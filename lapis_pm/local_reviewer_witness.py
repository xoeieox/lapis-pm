"""Local-reviewer witness module (local-reviewer-witness-v0).

Provides `run_local_reviewer_witness()` which dispatches the StarHouse local
Qwen (MoE Qwen3.6-35B-A3B) as a parallel third-reviewer witness alongside the
existing Claude (Opus) reviewer. Observational only — never displaces Claude's
verdict. v0 is data-collection only; no auto-merge gate change.

Key constants (REVIEWER_PROMPT_TEMPLATE, REVIEWER_JSON_SCHEMA, _clean_model_output,
_score_agreement) lifted verbatim from scripts/reviewer_spike.py (PR #93).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from lapis_pm.node_probe import node_reachable

_PROBE_TIMEOUT = 3    # seconds for connect probe before POST
_CONNECT_TIMEOUT = 5  # seconds connect cap on POST (120s read budget preserved)


# ---------------------------------------------------------------------------
# Lifted verbatim from scripts/reviewer_spike.py (PR #93 — authoritative)
# ---------------------------------------------------------------------------

REVIEWER_PROMPT_TEMPLATE = """You are reviewing PR #{pr_number} in repo `{repo}`.

## Spec context

{spec_summary}

## Role

You are a pure read-only judge. Read the diff below and return a verdict. Do not
suggest edits in prose; the verdict JSON is your only output.

## Diff

```diff
{diff}
```

## Output

Return JSON only, no prose outside it:
{{
  "verdict": "clean" | "fixable" | "needs-human",
  "issues": [{{"severity": "low" | "med" | "high", "path": "...", "note": "..."}}],
  "confidence": 0.0-1.0
}}

Verdict guide:
- `clean`: no issues that warrant a fix
- `fixable`: issues a fixer can address mechanically from the listed notes
- `needs-human`: issues require human judgment (spec ambiguity, scope question)

Severity guide:
- `high`: blocks merge (invariant violation, broken contract, missing test)
- `med`: should fix but not blocking
- `low`: nit, style, minor improvement

Note style: each `note` is at most 2 sentences. Do not include analysis,
chain-of-thought, or rumination inside `note` — just the issue and the fix.
If you have nothing to flag, return `verdict: clean` with an empty `issues` array.
"""

REVIEWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["clean", "fixable", "needs-human"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["low", "med", "high"]},
                    "path": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["severity", "path", "note"],
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["verdict", "issues", "confidence"],
}


def _clean_model_output(content: str) -> str:
    """Strip markdown fences and <think> blocks before JSON parsing."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content.strip())
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # Some models prepend a preamble before the JSON; grab the first {...} block
    if not content.startswith("{"):
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if m:
            content = m.group(0)
    return content


def _score_agreement(claude: dict, local: dict | None) -> str:
    """Categorize verdict agreement: agree | diverge_minor | diverge_major | local_failed."""
    if local is None:
        return "local_failed"
    cv = claude.get("verdict")
    lv = local.get("verdict")
    if cv == lv:
        return "agree"
    # clean vs fixable is a softer disagreement than either vs needs-human
    if {cv, lv} <= {"clean", "fixable"}:
        return "diverge_minor"
    return "diverge_major"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LocalReviewerWitnessResult:
    """Result envelope from a local-reviewer witness call.

    Populated on both success and failure paths. On failure, `agreement` is
    "local_failed" and `error` is set; verdict / issues / confidence are None.
    """
    verdict: Literal["clean", "fixable", "needs-human"] | None
    issues: list[dict]
    confidence: float | None
    agreement: Literal["agree", "diverge_minor", "diverge_major", "local_failed"]
    latency_ms: int
    json_valid: bool
    model: str | None          # populated from llama.cpp response
    prompt_hash: str           # sha256 of the prompt (for provenance)
    dispatched_at: str         # ISO8601
    error: str | None          # populated on failure

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "issues": self.issues,
            "confidence": self.confidence,
            "agreement": self.agreement,
            "latency_ms": self.latency_ms,
            "json_valid": self.json_valid,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "dispatched_at": self.dispatched_at,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Divergence note helper
# ---------------------------------------------------------------------------

def _format_divergence_note(claude_verdict: dict, witness: LocalReviewerWitnessResult) -> str:
    """Produce a human-readable diff between Claude's verdict and the witness result.

    Used when agreement == "diverge_major" to write a pm:reviewer-divergence
    observation comment.
    """
    cv = claude_verdict.get("verdict", "?")
    lv = witness.verdict or "?"
    cv_issues = claude_verdict.get("issues") or []
    lv_issues = witness.issues or []

    lines = [
        f"Reviewer divergence (major): Claude={cv}, Local={lv}",
        "",
        f"Claude issues ({len(cv_issues)}):",
    ]
    for i in cv_issues[:5]:
        lines.append(f"  [{i.get('severity','?')}] {i.get('path','?')}: {i.get('note','')}")
    if len(cv_issues) > 5:
        lines.append(f"  ... ({len(cv_issues) - 5} more)")

    lines.append(f"\nLocal issues ({len(lv_issues)}):")
    for i in lv_issues[:5]:
        lines.append(f"  [{i.get('severity','?')}] {i.get('path','?')}: {i.get('note','')}")
    if len(lv_issues) > 5:
        lines.append(f"  ... ({len(lv_issues) - 5} more)")

    lines.append(f"\nLocal model: {witness.model or '(unknown)'}")
    lines.append(f"Latency: {witness.latency_ms}ms")
    lines.append(f"Dispatched: {witness.dispatched_at}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_local_reviewer_witness(
    diff_text: str,
    repo: str,
    pr_number: int,
    spec_summary: str,
    claude_verdict: dict,           # used only for agreement classification
    *,
    model: str = "qwen3.6-35b-a3b.gguf",
    endpoint: str = "http://203.0.113.12:8081/v1/chat/completions",
    timeout: int = 120,
    max_diff_chars: int = 30000,
    max_spec_chars: int = 5000,
    _use_grammar: bool = True,      # internal; set False only in tests exercising parse-failure path
) -> LocalReviewerWitnessResult:
    """Run the local-reviewer witness call.

    Returns a populated LocalReviewerWitnessResult on both success and failure.
    Never raises.
    """
    import httpx  # local import to avoid top-level httpx dependency at module load

    dispatched_at = datetime.now(timezone.utc).isoformat()

    # Truncate inputs to budget
    if len(diff_text) > max_diff_chars:
        diff_text = diff_text[:max_diff_chars] + f"\n\n[... diff truncated; original chars exceeded {max_diff_chars}]"
    if len(spec_summary) > max_spec_chars:
        spec_summary = spec_summary[:max_spec_chars].rstrip() + "\n…[truncated]"

    prompt = REVIEWER_PROMPT_TEMPLATE.format(
        pr_number=pr_number,
        repo=repo,
        spec_summary=spec_summary,
        diff=diff_text,
    )
    prompt_hash = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _make_failure(latency_ms: int, error: str) -> LocalReviewerWitnessResult:
        return LocalReviewerWitnessResult(
            verdict=None,
            issues=[],
            confidence=None,
            agreement="local_failed",
            latency_ms=latency_ms,
            json_valid=False,
            model=None,
            prompt_hash=prompt_hash,
            dispatched_at=dispatched_at,
            error=error,
        )

    def _do_post(use_grammar: bool) -> tuple[dict | None, int, str | None, str | None]:
        """POST to endpoint. Returns (resp_json, latency_ms, raw_content, error)."""
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        if use_grammar:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ReviewerVerdict",
                    "strict": True,
                    "schema": REVIEWER_JSON_SCHEMA,
                },
            }
        start = time.time()
        try:
            resp = httpx.post(endpoint, json=body, timeout=httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT))
            lat = int((time.time() - start) * 1000)
            if resp.status_code in (400, 422):
                return None, lat, None, f"HTTP {resp.status_code}"
            resp.raise_for_status()
            rj = resp.json()
            raw = rj["choices"][0]["message"]["content"]
            return rj, lat, raw, None
        except httpx.TimeoutException as exc:
            return None, int((time.time() - start) * 1000), None, f"TimeoutException: {exc}"
        except httpx.ConnectError as exc:
            return None, int((time.time() - start) * 1000), None, f"ConnectError: {exc}"
        except Exception as exc:
            return None, int((time.time() - start) * 1000), None, f"{type(exc).__name__}: {exc}"

    # Probe gate: skip POST if endpoint is unreachable (avoids 45-120s SYN block)
    _t_probe = time.time()
    if not node_reachable(endpoint, timeout=_PROBE_TIMEOUT):
        return _make_failure(
            int((time.time() - _t_probe) * 1000),
            f"endpoint unreachable (probe): {endpoint}",
        )

    # First attempt: with grammar (json_schema)
    resp_json, latency_ms, raw, err = _do_post(_use_grammar)

    # Graceful degradation: retry once with json_object if 4xx on first attempt
    if err is not None and err.startswith("HTTP ") and _use_grammar:
        resp_json2, lat2, raw2, err2 = _do_post(False)
        latency_ms += lat2
        if err2 is not None:
            return _make_failure(latency_ms, f"json_schema: {err}; json_object: {err2}")
        resp_json, raw, err = resp_json2, raw2, None

    if err is not None:
        return _make_failure(latency_ms, err)

    if not raw:
        return _make_failure(latency_ms, "empty response content")

    # Extract model name from response
    resp_model = resp_json.get("model") if resp_json else None

    # Parse response
    cleaned = _clean_model_output(raw)
    try:
        parsed = json.loads(cleaned)
        json_valid = True
    except json.JSONDecodeError as exc:
        return LocalReviewerWitnessResult(
            verdict=None,
            issues=[],
            confidence=None,
            agreement="local_failed",
            latency_ms=latency_ms,
            json_valid=False,
            model=resp_model,
            prompt_hash=prompt_hash,
            dispatched_at=dispatched_at,
            error=f"JSONDecodeError: {exc}",
        )

    verdict = parsed.get("verdict")
    if verdict not in ("clean", "fixable", "needs-human"):
        verdict = None

    issues = parsed.get("issues") or []
    confidence = parsed.get("confidence")
    agreement = _score_agreement(claude_verdict, parsed if verdict else None)

    return LocalReviewerWitnessResult(
        verdict=verdict,
        issues=issues,
        confidence=confidence,
        agreement=agreement,
        latency_ms=latency_ms,
        json_valid=json_valid,
        model=resp_model,
        prompt_hash=prompt_hash,
        dispatched_at=dispatched_at,
        error=None,
    )
