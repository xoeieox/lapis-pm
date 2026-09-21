#!/usr/bin/env python3
"""Reviewer-quality spike: run a local model against historical PRs with known Claude verdicts.

Compares a local model's reviewer-shaped verdict against Claude's recorded verdict for
the same PR. Outputs a CSV row per (model, PR) pair and dumps raw model responses to a
sidecar dir for inspection.

Usage:
    # From /data/agents (so FORGEJO_TOKEN is on PATH):
    export $(grep FORGEJO_TOKEN /data/agents/config/conductor.env | xargs)
    python3 /srv/lapis/lapis-pm/scripts/reviewer_spike.py \\
        --pairs 'target1:42,target2:55' \\
        --model qwen3.6-35b-a3b.gguf \\
        --endpoint http://203.0.113.12:8081/v1/chat/completions \\
        --out /tmp/spike_results.csv

Run the same command per candidate model; the CSV appends across runs so all candidates
land in one file for ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, "/srv/lapis/lapis-pm")
sys.path.insert(0, "/srv/git/agents-core-working")

from lapis_pm import episodic  # noqa: E402
from agents_core.forgejo import get_pr_diff  # noqa: E402


# ---------------------------------------------------------------------------
# Verdict taxonomy — cr-bundle-lapis-pm-2026-09-21 item 5eb29b2087
# ---------------------------------------------------------------------------
# This spike (and the local_reviewer_witness module that lifted these
# constants verbatim, PR #94) uses the REDUCED 3-value taxonomy that the
# actual reviewer dispatch emits (see the user_prompt in
# pm_core._dispatch_reviewer and authority.SCREEN_SYSTEM):
#     "clean" | "fixable" | "needs-human"
#
# The 4-value taxonomy (clean | issues-noted | fix-eligible | needs-human)
# named in the original review finding is the *recorded-PR review contract*
# (reviewer verdict tags / code-reviewer findings), NOT the reviewer-dispatch
# output contract. The spike and witness compare against recorded Claude
# reviewer verdicts, which are stored in the 3-value dispatch taxonomy
# (pm:reviewer:pr=N:cycle=K:verdict=clean|fixable|needs-human), so the
# 3-value schema is the correct one for this comparison.
#
# Mapping layer at the emit boundary (the downstream emit pipeline's
# 4-value contract):
#   clean        -> clean
#   fixable      -> issues-noted        (issues listed; fixer-eligible when
#                                       any issue severity is med/high)
#   needs-human  -> needs-human
# The witness is observational-only (it never displaces Claude's verdict and
# feeds no auto-merge gate), so the mapping is documentation, not code: the
# emit pipeline consumes Claude's recorded verdict, not the witness output.
# If the witness is ever wired into the emit pipeline, implement the mapping
# there (at the emit boundary), NOT by changing this schema.

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


def _get_claude_verdict(target_id: str, pr_number: int) -> dict | None:
    """Pull most-recent clean reviewer verdict for this PR from episodic.

    Reads verdict from content (not tag). Filters out:
    - `_parse_error` records (Claude JSON parse failed; verdict tag is fallback)
    - post-fixer-retry tracking records (have `prior_index` / `status: addressed`)
    """
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    candidates: list[dict] = []
    for c in episodic.all_comments(target_id):
        has_verdict_tag = False
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                if t.split(":verdict=")[-1] != "pending":
                    has_verdict_tag = True
                    break
        if not has_verdict_tag:
            continue

        content = c.content or ""
        json_part = content.split("\n", 1)[-1].strip()
        try:
            body = json.loads(json_part)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(body, dict):
            continue
        if "prior_index" in body or body.get("status") == "addressed":
            continue
        if "_parse_error" in body:
            continue
        if body.get("verdict") not in ("clean", "fixable", "needs-human"):
            continue
        candidates.append(body)

    return candidates[-1] if candidates else None


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


# 3-value reviewer-dispatch taxonomy (see the mapping-layer note above this
# template for the relationship to the 4-value recorded-review contract).
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


def _run_local_reviewer(prompt: str, model: str, endpoint: str, timeout: int,
                        use_grammar: bool = True) -> dict:
    """Call the local OpenAI-compat endpoint. Returns dict with raw, parsed, latency_ms, json_valid, error.

    If use_grammar=True, sends a response_format json_schema constraint. llama.cpp enforces this at
    the sampling level. MLX-served endpoints generally do not honor this field and ignore it cleanly.
    """
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
        resp = httpx.post(endpoint, json=body, timeout=timeout)
        resp.raise_for_status()
        latency_ms = int((time.time() - start) * 1000)
        resp_body = resp.json()
        raw = resp_body["choices"][0]["message"]["content"]
    except Exception as exc:
        return {
            "raw": "",
            "parsed": None,
            "latency_ms": int((time.time() - start) * 1000),
            "json_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    cleaned = _clean_model_output(raw)
    try:
        parsed = json.loads(cleaned)
        return {"raw": raw, "parsed": parsed, "latency_ms": latency_ms, "json_valid": True, "error": None}
    except json.JSONDecodeError as exc:
        return {
            "raw": raw,
            "parsed": None,
            "latency_ms": latency_ms,
            "json_valid": False,
            "error": f"JSONDecodeError: {exc}",
        }


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


def _issue_recall(claude: dict, local: dict | None) -> tuple[int, int]:
    """How many of Claude's HIGH-severity issues did the local model surface (by path overlap)?

    Returns (matched_count, total_high_count). Very coarse — a real eval would need
    embedding-similarity on notes, but path-overlap is a decent first signal.
    """
    if local is None:
        return (0, 0)
    claude_high = [i for i in (claude.get("issues") or []) if i.get("severity") == "high"]
    if not claude_high:
        return (0, 0)
    local_paths = {(i.get("path") or "").strip() for i in (local.get("issues") or [])}
    matched = sum(1 for i in claude_high if (i.get("path") or "").strip() in local_paths and (i.get("path") or "").strip())
    return (matched, len(claude_high))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True,
                    help="Comma-separated target_id:pr_number pairs, e.g. 'foo:42,bar:55'")
    ap.add_argument("--model", required=True, help="Model id as served by the endpoint")
    ap.add_argument("--endpoint", default="http://203.0.113.12:8081/v1/chat/completions",
                    help="OpenAI-compat /chat/completions URL")
    ap.add_argument("--repo", default="lapis-pm", help="Repo name for Forgejo diff lookup")
    ap.add_argument("--out", required=True, help="CSV output path (appends if exists)")
    ap.add_argument("--raw-dir", default="/tmp/reviewer_spike_raw",
                    help="Dir for raw model response sidecars")
    ap.add_argument("--max-diff-chars", type=int, default=30000,
                    help="Truncate diffs longer than this many chars")
    ap.add_argument("--max-spec-chars", type=int, default=5000,
                    help="Truncate spec_summary to this many chars")
    ap.add_argument("--timeout", type=int, default=600, help="Per-PR LLM call timeout (s)")
    ap.add_argument("--no-grammar", action="store_true",
                    help="Disable response_format json_schema enforcement (use for MLX endpoints)")
    args = ap.parse_args()

    pairs: list[tuple[str, int]] = []
    for p in args.pairs.split(","):
        p = p.strip()
        if not p:
            continue
        if ":" not in p:
            print(f"ERROR: pair '{p}' must be target_id:pr_number", file=sys.stderr)
            return 2
        target, pr = p.split(":", 1)
        pairs.append((target.strip(), int(pr.strip())))

    raw_dir = Path(args.raw_dir) / re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for target_id, pr_number in pairs:
        print(f"\n=== {target_id} PR #{pr_number} ===", file=sys.stderr)

        claude_verdict = _get_claude_verdict(target_id, pr_number)
        if claude_verdict is None:
            print(f"  SKIP: no recorded Claude verdict for {target_id}/PR#{pr_number}", file=sys.stderr)
            continue

        try:
            diff = get_pr_diff(repo=args.repo, pr_number=pr_number)
        except Exception as exc:
            print(f"  SKIP: diff fetch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if not diff:
            print(f"  SKIP: empty diff for PR #{pr_number}", file=sys.stderr)
            continue

        orig_diff_chars = len(diff)
        if orig_diff_chars > args.max_diff_chars:
            diff = diff[: args.max_diff_chars] + f"\n\n[... diff truncated; original {orig_diff_chars} chars]"

        try:
            spec_summary = episodic.spec_summary(target_id, max_chars=args.max_spec_chars) or "(no spec summary available)"
        except Exception as exc:
            spec_summary = f"(spec_summary error: {type(exc).__name__})"

        prompt = REVIEWER_PROMPT_TEMPLATE.format(
            pr_number=pr_number,
            repo=args.repo,
            spec_summary=spec_summary,
            diff=diff,
        )

        local = _run_local_reviewer(
            prompt, args.model, args.endpoint, args.timeout,
            use_grammar=not args.no_grammar,
        )

        # Sidecar raw response for later inspection
        sidecar = raw_dir / f"{target_id}__pr{pr_number}.txt"
        sidecar.write_text(
            f"# Model: {args.model}\n"
            f"# Endpoint: {args.endpoint}\n"
            f"# Grammar: {'on' if not args.no_grammar else 'off'}\n"
            f"# Target: {target_id}\n# PR: {pr_number}\n"
            f"# Latency: {local['latency_ms']}ms\n"
            f"# JSON valid: {local['json_valid']}\n"
            f"# Error: {local.get('error')}\n"
            f"# --- Claude verdict ---\n{json.dumps(claude_verdict, indent=2)}\n"
            f"# --- Local raw response ---\n{local['raw']}\n",
            encoding="utf-8",
        )

        agreement = _score_agreement(claude_verdict, local.get("parsed"))
        high_matched, high_total = _issue_recall(claude_verdict, local.get("parsed"))
        parsed = local.get("parsed") or {}
        local_issues = parsed.get("issues") or []

        row = {
            "target_id": target_id,
            "pr_number": pr_number,
            "model": args.model,
            "claude_verdict": claude_verdict.get("verdict"),
            "claude_issue_count": len(claude_verdict.get("issues") or []),
            "claude_high_count": sum(1 for i in (claude_verdict.get("issues") or []) if i.get("severity") == "high"),
            "claude_confidence": claude_verdict.get("confidence"),
            "local_verdict": parsed.get("verdict"),
            "local_issue_count": len(local_issues),
            "local_high_count": sum(1 for i in local_issues if i.get("severity") == "high"),
            "local_confidence": parsed.get("confidence"),
            "agreement": agreement,
            "high_issue_path_recall": f"{high_matched}/{high_total}" if high_total else "n/a",
            "json_valid": local["json_valid"],
            "latency_ms": local["latency_ms"],
            "diff_chars_used": len(diff),
            "diff_chars_orig": orig_diff_chars,
            "error": local.get("error") or "",
            "raw_path": str(sidecar),
        }
        rows.append(row)

        print(
            f"  claude:{row['claude_verdict']} | local:{row['local_verdict']} | "
            f"{agreement} | high_recall={row['high_issue_path_recall']} | "
            f"{local['latency_ms']}ms | json_valid={local['json_valid']}",
            file=sys.stderr,
        )

    if not rows:
        print("\nNo rows written (all pairs skipped).", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {out_path}", file=sys.stderr)
    print(f"Raw responses in {raw_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
