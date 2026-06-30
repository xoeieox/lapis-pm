"""Functional critic: AC-anchored oracle verification (functional-critic-v0).

Ephemeral worktree at PR head SHA -> subprocess pytest -> GW-122B analysis ->
structured JSON verdict. Tears down the worktree in finally.

Mirrors eval_gate.py's worktree-isolation pattern. Zero paid API calls:
GW-122B (gravitywell-122b) is doorman-leased local-only.

Sandbox hardening (Fold 1): plants .critic-bin/ shims for pip/curl/wget/nc
before the GW agent call so network-egress and package-install are intercepted
at the process layer regardless of system-prompt instruction.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS_BASE = Path("/srv/lapis/spec-review-artifacts")
WORKTREE_PREFIX = "/tmp/functional-critic"
PYTEST_TIMEOUT_S = 120
GW_TIMEOUT_S = 300
DOORMAN_HOST = "127.0.0.1"
DOORMAN_PORT = 8407

# ---------------------------------------------------------------------------
# AC extraction (regex-only, no LLM)
# ---------------------------------------------------------------------------

# Matches lines starting with optional bold markers around "AC<digits>", e.g.:
#   AC1 (worktree isolation): ...
#   **AC2 (AC extraction):** ...
_AC_PATTERN = re.compile(r"^\*?\*?(AC\d+)\b.*?:", re.MULTILINE)


def extract_acs(spec_text: str) -> list[dict]:
    """Extract numbered ACs from spec markdown.

    Returns list of {"ac_id": "AC1", "claim": "..."} dicts.
    Specs with no numbered ACs return an empty list.
    Regex-only — no LLM call.
    """
    acs: list[dict] = []
    matches = list(_AC_PATTERN.finditer(spec_text))
    for i, m in enumerate(matches):
        ac_id = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(spec_text)
        claim = spec_text[start:end].strip()
        first_para = claim.split("\n\n")[0].strip()
        acs.append({"ac_id": ac_id, "claim": first_para[:400]})
    return acs


# ---------------------------------------------------------------------------
# Sandbox shims (Fold 1)
# ---------------------------------------------------------------------------

_SHIM_TMPL = """\
#!/bin/bash
echo "[functional-critic-sandbox] {cmd} blocked: not permitted in critic worktree" >&2
exit 1
"""


def _plant_sandbox_shims(worktree_path: str) -> None:
    """Plant .critic-bin/ intercept shims for pip, curl, wget, nc."""
    shim_dir = Path(worktree_path) / ".critic-bin"
    shim_dir.mkdir(exist_ok=True)
    for cmd in ("pip", "pip3", "curl", "wget", "nc"):
        shim = shim_dir / cmd
        shim.write_text(_SHIM_TMPL.format(cmd=cmd))
        shim.chmod(0o755)


# ---------------------------------------------------------------------------
# Worktree lifecycle
# ---------------------------------------------------------------------------

def _git_fetch(repo_path: str) -> None:
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            capture_output=True, text=True, timeout=30,
            cwd=repo_path,
        )
    except Exception as exc:
        logger.warning("functional_critic: git fetch failed for %s: %s", repo_path, exc)


def _create_worktree(repo_path: str, worktree_dir: str, head_sha: str) -> bool:
    """Create a detached worktree at head_sha. Returns True on success."""
    try:
        r = subprocess.run(
            ["git", "worktree", "add", "--detach", worktree_dir, head_sha],
            capture_output=True, text=True, cwd=repo_path, timeout=60,
        )
        if r.returncode != 0:
            logger.warning(
                "functional_critic: worktree creation failed for %s: %s",
                head_sha, r.stderr[:300],
            )
            return False
        return True
    except Exception as exc:
        logger.warning("functional_critic: worktree creation error: %s", exc)
        return False


def _remove_worktree(repo_path: str, worktree_dir: str) -> None:
    """Remove worktree unconditionally. Called in finally so never raises."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", worktree_dir],
            capture_output=True, cwd=repo_path, timeout=30,
        )
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True, cwd=repo_path, timeout=30,
        )
    except Exception as exc:
        logger.warning(
            "functional_critic: worktree remove failed for %s: %s", worktree_dir, exc
        )


# ---------------------------------------------------------------------------
# Test runner (direct subprocess, not via GW agent)
# ---------------------------------------------------------------------------

def _run_pytest(worktree_dir: str, timeout_s: int = PYTEST_TIMEOUT_S) -> dict:
    """Run pytest -v in worktree. Returns result dict with stdout/returncode."""
    start = time.monotonic()
    try:
        r = subprocess.run(
            ["python3", "-m", "pytest", "-v", "--tb=short", "--no-header", "-q"],
            capture_output=True, text=True,
            cwd=worktree_dir, timeout=timeout_s,
        )
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[:8000],
            "stderr": r.stderr[:2000],
            "elapsed_s": round(time.monotonic() - start, 1),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1, "stdout": "", "stderr": "pytest timed out",
            "elapsed_s": float(timeout_s), "timed_out": True,
        }
    except FileNotFoundError:
        return {
            "returncode": -1, "stdout": "", "stderr": "pytest not found in worktree",
            "elapsed_s": 0.0, "timed_out": False,
        }
    except Exception as exc:
        return {
            "returncode": -1, "stdout": "", "stderr": str(exc),
            "elapsed_s": 0.0, "timed_out": False,
        }


# ---------------------------------------------------------------------------
# GW-122B critic call (zero paid)
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = """\
You are a functional critic for the Lapis PM pipeline. Your job: given a set of
acceptance criteria (ACs), determine whether each AC is satisfied by the PR being
reviewed. You have read_file and grep to examine the worktree.

For each AC:
1. Find the relevant test(s) or verification surface (CLI entrypoint, library fn,
   HTTP endpoint, etc.)
2. Read the test code and assess oracle quality:
   - Does the test assert INTENT (would catch a wrong-but-plausible impl)?
   - Or does it mirror the impl (would pass even if the impl is trivially wrong)?
3. Cross-reference the pytest output provided in the prompt (already run for you).
4. Assign: pass / fail / could_not_exercise

For oracle_assessment.suspicious_tests, list any test file where the assertions
test implementation shape rather than behavioural intent. Only list concrete paths
you actually read via read_file.

Rules:
- could_not_exercise: surface not reachable from the worktree (no CLI, no test,
  browser-only, network-only).
- Empty evidence field is a schema error — always fill it.
- Do NOT install packages or make outbound network calls.
- Return ONLY valid JSON, no prose outside it.

Schema:
{
  "ac_verdicts": [
    {
      "ac_id": "AC1",
      "claim": "<one-line claim>",
      "verdict": "pass" | "fail" | "could_not_exercise",
      "evidence": "<what was found and why this verdict>",
      "notes": "<optional, may be empty string>"
    }
  ],
  "oracle_assessment": {
    "summary": "<overall test oracle quality>",
    "suspicious_tests": [
      {"path": "<relative path>", "reason": "<why suspicious>"}
    ]
  },
  "overall": "pass" | "partial" | "fail" | "unverifiable"
}

overall rules:
- "unverifiable": all ACs are could_not_exercise
- "partial": at least one pass or fail, at least one could_not_exercise
- "fail": at least one fail, none could_not_exercise
- "pass": all ACs pass
"""


def _check_doorman(host: str = DOORMAN_HOST, port: int = DOORMAN_PORT) -> bool:
    """Quick liveness probe for doorman."""
    try:
        import httpx
        r = httpx.get(f"http://{host}:{port}/healthz", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _call_gw_critic(
    acs: list[dict],
    diff_summary: str,
    pytest_result: dict,
    worktree_dir: str,
    run_id: str,
) -> dict | None:
    """Invoke GW-122B for functional critic analysis. Returns parsed dict or None.

    Returns None if GW is unavailable (doorman unreachable, wake failed, call error).
    Never raises — caller converts None to unverifiable.
    """
    if not _check_doorman():
        logger.warning("functional_critic: doorman unreachable — GW unavailable")
        return None

    try:
        from agents_core.gw_agent import call_gw_agent, DEFAULT_READONLY_TOOLS
    except ImportError as exc:
        logger.warning("functional_critic: agents_core.gw_agent unavailable: %s", exc)
        return None

    ac_text = "\n".join(
        f"- {a['ac_id']}: {a['claim']}" for a in acs
    ) if acs else "(no numbered ACs found)"

    pytest_block = (
        f"pytest exit code: {pytest_result['returncode']}\n"
        f"elapsed: {pytest_result['elapsed_s']}s\n"
        f"timed_out: {pytest_result.get('timed_out', False)}\n"
        f"stdout:\n{pytest_result['stdout'][:4000]}\n"
        f"stderr: {pytest_result['stderr'][:500]}"
    )

    prompt = (
        f"## Acceptance Criteria\n\n{ac_text}\n\n"
        f"## PR Diff (summary)\n\n{diff_summary[:3000]}\n\n"
        f"## Pytest Output (pre-run in worktree)\n\n```\n{pytest_block}\n```\n\n"
        f"Worktree path: {worktree_dir}\n\n"
        "Now read the relevant test files and source using read_file and grep. "
        "Produce the JSON verdict."
    )

    try:
        text, _transcript = call_gw_agent(
            prompt=prompt,
            system=_CRITIC_SYSTEM,
            cwd=worktree_dir,
            tools=DEFAULT_READONLY_TOOLS,
            json_mode=True,
            on_wake_fail="skip",
            return_transcript=True,
            work_id=run_id,
            timeout=GW_TIMEOUT_S,
        )
    except Exception as exc:
        logger.warning("functional_critic: call_gw_agent error: %s", exc)
        return None

    if text is None:
        logger.warning("functional_critic: GW did not run (on_wake_fail=skip)")
        return None

    # Strip markdown fences / think blocks
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip())
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned.startswith("{"):
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            cleaned = m.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("functional_critic: failed to parse GW JSON verdict: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Verdict validation
# ---------------------------------------------------------------------------

def validate_verdict(verdict: dict) -> list[str]:
    """Return list of schema errors (empty = valid).

    Checks: ac_verdicts is a list, each entry has non-empty evidence and valid
    verdict value, overall is a valid value, oracle_assessment is a dict.
    """
    errors: list[str] = []
    ac_verdicts = verdict.get("ac_verdicts")
    if not isinstance(ac_verdicts, list):
        errors.append("ac_verdicts must be a list")
    else:
        valid_verdicts = {"pass", "fail", "could_not_exercise"}
        for i, av in enumerate(ac_verdicts):
            if not isinstance(av, dict):
                errors.append(f"ac_verdicts[{i}] must be a dict")
                continue
            if not av.get("evidence"):
                errors.append(
                    f"ac_verdicts[{i}] ({av.get('ac_id', '?')}) has empty evidence (AC4)"
                )
            if av.get("verdict") not in valid_verdicts:
                errors.append(
                    f"ac_verdicts[{i}] invalid verdict: {av.get('verdict')!r}"
                )
    valid_overall = {"pass", "partial", "fail", "unverifiable"}
    if verdict.get("overall") not in valid_overall:
        errors.append(f"invalid overall: {verdict.get('overall')!r}")
    if not isinstance(verdict.get("oracle_assessment"), dict):
        errors.append("oracle_assessment must be a dict")
    return errors


# ---------------------------------------------------------------------------
# Artifact writer (AC5)
# ---------------------------------------------------------------------------

def write_verdict_artifact(run_id: str, pr_number: int, verdict: dict) -> Path:
    """Write verdict to /srv/lapis/spec-review-artifacts/<run_id>/functional-critic-<pr>.json."""
    artifact_dir = ARTIFACTS_BASE / run_id
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"functional-critic-{pr_number}.json"
        artifact_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
        logger.info("functional_critic: artifact written to %s", artifact_path)
        return artifact_path
    except Exception as exc:
        logger.warning("functional_critic: artifact write failed: %s", exc)
        return artifact_dir / f"functional-critic-{pr_number}.json"


# ---------------------------------------------------------------------------
# Repo path resolution
# ---------------------------------------------------------------------------

def _resolve_repo_path(repo: str) -> str:
    try:
        from agents_core.shaper import Shaper
        return Shaper.resolve_repo_cwd(repo)
    except Exception:
        return f"/srv/git/{repo}-working"


# ---------------------------------------------------------------------------
# Main entry point (AC1, AC2, AC3, AC4, AC6, AC7)
# ---------------------------------------------------------------------------

def run_functional_critic(
    target_id: str,
    pr_number: int,
    head_sha: str,
    repo: str,
    spec_text: str,
    run_id: str,
    diff_summary: str = "",
) -> dict:
    """Run the functional critic for a PR.

    Creates an ephemeral worktree at head_sha (AC1), extracts ACs from spec_text
    (AC2), runs pytest, calls GW-122B for oracle analysis (AC3/AC4), returns a
    verdict dict. Tears down the worktree unconditionally in finally (AC1).

    Returns a verdict dict that always conforms to the schema, even on failure
    (overall: "unverifiable" with a reason in oracle_assessment.summary).
    """
    repo_path = _resolve_repo_path(repo)
    short_sha = head_sha[:8]
    worktree_dir = f"{WORKTREE_PREFIX}-{pr_number}-{short_sha}"
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _unverifiable(reason: str) -> dict:
        return {
            "ac_verdicts": [],
            "oracle_assessment": {"summary": reason, "suspicious_tests": []},
            "overall": "unverifiable",
            "provenance": {
                "worktree": worktree_dir,
                "gw_model": "gravitywell-122b",
                "pr_head_sha": head_sha,
                "run_id": run_id,
                "ts": ts,
            },
        }

    # AC2: extract ACs (regex only)
    acs = extract_acs(spec_text)
    if not acs:
        logger.info(
            "functional_critic: no numbered ACs found in spec for %s — unverifiable",
            target_id,
        )
        return _unverifiable("spec has no numbered ACs (pattern AC\\d+:)")

    _git_fetch(repo_path)
    worktree_created = _create_worktree(repo_path, worktree_dir, head_sha)

    try:
        if not worktree_created:
            return _unverifiable(f"worktree creation failed for sha={head_sha[:12]}")

        # AC1 + Fold 1: sandbox shims in worktree
        _plant_sandbox_shims(worktree_dir)

        # Run pytest directly (execution layer, separate from GW analysis)
        pytest_result = _run_pytest(worktree_dir)

        # AC7: if GW unavailable, return unverifiable (no paid fallback)
        verdict = _call_gw_critic(acs, diff_summary, pytest_result, worktree_dir, run_id)

        if verdict is None:
            logger.warning(
                "functional_critic: GW unavailable for PR #%d — returning unverifiable",
                pr_number,
            )
            return _unverifiable("gw_unavailable")

        # Schema validation (AC4: evidence field required)
        errors = validate_verdict(verdict)
        if errors:
            logger.warning(
                "functional_critic: verdict schema errors for PR #%d: %s",
                pr_number, errors,
            )
            verdict["_schema_errors"] = errors

        # Add provenance
        verdict.setdefault("provenance", {})
        verdict["provenance"].update({
            "worktree": worktree_dir,
            "gw_model": "gravitywell-122b",
            "pr_head_sha": head_sha,
            "run_id": run_id,
            "ts": ts,
        })

        return verdict

    finally:
        # AC1: tear down unconditionally whether verdict passes or not
        if worktree_created:
            _remove_worktree(repo_path, worktree_dir)


# ---------------------------------------------------------------------------
# Verdict summary helpers (used by pm_core and CLI)
# ---------------------------------------------------------------------------

def verdict_summary_text(verdict: dict) -> str:
    """Build a human-readable summary for the brief ## Functional Critic section."""
    overall = verdict.get("overall", "unknown")
    ac_verdicts = verdict.get("ac_verdicts") or []
    oracle = verdict.get("oracle_assessment") or {}
    suspicious = oracle.get("suspicious_tests") or []
    prov = verdict.get("provenance") or {}

    pass_count = sum(1 for v in ac_verdicts if v.get("verdict") == "pass")
    fail_count = sum(1 for v in ac_verdicts if v.get("verdict") == "fail")
    nx_count = sum(1 for v in ac_verdicts if v.get("verdict") == "could_not_exercise")
    total = len(ac_verdicts)

    lines = [
        f"**Overall:** `{overall}`",
        f"**ACs:** {total} total — {pass_count} pass / {fail_count} fail / {nx_count} could_not_exercise",
    ]

    if oracle.get("summary"):
        lines.append(f"**Oracle:** {oracle['summary']}")

    if suspicious:
        lines.append(f"**Suspicious tests ({len(suspicious)}):**")
        for s in suspicious[:3]:
            lines.append(f"  - `{s.get('path', '?')}`: {s.get('reason', '')}")
        if len(suspicious) > 3:
            lines.append(f"  - ... ({len(suspicious) - 3} more)")

    if ac_verdicts:
        lines.append("**Per-AC verdicts:**")
        for av in ac_verdicts:
            icon = {"pass": "✓", "fail": "✗", "could_not_exercise": "?"}.get(
                av.get("verdict", ""), "?"
            )
            lines.append(
                f"  {icon} {av.get('ac_id', '?')}: {av.get('verdict', '?')}"
                + (f" — {av.get('evidence', '')[:120]}" if av.get("evidence") else "")
            )

    if prov.get("run_id"):
        lines.append(f"**Artifact:** `{prov.get('run_id')}/functional-critic-*.json`")

    return "\n".join(lines)


def unexercised_ac_count(verdict: dict) -> int:
    """Count ACs with could_not_exercise verdict."""
    return sum(
        1 for v in (verdict.get("ac_verdicts") or [])
        if v.get("verdict") == "could_not_exercise"
    )
