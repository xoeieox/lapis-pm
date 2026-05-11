"""lapis_pm.spec_review — Pre-bind parallel Opus + Mirror Council review.

Public entry point: run_spec_review(spec_path, council_voicing, timeout_s, repo_override)
Returns SpecReviewBrief. Synchronous; caller blocks until both passes complete or timeout.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_CLAUDE_QUEUE_COMPLETED = Path("/srv/lapis/claude-queue/completed")
_CLAUDE_QUEUE_FAILED = Path("/srv/lapis/claude-queue/failed")
_GPU_QUEUE_COMPLETED = Path("/srv/lapis/gpu-queue/completed")
_GPU_QUEUE_FAILED = Path("/srv/lapis/gpu-queue/failed")
_COUNCIL_DIR = Path("/srv/lapis/council")

# Council terminal statuses — per v0.next spec + "closed" for scene mode.
_COUNCIL_TERMINAL: frozenset[str] = frozenset(
    {"resolved", "open", "laid-down", "failed", "closed"}
)

# Frontmatter regexes — anchored, multiline, kebab-case inside backticks.
_TID_RE = re.compile(
    r"^\*\*Target ID:\*\*\s+`([a-z0-9][a-z0-9-]*[a-z0-9])`\s*$",
    re.MULTILINE,
)
_REPO_RE = re.compile(
    r"^\*\*Repo:\*\*\s+`([a-z0-9][a-z0-9-]*[a-z0-9])`\s*$",
    re.MULTILINE,
)

_POLL_CADENCE_S = 10  # fixed per Invariant 8


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SpecFrontmatterError(ValueError):
    """Raised when spec frontmatter is missing or malformed."""


class InvariantContextError(OSError):
    """Raised when a required invariant-context file is missing."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class _DispatchResult:
    """Minimal dispatch-result shape.  Mirrors agents_core.shaper.DispatchResult.
    Used in both stub path (directly) and real path (wraps the shaper result).
    Only task_id is consumed by the poll loop; other fields are for audit.
    """
    task_id: str
    agent_type: str = ""
    spec_path: str = ""
    spec_id: str = ""
    output_path: str = ""


@dataclass
class SpecReviewBrief:
    spec_path: Path
    target_id: str
    repo: str
    opus_verdict: Literal["clean", "fixable", "needs-human", "timeout", "error"]
    opus_issues: list[dict]
    opus_confidence: float
    opus_run_id: str
    council_status: Literal["resolved", "open", "laid-down", "failed", "timeout", "error", "closed"]
    council_landing: str
    council_open_questions: list[str]
    council_confidence: str
    council_positions: list[dict]
    council_run_id: str
    elapsed_s: float
    combined_recommendation: Literal[
        "proceed-to-bind", "amend-spec", "shape-with-Erah", "incomplete"
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_spec_frontmatter(spec_path: Path) -> tuple[str, str]:
    """Return (target_id, repo) parsed from first 50 lines of spec.

    Raises SpecFrontmatterError with a descriptive message on failure.
    """
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as e:
        raise SpecFrontmatterError(f"cannot read spec file: {e}") from e

    # Search first 50 lines only
    first_50 = "\n".join(text.splitlines()[:50])

    tid_m = _TID_RE.search(first_50)
    repo_m = _REPO_RE.search(first_50)

    if not tid_m and not repo_m:
        raise SpecFrontmatterError(
            f"missing **Target ID:** and **Repo:** in {spec_path} "
            "(expected `**Target ID:** \\`<kebab-id>\\`` anchored at line start)"
        )
    if not tid_m:
        raise SpecFrontmatterError(
            f"missing **Target ID:** in {spec_path} "
            "(expected `**Target ID:** \\`<kebab-id>\\`` anchored at line start)"
        )
    if not repo_m:
        raise SpecFrontmatterError(
            f"missing **Repo:** in {spec_path} "
            "(expected `**Repo:** \\`<kebab-id>\\`` anchored at line start)"
        )

    return tid_m.group(1), repo_m.group(1)


def _synth_target_id(parsed_target_id: str) -> str:
    """Generate a synthetic target_id for queue tracking. Never written to TargetStore."""
    return f"spec-review-{parsed_target_id}-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _load_invariant_context(repo: str) -> str:
    """Load CLAUDE.md + SPEC.md + Constitution-Kernel.md for the given repo.

    CLAUDE.md and Constitution-Kernel.md are REQUIRED.
    SPEC.md is optional (empty section if missing).
    Raises InvariantContextError if a required file is missing.
    No truncation in v0 (per Invariant 12).
    """
    claude_md_path = Path(f"/srv/git/{repo}-working/CLAUDE.md")
    spec_md_path = Path(f"/srv/git/{repo}-working/SPEC.md")
    kernel_path = Path("/srv/git/inertia-vault-working/Lapis/Constitution-Kernel.md")

    if not claude_md_path.exists():
        raise InvariantContextError(
            f"CLAUDE.md not found at {claude_md_path} — cannot load invariant context"
        )
    if not kernel_path.exists():
        raise InvariantContextError(
            f"Constitution-Kernel.md not found at {kernel_path} — cannot load invariant context"
        )

    claude_md = claude_md_path.read_text(encoding="utf-8")
    spec_md = spec_md_path.read_text(encoding="utf-8") if spec_md_path.exists() else ""
    kernel = kernel_path.read_text(encoding="utf-8")

    return (
        f"=== CLAUDE.md ({repo}) ===\n{claude_md}\n\n"
        f"=== SPEC.md ({repo}) ===\n{spec_md}\n\n"
        f"=== Constitution-Kernel.md ===\n{kernel}"
    )


def _find_reviewer_output(task_id: str) -> Path | None:
    """Check completed dirs for the spec_reviewer output file.

    claude_queue_runner writes output only to completed/ (OUTPUT_DIR).
    GPU queue follows the same convention.
    """
    for completed in (_CLAUDE_QUEUE_COMPLETED, _GPU_QUEUE_COMPLETED):
        p = completed / f"{task_id}-output.md"
        if p.exists():
            return p
    return None


def _extract_outermost_json_object(text: str) -> str | None:
    """Return the first complete {...} substring using bracket counting.

    Handles nested objects (e.g. an issues array with {} elements).
    Returns None if no balanced object is found.
    """
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _read_verdict_from_output(output_path: Path) -> dict:
    """Parse the verdict JSON from the spec_reviewer output file.

    The spec_reviewer template returns JSON only. Extracts the first JSON
    object found in the file content, stripping markdown fences if present.
    """
    content = output_path.read_text(encoding="utf-8")
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    # Try direct JSON parse first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fall back: bracket-counting scan for outermost {...} (handles nested objects)
    obj_str = _extract_outermost_json_object(content)
    if obj_str:
        try:
            return json.loads(obj_str)
        except json.JSONDecodeError:
            pass
    # Return error shape on parse failure
    return {"verdict": "error", "issues": [], "confidence": 0.0}


def _dispatch_spec_reviewer(
    spec_text: str,
    synth_target_id: str,
    parsed_target_id: str,
    repo: str,
    invariant_context: str,
) -> _DispatchResult:
    """Dispatch the spec_reviewer agent. Stub-aware."""
    if os.getenv("SPEC_REVIEWER_STUB") == "1":
        verdict = os.getenv("SPEC_REVIEWER_STUB_VERDICT", "clean")
        task_id = f"stub-{synth_target_id}"
        output_path = _CLAUDE_QUEUE_COMPLETED / f"{task_id}-output.md"
        _CLAUDE_QUEUE_COMPLETED.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"verdict": verdict, "issues": [], "confidence": 0.95}),
            encoding="utf-8",
        )
        return _DispatchResult(
            task_id=task_id,
            agent_type="spec_reviewer",
            spec_path="stub",
            spec_id="stub",
            output_path=str(output_path),
        )

    from lapis_pm.pm_core import _SHAPER
    vars_: dict = {
        "target_id": synth_target_id,
        "spec_summary": f"spec-review pass for {parsed_target_id}",
        "repo": repo,
        "invariant_context": invariant_context,
        "question": (
            "Review the spec document below for technical soundness, "
            "implementability, and DON'T-do/DoD coverage. The document follows."
        ),
        "pr_number": "",
        "slug": "spec-review",
        "existing_branch": f"lapis/{synth_target_id}/spec-review",
        "base_branch": "main",
    }
    real = _SHAPER.dispatch(
        "spec_reviewer",
        synth_target_id,
        user_prompt=spec_text,
        vars_=vars_,
    )
    return _DispatchResult(
        task_id=real.task_id,
        agent_type=real.agent_type,
        spec_path=real.spec_path,
        spec_id=real.spec_id,
        output_path=real.output_path,
    )


def _dispatch_council(
    spec_text: str,
    parsed_target_id: str,
    invariant_context: str,
    voicing: str,
) -> str:
    """Submit the council deliberation. Returns run_id.

    SPEC_REVIEW_COUNCIL_STUB=1: skip cmd_submit entirely, return a fake run_id.
    The poll loop will never find a YAML for the fake id, so the timeout fires
    naturally. Used in Phase 42 smoke to maintain hermetic (no real LLM calls).
    """
    if os.getenv("SPEC_REVIEW_COUNCIL_STUB") == "1":
        return f"stub-council-{uuid.uuid4().hex[:8]}"

    decision_text = (
        f"Review this spec for ecosystem fit and meaning: target {parsed_target_id}.\n"
        f"The technical-soundness question is being handled in parallel by an Opus pass.\n"
        f"Your role: invariant fit and meaning. Does this spec align with the Lapis\n"
        f"Constitution Kernel and the conductor/lapis-ecosystem chub? What does it\n"
        f"imply for what we're building?\n\n"
        f"=== INVARIANT CONTEXT ===\n{invariant_context}\n\n"
        f"=== SPEC ===\n{spec_text}"
    )
    from agents_core.council.cli import cmd_submit, DEFAULT_TURNS
    args = argparse.Namespace(
        decision=decision_text,
        voicing=voicing,
        mode="deliberation",
        n=None,
        turns=DEFAULT_TURNS,
        with_entity=None,
        narrator=False,
        narrator_voice=None,
        no_queue=False,
        notify=False,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_submit(args)
    output = buf.getvalue()
    m = re.search(r"task_id=(\S+)", output)
    if not m:
        raise RuntimeError(
            f"could not parse run_id from cmd_submit output: {output!r}"
        )
    return m.group(1)


def _poll_until_terminal(
    spec_reviewer_task_id: str,
    council_run_id: str,
    timeout_s: int,
    start_time: float,
) -> tuple[dict, dict]:
    """Block until both sides terminal or timeout elapses.

    Returns (opus_raw, council_raw) where each is a dict with keys:
      status, verdict/council_status, issues, confidence, run_id, ...
    """
    opus_result: dict | None = None
    council_result: dict | None = None

    while True:
        elapsed = time.time() - start_time
        timed_out = elapsed >= timeout_s

        # Check spec_reviewer terminal state
        if opus_result is None:
            output_path = _find_reviewer_output(spec_reviewer_task_id)
            if output_path is not None:
                raw = _read_verdict_from_output(output_path)
                failed = str(_CLAUDE_QUEUE_FAILED) in str(output_path) or \
                         str(_GPU_QUEUE_FAILED) in str(output_path)
                opus_result = {
                    "status": "failed" if failed else "processed",
                    "verdict": raw.get("verdict", "error"),
                    "issues": raw.get("issues", []),
                    "confidence": raw.get("confidence", 0.0),
                    "run_id": spec_reviewer_task_id,
                }
            elif timed_out:
                opus_result = {
                    "status": "timeout",
                    "verdict": "timeout",
                    "issues": [],
                    "confidence": 0.0,
                    "run_id": spec_reviewer_task_id,
                }

        # Check council terminal state
        if council_result is None:
            council_yaml_path = _COUNCIL_DIR / f"{council_run_id}.yaml"
            if council_yaml_path.exists():
                try:
                    import yaml as _yaml
                    run = _yaml.safe_load(council_yaml_path.read_text(encoding="utf-8"))
                    status = run.get("status", "")
                    if status in _COUNCIL_TERMINAL:
                        synthesis = run.get("synthesis") or {}
                        # synthesis may be a string (v0 format) or dict (v0.next)
                        if isinstance(synthesis, str):
                            synthesis = _parse_synthesis_str(synthesis)
                        council_result = {
                            "status": status,
                            "landing": synthesis.get("landing", ""),
                            "open_questions": synthesis.get("open_questions", []),
                            "confidence": synthesis.get("confidence", ""),
                            "positions": synthesis.get("positions", []),
                            "run_id": council_run_id,
                        }
                except Exception:
                    pass

            if council_result is None and timed_out:
                council_result = {
                    "status": "timeout",
                    "landing": "",
                    "open_questions": [],
                    "confidence": "",
                    "positions": [],
                    "run_id": council_run_id,
                }

        # Both terminal?
        if opus_result is not None and council_result is not None:
            return opus_result, council_result

        if timed_out:
            # Shouldn't reach here, but guard against logic gaps
            if opus_result is None:
                opus_result = {
                    "status": "timeout", "verdict": "timeout",
                    "issues": [], "confidence": 0.0, "run_id": spec_reviewer_task_id,
                }
            if council_result is None:
                council_result = {
                    "status": "timeout", "landing": "", "open_questions": [],
                    "confidence": "", "positions": [], "run_id": council_run_id,
                }
            return opus_result, council_result

        time.sleep(_POLL_CADENCE_S)


def _parse_synthesis_str(text: str) -> dict:
    """Parse LANDING / OPEN QUESTIONS / CONFIDENCE text format into a dict."""
    landing_m = re.search(
        r"LANDING:\s*(.+?)(?=\n\s*OPEN QUESTIONS:|$)", text, re.IGNORECASE | re.DOTALL
    )
    oq_m = re.search(
        r"OPEN QUESTIONS:\s*(.+?)(?=\n\s*CONFIDENCE:|$)", text, re.IGNORECASE | re.DOTALL
    )
    conf_m = re.search(r"CONFIDENCE:\s*(\w[\w-]*)", text, re.IGNORECASE)

    landing = landing_m.group(1).strip() if landing_m else ""
    oq_raw = oq_m.group(1).strip() if oq_m else ""
    oq_lines = [
        re.sub(r"^[\-\*\d\.]+\s*", "", ln).strip()
        for ln in oq_raw.splitlines()
        if ln.strip() and ln.strip() not in {"-", "none", "None"}
    ]
    confidence = conf_m.group(1).lower() if conf_m else "partial"
    return {"landing": landing, "open_questions": oq_lines, "confidence": confidence, "positions": []}


def _combined_recommendation(
    opus_verdict: str,
    opus_issues: list[dict],
    council_status: str,
    council_positions: list[dict],
) -> Literal["proceed-to-bind", "amend-spec", "shape-with-Erah", "incomplete"]:
    """Deterministic combined recommendation — no LLM call."""
    # incomplete: either side timeout or error
    if opus_verdict in {"timeout", "error"} or council_status in {"timeout", "error"}:
        return "incomplete"

    # shape-with-Erah: opus needs-human, council laid-down, or council open with blocks
    has_block = any(p.get("position") == "block" for p in council_positions)
    if (
        opus_verdict == "needs-human"
        or council_status == "laid-down"
        or (council_status == "open" and has_block)
    ):
        return "shape-with-Erah"

    # amend-spec: opus fixable, council open (no blocks), or any HIGH opus issue
    has_high = any(
        str(i.get("severity", "")).lower() == "high" for i in opus_issues
    )
    if opus_verdict == "fixable" or council_status == "open" or has_high:
        return "amend-spec"

    # proceed-to-bind
    if opus_verdict == "clean" and council_status == "resolved":
        return "proceed-to-bind"

    return "incomplete"


def _build_brief(
    opus_raw: dict,
    council_raw: dict,
    spec_path: Path,
    parsed_target_id: str,
    repo: str,
    elapsed_s: float,
) -> SpecReviewBrief:
    """Assemble SpecReviewBrief from raw poll results."""
    opus_verdict = opus_raw.get("verdict", "error")
    opus_issues = opus_raw.get("issues", [])
    opus_confidence = float(opus_raw.get("confidence", 0.0))
    opus_run_id = opus_raw.get("run_id", "")

    council_status = council_raw.get("status", "error")
    council_landing = council_raw.get("landing", "")
    council_open_questions = council_raw.get("open_questions", [])
    council_confidence = council_raw.get("confidence", "")
    council_positions = council_raw.get("positions", [])
    council_run_id = council_raw.get("run_id", "")

    recommendation = _combined_recommendation(
        opus_verdict, opus_issues, council_status, council_positions
    )

    return SpecReviewBrief(
        spec_path=spec_path,
        target_id=parsed_target_id,
        repo=repo,
        opus_verdict=opus_verdict,
        opus_issues=opus_issues,
        opus_confidence=opus_confidence,
        opus_run_id=opus_run_id,
        council_status=council_status,
        council_landing=council_landing,
        council_open_questions=council_open_questions,
        council_confidence=council_confidence,
        council_positions=council_positions,
        council_run_id=council_run_id,
        elapsed_s=elapsed_s,
        combined_recommendation=recommendation,
    )


def format_brief(brief: SpecReviewBrief) -> str:
    """Render SpecReviewBrief as markdown for stdout."""
    issues_lines = (
        "\n".join(
            f"    - [{i.get('severity','?').upper()}] "
            f"{i.get('citation', i.get('path','?'))}: {i.get('note','')}"
            for i in brief.opus_issues
        )
        if brief.opus_issues
        else "    - (none)"
    )

    positions_lines = (
        "\n".join(
            f"    - {p.get('entity', p.get('id','?'))}: "
            f"{p.get('position','?')} — {p.get('reason', p.get('note',''))}"
            for p in brief.council_positions
        )
        if brief.council_positions
        else "    - (none)"
    )

    oq_lines = (
        "\n".join(f"    - {q}" for q in brief.council_open_questions)
        if brief.council_open_questions
        else "    - (none)"
    )

    # Plain-English suggested next step — mention reservations for converged-with-reservation
    step_map = {
        "proceed-to-bind": (
            "Both passes returned clean signals. "
            "Proceed to `lapis-pm bind` after Erah confirms."
        ),
        "amend-spec": (
            "One or both passes flagged issues that can be fixed in the spec. "
            "Address the noted issues, update the spec, then re-run spec-review or proceed to bind "
            "with Erah's judgment."
        ),
        "shape-with-Erah": (
            "A pass returned a signal that warrants direct human judgment. "
            "Review the findings with Erah before proceeding. "
            "Do not bind until Erah explicitly clears the concern."
        ),
        "incomplete": (
            "One or both passes did not complete. "
            "Re-run spec-review, or proceed with caution after reviewing whatever completed."
        ),
    }
    next_step = step_map.get(brief.combined_recommendation, "")
    # Reservation note for converged-with-reservation
    if (
        brief.combined_recommendation == "proceed-to-bind"
        and brief.council_confidence == "converged-with-reservation"
    ):
        stood_aside = [
            p for p in brief.council_positions if p.get("position") == "stand-aside"
        ]
        if stood_aside:
            names = ", ".join(
                p.get("entity", p.get("id", "?")) for p in stood_aside
            )
            next_step += (
                f" Note: {names} stood aside (converged-with-reservation). "
                "Their reservation is noted in §Mirror Council above."
            )

    return f"""# Spec Review: {brief.target_id}

**Spec:** {brief.spec_path}
**Repo:** {brief.repo}
**Elapsed:** {brief.elapsed_s:.1f}s
**Recommendation:** {brief.combined_recommendation}

## Opus technical review
- **Verdict:** {brief.opus_verdict} (confidence {brief.opus_confidence:.2f})
- **Run ID:** {brief.opus_run_id}
- **Issues:**
{issues_lines}

## Mirror Council deliberation
- **Status:** {brief.council_status}, confidence {brief.council_confidence}
- **Run ID:** {brief.council_run_id}
- **Landing:** {brief.council_landing or '(none)'}
- **Positions:**
{positions_lines}
- **Open questions:**
{oq_lines}

## Suggested next step
{next_step}
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_spec_review(
    spec_path: Path,
    council_voicing: str = "local",
    timeout_s: int = 900,
    repo_override: str | None = None,
) -> SpecReviewBrief:
    """Run parallel Opus + Council review of a spec document. Synchronous."""
    start_time = time.time()

    # 1. Parse frontmatter
    parsed_target_id, parsed_repo = _parse_spec_frontmatter(spec_path)
    repo = repo_override if repo_override else parsed_repo

    # 2. Generate synthetic target_id (never written to TargetStore)
    synth_tid = _synth_target_id(parsed_target_id)

    # 3. Load invariant context
    invariant_context = _load_invariant_context(repo)

    # 4. Read full spec text
    spec_text = spec_path.read_text(encoding="utf-8")

    # 5. Dispatch spec_reviewer (Opus pass)
    dispatch_result = _dispatch_spec_reviewer(
        spec_text, synth_tid, parsed_target_id, repo, invariant_context
    )

    # 6. Dispatch council deliberation
    council_run_id = _dispatch_council(
        spec_text, parsed_target_id, invariant_context, council_voicing
    )

    # 7. Poll until terminal
    opus_raw, council_raw = _poll_until_terminal(
        dispatch_result.task_id, council_run_id, timeout_s, start_time
    )

    elapsed = time.time() - start_time

    # 8. Build and return brief
    return _build_brief(opus_raw, council_raw, spec_path, parsed_target_id, repo, elapsed)
