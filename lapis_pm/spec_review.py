"""lapis_pm.spec_review — Pre-bind Facets + Mirror Council review.

Default gate: Facets (technical-integrity + trickster personas, Haiku) for the PM /
technical domain, plus a Mirror Council deliberation for invariant-fit / meaning, plus
a standing Sonnet deep-reviewer leg that fires by default for advisory/hold specs as a
reference-only signal (never moves the recommendation — Facets + Council are the sole
drivers). The Sonnet leg can be disabled per-run via --no-sonnet-reviewer or
SPEC_REVIEW_SONNET_DISABLED=1.

Public entry point: run_spec_review(spec_path, council_voicing, timeout_s, repo_override,
authority, dispatch_facets, sonnet_reviewer, compare_opus). Returns SpecReviewBrief.
Synchronous; caller blocks until all dispatched passes complete or timeout.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

# Import the canonical facets deploy clone path from pm_core (source-enforce coupling).
# pm_core has no module-level spec_review import, so this is circular-free.
from lapis_pm.pm_core import _FACETS_DEPLOY_CLONE


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_CLAUDE_QUEUE_COMPLETED = Path("/srv/lapis/claude-queue/completed")
_CLAUDE_QUEUE_FAILED = Path("/srv/lapis/claude-queue/failed")
_GPU_QUEUE_COMPLETED = Path("/srv/lapis/gpu-queue/completed")
_GPU_QUEUE_FAILED = Path("/srv/lapis/gpu-queue/failed")
_COUNCIL_DIR = Path("/srv/lapis/council")
# Facets deploy clone path — the same path the post-land pull deploys and the gate injects on PYTHONPATH.
_FACETS_REPO_PATH = Path(_FACETS_DEPLOY_CLONE)

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
    # Required fields
    spec_path: Path
    target_id: str
    repo: str
    council_status: Literal["resolved", "open", "laid-down", "failed", "timeout", "error", "closed"]
    council_landing: str
    council_open_questions: list[str]
    council_confidence: str
    council_positions: list[dict]
    council_run_id: str
    elapsed_s: float
    combined_recommendation: Literal[
        "proceed-to-bind", "amend-spec", "shape-with-Erah", "incomplete", "parse_failed",
    ]
    # Sonnet deep-reviewer fields (reference-only; never steer the recommendation)
    sonnet_verdict: str = "error"
    sonnet_issues: list[dict] = field(default_factory=list)
    sonnet_confidence: float = 0.0
    sonnet_run_id: str = ""
    parse_error: dict | None = None
    facets_deliberation: dict | None = None  # FacetsDeliberation envelope; None if disabled/timeout
    # True whenever the Sonnet leg ran (always reference-only now)
    sonnet_advisory_only: bool = False
    facets_operator: str = "haiku"  # operator used for Facets personas + synthesis

    # ------------------------------------------------------------------
    # Deprecated read-aliases — remove 90 days after merge (2026-09-05).
    # Before deletion: grep the tree to confirm no callers remain.
    # ------------------------------------------------------------------

    @property
    def opus_verdict(self) -> str:
        """Deprecated alias for sonnet_verdict. Remove 90 days after merge."""
        return self.sonnet_verdict

    @property
    def opus_issues(self) -> list[dict]:
        """Deprecated alias for sonnet_issues. Remove 90 days after merge."""
        return self.sonnet_issues

    @property
    def opus_confidence(self) -> float:
        """Deprecated alias for sonnet_confidence. Remove 90 days after merge."""
        return self.sonnet_confidence

    @property
    def opus_run_id(self) -> str:
        """Deprecated alias for sonnet_run_id. Remove 90 days after merge."""
        return self.sonnet_run_id

    @property
    def opus_advisory_only(self) -> bool:
        """Deprecated alias for sonnet_advisory_only. Remove 90 days after merge."""
        return self.sonnet_advisory_only


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


def _parse_spec_authority(spec_path: Path) -> str:
    """Extract Authority from spec frontmatter (first 50 lines).

    Pattern: **Authority:** <auto-merge|advisory|hold> (followed by whitespace or end)
    Raises SpecFrontmatterError if not found.
    """
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as e:
        raise SpecFrontmatterError(f"cannot read spec file: {e}") from e

    first_50 = "\n".join(text.splitlines()[:50])
    m = re.search(
        r"^\*\*Authority:\*\*\s+(auto-merge|advisory|hold)\b",
        first_50,
        re.MULTILINE,
    )
    if not m:
        raise SpecFrontmatterError(
            f"missing **Authority:** in {spec_path} "
            "(expected **Authority:** <auto-merge|advisory|hold> at line start, "
            "e.g. '**Authority:** advisory — ...')"
        )
    return m.group(1)


def _dispatch_facets(
    spec_text: str,
    parsed_target_id: str,
    repo: str,
    authority: str,
    start_time: float,
    facets_operator: str = "haiku",
) -> str | None:
    """Invoke Facets deliberation synchronously via subprocess.

    Shells out to `python3 -m facets.adapter deliberate`. Blocks until Facets
    completes or times out (10 min). Returns the deliberation_id if successful;
    None on error, timeout, or if Facets is disabled via FACETS_DISPATCH_DISABLED=1.

    Deliberation envelope is written to /srv/lapis/facets/deliberations/ by the adapter.

    Spec content contract: spec_text is written into the context JSON file under
    the "spec_text" key so that CompositionPersona.deliberate() can inline it
    directly into each persona's prompt. This is the established channel for
    structured context (--context-file); stdin is not used. See:
    /srv/lapis/planning/specs/spec-review-facets-context-injection-v0.md

    Panel composition: only technical-integrity and trickster are invoked.
    mirror-rep is excluded because it is corpus-backed over /srv/lapis/council/speakers/
    (canonical/historical) and does not engage with novel-spec content by design.
    Including mirror-rep in spec-review panels produces stale output from prior
    deliberations rather than analysis of the spec under review. This exclusion is
    local to pre-bind spec-review; mirror-rep's use in other deliberation paths is
    unaffected.
    """
    if os.getenv("FACETS_DISPATCH_DISABLED") == "1":
        return None

    import subprocess
    import json as _json
    import tempfile

    try:
        context = {
            "source": "pre-bind-wire",
            "spec_path": f"/srv/lapis/planning/specs/{parsed_target_id}.md",
            "spec_text": spec_text,
            "additional_context": f"repo={repo}, authority={authority}",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as ctx_f:
            _json.dump(context, ctx_f)
            context_file = ctx_f.name

        try:
            facets_env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    p for p in (str(_FACETS_REPO_PATH), os.environ.get("PYTHONPATH", "")) if p
                ),
            }
            argv = [
                "python3", "-m", "facets.adapter", "deliberate",
                (
                    f"Scope and timing judgment for {parsed_target_id} "
                    f"(authority: {authority}, repo: {repo}). "
                    f"Review the spec for portfolio fit, risk-reward, and readiness."
                ),
                "--context-file", context_file,
                "--personas", "technical-integrity,trickster",
                "--format", "json",
            ]
            if facets_operator != "haiku":
                argv += ["--persona-operator", facets_operator, "--synthesis-operator", facets_operator]
            target_repo_path = Path(f"/srv/git/{repo}-working")
            if os.getenv("FACETS_GROUNDING_DISABLED") == "1":
                pass
            elif target_repo_path.is_dir():
                argv += ["--target-repo", str(target_repo_path)]
            else:
                print(
                    f"[spec-review:facets] no working tree for repo {repo!r} at "
                    f"{target_repo_path} - Mode-1 grounding inert",
                    file=sys.stderr,
                )
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=600,
                env=facets_env,
            )
        finally:
            try:
                os.unlink(context_file)
            except OSError:
                pass

        if result.returncode != 0:
            print(
                f"[spec-review:facets-dispatch-error] subprocess exited "
                f"{result.returncode}: {result.stderr}",
                file=sys.stderr,
            )
            return None

        deliberation_json = _json.loads(result.stdout)
        deliberation_id = deliberation_json.get("deliberation_id")
        if not deliberation_id:
            print(
                "[spec-review:facets-dispatch-error] no deliberation_id in output",
                file=sys.stderr,
            )
            return None

        elapsed_s = int(time.time() - start_time)
        print(
            f"[spec-review:facets-complete] deliberation_id={deliberation_id} "
            f"elapsed={elapsed_s}s",
            file=sys.stderr,
        )
        return deliberation_id

    except subprocess.TimeoutExpired:
        print(
            "[spec-review:facets-timeout] Facets did not complete within 10 minutes",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"[spec-review:facets-dispatch-error] {e}",
            file=sys.stderr,
        )
        return None


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


def _iter_fenced_json_bodies(text: str) -> Iterator[str]:
    """Yield each ```(json)? fence body found in text, in document order."""
    for m in re.finditer(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL):
        yield m.group(1)


def _iter_balanced_json_candidates(text: str) -> list[str]:
    """Return balanced {...} substrings from text, sorted largest-first.

    Pre-strips fenced blocks so this function only sees non-fenced content
    (Strategy 1 already tried fenced bodies; this handles prose-embedded JSON).
    State machine skips braces inside single-backtick spans and JSON string literals.
    """
    # Pre-strip fenced blocks
    de_fenced = re.sub(r"```(?:json)?\s*\n.*?\n```", "", text, flags=re.DOTALL)

    candidates: list[str] = []
    depth = 0
    start: int | None = None
    in_backtick = False
    in_string = False
    i = 0
    while i < len(de_fenced):
        ch = de_fenced[i]
        if ch == "`" and not in_string:
            in_backtick = not in_backtick
        elif ch == '"' and not in_backtick:
            # Count preceding backslashes to detect escaped quote
            num_bs = 0
            j = i - 1
            while j >= 0 and de_fenced[j] == "\\":
                num_bs += 1
                j -= 1
            if num_bs % 2 == 0:  # unescaped quote
                in_string = not in_string
        elif not in_backtick and not in_string:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(de_fenced[start : i + 1])
                    start = None
        i += 1

    candidates.sort(key=len, reverse=True)
    return candidates


def _read_verdict_from_output(output_path: Path) -> dict:
    """Parse the verdict JSON from the spec_reviewer output file.

    Multi-strategy pipeline — handles model narrate-then-emit patterns:
    1. Extract from ```(json)? fenced blocks (handles "narration + fence" pattern).
    2. Direct json.loads on stripped content (handles strict JSON-only output).
    3. Bracket-count with backtick/string-literal awareness, try largest-first
       (handles fence-less JSON embedded in prose).
    4. Return parse_failed envelope (richer diagnostics than bare "error").
    """
    content = output_path.read_text(encoding="utf-8")
    stripped = content.strip()

    # Strategy 1: fenced extraction
    for fence_body in _iter_fenced_json_bodies(stripped):
        try:
            return json.loads(fence_body)
        except json.JSONDecodeError:
            continue

    # Strategy 2: direct JSON parse on stripped content
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: bracket-count, skipping backtick spans and string literals,
    #             try each balanced {...} candidate from largest to smallest
    for candidate in _iter_balanced_json_candidates(content):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # All strategies failed: richer parse_failed envelope so operators can
    # distinguish "delivery path lost data" from "underlying agent failed"
    return {
        "verdict": "parse_failed",
        "issues": [],
        "confidence": 0.0,
        "parse_error": {
            "file_size": len(content),
            "head": content[:200],
            "tail": content[-200:],
        },
    }


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
        f"The technical-soundness question is being handled in parallel by a Sonnet pass.\n"
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
    council_run_id: str,
    timeout_s: int,
    start_time: float,
    spec_reviewer_task_id: str | None = None,
) -> tuple[dict | None, dict]:
    """Block until Council terminal (and optionally Sonnet) or timeout elapses.

    When spec_reviewer_task_id is None (Facets-only mode), only Council is polled;
    sonnet_result is returned as None. When spec_reviewer_task_id is provided
    (Sonnet deep-reviewer mode), both sides are polled.

    Returns (sonnet_raw_or_None, council_raw).
    """
    sonnet_result: dict | None = None
    council_result: dict | None = None
    # When no Sonnet leg, mark it immediately "done"
    sonnet_already_done = spec_reviewer_task_id is None

    while True:
        elapsed = time.time() - start_time
        timed_out = elapsed >= timeout_s

        # Check spec_reviewer terminal state (only when Sonnet was dispatched)
        if not sonnet_already_done and sonnet_result is None:
            output_path = _find_reviewer_output(spec_reviewer_task_id)  # type: ignore[arg-type]
            if output_path is not None:
                raw = _read_verdict_from_output(output_path)
                failed = str(_CLAUDE_QUEUE_FAILED) in str(output_path) or \
                         str(_GPU_QUEUE_FAILED) in str(output_path)
                sonnet_result = {
                    "status": "failed" if failed else "processed",
                    "verdict": raw.get("verdict", "error"),
                    "issues": raw.get("issues", []),
                    "confidence": raw.get("confidence", 0.0),
                    "run_id": spec_reviewer_task_id,
                    "parse_error": raw.get("parse_error"),
                }
                elapsed_s = int(time.time() - start_time)
                print(
                    f"[spec-review:sonnet-complete] task_id={spec_reviewer_task_id} "
                    f"elapsed={elapsed_s}s verdict={sonnet_result['verdict']}",
                    file=sys.stderr,
                )
            elif timed_out:
                sonnet_result = {
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
                        elapsed_s = int(time.time() - start_time)
                        print(
                            f"[spec-review:council-complete] run_id={council_run_id} "
                            f"elapsed={elapsed_s}s status={status}",
                            file=sys.stderr,
                        )
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

        # Determine effective sonnet done-ness for convergence + logging
        sonnet_effective_done = sonnet_already_done or sonnet_result is not None

        # Both terminal?
        if sonnet_effective_done and council_result is not None:
            if timed_out:
                sonnet_done = sonnet_already_done or (
                    sonnet_result is not None and sonnet_result.get("status") != "timeout"
                )
                council_done = council_result.get("status") != "timeout"
                elapsed_s = int(time.time() - start_time)
                print(
                    f"[spec-review:timeout] elapsed={elapsed_s}s "
                    f"sonnet_done={sonnet_done} council_done={council_done}",
                    file=sys.stderr,
                )
            return sonnet_result, council_result

        if timed_out:
            # Shouldn't reach here, but guard against logic gaps
            if not sonnet_already_done and sonnet_result is None:
                sonnet_result = {
                    "status": "timeout", "verdict": "timeout",
                    "issues": [], "confidence": 0.0,
                    "run_id": spec_reviewer_task_id or "",
                }
            if council_result is None:
                council_result = {
                    "status": "timeout", "landing": "", "open_questions": [],
                    "confidence": "", "positions": [], "run_id": council_run_id,
                }
            return sonnet_result, council_result

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
    sonnet_verdict: str = "skip",
    sonnet_issues: list[dict] = (),
    council_status: str = "error",
    council_positions: list[dict] = (),
    facets_escalation: str | None = None,
    facets_unreliable: bool = False,
    authority: str = "advisory",
    # Deprecated aliases — kept for callers that haven't migrated yet
    opus_verdict: str | None = None,
    opus_issues: list[dict] | None = None,
) -> Literal["proceed-to-bind", "amend-spec", "shape-with-Erah", "incomplete", "parse_failed"]:
    """Deterministic combined recommendation — Facets (PM) + Council (philosophical).

    sonnet_verdict="skip" signals that the Sonnet leg was not dispatched (Facets mode).
    The Sonnet leg is always reference-only: even when present, its verdict/issues are
    fed the "skip" sentinel into the recommendation so they cannot move the gate.
    Facets + Council are the sole drivers.

    When facets_escalation is provided and authority is advisory/hold, Facets
    signals take precedence. Falls through to Council-only logic on
    facets_escalation="proceed", when Facets is absent, or when
    facets_unreliable=True (parse failures present — Council-only branches apply).
    """
    # Support deprecated opus_* parameter aliases
    if opus_verdict is not None:
        sonnet_verdict = opus_verdict
    if opus_issues is not None:
        sonnet_issues = opus_issues

    # parse_failed: parser could not extract a verdict
    if sonnet_verdict == "parse_failed":
        return "parse_failed"

    # Facets logic — gated on authority, escalation signal, and reliability
    if authority in {"advisory", "hold"} and facets_escalation and not facets_unreliable:
        if facets_escalation == "claude-max":
            return "shape-with-Erah"
        if facets_escalation == "investigate-first":
            return "amend-spec"
        if facets_escalation == "brief-to-Erah":
            return "shape-with-Erah"
        # proceed → fall through to Council signal

    # Council timeout/error → incomplete
    if council_status in {"timeout", "error"}:
        return "incomplete"

    # Sonnet timeout/error → incomplete (only when Sonnet was dispatched, non-advisory-only)
    # NOTE: in the current design, sonnet_verdict is always fed as "skip" to this function
    # (advisory-only mode), so this branch only triggers in legacy non-advisory-only calls.
    if sonnet_verdict not in {"skip"} and sonnet_verdict in {"timeout", "error"}:
        return "incomplete"

    # shape-with-Erah: council laid-down, council open with blocks, or Sonnet needs-human
    has_block = any(p.get("position") == "block" for p in council_positions)
    if council_status == "laid-down" or (council_status == "open" and has_block):
        return "shape-with-Erah"
    if sonnet_verdict not in {"skip"} and sonnet_verdict == "needs-human":
        return "shape-with-Erah"

    # amend-spec: council open, or Sonnet fixable/HIGH-issue (when non-advisory-only)
    has_high = any(str(i.get("severity", "")).lower() == "high" for i in sonnet_issues)
    if council_status == "open":
        return "amend-spec"
    if sonnet_verdict not in {"skip"} and (sonnet_verdict == "fixable" or has_high):
        return "amend-spec"

    # proceed-to-bind: council resolved + (Sonnet clean or Sonnet not dispatched)
    if council_status == "resolved" and sonnet_verdict in {"skip", "clean"}:
        return "proceed-to-bind"

    return "incomplete"


def _build_brief(
    council_raw: dict,
    spec_path: Path,
    parsed_target_id: str,
    repo: str,
    elapsed_s: float,
    sonnet_raw: dict | None = None,
    facets_deliberation: dict | None = None,
    authority: str = "advisory",
    sonnet_advisory_only: bool = False,
    facets_operator: str = "haiku",
    # Deprecated parameter aliases — kept for callers that haven't migrated yet
    opus_raw: dict | None = None,
    opus_advisory_only: bool | None = None,
) -> SpecReviewBrief:
    """Assemble SpecReviewBrief from Facets + Council (and optionally Sonnet) results.

    sonnet_raw is optional — None when the Sonnet leg was not dispatched.
    facets_deliberation is the FacetsDeliberation envelope dict; None if disabled/timeout.
    sonnet_advisory_only — the Sonnet leg is always reference-only: its output is
    rendered for reference but excluded from the combined recommendation (Facets +
    Council drive the gate). This flag is set True whenever the Sonnet leg ran.
    """
    # Support deprecated opus_* parameter aliases
    if opus_raw is not None and sonnet_raw is None:
        sonnet_raw = opus_raw
    if opus_advisory_only is not None and sonnet_advisory_only is False:
        sonnet_advisory_only = opus_advisory_only

    # Sonnet fields — default to "skip" sentinel when not dispatched
    sonnet_verdict = sonnet_raw.get("verdict", "error") if sonnet_raw is not None else "skip"
    sonnet_issues = sonnet_raw.get("issues", []) if sonnet_raw is not None else []
    sonnet_confidence = float(sonnet_raw.get("confidence", 0.0)) if sonnet_raw is not None else 0.0
    sonnet_run_id = sonnet_raw.get("run_id", "") if sonnet_raw is not None else ""

    council_status = council_raw.get("status", "error")
    council_landing = council_raw.get("landing", "")
    council_open_questions = council_raw.get("open_questions", [])
    council_confidence = council_raw.get("confidence", "")
    council_positions = council_raw.get("positions", [])
    council_run_id = council_raw.get("run_id", "")

    # Extract Facets escalation signal and compute reliability
    facets_escalation = None
    facets_unreliable = False
    if facets_deliberation:
        synthesis = facets_deliberation.get("synthesis") or {}
        if isinstance(synthesis, dict):
            facets_escalation = synthesis.get("escalation_recommendation")
        stances = facets_deliberation.get("stances") or []
        facets_unreliable = (
            bool(synthesis.get("parse_failed"))
            or any(s.get("parse_failed") for s in stances)
        )

    # The Sonnet leg is always reference-only: feed the "skip" sentinel into the
    # recommendation so any Sonnet verdict/issues/timeout cannot move the gate.
    # Facets + Council remain the sole drivers. The real Sonnet fields are still
    # stored on the brief for rendering.
    rec_sonnet_verdict = "skip" if sonnet_advisory_only else sonnet_verdict
    rec_sonnet_issues = () if sonnet_advisory_only else sonnet_issues

    recommendation = _combined_recommendation(
        sonnet_verdict=rec_sonnet_verdict,
        sonnet_issues=rec_sonnet_issues,
        council_status=council_status,
        council_positions=council_positions,
        facets_escalation=facets_escalation,
        facets_unreliable=facets_unreliable,
        authority=authority,
    )

    return SpecReviewBrief(
        spec_path=spec_path,
        target_id=parsed_target_id,
        repo=repo,
        council_status=council_status,
        council_landing=council_landing,
        council_open_questions=council_open_questions,
        council_confidence=council_confidence,
        council_positions=council_positions,
        council_run_id=council_run_id,
        elapsed_s=elapsed_s,
        combined_recommendation=recommendation,
        sonnet_verdict=sonnet_verdict,
        sonnet_issues=sonnet_issues,
        sonnet_confidence=sonnet_confidence,
        sonnet_run_id=sonnet_run_id,
        parse_error=sonnet_raw.get("parse_error") if sonnet_raw is not None else None,
        facets_deliberation=facets_deliberation,
        sonnet_advisory_only=sonnet_advisory_only,
        facets_operator=facets_operator,
    )


def format_brief(brief: SpecReviewBrief) -> str:
    """Render SpecReviewBrief as markdown for stdout."""
    issues_lines = (
        "\n".join(
            f"    - [{i.get('severity','?').upper()}] "
            f"{i.get('citation', i.get('path','?'))}: {i.get('note','')}"
            for i in brief.sonnet_issues
        )
        if brief.sonnet_issues
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
        "parse_failed": (
            "The spec_reviewer agent likely succeeded but the parser could not extract "
            "a JSON verdict. See the Parse Error block below for diagnostics. "
            "After the runner fix lands (`spec-review-output-truncation-runner`), re-run spec-review."
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

    parse_error_block = ""
    if brief.parse_error:
        pe = brief.parse_error
        parse_error_block = f"""
## Parse error
- **File size:** {pe.get('file_size', 0)} bytes
- **Head (first 200 chars):** {pe.get('head', '')}
- **Tail (last 200 chars):** {pe.get('tail', '')}
- **Note:** the spec_reviewer agent likely succeeded; the orchestration \
could not extract a JSON verdict from the output. See chain-sibling \
`spec-review-output-truncation-runner` for the delivery-path fix.
"""

    # Facets section — omitted if facets_deliberation is None
    facets_section = ""
    if brief.facets_deliberation:
        fd = brief.facets_deliberation
        syn = fd.get("synthesis", {}) if isinstance(fd.get("synthesis"), dict) else {}
        stances = fd.get("stances", [])

        # Compute unreliability from the stored deliberation
        synth_parse_failed = bool(syn.get("parse_failed"))
        failed_stances = [s for s in stances if s.get("parse_failed")]
        facets_unreliable = synth_parse_failed or bool(failed_stances)

        unreliable_header_line = (
            "\n**Facets leg unreliable; combined recommendation derived from Council only**"
            if facets_unreliable else ""
        )

        reliable_line = ""
        if synth_parse_failed:
            reliable_line = (
                f"- **Reliable:** no  (synthesis parse_failed: {syn.get('parse_error', '')})\n"
            )
        elif facets_unreliable:
            reliable_line = "- **Reliable:** no  (persona parse failures)\n"

        failed_personas_line = ""
        if failed_stances:
            fp_entries = ", ".join(
                f"{s.get('persona', '?')} ({s.get('parse_error', 'unknown')})"
                for s in failed_stances
            )
            failed_personas_line = f"- **Personas with parse failures:** {fp_entries}\n"

        stances_md = "\n".join(
            f"    - **{s.get('persona', '?')}** "
            f"({'verified' if s.get('verified') else 'inferred'}/{s.get('confidence', '?')}): "
            f"{s.get('claim', '')}"
            for s in stances
        ) or "    - (none)"

        # Uncertainty bounds subsection
        scope_limits = syn.get("scope_limits") or syn.get("uncertainty")
        if scope_limits:
            uncertainty_line = f"- **Uncertainty bounds:** {scope_limits}"
        elif brief.facets_operator == "haiku":
            uncertainty_line = (
                "- **Uncertainty bounds:** not surfaced by this operator"
                " — consider re-running with sonnet"
            )
        else:
            uncertainty_line = "- **Uncertainty bounds:** not reported by synthesis"

        facets_section = f"""
## Facets deliberation (PM domain) [operator: {brief.facets_operator}]{unreliable_header_line}
{reliable_line}- **Consensus level:** {syn.get('consensus_level', '?')}
- **Escalation:** {syn.get('escalation_recommendation', 'proceed')}
- **Confidence:** {syn.get('confidence', '?')}
- **Recommendation:** {syn.get('recommendation', '')}
- **Stances:**
{stances_md}
{uncertainty_line}
{failed_personas_line}- **Run ID:** {fd.get('deliberation_id', '')}
"""

    # Sonnet section — omitted only when the leg did not run (verdict="skip").
    # This section is always reference-only (purpose stated in heading).
    # IMPORTANT: always-render is load-bearing, not cosmetic — the whole point is a
    # standing deep signal in every gate run. Do not remove this section or gate it
    # on any flag other than sonnet_verdict == "skip".
    sonnet_section = ""
    if brief.sonnet_verdict != "skip":
        sonnet_section = f"""
## Sonnet technical review — reference only (does not affect recommendation)
- **Verdict:** {brief.sonnet_verdict} (confidence {brief.sonnet_confidence:.2f})
- **Run ID:** {brief.sonnet_run_id}
- **Issues:**
{issues_lines}
"""

    return f"""# Spec Review: {brief.target_id}

**Spec:** {brief.spec_path}
**Repo:** {brief.repo}
**Elapsed:** {brief.elapsed_s:.1f}s
**Recommendation:** {brief.combined_recommendation}
{facets_section}{sonnet_section}
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
{parse_error_block}"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_spec_review(
    spec_path: Path,
    council_voicing: str = "gravitywell",
    timeout_s: int = 1800,
    repo_override: str | None = None,
    authority: str | None = None,
    dispatch_facets: bool = True,
    sonnet_reviewer: bool = True,
    facets_operator: str = "haiku",
    # Deprecated parameter — kept for back-compat; no-op (sonnet_reviewer is always-on)
    compare_opus: bool = False,
) -> SpecReviewBrief:
    """Run Facets (PM) + Council (philosophical) + Sonnet deep-reviewer review. Synchronous.

    The Sonnet spec_reviewer leg fires by default for advisory/hold specs (sonnet_reviewer=True).
    It is reference-only and never moves combined_recommendation — Facets + Council are
    the sole recommendation drivers. Disable with sonnet_reviewer=False or
    SPEC_REVIEW_SONNET_DISABLED=1.

    Facets deliberation is synchronous (blocks ~5 min); Council is async (polled up to
    timeout). The Sonnet leg is dispatched async BEFORE the Facets block (so it runs
    concurrently with Facets' blocking subprocess), polled alongside Council.

    dispatch_facets=False or FACETS_DISPATCH_DISABLED=1 skips Facets entirely (for
    smoke or testing). authority defaults to None — parsed from spec frontmatter; falls
    back to "advisory" if not found. Only advisory/hold specs dispatch Facets and Sonnet.

    facets_operator controls the Facets persona + synthesis model; default haiku.

    compare_opus is accepted for back-compat but is a no-op — the Sonnet leg is already
    on by default, so passing compare_opus=True has no additional effect. A deprecation
    note is emitted to stderr. Sunset: remove 90 days after merge (2026-09-05).
    """
    # Emit deprecation note when the caller still passes compare_opus=True
    if compare_opus:
        print(
            "[spec-review] WARNING: --compare-opus / compare_opus is deprecated and has no effect; "
            "the Sonnet reference leg now runs by default (sunset 90 days after merge).",
            file=sys.stderr,
        )

    start_time = time.time()

    # 1. Parse frontmatter
    parsed_target_id, parsed_repo = _parse_spec_frontmatter(spec_path)
    repo = repo_override if repo_override else parsed_repo

    # 2. Parse authority from spec (if not provided)
    effective_authority = authority
    if effective_authority is None:
        try:
            effective_authority = _parse_spec_authority(spec_path)
        except SpecFrontmatterError:
            effective_authority = "advisory"  # default when not found

    # 3. Read full spec text
    spec_text = spec_path.read_text(encoding="utf-8")

    # 4. Load invariant context
    invariant_context = _load_invariant_context(repo)

    # 4b. Sonnet deep-reviewer leg: dispatch async BEFORE the Facets block so it runs
    #     concurrently with Facets' blocking subprocess. Always-on for advisory/hold;
    #     reference-only (never moves the recommendation). Intentional — do not remove.
    spec_reviewer_task_id: str | None = None
    sonnet_disabled = (not sonnet_reviewer) or os.getenv("SPEC_REVIEW_SONNET_DISABLED") == "1"
    do_sonnet = (not sonnet_disabled) and effective_authority in {"advisory", "hold"}
    if do_sonnet:
        synth_target_id = _synth_target_id(parsed_target_id)
        try:
            spec_reviewer_task_id = _dispatch_spec_reviewer(
                spec_text=spec_text,
                synth_target_id=synth_target_id,
                parsed_target_id=parsed_target_id,
                repo=repo,
                invariant_context=invariant_context,
            ).task_id
            print(
                f"[spec-review:sonnet-reviewer] dispatched reference Sonnet pass "
                f"task_id={spec_reviewer_task_id}",
                file=sys.stderr,
            )
        except Exception as e:
            # Sonnet leg is reference-only; never let its dispatch failure abort the gate.
            print(
                f"[spec-review:sonnet-reviewer-dispatch-error] {e} — continuing Facets-only",
                file=sys.stderr,
            )
            spec_reviewer_task_id = None

    # 5. Dispatch Facets (synchronous; blocks until complete or timeout)
    facets_deliberation: dict | None = None
    if dispatch_facets and effective_authority in {"advisory", "hold"}:
        facets_deliberation_id = _dispatch_facets(
            spec_text, parsed_target_id, repo, effective_authority, start_time,
            facets_operator=facets_operator,
        )
        if facets_deliberation_id:
            try:
                import json as _json
                facets_path = Path(f"/srv/lapis/facets/deliberations/{facets_deliberation_id}.json")
                facets_deliberation = _json.loads(facets_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(
                    f"[spec-review:facets-read-error] could not read Facets result: {e}",
                    file=sys.stderr,
                )

    # 6. Dispatch Council (async; parallel with any remaining work)
    council_run_id = _dispatch_council(
        spec_text, parsed_target_id, invariant_context, council_voicing
    )

    # 7. Poll until terminal. When Sonnet leg was dispatched, both Sonnet and Council
    #    are polled; otherwise only Council is polled and sonnet_raw stays None.
    sonnet_raw, council_raw = _poll_until_terminal(
        council_run_id=council_run_id,
        timeout_s=timeout_s,
        start_time=start_time,
        spec_reviewer_task_id=spec_reviewer_task_id,
    )

    elapsed = time.time() - start_time

    # 8. Build and return brief. sonnet_raw is non-None when the Sonnet leg ran; it is
    #    rendered for reference but excluded from the recommendation (sonnet_advisory_only).
    return _build_brief(
        council_raw=council_raw,
        spec_path=spec_path,
        parsed_target_id=parsed_target_id,
        repo=repo,
        elapsed_s=elapsed,
        sonnet_raw=sonnet_raw,
        facets_deliberation=facets_deliberation,
        authority=effective_authority,
        sonnet_advisory_only=sonnet_raw is not None,
        facets_operator=facets_operator,
    )
