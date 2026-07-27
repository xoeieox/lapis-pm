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
import asyncio
import contextlib
import fcntl
import io
import json
import os
import re
import socket
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

# Import the canonical facets deploy clone path from pm_core (source-enforce coupling).
# pm_core has no module-level spec_review import, so this is circular-free.
from lapis_pm.pm_core import _FACETS_DEPLOY_CLONE
from agents_core.shared_deliberation.orchestrator import (
    run_deliberation,
    init_facets_semaphore,
)
from agents_core.shared_deliberation.envelope import DeliberationRequest
from agents_core.llm import swarm_model, swarm_serving
from agents_core.room_paths import room_path, room_str


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_CLAUDE_QUEUE_COMPLETED = room_path('claude_queue.completed')
_CLAUDE_QUEUE_FAILED = room_path('claude_queue.failed')
_GPU_QUEUE_COMPLETED = room_path('gpu_queue.completed')
_GPU_QUEUE_FAILED = room_path('gpu_queue.failed')
_COUNCIL_DIR = room_path('council')
# Facets deploy clone path — the same path the post-land pull deploys and the gate injects on PYTHONPATH.
_FACETS_REPO_PATH = Path(_FACETS_DEPLOY_CLONE)

# Council terminal statuses — per v0.next spec + "closed" for scene mode.
_COUNCIL_TERMINAL: frozenset[str] = frozenset(
    {"resolved", "open", "laid-down", "failed", "closed", "timeout", "error"}
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
_SPEC_REVIEW_LOCK_TIMEOUT_DEFAULT = 900  # seconds; longer than a normal ~12-min review
_ELEVATOR_GROUNDING_POLL_CADENCE_S = 5  # inter-poll sleep for grounding polls



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
    # Effective voicing/operator fields (read from provenance, unknown if absent)
    council_voicing_requested: str = "gravitywell"  # what was requested
    council_voicing_effective: str = "unknown"  # what actually ran (from run YAML)
    council_voicing_degraded: bool = False  # whether it fell back
    council_voicing_degraded_reason: str = ""  # reason for fallback (gw_not_serving, etc.)
    council_error_reason: str = ""  # infra failure reason; set ONLY when council leg failed outright or was not run
    facets_operator_requested: str = ""  # what was requested (empty if not degraded)
    facets_operator_effective: str = "unknown"  # what actually ran
    facets_operator_degraded: bool = False  # whether it fell back
    # GravityWell reference leg fields (reference-only; never steer the recommendation)
    gw_verdict: str = "skip"  # verdict from GW, or "skip" if not dispatched
    gw_ran: bool = False  # whether the GW leg actually ran
    gw_skip_reason: str = ""  # reason for skip (e.g., "slot2_unavailable", "no_parseable_hostname")
    gw_findings_count: int = 0  # number of findings/issues from GW
    elapsed_gw: float = 0.0  # wall-clock time for GW leg
    gw_transcript_ref: str = ""  # absolute path to GW transcript JSON file

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


_AUTHORITY_TEXT_RE = re.compile(
    r"^\*\*Authority:\*\*\s+(\S+)",
    re.MULTILINE,
)
_AUTHORITY_TEXT_ALIASES = {"auto-merge": "auto", "auto": "auto", "advisory": "advisory", "hold": "hold"}


def _parse_spec_authority_text(text: str) -> str | None:
    """Extract Authority from spec text (first 50 lines).

    Returns 'advisory' | 'auto' | 'hold', or None if the header is absent or
    unrecognized. Unlike _parse_spec_verification_text, this has NO safe default —
    callers own the fallback, because bind and the spec-review gate need different
    fallback semantics (see cmd_bind below vs. the existing _parse_spec_authority
    path-based function, which is untouched by this spec).
    """
    first_50 = "\n".join(text.splitlines()[:50])
    m = _AUTHORITY_TEXT_RE.search(first_50)
    if not m:
        return None
    val = m.group(1).lower().rstrip(".,;:!)")
    return _AUTHORITY_TEXT_ALIASES.get(val)


_VERIFICATION_RE = re.compile(
    r"^\*\*Verification:\*\*\s+(\S+)",
    re.MULTILINE,
)
_VERIFICATION_MACHINE_ALIASES = frozenset({"machine"})


def _parse_spec_verification_text(text: str) -> str:
    """Extract Verification from spec text (first 50 lines).

    Returns 'machine' or 'pm-live-test'. Absent or unrecognized => 'pm-live-test' (fail-safe).
    Accepts aliases: live-test, human => pm-live-test.
    """
    first_50 = "\n".join(text.splitlines()[:50])
    m = _VERIFICATION_RE.search(first_50)
    if not m:
        return "pm-live-test"
    val = m.group(1).lower().rstrip(".,;:!)")
    if val in _VERIFICATION_MACHINE_ALIASES:
        return "machine"
    return "pm-live-test"


def _parse_spec_verification(spec_path: Path) -> str:
    """Extract Verification from spec file (first 50 lines).

    Sibling to _parse_spec_authority. Returns 'machine' or 'pm-live-test'.
    Absent/unrecognized/unreadable => 'pm-live-test' (fail-safe, never hard-errors).
    """
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError:
        return "pm-live-test"
    return _parse_spec_verification_text(text)


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


def _gw_slot2_url() -> str | None:
    """Resolve Slot-2's base URL: GW_SLOT2_URL env if set, else GW_URL's host with
    the port swapped to GW_SLOT2_PORT (default 8082). Mirrors the derivation
    pattern in agents_core.doorman_server._NodeState._slot2_url() — that's an
    instance method and can't be imported directly, so the ~4-line pattern is
    replicated here rather than hardcoding a second Tailscale IP.

    Returns None when GW_URL has no parseable hostname — callers must treat
    that as "Slot-2 unresolved" and skip gracefully rather than build a
    malformed URL.
    """
    override = os.environ.get("GW_SLOT2_URL")
    if override:
        return override

    gw_url = os.environ.get("GW_URL", "http://203.0.113.11:8081")
    slot2_port = os.environ.get("GW_SLOT2_PORT", "8082")
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(gw_url)
    if not parts.hostname:
        return None
    netloc = f"{parts.hostname}:{slot2_port}"
    return urlunsplit((parts.scheme, netloc, "", "", ""))


def _gw_primary_url() -> str:
    """The primary GW voicing endpoint — same value Council/Facets voicing
    resolves through call_operator("gravitywell")."""
    return os.environ.get("GW_URL", "http://203.0.113.11:8081")


def _normalized_host_port(url: str) -> tuple[str, int] | None:
    """Parse url to a (lowercased-host, port) tuple, applying default-port
    rules (80/443) when no port is present. Returns None if unparseable."""
    from urllib.parse import urlsplit
    parts = urlsplit(url.rstrip("/"))
    if not parts.hostname:
        return None
    port = parts.port
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return (parts.hostname.lower(), port)


def _gw_endpoints_collapsed(slot2_url: str, primary_url: str) -> bool:
    """Two-stage collapse test: does slot2_url point at the same physical
    endpoint as primary_url?

    Stage 1 (fast path): normalized host:port string equality (trailing
    slash stripped, host lowercased, default-port rules applied).

    Stage 2 (DNS-alias hardening): if the host:port strings differ, resolve
    both hostnames via socket.getaddrinfo and compare (ip, port) — two
    distinct hostnames that resolve to the same physical IP:port are also a
    collapse. Any resolution error falls back to the (already-negative)
    stage 1 result rather than raising.
    """
    a = _normalized_host_port(slot2_url)
    b = _normalized_host_port(primary_url)
    if a is None or b is None:
        return False
    if a == b:
        return True

    try:
        ips_a = {res[4][0] for res in socket.getaddrinfo(a[0], None)}
        ips_b = {res[4][0] for res in socket.getaddrinfo(b[0], None)}
    except OSError:
        return False

    return a[1] == b[1] and bool(ips_a & ips_b)


def _dispatch_gw_reviewer(
    spec_text: str,
    synth_target_id: str,
    parsed_target_id: str,
    repo: str,
    run_id: str,
    gw_principal: str | None = None,
) -> tuple[str | None, list[dict], float, str]:
    """Dispatch and run the GW reference reviewer synchronously against Slot-2
    Devstral (:8082), lease-free — a distinct-model second opinion that runs
    concurrently with the 27B Facets/Council voicing on :8081 at zero lease
    contention.

    Returns (text, transcript, elapsed_s, skip_reason) where text is the verdict
    string (or None if GW did not run), transcript is the list of tool calls,
    elapsed_s is the wall-clock time, and skip_reason is set when GW is skipped.

    Stub-aware: if GW_REVIEW_STUB=1, uses GW_REVIEW_STUB_VERDICT env var.
    """
    start_time = time.time()

    # Stub path
    if os.getenv("GW_REVIEW_STUB") == "1":
        verdict = os.getenv("GW_REVIEW_STUB_VERDICT", "clean")
        elapsed = time.time() - start_time
        return verdict, [], elapsed, ""

    gw_slot2_url = _gw_slot2_url()
    if gw_slot2_url is None:
        elapsed = time.time() - start_time
        print(
            f"[spec-review:gw-reviewer] GW reference reviewer skipped: GW_URL has no "
            f"parseable hostname (Slot-2 unresolved). Spec-review will proceed with "
            f"council voicing only.",
            file=sys.stderr,
        )
        return None, [], elapsed, "no_parseable_hostname"

    primary_url = _gw_primary_url()
    if _gw_endpoints_collapsed(gw_slot2_url, primary_url):
        elapsed = time.time() - start_time
        print(
            f"[spec-review:gw-reviewer] GW reference reviewer skipped: Slot-2 "
            f"({gw_slot2_url}) has collapsed onto the primary GW endpoint "
            f"({primary_url}) — a single-model topology gets Council/Facets "
            f"voicing only, no self-contending reference leg.",
            file=sys.stderr,
        )
        return None, [], elapsed, "slot2_collapsed_to_primary"

    # Single bounded (4s) probe: doubles as the readiness gate AND the
    # provenance source (the served model), replacing the doorman-heartbeat
    # pre-check (irrelevant to a lease-free Slot-2 consumer). None -> not
    # serving -> skip; a model string -> up -> run, and that string IS the
    # provenance. Logged, not hard-matched (Slot-2 may advertise either
    # gravitywell-devstral or the gravitywell-slot2 alias).
    served_model = swarm_model(gw_slot2_url)
    if served_model is None:
        elapsed = time.time() - start_time
        print(
            f"[spec-review:gw-reviewer] GW reference reviewer skipped: Slot-2 not "
            f"serving (probed {gw_slot2_url}). Spec-review will proceed with council "
            f"voicing only.",
            file=sys.stderr,
        )
        return None, [], elapsed, "slot2_unavailable"

    print(
        f"[spec-review:gw-reviewer-provenance] endpoint={gw_slot2_url} "
        f"served_model={served_model}",
        file=sys.stderr,
    )

    try:
        from agents_core.gw_agent import call_gw_agent, DEFAULT_READONLY_TOOLS
    except ImportError as e:
        elapsed = time.time() - start_time
        print(
            f"[spec-review:gw-reviewer] agents_core import failed: {e} — skipping GW leg",
            file=sys.stderr,
        )
        return None, [], elapsed, "import_error"

    prompt = (
        f"Review this spec for technical soundness, implementability, and "
        f"coverage of DON'T-do constraints and Definition-of-Done. "
        f"Return a JSON verdict with structure: "
        f"{{\"verdict\": \"clean|fixable|needs-human\", "
        f"\"issues\": [{{\"severity\": \"HIGH|MED|LOW\", \"note\": \"...\"}}], "
        f"\"confidence\": 0.0-1.0}}\n\n"
        f"=== SPEC ===\n{spec_text}"
    )

    try:
        text, transcript = call_gw_agent(
            prompt=prompt,
            system="",
            cwd=f"/srv/git/{repo}-working",
            tools=DEFAULT_READONLY_TOOLS,
            json_mode=True,
            on_wake_fail="skip",
            return_transcript=True,
            work_id=run_id,
            timeout=300,
            principal=gw_principal,
            backend_url=gw_slot2_url,
            acquire_lease=False,
        )
        elapsed = time.time() - start_time
        if text is not None:
            print(
                f"[spec-review:gw-reviewer] completed task_id={run_id} "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )
            return text, transcript, elapsed, ""
        print(
            f"[spec-review:gw-reviewer] GW did not run (on_wake_fail=skip) "
            f"elapsed={elapsed:.1f}s",
            file=sys.stderr,
        )
        return text, transcript, elapsed, "on_wake_fail_skip"
    except Exception as e:
        elapsed = time.time() - start_time
        print(
            f"[spec-review:gw-reviewer] call_gw_agent failed: {e} — skipping",
            file=sys.stderr,
        )
        return None, [], elapsed, "call_error"


def _poll_sonnet_until_terminal(
    spec_reviewer_task_id: str | None,
    timeout_s: int,
    start_time: float,
) -> dict | None:
    """Poll Sonnet leg to terminal with independent timeout.

    Council polling is now handled by the shared orchestrator (run_deliberation),
    which has its own internal timeout. The Sonnet leg polls independently with
    its own deadline to avoid stalling when one leg is slow.

    Returns sonnet_raw dict or None if task_id is None (no Sonnet leg dispatched).
    """
    if spec_reviewer_task_id is None:
        return None

    sonnet_result: dict | None = None

    while True:
        elapsed = time.time() - start_time
        timed_out = elapsed >= timeout_s

        # Check spec_reviewer terminal state
        if sonnet_result is None:
            output_path = _find_reviewer_output(spec_reviewer_task_id)
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
                return sonnet_result
            elif timed_out:
                sonnet_result = {
                    "status": "timeout",
                    "verdict": "timeout",
                    "issues": [],
                    "confidence": 0.0,
                    "run_id": spec_reviewer_task_id,
                }
                elapsed_s = int(time.time() - start_time)
                print(
                    f"[spec-review:sonnet-timeout] task_id={spec_reviewer_task_id} "
                    f"elapsed={elapsed_s}s",
                    file=sys.stderr,
                )
                return sonnet_result

        time.sleep(_POLL_CADENCE_S)


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
    council_voicing_requested: str = "gravitywell",
    gw_verdict: str = "skip",
    gw_ran: bool = False,
    gw_skip_reason: str = "",
    gw_findings_count: int = 0,
    elapsed_gw: float = 0.0,
    gw_transcript_ref: str = "",
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
    council_voicing_requested — the voicing value that was requested (default gravitywell).
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
    council_error_reason = council_raw.get("error_reason", "")

    # Extract effective voicing from Council run provenance
    council_voicing_effective = council_raw.get("voicing_effective")
    council_voicing_degraded = council_raw.get("voicing_degraded", False)
    council_voicing_degraded_reason = council_raw.get("voicing_degraded_reason", "")

    # Extract Facets escalation signal and compute reliability
    facets_escalation = None
    facets_unreliable = False
    facets_operator_requested = ""
    facets_operator_effective = "unknown"
    facets_operator_degraded = False

    if facets_deliberation:
        synthesis = facets_deliberation.get("synthesis") or {}
        if isinstance(synthesis, dict):
            facets_escalation = synthesis.get("escalation_recommendation")
        stances = facets_deliberation.get("stances") or []
        facets_unreliable = (
            bool(synthesis.get("parse_failed"))
            or any(s.get("parse_failed") for s in stances)
        )

        # Extract Facets operator information from methodology
        methodology = facets_deliberation.get("methodology") or {}
        facets_operator_requested = methodology.get("operator_requested") or ""
        # Determine effective operator from persona operators or synthesis operator
        persona_ops = methodology.get("persona_operators") or {}
        synthesis_op = methodology.get("synthesis_operator")
        if synthesis_op:
            facets_operator_effective = synthesis_op
        elif persona_ops and all(v == persona_ops.get("technical-integrity") for v in persona_ops.values()):
            # All personas use same operator
            facets_operator_effective = persona_ops.get("technical-integrity", "unknown")
        else:
            facets_operator_effective = "unknown"
        # Degrade detected iff operator_requested is set AND differs from effective
        facets_operator_degraded = bool(
            facets_operator_requested and facets_operator_requested != facets_operator_effective
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
        council_voicing_requested=council_voicing_requested,
        council_voicing_effective=council_voicing_effective or "unknown",
        council_voicing_degraded=council_voicing_degraded,
        council_voicing_degraded_reason=council_voicing_degraded_reason,
        facets_operator_requested=facets_operator_requested,
        facets_operator_effective=facets_operator_effective,
        facets_operator_degraded=facets_operator_degraded,
        council_error_reason=council_error_reason,
        gw_verdict=gw_verdict,
        gw_ran=gw_ran,
        gw_skip_reason=gw_skip_reason,
        gw_findings_count=gw_findings_count,
        elapsed_gw=elapsed_gw,
        gw_transcript_ref=gw_transcript_ref,
    )


def _council_infra_guidance(reason: str, run_id: str = "") -> str:
    """Map a council infra failure reason to operator guidance (substring, most-specific first)."""
    if "heartbeat_stale" in reason or "no_heartbeat_after_startup" in reason:
        return (
            "Council worker stalled/died (GW self-contention or a slow voice). "
            "Re-gate SOLO in a clean window (GW serving big, no concurrent gate), "
            "or raise COUNCIL_STALL_S for slow nodes. Not a spec objection."
        )
    if "gw_not_serving" in reason:
        return (
            "GW not serving — gate not run on GW. Wake/serve GW big, then re-gate "
            "(or pass an explicit sonnet voicing flag). Not a spec objection."
        )
    if "ConnectTimeout" in reason or "Council poll error" in reason:
        return (
            "Voicing node asleep/unreachable. Wake it (e.g. wake-starhouse) then re-gate. "
            "Not a spec objection."
        )
    if "slot_queued_timeout" in reason or "wake_failed" in reason:
        return (
            "GW admission is starving the gate's own concurrent calls. "
            "Set GW_ADMISSION_MODE off/shadow in conductor.env, then re-gate. "
            "Not a spec objection."
        )
    run_ref = f" `/srv/lapis/council/logs/{run_id}.log`" if run_id else ""
    return (
        "Council leg failed for an infra reason. "
        "Re-gate ONCE in a clean window; if the SAME failure recurs, STOP and inspect"
        f"{run_ref} before retrying. Not a spec objection."
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

    # Format effective voicing lines for Council and Facets
    def _voicing_line(requested: str, effective: str, degraded: bool, reason: str = "") -> str:
        """Format a single voicing/operator display line."""
        if effective == "unknown":
            return f"{requested} (effective: unknown)"
        if degraded:
            reason_str = f" ({reason})" if reason else ""
            return f"{requested} → {effective} (degraded{reason_str})"
        if effective.lower() == "gravitywell":
            return f"{requested} (on GW)"
        return f"{effective} (no degrade)"

    council_voicing_line = _voicing_line(
        brief.council_voicing_requested,
        brief.council_voicing_effective,
        brief.council_voicing_degraded,
        brief.council_voicing_degraded_reason,
    )

    # Only render Facets operator line if Facets deliberation was dispatched
    facets_operator_line = ""
    if brief.facets_deliberation is not None:
        facets_operator_line = _voicing_line(
            brief.facets_operator_requested or brief.facets_operator,
            brief.facets_operator_effective,
            brief.facets_operator_degraded,
            "",
        )

    # Degradation summary line — unmissable alert when ANY leg fell back
    degraded_summary = ""
    degraded_legs = []
    if brief.council_voicing_degraded:
        reason_str = f": {brief.council_voicing_degraded_reason}" if brief.council_voicing_degraded_reason else ""
        degraded_legs.append(f"Council{reason_str}")
    if brief.facets_operator_degraded:
        degraded_legs.append("Facets")
    if degraded_legs:
        legs_str = ", ".join(degraded_legs)
        degraded_summary = f"\n⚠️ DEGRADED: 1+ leg fell off GravityWell to paid Claude ({legs_str}).\n"

    # INFRA banner — rendered when council leg failed for a known infra reason
    infra_banner = ""
    if brief.council_error_reason:
        infra_guidance = _council_infra_guidance(brief.council_error_reason, brief.council_run_id)
        infra_banner = (
            f"\n## ⚠️ INFRA FAILURE — Not a spec objection\n"
            f"- **Reason:** `{brief.council_error_reason}`\n"
            f"- **Action:** {infra_guidance}\n"
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
    # Override "incomplete" with infra-specific guidance when a council error reason is known
    if brief.council_error_reason and brief.combined_recommendation == "incomplete":
        infra_guidance = _council_infra_guidance(brief.council_error_reason, brief.council_run_id)
        step_map["incomplete"] = (
            f"**[INFRA — not a spec objection]** Council leg did not complete due to an infra failure.\n\n"
            f"**Reason:** `{brief.council_error_reason}`\n\n"
            f"{infra_guidance}"
        )
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

    # GravityWell section — rendered when the leg ran (gw_ran=True).
    # Like Sonnet, this is always reference-only and never steers the recommendation.
    gw_section = ""
    if brief.gw_ran:
        gw_section = f"""
## GravityWell reference leg — reference only (does not affect recommendation)
- **Verdict:** {brief.gw_verdict}
- **Findings:** {brief.gw_findings_count}
- **Elapsed:** {brief.elapsed_gw:.1f}s
- **Transcript:** {brief.gw_transcript_ref or '(not persisted)'}
"""
    elif brief.gw_skip_reason:
        gw_section = f"""
## GravityWell reference leg — reference only (does not affect recommendation)
- **GW reference reviewer: skipped — {brief.gw_skip_reason}**
"""

    # Render voicing section only if there's data to show
    voicing_lines = f"- Council voicing: {council_voicing_line}"
    if facets_operator_line:
        voicing_lines += f"\n- Facets operator: {facets_operator_line}"
    voicing_section = f"\n**Leg voicing / operator:**\n{voicing_lines}\n" if voicing_lines else ""

    return f"""# Spec Review: {brief.target_id}

**Spec:** {brief.spec_path}
**Repo:** {brief.repo}
**Elapsed:** {brief.elapsed_s:.1f}s
**Recommendation:** {brief.combined_recommendation}
{voicing_section}{degraded_summary}{infra_banner}{facets_section}{sonnet_section}{gw_section}
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
# Cross-session spec-review serialization lock
# ---------------------------------------------------------------------------

def _spec_review_lock_path() -> Path:
    """Canonical cross-session lock path derived from os.getuid().

    Aborts with RuntimeError if /run/user/<uid> is absent — that is a fatal
    config error, not a fallback opportunity. A divergent path would silently
    defeat the mutual exclusion guarantee.
    """
    uid = os.getuid()
    run_user_dir = Path(f"/run/user/{uid}")
    if not run_user_dir.is_dir():
        raise RuntimeError(
            f"[spec-review:lock-fatal] /run/user/{uid} does not exist — "
            f"fatal config error; cannot derive canonical lock path. "
            f"Ensure systemd user runtime dir is present (loginctl enable-linger)."
        )
    return run_user_dir / "lapis-pm-spec-review.lock"


def _read_lock_holder(fd: int) -> str:
    """Read holder identity written by the lock acquirer. Advisory — may race with writer."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, 4096)
        if not data:
            return "(no holder identity written yet)"
        info = json.loads(data.decode())
        return (
            f"pid={info.get('pid', '?')} "
            f"spec={info.get('spec_path', '?')} "
            f"started_at={info.get('started_at', '?')} "
            f"host={info.get('host', '?')}"
        )
    except Exception:
        return "(unable to read holder identity)"


@contextlib.contextmanager
def _spec_review_lock(
    spec_path: Path,
    *,
    _lock_path_override: Path | None = None,
) -> Iterator[None]:
    """Cross-session exclusive advisory flock serializing spec-review GW dispatch.

    Lock path: /run/user/<uid>/lapis-pm-spec-review.lock (canonical, no fallback).
    Serializes within BRIX — sufficient since spec-review runs on BRIX.
    Auto-queues waiters up to SPEC_REVIEW_LOCK_TIMEOUT seconds (default 900).
    Fail-CLOSED on timeout: a still-held lock means a live hung holder; abort + NORMAL
    Pushover alert. Never proceed degraded, never kill the holder (PID-reuse risk).

    Superseded when the H5 elevator organ activates and spec-review submits through it.
    Removable at that point — cite lapis-pm-spec-review-serial-lock-v0.
    """
    lock_path = (
        _lock_path_override
        if _lock_path_override is not None
        else _spec_review_lock_path()
    )
    timeout_s = int(
        os.environ.get("SPEC_REVIEW_LOCK_TIMEOUT", str(_SPEC_REVIEW_LOCK_TIMEOUT_DEFAULT))
    )

    print(f"[spec-review:lock] lock_path={lock_path}", file=sys.stderr)

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + timeout_s
        logged_waiting = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass

            if not logged_waiting:
                holder_desc = _read_lock_holder(fd)
                print(
                    f"[spec-review:lock-queued] queued behind active review — {holder_desc}",
                    file=sys.stderr,
                )
                logged_waiting = True

            if time.monotonic() >= deadline:
                # Lock still held past timeout: holder is alive and hung (a crashed holder
                # auto-releases via kernel flock, so still-blocked => live hang). Fail CLOSED.
                holder_desc = _read_lock_holder(fd)
                msg = (
                    f"[spec-review:lock-timeout] lock held for >{timeout_s}s by a live "
                    f"process (hung holder: {holder_desc}). Aborting review of {spec_path}. "
                    f"Do NOT delete {lock_path} — kernel owns lock state. "
                    f"Investigate and clear the hung process, then re-run."
                )
                print(msg, file=sys.stderr)
                try:
                    from agents_core.notify import send_notification, Priority as _P
                    send_notification(
                        message=msg,
                        title="spec-review: lock timeout — hung holder",
                        priority=_P.NORMAL,
                    )
                except Exception as _notify_err:
                    print(
                        f"[spec-review:lock-timeout-notify-error] {_notify_err}",
                        file=sys.stderr,
                    )
                raise RuntimeError(msg)

            time.sleep(2)

        # Acquired. Write holder identity so any waiting caller can name us in its log.
        holder_data = json.dumps({
            "pid": os.getpid(),
            "spec_path": str(spec_path),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
        }).encode()
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, holder_data)

        print(f"[spec-review:lock-acquired] acquired spec_path={spec_path}", file=sys.stderr)
        yield
    finally:
        # Closing the fd releases the flock at the kernel level.
        # Process death also releases it, so stale locks never wedge.
        try:
            os.close(fd)
        except OSError:
            pass


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

    # Cross-session exclusive lock: serializes concurrent GW-dispatch across sessions
    # to prevent GW lane contention. Auto-queues up to SPEC_REVIEW_LOCK_TIMEOUT (900s);
    # fail-CLOSED on timeout (hung holder). Superseded when the H5 elevator organ activates.
    # Removable at that point — cite lapis-pm-spec-review-serial-lock-v0.
    with _spec_review_lock(spec_path):
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

        # 4c. GW reference leg: submit to ThreadPoolExecutor BEFORE Facets block so it runs
        #     in parallel. Always-on for advisory/hold; reference-only (never moves the
        #     recommendation). Bounded timeout + doorman pre-flight so it NEVER stalls the gate.
        gw_future = None
        executor = None
        gw_run_id = str(uuid.uuid4())[:8]
        gw_principal = f"gw-gate-{uuid.uuid4().hex[:12]}"
        do_gw = effective_authority in {"advisory", "hold"}
        if do_gw:
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                gw_future = executor.submit(
                    _dispatch_gw_reviewer,
                    spec_text=spec_text,
                    synth_target_id=gw_run_id,
                    parsed_target_id=parsed_target_id,
                    repo=repo,
                    run_id=gw_run_id,
                    gw_principal=gw_principal,
                )
                print(
                    f"[spec-review:gw-reviewer] submitted to executor work_id={gw_run_id}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"[spec-review:gw-reviewer-submit-error] {e}",
                    file=sys.stderr,
                )
                gw_future = None

        # 5. Run shared orchestration (Facets + Council concurrently).
        #    This replaces the bespoke _dispatch_facets + _dispatch_council paths.
        #    Council runs with its own internal timeout; Sonnet gets its own poll deadline.
        facets_deliberation: dict | None = None
        envelope = None
        council_run_id: str | None = None
        council_not_run_reason = ""

        if dispatch_facets and effective_authority in {"advisory", "hold"}:
            # Set the council timeout from the caller's timeout_s
            # (Facets finding #3 / Trickster: dynamic timeout propagation)
            os.environ["SHARED_DELIBERATION_COUNCIL_TIMEOUT_S"] = str(timeout_s)

            # Pre-grounding step: elevator integration (ELEVATOR_ACTIVE-gated).
            # Runs BEFORE DeliberationRequest construction; outcome sets grounding_result_file.
            # Held path (ELEVATOR_ACTIVE != "true"): grounding_result_file=None, no enqueue.
            import tempfile as _tempfile
            grounding_result_file: str | None = None
            _grounding_tmp: str | None = None  # track for cleanup
            grounding_path: str = "inline:not-attempted"

            elevator_active = os.getenv("ELEVATOR_ACTIVE") == "true"
            if elevator_active:
                elevator_store_url = os.getenv("ELEVATOR_STORE_URL", "http://127.0.0.1:8405")
                elevator_poll_timeout_sec = int(
                    os.getenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "180")
                )
                elevator_claim_deadline_sec = int(
                    os.getenv("ELEVATOR_GROUNDING_CLAIM_DEADLINE_SEC", "75")
                )

                # Step 0: Readiness pre-check — canonical probe (~4s, SWARM_URL/GW_URL-aware).
                if not swarm_serving():
                    grounding_path = "inline:swarm-not-serving"
                    print(
                        "[spec-review:elevator-grounding] swarm not serving (pre-check ~4s); "
                        "falling back to inline grounding",
                        file=sys.stderr,
                    )
                else:
                    try:
                        import requests as _requests

                        # Step 1: Enqueue grounding item.
                        target_repo_path = f"/srv/git/{repo}-working"
                        enqueue_payload = {
                            "lane": "execution",
                            "kind": "grounding",
                            "payload": {
                                "spec_text": spec_text,
                                "target_repo": target_repo_path,
                                "backend": "swarm",
                            },
                            "principal": "lapis-pm-spec-review",
                            "latency_class": "batch",
                        }
                        enqueue_resp = _requests.post(
                            f"{elevator_store_url}/v0/elevator/enqueue",
                            json=enqueue_payload,
                            timeout=10,
                        )
                        if enqueue_resp.status_code not in (200, 201):
                            grounding_path = f"inline:enqueue-failed({enqueue_resp.status_code})"
                            print(
                                f"[spec-review:elevator-grounding] enqueue failed "
                                f"({enqueue_resp.status_code}); falling back to inline",
                                file=sys.stderr,
                            )
                        else:
                            item_id = enqueue_resp.json().get("item_id")
                            if not item_id:
                                grounding_path = "inline:no-item-id"
                                print(
                                    "[spec-review:elevator-grounding] no item_id in enqueue response; "
                                    "falling back to inline",
                                    file=sys.stderr,
                                )
                            else:
                                # Step 2: Poll with two deadlines.
                                # Claim-progress deadline (AC5b): if item stays pending >
                                # elevator_claim_deadline_sec, fall back immediately — guards
                                # against an up-but-dead handler that would hang the full poll.
                                poll_start = time.time()
                                first_claim_seen = False
                                while True:
                                    elapsed = time.time() - poll_start
                                    if elapsed >= elevator_poll_timeout_sec:
                                        grounding_path = f"inline:poll-timeout({int(elapsed)}s)"
                                        print(
                                            f"[spec-review:elevator-grounding] poll timeout "
                                            f"({int(elapsed)}s >= {elevator_poll_timeout_sec}s); "
                                            "falling back to inline",
                                            file=sys.stderr,
                                        )
                                        break
                                    try:
                                        poll_resp = _requests.get(
                                            f"{elevator_store_url}/v0/elevator/item/{item_id}",
                                            timeout=5,
                                        )
                                        if poll_resp.status_code == 200:
                                            item_json = poll_resp.json()
                                            item_status = item_json.get("status")
                                            if item_status not in ("pending",):
                                                first_claim_seen = True
                                            if item_status == "served":
                                                # Step 3: write-and-verify result file.
                                                item_result = item_json.get("result")
                                                if not item_result:
                                                    grounding_path = "inline:grounding-empty"
                                                    print(
                                                        f"[spec-review:elevator-grounding] "
                                                        f"item_id={item_id} served but result empty; "
                                                        "falling back to inline",
                                                        file=sys.stderr,
                                                    )
                                                else:
                                                    try:
                                                        import json as _json
                                                        with _tempfile.NamedTemporaryFile(
                                                            mode="w",
                                                            suffix=".json",
                                                            delete=False,
                                                        ) as grf:
                                                            if isinstance(item_result, str):
                                                                grf.write(item_result)
                                                            else:
                                                                _json.dump(item_result, grf)
                                                            _grounding_tmp = grf.name
                                                        # Verify file exists and is non-empty.
                                                        tmp_path_obj = Path(_grounding_tmp)
                                                        if not tmp_path_obj.exists() or tmp_path_obj.stat().st_size == 0:
                                                            grounding_path = "inline:grounding-write-failed"
                                                            print(
                                                                f"[spec-review:elevator-grounding] "
                                                                f"temp file verify failed for item_id={item_id}; "
                                                                "falling back to inline",
                                                                file=sys.stderr,
                                                            )
                                                            _grounding_tmp = None
                                                        else:
                                                            grounding_result_file = _grounding_tmp
                                                            grounding_path = f"swarm:{item_id}"
                                                            print(
                                                                f"[spec-review:elevator-grounding] "
                                                                f"served item_id={item_id} "
                                                                f"grounding_path={grounding_path}",
                                                                file=sys.stderr,
                                                            )
                                                    except Exception as write_err:
                                                        grounding_path = "inline:grounding-write-failed"
                                                        print(
                                                            f"[spec-review:elevator-grounding] "
                                                            f"write/verify failed ({write_err}); "
                                                            "falling back to inline",
                                                            file=sys.stderr,
                                                        )
                                                        _grounding_tmp = None
                                                break
                                            elif item_status in ("failed", "expired"):
                                                grounding_path = f"inline:item-{item_status}"
                                                print(
                                                    f"[spec-review:elevator-grounding] "
                                                    f"item terminal status={item_status}; "
                                                    "falling back to inline",
                                                    file=sys.stderr,
                                                )
                                                break
                                            # Check claim-progress deadline (AC5b).
                                            elif not first_claim_seen and elapsed >= elevator_claim_deadline_sec:
                                                grounding_path = "inline:swarm-stall-unclaimed"
                                                print(
                                                    f"[spec-review:elevator-grounding] "
                                                    f"item_id={item_id} still pending after "
                                                    f"{int(elapsed)}s (claim deadline "
                                                    f"{elevator_claim_deadline_sec}s); "
                                                    "falling back to inline grounding_path=inline:swarm-stall-unclaimed",
                                                    file=sys.stderr,
                                                )
                                                break
                                        else:
                                            grounding_path = f"inline:poll-http-{poll_resp.status_code}"
                                            print(
                                                f"[spec-review:elevator-grounding] "
                                                f"poll HTTP {poll_resp.status_code}; "
                                                "falling back to inline",
                                                file=sys.stderr,
                                            )
                                            break
                                    except Exception as poll_err:
                                        grounding_path = f"inline:poll-error({type(poll_err).__name__})"
                                        print(
                                            f"[spec-review:elevator-grounding] poll error {poll_err}; "
                                            "falling back to inline",
                                            file=sys.stderr,
                                        )
                                        break
                                    time.sleep(_ELEVATOR_GROUNDING_POLL_CADENCE_S)
                    except Exception as enq_err:
                        grounding_path = f"inline:enqueue-error({type(enq_err).__name__})"
                        print(
                            f"[spec-review:elevator-grounding] {enq_err}; falling back to inline",
                            file=sys.stderr,
                        )

                print(
                    f"[spec-review:elevator-grounding] grounding_path={grounding_path}",
                    file=sys.stderr,
                )

            request = DeliberationRequest(
                text=spec_text,
                context={
                    "spec_path": room_str('planning.specs', f'{parsed_target_id}.md'),
                    "target_id": parsed_target_id,
                    "repo": repo,
                    "authority": effective_authority,
                    "invariant_context": invariant_context,
                },
                triage="full",
                caller="spec-review",
                council_voicing=council_voicing,
                facets_operator=facets_operator,
                grounding_result_file=grounding_result_file,
                gw_principal=gw_principal,
            )

            # D1: GW-liveness preflight on the primary Council+Facets legs.
            # If GW is not serving when council_voicing==gravitywell, skip run_deliberation
            # entirely — do NOT launch a doomed deliberation that may fall back to paid Sonnet.
            if council_voicing == "gravitywell" and not swarm_serving():
                print(
                    "[spec-review:council-preflight] swarm not serving; skipping run_deliberation",
                    file=sys.stderr,
                )
                council_not_run_reason = "gw_not_serving"
                if _grounding_tmp:
                    try:
                        os.unlink(_grounding_tmp)
                    except OSError:
                        pass
            else:
                try:
                    async def _deliberate():
                        init_facets_semaphore(int(os.environ.get("SHARED_DELIBERATION_MAX_CONCURRENT", "2")))
                        return await run_deliberation(request)

                    envelope = asyncio.run(_deliberate())
                    print(
                        f"[spec-review:shared-deliberation-complete] "
                        f"facets_ok={envelope.facets_ok} council_ok={envelope.council_ok}",
                        file=sys.stderr,
                    )

                    # Extract facets deliberation if successful
                    if envelope.facets_ok and envelope.facets:
                        facets_deliberation = envelope.facets

                    # Extract council run_id for reference
                    council_run_id = envelope.council_run_id
                except Exception as e:
                    print(
                        f"[spec-review:shared-deliberation-error] {e}",
                        file=sys.stderr,
                    )
                    # Degrade gracefully: envelope will be None, council_raw will reflect the error below
                    envelope = None
                finally:
                    # Clean up grounding temp file if one was created.
                    if _grounding_tmp:
                        try:
                            os.unlink(_grounding_tmp)
                        except OSError:
                            pass

        # 7. Collect GW Future (with bounded timeout)
        gw_text: str | None = None
        gw_transcript: list[dict] = []
        gw_elapsed: float = 0.0
        gw_skip_reason: str = ""
        if gw_future is not None:
            try:
                gw_text, gw_transcript, gw_elapsed, gw_skip_reason = gw_future.result(timeout=300)
            except FuturesTimeoutError:
                print(
                    f"[spec-review:gw-reviewer-timeout] GW leg exceeded 300s timeout",
                    file=sys.stderr,
                )
                gw_text = None
                gw_elapsed = 300.0
                gw_skip_reason = "timeout"
            except Exception as e:
                print(
                    f"[spec-review:gw-reviewer-collect-error] {e}",
                    file=sys.stderr,
                )
                gw_text = None
                gw_skip_reason = "collection_error"
            finally:
                # Clean up executor to prevent thread pool leak
                if executor is not None:
                    executor.shutdown(wait=False)

        # 8. Poll Sonnet leg to terminal (with independent timeout from Council).
        #    Council results come directly from the envelope (already complete).
        sonnet_raw = _poll_sonnet_until_terminal(
            spec_reviewer_task_id=spec_reviewer_task_id,
            timeout_s=timeout_s,
            start_time=start_time,
        )

        # 8b. Build council_raw from the envelope. If envelope is None (error path),
        #     council_raw reflects the error.
        if envelope is not None and envelope.council_ok:
            council_raw = {
                "status": envelope.council_status or "error",
                "landing": envelope.council_landing or "",
                "open_questions": envelope.council_open_questions or [],
                "confidence": envelope.council_confidence or "",
                "positions": envelope.council_positions or [],
                "run_id": envelope.council_run_id or "",
                "voicing_effective": envelope.council_voicing_effective,
                "voicing_degraded": envelope.council_voicing_degraded,
                "voicing_degraded_reason": envelope.council_voicing_degraded_reason or "",
            }
        else:
            # Council leg failed or envelope is None: return error status
            status = "error"
            if envelope is not None:
                status = envelope.council_status or "error"
                if status not in _COUNCIL_TERMINAL:
                    status = "error"
            # D2: capture the worker's specific failure reason for legibility.
            # Priority: preflight-skip reason > envelope error string > empty.
            if council_not_run_reason:
                _err_reason = council_not_run_reason
            elif envelope is not None:
                _err_reason = envelope.errors.get("council", "")
            else:
                _err_reason = ""
            council_raw = {
                "status": status,
                "landing": "",
                "open_questions": [],
                "confidence": "",
                "positions": [],
                "run_id": envelope.council_run_id if envelope else "",
                "voicing_effective": None,
                "voicing_degraded": False,
                "voicing_degraded_reason": "",
                "error_reason": _err_reason,
            }

    elapsed = time.time() - start_time

    # 9. Persist GW transcript and divergence record
    gw_verdict: str = "skip"
    gw_ran: bool = gw_text is not None
    gw_findings_count: int = 0
    gw_transcript_ref: str = ""

    if gw_ran and gw_text:
        try:
            gw_verdict_obj = json.loads(gw_text)
            gw_verdict = gw_verdict_obj.get("verdict", "error")
            gw_findings_count = len(gw_verdict_obj.get("issues", []))
        except json.JSONDecodeError:
            gw_verdict = "error"
            gw_findings_count = 0

        # Write transcript JSON to /srv/lapis/spec-review-artifacts/<run_id>/gw-transcript.json
        artifacts_dir = room_path('spec_review_artifacts', gw_run_id)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = artifacts_dir / "gw-transcript.json"
        try:
            transcript_path.write_text(json.dumps(gw_transcript, indent=2), encoding="utf-8")
            gw_transcript_ref = str(transcript_path.resolve())
        except Exception as e:
            print(
                f"[spec-review:gw-transcript-write-error] {e}",
                file=sys.stderr,
            )

    # Write divergence record to mem via subprocess — always, even when gw_ran=False,
    # so that skip-rate visibility is preserved (gw_ran: bool field tracks success/failure)
    sonnet_verdict_str = sonnet_raw.get("verdict", "skip") if sonnet_raw else "skip"
    divergence_record = {
        "run_id": gw_run_id,
        "spec_path": str(spec_path),
        "repo": repo,
        "sonnet_verdict": sonnet_verdict_str,
        "gw_verdict": gw_verdict,
        "agree": (gw_verdict == sonnet_verdict_str) if gw_ran else None,
        "gw_ran": gw_ran,
        "gw_transcript_ref": gw_transcript_ref,
        "sonnet_findings_count": len(sonnet_raw.get("issues", [])) if sonnet_raw else 0,
        "gw_findings_count": gw_findings_count,
        "elapsed_gw": gw_elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        from lapis_pm import node_identity as _node_identity
        if _node_identity.resolve_node_identity().node_role != "master":
            print(
                "[spec-review:gw-divergence-record-skip] node_role != master; "
                "not writing to the shared mem ledger",
                file=sys.stderr,
            )
        else:
            import subprocess as _subprocess
            _subprocess.run(
                ["mem", "set", f"router/gw-review-divergence/{gw_run_id}",
                 json.dumps(divergence_record)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            print(
                f"[spec-review:gw-divergence-record] logged to mem",
                file=sys.stderr,
            )
    except Exception as e:
        print(
            f"[spec-review:gw-divergence-record-error] {e}",
            file=sys.stderr,
        )

    # 10. Build and return brief. sonnet_raw is non-None when the Sonnet leg ran; it is
    #     rendered for reference but excluded from the recommendation (sonnet_advisory_only).
    #     GW leg data is also reference-only and non-steering.
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
        council_voicing_requested=council_voicing,
        gw_verdict=gw_verdict,
        gw_ran=gw_ran,
        gw_skip_reason=gw_skip_reason,
        gw_findings_count=gw_findings_count,
        elapsed_gw=gw_elapsed,
        gw_transcript_ref=gw_transcript_ref,
    )
