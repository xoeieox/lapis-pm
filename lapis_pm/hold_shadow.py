"""hold_shadow.py — dual-hook shadow observer (lapis-pm-hold-shadow-observer-v0).

Act-free shadow records of every gate outcome and PR hold, for Erah's
Thursday-morning enforce-boundary reading. SHADOW MODE ONLY (ratified
2026-08-12, HANDOFF-2026-08-12-wake-actuator-and-four-units.md, unit 3):
this module and its siblings (hold_shadow_rules.py, hold_shadow_summary.py)
never act on a hold. They only ever append a record to their own files
under /srv/lapis/hold-shadow/, and — once a day, out of tick — deposit one
summary gem (hold_shadow_summary.py).

Isolation invariants (spec Scope 6, test-enforced by
tests/test_hold_shadow_boundary.py):
  - imports nothing from lapis_pm except hold_shadow_rules (also isolated,
    stdlib-only)
  - never imports/calls tick, tick_all, force_dispatch, merge_and_deploy,
    any _act_*, apply_decision, bind, reconcile_decided_gems, steer.*
    mutators, or episodic.write_*
  - never invokes run_spec_review / the spec-review CLI, never opens the
    gate lock
  - writes ONLY under /srv/lapis/hold-shadow/ (the single gem POST lives in the
    separate hold_shadow_summary.py, not here)

Both public hooks (observe_gate_outcome, observe_hold_fact) are wrapped
never-raise: a write failure is caught, logged to
/srv/lapis/hold-shadow/faults.jsonl (itself best-effort, never-raise), and
swallowed. Never-raise is not silent suppression (gate amendment, Facets
2026-08-12) — the fault log is what keeps an observation failure visible
Thursday instead of a blind spot. Observation must never affect the gate
result or the tick (cursor-loss hazard if a shadow pass raised inside
tick()).

Callers (each a one-line hook call — see spec_review.py's run_spec_review
return site and pm_core.py's _act_brief hold branch):
    hold_shadow.observe_gate_outcome(brief, authority=effective_authority)
    hold_shadow.observe_hold_fact(target_id=..., pr_number=..., ...)
"""
from __future__ import annotations

import inspect
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import hold_shadow_rules as rules

GATE_OUTCOME_SCHEMA = "gate-outcome/v1"
HOLD_FACT_SCHEMA = "hold-fact/v1"
FAULT_SCHEMA = "hold-shadow-fault/v1"

_GATE_OUTCOME_REQUIRED_KEYS = frozenset({
    "schema_version", "run_id", "ts_utc", "spec_path", "repo", "authority",
    "combined_recommendation", "council", "facets", "grounding", "invoked_by",
})
_HOLD_FACT_REQUIRED_KEYS = frozenset({
    "schema_version", "record_id", "observed_at_utc", "target_id",
    "brief_comment_id", "hold_comment_id", "pr_number", "repo", "pm_authority",
    "spec_bound_ts", "hold_class", "hold_reasons_verbatim", "would_have_action",
    "would_have_params", "confidence", "dedupe_key",
})


def validate_gate_outcome_record(record: dict) -> None:
    """Raise ValueError on any schema violation. Checks schema_version
    first — the versioned-JSONL reader idiom (see gw-topology-reach-event/v1
    in conductor's gw_topology.py): never assume fields without checking
    the version tag first."""
    if record.get("schema_version") != GATE_OUTCOME_SCHEMA:
        raise ValueError(f"unexpected schema_version: {record.get('schema_version')!r}")
    missing = _GATE_OUTCOME_REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"gate-outcome record missing keys: {sorted(missing)}")
    for k in ("status", "confidence", "open_questions_n", "blocks_n"):
        if k not in record["council"]:
            raise ValueError(f"gate-outcome record.council missing key: {k}")
    for k in ("consensus_level", "escalation", "escalation_reason"):
        if k not in record["facets"]:
            raise ValueError(f"gate-outcome record.facets missing key: {k}")
    for k in ("status", "sha"):
        if k not in record["grounding"]:
            raise ValueError(f"gate-outcome record.grounding missing key: {k}")


def validate_hold_fact_record(record: dict) -> None:
    """Raise ValueError on any schema violation. Checks schema_version
    first, same idiom as validate_gate_outcome_record."""
    if record.get("schema_version") != HOLD_FACT_SCHEMA:
        raise ValueError(f"unexpected schema_version: {record.get('schema_version')!r}")
    missing = _HOLD_FACT_REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"hold-fact record missing keys: {sorted(missing)}")
    if record["hold_class"] not in rules.HOLD_CLASSES:
        raise ValueError(f"invalid hold_class: {record['hold_class']!r}")
    if record["would_have_action"] not in rules.WOULD_HAVE_ACTIONS:
        raise ValueError(f"invalid would_have_action: {record['would_have_action']!r}")
    if record["pr_number"] is not None and not isinstance(record["pr_number"], int):
        raise ValueError("pr_number must be int or None")
    if record["dedupe_key"] != f"{record['target_id']}:{record['brief_comment_id']}":
        raise ValueError("dedupe_key does not match '<target_id>:<brief_comment_id>'")
    # Negative-leakage guardrail (spec DoD-3): a hold-fact record must never
    # carry the executable directive shape (brief.py's
    # /srv/lapis/directives/brief-decisions/<tid>__*.json — {brief_id, option_id}).
    if "brief_id" in record and "option_id" in record:
        raise ValueError("hold-fact record carries the executable {brief_id, option_id} shape")


def _room_root() -> Path:
    return Path(os.environ.get("ROOM_ROOT", "/room"))


def hold_shadow_dir() -> Path:
    return _room_root() / "hold-shadow"


def gate_outcomes_path() -> Path:
    return hold_shadow_dir() / "gate-outcomes.jsonl"


def hold_facts_path() -> Path:
    return hold_shadow_dir() / "hold-facts.jsonl"


def faults_path() -> Path:
    return hold_shadow_dir() / "faults.jsonl"


def _dedupe_index_path() -> Path:
    return hold_shadow_dir() / "dedupe-index.json"


def _now_utc_iso() -> str:
    """UTC, seconds precision — matches gem timestamps, never a raw comment
    ts (comments are Pacific ISO microseconds; see the spec's 'Notes for
    the implementer')."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _log_fault(hook: str, detail: str) -> None:
    """Best-effort, itself never-raise: a failure logging a failure must
    never propagate to the caller."""
    try:
        _append_jsonl(faults_path(), {
            "schema_version": FAULT_SCHEMA,
            "ts_utc": _now_utc_iso(),
            "hook": hook,
            "detail": detail[:2000],
        })
    except Exception:
        pass


def _detect_invoked_by() -> str:
    """Best-effort, deterministic caller identification via the call stack.
    No parameter threading needed at run_spec_review's call sites — this is
    what makes 'every invoker is captured' true (the hook lives inside
    run_spec_review itself) with zero signature changes there. Unknown
    callers fall back to 'unknown'; this never raises."""
    try:
        for frame_info in inspect.stack()[1:8]:
            name = Path(frame_info.filename).stem
            if name == "bundle_autodispatch":
                return "bundle_autodispatch"
            if name == "cli":
                return "cli"
            if "mcp" in name:
                return "mcp"
        return "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Hook 1: gate-outcome/v1 — spec_review.run_spec_review completion
# ---------------------------------------------------------------------------

def observe_gate_outcome(brief, authority: str) -> None:
    """Never-raise hook. Call once, at run_spec_review's return point."""
    try:
        _write_gate_outcome(brief, authority)
    except Exception as exc:
        _log_fault("gate-outcome", f"{type(exc).__name__}: {exc}")


def _write_gate_outcome(brief, authority: str) -> None:
    fd = getattr(brief, "facets_deliberation", None) or {}
    synthesis = fd.get("synthesis") if isinstance(fd, dict) else None
    if not isinstance(synthesis, dict):
        synthesis = {}
    council_positions = getattr(brief, "council_positions", None) or []
    blocks_n = sum(1 for p in council_positions if p.get("position") == "block")

    record = {
        "schema_version": GATE_OUTCOME_SCHEMA,
        "run_id": str(uuid.uuid4()),
        "ts_utc": _now_utc_iso(),
        "spec_path": str(getattr(brief, "spec_path", "")),
        "repo": getattr(brief, "repo", ""),
        "authority": authority,
        "combined_recommendation": getattr(brief, "combined_recommendation", ""),
        "council": {
            "status": getattr(brief, "council_status", ""),
            "confidence": getattr(brief, "council_confidence", ""),
            "open_questions_n": len(getattr(brief, "council_open_questions", None) or []),
            "blocks_n": blocks_n,
        },
        "facets": {
            "consensus_level": synthesis.get("consensus_level"),
            "escalation": synthesis.get("escalation_recommendation"),
            "escalation_reason": synthesis.get("escalation_reason"),
        },
        "grounding": {
            "status": getattr(brief, "grounding_status", ""),
            "sha": getattr(brief, "grounding_resolved_sha", ""),
        },
        "invoked_by": _detect_invoked_by(),
    }
    _append_jsonl(gate_outcomes_path(), record)


# ---------------------------------------------------------------------------
# Hook 2: hold-fact/v1 — pm_core._act_brief's hold branch
# ---------------------------------------------------------------------------

def observe_hold_fact(
    *,
    target_id: str,
    pr_number: int | None,
    repo: str,
    hold_reasons: list[str],
    hold_comment_id: str,
    brief_comment_id: str,
    pm_authority: str,
    spec_bound_ts: str,
    has_issues: bool = False,
) -> None:
    """Never-raise hook. Call once, at the end of _act_brief's hold branch,
    once both hold_comment_id and brief_comment_id are known.

    pr_number and repo come from the caller's PRClassification (cls.pr_number
    / cls.repo) — the same structured int/str that become the pm:pr= /
    pm:repo= comment tags, never a prose regex (contrast brief_gem.py's
    _extract_pr_number, which does scan prose).
    """
    try:
        _write_hold_fact(
            target_id=target_id,
            pr_number=pr_number,
            repo=repo,
            hold_reasons=hold_reasons,
            hold_comment_id=hold_comment_id,
            brief_comment_id=brief_comment_id,
            pm_authority=pm_authority,
            spec_bound_ts=spec_bound_ts,
            has_issues=has_issues,
        )
    except Exception as exc:
        _log_fault("hold-fact", f"{type(exc).__name__}: {exc}")


def _write_hold_fact(
    *,
    target_id: str,
    pr_number: int | None,
    repo: str,
    hold_reasons: list[str],
    hold_comment_id: str,
    brief_comment_id: str,
    pm_authority: str,
    spec_bound_ts: str,
    has_issues: bool,
) -> None:
    dedupe_key = f"{target_id}:{brief_comment_id}"
    if _already_recorded(dedupe_key):
        return

    hold_class, would_have_action, would_have_params, confidence = rules.classify_hold(
        hold_reasons, authority_level=pm_authority, has_issues=has_issues,
    )
    if would_have_action in {"merge_pr", "adopt_pr"} and pr_number is not None:
        would_have_params = {**would_have_params, "pr": pr_number}

    record = {
        "schema_version": HOLD_FACT_SCHEMA,
        "record_id": str(uuid.uuid4()),
        "observed_at_utc": _now_utc_iso(),
        "target_id": target_id,
        "brief_comment_id": brief_comment_id,
        "hold_comment_id": hold_comment_id,
        "pr_number": int(pr_number) if pr_number is not None else None,
        "repo": repo,
        "pm_authority": pm_authority,
        "spec_bound_ts": spec_bound_ts,
        "hold_class": hold_class,
        "hold_reasons_verbatim": list(hold_reasons),
        "would_have_action": would_have_action,
        "would_have_params": would_have_params,
        "confidence": confidence,
        "dedupe_key": dedupe_key,
    }
    _append_jsonl(hold_facts_path(), record)
    _mark_recorded(dedupe_key)


def _already_recorded(dedupe_key: str) -> bool:
    """Favor a duplicate record over losing one: any read failure is
    treated as not-yet-recorded."""
    try:
        idx_path = _dedupe_index_path()
        if not idx_path.exists():
            return False
        data = json.loads(idx_path.read_text(encoding="utf-8"))
        return dedupe_key in data
    except Exception:
        return False


def _mark_recorded(dedupe_key: str) -> None:
    idx_path = _dedupe_index_path()
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if idx_path.exists():
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[dedupe_key] = True
    idx_path.write_text(json.dumps(data), encoding="utf-8")
