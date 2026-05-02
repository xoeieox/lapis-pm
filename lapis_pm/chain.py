"""Target chaining — validation, event emission, state snapshots, chain advance.

Chain state lives in mem.db under:
  chain/<group>/event/<iso-timestamp>   — append-only event log (one key per event)
  chain/<group>/state                   — JSON snapshot rewritten on each transition

Chain advance runs after every target land: scan pm-bound targets whose
depends_on includes the just-landed tid; if ALL deps are satisfied, auto-fire
that target's initial_dispatch.

Invariants:
- Idempotent: re-entering check_chain_advance with the same tid never double-fires.
- No double-fire: _is_dispatched() consults the chain state snapshot. A leg whose
  chain status is anything other than 'pending' is treated as already fired —
  this catches dispatches that progressed past 'pending' (processed/failed/landed)
  without the chain advancing, so subsequent sibling lands won't re-fire them.
- Backwards compat: targets without chain fields are untouched by chain logic.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from agents_core.mem import MemoryStore
from agents_core.targets import TargetStore

_CHAIN_PREFIX = "chain"
_STATE_SUFFIX = "state"
_EVENT_SUFFIX = "event"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mem() -> MemoryStore:
    from lapis_pm.pm_core import _mem as _pm_mem
    return _pm_mem()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_BRANCH_SLUG_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def validate_legs(legs: list[dict]) -> None:
    """Validate a chain leg list. Raises ValueError with a clear message on failure.

    Checks:
    - No duplicate tids.
    - depends_on only references tids within the same leg set.
    - No cyclic deps (topological sort).
    - branch_slug present, non-empty, kebab-case, no slashes, no lapis/ prefix.
    """
    if not legs:
        raise ValueError("Chain must have at least one leg")

    tids = [leg["tid"] for leg in legs]

    # Duplicate tid check
    seen: set[str] = set()
    for tid in tids:
        if tid in seen:
            raise ValueError(f"Duplicate tid in chain: {tid!r}")
        seen.add(tid)

    tid_set = set(tids)

    # branch_slug validation per leg
    for leg in legs:
        tid = leg.get("tid", "<unknown>")
        slug = leg.get("branch_slug")
        if not slug:
            raise ValueError(
                f"Leg {tid!r} is missing required field 'branch_slug'. "
                "Add a non-empty kebab-case string (e.g. 'implement') to the leg."
            )
        if "/" in slug:
            raise ValueError(
                f"Leg {tid!r} branch_slug {slug!r} contains '/'; "
                "branch_slug must not contain slashes."
            )
        if slug.startswith("lapis/") or slug == "lapis":
            raise ValueError(
                f"Leg {tid!r} branch_slug {slug!r} must not start with 'lapis/'; "
                "the prefix is constructed automatically."
            )
        if not _BRANCH_SLUG_RE.match(slug):
            raise ValueError(
                f"Leg {tid!r} branch_slug {slug!r} is invalid; "
                "must be kebab-case (lowercase alphanumeric, hyphens allowed in the middle, "
                "no leading/trailing hyphens)."
            )

    # Unknown deps check
    for leg in legs:
        for dep in leg.get("depends_on") or []:
            if dep not in tid_set:
                raise ValueError(
                    f"Leg {leg['tid']!r} depends_on unknown tid {dep!r} "
                    f"(not declared in this chain set)"
                )

    # Cycle detection via iterative DFS (state: 0=unvisited, 1=in-progress, 2=done)
    deps: dict[str, set[str]] = {
        leg["tid"]: set(leg.get("depends_on") or []) for leg in legs
    }
    visited: dict[str, int] = {tid: 0 for tid in tids}

    def dfs(start: str) -> None:
        stack = [(start, iter(deps.get(start, set())))]
        visited[start] = 1
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if visited.get(child) == 1:
                    raise ValueError(
                        f"Cyclic dependency detected: {child!r} is in a dependency cycle"
                    )
                if visited.get(child, 0) == 0:
                    visited[child] = 1
                    stack.append((child, iter(deps.get(child, set()))))
            except StopIteration:
                visited[node] = 2
                stack.pop()

    for tid in tids:
        if visited[tid] == 0:
            dfs(tid)


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------

def emit_chain_event(
    group_id: str,
    kind: str,
    tid: str,
    details: dict | None = None,
) -> None:
    """Write a chain event to mem.

    kind: bind | dispatch | pr_open | pr_merged | land | auto_dispatch | complete
    Each call appends a new key chain/<group>/event/<ts>.
    """
    ts = _now_iso()
    event = {
        "kind": kind,
        "tid": tid,
        "group_id": group_id,
        "ts": ts,
        "details": details or {},
    }
    event_key = f"{_CHAIN_PREFIX}/{group_id}/{_EVENT_SUFFIX}/{ts}"
    _mem().set(
        event_key,
        json.dumps(event),
        tags=["lapis-pm", "chain", f"chain-group:{group_id}", f"chain-kind:{kind}"],
    )


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

def update_chain_state(group_id: str, legs: list[dict]) -> dict:
    """Rewrite the chain/<group>/state snapshot from the provided leg list.

    legs: list of dicts with at least {tid, status}; may include pr_num, landed_at.
    Returns the written state dict.
    """
    all_landed = all(leg.get("status") == "landed" for leg in legs)
    state: dict = {
        "group_id": group_id,
        "legs": legs,
        "complete": all_landed,
    }
    _mem().set(
        f"{_CHAIN_PREFIX}/{group_id}/{_STATE_SUFFIX}",
        json.dumps(state),
        tags=["lapis-pm", "chain", f"chain-group:{group_id}", "chain-state"],
    )
    return state


def get_chain_state(group_id: str) -> dict | None:
    """Return current chain state dict, or None if not found."""
    row = _mem().get(f"{_CHAIN_PREFIX}/{group_id}/{_STATE_SUFFIX}")
    if row is None:
        return None
    try:
        return json.loads(row["content"])
    except Exception:
        return None


def _update_leg_status_in_state(group_id: str, tid: str, new_status: str,
                                  extra: dict | None = None) -> None:
    """Update a single leg's status (and optional extra fields) in the state snapshot."""
    if not group_id:
        return
    state = get_chain_state(group_id)
    if state is None:
        return
    for leg in state.get("legs", []):
        if leg.get("tid") == tid:
            leg["status"] = new_status
            if extra:
                leg.update(extra)
            break
    all_landed = all(leg.get("status") == "landed" for leg in state.get("legs", []))
    state["complete"] = all_landed
    _mem().set(
        f"{_CHAIN_PREFIX}/{group_id}/{_STATE_SUFFIX}",
        json.dumps(state),
        tags=["lapis-pm", "chain", f"chain-group:{group_id}", "chain-state"],
    )


# ---------------------------------------------------------------------------
# Pushover brief (deprecated)
# ---------------------------------------------------------------------------

def send_auto_dispatch_brief(group_id: str, tid: str, triggered_by: str) -> bool:
    """Deprecated: chain auto-dispatch no longer pushes Pushover. Returns False."""
    return False


# ---------------------------------------------------------------------------
# Landed-tids helper
# ---------------------------------------------------------------------------

_LANDED_LIMIT = 10_000


def landed_tids() -> set[str]:
    """Return set of all tids that have pm/landed/<tid> entries in mem."""
    results = _mem().list_all(tag="landed", limit=_LANDED_LIMIT)
    if len(results) >= _LANDED_LIMIT:
        import logging
        logging.getLogger(__name__).warning(
            "landed_tids: result count hit limit=%d; landed tids may be truncated, "
            "chain advance may miss satisfied deps. Consider pruning pm/landed/* entries.",
            _LANDED_LIMIT,
        )
    tids: set[str] = set()
    for r in results:
        key = r.get("key", "")
        if key.startswith("pm/landed/"):
            tids.add(key[len("pm/landed/"):])
    return tids


# Keep private alias for internal callers in this module.
_landed_tids = landed_tids


def _is_dispatched(tid: str, chain_group: str = "") -> bool:
    """Return True if this leg has already been fired (idempotency guard).

    Canonical source: the chain state snapshot. A leg with status other than
    'pending' (i.e. 'dispatched' or 'landed') has already been fired — this
    correctly skips legs whose dispatch record has progressed past 'pending'
    (processed / failed / queue-terminal) without the chain having advanced.

    Fallback (no chain_group, or state missing): any dispatched record at all
    is treated as 'already fired'. Conservative — avoids re-firing on lifecycle
    states beyond 'pending'.
    """
    if chain_group:
        state = get_chain_state(chain_group)
        if state:
            for leg in state.get("legs", []):
                if leg.get("tid") == tid:
                    return leg.get("status") != "pending"
    from lapis_pm.pm_core import load_dispatched
    return bool(load_dispatched(tid))


# ---------------------------------------------------------------------------
# Chain advance (post-land hook)
# ---------------------------------------------------------------------------

def check_chain_advance(just_landed_tid: str) -> list[str]:
    """After just_landed_tid lands, scan for dependent chain legs to auto-fire.

    Algorithm:
    1. Collect all currently-landed tids (including the one just landed).
    2. For every pm-bound target that:
       a. Has depends_on containing just_landed_tid.
       b. Has all deps satisfied (all in landed set).
       c. Has initial_dispatch populated.
       d. Is NOT already dispatched (idempotency).
    3. Fire fixer dispatch for each qualified target.

    Returns list of tids that were auto-dispatched this call.
    """
    from lapis_pm import episodic

    store = TargetStore()
    landed = _landed_tids()
    landed.add(just_landed_tid)  # freshly landed — include even if mem write races

    auto_dispatched: list[str] = []

    for target in store.load_all():
        if not target.pm_bound:
            continue

        depends_on: list[str] = target.data.get("depends_on") or []
        if not depends_on:
            continue
        if just_landed_tid not in depends_on:
            continue

        # All deps must be satisfied
        if not all(dep in landed for dep in depends_on):
            continue

        initial_dispatch: str | None = target.data.get("initial_dispatch")
        if not initial_dispatch:
            continue

        chain_group: str = target.data.get("chain_group") or ""

        # Idempotency: skip if chain state shows this leg already fired
        # (status != 'pending'), or — without chain_group — any dispatch record exists.
        if _is_dispatched(target.id, chain_group):
            continue

        try:
            _fire_initial_dispatch(
                target=target,
                intent=initial_dispatch.strip(),
                chain_group=chain_group,
                triggered_by=just_landed_tid,
            )
            auto_dispatched.append(target.id)
        except Exception as e:
            # Log failure but don't abort — other legs can still fire
            episodic.write(
                target.id,
                f"chain-auto-dispatch FAILED: {e}",
                tags=["pm:chain-error"],
            )

    return auto_dispatched


def _fire_initial_dispatch(target, intent: str, chain_group: str, triggered_by: str) -> None:
    """Fire initial_dispatch for a chain leg via the same path as --force-dispatch."""
    from lapis_pm import pm_core, episodic

    spec_sum = episodic.spec_summary(target.id)
    vars_ = {
        "target_id": target.id,
        "spec_summary": spec_sum,
        "repo": target.pm_repo or "",
        "question": intent,
        "pr_number": "",
        "slug": "chain-auto",
    }
    res = pm_core._SHAPER.dispatch("fixer", target.id, intent, vars_=vars_)
    pm_core.append_dispatched(target.id, {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": "fixer",
        "intent": intent,
        "repo": target.pm_repo or "",
        "ts": pm_core._now_iso(),
        "status": "pending",
        "retry_count": 0,
    })
    episodic.write_dispatch(
        target.id,
        (
            f"Chain auto-dispatch: fixer → {res.task_id}\n"
            f"Triggered by: {triggered_by}\nIntent: {intent}"
        ),
        extra_tags=[
            f"pm:gpu={res.task_id}",
            "pm:agent=fixer",
            f"pm:chain-group={chain_group}",
            f"pm:chain-triggered-by={triggered_by}",
        ],
    )

    # Emit chain event + update state snapshot
    emit_chain_event(
        chain_group, "auto_dispatch", target.id,
        details={"triggered_by": triggered_by, "task_id": res.task_id},
    )
    _update_leg_status_in_state(chain_group, target.id, "dispatched")
    # chain_quiet YAML field is deprecated and currently a no-op (Pushover auto-dispatch removed).


# ---------------------------------------------------------------------------
# On-land hook (updates state snapshot for the landed leg)
# ---------------------------------------------------------------------------

def on_leg_landed(tid: str, chain_group: str) -> None:
    """Called after a chain leg lands. Updates state snapshot and emits event."""
    if not chain_group:
        return
    _update_leg_status_in_state(chain_group, tid, "landed",
                                 extra={"landed_at": _now_iso()})
    state = get_chain_state(chain_group)
    emit_chain_event(chain_group, "land", tid)
    if state and state.get("complete"):
        emit_chain_event(chain_group, "complete", tid)
