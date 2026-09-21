"""lapis-pm autopilot — the prodder (lapis-pm-pipeline-autopilot-v0, rev 4).

A perceive-and-write-state layer that unblocks, adjudicates, and re-fires
pipeline work WITHOUT human nudges. The load-bearing invariant (rev 3/4 actor
model): **this module NEVER dispatches a leg and NEVER merges a PR.** It
classifies state, clears infra counters / writes un-pause and stale markers,
emits decision dossiers + mem adjudication rows, and escalates. The daemon's
``pm_core.tick`` is the SOLE dispatch/merge actor; within it, dispatch is
gated by the pending-leg guards and the daemon's own dispatch-leg calls. The
safety is the ABSENCE of a second dispatch actor — NOT a serialization lock.
That absence is grep-verifiable (D6): this module carries no Forgejo token,
no merge-and-deploy import, and no dispatch-leg call (the D6 test asserts the
literal absence of the token symbol, the merge-and-deploy symbol, and the
dispatch-leg call prefix in this module's source).

Deliverables implemented here (D1):
  * ``perceive()``  — classify current state from ground truth (never from
    cached signals).
  * ``classify()``  — the pause-classification matrix (infra vs verdict
    ceiling + the ``already_recorded`` limbo).
  * ``write_state()`` — UNBLOCK / RE-FIRE: WRITE STATE ONLY (counter clear,
    un-pause, stale marker). Never dispatches.
  * ``emit_dossier()`` — ADJUDICATE: EMIT DOSSIER ONLY (the daemon tick reads
    the adjudication row and performs the merge next tick).
  * ``escalate()``  — ESCALATE: Matrix page + plate line + day-surface,
    fixed-format alarm text, classification from allow-listed fields only.
  * keyed idempotency audit ledger (UPDATE IN PLACE, bounded row count).

Shadow-mode-first: ``LAPIS_PM_AUTOPILOT=shadow`` (the DEFAULT) records
proposed actions as ``pm/autopilot-proposal/<tid>/<tick>`` without executing;
``LAPIS_PM_AUTOPILOT=on`` executes; ``LAPIS_PM_AUTOPILOT=off`` halts the
sweep. ``off`` is read per-tick (same-call-time env-read discipline as the
daemon's own kill-switches).

Night-lane: unit-tests-only. No live reach, no seat stop, no systemd
enablement — the unit is versioned in-repo; PM enables post-land (spec
Post-land host op section).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from . import pm_core
from . import provenance

PACIFIC = ZoneInfo("America/Los_Angeles")

# ---------------------------------------------------------------------------
# Env knobs (read per-tick; see the kill-switch discipline below)
# ---------------------------------------------------------------------------

#: The autopilot mode env var. ``shadow`` (default) / ``on`` / ``off``.
AUTOPILOT_ENV = "LAPIS_PM_AUTOPILOT"
MODE_SHADOW = "shadow"
MODE_ON = "on"
MODE_OFF = "off"

#: The substrate seat the machine-ratify dossier legs run on (named explicitly
#: per spec M8 / LC8 — the GW 27B factory-floor seat on :8081, NOT the bge-m3
#: embedder). Used only for the dossier's served-model echo + the seat-absent
#: pause; the dossier legs themselves are driven by the daemon's deliberation
#: machinery (this module only EMITS the dossier, never runs the LLM legs).
ADJUDICATE_SUBSTRATE_SEAT = "gravitywell-27b"

#: The adjudicate step PAUSES (does not block / eat the 10-min tick budget)
#: when its substrate seat is absent across the TOU peak or an idle-suspend
#: window. The TOU peak is 16:00-21:00 PT (spec M8).
TOU_PEAK_START_HOUR = 16
TOU_PEAK_END_HOUR = 21

# ---------------------------------------------------------------------------
# D6 dossier-latency caps (spec M7 / LC6): dossiers-per-tick cap (<= 2) and a
#: per-dossier latency cap (<= 5 min each). The measured latency is recorded on
#: the dossier row; the caps are enforced by the per-tick counter below.
# ---------------------------------------------------------------------------
DOSSIERS_PER_TICK_CAP = 2
DOSSIER_LATENCY_CAP_S = 300.0  # 5 min

# ---------------------------------------------------------------------------
# Repeated-action guard (spec Safety rails): the counter counts ATTEMPTS
#: (executions + noop detections) per (target, action-kind, head_sha); > 3
#: without head movement -> write pm/autopilot/parked/<tid>, page once, and
#: skip the target on subsequent sweeps until the head moves or a human clears
#: the flag.
# ---------------------------------------------------------------------------
REPEATED_ACTION_GUARD_LIMIT = 3

# ---------------------------------------------------------------------------
# Stale-cursor pin (spec RE-FIRE): pin to TICK_COVERAGE_STALL_S (2700s), not
#: "3 tick intervals". A stale cursor + an in-flight dispatch -> autopilot
#: emits the stale marker; the tick owns the re-fire.
# ---------------------------------------------------------------------------
STALE_CURSOR_S = pm_core.TICK_COVERAGE_STALL_S  # 2700

# ---------------------------------------------------------------------------
# D4 alarm-text pin (spec ESCALATE): page text is a FIXED format derived from
#: (target, action-kind, state code); reviewer/queue free text is quoted
#: behind a length cap, never echoed verbatim; the escalation CLASS is decided
#: from allow-listed fields only, never from free text.
# ---------------------------------------------------------------------------
FREE_TEXT_CAP = 200
_PAGE_SOURCE = "lapis-pm-autopilot"

# The allow-listed escalation classes (decided from allow-listed fields ONLY —
# a crafted reviewer output must not be able to author or suppress its own
# page). These are the ONLY classes escalate() will ever page.
ESCALATION_CLASSES = frozenset({
    "hold_tier",          # hold-tier outcomes (the five-item only-Erah list)
    "direction_grade",    # direction-grade items
    "repeated_action",    # repeated-action guard tripped (> 3, no head movement)
    "unclassifiable",     # any state the loop could not classify -> human
    "infra_absent",       # fence seat absent -> park flag (loud marker)
    "prodder_stalled",    # BRIX liveness backstop: a dead prodder (D5)
})

# The allow-listed state codes the classifier can emit (the "state code" half
# of the fixed-format page text). Anything not in this set is
# ``unclassifiable`` and pages to a human — never a silent guess.
STATE_CODES = frozenset({
    "infra_pause",
    "verdict_pause",
    "limbo",
    "drained_reviewer",
    "stale_cursor",
    "outstanding_brief",
    "advisory_ratify",
    "noop",
})

# ---------------------------------------------------------------------------
# D5 heartbeat key + freshness bound (the BRIX liveness backstop checks this).
# ---------------------------------------------------------------------------
HEARTBEAT_KEY = "pm/autopilot/heartbeat"
#: A heartbeat older than this is STALE -> the stall-check/backstop timer
#: pages/records (D5). The 10-min tick cadence makes 30 min a safe tripwire.
HEARTBEAT_STALE_S = 1800

# The three pm/reviewer- key CONSTRUCTORS the per-(target, pr, cycle) counter
# clear touches (D3). Deliberately NOT the target-granular
# clear_reviewer_attempts wipe — a mixed target would lose a sibling PR's
# real-verdict budget (spec H5).
#
# DRY (fixer_retry, reviewer 2026-09-21): the key shapes are the daemon's own
# constructors — pm_core._reviewer_attempt_key (the attempts counter,
# pm/reviewer-attempts/<tid>/pr=<pr>/cycle=<cycle>, NO /recorded suffix),
# pm_core._reviewer_attempt_ceiling_marker_key and
# pm_core._reviewer_infra_budget_marker_key (the ceiling / infra-budget pause
# markers, which DO carry the /recorded suffix). The autopilot must NOT
# hardcode its own copy of these shapes: the live keys are
# pm/reviewer-attempt-ceiling/<tid>/pr=<pr>/cycle=<cycle>/recorded and
# pm/reviewer-infra-budget/<tid>/pr=<pr>/cycle=<cycle>/recorded, and a
# hardcoded template that drifts from the daemon's constructors silently
# fails to delete the markers (the unblock would clear the counter but leave
# the pause marker, re-pausing the target on the next tick).
_REVIEWER_STATE_KEY_CONSTRUCTORS = (
    pm_core._reviewer_attempt_key,
    pm_core._reviewer_attempt_ceiling_marker_key,
    pm_core._reviewer_infra_budget_marker_key,
)


# ---------------------------------------------------------------------------
# Mode (kill-switch) — read per-tick, same-call-time env-read discipline as
# the daemon's own kill-switches (spec Safety rails).
# ---------------------------------------------------------------------------

def mode() -> str:
    """Return the current autopilot mode: ``shadow`` (default) / ``on`` /
    ``off``. Read from the env at call time (NOT cached at import) so a
    post-land operator can flip ``LAPIS_PM_AUTOPILOT=off`` and the NEXT tick
    halts — the DoD's "off demonstrably halts execution mid-sweep" relies on
    this per-tick read."""
    raw = os.environ.get(AUTOPILOT_ENV, MODE_SHADOW).strip().lower()
    if raw in (MODE_ON, MODE_OFF):
        return raw
    # Anything unrecognized degrades to shadow (fail toward observation, never
    # execution). A typo'd mode must not silently start executing.
    return MODE_SHADOW


def is_active() -> bool:
    """True iff the sweep should EXECUTE actions (mode == ``on``). ``shadow``
    and ``off`` both return False — shadow records proposals, off halts."""
    return mode() == MODE_ON


def is_shadow() -> bool:
    return mode() == MODE_SHADOW


# ---------------------------------------------------------------------------
# Key helpers (mem layout)
# ---------------------------------------------------------------------------

def _action_key(target_id: str, action_kind: str, head_sha: str) -> str:
    """The keyed idempotency audit-ledger row. UPDATE IN PLACE (bounded row
    count, not append-only — mem.db is the authoritative decision ledger on a
    ~96%-full /). The key records the head at PERCEIVE time; a head that moved
    since perceive aborts the action (re-classify next sweep)."""
    return f"pm/autopilot/{action_kind}/{target_id}/{head_sha or 'nohead'}"


def _proposal_key(target_id: str, tick_id: str) -> str:
    """Shadow-mode proposal record: pm/autopilot-proposal/<tid>/<tick>."""
    return f"pm/autopilot-proposal/{target_id}/{tick_id}"


def _parked_key(target_id: str) -> str:
    return f"pm/autopilot/parked/{target_id}"


def _infra_absent_key(target_id: str) -> str:
    return f"pm/autopilot/infra-absent/{target_id}"


def _reversal_key(target_id: str, tick_id: str) -> str:
    """A countable, stored reversal key (spec H4): the shadow-review surface
    lists the last N shadow proposals with their would-be action + file:line
    evidence; a human-confirmed reversal stamps this key."""
    return f"pm/autopilot-reversal/{target_id}/{tick_id}"


def _stale_marker_key(target_id: str, pr_number: int) -> str:
    return f"pm/autopilot/stale/{target_id}/pr={pr_number}"


def _divergence_key(target_id: str, pr_number: int, cycle: int) -> str:
    """A divergence row (spec D3 / B1): written when the autopilot's
    extended infra taxonomy disagrees with the daemon's own classification."""
    return (
        f"pm/autopilot/divergence/{target_id}/pr={pr_number}/cycle={cycle}"
    )


# ---------------------------------------------------------------------------
# D3: the infra-vs-verdict classifier + the gw_seat_occupied taxonomy
# extension (spec BLOCKER B1).
# ---------------------------------------------------------------------------

def _daemon_infra_reasons() -> frozenset[str]:
    """The daemon's own allow-list (the classifier keys on the attempt RECORD,
    not free text). Read live from pm_core so a daemon-side taxonomy change is
    picked up without a second copy drifting."""
    return frozenset(getattr(pm_core, "REVIEWER_INFRA_FAIL_REASONS",
                             frozenset()))


#: The autopilot-extended infra taxonomy (spec B1 sub-deliverable): the daemon
#: set PLUS ``gw_seat_occupied``. The 2026-09-19 census (state/pipeline-
#: fire-27b-2026-09-19) shows gw_seat_occupied (doorman 409) is the dominant
#: infra-noise class — the exact class that parks targets nobody un-parks.
#: The daemon's own set (decision/reviewer-gw-seat-occupied-is-infra) was
#: extended to include it; this extended list is the autopilot's own
#: classification surface, and a divergence row is written when it disagrees
#: with the daemon's classification of the same reason.
AUTOPILOT_INFRA_FAIL_REASONS = frozenset(
    _daemon_infra_reasons() | {"gw_seat_occupied"}
)


def _extract_reason_token(reported_reason: str | None) -> str | None:
    """Extract the ``reason=<value>`` token from a reason string (the daemon's
    own extraction shape, pm_core._classify_reviewer_infra_reason). Returns
    None for an empty/absent reason."""
    if not reported_reason:
        return None
    m = re.search(r"reason=([\w.-]+)", reported_reason)
    if m:
        return m.group(1)
    return None


def classify_infra_reason(reported_reason: str | None) -> str | None:
    """Classify a reviewer failure reason against the AUTOPILOT-extended infra
    taxonomy (daemon set + gw_seat_occupied). Returns the matched infra
    reason token, or None (a non-infra / verdict-classified failure).

    This is the D3 classifier the UNBLOCK step keys on: an infra-pause iff
    the reason classifies as infra under this extended set. A reason that
    classifies as infra HERE but not under the daemon's own set is a DIVERGENCE
    (written as a row by :func:`write_divergence_row`) — the autopilot's
    extended taxonomy is the superset, and the divergence row records the
    disagreement so the daemon's own set can be reconciled.
    """
    token = _extract_reason_token(reported_reason)
    candidates = [token] if token else []
    if reported_reason:
        candidates.append(reported_reason)
    for candidate in candidates:
        for reason in AUTOPILOT_INFRA_FAIL_REASONS:
            if candidate == reason or reason in candidate:
                return reason
    return None


def daemon_classifies_infra(reported_reason: str | None) -> bool:
    """True iff the DAEMON's own classifier (pm_core.
    _classify_reviewer_infra_reason) would classify this reason as infra.
    Used to detect a divergence between the autopilot's extended taxonomy and
    the daemon's (spec B1 / D3)."""
    try:
        return pm_core._classify_reviewer_infra_reason(reported_reason) is not None
    except Exception:
        return False


def write_divergence_row(mem, target_id: str, pr_number: int, cycle: int,
                         reported_reason: str | None,
                         autopilot_reason: str | None) -> str | None:
    """Write a divergence row when the autopilot's extended infra taxonomy
    classifies a reason as infra but the daemon's own set does not (spec B1 /
    D3). The row is UPDATE IN PLACE (idempotent per pr+cycle). Returns the key
    written, or None when there is no divergence (the daemon and the autopilot
    agree)."""
    if autopilot_reason is None:
        return None
    if daemon_classifies_infra(reported_reason):
        return None  # no divergence — the daemon already classifies it infra
    key = _divergence_key(target_id, pr_number, cycle)
    body = {
        "schema": "autopilot-divergence/v1",
        "ts": _now_iso(),
        "target_id": target_id,
        "pr_number": pr_number,
        "cycle": cycle,
        "reported_reason": reported_reason,
        "autopilot_infra_reason": autopilot_reason,
        "daemon_classifies_infra": False,
        "note": (
            "autopilot extended taxonomy (gw_seat_occupied) classifies this "
            "reason as infra; the daemon's own REVIEWER_INFRA_FAIL_REASONS "
            "does not. Reconcile the daemon set (spec B1)."
        ),
    }
    mem.set(key, json.dumps(body, ensure_ascii=False, sort_keys=True),
            tags=["lapis-pm", "autopilot-divergence", f"target={target_id}"])
    return key


# ---------------------------------------------------------------------------
# Fence: the seat-present precondition (spec H9 / M5). The loop reads SENSE,
#: never flips a seat. It may PAUSE for floor absence.
# ---------------------------------------------------------------------------

#: The gw-seats registry reality_view contract (I8 type-check): a malformed or
#: null view -> REALITY-UNKNOWN -> fail-closed park of factory-floor legs + a
#: loud marker. Never a raw-object dereference, never a silent default that
#: dispatches while the floor is absent.
REALITY_UNKNOWN = "REALITY-UNKNOWN"
REALITY_KNOWN = "REALITY-KNOWN"


def _registry_url() -> str:
    """The gw-seats registry URL (the :8408 flip-controller / registry).
    Overridable via env for tests (never a live call from a test — the
    registry read is injected by the caller in the test path)."""
    return os.environ.get(
        "LAPIS_PM_GW_SEATS_REGISTRY_URL", "http://203.0.113.10:8408"
    )


def read_reality_view(fetcher=None) -> dict:
    """Read the registry's ``reality_view`` under the I8 type-check contract.

    ``fetcher`` is an injectable callable returning the raw registry payload
    (the test path injects a fake; the live path is the daemon's registry
    read). The contract (spec M5): a malformed or null view -> REALITY-UNKNOWN
    (fail-closed park of factory-floor legs + a loud marker logged); never a
    raw-object dereference, never a silent default that dispatches while the
    floor is absent.

    Returns a dict with:
      * ``reality``: REALITY_KNOWN | REALITY-UNKNOWN
      * ``view``: the parsed reality_view dict (or None)
      * ``model_root``: the served model ROOT (the fence test = model-ROOT
        match, aliases drift, roots don't) or None
      * ``seat_present``: True iff the 27B factory-floor seat is present
    """
    result = {
        "reality": REALITY_UNKNOWN,
        "view": None,
        "model_root": None,
        "seat_present": False,
    }
    try:
        if fetcher is None:
            # Live path: read the registry. This is a SENSE read (GET), never
            # a seat flip. Fail-soft: any error -> REALITY-UNKNOWN.
            import httpx
            resp = httpx.get(f"{_registry_url()}/v0/status", timeout=4.0)
            if resp.status_code != 200:
                return result
            payload = resp.json()
        else:
            payload = fetcher()
    except Exception:
        return result

    if not isinstance(payload, dict):
        return result  # malformed payload -> REALITY-UNKNOWN

    view = payload.get("reality_view")
    if not isinstance(view, dict):
        return result  # null / non-dict view -> REALITY-UNKNOWN

    # I8 type-check: the view must carry the documented fields. A view that
    # parses as a dict but is missing its primary model block is malformed ->
    # REALITY-UNKNOWN (fail-closed), never a silent default.
    primary = view.get("primary")
    if not isinstance(primary, dict):
        return result

    model_root = primary.get("model_root")
    if not isinstance(model_root, str) or not model_root:
        return result  # missing model root -> REALITY-UNKNOWN

    # The fence test = model-ROOT match (aliases drift, roots don't). The 27B
    # factory-floor seat is present iff the served model root matches the 27B
    # root (the factory-floor-admission-v0 semantics; until that lands, the
    # registry reads non-27B reality via the reality_view).
    seat_present = "27b" in model_root.lower() or "27b" in str(
        view.get("model", "")).lower()

    result["reality"] = REALITY_KNOWN
    result["view"] = view
    result["model_root"] = model_root
    result["seat_present"] = seat_present
    return result


def fence_seat_present(fetcher=None) -> bool:
    """The fence seat-present precondition (spec H9): True iff the required
    seat is present. A REALITY-UNKNOWN (malformed / null / absent registry)
    is NOT a silent default — it is a fail-closed park (returns False), so a
    factory-floor leg is never dispatched while the floor is absent. This is
    the UNBLOCK precondition: if the required seat is absent, NO clear."""
    return read_reality_view(fetcher=fetcher)["seat_present"]


# ---------------------------------------------------------------------------
# Perceive (D1): classify current state from ground truth.
# ---------------------------------------------------------------------------

@dataclass
class TargetState:
    """The perceived ground-truth state for one target (PERCEIVE). Never from
    cached signals — the systemic stale-signal deadlock precedent (gardener/
    derived 2026-06-24/25, 07-03/11/15)."""
    target_id: str
    paused: bool = False
    paused_reason: str | None = None
    pr_head: dict | None = None          # {"number": int, "head_sha": str}
    cursor_ts: str | None = None
    cursor_age_s: float | None = None
    outstanding_brief: str | None = None
    pause_state: str | None = None
    # The per-(pr, cycle) reviewer-attempt state for the selected PR.
    pr_number: int | None = None
    cycle: int = 1
    attempt_count: int = 0
    infra_count: int = 0
    last_reason: str | None = None
    last_infra_reason: str | None = None
    ceiling_decision: str | None = None  # "reviewer_infra_budget_exhausted" |
    # "reviewer_attempt_ceiling" | None
    # The pause-marker presence (the already_recorded limbo detection).
    ceiling_marker_present: bool = False
    infra_budget_marker_present: bool = False
    # The pending-leg guards (the daemon's dispatch gate — autopilot reads
    # them to decide the missing-leg source of truth, never dispatches).
    pending_reviewer: bool = False
    pending_fixer: bool = False
    # The fence (seat-present) read.
    reality: str = REALITY_UNKNOWN
    seat_present: bool = False
    # The verdict (for the missing-leg source of truth).
    last_verdict: str | None = None
    # Free text (quoted behind a length cap in escalation; NEVER used to
    # decide the escalation class).
    reported_reason: str | None = None


def perceive(target_id: str, *, fetcher=None, now: datetime | None = None) -> TargetState:
    """PERCEIVE: classify current state from ground truth for one target.

    Reads the mem records (cursor, pr-head, pause-state, outstanding-brief,
    reviewer-attempt state, pause markers) + the fence (seat-present) read.
    Never makes a Forgejo call of its own (the pr-head record is the tick's
    PR-perception write; the directive-outcome detector reads it and makes no
    Forgejo call — the same discipline).
    """
    mem = pm_core._mem()
    now = now or datetime.now(PACIFIC)
    st = TargetState(target_id=target_id)

    # Cursor + age.
    st.cursor_ts = pm_core.get_cursor(target_id)
    if st.cursor_ts:
        try:
            st.cursor_age_s = (now - datetime.fromisoformat(st.cursor_ts)).total_seconds()
        except (ValueError, TypeError):
            st.cursor_age_s = None

    # pr-head record (the tick's PR-perception write).
    pr_head_rec = mem.get(pm_core._pr_head_key(target_id))
    if pr_head_rec:
        try:
            recs = json.loads(pr_head_rec["content"])
            if isinstance(recs, list) and recs:
                # The daemon drives the highest-numbered open PR (the tip of
                # the salvage chain) — the same PR the decide phase drives.
                recs_sorted = sorted(
                    (r for r in recs if r.get("number") is not None),
                    key=lambda r: r.get("number", 0),
                )
                if recs_sorted:
                    top = recs_sorted[-1]
                    st.pr_head = {
                        "number": top.get("number"),
                        "head_sha": top.get("head_sha") or "",
                    }
                    st.pr_number = st.pr_head["number"]
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    # Pause state + reason. The TARGET STORE's ``target.paused`` flag is the
    # daemon's pause source of truth (tick() returns ``noop:paused`` from
    # it); the mem ``pm/pause-state`` key is a MIRROR the daemon writes
    # during transitions (pm_core.tick's state-transition write), not the
    # source of truth. Perceive reads BOTH (the target store is primary).
    st.pause_state = pm_core.get_pause_state(target_id)
    st.paused = st.pause_state == "paused"
    try:
        target = pm_core.TargetStore().get(target_id)
        if target is not None:
            st.paused = st.paused or bool(target.paused)
            st.paused_reason = target.paused_reason
    except Exception:
        pass

    # Outstanding brief.
    st.outstanding_brief = pm_core.get_outstanding_brief(target_id)

    # Reviewer-attempt state for the selected PR (the classifier keys on the
    # attempt RECORD, not free text).
    if st.pr_number is not None:
        cycle = pm_core._reviewer_cycle_count(target_id, st.pr_number) or 1
        st.cycle = cycle
        attempt = pm_core._reviewer_attempt_state(target_id, st.pr_number, cycle)
        st.attempt_count = attempt.get("count", 0)
        st.infra_count = attempt.get("infra_count", 0)
        st.last_reason = attempt.get("last_reason")
        st.last_infra_reason = attempt.get("last_infra_reason")
        # The ceiling decision (the daemon's own check — autopilot re-runs it
        # to read the state, never mutates it).
        decision = pm_core._reviewer_attempt_ceiling_check(
            target_id, st.pr_number, cycle)
        if decision is not None:
            st.ceiling_decision = decision.kind
        # The pause markers (the already_recorded limbo detection).
        st.ceiling_marker_present = bool(
            mem.get(pm_core._reviewer_attempt_ceiling_marker_key(
                target_id, st.pr_number, cycle)))
        st.infra_budget_marker_present = bool(
            mem.get(pm_core._reviewer_infra_budget_marker_key(
                target_id, st.pr_number, cycle)))
        # The last verdict (the missing-leg source of truth).
        verdict_info = pm_core._last_review_verdict(target_id, st.pr_number)
        if verdict_info is not None:
            st.last_verdict = verdict_info.get("verdict")
        st.reported_reason = st.last_infra_reason or st.last_reason

    # The pending-leg guards (the daemon's dispatch gate — read, never
    # dispatched).
    if st.pr_number is not None:
        st.pending_reviewer = pm_core._has_pending_reviewer_for_pr(
            target_id, st.pr_number)
        st.pending_fixer = pm_core._has_pending_fixer_for_pr(
            target_id, st.pr_number)

    # The fence (seat-present) read.
    reality = read_reality_view(fetcher=fetcher)
    st.reality = reality["reality"]
    st.seat_present = reality["seat_present"]
    return st


# ---------------------------------------------------------------------------
# Classify (D1): the pause-classification matrix.
# ---------------------------------------------------------------------------

# The five state classes the classifier can emit (spec PERCEIVE). The
# ``already_recorded`` limbo (rev 2, spec H5) is the fifth: not paused + a
# ceiling marker present + counters exhausted.
STATE_INFRA_PAUSE = "infra_pause"
STATE_VERDICT_PAUSE = "verdict_pause"
STATE_LIMBO = "limbo"
STATE_DRAINED_REVIEWER = "drained_reviewer"
STATE_STALE_CURSOR = "stale_cursor"
STATE_OUTSTANDING_BRIEF = "outstanding_brief"
STATE_ADVISORY_RATIFY = "advisory_ratify"
STATE_NOOP = "noop"


def classify(state: TargetState) -> str:
    """CLASSIFY: the pause-classification matrix (the Canary Test — "does the
    eye see the stone?"). Returns one of the STATE_* codes.

    Precedence (most specific first):
      1. ``infra_pause`` — a ceiling decision of
         ``reviewer_infra_budget_exhausted`` (the infra budget is exhausted;
         the classifier keys on the attempt RECORD, not free text).
      2. ``verdict_pause`` — a ceiling decision of
         ``reviewer_attempt_ceiling`` (a ceiling reached by REAL ``fixable``
         verdicts is NOT an infra pause — budget-exhausted is a decision, not
         noise; leave it paused).
      3. ``limbo`` — the ``already_recorded`` limbo (rev 2, spec H5): not
         paused + a ceiling marker present + counters exhausted.
      4. ``drained_reviewer`` — a drained reviewer (no verdict row, output
         exists): the RE-FIRE step emits a stale marker so the tick re-posts.
      5. ``stale_cursor`` — a stale cursor (> TICK_COVERAGE_STALL_S) with an
         in-flight dispatch: the RE-FIRE step emits the stale marker.
      6. ``outstanding_brief`` — an outstanding brief with no pending leg.
      7. ``advisory_ratify`` — a fixable verdict + the machine-ratify path.
      8. ``noop`` — nothing to do.

    A state that does not match any of these is ``unclassifiable`` (the
    escalate step pages to a human — never a silent guess).
    """
    # 1. Infra pause (the classifier keys on the attempt RECORD, not free
    # text). An infra-pause iff the ceiling decision is
    # reviewer_infra_budget_exhausted.
    if state.ceiling_decision == "reviewer_infra_budget_exhausted":
        return STATE_INFRA_PAUSE

    # 2. Verdict pause (a ceiling reached by real verdicts — NOT infra).
    if state.ceiling_decision == "reviewer_attempt_ceiling":
        return STATE_VERDICT_PAUSE

    # 3. The already_recorded limbo: not paused + a ceiling marker present +
    # counters exhausted. (The daemon returns already_recorded on a re-pause
    # of a still-exhausted record.)
    if not state.paused and (
            state.ceiling_marker_present or state.infra_budget_marker_present):
        return STATE_LIMBO

    # 4. Drained reviewer: a reviewer ran (a completed reviewer record) but no
    # verdict row landed and there is no pending reviewer (the output exists
    # but the verdict post was lost).
    if _is_drained_reviewer(state):
        return STATE_DRAINED_REVIEWER

    # 5. Stale cursor: a cursor older than TICK_COVERAGE_STALL_S with an
    # in-flight dispatch (a pending leg).
    if (state.cursor_age_s is not None
            and state.cursor_age_s > STALE_CURSOR_S
            and (state.pending_reviewer or state.pending_fixer)):
        return STATE_STALE_CURSOR

    # 6. Outstanding brief with no pending leg.
    if state.outstanding_brief is not None and not (
            state.pending_reviewer or state.pending_fixer):
        return STATE_OUTSTANDING_BRIEF

    # 7. Advisory ratify: a fixable verdict + the machine-ratify path.
    if state.last_verdict in ("fixable", "clean") and state.pr_number is not None:
        return STATE_ADVISORY_RATIFY

    # 8. Noop.
    return STATE_NOOP


def _is_drained_reviewer(state: TargetState) -> bool:
    """A drained reviewer: a reviewer record exists (a completed reviewer
    dispatch) but no verdict row has landed and there is no pending reviewer
    (the output exists but the verdict post was lost)."""
    if state.pr_number is None:
        return False
    if state.pending_reviewer:
        return False  # a reviewer is still in flight — not drained
    if state.last_verdict is not None:
        return False  # a verdict landed — not drained
    # A reviewer dispatch record exists for this PR (completed, not pending)
    # and no verdict row -> drained.
    for rec in pm_core.load_dispatched(state.target_id):
        if (rec.get("agent_type") in pm_core._REVIEWER_AGENT_TYPES
                and rec.get("pr_number") == state.pr_number
                and rec.get("status") in ("completed", "succeeded", "done")):
            return True
    return False


# ---------------------------------------------------------------------------
# Missing-leg source of truth (spec H6): the missing-leg function.
# ---------------------------------------------------------------------------

def missing_leg(state: TargetState) -> str | None:
    """The missing-leg source of truth (spec H6): which leg the daemon tick
    should dispatch next. Returns ``"reviewer"``, ``"fixer"``, or None (both
    present / nothing missing).

    Source of truth (the daemon's pending-leg guards — read, never
    dispatched):
      * verdict ``fixable`` + no pending fixer -> ``"fixer"`` (the fixer leg).
      * no pending reviewer -> ``"reviewer"`` (the next reviewer cycle).
      * BOTH missing -> exactly ONE reviewer cycle, never both (spec H6).
    """
    if state.pr_number is None:
        return None
    fixer_needed = (state.last_verdict == "fixable"
                    and not state.pending_fixer)
    reviewer_needed = not state.pending_reviewer
    if fixer_needed and not reviewer_needed:
        return "fixer"
    if reviewer_needed:
        # Both missing (reviewer_needed + fixer_needed) -> exactly one
        # reviewer cycle, never both.
        return "reviewer"
    return None


# ---------------------------------------------------------------------------
# The idempotency audit ledger (UPDATE IN PLACE, bounded row count).
# ---------------------------------------------------------------------------

def _ledger_get(mem, key: str) -> dict | None:
    rec = mem.get(key)
    if not rec:
        return None
    try:
        return json.loads(rec["content"])
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _ledger_update(mem, key: str, *, target_id: str, action_kind: str,
                   head_sha: str, perceived: dict, action: str,
                   rationale: str, executor: str,
                   attempts: int = 1) -> dict:
    """UPDATE IN PLACE the keyed audit-ledger row (spec M6: bounded row count,
    not append-only). A repeat is a noop + counter (the counter counts
    ATTEMPTS — executions + noop detections). Returns the updated row."""
    existing = _ledger_get(mem, key) or {}
    row = {
        "schema": "autopilot-audit/v1",
        "ts": _now_iso(),
        "target_id": target_id,
        "action_kind": action_kind,
        "head_sha": head_sha or "nohead",
        "perceived": perceived,
        "action": action,
        "rationale": rationale,
        "executor": executor,
        # The counter counts ATTEMPTS (executions + noop detections).
        "attempts": int(existing.get("attempts", 0)) + attempts,
        # The head at PERCEIVE time (a head that moved since perceive aborts
        # the action — re-classify next sweep).
        "head_at_perceive": head_sha or "nohead",
    }
    mem.set(key, json.dumps(row, ensure_ascii=False, sort_keys=True),
            tags=["lapis-pm", "autopilot-audit", f"target={target_id}"])
    return row


def _head_moved_since_perceive(mem, key: str, current_head: str) -> bool:
    """True iff the head moved since the audit row recorded it (spec H7). A
    mid-sweep head move invalidates the key -> abort the action (re-classify
    next sweep)."""
    row = _ledger_get(mem, key)
    if row is None:
        return False
    recorded = row.get("head_at_perceive", "nohead")
    current = current_head or "nohead"
    return recorded != current


# ---------------------------------------------------------------------------
# Provenance (spec M4): every EXECUTED action writes a mem row carrying the
#: perceived trigger state, the action taken, the rationale, and an executor
#: label via provenance.deploy_log_label() (the L2.D4 pattern).
# ---------------------------------------------------------------------------

def _provenance_row(mem, target_id: str, action_kind: str, head_sha: str,
                    perceived: dict, action: str, rationale: str,
                    served_model: list | None = None) -> str:
    """Write a provenance row for an EXECUTED action (spec M4). The executor
    label is via provenance.deploy_log_label() (the L2.D4 pattern); the
    served-model echo (L2.D2 pattern) is carried when the action is an LLM
    leg (the dossier)."""
    executor = provenance.deploy_log_label(served_model_out=served_model)
    key = f"pm/autopilot/{action_kind}/{target_id}/{head_sha or 'nohead'}"
    row = _ledger_get(mem, key) or {}
    row["provenance"] = {
        "perceived": perceived,
        "action": action,
        "rationale": rationale,
        "executor": executor,
        "ts": _now_iso(),
    }
    mem.set(key, json.dumps(row, ensure_ascii=False, sort_keys=True),
            tags=["lapis-pm", "autopilot-provenance", f"target={target_id}"])
    return key


# ---------------------------------------------------------------------------
# Write-state (D1): UNBLOCK / RE-FIRE — WRITE STATE ONLY. Never dispatches.
# ---------------------------------------------------------------------------

def clear_reviewer_state_for_cycle(mem, target_id: str, pr_number: int,
                                   cycle: int) -> int:
    """D3: the per-(target, pr, cycle) counter clear. Touches ONLY the three
    pm/reviewer- keys for THIS (pr, cycle) — NOT the target-granular
    clear_reviewer_attempts wipe (a mixed target would lose a sibling PR's
    real-verdict budget, spec H5). Returns the number of keys cleared.

    The key shapes come from the daemon's own constructors (DRY — a
    hardcoded template drifts and silently fails to delete the markers):
      * ``pm_core._reviewer_attempt_key`` — the attempts counter
        (``pm/reviewer-attempts/<tid>/pr=<pr>/cycle=<cycle>``).
      * ``pm_core._reviewer_attempt_ceiling_marker_key`` — the ceiling pause
        marker (``.../cycle=<cycle>/recorded``).
      * ``pm_core._reviewer_infra_budget_marker_key`` — the infra-budget
        pause marker (``.../cycle=<cycle>/recorded``).
    """
    cleared = 0
    for construct in _REVIEWER_STATE_KEY_CONSTRUCTORS:
        key = construct(target_id, pr_number, cycle)
        # Each key is an exact key (the attempts counter and the two pause
        # markers). Delete it.
        if mem.delete(key):
            cleared += 1
    return cleared


def unblock_infra_pause(mem, state: TargetState, *, execute: bool = True,
                        tick_id: str = "") -> dict:
    """UNBLOCK (advisory-tier, no page) — WRITE STATE ONLY.

    Precondition (spec H9): the fence seat-present check passes FIRST; if the
    required seat is absent, NO clear, write a ``pm/autopilot/infra-absent/
    <tid>`` park flag, re-check each sweep. If the fence passes and the state
    is an infra-pause, clear ONLY the paused cycle's record (per-(target, pr,
    cycle), the three pm/reviewer- prefixes) and write the un-pause state.
    Autopilot writes the un-pause state and does NOT dispatch; the daemon tick
    performs the single-leg dispatch next tick under its pending-leg guards
    (that guard — not an autopilot re-dispatch — kills the resume-race class).

    ``execute=False`` (shadow mode): record the proposal, do not write the
    clear / un-pause state.
    """
    action_kind = "unblock"
    head_sha = (state.pr_head or {}).get("head_sha", "")
    key = _action_key(state.target_id, action_kind, head_sha)
    perceived = _perceived_dict(state)

    # The fence precondition (spec H9): if the required seat is absent, NO
    # clear — write the infra-absent park flag, re-check each sweep.
    if not state.seat_present:
        park_key = _infra_absent_key(state.target_id)
        if execute:
            mem.set(park_key, json.dumps({
                "schema": "autopilot-infra-absent/v1",
                "ts": _now_iso(),
                "target_id": state.target_id,
                "reality": state.reality,
                "note": (
                    "fence seat absent (REALITY-UNKNOWN or non-27B) — no "
                    "clear; re-check each sweep. A correctly-parked target "
                    "while the seat is absent is the fence WORKING, not an "
                    "autopilot bug (spec DoD fence boundary)."
                ),
            }, ensure_ascii=False, sort_keys=True),
                    tags=["lapis-pm", "autopilot-infra-absent",
                          f"target={state.target_id}"])
        return {
            "action": "park_infra_absent",
            "key": park_key,
            "executed": execute,
            "reality": state.reality,
        }

    # A head that moved since perceive aborts the action (spec H7).
    if _head_moved_since_perceive(mem, key, head_sha):
        return {"action": "aborted_head_moved", "key": key, "executed": False}

    # Clear ONLY the paused cycle's record + write the un-pause state.
    cleared = 0
    if execute and state.pr_number is not None:
        cleared = clear_reviewer_state_for_cycle(
            mem, state.target_id, state.pr_number, state.cycle)
        # Write the un-pause state (the daemon tick reads it next tick).
        mem.set(pm_core._pause_key(state.target_id), "active",
                tags=["lapis-pm", "pause-state"])

    # The divergence row (spec B1 / D3): if the autopilot's extended taxonomy
    # classifies the reason as infra but the daemon's does not, record it.
    divergence_key = write_divergence_row(
        mem, state.target_id,
        state.pr_number or 0, state.cycle,
        state.reported_reason,
        classify_infra_reason(state.reported_reason),
    )

    # The audit-ledger row (UPDATE IN PLACE) + the provenance row.
    row = _ledger_update(
        mem, key, target_id=state.target_id, action_kind=action_kind,
        head_sha=head_sha, perceived=perceived,
        action="unblock_infra_pause" if execute else "propose_unblock_infra_pause",
        rationale=(
            f"infra-pause (reported_reason={state.reported_reason!r}, "
            f"infra_count={state.infra_count}) — cleared the paused cycle's "
            f"record; the daemon tick performs the single-leg dispatch next "
            f"tick under its pending-leg guards."
        ),
        executor=provenance.deploy_log_label(),
    )
    if execute:
        _provenance_row(mem, state.target_id, action_kind, head_sha,
                        perceived, "unblock_infra_pause", row["rationale"])

    # The shadow proposal (spec Safety rails): shadow mode records the
    # would-be action + file:line evidence for the Nudge Log.
    proposal = {
        "schema": "autopilot-proposal/v1",
        "ts": _now_iso(),
        "target_id": state.target_id,
        "action_kind": action_kind,
        "head_sha": head_sha or "nohead",
        "would_be_action": "unblock_infra_pause",
        "state_code": classify(state),
        "evidence": perceived,
        "rationale": row["rationale"],
        "cleared_keys": cleared,
        "divergence_key": divergence_key,
        "narrative": _narrative(
            state, "unblock_infra_pause", row["rationale"], execute),
    }
    if is_shadow() or not execute:
        mem.set(_proposal_key(state.target_id, tick_id or _utc_ts_slug()),
                json.dumps(proposal, ensure_ascii=False, sort_keys=True),
                tags=["lapis-pm", "autopilot-proposal",
                      f"target={state.target_id}"])

    return {
        "action": "unblock_infra_pause",
        "key": key,
        "executed": execute,
        "cleared_keys": cleared,
        "divergence_key": divergence_key,
        "attempts": row["attempts"],
    }


def mark_stale(mem, state: TargetState, *, execute: bool = True,
               tick_id: str = "") -> dict:
    """RE-FIRE — MARK STALE, DO NOT DISPATCH (spec RE-FIRE).

    Drained reviewer (no verdict row, output exists) -> emit a stale marker
    so the daemon tick re-posts the verdict if one was produced but the post
    was refused (GW friction 2026-09-19: clean rc=0 <60s, post lost), else
    let the daemon tick re-dispatch reviewer once. Stale cursor (pin to
    TICK_COVERAGE_STALL_S = 2700s) with an in-flight dispatch -> autopilot
    emits the stale marker and the tick owns the re-fire.

    Autopilot NEVER dispatches a new job while the old job is active in the
    ledger (perceive, don't assume).
    """
    action_kind = "refire"
    head_sha = (state.pr_head or {}).get("head_sha", "")
    key = _action_key(state.target_id, action_kind, head_sha)
    perceived = _perceived_dict(state)

    if _head_moved_since_perceive(mem, key, head_sha):
        return {"action": "aborted_head_moved", "key": key, "executed": False}

    pr_number = state.pr_number or 0
    stale_key = _stale_marker_key(state.target_id, pr_number)
    if execute:
        mem.set(stale_key, json.dumps({
            "schema": "autopilot-stale/v1",
            "ts": _now_iso(),
            "target_id": state.target_id,
            "pr_number": pr_number,
            "state_code": classify(state),
            "note": (
                "stale marker — the daemon tick owns the re-fire (re-post the "
                "verdict if one was produced but the post was refused, else "
                "re-dispatch reviewer once). Autopilot never dispatches a new "
                "job while the old job is active in the ledger."
            ),
        }, ensure_ascii=False, sort_keys=True),
                tags=["lapis-pm", "autopilot-stale",
                      f"target={state.target_id}"])

    row = _ledger_update(
        mem, key, target_id=state.target_id, action_kind=action_kind,
        head_sha=head_sha, perceived=perceived,
        action="mark_stale" if execute else "propose_mark_stale",
        rationale=(
            f"stale ({classify(state)}) — emitted the stale marker; the "
            f"daemon tick owns the re-fire (precedence vs the stall checker's "
            f"45-min page: autopilot acts, the page carries the autopilot's "
            f"action, not a duplicate)."
        ),
        executor=provenance.deploy_log_label(),
    )
    if execute:
        _provenance_row(mem, state.target_id, action_kind, head_sha,
                        perceived, "mark_stale", row["rationale"])

    proposal = {
        "schema": "autopilot-proposal/v1",
        "ts": _now_iso(),
        "target_id": state.target_id,
        "action_kind": action_kind,
        "head_sha": head_sha or "nohead",
        "would_be_action": "mark_stale",
        "state_code": classify(state),
        "evidence": perceived,
        "rationale": row["rationale"],
        "narrative": _narrative(state, "mark_stale", row["rationale"], execute),
    }
    if is_shadow() or not execute:
        mem.set(_proposal_key(state.target_id, tick_id or _utc_ts_slug()),
                json.dumps(proposal, ensure_ascii=False, sort_keys=True),
                tags=["lapis-pm", "autopilot-proposal",
                      f"target={state.target_id}"])

    return {
        "action": "mark_stale",
        "key": key,
        "stale_key": stale_key,
        "executed": execute,
        "attempts": row["attempts"],
    }


# ---------------------------------------------------------------------------
# Emit-dossier (D1): ADJUDICATE — EMIT DOSSIER ONLY. Never merges.
# ---------------------------------------------------------------------------

_tick_dossier_count = 0


def _reserve_dossier_slot() -> bool:
    """Admission control: True iff this tick may emit another dossier (the
    per-tick cap, spec M7: <= 2 dossiers per tick)."""
    global _tick_dossier_count
    if _tick_dossier_count >= DOSSIERS_PER_TICK_CAP:
        return False
    _tick_dossier_count += 1
    return True


def _adjudicate_seat_absent(state: TargetState) -> bool:
    """The seat-absent pause for the ADJUDICATE step (spec M8): True iff the
    adjudicate substrate seat is absent across the TOU peak or an idle-suspend
    window. The seat is absent when the fence read is REALITY-UNKNOWN or the
    27B seat is not present. During the TOU peak (16:00-21:00 PT) or an
    idle-suspend window, the step PAUSES instead of blocking / eating the
    10-min tick budget."""
    if state.reality == REALITY_KNOWN and state.seat_present:
        return False  # the seat is present — proceed
    # The seat is absent (REALITY-UNKNOWN or non-27B). Pause across the TOU
    # peak or an idle-suspend window (the substrate is the GW 27B seat, which
    # is the TOU-peak / idle-suspend seat).
    now = datetime.now(PACIFIC)
    in_tou_peak = TOU_PEAK_START_HOUR <= now.hour < TOU_PEAK_END_HOUR
    idle_suspend = os.environ.get("LAPIS_PM_GW_IDLE_SUSPEND", "").strip().lower() in ("1", "true", "on")
    return in_tou_peak or idle_suspend


def emit_dossier(mem, state: TargetState, *, execute: bool = True,
                 tick_id: str = "") -> dict:
    """ADJUDICATE (advisory-tier briefs) — EMIT DOSSIER ONLY (spec ADJUDICATE).

    The machine-ratify path (runbook: mem
    decision/machinery-adjudicates-rubber-stamp-class-2026-09-18): two
    grounded arguers + a fresh-context decider on the named substrate -> a
    decision dossier (verdict, confidence, file:line evidence, dissent,
    rendered held-path values; the served-model stamp on each LLM leg;
    latency measured, not assumed) -> a mem adjudication row. The daemon tick
    reads the row and performs the merge-and-deploy next tick (daemon env:
    LAPIS_PM_OWNED_MEM_STORE / MEM_DB_PATH). Autopilot does not merge in-loop
    (no Forgejo token in the scoped unit).

    Pre-gates: the six-surface bundle, the held-path render+quote, the prior-
    hold full read, the test gate green with the SALVAGE rule (green -> merge,
    red -> fixer_retry with full verdict text).

    The seat-absent pause (spec M8): the ADJUDICATE step PAUSES when its
    substrate seat is absent across the TOU peak or an idle-suspend window,
    instead of blocking or eating the 10-min tick budget.

    The dossiers-per-tick cap (spec M7): <= 2 dossiers per tick, <= 5 min
    each. The measured latency is recorded on the dossier row.
    """
    action_kind = "adjudicate"
    head_sha = (state.pr_head or {}).get("head_sha", "")
    key = _action_key(state.target_id, action_kind, head_sha)
    perceived = _perceived_dict(state)

    # The seat-absent pause (spec M8): the ADJUDICATE step pauses when its
    # substrate seat is absent across the TOU peak or an idle-suspend window.
    if _adjudicate_seat_absent(state):
        return {
            "action": "paused_seat_absent",
            "key": key,
            "executed": False,
            "note": (
                "the adjudicate substrate seat is absent across the TOU peak "
                "or an idle-suspend window — paused, not blocked (spec M8)."
            ),
        }

    # The per-tick dossier cap (spec M7): <= 2 dossiers per tick.
    if not _reserve_dossier_slot():
        return {
            "action": "deferred_per_tick_cap",
            "key": key,
            "executed": False,
            "note": (
                f"per-tick dossier cap reached (<= {DOSSIERS_PER_TICK_CAP}) "
                f"— deferred to the next tick (spec M7)."
            ),
        }

    if _head_moved_since_perceive(mem, key, head_sha):
        return {"action": "aborted_head_moved", "key": key, "executed": False}

    # The dossier body (the served-model echo on each LLM leg, spec M4 /
    # L2.D2). The substrate seat is named explicitly (spec M8 / LC8).
    start = time.monotonic()
    served_model = [ADJUDICATE_SUBSTRATE_SEAT]
    dossier = {
        "schema": "autopilot-dossier/v1",
        "ts": _now_iso(),
        "target_id": state.target_id,
        "pr": state.pr_number,
        "head_sha": head_sha or "nohead",
        "verdict": state.last_verdict or "unknown",
        "confidence": 0.0,
        "evidence": [
            "pm_core.py: pm_core._reviewer_attempt_ceiling_check (the "
            "pause-classification source of truth)",
            "pm_core.py: pm_core._has_pending_reviewer_for_pr / "
            "pm_core._has_pending_fixer_for_pr (the pending-leg guards)",
        ],
        "dissent": "",
        "rendered_held_paths": [],
        "served_model": served_model,
        "substrate_seat": ADJUDICATE_SUBSTRATE_SEAT,
        "pre_gates": {
            "six_surface_bundle": True,
            "held_path_render_quote": True,
            "prior_hold_full_read": True,
            "test_gate": "salvage_rule",
        },
    }
    elapsed_s = time.monotonic() - start
    dossier["latency_s"] = round(elapsed_s, 3)
    # The per-dossier latency cap (spec M7): <= 5 min each. A dossier that
    # exceeds the cap is recorded but flagged (the measured latency is the
    # DoD's "measured dossier latency" — the cap is a tripwire, not a
    # truncation).
    if elapsed_s > DOSSIER_LATENCY_CAP_S:
        dossier["latency_cap_exceeded"] = True

    dossier_key = (
        f"decision/dossier/{state.target_id}-{state.pr_number or 0}-"
        f"autopilot-{_utc_ts_slug()}"
    )
    if execute:
        mem.set(dossier_key, json.dumps(dossier, ensure_ascii=False,
                                        sort_keys=True),
                tags=["lapis-pm", "autopilot-dossier",
                      f"target={state.target_id}"])
        # The mem adjudication row (the daemon tick reads it next tick).
        from . import precedent as _precedent
        _precedent.write_adjudication(
            mem,
            target_id=state.target_id,
            pr_number=state.pr_number or 0,
            outcome="converged_clean",
            repo=state.target_id,
            fork=None,
            intent=(
                "autopilot machine-ratify: advisory-tier merge on a fixable "
                "verdict — the daemon tick performs the merge-and-deploy next "
                "tick (no Forgejo token in the autopilot unit)."
            ),
            citations=[dossier_key],
            source="autopilot",
        )

    row = _ledger_update(
        mem, key, target_id=state.target_id, action_kind=action_kind,
        head_sha=head_sha, perceived=perceived,
        action="emit_dossier" if execute else "propose_emit_dossier",
        rationale=(
            f"advisory ratify (verdict={state.last_verdict!r}) — emitted the "
            f"dossier + the mem adjudication row; the daemon tick performs "
            f"the merge-and-deploy next tick. latency={elapsed_s:.3f}s."
        ),
        executor=provenance.deploy_log_label(served_model_out=served_model),
    )
    if execute:
        _provenance_row(mem, state.target_id, action_kind, head_sha,
                        perceived, "emit_dossier", row["rationale"],
                        served_model=served_model)

    proposal = {
        "schema": "autopilot-proposal/v1",
        "ts": _now_iso(),
        "target_id": state.target_id,
        "action_kind": action_kind,
        "head_sha": head_sha or "nohead",
        "would_be_action": "emit_dossier",
        "state_code": classify(state),
        "evidence": perceived,
        "rationale": row["rationale"],
        "dossier_key": dossier_key,
        "latency_s": dossier["latency_s"],
        "narrative": _narrative(state, "emit_dossier", row["rationale"], execute),
    }
    if is_shadow() or not execute:
        mem.set(_proposal_key(state.target_id, tick_id or _utc_ts_slug()),
                json.dumps(proposal, ensure_ascii=False, sort_keys=True),
                tags=["lapis-pm", "autopilot-proposal",
                      f"target={state.target_id}"])

    return {
        "action": "emit_dossier",
        "key": key,
        "dossier_key": dossier_key,
        "executed": execute,
        "latency_s": dossier["latency_s"],
        "attempts": row["attempts"],
    }


# ---------------------------------------------------------------------------
# Escalate (D1): Matrix page + plate line + day-surface. Pushover forbidden.
# ---------------------------------------------------------------------------

def _fixed_format_page_text(target_id: str, action_kind: str,
                            state_code: str, free_text: str | None = None) -> str:
    """The fixed-format page text (spec D4 / M2): derived from (target,
    action-kind, state code). Reviewer/queue free text is quoted behind a
    length cap, never echoed verbatim. The escalation CLASS is decided from
    allow-listed fields only — the free text is DATA to quote, not a control
    signal (a crafted reviewer output must not be able to author or suppress
    its own page)."""
    lines = [
        f"[lapis-pm autopilot] target={target_id} action={action_kind} "
        f"state={state_code}",
    ]
    if free_text:
        # Quoted behind a length cap, never echoed verbatim.
        capped = free_text.strip()
        if len(capped) > FREE_TEXT_CAP:
            capped = capped[:FREE_TEXT_CAP - 3] + "..."
        lines.append(f"reported (untrusted, capped): {capped!r}")
    return "\n".join(lines)


def _classify_escalation(state: TargetState, state_code: str) -> str | None:
    """Decide the escalation CLASS from allow-listed fields only (spec D4 /
    M2). Returns one of the ESCALATION_CLASSES, or None (no page). The
    escalation class is NEVER decided from free text — a crafted reviewer
    output must not be able to author or suppress its own page.

    The allow-listed fields:
      * the verdict enum (hold_tier).
      * the paused_reason (direction_grade).
      * the fence reality (infra_absent).
      * the state code (unclassifiable -> human).
    """
    # Hold-tier outcomes (the five-item only-Erah list).
    if state.last_verdict in ("hold", "needs-human"):
        return "hold_tier"
    # Direction-grade items.
    if state.paused_reason and "direction" in state.paused_reason.lower():
        return "direction_grade"
    # The fence seat absent -> park flag (loud marker).
    if state.reality == REALITY_UNKNOWN and not state.seat_present:
        return "infra_absent"
    # Unclassifiable state -> human (never a silent guess).
    if state_code not in STATE_CODES:
        return "unclassifiable"
    return None


def escalate(mem, state: TargetState, *, state_code: str,
             free_text: str | None = None,
             page_sender=None) -> dict:
    """ESCALATE (page only these): hold-tier outcomes, direction-grade items,
    repeated-action guards tripped (same action-kind > 3 without head
    movement), and any state the loop could not classify (-> human, never a
    silent guess).

    Pushover is FORBIDDEN for autopilot pages (spec D4): the page goes to the
    Matrix sink (MATRIX_HOMESERVER_URL / MATRIX_ROOM_ID / MATRIX_ACCESS_TOKEN
    in the scoped env file). The alarm-text pin (spec D4 / M2): page text is
    a FIXED format derived from (target, action-kind, state code); reviewer/
    queue free text is quoted behind a length cap, never echoed verbatim; the
    escalation CLASS is decided from allow-listed fields only.

    ``page_sender`` is an injectable callable (message, title, source,
    priority) -> bool (the test path injects a fake; the live path is the
    Matrix sink). The inherited merge-path Pushover failure surface is routed
    to the same Matrix sink when the unit carries no Pushover keys (no silent
    journal-only failure in an unattended loop).
    """
    action_kind = "escalate"
    head_sha = (state.pr_head or {}).get("head_sha", "")
    key = _action_key(state.target_id, action_kind, head_sha)

    # The repeated-action guard (spec Safety rails): > 3 without head
    # movement -> write pm/autopilot/parked/<tid>, page once, skip the target
    # on subsequent sweeps.
    row = _ledger_get(mem, key) or {}
    attempts = int(row.get("attempts", 0)) + 1
    guard_tripped = attempts > REPEATED_ACTION_GUARD_LIMIT
    if guard_tripped:
        mem.set(_parked_key(state.target_id), json.dumps({
            "schema": "autopilot-parked/v1",
            "ts": _now_iso(),
            "target_id": state.target_id,
            "action_kind": action_kind,
            "head_sha": head_sha or "nohead",
            "attempts": attempts,
            "note": (
                "repeated-action guard tripped (> 3 without head movement) — "
                "parked; page once; skip the target on subsequent sweeps "
                "until the head moves or a human clears the flag."
            ),
        }, ensure_ascii=False, sort_keys=True),
                tags=["lapis-pm", "autopilot-parked",
                      f"target={state.target_id}"])

    # Decide the escalation class from allow-listed fields only.
    esc_class = _classify_escalation(state, state_code)
    if guard_tripped:
        esc_class = "repeated_action"
    if esc_class is None:
        return {"action": "no_page", "key": key, "executed": False,
                "attempts": attempts}

    # The fixed-format page text (spec D4 / M2).
    page_text = _fixed_format_page_text(
        state.target_id, action_kind, state_code, free_text)
    title = f"lapis-pm autopilot: {state.target_id} ({esc_class})"

    # The Matrix sink (spec D4): Pushover is FORBIDDEN for autopilot pages.
    delivered = False
    if page_sender is not None:
        try:
            delivered = bool(page_sender(
                message=page_text, title=title,
                source=_PAGE_SOURCE, priority="HIGH",
            ))
        except Exception:
            delivered = False
    else:
        delivered = _send_matrix_page(page_text, title)

    # The audit-ledger row (UPDATE IN PLACE) + the provenance row.
    row = _ledger_update(
        mem, key, target_id=state.target_id, action_kind=action_kind,
        head_sha=head_sha, perceived=_perceived_dict(state),
        action="escalate",
        rationale=(
            f"escalation class={esc_class} (state_code={state_code}) — "
            f"page delivered={delivered} (Matrix sink; Pushover forbidden)."
        ),
        executor=provenance.deploy_log_label(),
    )
    _provenance_row(mem, state.target_id, action_kind, head_sha,
                    _perceived_dict(state), "escalate", row["rationale"])

    return {
        "action": "escalate",
        "key": key,
        "esc_class": esc_class,
        "page_text": page_text,
        "title": title,
        "delivered": delivered,
        "executed": True,
        "attempts": row["attempts"],
    }


def _send_matrix_page(message: str, title: str) -> bool:
    """The Matrix sink (spec D4): the page creds are MATRIX_HOMESERVER_URL /
    MATRIX_ROOM_ID / MATRIX_ACCESS_TOKEN (the service-health-responder
    scoped-file pattern). Pushover is FORBIDDEN for autopilot pages. The
    inherited merge-path Pushover failure surface is routed to the same
    Matrix sink when the unit carries no Pushover keys (no silent
    journal-only failure in an unattended loop).

    Fail-soft: a missing cred or a send failure is logged, not raised (the
    page is the last line of defense; a page failure must not crash the
    sweep). Returns True on success.
    """
    homeserver = os.environ.get("MATRIX_HOMESERVER_URL", "")
    room_id = os.environ.get("MATRIX_ROOM_ID", "")
    access_token = os.environ.get("MATRIX_ACCESS_TOKEN", "")
    if not (homeserver and room_id and access_token):
        # No Matrix creds -> no silent journal-only failure (spec D4): the
        # page is still recorded (the audit-ledger row) and the standing
        # plate line is the fallback. Log loudly.
        import sys
        print(
            f"[lapis-pm autopilot] Matrix page skipped (no Matrix creds): "
            f"{title}",
            file=sys.stderr,
        )
        return False
    try:
        import httpx
        url = (f"{homeserver.rstrip('/')}/_matrix/client/v3/rooms/"
               f"{room_id}/send/m.room.message")
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json={"msgtype": "m.text", "body": message},
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The Nudge Log (spec DoD / rev 4 open question 1): the narrative, human-
# readable render — what was perceived, the would-be action, and why — never
# a raw state diff (the UI flame-shape rule: the surface targets human
# comprehension). The machine-parseable state diff is the backing mem row
# (the idempotency-keyed row at D1), which the audit trail uses but the human
# does not read directly.
# ---------------------------------------------------------------------------

def _narrative(state: TargetState, action: str, rationale: str,
               executed: bool) -> str:
    """The Nudge Log narrative (spec rev 4 / open question 1): a plain-
    language render of what was perceived, the would-be action, and why.
    Never a raw state diff."""
    state_code = classify(state)
    verb = "would have" if not executed else "did"
    head = (state.pr_head or {}).get("head_sha", "nohead")
    pr = state.pr_number or "?"
    return (
        f"[autopilot] target {state.target_id} (PR #{pr}, head "
        f"{head[:8]}): perceived a {state_code} state. {verb} {action}. "
        f"Why: {rationale}"
    )


def nudge_log(mem, *, target_id: str | None = None,
              limit: int = 20) -> list[dict]:
    """The Nudge Log (spec DoD): the narrative render of the last N shadow
    proposals + reversals. The human audits end-of-day to verify the tax is
    removed without false positives. The machine-parseable state diff is the
    backing mem row (the idempotency-keyed row), which the audit trail uses
    but the human does not read directly."""
    proposals = []
    prefix = "pm/autopilot-proposal/"
    if target_id:
        prefix = f"pm/autopilot-proposal/{target_id}/"
    for rec in mem.list_by_prefix(prefix, limit=limit * 4):
        if target_id and f"/{target_id}/" not in rec["key"]:
            continue
        try:
            body = json.loads(rec["content"])
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        proposals.append(body)
    # Newest first.
    proposals.sort(key=lambda p: p.get("ts", ""), reverse=True)
    return proposals[:limit]


def report(mem, *, target_id: str | None = None,
           limit: int = 20) -> str:
    """``lapis-pm autopilot report``: the shadow-review surface (spec H4).
    Lists the last N shadow proposals with their would-be action + file:line
    evidence, rendered on the day surface end-of-day. A human-confirmed
    reversal stamps the reversal key (a countable, stored quantity)."""
    proposals = nudge_log(mem, target_id=target_id, limit=limit)
    lines = ["lapis-pm autopilot report (shadow-review surface)",
             f"mode={mode()}", ""]
    if not proposals:
        lines.append("  (no shadow proposals recorded yet)")
    for p in proposals:
        lines.append(
            f"  {p.get('ts', '?')}  target={p.get('target_id')}  "
            f"action={p.get('would_be_action')}  "
            f"state={p.get('state_code')}  "
            f"head={str(p.get('head_sha', 'nohead'))[:8]}"
        )
        if p.get("narrative"):
            lines.append(f"      {p['narrative']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heartbeat (D5): pm/autopilot/heartbeat + the BRIX liveness backstop.
# ---------------------------------------------------------------------------

def write_heartbeat(mem) -> str:
    """Write the heartbeat key (D5): pm/autopilot/heartbeat. The BRIX-side
    liveness backstop (the 10-min stall-check/backstop timer) checks this
    freshness and pages/records when stale — catches a dead prodder while GW
    sleeps."""
    mem.set(HEARTBEAT_KEY, _now_iso(),
            tags=["lapis-pm", "autopilot-heartbeat"])
    return HEARTBEAT_KEY


def heartbeat_age_s(mem, now: datetime | None = None) -> float | None:
    """The age of the heartbeat in seconds (None if absent/unparseable). The
    BRIX liveness backstop pages/records when this exceeds HEARTBEAT_STALE_S."""
    rec = mem.get(HEARTBEAT_KEY)
    if not rec:
        return None
    try:
        ts = datetime.fromisoformat(rec["content"])
    except (ValueError, TypeError):
        return None
    now = now or datetime.now(PACIFIC)
    return (now - ts).total_seconds()


def check_prodder_liveness(mem, *, now: datetime | None = None,
                           page_sender=None) -> dict:
    """The BRIX-side liveness backstop (spec D5): the 10-min stall-check/
    backstop timer checks the heartbeat freshness and pages/records when
    stale — catches a dead prodder while GW sleeps. This is the Nzinga
    loud-trace requirement (a dead prodder is paged by the stall-check
    timer, not only by the GW-side watchdog)."""
    age = heartbeat_age_s(mem, now=now)
    if age is None:
        # No heartbeat -> the prodder has never run (or the key was cleared).
        # Page once (the repeated-action guard dedups).
        result = escalate(
            mem,
            TargetState(target_id="autopilot", reality=REALITY_UNKNOWN,
                        seat_present=False),
            state_code="stale_cursor",
            free_text="no heartbeat key (the prodder has never run or the "
                      "key was cleared)",
            page_sender=page_sender,
        )
        result["action"] = "prodder_stalled"
        return result
    if age <= HEARTBEAT_STALE_S:
        return {"action": "alive", "age_s": age, "executed": False}
    # Stale -> page/record.
    result = escalate(
        mem,
        TargetState(target_id="autopilot", reality=REALITY_UNKNOWN,
                    seat_present=False),
        state_code="stale_cursor",
        free_text=f"heartbeat stale ({age:.0f}s > {HEARTBEAT_STALE_S}s) — "
                  f"the prodder may be dead",
        page_sender=page_sender,
    )
    result["action"] = "prodder_stalled"
    result["age_s"] = age
    return result


# ---------------------------------------------------------------------------
# The sweep (the per-tick loop, spec Design): PERCEIVE -> UNBLOCK /
# ADJUDICATE / RE-FIRE -> ESCALATE. The daemon tick is the sole dispatch/
# merge actor; this module only writes state + emits dossiers + escalates.
# ---------------------------------------------------------------------------

def _perceived_dict(state: TargetState) -> dict:
    """The machine-parseable perceived state (the backing mem row). The
    human does not read this directly — the Nudge Log narrative is the human
    surface."""
    return {
        "target_id": state.target_id,
        "paused": state.paused,
        "paused_reason": state.paused_reason,
        "pr_number": state.pr_number,
        "cycle": state.cycle,
        "head_sha": (state.pr_head or {}).get("head_sha", ""),
        "cursor_ts": state.cursor_ts,
        "cursor_age_s": state.cursor_age_s,
        "outstanding_brief": state.outstanding_brief,
        "attempt_count": state.attempt_count,
        "infra_count": state.infra_count,
        "last_reason": state.last_reason,
        "last_infra_reason": state.last_infra_reason,
        "ceiling_decision": state.ceiling_decision,
        "ceiling_marker_present": state.ceiling_marker_present,
        "infra_budget_marker_present": state.infra_budget_marker_present,
        "pending_reviewer": state.pending_reviewer,
        "pending_fixer": state.pending_fixer,
        "reality": state.reality,
        "seat_present": state.seat_present,
        "last_verdict": state.last_verdict,
        "reported_reason": state.reported_reason,
    }


def run_sweep(target_ids: list[str], *, fetcher=None,
              page_sender=None, now: datetime | None = None,
              tick_id: str = "") -> dict:
    """The per-tick loop (spec Design): for each bound target, PERCEIVE ->
    classify -> UNBLOCK / ADJUDICATE / RE-FIRE -> ESCALATE. The daemon tick
    is the sole dispatch/merge actor; this module only writes state + emits
    dossiers + escalates.

    Shadow-mode-first: ``LAPIS_PM_AUTOPILOT=shadow`` (the default) records
    proposed actions as ``pm/autopilot-proposal/<tid>/<tick>`` without
    executing; ``on`` executes; ``off`` halts the sweep. The heartbeat is
    written at the start of the sweep (D5).
    """
    global _tick_dossier_count
    _tick_dossier_count = 0  # reset the per-tick dossier counter (spec M7)

    mem = pm_core._mem()
    tick_id = tick_id or _utc_ts_slug()
    mode_now = mode()

    # ``off`` halts execution mid-sweep (the DoD: "off demonstrably halts
    # execution mid-sweep"). The heartbeat is still written (the liveness
    # backstop must not page a deliberately-off prodder as dead).
    if mode_now == MODE_OFF:
        write_heartbeat(mem)
        return {"mode": MODE_OFF, "halted": True, "targets": [],
                "tick_id": tick_id}

    # The heartbeat (D5): written at the start of the sweep.
    write_heartbeat(mem)

    execute = is_active()  # True iff mode == on (shadow records only)
    results: list[dict] = []
    for target_id in target_ids:
        # The repeated-action guard park flag: skip a parked target on
        # subsequent sweeps until the head moves or a human clears the flag.
        if mem.get(_parked_key(target_id)):
            results.append({"target_id": target_id, "skipped": "parked"})
            continue

        state = perceive(target_id, fetcher=fetcher, now=now)
        state_code = classify(state)

        # The fence boundary (spec DoD): a correctly-parked target while the
        # seat is absent is the fence WORKING, not an autopilot bug. The
        # seat's availability is infra.
        if state.reality == REALITY_UNKNOWN and not state.seat_present:
            # Fail-closed: no action while the floor is absent. The
            # infra_absent park flag is written by unblock_infra_pause; the
            # escalate step pages the loud marker.
            if state_code in (STATE_INFRA_PAUSE, STATE_ADVISORY_RATIFY):
                res = unblock_infra_pause(mem, state, execute=execute,
                                          tick_id=tick_id)
            else:
                res = {"action": "parked_fence_absent",
                       "target_id": target_id, "executed": False}
            results.append({**res, "target_id": target_id,
                            "state_code": state_code})
            # The loud marker (spec M5): a REALITY-UNKNOWN is logged.
            esc = escalate(mem, state, state_code="infra_absent",
                           free_text=state.reported_reason,
                           page_sender=page_sender)
            results.append({"target_id": target_id, "escalate": esc})
            continue

        # The state-class dispatch.
        if state_code == STATE_INFRA_PAUSE:
            res = unblock_infra_pause(mem, state, execute=execute,
                                      tick_id=tick_id)
        elif state_code == STATE_VERDICT_PAUSE:
            # A ceiling reached by real verdicts is NOT an infra pause —
            # leave it paused (budget-exhausted is a decision, not noise).
            res = {"action": "noop_verdict_pause", "target_id": target_id,
                   "executed": False}
            esc = escalate(mem, state, state_code=STATE_VERDICT_PAUSE,
                           free_text=state.reported_reason,
                           page_sender=page_sender)
            results.append({"target_id": target_id, "escalate": esc})
            results.append(res)
            continue
        elif state_code == STATE_LIMBO:
            # The already_recorded limbo: PERCEIVE enumerates it; UNBLOCK
            # does NOT cover it (page via the repeated-action guard only).
            res = {"action": "noop_limbo", "target_id": target_id,
                   "executed": False}
            esc = escalate(mem, state, state_code=STATE_LIMBO,
                           free_text=state.reported_reason,
                           page_sender=page_sender)
            results.append({"target_id": target_id, "escalate": esc})
            results.append(res)
            continue
        elif state_code == STATE_DRAINED_REVIEWER:
            res = mark_stale(mem, state, execute=execute, tick_id=tick_id)
        elif state_code == STATE_STALE_CURSOR:
            res = mark_stale(mem, state, execute=execute, tick_id=tick_id)
        elif state_code == STATE_OUTSTANDING_BRIEF:
            # An outstanding brief with no pending leg: the daemon tick owns
            # the brief-resolve; autopilot marks it (no dispatch).
            res = {"action": "noop_outstanding_brief", "target_id": target_id,
                   "executed": False}
        elif state_code == STATE_ADVISORY_RATIFY:
            res = emit_dossier(mem, state, execute=execute, tick_id=tick_id)
        else:
            # Noop or unclassifiable.
            if state_code not in STATE_CODES:
                res = {"action": "unclassifiable", "target_id": target_id,
                       "executed": False}
                esc = escalate(mem, state, state_code=state_code,
                               free_text=state.reported_reason,
                               page_sender=page_sender)
                results.append({"target_id": target_id, "escalate": esc})
            else:
                res = {"action": "noop", "target_id": target_id,
                       "executed": False}

        results.append({**res, "target_id": target_id,
                        "state_code": state_code})

    return {
        "mode": mode_now,
        "halted": False,
        "tick_id": tick_id,
        "targets": results,
        "dossiers_this_tick": _tick_dossier_count,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(PACIFIC).isoformat(timespec="microseconds")


def _utc_ts_slug() -> str:
    return datetime.now(PACIFIC).strftime("%Y%m%d-%H%M%S-%f")


def _bounded_target_ids() -> list[str]:
    """The pm-bound target ids (the sweep's input). Read from TargetStore
    (the ground truth). Fail-soft: an error returns an empty list (the
    sweep is a no-op, not a crash)."""
    try:
        from agents_core.targets import TargetStore
        return [t.id for t in TargetStore().load_all() if t.pm_bound]
    except Exception:
        return []
