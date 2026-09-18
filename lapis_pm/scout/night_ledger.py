"""Scout night-run observability lane — ledger, queue bootstrap, morning digest.

Implements the ``genesis-constitution-heartbeat-v0`` spec (rev 3). The scout
night lane has fired cleanly for four+ nights yet left zero observable output:
no ``pm/night/run/*`` mem rows, no ``/data/queue/scout_candidates.jsonl``, no
morning digest. This module is the *observability* fix — a ledger the night can
read, an explicit blank-page record when nothing ran, a direction-grade morning
digest served to a readable surface, and a loud-but-not-Pushover doorbell.

Verified-absence assertion (D1, mandatory):
    ``git grep -n "pm/night/run" origin/main -- '*.py'`` returns NO matches.
    The four-night silence is explained by the ABSENCE of any run-state writer,
    not by an uninvestigated lane-logic bug. This module is that writer. If a
    second writer is found during implementation, do not invent a parallel
    ledger — stop and say so.

Advisory boundary:
    - Shared-store writes are limited to ``pm/night/run/*`` mem keys and
      ``/srv/lapis/morning/`` files.
    - This lane NEVER sends Pushover (numb channel, ratified
      decision/loudness-doctrine-no-repush-2026-09-17). The only notify
      transport this lane may use is the configured non-Pushover transport
      (preferred: Matrix); when none is configured it falls back to
      file+ledger (the digest file is the source of truth, the notify is the
      doorbell).

run_id:
    The ledger key's own ``<utc-ts>`` suffix — deterministic, atomic with the
    write, no separate UUID generator. A ghost row (a row without a valid
    identifier) is impossible by construction because the identifier IS the key.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agents_core.room_paths import room_root

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Closed lane-status set (D2)
# ---------------------------------------------------------------------------
# Every value MUST carry a reason. Do not emit dispatched/ACTIVE when no task
# ran — that is hallucinated activity. A blank page is recorded as EXPLICIT
# evidence, never inferred activity.
LANE_STATUS = frozenset({"dispatched", "skipped", "noop", "blank", "failed"})

# The one closed status used when a lane produced zero candidates.
BLANK = "blank"
BLANK_REASON = "no_candidates"

# The single mem key prefix this lane owns (Invariants).
NIGHT_RUN_MEM_PREFIX = "pm/night/run/"

# Default queue path (D2). Overridable per-call for tests.
DEFAULT_QUEUE_PATH = Path("/data/queue/scout_candidates.jsonl")

# Parked decisions older than this re-surface in the morning digest (D3).
RESURFACE_HOURS = 48.0

# Morning digest target directory (D3): /srv/lapis/morning/YYYY-MM-DD.md
MORNING_DIR_NAME = "morning"


# ---------------------------------------------------------------------------
# Lane record
# ---------------------------------------------------------------------------

@dataclass
class LaneRecord:
    """One lane's outcome for a single night run.

    ``status`` MUST be a member of :data:`LANE_STATUS` and ``reason`` MUST be a
    non-empty string. ``candidates`` is the per-lane candidate count (0 for a
    blank page).
    """
    lane: str
    status: str
    reason: str
    candidates: int = 0

    def __post_init__(self) -> None:
        if self.status not in LANE_STATUS:
            raise ValueError(
                f"lane {self.lane!r} status {self.status!r} not in closed set "
                f"{sorted(LANE_STATUS)}"
            )
        if not self.reason or not str(self.reason).strip():
            raise ValueError(f"lane {self.lane!r} must carry a non-empty reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "status": self.status,
            "reason": self.reason,
            "candidates": self.candidates,
        }


# ---------------------------------------------------------------------------
# run_id helpers
# ---------------------------------------------------------------------------

# UTC timestamp suffix used both as the log_root run_tag and as the ledger key
# suffix. Deterministic + atomic with the write: the identifier IS the key.
_RUN_TS_FMT = "%Y%m%dT%H%M%SZ"


def utc_run_ts(now: datetime | None = None) -> str:
    """Return the UTC timestamp suffix for a ledger key / run_tag.

    e.g. ``20260918T053000Z``. This is the ``run_id`` — no separate UUID
    generator, so a ghost row is impossible by construction.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).strftime(_RUN_TS_FMT)


def ledger_key(run_id: str) -> str:
    """Return the full mem key for a run: ``pm/night/run/<utc-ts>``."""
    return f"{NIGHT_RUN_MEM_PREFIX}{run_id}"


def run_id_from_key(key: str) -> str:
    """Recover the run_id (the ``<utc-ts>`` suffix) from a ledger key.

    The run_id is the key's own suffix — atomic with the write.
    """
    if key.startswith(NIGHT_RUN_MEM_PREFIX):
        return key[len(NIGHT_RUN_MEM_PREFIX):]
    return key


# ---------------------------------------------------------------------------
# D2 — queue bootstrap
# ---------------------------------------------------------------------------

def ensure_queue_file(path: Path | None = None) -> Path:
    """Assert the scout candidate queue file exists, creating it if missing.

    The producer asserts ``/data/queue/scout_candidates.jsonl``. Returns the
    path of the (now-existing) file. Creates parent directories.
    """
    qpath = Path(path) if path is not None else DEFAULT_QUEUE_PATH
    qpath.parent.mkdir(parents=True, exist_ok=True)
    if not qpath.exists():
        qpath.touch()
    return qpath


# ---------------------------------------------------------------------------
# D1 — run ledger
# ---------------------------------------------------------------------------

@dataclass
class RunLedger:
    """A single night run's ledger state, written as mem rows.

    ``run_open`` is written at run start; ``run_summary`` at completion. Both
    share the same key ``pm/night/run/<utc-ts>`` (upserted), so the ledger row
    is a single, atomically-identified record whose ``run_id`` is the key's own
    ``<utc-ts>`` suffix.
    """
    run_id: str
    started_at: str
    lanes: list[LaneRecord] = field(default_factory=list)
    phase: str = "open"
    wall_seconds: float | None = None
    aborted: bool = False
    abort_reason: str = ""
    digest_path: str | None = None
    notify: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    @classmethod
    def open(
        cls,
        run_id: str,
        *,
        lanes: list[LaneRecord] | None = None,
        now: datetime | None = None,
    ) -> "RunLedger":
        """Build the run-open record for a freshly-started run."""
        now = now or datetime.now(timezone.utc)
        return cls(
            run_id=run_id,
            started_at=now.isoformat(),
            lanes=list(lanes or []),
            phase="open",
        )

    def summarize(
        self,
        *,
        wall_seconds: float | None = None,
        lanes: list[LaneRecord] | None = None,
        aborted: bool = False,
        abort_reason: str = "",
        now: datetime | None = None,
    ) -> None:
        """Transition to run-summary, recording wall time + final lane states."""
        if lanes is not None:
            self.lanes = list(lanes)
        self.phase = "summary"
        self.wall_seconds = wall_seconds
        self.aborted = aborted
        self.abort_reason = abort_reason
        _ = now  # retained for signature symmetry; started_at already fixed

    # -- serialization ------------------------------------------------------

    def _lane_status(self) -> dict[str, str]:
        """Map lane -> status for the ledger row (closed set enforced)."""
        return {r.lane: r.status for r in self.lanes}

    def _candidate_counts(self) -> dict[str, int]:
        return {r.lane: r.candidates for r in self.lanes}

    def to_open_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": "open",
            "started_at": self.started_at,
            "lane_status": self._lane_status(),
            "lane_reason": {r.lane: r.reason for r in self.lanes},
            "candidate_counts": self._candidate_counts(),
        }

    def to_summary_payload(self) -> dict[str, Any]:
        payload = self.to_open_payload()
        payload.update(
            {
                "phase": "summary",
                "wall_seconds": self.wall_seconds,
                "aborted": self.aborted,
                "abort_reason": self.abort_reason,
                "digest_path": self.digest_path,
                "notify": self.notify,
            }
        )
        if self.extra:
            payload["extra"] = self.extra
        return payload

    def to_payload(self) -> dict[str, Any]:
        return (
            self.to_summary_payload() if self.phase == "summary"
            else self.to_open_payload()
        )


def _default_mem_store():
    """Return a mem store for writing ledger rows.

    On the mem master (BRIX) this is a direct local MemoryStore. Off-master
    callers would proxy, but the scout night lane runs on BRIX (the master), so
    a direct store is correct here. Tests inject a store via ``mem_store=``.
    """
    from agents_core.mem import MemoryStore

    return MemoryStore()


def emit_run_ledger(
    ledger: RunLedger,
    *,
    mem_store: Any | None = None,
) -> str:
    """Write a run-ledger row (run-open or run-summary) to mem.

    Writes exactly one mem row at ``pm/night/run/<utc-ts>`` (tag
    ``pm-night-run``). The key is the run's own ``<utc-ts>`` suffix, so the
    row and its identifier are atomic. Returns the mem key written.
    """
    store = mem_store if mem_store is not None else _default_mem_store()
    key = ledger_key(ledger.run_id)
    payload = ledger.to_payload()
    store.set(
        key,
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        tags=["pm-night-run"],
    )
    return key


def read_run_ledger(
    run_id: str,
    *,
    mem_store: Any | None = None,
) -> dict[str, Any] | None:
    """Read a run-ledger row by run_id. Returns the parsed payload or None."""
    store = mem_store if mem_store is not None else _default_mem_store()
    entry = store.get(ledger_key(run_id))
    if entry is None:
        return None
    try:
        return json.loads(entry["content"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def read_latest_run_ledger(
    *,
    mem_store: Any | None = None,
) -> dict[str, Any] | None:
    """Return the most recent run-summary ledger row (by key order).

    Ledger keys are ``pm/night/run/<utc-ts>``; the latest run has the
    lexicographically-greatest ``<utc-ts>`` suffix.
    """
    store = mem_store if mem_store is not None else _default_mem_store()
    rows = store.list_by_prefix(NIGHT_RUN_MEM_PREFIX, limit=50)
    if not rows:
        return None
    latest = max(rows, key=lambda r: r["key"])
    try:
        return json.loads(latest["content"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# D2 — blank-page honesty
# ---------------------------------------------------------------------------

def blank_page_record(lane: str) -> LaneRecord:
    """Return the explicit blank-page record for a lane that produced nothing.

    ``lane_status=blank`` and ``reason=no_candidates`` — silence recorded as
    EXPLICIT evidence, never inferred activity.
    """
    return LaneRecord(lane=lane, status=BLANK, reason=BLANK_REASON, candidates=0)


def lane_records_for_run(
    lanes: list[str],
    *,
    dispatched: set[str] | None = None,
    skipped: dict[str, str] | None = None,
    noop: dict[str, str] | None = None,
    failed: dict[str, str] | None = None,
    candidate_counts: dict[str, int] | None = None,
) -> list[LaneRecord]:
    """Build the closed-set lane records for a run.

    Each lane is classified into exactly one of the closed statuses:
    ``dispatched`` / ``skipped`` / ``noop`` / ``failed`` / ``blank``. A lane
    with zero candidates that did not dispatch is recorded as a blank page.
    Every record carries a reason.
    """
    dispatched = dispatched or set()
    skipped = skipped or {}
    noop = noop or {}
    failed = failed or {}
    candidate_counts = candidate_counts or {}

    records: list[LaneRecord] = []
    for lane in lanes:
        if lane in dispatched:
            records.append(
                LaneRecord(
                    lane=lane,
                    status="dispatched",
                    reason="lane ran",
                    candidates=candidate_counts.get(lane, 0),
                )
            )
        elif lane in failed:
            records.append(
                LaneRecord(lane=lane, status="failed", reason=failed[lane])
            )
        elif lane in skipped:
            records.append(
                LaneRecord(lane=lane, status="skipped", reason=skipped[lane])
            )
        elif lane in noop:
            records.append(
                LaneRecord(lane=lane, status="noop", reason=noop[lane])
            )
        else:
            # Zero candidates, nothing ran -> explicit blank page.
            records.append(blank_page_record(lane))
    return records


# ---------------------------------------------------------------------------
# D3 — morning digest (NEW composer, NOT a wrapper around digest())
# ---------------------------------------------------------------------------
#
# This is a direction-grade Markdown composer. It reads the run-ledger +
# parked decisions and emits a letter Erah can read and trust. It MUST NOT
# reuse or wrap lapis_pm/scout/digest.py::digest() — that function is a
# failure-mode mapper whose output surface is a LapisToolReturn JSON envelope
# at /srv/lapis/scout/maps/<spec-id>.yaml, categorically incompatible with the
# Markdown morning contract.
#
# Register (direction-grade only): mechanism, purpose, why broken, what the
# decision turns on. No code, no identifiers. The cheap-ask rule is enforced:
# an item that cannot be made a cheap ask stays in deliberation (misrouted,
# not hard). The 'why broken' field MAY carry a one-line technical root-cause
# summary when the root cause is a code defect.


# Identifier-ish tokens that must not leak into the direction-grade register.
# Code identifiers (dotted module paths, snake_case function names, file
# extensions) are the cheap-ask guard's tripwire.
_IDENT_TOKEN_RE = re.compile(
    r"(?:"
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*"  # dotted path
    r"|[A-Za-z_][A-Za-z0-9_]*\.(?:py|yaml|json|toml|sh|md|txt)"  # file
    r"|[a-z][a-z0-9]+(?:_[a-z0-9]+)+"  # snake_case identifier (>=2 parts)
    r")"
)


@dataclass
class DigestItem:
    """One direction-grade decision item for the morning digest.

    ``why_broken`` is a one-line narrative. When the root cause is a code
    defect it MAY be a one-line technical root-cause summary (spec rev 3).
    """
    title: str
    mechanism: str
    purpose: str
    why_broken: str
    turns_on: str
    # True when this item can be made a cheap ask (the default). False means
    # it stays in deliberation (misrouted, not hard).
    cheap_ask: bool = True
    # Optional re-surface metadata (D3): a parked decision older than 48h.
    days_waiting: float | None = None
    recommendation: str | None = None

    def fields(self) -> dict[str, str]:
        return {
            "title": self.title,
            "mechanism": self.mechanism,
            "purpose": self.purpose,
            "why_broken": self.why_broken,
            "turns_on": self.turns_on,
        }


def _contains_identifier(text: str) -> bool:
    """Return True if ``text`` contains a code-identifier-looking token."""
    return bool(_IDENT_TOKEN_RE.search(text))


def cheap_ask_guard(item: DigestItem) -> bool:
    """Return True if ``item`` passes the direction-grade cheap-ask guard.

    An item is misrouted (rejected) if any of its direction-grade fields
    carries a code identifier — the register is mechanism/purpose/why-broken/
    what-it-turns-on, not code. A one-line technical root-cause summary is
    allowed in ``why_broken`` ONLY when it stays free of identifiers (it names
    the defect in words, not in dotted paths or function names).
    """
    for value in item.fields().values():
        if _contains_identifier(value):
            return False
    return True


def resurface_parked(
    parked: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    hours: float = RESURFACE_HOURS,
) -> list[dict[str, Any]]:
    """Re-surface parked decisions older than ``hours`` with days-waiting cost.

    ``parked`` is a list of dicts, each with at least ``spec_id`` and
    ``parked_at`` (ISO-8601). Returns the subset older than ``hours`` with a
    ``days_waiting`` float and a one-line ``recommendation``.
    """
    now = now or datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for p in parked:
        parked_at = p.get("parked_at")
        if not parked_at:
            continue
        try:
            ts = datetime.fromisoformat(str(parked_at))
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_hours = (now - ts).total_seconds() / 3600.0
        if age_hours < hours:
            continue
        days_waiting = age_hours / 24.0
        spec_id = p.get("spec_id", "unknown")
        reason = p.get("reason", "")
        recommendation = (
            f"Decide {spec_id}: it has waited {days_waiting:.1f} days "
            f"({reason or 'reason not recorded'}); a parked decision must not "
            f"wait on Erah silently."
        )
        out.append(
            {
                "spec_id": spec_id,
                "reason": reason,
                "days_waiting": days_waiting,
                "recommendation": recommendation,
            }
        )
    return out


def _digest_item_from_resurface(rs: dict[str, Any]) -> DigestItem:
    """Build a direction-grade item from a re-surfaced parked decision.

    The spec_id is a parked-decision identifier, so it is folded into the
    narrative in words (the register is code-free).
    """
    days = rs.get("days_waiting", 0.0)
    return DigestItem(
        title="A parked decision is waiting on you",
        mechanism="The night parked a decision and it has not been decided.",
        purpose="Nothing may wait on you silently — every parked decision re-surfaces until decided.",
        why_broken=(
            f"It has waited {days:.1f} days with no decision; silence is the "
            f"default the heading rule forbids."
        ),
        turns_on="Whether the parked decision should be decided now or explicitly parked again.",
        cheap_ask=True,
        days_waiting=days,
        recommendation=rs.get("recommendation"),
    )


def _render_item_md(item: DigestItem) -> str:
    lines = [f"### {item.title}"]
    lines.append(f"- **Mechanism:** {item.mechanism}")
    lines.append(f"- **Purpose:** {item.purpose}")
    lines.append(f"- **Why broken:** {item.why_broken}")
    lines.append(f"- **What the decision turns on:** {item.turns_on}")
    if item.days_waiting is not None:
        lines.append(f"- **Days waiting:** {item.days_waiting:.1f}")
    if item.recommendation:
        lines.append(f"- **Recommendation:** {item.recommendation}")
    if not item.cheap_ask:
        lines.append("- **Routing:** stays in deliberation (cannot be a cheap ask).")
    return "\n".join(lines)


def compose_morning_digest(
    *,
    ledger: dict[str, Any] | None = None,
    resurfaced: list[dict[str, Any]] | None = None,
    date_str: str | None = None,
) -> str:
    """Compose the direction-grade Markdown morning digest.

    ``ledger`` is a run-summary payload (from :func:`read_run_ledger` /
    :func:`read_latest_run_ledger`). ``resurfaced`` is the output of
    :func:`resurface_parked`. The digest is a letter, not a balance sheet — it
    carries narrative context on top of the cost figures.
    """
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ledger = ledger or {}

    run_id = ledger.get("run_id", "unknown")
    phase = ledger.get("phase", "summary")
    wall = ledger.get("wall_seconds")
    wall_str = f"{wall:.0f}s" if isinstance(wall, (int, float)) else "n/a"
    aborted = ledger.get("aborted", False)
    abort_reason = ledger.get("abort_reason", "")

    lane_status = ledger.get("lane_status", {}) or {}
    candidate_counts = ledger.get("candidate_counts", {}) or {}

    # Build the narrative lane lines.
    lane_lines: list[str] = []
    for lane in sorted(lane_status):
        status = lane_status[lane]
        cands = candidate_counts.get(lane, 0)
        if status == BLANK:
            lane_lines.append(
                f"- **{lane}:** blank page — no candidates; silence recorded "
                f"explicitly, not inferred as activity."
            )
        elif status == "dispatched":
            lane_lines.append(
                f"- **{lane}:** dispatched — {cands} candidate(s) produced."
            )
        else:
            lane_lines.append(f"- **{lane}:** {status}.")

    # Assemble the letter.
    md: list[str] = []
    md.append(f"# Morning — {date_str}")
    md.append("")
    md.append(
        "Good morning. This is the night's account: what ran, what stayed "
        "silent, and what is waiting on you. It is a letter, not a balance "
        "sheet — read it as the machinery telling you what it wants to do."
    )
    md.append("")

    md.append("## The night at a glance")
    md.append("")
    md.append(f"- **Run:** {run_id} ({phase})")
    md.append(f"- **Wall time:** {wall_str}")
    if aborted:
        md.append(f"- **Aborted:** {abort_reason or 'yes'}")
    if lane_lines:
        md.append("- **Lanes:**")
        md.extend(lane_lines)
    else:
        md.append("- **Lanes:** none recorded.")
    md.append("")

    # Decision items: re-surfaced parked decisions first, then any explicit
    # items the caller may pass via resurfaced (they are already direction-grade
    # source dicts).
    items: list[DigestItem] = []
    for rs in (resurfaced or []):
        items.append(_digest_item_from_resurface(rs))

    if items:
        md.append("## Decisions waiting on you")
        md.append("")
        for item in items:
            md.append(_render_item_md(item))
            md.append("")
    else:
        md.append("## Decisions waiting on you")
        md.append("")
        md.append("Nothing is parked and waiting. The morning is clear.")
        md.append("")

    md.append(
        "_Every parked decision re-surfaces until it is decided — nothing may "
        "wait on you silently._"
    )
    md.append("")
    return "\n".join(md)


def emit_morning_digest(
    *,
    ledger: dict[str, Any] | None = None,
    resurfaced: list[dict[str, Any]] | None = None,
    date_str: str | None = None,
    morning_dir: Path | None = None,
) -> Path:
    """Compose the morning digest and write it to ``/srv/lapis/morning/<date>.md``.

    Returns the written path. ``morning_dir`` overrides the default
    ``<room_root>/morning`` (tests use a temp dir).
    """
    if morning_dir is None:
        morning_dir = room_root() / MORNING_DIR_NAME
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    morning_dir.mkdir(parents=True, exist_ok=True)
    out_path = morning_dir / f"{date_str}.md"
    out_path.write_text(compose_morning_digest(
        ledger=ledger,
        resurfaced=resurfaced,
        date_str=date_str,
    ))
    return out_path


# ---------------------------------------------------------------------------
# D4 — loud, non-Pushover delivery
# ---------------------------------------------------------------------------
#
# This lane NEVER sends Pushover. The preferred transport is Matrix (per
# decision/loudness-doctrine-no-repush-2026-09-17); Matrix provisioning is OUT
# of scope for this PR. When no loud transport is configured, the fallback is
# file+ledger: the digest file is the source of truth, the ledger row records
# the digest path.

# The one transport this lane is forbidden from using.
PUSHOVER_TRANSPORT = "pushover"


def _configured_loud_transport() -> str | None:
    """Return the configured non-Pushover loud transport, or None.

    Reads ``LAPIS_PM_LOUD_TRANSPORT`` (preferred: ``matrix``). Pushover is
    never returned — this lane is numb-channel-exempt.
    """
    raw = os.environ.get("LAPIS_PM_LOUD_TRANSPORT", "").strip().lower()
    if not raw or raw == PUSHOVER_TRANSPORT:
        return None
    return raw


def deliver_morning_digest(
    digest_path: Path,
    *,
    transport: str | None = None,
    notify_fn: Callable[..., bool] | None = None,
) -> dict[str, Any]:
    """Best-effort non-Pushover delivery of the digest.

    Returns a record describing what happened (for the ledger row):
    ``{"transport": <name|None>, "delivered": bool, "fallback": bool}``.

    - ``transport`` defaults to the configured loud transport (or None).
    - ``notify_fn`` is the transport callable (message, title) -> bool. When
      None and a transport is configured, delivery is recorded as not
      delivered (Matrix provisioning is out of scope) and the file+ledger
      fallback applies.
    - Pushover is NEVER used: if ``transport`` is ``"pushover"``, it is
      ignored and the fallback applies.
    """
    if transport == PUSHOVER_TRANSPORT:
        # This lane never sends Pushover. Fall through to the file+ledger
        # fallback; the digest file is the source of truth.
        transport = None

    if transport is None and notify_fn is None:
        transport = _configured_loud_transport()

    if transport is None:
        # Fallback: file+ledger. No loud transport configured.
        return {
            "transport": None,
            "delivered": False,
            "fallback": True,
            "digest_path": str(digest_path),
        }

    delivered = False
    if notify_fn is not None:
        try:
            delivered = bool(
                notify_fn(
                    message=f"Morning digest ready: {digest_path}",
                    title="Morning digest",
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("loud transport %r failed: %s", transport, exc)
            delivered = False

    return {
        "transport": transport,
        "delivered": delivered,
        "fallback": not delivered,
        "digest_path": str(digest_path),
    }


# ---------------------------------------------------------------------------
# D1+D2+D3+D4 — end-to-end night-run observability helper
# ---------------------------------------------------------------------------

def record_night_run(
    *,
    lanes: list[str],
    dispatched: set[str] | None = None,
    skipped: dict[str, str] | None = None,
    noop: dict[str, str] | None = None,
    failed: dict[str, str] | None = None,
    candidate_counts: dict[str, int] | None = None,
    queue_path: Path | None = None,
    mem_store: Any | None = None,
    now: datetime | None = None,
) -> RunLedger:
    """Write the run-open ledger row and bootstrap the queue (D1 + D2).

    Called at night-run start. Returns the :class:`RunLedger` (phase=open) so
    the caller can :meth:`RunLedger.summarize` and re-emit at completion.
    """
    ensure_queue_file(queue_path)
    run_id = utc_run_ts(now)
    lane_records = lane_records_for_run(
        lanes,
        dispatched=dispatched,
        skipped=skipped,
        noop=noop,
        failed=failed,
        candidate_counts=candidate_counts,
    )
    ledger = RunLedger.open(run_id, lanes=lane_records, now=now)
    emit_run_ledger(ledger, mem_store=mem_store)
    return ledger


def emit_night_summary(
    ledger: RunLedger,
    *,
    wall_seconds: float | None = None,
    lanes: list[LaneRecord] | None = None,
    aborted: bool = False,
    abort_reason: str = "",
    now: datetime | None = None,
    mem_store: Any | None = None,
) -> str:
    """Write the run-summary ledger row (D1). Returns the mem key."""
    ledger.summarize(
        wall_seconds=wall_seconds,
        lanes=lanes,
        aborted=aborted,
        abort_reason=abort_reason,
        now=now,
    )
    return emit_run_ledger(ledger, mem_store=mem_store)


def serve_morning(
    *,
    ledger: dict[str, Any] | None = None,
    resurfaced: list[dict[str, Any]] | None = None,
    date_str: str | None = None,
    morning_dir: Path | None = None,
    notify_fn: Callable[..., bool] | None = None,
    transport: str | None = None,
    mem_store: Any | None = None,
    run_id: str | None = None,
) -> Path:
    """Compose + write the morning digest, deliver it, and record the path
    in the ledger row (D3 + D4).

    Returns the digest file path. The ledger row (if a ``run_id`` is given) is
    updated with the digest path + delivery record.
    """
    digest_path = emit_morning_digest(
        ledger=ledger,
        resurfaced=resurfaced,
        date_str=date_str,
        morning_dir=morning_dir,
    )
    delivery = deliver_morning_digest(
        digest_path, transport=transport, notify_fn=notify_fn
    )

    # Record the digest path + delivery on the ledger row (best-effort).
    if run_id is not None:
        store = mem_store if mem_store is not None else _default_mem_store()
        entry = store.get(ledger_key(run_id))
        try:
            payload = json.loads(entry["content"]) if entry else {}
        except (json.JSONDecodeError, KeyError, TypeError):
            payload = {}
        payload["digest_path"] = str(digest_path)
        payload["notify"] = delivery
        store.set(
            ledger_key(run_id),
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            tags=["pm-night-run"],
        )
    return digest_path
