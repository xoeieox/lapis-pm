"""Deploy-inventory repair proposer (deploy-inventory-repair-proposer-v0).

Replaces the per-finding Pushover relay in `pm_core._reconcile_deploy_inventory()`
with Desk-gem deposits: for every fresh HIGH finding, diagnose it (via
`call_gw_agent`, graceful-degrading to the finding's own `detail` string on any
failure), dedup against a lapis-pm-local ledger keyed by `(clone_path,
finding_kind)`, and deposit one gem per fresh finding-class on the weaver Desk
instead of paging Erah directly. See the bound spec for full design (D1-D11).

Second, independent instance of the repair-proposer pattern (the first is
`night-manager-repair-proposer-v0`, in the conductor repo, pointed at
night-DAG). No shared library extraction here - premature at n=2.

Advisory-only (D9): this module calls no bind/dispatch/authorize. A gem is a
proposal on a human surface. Fixing the underlying findings is separate,
un-gated remediation work.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from agents_core.room_paths import room_path

from .file_lock import FileLockTimeout, file_lock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEDGER_FILE = room_path('lapis_state') / "deploy-inventory-repair-ledger.json"
_LEDGER_LOCK_FILE = room_path('lapis_state') / "deploy-inventory-repair-ledger.lock"

DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN = int(
    os.environ.get("DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN", "5")
)
DEPLOY_INVENTORY_STALE_GEM_SECS = int(
    os.environ.get("DEPLOY_INVENTORY_STALE_GEM_SECS", str(7 * 24 * 3600))
)
DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS = float(
    os.environ.get("DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS", "5")
)

# Distinguishable action string for a lock-contended skip (D3): the pass
# deposits nothing and this must be visible in the actions list rather than
# looking like a clean empty pass.
LEDGER_LOCK_CONTENDED_ACTION = "skip:ledger-lock-contended"

# Fixed signature for the corrupt-status meta-finding (D8) - no per-clone
# signature applies, so it does not go through the D3 (clone_path, kind) ledger.
_CORRUPT_STATUS_SIGNATURE = "corrupt-status"

_ACK_WATCH_OPTION = "ack_watch"
_NOT_REAL_OPTION = "not_real"

_GEM_OPTIONS = [
    {"key": _ACK_WATCH_OPTION, "title": "Known - keep watching",
     "sub": "stay quiet unless it recurs after a fix", "primary": True},
    {"key": _NOT_REAL_OPTION, "title": "Not a real issue",
     "sub": "suppress this finding-class"},
]

_DIAGNOSTICIAN_SYSTEM = """You are a deploy-inventory diagnostician for the Lapis PM.

You are given one drift finding about a git clone that backs (or should back) a
live systemd unit on a host. Your job is to produce a short, mechanical
diagnosis - state what is true and what it backs, plainly. No narrative
framing, no editorializing on stakes beyond the facts given.

Respond with EXACTLY one JSON object, no prose outside it:
{"root_cause": "<one or two sentences>",
 "suggested_direction": "<one or two sentences, concrete next step>",
 "affected_files": ["<path>", ...],
 "confidence": "high"|"medium"|"low",
 "uncertain": true|false}
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# D3 - signature
# ---------------------------------------------------------------------------

def compute_signature(clone_path: str, finding_kind: str) -> str:
    """sha256(clone_path|finding_kind)[:16].

    Deliberately excludes `commits_behind`'s count and any timestamp - a clone
    going from 2 to 3 commits behind is the same fix-class, not a new one.
    """
    return hashlib.sha256(f"{clone_path}|{finding_kind}".encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ledger - atomic write-to-temp-then-os.replace (gate finding, mandatory),
# fail-open read.
# ---------------------------------------------------------------------------

def read_ledger(path: Path | None = None) -> dict:
    target = path or _LEDGER_FILE
    try:
        return json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_ledger(ledger: dict, path: Path | None = None) -> None:
    """Atomic write: write-to-temp-then-`os.replace`. Never a torn/partial file
    even if interrupted mid-flight (Facets gate finding, mandatory). This
    guards readers against a torn file; it does not by itself serialise
    concurrent read-modify-write. The reachable overlap is not two
    timer-fired PM ticks (systemd's `OnUnitActiveSec` cannot re-fire while
    `lapis-pm.service` - a `Type=oneshot` unit - is still running) but an
    operator running `lapis-pm tick --all` by hand concurrently with the
    timer, which starts a separate process outside systemd's job model.
    Callers take the `_LEDGER_LOCK_FILE` flock (see `run_repair_pass` /
    `propose_corrupt_status`) around their read-modify-write region to close
    that window; this function's atomicity and that lock are orthogonal and
    both required.
    """
    target = path or _LEDGER_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=".deploy-inventory-repair-ledger-", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(ledger, indent=2, sort_keys=True))
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except OSError as e:
        logger.warning("[deploy-inventory-repair] ledger write failed: %s", e)


# ---------------------------------------------------------------------------
# D4 - diagnosis (bounded, graceful-degrade)
# ---------------------------------------------------------------------------

def _degrade(finding: dict) -> dict:
    """Graceful degrade: fall back to the finding's own `detail` as the entire
    diagnosis, uncertain=true, never fabricate a root cause.
    """
    return {
        "root_cause": finding.get("detail", ""),
        "suggested_direction": "",
        "affected_files": [],
        "confidence": None,
        "uncertain": True,
    }


def diagnose_finding(
    clone_path: str,
    finding: dict,
    *,
    mapped: bool,
    commits_behind: int | None,
    branch: str | None,
    backing_units: list,
) -> tuple[dict, str]:
    """Returns (diagnosis_dict, model_used_or_"mechanical").

    Calls call_gw_agent (read-only, cwd=clone_path when it exists) to interpret
    the finding. Any non-dict/timeout/wake-fail result degrades to the
    finding's own detail string, uncertain=True, and reports "mechanical" as
    the model (D7 - never let a graceful-degrade misattribute to a model that
    did not actually produce the content).
    """
    prompt = (
        f"clone_path: {clone_path}\n"
        f"finding.kind: {finding.get('kind')}\n"
        f"finding.detail: {finding.get('detail')}\n"
        f"mapped: {mapped}\n"
        f"commits_behind: {commits_behind}\n"
        f"branch: {branch}\n"
        f"backing_units: {backing_units}\n\n"
        "Produce the JSON diagnosis now."
    )
    cwd = clone_path if clone_path and os.path.isdir(clone_path) else None
    served_model_out: list = []
    try:
        from agents_core.gw_agent import call_gw_agent
        text = call_gw_agent(
            prompt=prompt,
            system=_DIAGNOSTICIAN_SYSTEM,
            cwd=cwd,
            writeable=False,
            json_mode=True,
            max_steps=12,
            timeout=240,
            on_wake_fail="skip",
            served_model_out=served_model_out,
        )
    except Exception as exc:
        logger.warning(
            "[deploy-inventory-repair] call_gw_agent error for %s/%s: %s",
            clone_path, finding.get("kind"), exc,
        )
        return _degrade(finding), "mechanical"

    if not isinstance(text, str) or not text.strip():
        return _degrade(finding), "mechanical"

    try:
        from agents_core.llm import parse_json_object
        parsed = parse_json_object(text)
    except Exception as exc:
        logger.warning("[deploy-inventory-repair] diagnosis parse failed: %s", exc)
        parsed = None

    if not isinstance(parsed, dict):
        return _degrade(finding), "mechanical"

    diag = {
        "root_cause": parsed.get("root_cause") or finding.get("detail", ""),
        "suggested_direction": parsed.get("suggested_direction") or "",
        "affected_files": parsed.get("affected_files") or [],
        "confidence": parsed.get("confidence"),
        "uncertain": bool(parsed.get("uncertain", False)),
    }
    model_used = served_model_out[0] if served_model_out else "unknown"
    return diag, model_used


# ---------------------------------------------------------------------------
# D6 - stakes line (mechanical, no model phrasing)
# ---------------------------------------------------------------------------

def build_stakes_line(
    *, backing_units: list, mapped: bool, times_seen: int,
) -> str:
    """Pure f-string from structured facts already on hand. No model call."""
    live_units = [bu.get("unit") for bu in backing_units if bu.get("live")] if backing_units else []
    if live_units:
        line = f"backs {len(live_units)} live unit(s) ({', '.join(live_units)})"
    else:
        line = (
            "not currently backing a live unit; "
            + ("unmapped - invisible to the standard pull/restart path" if not mapped else "mapped but drifted")
        )
    if times_seen > 1:
        line += f"; seen across {times_seen} passes"
    return line


# ---------------------------------------------------------------------------
# D5 - Desk gem deposit (reuses brief_gem's httpx/_weaver_base_url/pytest-guard)
# ---------------------------------------------------------------------------

def _pytest_guard_blocks() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) and not os.environ.get("WEAVER_BASE_URL")


def deposit_gem(payload: dict) -> str | None:
    """POST /v0/decision-gems. Returns gem_id on 201, None on any failure
    (weaver unreachable, non-201, or under-pytest-without-mock guard).
    """
    if _pytest_guard_blocks():
        return None

    from . import brief_gem as _brief_gem

    try:
        import httpx
        base = _brief_gem._weaver_base_url()
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{base}/v0/decision-gems", json=payload)
            if resp.status_code != 201:
                logger.warning(
                    "[deploy-inventory-repair] deposit non-201 (%s): %s",
                    resp.status_code, resp.text[:300],
                )
                return None
            return resp.json().get("gem_id")
    except Exception as exc:
        logger.warning("[deploy-inventory-repair] deposit failed: %s", exc)
        return None


def _fetch_gems_by_state(state: str) -> list[dict]:
    """GET /v0/decision-gems?state=<state>. [] on any failure (fail-soft)."""
    if _pytest_guard_blocks():
        return []

    from . import brief_gem as _brief_gem

    try:
        import httpx
        base = _brief_gem._weaver_base_url()
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{base}/v0/decision-gems", params={"state": state, "limit": 100},
            )
            resp.raise_for_status()
            return resp.json().get("gems", [])
    except Exception as exc:
        logger.warning(
            "[deploy-inventory-repair] gem fetch (state=%s) failed: %s", state, exc,
        )
        return []


def build_gem_payload(
    *,
    clone_path: str,
    finding: dict,
    mapped: bool,
    branch: str | None,
    commits_behind: int | None,
    diagnosis: dict,
    stakes_line: str,
    times_seen: int,
    title_prefix: str = "",
) -> dict:
    kind = finding.get("kind", "?")
    name = Path(clone_path).name
    context = [
        {"label": "Symptom", "lines": [
            finding.get("detail", ""),
            f"clone={clone_path} mapped={mapped} branch={branch} behind={commits_behind}",
        ]},
        {"label": "Diagnosis", "lines": [diagnosis.get("root_cause", "")]},
        {"label": "Suggested direction", "lines":
            [diagnosis.get("suggested_direction", "")] + list(diagnosis.get("affected_files") or [])},
        {"label": "Stakes", "lines": [stakes_line]},
    ]
    if times_seen > 1:
        context.append({"label": "Recurrence", "lines": [f"seen {times_seen} time(s)"]})

    return {
        "title": f"{title_prefix}{kind}: {name}",
        "ask": finding.get("detail", f"{kind} on {name}"),
        "why": (
            f"Deploy-drift on a {'mapped' if mapped else 'unmapped'} clone - "
            "a config/host-ops call, not reversible-Tier-0."
        ),
        "context": context,
        "options": _GEM_OPTIONS,
        "state": "needs",
        "origin": "from · deploy-inventory",
        "agent": "repair-expert",
        "deposited_by": "deploy-inventory-repair-proposer-v0",
    }


# ---------------------------------------------------------------------------
# D7 - provenance (best-effort, non-blocking)
# ---------------------------------------------------------------------------

_DEPOSIT_RECORDER_SPEC = os.environ.get(
    "LAPIS_PM_DEPOSIT_RECORDER", "zephyr.attribution:get_recorder"
)


def emit_provenance(*, gem_id: str, signature: str, clone_path: str, model: str) -> None:
    """Best-effort Zephyr attribution deposit. Never raises, never blocks or
    rolls back a successful gem deposit (D7 gate finding, mandatory).
    """
    try:
        import importlib
        module_name, _, attr = _DEPOSIT_RECORDER_SPEC.partition(":")
        recorder = getattr(importlib.import_module(module_name), attr)()
        manifest = "sha256:" + hashlib.sha256(
            f"deploy-inventory-repair:{gem_id}:{signature}".encode("utf-8")
        ).hexdigest()
        prov = {
            "manifest_hash": manifest,
            "agent_id": "deploy-inventory-repair-proposer-v0",
            "tool": "lapis-pm:deploy-inventory-repair",
            "model": model,
            "timestamp": _now_iso(),
            "schema_version": "lapis-provenance-v0",
            "gem_id": gem_id,
            # D6: NOT "signature" — Zephyr's recorder reads that field name as a
            # cryptographic signature and rejects any deposit that carries one
            # without a matching pubkey_id (finding/deploy-inventory-repair-
            # provenance-signature-field-collision-2026-08-08). This is the
            # unit's own fix-class dedup key, not a crypto signature.
            "finding_signature": signature,
            "clone_path": clone_path,
        }
        recorder.record(prov, store_kind="gem", key=gem_id)
    except Exception as exc:
        logger.warning("[deploy-inventory-repair] provenance deposit failed: %s", exc)


# ---------------------------------------------------------------------------
# D3 - dedup + reconciliation
# ---------------------------------------------------------------------------

def reconcile_terminal_gems(ledger: dict) -> None:
    """Fetch decided + superseded gems once, update ledger `status` from
    `decision_json.option_key` (or "dismissed_fp" for a superseded gem with
    no option_key). Mutates `ledger` in place.
    """
    for gem in _fetch_gems_by_state("decided"):
        gem_id = gem.get("gem_id")
        decision = gem.get("decision_json") or {}
        option_key = decision.get("option_key")
        for entry in ledger.values():
            if entry.get("gem_id") != gem_id:
                continue
            if option_key == _ACK_WATCH_OPTION:
                entry["status"] = "acked"
            elif option_key == _NOT_REAL_OPTION:
                entry["status"] = "dismissed_fp"

    for gem in _fetch_gems_by_state("superseded"):
        gem_id = gem.get("gem_id")
        for entry in ledger.values():
            if entry.get("gem_id") == gem_id and entry.get("status") == "open":
                entry["status"] = "dismissed_fp"


def would_deposit(ledger: dict, signature: str) -> bool:
    """D1a's single shared dedup predicate: True when a candidate with this
    signature would actually be deposited (no ledger entry yet, or the entry
    was acked and this is a recurrence); False when it would be suppressed
    (an open gem, or a dismissed false-positive).

    This is the ONE predicate `propose_repair`'s own early-returns and the
    `run_repair_pass` cap partition (D1a) both consult - never reimplemented
    at either call site (DoD-10 pins them together).
    """
    entry = ledger.get(signature)
    if entry is None:
        return True
    return entry.get("status") == "acked"


def propose_repair(
    ledger: dict,
    *,
    clone_path: str,
    finding: dict,
    mapped: bool,
    branch: str | None,
    commits_behind: int | None,
    backing_units: list,
) -> str | None:
    """Apply the D3 dedup rule for one fresh finding, diagnose + deposit if
    warranted. Returns the action taken as a short string for logging, or
    None if the caller should skip (dedup rule details are on the ledger
    entry mutation).

    Mutates `ledger` in place. Caller persists via write_ledger() once per
    pass (not per finding).
    """
    signature = compute_signature(clone_path, finding.get("kind", ""))
    entry = ledger.get(signature)
    now = _now_iso()

    if not would_deposit(ledger, signature):
        entry["last_seen"] = now
        entry["times_seen"] = entry.get("times_seen", 1) + 1
        if entry.get("status") == "open":
            return "skip:still-open"
        return "skip:dismissed-fp"

    recurred_after_ack = entry is not None and entry.get("status") == "acked"

    diagnosis, model_used = diagnose_finding(
        clone_path, finding, mapped=mapped, commits_behind=commits_behind,
        branch=branch, backing_units=backing_units,
    )

    times_seen = (entry.get("times_seen", 0) + 1) if entry else 1
    stakes_line = build_stakes_line(
        backing_units=backing_units, mapped=mapped, times_seen=times_seen,
    )
    if recurred_after_ack:
        stakes_line += " (recurred-after-ack)"

    payload = build_gem_payload(
        clone_path=clone_path, finding=finding, mapped=mapped, branch=branch,
        commits_behind=commits_behind, diagnosis=diagnosis, stakes_line=stakes_line,
        times_seen=times_seen,
    )

    gem_id = deposit_gem(payload)
    if gem_id is None:
        # Non-201/unreachable: leave the ledger unrecorded for this signature
        # so the next pass retries (D5).
        return "skip:deposit-failed"

    ledger[signature] = {
        "gem_id": gem_id,
        "first_seen": entry.get("first_seen") if entry else now,
        "last_seen": now,
        "times_seen": times_seen,
        "status": "open",
        "staled_at": None,
        "clone_path": clone_path,
        "finding_kind": finding.get("kind"),
    }

    emit_provenance(gem_id=gem_id, signature=signature, clone_path=clone_path, model=model_used)

    return "deposited:recurred-after-ack" if recurred_after_ack else "deposited:new"


def propose_corrupt_status() -> str | None:
    """The corrupt-status meta-signal (D8): deposits at most once while the
    underlying corruption persists unresolved (fixed signature, no per-clone
    ledger entry applies). Uses the module-level ledger for its own dedup so
    it shares the same atomic-write path as the per-clone entries.

    Takes `_LEDGER_LOCK_FILE` around its own read-modify-write region: this
    is an independent writer from `run_repair_pass`, each with its own
    in-memory ledger copy, so the two must be mutually exclusive across
    processes (deploy-inventory-repair-ledger-lock-v0). On contention this
    fails closed - deposits nothing, logs WARNING, returns the
    distinguishable `LEDGER_LOCK_CONTENDED_ACTION` - never proceeds
    unlocked.
    """
    try:
        with file_lock(_LEDGER_LOCK_FILE, DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS):
            return _propose_corrupt_status_locked()
    except FileLockTimeout:
        logger.warning(
            "[deploy-inventory-repair] ledger lock contended (path=%s, timeout=%ss); "
            "skipping corrupt-status pass, depositing nothing",
            _LEDGER_LOCK_FILE, DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS,
        )
        return LEDGER_LOCK_CONTENDED_ACTION


def _propose_corrupt_status_locked() -> str | None:
    """Body of `propose_corrupt_status`, run under the ledger lock."""
    ledger = read_ledger()
    entry = ledger.get(_CORRUPT_STATUS_SIGNATURE)
    now = _now_iso()
    if entry is not None and entry.get("status") == "open":
        entry["last_seen"] = now
        entry["times_seen"] = entry.get("times_seen", 1) + 1
        write_ledger(ledger)
        return "skip:still-open"

    payload = {
        "title": "deploy-inventory: status file unreadable",
        "ask": (
            "prior deploy-inventory-status.json is present but unreadable - "
            "per-finding notification dedup is suppressed for this pass."
        ),
        "why": "Reconciler self-health signal, not a per-clone finding.",
        "context": [{"label": "Symptom", "lines": [
            "prior status JSON present but unreadable/unparseable",
        ]}],
        "options": _GEM_OPTIONS,
        "state": "needs",
        "origin": "from · deploy-inventory",
        "agent": "repair-expert",
        "deposited_by": "deploy-inventory-repair-proposer-v0",
    }
    gem_id = deposit_gem(payload)
    if gem_id is None:
        return "skip:deposit-failed"

    ledger[_CORRUPT_STATUS_SIGNATURE] = {
        "gem_id": gem_id,
        "first_seen": entry.get("first_seen") if entry else now,
        "last_seen": now,
        "times_seen": (entry.get("times_seen", 0) + 1) if entry else 1,
        "status": "open",
        "staled_at": None,
        "clone_path": None,
        "finding_kind": None,
    }
    write_ledger(ledger)
    emit_provenance(gem_id=gem_id, signature=_CORRUPT_STATUS_SIGNATURE, clone_path="", model="mechanical")
    return "deposited:corrupt-status"


# ---------------------------------------------------------------------------
# D11 - stale-gem escalation (prevents the Desk from becoming a second
# unread queue). Fires once per ledger entry (staled_at gate).
# ---------------------------------------------------------------------------

def escalate_stale_gems(ledger: dict, *, now: datetime | None = None) -> list[str]:
    """For any ledger entry with status=="open" whose first_seen is older
    than DEPLOY_INVENTORY_STALE_GEM_SECS and staled_at is unset, deposit one
    visible, one-time re-signal gem. Mutates `ledger` in place (sets
    staled_at). Returns signatures that fired this call.
    """
    now = now or _now_dt()
    fired: list[str] = []
    for signature, entry in ledger.items():
        if entry.get("status") != "open":
            continue
        if entry.get("staled_at"):
            continue
        first_seen_raw = entry.get("first_seen")
        if not first_seen_raw:
            continue
        try:
            first_seen = _parse_iso(first_seen_raw)
        except ValueError:
            continue
        age = (now - first_seen).total_seconds()
        if age < DEPLOY_INVENTORY_STALE_GEM_SECS:
            continue

        days = int(age // 86400)
        clone_path = entry.get("clone_path") or "?"
        finding_kind = entry.get("finding_kind") or "?"
        payload = {
            "title": f"[{days}d unresolved] {finding_kind}: {Path(clone_path).name if clone_path != '?' else '?'}",
            "ask": (
                f"This finding-class has been open on the Desk for {days} day(s) "
                "without a decision - still relevant, or should it be dismissed?"
            ),
            "why": "Stale-open-gem escalation - a gem sitting undecided indefinitely reproduces the queue this unit exists to close.",
            "context": [{"label": "Recurrence", "lines": [
                f"open for {days}d across {entry.get('times_seen', 1)} passes",
                f"original gem_id={entry.get('gem_id')}",
            ]}],
            "options": _GEM_OPTIONS,
            "state": "needs",
            "origin": "from · deploy-inventory",
            "agent": "repair-expert",
            "deposited_by": "deploy-inventory-repair-proposer-v0",
        }
        gem_id = deposit_gem(payload)
        # Fires once regardless of deposit outcome — a failed deposit still
        # marks staled_at so a transient weaver outage doesn't turn into an
        # unbounded retry loop (D11(b): no unbounded re-deposits).
        entry["staled_at"] = _now_iso()
        if gem_id is not None:
            entry["stale_gem_id"] = gem_id
        fired.append(signature)
    return fired


# ---------------------------------------------------------------------------
# D1/D4/D5/D6 - the entry point pm_core calls
# ---------------------------------------------------------------------------

def run_repair_pass(
    status: dict,
    *,
    prior_high_keys: set | None,
    corrupt: bool,
) -> list[str]:
    """Replaces the per-finding Pushover loop + corrupt-status branch in
    `pm_core._reconcile_deploy_inventory()`. Returns a list of action strings
    for logging. Fault-isolated per finding (D1/DoD-6): a deposit/diagnosis
    exception for one finding is logged and skipped, never aborts the pass.
    """
    actions: list[str] = []

    if corrupt:
        try:
            action = propose_corrupt_status()
            if action:
                actions.append(action)
        except Exception as exc:
            logger.warning("[deploy-inventory-repair] corrupt-status path failed: %s", exc)
        return actions

    # The read-modify-write below spans the propose loop, up to
    # `DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN * 240s` of `call_gw_agent` diagnosis
    # calls in the worst case - minutes, not milliseconds. Taking the ledger
    # lock across that whole region is deliberate (deploy-inventory-repair-
    # ledger-lock-v0 D2): the only reachable contender is an operator running
    # `lapis-pm tick --all` concurrently with the timer, and the correct
    # behaviour for that second process is to do nothing, not to interleave a
    # lost update. On contention this fails closed - deposits nothing, logs
    # WARNING, returns the distinguishable `LEDGER_LOCK_CONTENDED_ACTION` -
    # never proceeds unlocked.
    try:
        with file_lock(_LEDGER_LOCK_FILE, DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS):
            actions.extend(_run_repair_pass_locked(status, prior_high_keys=prior_high_keys))
    except FileLockTimeout:
        logger.warning(
            "[deploy-inventory-repair] ledger lock contended (path=%s, timeout=%ss); "
            "skipping repair pass, depositing nothing",
            _LEDGER_LOCK_FILE, DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS,
        )
        actions.append(LEDGER_LOCK_CONTENDED_ACTION)

    return actions


def _run_repair_pass_locked(
    status: dict,
    *,
    prior_high_keys: set | None,
) -> list[str]:
    """Body of `run_repair_pass`'s non-corrupt path, run under the ledger
    lock. Read-modify-write region: `read_ledger()` through `write_ledger()`.
    """
    actions: list[str] = []

    ledger = read_ledger()

    try:
        reconcile_terminal_gems(ledger)
    except Exception as exc:
        logger.warning("[deploy-inventory-repair] terminal-gem reconciliation failed: %s", exc)

    # D1: admit a finding if ANY of exempt / fresh / never-put-in-front-of-a-
    # human is true. `is_unproposed` is the backlog admit path - the whole
    # point of this unit - and is deliberately a *union* with freshness, not
    # a replacement for it (see spec D1: preserves the acked-recurrence case).
    fresh: list[tuple] = []
    for clone in status.get("clones", []):
        for finding in clone.get("findings", []):
            if finding.get("severity") != "HIGH":
                continue
            key = (clone.get("path"), finding.get("kind"))
            is_exempt = finding.get("kind") == "auto_recovery_restart_failed"
            is_fresh = prior_high_keys is None or key not in prior_high_keys
            signature = compute_signature(clone.get("path"), finding.get("kind", ""))
            is_unproposed = signature not in ledger
            if not (is_exempt or is_fresh or is_unproposed):
                continue
            fresh.append((clone, finding))

    # D1a: partition BEFORE capping. A candidate that would_deposit()==False
    # (an open gem or a dismissed_fp) must never consume a cap slot - under
    # D1 the same chronic, suppressed candidates would otherwise occupy every
    # top slot on every pass, forever, and the backlog would never drain.
    would_list: list[tuple] = []
    suppressed_list: list[tuple] = []
    for clone, finding in fresh:
        signature = compute_signature(clone.get("path"), finding.get("kind", ""))
        if would_deposit(ledger, signature):
            would_list.append((clone, finding))
        else:
            suppressed_list.append((clone, finding))

    # D3: deterministic ordering under the cap - exempt first, then most live
    # backing units, then unmapped before mapped, then clone_path ascending.
    # No model call participates; purely a function of the status dict.
    def _sort_key(item: tuple) -> tuple:
        clone, finding = item
        is_exempt = finding.get("kind") == "auto_recovery_restart_failed"
        live_count = sum(1 for bu in (clone.get("backing_units") or []) if bu.get("live"))
        return (
            0 if is_exempt else 1,
            -live_count,
            bool(clone.get("mapped")),
            clone.get("path") or "",
        )

    would_list.sort(key=_sort_key)

    dropped = 0
    if len(would_list) > DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN:
        dropped = len(would_list) - DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN
        would_list = would_list[:DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN]

    # Suppressed candidates still flow through propose_repair - it returns
    # skip:still-open / skip:dismissed-fp and bumps last_seen/times_seen (D11
    # staleness accounting, the Recurrence context block), but costs no GW
    # call and no deposit, and never touched the cap above.
    to_process = would_list + suppressed_list

    for clone, finding in to_process:
        try:
            action = propose_repair(
                ledger,
                clone_path=clone.get("path"),
                finding=finding,
                mapped=bool(clone.get("mapped")),
                branch=clone.get("branch"),
                commits_behind=clone.get("commits_behind"),
                backing_units=clone.get("backing_units") or [],
            )
            if action:
                actions.append(f"{action}:{clone.get('path')}:{finding.get('kind')}")
        except Exception as exc:
            logger.warning(
                "[deploy-inventory-repair] propose_repair failed for %s/%s (non-fatal): %s",
                clone.get("path"), finding.get("kind"), exc,
            )
            actions.append(f"error:{clone.get('path')}:{finding.get('kind')}")

    if dropped:
        logger.warning(
            "[deploy-inventory-repair] %d fresh finding-class(es) dropped this pass "
            "(cap=%d, no silent truncation)", dropped, DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN,
        )
        actions.append(f"dropped:{dropped}")

    try:
        escalate_stale_gems(ledger)
    except Exception as exc:
        logger.warning("[deploy-inventory-repair] stale-gem escalation failed: %s", exc)

    write_ledger(ledger)
    return actions
