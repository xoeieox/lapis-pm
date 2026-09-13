"""Served-model provenance plumbing (local-reviewer-identity-and-provenance-v0,
Leg 2: L2.D2 / L2.D3 / L2.D4).

The locality ledger (``/srv/lapis/locality/<UTC-date>.jsonl``) already records the
ACTUAL served model per LLM call. This module is the read side of that ledger
for two human-facing surfaces:

- dispatch records (L2.D2): the ``served_model`` field, populated best-effort
  from the completed queue task yaml (Leg 1's L1.D3 field). A void echo is
  explicit ``None`` — the seat alias is NEVER substituted into the model
  field (Erah 2026-09-06 explicit-void adjudication).
- the deploy-log label (L2.D4): ``<seat-alias>:<served-model>``, where the
  seat alias resolves statically to the operator default ``gravitywell``
  (the brief composer passes no ``model`` kwarg) and the served model is
  the same call's ``served_model_out`` when in-bounds, else the
  seam-filtered ledger fallback, else the explicit ``not-reported`` marker.

Every function here is existence-checked and fail-soft: a missing yaml, a
missing ledger file, or a malformed line degrades to ``None`` /
``not-reported`` — never an exception up the verdict-encode path.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

# L1.D1 bound: the served model is server-echoed; accept only bounded tokens.
# A violating token is VOID (never rendered / recorded as a value).
SERVED_MODEL_TOKEN_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")
SERVED_MODEL_MAX_LEN = 200

# L2.D4: the seat alias for the brief/deploy surface. The brief composer
# passes NO ``model`` kwarg to call_gw_agent (verified brief.py), so the alias
# resolves STATICALLY to the operator default — the ledger's
# ``requested_operator``. There is no ``brief`` row in registry.yaml; do not
# invent one.
BRIEF_SEAT_ALIAS = "gravitywell"

# The ledger seam that carries the served-model echo (the gw_agent chokepoint).
# The ledger ``host`` field is the full GW base URL, so filtering keys on the
# seam field ONLY — never a host literal.
_LEDGER_SEAM = "call_gw_agent"

# The explicit-void marker for human-facing text (L2.D4). Data surfaces carry
# None; only the label text uses this marker.
NOT_REPORTED = "not-reported"


def _locality_root() -> Path:
    """Resolved locality-ledger root (honors LOCALITY_LEDGER_ROOT; default
    /srv/lapis/locality). Read-only for this leg — no format changes."""
    try:
        from agents_core.locality import root as _locality_root_fn
        return _locality_root_fn()
    except Exception:
        import os
        override = os.environ.get("LOCALITY_LEDGER_ROOT")
        return Path(override) if override else Path("/srv/lapis/locality")


def valid_served_token(value) -> str | None:
    """Return the token if it is a valid served-model value, else None.

    None/empty and bound-violating tokens (charset or length) are VOID —
    the caller must treat them as "not reported", never as the seat alias.
    """
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or len(token) > SERVED_MODEL_MAX_LEN:
        return None
    if not SERVED_MODEL_TOKEN_RE.match(token):
        return None
    return token


def read_served_model_from_queue_yaml(task_id: str | None) -> str | None:
    """L2.D2 read source: the completed queue task yaml's ``served_model``
    field (Leg 1's L1.D3 stamp, beside the existing ``model:`` requested
    alias).

    Null-tolerant across every degraded case:
    - missing ``task_id`` (legacy gpu_queue-era record) -> None
    - yaml absent entirely (crashed runs; the claude_queue.failed/ dir) -> None
    - field missing (pre-Leg-1 yamls) -> None
    - field explicitly ``null`` (void echo) -> None — NEVER the seat alias.

    Existence-checked best-effort; never raises (the verdict-encode idiom at
    pm_core.py:8082/8103/8121 — a provenance read must never fail
    verdict-encode).
    """
    if not task_id:
        return None
    try:
        import yaml
        from . import pm_core as _pm_core
        # Namespace guard (cycle-1 review): prefer the claude_queue yamls for
        # claude_-prefixed task ids — a legacy gpu_queue yaml that happens to
        # share the name must not shadow the claude_queue record. The legacy
        # names are consulted only when absent.
        if task_id.startswith("claude_"):
            candidates = (
                _pm_core.CLAUDE_QUEUE_COMPLETED_DIR / f"{task_id}.yaml",
                _pm_core.CLAUDE_QUEUE_FAILED_DIR / f"{task_id}.yaml",
            )
        else:
            candidates = (
                _pm_core.CLAUDE_QUEUE_COMPLETED_DIR / f"{task_id}.yaml",
                _pm_core.CLAUDE_QUEUE_FAILED_DIR / f"{task_id}.yaml",
                _pm_core.COMPLETED_DIR / f"{task_id}.yaml",
                _pm_core.FAILED_DIR / f"{task_id}.yaml",
            )
        for path in candidates:
            if not path.exists():
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            return valid_served_token(data.get("served_model"))
    except Exception:
        return None
    return None


def _ledger_day_served_models(day: datetime) -> list[str | None]:
    """All ``served_model`` values for the given UTC day, filtered on
    ``seam == "call_gw_agent"`` ONLY, in file order. Malformed lines are
    skipped (the ledger's own read discipline)."""
    path = _locality_root() / f"{day.strftime('%Y-%m-%d')}.jsonl"
    if not path.exists():
        return []
    values: list[str | None] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("seam") != _LEDGER_SEAM:
            continue
        values.append(valid_served_token(entry.get("served_model")))
    return values


def _ledger_fallback_served(now_utc: datetime | None = None) -> str | None:
    """L2.D4 fallback (resolution step 2): the last in-bounds ``served_model``
    on the seam for the ledger's UTC date — and ONLY when that UTC day
    carries exactly ONE distinct in-bounds ``served_model`` (the
    multi-model guard: a mixed day yields the void marker, never a guess).

    Date convention: the ledger file is UTC-dated while the host clock is
    PDT, so "today" means the ledger's UTC date, with a fallback to the
    previous UTC day before ``not-reported``.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    for day in (now_utc, now_utc - timedelta(days=1)):
        values = _ledger_day_served_models(day)
        if not values:
            continue
        distinct = {v for v in values if v is not None}
        if len(distinct) == 1:
            return next(iter(distinct))
        if len(distinct) > 1:
            # A mixed day is unresolvable — do not fall back to the previous
            # day with a different model than the one that actually served
            # today's call.
            return None
        # Only void entries today (no in-bounds echo) — try the previous day.
    return None


def deploy_log_label(served_model_out: list | None = None,
                     now_utc: datetime | None = None) -> str:
    """L2.D4: the deploy-log provenance label ``<seat-alias>:<served-model>``.

    Resolution order (explicit-void semantics, Erah 2026-09-06):
      1. the same call's ``served_model_out`` (last entry) when it yields an
         in-bounds token (the L1.D1 bound);
      2. the seam-filtered, single-distinct-model locality-ledger fallback
         (UTC date, previous-UTC-day fallback);
      3. ``<seat-alias>:not-reported``.

    The seat alias is NEVER written into the served-model slot — a
    multi-model day or a missing echo yields ``not-reported``, not the
    alias. Fail-soft: any internal error degrades to the void marker.
    """
    try:
        alias = BRIEF_SEAT_ALIAS
        served = None
        if served_model_out:
            served = valid_served_token(served_model_out[-1])
        if served is None:
            served = _ledger_fallback_served(now_utc)
        return f"{alias}:{served if served is not None else NOT_REPORTED}"
    except Exception:
        return f"{BRIEF_SEAT_ALIAS}:{NOT_REPORTED}"


def served_tag_for(record: dict) -> str | None:
    """L2.D3: the ``pm:served=<model>`` episodic tag for a dispatch record.

    Returns the tag when the record carries a known served model; returns
    None (ABSENCE, not an error) for pre-Leg-1 yamls, crashed runs, and
    null-echo cases. The tag value is never the seat alias.
    """
    served = valid_served_token(record.get("served_model"))
    if served is None:
        return None
    # Erah 2026-09-06 ruling: the seat alias is a role, never a model
    # substance — even a corrupted/hand-edited record holding the alias must
    # not emit a pm:served tag with it (re-validation, not just upstream
    # discipline).
    if served == BRIEF_SEAT_ALIAS:
        return None
    return f"pm:served={served}"
