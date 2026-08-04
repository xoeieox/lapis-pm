"""Availability canary for the Phala contractor seat (D7, lapis-pm-reviewer-
peak-contractor-route-v0).

The re-gate ruling (2026-08-04) fixed D7's shape precisely: a real, minimal
completion run periodically OUT-OF-BAND, its verdict cached with a
timestamp, dispatch reads the cache. Dispatch pays a dictionary lookup; the
canary itself is real inference (never `GET /v1/models`, which is a static
catalog that answers `200` regardless of upstream state
(`agents_core/phala_tee_proxy.py:86-99`) and proves nothing about health);
neither the static-catalog trap nor the per-dispatch-latency trap is taken.

Two verified traps this module exists to not fall into:
  - `GET /v1/models` is NOT a health check (static catalog, always 200).
  - `x-ratelimit-limit: 0` from Phala is NOT a terminal account condition —
    it is emitted during ordinary transient throttling
    (correction/phala-test-keytransient-not-billing-2026-08-04).
    A `429` here is TRANSIENT and RETRYABLE, honouring `retry-after`; it is
    never read as "the seat is dead."

This module does NOT gate the D2 dispatch-time routing decision (that is by
the clock ONLY, per D2 — a health canary must never become a second,
undocumented routing input, or it reintroduces exactly the "GWParkedError
means call the contractor" pattern D2 forbids in reverse). It exists purely
so a human sees seat health before a review depends on it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

CONTRACTOR_BACKEND_URL = "http://127.0.0.1:8413"
CONTRACTOR_MODEL = "deepseek/deepseek-v4-flash"

CACHE_MEM_KEY = "pm/contractor-seat/health"

# Bounded retry for the transient states. Honours `retry-after` when present,
# falls back to exponential backoff otherwise. Never retries a terminal state
# (proxy_down, empty_content) — those are named and returned immediately.
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_S = 1.0

# States, named per D3/D7. "healthy" and "proxy_down" are terminal on sight;
# "upstream_throttled" and "upstream_disconnected" are the transient states
# a probe retries through before giving up and reporting the last one seen.
STATE_HEALTHY = "healthy"
STATE_PROXY_DOWN = "proxy_down"
STATE_UPSTREAM_THROTTLED = "upstream_throttled"
STATE_UPSTREAM_DISCONNECTED = "upstream_disconnected"
STATE_UPSTREAM_5XX = "upstream_5xx"
STATE_EMPTY_CONTENT = "empty_content"


@dataclass
class SeatHealth:
    state: str
    checked_at: str  # ISO8601
    detail: str = ""
    retries_used: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    # Local import to avoid a hard circular dependency on pm_core at module
    # load time; pm_core imports this module, not the reverse.
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def classify_response(status_code: int | None, body: dict | None, exc: Exception | None) -> tuple[str, str]:
    """Classify one probe attempt into (state, detail).

    `exc` takes precedence when the request never got a response at all
    (connection refused → proxy_down; a dropped connection mid-response →
    upstream_disconnected). Both traps this module exists to avoid are
    handled here explicitly, not inferred from a status code alone.
    """
    if exc is not None:
        exc_name = type(exc).__name__
        msg = str(exc)
        if "RemoteDisconnected" in exc_name or "RemoteDisconnected" in msg:
            return STATE_UPSTREAM_DISCONNECTED, msg
        if "ConnectionRefused" in exc_name or "ConnectionError" in exc_name or "Connect" in exc_name:
            return STATE_PROXY_DOWN, msg
        return STATE_PROXY_DOWN, msg

    if status_code == 429:
        # THE TRAP: x-ratelimit-limit: 0 on Phala does not mean "no
        # allowance" — see correction/phala-test-key
        # transient-not-billing-2026-08-04. A 429 is always transient here,
        # never a terminal health verdict, regardless of what the limit
        # header says.
        return STATE_UPSTREAM_THROTTLED, "429 (transient throttle, not a terminal account condition)"

    if status_code is not None and status_code >= 500:
        return STATE_UPSTREAM_5XX, f"upstream {status_code}"

    if status_code == 200:
        content = ""
        if body:
            choices = body.get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content") or ""
        if not content.strip():
            return STATE_EMPTY_CONTENT, "200 with empty content"
        return STATE_HEALTHY, "200, non-empty content"

    return STATE_PROXY_DOWN, f"unexpected status {status_code}"


def _retry_after_seconds(headers: dict | None) -> float | None:
    if not headers:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def probe_seat_once(backend_url: str = CONTRACTOR_BACKEND_URL, model: str = CONTRACTOR_MODEL,
                     timeout: float = 20.0) -> tuple[str, str, dict | None]:
    """One real, minimal completion call against the sealed proxy.

    Deliberately NOT `GET /v1/models` (static catalog, proves nothing about
    upstream state — see module docstring). Returns (state, detail, headers)
    for the caller to decide whether to retry.
    """
    import requests

    try:
        resp = requests.post(
            f"{backend_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 16,
            },
            timeout=timeout,
        )
    except Exception as e:
        state, detail = classify_response(None, None, e)
        return state, detail, None

    body = None
    try:
        body = resp.json()
    except Exception:
        body = None
    state, detail = classify_response(resp.status_code, body, None)
    return state, detail, dict(resp.headers)


def probe_seat(backend_url: str = CONTRACTOR_BACKEND_URL, model: str = CONTRACTOR_MODEL,
               max_retries: int = DEFAULT_MAX_RETRIES, backoff_s: float = DEFAULT_BACKOFF_S,
               sleep_fn=time.sleep) -> SeatHealth:
    """Bounded-retry probe. Retries only the transient states; a terminal
    state (proxy_down, empty_content) is returned immediately without
    retrying — retrying a dead proxy is just as wasteful as retrying a
    genuinely-exhausted account, and proxy_down/empty_content carry no
    signal that another attempt would change.
    """
    last_state, last_detail = STATE_PROXY_DOWN, "no attempt made"
    retries_used = 0
    for attempt in range(max_retries + 1):
        state, detail, headers = probe_seat_once(backend_url, model)
        last_state, last_detail = state, detail
        if state in (STATE_HEALTHY, STATE_PROXY_DOWN, STATE_EMPTY_CONTENT):
            break
        if attempt >= max_retries:
            break
        retries_used += 1
        wait = _retry_after_seconds(headers)
        if wait is None:
            wait = backoff_s * (2 ** attempt)
        sleep_fn(wait)

    return SeatHealth(state=last_state, checked_at=_now_iso(), detail=last_detail, retries_used=retries_used)


def refresh_cached_health(mem_store=None) -> SeatHealth:
    """Run one real probe and persist the verdict to mem. Call this from an
    out-of-band timer (systemd unit / cron), never from the dispatch path.
    """
    health = probe_seat()
    if mem_store is not None:
        mem_store.set(CACHE_MEM_KEY, str(health.to_dict()), tags=["lapis-pm", "contractor-seat", "health"])
    return health


def cached_health(mem_store) -> dict | None:
    """Dispatch-time read: a dictionary lookup, no inference call. Returns
    None if no canary has ever run (fail-open — a missing cache is not a
    reason to change the D2 routing decision).
    """
    row = mem_store.get(CACHE_MEM_KEY)
    if not row:
        return None
    import ast
    content = row.get("content") if isinstance(row, dict) else None
    if not content:
        return None
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return None
