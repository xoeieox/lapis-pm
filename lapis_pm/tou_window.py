"""Single source of truth for the TOU peak window (`tou-peak-window`).

Erah's TOU electricity plan carries a PEAK tier, 16:00-21:00 America/Los_Angeles,
every day, no weekend exemption (infra/tou-peak-window-guardrail-2026-06-27).
GravityWell is asleep for the duration by construction, so any consumer that
needs to know "is it peak right now" imports `is_peak_window()` from here
rather than re-hardcoding the hours.

As of 2026-08-04 no other module in agents-core or lapis-pm owns this
constant (agents-core's `tou-peak-sleep-guard-v0` spec proposes an
`is_offpeak()`/`tou_phase()` helper but has not landed one). Per
lapis-pm-reviewer-peak-contractor-route-v0 D2: "If no module owns it, that
is worth one shared constant, not two." This module is that constant. If
agents-core lands its own owner, re-point this module to re-export from
there rather than keeping two copies in sync by hand.
"""

from __future__ import annotations

from datetime import datetime, time as _time
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")

# 16:00-21:00 PT, daily, no weekend exemption.
PEAK_START = _time(16, 0)
PEAK_END = _time(21, 0)


def is_peak_window(now: datetime | None = None) -> bool:
    """True when `now` (default: current time) falls in the TOU peak window.

    `now` may be naive (assumed already Pacific-local) or tz-aware (converted
    to Pacific). The window is a plain half-open [16:00, 21:00) local-clock
    interval, daily.
    """
    if now is None:
        now = datetime.now(PACIFIC)
    elif now.tzinfo is not None:
        now = now.astimezone(PACIFIC)
    local_time = now.time()
    return PEAK_START <= local_time < PEAK_END
