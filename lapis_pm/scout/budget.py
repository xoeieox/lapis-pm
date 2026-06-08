"""NightBudget — safe stop conditions for Scout night runs.

Replaces _resolve_until_epoch's next-day roll with a bounded, fail-safe budget.
A run ends when either stop condition is met; in-flight unit work always finishes.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime


DEFAULT_DEADLINE_HOURS = 4


@dataclass
class NightBudget:
    """Hard deadline + optional unit ceiling.

    deadline: Unix epoch, never rolls to tomorrow.
    max_units: if set, stop after this many units regardless of time.
    """
    deadline: float
    max_units: int | None = None

    def is_expired(self, now: float | None = None, units_run: int = 0) -> bool:
        t = now if now is not None else time.time()
        if t >= self.deadline:
            return True
        if self.max_units is not None and units_run >= self.max_units:
            return True
        return False

    def deadline_in_past(self, now: float | None = None) -> bool:
        t = now if now is not None else time.time()
        return t >= self.deadline

    def remaining_seconds(self, now: float | None = None) -> float:
        t = now if now is not None else time.time()
        return self.deadline - t

    @classmethod
    def from_duration(
        cls, hours: float, max_units: int | None = None, now: float | None = None
    ) -> "NightBudget":
        t = now if now is not None else time.time()
        return cls(deadline=t + hours * 3600, max_units=max_units)

    @classmethod
    def from_hhmm(
        cls, hhmm: str, max_units: int | None = None, now: float | None = None
    ) -> "NightBudget":
        """Today-only: if HHMM is already past, deadline = now (immediate no-op)."""
        now_ts = now if now is not None else time.time()
        dt_now = datetime.fromtimestamp(now_ts)
        hh, mm = int(hhmm[:2]), int(hhmm[2:])
        target = dt_now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target.timestamp() <= now_ts:
            # Past — expire immediately; no next-day roll
            return cls(deadline=now_ts, max_units=max_units)
        return cls(deadline=target.timestamp(), max_units=max_units)

    @classmethod
    def default(cls, max_units: int | None = None, now: float | None = None) -> "NightBudget":
        return cls.from_duration(DEFAULT_DEADLINE_HOURS, max_units=max_units, now=now)

    @classmethod
    def from_cli(
        cls,
        until_hhmm: str | None = None,
        deadline_in: str | None = None,
        max_units: int | None = None,
        now: float | None = None,
    ) -> "NightBudget":
        """Build from CLI flags. Priority: --deadline-in > --until > default 4h."""
        if deadline_in is not None:
            hours = _parse_duration_hours(deadline_in)
            return cls.from_duration(hours, max_units=max_units, now=now)
        if until_hhmm is not None:
            return cls.from_hhmm(until_hhmm, max_units=max_units, now=now)
        return cls.default(max_units=max_units, now=now)


def _parse_duration_hours(s: str) -> float:
    """Parse '4h', '30m', '1h30m', '90m' → hours as float."""
    s = s.strip().lower()
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?", s)
    if not m or not s or (not m.group(1) and not m.group(2)):
        raise ValueError(
            f"Cannot parse duration {s!r}. Expected format like '4h', '30m', '1h30m'."
        )
    hours = float(m.group(1) or 0)
    minutes = float(m.group(2) or 0)
    total = hours + minutes / 60
    if total <= 0:
        raise ValueError(f"Duration must be positive, got: {s!r}")
    return total
