"""Tests for lapis_pm.tou_window — the single source of the TOU peak window
(16:00-21:00 America/Los_Angeles, daily) used by the peak-contractor route
(lapis-pm-reviewer-peak-contractor-route-v0, D2)."""

from __future__ import annotations

from datetime import datetime

from lapis_pm.tou_window import is_peak_window, PACIFIC


def _pt(hour, minute=0):
    return datetime(2026, 8, 4, hour, minute, tzinfo=PACIFIC)


class TestIsPeakWindow:

    def test_start_of_window_is_peak(self):
        assert is_peak_window(_pt(16, 0)) is True

    def test_middle_of_window_is_peak(self):
        assert is_peak_window(_pt(18, 30)) is True

    def test_end_of_window_is_exclusive(self):
        # [16:00, 21:00) — 21:00 itself is off-peak.
        assert is_peak_window(_pt(21, 0)) is False

    def test_just_before_end_is_peak(self):
        assert is_peak_window(_pt(20, 59)) is True

    def test_before_window_is_offpeak(self):
        assert is_peak_window(_pt(15, 59)) is False

    def test_morning_is_offpeak(self):
        assert is_peak_window(_pt(9, 0)) is False

    def test_super_offpeak_hour_is_offpeak(self):
        assert is_peak_window(_pt(2, 0)) is False

    def test_naive_datetime_treated_as_local(self):
        naive = datetime(2026, 8, 4, 17, 0)
        assert is_peak_window(naive) is True

    def test_utc_datetime_converted_to_pacific(self):
        from zoneinfo import ZoneInfo
        # 2026-08-04 17:00 PT == 2026-08-05 00:00 UTC (PDT, UTC-7)
        utc_time = datetime(2026, 8, 5, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert is_peak_window(utc_time) is True

    def test_daily_no_weekend_exemption(self):
        # 2026-08-08 is a Saturday; peak still applies.
        saturday_peak = datetime(2026, 8, 8, 17, 0, tzinfo=PACIFIC)
        assert is_peak_window(saturday_peak) is True
