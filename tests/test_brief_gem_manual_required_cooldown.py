"""Tests for the manual-required renotify cooldown (lapis-pm-containment-held-paths-v0,
Leg 5): _notify_and_observe_manual_required rate-limits repeat Pushover notifications
for a blocked decided gem, following the _DEPLOY_CURRENCY_COOLDOWN_SECS idiom.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from lapis_pm import brief_gem


def _mock_mem(cooldown_content: str | None):
    mem = MagicMock()

    def _get(key: str):
        if key.startswith("pm:brief-gem-manual-notify-cooldown:") and cooldown_content is not None:
            return {"content": cooldown_content}
        return None

    mem.get.side_effect = _get
    mem.set = MagicMock()
    return mem


def test_first_call_notifies_high_and_writes_cooldown():
    mem = _mock_mem(cooldown_content=None)
    with (
        patch("lapis_pm.pm_core._mem", return_value=mem),
        patch("lapis_pm.brief_gem.send_notification") as mock_notify,
        patch("lapis_pm.brief_gem.episodic.write_observation") as mock_obs,
    ):
        brief_gem._notify_and_observe_manual_required("tid", 42, "gem-1", "held path")

    mock_notify.assert_called_once()
    _, kw = mock_notify.call_args
    assert kw.get("priority") == brief_gem._NotifyPriority.HIGH
    mem.set.assert_called_once()
    mock_obs.assert_called_once()


def test_repeat_calls_within_cooldown_suppress_pushover_not_observation():
    """N consecutive blocked ticks within the cooldown -> 1 Pushover notification, not N.
    The episodic observation still fires on every call."""
    recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    mem = _mock_mem(cooldown_content=recent)
    with (
        patch("lapis_pm.pm_core._mem", return_value=mem),
        patch("lapis_pm.brief_gem.send_notification") as mock_notify,
        patch("lapis_pm.brief_gem.episodic.write_observation") as mock_obs,
    ):
        for _ in range(5):
            brief_gem._notify_and_observe_manual_required("tid", 42, "gem-1", "held path")

    mock_notify.assert_not_called()
    assert mock_obs.call_count == 5
    # Suppression is visible in the observation text, not silent.
    for call in mock_obs.call_args_list:
        assert "suppressed" in call.args[1].lower()


def test_cooldown_lapsed_renotifies_at_normal_priority():
    stale = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    mem = _mock_mem(cooldown_content=stale)
    with (
        patch("lapis_pm.pm_core._mem", return_value=mem),
        patch("lapis_pm.brief_gem.send_notification") as mock_notify,
        patch("lapis_pm.brief_gem.episodic.write_observation"),
    ):
        brief_gem._notify_and_observe_manual_required("tid", 42, "gem-1", "held path")

    mock_notify.assert_called_once()
    _, kw = mock_notify.call_args
    assert kw.get("priority") == brief_gem._NotifyPriority.NORMAL


def test_cooldown_read_error_fails_open_and_notifies():
    mem = MagicMock()
    mem.get.side_effect = RuntimeError("mem unavailable")
    mem.set = MagicMock()
    with (
        patch("lapis_pm.pm_core._mem", return_value=mem),
        patch("lapis_pm.brief_gem.send_notification") as mock_notify,
        patch("lapis_pm.brief_gem.episodic.write_observation"),
    ):
        brief_gem._notify_and_observe_manual_required("tid", 42, "gem-1", "held path")

    mock_notify.assert_called_once()
