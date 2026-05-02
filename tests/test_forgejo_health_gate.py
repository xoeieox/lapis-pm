"""Unit tests for the Forgejo health gate (lapis-pm-forgejo-health-gate spec).

Coverage:
  (a) probe success → normal tick (tick_all proceeds, targets processed)
  (b) probe failure → all targets skipped, unreachable log line printed, cursors not advanced
  (c) three consecutive failures → Pushover emitted exactly once
  (d) recovery after failures → consecutive-fail counter reset to 0
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(tid: str, bound: bool = True) -> MagicMock:
    t = MagicMock()
    t.id = tid
    t.pm_bound = bound
    t.pm_repo = "lapis-test"
    t.pm_authority = "advisory"
    t.paused = False
    t.paused_reason = None
    t.data = {}
    return t


def _mock_mem_store(initial_fails: int = 0):
    """Return a dict-backed MemoryStore mock."""
    store: dict[str, str] = {}
    if initial_fails:
        store[pm_core.FORGEJO_CONSECUTIVE_FAILS_KEY] = str(initial_fails)

    def _get(key):
        if key in store:
            return {"content": store[key]}
        return None

    def _set(key, value, tags=None):
        store[key] = value

    def _delete(key):
        store.pop(key, None)

    m = MagicMock()
    m.get.side_effect = _get
    m.set.side_effect = _set
    m.delete.side_effect = _delete
    return m, store


# ---------------------------------------------------------------------------
# (a) Probe success → normal tick proceeds
# ---------------------------------------------------------------------------

class TestProbeSuccess:

    def test_tick_all_proceeds_when_probe_succeeds(self, capsys):
        """When probe returns True, tick_all runs normally and resets fail counter."""
        mem_mock, mem_store = _mock_mem_store(initial_fails=1)

        target = _make_target("tid-a")
        tick_result = pm_core.TickResult("tid-a", False, "ok", 0, "noop")

        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(True, "")),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core.tick", return_value=tick_result) as mock_tick,
            patch("lapis_pm.pm_core._is_auto_land_eligible", return_value=False),
            patch("lapis_pm.pm_core._spec_bound_ts", return_value=""),
        ):
            MockStore.return_value.load_all.return_value = [target]
            results = pm_core.tick_all()

        # tick() was called for the target
        mock_tick.assert_called_once()
        assert results == [tick_result]

        # Consecutive-fail counter reset to 0
        assert mem_store.get(pm_core.FORGEJO_CONSECUTIVE_FAILS_KEY) == "0"

        # No unreachable line printed
        captured = capsys.readouterr()
        assert "[forgejo:unreachable]" not in captured.out


# ---------------------------------------------------------------------------
# (b) Probe failure → skip + log, cursors not advanced
# ---------------------------------------------------------------------------

class TestProbeFailure:

    def test_all_targets_skipped_on_unreachable(self, capsys):
        """Probe failure skips all targets and logs [forgejo:unreachable]."""
        mem_mock, mem_store = _mock_mem_store()

        targets = [_make_target("tid-1"), _make_target("tid-2")]

        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(False, "connect_error")),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core.tick") as mock_tick,
        ):
            MockStore.return_value.load_all.return_value = targets
            results = pm_core.tick_all()

        # tick() must NOT be called for any target
        mock_tick.assert_not_called()

        # All results are skipped with the right decision tag
        assert len(results) == 2
        for r in results:
            assert r.skipped is True
            assert r.reason == "forgejo_unreachable"
            assert r.encoded == 0
            assert r.decision == "skipped:forgejo_unreachable"

        # [forgejo:unreachable] line printed with reason
        captured = capsys.readouterr()
        assert "[forgejo:unreachable] reason=connect_error" in captured.out

    def test_cursors_not_advanced_on_unreachable(self):
        """Cursors must not be written to mem on a skipped tick."""
        mem_mock, mem_store = _mock_mem_store()

        target = _make_target("tid-cursor")

        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(False, "timeout")),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
        ):
            MockStore.return_value.load_all.return_value = [target]
            pm_core.tick_all()

        # Cursor key must not appear in mem writes
        cursor_key = f"pm/cursor/tid-cursor"
        assert cursor_key not in mem_store

    def test_consecutive_fail_counter_increments(self):
        """Each failed probe increments the consecutive-fail counter."""
        mem_mock, mem_store = _mock_mem_store(initial_fails=1)

        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(False, "http=503")),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
        ):
            MockStore.return_value.load_all.return_value = []
            pm_core.tick_all()

        assert mem_store[pm_core.FORGEJO_CONSECUTIVE_FAILS_KEY] == "2"

    def test_unreachable_reason_http_code(self, capsys):
        """HTTP error codes are formatted as http=<code> in the log line."""
        mem_mock, _ = _mock_mem_store()

        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(False, "http=503")),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
        ):
            MockStore.return_value.load_all.return_value = []
            pm_core.tick_all()

        captured = capsys.readouterr()
        assert "[forgejo:unreachable] reason=http=503" in captured.out


# ---------------------------------------------------------------------------
# (c) Three consecutive failures → Pushover emitted exactly once
# ---------------------------------------------------------------------------

class TestPushoverPolicy:

    def test_pushover_emitted_on_third_consecutive_failure(self):
        """Pushover fires exactly when consecutive fails reaches threshold (3)."""
        # Start at 2 fails — the next failure should trigger the notification.
        mem_mock, _ = _mock_mem_store(initial_fails=2)

        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(False, "connect_error")),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core._notify_forgejo_unreachable") as mock_notify,
        ):
            MockStore.return_value.load_all.return_value = []
            pm_core.tick_all()

        mock_notify.assert_called_once()

    def test_pushover_not_emitted_before_threshold(self):
        """Pushover must NOT fire on the first or second consecutive failure."""
        for initial_fails in (0, 1):
            mem_mock, _ = _mock_mem_store(initial_fails=initial_fails)

            with (
                patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(False, "connect_error")),
                patch("lapis_pm.pm_core._mem", return_value=mem_mock),
                patch("lapis_pm.pm_core.TargetStore") as MockStore,
                patch("lapis_pm.pm_core._notify_forgejo_unreachable") as mock_notify,
            ):
                MockStore.return_value.load_all.return_value = []
                pm_core.tick_all()

            mock_notify.assert_not_called(), f"Pushover fired at initial_fails={initial_fails}"

    def test_pushover_not_emitted_again_after_threshold(self):
        """Pushover must NOT fire again on the 4th, 5th, … failure (only on the 3rd)."""
        mem_mock, _ = _mock_mem_store(initial_fails=3)  # already at threshold

        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(False, "connect_error")),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core._notify_forgejo_unreachable") as mock_notify,
        ):
            MockStore.return_value.load_all.return_value = []
            pm_core.tick_all()

        # fails is now 4, which is != FORGEJO_UNREACHABLE_THRESHOLD (3)
        mock_notify.assert_not_called()


# ---------------------------------------------------------------------------
# (d) Recovery clears the consecutive-fail counter
# ---------------------------------------------------------------------------

class TestRecovery:

    def test_recovery_resets_consecutive_fail_counter(self):
        """A successful probe after failures resets the counter to 0."""
        mem_mock, mem_store = _mock_mem_store(initial_fails=5)

        target = _make_target("tid-rec")
        tick_result = pm_core.TickResult("tid-rec", False, "ok", 0, "noop")

        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(True, "")),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core.tick", return_value=tick_result),
            patch("lapis_pm.pm_core._is_auto_land_eligible", return_value=False),
            patch("lapis_pm.pm_core._spec_bound_ts", return_value=""),
        ):
            MockStore.return_value.load_all.return_value = [target]
            pm_core.tick_all()

        assert mem_store[pm_core.FORGEJO_CONSECUTIVE_FAILS_KEY] == "0"

    def test_recovery_allows_new_pushover_cycle(self):
        """After recovery (counter reset), three more failures trigger Pushover again."""
        mem_mock, mem_store = _mock_mem_store(initial_fails=0)

        # Simulate recovery (counter already 0) then two failures, then third
        # — just test that the third failure in a fresh run triggers notification.
        # Set counter to 2 (as if we just had two fresh failures post-recovery).
        mem_store[pm_core.FORGEJO_CONSECUTIVE_FAILS_KEY] = "2"

        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(False, "connect_error")),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core._notify_forgejo_unreachable") as mock_notify,
        ):
            MockStore.return_value.load_all.return_value = []
            pm_core.tick_all()

        mock_notify.assert_called_once()


# ---------------------------------------------------------------------------
# probe_forgejo_health unit tests
# ---------------------------------------------------------------------------

class TestProbeForgejoHealth:

    def test_returns_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            ok, reason = pm_core.probe_forgejo_health()
        assert ok is True
        assert reason == ""

    def test_returns_false_on_non_2xx(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("httpx.get", return_value=mock_resp):
            ok, reason = pm_core.probe_forgejo_health()
        assert ok is False
        assert reason == "http=503"

    def test_returns_false_on_timeout(self):
        import httpx
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            ok, reason = pm_core.probe_forgejo_health()
        assert ok is False
        assert reason == "timeout"

    def test_returns_false_on_connect_error(self):
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            ok, reason = pm_core.probe_forgejo_health()
        assert ok is False
        assert reason == "connect_error"
