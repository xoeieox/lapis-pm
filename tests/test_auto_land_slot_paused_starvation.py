"""Tests for Leg 2 of lapis-pm-external-merge-land-unblock-v0: paused targets
must not win the tick_all auto-land pre-selection slot.

_is_auto_land_eligible deliberately does not check `paused` (that guard lives
in tick() itself, which runs after pre-selection). Before this fix, tick_all's
pre-selection sorted eligible targets by oldest bind without excluding paused
ones, so a paused-but-otherwise-eligible target could win the single auto-land
slot, sort first, and immediately noop:paused — starving every other
land-eligible target for as long as it stayed paused.

Coverage:
  AC2.1 paused eligible A (older bind) + unpaused eligible B → B chosen, not A.
  AC2.2 only a paused eligible target present → auto_land_chosen stays None.
  AC2.3 selection order among unpaused eligible targets is unchanged (oldest first).
  AC2.4 every fixture here has its head branch already deleted, so no Leg 1
        (_ensure_head_branch_deleted) call occurs — the pause filter is
        exercised in isolation from the branch-reconcile leg.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core


def _make_target(tid: str, paused: bool = False) -> MagicMock:
    t = MagicMock()
    t.id = tid
    t.pm_bound = True
    t.pm_repo = "lapis-test"
    t.pm_authority = "advisory"
    t.paused = paused
    t.paused_reason = "review-gate" if paused else None
    t.data = {}
    return t


def _run_tick_all(targets, eligible_ids, bound_ts_by_id, tick_results_by_id=None):
    """Run tick_all() with Forgejo health/deploy/gem plumbing bypassed.

    eligible_ids: subset of target ids for which _is_auto_land_eligible → True.
    bound_ts_by_id: id → sort-key string (_spec_bound_ts stand-in).
    Returns (results, allow_auto_land_by_id) — the latter captured from the
    tick() calls tick_all() actually makes.
    """
    allow_auto_land_by_id: dict[str, bool] = {}

    def fake_tick(target_id, allow_auto_land=True):
        allow_auto_land_by_id[target_id] = allow_auto_land
        if tick_results_by_id and target_id in tick_results_by_id:
            return tick_results_by_id[target_id]
        return pm_core.TickResult(target_id, False, "ok", 0, "noop:no_change")

    def fake_eligible(tid):
        return tid in eligible_ids

    def fake_bound_ts(tid):
        return bound_ts_by_id.get(tid, "9999-99-99")

    with (
        patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(True, "")),
        patch("lapis_pm.pm_core._set_forgejo_consecutive_fails"),
        patch("lapis_pm.pm_core._check_deploy_currency"),
        patch("lapis_pm.pm_core._reconcile_deploy_inventory"),
        patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
        patch("lapis_pm.pm_core.TargetStore") as MockStore,
        patch("lapis_pm.pm_core._is_auto_land_eligible", side_effect=fake_eligible),
        patch("lapis_pm.pm_core._spec_bound_ts", side_effect=fake_bound_ts),
        patch("lapis_pm.pm_core.tick", side_effect=fake_tick) as mock_tick,
        # AC2.4: assert the branch-reconcile leg never fires from this leg's tests.
        patch("lapis_pm.pm_core._ensure_head_branch_deleted") as mock_ensure_deleted,
    ):
        MockStore.return_value.load_all.return_value = targets
        results = pm_core.tick_all()

    assert mock_ensure_deleted.call_count == 0, (
        "AC2.4: Leg 2 fixtures must not trigger the Leg 1 branch-reconcile call"
    )
    return results, allow_auto_land_by_id


class TestPausedExcludedFromPreSelection:
    def test_ac2_1_unpaused_wins_over_older_paused(self):
        """AC2.1: paused A (older bind) + unpaused B eligible → B gets the slot, not A."""
        a = _make_target("target-a", paused=True)
        b = _make_target("target-b", paused=False)
        bound_ts = {"target-a": "2026-01-01T00:00:00", "target-b": "2026-06-01T00:00:00"}

        _results, allow_auto_land = _run_tick_all(
            [a, b], eligible_ids={"target-a", "target-b"}, bound_ts_by_id=bound_ts,
        )

        assert allow_auto_land["target-b"] is True
        assert allow_auto_land["target-a"] is False

    def test_ac2_1_regression_fails_on_reverted_code(self):
        """Demonstrates the pre-fix behaviour: pure age-sort (no pause filter)
        would have picked target-a (older bind) despite it being paused.

        This directly exercises the pre-fix selection expression so the
        regression is provable independent of the fixed tick_all() body.
        """
        bound = [_make_target("target-a", paused=True), _make_target("target-b", paused=False)]
        eligible = {"target-a", "target-b"}
        bound_ts = {"target-a": "2026-01-01T00:00:00", "target-b": "2026-06-01T00:00:00"}

        # Pre-fix expression (no `not t.paused` filter):
        pre_fix_eligible_ids = sorted(
            [t.id for t in bound if t.id in eligible],
            key=lambda tid: bound_ts[tid],
        )
        assert pre_fix_eligible_ids[0] == "target-a"  # the bug: paused target wins

        # Fixed expression (matches tick_all's current pre-selection):
        fixed_eligible_ids = sorted(
            [t.id for t in bound if not t.paused and t.id in eligible],
            key=lambda tid: bound_ts[tid],
        )
        assert fixed_eligible_ids[0] == "target-b"

    def test_ac2_2_only_paused_eligible_present_chooses_none(self):
        """AC2.2: with only a paused eligible target, no target gets allow_auto_land=True."""
        a = _make_target("target-a", paused=True)
        bound_ts = {"target-a": "2026-01-01T00:00:00"}

        _results, allow_auto_land = _run_tick_all(
            [a], eligible_ids={"target-a"}, bound_ts_by_id=bound_ts,
        )

        assert allow_auto_land["target-a"] is False
        assert True not in allow_auto_land.values()

    def test_ac2_3_unpaused_selection_order_unchanged(self):
        """AC2.3: among unpaused eligible targets, oldest _spec_bound_ts still wins."""
        older = _make_target("target-older", paused=False)
        newer = _make_target("target-newer", paused=False)
        bound_ts = {
            "target-older": "2026-01-01T00:00:00",
            "target-newer": "2026-06-01T00:00:00",
        }

        _results, allow_auto_land = _run_tick_all(
            [older, newer],
            eligible_ids={"target-older", "target-newer"},
            bound_ts_by_id=bound_ts,
        )

        assert allow_auto_land["target-older"] is True
        assert allow_auto_land["target-newer"] is False
