"""Tests for the decide() tip-of-salvage-chain PR selection
(lapis-pm-salvage-tip-repin-v0).

The salvage chain opens a NEW, higher-numbered PR on each run and the
superseded PRs stay open. The prior rule picked the LOWEST-numbered open PR
(FIFO), which latched the daemon to the oldest (already-superseded) PR and
re-dispatched fixer_retry/reviewer against its stale branch forever
(finding/lapis-pm-daemon-pins-stale-salvage-pr-2026-09-19).

The fix: `_select_actionable_tip_pr` returns the HIGHEST-numbered actionable
PR (the chain tip), and the decide() phase excludes numberless PRs from the
actionable set (loud warning) so `max` can never latch onto a PR with no
resolvable number.

Coverage (T1-T7 per spec):
  T1  salvage chain [#100, #200, #300] -> helper returns #300 (the tip)
  T2  single PR [#55] -> returns #55 (min == max, behavior unchanged)
  T3  two independent branches [#100, #200] -> returns #200 (newest; LIFO tiebreak)
  T4  tip resolved -> walk down: [#100, #200] -> returns #200
  T5  decide() integration: stale #100 + tip #300 both open/unclassified ->
      decision drives #300, NOT #100 (the re-pin regression)
  T6  empty actionable list is handled upstream (noop:no_change guard); the
      helper is never called with an empty list
  T7  numberless-PR guard: open_prs = [#100, #300, {number: None,
      head.ref: lapis/tid/ghost}] -> decision drives #300, the numberless PR
      is excluded from actionable_prs, and a warning names its head ref
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pr(number: int | None, ref: str) -> dict:
    pr: dict = {"head": {"ref": ref}}
    if number is not None:
        pr["number"] = number
    return pr


def _mock_target(pm_repo: str = "myrepo", pm_authority: str = "advisory") -> MagicMock:
    t = MagicMock()
    t.pm_bound = True
    t.paused = False
    t.paused_reason = None
    t.pm_repo = pm_repo
    t.pm_authority = pm_authority
    t.data = {}
    return t


def _tick_with_patches(target: MagicMock, extra_patches: dict | None = None):
    """Run tick("my-target") with minimal I/O patches + optional extras.

    Mirrors tests/test_tick_decision_taxonomy.py::_tick_with_patches.
    """
    extra = extra_patches or {}
    base_patches = [
        ("lapis_pm.pm_core.TargetStore", MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=target)))),
        ("lapis_pm.pm_core.get_pause_state", MagicMock(return_value="active")),
        ("lapis_pm.pm_core.set_pause_state", MagicMock()),
        ("lapis_pm.pm_core._reconcile_dispatched_with_queue", MagicMock(return_value=0)),
        ("lapis_pm.pm_core.get_cursor", MagicMock(return_value=None)),
        ("lapis_pm.episodic.since", MagicMock(return_value=[])),
        ("lapis_pm.pm_core._perceive_prs", MagicMock(return_value=([], True))),
        ("lapis_pm.pm_core._encode_user_comments", MagicMock(return_value=[])),
        ("lapis_pm.pm_core._seen_pr_ids", MagicMock(return_value=set())),
        ("lapis_pm.pm_core._encode_new_prs", MagicMock(return_value=[])),
        ("lapis_pm.pm_core._encode_pr_sha_updates", MagicMock(return_value=0)),
        ("lapis_pm.pm_core._encode_pr_body_updates", MagicMock(return_value=0)),
        ("lapis_pm.pm_core._encode_gpu_results", MagicMock(return_value=(0, []))),
        ("lapis_pm.pm_core._encode_merged_prs", MagicMock(return_value=0)),
        ("lapis_pm.pm_core._consume_brief_decisions", MagicMock(return_value=None)),
        ("lapis_pm.pm_core._is_auto_land_eligible", MagicMock(return_value=False)),
        ("lapis_pm.pm_core._persist_review_state_cache", MagicMock()),
        ("lapis_pm.pm_core.set_cursor", MagicMock()),
        ("lapis_pm.episodic.write_observation", MagicMock()),
    ]
    with ExitStack() as stack:
        for target_str, mock_val in base_patches:
            stack.enter_context(patch(target_str, mock_val))
        for target_str, mock_val in extra.items():
            stack.enter_context(patch(target_str, mock_val))
        return pm_core.tick("my-target")


def _decide_patches(open_prs: list[dict], classified: set[int] = frozenset()) -> dict:
    """Extra patches driving tick() into the `elif open_prs:` decide branch."""
    return {
        "lapis_pm.pm_core._perceive_prs": MagicMock(return_value=(open_prs, True)),
        "lapis_pm.pm_core._repo_owner": MagicMock(return_value=("myrepo", "Erah")),
        "lapis_pm.pm_core.get_open_prs": MagicMock(return_value=[]),
        "lapis_pm.pm_core._reconcile_orphan_prs": MagicMock(),
        "lapis_pm.pm_core._classified_pr_ids": MagicMock(return_value=classified),
        "lapis_pm.pm_core.authority.classify": MagicMock(return_value=MagicMock()),
        "lapis_pm.pm_core.episodic.spec_summary": MagicMock(return_value="spec"),
        "lapis_pm.pm_core._has_pending_reviewer_for_pr": MagicMock(return_value=False),
        "lapis_pm.pm_core._has_pending_fixer_for_pr": MagicMock(return_value=False),
        "lapis_pm.pm_core._reviewer_cycle_count": MagicMock(return_value=0),
        "lapis_pm.pm_core._fixer_retry_count": MagicMock(return_value=0),
    }


# ---------------------------------------------------------------------------
# T1-T4: the pure helper
# ---------------------------------------------------------------------------

class TestSelectActionableTipPr:
    def test_t1_salvage_chain_returns_tip(self):
        """T1 (the regression that motivated this unit): the helper must return
        the HIGHEST-numbered PR — the tip of the salvage chain — not the oldest
        (stale) one the FIFO min() rule latched onto."""
        actionable = [
            _make_pr(100, "lapis/tid/branch-a"),
            _make_pr(200, "lapis/tid/branch-a-salv"),
            _make_pr(300, "lapis/tid/branch-a-salv-salv"),
        ]
        pr = pm_core._select_actionable_tip_pr(actionable)
        assert pr["number"] == 300
        assert pr["head"]["ref"] == "lapis/tid/branch-a-salv-salv"

    def test_t2_single_pr_unchanged(self):
        """T2 (single PR, common case): min == max == the only PR — behavior is
        byte-for-byte unchanged for the overwhelmingly common one-open-PR case."""
        pr = pm_core._select_actionable_tip_pr([_make_pr(55, "lapis/tid/branch")])
        assert pr["number"] == 55

    def test_t3_two_independent_branches_newest_wins(self):
        """T3 (two independent branches, no prefix): the newest PR wins — an
        acceptable LIFO tiebreak for independent PRs."""
        actionable = [
            _make_pr(100, "lapis/tid/branch-x"),
            _make_pr(200, "lapis/tid/branch-y"),
        ]
        pr = pm_core._select_actionable_tip_pr(actionable)
        assert pr["number"] == 200

    def test_t4_tip_resolved_walks_down(self):
        """T4 (tip resolved -> walk down): once the tip #300 is removed (resolved
        / closed), the selection walks the chain down to the next tip, #200."""
        actionable = [
            _make_pr(100, "lapis/tid/branch-a"),
            _make_pr(200, "lapis/tid/branch-a-salv"),
        ]
        pr = pm_core._select_actionable_tip_pr(actionable)
        assert pr["number"] == 200


# ---------------------------------------------------------------------------
# T5: decide() integration — the end-to-end re-pin regression
# ---------------------------------------------------------------------------

class TestDecideTipSelectionIntegration:
    def test_t5_decide_drives_tip_not_stale_oldest(self):
        """T5 (decide() integration - the end-to-end regression): with a stale
        #100 and a tip #300 both open and unclassified, decide() must drive
        #300 (the tip), NOT #100. This proves the re-pin is resolved: the
        downstream _decide_for_pr / _has_pending_*_for_pr / reviewer-attempt
        counters (all keyed by pr_number_sel) now key off the tip."""
        target = _mock_target()
        open_prs = [
            _make_pr(100, "lapis/my-target/branch-a"),
            _make_pr(300, "lapis/my-target/branch-a-salv-salv"),
        ]
        decide_for_pr = MagicMock(
            return_value=pm_core.Decision("noop_no_change", {})
        )
        result = _tick_with_patches(target, {
            **_decide_patches(open_prs),
            "lapis_pm.pm_core._decide_for_pr": decide_for_pr,
        })
        # _decide_for_pr receives (target_id, repo, pr, pm_authority,
        # pm_verification) — the pr argument proves which PR was selected.
        assert decide_for_pr.call_count == 1
        selected_pr = decide_for_pr.call_args.args[2]
        assert selected_pr["number"] == 300, (
            f"decide() drove PR #{selected_pr['number']}, not the tip #300 — "
            "the stale-salvage re-pin is back."
        )
        assert result.skipped is False


# ---------------------------------------------------------------------------
# T6: empty actionable list is handled upstream
# ---------------------------------------------------------------------------

class TestEmptyActionableList:
    def test_t6_all_classified_noop_and_helper_never_called(self):
        """T6 (empty list is handled upstream): when every open PR is
        classified, actionable_prs is empty and the existing
        `if not actionable_prs: noop:no_change` guard fires — the helper is
        never called with an empty list (and the decision is a clean noop)."""
        target = _mock_target()
        open_prs = [_make_pr(100, "lapis/my-target/branch-a")]
        decide_for_pr = MagicMock()
        result = _tick_with_patches(target, {
            **_decide_patches(open_prs, classified={100}),
            "lapis_pm.pm_core._decide_for_pr": decide_for_pr,
        })
        assert result.decision == "noop:no_change"
        assert decide_for_pr.call_count == 0
        # The helper itself must not be reachable with an empty list: the
        # upstream guard is the only protection, so assert the contract holds.
        with pytest.raises(ValueError):
            pm_core._select_actionable_tip_pr([])

    def test_t6_no_open_prs_never_reaches_selection(self):
        """T6 (no open PRs at all): the `elif open_prs:` branch is not entered,
        so the selection path (helper included) is not reached."""
        target = _mock_target()
        result = _tick_with_patches(target, {
            **_decide_patches([]),
        })
        assert result.decision == "noop:no_change"


# ---------------------------------------------------------------------------
# T7: numberless-PR guard (the Facets amendment)
# ---------------------------------------------------------------------------

class TestNumberlessPrGuard:
    def test_t7_numberless_pr_excluded_warning_and_tip_selected(self, caplog):
        """T7 (numberless-PR guard - the Facets amendment): with
        open_prs = [#100 branch A, #300 branch A-salv-salv,
        {number: None, head.ref: "lapis/tid/ghost"}], the decision drives
        #300 (the highest-numbered *numbered* PR), the numberless PR is
        excluded from actionable_prs, and a warning is logged naming the
        numberless PR's head ref (lapis/tid/ghost).

        This is the regression for the silent-failure mode where `max` + the
        old `p.get("number", 1 << 30)` default would have pinned the daemon to
        the numberless PR (the default made it the max)."""
        target = _mock_target()
        open_prs = [
            _make_pr(100, "lapis/my-target/branch-a"),
            _make_pr(300, "lapis/my-target/branch-a-salv-salv"),
            _make_pr(None, "lapis/tid/ghost"),
        ]
        decide_for_pr = MagicMock(
            return_value=pm_core.Decision("noop_no_change", {})
        )
        # Spy on the helper to prove it is never handed a numberless PR.
        real_helper = pm_core._select_actionable_tip_pr
        seen_inputs: list[list[dict]] = []

        def _spy(actionable_prs):
            seen_inputs.append(actionable_prs)
            return real_helper(actionable_prs)

        with caplog.at_level(logging.WARNING, logger="lapis_pm.pm_core"):
            result = _tick_with_patches(target, {
                **_decide_patches(open_prs),
                "lapis_pm.pm_core._decide_for_pr": decide_for_pr,
                "lapis_pm.pm_core._select_actionable_tip_pr": _spy,
            })

        # The numberless PR was never passed to the helper.
        assert seen_inputs, "helper was not called"
        for actionable in seen_inputs:
            assert all(p.get("number") is not None for p in actionable), (
                f"helper received a numberless PR: {actionable}"
            )

        # The decision drove #300 (the tip), not #100 and not the numberless PR.
        assert decide_for_pr.call_count == 1
        selected_pr = decide_for_pr.call_args.args[2]
        assert selected_pr["number"] == 300

        # A loud warning names the numberless PR's head ref.
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "lapis/tid/ghost" in r.getMessage()
        ]
        assert warnings, (
            "expected a WARNING naming the numberless PR head ref "
            f"'lapis/tid/ghost'; got: {[r.getMessage() for r in caplog.records]}"
        )
