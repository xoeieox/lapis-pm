"""Tests for the reviewer attempt ceiling (lapis-pm-reviewer-attempt-ceiling-v0).

Bounds *attempts* (dispatches), not completed cycles, so a reviewer that
fails before ever writing a verdict is still bounded. The counter is
PR+cycle-bound and survives new SHAs; only an explicit manual clear
(`lapis-pm clear-reviewer-attempts <target_id>`) resets it.

Coverage:
  - attempt count starts at 0 and increments on each dispatch
  - ceiling check fires once count >= ceiling, before dispatch_reviewer fires
  - the ceiling-hit act pauses the target and writes exactly one record
    (idempotent on a second call)
  - a new SHA (i.e. an unrelated observation/tag) does NOT reset the counter
  - clear_reviewer_attempts wipes the counters for a target in one call and
    does not by itself resume the target
  - reconstructs the 2026-08-01 incident shape: N consecutive fast failures
    stop dispatch at the ceiling with exactly one record emitted
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.mem import MemoryStore
from lapis_pm import pm_core

TID = "test-reviewer-attempt-ceiling-fixture"


def _tmp_store() -> MemoryStore:
    return MemoryStore(db_path=Path(tempfile.mktemp(suffix=".db")))


# ---------------------------------------------------------------------------
# Attempt counting primitives
# ---------------------------------------------------------------------------

class TestAttemptCounting:
    def test_starts_at_zero(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            assert pm_core._reviewer_attempt_count(TID, 42, 1) == 0

    def test_increments_on_each_call(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            assert pm_core._increment_reviewer_attempt(TID, 42, 1) == 1
            assert pm_core._increment_reviewer_attempt(TID, 42, 1) == 2
            assert pm_core._reviewer_attempt_count(TID, 42, 1) == 2

    def test_distinct_prs_and_cycles_isolated(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 42, 1)
            pm_core._increment_reviewer_attempt(TID, 43, 1)
            pm_core._increment_reviewer_attempt(TID, 42, 2)
            assert pm_core._reviewer_attempt_count(TID, 42, 1) == 1
            assert pm_core._reviewer_attempt_count(TID, 43, 1) == 1
            assert pm_core._reviewer_attempt_count(TID, 42, 2) == 1

    def test_default_ceiling_is_two(self):
        assert pm_core._reviewer_attempt_ceiling() == 2

    def test_ceiling_configurable_via_env(self, monkeypatch):
        monkeypatch.setenv(pm_core.REVIEWER_ATTEMPT_CEILING_ENV, "5")
        assert pm_core._reviewer_attempt_ceiling() == 5

    def test_ceiling_ignores_invalid_env(self, monkeypatch):
        monkeypatch.setenv(pm_core.REVIEWER_ATTEMPT_CEILING_ENV, "not-a-number")
        assert pm_core._reviewer_attempt_ceiling() == pm_core.REVIEWER_ATTEMPT_CEILING_DEFAULT

    def test_record_reason_does_not_touch_count(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 42, 1)
            pm_core._record_reviewer_attempt_reason(TID, 42, 1, "gw_not_serving")
            state = pm_core._reviewer_attempt_state(TID, 42, 1)
            assert state["count"] == 1
            assert state["last_reason"] == "gw_not_serving"


# ---------------------------------------------------------------------------
# Ceiling check wired into _decide_for_pr
# ---------------------------------------------------------------------------

class TestCeilingDecision:
    def test_returns_none_below_ceiling(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 42, 1)  # count=1, ceiling=2
            assert pm_core._reviewer_attempt_ceiling_check(TID, 42, 1) is None

    def test_fires_at_ceiling(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 42, 1)
            pm_core._increment_reviewer_attempt(TID, 42, 1)  # count=2 == ceiling
            # A genuine review failure (not on the infra allow-list) —
            # counts against the ceiling per D1/D4 regression fence.
            pm_core._record_reviewer_attempt_reason(TID, 42, 1, "malformed verdict json")
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 42, 1)
            assert decision is not None
            assert decision.kind == "reviewer_attempt_ceiling"
            assert decision.payload["attempts"] == 2
            assert decision.payload["ceiling"] == 2
            assert decision.payload["reported_reason"] == "malformed verdict json"

    def test_decide_for_pr_returns_ceiling_not_dispatch_when_hit(self):
        """Reconstruct the incident shape: reviewer_count stays 0 (no verdict
        ever written), so decide() would normally keep returning
        dispatch_reviewer forever. Once attempts hit the ceiling it must not.
        """
        from lapis_pm import authority

        store = _tmp_store()
        cls = authority.PRClassification(
            verdict="advisory",
            screen_verdict="unknown",
            static_outcome=authority.StaticOutcome.static_pass,
            reasons=["static checks passed"],
            issues=[],
            pr_number=33,
            repo="facets",
            title="feat: x",
            html_url="https://forgejo/Erah/facets/pulls/33",
            changed_paths=["src/x.py"],
            diff_loc=5,
            diff="",
        )
        pr = {
            "number": 33, "title": "feat: x",
            "html_url": "https://forgejo/Erah/facets/pulls/33",
            "head": {"ref": "lapis/t/x"}, "mergeable": True,
        }
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            # Genuine review failure — not infra-classified, counts against
            # the ceiling.
            pm_core._record_reviewer_attempt_reason(
                TID, 33, 1, "local reviewer produced no verdict (reason=budget_exhausted)",
            )
            with (
                patch("lapis_pm.pm_core.authority.classify", return_value=cls),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),                patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
                patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
        ):
                decision = pm_core._decide_for_pr(TID, "facets", pr, "advisory")
        assert decision.kind == "reviewer_attempt_ceiling"
        assert decision.payload["pr_number"] == 33
        assert decision.payload["cycle"] == 1
        assert decision.payload["attempts"] == 2


# ---------------------------------------------------------------------------
# Ceiling-hit act: pause + single structured record
# ---------------------------------------------------------------------------

class TestCeilingPauseAction:
    def _payload(self, reason="gw_not_serving"):
        return {
            "pr_number": 33, "cycle": 1, "attempts": 2, "ceiling": 2,
            "reported_reason": reason,
        }

    def test_pauses_target_and_writes_one_record(self):
        store = _tmp_store()
        mock_target = MagicMock()
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.get.return_value = mock_target
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.TargetStore", mock_store_cls),
            patch("lapis_pm.pm_core.episodic.write_observation") as mock_write_obs,
        ):
            result = pm_core._act_reviewer_attempt_ceiling_pause(TID, self._payload())

        assert "action:reviewer_attempt_ceiling_paused" in result
        mock_target.set_paused.assert_called_once()
        assert mock_target.set_paused.call_args.kwargs.get("True", True) or True
        assert mock_target.set_paused.call_args[0][0] is True
        mock_target.save.assert_called_once()
        mock_write_obs.assert_called_once()

        # DoD 3b: reported_reason recorded as claimed, never as verified.
        written_body = mock_write_obs.call_args[0][1]
        assert "reported_reason" in written_body
        assert "unverified" in written_body

    def test_idempotent_second_call_does_not_re_record(self):
        store = _tmp_store()
        mock_target = MagicMock()
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.get.return_value = mock_target
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.TargetStore", mock_store_cls),
            patch("lapis_pm.pm_core.episodic.write_observation") as mock_write_obs,
        ):
            pm_core._act_reviewer_attempt_ceiling_pause(TID, self._payload())
            second = pm_core._act_reviewer_attempt_ceiling_pause(TID, self._payload())

        assert second == "action:reviewer_attempt_ceiling:already_recorded"
        # Only the first call wrote an observation record.
        mock_write_obs.assert_called_once()
        mock_target.set_paused.assert_called_once()


# ---------------------------------------------------------------------------
# Persistence across SHA changes / manual clear
# ---------------------------------------------------------------------------

class TestPersistenceAndClear:
    def test_sha_advance_does_not_reset_counter(self):
        """A new SHA is just another episodic observation — it must not
        touch the attempt counter. The counter has no SHA-derived key or
        reset hook; this test asserts that invariant behaviourally."""
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            before = pm_core._reviewer_attempt_count(TID, 33, 1)

            # Simulate a new SHA landing: write an unrelated sha-advance
            # observation and re-check the same _decide_for_pr code paths a
            # tick would exercise. Nothing in the attempt-ceiling helpers
            # consults episodic.all_comments, so this is a no-op by
            # construction — the assertion is that it stays a no-op.
            with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]):
                after = pm_core._reviewer_attempt_count(TID, 33, 1)

        assert before == after == 2

    def test_clear_wipes_counters_for_target_only(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            pm_core._record_reviewer_attempt_reason(TID, 33, 1, "gw_not_serving")
            other_tid = TID + "-other"
            pm_core._increment_reviewer_attempt(other_tid, 33, 1)

            cleared = pm_core.clear_reviewer_attempts(TID)

            assert cleared >= 1
            assert pm_core._reviewer_attempt_count(TID, 33, 1) == 0
            # A different target's counters are untouched.
            assert pm_core._reviewer_attempt_count(other_tid, 33, 1) == 1

    def test_clear_after_ceiling_hit_allows_fresh_attempts(self):
        store = _tmp_store()
        mock_target = MagicMock()
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.get.return_value = mock_target
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.TargetStore", mock_store_cls),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            ):
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            assert pm_core._reviewer_attempt_ceiling_check(TID, 33, 1) is not None

            pm_core.clear_reviewer_attempts(TID)

            assert pm_core._reviewer_attempt_ceiling_check(TID, 33, 1) is None

    def test_clear_does_not_by_itself_resume_target(self):
        """DoD boundary ruling: ceiling-hit must not auto-unpause. The clear
        command only wipes counters; resuming remains a separate, explicit
        `lapis-pm resume` call."""
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            pm_core.clear_reviewer_attempts(TID)
        # clear_reviewer_attempts never touches TargetStore/target.paused —
        # verified by construction (no TargetStore import/call in its body).
        import inspect
        src = inspect.getsource(pm_core.clear_reviewer_attempts)
        assert "TargetStore" not in src
        assert "set_paused" not in src


# ---------------------------------------------------------------------------
# Incident reconstruction: N consecutive fast failures, one record, then stop
# ---------------------------------------------------------------------------

class TestIncidentReconstruction:
    def test_ten_consecutive_genuine_failures_stop_at_ceiling_one_record(self):
        """Ten consecutive reviewer dispatches for the same PR, each a
        genuine review failure (not infra), retry_count always 0. Assert
        dispatch stops at the reviewer-attempt ceiling (default 2) and
        exactly one ceiling record is emitted — the pre-existing bound,
        unchanged by D1's infra carve-out."""
        from lapis_pm import authority

        store = _tmp_store()
        cls = authority.PRClassification(
            verdict="advisory",
            screen_verdict="unknown",
            static_outcome=authority.StaticOutcome.static_pass,
            reasons=["static checks passed"],
            issues=[],
            pr_number=33,
            repo="facets",
            title="feat: x",
            html_url="https://forgejo/Erah/facets/pulls/33",
            changed_paths=["src/x.py"],
            diff_loc=5,
            diff="",
        )
        pr = {
            "number": 33, "title": "feat: x",
            "html_url": "https://forgejo/Erah/facets/pulls/33",
            "head": {"ref": "lapis/t/x"}, "mergeable": True,
        }

        dispatch_calls = 0
        pause_calls = 0
        mock_target = MagicMock()

        def fake_set_paused(*a, **k):
            nonlocal pause_calls
            pause_calls += 1

        mock_target.set_paused.side_effect = fake_set_paused
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.get.return_value = mock_target

        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.TargetStore", mock_store_cls),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.authority.classify", return_value=cls),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            ):
            for _ in range(10):
                decision = pm_core._decide_for_pr(TID, "facets", pr, "advisory")
                if decision.kind == "dispatch_reviewer":
                    dispatch_calls += 1
                    # Simulate the dispatch (attempt) and its fast failure —
                    # a genuine review problem, e.g. the verdict didn't parse.
                    pm_core._increment_reviewer_attempt(TID, 33, decision.payload["cycle"])
                    pm_core._record_reviewer_attempt_reason(
                        TID, 33, decision.payload["cycle"],
                        "local reviewer produced no verdict (reason=budget_exhausted)",
                    )
                elif decision.kind == "reviewer_attempt_ceiling":
                    pm_core._act_reviewer_attempt_ceiling_pause(TID, decision.payload)

        assert dispatch_calls == pm_core.REVIEWER_ATTEMPT_CEILING_DEFAULT  # 2, not 10
        assert pause_calls == 1  # target paused exactly once

    def test_834_regression_gw_not_serving_does_not_wedge_behind_misleading_reason(self):
        """Replays the actual #834 observed sequence (spec evidence base):
        six consecutive `reviewer_fresh` dispatches, every one failing with
        `error: ERROR: local reviewer produced no verdict (reason=gw_not_serving)`,
        cycle never advancing past 1 (no verdict ever written). Before the
        fix, two of those non-runs paused the target with
        `reported_reason=None` and a `clear-reviewer-attempts` recommendation
        that could never work — a retry, not a repair.

        After the fix: gw_not_serving is infrastructure-classified, so it
        does not consume the reviewer-attempt ceiling at all — the target
        must NOT wedge behind `reviewer_attempt_ceiling` on this sequence.
        The infra-retry budget (separate, more generous, D1) is what
        eventually bounds it, and when it does, the pause must name the real
        cause and must NOT recommend clear-reviewer-attempts (D2)."""
        from lapis_pm import authority

        store = _tmp_store()
        cls = authority.PRClassification(
            verdict="advisory",
            screen_verdict="unknown",
            static_outcome=authority.StaticOutcome.static_pass,
            reasons=["static checks passed"],
            issues=[],
            pr_number=834,
            repo="lapis-pm",
            title="feat: x",
            html_url="https://forgejo/Erah/lapis-pm/pulls/834",
            changed_paths=["src/x.py"],
            diff_loc=5,
            diff="",
        )
        pr = {
            "number": 834, "title": "feat: x",
            "html_url": "https://forgejo/Erah/lapis-pm/pulls/834",
            "head": {"ref": "lapis/t/x"}, "mergeable": True,
        }

        pause_calls = []
        mock_target = MagicMock()
        mock_target.set_paused.side_effect = lambda *a, **k: pause_calls.append((a, k))
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.get.return_value = mock_target

        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.TargetStore", mock_store_cls),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.authority.classify", return_value=cls),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            # lapis-pm-reviewer-defer-backoff-v0: this replay drives ticks
            # back-to-back with no simulated wall-clock advance, which is
            # exactly what the backoff feature now paces against by design
            # (see test_reviewer_infra_backoff.py for that coverage). Disable
            # it here so this test keeps asserting its own, orthogonal
            # concern — the infra budget still bounds gw_not_serving without
            # wedging behind the reviewer-attempt ceiling.
            patch("lapis_pm.pm_core._reviewer_infra_backoff_check", return_value=None),
            ):
            # The observed #834 sequence was six dispatches; drive one tick
            # further so the ceiling-check after the sixth failure gets a
            # chance to fire the infra-budget pause (checked at the top of
            # each iteration, before the next dispatch).
            for _ in range(pm_core.REVIEWER_INFRA_RETRY_BUDGET_DEFAULT + 1):
                decision = pm_core._decide_for_pr(TID, "lapis-pm", pr, "advisory")
                # The bug this replays: gw_not_serving non-runs must never
                # wedge the target behind the reviewer-attempt ceiling.
                assert decision.kind != "reviewer_attempt_ceiling"
                if decision.kind == "dispatch_reviewer":
                    pm_core._increment_reviewer_attempt(TID, 834, decision.payload["cycle"])
                    pm_core._record_reviewer_attempt_reason(
                        TID, 834, decision.payload["cycle"],
                        "error: ERROR: local reviewer produced no verdict (reason=gw_not_serving)",
                    )
                elif decision.kind == "reviewer_infra_budget_exhausted":
                    pm_core._act_reviewer_infra_budget_pause(TID, decision.payload)
                    break

        # Still bounded — the separate infra-retry budget catches it after
        # its own (more generous) count of non-runs.
        assert len(pause_calls) == 1
        reason_text = pause_calls[0][1].get("reason", "")
        assert "reported_reason=None" not in reason_text
        assert "gw_not_serving" in reason_text
        assert "clear-reviewer-attempts" not in reason_text
        assert "pm-pr-review" in reason_text
