"""Tests for lapis-pm-reviewer-attempts-not-consumed-by-infra-v0.

Root cause (established live 2026-08-03, PR #834): every reviewer_fresh
dispatch failed with `error: ERROR: local reviewer produced no verdict
(reason=gw_not_serving)`. The reviewer never ran — GravityWell was reported
unavailable — yet each non-run consumed a reviewer attempt, and two consumed
attempts paused the target with a misleading `reported_reason=None` and a
`clear-reviewer-attempts` recommendation that could not work.

Coverage (D4):
  - gw_not_serving does NOT increment the ceiling-relevant (non-infra) count
  - a genuine review failure DOES increment it (regression fence)
  - repeated infrastructure failures are still bounded (separate infra budget)
  - a ceiling/infra pause never renders reported_reason=None — falls back to
    the dispatch record's `error` field when no per-attempt reason exists
  - an infrastructure-caused pause does not recommend clear-reviewer-attempts
  - any ceiling/infra pause names the pm-pr-review brief path
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents_core.mem import MemoryStore
from lapis_pm import pm_core

TID = "test-reviewer-infra-non-run-fixture"


def _tmp_store() -> MemoryStore:
    return MemoryStore(db_path=Path(tempfile.mktemp(suffix=".db")))


# ---------------------------------------------------------------------------
# Classification (D1)
# ---------------------------------------------------------------------------

class TestClassification:
    def test_gw_not_serving_classified_infra(self):
        assert pm_core._classify_reviewer_infra_reason(
            "ERROR: local reviewer produced no verdict (reason=gw_not_serving)"
        ) == "gw_not_serving"

    def test_gw_unreachable_classified_infra(self):
        assert pm_core._classify_reviewer_infra_reason(
            "ERROR: local reviewer produced no verdict (reason=gw_unreachable)"
        ) == "gw_unreachable"

    def test_gw_deferred_swarm_not_classified(self):
        """D1: gw_deferred_swarm belongs to the llm.py operator-path
        vocabulary (GW_PROVENANCE_PRECEDENCE), not the reviewer's actual call
        path (shaped_runner -> gw_agent.call_gw_agent). It can never reach a
        reviewer dispatch record, so it must NOT be treated as infra here —
        importing that vocabulary would classify against reasons that can
        never arrive."""
        assert pm_core._classify_reviewer_infra_reason(
            "reason=gw_deferred_swarm"
        ) is None

    def test_genuine_failure_not_classified(self):
        assert pm_core._classify_reviewer_infra_reason(
            "verdict JSON failed to parse"
        ) is None

    def test_none_reason_not_classified(self):
        assert pm_core._classify_reviewer_infra_reason(None) is None

    def test_unrecognised_reason_fails_closed(self):
        """An unrecognised reason counts against the ceiling — the allow-list
        is hardcoded and explicit, not an open vocabulary."""
        assert pm_core._classify_reviewer_infra_reason(
            "reason=some_new_failure_mode_nobody_has_seen_yet"
        ) is None


# ---------------------------------------------------------------------------
# Ceiling accounting: infra attempts excluded, genuine failures counted (D4)
# ---------------------------------------------------------------------------

class TestCeilingAccounting:
    def test_gw_not_serving_does_not_increment_ceiling_relevant_count(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 1, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 1, 1, "ERROR: local reviewer produced no verdict (reason=gw_not_serving)",
            )
            pm_core._increment_reviewer_attempt(TID, 1, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 1, 1, "ERROR: local reviewer produced no verdict (reason=gw_not_serving)",
            )
            # Two dispatches, both infra non-runs — the reviewer-attempt
            # ceiling (default 2) must NOT have been reached.
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 1, 1)
            assert decision is None or decision.kind != "reviewer_attempt_ceiling"

    def test_genuine_review_failure_does_increment_ceiling_relevant_count(self):
        """Regression fence: a real review failure must still be bounded by
        the reviewer-attempt ceiling exactly as before."""
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 2, 1)
            pm_core._record_reviewer_attempt_reason(TID, 2, 1, "verdict JSON failed to parse")
            pm_core._increment_reviewer_attempt(TID, 2, 1)
            pm_core._record_reviewer_attempt_reason(TID, 2, 1, "verdict JSON failed to parse")
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 2, 1)
            assert decision is not None
            assert decision.kind == "reviewer_attempt_ceiling"
            assert decision.payload["attempts"] == 2

    def test_mixed_infra_and_genuine_only_genuine_counts(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 3, 1)
            pm_core._record_reviewer_attempt_reason(TID, 3, 1, "reason=gw_not_serving")
            pm_core._increment_reviewer_attempt(TID, 3, 1)
            pm_core._record_reviewer_attempt_reason(TID, 3, 1, "reason=gw_not_serving")
            pm_core._increment_reviewer_attempt(TID, 3, 1)
            pm_core._record_reviewer_attempt_reason(TID, 3, 1, "verdict JSON failed to parse")
            # Only the third (genuine) attempt should count — 1 < ceiling(2).
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 3, 1)
            assert decision is None

    def test_infra_retry_budget_configurable_via_env(self, monkeypatch):
        monkeypatch.setenv(pm_core.REVIEWER_INFRA_RETRY_BUDGET_ENV, "3")
        assert pm_core._reviewer_infra_retry_budget() == 3

    def test_infra_retry_budget_default_more_generous_than_ceiling(self):
        assert pm_core._reviewer_infra_retry_budget() > pm_core._reviewer_attempt_ceiling()

    def test_repeated_infra_failures_still_bounded(self):
        """Infra non-runs are exempt from the reviewer-attempt ceiling but
        must still be bounded by the separate infra-retry budget — an
        endlessly-absent GravityWell must not produce endless dispatches."""
        store = _tmp_store()
        budget = pm_core._reviewer_infra_retry_budget()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            decision = None
            for i in range(budget + 2):  # drive well past the budget
                decision = pm_core._reviewer_attempt_ceiling_check(TID, 4, 1)
                if decision is not None:
                    break
                pm_core._increment_reviewer_attempt(TID, 4, 1)
                pm_core._record_reviewer_attempt_reason(TID, 4, 1, "reason=gw_not_serving")
            assert decision is not None
            assert decision.kind == "reviewer_infra_budget_exhausted"
            assert decision.payload["infra_attempts"] == budget


# ---------------------------------------------------------------------------
# D3: never render reported_reason=None
# ---------------------------------------------------------------------------

class TestNullReasonImpossible:
    def test_ceiling_pause_falls_back_to_dispatch_record_error(self):
        store = _tmp_store()
        mock_target = MagicMock()
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.get.return_value = mock_target

        dispatched_records = [
            {
                "agent_type": "reviewer_fresh", "pr_number": 834, "cycle": 1,
                "status": "failed",
                "error": "ERROR: local reviewer produced no verdict (reason=gw_not_serving)",
            },
        ]
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.TargetStore", mock_store_cls),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=dispatched_records),
        ):
            payload = {
                "pr_number": 834, "cycle": 1, "attempts": 2, "ceiling": 2,
                "reported_reason": None,  # never populated per-attempt
            }
            pm_core._act_reviewer_attempt_ceiling_pause(TID, payload)

        reason_text = mock_target.set_paused.call_args[1]["reason"]
        assert "reported_reason=None" not in reason_text
        assert "gw_not_serving" in reason_text

    def test_ceiling_pause_with_no_reason_anywhere_still_avoids_bare_none(self):
        store = _tmp_store()
        mock_target = MagicMock()
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.get.return_value = mock_target
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.TargetStore", mock_store_cls),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        ):
            payload = {
                "pr_number": 1, "cycle": 1, "attempts": 2, "ceiling": 2,
                "reported_reason": None,
            }
            pm_core._act_reviewer_attempt_ceiling_pause(TID, payload)
        reason_text = mock_target.set_paused.call_args[1]["reason"]
        assert "reported_reason=None" not in reason_text


# ---------------------------------------------------------------------------
# D2: infra pause does not recommend clear-reviewer-attempts; always names
# the pm-pr-review brief escape
# ---------------------------------------------------------------------------

class TestPauseMessaging:
    def test_infra_pause_names_cause_and_omits_clear_recommendation(self):
        store = _tmp_store()
        mock_target = MagicMock()
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.get.return_value = mock_target
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.TargetStore", mock_store_cls),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        ):
            payload = {
                "pr_number": 834, "cycle": 1,
                "infra_attempts": 6, "infra_budget": 6,
                "reported_reason": "ERROR: local reviewer produced no verdict (reason=gw_not_serving)",
            }
            pm_core._act_reviewer_infra_budget_pause(TID, payload)
        reason_text = mock_target.set_paused.call_args[1]["reason"]
        assert "gw_not_serving" in reason_text
        assert "clear-reviewer-attempts" not in reason_text
        assert "pm-pr-review" in reason_text

    def test_genuine_ceiling_pause_still_names_pm_pr_review(self):
        """Every ceiling pause — infra or genuine — must name the
        human-judgment escape that needs no automated verdict."""
        store = _tmp_store()
        mock_target = MagicMock()
        mock_store_cls = MagicMock()
        mock_store_cls.return_value.get.return_value = mock_target
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.TargetStore", mock_store_cls),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        ):
            payload = {
                "pr_number": 42, "cycle": 1, "attempts": 2, "ceiling": 2,
                "reported_reason": "verdict JSON failed to parse",
            }
            pm_core._act_reviewer_attempt_ceiling_pause(TID, payload)
        reason_text = mock_target.set_paused.call_args[1]["reason"]
        assert "pm-pr-review" in reason_text
        # A genuine failure IS a legitimate case for the clear command.
        assert "clear-reviewer-attempts" in reason_text


# ---------------------------------------------------------------------------
# D3, verified against real transition ordering (Council open question 3)
# ---------------------------------------------------------------------------

class TestReasonPopulationOrdering:
    def test_reconcile_records_reason_when_no_output_file_ever_exists(self):
        """The actual #834 timing bug: a reviewer job that crashes before
        writing an output file gets flipped pending -> failed by the
        queue-reconcile path (_reconcile_dispatched_with_queue), which runs
        BEFORE _encode_gpu_results in the same tick. Once flipped, encode's
        `if status != pending: continue` skips it, so its is_failure branch
        (the only other _record_reviewer_attempt_reason call site) never
        runs. The reconcile path itself must record the reason, using the
        queue's `error` field."""
        store = _tmp_store()
        rec = {
            "gpu_id": "gpu-834", "agent_type": "reviewer_fresh",
            "pr_number": 834, "cycle": 1, "status": "pending",
            "ts": "2026-08-03T14:04:00Z",
        }

        class _FakeQueue:
            def get_recent_failed(self, limit=50):
                return [{
                    "id": "gpu-834",
                    "error": "ERROR: local reviewer produced no verdict (reason=gw_not_serving)",
                    "completed_at": "2026-08-03T14:04:30Z",
                }]

            def get_recent_completed(self, limit=50):
                return []

        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core._ClaudeQueue", _FakeQueue),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]),
            patch("lapis_pm.pm_core.episodic.write_observation"),
        ):
            pm_core._reconcile_dispatched_with_queue(TID)
            state = pm_core._reviewer_attempt_state(TID, 834, 1)

        assert state["last_reason"] == (
            "ERROR: local reviewer produced no verdict (reason=gw_not_serving)"
        )
        assert state["infra_count"] == 1
