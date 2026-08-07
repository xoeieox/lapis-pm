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


# ---------------------------------------------------------------------------
# lapis-pm-reviewer-defer-timeout-is-infra-v0: gw_defer_timeout completes the
# allow-list. A reviewer refused GPU admission (doorman defer, client retry
# budget exhausted) never ran — it's a capacity non-run, not a reviewer
# failure, and should be classified identically to gw_not_serving/
# gw_unreachable.
# ---------------------------------------------------------------------------

class TestDeferTimeoutClassification:
    def test_verbatim_shaped_runner_string_classified_infra(self):
        """DoD 2: the verbatim string shaped_runner.py:558 emits, not a
        paraphrase — the regex at pm_core.py:3538(ish) is what's under test."""
        assert pm_core._classify_reviewer_infra_reason(
            "ERROR: local reviewer produced no verdict (reason=gw_defer_timeout)"
        ) == "gw_defer_timeout"

    def test_bare_substring_fallback_also_matches(self):
        """DoD 3: the reason can also arrive without the `reason=` prefix —
        straight from a queue `error` field, the shape pm_core.py:6488
        records."""
        assert pm_core._classify_reviewer_infra_reason(
            "gw_defer_timeout"
        ) == "gw_defer_timeout"


class TestDeferTimeoutCeilingAccounting:
    def test_consecutive_defer_timeouts_below_infra_budget_do_not_ceiling(self):
        """DoD 4: N consecutive gw_defer_timeout attempts, N below the infra
        budget, must not trip reviewer_attempt_ceiling."""
        store = _tmp_store()
        budget = pm_core._reviewer_infra_retry_budget()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            for _ in range(budget - 1):
                pm_core._increment_reviewer_attempt(TID, 10, 1)
                pm_core._record_reviewer_attempt_reason(
                    TID, 10, 1,
                    "ERROR: local reviewer produced no verdict (reason=gw_defer_timeout)",
                )
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 10, 1)
            assert decision is None

    def test_defer_timeouts_at_infra_budget_produce_infra_exhausted_decision(self):
        """DoD 4: at exactly REVIEWER_INFRA_RETRY_BUDGET_DEFAULT (6), the
        pause fires as reviewer_infra_budget_exhausted — never
        reviewer_attempt_ceiling."""
        store = _tmp_store()
        budget = pm_core._reviewer_infra_retry_budget()
        assert budget == pm_core.REVIEWER_INFRA_RETRY_BUDGET_DEFAULT
        with patch("lapis_pm.pm_core._mem", return_value=store):
            decision = None
            for _ in range(budget):
                pm_core._increment_reviewer_attempt(TID, 11, 1)
                pm_core._record_reviewer_attempt_reason(
                    TID, 11, 1,
                    "ERROR: local reviewer produced no verdict (reason=gw_defer_timeout)",
                )
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 11, 1)
            assert decision is not None
            assert decision.kind == "reviewer_infra_budget_exhausted"
            assert decision.payload["infra_attempts"] == budget

    def test_unrecognised_reason_still_counts_against_ceiling(self):
        """DoD 6: fail-closed is preserved — a reason string in no allow-list
        still consumes the reviewer-attempt ceiling."""
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 12, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 12, 1, "reason=some_totally_novel_failure_mode",
            )
            pm_core._increment_reviewer_attempt(TID, 12, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 12, 1, "reason=some_totally_novel_failure_mode",
            )
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 12, 1)
            assert decision is not None
            assert decision.kind == "reviewer_attempt_ceiling"

    def test_poison_pill_genuine_defect_wearing_defer_timeout_reason_still_bounded(self):
        """DoD 7: a reviewer that is failing for a genuine, non-infra cause
        but whose emitted reason happens to be gw_defer_timeout must still be
        bounded — reclassification moves the bound from 2 to 6, not to
        infinity. Drives repeated failures past the infra budget and asserts
        the run halts at REVIEWER_INFRA_RETRY_BUDGET_DEFAULT with a
        reviewer_infra_budget_exhausted Decision naming the reason — never an
        unbounded redispatch loop."""
        store = _tmp_store()
        budget = pm_core._reviewer_infra_retry_budget()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            decision = None
            attempts_made = 0
            for _ in range(budget + 10):  # drive well past the budget
                decision = pm_core._reviewer_attempt_ceiling_check(TID, 13, 1)
                if decision is not None:
                    break
                pm_core._increment_reviewer_attempt(TID, 13, 1)
                # The reason string is a genuine defect masquerading behind
                # gw_defer_timeout's vocabulary (e.g. a misclassified crash).
                pm_core._record_reviewer_attempt_reason(
                    TID, 13, 1,
                    "ERROR: local reviewer produced no verdict (reason=gw_defer_timeout)",
                )
                attempts_made += 1
            assert decision is not None
            assert decision.kind == "reviewer_infra_budget_exhausted"
            assert decision.payload["infra_attempts"] == budget
            assert attempts_made == budget
            assert "gw_defer_timeout" in decision.payload["reported_reason"]


class TestDeferTimeoutReasonAndWaitTelemetry:
    def test_reason_survives_verbatim_into_attempt_state_and_payload(self):
        """DoD 8: the reason string survives verbatim into the attempt state
        and into the reviewer_infra_budget_exhausted payload's
        last_infra_reason, so a congestion pause is distinguishable from an
        absence pause without reading code."""
        store = _tmp_store()
        budget = pm_core._reviewer_infra_retry_budget()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            for _ in range(budget):
                pm_core._increment_reviewer_attempt(TID, 14, 1)
                pm_core._record_reviewer_attempt_reason(
                    TID, 14, 1,
                    "ERROR: local reviewer produced no verdict (reason=gw_defer_timeout)",
                )
            state = pm_core._reviewer_attempt_state(TID, 14, 1)
            assert state["last_infra_reason"] == "gw_defer_timeout"
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 14, 1)
            assert decision.payload["reported_reason"] == "gw_defer_timeout"

    def test_wait_seconds_recorded_beside_reason(self):
        """DoD 9: last_infra_wait_s is recorded beside last_infra_reason and
        surfaced in the reviewer_infra_budget_exhausted payload."""
        store = _tmp_store()
        budget = pm_core._reviewer_infra_retry_budget()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            for _ in range(budget):
                pm_core._increment_reviewer_attempt(TID, 15, 1)
                pm_core._record_reviewer_attempt_reason(
                    TID, 15, 1,
                    "ERROR: local reviewer produced no verdict (reason=gw_defer_timeout)",
                    wait_s=44.7,
                )
            state = pm_core._reviewer_attempt_state(TID, 15, 1)
            assert state["last_infra_wait_s"] == 44.7
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 15, 1)
            assert decision.payload["last_infra_wait_s"] == 44.7

    def test_unmeasurable_wait_records_none_not_zero(self):
        """DoD 9: absent/unmeasurable duration records None, never 0 — a
        fabricated 0 would be indistinguishable from an instant refusal."""
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 16, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 16, 1,
                "ERROR: local reviewer produced no verdict (reason=gw_defer_timeout)",
            )
            state = pm_core._reviewer_attempt_state(TID, 16, 1)
            assert state["last_infra_wait_s"] is None

    def test_dispatch_wait_seconds_computed_from_ts_and_completed_at(self):
        """_reviewer_dispatch_wait_seconds derives the duration from the
        dispatch record's ts -> completed_at span."""
        rec = {
            "ts": "2026-08-06T10:00:00-07:00",
            "completed_at": "2026-08-06T10:00:45-07:00",
        }
        assert pm_core._reviewer_dispatch_wait_seconds(rec) == 45.0

    def test_dispatch_wait_seconds_none_when_timestamps_missing(self):
        assert pm_core._reviewer_dispatch_wait_seconds({}) is None
        assert pm_core._reviewer_dispatch_wait_seconds({"ts": "2026-08-06T10:00:00-07:00"}) is None


# ---------------------------------------------------------------------------
# lapis-pm-reviewer-seat-dead-token-infra-classify-v0 (Unit B): a seat that
# cannot emit a tool call never rendered a verdict — see
# agents-core-reviewer-seat-tool-call-probe-v0 (Unit A), the emitter of
# `seat_no_tool_calls`. This classifies that token identically to the other
# infra-non-run reasons: excluded from the reviewer-attempt ceiling, still
# bounded by the separate, more generous infra-retry budget (D1/D5).
# ---------------------------------------------------------------------------

class TestSeatNoToolCallsClassification:
    def test_verbatim_wrapped_reason_classified_infra(self):
        """DoD 2: reason=seat_no_tool_calls extracted via the reason= regex."""
        assert pm_core._classify_reviewer_infra_reason(
            "ERROR: local reviewer produced no verdict (reason=seat_no_tool_calls)"
        ) == "seat_no_tool_calls"

    def test_bare_substring_fallback_also_matches(self):
        """DoD 3: also classifies via the bare-substring fallback loop at
        :3609-3611, for reasons arriving without the reason= wrapper."""
        assert pm_core._classify_reviewer_infra_reason(
            "seat_no_tool_calls"
        ) == "seat_no_tool_calls"


class TestPreExistingMembersUnaffected:
    def test_gw_not_serving_still_classifies(self):
        """DoD 4: regression fence — the three pre-existing members must
        still classify exactly as before this token was added."""
        assert pm_core._classify_reviewer_infra_reason(
            "ERROR: local reviewer produced no verdict (reason=gw_not_serving)"
        ) == "gw_not_serving"

    def test_gw_unreachable_still_classifies(self):
        assert pm_core._classify_reviewer_infra_reason(
            "ERROR: local reviewer produced no verdict (reason=gw_unreachable)"
        ) == "gw_unreachable"

    def test_gw_defer_timeout_still_classifies(self):
        assert pm_core._classify_reviewer_infra_reason(
            "ERROR: local reviewer produced no verdict (reason=gw_defer_timeout)"
        ) == "gw_defer_timeout"


class TestSeatNoToolCallsCeilingExclusion:
    def test_attempts_failing_with_seat_no_tool_calls_excluded_from_ceiling(self):
        """DoD 5b: end-to-end exclusion, not just classification. Drives
        _record_reviewer_attempt_reason with seat_no_tool_calls past what
        would be the reviewer-attempt ceiling (default 2) and asserts the
        resulting decision from _reviewer_attempt_ceiling_check is NOT
        reviewer_attempt_ceiling — the attempts must be excluded via
        state["infra_count"], the real consumer at :4318."""
        store = _tmp_store()
        ceiling = pm_core._reviewer_attempt_ceiling()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            for _ in range(ceiling + 1):
                pm_core._increment_reviewer_attempt(TID, 20, 1)
                pm_core._record_reviewer_attempt_reason(
                    TID, 20, 1,
                    "ERROR: local reviewer produced no verdict (reason=seat_no_tool_calls)",
                )
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 20, 1)
            assert decision is None or decision.kind != "reviewer_attempt_ceiling"

    def test_seat_no_tool_calls_still_bounded_by_infra_budget(self):
        """The exclusion is not unbounded — repeated seat_no_tool_calls
        failures still trip the separate infra-retry budget, exactly like
        the other infra reasons (D5: no threshold tuning)."""
        store = _tmp_store()
        budget = pm_core._reviewer_infra_retry_budget()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            decision = None
            for _ in range(budget + 2):
                decision = pm_core._reviewer_attempt_ceiling_check(TID, 21, 1)
                if decision is not None:
                    break
                pm_core._increment_reviewer_attempt(TID, 21, 1)
                pm_core._record_reviewer_attempt_reason(
                    TID, 21, 1,
                    "ERROR: local reviewer produced no verdict (reason=seat_no_tool_calls)",
                )
            assert decision is not None
            assert decision.kind == "reviewer_infra_budget_exhausted"
            assert decision.payload["infra_attempts"] == budget
            assert decision.payload["reported_reason"] == "seat_no_tool_calls"
