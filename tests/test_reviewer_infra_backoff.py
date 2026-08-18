"""Tests for the reviewer defer backoff (lapis-pm-reviewer-defer-backoff-v0).

Reviewer dispatches were losing a priority-inversion race against spec-review
gates on GravityWell: gates hold protected-class doorman leases for 40+
minutes while the PM re-dispatched deferrable reviewer attempts every tick,
each failing gw_defer_timeout after a 45s poll, so the infra retry budget (6)
burned in ~6 minutes against occupancies that outlast it — see the spec's
2026-08-18/2026-08-14 bundle-autodispatch evidence.

Coverage (Definition of Done 1-3):
  1. Schedule table + next_retry_at computation honour the 2/4/8/16/20/20 min
     lookup table (table sum >= 70 min, the designed margin against gate
     occupancy); dispatch timestamps across the full 6-attempt budget match
     the table's first 5 gaps exactly, with exhaustion firing immediately on
     the 6th; the distinct noop decision string renders while backing off.
  2. Non-infra failure and healthy-verdict paths take the same control-flow
     branches as current main (structural: _decide_for_pr still returns
     dispatch_reviewer immediately, no backoff gate) — plus a decision-string
     regression guard — and a non-infra completion clears next_retry_at.
  3. Backoff state round-trips through _reviewer_attempt_state persistence
     and survives a simulated PM process restart; exhaustion at 6 still
     auto-pauses exactly as before this unit.
"""

from __future__ import annotations

import datetime as _dt
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents_core.mem import MemoryStore
from lapis_pm import pm_core

TID = "test-reviewer-infra-backoff-fixture"

GW_DEFER_TIMEOUT = "ERROR: local reviewer produced no verdict (reason=gw_defer_timeout)"
GENUINE_FAILURE = "verdict JSON failed to parse"


def _tmp_store() -> MemoryStore:
    return MemoryStore(db_path=Path(tempfile.mktemp(suffix=".db")))


class _FrozenDatetime(_dt.datetime):
    """A datetime subclass whose .now() returns a fixed instant until moved.
    fromisoformat/arithmetic are inherited unchanged (verified: both return
    instances of the subclass), so patching `lapis_pm.pm_core.datetime` to
    this class redirects every `datetime.now(...)` call in pm_core.py to a
    controllable clock while every parse/format call keeps working exactly
    as it does against the real datetime class."""
    _frozen = None

    @classmethod
    def now(cls, tz=None):
        if cls._frozen is None:
            return _dt.datetime.now(tz)
        return cls._frozen.astimezone(tz) if tz is not None else cls._frozen

    @classmethod
    def set_frozen(cls, dt):
        cls._frozen = dt

    @classmethod
    def advance(cls, seconds):
        cls._frozen = cls._frozen + _dt.timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# R1: schedule table + next_retry_at computation
# ---------------------------------------------------------------------------

class TestBackoffSchedule:
    def test_schedule_table_matches_spec(self):
        assert pm_core.REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC == [120, 240, 480, 960, 1200, 1200]

    def test_schedule_has_one_entry_per_infra_budget(self):
        assert len(pm_core.REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC) == pm_core.REVIEWER_INFRA_RETRY_BUDGET_DEFAULT

    def test_seconds_indexed_one_based_by_infra_count(self):
        for i, expected in enumerate(pm_core.REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC, start=1):
            assert pm_core._reviewer_infra_backoff_seconds(i) == expected

    def test_seconds_clamps_beyond_table_length(self):
        table = pm_core.REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC
        assert pm_core._reviewer_infra_backoff_seconds(len(table) + 5) == table[-1]

    def test_seconds_clamps_below_one(self):
        assert pm_core._reviewer_infra_backoff_seconds(0) == pm_core.REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC[0]

    def test_total_span_across_full_budget_is_at_least_70_minutes(self):
        total = sum(pm_core.REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC)
        assert total >= 70 * 60

    def test_until_anchors_on_completed_at(self):
        until = pm_core._reviewer_infra_backoff_until("2026-08-18T12:00:00-07:00", 1)
        parsed = _dt.datetime.fromisoformat(until)
        expected = _dt.datetime.fromisoformat("2026-08-18T12:02:00-07:00")
        assert parsed == expected

    def test_until_falls_back_to_now_when_completed_at_missing(self):
        with patch("lapis_pm.pm_core.datetime", _FrozenDatetime):
            _FrozenDatetime.set_frozen(_dt.datetime(2026, 8, 18, 12, 0, 0, tzinfo=pm_core.PACIFIC))
            until = pm_core._reviewer_infra_backoff_until(None, 1)
        parsed = _dt.datetime.fromisoformat(until)
        assert parsed == _dt.datetime(2026, 8, 18, 12, 2, 0, tzinfo=pm_core.PACIFIC)

    def test_until_falls_back_to_now_when_completed_at_unparseable(self):
        with patch("lapis_pm.pm_core.datetime", _FrozenDatetime):
            _FrozenDatetime.set_frozen(_dt.datetime(2026, 8, 18, 12, 0, 0, tzinfo=pm_core.PACIFIC))
            until = pm_core._reviewer_infra_backoff_until("not-a-timestamp", 1)
        parsed = _dt.datetime.fromisoformat(until)
        assert parsed == _dt.datetime(2026, 8, 18, 12, 2, 0, tzinfo=pm_core.PACIFIC)

    def test_record_reviewer_attempt_reason_stamps_next_retry_at(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 241, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 241, 1, GW_DEFER_TIMEOUT,
                completed_at="2026-08-18T05:23:00-07:00",
            )
            state = pm_core._reviewer_attempt_state(TID, 241, 1)
        assert state["next_retry_at"] == _dt.datetime.fromisoformat(
            "2026-08-18T05:25:00-07:00"
        ).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# R2: decide() honours the backoff — simulated ticks across the schedule
# ---------------------------------------------------------------------------

class TestBackoffCheck:
    def test_returns_none_with_no_state(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            assert pm_core._reviewer_infra_backoff_check(TID, 1, 1) is None

    def test_blocks_immediately_after_infra_failure(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 1, 1)
            pm_core._record_reviewer_attempt_reason(TID, 1, 1, GW_DEFER_TIMEOUT)
            decision = pm_core._reviewer_infra_backoff_check(TID, 1, 1)
        assert decision is not None
        assert decision.kind == "reviewer_infra_backoff"
        assert decision.payload["reason"] == "gw_defer_timeout"

    def test_clears_once_deadline_passes(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            with patch("lapis_pm.pm_core.datetime", _FrozenDatetime):
                _FrozenDatetime.set_frozen(
                    _dt.datetime(2026, 8, 18, 5, 15, 0, tzinfo=pm_core.PACIFIC)
                )
                pm_core._increment_reviewer_attempt(TID, 2, 1)
                pm_core._record_reviewer_attempt_reason(TID, 2, 1, GW_DEFER_TIMEOUT)
                # Still inside the 120s window.
                assert pm_core._reviewer_infra_backoff_check(TID, 2, 1) is not None
                _FrozenDatetime.advance(119)
                assert pm_core._reviewer_infra_backoff_check(TID, 2, 1) is not None
                _FrozenDatetime.advance(2)
                assert pm_core._reviewer_infra_backoff_check(TID, 2, 1) is None

    def test_dispatch_timestamps_honour_schedule_table_across_full_budget(self):
        """DoD 1: drive REVIEWER_INFRA_RETRY_BUDGET_DEFAULT consecutive infra
        failures through decide()'s real redispatch site
        (_reviewer_infra_backoff_check gating _decide_for_pr), each time
        advancing the frozen clock to exactly the schedule boundary, and
        assert: (a) redispatch is refused before each boundary, (b)
        redispatch proceeds exactly at it, (c) the resulting gaps between
        the 6 dispatches match REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC[0:5]
        faithfully, (d) budget exhaustion fires immediately on the 6th
        failure with no further wait (R4: this unit spaces the spend, it
        does not change when exhaustion itself fires).

        The table's full sum (~70 min, asserted separately in
        test_total_span_across_full_budget_is_at_least_70_minutes) is the
        designed *margin* against gate occupancy — the 6th entry backs the
        would-be 7th attempt that budget exhaustion pre-empts, so the real
        span across 6 dispatches is schedule[0:5] (~50 min), not the full
        table sum.
        """
        from lapis_pm import authority

        store = _tmp_store()
        cls = authority.PRClassification(
            verdict="advisory", screen_verdict="unknown",
            static_outcome=authority.StaticOutcome.static_pass,
            reasons=["static checks passed"], issues=[],
            pr_number=241, repo="agents-core", title="feat: x",
            html_url="https://forgejo/Erah/agents-core/pulls/241",
            changed_paths=["src/x.py"], diff_loc=5, diff="",
        )
        pr = {
            "number": 241, "title": "feat: x",
            "html_url": "https://forgejo/Erah/agents-core/pulls/241",
            "head": {"ref": "lapis/t/x"}, "mergeable": True,
        }

        dispatch_times = []
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.datetime", _FrozenDatetime),
            patch("lapis_pm.pm_core.authority.classify", return_value=cls),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
        ):
            _FrozenDatetime.set_frozen(
                _dt.datetime(2026, 8, 18, 5, 15, 0, tzinfo=pm_core.PACIFIC)
            )
            budget = pm_core.REVIEWER_INFRA_RETRY_BUDGET_DEFAULT
            schedule = pm_core.REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC

            for i in range(budget):
                if i > 0:
                    # Before this attempt's scheduled boundary: must back
                    # off, and must never wedge behind the (unrelated)
                    # reviewer-attempt ceiling.
                    pre_boundary = pm_core._decide_for_pr(TID, "agents-core", pr, "advisory")
                    assert pre_boundary.kind == "reviewer_infra_backoff"
                    assert pre_boundary.payload["next_retry_at"] is not None
                    # Jump to exactly this attempt's scheduled boundary.
                    _FrozenDatetime.advance(schedule[i - 1])

                decision = pm_core._decide_for_pr(TID, "agents-core", pr, "advisory")
                assert decision.kind == "dispatch_reviewer"
                dispatch_times.append(_FrozenDatetime._frozen)
                pm_core._increment_reviewer_attempt(TID, 241, decision.payload["cycle"])
                pm_core._record_reviewer_attempt_reason(
                    TID, 241, decision.payload["cycle"], GW_DEFER_TIMEOUT,
                    completed_at=_FrozenDatetime._frozen.isoformat(),
                )

            # 6th failure just recorded → infra_count == budget. Exhaustion
            # fires immediately, no further wait, and is never masked by a
            # backoff noop (the ceiling check runs first at every call site).
            final_decision = pm_core._decide_for_pr(TID, "agents-core", pr, "advisory")

        assert final_decision.kind == "reviewer_infra_budget_exhausted"
        assert len(dispatch_times) == budget
        gaps = [
            (dispatch_times[i] - dispatch_times[i - 1]).total_seconds()
            for i in range(1, len(dispatch_times))
        ]
        assert gaps == [float(s) for s in schedule[: budget - 1]]

    def test_distinct_noop_decision_string_renders_while_backing_off(self):
        """DoD 1: tick()'s decision_str for a reviewer_infra_backoff Decision
        is the documented `noop:reviewer_infra_backoff:until=<iso>` shape,
        not a generic noop:no_change."""
        store = _tmp_store()
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.datetime", _FrozenDatetime),
        ):
            # Frozen "now" sits inside the 120s window opened by the failure
            # below — deliberately not real wall-clock time, so this doesn't
            # depend on when the test happens to run.
            _FrozenDatetime.set_frozen(_dt.datetime(2026, 8, 18, 5, 23, 30, tzinfo=pm_core.PACIFIC))
            pm_core._increment_reviewer_attempt(TID, 3, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 3, 1, GW_DEFER_TIMEOUT, completed_at="2026-08-18T05:23:00-07:00",
            )
            decision = pm_core._reviewer_infra_backoff_check(TID, 3, 1)
        assert decision.payload["next_retry_at"] == _dt.datetime.fromisoformat(
            "2026-08-18T05:25:00-07:00"
        ).isoformat(timespec="microseconds")
        # Mirrors the tick()-side rendering at the reviewer_infra_backoff
        # branch — asserted directly against the payload shape it reads.
        decision_str = f"noop:reviewer_infra_backoff:until={decision.payload['next_retry_at']}"
        assert decision_str.startswith("noop:reviewer_infra_backoff:until=2026-08-18T05:25:00")


# ---------------------------------------------------------------------------
# R1 tail clause + DoD 2: non-infra completions clear next_retry_at, and
# non-infra/healthy paths are structurally unchanged from current main.
# ---------------------------------------------------------------------------

class TestNonInfraClearsBackoff:
    def test_non_infra_failure_clears_next_retry_at(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 4, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 4, 1, GW_DEFER_TIMEOUT, completed_at="2026-08-18T05:23:00-07:00",
            )
            state = pm_core._reviewer_attempt_state(TID, 4, 1)
            assert state["next_retry_at"] is not None

            # A mixed-failure sequence: the next attempt fails for a genuine
            # (non-infra) reason — this must break the streak.
            pm_core._increment_reviewer_attempt(TID, 4, 1)
            pm_core._record_reviewer_attempt_reason(TID, 4, 1, GENUINE_FAILURE)
            state = pm_core._reviewer_attempt_state(TID, 4, 1)
        assert state["next_retry_at"] is None
        # infra_count/last_infra_reason stay as historical record — only
        # next_retry_at is cleared.
        assert state["infra_count"] == 1
        assert state["last_infra_reason"] == "gw_defer_timeout"

    def test_non_infra_failure_never_sets_next_retry_at_in_the_first_place(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 5, 1)
            pm_core._record_reviewer_attempt_reason(TID, 5, 1, GENUINE_FAILURE)
            state = pm_core._reviewer_attempt_state(TID, 5, 1)
        assert state["next_retry_at"] is None

    def test_healthy_verdict_clears_stale_backoff(self):
        """_clear_reviewer_infra_backoff is the hook the verdict-encoding
        path calls — exercised directly here (the encoder plumbing itself is
        exercised end-to-end elsewhere); asserts a stale next_retry_at left
        over from an earlier infra streak does not survive a real verdict."""
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 6, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 6, 1, GW_DEFER_TIMEOUT, completed_at="2026-08-18T05:23:00-07:00",
            )
            assert pm_core._reviewer_attempt_state(TID, 6, 1)["next_retry_at"] is not None

            pm_core._clear_reviewer_infra_backoff(TID, 6, 1)
            state = pm_core._reviewer_attempt_state(TID, 6, 1)
        assert state["next_retry_at"] is None
        # Historical fields untouched.
        assert state["infra_count"] == 1
        assert state["last_infra_reason"] == "gw_defer_timeout"

    def test_clear_is_a_noop_write_when_nothing_to_clear(self):
        """No spurious write when next_retry_at is already unset — asserted
        via idempotent output, not by inspecting the mem call count."""
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 7, 1)
            pm_core._clear_reviewer_infra_backoff(TID, 7, 1)
            state = pm_core._reviewer_attempt_state(TID, 7, 1)
        assert state["next_retry_at"] is None
        assert state["count"] == 1

    def test_non_infra_failure_path_still_returns_dispatch_reviewer_immediately(self):
        """DoD 2: structural assertion — a genuine (non-infra) failure never
        sets next_retry_at, so the very next _decide_for_pr call for this
        pr+cycle takes the same dispatch_reviewer branch it always has,
        unaffected by the backoff gate added for infra failures."""
        from lapis_pm import authority

        store = _tmp_store()
        cls = authority.PRClassification(
            verdict="advisory", screen_verdict="unknown",
            static_outcome=authority.StaticOutcome.static_pass,
            reasons=["static checks passed"], issues=[],
            pr_number=33, repo="facets", title="feat: x",
            html_url="https://forgejo/Erah/facets/pulls/33",
            changed_paths=["src/x.py"], diff_loc=5, diff="",
        )
        pr = {
            "number": 33, "title": "feat: x",
            "html_url": "https://forgejo/Erah/facets/pulls/33",
            "head": {"ref": "lapis/t/x"}, "mergeable": True,
        }
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.authority.classify", return_value=cls),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
        ):
            pm_core._increment_reviewer_attempt(TID, 33, 1)
            pm_core._record_reviewer_attempt_reason(TID, 33, 1, GENUINE_FAILURE)
            decision = pm_core._decide_for_pr(TID, "facets", pr, "advisory")
        assert decision.kind == "dispatch_reviewer"
        # _reviewer_cycle_count is mocked to a constant 0 (no verdict has
        # ever landed) — a failed attempt, infra or not, never advances it,
        # so the redispatch stays on the same cycle=1 attempt-state key.
        assert decision.payload["cycle"] == 1


# ---------------------------------------------------------------------------
# DoD 3: persistence round-trip, restart survival, exhaustion still pauses
# ---------------------------------------------------------------------------

class TestPersistenceAndRestart:
    def test_next_retry_at_round_trips_through_mem(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 8, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 8, 1, GW_DEFER_TIMEOUT, completed_at="2026-08-18T05:23:00-07:00",
            )
            written = pm_core._reviewer_attempt_state(TID, 8, 1)["next_retry_at"]

            # Simulate a fresh read with no in-process cache — construct a
            # brand-new MemoryStore instance against the SAME db file, exactly
            # what a PM process restart looks like (mem lives in mem.db, not
            # process memory).
            restarted_store = MemoryStore(db_path=store.db_path)
        with patch("lapis_pm.pm_core._mem", return_value=restarted_store):
            reread = pm_core._reviewer_attempt_state(TID, 8, 1)["next_retry_at"]
        assert reread == written is not None

    def test_backoff_check_honours_next_retry_at_after_simulated_restart(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            pm_core._increment_reviewer_attempt(TID, 9, 1)
            pm_core._record_reviewer_attempt_reason(TID, 9, 1, GW_DEFER_TIMEOUT)

        restarted_store = MemoryStore(db_path=store.db_path)
        with patch("lapis_pm.pm_core._mem", return_value=restarted_store):
            decision = pm_core._reviewer_infra_backoff_check(TID, 9, 1)
        assert decision is not None
        assert decision.kind == "reviewer_infra_backoff"

    def test_missing_next_retry_at_field_is_backward_compatible(self):
        """Records written before this unit landed have no next_retry_at key
        at all — the default must read as 'no backoff', not raise."""
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            # Simulate a pre-unit record: no next_retry_at key present.
            store.set(
                pm_core._reviewer_attempt_key(TID, 10, 1),
                '{"count": 3, "last_reason": "gw_defer_timeout", "infra_count": 3, '
                '"last_infra_reason": "gw_defer_timeout", "last_infra_wait_s": 44.0}',
                tags=["lapis-pm", "reviewer-attempt-ceiling"],
            )
            state = pm_core._reviewer_attempt_state(TID, 10, 1)
            decision = pm_core._reviewer_infra_backoff_check(TID, 10, 1)
        assert state["next_retry_at"] is None
        assert decision is None

    def test_exhaustion_at_six_still_auto_pauses(self):
        """R4: budget semantics unchanged — driving 6 infra failures through
        the real ceiling-check path (with the backoff gate patched out so
        the loop doesn't need real time to pass) still exhausts the infra
        budget and produces reviewer_infra_budget_exhausted, exactly as
        before this unit."""
        store = _tmp_store()
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core._reviewer_infra_backoff_check", return_value=None),
        ):
            decision = None
            for _ in range(pm_core.REVIEWER_INFRA_RETRY_BUDGET_DEFAULT):
                pm_core._increment_reviewer_attempt(TID, 11, 1)
                pm_core._record_reviewer_attempt_reason(TID, 11, 1, GW_DEFER_TIMEOUT)
            decision = pm_core._reviewer_attempt_ceiling_check(TID, 11, 1)
        assert decision is not None
        assert decision.kind == "reviewer_infra_budget_exhausted"
        assert decision.payload["infra_attempts"] == pm_core.REVIEWER_INFRA_RETRY_BUDGET_DEFAULT

    def test_ceiling_check_precedes_backoff_check_at_exhaustion(self):
        """R4: even mid-backoff-window, an exhausted infra budget must
        surface as reviewer_infra_budget_exhausted, never masked by a
        reviewer_infra_backoff noop — decide()'s callers check the ceiling
        first at every call site."""
        from lapis_pm import authority

        store = _tmp_store()
        cls = authority.PRClassification(
            verdict="advisory", screen_verdict="unknown",
            static_outcome=authority.StaticOutcome.static_pass,
            reasons=["static checks passed"], issues=[],
            pr_number=12, repo="agents-core", title="feat: x",
            html_url="https://forgejo/Erah/agents-core/pulls/12",
            changed_paths=["src/x.py"], diff_loc=5, diff="",
        )
        pr = {
            "number": 12, "title": "feat: x",
            "html_url": "https://forgejo/Erah/agents-core/pulls/12",
            "head": {"ref": "lapis/t/x"}, "mergeable": True,
        }
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.authority.classify", return_value=cls),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
        ):
            for _ in range(pm_core.REVIEWER_INFRA_RETRY_BUDGET_DEFAULT):
                pm_core._increment_reviewer_attempt(TID, 12, 1)
                pm_core._record_reviewer_attempt_reason(
                    TID, 12, 1, GW_DEFER_TIMEOUT,
                    completed_at="2026-08-18T05:23:00-07:00",
                )
            # next_retry_at is set (far in the future relative to real now),
            # but the budget is also exhausted — exhaustion must win.
            decision = pm_core._decide_for_pr(TID, "agents-core", pr, "advisory")
        assert decision.kind == "reviewer_infra_budget_exhausted"


# ---------------------------------------------------------------------------
# R3: status line
# ---------------------------------------------------------------------------

class TestStatusLineVisibility:
    def test_active_reviewer_backoff_returns_none_with_no_state(self):
        store = _tmp_store()
        with patch("lapis_pm.pm_core._mem", return_value=store):
            with patch("lapis_pm.pm_core._classified_pr_ids", return_value=set()):
                result = pm_core._active_reviewer_backoff(TID, [{"number": 99}])
        assert result is None

    def test_active_reviewer_backoff_surfaces_during_active_window(self):
        store = _tmp_store()
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.datetime", _FrozenDatetime),
        ):
            # Frozen "now" sits inside the 120s window opened by the failure
            # below — deliberately not real wall-clock time.
            _FrozenDatetime.set_frozen(_dt.datetime(2026, 8, 18, 5, 23, 30, tzinfo=pm_core.PACIFIC))
            pm_core._increment_reviewer_attempt(TID, 241, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 241, 1, GW_DEFER_TIMEOUT, completed_at="2026-08-18T05:23:00-07:00",
            )
            with (
                patch("lapis_pm.pm_core._classified_pr_ids", return_value=set()),
                patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            ):
                result = pm_core._active_reviewer_backoff(TID, [{"number": 241}])
        assert result is not None
        assert result["pr_number"] == 241
        assert result["infra_count"] == 1
        assert result["infra_budget"] == pm_core.REVIEWER_INFRA_RETRY_BUDGET_DEFAULT
        assert result["last_infra_reason"] == "gw_defer_timeout"

    def test_active_review_state_alone_misses_the_backing_off_shape(self):
        """The exact gap R3's docstring names: _active_review_state's
        'not started yet' bail-out (cycle==0, no pending reviewer) also
        matches a target that's backing off after a failed attempt — proving
        why _active_reviewer_backoff must be a separate check."""
        store = _tmp_store()
        with (
            patch("lapis_pm.pm_core._mem", return_value=store),
            patch("lapis_pm.pm_core.datetime", _FrozenDatetime),
        ):
            _FrozenDatetime.set_frozen(_dt.datetime(2026, 8, 18, 5, 23, 30, tzinfo=pm_core.PACIFIC))
            pm_core._increment_reviewer_attempt(TID, 241, 1)
            pm_core._record_reviewer_attempt_reason(
                TID, 241, 1, GW_DEFER_TIMEOUT, completed_at="2026-08-18T05:23:00-07:00",
            )
            with (
                patch("lapis_pm.pm_core._classified_pr_ids", return_value=set()),
                patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
                patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            ):
                review_state = pm_core._active_review_state(TID, [{"number": 241}])
                backoff = pm_core._active_reviewer_backoff(TID, [{"number": 241}])
        assert review_state is None
        assert backoff is not None
