"""Tests for lapis-pm-phala-test-key D3.4/D5.2: the dead-seat
gap-spanner reroute is removed. A prior seat_no_tool_calls failure on this
exact PR/cycle no longer redirects the next dispatch to a contractor seat —
there is no alternate seat anymore. Instead the dispatch log entry
(episodic.write_dispatch) must make the dead seat visible: state that the
seat is dead, that no alternate seat is configured, and that an operator
directive is the remedy.

Formerly (lapis-pm-reviewer-seat-phala-test-key) this file asserted
the opposite: that seat_no_tool_calls rerouted to reviewer_fresh_contractor.
That route, and the seat backing it, are gone (Erah, 2026-08-19: "Drop
Phala as contractor, the rotating setup breaks whatever we have often
enough that it shouldn't be relied upon for Machinery."). Rewritten, not
deleted, per spec D5.6.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from lapis_pm import authority, pm_core
from lapis_pm.tou_window import PACIFIC

PR = {
    "number": 42,
    "title": "feat: add widgets",
    "html_url": "https://forgejo/Erah/myrepo/pulls/42",
    "head": {"ref": "lapis/my-target/widgets"},
    "base": {"ref": "main"},
    "mergeable": True,
}

CLS = authority.PRClassification(
    verdict="advisory",
    screen_verdict="unknown",
    static_outcome=authority.StaticOutcome.static_pass,
    reasons=["static checks passed"],
    issues=[],
    pr_number=42,
    repo="myrepo",
    title="feat: add widgets",
    html_url="https://forgejo/Erah/myrepo/pulls/42",
    changed_paths=["src/foo.py"],
    diff_loc=30,
    diff="",
)

OFFPEAK_TIME = datetime(2026, 8, 4, 10, 0, tzinfo=PACIFIC)  # 10:00 PT
PEAK_TIME = datetime(2026, 8, 4, 18, 0, tzinfo=PACIFIC)     # 18:00 PT — formerly 16:00-21:00 peak


def _make_dispatch_result():
    result = MagicMock()
    result.task_id = "task-abc"
    result.spec_id = "spec-abc"
    return result


def _call_reviewer(mode, now, prior_state):
    """Dispatch a reviewer with a stubbed prior reviewer-attempt state, and
    return (captured_agent_type, dispatched_records, episodic_dispatch_texts)."""
    captured_agent_type = {}
    dispatched_records = []
    dispatch_texts = []

    def capture_dispatch(agent_type, target_id, user_prompt, vars_=None, **kw):
        captured_agent_type["value"] = agent_type
        return _make_dispatch_result()

    def fake_append(target_id, record):
        dispatched_records.append(record)

    def fake_write_dispatch(target_id, text, extra_tags=None):
        dispatch_texts.append(text)

    with (
        patch.object(pm_core._SHAPER, "dispatch", side_effect=capture_dispatch),
        patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd", return_value="/srv/git/myrepo-working"),
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
        patch("lapis_pm.pm_core.episodic.write_dispatch", side_effect=fake_write_dispatch),
        patch("lapis_pm.pm_core.append_dispatched", side_effect=fake_append),
        patch("lapis_pm.pm_core._increment_review_gate_counter", return_value=1),
        patch("lapis_pm.pm_core._increment_reviewer_attempt"),
        patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        patch("agents_core.forgejo.get_pr_diff", return_value="diff text"),
        patch("lapis_pm.pm_core._reviewer_attempt_state", return_value=prior_state),
        patch("lapis_pm.tou_window.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = now
        pm_core._act_dispatch_reviewer("tid", PR, CLS, mode=mode, cycle=1)

    return captured_agent_type.get("value"), dispatched_records, dispatch_texts


def _state(last_infra_reason=None):
    return {
        "count": 1,
        "last_reason": last_infra_reason,
        "infra_count": 1 if last_infra_reason else 0,
        "last_infra_reason": last_infra_reason,
        "last_infra_wait_s": None,
    }


class TestDeadSeatVisibleNoReroute:

    def test_prior_seat_no_tool_calls_fresh_mode_dispatches_local_seat(self):
        agent_type, records, texts = _call_reviewer(
            "fresh", OFFPEAK_TIME, _state("seat_no_tool_calls")
        )
        assert agent_type == "reviewer_fresh"
        assert "route" not in records[0]
        assert records[0]["agent_type"] == "reviewer_fresh"

    def test_prior_seat_no_tool_calls_dispatch_log_carries_no_alternate_seat_note(self):
        _, _, texts = _call_reviewer(
            "fresh", OFFPEAK_TIME, _state("seat_no_tool_calls")
        )
        assert any("seat_no_tool_calls" in t for t in texts)
        assert any("dead" in t.lower() for t in texts)
        assert any("no alternate seat" in t.lower() for t in texts)

    def test_prior_gw_not_serving_fresh_mode_no_note(self):
        agent_type, records, texts = _call_reviewer(
            "fresh", OFFPEAK_TIME, _state("gw_not_serving")
        )
        assert agent_type == "reviewer_fresh"
        assert "route" not in records[0]
        assert not any("seat_no_tool_calls" in t for t in texts)

    def test_prior_seat_no_tool_calls_same_mode_no_note(self):
        """same mode is never routed through the reviewer_fresh dead-seat
        check at all — the note is scoped to reviewer_fresh dispatches."""
        agent_type, records, texts = _call_reviewer(
            "same", OFFPEAK_TIME, _state("seat_no_tool_calls")
        )
        assert agent_type == "reviewer"
        assert "route" not in records[0]
        assert not any("seat_no_tool_calls" in t for t in texts)

    def test_former_peak_window_no_longer_affects_routing(self):
        agent_type, records, texts = _call_reviewer(
            "fresh", PEAK_TIME, _state(None)
        )
        assert agent_type == "reviewer_fresh"
        assert "route" not in records[0]
