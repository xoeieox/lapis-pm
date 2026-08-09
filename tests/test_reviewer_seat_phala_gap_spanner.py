"""Tests for lapis-pm-reviewer-seat-phala-test-key: routing a
definitively-dead local reviewer seat to the Phala contractor instead of
letting it burn the rest of the infra budget toward a pause.

Invariants under test (spec Deliverable 4):
  - prior infra reason == "seat_no_tool_calls" and mode == "fresh" reroutes
    to reviewer_fresh_contractor, outside the TOU window.
  - prior infra reason == "gw_not_serving" (not seat_no_tool_calls) does NOT
    reroute — falls through to the existing ceiling/pause path unchanged.
  - mode == "same" never reroutes, regardless of prior reason.
  - the TOU-window branch still fires when neither gap-spanner condition
    applies (regression guard on the existing D2 carve-out).
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

OFFPEAK_TIME = datetime(2026, 8, 4, 10, 0, tzinfo=PACIFIC)  # 10:00 PT — outside TOU peak
PEAK_TIME = datetime(2026, 8, 4, 18, 0, tzinfo=PACIFIC)     # 18:00 PT — inside 16:00-21:00


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


class TestGapSpannerReroute:

    def test_prior_seat_no_tool_calls_fresh_mode_offpeak_reroutes_to_contractor(self):
        agent_type, records, texts = _call_reviewer(
            "fresh", OFFPEAK_TIME, _state("seat_no_tool_calls")
        )
        assert agent_type == "reviewer_fresh_contractor"
        assert records[0]["route"] == "gap-spanner-dead-seat-reroute"
        assert any("Gap-spanner reroute" in t and "seat_no_tool_calls" in t for t in texts)
        # Log the symptom only — never a diagnosed cause.
        assert not any("prefix-cache" in t.lower() or "poisoning" in t.lower() for t in texts)

    def test_prior_gw_not_serving_fresh_mode_does_not_reroute(self):
        agent_type, records, _ = _call_reviewer(
            "fresh", OFFPEAK_TIME, _state("gw_not_serving")
        )
        assert agent_type == "reviewer_fresh"
        assert "route" not in records[0]

    def test_prior_seat_no_tool_calls_same_mode_does_not_reroute(self):
        agent_type, records, _ = _call_reviewer(
            "same", OFFPEAK_TIME, _state("seat_no_tool_calls")
        )
        assert agent_type == "reviewer"
        assert "route" not in records[0]

    def test_tou_peak_branch_still_fires_when_no_gap_spanner_condition(self):
        agent_type, records, _ = _call_reviewer(
            "fresh", PEAK_TIME, _state(None)
        )
        assert agent_type == "reviewer_fresh_contractor"
        assert records[0]["route"] == "peak-window-policy"
