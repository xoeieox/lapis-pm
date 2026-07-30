"""Tests for lapis-pm-force-dispatch-pr-number-cycle-fix-v0.

Coverage:
  - force_dispatch("fixer_retry", ...) against an open PR stamps pr_number on the record
  - _fixer_retry_count counts force-dispatched fixer_retry records once pr_number is present
  - force_dispatch("reviewer_fresh", ...) stamps pr_number + next cycle, and threads the
    pm:reviewer:pr=N:cycle=K:verdict=pending tag into the episodic write
  - force_dispatch("fixer", ...) against a target with no open PR leaves pr_number absent
    (regression guard: true initial dispatch unaffected)
  - _decide_for_pr self-heals a reviewer_count < fixer_count skew (the 2026-07-30 incident)
  - replay-style integration: two force-dispatched fixer_retry calls against the same open
    PR followed by a tick dispatches a reviewer instead of no-op'ing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core
from lapis_pm import authority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(target_id: str = "test-tid", pm_repo: str = "myrepo") -> MagicMock:
    t = MagicMock()
    t.pm_bound = True
    t.pm_repo = pm_repo
    t.data = {}
    return t


def _make_dispatch_result(task_id: str = "task-abc") -> MagicMock:
    res = MagicMock()
    res.task_id = task_id
    res.spec_id = "spec-abc"
    res.output_path = "/tmp/fake-output"
    return res


def _make_agent(model: str = "opus") -> MagicMock:
    a = MagicMock()
    a.model = model
    return a


def _make_pr(number: int = 820, target_id: str = "test-tid", ref: str | None = None,
             base_ref: str = "main") -> dict:
    return {
        "number": number,
        "head": {"ref": ref or f"lapis/{target_id}/forced"},
        "base": {"ref": base_ref},
    }


@dataclass
class _FakeComment:
    content: str
    tags: list = field(default_factory=list)


def _force_dispatch(
    target_id: str,
    agent_type: str,
    intent: str,
    *,
    open_prs: list[dict] | None = None,
    dispatched_list: list | None = None,
    append_dispatched_capture: list | None = None,
    reviewer_cycle_count_return: int = 0,
) -> tuple[str, dict | None, MagicMock]:
    """Call force_dispatch with full mocking, capturing the append_dispatched record."""
    if dispatched_list is None:
        dispatched_list = []
    captured = append_dispatched_capture if append_dispatched_capture is not None else []

    def _capture(tid, record):
        captured.append(record)

    write_dispatch_mock = MagicMock()

    with (
        patch("lapis_pm.pm_core.TargetStore") as mock_store_cls,
        patch.object(pm_core._SHAPER, "dispatch", return_value=_make_dispatch_result()),
        patch.object(pm_core._SHAPER, "get_agent", return_value=_make_agent()),
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec summary"),
        patch("lapis_pm.pm_core.episodic.write_dispatch", write_dispatch_mock),
        patch("lapis_pm.pm_core.append_dispatched", side_effect=_capture),
        patch("lapis_pm.pm_core.load_dispatched", return_value=dispatched_list),
        patch("agents_core.forgejo.get_open_prs", return_value=open_prs or []),
        patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=reviewer_cycle_count_return),
        patch("lapis_pm.router_portfolio.emit_decision_dispatch"),
    ):
        mock_store_cls.return_value.get.return_value = _make_target(target_id)
        task_id = pm_core.force_dispatch(target_id, agent_type, intent)

    record = captured[-1] if captured else None
    return task_id, record, write_dispatch_mock


# ---------------------------------------------------------------------------
# fixer_retry stamps pr_number
# ---------------------------------------------------------------------------

def test_fixer_retry_stamps_pr_number_against_open_pr():
    pr = _make_pr(number=820)
    _, record, _ = _force_dispatch(
        "test-tid", "fixer_retry", "fix it",
        open_prs=[pr],
    )
    assert record is not None
    assert record["pr_number"] == 820


def test_fixer_retry_count_nonzero_after_two_force_dispatches():
    """After two force_dispatch('fixer_retry', ...) calls against the same open PR,
    once the SHA-advance perceiver marks them processed, _fixer_retry_count must
    reflect both — not silently stay 0 because pr_number was never stamped."""
    pr = _make_pr(number=820)
    captured: list[dict] = []

    _force_dispatch(
        "test-tid", "fixer_retry", "fix it #1",
        open_prs=[pr], append_dispatched_capture=captured,
    )
    _force_dispatch(
        "test-tid", "fixer_retry", "fix it #2",
        open_prs=[pr], append_dispatched_capture=captured,
    )

    assert len(captured) == 2
    for rec in captured:
        assert rec["pr_number"] == 820
        rec["status"] = "processed"  # simulate SHA-advance perceiver confirming push

    with patch("lapis_pm.pm_core.load_dispatched", return_value=captured):
        assert pm_core._fixer_retry_count("test-tid", 820) == 2


# ---------------------------------------------------------------------------
# reviewer_fresh stamps pr_number + cycle, threads episodic tag
# ---------------------------------------------------------------------------

def test_reviewer_fresh_stamps_pr_number_and_next_cycle():
    pr = _make_pr(number=820)
    _, record, write_dispatch_mock = _force_dispatch(
        "test-tid", "reviewer_fresh", "review it",
        open_prs=[pr], reviewer_cycle_count_return=1,
    )
    assert record is not None
    assert record["pr_number"] == 820
    assert record["cycle"] == 2  # prior_reviewer_count(1) + 1

    tags = write_dispatch_mock.call_args.kwargs["extra_tags"]
    assert "pm:reviewer:pr=820:cycle=2:verdict=pending" in tags


def test_reviewer_cycle_count_picks_up_terminal_verdict_tag():
    """Once the pending tag written by force_dispatch is superseded by a terminal
    verdict comment carrying the same pr/cycle, the real _reviewer_cycle_count
    (not mocked here) must count it."""
    comments = [
        _FakeComment(
            content="Reviewer verdict for PR #820:\n{}",
            tags=["pm:reviewer:pr=820:cycle=2:verdict=clean"],
        ),
    ]
    with patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments):
        assert pm_core._reviewer_cycle_count("test-tid", 820) == 1


# ---------------------------------------------------------------------------
# Initial dispatch (no open PR) unaffected — regression guard
# ---------------------------------------------------------------------------

def test_initial_fixer_no_open_pr_leaves_pr_number_absent():
    _, record, _ = _force_dispatch(
        "test-tid", "fixer", "implement it",
        open_prs=[],
    )
    assert record is not None
    assert "pr_number" not in record
    assert "cycle" not in record


# ---------------------------------------------------------------------------
# _decide_for_pr: reviewer_count < fixer_count self-heal (2026-07-30 incident)
# ---------------------------------------------------------------------------

def _make_cls(pm_authority: str = "advisory") -> MagicMock:
    cls = MagicMock(spec=authority.PRClassification)
    cls.static_outcome = "normal"
    cls.verdict = "advisory"
    cls.pr_number = 820
    cls.repo = "lapis-pm"
    cls.title = "test PR"
    cls.html_url = "http://forgejo/pr/820"
    cls.changed_paths = []
    cls.diff_loc = 10
    cls.diff = ""
    cls.issues = []
    cls.screen_verdict = "fixable"
    return cls


def test_decide_for_pr_reviewer_lt_fixer_dispatches_next_reviewer_cycle():
    """Exact skew from the 2026-07-30 incident: reviewer_count=1, fixer_count=2
    (two rapid force-dispatched fixer_retry calls, no intervening review).
    Must dispatch_reviewer cycle=2, not noop_no_change."""
    pr = {"number": 820, "head": {"sha": "abc123"}}
    cls = _make_cls()

    with (
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value=""),
        patch("lapis_pm.authority.classify", return_value=cls),
        patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
        patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
        patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
        patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
        patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
        patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
    ):
        decision = pm_core._decide_for_pr(
            "my-target", "lapis-pm", pr, pm_authority="advisory"
        )

    assert decision.kind == "dispatch_reviewer", (
        f"Expected dispatch_reviewer (self-heal), got {decision.kind!r}. "
        "reviewer_count < fixer_count must not fall through to noop_no_change."
    )
    assert decision.payload["cycle"] == 2


def test_decide_for_pr_reviewer_lt_fixer_respects_kill_switch():
    """The self-heal branch still honors the review-gate kill-switch."""
    pr = {"number": 820, "head": {"sha": "abc123"}}
    cls = _make_cls()

    with (
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value=""),
        patch("lapis_pm.authority.classify", return_value=cls),
        patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
        patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
        patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
        patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
        patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
        patch("lapis_pm.pm_core._review_gate_counter", return_value=pm_core.REVIEW_GATE_THRESHOLD),
        patch("lapis_pm.pm_core._set_review_gate_paused") as set_paused_mock,
    ):
        decision = pm_core._decide_for_pr(
            "my-target", "lapis-pm", pr, pm_authority="advisory"
        )

    assert decision.kind == "review_gate_pause"
    set_paused_mock.assert_called_once_with(True)


# ---------------------------------------------------------------------------
# Replay-style integration: two rapid force-dispatched fixer_retry calls, then tick
# ---------------------------------------------------------------------------

def test_replay_two_force_dispatched_fixer_retries_then_tick_dispatches_reviewer():
    """Mirrors the 2026-07-30 incident: force-dispatch fixer_retry twice against the
    same open PR (simulating rapid manual retries), then decide — must dispatch a
    reviewer rather than getting stuck at noop:no_change forever."""
    target_id = "conductor-gw-topology-dual-a3b-coder-v0"
    pr_dict = _make_pr(number=820, target_id=target_id)
    captured: list[dict] = []

    _force_dispatch(
        target_id, "fixer_retry", "retry #1",
        open_prs=[pr_dict], append_dispatched_capture=captured,
    )
    _force_dispatch(
        target_id, "fixer_retry", "retry #2",
        open_prs=[pr_dict], append_dispatched_capture=captured,
    )
    for rec in captured:
        rec["status"] = "processed"
    # One completed reviewer cycle predates both retries (matches the incident: a
    # single earlier reviewer verdict, then two force-dispatched retries with no
    # intervening review).
    captured.insert(0, {
        "agent_type": "reviewer",
        "pr_number": 820,
        "status": "processed",
        "cycle": 1,
        "ts": "2026-07-30T10:00:00-07:00",
        "gpu_id": "gpu-reviewer-1",
    })

    pr = {"number": 820, "head": {"sha": "abc123"}}
    cls = _make_cls()

    with (
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value=""),
        patch("lapis_pm.authority.classify", return_value=cls),
        patch("lapis_pm.pm_core.load_dispatched", return_value=captured),
        patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
        patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
        patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
        patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
        patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
    ):
        decision = pm_core._decide_for_pr(
            target_id, "lapis-pm", pr, pm_authority="advisory"
        )

    assert decision.kind == "dispatch_reviewer", (
        f"Expected dispatch_reviewer, got {decision.kind!r} — the daemon must "
        "self-heal the reviewer_count < fixer_count skew instead of no-op'ing."
    )
    assert decision.payload["cycle"] == 2
