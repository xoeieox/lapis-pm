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

AMENDED 2026-07-30 additions (cycle-1 review found the reviewer half couldn't execute):
  - vars_["pr_number"] must carry the resolved PR number, not a hardcoded ""
  - agent_type == "reviewer" must not raise KeyError('prior_review') inside
    _SHAPER.dispatch's system-template render
  - template-render test: every {placeholder} in the REAL reviewer/reviewer_fresh/
    fixer_retry system_template (registry.yaml) must be present in the vars_ dict
    force_dispatch builds -- catches the next unthreaded template var too
"""

from __future__ import annotations

import re
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


def _force_dispatch_capture_vars(
    target_id: str,
    agent_type: str,
    intent: str,
    *,
    open_prs: list[dict] | None = None,
    reviewer_cycle_count_return: int = 0,
    review_verdict_for_cycle_return: dict | None = None,
) -> dict:
    """Call force_dispatch mocking only _SHAPER.dispatch (NOT get_agent), so the
    REAL registry.yaml system_template is used when the caller composes it
    against the captured vars_. This is what the 8 original tests never did --
    they mocked get_agent too, which is exactly why the KeyError('prior_review')
    and hardcoded pr_number="" defects both slipped past review.
    """
    dispatch_mock = MagicMock(return_value=_make_dispatch_result())

    with (
        patch("lapis_pm.pm_core.TargetStore") as mock_store_cls,
        patch.object(pm_core._SHAPER, "dispatch", dispatch_mock),
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec summary"),
        patch("lapis_pm.pm_core.episodic.write_dispatch"),
        patch("lapis_pm.pm_core.append_dispatched"),
        patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        patch("agents_core.forgejo.get_open_prs", return_value=open_prs or []),
        patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=reviewer_cycle_count_return),
        patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=review_verdict_for_cycle_return),
        patch("lapis_pm.router_portfolio.emit_decision_dispatch"),
    ):
        mock_store_cls.return_value.get.return_value = _make_target(target_id)
        pm_core.force_dispatch(target_id, agent_type, intent)

    return dispatch_mock.call_args.kwargs["vars_"]


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


# ---------------------------------------------------------------------------
# AMENDED 2026-07-30: vars_["pr_number"] must be the resolved PR number, not ""
# ---------------------------------------------------------------------------

def test_reviewer_fresh_vars_carry_resolved_pr_number_not_blank():
    """Defect (1): vars_["pr_number"] was hardcoded "" and never updated from the
    pr_number force_dispatch had just resolved -- a force-dispatched reviewer_fresh
    was dispatched successfully but told to review PR "" while its record claimed
    a real pr_number/cycle."""
    pr = _make_pr(number=820)
    vars_ = _force_dispatch_capture_vars(
        "test-tid", "reviewer_fresh", "review it",
        open_prs=[pr], reviewer_cycle_count_return=1,
    )
    assert vars_["pr_number"] == "820"
    assert vars_["pr_number"] != ""


def test_fixer_no_open_pr_vars_pr_number_stays_blank():
    """Regression guard for the true initial-dispatch case: no open PR resolved
    means vars_["pr_number"] must stay "", matching the reviewer template's
    long-standing contract for a not-yet-existing PR."""
    vars_ = _force_dispatch_capture_vars(
        "test-tid", "fixer", "implement it",
        open_prs=[],
    )
    assert vars_["pr_number"] == ""


# ---------------------------------------------------------------------------
# AMENDED 2026-07-30: agent_type == "reviewer" must not raise KeyError('prior_review')
# ---------------------------------------------------------------------------

def test_reviewer_dispatch_does_not_raise_missing_prior_review():
    """Defect (2): the `reviewer` template requires {prior_review}, but force_dispatch's
    vars_ never supplied it, so _SHAPER.dispatch -> _compose_system ->
    system_template.format(**vars_) raised KeyError('prior_review') before the new
    record-construction code was ever reached -- making the whole `reviewer` branch
    dead code when invoked via --force-dispatch. This test renders the REAL template
    (not a mock) against the vars_ force_dispatch builds, so a regression here fails
    loud rather than crashing silently at runtime."""
    pr = _make_pr(number=820)
    vars_ = _force_dispatch_capture_vars(
        "test-tid", "reviewer", "review it",
        open_prs=[pr], reviewer_cycle_count_return=1,
    )
    assert "prior_review" in vars_

    agent = pm_core._SHAPER.get_agent("reviewer")
    vars_.setdefault("repo_cwd", "/tmp/fake-cwd")
    # Must not raise KeyError — this is the exact crash reproduced live 2026-07-30 21:26 PT.
    rendered = pm_core._SHAPER._compose_system(agent, vars_)
    assert "PR 820" in rendered


def test_reviewer_first_cycle_has_empty_prior_review():
    """cycle == 1 (no prior verdict yet) must fall back to prior_review="" rather
    than raising or fabricating a prior verdict."""
    pr = _make_pr(number=820)
    vars_ = _force_dispatch_capture_vars(
        "test-tid", "reviewer", "review it",
        open_prs=[pr], reviewer_cycle_count_return=0,
    )
    assert vars_["prior_review"] == ""


def test_reviewer_second_cycle_populates_prior_review_from_verdict():
    """cycle > 1 must build the prior-review block the same way
    _act_dispatch_reviewer does (same-reviewer mode), from the previous cycle's
    recorded verdict."""
    pr = _make_pr(number=820)
    prior_verdict = {
        "verdict": "fixable",
        "issues": [{"severity": "high", "path": "foo.py", "note": "bug"}],
    }
    vars_ = _force_dispatch_capture_vars(
        "test-tid", "reviewer", "review it",
        open_prs=[pr], reviewer_cycle_count_return=1,
        review_verdict_for_cycle_return=prior_verdict,
    )
    assert "Prior review context (cycle 1)" in vars_["prior_review"]
    assert "foo.py" in vars_["prior_review"]


# ---------------------------------------------------------------------------
# AMENDED 2026-07-30: template-render test -- every real placeholder must be threaded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "agent_type,reviewer_cycle_count_return",
    [("reviewer", 1), ("reviewer_fresh", 1), ("fixer_retry", 0)],
)
def test_template_placeholders_all_present_in_vars(agent_type, reviewer_cycle_count_return):
    """Cheap structural test, no GPU/network: take the agent's REAL system_template
    from registry.yaml, extract its {placeholder} names, and assert every one is
    present in the vars_ dict force_dispatch builds. Generalizes -- catches the
    next template that grows a placeholder nobody threaded into force_dispatch."""
    pr = _make_pr(number=820, target_id="test-tid")
    vars_ = _force_dispatch_capture_vars(
        "test-tid", agent_type, "do it",
        open_prs=[pr], reviewer_cycle_count_return=reviewer_cycle_count_return,
    )
    agent = pm_core._SHAPER.get_agent(agent_type)
    placeholders = set(re.findall(r"\{(\w+)\}", agent.system_template))
    missing = placeholders - set(vars_.keys())
    assert not missing, f"{agent_type} template placeholders not in vars_: {missing}"
