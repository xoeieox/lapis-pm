"""Tests for the review-gate loop (lapis-pm-review-gate-loop spec).

Coverage (per spec deliverable 8):
  - reviewer dispatch fires when PR perceived + static checks pass (advisory/hold authority)
  - cycle count walks episodic correctly
  - fixable + cycles<budget → fixer retry dispatched with issues as intent
  - fixable + cycles==budget → exhausted brief synthesized
  - needs-human verdict → immediate brief, no retry
  - clean verdict + auto-merge authority → merge action
  - clean verdict + advisory authority → advisory brief
  - fresh-reviewer mode on hold PRs does NOT receive prior_review var
  - same-reviewer mode on advisory PRs DOES receive prior_review var (when cycle>1)
  - human directive during active loop preempts (existing decide priority preserved)
  - kill-switch: threshold-exceeded → reviewer dispatch skipped, fallback + single pause brief
  - lapis-pm review-gate resume → counter resets, review-gate enables again
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import authority, pm_core


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PR_TEMPLATE = {
    "number": 42,
    "title": "feat: add widgets",
    "html_url": "https://forgejo/Erah/myrepo/pulls/42",
    "head": {"ref": "lapis/my-target/widgets"},
    "mergeable": True,
}

CLS_STATIC_PASS = authority.PRClassification(
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
    diff="--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n+pass",
)

CLS_HELD_PATH = authority.PRClassification(
    verdict="hold",
    screen_verdict="unknown",
    static_outcome=authority.StaticOutcome.auto_hold_path,
    reasons=["held path(s) touched: infra/deploy.sh"],
    issues=[],
    pr_number=42,
    repo="myrepo",
    title="feat: add widgets",
    html_url="https://forgejo/Erah/myrepo/pulls/42",
    changed_paths=["infra/deploy.sh"],
    diff_loc=5,
    diff="",
)

CLS_AUTO_CLEAN = authority.PRClassification(
    verdict="auto",
    screen_verdict="clean",
    static_outcome=authority.StaticOutcome.static_pass,
    reasons=["clean screen, 10 LOC, mergeable"],
    issues=[],
    pr_number=42,
    repo="myrepo",
    title="feat: add widgets",
    html_url="https://forgejo/Erah/myrepo/pulls/42",
    changed_paths=["docs/README.md"],
    diff_loc=10,
    diff="",
)

ISSUES = [{"severity": "low", "path": "src/foo.py", "note": "missing docstring"}]


def _make_review_comment(pr_number: int, cycle: int, verdict: str,
                          issues: list | None = None) -> MagicMock:
    """Build a mock Comment with reviewer episodic tags."""
    c = MagicMock()
    c.tags = [
        "pm:result",
        f"pm:reviewer:pr={pr_number}:cycle={cycle}:verdict={verdict}",
        f"pm:pr={pr_number}",
    ]
    payload = {"verdict": verdict, "issues": issues or [], "confidence": 0.9}
    c.content = f"Reviewer verdict for PR #{pr_number}:\n{json.dumps(payload)}"
    return c


# ---------------------------------------------------------------------------
# authority.classify() static checks
# ---------------------------------------------------------------------------

class TestAuthorityClassify:
    """Verify authority.classify behaves correctly for different authority levels."""

    def test_held_path_always_triggers_hold(self):
        """Held path returns auto_hold_path regardless of authority."""
        pr = {"title": "t", "html_url": "", "mergeable": True}
        diff = "+++ b/infra/deploy.sh\n+echo hi"
        with (
            patch("lapis_pm.authority.get_pr", return_value=pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=diff),
        ):
            cls = authority.classify("myrepo", 1, "spec", pm_authority="advisory")
        assert cls.static_outcome == authority.StaticOutcome.auto_hold_path
        assert cls.verdict == "hold"
        assert cls.screen_verdict == "unknown"

    def test_advisory_no_llm_call(self):
        """Advisory authority classify() must NOT call screen() (inline LLM)."""
        pr = {"title": "t", "html_url": "", "mergeable": True}
        diff = "+++ b/src/foo.py\n+pass"
        with (
            patch("lapis_pm.authority.get_pr", return_value=pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=diff),
            patch("lapis_pm.authority.screen") as mock_screen,
        ):
            cls = authority.classify("myrepo", 1, "spec", pm_authority="advisory")
        mock_screen.assert_not_called()
        assert cls.static_outcome == authority.StaticOutcome.static_pass
        assert cls.screen_verdict == "unknown"

    def test_hold_no_llm_call(self):
        """Hold authority classify() must NOT call screen() (inline LLM)."""
        pr = {"title": "t", "html_url": "", "mergeable": True}
        diff = "+++ b/src/foo.py\n+pass"
        with (
            patch("lapis_pm.authority.get_pr", return_value=pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=diff),
            patch("lapis_pm.authority.screen") as mock_screen,
        ):
            cls = authority.classify("myrepo", 1, "spec", pm_authority="hold")
        mock_screen.assert_not_called()
        assert cls.static_outcome == authority.StaticOutcome.static_pass

    def test_auto_calls_screen(self):
        """Auto-merge authority classify() calls screen() inline."""
        pr = {"title": "t", "html_url": "", "mergeable": True}
        diff = "+++ b/docs/README.md\n+text"
        screen_result = {"verdict": "clean", "issues": [], "confidence": 0.95}
        with (
            patch("lapis_pm.authority.get_pr", return_value=pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=diff),
            patch("lapis_pm.authority.screen", return_value=screen_result) as mock_screen,
        ):
            cls = authority.classify("myrepo", 1, "spec", pm_authority="auto")
        mock_screen.assert_called_once()
        assert cls.screen_verdict == "clean"
        assert cls.verdict == "auto"


# ---------------------------------------------------------------------------
# Cycle count helpers
# ---------------------------------------------------------------------------

class TestCycleCountHelpers:

    def test_reviewer_cycle_count_zero_on_no_comments(self):
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]):
            count = pm_core._reviewer_cycle_count("tid", 42)
        assert count == 0

    def test_reviewer_cycle_count_counts_non_pending(self):
        comments = [
            _make_review_comment(42, 1, "fixable"),
            _make_review_comment(42, 2, "clean"),
        ]
        # Add a pending dispatch comment (should NOT be counted)
        pending = MagicMock()
        pending.tags = ["pm:dispatch", "pm:reviewer:pr=42:cycle=3:verdict=pending"]
        pending.content = ""
        comments.append(pending)

        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments):
            count = pm_core._reviewer_cycle_count("tid", 42)
        assert count == 2  # only completed ones

    def test_reviewer_cycle_count_skips_other_prs(self):
        comments = [
            _make_review_comment(42, 1, "clean"),
            _make_review_comment(99, 1, "fixable"),  # different PR
        ]
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments):
            count = pm_core._reviewer_cycle_count("tid", 42)
        assert count == 1

    def test_last_review_verdict_none_on_empty(self):
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]):
            result = pm_core._last_review_verdict("tid", 42)
        assert result is None

    def test_last_review_verdict_returns_latest(self):
        comments = [
            _make_review_comment(42, 1, "fixable", ISSUES),
            _make_review_comment(42, 2, "clean"),
        ]
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments):
            result = pm_core._last_review_verdict("tid", 42)
        assert result is not None
        assert result["verdict"] == "clean"

    def test_last_review_verdict_excludes_pending(self):
        c1 = _make_review_comment(42, 1, "fixable")
        pending = MagicMock()
        pending.tags = ["pm:reviewer:pr=42:cycle=2:verdict=pending"]
        pending.content = ""
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[c1, pending]):
            result = pm_core._last_review_verdict("tid", 42)
        assert result["verdict"] == "fixable"


# ---------------------------------------------------------------------------
# _decide_for_pr routing
# ---------------------------------------------------------------------------

class TestDecideForPr:

    def test_held_path_returns_hold_brief_immediately(self):
        """Held path → hold_brief without reviewer dispatch (spec invariant)."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_HELD_PATH),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "hold_brief"

    def test_auto_authority_clean_returns_merge(self):
        """Auto authority + clean screen → merge."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_AUTO_CLEAN),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "auto")
        assert d.kind == "merge"

    def test_auto_authority_fixable_returns_advisory_brief(self):
        """Auto authority + fixable verdict → advisory_brief."""
        cls = authority.PRClassification(
            verdict="advisory", screen_verdict="fixable",
            static_outcome=authority.StaticOutcome.static_pass,
            reasons=["fixable issues"], issues=ISSUES,
            pr_number=42, repo="myrepo", title="t", html_url="",
            changed_paths=[], diff_loc=10, diff="",
        )
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=cls),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "auto")
        assert d.kind == "advisory_brief"

    def test_advisory_never_reviewed_dispatches_reviewer(self):
        """Advisory PR, static pass, no prior review → dispatch reviewer."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "dispatch_reviewer"
        assert d.payload["mode"] == "same"
        assert d.payload["cycle"] == 1

    def test_hold_authority_uses_fresh_mode(self):
        """Hold authority → dispatch reviewer with fresh mode."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "hold")
        assert d.kind == "dispatch_reviewer"
        assert d.payload["mode"] == "fresh"

    def test_pending_reviewer_returns_noop(self):
        """Reviewer already pending for this PR → noop (no double-dispatch)."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=True),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "noop"

    def test_pending_fixer_returns_noop(self):
        """Fixer already pending → noop."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=True),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "noop"

    def test_fixable_cycles_below_budget_dispatches_fixer(self):
        """Reviewed (cycle 1) fixable + cycles < budget → dispatch_fixer_retry."""
        verdict = {"verdict": "fixable", "issues": ISSUES, "confidence": 0.8}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "dispatch_fixer_retry"
        assert d.payload["issues"] == ISSUES
        assert d.payload["cycle"] == 1  # reviewer cycle that returned fixable

    def test_fixable_cycles_at_budget_returns_exhausted_brief(self):
        """Reviewed (cycle 2) fixable + cycles >= budget → review_exhausted_brief."""
        verdict = {"verdict": "fixable", "issues": ISSUES, "confidence": 0.8}
        history = [{"cycle": 1, "verdict": "fixable", "issues": ISSUES},
                   {"cycle": 2, "verdict": "fixable", "issues": ISSUES}]
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=2),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
            patch("lapis_pm.pm_core._collect_review_history", return_value=history),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "review_exhausted_brief"
        assert d.payload["history"] == history

    def test_needs_human_verdict_returns_hold_brief(self):
        """needs-human verdict → immediate hold_brief, no fixer retry."""
        verdict = {"verdict": "needs-human", "issues": [], "confidence": 0.6}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "hold_brief"

    def test_clean_verdict_advisory_returns_advisory_brief(self):
        """clean verdict + advisory authority → advisory brief."""
        verdict = {"verdict": "clean", "issues": [], "confidence": 0.95}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "advisory_brief"

    def test_clean_verdict_hold_returns_hold_brief(self):
        """clean verdict + hold authority → hold brief (human still decides)."""
        verdict = {"verdict": "clean", "issues": [], "confidence": 0.95}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "hold")
        assert d.kind == "hold_brief"

    def test_post_fixer_dispatches_reviewer_again(self):
        """After fixer completes, next tick should dispatch reviewer (cycle 2)."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),  # fixer done
            patch("lapis_pm.pm_core._review_gate_counter", return_value=5),  # below threshold
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "dispatch_reviewer"
        assert d.payload["cycle"] == 2


# ---------------------------------------------------------------------------
# Reviewer mode vars (fresh vs same)
# ---------------------------------------------------------------------------

class TestReviewerModeVars:

    def _make_dispatch_result(self):
        result = MagicMock()
        result.task_id = "task-abc"
        result.spec_id = "spec-abc"
        return result

    def test_reviewer_fresh_dispatches_reviewer_fresh_agent(self):
        """Hold PRs dispatch reviewer_fresh agent type."""
        captured_agent = {}

        def capture_dispatch(agent_type, target_id, user_prompt, vars_=None, **kw):
            captured_agent["type"] = agent_type
            return self._make_dispatch_result()

        with (
            patch("lapis_pm.pm_core.shaper.dispatch", side_effect=capture_dispatch),
            patch("lapis_pm.pm_core.shaper._resolve_repo_cwd",
                  return_value="/srv/git/myrepo-working"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core._increment_review_gate_counter", return_value=1),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("agents_core.forgejo.get_pr_diff", return_value="diff"),
        ):
            pm_core._act_dispatch_reviewer("tid", PR_TEMPLATE, CLS_STATIC_PASS,
                                            mode="fresh", cycle=1)
        assert captured_agent["type"] == "reviewer_fresh"

    def test_reviewer_same_dispatches_reviewer_agent(self):
        """Advisory PRs dispatch reviewer agent type (same-reviewer mode)."""
        captured_agent = {}

        def capture_dispatch(agent_type, target_id, user_prompt, vars_=None, **kw):
            captured_agent["type"] = agent_type
            return self._make_dispatch_result()

        with (
            patch("lapis_pm.pm_core.shaper.dispatch", side_effect=capture_dispatch),
            patch("lapis_pm.pm_core.shaper._resolve_repo_cwd",
                  return_value="/srv/git/myrepo-working"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core._increment_review_gate_counter", return_value=1),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("agents_core.forgejo.get_pr_diff", return_value="diff"),
        ):
            pm_core._act_dispatch_reviewer("tid", PR_TEMPLATE, CLS_STATIC_PASS,
                                            mode="same", cycle=1)
        assert captured_agent["type"] == "reviewer"

    def test_same_reviewer_cycle2_injects_prior_review(self):
        """Same-reviewer mode cycle>1: prior_review var is populated."""
        captured_vars = {}

        def capture_dispatch(agent_type, target_id, user_prompt, vars_=None, **kw):
            captured_vars.update(vars_ or {})
            return self._make_dispatch_result()

        prior_verdict = {"verdict": "fixable", "issues": ISSUES, "confidence": 0.8}
        with (
            patch("lapis_pm.pm_core.shaper.dispatch", side_effect=capture_dispatch),
            patch("lapis_pm.pm_core.shaper._resolve_repo_cwd",
                  return_value="/srv/git/myrepo-working"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core._increment_review_gate_counter", return_value=2),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=prior_verdict),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("agents_core.forgejo.get_pr_diff", return_value="diff"),
        ):
            pm_core._act_dispatch_reviewer("tid", PR_TEMPLATE, CLS_STATIC_PASS,
                                            mode="same", cycle=2)

        assert "prior_review" in captured_vars
        assert "fixable" in captured_vars["prior_review"]  # prior verdict injected

    def test_same_reviewer_cycle1_has_empty_prior_review(self):
        """Same-reviewer mode cycle=1: prior_review is empty string."""
        captured_vars = {}

        def capture_dispatch(agent_type, target_id, user_prompt, vars_=None, **kw):
            captured_vars.update(vars_ or {})
            return self._make_dispatch_result()

        with (
            patch("lapis_pm.pm_core.shaper.dispatch", side_effect=capture_dispatch),
            patch("lapis_pm.pm_core.shaper._resolve_repo_cwd",
                  return_value="/srv/git/myrepo-working"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core._increment_review_gate_counter", return_value=1),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("agents_core.forgejo.get_pr_diff", return_value="diff"),
        ):
            pm_core._act_dispatch_reviewer("tid", PR_TEMPLATE, CLS_STATIC_PASS,
                                            mode="same", cycle=1)

        assert captured_vars.get("prior_review", "") == ""


# ---------------------------------------------------------------------------
# Kill-switch
# ---------------------------------------------------------------------------

class TestKillSwitch:

    def test_threshold_exceeded_returns_review_gate_pause(self):
        """Counter >= threshold → review_gate_pause decision (not dispatch_reviewer)."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._review_gate_counter",
                  return_value=pm_core.REVIEW_GATE_THRESHOLD),
            patch("lapis_pm.pm_core._set_review_gate_paused") as mock_pause,
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "review_gate_pause"
        mock_pause.assert_called_once_with(True)

    def test_paused_gate_falls_back_to_inline_sonnet(self):
        """When paused → fallback to inline Sonnet screen (not dispatch_reviewer)."""
        screen_result = {"verdict": "clean", "issues": [], "confidence": 0.9}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=True),
            patch("lapis_pm.pm_core.authority.screen", return_value=screen_result),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind != "dispatch_reviewer"
        assert d.kind in ("advisory_brief", "hold_brief", "merge")

    def test_act_review_gate_pause_posts_brief_once(self):
        """Kill-switch brief is emitted only once (idempotent on first call)."""
        brief_mock = MagicMock()
        brief_mock.comment_id = "brief-1"
        mem = MagicMock()
        mem.get.return_value = None  # no existing brief

        with (
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=40),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=brief_mock),
            patch("lapis_pm.pm_core.set_outstanding_brief"),
        ):
            result = pm_core._act_review_gate_pause(
                "tid", {"pr": PR_TEMPLATE, "cls": CLS_STATIC_PASS}
            )
        assert "review_gate_paused" in result
        assert brief_mock.comment_id in result

    def test_act_review_gate_pause_idempotent(self):
        """Second call to _act_review_gate_pause returns already_briefed."""
        mem = MagicMock()
        mem.get.return_value = {"content": "brief-1"}  # existing brief

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            result = pm_core._act_review_gate_pause(
                "tid", {"pr": PR_TEMPLATE, "cls": CLS_STATIC_PASS}
            )
        assert "already_briefed" in result

    def test_review_gate_resume_resets_counter(self):
        """review_gate_resume() resets counter to 0 and clears paused flag."""
        mem = MagicMock()
        # get() called for counter (returns 25) and for paused check
        mem.get.side_effect = [{"content": "25"}, None]

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            prev = pm_core.review_gate_resume()

        assert prev == 25
        set_calls = [c for c in mem.set.call_args_list
                     if c[0][0] == pm_core.REVIEW_GATE_COUNTER_KEY]
        assert set_calls
        assert set_calls[0][0][1] == "0"

    def test_review_gate_status_returns_state(self):
        """review_gate_status() returns counter, threshold, paused."""
        with (
            patch("lapis_pm.pm_core._review_gate_counter", return_value=15),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
        ):
            state = pm_core.review_gate_status()
        assert state["counter"] == 15
        assert state["threshold"] == pm_core.REVIEW_GATE_THRESHOLD
        assert state["paused"] is False


# ---------------------------------------------------------------------------
# Encode GPU results — reviewer-specific episodic tagging
# ---------------------------------------------------------------------------

class TestEncodeGpuResultsReviewer:

    def _make_pending_reviewer_record(self, pr_number=42, cycle=1):
        return {
            "gpu_id": "gpu-abc123",
            "spec_id": "spec-abc",
            "agent_type": "reviewer",
            "intent": f"review PR #{pr_number} cycle {cycle}",
            "repo": "myrepo",
            "pr_number": pr_number,
            "cycle": cycle,
            "status": "pending",
        }

    def test_reviewer_completion_writes_verdict_episodic_tag(self):
        """When reviewer completes, episodic entry has pm:reviewer:pr=N:cycle=K:verdict=V tag."""
        record = self._make_pending_reviewer_record()
        output = json.dumps({"verdict": "fixable", "issues": ISSUES, "confidence": 0.8})
        written_tags = []

        def capture_write_result(target_id, content, extra_tags=None):
            written_tags.extend(extra_tags or [])
            c = MagicMock()
            c.id = "comment-1"
            return c

        mock_path = MagicMock()
        mock_path.read_text.return_value = output
        mock_path.parent = object()  # not FAILED_DIR

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=mock_path),
            patch("lapis_pm.pm_core._read_fixer_meta", return_value=None),
            patch("lapis_pm.pm_core._consume_fixer_meta"),
            patch("lapis_pm.pm_core.episodic.write_result",
                  side_effect=capture_write_result),
            patch("lapis_pm.pm_core.FAILED_DIR", object()),  # different object
        ):
            encoded, failed = pm_core._encode_gpu_results("tid")

        reviewer_tags = [t for t in written_tags
                         if "pm:reviewer:pr=42:cycle=1:verdict=fixable" in t]
        assert reviewer_tags, f"Expected reviewer tag in {written_tags}"
        assert not failed  # reviewer success not a retry failure


# ---------------------------------------------------------------------------
# CLI: review-gate subcommand
# ---------------------------------------------------------------------------

class TestReviewGateCli:

    def test_review_gate_status_command(self, capsys):
        """lapis-pm review-gate status prints counter + threshold."""
        from lapis_pm.cli import main
        with (
            patch("lapis_pm.cli.pm_core.review_gate_status",
                  return_value={"counter": 10, "threshold": 40, "paused": False}),
        ):
            ret = main(["review-gate", "status"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "10" in out
        assert "40" in out

    def test_review_gate_resume_command(self):
        """lapis-pm review-gate resume resets counter."""
        from lapis_pm.cli import main
        with patch("lapis_pm.cli.pm_core.review_gate_resume", return_value=25) as mock_resume:
            ret = main(["review-gate", "resume"])
        assert ret == 0
        mock_resume.assert_called_once()


# ---------------------------------------------------------------------------
# fixer_retry completion observation (encode path)
# ---------------------------------------------------------------------------

class TestFixerRetryCompletion:
    """Tests for fixer_retry completion detection via PR SHA advancement."""

    def _pending_fixer_record(self, pr_number=42, cycle=1,
                               ts="2026-01-01T10:00:00") -> dict:
        return {
            "gpu_id": "claude_abc123_fixer_retry",
            "spec_id": "spec-abc",
            "agent_type": "fixer_retry",
            "intent": f"fix PR #{pr_number} after reviewer cycle {cycle}",
            "repo": "myrepo",
            "pr_number": pr_number,
            "cycle": cycle,
            "ts": ts,
            "status": "pending",
            "retry_count": 0,
        }

    def _sha_comment(self, pr_number: int, sha: str, ts: str) -> MagicMock:
        c = MagicMock()
        c.ts = ts
        c.tags = [f"pm:pr={pr_number}", f"pm:pr={pr_number}:sha={sha}"]
        c.content = f"PR #{pr_number} head SHA: {sha}"
        return c

    def test_fixer_retry_processed_when_sha_advances(self):
        """fixer_retry transitions to processed when a SHA observation appears after dispatch ts."""
        record = self._pending_fixer_record(ts="2026-01-01T10:00:00")
        sha_obs = self._sha_comment(42, "abc123newsha", "2026-01-01T10:05:00")

        written = []

        def capture_result(target_id, content, extra_tags=None):
            written.append(content)
            c = MagicMock()
            c.id = "result-1"
            return c

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched") as mock_save,
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[sha_obs]),
            patch("lapis_pm.pm_core.episodic.write_result", side_effect=capture_result),
        ):
            encoded, failed = pm_core._encode_gpu_results("tid")

        assert encoded == 1
        assert record["status"] == "processed"
        assert record["completed_at"] == sha_obs.ts
        mock_save.assert_called_once()
        assert not failed
        assert written  # episodic result was written

    def test_fixer_retry_stays_pending_when_sha_unchanged(self):
        """fixer_retry stays pending if no SHA observation exists after dispatch ts."""
        record = self._pending_fixer_record(ts="2026-01-01T10:00:00")
        # SHA observation exists but BEFORE dispatch_ts — does not qualify
        old_obs = self._sha_comment(42, "old_sha_xyz", "2026-01-01T09:00:00")

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched") as mock_save,
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[old_obs]),
        ):
            encoded, failed = pm_core._encode_gpu_results("tid")

        assert encoded == 0
        assert record["status"] == "pending"
        mock_save.assert_not_called()
        assert not failed

    def test_fixer_retry_completion_idempotent(self):
        """Already-processed fixer_retry records are not re-encoded."""
        record = self._pending_fixer_record(ts="2026-01-01T10:00:00")
        record["status"] = "processed"
        record["completed_at"] = "2026-01-01T10:05:00"
        sha_obs = self._sha_comment(42, "abc123", "2026-01-01T10:05:00")

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched") as mock_save,
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[sha_obs]),
        ):
            encoded, failed = pm_core._encode_gpu_results("tid")

        assert encoded == 0
        mock_save.assert_not_called()

    def test_reviewer_completion_still_uses_gpu_output(self):
        """Reviewer (non-fixer_retry) still uses GPU output file path."""
        reviewer_record = {
            "gpu_id": "gpu_rev_abc",
            "spec_id": "spec-rev",
            "agent_type": "reviewer",
            "intent": "review PR #42 cycle 1",
            "repo": "myrepo",
            "pr_number": 42,
            "cycle": 1,
            "ts": "2026-01-01T09:00:00",
            "status": "pending",
            "retry_count": 0,
        }
        output = json.dumps({"verdict": "fixable", "issues": ISSUES, "confidence": 0.8})

        mock_path = MagicMock()
        mock_path.read_text.return_value = output
        mock_path.parent = object()  # not FAILED_DIR or CLAUDE_QUEUE_FAILED_DIR

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_record]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=mock_path),
            patch("lapis_pm.pm_core._read_fixer_meta", return_value=None),
            patch("lapis_pm.pm_core._consume_fixer_meta"),
            patch("lapis_pm.pm_core.episodic.write_result", return_value=MagicMock()),
            patch("lapis_pm.pm_core.FAILED_DIR", object()),
            patch("lapis_pm.pm_core.CLAUDE_QUEUE_FAILED_DIR", object()),
        ):
            encoded, failed = pm_core._encode_gpu_results("tid")

        assert encoded == 1
        assert reviewer_record["status"] == "processed"


# ---------------------------------------------------------------------------
# Cycle K→K+1 progression (decide path)
# ---------------------------------------------------------------------------

class TestCycleKProgression:
    """Tests for reviewer cycle K+1 dispatch after fixer_retry completion."""

    def _reviewer_dispatch_record(self, pr_number=42, cycle=1,
                                   ts="2026-01-01T09:00:00") -> dict:
        return {
            "gpu_id": "gpu_rev_abc",
            "agent_type": "reviewer",
            "pr_number": pr_number,
            "cycle": cycle,
            "ts": ts,
            "status": "processed",
        }

    def _sha_observation(self, pr_number: int, sha: str, ts: str) -> MagicMock:
        c = MagicMock()
        c.ts = ts
        c.tags = [f"pm:pr={pr_number}", f"pm:pr={pr_number}:sha={sha}"]
        c.content = ""
        return c

    def test_cycle2_dispatched_when_fixer_processed_and_sha_advanced(self):
        """reviewer=1 fixer=1 SHA advanced → dispatch reviewer cycle 2 (advisory)."""
        reviewer_rec = self._reviewer_dispatch_record(cycle=1, ts="2026-01-01T09:00:00")
        sha_obs = self._sha_observation(42, "newsha_abc", "2026-01-01T10:00:00")

        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[sha_obs]),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")

        assert d.kind == "dispatch_reviewer"
        assert d.payload["cycle"] == 2
        assert d.payload["mode"] == "same"  # advisory keeps same-reviewer mode

    def test_no_progression_guard_sha_unchanged(self):
        """reviewer=1 fixer=1 SHA NOT advanced → noop (no-progression guard)."""
        reviewer_rec = self._reviewer_dispatch_record(cycle=1, ts="2026-01-01T09:00:00")
        # SHA observation exists but BEFORE reviewer dispatch ts — does not count
        old_sha = self._sha_observation(42, "oldsha", "2026-01-01T08:00:00")

        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[old_sha]),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")

        assert d.kind == "noop"

    def test_budget_exhaustion_after_cycle2_fixable(self):
        """reviewer=2 fixer=2 SHA advanced → review_exhausted_brief (advisory budget=2)."""
        # Cycle 2 reviewer dispatch record
        reviewer_rec = self._reviewer_dispatch_record(cycle=2, ts="2026-01-01T11:00:00")
        sha_obs = self._sha_observation(42, "sha2", "2026-01-01T11:30:00")

        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=2),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[sha_obs]),
            patch("lapis_pm.pm_core._collect_review_history", return_value=[
                {"cycle": 1, "verdict": "fixable", "issues": ISSUES},
                {"cycle": 2, "verdict": "fixable", "issues": ISSUES},
            ]),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")

        assert d.kind == "review_exhausted_brief"
        assert len(d.payload["history"]) == 2

    def test_same_reviewer_mode_preserved_at_cycle2_advisory(self):
        """Advisory: cycle 2 reviewer uses mode=same (same-reviewer context)."""
        reviewer_rec = self._reviewer_dispatch_record(cycle=1, ts="2026-01-01T09:00:00")
        sha_obs = self._sha_observation(42, "newsha", "2026-01-01T10:00:00")

        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[sha_obs]),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")

        assert d.kind == "dispatch_reviewer"
        assert d.payload["mode"] == "same"

    def test_fresh_reviewer_mode_preserved_at_cycle2_hold(self):
        """Hold: cycle 2 reviewer uses mode=fresh (fresh-reviewer, no prior context)."""
        reviewer_rec = self._reviewer_dispatch_record(cycle=1, ts="2026-01-01T09:00:00")
        reviewer_rec["agent_type"] = "reviewer_fresh"
        sha_obs = self._sha_observation(42, "newsha", "2026-01-01T10:00:00")

        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[sha_obs]),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "hold")

        assert d.kind == "dispatch_reviewer"
        assert d.payload["mode"] == "fresh"


# ---------------------------------------------------------------------------
# _recover_reviewer_verdict — parse hardening
# ---------------------------------------------------------------------------

class TestRecoverReviewerVerdict:

    def test_strict_valid_json_unaffected(self):
        """Recovery helper is not involved when json.loads succeeds — verify
        that the helper itself also parses clean JSON correctly."""
        raw = json.dumps({"verdict": "fixable", "issues": ISSUES, "confidence": 0.9})
        result = pm_core._recover_reviewer_verdict(raw)
        assert result is not None
        assert result["verdict"] == "fixable"

    def test_code_fence_json_recovered(self):
        """JSON wrapped in ```json ... ``` fences is recovered."""
        payload = {"verdict": "fixable", "issues": ISSUES, "confidence": 0.8}
        raw = f"```json\n{json.dumps(payload)}\n```"
        result = pm_core._recover_reviewer_verdict(raw)
        assert result is not None
        assert result["verdict"] == "fixable"
        assert result["issues"] == ISSUES

    def test_plain_code_fence_json_recovered(self):
        """JSON wrapped in plain ``` ... ``` fences is recovered."""
        payload = {"verdict": "clean", "issues": [], "confidence": 0.95}
        raw = f"```\n{json.dumps(payload)}\n```"
        result = pm_core._recover_reviewer_verdict(raw)
        assert result is not None
        assert result["verdict"] == "clean"

    def test_prose_prefixed_json_recovered(self):
        """JSON preceded by prose ('Here is my analysis: {...}') is recovered."""
        payload = {"verdict": "fixable", "issues": ISSUES, "confidence": 0.7}
        raw = f"Here is my analysis:\n{json.dumps(payload)}"
        result = pm_core._recover_reviewer_verdict(raw)
        assert result is not None
        assert result["verdict"] == "fixable"
        assert result["issues"] == ISSUES

    def test_truncated_json_with_verdict_literal_gives_verdict_only(self):
        """Truncated JSON where full structure is unparseable but verdict is
        visible falls back to regex recovery: issues=[], confidence=0.0."""
        raw = '{"verdict": "fixable", "issues": [{"severity": "high", "note": "broken'
        result = pm_core._recover_reviewer_verdict(raw)
        assert result is not None
        assert result["verdict"] == "fixable"
        assert result["issues"] == []
        assert result["confidence"] == 0.0

    def test_garbage_no_recognizable_verdict_returns_none(self):
        """Completely unparseable input with no recognizable verdict → None."""
        raw = "I could not complete the review due to an error. Please retry."
        result = pm_core._recover_reviewer_verdict(raw)
        assert result is None

    def test_brace_inside_string_does_not_close_object_early(self):
        """Regression: a `}` inside a JSON string literal must not be treated
        as the closing brace by the balanced-brace scan."""
        payload = {"verdict": "fixable", "issues": [
            {"severity": "high", "note": "saw `} oops` in code"},
        ], "confidence": 0.6}
        raw = f"Reviewer notes:\n{json.dumps(payload)}\ntrailing prose"
        result = pm_core._recover_reviewer_verdict(raw)
        assert result is not None
        assert result["verdict"] == "fixable"
        assert result["issues"] == payload["issues"]
        assert result["confidence"] == 0.6

    def test_escaped_quote_in_string_does_not_break_scan(self):
        """An escaped quote inside a string must not flip the in-string state
        and let a subsequent `}` close the object early."""
        payload = {"verdict": "clean", "issues": [
            {"severity": "low", "note": 'has \\"quote\\" and } brace'},
        ], "confidence": 0.9}
        raw = f"prefix\n{json.dumps(payload)}"
        result = pm_core._recover_reviewer_verdict(raw)
        assert result is not None
        assert result["verdict"] == "clean"

    def test_encode_gpu_results_uses_recovery_on_bad_json(self):
        """When reviewer output fails strict parse, _encode_gpu_results uses
        recovery and tags the episodic entry with pm:reviewer:parse-recovered."""
        record = {
            "gpu_id": "gpu-recover1",
            "spec_id": "spec-r",
            "agent_type": "reviewer",
            "intent": "review PR #42 cycle 1",
            "repo": "myrepo",
            "pr_number": 42,
            "cycle": 1,
            "status": "pending",
        }
        payload = {"verdict": "fixable", "issues": ISSUES, "confidence": 0.8}
        # Wrap in prose to force json.loads failure, but balanced-brace scan recovers it
        bad_json = f"Here is my analysis:\n{json.dumps(payload)}"

        written_tags = []
        written_content = []

        def capture_write_result(target_id, content, extra_tags=None):
            written_tags.extend(extra_tags or [])
            written_content.append(content)
            c = MagicMock()
            c.id = "comment-recover"
            return c

        mock_path = MagicMock()
        mock_path.read_text.return_value = bad_json
        mock_path.parent = object()

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=mock_path),
            patch("lapis_pm.pm_core._read_fixer_meta", return_value=None),
            patch("lapis_pm.pm_core._consume_fixer_meta"),
            patch("lapis_pm.pm_core.episodic.write_result",
                  side_effect=capture_write_result),
            patch("lapis_pm.pm_core.FAILED_DIR", object()),
        ):
            encoded, failed = pm_core._encode_gpu_results("tid")

        assert encoded == 1
        assert not failed
        assert any("pm:reviewer:parse-recovered" in t for t in written_tags), \
            f"parse-recovered tag missing from {written_tags}"
        verdict_tags = [t for t in written_tags if "verdict=fixable" in t]
        assert verdict_tags, f"Expected verdict=fixable tag in {written_tags}"

    def test_encode_gpu_results_needs_human_on_unrecoverable(self):
        """When recovery also fails, verdict stays needs-human and no
        parse-recovered tag is emitted."""
        record = {
            "gpu_id": "gpu-garbage",
            "spec_id": "spec-g",
            "agent_type": "reviewer",
            "intent": "review PR #42 cycle 1",
            "repo": "myrepo",
            "pr_number": 42,
            "cycle": 1,
            "status": "pending",
        }
        garbage = "Completely unparseable output with no verdict whatsoever!!!"

        written_tags = []

        def capture_write_result(target_id, content, extra_tags=None):
            written_tags.extend(extra_tags or [])
            c = MagicMock()
            c.id = "comment-garbage"
            return c

        mock_path = MagicMock()
        mock_path.read_text.return_value = garbage
        mock_path.parent = object()

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=mock_path),
            patch("lapis_pm.pm_core._read_fixer_meta", return_value=None),
            patch("lapis_pm.pm_core._consume_fixer_meta"),
            patch("lapis_pm.pm_core.episodic.write_result",
                  side_effect=capture_write_result),
            patch("lapis_pm.pm_core.FAILED_DIR", object()),
        ):
            encoded, failed = pm_core._encode_gpu_results("tid")

        assert encoded == 1
        assert not any("pm:reviewer:parse-recovered" in t for t in written_tags), \
            f"Unexpected parse-recovered tag in {written_tags}"
        verdict_tags = [t for t in written_tags if "verdict=needs-human" in t]
        assert verdict_tags, f"Expected verdict=needs-human tag in {written_tags}"
