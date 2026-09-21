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
"""

from __future__ import annotations

import json
from contextlib import ExitStack
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
            cls = authority.classify("myrepo", 1, "spec", pm_authority="auto", verification="machine")
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

    @pytest.fixture(autouse=True)
    def _healthy_panel(self):
        # These tests exercise _decide_for_pr's fixable/needs-human/clean/budget
        # routing, not panel-starvation (Leg 1, covered by test_panel_starvation*.py).
        # Their verdict fixtures predate the corroboration/witness blocks, so
        # without this they'd all read as starved (absent block == down) and
        # route to advisory_brief regardless of verdict content. Assume a
        # healthy (fully-corroborated) panel here.
        with patch("lapis_pm.panel_starvation.verdict_is_starved", return_value=False):
            yield

    def test_held_path_dispatches_reviewer_before_hold_brief(self):
        """Held path → reviewer dispatched once first (lapis-pm-containment-held-paths-v0
        Leg 3), THEN hold_brief once the verdict is in — never skips the machine read."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_HELD_PATH),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "dispatch_reviewer"
        assert d.payload["held_path"] is True
        assert d.payload["cycle"] == 1

    def test_held_path_pending_reviewer_is_noop(self):
        """A held-path reviewer already in flight → noop, not a re-dispatch."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_HELD_PATH),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=True),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "noop_reviewer_in_flight"

    def test_held_path_verdict_ready_raises_hold_brief_with_verdict(self):
        """Reviewer already completed once → hold_brief carrying that verdict, never
        dispatched a second time regardless of verdict content."""
        verdict_info = {"verdict": "needs-human", "issues": ISSUES}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_HELD_PATH),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict_info),
            ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "hold_brief"
        assert d.payload["reviewer_verdict"] == verdict_info

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
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "dispatch_reviewer"
        assert d.payload["mode"] == "same"
        assert d.payload["cycle"] == 1

    def test_hold_authority_uses_fresh_mode(self):
        """Hold authority → dispatch reviewer with fresh mode."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=0),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "hold")
        assert d.kind == "dispatch_reviewer"
        assert d.payload["mode"] == "fresh"

    def test_pending_reviewer_returns_noop(self):
        """Reviewer already pending for this PR → noop_reviewer_in_flight (no double-dispatch)."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=True),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "noop_reviewer_in_flight"
        assert d.payload["pr_number"] == PR_TEMPLATE["number"]
        assert d.payload["cycle"] == 1

    def test_pending_fixer_returns_noop(self):
        """Fixer already pending → noop_fixer_in_flight."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=True),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[
                {"status": "pending", "agent_type": "fixer_retry",
                 "pr_number": PR_TEMPLATE["number"], "gpu_id": "task-abc123"}
            ]),
            ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "noop_fixer_in_flight"
        assert d.payload["dispatch_id"] == "task-abc123"

    def test_fixable_cycles_below_budget_dispatches_fixer(self):
        """Reviewed (cycle 1) fixable + cycles < budget → dispatch_fixer_retry."""
        verdict = {"verdict": "fixable", "issues": ISSUES, "confidence": 0.8}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
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
        """Reviewed (cycle 3) fixable + cycles >= budget + MED issue → review_exhausted_brief.

        Advisory budget is 3 (cr-bundle item df1ac58425): one full
        fixer-fix-reviewer round-trip plus a follow-up review before the
        loud budget-exhausted brief fires."""
        med_issues = [{"severity": "med", "path": "src/foo.py", "note": "missing test coverage"}]
        verdict = {"verdict": "fixable", "issues": med_issues, "confidence": 0.8}
        history = [{"cycle": 1, "verdict": "fixable", "issues": med_issues},
                   {"cycle": 2, "verdict": "fixable", "issues": med_issues},
                   {"cycle": 3, "verdict": "fixable", "issues": med_issues}]
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=3),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
            patch("lapis_pm.pm_core._collect_review_history", return_value=history),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "review_exhausted_brief"
        assert d.payload["history"] == history

    def test_fixable_budget_exhausted_low_only_advisory_returns_advisory_brief(self):
        """LOW-only issues + budget exhausted + advisory → advisory_brief."""
        low_issues = [{"severity": "low", "path": "src/foo.py", "note": "nit"}]
        verdict = {"verdict": "fixable", "issues": low_issues, "confidence": 0.8}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=3),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
            ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "advisory_brief"
        assert d.payload["classification"].issues == low_issues

    def test_fixable_budget_exhausted_low_only_hold_returns_exhausted_brief(self):
        """LOW-only issues + budget exhausted + hold → review_exhausted_brief (hold always escalates).
        Hold budget is 4 cycles, so reviewer_count=4 simulates exhaustion."""
        low_issues = [{"severity": "low", "path": "src/foo.py", "note": "nit"}]
        verdict = {"verdict": "fixable", "issues": low_issues, "confidence": 0.8}
        history = [{"cycle": 4, "verdict": "fixable", "issues": low_issues}]
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=4),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=3),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
            patch("lapis_pm.pm_core._collect_review_history", return_value=history),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "hold")
        assert d.kind == "review_exhausted_brief"

    def test_fixable_budget_exhausted_med_present_advisory_returns_exhausted_brief(self):
        """MED issue present + budget exhausted + advisory → review_exhausted_brief."""
        med_issues = [{"severity": "med", "path": "src/bar.py", "note": "missing test"}]
        verdict = {"verdict": "fixable", "issues": med_issues, "confidence": 0.8}
        history = [{"cycle": 3, "verdict": "fixable", "issues": med_issues}]
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=3),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
            patch("lapis_pm.pm_core._collect_review_history", return_value=history),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "review_exhausted_brief"

    def test_fixable_budget_exhausted_high_present_advisory_returns_exhausted_brief(self):
        """HIGH issue present + budget exhausted + advisory → review_exhausted_brief."""
        high_issues = [{"severity": "high", "path": "src/baz.py", "note": "security hole"}]
        verdict = {"verdict": "fixable", "issues": high_issues, "confidence": 0.8}
        history = [{"cycle": 3, "verdict": "fixable", "issues": high_issues}]
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=3),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
            patch("lapis_pm.pm_core._collect_review_history", return_value=history),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "review_exhausted_brief"

    def test_fixable_budget_exhausted_mixed_low_med_advisory_returns_exhausted_brief(self):
        """Mixed LOW+MED issues + budget exhausted + advisory → review_exhausted_brief."""
        mixed_issues = [
            {"severity": "low", "path": "src/foo.py", "note": "nit"},
            {"severity": "med", "path": "src/bar.py", "note": "missing test"},
        ]
        verdict = {"verdict": "fixable", "issues": mixed_issues, "confidence": 0.8}
        history = [{"cycle": 3, "verdict": "fixable", "issues": mixed_issues}]
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=3),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
            patch("lapis_pm.pm_core._collect_review_history", return_value=history),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "review_exhausted_brief"

    def test_fixable_budget_exhausted_empty_issues_advisory_returns_exhausted_brief(self):
        """Empty issues list + budget exhausted + advisory → review_exhausted_brief (defensive)."""
        verdict = {"verdict": "fixable", "issues": [], "confidence": 0.8}
        history = [{"cycle": 3, "verdict": "fixable", "issues": []}]
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=3),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
            patch("lapis_pm.pm_core._collect_review_history", return_value=history),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "review_exhausted_brief"

    def test_needs_human_verdict_returns_hold_brief(self):
        """needs-human verdict → immediate hold_brief, no fixer retry."""
        verdict = {"verdict": "needs-human", "issues": [], "confidence": 0.6}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
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
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
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
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),  # fixer done
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
            patch("lapis_pm.pm_core._SHAPER.dispatch", side_effect=capture_dispatch),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd",
                  return_value="/srv/git/myrepo-working"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
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
            patch("lapis_pm.pm_core._SHAPER.dispatch", side_effect=capture_dispatch),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd",
                  return_value="/srv/git/myrepo-working"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
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
            patch("lapis_pm.pm_core._SHAPER.dispatch", side_effect=capture_dispatch),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd",
                  return_value="/srv/git/myrepo-working"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=prior_verdict),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("agents_core.forgejo.get_pr_diff", return_value="diff"),
        ):
            pm_core._act_dispatch_reviewer("tid", PR_TEMPLATE, CLS_STATIC_PASS,
                                            mode="same", cycle=2)

        assert "prior_review" in captured_vars
        assert "fixable" in captured_vars["prior_review"]  # prior verdict injected
        # Delta-classification framing is injected (§1)
        assert "prior_resolution" in captured_vars["prior_review"]
        assert "prior_index" in captured_vars["prior_review"]

    def test_same_reviewer_cycle1_has_empty_prior_review(self):
        """Same-reviewer mode cycle=1: prior_review is empty string."""
        captured_vars = {}

        def capture_dispatch(agent_type, target_id, user_prompt, vars_=None, **kw):
            captured_vars.update(vars_ or {})
            return self._make_dispatch_result()

        with (
            patch("lapis_pm.pm_core._SHAPER.dispatch", side_effect=capture_dispatch),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd",
                  return_value="/srv/git/myrepo-working"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("agents_core.forgejo.get_pr_diff", return_value="diff"),
            ):
            pm_core._act_dispatch_reviewer("tid", PR_TEMPLATE, CLS_STATIC_PASS,
                                            mode="same", cycle=1)

        assert captured_vars.get("prior_review", "") == ""


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
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[sha_obs]),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")

        assert d.kind == "dispatch_reviewer"
        assert d.payload["cycle"] == 2
        assert d.payload["mode"] == "same"  # advisory keeps same-reviewer mode

    def test_no_progression_guard_sha_unchanged(self):
        """reviewer=1 fixer=1 SHA NOT advanced → noop_no_change (no-progression guard)."""
        reviewer_rec = self._reviewer_dispatch_record(cycle=1, ts="2026-01-01T09:00:00")
        # SHA observation exists but BEFORE reviewer dispatch ts — does not count
        old_sha = self._sha_observation(42, "oldsha", "2026-01-01T08:00:00")

        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[old_sha]),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")

        assert d.kind == "noop_no_change"

    def test_budget_exhaustion_after_cycle3_fixable(self):
        """reviewer=3 fixer=2 SHA advanced → review_exhausted_brief (advisory budget=3).

        cr-bundle item df1ac58425: the advisory budget is 3, so exhaustion
        requires one full fixer round-trip (reviewer 1 → fixer → reviewer 2)
        plus a follow-up review (reviewer 3) before the loud brief fires."""
        # Cycle 3 reviewer dispatch record. The SHA observation must postdate
        # the cycle-3 dispatch ts — _pr_advanced_since compares against the
        # most recent reviewer dispatch (the "fixer pushed nothing" guard),
        # not cycle 2's.
        reviewer_rec = self._reviewer_dispatch_record(cycle=3, ts="2026-01-01T09:00:00")
        sha_obs = self._sha_observation(42, "sha3", "2026-01-01T11:30:00")

        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=3),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[sha_obs]),
            # The "fixer pushed nothing" guard compares the cycle-3 dispatch
            # ts against the SHA stream; the fixer's commit (sha3) landed
            # after it, so the guard passes.
            patch("lapis_pm.pm_core._pr_advanced_since", return_value=True),
            patch("lapis_pm.pm_core._last_review_verdict",
                  return_value={"verdict": "fixable", "issues": ISSUES, "confidence": 0.8}),
            patch("lapis_pm.pm_core._collect_review_history", return_value=[
                {"cycle": 1, "verdict": "fixable", "issues": ISSUES},
                {"cycle": 2, "verdict": "fixable", "issues": ISSUES},
                {"cycle": 3, "verdict": "fixable", "issues": ISSUES},
            ]),
            ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")

        assert d.kind == "review_exhausted_brief"
        assert len(d.payload["history"]) == 3

    def test_same_reviewer_mode_preserved_at_cycle2_advisory(self):
        """Advisory: cycle 2 reviewer uses mode=same (same-reviewer context)."""
        reviewer_rec = self._reviewer_dispatch_record(cycle=1, ts="2026-01-01T09:00:00")
        sha_obs = self._sha_observation(42, "newsha", "2026-01-01T10:00:00")

        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
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
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),            patch("lapis_pm.pm_core.load_dispatched", return_value=[reviewer_rec]),
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

# ---------------------------------------------------------------------------
# §6 Audit gate — delta classification tests
# ---------------------------------------------------------------------------

# Shared prior issues (3 items for most tests)
_PRIOR_ISSUES_3 = [
    {"severity": "high", "path": "tests/test_x.py", "note": "test 9 unmocked"},
    {"severity": "med",  "path": "src/validator.py", "note": "naming convention"},
    {"severity": "low",  "path": "tests/test_y.py",  "note": "test quality"},
]
_PRIOR_VERDICT_3 = {"verdict": "fixable", "issues": _PRIOR_ISSUES_3, "confidence": 0.8}

# Shared prior issues (2 items)
_PRIOR_ISSUES_2 = [
    {"severity": "high", "path": "tests/test_x.py", "note": "test 9 unmocked"},
    {"severity": "med",  "path": "src/validator.py", "note": "naming convention"},
]
_PRIOR_VERDICT_2 = {"verdict": "fixable", "issues": _PRIOR_ISSUES_2, "confidence": 0.8}


def _audit_gate_patches(
    reviewer_count: int,
    fixer_count: int,
    current_verdict: dict,
    prior_verdict: dict | None,
    pm_authority: str = "advisory",
) -> ExitStack:
    """Return an ExitStack context manager with audit-gate _decide_for_pr patches active."""
    patches = [
        patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),        patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
        patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=reviewer_count),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=fixer_count),
        patch("lapis_pm.pm_core._last_review_verdict", return_value=current_verdict),
            patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=prior_verdict),
        patch("lapis_pm.pm_core.episodic.write_observation"),
    ]
    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


class TestAuditGateDeltaClassification:
    """§6 Tests — reviewer same-mode delta classification audit gate."""

    @pytest.fixture(autouse=True)
    def _healthy_panel(self):
        # Same rationale as TestDecideForPr._healthy_panel — these fixtures
        # predate corroboration/witness blocks and test the audit gate, not
        # panel starvation.
        with patch("lapis_pm.panel_starvation.verdict_is_starved", return_value=False):
            yield

    # -----------------------------------------------------------------------
    # Test 1: Cycle 1 — audit gate is a no-op
    # -----------------------------------------------------------------------
    def test_cycle1_audit_gate_noop(self):
        """mode=same, cycle=1 — reviewer_count=1 < 2, audit gate not engaged."""
        verdict = {"verdict": "fixable", "issues": ISSUES, "confidence": 0.9}
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
            patch("lapis_pm.pm_core._review_verdict_for_cycle") as mock_rvfc,
            patch("lapis_pm.pm_core.episodic.write_observation"),
            ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        # Gate not triggered for cycle 1
        mock_rvfc.assert_not_called()
        assert d.kind == "dispatch_fixer_retry"
        assert d.payload["issues"] == ISSUES

    # -----------------------------------------------------------------------
    # Test 2: Cycle 2 — all priors addressed → advisory_brief, issues == []
    # -----------------------------------------------------------------------
    def test_cycle2_all_addressed_downgrades_to_clean(self):
        """Cycle 2: reviewer returns clean + all 3 priors addressed → advisory_brief."""
        current_verdict = {
            "verdict": "clean",
            "issues": [],
            "prior_resolution": [
                {"prior_index": 0, "status": "addressed",
                 "evidence": "tests/test_x.py:680 mock_record = MagicMock()"},
                {"prior_index": 1, "status": "addressed",
                 "evidence": "src/validator.py:45 renamed to validate_record"},
                {"prior_index": 2, "status": "addressed",
                 "evidence": "tests/test_y.py:100 added edge-case assertions"},
            ],
            "confidence": 0.95,
        }
        with _audit_gate_patches(
            reviewer_count=2, fixer_count=1,
            current_verdict=current_verdict,
            prior_verdict=_PRIOR_VERDICT_3,
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "advisory_brief"
        assert d.payload["classification"].issues == []

    # -----------------------------------------------------------------------
    # Test 3: Cycle 2 — one still_present (with prior_index) → fixer retry
    # (budget patched to 3 so cycle-2 advisory can still dispatch fixer retry)
    # -----------------------------------------------------------------------
    def test_cycle2_one_still_present_carried_to_fixer_retry(self):
        """Cycle 2: one prior still_present → dispatch_fixer_retry with 1 issue.

        The advisory budget is 3 (cr-bundle item df1ac58425), so cycle 2 can
        still dispatch a fixer retry without patching the budget. The test
        verifies audit-gate carries the still_present issue and that it has
        prior_index=1.
        """
        carried_issue = {
            "severity": "med", "path": "src/validator.py",
            "note": "still uses wrong naming at src/validator.py:30",
            "prior_index": 1,
        }
        current_verdict = {
            "verdict": "fixable",
            "issues": [carried_issue],
            "prior_resolution": [
                {"prior_index": 0, "status": "addressed",
                 "evidence": "tests/test_x.py:680 mock_record = MagicMock()"},
                {"prior_index": 1, "status": "still_present",
                 "evidence": "src/validator.py:30 still uses id-substring"},
                {"prior_index": 2, "status": "addressed",
                 "evidence": "tests/test_y.py:100 added assertions"},
            ],
            "confidence": 0.85,
        }
        stack = _audit_gate_patches(
            reviewer_count=2, fixer_count=1,
            current_verdict=current_verdict,
            prior_verdict=_PRIOR_VERDICT_3,
        )
        with stack:
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "dispatch_fixer_retry"
        assert len(d.payload["issues"]) == 1
        assert d.payload["issues"][0]["prior_index"] == 1

    # -----------------------------------------------------------------------
    # Test 4: Cycle 2 — unsubstantiated still_present (empty evidence) → clean
    # -----------------------------------------------------------------------
    def test_cycle2_unsubstantiated_still_present_drops_all(self):
        """Cycle 2: all 3 resolutions have empty evidence → dropped, verdict→clean."""
        current_verdict = {
            "verdict": "fixable",
            "issues": [
                {"severity": "high", "path": "tests/test_x.py",
                 "note": "still unmocked", "prior_index": 0},
                {"severity": "med",  "path": "src/validator.py",
                 "note": "still bad naming", "prior_index": 1},
                {"severity": "low",  "path": "tests/test_y.py",
                 "note": "still low quality", "prior_index": 2},
            ],
            "prior_resolution": [
                {"prior_index": 0, "status": "still_present", "evidence": ""},
                {"prior_index": 1, "status": "still_present", "evidence": ""},
                {"prior_index": 2, "status": "still_present", "evidence": ""},
            ],
            "confidence": 0.7,
        }
        with _audit_gate_patches(
            reviewer_count=2, fixer_count=1,
            current_verdict=current_verdict,
            prior_verdict=_PRIOR_VERDICT_3,
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        # All resolutions malformed (empty evidence) → still_present_set empty
        # → all prior_index issues dropped → verdict downgraded fixable→clean
        assert d.kind == "advisory_brief"
        assert d.payload["classification"].issues == []

    # -----------------------------------------------------------------------
    # Test 5: Cycle 2 — malformed prior_resolution entries dropped
    # -----------------------------------------------------------------------
    def test_cycle2_malformed_resolution_entries_dropped(self, caplog):
        """Cycle 2: out-of-range prior_index + unknown status → both dropped."""
        current_verdict = {
            "verdict": "fixable",
            "issues": [
                {"severity": "high", "path": "tests/test_x.py",
                 "note": "still present", "prior_index": 0},
            ],
            "prior_resolution": [
                # Out-of-range index
                {"prior_index": 99, "status": "still_present",
                 "evidence": "some/file.py:1"},
                # Unknown status
                {"prior_index": 0, "status": "unknown_status",
                 "evidence": "tests/test_x.py:50"},
            ],
            "confidence": 0.7,
        }
        import logging
        stack = _audit_gate_patches(
            reviewer_count=2, fixer_count=1,
            current_verdict=current_verdict,
            prior_verdict=_PRIOR_VERDICT_3,
        )
        stack.enter_context(caplog.at_level(logging.WARNING, logger="lapis_pm.pm_core"))
        with stack:
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        # Both malformed → still_present_set empty → prior_index=0 issue dropped
        # → verdict downgraded fixable→clean
        assert d.kind == "advisory_brief"
        assert "malformed" in caplog.text.lower()

    # -----------------------------------------------------------------------
    # Test 6: Cycle 2 — new issue (no prior_index) + still_present → both kept
    # -----------------------------------------------------------------------
    def test_cycle2_new_issue_and_still_present_both_kept(self):
        """Cycle 2: 1 still_present + 1 new issue → 2 issues in live set."""
        still_present_issue = {
            "severity": "high", "path": "tests/test_x.py",
            "note": "still unmocked at line 680", "prior_index": 0,
        }
        new_issue = {
            "severity": "low", "path": "src/new_file.py",
            "note": "newly introduced typo",
            # No prior_index — this is a new issue
        }
        current_verdict = {
            "verdict": "fixable",
            "issues": [still_present_issue, new_issue],
            "prior_resolution": [
                {"prior_index": 0, "status": "still_present",
                 "evidence": "tests/test_x.py:680 still no MagicMock"},
                {"prior_index": 1, "status": "addressed",
                 "evidence": "src/validator.py:45 renamed correctly"},
            ],
            "confidence": 0.8,
        }
        stack = _audit_gate_patches(
            reviewer_count=2, fixer_count=1,
            current_verdict=current_verdict,
            prior_verdict=_PRIOR_VERDICT_2,
        )
        with stack:
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "dispatch_fixer_retry"
        assert len(d.payload["issues"]) == 2
        # The new issue (no prior_index) is preserved
        new_issues = [i for i in d.payload["issues"] if "prior_index" not in i]
        assert len(new_issues) == 1
        assert new_issues[0]["path"] == "src/new_file.py"
        # The still_present issue is preserved
        carried = [i for i in d.payload["issues"] if i.get("prior_index") == 0]
        assert len(carried) == 1

    # -----------------------------------------------------------------------
    # Test 7: Fresh mode cycle 2 — audit gate not engaged
    # -----------------------------------------------------------------------
    def test_fresh_mode_cycle2_audit_gate_not_engaged(self):
        """mode=fresh (hold authority), cycle=2 — audit gate skipped entirely."""
        current_verdict = {
            "verdict": "clean",
            "issues": [],
            # prior_resolution would be ignored in fresh mode
            "prior_resolution": [
                {"prior_index": 0, "status": "addressed", "evidence": "src/foo.py:1"},
            ],
            "confidence": 0.9,
        }
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=3),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=current_verdict),
            patch("lapis_pm.pm_core._review_verdict_for_cycle") as mock_rvfc,
            patch("lapis_pm.pm_core.episodic.write_observation"),
            ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "hold")
        # Fresh mode → audit gate not triggered → _review_verdict_for_cycle not called
        mock_rvfc.assert_not_called()
        assert d.kind == "hold_brief"

    # -----------------------------------------------------------------------
    # Test 8: Telemetry writes
    # -----------------------------------------------------------------------
    def test_cycle2_telemetry_writes_per_resolution_and_rollup(self, tmp_path):
        """Cycle 3 (advisory budget=3, cr-bundle item df1ac58425): 2 valid
        resolutions → 2 prior-resolution observations + 1 rollup."""
        current_verdict = {
            "verdict": "fixable",
            "issues": [
                {"severity": "high", "path": "tests/test_x.py",
                 "note": "still unmocked", "prior_index": 0},
            ],
            "prior_resolution": [
                {"prior_index": 0, "status": "still_present",
                 "evidence": "tests/test_x.py:680 line unchanged"},
                {"prior_index": 1, "status": "addressed",
                 "evidence": "src/validator.py:45 renamed"},
            ],
            "confidence": 0.8,
        }
        written_obs: list[tuple[str, list[str]]] = []

        def capture_write_obs(target_id, content, extra_tags=None):
            written_obs.append((content, list(extra_tags or [])))

        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=3),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=2),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=current_verdict),
            patch("lapis_pm.pm_core._review_verdict_for_cycle",
                  return_value=_PRIOR_VERDICT_2),
            patch("lapis_pm.pm_core.episodic.write_observation",
                  side_effect=capture_write_obs),
            ):
            pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")

        resolution_obs = [
            (c, t) for c, t in written_obs
            if any("pm:reviewer-prior-resolution" in tag for tag in t)
        ]
        rollup_obs = [
            (c, t) for c, t in written_obs
            if any("pm:reviewer-audit-gate" in tag for tag in t)
        ]
        assert len(resolution_obs) == 2, (
            f"Expected 2 prior-resolution observations, got {len(resolution_obs)}: {written_obs}"
        )
        assert len(rollup_obs) == 1, (
            f"Expected 1 rollup observation, got {len(rollup_obs)}: {written_obs}"
        )
        # Rollup contains cycle info
        rollup_content = rollup_obs[0][0]
        assert "audit-gate:" in rollup_content
        assert "cycle=3" in rollup_content

    # -----------------------------------------------------------------------
    # Test 9: Cycle 2 — prior cycle 1 was clean (issues=[]) → audit gate no-op
    # -----------------------------------------------------------------------
    def test_cycle2_clean_prior_audit_gate_noop(self):
        """Cycle 2: prior cycle returned clean with issues=[] → no priors → gate is no-op."""
        prior_clean = {"verdict": "clean", "issues": [], "confidence": 0.95}
        current_verdict = {
            "verdict": "fixable",
            "issues": ISSUES,
            "confidence": 0.8,
        }
        stack = _audit_gate_patches(
            reviewer_count=2, fixer_count=1,
            current_verdict=current_verdict,
            prior_verdict=prior_clean,
        )
        with stack:
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        # No prior issues → audit gate is a no-op → issues pass through unchanged
        assert d.kind == "dispatch_fixer_retry"
        assert d.payload["issues"] == ISSUES

    # -----------------------------------------------------------------------
    # Test 10: Defense-in-depth path-duplicate without prior_index
    # -----------------------------------------------------------------------
    def test_cycle2_path_duplicate_without_prior_index_dropped(self, caplog):
        """Cycle 2: issue matches prior path but omits prior_index + prior is addressed → drop."""
        # Reviewer omits prior_index on an issue that matches a prior path
        uncited_issue = {
            "severity": "high", "path": "tests/test_x.py",
            "note": "still missing mock",
            # No prior_index — defense-in-depth path matching kicks in
        }
        current_verdict = {
            "verdict": "fixable",
            "issues": [uncited_issue],
            "prior_resolution": [
                # Prior for tests/test_x.py (index 0) is ADDRESSED
                {"prior_index": 0, "status": "addressed",
                 "evidence": "tests/test_x.py:680 mock_record = MagicMock()"},
                {"prior_index": 1, "status": "addressed",
                 "evidence": "src/validator.py:45 renamed"},
                {"prior_index": 2, "status": "addressed",
                 "evidence": "tests/test_y.py:100 assertions added"},
            ],
            "confidence": 0.7,
        }
        import logging
        stack = _audit_gate_patches(
            reviewer_count=2, fixer_count=1,
            current_verdict=current_verdict,
            prior_verdict=_PRIOR_VERDICT_3,
        )
        stack.enter_context(caplog.at_level(logging.WARNING, logger="lapis_pm.pm_core"))
        with stack:
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        # Path matches a prior but that prior is addressed → dropped via defense-in-depth
        assert d.kind == "advisory_brief"
        assert "uncited prior-path" in caplog.text

    # -----------------------------------------------------------------------
    # Test 11: Verdict downgrade needs-human→clean
    # -----------------------------------------------------------------------
    def test_cycle2_needs_human_all_priors_addressed_downgrades_to_clean(self):
        """Cycle 2: needs-human + issues=[] + all priors addressed → downgrade to clean."""
        current_verdict = {
            "verdict": "needs-human",
            "issues": [],
            "prior_resolution": [
                {"prior_index": 0, "status": "addressed",
                 "evidence": "tests/test_x.py:680 mock_record = MagicMock()"},
                {"prior_index": 1, "status": "addressed",
                 "evidence": "src/validator.py:45 renamed"},
            ],
            "confidence": 0.85,
        }
        with _audit_gate_patches(
            reviewer_count=2, fixer_count=1,
            current_verdict=current_verdict,
            prior_verdict=_PRIOR_VERDICT_2,
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        # All 2 priors addressed, no new issues → downgrade needs-human→clean
        assert d.kind == "advisory_brief"
        assert d.payload["classification"].issues == []


# ---------------------------------------------------------------------------
# _review_verdict_for_cycle helper
# ---------------------------------------------------------------------------

class TestReviewVerdictForCycle:

    def test_returns_none_on_no_comments(self):
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]):
            result = pm_core._review_verdict_for_cycle("tid", 42, 1)
        assert result is None

    def test_returns_verdict_for_specific_cycle(self):
        c1 = _make_review_comment(42, 1, "fixable", ISSUES)
        c2 = _make_review_comment(42, 2, "clean")
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[c1, c2]):
            result1 = pm_core._review_verdict_for_cycle("tid", 42, 1)
            result2 = pm_core._review_verdict_for_cycle("tid", 42, 2)
        assert result1 is not None
        assert result1["verdict"] == "fixable"
        assert result1["issues"] == ISSUES
        assert result2 is not None
        assert result2["verdict"] == "clean"

    def test_ignores_pending_cycle(self):
        c1 = _make_review_comment(42, 1, "fixable", ISSUES)
        pending = MagicMock()
        pending.tags = ["pm:reviewer:pr=42:cycle=1:verdict=pending"]
        pending.content = ""
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[pending, c1]):
            result = pm_core._review_verdict_for_cycle("tid", 42, 1)
        assert result["verdict"] == "fixable"

    def test_different_pr_not_returned(self):
        c = _make_review_comment(99, 1, "clean")
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[c]):
            result = pm_core._review_verdict_for_cycle("tid", 42, 1)
        assert result is None


# ---------------------------------------------------------------------------
# Registry model selection (lapis-pm-advisory-sonnet-reviewer-v0)
# ---------------------------------------------------------------------------

class TestRegistryReviewerModels:
    """Verify registry.yaml reviewer agent model assignments.

    Advisory authority → reviewer (model=sonnet).
    Hold authority → reviewer_fresh (model=opus).
    Spec-review → spec_reviewer (model=opus).
    """

    def test_advisory_reviewer_agent_uses_sonnet(self):
        """Advisory authority selects reviewer agent, which must be model=sonnet."""
        agent = pm_core._SHAPER.get_agent("reviewer")
        assert agent.model == "sonnet", (
            f"reviewer agent model is '{agent.model}'; expected 'sonnet' "
            "(lapis-pm-advisory-sonnet-reviewer-v0)"
        )

    def test_hold_reviewer_fresh_agent_uses_opus(self):
        """Hold authority selects reviewer_fresh agent, which must remain model=opus."""
        agent = pm_core._SHAPER.get_agent("reviewer_fresh")
        assert agent.model == "opus", (
            f"reviewer_fresh agent model is '{agent.model}'; expected 'opus' "
            "(hold authority unchanged)"
        )

    def test_spec_reviewer_agent_uses_sonnet(self):
        """Spec-review gate uses spec_reviewer agent, switched to model=sonnet (reference-only leg)."""
        agent = pm_core._SHAPER.get_agent("spec_reviewer")
        assert agent.model == "sonnet", (
            f"spec_reviewer agent model is '{agent.model}'; expected 'sonnet' "
            "(lapis-pm-spec-review-sonnet-reference-default-v0: switched from opus)"
        )
