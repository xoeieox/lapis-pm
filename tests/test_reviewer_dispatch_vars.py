"""Tests for reviewer dispatch vars — lapis-pm-reviewer-full-context-v0.

Asserts that _act_dispatch_reviewer populates vars_ with existing_branch and
base_branch keys whose values match the dispatched PR's head/base refs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lapis_pm import authority, pm_core


PR_WITH_REFS = {
    "number": 42,
    "title": "feat: add widgets",
    "html_url": "https://forgejo/Erah/myrepo/pulls/42",
    "head": {"ref": "lapis/my-target/widgets"},
    "base": {"ref": "main"},
    "mergeable": True,
}

PR_NO_BASE = {
    "number": 7,
    "title": "feat: no base ref",
    "html_url": "https://forgejo/Erah/myrepo/pulls/7",
    "head": {"ref": "lapis/my-target/branch"},
    "base": {},
    "mergeable": True,
}

PR_NO_HEAD = {
    "number": 8,
    "title": "feat: no head ref",
    "html_url": "https://forgejo/Erah/myrepo/pulls/8",
    "head": {},
    "base": {"ref": "develop"},
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


def _make_dispatch_result():
    result = MagicMock()
    result.task_id = "task-abc"
    result.spec_id = "spec-abc"
    return result


def _call_reviewer(pr, cls=None, mode="same", cycle=1):
    """Call _act_dispatch_reviewer with full mocking; return captured vars_."""
    if cls is None:
        cls = CLS
    captured = {}

    def capture_dispatch(agent_type, target_id, user_prompt, vars_=None, **kw):
        captured.update(vars_ or {})
        return _make_dispatch_result()

    with (
        patch.object(pm_core._SHAPER, "dispatch", side_effect=capture_dispatch),
        patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd",
              return_value="/srv/git/myrepo-working"),
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
        patch("lapis_pm.pm_core.episodic.write_dispatch"),
        patch("lapis_pm.pm_core.append_dispatched"),
        patch("lapis_pm.pm_core._increment_review_gate_counter", return_value=1),
        patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        patch("agents_core.forgejo.get_pr_diff", return_value="diff text"),
    ):
        pm_core._act_dispatch_reviewer("tid", pr, cls, mode=mode, cycle=cycle)

    return captured


class TestReviewerDispatchBranchVars:

    def test_existing_branch_matches_pr_head_ref(self):
        """existing_branch in vars_ must equal pr['head']['ref']."""
        vars_ = _call_reviewer(PR_WITH_REFS)
        assert "existing_branch" in vars_, "existing_branch missing from vars_"
        assert vars_["existing_branch"] == "lapis/my-target/widgets"

    def test_base_branch_matches_pr_base_ref(self):
        """base_branch in vars_ must equal pr['base']['ref']."""
        vars_ = _call_reviewer(PR_WITH_REFS)
        assert "base_branch" in vars_, "base_branch missing from vars_"
        assert vars_["base_branch"] == "main"

    def test_base_branch_falls_back_to_main_when_missing(self):
        """base_branch falls back to 'main' when pr['base']['ref'] is absent."""
        vars_ = _call_reviewer(PR_NO_BASE)
        assert vars_["base_branch"] == "main"

    def test_existing_branch_falls_back_when_head_ref_missing(self):
        """existing_branch falls back to lapis/<tid>/<pr_number> when head ref absent."""
        vars_ = _call_reviewer(PR_NO_HEAD)
        assert vars_["existing_branch"].startswith("lapis/")
        assert "8" in vars_["existing_branch"]

    def test_both_branch_vars_present_in_same_call(self):
        """Both existing_branch and base_branch must be present in the same vars_."""
        vars_ = _call_reviewer(PR_WITH_REFS)
        assert "existing_branch" in vars_
        assert "base_branch" in vars_

    def test_branch_vars_present_in_fresh_mode(self):
        """Branch vars are present even in fresh-reviewer mode."""
        vars_ = _call_reviewer(PR_WITH_REFS, mode="fresh", cycle=1)
        assert vars_["existing_branch"] == "lapis/my-target/widgets"
        assert vars_["base_branch"] == "main"

    def test_branch_vars_present_in_cycle2(self):
        """Branch vars are passed through on cycle 2 (same-reviewer mode)."""
        prior_verdict = {"verdict": "fixable", "issues": [], "confidence": 0.8}
        with patch("lapis_pm.pm_core._last_review_verdict", return_value=prior_verdict):
            vars_ = _call_reviewer(PR_WITH_REFS, mode="same", cycle=2)
        assert vars_["existing_branch"] == "lapis/my-target/widgets"
        assert vars_["base_branch"] == "main"
