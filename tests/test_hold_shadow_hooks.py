"""Wiring tests for lapis-pm-hold-shadow-observer-v0's two hook call sites:
spec_review.run_spec_review's return point, and pm_core._act_brief's hold
branch. Both hooks themselves (schema, classification, dedupe, never-raise)
are covered by tests/test_hold_shadow_records.py — this file only proves the
hook is actually reached from the two call sites named in the spec.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from lapis_pm import authority, pm_core, spec_review


def test_run_spec_review_calls_observe_gate_outcome_at_its_return_site():
    """Source-presence check rather than a full run_spec_review integration
    test: run_spec_review dispatches real Facets/Council/GW legs, which is
    out of scope for a unit test. The hook itself (observe_gate_outcome) is
    exercised directly, with a real _build_brief-constructed brief, in
    test_hold_shadow_records.py."""
    source = inspect.getsource(spec_review.run_spec_review)
    assert "hold_shadow.observe_gate_outcome(brief" in source
    assert "authority=effective_authority" in source


def _make_hold_cls(pr_number: int = 42, reasons=None) -> authority.PRClassification:
    return authority.PRClassification(
        verdict="hold",
        screen_verdict="unknown",
        static_outcome=authority.StaticOutcome.auto_hold_path,
        reasons=reasons or ["held path(s) touched: infra/foo.sh"],
        issues=[],
        pr_number=pr_number,
        repo="lapis-pm",
        title="Test held PR",
        html_url="http://forgejo/pr/42",
        changed_paths=["infra/foo.sh"],
        diff_loc=3,
        diff="diff --git a/infra/foo.sh b/infra/foo.sh\n+echo hi\n",
    )


def test_act_brief_hold_branch_calls_observe_hold_fact():
    cls = _make_hold_cls()
    hold_comment = MagicMock()
    hold_comment.id = "hold-cid-1"
    brief_result = MagicMock()
    brief_result.comment_id = "brief-cid-1"
    brief_result.synthesis_failed = False
    brief_result.pushed = False
    fake_target = MagicMock()
    fake_target.pm_authority = "hold"

    with (
        patch("lapis_pm.pm_core.episodic.write_hold", return_value=hold_comment),
        patch("lapis_pm.pm_core.TargetStore") as mock_ts,
        patch("lapis_pm.pm_core._last_review_verdict", return_value=None),
        patch("lapis_pm.pm_core.brief.synthesize", return_value=brief_result),
        patch("lapis_pm.pm_core._mark_pr_classified"),
        patch("lapis_pm.pm_core._set_brief_outstanding"),
        patch("lapis_pm.pm_core._spec_bound_ts", return_value="2026-08-01T00:00:00-07:00"),
        patch("lapis_pm.hold_shadow.observe_hold_fact") as mock_observe,
    ):
        mock_ts.return_value.get.return_value = fake_target
        result = pm_core._act_brief(
            "hold-integration-tid", trigger="held PR", hold=True,
            payload={"classification": cls},
        )

    assert result == "action:brief_emitted:kind=hold:cid=brief-cid-1"
    mock_observe.assert_called_once()
    _, kwargs = mock_observe.call_args
    assert kwargs["target_id"] == "hold-integration-tid"
    assert kwargs["pr_number"] == 42
    assert kwargs["repo"] == "lapis-pm"
    assert kwargs["hold_reasons"] == ["held path(s) touched: infra/foo.sh"]
    assert kwargs["hold_comment_id"] == "hold-cid-1"
    assert kwargs["brief_comment_id"] == "brief-cid-1"
    assert kwargs["pm_authority"] == "hold"
    assert kwargs["spec_bound_ts"] == "2026-08-01T00:00:00-07:00"
    assert kwargs["has_issues"] is False


def test_act_brief_advisory_branch_never_calls_observe_hold_fact():
    """The gate-1 sibling of the wiring test above: an advisory (non-hold)
    brief must not fire the hold-fact hook at all."""
    cls = authority.PRClassification(
        verdict="advisory", screen_verdict="clean",
        static_outcome=authority.StaticOutcome.static_pass,
        reasons=[], issues=[], pr_number=7, repo="lapis-pm", title="t",
        html_url="http://x/7", changed_paths=["foo.py"], diff_loc=1, diff="",
    )
    brief_result = MagicMock()
    brief_result.comment_id = "brief-cid-adv"
    brief_result.synthesis_failed = False
    brief_result.pushed = False

    with (
        patch("lapis_pm.pm_core.TargetStore") as mock_ts,
        patch("lapis_pm.pm_core._last_review_verdict", return_value=None),
        patch("lapis_pm.pm_core.brief.synthesize", return_value=brief_result),
        patch("lapis_pm.pm_core._mark_pr_classified"),
        patch("lapis_pm.pm_core._set_brief_outstanding"),
        patch("lapis_pm.hold_shadow.observe_hold_fact") as mock_observe,
    ):
        mock_ts.return_value.get.return_value = None
        pm_core._act_brief(
            "advisory-tid", trigger="advisory PR", hold=False,
            payload={"classification": cls},
        )

    mock_observe.assert_not_called()
