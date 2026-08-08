"""Integration tests for lapis-pm-degraded-panel-confidence-and-absence-claims-v0:
the R1 advisory-not-a-gate downgrade in pm_core._decide_for_pr, and the Leg 3
no-op-retry escalation via the outstanding-brief mechanism.

Coverage (DoD #8):
  - the advisory downgrade blocking auto-dispatch (a starved `fixable`
    verdict must NOT dispatch_fixer_retry)
  - the no-op-retry escalation reaching an outstanding brief
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import authority, pm_core

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

STARVED_ISSUE = {
    "severity": "high",
    "path": "agents_core/phala_tee.py",
    "note": "`_check_equal` is not defined anywhere in the diff or in the existing code I read.",
}


def _starved_verdict(verdict="fixable", issues=None):
    """Replay of the PR #220 incident: all three legs down, confidence=0.9."""
    return {
        "verdict": verdict,
        "issues": issues if issues is not None else [STARVED_ISSUE],
        "confidence": 0.0,  # already attenuated by apply_panel_starvation
        "panel_starvation": {
            "legs_down": ["local_witness", "corroboration", "second_node"],
            "starved": True,
            "confidence_raw": 0.9,
        },
    }


class TestR1AdvisoryDowngradeBlocksAutoDispatch:
    def test_starved_fixable_verdict_does_not_dispatch_fixer_retry(self):
        """A starved panel's `fixable` verdict must go to advisory_brief, never
        dispatch_fixer_retry — R1: it may inform, it may not auto-dispatch a
        fixer retry on its own authority."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=_starved_verdict()),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "advisory_brief"

    def test_starved_needs_human_verdict_does_not_hold_block(self):
        """A starved panel's `needs-human` verdict must not hold-gate the
        target either — advisory only, never a block."""
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._last_review_verdict",
                  return_value=_starved_verdict(verdict="needs-human")),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "hold")
        assert d.kind == "advisory_brief"

    def test_healthy_panel_fixable_still_dispatches_fixer_retry(self):
        """Regression fence: a fully-corroborated panel's `fixable` verdict
        keeps auto-dispatching exactly as before this spec."""
        healthy = {
            "verdict": "fixable",
            "issues": [{"severity": "low", "path": "src/foo.py", "note": "missing docstring"}],
            "confidence": 0.8,
            "panel_starvation": {"legs_down": [], "starved": False, "confidence_raw": 0.8},
        }
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=healthy),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "dispatch_fixer_retry"

    def test_verdict_missing_panel_starvation_block_treated_as_healthy(self):
        """A verdict written before this spec's `panel_starvation` annotation
        landed still carries the pre-existing corroboration/witness passes
        (those predate this spec) — the gate falls back to computing
        starvation fresh from those raw blocks, and a healthy pair of raw
        blocks must not be treated as starved just because the precomputed
        summary key is missing."""
        legacy_verdict = {
            "verdict": "fixable",
            "issues": [{"severity": "low", "path": "src/foo.py", "note": "missing docstring"}],
            "confidence": 0.8,
            "local_reviewer_witness": {"agreement": "agree"},
            "corroboration_result": {
                "verdict": "clean",
                "claim": "identifiers checked",
                "cross_node_divergence": "agree",
            },
        }
        with (
            patch("lapis_pm.pm_core.authority.classify", return_value=CLS_STATIC_PASS),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=0),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=legacy_verdict),
        ):
            d = pm_core._decide_for_pr("tid", "myrepo", PR_TEMPLATE, "advisory")
        assert d.kind == "dispatch_fixer_retry"


class TestNoopRetryEscalation:
    def test_noop_retry_against_starved_verdict_reaches_outstanding_brief(self):
        rec = {"gpu_id": "task-1", "agent_type": "fixer_retry", "pr_number": 7, "cycle": 1}
        starved = _starved_verdict()
        with (
            patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=starved),
            patch("lapis_pm.pm_core._mem") as mock_mem,
            patch("lapis_pm.pm_core.episodic.write_hold") as mock_hold,
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
            patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set_outstanding,
        ):
            mock_mem.return_value.get.return_value = None  # marker not yet recorded
            mock_brief = MagicMock()
            mock_brief.comment_id = "cid-123"
            mock_synth.return_value = mock_brief

            action = pm_core._escalate_noop_retry_if_degraded("tid", rec, 7)

        assert action is not None
        assert "noop_retry_degraded_verdict_brief" in action
        mock_hold.assert_called_once()
        mock_synth.assert_called_once()
        mock_set_outstanding.assert_called_once_with("tid", mock_brief)
        # Idempotency marker persisted so a later tick doesn't re-escalate.
        mock_mem.return_value.set.assert_called_once()
        marker_key = mock_mem.return_value.set.call_args[0][0]
        assert "noop-retry-escalation" in marker_key
        assert "tid" in marker_key
        assert "pr=7" in marker_key

    def test_noop_retry_against_refuted_absence_claim_also_escalates(self):
        rec = {"gpu_id": "task-1", "agent_type": "fixer_retry", "pr_number": 7, "cycle": 1}
        refuted_verdict = {
            "verdict": "fixable",
            "issues": [],
            "confidence": 0.9,
            "panel_starvation": {"legs_down": [], "starved": False, "confidence_raw": 0.9},
            "refuted_absence_findings": [
                {"severity": "high", "path": "foo.py", "note": "...", "absence_check": "refuted",
                 "refuted_location": "foo.py@deadbeef"},
            ],
        }
        with (
            patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=refuted_verdict),
            patch("lapis_pm.pm_core._mem") as mock_mem,
            patch("lapis_pm.pm_core.episodic.write_hold"),
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
            patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set_outstanding,
        ):
            mock_mem.return_value.get.return_value = None
            mock_brief = MagicMock()
            mock_brief.comment_id = "cid-456"
            mock_synth.return_value = mock_brief

            action = pm_core._escalate_noop_retry_if_degraded("tid", rec, 7)

        assert action is not None
        mock_set_outstanding.assert_called_once()

    def test_noop_retry_against_healthy_unrefuted_verdict_does_not_escalate(self):
        """A no-op retry against a healthy verdict is NOT this escalation's
        concern — it's handled (or not) elsewhere; this mechanism only fires
        on strong evidence the verdict itself was false."""
        rec = {"gpu_id": "task-1", "agent_type": "fixer_retry", "pr_number": 7, "cycle": 1}
        healthy_verdict = {
            "verdict": "fixable",
            "issues": [{"severity": "low", "path": "foo.py", "note": "nit"}],
            "confidence": 0.8,
            "panel_starvation": {"legs_down": [], "starved": False, "confidence_raw": 0.8},
        }
        with (
            patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=healthy_verdict),
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
        ):
            action = pm_core._escalate_noop_retry_if_degraded("tid", rec, 7)

        assert action is None
        mock_synth.assert_not_called()

    def test_noop_retry_escalation_idempotent_when_marker_already_recorded(self):
        rec = {"gpu_id": "task-1", "agent_type": "fixer_retry", "pr_number": 7, "cycle": 1}
        starved = _starved_verdict()
        with (
            patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=starved),
            patch("lapis_pm.pm_core._mem") as mock_mem,
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
        ):
            mock_mem.return_value.get.return_value = {"content": "already recorded"}
            action = pm_core._escalate_noop_retry_if_degraded("tid", rec, 7)

        assert action is None
        mock_synth.assert_not_called()

    def test_noop_retry_with_no_verdict_found_does_not_escalate(self):
        rec = {"gpu_id": "task-1", "agent_type": "fixer_retry", "pr_number": 7, "cycle": 1}
        with (
            patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=None),
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
        ):
            action = pm_core._escalate_noop_retry_if_degraded("tid", rec, 7)
        assert action is None
        mock_synth.assert_not_called()
