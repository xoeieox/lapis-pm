"""Tests for cycle-accounting purity (lapis-pm-cycle-accounting-sha-truth).

Coverage:
  - _fixer_retry_count counts only processed records (not failed, not pending)
  - _fixer_retry_count returns 0 when all records are failed (cascade regression)
  - _fixer_retry_count defensively ignores records with no status field
  - decide-branch end-to-end: 1 processed + 1 failed fixer_retry + cycle-1 reviewer
    verdict=fixable → decision is dispatch_reviewer cycle=2, NOT dispatch_fixer_retry
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core
from lapis_pm import authority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixer_retry_record(pr_number: int, status: str) -> dict:
    return {
        "agent_type": "fixer_retry",
        "pr_number": pr_number,
        "status": status,
        "gpu_id": f"gpu-{status}-{pr_number}",
        "ts": "2026-04-27T10:00:00-07:00",
    }


def _reviewer_record(pr_number: int, cycle: int, status: str = "processed") -> dict:
    return {
        "agent_type": "reviewer",
        "pr_number": pr_number,
        "status": status,
        "cycle": cycle,
        "ts": "2026-04-27T09:00:00-07:00",
        "gpu_id": f"gpu-reviewer-{pr_number}-{cycle}",
    }


def _make_cls(pm_authority: str = "advisory") -> MagicMock:
    """Minimal PRClassification mock for advisory path."""
    cls = MagicMock(spec=authority.PRClassification)
    cls.static_outcome = "normal"  # not auto_hold_path
    cls.verdict = "advisory"       # not "auto" or "hold" → advisory path
    cls.pr_number = 1
    cls.repo = "lapis-pm"
    cls.title = "test PR"
    cls.html_url = "http://forgejo/pr/1"
    cls.changed_paths = []
    cls.diff_loc = 10
    cls.diff = ""
    cls.issues = []
    cls.screen_verdict = "fixable"
    return cls


# ---------------------------------------------------------------------------
# _fixer_retry_count: processed-only filter
# ---------------------------------------------------------------------------

def test_fixer_retry_count_processed_only():
    """One processed + one failed → count = 1 (not 2)."""
    records = [
        _fixer_retry_record(pr_number=1, status="processed"),
        _fixer_retry_record(pr_number=1, status="failed"),
    ]
    with patch("lapis_pm.pm_core.load_dispatched", return_value=records):
        assert pm_core._fixer_retry_count("my-target", 1) == 1


def test_fixer_retry_count_all_failed_returns_zero():
    """Cascade regression: one failed + one failed → count = 0.

    This is the 2026-04-27 scenario that caused the cascade dispatch — the
    phantom failed record was counted equally with processed, making
    _fixer_retry_count return 2 instead of 1.
    """
    records = [
        _fixer_retry_record(pr_number=1, status="failed"),
        _fixer_retry_record(pr_number=1, status="failed"),
    ]
    with patch("lapis_pm.pm_core.load_dispatched", return_value=records):
        assert pm_core._fixer_retry_count("my-target", 1) == 0


def test_fixer_retry_count_no_status_field_not_counted():
    """Defensive: records missing status field are not counted."""
    records = [
        {"agent_type": "fixer_retry", "pr_number": 1},  # no status
        _fixer_retry_record(pr_number=1, status="processed"),
    ]
    with patch("lapis_pm.pm_core.load_dispatched", return_value=records):
        assert pm_core._fixer_retry_count("my-target", 1) == 1


def test_fixer_retry_count_pending_not_counted():
    """Pending records do not advance cycle accounting."""
    records = [
        _fixer_retry_record(pr_number=1, status="pending"),
        _fixer_retry_record(pr_number=1, status="processed"),
    ]
    with patch("lapis_pm.pm_core.load_dispatched", return_value=records):
        assert pm_core._fixer_retry_count("my-target", 1) == 1


def test_fixer_retry_count_different_pr_not_counted():
    """Records for a different PR do not count."""
    records = [
        _fixer_retry_record(pr_number=2, status="processed"),
        _fixer_retry_record(pr_number=1, status="processed"),
    ]
    with patch("lapis_pm.pm_core.load_dispatched", return_value=records):
        assert pm_core._fixer_retry_count("my-target", 1) == 1


# ---------------------------------------------------------------------------
# Decide-branch end-to-end: phantom-record cascade regression
# ---------------------------------------------------------------------------

def test_decide_branch_dispatches_reviewer_not_fixer_retry():
    """End-to-end: 1 processed + 1 failed fixer_retry + cycle-1 reviewer verdict=fixable
    → decision is dispatch_reviewer cycle=2, NOT dispatch_fixer_retry.

    This is the 2026-04-27 cascade scenario. Without the fix, _fixer_retry_count
    would return 2 (counting the failed record), making fixer_count=2 > reviewer_count=1,
    which triggered another fixer_retry dispatch. With the fix, fixer_count=1 ==
    reviewer_count=1, so the cycle advances to reviewer cycle 2.
    """
    pr_number = 42
    dispatched_records = [
        # One fixer_retry that actually pushed code → processed
        {
            "agent_type": "fixer_retry",
            "pr_number": pr_number,
            "status": "processed",
            "gpu_id": "gpu-processed",
            "ts": "2026-04-27T08:00:00-07:00",
        },
        # One phantom failed fixer_retry (the 2026-04-27 cascade source)
        {
            "agent_type": "fixer_retry",
            "pr_number": pr_number,
            "status": "failed",
            "gpu_id": "gpu-failed",
            "ts": "2026-04-27T07:00:00-07:00",
            "error": "ERROR: transient runner failure",
        },
        # Reviewer cycle 1 dispatch record (for _reviewer_dispatch_ts)
        {
            "agent_type": "reviewer",
            "pr_number": pr_number,
            "status": "processed",
            "cycle": 1,
            "ts": "2026-04-27T06:00:00-07:00",
            "gpu_id": "gpu-reviewer-1",
        },
    ]

    pr = {"number": pr_number, "head": {"sha": "abc123"}}
    cls = _make_cls()

    with patch("lapis_pm.pm_core.episodic.spec_summary", return_value=""), \
         patch("lapis_pm.authority.classify", return_value=cls), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=dispatched_records), \
         patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False), \
         patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False), \
         patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1), \
         patch("lapis_pm.pm_core._last_review_verdict",
               return_value={"verdict": "fixable", "issues": []}), \
         patch("lapis_pm.pm_core._reviewer_dispatch_ts",
               return_value="2026-04-27T06:00:00-07:00"), \
         patch("lapis_pm.pm_core._pr_advanced_since", return_value=True):

        decision = pm_core._decide_for_pr(
            "my-target", "lapis-pm", pr, pm_authority="advisory"
        )

    assert decision.kind == "dispatch_reviewer", (
        f"Expected dispatch_reviewer (cycle 2), got {decision.kind!r}. "
        "This is the 2026-04-27 cascade regression — _fixer_retry_count must "
        "not count failed records."
    )
    assert decision.payload["cycle"] == 2
