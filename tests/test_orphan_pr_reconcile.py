"""Tests for L2.D1 orphan PR reconciliation (lapis-pm-fixer-orphan-reconcile-v0).

These tests verify:
1. Marker extraction from PR bodies
2. Traceability detection
3. Auto-adoption of traceable deviant-branch PRs
4. Brief generation for untraceable PRs
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core


class TestMarkerExtraction:
    """Test _extract_pr_markers function."""

    def test_extract_markers_both_present(self):
        """Extract both gpu_id and tid markers from PR body."""
        body = """
        Some PR description here.

        <!-- lapis-gpu-id: gpu-123-abc -->
        <!-- lapis-tid: my-target-id -->

        More text after markers.
        """
        markers = pm_core._extract_pr_markers(body)
        assert markers["gpu_id"] == "gpu-123-abc"
        assert markers["tid"] == "my-target-id"

    def test_extract_markers_gpu_id_only(self):
        """Extract only gpu_id marker."""
        body = "Some text\n<!-- lapis-gpu-id: gpu-456-def -->\nMore text"
        markers = pm_core._extract_pr_markers(body)
        assert markers["gpu_id"] == "gpu-456-def"
        assert markers["tid"] is None

    def test_extract_markers_tid_only(self):
        """Extract only tid marker."""
        body = "Some text\n<!-- lapis-tid: target-123 -->\nMore text"
        markers = pm_core._extract_pr_markers(body)
        assert markers["gpu_id"] is None
        assert markers["tid"] == "target-123"

    def test_extract_markers_none_present(self):
        """Return None for both when no markers found."""
        body = "Some PR description with no markers"
        markers = pm_core._extract_pr_markers(body)
        assert markers["gpu_id"] is None
        assert markers["tid"] is None

    def test_extract_markers_empty_body(self):
        """Handle empty or None body."""
        assert pm_core._extract_pr_markers(None) == {"gpu_id": None, "tid": None}
        assert pm_core._extract_pr_markers("") == {"gpu_id": None, "tid": None}

    def test_extract_markers_malformed(self):
        """Handle malformed marker comments gracefully."""
        body = "<!-- lapis-gpu-id invalid --> <!-- lapis-tid: valid-id -->"
        markers = pm_core._extract_pr_markers(body)
        # gpu_id malformed, should not match
        assert markers["gpu_id"] is None
        assert markers["tid"] == "valid-id"


class TestTraceability:
    """Test _is_pr_traceable_to_target function."""

    def test_traceable_via_tid_marker(self):
        """PR is traceable if tid marker matches target."""
        pr = {
            "number": 42,
            "body": "PR description\n<!-- lapis-tid: my-target -->"
        }
        assert pm_core._is_pr_traceable_to_target("my-target", pr) is True

    def test_traceable_via_gpu_id_marker(self):
        """PR is traceable if gpu_id marker matches dispatched record."""
        pr = {
            "number": 42,
            "body": "PR description\n<!-- lapis-gpu-id: gpu-123 -->"
        }

        with patch("lapis_pm.pm_core.load_dispatched") as mock_load:
            mock_load.return_value = [
                {"gpu_id": "gpu-123", "agent_type": "fixer"},
            ]
            assert pm_core._is_pr_traceable_to_target("my-target", pr) is True

    def test_not_traceable_no_markers(self):
        """PR is not traceable if no markers and no dispatched record."""
        pr = {
            "number": 42,
            "body": "PR description with no markers"
        }

        with patch("lapis_pm.pm_core.load_dispatched") as mock_load:
            mock_load.return_value = []
            assert pm_core._is_pr_traceable_to_target("my-target", pr) is False

    def test_not_traceable_wrong_tid(self):
        """PR is not traceable if tid marker doesn't match target."""
        pr = {
            "number": 42,
            "body": "<!-- lapis-tid: other-target -->"
        }
        assert pm_core._is_pr_traceable_to_target("my-target", pr) is False

    def test_not_traceable_gpu_id_no_match(self):
        """PR is not traceable if gpu_id doesn't match any dispatch."""
        pr = {
            "number": 42,
            "body": "<!-- lapis-gpu-id: gpu-999 -->"
        }

        with patch("lapis_pm.pm_core.load_dispatched") as mock_load:
            mock_load.return_value = [
                {"gpu_id": "gpu-123", "agent_type": "fixer"},
            ]
            assert pm_core._is_pr_traceable_to_target("my-target", pr) is False


class TestReconciliation:
    """Test _reconcile_orphan_prs function."""

    def test_auto_adopt_traceable_pr(self):
        """Auto-adopt a traceable deviant-branch PR."""
        target = MagicMock()
        target.data = {}

        pr = {
            "number": 42,
            "body": "<!-- lapis-tid: my-target -->\nPR description",
            "head": {"ref": "lapis/unrelated-fix"}
        }

        with (
            patch("lapis_pm.pm_core._is_pr_traceable_to_target", return_value=True),
            patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs,
        ):
            pm_core._reconcile_orphan_prs("my-target", target, "my-repo", [pr])

        # Check adoption was recorded
        assert target.data["adopted_head_branch"] == "lapis/unrelated-fix"
        assert target.data["adopted_pr_number"] == 42
        target.save.assert_called_once()

        # Check observation was written
        mock_obs.assert_called_once()
        call_args = mock_obs.call_args
        assert "pm:orphan-adopted" in call_args.kwargs.get("extra_tags", [])

    def test_skip_canonical_branch_pr(self):
        """Skip PRs on canonical lapis/<tid>/ branches."""
        target = MagicMock()
        target.data = {}

        pr = {
            "number": 42,
            "body": "No markers",
            "head": {"ref": "lapis/my-target/some-fix"}
        }

        with patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs:
            pm_core._reconcile_orphan_prs("my-target", target, "my-repo", [pr])

        # Should not create a brief for canonical branches
        mock_obs.assert_not_called()
        target.save.assert_not_called()

    def test_skip_already_adopted_pr(self):
        """Skip PR if already adopted."""
        target = MagicMock()
        target.data = {"adopted_pr_number": 42}

        pr = {
            "number": 42,
            "body": "No markers",
            "head": {"ref": "some/deviant/branch"}
        }

        with patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs:
            pm_core._reconcile_orphan_prs("my-target", target, "my-repo", [pr])

        # Should not process already-adopted PR
        mock_obs.assert_not_called()
        target.save.assert_not_called()

    def test_brief_for_untraceable_pr(self):
        """Raise a brief for untraceable deviant PR."""
        target = MagicMock()
        target.data = {}

        pr = {
            "number": 42,
            "body": "No markers",
            "head": {"ref": "some/random/branch"}
        }

        with (
            patch("lapis_pm.pm_core._is_pr_traceable_to_target", return_value=False),
            patch("lapis_pm.pm_core.brief.synthesize") as mock_brief,
            patch("lapis_pm.pm_core.set_outstanding_brief") as mock_set_brief,
            patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs,
        ):
            mock_brief_obj = MagicMock()
            mock_brief_obj.comment_id = "cid-123"
            mock_brief.return_value = mock_brief_obj

            pm_core._reconcile_orphan_prs("my-target", target, "my-repo", [pr])

        # Should not adopt
        target.save.assert_not_called()

        # Should create a brief
        mock_brief.assert_called_once()
        call_args = mock_brief.call_args
        assert "orphan" in call_args.kwargs["trigger"].lower()
        assert "not traceable" in call_args.kwargs["query"].lower()

        # Should set outstanding brief
        mock_set_brief.assert_called_once_with("my-target", "cid-123")

        # Should write observation
        mock_obs.assert_called()
        call_args = mock_obs.call_args
        assert "pm:orphan-untraceable" in call_args.kwargs.get("extra_tags", [])
