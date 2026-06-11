"""Tests for L2.D1 orphan PR reconciliation (lapis-pm-fixer-orphan-reconcile-v0).

These tests verify:
1. Marker extraction from PR bodies
2. Traceability detection
3. Auto-adoption of traceable deviant-branch PRs
4. Brief generation for untraceable PRs
5. Brief option resolution via adopt_pr action
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core, brief


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
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
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
        assert call_args.kwargs["trigger"] == "orphan-pr-untraceable"
        assert "not traceable" in call_args.kwargs["query"].lower()

        # Should set outstanding brief
        mock_set_brief.assert_called_once_with("my-target", "cid-123")

        # Should write observation
        mock_obs.assert_called()
        call_args = mock_obs.call_args
        assert "pm:orphan-untraceable" in call_args.kwargs.get("extra_tags", [])

    def test_idempotency_skip_existing_brief(self):
        """Skip brief synthesis if one already exists for the target."""
        target = MagicMock()
        target.data = {}

        pr = {
            "number": 42,
            "body": "No markers",
            "head": {"ref": "some/random/branch"}
        }

        with (
            patch("lapis_pm.pm_core._is_pr_traceable_to_target", return_value=False),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="cid-existing"),
            patch("lapis_pm.pm_core.brief.synthesize") as mock_brief,
            patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs,
        ):
            pm_core._reconcile_orphan_prs("my-target", target, "my-repo", [pr])

        # Should NOT synthesize a new brief (idempotency guard)
        mock_brief.assert_not_called()
        # Should NOT write observation for new brief
        mock_obs.assert_not_called()


class TestTemplateMarkers:
    """Test that the fixer template includes HTML comment traceability markers.

    L2.D0 requires fixer template to embed HTML-comment markers for orphan PR
    reconciliation. This is an integration test of variable substitution.
    """

    def test_fixer_template_contains_markers(self):
        """Verify fixer template YAML includes lapis-gpu-id and lapis-tid markers."""
        import yaml
        from pathlib import Path

        registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
        with open(registry_path) as f:
            registry_data = yaml.safe_load(f)

        fixer_template = registry_data.get("agents", {}).get("fixer", {}).get("system_template", "")
        assert "<!-- lapis-gpu-id:" in fixer_template, "Missing lapis-gpu-id marker in fixer template"
        assert "<!-- lapis-tid:" in fixer_template, "Missing lapis-tid marker in fixer template"

    def test_fixer_template_markers_in_pr_body(self):
        """Verify markers are positioned in PR body creation for substitution."""
        import yaml
        from pathlib import Path

        registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
        with open(registry_path) as f:
            registry_data = yaml.safe_load(f)

        fixer_template = registry_data.get("agents", {}).get("fixer", {}).get("system_template", "")
        # Markers should appear in the PR body creation section (around create_pr call)
        assert "body=" in fixer_template
        body_start = fixer_template.find("body=")
        marker_section = fixer_template[body_start:]
        assert "lapis-gpu-id" in marker_section, "lapis-gpu-id not in PR body section"
        assert "lapis-tid" in marker_section, "lapis-tid not in PR body section"


class TestAdoptAction:
    """Test the adopt_pr brief option resolution."""

    def test_adopt_pr_action_sets_adoption_fields(self):
        """Adopting a PR via brief option sets adopted_head_branch and adopted_pr_number."""
        # Mock the target
        mock_target = MagicMock()
        mock_target.pm_repo = "my-repo"
        mock_target.data = {}

        # Mock TargetStore.get to return our mock target
        with (
            patch("agents_core.targets.TargetStore") as mock_store_class,
            patch("agents_core.forgejo.get_pr") as mock_get_pr,
            patch("lapis_pm.episodic.write_observation") as mock_obs,
        ):
            mock_store = MagicMock()
            mock_store.get.return_value = mock_target
            mock_store_class.return_value = mock_store

            # Mock the PR return value from get_pr
            mock_get_pr.return_value = {
                "number": 42,
                "head": {"ref": "lapis/feature-branch"}
            }

            # Call _act_adopt_pr
            result = brief._act_adopt_pr("my-target", pr_number=42)

            # Assert adoption fields were set
            assert mock_target.data["adopted_head_branch"] == "lapis/feature-branch"
            assert mock_target.data["adopted_pr_number"] == 42
            mock_target.save.assert_called_once()

            # Assert observation was emitted with pm:orphan-adopted tag
            mock_obs.assert_called_once()
            call_args = mock_obs.call_args
            assert "pm:orphan-adopted" in call_args.kwargs.get("extra_tags", [])

            # Assert result message indicates successful adoption
            assert "adopted PR #42" in result

    def test_adopt_pr_action_via_apply_decision(self):
        """Test adopt_pr action through the full apply_decision path."""
        # Create mock target
        mock_target = MagicMock()
        mock_target.pm_repo = "my-repo"
        mock_target.data = {}

        # Create the brief options JSON that would be in a pm:brief-options comment
        options_json = {
            "brief_id": "cid-123",
            "trigger": "orphan-pr-untraceable",
            "options": [
                {
                    "id": "A",
                    "label": "Adopt this PR into the PM loop",
                    "action": {"kind": "adopt_pr", "pr": 42}
                },
                {
                    "id": "B",
                    "label": "Dismiss (leave PR open)",
                    "action": {"kind": "acknowledge_and_clear"}
                },
            ]
        }

        with (
            patch("agents_core.targets.TargetStore") as mock_store_class,
            patch("agents_core.forgejo.get_pr") as mock_get_pr,
            patch("lapis_pm.episodic.write_observation") as mock_obs,
            patch("lapis_pm.brief.read_options") as mock_read_options,
            patch("lapis_pm.pm_core._mem") as mock_mem,
            patch("lapis_pm.pm_core.get_outstanding_brief") as mock_get_brief,
            patch("lapis_pm.pm_core.clear_outstanding_brief") as mock_clear_brief,
        ):
            mock_store = MagicMock()
            mock_store.get.return_value = mock_target
            mock_store_class.return_value = mock_store

            # Mock get_pr to return PR details
            mock_get_pr.return_value = {
                "number": 42,
                "head": {"ref": "lapis/feature-branch"}
            }

            # Mock read_options to return our options JSON
            mock_read_options.return_value = options_json

            # Mock mem to indicate no prior resolution
            mock_mem.return_value.get.return_value = None
            mock_mem.return_value.set = MagicMock()

            # Mock current outstanding brief
            mock_get_brief.return_value = "cid-123"

            # Call apply_decision with adopt option
            result = brief.apply_decision("my-target", "cid-123", "A")

            # Assert success
            assert result["ok"] is True
            assert result["action_kind"] == "adopt_pr"

            # Assert adoption fields were set on target
            assert mock_target.data["adopted_head_branch"] == "lapis/feature-branch"
            assert mock_target.data["adopted_pr_number"] == 42
            mock_target.save.assert_called_once()

            # Assert observation was written with pm:orphan-adopted tag
            obs_calls = mock_obs.call_args_list
            # Two observations: one from _act_adopt_pr, one from apply_decision (brief-resolved)
            orphan_adopted_obs = [
                c for c in obs_calls
                if "pm:orphan-adopted" in c.kwargs.get("extra_tags", [])
            ]
            assert len(orphan_adopted_obs) > 0, "Missing pm:orphan-adopted observation"

            # Assert outstanding brief was cleared
            mock_clear_brief.assert_called_once()
