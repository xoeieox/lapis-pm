"""Tests for Leg 1 dispatch-dedup guards (lapis-pm-fixer-dispatch-dedup-v0).

Coverage:
  - L1.D1: Initial fixer raises when an open lapis/<tid>/ PR exists
  - L1.D1: Forgejo-unreachable during scan → dispatch proceeds (degrade-open)
  - L1.D3: Initial fixer raises when a pending fixer/fixer_retry record exists
  - L1.D2: When an open PR exists, vars_["existing_branch"] resolves to PR branch
  - L1.D2: When an open PR exists, vars_["slug"] resolves to the branch last component
  - Regression: Adopted-PR guard still raises for initial fixer on adopted target
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock
import pytest

from lapis_pm import pm_core
from agents_core.targets import TargetStore


@pytest.fixture
def target_store_mock():
    """Mock TargetStore with a test target."""
    with patch("lapis_pm.pm_core.TargetStore") as mock:
        yield mock


@pytest.fixture
def episodic_mock():
    """Mock episodic spec_summary."""
    with patch("lapis_pm.pm_core.episodic.spec_summary") as mock:
        mock.return_value = "test spec summary"
        yield mock


@pytest.fixture
def shaper_mock():
    """Mock the _SHAPER dispatch."""
    with patch("lapis_pm.pm_core._SHAPER.dispatch") as mock:
        result = MagicMock()
        result.task_id = "task-123"
        result.spec_id = "spec-456"
        mock.return_value = result
        yield mock


@pytest.fixture
def mem_mock():
    """Mock _mem for dispatch record storage."""
    with patch("lapis_pm.pm_core._mem") as mock:
        mem_instance = MagicMock()
        mem_instance.get.return_value = None
        mem_instance.set.return_value = None
        mock.return_value = mem_instance
        yield mock


@pytest.fixture
def router_portfolio_mock():
    """Mock router_portfolio emit."""
    with patch("lapis_pm.pm_core.emit_decision_dispatch") as mock:
        yield mock


class TestL1D1InitialFixerGuardOpenPR:
    """L1.D1: Initial fixer must not dispatch when target has an open lapis/<tid>/ PR."""

    def test_initial_fixer_raises_when_open_pr_exists(
        self, target_store_mock, episodic_mock, mem_mock
    ):
        """Initial fixer raises when an open PR with canonical branch exists."""
        target = MagicMock()
        target.pm_repo = "lapis/coderag"
        target.data = {}
        target_store_mock.return_value.get.return_value = target

        # Mock get_open_prs to return an open PR on canonical branch
        open_prs = [
            {
                "number": 42,
                "head": {"ref": "lapis/my-target/fix-bug"},
                "base": {"ref": "main"},
            }
        ]
        with patch(
            "agents_core.forgejo.get_open_prs", return_value=open_prs
        ) as mock_get_open_prs:
            with pytest.raises(ValueError, match="already has open PR #42"):
                pm_core.force_dispatch("my-target", "fixer", "Test intent")

    def test_initial_fixer_raises_when_adopted_branch_matches_open_pr(
        self, target_store_mock, episodic_mock, mem_mock
    ):
        """Initial fixer raises when an open PR matches adopted_head_branch."""
        target = MagicMock()
        target.pm_repo = "lapis/coderag"
        target.data = {"adopted_head_branch": "lapis/custom-branch"}
        target_store_mock.return_value.get.return_value = target

        open_prs = [
            {
                "number": 43,
                "head": {"ref": "lapis/custom-branch"},
                "base": {"ref": "main"},
            }
        ]
        with patch(
            "agents_core.forgejo.get_open_prs", return_value=open_prs
        ) as mock_get_open_prs:
            with pytest.raises(ValueError, match="already has open PR #43"):
                pm_core.force_dispatch("my-target", "fixer", "Test intent")

    def test_initial_fixer_proceeds_when_forgejo_unreachable(
        self, target_store_mock, episodic_mock, mem_mock, shaper_mock
    ):
        """Initial fixer proceeds (degrade-open) when Forgejo scan raises."""
        target = MagicMock()
        target.pm_repo = "lapis/coderag"
        target.data = {}
        target_store_mock.return_value.get.return_value = target

        with patch("lapis_pm.pm_core.get_open_prs", side_effect=Exception("Forgejo down")):
            # Should not raise; proceeds with dispatch
            task_id = pm_core.force_dispatch("my-target", "fixer", "Test intent")
            assert task_id == "task-123"

    def test_initial_fixer_not_checked_when_no_pm_repo(
        self, target_store_mock, episodic_mock, mem_mock, shaper_mock
    ):
        """L1.D1 guard skipped when target has no pm_repo."""
        target = MagicMock()
        target.pm_repo = None  # No pm_repo
        target.data = {}
        target_store_mock.return_value.get.return_value = target

        # Should not call get_open_prs; should dispatch normally
        with patch("agents_core.forgejo.get_open_prs") as mock_get_open_prs:
            task_id = pm_core.force_dispatch("my-target", "fixer", "Test intent")
            assert task_id == "task-123"
            # Verify get_open_prs was NOT called
            mock_get_open_prs.assert_not_called()


class TestL1D3TargetLevelConcurrencyGuard:
    """L1.D3: Initial fixer must not dispatch when any pending fixer/fixer_retry exists."""

    def test_initial_fixer_raises_when_pending_fixer_exists(
        self, target_store_mock, episodic_mock, mem_mock
    ):
        """Initial fixer raises when a pending fixer record exists."""
        target = MagicMock()
        target.pm_repo = None
        target.data = {}
        target_store_mock.return_value.get.return_value = target

        # Mock load_dispatched to return a pending fixer record
        pending_record = {
            "gpu_id": "gpu-999",
            "status": "pending",
            "agent_type": "fixer",
        }
        with patch(
            "lapis_pm.pm_core.load_dispatched", return_value=[pending_record]
        ):
            with pytest.raises(ValueError, match="has a pending fixer dispatch"):
                pm_core.force_dispatch("my-target", "fixer", "Test intent")

    def test_initial_fixer_raises_when_pending_fixer_retry_exists(
        self, target_store_mock, episodic_mock, mem_mock
    ):
        """Initial fixer raises when a pending fixer_retry record exists."""
        target = MagicMock()
        target.pm_repo = None
        target.data = {}
        target_store_mock.return_value.get.return_value = target

        pending_record = {
            "gpu_id": "gpu-888",
            "status": "pending",
            "agent_type": "fixer_retry",
        }
        with patch(
            "lapis_pm.pm_core.load_dispatched", return_value=[pending_record]
        ):
            with pytest.raises(ValueError, match="has a pending fixer_retry dispatch"):
                pm_core.force_dispatch("my-target", "fixer", "Test intent")

    def test_initial_fixer_proceeds_when_no_pending_fixer(
        self, target_store_mock, episodic_mock, mem_mock, shaper_mock
    ):
        """Initial fixer proceeds when no pending fixer/fixer_retry exists."""
        target = MagicMock()
        target.pm_repo = None
        target.data = {}
        target_store_mock.return_value.get.return_value = target

        # Empty or completed records only
        completed_record = {
            "gpu_id": "gpu-777",
            "status": "completed",
            "agent_type": "fixer",
        }
        with patch(
            "lapis_pm.pm_core.load_dispatched", return_value=[completed_record]
        ):
            task_id = pm_core.force_dispatch("my-target", "fixer", "Test intent")
            assert task_id == "task-123"


class TestL1D2CanonicalBranchReuse:
    """L1.D2: Initial fixer reuses canonical/adopted branch, updates slug correctly."""

    def test_fixer_resolves_existing_branch_from_open_pr(
        self, target_store_mock, episodic_mock, mem_mock, shaper_mock
    ):
        """When an open PR exists, vars_["existing_branch"] is set to the PR branch."""
        target = MagicMock()
        target.pm_repo = "lapis/coderag"
        target.data = {}
        target_store_mock.return_value.get.return_value = target

        open_prs = [
            {
                "number": 50,
                "head": {"ref": "lapis/my-target/actual-fix"},
                "base": {"ref": "develop"},
            }
        ]
        with patch("agents_core.forgejo.get_open_prs", return_value=open_prs):
            # Use fixer_retry to avoid L1.D1 guard and L1.D3 check (already tested)
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[]):
                pm_core.force_dispatch("my-target", "fixer_retry", "Test intent")
                # Verify shaper.dispatch was called with the correct vars_
                call_args = shaper_mock.call_args
                assert call_args is not None
                vars_ = call_args[1]["vars_"]
                assert vars_["existing_branch"] == "lapis/my-target/actual-fix"
                assert vars_["base_branch"] == "develop"

    def test_fixer_updates_slug_from_branch_component(
        self, target_store_mock, episodic_mock, mem_mock, shaper_mock
    ):
        """When an open PR exists, vars_["slug"] is set to the branch last component."""
        target = MagicMock()
        target.pm_repo = "lapis/coderag"
        target.data = {}
        target_store_mock.return_value.get.return_value = target

        open_prs = [
            {
                "number": 51,
                "head": {"ref": "lapis/my-target/some-branch-name"},
                "base": {"ref": "main"},
            }
        ]
        with patch("agents_core.forgejo.get_open_prs", return_value=open_prs):
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[]):
                pm_core.force_dispatch("my-target", "fixer_retry", "Test intent")
                call_args = shaper_mock.call_args
                assert call_args is not None
                vars_ = call_args[1]["vars_"]
                assert vars_["slug"] == "some-branch-name"

    def test_fixer_defaults_to_forced_slug_when_no_open_pr(
        self, target_store_mock, episodic_mock, mem_mock, shaper_mock
    ):
        """When no open PR exists, vars_["slug"] defaults to "forced"."""
        target = MagicMock()
        target.pm_repo = "lapis/coderag"
        target.data = {}
        target_store_mock.return_value.get.return_value = target

        with patch("agents_core.forgejo.get_open_prs", return_value=[]):
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[]):
                pm_core.force_dispatch("my-target", "fixer_retry", "Test intent")
                call_args = shaper_mock.call_args
                assert call_args is not None
                vars_ = call_args[1]["vars_"]
                assert vars_["slug"] == "forced"


class TestRegressionAdoptedPRGuard:
    """Regression: Adopted-PR guard still blocks initial fixer on adopted targets."""

    def test_adopted_pr_guard_still_blocks_initial_fixer(
        self, target_store_mock, episodic_mock, mem_mock
    ):
        """Adopted PR guard (unchanged) still raises for initial fixer."""
        target = MagicMock()
        target.pm_repo = None
        target.data = {"adopted_pr_number": 99}
        target_store_mock.return_value.get.return_value = target

        with patch("lapis_pm.pm_core.load_dispatched", return_value=[]):
            with pytest.raises(ValueError, match="has an adopted PR"):
                pm_core.force_dispatch("my-target", "fixer", "Test intent")

    def test_adopted_pr_guard_allows_fixer_after_first_one_landed(
        self, target_store_mock, episodic_mock, mem_mock, shaper_mock
    ):
        """Adopted PR guard allows fixer if one already landed (not the initial)."""
        target = MagicMock()
        target.pm_repo = None
        target.data = {"adopted_pr_number": 99}
        target_store_mock.return_value.get.return_value = target

        # First fixer already dispatched and completed
        prior_record = {
            "gpu_id": "gpu-old",
            "status": "completed",
            "agent_type": "fixer",
        }
        with patch(
            "lapis_pm.pm_core.load_dispatched", return_value=[prior_record]
        ):
            task_id = pm_core.force_dispatch("my-target", "fixer", "Test intent")
            assert task_id == "task-123"
