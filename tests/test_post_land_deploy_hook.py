"""Unit tests for _post_land_deploy_hook in pm_core."""

import subprocess
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from lapis_pm import pm_core


def _make_completed_process(returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=stderr
    )


class TestPostLandDeployHook:
    """Tests for _post_land_deploy_hook."""

    def test_mapped_repo_fires_restart_per_unit(self, capsys):
        """Mapped repo causes git pull then one subprocess.run call per service unit."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("agents-core")

        # git pull first, then one restart per service unit
        assert len(calls) == 3
        assert calls[0] == ["git", "-C", "/srv/git/agents-core-working", "pull", "--ff-only", "origin", "main"]
        assert calls[1] == ["sudo", "-n", "systemctl", "restart", "claude-queue-runner.service"]
        assert calls[2] == ["sudo", "-n", "systemctl", "restart", "gpu-queue-runner.service"]

    def test_unmapped_repo_is_noop(self):
        """Unmapped repo results in zero subprocess.run calls."""
        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run") as mock_run:
                pm_core._post_land_deploy_hook("foyer")
        mock_run.assert_not_called()

    def test_none_repo_is_noop(self):
        """None repo results in zero subprocess.run calls."""
        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run") as mock_run:
                pm_core._post_land_deploy_hook(None)
        mock_run.assert_not_called()

    def test_restart_failure_does_not_raise(self, capsys):
        """Non-zero returncode is logged to stderr but does not raise."""
        def fake_run(cmd, **kwargs):
            return _make_completed_process(returncode=1, stderr="Failed to restart unit")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("agents-core")  # must not raise

        captured = capsys.readouterr()
        assert "claude-queue-runner.service" in captured.err
        assert "rc=1" in captured.err

    def test_timeout_does_not_raise(self, capsys):
        """subprocess.TimeoutExpired is caught and logged; no exception propagates."""
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=60)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("agents-core")  # must not raise

        captured = capsys.readouterr()
        assert "errored" in captured.err


class TestPostLandGitPull:
    """Tests for _post_land_git_pull."""

    def test_pull_mapped_repo_fires_git_pull(self):
        """Mapped repo causes a git pull call with the correct args."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("lapis-pm")

        assert len(calls) == 1
        assert calls[0] == ["git", "-C", "/srv/lapis/lapis-pm", "pull", "--ff-only", "origin", "main"]

    def test_pull_unmapped_repo_is_noop(self):
        """Unmapped repo results in zero subprocess.run calls."""
        with patch("lapis_pm.pm_core.subprocess.run") as mock_run:
            pm_core._post_land_git_pull("foyer")
        mock_run.assert_not_called()

    def test_pull_none_repo_is_noop(self):
        """None repo results in zero subprocess.run calls."""
        with patch("lapis_pm.pm_core.subprocess.run") as mock_run:
            pm_core._post_land_git_pull(None)
        mock_run.assert_not_called()

    def test_pull_failure_does_not_block_restart(self):
        """A failing git pull must not prevent the subsequent systemctl restart."""
        sudo_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git":
                return _make_completed_process(returncode=1, stderr="not ff")
            sudo_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("agents-core")

        assert len(sudo_calls) == 2
        assert sudo_calls[0] == ["sudo", "-n", "systemctl", "restart", "claude-queue-runner.service"]
        assert sudo_calls[1] == ["sudo", "-n", "systemctl", "restart", "gpu-queue-runner.service"]

    def test_pull_ff_only_in_args(self):
        """--ff-only is always present in the git pull args."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("lapis-pm")

        assert len(calls) == 1
        assert "--ff-only" in calls[0]

    def test_pull_oserror_does_not_raise(self):
        """OSError from subprocess.run must not propagate."""
        def fake_run(cmd, **kwargs):
            raise OSError("git not found")

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("lapis-pm")  # must not raise

    def test_pull_timeout_does_not_raise(self, capsys):
        """subprocess.TimeoutExpired must not propagate; stderr contains 'errored'."""
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("lapis-pm")  # must not raise

        captured = capsys.readouterr()
        assert "errored" in captured.err
