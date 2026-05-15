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
        """Mapped repo causes one subprocess.run call per service unit."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("agents-core")

        assert len(calls) == 2
        assert calls[0] == ["sudo", "-n", "systemctl", "restart", "claude-queue-runner.service"]
        assert calls[1] == ["sudo", "-n", "systemctl", "restart", "gpu-queue-runner.service"]

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
