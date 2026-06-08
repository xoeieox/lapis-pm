"""Unit tests for _post_land_deploy_hook in pm_core."""

import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from lapis_pm import pm_core


def _make_completed_process(returncode=0, stderr="", stdout=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestPostLandDeployHook:
    """Tests for _post_land_deploy_hook."""

    def test_mapped_repo_fires_restart_per_unit(self, capsys):
        """Mapped repo causes git pull then one subprocess.run call per service unit."""
        pull_calls = []
        restart_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "sudo":
                restart_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("agents-core")

        assert len(pull_calls) == 1
        assert pull_calls[0] == [
            "git", "-C", "/srv/git/agents-core-working", "pull", "--ff-only", "origin", "main"
        ]
        assert len(restart_calls) == 2
        assert restart_calls[0] == ["sudo", "-n", "systemctl", "restart", "claude-queue-runner.service"]
        assert restart_calls[1] == ["sudo", "-n", "systemctl", "restart", "gpu-queue-runner.service"]

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
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            return _make_completed_process(returncode=1, stderr="Failed to restart unit")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("agents-core")  # must not raise

        captured = capsys.readouterr()
        # The pull fails (rc=1) and the restart also fails — both are logged
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

    def test_pull_lapis_pm_targets_deploy_clone_only(self):
        """lapis-pm pull targets only /srv/git/lapis-pm (the deploy clone = run path).

        -working is intentionally excluded: it is the dev/PM tree, not the run path.
        Dirty state there must never block a deploy.
        """
        pull_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            pull_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="")

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("lapis_pm.pm_core._write_deploy_log"),
        ):
            pm_core._post_land_git_pull("lapis-pm")

        assert len(pull_calls) == 1
        assert pull_calls[0] == [
            "git", "-C", "/srv/git/lapis-pm", "pull", "--ff-only", "origin", "main"
        ]
        assert not any("/srv/lapis/lapis-pm" in str(c) for c in pull_calls), (
            "-working must not be pulled"
        )

    def test_lapis_pm_not_in_post_land_restart(self):
        """lapis-pm must not be in _POST_LAND_RESTART: tick picks up code on next fire."""
        assert "lapis-pm" not in pm_core._POST_LAND_RESTART

    def test_pull_mapped_repo_fires_git_pull(self):
        """Mapped repo (agents-core) causes a git pull call with the correct args."""
        pull_calls = []

        def fake_run(cmd, **kwargs):
            if "pull" in cmd:
                pull_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("agents-core")

        assert len(pull_calls) == 1
        assert pull_calls[0] == [
            "git", "-C", "/srv/git/agents-core-working", "pull", "--ff-only", "origin", "main"
        ]

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
            if cmd[0] == "git" and "pull" in cmd:
                return _make_completed_process(returncode=1, stderr="not ff")
            if cmd[0] == "git" and "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            sudo_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("agents-core")

        assert len(sudo_calls) == 2
        assert sudo_calls[0] == ["sudo", "-n", "systemctl", "restart", "claude-queue-runner.service"]
        assert sudo_calls[1] == ["sudo", "-n", "systemctl", "restart", "gpu-queue-runner.service"]

    def test_pull_ff_only_in_all_args(self):
        """--ff-only is present in the git pull call for lapis-pm."""
        pull_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            pull_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="")

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("lapis_pm.pm_core._write_deploy_log"),
        ):
            pm_core._post_land_git_pull("lapis-pm")

        assert len(pull_calls) == 1
        assert "--ff-only" in pull_calls[0]

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

    def test_lapis_pm_pull_failure_sends_pushover(self):
        """A failed pull against the lapis-pm deploy clone triggers a Pushover alert."""
        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            return _make_completed_process(returncode=1, stderr="not fast-forward")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "title": title, "priority": priority})
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_git_pull("lapis-pm")

        assert len(notify_calls) == 1, "Expected exactly one Pushover call on pull failure"
        assert "deploy pull failed" in notify_calls[0]["title"]
        assert "lapis-pm" in notify_calls[0]["title"]

    def test_agents_core_pull_failure_does_not_send_pushover(self):
        """A failed pull for agents-core does NOT trigger a Pushover alert."""
        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            return _make_completed_process(returncode=1, stderr="not ff")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_git_pull("agents-core")

        assert len(notify_calls) == 0, "agents-core pull failure must not send Pushover"

    def test_lapis_pm_pull_success_writes_deploy_log_on_advance(self, tmp_path):
        """A successful lapis-pm pull that advances HEAD writes a deploy log entry."""
        log_file = tmp_path / "lapis-pm-deploy-log.md"
        log_file.write_text("# log\n---\n")

        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            if "rev-parse" in cmd:
                # Return different SHAs for pre and post
                return _make_completed_process(
                    returncode=0,
                    stdout="oldsha0" if call_count["n"] <= 1 else "newsha1",
                )
            return _make_completed_process(returncode=0, stdout="")

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
        ):
            pm_core._post_land_git_pull("lapis-pm")

        contents = log_file.read_text()
        assert "synced" in contents
        assert "post-land-hook" in contents

    def test_lapis_pm_pull_no_log_when_already_current(self, tmp_path):
        """No deploy log entry when pull succeeds but HEAD was already current."""
        log_file = tmp_path / "lapis-pm-deploy-log.md"
        log_file.write_text("# log\n---\n")

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="same1234")
            return _make_completed_process(returncode=0, stdout="")

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
        ):
            pm_core._post_land_git_pull("lapis-pm")

        contents = log_file.read_text()
        assert "synced" not in contents  # no new entry


class TestDeployCurrencyCheck:
    """Tests for _check_deploy_currency — deploy-currency health monitoring."""

    def test_alerts_when_deploy_clone_is_behind(self):
        """When local HEAD differs from origin/main, Pushover fires and mem is stamped."""
        notify_calls = []
        mem_sets = {}

        def fake_run(cmd, **kwargs):
            if "fetch" in cmd:
                return _make_completed_process(returncode=0)
            if "rev-parse" in cmd:
                if "origin/main" in cmd:
                    return _make_completed_process(returncode=0, stdout="newsha0new")
                return _make_completed_process(returncode=0, stdout="oldsha0old")
            return _make_completed_process(returncode=0)

        fake_mem = MagicMock()
        fake_mem.get.return_value = None  # no prior alert
        fake_mem.set.side_effect = lambda k, v, **kw: mem_sets.update({k: v})

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "title": title})
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._check_deploy_currency()

        assert len(notify_calls) == 1, "Expected one Pushover alert on stale deploy clone"
        assert "stale" in notify_calls[0]["title"]
        assert pm_core._DEPLOY_CURRENCY_STALE_KEY in mem_sets

    def test_no_alert_when_deploy_clone_is_current(self):
        """When local HEAD matches origin/main, no alert is sent."""
        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "fetch" in cmd:
                return _make_completed_process(returncode=0)
            return _make_completed_process(returncode=0, stdout="same1234same")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._check_deploy_currency()

        assert len(notify_calls) == 0

    def test_cooldown_suppresses_repeat_alert(self):
        """Within _DEPLOY_CURRENCY_COOLDOWN_SECS, a second staleness check does not re-alert."""
        from datetime import datetime, timezone, timedelta
        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "fetch" in cmd:
                return _make_completed_process(returncode=0)
            if "origin/main" in cmd:
                return _make_completed_process(returncode=0, stdout="newsha0new")
            return _make_completed_process(returncode=0, stdout="oldsha0old")

        recent_ts = (
            datetime.now(timezone.utc) - timedelta(seconds=60)
        ).isoformat()  # 1 minute ago, well within 1h cooldown

        fake_mem = MagicMock()
        fake_mem.get.return_value = recent_ts

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._check_deploy_currency()

        assert len(notify_calls) == 0, "Alert was suppressed by cooldown — no Pushover expected"

    def test_exception_during_check_does_not_raise(self):
        """Any exception in _check_deploy_currency must not propagate (best-effort)."""
        def fake_run(cmd, **kwargs):
            raise OSError("git not found")

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._check_deploy_currency()  # must not raise


class TestCodeReviewerDeploy:
    """Tests for code-reviewer entry in _POST_LAND_PULL (lapis-pm-deploy-pull-code-reviewer-v0)."""

    def test_code_reviewer_not_in_post_land_restart(self):
        """code-reviewer must not be in _POST_LAND_RESTART: timers re-import on each fire."""
        assert "code-reviewer" not in pm_core._POST_LAND_RESTART

    def test_code_reviewer_not_in_post_land_pull_critical(self):
        """code-reviewer pull failure is LOW signal, not critical — must not be in CRITICAL set."""
        assert "code-reviewer" not in pm_core._POST_LAND_PULL_CRITICAL

    def test_code_reviewer_pull_triggers_git_pull_no_restart(self):
        """Landing a code-reviewer PR fires exactly one git pull and zero systemctl calls."""
        pull_calls = []
        restart_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "sudo":
                restart_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("code-reviewer")

        assert len(pull_calls) == 1
        assert pull_calls[0] == [
            "git", "-C", "/srv/git/code-reviewer-working", "pull", "--ff-only", "origin", "main"
        ]
        assert len(restart_calls) == 0, "code-reviewer is Type=oneshot — no systemctl restart"

    def test_code_reviewer_pull_failure_sends_low_priority_notify(self):
        """A failed code-reviewer pull emits exactly one LOW-priority notification."""
        from agents_core.notify import Priority

        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            return _make_completed_process(returncode=1, stderr="not fast-forward")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "title": title, "priority": priority})
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_git_pull("code-reviewer")

        assert len(notify_calls) == 1, "Expected exactly one notification on code-reviewer pull failure"
        assert notify_calls[0]["priority"] == Priority.LOW, (
            f"Expected Priority.LOW, got {notify_calls[0]['priority']}"
        )
        assert "code-reviewer" in notify_calls[0]["message"]
        assert "/srv/git/code-reviewer-working" in notify_calls[0]["message"]

    def test_code_reviewer_pull_failure_does_not_raise(self):
        """A failed code-reviewer pull must not raise — landing must still complete."""
        def fake_run(cmd, **kwargs):
            return _make_completed_process(returncode=1, stderr="diverged")

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("code-reviewer")  # must not raise

    def test_code_reviewer_pull_failure_not_critical_channel(self):
        """code-reviewer pull failure must NOT emit NORMAL or HIGH priority — low signal only."""
        from agents_core.notify import Priority

        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            return _make_completed_process(returncode=1, stderr="not ff")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(priority)
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_git_pull("code-reviewer")

        assert all(p == Priority.LOW for p in notify_calls), (
            "code-reviewer pull failure must only emit LOW priority — never NORMAL or HIGH"
        )
