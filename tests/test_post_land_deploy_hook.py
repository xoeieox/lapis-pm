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
        """Mapped repo causes git pull (both paths) then system + user restarts."""
        pull_calls = []
        sudo_restart_calls = []
        user_restart_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "sudo":
                sudo_restart_calls.append(cmd)
            elif cmd[0] == "systemctl" and "--user" in cmd and "restart" in cmd:
                user_restart_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                    pm_core._post_land_deploy_hook("agents-core")

        assert len(pull_calls) == 2
        pull_paths = [c[2] for c in pull_calls]
        assert "/srv/git/agents-core-working" in pull_paths
        assert "/data/agents" in pull_paths
        assert len(sudo_restart_calls) == 2
        assert sudo_restart_calls[0] == ["sudo", "-n", "systemctl", "restart", "claude-queue-runner.service"]
        assert sudo_restart_calls[1] == ["sudo", "-n", "systemctl", "restart", "gpu-queue-runner.service"]
        assert len(user_restart_calls) == 2
        assert user_restart_calls[0] == ["systemctl", "--user", "restart", "doorman-server.service"]
        assert user_restart_calls[1] == ["systemctl", "--user", "restart", "slot-server.service"]

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

    def test_user_unit_restart_uses_no_sudo(self, capsys):
        """User-unit restarts must use `systemctl --user restart`, never sudo."""
        sudo_calls = []
        user_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "sudo":
                sudo_calls.append(cmd)
            elif cmd[0] == "systemctl" and "--user" in cmd:
                user_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                    pm_core._post_land_deploy_hook("agents-core")

        restart_user = [c for c in user_calls if "restart" in c]
        assert len(restart_user) == 2
        for c in restart_user:
            assert c[0] == "systemctl"
            assert "--user" in c
            assert "sudo" not in c
        sudo_restart = [c for c in sudo_calls if "restart" in c]
        assert len(sudo_restart) == 2
        for c in sudo_restart:
            assert "--user" not in c

    def test_user_unit_restart_failure_logged_does_not_raise(self, capsys):
        """Non-zero rc from a user-unit restart is logged to stderr and does not raise."""
        def fake_run(cmd, **kwargs):
            if cmd[0] == "systemctl" and "--user" in cmd and "restart" in cmd:
                return _make_completed_process(returncode=1, stderr="unit failed")
            if cmd[0] == "systemctl" and "--user" in cmd and "is-active" in cmd:
                return _make_completed_process(returncode=3, stdout="failed")
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                    pm_core._post_land_deploy_hook("agents-core")  # must not raise

        captured = capsys.readouterr()
        assert "rc=1" in captured.err or "failed" in captured.err

    def test_repo_without_user_units_issues_no_user_restarts(self, capsys):
        """A repo with no _POST_LAND_RESTART_USER entry must not issue systemctl --user calls."""
        user_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "systemctl" and "--user" in cmd:
                user_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                    pm_core._post_land_deploy_hook("lapis-pm")

        assert len(user_calls) == 0

    def test_deploy_hook_disabled_skips_pull_and_restarts(self, capsys):
        """LAPIS_PM_DEPLOY_HOOK_DISABLE=1 skips all pulls and restarts."""
        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", True):
            with patch("lapis_pm.pm_core.subprocess.run") as mock_run:
                pm_core._post_land_deploy_hook("agents-core")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "disabled" in captured.err

    def test_xdg_runtime_dir_unset_skips_user_restarts_logs_error(self, capsys):
        """XDG_RUNTIME_DIR unset → no systemctl --user calls issued, loud error logged."""
        user_calls = []
        sudo_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "systemctl" and "--user" in cmd:
                user_calls.append(cmd)
            elif cmd[0] == "sudo":
                sudo_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="active")

        env_without_xdg = {k: v for k, v in __import__("os").environ.items() if k != "XDG_RUNTIME_DIR"}
        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.dict("os.environ", env_without_xdg, clear=True):
                    pm_core._post_land_deploy_hook("agents-core")

        assert len(user_calls) == 0, "No systemctl --user calls when XDG_RUNTIME_DIR unset"
        assert len(sudo_calls) == 2, "System-unit restarts still fire"
        captured = capsys.readouterr()
        assert "XDG_RUNTIME_DIR" in captured.err

    def test_user_unit_not_active_after_restart_logs_loudly(self, capsys):
        """If is-active returns non-active after restart, loud stderr log, no raise."""
        def fake_run(cmd, **kwargs):
            if cmd[0] == "systemctl" and "--user" in cmd and "is-active" in cmd:
                return _make_completed_process(returncode=3, stdout="failed\n")
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                    pm_core._post_land_deploy_hook("agents-core")  # must not raise

        captured = capsys.readouterr()
        assert "not active after restart" in captured.err


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
        """Mapped repo (agents-core) causes git pull for both clone paths."""
        pull_calls = []

        def fake_run(cmd, **kwargs):
            if "pull" in cmd:
                pull_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("agents-core")

        assert len(pull_calls) == 2
        pull_paths = [c[2] for c in pull_calls]
        assert "/srv/git/agents-core-working" in pull_paths
        assert "/data/agents" in pull_paths
        for c in pull_calls:
            assert "--ff-only" in c

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
        """A failing git pull must not prevent subsequent system or user-unit restarts."""
        sudo_calls = []
        user_restart_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "pull" in cmd:
                return _make_completed_process(returncode=1, stderr="not ff")
            if cmd[0] == "git" and "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            if cmd[0] == "sudo":
                sudo_calls.append(cmd)
            elif cmd[0] == "systemctl" and "--user" in cmd:
                user_restart_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                    pm_core._post_land_deploy_hook("agents-core")

        assert len(sudo_calls) == 2
        assert sudo_calls[0] == ["sudo", "-n", "systemctl", "restart", "claude-queue-runner.service"]
        assert sudo_calls[1] == ["sudo", "-n", "systemctl", "restart", "gpu-queue-runner.service"]
        # user restarts: 2 restart + 2 is-active liveness checks
        restart_cmds = [c for c in user_restart_calls if "restart" in c]
        assert len(restart_cmds) == 2
        assert restart_cmds[0] == ["systemctl", "--user", "restart", "doorman-server.service"]
        assert restart_cmds[1] == ["systemctl", "--user", "restart", "slot-server.service"]

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

    def test_agents_core_pull_failure_sends_pushover(self):
        """A failed pull for agents-core triggers a Pushover alert (agents-core is CRITICAL).

        /data/agents is the editable-install root for --user units; a pull failure there
        means doorman-server and slot-server silently run stale code — hence CRITICAL.
        """
        from agents_core.notify import Priority

        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            return _make_completed_process(returncode=1, stderr="not ff")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"title": title, "priority": priority})
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_git_pull("agents-core")

        assert len(notify_calls) >= 1, "agents-core pull failure must send Pushover (CRITICAL repo)"
        assert any("deploy pull failed" in c["title"] for c in notify_calls)
        assert all(c["priority"] == Priority.NORMAL for c in notify_calls)

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
