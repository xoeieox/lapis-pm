"""Unit tests for _post_land_deploy_hook in pm_core."""

import importlib.util
import json
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest
import yaml

from lapis_pm import pm_core, spec_review

# R4 (night-plan-conductor-deploy-sync-v0): committed fixture reproducing conductor's
# real scripts/-local import edges (verified against origin/main = bede656 this
# cycle) — see tests/fixtures/conductor-night-closure/ for provenance notes per file.
_CLOSURE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "conductor-night-closure"

# Matches both `from X import ...` and `import X` at any indent (module-level or
# deferred inside a function) — deliberately indent-agnostic since the whole point
# of R4 is to catch deferred (function-local) sibling imports, not just top-level ones.
_IMPORT_RE = re.compile(
    r'^\s*(?:from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\b|import\s+([A-Za-z_][A-Za-z0-9_]*)\b)',
    re.MULTILINE,
)


def _local_import_targets(source: str) -> set:
    """Every bare module name referenced by an import statement in `source`."""
    targets = set()
    for m in _IMPORT_RE.finditer(source):
        name = m.group(1) or m.group(2)
        if name:
            targets.add(name)
    return targets


def _compute_import_closure(seed_modules, source_for) -> set:
    """Fixpoint transitive closure over `seed_modules` (R4 spec algorithm).

    `source_for(mod)` returns mod's source text, or None if `mod` is not a
    scripts/-local sibling (stdlib/agents_core/pip packages are not edges).
    """
    closure = set()
    frontier = set(seed_modules)
    while frontier:
        mod = frontier.pop()
        if mod in closure:
            continue
        src = source_for(mod)
        if src is None:
            continue
        closure.add(mod)
        for candidate in _local_import_targets(src):
            if candidate not in closure:
                frontier.add(candidate)
    return closure


def _make_completed_process(returncode=0, stderr="", stdout=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _advancing_fake_run(extra=None):
    """Return a fake_run that makes each git pull advance HEAD (pre != post SHA)."""
    revparse_count = {}

    def fake_run(cmd, **kwargs):
        if cmd[0] == "git" and "rev-parse" in cmd:
            path = cmd[2]
            revparse_count[path] = revparse_count.get(path, 0) + 1
            sha = "presha111" if revparse_count[path] == 1 else "postsha222"
            return _make_completed_process(returncode=0, stdout=sha)
        if extra:
            return extra(cmd, **kwargs)
        return _make_completed_process(returncode=0, stdout="active")

    return fake_run


class TestPostLandDeployHook:
    """Tests for _post_land_deploy_hook."""

    def test_mapped_repo_fires_restart_per_unit(self, capsys, tmp_path):
        """Mapped repo causes git pull (both paths) then system + user restarts when HEAD advances."""
        pull_calls = []
        sudo_restart_calls = []
        user_restart_calls = []
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "sudo":
                sudo_restart_calls.append(cmd)
            elif cmd[0] == "systemctl" and "--user" in cmd and "restart" in cmd:
                user_restart_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                        with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                            with patch.object(pm_core, "_read_restart_pending", return_value=None):
                                pm_core._post_land_deploy_hook("agents-core")

        assert len(pull_calls) == 2
        pull_paths = [c[2] for c in pull_calls]
        assert "/srv/git/agents-core-working" in pull_paths
        assert "/data/agents" in pull_paths
        restart_units = {c[4] for c in sudo_restart_calls if "restart" in c}
        assert len(sudo_restart_calls) == 2
        assert "claude-queue-runner.service" in restart_units
        assert "gpu-queue-runner.service" in restart_units
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

    def test_restart_failure_does_not_raise(self, capsys, tmp_path):
        """Non-zero returncode from restart is logged to stderr but does not raise."""
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "sudo" and "restart" in cmd:
                return _make_completed_process(returncode=1, stderr="Failed to restart unit")
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                        with patch.object(pm_core, "_read_restart_pending", return_value=None):
                            pm_core._post_land_deploy_hook("agents-core")  # must not raise

        captured = capsys.readouterr()
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

    def test_user_unit_restart_uses_no_sudo(self, capsys, tmp_path):
        """User-unit restarts must use `systemctl --user restart`, never sudo."""
        sudo_calls = []
        user_calls = []
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            if cmd[0] == "sudo":
                sudo_calls.append(cmd)
            elif cmd[0] == "systemctl" and "--user" in cmd:
                user_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                        with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                            with patch.object(pm_core, "_read_restart_pending", return_value=None):
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

    def test_user_unit_restart_failure_logged_does_not_raise(self, capsys, tmp_path):
        """Non-zero rc from a user-unit restart is logged to stderr and does not raise."""
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "systemctl" and "--user" in cmd and "restart" in cmd:
                return _make_completed_process(returncode=1, stderr="unit failed")
            if cmd[0] == "systemctl" and "--user" in cmd and "is-active" in cmd:
                return _make_completed_process(returncode=3, stdout="failed")
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                        with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                            with patch.object(pm_core, "_read_restart_pending", return_value=None):
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

    def test_xdg_runtime_dir_unset_skips_user_restarts_logs_error(self, capsys, tmp_path):
        """XDG_RUNTIME_DIR unset → no systemctl --user calls issued, loud error logged."""
        user_calls = []
        sudo_calls = []
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            if cmd[0] == "systemctl" and "--user" in cmd:
                user_calls.append(cmd)
            elif cmd[0] == "sudo":
                sudo_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="active")

        env_without_xdg = {k: v for k, v in __import__("os").environ.items() if k != "XDG_RUNTIME_DIR"}
        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", env_without_xdg, clear=True):
                        with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                            with patch.object(pm_core, "_read_restart_pending", return_value=None):
                                pm_core._post_land_deploy_hook("agents-core")

        assert len(user_calls) == 0, "No systemctl --user calls when XDG_RUNTIME_DIR unset"
        assert len(sudo_calls) == 2, "System-unit restarts still fire"
        captured = capsys.readouterr()
        assert "XDG_RUNTIME_DIR" in captured.err

    def test_user_unit_not_active_after_restart_logs_loudly(self, capsys, tmp_path):
        """If is-active returns non-active after restart, loud stderr log, no raise."""
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            if cmd[0] == "systemctl" and "--user" in cmd and "is-active" in cmd:
                return _make_completed_process(returncode=3, stdout="failed\n")
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                        with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                            with patch.object(pm_core, "_read_restart_pending", return_value=None):
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
            if "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
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

    def test_pull_failure_skips_restart(self, capsys):
        """A failing git pull means HEAD did not advance, so no restart is issued.

        Under the HEAD-advance gate, a pull failure is treated as a no-op: the
        deploy hook does not know whether new code exists, so it conservatively
        skips the restart. The pull failure is still logged/alerted via the existing
        path; the noop is logged to stderr.
        """
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
                    with patch.object(pm_core, "_read_restart_pending", return_value=None):
                        pm_core._post_land_deploy_hook("agents-core")

        # Pull failed → HEAD unchanged → restart must be skipped
        assert len(sudo_calls) == 0, "No restart when pull failed (HEAD did not advance)"
        restart_cmds = [c for c in user_restart_calls if "restart" in c]
        assert len(restart_cmds) == 0, "No user-unit restart when pull failed"
        captured = capsys.readouterr()
        assert "noop" in captured.err or "HEAD unchanged" in captured.err or "rc=1" in captured.err

    def test_pull_ff_only_in_all_args(self):
        """--ff-only is present in the git pull call for lapis-pm."""
        pull_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            if "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
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


class TestFacetsDeploy:
    """Tests for facets entry in _POST_LAND_PULL (lapis-pm-deploy-pull-facets-v0)."""

    def test_facets_not_in_post_land_restart(self):
        """facets must not be in _POST_LAND_RESTART: adapter invoked per-fire, no daemon."""
        assert "facets" not in pm_core._POST_LAND_RESTART

    def test_facets_not_in_post_land_restart_user(self):
        """facets must not be in _POST_LAND_RESTART_USER: no long-running user service."""
        assert "facets" not in pm_core._POST_LAND_RESTART_USER

    def test_facets_not_in_post_land_pull_critical(self):
        """facets pull failure is LOW signal, not critical — must not be in CRITICAL set."""
        assert "facets" not in pm_core._POST_LAND_PULL_CRITICAL

    def test_facets_in_post_land_pull_low_signal(self):
        """facets pull failure emits LOW-priority notification (advisory, per-invocation)."""
        assert "facets" in pm_core._POST_LAND_PULL_LOW_SIGNAL

    def test_facets_pull_triggers_git_pull_no_restart(self):
        """Landing a facets PR fires exactly one git pull and zero systemctl calls."""
        pull_calls = []
        restart_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "sudo":
                restart_calls.append(cmd)
            elif cmd[0] == "systemctl" and "--user" in cmd:
                restart_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("facets")

        assert len(pull_calls) == 1
        assert pull_calls[0] == [
            "git", "-C", "/srv/git/facets-working", "pull", "--ff-only", "origin", "main"
        ]
        assert len(restart_calls) == 0, "facets is Type=oneshot per-fire adapter — no systemctl restart"

    def test_facets_pull_failure_sends_low_priority_notify(self):
        """A failed facets pull emits exactly one LOW-priority notification."""
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
            pm_core._post_land_git_pull("facets")

        assert len(notify_calls) == 1, "Expected exactly one notification on facets pull failure"
        assert notify_calls[0]["priority"] == Priority.LOW, (
            f"Expected Priority.LOW, got {notify_calls[0]['priority']}"
        )
        assert "facets" in notify_calls[0]["message"]
        assert "/srv/git/facets-working" in notify_calls[0]["message"]

    def test_facets_pull_failure_does_not_raise(self):
        """A failed facets pull must not raise — landing must still complete."""
        def fake_run(cmd, **kwargs):
            return _make_completed_process(returncode=1, stderr="diverged")

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("facets")  # must not raise

    def test_facets_pull_path_synced_with_spec_review(self):
        """Guard: facets pull path MUST match spec_review._FACETS_REPO_PATH."""
        assert pm_core._POST_LAND_PULL["facets"] == [str(spec_review._FACETS_REPO_PATH)], (
            "facets pull path diverged from spec_review._FACETS_REPO_PATH — "
            "post-land pull and PYTHONPATH injection must target the same clone"
        )


class TestSynapseDeploy:
    """Tests for synapse entry in _POST_LAND_PULL and _POST_LAND_RESTART.

    synapse is the central BRIX context-injection substrate. It is unique among
    the mapped repos in that it has a SINGLE tree (/srv/git/synapse-working)
    serving both PM investigation and runtime, and the service is Type=simple
    (long-running daemon) requiring a restart after pull.
    """

    def test_synapse_in_post_land_pull(self):
        """synapse must be mapped to /srv/git/synapse-working in _POST_LAND_PULL."""
        assert "synapse" in pm_core._POST_LAND_PULL
        assert pm_core._POST_LAND_PULL["synapse"] == ["/srv/git/synapse-working"]

    def test_synapse_in_post_land_restart(self):
        """synapse must be mapped to synapse.service in _POST_LAND_RESTART."""
        assert "synapse" in pm_core._POST_LAND_RESTART
        assert pm_core._POST_LAND_RESTART["synapse"] == ("synapse.service",)

    def test_synapse_in_post_land_pull_critical(self):
        """synapse must be in _POST_LAND_PULL_CRITICAL (central substrate, silent failure = critical)."""
        assert "synapse" in pm_core._POST_LAND_PULL_CRITICAL

    def test_synapse_not_in_post_land_restart_user(self):
        """synapse.service is a system unit, not a --user unit — must not be in _RESTART_USER."""
        assert "synapse" not in pm_core._POST_LAND_RESTART_USER

    def test_synapse_pull_triggers_git_pull_and_system_restart(self, tmp_path):
        """Landing a synapse PR fires exactly one git pull and one sudo systemctl restart."""
        pull_calls = []
        restart_calls = []
        user_restart_calls = []
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "sudo" and "restart" in cmd:
                restart_calls.append(cmd)
            elif cmd[0] == "systemctl" and "--user" in cmd:
                user_restart_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                        pm_core._post_land_deploy_hook("synapse")

        assert len(pull_calls) == 1
        assert pull_calls[0] == [
            "git", "-C", "/srv/git/synapse-working", "pull", "--ff-only", "origin", "main"
        ]
        assert len(restart_calls) == 1
        assert restart_calls[0] == ["sudo", "-n", "systemctl", "restart", "synapse.service"]
        assert len(user_restart_calls) == 0, "synapse.service is a system unit, not --user"

    def test_synapse_pull_failure_sends_pushover_normal_priority(self):
        """A failed synapse pull triggers a Pushover alert with Priority.NORMAL (CRITICAL repo)."""
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
            pm_core._post_land_git_pull("synapse")

        assert len(notify_calls) >= 1, "synapse pull failure must send Pushover (CRITICAL repo)"
        assert any("deploy pull failed" in c["title"] for c in notify_calls)
        assert all(c["priority"] == Priority.NORMAL for c in notify_calls)

    def test_synapse_pull_failure_title_contains_synapse(self):
        """Synapse pull failure title is parameterized and contains 'synapse'."""
        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            return _make_completed_process(returncode=1, stderr="diverged")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"title": title})
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_git_pull("synapse")

        assert len(notify_calls) >= 1
        assert any("synapse" in c["title"] for c in notify_calls), (
            "synapse pull failure title must contain 'synapse' (parameterized per repo)"
        )

    def test_synapse_restart_failure_does_not_raise(self, capsys, tmp_path):
        """Non-zero returncode from synapse.service restart is logged to stderr, does not raise."""
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "pull" in cmd:
                return _make_completed_process(returncode=0)
            if cmd[0] == "sudo" and "restart" in cmd:
                return _make_completed_process(returncode=1, stderr="unit failed to restart")
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    pm_core._post_land_deploy_hook("synapse")  # must not raise

        captured = capsys.readouterr()
        assert "rc=1" in captured.err or "failed" in captured.err

    def test_synapse_restart_failure_is_stderr_only_not_pushover(self):
        """synapse restart failure is logged to stderr/journal only (no Pushover).

        This documents the §2 deferral: loud restart-failure signaling is deferred to
        lapis-pm-deploy-restart-gate-on-advance-v0. Restart failure is stderr/journal
        best-effort with Restart=on-failure as the self-heal backstop, consistent with
        all other repos (agents-core included).
        """
        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "pull" in cmd:
                return _make_completed_process(returncode=0)
            if cmd[0] == "sudo" and "restart" in cmd:
                return _make_completed_process(returncode=1, stderr="unit failed")
            return _make_completed_process(returncode=0)

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"title": title})
            return True

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch("agents_core.notify.send_notification", fake_notify):
                    pm_core._post_land_deploy_hook("synapse")

        # No Pushover for restart failure — deferred to lapis-pm-deploy-restart-gate-on-advance-v0
        assert not any("restart" in c.get("title", "") for c in notify_calls)

    def test_lapis_pm_pull_failure_title_still_contains_lapis_pm(self):
        """Regression guard: lapis-pm pull failure title is 'lapis-pm: deploy pull failed' after parameterization.

        With the title parameterized as f'{repo}: deploy pull failed', lapis-pm still
        gets 'lapis-pm: deploy pull failed'. This ensures the parameterization doesn't
        break the existing lapis-pm test.
        """
        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            return _make_completed_process(returncode=1, stderr="not ff")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"title": title})
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_git_pull("lapis-pm")

        assert len(notify_calls) >= 1
        assert any("lapis-pm" in c["title"] for c in notify_calls), (
            "lapis-pm pull failure title must contain 'lapis-pm' (regression guard after parameterization)"
        )


class TestGardenerDeploy:
    """Tests for gardener entry in _POST_LAND_PULL (lapis-pm-deploy-pull-gardener-v0)."""

    def test_gardener_not_in_post_land_restart(self):
        """gardener must not be in _POST_LAND_RESTART: timer re-imports on each fire."""
        assert "gardener" not in pm_core._POST_LAND_RESTART

    def test_gardener_not_in_post_land_restart_user(self):
        """gardener must not be in _POST_LAND_RESTART_USER: no long-running user service."""
        assert "gardener" not in pm_core._POST_LAND_RESTART_USER

    def test_gardener_not_in_post_land_pull_critical(self):
        """gardener pull failure is LOW signal, not critical — must not be in CRITICAL set."""
        assert "gardener" not in pm_core._POST_LAND_PULL_CRITICAL

    def test_gardener_in_post_land_pull_low_signal(self):
        """gardener pull failure emits LOW-priority notification (timer oneshot nightly)."""
        assert "gardener" in pm_core._POST_LAND_PULL_LOW_SIGNAL

    def test_gardener_pull_triggers_git_pull_no_restart(self):
        """Landing a gardener PR fires exactly one git pull and zero systemctl calls."""
        pull_calls = []
        restart_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "sudo":
                restart_calls.append(cmd)
            elif cmd[0] == "systemctl" and "--user" in cmd:
                restart_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("gardener")

        assert len(pull_calls) == 1
        assert pull_calls[0] == [
            "git", "-C", "/srv/git/gardener-working", "pull", "--ff-only", "origin", "main"
        ]
        assert len(restart_calls) == 0, "gardener is Type=oneshot — no systemctl restart"

    def test_gardener_pull_failure_sends_low_priority_notify(self):
        """A failed gardener pull emits exactly one LOW-priority notification."""
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
            pm_core._post_land_git_pull("gardener")

        assert len(notify_calls) == 1, "Expected exactly one notification on gardener pull failure"
        assert notify_calls[0]["priority"] == Priority.LOW, (
            f"Expected Priority.LOW, got {notify_calls[0]['priority']}"
        )
        assert "gardener" in notify_calls[0]["message"]
        assert "/srv/git/gardener-working" in notify_calls[0]["message"]

    def test_gardener_pull_failure_does_not_raise(self):
        """A failed gardener pull must not raise — landing must still complete."""
        def fake_run(cmd, **kwargs):
            return _make_completed_process(returncode=1, stderr="diverged")

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("gardener")  # must not raise

    def test_gardener_pull_failure_not_critical_channel(self):
        """gardener pull failure must NOT emit NORMAL or HIGH priority — low signal only."""
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
            pm_core._post_land_git_pull("gardener")

        assert all(p == Priority.LOW for p in notify_calls), (
            "gardener pull failure must only emit LOW priority — never NORMAL or HIGH"
        )


class TestCockpitDeploy:
    """Tests for cockpit entry in _POST_LAND_PULL / _POST_LAND_RESTART_USER
    (lapis-pm-deploy-pull-restart-cockpit-v0)."""

    def test_cockpit_pull_path(self):
        """cockpit pulls the single-tree PYTHONPATH-import serving clone."""
        assert "cockpit" in pm_core._POST_LAND_PULL
        assert pm_core._POST_LAND_PULL["cockpit"] == ["/srv/git/cockpit-working"]

    def test_cockpit_restart_user_unit(self):
        """cockpit.service is restarted via the --user (no-sudo) path."""
        assert pm_core._POST_LAND_RESTART_USER["cockpit"] == ("cockpit.service",)

    def test_cockpit_not_in_post_land_restart(self):
        """cockpit is a --user unit, never the sudo/system path."""
        assert "cockpit" not in pm_core._POST_LAND_RESTART

    def test_cockpit_not_in_post_land_pull_critical(self):
        """cockpit pull failure is LOW signal, not critical — must not be in CRITICAL set."""
        assert "cockpit" not in pm_core._POST_LAND_PULL_CRITICAL

    def test_cockpit_in_post_land_pull_low_signal(self):
        """cockpit pull failure emits LOW-priority notification (advisory, self-announcing)."""
        assert "cockpit" in pm_core._POST_LAND_PULL_LOW_SIGNAL

    def test_cockpit_pull_and_restart_dispatch(self, capsys, tmp_path):
        """Landing a cockpit PR with HEAD advancing issues exactly one git pull and
        one `systemctl --user restart cockpit.service` call — no sudo call anywhere."""
        pull_calls = []
        sudo_calls = []
        user_restart_calls = []
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "sudo":
                sudo_calls.append(cmd)
            elif cmd[0] == "systemctl" and "--user" in cmd and "restart" in cmd:
                user_restart_calls.append(cmd)
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                        pm_core._post_land_deploy_hook("cockpit")

        assert len(pull_calls) == 1
        assert pull_calls[0] == [
            "git", "-C", "/srv/git/cockpit-working", "pull", "--ff-only", "origin", "main"
        ]
        assert len(user_restart_calls) == 1
        assert user_restart_calls[0] == ["systemctl", "--user", "restart", "cockpit.service"]
        assert len(sudo_calls) == 0, "cockpit must never be restarted via sudo"

    def test_cockpit_noop_on_unchanged_head(self, capsys):
        """No-op pull (HEAD unchanged) must not issue a cockpit.service restart."""
        user_restart_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="sameshasha")
            if cmd[0] == "systemctl" and "--user" in cmd and "restart" in cmd:
                user_restart_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                    pm_core._post_land_deploy_hook("cockpit")

        assert len(user_restart_calls) == 0, "No restart when HEAD unchanged"

    def test_cockpit_pull_failure_sends_low_priority_notify(self):
        """A failed cockpit pull emits exactly one LOW-priority notification."""
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
            pm_core._post_land_git_pull("cockpit")

        assert len(notify_calls) == 1, "Expected exactly one notification on cockpit pull failure"
        assert notify_calls[0]["priority"] == Priority.LOW, (
            f"Expected Priority.LOW, got {notify_calls[0]['priority']}"
        )
        assert "cockpit" in notify_calls[0]["message"]
        assert "/srv/git/cockpit-working" in notify_calls[0]["message"]

    def test_cockpit_pull_failure_not_critical_channel(self):
        """cockpit pull failure must NOT emit NORMAL or HIGH priority — LOW only."""
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
            pm_core._post_land_git_pull("cockpit")

        assert all(p == Priority.LOW for p in notify_calls), (
            "cockpit pull failure must only emit LOW priority — never NORMAL or HIGH"
        )

    def test_cockpit_pull_failure_does_not_raise(self):
        """A failed cockpit pull must not raise — landing must still complete."""
        def fake_run(cmd, **kwargs):
            return _make_completed_process(returncode=1, stderr="diverged")

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("cockpit")  # must not raise

    def test_cockpit_restart_failure_does_not_raise(self, capsys, tmp_path):
        """Non-zero rc from cockpit.service restart is logged to stderr, does not raise."""
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "systemctl" and "--user" in cmd and "restart" in cmd:
                return _make_completed_process(returncode=1, stderr="unit failed")
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                        pm_core._post_land_deploy_hook("cockpit")  # must not raise

        captured = capsys.readouterr()
        assert "rc=1" in captured.err or "failed" in captured.err


class TestRagOpsDeploy:
    """Tests for rag-ops entry in _POST_LAND_PULL (lapis-pm-deploy-pull-rag-ops-v0).

    rag-ops is pure `docker compose` config + a bash script — no daemon imports it,
    so this mirrors TestGardenerDeploy's shape exactly: LOW signal, pull only, no
    restart. See pm_core.py's _POST_LAND_PULL["rag-ops"] comment for why a pull
    alone does not make docker re-read the compose file (out of scope here, §0/§2
    of the spec).
    """

    def test_rag_ops_not_in_post_land_restart(self):
        """rag-ops must not be in _POST_LAND_RESTART: no daemon to restart."""
        assert "rag-ops" not in pm_core._POST_LAND_RESTART

    def test_rag_ops_not_in_post_land_restart_user(self):
        """rag-ops must not be in _POST_LAND_RESTART_USER: no long-running user service."""
        assert "rag-ops" not in pm_core._POST_LAND_RESTART_USER

    def test_rag_ops_not_in_post_land_pull_critical(self):
        """rag-ops pull failure is LOW signal, not critical — must not be in CRITICAL set."""
        assert "rag-ops" not in pm_core._POST_LAND_PULL_CRITICAL

    def test_rag_ops_in_post_land_pull_low_signal(self):
        """rag-ops pull failure emits LOW-priority notification (attributable, never paging)."""
        assert "rag-ops" in pm_core._POST_LAND_PULL_LOW_SIGNAL

    def test_rag_ops_pull_triggers_git_pull_no_restart(self):
        """Landing a rag-ops PR fires exactly one git pull and zero systemctl/docker calls."""
        pull_calls = []
        restart_calls = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "sudo":
                restart_calls.append(cmd)
            elif cmd[0] == "systemctl":
                restart_calls.append(cmd)
            elif cmd[0] == "docker":
                restart_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                pm_core._post_land_deploy_hook("rag-ops")

        assert len(pull_calls) == 1
        assert pull_calls[0] == [
            "git", "-C", "/data/rag", "pull", "--ff-only", "origin", "main"
        ]
        assert len(restart_calls) == 0, (
            "rag-ops is pure docker compose config — no systemctl/docker invocation "
            "of any kind, per spec §1/§2"
        )

    def test_rag_ops_pull_failure_sends_low_priority_notify(self):
        """A failed rag-ops pull emits exactly one LOW-priority notification."""
        from agents_core.notify import Priority

        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            if "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            return _make_completed_process(returncode=1, stderr="not fast-forward")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "title": title, "priority": priority})
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_git_pull("rag-ops")

        assert len(notify_calls) == 1, "Expected exactly one notification on rag-ops pull failure"
        assert notify_calls[0]["priority"] == Priority.LOW, (
            f"Expected Priority.LOW, got {notify_calls[0]['priority']}"
        )
        assert "rag-ops" in notify_calls[0]["message"]
        assert "/data/rag" in notify_calls[0]["message"]

    def test_rag_ops_pull_failure_does_not_raise(self):
        """A failed rag-ops pull must not raise — landing must still complete."""
        def fake_run(cmd, **kwargs):
            if "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            return _make_completed_process(returncode=1, stderr="diverged")

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._post_land_git_pull("rag-ops")  # must not raise

    def test_rag_ops_pull_failure_not_critical_channel(self):
        """rag-ops pull failure must NOT emit NORMAL or HIGH priority — low signal only."""
        from agents_core.notify import Priority

        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            if "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            return _make_completed_process(returncode=1, stderr="not ff")

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(priority)
            return True

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_git_pull("rag-ops")

        assert all(p == Priority.LOW for p in notify_calls), (
            "rag-ops pull failure must only emit LOW priority — never NORMAL or HIGH"
        )


class TestPostLandGitPullDirtyTreeAndDivergence:
    """Poison-pill hardening for the shared _post_land_git_pull (Facets + Council
    spec-review, lapis-pm-deploy-pull-rag-ops-v0 §1.3). Both tests exercise the
    real git binary against a throwaway repo pair — not mocked subprocess — since
    the load-bearing behavior here is git's own actual dirty/divergence detection.
    """

    @staticmethod
    def _make_origin_and_clone(tmp_path):
        """A tiny origin repo (branch `main`, one commit) and a working clone of it."""
        origin = tmp_path / "origin"
        origin.mkdir()
        subprocess.run(["git", "init", "-b", "main", str(origin)], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=origin, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=origin, check=True, capture_output=True)
        (origin / "f.txt").write_text("v1\n")
        subprocess.run(["git", "add", "."], cwd=origin, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=origin, check=True, capture_output=True)

        clone = tmp_path / "clone"
        subprocess.run(
            ["git", "clone", str(origin), str(clone)], check=True, capture_output=True
        )
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=clone, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=clone, check=True, capture_output=True)
        return origin, clone

    def test_post_land_git_pull_dirty_tree_logs_specific_warning(self, tmp_path, capsys):
        """A dirty mapped clone (uncommitted local modification) must log the distinct
        dirty-tree message and must NOT attempt the `git pull` subprocess at all."""
        _origin, clone = self._make_origin_and_clone(tmp_path)
        (clone / "f.txt").write_text("locally modified, uncommitted\n")

        calls = []
        real_run = subprocess.run

        def spy_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return real_run(cmd, *args, **kwargs)

        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"gardener": [str(clone)]}),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=spy_run),
        ):
            advanced = pm_core._post_land_git_pull("gardener")

        assert advanced is False
        captured = capsys.readouterr()
        assert "dirty working tree" in captured.err
        assert not any(
            cmd[:2] == ["git", "-C"] and "pull" in cmd for cmd in calls
        ), "a dirty tree must skip the git pull subprocess entirely"

    def test_post_land_git_pull_non_fastforward_fails_clean(self, tmp_path, capsys):
        """A mapped clone whose local main has diverged from origin/main (non-fast-forward,
        distinct from the dirty-uncommitted-changes case above) must fail the pull cleanly:
        return False, not raise, and leave no merge-in-progress state behind."""
        origin, clone = self._make_origin_and_clone(tmp_path)

        # Advance origin/main independently...
        (origin / "f.txt").write_text("v2-on-origin\n")
        subprocess.run(["git", "add", "."], cwd=origin, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "origin-advances"], cwd=origin, check=True, capture_output=True)

        # ...while the clone's local main also gains a commit origin never saw.
        (clone / "g.txt").write_text("local-only commit\n")
        subprocess.run(["git", "add", "."], cwd=clone, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "clone-diverges"], cwd=clone, check=True, capture_output=True)

        with patch.object(pm_core, "_POST_LAND_PULL", {"gardener": [str(clone)]}):
            advanced = pm_core._post_land_git_pull("gardener")  # must not raise

        assert advanced is False
        status = subprocess.run(
            ["git", "-C", str(clone), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        assert status.returncode == 0
        # No merge-in-progress markers left behind by the failed --ff-only attempt.
        assert not (clone / ".git" / "MERGE_HEAD").exists()
        assert "both modified" not in status.stdout
        assert "Unmerged paths" not in status.stdout


class TestMergeAndDeploy:
    """Tests for merge_and_deploy choicepoint."""

    def test_merge_and_deploy_fires_hook_for_mapped_repo(self):
        """merge_and_deploy merges the PR and fires the deploy hook for a mapped repo."""
        merge_calls = []
        pull_calls = []

        def fake_merge(*args, **kwargs):
            merge_calls.append((args, kwargs))
            return {"merged": True}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "pull" in cmd:
                pull_calls.append(cmd)
            elif cmd[0] == "git" and "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="abc12345")
            return _make_completed_process(returncode=0)

        with (
            patch("lapis_pm.pm_core.merge_pr", side_effect=fake_merge),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
            patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}),
        ):
            result = pm_core.merge_and_deploy("agents-core", 42)

        assert result == {"merged": True}
        assert len(merge_calls) == 1
        assert merge_calls[0] == (("agents-core", 42), {"owner": None})
        assert len(pull_calls) == 2

    def test_merge_and_deploy_hook_exception_does_not_propagate(self):
        """Post-merge hook/branch-delete exception does not mask merge success."""
        merge_calls = []

        def fake_merge(*args, **kwargs):
            merge_calls.append((args, kwargs))
            return {"merged": True}

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)

        with (
            patch("lapis_pm.pm_core.merge_pr", side_effect=fake_merge),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
        ):
            result = pm_core.merge_and_deploy("agents-core", 42)

        assert result == {"merged": True}
        assert len(merge_calls) == 1

    def test_merge_and_deploy_uses_post_merge_hook_trigger(self, tmp_path):
        """merge_and_deploy passes trigger='post-merge-hook' to the hook."""
        log_file = tmp_path / "deploy-log.md"
        log_file.write_text("# log\n---\n")

        call_count = {"n": 0}

        def fake_merge(*args, **kwargs):
            return {"merged": True}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            if "rev-parse" in cmd:
                return _make_completed_process(
                    returncode=0,
                    stdout="oldsha0" if call_count["n"] <= 1 else "newsha1",
                )
            return _make_completed_process(returncode=0)

        with (
            patch("lapis_pm.pm_core.merge_pr", side_effect=fake_merge),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
        ):
            pm_core.merge_and_deploy("lapis-pm", 42)

        contents = log_file.read_text()
        assert "post-merge-hook" in contents

    def test_merge_failure_propagates(self):
        """A merge_pr failure raises and never calls the post-merge steps."""
        def fake_merge(*args, **kwargs):
            raise RuntimeError("Merge API error")

        with patch("lapis_pm.pm_core.merge_pr", side_effect=fake_merge):
            with patch("lapis_pm.pm_core._post_land_deploy_hook") as mock_hook:
                try:
                    pm_core.merge_and_deploy("agents-core", 42)
                    assert False, "Should have raised"
                except RuntimeError as e:
                    assert "Merge API error" in str(e)
                    mock_hook.assert_not_called()


class TestEnsureHeadBranchDeleted:
    """Tests for _ensure_head_branch_deleted branch hygiene."""

    def test_branch_already_deleted_returns_early(self):
        """If get_branch returns 404 (branch already deleted), no DELETE is issued."""
        delete_calls = []

        def fake_get_pr(*args, **kwargs):
            return {"head": {"ref": "lapis/tid/slug"}}

        def fake_get_branch(*args, **kwargs):
            exc = Exception("404 Not Found")
            exc.args = ("404 Not Found",)
            raise exc

        with (
            patch("lapis_pm.pm_core._forgejo_get_pr", side_effect=fake_get_pr),
            patch("lapis_pm.pm_core._forgejo_get_branch", side_effect=fake_get_branch),
        ):
            pm_core._ensure_head_branch_deleted("agents-core", 42)
            # No exception, no DELETE attempt

    def test_branch_present_issues_explicit_delete(self):
        """If branch still exists, an explicit DELETE is issued via httpx."""
        delete_calls = []

        def fake_get_pr(*args, **kwargs):
            return {"head": {"ref": "lapis/tid/slug"}}

        def fake_get_branch(*args, **kwargs):
            return {}  # branch exists

        def fake_delete(url, **kwargs):
            delete_calls.append(url)
            resp = MagicMock()
            resp.status_code = 204
            return resp

        with (
            patch("lapis_pm.pm_core._forgejo_get_pr", side_effect=fake_get_pr),
            patch("lapis_pm.pm_core._forgejo_get_branch", side_effect=fake_get_branch),
            patch.dict("os.environ", {"FORGEJO_TOKEN": "fake-token"}),
        ):
            import httpx
            with patch.object(httpx, "delete", side_effect=fake_delete):
                pm_core._ensure_head_branch_deleted("agents-core", 42)

        assert len(delete_calls) == 1
        assert "lapis/tid/slug" in delete_calls[0]

    def test_delete_exception_swallowed(self):
        """If DELETE raises, the exception is logged and does not propagate."""
        def fake_get_pr(*args, **kwargs):
            return {"head": {"ref": "lapis/tid/slug"}}

        def fake_get_branch(*args, **kwargs):
            return {}  # branch exists

        def fake_delete(url, **kwargs):
            raise OSError("Network error")

        with (
            patch("lapis_pm.pm_core._forgejo_get_pr", side_effect=fake_get_pr),
            patch("lapis_pm.pm_core._forgejo_get_branch", side_effect=fake_get_branch),
            patch.dict("os.environ", {"FORGEJO_TOKEN": "fake-token"}),
        ):
            import httpx
            with patch.object(httpx, "delete", side_effect=fake_delete):
                pm_core._ensure_head_branch_deleted("agents-core", 42)
                # Must not raise

    def test_no_forgejo_apis_is_noop(self):
        """If forgejo APIs are unavailable, the function is a noop."""
        with (
            patch("lapis_pm.pm_core._forgejo_get_pr", None),
            patch("lapis_pm.pm_core._forgejo_get_branch", None),
        ):
            pm_core._ensure_head_branch_deleted("agents-core", 42)
            # Must not raise


class TestPostLandDeployHookTrigger:
    """Tests for trigger parameter plumbing."""

    def test_post_land_deploy_hook_defaults_trigger_to_post_land_hook(self, tmp_path):
        """_post_land_deploy_hook defaults trigger to 'post-land-hook'."""
        log_file = tmp_path / "deploy-log.md"
        log_file.write_text("# log\n---\n")

        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            if "rev-parse" in cmd:
                return _make_completed_process(
                    returncode=0,
                    stdout="oldsha0" if call_count["n"] <= 1 else "newsha1",
                )
            return _make_completed_process(returncode=0)

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
        ):
            pm_core._post_land_deploy_hook("lapis-pm")

        contents = log_file.read_text()
        assert "post-land-hook" in contents

    def test_post_land_deploy_hook_custom_trigger(self, tmp_path):
        """_post_land_deploy_hook forwards custom trigger to _post_land_git_pull."""
        log_file = tmp_path / "deploy-log.md"
        log_file.write_text("# log\n---\n")

        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            if "rev-parse" in cmd:
                return _make_completed_process(
                    returncode=0,
                    stdout="oldsha0" if call_count["n"] <= 1 else "newsha1",
                )
            return _make_completed_process(returncode=0)

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
        ):
            pm_core._post_land_deploy_hook("lapis-pm", trigger="post-merge-hook")

        contents = log_file.read_text()
        assert "post-merge-hook" in contents

    def test_double_fire_already_current_no_log(self, tmp_path):
        """When git pull is a no-op (already current), second deploy-log line is not added."""
        log_file = tmp_path / "deploy-log.md"
        log_file.write_text("# log\n---\n")

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="same1234")
            return _make_completed_process(returncode=0)

        with (
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
        ):
            pm_core._post_land_deploy_hook("lapis-pm", trigger="post-merge-hook")

        contents = log_file.read_text()
        assert "synced" not in contents  # no new entry when already current


class TestHeadAdvanceGate:
    """Tests for the HEAD-advance gate and in-flight-aware deferral."""

    def _make_advancing_run(self, pull_calls=None, sudo_calls=None):
        """Return a fake_run that advances HEAD on pull."""
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "pre111sha" if revparse_count[path] == 1 else "post222sha"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            if cmd[0] == "git" and "pull" in cmd:
                if pull_calls is not None:
                    pull_calls.append(cmd)
                return _make_completed_process(returncode=0)
            if cmd[0] == "sudo" and "restart" in cmd:
                if sudo_calls is not None:
                    sudo_calls.append(cmd)
                return _make_completed_process(returncode=0)
            return _make_completed_process(returncode=0, stdout="active")

        return fake_run

    def test_noop_pull_skips_restart_and_logs(self, capsys):
        """No-op pull (HEAD unchanged) must not invoke systemctl restart and logs noop."""
        sudo_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="sameshasha")
            if cmd[0] == "sudo":
                sudo_calls.append(cmd)
            return _make_completed_process(returncode=0)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.object(pm_core, "_read_restart_pending", return_value=None):
                    pm_core._post_land_deploy_hook("agents-core")

        assert len(sudo_calls) == 0, "No restart when HEAD unchanged"
        captured = capsys.readouterr()
        assert "noop" in captured.err
        assert "HEAD unchanged" in captured.err

    def test_head_advance_idle_queue_restarts_now(self, capsys, tmp_path):
        """HEAD advance + idle queue → both units restart immediately."""
        sudo_calls = []
        fake_run = self._make_advancing_run(sudo_calls=sudo_calls)

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                        with patch.object(pm_core, "_read_restart_pending", return_value=None):
                            with patch.object(pm_core, "_write_restart_pending") as mock_write:
                                pm_core._post_land_deploy_hook("agents-core")

        restart_units = [c[4] for c in sudo_calls if "restart" in c]
        assert "claude-queue-runner.service" in restart_units
        assert "gpu-queue-runner.service" in restart_units
        mock_write.assert_not_called()

    def test_head_advance_busy_queue_defers_claude_runner(self, capsys, tmp_path):
        """HEAD advance + busy queue → claude-queue-runner deferred, marker written."""
        sudo_calls = []
        written = {}
        fake_run = self._make_advancing_run(sudo_calls=sudo_calls)

        def fake_write(repo, units, first_deferred_at):
            written["repo"] = repo
            written["units"] = units

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.object(pm_core, "_count_inflight_fixers", return_value=3):
                        with patch.object(pm_core, "_read_restart_pending", return_value=None):
                            with patch.object(pm_core, "_write_restart_pending", side_effect=fake_write):
                                with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                                    pm_core._post_land_deploy_hook("agents-core")

        # gpu-queue-runner should restart immediately; claude-queue-runner is deferred
        restart_units = [c[4] for c in sudo_calls if "restart" in c]
        assert "gpu-queue-runner.service" in restart_units
        assert "claude-queue-runner.service" not in restart_units
        assert written.get("repo") == "agents-core"
        assert "claude-queue-runner.service" in written.get("units", [])
        captured = capsys.readouterr()
        assert "deferring" in captured.err
        assert "in-flight" in captured.err

    def test_deferred_then_idle_fires_restart(self, capsys):
        """Pending marker + idle queue → deferred restart fires, marker cleared."""
        from datetime import datetime, timezone, timedelta
        sudo_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="sameshasha")
            if cmd[0] == "sudo" and "restart" in cmd:
                sudo_calls.append(cmd)
                return _make_completed_process(returncode=0)
            return _make_completed_process(returncode=0)

        first_deferred_at = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        pending = {
            "repo": "agents-core",
            "units": ["claude-queue-runner.service"],
            "first_deferred_at": first_deferred_at,
            "post_head": "",
        }

        cleared = {}

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                    with patch.object(pm_core, "_read_restart_pending", return_value=pending):
                        with patch.object(pm_core, "_clear_restart_pending",
                                          side_effect=lambda r: cleared.update({"repo": r})):
                            pm_core._post_land_deploy_hook("agents-core")

        restart_units = [c[4] for c in sudo_calls if "restart" in c]
        assert "claude-queue-runner.service" in restart_units
        assert cleared.get("repo") == "agents-core"
        captured = capsys.readouterr()
        assert "deferred restart now firing" in captured.err

    def test_defer_budget_exceeded_restarts_anyway(self, capsys):
        """Pending marker + busy queue + budget exceeded → forced restart + CRITICAL alert."""
        from datetime import datetime, timezone, timedelta
        sudo_calls = []
        notify_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="sameshasha")
            if cmd[0] == "sudo" and "restart" in cmd:
                sudo_calls.append(cmd)
                return _make_completed_process(returncode=0)
            return _make_completed_process(returncode=0)

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"title": title, "priority": priority, "message": message})
            return True

        # First deferred 31 minutes ago (past the 1800s default)
        first_deferred_at = (
            datetime.now(timezone.utc) - timedelta(seconds=pm_core.RESTART_DEFER_MAX_S + 60)
        ).isoformat()
        pending = {
            "repo": "agents-core",
            "units": ["claude-queue-runner.service"],
            "first_deferred_at": first_deferred_at,
            "post_head": "",
        }

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.object(pm_core, "_count_inflight_fixers", return_value=2):
                    with patch.object(pm_core, "_read_restart_pending", return_value=pending):
                        with patch.object(pm_core, "_clear_restart_pending"):
                            with patch("agents_core.notify.send_notification", fake_notify):
                                pm_core._post_land_deploy_hook("agents-core")

        restart_units = [c[4] for c in sudo_calls if "restart" in c]
        assert "claude-queue-runner.service" in restart_units, "Must restart at budget deadline"
        assert len(notify_calls) == 1
        assert "budget exceeded" in notify_calls[0]["title"] or "budget exceeded" in notify_calls[0]["message"]
        captured = capsys.readouterr()
        assert "CRITICAL" in captured.err
        assert "budget exceeded" in captured.err

    def test_no_duplicate_stacking_while_pending(self, capsys):
        """Two ticks with pending marker + busy queue → no new marker, no extra restart."""
        from datetime import datetime, timezone, timedelta
        sudo_calls = []
        write_calls = []

        def fake_run(cmd, **kwargs):
            if "rev-parse" in cmd:
                return _make_completed_process(returncode=0, stdout="sameshasha")
            if cmd[0] == "sudo" and "restart" in cmd:
                sudo_calls.append(cmd)
                return _make_completed_process(returncode=0)
            return _make_completed_process(returncode=0)

        first_deferred_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        pending = {
            "repo": "agents-core",
            "units": ["claude-queue-runner.service"],
            "first_deferred_at": first_deferred_at,
            "post_head": "",
        }

        for _ in range(2):
            with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.object(pm_core, "_count_inflight_fixers", return_value=5):
                        with patch.object(pm_core, "_read_restart_pending", return_value=pending):
                            with patch.object(pm_core, "_write_restart_pending",
                                              side_effect=lambda *a, **kw: write_calls.append(a)):
                                with patch.object(pm_core, "_clear_restart_pending"):
                                    pm_core._post_land_deploy_hook("agents-core")

        assert len(sudo_calls) == 0, "No restart while pending marker is live and queue busy"
        assert len(write_calls) == 0, "No new marker written while live marker exists"


def _conductor_git_fake_run(
    *, pull_rc=0, fetch_rc=0, fetch_stderr="", head_sha="cafe1111",
    main_sha="cafe1111", status_stdout="", branch="main",
):
    """Fake subprocess.run dispatcher for the conductor pull + R7 freshness-gate calls.

    Handles: `git pull --ff-only`, `git fetch origin main`, `git rev-parse HEAD`,
    `git rev-parse origin/main`, `git rev-parse --abbrev-ref HEAD`, `git status --porcelain`.
    """
    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            sub = cmd[3] if len(cmd) > 3 else ""
            if sub == "pull":
                return _make_completed_process(
                    returncode=pull_rc, stderr="pull failed" if pull_rc else ""
                )
            if sub == "fetch":
                return _make_completed_process(returncode=fetch_rc, stderr=fetch_stderr)
            if sub == "rev-parse":
                if "--abbrev-ref" in cmd:
                    return _make_completed_process(returncode=0, stdout=branch)
                if cmd[-1] == "origin/main":
                    return _make_completed_process(returncode=0, stdout=main_sha)
                return _make_completed_process(returncode=0, stdout=head_sha)
            if sub == "status":
                return _make_completed_process(returncode=0, stdout=status_stdout)
        return _make_completed_process(returncode=0)
    return fake_run


class TestConductorNightPlanDeploy:
    """Tests for night-plan-conductor-deploy-sync-v0.

    Adds conductor to the post-land deploy machinery: (R1) a deploy-clone pull, (R2/R9)
    an atomic, per-file-isolated copy of the night-plan script closure into
    /data/agents/scripts, gated on (R3/R7) a self-fetching source-freshness check, with
    (R10) single-alert ownership so a dirty/stale source pages exactly once.
    """

    # --- R1: classification ---

    def test_conductor_pull_path(self):
        assert pm_core._POST_LAND_PULL["conductor"] == ["/srv/git/conductor"]

    def test_conductor_in_post_land_pull_low_signal(self):
        assert "conductor" in pm_core._POST_LAND_PULL_LOW_SIGNAL

    def test_conductor_not_in_post_land_pull_critical(self):
        assert "conductor" not in pm_core._POST_LAND_PULL_CRITICAL

    def test_conductor_not_in_post_land_restart(self):
        """R5: night scripts are timer-oneshot — no restart map entry."""
        assert "conductor" not in pm_core._POST_LAND_RESTART

    def test_conductor_not_in_post_land_restart_user(self):
        assert "conductor" not in pm_core._POST_LAND_RESTART_USER

    # --- R4: closure-correctness anchor ---

    def test_manifest_includes_night_plan_and_deep_deps(self):
        """The deep deps a naive depth-2 read misses (gpu_lane/rsi_ingest/podcast_engine)
        must be present, not just the top-level entrypoint + yaml modules."""
        for fname in (
            "night_plan.py", "night_coordinator.py", "gpu_lane.py",
            "rsi_ingest.py", "podcast_engine.py", "night_producers.yaml",
        ):
            assert fname in pm_core._CONDUCTOR_NIGHT_SCRIPTS

    def test_fixture_closure_matches_manifest(self):
        """Always-on drift guard: recompute the transitive scripts/-local import
        closure from the committed fixture (seed = {night_plan, night_coordinator,
        every yaml `module:`}; edge = any `from X`/`import X` where X.py exists;
        fixpoint) and assert it EQUALS _CONDUCTOR_NIGHT_SCRIPTS exactly. A manifest
        missing a deep dep (the round-1 hole: gpu_lane/rsi_ingest/podcast_engine)
        fails this test, not a night run. Runs with no dependency on the live
        conductor path — the fixture is the primary always-on vehicle."""
        producers = yaml.safe_load(
            (_CLOSURE_FIXTURE_DIR / "night_producers.yaml").read_text()
        )
        seed = {"night_plan", "night_coordinator"} | {
            p["module"] for p in producers["producers"]
        }

        def source_for(mod):
            f = _CLOSURE_FIXTURE_DIR / f"{mod}.py"
            return f.read_text() if f.is_file() else None

        closure = _compute_import_closure(seed, source_for)
        expected_files = {f"{mod}.py" for mod in closure} | {"night_producers.yaml"}
        assert set(pm_core._CONDUCTOR_NIGHT_SCRIPTS) == expected_files

    def test_load_test_gpu_lane_registers_and_deferred_siblings_import(self):
        """Load-test exercising DEFERRED imports (the ones a bare top-level module
        import does not touch): night_plan + night_coordinator import, the "gpu" lane
        actually REGISTERS (gpu_lane's import is try/except-guarded — a bare `import
        night_coordinator` passes even when gpu_lane.py is absent, so asserting the
        module imports is not sufficient), and each active producer's deferred
        siblings import (arxiv_watch -> rsi_ingest, idea_collider -> podcast_engine)."""
        fixture_modules = [
            "night_plan", "night_coordinator", "gpu_lane", "scout_producer",
            "arxiv_producer", "arxiv_watch", "rsi_ingest",
            "idea_collider_night_batch_producer", "idea_collider",
            "idea_collider_night_batch", "podcast_engine", "kami_producer",
            "kami_batch", "kami_selector", "kami_sweep", "kami_adjudicator",
            "kami_small", "enlightenment_producer", "enlightenment_reader",
            "research_headings",
        ]
        for mod in fixture_modules:
            sys.modules.pop(mod, None)
        fixture_dir = str(_CLOSURE_FIXTURE_DIR)
        sys.path.insert(0, fixture_dir)
        try:
            import night_plan  # noqa: F401
            import night_coordinator

            assert "gpu" in night_coordinator._LANE_HANDLERS, (
                "gpu_lane's guarded import must actually register the 'gpu' lane "
                "for all three gpu-lane producers — a bare `import night_coordinator` "
                "succeeding is not sufficient (guarded-import trap)"
            )

            import arxiv_watch  # noqa: F401
            assert "rsi_ingest" in sys.modules

            import idea_collider  # noqa: F401
            assert "podcast_engine" in sys.modules
        finally:
            sys.path.remove(fixture_dir)
            for mod in fixture_modules:
                sys.modules.pop(mod, None)

    @pytest.mark.skipif(
        importlib.util.find_spec("lapis_engine") is None,
        reason="lapis_engine not importable in this environment — the scout "
        "cross-package precondition is a documented go-live requirement, not "
        "something the closure copy delivers, so it cannot be checked here",
    )
    def test_scout_cross_package_precondition_importable(self):
        """Honest limit: scout_producer's cross-package deps (lapis_pm.scout.*,
        lapis_engine.adapters) are function-deferred, so a bare `import
        scout_producer` does not exercise them, and the closure copy does not (and
        must not claim to) deliver them — they are installed packages, not
        scripts/-local siblings. This only asserts the /data/agents runtime
        precondition that go-live step 3 confirms."""
        import lapis_pm.scout.selection  # noqa: F401
        import lapis_engine.adapters  # noqa: F401

    @pytest.mark.skipif(
        not (Path("/srv/git/conductor") / ".git").is_dir(),
        reason="live conductor deploy clone not present in this environment",
    )
    def test_live_conductor_closure_matches_fixture(self):
        """Skipif live-drift test: recomputes the closure from conductor's
        CANONICAL origin/main (not the local checkout, which may be on a stale
        feature branch — see R3/R7) and asserts it matches _CONDUCTOR_NIGHT_SCRIPTS,
        so fixture staleness vs the real conductor repo is caught wherever this can
        run (BRIX). Soft-skips (not fails) on network/fetch trouble — it verifies
        fixture freshness, it does not gate the deploy."""
        clone = "/srv/git/conductor"
        try:
            fetch = subprocess.run(
                ["git", "-C", clone, "fetch", "origin", "main"],
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            pytest.skip(f"could not fetch origin/main from live conductor clone: {exc}")
        if fetch.returncode != 0:
            pytest.skip(f"git fetch origin main failed: {fetch.stderr!r}")

        def source_for(mod):
            r = subprocess.run(
                ["git", "-C", clone, "show", f"origin/main:scripts/{mod}.py"],
                capture_output=True, text=True, timeout=10,
            )
            return r.stdout if r.returncode == 0 else None

        yaml_result = subprocess.run(
            ["git", "-C", clone, "show", "origin/main:scripts/night_producers.yaml"],
            capture_output=True, text=True, timeout=10,
        )
        if yaml_result.returncode != 0:
            pytest.skip("could not read origin/main:scripts/night_producers.yaml")

        producers = yaml.safe_load(yaml_result.stdout)
        seed = {"night_plan", "night_coordinator"} | {
            p["module"] for p in producers["producers"]
        }
        closure = _compute_import_closure(seed, source_for)
        expected_files = {f"{mod}.py" for mod in closure} | {"night_producers.yaml"}
        assert set(pm_core._CONDUCTOR_NIGHT_SCRIPTS) == expected_files

    # --- R7: source-freshness gate ---

    def test_fetch_fail_with_pending_change_emits_normal_and_skips_copy(self, tmp_path):
        """A fetch failure with a real delivery blocked (dest missing a file the
        source has) must SKIP the copy entirely and alert NORMAL, not copy stale."""
        from agents_core.notify import Priority

        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('v1')")

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "priority": priority})
            return True

        fake_run = _conductor_git_fake_run(fetch_rc=1, fetch_stderr="could not resolve host")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")

        assert list(dest_dir.iterdir()) == [], "fetch-fail must copy zero files, not stale content"
        assert len(notify_calls) == 1
        assert notify_calls[0]["priority"] == Priority.NORMAL

    def test_fetch_fail_with_no_pending_change_emits_low(self, tmp_path):
        """A fetch failure when the runtime already matches the local source is a LOW
        signal (nothing lost, just couldn't re-verify) — not NORMAL."""
        from agents_core.notify import Priority

        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        content = "print('same')"
        (src_dir / "night_plan.py").write_text(content)
        (dest_dir / "night_plan.py").write_text(content)

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(priority)
            return True

        fake_run = _conductor_git_fake_run(fetch_rc=1, fetch_stderr="network unreachable")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")

        assert (dest_dir / "night_plan.py").read_text() == content
        assert len(notify_calls) == 1
        assert notify_calls[0] == Priority.LOW

    def test_dirty_source_skips_copy_emits_normal(self, tmp_path):
        """A dirty source (uncommitted changes) must SKIP the copy, not deliver
        partially-modified content, and alert NORMAL naming remediation."""
        from agents_core.notify import Priority

        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('v1')")

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "priority": priority})
            return True

        fake_run = _conductor_git_fake_run(
            fetch_rc=0, head_sha="aaa", main_sha="aaa",
            status_stdout="M scripts/night_plan.py\n", branch="main",
        )

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")

        assert list(dest_dir.iterdir()) == [], "dirty source must not deliver any file"
        assert len(notify_calls) == 1
        assert notify_calls[0]["priority"] == Priority.NORMAL
        assert "clean-on-main" in notify_calls[0]["message"]

    def test_off_main_source_skips_copy_emits_normal(self, tmp_path):
        """A source checked out on a feature branch (today's live state) must SKIP
        the copy even if clean, else the runtime can regress to a stale manifest."""
        from agents_core.notify import Priority

        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('v1')")

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "priority": priority})
            return True

        fake_run = _conductor_git_fake_run(
            fetch_rc=0, head_sha="aaa", main_sha="bbb", status_stdout="",
            branch="lapis/chub-register-cockpit-8408",
        )

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")

        assert list(dest_dir.iterdir()) == []
        assert len(notify_calls) == 1
        assert notify_calls[0]["priority"] == Priority.NORMAL

    # --- R2/R3/R9: copy mechanics (fresh source) ---

    def test_fresh_source_creates_missing_file(self, tmp_path):
        """The exact go-live case: night_plan.py absent at dest, source verified fresh."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        log_file = tmp_path / "deploy-log.md"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('v1')")

        fake_run = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test-trigger")

        assert (dest_dir / "night_plan.py").read_text() == "print('v1')"
        log_text = log_file.read_text()
        assert "night_plan.py" in log_text
        assert "created" in log_text
        assert "test-trigger" in log_text

    def test_fresh_source_updates_changed_file(self, tmp_path):
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        log_file = tmp_path / "deploy-log.md"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('v2')")
        (dest_dir / "night_plan.py").write_text("print('v1')")

        fake_run = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")

        assert (dest_dir / "night_plan.py").read_text() == "print('v2')"
        log_text = log_file.read_text()
        assert ".." in log_text, "changed-file log line must show old8..new8, not 'created'"
        assert "created" not in log_text

    def test_fresh_source_noop_when_current(self, tmp_path):
        """Idempotent no-op: identical content at dest → no log write, no mutation."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        log_file = tmp_path / "deploy-log.md"
        src_dir.mkdir()
        dest_dir.mkdir()
        content = "print('same')"
        (src_dir / "night_plan.py").write_text(content)
        (dest_dir / "night_plan.py").write_text(content)

        fake_run = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")

        assert (dest_dir / "night_plan.py").read_text() == content
        assert not log_file.exists(), "no-op copy must not write a provenance line"

    def test_source_file_absent_is_skipped_not_raised(self, tmp_path):
        """A manifest entry naming a file conductor doesn't have must not raise —
        it's a manifest bug caught by the drift guard, not a deploy-time failure."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        # src_dir intentionally left without any of the manifested files.

        fake_run = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")  # must not raise

        assert list(dest_dir.iterdir()) == []

    def test_per_file_isolation_oserror_continues_to_next_file(self, tmp_path):
        """An OSError copying file N must not abort the closure — file N+1 still copies."""
        import shutil as _shutil

        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('plan')")
        (src_dir / "night_coordinator.py").write_text("print('coordinator')")

        real_copyfileobj = _shutil.copyfileobj

        def flaky_copyfileobj(fsrc, fdst, *args, **kwargs):
            if getattr(fsrc, "name", "").endswith("night_plan.py"):
                raise OSError("simulated disk-full")
            return real_copyfileobj(fsrc, fdst, *args, **kwargs)

        fake_run = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("shutil.copyfileobj", side_effect=flaky_copyfileobj),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")  # must not raise

        assert not (dest_dir / "night_plan.py").exists(), "the failing file must not land"
        assert (dest_dir / "night_coordinator.py").read_text() == "print('coordinator')", (
            "the file after the failing one must still be copied"
        )
        assert list(dest_dir.glob(".*.tmp")) == [], "no stray temp file after the failure"

    def test_atomic_replace_failure_leaves_dest_whole_and_cleans_temp(self, tmp_path):
        """A failure between temp-write and os.replace must leave the pre-existing
        dest file intact (never partial) and must not leave a stray .tmp file."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('new')")
        (dest_dir / "night_plan.py").write_text("print('old')")

        fake_run = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        def failing_replace(src, dst):
            raise OSError("simulated failure during replace")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("os.replace", side_effect=failing_replace),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")  # must not raise

        assert (dest_dir / "night_plan.py").read_text() == "print('old')", (
            "dest must remain whole (old content) after a failed replace"
        )
        assert list(dest_dir.glob(".*.tmp")) == [], "temp file must be cleaned up on failure"

    # --- R10: single-alert ownership ---

    def test_dirty_source_via_full_hook_emits_exactly_one_alert(self, tmp_path):
        """A generic conductor pull failure PLUS a dirty/off-main freshness-gate source
        is one root cause — the hook must emit exactly one Pushover, not two."""
        from agents_core.notify import Priority

        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        log_file = tmp_path / "deploy-log.md"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('v1')")

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(priority)
            return True

        # The generic `_post_land_git_pull` pull attempt fails (would normally emit a
        # LOW alert for a low-signal repo); the freshness gate's OWN fetch succeeds but
        # finds the source off-main — it, not the generic pull-fail path, must own the
        # single alert emitted here.
        fake_run = _conductor_git_fake_run(
            pull_rc=1, fetch_rc=0, head_sha="aaa", main_sha="bbb",
            branch="lapis/some-feature-branch",
        )

        with (
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_deploy_hook("conductor")

        assert len(notify_calls) == 1, f"expected exactly one alert, got {notify_calls}"
        assert notify_calls[0] == Priority.NORMAL
        assert list(dest_dir.iterdir()) == []

    def test_transient_fetch_fail_via_full_hook_emits_exactly_one_alert(self, tmp_path):
        """The round-4 HIGH: local refs equal+clean but the R7 fetch itself fails must
        still emit exactly ONE alert (not zero) and must not copy stale content."""
        from agents_core.notify import Priority

        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        log_file = tmp_path / "deploy-log.md"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('v1')")
        # dest missing night_plan.py — a real delivery would be blocked by the skip.

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(priority)
            return True

        # generic pull succeeds trivially (no-op, local refs already equal — pre==post);
        # the freshness gate's OWN fetch fails.
        fake_run = _conductor_git_fake_run(
            pull_rc=0, fetch_rc=1, fetch_stderr="could not resolve host",
            head_sha="aaa", main_sha="aaa",
        )

        with (
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._post_land_deploy_hook("conductor")

        assert len(notify_calls) == 1, f"expected exactly one alert, got {notify_calls}"
        assert notify_calls[0] == Priority.NORMAL
        assert list(dest_dir.iterdir()) == [], "a failed fetch must never copy stale content"

    # --- R8: no-producer-drop (copy preserves whatever the canonical source has) ---

    def test_night_producers_yaml_copy_preserves_full_producer_set(self, tmp_path):
        """Copying night_producers.yaml from a verified-fresh source must deliver its
        full byte-for-byte content — no filtering/dropping of any producer entry."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        log_file = tmp_path / "deploy-log.md"
        src_dir.mkdir()
        dest_dir.mkdir()
        yaml_content = (
            "producers:\n"
            "  - module: scout_producer\n    active: true\n"
            "  - module: arxiv_producer\n    active: true\n"
            "  - module: idea_collider_night_batch_producer\n    active: true\n"
        )
        (src_dir / "night_producers.yaml").write_text(yaml_content)
        # runtime previously had a stale/partial yaml — the copy must overwrite it
        # with the full canonical content, not merge or filter.
        (dest_dir / "night_producers.yaml").write_text("producers:\n  - module: scout_producer\n")

        fake_run = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
        ):
            pm_core._deploy_conductor_night_scripts(trigger="test")

        result_yaml = (dest_dir / "night_producers.yaml").read_text()
        assert "idea_collider_night_batch_producer" in result_yaml
        assert result_yaml == yaml_content
