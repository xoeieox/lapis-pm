"""Unit tests for _post_land_deploy_hook in pm_core."""

import hashlib
import importlib.util
import json
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
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

    def test_agents_core_map_contains_mem_server(self):
        """mem-hygiene-postland-restart-map-v0 D-1: mem-server.service is in the
        agents-core user-unit restart map (the /v0/hygiene/* run surface must not
        silently no-serve after a land)."""
        assert "mem-server.service" in pm_core._POST_LAND_RESTART_USER["agents-core"]
        # The existing two entries are untouched — this is an additive entry.
        assert pm_core._POST_LAND_RESTART_USER["agents-core"] == (
            "doorman-server.service", "slot-server.service", "mem-server.service",
        )

    def test_agents_core_pull_critical_untouched(self):
        """D-1 is a restart-surface change only: the pull critical set is untouched."""
        assert pm_core._POST_LAND_PULL_CRITICAL == frozenset({"lapis-pm", "agents-core", "synapse"})

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
        assert len(user_restart_calls) == 3
        assert user_restart_calls[0] == ["systemctl", "--user", "restart", "doorman-server.service"]
        assert user_restart_calls[1] == ["systemctl", "--user", "restart", "slot-server.service"]
        assert user_restart_calls[2] == ["systemctl", "--user", "restart", "mem-server.service"]

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
        assert len(restart_user) == 3
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


# A synthetic user unit that imports agents_core but is NOT in the restart map
# (the guard must flag it). Module form, mirroring slot-server.service's
# `python3 -m agents_core.slot_server` ExecStart shape.
_SYNTHETIC_UNMAPPED_UNIT = """[Unit]
Description=synthetic agents_core importer (test fixture)

[Service]
Type=simple
ExecStart=/usr/bin/env python3 -m agents_core.synthetic_worker
Restart=on-failure

[Install]
WantedBy=default.target
"""

# A synthetic user unit that does NOT import agents_core (the guard must stay
# quiet about it).
_SYNTHETIC_UNRELATED_UNIT = """[Unit]
Description=synthetic unrelated service (test fixture)

[Service]
Type=simple
ExecStart=/usr/bin/env python3 /opt/something/unrelated.py
Restart=on-failure

[Install]
WantedBy=default.target
"""


class TestAgentsCoreRestartMapAudit:
    """mem-hygiene-postland-restart-map-v0 D-2/D-3: the §2.3 growth obligation
    fired with mem-server.service as the third agents-core entry — the restart
    map is now self-auditing. The guard scans the user-unit dir for agents_core
    importers and logs LOUDLY for any absent from _POST_LAND_RESTART_USER.
    LOG-LOUD, never FAIL-STOP: findings never raise, never block a land.
    """

    def _write_units(self, tmp_path: Path, names: dict[str, str]) -> Path:
        unit_dir = tmp_path / "user"
        unit_dir.mkdir()
        for name, text in names.items():
            (unit_dir / name).write_text(text)
        return unit_dir

    def test_guard_flags_synthetic_unmapped_importer(self, tmp_path, capsys):
        """(b) The guard flags a synthetic user unit that imports agents_core
        but is absent from the map — loud stderr + returned finding."""
        unit_dir = self._write_units(
            tmp_path,
            {
                "synthetic-worker.service": _SYNTHETIC_UNMAPPED_UNIT,
                "unrelated.service": _SYNTHETIC_UNRELATED_UNIT,
            },
        )
        findings = pm_core._audit_agents_core_user_units(unit_dir)
        assert findings == ["synthetic-worker.service"]
        captured = capsys.readouterr()
        assert "AUDIT" in captured.err
        assert "synthetic-worker.service" in captured.err
        assert "ABSENT from _POST_LAND_RESTART_USER" in captured.err

    def test_guard_quiet_when_map_complete(self, tmp_path, capsys):
        """(b) The guard stays quiet when every importer is in the map."""
        # The real map (agents-core tuple incl. mem-server.service) + the
        # synthetic unit's name → complete.
        restart_user = dict(pm_core._POST_LAND_RESTART_USER)
        restart_user["agents-core"] = restart_user["agents-core"] + ("synthetic-worker.service",)
        unit_dir = self._write_units(
            tmp_path,
            {
                "synthetic-worker.service": _SYNTHETIC_UNMAPPED_UNIT,
                "unrelated.service": _SYNTHETIC_UNRELATED_UNIT,
            },
        )
        findings = pm_core._audit_agents_core_user_units(unit_dir, restart_user=restart_user)
        assert findings == []
        captured = capsys.readouterr()
        assert "AUDIT" not in captured.err

    def test_guard_ignores_non_importer_units(self, tmp_path, capsys):
        """A unit that never imports agents_core is not a finding, even if
        absent from the map."""
        unit_dir = self._write_units(
            tmp_path,
            {"unrelated.service": _SYNTHETIC_UNRELATED_UNIT},
        )
        findings = pm_core._audit_agents_core_user_units(unit_dir)
        assert findings == []
        assert "AUDIT" not in capsys.readouterr().err

    def test_guard_detects_wrapper_script_importer(self, tmp_path):
        """Known-wrapper-set detection: a unit whose ExecStart runs a script
        under a wrapper root that itself imports agents_core is an importer
        (the doorman pattern — the .service text never names agents_core)."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "wrapper.py"
        script.write_text("import sys\nfrom agents_core.wrapper_server import main\n")
        unit_dir = self._write_units(
            tmp_path,
            {
                "wrapper.service": (
                    "[Service]\n"
                    f"ExecStart=/usr/bin/env python3 {script}\n"
                    "Restart=on-failure\n"
                ),
            },
        )
        # The script is not under the real wrapper root, so point the root at
        # the temp dir: detection is root-relative by construction.
        findings = pm_core._audit_agents_core_user_units(
            unit_dir, wrapper_roots=(str(scripts_dir) + "/",),
        )
        assert findings == ["wrapper.service"]

    def test_guard_missing_dir_is_loud_not_fatal(self, tmp_path, capsys):
        """A missing/unreadable unit dir is one loud line + empty findings —
        the guard must never raise or block a land."""
        findings = pm_core._audit_agents_core_user_units(tmp_path / "does-not-exist")
        assert findings == []
        captured = capsys.readouterr()
        assert "AUDIT" in captured.err
        assert "SKIPPED" in captured.err

    def test_guard_hook_integration_agents_core(self, tmp_path, capsys):
        """End-to-end: the agents-core hook pass runs the audit (findings land
        on the deploy-pass-report stderr surface) and the land is never blocked."""
        unit_dir = self._write_units(
            tmp_path,
            {"synthetic-worker.service": _SYNTHETIC_UNMAPPED_UNIT},
        )
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                        with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                            with patch.object(pm_core, "_read_restart_pending", return_value=None):
                                with patch("pathlib.Path.home", return_value=tmp_path / "home"):
                                    (tmp_path / "home" / ".config" / "systemd" / "user").mkdir(parents=True)
                                    (tmp_path / "home" / ".config" / "systemd" / "user" / "synthetic-worker.service").write_text(_SYNTHETIC_UNMAPPED_UNIT)
                                    pm_core._post_land_deploy_hook("agents-core")  # must not raise

        captured = capsys.readouterr()
        assert "AUDIT" in captured.err
        assert "synthetic-worker.service" in captured.err

    def test_guard_hook_not_run_for_other_repos(self, tmp_path, capsys):
        """The audit is agents-core-scoped: a non-agents-core land never runs it
        (no scanner noise on unrelated lands)."""
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                path = cmd[2]
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _make_completed_process(returncode=0, stdout=sha)
            if cmd[0] == "git" and "status" in cmd:
                return _make_completed_process(returncode=0, stdout="")
            return _make_completed_process(returncode=0, stdout="active")

        with patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False):
            with patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"):
                with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                    with patch.dict("os.environ", {"XDG_RUNTIME_DIR": "/run/user/1000"}):
                        with patch.object(pm_core, "_count_inflight_fixers", return_value=0):
                            with patch.object(pm_core, "_read_restart_pending", return_value=None):
                                with patch("pathlib.Path.home", return_value=tmp_path / "home"):
                                    (tmp_path / "home" / ".config" / "systemd" / "user").mkdir(parents=True)
                                    (tmp_path / "home" / ".config" / "systemd" / "user" / "synthetic-worker.service").write_text(_SYNTHETIC_UNMAPPED_UNIT)
                                    pm_core._post_land_deploy_hook("cockpit")  # must not raise

        captured = capsys.readouterr()
        assert "synthetic-worker.service" not in captured.err


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
            # Slice 1: the selfheal pass owns critical-repo failures; mock it
            # False so this test exercises the legacy notify FALLBACK.
            patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure", return_value=False),
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
            # Slice 1: mock the selfheal pass False to exercise the legacy notify
            # fallback (the machine did not own this failure).
            patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure", return_value=False),
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
    """Tests for _check_deploy_currency — deploy-currency health monitoring.

    Generalized (agents-core-deploy-drift-backstop-v0 §A) to iterate every
    clone path across all _POST_LAND_PULL_CRITICAL repos (lapis-pm,
    agents-core's two clones, synapse), not just the lapis-pm deploy clone.
    `_by_path_fake_run` drives each clone's local/origin SHA independently so
    tests can force exactly one clone stale while the rest stay current.
    """

    @pytest.fixture(autouse=True)
    def _isolate_currency_status_file(self, tmp_path):
        """Every check now also writes the §D commit-distance status file
        (agents-core-deploy-drift-backstop-v0). Redirect it to tmp_path so these
        tests never touch the real /srv/lapis/lapis-state/deploy-currency-status.json.
        """
        with patch.object(
            pm_core, "_DEPLOY_CURRENCY_STATUS_FILE", tmp_path / "deploy-currency-status.json",
        ):
            yield

    @staticmethod
    def _by_path_fake_run(stale_paths, distance=5):
        """subprocess.run stub: clones in `stale_paths` report local != origin/main
        (and a fixed rev-list --count distance); every other mapped clone reports
        current (local == origin/main)."""
        def fake_run(cmd, **kwargs):
            path = cmd[2] if len(cmd) > 2 and cmd[1] == "-C" else None
            if "fetch" in cmd:
                return _make_completed_process(returncode=0)
            if "rev-list" in cmd:
                return _make_completed_process(returncode=0, stdout=str(distance))
            if "rev-parse" in cmd:
                if path in stale_paths:
                    if "origin/main" in cmd:
                        return _make_completed_process(returncode=0, stdout="newsha0new")
                    return _make_completed_process(returncode=0, stdout="oldsha0old")
                return _make_completed_process(returncode=0, stdout="same1234same")
            return _make_completed_process(returncode=0)
        return fake_run

    def test_alerts_when_deploy_clone_is_behind(self):
        """Exactly one clone forced N-behind: one NORMAL alert naming repo, path,
        and commit-distance; mem is stamped with a per-clone key; no other
        clone's currency is disturbed (AC1)."""
        notify_calls = []
        mem_sets = {}

        fake_mem = MagicMock()
        fake_mem.get.return_value = None  # no prior alert
        fake_mem.set.side_effect = lambda k, v, **kw: mem_sets.update({k: v})

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "title": title, "priority": priority})
            return True

        with (
            patch(
                "lapis_pm.pm_core.subprocess.run",
                side_effect=self._by_path_fake_run({"/srv/git/lapis-pm"}, distance=7),
            ),
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._check_deploy_currency()

        assert len(notify_calls) == 1, "Expected exactly one Pushover alert on the one stale clone"
        assert "stale" in notify_calls[0]["title"]
        assert "lapis-pm" in notify_calls[0]["title"]
        assert "/srv/git/lapis-pm" in notify_calls[0]["message"]
        assert "7 commits behind" in notify_calls[0]["message"]
        from agents_core.notify import Priority
        assert notify_calls[0]["priority"] == Priority.NORMAL
        assert f"{pm_core._DEPLOY_CURRENCY_STALE_KEY}:/srv/git/lapis-pm" in mem_sets

    def test_no_alert_when_deploy_clone_is_current(self):
        """When every critical clone's local HEAD matches origin/main, no alert fires."""
        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        with (
            patch(
                "lapis_pm.pm_core.subprocess.run",
                side_effect=self._by_path_fake_run(set()),
            ),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            pm_core._check_deploy_currency()

        assert len(notify_calls) == 0

    def test_multiple_stale_clones_alert_independently(self):
        """Two distinct stale clones each fire their own alert in one check (AC1/AC2)."""
        notify_calls = []
        fake_mem = MagicMock()
        fake_mem.get.return_value = None

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "title": title})
            return True

        with (
            patch(
                "lapis_pm.pm_core.subprocess.run",
                side_effect=self._by_path_fake_run({"/srv/git/lapis-pm", "/data/agents"}),
            ),
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._check_deploy_currency()

        assert len(notify_calls) == 2
        messages = [c["message"] for c in notify_calls]
        assert any("/srv/git/lapis-pm" in m for m in messages)
        assert any("/data/agents" in m for m in messages)

    def test_cooldown_suppresses_repeat_alert_per_clone(self):
        """Within cooldown, the previously-alerted clone stays silent, but a
        DIFFERENT drifted clone still alerts independently (AC2)."""
        from datetime import datetime, timezone, timedelta
        notify_calls = []

        recent_ts = (
            datetime.now(timezone.utc) - timedelta(seconds=60)
        ).isoformat()  # 1 minute ago, well within 1h cooldown

        fake_mem = MagicMock()
        fake_mem.get.side_effect = (
            lambda key: {"content": recent_ts} if key.endswith("/srv/git/lapis-pm") else None
        )

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "title": title})
            return True

        with (
            patch(
                "lapis_pm.pm_core.subprocess.run",
                side_effect=self._by_path_fake_run({"/srv/git/lapis-pm", "/data/agents"}),
            ),
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._check_deploy_currency()

        assert len(notify_calls) == 1, (
            "Cooled-down clone stayed silent; the other stale clone still alerted"
        )
        assert "/data/agents" in notify_calls[0]["message"]

    def test_exception_during_check_does_not_raise(self):
        """Any exception in _check_deploy_currency must not propagate (best-effort)."""
        def fake_run(cmd, **kwargs):
            raise OSError("git not found")

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
            pm_core._check_deploy_currency()  # must not raise

    def test_status_file_records_commit_distance_for_every_clone(self):
        """§D: every check (current or stale) stamps a commit-distance snapshot for
        every critical clone into the shared status file, keyed by clone path —
        the quantitative meter a visible surface (e.g. doorman `/status`) reads."""
        with patch(
            "lapis_pm.pm_core.subprocess.run",
            side_effect=self._by_path_fake_run({"/data/agents"}, distance=3),
            ):
            pm_core._check_deploy_currency()

        data = json.loads(pm_core._DEPLOY_CURRENCY_STATUS_FILE.read_text())
        assert data["/data/agents"]["commits_behind"] == 3
        assert data["/data/agents"]["sha"] == "oldsha0old"
        assert data["/data/agents"]["repo"] == "agents-core"
        # A current clone is recorded too, with commits_behind == 0.
        assert data["/srv/git/lapis-pm"]["commits_behind"] == 0

    def test_status_file_write_failure_does_not_raise(self):
        """A status-file write failure (e.g. unwritable path) is best-effort and must
        not block the alert path."""
        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        with (
            patch(
                "lapis_pm.pm_core.subprocess.run",
                side_effect=self._by_path_fake_run({"/srv/git/lapis-pm"}),
            ),
            patch.object(
                pm_core, "_DEPLOY_CURRENCY_STATUS_FILE",
                Path("/nonexistent-root/deploy-currency-status.json"),
            ),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            pm_core._check_deploy_currency()  # must not raise

        assert len(notify_calls) == 1


class TestReconcileDeployInventoryDedup:
    """Tests for _reconcile_deploy_inventory's cooldown-read fix (Defect A) and
    Pushover notification dedup against the prior snapshot (Defect B) —
    lapis-pm-deploy-inventory-notify-dedup-v0.
    """

    @pytest.fixture(autouse=True)
    def _isolate_status_file(self, tmp_path):
        from lapis_pm import deploy_inventory
        with patch.object(
            deploy_inventory, "_STATUS_FILE", tmp_path / "deploy-inventory-status.json",
        ):
            yield

    @staticmethod
    def _status(findings_by_clone):
        """{clone_path: [(kind, severity, detail), ...]} -> a minimal status dict."""
        clones = []
        for path, findings in findings_by_clone.items():
            clones.append({
                "path": path,
                "findings": [
                    {"kind": kind, "severity": sev, "detail": detail}
                    for kind, sev, detail in findings
                ],
            })
        return {"generated": "2026-07-22T00:00:00+00:00", "clones": clones}

    def test_cooldown_within_window_skips_reconcile_pass(self):
        """A last-run key read via the correct ["content"] idiom, within the
        cooldown window, skips the pass entirely (Defect A)."""
        from datetime import datetime, timezone, timedelta
        recent_ts = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        fake_mem = MagicMock()
        fake_mem.get.return_value = {"content": recent_ts}

        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass") as fake_run_pass,
        ):
            pm_core._reconcile_deploy_inventory()

        fake_run_pass.assert_not_called()

    def test_cooldown_stale_runs_reconcile_pass(self):
        """A last-run key older than the cooldown window runs the pass (Defect A)."""
        from datetime import datetime, timezone, timedelta
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=7200)).isoformat()
        fake_mem = MagicMock()
        fake_mem.get.return_value = {"content": old_ts}

        status = self._status({})
        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass", return_value=status) as fake_run_pass,
        ):
            pm_core._reconcile_deploy_inventory()

        fake_run_pass.assert_called_once()

    def test_finding_unchanged_since_prior_pass_does_not_notify(self):
        from lapis_pm import deploy_inventory
        prior = self._status({"/srv/git/lapis-pm": [("tracked_dirty_tree", "HIGH", "dirty")]})
        deploy_inventory.write_status_json(prior)

        current = self._status({"/srv/git/lapis-pm": [("tracked_dirty_tree", "HIGH", "dirty")]})

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        fake_mem = MagicMock()
        fake_mem.get.return_value = None  # cooldown clear

        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass", return_value=current),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            pm_core._reconcile_deploy_inventory()

        assert notify_calls == []

    def test_new_finding_since_prior_pass_notifies(self):
        from lapis_pm import deploy_inventory
        prior = self._status({"/srv/git/lapis-pm": [("tracked_dirty_tree", "HIGH", "dirty")]})
        deploy_inventory.write_status_json(prior)

        current = self._status({
            "/srv/git/lapis-pm": [
                ("tracked_dirty_tree", "HIGH", "dirty"),
                ("stray_branch", "HIGH", "on a branch"),
            ],
        })

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        fake_mem = MagicMock()
        fake_mem.get.return_value = None

        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass", return_value=current),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            pm_core._reconcile_deploy_inventory()

        assert notify_calls == ["deploy-inventory: stray_branch"]

    def test_first_ever_pass_notifies_all_current_high_findings(self):
        """No prior snapshot on disk: every current HIGH finding notifies once,
        not suppressed (spec item 2)."""
        current = self._status({
            "/srv/git/lapis-pm": [("tracked_dirty_tree", "HIGH", "dirty")],
            "/data/agents": [("stale_behind_origin", "HIGH", "behind")],
        })

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        fake_mem = MagicMock()
        fake_mem.get.return_value = None

        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass", return_value=current),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            pm_core._reconcile_deploy_inventory()

        assert len(notify_calls) == 2

    def test_corrupt_prior_snapshot_fires_single_audit_and_suppresses_per_finding_pushes(
        self, tmp_path,
    ):
        """A present-but-unparseable prior snapshot must not fail-open (notifying
        every current finding) nor fail-silent (no signal at all): exactly one
        HIGH audit push fires, and per-finding pushes are suppressed for this
        pass. The next pass is gated by the (now-functioning) coarse cooldown."""
        status_path = tmp_path / "deploy-inventory-status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text("{not valid json")

        current = self._status({
            "/srv/git/lapis-pm": [("tracked_dirty_tree", "HIGH", "dirty")],
            "/data/agents": [("stray_branch", "HIGH", "branch")],
        })

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        fake_mem = MagicMock()
        fake_mem.get.return_value = None

        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass", return_value=current),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            pm_core._reconcile_deploy_inventory()

        assert notify_calls == ["deploy-inventory: status file unreadable"]

        # Next tick: the coarse cooldown (now correctly reading ["content"])
        # gates the pass before the corrupt-snapshot logic ever re-runs.
        fake_mem.get.return_value = {"content": datetime.now(timezone.utc).isoformat()}
        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass") as fake_run_pass_2,
        ):
            pm_core._reconcile_deploy_inventory()

        fake_run_pass_2.assert_not_called()


class TestExpertsDeployMapEntry:
    """Part A regression coverage (lapis-pm-deploy-inventory-auto-recovery-v0):
    experts must be pull- and restart-mapped, and absent from the critical/
    low-signal sets (spec item 3 — nothing depends on that distinction here).
    """

    def test_experts_in_post_land_pull(self):
        assert pm_core._POST_LAND_PULL.get("experts") == ["/srv/git/experts"]

    def test_experts_in_post_land_restart_system_scope_not_user(self):
        assert pm_core._POST_LAND_RESTART.get("experts") == (
            "inertia-expert.service", "lapis-expert.service",
        )
        assert "experts" not in pm_core._POST_LAND_RESTART_USER

    def test_experts_not_critical_or_low_signal(self):
        assert "experts" not in pm_core._POST_LAND_PULL_CRITICAL
        assert "experts" not in pm_core._POST_LAND_PULL_LOW_SIGNAL

    def test_experts_mapped_true_in_synthetic_inventory_fixture(self, tmp_path):
        from lapis_pm import deploy_inventory
        clone_dir = tmp_path / "experts"
        clone_dir.mkdir()
        reverse_map = deploy_inventory._reverse_pull_map({"experts": [str(clone_dir)]})
        assert reverse_map[str(clone_dir.resolve())] == "experts"


class TestDeployInventoryAutoRecovery:
    """Tests for script-tier auto-recovery in _reconcile_deploy_inventory
    (lapis-pm-deploy-inventory-auto-recovery-v0)."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path):
        from lapis_pm import deploy_inventory
        with (
            patch.object(deploy_inventory, "_STATUS_FILE", tmp_path / "deploy-inventory-status.json"),
            patch.object(pm_core, "_DEPLOY_PULL_LOCK_DIR", tmp_path / "lock"),
            patch.object(pm_core, "_AUTO_RECOVERY_RESTART_POLL_INTERVAL_SECS", 0.001),
            patch.object(pm_core, "_AUTO_RECOVERY_RESTART_POLL_WINDOW_SECS", 0.01),
        ):
            yield

    @staticmethod
    def _clone(path, *, mapped=True, commits_behind=0, branch="main",
               tracked_dirty=False, untracked_present=False, findings=()):
        return {
            "path": path,
            "remote": "origin",
            "mapped": mapped,
            "acked": False,
            "backing_units": [],
            "head": "abc123",
            "branch": branch,
            "commits_behind": commits_behind,
            "tracked_dirty": tracked_dirty,
            "untracked_present": untracked_present,
            "lock_age_secs": None,
            "findings": [{"kind": k, "severity": s, "detail": d} for k, s, d in findings],
        }

    @staticmethod
    def _status(clones):
        return {"generated": "2026-07-23T00:00:00+00:00", "clones": clones}

    @staticmethod
    def _clean_currency():
        from lapis_pm import deploy_inventory
        return deploy_inventory.CurrencyResult(
            head="deadbeef", branch="main", commits_behind=0,
            tracked_dirty=False, untracked_present=False, findings=[],
        )

    # --- eligibility gate (item 4) ---

    def test_stale_only_mapped_unlocked_eligible(self):
        clone = self._clone("/srv/git/experts", commits_behind=3,
                             findings=[("stale_behind_origin", "HIGH", "3 behind")])
        assert pm_core._auto_recovery_eligible(clone) is True

    def test_stray_branch_alongside_stale_not_eligible(self):
        clone = self._clone("/srv/git/experts", branch="feature-x",
                             findings=[
                                 ("stale_behind_origin", "HIGH", "3 behind"),
                                 ("stray_branch", "HIGH", "on feature-x"),
                             ])
        assert pm_core._auto_recovery_eligible(clone) is False

    def test_tracked_dirty_alongside_stale_not_eligible(self):
        clone = self._clone("/srv/git/experts", tracked_dirty=True,
                             findings=[
                                 ("stale_behind_origin", "HIGH", "3 behind"),
                                 ("tracked_dirty_tree", "HIGH", "dirty"),
                             ])
        assert pm_core._auto_recovery_eligible(clone) is False

    def test_detached_head_already_excluded_via_stray_branch_finding(self):
        """A mid-rebase/bisect clone shows a DETACHED branch, which already
        trips stray_branch — no separate in-progress-op marker check needed."""
        clone = self._clone("/srv/git/experts", branch="DETACHED",
                             findings=[
                                 ("stale_behind_origin", "HIGH", "3 behind"),
                                 ("stray_branch", "HIGH", "on 'DETACHED'"),
                             ])
        assert pm_core._auto_recovery_eligible(clone) is False

    def test_conflicted_tree_already_excluded_via_tracked_dirty_finding(self):
        """A conflicted cherry-pick/am/merge leaves tracked modifications,
        which already trips tracked_dirty_tree — same reasoning as above."""
        clone = self._clone("/srv/git/experts", tracked_dirty=True,
                             findings=[
                                 ("stale_behind_origin", "HIGH", "3 behind"),
                                 ("tracked_dirty_tree", "HIGH", "conflict markers present"),
                             ])
        assert pm_core._auto_recovery_eligible(clone) is False

    def test_unmapped_clone_not_eligible(self):
        clone = self._clone("/srv/git/experts", mapped=False,
                             findings=[("stale_behind_origin", "HIGH", "3 behind")])
        assert pm_core._auto_recovery_eligible(clone) is False

    def test_locked_clone_not_eligible_regardless_of_critical(self):
        clone_path = "/srv/git/lapis-pm"  # a _POST_LAND_PULL_CRITICAL repo
        pm_core._write_deploy_pull_lock(clone_path, "genuine divergence", 5)
        clone = self._clone(clone_path,
                             findings=[("stale_behind_origin", "HIGH", "3 behind")])
        assert pm_core._auto_recovery_eligible(clone) is False

    # --- full recovery step (items 5, 6, 7) ---

    def test_recovery_attempted_calls_hook_with_repo_and_trigger(self):
        clone = self._clone("/srv/git/experts", commits_behind=3,
                             findings=[("stale_behind_origin", "HIGH", "3 behind")])
        status = self._status([clone])

        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"experts": ["/srv/git/experts"]}),
            patch.object(pm_core, "_POST_LAND_RESTART",
                         {"experts": ("inertia-expert.service", "lapis-expert.service")}),
            patch.object(pm_core, "_POST_LAND_RESTART_USER", {}),
            patch.object(pm_core, "_post_land_deploy_hook") as fake_hook,
            patch("lapis_pm.deploy_inventory.check_currency", return_value=self._clean_currency()),
            patch("lapis_pm.pm_core.subprocess.run",
                  return_value=MagicMock(returncode=0, stdout="active\n")),
            patch("agents_core.notify.send_notification", lambda **kw: True),
            ):
            pm_core._run_deploy_inventory_auto_recovery(status)

        fake_hook.assert_called_once_with("experts", trigger="deploy-inventory-auto-recovery")

    def test_ffonly_refusal_leaves_finding_no_crash_no_low_ping(self):
        """A diverged clone (ff-only refuses) is treated like any other failed
        pull: post-recheck still shows the HIGH finding, no exception raised,
        no LOW ping sent."""
        clone = self._clone("/srv/git/experts", commits_behind=3,
                             findings=[("stale_behind_origin", "HIGH", "3 behind")])
        status = self._status([clone])
        from lapis_pm import deploy_inventory
        still_stale = deploy_inventory.CurrencyResult(
            head="oldsha", branch="main", commits_behind=3, tracked_dirty=False,
            untracked_present=False,
            findings=[deploy_inventory.Finding("stale_behind_origin", "HIGH", "still 3 behind")],
        )

        notify_calls = []
        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"experts": ["/srv/git/experts"]}),
            patch.object(pm_core, "_POST_LAND_RESTART", {}),
            patch.object(pm_core, "_POST_LAND_RESTART_USER", {}),
            patch.object(pm_core, "_post_land_deploy_hook"),
            patch("lapis_pm.deploy_inventory.check_currency", return_value=still_stale),
            patch("agents_core.notify.send_notification",
                  lambda **kw: notify_calls.append(kw) or True),
        ):
            pm_core._run_deploy_inventory_auto_recovery(status)

        assert notify_calls == []
        assert status["clones"][0]["commits_behind"] == 3
        assert status["clones"][0]["findings"][0]["kind"] == "stale_behind_origin"

    def test_restart_unit_never_active_synthesizes_high_finding_no_low_ping(self):
        clone = self._clone("/srv/git/experts", commits_behind=2,
                             findings=[("stale_behind_origin", "HIGH", "2 behind")])
        status = self._status([clone])

        notify_calls = []
        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"experts": ["/srv/git/experts"]}),
            patch.object(pm_core, "_POST_LAND_RESTART",
                         {"experts": ("inertia-expert.service", "lapis-expert.service")}),
            patch.object(pm_core, "_POST_LAND_RESTART_USER", {}),
            patch.object(pm_core, "_post_land_deploy_hook"),
            patch("lapis_pm.deploy_inventory.check_currency", return_value=self._clean_currency()),
            patch("lapis_pm.pm_core.subprocess.run",
                  return_value=MagicMock(returncode=0, stdout="inactive\n")),
            patch("agents_core.notify.send_notification",
                  lambda **kw: notify_calls.append(kw) or True),
        ):
            pm_core._run_deploy_inventory_auto_recovery(status)

        assert notify_calls == []
        findings = status["clones"][0]["findings"]
        kinds = [f["kind"] for f in findings]
        assert "auto_recovery_restart_failed" in kinds
        high = next(f for f in findings if f["kind"] == "auto_recovery_restart_failed")
        assert high["severity"] == "HIGH"

    def test_restart_unit_activating_then_active_treated_as_success(self):
        """The retry window absorbs normal restart latency instead of
        false-positiving on an immediate first check (spec-review R2)."""
        clone = self._clone("/srv/git/experts", commits_behind=1,
                             findings=[("stale_behind_origin", "HIGH", "1 behind")])
        status = self._status([clone])

        notify_calls = []
        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"experts": ["/srv/git/experts"]}),
            patch.object(pm_core, "_POST_LAND_RESTART", {"experts": ("inertia-expert.service",)}),
            patch.object(pm_core, "_POST_LAND_RESTART_USER", {}),
            patch.object(pm_core, "_post_land_deploy_hook"),
            patch("lapis_pm.deploy_inventory.check_currency", return_value=self._clean_currency()),
            patch(
                "lapis_pm.pm_core.subprocess.run",
                side_effect=[
                    MagicMock(returncode=0, stdout="activating\n"),
            MagicMock(returncode=0, stdout="activating\n"),
                    MagicMock(returncode=0, stdout="active\n"),
                ],
            ),
            patch("agents_core.notify.send_notification",
                  lambda **kw: notify_calls.append(kw) or True),
        ):
            pm_core._run_deploy_inventory_auto_recovery(status)

        assert len(notify_calls) == 1
        from agents_core.notify import Priority
        assert notify_calls[0]["priority"] == Priority.LOW
        kinds = [f["kind"] for f in status["clones"][0]["findings"]]
        assert "auto_recovery_restart_failed" not in kinds

    def test_empty_restart_covered_set_is_vacuously_successful(self):
        """code-reviewer/facets/gardener/conductor/rag-ops/lapis-pm style repos
        (no restart entry) recover on git currency alone — an empty
        restart-covered set must read as success, not as drift."""
        clone = self._clone("/srv/git/code-reviewer-working", commits_behind=1,
                             findings=[("stale_behind_origin", "HIGH", "1 behind")])
        status = self._status([clone])

        notify_calls = []
        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"code-reviewer": ["/srv/git/code-reviewer-working"]}),
            patch.object(pm_core, "_POST_LAND_RESTART", {}),
            patch.object(pm_core, "_POST_LAND_RESTART_USER", {}),
            patch.object(pm_core, "_post_land_deploy_hook"),
            patch("lapis_pm.deploy_inventory.check_currency", return_value=self._clean_currency()),
            patch("lapis_pm.pm_core.subprocess.run") as fake_run,
            patch("agents_core.notify.send_notification",
                  lambda **kw: notify_calls.append(kw) or True),
        ):
            pm_core._run_deploy_inventory_auto_recovery(status)

        fake_run.assert_not_called()  # nothing to poll — never even called
        assert len(notify_calls) == 1
        from agents_core.notify import Priority
        assert notify_calls[0]["priority"] == Priority.LOW
        kinds = [f["kind"] for f in status["clones"][0]["findings"]]
        assert "auto_recovery_restart_failed" not in kinds

    # --- integration through _reconcile_deploy_inventory (items 7, 8) ---

    def test_reconcile_integration_success_persists_fresh_state_and_low_ping(self):
        from lapis_pm import deploy_inventory
        current = self._status([
            self._clone("/srv/git/experts", commits_behind=2,
                        findings=[("stale_behind_origin", "HIGH", "2 behind")]),
        ])

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append((title, priority))
            return True

        fake_mem = MagicMock()
        fake_mem.get.return_value = None

        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch.object(pm_core, "_POST_LAND_PULL", {"experts": ["/srv/git/experts"]}),
            patch.object(pm_core, "_POST_LAND_RESTART",
                         {"experts": ("inertia-expert.service", "lapis-expert.service")}),
            patch.object(pm_core, "_POST_LAND_RESTART_USER", {}),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass", return_value=current),
            patch.object(pm_core, "_post_land_deploy_hook"),
            patch("lapis_pm.deploy_inventory.check_currency", return_value=self._clean_currency()),
            patch("lapis_pm.pm_core.subprocess.run",
                  return_value=MagicMock(returncode=0, stdout="active\n")),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            pm_core._reconcile_deploy_inventory()

        from agents_core.notify import Priority
        assert notify_calls == [("deploy-inventory: auto-recovered", Priority.LOW)]

        persisted = deploy_inventory.read_status_json()
        assert persisted["clones"][0]["commits_behind"] == 0
        assert persisted["clones"][0]["findings"] == []

    def test_reconcile_integration_failed_recovery_falls_through_to_existing_alert(self):
        from lapis_pm import deploy_inventory
        current = self._status([
            self._clone("/srv/git/experts", commits_behind=2,
                        findings=[("stale_behind_origin", "HIGH", "2 behind")]),
        ])
        still_stale = deploy_inventory.CurrencyResult(
            head="oldsha", branch="main", commits_behind=2, tracked_dirty=False,
            untracked_present=False,
            findings=[deploy_inventory.Finding("stale_behind_origin", "HIGH", "still 2 behind")],
        )

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        fake_mem = MagicMock()
        fake_mem.get.return_value = None

        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch.object(pm_core, "_POST_LAND_PULL", {"experts": ["/srv/git/experts"]}),
            patch.object(pm_core, "_POST_LAND_RESTART", {}),
            patch.object(pm_core, "_POST_LAND_RESTART_USER", {}),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass", return_value=current),
            patch.object(pm_core, "_post_land_deploy_hook"),
            patch("lapis_pm.deploy_inventory.check_currency", return_value=still_stale),
            patch("agents_core.notify.send_notification", fake_notify),
        ):
            pm_core._reconcile_deploy_inventory()

        assert notify_calls == ["deploy-inventory: stale_behind_origin"]

    def test_auto_recovery_restart_failed_exempt_from_dedup_fires_every_pass(self):
        """auto_recovery_restart_failed is the one HIGH kind that does NOT go
        quiet while persisting (spec-review R4 dedup exemption) — every other
        kind still follows the 2026-07-22 no-re-alert-floor ruling unchanged."""
        from lapis_pm import deploy_inventory
        finding = ("auto_recovery_restart_failed", "HIGH", "restart did not take")
        # mapped: False so the auto-recovery step itself is a no-op here —
        # isolates the notify-loop dedup-exemption behavior specifically.
        current = self._status([self._clone("/srv/git/experts", mapped=False, findings=[finding])])

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(title)
            return True

        fake_mem = MagicMock()
        fake_mem.get.return_value = None

        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass", return_value=current),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            pm_core._reconcile_deploy_inventory()

        assert notify_calls == ["deploy-inventory: auto_recovery_restart_failed"]

        # Second pass: the prior snapshot on disk (written by pass 1) now equals
        # `current`, same (clone_path, kind) key persists — it fires again,
        # unlike every other HIGH kind (see test_finding_unchanged_since_prior_pass_does_not_notify).
        with (
            patch("lapis_pm.pm_core._mem", return_value=fake_mem),
            patch("lapis_pm.deploy_inventory.run_reconcile_pass", return_value=current),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            pm_core._reconcile_deploy_inventory()

        assert notify_calls == [
            "deploy-inventory: auto_recovery_restart_failed",
            "deploy-inventory: auto_recovery_restart_failed",
        ]


class TestAgentsCoreDeployBackstopTimer:
    """§C (agents-core-deploy-drift-backstop-v0): a standing ~10-min timer that
    reconciles /data/agents (+ /srv/git/agents-core-working) independent of any
    land event, mirroring lapis-pm-deploy.{service,timer} (AC5)."""

    _SYSTEMD_DIR = Path(__file__).parent.parent / "systemd"

    def test_timer_and_service_files_exist(self):
        assert (self._SYSTEMD_DIR / "agents-core-deploy.timer").is_file()
        assert (self._SYSTEMD_DIR / "agents-core-deploy.service").is_file()

    def test_timer_fires_on_a_ten_minute_cadence_independent_of_land_events(self):
        timer_text = (self._SYSTEMD_DIR / "agents-core-deploy.timer").read_text()
        assert "OnUnitActiveSec=10min" in timer_text
        assert "Unit=agents-core-deploy.service" in timer_text

    def test_service_invokes_the_robust_pull_for_agents_core(self):
        service_text = (self._SYSTEMD_DIR / "agents-core-deploy.service").read_text()
        assert "_post_land_git_pull" in service_text
        assert "'agents-core'" in service_text
        # Runs from the lapis-pm deploy clone (kept current by its own PR #116
        # backstop timer), not -working — so this detector can't be silenced by
        # agents-core's own drift.
        assert "WorkingDirectory=/srv/git/lapis-pm" in service_text
        assert "/srv/lapis/lapis-pm" not in service_text


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
            patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure", return_value=False),
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
            patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure", return_value=False),
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
            patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure", return_value=False),
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


class TestGradedCollisionSafety:
    """Tests for the graded collision-safety model in _post_land_git_pull
    (agents-core-deploy-drift-backstop-v0 §B, AC3/AC4). Exercises the real git
    binary against throwaway repo pairs — the load-bearing behavior is git's own
    untracked-collision/divergence detection, same rationale as
    TestPostLandGitPullDirtyTreeAndDivergence. Every test maps the throwaway
    clone under repo="agents-core" (already in _POST_LAND_PULL_CRITICAL) so the
    collision-handling branch is live without touching that frozenset.
    """

    @staticmethod
    def _make_origin_and_clone(tmp_path):
        origin = tmp_path / "origin"
        origin.mkdir()
        subprocess.run(["git", "init", "-b", "main", str(origin)], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=origin, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=origin, check=True, capture_output=True)
        (origin / "f.txt").write_text("v1\n")
        subprocess.run(["git", "add", "."], cwd=origin, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=origin, check=True, capture_output=True)

        clone = tmp_path / "clone"
        subprocess.run(["git", "clone", str(origin), str(clone)], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=clone, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=clone, check=True, capture_output=True)
        return origin, clone

    def test_debris_outside_sacred_zone_quarantined_and_pull_completes(self, tmp_path):
        """AC3(a): an untracked collision OUTSIDE every sacred zone is quarantined
        (moved aside) and the ff-pull then completes."""
        origin, clone = self._make_origin_and_clone(tmp_path)
        (origin / "junk.txt").write_text("from upstream\n")
        subprocess.run(["git", "add", "."], cwd=origin, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add junk.txt"], cwd=origin, check=True, capture_output=True)
        (clone / "junk.txt").write_text("local untracked debris\n")

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"title": title, "priority": priority})
            return True

        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"agents-core": [str(clone)]}),
            patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"),
            patch.object(pm_core, "_DEPLOY_PULL_LOCK_DIR", tmp_path / "lock"),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            advanced = pm_core._post_land_git_pull("agents-core")

        assert advanced is True, "pull should self-recover once debris is quarantined"
        assert (clone / "junk.txt").read_text() == "from upstream\n"
        quarantined = list((clone / ".deploy-quarantine").rglob("junk.txt"))
        assert len(quarantined) == 1
        assert quarantined[0].read_text() == "local untracked debris\n"
        assert notify_calls == [], "a silently self-healed collision must not alert"

    def test_sacred_junk_quarantined(self, tmp_path):
        """AC3(c): non-runtime-pattern junk inside a sacred zone (e.g. `0`) is
        quarantined like any other debris."""
        origin, clone = self._make_origin_and_clone(tmp_path)
        (origin / "config").mkdir()
        (origin / "config" / "0").write_text("from upstream\n")
        subprocess.run(["git", "add", "."], cwd=origin, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add config/0"], cwd=origin, check=True, capture_output=True)
        (clone / "config").mkdir()
        (clone / "config" / "0").write_text("stray junk\n")

        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"agents-core": [str(clone)]}),
            patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"),
            patch.object(pm_core, "_DEPLOY_PULL_LOCK_DIR", tmp_path / "lock"),
        ):
            advanced = pm_core._post_land_git_pull("agents-core")

        assert advanced is True
        assert (clone / "config" / "0").read_text() == "from upstream\n"
        assert list((clone / ".deploy-quarantine").rglob("0"))

    def test_sacred_runtime_collision_halts_and_alerts(self, tmp_path):
        """AC3(b): an untracked collision INSIDE a sacred zone matching a
        RUNTIME_PATTERNS glob halts the pull and fires a CRITICAL (HIGH-priority)
        alert naming the file — the file itself is NOT moved."""
        origin, clone = self._make_origin_and_clone(tmp_path)
        (origin / "data").mkdir()
        (origin / "data" / "claude_queue.db").write_text("upstream schema\n")
        subprocess.run(["git", "add", "."], cwd=origin, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add data/claude_queue.db"], cwd=origin, check=True, capture_output=True
        )
        (clone / "data").mkdir()
        (clone / "data" / "claude_queue.db").write_text("live runtime queue state\n")

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "title": title, "priority": priority})
            return True

        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"agents-core": [str(clone)]}),
            patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"),
            patch.object(pm_core, "_DEPLOY_PULL_LOCK_DIR", tmp_path / "lock"),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            advanced = pm_core._post_land_git_pull("agents-core")

        assert advanced is False
        # Sacred runtime state must never be moved.
        assert (clone / "data" / "claude_queue.db").read_text() == "live runtime queue state\n"
        assert not (clone / ".deploy-quarantine").exists()
        assert len(notify_calls) == 1
        from agents_core.notify import Priority
        assert notify_calls[0]["priority"] == Priority.HIGH
        assert "claude_queue.db" in notify_calls[0]["message"]
        assert "halted" in notify_calls[0]["title"]
        # No divergence lock — this is a collision halt, not a lineage divergence.
        assert pm_core._deploy_pull_locked(str(clone)) is None

    def test_genuine_divergence_locks_alerts_and_does_not_loop(self, tmp_path):
        """AC4: real lineage divergence (not an untracked collision) writes a lock
        sentinel and fires a CRITICAL (HIGH) alert with the commit-distance — and
        a subsequent tick must not re-attempt the pull (no loop, no reset --hard)."""
        origin, clone = self._make_origin_and_clone(tmp_path)
        (origin / "f.txt").write_text("v2-on-origin\n")
        subprocess.run(["git", "add", "."], cwd=origin, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "origin-advances"], cwd=origin, check=True, capture_output=True)
        (clone / "g.txt").write_text("local-only commit\n")
        subprocess.run(["git", "add", "."], cwd=clone, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "clone-diverges"], cwd=clone, check=True, capture_output=True)

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append({"message": message, "title": title, "priority": priority})
            return True

        lock_dir = tmp_path / "lock"
        with (
            patch.object(pm_core, "_POST_LAND_PULL", {"agents-core": [str(clone)]}),
            patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"),
            patch.object(pm_core, "_DEPLOY_PULL_LOCK_DIR", lock_dir),
            patch("agents_core.notify.send_notification", fake_notify),
            ):
            advanced = pm_core._post_land_git_pull("agents-core")

            assert advanced is False
            lock = pm_core._deploy_pull_locked(str(clone))
            assert lock is not None
            assert lock["commits_behind"] == 1

            assert len(notify_calls) == 1
            from agents_core.notify import Priority
            assert notify_calls[0]["priority"] == Priority.HIGH
            assert "diverged" in notify_calls[0]["title"] or "divergence" in notify_calls[0]["message"]

            # Second tick: locked clone must be skipped entirely — no subprocess
            # call at all for it (no loop, no reset --hard).
            with patch("lapis_pm.pm_core.subprocess.run") as spy_run:
                advanced_again = pm_core._post_land_git_pull("agents-core")
            spy_run.assert_not_called()
            assert advanced_again is False
            assert len(notify_calls) == 1, "locked clone must not re-alert on every tick"

        # Manual-recovery escape hatch clears the lock.
        pm_core._clear_deploy_pull_lock(str(clone))
        assert pm_core._deploy_pull_locked(str(clone)) is None


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


class TestReconcileSurvivingHeadBranch:
    """Tests for _reconcile_surviving_head_branch (Leg 1, AC1.1-AC1.6).

    A merge performed outside merge_and_deploy (e.g. a Forgejo web-UI merge
    with the delete-branch box unchecked) leaves the head branch alive, which
    permanently blocks _is_auto_land_eligible. This reconcile deletes it.
    """

    def _merged_comment(self, pr_num: int) -> MagicMock:
        c = MagicMock()
        c.tags = [f"pm:pr-merged:{pr_num}", f"pm:pr={pr_num}", "pm:observation"]
        c.content = f"PR #{pr_num} merged at 2026-08-02T00:00:00Z, created_at=, head_sha=abc"
        return c

    def _closed_not_merged_comment(self, pr_num: int) -> MagicMock:
        # Closed-not-merged PRs never get a pm:pr-merged tag written by
        # _encode_merged_prs, so this simulates the "seen but not merged" case.
        c = MagicMock()
        c.tags = [f"pm:pr={pr_num}", "pm:observation"]
        c.content = f"PR #{pr_num} observed"
        return c

    def test_ac1_1_surviving_branch_triggers_delete(self):
        """AC1.1: a merged PR with a still-resolving head branch gets deleted."""
        comments = [self._merged_comment(9)]
        delete_call = {}

        def fake_ensure_deleted(repo, pr_number, *, owner=None):
            delete_call["args"] = (repo, pr_number, owner)

        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
            patch("lapis_pm.pm_core._ensure_head_branch_deleted",
                  side_effect=fake_ensure_deleted) as mock_ensure,
        ):
            pm_core._reconcile_surviving_head_branch("my-target", "Erah/lapis-pm")

        mock_ensure.assert_called_once()
        assert delete_call["args"] == ("lapis-pm", 9, "Erah")

    def test_ac1_2_closed_not_merged_never_triggers(self):
        """AC1.2: a closed-but-not-merged PR (no pm:pr-merged tag) never reconciles."""
        comments = [self._closed_not_merged_comment(9)]
        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
            patch("lapis_pm.pm_core._ensure_head_branch_deleted") as mock_ensure,
        ):
            pm_core._reconcile_surviving_head_branch("my-target", "Erah/lapis-pm")
        mock_ensure.assert_not_called()

    def test_ac1_3_zero_calls_with_no_merged_observation(self):
        """AC1.3: no pm:pr-merged observation → zero Forgejo calls (early return)."""
        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]),
            patch("lapis_pm.pm_core._ensure_head_branch_deleted") as mock_ensure,
        ):
            pm_core._reconcile_surviving_head_branch("my-target", "Erah/lapis-pm")
        mock_ensure.assert_not_called()

    def test_ac1_3_zero_calls_once_branch_already_gone(self):
        """AC1.3: steady-state cost does not grow once the branch is confirmed gone.

        _ensure_head_branch_deleted itself is the sole probe surface (already
        idempotent/no-op on a 404) — the reconcile issues no calls beyond it and
        does not escalate call volume across repeated ticks.
        """
        comments = [self._merged_comment(9)]
        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
            patch("lapis_pm.pm_core._ensure_head_branch_deleted") as mock_ensure,
        ):
            pm_core._reconcile_surviving_head_branch("my-target", "Erah/lapis-pm")
            pm_core._reconcile_surviving_head_branch("my-target", "Erah/lapis-pm")
        # Exactly one call per invocation — no retry/escalation on repeat ticks.
        assert mock_ensure.call_count == 2

    def test_ac1_4_forgejo_failure_does_not_raise(self):
        """AC1.4: a Forgejo failure inside the reconcile never raises out of tick()."""
        comments = [self._merged_comment(9)]
        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
            patch("lapis_pm.pm_core._ensure_head_branch_deleted",
                  side_effect=RuntimeError("forgejo unreachable")),
            patch("lapis_pm.pm_core.episodic.write_observation") as mock_write_obs,
        ):
            pm_core._reconcile_surviving_head_branch("my-target", "Erah/lapis-pm")
            # Must not raise.
        mock_write_obs.assert_called_once()

    def test_ac1_5_idempotent_on_already_deleted_branch(self):
        """AC1.5: calling the reconcile when the branch is already gone is a no-op.

        Pins existing behaviour in _ensure_head_branch_deleted (404 from
        get_branch → return early, no DELETE) via the reconcile wrapper.
        """
        comments = [self._merged_comment(9)]

        def fake_get_pr(*args, **kwargs):
            return {"head": {"ref": "lapis/my-target/forced"}}

        def fake_get_branch(*args, **kwargs):
            exc = Exception("404 Not Found")
            raise exc

        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
            patch("lapis_pm.pm_core._forgejo_get_pr", side_effect=fake_get_pr),
            patch("lapis_pm.pm_core._forgejo_get_branch", side_effect=fake_get_branch),
            ):
            pm_core._reconcile_surviving_head_branch("my-target", "Erah/lapis-pm")
            # No exception, no DELETE attempt (verified by _ensure_head_branch_deleted's
            # own contract, exercised here end-to-end through the reconcile wrapper).

    def test_ac1_6_failure_writes_pm_error_observation(self):
        """AC1.6: a reconcile failure writes a pm:error-tagged episodic observation."""
        comments = [self._merged_comment(9)]
        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
            patch("lapis_pm.pm_core._ensure_head_branch_deleted",
                  side_effect=RuntimeError("forgejo unreachable")),
            patch("lapis_pm.pm_core.episodic.write_observation") as mock_write_obs,
        ):
            pm_core._reconcile_surviving_head_branch("my-target", "Erah/lapis-pm")

        mock_write_obs.assert_called_once()
        args, kwargs = mock_write_obs.call_args
        assert args[0] == "my-target"
        extra_tags = kwargs.get("extra_tags", [])
        assert "pm:error" in extra_tags

    def test_no_repo_is_noop(self):
        """Empty repo → early return, no episodic or Forgejo calls."""
        with (
            patch("lapis_pm.pm_core.episodic.all_comments") as mock_all_comments,
            patch("lapis_pm.pm_core._ensure_head_branch_deleted") as mock_ensure,
        ):
            pm_core._reconcile_surviving_head_branch("my-target", "")
        mock_all_comments.assert_not_called()
        mock_ensure.assert_not_called()


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

    def test_manifest_includes_gw_phase_models(self):
        """lapis-pm-conductor-brix-gw-runtime-deploy-manifest-v0: gw_phase_models.py
        is a true night_plan.py closure member (a top-level import since 2026-08-01,
        c35e8f2) — it belongs in _CONDUCTOR_NIGHT_SCRIPTS itself, NOT in the new
        sibling _CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS manifest."""
        assert "gw_phase_models.py" in pm_core._CONDUCTOR_NIGHT_SCRIPTS
        assert "gw_phase_models.py" not in pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS

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
        expected_files = {f"{mod}.py" for mod in closure} | {
            "night_producers.yaml", "night-task-menu.yaml",
        }
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
            "research_headings", "gw_phase_models", "gw_seat_lane",
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
        expected_files = {f"{mod}.py" for mod in closure} | {
            "night_producers.yaml", "night-task-menu.yaml",
        }
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
            # host-script-deploy-on-land-v0: the full hook now also fires the GW
            # host-script install for repo=="conductor" — hermetic isolation for
            # its own drift ledger, exactly as _DEPLOY_LOG is isolated above.
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-drift-ledger.json"),
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
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-drift-ledger.json"),
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


class TestConductorBrixGwRuntimeDeploy:
    """Tests for lapis-pm-conductor-brix-gw-runtime-deploy-manifest-v0.

    Adds the BRIX-side GW actuator family (doorman, flip-controller, gw_topology's
    own helper-hash reference copy, scout-night-pre) to the post-land local-copy
    machinery, as a manifest sibling to _CONDUCTOR_NIGHT_SCRIPTS delivered by the
    SAME _deploy_conductor_night_scripts copy pass (see that function's docstring
    for why this is one combined pass, not a second independent call: R10
    single-alert-ownership). The manifest itself stays separate because this
    family sits outside night_plan.py's producer-import closure, so it cannot be
    a naive append to _CONDUCTOR_NIGHT_SCRIPTS (see TestConductorNightPlanDeploy's
    exact-equality tests above).
    """

    # --- manifest membership ---

    @pytest.mark.parametrize("fname", [
        "gw_topology.py", "gw_actuator.py", "gw_host_safety.py", "flip_controller.py",
        "gw-topology", "gw-night-pre.py", "gw-night-post.py", "scout-night-pre.py",
        # night-deploy-manifest-attestation-v0: the three DAG-command scripts that
        # were unmanifested (the measured 2026-09-07 hole).
        "mini_1f916_night.py", "keeper_v0.py", "council_sweep.py",
    ])
    def test_manifest_includes_gw_runtime_family_member(self, fname):
        assert fname in pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS

    def test_manifest_is_exactly_the_eleven_files(self):
        """No more, no fewer — the deliberately-out-of-scope files
        (gw_topology_cache.py, gw_enforce_rearm_ab.py, gw-serve, gw-dual) must not
        be present. night-deploy-manifest-attestation-v0 adds the three
        DAG-command scripts (mini_1f916_night.py, keeper_v0.py, council_sweep.py)
        that were unmanifested (the measured 2026-09-07 hole)."""
        assert set(pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS) == {
            "gw_topology.py", "gw_actuator.py", "gw_host_safety.py",
            "flip_controller.py", "gw-topology", "gw-night-pre.py",
            "gw-night-post.py", "scout-night-pre.py",
            "mini_1f916_night.py", "keeper_v0.py", "council_sweep.py",
        }

    def test_manifest_disjoint_from_night_scripts(self):
        """The two manifests must not overlap — each file has exactly one owner."""
        assert not (
            set(pm_core._CONDUCTOR_NIGHT_SCRIPTS) & set(pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS)
        )

    # --- real-filesystem copy: delivery + mode preservation ---

    def test_real_fs_copy_delivers_all_files_with_modes_preserved(self, tmp_path):
        """DoD (scope §3): a real-filesystem copy test confirming the new family
        lands with modes preserved — copystat preserves the repo mode exactly as
        the night-plan family's own copy mechanics already do. Driven through the
        actual production entry point (_deploy_conductor_night_scripts), which now
        carries both manifests in one pass."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        log_file = tmp_path / "deploy-log.md"
        src_dir.mkdir()
        dest_dir.mkdir()

        contents = {}
        modes = {}
        for i, fname in enumerate(pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS):
            content = f"# {fname} v1\n"
            contents[fname] = content
            mode = 0o755 if i % 2 == 0 else 0o644
            modes[fname] = mode
            path = src_dir / fname
            path.write_text(content)
            path.chmod(mode)

        fake_run = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            ):
            pm_core._deploy_conductor_night_scripts(trigger="test")

        log_text = log_file.read_text()
        for fname in pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS:
            dest_path = dest_dir / fname
            assert dest_path.read_text() == contents[fname]
            assert stat.S_IMODE(dest_path.stat().st_mode) == modes[fname], (
                f"{fname} mode not preserved by copystat"
            )
            assert f"conductor:scripts/{fname}" in log_text
        assert "created" in log_text

    def test_fetch_fail_skips_copy_of_both_manifests_and_alerts_once(self, tmp_path):
        """R10 single-alert-ownership, now covering both manifests: a fetch
        failure must skip BOTH families' copies (never deliver stale content) and
        alert exactly once — not once per manifest."""
        from agents_core.notify import Priority

        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('v1')")
        (src_dir / "gw_topology.py").write_text("print('v1')")

        notify_calls = []

        def fake_notify(message, title, priority, **kwargs):
            notify_calls.append(priority)
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
        assert len(notify_calls) == 1, f"expected exactly one alert, got {notify_calls}"
        assert notify_calls[0] == Priority.NORMAL

    def test_fired_from_post_land_hook_alongside_gw_host_install(self, tmp_path):
        """Scope §2: fired from _post_land_deploy_hook for repo=='conductor'
        alongside the existing GW-host-script-install call — both local-copy
        manifests land in the same pass (DoD 3's 'same pass' requirement), and the
        cross-host GW-host install still runs independently."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        log_file = tmp_path / "deploy-log.md"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('night')")
        (src_dir / "gw_topology.py").write_text("print('topology')")
        (src_dir / "gw-topology").write_text("#!/bin/bash\necho hi\n")

        fake_run = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        with (
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            # host-script-deploy-on-land-v0's cross-host install also fires for
            # repo=="conductor" — hermetic isolation for its own drift ledger,
            # exactly as _DEPLOY_LOG is isolated above.
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-drift-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run),
            ):
            pm_core._post_land_deploy_hook("conductor")

        assert (dest_dir / "night_plan.py").read_text() == "print('night')"
        assert (dest_dir / "gw_topology.py").read_text() == "print('topology')"
        assert (dest_dir / "gw-topology").read_text() == "#!/bin/bash\necho hi\n"


# ---------------------------------------------------------------------------
# host-script-deploy-on-land-v0: GravityWell host-script install
# ---------------------------------------------------------------------------

def _git(cwd, *args):
    result = subprocess.run(["git"] + list(args), cwd=str(cwd), capture_output=True, text=True)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


def _git_out(cwd, *args) -> str:
    return _git(cwd, *args).stdout.strip()


def _init_real_conductor_clone(clone: Path) -> None:
    """A real (not mocked) tiny git repo shaped like the conductor deploy clone,
    for the D3/D4 tests that must run against real git behavior rather than a
    fixture (spec DoD 3/4)."""
    clone.mkdir(parents=True, exist_ok=True)
    _git(clone, "init", "-q")
    _git(clone, "config", "user.email", "test@test")
    _git(clone, "config", "user.name", "test")


def _real_git_with_stubbed_fetch(real_run):
    """Passes every git call through to the REAL subprocess.run against the tmp
    repo except `git fetch origin main`, which is stubbed to a clean success (no
    real 'origin' remote exists on the tmp repo, and no network is available in
    the test sandbox)."""
    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "git" and len(cmd) > 3 and cmd[3] == "fetch":
            return _make_completed_process(returncode=0)
        return real_run(cmd, **kwargs)
    return fake_run


def _gw_fake_run(
    *, git_fake=None, reachable=True, pre_hash=None, post_hash=None,
    scp_rc=0, install_rc=0, self_hash_line=None, calls=None,
):
    """Fake subprocess.run dispatcher for the GW host-script install calls
    (reachability probe, sha256sum pre/post-install, scp, sudo install, --help
    self-hash, rm cleanup) — delegates `git` calls to `git_fake` (a
    `_conductor_git_fake_run`-shaped callable) so the shared freshness gate
    behaves exactly as it does for the local night-script copy (D2 — gate
    sharing).

    pre_hash / post_hash / self_hash_line may be a plain string (applies to
    every dest) or a {dest: value} dict (per-entry, for mixed-pass tests).
    `calls` — if given, every dispatched cmd is appended to it (call log for
    assertions like "no scp/install happened").
    """
    sha_call_counts: dict = {}

    def _resolve(value, dest):
        if isinstance(value, dict):
            return value.get(dest)
        return value

    def fake_run(cmd, **kwargs):
        if calls is not None:
            calls.append(list(cmd))
        if cmd and cmd[0] == "git":
            if git_fake is not None:
                return git_fake(cmd, **kwargs)
            return _make_completed_process(returncode=0)
        if cmd and cmd[0] == "ssh" and len(cmd) > 1 and cmd[1] == "-o":
            return _make_completed_process(returncode=0 if reachable else 1)
        if cmd and cmd[0] == "scp":
            return _make_completed_process(
                returncode=scp_rc, stderr="" if scp_rc == 0 else "scp failed",
            )
        if cmd and cmd[0] == "ssh":
            remote = cmd[2:]
            if remote[:1] == ["sha256sum"]:
                dest = remote[1]
                n = sha_call_counts.get(dest, 0)
                sha_call_counts[dest] = n + 1
                value = _resolve(pre_hash, dest) if n == 0 else _resolve(post_hash, dest)
                if value is None:
                    return _make_completed_process(returncode=1)
                return _make_completed_process(returncode=0, stdout=f"{value}  {dest}\n")
            if remote[:1] == ["rm"]:
                return _make_completed_process(returncode=0)
            if remote[:3] == ["sudo", "-n", "install"]:
                return _make_completed_process(
                    returncode=install_rc, stderr="" if install_rc == 0 else "install failed",
                )
            if len(remote) >= 2 and remote[-1] == "--help":
                dest = remote[0]
                value = _resolve(self_hash_line, dest)
                return _make_completed_process(returncode=0, stdout=value or "")
        return _make_completed_process(returncode=0)

    return fake_run


class TestGwHostScriptDeploy:
    """Tests for host-script-deploy-on-land-v0 — the cross-host sibling of
    _deploy_conductor_night_scripts that installs conductor's repo-held
    GravityWell scripts (gw-topology, gw-idle-suspend.sh) onto
    gravitywell:/usr/local/sbin/.
    """

    # --- D1: manifest ---

    def test_manifest_has_both_scripts_correctly_shaped(self):
        by_dest = {e.dest: e for e in pm_core._CONDUCTOR_GW_HOST_SCRIPTS}
        assert "/usr/local/sbin/gw-topology" in by_dest
        assert "/usr/local/sbin/gw-idle-suspend.sh" in by_dest
        assert "/usr/local/sbin/gw_resume_grace.py" in by_dest
        assert "/usr/local/sbin/gw-gpu1-lease.sh" in by_dest
        assert "/usr/local/bin/gw-serve" in by_dest

        topology = by_dest["/usr/local/sbin/gw-topology"]
        assert topology.src_name == "gw-topology"
        assert topology.mode == 0o755
        assert topology.owner == "root:root"
        assert topology.self_attesting is True
        assert topology.execution_sensitive is True

        idle = by_dest["/usr/local/sbin/gw-idle-suspend.sh"]
        assert idle.src_name == "gw-idle-suspend.sh"
        assert idle.self_attesting is False
        assert idle.execution_sensitive is True

        # gw_resume_grace.py (lapis-pm-gw-host-script-manifest-add-resume-grace-v0):
        # gw-idle-suspend.sh's guard-rule-0b companion, delivered but not
        # execution_sensitive — see spec Scope for the execution_sensitive=False
        # rationale.
        grace = by_dest["/usr/local/sbin/gw_resume_grace.py"]
        assert grace.src_name == "gw_resume_grace.py"
        assert grace.mode == 0o755
        assert grace.owner == "root:root"
        assert grace.self_attesting is False
        assert grace.execution_sensitive is False

        # gw-gpu1-lease.sh (gpu1-multi-tenant-lease arbitration landing
        # 2026-08-22/23): the tenancy fast-follow debt — hand-installed on GW
        # with the manifest entry never landed; manifested here to make
        # subsequent conductor deploys deliver it durably.
        lease = by_dest["/usr/local/sbin/gw-gpu1-lease.sh"]
        assert lease.src_name == "gw-gpu1-lease.sh"
        assert lease.mode == 0o755
        assert lease.owner == "root:root"
        assert lease.self_attesting is False
        assert lease.execution_sensitive is False

        # gw-serve (gw-gpu1-berth-standing-seat-v0 leg 3): gained a repo
        # source (conductor scripts/gw-serve, bootstrapped + GPU 1 stop-path
        # extension) via leg 1 (conductor PR #896). Dest is /usr/local/bin
        # (arbitrary per entry); execution_sensitive=True — it flips serving
        # state, so the D5 in-flight-flip skip applies.
        serve = by_dest["/usr/local/bin/gw-serve"]
        assert serve.src_name == "gw-serve"
        assert serve.mode == 0o755
        assert serve.owner == "root:root"
        assert serve.self_attesting is False
        assert serve.execution_sensitive is True

    def test_gw_dual_not_manifested(self):
        """Out-of-scope: gw-dual has no repo source, so no manifest entry
        (spec Out-of-scope). gw-serve DID gain a repo source via
        gw-gpu1-berth-standing-seat-v0 leg 1 and is asserted in
        test_manifest_has_both_scripts_correctly_shaped above."""
        dests = {e.dest for e in pm_core._CONDUCTOR_GW_HOST_SCRIPTS}
        assert "/usr/local/sbin/gw-dual" not in dests

    # --- D2: gate sharing ---

    def test_dirty_source_skips_both_local_copy_and_gw_install(self, tmp_path):
        """One gate, two consumers (D2): a dirty deploy clone must skip BOTH the
        local night-script copy and the GravityWell host-script install
        identically — never re-implemented, never diverging."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        (src_dir / "night_plan.py").write_text("print('v1')")
        (src_dir / "gw-topology").write_text("#!/bin/sh\necho hi\n")
        (src_dir / "gw-idle-suspend.sh").write_text("#!/bin/sh\necho hi\n")

        git_fake = _conductor_git_fake_run(
            fetch_rc=0, head_sha="aaa", main_sha="aaa",
            status_stdout="M scripts/night_plan.py\n",
        )
        gw_fake = _gw_fake_run(git_fake=git_fake)

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", str(dest_dir)),
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=gw_fake),
            ):
            pm_core._deploy_conductor_night_scripts(trigger="test")
            pm_core._deploy_conductor_gw_host_scripts(trigger="test")

        assert list(dest_dir.iterdir()) == [], "local copy must be skipped on dirty source"
        # No install artifacts possible to observe directly (remote), but the
        # gate must have short-circuited before any scp/install call fired.

    # --- D3: untracked-vs-tracked dirt, verified against a real clone ---

    def test_untracked_editor_dirs_no_longer_block_freshness_gate(self, tmp_path):
        """DoD 3: git status --porcelain --untracked-files=no is in use, verified
        against a REAL git clone (not a fixture) carrying the exact live finding
        — untracked .claude/ + .vscode/, neither gitignored — which must no
        longer skip delivery."""
        clone = tmp_path / "clone"
        _init_real_conductor_clone(clone)
        (clone / "scripts").mkdir()
        (clone / "scripts" / "gw-topology").write_text("#!/bin/sh\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-q", "-m", "init")
        head_sha = _git_out(clone, "rev-parse", "HEAD")
        _git(clone, "update-ref", "refs/remotes/origin/main", head_sha)

        # the exact live finding (finding/lapis-pm-conductor-deploy-gate-blocked-
        # by-untracked-dirs-2026-08-11): untracked .claude/ + .vscode/ present.
        (clone / ".claude").mkdir()
        (clone / ".claude" / "settings.json").write_text("{}")
        (clone / ".vscode").mkdir()
        (clone / ".vscode" / "settings.json").write_text("{}")

        with patch(
            "lapis_pm.pm_core.subprocess.run",
            side_effect=_real_git_with_stubbed_fetch(subprocess.run),
            ):
            is_fresh, reason = pm_core._conductor_source_freshness_gate(str(clone))

        assert is_fresh is True, f"untracked .claude/.vscode must not block freshness: {reason}"

    def test_tracked_dirt_still_blocks_freshness_gate(self, tmp_path):
        """The other half of D3: a TRACKED modification must still block — the
        looser untracked check must not become no check at all."""
        clone = tmp_path / "clone"
        _init_real_conductor_clone(clone)
        (clone / "scripts").mkdir()
        tracked = clone / "scripts" / "gw-topology"
        tracked.write_text("#!/bin/sh\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-q", "-m", "init")
        head_sha = _git_out(clone, "rev-parse", "HEAD")
        _git(clone, "update-ref", "refs/remotes/origin/main", head_sha)

        tracked.write_text("#!/bin/sh\necho modified\n")  # tracked, uncommitted

        with patch(
            "lapis_pm.pm_core.subprocess.run",
            side_effect=_real_git_with_stubbed_fetch(subprocess.run),
            ):
            is_fresh, reason = pm_core._conductor_source_freshness_gate(str(clone))

        assert is_fresh is False
        assert "clean-on-main" in reason

    # --- per-file tracked check ---

    def test_untracked_manifested_file_skipped_even_when_tree_clean(self, tmp_path):
        """An untracked-but-manifested file must be skipped even though the
        tree-level check (which ignores untracked files) reads clean."""
        git_fake = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")

        def ls_files_untracked(cmd, **kwargs):
            if cmd[3] == "ls-files":
                return _make_completed_process(returncode=1)  # not tracked
            return git_fake(cmd, **kwargs)

        with patch("lapis_pm.pm_core.subprocess.run", side_effect=ls_files_untracked):
            clean = pm_core._conductor_file_verified_clean("/fake/clone", "scripts/gw-topology")

        assert clean is False

    def test_per_file_check_baseline_is_head_not_stale_origin_main(self, tmp_path):
        """DoD 4: the baseline is HEAD, not origin/main. With HEAD ahead of a
        stale local origin/main ref, a file matching HEAD is delivered and a
        file matching only the stale ref is not."""
        clone = tmp_path / "clone"
        _init_real_conductor_clone(clone)
        (clone / "scripts").mkdir()
        f = clone / "scripts" / "gw-idle-suspend.sh"

        f.write_text("old content\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-q", "-m", "old")
        old_sha = _git_out(clone, "rev-parse", "HEAD")
        # A stale local origin/main ref — simulates being mid-land, before the
        # next fetch updates it.
        _git(clone, "update-ref", "refs/remotes/origin/main", old_sha)

        # HEAD advances (the ff-only pull landing a new merge commit).
        f.write_text("new content\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-q", "-m", "new")

        # Working tree matches HEAD -> verified clean -> would be delivered.
        assert pm_core._conductor_file_verified_clean(str(clone), "scripts/gw-idle-suspend.sh") is True

        # Working tree reverted to match ONLY the stale origin/main content ->
        # must NOT verify clean — proves the baseline is HEAD, not origin/main.
        f.write_text("old content\n")
        assert pm_core._conductor_file_verified_clean(str(clone), "scripts/gw-idle-suspend.sh") is False

    # --- D4: unreachable host ---

    def test_unreachable_host_skips_all_no_wake_no_merge_failure(self, tmp_path):
        """DoD 5: an unreachable GravityWell produces a clean LOW skip — no wake
        attempt, no scp/install call, and the function never raises."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "gw-topology").write_text("#!/bin/sh\n")
        (src_dir / "gw-idle-suspend.sh").write_text("#!/bin/sh\n")

        git_fake = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")
        calls: list = []
        gw_fake = _gw_fake_run(git_fake=git_fake, reachable=False, calls=calls)

        with (
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=gw_fake),
            ):
            pm_core._deploy_conductor_gw_host_scripts(trigger="test")  # must not raise

        ssh_calls = [c for c in calls if c and c[0] in ("ssh", "scp")]
        assert len(ssh_calls) == 1, f"expected only the reachability probe, got {ssh_calls}"
        assert ssh_calls[0][:2] == ["ssh", "-o"], "the one call must be the probe, not scp/install"
        for c in calls:
            joined = " ".join(str(x) for x in c)
            assert "wake" not in joined.lower() and "wol" not in joined.lower()

    # --- D5: in_flight_flip, execution_sensitive per-entry ---

    def test_in_flight_flip_skips_execution_sensitive_only_mixed_pass(self, tmp_path):
        """DoD 6 + DoD 9: in_flight_flip=True skips ONLY manifest entries marked
        execution_sensitive; a non-sensitive entry still installs. The resulting
        mixed pass (one delivered, one skipped) must report BOTH sets — a
        partial delivery must never read as a clean state."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "sensitive.sh").write_text("#!/bin/sh\nsensitive\n")
        (src_dir / "not-sensitive.sh").write_text("#!/bin/sh\nnot sensitive\n")

        sensitive_hash = hashlib.sha256((src_dir / "sensitive.sh").read_bytes()).hexdigest()
        other_hash = hashlib.sha256((src_dir / "not-sensitive.sh").read_bytes()).hexdigest()

        manifest = (
            pm_core._GwHostScript("sensitive.sh", "/usr/local/sbin/sensitive.sh", 0o755, "root:root", False, True),
            pm_core._GwHostScript("not-sensitive.sh", "/usr/local/sbin/not-sensitive.sh", 0o755, "root:root", False, False),
        )

        git_fake = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")
        gw_fake = _gw_fake_run(
            git_fake=git_fake,
            pre_hash={"/usr/local/sbin/sensitive.sh": "stale", "/usr/local/sbin/not-sensitive.sh": "stale"},
            post_hash={"/usr/local/sbin/not-sensitive.sh": other_hash},
        )

        captured_payload = {}

        def fake_deposit_gem(payload):
            captured_payload.update(payload)
            return "gem-123"

        with (
            patch.object(pm_core, "_CONDUCTOR_GW_HOST_SCRIPTS", manifest),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", tmp_path / "deploy-log.md"),
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=gw_fake),
            patch("agents_core.llm.gw_serving_state") as mock_state,
            patch("lapis_pm.deploy_inventory_repair.deposit_gem", side_effect=fake_deposit_gem),
            ):
            mock_state.return_value = MagicMock(in_flight_flip=True)
            pm_core._deploy_conductor_gw_host_scripts(trigger="test")

        assert "sensitive.sh: in_flight_flip" in captured_payload.get("ask", ""), captured_payload
        undelivered_lines = next(
            c["lines"] for c in captured_payload.get("context", []) if c["label"] == "Undelivered"
        )
        delivered_lines = next(
            c["lines"] for c in captured_payload.get("context", []) if c["label"] == "Delivered this pass"
        )
        assert any("sensitive.sh" in ln and "in_flight_flip" in ln for ln in undelivered_lines)
        assert any("not-sensitive.sh" in ln for ln in delivered_lines)

    # --- D6: verify mismatch ---

    def test_post_install_hash_mismatch_is_not_recorded_as_delivered(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "gw-idle-suspend.sh").write_text("#!/bin/sh\n")

        manifest = (
            pm_core._GwHostScript(
                "gw-idle-suspend.sh", "/usr/local/sbin/gw-idle-suspend.sh", 0o755, "root:root", False, True,
            ),
        )
        git_fake = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")
        log_file = tmp_path / "deploy-log.md"
        gw_fake = _gw_fake_run(
            git_fake=git_fake, pre_hash="stale", post_hash="wrong-hash-after-install",
        )

        with (
            patch.object(pm_core, "_CONDUCTOR_GW_HOST_SCRIPTS", manifest),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=gw_fake),
            patch("agents_core.llm.gw_serving_state") as mock_state,
        ):
            mock_state.return_value = MagicMock(in_flight_flip=False)
            pm_core._deploy_conductor_gw_host_scripts(trigger="test")

        assert not log_file.exists(), "a verify mismatch must never write a delivered provenance line"

    def test_self_hash_mismatch_is_not_recorded_as_delivered(self, tmp_path):
        """The --help self-hash verify (self-attesting entries only): a mismatch
        must be reported failed, even though the plain re-hash matched."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        content = "#!/bin/sh\necho self-attest\n"
        (src_dir / "gw-topology").write_text(content)
        local_hash = hashlib.sha256(content.encode()).hexdigest()

        manifest = (
            pm_core._GwHostScript("gw-topology", "/usr/local/sbin/gw-topology", 0o755, "root:root", True, True),
        )
        git_fake = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")
        log_file = tmp_path / "deploy-log.md"
        gw_fake = _gw_fake_run(
            git_fake=git_fake, pre_hash="stale", post_hash=local_hash,
            self_hash_line="self-hash: 000000000000\n",  # deliberately wrong
        )

        with (
            patch.object(pm_core, "_CONDUCTOR_GW_HOST_SCRIPTS", manifest),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=gw_fake),
            patch("agents_core.llm.gw_serving_state") as mock_state,
        ):
            mock_state.return_value = MagicMock(in_flight_flip=False)
            pm_core._deploy_conductor_gw_host_scripts(trigger="test")

        assert not log_file.exists(), "a self-hash mismatch must never write a delivered provenance line"

    def test_self_hash_verify_passes_and_records_delivery(self, tmp_path):
        """The green-path counterpart: matching re-hash AND matching self-hash
        records a delivered provenance line — a green verify here is exactly
        what stops _assert_helper_compatible refusing the next reach()."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        content = "#!/bin/sh\necho self-attest\n"
        (src_dir / "gw-topology").write_text(content)
        local_hash = hashlib.sha256(content.encode()).hexdigest()

        manifest = (
            pm_core._GwHostScript("gw-topology", "/usr/local/sbin/gw-topology", 0o755, "root:root", True, True),
        )
        git_fake = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")
        log_file = tmp_path / "deploy-log.md"
        gw_fake = _gw_fake_run(
            git_fake=git_fake, pre_hash="stale", post_hash=local_hash,
            self_hash_line=f"self-hash: {local_hash[:12]}\n",
        )

        with (
            patch.object(pm_core, "_CONDUCTOR_GW_HOST_SCRIPTS", manifest),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=gw_fake),
            patch("agents_core.llm.gw_serving_state") as mock_state,
        ):
            mock_state.return_value = MagicMock(in_flight_flip=False)
            pm_core._deploy_conductor_gw_host_scripts(trigger="test")

        log_text = log_file.read_text()
        assert "gravitywell:/usr/local/sbin/gw-topology" in log_text
        assert local_hash[:8] in log_text

    def test_gw_resume_grace_full_install_path_mode_owner_atomic_verified(self, tmp_path):
        """lapis-pm-gw-host-script-manifest-add-resume-grace-v0 DoD 2: full
        install-path coverage for the new gw_resume_grace.py entry — mirrors
        the gw-topology / gw-idle-suspend.sh coverage above. Confirms delivery
        at mode 0o755, owner root:root, via the atomic scp-to-temp +
        `sudo install` path (never a direct overwrite of dest), with the
        post-install re-hash verify, and that self_attesting=False correctly
        skips the --help self-hash check (unlike gw-topology)."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        content = "#!/usr/bin/env python3\nprint('grace')\n"
        (src_dir / "gw_resume_grace.py").write_text(content)
        local_hash = hashlib.sha256(content.encode()).hexdigest()

        manifest = (
            pm_core._GwHostScript(
                "gw_resume_grace.py", "/usr/local/sbin/gw_resume_grace.py", 0o755, "root:root", False, False,
            ),
        )
        git_fake = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")
        log_file = tmp_path / "deploy-log.md"
        calls: list = []
        gw_fake = _gw_fake_run(
            git_fake=git_fake, pre_hash="stale", post_hash=local_hash, calls=calls,
        )

        with (
            patch.object(pm_core, "_CONDUCTOR_GW_HOST_SCRIPTS", manifest),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=gw_fake),
            patch("agents_core.llm.gw_serving_state") as mock_state,
        ):
            mock_state.return_value = MagicMock(in_flight_flip=False)
            pm_core._deploy_conductor_gw_host_scripts(trigger="test")

        log_text = log_file.read_text()
        assert "gravitywell:/usr/local/sbin/gw_resume_grace.py" in log_text
        assert local_hash[:8] in log_text

        scp_calls = [c for c in calls if c and c[0] == "scp"]
        assert len(scp_calls) == 1, f"expected exactly one scp call, got {scp_calls}"
        assert scp_calls[0][:2] == ["scp", "-p"]
        tmp_remote = scp_calls[0][3].split(":", 1)[1]
        assert tmp_remote.startswith("/tmp/"), "must scp to a temp path, never overwrite dest directly (atomic install)"

        install_calls = [
            c for c in calls
            if c and c[0] == "ssh" and len(c) > 4 and c[2:5] == ["sudo", "-n", "install"]
        ]
        assert len(install_calls) == 1, f"expected exactly one install call, got {install_calls}"
        install_cmd = install_calls[0][2:]
        assert install_cmd[:5] == ["sudo", "-n", "install", "-o", "root"], install_cmd
        assert install_cmd[5:7] == ["-g", "root"], install_cmd
        assert install_cmd[7:9] == ["-m", "755"], install_cmd
        assert install_cmd[-2] == tmp_remote, "install must move the scp'd temp path, not re-fetch"
        assert install_cmd[-1] == "/usr/local/sbin/gw_resume_grace.py"

        help_calls = [c for c in calls if c and c[0] == "ssh" and len(c) >= 4 and c[-1] == "--help"]
        assert help_calls == [], "self_attesting=False must skip the --help self-hash verify"

    # --- D7/D8: no-op path ---

    def test_noop_when_remote_hash_already_matches_no_reinstall(self, tmp_path):
        """DoD 8: a matching remote hash is a logged no-op — no scp, no install."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        content = "#!/bin/sh\n"
        (src_dir / "gw-idle-suspend.sh").write_text(content)
        local_hash = hashlib.sha256(content.encode()).hexdigest()

        manifest = (
            pm_core._GwHostScript(
                "gw-idle-suspend.sh", "/usr/local/sbin/gw-idle-suspend.sh", 0o755, "root:root", False, True,
            ),
        )
        git_fake = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")
        log_file = tmp_path / "deploy-log.md"
        calls: list = []
        gw_fake = _gw_fake_run(git_fake=git_fake, pre_hash=local_hash, calls=calls)

        with (
            patch.object(pm_core, "_CONDUCTOR_GW_HOST_SCRIPTS", manifest),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=gw_fake),
            patch("agents_core.llm.gw_serving_state") as mock_state,
        ):
            mock_state.return_value = MagicMock(in_flight_flip=False)
            pm_core._deploy_conductor_gw_host_scripts(trigger="test")

        scp_or_install_calls = [
            c for c in calls
            if (c and c[0] == "scp") or (c and c[0] == "ssh" and "install" in c)
        ]
        assert scp_or_install_calls == [], f"no-op must never scp/install: {scp_or_install_calls}"
        assert not log_file.exists(), "no-op must not write a provenance line"

    # --- fault isolation between manifest entries ---

    def test_fault_isolation_one_entry_error_does_not_block_the_other(self, tmp_path):
        """An unexpected exception installing one manifest entry must not prevent
        the other entry from being attempted (mirrors _copy_one_conductor_script's
        per-file isolation for the local copy)."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "broken.sh").write_text("#!/bin/sh\n")
        (src_dir / "fine.sh").write_text("#!/bin/sh\n")
        fine_hash = hashlib.sha256((src_dir / "fine.sh").read_bytes()).hexdigest()

        manifest = (
            pm_core._GwHostScript("broken.sh", "/usr/local/sbin/broken.sh", 0o755, "root:root", False, False),
            pm_core._GwHostScript("fine.sh", "/usr/local/sbin/fine.sh", 0o755, "root:root", False, False),
        )
        git_fake = _conductor_git_fake_run(fetch_rc=0, head_sha="aaa", main_sha="aaa")
        # ONE shared closure instance (not reconstructed per-call) so its internal
        # pre/post sha256sum call counter persists across dispatched commands.
        inner_fake = _gw_fake_run(
            git_fake=git_fake, pre_hash="stale", post_hash={"/usr/local/sbin/fine.sh": fine_hash},
        )

        def raising_gw_fake(cmd, **kwargs):
            if (
                cmd and cmd[0] == "ssh" and len(cmd) > 3
                and cmd[2:4] == ["sha256sum", "/usr/local/sbin/broken.sh"]
            ):
                raise RuntimeError("simulated transport corruption")
            return inner_fake(cmd, **kwargs)

        log_file = tmp_path / "deploy-log.md"

        with (
            patch.object(pm_core, "_CONDUCTOR_GW_HOST_SCRIPTS", manifest),
            patch.object(pm_core, "_CONDUCTOR_DEPLOY_CLONE", str(tmp_path / "clone")),
            patch.object(pm_core, "_CONDUCTOR_SCRIPTS_SRC", str(src_dir)),
            patch.object(pm_core, "_DEPLOY_LOG", log_file),
            patch.object(pm_core, "_GW_HOST_SCRIPT_DRIFT_LEDGER", tmp_path / "gw-ledger.json"),
            patch("lapis_pm.pm_core.subprocess.run", side_effect=raising_gw_fake),
            patch("agents_core.llm.gw_serving_state") as mock_state,
        ):
            mock_state.return_value = MagicMock(in_flight_flip=False)
            pm_core._deploy_conductor_gw_host_scripts(trigger="test")  # must not raise

        log_text = log_file.read_text()
        assert "gravitywell:/usr/local/sbin/fine.sh" in log_text, (
            "the entry after the failing one must still install"
        )
        assert "broken.sh" not in log_text
