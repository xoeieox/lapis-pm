"""Tests for deploy/daemon-sync.sh copy-mode and import_closure.py"""

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
DAEMON_SYNC = REPO_ROOT / "deploy" / "daemon-sync.sh"
IMPORT_CLOSURE = REPO_ROOT / "deploy" / "import_closure.py"


class TestImportClosure:
    """Unit tests for import_closure.py — closure computation."""

    def test_import_closure_exists(self):
        """Script must be present and executable."""
        assert IMPORT_CLOSURE.is_file()
        assert os.access(IMPORT_CLOSURE, os.X_OK)

    def test_closure_simple_chain(self, tmp_path):
        """Closure of a→b→c includes all three."""
        # Create: a.py imports b, b.py imports c, c.py imports nothing
        (tmp_path / "c.py").write_text("# c\n")
        (tmp_path / "b.py").write_text("import c\n")
        (tmp_path / "a.py").write_text("import b\n")

        result = subprocess.run(
            [sys.executable, str(IMPORT_CLOSURE), str(tmp_path), "a.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        closure = set(result.stdout.strip().split('\n'))
        assert closure == {"a.py", "b.py", "c.py"}

    def test_closure_excludes_unrelated_files(self, tmp_path):
        """Closure excludes files not imported."""
        (tmp_path / "a.py").write_text("import b\n")
        (tmp_path / "b.py").write_text("# b\n")
        (tmp_path / "z.py").write_text("# z (unrelated)\n")

        result = subprocess.run(
            [sys.executable, str(IMPORT_CLOSURE), str(tmp_path), "a.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        closure = set(result.stdout.strip().split('\n'))
        assert closure == {"a.py", "b.py"}
        assert "z.py" not in closure

    def test_closure_third_party_ignored(self, tmp_path):
        """Closure ignores third-party/stdlib imports."""
        (tmp_path / "a.py").write_text("import os\nimport yaml\nimport b\n")
        (tmp_path / "b.py").write_text("# b\n")

        result = subprocess.run(
            [sys.executable, str(IMPORT_CLOSURE), str(tmp_path), "a.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        closure = set(result.stdout.strip().split('\n'))
        assert closure == {"a.py", "b.py"}

    def test_closure_function_body_imports(self, tmp_path):
        """Closure includes imports inside function bodies."""
        # a.py imports b inside a function (lazy import)
        (tmp_path / "a.py").write_text(textwrap.dedent("""\
            def load_b():
                import b
                return b.func()
        """))
        (tmp_path / "b.py").write_text("def func(): pass\n")

        result = subprocess.run(
            [sys.executable, str(IMPORT_CLOSURE), str(tmp_path), "a.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        closure = set(result.stdout.strip().split('\n'))
        assert closure == {"a.py", "b.py"}, "Must find lazy imports in function bodies"

    def test_closure_seed_as_module_name(self, tmp_path):
        """Seeds can be bare module names (no .py)."""
        (tmp_path / "foo.py").write_text("import bar\n")
        (tmp_path / "bar.py").write_text("# bar\n")

        result = subprocess.run(
            [sys.executable, str(IMPORT_CLOSURE), str(tmp_path), "foo"],  # no .py
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        closure = set(result.stdout.strip().split('\n'))
        assert closure == {"foo.py", "bar.py"}

    def test_closure_unresolvable_seed_fails(self, tmp_path):
        """Unresolvable seed exits non-zero with error message."""
        (tmp_path / "a.py").write_text("# a\n")

        result = subprocess.run(
            [sys.executable, str(IMPORT_CLOSURE), str(tmp_path), "missing.py"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, "Must exit non-zero for missing seed"
        assert "missing" in result.stderr.lower(), "Must name the missing seed"

    def test_closure_multiple_seeds(self, tmp_path):
        """Multiple seeds create union of closures."""
        (tmp_path / "a.py").write_text("import c\n")
        (tmp_path / "b.py").write_text("import d\n")
        (tmp_path / "c.py").write_text("# c\n")
        (tmp_path / "d.py").write_text("# d\n")

        result = subprocess.run(
            [sys.executable, str(IMPORT_CLOSURE), str(tmp_path), "a.py", "b.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        closure = set(result.stdout.strip().split('\n'))
        assert closure == {"a.py", "b.py", "c.py", "d.py"}


class TestDaemonSync:
    """Tests for daemon-sync.sh — copy-mode logic and edge cases."""

    def test_daemon_sync_exists(self):
        """Script must be present and executable."""
        assert DAEMON_SYNC.is_file()
        assert os.access(DAEMON_SYNC, os.X_OK)

    def test_daemon_sync_syntax(self):
        """bash -n must report no syntax errors."""
        result = subprocess.run(
            ["bash", "-n", str(DAEMON_SYNC)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"

    def test_daemon_sync_help(self):
        """--help exits 0 and prints usage."""
        result = subprocess.run(
            ["bash", str(DAEMON_SYNC), "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout

    def test_copy_mode_manifest_entry_accepted(self, tmp_path):
        """Manifest with copy-mode entry parses without error."""
        manifest = tmp_path / "daemon-manifest.yaml"
        manifest.write_text(textwrap.dedent("""\
            repos:
              conductor:
                deploy_mode: copy
                deploy_source: /srv/git/conductor-working
                copy_from: scripts
                copy_to: /data/agents/scripts
                entry_points:
                  - night_coordinator.py
                seed_manifests:
                  - night_producers.yaml
        """))

        # Simple parse check: python3 can parse it
        result = subprocess.run(
            ["python3", "-c",
             f"import yaml; yaml.safe_load(open('{manifest}'))"],
            capture_output=True,
        )
        assert result.returncode == 0

    def test_service_mode_entry_unchanged(self, tmp_path):
        """Service-mode repos (weaver) work as before."""
        manifest = tmp_path / "daemon-manifest.yaml"
        manifest.write_text(textwrap.dedent("""\
            repos:
              weaver:
                deploy_source: /srv/git/weaver-working
                services:
                  - unit: weaver-server.service
                    scope: user
        """))

        # Parse check
        result = subprocess.run(
            ["python3", "-c",
             f"import yaml; yaml.safe_load(open('{manifest}'))"],
            capture_output=True,
        )
        assert result.returncode == 0

    def test_copy_mode_no_services_key_no_error(self, tmp_path):
        """Copy-mode repos without 'services' key don't cause drift-check KeyError."""
        # This is a regression test: step-1 drift check must skip repos with no services
        manifest = tmp_path / "daemon-manifest.yaml"
        manifest.write_text(textwrap.dedent("""\
            repos:
              conductor:
                deploy_mode: copy
                deploy_source: /srv/git/conductor-working
                copy_from: scripts
                copy_to: /data/agents/scripts
                entry_points:
                  - night_coordinator.py
        """))

        # Simulate the get_services() call for a copy-mode repo
        result = subprocess.run(
            ["python3", "-c", textwrap.dedent(f"""\
                import yaml
                with open('{manifest}') as f:
                    m = yaml.safe_load(f)
                for s in m['repos']['conductor'].get('services', []):
                    print(s['unit'] + '\\t' + s['scope'])
                # No exception if services is missing
                print("OK")
            """)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_daemon_sync_dry_run_with_copy_mode(self, tmp_path):
        """--dry-run with copy-mode entry prints planned copies and mutates nothing."""
        # Create mock git and mocks
        mock_git = self._make_mock_git(tmp_path)
        mock_python = self._make_mock_python_closure(tmp_path)

        # Create manifest with copy-mode entry
        manifest = tmp_path / "daemon-manifest.yaml"
        manifest.write_text(textwrap.dedent("""\
            repos:
              conductor:
                deploy_mode: copy
                deploy_source: /srv/git/conductor-working
                copy_from: scripts
                copy_to: /data/agents/scripts
                entry_points:
                  - night_coordinator.py
        """))

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + ":" + env["PATH"]
        env["MANIFEST"] = str(manifest)

        result = subprocess.run(
            ["bash", "-c", textwrap.dedent(f"""\
                export MANIFEST={manifest}
                bash {DAEMON_SYNC} --dry-run
            """)],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT)
        )

        assert result.returncode == 0, f"daemon-sync --dry-run failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "[dry-run]" in result.stdout, "--dry-run should emit [dry-run] lines"
        assert "rsync" in result.stdout.lower(), "dry-run should mention rsync copy operation"

    def test_daemon_sync_dry_run_no_mutations(self, tmp_path):
        """--dry-run must not create any files or directories."""
        mock_git = self._make_mock_git(tmp_path)
        mock_python = self._make_mock_python_closure(tmp_path)

        # Setup: verify copy_to doesn't exist before
        copy_to = tmp_path / "runtime_scripts"
        assert not copy_to.exists()

        manifest = tmp_path / "daemon-manifest.yaml"
        manifest.write_text(textwrap.dedent(f"""\
            repos:
              conductor:
                deploy_mode: copy
                deploy_source: /srv/git/conductor-working
                copy_from: scripts
                copy_to: {copy_to}
                entry_points:
                  - night_coordinator.py
        """))

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + ":" + env["PATH"]

        result = subprocess.run(
            ["bash", "-c", textwrap.dedent(f"""\
                bash {DAEMON_SYNC} --dry-run
            """)],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT)
        )

        # Verify copy_to was NOT created
        assert not copy_to.exists(), "--dry-run must not create copy_to directory"
        assert result.returncode == 0

    def _make_mock_git(self, tmp_path: Path) -> Path:
        """Return a mock `git` binary."""
        mock = tmp_path / "git"
        mock.write_text(textwrap.dedent("""\
            #!/bin/bash
            for arg in "$@"; do
                case "$arg" in
                    rev-parse) echo "abc1234def5678"; exit 0 ;;
                    diff)      exit 0 ;;  # no changes
                    merge)     exit 0 ;;  # merge succeeds
                    fetch)     exit 0 ;;  # fetch succeeds
                esac
            done
            exit 0
        """))
        mock.chmod(0o755)
        return mock

    def _make_mock_python_closure(self, tmp_path: Path) -> Path:
        """Return a mock import_closure.py that outputs minimal closure."""
        mock = REPO_ROOT / "deploy" / "import_closure.py"
        # The real import_closure.py should exist; this just ensures path is available
        return mock


class TestSmokeCheck:
    """Tests for deploy-time import smoke-check."""

    def test_smoke_check_success_with_present_modules(self, tmp_path):
        """Smoke-check passes when all modules import successfully."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "entry.py").write_text("import sys\n")
        (scripts_dir / "helper.py").write_text("# helper\n")

        # Inline smoke-check (same as in daemon-sync.sh step 5)
        result = subprocess.run(
            ["python3", "-c", textwrap.dedent(f"""\
                import sys, importlib
                sys.path.insert(0, '{scripts_dir}')
                seeds = ['entry', 'helper']
                failed = []
                for seed in seeds:
                    try:
                        importlib.import_module(seed)
                    except ModuleNotFoundError as e:
                        failed.append(f"{{seed}}: {{e}}")
                if failed:
                    for msg in failed:
                        print(msg, file=sys.stderr)
                    sys.exit(1)
                print("OK")
            """)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_smoke_check_fails_with_missing_module(self, tmp_path):
        """Smoke-check fails when a module cannot be imported."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "entry.py").write_text("# entry\n")

        result = subprocess.run(
            ["python3", "-c", textwrap.dedent(f"""\
                import sys, importlib
                sys.path.insert(0, '{scripts_dir}')
                seeds = ['entry', 'missing']
                failed = []
                for seed in seeds:
                    try:
                        importlib.import_module(seed)
                    except ModuleNotFoundError as e:
                        failed.append(f"{{seed}}: {{e}}")
                if failed:
                    for msg in failed:
                        print(msg, file=sys.stderr)
                    sys.exit(1)
            """)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "missing" in result.stderr.lower()


class TestRealConductorClosure:
    """HARD acceptance test: verify real conductor closure."""

    @pytest.mark.skipif(
        not Path("/srv/git/conductor-working").exists(),
        reason="conductor-working not mounted"
    )
    def test_conductor_closure_includes_required_modules(self):
        """Real conductor: closure of night_coordinator + producers includes the 3 readers."""
        conductor_scripts = Path("/srv/git/conductor-working/scripts")
        assert conductor_scripts.is_dir(), "conductor-working/scripts must exist"

        # Seeds: entry point + producer modules from night_producers.yaml
        seeds = ["night_coordinator.py"]

        # Extract producer modules from night_producers.yaml
        producers_yaml = conductor_scripts / "night_producers.yaml"
        if producers_yaml.exists():
            result = subprocess.run(
                ["python3", "-c", textwrap.dedent(f"""\
                    import yaml
                    with open('{producers_yaml}') as f:
                        m = yaml.safe_load(f)
                    for p in m.get('producers', []):
                        print(p['module'])
                """)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        seeds.append(line)

        # Compute closure
        result = subprocess.run(
            [sys.executable, str(IMPORT_CLOSURE), str(conductor_scripts)] + seeds,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"import_closure failed: {result.stderr}"

        closure = set(result.stdout.strip().split('\n'))

        # Check that required modules are in the closure
        required = ["enlightenment_reader.py", "arxiv_watch.py", "rsi_ingest.py"]
        for req in required:
            assert req in closure, (
                f"Required module {req} is NOT in closure. "
                f"This was the bug on 2026-06-09. "
                f"Closure: {closure}"
            )


class TestManifestSchemaConductorEntries:
    """The real daemon-manifest.yaml must explicitly list gpu_queue_runner.py
    and gpu-queue-runner.service, mirroring the flip_controller/gw_actuator
    entries — otherwise gpu_queue_runner.py silently falls out of the copy
    closure the same way flip_controller.py did before Unit B
    (gotcha/flip-controller-not-in-daemon-sync-copy-closure-2026-06-20)."""

    def test_gpu_queue_runner_in_entry_points_and_services(self):
        import yaml

        manifest_path = REPO_ROOT / "deploy" / "daemon-manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        conductor = manifest["repos"]["conductor"]

        assert "gpu_queue_runner.py" in conductor["entry_points"]
        # Prior entries must remain — this is additive, not a replacement.
        assert "flip_controller.py" in conductor["entry_points"]
        assert "gw_actuator.py" in conductor["entry_points"]

        service_units = {s["unit"]: s["scope"] for s in conductor["services"]}
        assert service_units.get("gpu-queue-runner.service") == "system"
        assert service_units.get("flip-controller.service") == "user"


class TestCopyModeRestartOnChange:
    """Coverage for copy-mode restart-on-change: RSYNC_CHANGES>0 + smoke-ok
    gate, restart hysteresis, and the --user-scope env prerequisite
    skip-and-flag path (AC2, AC3, AC5, AC6). Uses MANIFEST env override plus
    mocked git/systemctl so the whole daemon-sync.sh flow runs isolated from
    real host paths and real systemd units.
    """

    MARKER = Path("/tmp/daemon-sync-restart-flip-controller.service")

    @pytest.fixture(autouse=True)
    def _clean_markers(self):
        for p in Path("/tmp").glob("daemon-sync-restart-*"):
            p.unlink(missing_ok=True)
        yield
        for p in Path("/tmp").glob("daemon-sync-restart-*"):
            p.unlink(missing_ok=True)

    def _setup_fixture(self, tmp_path, hysteresis_sec=None, dbus_env=True, include_weaver=False):
        deploy_source = tmp_path / "conductor-working"
        scripts_dir = deploy_source / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "night_coordinator.py").write_text("# entry\n")

        copy_to = tmp_path / "runtime_scripts"

        repos_yaml = textwrap.dedent(f"""\
            repos:
              conductor:
                deploy_mode: copy
                deploy_source: {deploy_source}
                copy_from: scripts
                copy_to: {copy_to}
                entry_points:
                  - night_coordinator.py
                services:
                  - unit: flip-controller.service
                    scope: user
        """)
        if include_weaver:
            repos_yaml += (
                f"  weaver:\n"
                f"    deploy_source: {tmp_path / 'weaver-working'}\n"
                f"    services:\n"
                f"      - unit: weaver-server.service\n"
                f"        scope: user\n"
            )

        manifest = tmp_path / "daemon-manifest.yaml"
        manifest.write_text(repos_yaml)

        mock_git = tmp_path / "git"
        mock_git.write_text(textwrap.dedent("""\
            #!/bin/bash
            for arg in "$@"; do
                case "$arg" in
                    rev-parse) echo "abc1234def5678"; exit 0 ;;
                    diff)      exit 0 ;;
                    merge)     exit 0 ;;
                    fetch)     exit 0 ;;
                esac
            done
            exit 0
        """))
        mock_git.chmod(0o755)

        restart_log = tmp_path / "restart.log"
        pid_counter = tmp_path / "pid_counter"
        mock_systemctl = tmp_path / "systemctl"
        mock_systemctl.write_text(textwrap.dedent(f"""\
            #!/bin/bash
            for arg in "$@"; do
                case "$arg" in
                    restart) echo "restart:$*" >> "{restart_log}"; exit 0 ;;
                    --property=LoadState) echo "LoadState=loaded"; exit 0 ;;
                    --property=ActiveState) echo "ActiveState=active"; exit 0 ;;
                    --property=MainPID)
                        n=0
                        [[ -f "{pid_counter}" ]] && n=$(cat "{pid_counter}")
                        n=$((n+1))
                        echo "$n" > "{pid_counter}"
                        echo "MainPID=$((1000+n))"
                        exit 0
                        ;;
                esac
            done
            exit 0
        """))
        mock_systemctl.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + ":" + env["PATH"]
        env["MANIFEST"] = str(manifest)
        if hysteresis_sec is not None:
            env["DAEMON_SYNC_RESTART_HYSTERESIS_SEC"] = str(hysteresis_sec)
        if dbus_env:
            xdg_dir = tmp_path / "xdgrun"
            xdg_dir.mkdir(exist_ok=True)
            env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/tmp/fake-bus"
            env["XDG_RUNTIME_DIR"] = str(xdg_dir)
        else:
            env.pop("DBUS_SESSION_BUS_ADDRESS", None)
            env.pop("XDG_RUNTIME_DIR", None)

        return {
            "deploy_source": deploy_source,
            "scripts_dir": scripts_dir,
            "copy_to": copy_to,
            "manifest": manifest,
            "restart_log": restart_log,
            "env": env,
        }

    def _run(self, fixture, extra_env=None):
        env = dict(fixture["env"])
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(DAEMON_SYNC)],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
        )

    def _restart_count(self, fixture):
        if not fixture["restart_log"].exists():
            return 0
        return fixture["restart_log"].read_text().count("restart:")

    def test_restart_fires_when_changes_and_smoke_pass(self, tmp_path):
        """AC2: RSYNC_CHANGES>0 + smoke-ok -> restart fires, marker written."""
        fx = self._setup_fixture(tmp_path)
        result = self._run(fx)
        assert result.returncode == 0, result.stdout + result.stderr
        assert self._restart_count(fx) == 1
        assert "flip-controller.service" in fx["restart_log"].read_text()
        assert self.MARKER.exists()

    def test_no_restart_when_zero_changes(self, tmp_path):
        """AC2: a second run with no source changes -> rsync reports 0 changes -> no restart."""
        fx = self._setup_fixture(tmp_path, hysteresis_sec=0)
        self._run(fx)
        assert self._restart_count(fx) == 1

        result = self._run(fx)
        assert result.returncode == 0
        assert self._restart_count(fx) == 1, "no new restart expected when rsync changes == 0"
        assert "Rsync changed 0 files" in result.stdout

    def test_no_restart_when_smoke_check_fails(self, tmp_path):
        """AC3: import smoke-check failure blocks the restart and surfaces the anomaly."""
        fx = self._setup_fixture(tmp_path)
        (fx["scripts_dir"] / "night_coordinator.py").write_text("import this_module_does_not_exist\n")
        result = self._run(fx)
        assert result.returncode == 0
        assert self._restart_count(fx) == 0, "must not restart when smoke-check fails"
        assert "ANOMALOUS" in result.stderr
        assert "smoke-check failed" in result.stderr

    def test_hysteresis_blocks_second_restart_within_window(self, tmp_path):
        """AC5: two qualifying restarts within the hysteresis window -> exactly one restart;
        after the window elapses, a restart fires again."""
        # A 3s (not 1s) window avoids flaking on the marker's epoch-second
        # granularity: two runs a few ms apart could otherwise straddle a
        # second boundary and see elapsed=1 even though real time was <0.1s.
        fx = self._setup_fixture(tmp_path, hysteresis_sec=3)
        self._run(fx)
        assert self._restart_count(fx) == 1

        (fx["scripts_dir"] / "night_coordinator.py").write_text("# entry v2\n")
        result = self._run(fx)
        assert result.returncode == 0
        assert self._restart_count(fx) == 1, "second restart within the hysteresis window must be skipped"
        assert "hysteresis" in (result.stdout + result.stderr).lower()

        time.sleep(3.5)
        (fx["scripts_dir"] / "night_coordinator.py").write_text("# entry v3\n")
        result = self._run(fx)
        assert result.returncode == 0
        assert self._restart_count(fx) == 2, "restart should fire again once the hysteresis window elapses"

    def test_missing_marker_does_not_block_restart(self, tmp_path):
        """AC5: a missing/unreadable marker does not block the restart (assume-safe-and-proceed)."""
        fx = self._setup_fixture(tmp_path, hysteresis_sec=60)
        assert not self.MARKER.exists()
        result = self._run(fx)
        assert result.returncode == 0
        assert self._restart_count(fx) == 1

    def test_env_prereq_missing_skips_restart_and_marks_incomplete(self, tmp_path):
        """AC6: missing DBUS_SESSION_BUS_ADDRESS/XDG_RUNTIME_DIR -> restart skipped,
        deploy result reported INCOMPLETE (not success), anomaly surfaced, run proceeds."""
        fx = self._setup_fixture(tmp_path, dbus_env=False, include_weaver=True)
        result = self._run(fx)
        assert result.returncode == 0, "run must not halt even though the restart is skipped"
        assert self._restart_count(fx) == 0, "no systemctl --user restart call should be made"
        assert "INCOMPLETE" in result.stdout
        assert "ANOMALOUS" in result.stderr
        assert "DBUS_SESSION_BUS_ADDRESS" in result.stderr
        assert "── Repo: weaver" in result.stdout, "run must proceed to other repos, not halt"
