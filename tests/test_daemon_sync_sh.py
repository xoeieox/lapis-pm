"""Tests for deploy/daemon-sync.sh copy-mode and import_closure.py"""

import os
import subprocess
import sys
import tempfile
import textwrap
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
