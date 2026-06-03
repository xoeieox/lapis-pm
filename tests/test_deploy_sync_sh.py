"""Tests for deploy/sync.sh — syntax check and dry-run smoke."""

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

# Absolute path to the script under test
REPO_ROOT = Path(__file__).parent.parent
SYNC_SH = REPO_ROOT / "deploy" / "sync.sh"


def test_sync_sh_exists():
    """Script must be present and executable."""
    assert SYNC_SH.is_file(), "deploy/sync.sh not found"
    assert os.access(SYNC_SH, os.X_OK), "deploy/sync.sh is not executable"


def test_sync_sh_syntax():
    """bash -n must report no syntax errors."""
    result = subprocess.run(
        ["bash", "-n", str(SYNC_SH)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


def test_sync_sh_help():
    """--help exits 0 and prints usage."""
    result = subprocess.run(
        ["bash", str(SYNC_SH), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_sync_sh_unknown_arg():
    """Unknown argument exits non-zero."""
    result = subprocess.run(
        ["bash", str(SYNC_SH), "--bogus"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def _make_mock_git(tmp_path: Path, fake_head: str = "abc1234def5678") -> Path:
    """Return a mock `git` binary that handles rev-parse and diff without touching disk."""
    mock = tmp_path / "git"
    mock.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        for arg in "$@"; do
            case "$arg" in
                rev-parse) echo "{fake_head}"; exit 0 ;;
                diff)      exit 0 ;;  # no pyproject.toml change
            esac
        done
        exit 0
    """))
    mock.chmod(0o755)
    return mock


def _make_forbidden_sudo(tmp_path: Path) -> Path:
    """Return a mock `sudo` that fails loudly — must not be called in dry-run."""
    mock = tmp_path / "sudo"
    mock.write_text(textwrap.dedent("""\
        #!/bin/bash
        echo "SUDO_CALLED_IN_DRY_RUN: $*" >&2
        exit 1
    """))
    mock.chmod(0o755)
    return mock


def test_sync_sh_dry_run_no_sudo_calls(tmp_path):
    """--dry-run must print planned actions and never invoke sudo."""
    _make_mock_git(tmp_path)
    _make_forbidden_sudo(tmp_path)

    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env["PATH"]

    result = subprocess.run(
        ["bash", str(SYNC_SH), "--dry-run"],
        capture_output=True, text=True, env=env,
    )

    assert result.returncode == 0, f"dry-run exited non-zero:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "SUDO_CALLED_IN_DRY_RUN" not in result.stderr, (
        "sudo was invoked during --dry-run"
    )
    assert "[dry-run]" in result.stdout, "--dry-run should emit [dry-run] lines"


def test_sync_sh_dry_run_prints_git_fetch_and_merge(tmp_path):
    """--dry-run output includes the git fetch and merge commands."""
    _make_mock_git(tmp_path)
    _make_forbidden_sudo(tmp_path)

    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env["PATH"]

    result = subprocess.run(
        ["bash", str(SYNC_SH), "--dry-run"],
        capture_output=True, text=True, env=env,
    )

    assert result.returncode == 0
    assert "fetch origin" in result.stdout
    assert "merge --ff-only" in result.stdout


def test_sync_sh_dry_run_no_cp_calls(tmp_path):
    """--dry-run must not call cp (no mutation of /etc/systemd/system/)."""
    _make_mock_git(tmp_path)
    _make_forbidden_sudo(tmp_path)

    # Intercept cp as well
    fake_cp = tmp_path / "cp"
    fake_cp.write_text(textwrap.dedent("""\
        #!/bin/bash
        echo "CP_CALLED_IN_DRY_RUN: $*" >&2
        exit 1
    """))
    fake_cp.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env["PATH"]

    result = subprocess.run(
        ["bash", str(SYNC_SH), "--dry-run"],
        capture_output=True, text=True, env=env,
    )

    assert result.returncode == 0, f"dry-run crashed: {result.stderr}"
    assert "CP_CALLED_IN_DRY_RUN" not in result.stderr


def test_sync_sh_dry_run_summary_line(tmp_path):
    """--dry-run output ends with a Summary line."""
    _make_mock_git(tmp_path)
    _make_forbidden_sudo(tmp_path)

    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env["PATH"]

    result = subprocess.run(
        ["bash", str(SYNC_SH), "--dry-run"],
        capture_output=True, text=True, env=env,
    )

    assert result.returncode == 0
    assert "Summary:" in result.stdout
