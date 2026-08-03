"""Tests for the GW reference leg's pinned origin/main worktree (U3c).

_dispatch_gw_reviewer (lapis_pm.spec_review) grounds the GW leg against a
detached origin/main worktree of the local /srv/git/<repo>-working clone,
mirroring the pattern agents-core's Facets orchestrator uses (PR #201) —
never the shared -working tree itself, which carries no guarantee of being at
origin/main, clean, or even on main.

These tests exercise the real worktree machinery (_create_gw_worktree /
_remove_gw_worktree) against a throwaway local git repo, overriding the
hermetic default that tests/conftest.py installs for every other spec-review
test. That default fakes worktree creation so unrelated tests stay
independent of host filesystem state; these tests exist specifically to
verify the real thing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.spec_review import (
    GwLegProvenance,
    _dispatch_gw_reviewer,
)
from lapis_pm.spec_review import _create_gw_worktree as _real_create_gw_worktree
from lapis_pm.spec_review import _remove_gw_worktree as _real_remove_gw_worktree


def _run_git(args, cwd):
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


def _mock_gw_module(call_gw_agent_mock):
    mock_module = MagicMock()
    mock_module.call_gw_agent = call_gw_agent_mock
    mock_module.DEFAULT_READONLY_TOOLS = []
    return mock_module


@pytest.fixture
def real_gw_repo():
    """A real /srv/git/<name>-working clone with a commit on main and a faked
    origin/main remote-tracking ref (no real remote needed — a local ref is
    enough for `git worktree add --detach <tmp> origin/main` to resolve exactly
    as it would against a real fetched clone). Torn down at test end."""
    repo_name = f"gwlegtest-{uuid.uuid4().hex[:10]}"
    working_dir = f"/srv/git/{repo_name}-working"
    os.makedirs(working_dir)
    try:
        _run_git(["init", "-q", "-b", "main"], working_dir)
        _run_git(["config", "user.email", "test@example.com"], working_dir)
        _run_git(["config", "user.name", "Test"], working_dir)
        (Path(working_dir) / "README.md").write_text("gw leg worktree test fixture\n")
        _run_git(["add", "README.md"], working_dir)
        _run_git(["commit", "-q", "-m", "init"], working_dir)
        sha = _run_git(["rev-parse", "HEAD"], working_dir)
        _run_git(["update-ref", "refs/remotes/origin/main", sha], working_dir)
        yield repo_name, working_dir, sha
    finally:
        subprocess.run(["git", "-C", working_dir, "worktree", "prune"], capture_output=True)
        shutil.rmtree(working_dir, ignore_errors=True)


def _use_real_worktree_helpers(monkeypatch):
    monkeypatch.setattr("lapis_pm.spec_review._create_gw_worktree", _real_create_gw_worktree)
    monkeypatch.setattr("lapis_pm.spec_review._remove_gw_worktree", _real_remove_gw_worktree)


# ---------------------------------------------------------------------------
# DoD 2: call_gw_agent invoked with cwd == worktree path, not -working
# ---------------------------------------------------------------------------

def test_call_gw_agent_invoked_with_worktree_path_as_cwd(real_gw_repo, monkeypatch):
    repo_name, working_dir, sha = real_gw_repo
    _use_real_worktree_helpers(monkeypatch)

    mock_call_gw = MagicMock(return_value=("clean", []))
    mock_gw_module = _mock_gw_module(mock_call_gw)

    with patch.dict(os.environ, {"GW_URL": "http://203.0.113.11:8081"}, clear=False):
        os.environ.pop("GW_SLOT2_URL", None)
        with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
            with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
                text, transcript, elapsed, skip_reason, provenance = _dispatch_gw_reviewer(
                    spec_text="spec",
                    synth_target_id="tid",
                    parsed_target_id="id",
                    repo=repo_name,
                    run_id="run-cwd",
                )

    mock_call_gw.assert_called_once()
    used_cwd = mock_call_gw.call_args.kwargs.get("cwd")
    assert used_cwd != working_dir
    assert used_cwd.startswith(tempfile.gettempdir())
    assert "run-cwd" in used_cwd
    assert provenance is not None
    assert provenance.worktree_path == used_cwd
    # Cleaned up by the finally block after the call returns.
    assert not os.path.exists(used_cwd)


# ---------------------------------------------------------------------------
# DoD 3: a git/worktree failure returns the skip-tuple shape, never raises
# ---------------------------------------------------------------------------

def test_worktree_git_failure_returns_skip_tuple_no_raise(monkeypatch):
    _use_real_worktree_helpers(monkeypatch)

    mock_call_gw = MagicMock(return_value=("clean", []))
    mock_gw_module = _mock_gw_module(mock_call_gw)

    with patch.dict(os.environ, {"GW_URL": "http://203.0.113.11:8081"}, clear=False):
        os.environ.pop("GW_SLOT2_URL", None)
        with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
            with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
                result = _dispatch_gw_reviewer(
                    spec_text="spec",
                    synth_target_id="tid",
                    parsed_target_id="id",
                    repo=f"nonexistent-repo-{uuid.uuid4().hex[:8]}",
                    run_id="run-fail",
                )

    text, transcript, elapsed, skip_reason, provenance = result
    assert text is None
    assert transcript == []
    assert skip_reason == "worktree_unavailable"
    assert provenance is None
    mock_call_gw.assert_not_called()


# ---------------------------------------------------------------------------
# DoD 4: provenance.resolved_sha comes from `git rev-parse HEAD` inside the
# worktree itself, matching what the leg actually reads.
# ---------------------------------------------------------------------------

def test_provenance_resolved_sha_matches_worktree_head(real_gw_repo, monkeypatch):
    repo_name, working_dir, sha = real_gw_repo
    _use_real_worktree_helpers(monkeypatch)

    captured = {}

    def fake_call_gw_agent(*args, **kwargs):
        cwd = kwargs["cwd"]
        captured["head_in_worktree"] = _run_git(["rev-parse", "HEAD"], cwd)
        return "clean", []

    mock_gw_module = MagicMock()
    mock_gw_module.call_gw_agent = fake_call_gw_agent
    mock_gw_module.DEFAULT_READONLY_TOOLS = []

    with patch.dict(os.environ, {"GW_URL": "http://203.0.113.11:8081"}, clear=False):
        os.environ.pop("GW_SLOT2_URL", None)
        with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
            with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
                text, transcript, elapsed, skip_reason, provenance = _dispatch_gw_reviewer(
                    spec_text="spec",
                    synth_target_id="tid",
                    parsed_target_id="id",
                    repo=repo_name,
                    run_id="run-sha",
                )

    assert isinstance(provenance, GwLegProvenance)
    assert provenance.resolved_sha == sha
    assert provenance.resolved_sha == captured["head_in_worktree"]
    assert sha in provenance.staleness_warning


# ---------------------------------------------------------------------------
# DoD 5: staleness-warning line emitted to stderr with the family prefix
# ---------------------------------------------------------------------------

def test_staleness_warning_emitted_to_stderr(real_gw_repo, monkeypatch, capsys):
    repo_name, working_dir, sha = real_gw_repo
    _use_real_worktree_helpers(monkeypatch)

    mock_call_gw = MagicMock(return_value=("clean", []))
    mock_gw_module = _mock_gw_module(mock_call_gw)

    with patch.dict(os.environ, {"GW_URL": "http://203.0.113.11:8081"}, clear=False):
        os.environ.pop("GW_SLOT2_URL", None)
        with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
            with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
                _dispatch_gw_reviewer(
                    spec_text="spec",
                    synth_target_id="tid",
                    parsed_target_id="id",
                    repo=repo_name,
                    run_id="run-warn",
                )

    captured = capsys.readouterr()
    assert "[spec-review:gw-reviewer-pinned]" in captured.err
    assert sha in captured.err
    assert "may not reflect the current remote state" in captured.err


# ---------------------------------------------------------------------------
# DoD 6: concurrent run_ids never collide on worktree path
# ---------------------------------------------------------------------------

def test_concurrent_run_ids_produce_distinct_worktree_paths(real_gw_repo):
    repo_name, working_dir, sha = real_gw_repo

    path_a, sha_a, err_a = _real_create_gw_worktree(repo_name, "run-a")
    path_b, sha_b, err_b = _real_create_gw_worktree(repo_name, "run-b")
    try:
        assert err_a == "" and err_b == ""
        assert path_a is not None and path_b is not None
        assert path_a != path_b
        assert "run-a" in path_a
        assert "run-b" in path_b
        assert os.path.isdir(path_a)
        assert os.path.isdir(path_b)
    finally:
        _real_remove_gw_worktree(repo_name, path_a)
        _real_remove_gw_worktree(repo_name, path_b)


# ---------------------------------------------------------------------------
# DoD 7 + 9: worktree removed after both a successful and a failed call;
# provenance stays populated when call_gw_agent itself fails after a
# successful worktree.
# ---------------------------------------------------------------------------

def test_worktree_removed_after_successful_call(real_gw_repo, monkeypatch):
    repo_name, working_dir, sha = real_gw_repo
    _use_real_worktree_helpers(monkeypatch)

    mock_call_gw = MagicMock(return_value=("clean", []))
    mock_gw_module = _mock_gw_module(mock_call_gw)

    with patch.dict(os.environ, {"GW_URL": "http://203.0.113.11:8081"}, clear=False):
        os.environ.pop("GW_SLOT2_URL", None)
        with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
            with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
                _dispatch_gw_reviewer(
                    spec_text="spec",
                    synth_target_id="tid",
                    parsed_target_id="id",
                    repo=repo_name,
                    run_id="run-cleanup-ok",
                )

    listing = _run_git(["worktree", "list"], working_dir)
    assert "run-cleanup-ok" not in listing


def test_worktree_removed_after_failed_call_provenance_still_populated(real_gw_repo, monkeypatch):
    repo_name, working_dir, sha = real_gw_repo
    _use_real_worktree_helpers(monkeypatch)

    mock_call_gw = MagicMock(side_effect=Exception("boom"))
    mock_gw_module = _mock_gw_module(mock_call_gw)

    with patch.dict(os.environ, {"GW_URL": "http://203.0.113.11:8081"}, clear=False):
        os.environ.pop("GW_SLOT2_URL", None)
        with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
            with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
                text, transcript, elapsed, skip_reason, provenance = _dispatch_gw_reviewer(
                    spec_text="spec",
                    synth_target_id="tid",
                    parsed_target_id="id",
                    repo=repo_name,
                    run_id="run-cleanup-fail",
                )

    assert text is None
    assert skip_reason == "call_error"
    assert provenance is not None
    assert provenance.resolved_sha == sha

    listing = _run_git(["worktree", "list"], working_dir)
    assert "run-cleanup-fail" not in listing


# ---------------------------------------------------------------------------
# DoD 8: `git worktree add` failing means the finally block never attempts
# `git worktree remove` for a worktree that was never created.
# ---------------------------------------------------------------------------

def test_worktree_add_failure_no_removal_attempted(monkeypatch):
    fake_create = MagicMock(return_value=(None, None, "git_worktree_add_failed:simulated"))
    fake_remove = MagicMock()
    monkeypatch.setattr("lapis_pm.spec_review._create_gw_worktree", fake_create)
    monkeypatch.setattr("lapis_pm.spec_review._remove_gw_worktree", fake_remove)

    mock_call_gw = MagicMock(return_value=("clean", []))
    mock_gw_module = _mock_gw_module(mock_call_gw)

    with patch.dict(os.environ, {"GW_URL": "http://203.0.113.11:8081"}, clear=False):
        os.environ.pop("GW_SLOT2_URL", None)
        with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
            with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
                text, transcript, elapsed, skip_reason, provenance = _dispatch_gw_reviewer(
                    spec_text="spec",
                    synth_target_id="tid",
                    parsed_target_id="id",
                    repo="whatever-repo",
                    run_id="run-add-fail",
                )

    assert skip_reason == "worktree_unavailable"
    assert provenance is None
    mock_call_gw.assert_not_called()
    fake_remove.assert_not_called()


# ---------------------------------------------------------------------------
# DoD 10: a `git worktree remove` failure in the finally block is logged as a
# warning and never masks the original try-body result.
# ---------------------------------------------------------------------------

def test_worktree_remove_raise_in_finally_preserves_result_and_logs_warning(
    real_gw_repo, monkeypatch, capsys
):
    repo_name, working_dir, sha = real_gw_repo
    monkeypatch.setattr("lapis_pm.spec_review._create_gw_worktree", _real_create_gw_worktree)

    def raising_remove(repo, worktree_path):
        raise RuntimeError("simulated remove failure")

    monkeypatch.setattr("lapis_pm.spec_review._remove_gw_worktree", raising_remove)

    mock_call_gw = MagicMock(return_value=("clean", []))
    mock_gw_module = _mock_gw_module(mock_call_gw)

    with patch.dict(os.environ, {"GW_URL": "http://203.0.113.11:8081"}, clear=False):
        os.environ.pop("GW_SLOT2_URL", None)
        with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
            with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
                text, transcript, elapsed, skip_reason, provenance = _dispatch_gw_reviewer(
                    spec_text="spec",
                    synth_target_id="tid",
                    parsed_target_id="id",
                    repo=repo_name,
                    run_id="run-remove-raise",
                )

    assert text == "clean"
    assert skip_reason == ""
    assert provenance is not None

    captured = capsys.readouterr()
    assert "[spec-review:gw-reviewer-cleanup-warning]" in captured.err
    assert "simulated remove failure" in captured.err

    # The fake raised instead of actually removing the worktree — clean it up
    # for real so the fixture teardown doesn't have to warn about it.
    _real_remove_gw_worktree(repo_name, provenance.worktree_path)


# ---------------------------------------------------------------------------
# DoD 11: the abandoned_event branch is unaffected — cleanup still happens
# via the same unconditional finally.
# ---------------------------------------------------------------------------

def test_abandoned_leg_worktree_still_cleaned_up(real_gw_repo, monkeypatch, capsys):
    repo_name, working_dir, sha = real_gw_repo
    _use_real_worktree_helpers(monkeypatch)

    mock_call_gw = MagicMock(return_value=("clean", []))
    mock_gw_module = _mock_gw_module(mock_call_gw)

    event = threading.Event()
    event.set()

    with patch.dict(os.environ, {"GW_URL": "http://203.0.113.11:8081"}, clear=False):
        os.environ.pop("GW_SLOT2_URL", None)
        with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
            with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
                text, transcript, elapsed, skip_reason, provenance = _dispatch_gw_reviewer(
                    spec_text="spec",
                    synth_target_id="tid",
                    parsed_target_id="id",
                    repo=repo_name,
                    run_id="run-abandoned-cleanup",
                    abandoned_event=event,
                )

    captured = capsys.readouterr()
    assert "[spec-review:gw-reviewer-abandoned]" in captured.err

    listing = _run_git(["worktree", "list"], working_dir)
    assert "run-abandoned-cleanup" not in listing
