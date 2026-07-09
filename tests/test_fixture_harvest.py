"""Offline tests for the single-commit fixture harvest (lapis-pm-fixture-harvest-symptom-first-v0).

All tests run against a self-contained TOY git repo with mock_launder=True and
skip_base_runs=True - fully offline, no live pytest, no GW dependency. This only
exercises the BLIND path (fail_first_confirmed is never computed without real base
runs) plus the symptom-first framing / reproduce-pointer / leak-guard logic, which is
the correct machine-verifiable surface for this unit (see spec DON'T section).
"""

import argparse
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from lapis_pm.batched_fixer_eval import (
    FixtureRecord,
    harvest_one,
    _assert_no_intent_leak,
    REPRODUCE_POINTER_TMPL,
    TargetShaNotResolvedError,
    NonDiscriminatesHarvestError,
)


def _make_harvest_toy_repo(path: Path) -> tuple[str, str]:
    """Build a toy repo: parent commit, then a fix commit with a source change plus
    one co-committed test. Returns (parent_sha, fix_sha)."""
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=path, check=True, capture_output=True)

    (path / "module.py").write_text("def foo():\n    return 'old'\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)
    parent_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True
    ).stdout.strip()

    (path / "module.py").write_text("def foo():\n    return 'new'\n")
    tests_dir = path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_module.py").write_text(
        "from module import foo\n\n\ndef test_foo_returns_new():\n    assert foo() == 'new'\n"
    )
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "fix(module): correct foo() return value"],
        cwd=path, check=True, capture_output=True,
    )
    fix_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True
    ).stdout.strip()

    return parent_sha, fix_sha


# ---------------------------------------------------------------------------
# harvest_one: field derivation, symptom-first framing, leak guard
# ---------------------------------------------------------------------------


def test_harvest_one_field_derivation(tmp_path):
    toy_repo = tmp_path / "toy_repo"
    toy_repo.mkdir()
    parent_sha, fix_sha = _make_harvest_toy_repo(toy_repo)

    fixture = harvest_one(
        fix_sha, toy_repo, "toy", skip_base_runs=True, mock_launder=True,
    )

    assert fixture is not None
    assert isinstance(fixture, FixtureRecord)
    assert fixture.repo == "toy"
    assert fixture.sha == fix_sha
    assert fixture.parent_sha == parent_sha
    assert fixture.scoped_test_files == ["tests/test_module.py"]
    assert fixture.golden_test_ids == ["tests/test_module.py::test_foo_returns_new"]
    # skip_base_runs=True never computes fail_first_confirmed -> BLIND, not DISCRIMINATES
    # (the offline test's correct surface per the spec's DON'T section).
    assert fixture.checker_class == "BLIND"


def test_harvest_one_reproduce_pointer_present_no_leak(tmp_path):
    toy_repo = tmp_path / "toy_repo2"
    toy_repo.mkdir()
    _parent_sha, fix_sha = _make_harvest_toy_repo(toy_repo)

    fixture = harvest_one(
        fix_sha, toy_repo, "toy", skip_base_runs=True, mock_launder=True,
    )

    assert fixture is not None
    expected_pointer = REPRODUCE_POINTER_TMPL.format(test_files="tests/test_module.py")
    assert expected_pointer in fixture.task_intent_paraphrased
    for tid in fixture.golden_test_ids:
        assert tid not in fixture.task_intent_paraphrased


def test_leak_guard_raises_on_injected_golden_test_id():
    golden_test_ids = ["tests/test_module.py::test_foo_returns_new"]
    paraphrased = "Investigate this. See tests/test_module.py::test_foo_returns_new for details."
    with pytest.raises(RuntimeError, match="Leak guard violated"):
        _assert_no_intent_leak(paraphrased, golden_test_ids, "", "", "deadbeef")


def test_harvest_one_leak_guard_raises_when_launder_intent_leaks(tmp_path, monkeypatch):
    import lapis_pm.batched_fixer_eval as bfe

    toy_repo = tmp_path / "toy_repo_leak"
    toy_repo.mkdir()
    _parent_sha, fix_sha = _make_harvest_toy_repo(toy_repo)

    def _leaking_launder(raw, mock_mode=False, scoped_test_files=None):
        return "tests/test_module.py::test_foo_returns_new leaked verbatim", "mock"

    monkeypatch.setattr(bfe, "launder_intent", _leaking_launder)

    with pytest.raises(RuntimeError, match="Leak guard violated"):
        bfe.harvest_one(fix_sha, toy_repo, "toy", skip_base_runs=True, mock_launder=True)


def test_harvest_one_invalid_sha_raises_target_sha_not_resolved(tmp_path):
    toy_repo = tmp_path / "toy_repo_invalid"
    toy_repo.mkdir()
    _make_harvest_toy_repo(toy_repo)

    with pytest.raises(TargetShaNotResolvedError):
        harvest_one("0" * 40, toy_repo, "toy", skip_base_runs=True, mock_launder=True)


# ---------------------------------------------------------------------------
# --target-sha routing through run_eval
# ---------------------------------------------------------------------------


def test_target_sha_routes_run_eval_to_single_fixture(tmp_path, monkeypatch):
    import lapis_pm.batched_fixer_eval as bfe

    toy_repo = tmp_path / "toy_repo_route"
    toy_repo.mkdir()
    _parent_sha, fix_sha = _make_harvest_toy_repo(toy_repo)

    monkeypatch.setattr(bfe, "LAPIS_PM_REPO", toy_repo)
    monkeypatch.setattr(bfe, "CORPUS_DIR", tmp_path / "corpus")

    result = bfe.run_eval(
        phase="build-corpus", mock_mode=True,
        target_sha=fix_sha, repo_only="lapis-pm",
    )
    assert result is None

    json_files = list((tmp_path / "corpus").glob("*.json"))
    assert len(json_files) == 1


# ---------------------------------------------------------------------------
# CLI guards: invalid sha (exit 2), non-DISCRIMINATES (exit 1)
# ---------------------------------------------------------------------------


def test_cli_invalid_target_sha_exits_2_no_save(tmp_path, monkeypatch, capsys):
    import lapis_pm.batched_fixer_eval as bfe
    from lapis_pm import cli

    toy_repo = tmp_path / "toy_repo_cli_invalid"
    toy_repo.mkdir()
    _make_harvest_toy_repo(toy_repo)

    monkeypatch.setattr(bfe, "LAPIS_PM_REPO", toy_repo)

    saved = []
    monkeypatch.setattr(bfe, "save_corpus", lambda c: saved.extend(c))

    args = argparse.Namespace(
        phase="build-corpus", run_id=None, mock=True,
        target_sha="0" * 40, repo_only="lapis-pm", json=False,
    )
    rc = cli.cmd_batched_fixer_eval(args)

    assert rc == 2
    assert saved == []
    captured = capsys.readouterr()
    assert "error: --target-sha" in captured.err
    assert "could not be resolved in repo 'lapis-pm'" in captured.err


def test_cli_non_discriminates_harvest_exits_1_no_save(tmp_path, monkeypatch, capsys):
    import lapis_pm.batched_fixer_eval as bfe
    from lapis_pm import cli

    # Startup GW probe only fires when mock_mode is False; stub it to "gw" so the
    # non-DISCRIMINATES guard (which only fires in real/non-mock mode) is what's
    # actually under test here, not GW availability.
    monkeypatch.setattr(bfe, "launder_intent", lambda *a, **kw: ("stub probe ok", "gw"))

    dummy_fixture = FixtureRecord(
        repo="lapis-pm", sha="abc123def456", parent_sha="def456abc123",
        pr_number=None, path="module.py", file_loc="line 1-2", changed_lines=5,
        tier="T1", checker_class="BLIND",
    )
    monkeypatch.setattr(bfe, "harvest_one", lambda *a, **kw: dummy_fixture)

    saved = []
    monkeypatch.setattr(bfe, "save_corpus", lambda c: saved.extend(c))

    args = argparse.Namespace(
        phase="build-corpus", run_id=None, mock=False,
        target_sha="abc123def456", repo_only="lapis-pm", json=False,
    )
    rc = cli.cmd_batched_fixer_eval(args)

    assert rc == 1
    assert saved == []
    captured = capsys.readouterr()
    assert "warning: harvested fixture for abc123def456 is BLIND" in captured.err
    assert "refusing to save" in captured.err


def test_cli_mock_mode_non_discriminates_guard_exempt(tmp_path, monkeypatch):
    """In --mock/skip_base_runs mode the checker_class is not real, so the
    non-DISCRIMINATES guard must NOT fire (offline test path is exempt)."""
    import lapis_pm.batched_fixer_eval as bfe
    from lapis_pm import cli

    toy_repo = tmp_path / "toy_repo_mock_exempt"
    toy_repo.mkdir()
    _parent_sha, fix_sha = _make_harvest_toy_repo(toy_repo)

    monkeypatch.setattr(bfe, "LAPIS_PM_REPO", toy_repo)
    monkeypatch.setattr(bfe, "CORPUS_DIR", tmp_path / "corpus")

    args = argparse.Namespace(
        phase="build-corpus", run_id=None, mock=True,
        target_sha=fix_sha, repo_only="lapis-pm", json=False,
    )
    rc = cli.cmd_batched_fixer_eval(args)

    assert rc == 0
    json_files = list((tmp_path / "corpus").glob("*.json"))
    assert len(json_files) == 1


# ---------------------------------------------------------------------------
# Shared-tree safety: dedicated_clone used, clone path (never repo_path) passed
# to run_base_tests_n_times/verify_fail_first, and no mutating git op on repo_path.
# ---------------------------------------------------------------------------


def test_harvest_one_shared_tree_safety(tmp_path, monkeypatch):
    import lapis_pm.batched_fixer_eval as bfe

    toy_repo = tmp_path / "toy_repo_safety"
    toy_repo.mkdir()
    _parent_sha, fix_sha = _make_harvest_toy_repo(toy_repo)

    real_dedicated_clone = bfe.dedicated_clone
    clone_entries = []

    @contextmanager
    def spy_dedicated_clone(repo_path, run_id):
        clone_entries.append(repo_path)
        with real_dedicated_clone(repo_path, run_id) as cp:
            yield cp

    base_runs_calls = []

    def spy_run_base_tests_n_times(clone_path, parent_sha, sha, test_files, n=3):
        base_runs_calls.append(clone_path)
        return [], [], []

    ff_calls = []

    def spy_verify_fail_first(clone_path, parent_sha, golden_test_diff, golden_source_diff, golden_test_ids, n=3):
        ff_calls.append(clone_path)
        return False, False, False

    monkeypatch.setattr(bfe, "dedicated_clone", spy_dedicated_clone)
    monkeypatch.setattr(bfe, "run_base_tests_n_times", spy_run_base_tests_n_times)
    monkeypatch.setattr(bfe, "verify_fail_first", spy_verify_fail_first)

    real_subprocess_run = subprocess.run
    repo_path_cmds = []

    def spy_subprocess_run(cmd, *args, **kwargs):
        if kwargs.get("cwd") == toy_repo:
            repo_path_cmds.append(list(cmd))
        return real_subprocess_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy_subprocess_run)

    fixture = bfe.harvest_one(
        fix_sha, toy_repo, "toy", skip_base_runs=False, mock_launder=True,
    )

    assert fixture is not None
    assert clone_entries == [toy_repo], "dedicated_clone must be entered exactly once, with repo_path"
    assert base_runs_calls, "run_base_tests_n_times must be invoked"
    assert all(cp != toy_repo for cp in base_runs_calls), (
        "run_base_tests_n_times must be invoked with the clone path, never repo_path"
    )
    assert ff_calls, "verify_fail_first must be invoked"
    assert all(cp != toy_repo for cp in ff_calls), (
        "verify_fail_first must be invoked with the clone path, never repo_path"
    )

    # No worktree-add / checkout / mutating op ever targets repo_path directly - only
    # read-only metadata reads (rev-parse, show) are allowed against the shared tree.
    for cmd in repo_path_cmds:
        assert not (len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "worktree"), (
            f"git worktree op must never target repo_path: {cmd}"
        )
        assert not (len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "checkout"), (
            f"git checkout must never target repo_path: {cmd}"
        )
    assert any(cmd[:2] == ["git", "rev-parse"] for cmd in repo_path_cmds), (
        "the read-only git rev-parse must be present against repo_path"
    )
