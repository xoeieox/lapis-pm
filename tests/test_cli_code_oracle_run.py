"""Tests for `lapis-pm code-oracle-run` (lapis-pm-code-oracle-runner-v0 spec).

Thin CLI wiring to the existing `run_code_oracle_experiment` harness
(lapis_pm/code_oracle_run.py). Covers all 7 ACs; `load_corpus` and
`run_code_oracle_experiment` are mocked throughout - the harness itself is
covered by tests/test_code_oracle_run.py.
"""

from __future__ import annotations

import io
import subprocess as sp
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from lapis_pm.batched_fixer_eval import FixtureRecord, LAPIS_PM_REPO
from lapis_pm.cli import build_parser, main


def _make_fixture(repo="lapis-pm", sha="66a7d2ea" + "0" * 32, parent_sha="deadbeef"):
    return FixtureRecord(
        repo=repo,
        sha=sha,
        parent_sha=parent_sha,
        pr_number=1,
        path="lapis_pm/foo.py",
        file_loc="line 1-2",
        changed_lines=2,
        tier="T1",
        task_intent_raw="fix foo",
        task_intent_paraphrased="fix foo",
    )


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", str(repo)], check=True, capture_output=True)
    sp.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "config", "user.name", "T"], cwd=repo, check=True, capture_output=True)
    (repo / "f.txt").write_text("x")
    sp.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)
    sha = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    return repo, sha


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# AC1 - subcommand exists & parses, literal defaults, required-arg enforcement
# ---------------------------------------------------------------------------


def test_ac1_parses_full_arg_set_with_defaults():
    parser = build_parser()
    args = parser.parse_args([
        "code-oracle-run",
        "--fixture-id", "lapis-pm-66a7d2ea",
        "--agent-backend-url", "http://203.0.113.11:8082/v1",
    ])
    assert args.fixture_id == "lapis-pm-66a7d2ea"
    assert args.agent_backend_url == "http://203.0.113.11:8082/v1"
    assert args.repo_path == LAPIS_PM_REPO
    assert args.out_dir == Path("/tmp/code-oracle-run")
    assert args.run_id is None
    assert args.max_steps == 60
    assert args.timeout == 1800


def test_ac1_missing_required_arg_exits_2():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["code-oracle-run", "--fixture-id", "lapis-pm-66a7d2ea"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# AC2 - fixture lookup by id
# ---------------------------------------------------------------------------


def test_ac2_selects_correct_fixture_by_id(tmp_path):
    repo, sha = _git_repo(tmp_path)
    target = _make_fixture(parent_sha=sha)
    other = _make_fixture(repo="conductor", sha="ffffffff" + "0" * 32, parent_sha=sha)

    with patch("lapis_pm.batched_fixer_eval.load_corpus", return_value=[other, target]), \
         patch("lapis_pm.code_oracle_run.run_code_oracle_experiment") as mock_run:
        mock_run.return_value = {
            "run_id": "lapis-pm-66a7d2ea", "fixture_id": "lapis-pm-66a7d2ea",
            "agent_backend_url": "http://x/v1", "oracle_outcome": "pass",
            "apply_status": "success", "agent_self_report": None,
        }
        code, _, _ = _run([
            "code-oracle-run",
            "--fixture-id", "lapis-pm-66a7d2ea",
            "--agent-backend-url", "http://x/v1",
            "--repo-path", str(repo),
        ])

    assert code == 0
    assert mock_run.call_args.kwargs["fixture"] is target


# ---------------------------------------------------------------------------
# AC3 - unknown fixture id is a loud, CLI-layer, non-zero exit
# ---------------------------------------------------------------------------


def test_ac3_unknown_fixture_id_exits_1_no_harness_call(tmp_path):
    repo, sha = _git_repo(tmp_path)
    known = _make_fixture(parent_sha=sha)

    with patch("lapis_pm.batched_fixer_eval.load_corpus", return_value=[known]), \
         patch("lapis_pm.code_oracle_run.run_code_oracle_experiment") as mock_run:
        code, _, err = _run([
            "code-oracle-run",
            "--fixture-id", "does-not-exist",
            "--agent-backend-url", "http://x/v1",
            "--repo-path", str(repo),
        ])

    assert code == 1
    assert "does-not-exist" in err
    assert "lapis-pm-66a7d2ea" in err
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# AC4 - exact kwarg delegation
# ---------------------------------------------------------------------------


def test_ac4_delegates_with_exact_kwarg_names(tmp_path):
    repo, sha = _git_repo(tmp_path)
    fixture = _make_fixture(parent_sha=sha)
    out_dir = tmp_path / "out"

    with patch("lapis_pm.batched_fixer_eval.load_corpus", return_value=[fixture]), \
         patch("lapis_pm.code_oracle_run.run_code_oracle_experiment") as mock_run:
        mock_run.return_value = {
            "run_id": "custom-run", "fixture_id": "lapis-pm-66a7d2ea",
            "agent_backend_url": "http://x/v1", "oracle_outcome": "pass",
            "apply_status": "success", "agent_self_report": {"passed": 1},
        }
        code, _, _ = _run([
            "code-oracle-run",
            "--fixture-id", "lapis-pm-66a7d2ea",
            "--agent-backend-url", "http://x/v1",
            "--repo-path", str(repo),
            "--out-dir", str(out_dir),
            "--run-id", "custom-run",
            "--max-steps", "10",
            "--timeout", "99",
        ])

    assert code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args.args == ()
    assert mock_run.call_args.kwargs == {
        "fixture": fixture,
        "agent_backend_url": "http://x/v1",
        "repo_path": repo,
        "out_dir": out_dir,
        "run_id": "custom-run",
        "max_steps": 10,
        "timeout_s": 99,
    }


# ---------------------------------------------------------------------------
# AC5 - result surfaced (fixed field set) + exit 0
# ---------------------------------------------------------------------------


def test_ac5_prints_fixed_field_set_and_exits_0(tmp_path):
    repo, sha = _git_repo(tmp_path)
    fixture = _make_fixture(parent_sha=sha)
    out_dir = tmp_path / "out"

    result = {
        "run_id": "lapis-pm-66a7d2ea",
        "fixture_id": "lapis-pm-66a7d2ea",
        "agent_backend_url": "http://x/v1",
        "oracle_outcome": "pass",
        "apply_status": "success",
        "agent_self_report": {"passed": 3, "failed": 0},
        "final_diff": "SHOULD NOT BE PRINTED super-secret-diff-content",
        "transcript_path": str(out_dir / "lapis-pm-66a7d2ea.transcript.json"),
    }

    with patch("lapis_pm.batched_fixer_eval.load_corpus", return_value=[fixture]), \
         patch("lapis_pm.code_oracle_run.run_code_oracle_experiment", return_value=result):
        code, out, _ = _run([
            "code-oracle-run",
            "--fixture-id", "lapis-pm-66a7d2ea",
            "--agent-backend-url", "http://x/v1",
            "--repo-path", str(repo),
            "--out-dir", str(out_dir),
        ])

    assert code == 0
    assert "lapis-pm-66a7d2ea" in out
    assert "http://x/v1" in out
    assert "pass" in out
    assert "success" in out
    assert "{'passed': 3, 'failed': 0}" in out
    assert str(out_dir / "lapis-pm-66a7d2ea.json") in out
    assert "SHOULD NOT BE PRINTED" not in out
    assert "transcript" not in out.lower()


# ---------------------------------------------------------------------------
# AC6 - harness exceptions caught -> ERROR: <msg>, exit 1
# ---------------------------------------------------------------------------


def test_ac6_harness_exception_caught(tmp_path):
    repo, sha = _git_repo(tmp_path)
    fixture = _make_fixture(parent_sha=sha)

    with patch("lapis_pm.batched_fixer_eval.load_corpus", return_value=[fixture]), \
         patch(
             "lapis_pm.code_oracle_run.run_code_oracle_experiment",
             side_effect=RuntimeError("agent backend unreachable"),
         ):
        code, _, err = _run([
            "code-oracle-run",
            "--fixture-id", "lapis-pm-66a7d2ea",
            "--agent-backend-url", "http://x/v1",
            "--repo-path", str(repo),
        ])

    assert code == 1
    assert "ERROR: agent backend unreachable" in err


# ---------------------------------------------------------------------------
# AC7 - CLI performs no writes under repo_path; pre-flight guard
# ---------------------------------------------------------------------------


def test_ac7_no_writes_under_repo_path(tmp_path):
    repo, sha = _git_repo(tmp_path)
    fixture = _make_fixture(parent_sha=sha)

    before = sp.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    ).stdout
    before_listing = sorted(p.relative_to(repo) for p in repo.rglob("*"))

    with patch("lapis_pm.batched_fixer_eval.load_corpus", return_value=[fixture]), \
         patch("lapis_pm.code_oracle_run.run_code_oracle_experiment") as mock_run:
        mock_run.return_value = {
            "run_id": "lapis-pm-66a7d2ea", "fixture_id": "lapis-pm-66a7d2ea",
            "agent_backend_url": "http://x/v1", "oracle_outcome": "pass",
            "apply_status": "success", "agent_self_report": None,
        }
        code, _, _ = _run([
            "code-oracle-run",
            "--fixture-id", "lapis-pm-66a7d2ea",
            "--agent-backend-url", "http://x/v1",
            "--repo-path", str(repo),
        ])

    assert code == 0
    after = sp.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    ).stdout
    after_listing = sorted(p.relative_to(repo) for p in repo.rglob("*"))
    assert before == after == ""
    assert before_listing == after_listing


def test_ac7_preflight_rejects_invalid_repo_path(tmp_path):
    fixture = _make_fixture(parent_sha="deadbeef")
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    with patch("lapis_pm.batched_fixer_eval.load_corpus", return_value=[fixture]), \
         patch("lapis_pm.code_oracle_run.run_code_oracle_experiment") as mock_run:
        code, _, err = _run([
            "code-oracle-run",
            "--fixture-id", "lapis-pm-66a7d2ea",
            "--agent-backend-url", "http://x/v1",
            "--repo-path", str(not_a_repo),
        ])

    assert code == 1
    assert "not a valid git repo" in err
    mock_run.assert_not_called()


def test_ac7_preflight_rejects_missing_parent_sha(tmp_path):
    repo, sha = _git_repo(tmp_path)
    fixture = _make_fixture(parent_sha="0" * 40)

    with patch("lapis_pm.batched_fixer_eval.load_corpus", return_value=[fixture]), \
         patch("lapis_pm.code_oracle_run.run_code_oracle_experiment") as mock_run:
        code, _, err = _run([
            "code-oracle-run",
            "--fixture-id", "lapis-pm-66a7d2ea",
            "--agent-backend-url", "http://x/v1",
            "--repo-path", str(repo),
        ])

    assert code == 1
    assert "not found" in err
    mock_run.assert_not_called()
