"""Offline test for lapis_pm.code_oracle_run.

Exercises the REAL oracle + REAL git + REAL pytest against a self-contained
toy repo. The only thing stubbed is call_gw_agent (no network / no LLM call) -
the whole point of this unit is proving the clone -> worktree -> agent ->
oracle -> record -> reset composition works, not re-testing the oracle itself
(that's covered by test_batched_fixer_eval.py).
"""

import json
import subprocess as sp
from pathlib import Path

from lapis_pm.batched_fixer_eval import FixtureRecord
from lapis_pm.code_oracle_run import run_code_oracle_experiment


def _build_toy_repo(tmp_path, name="repo"):
    repo = tmp_path / name
    repo.mkdir()
    (repo / "lapis_pm").mkdir()
    sp.run(["git", "init", str(repo)], check=True, capture_output=True)
    sp.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "config", "user.name", "T"], cwd=repo, check=True, capture_output=True)
    (repo / "lapis_pm" / "__init__.py").write_text("")
    (repo / "lapis_pm" / "module.py").write_text("def foo():\n    return 'old'\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("")
    sp.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)
    parent_sha = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    return repo, parent_sha


GOLDEN_TEST_DIFF = (
    "diff --git a/tests/test_module.py b/tests/test_module.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n+++ b/tests/test_module.py\n"
    "@@ -0,0 +1,3 @@\n"
    "+from lapis_pm.module import foo\n"
    "+def test_foo_new():\n"
    "+    assert foo() == 'new'\n"
)

FIXING_DIFF = (
    "diff --git a/lapis_pm/module.py b/lapis_pm/module.py\n"
    "--- a/lapis_pm/module.py\n+++ b/lapis_pm/module.py\n"
    "@@ -1,2 +1,2 @@\n def foo():\n-    return 'old'\n+    return 'new'\n"
)


def _make_fixture(parent_sha):
    return FixtureRecord(
        repo="toy",
        sha="cafe0000cafe0001",
        parent_sha=parent_sha,
        pr_number=1,
        path="lapis_pm/module.py",
        file_loc="line 1-2",
        changed_lines=2,
        tier="T1",
        task_intent_raw="fix foo() to return 'new' instead of 'old'",
        task_intent_paraphrased="fix foo() to return 'new' instead of 'old'",
        checker_class="DISCRIMINATES",
        golden_test_diff=GOLDEN_TEST_DIFF,
        golden_test_ids=["tests/test_module.py::test_foo_new"],
        fail_first_confirmed=True,
        scoped_test_files=[],
        base_stable_fail_set=[],
        base_flaky_set=[],
    )


def _canned_agent(final_diff, last_test_outcome=None):
    """Build a call_gw_agent stub that returns a canned (fixer_result, transcript)
    without any network call. Signature matches code_oracle_run's kwarg-only call."""

    def _stub(*, prompt, system, cwd, writeable, backend_url, acquire_lease, timeout, max_steps, work_id, on_wake_fail):
        fixer_result = {
            "final_diff": final_diff,
            "last_test_outcome": last_test_outcome,
            "concluded": True,
            "max_steps_reached": False,
            "no_progress": False,
        }
        transcript = [{"step": 1, "note": "canned agent stub, no network call"}]
        return fixer_result, transcript

    return _stub


def test_run_code_oracle_experiment_pass_on_fixing_diff(tmp_path, monkeypatch):
    repo, parent_sha = _build_toy_repo(tmp_path)
    fixture = _make_fixture(parent_sha)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "lapis_pm.code_oracle_run.call_gw_agent",
        _canned_agent(FIXING_DIFF, last_test_outcome={"passed": 1, "failed": 0, "errors": 0}),
    )

    before_status = sp.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    ).stdout

    result = run_code_oracle_experiment(
        fixture, "http://fake-endpoint:9", repo_path=repo, out_dir=out_dir, run_id="run-pass",
    )

    assert result["oracle_outcome"] == "pass"
    assert result["apply_status"] == "success"
    assert result["run_id"] == "run-pass"
    assert result["fixture_id"] == "toy-cafe0000"
    assert result["agent_backend_url"] == "http://fake-endpoint:9"
    assert result["final_diff"] == FIXING_DIFF
    assert result["post_failures"] == []
    assert result["agent_self_report"] == {"passed": 1, "failed": 0, "errors": 0}
    assert result["agent_concluded"] is True
    assert result["agent_max_steps_reached"] is False
    assert result["agent_no_progress"] is False

    result_path = out_dir / "run-pass.json"
    transcript_path = out_dir / "run-pass.transcript.json"
    assert result_path.exists()
    assert transcript_path.exists()
    assert json.loads(result_path.read_text())["oracle_outcome"] == "pass"
    assert json.loads(transcript_path.read_text()) == [
        {"step": 1, "note": "canned agent stub, no network call"}
    ]
    assert result["transcript_path"] == str(transcript_path)

    after_status = sp.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert before_status == after_status == ""

    clone_path = Path(f"/tmp/bfe-{repo.name}-run-pass")
    assert not clone_path.exists()
    assert not (clone_path.parent / "agent_run-pass").exists()
    assert not (clone_path.parent / "oracle_run-pass").exists()


def test_run_code_oracle_experiment_non_pass_on_empty_diff(tmp_path, monkeypatch):
    repo, parent_sha = _build_toy_repo(tmp_path, name="repo2")
    fixture = _make_fixture(parent_sha)
    out_dir = tmp_path / "out2"

    # Canned self-report CLAIMS success even though the diff does not fix the
    # bug - the oracle verdict must be independent of this self-report.
    monkeypatch.setattr(
        "lapis_pm.code_oracle_run.call_gw_agent",
        _canned_agent("", last_test_outcome={"passed": 3, "failed": 0, "errors": 0}),
    )

    result = run_code_oracle_experiment(
        fixture, "http://fake-endpoint:9", repo_path=repo, out_dir=out_dir, run_id="run-fail",
    )

    assert result["oracle_outcome"] != "pass"
    assert result["agent_self_report"] == {"passed": 3, "failed": 0, "errors": 0}
    assert (out_dir / "run-fail.json").exists()
    assert (out_dir / "run-fail.transcript.json").exists()

    after_status = sp.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert after_status == ""

    clone_path = Path(f"/tmp/bfe-{repo.name}-run-fail")
    assert not clone_path.exists()
    assert not (clone_path.parent / "agent_run-fail").exists()
    assert not (clone_path.parent / "oracle_run-fail").exists()


def test_run_code_oracle_experiment_default_run_id(tmp_path, monkeypatch):
    repo, parent_sha = _build_toy_repo(tmp_path, name="repo3")
    fixture = _make_fixture(parent_sha)
    out_dir = tmp_path / "out3"

    monkeypatch.setattr(
        "lapis_pm.code_oracle_run.call_gw_agent",
        _canned_agent(FIXING_DIFF),
    )

    result = run_code_oracle_experiment(
        fixture, "http://fake-endpoint:9", repo_path=repo, out_dir=out_dir,
    )

    assert result["run_id"] == "toy-cafe0000"
    assert result["fixture_id"] == "toy-cafe0000"
    assert (out_dir / "toy-cafe0000.json").exists()
