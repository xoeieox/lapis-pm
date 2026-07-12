"""Unit tests for candidate-apply-robustness-v0 (SEARCH/REPLACE candidate format).

All tests are fully offline: swarm mocked, no GW, no live network, no real git-apply
dependency for the candidate itself (only apply_search_replace text substitution).
"""

import subprocess as sp
from dataclasses import asdict

import pytest

from lapis_pm.batched_fixer_eval import (
    FixtureRecord,
    apply_search_replace,
    dedicated_clone,
    generate_candidates,
    oracle_evaluate_candidate,
)


# ---------------------------------------------------------------------------
# apply_search_replace - pure function unit tests
# ---------------------------------------------------------------------------


def test_single_block_applies():
    original = "def foo():\n    return 'old'\n"
    candidate = (
        "<<<<<<< SEARCH\n"
        "def foo():\n    return 'old'\n"
        "=======\n"
        "def foo():\n    return 'new'\n"
        ">>>>>>> REPLACE\n"
    )
    new_text, status = apply_search_replace(original, candidate)
    assert status == "applied"
    assert new_text == "def foo():\n    return 'new'\n"


def test_multiple_blocks_all_apply():
    original = "a = 1\nb = 2\nc = 3\n"
    candidate = (
        "<<<<<<< SEARCH\n"
        "a = 1\n"
        "=======\n"
        "a = 100\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "c = 3\n"
        "=======\n"
        "c = 300\n"
        ">>>>>>> REPLACE\n"
    )
    new_text, status = apply_search_replace(original, candidate)
    assert status == "applied"
    assert new_text == "a = 100\nb = 2\nc = 300\n"


def test_search_not_found():
    original = "a = 1\n"
    candidate = (
        "<<<<<<< SEARCH\n"
        "does not exist\n"
        "=======\n"
        "replacement\n"
        ">>>>>>> REPLACE\n"
    )
    new_text, status = apply_search_replace(original, candidate)
    assert new_text is None
    assert status == "search-not-found"


def test_search_ambiguous():
    original = "x = 1\nx = 1\n"
    candidate = (
        "<<<<<<< SEARCH\n"
        "x = 1\n"
        "=======\n"
        "x = 2\n"
        ">>>>>>> REPLACE\n"
    )
    new_text, status = apply_search_replace(original, candidate)
    assert new_text is None
    assert status == "search-ambiguous"


def test_no_blocks_parsed_on_malformed_or_absent_fences():
    original = "a = 1\n"
    new_text, status = apply_search_replace(original, "this is not a diff at all\n")
    assert new_text is None
    assert status == "no-blocks-parsed"


def test_never_raises_on_garbage_input():
    # A grab-bag of adversarial inputs must always return a (None, reason) tuple.
    original = "a = 1\n"
    garbage_inputs = [
        "",
        "<<<<<<< SEARCH",
        ">>>>>>> REPLACE\n=======\n<<<<<<< SEARCH\n",
        "\x00\x01binary junk\xff",
    ]
    for g in garbage_inputs:
        new_text, status = apply_search_replace(original, g)
        assert new_text is None
        assert isinstance(status, str) and status


def test_malformed_block_missing_closing_marker():
    original = "a = 1\n"
    candidate = (
        "<<<<<<< SEARCH\n"
        "a = 1\n"
        "=======\n"
        "a = 2\n"
        # no closing >>>>>>> REPLACE
    )
    new_text, status = apply_search_replace(original, candidate)
    assert new_text is None
    assert status == "malformed-block"


def test_malformed_block_orphan_divider():
    original = "a = 1\n"
    candidate = "=======\na = 2\n>>>>>>> REPLACE\n"
    new_text, status = apply_search_replace(original, candidate)
    assert new_text is None
    assert status == "malformed-block"


def test_delimiter_collision_in_body_surfaces_as_recorded_failure():
    """A bare '=======' line inside a SEARCH body mis-parses the block boundary,
    but must surface as a recorded failure (search-not-found or malformed-block),
    never as a silently-corrupted 'applied' result (Sonnet MED gate)."""
    original = "line1\n=======\nline2\nline1\n"  # 'line1' appears twice: ambiguous/not-unique
    candidate = (
        "<<<<<<< SEARCH\n"
        "line1\n=======\nline2\n"
        "=======\n"
        "replacement\n"
        ">>>>>>> REPLACE\n"
    )
    new_text, status = apply_search_replace(original, candidate)
    assert new_text is None
    assert status in ("search-not-found", "search-ambiguous", "malformed-block")


def test_multi_block_sequential_second_targets_first_result():
    """Block 2's SEARCH depends on text left behind by block 1 (Sonnet MED 2026-07-10 gate)."""
    original = "def foo():\n    return 'old'\n"
    candidate = (
        "<<<<<<< SEARCH\n"
        "return 'old'\n"
        "=======\n"
        "return 'intermediate'\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "return 'intermediate'\n"
        "=======\n"
        "return 'final'\n"
        ">>>>>>> REPLACE\n"
    )
    new_text, status = apply_search_replace(original, candidate)
    assert status == "applied"
    assert new_text == "def foo():\n    return 'final'\n"


def test_multi_block_first_failure_fails_whole_candidate():
    original = "a = 1\nb = 2\n"
    candidate = (
        "<<<<<<< SEARCH\n"
        "does not exist\n"
        "=======\n"
        "a = 100\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "b = 2\n"
        "=======\n"
        "b = 200\n"
        ">>>>>>> REPLACE\n"
    )
    new_text, status = apply_search_replace(original, candidate)
    assert new_text is None
    assert status == "search-not-found"


# ---------------------------------------------------------------------------
# generate_candidates mock-mode format parity
# ---------------------------------------------------------------------------


def test_mock_generate_candidates_are_search_replace_and_apply():
    fixture = FixtureRecord(
        repo="test", sha="sha123", parent_sha="parent",
        pr_number=1, path="test.py", file_loc="unknown", changed_lines=1, tier="T1",
        pre_state_slice="def foo():\n    return 'old'\n",
    )
    candidates = generate_candidates(fixture, n=2, mock_mode=True)
    assert len(candidates) == 2
    for c, retries in candidates:
        assert c is not None
        assert retries == 0
        assert "<<<<<<< SEARCH" in c
        new_text, status = apply_search_replace(fixture.pre_state_slice, c)
        assert status == "applied", f"mock candidate rejected: {status}"
        assert new_text is not None


# ---------------------------------------------------------------------------
# oracle_evaluate_candidate - apply path via search/replace
# ---------------------------------------------------------------------------


def _build_toy_discriminates_fixture():
    return FixtureRecord(
        repo="lapis-pm", sha="feedfeed0001", parent_sha="placeholder",
        pr_number=None, path="lapis_pm/module.py", file_loc="unknown",
        changed_lines=1, tier="T1",
        task_intent_paraphrased="Fix foo() to return the new value",
        pre_state_slice="def foo():\n    return 'old'\n",
        golden_source_diff=(
            "diff --git a/lapis_pm/module.py b/lapis_pm/module.py\n"
            "--- a/lapis_pm/module.py\n+++ b/lapis_pm/module.py\n"
            "@@ -1,2 +1,2 @@\n def foo():\n-    return 'old'\n+    return 'new'\n"
        ),
        golden_test_diff=(
            "diff --git a/tests/test_module.py b/tests/test_module.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n+++ b/tests/test_module.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+from lapis_pm.module import foo\n"
            "+def test_foo_new():\n"
            "+    assert foo() == 'new'\n"
        ),
        golden_test_ids=["tests/test_module.py::test_foo_new"],
        fail_first_confirmed=True,
        checker_class="DISCRIMINATES",
        scoped_test_files=[],
        base_stable_fail_set=[],
        base_flaky_set=[],
    )


def _make_toy_worktree_repo(tmp_path, name):
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


def test_oracle_search_replace_candidate_reaches_test_step(tmp_path):
    fixture = _build_toy_discriminates_fixture()
    repo, parent_sha = _make_toy_worktree_repo(tmp_path, "repo_sr_pass")
    fixture = FixtureRecord(**{**asdict(fixture), "parent_sha": parent_sha})

    candidate = (
        "<<<<<<< SEARCH\n"
        "def foo():\n    return 'old'\n"
        "=======\n"
        "def foo():\n    return 'new'\n"
        ">>>>>>> REPLACE\n"
    )

    with dedicated_clone(repo, "test-sr-pass") as clone_path:
        status, failures, outcome = oracle_evaluate_candidate(clone_path, fixture, candidate, "wt_sr_pass")

    assert status == "success"
    assert outcome == "pass"


def test_oracle_non_matching_candidate_returns_failed(tmp_path):
    fixture = _build_toy_discriminates_fixture()
    repo, parent_sha = _make_toy_worktree_repo(tmp_path, "repo_sr_fail")
    fixture = FixtureRecord(**{**asdict(fixture), "parent_sha": parent_sha})

    candidate = (
        "<<<<<<< SEARCH\n"
        "this text does not exist in module.py\n"
        "=======\n"
        "def foo():\n    return 'new'\n"
        ">>>>>>> REPLACE\n"
    )

    with dedicated_clone(repo, "test-sr-fail") as clone_path:
        status, failures, outcome = oracle_evaluate_candidate(clone_path, fixture, candidate, "wt_sr_fail")

    assert status == "failed"


# ---------------------------------------------------------------------------
# Leak-guard regression (prompt side + output side)
# ---------------------------------------------------------------------------


def test_leak_guard_prompt_side_still_raises():
    fixture = FixtureRecord(
        repo="test", sha="leaksha1", parent_sha="parent",
        pr_number=1, path="module.py", file_loc="unknown", changed_lines=5, tier="T1",
        task_intent_paraphrased="Fix the bug",
        pre_state_slice=(
            "diff --git a/tests/test_m.py b/tests/test_m.py\n"
            "--- /dev/null\n+++ b/tests/test_m.py\n"
            "+def test_secret_oracle():\n+    assert True\n"
        ),
        golden_test_diff=(
            "diff --git a/tests/test_m.py b/tests/test_m.py\n"
            "--- /dev/null\n+++ b/tests/test_m.py\n"
            "+def test_secret_oracle():\n+    assert True\n"
        ),
        golden_test_ids=["tests/test_m.py::test_secret_oracle"],
        checker_class="DISCRIMINATES",
        fail_first_confirmed=True,
    )
    with pytest.raises(RuntimeError):
        generate_candidates(fixture, n=1, mock_mode=False)


def test_leak_guard_output_side_drops_candidate(monkeypatch):
    """When the swarm returns a candidate containing a golden_test_id verbatim,
    generate_candidates must drop it to None (recorded as a none/health signal),
    never surface it as an applied candidate."""
    fixture = FixtureRecord(
        repo="test", sha="leaksha2", parent_sha="parent",
        pr_number=1, path="module.py", file_loc="unknown", changed_lines=5, tier="T1",
        task_intent_paraphrased="Fix the bug",
        pre_state_slice="def foo():\n    return 'old'\n",
        golden_test_diff="",
        golden_test_ids=["tests/test_m.py::test_secret_oracle"],
        checker_class="DISCRIMINATES",
        fail_first_confirmed=True,
    )

    leaking_candidate = (
        "<<<<<<< SEARCH\n"
        "def foo():\n    return 'old'\n"
        "=======\n"
        "def foo():\n    return 'new'  # tests/test_m.py::test_secret_oracle\n"
        ">>>>>>> REPLACE\n"
    )

    def fake_call_swarm(prompts, **kwargs):
        return [leaking_candidate for _ in prompts]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    candidates = generate_candidates(fixture, n=2, mock_mode=False)
    assert len(candidates) == 2
    assert all(c is None for c, _ in candidates)
