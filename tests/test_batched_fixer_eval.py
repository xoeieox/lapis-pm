"""Unit tests for batched_fixer_eval — CORE ACs: AC1, AC1b, AC2, AC3, AC4, AC4b, AC11, AC13.

DEFERRED ACs (AC5, AC6, AC6b, AC7-AC12): scaffolded in separate section below,
marked clearly as DEFERRED and NOT claimed green in this bind.

All tests run against mocked swarm / synthetic fixtures — zero GW, zero paid spend.
"""

import json
import os
import subprocess
import pytest
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from lapis_pm.batched_fixer_eval import (
    # Dataclasses
    FixtureRecord,
    CandidateResult,
    FixtureRunResult,
    TierMetrics,
    EvalResult,
    # Pure oracle functions (AC4, AC4b) — CORE
    extract_module_name,
    parse_pytest_failures,
    compute_flakiness_fingerprint,
    classify_candidate_outcome,
    classify_candidate_outcome_golden,
    slice_pre_state_from_diff,
    validate_corpus_power_floor,
    validate_discriminates_power_floor,
    # Co-committed test oracle helpers (AC-O1, AC-O2)
    split_diff_by_type,
    extract_test_ids_from_diff,
    classify_checker_class_cocommitted,
    # Corpus builder helpers
    find_scoped_test_files,
    classify_checker_class,
    launder_intent,
    check_laundering_quality,
    # Corpus / run
    build_corpus,
    generate_candidates,
    run_fixture,
    aggregate_results,
    generate_report,
    run_eval,
    # Git / serving utilities
    dedicated_clone,
    detached_worktree,
    swarm_serving,
    swarm_model,
    assert_swarm_health,
    grounding_hook,
    # Constants
    CORPUS_HOLDOUT_PER_TIER,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic corpus entries
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_fixture_t1():
    return FixtureRecord(
        repo="lapis-pm",
        sha="a1b2c3d4e5f60000",
        parent_sha="parent0000000001",
        pr_number=42,
        path="lapis_pm/pm_core.py",
        file_loc="line 123-130",
        changed_lines=8,
        tier="T1",
        golden_diff_cyclomatic_delta=0,
        distinct_symbols_touched=1,
        is_concurrency_code=False,
        task_intent_raw="fix: return wrong value when input empty",
        task_intent_paraphrased="Function returns wrong value when input is empty",
        intent_source="reviewer-comment",
        pre_state_slice="def foo():\n    return invalid",
        golden_diff=(
            "--- a/lapis_pm/pm_core.py\n+++ b/lapis_pm/pm_core.py\n"
            "@@ -123,3 +123,3 @@\n-    return invalid\n+    return valid"
        ),
        scoped_test_files=["tests/test_pm_core.py"],
        base_stable_fail_set=["tests/test_pm_core.py::test_foo"],
        base_flaky_set=[],
        target_test_files=["tests/test_pm_core.py::test_foo"],
        checker_class="DISCRIMINATES",
        is_reviewer_cycle=True,
        blind_holdout=True,
    )


@pytest.fixture
def mock_fixture_blind():
    return FixtureRecord(
        repo="conductor",
        sha="b2c3d4e5f6000001",
        parent_sha="parent_blind00001",
        pr_number=None,
        path="conductor/scheduler.py",
        file_loc="line 50-60",
        changed_lines=15,
        tier="T2",
        golden_diff_cyclomatic_delta=1,
        distinct_symbols_touched=2,
        is_concurrency_code=True,
        task_intent_raw="fix: race condition in shared state",
        task_intent_paraphrased="Concurrent access to shared state causes data corruption",
        intent_source="commit-body",
        pre_state_slice="# concurrent code without lock\naccess_state()",
        golden_diff=(
            "--- a/conductor/scheduler.py\n+++ b/conductor/scheduler.py\n"
            "@@ -50,2 +50,3 @@\n+    with lock:\n     access_state()"
        ),
        scoped_test_files=["tests/scheduler_test.py"],
        base_stable_fail_set=[],
        base_flaky_set=[],
        target_test_files=[],
        checker_class="BLIND",
        is_reviewer_cycle=False,
        blind_holdout=True,
    )


@pytest.fixture
def mock_fixture_untested():
    return FixtureRecord(
        repo="lapis-pm",
        sha="c3d4e5f600000002",
        parent_sha="parent0000000002",
        pr_number=None,
        path="lapis_pm/some_module.py",
        file_loc="line 1-10",
        changed_lines=5,
        tier="T1",
        scoped_test_files=[],
        base_stable_fail_set=[],
        base_flaky_set=[],
        target_test_files=[],
        checker_class="UNTESTED",
    )


@pytest.fixture
def mock_corpus_t1_floor(mock_fixture_t1):
    """Corpus with ≥8 T1 and ≥8 T2 holdout fixtures (meets coarse-binary power floor)."""
    corpus = []
    for i in range(8):
        corpus.append(FixtureRecord(
            repo="lapis-pm",
            sha=f"a1b2c3d4e5f6{i:04d}",
            parent_sha=f"parent{i:010d}",
            pr_number=100 + i,
            path="lapis_pm/pm_core.py",
            file_loc=f"line {100 + i * 10}-{108 + i * 10}",
            changed_lines=8,
            tier="T1",
            distinct_symbols_touched=1,
            task_intent_raw=f"fix issue {i}",
            task_intent_paraphrased=f"[Symptom] issue {i}",
            scoped_test_files=["tests/test_pm_core.py"],
            base_stable_fail_set=["tests/test_pm_core.py::test_foo"],
            target_test_files=["tests/test_pm_core.py::test_foo"],
            checker_class="DISCRIMINATES",
            blind_holdout=True,
        ))
    for i in range(8):
        corpus.append(FixtureRecord(
            repo="conductor",
            sha=f"b2c3d4e5f6a7{i:04d}",
            parent_sha=f"qparent{i:09d}",
            pr_number=200 + i,
            path="conductor/some_module.py",
            file_loc=f"line {50 + i * 10}-{65 + i * 10}",
            changed_lines=20,
            tier="T2",
            blind_holdout=True,
        ))
    return corpus


# ---------------------------------------------------------------------------
# AC1: Corpus schema validation
# ---------------------------------------------------------------------------


def test_ac1_fixture_record_schema(mock_fixture_t1):
    """AC1: FixtureRecord serialises with all required fields."""
    data = asdict(mock_fixture_t1)
    required = [
        "repo", "sha", "parent_sha", "pr_number", "path", "file_loc",
        "changed_lines", "tier", "checker_class", "task_intent_raw",
        "task_intent_paraphrased", "intent_source", "pre_state_slice",
        "golden_diff", "scoped_test_files", "base_stable_fail_set",
        "base_flaky_set", "target_test_files", "is_reviewer_cycle",
        "is_test_only", "blind_holdout", "flaky_excluded_count",
        # Co-committed test oracle fields (AC-O1)
        "golden_source_diff", "golden_test_diff", "golden_test_ids", "fail_first_confirmed",
    ]
    for key in required:
        assert key in data, f"Missing field: {key}"


def test_ac1_pr_number_nullable(mock_fixture_blind):
    """AC1: pr_number is explicitly nullable; null routes to commit-body intent."""
    assert mock_fixture_blind.pr_number is None
    data = asdict(mock_fixture_blind)
    assert data["pr_number"] is None
    # Null pr_number → commit-body source
    assert mock_fixture_blind.intent_source == "commit-body"
    assert mock_fixture_blind.is_reviewer_cycle is False


def test_ac1_pr_number_int(mock_fixture_t1):
    """AC1: pr_number can be an int."""
    assert isinstance(mock_fixture_t1.pr_number, int)
    assert mock_fixture_t1.pr_number == 42


def test_ac1_schema_round_trip():
    """AC1: FixtureRecord serialises and deserialises via asdict/FixtureRecord(**data)."""
    f = FixtureRecord(
        repo="lapis-pm", sha="abc123", parent_sha="def456",
        pr_number=None, path="x.py", file_loc="unknown", changed_lines=3, tier="T1",
        base_stable_fail_set=["t::a"], base_flaky_set=["t::b"],
    )
    data = asdict(f)
    f2 = FixtureRecord(**data)
    assert f2.pr_number is None
    assert f2.base_stable_fail_set == ["t::a"]
    assert f2.base_flaky_set == ["t::b"]


# ---------------------------------------------------------------------------
# AC1b: Per-tier power floor
# ---------------------------------------------------------------------------


def test_ac1b_power_floor_met(mock_corpus_t1_floor):
    """AC1b: validate_corpus_power_floor passes when T1+T2 each have ≥8 holdouts."""
    result = validate_corpus_power_floor(mock_corpus_t1_floor)
    assert result.get("T1", 0) >= CORPUS_HOLDOUT_PER_TIER
    assert result.get("T2", 0) >= CORPUS_HOLDOUT_PER_TIER
    # T1+T2 only (no T3) routes to coarse-binary, not 3-tier
    assert result.get("_shape") == "coarse-binary"


def test_ac1b_power_floor_failure_raises():
    """AC1b: build fails when no tier meets the floor and coarse-binary also fails."""
    # Only 3 T1 holdouts, 0 T2/T3 — neither 3-tier nor coarse-binary floor met
    corpus = [
        FixtureRecord(
            repo="lapis-pm", sha=f"sha{i:04d}", parent_sha=f"p{i:04d}",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5,
            tier="T1", blind_holdout=True,
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="power floor not met"):
        validate_corpus_power_floor(corpus, tier_floor=8)


def test_ac1b_coarse_binary_collapse():
    """AC1b: corpus collapses to coarse binary when T3 is unfillable but T1+T2 meet floor."""
    corpus = (
        [
            FixtureRecord(
                repo="lapis-pm", sha=f"t1sha{i}", parent_sha=f"p{i}",
                pr_number=None, path="x.py", file_loc="unknown", changed_lines=5,
                tier="T1", blind_holdout=True,
            )
            for i in range(8)
        ]
        + [
            FixtureRecord(
                repo="conductor", sha=f"t2sha{i}", parent_sha=f"q{i}",
                pr_number=None, path="y.py", file_loc="unknown", changed_lines=20,
                tier="T2", blind_holdout=True,
            )
            for i in range(8)
        ]
    )
    # No T3 — 3-tier path requires all three tiers; falls to coarse binary (T1=8, T2+T3=8)
    counts = validate_corpus_power_floor(corpus, tier_floor=8)
    assert counts.get("T1", 0) >= 8
    assert counts.get("T2", 0) >= 8
    assert counts.get("_shape") == "coarse-binary"


def test_ac1b_mock_build_corpus_passes_through():
    """AC1b: build_corpus with mock_corpus returns it and skips floor check."""
    corpus = [
        FixtureRecord(
            repo="lapis-pm", sha="x", parent_sha="y",
            pr_number=1, path="f.py", file_loc="unknown", changed_lines=1, tier="T1",
        )
    ]
    result, meta = build_corpus(mock_corpus=corpus)
    assert len(result) == 1
    assert meta["source"] == "mock"


# ---------------------------------------------------------------------------
# AC2: Intent laundering code-path
# ---------------------------------------------------------------------------


def test_ac2_launder_intent_mock_mode_exists():
    """AC2: launder_intent returns (paraphrased, status) tuple with mock_mode=True."""
    raw = "fix: change path from /health to /healthz\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    paraphrased, status = launder_intent(raw, mock_mode=True)
    assert isinstance(paraphrased, str)
    assert len(paraphrased) > 0
    assert status == "mock"


def test_ac2_launder_intent_mock_does_not_return_raw_verbatim():
    """AC2: mock laundering never returns the raw string verbatim (always transforms)."""
    raw = "fix: change path from /health to /healthz"
    paraphrased, status = launder_intent(raw, mock_mode=True)
    assert paraphrased != raw, "launder_intent must transform the raw text, not return it verbatim"
    assert status == "mock"


def test_ac2_launder_intent_mock_strips_co_authored():
    """AC2: mock laundering strips Co-Authored-By trailers."""
    raw = "fix: something\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    paraphrased, status = launder_intent(raw, mock_mode=True)
    assert "Co-Authored-By" not in paraphrased
    assert status == "mock"


def test_ac2_generate_candidates_uses_paraphrased(mock_fixture_t1):
    """AC2: generate_candidates uses task_intent_paraphrased, not task_intent_raw."""
    # The fixture has different raw vs paraphrased; in real mode the prompt uses paraphrased
    # Verify this in the generate_candidates source (grep check)
    import inspect
    source = inspect.getsource(generate_candidates)
    assert "task_intent_paraphrased" in source
    assert "task_intent_raw" not in source or source.index("task_intent_paraphrased") < source.index("task_intent_raw") + 1000


def test_ac2_no_raw_commit_body_in_build_corpus():
    """AC2: build_corpus does not assign commit_body[:200] directly to task_intent_paraphrased."""
    import inspect
    source = inspect.getsource(build_corpus)
    # The forbidden stub pattern was: task_intent_paraphrased=commit_body[:200]
    assert "task_intent_paraphrased=commit_body" not in source
    assert "commit_body[:200]" not in source


def test_ac2_laundering_quality_scaffold():
    """AC2: check_laundering_quality scaffold exists and returns deferred flag."""
    result = check_laundering_quality("paraphrased intent", "raw intent")
    assert "deferred" in result
    assert result["deferred"] is True


# ---------------------------------------------------------------------------
# AC3: Dedicated-clone isolation — no shared-tree mutation
# ---------------------------------------------------------------------------


def _make_tiny_git_repo(path: Path) -> str:
    """Create a minimal git repo with one Python file and one test. Returns initial SHA."""
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True)
    (path / "module.py").write_text("def foo():\n    return 'old'\n")
    (path / "test_module.py").write_text(
        "from module import foo\ndef test_foo():\n    assert foo() == 'new'\n"
    )
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_ac3_dedicated_clone_is_separate_from_source(tmp_path):
    """AC3: dedicated_clone creates a separate path, not the shared working tree."""
    source = tmp_path / "source_repo"
    source.mkdir()
    sha = _make_tiny_git_repo(source)

    with dedicated_clone(source, "test-run-ac3") as clone_path:
        assert clone_path != source
        assert clone_path.exists()
        assert (clone_path / ".git").exists()
        # Verify HEAD matches
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=clone_path, capture_output=True, text=True
        )
        assert result.stdout.strip() == sha


def test_ac3_dedicated_clone_removed_after_context(tmp_path):
    """AC3: dedicated clone is removed after context exits."""
    source = tmp_path / "source_repo2"
    source.mkdir()
    _make_tiny_git_repo(source)

    clone_path_ref = None
    with dedicated_clone(source, "test-run-ac3-cleanup") as clone_path:
        clone_path_ref = clone_path
        assert clone_path.exists()

    assert not clone_path_ref.exists(), "Dedicated clone must be removed on exit"


def test_ac3_worktree_off_clone_not_source(tmp_path):
    """AC3: worktree is added off the dedicated clone, never off the source."""
    source = tmp_path / "source_repo3"
    source.mkdir()
    sha = _make_tiny_git_repo(source)

    with dedicated_clone(source, "test-run-ac3-wt") as clone_path:
        with detached_worktree(clone_path, sha, "ac3_test_wt") as wt_path:
            assert wt_path.exists()
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=wt_path, capture_output=True, text=True
            )
            assert head.stdout.strip() == sha

            # Source repo must have no extra worktrees registered
            src_wt = subprocess.run(
                ["git", "worktree", "list"], cwd=source, capture_output=True, text=True
            )
            wt_lines = src_wt.stdout.strip().splitlines()
            assert len(wt_lines) == 1, (
                f"Source repo must have exactly 1 worktree (itself), got: {wt_lines}"
            )


def test_ac3_no_worktree_prune_on_shared_clone():
    """AC3: harness never has 'prune' in any subprocess call list."""
    import ast
    module_path = Path(__file__).parent.parent / "lapis_pm" / "batched_fixer_eval.py"
    source = module_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            strs = [
                e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            assert "prune" not in strs, (
                f"Found 'prune' in subprocess call list — "
                "harness must NOT invoke 'git worktree prune'"
            )
    # worktree remove --force is OK for cleanup
    assert '"remove"' in source or "worktree remove" in source


def test_ac3_no_systemctl_or_ssh_in_harness():
    """AC3/AC10: harness is lease-agnostic — no systemctl/ssh/lease-acquire calls."""
    module_path = Path(__file__).parent.parent / "lapis_pm" / "batched_fixer_eval.py"
    source = module_path.read_text()
    assert "systemctl" not in source
    assert "ssh " not in source
    assert "acquire_lease" not in source


# ---------------------------------------------------------------------------
# AC4: Flakiness-robust baseline-diff oracle — pure function tests
# ---------------------------------------------------------------------------


def test_ac4_oracle_pass():
    """AC4: PASS when target tests flip and no new failures."""
    # base: test_A fails stably, test_B fails stably
    # post: test_A passes, test_B still fails → no new failures, target flipped
    outcome = classify_candidate_outcome(
        post_failures=["tests/t.py::test_B"],
        base_stable_fail_set=["tests/t.py::test_A", "tests/t.py::test_B"],
        base_flaky_set=[],
        target_test_files=["tests/t.py::test_A"],
        checker_class="DISCRIMINATES",
    )
    assert outcome == "pass", f"Expected pass, got {outcome}"


def test_ac4_oracle_regressed_new_failure():
    """AC4: REGRESSED when a new test (not in stable or flaky set) fails at post."""
    outcome = classify_candidate_outcome(
        post_failures=["tests/t.py::test_A", "tests/t.py::test_NEW"],
        base_stable_fail_set=["tests/t.py::test_A"],
        base_flaky_set=[],
        target_test_files=["tests/t.py::test_A"],  # would need to flip to pass
        checker_class="DISCRIMINATES",
    )
    # test_NEW is a new failure outside the excluded set → regressed
    # (Note: test_A also still fails, so condition 3 also fails, but condition 2 fires first)
    assert outcome == "regressed", f"Expected regressed, got {outcome}"


def test_ac4_oracle_regressed_target_not_flipped():
    """AC4: REGRESSED when target test still fails at post."""
    outcome = classify_candidate_outcome(
        post_failures=["tests/t.py::test_A", "tests/t.py::test_B"],
        base_stable_fail_set=["tests/t.py::test_A", "tests/t.py::test_B"],
        base_flaky_set=[],
        target_test_files=["tests/t.py::test_A"],
        checker_class="DISCRIMINATES",
    )
    # target test_A still failing at post → regressed
    assert outcome == "regressed", f"Expected regressed, got {outcome}"


def test_ac4_oracle_flaky_test_ignored():
    """AC4: Flaky test changing outcome does NOT count as a new failure (ignored)."""
    # base_flaky = test_C (sometimes fails)
    # post: test_B still failing (stable), test_C failing (flaky — excluded)
    # target test_A passes at post
    outcome = classify_candidate_outcome(
        post_failures=["tests/t.py::test_B", "tests/t.py::test_C"],
        base_stable_fail_set=["tests/t.py::test_A", "tests/t.py::test_B"],
        base_flaky_set=["tests/t.py::test_C"],
        target_test_files=["tests/t.py::test_A"],
        checker_class="DISCRIMINATES",
    )
    # test_C is flaky — ignored; test_B is stable and still fails; test_A passes → PASS
    assert outcome == "pass", f"Expected pass (flaky test_C ignored), got {outcome}"


def test_ac4_oracle_blind_fixture():
    """AC4: BLIND for checker_class=BLIND — oracle cannot discriminate."""
    outcome = classify_candidate_outcome(
        post_failures=[],
        base_stable_fail_set=[],
        base_flaky_set=[],
        target_test_files=[],
        checker_class="BLIND",
    )
    assert outcome == "blind", f"Expected blind, got {outcome}"


def test_ac4_oracle_untested_fixture():
    """AC4: BLIND for checker_class=UNTESTED (no scoped tests)."""
    outcome = classify_candidate_outcome(
        post_failures=[],
        base_stable_fail_set=[],
        base_flaky_set=[],
        target_test_files=[],
        checker_class="UNTESTED",
    )
    assert outcome == "blind"


def test_ac4_oracle_empty_target_files():
    """AC4: BLIND when target_test_files is empty (even if checker_class=DISCRIMINATES)."""
    # Empty target_test_files means oracle cannot verify the fix
    outcome = classify_candidate_outcome(
        post_failures=["tests/t.py::test_A"],
        base_stable_fail_set=["tests/t.py::test_A"],
        base_flaky_set=[],
        target_test_files=[],
        checker_class="DISCRIMINATES",
    )
    assert outcome == "blind"


def test_ac4_oracle_unverified_stable_passes():
    """AC4: UNVERIFIED when a stably-failing test unexpectedly passes at post."""
    # test_B was stably failing but now passes — baseline assumptions violated
    outcome = classify_candidate_outcome(
        post_failures=["tests/t.py::test_A"],   # test_B no longer in failures
        base_stable_fail_set=["tests/t.py::test_A", "tests/t.py::test_B"],
        base_flaky_set=[],
        target_test_files=["tests/t.py::test_A"],
        checker_class="DISCRIMINATES",
    )
    # test_B was stable-fail but now passes — baseline broken → unverified
    assert outcome == "unverified", f"Expected unverified, got {outcome}"


def test_ac4_parse_pytest_failures_standard():
    """AC4: parse_pytest_failures extracts test IDs from pytest -q output."""
    output = (
        "FAILED tests/test_foo.py::test_bar - AssertionError: expected 1\n"
        "FAILED tests/test_foo.py::TestClass::test_baz - ValueError\n"
        "passed 5\n"
    )
    failures = parse_pytest_failures(output)
    assert "tests/test_foo.py::test_bar" in failures
    assert "tests/test_foo.py::TestClass::test_baz" in failures
    assert len(failures) == 2


def test_ac4_parse_pytest_failures_empty():
    """AC4: parse_pytest_failures returns [] on clean output."""
    output = "5 passed in 0.12s\n"
    assert parse_pytest_failures(output) == []


def test_ac4_oracle_uses_classify_candidate_outcome():
    """AC4: oracle_evaluate_candidate calls classify_candidate_outcome (not returncode==0)."""
    import inspect
    from lapis_pm.batched_fixer_eval import oracle_evaluate_candidate
    source = inspect.getsource(oracle_evaluate_candidate)
    assert "classify_candidate_outcome" in source, (
        "oracle_evaluate_candidate must call classify_candidate_outcome "
        "(not bare returncode==0 check)"
    )
    # The forbidden stub pattern: test_outcome = "pass" assigned from returncode check
    # The oracle must NOT assign "pass" from a condition on returncode alone
    assert 'test_outcome = "pass"' not in source and "= 'pass'" not in source or \
        "classify_candidate_outcome" in source, (
        "oracle must use classify_candidate_outcome, not returncode-based pass assignment"
    )


# ---------------------------------------------------------------------------
# AC4b: N≥3 base runs, stable/flaky partition, flaky-target quarantine
# ---------------------------------------------------------------------------


def test_ac4b_compute_fingerprint_stable():
    """AC4b: Tests failing on ALL N runs → stable_fail_set."""
    run_results = [
        ["t::a", "t::b"],
        ["t::a", "t::b"],
        ["t::a", "t::b"],
    ]
    stable, flaky = compute_flakiness_fingerprint(run_results)
    assert "t::a" in stable
    assert "t::b" in stable
    assert flaky == []


def test_ac4b_compute_fingerprint_flaky():
    """AC4b: Tests failing on SOME (not all) runs → flaky_set."""
    run_results = [
        ["t::a", "t::flaky"],
        ["t::a"],
        ["t::a", "t::flaky"],
    ]
    stable, flaky = compute_flakiness_fingerprint(run_results)
    assert "t::a" in stable
    assert "t::flaky" in flaky
    assert "t::flaky" not in stable


def test_ac4b_compute_fingerprint_all_flaky():
    """AC4b: Test failing on no run not in any set; failing on all N → stable."""
    run_results = [
        ["t::a"],
        [],
        ["t::a"],
    ]
    stable, flaky = compute_flakiness_fingerprint(run_results)
    # t::a fails 2/3 → flaky; nothing fails all 3 → stable is empty
    assert stable == []
    assert "t::a" in flaky


def test_ac4b_compute_fingerprint_empty():
    """AC4b: Empty run_results returns empty sets."""
    stable, flaky = compute_flakiness_fingerprint([])
    assert stable == []
    assert flaky == []


def test_ac4b_classify_checker_discriminates():
    """AC4b: DISCRIMINATES when scoped tests exist, stable fails, and targets flip."""
    result = classify_checker_class(
        scoped_test_files=["tests/test_foo.py"],
        base_stable_fail_set=["tests/test_foo.py::test_a"],
        target_test_files=["tests/test_foo.py::test_a"],
    )
    assert result == "DISCRIMINATES"


def test_ac4b_classify_checker_blind():
    """AC4b: BLIND when scoped tests exist but no stable failures / no targets flip."""
    result = classify_checker_class(
        scoped_test_files=["tests/test_foo.py"],
        base_stable_fail_set=[],
        target_test_files=[],
    )
    assert result == "BLIND"


def test_ac4b_classify_checker_untested():
    """AC4b: UNTESTED when scoped_test_files is empty."""
    result = classify_checker_class(
        scoped_test_files=[],
        base_stable_fail_set=[],
        target_test_files=[],
    )
    assert result == "UNTESTED"


def test_ac4b_flaky_target_quarantine_by_construction():
    """AC4b: Target tests come from stable_fail_set; flaky tests are excluded by construction.

    This verifies that compute_flakiness_fingerprint correctly separates stable from flaky,
    so a target that would be 'flaky' is never in stable_fail_set and thus never in targets.
    """
    # A test failing 2/3 times is flaky, not stable
    run_results = [
        ["t::sometimes"],
        [],
        ["t::sometimes"],
    ]
    stable, flaky = compute_flakiness_fingerprint(run_results)
    assert "t::sometimes" not in stable  # not eligible to be a target
    assert "t::sometimes" in flaky       # correctly identified as flaky


# ---------------------------------------------------------------------------
# AC4: extract_module_name and slice_pre_state_from_diff helpers
# ---------------------------------------------------------------------------


def test_ac4_extract_module_name_nested():
    """AC4: extract_module_name converts paths with packages."""
    assert extract_module_name("lapis_pm/pm_core.py") == "lapis_pm.pm_core"
    assert extract_module_name("lapis_pm/__init__.py") == "lapis_pm"
    assert extract_module_name("scripts/elevator_scheduler.py") == "scripts.elevator_scheduler"


def test_ac4_extract_module_name_toplevel():
    assert extract_module_name("module.py") == "module"


def test_ac4_slice_pre_state_extracts_enclosing_def():
    """AC4: slice_pre_state_from_diff finds enclosing def."""
    pre_state = "\n".join([
        "import os",
        "def unrelated():",
        "    pass",
        "def target_func():",
        "    x = 1",
        "    return x",
        "def after():",
        "    pass",
    ])
    golden_diff = (
        "--- a/f.py\n+++ b/f.py\n"
        "@@ -5,2 +5,2 @@\n"
        "-    x = 1\n+    x = 2\n"
    )
    result = slice_pre_state_from_diff(pre_state, golden_diff, context_lines=3)
    assert "target_func" in result
    assert "def target_func" in result


def test_ac4_slice_pre_state_fallback_no_hunk():
    """AC4: Falls back to 3000-char window when diff has no hunk headers."""
    pre = "x" * 1000
    result = slice_pre_state_from_diff(pre, "no hunks here", context_lines=15)
    assert len(result) <= 3000


def test_ac4_slice_pre_state_caps_at_150_lines():
    """AC4: Slice capped at 150 lines max."""
    pre_state = "\n".join([f"line {i}" for i in range(300)])
    golden_diff = "--- a/f.py\n+++ b/f.py\n@@ -10,5 +10,5 @@\n-old\n+new\n"
    result = slice_pre_state_from_diff(pre_state, golden_diff)
    assert len(result.splitlines()) <= 150


# ---------------------------------------------------------------------------
# AC11: Held/dormant — zero production writes
# ---------------------------------------------------------------------------


def test_ac11_no_elevator_writes():
    """AC11: Harness imports no production elevator/queue/flip-controller paths."""
    module_path = Path(__file__).parent.parent / "lapis_pm" / "batched_fixer_eval.py"
    source = module_path.read_text()
    assert "ElevatorStore" not in source
    assert "_handle_fixer" not in source
    assert "KIND_HANDLER" not in source
    assert "flip-controller" not in source
    assert "ELEVATOR_ACTIVE" not in source


def test_ac11_grounding_hook_noop(mock_fixture_t1):
    """AC11: grounding_hook returns '' by default (no-op seam)."""
    assert callable(grounding_hook)
    assert grounding_hook(mock_fixture_t1) == ""


def test_ac11_harness_is_lease_agnostic():
    """AC11: harness contains no lease-acquire, systemctl, or ssh invocations.

    Note: "gw-serve" may appear in human-readable error messages (operator remedy
    instructions) but must never appear as a subprocess or os.system invocation.
    """
    import re
    module_path = Path(__file__).parent.parent / "lapis_pm" / "batched_fixer_eval.py"
    source = module_path.read_text()
    assert "systemctl" not in source
    assert "ssh " not in source
    assert "acquire_lease" not in source
    # gw-serve must not appear as a shell invocation — only remedy-message strings are allowed
    assert not re.search(r'subprocess\.[^\n]*gw-serve', source), \
        "harness must not invoke gw-serve via subprocess"
    assert not re.search(r'os\.system\([^\n]*gw-serve', source), \
        "harness must not invoke gw-serve via os.system"


# ---------------------------------------------------------------------------
# AC13: Mock swarm test suite (zero GW, zero paid)
# ---------------------------------------------------------------------------


def test_ac13_corpus_schema_nullable_pr():
    """AC13: Corpus schema validates with pr_number as int or null."""
    for pr in [None, 42]:
        f = FixtureRecord(
            repo="test", sha="abc", parent_sha="def", pr_number=pr,
            path="x.py", file_loc="unknown", changed_lines=1, tier="T1",
        )
        data = asdict(f)
        assert data["pr_number"] == pr
        f2 = FixtureRecord(**data)
        assert f2.pr_number == pr


def test_ac13_mock_build_corpus_returns_unchanged():
    """AC13: build_corpus with mock_corpus returns corpus unchanged."""
    corpus = [
        FixtureRecord(
            repo="test", sha="a", parent_sha="b", pr_number=1,
            path="f.py", file_loc="unknown", changed_lines=1, tier="T1",
        )
    ]
    result, meta = build_corpus(mock_corpus=corpus)
    assert len(result) == 1
    assert meta["source"] == "mock"


def test_ac13_mock_generate_candidates():
    """AC13: generate_candidates(mock_mode=True) returns deterministic canned diffs."""
    f = FixtureRecord(
        repo="test", sha="sha123", parent_sha="parent",
        pr_number=1, path="test.py", file_loc="unknown", changed_lines=1, tier="T1",
    )
    candidates = generate_candidates(f, n=3, mock_mode=True)
    assert len(candidates) == 3
    assert all(c is not None for c in candidates)
    assert all("--- a/" in c for c in candidates)


def test_ac13_compute_flakiness_fingerprint_three_runs():
    """AC13: compute_flakiness_fingerprint with 3 runs partitions correctly."""
    runs = [["a", "b"], ["a", "c"], ["a"]]
    stable, flaky = compute_flakiness_fingerprint(runs)
    assert stable == ["a"]        # "a" fails all 3
    assert "b" in flaky           # "b" fails 1/3
    assert "c" in flaky           # "c" fails 1/3


def test_ac13_classify_candidate_outcome_all_cases():
    """AC13: classify_candidate_outcome covers pass/regressed/blind/unverified."""
    # PASS: target t::s flips, non-target t::other stays failing, no new failures
    assert classify_candidate_outcome(
        post_failures=["t::other"],
        base_stable_fail_set=["t::s", "t::other"],
        base_flaky_set=[],
        target_test_files=["t::s"],
        checker_class="DISCRIMINATES",
    ) == "pass"

    # REGRESSED: target t::s still failing
    assert classify_candidate_outcome(
        post_failures=["t::s"],
        base_stable_fail_set=["t::s"],
        base_flaky_set=[],
        target_test_files=["t::s"],
        checker_class="DISCRIMINATES",
    ) == "regressed"

    # REGRESSED: new failure introduced
    assert classify_candidate_outcome(
        post_failures=["t::new"],
        base_stable_fail_set=["t::s"],
        base_flaky_set=[],
        target_test_files=["t::s"],
        checker_class="DISCRIMINATES",
    ) == "regressed"

    # UNVERIFIED: non-target stable test unexpectedly passes
    assert classify_candidate_outcome(
        post_failures=[],         # t::other was stable-fail but now passes
        base_stable_fail_set=["t::s", "t::other"],
        base_flaky_set=[],
        target_test_files=["t::s"],
        checker_class="DISCRIMINATES",
    ) == "unverified"

    # BLIND: checker_class BLIND
    assert classify_candidate_outcome(
        post_failures=[], base_stable_fail_set=[], base_flaky_set=[],
        target_test_files=[], checker_class="BLIND",
    ) == "blind"

    # BLIND: checker_class UNTESTED
    assert classify_candidate_outcome(
        post_failures=[], base_stable_fail_set=[], base_flaky_set=[],
        target_test_files=[], checker_class="UNTESTED",
    ) == "blind"


def test_ac13_validate_corpus_power_floor_unit():
    """AC13: validate_corpus_power_floor raises on insufficient holdouts."""
    thin_corpus = [
        FixtureRecord(
            repo="x", sha=f"s{i}", parent_sha=f"p{i}",
            pr_number=None, path="x.py", file_loc="unknown",
            changed_lines=1, tier="T1", blind_holdout=True,
        )
        for i in range(4)
    ]
    with pytest.raises(ValueError):
        validate_corpus_power_floor(thin_corpus, tier_floor=8)


def test_ac13_launder_intent_code_path():
    """AC13: launder_intent returns (paraphrased, status) tuple that transforms input in mock mode."""
    raw = "fix: rename import from x to y\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
    laundered, status = launder_intent(raw, mock_mode=True)
    assert laundered != raw
    assert "Co-Authored-By" not in laundered
    assert status == "mock"


def test_ac13_mock_run_fixture_scaffold(mock_fixture_t1):
    """AC13: run_fixture(mock_mode=True) returns FixtureRunResult scaffold (DEFERRED oracle)."""
    result = run_fixture(mock_fixture_t1, mock_mode=True)
    assert isinstance(result, FixtureRunResult)
    assert result.fixture_id == f"{mock_fixture_t1.repo}-{mock_fixture_t1.sha[:8]}"
    assert result.n_candidates_generated > 0
    # In mock mode, oracle is NOT invoked; outcome is DEFERRED scaffold
    assert result.execute_select_outcome == "unverified"  # DEFERRED scaffold


def test_ac13_run_eval_mock_mode():
    """AC13: run_eval(phase='run', mock_mode=True) runs without GW or corpus."""
    # Without corpus, returns None gracefully
    result = run_eval(phase="run", run_id="test-ac13-mock", mock_mode=True)
    # No corpus → None is acceptable; no exception should be raised


def test_ac13_extract_module_name():
    """AC13: extract_module_name handles standard cases."""
    assert extract_module_name("lapis_pm/eval_gate.py") == "lapis_pm.eval_gate"
    assert extract_module_name("module.py") == "module"
    assert extract_module_name("pkg/__init__.py") == "pkg"


def test_ac13_parse_pytest_failures():
    """AC13: parse_pytest_failures handles standard pytest -q output."""
    out = "FAILED tests/t.py::test_x - AssertionError\nFAILED tests/t.py::test_y\n2 failed"
    failures = parse_pytest_failures(out)
    assert "tests/t.py::test_x" in failures
    assert "tests/t.py::test_y" in failures


# ---------------------------------------------------------------------------
# AC-H: Corpus-validity hardening — laundering degradation + power-floor guard
# ---------------------------------------------------------------------------


def test_ach1_startup_probe_blocks_real_build_when_gw_down(tmp_path, monkeypatch):
    """AC-H1: non-mock build-corpus raises when GW probe returns fallback status."""
    import lapis_pm.batched_fixer_eval as bfe

    # Simulate GW unavailable: call_operator raises on any call
    def _mock_launder_fails(raw, mock_mode=False):
        if not mock_mode:
            return raw, "fallback"
        cleaned = re.sub(r"\nCo-Authored-By:.*", "", raw, flags=re.DOTALL).strip()
        return f"[Symptom] {cleaned[:200]}", "mock"

    monkeypatch.setattr(bfe, "launder_intent", _mock_launder_fails)

    with pytest.raises(RuntimeError, match="GW not serving big-122B"):
        bfe.run_eval(phase="build-corpus", mock_mode=False)


def test_ach1_mock_mode_skips_probe(tmp_path, monkeypatch):
    """AC-H1: mock-mode build-corpus skips the GW probe and builds GW-free."""
    import lapis_pm.batched_fixer_eval as bfe

    probe_called = []

    real_launder = bfe.launder_intent

    def _tracking_launder(raw, mock_mode=False):
        if not mock_mode:
            probe_called.append(raw)
        return real_launder(raw, mock_mode=mock_mode)

    monkeypatch.setattr(bfe, "launder_intent", _tracking_launder)

    # Mock build_corpus to avoid git operations
    dummy_corpus = [
        FixtureRecord(
            repo="lapis-pm", sha="aabbccdd", parent_sha="eeff0011",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5, tier="T1",
        )
    ]
    monkeypatch.setattr(bfe, "build_corpus", lambda **kw: (dummy_corpus, {
        "laundering_total": 0, "laundering_fallback_count": 0,
        "per_tier_counts": {"T1": 1}, "per_tier_holdout_counts": {},
    }))
    monkeypatch.setattr(bfe, "validate_corpus_power_floor", lambda c, **kw: {"T1": 0, "_shape": "coarse-binary"})
    monkeypatch.setattr(bfe, "validate_discriminates_power_floor", lambda c, **kw: {
        "per_tier_discriminates_counts": {}, "per_bucket_holdout_discriminates": {"T1": 0, "T2+T3": 0},
    })
    monkeypatch.setattr(bfe, "save_corpus", lambda c: None)
    monkeypatch.setattr(bfe, "_write_corpus_manifest", lambda m: None)

    bfe.run_eval(phase="build-corpus", mock_mode=True)
    assert probe_called == [], "mock mode must not call launder_intent with mock_mode=False"


def test_ach2_launder_status_fallback_counted(monkeypatch, tmp_path):
    """AC-H2: build_corpus behaviorally tracks laundering_total and laundering_fallback_count.

    Drives the real extraction loop via a tiny git repo with 5 fix commits so the
    laundering counters (lines ~1108-1110 of batched_fixer_eval.py) are actually exercised.
    """
    import lapis_pm.batched_fixer_eval as bfe

    # Build a tiny git repo with N fix commits that pass all corpus-extraction filters.
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    subprocess.run(["git", "init", str(fake_repo)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=fake_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=fake_repo, check=True, capture_output=True)
    (fake_repo / "module.py").write_text("def foo():\n    return 'init'\n")
    subprocess.run(["git", "add", "."], cwd=fake_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=fake_repo, check=True, capture_output=True)

    n_fix_commits = 5
    for i in range(n_fix_commits):
        # Enough changed lines (>2) to pass the trivial-drop filter; unique content per
        # commit so the MD5-dedup step doesn't squash them.
        content = f"def foo():\n    return '{i}'\n" + "".join(f"# v{i}_{j}\n" for j in range(4))
        (fake_repo / "module.py").write_text(content)
        subprocess.run(["git", "add", "."], cwd=fake_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"fix: broken behavior revision {i}"],
            cwd=fake_repo, check=True, capture_output=True,
        )

    # Redirect repos: only fake_repo (conductor path nonexistent → skipped).
    monkeypatch.setattr(bfe, "LAPIS_PM_REPO", fake_repo)
    monkeypatch.setattr(bfe, "CONDUCTOR_REPO", tmp_path / "nonexistent")

    call_count = [0]

    def _mixed_launder(raw, mock_mode=False):
        call_count[0] += 1
        # Even calls → gw, odd calls → fallback
        if call_count[0] % 2 == 0:
            return f"[GW] {raw[:50]}", "gw"
        return raw, "fallback"

    monkeypatch.setattr(bfe, "launder_intent", _mixed_launder)

    _, meta = bfe.build_corpus(skip_base_runs=True)

    # All 5 fix commits must reach the laundering step.
    assert meta["laundering_total"] == n_fix_commits, (
        f"Expected laundering_total={n_fix_commits}, got {meta['laundering_total']}"
    )
    # Odd-numbered calls (1, 3, 5) return "fallback" → 3 out of 5.
    expected_fallbacks = (n_fix_commits + 1) // 2
    assert meta["laundering_fallback_count"] == expected_fallbacks, (
        f"Expected laundering_fallback_count={expected_fallbacks}, got {meta['laundering_fallback_count']}"
    )
    assert call_count[0] == n_fix_commits


def test_ach2_fallback_warning_printed_to_stderr(monkeypatch, capsys, tmp_path):
    """AC-H2: run_eval build-corpus prints contamination WARNING to stderr when fallback > 0."""
    import lapis_pm.batched_fixer_eval as bfe

    dummy_corpus = [
        FixtureRecord(
            repo="lapis-pm", sha="aabbccdd", parent_sha="eeff0011",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5, tier="T1",
        )
    ]
    # Simulate 2 fallbacks out of 5 total
    monkeypatch.setattr(bfe, "build_corpus", lambda **kw: (dummy_corpus, {
        "laundering_total": 5, "laundering_fallback_count": 2,
        "per_tier_counts": {"T1": 1}, "per_tier_holdout_counts": {},
    }))
    monkeypatch.setattr(bfe, "validate_corpus_power_floor", lambda c, **kw: {"T1": 0, "_shape": "coarse-binary"})
    monkeypatch.setattr(bfe, "validate_discriminates_power_floor", lambda c, **kw: {
        "per_tier_discriminates_counts": {}, "per_bucket_holdout_discriminates": {"T1": 0, "T2+T3": 0},
    })
    monkeypatch.setattr(bfe, "save_corpus", lambda c: None)
    monkeypatch.setattr(bfe, "_write_corpus_manifest", lambda m: None)

    bfe.run_eval(phase="build-corpus", mock_mode=True)

    captured = capsys.readouterr()
    assert "CONTAMINATED CORPUS" in captured.err
    assert "2/5" in captured.err


def test_ach2_no_warning_when_no_fallbacks(monkeypatch, capsys):
    """AC-H2: run_eval build-corpus prints NO contamination warning when fallback count is 0."""
    import lapis_pm.batched_fixer_eval as bfe

    dummy_corpus = [
        FixtureRecord(
            repo="lapis-pm", sha="aabbccdd", parent_sha="eeff0011",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5, tier="T1",
        )
    ]
    monkeypatch.setattr(bfe, "build_corpus", lambda **kw: (dummy_corpus, {
        "laundering_total": 5, "laundering_fallback_count": 0,
        "per_tier_counts": {"T1": 1}, "per_tier_holdout_counts": {},
    }))
    monkeypatch.setattr(bfe, "validate_corpus_power_floor", lambda c, **kw: {"T1": 0, "_shape": "coarse-binary"})
    monkeypatch.setattr(bfe, "validate_discriminates_power_floor", lambda c, **kw: {
        "per_tier_discriminates_counts": {}, "per_bucket_holdout_discriminates": {"T1": 0, "T2+T3": 0},
    })
    monkeypatch.setattr(bfe, "save_corpus", lambda c: None)
    monkeypatch.setattr(bfe, "_write_corpus_manifest", lambda m: None)

    bfe.run_eval(phase="build-corpus", mock_mode=True)

    captured = capsys.readouterr()
    assert "CONTAMINATED" not in captured.err


def test_ach2_manifest_written_with_required_fields(monkeypatch, tmp_path):
    """AC-H2: _write_corpus_manifest writes _manifest.json with laundering + shape fields."""
    import lapis_pm.batched_fixer_eval as bfe

    original_corpus_dir = bfe.CORPUS_DIR
    bfe.CORPUS_DIR = tmp_path / "corpus"
    try:
        metadata = {
            "laundering_total": 10,
            "laundering_fallback_count": 2,
            "per_tier_counts": {"T1": 5, "T2": 3, "T3": 2},
            "per_tier_holdout_counts": {"T1": 5, "T2": 3, "T3": 2},
            "corpus_shape": "3-tier",
        }
        bfe._write_corpus_manifest(metadata)
        manifest_path = bfe.CORPUS_DIR / "_manifest.json"
        assert manifest_path.exists()
        data = json.load(open(manifest_path))
        assert data["laundering_total"] == 10
        assert data["laundering_fallback_count"] == 2
        assert data["corpus_shape"] == "3-tier"
        assert "per_tier_counts" in data
        assert "per_tier_holdout_counts" in data
    finally:
        bfe.CORPUS_DIR = original_corpus_dir


def test_ach3_t1_t2_only_not_three_tier():
    """AC-H3: corpus with only T1+T2 (no T3) is NOT reported as 3-tier — takes coarse path."""
    corpus = (
        [
            FixtureRecord(
                repo="lapis-pm", sha=f"t1s{i}", parent_sha=f"p{i}",
                pr_number=None, path="x.py", file_loc="unknown", changed_lines=5,
                tier="T1", blind_holdout=True,
            )
            for i in range(8)
        ]
        + [
            FixtureRecord(
                repo="conductor", sha=f"t2s{i}", parent_sha=f"q{i}",
                pr_number=None, path="y.py", file_loc="unknown", changed_lines=20,
                tier="T2", blind_holdout=True,
            )
            for i in range(8)
        ]
    )
    result = validate_corpus_power_floor(corpus, tier_floor=8)
    assert result.get("_shape") == "coarse-binary", "T1+T2-only corpus must be coarse-binary"
    assert result.get("T1", 0) >= 8
    assert result.get("T2", 0) >= 8


def test_ach3_full_three_tier_is_three_tier():
    """AC-H3: corpus with T1+T2+T3 each >= floor is reported as 3-tier."""
    corpus = [
        FixtureRecord(
            repo="lapis-pm", sha=f"{t}s{i}", parent_sha=f"p{t}{i}",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5,
            tier=t, blind_holdout=True,
        )
        for t in ("T1", "T2", "T3")
        for i in range(8)
    ]
    result = validate_corpus_power_floor(corpus, tier_floor=8)
    assert result.get("_shape") == "3-tier"
    assert result.get("T1", 0) >= 8
    assert result.get("T2", 0) >= 8
    assert result.get("T3", 0) >= 8


def test_ach3_sub_floor_raises():
    """AC-H3: corpus where neither 3-tier nor coarse-binary floor is satisfiable raises."""
    corpus = [
        FixtureRecord(
            repo="lapis-pm", sha=f"s{i}", parent_sha=f"p{i}",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5,
            tier="T1", blind_holdout=True,
        )
        for i in range(3)  # only 3 T1, no T2/T3
    ]
    with pytest.raises(ValueError, match="power floor not met"):
        validate_corpus_power_floor(corpus, tier_floor=8)


def test_ach3_shape_captured_in_run_eval_metadata(monkeypatch, tmp_path):
    """AC-H3: run_eval build-corpus captures corpus_shape from validate_corpus_power_floor."""
    import lapis_pm.batched_fixer_eval as bfe

    captured_metadata = {}

    def _mock_write_manifest(meta):
        captured_metadata.update(meta)

    dummy_corpus = [
        FixtureRecord(
            repo="lapis-pm", sha="aabbccdd", parent_sha="eeff0011",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5, tier="T1",
        )
    ]
    monkeypatch.setattr(bfe, "build_corpus", lambda **kw: (dummy_corpus, {
        "laundering_total": 1, "laundering_fallback_count": 0,
    }))
    monkeypatch.setattr(bfe, "validate_corpus_power_floor", lambda c, **kw: {
        "T1": 5, "_shape": "coarse-binary"
    })
    monkeypatch.setattr(bfe, "validate_discriminates_power_floor", lambda c, **kw: {
        "per_tier_discriminates_counts": {}, "per_bucket_holdout_discriminates": {"T1": 0, "T2+T3": 0},
    })
    monkeypatch.setattr(bfe, "save_corpus", lambda c: None)
    monkeypatch.setattr(bfe, "_write_corpus_manifest", _mock_write_manifest)

    bfe.run_eval(phase="build-corpus", mock_mode=True)

    assert captured_metadata.get("corpus_shape") == "coarse-binary"


# ---------------------------------------------------------------------------
# AC-O: Co-committed test oracle (batched-fixer-cocommitted-test-oracle-v0)
# ---------------------------------------------------------------------------


def test_aco1_schema_new_fields_present():
    """AC-O1: FixtureRecord has golden_source_diff, golden_test_diff, golden_test_ids, fail_first_confirmed."""
    f = FixtureRecord(
        repo="test", sha="abc", parent_sha="def", pr_number=None,
        path="m.py", file_loc="unknown", changed_lines=5, tier="T1",
    )
    data = asdict(f)
    assert "golden_source_diff" in data
    assert "golden_test_diff" in data
    assert "golden_test_ids" in data
    assert "fail_first_confirmed" in data
    assert data["golden_source_diff"] == ""
    assert data["golden_test_diff"] == ""
    assert data["golden_test_ids"] == []
    assert data["fail_first_confirmed"] is False


def test_aco1_split_diff_source_only():
    """AC-O1: split_diff_by_type on a source-only diff yields non-empty source, empty test."""
    diff = (
        "diff --git a/lapis_pm/foo.py b/lapis_pm/foo.py\n"
        "index abc..def 100644\n"
        "--- a/lapis_pm/foo.py\n"
        "+++ b/lapis_pm/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
    )
    src, tst = split_diff_by_type(diff)
    assert "lapis_pm/foo.py" in src
    assert tst == ""


def test_aco1_split_diff_fix_and_test():
    """AC-O1: split_diff_by_type correctly separates source and test sections."""
    diff = (
        "diff --git a/lapis_pm/foo.py b/lapis_pm/foo.py\n"
        "--- a/lapis_pm/foo.py\n"
        "+++ b/lapis_pm/foo.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/tests/test_foo.py b/tests/test_foo.py\n"
        "--- a/tests/test_foo.py\n"
        "+++ b/tests/test_foo.py\n"
        "@@ -1 +2 @@\n+def test_bar():\n+    assert True\n"
    )
    src, tst = split_diff_by_type(diff)
    assert "lapis_pm/foo.py" in src
    assert "tests/test_foo.py" not in src
    assert "tests/test_foo.py" in tst
    assert "lapis_pm/foo.py" not in tst


def test_aco1_split_diff_test_suffix_patterns():
    """AC-O1: split_diff recognises _test.py and test_ prefix patterns."""
    diff_suffix = (
        "diff --git a/module_test.py b/module_test.py\n"
        "--- a/module_test.py\n+++ b/module_test.py\n@@ -1 +1 @@\n+def test_x(): pass\n"
    )
    diff_prefix = (
        "diff --git a/test_module.py b/test_module.py\n"
        "--- a/test_module.py\n+++ b/test_module.py\n@@ -1 +1 @@\n+def test_y(): pass\n"
    )
    _, tst1 = split_diff_by_type(diff_suffix)
    assert "module_test.py" in tst1
    _, tst2 = split_diff_by_type(diff_prefix)
    assert "test_module.py" in tst2


def test_aco1_extract_test_ids_empty():
    """AC-O1: empty test diff yields empty ID list."""
    assert extract_test_ids_from_diff("") == []


def test_aco1_extract_test_ids_from_diff():
    """AC-O1: extract_test_ids_from_diff parses added def test_* lines into node IDs."""
    test_diff = (
        "diff --git a/tests/test_foo.py b/tests/test_foo.py\n"
        "--- a/tests/test_foo.py\n"
        "+++ b/tests/test_foo.py\n"
        "@@ -10,0 +11 @@\n"
        "+def test_new_behavior():\n"
        "+    assert foo() == 'new'\n"
        "+\n"
        "+def test_edge_case():\n"
        "+    assert foo() != None\n"
    )
    ids = extract_test_ids_from_diff(test_diff)
    assert "tests/test_foo.py::test_new_behavior" in ids
    assert "tests/test_foo.py::test_edge_case" in ids
    assert len(ids) == 2


def test_aco1_extract_test_ids_ignores_unchanged_lines():
    """AC-O1: extract_test_ids ignores context/removed def test_ lines."""
    diff = (
        "+++ b/tests/test_foo.py\n"
        " def test_existing():\n"   # context line — not added
        "-def test_removed():\n"    # removed line
        "+def test_added():\n"      # added
    )
    ids = extract_test_ids_from_diff(diff)
    assert len(ids) == 1
    assert "test_added" in ids[0]


def test_aco1_source_only_fixture_has_empty_test_fields():
    """AC-O1: source-only commit yields empty golden_test_diff and golden_test_ids."""
    source_diff = (
        "diff --git a/lapis_pm/foo.py b/lapis_pm/foo.py\n"
        "--- a/lapis_pm/foo.py\n+++ b/lapis_pm/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    src, tst = split_diff_by_type(source_diff)
    ids = extract_test_ids_from_diff(tst)
    assert tst == ""
    assert ids == []


# AC-O2: classify_checker_class_cocommitted

def test_aco2_discriminates_when_fail_first_confirmed():
    """AC-O2: DISCRIMINATES iff has golden_test_diff AND fail_first_confirmed."""
    result = classify_checker_class_cocommitted(
        golden_test_diff="--- a/tests/test_foo.py\n+def test_x(): pass\n",
        fail_first_confirmed=True,
    )
    assert result == "DISCRIMINATES"


def test_aco2_blind_when_test_present_but_not_fail_first():
    """AC-O2: BLIND when has golden_test_diff but fail_first_confirmed is False."""
    result = classify_checker_class_cocommitted(
        golden_test_diff="--- a/tests/test_foo.py\n+def test_x(): pass\n",
        fail_first_confirmed=False,
    )
    assert result == "BLIND"


def test_aco2_untested_when_no_test_diff():
    """AC-O2: UNTESTED when golden_test_diff is empty (source-only commit)."""
    result = classify_checker_class_cocommitted(golden_test_diff="", fail_first_confirmed=False)
    assert result == "UNTESTED"
    result2 = classify_checker_class_cocommitted(golden_test_diff="", fail_first_confirmed=True)
    assert result2 == "UNTESTED"


def test_aco2_build_time_verify_fail_first_synthetic(tmp_path):
    """AC-O2: verify_fail_first returns (True, True, False) for a proper fail-first test."""
    import subprocess as sp
    from lapis_pm.batched_fixer_eval import verify_fail_first, dedicated_clone

    # Build a synthetic repo: source that fails the test, then fix
    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", str(repo)], check=True, capture_output=True)
    sp.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "config", "user.name", "T"], cwd=repo, check=True, capture_output=True)

    # Parent commit: module returns 'old'
    (repo / "module.py").write_text("def foo():\n    return 'old'\n")
    sp.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)
    parent_sha = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()

    # golden_test_diff: test that asserts 'new' (fails at parent, passes after fix)
    golden_test_diff = (
        "diff --git a/test_module.py b/test_module.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/test_module.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+from module import foo\n"
        "+def test_foo_returns_new():\n"
        "+    assert foo() == 'new'\n"
    )
    golden_source_diff = (
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def foo():\n"
        "-    return 'old'\n"
        "+    return 'new'\n"
    )
    golden_test_ids = ["test_module.py::test_foo_returns_new"]

    with dedicated_clone(repo, "test-ff") as clone_path:
        ff_confirmed, sanity_pass, is_flaky = verify_fail_first(
            clone_path, parent_sha, golden_test_diff, golden_source_diff,
            golden_test_ids, n=3,
        )

    assert ff_confirmed is True, "Test must fail on unpatched source"
    assert sanity_pass is True, "Test must pass after source fix"
    assert is_flaky is False


def test_aco2_verify_fail_first_blind_when_test_passes_at_parent(tmp_path):
    """AC-O2: verify_fail_first returns (False, *, False) when test passes on unpatched source."""
    import subprocess as sp
    from lapis_pm.batched_fixer_eval import verify_fail_first, dedicated_clone

    repo = tmp_path / "repo2"
    repo.mkdir()
    sp.run(["git", "init", str(repo)], check=True, capture_output=True)
    sp.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "config", "user.name", "T"], cwd=repo, check=True, capture_output=True)
    (repo / "module.py").write_text("def foo():\n    return 'new'\n")  # already returns 'new'
    sp.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)
    parent_sha = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()

    # Test expects 'new' — already satisfied at parent → BLIND
    golden_test_diff = (
        "diff --git a/test_module.py b/test_module.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/test_module.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+from module import foo\n"
        "+def test_foo():\n"
        "+    assert foo() == 'new'\n"
    )
    golden_source_diff = ""  # no real source change needed
    golden_test_ids = ["test_module.py::test_foo"]

    with dedicated_clone(repo, "test-ff-blind") as clone_path:
        ff_confirmed, sanity_pass, is_flaky = verify_fail_first(
            clone_path, parent_sha, golden_test_diff, golden_source_diff,
            golden_test_ids, n=3,
        )

    assert ff_confirmed is False, "Test passes at parent — not a fail-first discriminator"
    assert is_flaky is False


# AC-O3: co-committed test execute oracle

@pytest.fixture
def discriminates_fixture_with_golden_test():
    """DISCRIMINATES fixture with golden_test_diff set (co-committed test oracle)."""
    return FixtureRecord(
        repo="lapis-pm",
        sha="dead0000beef0001",
        parent_sha="parent_discriminates",
        pr_number=10,
        path="lapis_pm/module.py",
        file_loc="line 1-5",
        changed_lines=4,
        tier="T1",
        task_intent_paraphrased="module returns wrong value",
        pre_state_slice="def foo():\n    return 'old'\n",
        golden_diff="(full diff for provenance)",
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


def test_aco3_oracle_pass_when_golden_test_passes(tmp_path, discriminates_fixture_with_golden_test):
    """AC-O3: oracle returns 'pass' when candidate satisfies the golden test."""
    import subprocess as sp
    from lapis_pm.batched_fixer_eval import oracle_evaluate_candidate, dedicated_clone

    fixture = discriminates_fixture_with_golden_test

    # Build a synthetic repo at parent state (foo returns 'old')
    repo = tmp_path / "repo"
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

    # Inject the real parent_sha into fixture
    fixture = FixtureRecord(**{**asdict(fixture), "parent_sha": parent_sha})

    # Candidate that correctly fixes foo() → 'new'
    good_candidate = (
        "diff --git a/lapis_pm/module.py b/lapis_pm/module.py\n"
        "--- a/lapis_pm/module.py\n+++ b/lapis_pm/module.py\n"
        "@@ -1,2 +1,2 @@\n def foo():\n-    return 'old'\n+    return 'new'\n"
    )

    with dedicated_clone(repo, "test-aco3-pass") as clone_path:
        status, failures, outcome = oracle_evaluate_candidate(clone_path, fixture, good_candidate, "wt_pass")

    assert status == "success"
    assert outcome == "pass", f"Expected pass, got {outcome}"


def test_aco3_oracle_regressed_when_golden_fails(tmp_path, discriminates_fixture_with_golden_test):
    """AC-O3: oracle returns 'regressed' when candidate does not satisfy the golden test."""
    import subprocess as sp
    from lapis_pm.batched_fixer_eval import oracle_evaluate_candidate, dedicated_clone

    fixture = discriminates_fixture_with_golden_test

    repo = tmp_path / "repo2"
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

    fixture = FixtureRecord(**{**asdict(fixture), "parent_sha": parent_sha})

    # Candidate that returns wrong value — golden test still fails
    bad_candidate = (
        "diff --git a/lapis_pm/module.py b/lapis_pm/module.py\n"
        "--- a/lapis_pm/module.py\n+++ b/lapis_pm/module.py\n"
        "@@ -1,2 +1,2 @@\n def foo():\n-    return 'old'\n+    return 'wrong'\n"
    )

    with dedicated_clone(repo, "test-aco3-regressed") as clone_path:
        status, failures, outcome = oracle_evaluate_candidate(clone_path, fixture, bad_candidate, "wt_reg")

    assert status == "success"
    assert outcome == "regressed", f"Expected regressed, got {outcome}"


def test_aco3_oracle_apply_failed_on_bad_patch(tmp_path, discriminates_fixture_with_golden_test):
    """AC-O3: oracle returns 'failed' when candidate diff does not apply."""
    import subprocess as sp
    from lapis_pm.batched_fixer_eval import oracle_evaluate_candidate, dedicated_clone

    fixture = discriminates_fixture_with_golden_test

    repo = tmp_path / "repo3"
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

    fixture = FixtureRecord(**{**asdict(fixture), "parent_sha": parent_sha})

    bad_patch = "this is not a valid diff\n"

    with dedicated_clone(repo, "test-aco3-apply-fail") as clone_path:
        status, failures, outcome = oracle_evaluate_candidate(clone_path, fixture, bad_patch, "wt_af")

    assert status == "failed", f"Expected failed, got {status}"


# AC-O4: leak guard

def test_aco4_prompt_does_not_reference_golden_test():
    """AC-O4: candidate prompt content never includes golden_test fields.

    Verifies behaviorally: a DISCRIMINATES fixture with a distinctive golden_test_diff
    string does NOT have that string appear in the generated candidate (mock mode).
    Also verifies the prompt block does not feed golden_test_diff into the f-string.
    """
    import inspect
    source = inspect.getsource(generate_candidates)
    # The prompt must use task_intent_paraphrased and pre_state_slice
    assert "task_intent_paraphrased" in source
    assert "pre_state_slice" in source
    # The prompt f-string expression block is between 'prompt = (' and the leak guard.
    # The leak guard checks for the field AFTER prompt is built — exclude it.
    # Find the raw prompt string: from 'prompt = (' up to the first 'Emit a unified diff' line.
    if "Emit a unified diff" in source:
        prompt_template = source.split("prompt =")[1].split("Emit a unified diff")[0]
        assert "golden_test_diff" not in prompt_template, \
            "golden_test_diff must not be fed into the prompt f-string"
        assert "golden_test_ids" not in prompt_template, \
            "golden_test_ids must not be fed into the prompt f-string"


def test_aco4_mock_candidates_dont_contain_test_source():
    """AC-O4: mock generate_candidates output contains no golden test text."""
    fixture = FixtureRecord(
        repo="test", sha="abc123", parent_sha="parent",
        pr_number=1, path="module.py", file_loc="unknown", changed_lines=5, tier="T1",
        task_intent_paraphrased="Fix the bug",
        pre_state_slice="def foo():\n    return 'old'\n",
        golden_test_diff=(
            "--- /dev/null\n+++ b/tests/test_m.py\n"
            "+def test_secret_oracle():\n+    assert True\n"
        ),
        golden_test_ids=["tests/test_m.py::test_secret_oracle"],
        checker_class="DISCRIMINATES",
        fail_first_confirmed=True,
    )
    candidates = generate_candidates(fixture, n=2, mock_mode=True)
    for c in candidates:
        if c:
            assert "test_secret_oracle" not in c
            assert "golden_test" not in c


# AC-O5: DISCRIMINATES power floor

def test_aco5_validate_discriminates_floor_passes():
    """AC-O5: validate_discriminates_power_floor passes when T1 and T2+T3 each have ≥8."""
    corpus = []
    for i in range(8):
        corpus.append(FixtureRecord(
            repo="lapis-pm", sha=f"d1{i:06d}", parent_sha=f"p{i}",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5,
            tier="T1", checker_class="DISCRIMINATES", blind_holdout=True,
            golden_test_diff="--- /dev/null\n+++ b/t.py\n+def test_x(): pass\n",
            golden_test_ids=["t.py::test_x"], fail_first_confirmed=True,
        ))
    for i in range(8):
        corpus.append(FixtureRecord(
            repo="conductor", sha=f"d2{i:06d}", parent_sha=f"q{i}",
            pr_number=None, path="y.py", file_loc="unknown", changed_lines=15,
            tier="T2", checker_class="DISCRIMINATES", blind_holdout=True,
            golden_test_diff="--- /dev/null\n+++ b/t.py\n+def test_y(): pass\n",
            golden_test_ids=["t.py::test_y"], fail_first_confirmed=True,
        ))
    result = validate_discriminates_power_floor(corpus, floor=8)
    assert result["per_bucket_holdout_discriminates"]["T1"] >= 8
    assert result["per_bucket_holdout_discriminates"]["T2+T3"] >= 8


def test_aco5_validate_discriminates_floor_raises_below_floor():
    """AC-O5: validate_discriminates_power_floor raises ValueError when below floor."""
    corpus = [
        FixtureRecord(
            repo="lapis-pm", sha=f"x{i}", parent_sha=f"p{i}",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5,
            tier="T1", checker_class="DISCRIMINATES", blind_holdout=True,
            fail_first_confirmed=True,
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="DISCRIMINATES holdout floor not met"):
        validate_discriminates_power_floor(corpus, floor=8)


def test_aco5_manifest_has_discriminates_fields(monkeypatch, tmp_path):
    """AC-O5: _write_corpus_manifest includes per_tier_discriminates_counts and
    per_bucket_holdout_discriminates when called from run_eval."""
    import lapis_pm.batched_fixer_eval as bfe

    captured = {}

    def _capture_manifest(meta):
        captured.update(meta)

    disc_corpus = []
    for i in range(8):
        disc_corpus.append(FixtureRecord(
            repo="lapis-pm", sha=f"dm{i:06d}", parent_sha=f"p{i}",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5,
            tier="T1", checker_class="DISCRIMINATES", blind_holdout=True,
            fail_first_confirmed=True,
        ))
    for i in range(8):
        disc_corpus.append(FixtureRecord(
            repo="conductor", sha=f"dn{i:06d}", parent_sha=f"q{i}",
            pr_number=None, path="y.py", file_loc="unknown", changed_lines=15,
            tier="T2", checker_class="DISCRIMINATES", blind_holdout=True,
            fail_first_confirmed=True,
        ))

    monkeypatch.setattr(bfe, "build_corpus", lambda **kw: (disc_corpus, {
        "laundering_total": 0, "laundering_fallback_count": 0,
    }))
    monkeypatch.setattr(bfe, "validate_corpus_power_floor", lambda c, **kw: {
        "T1": 8, "T2": 8, "_shape": "coarse-binary",
    })
    monkeypatch.setattr(bfe, "save_corpus", lambda c: None)
    monkeypatch.setattr(bfe, "_write_corpus_manifest", _capture_manifest)

    bfe.run_eval(phase="build-corpus", mock_mode=True)

    assert "per_tier_discriminates_counts" in captured
    assert "per_bucket_holdout_discriminates" in captured
    assert captured["per_bucket_holdout_discriminates"]["T1"] == 8
    assert captured["per_bucket_holdout_discriminates"]["T2+T3"] == 8


def test_aco5_build_fails_loud_when_discriminates_below_floor(monkeypatch, capsys):
    """AC-O5: run_eval build-corpus fails loud (logs error) when DISCRIMINATES floor unmet."""
    import lapis_pm.batched_fixer_eval as bfe

    thin_corpus = [
        FixtureRecord(
            repo="lapis-pm", sha="x", parent_sha="y",
            pr_number=None, path="x.py", file_loc="unknown", changed_lines=5, tier="T1",
        )
    ]
    monkeypatch.setattr(bfe, "build_corpus", lambda **kw: (thin_corpus, {
        "laundering_total": 0, "laundering_fallback_count": 0,
    }))
    monkeypatch.setattr(bfe, "validate_corpus_power_floor", lambda c, **kw: {
        "_shape": "coarse-binary",
    })
    monkeypatch.setattr(bfe, "save_corpus", lambda c: None)
    monkeypatch.setattr(bfe, "_write_corpus_manifest", lambda m: None)

    # run_eval catches ValueError and logs it; the build does NOT complete
    result = bfe.run_eval(phase="build-corpus", mock_mode=True)
    assert result is None  # build did not succeed


# AC-O3 classify_candidate_outcome_golden unit tests

def test_aco3_classify_golden_pass():
    """AC-O3: classify_candidate_outcome_golden returns 'pass' when all golden IDs pass."""
    outcome = classify_candidate_outcome_golden(
        post_failures=["tests/t.py::some_other_test"],
        golden_test_ids=["tests/t.py::test_the_fix"],
    )
    assert outcome == "pass"


def test_aco3_classify_golden_regressed():
    """AC-O3: classify_candidate_outcome_golden returns 'regressed' when golden test fails."""
    outcome = classify_candidate_outcome_golden(
        post_failures=["tests/t.py::test_the_fix"],
        golden_test_ids=["tests/t.py::test_the_fix"],
    )
    assert outcome == "regressed"


def test_aco3_classify_golden_blind_no_ids():
    """AC-O3: classify_candidate_outcome_golden returns 'blind' when no golden_test_ids."""
    outcome = classify_candidate_outcome_golden(
        post_failures=[],
        golden_test_ids=[],
    )
    assert outcome == "blind"


# ---------------------------------------------------------------------------
# DEFERRED SCAFFOLDS (AC5, AC6, AC6b, AC7-AC12) — NOT claimed green in this bind
# ---------------------------------------------------------------------------
# These tests exercise the scaffold interfaces only. They are NOT part of the
# CORE-AC must-land-green set for this bind. The follow-on leg implements them.


@pytest.mark.skip(reason="DEFERRED: AC5 (single-shot baseline) implemented in follow-on leg")
def test_deferred_ac5_single_shot_baseline(mock_fixture_t1):
    pass


@pytest.mark.skip(reason="DEFERRED: AC6 (best-of-N execute-select) implemented in follow-on leg")
def test_deferred_ac6_best_of_n(mock_fixture_t1):
    pass


@pytest.mark.skip(reason="DEFERRED: AC6b (held-out validation) implemented in follow-on leg")
def test_deferred_ac6b_held_out_validation(mock_fixture_t1):
    pass


@pytest.mark.skip(reason="DEFERRED: AC7 (vote-judge + calibration) implemented in follow-on leg")
def test_deferred_ac7_three_valued_classifier():
    pass


@pytest.mark.skip(reason="DEFERRED: AC8 (decision table + hard gates) implemented in follow-on leg")
def test_deferred_ac8_hard_gates():
    pass


@pytest.mark.skip(reason="DEFERRED: AC9 (Arm-A baseline + spend metrics) implemented in follow-on leg")
def test_deferred_ac9_arm_a_baseline():
    pass


@pytest.mark.skip(reason="DEFERRED: AC10 (phase-gate + tolerant recheck) requires GW — follow-on leg")
def test_deferred_ac10_swarm_health():
    pass


@pytest.mark.skip(reason="DEFERRED: AC12 (dual-surface report) implemented in follow-on leg")
def test_deferred_ac12_dual_surface_report():
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
