"""Batched-fixer patch-quality eval harness (H5 U4) — KEEP HALF (steps 1-2).

AMENDED §0c: This bind is scoped to build-order steps 1-2 only.
- Step 1 (AC1, AC1b, AC2 stub): Corpus builder + classifier + intent-laundering.
- Step 2 (AC3, AC4, AC4b): Isolated-worktree checker + apply-check +
  flakiness-robust baseline-diff oracle.

DEFERRED to follow-on leg (AC5, AC6, AC6b, AC7-AC12):
- EXECUTE-selector + best-of-N run phase
- VOTE-selector + judge-calibration
- Arm-A baseline + decision-table aggregator + dual-surface report

Mocking scope: SWARM/candidate-generation ONLY (call_swarm / call_operator).
The oracle and corpus builder are REAL deterministic local code with no GW dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

EVAL_BASE = Path("/tmp/batched-fixer-eval")
CORPUS_DIR = Path(__file__).parent / "fixtures/batched_fixer_corpus"
RUNS_DIR = EVAL_BASE / "runs"
REPORTS_DIR = Path("/srv/lapis/planning/evals")

LAPIS_PM_REPO = Path("/srv/lapis/lapis-pm")
CONDUCTOR_REPO = Path("/srv/git/conductor-working")

CLONE_PREFIX = "/tmp/bfe"

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

PR_EVAL_TIMEOUT_S = 180
SWARM_PROBE_TIMEOUT_S = 4
SWARM_RECHECK_TIMEOUT_S = 10
SWARM_HEALTH_RETRIES = 2
TEARDOWN_WAIT_S = 5.0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_SWARM_MODEL = "Qwen3-Coder-Next"

CORPUS_MIN_SIZE = 40
CORPUS_MAX_SIZE = 60
CORPUS_HOLDOUT_PER_TIER = 8
FLAKE_RUN_COUNT = 3

N_CANDIDATES_RANGE = [1, 2, 3, 4]

EXECUTE_SUCCESS_BASELINE_TARGET = 0.70
BEST_OF_N_LIFT_PP = 15
HELD_OUT_FAIL_GATE = 0.05
JUDGE_ACCURACY_GATE = 0.80
BLIND_HOLDOUT_SIZE_GATE = 8
BLIND_SHARE_GATE = 0.30

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FixtureRecord:
    """One frozen corpus record per landed fix fixture."""
    repo: str
    sha: str
    parent_sha: str
    pr_number: Optional[int]
    path: str
    file_loc: str
    changed_lines: int
    tier: Literal["T1", "T2", "T3"]
    golden_diff_cyclomatic_delta: Optional[int] = None
    distinct_symbols_touched: int = 0
    is_concurrency_code: bool = False
    task_intent_raw: str = ""
    task_intent_paraphrased: str = ""
    intent_source: Literal["commit-body", "reviewer-comment"] = "commit-body"
    pre_state_slice: str = ""
    golden_diff: str = ""
    scoped_test_files: list[str] = field(default_factory=list)
    base_stable_fail_set: list[str] = field(default_factory=list)
    base_flaky_set: list[str] = field(default_factory=list)
    target_test_files: list[str] = field(default_factory=list)
    checker_class: Literal["DISCRIMINATES", "BLIND", "UNTESTED"] = "UNTESTED"
    is_reviewer_cycle: bool = False
    is_test_only: bool = False
    blind_holdout: bool = False
    flaky_excluded_count: int = 0


@dataclass
class CandidateResult:
    """Result of running one candidate through the oracle."""
    fixture_id: str
    candidate_seed: int
    candidate_text: Optional[str]
    apply_status: Literal["success", "failed", "apply_error"] = "apply_error"
    apply_error: str = ""
    scoped_test_outcome: Literal["pass", "regressed", "blind", "unverified"] = "unverified"
    test_failures: list[str] = field(default_factory=list)
    test_errors: list[str] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class FixtureRunResult:
    """Aggregated result for one fixture across N candidates.

    NOTE: execute_select, held_out_test_outcome, vote_judge_pick are
    DEFERRED (AC6, AC6b, AC7) — populated only in the follow-on leg.
    """
    fixture_id: str
    tier: str
    repo: str
    n_candidates_generated: int
    none_count: int
    candidates: list[CandidateResult] = field(default_factory=list)
    # DEFERRED: AC6 execute-select fields
    execute_select_outcome: Literal[
        "pass", "regressed", "blind", "no_passing_candidate", "unverified"
    ] = "unverified"
    first_passing_candidate_seed: Optional[int] = None
    passing_candidate_count: int = 0
    selected_candidate: Optional[CandidateResult] = None
    # DEFERRED: AC6b held-out validation
    held_out_test_outcome: Literal["pass", "regressed", "unverified"] = "unverified"
    # DEFERRED: AC7 vote/judge fields
    vote_judge_pick: Optional[int] = None
    vote_judge_rationale: str = ""
    judge_agreed_with_execute: Optional[bool] = None


@dataclass
class TierMetrics:
    """Aggregated metrics for one (tier, N) cell. DEFERRED: AC8."""
    tier: str
    n: int
    repo_breakdown: dict[str, dict]
    execute_success_rate: float
    execute_success_ci: tuple[float, float]
    candidates_to_first_pass_median: float
    selected_but_fails_held_out_rate: float
    held_out_ci: tuple[float, float]
    holdout_n: int
    held_out_fail_gate_passed: bool
    blind_vote_accuracy: Optional[float] = None
    blind_holdout_n: int = 0
    blind_share: float = 0.0
    judge_accuracy_on_discriminating: Optional[float] = None
    judge_gate_passed: bool = False
    blind_gate_passed: bool = False
    routing_recommendation: Literal["swarm-lane", "big-lane", "INSUFFICIENT_POWER"] = "big-lane"
    gate_failure_reason: str = ""
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0


@dataclass
class EvalResult:
    """Final eval result. DEFERRED: report fields populated in follow-on leg."""
    run_id: str
    generated_at: str
    served_model_id: str
    served_model_backend: str
    tier_metrics: dict[str, dict[int, TierMetrics]]
    arm_a_latency_p50: float = 0.0
    arm_a_latency_p95: float = 0.0
    arm_b_total_latency_s: float = 0.0
    verdict: dict[str, Any] = field(default_factory=dict)
    report_path: str = ""


# ---------------------------------------------------------------------------
# Serving assertions (AC10 — DEFERRED run phase uses these)
# ---------------------------------------------------------------------------


def swarm_serving() -> bool:
    from agents_core.llm import swarm_serving as _swarm_serving
    return _swarm_serving()


def swarm_model() -> Optional[str]:
    from agents_core.llm import swarm_model as _swarm_model
    return _swarm_model()


def assert_swarm_health(strict: bool = True) -> bool:
    """Assert swarm serving state. strict=startup, tolerant=between-batch."""
    timeout = SWARM_PROBE_TIMEOUT_S if strict else SWARM_RECHECK_TIMEOUT_S
    retries = 1 if not strict else 0

    for attempt in range(1 + retries):
        try:
            if not swarm_serving():
                if strict:
                    logger.error(
                        "GW swarm not serving — operator must open a swarm window; "
                        "see §4 procedure"
                    )
                    return False
                logger.warning("Swarm recheck: not serving (attempt %d/%d)", attempt + 1, retries + 1)
                if attempt < retries:
                    time.sleep(2)
                continue

            model = swarm_model()
            if model != EXPECTED_SWARM_MODEL:
                if strict:
                    logger.error(
                        "GW swarm model mismatch: got %r, expected %r",
                        model, EXPECTED_SWARM_MODEL,
                    )
                    return False
                logger.warning(
                    "Swarm recheck: model %r (expected %r), attempt %d/%d",
                    model, EXPECTED_SWARM_MODEL, attempt + 1, retries + 1,
                )
                if attempt < retries:
                    time.sleep(2)
                continue

            return True

        except Exception as e:
            if strict:
                logger.error("Swarm health check failed (strict): %s", e)
                return False
            logger.warning("Swarm recheck slow/failed (attempt %d/%d): %s", attempt + 1, retries + 1, e)
            if attempt < retries:
                time.sleep(2)

    logger.error("Swarm recheck failed after %d attempts — aborting with partial-results dump", retries + 1)
    return False


# ---------------------------------------------------------------------------
# Git / worktree utilities (AC3)
# ---------------------------------------------------------------------------


def cleanup_leaked_worktrees(repo_path: Path):
    """Remove leaked worktrees off a clone (never touches shared working tree)."""
    try:
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for i, line in enumerate(result.stdout.splitlines()):
            if not line.strip() or i == 0:
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            wt = parts[0]
            if not wt or wt == "worktree":
                continue
            try:
                logger.warning("Removing leaked worktree: %s", wt)
                subprocess.run(
                    ["git", "worktree", "remove", "--force", wt],
                    cwd=repo_path,
                    capture_output=True,
                    timeout=10,
                )
            except Exception as exc:
                logger.warning("Could not remove worktree %s: %s", wt, exc)
    except Exception as exc:
        logger.warning("Could not list worktrees in %s: %s", repo_path, exc)


def cleanup_old_clones(prefix: str = CLONE_PREFIX):
    """Remove leaked dedicated clones from prior crashed runs (bfe-* prefix only)."""
    try:
        parent = Path(prefix).parent
        pattern = Path(prefix).name
        for clone in parent.glob(f"{pattern}-*"):
            if clone.is_dir():
                logger.warning("Removing leaked clone: %s", clone)
                shutil.rmtree(clone, ignore_errors=True)
    except Exception as exc:
        logger.warning("Could not cleanup old clones: %s", exc)


@contextmanager
def dedicated_clone(repo_path: Path, run_id: str):
    """Create a dedicated per-run throwaway clone — NEVER mutates the shared working tree.

    Yields the clone path. Auto-removes on exit.
    The harness NEVER calls `git worktree prune` on any shared clone.
    """
    clone_path = Path(f"{CLONE_PREFIX}-{repo_path.name}-{run_id}")
    try:
        if clone_path.exists():
            shutil.rmtree(clone_path, ignore_errors=True)
        logger.info("Creating dedicated clone: %s", clone_path)
        subprocess.run(
            ["git", "clone", "--local", "--no-hardlinks", str(repo_path), str(clone_path)],
            capture_output=True,
            timeout=60,
            check=True,
        )
        yield clone_path
    finally:
        if clone_path.exists():
            cleanup_leaked_worktrees(clone_path)
            shutil.rmtree(clone_path, ignore_errors=True)
            logger.info("Removed dedicated clone: %s", clone_path)


@contextmanager
def detached_worktree(clone_path: Path, sha: str, wt_name: str = "eval_wt"):
    """Create a detached worktree at sha off the dedicated clone.

    Verifies HEAD == sha before yielding. Auto-removes on exit.
    Never adds a worktree off the shared working tree.
    """
    wt_path = clone_path.parent / wt_name
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_path), sha],
            cwd=clone_path,
            capture_output=True,
            timeout=15,
            check=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        actual_sha = result.stdout.strip()
        if actual_sha != sha:
            raise RuntimeError(f"Worktree HEAD mismatch: expected {sha}, got {actual_sha}")
        yield wt_path
    finally:
        if wt_path.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt_path)],
                cwd=clone_path,
                capture_output=True,
                timeout=10,
            )


# ---------------------------------------------------------------------------
# Corpus loading / saving
# ---------------------------------------------------------------------------


def load_corpus() -> list[FixtureRecord]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    corpus = []
    for json_file in sorted(CORPUS_DIR.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                corpus.append(FixtureRecord(**data))
        except Exception as exc:
            logger.warning("Could not load fixture %s: %s", json_file, exc)
    return corpus


def save_corpus(corpus: list[FixtureRecord]):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for fixture in corpus:
        fixture_id = f"{fixture.repo}-{fixture.sha[:8]}"
        with open(CORPUS_DIR / f"{fixture_id}.json", "w") as f:
            json.dump(asdict(fixture), f, indent=2)
        logger.info("Saved fixture: %s", fixture_id)


def save_run_results(run_id: str, fixture_results: list[FixtureRunResult]):
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "fixture_results.jsonl", "w") as f:
        for result in fixture_results:
            f.write(json.dumps(asdict(result), default=str) + "\n")
    logger.info("Saved run results: %s (%d fixtures)", run_dir, len(fixture_results))


def load_run_results(run_id: str) -> list[FixtureRunResult]:
    results_file = RUNS_DIR / run_id / "fixture_results.jsonl"
    if not results_file.exists():
        return []
    results = []
    with open(results_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "candidates" in data:
                    data["candidates"] = [CandidateResult(**c) for c in data["candidates"]]
                if data.get("selected_candidate"):
                    data["selected_candidate"] = CandidateResult(**data["selected_candidate"])
                results.append(FixtureRunResult(**data))
            except Exception as exc:
                logger.warning("Could not load result: %s", exc)
    return results


# ---------------------------------------------------------------------------
# Pure oracle functions (AC4, AC4b) — CORE, no GW / git dependency
# ---------------------------------------------------------------------------


def extract_module_name(path: str) -> str:
    """Convert a file path to a Python module name.

    E.g. "lapis_pm/pm_core.py" → "lapis_pm.pm_core"
    """
    p = Path(path)
    parts = list(p.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parse_pytest_failures(output: str) -> list[str]:
    """Extract failing test IDs from pytest output.

    Matches lines of the form:
      FAILED tests/test_foo.py::test_bar - AssertionError: ...
    """
    failures = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED "):
            test_id = stripped[len("FAILED "):].split(" - ")[0].strip()
            if test_id:
                failures.append(test_id)
    return failures


def compute_flakiness_fingerprint(
    run_results: list[list[str]],
) -> tuple[list[str], list[str]]:
    """Compute stable and flaky fail sets from N≥3 base runs.

    Args:
        run_results: Each element is the list of failing test IDs from one run.

    Returns:
        (stable_fail_set, flaky_set) where:
        - stable_fail_set: tests that fail on EVERY run
        - flaky_set: tests that fail on SOME but not ALL runs
    """
    if not run_results:
        return [], []
    n = len(run_results)
    fail_counts: dict[str, int] = {}
    for failures in run_results:
        for test_id in failures:
            fail_counts[test_id] = fail_counts.get(test_id, 0) + 1
    stable = sorted(t for t, c in fail_counts.items() if c == n)
    flaky = sorted(t for t, c in fail_counts.items() if 0 < c < n)
    return stable, flaky


def classify_candidate_outcome(
    post_failures: list[str],
    base_stable_fail_set: list[str],
    base_flaky_set: list[str],
    target_test_files: list[str],
    checker_class: str,
) -> Literal["pass", "regressed", "blind", "unverified"]:
    """Classify a candidate's outcome using the flakiness-robust baseline-diff oracle.

    PASS iff ALL three conditions hold:
    1. Every test in base_stable_fail_set still fails at post (baseline unchanged).
    2. No NEW failure at post outside base_stable_fail_set ∪ base_flaky_set.
    3. All target tests transition FAIL@base → PASS@post (not in post_failures).

    Tests in base_flaky_set are ignored on both sides (excluded from differential).
    Returns "blind" for BLIND/UNTESTED fixtures — oracle cannot discriminate.

    NEVER performs a bare returncode==0 check. This is the spec-required oracle (AC4).
    """
    if checker_class in ("BLIND", "UNTESTED") or not target_test_files:
        return "blind"

    post_set = set(post_failures)
    stable_set = set(base_stable_fail_set)
    flaky_set_s = set(base_flaky_set)
    target_set = set(target_test_files)
    excluded = stable_set | flaky_set_s

    # Condition 1: non-target stable failures still fail at post.
    # Target tests are EXPECTED to transition fail→pass, so they are excluded
    # from this check. Only non-target stably-failing tests must stay failing.
    non_target_stable = stable_set - target_set
    missing_stable = non_target_stable - post_set
    if missing_stable:
        # A non-target stably-failing test now passes — baseline assumptions violated
        return "unverified"

    # Condition 2: no new failures outside excluded set
    new_failures = post_set - excluded
    if new_failures:
        return "regressed"

    # Condition 3: target tests transitioned FAIL@base → PASS@post
    still_failing_targets = target_set & post_set
    if still_failing_targets:
        return "regressed"

    return "pass"


def slice_pre_state_from_diff(
    pre_state: str,
    golden_diff: str,
    context_lines: int = 15,
) -> str:
    """Slice pre-fix file to enclosing function/block ± context_lines.

    Parses @@ hunk headers from golden_diff to find the changed region.
    Walks backwards to the nearest def/class. Caps slice at 150 lines.
    Falls back to a 3000-char window if no hunk headers found.
    """
    if not pre_state or not golden_diff:
        return pre_state[:3000]

    lines = pre_state.splitlines()
    if not lines:
        return pre_state[:3000]

    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)
    matches = list(hunk_pattern.finditer(golden_diff))
    if not matches:
        return pre_state[:3000]

    first_line = int(matches[0].group(1)) - 1  # 0-indexed
    last_line = int(matches[-1].group(1)) - 1

    # Walk backwards to enclosing def/class
    enclosing_start = max(0, first_line - context_lines)
    for i in range(first_line, -1, -1):
        if i < len(lines):
            stripped = lines[i].lstrip()
            if (
                stripped.startswith("def ")
                or stripped.startswith("async def ")
                or stripped.startswith("class ")
            ):
                enclosing_start = i
                break

    end_line = min(len(lines), last_line + context_lines + 1)

    slice_lines = lines[enclosing_start:end_line]
    if len(slice_lines) > 150:
        slice_lines = slice_lines[:150]

    return "\n".join(slice_lines)


def validate_corpus_power_floor(
    corpus: list[FixtureRecord],
    tier_floor: int = CORPUS_HOLDOUT_PER_TIER,
) -> dict[str, Any]:
    """Validate corpus meets the holdout power floor.

    Checks strict 3-tier floor first (ALL of {T1,T2,T3} present AND each ≥ floor).
    A corpus missing any tier routes to coarse binary (T1 vs T2+T3).
    Raises ValueError if neither floor is satisfiable.

    Returns per-tier holdout counts plus "_shape" key ("3-tier" or "coarse-binary").
    Callers should treat "_shape" as metadata; tier keys are the holdout counts.
    """
    holdout_by_tier: dict[str, int] = defaultdict(int)
    for f in corpus:
        if f.blind_holdout:
            holdout_by_tier[f.tier] += 1

    present_tiers = {f.tier for f in corpus}
    required_tiers = {"T1", "T2", "T3"}

    # Strict 3-tier floor: ALL three tiers must be present AND each meet the floor.
    # A corpus missing any tier is NOT reported 3-tier regardless of per-tier counts.
    if (
        present_tiers >= required_tiers
        and all(holdout_by_tier.get(t, 0) >= tier_floor for t in required_tiers)
    ):
        result = dict(holdout_by_tier)
        result["_shape"] = "3-tier"
        return result

    # Coarse binary: T1 (small-single-file) vs T2+T3 (larger/cross-cutting)
    small_n = holdout_by_tier.get("T1", 0)
    larger_n = holdout_by_tier.get("T2", 0) + holdout_by_tier.get("T3", 0)
    if small_n >= tier_floor and larger_n >= tier_floor:
        logger.info(
            "3-tier floor not met; collapsed to coarse binary (T1=%d, T2+T3=%d)",
            small_n, larger_n,
        )
        result = dict(holdout_by_tier)
        result["_shape"] = "coarse-binary"
        return result

    raise ValueError(
        f"Corpus power floor not met: holdouts by tier = {dict(holdout_by_tier)}, "
        f"floor = {tier_floor}. Neither 3-tier nor coarse-binary floor is satisfiable. "
        f"Extend corpus extraction or relax holdout floor."
    )


# ---------------------------------------------------------------------------
# Corpus builder helpers (AC1, AC2, AC4b) — CORE, local git only
# ---------------------------------------------------------------------------


def find_scoped_test_files(repo_path: Path, changed_path: str) -> list[str]:
    """Find test files that import the changed module.

    Greps test directories for import statements referencing the module.
    Returns paths relative to repo_path.
    """
    module_name = extract_module_name(changed_path)
    stem = Path(changed_path).stem

    # Build import patterns
    patterns: set[str] = set()
    if "." in module_name:
        pkg, mod = module_name.rsplit(".", 1)
        patterns.add(f"from {module_name} import")
        patterns.add(f"from {pkg} import {mod}")
        patterns.add(f"import {module_name}")
    patterns.add(f"import {stem}")
    patterns.add(f"from {stem} import")
    if stem not in ("__init__", "conftest", "setup", "main"):
        patterns.add(stem)

    test_dirs = ["tests"]
    for extra in ["scripts/tests", "dashboard/cockpit/tests"]:
        if (repo_path / extra).exists():
            test_dirs.append(extra)

    matched: set[str] = set()
    for test_dir_name in test_dirs:
        test_dir = repo_path / test_dir_name
        if not test_dir.exists():
            continue
        for tf in sorted(test_dir.rglob("*.py")):
            name = tf.name
            if not (name.startswith("test_") or name.endswith("_test.py")):
                continue
            try:
                content = tf.read_text(errors="replace")
            except Exception:
                continue
            for pat in patterns:
                if pat in content:
                    try:
                        matched.add(str(tf.relative_to(repo_path)))
                    except ValueError:
                        matched.add(str(tf))
                    break

    return sorted(matched)


def run_scoped_tests_once(
    wt_path: Path,
    test_files: list[str],
    timeout: int = PR_EVAL_TIMEOUT_S,
) -> tuple[list[str], str]:
    """Run scoped tests in a worktree; return (failing_test_ids, raw_outcome).

    raw_outcome: "ran" | "timeout" | "error" | "no-tests"
    Uses -q --tb=no so output is parseable by parse_pytest_failures.
    """
    if not test_files:
        return [], "no-tests"

    cmd = (
        ["python3", "-m", "pytest"]
        + list(test_files)
        + ["-p", "no:cacheprovider", "-q", "--tb=no", "--no-header"]
    )
    try:
        result = subprocess.run(
            cmd,
            cwd=wt_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        failures = parse_pytest_failures(result.stdout + result.stderr)
        return failures, "ran"
    except subprocess.TimeoutExpired:
        logger.warning("Test run timed out in %s", wt_path)
        return [], "timeout"
    except Exception as exc:
        logger.warning("Test run error in %s: %s", wt_path, exc)
        return [], "error"


def run_base_tests_n_times(
    clone_path: Path,
    parent_sha: str,
    sha: str,
    test_files: list[str],
    n: int = FLAKE_RUN_COUNT,
) -> tuple[list[str], list[str], list[str]]:
    """Run scoped tests N≥3 times at parent_sha to build flakiness fingerprint.

    Also runs once at sha to identify target tests (FAIL@base → PASS@sha).

    Returns:
        (stable_fail_set, flaky_set, target_test_files) where:
        - stable_fail_set: test IDs that fail on EVERY base run
        - flaky_set: test IDs that fail on SOME (not all) base runs
        - target_test_files: test IDs in stable_fail_set that pass at sha
          (these are the tests the fix exercises, quarantined if flaky)
    """
    if not test_files:
        return [], [], []

    run_results: list[list[str]] = []
    for i in range(n):
        try:
            with detached_worktree(clone_path, parent_sha, f"base_run_{i}") as wt_path:
                failures, outcome = run_scoped_tests_once(wt_path, test_files)
                if outcome == "timeout":
                    logger.warning("Base run %d timed out; including partial results", i)
                run_results.append(failures)
        except Exception as exc:
            logger.warning("Base run %d failed: %s; skipping this run", i, exc)

    if not run_results:
        logger.warning("All base runs failed for %s at %s", test_files, parent_sha)
        return [], [], []

    stable_fail_set, flaky_set = compute_flakiness_fingerprint(run_results)

    # Run at sha to identify target tests
    target_test_files: list[str] = []
    try:
        with detached_worktree(clone_path, sha, "sha_run") as wt_path:
            sha_failures, sha_outcome = run_scoped_tests_once(wt_path, test_files)
            if sha_outcome == "ran":
                sha_fail_set = set(sha_failures)
                # Targets: stable failures that now pass at sha
                target_test_files = [t for t in stable_fail_set if t not in sha_fail_set]
            else:
                logger.warning("SHA run outcome %r; no target tests identified", sha_outcome)
    except Exception as exc:
        logger.warning("SHA run failed: %s", exc)

    return stable_fail_set, flaky_set, target_test_files


def classify_checker_class(
    scoped_test_files: list[str],
    base_stable_fail_set: list[str],
    target_test_files: list[str],
) -> Literal["DISCRIMINATES", "BLIND", "UNTESTED"]:
    """Classify a fixture's checker_class (three-valued, AC7).

    DISCRIMINATES: scoped tests exist, some stable-fail at base, some flip at sha.
    BLIND: scoped tests exist but no tests stably fail or nothing flips.
    UNTESTED: no scoped tests found.
    """
    if not scoped_test_files:
        return "UNTESTED"
    if target_test_files:
        return "DISCRIMINATES"
    return "BLIND"


def launder_intent(
    task_intent_raw: str,
    mock_mode: bool = False,
) -> tuple[str, Literal["gw", "fallback", "mock"]]:
    """Rewrite task intent as a from-symptom description with solution-naming stripped.

    Real mode: calls call_operator('gravitywell', ...) — zero paid, local 122B.
    Mock mode (AC2 testing / CI): returns a stub paraphrase without calling GW.

    Returns (paraphrased, status) where status is "gw", "fallback", or "mock".
    Callers must track status to detect contamination (raw-body fallback = contaminated).

    The harness ALWAYS feeds task_intent_paraphrased (never raw commit body) to the
    fixer model. This code-path must exist even when the call is mocked (AC2).
    """
    if mock_mode:
        # MOCK: strip Co-Authored-By trailers and obvious solution-naming; never return raw verbatim
        cleaned = re.sub(r"\nCo-Authored-By:.*", "", task_intent_raw, flags=re.DOTALL).strip()
        cleaned = re.sub(r"PR #\d+", "", cleaned).strip()
        paraphrased = f"[Symptom] {cleaned[:200]}" if cleaned else "[Symptom: unspecified]"
        return paraphrased, "mock"

    try:
        from agents_core.llm import call_operator

        prompt = (
            "Rewrite the following task description as a from-symptom specification.\n"
            "Remove ALL solution-naming (e.g. 'change X to Y', 'add import Z', 'set path to ...', "
            "'fix the function').\n"
            "Describe only the observable symptom or failing behavior in 1-2 sentences.\n\n"
            f"Original:\n{task_intent_raw}\n\n"
            "Rewritten symptom description (no solution-naming):"
        )
        result = call_operator("gravitywell", prompt)
        paraphrased = result.strip() if result else task_intent_raw
        return paraphrased, "gw"
    except Exception as exc:
        logger.warning("Intent laundering failed: %s; using cleaned raw", exc)
        cleaned = re.sub(r"\nCo-Authored-By:.*", "", task_intent_raw, flags=re.DOTALL).strip()
        paraphrased = cleaned[:500] if cleaned else task_intent_raw
        return paraphrased, "fallback"


def check_laundering_quality(
    paraphrased: str,
    raw: str,
) -> dict[str, Any]:
    """Spot-check laundering quality on a sample (AC2 — test-only-stub-acceptable).

    DEFERRED: real implementation calls gravitywell as an independent judge.
    Returns pass rates for 'names_no_solution' and 'specifies_task'.
    """
    # DEFERRED SCAFFOLD: real implementation dispatches gravitywell judge
    return {
        "names_no_solution": None,  # DEFERRED: not computed in this bind
        "specifies_task": None,     # DEFERRED: not computed in this bind
        "deferred": True,
    }


def _compute_difficulty_signals(diff: str, pre_state: str) -> dict[str, Any]:
    """Best-effort orthogonal difficulty signals for a fixture."""
    concurrency_patterns = re.compile(
        r"\b(async\s+def|await\s+|asyncio\.|threading\.|lock\.|Lock\(\)|RLock\(\)|Semaphore\()",
        re.IGNORECASE,
    )
    is_concurrency = bool(concurrency_patterns.search(diff) or concurrency_patterns.search(pre_state))

    # Cyclomatic delta: count added branches in diff ('+' lines with if/for/while/and/or)
    branch_pattern = re.compile(r"^\+(?!\+\+)\s+(?:if |for |while |elif |\band\b|\bor\b)", re.MULTILINE)
    cyclomatic_delta = len(branch_pattern.findall(diff))

    # Distinct symbols: unique names after 'def ' or 'class ' in the diff hunks
    symbol_pattern = re.compile(r"^[+-](?![+-])\s*(?:def|class|async\s+def)\s+(\w+)", re.MULTILINE)
    symbols = {m.group(1) for m in symbol_pattern.finditer(diff)}

    return {
        "is_concurrency_code": is_concurrency,
        "golden_diff_cyclomatic_delta": cyclomatic_delta,
        "distinct_symbols_touched": len(symbols),
    }


# ---------------------------------------------------------------------------
# Build corpus (AC1, AC1b, AC2, AC4b) — CORE
# ---------------------------------------------------------------------------


def build_corpus(
    target_size: int = 50,
    tier_floor: int = CORPUS_HOLDOUT_PER_TIER,
    mock_corpus: Optional[list[FixtureRecord]] = None,
    launder_mock_mode: bool = False,
    skip_base_runs: bool = False,
) -> tuple[list[FixtureRecord], dict[str, Any]]:
    """Build the frozen fixture corpus from git history.

    Real mode: extracts commits from lapis-pm + conductor, computes scoped tests,
    runs N≥3 base runs for flakiness fingerprint, classifies checker_class, launders intent.

    Args:
        mock_corpus: If provided, skip git extraction (for unit testing AC13).
        launder_mock_mode: Use mock laundering (no GW call) — for testing AC2 code-path.
        skip_base_runs: Skip N≥3 base test runs (faster corpus build without fingerprint).
    """
    if mock_corpus is not None:
        logger.info("Using mock corpus (%d fixtures)", len(mock_corpus))
        return mock_corpus, {"source": "mock", "fixtures_count": len(mock_corpus)}

    logger.info("Building corpus from git history (lapis-pm + conductor)")

    corpus: list[FixtureRecord] = []
    metadata: dict[str, Any] = {
        "lapis_pm_count": 0,
        "conductor_count": 0,
        "quarantine_count": 0,
        "total_extracted": 0,
        "deduped_dropped": 0,
        "trivial_dropped": 0,
        "flaky_target_quarantine_count": 0,
        "flaky_excluded_count_lapis_pm": 0,
        "flaky_excluded_count_conductor": 0,
        "laundering_total": 0,
        "laundering_fallback_count": 0,
    }

    repos = [
        (LAPIS_PM_REPO, "lapis-pm"),
        (CONDUCTOR_REPO, "conductor"),
    ]

    seen_diffs: set[str] = set()
    fix_pattern = re.compile(r"^fix[\(\:\s]", re.IGNORECASE)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-corpus")

    for repo_path, repo_name in repos:
        if not repo_path.exists():
            logger.warning("Repo not found: %s", repo_path)
            continue

        logger.info("Extracting from %s...", repo_name)

        # Dedicated clone for base-run tests (never touches shared tree)
        clone_path: Optional[Path] = None
        clone_ctx = None
        if not skip_base_runs:
            clone_ctx = dedicated_clone(repo_path, f"{run_id}-{repo_name}")

        def _run_extraction(clone_path_inner: Optional[Path]):
            nonlocal corpus

            try:
                result = subprocess.run(
                    ["git", "log", "--no-merges", "--format=%H %s"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception as exc:
                logger.error("git log failed for %s: %s", repo_name, exc)
                return

            for line in result.stdout.strip().split("\n"):
                if not line.strip() or len(corpus) >= target_size:
                    break

                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                sha, subject = parts[0], parts[1]

                if not fix_pattern.match(subject):
                    continue

                try:
                    # Commit body
                    cb_result = subprocess.run(
                        ["git", "show", "--format=%B", "--no-patch", sha],
                        cwd=repo_path, capture_output=True, text=True, timeout=10,
                    )
                    commit_body = cb_result.stdout.strip()

                    # Changed files
                    ns_result = subprocess.run(
                        ["git", "show", "--numstat", "--format=", sha],
                        cwd=repo_path, capture_output=True, text=True, timeout=10,
                    )
                    changed_files = []
                    total_changed = 0
                    for fline in ns_result.stdout.strip().split("\n"):
                        if not fline.strip():
                            continue
                        fp = fline.split()
                        if len(fp) >= 3:
                            try:
                                chg = int(fp[0]) + int(fp[1])
                                changed_files.append((fp[2], chg))
                                total_changed += chg
                            except ValueError:
                                pass

                    if not changed_files:
                        continue
                    if len(changed_files) > 2:
                        continue
                    if len(changed_files) == 2 and not any("test" in f[0] for f in changed_files):
                        continue
                    if total_changed > 80:
                        continue

                    # Quarantine test-only commits
                    is_test_only = all(
                        "test" in f[0] or f[0].endswith("_test.py")
                        for f in changed_files
                    )
                    if is_test_only:
                        metadata["quarantine_count"] += 1
                        continue

                    # Exclude trivial 1-line value swaps (pure +1/-1 docstring/URL fixes)
                    if total_changed <= 2:
                        metadata["trivial_dropped"] += 1
                        continue

                    # Dedup by diff hash
                    diff_result = subprocess.run(
                        ["git", "show", "--format=", sha],
                        cwd=repo_path, capture_output=True, text=True, timeout=10,
                    )
                    diff_hash = hashlib.md5(diff_result.stdout.encode()).hexdigest()
                    if diff_hash in seen_diffs:
                        metadata["deduped_dropped"] += 1
                        continue
                    seen_diffs.add(diff_hash)

                    # PR number resolution (nullable, deterministic)
                    pr_number: Optional[int] = None
                    pr_match = re.search(r"(?:PR #|pull/|#)(\d+)", commit_body)
                    if pr_match:
                        pr_number = int(pr_match.group(1))

                    # parent SHA
                    parent_result = subprocess.run(
                        ["git", "rev-parse", f"{sha}^"],
                        cwd=repo_path, capture_output=True, text=True, timeout=5,
                    )
                    parent_sha = parent_result.stdout.strip()
                    if not parent_sha:
                        continue

                    # Pick the source (non-test) file
                    source_files = [f for f in changed_files if "test" not in f[0]]
                    if not source_files:
                        continue
                    path = source_files[0][0]

                    # Pre-state content at parent
                    pre_result = subprocess.run(
                        ["git", "show", f"{parent_sha}:{path}"],
                        cwd=repo_path, capture_output=True, text=True, timeout=10,
                    )
                    pre_state = pre_result.stdout if pre_result.returncode == 0 else ""

                    # Golden diff
                    gd_result = subprocess.run(
                        ["git", "show", "--format=", sha, "--", path],
                        cwd=repo_path, capture_output=True, text=True, timeout=10,
                    )
                    golden_diff = gd_result.stdout

                    # Proper pre-state slice using diff hunk headers
                    pre_state_slice = slice_pre_state_from_diff(pre_state, golden_diff)

                    # Tier classification
                    if len(changed_files) == 2 or total_changed > 30:
                        tier: Literal["T1", "T2", "T3"] = "T2"
                    else:
                        tier = "T1"
                    # T3: cross-module / cross-file with > 1 non-test file
                    non_test_files = [f for f in changed_files if "test" not in f[0]]
                    if len(non_test_files) > 1:
                        tier = "T3"

                    # Difficulty signals (best-effort)
                    signals = _compute_difficulty_signals(golden_diff, pre_state)

                    # Intent: use reviewer comment if pr_number resolved, else commit body
                    if pr_number is not None:
                        task_intent_raw = commit_body  # Could query Forgejo; fall back to body
                        intent_source: Literal["commit-body", "reviewer-comment"] = "commit-body"
                        is_reviewer_cycle = False
                    else:
                        task_intent_raw = commit_body
                        intent_source = "commit-body"
                        is_reviewer_cycle = False

                    # Intent laundering (code-path always exists; call mocked in CI)
                    task_intent_paraphrased, launder_status = launder_intent(
                        task_intent_raw, mock_mode=launder_mock_mode
                    )
                    metadata["laundering_total"] += 1
                    if launder_status == "fallback":
                        metadata["laundering_fallback_count"] += 1

                    # Scoped test files (grep importers of changed symbols)
                    scoped_test_files = find_scoped_test_files(repo_path, path)

                    # N≥3 base runs for flakiness fingerprint
                    base_stable_fail_set: list[str] = []
                    base_flaky_set: list[str] = []
                    target_test_files: list[str] = []
                    flaky_excluded = 0

                    if scoped_test_files and clone_path_inner and not skip_base_runs:
                        try:
                            (
                                base_stable_fail_set,
                                base_flaky_set,
                                target_test_files,
                            ) = run_base_tests_n_times(
                                clone_path_inner,
                                parent_sha,
                                sha,
                                scoped_test_files,
                                n=FLAKE_RUN_COUNT,
                            )
                            flaky_excluded = len(base_flaky_set)

                            # Quarantine if target tests are flaky (should not happen by construction
                            # since target_test_files come from stable_fail_set, but sanity-check)
                            if any(t in base_flaky_set for t in target_test_files):
                                logger.warning(
                                    "Fixture %s has flaky target tests — quarantining", sha[:8]
                                )
                                metadata["flaky_target_quarantine_count"] += 1
                                continue

                        except Exception as exc:
                            logger.warning("Base runs failed for %s: %s", sha[:8], exc)

                    if repo_name == "lapis-pm":
                        metadata["flaky_excluded_count_lapis_pm"] += flaky_excluded
                    else:
                        metadata["flaky_excluded_count_conductor"] += flaky_excluded

                    # Classify checker_class
                    checker_class = classify_checker_class(
                        scoped_test_files, base_stable_fail_set, target_test_files
                    )

                    fixture = FixtureRecord(
                        repo=repo_name,
                        sha=sha,
                        parent_sha=parent_sha,
                        pr_number=pr_number,
                        path=path,
                        file_loc=f"line 1-{len(pre_state.splitlines())}",
                        changed_lines=total_changed,
                        tier=tier,
                        golden_diff_cyclomatic_delta=signals["golden_diff_cyclomatic_delta"],
                        distinct_symbols_touched=signals["distinct_symbols_touched"],
                        is_concurrency_code=signals["is_concurrency_code"],
                        task_intent_raw=task_intent_raw,
                        task_intent_paraphrased=task_intent_paraphrased,
                        intent_source=intent_source,
                        pre_state_slice=pre_state_slice,
                        golden_diff=golden_diff,
                        scoped_test_files=scoped_test_files,
                        base_stable_fail_set=base_stable_fail_set,
                        base_flaky_set=base_flaky_set,
                        target_test_files=target_test_files,
                        checker_class=checker_class,
                        is_reviewer_cycle=is_reviewer_cycle,
                        is_test_only=False,
                        blind_holdout=False,
                        flaky_excluded_count=flaky_excluded,
                    )

                    corpus.append(fixture)
                    metadata["total_extracted"] += 1
                    if repo_name == "lapis-pm":
                        metadata["lapis_pm_count"] += 1
                    else:
                        metadata["conductor_count"] += 1

                except Exception as exc:
                    logger.warning("Error processing commit %s: %s", sha, exc)

        if clone_ctx is not None:
            with clone_ctx as cp:
                _run_extraction(cp)
        else:
            _run_extraction(None)

    # Assign holdout flags (first tier_floor per tier)
    holdout_counts: dict[str, int] = {}
    for f in corpus:
        if holdout_counts.get(f.tier, 0) < tier_floor:
            f.blind_holdout = True
            holdout_counts[f.tier] = holdout_counts.get(f.tier, 0) + 1

    tier_counts = {t: sum(1 for f in corpus if f.tier == t) for t in ("T1", "T2", "T3")}
    metadata["per_tier_counts"] = tier_counts
    metadata["per_tier_holdout_counts"] = dict(holdout_counts)
    metadata["deduped_seen"] = len(seen_diffs)

    logger.info(
        "Corpus built: %d fixtures, tiers=%s, holdouts=%s",
        len(corpus), tier_counts, holdout_counts,
    )

    return corpus, metadata


# ---------------------------------------------------------------------------
# Grounding hook (AC11 — no-op seam, out-of-scope for v0)
# ---------------------------------------------------------------------------


def grounding_hook(fixture: FixtureRecord) -> str:
    """Expert grounding injection seam. Default no-op. Returns '' always."""
    return ""


# ---------------------------------------------------------------------------
# Candidate generation (DEFERRED run phase — swarm mocked in CI)
# ---------------------------------------------------------------------------


def generate_candidates(
    fixture: FixtureRecord,
    n: int = 1,
    temperature: float = 0.7,
    mock_mode: bool = True,
) -> list[Optional[str]]:
    """Generate N candidate diffs via call_swarm (mocked in CI).

    In mock_mode, returns deterministic canned diffs.
    None entries = swarm health signals (not failed patches) — recorded separately.
    """
    if mock_mode:
        return [
            f"--- a/{fixture.path}\n+++ b/{fixture.path}\n"
            f"@@ -1,3 +1,3 @@\n"
            f" # Mock candidate {i + 1} for {fixture.sha[:8]}\n"
            f"-old line\n+new line\n unchanged\n"
            for i in range(n)
        ]

    from agents_core.llm import call_swarm

    grounding = grounding_hook(fixture)
    prompt = (
        f"Fix the following issue in {fixture.path}:\n\n"
        f"Task: {fixture.task_intent_paraphrased}\n\n"
        f"{('Context:\n' + grounding + chr(10) + chr(10)) if grounding else ''}"
        f"Current code:\n```\n{fixture.pre_state_slice}\n```\n\n"
        "Emit a unified diff in a ```diff ... ``` fence. Minimal changes only."
    )
    system = (
        "You are a code fixer. Emit only a unified diff in a ```diff``` fenced block. "
        "No explanations."
    )

    try:
        raw = call_swarm(
            [prompt] * n,
            system=system,
            temperature=temperature,
            timeout=60,
            max_concurrent=min(4, n),
        )
        extracted = []
        for c in raw:
            if c is None:
                extracted.append(None)
            else:
                m = re.search(r"```diff\n(.*?)\n```", c, re.DOTALL)
                extracted.append(m.group(1) if m else c)
        return extracted
    except Exception as exc:
        logger.error("call_swarm failed: %s", exc)
        return [None] * n


# ---------------------------------------------------------------------------
# Oracle: evaluate one candidate in an isolated worktree (AC3, AC4)
# ---------------------------------------------------------------------------


def oracle_evaluate_candidate(
    clone_path: Path,
    fixture: FixtureRecord,
    candidate_text: str,
    wt_suffix: str,
) -> tuple[Literal["success", "failed", "apply_error"], list[str], Literal["pass", "regressed", "blind", "unverified"]]:
    """Apply a candidate diff and evaluate it using the flakiness-robust oracle.

    Returns (apply_status, post_failures, scoped_test_outcome).
    NEVER uses returncode==0 as the pass criterion (AC4 requirement).
    """
    with detached_worktree(clone_path, fixture.parent_sha, wt_suffix) as wt_path:
        # Apply-check gate
        check = subprocess.run(
            ["git", "apply", "--check"],
            input=candidate_text,
            cwd=wt_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if check.returncode != 0:
            return "failed", [], "unverified"

        # Apply patch
        apply = subprocess.run(
            ["git", "apply"],
            input=candidate_text,
            cwd=wt_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if apply.returncode != 0:
            return "apply_error", [], "unverified"

        # If no scoped tests, oracle cannot discriminate
        if not fixture.scoped_test_files:
            return "success", [], "blind"

        # Run scoped tests
        post_failures, outcome = run_scoped_tests_once(
            wt_path, fixture.scoped_test_files, timeout=PR_EVAL_TIMEOUT_S
        )

        if outcome == "timeout":
            return "success", [], "unverified"
        if outcome == "error":
            return "success", [], "unverified"

        # Classify using flakiness-robust baseline-diff (AC4)
        test_outcome = classify_candidate_outcome(
            post_failures=post_failures,
            base_stable_fail_set=fixture.base_stable_fail_set,
            base_flaky_set=fixture.base_flaky_set,
            target_test_files=fixture.target_test_files,
            checker_class=fixture.checker_class,
        )

        return "success", post_failures, test_outcome


# ---------------------------------------------------------------------------
# Run fixture (DEFERRED: best-of-N execute-select is AC6; scaffold only)
# ---------------------------------------------------------------------------


def run_fixture(
    fixture: FixtureRecord,
    n_range: list[int] = None,
    mock_mode: bool = True,
    repo_path: Optional[Path] = None,
) -> FixtureRunResult:
    """Drive candidates through the oracle for one fixture.

    The per-candidate oracle evaluation (apply + classify) is CORE (AC4).
    The execute-select (keep first passing) and held-out validation are
    DEFERRED (AC6, AC6b) — scaffolded below but NOT claimed green.

    mock_mode: mocks candidate GENERATION only (swarm); oracle is real code.
    """
    if n_range is None:
        n_range = N_CANDIDATES_RANGE

    fixture_id = f"{fixture.repo}-{fixture.sha[:8]}"
    max_n = max(n_range)

    candidates_raw = generate_candidates(fixture, n=max_n, mock_mode=mock_mode)

    result = FixtureRunResult(
        fixture_id=fixture_id,
        tier=fixture.tier,
        repo=fixture.repo,
        n_candidates_generated=max_n,
        none_count=sum(1 for c in candidates_raw if c is None),
    )

    if mock_mode:
        # DEFERRED SCAFFOLD: mock mode for testing deferred AC5/AC6 surfaces.
        # The oracle is NOT invoked here — mock mode tests swarm generation plumbing only.
        # AC4 oracle tests use classify_candidate_outcome directly (see test_batched_fixer_eval.py).
        for i, candidate_text in enumerate(candidates_raw):
            candidate = CandidateResult(
                fixture_id=fixture_id,
                candidate_seed=i,
                candidate_text=candidate_text,
                apply_status="success" if candidate_text else "apply_error",
                scoped_test_outcome="pass" if i == 0 else "unverified",  # SCAFFOLD
            )
            result.candidates.append(candidate)
        # DEFERRED: execute-select (AC6)
        result.execute_select_outcome = "unverified"  # DEFERRED
        return result

    # Real mode: oracle evaluation per candidate
    if repo_path is None:
        repo_path = LAPIS_PM_REPO if fixture.repo == "lapis-pm" else CONDUCTOR_REPO

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    try:
        with dedicated_clone(repo_path, run_id) as clone_path:
            for i, candidate_text in enumerate(candidates_raw):
                if candidate_text is None:
                    result.candidates.append(
                        CandidateResult(
                            fixture_id=fixture_id,
                            candidate_seed=i,
                            candidate_text=None,
                            apply_status="apply_error",
                            scoped_test_outcome="unverified",
                        )
                    )
                    continue

                try:
                    apply_status, post_failures, test_outcome = oracle_evaluate_candidate(
                        clone_path, fixture, candidate_text, f"wt_{run_id}_{i}"
                    )
                    result.candidates.append(
                        CandidateResult(
                            fixture_id=fixture_id,
                            candidate_seed=i,
                            candidate_text=candidate_text,
                            apply_status=apply_status,
                            scoped_test_outcome=test_outcome,
                            test_failures=post_failures,
                        )
                    )
                except Exception as exc:
                    logger.warning("Candidate %d oracle error: %s", i, exc)
                    result.candidates.append(
                        CandidateResult(
                            fixture_id=fixture_id,
                            candidate_seed=i,
                            candidate_text=candidate_text,
                            apply_status="apply_error",
                            apply_error=str(exc),
                            scoped_test_outcome="unverified",
                        )
                    )

            # DEFERRED (AC6): execute-select — find first passing candidate
            # Scaffolded here; NOT claimed green in this bind.
            passing = [c for c in result.candidates if c.scoped_test_outcome == "pass"]
            if passing:
                result.execute_select_outcome = "pass"  # DEFERRED AC6
                result.selected_candidate = passing[0]  # DEFERRED AC6
                result.first_passing_candidate_seed = passing[0].candidate_seed  # DEFERRED AC6
                result.passing_candidate_count = len(passing)  # DEFERRED AC6
            else:
                result.execute_select_outcome = "no_passing_candidate"  # DEFERRED AC6

            # DEFERRED (AC6b): held-out validation of selected candidate
            # Scaffold only — not implemented in this bind.
            result.held_out_test_outcome = "unverified"  # DEFERRED AC6b

    except Exception as exc:
        logger.error("Fixture %s run failed: %s", fixture_id, exc)
        result.execute_select_outcome = "unverified"

    return result


# ---------------------------------------------------------------------------
# Wilson CI + aggregate (DEFERRED: AC5, AC8 — scaffold only)
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success rate."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = successes / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * ((p_hat * (1 - p_hat) / n) + z * z / (4 * n * n)) ** 0.5 / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def aggregate_results(
    fixture_results: list[FixtureRunResult],
    corpus: list[FixtureRecord],
) -> dict[str, dict[int, TierMetrics]]:
    """Aggregate per-fixture results into decision table. DEFERRED: AC5, AC8."""
    # DEFERRED SCAFFOLD — this aggregation is AC8 (follow-on leg)
    corpus_by_id = {f"{f.repo}-{f.sha[:8]}": f for f in corpus}
    results_by_tier: dict[str, list[FixtureRunResult]] = defaultdict(list)
    for r in fixture_results:
        results_by_tier[r.tier].append(r)

    tier_metrics: dict[str, dict[int, TierMetrics]] = {}
    for tier, tier_results in results_by_tier.items():
        tier_metrics[tier] = {}
        holdout = [r for r in tier_results if corpus_by_id.get(r.fixture_id, FixtureRecord(
            repo="", sha="", parent_sha="", pr_number=None, path="",
            file_loc="", changed_lines=0, tier="T1",
        )).blind_holdout]

        for n in N_CANDIDATES_RANGE:
            total = len(holdout)
            if total == 0:
                continue
            successes = sum(1 for r in holdout if r.execute_select_outcome == "pass")
            rate = successes / total
            ci = wilson_ci(successes, total)

            tier_metrics[tier][n] = TierMetrics(
                tier=tier, n=n,
                repo_breakdown={},
                execute_success_rate=rate,
                execute_success_ci=ci,
                candidates_to_first_pass_median=0.0,
                selected_but_fails_held_out_rate=0.0,
                held_out_ci=(0.0, 1.0),
                holdout_n=total,
                held_out_fail_gate_passed=True,
                routing_recommendation="big-lane",  # DEFERRED: gates not applied yet
                gate_failure_reason="DEFERRED: AC8 not implemented in this bind",
            )

    return tier_metrics


def generate_report(
    eval_result: EvalResult,
    corpus: list[FixtureRecord],
    mock_mode: bool = True,
) -> str:
    """Generate verdict markdown. DEFERRED: AC12 (follow-on leg)."""
    return (
        f"# Batched-Fixer Patch-Quality Eval — {eval_result.run_id}\n\n"
        f"**DEFERRED**: Report generation is AC12, implemented in the follow-on leg.\n\n"
        f"Generated: {eval_result.generated_at}\n"
        f"Served model: {eval_result.served_model_id} ({eval_result.served_model_backend})\n"
    )


# ---------------------------------------------------------------------------
# Corpus manifest helper
# ---------------------------------------------------------------------------


def _write_corpus_manifest(metadata: dict[str, Any]) -> None:
    """Write corpus metadata to _manifest.json in CORPUS_DIR for ratification checkpoint."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CORPUS_DIR / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Corpus manifest written: %s", manifest_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_eval(
    phase: Literal["build-corpus", "run", "report"] = "run",
    run_id: Optional[str] = None,
    mock_mode: bool = False,
) -> Optional[EvalResult]:
    """Main harness entry point.

    CORE phases (this bind): build-corpus.
    DEFERRED phases (follow-on leg): run, report.
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    logger.info("=== Batched-Fixer Eval %s (phase=%s, mock=%s) ===", run_id, phase, mock_mode)
    cleanup_old_clones()

    try:
        if phase == "build-corpus":
            # Startup probe: verify GW laundering is live before building anything (real mode only).
            # A down or wrong-mode GW causes every fixture to silently use raw-body intent
            # (contaminated). Fail loud before building, not after.
            if not mock_mode:
                _, probe_status = launder_intent("probe", mock_mode=False)
                if probe_status == "fallback":
                    raise RuntimeError(
                        "GW not serving big-122B — intent-laundering would silently degrade to "
                        "raw-body (contaminated) intent. Wake GW and run `gw-serve big`, confirm "
                        "doorman /status serving:true, then re-run build-corpus."
                    )

            corpus, metadata = build_corpus(launder_mock_mode=mock_mode)

            floor_result = validate_corpus_power_floor(corpus)
            corpus_shape = floor_result.pop("_shape", "unknown")
            metadata["corpus_shape"] = corpus_shape

            # Surface contamination loudly — a buried logger.warning is not enough.
            fallback_count = metadata.get("laundering_fallback_count", 0)
            if fallback_count > 0:
                total = metadata.get("laundering_total", 0)
                print(
                    f"\n⚠ CONTAMINATED CORPUS: {fallback_count}/{total} fixtures used "
                    f"raw-body intent (laundering fallback). The PM checkpoint must require "
                    f"this be 0; re-run with GW serving big.",
                    file=sys.stderr,
                )

            _write_corpus_manifest(metadata)
            save_corpus(corpus)
            logger.info("Corpus built: %d fixtures, metadata: %s", len(corpus), metadata)
            return None

        if phase == "run":
            # DEFERRED: AC5/AC6 run phase is implemented in the follow-on leg.
            # Scaffold: load corpus + run oracle on each fixture (no execute-select).
            if not mock_mode:
                if not assert_swarm_health(strict=True):
                    logger.error("Swarm health check failed; aborting (see §4 procedure)")
                    return None

            corpus = load_corpus()
            if not corpus:
                logger.warning("No corpus loaded; run 'build-corpus' first")
                return None

            fixture_results = []
            for fixture in corpus:
                try:
                    r = run_fixture(fixture, mock_mode=mock_mode)
                    fixture_results.append(r)
                except Exception as exc:
                    logger.error("Fixture %s failed: %s", fixture.sha[:8], exc)

            save_run_results(run_id, fixture_results)
            tier_metrics = aggregate_results(fixture_results, corpus)
            return EvalResult(
                run_id=run_id,
                generated_at=datetime.now(timezone.utc).isoformat(),
                served_model_id=EXPECTED_SWARM_MODEL if not mock_mode else "mock",
                served_model_backend="unknown",
                tier_metrics=tier_metrics,
            )

        if phase == "report":
            logger.warning("Report phase is DEFERRED (AC12) — implemented in follow-on leg")
            return None

    except RuntimeError:
        raise
    except Exception as exc:
        logger.exception("Eval phase %s failed: %s", phase, exc)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_eval(mock_mode=True)
    if result:
        print(f"Result: {result}")
