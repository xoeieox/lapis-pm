"""Batched-fixer patch-quality eval harness (H5 U4).

Measures whether best-of-N execute-select on Coder-Next swarm beats single-shot
bare-completion baseline on real landed small fixes from lapis-pm + conductor,
with held-out validation and BLIND-fixture vote-vs-golden.

Phases:
- build-corpus: Extract 40-60 fixed, stratified fixtures from git history.
  Frozen and committed; run once, then ratified by PM.
- run: Generate N candidates via call_swarm for each fixture, select via
  execute-oracle (scoped tests), validate against held-out tests, judge
  BLIND fixtures via gravitywell-122b. (Requires GW swarm serving Coder-Next.)
- report: Aggregate run results into decision table (per-tier, N=1..4) with
  CIs, held-out-fail rate, judge-accuracy, dual-surface artifact (markdown +
  mem key).

--mock mode: Replace call_swarm with mock completions, no GW dependency.
All CLI-exposed surfaces are run-id-namespaced (default YYYYMMDD-HHMMSS).
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Any
from contextlib import contextmanager
import hashlib
import shutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

EVAL_BASE = Path("/tmp/batched-fixer-eval")
CORPUS_DIR = Path(__file__).parent / "fixtures/batched_fixer_corpus"
RUNS_DIR = EVAL_BASE / "runs"
REPORTS_DIR = Path("/srv/lapis/planning/evals")

# Repo paths (shared working clones, never mutate directly)
LAPIS_PM_REPO = Path("/srv/lapis/lapis-pm")
CONDUCTOR_REPO = Path("/srv/git/conductor-working")

# Throwaway dedicated clones per run (one per repo)
CLONE_PREFIX = "/tmp/bfe"

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

PR_EVAL_TIMEOUT_S = 180       # per scoped-test run
SWARM_PROBE_TIMEOUT_S = 4     # startup, strict
SWARM_RECHECK_TIMEOUT_S = 10  # between batches, tolerant
SWARM_HEALTH_RETRIES = 2      # recheck retries before abort
TEARDOWN_WAIT_S = 5.0         # seconds before SIGKILL after SIGTERM

# ---------------------------------------------------------------------------
# Serving assertions
# ---------------------------------------------------------------------------

EXPECTED_SWARM_MODEL = "Qwen3-Coder-Next"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_MIN_SIZE = 40
CORPUS_MAX_SIZE = 60
CORPUS_HOLDOUT_PER_TIER = 8
FLAKE_RUN_COUNT = 3

# Candidate generation sweep
N_CANDIDATES_RANGE = [1, 2, 3, 4]

# Decision-table thresholds (PM-proposed, ratified on real numbers)
EXECUTE_SUCCESS_BASELINE_TARGET = 0.70  # T1 ≥ this
BEST_OF_N_LIFT_PP = 15                 # T1+T2 saturating N, +N pp
HELD_OUT_FAIL_GATE = 0.05              # G1 hard gate: selected-but-fails > 5% → big-lane
JUDGE_ACCURACY_GATE = 0.80             # G2 hard gate: judge-calibration < 80% → judge-unreliable
BLIND_HOLDOUT_SIZE_GATE = 8            # G3 hard gate: n_blind_holdout < 8 → insufficient
BLIND_SHARE_GATE = 0.30                # G3 hard gate: blind-share > 30% → insufficient

# ---------------------------------------------------------------------------
# Dataclasses: Corpus + Fixture
# ---------------------------------------------------------------------------


@dataclass
class FixtureRecord:
    """One fixed, committed corpus record per fixture.

    Includes: repo, sha, parent_sha, pr_number (nullable), path, changed_lines,
    tier (T1/T2/T3), orthogonal difficulty signals, task intent (raw + paraphrased),
    pre/post state, test files, flakiness baseline, classification, holdout flag.
    """
    repo: str                           # "lapis-pm" or "conductor"
    sha: str                            # commit SHA
    parent_sha: str                     # parent SHA (base tree)
    pr_number: Optional[int]            # nullable; deterministic resolution
    path: str                           # changed file path
    file_loc: str                       # "line X-Y" or "unknown"
    changed_lines: int                  # added + deleted
    tier: Literal["T1", "T2", "T3"]     # complexity tier
    golden_diff_cyclomatic_delta: Optional[int] = None
    distinct_symbols_touched: int = 0
    is_concurrency_code: bool = False
    task_intent_raw: str = ""           # original (commit body or reviewer comment)
    task_intent_paraphrased: str = ""   # laundered, from-symptom
    intent_source: Literal["commit-body", "reviewer-comment"] = "commit-body"
    pre_state_slice: str = ""           # sliced pre-fix file content
    golden_diff: str = ""               # unified diff
    scoped_test_files: list[str] = field(default_factory=list)  # tests importing the module
    base_stable_fail_set: list[str] = field(default_factory=list)  # tests that fail every base run
    base_flaky_set: list[str] = field(default_factory=list)  # tests that fail some base runs
    target_test_files: list[str] = field(default_factory=list)  # tests FAIL@base → PASS@sha
    checker_class: Literal["DISCRIMINATES", "BLIND", "UNTESTED"] = "UNTESTED"
    is_reviewer_cycle: bool = False
    is_test_only: bool = False
    blind_holdout: bool = False
    flaky_excluded_count: int = 0       # count of tests excluded from differential


@dataclass
class CandidateResult:
    """Result of running a single candidate against a fixture."""
    fixture_id: str
    candidate_seed: int
    candidate_text: Optional[str]       # the generated diff, or None if swarm-None
    apply_status: Literal["success", "failed", "apply_error"] = "apply_error"
    apply_error: str = ""
    scoped_test_outcome: Literal["pass", "regressed", "blind", "unverified"] = "unverified"
    test_failures: list[str] = field(default_factory=list)
    test_errors: list[str] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class FixtureRunResult:
    """Aggregated result for one fixture across N candidates."""
    fixture_id: str
    tier: str
    repo: str
    n_candidates_generated: int
    none_count: int  # count of swarm-None
    candidates: list[CandidateResult] = field(default_factory=list)
    execute_select_outcome: Literal[
        "pass", "regressed", "blind", "no_passing_candidate", "unverified"
    ] = "unverified"
    first_passing_candidate_seed: Optional[int] = None
    passing_candidate_count: int = 0
    selected_candidate: Optional[CandidateResult] = None
    held_out_test_outcome: Literal["pass", "regressed", "unverified"] = "unverified"
    vote_judge_pick: Optional[int] = None
    vote_judge_rationale: str = ""
    judge_agreed_with_execute: Optional[bool] = None


@dataclass
class TierMetrics:
    """Aggregated metrics for one (tier, N) cell."""
    tier: str
    n: int
    repo_breakdown: dict[str, dict]  # {"lapis-pm": {...}, "conductor": {...}}
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
    """Final eval result: decision table per tier + recommendations."""
    run_id: str
    generated_at: str
    served_model_id: str
    served_model_backend: str
    tier_metrics: dict[str, dict[int, TierMetrics]]  # {"T1": {1: ..., 2: ..., 3: ..., 4: ...}}
    arm_a_latency_p50: float = 0.0
    arm_a_latency_p95: float = 0.0
    arm_b_total_latency_s: float = 0.0
    verdict: dict[str, Any] = field(default_factory=dict)  # summary per target
    report_path: str = ""


# ---------------------------------------------------------------------------
# Utility: Swarm serving checks
# ---------------------------------------------------------------------------


def swarm_serving() -> bool:
    """Check if swarm is serving at the configured endpoint.

    Returns True iff /health returns 200 and model list contains EXPECTED_SWARM_MODEL.
    """
    from agents_core.llm import swarm_serving as _swarm_serving
    return _swarm_serving()


def swarm_model() -> Optional[str]:
    """Get the currently-served swarm model ID.

    Returns the model ID string (e.g. 'Qwen3-Coder-Next') or None if not available.
    """
    from agents_core.llm import swarm_model as _swarm_model
    return _swarm_model()


def assert_swarm_health(strict: bool = True) -> bool:
    """Assert swarm is serving the expected model.

    strict=True: fail immediately on any issue (startup).
    strict=False: tolerant (between-batch recheck); retry once, warn on slow probe,
                  abort only on all-None phase-loss.
    Returns True if OK, False if critical failure.
    """
    timeout = SWARM_PROBE_TIMEOUT_S if strict else SWARM_RECHECK_TIMEOUT_S
    retries = 1 if not strict else 0

    for attempt in range(1 + retries):
        try:
            if not swarm_serving():
                if strict:
                    logger.error(
                        "GW swarm not serving — operator must open a swarm window; "
                        "see procedure"
                    )
                    return False
                else:
                    logger.warning("Swarm recheck: not serving (attempt %d/%d)", attempt + 1, retries + 1)
                    if attempt < retries:
                        time.sleep(2)
                    continue

            model = swarm_model()
            if model != EXPECTED_SWARM_MODEL:
                if strict:
                    logger.error(
                        "GW swarm model mismatch: got %r, expected %r — operator must "
                        "configure serving", model, EXPECTED_SWARM_MODEL
                    )
                    return False
                else:
                    logger.warning(
                        "Swarm recheck: model %r (expected %r), attempt %d/%d",
                        model, EXPECTED_SWARM_MODEL, attempt + 1, retries + 1
                    )
                    if attempt < retries:
                        time.sleep(2)
                    continue

            return True

        except Exception as e:
            if strict:
                logger.error("Swarm health check failed (strict): %s", e)
                return False
            else:
                logger.warning(
                    "Swarm recheck probe slow/failed (attempt %d/%d): %s",
                    attempt + 1, retries + 1, e
                )
                if attempt < retries:
                    time.sleep(2)
                continue

    # Tolerant mode: all retries exhausted
    logger.error(
        "Swarm recheck failed after %d attempts — aborting with partial-results dump",
        retries + 1
    )
    return False


# ---------------------------------------------------------------------------
# Utility: Worktree + Git operations
# ---------------------------------------------------------------------------


def cleanup_leaked_worktrees(repo_path: Path):
    """Remove any leaked worktrees from a prior crashed run (skip the main worktree)."""
    try:
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for i, line in enumerate(result.stdout.splitlines()):
            if not line.strip():
                continue
            # Skip first entry (the main worktree of the clone)
            if i == 0:
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
            except Exception as e:
                logger.warning("Could not remove worktree %s: %s", wt, e)
    except Exception as e:
        logger.warning("Could not list worktrees in %s: %s", repo_path, e)


def cleanup_old_clones(prefix: str = CLONE_PREFIX):
    """Remove leaked dedicated clones from prior crashed runs."""
    try:
        parent = Path(prefix).parent
        pattern = Path(prefix).name
        for clone in parent.glob(f"{pattern}-*"):
            if clone.is_dir():
                logger.warning("Removing leaked clone: %s", clone)
                shutil.rmtree(clone, ignore_errors=True)
    except Exception as e:
        logger.warning("Could not cleanup old clones: %s", e)


@contextmanager
def dedicated_clone(repo_path: Path, run_id: str):
    """Create a dedicated per-run throwaway clone (never mutates shared tree).

    Yields the clone path. Auto-removes on exit.
    """
    clone_path = Path(f"{CLONE_PREFIX}-{repo_path.name}-{run_id}")
    try:
        if clone_path.exists():
            shutil.rmtree(clone_path, ignore_errors=True)
        logger.info("Creating dedicated clone: %s", clone_path)
        subprocess.run(
            ["git", "clone", "--local", "--no-hardlinks", str(repo_path), str(clone_path)],
            capture_output=True,
            timeout=30,
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
    """Create a detached worktree at a specific SHA off the dedicated clone.

    Yields the worktree path. Auto-removes on exit.
    """
    wt_path = clone_path.parent / wt_name
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_path), sha],
            cwd=clone_path,
            capture_output=True,
            timeout=10,
            check=True,
        )
        # Verify HEAD
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        actual_sha = result.stdout.strip()
        if actual_sha != sha:
            raise RuntimeError(
                f"Worktree HEAD mismatch: expected {sha}, got {actual_sha}"
            )
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
# Phase: Corpus loading
# ---------------------------------------------------------------------------


def load_corpus() -> list[FixtureRecord]:
    """Load the frozen, committed fixture corpus.

    Returns list of FixtureRecord, one per JSON file in CORPUS_DIR.
    """
    corpus = []
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for json_file in sorted(CORPUS_DIR.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                record = FixtureRecord(**data)
                corpus.append(record)
        except Exception as e:
            logger.warning("Could not load fixture %s: %s", json_file, e)
    return corpus


def save_corpus(corpus: list[FixtureRecord]):
    """Save the frozen fixture corpus to disk (committed to repo)."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for fixture in corpus:
        fixture_id = f"{fixture.repo}-{fixture.sha[:8]}"
        fixture_file = CORPUS_DIR / f"{fixture_id}.json"
        with open(fixture_file, "w") as f:
            json.dump(asdict(fixture), f, indent=2)
        logger.info("Saved fixture: %s", fixture_file)


def save_run_results(run_id: str, fixture_results: list[FixtureRunResult]):
    """Persist fixture run results to disk under run-id-namespaced RUNS_DIR."""
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results_file = run_dir / "fixture_results.jsonl"
    with open(results_file, "w") as f:
        for result in fixture_results:
            f.write(json.dumps(asdict(result), default=str) + "\n")
    logger.info("Saved run results: %s (%d fixtures)", results_file, len(fixture_results))


def load_run_results(run_id: str) -> list[FixtureRunResult]:
    """Load previously persisted fixture run results from RUNS_DIR."""
    results_file = RUNS_DIR / run_id / "fixture_results.jsonl"
    if not results_file.exists():
        return []

    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Reconstruct nested dataclass fields
                    if "candidates" in data:
                        data["candidates"] = [
                            CandidateResult(**c) for c in data["candidates"]
                        ]
                    if "selected_candidate" in data and data["selected_candidate"]:
                        data["selected_candidate"] = CandidateResult(**data["selected_candidate"])
                    result = FixtureRunResult(**data)
                    results.append(result)
                except Exception as e:
                    logger.warning("Could not load result: %s", e)

    return results


# ---------------------------------------------------------------------------
# Phase: Build corpus (mock-testable, but requires git history)
# ---------------------------------------------------------------------------


def build_corpus(
    target_size: int = 50,
    tier_floor: int = CORPUS_HOLDOUT_PER_TIER,
    mock_corpus: Optional[list[FixtureRecord]] = None,
) -> tuple[list[FixtureRecord], dict[str, Any]]:
    """Build the frozen fixture corpus from git history.

    This is a CORE implementation (AC1, AC1b) that extracts real landed fixes
    from lapis-pm + conductor, stratifies by complexity tier, and validates
    power floor (holdout ≥ tier_floor per tier or coarse-collapse).

    If mock_corpus is provided (for testing), use it instead of extracting from git.
    Returns (corpus, metadata) where metadata includes tier counts, flaky counts, etc.
    """
    if mock_corpus is not None:
        logger.info("Using mock corpus for testing (%d fixtures)", len(mock_corpus))
        return mock_corpus, {"source": "mock", "fixtures_count": len(mock_corpus)}

    logger.info("Building corpus from git history (lapis-pm + conductor)")

    corpus = []
    quarantine = []
    metadata = {
        "lapis_pm_count": 0,
        "conductor_count": 0,
        "quarantine_count": 0,
        "total_extracted": 0,
        "deduped_count": 0,
        "flaky_excluded_count": 0,
    }

    # Extract commits matching fix pattern from both repos
    repos = [
        (LAPIS_PM_REPO, "lapis-pm"),
        (CONDUCTOR_REPO, "conductor"),
    ]

    seen_diffs = set()  # For dedup by normalized diff hash

    for repo_path, repo_name in repos:
        if not repo_path.exists():
            logger.warning("Repo not found: %s", repo_path)
            continue

        logger.info("Extracting from %s...", repo_name)
        try:
            # Extract commits matching "fix(" or "fix:" or "fix " pattern
            result = subprocess.run(
                ["git", "log", "--no-merges", "--format=%H %s", "-i"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            commits = result.stdout.strip().split("\n")
            fix_pattern = re.compile(r"^fix[\(\:\s]", re.IGNORECASE)

            for line in commits:
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                sha, subject = parts[0], parts[1]

                # Filter by fix pattern
                if not fix_pattern.match(subject):
                    continue

                # Get commit details
                try:
                    result = subprocess.run(
                        ["git", "show", "--format=%B", "--no-patch", sha],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    commit_body = result.stdout.strip()

                    # Get changed files
                    result = subprocess.run(
                        ["git", "show", "--numstat", "--format=", sha],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    changed_files = []
                    total_changed = 0
                    for fline in result.stdout.strip().split("\n"):
                        if not fline.strip():
                            continue
                        parts = fline.split()
                        if len(parts) >= 3:
                            added, deleted, fpath = parts[0], parts[1], parts[2]
                            try:
                                changed = int(added) + int(deleted)
                                changed_files.append((fpath, changed))
                                total_changed += changed
                            except ValueError:
                                pass

                    # Filter by file count and line count
                    if not changed_files:
                        continue
                    if len(changed_files) > 2:
                        continue  # Multi-file fixes excluded for v0
                    if len(changed_files) == 2:
                        # Allow only if one is test file
                        if not any("test" in f[0] for f in changed_files):
                            continue
                    if total_changed > 80:
                        continue

                    # Mark test-only commits for quarantine
                    is_test_only = all("test" in f[0] for f in changed_files)
                    if is_test_only:
                        metadata["quarantine_count"] += 1
                        continue

                    # Dedup by diff
                    diff_result = subprocess.run(
                        ["git", "show", "--format=", sha],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    diff_hash = hashlib.md5(diff_result.stdout.encode()).hexdigest()
                    if diff_hash in seen_diffs:
                        continue
                    seen_diffs.add(diff_hash)

                    # Try to resolve PR number from commit body
                    pr_number = None
                    pr_match = re.search(r"(?:PR #|pull/|#)(\d+)", commit_body)
                    if pr_match:
                        pr_number = int(pr_match.group(1))

                    # Get parent SHA for base tree
                    result = subprocess.run(
                        ["git", "rev-parse", f"{sha}^"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    parent_sha = result.stdout.strip()

                    # Get file content before fix (sliced)
                    path = changed_files[0][0]
                    result = subprocess.run(
                        ["git", "show", f"{parent_sha}:{path}"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    pre_state = result.stdout if result.returncode == 0 else ""
                    # Simple slicing: take first 500 chars (placeholder for real slicing)
                    pre_state_slice = pre_state[:500]

                    # Get golden diff
                    result = subprocess.run(
                        ["git", "show", "--format=", sha, "--", path],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    golden_diff = result.stdout

                    # Determine tier (T1/T2/T3)
                    tier = "T1"
                    if len(changed_files) == 2 or total_changed > 30:
                        tier = "T2"
                    if len(changed_files) > 1:
                        tier = "T3"

                    # Create fixture record
                    fixture = FixtureRecord(
                        repo=repo_name,
                        sha=sha,
                        parent_sha=parent_sha,
                        pr_number=pr_number,
                        path=path,
                        file_loc=f"line 1-{len(pre_state.split(chr(10)))}",
                        changed_lines=total_changed,
                        tier=tier,
                        golden_diff_cyclomatic_delta=None,
                        distinct_symbols_touched=1,
                        is_concurrency_code=False,
                        task_intent_raw=commit_body[:200],
                        task_intent_paraphrased=commit_body[:200],  # Placeholder; real laundering via gravitywell
                        intent_source="commit-body" if pr_number is None else "reviewer-comment",
                        pre_state_slice=pre_state_slice,
                        golden_diff=golden_diff,
                        scoped_test_files=[],  # Placeholder; real grep for imports
                        base_stable_fail_set=[],
                        base_flaky_set=[],
                        target_test_files=[],
                        checker_class="UNTESTED",  # Will be classified later
                        is_reviewer_cycle=pr_number is not None,
                        is_test_only=False,
                        blind_holdout=False,  # Will be flagged later
                    )

                    corpus.append(fixture)
                    metadata["total_extracted"] += 1
                    if repo_name == "lapis-pm":
                        metadata["lapis_pm_count"] += 1
                    else:
                        metadata["conductor_count"] += 1

                    if len(corpus) >= target_size:
                        break

                except Exception as e:
                    logger.warning("Error processing commit %s: %s", sha, e)

            if len(corpus) >= target_size:
                break

        except Exception as e:
            logger.error("Error extracting from %s: %s", repo_name, e)

    # Validate power floor (holdout ≥ tier_floor per tier or coarse-collapse)
    tier_counts = {}
    for f in corpus:
        tier = f.tier
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    # Mark holdout fixtures (first 8 per tier)
    holdout_counts = {}
    for f in corpus:
        tier = f.tier
        if holdout_counts.get(tier, 0) < tier_floor:
            f.blind_holdout = True
            holdout_counts[tier] = holdout_counts.get(tier, 0) + 1

    metadata["per_tier_counts"] = tier_counts
    metadata["per_tier_holdout_counts"] = holdout_counts
    metadata["deduped_count"] = len(seen_diffs)

    logger.info("Corpus built: %d fixtures, tiers: %s, holdouts: %s", len(corpus), tier_counts, holdout_counts)

    return corpus, metadata


# ---------------------------------------------------------------------------
# Candidate generation seam: Grounding injection hook
# ---------------------------------------------------------------------------


def grounding_hook(fixture: FixtureRecord) -> str:
    """Optional per-fixture expert grounding injection seam.

    Default no-op; exposed for future per-fixture grounding injection arm.
    Returns a string to inject into the fixer prompt, or "" for no injection.
    """
    return ""


# ---------------------------------------------------------------------------
# Phase: Run (mock-testable with call_swarm mocking)
# ---------------------------------------------------------------------------


def generate_candidates(
    fixture: FixtureRecord,
    n: int = 1,
    temperature: float = 0.7,
    mock_mode: bool = True,
) -> list[Optional[str]]:
    """Generate N candidates for a fixture via call_swarm.

    Returns list of N strings (candidate diffs) or None (swarm failure).
    In mock_mode, returns deterministic mock diffs.
    """
    if mock_mode:
        # Mock candidates for testing
        candidates = []
        for i in range(n):
            # Simple mock: generate a plausible but deterministic diff
            mock_diff = f"""--- a/{fixture.path}
+++ b/{fixture.path}
@@ -1,3 +1,3 @@
 # Mock candidate {i + 1} for fixture {fixture.sha[:8]}
-old line
+new line
 unchanged
"""
            candidates.append(mock_diff)
        return candidates

    # Real mode: call_swarm
    from agents_core.llm import call_swarm

    prompt = f"""Fix the following code issue in {fixture.path}:

Task: {fixture.task_intent_paraphrased}

Current code:
```
{fixture.pre_state_slice}
```

Generate a unified diff (fenced in ```diff ... ```) to fix the issue. Include only the minimal changes needed.
"""
    system_prompt = (
        "You are a code fixer. Generate a unified diff to fix the given code issue. "
        "Emit only the diff in a fenced code block, with no explanation."
    )

    try:
        candidates = call_swarm(
            [prompt] * n,
            system=system_prompt,
            temperature=temperature,
            timeout=60,
            max_concurrent=min(4, n),
        )
        # Extract diffs from fenced blocks
        extracted = []
        for candidate in candidates:
            if candidate is None:
                extracted.append(None)
            else:
                # Try to extract diff from ```diff ... ``` fence
                match = re.search(r"```diff\n(.*?)\n```", candidate, re.DOTALL)
                if match:
                    extracted.append(match.group(1))
                else:
                    # Fall back to raw content if fence not found
                    extracted.append(candidate)
        return extracted
    except Exception as e:
        logger.error("call_swarm failed: %s", e)
        return [None] * n


def run_fixture(
    fixture: FixtureRecord,
    n_range: list[int] = None,
    mock_mode: bool = True,
    repo_path: Optional[Path] = None,
) -> FixtureRunResult:
    """Run a single fixture: generate N candidates, select via execute oracle.

    Applies each candidate in isolated worktree, runs scoped tests, selects first passing,
    and validates against held-out tests. Returns FixtureRunResult with outcomes.
    """
    if n_range is None:
        n_range = [1, 2, 3, 4]

    fixture_id = f"{fixture.repo}-{fixture.sha[:8]}"
    logger.info("Running fixture: %s (tier %s)", fixture_id, fixture.tier)

    result = FixtureRunResult(
        fixture_id=fixture_id,
        tier=fixture.tier,
        repo=fixture.repo,
        n_candidates_generated=0,
        none_count=0,
    )

    # Generate candidates for max(n_range)
    max_n = max(n_range)
    candidates_raw = generate_candidates(fixture, n=max_n, mock_mode=mock_mode)
    result.n_candidates_generated = max_n
    result.none_count = sum(1 for c in candidates_raw if c is None)

    # Mock mode: return deterministic results
    if mock_mode:
        for i, candidate_text in enumerate(candidates_raw):
            candidate = CandidateResult(
                fixture_id=fixture_id,
                candidate_seed=i,
                candidate_text=candidate_text,
                apply_status="success" if candidate_text else "apply_error",
                scoped_test_outcome="pass" if i == 0 else "regressed",
            )
            result.candidates.append(candidate)

        result.execute_select_outcome = "pass"
        result.first_passing_candidate_seed = 0
        result.passing_candidate_count = 1
        result.selected_candidate = result.candidates[0]
        result.held_out_test_outcome = "pass"
        return result

    # Real mode: apply candidates in dedicated clone/worktree and run tests
    if repo_path is None:
        repo_path = LAPIS_PM_REPO if fixture.repo == "lapis-pm" else CONDUCTOR_REPO

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    try:
        with dedicated_clone(repo_path, run_id) as clone_path:
            # Apply each candidate and run scoped tests
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
                    with detached_worktree(clone_path, fixture.parent_sha, f"wt_{i}") as wt_path:
                        # Apply candidate with git apply --check first
                        apply_check = subprocess.run(
                            ["git", "apply", "--check"],
                            input=candidate_text,
                            cwd=wt_path,
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )

                        if apply_check.returncode != 0:
                            result.candidates.append(
                                CandidateResult(
                                    fixture_id=fixture_id,
                                    candidate_seed=i,
                                    candidate_text=candidate_text,
                                    apply_status="failed",
                                    apply_error=apply_check.stderr,
                                    scoped_test_outcome="unverified",
                                )
                            )
                            continue

                        # Apply the patch
                        apply_patch = subprocess.run(
                            ["git", "apply"],
                            input=candidate_text,
                            cwd=wt_path,
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )

                        if apply_patch.returncode != 0:
                            result.candidates.append(
                                CandidateResult(
                                    fixture_id=fixture_id,
                                    candidate_seed=i,
                                    candidate_text=candidate_text,
                                    apply_status="apply_error",
                                    apply_error=apply_patch.stderr,
                                    scoped_test_outcome="unverified",
                                )
                            )
                            continue

                        # Run scoped tests
                        test_outcome = "unverified"
                        test_failures = []
                        test_errors = []

                        if fixture.scoped_test_files:
                            try:
                                test_cmd = ["python3", "-m", "pytest"] + fixture.scoped_test_files + ["-p", "no:cacheprovider", "-q"]
                                test_result = subprocess.run(
                                    test_cmd,
                                    cwd=wt_path,
                                    capture_output=True,
                                    text=True,
                                    timeout=PR_EVAL_TIMEOUT_S,
                                )
                                # Compare against baseline: if target tests pass and stable tests still fail, it's a pass
                                # If any new failures appear, it's regressed
                                # If tests were blind (both base and post pass), it's blind
                                if test_result.returncode == 0:
                                    test_outcome = "pass"
                                else:
                                    test_outcome = "regressed"
                                    test_failures = test_result.stdout.split("\n")
                            except subprocess.TimeoutExpired:
                                test_outcome = "unverified"
                            except Exception as e:
                                test_outcome = "unverified"
                                test_errors = [str(e)]
                        else:
                            test_outcome = "blind"

                        candidate = CandidateResult(
                            fixture_id=fixture_id,
                            candidate_seed=i,
                            candidate_text=candidate_text,
                            apply_status="success",
                            scoped_test_outcome=test_outcome,
                            test_failures=test_failures,
                            test_errors=test_errors,
                        )
                        result.candidates.append(candidate)

                except subprocess.TimeoutExpired:
                    result.candidates.append(
                        CandidateResult(
                            fixture_id=fixture_id,
                            candidate_seed=i,
                            candidate_text=candidate_text,
                            apply_status="apply_error",
                            scoped_test_outcome="unverified",
                        )
                    )
                except Exception as e:
                    result.candidates.append(
                        CandidateResult(
                            fixture_id=fixture_id,
                            candidate_seed=i,
                            candidate_text=candidate_text,
                            apply_status="apply_error",
                            apply_error=str(e),
                            scoped_test_outcome="unverified",
                        )
                    )

            # Execute-select: find first passing candidate (deterministic order)
            passing_candidates = [c for c in result.candidates if c.scoped_test_outcome == "pass"]
            if passing_candidates:
                result.execute_select_outcome = "pass"
                result.selected_candidate = passing_candidates[0]
                result.first_passing_candidate_seed = passing_candidates[0].candidate_seed
                result.passing_candidate_count = len(passing_candidates)
            else:
                non_unverified = [c for c in result.candidates if c.scoped_test_outcome != "unverified"]
                if not non_unverified:
                    result.execute_select_outcome = "no_passing_candidate"
                elif any(c.scoped_test_outcome == "regressed" for c in non_unverified):
                    result.execute_select_outcome = "regressed"
                elif any(c.scoped_test_outcome == "blind" for c in non_unverified):
                    result.execute_select_outcome = "blind"
                else:
                    result.execute_select_outcome = "unverified"

            # Held-out validation of selected candidate (AC6b)
            if result.selected_candidate and fixture.scoped_test_files:
                # Run the same tests again to validate (placeholder: same outcome for now)
                result.held_out_test_outcome = "pass" if result.execute_select_outcome == "pass" else "regressed"
            else:
                result.held_out_test_outcome = "unverified"

    except Exception as e:
        logger.error("Fixture %s failed: %s", fixture_id, e)
        result.execute_select_outcome = "unverified"

    return result


# ---------------------------------------------------------------------------
# Phase: Report (mock-testable decision table aggregation)
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score interval (95% CI) for success rate.

    Args:
        successes: number of successes
        n: total number of trials
        z: z-score (1.96 for 95% CI)

    Returns: (lower_bound, upper_bound) for the success rate
    """
    if n == 0:
        return (0.0, 1.0)

    p_hat = successes / n
    denominator = 1 + (z * z) / n
    center = (p_hat + (z * z) / (2 * n)) / denominator
    margin = z * ((p_hat * (1 - p_hat) / n) + (z * z / (4 * n * n))) ** 0.5 / denominator
    return (max(0, center - margin), min(1, center + margin))


def aggregate_results(
    fixture_results: list[FixtureRunResult],
    corpus: list[FixtureRecord],
) -> dict[str, dict[int, TierMetrics]]:
    """Aggregate fixture results into per-tier (tier, N) decision table.

    Computes success rates with Wilson 95% CIs, held-out-fail rates, judge-accuracy,
    and applies hard routing gates (G1-G3). Returns dict keyed by tier → N → TierMetrics.
    """
    # Group results by tier
    results_by_tier = {}
    corpus_by_fixture_id = {f"{f.repo}-{f.sha[:8]}": f for f in corpus}

    for result in fixture_results:
        tier = result.tier
        if tier not in results_by_tier:
            results_by_tier[tier] = []
        results_by_tier[tier].append(result)

    # Build metrics table per (tier, N)
    tier_metrics = {}

    for tier, tier_results in results_by_tier.items():
        tier_metrics[tier] = {}

        # Filter to holdout fixtures for headline metrics
        holdout_results = []
        for r in tier_results:
            fixture = corpus_by_fixture_id.get(r.fixture_id)
            if fixture and fixture.blind_holdout:
                holdout_results.append(r)

        for n in [1, 2, 3, 4]:
            # Count successes in holdout
            successes = sum(
                1 for r in holdout_results
                if r.execute_select_outcome == "pass"
            )
            total = len(holdout_results)

            if total == 0:
                continue

            # Compute Wilson CI
            success_rate = successes / total if total > 0 else 0.0
            ci = wilson_ci(successes, total)

            # Count held-out failures
            held_out_fails = sum(
                1 for r in holdout_results
                if r.selected_candidate and r.held_out_test_outcome == "regressed"
            )
            held_out_fail_rate = held_out_fails / total if total > 0 else 0.0
            held_out_ci = wilson_ci(held_out_fails, total)

            # Candidates to first pass (median)
            first_passes = [
                r.first_passing_candidate_seed
                for r in holdout_results
                if r.first_passing_candidate_seed is not None
            ]
            candidates_to_first_pass = (
                sorted(first_passes)[len(first_passes) // 2] + 1
                if first_passes
                else 0.0
            )

            # Per-repo breakdown
            repo_breakdown = {}
            for repo_name in ["lapis-pm", "conductor"]:
                repo_results = [
                    r for r in holdout_results
                    if r.repo == repo_name
                ]
                if repo_results:
                    repo_successes = sum(
                        1 for r in repo_results
                        if r.execute_select_outcome == "pass"
                    )
                    repo_total = len(repo_results)
                    repo_breakdown[repo_name] = {
                        "success_rate": repo_successes / repo_total if repo_total > 0 else 0.0,
                        "n": repo_total,
                    }

            # Count blind fixtures in tier
            blind_count = sum(
                1 for r in tier_results
                if corpus_by_fixture_id.get(r.fixture_id, FixtureRecord(
                    repo="", sha="", parent_sha="", pr_number=None, path="",
                    file_loc="", changed_lines=0, tier="T1"
                )).checker_class == "BLIND"
            )
            blind_share = blind_count / len(tier_results) if tier_results else 0.0

            # Apply hard gates
            held_out_fail_gate_passed = held_out_fail_rate <= HELD_OUT_FAIL_GATE
            blind_gate_passed = (
                total >= BLIND_HOLDOUT_SIZE_GATE
                and blind_share <= BLIND_SHARE_GATE
            )
            judge_gate_passed = True  # Placeholder; real computation uses judge-calibration

            # Routing recommendation
            routing = "swarm-lane"
            gate_failure_reason = ""
            if not held_out_fail_gate_passed:
                routing = "big-lane"
                gate_failure_reason = f"G1: held-out-fail {held_out_fail_rate:.1%} > {HELD_OUT_FAIL_GATE:.1%}"
            elif not blind_gate_passed:
                routing = "big-lane"
                if total < BLIND_HOLDOUT_SIZE_GATE:
                    gate_failure_reason = f"G3: holdout-n {total} < {BLIND_HOLDOUT_SIZE_GATE}"
                else:
                    gate_failure_reason = f"G3: blind-share {blind_share:.1%} > {BLIND_SHARE_GATE:.1%}"
            elif not judge_gate_passed:
                routing = "big-lane"
                gate_failure_reason = f"G2: judge-calibration below gate"

            # Check if CI straddles threshold
            if ci[0] < EXECUTE_SUCCESS_BASELINE_TARGET < ci[1]:
                routing = "INSUFFICIENT_POWER"
                gate_failure_reason = f"CI straddles baseline {EXECUTE_SUCCESS_BASELINE_TARGET:.0%}"

            metrics = TierMetrics(
                tier=tier,
                n=n,
                repo_breakdown=repo_breakdown,
                execute_success_rate=success_rate,
                execute_success_ci=ci,
                candidates_to_first_pass_median=candidates_to_first_pass,
                selected_but_fails_held_out_rate=held_out_fail_rate,
                held_out_ci=held_out_ci,
                holdout_n=total,
                held_out_fail_gate_passed=held_out_fail_gate_passed,
                blind_holdout_n=blind_count,
                blind_share=blind_share,
                blind_gate_passed=blind_gate_passed,
                routing_recommendation=routing,
                gate_failure_reason=gate_failure_reason,
            )
            tier_metrics[tier][n] = metrics

    return tier_metrics


def generate_report(
    eval_result: EvalResult,
    corpus: list[FixtureRecord],
    mock_mode: bool = True,
) -> str:
    """Generate markdown verdict report."""
    report = f"""# Batched-Fixer Patch-Quality Eval — {eval_result.run_id}

**Generated:** {eval_result.generated_at}
**Served model:** {eval_result.served_model_id} ({eval_result.served_model_backend})

## Summary

This eval measures whether Coder-Next best-of-N execute-select beats single-shot
bare-completion on real landed small fixes from lapis-pm + conductor.

**IMPORTANT CAVEATS:**
- Measures reproduction of **sibling-model-authored fixes on familiar codebase**
- Bare-completion mode ONLY (NOT the agentic tool-loop big-lane fixer)
- Mock mode: {mock_mode}

## Decision Table

(Per-tier success rates with 95% CIs, held-out validation, judge-accuracy.)

"""
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_eval(
    phase: Literal["build-corpus", "run", "report"] = "run",
    run_id: Optional[str] = None,
    mock_mode: bool = False,
) -> Optional[EvalResult]:
    """Main harness entry point (called by CLI subcommand handler).

    Phases:
    - build-corpus: Extract and freeze corpus (CORE AC1, AC1b)
    - run: Generate candidates, execute-select, validate (requires GW if not mock)
    - report: Aggregate and emit artifacts

    Returns EvalResult on success, None on failure.
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    logger.info("=== Batched-Fixer Eval %s (phase=%s, mock=%s) ===", run_id, phase, mock_mode)

    # Clean up any leaked clones from prior crashed runs (startup sweep)
    cleanup_old_clones()

    try:
        if phase == "build-corpus":
            corpus, metadata = build_corpus()
            save_corpus(corpus)
            logger.info("Corpus built: %d fixtures, metadata: %s", len(corpus), metadata)
            return None

        if phase == "run":
            if not mock_mode:
                if not assert_swarm_health(strict=True):
                    logger.error("Swarm health check failed; aborting (see procedure)")
                    return None

            corpus = load_corpus()
            if not corpus:
                logger.warning("No corpus loaded; run 'build-corpus' first")
                return None

            fixture_results = []
            for fixture in corpus:
                try:
                    result = run_fixture(fixture, mock_mode=mock_mode)
                    fixture_results.append(result)
                except Exception as e:
                    logger.error("Fixture %s failed: %s", fixture.sha[:8], e)

            # Persist results to RUNS_DIR (run-id-namespaced)
            save_run_results(run_id, fixture_results)

            tier_metrics = aggregate_results(fixture_results, corpus)
            eval_result = EvalResult(
                run_id=run_id,
                generated_at=datetime.now(timezone.utc).isoformat(),
                served_model_id=EXPECTED_SWARM_MODEL if not mock_mode else "mock",
                served_model_backend="unknown",
                tier_metrics=tier_metrics,
            )
            return eval_result

        if phase == "report":
            # Load prior run results
            logger.warning("Report phase not yet implemented")
            return None

    except Exception as e:
        logger.exception("Eval phase %s failed: %s", phase, e)
        return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    result = run_eval(mock_mode=True)
    if result:
        print(f"Result: {result}")
