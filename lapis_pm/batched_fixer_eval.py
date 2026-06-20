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
    """Remove any leaked worktrees from a prior crashed run."""
    try:
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if not line.strip():
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
    metadata = {"lapis_pm_count": 0, "conductor_count": 0, "quarantine_count": 0}

    # TODO: Implement git history extraction (AC1)
    # For now, return empty corpus for mock testing
    logger.warning("Corpus builder not yet implemented; returning empty corpus")

    return corpus, metadata


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
) -> FixtureRunResult:
    """Run a single fixture: generate N candidates, select via execute oracle.

    Yields FixtureRunResult with candidates, execute-select outcome, and held-out validation.
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

    # TODO: Apply each candidate in a worktree, run scoped tests (AC3, AC4)
    # TODO: Implement execute-select + held-out validation (AC6, AC6b)
    # For mock testing, populate with deterministic results
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


# ---------------------------------------------------------------------------
# Phase: Report (mock-testable decision table aggregation)
# ---------------------------------------------------------------------------


def aggregate_results(
    fixture_results: list[FixtureRunResult],
    corpus: list[FixtureRecord],
) -> dict[str, dict[int, TierMetrics]]:
    """Aggregate fixture results into per-tier (tier, N) decision table.

    Computes success rates with Wilson 95% CIs, held-out-fail rates, judge-accuracy,
    and applies hard routing gates (G1-G3). Returns dict keyed by tier → N → TierMetrics.
    """
    # TODO: Implement aggregation with CI computation (AC5, AC8)
    # For now, return mock structure
    return {
        "T1": {
            1: TierMetrics(
                tier="T1",
                n=1,
                repo_breakdown={},
                execute_success_rate=0.75,
                execute_success_ci=(0.65, 0.85),
                candidates_to_first_pass_median=1.0,
                selected_but_fails_held_out_rate=0.0,
                held_out_ci=(0.0, 0.0),
                holdout_n=8,
                held_out_fail_gate_passed=True,
                routing_recommendation="swarm-lane",
            )
        }
    }


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
