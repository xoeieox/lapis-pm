"""Synapse eval-gate v1: retrieval-quality check on PR branches.

Runs ``synapse eval replay`` against the PR branch, compares to a stored
baseline, and returns EvalResult.  The gate runs during tick decision-selection
for PRs in the synapse repo that touch retrieval-relevant paths.

Status semantics
----------------
| baseline_stale | regressed | status       | Brief emitted?                    |
|----------------|-----------|--------------|-----------------------------------|
| False          | False     | ``clean``    | No (percept tag only)             |
| False          | True      | ``regressed``| Yes - advisory with delta table   |
| True           | False     | ``unverified``| Yes - advisory with rebaseline nudge |
| True           | True      | ``regressed``| Yes - advisory + staleness flag   |

Concurrency assumption: v1 assumes serial tick execution.  No in-process
locks needed.  If a future tick-concurrency model lands, document the
assumption change here rather than adding locks.

Cache: eval results are keyed by ``pr.head.sha`` under RUNS_DIR.  A cache hit
short-circuits re-evaluation.  Cache is invalidated only by a new head SHA.
An ``actioned`` flag inside the cache JSON prevents duplicate briefs on
subsequent ticks that hit the same cached result.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path filter
# ---------------------------------------------------------------------------

RETRIEVAL_PATHS = ("synapse/service/", "synapse/eval/", "synapse/__init__.py")

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

EVAL_BASE = Path("/data/synapse/eval")
FIXTURE_PATH = EVAL_BASE / "fixtures/baseline.ndjson"
MANIFEST_PATH = EVAL_BASE / "fixtures/baseline.manifest.json"
BASELINE_PATH = EVAL_BASE / "baselines/main.json"
RUNS_DIR = EVAL_BASE / "runs"
CONFIG_PATH = EVAL_BASE / "config.yaml"

# Synapse working clone (for git operations: rev-parse origin/main, worktree)
SYNAPSE_REPO_PATH = Path("/srv/git/synapse-working")

# Smoke-test fixture bundled with this package (used by tests only).
SMOKE_FIXTURE_PATH = Path(__file__).parent / "smoke_fixtures/synapse_eval/baseline.ndjson"

# ---------------------------------------------------------------------------
# Port constants
# ---------------------------------------------------------------------------

_SYNAPSE_PRODUCTION_PORT = int(os.environ.get("SYNAPSE_PORT", "8401"))
EPHEMERAL_PORT_BASE = _SYNAPSE_PRODUCTION_PORT + 100   # 8501
EPHEMERAL_PORT_MAX = _SYNAPSE_PRODUCTION_PORT + 110    # 8511

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

PR_EVAL_TIMEOUT_S = 180       # wall-clock for full PR-branch eval
REGEN_TIMEOUT_S = 240         # wall-clock for baseline regeneration
READINESS_POLL_S = 0.25       # poll interval for /health
READINESS_CEILING_S = 30.0    # max wait for service readiness
TEARDOWN_WAIT_S = 5.0         # seconds before SIGKILL after SIGTERM

# Baseline regeneration rate-limit
REGEN_RATE_LIMIT_KEY = "pm/synapse-eval/regen-last-attempt"
REGEN_RATE_LIMIT_S = 1800    # 30 minutes between failed attempts

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, dict] = {
    "chub_jaccard_at_k_mean": {"direction": "higher_better", "regression": 0.05},
    "top1_stability":          {"direction": "higher_better", "regression": 0.05},
    "mem_jaccard_at_k_mean":   {"direction": "higher_better", "regression": 0.10},
    "latency_p50_ms":          {"direction": "lower_better",  "regression_pct": 25.0},
}


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalResult:
    pr_number: int
    head_sha: str
    baseline_sha: str
    baseline_stale: bool
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    deltas: dict[str, float]
    regressed: bool
    regressed_metrics: list[str]
    summary_text: str
    run_path: str
    status: Literal["clean", "regressed", "unverified"] = "clean"


# ---------------------------------------------------------------------------
# Path filter
# ---------------------------------------------------------------------------

def pr_touches_retrieval(pr_diff: str) -> bool:
    """True iff the PR diff touches any path under RETRIEVAL_PATHS."""
    for line in pr_diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        # "diff --git a/<path> b/<path>"
        parts = line.split(" ")
        if len(parts) < 4:
            continue
        a_side = parts[2]   # "a/<path>"
        if not a_side.startswith("a/"):
            continue
        file_path = a_side[2:]   # strip "a/"
        for rp in RETRIEVAL_PATHS:
            if rp.endswith("/"):
                if file_path.startswith(rp):
                    return True
            else:
                if file_path == rp:
                    return True
    return False


# ---------------------------------------------------------------------------
# Threshold loading + metric comparison
# ---------------------------------------------------------------------------

def _load_thresholds() -> dict[str, dict]:
    """Load threshold overrides from CONFIG_PATH; fall back to defaults on error."""
    if not CONFIG_PATH.exists():
        return DEFAULT_THRESHOLDS.copy()
    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(CONFIG_PATH.read_text())
        overrides = raw.get("thresholds") or {}
    except Exception as exc:
        logger.warning("eval_gate: failed to load %s (%s); using defaults", CONFIG_PATH, exc)
        return DEFAULT_THRESHOLDS.copy()

    merged = DEFAULT_THRESHOLDS.copy()
    if not isinstance(overrides, dict):
        logger.warning("eval_gate: thresholds in %s is not a mapping; using defaults", CONFIG_PATH)
        return merged
    for metric, cfg in overrides.items():
        if metric not in merged:
            logger.warning("eval_gate: unrecognized metric %r in config; ignoring", metric)
            continue
        merged[metric] = dict(cfg)
    return merged


def _metric_regressed(metric: str, current: float, baseline: float, cfg: dict) -> bool:
    """Return True iff current vs baseline crosses the regression threshold."""
    direction = cfg.get("direction", "higher_better")
    if direction == "higher_better":
        threshold = float(cfg.get("regression", 0.05))
        return (baseline - current) > threshold
    else:  # lower_better
        if "regression_pct" in cfg:
            if baseline == 0:
                return False
            pct_increase = (current - baseline) / abs(baseline) * 100.0
            return pct_increase > float(cfg["regression_pct"])
        else:
            threshold = float(cfg.get("regression", 25.0))
            return (current - baseline) > threshold


def _build_summary_text(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    deltas: dict[str, float],
    regressed_metrics: list[str],
    baseline_stale: bool,
    baseline_sha: str,
    current_main_sha: str = "",
    run_path: str = "",
) -> str:
    """Build human-readable delta table for brief body."""
    header = "Metric                     | Baseline | Current  | Delta    | Status\n"
    sep    = "---------------------------|----------|----------|----------|---------\n"
    rows = []
    for m in sorted(set(list(metrics.keys()) + list(baseline_metrics.keys()))):
        bv = baseline_metrics.get(m)
        cv = metrics.get(m)
        dv = deltas.get(m)
        bvs = f"{bv:.4f}" if bv is not None else "n/a"
        cvs = f"{cv:.4f}" if cv is not None else "n/a"
        dvs = f"{dv:+.4f}" if dv is not None else "n/a"
        st = "REGRESSED" if m in regressed_metrics else "ok"
        rows.append(f"{m:<27}| {bvs:<8} | {cvs:<8} | {dvs:<8} | {st}")
    table = header + sep + "\n".join(rows)
    parts = [table]
    if baseline_stale:
        note = (
            f"\nWARNING: baseline SHA {baseline_sha[:12]} is stale - "
            f"current origin/main is {current_main_sha[:12] if current_main_sha else '(unknown)'}. "
            "Rebaseline recommended: `lapis-pm eval-gate baseline regenerate`"
        )
        parts.append(note)
    if run_path:
        parts.append(f"\nRun record: {run_path}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------

def load_baseline() -> dict | None:
    """Load /data/synapse/eval/baselines/main.json; return None if missing/corrupt."""
    if not BASELINE_PATH.exists():
        return None
    try:
        return json.loads(BASELINE_PATH.read_text())
    except Exception as exc:
        logger.warning("eval_gate: failed to load baseline (%s)", exc)
        return None


def _current_synapse_main_sha(synapse_repo: Path = SYNAPSE_REPO_PATH) -> str:
    """Return git rev-parse origin/main for synapse, or '' on error."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            capture_output=True, text=True, timeout=15,
            cwd=str(synapse_repo),
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _check_baseline_stale(
    baseline: dict,
    synapse_repo: Path = SYNAPSE_REPO_PATH,
) -> tuple[bool, str]:
    """Return (is_stale, current_main_sha).

    Stale means baseline["sha"] != current origin/main HEAD.
    Falls back to (True, "") on any git error.
    """
    baseline_sha = baseline.get("sha", "")
    current_sha = _current_synapse_main_sha(synapse_repo)
    if not current_sha:
        # git unavailable or repo missing - treat as stale
        return True, ""
    return (baseline_sha != current_sha), current_sha


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _run_cache_path(head_sha: str) -> Path:
    return RUNS_DIR / f"{head_sha}.json"


def _load_cached_result(head_sha: str) -> EvalResult | None:
    """Return cached EvalResult if present; None otherwise."""
    p = _run_cache_path(head_sha)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return EvalResult(
            pr_number=data["pr_number"],
            head_sha=data["head_sha"],
            baseline_sha=data["baseline_sha"],
            baseline_stale=data["baseline_stale"],
            metrics=data["metrics"],
            baseline_metrics=data["baseline_metrics"],
            deltas=data["deltas"],
            regressed=data["regressed"],
            regressed_metrics=data["regressed_metrics"],
            summary_text=data["summary_text"],
            run_path=data["run_path"],
            status=data.get("status", "clean"),
        )
    except Exception as exc:
        logger.warning("eval_gate: failed to load cached result for %s (%s)", head_sha, exc)
        return None


def _save_result(result: EvalResult, actioned: bool = False) -> None:
    """Write EvalResult to run cache JSON."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(result)
    data["actioned"] = actioned
    _run_cache_path(result.head_sha).write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )


def is_eval_actioned(head_sha: str) -> bool:
    """True if the cached eval for this SHA has already been actioned (brief emitted or percept tag written)."""
    p = _run_cache_path(head_sha)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("actioned", False)
    except Exception:
        return False


def mark_eval_actioned(head_sha: str) -> None:
    """Set actioned=True in the run cache JSON for this SHA."""
    p = _run_cache_path(head_sha)
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
        data["actioned"] = True
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.warning("eval_gate: failed to mark_eval_actioned for %s (%s)", head_sha, exc)


# ---------------------------------------------------------------------------
# Ephemeral synapse process management
# ---------------------------------------------------------------------------

def _find_available_port(start: int, end: int) -> int | None:
    """Try ports from start to end (inclusive); return first available or None."""
    import socket
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    return None


def _poll_health(host: str, port: int, ceiling: float) -> str | None:
    """Poll GET http://<host>:<port>/health every READINESS_POLL_S seconds.

    Returns the bound host from the /health response once the service is ready,
    or None on timeout.  The host is read from the JSON response if possible
    (post synapse-port-drift-check-v0); otherwise uses the supplied host.
    """
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + ceiling
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    body = resp.read()
                    try:
                        info = json.loads(body)
                        return info.get("host", host)
                    except Exception:
                        return host
        except Exception:
            pass
        time.sleep(READINESS_POLL_S)
    return None


def _teardown_process(proc: subprocess.Popen) -> None:
    """Graceful SIGTERM → 5s wait → SIGKILL teardown."""
    try:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=TEARDOWN_WAIT_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=3)
    except Exception as exc:
        logger.warning("eval_gate: teardown error: %s", exc)


def _remove_worktree(synapse_repo: Path, worktree_path: str) -> None:
    """Remove the git worktree at worktree_path from synapse_repo."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", worktree_path],
            capture_output=True, cwd=str(synapse_repo), timeout=30,
        )
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True, cwd=str(synapse_repo), timeout=30,
        )
    except Exception as exc:
        logger.warning("eval_gate: worktree remove failed for %s: %s", worktree_path, exc)


# ---------------------------------------------------------------------------
# Core eval runner (shared by evaluate_pr and regenerate_baseline)
# ---------------------------------------------------------------------------

def _run_replay(
    worktree_path: str,
    fixture: Path,
    timeout_remaining: float,
) -> dict | None:
    """Spin ephemeral synapse in worktree, run replay, return parsed JSON or None.

    Tears down the process unconditionally.
    """
    port = _find_available_port(EPHEMERAL_PORT_BASE, EPHEMERAL_PORT_MAX)
    if port is None:
        logger.warning(
            "eval_gate: no available ephemeral port in range %d-%d",
            EPHEMERAL_PORT_BASE, EPHEMERAL_PORT_MAX,
        )
        return None

    env = {**os.environ, "SYNAPSE_PORT": str(port)}
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            ["python", "-m", "synapse.service"],
            cwd=worktree_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # so killpg works
        )

        # Readiness polling - bind host determined from /health JSON
        health_deadline = time.monotonic() + min(READINESS_CEILING_S, timeout_remaining - 5)
        # Try 127.0.0.1 first (test environments); production will bind Tailscale IP
        bound_host = _poll_health("127.0.0.1", port, max(health_deadline - time.monotonic(), 5))
        if bound_host is None:
            # Try the Tailscale IP
            ts_ip = os.environ.get("SYNAPSE_HOST", "203.0.113.10")
            bound_host = _poll_health(ts_ip, port, max(health_deadline - time.monotonic(), 5))
        if bound_host is None:
            logger.warning(
                "eval_gate: ephemeral synapse not ready after %.0fs on port %d",
                READINESS_CEILING_S, port,
            )
            return None

        endpoint = f"http://{bound_host}:{port}"
        replay_timeout = timeout_remaining - READINESS_CEILING_S - 5
        if replay_timeout <= 0:
            logger.warning("eval_gate: no time remaining for replay after startup")
            return None

        r = subprocess.run(
            ["synapse", "eval", "replay", str(fixture), "--endpoint", endpoint],
            capture_output=True, text=True,
            timeout=replay_timeout,
            cwd=worktree_path,
        )
        if r.returncode != 0:
            logger.warning("eval_gate: replay subprocess exited %d: %s", r.returncode, r.stderr[:500])
            return None

        return json.loads(r.stdout)

    except subprocess.TimeoutExpired:
        logger.warning("eval_gate: replay subprocess timed out")
        return None
    except json.JSONDecodeError as exc:
        logger.warning("eval_gate: failed to parse replay JSON: %s", exc)
        return None
    except Exception as exc:
        logger.warning("eval_gate: replay error: %s", exc)
        return None
    finally:
        if proc is not None:
            _teardown_process(proc)


# ---------------------------------------------------------------------------
# evaluate_pr - main entry point
# ---------------------------------------------------------------------------

def evaluate_pr(
    target_id: str,
    pr: dict,
    repo: str = "synapse",
) -> EvalResult | None:
    """Run synapse eval replay against the PR branch, compare to baseline.

    Returns EvalResult, or None if:
    - repo is not "synapse"
    - path filter not matched
    - fixture missing
    - worktree/eval failure (logs warning; gate degrades gracefully)
    Previously evaluated PRs (by head_sha) return the cached result immediately.
    """
    if repo != "synapse":
        return None

    head = pr.get("head") or {}
    head_sha = head.get("sha", "")
    pr_number = pr.get("number", 0)
    if not head_sha:
        return None

    # Cache hit
    cached = _load_cached_result(head_sha)
    if cached is not None:
        return cached

    # Fetch diff and apply path filter
    try:
        from agents_core.forgejo import get_pr_diff
        diff = get_pr_diff(repo, pr_number)
    except Exception as exc:
        logger.warning("eval_gate: failed to fetch diff for PR #%d: %s", pr_number, exc)
        return None

    if not pr_touches_retrieval(diff):
        return None

    # Fixture check
    if not FIXTURE_PATH.exists():
        logger.warning(
            "eval_gate: pinned fixture missing at %s - "
            "run 'synapse eval export' to create",
            FIXTURE_PATH,
        )
        return None

    # Baseline check
    baseline = load_baseline()
    if baseline is None:
        logger.warning("eval_gate: no baseline found at %s - regenerate first", BASELINE_PATH)
        # Return a synthetic "unverified" result so the tick can surface the gap
        run_p = str(_run_cache_path(head_sha))
        result = EvalResult(
            pr_number=pr_number,
            head_sha=head_sha,
            baseline_sha="",
            baseline_stale=True,
            metrics={},
            baseline_metrics={},
            deltas={},
            regressed=False,
            regressed_metrics=[],
            summary_text=(
                "No baseline found. Run `lapis-pm eval-gate baseline regenerate` "
                "to create the initial baseline before eval-gate can assess this PR."
            ),
            run_path=run_p,
            status="unverified",
        )
        _save_result(result)
        return result

    baseline_stale, current_main_sha = _check_baseline_stale(baseline)
    baseline_sha = baseline.get("sha", "")
    baseline_metrics: dict[str, float] = baseline.get("metrics", {})

    # Worktree creation
    short_sha = head_sha[:8]
    worktree_dir = f"/tmp/synapse-eval-pr{pr_number}-{short_sha}"
    synapse_repo = SYNAPSE_REPO_PATH

    deadline = time.monotonic() + PR_EVAL_TIMEOUT_S

    # Fetch so the head SHA from Forgejo exists in the local clone.
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            capture_output=True, text=True, timeout=30,
            cwd=str(synapse_repo),
        )
    except Exception as exc:
        logger.warning("eval_gate: git fetch before worktree failed for PR #%d: %s", pr_number, exc)

    try:
        r = subprocess.run(
            ["git", "worktree", "add", "--detach", worktree_dir, head_sha],
            capture_output=True, text=True, cwd=str(synapse_repo), timeout=60,
        )
        if r.returncode != 0:
            logger.warning(
                "eval_gate: worktree creation failed for %s: %s", head_sha, r.stderr[:300]
            )
            return None
    except Exception as exc:
        logger.warning("eval_gate: worktree creation error: %s", exc)
        return None

    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("eval_gate: timeout before replay for PR #%d", pr_number)
            return None

        replay_result = _run_replay(worktree_dir, FIXTURE_PATH, remaining)
        if replay_result is None:
            return None

        current_metrics: dict[str, float] = {
            k: float(v) for k, v in replay_result.items()
            if isinstance(v, (int, float))
        }

        # Compute deltas and regressions
        thresholds = _load_thresholds()
        deltas: dict[str, float] = {}
        regressed_metrics: list[str] = []
        for metric, cfg in thresholds.items():
            cv = current_metrics.get(metric)
            bv = baseline_metrics.get(metric)
            if cv is None or bv is None:
                continue
            delta = cv - bv
            deltas[metric] = delta
            if _metric_regressed(metric, cv, bv, cfg):
                regressed_metrics.append(metric)

        regressed = bool(regressed_metrics)

        # Status semantics (Council Amendment 1)
        if baseline_stale:
            status = "regressed" if regressed else "unverified"
        else:
            status = "regressed" if regressed else "clean"

        run_path = str(_run_cache_path(head_sha))
        summary_text = _build_summary_text(
            current_metrics, baseline_metrics, deltas, regressed_metrics,
            baseline_stale, baseline_sha, current_main_sha, run_path,
        )

        result = EvalResult(
            pr_number=pr_number,
            head_sha=head_sha,
            baseline_sha=baseline_sha,
            baseline_stale=baseline_stale,
            metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            deltas=deltas,
            regressed=regressed,
            regressed_metrics=regressed_metrics,
            summary_text=summary_text,
            run_path=run_path,
            status=status,
        )
        _save_result(result)
        return result

    finally:
        _remove_worktree(synapse_repo, worktree_dir)


# ---------------------------------------------------------------------------
# regenerate_baseline
# ---------------------------------------------------------------------------

def regenerate_baseline(
    synapse_repo_path: str = str(SYNAPSE_REPO_PATH),
) -> dict | None:
    """Worktree synapse at origin/main HEAD, run replay, write baselines/main.json.

    Returns the written baseline dict on success, None on failure.
    Tears down worktree unconditionally (try/finally).
    """
    synapse_repo = Path(synapse_repo_path)

    # Get current main HEAD sha
    main_sha = _current_synapse_main_sha(synapse_repo)
    if not main_sha:
        logger.warning("eval_gate: cannot determine origin/main HEAD for %s", synapse_repo)
        return None

    if not FIXTURE_PATH.exists():
        logger.warning(
            "eval_gate: pinned fixture missing at %s; cannot regenerate baseline", FIXTURE_PATH
        )
        return None

    short_sha = main_sha[:8]
    worktree_dir = f"/tmp/synapse-eval-baseline-{short_sha}"

    deadline = time.monotonic() + REGEN_TIMEOUT_S

    # Fetch so origin/main is current in the local clone.
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            capture_output=True, text=True, timeout=30,
            cwd=str(synapse_repo),
        )
    except Exception as exc:
        logger.warning("eval_gate: git fetch before baseline worktree failed: %s", exc)

    try:
        r = subprocess.run(
            ["git", "worktree", "add", "--detach", worktree_dir, main_sha],
            capture_output=True, text=True, cwd=str(synapse_repo), timeout=60,
        )
        if r.returncode != 0:
            logger.warning(
                "eval_gate: baseline worktree creation failed: %s", r.stderr[:300]
            )
            return None
    except Exception as exc:
        logger.warning("eval_gate: baseline worktree error: %s", exc)
        return None

    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("eval_gate: timeout before baseline replay")
            return None

        replay_result = _run_replay(worktree_dir, FIXTURE_PATH, remaining)
        if replay_result is None:
            logger.warning("eval_gate: baseline replay returned no result")
            return None

        metrics = {
            k: float(v) for k, v in replay_result.items()
            if isinstance(v, (int, float))
        }

        baseline = {
            "sha": main_sha,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": metrics,
            "fixture_path": str(FIXTURE_PATH),
        }

        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2))
        logger.info("eval_gate: baseline written for sha=%s", main_sha)
        return baseline

    finally:
        _remove_worktree(synapse_repo, worktree_dir)


# ---------------------------------------------------------------------------
# Baseline regeneration rate-limit helpers (used by pm_core)
# ---------------------------------------------------------------------------

def _mem():
    from agents_core.mem import MemoryStore
    return MemoryStore()


def should_regenerate_baseline(repo: str) -> bool:
    """Return True if a baseline regeneration should be triggered.

    Conditions (all must hold):
    - repo == "synapse"
    - Baseline is missing or stale (sha != current origin/main HEAD)
    - Not rate-limited (30 min since last failed attempt)
    """
    if repo != "synapse":
        return False

    baseline = load_baseline()
    if baseline is None:
        stale = True
    else:
        stale, _ = _check_baseline_stale(baseline)

    if not stale:
        return False

    # Rate-limit check
    try:
        rec = _mem().get(REGEN_RATE_LIMIT_KEY)
        if rec:
            data = json.loads(rec["content"])
            last_failed_at = data.get("failed_at")
            if last_failed_at:
                last_ts = datetime.fromisoformat(last_failed_at).timestamp()
                if (time.time() - last_ts) < REGEN_RATE_LIMIT_S:
                    return False
    except Exception:
        pass

    return True


def record_regen_attempt(success: bool) -> None:
    """Persist regen attempt result for rate-limiting and observability."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data: dict = {"attempted_at": now}
    if not success:
        data["failed_at"] = now
    try:
        _mem().set(
            REGEN_RATE_LIMIT_KEY,
            json.dumps(data, ensure_ascii=False),
            tags=["lapis-pm", "synapse-eval"],
        )
    except Exception as exc:
        logger.warning("eval_gate: failed to record regen attempt: %s", exc)


# ---------------------------------------------------------------------------
# CLI helpers (used by cmd_eval_gate in cli.py)
# ---------------------------------------------------------------------------

def eval_gate_status(open_prs: list[dict], repo: str = "synapse") -> list[dict]:
    """Return status summary for each open PR (used by CLI `eval-gate status`)."""
    out = []
    for pr in open_prs:
        head = pr.get("head") or {}
        sha = head.get("sha", "")
        pr_number = pr.get("number", 0)
        cached = _load_cached_result(sha) if sha else None
        out.append({
            "pr_number": pr_number,
            "head_sha": sha[:8] if sha else "",
            "status": cached.status if cached else "not_evaluated",
            "regressed_metrics": cached.regressed_metrics if cached else [],
            "actioned": is_eval_actioned(sha) if sha else False,
            "run_path": cached.run_path if cached else "",
        })
    return out
