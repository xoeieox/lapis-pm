"""Pipeline-attested machine verification (U1, lapis-pm-autonomy-actuator-v0).

The daemon runs the repo's suite itself on the PR head and on origin/main
(the same two-worktree failure-set-diff recipe the `pm-pr-review` skill
prescribes) and grants `pm_verification=machine` from the *observed run*,
not from spec prose. This dissolves clause 6 of the conservative
auto-resolve predicate for clean PRs without weakening the containment
invariant: the grant source is the pipeline's observation, which
interpolated bundle content cannot fake (the containment pin in
`bundle_autodispatch._bind` STAYS — it keeps bind-time state fail-safe;
this module flips verification later, from observation, at PR time).

Runner-control rule (Many Eyes B1): if the PR's `changed_paths` intersect
the test-runner control set, the result is `inconclusive` — a PR may not
rewrite the instrument that measures it.

Worktrees are ephemeral (tempfile.mkdtemp prefix="lapis-attest-", 0700)
and removed `--force` on ALL exit paths (M3: /tmp is near-full on the BRIX
root disk). One run per head SHA: the result caches in
`target.data["pm_verification_attestation"]` and is re-used while the head
SHA is unchanged (a force-push re-runs it).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Wall budget per head (Design 2). Exceeded = inconclusive = fail-safe.
ATTESTATION_WALL_BUDGET_S = 900

# Test-runner control set (Design 2 / B1): a PR touching any of these
# attests `inconclusive` — it may not rewrite the instrument that measures
# it. Root-level filenames plus conftest.py / tests/*.cfg|ini at any depth.
_RUNNER_CONTROL_ROOT_NAMES = {
    "smoke.sh", "Makefile", "pyproject.toml", "setup.py",
    "setup.cfg", "pytest.ini", "tox.ini",
}
_TESTS_CFG_INI_RE = re.compile(r"^tests/.*\.(cfg|ini)$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_clone_path(repo: str) -> str:
    """Resolve the local working clone path for a repo name.

    Mirrors the Shaper convention (`/srv/git/<name>-working`); the attestation
    runs git worktree commands from this clone.
    """
    return f"/srv/git/{repo}-working"


def _head_branch_for(target_id: str, slug: str = "forced") -> str:
    """The PR head branch. Fixer dispatches use `lapis/<tid>/forced`; a
    dispatch record's slug (when present) wins."""
    return f"lapis/{target_id}/{slug or 'forced'}"


def _base_ref(base_branch: str) -> str:
    return f"origin/{base_branch}"


def _pr_failures_from_output(output: str) -> list[str]:
    """Extract failing test ids from pytest -q output (or smoke.sh output).

    pytest -q emits `FAILED path::test_name` lines. Best-effort: any line
    starting with FAILED is a failure; the set is what matters (the diff is
    on sets, not ordering)."""
    out = set()
    for line in (output or "").splitlines():
        line = line.strip()
        if line.startswith("FAILED "):
            # "FAILED tests/test_x.py::test_y - AssertionError"
            ident = line[len("FAILED "):].split(" - ")[0].strip()
            if ident:
                out.add(ident)
    return sorted(out)


def _run_suite_in_worktree(worktree: Path) -> tuple[str, list[str]]:
    """Run the repo's suite in `worktree`. Returns (suite, failures).

    Suite selection (Design 2): `bash smoke.sh` if the file exists in the
    repo root, else `python3 -m pytest -q` — exactly the `pm-pr-review`
    skill's repo-entry-point rule.

    Raises on any runner failure (timeout, missing binary, non-zero rc is
    NOT an error — red suites are data, not failures).
    """
    if (worktree / "smoke.sh").exists():
        suite = "bash smoke.sh"
        cmd = ["bash", "smoke.sh"]
    else:
        suite = "python3 -m pytest -q"
        cmd = ["python3", "-m", "pytest", "-q"]
    proc = subprocess.run(
        cmd,
        cwd=str(worktree),
        capture_output=True,
        text=True,
        timeout=ATTESTATION_WALL_BUDGET_S,
    )
    # A red suite is DATA (its failures feed the diff), not an error. Only
    # timeouts / missing binaries raise (caught upstream -> inconclusive).
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return suite, _pr_failures_from_output(output)


def _create_worktree(clone: str, ref: str, dest: Path) -> None:
    subprocess.run(
        ["git", "-C", clone, "worktree", "add", "--detach", str(dest), ref],
        capture_output=True, text=True, timeout=120, check=True,
    )


def _remove_worktree(clone: str, dest: Path) -> None:
    """Remove `--force` on all exit paths (Design 2 / M3)."""
    try:
        subprocess.run(
            ["git", "-C", clone, "worktree", "remove", "--force", str(dest)],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        # Last-resort: the worktree dir may be gone; prune any stale admin entry.
        try:
            shutil.rmtree(str(dest), ignore_errors=True)
            subprocess.run(
                ["git", "-C", clone, "worktree", "prune"],
                capture_output=True, text=True, timeout=60,
            )
        except Exception:
            pass


def _runner_control_hit(changed_paths: list[str]) -> bool:
    """True when the PR touches the test-runner control set (B1)."""
    for p in changed_paths or []:
        base = p.rsplit("/", 1)[-1]
        if base in _RUNNER_CONTROL_ROOT_NAMES:
            return True
        if base == "conftest.py":
            return True
        if _TESTS_CFG_INI_RE.match(p):
            return True
    return False


def attest(
    repo: str,
    pr_number: int,
    head_sha: str,
    base_branch: str = "main",
    *,
    changed_paths: list[str] | None = None,
    target_id: str | None = None,
    slug: str = "forced",
) -> dict:
    """Run the two-worktree failure-set diff and return the attestation.

    Returns:
        {result: "attested" | "unattested" | "inconclusive",
         pr_failures: [str], preexisting: [str],
         suite: str, ts: iso}

    `attested`    — PR head introduces no new failures vs origin/main.
    `unattested`  — PR head introduces >=1 failure not on main.
    `inconclusive`— runner-control hit, worktree failure, suite timeout,
                    missing clone, or missing head SHA. Fail-safe: the
                    caller never grants machine verification on this.
    """
    ts = _now_iso()
    if not head_sha:
        return {"result": "inconclusive", "pr_failures": [], "preexisting": [],
                "suite": "", "ts": ts, "reason": "missing_head_sha"}
    if _runner_control_hit(changed_paths or []):
        return {"result": "inconclusive", "pr_failures": [], "preexisting": [],
                "suite": "", "ts": ts, "reason": "runner_control_intersection"}

    clone = _repo_clone_path(repo)
    if not Path(clone).is_dir():
        return {"result": "inconclusive", "pr_failures": [], "preexisting": [],
                "suite": "", "ts": ts, "reason": f"clone_missing:{clone}"}

    base_ref = _base_ref(base_branch)
    head_ref = _head_branch_for(target_id or f"pr{pr_number}", slug)

    root = Path(tempfile.mkdtemp(prefix="lapis-attest-"))
    try:
        os.chmod(str(root), 0o700)
        pr_wt = root / "pr"
        main_wt = root / "main"
        try:
            _create_worktree(clone, head_ref, pr_wt)
            _create_worktree(clone, base_ref, main_wt)
            suite, pr_failures = _run_suite_in_worktree(pr_wt)
            _main_suite, main_failures = _run_suite_in_worktree(main_wt)
        except subprocess.TimeoutExpired:
            return {"result": "inconclusive", "pr_failures": [], "preexisting": [],
                    "suite": "", "ts": ts, "reason": "wall_budget_exceeded"}
        except subprocess.CalledProcessError as e:
            return {"result": "inconclusive", "pr_failures": [], "preexisting": [],
                    "suite": "", "ts": ts,
                    "reason": f"worktree_failure:{type(e).__name__}"}
        except Exception as e:
            return {"result": "inconclusive", "pr_failures": [], "preexisting": [],
                    "suite": "", "ts": ts,
                    "reason": f"runner_error:{type(e).__name__}"}

        pr_set = set(pr_failures)
        main_set = set(main_failures)
        new_failures = sorted(pr_set - main_set)
        preexisting = sorted(pr_set & main_set)
        result = "unattested" if new_failures else "attested"
        return {
            "result": result,
            "pr_failures": pr_failures,
            "preexisting": preexisting,
            "suite": suite,
            "ts": ts,
        }
    finally:
        # ALL exit paths (success, timeout, worktree failure, exception).
        _remove_worktree(clone, pr_wt)
        _remove_worktree(clone, main_wt)
        shutil.rmtree(str(root), ignore_errors=True)


def cache_key(target_id: str, pr_number: int) -> str:
    """mem key for the attestation result (one run per head SHA)."""
    return f"pm/verification-attestation/{target_id}/{pr_number}"


def read_cache(mem, target_id: str, pr_number: int, head_sha: str) -> dict | None:
    """Return the cached attestation for the CURRENT head SHA, else None.

    A force-push (new head SHA) invalidates the cache -> re-run."""
    try:
        rec = mem.get(cache_key(target_id, pr_number))
        if not rec:
            return None
        data = json.loads(rec.get("content", ""))
        if not isinstance(data, dict):
            return None
        if data.get("head_sha") != head_sha:
            return None
        return data
    except Exception:
        return None


def write_cache(mem, target_id: str, pr_number: int, head_sha: str,
                result: dict, *, shadow: bool = False) -> str:
    """Persist the attestation result. `shadow=True` tags it shadow (no grant)."""
    payload = {
        "head_sha": head_sha,
        "result": result.get("result"),
        "pr_failures": result.get("pr_failures", []),
        "preexisting": result.get("preexisting", []),
        "suite": result.get("suite", ""),
        "ts": result.get("ts", _now_iso()),
        "shadow": shadow,
    }
    if result.get("reason"):
        payload["reason"] = result["reason"]
    tags = ["lapis-pm", "pm:verification-attestation"]
    if shadow:
        tags.append("shadow")
    key = cache_key(target_id, pr_number)
    mem.set(key, json.dumps(payload, ensure_ascii=False), tags=tags)
    return key


def record_preexisting(mem, repo: str, preexisting: list[str], now: str | None = None) -> None:
    """Record pre-existing failures as `finding/` keys (once per
    (repo, failing-test) per day, deduped) per the skill's recipe."""
    now = now or _now_iso()
    day = now[:10]
    for test in preexisting:
        key = f"finding/preexisting-failure/{repo}/{test}/{day}"
        try:
            if mem.get(key):
                continue
            mem.set(key, json.dumps({"repo": repo, "test": test, "ts": now}),
                    tags=["lapis-pm", "finding", "preexisting-failure"])
        except Exception:
            pass
