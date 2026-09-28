"""bundle_triage — per-item mechanical/fork classification for debt-bundle
items, run BEFORE the full spec-review gate (spec bundle-item-level-triage-v0),
plus the Verification Contract main-baseline machinery (spec
verification-contract-main-baseline-v0).

Why: the nightly debt-bundle gate has bound zero items across 15 consecutive
runs — every bundle item, however mechanical, rides the same philosophical
council as a genuine design fork, and the council escalates everything. This
module splits each bundle item into:

  - "mechanical" — code-grounded, concrete fix, small blast radius, and an
    EXTRACTED (never formulated) Verification Contract. Cooked into its own
    fix spec and bound directly, advisory, no council.
  - "fork" — anything that fails a tier. Stays in the bundle and reaches the
    full gate unchanged, so genuine invariant/ethics/UX/vision-fork items
    still route to Erah exactly as before.

Three-tier check, binary, fail-closed to "fork" on any tier miss:
  1. Static check       — code-grounded fix, no denylisted path touched.
  2. Blast-radius check — small named-file count, no denylisted path.
  3. Behavioral-invariance check — a POST-COOK BIND-GATE VERIFIER, NOT a
     classification-time tier. `classify_item` / `is_mechanical` cover tiers
     1-2 only. Tier 3 is `verify_behavioral_invariance`, called by
     `bundle_autodispatch._reconcile_one` AFTER `cook_item_to_spec` and
     BEFORE the mechanical bind fires. A tier-3 miss re-routes the item to
     fork with reason "invariance" — never a silent bind.

The Verification Contract is EXTRACTION-ONLY: `extract_verification_contract`
locates existing proof (a named test that already pins the touched surface's
behavior) — it never generates a new assertion or interprets meaning to
synthesize one. No extractable contract means the item can never classify
mechanical, however simple the fix looks (DoD-1, fail-closed).

The contract's "passes unchanged" clause is now a VERIFIED FACT, not an
inference (spec verification-contract-main-baseline-v0): `extract_verification_contract`
runs `baseline_suite_for_item` — fetches origin, resolves origin/main,
creates-or-reuses a detached worktree AT that main sha, runs the named test
file there, and records the result (main sha, red node ids, counts, duration)
in `item["verification_baseline"]`. The rendered contract string is
sha-anchored and states what was actually measured; an unrunnable file
(fetch failure, absence at main sha, collection error, timeout, run-budget
exhaustion) yields no contract at all (state `unverified`) — fail-closed,
same as the "no locatable test" case.

The tier-1/2 interface-and-invariant denylist SOURCES from the existing
`lapis_pm.authority.HELD_PATTERNS` rather than inventing a parallel list.

`cook_item_to_spec` is IN-REPO: it renders a spec from a plain dict via an
f-string template — no import of, or call into, conductor or any other repo.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lapis_pm.authority import is_held_path

# ---------------------------------------------------------------------------
# Tier thresholds / reject lists
# ---------------------------------------------------------------------------

# Tier 2: "a small named-file count" (v0 stand-in for the Council's fuller
# dependency-topology read — see spec Design §3).
MAX_BLAST_RADIUS_FILES = 3

# Case-insensitive exact-match placeholder tokens for fix_description — a
# non-responsive value is not a concrete fix, mirroring the C7 Consumer-line
# reject-token convention in spec_review.py.
_FIX_DESCRIPTION_REJECT_TOKENS: frozenset[str] = frozenset({
    "tbd", "n/a", "na", "none", "unknown", "todo", "",
})

_FORK_REASONS = ("static", "blast-radius", "invariance")


# ---------------------------------------------------------------------------
# Tier 1 + 2: the classification-time predicate
# ---------------------------------------------------------------------------

def _looks_placeholder(fix_description: str) -> bool:
    normalized = re.sub(r"\s+", " ", fix_description.strip().lower())
    return normalized in _FIX_DESCRIPTION_REJECT_TOKENS


def _static_check(fix_description: str, touched_surfaces: list[str]) -> bool:
    """Tier 1: the fix is code-grounded (a concrete, non-placeholder fix
    description naming concrete touched files) and touches no denylisted
    path."""
    if not fix_description or not fix_description.strip():
        return False
    if _looks_placeholder(fix_description):
        return False
    if not touched_surfaces:
        return False
    if any(is_held_path(p) for p in touched_surfaces):
        return False
    return True


def _blast_radius_check(touched_surfaces: list[str]) -> bool:
    """Tier 2: a small named-file count, no denylisted path. Denylist is
    re-checked defensively — this function must stand alone as a correct
    tier-2 predicate even if a future caller runs it without tier 1."""
    if not touched_surfaces:
        return False
    if len(touched_surfaces) > MAX_BLAST_RADIUS_FILES:
        return False
    if any(is_held_path(p) for p in touched_surfaces):
        return False
    return True


def is_mechanical(fix_description: str, touched_surfaces: list[str]) -> bool:
    """The mechanical predicate, tiers 1+2 only — separately callable so a
    second call site (seam 4, `adjudication-boundary-four-trigger-alignment-v0`
    DoD-3) can share this exact definition rather than forking it.

    Pure, deterministic, no I/O. Tier 3 (behavioral-invariance) is
    deliberately NOT part of this predicate — see module docstring and
    `verify_behavioral_invariance`.
    """
    return (
        _static_check(fix_description, touched_surfaces)
        and _blast_radius_check(touched_surfaces)
    )


def classify_item(item: dict) -> tuple[str, str | None]:
    """Thin wrapper over `is_mechanical` — tiers 1+2 only.

    Returns (class_, fork_reason):
      - ("mechanical", None) — both tiers pass.
      - ("fork", "static")       — tier 1 (code-grounded / denylist) failed.
      - ("fork", "blast-radius") — tier 1 passed, tier 2 (file count) failed.

    Never returns reason "invariance" — that categorized reason belongs only
    to the post-cook tier-3 verifier in bundle_autodispatch._reconcile_one,
    which re-routes an already-"mechanical" item to fork when no
    Verification Contract can be extracted.
    """
    fix_description = item.get("fix_description") or ""
    touched_surfaces = item.get("touched_surfaces") or []

    if not _static_check(fix_description, touched_surfaces):
        return "fork", "static"
    if not _blast_radius_check(touched_surfaces):
        return "fork", "blast-radius"
    return "mechanical", None


# ---------------------------------------------------------------------------
# Tier 3: the post-cook bind-gate verifier (Verification Contract) + the
# main-baseline machinery (verification-contract-main-baseline-v0)
# ---------------------------------------------------------------------------

# Per-file baseline subprocess timebox — matches batched_fixer_eval.PR_EVAL_TIMEOUT_S.
BASELINE_TIMEOUT_S = 180
# Per-reconcile-run deadline: exhausted -> unverified/budget_exhausted -> fork.
BASELINE_RUN_BUDGET_S = 3600

# Detached-worktree scratch prefix. Strictly scoped — the prune pass below
# never touches anything outside this prefix (batched_fixer_eval's "NEVER
# calls git worktree prune on any shared clone" invariant is honored: we
# only ever remove our own registrations here).
_BASELINE_WORKTREE_ROOT = Path("/tmp/lapis-pm-baselines")
_WORKTREE_DIR_RE = re.compile(r"^(?P<repo>.+)-(?P<sha7>[0-9a-f]{7})$")


def _repo_root(repo: str) -> Path:
    """Convention path for a repo's local working clone (AGENTS.md 'Repos:'
    section: /srv/git/<repo>-working). A thin, monkeypatchable indirection —
    tests override this rather than touching the real filesystem."""
    return Path(f"/srv/git/{repo}-working")


def _locate_existing_test(root: Path, surface: str) -> str | None:
    """Static predicate over the tree: does a conventionally-named existing
    test already cover *surface*? Checked locations, in order:
      - <root>/tests/test_<stem>.py           (repo-root tests/ convention)
      - <surface's dir>/test_<stem>.py         (co-located test)
      - <surface's dir>/tests/test_<stem>.py   (package-local tests/)
    Returns the path (relative to root when possible) of the first hit, or
    None. Never writes, never runs anything — presence on disk only."""
    stem = Path(surface).stem
    surface_dir = Path(surface).parent
    candidates = [
        root / "tests" / f"test_{stem}.py",
        root / surface_dir / f"test_{stem}.py",
        root / surface_dir / "tests" / f"test_{stem}.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.relative_to(root))
            except ValueError:
                return str(candidate)
    return None


@dataclass
class Baseline:
    """The recorded, verified fact behind a Verification Contract string.
    Rides in `item["verification_baseline"]` (caller-stored, additive,
    symmetric with `item["verification_contract"]`)."""

    main_sha: str | None
    test_rel: str | None
    state: str  # "green" | "annotated" | "unverified"
    red: list[str]
    n_tests: int
    duration_s: float
    ts: str
    reason: str | None = None   # categorized reason when state == "unverified"
    surface: str | None = None  # which touched_surface matched test_rel (rendering aid)

    def to_dict(self) -> dict:
        return {
            "main_sha": self.main_sha,
            "test_rel": self.test_rel,
            "state": self.state,
            "red": list(self.red),
            "n_tests": self.n_tests,
            "duration_s": self.duration_s,
            "ts": self.ts,
            "reason": self.reason,
            "surface": self.surface,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- runner seams — each independently monkeypatchable, mirroring _repo_root ---

def _resolve_main_sha(repo_root: Path) -> str | None:
    """`git fetch origin` then `git rev-parse origin/main` on the -working
    clone. Returns the resolved sha, or None on any fetch/rev-parse failure
    (fail-closed — reason `fetch_failed`). The sole sha-resolution seam;
    reused standalone by tier-3's fast path and by the run-start prune pass,
    so all three share identical fetch semantics."""
    try:
        fetch = subprocess.run(
            ["git", "fetch", "origin"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=60,
        )
        if fetch.returncode != 0:
            return None
        rev_parse = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=15,
        )
        if rev_parse.returncode != 0:
            return None
        sha = rev_parse.stdout.strip()
        return sha or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_worktree_add(repo_root: Path, wt_path: Path, sha: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_path), sha],
            cwd=str(repo_root), capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _ensure_baseline_worktree(repo: str, repo_root: Path, main_sha: str) -> Path | None:
    """Create-or-reuse a detached worktree at main_sha under
    `_BASELINE_WORKTREE_ROOT/<repo>-<sha7>`. The `.exists()` check below IS
    the reuse path (worktree creation is the slow part; the suite run is
    seconds) — a second call at the same sha never re-invokes
    `git worktree add`."""
    sha7 = main_sha[:7]
    wt_path = _BASELINE_WORKTREE_ROOT / f"{repo}-{sha7}"
    if wt_path.exists():
        return wt_path
    try:
        _BASELINE_WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return wt_path if _git_worktree_add(repo_root, wt_path, main_sha) else None


def _run_baseline_pytest(worktree: Path, test_rel: str, timeout_s: int) -> tuple[str, str, float]:
    """Run the named test file's suite in the worktree. Returns
    (outcome, output, duration_s); outcome is "ran" | "timeout" |
    "collection-error" | "error". Invocation shape is pinned — the
    parser+invocation parity test locks this exact command:
    `python3 -m pytest <rel> -q --no-header -p no:cacheprovider`, cwd=worktree,
    env PYTHONUSERBASE=/home/user/.local (worktree-local .pyuserbase
    friction — see lapis-pm/conftest.py:1-9)."""
    cmd = ["python3", "-m", "pytest", test_rel, "-q", "--no-header", "-p", "no:cacheprovider"]
    env = dict(os.environ)
    env["PYTHONUSERBASE"] = "/home/user/.local"
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, cwd=str(worktree), capture_output=True, text=True,
            timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return "timeout", "", time.monotonic() - start
    except (OSError, subprocess.SubprocessError) as exc:
        return "error", str(exc), time.monotonic() - start
    duration_s = time.monotonic() - start
    output = result.stdout + result.stderr
    from lapis_pm.batched_fixer_eval import parse_pytest_failures
    if result.returncode in (2, 4) and not parse_pytest_failures(output):
        # 2 = interrupted (collection error), 4 = no collectors found — both
        # mean the file could not be run at all, mirroring
        # batched_fixer_eval.run_scoped_tests_once's own heuristic.
        return "collection-error", output, duration_s
    return "ran", output, duration_s


_PYTEST_SUMMARY_COUNT_RE = re.compile(
    r"(\d+)\s+(?:passed|failed|error|errors|skipped|xfailed|xpassed)"
)


def _count_pytest_total(output: str) -> int:
    return sum(int(n) for n in _PYTEST_SUMMARY_COUNT_RE.findall(output))


def baseline_suite_for_item(item: dict, deadline_monotonic: float | None = None) -> Baseline:
    """Turn the Verification Contract's "passes unchanged" claim into a
    recorded fact (spec Design steps 1-5): resolve origin/main on the
    -working clone, RE-LOCATE the named test against a detached worktree AT
    that main sha (never the drifted -working tree), create-or-reuse the
    worktree, run the named file, and parse its outcome with the existing
    `batched_fixer_eval.parse_pytest_failures`.

    `deadline_monotonic` is the caller's optional per-run budget
    (time.monotonic() deadline, BASELINE_RUN_BUDGET_S from run start).
    Exceeded -> state=unverified reason=budget_exhausted, no I/O attempted.

    Never raises — every failure mode (missing repo/surfaces, no locatable
    test anywhere, fetch failure, worktree failure, absence at main sha,
    collection error, timeout, budget exhaustion) returns state="unverified"
    with a categorized `reason`, fail-closed (DoD-2)."""
    ts = _now_iso()

    if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
        return Baseline(None, None, "unverified", [], 0, 0.0, ts, reason="budget_exhausted")

    repo = item.get("repo")
    touched_surfaces = item.get("touched_surfaces") or []
    if not repo or not touched_surfaces:
        return Baseline(None, None, "unverified", [], 0, 0.0, ts, reason="no_repo_or_surfaces")

    root = _repo_root(repo)
    test_rel = None
    surface = None
    for s in touched_surfaces:
        located = _locate_existing_test(root, s)
        if located is not None:
            test_rel, surface = located, s
            break
    if test_rel is None:
        return Baseline(None, None, "unverified", [], 0, 0.0, ts, reason="not_located")

    main_sha = _resolve_main_sha(root)
    if main_sha is None:
        return Baseline(
            None, test_rel, "unverified", [], 0, 0.0, ts,
            reason="fetch_failed", surface=surface,
        )

    worktree = _ensure_baseline_worktree(repo, root, main_sha)
    if worktree is None:
        return Baseline(
            main_sha, test_rel, "unverified", [], 0, 0.0, ts,
            reason="worktree_failed", surface=surface,
        )

    # Re-locate against the MAIN worktree root, not the -working tree — the
    # proof must exist at main sha; a test present in a drifted -working
    # tree but absent on main is not main-state proof.
    if not (worktree / test_rel).exists():
        return Baseline(
            main_sha, test_rel, "unverified", [], 0, 0.0, ts,
            reason="not_on_main", surface=surface,
        )

    outcome, output, duration_s = _run_baseline_pytest(worktree, test_rel, BASELINE_TIMEOUT_S)
    if outcome == "timeout":
        return Baseline(
            main_sha, test_rel, "unverified", [], 0, duration_s, ts,
            reason="timeout", surface=surface,
        )
    if outcome != "ran":
        return Baseline(
            main_sha, test_rel, "unverified", [], 0, duration_s, ts,
            reason="collection_error", surface=surface,
        )

    from lapis_pm.batched_fixer_eval import parse_pytest_failures
    red = parse_pytest_failures(output)
    n_tests = _count_pytest_total(output)
    state = "annotated" if red else "green"
    return Baseline(main_sha, test_rel, state, red, n_tests, duration_s, ts, surface=surface)


def _remove_baseline_worktree(repo: str, wt_path: Path) -> None:
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt_path)],
            cwd=str(_repo_root(repo)), capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    if wt_path.exists():
        shutil.rmtree(wt_path, ignore_errors=True)


def prune_stale_baseline_worktrees() -> list[str]:
    """Run ONCE at the START of each reconcile run, before any baseline
    (gate hardening fold: the `/` partition was 87% full at spec time, and
    pruning only at worktree-creation time leaves stale entries from dead
    runs to accumulate — a disk-exhaustion cascade into `unverified` for
    every item in the run). Removes registrations under
    `_BASELINE_WORKTREE_ROOT` whose sha no longer matches that repo's
    current origin/main. Strictly scoped to this prefix — never runs
    `git worktree prune` against a shared clone.

    Returns the list of removed worktree directory names (for logging).
    Never raises."""
    removed: list[str] = []
    try:
        if not _BASELINE_WORKTREE_ROOT.exists():
            return removed
        entries = sorted(p for p in _BASELINE_WORKTREE_ROOT.iterdir() if p.is_dir())
    except OSError:
        return removed

    current_sha7_by_repo: dict[str, str | None] = {}
    for entry in entries:
        m = _WORKTREE_DIR_RE.match(entry.name)
        if not m:
            continue
        repo, sha7 = m.group("repo"), m.group("sha7")
        if repo not in current_sha7_by_repo:
            resolved = _resolve_main_sha(_repo_root(repo))
            current_sha7_by_repo[repo] = resolved[:7] if resolved else None
        current_sha7 = current_sha7_by_repo[repo]
        if current_sha7 is not None and sha7 == current_sha7:
            continue  # current — keep
        _remove_baseline_worktree(repo, entry)
        removed.append(entry.name)
    return removed


def extract_verification_contract(item: dict, deadline_monotonic: float | None = None) -> str | None:
    """EXTRACTION-ONLY Verification Contract: run `baseline_suite_for_item`
    (which itself locates existing proof for one of the item's touched
    surfaces — never formulate or generate a new assertion) and render the
    verified fact it records. Returns None when no touched surface has a
    locatable test, or the baseline itself could not be verified — fail-
    closed (DoD-1): no extractable contract, no matter how simple the fix
    looks. Always records `item["verification_baseline"]` once repo/surfaces
    are present, even on a None return, so a fork/invariance re-route still
    carries its categorized reason for Erah's audit.
    """
    repo = item.get("repo")
    touched_surfaces = item.get("touched_surfaces") or []
    if not repo or not touched_surfaces:
        return None

    baseline = baseline_suite_for_item(item, deadline_monotonic=deadline_monotonic)
    item["verification_baseline"] = baseline.to_dict()

    if baseline.state == "unverified":
        return None

    sha7 = (baseline.main_sha or "")[:7]
    if baseline.state == "green":
        return (
            f"existing test `{baseline.test_rel}` pins behavior for `{baseline.surface}` "
            f"and passes unchanged (main@{sha7}: suite green at baseline, "
            f"{baseline.n_tests} tests in {baseline.duration_s:.1f}s, {baseline.ts})"
        )

    # state == "annotated": file ran, pre-existing reds present — the
    # contract holds, scoped explicitly to the file-level suite fingerprint
    # so it cannot be misread as a claim about any single test.
    red_preview = ", ".join(baseline.red[:5])
    return (
        f"existing test `{baseline.test_rel}` pins behavior for `{baseline.surface}`; "
        f"main@{sha7} file-level suite fingerprint at baseline: {len(baseline.red)} "
        f"pre-existing red in this file ({red_preview}, full list in route record, "
        f"{baseline.ts}) - the 'unchanged' clause is scoped to this FILE-LEVEL "
        f"fingerprint, not to any single test"
    )


def verify_behavioral_invariance(
    item: dict, deadline_monotonic: float | None = None,
) -> tuple[bool, str | None]:
    """Tier 3 — the POST-COOK BIND-GATE VERIFIER (not a classification-time
    tier; see module docstring). Called by
    `bundle_autodispatch._reconcile_one` AFTER `cook_item_to_spec` renders
    the fix spec and BEFORE the mechanical bind fires.

    Sha-compare fast path + conditional re-run at advanced sha (spec Design
    "Tier 3 re-verification", option c): when a prior baseline is already
    recorded on this item (the normal caller flow — extract_verification_contract
    ran first), a cheap `rev-parse origin/main` re-confirms the recorded sha
    still matches current main — no worktree, no suite re-run. Sha
    unchanged -> holds against the recorded contract. Sha advanced (a merge
    landed between extract and verify), or no prior baseline recorded at all
    (e.g. called standalone) -> re-run the baseline fresh at whatever sha is
    live now.

    Returns (holds, contract). holds=False means: re-route the item to fork
    with reason "invariance" — never a silent bind.
    """
    recorded = item.get("verification_baseline")
    contract = item.get("verification_contract")

    if recorded and contract:
        root = _repo_root(item.get("repo"))
        current_sha = _resolve_main_sha(root)
        if current_sha is not None and current_sha == recorded.get("main_sha"):
            return contract is not None, contract
        # sha advanced (or the compare itself failed to resolve) — the
        # recorded baseline no longer speaks for current main; fall through
        # to a fresh re-run below.

    new_contract = extract_verification_contract(item, deadline_monotonic=deadline_monotonic)
    return new_contract is not None, new_contract


# ---------------------------------------------------------------------------
# The in-repo cook engine
# ---------------------------------------------------------------------------

def mechanical_item_target_id(repo: str, debt_id: str) -> str:
    """Deterministic kebab-case target id for a cooked mechanical-item spec
    — matches the **Target ID:**/**Repo:** anchored-header regex
    (spec_review.py's _TID_RE / _REPO_RE)."""
    slug = re.sub(r"[^a-z0-9]+", "-", debt_id.lower()).strip("-")
    return f"cr-bundle-item-{repo}-{slug}"


def cook_item_to_spec(item: dict, out_dir: Path | None = None) -> Path:
    """IN-REPO cook engine (bundle-item-level-triage-v0, 2026-08-17 gate
    amendment): renders a minimal, gate-admissible fix spec for one
    mechanical bundle item — four C7 header lines in the first 50 lines
    (**Target ID:**, **Repo:**, **Authority:**, **Consumer:**), the item's
    concrete fix as the sole deliverable, and the Verification Contract
    copied VERBATIM into the DoD.

    Rendering, not analysis: this function never extracts or re-derives the
    contract itself — it renders whatever `item["verification_contract"]`
    already holds (set by the caller from `extract_verification_contract` /
    `verify_behavioral_invariance` before cooking). No cross-repo import, no
    cross-repo call — conductor's `_run_cook_pass` is prior art for the
    pattern only, never a dependency.
    """
    from agents_core.room_paths import room_path

    repo = item["repo"]
    debt_id = item["debt_id"]
    target_id = mechanical_item_target_id(repo, debt_id)
    spec_dir = out_dir if out_dir is not None else room_path("planning.specs")
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{target_id}.md"

    surfaces = item.get("touched_surfaces") or []
    files_line = ", ".join(f"`{s}`" for s in surfaces) if surfaces else "(none named)"
    fix_description = item.get("fix_description") or "(no fix description captured)"
    contract = item.get("verification_contract") or "(no Verification Contract recorded)"
    source_pr = item.get("source_pr")

    text = (
        "---\n"
        f"spec_id: {target_id}\n"
        "status: draft\n"
        f"source: bundle-item-level-triage-v0 mechanical cook (debt_id={debt_id})\n"
        "---\n"
        "\n"
        f"# Spec: {target_id} — mechanical debt-item fix (cooked, no council)\n"
        "\n"
        f"**Target ID:** `{target_id}`\n"
        f"**Repo:** `{repo}`\n"
        "**Authority:** advisory\n"
        "**Consumer:** the night code-review debt-bundle pipeline "
        "(`lapis_pm/bundle_autodispatch.py` `reconcile()`) — the reader of "
        "each bundle item's triage class and the auto-bound fix it produces\n"
        "\n"
        "## Origin\n"
        "\n"
        f"Cooked by `lapis_pm.bundle_triage.cook_item_to_spec` "
        f"(bundle-item-level-triage-v0) from debt item `{debt_id}`"
        f"{f' (source PR #{source_pr})' if source_pr else ''} in the nightly "
        f"debt-bundle sweep for `{repo}`. Classified `mechanical` by the "
        "three-tier triage check (static + blast-radius both pass; "
        "behavioral-invariance verified) — bound directly, advisory, no "
        "philosophical council.\n"
        "\n"
        "## Deliverable\n"
        "\n"
        f"{fix_description}\n"
        "\n"
        f"**Touched surfaces:** {files_line}\n"
        "\n"
        "## Definition of Done\n"
        "\n"
        "- The fix above is applied.\n"
        "- **Verification Contract** (extraction-only, copied verbatim from "
        f"the triage record): {contract}\n"
    )
    spec_path.write_text(text, encoding="utf-8")
    return spec_path


# ---------------------------------------------------------------------------
# Per-item routing record (deliverable (c) — the sidecar the conductor-side
# day-surface renderer can read without importing lapis-pm)
# ---------------------------------------------------------------------------

def write_route_record(spec_path: Path, repo: str, triage_records: list[dict]) -> Path:
    """Write the per-item machine-readable route record adjacent to the
    bundle spec under /srv/lapis/planning/specs/ — deliverable (c). This is NOT
    the day-surface rendering (conductor-side, out of scope for this unit;
    see spec Deliverables)."""
    record_path = Path(str(spec_path) + ".route.json")
    payload = {"spec": spec_path.name, "repo": repo, "items": triage_records}
    record_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return record_path
