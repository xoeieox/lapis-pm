"""Demotion backstop (U3, lapis-pm-autonomy-actuator-v0).

A nightly sweep re-runs the suite on main for recently auto/precedent-
resolved merges; a red suite attributable to the merged PR's touched files
opens a HIGH regression brief and demotes that fork class back to
human-gated. Demotion is automatic; promotion is manual (Erah, from the
Thursday reading). The machine demotes itself; the human promotes.

Attribution is deliberately coarse (Design 7): red main suite + failing test
files intersecting a resolved PR's `changed_paths` = "suspect", not
"proven". The regression brief says so. The demotion fires on suspect.

Cost bound (Design 9 / H2): the sweep processes AT MOST ONE repo's main
baseline per tick (round-robin cursor persisted in the class records), so a
multi-repo sweep spans consecutive ticks.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import verification_attestation as _va
from . import precedent as _precedent

logger = logging.getLogger(__name__)

# Lazy mem accessor (same pattern as router_portfolio._mem): resolved on first
# use so the leaf module imports cleanly off the mem master and in tests.
_mem_store = None


def _mem():
    """Return the shared module-level mem accessor (lazy init).

    On the mem master (BRIX) this is a direct MemoryStore (local = master). Off
    master it is the pm_core shared store (node-identity-checked), so backstop
    writes never diverge into a local sqlite that reverse-replication would
    clobber. Import is best-effort — a missing pm_core (test isolation) falls
    back to a fresh MemoryStore."""
    global _mem_store
    if _mem_store is None:
        try:
            from . import pm_core as _pm_core
            _mem_store = _pm_core._mem()
        except Exception:
            from agents_core.mem import MemoryStore
            _mem_store = MemoryStore()
    return _mem_store


# 14-day lookback for resolved merges (U3.2).
REGRESSION_WINDOW_DAYS = 14

# 14-day cooldown for promotion refusal (Design 7 / Facets revision).
PROMOTE_COOLDOWN_DAYS = 14

# Demoted more than twice within 14d -> --force requires --justification.
PROMOTE_FORCE_JUSTIFY_THRESHOLD = 2

# D4 (lapis-pm-reviewer-leg-repair-v0): the sweep moves OUT of the 60s tick
# into a dedicated lapis-pm-backstop unit (15m cycle, 1200s timeout). The
# in-flight guard in pm_core is process-local (each tick is a fresh process),
# so serialization is enforced by the persistent sweep_started_ts stamp
# (written at sweep START) + a minimum interval >= the sweep cost (~900s).
# The 15m cycle floor is the minimum interval; a sweep that started < 15m ago
# is skipped (no re-sweep of an in-flight or just-completed sweep).
SWEEP_MIN_INTERVAL_S = 900.0  # 15m cycle floor (>= ~900s sweep cost)

# D4 (lapis-pm-reviewer-leg-repair-v0): class-dependent eligibility.
# docs/test-only class: 24h daily window (the existing _autonomy_op_daily_gated
# semantics, per class). running-path class: eligible every sweep cycle (15m)
# ONLY on a NEW demotion (last_regression_ts > last_sweep_ts) — no re-sweeping
# the same red state ~every 16 minutes until green.
DOCS_ONLY_WINDOW_S = 86400.0  # 24h daily window for docs/test-only class
SWEEP_STAMP_KEY = "pm/autonomy-actuator/backstop-sweep-started"

SHADOW_DIR = Path("/srv/lapis/autonomy-shadow")
SHADOW_SWEEP_OUTCOMES = SHADOW_DIR / "sweep-outcomes.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(ts: str):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _failing_test_files(failures: list[str]) -> set:
    """Extract file paths from failing test ids (`path::test` or `path.py::test`)."""
    files = set()
    for f in failures or []:
        path = f.split("::")[0].strip()
        if path:
            files.add(path)
    return files


def _intersect(failing_files: set, changed_paths: list[str]) -> bool:
    """Coarse attribution (Design 7): failing test files intersect the PR's
    changed_paths. Suspect, not proven."""
    if not failing_files or not changed_paths:
        return False
    changed = set(changed_paths)
    for ff in failing_files:
        if ff in changed:
            return True
        # Also match by basename (test file moved / path prefix differs).
        base = ff.rsplit("/", 1)[-1]
        for cp in changed:
            if cp.rsplit("/", 1)[-1] == base:
                return True
    return False


def _recent_resolved(mem, now: str) -> list[dict]:
    """Collect merged-via auto_resolved / precedent_resolved PRs from the
    last 14 days (mem keys `pm/auto-resolved/<tid>/<pr>` + `adjudication`
    records with `source: "precedent"` and their cited PR)."""
    cutoff = _parse_iso(now) - timedelta(days=REGRESSION_WINDOW_DAYS)
    out = []
    seen = set()

    def _add(tid, pr, repo, changed_paths, source, key):
        sig = (tid, pr)
        if sig in seen:
            return
        seen.add(sig)
        out.append({
            "target_id": tid, "pr": pr, "repo": repo,
            "changed_paths": changed_paths or [],
            "source": source, "key": key,
        })

    # Auto-resolved merges.
    try:
        for key in mem.search("pm/auto-resolved") or []:
            raw = mem.get(key)
            if not raw:
                continue
            data = json.loads(raw.get("content", ""))
            if not isinstance(data, dict):
                continue
            ts = _parse_iso(data.get("ts"))
            if ts is None or ts < cutoff:
                continue
            _add(data.get("target_id"), data.get("pr_number"),
                 data.get("repo"), data.get("changed_paths"),
                 "auto_resolved", key)
    except Exception:
        pass

    # Precedent resolutions.
    try:
        for key in mem.search("adjudication") or []:
            raw = mem.get(key)
            if not raw:
                continue
            m = re.search(r"```json\s*(\{.*?\})\s*```",
                          raw.get("content", ""), re.DOTALL)
            if not m:
                continue
            data = json.loads(m.group(1))
            if not isinstance(data, dict) or data.get("source") != "precedent":
                continue
            ts = _parse_iso(data.get("ts"))
            if ts is None or ts < cutoff:
                continue
            _add(data.get("target_id") or key.split("/")[1],
                 data.get("pr"), data.get("repo"),
                 data.get("changed_paths"), "precedent_resolved", key)
    except Exception:
        pass
    return out


def _repos_with_class_records(mem) -> list[str]:
    """Distinct repos that have at least one autonomy-class record (the
    round-robin sweep candidates)."""
    repos = set()
    try:
        for key in mem.search("pm/autonomy-class") or []:
            parts = key.split("/")
            if len(parts) >= 3 and parts[0] == "pm" and parts[1] == "autonomy-class":
                repos.add(parts[2])
    except Exception:
        pass
    return sorted(repos)


def _class_records_for_repo(mem, repo: str) -> list[dict]:
    """All class records for a repo (key + parsed body)."""
    out = []
    try:
        for key in mem.search(f"pm/autonomy-class/{repo}/") or []:
            raw = mem.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw.get("content", ""))
            except (ValueError, TypeError):
                continue
            if isinstance(data, dict):
                out.append({"key": key, "data": data})
    except Exception:
        pass
    return out


def _demote_class(mem, key: str, now: str) -> None:
    raw = mem.get(key)
    data = {}
    if raw:
        try:
            data = json.loads(raw.get("content", ""))
        except (ValueError, TypeError):
            data = {}
    if not isinstance(data, dict):
        data = {}
    data["promoted"] = False
    data["regression_count"] = int(data.get("regression_count", 0)) + 1
    data["last_regression_ts"] = now
    mem.set(key, json.dumps(data, ensure_ascii=False, sort_keys=True),
            tags=["lapis-pm", "pm:autonomy-class"])


def _sweep_started_stamp(mem) -> str | None:
    """Read the persistent sweep_started_ts stamp (D4: written at sweep
    START, not completion — a killed sweep still leaves the stamp, so the
    next cycle sees it and skips rather than stacking)."""
    raw = mem.get(SWEEP_STAMP_KEY)
    if not raw:
        return None
    return str(raw.get("content", "")).strip() or None


def _write_sweep_started_stamp(mem, now: str) -> None:
    mem.set(SWEEP_STAMP_KEY, now, tags=["lapis-pm", "pm:autonomy-actuator"])


def _repo_change_class(recs: list[dict]) -> list[str]:
    """Derive the repo's change_class from its class records' change_classes
    (D4: the class is derived from changed paths via derive_change_classes
    and recorded in the finding/shadow line)."""
    classes = set()
    for r in recs:
        for c in r["data"].get("change_classes", []) or []:
            classes.add(c)
    return sorted(classes)


def _is_docs_only(change_classes: list[str]) -> bool:
    """True when the repo's change_class is docs/test-only (no code/config/
    deploy). D4: docs/test-only class gets the 24h daily window; running-path
    class (any code/config/deploy) gets the 15m cycle eligibility."""
    if not change_classes:
        return False  # unknown class -> treat as running-path (conservative)
    return all(c in ("docs", "test") for c in change_classes)


def _repo_eligible(mem, repo: str, recs: list[dict], now: str) -> tuple[bool, str]:
    """D4 (lapis-pm-reviewer-leg-repair-v0): class-dependent eligibility.

    Returns (eligible, reason). The gate runs POST-selection (the repo must
    be known before its class can be derived — the rev-1 global pre-selection
    gate was wrong).

    - docs/test-only class: 24h daily window (at most one sweep per 24h).
    - running-path class: eligible every 15m cycle ONLY on a NEW demotion
      (last_regression_ts > last_sweep_ts) — no re-sweeping the same red
      state ~every 16 minutes until green (the unbounded re-eligibility rev-1
      implied, ~9/day).
    """
    change_classes = _repo_change_class(recs)
    _now_dt = _parse_iso(now) or datetime.now(timezone.utc)
    if _is_docs_only(change_classes):
        # 24h daily window: at most one sweep per 24h.
        for r in recs:
            last = r["data"].get("last_sweep_ts")
            if last:
                last_dt = _parse_iso(last)
                if last_dt and (_now_dt - last_dt).total_seconds() < DOCS_ONLY_WINDOW_S:
                    return False, "docs_only_window"
        return True, "docs_only_eligible"
    # running-path class: eligible only on a NEW demotion since the last
    # sweep. last_regression_ts > last_sweep_ts means a new demotion occurred
    # after the last sweep — the repo is red and needs re-checking.
    for r in recs:
        last_reg = r["data"].get("last_regression_ts")
        last_sweep = r["data"].get("last_sweep_ts")
        if last_reg:
            last_reg_dt = _parse_iso(last_reg)
            last_sweep_dt = _parse_iso(last_sweep) if last_sweep else None
            if last_reg_dt and (last_sweep_dt is None or last_reg_dt > last_sweep_dt):
                return True, "new_demotion"
    return False, "no_new_demotion"


def sweep(mem, now: str | None = None, repo_cursor: str | None = None) -> dict:
    """Run ONE repo's sweep step (Design 9 round-robin; cursor persists in
    the class records' last_sweep_ts).

    D4 (lapis-pm-reviewer-leg-repair-v0): the sweep runs on a dedicated
    15m cycle (not the 60s tick). Serialization: the persistent
    sweep_started_ts stamp (written at sweep START) + the 15m minimum
    interval. Class-dependent eligibility (docs/test-only 24h, running-path
    on new demotion) is checked POST-selection.

    Returns a per-repo outcome dict:
        {repo, status: "green"|"red"|"inconclusive"|"no_candidates"|
         "skipped_in_flight"|"skipped_not_eligible",
         suspect_prs: [pr], demoted_classes: [key], findings: [key], ...}
    """
    now = now or _now_iso()
    repos = _repos_with_class_records(mem)
    if not repos:
        return {"repo": None, "status": "no_candidates", "suspect_prs": [],
                "demoted_classes": [], "findings": [], "ts": now}

    # D4: serialization — the persistent sweep_started_ts stamp (written at
    # sweep START) + the 15m minimum interval. A sweep that started < 15m ago
    # is skipped (no re-sweep of an in-flight or just-completed sweep). The
    # interval is measured against the `now` parameter (test-deterministic);
    # a real clock is used only when `now` is absent (the default).
    started = _sweep_started_stamp(mem)
    if started:
        started_dt = _parse_iso(started)
        _now_dt = _parse_iso(now) or datetime.now(timezone.utc)
        if started_dt and (_now_dt - started_dt).total_seconds() < SWEEP_MIN_INTERVAL_S:
            return {"repo": None, "status": "skipped_in_flight",
                    "suspect_prs": [], "demoted_classes": [], "findings": [],
                    "ts": now, "started_ts": started}

    # Round-robin: pick the repo whose last_sweep_ts is oldest (or the
    # cursor if given and present).
    def _last_sweep(recs):
        stamps = [r["data"].get("last_sweep_ts") for r in recs
                  if r["data"].get("last_sweep_ts")]
        return min(stamps) if stamps else ""

    if repo_cursor and repo_cursor in repos:
        repo = repo_cursor
    else:
        repo = min(repos, key=lambda r: _last_sweep(_class_records_for_repo(mem, r)))

    recs = _class_records_for_repo(mem, repo)

    # D4: class-dependent eligibility (POST-selection — the repo must be
    # known before its class can be derived).
    eligible, reason = _repo_eligible(mem, repo, recs, now)
    if not eligible:
        return {"repo": repo, "status": "skipped_not_eligible",
                "suspect_prs": [], "demoted_classes": [], "findings": [],
                "ts": now, "reason": reason,
                "change_class": _repo_change_class(recs)}

    # D4: write the persistent sweep_started_ts stamp at sweep START (not
    # completion — a killed sweep still leaves the stamp, so the next cycle
    # sees it and skips rather than stacking).
    _write_sweep_started_stamp(mem, now)

    # Re-run the suite on origin/main in a throwaway worktree (same budget /
    # cleanup discipline as U1). Dedicated main-baseline run: ONE detached
    # worktree at origin/main — no PR head, no _head_branch_for.
    att = _va.attest_main_baseline(repo, base_branch="main",
                                   target_id="__backstop__")
    main_failures = att.get("pr_failures", [])
    result = att.get("result", "inconclusive")

    findings = []
    suspect_prs = []
    demoted_classes = []

    if result == "inconclusive":
        status = "inconclusive"
    elif not main_failures:
        status = "green"
    else:
        status = "red"
        failing_files = _failing_test_files(main_failures)
        for pr in _recent_resolved(mem, now):
            if pr.get("repo") and pr["repo"] != repo:
                continue
            if _intersect(failing_files, pr.get("changed_paths", [])):
                suspect_prs.append(pr["pr"])
                # Demote every fork class implied by that PR.
                for rec in recs:
                    _demote_class(mem, rec["key"], now)
                    demoted_classes.append(rec["key"])

    # D4: sweep count — how many sweeps this repo has had (the steady-state
    # signal: a docs-only repo swept once and a running-path repo swept 20x
    # leave DIFFERENT state, not identical).
    sweep_count = sum(1 for r in recs if r["data"].get("last_sweep_ts"))
    change_class = _repo_change_class(recs)

    # Record the sweep outcome (finding/ key, red or green, per repo).
    # D4: change_class + sweep_count in the finding line (the operator
    # visibility surface — the rev-1 finding had no class field, so a
    # docs-only repo and a running-path repo left identical state).
    finding_key = f"finding/backstop-sweep/{repo}/{today}"
    try:
        mem.set(finding_key, json.dumps({
            "repo": repo, "status": status, "ts": now,
            "main_failures": main_failures,
            "suspect_prs": suspect_prs,
            "demoted_classes": demoted_classes,
            "change_class": change_class,
            "sweep_count": sweep_count,
        }, ensure_ascii=False, sort_keys=True),
                tags=["lapis-pm", "finding", "backstop-sweep"])
        findings.append(finding_key)
    except Exception:
        pass

    # Update last_sweep_ts on every class record for this repo. Re-read the
    # current data (a demotion above may have updated it) so the sweep stamp
    # does not clobber the demotion.
    for rec in recs:
        raw = mem.get(rec["key"])
        data = {}
        if raw:
            try:
                data = json.loads(raw.get("content", ""))
            except (ValueError, TypeError):
                data = {}
        if not isinstance(data, dict):
            data = {}
        data["last_sweep_ts"] = now
        mem.set(rec["key"], json.dumps(data, ensure_ascii=False, sort_keys=True),
                tags=["lapis-pm", "pm:autonomy-class"])

    # Shadow file (Design 10). D4: change_class + sweep_count in the shadow
    # line (the operator visibility surface).
    _record_shadow_sweep(repo, status, main_failures, suspect_prs,
                          demoted_classes, now,
                          change_class=change_class,
                          sweep_count=sweep_count)

    return {
        "repo": repo, "status": status, "suspect_prs": suspect_prs,
        "demoted_classes": demoted_classes, "findings": findings,
        "main_failures": main_failures, "ts": now,
        "change_class": change_class, "sweep_count": sweep_count,
    }


def _record_shadow_sweep(repo, status, main_failures, suspect_prs,
                          demoted_classes, now, *,
                          change_class: list[str] | None = None,
                          sweep_count: int | None = None) -> None:
    try:
        SHADOW_DIR.mkdir(parents=True, exist_ok=True)
        line = {
            "schema_version": "sweep-outcome/v1",
            "ts_utc": now,
            "repo": repo,
            "status": status,
            "main_failures": main_failures,
            "suspect_prs": suspect_prs,
            "demoted_classes": demoted_classes,
            # D4 (lapis-pm-reviewer-leg-repair-v0): change_class + sweep_count
            # in the shadow line (the operator visibility surface — the rev-1
            # line had no class field, so a docs-only repo and a running-path
            # repo left identical state).
            "change_class": change_class or [],
            "sweep_count": sweep_count or 0,
        }
        with open(SHADOW_SWEEP_OUTCOMES, "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("backstop shadow sweep write failed: %s", e)


def open_regression_brief(repo: str, suspect_pr: int, failing_tests: list[str],
                          now: str | None = None) -> None:
    """Open a Pushover HIGH regression brief (U3.2). Suspect, not proven.

    D4 (lapis-pm-reviewer-leg-repair-v0): brief cooldown — one HIGH per
    (repo, suspect PR) per day. A red-storm of re-sweeps (a persistently-red
    running-path repo) must not fire a HIGH brief on every red sweep; the
    cooldown dedups by (repo, suspect_pr) over a 24h window.
    """
    now = now or _now_iso()
    # D4: brief cooldown — one HIGH per (repo, suspect PR) per day.
    cooldown_key = f"finding/backstop-brief-cooldown/{repo}/{suspect_pr}"
    try:
        mem = _mem()
        raw = mem.get(cooldown_key)
        if raw:
            last = _parse_iso(str(raw.get("content", "")))
            if last and (datetime.now(timezone.utc) - last).total_seconds() < DOCS_ONLY_WINDOW_S:
                logger.info(
                    "backstop regression brief suppressed (cooldown): %s PR #%s",
                    repo, suspect_pr,
                )
                return
        mem.set(cooldown_key, now, tags=["lapis-pm", "finding", "backstop-brief-cooldown"])
    except Exception:
        pass  # fail-open: a cooldown read/write error lets the brief fire

    try:
        from agents_core.notify import send_notification, Priority
        send_notification(
            message=(
                f"Backstop sweep: red main suite on {repo}. Suspect (not "
                f"proven) regression from auto/precedent-resolved PR #{suspect_pr}. "
                f"Failing tests: {', '.join(failing_tests[:10])}. "
                f"Fork class demoted to human-gated."
            ),
            title=f"lapis-pm backstop: {repo} PR #{suspect_pr} suspect regression",
            priority=Priority.HIGH,
        )
    except Exception as e:
        logger.warning("backstop regression brief failed: %s", e)


def promote(mem, repo: str, verdict_class: str, change_classes: list[str],
            loc_bucket: str, *, force: bool = False,
            justification: str | None = None) -> dict:
    """Manual promotion path (Design 7/8). Refuses to promote a class with
    regression_count > 0 and last_regression_ts within 14 days without
    --force; with regression_count > 2 within 14 days, even --force requires
    a --justification (logged on the class record)."""
    key = _precedent.class_record_key(repo, verdict_class, change_classes,
                                      loc_bucket)
    raw = mem.get(key)
    data = {}
    if raw:
        try:
            data = json.loads(raw.get("content", ""))
        except (ValueError, TypeError):
            data = {}
    if not isinstance(data, dict):
        data = {}

    now = _now_iso()
    reg_count = int(data.get("regression_count", 0))
    last_reg = _parse_iso(data.get("last_regression_ts"))
    within_cooldown = (
        last_reg is not None
        and (datetime.now(timezone.utc) - last_reg)
        < timedelta(days=PROMOTE_COOLDOWN_DAYS)
    )

    if reg_count > 0 and within_cooldown:
        if not force:
            return {"promoted": False, "key": key,
                    "reason": "regression_cooldown",
                    "regression_count": reg_count,
                    "last_regression_ts": data.get("last_regression_ts")}
        if reg_count > PROMOTE_FORCE_JUSTIFY_THRESHOLD and not justification:
            return {"promoted": False, "key": key,
                    "reason": "force_requires_justification",
                    "regression_count": reg_count}

    data["promoted"] = True
    data["promoted_by"] = "Erah-force" if force else "Erah"
    data["promoted_ts"] = now
    if force and justification:
        data["force_justification"] = justification
        data["force_justification_ts"] = now
    mem.set(key, json.dumps(data, ensure_ascii=False, sort_keys=True),
            tags=["lapis-pm", "pm:autonomy-class"])
    return {"promoted": True, "key": key, "promoted_by": data["promoted_by"],
            "ts": now}
