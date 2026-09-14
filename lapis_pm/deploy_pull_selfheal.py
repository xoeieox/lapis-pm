"""Deploy-pull backstop selfheal (deploy-pull-selfheal-machine-dispatch-v0).

The agents-core-deploy.timer backstop (and the post-land hook) both call
`pm_core._post_land_git_pull`; when a mapped deploy clone is stuck (dead local
branch, staged adds, divergence, credentials) the backstop's only historical
response was one NORMAL Pushover per 10-minute cycle, forever (the 2026-09-02
~370-page storm). This module is the structural fix:

  D1 `classify_pull_failure` - a pure classifier over git state (no LLM call).
  D2 safe-zone self-repair - the 2026-09-02 manual fix's operations
     (reset / switch / pull / branch-delete), gated on the D1 evidence flags,
     silent on success (ledger entry + deploy-log line, zero Pushover).
  D3 repair ledger + one Desk gem per stuck condition - the ledger
     (/srv/lapis/lapis-state/deploy-pull-repair-ledger.json) is the dedup
     authority; only state transitions page (I2).
  D4 windowed machine action - after the 20-minute human window (or a
     `salvage_now` gem option) the host actor salvages local work to a pushed
     `lapis/<target_id>/salvage` branch + PR, restores the tree to main, and
     closes the loop on healthy verification.
  D5 PR-path-stall hold - the D3/D4 pass is the named observer; a stalled
     salvage PR becomes `worker_failed` with exactly one HIGH page.
  D6 `fetch_failed` (credentials/network) - an Erah-gate class: gem +
     station incident, one HIGH page at 20 minutes, then silence.
  D7 named read-only seams only (ledger, station db, targets, gems, salvage
     PRs); `pm_core.force_dispatch` is the documented LLM-dispatch seam and is
     never called from this module.
  D8 no new env vars, no systemd changes.

Lane declaration (I8): the machine/Erah split below is encoded from Erah's
documented declarations - see the docstrings on MACHINE_LANE_ACTIONS /
ERAH_GATE_CLASSES.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from agents_core.room_paths import room_path

from .file_lock import FileLockTimeout, file_lock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEDGER_FILE = room_path("lapis_state") / "deploy-pull-repair-ledger.json"
_LEDGER_LOCK_FILE = room_path("lapis_state") / "deploy-pull-repair-ledger.lock"
_LEDGER_LOCK_TIMEOUT_S = 5.0

# D1: the one fetch the classifier performs, pinned (H2: the 120s unit
# deadline). A failed fetch is a class (fetch_failed), not an error.
FETCH_TIMEOUT_S = 15.0
# D2/D4: every git surgery op is list-args with this timeout (the existing
# pm_core.py pull-timeout pattern).
GIT_OP_TIMEOUT_S = 30.0
# D1: dirty tracked files touched within this window are "active" - a human
# or agent may be in the tree: no action, no page, no ledger state (I3).
ACTIVE_DIRTY_WINDOW_S = 1800
# D1: porcelain is capped in every store this module writes (M9).
PORCELAIN_CAP = 200
# D4: Mirror Council 2026-09-02 gate (Erah-ratified): the human window is
# 2 cycles of the 10-minute backstop = 20 minutes wall-clock since the
# persisted first_seen. `salvage_now` bypasses the window.
HUMAN_WINDOW_CYCLES = 2
_BACKSTOP_PERIOD_S = 600  # agents-core-deploy.timer OnUnitActiveSec=10min
HUMAN_WINDOW_S = HUMAN_WINDOW_CYCLES * _BACKSTOP_PERIOD_S
# D6: fetch_failed pages once, 20 wall-clock minutes after first_seen.
FETCH_FAILED_PAGE_AFTER_S = HUMAN_WINDOW_S
# D4: the temporary local ref the stale-dirty salvage commit lands on.
# NEVER commit directly on main (Facets gate condition 2, 2026-09-02).
_SALVAGE_TMP_REF = "deploy-pull-salvage-tmp"
_DEPOSITED_BY = "deploy-pull-selfheal-core-v0"

# D1 - classification labels (first match wins over the evidence; any
# classifier exception or uncollectable input is `unknown`, never a safe
# class - I1).
CLASS_FETCH_FAILED = "fetch_failed"
CLASS_SAFE_DEAD_BRANCH = "safe_dead_branch"
CLASS_SAFE_STAGED_ADDS = "safe_staged_adds"
CLASS_DIVERGED = "diverged"
CLASS_ACTIVE_DIRTY = "active_dirty"
CLASS_STALE_DIRTY = "stale_dirty"
CLASS_UNKNOWN = "unknown"
_SAFE_CLASSES = frozenset({CLASS_SAFE_DEAD_BRANCH, CLASS_SAFE_STAGED_ADDS})
# D4: the classes the windowed salvage acts on.
_SALVAGE_CLASSES = frozenset({CLASS_STALE_DIRTY, CLASS_DIVERGED})

# ---------------------------------------------------------------------------
# I8 - the machine/Erah lane split, encoded from Erah's documented
# declarations. These constants are the source of record for which failures
# the machine acts on autonomously and which it stops and surfaces.
#
# decision/productive-autonomy-held-node-2026-07-27 (Erah voice, ratified):
#   "HELD NODE = MACHINE SELF-OPERABILITY: the pipeline investigates,
#   documents, repairs itself, escalating to Erah ONLY at invariant/ethical
#   gates." The load-bearing split is machine-addressable work vs Erah-
#   specific work; only the latter belongs on his plate.
# decision/stalled-target-triage-to-agent-not-Erah-2026-06-24 (Erah voice):
#   "the Pushover notifications honestly shouldn't be for me, they should be
#   for an agent" - stall events route to agent triage, not the operator.
# decision/Erah-pushover-signal-policy-llm-first-responder-2026-09-02
#   (Erah, 2026-09-02): Pushover is "you have to do something" only;
#   routine/health/success classes are silent (visibility lives in the
#   deploy log, ledger entries, and Desk gems); the first responder for
#   Lapis-machinery-domain notifications is an LLM, not Erah.
# ---------------------------------------------------------------------------

# Critical fix #3 (rev-2): the I8 lane-constant mem-key assertion reads
# `MACHINE_LANE_ACTIONS.__doc__`. A bare `frozenset.__doc__` is the builtin
# (read-only, citing no mem key) and a bare string literal after the
# assignment does not attach - so the constants are declared as CLASSES whose
# class docstring IS the `.__doc__` the test reads. They are documentation
# sentinels only (never iterated / used for membership), so the class shape
# is behavior-preserving.
class MACHINE_LANE_ACTIONS:
    """Machine-lane actions: machine-addressable work the backstop executes
autonomously (decision/productive-autonomy-held-node-2026-07-27). No LLM
call appears in any of these (I9); the LLM appears only in the downstream
review/ratify loop. The machine surfaces only via the invisible-affordance
channels Erah ratified (decision/Erah-invisible-affordances-2026-06-08): the
Desk gem + the deploy-log provenance line, never a page storm."""
    VALUE = frozenset({
        "classify_pull_failure",   # D1 - mechanical, LLM-free
        "safe_zone_self_repair",   # D2 - git surgery under the classifier's proof
        "salvage_and_restore",     # D4 - commit-not-discard salvage + PR (Slice 2)
        "ledger_write",            # D3 - the dedup authority
        "desk_gem_deposit",        # D3 - one gem per stuck condition
        "close_the_loop",          # D4.5 - healthy verification
    })


class ERAH_GATE_CLASSES:
    """Erah's gates: the classes where this spec stops and surfaces, never
pages repeatedly (decision/productive-autonomy-held-node-2026-07-27: escalate
to Erah ONLY at invariant/ethical gates; the signal policy is
decision/Erah-pushover-signal-policy-llm-first-responder-2026-09-02). In Slice
Slice 2 all three gates are live: D4 salvage_pr_land_or_discard (ratify the
salvage PR or close it), D5 worker_failed_hold (PR path stalled; PM
judgment), D6 fetch_failed (credentials/network). Stall routing follows
decision/stalled-target-triage-to-agent-not-Erah-2026-06-24."""
    VALUE = frozenset({
        "salvage_pr_land_or_discard",  # D4 - ratify the salvage PR or close it
        "worker_failed_hold",          # D5 - PR path stalled; PM judgment
        "fetch_failed",                # D6 - credentials/network; a host actor
                                       # cannot rotate a token
    })


_GEM_OPTIONS = [
    {"key": "salvage_now", "title": "Salvage now",
     "sub": "skip the 20-minute window - the machine salvages local work to a "
            "pushed salvage branch + PR and restores the tree to main",
     "primary": True},
    {"key": "ack_watch", "title": "Known - keep watching",
     "sub": "suppress the machine action while this stays open"},
    {"key": "not_real", "title": "Not a real issue",
     "sub": "dismiss as a false positive"},
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _git(path: str, *args: str, timeout: float = GIT_OP_TIMEOUT_S) -> subprocess.CompletedProcess:
    """List-args subprocess git (no shell=True; refs pass as single args)."""
    return subprocess.run(
        ["git", "-C", path, *args],
        capture_output=True, text=True, timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Credential redaction (H1: both agents-core clones embed the Forgejo token in
# the remote URL and git echoes the tokenized URL in fetch-failure stderr;
# the gem store is no-auth and tailnet-reachable, so the redacted string is
# what lands in every store this module writes). In-repo precedent:
# agents_core/forgejo.py scrubs the token from push URLs.
# ---------------------------------------------------------------------------

_USERINFO_RE = re.compile(r"(https?://)[^/@\s]+@")


def redact_credentials(text: str) -> str:
    """Mask user:pass in http(s)://user:pass@host to http(s)://***@host."""
    if not text:
        return text
    return _USERINFO_RE.sub(r"\1***@", text)


def _capped_porcelain(lines: list[str]) -> list[str]:
    """Cap porcelain at PORCELAIN_CAP lines + '+N more' (M9)."""
    if len(lines) <= PORCELAIN_CAP:
        return list(lines)
    return list(lines[:PORCELAIN_CAP]) + [f"+{len(lines) - PORCELAIN_CAP} more"]


# ---------------------------------------------------------------------------
# D1 - classification
# ---------------------------------------------------------------------------

def _evidence_has_anomaly(evidence: dict) -> bool:
    """True when the evidence shows a real (if unclassifiable) stuck condition
    vs a clean tree. A clean tree (branch=main, ahead=0, no dirty porcelain,
    fetch_rc=0) classifies `unknown` with NO anomaly - D4.5 closes any open
    entry for the path in place (the tree already healed) and no gem is
    deposited. Any divergence from that baseline is an anomaly worth a gem."""
    if evidence.get("fetch_rc"):
        return True
    if evidence.get("branch") not in (None, "main"):
        return True
    if (evidence.get("ahead") or 0) > 0:
        return True
    if evidence.get("porcelain"):
        return True
    return False


def _collect_evidence(path: str, now: float | None = None) -> dict:
    """Collect the git-state evidence the classifier decides over.

    Raises on any uncollectable input (bad path, git failure) - the caller
    maps that to `unknown` (I1: error-unknown, never a safe class).
    """
    now = now if now is not None else time.time()

    fetch = _git(path, "fetch", "origin", "main", timeout=FETCH_TIMEOUT_S)
    fetch_stderr = redact_credentials(fetch.stderr or "")

    # fetch_failed short-circuits BEFORE branch/rev-list/status: the fetch
    # failure is the first-match class (credentials/network), the rest of the
    # evidence is not collectable (a failed fetch leaves the refs unreliable)
    # and a subsequent git failure would wrongly map this to `unknown` instead
    # of `fetch_failed` (the I1 error-unknown rule must not shadow the
    # higher-precedence fetch-failed class - test_fetch_failed_precedence).
    if fetch.returncode != 0:
        return {
            "branch": None,
            "ahead": None,
            "behind": None,
            "fetch_rc": fetch.returncode,
            "fetch_stderr": fetch_stderr,
            "porcelain": [], "dirty_files": [], "mtimes": [],
            "now": now,
            "dead_branch": False, "staged_adds": False,
        }

    branch_res = _git(path, "branch", "--show-current")
    if branch_res.returncode != 0:
        raise RuntimeError(f"branch --show-current failed: {branch_res.stderr.strip()[:200]}")
    branch = branch_res.stdout.strip()

    ahead = behind = None
    lr = _git(path, "rev-list", "--left-right", "--count", "HEAD...origin/main")
    if lr.returncode == 0:
        # output is "<left>\t<right>" (left = HEAD-only, right = origin/main-
        # only); tab- or space-separated.
        parts = (lr.stdout or "").replace("\t", " ").split()
        if len(parts) == 2:
            ahead, behind = int(parts[0]), int(parts[1])
    if ahead is None:
        raise RuntimeError(f"rev-list left-right count failed: {(lr.stderr or '').strip()[:200]}")

    status = _git(path, "status", "--porcelain")
    if status.returncode != 0:
        raise RuntimeError(f"status --porcelain failed: {status.stderr.strip()[:200]}")
    porcelain = [line for line in status.stdout.splitlines() if line]

    # dirty tracked files = non-?? lines; mtime of each (oldest is the I3
    # signal and the D3 oldest_dirty_mtime schema field).
    dirty_files: list[str] = []
    mtimes: list[float] = []
    for line in porcelain:
        if line.startswith("??"):
            continue
        # porcelain format: XY PATH (XY is two chars, space, then the path).
        # Renames appear as "R  old -> new" (or "RM"): the NEW path is the
        # one that exists on disk - stat the old side and it reads as
        # "missing", which is exactly the conservative fallback below.
        rel = line[3:]
        if " -> " in rel:
            rel = rel.rsplit(" -> ", 1)[1]
        if rel.startswith('"') and rel.endswith('"'):
            rel = rel[1:-1]
        dirty_files.append(rel)
        try:
            mtimes.append((Path(path) / rel).stat().st_mtime)
        except OSError:
            # renamed/unresolvable: use the repo mtime as a conservative
            # (older) fallback so a missing mtime never reads as "active".
            try:
                mtimes.append(Path(path).stat().st_mtime)
            except OSError:
                mtimes.append(0.0)

    return {
        "branch": branch,
        "ahead": ahead,
        "behind": behind,
        "fetch_rc": fetch.returncode,
        "fetch_stderr": fetch_stderr,
        "porcelain": _capped_porcelain(porcelain),
        "dirty_files": list(dirty_files),
        "mtimes": list(mtimes),
        "now": now,
        # evidence flags (composition rule: downstream repairs gate on these,
        # never on the label string - M2):
        "dead_branch": bool(branch != "main" and ahead == 0),
        "staged_adds": bool(
            dirty_files
            and all(line[:2] == "A " or line[:1] == "A" for line in porcelain if not line.startswith("??"))
        ),
    }


def classify_pull_failure(path: str, trigger: str, *, now: float | None = None) -> tuple[str, dict]:
    """Classify a failed pull at `path` (D1).

    Returns (class, evidence). First match wins over the evidence; any
    exception or uncollectable input is `unknown` (I1). Pure over
    subprocessed git output - no LLM call (I9).
    """
    try:
        ev = _collect_evidence(path, now=now)
    except Exception as exc:  # noqa: BLE001 - I1: error-unknown, never safe
        return CLASS_UNKNOWN, {
            "branch": None, "ahead": None, "behind": None,
            "fetch_rc": None, "fetch_stderr": "",
            "porcelain": [], "dirty_files": [], "mtimes": [],
            "now": now if now is not None else time.time(),
            "dead_branch": False, "staged_adds": False,
            "error": str(exc)[:300],
        }

    if ev["fetch_rc"] != 0:
        return CLASS_FETCH_FAILED, ev

    dirty = any(not line.startswith("??") for line in ev["porcelain"])

    if ev["dead_branch"]:
        return CLASS_SAFE_DEAD_BRANCH, ev
    if ev["staged_adds"]:
        return CLASS_SAFE_STAGED_ADDS, ev
    if ev["ahead"] is not None and ev["ahead"] > 0:
        return CLASS_DIVERGED, ev
    if dirty:
        window = ACTIVE_DIRTY_WINDOW_S
        if any((ev["now"] - m) < window for m in ev["mtimes"]):
            return CLASS_ACTIVE_DIRTY, ev
        return CLASS_STALE_DIRTY, ev
    return CLASS_UNKNOWN, ev


# ---------------------------------------------------------------------------
# D3 - ledger (atomic tmp + flush + fsync + replace; flock; corrupt
# quarantine). The ledger, not snapshot-newness, decides gem + action dedup
# (I4).
# ---------------------------------------------------------------------------

def compute_signature(path: str, cls: str) -> str:
    """sha256(path|class)[:16] - a stable id over (path, class) only (the
    compute_signature shape of deploy_inventory_repair.py)."""
    return hashlib.sha256(f"{path}|{cls}".encode("utf-8")).hexdigest()[:16]


def _ledger_paths() -> tuple[Path, Path]:
    return _LEDGER_FILE, _LEDGER_LOCK_FILE


def read_ledger(path: Path | None = None) -> dict:
    """Read the ledger; a corrupt file is quarantined (renamed aside) and
    reads as empty (I5).

    Critical fix #2 (rev-2): a MISSING file is the normal first-run state and
    reads as EMPTY, silently (debug log) - NOT as corrupt. Only an existing
    but unparseable file quarantines. The head treated FileNotFoundError
    (an OSError subclass) as corrupt, so the first run logged a scary
    'quarantine failed' warning and attempted a rename of an absent file."""
    target = path or _LEDGER_FILE
    try:
        raw = target.read_text()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("ledger is not a dict")
        return data
    except FileNotFoundError:
        # Normal first-run state: the ledger does not exist yet.
        logger.debug("[deploy-pull-selfheal] ledger %s absent - empty ledger", target)
        return {}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        try:
            q = target.with_suffix(target.suffix + f".corrupt-{int(time.time())}")
            target.replace(q)
            logger.warning(
                "[deploy-pull-selfheal] ledger %s corrupt (%s); quarantined to %s",
                target, exc, q,
            )
        except OSError:
            logger.warning(
                "[deploy-pull-selfheal] ledger %s corrupt (%s); quarantine failed",
                target, exc,
            )
        return {}


def write_ledger(ledger: dict, path: Path | None = None) -> None:
    """Atomic write: tmp + flush + fsync + os.replace (I5). The fsync is the
    one line the landed deploy_inventory_repair sibling lacks."""
    target = path or _LEDGER_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=".deploy-pull-repair-ledger-", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(ledger, indent=2, sort_keys=True))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except OSError as e:
        logger.warning("[deploy-pull-selfheal] ledger write failed: %s", e)


def _new_entry(path: str, repo: str, cls: str, evidence: dict, now: str) -> dict:
    mtimes = [m for m in evidence.get("mtimes", []) if m]
    return {
        "signature": compute_signature(path, cls),
        "path": path,
        "repo": repo,
        "class": cls,
        "first_seen": now,
        "last_seen": now,
        "times_seen": 1,
        "status": "open",
        "gem_id": None,
        "pr_number": None,
        "salvage_branch": None,
        "dispatch_error": None,
        "attempts": 0,
        "oldest_dirty_mtime": min(mtimes) if mtimes else None,
    }


# ---------------------------------------------------------------------------
# D3 - Desk gem deposit + provenance + station escalation (reuses the
# landed deploy_inventory_repair machinery rather than reinventing it).
# ---------------------------------------------------------------------------

def _stakes_line(repo: str, path: str, times_seen: int) -> str:
    """Mechanical stakes line: read the last deploy-inventory-status.json
    pass fail-soft; else the backstop-page-storm fact."""
    try:
        status = json.loads(
            (room_path("lapis_state") / "deploy-inventory-status.json").read_text()
        )
        for clone in status.get("clones", []):
            if clone.get("path") == path:
                live = [bu.get("unit") for bu in (clone.get("backing_units") or []) if bu.get("live")]
                if live:
                    line = f"backs {len(live)} live unit(s) ({', '.join(live)})"
                else:
                    line = "not currently backing a live unit"
                if times_seen > 1:
                    line += f"; seen across {times_seen} passes"
                return line
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    line = (
        "backstop pages the operator every 10 min while stuck; runtime "
        "imports come from the sibling clone, impact low"
    )
    if times_seen > 1:
        line += f"; seen across {times_seen} passes"
    return line


def build_gem_payload(*, repo: str, path: str, cls: str, evidence: dict,
                      times_seen: int, first_seen: str) -> dict:
    """The D3 gem body. The state dump IS the diagnosis - no model call."""
    name = Path(path).name
    state_dump = [
        f"branch={evidence.get('branch')} ahead={evidence.get('ahead')} "
        f"behind={evidence.get('behind')} fetch_rc={evidence.get('fetch_rc')}",
        f"class={cls} first_seen={first_seen}",
    ]
    for line in evidence.get("porcelain", [])[:20]:
        state_dump.append(line)
    if evidence.get("fetch_stderr"):
        state_dump.append(f"fetch: {evidence['fetch_stderr'][:300]}")

    context = [
        {"label": "Symptom", "lines": [
            f"deploy pull stuck on {path} ({repo}): {cls} since {first_seen} "
            f"({times_seen} cycle(s) failed)",
        ]},
        {"label": "Diagnosis", "lines": state_dump},
        {"label": "Suggested direction", "lines": [
            "after the 20-minute window the machine salvages local work to a "
            "pushed salvage branch + PR and restores the tree to main - "
            "click salvage_now to do it now",
        ]},
        {"label": "Stakes", "lines": [_stakes_line(repo, path, times_seen)]},
    ]
    if times_seen > 1:
        context.append({"label": "Recurrence", "lines": [f"seen {times_seen} time(s)"]})

    return {
        "title": f"deploy pull stuck: {repo} {name}",
        "ask": f"deploy pull stuck on {name} ({cls}) - ratify the machine action or act",
        "why": (
            "A mapped deploy clone is stuck and the machine owns it until a "
            "human acts (deploy-pull-selfheal-core-v0)."
        ),
        "context": context,
        "options": _GEM_OPTIONS,
        "state": "needs",
        "origin": "from · deploy-pull-selfheal",
        "agent": "repair-expert",
        "deposited_by": _DEPOSITED_BY,
    }


def deposit_gem(payload: dict) -> str | None:
    """POST /v0/decision-gems (the landed deploy_inventory_repair helper).
    Returns gem_id on 201, None on any failure (fail-soft)."""
    from . import deploy_inventory_repair as _dir
    return _dir.deposit_gem(payload)


def emit_provenance(*, gem_id: str, signature: str, clone_path: str, model: str) -> None:
    """Attribution rail (I5): a gem with no attribution entry is a provenance
    hole. Reuses the landed helper (model='mechanical' - no model call fired)."""
    from . import deploy_inventory_repair as _dir
    try:
        _dir.emit_provenance(gem_id=gem_id, signature=signature,
                             clone_path=clone_path, model=model)
    except Exception as exc:  # best-effort, never blocks the deposit
        logger.warning("[deploy-pull-selfheal] provenance deposit failed: %s", exc)


def station_escalate(*, repo: str, path: str, cls: str, first_seen: str,
                     times_seen: int) -> None:
    """D7: write the non-load-bearing repair-station record (the 2026-08-01
    reader gap: nothing reads intake yet; this is history for the future
    reader). Never raises.

    Slice-2 corrected import (C1): `first` lives in
    agents_core.repair_station.types (the EscalationPolicy factory), NOT in
    .escalate - the PR #314 head import (`from ...escalate import escalate,
    first`) raised ImportError, which this wrapper's except clause would have
    swallowed into a silent warning (the parent's dead-call bug)."""
    try:
        from agents_core.repair_station.escalate import escalate
        from agents_core.repair_station.types import Tier, first
        escalate(
            station_id=f"deploy/pull-{repo}",
            stable_pointer=str(_LEDGER_FILE),
            error_signal={
                "path": path, "class": cls,
                "first_seen": first_seen, "times_seen": times_seen,
            },
            author_intent=(
                "a mapped deploy clone is stuck and the machine owns it "
                "until a human acts"
            ),
            escalation_policy=first(),
            tier=Tier.NORMAL,
            signature_fields=["path", "class"],
        )
    except Exception as exc:
        logger.warning("[deploy-pull-selfheal] station escalate failed: %s", exc)


def _send_page(message: str, title: str, priority) -> None:
    try:
        from agents_core.notify import send_notification
        send_notification(message=message, title=title, priority=priority)
    except Exception as exc:
        logger.warning("[deploy-pull-selfheal] pushover failed: %s", exc)


# ---------------------------------------------------------------------------
# D2 - safe-zone self-repair (machine lane; both triggers; silent on success)
# ---------------------------------------------------------------------------

def self_repair(path: str, evidence: dict, trigger: str) -> bool:
    """D2: the 2026-09-02 manual fix's operations, gated on the evidence
    flags (never the label - M2). Returns True on success + verify; on any
    failure the caller routes to D3 as `unknown`.

    Success = steps 1-3 + verify. The best-effort branch delete (step 4)
    never routes a verified-healthy tree to failure.
    """
    steps: list[str] = []
    try:
        if evidence.get("staged_adds"):
            r = _git(path, "reset")
            if r.returncode != 0:
                return False
            steps.append("reset")
        if evidence.get("dead_branch"):
            r = _git(path, "switch", "main")
            if r.returncode != 0:
                return False
            steps.append("switch-main")
        r = _git(path, "pull", "--ff-only", "origin", "main")
        if r.returncode != 0:
            return False
        steps.append("pull-ff-only")

        # step 4 - best-effort dead-branch delete, never a failure route.
        if evidence.get("dead_branch"):
            branch = evidence.get("branch")
            if branch and branch != "main" and evidence.get("ahead") == 0:
                r = _git(path, "branch", "-d", branch)
                if r.returncode != 0:
                    logger.warning(
                        "[deploy-pull-selfheal] D2 branch -d %s failed (best-effort): %s",
                        branch, (r.stderr or "").strip()[:200],
                    )
                else:
                    steps.append(f"branch-d:{branch}")

        # step 5 - verify: main / 0/0 / clean.
        ok, _ev = _verify_healthy(path)
        if not ok:
            return False
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("[deploy-pull-selfheal] D2 self-repair errored at %s: %s", path, exc)
        return False

    # Success: durable record under the same flock discipline (ledger entry
    # status=resolved + deploy-log line), NO Pushover (I2 silence default).
    _record_resolved(path, evidence, trigger, steps)
    return True


def _verify_healthy(path: str) -> tuple[bool, dict]:
    """The cheap local verify predicate (no network): branch==main,
    0/0, no non-?? porcelain."""
    try:
        branch_res = _git(path, "branch", "--show-current")
        lr = _git(path, "rev-list", "--left-right", "--count", "HEAD...origin/main")
        status = _git(path, "status", "--porcelain")
    except (subprocess.TimeoutExpired, OSError):
        return False, {}
    if branch_res.returncode != 0 or lr.returncode != 0 or status.returncode != 0:
        return False, {}
    branch = branch_res.stdout.strip()
    parts = (lr.stdout or "").split()
    if len(parts) != 2 or parts[0] != "0" or parts[1] != "0":
        return False, {}
    dirty = any(line and not line.startswith("??") for line in status.stdout.splitlines())
    if dirty:
        return False, {}
    return True, {"branch": branch}


def _record_resolved(path: str, evidence: dict, trigger: str, steps: list[str]) -> None:
    """D2 success record: ledger entry (status=resolved, first_seen=
    last_seen=now, times_seen=1, attempts=1) + deploy-log line, under the
    ledger flock. A resolved-by-self-repair entry that recurs later re-opens
    in place per the D3 recurrence rule."""
    now = _now_iso()
    cls = CLASS_SAFE_DEAD_BRANCH if evidence.get("dead_branch") else CLASS_SAFE_STAGED_ADDS
    try:
        with file_lock(_LEDGER_LOCK_FILE, _LEDGER_LOCK_TIMEOUT_S):
            ledger = read_ledger()
            signature = compute_signature(path, cls)
            entry = ledger.get(signature)
            if entry is None:
                entry = _new_entry(path, "agents-core", cls, evidence, now)
                entry["status"] = "resolved"
                entry["attempts"] = 1
                ledger[signature] = entry
            else:
                entry["status"] = "resolved"
                entry["last_seen"] = now
                entry["times_seen"] = 1
                entry["attempts"] = 1
            write_ledger(ledger)
    except (FileLockTimeout, Exception) as exc:  # noqa: BLE001 - best-effort record
        logger.warning("[deploy-pull-selfheal] D2 ledger record failed: %s", exc)
    _deploy_log_line(path, f"self-repaired ({', '.join(steps) or 'pull'}) | {trigger}")


def _deploy_log_line(path: str, summary: str) -> None:
    """Append one provenance line to the deploy log (best-effort)."""
    try:
        from . import pm_core as _pm
        with open(_pm._DEPLOY_LOG, "a") as f:
            f.write(f"- `{_now_iso()}` | {path} | {summary}\n")
    except Exception as exc:
        logger.warning("[deploy-pull-selfheal] deploy-log write failed: %s", exc)


# ---------------------------------------------------------------------------
# D3 - first-detection / recurrence handling (under the ledger flock)
# ---------------------------------------------------------------------------

def _gem_decisions() -> dict:
    """Fetch decided + superseded gems once; map gem_id -> option_key (or
    'superseded'). Fail-soft: {} on any failure."""
    from . import deploy_inventory_repair as _dir
    out: dict = {}
    for gem in _dir._fetch_gems_by_state("decided"):
        decision = gem.get("decision_json") or {}
        out[gem.get("gem_id")] = decision.get("option_key")
    for gem in _dir._fetch_gems_by_state("superseded"):
        out.setdefault(gem.get("gem_id"), "superseded")
    return out


def _reconcile_terminal(ledger: dict, decisions: dict) -> set[str]:
    """I10: every gem option has a consumer. ack_watch -> acked (D4
    suppressed while acked); not_real -> dismissed_fp; superseded open gem ->
    resolved (close-the-loop already handled the tree).

    Returns the set of GEM IDs this pass changed to a terminal state (acked /
    dismissed_fp / resolved) via a gem decision. A just-dismissed or
    just-acked entry must NOT be re-opened / re-deposited by the D3 logic in
    the same pass (a dismissed_fp set HERE means the operator just dismissed
    it - do not immediately re-open)."""
    changed_gem_ids: set[str] = set()
    for entry in ledger.values():
        gem_id = entry.get("gem_id")
        if not gem_id or gem_id not in decisions:
            continue
        decision = decisions[gem_id]
        if entry.get("status") != "open":
            continue
        if decision == "ack_watch":
            entry["status"] = "acked"
            changed_gem_ids.add(gem_id)
        elif decision == "not_real":
            entry["status"] = "dismissed_fp"
            changed_gem_ids.add(gem_id)
        elif decision == "superseded":
            entry["status"] = "resolved"
            changed_gem_ids.add(gem_id)
    return changed_gem_ids


def _first_detection(ledger: dict, signature: str, entry: dict, *, repo: str,
                     evidence: dict, now: str, recurred: bool = False) -> str:
    """Open or re-open the entry (first detection, dismissed_fp, or
    resolved-recurrence - the recurrence resets first_seen + times_seen so a
    fresh stuck episode re-arms the human window instead of acting
    immediately). Deposits one gem + station incident. Returns an action
    string.

    Critical fix #7 (rev-2): `recurred` is passed EXPLICITLY by the caller
    from the ORIGINAL ledger lookup (the pre-base entry). The head computed
    `recurred = entry is not None` on the always-instantiated `base`, which
    was never None - so the flag was ALWAYS true and the `deposited:new`
    label never fired. The two labels are now distinct."""
    first_seen = now
    times_seen = 1
    if recurred:
        entry = dict(entry)
        entry["status"] = "open"
        entry["first_seen"] = now
        entry["times_seen"] = 1
        entry["gem_id"] = None
        entry["pr_number"] = None
        entry["salvage_branch"] = None
        entry["attempts"] = 0
        # C6: a recurring fetch_failed episode re-arms the 20-minute page
        # timer; the page flag must reset with it or the re-armed episode
        # never re-pages (silent defeat of the D6 gate on recurrence).
        entry["fetch_page_sent"] = False

    payload = build_gem_payload(
        repo=repo, path=entry["path"], cls=entry["class"],
        evidence=evidence, times_seen=times_seen, first_seen=first_seen,
    )
    gem_id = deposit_gem(payload)
    if gem_id is None:
        # Non-201/unreachable: leave the ledger unrecorded for this signature
        # so the next pass retries (the landed sibling's D5 discipline).
        return "skip:deposit-failed"

    entry["gem_id"] = gem_id
    entry["first_seen"] = first_seen
    entry["last_seen"] = now
    entry["times_seen"] = times_seen
    ledger[signature] = entry
    emit_provenance(gem_id=gem_id, signature=signature,
                    clone_path=entry["path"], model="mechanical")
    # D7 (Slice 2): the non-load-bearing station incident - read-only seam
    # toward the station brain, never raises.
    station_escalate(
        repo=repo, path=entry["path"], cls=entry["class"],
        first_seen=first_seen, times_seen=times_seen,
    )
    return "deposited:recurred" if recurred else "deposited:new"


# ---------------------------------------------------------------------------
# D4 - windowed machine action (the host actor)
# ---------------------------------------------------------------------------

def _window_elapsed(entry: dict, now: float | None = None) -> bool:
    """Wall-clock since the persisted first_seen >= HUMAN_WINDOW_S (D4).
    `last_seen` is deliberately not part of the condition (it feeds only the
    Recurrence block)."""
    now = now if now is not None else time.time()
    try:
        first_seen = _parse_iso(entry["first_seen"]).timestamp()
    except (KeyError, ValueError, TypeError):
        return False
    return (now - first_seen) >= HUMAN_WINDOW_S


def _salvage_now_requested(entry: dict, decisions: dict) -> bool:
    gem_id = entry.get("gem_id")
    return bool(gem_id and decisions.get(gem_id) == "salvage_now")


def _ensure_target(repo: str, signature: str) -> str | None:
    """Ensure the repair target deploy-repair-{repo}-{sig6} exists (the
    bundle_autodispatch._bind precedent, in-process). Returns the target_id
    or None (best-effort - a missing target does not block the salvage)."""
    target_id = f"deploy-repair-{repo}-{signature[:6]}"
    try:
        from agents_core.targets import TargetStore
        store = TargetStore()
        if store.get(target_id) is not None:
            return target_id
        target = store.create(
            target_id,
            title=f"deploy-pull repair: {repo} {signature[:6]}",
            description=(
                f"Salvage + repair target for a stuck deploy clone "
                f"(deploy-pull-selfheal-machine-dispatch-v0). Signature "
                f"{signature}."
            ),
        )
        target.bind_pm(repo, "advisory")
        target.save()
        return target_id
    except Exception as exc:
        logger.warning("[deploy-pull-selfheal] ensure target %s failed: %s", target_id, exc)
        return None


def _open_salvage_pr_exists(repo: str, target_id: str) -> bool | None:
    """Check for an existing open PR on lapis/<target_id>/ (the same Forgejo
    scan L1.D1 uses). True/False; None (degrade-open) on error."""
    try:
        from agents_core.forgejo import get_open_prs
        repo_name = repo.split("/")[-1]
        owner = repo.split("/", 1)[0] if "/" in repo else None
        for pr in get_open_prs(repo_name, owner=owner):
            ref = (pr.get("head") or {}).get("ref", "")
            if ref.startswith(f"lapis/{target_id}/"):
                return True
        return False
    except Exception as exc:
        logger.warning("[deploy-pull-selfheal] open-PR scan failed (%s): %s", repo, exc)
        return None


def _salvage(path: str, repo: str, entry: dict, target_id: str) -> tuple[bool, dict]:
    """D4.2: the mechanical, lossless-by-construction salvage. Returns
    (ok, info). Info carries the salvage branch + the HIGH-page facts.

    The losslessness invariant (not the sequence): every commit reachable
    before the operation is reachable after, and the tree ends at
    main/0/0/clean.

    C9 (pre-op ref hygiene, git 2.43 semantics live-verified 2026-09-02): a
    crashed prior op can leave a stale `deploy-pull-salvage-tmp`, and a
    resolved prior episode can leave the LOCAL `lapis/<target_id>/salvage`
    ref (the remote side is deleted on merge, the local side never is).
    A pre-existing dst ref is handled explicitly: force-updated when it
    carries no commits unique to it, else the op aborts (the entry stays
    open with dispatch_error set - a named state, never a silent retry
    loop).

    The park step (reviewer fix, cycle-1 med-2) uses `update-ref` as the
    converge primitive, NOT `branch -m <src> <dst>`: an existing-dst
    rename is refused (rc=128, live-verified on git 2.43 - refused even
    when both refs are at the same commit), so the pre-op converge must
    keep the dst ref present and the park step force-updates it to the
    parked commit (temp-commit path: the temp ref's commit; non-temp
    paths: HEAD) and deletes the source ref (temp ref / dead branch).
    The delete-then-recreate alternative could leave the salvage ref
    MISSING on the non-temp (diverged) path, which would break the
    losslessness verify (the post-op rev-list would not include the
    salvage ref) and the PR push.
    """
    cls = entry.get("class")
    salvage_ref = f"lapis/{target_id}/salvage"
    info = {"salvage_branch": salvage_ref, "temp_commit": False, "target_id": target_id}
    try:
        # pre-op ref hygiene (C9) - before any branch surgery.
        dst = _git(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{salvage_ref}")
        if dst.returncode == 0:
            # A local salvage ref from a resolved episode. Converge only
            # when it carries no commits unique to it (everything on it is
            # already reachable from HEAD - `rev-list <ref>..HEAD` counts
            # commits reachable from HEAD but not from <ref>) - otherwise
            # abort (named state).
            uniq = _git(path, "rev-list", "--count", f"HEAD..{salvage_ref}")
            if uniq.returncode != 0 or (uniq.stdout or "").strip() != "0":
                entry["dispatch_error"] = "salvage-ref-collision"
                return False, info
            # Converge the dst ref to HEAD WITHOUT deleting it (reviewer
            # fix, cycle-1 med-2): the delete-then-recreate sequence could
            # leave the salvage ref MISSING on the non-temp (diverged)
            # path, which would break the losslessness verify and the PR
            # push. A non-destructive `update-ref` converge keeps the ref
            # present at HEAD; the park step below force-updates it to
            # the parked commit (see the docstring).
            r = _git(path, "update-ref", f"refs/heads/{salvage_ref}", "HEAD")
            if r.returncode != 0:
                entry["dispatch_error"] = "salvage-ref-collision"
                return False, info
        # a stale temp ref from a crashed prior op: `switch -c` onto an
        # existing ref FAILS (rc=128) - a same-name `branch -m` is a no-op
        # (rc=0), so the post-rename crash window converges, but the op
        # itself must converge or it wedges D4 into a silent per-cycle
        # retry. Converge: delete the stale temp ref ONLY when it carries
        # no commits unique to ITSELF (a crashed mid-op ref at HEAD is the
        # normal shape); otherwise abort (named state - the unique commits
        # are preserved, the entry stays open). Direction matters (C9
        # crash shape, spec I6 losslessness): `HEAD..{tmp}` counts the
        # commits reachable from the tmp ref but NOT from HEAD - those are
        # the commits unique to the tmp ref. A tmp ref one commit ahead of
        # HEAD (a crashed op that committed before dying) must ABORT, not
        # be `branch -D`'d: deleting it would strand that commit and break
        # the losslessness invariant (every pre-op reachable commit
        # reachable post-op). The at-HEAD shape (0 unique commits) is the
        # one that converges.
        tmp_probe = _git(path, "rev-parse", "--verify", "--quiet",
                         f"refs/heads/{_SALVAGE_TMP_REF}")
        if tmp_probe.returncode == 0:
            uniq = _git(path, "rev-list", "--count", f"HEAD..{_SALVAGE_TMP_REF}")
            if uniq.returncode != 0 or (uniq.stdout or "").strip() != "0":
                entry["dispatch_error"] = "salvage-ref-collision"
                return False, info
            r = _git(path, "branch", "-D", _SALVAGE_TMP_REF)
            if r.returncode != 0:
                entry["dispatch_error"] = "salvage-ref-collision"
                return False, info

        # capture the pre-op reachable-commit set (the losslessness check).
        pre = _git(path, "rev-list", "HEAD")
        if pre.returncode != 0:
            return False, info
        pre_commits = set((pre.stdout or "").split())

        temp_commit = False
        if cls == CLASS_STALE_DIRTY:
            r = _git(path, "add", "-A")
            if r.returncode != 0:
                return False, info
            # non-empty cache check: is there anything staged now?
            r = _git(path, "diff", "--cached", "--name-only")
            if r.returncode != 0 or not (r.stdout or "").strip():
                # nothing to commit (e.g. all untracked) - fall through to
                # the diverged shape.
                pass
            else:
                # move to the temporary local ref FIRST - never commit
                # directly on main (Facets gate condition 2).
                r = _git(path, "switch", "-c", _SALVAGE_TMP_REF)
                if r.returncode != 0:
                    return False, info
                ts = _now_iso()
                r = _git(path, "commit", "-m",
                         f"salvage({path}): local work parked by deploy-pull selfheal {ts}")
                if r.returncode != 0:
                    return False, info
                temp_commit = True
                info["temp_commit"] = True

        # park the local work under the canonical salvage ref.
        branch_res = _git(path, "branch", "--show-current")
        current = branch_res.stdout.strip()
        if temp_commit:
            # The temp ref carries the new commit; the salvage ref
            # (possibly the C9-converged one at HEAD) is force-updated to
            # the temp ref's commit and the temp ref deleted. A
            # `branch -m <tmp> <dst>` onto an EXISTING dst fails (rc=128 -
            # live-verified on git 2.43: an existing-dst rename is refused
            # even when both refs are at the same commit), so the
            # update-ref is the converge primitive, not the rename.
            r = _git(path, "update-ref", f"refs/heads/{salvage_ref}",
                     f"refs/heads/{_SALVAGE_TMP_REF}")
            if r.returncode != 0:
                return False, info
        elif current != "main":
            # diverged: the unique local commit is on `current` (== HEAD).
            # The salvage ref (possibly the C9-converged one at HEAD) is
            # force-updated to HEAD and the dead branch deleted. A
            # `branch -m <current> <dst>` onto an existing dst fails
            # (rc=128), so the update-ref is the converge primitive here
            # too (reviewer fix, cycle-1 med-2: the delete-then-recreate
            # sequence could leave the ref missing on this path and break
            # the losslessness verify + PR push).
            r = _git(path, "update-ref", f"refs/heads/{salvage_ref}", "HEAD")
            if r.returncode != 0:
                return False, info
            r = _git(path, "branch", "-D", current)
            if r.returncode != 0:
                return False, info
        else:
            # stale-dirty-on-clean-main with nothing to commit: nothing to
            # park; the salvage ref (converged or fresh) sits at HEAD.
            r = _git(path, "update-ref", f"refs/heads/{salvage_ref}", "HEAD")
            if r.returncode != 0:
                return False, info

        # restore: switch main FIRST (the temp ref cannot be deleted while
        # it is the checked-out branch - `branch -D` refuses, rc=128; the
        # commit is safe on the salvage ref and main was never dirtied by
        # it), then delete the temp ref, reset --hard origin/main when the
        # temp-ref commit was made, and ff-only pull.
        r = _git(path, "switch", "main")
        if r.returncode != 0:
            return False, info
        if temp_commit:
            r = _git(path, "branch", "-D", _SALVAGE_TMP_REF)
            if r.returncode != 0:
                return False, info
        if temp_commit:
            r = _git(path, "reset", "--hard", "origin/main")
            if r.returncode != 0:
                return False, info
        r = _git(path, "pull", "--ff-only", "origin", "main")
        if r.returncode != 0:
            return False, info

        # losslessness verify: every pre-op reachable commit is reachable
        # after, AND the tree ends main/0/0/clean.
        post = _git(path, "rev-list", "HEAD", salvage_ref)
        if post.returncode != 0:
            return False, info
        post_commits = set((post.stdout or "").split())
        if not pre_commits.issubset(post_commits):
            missing = pre_commits - post_commits
            logger.warning(
                "[deploy-pull-selfheal] losslessness check failed at %s: "
                "%d pre-op commit(s) unreachable (e.g. %s)",
                path, len(missing), next(iter(missing)) if missing else "?",
            )
            return False, info
        ok, _ev = _verify_healthy(path)
        if not ok:
            # M2 (named behavior): a tree that is ahead AND dirty classifies
            # `diverged` (the classifier checks ahead > 0 before the dirty
            # check) and this path never commits dirty files (the temp commit
            # is stale_dirty-only) - so the first salvage attempt of a
            # dirty+diverged tree fails its own verify HERE: no work lost, no
            # PR opened, the entry stays open, and the next cycle
            # re-classifies it `stale_dirty` and completes.
            return False, info
        # C8: the losslessness verify is durable, not just in-memory - the
        # pre/post reachable-set counts ride on `info` so the deploy-log
        # line and the PR body record them (I6 post-hoc auditability).
        info["losslessness"] = (
            f"salvage verified: {len(pre_commits)}/{len(pre_commits)} "
            f"pre-op commits reachable; tree at main 0/0"
        )
        return True, info
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("[deploy-pull-selfheal] D4 salvage errored at %s: %s", path, exc)
        return False, info


def _push_and_open_pr(path: str, repo: str, entry: dict, info: dict) -> int | None:
    """D4.3: push the salvage branch + forgejo.create_pr. Returns the PR
    number, or None on any failure (the entry stays open; the open-PR guard
    makes a retry idempotent)."""
    salvage_ref = info["salvage_branch"]
    try:
        r = _git(path, "push", "origin", salvage_ref)
        if r.returncode != 0:
            logger.warning("[deploy-pull-selfheal] push %s failed: %s",
                           salvage_ref, (r.stderr or "").strip()[:200])
            return None
        from agents_core.forgejo import create_pr
        date = _now_iso()[:10]
        # L3: the ledger entry never carries a `porcelain` key (the head's
        # `entry.get("porcelain")` was a dead access) - the dump is the
        # class + path plus the C8 losslessness record.
        state_dump = "\n".join(
            [
                f"class={entry.get('class')} path={path}",
                info.get("losslessness", ""),
            ]
        ).strip()
        body = (
            f"{state_dump}\n\n"
            "Opened automatically by deploy-pull selfheal "
            "(deploy-pull-selfheal-machine-dispatch-v0). Local work was "
            "committed (not discarded) to this salvage branch and the tree "
            "was restored to main. Ratify to land, close to discard."
        )
        result = create_pr(
            repo=repo,
            title=f"salvage: local work from {path} ({date})",
            head=salvage_ref,
            base="main",
            body=body,
        )
        pr_number = result.get("number") if isinstance(result, dict) else None
        return int(pr_number) if pr_number is not None else None
    except Exception as exc:
        logger.warning("[deploy-pull-selfheal] create_pr failed for %s: %s", path, exc)
        return None


def _supersede_gem(gem_id: str, reason: str) -> None:
    try:
        from . import brief_gem as _bg
        _bg._call_supersede_endpoint(gem_id, reason=reason, by=_DEPOSITED_BY)
    except Exception as exc:
        logger.warning("[deploy-pull-selfheal] supersede failed for %s: %s", gem_id, exc)


def _pr_state(repo: str, pr_number: int) -> str | None:
    """Forgejo PR state: 'open' | 'merged' | 'closed' | None (unknown)."""
    try:
        from agents_core.forgejo import get_pr
        repo_name = repo.split("/")[-1]
        owner = repo.split("/", 1)[0] if "/" in repo else None
        pr = get_pr(repo_name, pr_number, owner=owner)
        if not pr:
            return None
        state = pr.get("state")
        if state == "merged":
            return "merged"
        if state in ("closed", "open"):
            return state
        return None
    except Exception as exc:
        logger.warning("[deploy-pull-selfheal] get_pr failed for %s#%s: %s",
                       repo, pr_number, exc)
        return None


def _rejection_count(target_id: str) -> int:
    """Count fixer rejections in mem pm/dispatched/<tid> (the D5 named
    signal (b): two rejections = a stalled review path)."""
    try:
        from agents_core.mem import MemoryStore
        mem = MemoryStore()
        raw = mem.get(f"pm/dispatched/{target_id}")
        if not raw:
            return 0
        data = raw.get("content") if isinstance(raw, dict) else raw
        records = json.loads(data) if isinstance(data, str) else data
    except Exception:
        return 0
    count = 0
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        if rec.get("agent_type") in ("fixer", "fixer_retry") and rec.get("verdict") == "rejected":
            count += 1
    return count


def _target_paused_review_gate(target_id: str) -> bool:
    """D5 named signal (b): the target's paused state with the review-gate
    reason."""
    try:
        from agents_core.mem import MemoryStore
        mem = MemoryStore()
        raw = mem.get(f"pm/pause-state/{target_id}")
        if not raw:
            return False
        data = raw.get("content") if isinstance(raw, dict) else raw
        if isinstance(data, str):
            return data == "paused"
        return bool(data) and data.get("state") == "paused"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# D3/D4/D5/D6 - the pass (called from _post_land_git_pull's failure branch,
# before the generic notify). Best-effort: never raises (I7).
# ---------------------------------------------------------------------------

def run_pass(repo: str, path: str, trigger: str, *, now: float | None = None) -> list[str]:
    """One selfheal pass for one stuck path. Returns action strings for
    logging. Never raises (I7)."""
    actions: list[str] = []
    try:
        cls, evidence = classify_pull_failure(path, trigger, now=now)
        actions.append(f"classified:{cls}")

        # D2 - safe-zone self-repair (machine lane, both triggers).
        if cls in _SAFE_CLASSES:
            if self_repair(path, evidence, trigger):
                actions.append("self_repaired")
                return actions
            # fall through to D3 as unknown (D2 spec: on any step 1-3
            # failure or verify failure, fall through with class unknown).
            cls = CLASS_UNKNOWN
            evidence = {**evidence, "error": "self-repair verify failed"}
            actions.append("self_repair_failed:unknown")

        # D3/D4/D5/D6 - under the ledger flock (read-decide-deposit-write,
        # fail-closed on contention).
        try:
            with file_lock(_LEDGER_LOCK_FILE, _LEDGER_LOCK_TIMEOUT_S):
                actions.extend(_run_pass_locked(repo, path, cls, evidence, trigger, now=now))
        except FileLockTimeout:
            logger.warning(
                "[deploy-pull-selfheal] ledger lock contended (path=%s); skipping "
                "pass, depositing nothing (fail-closed)", path,
            )
            actions.append("skip:ledger-lock-contended")
    except Exception as exc:  # noqa: BLE001 - I7: best-effort, never raises
        logger.warning("[deploy-pull-selfheal] pass failed for %s: %s", path, exc)
        actions.append(f"error:{type(exc).__name__}")
    return actions


def _run_pass_locked(repo: str, path: str, cls: str, evidence: dict,
                     trigger: str, *, now: float | None = None) -> list[str]:
    """Body of run_pass, under the ledger flock."""
    actions: list[str] = []
    now_ts = _now_iso()
    ledger = read_ledger()
    decisions = _gem_decisions()
    reconcile_changed_gem_ids = _reconcile_terminal(ledger, decisions)

    signature = compute_signature(path, cls)
    entry = ledger.get(signature)

    # I10: an ack_watch decision put this entry into "acked" (reconcile above) -
    # the machine action is suppressed while acked. No bump, no deposit, no
    # page; the pass holds. (Deploy-pull-selfheal-core-v0 Slice 1: acked is a
    # terminal hold here; D4's salvage-window bypass returns in Slice 2.)
    if entry is not None and entry.get("status") == "acked":
        actions.append("hold:acked")
        write_ledger(ledger)
        return actions

    # A just-reconciled terminal entry (dismissed_fp / resolved set THIS pass
    # by a gem decision) is not re-opened or re-deposited - the operator just
    # acted (I10). Only a dismissed_fp/resolved carried from a PRIOR pass (a
    # fresh episode) re-opens.
    if entry is not None and entry.get("gem_id") in reconcile_changed_gem_ids:
        actions.append(f"hold:{entry.get('status')}:reconciled")
        write_ledger(ledger)
        return actions

    # D3 - first detection / recurrence / bump.
    if cls == CLASS_ACTIVE_DIRTY:
        # I3: pure defer - no action, no page, no ledger state.
        write_ledger(ledger)
        actions.append("defer:active_dirty")
        return actions

    if cls in _SALVAGE_CLASSES or cls in (CLASS_FETCH_FAILED, CLASS_UNKNOWN):
        # A clean tree classifies `unknown` (no predicate matches - 0/0,
        # branch=main, no dirty porcelain). That is NOT a stuck condition:
        # do not deposit a gem (D4.5 below closes any OPEN entry for this path
        # in place - the tree already healed). Only a genuinely unclassifiable
        # tree (evidence of an anomaly) is a first-detection deposit.
        if cls == CLASS_UNKNOWN and not _evidence_has_anomaly(evidence):
            actions.append("clean:unknown")
        elif entry is None or entry.get("status") in ("dismissed_fp", "resolved"):
            base = entry if entry is not None else _new_entry(path, repo, cls, evidence, now_ts)
            action = _first_detection(ledger, signature, base, repo=repo,
                                      evidence=evidence, now=now_ts,
                                      recurred=entry is not None)
            actions.append(action)
        else:
            # recurring cycle with a live entry: bump last_seen/times_seen
            # only. No gem, no page (I2 - only state transitions page).
            entry["last_seen"] = now_ts
            entry["times_seen"] = entry.get("times_seen", 1) + 1
            actions.append("bump:open")

    # D6 - fetch_failed: no machine action; one HIGH page at 20 minutes
    # (FETCH_FAILED_PAGE_AFTER_S), then silence (state-transition rule, I2).
    # The gem + station incident were recorded at first detection (D3/D7).
    if cls == CLASS_FETCH_FAILED and entry is not None:
        if not entry.get("fetch_page_sent"):
            try:
                age = time.time() - _parse_iso(entry["first_seen"]).timestamp()
            except (ValueError, TypeError):
                age = 0
            if age >= FETCH_FAILED_PAGE_AFTER_S:
                _send_page(
                    message=(
                        f"{repo}: deploy pull fetch failed - credentials/network "
                        f"suspected (gem {entry.get('gem_id')})"
                    ),
                    title=f"{repo}: deploy pull fetch failed",
                    priority=_high_priority(),
                )
                entry["fetch_page_sent"] = True
                actions.append("paged:fetch_failed")

    # D4 - windowed salvage for open stale_dirty/diverged entries (the
    # salvage_now gem option bypasses the window).
    if cls in _SALVAGE_CLASSES and entry is not None and entry.get("status") == "open":
        if _window_elapsed(entry, now=now) or _salvage_now_requested(entry, decisions):
            actions.extend(_salvage_entry(ledger, entry, repo, path, evidence, trigger))

    # D4.5 - close the loop: cheap local verify on the success path for
    # open/salvaged entries.
    healthy, _ev = _verify_healthy(path)
    if healthy:
        for e in ledger.values():
            if e.get("path") != path:
                continue
            if e.get("status") == "open":
                # the human or another actor fixed it.
                e["status"] = "resolved"
                actions.append(f"resolved:healthy:{e.get('signature', '?')[:8]}")
                if e.get("gem_id"):
                    _supersede_gem(e["gem_id"], "tree verified healthy (close-the-loop)")
            elif e.get("status") == "salvaged":
                pr = e.get("pr_number")
                if pr:
                    state = _pr_state(repo, pr)
                    if state in ("merged", "closed"):
                        e["status"] = "resolved"
                        actions.append(f"resolved:salvaged-{state}:{e.get('signature', '?')[:8]}")
                        if e.get("gem_id"):
                            _supersede_gem(e["gem_id"], f"salvage PR #{pr} {state}; tree healthy")

    # D5 - PR-path-stall hold (this pass is the named observer). A stalled
    # salvage PR becomes `worker_failed` with exactly one HIGH page. Named
    # signal preconditions (M3): signal (a) closed-unmerged is a live Forgejo
    # read; signals (b) two-rejections and (c) review-gate-cap pause read
    # pm/dispatched/<tid> / pm/pause-state/<tid> state that exists only if
    # the deploy-repair-* target is actually dispatched/paused - if the
    # advisory salvage PR is never dispatched, D5 effectively fires only on
    # (a). All three reads are error-safe in the conservative direction
    # (read error -> not stalled, no false worker_failed).
    for e in ledger.values():
        if e.get("path") != path or e.get("status") != "salvaged":
            continue
        pr = e.get("pr_number")
        if not pr:
            continue
        target_id = e.get("target_id") or ""
        state = _pr_state(repo, pr)
        stalled = state == "closed"  # (a) closed-unmerged
        if not stalled and target_id:
            stalled = _rejection_count(target_id) >= 2  # (b) two rejections
        if not stalled and target_id:
            stalled = _target_paused_review_gate(target_id)  # (c) gate-cap pause
        if stalled:
            e["status"] = "worker_failed"
            # L4: the gem id rides in the page text (ledger-only provenance
            # is the named accepted state, I5).
            _send_page(
                message=(
                    f"{repo}: deploy repair stalled - holding for PM "
                    f"(target {target_id or '?'}, PR #{pr}, "
                    f"gem {e.get('gem_id') or '?'})"
                ),
                title=f"{repo}: deploy repair stalled",
                priority=_high_priority(),
            )
            actions.append("worker_failed")

    write_ledger(ledger)
    return actions


def close_ledger_entries_for_path(path: str, *, reason: str = "") -> None:
    """lapis-pm-daemon-silent-gaps-v0 B4b ledger closure: mark every open
    ledger entry for `path` as `closed` (NOT deleted) so a subsequent
    re-lock of the same path is a fresh state transition that pages,
    instead of being suppressed by the "already-paged" dedup (the silent
    re-lock trap). Called on a B4b self-clear; the manual escape hatch
    (_clear_deploy_pull_lock) stays as-is per the spec. Best-effort:
    never raises."""
    now = _now_iso()
    try:
        with file_lock(_LEDGER_LOCK_FILE, _LEDGER_LOCK_TIMEOUT_S):
            ledger = read_ledger()
            changed = False
            for entry in ledger.values():
                if entry.get("path") != path:
                    continue
                if entry.get("status") != "open":
                    continue
                entry["status"] = "closed"
                entry["closed_at"] = now
                if reason:
                    entry["closed_reason"] = reason
                changed = True
            if changed:
                write_ledger(ledger)
    except Exception as exc:  # noqa: BLE001 - best-effort, never raises
        logger.warning("[deploy-pull-selfheal] close_ledger_entries_for_path "
                       "failed for %s: %s", path, exc)


def notify_locked_clone_stuck(repo: str, path: str, cls: str,
                              trigger: str) -> None:
    """lapis-pm-daemon-silent-gaps-v0 B4b still-stuck page: exactly one
    NORMAL page per (path, class) state transition, deduped on the D3
    ledger signature (compute_signature = sha256(path|class)). A stable
    class stuck for days pages exactly once (subsequent cycles are silent
    `bump:open` - the 2026-09-02 370-page storm must not recur). Best-
    effort: never raises."""
    try:
        with file_lock(_LEDGER_LOCK_FILE, _LEDGER_LOCK_TIMEOUT_S):
            ledger = read_ledger()
            signature = compute_signature(path, cls)
            entry = ledger.get(signature)
            if entry is not None and entry.get("status") == "open":
                # Stable class: bump only - no page (I2 - only state
                # transitions page).
                entry["last_seen"] = _now_iso()
                entry["times_seen"] = entry.get("times_seen", 1) + 1
                write_ledger(ledger)
                return
            # First detection (or a class transition, or a RE-LOCK after a
            # B4b self-clear closed the entry) - page once: a closed entry
            # re-opening is a fresh state transition, not suppressed by
            # dedup (the silent re-lock trap).
            now = _now_iso()
            base = entry if entry is not None else _new_entry(
                path, repo, cls, {}, now,
            )
            base["status"] = "open"
            base["last_seen"] = now
            base["times_seen"] = 1
            ledger[signature] = base
            write_ledger(ledger)
        _send_page(
            message=(
                f"{repo}: deploy clone {path} is LOCKED and still stuck "
                f"(class={cls}); the lock is kept and the pull is skipped. "
                f"Manual intervention required (clear the lock once the "
                f"divergence is resolved)."
            ),
            title=f"{repo}: deploy pull lock still stuck ({cls})",
            priority=_normal_priority(),
        )
    except Exception as exc:  # noqa: BLE001 - best-effort, never raises
        logger.warning("[deploy-pull-selfheal] notify_locked_clone_stuck "
                       "failed for %s: %s", path, exc)


def _normal_priority():
    from agents_core.notify import Priority
    return Priority.NORMAL


def _salvage_entry(ledger: dict, entry: dict, repo: str, path: str,
                   evidence: dict, trigger: str) -> list[str]:
    """D4.1-D4.4 for one open entry meeting the window condition."""
    actions: list[str] = []
    signature = entry.get("signature") or compute_signature(path, entry.get("class", "?"))
    # no-op guard: a salvaged entry is never re-salvaged (status gate above).
    target_id = _ensure_target(repo, signature)
    if target_id is None:
        entry["dispatch_error"] = "target-ensure-failed"
        actions.append("salvage:target-failed")
        return actions
    entry["target_id"] = target_id

    # no-op guard: an existing open PR on the salvage branch skips push/create
    # (a mid-pass kill cannot double-open a PR).
    if _open_salvage_pr_exists(repo, target_id) is True:
        actions.append("salvage:pr-exists")
        return actions

    ok, info = _salvage(path, repo, entry, target_id)
    if not ok:
        # _salvage may have set a more specific dispatch_error (e.g.
        # `salvage-ref-collision` for a pre-existing ref carrying unique
        # commits - the C9 named state); preserve it. The generic
        # `salvage-failed` is the fallback for any other failure.
        entry["dispatch_error"] = entry.get("dispatch_error") or "salvage-failed"
        entry["attempts"] = entry.get("attempts", 0) + 1
        actions.append("salvage:failed")
        return actions

    entry["salvage_branch"] = info["salvage_branch"]
    pr_number = _push_and_open_pr(path, repo, entry, info)
    if pr_number is None:
        entry["dispatch_error"] = "push-or-pr-failed"
        entry["attempts"] = entry.get("attempts", 0) + 1
        actions.append("salvage:push-pr-failed")
        return actions

    entry["pr_number"] = pr_number
    entry["status"] = "salvaged"
    entry["attempts"] = entry.get("attempts", 0) + 1
    # D4.4: the one HIGH page - command-shaped, not a report. The only
    # routine page in the spec (the land-or-discard judgment gate, I2).
    _send_page(
        message=(
            f"{repo}: deploy pull stuck - machine salvaged local work to "
            f"PR #{pr_number} (target {target_id}, gem {entry.get('gem_id')}); "
            f"tree restored to main. Ratify to land, close to discard."
        ),
        title=f"{repo}: deploy pull salvaged",
        priority=_high_priority(),
    )
    # C8: the losslessness record (pre/post reachable-set counts) is durable
    # in the deploy-log line, not just in-memory.
    _deploy_log_line(
        path,
        f"salvaged to PR #{pr_number} ({target_id}); "
        f"{info.get('losslessness', 'salvage verified')} | {trigger}",
    )
    actions.append(f"salvaged:pr={pr_number}")
    return actions


def _high_priority():
    from agents_core.notify import Priority
    return Priority.HIGH


def close_out_sweep(repo: str, path: str) -> list[str]:
    """OQ-4 option-3 (Erah-approved 2026-09-02, Mirror-Council architecture):
    the lightweight Close-Out Sweep the seam runs on a SUCCESSFUL pull
    (`pull_rc == 0`). Bounded + idempotent: a path-scoped ledger read (no
    ledger-wide scan), at most one Forgejo PR-state read per salvage entry
    (5s timeout), and only the two named transitions - `salvaged` ->
    `resolved` (the salvage PR merged/closed) and `salvaged` ->
    `worker_failed` (the PR path stalled). No salvage surgery on success,
    well inside the 120s unit budget. Never raises (I7)."""
    actions: list[str] = []
    try:
        with file_lock(_LEDGER_LOCK_FILE, _LEDGER_LOCK_TIMEOUT_S):
            ledger = read_ledger()
            changed = False
            for e in ledger.values():
                if e.get("path") != path or e.get("status") != "salvaged":
                    continue
                pr = e.get("pr_number")
                if not pr:
                    continue
                state = _pr_state(repo, pr)
                if state in ("merged", "closed"):
                    # PR merged -> resolved. PR closed WITHOUT merging
                    # (the operator discarded the salvage) -> worker_failed:
                    # the PR path stalled and the salvage work is parked on
                    # the salvage branch, not landed. (This is the D5
                    # signal (a) - closed-unmerged - and it is NOT a
                    # resolved outcome.)
                    if state == "merged":
                        e["status"] = "resolved"
                        changed = True
                        actions.append(
                            f"resolved:salvaged-{state}:{e.get('signature', '?')[:8]}"
                        )
                        if e.get("gem_id"):
                            _supersede_gem(
                                e["gem_id"],
                                f"salvage PR #{pr} {state} (close-out sweep)",
                            )
                    else:
                        e["status"] = "worker_failed"
                        changed = True
                        # I2 (exactly one HIGH page per transition): the
                        # close-out sweep is the named D5 observer on the
                        # SUCCESS path (the failure path's D5 block only
                        # runs when a pull fails) - the `salvaged` ->
                        # `worker_failed` transition fires the same one
                        # HIGH page the D5 block fires (the page is
                        # transition-gated: a worker_failed entry is never
                        # re-processed by the sweep, so no re-page).
                        # L4: the gem id rides in the page text.
                        _send_page(
                            message=(
                                f"{repo}: deploy repair stalled - holding for PM "
                                f"(target {e.get('target_id') or '?'}, PR #{pr}, "
                                f"gem {e.get('gem_id') or '?'})"
                            ),
                            title=f"{repo}: deploy repair stalled",
                            priority=_high_priority(),
                        )
                        actions.append("worker_failed:close-out")
                else:
                    target_id = e.get("target_id") or ""
                    stalled = False
                    if target_id:
                        stalled = _rejection_count(target_id) >= 2
                    if not stalled and target_id:
                        stalled = _target_paused_review_gate(target_id)
                    if stalled:
                        e["status"] = "worker_failed"
                        changed = True
                        _send_page(
                            message=(
                                f"{repo}: deploy repair stalled - holding for PM "
                                f"(target {target_id or '?'}, PR #{pr}, "
                                f"gem {e.get('gem_id') or '?'})"
                            ),
                            title=f"{repo}: deploy repair stalled",
                            priority=_high_priority(),
                        )
                        actions.append("worker_failed:close-out")
            if changed:
                write_ledger(ledger)
    except Exception as exc:  # noqa: BLE001 - I7: best-effort, never raises
        logger.warning("[deploy-pull-selfheal] close-out sweep failed for %s: %s",
                       path, exc)
    return actions


def pass_handled_failure(repo: str, path: str, trigger: str, *,
                         pull_rc: int, pull_stderr: str = "") -> bool:
    """Wiring for _post_land_git_pull (OQ-4 option-3: called on EVERY pull -
    success AND failure - with the real `pull_rc`; the machine branches
    internally).

    `pull_rc == 0`: the lightweight Close-Out Sweep (bounded, idempotent) -
    closes the `salvaged` -> `resolved` / `worker_failed` loop the parent's
    D4 step-5 success-path promise named. `pull_rc != 0`: the heavy D1-D7
    salvage/recovery path.

    Returns True when the selfheal pass owns this outcome (the caller skips
    the legacy generic notify): a self-repair succeeded, the ledger
    recorded/acted on the condition, or the pass is holding a state it owns
    (C5). Returns False when the pass could not classify the failure (the
    legacy alert remains the fallback - I7). Never raises (I7).

    C5 (SECURITY fix, Erah-reclassified 2026-09-02): the `hold:` and `clean:`
    action prefixes are in the True set - the pass owns those states. The
    Slice-1 True set omitted them, so for a stuck tree in `hold:acked` /
    `clean:unknown` the pass returned False and pm_core's legacy generic
    notify fired EVERY cycle - a NORMAL Pushover carrying unredacted
    `stderr[:300]` (on a `fetch_failed` root cause git echoes the tokenized
    remote URL; both mapped deploy trees embed the Forgejo token in the URL)
    into Pushover and the no-auth audit store
    /srv/lapis/notify-audit/captured.jsonl (`send_notification` has no
    message-level dedup) - a credential leak, not a page-budget nuisance.
    Named known-leak exceptions (same treatment as the Slice-1 journal
    line): the pm_core journal line (pm_core.py:1072, Slice-1 fix #6) and
    the legacy page's `stderr[:300]` (pm_core.py:1206) remain unredacted.
    """
    try:
        if pull_rc == 0:
            actions = close_out_sweep(repo, path)
        else:
            actions = run_pass(repo, path, trigger)
    except Exception as exc:  # noqa: BLE001 - I7
        logger.warning("[deploy-pull-selfheal] pass_handled_failure errored: %s", exc)
        return False
    if "self_repaired" in actions:
        return True
    # The pass owns the outcome when it recorded or acted on the condition.
    for action in actions:
        if action.startswith(("deposited:", "bump:", "salvage", "salvaged:",
                              "paged:", "worker_failed", "defer:",
                              "resolved:", "hold:", "clean:",
                              "skip:ledger-lock-contended",
                              "skip:deposit-failed")):
            return True
    # classified:unknown with no ledger action - the legacy alert covers it.
    return False


# ---------------------------------------------------------------------------
# D7 - seams (named, read-only, no API coupling). force_dispatch
# (pm_core.py:3068) is the documented LLM-dispatch seam; it is never called
# from this module (the fixer's current surface is read-only git +
# per-task worktree cwd - it cannot execute host-tree ops).
# ---------------------------------------------------------------------------

def read_seams() -> dict:
    """The D7 read-only seams the brix-repair-station brain (when it lands)
    can read: the ledger, the station db path, the target family, the gems,
    the salvage PRs. No behavior depends on any of these being read."""
    try:
        ledger = read_ledger()
    except Exception:
        ledger = {}
    return {
        "ledger": ledger,
        "ledger_path": str(_LEDGER_FILE),
        "station_db": "/srv/lapis/repair-station/repair_station.db",
        "target_family": "deploy-repair-*",
        "salvage_branch_family": "lapis/deploy-repair-*/salvage",
    }
