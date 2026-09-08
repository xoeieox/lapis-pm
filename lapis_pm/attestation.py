"""Deploy-manifest completeness attestation.

Target: `night-deploy-manifest-attestation-v0` (spec rev 4, Erah-cleared
2026-09-07 — Option A: land is NEVER refused; loud same-window node + land
reporting + expiring waivers + historical ledger; no terrain-map consumer
hold).

The tractable invariant (spec D1): **every python file named by a scheduled
node's command must appear in the manifest that deploys it, or be explicitly
waived.** Correctness of a deployed script stays with the test suite — this
module attests completeness, not correctness.

Design pins (spec D1-D7, I1-I7):

- The checker is a module of THIS repo, invoked as a module from the
  /srv/git/lapis-pm deploy clone (freshness via `_POST_LAND_PULL["lapis-pm"]`)
  — it needs no manifest entry; what it audits does.
- It parses ONLY the narrow, stable subset of the plan file (node `id`,
  `command`, `lane`) and ends `held` with reason `plan_schema_drift` if an
  unrecognized top-level key appears (D1). It never couples to the
  conductor's internal parser.
- Waiver file `{script, reason, added, owner, expires}` (D2): a waived script
  that later drifts is re-reported; a forever-waiver (missing/absurd
  `expires`) is a bug the node reports.
- Verdicts: `clean` / `held` (drift detected — alarms, deduped — never
  `failed`/`idle`) with structured reasons.
- The sibling artifact `/data/slots/deploy-attestation.json` is
  append-ledger-shaped (not a snapshot) and is written atomically (I7).
- Per-host attestation (D6): the deploy clone, /data/agents/scripts/, and
  GW /usr/local/sbin|bin via bounded never-wake ssh (list argv, never
  raises, None on failure).
- No node may declare `after: [night-deploy-attestation]` — validation
  rejects a plan that does (DoD-6, I1/I3).

The checker NEVER deploys (I6): it reports; deploy is land's job.
"""

from __future__ import annotations

import filecmp
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths — module-level constants (not inlined) so tests can patch them to
# tmp paths, the same pattern the post-land deploy suite uses for
# _CONDUCTOR_SCRIPTS_SRC/_DEST and _DEPLOY_LOG.
# ---------------------------------------------------------------------------

#: The night plan file (host declaration, outside git).
NIGHT_PLAN_PATH = Path(os.environ.get("LAPIS_PM_NIGHT_PLAN", "/data/slots/night-plan.yaml"))

#: The waiver file (host declaration, outside git; seeded by landing ops).
WAIVER_FILE = Path(os.environ.get("LAPIS_PM_WAIVER_FILE", "/data/slots/deploy-manifest-waivers.yaml"))

#: The sibling attestation artifact — append-ledger-shaped, atomic writes (I7).
ATTESTATION_LEDGER_PATH = Path(
    os.environ.get("LAPIS_PM_ATTESTATION_LEDGER", "/data/slots/deploy-attestation.json")
)

#: The night-plan run record — its per-node verdict/reason is the channel by
#: which a held attestation node reaches the morning brief + agora intake leg
#: (spec DoD-0/D5 pinned consumer).
RUN_RECORD_PATH = Path(
    os.environ.get("LAPIS_PM_RUN_RECORD", "/data/slots/night-plan-runs/latest.json")
)

#: The deploy clone this module runs from (freshness via _POST_LAND_PULL).
LAPIS_PM_DEPLOY_CLONE = Path(
    os.environ.get("LAPIS_PM_DEPLOY_CLONE", "/srv/git/lapis-pm")
)

#: The conductor source of truth for scripts (same seam pm_core uses).
CONDUCTOR_SCRIPTS_SRC = Path(os.environ.get("LAPIS_PM_CONDUCTOR_SCRIPTS_SRC", "/srv/git/conductor/scripts"))

#: The conductor deploy clone root. A script a node command invokes FROM THIS
#: TREE (e.g. /srv/git/conductor/scripts/clone_currency_sync.py) executes
#: directly from the deploy clone, which _POST_LAND_PULL["conductor"] keeps
#: current — so it is NOT a deploy-manifest gap (the D6 audit surface is the
#: three execution locations: deploy clone, /data/agents/scripts/, GW). Only
#: scripts invoked from a SEPARATE execution location (/data/agents/scripts/)
#: that must be DELIVERED there by a manifest are drift rows.
CONDUCTOR_DEPLOY_CLONE = Path(os.environ.get("LAPIS_PM_CONDUCTOR_DEPLOY_CLONE", "/srv/git/conductor"))

#: The BRIX host-op runtime dir (same seam pm_core uses).
AGENTS_SCRIPTS_DIR = Path(os.environ.get("LAPIS_PM_AGENTS_SCRIPTS", "/data/agents/scripts"))

#: The GW host + ssh timeouts — the never-wake discipline (D6): short-timeout
#: BatchMode ssh probe, never DoormanClient/WoL.
_GW_SSH_HOST = os.environ.get("LAPIS_PM_GW_SSH_HOST", "gravitywell")
_GW_SSH_CONNECT_TIMEOUT_SECS = int(os.environ.get("LAPIS_PM_GW_SSH_CONNECT_TIMEOUT_SECS", "5"))
_GW_SSH_CMD_TIMEOUT_SECS = int(os.environ.get("LAPIS_PM_GW_SSH_CMD_TIMEOUT_SECS", "20"))

#: The DAG node id this attestation runs under (the host-declared plan node).
ATTTESTATION_NODE_ID = "night-deploy-attestation"

#: The alarm-dedup signature (the _GW_HOST_SCRIPT_DRIFT_LEDGER shape, D7).
ALARM_LEDGER_SIGNATURE = "night-deploy-attestation-drift"

#: The alarm-dedup ledger — dedups the Desk-gem deposit so a persisting held
#: pages once per day, not every night (D7, DoD-8).
ALARM_LEDGER_PATH = Path(
    os.environ.get("LAPIS_PM_ALARM_LEDGER", "/srv/lapis/lapis-state/lapis-pm/night-deploy-attestation-drift-ledger.json")
)

#: The alarm cooldown — a persisting held pages once per day (D7: 3600s floor;
#: the persisting-drift page is daily, the cooldown gates re-deposit).
ALARM_COOLDOWN_SECS = int(os.environ.get("LAPIS_PM_ALARM_COOLDOWN_SECS", "3600"))

#: A waiver expiry beyond this horizon is "absurd" — a forever-waiver is a bug
#: the node reports (D2, Q-c: an expiry defeats the ritual).
_WAIVER_MAX_HORIZON_DAYS = 365

#: The narrow top-level keys the plan-file parser accepts (D1). Anything else
#: -> held with reason plan_schema_drift, never a silent pass.
_PLAN_TOP_LEVEL_KEYS = frozenset({"schema_version", "nodes"})

#: The narrow per-node keys the parser accepts (D1).
_PLAN_NODE_KEYS = frozenset(
    {
        "id", "command", "lane", "repo", "gw_phase", "bound", "gate",
        "success_predicate", "edge_policy", "idempotency", "autonomy_tier",
        "producer", "after", "active", "rationale", "gw_resource",
        "compute_target", "compute_targets", "decision_slot", "units",
    }
)

#: Exit codes the DAG node's success_predicate maps (host-declared plan node):
#: 0 -> clean, 2 -> held (drift / schema drift / waiver bug), 3 -> failed
#: (internal error — a detector bug, I2).
EXIT_CLEAN = 0
EXIT_HELD = 2
EXIT_FAILED = 3


# ---------------------------------------------------------------------------
# Verdict vocabulary
# ---------------------------------------------------------------------------

VERDICT_CLEAN = "clean"
VERDICT_HELD = "held"

REASON_NONE = None
REASON_DRIFT = "drift"
REASON_PLAN_SCHEMA_DRIFT = "plan_schema_drift"
REASON_WAIVER_EXPIRED = "waiver_expired"
REASON_WAIVER_FOREVER = "waiver_forever"
REASON_INTERNAL_ERROR = "internal_error"


# ---------------------------------------------------------------------------
# Narrow plan-file parse (D1) — never coupled to the conductor's parser
# ---------------------------------------------------------------------------

@dataclass
class PlanNode:
    """One node's narrow subset: id, command, lane (+ after, for DoD-6)."""
    id: str
    command: str
    lane: str
    after: list[str] = field(default_factory=list)


@dataclass
class PlanParseResult:
    """Narrow plan parse. `schema_drift` is True iff an unrecognized top-level
    key appeared (or the nodes list was malformed) — the caller ends `held`
    with reason `plan_schema_drift`, never a silent pass."""
    nodes: list[PlanNode]
    schema_drift: bool
    schema_drift_detail: str = ""


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def parse_plan_narrow(plan_text: str) -> PlanParseResult:
    """Parse ONLY the narrow stable subset of the plan file (node id/command/
    lane, plus `after` for the DoD-6 invariant).

    An unrecognized TOP-LEVEL key -> schema_drift (the plan file's schema is a
    moving target across conductor PRs; a checker that re-implements the
    conductor's full parser drifts and erodes trust — spec D1 / Q-d).
    Per-node: only `id`, `command`, `lane`, `after` are consumed; other
    per-node keys are tolerated (they are not the schema this checker owns).
    """
    try:
        doc = yaml.safe_load(plan_text)
    except yaml.YAMLError:
        return PlanParseResult(nodes=[], schema_drift=True,
                               schema_drift_detail="yaml parse error")
    if not isinstance(doc, dict):
        return PlanParseResult(nodes=[], schema_drift=True,
                               schema_drift_detail="top-level is not a mapping")

    unknown = sorted(set(doc.keys()) - _PLAN_TOP_LEVEL_KEYS)
    if unknown:
        return PlanParseResult(
            nodes=[], schema_drift=True,
            schema_drift_detail=f"unrecognized top-level key(s): {unknown}",
        )

    raw_nodes = doc.get("nodes")
    if raw_nodes is None:
        return PlanParseResult(nodes=[], schema_drift=False)
    if not isinstance(raw_nodes, list):
        return PlanParseResult(nodes=[], schema_drift=True,
                               schema_drift_detail="nodes is not a list")

    nodes: list[PlanNode] = []
    for i, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            return PlanParseResult(
                nodes=nodes, schema_drift=True,
                schema_drift_detail=f"node[{i}] is not a mapping",
            )
        nid = raw.get("id")
        if not isinstance(nid, str) or not nid:
            return PlanParseResult(
                nodes=nodes, schema_drift=True,
                schema_drift_detail=f"node[{i}] missing/invalid id",
            )
        command = raw.get("command") or ""
        if not isinstance(command, str):
            command = str(command)
        lane = raw.get("lane")
        if not isinstance(lane, str):
            lane = ""
        after = raw.get("after") or []
        if isinstance(after, str):
            after = [after]
        if not isinstance(after, list):
            after = []
        nodes.append(PlanNode(id=nid, command=command, lane=lane,
                              after=[str(a) for a in after]))
    return PlanParseResult(nodes=nodes, schema_drift=False)


# ---------------------------------------------------------------------------
# Script resolution from a node command (D1)
# ---------------------------------------------------------------------------

#: Absolute-path tokens that are python scripts. Resolves the invoked script
#: paths — the check does NOT attempt to resolve arbitrary inline shell (D1).
_SCRIPT_PATH_RE = re.compile(r"(/[^\s'\";|&()]+\.py)\b")


def resolve_scripts(command: str) -> list[str]:
    """Every absolute `.py` path a node command invokes, de-duplicated in order.

    Only absolute paths are resolved (a bare `python3` with no script arg, or a
    relative path, is not a resolvable artifact — it is not a drift row).
    """
    seen: list[str] = []
    for m in _SCRIPT_PATH_RE.finditer(command):
        p = m.group(1)
        if p not in seen:
            seen.append(p)
    return seen


# ---------------------------------------------------------------------------
# Manifest closure (DoD-0) — who deploys a given script
# ---------------------------------------------------------------------------

@dataclass
class ManifestClosure:
    """The three deploy manifests (imported from pm_core so the checker audits
    the SAME tuples the deploy hook writes — no re-declaration that can drift).

    Counts verified at PR time by an AST pass handling AnnAssign (DoD-0):
    _CONDUCTOR_NIGHT_SCRIPTS (25, :532), _CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS
    (8->11, :645), _CONDUCTOR_GW_HOST_SCRIPTS (5, :602).
    """
    night_scripts: tuple[str, ...]
    brix_gw_runtime: tuple[str, ...]
    gw_host_dests: tuple[str, ...]

    def local_manifested(self) -> set[str]:
        """Basename set of every script the local deploy pass (deploy clone ->
        /data/agents/scripts) delivers — the NIGHT + BRIX_GW_RUNTIME families
        share one destination dir."""
        return set(self.night_scripts) | set(self.brix_gw_runtime)

    def gw_host_dest_basenames(self) -> set[str]:
        return {Path(d).name for d in self.gw_host_dests}


def manifest_closure_from_pm_core() -> ManifestClosure:
    """Import the live tuples from pm_core (single source of truth)."""
    from . import pm_core
    return ManifestClosure(
        night_scripts=tuple(pm_core._CONDUCTOR_NIGHT_SCRIPTS),
        brix_gw_runtime=tuple(pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS),
        gw_host_dests=tuple(e.dest for e in pm_core._CONDUCTOR_GW_HOST_SCRIPTS),
    )


def manifest_counts_ast() -> dict[str, int]:
    """DoD-0 verification: re-verify the manifest counts by an AST pass that
    handles BOTH Assign and AnnAssign (the :532 tuple is an AnnAssign; a grep
    or Assign-only AST pass silently misses it — the I4 near-miss).

    Reads the ON-DISK pm_core.py source file (the module this checker audits —
    single source of truth, not a re-declaration that can drift). The on-disk
    read is deliberate: an in-memory read (inspect.getsource) verifies the
    imported module rather than the file, so it cannot catch a case where the
    source file and the imported module genuinely disagree (e.g. a stale
    deploy clone). Falls back to inspect.getsource when the on-disk file is
    unreadable. Returns {name: count}; a missing name -> 0.
    """
    import ast
    import inspect

    from . import pm_core
    source: str | None = None
    try:
        source = Path(inspect.getsourcefile(pm_core)).read_text()
    except (OSError, TypeError):
        pass
    if source is None:
        try:
            source = inspect.getsource(pm_core)
        except (OSError, SyntaxError):
            return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    names = {
        "_CONDUCTOR_NIGHT_SCRIPTS",
        "_CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS",
        "_CONDUCTOR_GW_HOST_SCRIPTS",
    }
    out: dict[str, int] = {n: 0 for n in names}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id in names:
                if isinstance(value, ast.Tuple):
                    out[t.id] = len(value.elts)
    return out


# ---------------------------------------------------------------------------
# Waivers (D2) — explicit, first-class, expiring
# ---------------------------------------------------------------------------

@dataclass
class Waiver:
    script: str
    reason: str
    added: str
    owner: str
    expires: str | None

    def is_expired(self, now: datetime | None = None) -> bool:
        if not self.expires:
            return False
        try:
            exp = datetime.fromisoformat(str(self.expires).replace("Z", "+00:00"))
        except ValueError:
            return False
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        now = now or datetime.now(timezone.utc)
        return exp <= now

    def is_forever(self) -> bool:
        """A forever-waiver (missing/absurd expiry) is a bug the node reports
        (D2, Q-c). Absurd = no expiry, or an expiry beyond the max horizon."""
        if not self.expires:
            return True
        try:
            exp = datetime.fromisoformat(str(self.expires).replace("Z", "+00:00"))
        except ValueError:
            return True
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        horizon = timedelta(days=_WAIVER_MAX_HORIZON_DAYS)
        return (exp - now) > horizon


def load_waivers(path: Path | None = None) -> tuple[list[Waiver], list[str]]:
    """Load the waiver file. Returns (waivers, problems).

    problems are structured strings for a waiver that is missing a required
    field (script/reason/owner) — those are reported, not silently dropped.
    A forever-waiver is NOT a problem here (it is reported by the attestation
    itself as a bug, D2); it is loaded and carried so the caller can flag it.
    """
    path = path or WAIVER_FILE
    try:
        text = path.read_text()
    except OSError:
        return [], []
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return [], ["waiver file is not valid yaml"]
    if doc is None:
        return [], []
    entries = doc if isinstance(doc, list) else doc.get("waivers", [])
    if not isinstance(entries, list):
        return [], ["waiver file top-level is not a list"]

    waivers: list[Waiver] = []
    problems: list[str] = []
    for i, raw in enumerate(entries):
        if not isinstance(raw, dict):
            problems.append(f"waiver[{i}] is not a mapping")
            continue
        script = raw.get("script")
        reason = raw.get("reason")
        owner = raw.get("owner")
        if not script:
            problems.append(f"waiver[{i}] missing script")
            continue
        if not reason:
            problems.append(f"waiver[{i}] ({script}) missing reason")
            continue
        if not owner:
            problems.append(f"waiver[{i}] ({script}) missing owner")
            continue
        waivers.append(Waiver(
            script=str(script), reason=str(reason),
            added=str(raw.get("added", "")), owner=str(owner),
            expires=raw.get("expires"),
        ))
    return waivers, problems


def waiver_for(script: str, waivers: list[Waiver]) -> Waiver | None:
    """The waiver whose basename matches `script`'s basename, if any."""
    base = Path(script).name
    for w in waivers:
        if Path(w.script).name == base:
            return w
    return None


# ---------------------------------------------------------------------------
# Per-host verification (D6) — local cmp + mtimes; GW via bounded ssh
# ---------------------------------------------------------------------------

@dataclass
class HostRow:
    """One script's per-host attestation row."""
    script: str
    host: str
    status: str  # "clean" | "drift" | "absent" | "unverifiable" | "waived"
    source_mtime: str | None = None
    live_mtime: str | None = None
    cmp: str | None = None  # "equal" | "differ" | "absent" | "unverifiable"
    age_days: float | None = None
    waiver: str | None = None
    detail: str = ""


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _age_days(src_mtime: float, live_mtime: float) -> float:
    return max(0.0, (src_mtime - live_mtime) / 86400.0)


def verify_local_script(script: str, src_dir: Path, dest_dir: Path) -> HostRow:
    """Local host verification (deploy clone source vs /data/agents/scripts).

    `script` is the absolute live path the node command invokes. The source of
    truth is CONDUCTOR_SCRIPTS_SRC/<basename>; the live artifact is the invoked
    path (or the /data/agents/scripts copy if the invoked path is absent).
    """
    base = Path(script).name
    src = src_dir / base
    live = Path(script)
    if not live.exists():
        # Fall back to the host-op dir copy (the invoked path may be a
        # deploy-clone path that the runtime reads from /data/agents/scripts).
        alt = dest_dir / base
        if alt.exists():
            live = alt
        else:
            return HostRow(script=script, host="brix", status="absent",
                           cmp="absent", detail=f"live artifact absent: {script}")

    if not src.exists():
        # No source of truth in the conductor tree — the script is not a
        # conductor-held artifact; the checker cannot attest its completeness.
        # (Reported as unverifiable, not a silent pass.)
        return HostRow(script=script, host="brix", status="unverifiable",
                       cmp="unverifiable",
                       detail=f"no conductor source for {base}")

    src_mt = src.stat().st_mtime
    live_mt = live.stat().st_mtime
    same = filecmp.cmp(str(src), str(live), shallow=False)
    status = "clean" if same else "drift"
    return HostRow(
        script=script, host="brix", status=status,
        source_mtime=_mtime_iso(src), live_mtime=_mtime_iso(live),
        cmp="equal" if same else "differ",
        age_days=round(_age_days(src_mt, live_mt), 3),
    )


def _gw_ssh_capture(remote_cmd: list, timeout: int = _GW_SSH_CMD_TIMEOUT_SECS) -> str | None:
    """Bounded never-wake ssh probe (D6): list argv, never raises, None on
    failure.

    DELEGATES to pm_core._gw_ssh_capture — the ssh discipline (BatchMode
    never-wake, ConnectTimeout bound, None-on-failure) is defined in ONE place
    (pm_core, the deploy pass) so the two call sites cannot drift. This module
    only owns the audit surface (which hosts, which rows), not the transport.
    """
    from . import pm_core
    return pm_core._gw_ssh_capture(remote_cmd, timeout=timeout)


def _gw_reachable() -> bool:
    """Reachability probe — delegates to pm_core._gw_host_reachable (the same
    single-source-of-truth discipline as _gw_ssh_capture above)."""
    from . import pm_core
    return pm_core._gw_host_reachable()


def verify_gw_scripts(src_dir: Path, gw_dests: tuple[str, ...]) -> list[HostRow]:
    """GW /usr/local/sbin|bin attestation via bounded never-wake ssh (D6).

    A GW-only gap is reported separately (its own rows), never collapsed into
    the local boolean. Unreachable GW -> rows with status "unverifiable"
    (not a silent pass, I6: the checker reports what it cannot verify).
    """
    rows: list[HostRow] = []
    if not _gw_reachable():
        for dest in gw_dests:
            base = Path(dest).name
            src = src_dir / base
            rows.append(HostRow(
                script=dest, host="gravitywell", status="unverifiable",
                cmp="unverifiable",
                detail="gw unreachable (never-wake probe failed)",
            ))
        return rows

    for dest in gw_dests:
        base = Path(dest).name
        src = src_dir / base
        if not src.exists():
            rows.append(HostRow(script=dest, host="gravitywell",
                                status="unverifiable", cmp="unverifiable",
                                detail=f"no conductor source for {base}"))
            continue
        local_hash = _sha256_file(src)
        remote_hash_out = _gw_ssh_capture(["sha256sum", dest])
        remote_hash = remote_hash_out.split()[0] if remote_hash_out else None
        if remote_hash is None:
            rows.append(HostRow(script=dest, host="gravitywell",
                                status="unverifiable", cmp="unverifiable",
                                detail="remote hash unreadable"))
            continue
        same = (remote_hash == local_hash)
        rows.append(HostRow(
            script=dest, host="gravitywell",
            status="clean" if same else "drift",
            cmp="equal" if same else "differ",
            detail=f"local={local_hash[:12]} remote={(remote_hash or '')[:12]}",
        ))
    return rows


def _is_under(path: str, root: Path) -> bool:
    """True if `path` is inside `root` (path containment, symlink-naive — the
    deploy-clone paths are real dirs, not symlinks, on BRIX)."""
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _sha256_file(path: Path) -> str:
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# The attestation pass
# ---------------------------------------------------------------------------

@dataclass
class AttestationResult:
    verdict: str  # "clean" | "held"
    reason: str | None
    rows: list[HostRow]
    unmanifested: list[str]
    waived: list[str]
    waiver_problems: list[str]
    schema_drift: bool = False
    schema_drift_detail: str = ""
    counts: dict[str, int] = field(default_factory=dict)

    def drift_rows(self) -> list[HostRow]:
        return [r for r in self.rows if r.status == "drift"]

    def to_record(self, run_id: str | None, ts: str) -> dict:
        """One ledger record (the append-ledger entry, D5)."""
        return {
            "ts": ts,
            "run_id": run_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "schema_drift": self.schema_drift,
            "schema_drift_detail": self.schema_drift_detail,
            "counts": self.counts,
            "unmanifested": self.unmanifested,
            "waived": self.waived,
            "waiver_problems": self.waiver_problems,
            "rows": [
                {
                    "script": r.script, "host": r.host, "status": r.status,
                    "source_mtime": r.source_mtime, "live_mtime": r.live_mtime,
                    "cmp": r.cmp, "age_days": r.age_days,
                    "waiver": r.waiver, "detail": r.detail,
                }
                for r in self.rows
            ],
        }


def run_attestation(
    *,
    plan_path: Path | None = None,
    waiver_path: Path | None = None,
    src_dir: Path | None = None,
    dest_dir: Path | None = None,
    closure: ManifestClosure | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
) -> AttestationResult:
    """Run the full attestation pass. Never raises — a detector bug is a
    `held`/`failed` verdict with reason `internal_error`, not a crash (I2)."""
    now = now or datetime.now(timezone.utc)
    try:
        return _run_attestation_impl(
            plan_path=plan_path, waiver_path=waiver_path,
            src_dir=src_dir, dest_dir=dest_dir, closure=closure,
            run_id=run_id, now=now,
        )
    except Exception as e:  # noqa: BLE001 - I2: a detector bug is a verdict,
        # not a crash. `held` + internal_error so it is never mistaken for
        # drift and never reads as a silent pass.
        print(f"[attestation] detector error (non-fatal): {type(e).__name__}: {e}",
              file=sys.stderr)
        return AttestationResult(
            verdict=VERDICT_HELD, reason=REASON_INTERNAL_ERROR,
            rows=[], unmanifested=[], waived=[], waiver_problems=[],
        )


def _run_attestation_impl(
    *,
    plan_path: Path | None,
    waiver_path: Path | None,
    src_dir: Path | None,
    dest_dir: Path | None,
    closure: ManifestClosure | None,
    run_id: str | None,
    now: datetime,
) -> AttestationResult:
    plan_path = plan_path or NIGHT_PLAN_PATH
    waiver_path = waiver_path or WAIVER_FILE
    src_dir = src_dir or CONDUCTOR_SCRIPTS_SRC
    dest_dir = dest_dir or AGENTS_SCRIPTS_DIR
    closure = closure or manifest_closure_from_pm_core()

    # --- plan parse (D1) ---
    try:
        plan_text = plan_path.read_text()
    except OSError as e:
        return AttestationResult(
            verdict=VERDICT_HELD, reason=REASON_PLAN_SCHEMA_DRIFT,
            rows=[], unmanifested=[], waived=[], waiver_problems=[],
            schema_drift=True, schema_drift_detail=f"plan unreadable: {e}",
        )
    parsed = parse_plan_narrow(plan_text)
    if parsed.schema_drift:
        return AttestationResult(
            verdict=VERDICT_HELD, reason=REASON_PLAN_SCHEMA_DRIFT,
            rows=[], unmanifested=[], waived=[], waiver_problems=[],
            schema_drift=True, schema_drift_detail=parsed.schema_drift_detail,
        )

    waivers, waiver_problems = load_waivers(waiver_path)

    # --- DoD-6: no node may depend on the attestation node ---
    for node in parsed.nodes:
        if ATTTESTATION_NODE_ID in node.after:
            return AttestationResult(
                verdict=VERDICT_HELD, reason=REASON_PLAN_SCHEMA_DRIFT,
                rows=[], unmanifested=[], waived=[], waiver_problems=[],
                schema_drift=True,
                schema_drift_detail=(
                    f"DoD-6 violated: node {node.id!r} declares "
                    f"after: [{ATTTESTATION_NODE_ID}]"
                ),
            )

    # --- resolve every script a node command invokes ---
    scripted: dict[str, list[str]] = {}  # script path -> node ids
    for node in parsed.nodes:
        for script in resolve_scripts(node.command):
            scripted.setdefault(script, []).append(node.id)

    rows: list[HostRow] = []
    unmanifested: list[str] = []
    waived: list[str] = []

    local_manifested = closure.local_manifested()
    for script, node_ids in sorted(scripted.items()):
        base = Path(script).name
        w = waiver_for(script, waivers)

        # A script invoked FROM the conductor deploy clone executes directly
        # from a tree _POST_LAND_PULL keeps current — it is not a
        # deploy-manifest gap (D6 audit surface: deploy clone / agents-scripts
        # / GW). Only scripts invoked from a SEPARATE execution location
        # (/data/agents/scripts/) that must be delivered there by a manifest
        # are drift rows.
        if _is_under(script, CONDUCTOR_DEPLOY_CLONE):
            continue

        # A script that lives in /data/agents/scripts and is NOT in any local
        # manifest is the exact drift row (unmanifested). A script that is in
        # a manifest gets a live-vs-source cmp row.
        if base not in local_manifested:
            # Unmanifested: is it waived?
            if w is not None:
                if w.is_expired(now):
                    # An expired waiver is re-reported as drift (D2).
                    row = _drift_row(script, base, src_dir, dest_dir)
                    row.waiver = f"expired ({w.expires})"
                    rows.append(row)
                    unmanifested.append(script)
                else:
                    waived.append(script)
                    rows.append(HostRow(
                        script=script, host="brix", status="waived",
                        waiver=f"{w.owner}: {w.reason}",
                        detail=f"nodes: {', '.join(node_ids)}",
                    ))
                continue
            # Unmanifested and not waived -> drift row (the named gap).
            row = _unmanifested_row(script, base, src_dir, dest_dir, node_ids)
            rows.append(row)
            unmanifested.append(script)
            continue

        # Manifested: live-vs-source cmp (D6 local host).
        row = verify_local_script(script, src_dir, dest_dir)
        if w is not None and row.status == "drift":
            # A waived script that later drifts is re-reported (D2).
            row.waiver = f"waived-but-drift ({w.owner})"
        rows.append(row)

    # --- GW host rows (D6) — reported separately, never collapsed ---
    rows.extend(verify_gw_scripts(src_dir, closure.gw_host_dests))

    # --- waiver bugs (D2): forever-waivers are a bug the node reports ---
    for w in waivers:
        if w.is_forever():
            waiver_problems.append(
                f"forever-waiver: {w.script} (no/absurd expires — a bug, D2)"
            )

    # --- verdict ---
    drift = [r for r in rows if r.status == "drift"]
    if unmanifested or drift:
        return AttestationResult(
            verdict=VERDICT_HELD, reason=REASON_DRIFT, rows=rows,
            unmanifested=unmanifested, waived=waived,
            waiver_problems=waiver_problems,
        )
    if waiver_problems:
        return AttestationResult(
            verdict=VERDICT_HELD, reason=REASON_WAIVER_FOREVER, rows=rows,
            unmanifested=unmanifested, waived=waived,
            waiver_problems=waiver_problems,
        )
    return AttestationResult(
        verdict=VERDICT_CLEAN, reason=REASON_NONE, rows=rows,
        unmanifested=unmanifested, waived=waived,
        waiver_problems=waiver_problems,
    )


def _unmanifested_row(script: str, base: str, src_dir: Path, dest_dir: Path,
                      node_ids: list[str]) -> HostRow:
    """An unmanifested script's row: does a live copy exist and does it match
    the merged source? Both mtimes + cmp are reported (DoD-1/DoD-3)."""
    src = src_dir / base
    live = Path(script)
    if not live.exists():
        alt = dest_dir / base
        if alt.exists():
            live = alt
    if not src.exists():
        return HostRow(script=script, host="brix", status="drift",
                       cmp="absent",
                       detail=f"unmanifested; no conductor source; nodes: {', '.join(node_ids)}")
    if not live.exists():
        return HostRow(script=script, host="brix", status="drift",
                       cmp="absent",
                       detail=f"unmanifested; live absent; nodes: {', '.join(node_ids)}")
    same = filecmp.cmp(str(src), str(live), shallow=False)
    # Unmanifested is ALWAYS a drift row (DoD-1): the completeness invariant is
    # violated regardless of whether the live copy happens to match the source.
    # The cmp field still records whether the content matches (DoD-3).
    return HostRow(
        script=script, host="brix", status="drift",
        source_mtime=_mtime_iso(src), live_mtime=_mtime_iso(live),
        cmp="equal" if same else "differ",
        age_days=round(_age_days(src.stat().st_mtime, live.stat().st_mtime), 3),
        detail=f"unmanifested (not in any deploy manifest); nodes: {', '.join(node_ids)}",
    )


def _drift_row(script: str, base: str, src_dir: Path, dest_dir: Path) -> HostRow:
    """A waived-but-now-drift script's row (re-reported, D2)."""
    src = src_dir / base
    live = Path(script)
    if not live.exists():
        live = dest_dir / base
    if not src.exists() or not live.exists():
        return HostRow(script=script, host="brix", status="drift", cmp="absent")
    same = filecmp.cmp(str(src), str(live), shallow=False)
    return HostRow(
        script=script, host="brix", status="drift" if not same else "clean",
        source_mtime=_mtime_iso(src), live_mtime=_mtime_iso(live),
        cmp="equal" if same else "differ",
        age_days=round(_age_days(src.stat().st_mtime, live.stat().st_mtime), 3),
    )


# ---------------------------------------------------------------------------
# Ledger (D5) — append-ledger-shaped, atomic writes (I7)
# ---------------------------------------------------------------------------

def read_ledger(path: Path | None = None) -> list[dict]:
    path = path or ATTESTATION_LEDGER_PATH
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(doc, dict):
        return doc.get("records", [])
    if isinstance(doc, list):
        return doc
    return []


def append_ledger_record(record: dict, path: Path | None = None) -> None:
    """Append one record to the ledger, atomically (temp + os.replace, I7).

    The ledger is append-shaped (a historical record, not a snapshot) — the
    Murasaki bridge from the Council: the artifact is the enduring record.
    Bounded: keep the most recent _LEDGER_MAX_RECORDS so it does not grow
    unboundedly across nights.
    """
    path = path or ATTESTATION_LEDGER_PATH
    records = read_ledger(path)
    records.append(record)
    records = records[-_LEDGER_MAX_RECORDS:]
    _atomic_write_json(path, {"records": records})


_LEDGER_MAX_RECORDS = 200


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Alarm dedup (D7, DoD-8) — the _GW_HOST_SCRIPT_DRIFT_LEDGER shape
# ---------------------------------------------------------------------------

def read_alarm_ledger(path: Path | None = None) -> dict:
    path = path or ALARM_LEDGER_PATH
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_alarm_ledger(ledger: dict, path: Path | None = None) -> None:
    path = path or ALARM_LEDGER_PATH
    _atomic_write_json(path, ledger)


def _alarm_cooldown_elapsed(entry: dict, now: datetime) -> bool:
    """True if enough time has passed since the last alarm to page again."""
    last = entry.get("last_alarm_at")
    if not last:
        return True
    try:
        dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
    except ValueError:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds() >= ALARM_COOLDOWN_SECS


def maybe_alarm_drift(
    result: AttestationResult,
    *,
    alarm_ledger_path: Path | None = None,
    deposit=None,
    now: datetime | None = None,
) -> dict:
    """D7/DoD-8: deposit a deduped alarm for a persisting drift.

    `deposit` is a callable taking a payload dict and returning a gem id (or
    None) — injected so tests can observe the deposit without hitting the Desk.
    A persisting held pages once per day (the cooldown), not every night.
    Returns a small status dict (deposited / suppressed / resolved).
    """
    now = now or datetime.now(timezone.utc)
    ledger = read_alarm_ledger(alarm_ledger_path)
    entry = ledger.get(ALARM_LEDGER_SIGNATURE)
    drift = result.drift_rows()
    unmanifested = result.unmanifested

    if not drift and not unmanifested:
        # Healthy: resolve any open entry so a later regression pages fresh.
        if entry is not None and entry.get("status") == "open":
            entry["status"] = "resolved"
            entry["last_seen"] = now.isoformat()
            _write_alarm_ledger(ledger, alarm_ledger_path)
        return {"status": "resolved"}

    would_page = (
        entry is None
        or entry.get("status") in ("acked", "resolved")
        or _alarm_cooldown_elapsed(entry, now)
    )
    if not would_page:
        entry["last_seen"] = now.isoformat()
        entry["times_seen"] = entry.get("times_seen", 1) + 1
        _write_alarm_ledger(ledger, alarm_ledger_path)
        return {"status": "suppressed"}

    payload = _build_alarm_payload(result, now)
    gem_id = None
    if deposit is not None:
        try:
            gem_id = deposit(payload)
        except Exception:
            gem_id = None
    if gem_id is None:
        # No deposit available (or it failed) — still record last_seen so the
        # cooldown is honest, but mark the entry unresolved so the next pass
        # retries the page.
        entry = {
            "first_seen": (entry or {}).get("first_seen") or now.isoformat(),
            "last_seen": now.isoformat(),
            "times_seen": (entry or {}).get("times_seen", 0) + 1,
            "status": "open",
        }
        ledger[ALARM_LEDGER_SIGNATURE] = entry
        _write_alarm_ledger(ledger, alarm_ledger_path)
        return {"status": "deposit_failed"}

    ledger[ALARM_LEDGER_SIGNATURE] = {
        "gem_id": gem_id,
        "first_seen": (entry or {}).get("first_seen") or now.isoformat(),
        "last_seen": now.isoformat(),
        "last_alarm_at": now.isoformat(),
        "times_seen": (entry or {}).get("times_seen", 0) + 1,
        "status": "open",
    }
    _write_alarm_ledger(ledger, alarm_ledger_path)
    return {"status": "deposited", "gem_id": gem_id}


def _build_alarm_payload(result: AttestationResult, now: datetime) -> dict:
    drift_lines = [
        f"{r.script}: {r.host} {r.cmp} (age {r.age_days}d)" for r in result.drift_rows()
    ]
    unmanifested_lines = [f"{s}: unmanifested" for s in result.unmanifested]
    lines = drift_lines + unmanifested_lines
    return {
        "title": f"night-deploy-attestation: {len(lines)} drifted/unmanifested script(s)",
        "ask": "; ".join(lines),
        "why": (
            "Every script a scheduled node executes must be deployed from the "
            "tree it merged into, or explicitly waived. A drift here means a "
            "node is running stale code and reporting green."
        ),
        "context": [
            {"label": "Drifted", "lines": drift_lines or ["(none)"]},
            {"label": "Unmanifested", "lines": unmanifested_lines or ["(none)"]},
        ],
        "options": [
            {"key": "ack_watch", "title": "Known - keep watching",
             "sub": "stay quiet unless it recurs after a fix", "primary": True},
            {"key": "not_real", "title": "Not a real issue",
             "sub": "suppress this finding-class"},
        ],
        "state": "needs",
        "origin": "from · night-deploy-attestation",
        "agent": "repair-expert",
        "deposited_by": "lapis-pm-night-deploy-manifest-attestation-v0",
    }


# ---------------------------------------------------------------------------
# Node entry point (the DAG node's command) — prints a machine-readable verdict
# ---------------------------------------------------------------------------

def _read_run_id(run_record_path: Path | None = None) -> str | None:
    run_record_path = run_record_path or RUN_RECORD_PATH
    try:
        doc = json.loads(run_record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(doc, dict):
        return doc.get("run_id")
    return None


def node_main(
    *,
    plan_path: Path | None = None,
    waiver_path: Path | None = None,
    ledger_path: Path | None = None,
    src_dir: Path | None = None,
    dest_dir: Path | None = None,
    closure: ManifestClosure | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
    emit: bool = True,
) -> int:
    """The DAG node's entry point. Runs the attestation, appends the ledger
    record, fires the deduped alarm, and returns the exit code the node's
    success_predicate maps (0 clean / 2 held / 3 failed).

    Prints a single machine-readable JSON line to stdout (the node's log
    surface) — the verdict + reason reach the run record's per-node block,
    which the morning brief + agora intake leg already read (DoD-0/D5).
    """
    now = now or datetime.now(timezone.utc)
    result = run_attestation(
        plan_path=plan_path, waiver_path=waiver_path,
        src_dir=src_dir, dest_dir=dest_dir, closure=closure,
        run_id=run_id, now=now,
    )
    if result.reason == REASON_INTERNAL_ERROR:
        # A detector bug is `failed` (not held) so it is not mistaken for
        # drift (I2: undetected drift is a detector bug).
        code = EXIT_FAILED
    elif result.verdict == VERDICT_HELD:
        code = EXIT_HELD
    else:
        code = EXIT_CLEAN

    run_id = run_id or _read_run_id()
    record = result.to_record(run_id, now.isoformat())
    try:
        append_ledger_record(record, ledger_path)
    except Exception as e:  # noqa: BLE001
        print(f"[attestation] ledger write failed (non-fatal): {e}", file=sys.stderr)

    # Deduped alarm (D7) — best-effort; a deposit failure never changes the
    # node's verdict.
    try:
        maybe_alarm_drift(result, now=now)
    except Exception as e:  # noqa: BLE001
        print(f"[attestation] alarm pass failed (non-fatal): {e}", file=sys.stderr)

    if emit:
        print(json.dumps({
            "node": ATTTESTATION_NODE_ID,
            "verdict": result.verdict,
            "reason": result.reason,
            "drift": len(result.drift_rows()),
            "unmanifested": result.unmanifested,
            "waived": result.waived,
            "waiver_problems": result.waiver_problems,
        }))
    return code


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="lapis-pm attestation-check")
    p.add_argument("--plan", default=None, help="Night plan path (default /data/slots/night-plan.yaml)")
    p.add_argument("--waivers", default=None, help="Waiver file path")
    p.add_argument("--ledger", default=None, help="Attestation ledger path")
    p.add_argument("--src", default=None, help="Conductor scripts source dir")
    p.add_argument("--dest", default=None, help="BRIX host-op scripts dir")
    p.add_argument("--no-emit", action="store_true", help="Do not print the JSON line")
    args = p.parse_args(argv)

    def _p(v):
        return Path(v) if v else None

    return node_main(
        plan_path=_p(args.plan), waiver_path=_p(args.waivers),
        ledger_path=_p(args.ledger), src_dir=_p(args.src),
        dest_dir=_p(args.dest), emit=not args.no_emit,
    )


if __name__ == "__main__":
    sys.exit(main())
