"""Attestation-drift function for `/srv/lapis/systems` specs.

Target: `system-spec-drift-attestation-v0`. Ratified 2026-08-06 (gate run
`2026-08-06-144529`; revision 2 amendments a/b/c — see the spec's §Gate
record). Implements Direction A only (corpus -> attestation, i.e. drift
accrual on land); Direction B (mem.db decision seeding into repowise) is a
separate, non-hot-path concern — see `seed_decisions_from_mem` below.

Hooked from `_post_land_deploy_hook` at its **definition**
(`pm_core.py:1627`) via `run_post_land_attestation()`, so all four call
sites (`:567`, `:1463`, `:3120`, `:3200`) accrue drift, not just the
`merge_and_deploy` path. Best-effort and never raises — this mirrors that
hook's own documented contract ("Never raises — landing must complete even
if the pull or restart fails", pm_core.py:1628-1633) and the contract at
pm_core.py:555-560 ("a merge that succeeded must never be reported as
failed").

Invariant 6 (hard, ratified Erah 2026-08-06): **silence is never evidence
of stability.** Before this module renders "no drift" for a spec, it must
prove the watcher actually ran against current state — a stale or missing
repowise index renders `STALE-GRAPH` / `NO-INDEX`, never a clean row. See
`_graph_status()`.

Not in this module: host-path watching (systemd units, `~/.claude/*`,
GravityWell host paths — out of scope, see spec §Out of scope), the seven
unindexed-repo / gitignore host-ops prerequisites (host ops, no repo PR),
and the `[keys.systems]` agents-core classmap entry (a separate cross-repo
change — `_systems_root()` degrades gracefully until it lands).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --- graph-status sentinels (invariant 6: never render silence as clean) ---
STATUS_OK = "OK"
STALE_GRAPH = "STALE-GRAPH"
NO_INDEX = "NO-INDEX"
PARTIAL_COVERAGE = "PARTIAL-COVERAGE"

_DRIFT_ORDER = {"green": 0, "amber": 1, "red": 2}
_DRIFT_NEXT = {"green": "amber", "amber": "red", "red": "red"}

# Convention on BRIX (measured 2026-08-06): deploy clone at /srv/git/<repo>,
# shared PM working tree at /srv/git/<repo>-working. Rule (Erah 2026-08-06,
# reversing revision 1): prefer the deploy clone where one exists, fall back
# to the working tree, record which was used and at what sha.
_CLONE_ROOT = Path(os.environ.get("LAPIS_PM_CLONE_ROOT", "/srv/git"))

# mem.db key prefixes Direction B is allowed to read. Everything else
# (router/, pm/dispatched/, pm/cursor/, pm/brief-gem/, arc-scratch-*) is
# operational exhaust and must be culled before it ever reaches repowise —
# "this cull is a gate, not a filter applied later" (spec §Direction B).
_MEM_ALLOWED_PREFIXES = ("decision/", "finding/", "correction/", "reference/")


# --------------------------------------------------------------------------
# /srv/lapis/systems access
# --------------------------------------------------------------------------

def _systems_root() -> Path:
    """Resolve `/srv/lapis/systems` via the room_paths seam.

    `[keys.systems]` is not yet in the agents-core classmap — grepping it
    returns nothing (spec §Prerequisite). That is a small, separate
    cross-repo change. Until it lands, fall back to the same ROOM_ROOT
    env-var resolution `room_paths.py` itself uses, never a bare hardcoded
    `/srv/lapis/systems` (that would defeat ROOM_ROOT relocation, the whole
    point of the indirection).
    """
    try:
        from agents_core.room_paths import room_path
        return room_path("systems")
    except (KeyError, ImportError):
        room_root = os.environ.get("ROOM_ROOT", "/room")
        return Path(room_root) / "systems"


def _manifest_path() -> Path:
    return _systems_root() / "_manifest.md"


def iter_spec_paths(root: Path | None = None) -> list[Path]:
    """All `<slug>.md` specs under /srv/lapis/systems, excluding `_*.md` doctrine files."""
    base = root if root is not None else _systems_root()
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob("*.md") if not p.name.startswith("_"))


# --------------------------------------------------------------------------
# code-path classification (spec §Step zero — measured three-way split)
# --------------------------------------------------------------------------

def classify_code_path(raw: str) -> str:
    """Classify one `code-paths` entry per the three-way split measured in the spec.

    - repo-relative (`gardener/gardener/inputs.py`) — repowise-visible
    - absolute / home (`~/.claude/hooks/synapse-inject.py`) — not visible
    - host-scoped (`gravitywell:/usr/local/bin/gw-serve`, `systemd: unit`) — not visible
    """
    raw = raw.strip()
    first_segment = raw.split("/", 1)[0]
    if ":" in first_segment:
        return "host_scoped"
    if raw.startswith("~") or raw.startswith("/"):
        return "absolute"
    return "repo_relative"


def split_repo_relative(raw: str) -> tuple[str, str]:
    """`gardener/gardener/inputs.py` -> ("gardener", "gardener/inputs.py")."""
    if "/" not in raw:
        return raw, ""
    repo, relpath = raw.split("/", 1)
    return repo, relpath


# --------------------------------------------------------------------------
# spec frontmatter
# --------------------------------------------------------------------------

@dataclass
class Spec:
    path: Path
    slug: str
    frontmatter: dict[str, Any]
    body: str

    @property
    def drift(self) -> str:
        return str(self.frontmatter.get("drift", "green")).strip()

    @property
    def raw_code_paths(self) -> list[str]:
        cp = self.frontmatter.get("code-paths") or []
        if isinstance(cp, str):
            cp = [cp]
        return [str(p) for p in cp]

    def code_paths_for_repo(self, repo: str) -> list[str]:
        """repo-relative code-paths entries belonging to `repo`, as relpaths within it."""
        out = []
        for raw in self.raw_code_paths:
            if classify_code_path(raw) != "repo_relative":
                continue
            r, relpath = split_repo_relative(raw)
            if r == repo and relpath:
                out.append(relpath)
        return out

    def has_non_repo_relative_paths(self) -> bool:
        return any(classify_code_path(p) != "repo_relative" for p in self.raw_code_paths)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse the `---\\n<yaml>\\n---\\n<body>` contract from `_format.md`."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    _, fm_text, body = parts
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return fm, body


def load_spec(path: Path) -> Spec | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    fm, body = parse_frontmatter(text)
    if not fm:
        return None
    slug = str(fm.get("system") or path.stem)
    return Spec(path=path, slug=slug, frontmatter=fm, body=body)


def load_all_specs(root: Path | None = None) -> list[Spec]:
    specs = []
    for p in iter_spec_paths(root):
        s = load_spec(p)
        if s is not None:
            specs.append(s)
    return specs


# --------------------------------------------------------------------------
# repo clone resolution (deploy clone preferred, invariant 7: record used clone+sha)
# --------------------------------------------------------------------------

@dataclass
class CloneInfo:
    repo: str
    path: Path
    kind: str  # "deploy" | "working"
    sha: str | None
    dirty_count: int | None  # only set on a working-tree fallback


def _git(path: Path, *args: str, timeout: float = 8.0) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def resolve_repo_clone(repo: str, clone_root: Path | None = None) -> CloneInfo | None:
    """Deploy clone (`<root>/<repo>`) preferred, working tree (`<root>/<repo>-working`)
    fallback. Records which was used and at what sha — never inferred (invariant 7).
    """
    root = clone_root if clone_root is not None else _CLONE_ROOT
    deploy = root / repo
    working = root / f"{repo}-working"

    if (deploy / ".git").exists():
        sha = _git(deploy, "rev-parse", "HEAD")
        return CloneInfo(repo=repo, path=deploy, kind="deploy", sha=sha, dirty_count=None)

    if (working / ".git").exists():
        sha = _git(working, "rev-parse", "HEAD")
        dirty_out = _git(working, "status", "--porcelain") or ""
        dirty_count = len([ln for ln in dirty_out.splitlines() if ln.strip()])
        return CloneInfo(repo=repo, path=working, kind="working", sha=sha, dirty_count=dirty_count)

    return None


# --------------------------------------------------------------------------
# repowise: staleness (STALE-GRAPH, hard invariant) + dependency-graph reachability
# --------------------------------------------------------------------------

def _repowise_state(clone_path: Path) -> dict[str, Any] | None:
    state_file = clone_path / ".repowise" / "state.json"
    try:
        return json.loads(state_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def graph_status(clone: CloneInfo | None) -> tuple[str, str | None, str | None]:
    """`(status, index_sha, head_sha)`.

    Invariant 6 (hard): a stale or absent index must never let a spec render
    as "no drift" — it renders STALE-GRAPH or NO-INDEX instead. Raised by a
    Council stand-aside the landing omitted: "essential to distinguish
    between a healthy silence and a broken tool."
    """
    if clone is None:
        return NO_INDEX, None, None
    state = _repowise_state(clone.path)
    if state is None:
        return NO_INDEX, None, clone.sha
    index_sha = state.get("last_sync_commit")
    head_sha = clone.sha
    if not index_sha or not head_sha:
        return NO_INDEX, index_sha, head_sha
    if index_sha == head_sha or index_sha.startswith(head_sha) or head_sha.startswith(index_sha):
        return STATUS_OK, index_sha, head_sha
    return STALE_GRAPH, index_sha, head_sha


def _load_knowledge_graph(clone_path: Path) -> dict[str, Any] | None:
    """Read repowise's per-repo dependency graph directly from the exported
    `knowledge-graph.json` artifact.

    Deliberately reads the on-disk graph rather than shelling out to
    `repowise export --full` / `dead-code` — both were found broken against
    the installed 0.39.0 (`AttributeError: 'DeadCodeFinding' object has no
    attribute 'finding_type'` on `--full`, measured this session) with
    `docs_enabled: false` indexes. `knowledge-graph.json` is the stable,
    already-materialized artifact every index writes.
    """
    gpath = clone_path / ".repowise" / "knowledge-graph.json"
    try:
        return json.loads(gpath.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def reachable_from(
    clone_path: Path, start_relpaths: list[str], max_nodes: int = 5000
) -> set[str]:
    """Blast radius: files transitively **reachable from** `start_relpaths` —
    i.e. the files each start file structurally depends on — per repowise's
    dependency graph.

    Called with a spec's declared `code-paths` as the start set. An edge
    `{source, target, type: "imports", ...}` means `source` depends on
    `target`; if `target` changes, `source`'s behaviour (and so a spec
    attesting to `source`) may no longer hold, even though `target` itself
    is never named in the spec's `code-paths`. We BFS *forward* from each
    declared path along all edge types (imports, tested_by, calls, ...) —
    broad by design, since under-detection is the exact failure this target
    exists to close (spec: "naive path-matching ... under-detects").
    """
    graph = _load_knowledge_graph(clone_path)
    if graph is None:
        return set()

    dependencies: dict[str, set[str]] = {}
    for edge in graph.get("edges", []):
        target = edge.get("target", "")
        source = edge.get("source", "")
        if not target.startswith("file:") or not source.startswith("file:"):
            continue
        dependencies.setdefault(source, set()).add(target)

    frontier = {f"file:{rp}" for rp in start_relpaths}
    seen: set[str] = set()
    visited_nodes = 0
    while frontier and visited_nodes < max_nodes:
        nxt: set[str] = set()
        for node in frontier:
            for dep in dependencies.get(node, ()):
                if dep not in seen:
                    nxt.add(dep)
            visited_nodes += 1
        seen |= nxt
        frontier = nxt

    return {n[len("file:"):] for n in seen}


# --------------------------------------------------------------------------
# Direction A: land -> spec drift accrual
# --------------------------------------------------------------------------

@dataclass
class SpecLandResult:
    spec: Spec
    matched: bool
    match_kind: str | None  # "named" | "reachable" | None
    matched_paths: list[str] = field(default_factory=list)
    graph_status: str = STATUS_OK
    index_sha: str | None = None
    head_sha: str | None = None
    partial_coverage: bool = False


def evaluate_spec_for_land(
    spec: Spec, repo: str, changed_relpaths: list[str], clone: CloneInfo | None
) -> SpecLandResult | None:
    """None means this spec has no code-paths in `repo` at all — not relevant to this land."""
    repo_paths = spec.code_paths_for_repo(repo)
    partial = spec.has_non_repo_relative_paths()
    if not repo_paths:
        return None

    changed_set = set(changed_relpaths)
    named_matches = sorted(changed_set & set(repo_paths))

    status, index_sha, head_sha = graph_status(clone)

    if named_matches:
        return SpecLandResult(
            spec=spec, matched=True, match_kind="named", matched_paths=named_matches,
            graph_status=status, index_sha=index_sha, head_sha=head_sha,
            partial_coverage=partial,
        )

    # No named hit. Invariant 6: before declaring "no drift" via the
    # reachability check, prove the watcher ran against current state.
    if status != STATUS_OK:
        return SpecLandResult(
            spec=spec, matched=False, match_kind=None,
            graph_status=status, index_sha=index_sha, head_sha=head_sha,
            partial_coverage=partial,
        )

    # Start from the spec's declared code-paths and see whether the land's
    # changed files fall in their transitive dependency set (blast radius).
    reachable = reachable_from(clone.path, repo_paths) if clone else set()
    reach_hits = sorted(changed_set & reachable)
    if reach_hits:
        return SpecLandResult(
            spec=spec, matched=True, match_kind="reachable", matched_paths=reach_hits,
            graph_status=status, index_sha=index_sha, head_sha=head_sha,
            partial_coverage=partial,
        )

    return SpecLandResult(
        spec=spec, matched=False, match_kind=None,
        graph_status=status, index_sha=index_sha, head_sha=head_sha,
        partial_coverage=partial,
    )


def _degrade(drift: str) -> str:
    """Degrade-only: green -> amber -> red. Never restores (invariant 1)."""
    return _DRIFT_NEXT.get(drift, drift)


def _write_spec_frontmatter(spec: Spec, updates: dict[str, Any]) -> None:
    """Rewrite only frontmatter fields — spec content (the body) is never touched
    by machine (invariant 2)."""
    fm = dict(spec.frontmatter)
    fm.update(updates)
    new_text = "---\n" + yaml.safe_dump(fm, sort_keys=False, default_flow_style=False) + "---" + spec.body
    spec.path.write_text(new_text)


_MANIFEST_ROW_RE_TEMPLATE = r"(\|[^\n]*\[{slug_pattern}\][^\n]*\|)"


def _update_manifest_row(spec_filename: str, new_note: str, manifest: Path | None = None) -> None:
    """Best-effort: append `new_note` to the manifest row whose link targets
    `spec_filename`. Never raises; a missing/unparseable manifest is a no-op.
    """
    import re

    if manifest is None:
        manifest = _manifest_path()
    try:
        text = manifest.read_text()
    except OSError:
        return

    pattern = re.compile(
        r"^(\|.*\[" + re.escape(spec_filename) + r"\]\(" + re.escape(spec_filename) + r"\).*)\|(\s*)$",
        re.MULTILINE,
    )

    def _repl(m: "re.Match[str]") -> str:
        row = m.group(1)
        if new_note in row:
            return m.group(0)
        return row.rstrip() + f" {new_note} |"

    new_text, n = pattern.subn(_repl, text)
    if n == 0:
        return
    try:
        manifest.write_text(new_text)
    except OSError:
        pass


def apply_land_result(result: SpecLandResult, target_id: str | None, trigger: str) -> None:
    """Write the spec's frontmatter + its manifest row. Degrade-only (invariant 1),
    frontmatter-only (invariant 2), never raises (invariant 3 / best-effort)."""
    spec = result.spec
    who = target_id or f"trigger:{trigger}"
    manifest = spec.path.parent / "_manifest.md"

    if result.graph_status == STALE_GRAPH:
        _write_spec_frontmatter(spec, {
            "graph-status": STALE_GRAPH,
            "graph-index-sha": result.index_sha,
            "graph-head-sha": result.head_sha,
        })
        note = f"`{STALE_GRAPH}` (index {str(result.index_sha)[:8]} vs head {str(result.head_sha)[:8]})"
        _update_manifest_row(spec.path.name, note, manifest=manifest)
        return

    if result.graph_status == NO_INDEX:
        _write_spec_frontmatter(spec, {"graph-status": NO_INDEX})
        _update_manifest_row(spec.path.name, f"`{NO_INDEX}`", manifest=manifest)
        return

    if result.partial_coverage:
        _update_manifest_row(spec.path.name, f"`{PARTIAL_COVERAGE}`", manifest=manifest)

    if not result.matched:
        return

    new_drift = _degrade(spec.drift)
    if _DRIFT_ORDER.get(new_drift, 0) <= _DRIFT_ORDER.get(spec.drift, 0):
        # Already at/above this drift level (e.g. red -> red) — invariant 1
        # forbids "restoring", but a no-op re-write is also unnecessary.
        return

    _write_spec_frontmatter(spec, {
        "drift": new_drift,
        "drift-last-target": who,
        "graph-status": STATUS_OK,
    })
    note = (
        f"drift {spec.drift}->{new_drift} by `{who}` "
        f"({result.match_kind}: {', '.join(result.matched_paths[:3])})"
    )
    _update_manifest_row(spec.path.name, note, manifest=manifest)


def run_post_land_attestation(
    repo: str | None,
    changed_paths: list[str] | None,
    *,
    target_id: str | None = None,
    trigger: str = "post-land-hook",
) -> None:
    """Entry point called from `_post_land_deploy_hook` (pm_core.py:1627).

    Best-effort, never raises — a failure here must never be reported as a
    failed land (pm_core.py:555-560, :1628-1633).
    """
    try:
        _run(repo, changed_paths or [], target_id, trigger)
    except Exception as e:  # noqa: BLE001 - best-effort by contract
        print(f"[spec-attestation] pass failed (non-fatal): {e}", file=sys.stderr)


def _run(repo: str | None, changed_paths: list[str], target_id: str | None, trigger: str) -> None:
    if not repo or not changed_paths:
        return
    specs = load_all_specs()
    if not specs:
        return
    clone = resolve_repo_clone(repo)
    for spec in specs:
        result = evaluate_spec_for_land(spec, repo, changed_paths, clone)
        if result is None:
            continue
        apply_land_result(result, target_id, trigger)
        if result.matched:
            print(
                f"[spec-attestation] {spec.slug}: drift accrued by "
                f"{target_id or trigger} ({result.match_kind})",
                file=sys.stderr,
            )
        elif result.graph_status != STATUS_OK:
            print(
                f"[spec-attestation] {spec.slug}: {result.graph_status} "
                f"(index vs head mismatch for {repo})",
                file=sys.stderr,
            )


# --------------------------------------------------------------------------
# Direction B: mem.db ledger -> repowise decision layer (batched, not hot-path)
# --------------------------------------------------------------------------
#
# Open question #3 (spec §Gate record) is explicitly left open by this
# revision: per-land vs batched refresh. Not load-bearing for Direction A,
# so this module does not block on it. Choice made here: **batched** — run
# via a standalone CLI/cron invocation, not wired into
# `_post_land_deploy_hook`. Reason: `repowise decision add` is an
# interactive-only CLI (no flags to pass content non-interactively, checked
# this session against repowise 0.39.0), so seeding is a slower, closer-to-
# offline operation and does not belong on a land's best-effort hot path.

def cull_mem_keys_for_decisions(keys: list[str]) -> list[str]:
    """The gate mandated by spec §Direction B: keep only `decision/`,
    `finding/`, `correction/`, `reference/` keys. Everything else
    (`router/`, `pm/dispatched/`, `pm/cursor/`, `pm/brief-gem/`,
    `arc-scratch-*`, ...) is operational exhaust and is dropped here —
    "This cull is a gate, not a filter applied later."
    """
    return [k for k in keys if k.startswith(_MEM_ALLOWED_PREFIXES)]


def repo_for_mem_key(key: str, specs: list[Spec] | None = None) -> str | None:
    """Best-effort: take the repo from the related spec's `code-paths` where
    available (spec §Direction B — "taking the repo from the related spec's
    code-paths where available"). No match -> None, caller skips seeding
    rather than guessing.
    """
    specs = specs if specs is not None else load_all_specs()
    key_slug = key.split("/", 1)[-1] if "/" in key else key
    for spec in specs:
        if spec.slug and spec.slug in key_slug:
            for raw in spec.raw_code_paths:
                if classify_code_path(raw) == "repo_relative":
                    repo, _ = split_repo_relative(raw)
                    return repo
    return None
