"""Deploy inventory reconciler (lapis-pm-deploy-inventory-reconciler-v0).

Derives the actual deployment surface from the live host — systemd units
(system + --user) and editable pip installs — and reconciles it against the
hand-maintained `_POST_LAND_PULL` / `_POST_LAND_RESTART` maps in `pm_core`.
The maps are allowlists a human must remember to update; nothing previously
compared them against what is actually running. This module is the
comparison.

It also detects a related but distinct hazard (lapis-pm-deploy-inventory-
runtime-symlink-into-dev-tree-v0): a runtime config path resolving, through
a symlink chain, into a `-working` PM investigation tree the maps above do
not cover — see `find_runtime_symlink_findings`.

Detection-only (spec item 8): this module never mutates a clone or a unit.
It only fetches (read-only), inspects, and writes a status JSON. No pull,
checkout, reset, branch change, or service restart lives here.

Fail-open on bad probe data (spec Definitions): every `systemctl`/`git` call
is wrapped with a timeout and checked for a clean, non-empty, zero-exit
result. A malformed probe logs a warning and is skipped — it never
manufactures a finding.
"""

from __future__ import annotations

import configparser
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from agents_core.room_paths import room_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROBE_TIMEOUT_SECS = 5      # per-call timeout for systemctl/git probes (spec item 7)
_FETCH_TIMEOUT_SECS = 20     # `git fetch` gets a longer budget (network call)

_ACK_MARKER = ".deploy-unmapped-ack"
_STALE_LOCK_THRESHOLD_SECS = 6 * 3600  # default 6h (spec item 4)

# PM-managed clone roots — only paths resolving under one of these are inventoried
# (spec: editable-install scope + PYTHONPATH boundary / Council dependency-shadow reservation).
_CLONE_ROOT_PREFIXES = ("/srv/git", "/data/agents")

# A PM investigation ("-working") tree, by definition (D1), is a direct child
# of THIS root whose basename ends in "-working" — distinct from
# _CLONE_ROOT_PREFIXES (where we *walk* for symlinks, which also includes
# /data/agents) since a "-working" tree can only ever live under /srv/git.
_DEV_TREE_ROOT = "/srv/git"

# ELOOP-style bound on symlink chain length (D1a — transitive resolution).
# Matches the usual Linux kernel default; a chain this long is itself a
# malformed-input signal, not a real config route.
_MAX_SYMLINK_HOPS = 40

# The two finding kinds a bare/legacy `.deploy-unmapped-ack` (no content, or
# content naming nothing) has always silenced, by the pre-existing per-clone
# boolean at reconcile_inventory(). `runtime_symlink_into_dev_tree` is
# deliberately excluded from this set (D4) — see `_ack_kinds()`.
_LEGACY_ACK_KINDS = frozenset({"unmapped_live_clone", "pull_without_restart"})

_STATUS_FILE = room_path('lapis_state') / "deploy-inventory-status.json"
_LOCK_DIR = room_path('lapis_state') / "deploy-pull-lock"

_EXECSTART_ARGV_RE = re.compile(r'argv\[\]=(.*?)\s*;')

_SEVERITY_ORDER = {"HIGH": 0, "NORMAL": 1, "INFO": 2, None: 3}


# ---------------------------------------------------------------------------
# Data model (names/shape normative per spec item 1)
# ---------------------------------------------------------------------------

@dataclass
class BackingUnit:
    unit: str            # e.g. "lapis-expert.service"
    field: str            # "WorkingDirectory" | "ExecStart" | "PYTHONPATH"
    scope: str            # "system" | "user"
    live: bool
    systemd_type: str     # "simple" | "notify" | "oneshot" | "timer" | ...


@dataclass
class CloneEntry:
    path: str                                     # realpath'd clone root
    remote: str | None = None                     # origin URL, or None
    backing_units: list = field(default_factory=list)  # list[BackingUnit]
    is_editable_install: bool = False


@dataclass
class Finding:
    kind: str
    severity: str   # "HIGH" | "NORMAL" | "INFO"
    detail: str


@dataclass
class CurrencyResult:
    head: str | None
    branch: str | None
    commits_behind: int | None
    tracked_dirty: bool
    untracked_present: bool
    findings: list  # list[Finding]


# ---------------------------------------------------------------------------
# Fail-open subprocess wrapper — the single seam every git/systemctl call
# goes through, so tests can intercept every host call from one mock point.
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = _PROBE_TIMEOUT_SECS) -> subprocess.CompletedProcess | None:
    """Run `cmd` with an explicit timeout. Returns None on any hard failure
    (timeout, missing binary) — never raises. Caller still must check
    `.returncode` and `.stdout` for the empty/non-zero/unparseable cases
    (fail-open policy applies to both: skip the unit/clone, no finding).
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("[deploy-inventory] probe failed: %s (%s)", cmd, e)
        return None


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


# ---------------------------------------------------------------------------
# systemd enumeration
# ---------------------------------------------------------------------------

def _list_units(scope: str) -> list[str]:
    """Unit names (service + timer) for scope 'system' or 'user'. [] on probe failure."""
    cmd = ["systemctl"] + (["--user"] if scope == "user" else []) + [
        "list-units", "--all", "--no-legend", "--plain", "--type=service,timer",
    ]
    result = _run(cmd)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        logger.warning("[deploy-inventory] unit list failed for scope=%s", scope)
        return []
    units = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts:
            units.append(parts[0])
    return units


_SHOW_PROPS = ("ActiveState", "SubState", "Type", "WorkingDirectory", "ExecStart", "Environment")


def _show_unit(unit: str, scope: str) -> dict | None:
    """`systemctl show` properties for `unit`. None on any probe failure (fail-open)."""
    cmd = ["systemctl"] + (["--user"] if scope == "user" else []) + [
        "show", unit, "--no-page", "-p", ",".join(_SHOW_PROPS),
    ]
    result = _run(cmd)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        logger.warning("[deploy-inventory] systemctl show failed for %s (scope=%s)", unit, scope)
        return None
    props: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        props[k] = v
    return props


def _base_name(unit: str) -> str:
    return unit.rsplit(".", 1)[0]


def _unit_is_live(props: dict, timer_props: dict | None) -> bool:
    """Per spec Definitions §"live" unit: ActiveState=active + SubState=running,
    except Type=oneshot (timer-driven) units, which are live iff their timer is
    active (a scheduled oneshot still deploys code)."""
    if props.get("Type") == "oneshot":
        return timer_props is not None and timer_props.get("ActiveState") == "active"
    return props.get("ActiveState") == "active" and props.get("SubState") == "running"


def _parse_execstart_paths(exec_start_raw: str) -> list[str]:
    """Extract argv[] tokens from a `systemctl show` ExecStart value, e.g.
    '{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 -m foo.bar ; ignore_errors=no ; ... }'.
    """
    tokens: list[str] = []
    for m in _EXECSTART_ARGV_RE.finditer(exec_start_raw):
        tokens.extend(m.group(1).split())
    return tokens


def _resolve_git_root(path_str: str) -> str | None:
    """Path resolution per spec Definitions: realpath first (follow symlinks),
    then `git rev-parse --show-toplevel`. None if not a git-backed clone under
    a PM-managed root, or on any probe failure (fail-open).
    """
    if not path_str:
        return None
    try:
        resolved = str(Path(path_str).resolve())
    except OSError:
        return None
    result = _run(["git", "-C", resolved, "rev-parse", "--show-toplevel"])
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return None
    root = result.stdout.strip()
    if not any(root.startswith(prefix) for prefix in _CLONE_ROOT_PREFIXES):
        return None
    return root


def _candidate_paths(props: dict) -> list[tuple[str, str]]:
    """(raw_path, field_name) pairs extracted from a unit's WorkingDirectory,
    ExecStart, and Environment=PYTHONPATH= properties."""
    candidates: list[tuple[str, str]] = []

    wd = props.get("WorkingDirectory", "")
    if wd:
        candidates.append((wd, "WorkingDirectory"))

    for tok in _parse_execstart_paths(props.get("ExecStart", "")):
        if tok.startswith("/"):
            candidates.append((tok, "ExecStart"))

    env = props.get("Environment", "")
    for tok in env.split():
        if tok.startswith("PYTHONPATH="):
            for p in tok[len("PYTHONPATH="):].split(":"):
                if p:
                    candidates.append((p, "PYTHONPATH"))

    return candidates


def _get_remote(clone_path: str) -> str | None:
    """origin remote URL, read directly from .git/config (no subprocess needed —
    keeps this off the detection-only guard's subprocess surface entirely)."""
    config_path = Path(clone_path) / ".git" / "config"
    try:
        text = config_path.read_text()
    except OSError:
        return None
    parser = configparser.ConfigParser(strict=False)
    try:
        parser.read_string(text)
    except configparser.Error:
        return None
    section = 'remote "origin"'
    if parser.has_section(section) and parser.has_option(section, "url"):
        return parser.get(section, "url")
    return None


def derive_deploy_inventory() -> dict[str, CloneEntry]:
    """Enumerate the real deployment surface from the live host.

    Returns a dict keyed by resolved clone root. Fail-open throughout: a
    unit or clone whose probe data is truncated/empty/non-zero/unparseable
    is logged and skipped, never turned into a finding.
    """
    inventory: dict[str, CloneEntry] = {}

    for scope in ("system", "user"):
        unit_names = _list_units(scope)
        service_units = [u for u in unit_names if u.endswith(".service")]
        timer_units = [u for u in unit_names if u.endswith(".timer")]

        timer_props_by_base: dict[str, dict] = {}
        for tu in timer_units:
            props = _show_unit(tu, scope)
            if props is not None:
                timer_props_by_base[_base_name(tu)] = props

        for su in service_units:
            props = _show_unit(su, scope)
            if props is None:
                continue

            unit_type = props.get("Type", "")
            live = _unit_is_live(props, timer_props_by_base.get(_base_name(su)))

            seen_roots_for_unit: set[str] = set()
            for raw_path, field_name in _candidate_paths(props):
                root = _resolve_git_root(raw_path)
                if root is None or root in seen_roots_for_unit:
                    continue
                seen_roots_for_unit.add(root)
                entry = inventory.setdefault(root, CloneEntry(path=root))
                entry.backing_units.append(BackingUnit(
                    unit=su, field=field_name, scope=scope, live=live, systemd_type=unit_type,
                ))

    for _name, root in _discover_editable_installs().items():
        entry = inventory.setdefault(root, CloneEntry(path=root))
        entry.is_editable_install = True

    for root, entry in inventory.items():
        entry.remote = _get_remote(root)

    return inventory


# ---------------------------------------------------------------------------
# Editable pip installs
# ---------------------------------------------------------------------------

def _editable_location(dist) -> str | None:
    """The resolved filesystem location of an editable install, or None.

    `dist` is anything exposing importlib.metadata.Distribution's interface
    (`.metadata["Name"]`, `.read_text("direct_url.json")`) — tests pass fakes.
    """
    try:
        direct_url_text = dist.read_text("direct_url.json")
    except Exception:
        return None
    if not direct_url_text:
        return None
    try:
        data = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None
    if not data.get("dir_info", {}).get("editable"):
        return None
    url = data.get("url", "")
    if not url.startswith("file://"):
        return None
    from urllib.parse import urlparse, unquote
    return unquote(urlparse(url).path)


def _discover_editable_installs(distributions=None) -> dict[str, str]:
    """{package_name: resolved_clone_root} for editable installs whose resolved
    location is under a PM-managed clone root (spec: editable-install scope —
    discovered by scanning, never a hardcoded package list).
    """
    if distributions is None:
        import importlib.metadata
        distributions = importlib.metadata.distributions()

    result: dict[str, str] = {}
    for dist in distributions:
        try:
            name = dist.metadata["Name"]
        except Exception:
            continue
        location = _editable_location(dist)
        if not location:
            continue
        try:
            resolved = str(Path(location).resolve())
        except OSError:
            continue
        if any(resolved.startswith(p) for p in _CLONE_ROOT_PREFIXES):
            result[name] = resolved
    return result


# ---------------------------------------------------------------------------
# Reconciler — gap findings (spec item 2)
# ---------------------------------------------------------------------------

def _reverse_pull_map(post_land_pull: dict[str, list[str]]) -> dict[str, str]:
    """resolved_clone_path -> repo, over the union of every _POST_LAND_PULL entry."""
    reverse: dict[str, str] = {}
    for repo, paths in post_land_pull.items():
        for p in paths:
            try:
                resolved = str(Path(p).resolve())
            except OSError:
                continue
            reverse[resolved] = repo
    return reverse


def reconcile_inventory(
    inventory: dict[str, CloneEntry],
    post_land_pull: dict[str, list[str]],
    post_land_restart: dict[str, tuple[str, ...]],
    post_land_restart_user: dict[str, tuple[str, ...]],
) -> dict[str, list[Finding]]:
    """Compare the derived inventory against the pull/restart maps. Emits
    `unmapped_live_clone` and `pull_without_restart` findings (spec item 2).
    Both are NORMAL severity, downgraded to INFO by the `.deploy-unmapped-ack`
    escape hatch (spec item 2 + escalation policy item 5).
    """
    reverse_map = _reverse_pull_map(post_land_pull)
    restart_covered = set(post_land_restart) | set(post_land_restart_user)

    findings_by_clone: dict[str, list[Finding]] = {}
    for path, entry in inventory.items():
        findings: list[Finding] = []
        repo = reverse_map.get(path)
        acked = (Path(path) / _ACK_MARKER).exists()
        is_live = entry.is_editable_install or any(bu.live for bu in entry.backing_units)

        if repo is None:
            if is_live:
                severity = "INFO" if acked else "NORMAL"
                live_units = [bu.unit for bu in entry.backing_units if bu.live]
                findings.append(Finding(
                    "unmapped_live_clone", severity,
                    f"{path} backs live unit(s) {live_units or ['<editable install>']} "
                    f"but is absent from every _POST_LAND_PULL entry",
                ))
        elif repo not in restart_covered:
            for bu in entry.backing_units:
                if bu.live and bu.systemd_type in ("simple", "notify"):
                    severity = "INFO" if acked else "NORMAL"
                    findings.append(Finding(
                        "pull_without_restart", severity,
                        f"{repo} ({path}) is pull-covered but backs live {bu.systemd_type} "
                        f"unit {bu.unit} with no _POST_LAND_RESTART/_POST_LAND_RESTART_USER entry",
                    ))
                    break  # one finding per clone is sufficient signal

        if findings:
            findings_by_clone[path] = findings

    return findings_by_clone


def _ack_kinds(clone_path: str) -> frozenset[str]:
    """Which finding kinds the `.deploy-unmapped-ack` marker at `clone_path`
    acknowledges (D4).

    The marker's mere *presence* — empty file, or content naming nothing —
    acknowledges only `_LEGACY_ACK_KINDS`, matching the pre-existing
    per-clone-boolean behaviour every current ack relies on
    (`reconcile_inventory`'s `acked = (Path(path) / _ACK_MARKER).exists()`).
    To additionally acknowledge `runtime_symlink_into_dev_tree` (or any future
    kind), the marker's CONTENTS must name it explicitly, one kind per line
    (blank lines and `#`-comments ignored). This is the collision guard: a
    bare legacy ack dropped to silence an unrelated `unmapped_live_clone`
    finding must never also silence a HIGH symlink-into-dev-tree finding on
    the same clone.

    Returns frozenset() if no marker exists at all.
    """
    marker = Path(clone_path) / _ACK_MARKER
    if not marker.exists():
        return frozenset()
    try:
        text = marker.read_text()
    except OSError:
        # Present but unreadable — fail open to the legacy existence-only
        # behaviour rather than silently un-acking every clone with a
        # permission-restricted marker.
        return _LEGACY_ACK_KINDS
    named = {
        line.strip() for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return _LEGACY_ACK_KINDS | named if named else _LEGACY_ACK_KINDS


# ---------------------------------------------------------------------------
# Runtime symlink into dev-tree detection (D1/D1a/D2-D7) — a runtime config
# path that resolves, through a symlink chain, into a `-working` PM
# investigation tree makes a production service's live config whatever
# branch that tree happens to have checked out. This is a distinct defect
# class from the gap findings above: it is not about an unmapped clone, it
# is about a clone silently reading a DIFFERENT clone's dev copy.
# ---------------------------------------------------------------------------

def _working_tree_root(path_str: str) -> str | None:
    """If `path_str` lies under a `-working` PM investigation tree directly
    beneath `_DEV_TREE_ROOT`, return that tree's root path. Else None.

    A "-working" tree is, by definition (D1), a direct child of
    `_DEV_TREE_ROOT` whose basename ends in "-working" — e.g.
    `/srv/git/conductor-working`, not `/srv/git/facets-working/deep/dir`'s
    grandparent-of-grandparent or some other indirect ancestor match.
    """
    try:
        rel = Path(path_str).relative_to(_DEV_TREE_ROOT)
    except ValueError:
        return None
    if not rel.parts:
        return None
    top = rel.parts[0]
    if not top.endswith("-working"):
        return None
    return str(Path(_DEV_TREE_ROOT) / top)


def _iter_symlinks(root: str):
    """Yield every symlink (file or directory) under `root`, fail-open (D5)
    on a missing root or any per-entry/per-directory probe error — logged
    and skipped, never raised.

    `followlinks=False` (the default) keeps `os.walk` from descending
    *through* a symlinked directory — it still yields the symlink itself as
    a dirnames/filenames entry, it just never recurses into its target a
    second time (avoids both duplicate scanning and infinite recursion on a
    self-referential symlinked directory).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return

    def _on_walk_error(err: OSError) -> None:
        logger.warning("[deploy-inventory] symlink walk error under %s: %s", root, err)

    for dirpath, dirnames, filenames in os.walk(root, onerror=_on_walk_error, followlinks=False):
        # Never descend into a clone's .git internals — irrelevant to runtime
        # config routing and can be enormous.
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in dirnames + filenames:
            candidate = Path(dirpath) / name
            try:
                if candidate.is_symlink():
                    yield candidate
            except OSError as e:
                logger.warning("[deploy-inventory] symlink stat failed for %s: %s", candidate, e)


def _resolve_symlink_chain(symlink_path: Path) -> tuple[list[str], str] | None:
    """Follow `symlink_path`'s symlink chain hop by hop, TRANSITIVELY (D1a).

    `Path.resolve()` would do this silently in one call; this function makes
    each hop explicit so a finding's `detail` can name the provenance chain
    (D1a's DoD requirement) and so a future refactor to a one-hop primitive
    (e.g. bare `os.readlink()`) would visibly fail this function's own tests
    rather than silently regressing.

    Returns `(hops, terminus)`: `hops` is every path visited in order,
    starting with `symlink_path` itself and then each link's resolved
    target; `terminus` is the final hop — either a real, existing,
    non-symlink path, or the first hop that lands outside `_DEV_TREE_ROOT` or
    inside a `-working` tree (D1a: both are deliberate stopping points, not
    failures — the point is the resolved *route*, not whether one specific
    leaf file exists).

    Returns None — fail-open (D5) — on an unreadable link, a dangling/broken
    terminus, a resolution loop, or a chain longer than
    `_MAX_SYMLINK_HOPS`. Never raises.
    """
    current = symlink_path
    hops = [str(symlink_path)]
    seen = {str(symlink_path)}
    dev_root = _DEV_TREE_ROOT.rstrip("/")

    for _ in range(_MAX_SYMLINK_HOPS):
        try:
            is_link = current.is_symlink()
        except OSError as e:
            logger.warning("[deploy-inventory] symlink probe failed for %s: %s", current, e)
            return None

        if not is_link:
            if not current.exists():
                logger.warning(
                    "[deploy-inventory] broken symlink chain — %s does not exist (chain: %s)",
                    current, hops,
                )
                return None  # dangling terminus — fail open, no finding
            return hops, str(current)

        try:
            raw_target = os.readlink(current)
        except OSError as e:
            logger.warning("[deploy-inventory] unreadable symlink %s: %s", current, e)
            return None

        target = Path(raw_target)
        next_path = target if target.is_absolute() else (current.parent / target)
        next_str = str(Path(os.path.normpath(str(next_path))))

        if next_str in seen:
            logger.warning(
                "[deploy-inventory] symlink resolution loop at %s (chain: %s)", next_str, hops,
            )
            return None
        seen.add(next_str)
        hops.append(next_str)

        # D1a stop conditions — leaving the dev-tree root, or landing inside
        # a `-working` tree, both terminate the chase here without an
        # existence check.
        if next_str != dev_root and not next_str.startswith(dev_root + "/"):
            return hops, next_str
        if _working_tree_root(next_str) is not None:
            return hops, next_str

        current = Path(next_str)

    logger.warning(
        "[deploy-inventory] symlink chain exceeded %d hops starting at %s — treating as a "
        "resolution failure (fail-open)", _MAX_SYMLINK_HOPS, symlink_path,
    )
    return None


def find_runtime_symlink_findings(post_land_pull: dict[str, list[str]]) -> dict[str, list[Finding]]:
    """D1/D1a/D2-D5: walk `_CLONE_ROOT_PREFIXES` for symlinks whose fully
    (transitively) resolved target lies inside a `-working` PM investigation
    tree that `_POST_LAND_PULL` does not cover. One HIGH
    `runtime_symlink_into_dev_tree` finding per such symlink (D2), keyed by
    the clone that CONTAINS the symlink (D7 — so it renders on that clone's
    own operator-board row, e.g. `/data/agents` for the two 2026-08-13
    instances), downgraded to INFO by a kind-scoped ack (D4).

    A symlink that already lives *inside* the same `-working` tree it
    resolves into is excluded — that is ordinary internal tooling plumbing
    (a venv's `bin/python` symlink, a `node_modules/.bin` shim), not a
    runtime path crossing from a deploy clone into a foreign dev tree. Without
    this exclusion the live host produces dozens of false positives from
    every `-working` tree's own virtualenv and `node_modules` (verified
    2026-08-13 against the actual host — see DoD item 8).

    Fail-open throughout (D5): every per-symlink resolution failure is
    logged and skipped by `_resolve_symlink_chain`/`_iter_symlinks`, never
    turned into a finding. Detection only (D6): no subprocess call here
    mutates anything — `_resolve_git_root` (used to attribute the "owning
    clone") issues only `git rev-parse --show-toplevel`, already in this
    module's allowed read-only verb set.
    """
    reverse_map = _reverse_pull_map(post_land_pull)  # resolved_path -> repo
    findings_by_clone: dict[str, list[Finding]] = {}

    for root in _CLONE_ROOT_PREFIXES:
        for symlink_path in _iter_symlinks(root):
            resolved = _resolve_symlink_chain(symlink_path)
            if resolved is None:
                continue
            hops, terminus = resolved

            working_tree = _working_tree_root(terminus)
            if working_tree is None:
                continue  # never landed inside a `-working` tree — not this defect

            if _working_tree_root(str(symlink_path)) == working_tree:
                continue  # symlink already lives inside the tree it points to

            if working_tree in reverse_map:
                continue  # the tree IS post-land-pulled — no defect

            owning_clone = _resolve_git_root(str(symlink_path.parent)) or str(symlink_path.parent)
            ack_kinds = _ack_kinds(owning_clone)
            severity = "INFO" if "runtime_symlink_into_dev_tree" in ack_kinds else "HIGH"

            chain_desc = " -> ".join(hops)
            detail = (
                f"{symlink_path} resolves to {terminus} (chain: {chain_desc}) inside PM "
                f"investigation tree {working_tree}, which is not present in _POST_LAND_PULL "
                f"and so is never pulled after a land"
            )
            findings_by_clone.setdefault(owning_clone, []).append(
                Finding("runtime_symlink_into_dev_tree", severity, detail)
            )

    return findings_by_clone


# ---------------------------------------------------------------------------
# Currency assertion (spec item 3) — three independent checks per clone
# ---------------------------------------------------------------------------

def _git_rev_parse_head(clone_path: str) -> str | None:
    result = _run(["git", "-C", clone_path, "rev-parse", "HEAD"])
    if result is None or result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _git_current_branch(clone_path: str) -> str | None:
    """Current branch name, or "DETACHED" for a detached HEAD (a real, known
    state per the status-JSON schema) — distinct from a probe failure, which
    returns None (indeterminate, no finding).
    """
    result = _run(["git", "-C", clone_path, "symbolic-ref", "--short", "HEAD"])
    if result is None:
        return None
    if result.returncode != 0:
        return "DETACHED"
    branch = result.stdout.strip()
    return branch or None


def _commits_behind(clone_path: str) -> int | None:
    result = _run(["git", "-C", clone_path, "rev-list", "--count", "HEAD..origin/main"])
    if result is None or result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _tree_status(clone_path: str) -> tuple[bool, bool]:
    """(tracked_dirty, untracked_present). Untracked ('??') files never count
    as tracked-dirty (spec Definitions: "clean tree" — untracked is normal
    steady state for runtime DBs/caches and must never trip the alert).
    """
    result = _run(["git", "-C", clone_path, "status", "--porcelain=v1"])
    if result is None or result.returncode != 0:
        return False, False
    tracked_dirty = False
    untracked_present = False
    for line in result.stdout.splitlines():
        if not line:
            continue
        if line[:2] == "??":
            untracked_present = True
        else:
            tracked_dirty = True
    return tracked_dirty, untracked_present


def check_currency(clone_path: str, *, fetch_done: set | None = None) -> CurrencyResult:
    """The three independent currency assertions (spec item 3). Fetches
    `origin main` once per reconcile pass per clone (memoized via `fetch_done`
    — pass the same set across every clone in one pass so none is fetched
    twice, per the exact fetch strategy in the spec).
    """
    if fetch_done is None or clone_path not in fetch_done:
        _run(["git", "-C", clone_path, "fetch", "--quiet", "origin", "main"], timeout=_FETCH_TIMEOUT_SECS)
        if fetch_done is not None:
            fetch_done.add(clone_path)

    head = _git_rev_parse_head(clone_path)
    branch = _git_current_branch(clone_path)
    commits_behind = _commits_behind(clone_path)
    tracked_dirty, untracked_present = _tree_status(clone_path)

    findings: list[Finding] = []
    if commits_behind is not None and commits_behind > 0:
        findings.append(Finding(
            "stale_behind_origin", "HIGH",
            f"{clone_path} is {commits_behind} commit(s) behind origin/main",
        ))
    if branch is not None and branch != "main":
        findings.append(Finding(
            "stray_branch", "HIGH",
            f"{clone_path} is on {branch!r}, not main",
        ))
    if tracked_dirty:
        findings.append(Finding(
            "tracked_dirty_tree", "HIGH",
            f"{clone_path} has tracked modifications (git status --porcelain)",
        ))

    return CurrencyResult(head, branch, commits_behind, tracked_dirty, untracked_present, findings)


# ---------------------------------------------------------------------------
# Stale-lock escalation (spec item 4)
# ---------------------------------------------------------------------------

def _lock_path(clone_path: str) -> Path:
    """Same naming convention as pm_core._deploy_pull_lock_path — reads the
    same lock files that path writes, without importing pm_core (avoids a
    circular import; pm_core imports this module)."""
    name = clone_path.strip("/").replace("/", "_") + ".json"
    return _LOCK_DIR / name


def check_stale_lock(clone_path: str, *, now: datetime | None = None) -> tuple[Finding | None, int | None]:
    """(finding, age_secs) for the deploy-pull-lock at `clone_path`, or
    (None, None) if no lock exists / is unreadable. A lock present but younger
    than the threshold still yields a NORMAL finding (visible, not silent);
    past the threshold it escalates to HIGH (spec item 4 — today `locked_at`
    is written but never read back).
    """
    try:
        data = json.loads(_lock_path(clone_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None, None

    locked_at_raw = data.get("locked_at")
    if not locked_at_raw:
        return None, None
    try:
        locked_at = datetime.fromisoformat(locked_at_raw)
    except ValueError:
        return None, None
    if locked_at.tzinfo is None:
        locked_at = locked_at.replace(tzinfo=timezone.utc)

    now = now or datetime.now(locked_at.tzinfo)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age = (now - locked_at).total_seconds()
    severity = "HIGH" if age >= _STALE_LOCK_THRESHOLD_SECS else "NORMAL"
    finding = Finding(
        "stale_deploy_pull_lock", severity,
        f"deploy-pull-lock at {clone_path} is {age / 3600:.1f}h old "
        f"(threshold {_STALE_LOCK_THRESHOLD_SECS / 3600:.0f}h)",
    )
    return finding, int(age)


# ---------------------------------------------------------------------------
# Full reconcile pass + status JSON (spec items 3, 6)
# ---------------------------------------------------------------------------

def _apply_ack_downgrade(findings: list[Finding], acked: bool) -> list[Finding]:
    if not acked:
        return findings
    return [
        Finding(f.kind, "INFO", f.detail) if f.kind in ("unmapped_live_clone", "pull_without_restart") else f
        for f in findings
    ]


def run_reconcile_pass(
    post_land_pull: dict[str, list[str]],
    post_land_restart: dict[str, tuple[str, ...]],
    post_land_restart_user: dict[str, tuple[str, ...]],
    *,
    now: datetime | None = None,
) -> dict:
    """One full reconcile pass: derive inventory, reconcile against the maps,
    run the three currency assertions + stale-lock check on every mapped-or-
    derived clone (spec item 3). Returns the status dict per spec item 6's
    schema. Read-only, detection-only — see module docstring.
    """
    inventory = derive_deploy_inventory()
    findings_by_clone = reconcile_inventory(inventory, post_land_pull, post_land_restart, post_land_restart_user)
    symlink_findings_by_clone = find_runtime_symlink_findings(post_land_pull)
    reverse_map = _reverse_pull_map(post_land_pull)

    all_paths = sorted(set(inventory.keys()) | set(reverse_map.keys()) | set(symlink_findings_by_clone.keys()))
    fetch_done: set = set()

    clones_out = []
    for path in all_paths:
        entry = inventory.get(path)
        repo = reverse_map.get(path)
        acked = (Path(path) / _ACK_MARKER).exists()

        currency = check_currency(path, fetch_done=fetch_done)
        lock_finding, lock_age = check_stale_lock(path, now=now)

        findings = list(findings_by_clone.get(path, []))
        findings = _apply_ack_downgrade(findings, acked)
        # Severity for runtime_symlink_into_dev_tree is already final (D4's
        # kind-scoped ack was applied inside find_runtime_symlink_findings) —
        # _apply_ack_downgrade above only ever touches _LEGACY_ACK_KINDS, so
        # these pass through untouched regardless of call order.
        findings.extend(symlink_findings_by_clone.get(path, []))
        findings.extend(currency.findings)
        if lock_finding is not None:
            findings.append(lock_finding)

        clones_out.append({
            "path": path,
            "remote": entry.remote if entry else _get_remote(path),
            "mapped": repo is not None,
            "acked": acked,
            "backing_units": [
                {"unit": bu.unit, "field": bu.field, "scope": bu.scope, "live": bu.live, "type": bu.systemd_type}
                for bu in (entry.backing_units if entry else [])
            ],
            "head": currency.head,
            "branch": currency.branch,
            "commits_behind": currency.commits_behind,
            "tracked_dirty": currency.tracked_dirty,
            "untracked_present": currency.untracked_present,
            "lock_age_secs": lock_age,
            "findings": [asdict(f) for f in findings],
        })

    return {"generated": _now_iso(now), "clones": clones_out}


def write_status_json(status: dict, path: Path | None = None) -> None:
    """Best-effort write — a write failure never blocks the reconcile pass."""
    target = path or _STATUS_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(status, indent=2, sort_keys=True))
    except OSError as e:
        logger.warning("[deploy-inventory] status file write failed: %s", e)


def read_status_json(path: Path | None = None) -> dict | None:
    """Read the previously-written status JSON.

    Returns None when no file exists yet (legitimate first-ever-pass
    bootstrap). Raises (OSError, json.JSONDecodeError) when the file is
    present but unreadable/unparseable, so callers can distinguish bootstrap
    (quiet) from corruption (loud) - lapis-pm-deploy-inventory-notify-dedup-v0
    spec item 4.
    """
    target = path or _STATUS_FILE
    if not target.exists():
        return None
    return json.loads(target.read_text())


def high_finding_keys(status: dict) -> set[tuple[str, str]]:
    """(clone_path, finding_kind) pairs for every HIGH-severity finding in
    `status` — the identity used for cross-pass Pushover dedup
    (lapis-pm-deploy-inventory-notify-dedup-v0 spec item 2).
    """
    keys: set[tuple[str, str]] = set()
    for clone in status.get("clones", []):
        for finding in clone.get("findings", []):
            if finding.get("severity") == "HIGH":
                keys.add((clone.get("path"), finding.get("kind")))
    return keys


# ---------------------------------------------------------------------------
# `lapis-pm deploy-status` rendering (spec item 6 — human affordance)
# ---------------------------------------------------------------------------

_COLOR_HIGH = "\033[31m"
_COLOR_NORMAL = "\033[33m"
_COLOR_INFO = "\033[2m"
_COLOR_AMBER_BOLD_UNDERLINE = "\033[1;4;33m"
_RESET = "\033[0m"
_COLORS = {"HIGH": _COLOR_HIGH, "NORMAL": _COLOR_NORMAL, "INFO": _COLOR_INFO}


def _worst_severity(findings: list[dict]) -> str | None:
    best = None
    for f in findings:
        sev = f.get("severity")
        if _SEVERITY_ORDER.get(sev, 3) < _SEVERITY_ORDER.get(best, 3):
            best = sev
    return best


def _worst_finding(findings: list[dict]) -> dict | None:
    """The single finding dict carrying the worst (lowest _SEVERITY_ORDER)
    severity — ties broken by first-seen order. Used to name the finding
    *kind* on the operator board (D7): a bare severity color tells Erah
    something is wrong but not which finding — e.g. `runtime_symlink_into_
    dev_tree` needs to be legible as itself, not folded into an undifferentiated
    "HIGH".
    """
    best = None
    for f in findings:
        sev = f.get("severity")
        if best is None or _SEVERITY_ORDER.get(sev, 3) < _SEVERITY_ORDER.get(best.get("severity"), 3):
            best = f
    return best


def _truncate_units(backing_units: list[dict], width: int = 30) -> str:
    names = [bu["unit"] for bu in backing_units]
    joined = ", ".join(names)
    if len(joined) <= width:
        return joined
    shown: list[str] = []
    total = 0
    for name in names:
        if total + len(name) > width - 3:
            break
        shown.append(name)
        total += len(name) + 2
    remaining = len(names) - len(shown)
    return ", ".join(shown) + (f" +{remaining} more" if remaining > 0 else "")


def render_deploy_status(status: dict) -> str:
    """One row per clone, sorted by worst-finding severity (HIGH > NORMAL >
    INFO > none) then path — per spec item 6's rendering spec.
    """
    clones = status.get("clones", [])
    rows = sorted(clones, key=lambda c: (_SEVERITY_ORDER.get(_worst_severity(c.get("findings", [])), 3), c["path"]))

    lines = [f"generated: {status.get('generated', '?')}", ""]
    header = f"{'CLONE':<24} {'BACKING UNIT(S)':<32} {'BEHIND':>7} {'BRANCH-OK':>10} {'CLEAN':>7} {'MAPPED':>7}  WORST-FINDING"
    lines.append(header)
    lines.append("-" * len(header))

    for c in rows:
        name = Path(c["path"]).name
        units = _truncate_units(c.get("backing_units", []))
        behind = c.get("commits_behind")
        behind_repr = str(behind) if behind is not None else "?"
        branch_ok = "yes" if c.get("branch") == "main" else "no"
        clean_repr = "no" if c.get("tracked_dirty") else "yes"
        if c.get("untracked_present"):
            clean_repr = f"{_COLOR_AMBER_BOLD_UNDERLINE}{clean_repr}{_RESET}"
        mapped = "yes" if c.get("mapped") else "no"
        worst_finding = _worst_finding(c.get("findings", []))
        worst = worst_finding.get("severity") if worst_finding else None
        color = _COLORS.get(worst, "")
        label = f"{worst} {worst_finding.get('kind', '?')}" if worst_finding else "-"
        worst_repr = f"{color}{label}{_RESET}" if color else label
        lines.append(
            f"{name:<24} {units:<32} {behind_repr:>7} {branch_ok:>10} {clean_repr:>7} {mapped:>7}  {worst_repr}"
        )

    return "\n".join(lines)
