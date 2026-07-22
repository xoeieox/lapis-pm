"""Deploy inventory reconciler (lapis-pm-deploy-inventory-reconciler-v0).

Derives the actual deployment surface from the live host — systemd units
(system + --user) and editable pip installs — and reconciles it against the
hand-maintained `_POST_LAND_PULL` / `_POST_LAND_RESTART` maps in `pm_core`.
The maps are allowlists a human must remember to update; nothing previously
compared them against what is actually running. This module is the
comparison.

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
    reverse_map = _reverse_pull_map(post_land_pull)

    all_paths = sorted(set(inventory.keys()) | set(reverse_map.keys()))
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
        worst = _worst_severity(c.get("findings", []))
        color = _COLORS.get(worst, "")
        worst_repr = f"{color}{worst or '-'}{_RESET}" if color else (worst or "-")
        lines.append(
            f"{name:<24} {units:<32} {behind_repr:>7} {branch_ok:>10} {clean_repr:>7} {mapped:>7}  {worst_repr}"
        )

    return "\n".join(lines)
