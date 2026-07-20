"""Node identity resolution and structural read/write ownership boundaries.

Every lapis-pm process resolves its node identity exactly once at start (CLI
dispatch in cli.py, the daemon tick loop, any standalone runner) via
resolve_node_identity(). Two fail-closed tiers distinguish a hard safety
violation from an accidental misconfiguration -- neither ever runs, neither
is a resumable pause:

  - NodeIdentityViolation ("treason"): an independent node's config points at
    a foreign/shared store, or the node attempts to forward writes to master.
  - NodeConfigError: the owned store path is missing/invalid, or node_role is
    not a recognized literal -- a legible "fix your config" refusal.

A valid-but-empty owned store is a normal fresh-node state and starts fine.

writable_store() is the only sanctioned way lapis_pm/ code obtains a writable
MemoryStore. peer_store() returns a read-only handle onto a declared peer
store. See tests/test_node_identity_bypass_guard.py for the CI-enforced
perimeter (I7) and its documented residual gap: runtime monkey-patching,
dynamic imports of rogue modules, and a compromised agents-core are outside
this target's threat model (Erah's ruling, 2026-07-19 -- a runtime
import-hook was considered and declined as disproportionate/fragile).

node_role == master (BRIX) reproduces today's behavior exactly, including
honoring MEM_MASTER_URL -- every gate in this module is a no-op for master.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class NodeIdentityViolation(Exception):
    """Hard safety violation: an independent node's config points at, or an
    action targets, a foreign/shared store, controller, Forgejo, or queue
    root -- or the node attempts to forward writes to master. Fail-closed:
    the caller must never proceed past this exception."""


class NodeConfigError(Exception):
    """Accidental misconfiguration: missing/invalid owned-store path or an
    unrecognized node_role literal. Distinct from NodeIdentityViolation --
    a legible "fix your config" refusal, equally fail-closed."""


_VALID_ROLES = ("master", "independent")


@dataclass(frozen=True)
class NodeIdentity:
    node_role: str
    owned_mem_store: str
    readonly_peer_stores: tuple
    gw_flip_authorized: bool
    owned_forgejo: str
    owned_queue_root: str


_IDENTITY: Optional[NodeIdentity] = None


def _bool_val(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _load_profile() -> dict:
    """Optional node-profile YAML, second in the precedence order (below
    explicit env vars, above safe defaults). Missing/unreadable -> empty."""
    path = os.environ.get("LAPIS_PM_NODE_PROFILE")
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    import yaml
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception as e:
        logger.warning("node_identity: failed to load LAPIS_PM_NODE_PROFILE %s: %s", path, e)
        return {}


def _field(env_var: str, profile: dict, key: str, default):
    val = os.environ.get(env_var)
    if val is not None and val != "":
        return val
    if key in profile and profile[key] not in (None, ""):
        return profile[key]
    return default


def _default_forgejo() -> str:
    try:
        from agents_core.forgejo import FORGEJO_URL
        return FORGEJO_URL
    except Exception:
        return os.environ.get("FORGEJO_URL", "http://203.0.113.10:3000")


def _default_queue_root() -> str:
    return os.environ.get("ROOM_ROOT", "/room")


def _resolve() -> NodeIdentity:
    profile = _load_profile()

    node_role = str(_field("LAPIS_PM_NODE_ROLE", profile, "node_role", "independent")).strip()

    owned_mem_store = str(_field(
        "LAPIS_PM_OWNED_MEM_STORE", profile, "owned_mem_store",
        os.environ.get("MEM_DB_PATH", ""),
    )).strip()

    peer_raw = _field("LAPIS_PM_READONLY_PEER_STORES", profile, "readonly_peer_stores", "")
    if isinstance(peer_raw, (list, tuple)):
        readonly_peer_stores = tuple(str(p).strip() for p in peer_raw if str(p).strip())
    else:
        readonly_peer_stores = tuple(p.strip() for p in str(peer_raw).split(",") if p.strip())

    gw_flip_authorized = _bool_val(
        _field("LAPIS_PM_GW_FLIP_AUTHORIZED", profile, "gw_flip_authorized", False)
    )

    owned_forgejo = str(_field(
        "LAPIS_PM_OWNED_FORGEJO", profile, "owned_forgejo", _default_forgejo()
    )).strip()

    owned_queue_root = str(_field(
        "LAPIS_PM_OWNED_QUEUE_ROOT", profile, "owned_queue_root", _default_queue_root()
    )).strip()

    return NodeIdentity(
        node_role=node_role,
        owned_mem_store=owned_mem_store,
        readonly_peer_stores=readonly_peer_stores,
        gw_flip_authorized=gw_flip_authorized,
        owned_forgejo=owned_forgejo,
        owned_queue_root=owned_queue_root,
    )


def _path_under(child: Path, ancestor: Path) -> bool:
    try:
        child.relative_to(ancestor)
        return True
    except ValueError:
        return False


def _validate_startup(identity: NodeIdentity) -> None:
    """Two-tier startup invariant. Order matters: role/presence (Tier 2) must
    be legible before a foreign-store (Tier 1) comparison can even be made."""
    if identity.node_role not in _VALID_ROLES:
        raise NodeConfigError(
            f"LAPIS_PM_NODE_ROLE={identity.node_role!r} is not a recognized role "
            f"(expected one of {_VALID_ROLES}); set LAPIS_PM_NODE_ROLE to 'master' "
            "or 'independent'"
        )

    if not identity.owned_mem_store:
        raise NodeConfigError(
            "owned_mem_store is unset -- set LAPIS_PM_OWNED_MEM_STORE (or MEM_DB_PATH) "
            "to a writable path this node owns"
        )

    owned_path = Path(identity.owned_mem_store)
    parent = owned_path.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise NodeConfigError(
            f"owned_mem_store parent directory {parent} is missing or unwritable -- "
            "set LAPIS_PM_OWNED_MEM_STORE to a writable path this node owns"
        )

    if identity.node_role == "independent":
        owned_resolved = owned_path.resolve()
        for peer in identity.readonly_peer_stores:
            peer_resolved = Path(peer).resolve()
            if owned_resolved == peer_resolved or _path_under(owned_resolved, peer_resolved):
                raise NodeIdentityViolation(
                    f"owned_mem_store {owned_resolved} resolves under declared peer "
                    f"store {peer_resolved} -- an independent node must never own a "
                    "path inside a foreign/shared store"
                )
        if os.environ.get("MEM_MASTER_URL"):
            raise NodeIdentityViolation(
                "MEM_MASTER_URL is set but node_role=independent -- an independent "
                "node never forwards writes to master; unset MEM_MASTER_URL or set "
                "LAPIS_PM_NODE_ROLE=master if this node is BRIX"
            )


def resolve_node_identity(force: bool = False) -> NodeIdentity:
    """Resolve (and cache) this process's node identity. Idempotent after the
    first successful call -- safe to call from every entry point and from
    every gate in this module."""
    global _IDENTITY
    if _IDENTITY is not None and not force:
        return _IDENTITY
    identity = _resolve()
    _validate_startup(identity)
    _IDENTITY = identity
    return identity


def _reset_for_tests() -> None:
    """Test-only: clear the cached identity so a test can re-resolve under a
    patched env. Not part of the public interface."""
    global _IDENTITY
    _IDENTITY = None


# ---------------------------------------------------------------------------
# 1. writable_store() / ReadOnlyPeerStore / peer_store() -- the structural
#    read/write separation (Design §2).
# ---------------------------------------------------------------------------

def writable_store(path: str | None = None):
    """Return a writable MemoryStore bound to this node's owned_mem_store.

    The only sanctioned way to obtain a writable MemoryStore in lapis_pm/.
    A request for any other path raises NodeIdentityViolation -- it never
    returns a usable object.
    """
    identity = resolve_node_identity()
    owned_resolved = Path(identity.owned_mem_store).resolve()
    if path is not None:
        requested = Path(path).resolve()
        if requested != owned_resolved:
            raise NodeIdentityViolation(
                f"writable_store requested for {requested}, which is not this "
                f"node's owned_mem_store ({owned_resolved})"
            )
    from agents_core.mem import MemoryStore
    return MemoryStore(owned_resolved)


class ReadOnlyPeerStore:
    """Read-only view onto a declared peer store.

    NOT a MemoryStore subclass -- holds the inner store only via private
    (name-mangled) composition and exposes only get/search/list_all. There is
    no public attribute path to the inner object, so this cannot be cast to
    or used as a writable store.
    """

    def __init__(self, path: Path):
        from agents_core.mem import MemoryStore
        self.__inner = MemoryStore(path)

    def get(self, key: str):
        return self.__inner.get(key)

    def search(self, query: str, tag: str = "", limit: int = 20):
        return self.__inner.search(query, tag=tag, limit=limit)

    def list_all(self, tag: str = "", tags=None, since: str = "", limit: int = 50):
        return self.__inner.list_all(tag=tag, tags=tags, since=since, limit=limit)


def peer_store(path: str) -> ReadOnlyPeerStore:
    """Return a ReadOnlyPeerStore for `path` only if it is a declared
    readonly_peer_stores entry; otherwise raise NodeIdentityViolation."""
    identity = resolve_node_identity()
    requested = Path(path).resolve()
    for peer in identity.readonly_peer_stores:
        peer_resolved = Path(peer).resolve()
        if requested == peer_resolved:
            return ReadOnlyPeerStore(peer_resolved)
    raise NodeIdentityViolation(
        f"peer_store requested for {requested}, which is not a declared "
        f"readonly_peer_stores entry ({list(identity.readonly_peer_stores)})"
    )


# ---------------------------------------------------------------------------
# 2. GW flip authorization (Design §3b).
# ---------------------------------------------------------------------------

def ensure_gw_flip_authorized(controller_url: str) -> None:
    """Raise NodeIdentityViolation unless this node is authorized to flip the
    physical GravityWell controller. gw_flip_authorized is the single flag
    modeling ownership of the (singular, in-scope) flip controller -- there
    is no separate per-URL ownership list in this spec."""
    identity = resolve_node_identity()
    if identity.node_role == "master":
        return
    if not identity.gw_flip_authorized:
        raise NodeIdentityViolation(
            f"GW flip refused: node_role={identity.node_role} is not authorized "
            f"to flip {controller_url} -- set LAPIS_PM_GW_FLIP_AUTHORIZED=true "
            "only on the node that physically owns the GravityWell flip controller"
        )


# ---------------------------------------------------------------------------
# 3. Forgejo merge / branch-delete ownership (Design §3c).
# ---------------------------------------------------------------------------

def _same_host(url_a: str, url_b: str) -> bool:
    return urlparse(url_a).netloc == urlparse(url_b).netloc


def ensure_owned_forgejo(forgejo_url: str) -> None:
    """Raise NodeIdentityViolation unless this node owns `forgejo_url` (i.e.
    node_role == master and the URL matches the resolved owned_forgejo)."""
    identity = resolve_node_identity()
    if identity.node_role == "master" and _same_host(forgejo_url, identity.owned_forgejo):
        return
    raise NodeIdentityViolation(
        f"Forgejo merge/branch-delete refused: node_role={identity.node_role} does "
        f"not own {forgejo_url} (owned_forgejo={identity.owned_forgejo}) -- only "
        "node_role=master may merge PRs or delete branches on its owned Forgejo"
    )


# ---------------------------------------------------------------------------
# 4. Dispatch / queue / working-clone ownership (Design §3d).
# ---------------------------------------------------------------------------

def ensure_dispatch_target_owned(*paths: str) -> None:
    """Raise NodeIdentityViolation if any path in `paths` (a queue dir or a
    fixer working-clone dir) is not owned by this node.

    No-op for node_role == master (I6). For an independent node: refuses a
    path under a declared peer store, and refuses any path outside
    owned_queue_root (which defaults to ROOM_ROOT, so the working-clone tree
    under /srv/git needs an explicit LAPIS_PM_OWNED_QUEUE_ROOT on a real
    independent deployment -- fail-closed until configured, matching the
    gw_flip_authorized default-false posture).
    """
    identity = resolve_node_identity()
    if identity.node_role == "master":
        return
    owned_root = Path(identity.owned_queue_root).resolve()
    for raw in paths:
        resolved = Path(raw).resolve()
        for peer in identity.readonly_peer_stores:
            peer_resolved = Path(peer).resolve()
            if resolved == peer_resolved or _path_under(resolved, peer_resolved):
                raise NodeIdentityViolation(
                    f"dispatch target {resolved} resolves under declared peer store "
                    f"{peer_resolved} -- refusing to dispatch/clone into a foreign tree"
                )
        if owned_root != Path("/") and resolved != owned_root and not _path_under(resolved, owned_root):
            raise NodeIdentityViolation(
                f"dispatch target {resolved} is outside owned_queue_root "
                f"({owned_root}) -- an independent node may not dispatch into or "
                "clone within a queue/working tree it does not own"
            )
