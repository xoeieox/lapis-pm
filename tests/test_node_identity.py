"""Tests for lapis_pm.node_identity — node-owned write boundaries + non-interference
guard (spec: lapis-pm-node-write-ownership-v0).

Coverage:
  resolve_node_identity:
  - defaults to independent + owns-nothing when no config is present
  - env vars take precedence over a node-profile YAML
  - profile YAML is used when the matching env var is absent
  Two-tier startup invariant:
  - Tier 1 (NodeIdentityViolation): owned_mem_store resolves under a declared peer store
  - Tier 1 (NodeIdentityViolation): MEM_MASTER_URL set while node_role=independent
  - Tier 2 (NodeConfigError): owned_mem_store unset
  - Tier 2 (NodeConfigError): owned_mem_store parent dir missing
  - Tier 2 (NodeConfigError): node_role not a recognized literal
  - a valid-but-empty owned store starts normally
  - node_role=master reproduces today's behavior, including MEM_MASTER_URL
  writable_store / ReadOnlyPeerStore / peer_store:
  - writable_store() with no path returns a store bound to owned_mem_store
  - writable_store(path) for a non-owned path raises NodeIdentityViolation
  - ReadOnlyPeerStore exposes only get/search/list_all, no write surface
  - peer_store() returns a working reader for a declared peer path
  - peer_store() raises for an undeclared path
  Gates:
  - ensure_gw_flip_authorized: refuses when unauthorized, no-ops for master
  - ensure_owned_forgejo: refuses for non-master / mismatched host, allows master+match
  - ensure_dispatch_target_owned: no-ops for master; refuses foreign/out-of-root paths
    for independent; allows an in-root path for independent
"""
from __future__ import annotations

import os

import pytest

from lapis_pm import node_identity


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "LAPIS_PM_NODE_ROLE",
        "LAPIS_PM_OWNED_MEM_STORE",
        "LAPIS_PM_READONLY_PEER_STORES",
        "LAPIS_PM_GW_FLIP_AUTHORIZED",
        "LAPIS_PM_OWNED_FORGEJO",
        "LAPIS_PM_OWNED_QUEUE_ROOT",
        "LAPIS_PM_NODE_PROFILE",
        "MEM_MASTER_URL",
        "ROOM_ROOT",
        "FORGEJO_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    node_identity._reset_for_tests()
    yield
    node_identity._reset_for_tests()


# ---------------------------------------------------------------------------
# resolve_node_identity — precedence + defaults
# ---------------------------------------------------------------------------

def test_defaults_to_independent_owns_nothing(monkeypatch, tmp_path):
    mem_path = tmp_path / "mem.db"
    monkeypatch.setenv("MEM_DB_PATH", str(mem_path))
    identity = node_identity.resolve_node_identity()
    assert identity.node_role == "independent"
    assert identity.readonly_peer_stores == ()
    assert identity.gw_flip_authorized is False
    assert identity.owned_queue_root == "/room"


def test_env_var_takes_precedence_over_profile(monkeypatch, tmp_path):
    mem_path = tmp_path / "mem.db"
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("node_role: master\n")
    monkeypatch.setenv("LAPIS_PM_NODE_PROFILE", str(profile_path))
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(mem_path))
    identity = node_identity.resolve_node_identity()
    assert identity.node_role == "independent"


def test_profile_yaml_used_when_env_absent(monkeypatch, tmp_path):
    mem_path = tmp_path / "mem.db"
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        f"node_role: master\nowned_mem_store: {mem_path}\n"
    )
    monkeypatch.setenv("LAPIS_PM_NODE_PROFILE", str(profile_path))
    identity = node_identity.resolve_node_identity()
    assert identity.node_role == "master"
    assert identity.owned_mem_store == str(mem_path)


def test_resolve_node_identity_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("MEM_DB_PATH", str(tmp_path / "mem.db"))
    first = node_identity.resolve_node_identity()
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "master")
    second = node_identity.resolve_node_identity()
    assert second is first
    third = node_identity.resolve_node_identity(force=True)
    assert third.node_role == "master"


# ---------------------------------------------------------------------------
# Two-tier startup invariant
# ---------------------------------------------------------------------------

def test_tier1_foreign_store_raises_node_identity_violation(monkeypatch, tmp_path):
    peer_dir = tmp_path / "peer"
    peer_dir.mkdir()
    owned = peer_dir / "mem.db"  # lives *inside* the declared peer store
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(owned))
    monkeypatch.setenv("LAPIS_PM_READONLY_PEER_STORES", str(peer_dir))
    with pytest.raises(node_identity.NodeIdentityViolation):
        node_identity.resolve_node_identity()


def test_tier1_mem_master_url_set_on_independent_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    monkeypatch.setenv("MEM_MASTER_URL", "http://203.0.113.10:8404")
    with pytest.raises(node_identity.NodeIdentityViolation):
        node_identity.resolve_node_identity()


def test_tier2_unset_owned_store_raises_node_config_error(monkeypatch):
    monkeypatch.setenv("MEM_DB_PATH", "")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", "")
    with pytest.raises(node_identity.NodeConfigError):
        node_identity.resolve_node_identity()


def test_tier2_missing_parent_dir_raises_node_config_error(monkeypatch, tmp_path):
    missing = tmp_path / "does-not-exist" / "mem.db"
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(missing))
    with pytest.raises(node_identity.NodeConfigError):
        node_identity.resolve_node_identity()


def test_tier2_bad_role_literal_raises_node_config_error(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "starhouse")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    with pytest.raises(node_identity.NodeConfigError):
        node_identity.resolve_node_identity()


def test_valid_empty_owned_store_starts_normally(monkeypatch, tmp_path):
    """A fresh node whose owned store file doesn't exist yet (parent dir does) starts fine."""
    fresh = tmp_path / "mem.db"
    assert not fresh.exists()
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(fresh))
    identity = node_identity.resolve_node_identity()
    assert identity.owned_mem_store == str(fresh)


def test_master_role_honors_mem_master_url_without_new_failure(monkeypatch, tmp_path):
    """I6: node_role=master reproduces today's behavior, including MEM_MASTER_URL."""
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "master")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    monkeypatch.setenv("MEM_MASTER_URL", "http://203.0.113.10:8404")
    identity = node_identity.resolve_node_identity()
    assert identity.node_role == "master"


# ---------------------------------------------------------------------------
# writable_store / ReadOnlyPeerStore / peer_store
# ---------------------------------------------------------------------------

def test_writable_store_binds_to_owned_mem_store(monkeypatch, tmp_path):
    owned = tmp_path / "mem.db"
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(owned))
    store = node_identity.writable_store()
    store.set("pattern/x", "hello", tags=["t"])
    assert store.get("pattern/x")["content"] == "hello"
    store.close()


def test_writable_store_rejects_non_owned_path(monkeypatch, tmp_path):
    owned = tmp_path / "mem.db"
    other = tmp_path / "other.db"
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(owned))
    with pytest.raises(node_identity.NodeIdentityViolation):
        node_identity.writable_store(str(other))


def test_readonly_peer_store_exposes_only_read_surface(tmp_path):
    inner_path = tmp_path / "peer.db"
    peer = node_identity.ReadOnlyPeerStore(inner_path)
    assert hasattr(peer, "get")
    assert hasattr(peer, "search")
    assert hasattr(peer, "list_all")
    assert not hasattr(peer, "set")
    assert not hasattr(peer, "delete")
    # No public attribute path to the inner MemoryStore.
    assert not any(a for a in vars(peer) if not a.startswith("_ReadOnlyPeerStore__"))


def test_peer_store_returns_working_reader_for_declared_path(monkeypatch, tmp_path):
    owned = tmp_path / "mine.db"
    peer_path = tmp_path / "peer.db"
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(owned))
    monkeypatch.setenv("LAPIS_PM_READONLY_PEER_STORES", str(peer_path))

    # Seed the peer store directly via the writable factory bound to that path...
    from agents_core.mem import MemoryStore
    seed = MemoryStore(peer_path)
    seed.set("decision/from-peer", "peer content")
    seed.close()

    reader = node_identity.peer_store(str(peer_path))
    result = reader.get("decision/from-peer")
    assert result["content"] == "peer content"


def test_peer_store_raises_for_undeclared_path(monkeypatch, tmp_path):
    owned = tmp_path / "mine.db"
    undeclared = tmp_path / "not-declared.db"
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(owned))
    monkeypatch.setenv("LAPIS_PM_READONLY_PEER_STORES", "")
    with pytest.raises(node_identity.NodeIdentityViolation):
        node_identity.peer_store(str(undeclared))


# ---------------------------------------------------------------------------
# ensure_gw_flip_authorized
# ---------------------------------------------------------------------------

def test_gw_flip_refused_when_unauthorized(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    with pytest.raises(node_identity.NodeIdentityViolation):
        node_identity.ensure_gw_flip_authorized("http://203.0.113.10:8408")


def test_gw_flip_allowed_when_authorized(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    monkeypatch.setenv("LAPIS_PM_GW_FLIP_AUTHORIZED", "true")
    node_identity.ensure_gw_flip_authorized("http://203.0.113.10:8408")  # does not raise


def test_gw_flip_noop_for_master(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "master")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    node_identity.ensure_gw_flip_authorized("http://203.0.113.10:8408")  # does not raise


# ---------------------------------------------------------------------------
# ensure_owned_forgejo
# ---------------------------------------------------------------------------

def test_owned_forgejo_refused_for_independent(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    with pytest.raises(node_identity.NodeIdentityViolation):
        node_identity.ensure_owned_forgejo("http://203.0.113.10:3000")


def test_owned_forgejo_refused_for_master_with_mismatched_host(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "master")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    monkeypatch.setenv("LAPIS_PM_OWNED_FORGEJO", "http://203.0.113.10:3000")
    with pytest.raises(node_identity.NodeIdentityViolation):
        node_identity.ensure_owned_forgejo("http://10.0.0.9:3000")


def test_owned_forgejo_allowed_for_master_with_matching_host(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "master")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    monkeypatch.setenv("LAPIS_PM_OWNED_FORGEJO", "http://203.0.113.10:3000")
    node_identity.ensure_owned_forgejo("http://203.0.113.10:3000")  # does not raise


# ---------------------------------------------------------------------------
# ensure_dispatch_target_owned
# ---------------------------------------------------------------------------

def test_dispatch_target_noop_for_master(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "master")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    node_identity.ensure_dispatch_target_owned("/srv/git/anything-working")  # does not raise


def test_dispatch_target_refused_outside_owned_root_for_independent(monkeypatch, tmp_path):
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    monkeypatch.setenv("LAPIS_PM_OWNED_QUEUE_ROOT", str(tmp_path / "room"))
    with pytest.raises(node_identity.NodeIdentityViolation):
        node_identity.ensure_dispatch_target_owned("/srv/git/foreign-working")


def test_dispatch_target_refused_under_peer_store_for_independent(monkeypatch, tmp_path):
    peer_dir = tmp_path / "peer-tree"
    peer_dir.mkdir()
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    monkeypatch.setenv("LAPIS_PM_READONLY_PEER_STORES", str(peer_dir))
    monkeypatch.setenv("LAPIS_PM_OWNED_QUEUE_ROOT", "/")
    with pytest.raises(node_identity.NodeIdentityViolation):
        node_identity.ensure_dispatch_target_owned(str(peer_dir / "repo-working"))


def test_dispatch_target_allowed_inside_owned_root_for_independent(monkeypatch, tmp_path):
    owned_root = tmp_path / "room"
    owned_root.mkdir()
    monkeypatch.setenv("LAPIS_PM_NODE_ROLE", "independent")
    monkeypatch.setenv("LAPIS_PM_OWNED_MEM_STORE", str(tmp_path / "mem.db"))
    monkeypatch.setenv("LAPIS_PM_OWNED_QUEUE_ROOT", str(owned_root))
    node_identity.ensure_dispatch_target_owned(str(owned_root / "gpu-queue"))  # does not raise
