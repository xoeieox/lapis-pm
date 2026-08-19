"""Tests for the land-time debt-ledger close (spec: debt-ledger-merge-flip-v0).

Coverage:
  T1  - flip: open entry -> resolved, fields preserved, harvest-shape (debt-open
        tag bucket) no longer returns the key.
  T2  - non-bundle tid -> False, zero writes.
  T2b - regex boundary: substring / 9-hex / 11-hex tids never match (anchored
        full-string only).
  T3  - missing ledger key -> False, warning logged, no exception.
  T4  - no-clobber: already-invalid entry is left byte-identical.
  T5  - store.set failure never raises.
  T6  - choke point: the close-then-clear_landed_state pairing used at each
        land call site flips a bundle-item entry; a non-bundle tid through
        the same pairing writes nothing to the ledger.

Uses the real MemoryStore against the isolated MEM_DB_PATH the root
conftest.py pins for the whole test session (conftest.py:28-38) — the same
store pm_core._mem() / node_identity.writable_store() resolve to.
"""
from __future__ import annotations

import yaml
import pytest

from lapis_pm import pm_core
from lapis_pm import node_identity


def _seed_debt(store, repo: str, debt_id: str, **extra) -> str:
    key = f"review/debt/{repo}/{debt_id}"
    body = {
        "status": "open",
        "note": "some finding",
        "paths": ["a/b.py"],
        "opened_at": "2026-06-01T00:00:00Z",
        "source_pr": 100,
        **extra,
    }
    store.set(
        key,
        yaml.safe_dump(body, sort_keys=False),
        tags=["review-debt", f"repo-{repo}", f"debt-{body['status']}"],
        source="code-reviewer-webhook",
    )
    return key


@pytest.fixture()
def store():
    return node_identity.writable_store()


# ---------------------------------------------------------------------------
# T1 — flip
# ---------------------------------------------------------------------------

def test_t1_flip_resolves_open_entry(store):
    key = _seed_debt(store, "fakerepo", "aabbccdde1")
    tid = "cr-bundle-item-fakerepo-aabbccdde1"

    result = pm_core.close_resolved_debt_for_target(
        tid, resolved_by_pr=42, ground="auto-land pr=42",
    )
    assert result is True

    rec = store.get(key)
    body = yaml.safe_load(rec["content"])
    assert body["status"] == "resolved"
    assert body["resolved_by_pr"] == 42
    assert body["resolved_at"]
    assert body["resolved_ground"] == "auto-land pr=42"
    # original fields preserved
    assert body["note"] == "some finding"
    assert body["paths"] == ["a/b.py"]
    assert body["opened_at"] == "2026-06-01T00:00:00Z"
    assert body["source_pr"] == 100

    tags = set(rec["tags"].split(","))
    assert tags == {"review-debt", "repo-fakerepo", "debt-resolved"}
    assert rec["source"] == "lapis-pm-land-close"

    # Harvest-shape read: the debt-open bucket no longer contains the key.
    open_bucket = store.list_all(tag="debt-open", limit=100)
    assert key not in {r["key"] for r in open_bucket}


# ---------------------------------------------------------------------------
# T2 / T2b — non-bundle / regex boundary
# ---------------------------------------------------------------------------

def test_t2_non_bundle_tid_is_noop(store):
    assert pm_core.close_resolved_debt_for_target("some-target-v0", ground="x") is False


@pytest.mark.parametrize("tid", [
    "xxcr-bundle-item-fakerepo-aabbccdde1yy",   # substring, not full match
    "cr-bundle-item-fakerepo-aabbccdde",        # 9-hex suffix
    "cr-bundle-item-fakerepo-aabbccdde1f",      # 11-hex suffix
])
def test_t2b_regex_boundary_rejects_non_anchored_tids(store, tid):
    _seed_debt(store, "fakerepo", "aabbccdde1")
    assert pm_core.close_resolved_debt_for_target(tid, ground="x") is False
    # zero writes: the seeded entry stays open/untouched
    rec = store.get("review/debt/fakerepo/aabbccdde1")
    assert yaml.safe_load(rec["content"])["status"] == "open"


# ---------------------------------------------------------------------------
# T3 — missing key
# ---------------------------------------------------------------------------

def test_t3_missing_key_returns_false_no_raise(store, caplog):
    tid = "cr-bundle-item-ghostrepo-0123456789"
    result = pm_core.close_resolved_debt_for_target(tid, ground="x")
    assert result is False


# ---------------------------------------------------------------------------
# T4 — no-clobber
# ---------------------------------------------------------------------------

def test_t4_no_clobber_invalid_entry(store):
    key = _seed_debt(
        store, "fakerepo", "ffeeddcc11",
        status="invalid", invalid_ground="salvage: superseded",
    )
    before = store.get(key)

    tid = "cr-bundle-item-fakerepo-ffeeddcc11"
    result = pm_core.close_resolved_debt_for_target(tid, ground="auto-land pr=1")
    assert result is False

    after = store.get(key)
    assert after["content"] == before["content"]
    assert after["source"] == before["source"]


# ---------------------------------------------------------------------------
# T5 — store failure never raises
# ---------------------------------------------------------------------------

def test_t5_store_set_failure_never_raises(store, monkeypatch):
    key = _seed_debt(store, "fakerepo", "aa11bb22cc")
    tid = "cr-bundle-item-fakerepo-aa11bb22cc"

    def _boom(*a, **kw):
        raise RuntimeError("simulated store failure")

    monkeypatch.setattr(node_identity, "writable_store", lambda path=None: _FailingStore(store, _boom))

    result = pm_core.close_resolved_debt_for_target(tid, ground="auto-land pr=1")
    assert result is False


class _FailingStore:
    """Wraps a real store but raises on .set(), for T5."""

    def __init__(self, inner, boom):
        self._inner = inner
        self._boom = boom

    def get(self, key):
        return self._inner.get(key)

    def set(self, *a, **kw):
        self._boom()


# ---------------------------------------------------------------------------
# T6 — choke point: close-then-clear_landed_state pairing
# ---------------------------------------------------------------------------

def test_t6_choke_point_flips_bundle_item_via_land_pairing(store):
    key = _seed_debt(store, "fakerepo", "12ab34cd56")
    tid = "cr-bundle-item-fakerepo-12ab34cd56"

    # Mirrors the pattern used at each of the three land call sites: the
    # flip fires immediately before clear_landed_state.
    pm_core.close_resolved_debt_for_target(tid, resolved_by_pr=7, ground="auto-land pr=7")
    summary = pm_core.clear_landed_state(tid)
    assert isinstance(summary, dict)

    rec = store.get(key)
    body = yaml.safe_load(rec["content"])
    assert body["status"] == "resolved"
    assert body["resolved_by_pr"] == 7


def test_t6_choke_point_non_bundle_tid_writes_nothing(store):
    tid = "some-other-target-v0"

    result = pm_core.close_resolved_debt_for_target(tid, ground="auto-land pr=7")
    pm_core.clear_landed_state(tid)

    assert result is False
    # No review/debt/* key was ever created for a non-bundle tid.
    assert store.get(f"review/debt/{tid}") is None
