"""Tests for the PM→Zephyr slot-close + deposit seam (gate-5).

Coverage:
  - Happy path: processed dispatch → slot transitions to 'landed' + deposit row with
    correct provenance including non-null model.
  - Failure path: failed dispatch → slot transitions to 'abandoned' + deposit still
    emitted with correct provenance.
  - Idempotency (pending-guard): pending status is not in _SLOT_STATUS_MAP → no-op.
  - Manifest-hash determinism: same inputs produce the same hash, so INSERT OR IGNORE
    correctly deduplicates on a second completion call.
  - Best-effort posture: SlotStore failure does not raise or block deposit.
  - Best-effort posture: recorder failure does not raise or block caller.
  - Shaper model lookup failure falls back to model=None without crashing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agents_core.slots  # ensure module is in sys.modules before any patching
from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dispatch_record(status: str, agent_type: str = "fixer") -> dict:
    return {
        "gpu_id": "task-abc",
        "spec_id": "spec-xyz",
        "agent_type": agent_type,
        "intent": "implement the thing",
        "repo": "lapis-pm",
        "ts": "2026-06-02T10:00:00-07:00",
        "status": status,
        "completed_at": "2026-06-02T11:00:00-07:00",
        "retry_count": 0,
    }


def _shaper_mock(model: str = "sonnet") -> MagicMock:
    agent_def = MagicMock()
    agent_def.model = model
    shaper = MagicMock()
    shaper.get_agent.return_value = agent_def
    return shaper


def _make_recorder_module(recorder_instance: MagicMock) -> MagicMock:
    """Return a mock module whose get_recorder() returns recorder_instance."""
    mod = MagicMock()
    mod.get_recorder.return_value = recorder_instance
    return mod


# Patch order matters: SlotStore must be patched before importlib.import_module
# is patched, because patch() itself uses importlib internally to resolve the
# target module. Applying the importlib patch first causes the SlotStore patch
# to target the wrong object.
_SLOT_STORE_PATH = "agents_core.slots.SlotStore"
_IMPORTLIB_PATH = "lapis_pm.pm_core.importlib.import_module"


# ---------------------------------------------------------------------------
# Happy path: processed → landed + correct provenance with non-null model
# ---------------------------------------------------------------------------

def test_happy_path_landed_with_non_null_model():
    """Processed dispatch closes slot as 'landed' and deposits provenance with model."""
    rec = _dispatch_record("processed", agent_type="fixer")
    recorder_instance = MagicMock()
    slot_store_instance = MagicMock()

    with patch(_SLOT_STORE_PATH, return_value=slot_store_instance), \
         patch.object(pm_core, "_SHAPER", _shaper_mock("sonnet")), \
         patch(_IMPORTLIB_PATH, return_value=_make_recorder_module(recorder_instance)):

        pm_core._close_slot_and_deposit(rec, "my-project")

    # Slot must be closed as 'landed'
    slot_store_instance.update_status.assert_called_once_with(
        "spec-xyz", "landed", by="task-abc"
    )

    # Deposit must have been recorded exactly once with correct provenance
    recorder_instance.record.assert_called_once()
    prov = recorder_instance.record.call_args[0][0]

    assert prov["model"] == "sonnet", "model must be non-null (sourced from shaper registry)"
    assert prov["agent_id"] == "task-abc"
    assert prov["tool"] == "lapis-pm:fixer"
    assert prov["slot"]["status"] == "landed"
    assert prov["slot"]["slot_id"] == "spec-xyz"
    assert prov["slot"]["project_id"] == "my-project"
    assert prov["manifest_hash"].startswith("sha256:")


# ---------------------------------------------------------------------------
# Failure path: failed → abandoned + deposit still correct
# ---------------------------------------------------------------------------

def test_failure_path_abandoned_with_correct_provenance():
    """Failed dispatch closes slot as 'abandoned' and still deposits provenance."""
    rec = _dispatch_record("failed", agent_type="fixer")
    recorder_instance = MagicMock()
    slot_store_instance = MagicMock()

    with patch(_SLOT_STORE_PATH, return_value=slot_store_instance), \
         patch.object(pm_core, "_SHAPER", _shaper_mock("sonnet")), \
         patch(_IMPORTLIB_PATH, return_value=_make_recorder_module(recorder_instance)):

        pm_core._close_slot_and_deposit(rec, "my-project")

    slot_store_instance.update_status.assert_called_once_with(
        "spec-xyz", "abandoned", by="task-abc"
    )

    prov = recorder_instance.record.call_args[0][0]
    assert prov["model"] == "sonnet"
    assert prov["slot"]["status"] == "abandoned"


# ---------------------------------------------------------------------------
# Idempotency: pending status → no-op (pending-guard)
# ---------------------------------------------------------------------------

def test_pending_status_is_noop():
    """A pending dispatch record is not in _SLOT_STATUS_MAP → no slot-close or deposit."""
    rec = _dispatch_record("pending", agent_type="fixer")
    recorder_instance = MagicMock()
    slot_store_instance = MagicMock()

    with patch(_SLOT_STORE_PATH, return_value=slot_store_instance), \
         patch(_IMPORTLIB_PATH, return_value=_make_recorder_module(recorder_instance)):

        pm_core._close_slot_and_deposit(rec, "my-project")

    # Nothing must be touched — pending is not a terminal status
    slot_store_instance.update_status.assert_not_called()
    recorder_instance.record.assert_not_called()


# ---------------------------------------------------------------------------
# Manifest-hash determinism (dedup guard for double-deposit)
# ---------------------------------------------------------------------------

def test_manifest_hash_is_deterministic():
    """Same slot_id + slot_status + completed_at → same manifest_hash.

    The deposit recorder uses INSERT OR IGNORE on manifest_hash to deduplicate.
    This test asserts the hash is deterministic so that guard works correctly
    when _close_slot_and_deposit is called twice on the same record.
    """
    rec = _dispatch_record("processed", agent_type="fixer")
    provs = []

    for _ in range(2):
        recorder_instance = MagicMock()
        slot_store_instance = MagicMock()

        with patch(_SLOT_STORE_PATH, return_value=slot_store_instance), \
             patch.object(pm_core, "_SHAPER", _shaper_mock("sonnet")), \
             patch(_IMPORTLIB_PATH, return_value=_make_recorder_module(recorder_instance)):

            pm_core._close_slot_and_deposit(rec, "my-project")
            provs.append(recorder_instance.record.call_args[0][0])

    assert provs[0]["manifest_hash"] == provs[1]["manifest_hash"], (
        "manifest_hash must be deterministic so INSERT OR IGNORE deduplicates second deposit"
    )


# ---------------------------------------------------------------------------
# Best-effort: SlotStore failure does not raise or block deposit
# ---------------------------------------------------------------------------

def test_slot_store_failure_does_not_block_deposit():
    """SlotStore.update_status raising must not prevent deposit emission."""
    rec = _dispatch_record("processed", agent_type="fixer")
    recorder_instance = MagicMock()
    slot_store_instance = MagicMock()
    slot_store_instance.update_status.side_effect = RuntimeError("DB locked")

    with patch(_SLOT_STORE_PATH, return_value=slot_store_instance), \
         patch.object(pm_core, "_SHAPER", _shaper_mock("sonnet")), \
         patch(_IMPORTLIB_PATH, return_value=_make_recorder_module(recorder_instance)):

        # Must not raise
        pm_core._close_slot_and_deposit(rec, "my-project")

    # Deposit still attempted despite slot-close failure
    recorder_instance.record.assert_called_once()


# ---------------------------------------------------------------------------
# Best-effort: recorder failure does not raise
# ---------------------------------------------------------------------------

def test_recorder_failure_does_not_raise():
    """recorder.record() raising must not propagate to caller."""
    rec = _dispatch_record("processed", agent_type="fixer")
    recorder_instance = MagicMock()
    recorder_instance.record.side_effect = RuntimeError("zephyr unavailable")
    slot_store_instance = MagicMock()

    with patch(_SLOT_STORE_PATH, return_value=slot_store_instance), \
         patch.object(pm_core, "_SHAPER", _shaper_mock("sonnet")), \
         patch(_IMPORTLIB_PATH, return_value=_make_recorder_module(recorder_instance)):

        # Must not raise
        pm_core._close_slot_and_deposit(rec, "my-project")


# ---------------------------------------------------------------------------
# Shaper model lookup failure falls back gracefully (model=None, not crash)
# ---------------------------------------------------------------------------

def test_shaper_model_lookup_failure_deposits_null_model():
    """If _SHAPER.get_agent() raises, deposit still proceeds with model=None."""
    rec = _dispatch_record("processed", agent_type="fixer")
    recorder_instance = MagicMock()
    slot_store_instance = MagicMock()

    bad_shaper = MagicMock()
    bad_shaper.get_agent.side_effect = KeyError("unknown agent")

    with patch(_SLOT_STORE_PATH, return_value=slot_store_instance), \
         patch.object(pm_core, "_SHAPER", bad_shaper), \
         patch(_IMPORTLIB_PATH, return_value=_make_recorder_module(recorder_instance)):

        pm_core._close_slot_and_deposit(rec, "my-project")

    prov = recorder_instance.record.call_args[0][0]
    assert prov["model"] is None  # graceful fallback, not a crash
