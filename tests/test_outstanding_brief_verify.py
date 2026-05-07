"""Tests for set_outstanding_brief_verified and _post_write_sweep_brief.

Mocking strategy: _mem() returns a fresh MemoryStore() per call, so
patching a single instance's .get does not affect subsequent calls.
Tests that need controlled read behaviour patch MemoryStore.get at the
class level via patch.object(MemoryStore, "get", ...) as required by
the spec.  Test isolation (not touching the prod mem.db) is achieved by
also patching _mem to return a MemoryStore backed by a fresh temp file.
"""

from __future__ import annotations

import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_core.mem import MemoryStore
from lapis_pm.pm_core import (
    OutstandingBriefWriteError,
    _brief_key,
    _post_write_sweep_brief,
    clear_outstanding_brief,
    get_outstanding_brief,
    set_outstanding_brief_verified,
)

TID = "test-outstanding-brief-verify-fixture"
CID = "cid-fixture-abc"


def _tmp_store() -> MemoryStore:
    """Return a MemoryStore backed by a fresh temp SQLite file."""
    return MemoryStore(db_path=Path(tempfile.mktemp(suffix=".db")))


# ---------------------------------------------------------------------------
# Case 1: happy path — write succeeds, key present after return
# ---------------------------------------------------------------------------


def test_set_outstanding_brief_verified_writes_and_verifies():
    """Happy path: key is present immediately after set_outstanding_brief_verified."""
    store = _tmp_store()

    with patch("lapis_pm.pm_core._mem", return_value=store):
        set_outstanding_brief_verified(TID, CID)
        assert get_outstanding_brief(TID) == CID
        # Teardown
        clear_outstanding_brief(TID)


# ---------------------------------------------------------------------------
# Case 2: first verify read misses; second succeeds — retry + WARN logged
# ---------------------------------------------------------------------------


def test_set_outstanding_brief_verified_retries_on_first_miss(capsys):
    """First read misses (returns None); second read succeeds.

    Expects:
    - [outstanding-brief:write-mismatch] line on stderr with attempt=1
    - Function returns normally (no exception)
    """
    store = _tmp_store()

    # First .get() returns None (simulate miss); second returns the real record.
    real_rec = {"content": CID, "key": _brief_key(TID), "tags": ["lapis-pm"]}
    side_effects = [None, real_rec]

    with patch("lapis_pm.pm_core._mem", return_value=store):
        with patch.object(MemoryStore, "get", side_effect=side_effects):
            set_outstanding_brief_verified(TID, CID)  # must not raise

    captured = capsys.readouterr()
    assert "[outstanding-brief:write-mismatch]" in captured.err
    assert f"tid={TID}" in captured.err
    assert f"cid={CID}" in captured.err
    assert "attempt=1" in captured.err
    assert "attempt=2" not in captured.err


# ---------------------------------------------------------------------------
# Case 3: all reads miss — raises OutstandingBriefWriteError, two WARN lines
# ---------------------------------------------------------------------------


def test_set_outstanding_brief_verified_raises_after_retry(capsys):
    """All reads return None; function must raise OutstandingBriefWriteError
    and emit two [outstanding-brief:write-mismatch] lines (attempt=1 and attempt=2).
    """
    store = _tmp_store()

    with patch("lapis_pm.pm_core._mem", return_value=store):
        with patch.object(MemoryStore, "get", return_value=None):
            with pytest.raises(OutstandingBriefWriteError):
                set_outstanding_brief_verified(TID, CID)

    captured = capsys.readouterr()
    assert captured.err.count("[outstanding-brief:write-mismatch]") == 2
    assert "attempt=1" in captured.err
    assert "attempt=2" in captured.err


# ---------------------------------------------------------------------------
# Case 4: key disappears between verify and sweep — sweep logs, never raises
# ---------------------------------------------------------------------------


def test_post_write_sweep_logs_on_disappearance(capsys):
    """Key is written and verified, then deleted externally.

    _post_write_sweep_brief must:
    - emit exactly one [outstanding-brief:disappeared-post-write] line to stderr
    - NOT raise any exception
    """
    store = _tmp_store()

    with patch("lapis_pm.pm_core._mem", return_value=store):
        # Write and verify succeeds
        set_outstanding_brief_verified(TID, CID)
        assert get_outstanding_brief(TID) == CID

        # Simulate external deletion
        store.delete(_brief_key(TID))
        assert get_outstanding_brief(TID) is None

        # Sweep must log, not raise
        _post_write_sweep_brief(TID, CID)

    captured = capsys.readouterr()
    assert captured.err.count("[outstanding-brief:disappeared-post-write]") == 1
    assert f"tid={TID}" in captured.err
    assert f"cid={CID}" in captured.err
