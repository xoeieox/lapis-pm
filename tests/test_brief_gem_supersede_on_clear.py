"""Tests for lapis-pm-hold-brief-optionless-gem-fix-v0 Fix 2: gem-supersede-on-clear.

clear_outstanding_brief (pm_core.py) is the single choke-point all four
clear-brief call paths funnel through: cli.py unbind, brief.py
principal_decision, pm_core.py auto_land, pm_core.py user_ack. Before this
fix it deleted the outstanding-brief mem key without ever telling the Desk
gem the brief was resolved, so a gem could outlive its brief.

These tests verify: each of the four reasons supersedes a live gem tied to
the cleared brief; a clear with no live gem is a silent no-op; a
supersede-lookup failure never blocks the underlying delete (fail-soft,
matching the function's existing style).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_core.mem import MemoryStore
from lapis_pm import brief_gem
from lapis_pm.pm_core import clear_outstanding_brief, get_outstanding_brief, set_outstanding_brief


def _tmp_store() -> MemoryStore:
    return MemoryStore(db_path=Path(tempfile.mktemp(suffix=".db")))


@pytest.mark.parametrize("reason", ["unbind", "principal_decision", "auto_land", "user_ack"])
def test_clear_outstanding_brief_supersedes_live_gem(reason):
    """Each of the four clear-brief reasons supersedes a live gem tied to the brief."""
    store = _tmp_store()
    tid = "test-supersede-target"
    cid = "cid-supersede-abc"
    gem_id = "gem-supersede-001"

    rkey = brief_gem._reverse_key(tid, cid)
    store.set(rkey, gem_id, tags=["lapis-pm", "brief-gem"])

    supersede_calls: list[tuple] = []
    update_calls: list[tuple] = []

    def _fake_supersede(gid, reason, by):
        supersede_calls.append((gid, reason, by))
        return True

    def _fake_update(gid, status, annotation=None, actioned_ts=None):
        update_calls.append((gid, status, annotation))

    with (
        patch("lapis_pm.pm_core._mem", return_value=store),
        patch("lapis_pm.brief_gem._call_supersede_endpoint", side_effect=_fake_supersede),
        patch("lapis_pm.brief_gem._update_map_status", side_effect=_fake_update),
    ):
        set_outstanding_brief(tid, cid)
        clear_outstanding_brief(tid, reason=reason)

    assert supersede_calls == [(gem_id, f"brief cleared: {reason}", "lapis-pm:brief-cleared")]
    assert update_calls == [(gem_id, "superseded", f"brief cleared: {reason}")]

    with patch("lapis_pm.pm_core._mem", return_value=store):
        assert get_outstanding_brief(tid) is None


def test_clear_outstanding_brief_no_live_gem_is_silent_noop():
    """No gem was ever deposited for this brief -> no supersede call, no exception."""
    store = _tmp_store()
    tid = "test-no-gem-target"
    cid = "cid-no-gem-abc"

    with (
        patch("lapis_pm.pm_core._mem", return_value=store),
        patch("lapis_pm.brief_gem._call_supersede_endpoint") as mock_sup,
    ):
        set_outstanding_brief(tid, cid)
        clear_outstanding_brief(tid, reason="unbind")  # must not raise

    mock_sup.assert_not_called()
    with patch("lapis_pm.pm_core._mem", return_value=store):
        assert get_outstanding_brief(tid) is None


def test_clear_outstanding_brief_no_prior_brief_is_silent_noop():
    """No outstanding brief at all (observed is None) -> no supersede lookup, no exception."""
    store = _tmp_store()
    tid = "test-absent-brief-target"

    with (
        patch("lapis_pm.pm_core._mem", return_value=store),
        patch("lapis_pm.brief_gem._reverse_key") as mock_rkey,
    ):
        clear_outstanding_brief(tid, reason="unbind")  # must not raise

    mock_rkey.assert_not_called()


def test_clear_outstanding_brief_409_leaves_forward_map_untouched():
    """Supersede endpoint returns False (409, already terminal) -> no forward-map update."""
    store = _tmp_store()
    tid = "test-409-target"
    cid = "cid-409-abc"
    gem_id = "gem-409-001"

    rkey = brief_gem._reverse_key(tid, cid)
    store.set(rkey, gem_id, tags=["lapis-pm", "brief-gem"])

    with (
        patch("lapis_pm.pm_core._mem", return_value=store),
        patch("lapis_pm.brief_gem._call_supersede_endpoint", return_value=False),
        patch("lapis_pm.brief_gem._update_map_status") as mock_update,
    ):
        set_outstanding_brief(tid, cid)
        clear_outstanding_brief(tid, reason="auto_land")

    mock_update.assert_not_called()


def test_clear_outstanding_brief_supersede_failure_does_not_block_clear(capsys):
    """A supersede-lookup error must not block the outstanding-brief delete (fail-soft)."""
    store = _tmp_store()
    tid = "test-supersede-error-target"
    cid = "cid-supersede-error"
    gem_id = "gem-error-001"

    rkey = brief_gem._reverse_key(tid, cid)
    store.set(rkey, gem_id, tags=["lapis-pm", "brief-gem"])

    with (
        patch("lapis_pm.pm_core._mem", return_value=store),
        patch("lapis_pm.brief_gem._call_supersede_endpoint",
              side_effect=RuntimeError("weaver down")),
    ):
        set_outstanding_brief(tid, cid)
        clear_outstanding_brief(tid, reason="auto_land")  # must not raise

    with patch("lapis_pm.pm_core._mem", return_value=store):
        assert get_outstanding_brief(tid) is None

    captured = capsys.readouterr()
    assert "[outstanding-brief:supersede-gem-error]" in captured.err
