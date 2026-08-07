"""Tests for the stale-pending dispatch reaper
(lapis-pm-stale-pending-dispatch-reaper-v0), Legs 1 and 2.

Coverage per spec acceptance criteria:
  AC1 — pending record whose gpu_id is in no queue directory at all →
        flipped to failed, failure_reason="reaped:task_absent_from_queue",
        one de-duplicated episodic audit comment.
  AC2 — pending record whose gpu_id is in active/ past its own
        timeout_seconds + STARTUP_STALE_GRACE_S → flipped to failed,
        failure_reason="reaped:active_past_timeout".
  AC3 — pending record whose gpu_id is in active/ and WITHIN its own
        timeout → left pending (a live long-running fixer is never reaped).
  AC4 — carve-outs (fixer_retry+completed, reviewer+completed) still stay
        pending; already covered by tests/test_dispatch_reconciliation.py,
        re-asserted here for a case that also has no terminal match (the
        carve-out record is genuinely still queued, not reaped either).
  AC5 — a queue read that raises reaps nothing and returns 0.
  AC6 — force_dispatch("fixer", ...) against a target whose only pending
        record is provably dead succeeds (Leg 2): on current main this
        raises ValueError; post-fix the reconcile-before-guard call clears
        the dead record first.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pending_record(gpu_id: str, agent_type: str = "fixer", ts: str | None = None) -> dict:
    return {
        "gpu_id": gpu_id,
        "spec_id": "spec-abc",
        "agent_type": agent_type,
        "intent": "do the thing",
        "repo": "myrepo",
        "ts": ts or "2026-04-25T16:51:00-07:00",
        "status": "pending",
        "retry_count": 0,
        "pr_number": None,
    }


def _make_queue(failed=None, completed=None, active=None, pending=None):
    cq = MagicMock()
    cq.get_recent_failed.return_value = failed or []
    cq.get_recent_completed.return_value = completed or []
    cq.get_active.return_value = active or []
    cq.get_pending.return_value = pending or []
    return cq


def _run_reconcile(rec, queue):
    """Run _reconcile_dispatched_with_queue against a single record, fully
    mocked, and return the flipped count."""
    written = []

    def fake_write_observation(target_id, content, extra_tags=None):
        c = MagicMock()
        c.tags = extra_tags or []
        c.content = content
        written.append(c)
        return c

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save, \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation", side_effect=fake_write_observation):
        count = pm_core._reconcile_dispatched_with_queue("my-target")

    return count, mock_save, written


# ---------------------------------------------------------------------------
# AC1 — provably gone
# ---------------------------------------------------------------------------

def test_ac1_absent_from_all_queue_dirs_reaps_to_failed():
    gpu_id = "claude_20260805_100000_0001_fixer_myrepo"
    rec = _pending_record(gpu_id)
    queue = _make_queue()  # everything empty — gone from every directory

    count, mock_save, written = _run_reconcile(rec, queue)

    assert count == 1
    assert rec["status"] == "failed"
    assert rec["failure_reason"] == "reaped:task_absent_from_queue"
    assert rec.get("completed_at")
    mock_save.assert_called_once()
    assert len(written) == 1
    assert "pm:dispatch-reaped" in written[0].tags
    assert any(t.startswith("pm:dispatch-reconciled:gpu=") for t in written[0].tags)


def test_ac1_dedup_second_call_writes_nothing():
    """Second reconcile call against an already-reaped tag writes no new comment."""
    gpu_id = "claude_20260805_100000_0002_fixer_myrepo"
    rec = _pending_record(gpu_id)
    queue = _make_queue()
    dedup_tag = f"pm:dispatch-reconciled:gpu={gpu_id}"
    existing_comment = MagicMock()
    existing_comment.tags = [dedup_tag]

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched"), \
         patch("lapis_pm.episodic.all_comments", return_value=[existing_comment]), \
         patch("lapis_pm.episodic.write_observation") as mock_obs:
        count = pm_core._reconcile_dispatched_with_queue("my-target")

    # Still reaps (status flip is independent of the audit comment), but
    # writes no duplicate comment.
    assert count == 1
    assert rec["status"] == "failed"
    mock_obs.assert_not_called()


# ---------------------------------------------------------------------------
# AC2 — stuck active past its own timeout
# ---------------------------------------------------------------------------

def test_ac2_active_past_timeout_reaps_to_failed():
    gpu_id = "claude_20260805_100000_0003_fixer_myrepo"
    rec = _pending_record(gpu_id)
    # Started 2 hours ago with a 300s timeout — way past timeout + grace (300s).
    queue = _make_queue(active=[{
        "id": gpu_id,
        "started_at": "2026-08-05T08:00:00-07:00",
        "timeout_seconds": 300,
    }])

    with patch("lapis_pm.pm_core.datetime") as mock_dt:
        # Only used indirectly via pm_core.datetime.now(PACIFIC) inside
        # _reap_verdict; freeze "now" far enough past started_at.
        import datetime as _real_datetime
        mock_dt.now.return_value = _real_datetime.datetime(
            2026, 8, 5, 10, 30, 0, tzinfo=pm_core.PACIFIC)
        mock_dt.fromisoformat.side_effect = _real_datetime.datetime.fromisoformat

        count, mock_save, written = _run_reconcile(rec, queue)

    assert count == 1
    assert rec["status"] == "failed"
    assert rec["failure_reason"] == "reaped:active_past_timeout"
    mock_save.assert_called_once()
    assert len(written) == 1
    assert "pm:dispatch-reaped" in written[0].tags


# ---------------------------------------------------------------------------
# AC3 — active but within timeout: never reaped
# ---------------------------------------------------------------------------

def test_ac3_active_within_timeout_stays_pending():
    gpu_id = "claude_20260805_100000_0004_fixer_myrepo"
    rec = _pending_record(gpu_id)
    queue = _make_queue(active=[{
        "id": gpu_id,
        "started_at": "2026-08-05T10:00:00-07:00",
        "timeout_seconds": 3600,
    }])

    with patch("lapis_pm.pm_core.datetime") as mock_dt:
        import datetime as _real_datetime
        # Only 5 minutes in — well within a 1-hour timeout.
        mock_dt.now.return_value = _real_datetime.datetime(
            2026, 8, 5, 10, 5, 0, tzinfo=pm_core.PACIFIC)
        mock_dt.fromisoformat.side_effect = _real_datetime.datetime.fromisoformat

        count, mock_save, written = _run_reconcile(rec, queue)

    assert count == 0
    assert rec["status"] == "pending"
    mock_save.assert_not_called()
    assert written == []


# ---------------------------------------------------------------------------
# AC4 — carve-out record with no terminal match and still genuinely queued
# ---------------------------------------------------------------------------

def test_ac4_fixer_retry_still_pending_in_queue_stays_pending():
    gpu_id = "claude_20260805_100000_0005_fixer_retrymyrepo"
    rec = _pending_record(gpu_id, agent_type="fixer_retry")
    queue = _make_queue(pending=[{"id": gpu_id}])

    count, mock_save, written = _run_reconcile(rec, queue)

    assert count == 0
    assert rec["status"] == "pending"
    mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# AC5 — queue read raises: reaps nothing, returns 0
# ---------------------------------------------------------------------------

def test_ac5_queue_read_raises_reaps_nothing():
    gpu_id = "claude_20260805_100000_0006_fixer_myrepo"
    rec = _pending_record(gpu_id)
    queue = MagicMock()
    queue.get_recent_failed.side_effect = RuntimeError("queue unreadable")

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save, \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation") as mock_obs:
        count = pm_core._reconcile_dispatched_with_queue("my-target")

    assert count == 0
    assert rec["status"] == "pending"
    mock_save.assert_not_called()
    mock_obs.assert_not_called()

    # The _ex form additionally reports the failure for Leg 2 (force_dispatch)
    # to distinguish "nothing to reconcile" from "queue read failed".
    with patch.object(pm_core, "_ClaudeQueue", return_value=queue):
        flipped, ok = pm_core._reconcile_dispatched_with_queue_ex("my-target")
    assert flipped == 0
    assert ok is False


# ---------------------------------------------------------------------------
# AC6 — force_dispatch self-heals past a dead pending record (Leg 2)
# ---------------------------------------------------------------------------

def _make_target(target_id: str = "test-tid", pm_repo: str = "myrepo") -> MagicMock:
    t = MagicMock()
    t.pm_bound = True
    t.pm_repo = pm_repo
    t.data = {}
    return t


def _make_dispatch_result(task_id: str = "task-abc") -> MagicMock:
    res = MagicMock()
    res.task_id = task_id
    res.spec_id = "spec-abc"
    res.output_path = "/tmp/fake-output"
    return res


def _make_agent(model: str = "opus") -> MagicMock:
    a = MagicMock()
    a.model = model
    return a


def test_ac6_force_dispatch_self_heals_past_dead_pending_record():
    target_id = "test-reaper-tid"
    dead_gpu_id = "claude_20260805_090000_0007_fixer_myrepo"
    dead_rec = _pending_record(dead_gpu_id, agent_type="fixer")
    dispatched_list = [dead_rec]

    queue = _make_queue()  # gone from every directory — Case A

    with (
        patch("lapis_pm.pm_core.TargetStore") as mock_store_cls,
        patch.object(pm_core, "_ClaudeQueue", return_value=queue),
        patch.object(pm_core._SHAPER, "dispatch", return_value=_make_dispatch_result()),
        patch.object(pm_core._SHAPER, "get_agent", return_value=_make_agent()),
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec summary"),
        patch("lapis_pm.pm_core.episodic.write_dispatch"),
        patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]),
        patch("lapis_pm.pm_core.episodic.write_observation"),
        patch("lapis_pm.pm_core.append_dispatched"),
        patch("lapis_pm.pm_core.load_dispatched", return_value=dispatched_list),
        patch("lapis_pm.pm_core.save_dispatched"),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        patch("lapis_pm.router_portfolio.emit_decision_dispatch"),
    ):
        mock_store_cls.return_value.get.return_value = _make_target(target_id)
        # On unfixed main this raises ValueError("... has a pending fixer
        # dispatch ... not firing a concurrent initial fixer"). Post-fix it
        # must succeed because force_dispatch reconciles the dead record
        # away before the L1.D3 guard evaluates.
        task_id = pm_core.force_dispatch(target_id, "fixer", "do the thing again")

    assert task_id == "task-abc"
    # The dead record was reaped in place as part of the self-heal.
    assert dead_rec["status"] == "failed"
    assert dead_rec["failure_reason"] == "reaped:task_absent_from_queue"


def test_ac6_guard_still_fires_when_reap_cannot_prove_death():
    """Sanity check for the negative case: a genuinely live pending record
    (still in active/, within its own timeout) still blocks a concurrent
    initial fixer — Leg 2 does not weaken the guard, only unwedge provably
    dead records."""
    target_id = "test-reaper-tid-live"
    live_gpu_id = "claude_20260805_100000_0008_fixer_myrepo"
    live_rec = _pending_record(live_gpu_id, agent_type="fixer")
    dispatched_list = [live_rec]

    queue = _make_queue(active=[{
        "id": live_gpu_id,
        "started_at": "2026-08-05T10:00:00-07:00",
        "timeout_seconds": 3600,
    }])

    with (
        patch("lapis_pm.pm_core.TargetStore") as mock_store_cls,
        patch.object(pm_core, "_ClaudeQueue", return_value=queue),
        patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]),
        patch("lapis_pm.pm_core.episodic.write_observation"),
        patch("lapis_pm.pm_core.load_dispatched", return_value=dispatched_list),
        patch("lapis_pm.pm_core.save_dispatched"),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        patch("lapis_pm.pm_core.datetime") as mock_dt,
    ):
        import datetime as _real_datetime
        # Freeze "now" 5 minutes past started_at — well within the 1-hour timeout.
        mock_dt.now.return_value = _real_datetime.datetime(
            2026, 8, 5, 10, 5, 0, tzinfo=pm_core.PACIFIC)
        mock_dt.fromisoformat.side_effect = _real_datetime.datetime.fromisoformat

        mock_store_cls.return_value.get.return_value = _make_target(target_id)
        try:
            pm_core.force_dispatch(target_id, "fixer", "do the thing again")
            raised = False
        except ValueError as e:
            raised = True
            msg = str(e)

    assert raised
    assert "clear-dispatch" in msg
    assert live_rec["status"] == "pending"
