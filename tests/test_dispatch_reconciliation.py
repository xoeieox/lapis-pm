"""Tests for _reconcile_dispatched_with_queue (dispatch–queue reconciliation).

Coverage per spec:
  - pending dispatch + queue failed entry matching gpu_id → flipped to failed
  - pending dispatch + queue completed entry matching gpu_id → flipped to processed
    (non-fixer_retry only — fixer_retry completed is left pending per carve-out)
  - pending dispatch + no matching queue entry → left as pending (no-op)
  - already-failed dispatch + matching queue failed entry → left untouched
  - audit comment written exactly once per flip (de-dup: second call writes nothing)
  - fixer_retry carve-out: queue completed → stays pending (SHA-advance owns processed)
  - fixer_retry carve-out: queue failed → still flipped to failed
  - non-fixer_retry (fixer): queue completed → flipped to processed (regression)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mem(dispatched: list[dict] | None = None, comments: list | None = None):
    """Return a mock MemoryStore seeded with dispatched records + comments."""
    mem = MagicMock()
    dispatched_json = json.dumps(dispatched or [])
    mem.get.return_value = dispatched_json
    return mem


def _make_comment_store(comments: list | None = None):
    """Return a mock CommentStore that lists the given Comment-like objects."""
    cs = MagicMock()
    cs.list.return_value = comments or []
    return cs


def _make_comment(tags: list[str]) -> MagicMock:
    c = MagicMock()
    c.tags = tags
    return c


def _pending_record(gpu_id: str, agent_type: str = "fixer_retry",
                    pr_number: int = 1) -> dict:
    return {
        "gpu_id": gpu_id,
        "spec_id": "spec-abc",
        "agent_type": agent_type,
        "intent": f"fix PR #{pr_number}",
        "repo": "myrepo",
        "ts": "2026-04-25T16:51:00-07:00",
        "status": "pending",
        "retry_count": 0,
        "pr_number": pr_number,
    }


# ---------------------------------------------------------------------------
# Fixtures: fake ClaudeQueue
# ---------------------------------------------------------------------------

def _make_queue(failed: list[dict] | None = None,
                completed: list[dict] | None = None):
    cq = MagicMock()
    cq.get_recent_failed.return_value = failed or []
    cq.get_recent_completed.return_value = completed or []
    return cq


# ---------------------------------------------------------------------------
# Test: failed flip
# ---------------------------------------------------------------------------

def test_pending_flips_to_failed_on_queue_failure():
    """Pending record with gpu_id=X; queue reports X failed → status=failed, error carried."""
    gpu_id = "claude_20260425_164727_3432_fixer_retryclaudeviewtar"
    rec = _pending_record(gpu_id)
    error_msg = "ERROR: call_claude_cli returned None"

    queue = _make_queue(
        failed=[{"id": gpu_id, "error": error_msg, "completed_at": "2026-04-25T16:51:00-07:00"}],
    )

    written_comments = []

    def fake_write_observation(target_id, content, extra_tags=None):
        c = MagicMock()
        c.tags = extra_tags or []
        c.content = content
        written_comments.append(c)
        return c

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save, \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation", side_effect=fake_write_observation):

        count = pm_core._reconcile_dispatched_with_queue("my-target")

    assert count == 1
    assert rec["status"] == "failed"
    assert rec["error"] == error_msg
    mock_save.assert_called_once()
    # Audit comment written
    assert len(written_comments) == 1
    assert "pm:dispatch-reconciled" in written_comments[0].tags
    assert f"pm:dispatch-reconciled:gpu={gpu_id}" in written_comments[0].tags


# ---------------------------------------------------------------------------
# Test: processed flip
# ---------------------------------------------------------------------------

def test_pending_flips_to_processed_on_queue_completion():
    """Pending non-fixer_retry record + queue completed → status=processed, completed_at set.

    Uses agent_type="fixer" — non-fixer_retry agents are flipped to processed by the
    reconciler. fixer_retry is handled separately (carve-out tests below).
    """
    gpu_id = "claude_20260426_120000_0001_fixer_myrepo"
    rec = _pending_record(gpu_id, agent_type="fixer")  # NOT fixer_retry
    completed_at = "2026-04-26T12:05:00-07:00"

    queue = _make_queue(
        completed=[{"id": gpu_id, "completed_at": completed_at}],
    )

    written_comments = []

    def fake_write_observation(target_id, content, extra_tags=None):
        c = MagicMock()
        c.tags = extra_tags or []
        written_comments.append(c)
        return c

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save, \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation", side_effect=fake_write_observation):

        count = pm_core._reconcile_dispatched_with_queue("my-target")

    assert count == 1
    assert rec["status"] == "processed"
    assert rec["completed_at"] == completed_at
    mock_save.assert_called_once()
    assert len(written_comments) == 1
    assert f"pm:dispatch-reconciled:gpu={gpu_id}" in written_comments[0].tags


# ---------------------------------------------------------------------------
# Test: no-match → no-op
# ---------------------------------------------------------------------------

def test_pending_no_match_left_as_pending():
    """Pending record with gpu_id=Z; queue returns nothing matching → left as pending."""
    gpu_id = "claude_20260426_130000_0002_fixer_myrepo"
    rec = _pending_record(gpu_id)

    queue = _make_queue(
        failed=[{"id": "some-other-id", "error": "ERROR: unrelated"}],
    )

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


# ---------------------------------------------------------------------------
# Test: already-terminal → untouched
# ---------------------------------------------------------------------------

def test_already_failed_record_not_re_flipped():
    """Already-failed record with gpu_id=W; queue's failed list includes W → left alone."""
    gpu_id = "claude_20260424_094637_3406_scoutlapisengineclaudequ"
    rec = _pending_record(gpu_id)
    rec["status"] = "failed"
    rec["error"] = "ERROR: previous failure"

    queue = _make_queue(
        failed=[{"id": gpu_id, "error": "ERROR: new failure message"}],
    )

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save, \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation") as mock_obs:

        count = pm_core._reconcile_dispatched_with_queue("my-target")

    assert count == 0
    # Error field unchanged — the original failure message is kept.
    assert rec["error"] == "ERROR: previous failure"
    mock_save.assert_not_called()
    mock_obs.assert_not_called()


# ---------------------------------------------------------------------------
# Test: audit comment de-dup (second call writes nothing new)
# ---------------------------------------------------------------------------

def test_audit_comment_dedup_across_calls():
    """Calling reconcile twice: first call writes audit comment; second call does not."""
    gpu_id = "claude_20260425_164727_3432_fixer_retryclaudeviewtar"
    dedup_tag = f"pm:dispatch-reconciled:gpu={gpu_id}"

    # First call: no prior dedup tag, record is pending
    rec_first = _pending_record(gpu_id)
    queue = _make_queue(
        failed=[{"id": gpu_id, "error": "ERROR: foo", "completed_at": "2026-04-25T17:00:00-07:00"}],
    )

    written_first = []

    def fake_write_obs_first(target_id, content, extra_tags=None):
        c = MagicMock()
        c.tags = extra_tags or []
        written_first.append(c)
        return c

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec_first]), \
         patch("lapis_pm.pm_core.save_dispatched"), \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation", side_effect=fake_write_obs_first):

        count1 = pm_core._reconcile_dispatched_with_queue("my-target")

    assert count1 == 1
    assert len(written_first) == 1

    # Second call: record is now "failed" (terminal), AND dedup tag exists in comments.
    rec_second = dict(rec_first)  # copy, already flipped to failed
    existing_comment = _make_comment([dedup_tag, "pm:dispatch-reconciled", "pm:observation"])

    written_second = []

    def fake_write_obs_second(target_id, content, extra_tags=None):
        c = MagicMock()
        c.tags = extra_tags or []
        written_second.append(c)
        return c

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec_second]), \
         patch("lapis_pm.pm_core.save_dispatched"), \
         patch("lapis_pm.episodic.all_comments", return_value=[existing_comment]), \
         patch("lapis_pm.episodic.write_observation", side_effect=fake_write_obs_second):

        count2 = pm_core._reconcile_dispatched_with_queue("my-target")

    # record is already terminal → 0 flips, 0 new comments
    assert count2 == 0
    assert len(written_second) == 0


# ---------------------------------------------------------------------------
# Test: _ClaudeQueue unavailable → no-op (defensive import)
# ---------------------------------------------------------------------------

def test_no_op_when_claude_queue_unavailable():
    """When _ClaudeQueue is None (agents_core not on path), reconcile returns 0 gracefully."""
    rec = _pending_record("any-gpu-id")

    with patch.object(pm_core, "_ClaudeQueue", None), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]) as mock_load, \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save:

        count = pm_core._reconcile_dispatched_with_queue("my-target")

    assert count == 0
    mock_load.assert_not_called()
    mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Test: reconciled count surfaces in TickResult
# ---------------------------------------------------------------------------

def test_tick_result_includes_reconciled_count():
    """When reconcile flips a record, TickResult.reconciled is non-zero."""
    gpu_id = "claude_20260425_164727_3432_fixer_retryclaudeviewtar"
    rec = _pending_record(gpu_id)

    queue = _make_queue(
        failed=[{"id": gpu_id, "error": "ERROR: transient", "completed_at": "2026-04-25T17:00:00"}],
    )

    from agents_core.targets import Target

    mock_target = MagicMock(spec=Target)
    mock_target.pm_bound = True
    mock_target.paused = False
    mock_target.pm_repo = ""
    mock_target.pm_authority = "advisory"

    with patch("lapis_pm.pm_core._ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched"), \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation"), \
         patch("lapis_pm.pm_core.TargetStore") as MockStore, \
         patch("lapis_pm.pm_core.get_cursor", return_value=None), \
         patch("lapis_pm.pm_core.set_cursor"), \
         patch("lapis_pm.pm_core.get_pause_state", return_value=None), \
         patch("lapis_pm.pm_core.set_pause_state"), \
         patch("lapis_pm.episodic.since", return_value=[]):

        MockStore.return_value.get.return_value = mock_target
        result = pm_core.tick("my-target")

    assert result.reconciled == 1


# ---------------------------------------------------------------------------
# fixer_retry carve-out: SHA-advance perceiver owns fixer_retry → processed
# ---------------------------------------------------------------------------

def test_fixer_retry_queue_completed_stays_pending():
    """fixer_retry + queue completed → stays pending (SHA-advance perceiver owns processed).

    The reconciler must NOT flip a fixer_retry to processed based on queue
    completion alone. The SHA-advance perceiver (~pm_core _perceive_pr_sha_advance)
    is the sole authority for that transition.
    """
    gpu_id = "claude_20260427_210000_0001_fixer_retry_myrepo"
    rec = _pending_record(gpu_id, agent_type="fixer_retry")

    queue = _make_queue(
        completed=[{"id": gpu_id, "completed_at": "2026-04-27T21:05:00-07:00"}],
    )

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save, \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation") as mock_obs:

        count = pm_core._reconcile_dispatched_with_queue("my-target")

    assert count == 0, "fixer_retry queue-completed must not count as a reconciled flip"
    assert rec["status"] == "pending", "fixer_retry must remain pending after queue completion"
    mock_save.assert_not_called()
    mock_obs.assert_not_called()


def test_fixer_retry_queue_failed_flips_to_failed():
    """fixer_retry + queue failed → flips to failed (failure carve-out still permitted).

    Failed retries do not advance cycle accounting, so the reconciler is allowed
    to flip them to failed — this is safe and necessary for operational hygiene.
    """
    gpu_id = "claude_20260427_070000_0001_fixer_retry_myrepo"
    error_msg = "ERROR: runner exited non-zero"
    rec = _pending_record(gpu_id, agent_type="fixer_retry")

    queue = _make_queue(
        failed=[{"id": gpu_id, "error": error_msg, "completed_at": "2026-04-27T07:05:00-07:00"}],
    )

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save, \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation"):

        count = pm_core._reconcile_dispatched_with_queue("my-target")

    assert count == 1
    assert rec["status"] == "failed"
    assert rec["error"] == error_msg
    mock_save.assert_called_once()


def test_non_fixer_retry_queue_completed_flips_to_processed():
    """Non-fixer_retry (e.g. fixer) + queue completed → flipped to processed.

    Regression: the carve-out must not affect other agent types. fixer, reviewer,
    and scout records are reconciled normally in both directions.
    """
    gpu_id = "claude_20260427_080000_0001_fixer_myrepo"
    completed_at = "2026-04-27T08:05:00-07:00"
    rec = _pending_record(gpu_id, agent_type="fixer")

    queue = _make_queue(
        completed=[{"id": gpu_id, "completed_at": completed_at}],
    )

    with patch.object(pm_core, "_ClaudeQueue", return_value=queue), \
         patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]), \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save, \
         patch("lapis_pm.episodic.all_comments", return_value=[]), \
         patch("lapis_pm.episodic.write_observation"):

        count = pm_core._reconcile_dispatched_with_queue("my-target")

    assert count == 1
    assert rec["status"] == "processed"
    assert rec["completed_at"] == completed_at
    mock_save.assert_called_once()
