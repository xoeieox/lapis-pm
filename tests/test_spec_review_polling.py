"""Tests for _poll_reference_until_terminal (lapis_pm/spec_review.py:1127-1188).

Rewritten against the current reference-leg-only contract. The old
`_poll_until_terminal` this file tested (two-leg poll, `(sonnet_raw,
council_raw)` tuple return) was renamed to `_poll_sonnet_until_terminal` by
5bd2b7f (2026-06-19, shared-deliberation orchestrator migration) and then to
`_poll_reference_until_terminal` by 78bc873 (2026-08-01). This file was
carried through neither rename, so it imported a symbol that no longer
existed and broke collection for the whole repo. See spec
lapis-pm-orphan-poll-test-repair-v0.

Council polling now lives in the shared orchestrator's `run_deliberation`
and is no longer this function's concern, so the old file's council-side
assertions have no successor here and are not reconstructed.

Every test in this file gets its production queue/council paths redirected
to `tmp_path` by the `_isolated_queues` fixture below (function-scoped,
autouse) — closes review/debt/lapis-pm/bbc4b922d5. No test may write under
`/room`.

The isolation is behavioural, not nominal: `test_no_test_artifact_reaches_the_live_room_root`
asserts that a poll run driven by this file's fixture resolves the completed
queue under `tmp_path` and leaves the live production roots untouched, so a
regression that makes the patch silently ineffective (e.g. a future
import-time rebinding or a renamed constant) fails loudly instead of letting
test artifacts reappear in the live queue where a concurrent queue runner
could consume them.
"""
from __future__ import annotations

import json
import time

import pytest

import lapis_pm.spec_review as spec_review


@pytest.fixture(autouse=True)
def _isolated_queues(tmp_path, monkeypatch):
    """Redirect every production queue/council path to a tmp_path-relative dir.

    Patches the production module's namespace (the dotted-string form of
    monkeypatch.setattr targeting "lapis_pm.spec_review"), not a name
    imported into this test module — `_find_reviewer_output`
    (lapis_pm/spec_review.py:533-544) reads these constants from its own
    module globals at call time (:539), so rebinding a name this test module
    imported would silently leave production code resolving the real /room
    paths while the test appeared to pass.
    """
    claude_completed = tmp_path / "claude-queue" / "completed"
    gpu_completed = tmp_path / "gpu-queue" / "completed"
    claude_failed = tmp_path / "claude-queue" / "failed"
    gpu_failed = tmp_path / "gpu-queue" / "failed"
    council_dir = tmp_path / "council"
    monkeypatch.setattr("lapis_pm.spec_review._CLAUDE_QUEUE_COMPLETED", claude_completed)
    monkeypatch.setattr("lapis_pm.spec_review._GPU_QUEUE_COMPLETED", gpu_completed)
    monkeypatch.setattr("lapis_pm.spec_review._CLAUDE_QUEUE_FAILED", claude_failed)
    monkeypatch.setattr("lapis_pm.spec_review._GPU_QUEUE_FAILED", gpu_failed)
    monkeypatch.setattr("lapis_pm.spec_review._COUNCIL_DIR", council_dir)
    return {
        "claude_completed": claude_completed,
        "gpu_completed": gpu_completed,
        "claude_failed": claude_failed,
        "gpu_failed": gpu_failed,
        "council_dir": council_dir,
    }


def _write_output(directory, task_id, verdict="clean", **extra):
    """Write a fake reviewer output file under `directory` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{task_id}-output.md"
    payload = {"verdict": verdict, "issues": [], "confidence": 0.95}
    payload.update(extra)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. Null task id short-circuits (:1140-1141)
# ---------------------------------------------------------------------------

def test_null_task_id_returns_none_without_polling():
    result = spec_review._poll_reference_until_terminal(
        spec_reviewer_task_id=None,
        timeout_s=60,
        start_time=time.time(),
    )
    assert result is None


# ---------------------------------------------------------------------------
# 2. Output already present returns on the first check (:1150-1163)
# ---------------------------------------------------------------------------

def test_output_present_returns_processed_on_first_check(_isolated_queues):
    tid = "poll-test-processed"
    _write_output(
        _isolated_queues["claude_completed"], tid,
        verdict="clean", issues=["nit"], confidence=0.82,
    )

    result = spec_review._poll_reference_until_terminal(
        spec_reviewer_task_id=tid,
        timeout_s=60,
        start_time=time.time(),
    )

    assert result["status"] == "processed"
    assert result["verdict"] == "clean"
    assert result["issues"] == ["nit"]
    assert result["confidence"] == 0.82
    assert result["run_id"] == tid


# ---------------------------------------------------------------------------
# 3. Failed-queue detection (:1154-1156) — no coverage before this unit
# ---------------------------------------------------------------------------

def test_failed_queue_output_yields_failed_status(_isolated_queues, monkeypatch):
    """Output resolved from the failed root still returns its parsed verdict,
    but with status="failed" instead of "processed".

    _find_reviewer_output only ever searches the two COMPLETED roots (:539);
    the "failed" branch fires on a string-containment check against the
    _FAILED constants (:1154-1156), so the production code only takes this
    branch when a completed-root hit's path also matches a failed root.
    Reproduce that condition directly by pointing _CLAUDE_QUEUE_COMPLETED at
    the same directory already patched onto _CLAUDE_QUEUE_FAILED.
    """
    tid = "poll-test-failed"
    failed_dir = _isolated_queues["claude_failed"]
    monkeypatch.setattr("lapis_pm.spec_review._CLAUDE_QUEUE_COMPLETED", failed_dir)
    _write_output(failed_dir, tid, verdict="fixable")

    result = spec_review._poll_reference_until_terminal(
        spec_reviewer_task_id=tid,
        timeout_s=60,
        start_time=time.time(),
    )

    assert result["status"] == "failed"
    assert result["verdict"] == "fixable"


# ---------------------------------------------------------------------------
# 4. Timeout marker (:1172-1180) — back-dated start_time, no real sleep
# ---------------------------------------------------------------------------

def test_timeout_with_no_output_returns_timeout_marker():
    tid = "poll-test-timeout"
    start = time.time() - 10_000  # already past any reasonable timeout_s

    result = spec_review._poll_reference_until_terminal(
        spec_reviewer_task_id=tid,
        timeout_s=10,
        start_time=start,
    )

    assert result["status"] == "timeout"
    assert result["verdict"] == "timeout"
    assert result["confidence"] == 0.0
    assert result["issues"] == []
    assert result["run_id"] == tid


# ---------------------------------------------------------------------------
# 5. Passthrough of parse_error and claims_checked (:1161-1162)
# ---------------------------------------------------------------------------

def test_parse_error_and_claims_checked_pass_through(_isolated_queues):
    tid = "poll-test-passthrough"
    claims_checked = [{"claim": "x", "verified": True}]
    parse_error = {"file_size": 12, "head": "abc", "tail": "xyz"}
    _write_output(
        _isolated_queues["claude_completed"], tid,
        verdict="clean", claims_checked=claims_checked, parse_error=parse_error,
    )

    result = spec_review._poll_reference_until_terminal(
        spec_reviewer_task_id=tid,
        timeout_s=60,
        start_time=time.time(),
    )

    assert result["claims_checked"] == claims_checked
    assert result["parse_error"] == parse_error


# ---------------------------------------------------------------------------
# 6. bbc4b922d5 isolation is effective (meta-test)
# ---------------------------------------------------------------------------

def test_no_test_artifact_reaches_the_live_room_root(_isolated_queues):
    """The `_isolated_queues` patch must actually redirect `_find_reviewer_output`
    to tmp_path, and a poll run must not touch the live production roots.

    This is the missing coverage for review/debt/lapis-pm/bbc4b922d5: the
    fixture above patches the production module's constants, but nothing
    previously asserted the patch is effective — a regression (renamed
    constant, import-time rebinding, a new queue root added to the module)
    would let this file's `-output.md` artifacts land under the live
    /srv/lapis/claude-queue and /srv/lapis/council roots, where a concurrently running
    queue runner could consume them as real task output. The assertions here
    pin the behavioural contract:

    1. `_find_reviewer_output` resolves the task's output file under the
       tmp_path-completed dir (not the live root).
    2. A full `_poll_reference_until_terminal` run reads it from there and
       returns "processed".
    3. The live production roots (`_CLAUDE_QUEUE_COMPLETED`'s original
       /room value and `_COUNCIL_DIR`'s original /room value) contain no
       artifact for the task id.
    """
    # Capture the LIVE production roots before the fixture's patch took
    # effect — the fixture patched the module globals, so re-importing the
    # module would show the patched values; the original /room values are
    # what a regression would silently write to.
    import agents_core.room_paths as _rp
    live_completed = _rp.room_path("claude_queue.completed")
    live_council = _rp.room_path("council")

    tid = "poll-test-room-isolation"
    completed_dir = _isolated_queues["claude_completed"]

    # 1. Resolution goes to tmp_path, not the live root.
    resolved = spec_review._find_reviewer_output(tid)
    assert resolved is None  # no file yet
    p = _write_output(completed_dir, tid, verdict="clean")
    resolved = spec_review._find_reviewer_output(tid)
    assert resolved is not None
    assert resolved.parent == completed_dir
    assert str(resolved).startswith(str(_isolated_queues["claude_completed"].parent.parent))

    # 2. The poll reads the tmp_path artifact and reports processed.
    result = spec_review._poll_reference_until_terminal(
        spec_reviewer_task_id=tid,
        timeout_s=60,
        start_time=time.time(),
    )
    assert result is not None
    assert result["status"] == "processed"
    assert result["run_id"] == tid

    # 3. The live production roots hold no artifact for this task id.
    assert not (live_completed / f"{tid}-output.md").exists(), (
        "test artifact leaked into the live claude-queue completed root "
        f"({live_completed}) — bbc4b922d5 regression"
    )
    assert not (live_council / f"{tid}.yaml").exists(), (
        "test artifact leaked into the live council root "
        f"({live_council}) — bbc4b922d5 regression"
    )
