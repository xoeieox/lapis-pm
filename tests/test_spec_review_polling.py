"""Tests for _poll_until_terminal."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lapis_pm.spec_review import (
    _CLAUDE_QUEUE_COMPLETED,
    _COUNCIL_DIR,
    _poll_until_terminal,
)


def _write_reviewer_output(task_id: str, verdict: str = "clean") -> Path:
    """Write a fake reviewer output file and return its path."""
    _CLAUDE_QUEUE_COMPLETED.mkdir(parents=True, exist_ok=True)
    p = _CLAUDE_QUEUE_COMPLETED / f"{task_id}-output.md"
    p.write_text(json.dumps({"verdict": verdict, "issues": [], "confidence": 0.95}))
    return p


def _write_council_yaml(run_id: str, status: str = "resolved") -> Path:
    """Write a fake council run YAML and return its path."""
    import yaml
    _COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    p = _COUNCIL_DIR / f"{run_id}.yaml"
    run = {
        "run_id": run_id,
        "status": status,
        "mode": "deliberation",
        "synthesis": {
            "confidence": "converged",
            "landing": "test landing",
            "open_questions": [],
            "positions": [{"entity": "ent-a", "position": "agree"}],
        },
    }
    p.write_text(yaml.safe_dump(run))
    return p


# ---------------------------------------------------------------------------
# Happy path: both terminal simultaneously
# ---------------------------------------------------------------------------

def test_both_terminal_immediately():
    """Both files present before poll starts — loop exits on first check."""
    tid = f"poll-test-{int(time.time())}"
    run_id = f"council-poll-{int(time.time())}"
    out_path = None
    council_path = None
    try:
        out_path = _write_reviewer_output(tid)
        council_path = _write_council_yaml(run_id)

        sonnet_raw, council_raw = _poll_until_terminal(
            spec_reviewer_task_id=tid,
            council_run_id=run_id,
            timeout_s=60,
            start_time=time.time(),
        )
        assert sonnet_raw["verdict"] == "clean"
        assert council_raw["status"] == "resolved"
    finally:
        if out_path and out_path.exists():
            out_path.unlink()
        if council_path and council_path.exists():
            council_path.unlink()


# ---------------------------------------------------------------------------
# Timeout: pending side gets "timeout" marker
# ---------------------------------------------------------------------------

def test_timeout_marks_pending_council():
    """spec_reviewer output present, council never writes → council gets timeout marker."""
    tid = f"poll-timeout-{int(time.time())}"
    run_id = f"council-timeout-{int(time.time())}"
    out_path = None
    try:
        out_path = _write_reviewer_output(tid, verdict="fixable")

        # Start with elapsed time already past timeout
        start = time.time() - 100  # 100s elapsed → timeout_s=10 already exceeded

        sonnet_raw, council_raw = _poll_until_terminal(
            spec_reviewer_task_id=tid,
            council_run_id=run_id,
            timeout_s=10,
            start_time=start,
        )
        assert sonnet_raw["verdict"] == "fixable"
        assert council_raw["status"] == "timeout"
    finally:
        if out_path and out_path.exists():
            out_path.unlink()
        # council YAML should NOT exist (we never wrote it)
        council_yaml = _COUNCIL_DIR / f"{run_id}.yaml"
        if council_yaml.exists():
            council_yaml.unlink()


def test_timeout_marks_pending_reviewer():
    """Council YAML written + terminal, reviewer never writes → reviewer gets timeout."""
    tid = f"poll-timeout-rev-{int(time.time())}"
    run_id = f"council-present-{int(time.time())}"
    council_path = None
    try:
        council_path = _write_council_yaml(run_id, status="resolved")

        start = time.time() - 100  # already timed out

        sonnet_raw, council_raw = _poll_until_terminal(
            spec_reviewer_task_id=tid,
            council_run_id=run_id,
            timeout_s=10,
            start_time=start,
        )
        assert sonnet_raw["status"] == "timeout"
        assert sonnet_raw["verdict"] == "timeout"
        assert council_raw["status"] == "resolved"
    finally:
        if council_path and council_path.exists():
            council_path.unlink()
        # Ensure no stray reviewer output was written
        out = _CLAUDE_QUEUE_COMPLETED / f"{tid}-output.md"
        if out.exists():
            out.unlink()


# ---------------------------------------------------------------------------
# Progress log lines — §4 and §5
# ---------------------------------------------------------------------------

def test_poll_timeout_log_emitted(capsys):
    """Both sides timeout → [spec-review:timeout] sonnet_done=False council_done=False."""
    tid = f"poll-timeout-log-{int(time.time())}"
    run_id = f"council-timeout-log-{int(time.time())}"
    try:
        # No output files written — both sides hit timeout
        sonnet_raw, council_raw = _poll_until_terminal(
            spec_reviewer_task_id=tid,
            council_run_id=run_id,
            timeout_s=60,
            start_time=time.time() - 10000,
        )
        assert sonnet_raw["status"] == "timeout"
        assert council_raw["status"] == "timeout"
        captured = capsys.readouterr()
        assert "[spec-review:timeout] elapsed=" in captured.err
        assert "sonnet_done=False" in captured.err
        assert "council_done=False" in captured.err
        assert "sonnet-complete" not in captured.err
        assert "council-complete" not in captured.err
    finally:
        out = _CLAUDE_QUEUE_COMPLETED / f"{tid}-output.md"
        if out.exists():
            out.unlink()
        council_yaml = _COUNCIL_DIR / f"{run_id}.yaml"
        if council_yaml.exists():
            council_yaml.unlink()


def test_sonnet_complete_log_emitted(capsys):
    """Reviewer output found, council times out → sonnet-complete + timeout logs."""
    tid = f"sonnet-complete-log-{int(time.time())}"
    run_id = f"council-sonnet-timeout-{int(time.time())}"
    out_path = None
    try:
        out_path = _write_reviewer_output(tid, "clean")
        sonnet_raw, council_raw = _poll_until_terminal(
            spec_reviewer_task_id=tid,
            council_run_id=run_id,
            timeout_s=60,
            start_time=time.time() - 10000,
        )
        assert sonnet_raw["verdict"] == "clean"
        assert council_raw["status"] == "timeout"
        captured = capsys.readouterr()
        assert f"[spec-review:sonnet-complete] task_id={tid}" in captured.err
        assert "elapsed=" in captured.err
        assert "verdict=clean" in captured.err
        assert "[spec-review:timeout]" in captured.err
        assert "sonnet_done=True" in captured.err
        assert "council_done=False" in captured.err
    finally:
        if out_path and out_path.exists():
            out_path.unlink()
        council_yaml = _COUNCIL_DIR / f"{run_id}.yaml"
        if council_yaml.exists():
            council_yaml.unlink()


def test_council_complete_log_emitted(capsys):
    """Council yaml found (status=failed), reviewer times out → council-complete + timeout."""
    tid = f"council-complete-log-{int(time.time())}"
    run_id = f"council-complete-yaml-{int(time.time())}"
    council_path = None
    try:
        council_path = _write_council_yaml(run_id, "failed")
        sonnet_raw, council_raw = _poll_until_terminal(
            spec_reviewer_task_id=tid,
            council_run_id=run_id,
            timeout_s=60,
            start_time=time.time() - 10000,
        )
        assert sonnet_raw["status"] == "timeout"
        assert council_raw["status"] == "failed"
        captured = capsys.readouterr()
        assert f"[spec-review:council-complete] run_id={run_id}" in captured.err
        assert "elapsed=" in captured.err
        assert "status=failed" in captured.err
        assert "[spec-review:timeout]" in captured.err
        assert "sonnet_done=False" in captured.err
        assert "council_done=True" in captured.err
    finally:
        if council_path and council_path.exists():
            council_path.unlink()
        out = _CLAUDE_QUEUE_COMPLETED / f"{tid}-output.md"
        if out.exists():
            out.unlink()


# ---------------------------------------------------------------------------
# No cancellation signal: task continues after poll returns
# ---------------------------------------------------------------------------

def test_no_cancel_signal_sent(tmp_path):
    """After poll returns on timeout, no side-channel cancel is sent.

    Verified by checking that the dispatched task's output directory is
    untouched — the poll loop does not delete or modify any queue files.
    """
    tid = f"no-cancel-{int(time.time())}"
    run_id = f"no-cancel-council-{int(time.time())}"
    out_path = None
    try:
        out_path = _write_reviewer_output(tid)
        start = time.time() - 200  # definitely timed out

        _poll_until_terminal(
            spec_reviewer_task_id=tid,
            council_run_id=run_id,
            timeout_s=10,
            start_time=start,
        )

        # The reviewer output file must still exist (not deleted by poll loop)
        assert out_path.exists(), "poll loop must not delete output files"
    finally:
        if out_path and out_path.exists():
            out_path.unlink()
        council_yaml = _COUNCIL_DIR / f"{run_id}.yaml"
        if council_yaml.exists():
            council_yaml.unlink()
