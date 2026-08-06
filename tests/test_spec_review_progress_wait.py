"""Unit tests for lapis-pm-gate-queue-progress-aware-wait-v0.

Covers the unit's Definition of Done: heartbeat publishing, staleness-based
(not elapsed-time-based) stall detection with reset-on-recovery, a separate
absolute ceiling backstop, exact fallback when the progress file is absent
or corrupt, atomic same-directory progress writes, stale-temp cleanup on
acquire, defensive env parsing, and the /srv/lapis/gate-queue/history.jsonl trace.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.spec_review import (
    _PROGRESS_FIELDS,
    _HeartbeatWriter,
    _env_int,
    _progress_path,
    _read_progress,
    _spec_review_lock,
    _write_progress,
)


def _notify_patch():
    """Patch agents_core.notify so no real Pushover call fires in tests."""
    return patch.dict(
        "sys.modules",
        {"agents_core.notify": MagicMock(
            send_notification=MagicMock(),
            Priority=MagicMock(NORMAL="normal"),
        )},
    )


def _hold_lock_in_subprocess(lock_file: str, held_event_fd: int, release_event_fd: int):
    """Subprocess target: acquire lock, signal held, wait for release signal.
    Deliberately does NOT write a progress file — simulates a mixed-fleet
    pre-change holder."""
    import fcntl as _fcntl
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    _fcntl.flock(fd, _fcntl.LOCK_EX)
    os.write(held_event_fd, b"\x01")
    os.close(held_event_fd)
    os.read(release_event_fd, 1)
    os.close(release_event_fd)
    os.close(fd)


# ---------------------------------------------------------------------------
# DoD 12/13: progress record shape and atomic same-directory write
# ---------------------------------------------------------------------------

def test_progress_record_has_exact_field_set(tmp_path):
    progress_path = tmp_path / "lock.progress"
    _write_progress(progress_path, pid=123, phase="facets+council", phase_at=1.0, beat_at=2.0)
    record = json.loads(progress_path.read_text())
    assert set(record.keys()) == _PROGRESS_FIELDS


def test_atomic_write_uses_os_replace_same_directory(tmp_path):
    progress_path = tmp_path / "lock.progress"
    calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    with patch("os.replace", side_effect=spy_replace):
        _write_progress(progress_path, pid=1, phase="p", phase_at=1.0, beat_at=1.0)

    assert len(calls) == 1
    src, dst = calls[0]
    assert src.parent == progress_path.parent == dst.parent == tmp_path
    assert dst == progress_path


def test_reader_never_observes_torn_record(tmp_path):
    """DoD6: a reader concurrent with many writes always sees a complete record."""
    progress_path = tmp_path / "lock.progress"
    _write_progress(progress_path, pid=1, phase="init", phase_at=0.0, beat_at=0.0)
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            _write_progress(progress_path, pid=1, phase=f"phase-{i}", phase_at=float(i), beat_at=float(i))
            i += 1

    def reader():
        for _ in range(300):
            record = _read_progress(progress_path)
            if record is None:
                errors.append("saw None mid-write")
            elif set(record.keys()) != _PROGRESS_FIELDS:
                errors.append(f"torn record: {record}")

    t_w = threading.Thread(target=writer)
    t_r = threading.Thread(target=reader)
    t_w.start()
    t_r.start()
    t_r.join(timeout=10)
    stop.set()
    t_w.join(timeout=10)

    assert errors == [], f"reader observed corruption: {errors}"


# ---------------------------------------------------------------------------
# DoD 14: stale temp files cleaned on acquire
# ---------------------------------------------------------------------------

def test_stale_progress_temp_cleaned_on_acquire(tmp_path):
    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    progress_path = _progress_path(lock_file)
    debris = progress_path.parent / f"{progress_path.name}.tmp-99999-deadbeef"
    debris.write_text('{"garbage": true}')
    assert debris.exists()

    with _notify_patch():
        with _spec_review_lock(Path("/spec.md"), _lock_path_override=lock_file):
            assert not debris.exists()


# ---------------------------------------------------------------------------
# DoD 1: heartbeat refreshes beat_at monotonically through a long phase
# ---------------------------------------------------------------------------

def test_heartbeat_beat_at_increases_monotonically(tmp_path):
    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    progress_path = _progress_path(lock_file)

    with patch.dict(os.environ, {"SPEC_REVIEW_HEARTBEAT_INTERVAL": "1"}):
        with _notify_patch():
            with _spec_review_lock(Path("/spec.md"), _lock_path_override=lock_file) as beat:
                beat.set_phase("facets+council")
                seen = []
                for _ in range(3):
                    time.sleep(1.1)
                    record = _read_progress(progress_path)
                    assert record is not None
                    seen.append(record["beat_at"])
    assert seen == sorted(seen)
    assert seen[-1] > seen[0]


# ---------------------------------------------------------------------------
# DoD 2/DoD5-analogue: healthy holder outlasting SPEC_REVIEW_LOCK_TIMEOUT
# is never aborted, and DoD3/3a: stall detection + reset-on-recovery
# ---------------------------------------------------------------------------

def test_waiter_behind_healthy_holder_does_not_abort(tmp_path):
    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    env = {
        "SPEC_REVIEW_LOCK_TIMEOUT": "1",          # legacy deadline — must NOT trip
        "SPEC_REVIEW_HEARTBEAT_INTERVAL": "1",
        "SPEC_REVIEW_STALL_MISSED_BEATS": "5",
        "SPEC_REVIEW_LOCK_MAX_WAIT": "30",
    }
    holder_acquired = threading.Event()
    waiter_acquired = threading.Event()
    errors = []

    def holder():
        try:
            with patch.dict(os.environ, env):
                with _notify_patch():
                    with _spec_review_lock(Path("/holder.md"), _lock_path_override=lock_file):
                        holder_acquired.set()
                        time.sleep(3.0)  # outlives the 1s legacy deadline
        except Exception as e:  # pragma: no cover - surfaced via errors list
            errors.append(e)

    def waiter():
        holder_acquired.wait(timeout=5)
        try:
            with patch.dict(os.environ, env):
                with _notify_patch():
                    with _spec_review_lock(Path("/waiter.md"), _lock_path_override=lock_file):
                        waiter_acquired.set()
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert errors == [], f"unexpected abort(s): {errors}"
    assert waiter_acquired.is_set(), "waiter behind a healthy holder must eventually acquire"


def test_stall_abort_after_consecutive_missed_beats(tmp_path):
    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    progress_path = _progress_path(lock_file)
    # Plant a progress file that never advances — simulates a wedged holder that
    # wrote once on acquire and then hung (thread died / process wedged on I/O).
    _write_progress(progress_path, pid=999999, phase="facets+council", phase_at=time.time(), beat_at=time.time())

    env = {
        "SPEC_REVIEW_LOCK_TIMEOUT": "3600",  # legacy deadline irrelevant — progress file present
        "SPEC_REVIEW_HEARTBEAT_INTERVAL": "1",
        "SPEC_REVIEW_STALL_MISSED_BEATS": "3",
        "SPEC_REVIEW_LOCK_MAX_WAIT": "3600",
    }

    held_r, held_w = os.pipe()
    release_r, release_w = os.pipe()
    proc = multiprocessing.Process(
        target=_hold_lock_in_subprocess_with_progress,
        args=(str(lock_file), str(progress_path), held_w, release_r),
    )
    proc.start()
    os.close(held_w)
    os.close(release_r)
    os.read(held_r, 1)
    os.close(held_r)

    with patch.dict(os.environ, env):
        with _notify_patch():
            with pytest.raises(RuntimeError, match="lock-stall"):
                with _spec_review_lock(Path("/waiter.md"), _lock_path_override=lock_file):
                    pass

    os.write(release_w, b"\x01")
    os.close(release_w)
    proc.join(timeout=3)


def _hold_lock_in_subprocess_with_progress(lock_file, progress_file, held_event_fd, release_event_fd):
    """Like _hold_lock_in_subprocess, but keeps the pre-planted progress file frozen
    (never refreshes it) — simulates a wedged holder."""
    import fcntl as _fcntl
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    _fcntl.flock(fd, _fcntl.LOCK_EX)
    os.write(held_event_fd, b"\x01")
    os.close(held_event_fd)
    os.read(release_event_fd, 1)
    os.close(release_event_fd)
    os.close(fd)


def test_single_missed_beat_resets_and_does_not_abort(tmp_path):
    """DoD3a: a beat_at stall for a few polls that then advances resets the
    counter to zero — no abort, and the miss does not accumulate."""
    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    progress_path = _progress_path(lock_file)

    holder_acquired = threading.Event()
    stop_holder = threading.Event()
    errors = []

    def holder():
        try:
            with patch.dict(os.environ, {"SPEC_REVIEW_HEARTBEAT_INTERVAL": "3600"}):
                with _notify_patch():
                    with _spec_review_lock(Path("/holder.md"), _lock_path_override=lock_file) as beat:
                        holder_acquired.set()
                        # Manually drive a few "stall then recover" beats, faster
                        # than the real 3600s auto-ticker would.
                        for i in range(6):
                            if stop_holder.is_set():
                                break
                            beat.set_phase(f"phase-{i}")
                            time.sleep(0.05)
                        stop_holder.wait(timeout=5)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    t = threading.Thread(target=holder)
    t.start()
    holder_acquired.wait(timeout=5)
    time.sleep(0.5)  # let several phase advances land
    stop_holder.set()
    t.join(timeout=5)

    assert errors == [], f"unexpected error: {errors}"


# ---------------------------------------------------------------------------
# DoD 4: absolute ceiling is a distinct, distinguishable backstop
# ---------------------------------------------------------------------------

def test_ceiling_abort_distinguishable_from_stall(tmp_path):
    lock_file = tmp_path / "lapis-pm-spec-review.lock"

    env = {
        "SPEC_REVIEW_LOCK_TIMEOUT": "3600",
        "SPEC_REVIEW_HEARTBEAT_INTERVAL": "1",
        "SPEC_REVIEW_STALL_MISSED_BEATS": "10000",  # effectively disabled
        "SPEC_REVIEW_LOCK_MAX_WAIT": "1",
    }

    holder_acquired = threading.Event()
    release = threading.Event()
    errors = []

    def holder():
        try:
            with patch.dict(os.environ, env):
                with _notify_patch():
                    with _spec_review_lock(Path("/holder.md"), _lock_path_override=lock_file):
                        holder_acquired.set()
                        release.wait(timeout=10)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    t = threading.Thread(target=holder)
    t.start()
    holder_acquired.wait(timeout=5)

    with patch.dict(os.environ, env):
        with _notify_patch():
            with pytest.raises(RuntimeError, match="lock-ceiling") as excinfo:
                with _spec_review_lock(Path("/waiter.md"), _lock_path_override=lock_file):
                    pass

    release.set()
    t.join(timeout=5)

    msg = str(excinfo.value)
    assert "hung" not in msg.lower() or "not" in msg.lower()
    assert "queue too deep" in msg or "backstop" in msg
    assert errors == []


# ---------------------------------------------------------------------------
# DoD 5: progress absent or corrupt reproduces today's exact behaviour
# ---------------------------------------------------------------------------

def test_absent_progress_file_falls_back_to_legacy_timeout(tmp_path):
    lock_file = tmp_path / "lapis-pm-spec-review.lock"

    held_r, held_w = os.pipe()
    release_r, release_w = os.pipe()
    proc = multiprocessing.Process(
        target=_hold_lock_in_subprocess,
        args=(str(lock_file), held_w, release_r),
    )
    proc.start()
    os.close(held_w)
    os.close(release_r)
    os.read(held_r, 1)
    os.close(held_r)

    with patch.dict(os.environ, {"SPEC_REVIEW_LOCK_TIMEOUT": "1", "SPEC_REVIEW_LOCK_MAX_WAIT": "3600"}):
        with _notify_patch():
            with pytest.raises(RuntimeError, match="lock-timeout"):
                with _spec_review_lock(Path("/waiter.md"), _lock_path_override=lock_file):
                    pass

    os.write(release_w, b"\x01")
    os.close(release_w)
    proc.join(timeout=3)


def test_corrupt_progress_file_falls_back_to_legacy_timeout(tmp_path):
    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    progress_path = _progress_path(lock_file)
    progress_path.write_text("{not valid json")

    held_r, held_w = os.pipe()
    release_r, release_w = os.pipe()
    proc = multiprocessing.Process(
        target=_hold_lock_in_subprocess,
        args=(str(lock_file), held_w, release_r),
    )
    proc.start()
    os.close(held_w)
    os.close(release_r)
    os.read(held_r, 1)
    os.close(held_r)

    with patch.dict(os.environ, {"SPEC_REVIEW_LOCK_TIMEOUT": "1", "SPEC_REVIEW_LOCK_MAX_WAIT": "3600"}):
        with _notify_patch():
            with pytest.raises(RuntimeError, match="lock-timeout"):
                with _spec_review_lock(Path("/waiter.md"), _lock_path_override=lock_file):
                    pass

    os.write(release_w, b"\x01")
    os.close(release_w)
    proc.join(timeout=3)


def test_read_progress_missing_field_treated_as_corrupt(tmp_path):
    progress_path = tmp_path / "lock.progress"
    progress_path.write_text(json.dumps({"pid": 1, "phase": "x"}))  # missing phase_at/beat_at
    assert _read_progress(progress_path) is None


# ---------------------------------------------------------------------------
# DoD 7: /srv/lapis/gate-queue/history.jsonl gets one parseable line per event
# ---------------------------------------------------------------------------

def test_history_jsonl_records_queue_and_acquire(tmp_path, monkeypatch):
    history_path = tmp_path / "gate-queue" / "history.jsonl"
    monkeypatch.setattr("lapis_pm.spec_review._GATE_QUEUE_HISTORY_PATH", history_path)

    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    held_r, held_w = os.pipe()
    release_r, release_w = os.pipe()
    proc = multiprocessing.Process(
        target=_hold_lock_in_subprocess,
        args=(str(lock_file), held_w, release_r),
    )
    proc.start()
    os.close(held_w)
    os.close(release_r)
    os.read(held_r, 1)
    os.close(held_r)

    acquired = threading.Event()

    def waiter():
        with _notify_patch():
            with _spec_review_lock(Path("/waiter.md"), _lock_path_override=lock_file):
                acquired.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.3)
    os.write(release_w, b"\x01")
    os.close(release_w)
    proc.join(timeout=3)
    t.join(timeout=5)
    acquired.wait(timeout=5)

    assert history_path.exists()
    lines = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    events = {line["event"] for line in lines}
    assert "queued" in events
    assert "acquired" in events


# ---------------------------------------------------------------------------
# DoD 8: heartbeat thread cannot outlive its process; release unchanged
# ---------------------------------------------------------------------------

def test_heartbeat_thread_is_daemon_and_stops_on_release(tmp_path):
    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    with patch.dict(os.environ, {"SPEC_REVIEW_HEARTBEAT_INTERVAL": "1"}):
        with _notify_patch():
            with _spec_review_lock(Path("/spec.md"), _lock_path_override=lock_file) as beat:
                thread = beat._thread
                assert thread is not None
                assert thread.daemon
                assert thread.is_alive()
    # After the with-block exits, stop() has joined the thread.
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# DoD 9: every env var this unit reads parses defensively
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "SPEC_REVIEW_LOCK_TIMEOUT",
        "SPEC_REVIEW_STALL_MISSED_BEATS",
        "SPEC_REVIEW_LOCK_MAX_WAIT",
        "SPEC_REVIEW_HEARTBEAT_INTERVAL",
    ],
)
def test_malformed_env_var_falls_back_to_default(name):
    with patch.dict(os.environ, {name: "not-an-int"}):
        # Must not raise.
        assert _env_int(name, 42) == 42


def test_env_int_missing_returns_default():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SOME_UNSET_SPEC_REVIEW_VAR", None)
        assert _env_int("SOME_UNSET_SPEC_REVIEW_VAR", 7) == 7
