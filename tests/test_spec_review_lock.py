"""Unit tests for spec-review cross-session flock serialization.

Covers AC1–AC5 and AC7 from lapis-pm-spec-review-serial-lock-v0:
  AC1: Two concurrent callers serialize (second blocks until first releases).
  AC2: A holder that dies auto-releases; subsequent caller acquires immediately.
  AC3: A waiting caller logs "queued" with the holder's spec_path/pid.
  AC4: Fail-CLOSED on timeout: abort + Pushover alert, never proceed.
  AC5: Single-caller path: lock acquired immediately, no added latency.
  AC7: Lock path = /run/user/<uid>/lapis-pm-spec-review.lock, independent of
       XDG_RUNTIME_DIR/cwd; missing dir is fatal config error (no fallback).

flock is per-open-fd: tests use _spec_review_lock directly with a tmp-dir override,
or subprocesses/threads that each open the file independently.
"""
from __future__ import annotations

import io
import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.spec_review import (
    _SPEC_REVIEW_LOCK_TIMEOUT_DEFAULT,
    _read_lock_holder,
    _spec_review_lock,
    _spec_review_lock_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lock_with_tmp(tmp_path: Path, spec_path: Path | None = None, **kwargs):
    """Return _spec_review_lock using a tmp-dir lock file."""
    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    return _spec_review_lock(
        spec_path or Path("/tmp/test-spec.md"),
        _lock_path_override=lock_file,
        **kwargs,
    )


def _hold_lock_in_subprocess(lock_file: str, held_event_fd: int, release_event_fd: int):
    """Subprocess target: acquire lock, signal held, wait for release signal."""
    import fcntl as _fcntl
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    _fcntl.flock(fd, _fcntl.LOCK_EX)
    # Write holder identity
    holder_data = json.dumps({
        "pid": os.getpid(),
        "spec_path": "/tmp/holder-spec.md",
        "started_at": "2026-06-19T00:00:00Z",
        "host": "test-host",
    }).encode()
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, holder_data)
    # Signal that the lock is held
    os.write(held_event_fd, b"\x01")
    os.close(held_event_fd)
    # Wait for release signal
    os.read(release_event_fd, 1)
    os.close(release_event_fd)
    os.close(fd)


# ---------------------------------------------------------------------------
# AC5: Single-caller path — immediate acquire, no latency
# ---------------------------------------------------------------------------

def test_single_caller_acquires_immediately(tmp_path):
    """AC5: no contention → lock acquired without delay."""
    t0 = time.monotonic()
    with _lock_with_tmp(tmp_path):
        elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"single-caller lock should be instant, took {elapsed:.2f}s"


def test_single_caller_lockfile_contains_holder_identity(tmp_path):
    """AC5: after acquire, lockfile contains pid/spec_path/started_at/host."""
    lock_file = tmp_path / "lapis-pm-spec-review.lock"
    spec = Path("/tmp/my-spec.md")
    with _spec_review_lock(spec, _lock_path_override=lock_file):
        raw = lock_file.read_bytes()
        info = json.loads(raw)
    assert info["pid"] == os.getpid()
    assert info["spec_path"] == str(spec)
    assert "started_at" in info
    assert "host" in info


# ---------------------------------------------------------------------------
# AC1: Two concurrent callers serialize
# ---------------------------------------------------------------------------

def test_two_threads_serialize(tmp_path):
    """AC1: second caller blocks until first releases; both succeed in order."""
    order = []
    lock_file = tmp_path / "lapis-pm-spec-review.lock"

    def first():
        with _spec_review_lock(Path("/spec1.md"), _lock_path_override=lock_file):
            order.append("first-in")
            time.sleep(0.3)
            order.append("first-out")

    def second():
        # Give first a moment to acquire before we try
        time.sleep(0.05)
        with _spec_review_lock(
            Path("/spec2.md"),
            _lock_path_override=lock_file,
        ):
            order.append("second-in")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert order == ["first-in", "first-out", "second-in"], (
        f"second caller should wait until first releases; got order={order}"
    )


# ---------------------------------------------------------------------------
# AC2: Holder death auto-releases (kernel flock)
# ---------------------------------------------------------------------------

def test_holder_death_autoreleases(tmp_path):
    """AC2: when the holder process dies, the lock is released by the kernel."""
    lock_file = tmp_path / "lapis-pm-spec-review.lock"

    # Pipe pair: child signals "held", parent signals "release"
    held_r, held_w = os.pipe()
    release_r, release_w = os.pipe()

    proc = multiprocessing.Process(
        target=_hold_lock_in_subprocess,
        args=(str(lock_file), held_w, release_r),
    )
    proc.start()
    os.close(held_w)
    os.close(release_r)

    # Wait until child signals it holds the lock
    os.read(held_r, 1)
    os.close(held_r)

    # Kill the child without signaling release — simulates crash
    proc.terminate()
    proc.join(timeout=3)
    os.close(release_w)

    # Kernel should have released the flock; we should acquire immediately
    t0 = time.monotonic()
    with _spec_review_lock(Path("/spec.md"), _lock_path_override=lock_file):
        elapsed = time.monotonic() - t0

    assert elapsed < 2.0, (
        f"lock should auto-release on holder death; took {elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# AC3: Waiting caller logs queued message naming holder's spec_path/pid
# ---------------------------------------------------------------------------

def test_waiter_logs_queued_with_holder_identity(tmp_path):
    """AC3: while waiting, the caller logs 'queued' once with holder pid/spec_path."""
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

    stderr_buf = io.StringIO()
    acquired = threading.Event()

    def waiter():
        with patch("sys.stderr", stderr_buf):
            with _spec_review_lock(Path("/waiter-spec.md"), _lock_path_override=lock_file):
                acquired.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

    # Give the waiter a moment to log the queued message
    time.sleep(0.3)

    # Release the holder
    os.write(release_w, b"\x01")
    os.close(release_w)
    proc.join(timeout=3)
    t.join(timeout=5)
    acquired.wait(timeout=5)

    stderr_out = stderr_buf.getvalue()
    assert "lock-queued" in stderr_out, f"expected 'lock-queued' in stderr; got:\n{stderr_out}"
    assert "pid=" in stderr_out, f"expected pid in queued log; got:\n{stderr_out}"
    assert "/tmp/holder-spec.md" in stderr_out, (
        f"expected holder spec_path in queued log; got:\n{stderr_out}"
    )


# ---------------------------------------------------------------------------
# AC4: Fail-CLOSED on timeout — abort + Pushover, never proceed
# ---------------------------------------------------------------------------

def test_timeout_raises_and_sends_pushover(tmp_path):
    """AC4: when lock held past timeout, waiter raises RuntimeError + sends NORMAL Pushover."""
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

    notify_calls = []

    def mock_send_notification(message, title, priority):
        notify_calls.append({"message": message, "title": title, "priority": priority})

    mock_priority = MagicMock()
    mock_priority.NORMAL = "normal"
    with patch.dict(os.environ, {"SPEC_REVIEW_LOCK_TIMEOUT": "3"}):
        with patch.dict(
            "sys.modules",
            {"agents_core.notify": MagicMock(
                send_notification=mock_send_notification,
                Priority=mock_priority,
            )},
        ):
            with pytest.raises(RuntimeError, match="lock-timeout"):
                with _spec_review_lock(
                    Path("/waiter-spec.md"),
                    _lock_path_override=lock_file,
                ):
                    pass  # should never reach here

    # Release and clean up
    os.write(release_w, b"\x01")
    os.close(release_w)
    proc.join(timeout=3)

    # AC4: notification was sent with NORMAL priority
    assert len(notify_calls) == 1, f"expected 1 Pushover call; got {len(notify_calls)}"
    assert notify_calls[0]["priority"] == "normal", (
        f"expected priority='normal'; got {notify_calls[0]['priority']!r}"
    )


def test_timeout_does_not_proceed(tmp_path):
    """AC4: on timeout, the body of the with block is never executed."""
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

    body_executed = False

    with patch.dict(os.environ, {"SPEC_REVIEW_LOCK_TIMEOUT": "3"}):
        with patch.dict("sys.modules", {
            "agents_core.notify": MagicMock(
                send_notification=MagicMock(),
                Priority=MagicMock(NORMAL="normal"),
            ),
        }):
            try:
                with _spec_review_lock(
                    Path("/waiter-spec.md"),
                    _lock_path_override=lock_file,
                ):
                    body_executed = True
            except RuntimeError:
                pass

    os.write(release_w, b"\x01")
    os.close(release_w)
    proc.join(timeout=3)

    assert not body_executed, "body must not execute on timeout (fail-CLOSED)"


# ---------------------------------------------------------------------------
# AC7: Lock path determinism
# ---------------------------------------------------------------------------

def test_lock_path_is_canonical_getuid(tmp_path):
    """AC7: resolved lock path uses os.getuid(), not XDG_RUNTIME_DIR or cwd."""
    uid = os.getuid()
    expected = Path(f"/run/user/{uid}/lapis-pm-spec-review.lock")

    # Changing XDG_RUNTIME_DIR must NOT affect the resolved path
    with patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(tmp_path)}):
        with patch.object(Path, "is_dir", return_value=True):
            result = _spec_review_lock_path()

    assert result == expected, (
        f"lock path must be /run/user/<uid>/... independent of XDG_RUNTIME_DIR; "
        f"got {result}"
    )


def test_lock_path_missing_runuser_dir_is_fatal(tmp_path):
    """AC7: if /run/user/<uid> does not exist, abort with RuntimeError — no fallback."""
    uid = os.getuid()
    with patch.object(Path, "is_dir", return_value=False):
        with pytest.raises(RuntimeError, match="fatal config error"):
            _spec_review_lock_path()


def test_lock_path_no_xdg_fallback(tmp_path):
    """AC7: lock path never falls back to /tmp or XDG_RUNTIME_DIR even if /run/user absent."""
    with patch.object(Path, "is_dir", return_value=False):
        with pytest.raises(RuntimeError):
            _spec_review_lock_path()
        # Verify no tmp or home path was silently returned
        # (test passes if RuntimeError was raised — no silent fallback)


# ---------------------------------------------------------------------------
# Misc: default timeout constant
# ---------------------------------------------------------------------------

def test_default_timeout_is_900s():
    """Default lock timeout is 900s — longer than a normal review so queuing is the norm."""
    assert _SPEC_REVIEW_LOCK_TIMEOUT_DEFAULT == 900
