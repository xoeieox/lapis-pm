"""Tests for lapis_pm.file_lock (deploy-inventory-repair-ledger-lock-v0).

Stdlib-only leaf module - these tests have no agents-core dependency and
exercise the lock primitive directly: mutual exclusion across genuine OS
processes (DoD-1), bounded timeout (DoD-4), release on process death
(DoD-7), and EINTR resilience (DoD-10).
"""
import multiprocessing
import os
import signal
import time
from pathlib import Path

import pytest

from lapis_pm.file_lock import FileLockTimeout, file_lock


def _hold_lock_and_record(lock_path, hold_s, order_list, idx):
    with file_lock(lock_path, timeout_s=10):
        order_list.append(idx)
        time.sleep(hold_s)


def _acquire_once(lock_path, hold_s):
    with file_lock(lock_path, timeout_s=30):
        time.sleep(hold_s)


def _hold_then_die(lock_path, ready_event):
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_EX)
    ready_event.set()
    # Die without releasing - flock must still be released by the kernel.
    os.kill(os.getpid(), signal.SIGKILL)


class TestCrossProcessMutualExclusion:
    def test_two_processes_serialize_not_interleave(self, tmp_path):
        """DoD-1: genuine OS processes (multiprocessing.Process, not
        threading.Thread) contending for the same lock file serialize -
        the second cannot enter its critical section until the first
        releases."""
        lock_path = tmp_path / "test.lock"
        manager = multiprocessing.Manager()
        order = manager.list()

        p1 = multiprocessing.Process(target=_hold_lock_and_record, args=(lock_path, 0.5, order, 1))
        p2 = multiprocessing.Process(target=_hold_lock_and_record, args=(lock_path, 0.1, order, 2))
        p1.start()
        time.sleep(0.15)  # ensure p1 acquires first
        p2.start()
        p1.join(timeout=10)
        p2.join(timeout=10)

        assert p1.exitcode == 0
        assert p2.exitcode == 0
        assert list(order) == [1, 2]


class TestTimeoutBounded:
    def test_timeout_raises_and_is_bounded(self, tmp_path):
        """DoD-4: acquisition against an already-held lock times out at
        roughly the configured value, not indefinitely."""
        lock_path = tmp_path / "test.lock"
        holder = multiprocessing.Process(target=_acquire_once, args=(lock_path, 2.0))
        holder.start()
        time.sleep(0.2)  # ensure holder has the lock

        start = time.monotonic()
        with pytest.raises(FileLockTimeout):
            with file_lock(lock_path, timeout_s=0.5):
                pass
        elapsed = time.monotonic() - start
        assert 0.4 < elapsed < 2.0  # bounded near the configured timeout

        holder.join(timeout=10)
        assert holder.exitcode == 0


class TestReleaseOnProcessDeath:
    def test_killed_holder_does_not_wedge_next_acquisition(self, tmp_path):
        """DoD-7: flock is released at the kernel level on fd close *and on
        process death*. A process SIGKILLed while holding the lock must not
        wedge the next acquisition - asserted rather than assumed."""
        lock_path = tmp_path / "test.lock"
        manager = multiprocessing.Manager()
        ready = manager.Event()

        holder = multiprocessing.Process(target=_hold_then_die, args=(lock_path, ready))
        holder.start()
        assert ready.wait(timeout=10), "holder never acquired the lock"
        holder.join(timeout=10)
        assert holder.exitcode is not None and holder.exitcode != 0  # died via SIGKILL

        # A fresh acquisition must succeed promptly - no reaper needed.
        start = time.monotonic()
        with file_lock(lock_path, timeout_s=5):
            pass
        elapsed = time.monotonic() - start
        assert elapsed < 2.0


class TestEintrResilience:
    def test_signal_during_contended_wait_does_not_crash_acquisition(self, tmp_path):
        """DoD-10 (Facets transmuter gate finding, mandatory): a signal
        delivered while the helper is waiting on a contended lock must not
        raise InterruptedError/OSError(EINTR) - it must retry until the
        deadline, then raise the typed timeout. Uses a real SIGALRM with a
        handler installed so the poll loop's own syscalls are genuinely
        interrupted, not just asserted from PEP 475 behaviour."""
        lock_path = tmp_path / "test.lock"
        holder = multiprocessing.Process(target=_acquire_once, args=(lock_path, 1.5))
        holder.start()
        time.sleep(0.1)

        def _noop_handler(signum, frame):
            pass

        old_handler = signal.signal(signal.SIGALRM, _noop_handler)
        try:
            # Fire a real signal partway through the contended wait.
            signal.setitimer(signal.ITIMER_REAL, 0.2, 0.2)
            try:
                with file_lock(lock_path, timeout_s=3):
                    pass
            except FileLockTimeout:
                pytest.fail(
                    "file_lock raised FileLockTimeout despite the holder "
                    "releasing well within the 3s budget - EINTR handling "
                    "likely ate wall-clock time it shouldn't have, or "
                    "propagated an error that unwound the loop early"
                )
            except (InterruptedError, OSError) as exc:
                pytest.fail(f"EINTR propagated instead of being retried: {exc!r}")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

        holder.join(timeout=10)
        assert holder.exitcode == 0

    def test_signal_storm_eventually_times_out_cleanly(self, tmp_path):
        """A lock that can never be acquired (holder never releases within
        the window) must still raise the typed timeout - not an
        interrupted-syscall error - even under a continuous signal storm."""
        lock_path = tmp_path / "test.lock"
        holder = multiprocessing.Process(target=_acquire_once, args=(lock_path, 5.0))
        holder.start()
        time.sleep(0.1)

        def _noop_handler(signum, frame):
            pass

        old_handler = signal.signal(signal.SIGALRM, _noop_handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.05, 0.05)
            try:
                with file_lock(lock_path, timeout_s=0.6):
                    pass
                pytest.fail("expected FileLockTimeout")
            except FileLockTimeout:
                pass  # correct: typed timeout, not an unhandled EINTR error
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            holder.terminate()
            holder.join(timeout=10)
