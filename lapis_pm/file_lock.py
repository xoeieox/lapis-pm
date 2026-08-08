"""Stdlib-only leaf module: a path-parameterised `flock` context manager.

Exists because `pm_core._friction_queue_lock` (`pm_core.py:5946-5984`) cannot be
imported here: `pm_core` imports `deploy_inventory_repair`
(`pm_core.py:1626`), and `deploy_inventory_repair` needs this lock, so
importing `pm_core` back would be circular. A leaf module - stdlib only,
imported by `deploy_inventory_repair` and nothing else pulling it back in -
cannot participate in that cycle.

Mirrors `_friction_queue_lock`'s shape deliberately: `os.open(O_CREAT|O_RDWR,
0o600)` -> `fcntl.flock(fd, LOCK_EX|LOCK_NB)` poll loop with a deadline and
`time.sleep(0.1)`, `os.close(fd)` in `finally`, a typed timeout on
exhaustion. Do not invent a new locking strategy here.

`flock` is released automatically on fd close *and on process death*, so
there is no stale-lock class and no reaper is needed - this is a real mutex,
unlike the deploy-pull sentinel (`pm_core.py:645-661`), which is a
human-cleared JSON circuit-breaker.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import time
from pathlib import Path

LOCK_TIMEOUT_DEFAULT = 5.0


class FileLockTimeout(Exception):
    """Raised when a `flock` is not acquired within the timeout."""


@contextlib.contextmanager
def file_lock(lock_path: Path | str, timeout_s: float | None = None):
    """Exclusive flock on `lock_path`, polling every 0.1s until acquired or
    `timeout_s` elapses.

    Raises `FileLockTimeout` on timeout - callers decide the failure mode.

    A signal delivered while waiting on `flock` or `sleep` (operator Ctrl-C,
    a `SIGTERM` from `TimeoutStartSec`, any handled signal) can raise
    `InterruptedError`/`OSError(errno=EINTR)`. That must be treated as
    "retry until the deadline", not as an acquisition failure and not as an
    unhandled crash - a lock helper that dies on a signal turns a routine
    timeout into a stack trace inside a pass whose whole contract is
    fail-soft. PEP 475 (3.5+) retries most interrupted syscalls
    automatically, but not every path is covered, so this is handled
    explicitly rather than assumed from the runtime version.
    """
    if timeout_s is None:
        timeout_s = LOCK_TIMEOUT_DEFAULT
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise FileLockTimeout(
                        f"file lock timed out after {timeout_s}s: {lock_path}"
                    )
                try:
                    time.sleep(0.1)
                except InterruptedError:
                    # EINTR mid-sleep: retry until the deadline.
                    pass
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    # EINTR mid-flock: retry until the deadline.
                    if time.monotonic() >= deadline:
                        raise FileLockTimeout(
                            f"file lock timed out after {timeout_s}s: {lock_path}"
                        )
                    continue
                raise
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
