"""Cross-process ledger-lock tests (deploy-inventory-repair-ledger-lock-v0).

DoD-1/2/5 require genuine OS processes, not threads and not a mocked lock -
`threading.Lock` would pass a thread test while providing nothing against the
actual cross-process contention model this unit exists to close. Workers run
via `tests/_ledger_lock_worker.py` as real `subprocess.Popen` processes.

DoD-3/4 (contended -> skip, bounded timeout) and part of DoD-5 run in-process
against the real module, holding the lock directly via `lapis_pm.file_lock`
on a second file descriptor - `flock` is scoped to the open file description,
not the process, so two `file_lock()` calls in the same test process still
contend correctly.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lapis_pm import deploy_inventory_repair as dir_mod
from lapis_pm.file_lock import file_lock

_WORKER = str(Path(__file__).resolve().parent / "_ledger_lock_worker.py")
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def _subprocess_env():
    env = dict(os.environ)
    extra = [
        _REPO_ROOT,
        "/srv/git/archetypes-core",
        "/home/user/.local/lib/python3.12/site-packages",
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra + ([existing] if existing else []))
    return env


def _spawn(mode, ledger_path, lock_path, clone_path="", finding_kind="", sleep_s="0",
           timeout_s=""):
    return subprocess.Popen(
        [sys.executable, _WORKER, mode, str(ledger_path), str(lock_path),
         clone_path, finding_kind, str(sleep_s), str(timeout_s)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_subprocess_env(),
    )


class TestCrossProcessMutualExclusion:
    def test_two_processes_do_not_lose_each_others_entry(self, tmp_path):
        """DoD-1/DoD-2: process A reads the ledger, holds the lock across a
        slow 'diagnosis' (simulating the GW call), and writes its entry;
        process B does the same for a different finding. Without the lock
        this is exactly the lost-update shape (last writer wins wholesale)
        the spec names - the loser's entry vanishes. Under the lock both
        entries must survive.
        """
        ledger_path = tmp_path / "ledger.json"
        lock_path = tmp_path / "ledger.lock"

        proc_a = _spawn("repair", ledger_path, lock_path,
                         clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                         sleep_s="1.0", timeout_s="10")
        # Give A a head start so it acquires the lock first.
        time.sleep(0.2)
        proc_b = _spawn("repair", ledger_path, lock_path,
                         clone_path="/srv/git/bar", finding_kind="stray_branch",
                         sleep_s="0.1", timeout_s="10")

        out_a, err_a = proc_a.communicate(timeout=30)
        out_b, err_b = proc_b.communicate(timeout=30)
        assert proc_a.returncode == 0, err_a
        assert proc_b.returncode == 0, err_b

        actions_a = json.loads(out_a)
        actions_b = json.loads(out_b)
        assert any(a.startswith("deposited:") for a in actions_a), actions_a
        assert any(a.startswith("deposited:") for a in actions_b), actions_b

        ledger = json.loads(ledger_path.read_text())
        sig_a = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        sig_b = dir_mod.compute_signature("/srv/git/bar", "stray_branch")
        # DoD-2: neither writer's entry is erased by the other's whole-file write.
        assert sig_a in ledger, "process A's entry was lost"
        assert sig_b in ledger, "process B's entry was lost"

    def test_run_repair_pass_and_propose_corrupt_status_share_the_lock(self, tmp_path):
        """DoD-5: propose_corrupt_status (its own independent read_ledger/
        write_ledger) and run_repair_pass are mutually exclusive across
        processes - both take the same lock, so the ledger-write regions
        cannot interleave and lose each other's entries either.
        """
        ledger_path = tmp_path / "ledger.json"
        lock_path = tmp_path / "ledger.lock"

        proc_repair = _spawn("repair", ledger_path, lock_path,
                              clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                              sleep_s="1.0", timeout_s="10")
        time.sleep(0.2)
        proc_corrupt = _spawn("corrupt", ledger_path, lock_path, timeout_s="10")

        out_repair, err_repair = proc_repair.communicate(timeout=30)
        out_corrupt, err_corrupt = proc_corrupt.communicate(timeout=30)
        assert proc_repair.returncode == 0, err_repair
        assert proc_corrupt.returncode == 0, err_corrupt

        actions_repair = json.loads(out_repair)
        action_corrupt = json.loads(out_corrupt)
        assert any(a.startswith("deposited:") for a in actions_repair), actions_repair
        assert action_corrupt == "deposited:corrupt-status"

        ledger = json.loads(ledger_path.read_text())
        sig_repair = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        assert sig_repair in ledger
        assert dir_mod._CORRUPT_STATUS_SIGNATURE in ledger


class TestContendedSkipFailsClosed:
    def test_run_repair_pass_skips_and_never_deposits_when_locked(self, tmp_path):
        """DoD-3: with the lock already held, a repair pass deposits
        nothing, and returns the distinguishable skip action - never
        proceeds unlocked."""
        ledger_path = tmp_path / "ledger.json"
        lock_path = tmp_path / "ledger.lock"
        dir_mod._LEDGER_FILE = ledger_path
        dir_mod._LEDGER_LOCK_FILE = lock_path

        status = {"clones": [{
            "path": "/srv/git/foo", "mapped": True, "branch": "main",
            "commits_behind": 1, "backing_units": [],
            "findings": [{"kind": "stale_behind_origin", "severity": "HIGH", "detail": "x"}],
        }]}

        with file_lock(lock_path, timeout_s=10):
            with (
                patch.object(dir_mod, "DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS", 0.3),
                patch.object(dir_mod, "deposit_gem") as mock_deposit,
                patch.object(dir_mod, "diagnose_finding") as mock_diagnose,
            ):
                actions = dir_mod.run_repair_pass(status, prior_high_keys=set(), corrupt=False)

        assert actions == [dir_mod.LEDGER_LOCK_CONTENDED_ACTION]
        assert not mock_deposit.called
        assert not mock_diagnose.called

    def test_propose_corrupt_status_skips_and_never_deposits_when_locked(self, tmp_path):
        ledger_path = tmp_path / "ledger.json"
        lock_path = tmp_path / "ledger.lock"
        dir_mod._LEDGER_FILE = ledger_path
        dir_mod._LEDGER_LOCK_FILE = lock_path

        with file_lock(lock_path, timeout_s=10):
            with (
                patch.object(dir_mod, "DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS", 0.3),
                patch.object(dir_mod, "deposit_gem") as mock_deposit,
            ):
                action = dir_mod.propose_corrupt_status()

        assert action == dir_mod.LEDGER_LOCK_CONTENDED_ACTION
        assert not mock_deposit.called

    def test_warning_logged_on_contention(self, tmp_path, caplog):
        ledger_path = tmp_path / "ledger.json"
        lock_path = tmp_path / "ledger.lock"
        dir_mod._LEDGER_FILE = ledger_path
        dir_mod._LEDGER_LOCK_FILE = lock_path

        with file_lock(lock_path, timeout_s=10):
            with (
                patch.object(dir_mod, "DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS", 0.3),
                caplog.at_level("WARNING"),
            ):
                dir_mod.propose_corrupt_status()

        assert any("ledger lock contended" in rec.message for rec in caplog.records)


class TestContendedTimeoutBounded:
    def test_timeout_is_bounded_by_configured_value_not_a_gw_length_wait(self, tmp_path):
        """DoD-4: a contended pass returns in roughly the configured timeout,
        not after (e.g.) a 240s GW-length wait."""
        ledger_path = tmp_path / "ledger.json"
        lock_path = tmp_path / "ledger.lock"
        dir_mod._LEDGER_FILE = ledger_path
        dir_mod._LEDGER_LOCK_FILE = lock_path

        timeout_s = 0.5
        status = {"clones": []}
        with file_lock(lock_path, timeout_s=10):
            with patch.object(dir_mod, "DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS", timeout_s):
                start = time.monotonic()
                actions = dir_mod.run_repair_pass(status, prior_high_keys=set(), corrupt=False)
                elapsed = time.monotonic() - start

        assert actions == [dir_mod.LEDGER_LOCK_CONTENDED_ACTION]
        # Bounded by the configured timeout with slack for scheduling jitter,
        # nowhere near a GW-length (240s) wait.
        assert elapsed < timeout_s + 2.0


class TestAtomicWritePreservedUnderLock:
    def test_write_ledger_body_unchanged(self, tmp_path):
        """D4/DoD-6 (belt-and-suspenders alongside the parent's unmodified
        TestLedgerAtomicWrite): write_ledger still round-trips and is still
        the atomic temp-then-replace implementation - the lock wraps it, it
        does not replace it."""
        ledger_file = tmp_path / "ledger.json"
        ledger = {"abc123": {"gem_id": "g1", "status": "open"}}
        dir_mod.write_ledger(ledger, path=ledger_file)
        assert dir_mod.read_ledger(path=ledger_file) == ledger

        with patch("os.replace", side_effect=OSError("disk full")):
            dir_mod.write_ledger({"new": {"status": "open"}}, path=ledger_file)
        # prior content untouched on a failed replace
        assert json.loads(ledger_file.read_text()) == ledger
