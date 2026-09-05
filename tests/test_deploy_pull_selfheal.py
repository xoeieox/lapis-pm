"""Hermetic tests for lapis_pm.deploy_pull_selfheal (Slice 1,
deploy-pull-selfheal-core-v0).

The rebased 30-test suite (spec Critical fix #5 gate: 38 at PR #314 head -
10 named removals + 2 new). No live weaver/Desk dependency: every test that
drives a first-detection `run_pass` monkeypatches `deposit_gem` /
`emit_provenance` (the hermeticity requirement, rev-1 M3), and the page
budget is pinned with an `agents_core.notify.send_notification` spy (I2:
no page-emitting transitions in this slice).

Coverage: D1 classifier (every row, composition, error-unknown,
fetch-failed precedence, redaction incl. the bare-token shape); window math
(active-defer vs stale); the I2 page-budget spy; I3 active-defer; ledger
bootstrap + atomic write (fsync) + corrupt quarantine + absent-file
(Critical fix #2/#5) + flock contention; I10 ack_watch/not_real
consumption; the I8 lane-constant docstrings (Critical fix #3); the
`recurred`-flag distinctness (Critical fix #7); the `_finish_success`
wiring (Critical fix #4 / DoD-8, new test); the rebased 5-cycle and
fetch_failed contracts.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from lapis_pm import deploy_pull_selfheal as dps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_cp(rc=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["git"], returncode=rc,
                                       stdout=stdout, stderr=stderr)


def _patch_git(monkeypatch, fetch_rc=0, branch="main", ahead="0", behind="0",
               porcelain="", fetch_stderr=""):
    """Patch dps._git so classify runs hermetically."""
    def fake_git(path, *args, timeout=dps.GIT_OP_TIMEOUT_S):
        if args[:1] == ("fetch",):
            return _git_cp(rc=fetch_rc, stderr=fetch_stderr)
        if args[:2] == ("branch", "--show-current"):
            return _git_cp(stdout=branch + "\n")
        if args[:1] == ("rev-list",):
            return _git_cp(stdout=f"{ahead}\t{behind}\n")
        if args[:1] == ("status",):
            return _git_cp(stdout=porcelain)
        return _git_cp()
    monkeypatch.setattr(dps, "_git", fake_git)


def _evidence(branch="main", ahead=0, behind=0, porcelain=(), fetch_rc=0,
              fetch_stderr="", mtime=None, now=None):
    now = now if now is not None else time.time()
    dirty = [l for l in porcelain if not l.startswith("??")]
    mtimes = [mtime if mtime is not None else 0.0] * len(dirty)
    return {
        "branch": branch, "ahead": ahead, "behind": behind,
        "fetch_rc": fetch_rc, "fetch_stderr": fetch_stderr,
        "porcelain": list(porcelain), "dirty_files": [l[3:] for l in dirty],
        "mtimes": mtimes, "now": now,
        "dead_branch": bool(branch != "main" and ahead == 0),
        "staged_adds": bool(
            dirty and all(l[:2] == "A " or l[:1] == "A" for l in dirty)
        ),
    }


def _mock_desk(monkeypatch, gem_id="gem-1"):
    """The hermeticity requirement: mock the Desk-facing calls."""
    deposit = MagicMock(return_value=gem_id)
    monkeypatch.setattr(dps, "deposit_gem", deposit)
    monkeypatch.setattr(dps, "emit_provenance", MagicMock())
    monkeypatch.setattr(dps, "_gem_decisions", lambda: {})
    monkeypatch.setattr(dps, "_supersede_gem", MagicMock())
    return deposit


def _page_spy(monkeypatch):
    """I2 page-budget spy on agents_core.notify.send_notification."""
    spy = MagicMock()
    monkeypatch.setattr("agents_core.notify.send_notification", spy)
    return spy


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path, monkeypatch):
    """Hermeticity: every test gets a fresh, isolated ledger + lock in tmp_path.

    The module's _LEDGER_FILE points at the REAL production ledger
    (/srv/lapis/lapis-state/deploy-pull-repair-ledger.json), which accumulates
    entries from prior runs - a run_pass test against /x would otherwise read
    stale open entries (bump/salvage instead of a clean deposited:new).
    Redirect to a tmp ledger so each test is deterministic and never touches
    production state."""
    monkeypatch.setattr(dps, "_LEDGER_FILE", tmp_path / "ledger.json")
    monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
    monkeypatch.setattr(dps, "_LEDGER_LOCK_TIMEOUT_S", 0.2)


# ---------------------------------------------------------------------------
# B4b (lapis-pm-daemon-silent-gaps-v0) - locked-clone re-evaluation helpers
# ---------------------------------------------------------------------------

class TestB4bLockedCloneHelpers:
    """close_ledger_entries_for_path (ledger closure on clear) +
    notify_locked_clone_stuck (the still-stuck NORMAL page, deduped on the
    D3 ledger signature)."""

    def test_close_marks_open_entries_closed_not_deleted(self, monkeypatch):
        """After a B4b self-clear the D3 ledger entry for the path is
        `closed` (never deleted) - so a re-lock of the same path is a fresh
        state transition that pages instead of being dedup-suppressed."""
        from lapis_pm import deploy_pull_selfheal as _dps
        ledger = {
            "sig-open": {
                "signature": "sig-open", "path": "/srv/git/agents-core",
                "repo": "agents-core", "class": "diverged",
                "first_seen": "2026-09-02T16:04:00+00:00",
                "last_seen": "2026-09-02T16:04:00+00:00",
                "times_seen": 3, "status": "open",
            },
            "sig-resolved": {
                "signature": "sig-resolved", "path": "/srv/git/agents-core",
                "repo": "agents-core", "class": "safe_dead_branch",
                "status": "resolved",
            },
            "sig-other": {
                "signature": "sig-other", "path": "/srv/git/other",
                "repo": "other", "class": "diverged", "status": "open",
            },
        }
        monkeypatch.setattr(_dps, "read_ledger", lambda path=None: dict(ledger))
        written = []
        monkeypatch.setattr(_dps, "write_ledger",
                            lambda ledger, path=None: written.append(dict(ledger)))
        _dps.close_ledger_entries_for_path("/srv/git/agents-core",
                                           reason="b4b-self-clear")
        assert written, "ledger must be re-written"
        out = written[-1]
        assert "sig-open" in out, "entry must NOT be deleted"
        assert out["sig-open"]["status"] == "closed"
        assert out["sig-open"]["closed_reason"] == "b4b-self-clear"
        assert out["sig-open"].get("closed_at"), "closed_at must be set"
        # non-open and other-path entries are untouched
        assert out["sig-resolved"]["status"] == "resolved"
        assert out["sig-other"]["status"] == "open"

    def test_close_noop_when_no_open_entries(self, monkeypatch):
        from lapis_pm import deploy_pull_selfheal as _dps
        monkeypatch.setattr(_dps, "read_ledger", lambda path=None: {})
        written = []
        monkeypatch.setattr(_dps, "write_ledger",
                            lambda ledger, path=None: written.append(ledger))
        _dps.close_ledger_entries_for_path("/srv/git/agents-core")
        assert written == [], "no write when nothing to close"

    def test_still_stuck_pages_once_then_silent(self, monkeypatch):
        """A locked clone in `diverged` state: exactly one NORMAL page on
        first detection; a second backstop cycle in the same state emits no
        additional page (the 2026-09-02 370-page storm must not recur)."""
        from agents_core.notify import Priority
        pages = []

        def fake_send(message, title, priority, **kwargs):
            pages.append((title, priority))
            return True

        monkeypatch.setattr("agents_core.notify.send_notification", fake_send)
        monkeypatch.setattr(dps, "_send_page",
                            lambda message, title, priority: pages.append((title, priority)))

        dps.notify_locked_clone_stuck("agents-core", "/srv/git/agents-core",
                                      dps.CLASS_DIVERGED, "backstop-timer")
        assert len(pages) == 1
        assert pages[0][1] == Priority.NORMAL

        # Second cycle, same (path, class): silent bump, no page.
        dps.notify_locked_clone_stuck("agents-core", "/srv/git/agents-core",
                                      dps.CLASS_DIVERGED, "backstop-timer")
        assert len(pages) == 1

        ledger = dps.read_ledger()
        sig = dps.compute_signature("/srv/git/agents-core", dps.CLASS_DIVERGED)
        assert sig in ledger
        assert ledger[sig]["status"] == "open"
        assert ledger[sig]["times_seen"] == 2

    def test_class_transition_pages_again(self, monkeypatch):
        """A class change for the same path is a fresh state transition:
        the new (path, class) signature pages once."""
        pages = []
        monkeypatch.setattr(dps, "_send_page",
                            lambda message, title, priority: pages.append(title))
        dps.notify_locked_clone_stuck("agents-core", "/srv/git/agents-core",
                                      dps.CLASS_DIVERGED, "backstop-timer")
        dps.notify_locked_clone_stuck("agents-core", "/srv/git/agents-core",
                                      dps.CLASS_FETCH_FAILED, "backstop-timer")
        assert len(pages) == 2

    def test_relock_after_closed_entry_pages_fresh(self, monkeypatch):
        """DoD 9: after a B4b self-clear (entry `closed`), a re-lock of the
        same path emits a fresh page (the re-lock is a new state
        transition, not suppressed by dedup)."""
        pages = []
        monkeypatch.setattr(dps, "_send_page",
                            lambda message, title, priority: pages.append(title))
        dps.notify_locked_clone_stuck("agents-core", "/srv/git/agents-core",
                                      dps.CLASS_DIVERGED, "backstop-timer")
        assert len(pages) == 1
        dps.close_ledger_entries_for_path("/srv/git/agents-core",
                                          reason="b4b-self-clear")
        # Re-lock: the closed entry is re-opened as a fresh transition.
        dps.notify_locked_clone_stuck("agents-core", "/srv/git/agents-core",
                                      dps.CLASS_DIVERGED, "backstop-timer")
        assert len(pages) == 2
        ledger = dps.read_ledger()
        sig = dps.compute_signature("/srv/git/agents-core", dps.CLASS_DIVERGED)
        assert ledger[sig]["status"] == "open"


# ---------------------------------------------------------------------------
# D1 - classifier
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_fetch_failed_precedence(self, monkeypatch):
        monkeypatch.setattr(dps, "_git", lambda *a, **kw: _git_cp(rc=1,
                         stderr="fatal: unable to access 'http://tok@h/r'"))
        cls, ev = dps.classify_pull_failure("/x", "backstop-timer")
        assert cls == dps.CLASS_FETCH_FAILED
        assert ev["fetch_rc"] == 1
        assert "tok" not in ev["fetch_stderr"]

    def test_safe_dead_branch(self, monkeypatch):
        _patch_git(monkeypatch, branch="feature/x", ahead="0", behind="0")
        cls, ev = dps.classify_pull_failure("/x", "backstop-timer")
        assert cls == dps.CLASS_SAFE_DEAD_BRANCH
        assert ev["dead_branch"] is True

    def test_safe_staged_adds(self, monkeypatch):
        _patch_git(monkeypatch, porcelain="A  new.py\nA  other.py\n")
        cls, ev = dps.classify_pull_failure("/x", "backstop-timer")
        assert cls == dps.CLASS_SAFE_STAGED_ADDS
        assert ev["staged_adds"] is True

    def test_diverged(self, monkeypatch):
        _patch_git(monkeypatch, ahead="2", behind="0")
        cls, _ev = dps.classify_pull_failure("/x", "backstop-timer")
        assert cls == dps.CLASS_DIVERGED

    def test_active_dirty_within_window(self, monkeypatch):
        now = time.time()
        _patch_git(monkeypatch, porcelain=" M tracked.py\n",
                   fetch_stderr="")
        monkeypatch.setattr(dps, "_git", lambda *a, **kw: _git_cp())
        # build evidence by hand via _collect_evidence patching
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M tracked.py",),
                                             mtime=now - 60, now=now),
        )
        cls, _ev = dps.classify_pull_failure("/x", "backstop-timer", now=now)
        assert cls == dps.CLASS_ACTIVE_DIRTY

    def test_stale_dirty_outside_window(self, monkeypatch):
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M tracked.py",),
                                             mtime=now - 3600, now=now),
        )
        cls, _ev = dps.classify_pull_failure("/x", "backstop-timer", now=now)
        assert cls == dps.CLASS_STALE_DIRTY

    def test_unknown_clean_tree(self, monkeypatch):
        _patch_git(monkeypatch)
        cls, _ev = dps.classify_pull_failure("/x", "backstop-timer")
        assert cls == dps.CLASS_UNKNOWN

    def test_error_unknown_never_safe(self, monkeypatch):
        def boom(path, *args, **kw):
            raise RuntimeError("git exploded")
        monkeypatch.setattr(dps, "_git", boom)
        cls, ev = dps.classify_pull_failure("/x", "backstop-timer")
        assert cls == dps.CLASS_UNKNOWN
        assert ev["dead_branch"] is False
        assert ev["staged_adds"] is False
        assert "error" in ev

    def test_composition_dead_branch_plus_staged_adds(self, monkeypatch):
        """The exact 2026-09-02 shape: dead branch + staged adds labels
        safe_dead_branch (first match) with BOTH evidence flags set."""
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(
                branch="feature/x", ahead=0, behind=0,
                porcelain=("A  new.py",), now=now),
        )
        cls, ev = dps.classify_pull_failure("/x", "backstop-timer", now=now)
        assert cls == dps.CLASS_SAFE_DEAD_BRANCH
        assert ev["dead_branch"] is True
        assert ev["staged_adds"] is True

    def test_untracked_only_is_not_dirty(self, monkeypatch):
        _patch_git(monkeypatch, porcelain="?? untracked.txt\n")
        cls, _ev = dps.classify_pull_failure("/x", "backstop-timer")
        assert cls == dps.CLASS_UNKNOWN


# ---------------------------------------------------------------------------
# Redaction (Critical fix #6: the bare-token shape)
# ---------------------------------------------------------------------------

class TestRedactCredentials:
    def test_user_pass_shape(self):
        out = dps.redact_credentials(
            "fatal: unable to access 'http://user:pass@203.0.113.10:3000/r/'")
        assert "user:pass" not in out
        assert "http://***@" in out

    def test_bare_token_shape(self):
        """The mapped clones embed a BARE Forgejo token with no colon
        (verified /srv/git/lapis-pm/.git/config:7) - the mask must cover
        it."""
        out = dps.redact_credentials(
            "fatal: unable to access 'http://96b48ef01ad99e6672324f9fb992f598c223ae6a@203.0.113.10:3000/r/'")
        assert "96b48ef01ad99e6672324f9fb992f598c223ae6a" not in out
        assert "http://***@" in out

    def test_empty_and_plain(self):
        assert dps.redact_credentials("") == ""
        assert dps.redact_credentials("no url here") == "no url here"


# ---------------------------------------------------------------------------
# Window math (active-defer vs stale; no windowed salvage in Slice 1)
# ---------------------------------------------------------------------------

class TestWindowMath:
    def test_active_defer_within_1800s(self, monkeypatch):
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 1799, now=now),
        )
        cls, _ev = dps.classify_pull_failure("/x", "backstop-timer", now=now)
        assert cls == dps.CLASS_ACTIVE_DIRTY

    def test_stale_beyond_1800s(self, monkeypatch):
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 1801, now=now),
        )
        cls, _ev = dps.classify_pull_failure("/x", "backstop-timer", now=now)
        assert cls == dps.CLASS_STALE_DIRTY


# ---------------------------------------------------------------------------
# I2 - page budget (0 pages on the unsafe, self-repaired, and fetch_failed
# shapes)
# ---------------------------------------------------------------------------

class TestPageBudget:
    def test_stale_dirty_first_detection_no_page(self, monkeypatch, tmp_path):
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        _mock_desk(monkeypatch)
        pages = _page_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        assert "deposited:new" in actions
        assert pages.call_count == 0

    def test_fetch_failed_first_detection_no_page(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(fetch_rc=1,
                                             fetch_stderr="fatal: auth failed"),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        _mock_desk(monkeypatch)
        pages = _page_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert "deposited:new" in actions
        assert pages.call_count == 0
        # the Diagnosis carries the redacted fetch stderr
        payload = dps.deposit_gem.call_args.args[0]
        diag = [b for b in payload["context"] if b["label"] == "Diagnosis"][0]
        assert any("auth failed" in line for line in diag["lines"])

    def test_five_cycles_stuck_unsafe_exactly_zero_pages(self, monkeypatch, tmp_path):
        """Rebased contract (Critical fix #3): 5 consecutive cycles against a
        stuck unsafe tree - exactly 0 pages, the gem deposits ONCE, and the
        open entry is bump-only on cycles 2-5."""
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        deposit = _mock_desk(monkeypatch)
        pages = _page_spy(monkeypatch)
        all_actions = []
        for i in range(5):
            all_actions.extend(dps.run_pass("agents-core", "/x", "backstop-timer",
                                            now=now + i))
        assert deposit.call_count == 1
        assert pages.call_count == 0
        assert all_actions.count("bump:open") == 4

    def test_self_repaired_tree_zero_pages(self, monkeypatch, tmp_path):
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=("A  new.py",), now=now),
        )
        monkeypatch.setattr(dps, "self_repair", lambda path, ev, trigger: True)
        pages = _page_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        assert "self_repaired" in actions
        assert pages.call_count == 0


# ---------------------------------------------------------------------------
# I3 - active-defer (no action, no page, no ledger state)
# ---------------------------------------------------------------------------

class TestActiveDefer:
    def test_active_dirty_defer_no_ledger_state(self, monkeypatch, tmp_path):
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 60, now=now),
        )
        _mock_desk(monkeypatch)
        pages = _page_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        assert "defer:active_dirty" in actions
        assert pages.call_count == 0
        ledger = dps.read_ledger()
        assert not any(e.get("path") == "/x" for e in ledger.values())


# ---------------------------------------------------------------------------
# D3 - ledger (bootstrap, atomic write, quarantine, absent file, flock)
# ---------------------------------------------------------------------------

class TestLedger:
    def test_write_then_read_roundtrip(self, tmp_path):
        ledger_file = tmp_path / "ledger.json"
        ledger = {"abc123": {"gem_id": "g1", "status": "open"}}
        dps.write_ledger(ledger, path=ledger_file)
        assert dps.read_ledger(path=ledger_file) == ledger

    def test_write_creates_parent_dir(self, tmp_path):
        """The ledger parent dir must exist for the atomic write (the
        [Errno 2] class)."""
        ledger_file = tmp_path / "nested" / "dir" / "ledger.json"
        dps.write_ledger({"k": {"status": "open"}}, path=ledger_file)
        assert ledger_file.exists()

    def test_write_is_atomic_fsync_then_replace(self, tmp_path):
        """fsync before os.replace (I5): a mid-flight failure leaves the
        prior content intact."""
        ledger_file = tmp_path / "ledger.json"
        dps.write_ledger({"orig": {"status": "open"}}, path=ledger_file)
        prior = ledger_file.read_text()
        with patch("os.replace", side_effect=OSError("disk full")):
            dps.write_ledger({"new": {"status": "open"}}, path=ledger_file)
        assert ledger_file.read_text() == prior

    def test_corrupt_ledger_quarantined(self, tmp_path):
        ledger_file = tmp_path / "ledger.json"
        ledger_file.write_text("{not valid json")
        assert dps.read_ledger(path=ledger_file) == {}
        assert not ledger_file.exists()
        quarantined = [p for p in tmp_path.iterdir() if "corrupt" in p.name]
        assert len(quarantined) == 1

    def test_absent_ledger_reads_as_empty_silently(self, tmp_path, caplog):
        """Critical fix #2/#5: a MISSING file is the normal first-run state -
        it reads as empty WITHOUT a quarantine warning (the [Errno 2]
        read-path fix)."""
        import logging
        with caplog.at_level(logging.WARNING, logger="lapis_pm.deploy_pull_selfheal"):
            assert dps.read_ledger(path=tmp_path / "nope.json") == {}
        assert not any("quarantine" in r.message for r in caplog.records)
        assert not any("corrupt" in r.message for r in caplog.records)

    def test_flock_contention_two_passes_one_gem(self, monkeypatch, tmp_path):
        """Two contending passes deposit exactly one gem: the second pass
        hits the 5s lock timeout (fail-closed) and skips."""
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        deposit = _mock_desk(monkeypatch)
        pages = _page_spy(monkeypatch)
        from lapis_pm.file_lock import file_lock, FileLockTimeout
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        monkeypatch.setattr(dps, "_LEDGER_FILE", tmp_path / "ledger.json")
        monkeypatch.setattr(dps, "_LEDGER_LOCK_TIMEOUT_S", 0.2)

        # hold the lock in this process: open + flock it, then run the pass
        # which must time out on the real (unpatched) file_lock.
        import fcntl
        lock_fd = os.open(str(tmp_path / "ledger.lock"), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            actions = dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        finally:
            os.close(lock_fd)
        assert "skip:ledger-lock-contended" in actions
        assert deposit.call_count == 0
        assert pages.call_count == 0

    def test_signature_stable_over_path_class(self):
        a = dps.compute_signature("/srv/git/x", "stale_dirty")
        b = dps.compute_signature("/srv/git/x", "stale_dirty")
        c = dps.compute_signature("/srv/git/x", "diverged")
        assert a == b
        assert a != c
        assert len(a) == 16


# ---------------------------------------------------------------------------
# I10 - gem options are live (ack_watch / not_real consumption)
# ---------------------------------------------------------------------------

class TestGemOptions:
    def test_ack_watch_consumed(self, monkeypatch, tmp_path):
        sig = dps.compute_signature("/x", "stale_dirty")
        monkeypatch.setattr(dps, "_LEDGER_FILE", tmp_path / "ledger.json")
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        ledger = {sig: {"gem_id": "g1", "status": "open", "path": "/x",
                        "class": "stale_dirty", "times_seen": 1,
                        "first_seen": "2026-09-02T00:00:00+00:00"}}
        monkeypatch.setattr(dps, "read_ledger", lambda path=None: ledger)
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        # the run must classify stale_dirty so the seeded signature is found;
        # patch _collect_evidence (the real one on /x fails git -> fetch_failed
        # via the fetch short-circuit, a DIFFERENT signature)
        monkeypatch.setattr(dps, "_collect_evidence",
                            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                                             mtime=time.time() - 3600,
                                                             now=time.time()))
        _mock_desk(monkeypatch)
        # the decision mock must be set AFTER _mock_desk (which sets
        # _gem_decisions to {} by default - clobbering this would empty it)
        monkeypatch.setattr(dps, "_gem_decisions", lambda: {"g1": "ack_watch"})
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert ledger[sig]["status"] == "acked"
        # an acked entry is not re-deposited and not bumped
        assert "bump:open" not in actions
        assert "deposited:new" not in actions

    def test_not_real_consumed(self, monkeypatch, tmp_path):
        sig = dps.compute_signature("/x", "stale_dirty")
        monkeypatch.setattr(dps, "_LEDGER_FILE", tmp_path / "ledger.json")
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        ledger = {sig: {"gem_id": "g1", "status": "open", "path": "/x",
                        "class": "stale_dirty", "times_seen": 1,
                        "first_seen": "2026-09-02T00:00:00+00:00"}}
        monkeypatch.setattr(dps, "read_ledger", lambda path=None: ledger)
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        monkeypatch.setattr(dps, "_collect_evidence",
                            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                                             mtime=time.time() - 3600,
                                                             now=time.time()))
        _mock_desk(monkeypatch)
        monkeypatch.setattr(dps, "_gem_decisions", lambda: {"g1": "not_real"})
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert ledger[sig]["status"] == "dismissed_fp"

    def test_gem_payload_shape(self, monkeypatch):
        """Pure unit assertion on the payload structure passed to the mocked
        deposit_gem (the rev-2 council-Q1 clarification: no live mock server,
        no :8403 connection - the mock records the call, the test asserts
        the shape)."""
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        deposit = _mock_desk(monkeypatch)
        dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        payload = deposit.call_args.args[0]
        assert payload["title"] == "deploy pull stuck: agents-core x"
        assert payload["state"] == "needs"
        assert payload["deposited_by"] == "deploy-pull-selfheal-core-v0"
        keys = [o["key"] for o in payload["options"]]
        assert keys == ["ack_watch", "not_real"]
        assert "salvage_now" not in keys
        labels = [b["label"] for b in payload["context"]]
        assert "Symptom" in labels and "Diagnosis" in labels
        suggested = [b for b in payload["context"] if b["label"] == "Suggested direction"][0]
        assert any("ack_watch" in line for line in suggested["lines"])


# ---------------------------------------------------------------------------
# I8 - lane constants cite the mem keys (Critical fix #3)
# ---------------------------------------------------------------------------

class TestLaneConstants:
    def test_lane_constants_cite_mem_keys(self):
        assert dps.MACHINE_LANE_ACTIONS.__doc__ is not None
        assert "productive-autonomy-held-node-2026-07-27" in dps.MACHINE_LANE_ACTIONS.__doc__
        assert "Erah-invisible-affordances-2026-06-08" in dps.MACHINE_LANE_ACTIONS.__doc__
        assert dps.ERAH_GATE_CLASSES.__doc__ is not None
        assert "productive-autonomy-held-node-2026-07-27" in dps.ERAH_GATE_CLASSES.__doc__
        assert "stalled-target-triage-to-agent-not-Erah-2026-06-24" in dps.ERAH_GATE_CLASSES.__doc__
        assert "Erah-pushover-signal-policy-llm-first-responder-2026-09-02" in dps.ERAH_GATE_CLASSES.__doc__


# ---------------------------------------------------------------------------
# Critical fix #7 - recurred flag distinctness
# ---------------------------------------------------------------------------

class TestRecurredFlag:
    def test_first_detection_labels_new(self, monkeypatch, tmp_path):
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        _mock_desk(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        assert "deposited:new" in actions

    def test_resolved_recurrence_reopens_in_place(self, monkeypatch, tmp_path):
        """A resolved entry that recurs re-opens in place: status -> open,
        first_seen/times_seen reset, a fresh gem deposited, and the action
        is `deposited:recurred` (distinct from `deposited:new`)."""
        now = time.time()
        sig = dps.compute_signature("/x", "stale_dirty")
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        deposit = _mock_desk(monkeypatch)
        first = dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        assert "deposited:new" in first
        ledger = dps.read_ledger()
        # resolve the entry (e.g. the tree healed and D4.5 closed it)
        ledger[sig]["status"] = "resolved"
        dps.write_ledger(ledger)
        second = dps.run_pass("agents-core", "/x", "backstop-timer", now=now + 600)
        assert "deposited:recurred" in second
        assert deposit.call_count == 2
        entry = dps.read_ledger()[sig]
        assert entry["status"] == "open"
        assert entry["times_seen"] == 1
        assert entry["gem_id"] == "gem-1"


# ---------------------------------------------------------------------------
# D4.5 - close-the-loop verify-healthy sweep
# ---------------------------------------------------------------------------

class TestCloseTheLoop:
    def test_healthy_tree_closes_open_entry(self, monkeypatch, tmp_path):
        now = time.time()
        sig = dps.compute_signature("/x", "stale_dirty")
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        _mock_desk(monkeypatch)
        first = dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        assert "deposited:new" in first
        # the tree heals: the next pass classifies unknown (clean 0/0) and
        # D4.5 closes the open entry in place.
        monkeypatch.setattr(dps, "_collect_evidence",
                            lambda path, now=None: _evidence())
        supersede = MagicMock()
        monkeypatch.setattr(dps, "_supersede_gem", supersede)
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {"branch": "main"}))
        second = dps.run_pass("agents-core", "/x", "backstop-timer", now=now + 600)
        assert any(a.startswith("resolved:healthy:") for a in second)
        entry = dps.read_ledger()[sig]
        assert entry["status"] == "resolved"
        supersede.assert_called_once()


# ---------------------------------------------------------------------------
# Critical fix #4 / DoD-8 - _finish_success wiring (NEW test)
# ---------------------------------------------------------------------------

class TestFinishSuccessWiring:
    def test_self_repaired_pull_fires_finish_success(self, monkeypatch, tmp_path):
        """A self-repaired post-land pull must fire _finish_success(pre_head):
        the HEAD-advance restart gate (any_advanced=True) + the canonical
        `synced old..new` deploy-log line. The 5 head reds do NOT cover
        pm_core wiring, so this test is new."""
        from lapis_pm import pm_core

        path = "/srv/git/agents-core-working"
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "presha111" if revparse_count[path] == 1 else "postsha222"
                return _git_cp(stdout=sha + "\n")
            if cmd[0] == "git" and "status" in cmd:
                return _git_cp(stdout="")
            if cmd[0] == "git" and "pull" in cmd:
                return _git_cp(rc=1, stderr="fatal: Not possible to fast-forward")
            return _git_cp(stdout="active")

        deploy_log = tmp_path / "deploy-log.md"
        with patch.object(pm_core, "_DEPLOY_LOG", deploy_log):
            with patch.object(pm_core, "_deploy_pull_locked", return_value=None):
                with patch.object(pm_core, "_commit_distance", return_value=0):
                    with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                        with patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure",
                                   return_value=True) as mock_pass:
                            advanced = pm_core._post_land_git_pull("agents-core",
                                                                   trigger="backstop-timer")
        mock_pass.assert_called()  # the wiring fires the selfheal pass
        assert advanced is True
        log_text = deploy_log.read_text()
        # SHAs are truncated to 8 chars in the deploy log (pm_core.py:1027):
        # presha111 -> presha11, postsha222 -> postsha2
        assert "synced presha11..postsha2" in log_text

    def test_unrepaired_recorded_failure_keeps_legacy_path(self, monkeypatch, tmp_path):
        """A ledger-recorded (non-self-repaired) failure returns False from
        pass_handled_failure: no _finish_success, no restart gate."""
        from lapis_pm import pm_core

        path = "/srv/git/agents-core-working"
        revparse_count = {}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "rev-parse" in cmd:
                revparse_count[path] = revparse_count.get(path, 0) + 1
                sha = "samehead1"  # HEAD did not advance
                return _git_cp(stdout=sha + "\n")
            if cmd[0] == "git" and "status" in cmd:
                return _git_cp(stdout="")
            if cmd[0] == "git" and "pull" in cmd:
                return _git_cp(rc=1, stderr="fatal: Not possible to fast-forward")
            return _git_cp(stdout="active")

        deploy_log = tmp_path / "deploy-log.md"
        with patch.object(pm_core, "_DEPLOY_LOG", deploy_log):
            with patch.object(pm_core, "_deploy_pull_locked", return_value=None):
                with patch.object(pm_core, "_commit_distance", return_value=0):
                    with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                        with patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure",
                                   return_value=False):
                            advanced = pm_core._post_land_git_pull("agents-core",
                                                                   trigger="backstop-timer")
        assert advanced is False
        assert not deploy_log.exists() or "synced" not in deploy_log.read_text()
