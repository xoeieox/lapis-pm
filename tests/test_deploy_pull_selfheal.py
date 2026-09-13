"""Hermetic tests for lapis_pm.deploy_pull_selfheal (Slice 2,
deploy-pull-selfheal-slice2-v0, grafted onto the Slice 1 base
deploy-pull-selfheal-core-v0).

The Slice-1 36-test suite (2 re-pinned in place per Scope item 6 / C5 /
C6, the rest unchanged) plus the re-added D4/D5/D6/D7 tests (the 10 named
removals in the core spec's fix #5) plus the panel-added shapes 10-12
plus the OQ-4 wiring test (shape 13). No live weaver/Desk dependency:
every test that drives a first-detection `run_pass` monkeypatches
`deposit_gem` / `emit_provenance` (the hermeticity requirement), and the
page budget is pinned with an `agents_core.notify.send_notification` spy
(I2: at most one page per state transition).

Coverage: D1 classifier (every row, composition, error-unknown,
fetch-failed precedence, redaction incl. the bare-token shape); window
math (active-defer vs stale); the I2 page-budget spy (re-pinned: 5 cycles
stuck unsafe -> exactly 1 salvaged page + 0 re-pages; C5 held-states
pins: 5 cycles stuck acked / clean-unknown -> 0 pages); I3 active-defer;
ledger bootstrap + atomic write (fsync) + corrupt quarantine + absent-file
+ flock contention + Slice-1-shaped entry read-back (M1); I10
ack_watch/not_real/salvage_now consumption; the I8 lane-constant
docstrings + `.VALUE` membership (L2); the `recurred`-flag distinctness
(Critical fix #7); the `_finish_success` wiring; D4 salvage losslessness
(stale-dirty + diverged on scratch clones), no-double-PR, dirty+diverged
churn (M2), pre-existing ref collision (C9/M4); D5 worker_failed hold +
one page + no re-page + never auto-closed + error-safe signals (M3); D6
fetch_failed 20-min page + silence + C6 recurrence re-page; D7 escalate
with the corrected import (C1); the OQ-4 option-3 success-path seam
wiring (shape 13, blocking).
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
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
        """C6 re-pin: first detection records the gem + station incident and
        pages NOTHING; the 20-minute gate pages once, then silence."""
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
        # the 20-minute gate: one HIGH page, then silence (state-transition
        # rule, I2).
        ledger = dps.read_ledger()
        for e in ledger.values():
            e["first_seen"] = (datetime.now(timezone.utc)
                               - timedelta(minutes=21)).isoformat(
                                   timespec="seconds")
        dps.write_ledger(ledger)
        pages.reset_mock()
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert "paged:fetch_failed" in actions
        assert pages.call_count == 1
        pages.reset_mock()
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert "paged:fetch_failed" not in actions
        assert pages.call_count == 0

    def test_five_cycles_stuck_unsafe_exactly_one_page(self, monkeypatch, tmp_path):
        """Re-pinned contract (I2, Slice 2): 5 consecutive cycles against a
        stuck unsafe tree PAST the 20-minute window - exactly 1 page
        (salvaged) + 0 re-pages; the gem deposits ONCE; the entry
        transitions to `salvaged` and the later cycles hold."""
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        monkeypatch.setattr(dps, "_ensure_target",
                            lambda repo, sig: "deploy-repair-agents-core-abc123")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: False)
        monkeypatch.setattr(dps, "_salvage", lambda *a, **kw: (True, {
            "salvage_branch": "lapis/deploy-repair-agents-core-abc123/salvage",
            "temp_commit": False, "target_id": "deploy-repair-agents-core-abc123",
        }))
        monkeypatch.setattr(dps, "_push_and_open_pr", lambda *a, **kw: 42)
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **kw: None)
        deposit = _mock_desk(monkeypatch)
        pages = _page_spy(monkeypatch)
        all_actions = []
        # cycle 1: first detection (gem, no page; the window is not yet
        # elapsed - fresh first_seen).
        all_actions.extend(dps.run_pass("agents-core", "/x", "backstop-timer",
                                        now=now))
        # age the entry past the 20-minute window (simulate 5 backstop
        # cycles of 10 min each).
        ledger = dps.read_ledger()
        for e in ledger.values():
            e["first_seen"] = (datetime.now(timezone.utc)
                               - timedelta(minutes=51)).isoformat(
                                   timespec="seconds")
        dps.write_ledger(ledger)
        # cycles 2-5: exactly one salvaged page total, then silence.
        for i in range(4):
            all_actions.extend(dps.run_pass("agents-core", "/x", "backstop-timer",
                                            now=now + i))
        assert deposit.call_count == 1
        # exactly one page, command-shaped (the only routine page, I2)
        assert pages.call_count == 1
        assert "salvaged" in pages.call_args.kwargs.get("message", "").lower()
        assert "Ratify to land, close to discard." in pages.call_args.kwargs["message"]
        entry = dps.read_ledger()[dps.compute_signature("/x", "stale_dirty")]
        assert entry["status"] == "salvaged"
        assert entry["pr_number"] == 42

    def test_five_cycles_stuck_acked_zero_pages(self, monkeypatch, tmp_path):
        """C5 SECURITY pin: 5 cycles against a stuck acked tree - 0 pages
        from this module (the pass returns True for `hold:acked`; the Slice-1
        True-set gap let pm_core's legacy generic notify - unredacted
        stderr[:300] incl. the tokenized remote URL - fire every cycle)."""
        now = time.time()
        sig = dps.compute_signature("/x", "stale_dirty")
        ledger = {sig: {"gem_id": "g1", "status": "open", "path": "/x",
                        "class": "stale_dirty", "times_seen": 1,
                        "first_seen": (datetime.now(timezone.utc)
                                       - timedelta(minutes=21)).isoformat(
                                           timespec="seconds")}}
        monkeypatch.setattr(dps, "read_ledger", lambda path=None: dict(ledger))
        written = []
        monkeypatch.setattr(dps, "write_ledger",
                            lambda l, path=None: written.append(dict(l)))
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        _mock_desk(monkeypatch)
        monkeypatch.setattr(dps, "_gem_decisions", lambda: {"g1": "ack_watch"})
        pages = _page_spy(monkeypatch)
        for i in range(5):
            actions = dps.run_pass("agents-core", "/x", "backstop-timer", now=now + i)
            assert "hold:acked" in actions
            assert dps.pass_handled_failure(
                "agents-core", "/x", "backstop-timer",
                pull_rc=1, pull_stderr="fatal: auth failed") is True
        assert pages.call_count == 0

    def test_five_cycles_stuck_clean_unknown_zero_pages(self, monkeypatch, tmp_path):
        """C5 SECURITY pin: 5 cycles against a stuck clean-unknown tree - 0
        pages from this module (`clean:unknown` is in the True set; the pass
        owns the state)."""
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(),  # clean 0/0 -> unknown, no anomaly
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        _mock_desk(monkeypatch)
        pages = _page_spy(monkeypatch)
        for i in range(5):
            actions = dps.run_pass("agents-core", "/x", "backstop-timer", now=now + i)
            assert "clean:unknown" in actions
            assert dps.pass_handled_failure(
                "agents-core", "/x", "backstop-timer",
                pull_rc=1, pull_stderr="fatal: auth failed") is True
        assert pages.call_count == 0

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
        the shape). Re-pinned in place for Slice 2 (OQ-1): the identity
        stays `deploy-pull-selfheal-core-v0` (same machinery, more actions)
        and the `salvage_now` gem option is live (I10 - no dead button)."""
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
        assert keys == ["salvage_now", "ack_watch", "not_real"]
        salvage_now = [o for o in payload["options"] if o["key"] == "salvage_now"][0]
        assert salvage_now.get("primary") is True
        labels = [b["label"] for b in payload["context"]]
        assert "Symptom" in labels and "Diagnosis" in labels
        suggested = [b for b in payload["context"] if b["label"] == "Suggested direction"][0]
        assert any("salvage_now" in line for line in suggested["lines"])


# ---------------------------------------------------------------------------
# I8 - lane constants cite the mem keys (Critical fix #3)
# ---------------------------------------------------------------------------

class TestLaneConstants:
    def test_lane_constants_cite_mem_keys(self):
        """L2 re-pin: main's class shape preserved (the `.__doc__` the
        assertion reads IS the class docstring); the Slice-2 verbs are in
        `.VALUE`."""
        assert dps.MACHINE_LANE_ACTIONS.__doc__ is not None
        assert "productive-autonomy-held-node-2026-07-27" in dps.MACHINE_LANE_ACTIONS.__doc__
        assert "Erah-invisible-affordances-2026-06-08" in dps.MACHINE_LANE_ACTIONS.__doc__
        assert dps.ERAH_GATE_CLASSES.__doc__ is not None
        assert "productive-autonomy-held-node-2026-07-27" in dps.ERAH_GATE_CLASSES.__doc__
        assert "stalled-target-triage-to-agent-not-Erah-2026-06-24" in dps.ERAH_GATE_CLASSES.__doc__
        assert "Erah-pushover-signal-policy-llm-first-responder-2026-09-02" in dps.ERAH_GATE_CLASSES.__doc__
        # Slice-2 `.VALUE` membership (class shape preserved - L2)
        assert "salvage_and_restore" in dps.MACHINE_LANE_ACTIONS.VALUE
        assert "classify_pull_failure" in dps.MACHINE_LANE_ACTIONS.VALUE
        assert "close_the_loop" in dps.MACHINE_LANE_ACTIONS.VALUE
        assert "salvage_pr_land_or_discard" in dps.ERAH_GATE_CLASSES.VALUE
        assert "worker_failed_hold" in dps.ERAH_GATE_CLASSES.VALUE
        assert "fetch_failed" in dps.ERAH_GATE_CLASSES.VALUE


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


# ---------------------------------------------------------------------------
# Slice 2 helpers (D4/D5/D6/D7 - scratch-clone salvage machinery, ported
# from the PR #314 head test matrix and grafted onto this suite)
# ---------------------------------------------------------------------------

def _git(path, *args, check=True):
    r = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True, text=True, timeout=30,
    )
    if check and r.returncode != 0:
        raise AssertionError(f"git {args} failed: {r.stderr}")
    return r


def _make_origin(tmp_path: Path, files: dict | None = None) -> Path:
    """A bare origin repo with main + one commit."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    subprocess.run(["git", "init", "--bare", str(origin)], check=True,
                   capture_output=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init")
    _git(seed, "config", "user.email", "t@t")
    _git(seed, "config", "user.name", "t")
    for name, content in (files or {"a.txt": "one\n"}).items():
        (seed / name).write_text(content)
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "branch", "-M", "main")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "origin", "main")
    _git(origin, "symbolic-ref", "HEAD", "refs/heads/main")
    return origin


def _clone(origin: Path, tmp_path: Path, name: str = "clone") -> Path:
    clone = tmp_path / name
    subprocess.run(["git", "clone", str(origin), str(clone)], check=True,
                   capture_output=True)
    _git(clone, "config", "user.email", "t@t")
    _git(clone, "config", "user.name", "t")
    return clone


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _entry(path: str, repo: str, cls: str, *, status: str = "open",
           first_seen: str | None = None, gem_id: str | None = "gem-1",
           times_seen: int = 1) -> dict:
    now = _iso(datetime.now(timezone.utc))
    return {
        "signature": dps.compute_signature(path, cls),
        "path": path, "repo": repo, "class": cls,
        "first_seen": first_seen or now,
        "last_seen": now, "times_seen": times_seen, "status": status,
        "gem_id": gem_id, "pr_number": None, "salvage_branch": None,
        "dispatch_error": None, "attempts": 0, "oldest_dirty_mtime": None,
    }


def _notify_spy(monkeypatch):
    """I2 page-budget spy on agents_core.notify.send_notification."""
    pages: list = []
    monkeypatch.setattr(
        sys.modules["agents_core.notify"], "send_notification",
        lambda **kw: pages.append(kw),
    )
    return pages


# ---------------------------------------------------------------------------
# Shape 1 - window math (active-within-20min vs stale-beyond; salvage_now
# bypass)
# ---------------------------------------------------------------------------

class TestWindowMathSlice2:
    def test_window_not_before_20min(self):
        e = _entry("/x", "agents-core", dps.CLASS_STALE_DIRTY,
                   first_seen=_iso(datetime.now(timezone.utc) - timedelta(minutes=10)))
        assert dps._window_elapsed(e) is False
        e["first_seen"] = _iso(datetime.now(timezone.utc) - timedelta(minutes=20, seconds=1))
        assert dps._window_elapsed(e) is True
        e["first_seen"] = _iso(datetime.now(timezone.utc) - timedelta(minutes=61))
        assert dps._window_elapsed(e) is True

    def test_salvage_now_bypasses_window(self):
        e = _entry("/x", "agents-core", dps.CLASS_STALE_DIRTY,
                   first_seen=_iso(datetime.now(timezone.utc) - timedelta(minutes=1)),
                   gem_id="g1")
        assert dps._window_elapsed(e) is False
        assert dps._salvage_now_requested(e, {"g1": "salvage_now"}) is True
        assert dps._salvage_now_requested(e, {"g1": "ack_watch"}) is False


# ---------------------------------------------------------------------------
# Shape 2 - D4 salvage losslessness (stale-dirty + diverged, scratch clones)
# ---------------------------------------------------------------------------

class TestSalvageLosslessness:
    def test_salvage_stale_dirty_temp_clone(self, tmp_path, monkeypatch):
        """D4 on a scratch clone: the commit lands on the temp ref (main
        untouched mid-operation), the salvage ref carries the work, the tree
        ends main/0/0/clean, and every pre-op commit is still reachable."""
        origin = _make_origin(tmp_path)
        clone = _clone(origin, tmp_path)
        (clone / "work.txt").write_text("precious\n")
        _git(clone, "add", "-A")
        pre_commits = set(_git(clone, "rev-list", "HEAD").stdout.split())

        pages = _notify_spy(monkeypatch)
        monkeypatch.setattr(dps, "_ensure_target",
                            lambda repo, sig: "deploy-repair-agents-core-abc123")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: False)
        # the close-the-loop verify predicate is not the losslessness check -
        # mock it; assert the real end state (main/0/0/clean) + losslessness
        # via git below.
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        pr_numbers: list = []
        monkeypatch.setattr(
            dps, "_push_and_open_pr",
            lambda path, repo, entry, info: (pr_numbers.append(42) or 42),
        )

        sig = dps.compute_signature(str(clone), dps.CLASS_STALE_DIRTY)
        entry = _entry(str(clone), "agents-core", dps.CLASS_STALE_DIRTY)
        entry["signature"] = sig
        ledger = {sig: entry}

        actions = dps._salvage_entry(ledger, entry, "agents-core", str(clone),
                                     {}, "backstop-timer")
        assert any(a.startswith("salvaged:") for a in actions), actions

        # the salvage branch exists and carries the local work
        ref = "lapis/deploy-repair-agents-core-abc123/salvage"
        show = _git(clone, "show", f"{ref}:work.txt")
        assert show.stdout == "precious\n"
        # main is at 0/0 and clean
        assert _git(clone, "branch", "--show-current").stdout.strip() == "main"
        lr = _git(clone, "rev-list", "--left-right", "--count", "HEAD...origin/main")
        assert lr.stdout.split() == ["0", "0"]
        status = _git(clone, "status", "--porcelain").stdout
        assert not any(line and not line.startswith("??") for line in status.splitlines())
        # losslessness: every pre-op commit reachable post-op
        post_commits = set(_git(clone, "rev-list", "HEAD", ref).stdout.split())
        assert pre_commits.issubset(post_commits)
        # exactly one HIGH page, command-shaped (I2)
        assert len(pages) == 1
        assert pages[0]["priority"] == sys.modules["agents_core.notify"].Priority.HIGH
        assert "Ratify to land, close to discard." in pages[0]["message"]
        assert "PR #42" in pages[0]["message"]
        # ledger entry salvaged with pr_number + salvage_branch
        assert entry["status"] == "salvaged"
        assert entry["pr_number"] == 42
        assert entry["salvage_branch"] == ref

    def test_salvage_diverged_losslessness(self, tmp_path, monkeypatch):
        """Diverged shape: the local unique commit is reachable on the
        salvage ref post-operation (losslessness), tree at main/0/0."""
        origin = _make_origin(tmp_path)
        clone = _clone(origin, tmp_path)
        (clone / "local.txt").write_text("local\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "local unique")
        pre_commits = set(_git(clone, "rev-list", "HEAD").stdout.split())

        monkeypatch.setattr(dps, "_ensure_target",
                            lambda repo, sig: "deploy-repair-agents-core-def456")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: False)
        # the close-the-loop verify predicate (branch==main + 0/0 + clean) is
        # NOT the losslessness check - the diverged shape ends at main/clean
        # with the unique commit parked on the salvage ref (ahead>0 by
        # construction). Mock the verify; assert losslessness via rev-list.
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        monkeypatch.setattr(dps, "_push_and_open_pr", lambda p, r, e, i: 7)

        sig = dps.compute_signature(str(clone), dps.CLASS_DIVERGED)
        entry = _entry(str(clone), "agents-core", dps.CLASS_DIVERGED)
        entry["signature"] = sig
        actions = dps._salvage_entry({sig: entry}, entry, "agents-core", str(clone),
                                     {}, "backstop-timer")
        assert any(a.startswith("salvaged:") for a in actions), actions
        ref = "lapis/deploy-repair-agents-core-def456/salvage"
        post_commits = set(_git(clone, "rev-list", "HEAD", ref).stdout.split())
        assert pre_commits.issubset(post_commits)  # losslessness
        assert _git(clone, "branch", "--show-current").stdout.strip() == "main"

    def test_no_double_pr_when_open_pr_exists(self, tmp_path, monkeypatch):
        """No-op guard: an existing open PR on the salvage branch skips
        push/create (a mid-pass kill cannot double-open a PR)."""
        origin = _make_origin(tmp_path)
        clone = _clone(origin, tmp_path)
        monkeypatch.setattr(dps, "_ensure_target", lambda repo, sig: "tid-1")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: True)
        pushed: list = []
        monkeypatch.setattr(dps, "_push_and_open_pr",
                            lambda p, r, e, i: pushed.append(1) or 9)
        sig = dps.compute_signature(str(clone), dps.CLASS_STALE_DIRTY)
        entry = _entry(str(clone), "agents-core", dps.CLASS_STALE_DIRTY)
        entry["signature"] = sig
        actions = dps._salvage_entry({sig: entry}, entry, "agents-core", str(clone),
                                     {}, "backstop-timer")
        assert actions == ["salvage:pr-exists"]
        assert pushed == []
        assert entry["status"] == "open"  # unchanged


# ---------------------------------------------------------------------------
# Shape 10 (M2) - dirty+diverged churn: first attempt fails its own verify
# ---------------------------------------------------------------------------

class TestDirtyDivergedChurn:
    def test_dirty_diverged_first_attempt_fails_verify_no_loss_no_pr(
            self, tmp_path, monkeypatch):
        """A tree ahead AND dirty classifies `diverged` (the classifier
        checks ahead > 0 before the dirty check). The diverged salvage path
        never commits dirty files (the temp commit is stale_dirty-only), so
        the first attempt fails its own `_verify_healthy`: no work lost, no
        PR opened, the entry stays open, and the next cycle re-classifies it
        `stale_dirty` and completes. Pin the churn shape; no double-PR."""
        origin = _make_origin(tmp_path)
        clone = _clone(origin, tmp_path)
        (clone / "local.txt").write_text("local\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "local unique")
        # a MODIFIED tracked file (not a staged add - the classifier checks
        # staged_adds before ahead>0): ahead AND dirty.
        (clone / "a.txt").write_text("modified\n")
        old = time.time() - 3600
        os.utime(clone / "a.txt", (old, old))

        # the named M2 behavior: a dirty+diverged tree fails its own
        # `_verify_healthy` on the first attempt (the diverged path never
        # commits dirty files - the temp commit is stale_dirty-only) - no
        # work lost, no PR opened, the entry stays open. The REAL
        # `_verify_healthy` (unmocked) is what fails here: after the
        # salvage parks the unique commit on the salvage ref, main is
        # still ahead (1/0) and the dirty file is still dirty - the
        # close-the-loop predicate rejects it.
        pages = _notify_spy(monkeypatch)
        monkeypatch.setattr(dps, "_ensure_target",
                            lambda repo, sig: "deploy-repair-agents-core-m211")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: False)
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        pr_numbers: list = []
        monkeypatch.setattr(
            dps, "_push_and_open_pr",
            lambda p, r, e, i: (pr_numbers.append(77) or 77),
        )

        # cycle 1: classifies diverged (ahead > 0 checked before dirty)
        cls1, _ev1 = dps.classify_pull_failure(str(clone), "backstop-timer")
        assert cls1 == dps.CLASS_DIVERGED

        sig = dps.compute_signature(str(clone), dps.CLASS_DIVERGED)
        entry = _entry(str(clone), "agents-core", dps.CLASS_DIVERGED)
        entry["signature"] = sig
        actions = dps._salvage_entry({sig: entry}, entry, "agents-core",
                                     str(clone), _ev1, "backstop-timer")
        assert "salvaged:" not in " ".join(actions), actions
        assert "salvage:failed" in actions, actions
        # no work lost: the dirty file and the unique commit are intact
        assert (clone / "a.txt").read_text() == "modified\n"
        assert "local unique" in _git(clone, "log", "-1", "--format=%s",
                                      "HEAD").stdout
        # no PR opened, entry stays open, no page
        assert pr_numbers == []
        assert pages == []
        assert entry["status"] == "open"
        assert entry["dispatch_error"] == "salvage-failed"

        # cycle 2: the tree re-classifies stale_dirty (the unique commit is
        # committed by a human onto origin/main, leaving a clean-main +
        # stale-dirty shape) and the next salvage attempt completes.
        # Simulate: the unique commit is now on origin/main (a human push),
        # leaving a clean-main + stale-dirty shape (the dirty file is still
        # dirty). The REAL `_verify_healthy` passes once the salvage
        # commits the dirty file to the salvage ref and restores main to
        # 0/0/clean. The salvage ref from cycle 1 (carrying the parked
        # unique commit) is deleted: the human push put that commit on
        # origin/main, so the ref carries no unique commits and the C9
        # converge path would force-update it - but a fresh salvage ref
        # is the clean shape for cycle 2 (the real end state after a
        # merged salvage PR deletes the remote side; the local side is
        # the residue this test is NOT pinning here).
        _git(clone, "switch", "main")
        _git(clone, "reset", "--hard", "origin/main")
        _git(clone, "branch", "-D", "lapis/deploy-repair-agents-core-m211/salvage")
        (clone / "a.txt").write_text("modified2\n")
        old = time.time() - 3600
        os.utime(clone / "a.txt", (old, old))
        cls2, _ev2 = dps.classify_pull_failure(str(clone), "backstop-timer")
        assert cls2 == dps.CLASS_STALE_DIRTY
        sig2 = dps.compute_signature(str(clone), dps.CLASS_STALE_DIRTY)
        entry2 = _entry(str(clone), "agents-core", dps.CLASS_STALE_DIRTY)
        entry2["signature"] = sig2
        actions2 = dps._salvage_entry({sig2: entry2}, entry2, "agents-core",
                                      str(clone), _ev2, "backstop-timer")
        assert any(a.startswith("salvaged:") for a in actions2), actions2
        # exactly one PR total across both cycles (no double-PR)
        assert pr_numbers == [77]
        assert len(pages) == 1


# ---------------------------------------------------------------------------
# Shape 11 (C9/M4) - pre-existing ref collision
# ---------------------------------------------------------------------------

class TestPreOpRefHygiene:
    def test_stale_tmp_ref_converges(self, tmp_path, monkeypatch):
        """A stale `deploy-pull-salvage-tmp` from a crashed prior op: the
        same-name rename is a no-op (rc=0) - the salvage converges, 0 pages
        from the ref itself (the single salvaged page is the named one)."""
        origin = _make_origin(tmp_path)
        clone = _clone(origin, tmp_path)
        # simulate a crashed prior op: the temp ref exists at main.
        _git(clone, "branch", dps._SALVAGE_TMP_REF, "main")
        (clone / "work.txt").write_text("precious\n")
        _git(clone, "add", "-A")

        pages = _notify_spy(monkeypatch)
        monkeypatch.setattr(dps, "_ensure_target",
                            lambda repo, sig: "deploy-repair-agents-core-c901")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: False)
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        monkeypatch.setattr(dps, "_push_and_open_pr", lambda p, r, e, i: 51)

        sig = dps.compute_signature(str(clone), dps.CLASS_STALE_DIRTY)
        entry = _entry(str(clone), "agents-core", dps.CLASS_STALE_DIRTY)
        entry["signature"] = sig
        actions = dps._salvage_entry({sig: entry}, entry, "agents-core",
                                     str(clone), {}, "backstop-timer")
        assert any(a.startswith("salvaged:") for a in actions), actions
        # the temp ref no longer exists (converged: deleted before the op)
        probe = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "--verify", "--quiet",
             f"refs/heads/{dps._SALVAGE_TMP_REF}"],
            capture_output=True, text=True, timeout=30)
        assert probe.returncode != 0
        # exactly the one named salvaged page
        assert len(pages) == 1

    def test_preexisting_local_salvage_ref_converges(self, tmp_path, monkeypatch):
        """A pre-existing local `lapis/<target_id>/salvage` from a resolved
        episode (the remote side is deleted on merge, the local side never
        is): force-updated when it carries no unique commits - converges."""
        origin = _make_origin(tmp_path)
        clone = _clone(origin, tmp_path)
        ref = "lapis/deploy-repair-agents-core-c902/salvage"
        _git(clone, "branch", ref, "main")  # no unique commits
        (clone / "work.txt").write_text("precious\n")
        _git(clone, "add", "-A")

        pages = _notify_spy(monkeypatch)
        monkeypatch.setattr(dps, "_ensure_target",
                            lambda repo, sig: "deploy-repair-agents-core-c902")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: False)
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        monkeypatch.setattr(dps, "_push_and_open_pr", lambda p, r, e, i: 52)

        sig = dps.compute_signature(str(clone), dps.CLASS_STALE_DIRTY)
        entry = _entry(str(clone), "agents-core", dps.CLASS_STALE_DIRTY)
        entry["signature"] = sig
        actions = dps._salvage_entry({sig: entry}, entry, "agents-core",
                                     str(clone), {}, "backstop-timer")
        assert any(a.startswith("salvaged:") for a in actions), actions
        # the ref was force-updated to carry the salvaged work
        assert _git(clone, "show", f"{ref}:work.txt").stdout == "precious\n"
        assert len(pages) == 1

    def test_preexisting_local_salvage_ref_with_unique_commits_aborts(
            self, tmp_path, monkeypatch):
        """A pre-existing local salvage ref carrying UNIQUE commits: the op
        aborts with a named ledger state (`salvage-ref-collision`) - never a
        silent per-cycle retry; 0 pages."""
        origin = _make_origin(tmp_path)
        clone = _clone(origin, tmp_path)
        ref = "lapis/deploy-repair-agents-core-c903/salvage"
        # a unique commit on the salvage ref (not reachable from HEAD):
        # the ref must be AHEAD of HEAD (rev-list <ref>..HEAD counts
        # commits reachable from the ref but not from HEAD - the spec's
        # "unique to it" direction).
        _git(clone, "switch", "-c", ref)
        (clone / "unique.txt").write_text("unique\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "unique on salvage ref")
        _git(clone, "switch", "main")
        (clone / "work.txt").write_text("precious\n")
        _git(clone, "add", "-A")

        pages = _notify_spy(monkeypatch)
        monkeypatch.setattr(dps, "_ensure_target",
                            lambda repo, sig: "deploy-repair-agents-core-c903")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: False)
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        pushed: list = []
        monkeypatch.setattr(dps, "_push_and_open_pr",
                            lambda p, r, e, i: pushed.append(1) or 53)

        sig = dps.compute_signature(str(clone), dps.CLASS_STALE_DIRTY)
        entry = _entry(str(clone), "agents-core", dps.CLASS_STALE_DIRTY)
        entry["signature"] = sig
        actions = dps._salvage_entry({sig: entry}, entry, "agents-core",
                                     str(clone), {}, "backstop-timer")
        assert "salvage:failed" in actions, actions
        assert entry["dispatch_error"] == "salvage-ref-collision"
        assert entry["status"] == "open"  # named state, not silent
        # the unique commit is untouched
        assert "unique on salvage ref" in _git(
            clone, "log", "-1", "--format=%s", ref).stdout
        # no push, no PR, no page
        assert pushed == []
        assert pages == []

    def test_diverged_path_salvage_ref_present_after_converge(
            self, tmp_path, monkeypatch):
        """C9 (reviewer fix, cycle-1 med-2): on the non-temp (diverged)
        path with a pre-existing local salvage ref, the converge must
        leave the salvage ref PRESENT carrying the unique commit - the
        delete-then-recreate sequence could leave it missing (a
        `branch <dst>` from a non-main checked-out branch fails rc=128),
        which would break the losslessness verify (the post-op rev-list
        would not include the salvage ref) and the PR push."""
        origin = _make_origin(tmp_path)
        clone = _clone(origin, tmp_path)
        # a unique local commit (the diverged shape).
        (clone / "local.txt").write_text("local\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "local unique")
        ref = "lapis/deploy-repair-agents-core-c904/salvage"
        # a pre-existing local salvage ref from a resolved episode at the
        # OLD HEAD (no unique commits relative to the new HEAD - the
        # converge direction).
        _git(clone, "branch", ref, "origin/main")

        pages = _notify_spy(monkeypatch)
        monkeypatch.setattr(dps, "_ensure_target",
                            lambda repo, sig: "deploy-repair-agents-core-c904")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: False)
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        pr_numbers: list = []
        monkeypatch.setattr(dps, "_push_and_open_pr",
                            lambda p, r, e, i: (pr_numbers.append(54) or 54))

        sig = dps.compute_signature(str(clone), dps.CLASS_DIVERGED)
        entry = _entry(str(clone), "agents-core", dps.CLASS_DIVERGED)
        entry["signature"] = sig
        actions = dps._salvage_entry({sig: entry}, entry, "agents-core",
                                     str(clone), {}, "backstop-timer")
        assert any(a.startswith("salvaged:") for a in actions), actions
        # the salvage ref is PRESENT post-op and carries the unique commit
        # (losslessness: the unique commit is reachable via the ref)
        probe = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "--verify", "--quiet",
             f"refs/heads/{ref}"],
            capture_output=True, text=True, timeout=30)
        assert probe.returncode == 0, "salvage ref missing post-op"
        assert "local unique" in _git(clone, "log", "-1", "--format=%s",
                                      ref).stdout
        assert _git(clone, "branch", "--show-current").stdout.strip() == "main"
        # the PR push ran (the ref was present to push)
        assert pr_numbers == [54]
        assert len(pages) == 1


# ---------------------------------------------------------------------------
# Shape 4 (D5) - worker_failed hold
# ---------------------------------------------------------------------------

class TestStallHold:
    def _salvaged_entry(self, tmp_path, monkeypatch) -> tuple:
        ledger_path = tmp_path / "ledger.json"
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        e = _entry("/x", "agents-core", dps.CLASS_STALE_DIRTY, status="salvaged",
                   gem_id="g1")
        e["pr_number"] = 12
        e["target_id"] = "deploy-repair-agents-core-abc123"
        dps.write_ledger({e["signature"]: e}, path=ledger_path)
        monkeypatch.setattr(dps, "_gem_decisions", lambda: {})
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        return ledger_path, e

    def test_closed_unmerged_pr_one_worker_failed_page(self, tmp_path, monkeypatch):
        ledger_path, e = self._salvaged_entry(tmp_path, monkeypatch)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "closed")
        monkeypatch.setattr(dps, "_rejection_count", lambda tid: 0)
        monkeypatch.setattr(dps, "_target_paused_review_gate", lambda tid: False)
        pages = _notify_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert "worker_failed" in actions
        assert len(pages) == 1  # exactly one HIGH page (I2)
        assert pages[0]["priority"] == sys.modules["agents_core.notify"].Priority.HIGH
        assert "holding for PM" in pages[0]["message"]
        # L4: the gem id rides in the page text
        assert "gem g1" in pages[0]["message"]
        data = json.loads(ledger_path.read_text())
        assert data[e["signature"]]["status"] == "worker_failed"

    def test_two_rejections_stall(self, tmp_path, monkeypatch):
        ledger_path, e = self._salvaged_entry(tmp_path, monkeypatch)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "open")
        monkeypatch.setattr(dps, "_rejection_count", lambda tid: 2)
        monkeypatch.setattr(dps, "_target_paused_review_gate", lambda tid: False)
        pages = _notify_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert "worker_failed" in actions
        assert len(pages) == 1

    def test_gate_cap_pause_stall(self, tmp_path, monkeypatch):
        ledger_path, e = self._salvaged_entry(tmp_path, monkeypatch)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "open")
        monkeypatch.setattr(dps, "_rejection_count", lambda tid: 0)
        monkeypatch.setattr(dps, "_target_paused_review_gate", lambda tid: True)
        pages = _notify_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert "worker_failed" in actions
        assert len(pages) == 1

    def test_worker_failed_not_repaged(self, tmp_path, monkeypatch):
        """I2: no per-cycle re-page - a second pass against a worker_failed
        entry pages zero times."""
        ledger_path, e = self._salvaged_entry(tmp_path, monkeypatch)
        data = json.loads(ledger_path.read_text())
        data[e["signature"]]["status"] = "worker_failed"
        dps.write_ledger(data, path=ledger_path)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "closed")
        pages = _notify_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert pages == []
        assert "worker_failed" not in actions

    def test_worker_failed_never_auto_closed(self, tmp_path, monkeypatch):
        """worker_failed is not auto-closed by the healthy verify."""
        ledger_path, e = self._salvaged_entry(tmp_path, monkeypatch)
        data = json.loads(ledger_path.read_text())
        data[e["signature"]]["status"] = "worker_failed"
        dps.write_ledger(data, path=ledger_path)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "merged")
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {}))
        dps.run_pass("agents-core", "/x", "backstop-timer")
        data = json.loads(ledger_path.read_text())
        assert data[e["signature"]]["status"] == "worker_failed"

    def test_error_safe_signals_no_false_worker_failed(self, tmp_path, monkeypatch):
        """M3: each of the three signal reads erroring -> not stalled (no
        false worker_failed)."""
        ledger_path, e = self._salvaged_entry(tmp_path, monkeypatch)

        def boom(*a, **kw):
            raise RuntimeError("mem down")
        # signal (a) _pr_state errors -> None (not "closed")
        monkeypatch.setattr(dps, "_pr_state", boom)
        monkeypatch.setattr(dps, "_rejection_count", boom)
        monkeypatch.setattr(dps, "_target_paused_review_gate", boom)
        pages = _notify_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert "worker_failed" not in actions
        assert pages == []
        data = json.loads(ledger_path.read_text())
        assert data[e["signature"]]["status"] == "salvaged"


# ---------------------------------------------------------------------------
# Shape 6 (D6) - fetch_failed 20-min page + C6 recurrence
# ---------------------------------------------------------------------------

class TestFetchFailedSlice2:
    def test_page_at_20min_then_silence(self, tmp_path, monkeypatch):
        ledger_path = tmp_path / "ledger.json"
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        old = _iso(datetime.now(timezone.utc) - timedelta(minutes=25))
        e = _entry("/x", "agents-core", dps.CLASS_FETCH_FAILED, first_seen=old)
        dps.write_ledger({e["signature"]: e}, path=ledger_path)
        monkeypatch.setattr(dps, "_gem_decisions", lambda: {})
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        pages = _notify_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert "paged:fetch_failed" in actions
        assert len(pages) == 1
        assert "credentials/network" in pages[0]["message"]
        # second pass: silent (state-transition rule)
        pages.clear()
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert pages == []
        assert "paged:fetch_failed" not in actions

    def test_no_page_before_20min(self, tmp_path, monkeypatch):
        ledger_path = tmp_path / "ledger.json"
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        e = _entry("/x", "agents-core", dps.CLASS_FETCH_FAILED,
                   first_seen=_iso(datetime.now(timezone.utc) - timedelta(minutes=5)))
        dps.write_ledger({e["signature"]: e}, path=ledger_path)
        monkeypatch.setattr(dps, "_gem_decisions", lambda: {})
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        pages = _notify_spy(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert pages == []
        assert "paged:fetch_failed" not in actions

    def test_recurrence_repages_at_20min(self, tmp_path, monkeypatch):
        """C6: a recurring fetch_failed (resolved -> re-failed) re-pages once
        at 20 min - the recurrence reset clears `fetch_page_sent`."""
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(fetch_rc=1,
                                             fetch_stderr="fatal: auth failed"),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        _mock_desk(monkeypatch)
        pages = _page_spy(monkeypatch)
        # episode 1: first detection, no page; age past the window; page once.
        dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        ledger = dps.read_ledger()
        for e in ledger.values():
            e["first_seen"] = _iso(datetime.now(timezone.utc) - timedelta(minutes=21))
        dps.write_ledger(ledger)
        dps.run_pass("agents-core", "/x", "backstop-timer", now=now + 1)
        assert pages.call_count == 1
        # the tree heals: D4.5 resolves the entry.
        monkeypatch.setattr(dps, "_collect_evidence",
                            lambda path, now=None: _evidence())
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {"branch": "main"}))
        monkeypatch.setattr(dps, "_supersede_gem", MagicMock())
        actions = dps.run_pass("agents-core", "/x", "backstop-timer", now=now + 2)
        assert any(a.startswith("resolved:healthy:") for a in actions)
        # episode 2 (recurrence): re-failed; the recurrence reset clears
        # fetch_page_sent; age past the window; re-pages once.
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(fetch_rc=1,
                                             fetch_stderr="fatal: auth failed"),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        pages.reset_mock()
        second = dps.run_pass("agents-core", "/x", "backstop-timer", now=now + 3)
        assert "deposited:recurred" in second
        entry = dps.read_ledger()[dps.compute_signature("/x", "fetch_failed")]
        assert entry["fetch_page_sent"] is False  # C6 reset
        ledger = dps.read_ledger()
        for e in ledger.values():
            e["first_seen"] = _iso(datetime.now(timezone.utc) - timedelta(minutes=21))
        dps.write_ledger(ledger)
        dps.run_pass("agents-core", "/x", "backstop-timer", now=now + 4)
        assert pages.call_count == 1  # re-paged once at 20 min


# ---------------------------------------------------------------------------
# Shape 7 (D7) - escalate with the corrected import
# ---------------------------------------------------------------------------

class TestStationEscalate:
    def test_escalate_corrected_import(self, monkeypatch):
        """C1: `station_escalate` writes the repair-station incident with the
        corrected import (`first` from agents_core.repair_station.types, not
        .escalate - the head import raised ImportError)."""
        import sys
        import agents_core.repair_station as rs
        # NOTE: `import agents_core.repair_station.escalate as m` binds the
        # FUNCTION (the package __init__ shadows the submodule in the
        # package namespace) - the real module is in sys.modules.
        rs_escalate_mod = sys.modules["agents_core.repair_station.escalate"]
        calls: list = []

        def fake_escalate(*a, **kw):
            calls.append((a, kw))
            return "incident-1"
        # the corrected import reads `escalate` from the module namespace -
        # patch it there (the package `__init__` re-exports it too).
        monkeypatch.setattr(rs_escalate_mod, "escalate", fake_escalate)
        monkeypatch.setattr(rs, "escalate", fake_escalate)
        # the real import path must resolve: `first` from types, `escalate`
        # from escalate (the corrected import - no ImportError).
        from agents_core.repair_station.escalate import escalate  # noqa: F401
        from agents_core.repair_station.types import Tier, first  # noqa: F401
        assert first().kind == "first"
        assert Tier.NORMAL.value == 2

        dps.station_escalate(repo="agents-core", path="/x", cls="stale_dirty",
                             first_seen="2026-09-02T00:00:00+00:00",
                             times_seen=1)
        assert len(calls) == 1
        a, kw = calls[0]
        assert kw["station_id"] == "deploy/pull-agents-core"  # station family
        assert kw["escalation_policy"].kind == "first"
        assert kw["tier"] == Tier.NORMAL
        assert kw["signature_fields"] == ["path", "class"]
        assert kw["error_signal"]["path"] == "/x"
        assert kw["error_signal"]["class"] == "stale_dirty"

    def test_first_detection_fires_station_escalate(self, monkeypatch, tmp_path):
        """The _first_detection call site fires station_escalate (the grafted
        D7 call after emit_provenance)."""
        now = time.time()
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=now - 3600, now=now),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        _mock_desk(monkeypatch)
        esc = MagicMock()
        monkeypatch.setattr(dps, "station_escalate", esc)
        dps.run_pass("agents-core", "/x", "backstop-timer", now=now)
        esc.assert_called_once()
        assert esc.call_args.kwargs["repo"] == "agents-core"
        assert esc.call_args.kwargs["cls"] == "stale_dirty"


# ---------------------------------------------------------------------------
# Shape 8 (M1) - ledger schema
# ---------------------------------------------------------------------------

class TestLedgerSchema:
    def test_slice1_shaped_entry_reads_back(self, monkeypatch, tmp_path):
        """M1: a Slice-1-shaped entry (identical schema - _new_entry
        unchanged) reads back without crashing; the re-added code uses
        `.get()` for every Slice-2 field."""
        sig = dps.compute_signature("/x", "stale_dirty")
        ledger = {sig: {
            "signature": sig, "path": "/x", "repo": "agents-core",
            "class": "stale_dirty", "first_seen": "2026-09-02T00:00:00+00:00",
            "last_seen": "2026-09-02T00:00:00+00:00", "times_seen": 1,
            "status": "open", "gem_id": "g1", "pr_number": None,
            "salvage_branch": None, "dispatch_error": None, "attempts": 0,
            "oldest_dirty_mtime": None,
        }}
        monkeypatch.setattr(dps, "read_ledger", lambda path=None: dict(ledger))
        monkeypatch.setattr(
            dps, "_collect_evidence",
            lambda path, now=None: _evidence(porcelain=(" M t.py",),
                                             mtime=time.time() - 3600,
                                             now=time.time()),
        )
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        _mock_desk(monkeypatch)
        actions = dps.run_pass("agents-core", "/x", "backstop-timer")
        assert "bump:open" in actions  # read back + bumped, no crash

    def test_salvaged_transition_writes_action_fields(self, tmp_path, monkeypatch):
        """After a salvaged transition, status == 'salvaged' with
        pr_number/salvage_branch non-null; worker_failed is a status value,
        not a field - no phantom fields in the durable store."""
        origin = _make_origin(tmp_path)
        clone = _clone(origin, tmp_path)
        (clone / "work.txt").write_text("precious\n")
        _git(clone, "add", "-A")
        pages = _notify_spy(monkeypatch)
        monkeypatch.setattr(dps, "_ensure_target",
                            lambda repo, sig: "deploy-repair-agents-core-m101")
        monkeypatch.setattr(dps, "_open_salvage_pr_exists", lambda repo, tid: False)
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (True, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        monkeypatch.setattr(dps, "_push_and_open_pr", lambda p, r, e, i: 88)

        sig = dps.compute_signature(str(clone), dps.CLASS_STALE_DIRTY)
        entry = _entry(str(clone), "agents-core", dps.CLASS_STALE_DIRTY)
        entry["signature"] = sig
        ledger = {sig: entry}
        actions = dps._salvage_entry(ledger, entry, "agents-core", str(clone),
                                     {}, "backstop-timer")
        assert any(a.startswith("salvaged:") for a in actions)
        assert entry["status"] == "salvaged"
        assert entry["pr_number"] == 88
        assert entry["salvage_branch"] == "lapis/deploy-repair-agents-core-m101/salvage"
        assert entry["target_id"] == "deploy-repair-agents-core-m101"
        # no phantom fields: worker_failed is a status value, not a field
        assert "worker_failed" not in entry
        assert "salvaged" not in entry  # status value, not a field

    def test_worker_failed_is_status_value(self, tmp_path, monkeypatch):
        ledger_path = tmp_path / "ledger.json"
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        e = _entry("/x", "agents-core", dps.CLASS_STALE_DIRTY, status="salvaged")
        e["pr_number"] = 12
        e["target_id"] = "deploy-repair-agents-core-abc123"
        dps.write_ledger({e["signature"]: e}, path=ledger_path)
        monkeypatch.setattr(dps, "_gem_decisions", lambda: {})
        monkeypatch.setattr(dps, "_verify_healthy", lambda path: (False, {}))
        monkeypatch.setattr(dps, "_deploy_log_line", lambda *a, **k: None)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "closed")
        monkeypatch.setattr(dps, "_rejection_count", lambda tid: 0)
        monkeypatch.setattr(dps, "_target_paused_review_gate", lambda tid: False)
        _notify_spy(monkeypatch)
        dps.run_pass("agents-core", "/x", "backstop-timer")
        data = json.loads(ledger_path.read_text())
        entry = data[e["signature"]]
        assert entry["status"] == "worker_failed"
        assert "worker_failed" not in entry  # status value, not a field


# ---------------------------------------------------------------------------
# Shape 13 (OQ-4, BLOCKING) - the success-path seam wiring, end-to-end
# ---------------------------------------------------------------------------

class TestOQ4Wiring:
    def test_success_pull_reaches_close_out_sweep(self, monkeypatch, tmp_path):
        """Exercises the production `_post_land_git_pull` success path
        END-TO-END with a mocked successful `git pull` - asserting
        `pass_handled_failure` is invoked with `pull_rc=0` (NOT by calling
        it directly - a direct call cannot capture the pm_core wiring
        change). The Close-Out Sweep runs: a `salvaged` ledger entry for
        that path with a merged PR transitions to `resolved`. Carries a
        wall-time assertion (<5s) + a path-scoped ledger read to protect
        the 120s budget."""
        from lapis_pm import pm_core

        path = "/srv/git/agents-core-working"
        ledger_path = tmp_path / "ledger.json"
        sig = dps.compute_signature(path, dps.CLASS_STALE_DIRTY)
        e = _entry(path, "agents-core", dps.CLASS_STALE_DIRTY, status="salvaged")
        e["pr_number"] = 12
        e["target_id"] = "deploy-repair-agents-core-abc123"
        dps.write_ledger({sig: e}, path=ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        monkeypatch.setattr(dps, "_LEDGER_LOCK_TIMEOUT_S", 0.2)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "merged")
        supersede = MagicMock()
        monkeypatch.setattr(dps, "_supersede_gem", supersede)
        # the sweep's close-the-loop reads must not hit the network
        # (bounded + idempotent: the PR-state read is the one Forgejo
        # call, mocked above).
        monkeypatch.setattr(dps, "_rejection_count", lambda tid: 0)
        monkeypatch.setattr(dps, "_target_paused_review_gate", lambda tid: False)

        # the production path: a successful pull for a critical repo. The
        # REAL pass_handled_failure runs (the wiring under test) - the
        # pm_core call site is what's being verified, not the machine.
        calls: list = []
        real_pass = dps.pass_handled_failure

        def spy_pass(repo, path_, trigger, *, pull_rc, pull_stderr=""):
            calls.append((repo, path_, pull_rc))
            return real_pass(repo, path_, trigger, pull_rc=pull_rc,
                             pull_stderr=pull_stderr)

        monkeypatch.setattr(dps, "pass_handled_failure", spy_pass)
        with patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure",
                   side_effect=spy_pass):
            # record the ledger read scope: the sweep must be path-scoped.
            reads: list = []
            real_read_ledger = dps.read_ledger

            def scoped_read_ledger(p=None):
                reads.append(p)
                return real_read_ledger(p)

            monkeypatch.setattr(dps, "read_ledger", scoped_read_ledger)

            # The fake models the REAL _finish_success rev-parse sequence:
            # `git -C <path> rev-parse HEAD` (the pre_head capture at pass
            # start and the post_head capture inside _finish_success)
            # returns the same HEAD (a no-advance pull), and the dirty
            # check (`git -C <path> status --porcelain`) is clean. A
            # rev-parse failure (rc != 0) would leave pre_head empty and
            # skip the success-path seam entirely (the low fix pins the
            # rev-parse-failure shape separately below).
            def fake_run(cmd, **kwargs):
                if cmd[0] == "git" and "rev-parse" in cmd:
                    return _git_cp(rc=0, stdout="samehead1\n")
                if cmd[0] == "git" and "status" in cmd:
                    return _git_cp(rc=0, stdout="")
                if cmd[0] == "git" and "pull" in cmd:
                    return _git_cp(rc=0, stdout="Already up to date.")
                return _git_cp(rc=0, stdout="active")

            t0 = time.time()
            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.object(pm_core, "_deploy_pull_locked",
                                  return_value=None):
                    advanced = pm_core._post_land_git_pull(
                        "agents-core", trigger="backstop-timer")
            elapsed = time.time() - t0
        # the seam is called on the SUCCESS path with the real pull_rc=0
        assert ("agents-core", path, 0) in calls, calls
        assert advanced is False  # HEAD did not advance
        # the sweep ran path-scoped (no ledger-wide scan) and transitioned
        # the salvaged entry to resolved (merged PR).
        data = json.loads(ledger_path.read_text())
        assert data[sig]["status"] == "resolved"
        supersede.assert_called_once()
        assert elapsed < 5.0

    def test_success_pull_close_out_stalled_pr_worker_failed(self, monkeypatch, tmp_path):
        """A `salvaged` entry with a stalled PR (two rejections) transitions
        `worker_failed` on the success-path sweep."""
        from lapis_pm import pm_core

        path = "/srv/git/agents-core-working"
        ledger_path = tmp_path / "ledger.json"
        sig = dps.compute_signature(path, dps.CLASS_STALE_DIRTY)
        e = _entry(path, "agents-core", dps.CLASS_STALE_DIRTY, status="salvaged")
        e["pr_number"] = 13
        e["target_id"] = "deploy-repair-agents-core-abc123"
        dps.write_ledger({sig: e}, path=ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        monkeypatch.setattr(dps, "_LEDGER_LOCK_TIMEOUT_S", 0.2)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "open")
        monkeypatch.setattr(dps, "_rejection_count", lambda tid: 2)
        monkeypatch.setattr(dps, "_target_paused_review_gate", lambda tid: False)

        calls: list = []
        real_pass = dps.pass_handled_failure

        def spy_pass(repo, path_, trigger, *, pull_rc, pull_stderr=""):
            calls.append((repo, path_, pull_rc))
            return real_pass(repo, path_, trigger, pull_rc=pull_rc,
                             pull_stderr=pull_stderr)

        monkeypatch.setattr(dps, "pass_handled_failure", spy_pass)
        with patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure",
                   side_effect=spy_pass):
            def fake_run(cmd, **kwargs):
                if cmd[0] == "git" and "rev-parse" in cmd:
                    return _git_cp(stdout="samehead1\n")
                if cmd[0] == "git" and "status" in cmd:
                    return _git_cp(stdout="")
                if cmd[0] == "git" and "pull" in cmd:
                    return _git_cp(rc=0, stdout="Already up to date.")
                return _git_cp(stdout="active")

            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.object(pm_core, "_deploy_pull_locked", return_value=None):
                    pm_core._post_land_git_pull("agents-core", trigger="backstop-timer")
        assert ("agents-core", path, 0) in calls
        data = json.loads(ledger_path.read_text())
        assert data[sig]["status"] == "worker_failed"

    def test_close_out_stall_transition_fires_one_high_page(self, tmp_path,
                                                            monkeypatch):
        """I2 alignment (reviewer fix, cycle-1 med-1): the close-out
        sweep's `salvaged` -> `worker_failed` transition fires the SAME
        one HIGH page the D5 block in _run_pass_locked fires (the sweep
        is the named D5 observer on the SUCCESS path - the failure-path
        D5 block only runs when a pull fails). Transition-gated: a second
        sweep over the worker_failed entry re-pages nothing."""
        path = "/srv/git/agents-core-working"
        ledger_path = tmp_path / "ledger.json"
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        monkeypatch.setattr(dps, "_LEDGER_LOCK_TIMEOUT_S", 0.2)
        sig = dps.compute_signature(path, dps.CLASS_STALE_DIRTY)
        e = _entry(path, "agents-core", dps.CLASS_STALE_DIRTY, status="salvaged",
                   gem_id="g1")
        e["pr_number"] = 13
        e["target_id"] = "deploy-repair-agents-core-abc123"
        dps.write_ledger({sig: e}, path=ledger_path)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "open")
        monkeypatch.setattr(dps, "_rejection_count", lambda tid: 2)
        monkeypatch.setattr(dps, "_target_paused_review_gate", lambda tid: False)
        pages = _notify_spy(monkeypatch)

        actions = dps.close_out_sweep("agents-core", path)
        assert "worker_failed:close-out" in actions, actions
        # exactly one HIGH page, the D5 command-shaped text, gem id in
        # the text (L4)
        assert len(pages) == 1
        assert pages[0]["priority"] == sys.modules["agents_core.notify"].Priority.HIGH
        assert "holding for PM" in pages[0]["message"]
        assert "gem g1" in pages[0]["message"]
        data = json.loads(ledger_path.read_text())
        assert data[sig]["status"] == "worker_failed"

        # idempotent: a second sweep over the worker_failed entry re-pages
        # nothing (the page is transition-gated, I2)
        pages.clear()
        actions2 = dps.close_out_sweep("agents-core", path)
        assert actions2 == []
        assert pages == []

    def test_close_out_closed_unmerged_pr_is_worker_failed_not_resolved(
            self, tmp_path, monkeypatch):
        """The dead-code fix (cycle-1 med-1): a CLOSED (unmerged) salvage
        PR is the D5 signal (a) - the operator discarded the salvage - so
        the sweep transitions `worker_failed` (one HIGH page), NOT
        `resolved` (the pre-fix code resolved a closed PR and the
        `elif state == "closed"` branch below it was unreachable)."""
        path = "/srv/git/agents-core-working"
        ledger_path = tmp_path / "ledger.json"
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        monkeypatch.setattr(dps, "_LEDGER_LOCK_TIMEOUT_S", 0.2)
        sig = dps.compute_signature(path, dps.CLASS_STALE_DIRTY)
        e = _entry(path, "agents-core", dps.CLASS_STALE_DIRTY, status="salvaged",
                   gem_id="g1")
        e["pr_number"] = 14
        e["target_id"] = "deploy-repair-agents-core-abc123"
        dps.write_ledger({sig: e}, path=ledger_path)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "closed")
        monkeypatch.setattr(dps, "_rejection_count", lambda tid: 0)
        monkeypatch.setattr(dps, "_target_paused_review_gate", lambda tid: False)
        pages = _notify_spy(monkeypatch)

        actions = dps.close_out_sweep("agents-core", path)
        assert "worker_failed:close-out" in actions, actions
        assert not any(a.startswith("resolved:") for a in actions), actions
        assert len(pages) == 1
        assert pages[0]["priority"] == sys.modules["agents_core.notify"].Priority.HIGH
        data = json.loads(ledger_path.read_text())
        assert data[sig]["status"] == "worker_failed"

    def test_close_out_sweep_direct_bounded_idempotent(self, tmp_path, monkeypatch):
        """The sweep is bounded (path-scoped read, one PR-state read per
        salvaged entry) and idempotent (a second run over a resolved entry
        is a no-op)."""
        ledger_path = tmp_path / "ledger.json"
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        monkeypatch.setattr(dps, "_LEDGER_LOCK_TIMEOUT_S", 0.2)
        path = "/srv/git/agents-core-working"
        sig = dps.compute_signature(path, dps.CLASS_STALE_DIRTY)
        e = _entry(path, "agents-core", dps.CLASS_STALE_DIRTY, status="salvaged")
        e["pr_number"] = 12
        e["target_id"] = "deploy-repair-agents-core-abc123"
        # a second entry on a DIFFERENT path must not be touched
        other = _entry("/other", "agents-core", dps.CLASS_STALE_DIRTY,
                       status="salvaged")
        other["pr_number"] = 99
        other["target_id"] = "tid-other"
        dps.write_ledger({sig: e, other["signature"]: other}, path=ledger_path)

        pr_reads: list = []
        monkeypatch.setattr(
            dps, "_pr_state",
            lambda repo, pr: (pr_reads.append(pr) or "merged"),
        )
        pages = _notify_spy(monkeypatch)
        actions = dps.close_out_sweep("agents-core", path)
        assert any(a.startswith("resolved:salvaged-merged:") for a in actions)
        assert pr_reads == [12]  # path-scoped: only this path's PR read
        data = json.loads(ledger_path.read_text())
        assert data[sig]["status"] == "resolved"
        assert data[other["signature"]]["status"] == "salvaged"  # untouched
        # resolved is a close-out, not a page (I2: the sweep pages only on
        # the worker_failed transition)
        assert pages == []
        # idempotent: a second run is a no-op (no PR read, no write)
        pr_reads.clear()
        actions2 = dps.close_out_sweep("agents-core", path)
        assert pr_reads == []
        assert actions2 == []
        data2 = json.loads(ledger_path.read_text())
        assert data2[sig]["status"] == "resolved"

    def test_success_pull_close_out_sweep_fires_when_revparse_fails(
            self, monkeypatch, tmp_path):
        """Low fix (cycle-1): the success-path close-out seam fires even
        when the pre_head rev-parse FAILS (pre_head empty) - the OQ-4
        option-3 contract is that the seam runs on EVERY successful pull
        for a critical repo, and a rev-parse hiccup must not drop the
        Close-Out Sweep (the salvaged -> resolved loop)."""
        from lapis_pm import pm_core

        path = "/srv/git/agents-core-working"
        ledger_path = tmp_path / "ledger.json"
        sig = dps.compute_signature(path, dps.CLASS_STALE_DIRTY)
        e = _entry(path, "agents-core", dps.CLASS_STALE_DIRTY, status="salvaged")
        e["pr_number"] = 15
        e["target_id"] = "deploy-repair-agents-core-abc123"
        dps.write_ledger({sig: e}, path=ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_FILE", ledger_path)
        monkeypatch.setattr(dps, "_LEDGER_LOCK_FILE", tmp_path / "ledger.lock")
        monkeypatch.setattr(dps, "_LEDGER_LOCK_TIMEOUT_S", 0.2)
        monkeypatch.setattr(dps, "_pr_state", lambda repo, pr: "merged")
        supersede = MagicMock()
        monkeypatch.setattr(dps, "_supersede_gem", supersede)
        monkeypatch.setattr(dps, "_rejection_count", lambda tid: 0)
        monkeypatch.setattr(dps, "_target_paused_review_gate", lambda tid: False)

        calls: list = []
        real_pass = dps.pass_handled_failure

        def spy_pass(repo, path_, trigger, *, pull_rc, pull_stderr=""):
            calls.append((repo, path_, pull_rc))
            return real_pass(repo, path_, trigger, pull_rc=pull_rc,
                             pull_stderr=pull_stderr)

        monkeypatch.setattr(dps, "pass_handled_failure", spy_pass)
        with patch("lapis_pm.deploy_pull_selfheal.pass_handled_failure",
                   side_effect=spy_pass):
            # rev-parse FAILS (rc=128): pre_head stays empty. The pull
            # itself succeeds; the close-out seam must still fire.
            def fake_run(cmd, **kwargs):
                if cmd[0] == "git" and "rev-parse" in cmd:
                    return _git_cp(rc=128, stderr="fatal: not a git repository")
                if cmd[0] == "git" and "status" in cmd:
                    return _git_cp(rc=0, stdout="")
                if cmd[0] == "git" and "pull" in cmd:
                    return _git_cp(rc=0, stdout="Already up to date.")
                return _git_cp(rc=0, stdout="active")

            with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
                with patch.object(pm_core, "_deploy_pull_locked",
                                  return_value=None):
                    advanced = pm_core._post_land_git_pull(
                        "agents-core", trigger="backstop-timer")
        assert advanced is False
        assert ("agents-core", path, 0) in calls, calls
        # the sweep transitioned the salvaged entry to resolved (merged
        # PR) despite the rev-parse failure
        data = json.loads(ledger_path.read_text())
        assert data[sig]["status"] == "resolved"
        supersede.assert_called_once()
