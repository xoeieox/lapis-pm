"""D6/D7/D9/D10 tests (attestation-contract-v0 leg 2).

Coverage (per spec Tests section items 7-11):
  D6  tick-coverage stall checker (dedicated unit, rev-2 pins):
      46-min-old ACTIVE target pages once with the required content; a
      fresh cursor does not page; a cursor advance clears the episode;
      a PAUSED target with a 3-day-old cursor produces NO page; an
      ACTIVE target with NO cursor record uses its bound ts as the
      watermark; a Forgejo outage at/above the 3-strike threshold -> NO
      checker pages; the checker module carries no LLM imports and the
      unit file carries no EnvironmentFile lines.
  D7  directive-outcome stall page:
      directive + pm:pr-head baseline with no outcome >15 min pages once
      naming the directive + the force-dispatch command; a dispatch /
      PR-head advance past the baseline / brief action suppresses it; a
      none-baseline + open PR counts as an advance; the detector makes
      no Forgejo call.
  D9  land-pass ghost-file resolution:
      the #307 shape (diff names files absent from the merged head + a
      test file) yields zero ghost rows; a genuinely missing runtime
      script still marks DRIFT.
  D10 backstop unit files:
      the repo systemd/lapis-pm-backstop.{service,timer} files match the
      live units byte-for-byte when the live units are present (host op
      keeps them in sync; skipped when absent).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core
from lapis_pm import node_identity

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem_store(monkeypatch, tmp_path):
    """A hermetic MemoryStore bound to a tmp file (tests-only isolation)."""
    from agents_core.mem import MemoryStore
    store = MemoryStore(db_path=tmp_path / "mem.db")
    monkeypatch.setattr(node_identity, "writable_store",
                        lambda path=None: store)
    return store


def _fake_target(tid: str, *, pm_bound: bool = True, paused: bool = False):
    t = MagicMock()
    t.id = tid
    t.pm_bound = pm_bound
    t.paused = paused
    t.data = {}
    return t


def _comment(ts: str, content: str, tags: list[str]):
    c = MagicMock()
    c.ts = ts
    c.content = content
    c.tags = tags
    c.author = "Erah"
    c.id = f"cid-{ts}"
    return c


def _now_pacific() -> datetime:
    return datetime.now(pm_core.PACIFIC)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# D6 - tick-coverage stall checker
# ---------------------------------------------------------------------------

class TestTickCoverageStall:

    def test_active_target_46min_old_cursor_pages_once(self, mem_store):
        """An ACTIVE bound target with a synthetic 46-min-old cursor pages
        once with the required content (target id, cursor age, resume
        command; the last-decision line when present)."""
        tid = "stall-tid-1"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        old = now - timedelta(minutes=46)
        mem_store.set(pm_core._cursor_key(tid), _iso(old),
                      tags=["lapis-pm", "cursor"])
        mem_store.set(pm_core._last_decision_key(tid), "noop:no_change",
                      tags=["lapis-pm", "last-decision"])

        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = []
            n = pm_core._check_tick_stalls(store, now)

        assert n == 1
        assert mock_notify.call_count == 1
        call = mock_notify.call_args
        assert call.args[0]  # the message is the first positional arg
        assert call.kwargs["source"] == "lapis-pm-stall-check"
        assert call.kwargs["priority"].name == "HIGH"
        msg = call.args[0]
        assert tid in msg
        assert "46 min" in msg
        assert "last-decision=noop:no_change" in msg
        assert f"tick --target {tid}" in msg
        # Episode dedup stamped.
        assert mem_store.get(pm_core._tick_stall_key(tid)) is not None

    def test_fresh_cursor_does_not_page(self, mem_store):
        tid = "stall-tid-2"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        mem_store.set(pm_core._cursor_key(tid), _iso(now - timedelta(minutes=5)),
                      tags=["lapis-pm", "cursor"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = []
            n = pm_core._check_tick_stalls(store, now)
        assert n == 0
        assert mock_notify.call_count == 0

    def test_cursor_advance_clears_episode(self, mem_store):
        """A cursor advance clears the episode: after a page, a newer
        cursor (the stall recovered) resets the dedup so the NEXT stall
        episode can page again."""
        tid = "stall-tid-3"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        old = now - timedelta(minutes=46)
        mem_store.set(pm_core._cursor_key(tid), _iso(old),
                      tags=["lapis-pm", "cursor"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = []
            assert pm_core._check_tick_stalls(store, now) == 1
            # Same stale cursor, same episode -> no second page.
            assert pm_core._check_tick_stalls(store, now) == 0
            assert mock_notify.call_count == 1
        # Recovery: the cursor advanced past the last-paged ts.
        mem_store.set(pm_core._cursor_key(tid), _iso(now - timedelta(minutes=1)),
                      tags=["lapis-pm", "cursor"])
        # Then it stalls again (a NEW episode: the cursor is stale again
        # AND the stall dedup was cleared by the cursor advance).
        stale2 = now - timedelta(minutes=46)
        mem_store.set(pm_core._cursor_key(tid), _iso(stale2),
                      tags=["lapis-pm", "cursor"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = []
            assert pm_core._check_tick_stalls(store, now) == 1
        assert mock_notify.call_count == 1

    def test_paused_target_3day_old_cursor_no_page(self, mem_store):
        """PAUSED targets are EXCLUDED by design: a paused target with a
        3-day-old cursor produces NO page."""
        tid = "stall-tid-paused"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid, paused=True)]
        now = _now_pacific()
        mem_store.set(pm_core._cursor_key(tid),
                      _iso(now - timedelta(days=3)),
                      tags=["lapis-pm", "cursor"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = []
            n = pm_core._check_tick_stalls(store, now)
        assert n == 0
        assert mock_notify.call_count == 0

    def test_no_cursor_uses_bound_ts_watermark(self, mem_store):
        """An ACTIVE target with NO cursor record uses its bound ts (the
        spec:bound comment ts) as the watermark: a 46-min-old bound ts
        pages."""
        tid = "stall-tid-nocursor"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        bound_ts = _iso(now - timedelta(minutes=46))
        with (
            patch.object(pm_core, "_spec_bound_ts", return_value=bound_ts),

            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [
                _comment(bound_ts, "spec body", ["spec:bound"])
            ]
            n = pm_core._check_tick_stalls(store, now)
        assert n == 1
        assert mock_notify.call_count == 1
        assert tid in mock_notify.call_args.args[0]

    def test_forgejo_outage_at_threshold_no_pages(self, mem_store):
        """A Forgejo outage at or above the 3-strike threshold -> NO
        checker pages for any target (the 3-strike page owns that
        episode - I5)."""
        tid = "stall-tid-forgejo"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        mem_store.set(pm_core._cursor_key(tid),
                      _iso(now - timedelta(minutes=46)),
                      tags=["lapis-pm", "cursor"])
        for fails in (pm_core.FORGEJO_UNREACHABLE_THRESHOLD,
                      pm_core.FORGEJO_UNREACHABLE_THRESHOLD + 2):
            mem_store.set(pm_core.FORGEJO_CONSECUTIVE_FAILS_KEY, str(fails),
                          tags=["lapis-pm", "forgejo-health"])
            with (
                patch.object(pm_core, "episodic") as fake_ep,
                patch("agents_core.notify.send_notification") as mock_notify,
            ):
                fake_ep.all_comments.return_value = []
                assert pm_core._check_tick_stalls(store, now) == 0
            assert mock_notify.call_count == 0

    def test_threshold_constant_justified(self):
        # The constant is named and the justification is in the module
        # docstring/comment (spec DoD).
        assert pm_core.TICK_COVERAGE_STALL_S == 2700
        assert pm_core.DIRECTIVE_OUTCOME_STALL_S == 900

    def test_stall_check_module_has_no_llm_imports(self):
        """The checker is stdlib + mem reads only: no LLM-call machinery
        (ClaudeQueue / call_claude_cli / call_gw_agent) imported or called
        in stall_check.py (and the detector path it drives uses no LLM
        call)."""
        src = (REPO_ROOT / "lapis_pm" / "stall_check.py").read_text()
        import ast
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.update(
                    f"{node.module}.{a.name}" for a in node.names
                )
        for name in ("ClaudeQueue", "call_claude_cli", "call_gw_agent"):
            assert not any(name in i for i in imported), name

    def test_stall_check_unit_carries_no_environment_file(self):
        """The stall-check unit file carries NO EnvironmentFile lines
        (a stdlib-only unit reading no credentials must not inherit
        conductor.env / phala.env)."""
        svc = (REPO_ROOT / "systemd" / "lapis-pm-stall-check.service").read_text()
        assert "EnvironmentFile" not in svc
        # The timer is independent of the tick's lifecycle.
        timer = (REPO_ROOT / "systemd" / "lapis-pm-stall-check.timer").read_text()
        assert "OnUnitActiveSec=600" in timer

    def test_lapis_pm_service_timeout_bumped_to_1800(self):
        """D6b: TimeoutStartSec=300 -> 1800 in the repo unit file."""
        svc = (REPO_ROOT / "systemd" / "lapis-pm.service").read_text()
        assert "TimeoutStartSec=1800" in svc
        assert "TimeoutStartSec=300" not in svc


# ---------------------------------------------------------------------------
# D7 - directive-outcome stall page
# ---------------------------------------------------------------------------

class TestDirectiveOutcomeStall:

    def test_directive_no_outcome_pages_once(self, mem_store):
        """A directive-seen (with pm:pr-head baseline) and no outcome for
        >15 min pages once naming the directive + the force-dispatch
        command."""
        tid = "dir-tid-1"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=16))
        obs = _comment(
            dir_ts,
            "Directive received from Erah at " + dir_ts + ":\nstop the burn",
            ["pm:observation", "pm:directive-seen",
             "pm:directive-id=d-42", "pm:pr-head=abc123"],
        )
        # No dispatch after the directive, PR head unchanged (abc123),
        # outstanding brief still set (not actioned).
        mem_store.set(pm_core._dispatched_key(tid), "[]",
                      tags=["lapis-pm", "dispatched"])
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 7, "head_sha": "abc123"}]),
                      tags=["lapis-pm", "pr-head"])
        mem_store.set(pm_core._brief_key(tid), "cid-brief-1",
                      tags=["lapis-pm", "outstanding-brief"])

        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs]
            n = pm_core._check_directive_stalls(store, now)

        assert n == 1
        assert mock_notify.call_count == 1
        call = mock_notify.call_args
        assert call.args[0]  # the message is the first positional arg
        assert call.kwargs["source"] == "lapis-pm-directive-stall"
        assert call.kwargs["priority"].name == "HIGH"
        msg = call.args[0]
        assert tid in msg
        assert "Erah" in msg
        assert "stop the burn" in msg
        assert "--force-dispatch" in msg
        assert f"tick --target {tid}" in msg
        # Dedup stamped.
        assert mem_store.get(
            pm_core._directive_stall_key(tid, "d-42")) is not None

    def test_dispatch_after_directive_suppresses(self, mem_store):
        tid = "dir-tid-2"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=16))
        obs = _comment(dir_ts, "Directive received:\nx",
                       ["pm:directive-seen", "pm:directive-id=d-1",
                        "pm:pr-head=abc123"])
        # A dispatch recorded AFTER the directive ts -> outcome (a).
        rec = json.dumps([{"ts": _iso(now - timedelta(minutes=5)),
                           "agent_type": "fixer"}])
        mem_store.set(pm_core._dispatched_key(tid), rec,
                      tags=["lapis-pm", "dispatched"])
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 7, "head_sha": "abc123"}]),
                      tags=["lapis-pm", "pr-head"])
        mem_store.set(pm_core._brief_key(tid), "cid-brief",
                      tags=["lapis-pm", "outstanding-brief"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs]
            assert pm_core._check_directive_stalls(store, now) == 0
        assert mock_notify.call_count == 0

    def test_pr_head_advance_past_baseline_suppresses(self, mem_store):
        tid = "dir-tid-3"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=16))
        obs = _comment(dir_ts, "Directive received:\nx",
                       ["pm:directive-seen", "pm:directive-id=d-1",
                        "pm:pr-head=abc123"])
        mem_store.set(pm_core._dispatched_key(tid), "[]",
                      tags=["lapis-pm", "dispatched"])
        # The open PR head advanced past the baseline -> outcome (b).
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 7, "head_sha": "def456"}]),
                      tags=["lapis-pm", "pr-head"])
        mem_store.set(pm_core._brief_key(tid), "cid-brief",
                      tags=["lapis-pm", "outstanding-brief"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs]
            assert pm_core._check_directive_stalls(store, now) == 0
        assert mock_notify.call_count == 0

    def test_brief_actioned_suppresses(self, mem_store):
        tid = "dir-tid-4"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=16))
        obs = _comment(dir_ts, "Directive received:\nx",
                       ["pm:directive-seen", "pm:directive-id=d-1",
                        "pm:pr-head=abc123"])
        mem_store.set(pm_core._dispatched_key(tid), "[]",
                      tags=["lapis-pm", "dispatched"])
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 7, "head_sha": "abc123"}]),
                      tags=["lapis-pm", "pr-head"])
        # The directive's brief was actioned (outstanding brief consumed)
        # -> outcome (c).
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs]
            assert pm_core._check_directive_stalls(store, now) == 0
        assert mock_notify.call_count == 0

    def test_none_baseline_open_pr_counts_as_advance(self, mem_store):
        tid = "dir-tid-5"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=16))
        # pm:pr-head=none baseline (no PR open at directive time).
        obs = _comment(dir_ts, "Directive received:\nx",
                       ["pm:directive-seen", "pm:directive-id=d-1",
                        "pm:pr-head=none"])
        mem_store.set(pm_core._dispatched_key(tid), "[]",
                      tags=["lapis-pm", "dispatched"])
        # Any open PR now counts as an advance.
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 9, "head_sha": "fff789"}]),
                      tags=["lapis-pm", "pr-head"])
        mem_store.set(pm_core._brief_key(tid), "cid-brief",
                      tags=["lapis-pm", "outstanding-brief"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs]
            assert pm_core._check_directive_stalls(store, now) == 0
        assert mock_notify.call_count == 0

    def test_recent_directive_does_not_page(self, mem_store):
        tid = "dir-tid-6"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=5))  # < 900s
        obs = _comment(dir_ts, "Directive received:\nx",
                       ["pm:directive-seen", "pm:directive-id=d-1",
                        "pm:pr-head=abc123"])
        mem_store.set(pm_core._dispatched_key(tid), "[]",
                      tags=["lapis-pm", "dispatched"])
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 7, "head_sha": "abc123"}]),
                      tags=["lapis-pm", "pr-head"])
        mem_store.set(pm_core._brief_key(tid), "cid-brief",
                      tags=["lapis-pm", "outstanding-brief"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs]
            assert pm_core._check_directive_stalls(store, now) == 0
        assert mock_notify.call_count == 0

    def test_detector_makes_no_forgejo_call(self):
        """The detector path makes no direct Forgejo call (it is mem +
        episodic reads only - the current PR head comes from the
        pm/pr-head/<tid> record the tick's PR-perception site writes)."""
        import inspect
        for fn in (pm_core._check_directive_stalls,
                   pm_core._directive_outcome_holds,
                   pm_core._directive_seen_observations):
            src = inspect.getsource(fn)
            assert "get_open_prs" not in src, fn.__name__
            assert "_forgejo_get_pr" not in src, fn.__name__
            assert "httpx" not in src, fn.__name__

    def test_directive_branch_carries_pr_head_baseline(self):
        """The directive-seen observation's extra_tags seam carries
        pm:pr-head=<sha> (the open PR head at directive time) or
        pm:pr-head=none when no PR is open."""
        import inspect
        src = inspect.getsource(pm_core.tick)
        assert "pm:pr-head=" in src
        assert '"pm:pr-head={_dir_baseline}"' in src


# ---------------------------------------------------------------------------
# D9 - land-pass ghost-file resolution
# ---------------------------------------------------------------------------

class TestJustMergedPrScripts:

    def _run(self, clone: Path, diff_names: list[str],
             head_files: list[str] | None) -> list[str]:
        """Run _just_merged_pr_scripts against a real git clone."""
        log_line = (f"2026-09-08T12:00:00Z post-land-hook {clone} "
                    f"old=aaaaaaaaaaaa new=bbbbbbbbbbbb")

        def fake_git(args, **kw):
            if args[:2] == ["git", "-C"] and "diff" in args:
                return subprocess.CompletedProcess(
                    args, 0, stdout="\n".join(diff_names) + "\n")
            if args[:2] == ["git", "-C"] and "ls-tree" in args:
                if head_files is None:
                    return subprocess.CompletedProcess(args, 1, stdout="")
                return subprocess.CompletedProcess(
                    args, 0, stdout="\n".join(head_files) + "\n")
            return subprocess.CompletedProcess(args, 0, stdout="")

        with (
            patch("subprocess.run", side_effect=fake_git),
            patch.object(pm_core, "_last_deploy_log_shas",
                         return_value=("a" * 12, "b" * 12)),
        ):
            return pm_core._just_merged_pr_scripts("lapis-pm", [str(clone)])

    def test_ghost_307_shape_yields_no_ghost_rows(self):
        """The #307 shape re-run: a PR whose diff range names files absent
        from the merged head (cli.py, orchestrator.py - salvage-branch
        residue) + a test file (test_council_resilience.py) -> no ghost
        DRIFT rows."""
        diff_names = [
            "cli.py",
            "orchestrator.py",
            "tests/test_council_resilience.py",
            "lapis_pm/pm_core.py",
        ]
        head_files = ["lapis_pm/pm_core.py", "lapis_pm/cli.py"]
        out = self._run(Path("/tmp/fake-clone"), diff_names, head_files)
        # cli.py is absent from the merged head -> no ghost row.
        assert "cli.py" not in out
        # orchestrator.py is absent from the merged head -> no ghost row.
        assert "orchestrator.py" not in out
        # test_council_resilience.py is a test-path file -> excluded.
        assert "test_council_resilience.py" not in out
        # The genuinely-present runtime script is a candidate.
        assert out == ["pm_core.py"]

    def test_genuinely_missing_runtime_script_still_marks(self):
        """A genuinely missing runtime script (present at the merged HEAD,
        named by the diff) still marks DRIFT - the signal's real purpose is
        preserved."""
        diff_names = ["lapis_pm/foo.py", "lapis_pm/bar.py"]
        head_files = ["lapis_pm/foo.py", "lapis_pm/bar.py"]
        out = self._run(Path("/tmp/fake-clone"), diff_names, head_files)
        assert out == ["foo.py", "bar.py"]

    def test_test_path_exclusion_rules(self):
        """tests/ path component OR test_/conftest basename -> excluded."""
        diff_names = [
            "tests/sub/helper.py",          # tests/ path component
            "lapis_pm/test_widget.py",      # test_ basename
            "lapis_pm/conftest.py",         # conftest basename
            "lapis_pm/real.py",
        ]
        head_files = [
            "tests/sub/helper.py",
            "lapis_pm/test_widget.py",
            "lapis_pm/conftest.py",
            "lapis_pm/real.py",
        ]
        out = self._run(Path("/tmp/fake-clone"), diff_names, head_files)
        assert out == ["real.py"]


# ---------------------------------------------------------------------------
# D10 - backstop unit files repo-tracked
# ---------------------------------------------------------------------------

class TestBackstopUnitsRepoTracked:

    LIVE_SVC = Path("/etc/systemd/system/lapis-pm-backstop.service")
    LIVE_TIMER = Path("/etc/systemd/system/lapis-pm-backstop.timer")

    def test_repo_backstop_unit_files_exist(self):
        svc = REPO_ROOT / "systemd" / "lapis-pm-backstop.service"
        timer = REPO_ROOT / "systemd" / "lapis-pm-backstop.timer"
        assert svc.exists()
        assert timer.exists()
        svc_text = svc.read_text()
        timer_text = timer.read_text()
        assert "Type=oneshot" in svc_text
        assert "backstop-sweep" in svc_text
        assert "TimeoutStartSec=1200" in svc_text
        assert "OnUnitActiveSec=900" in timer_text

    @pytest.mark.skipif(
        not (LIVE_SVC.exists() and LIVE_TIMER.exists()),
        reason="live units only present on the host that runs them",
    )
    def test_repo_files_match_live_units(self):
        """The repo files match the live units byte-for-byte (the live
        text is the source of truth)."""
        assert (REPO_ROOT / "systemd" / "lapis-pm-backstop.service").read_text() \
            == self.LIVE_SVC.read_text()
        assert (REPO_ROOT / "systemd" / "lapis-pm-backstop.timer").read_text() \
            == self.LIVE_TIMER.read_text()
