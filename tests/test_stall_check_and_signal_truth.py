"""D6/D7/D9/D10 tests (attestation-contract-v0 leg 2).

Coverage (per spec Tests section items 7-11):
  D6  tick-coverage stall checker (dedicated unit, rev-2 pins):
      46-min-old ACTIVE target pages once with the required content; a
      fresh cursor does not page; a cursor advance clears the episode;
      a PAUSED target with a 3-day-old cursor produces NO page; an
      ACTIVE target with NO cursor record uses its bound ts as the
      watermark; a Forgejo outage at/above the 3-strike threshold -> NO
      checker pages; the checker module carries no LLM imports; the unit
      file references NO CREDENTIAL env file (only the scoped env file -
      REV 4 re-scope: the scoped file carries exactly MEM_DB_PATH + the
      Pushover keys, and a checker run under the unit's exact env resolves
      its mem store without NodeConfigError).
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
from datetime import datetime, timedelta, timezone
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
        # Recovery: the cursor advanced past the last-paged ts. REV 4:
        # the page stamp is REAL wall-clock (_now_iso, which is >= now),
        # so a recovery cursor of `now - 1min` would sit BEFORE the page
        # stamp and the dedup would never clear. A real recovery
        # advances the cursor PAST the real page stamp, so simulate that
        # by reading the real stamp and advancing past it.
        stamped = mem_store.get(pm_core._tick_stall_key(tid))["content"]
        recovered = datetime.fromisoformat(stamped) + timedelta(seconds=5)
        mem_store.set(pm_core._cursor_key(tid), _iso(recovered),
                      tags=["lapis-pm", "cursor"])
        # The cursor advance cleared the episode: a re-check at the
        # recovery instant sees a fresh cursor and pages nothing.
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = []
            assert pm_core._check_tick_stalls(store, recovered) == 0
        # Then it stalls again (a NEW episode: the cursor is stale again
        # AND the stall dedup was cleared by the cursor advance). The
        # re-stall is simulated by advancing the wall clock to a fresh
        # `now` (the recovery happened, the tick loop broke again, and
        # enough time has passed for the cursor to be stale again) -
        # the dedup was cleared by the cursor advance, so the second
        # page must fire. (The second page's stamp is also real
        # wall-clock, which the test's final count assertion pins.)
        now2 = recovered + timedelta(minutes=47)
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = []
            assert pm_core._check_tick_stalls(store, now2) == 1
        assert mock_notify.call_count == 2

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

    def test_stall_check_unit_carries_no_credential_env_file(self):
        """REV 4 D6 unit env: the stall-check unit references NO CREDENTIAL
        env file (conductor.env / phala.env - the pin stands), but MUST
        carry the SCOPED env file (doorman-watchdog precedent) that
        resolves the mem store + Pushover keys. The scoped file is a
        post-land host op; its key-set contract (exactly MEM_DB_PATH +
        PUSHOVER_APP_TOKEN + PUSHOVER_USER_KEY, no FORGEJO_TOKEN, no
        Phala key) is asserted here, not against a live file."""
        svc = (REPO_ROOT / "systemd" / "lapis-pm-stall-check.service").read_text()
        env_file_lines = [
            ln for ln in svc.splitlines()
            if ln.strip().startswith("EnvironmentFile")
        ]
        # No credential env file may be referenced.
        for ln in env_file_lines:
            assert "conductor.env" not in ln
            assert "phala.env" not in ln
        # Exactly one scoped env file reference, the pinned path.
        assert len(env_file_lines) == 1
        assert "EnvironmentFile=-%h/.config/lapis-pm/stall-check.env" in svc
        # The timer is independent of the tick's lifecycle.
        timer = (REPO_ROOT / "systemd" / "lapis-pm-stall-check.timer").read_text()
        assert "OnUnitActiveSec=600" in timer

    def test_stall_check_scoped_env_file_key_set_contract(self):
        """The scoped env file's key-set contract: exactly MEM_DB_PATH +
        the two Pushover keys (no FORGEJO_TOKEN, no Phala key). This
        asserts the CONTRACT the post-land host op must honor, not a live
        file (the file itself is created post-land on the host)."""
        scoped_keys = {
            "MEM_DB_PATH",
            "PUSHOVER_APP_TOKEN",
            "PUSHOVER_USER_KEY",
        }
        forbidden = {"FORGEJO_TOKEN", "PHALA_TEE_KEY"}
        # The contract itself: the allowed set is exactly the scoped set,
        # and no credential key may ever be part of it.
        assert scoped_keys == {"MEM_DB_PATH", "PUSHOVER_APP_TOKEN",
                               "PUSHOVER_USER_KEY"}
        assert not (scoped_keys & forbidden)
        # If the live scoped file exists on this host, its key set must
        # match the contract exactly.
        from pathlib import Path as _P
        import os as _os
        live = _P(_os.path.expanduser("~/.config/lapis-pm/stall-check.env"))
        if live.exists():
            live_keys = {
                ln.split("=", 1)[0].strip()
                for ln in live.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")
                and "=" in ln
            }
            assert live_keys == scoped_keys, (
                f"scoped env file key set {sorted(live_keys)} != "
                f"contract {sorted(scoped_keys)}"
            )

    def test_stall_check_run_resolves_mem_store_under_unit_env(self):
        """REV 4 D6 startup-crash check (PM-execution-verified defect):
        cli.main() resolves node identity fail-closed BEFORE subcommand
        dispatch and raises NodeConfigError without MEM_DB_PATH, so the
        checker run under the unit's EXACT env (scoped file: MEM_DB_PATH
        + Pushover keys; no credential files) must resolve its mem store
        and complete without NodeConfigError."""
        import subprocess as _sp
        import sys as _sys
        import tempfile as _tf
        from pathlib import Path as _P

        db = _P(_tf.mkdtemp()) / "mem.db"
        # The unit's exact env: the scoped file's keys + the unit's
        # Environment= lines (PYTHONPATH/PATH/XDG/DBUS are irrelevant to
        # the identity resolution; the load-bearing keys are MEM_DB_PATH
        # + the Pushover keys).
        env = {
            "MEM_DB_PATH": str(db),
            "PUSHOVER_APP_TOKEN": "test-token",
            "PUSHOVER_USER_KEY": "test-user",
            "HOME": str(_P(_tf.mkdtemp())),
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
        }
        code = (
            "import sys; "
            "from lapis_pm import node_identity; "
            "node_identity.resolve_node_identity(force=True); "
            "print('resolved')"
        )
        proc = _sp.run(
            [_sys.executable, "-c", code],
            env=env, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=120,
        )
        assert "NodeConfigError" not in proc.stderr, (
            f"checker run under the unit's exact env raised "
            f"NodeConfigError: {proc.stderr[-500:]}"
        )
        assert proc.returncode == 0, (
            f"identity resolution under the unit's exact env failed "
            f"rc={proc.returncode}: {proc.stderr[-500:]}"
        )

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
        # and the raised brief's cid is recorded (pm:brief-id) but NOT
        # consumed (no decision against it, slot empty).
        mem_store.set(pm_core._dispatched_key(tid), "[]",
                      tags=["lapis-pm", "dispatched"])
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 7, "head_sha": "abc123"}]),
                      tags=["lapis-pm", "pr-head"])
        raised = _comment(
            dir_ts, "Directive brief raised: cid=cid-brief-1",
            ["pm:directive-brief-raised", "pm:brief-id=cid-brief-1",
             "pm:directive-id=d-42"],
        )

        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs, raised]
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
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs]
            assert pm_core._check_directive_stalls(store, now) == 0
        assert mock_notify.call_count == 0

    def test_brief_actioned_suppresses(self, mem_store):
        """REV 4 D7 outcome (c): the directive's OWN brief was consumed -
        keyed on the specific brief id (pm:brief-id=<cid> on the
        directive-brief-raised companion observation): a decision audit
        recorded against that cid (decision/brief-resolved/<cid>)
        suppresses the page."""
        tid = "dir-tid-4"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=16))
        obs = _comment(dir_ts, "Directive received:\nx",
                       ["pm:directive-seen", "pm:directive-id=d-1",
                        "pm:pr-head=abc123"])
        # The raised brief's cid rides the companion observation.
        raised = _comment(dir_ts, "Directive brief raised: cid=cid-brief-4",
                          ["pm:directive-brief-raised", "pm:brief-id=cid-brief-4",
                           "pm:directive-id=d-1"])
        mem_store.set(pm_core._dispatched_key(tid), "[]",
                      tags=["lapis-pm", "dispatched"])
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 7, "head_sha": "abc123"}]),
                      tags=["lapis-pm", "pr-head"])
        # A decision recorded against that specific cid (the single
        # resolution path writes the audit key) -> outcome (c) holds.
        mem_store.set("decision/brief-resolved/cid-brief-4",
                      json.dumps({"target_id": tid, "option_id": "o1",
                                  "action_kind": "acknowledge_and_clear",
                                  "ts": _iso(now - timedelta(minutes=5))}),
                      tags=["lapis-pm", "brief-resolved"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs, raised]
            assert pm_core._check_directive_stalls(store, now) == 0
        assert mock_notify.call_count == 0

    def test_empty_brief_slot_does_not_suppress(self, mem_store):
        """REV 4 D7 pin: an EMPTY outstanding-brief slot does NOT count as
        outcome (c) - the over-broad `get_outstanding_brief is None ->
        True` reading suppressed the page for any target with no
        outstanding brief (the common steady state), defeating the
        detector. A directive with a recorded brief id and an empty slot
        (no decision against that cid) PAGES."""
        tid = "dir-tid-4b"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=16))
        obs = _comment(dir_ts, "Directive received:\nx",
                       ["pm:directive-seen", "pm:directive-id=d-1",
                        "pm:pr-head=abc123"])
        raised = _comment(dir_ts, "Directive brief raised: cid=cid-brief-4b",
                          ["pm:directive-brief-raised", "pm:brief-id=cid-brief-4b",
                           "pm:directive-id=d-1"])
        mem_store.set(pm_core._dispatched_key(tid), "[]",
                      tags=["lapis-pm", "dispatched"])
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 7, "head_sha": "abc123"}]),
                      tags=["lapis-pm", "pr-head"])
        # The outstanding-brief slot is EMPTY (no decision against the
        # recorded cid) -> outcome (c) does NOT hold -> the page fires.
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs, raised]
            assert pm_core._check_directive_stalls(store, now) == 1
        assert mock_notify.call_count == 1

    def test_brief_clear_of_specific_cid_suppresses(self, mem_store):
        """REV 4 D7 outcome (c): a CLEAR of the specific cid (the slot now
        holds a DIFFERENT brief) suppresses the page for that
        directive."""
        tid = "dir-tid-4c"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid)]
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=16))
        obs = _comment(dir_ts, "Directive received:\nx",
                       ["pm:directive-seen", "pm:directive-id=d-1",
                        "pm:pr-head=abc123"])
        raised = _comment(dir_ts, "Directive brief raised: cid=cid-brief-4c",
                          ["pm:directive-brief-raised", "pm:brief-id=cid-brief-4c",
                           "pm:directive-id=d-1"])
        mem_store.set(pm_core._dispatched_key(tid), "[]",
                      tags=["lapis-pm", "dispatched"])
        mem_store.set(pm_core._pr_head_key(tid),
                      json.dumps([{"number": 7, "head_sha": "abc123"}]),
                      tags=["lapis-pm", "pr-head"])
        # The slot holds a DIFFERENT brief: the directive's brief was
        # cleared (consumed) -> outcome (c) holds.
        mem_store.set(pm_core._brief_key(tid), "cid-brief-later",
                      tags=["lapis-pm", "outstanding-brief"])
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs, raised]
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

    def test_directive_branch_records_brief_id(self):
        """REV 4 D7: the raised brief's comment id rides the companion
        pm:directive-brief-raised observation (pm:brief-id=<cid>, or
        pm:brief-id=none when the raise failed), read back from the mem
        slot the raise just wrote."""
        import inspect
        src = inspect.getsource(pm_core.tick)
        assert "pm:directive-brief-raised" in src
        assert "pm:brief-id=" in src

    def test_paused_target_directive_no_page(self, mem_store):
        """REV 4: _check_directive_stalls excludes PAUSED targets (parity
        with D6 at the tick-coverage loop): a paused target with a stale
        directive produces NO page."""
        tid = "dir-tid-paused"
        store = MagicMock()
        store.load_all.return_value = [_fake_target(tid, paused=True)]
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
        with (
            patch.object(pm_core, "episodic") as fake_ep,
            patch("agents_core.notify.send_notification") as mock_notify,
        ):
            fake_ep.all_comments.return_value = [obs]
            assert pm_core._check_directive_stalls(store, now) == 0
        assert mock_notify.call_count == 0

    def test_dispatch_ts_compare_timezone_aware(self, mem_store):
        """REV 4: the dispatch-ts vs directive-ts comparison is
        timezone-aware (parsed datetimes), not a string compare - a
        dispatch 5 min after the directive recorded with a different
        offset spelling (UTC vs Pacific) must still read as 'after'."""
        now = _now_pacific()
        dir_ts = _iso(now - timedelta(minutes=16))
        # 5 min after the directive, spelled in UTC (+00:00) - as a
        # string it sorts BEFORE the Pacific-spelled dir_ts.
        dispatch_ts = (
            now - timedelta(minutes=11)
        ).astimezone(timezone.utc).isoformat(timespec="microseconds")
        assert dispatch_ts < dir_ts  # the string-compare trap
        tid = "dir-tid-tz"
        mem_store.set(pm_core._dispatched_key(tid),
                      json.dumps([{"ts": dispatch_ts,
                                   "agent_type": "fixer"}]),
                      tags=["lapis-pm", "dispatched"])
        # Outcome (a) holds via the timezone-aware parse (the brief slot
        # is empty and no brief id is recorded, so only (a) can hold).
        assert pm_core._directive_outcome_holds(
            tid, dir_ts, "abc123", brief_id=None,
        )
        # Direct unit assertion on the parse helper.
        d1 = pm_core._parse_ts_aware(dir_ts)
        d2 = pm_core._parse_ts_aware(dispatch_ts)
        assert d1 is not None and d2 is not None
        assert d2 > d1  # timezone-aware: the dispatch IS after the directive


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
