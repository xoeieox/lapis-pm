"""Tests for U3 - demotion backstop (lapis-pm-autonomy-actuator-v0).

Coverage (per spec AC list):
  AC5  hook exception             -> tick-path never-raise, fail-soft to logging
  AC6  idempotency: re-running backstop-sweep within the same daily window
       processes at most one repo per tick (cursor advances, no re-run of a
       repo already swept this window)
  U3.1 sweep: one repo per call (round-robin cursor), red/green outcomes
  U3.2 attribution: red main + intersecting changed_paths -> suspect ->
       demote + HIGH brief (says "suspect, not proven")
  U3.3 autonomy-promote: manual promotion; demotion-cooldown guard refuses
       promotion with regression_count>0 and last_regression_ts within 14d
       without --force; --force recorded with promoted_by: "Erah-force"
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import autonomy_backstop as ab
from lapis_pm import precedent as pc
from lapis_pm import verification_attestation as va


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mem(store: dict | None = None) -> MagicMock:
    data: dict[str, dict] = {}
    if store:
        data.update(store)

    mock = MagicMock()

    def _set(key: str, content: str, tags: list | None = None, source: str = "") -> bool:
        data[key] = {"key": key, "content": content, "tags": tags or []}
        return True

    def _get(key: str) -> dict | None:
        return data.get(key)

    def _search(query: str = "", tags: list | None = None, limit: int = 100) -> list[str]:
        out = []
        for k, v in data.items():
            if query and (query in k or any(query in t for t in (v.get("tags") or []))):
                out.append(k)
        return out[:limit]

    mock.set.side_effect = _set
    mock.get.side_effect = _get
    mock.search.side_effect = _search
    mock._data = data
    return mock


def _att(result: str, pr_failures: list[str] | None = None, **kw) -> dict:
    base = {"result": result, "pr_failures": pr_failures or [],
            "preexisting": [], "suite": "python3 -m pytest -q",
            "ts": "2026-09-06T00:00:00Z"}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# U3.1 - sweep: one repo per call, red/green outcomes
# ---------------------------------------------------------------------------

class TestSweepOneRepoPerCall:

    def test_sweep_processes_single_repo_green(self):
        mem = _make_mem()
        key = "pm/autonomy-class/lapis-pm/clean/code/lt100"
        mem.set(key, json.dumps({"promoted": True, "auto_resolved_count": 0,
                                 "regression_count": 0}),
                tags=["lapis-pm", "pm:autonomy-class"])

        with (
            patch.object(va, "attest_main_baseline",
                         return_value=_att("attested", pr_failures=[])),
            patch("lapis_pm.episodic.write_observation"),
        ):
            out = ab.sweep(mem, now="2026-09-06T00:00:00Z")

        # Exactly one repo processed, green.
        assert out["repo"] == "lapis-pm"
        assert out["status"] == "green"
        assert out["suspect_prs"] == []

    def test_sweep_red_with_resolved_prs_demotes(self):
        mem = _make_mem()
        key = "pm/autonomy-class/lapis-pm/clean/code/lt100"
        mem.set(key, json.dumps({"promoted": True, "auto_resolved_count": 1,
                                 "regression_count": 0}),
                tags=["lapis-pm", "pm:autonomy-class"])
        # A resolved PR record (auto_resolved) with changed_paths.
        mem.set("pm/auto-resolved/my-target/42",
                json.dumps({"target_id": "my-target", "pr_number": 42,
                            "repo": "lapis-pm",
                            "changed_paths": ["lapis_pm/foo.py"],
                            "ts": "2026-09-05T00:00:00Z"}),
                tags=["lapis-pm", "pm:auto-resolved"])

        with (
            patch.object(va, "attest_main_baseline",
                         return_value=_att("unattested",
                                           pr_failures=["lapis_pm/foo.py::test_x"])),
            patch("lapis_pm.episodic.write_observation"),
        ):
            out = ab.sweep(mem, now="2026-09-06T00:00:00Z")

        assert out["status"] == "red"
        assert out["suspect_prs"]  # attributed to the resolved PR
        # Demotion fired on the suspect PR's fork class.
        rec = json.loads(mem._data[key]["content"])
        assert rec["promoted"] is False
        assert rec["regression_count"] == 1

    def test_sweep_green_no_demotion(self):
        mem = _make_mem()
        key = "pm/autonomy-class/lapis-pm/clean/code/lt100"
        mem.set(key, json.dumps({"promoted": True, "auto_resolved_count": 1,
                                 "regression_count": 0}),
                tags=["lapis-pm", "pm:autonomy-class"])
        mem.set("pm/auto-resolved/my-target/42",
                json.dumps({"target_id": "my-target", "pr_number": 42,
                            "repo": "lapis-pm",
                            "changed_paths": ["lapis_pm/foo.py"],
                            "ts": "2026-09-05T00:00:00Z"}),
                tags=["lapis-pm", "pm:auto-resolved"])

        with (
            patch.object(va, "attest_main_baseline",
                         return_value=_att("attested", pr_failures=[])),
            patch("lapis_pm.episodic.write_observation"),
        ):
            out = ab.sweep(mem, now="2026-09-06T00:00:00Z")

        assert out["status"] == "green"
        rec = json.loads(mem._data[key]["content"])
        assert rec["promoted"] is True  # no demotion on green


# ---------------------------------------------------------------------------
# AC6 - idempotency: daily gate + cursor advance
# ---------------------------------------------------------------------------

class TestSweepIdempotency:

    def test_daily_gate_skips_within_window(self):
        # AC6: re-running backstop-sweep within the same daily window does not
        # re-run a repo already swept this window.
        mem = _make_mem()
        key = "pm/autonomy-class/lapis-pm/clean/code/lt100"
        # Already swept today.
        mem.set(key, json.dumps({"promoted": True, "auto_resolved_count": 0,
                                 "regression_count": 0,
                                 "last_sweep_ts": "2026-09-06T00:00:00Z"}),
                tags=["lapis-pm", "pm:autonomy-class"])

        with (
            patch.object(va, "attest_main_baseline") as mock_baseline,
            patch("lapis_pm.episodic.write_observation"),
        ):
            out = ab.sweep(mem, now="2026-09-06T12:00:00Z")

        # The repo was already swept today -> skipped, no suite run.
        assert out["status"] == "skipped_already_swept_today"
        assert not mock_baseline.called

    def test_cursor_advances_across_repos(self):
        # AC6: cursor advances - a second tick processes the next repo.
        mem = _make_mem()
        for repo in ("lapis-pm", "agents-core"):
            key = f"pm/autonomy-class/{repo}/clean/code/lt100"
            mem.set(key, json.dumps({"promoted": True, "auto_resolved_count": 0,
                                     "regression_count": 0}),
                    tags=["lapis-pm", "pm:autonomy-class"])

        with (
            patch.object(va, "attest_main_baseline",
                         return_value=_att("attested", pr_failures=[])),
            patch("lapis_pm.episodic.write_observation"),
        ):
            out1 = ab.sweep(mem, now="2026-09-06T00:00:00Z")
        first_repo = out1["repo"]
        # After the first sweep, lapis-pm's last_sweep_ts is set. The next
        # sweep (next day) should pick the repo with the oldest (empty) stamp.
        with (
            patch.object(va, "attest_main_baseline",
                         return_value=_att("attested", pr_failures=[])),
            patch("lapis_pm.episodic.write_observation"),
        ):
            out2 = ab.sweep(mem, now="2026-09-07T00:00:00Z")
        second_repo = out2["repo"]
        assert first_repo != second_repo

    def test_sweep_uses_main_baseline_not_attest(self):
        # The sweep must run the dedicated main-baseline path (one detached
        # worktree at origin/main), NOT attest() with a fake target_id
        # (attest(target_id="__backstop__") would build a nonexistent branch
        # "lapis/__backstop__/main" and always return inconclusive).
        mem = _make_mem()
        key = "pm/autonomy-class/lapis-pm/clean/code/lt100"
        mem.set(key, json.dumps({"promoted": True, "auto_resolved_count": 0,
                                 "regression_count": 0}),
                tags=["lapis-pm", "pm:autonomy-class"])

        def _boom(*args, **kwargs):
            raise AssertionError(
                "sweep must not call va.attest (it was called with "
                f"args={args}, kwargs={kwargs})"
            )

        with (
            patch.object(va, "attest", side_effect=_boom),
            patch.object(va, "attest_main_baseline",
                         return_value=_att("attested", pr_failures=[])),
            patch("lapis_pm.episodic.write_observation"),
        ):
            out = ab.sweep(mem, now="2026-09-06T00:00:00Z")

        # The sweep completed with the stubbed main-baseline result.
        assert out["repo"] == "lapis-pm"
        assert out["status"] == "green"


# ---------------------------------------------------------------------------
# U3.2 - open_regression_brief: HIGH, "suspect, not proven"
# ---------------------------------------------------------------------------

class TestRegressionBrief:

    def test_brief_says_suspect_not_proven(self):
        with patch("agents_core.notify.send_notification") as mock_notify:
            ab.open_regression_brief("lapis-pm", 42,
                                     ["lapis_pm/foo.py::test_x"])
        assert mock_notify.called
        msg = mock_notify.call_args.kwargs.get("message", "")
        assert "suspect" in msg.lower() or "not proven" in msg.lower()
        assert "lapis-pm" in msg
        assert "42" in msg


# ---------------------------------------------------------------------------
# U3.3 - autonomy-promote: demotion-cooldown guard
# ---------------------------------------------------------------------------

class TestAutonomyPromote:

    def _class_key(self):
        return pc.class_record_key("lapis-pm", "clean", ["code"], "lt100")

    def test_promote_refused_within_14d_regression(self):
        # Demotion-cooldown guard: regression_count>0 and last_regression_ts
        # within 14 days -> refused without --force.
        mem = _make_mem()
        key = self._class_key()
        # last_regression_ts within 14 days of now (2026-09-05 is ~1 day ago).
        mem.set(key, json.dumps({"promoted": False, "auto_resolved_count": 0,
                                 "regression_count": 1,
                                 "last_regression_ts": "2026-09-05T00:00:00Z"}),
                tags=["lapis-pm", "pm:autonomy-class"])

        res = ab.promote(mem, repo="lapis-pm", verdict_class="clean",
                         change_classes=["code"], loc_bucket="lt100",
                         force=False, justification=None)
        assert res["promoted"] is False
        assert res["reason"] == "regression_cooldown"
        rec = json.loads(mem._data[key]["content"])
        assert rec["promoted"] is False  # unchanged

    def test_promote_with_force_after_2_demotions_requires_justification(self):
        # regression_count > 2 within 14d: even --force requires --justification.
        mem = _make_mem()
        key = self._class_key()
        mem.set(key, json.dumps({"promoted": False, "auto_resolved_count": 0,
                                 "regression_count": 3,
                                 "last_regression_ts": "2026-09-05T00:00:00Z"}),
                tags=["lapis-pm", "pm:autonomy-class"])

        # --force WITHOUT justification -> refused.
        res = ab.promote(mem, repo="lapis-pm", verdict_class="clean",
                         change_classes=["code"], loc_bucket="lt100",
                         force=True, justification=None)
        assert res["promoted"] is False
        assert res["reason"] == "force_requires_justification"

        # --force WITH justification -> promoted, recorded as Erah-force.
        res = ab.promote(mem, repo="lapis-pm", verdict_class="clean",
                         change_classes=["code"], loc_bucket="lt100",
                         force=True, justification="manual review done")
        assert res["promoted"] is True
        rec = json.loads(mem._data[key]["content"])
        assert rec["promoted"] is True
        assert rec["promoted_by"] == "Erah-force"
        assert rec["force_justification"] == "manual review done"

    def test_promote_clean_class(self):
        mem = _make_mem()
        key = self._class_key()
        mem.set(key, json.dumps({"promoted": False, "auto_resolved_count": 0,
                                 "regression_count": 0}),
                tags=["lapis-pm", "pm:autonomy-class"])

        res = ab.promote(mem, repo="lapis-pm", verdict_class="clean",
                         change_classes=["code"], loc_bucket="lt100",
                         force=False, justification=None)
        assert res["promoted"] is True
        rec = json.loads(mem._data[key]["content"])
        assert rec["promoted"] is True
        assert rec["promoted_by"] == "Erah"


# ---------------------------------------------------------------------------
# AC5 - tick-path never-raise, fail-soft to logging
# ---------------------------------------------------------------------------

class TestTickPathNeverRaise:

    def test_tick_sweep_hook_never_raises(self):
        from lapis_pm import pm_core

        with (
            patch.object(pm_core, "_mem", side_effect=RuntimeError("mem down")),
            patch.object(pm_core, "_autonomy_inflight", return_value=False),
            patch.object(pm_core, "_autonomy_op_daily_gated", return_value=True),
        ):
            # The tick-path hook must not propagate the exception.
            pm_core._backstop_sweep_tick()
