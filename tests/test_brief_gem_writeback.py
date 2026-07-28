"""Tests for lapis-pm-brief-gem-writeback-v0.

Coverage:
  1. test_brief_raise_deposits_gem: raising a brief deposits one gem with mapped options;
     re-raise of same brief is idempotent; weaver-unreachable fails soft (brief still set,
     warning logged).
  2. test_reconciler_executes_decided_gem: decided gem with status=open -> apply_decision
     called with right args; gem marked actioned; second tick does not re-apply.
  3. test_reconciler_held_path_blocks_automerge: merge_pr gem whose fresh diff touches
     a held path is not executed; gem stays open + annotated; brief intact; HIGH notify
     + observation fire.
  4. test_reconciler_held_path_fetch_failure_fails_closed: diff fetch error -> reconciler
     fails closed (no merge), same manual-required path.
  5. test_reconciler_hold_authority_blocks: pm_authority="hold" target -> not auto-executed.
  6. test_reconciler_supersession: outstanding brief cleared (resolved via skill) -> gem
     marked superseded; apply_decision NOT called.
  7. test_reconciler_permanent_error_supersedes: apply_decision returns ok=False,
     error="unknown_option_id" -> gem superseded, not retried.
  8. test_non_merge_action_executes: decided force_dispatch_retry gem executes directly
     (no held-path gate needed).

Patch targets:
  - lapis_pm.pm_core._mem       -> MemoryStore factory (returns mock mem)
  - lapis_pm.pm_core.get_outstanding_brief -> outstanding brief lookup
  - lapis_pm.brief_gem._brief.read_options / apply_decision  (module-level _brief alias)
  - lapis_pm.brief_gem._fetch_decided_gems
  - lapis_pm.brief_gem._held_path_check_live
  - lapis_pm.brief_gem.send_notification  (module-level import)
  - lapis_pm.brief_gem.episodic.write_observation  (module-level episodic alias)
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import brief_gem


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_brief(
    target_id: str = "my-target",
    comment_id: str = "brief-cid-001",
    body: str = (
        "## State\nPR is open on lapis-pm #42.\n"
        "## Recent activity\n- Fixer dispatched.\n"
        "## Decision needed\nMerge or retry the PR?\n"
        "## Options\n- A: Merge\n- B: Retry"
    ),
    synthesis_failed: bool = False,
    pushed: bool = True,
) -> MagicMock:
    b = MagicMock()
    b.target_id = target_id
    b.comment_id = comment_id
    b.body = body
    b.synthesis_failed = synthesis_failed
    b.pushed = pushed
    return b


def _make_mock_mem(
    *,
    forward: dict | None = None,
    reverse: str | None = None,
) -> MagicMock:
    """Return a mock MemoryStore with configurable get() side-effect."""
    mem = MagicMock()
    set_calls: list[tuple] = []

    def _get(key: str):
        if reverse is not None and key.startswith("pm/brief-gem/by-brief/"):
            return {"content": reverse}
        if forward is not None and key.startswith("pm/brief-gem/map/"):
            return {"content": json.dumps(forward)}
        return None

    def _set(key, value, **kwargs):
        set_calls.append((key, value))

    mem.get.side_effect = _get
    mem.set.side_effect = _set
    mem._set_calls = set_calls
    return mem


def _minimal_forward(
    target_id: str = "my-target",
    brief_comment_id: str = "brief-cid-001",
    status: str = "open",
    pr_number: int | None = 42,
) -> dict:
    return {
        "target_id": target_id,
        "brief_comment_id": brief_comment_id,
        "pr_number": pr_number,
        "deposited_ts": "2026-06-25T10:00:00+00:00",
        "actioned_ts": None,
        "status": status,
        "annotation": None,
    }


def _make_advisory_opts(brief_id: str = "brief-cid-001", pr: int = 42) -> dict:
    return {
        "brief_id": brief_id,
        "trigger": "advisory-clean",
        "options": [
            {"id": "A", "label": "Merge the PR", "action": {"kind": "merge_pr", "pr": pr}},
            {"id": "B", "label": "Acknowledge and clear",
             "action": {"kind": "acknowledge_and_clear"}},
        ],
    }


# ---------------------------------------------------------------------------
# 1. test_brief_raise_deposits_gem
# ---------------------------------------------------------------------------

class TestBriefRaiseDepositsGem:

    def test_deposit_called_once_on_brief_raise(self):
        """Raising a brief deposits one gem and stores forward + reverse mapping keys."""
        b = _make_brief()
        opts = _make_advisory_opts(b.comment_id)
        mock_mem = _make_mock_mem()  # no existing gem

        fake_resp = MagicMock()
        fake_resp.json.return_value = {"gem_id": "gem-uuid-001"}
        fake_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        deposited: list[dict] = []

        def _capture_post(url, json=None, **kwargs):
            deposited.append(json or {})
            return fake_resp

        mock_client.post.side_effect = _capture_post

        with (
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("httpx.Client", return_value=mock_client),
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            gem_id = brief_gem.deposit_brief_gem("my-target", b)

        assert gem_id == "gem-uuid-001"
        assert len(deposited) == 1
        payload = deposited[0]
        assert payload["origin"] == "PM brief · my-target"
        assert payload["deposited_by"] == "lapis-pm:brief"
        assert payload["source_thread_id"] == "my-target"
        assert payload["agent"] == "PM"
        # Options mapped: key=id, title=label, sub=""
        assert len(payload["options"]) == 2
        assert payload["options"][0]["key"] == "A"
        assert payload["options"][0]["title"] == "Merge the PR"
        assert payload["options"][0]["sub"] == ""
        assert payload["options"][1]["key"] == "B"
        # PR number extracted from merge_pr action
        assert payload["why"] == "Brief for target my-target PR #42"

        # Forward and reverse keys written to mem
        fwd_keys = [k for k, _ in mock_mem._set_calls if k.startswith("pm/brief-gem/map/")]
        rev_keys = [k for k, _ in mock_mem._set_calls if k.startswith("pm/brief-gem/by-brief/")]
        assert len(fwd_keys) == 1
        assert len(rev_keys) == 1
        assert rev_keys[0] == "pm/brief-gem/by-brief/my-target/brief-cid-001"

        # Forward record has correct shape
        fwd_rec = json.loads(next(v for k, v in mock_mem._set_calls if k.startswith("pm/brief-gem/map/")))
        assert fwd_rec["target_id"] == "my-target"
        assert fwd_rec["brief_comment_id"] == "brief-cid-001"
        assert fwd_rec["pr_number"] == 42
        assert fwd_rec["status"] == "open"

    def test_re_raise_same_brief_is_idempotent(self):
        """Re-raising the same brief does not create a duplicate gem."""
        b = _make_brief()
        # Reverse index already has a gem_id
        mock_mem = _make_mock_mem(reverse="existing-gem-id")

        with (
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("httpx.Client") as mock_httpx,
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            gem_id = brief_gem.deposit_brief_gem("my-target", b)

        assert gem_id == "existing-gem-id"
        mock_httpx.assert_not_called()  # no HTTP call made

    def test_weaver_unreachable_fails_soft_returns_none(self):
        """Weaver unreachable: returns None, no mapping stored."""
        import httpx as _httpx
        b = _make_brief()
        mock_mem = _make_mock_mem()  # empty

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = _httpx.ConnectError("weaver down")

        with (
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=None),
            patch("httpx.Client", return_value=mock_client),
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            gem_id = brief_gem.deposit_brief_gem("my-target", b)

        assert gem_id is None
        # No mapping stored on failure
        assert mock_mem._set_calls == []

    def test_deposit_called_from_set_brief_outstanding_success_path(self):
        """_set_brief_outstanding calls deposit_brief_gem when brief is set (normal path)."""
        from lapis_pm import pm_core
        b = _make_brief(synthesis_failed=False)
        deposit_calls: list = []

        with (
            patch("lapis_pm.pm_core.set_outstanding_brief"),
            # brief_gem.deposit_brief_gem is the canonical function location
            patch("lapis_pm.brief_gem.deposit_brief_gem",
                  side_effect=lambda tid, br: deposit_calls.append((tid, br.comment_id)) or "g"),
        ):
            result = pm_core._set_brief_outstanding("my-target", b, verified=False)

        assert result is True
        assert deposit_calls == [("my-target", b.comment_id)]

    def test_deposit_error_in_set_brief_outstanding_is_swallowed(self):
        """A deposit error inside _set_brief_outstanding is non-fatal; brief still set."""
        from lapis_pm import pm_core
        b = _make_brief(synthesis_failed=False)

        with (
            patch("lapis_pm.pm_core.set_outstanding_brief"),
            patch("lapis_pm.brief_gem.deposit_brief_gem",
                  side_effect=RuntimeError("weaver gone")),
        ):
            result = pm_core._set_brief_outstanding("my-target", b, verified=False)

        assert result is True  # brief still set; deposit error is non-fatal


# ---------------------------------------------------------------------------
# 2. test_reconciler_executes_decided_gem
# ---------------------------------------------------------------------------

class TestReconcilerExecutesDecidedGem:

    def test_decided_gem_calls_apply_decision_and_marks_actioned(self):
        """Decided gem with status=open -> apply_decision called; gem actioned."""
        fwd = _minimal_forward(pr_number=None)
        mock_mem = _make_mock_mem(forward=fwd)

        opts = {
            "brief_id": "brief-cid-001",
            "options": [
                {"id": "B", "label": "Acknowledge and clear",
                 "action": {"kind": "acknowledge_and_clear"}},
            ],
        }
        apply_result = {
            "ok": True,
            "action_kind": "acknowledge_and_clear",
            "detail": "acknowledged and cleared",
        }

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-1", "state": "decided",
                                 "decision_json": {"option_key": "B"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision",
                  return_value=apply_result) as mock_apply,
            patch("agents_core.targets.TargetStore"),
        ):
            actions = brief_gem.reconcile_decided_gems()

        mock_apply.assert_called_once_with("my-target", "brief-cid-001", "B")
        assert any("actioned" in a and "acknowledge_and_clear" in a for a in actions)

        # Forward map updated to "actioned"
        updated = next(
            (json.loads(v) for k, v in mock_mem._set_calls if "pm/brief-gem/map/" in k),
            None,
        )
        assert updated is not None
        assert updated["status"] == "actioned"
        assert updated["actioned_ts"] is not None

    def test_second_tick_skips_actioned_gem(self):
        """A gem already marked actioned is skipped on the next tick."""
        fwd = _minimal_forward(status="actioned")
        mock_mem = _make_mock_mem(forward=fwd)

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-1", "state": "decided",
                                 "decision_json": {"option_key": "B"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.brief_gem._brief.apply_decision") as mock_apply,
        ):
            actions = brief_gem.reconcile_decided_gems()

        mock_apply.assert_not_called()
        assert actions == []


# ---------------------------------------------------------------------------
# 3. test_reconciler_held_path_blocks_automerge
# ---------------------------------------------------------------------------

class TestReconcilerHeldPathBlocksAutomerge:

    def test_held_path_blocks_merge_notifies_and_leaves_open(self):
        """merge_pr gem whose live diff touches a held path: not executed, notified, open."""
        fwd = _minimal_forward(pr_number=42)
        mock_mem = _make_mock_mem(forward=fwd)

        opts = _make_advisory_opts(pr=42)
        notify_calls: list = []
        obs_calls: list = []

        mock_target = MagicMock()
        mock_target.pm_repo = "lapis-pm"
        mock_target.pm_authority = "advisory"

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-merge", "state": "decided",
                                 "decision_json": {"option_key": "A"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision") as mock_apply,
            patch("agents_core.targets.TargetStore") as MockStore,
            patch("lapis_pm.brief_gem._held_path_check_live",
                  return_value=(True, "held path(s): lapis_pm/SPEC.md")),
            patch("lapis_pm.brief_gem.send_notification",
                  side_effect=lambda msg, **kw: notify_calls.append((msg, kw)) or True),
            patch("lapis_pm.brief_gem.episodic.write_observation",
                  side_effect=lambda tid, msg, **kw: obs_calls.append((tid, msg))),
        ):
            MockStore.return_value.get.return_value = mock_target
            actions = brief_gem.reconcile_decided_gems()

        # apply_decision NOT called
        mock_apply.assert_not_called()
        # Action reflects manual-required held-path
        assert any("manual-required" in a and "held-path" in a for a in actions)
        # HIGH-priority notify sent
        assert len(notify_calls) == 1
        notify_msg, notify_kw = notify_calls[0]
        assert "manual merge" in notify_msg.lower()
        assert notify_kw.get("priority") == brief_gem._NotifyPriority.HIGH
        # Observation written with correct tag
        assert len(obs_calls) == 1
        assert "manual merge required" in obs_calls[0][1].lower()
        # Gem status stays "open" with annotation
        updated = next(
            (json.loads(v) for k, v in mock_mem._set_calls if "pm/brief-gem/map/" in k),
            None,
        )
        assert updated is not None
        assert updated["status"] == "open"
        assert "held" in (updated.get("annotation") or "").lower()

    def test_brief_not_cleared_on_held_block(self):
        """Outstanding brief NOT cleared when held-path blocks execution."""
        fwd = _minimal_forward(pr_number=42)
        mock_mem = _make_mock_mem(forward=fwd)
        opts = _make_advisory_opts(pr=42)
        mock_target = MagicMock()
        mock_target.pm_repo = "lapis-pm"
        mock_target.pm_authority = "advisory"

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-merge", "state": "decided",
                                 "decision_json": {"option_key": "A"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision") as mock_apply,
            patch("agents_core.targets.TargetStore") as MockStore,
            patch("lapis_pm.brief_gem._held_path_check_live",
                  return_value=(True, "held path(s): systemd/lapis-pm.service")),
            patch("lapis_pm.brief_gem.send_notification"),
            patch("lapis_pm.brief_gem.episodic.write_observation"),
        ):
            MockStore.return_value.get.return_value = mock_target
            brief_gem.reconcile_decided_gems()

        # apply_decision not called = brief not cleared
        mock_apply.assert_not_called()


# ---------------------------------------------------------------------------
# 4. test_reconciler_held_path_fetch_failure_fails_closed
# ---------------------------------------------------------------------------

class TestReconcilerHeldPathFetchFailure:

    def test_diff_fetch_error_fails_closed(self):
        """Diff fetch error (branch gone/network) -> reconciler fails closed, no merge."""
        fwd = _minimal_forward(pr_number=42)
        mock_mem = _make_mock_mem(forward=fwd)
        opts = _make_advisory_opts(pr=42)
        mock_target = MagicMock()
        mock_target.pm_repo = "lapis-pm"
        mock_target.pm_authority = "advisory"

        notify_calls: list = []

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-merge", "state": "decided",
                                 "decision_json": {"option_key": "A"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision") as mock_apply,
            patch("agents_core.targets.TargetStore") as MockStore,
            # _held_path_check_live catches errors internally and returns blocked=True
            patch("lapis_pm.brief_gem._held_path_check_live",
                  return_value=(True, "diff fetch error: branch gone")),
            patch("lapis_pm.brief_gem.send_notification",
                  side_effect=lambda msg, **kw: notify_calls.append(msg) or True),
            patch("lapis_pm.brief_gem.episodic.write_observation"),
        ):
            MockStore.return_value.get.return_value = mock_target
            actions = brief_gem.reconcile_decided_gems()

        mock_apply.assert_not_called()
        assert any("manual-required" in a for a in actions)
        assert len(notify_calls) == 1

    def test_held_path_check_live_returns_blocked_on_exception(self):
        """_held_path_check_live: exception from get_pr_diff -> (True, error-reason)."""
        with (
            patch("lapis_pm.brief_gem.authority.changed_paths", return_value=[]),
            patch("lapis_pm.brief_gem.authority.is_held_path", return_value=False),
            patch("agents_core.forgejo.get_pr", return_value={"number": 42}),
            patch("agents_core.forgejo.get_pr_diff",
                  side_effect=ConnectionError("network gone")),
        ):
            blocked, reason = brief_gem._held_path_check_live("my-target", 42, "lapis-pm")

        assert blocked is True
        assert "diff fetch error" in reason.lower() or "network" in reason.lower()

    def test_held_path_check_live_returns_blocked_when_pr_not_found(self):
        """_held_path_check_live: PR not found in Forgejo -> (True, reason)."""
        with patch("agents_core.forgejo.get_pr", return_value=None):
            blocked, reason = brief_gem._held_path_check_live("my-target", 42, "lapis-pm")

        assert blocked is True
        assert "not found" in reason.lower()


# ---------------------------------------------------------------------------
# 5. test_reconciler_hold_authority_blocks
# ---------------------------------------------------------------------------

class TestReconcilerHoldAuthorityBlocks:

    def test_hold_authority_target_not_auto_executed(self):
        """Target with pm_authority=hold -> manual required, HIGH notify, no execution."""
        fwd = _minimal_forward(pr_number=42)
        mock_mem = _make_mock_mem(forward=fwd)
        opts = _make_advisory_opts(pr=42)
        mock_target = MagicMock()
        mock_target.pm_repo = "lapis-pm"
        mock_target.pm_authority = "hold"

        notify_calls: list[tuple] = []

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-hold", "state": "decided",
                                 "decision_json": {"option_key": "A"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision") as mock_apply,
            patch("agents_core.targets.TargetStore") as MockStore,
            patch("lapis_pm.brief_gem._held_path_check_live") as mock_held,
            patch("lapis_pm.brief_gem.send_notification",
                  side_effect=lambda msg, **kw: notify_calls.append((msg, kw)) or True),
            patch("lapis_pm.brief_gem.episodic.write_observation"),
        ):
            MockStore.return_value.get.return_value = mock_target
            actions = brief_gem.reconcile_decided_gems()

        mock_apply.assert_not_called()
        # Hold-authority check fires before held-path check
        mock_held.assert_not_called()
        assert any("hold-authority" in a for a in actions)
        assert len(notify_calls) == 1
        _, kw = notify_calls[0]
        assert kw.get("priority") == brief_gem._NotifyPriority.HIGH


# ---------------------------------------------------------------------------
# 6. test_reconciler_supersession
# ---------------------------------------------------------------------------

class TestReconcilerSupersession:

    def test_brief_already_cleared_marks_gem_superseded(self):
        """Brief cleared (e.g. via pm-pr-review) before reconciler runs -> gem superseded."""
        fwd = _minimal_forward(status="open")
        mock_mem = _make_mock_mem(forward=fwd)

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-1", "state": "decided",
                                 "decision_json": {"option_key": "A"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            # Outstanding brief is now None (already cleared)
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.brief_gem._brief.apply_decision") as mock_apply,
        ):
            actions = brief_gem.reconcile_decided_gems()

        mock_apply.assert_not_called()
        assert any("superseded" in a for a in actions)

    def test_new_brief_replaces_old_marks_gem_superseded(self):
        """Newer brief outstanding -> old gem superseded, apply_decision NOT called."""
        fwd = _minimal_forward(brief_comment_id="old-brief-cid", status="open")
        mock_mem = _make_mock_mem(forward=fwd)

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-1", "state": "decided",
                                 "decision_json": {"option_key": "A"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            # A different (newer) brief is now outstanding
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="new-brief-cid"),
            patch("lapis_pm.brief_gem._brief.apply_decision") as mock_apply,
        ):
            actions = brief_gem.reconcile_decided_gems()

        mock_apply.assert_not_called()
        assert any("superseded" in a for a in actions)
        updated = next(
            (json.loads(v) for k, v in mock_mem._set_calls if "pm/brief-gem/map/" in k),
            None,
        )
        assert updated is not None
        assert updated["status"] == "superseded"


# ---------------------------------------------------------------------------
# 7. test_reconciler_permanent_error_supersedes
# ---------------------------------------------------------------------------

class TestReconcilerPermanentError:

    def test_unknown_option_id_supersedes_gem(self):
        """apply_decision ok=False, error='unknown_option_id:Z' -> gem superseded."""
        fwd = _minimal_forward(pr_number=None)
        mock_mem = _make_mock_mem(forward=fwd)

        opts = {
            "brief_id": "brief-cid-001",
            "options": [
                {"id": "A", "label": "Ack", "action": {"kind": "acknowledge_and_clear"}},
            ],
        }
        apply_result = {"ok": False, "error": "unknown_option_id:Z"}

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-1", "state": "decided",
                                 "decision_json": {"option_key": "Z"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision", return_value=apply_result),
            patch("agents_core.targets.TargetStore"),
        ):
            actions = brief_gem.reconcile_decided_gems()

        assert any("superseded:permanent-error" in a for a in actions)
        assert not any("retry" in a for a in actions)

    def test_options_not_found_supersedes_gem(self):
        """apply_decision ok=False, error='options_not_found' -> gem superseded."""
        fwd = _minimal_forward(pr_number=None)
        mock_mem = _make_mock_mem(forward=fwd)

        opts = {
            "brief_id": "brief-cid-001",
            "options": [{"id": "B", "label": "Retry",
                          "action": {"kind": "force_dispatch_retry"}}],
        }
        apply_result = {"ok": False, "error": "options_not_found"}

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-1", "state": "decided",
                                 "decision_json": {"option_key": "B"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision", return_value=apply_result),
            patch("agents_core.targets.TargetStore"),
        ):
            actions = brief_gem.reconcile_decided_gems()

        assert any("superseded:permanent-error" in a for a in actions)

    def test_transient_error_leaves_gem_open_for_retry(self):
        """apply_decision ok=False, error='action_failed:...' -> gem stays open (transient)."""
        fwd = _minimal_forward(pr_number=None)
        mock_mem = _make_mock_mem(forward=fwd)

        opts = {
            "brief_id": "brief-cid-001",
            "options": [{"id": "B", "label": "Retry",
                          "action": {"kind": "force_dispatch_retry"}}],
        }
        apply_result = {
            "ok": False,
            "error": "action_failed:force_dispatch_retry:network error",
        }

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-1", "state": "decided",
                                 "decision_json": {"option_key": "B"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision", return_value=apply_result),
            patch("agents_core.targets.TargetStore"),
        ):
            actions = brief_gem.reconcile_decided_gems()

        assert any("retry" in a for a in actions)
        assert not any("superseded" in a for a in actions)


# ---------------------------------------------------------------------------
# 8. test_non_merge_action_executes
# ---------------------------------------------------------------------------

class TestNonMergeActionExecutes:

    def test_force_dispatch_retry_executes_without_held_path_gate(self):
        """force_dispatch_retry decided gem executes directly — no held-path check."""
        fwd = _minimal_forward(pr_number=None)
        mock_mem = _make_mock_mem(forward=fwd)

        opts = {
            "brief_id": "brief-cid-001",
            "options": [
                {"id": "B", "label": "Retry fixer",
                 "action": {"kind": "force_dispatch_retry"}},
            ],
        }
        apply_result = {
            "ok": True,
            "action_kind": "force_dispatch_retry",
            "detail": "force_dispatch fixer task_id=task-123",
        }

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-retry", "state": "decided",
                                 "decision_json": {"option_key": "B"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision",
                  return_value=apply_result) as mock_apply,
            patch("lapis_pm.brief_gem._held_path_check_live") as mock_held,
            patch("agents_core.targets.TargetStore") as MockStore,
        ):
            mock_target = MagicMock()
            mock_target.pm_repo = "lapis-pm"
            mock_target.pm_authority = "advisory"
            MockStore.return_value.get.return_value = mock_target
            actions = brief_gem.reconcile_decided_gems()

        mock_apply.assert_called_once_with("my-target", "brief-cid-001", "B")
        mock_held.assert_not_called()  # no held-path check for non-merge actions
        assert any("actioned" in a and "force_dispatch_retry" in a for a in actions)

    def test_acknowledge_and_clear_executes_without_held_path_gate(self):
        """acknowledge_and_clear executes without held-path check."""
        fwd = _minimal_forward(pr_number=None)
        mock_mem = _make_mock_mem(forward=fwd)

        opts = {
            "brief_id": "brief-cid-001",
            "options": [
                {"id": "A", "label": "Ack", "action": {"kind": "acknowledge_and_clear"}},
            ],
        }
        apply_result = {
            "ok": True,
            "action_kind": "acknowledge_and_clear",
            "detail": "acknowledged and cleared",
        }

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems",
                  return_value=[{"gem_id": "gem-ack", "state": "decided",
                                 "decision_json": {"option_key": "A"}}]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-cid-001"),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("lapis_pm.brief_gem._brief.apply_decision",
                  return_value=apply_result) as mock_apply,
            patch("lapis_pm.brief_gem._held_path_check_live") as mock_held,
            patch("agents_core.targets.TargetStore") as MockStore,
        ):
            mock_target = MagicMock()
            mock_target.pm_repo = "lapis-pm"
            mock_target.pm_authority = "advisory"
            MockStore.return_value.get.return_value = mock_target
            actions = brief_gem.reconcile_decided_gems()

        mock_apply.assert_called_once_with("my-target", "brief-cid-001", "A")
        mock_held.assert_not_called()
        assert any("actioned" in a for a in actions)


# ---------------------------------------------------------------------------
# Default-options synthesis (lapis-pm-hold-brief-optionless-gem-fix-v0, Fix 1 + 3)
# ---------------------------------------------------------------------------

class TestDefaultOptionsSynthesis:

    def _deposit(self, b, opts, mock_mem):
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"gem_id": "gem-synth-001"}
        fake_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        deposited: list[dict] = []

        def _capture_post(url, json=None, **kwargs):
            deposited.append(json or {})
            return fake_resp

        mock_client.post.side_effect = _capture_post

        with (
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=opts),
            patch("httpx.Client", return_value=mock_client),
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            gem_id = brief_gem.deposit_brief_gem(b.target_id, b)

        assert gem_id == "gem-synth-001"
        return deposited[0]

    def test_no_options_comment_deposits_ack_only(self):
        """No pm:brief-options comment, no PR mention -> ack-only, non-empty options."""
        b = _make_brief(body=(
            "## State\nSomething needs a call.\n"
            "## Decision needed\nRetry or escalate?\n"
        ))
        mock_mem = _make_mock_mem()

        payload = self._deposit(b, None, mock_mem)

        assert payload["options"] == [{"key": "ack", "title": "Acknowledge", "sub": ""}]
        assert payload["why"] == "Brief for target my-target"

    def test_empty_options_list_also_synthesizes_ack(self):
        """read_options returns a shape with an empty options list -> same as None case."""
        b = _make_brief(body=(
            "## State\nSomething needs a call.\n"
            "## Decision needed\nRetry or escalate?\n"
        ))
        mock_mem = _make_mock_mem()
        opts = {"brief_id": b.comment_id, "options": []}

        payload = self._deposit(b, opts, mock_mem)

        assert payload["options"] == [{"key": "ack", "title": "Acknowledge", "sub": ""}]

    def test_pr_number_in_title_synthesizes_merge_option_and_why(self):
        """Title/body names 'PR #N' with no options comment -> merge option + why names it."""
        b = _make_brief(body=(
            "## State\nPR #11 is held after static checks passed.\n"
            "## Decision needed\nMerge PR #11?\n"
        ))
        mock_mem = _make_mock_mem()

        payload = self._deposit(b, None, mock_mem)

        assert payload["options"] == [
            {"key": "merge_pr", "title": "Merge PR #11", "sub": ""},
            {"key": "ack", "title": "Acknowledge", "sub": ""},
        ]
        assert payload["why"] == "Brief for target my-target PR #11"

        fwd_rec = json.loads(next(
            v for k, v in mock_mem._set_calls if k.startswith("pm/brief-gem/map/")
        ))
        assert fwd_rec["pr_number"] == 11

    def test_real_options_present_skips_synthesis(self):
        """When read_options returns real options, no synthesis happens (unchanged behavior)."""
        b = _make_brief()
        opts = _make_advisory_opts(b.comment_id)
        mock_mem = _make_mock_mem()

        payload = self._deposit(b, opts, mock_mem)

        keys = [o["key"] for o in payload["options"]]
        assert "ack" not in keys
        assert keys == ["A", "B"]


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestExtractPrNumber:

    def test_extracts_pr_number(self):
        assert brief_gem._extract_pr_number("Merge PR #11?") == 11

    def test_no_match_returns_none(self):
        assert brief_gem._extract_pr_number("Retry or escalate?") is None

    def test_case_insensitive_and_spacing(self):
        assert brief_gem._extract_pr_number("please merge pr # 7 now") == 7


class TestParseBriefTitleAndAsk:

    def test_parses_state_and_decision_needed(self):
        body = (
            "## State\nPR is open on lapis-pm #42.\n"
            "## Recent activity\n- Fixer dispatched.\n"
            "## Decision needed\nMerge or retry the PR?\n"
            "## Options\n- A: Merge\n- B: Retry"
        )
        title, ask = brief_gem._parse_brief_title_and_ask(body)
        assert "PR is open" in title
        assert "Merge or retry" in ask

    def test_defaults_on_missing_sections(self):
        title, ask = brief_gem._parse_brief_title_and_ask("No structured sections here")
        assert title == "PM brief"
        assert ask == "Awaiting decision"

    def test_none_decision_becomes_awaiting(self):
        body = "## State\nAll good.\n## Decision needed\nnone\n"
        title, ask = brief_gem._parse_brief_title_and_ask(body)
        assert ask == "Awaiting decision"

    def test_multiline_state_uses_first_line_only(self):
        body = "## State\nFirst line.\nSecond line.\n## Decision needed\nAct now.\n"
        title, ask = brief_gem._parse_brief_title_and_ask(body)
        assert title == "First line."
        assert ask == "Act now."


class TestWeaverUrlResolution:

    def test_env_var_override(self):
        import os
        with patch.dict(os.environ, {"WEAVER_BASE_URL": "http://custom:9999"}):
            assert brief_gem._weaver_base_url() == "http://custom:9999"

    def test_bind_port_override(self):
        import os
        with patch.dict(os.environ, {"WEAVER_BIND_PORT": "4567"},
                        clear=False):
            # Ensure WEAVER_BASE_URL is not set
            env = {k: v for k, v in os.environ.items() if k != "WEAVER_BASE_URL"}
            env["WEAVER_BIND_PORT"] = "4567"
            with patch.dict(os.environ, env, clear=True):
                assert brief_gem._weaver_base_url() == "http://127.0.0.1:4567"

    def test_brix_default(self):
        import os
        env = {k: v for k, v in os.environ.items()
               if k not in ("WEAVER_BASE_URL", "WEAVER_BIND_PORT")}
        with patch.dict(os.environ, env, clear=True):
            url = brief_gem._weaver_base_url()
        assert "203.0.113.10:8403" in url


# ---------------------------------------------------------------------------
# Test-isolation guard (lapis-pm-brief-gem-test-isolation-v0)
# ---------------------------------------------------------------------------

class TestPytestGuard:
    """deposit_brief_gem and _fetch_decided_gems must not reach live weaver under pytest."""

    def test_deposit_brief_gem_noop_under_pytest(self):
        """With PYTEST_CURRENT_TEST set (always true) and no WEAVER_BASE_URL, deposit is a no-op."""
        b = _make_brief()
        env = {k: v for k, v in os.environ.items() if k != "WEAVER_BASE_URL"}
        env["PYTEST_CURRENT_TEST"] = "test_deposit_brief_gem_noop_under_pytest"
        with (
            patch.dict(os.environ, env, clear=True),
            patch("httpx.Client") as mock_httpx,
        ):
            result = brief_gem.deposit_brief_gem("test-target", b)
        assert result is None
        mock_httpx.assert_not_called()

    def test_deposit_brief_gem_runs_when_weaver_base_url_set(self):
        """With WEAVER_BASE_URL set, deposit bypasses the guard and hits the mock weaver."""
        b = _make_brief()
        mock_mem = _make_mock_mem()

        fake_resp = MagicMock()
        fake_resp.json.return_value = {"gem_id": "gem-intent-001"}
        fake_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = fake_resp

        with (
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.brief_gem._brief.read_options", return_value=None),
            patch("httpx.Client", return_value=mock_client),
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://test-weaver:9999"}),
        ):
            result = brief_gem.deposit_brief_gem("test-target", b)
        assert result == "gem-intent-001"
        mock_client.post.assert_called_once()

    def test_set_brief_outstanding_no_live_deposit(self):
        """_set_brief_outstanding under pytest deposits no gem to live weaver (original leak)."""
        from lapis_pm import pm_core

        b = _make_brief(synthesis_failed=False)
        mock_mem = _make_mock_mem()
        env = {k: v for k, v in os.environ.items() if k != "WEAVER_BASE_URL"}
        env["PYTEST_CURRENT_TEST"] = "test_set_brief_outstanding_no_live_deposit"
        with (
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch.dict(os.environ, env, clear=True),
            patch("httpx.Client") as mock_httpx,
        ):
            pm_core._set_brief_outstanding("my-target", b)
        mock_httpx.assert_not_called()

    def test_call_supersede_endpoint_noop_under_pytest(self):
        """_call_supersede_endpoint with no WEAVER_BASE_URL is a no-op under pytest."""
        env = {k: v for k, v in os.environ.items() if k != "WEAVER_BASE_URL"}
        env["PYTEST_CURRENT_TEST"] = "test_call_supersede_endpoint_noop_under_pytest"
        with (
            patch.dict(os.environ, env, clear=True),
            patch("httpx.Client") as mock_httpx,
        ):
            result = brief_gem._call_supersede_endpoint("gem-x", "test reason", "test-by")
        assert result is None
        mock_httpx.assert_not_called()


# ---------------------------------------------------------------------------
# Orphan open-gem sweep tests
# ---------------------------------------------------------------------------

def _make_map_entry(gem_id: str, target_id: str, brief_comment_id: str,
                    status: str = "open") -> dict:
    content = json.dumps({
        "target_id": target_id,
        "brief_comment_id": brief_comment_id,
        "pr_number": None,
        "deposited_ts": "2026-06-25T10:00:00+00:00",
        "actioned_ts": None,
        "status": status,
        "annotation": None,
    })
    return {"key": f"pm/brief-gem/map/{gem_id}", "content": content}


class TestReconcileOrphanSweep:
    """test_reconcile_supersedes_orphan_open_gem: orphan detection and supersession."""

    def _make_mem_with_entries(self, entries: list[dict]) -> MagicMock:
        """Mock mem that returns entries from list_by_prefix and supports get/set."""
        mem = MagicMock()
        set_calls: list[tuple] = []

        def _get(key: str):
            for e in entries:
                if e["key"] == key:
                    return {"content": e["content"]}
            return None

        def _list_by_prefix(prefix: str, limit: int = 50):
            return [e for e in entries if e["key"].startswith(prefix)]

        def _set(key, value, **kwargs):
            set_calls.append((key, value))

        mem.get.side_effect = _get
        mem.list_by_prefix.side_effect = _list_by_prefix
        mem.set.side_effect = _set
        mem._set_calls = set_calls
        return mem

    def test_orphan_gem_superseded_when_brief_cleared(self):
        """Open gem whose brief is no longer outstanding → supersede called, mapping superseded."""
        entries = [_make_map_entry("gem-orphan", "target-a", "brief-old-cid")]
        mock_mem = self._make_mem_with_entries(entries)

        supersede_calls: list[tuple] = []

        def _fake_supersede(gem_id, reason, by):
            supersede_calls.append((gem_id, reason, by))
            return True

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems", return_value=[]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.brief_gem._call_supersede_endpoint",
                  side_effect=_fake_supersede),
        ):
            actions = brief_gem.reconcile_decided_gems()

        assert len(supersede_calls) == 1
        assert supersede_calls[0][0] == "gem-orphan"
        assert any("orphan-superseded" in a for a in actions)

        # Mapping updated to superseded
        updated = next(
            (json.loads(v) for k, v in mock_mem._set_calls if "pm/brief-gem/map/" in k),
            None,
        )
        assert updated is not None
        assert updated["status"] == "superseded"
        assert "orphan" in (updated.get("annotation") or "").lower()

    def test_gem_with_brief_still_outstanding_untouched(self):
        """Open gem whose brief is still outstanding is NOT superseded."""
        entries = [_make_map_entry("gem-live", "target-b", "brief-live-cid")]
        mock_mem = self._make_mem_with_entries(entries)

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems", return_value=[]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-live-cid"),
            patch("lapis_pm.brief_gem._call_supersede_endpoint") as mock_sup,
        ):
            actions = brief_gem.reconcile_decided_gems()

        mock_sup.assert_not_called()
        assert not any("orphan" in a for a in actions)
        assert mock_mem._set_calls == []

    def test_changed_brief_triggers_orphan_supersede(self):
        """Brief replaced by a newer one → old gem is superseded."""
        entries = [_make_map_entry("gem-stale", "target-c", "brief-old")]
        mock_mem = self._make_mem_with_entries(entries)

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems", return_value=[]),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            # A different brief is now outstanding
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-new"),
            patch("lapis_pm.brief_gem._call_supersede_endpoint", return_value=True),
        ):
            actions = brief_gem.reconcile_decided_gems()

        assert any("orphan-superseded" in a for a in actions)


class TestReconcileOrphanIsSilent:
    """test_reconcile_orphan_is_silent: no Pushover / notify on orphan supersede."""

    def test_no_notification_on_orphan_supersede(self):
        """Orphan supersede fires no send_notification call (silent auto-clear)."""
        entries = [_make_map_entry("gem-silent", "target-d", "brief-gone")]
        mem = MagicMock()
        mem.list_by_prefix.return_value = entries
        mem.get.return_value = None
        mem.set.return_value = None

        notify_calls: list = []

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems", return_value=[]),
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.brief_gem._call_supersede_endpoint", return_value=True),
            patch("lapis_pm.brief_gem.send_notification",
                  side_effect=lambda *a, **kw: notify_calls.append(a)),
        ):
            brief_gem.reconcile_decided_gems()

        assert notify_calls == []


class TestReconcileOrphanIdempotent:
    """test_reconcile_orphan_idempotent: superseded mapping is not re-processed."""

    def test_superseded_mapping_skipped(self):
        """A mapping with status='superseded' is not passed to _call_supersede_endpoint."""
        entries = [_make_map_entry("gem-done", "target-e", "brief-x", status="superseded")]
        mem = MagicMock()
        mem.list_by_prefix.return_value = entries
        mem.get.return_value = None

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems", return_value=[]),
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.brief_gem._call_supersede_endpoint") as mock_sup,
        ):
            actions = brief_gem.reconcile_decided_gems()

        mock_sup.assert_not_called()
        assert not any("orphan" in a for a in actions)

    def test_actioned_mapping_skipped(self):
        """A mapping with status='actioned' is not passed to _call_supersede_endpoint."""
        entries = [_make_map_entry("gem-done2", "target-f", "brief-y", status="actioned")]
        mem = MagicMock()
        mem.list_by_prefix.return_value = entries
        mem.get.return_value = None

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems", return_value=[]),
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.brief_gem._call_supersede_endpoint") as mock_sup,
        ):
            actions = brief_gem.reconcile_decided_gems()

        mock_sup.assert_not_called()

    def test_409_leaves_mapping_open(self):
        """409 from supersede endpoint leaves mapping status=open (decided-gem path heals it)."""
        entries = [_make_map_entry("gem-conflict", "target-g", "brief-decided")]
        mem = MagicMock()
        mem.list_by_prefix.return_value = entries
        mem.get.return_value = None
        set_calls: list = []
        mem.set.side_effect = lambda k, v, **kw: set_calls.append((k, v))

        with (
            patch("lapis_pm.brief_gem._fetch_decided_gems", return_value=[]),
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.brief_gem._call_supersede_endpoint", return_value=False),
        ):
            actions = brief_gem.reconcile_decided_gems()

        # No map update on 409
        assert set_calls == []
        # No orphan-superseded action
        assert not any("orphan-superseded" in a for a in actions)
