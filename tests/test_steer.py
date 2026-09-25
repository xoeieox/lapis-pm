"""Tests for lapis_pm/steer.py — typed steer channel.

Covers:
  - file_steer round-trip and invalid type rejection
  - consume_pending_steers moves to applied/ with applied_ts
  - peek_emergencies is non-destructive
  - mark_applied is idempotent
  - set/get/consume_directive_overlay lifecycle
  - inject_overlay: fixer consumes, reviewer does not, fixer with no overlay → ""
  - Directive supersede audit: pm:steer:directive:superseded observation written
  - Template-render guard (B1 regression): fixer and fixer_retry system_templates
    render without KeyError with and without overlay
  - Encode integration: steer encode path in tick()
  - Emergency decide 4.0a: pauses target, marks file applied, correct TickResult
  - Emergency durability: file stays pending until after save()
  - Co-pending directive overlay survives emergency pause
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_steer_dir(tmp_path: Path) -> Path:
    d = tmp_path / "steers"
    d.mkdir()
    return d


def _patch_steers_dir(tmp_path: Path):
    """Patch _steers_dir() to use tmp_path/steers."""
    steers = tmp_path / "steers"
    steers.mkdir(exist_ok=True)
    return patch("lapis_pm.steer._steers_dir", return_value=steers)


# ---------------------------------------------------------------------------
# file_steer — basic round-trip
# ---------------------------------------------------------------------------

class TestFileSteer:
    def test_round_trip_context(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            path = steer.file_steer("my-target-v0", "context", "FYI CI config changed")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["type"] == "context"
        assert data["message"] == "FYI CI config changed"
        assert data["target_id"] == "my-target-v0"
        assert data["filed_by"] == "pm"
        assert "ts" in data

    def test_round_trip_directive(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            path = steer.file_steer("my-target-v0", "directive", "skip the CLI shim")
        data = json.loads(path.read_text())
        assert data["type"] == "directive"

    def test_round_trip_emergency(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            path = steer.file_steer("my-target-v0", "emergency", "stop now")
        data = json.loads(path.read_text())
        assert data["type"] == "emergency"

    def test_invalid_type_raises_value_error(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            with pytest.raises(ValueError, match="Invalid steer type"):
                steer.file_steer("my-target-v0", "bogus", "nope")

    def test_filename_contains_target_id_and_type(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            path = steer.file_steer("tid-abc", "directive", "msg")
        assert "tid-abc" in path.name
        assert "directive" in path.name


# ---------------------------------------------------------------------------
# consume_pending_steers
# ---------------------------------------------------------------------------

class TestConsumePendingSteer:
    def test_consume_moves_to_applied(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            steer.file_steer("t1", "context", "hello")
            consumed = steer.consume_pending_steers("t1", types=("context",))
        assert len(consumed) == 1
        assert consumed[0]["type"] == "context"
        assert consumed[0]["message"] == "hello"
        applied = list((tmp_path / "steers" / "applied").glob("t1__*.json"))
        assert len(applied) == 1
        data = json.loads(applied[0].read_text())
        assert "applied_ts" in data
        assert data["disposition"] == "consumed"
        pending = list((tmp_path / "steers").glob("t1__*.json"))
        assert len(pending) == 0

    def test_consume_excludes_emergency_by_default(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            steer.file_steer("t2", "emergency", "stop")
            steer.file_steer("t2", "context", "info")
            consumed = steer.consume_pending_steers("t2")
        assert len(consumed) == 1
        assert consumed[0]["type"] == "context"
        pending = list((tmp_path / "steers").glob("t2__*.json"))
        # emergency still pending
        assert len(pending) == 1
        assert "emergency" in pending[0].name

    def test_consume_filters_by_target_id(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            steer.file_steer("target-a", "directive", "msg-a")
            steer.file_steer("target-b", "directive", "msg-b")
            consumed = steer.consume_pending_steers("target-a", types=("directive",))
        assert len(consumed) == 1
        assert consumed[0]["target_id"] == "target-a"
        # target-b stays pending
        pending = list((tmp_path / "steers").glob("target-b__*.json"))
        assert len(pending) == 1

    def test_consume_returns_empty_when_no_pending(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            consumed = steer.consume_pending_steers("no-such-target")
        assert consumed == []


# ---------------------------------------------------------------------------
# peek_emergencies
# ---------------------------------------------------------------------------

class TestPeekEmergencies:
    def test_peek_is_nondestructive(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            steer.file_steer("t3", "emergency", "halt")
            result = steer.peek_emergencies("t3")
        assert len(result) == 1
        assert result[0]["type"] == "emergency"
        assert result[0]["message"] == "halt"
        assert "_path" in result[0]
        # file must still be in pending dir
        pending = list((tmp_path / "steers").glob("t3__*.json"))
        assert len(pending) == 1

    def test_peek_ignores_non_emergency(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            steer.file_steer("t4", "context", "info")
            steer.file_steer("t4", "directive", "redo")
            result = steer.peek_emergencies("t4")
        assert result == []

    def test_peek_returns_empty_when_no_dir(self, tmp_path):
        from lapis_pm import steer
        empty = tmp_path / "steers-empty"
        with patch("lapis_pm.steer._steers_dir", return_value=empty):
            result = steer.peek_emergencies("noop")
        assert result == []


# ---------------------------------------------------------------------------
# mark_applied
# ---------------------------------------------------------------------------

class TestMarkApplied:
    def test_mark_applied_moves_file(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            path = steer.file_steer("t5", "emergency", "stop")
            steer.mark_applied(path, disposition="emergency_paused")
        assert not path.exists()
        applied = list((tmp_path / "steers" / "applied").glob("t5__*.json"))
        assert len(applied) == 1
        data = json.loads(applied[0].read_text())
        assert data["disposition"] == "emergency_paused"
        assert "applied_ts" in data

    def test_mark_applied_is_idempotent(self, tmp_path):
        with _patch_steers_dir(tmp_path):
            from lapis_pm import steer
            path = steer.file_steer("t6", "context", "x")
            steer.mark_applied(path)
            steer.mark_applied(path)  # second call must not raise


# ---------------------------------------------------------------------------
# Directive overlay lifecycle
# ---------------------------------------------------------------------------

class TestDirectiveOverlay:
    def _mem_mock(self):
        store = {}

        def _set(key, value, tags=None):
            store[key] = value
            return True

        def _get(key):
            if key in store:
                return {"content": store[key]}
            return None

        def _delete(key):
            store.pop(key, None)
            return True

        m = MagicMock()
        m.set.side_effect = _set
        m.get.side_effect = _get
        m.delete.side_effect = _delete
        return m

    def test_set_get_consume_lifecycle(self):
        mem = self._mem_mock()
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            from lapis_pm import steer
            steer.set_directive_overlay("t7", "focus on API surface")
            assert steer.get_directive_overlay("t7") == "focus on API surface"
            consumed = steer.consume_directive_overlay("t7")
            assert consumed == "focus on API surface"
            # second consume → None
            assert steer.consume_directive_overlay("t7") is None

    def test_second_get_after_consume_returns_none(self):
        mem = self._mem_mock()
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            from lapis_pm import steer
            steer.set_directive_overlay("t8", "msg")
            steer.consume_directive_overlay("t8")
            assert steer.get_directive_overlay("t8") is None

    def test_supersede_audit_observation(self):
        mem = self._mem_mock()
        obs_calls = []
        from lapis_pm import steer
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation",
                   side_effect=lambda *a, **kw: obs_calls.append((a, kw))):
            steer.set_directive_overlay("t9", "first directive")
            # second filing while first is unconsumed
            steer.set_directive_overlay("t9", "second directive")
            # overlay holds the second directive (check inside the patch context)
            second_overlay = steer.get_directive_overlay("t9")
        # superseded observation must have been written for the first
        supersede_calls = [
            (a, kw) for a, kw in obs_calls
            if "pm:steer:directive:superseded" in kw.get("extra_tags", [])
        ]
        assert len(supersede_calls) == 1
        assert "first directive" in supersede_calls[0][0][1]
        assert second_overlay == "second directive"


# ---------------------------------------------------------------------------
# inject_overlay
# ---------------------------------------------------------------------------

class TestInjectOverlay:
    def _mem_mock(self, value=None):
        m = MagicMock()
        m.get.return_value = {"content": value} if value else None
        m.delete.return_value = True
        return m

    def test_fixer_with_overlay_sets_block_and_consumes(self):
        mem = self._mem_mock("skip the CLI shim")
        obs_calls = []
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation",
                   side_effect=lambda *a, **kw: obs_calls.append((a, kw))):
            from lapis_pm import steer
            vars_ = {}
            steer.inject_overlay("tid", vars_, "fixer")
        assert "steer_directive_block" in vars_
        assert "skip the CLI shim" in vars_["steer_directive_block"]
        assert "Active PM directive" in vars_["steer_directive_block"]
        mem.delete.assert_called_once()
        consumed_calls = [
            (a, kw) for a, kw in obs_calls
            if "pm:steer:directive:consumed" in kw.get("extra_tags", [])
        ]
        assert len(consumed_calls) == 1

    def test_fixer_retry_with_overlay_sets_block_and_consumes(self):
        mem = self._mem_mock("reprioritize: focus on X")
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            from lapis_pm import steer
            vars_ = {}
            steer.inject_overlay("tid", vars_, "fixer_retry")
        assert "reprioritize: focus on X" in vars_["steer_directive_block"]
        mem.delete.assert_called_once()

    def test_fixer_staged_with_overlay_sets_block_and_consumes(self):
        """lapis-pm-fixers-harness-registry-v0 Leg 2: the fixer_staged agent_type
        consumes the single-shot directive overlay (the mission fence rides the
        steer channel into {steer_directive_block})."""
        mem = self._mem_mock("```mission\npre_aimed: true\n```")
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            from lapis_pm import steer
            vars_ = {}
            steer.inject_overlay("tid", vars_, "fixer_staged")
        assert "pre_aimed: true" in vars_["steer_directive_block"]
        assert "Active PM directive" in vars_["steer_directive_block"]
        mem.delete.assert_called_once()

    def test_fixer_flash_with_overlay_sets_block_and_consumes(self):
        """flashnext-fixer-trial-v0 (leg 1): the fixer_flash trial tier
        consumes the single-shot directive overlay - a mid-run PM directive
        filed against a trial target must reach the trial dispatch rather
        than silently persist past it."""
        mem = self._mem_mock("trial directive: focus on the pin")
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            from lapis_pm import steer
            vars_ = {}
            steer.inject_overlay("tid", vars_, "fixer_flash")
        assert "trial directive: focus on the pin" in vars_["steer_directive_block"]
        assert "Active PM directive" in vars_["steer_directive_block"]
        mem.delete.assert_called_once()

    def test_fixer_flash_without_overlay_sets_empty_string(self):
        mem = self._mem_mock(None)
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            from lapis_pm import steer
            vars_ = {}
            steer.inject_overlay("tid", vars_, "fixer_flash")
        assert vars_["steer_directive_block"] == ""
        mem.delete.assert_not_called()

    def test_fixer_staged_without_overlay_sets_empty_string(self):
        mem = self._mem_mock(None)
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            from lapis_pm import steer
            vars_ = {}
            steer.inject_overlay("tid", vars_, "fixer_staged")
        assert vars_["steer_directive_block"] == ""
        mem.delete.assert_not_called()

    def test_fixer_without_overlay_sets_empty_string(self):
        mem = self._mem_mock(None)
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            from lapis_pm import steer
            vars_ = {}
            steer.inject_overlay("tid", vars_, "fixer")
        assert vars_["steer_directive_block"] == ""
        mem.delete.assert_not_called()

    def test_reviewer_with_overlay_sets_empty_and_does_not_consume(self):
        mem = self._mem_mock("some directive")
        from lapis_pm import steer
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            vars_ = {}
            steer.inject_overlay("tid", vars_, "reviewer")
            # overlay still readable (non-destructive)
            still_present = steer.get_directive_overlay("tid")
        assert vars_["steer_directive_block"] == ""
        mem.delete.assert_not_called()
        assert still_present == "some directive"

    def test_reviewer_fresh_with_overlay_does_not_consume(self):
        mem = self._mem_mock("directive")
        with patch("lapis_pm.steer._mem", return_value=mem), \
             patch("lapis_pm.episodic.write_observation"):
            from lapis_pm import steer
            vars_ = {}
            steer.inject_overlay("tid", vars_, "reviewer_fresh")
        assert vars_["steer_directive_block"] == ""
        mem.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Template-render guard (B1 regression)
# ---------------------------------------------------------------------------

REGISTRY_PATH = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"


def _load_template(agent_type: str) -> str:
    data = yaml.safe_load(REGISTRY_PATH.read_text())
    return data["agents"][agent_type]["system_template"]


class TestTemplateRenderGuard:
    """Regression guard: fixer and fixer_retry templates must not KeyError
    when rendered via str.format(**vars_) with vars_ produced by inject_overlay,
    both with and without an active overlay."""

    def _base_vars(self, overlay_text: str | None) -> dict:
        """Build a vars_ that mirrors what pm_core builds before inject_overlay."""
        from lapis_pm import steer
        mem_mock = MagicMock()
        mem_mock.get.return_value = {"content": overlay_text} if overlay_text else None
        mem_mock.delete.return_value = True
        vars_ = {
            "target_id": "test-target-v0",
            "spec_summary": "a spec",
            "repo": "lapis-pm",
            "question": "implement X",
            "pr_number": "",
            "slug": "forced",
            "existing_branch": "lapis/test-target-v0/forced",
            "base_branch": "main",
            "intent_block": "## Intent\nDo the thing.",
        }
        with patch("lapis_pm.steer._mem", return_value=mem_mock), \
             patch("lapis_pm.episodic.write_observation"):
            steer.inject_overlay("test-target-v0", vars_, "fixer")
        return vars_

    def test_fixer_template_renders_without_overlay(self):
        tpl = _load_template("fixer")
        vars_ = self._base_vars(None)
        rendered = tpl.format(**vars_)
        assert "test-target-v0" in rendered

    def test_fixer_template_renders_with_overlay(self):
        tpl = _load_template("fixer")
        vars_ = self._base_vars("focus on API surface")
        rendered = tpl.format(**vars_)
        assert "focus on API surface" in rendered

    def test_fixer_retry_template_renders_without_overlay(self):
        tpl = _load_template("fixer_retry")
        # inject_overlay for fixer_retry is the same key; reuse vars
        vars_ = self._base_vars(None)
        rendered = tpl.format(**vars_)
        assert "test-target-v0" in rendered

    def test_fixer_retry_template_renders_with_overlay(self):
        tpl = _load_template("fixer_retry")
        vars_ = self._base_vars("skip the CLI shim")
        rendered = tpl.format(**vars_)
        assert "skip the CLI shim" in rendered

    def test_fixer_template_no_key_error_missing_overlay(self):
        """Explicit: ensure KeyError is not raised when steer_directive_block is ""."""
        tpl = _load_template("fixer")
        vars_ = self._base_vars(None)
        assert vars_["steer_directive_block"] == ""
        try:
            tpl.format(**vars_)
        except KeyError as e:
            pytest.fail(f"fixer template raised KeyError: {e}")

    def test_fixer_retry_template_no_key_error_missing_overlay(self):
        tpl = _load_template("fixer_retry")
        vars_ = self._base_vars(None)
        try:
            tpl.format(**vars_)
        except KeyError as e:
            pytest.fail(f"fixer_retry template raised KeyError: {e}")

    def test_fixer_staged_template_renders_with_overlay(self):
        """lapis-pm-fixers-harness-registry-v0 Leg 2: the fixer_staged
        system_template carries {steer_directive_block} and renders the
        mission fence without KeyError."""
        tpl = _load_template("fixer_staged")
        assert "{steer_directive_block}" in tpl
        vars_ = self._base_vars("```mission\npre_aimed: true\n```")
        rendered = tpl.format(**vars_)
        assert "pre_aimed: true" in rendered
        assert "test-target-v0" in rendered

    def test_fixer_staged_template_no_key_error_missing_overlay(self):
        tpl = _load_template("fixer_staged")
        vars_ = self._base_vars(None)
        try:
            tpl.format(**vars_)
        except KeyError as e:
            pytest.fail(f"fixer_staged template raised KeyError: {e}")


# ---------------------------------------------------------------------------
# CLI — invalid type returns non-zero
# ---------------------------------------------------------------------------

class TestCLISteer:
    def test_invalid_type_exits_nonzero(self, tmp_path):
        from lapis_pm.cli import cmd_steer
        args = MagicMock()
        args.target_id = "t"
        args.type = "bogus"
        args.message = "x"
        ret = cmd_steer(args)
        assert ret != 0

    def test_valid_type_writes_file(self, tmp_path):
        from lapis_pm.cli import cmd_steer
        args = MagicMock()
        args.target_id = "cli-target"
        args.type = "context"
        args.message = "hello from cli"
        with patch("lapis_pm.steer._steers_dir", return_value=tmp_path / "steers"):
            ret = cmd_steer(args)
        assert ret == 0


# ---------------------------------------------------------------------------
# Encode integration: context/directive steers encoded in tick(), emergency not
# ---------------------------------------------------------------------------

def _mock_target_for_steer(paused: bool = False) -> MagicMock:
    t = MagicMock()
    t.pm_bound = True
    t.paused = paused
    t.paused_reason = None
    t.pm_repo = ""
    t.pm_authority = "advisory"
    t.data = {}
    return t


def _tick_with_steer_patches(target, steer_patches: dict | None = None):
    """Run tick() with minimal patches plus steer-specific overrides."""
    from contextlib import ExitStack
    from lapis_pm import pm_core

    base = {
        "lapis_pm.pm_core.TargetStore": MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=target))),
        "lapis_pm.pm_core.get_pause_state": MagicMock(return_value="active"),
        "lapis_pm.pm_core.set_pause_state": MagicMock(),
        "lapis_pm.pm_core._reconcile_dispatched_with_queue": MagicMock(return_value=0),
        "lapis_pm.pm_core.get_cursor": MagicMock(return_value=None),
        "lapis_pm.episodic.since": MagicMock(return_value=[]),
        "lapis_pm.pm_core._perceive_prs": MagicMock(return_value=([], True)),
        "lapis_pm.pm_core._encode_user_comments": MagicMock(return_value=[]),
        "lapis_pm.pm_core._seen_pr_ids": MagicMock(return_value=set()),
        "lapis_pm.pm_core._encode_new_prs": MagicMock(return_value=[]),
        "lapis_pm.pm_core._encode_pr_sha_updates": MagicMock(return_value=0),
        "lapis_pm.pm_core._encode_pr_body_updates": MagicMock(return_value=0),
        "lapis_pm.pm_core._encode_gpu_results": MagicMock(return_value=(0, [])),
        "lapis_pm.pm_core._encode_merged_prs": MagicMock(return_value=0),
        "lapis_pm.pm_core._consume_brief_decisions": MagicMock(return_value=None),
        "lapis_pm.pm_core._is_auto_land_eligible": MagicMock(return_value=False),
        "lapis_pm.pm_core._persist_review_state_cache": MagicMock(),
        "lapis_pm.pm_core.set_cursor": MagicMock(),
        "lapis_pm.pm_core.episodic.write_observation": MagicMock(),
        "lapis_pm.pm_core._find_lost_fixer_dispatches": MagicMock(return_value=([], [])),
        "lapis_pm.pm_core.load_dispatched": MagicMock(return_value=[]),
    }
    merged = {**base, **(steer_patches or {})}
    with ExitStack() as stack:
        for k, v in merged.items():
            stack.enter_context(patch(k, v))
        return pm_core.tick("my-target")


class TestEncodeIntegration:
    def test_context_steer_increments_encoded(self, tmp_path):
        """A context steer is consumed during encode and increments encoded count."""
        target = _mock_target_for_steer()
        context_steer = {
            "type": "context",
            "message": "CI config changed",
            "ts": "2026-06-30T10:00:00+00:00",
            "_path": tmp_path / "fake.json",
        }
        with patch("lapis_pm.steer.consume_pending_steers", return_value=[context_steer]), \
             patch("lapis_pm.steer.peek_emergencies", return_value=[]):
            result = _tick_with_steer_patches(target)
        assert result.encoded >= 1

    def test_directive_steer_registers_overlay(self, tmp_path):
        """A directive steer calls set_directive_overlay during encode."""
        target = _mock_target_for_steer()
        directive_steer = {
            "type": "directive",
            "message": "focus on API surface",
            "ts": "2026-06-30T10:00:00+00:00",
            "_path": tmp_path / "fake.json",
        }
        overlay_calls = []
        with patch("lapis_pm.steer.consume_pending_steers", return_value=[directive_steer]), \
             patch("lapis_pm.steer.peek_emergencies", return_value=[]), \
             patch("lapis_pm.steer.set_directive_overlay",
                   side_effect=lambda tid, msg: overlay_calls.append((tid, msg))), \
             patch("lapis_pm.steer.inject_overlay"):
            _tick_with_steer_patches(target)
        assert len(overlay_calls) == 1
        assert overlay_calls[0] == ("my-target", "focus on API surface")

    def test_emergency_not_consumed_during_encode(self, tmp_path):
        """An emergency steer file stays pending after the encode phase
        (peek_emergencies is the appropriate path, not consume_pending_steers)."""
        target = _mock_target_for_steer()
        consume_calls = []

        def fake_consume(tid, types=("context", "directive")):
            consume_calls.append(types)
            return []

        with patch("lapis_pm.steer.consume_pending_steers", side_effect=fake_consume), \
             patch("lapis_pm.steer.peek_emergencies", return_value=[]), \
             patch("lapis_pm.steer.inject_overlay"):
            _tick_with_steer_patches(target)

        # emergency must NOT appear in the types passed to consume_pending_steers
        for types in consume_calls:
            assert "emergency" not in types


# ---------------------------------------------------------------------------
# Emergency decide integration
# ---------------------------------------------------------------------------

class TestEmergencyDecide:
    def test_emergency_pauses_target_and_marks_applied(self, tmp_path):
        """Emergency steer pauses target, marks file applied, returns correct decision."""
        target = _mock_target_for_steer()
        applied_calls = []
        emergency = {
            "type": "emergency",
            "message": "security hole detected",
            "ts": "2026-06-30T10:00:00+00:00",
            "_path": tmp_path / "t-e.json",
        }
        with patch("lapis_pm.steer.consume_pending_steers", return_value=[]), \
             patch("lapis_pm.steer.peek_emergencies", return_value=[emergency]), \
             patch("lapis_pm.steer.mark_applied",
                   side_effect=lambda p, disposition="applied": applied_calls.append((p, disposition))), \
             patch("lapis_pm.steer.inject_overlay"):
            result = _tick_with_steer_patches(target, {
                "lapis_pm.pm_core.set_pause_state": MagicMock(),
            })

        assert result.decision.startswith("action:steer_emergency_applied:")
        assert "security hole detected" in result.decision or \
               "2026-06-30T10:00:00" in result.decision
        # mark_applied must have been called with emergency_paused
        assert any(d == "emergency_paused" for _, d in applied_calls)
        # target paused
        target.set_paused.assert_called_once_with(
            True, reason="steer:emergency: security hole detected"[:110]
        )
        target.save.assert_called_once()

    def test_emergency_save_before_mark_applied_ordering(self, tmp_path):
        """save() must be called before mark_applied() (durability ordering)."""
        target = _mock_target_for_steer()
        call_order = []
        emergency = {
            "type": "emergency",
            "message": "halt",
            "ts": "2026-06-30T10:00:00+00:00",
            "_path": tmp_path / "e.json",
        }
        target.save.side_effect = lambda: call_order.append("save")

        with patch("lapis_pm.steer.consume_pending_steers", return_value=[]), \
             patch("lapis_pm.steer.peek_emergencies", return_value=[emergency]), \
             patch("lapis_pm.steer.mark_applied",
                   side_effect=lambda *a, **kw: call_order.append("mark_applied")), \
             patch("lapis_pm.steer.inject_overlay"):
            _tick_with_steer_patches(target)

        assert call_order.index("save") < call_order.index("mark_applied")

    def test_co_pending_directive_overlay_survives_emergency_pause(self, tmp_path):
        """A directive overlay registered in the same tick survives an emergency pause.
        inject_overlay must NOT be called for emergency path (no fixer dispatch)."""
        target = _mock_target_for_steer()
        inject_calls = []
        emergency = {
            "type": "emergency",
            "message": "halt",
            "ts": "2026-06-30T10:00:00+00:00",
            "_path": tmp_path / "e2.json",
        }
        # Directive steer is encoded, then emergency halts in decide
        directive_steer = {
            "type": "directive",
            "message": "focus on API",
            "ts": "2026-06-30T10:00:00+00:00",
            "_path": tmp_path / "d.json",
        }
        overlay_store = {}

        def fake_set_overlay(tid, msg):
            overlay_store[tid] = msg

        with patch("lapis_pm.steer.consume_pending_steers", return_value=[directive_steer]), \
             patch("lapis_pm.steer.peek_emergencies", return_value=[emergency]), \
             patch("lapis_pm.steer.set_directive_overlay", side_effect=fake_set_overlay), \
             patch("lapis_pm.steer.mark_applied"), \
             patch("lapis_pm.steer.inject_overlay",
                   side_effect=lambda *a, **kw: inject_calls.append(a)):
            _tick_with_steer_patches(target)

        # directive overlay was registered
        assert overlay_store.get("my-target") == "focus on API"
        # inject_overlay was NOT called for the emergency path (no dispatch happened)
        # The emergency branch returns early before any _SHAPER.dispatch call
        # so inject_overlay calls (if any) come only from _SHAPER.dispatch wrappers.
        # In this minimal test no dispatch occurs, so inject_overlay should not be called.
        assert len(inject_calls) == 0

    def test_emergency_durability_re_applies_if_file_stays_pending(self, tmp_path):
        """Simulate crash after save() but before mark_applied: next tick re-pauses."""
        target = _mock_target_for_steer()
        target.paused = False  # not yet paused on second tick
        emergency = {
            "type": "emergency",
            "message": "halt",
            "ts": "2026-06-30T10:00:00+00:00",
            "_path": tmp_path / "e3.json",
        }
        applied = []

        with patch("lapis_pm.steer.consume_pending_steers", return_value=[]), \
             patch("lapis_pm.steer.peek_emergencies", return_value=[emergency]), \
             patch("lapis_pm.steer.mark_applied",
                   side_effect=lambda p, disposition="applied": applied.append(1)), \
             patch("lapis_pm.steer.inject_overlay"):
            result = _tick_with_steer_patches(target)

        assert result.decision.startswith("action:steer_emergency_applied:")
        # target paused again (idempotent)
        target.set_paused.assert_called()
        # mark_applied completed this time
        assert applied
