"""Tests for lapis_pm.chain — validation, event emission, state, chain advance.

Coverage:
  Schema:
  - validate_legs: valid chain passes
  - validate_legs: duplicate tid raises ValueError
  - validate_legs: cyclic dep raises ValueError
  - validate_legs: unknown dep in depends_on raises ValueError

  Event emission:
  - emit_chain_event writes mem key chain/<group>/event/<ts> with correct fields

  State snapshot:
  - update_chain_state writes chain/<group>/state with legs + complete=False
  - complete flips True when all legs are "landed"
  - get_chain_state returns dict or None

  Chain advance:
  - check_chain_advance fires initial_dispatch for leg whose only dep just landed
  - check_chain_advance does NOT fire when not all deps are satisfied
  - check_chain_advance is idempotent (no double-fire when dispatch pending)
  - on_leg_landed updates state snapshot for the landed leg

  CLI:
  - bind parser accepts hold as --authority choice
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.chain import (
    validate_legs,
    emit_chain_event,
    update_chain_state,
    get_chain_state,
    check_chain_advance,
    on_leg_landed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mem_store():
    """Return an in-memory MemoryStore backed by a temp file."""
    from agents_core.mem import MemoryStore
    tmp = tempfile.mktemp(suffix=".db")
    return MemoryStore(db_path=Path(tmp))


# ---------------------------------------------------------------------------
# validate_legs
# ---------------------------------------------------------------------------

class TestValidateLegs:
    def test_valid_single_leg(self):
        legs = [{"tid": "a", "repo": "r", "authority": "advisory", "intent": "do it",
                 "branch_slug": "implement"}]
        validate_legs(legs)  # should not raise

    def test_valid_chain_linear(self):
        legs = [
            {"tid": "a", "depends_on": [], "branch_slug": "step-a"},
            {"tid": "b", "depends_on": ["a"], "branch_slug": "step-b"},
            {"tid": "c", "depends_on": ["b"], "branch_slug": "step-c"},
        ]
        validate_legs(legs)

    def test_valid_parallel_rejoin(self):
        # legs b and c both depend on a; d depends on b and c
        legs = [
            {"tid": "a", "depends_on": [], "branch_slug": "a"},
            {"tid": "b", "depends_on": ["a"], "branch_slug": "b"},
            {"tid": "c", "depends_on": ["a"], "branch_slug": "c"},
            {"tid": "d", "depends_on": ["b", "c"], "branch_slug": "d"},
        ]
        validate_legs(legs)

    # --- branch_slug validation ---

    def test_branch_slug_missing_raises(self):
        legs = [{"tid": "a", "repo": "r", "authority": "advisory", "intent": "x"}]
        with pytest.raises(ValueError, match="branch_slug"):
            validate_legs(legs)

    def test_branch_slug_empty_raises(self):
        legs = [{"tid": "a", "branch_slug": ""}]
        with pytest.raises(ValueError, match="branch_slug"):
            validate_legs(legs)

    def test_branch_slug_with_slash_raises(self):
        legs = [{"tid": "a", "branch_slug": "foo/bar"}]
        with pytest.raises(ValueError, match="slash"):
            validate_legs(legs)

    def test_branch_slug_with_lapis_prefix_raises(self):
        legs = [{"tid": "a", "branch_slug": "lapis/implement"}]
        with pytest.raises(ValueError, match="lapis/"):
            validate_legs(legs)

    def test_branch_slug_leading_dash_raises(self):
        legs = [{"tid": "a", "branch_slug": "-foo"}]
        with pytest.raises(ValueError, match="invalid"):
            validate_legs(legs)

    def test_branch_slug_trailing_dash_raises(self):
        legs = [{"tid": "a", "branch_slug": "foo-"}]
        with pytest.raises(ValueError, match="invalid"):
            validate_legs(legs)

    def test_branch_slug_single_char_valid(self):
        legs = [{"tid": "a", "branch_slug": "v"}]
        validate_legs(legs)  # single char is valid

    def test_branch_slug_with_numbers_valid(self):
        legs = [{"tid": "a", "branch_slug": "impl-v2"}]
        validate_legs(legs)

    def test_branch_slug_uppercase_raises(self):
        legs = [{"tid": "a", "branch_slug": "Implement"}]
        with pytest.raises(ValueError, match="invalid"):
            validate_legs(legs)

    def test_duplicate_tid(self):
        legs = [{"tid": "a", "branch_slug": "s"}, {"tid": "a", "branch_slug": "s"}]
        with pytest.raises(ValueError, match="Duplicate tid"):
            validate_legs(legs)

    def test_cyclic_direct(self):
        legs = [
            {"tid": "a", "depends_on": ["b"], "branch_slug": "s"},
            {"tid": "b", "depends_on": ["a"], "branch_slug": "s"},
        ]
        with pytest.raises(ValueError, match="Cyclic"):
            validate_legs(legs)

    def test_cyclic_indirect(self):
        legs = [
            {"tid": "a", "depends_on": ["c"], "branch_slug": "s"},
            {"tid": "b", "depends_on": ["a"], "branch_slug": "s"},
            {"tid": "c", "depends_on": ["b"], "branch_slug": "s"},
        ]
        with pytest.raises(ValueError, match="Cyclic"):
            validate_legs(legs)

    def test_unknown_dep(self):
        legs = [{"tid": "a", "depends_on": ["nonexistent"], "branch_slug": "s"}]
        with pytest.raises(ValueError, match="unknown tid"):
            validate_legs(legs)

    def test_empty_legs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_legs([])


# ---------------------------------------------------------------------------
# emit_chain_event + get_chain_state
# ---------------------------------------------------------------------------

class TestChainEventAndState:
    def test_emit_chain_event_writes_mem_key(self):
        mem = _make_mem_store()
        with patch("lapis_pm.chain._mem", return_value=mem):
            emit_chain_event("my-group", "bind", "leg-a", details={"legs": ["leg-a"]})

        # Key should be chain/my-group/event/<ts>
        results = mem.list_all(tag="chain-group:my-group", limit=50)
        event_rows = [r for r in results if "/event/" in r.get("key", "")]
        assert len(event_rows) == 1
        data = json.loads(event_rows[0]["content"])
        assert data["kind"] == "bind"
        assert data["tid"] == "leg-a"
        assert data["group_id"] == "my-group"
        assert data["details"] == {"legs": ["leg-a"]}

    def test_update_chain_state_not_complete(self):
        mem = _make_mem_store()
        with patch("lapis_pm.chain._mem", return_value=mem):
            legs = [
                {"tid": "a", "status": "dispatched"},
                {"tid": "b", "status": "pending"},
            ]
            state = update_chain_state("grp", legs)

        assert state["complete"] is False
        assert state["group_id"] == "grp"
        assert len(state["legs"]) == 2

    def test_update_chain_state_complete(self):
        mem = _make_mem_store()
        with patch("lapis_pm.chain._mem", return_value=mem):
            legs = [
                {"tid": "a", "status": "landed"},
                {"tid": "b", "status": "landed"},
            ]
            state = update_chain_state("grp2", legs)

        assert state["complete"] is True

    def test_get_chain_state_returns_none_missing(self):
        mem = _make_mem_store()
        with patch("lapis_pm.chain._mem", return_value=mem):
            result = get_chain_state("no-such-group")
        assert result is None

    def test_get_chain_state_roundtrip(self):
        mem = _make_mem_store()
        with patch("lapis_pm.chain._mem", return_value=mem):
            legs = [{"tid": "x", "status": "pending"}]
            update_chain_state("grp3", legs)
            result = get_chain_state("grp3")
        assert result is not None
        assert result["group_id"] == "grp3"
        assert result["legs"][0]["tid"] == "x"


# ---------------------------------------------------------------------------
# on_leg_landed
# ---------------------------------------------------------------------------

class TestOnLegLanded:
    def test_updates_leg_status(self):
        mem = _make_mem_store()
        with patch("lapis_pm.chain._mem", return_value=mem):
            update_chain_state("g", [
                {"tid": "a", "status": "dispatched"},
                {"tid": "b", "status": "pending"},
            ])
            on_leg_landed("a", "g")
            state = get_chain_state("g")

        legs_by_tid = {leg["tid"]: leg for leg in state["legs"]}
        assert legs_by_tid["a"]["status"] == "landed"
        assert legs_by_tid["b"]["status"] == "pending"
        assert state["complete"] is False

    def test_all_landed_flips_complete(self):
        mem = _make_mem_store()
        with patch("lapis_pm.chain._mem", return_value=mem):
            update_chain_state("g2", [
                {"tid": "x", "status": "dispatched"},
            ])
            on_leg_landed("x", "g2")
            state = get_chain_state("g2")

        assert state["complete"] is True

    def test_no_group_is_noop(self):
        # Should not raise even if chain_group is empty
        mem = _make_mem_store()
        with patch("lapis_pm.chain._mem", return_value=mem):
            on_leg_landed("some-tid", "")  # empty group_id


# ---------------------------------------------------------------------------
# check_chain_advance
# ---------------------------------------------------------------------------

class TestCheckChainAdvance:
    def _make_target(self, tid, depends_on, initial_dispatch, chain_group,
                     pm_repo="test-repo"):
        """Build a minimal mock target object."""
        t = MagicMock()
        t.id = tid
        t.pm_bound = True
        t.pm_repo = pm_repo
        t.data = {
            "depends_on": depends_on,
            "initial_dispatch": initial_dispatch,
            "chain_group": chain_group,
        }
        return t

    def test_fires_when_single_dep_satisfied(self):
        """check_chain_advance's leg-advance fire must route through
        pm_core.force_dispatch — same guarded path as --force-dispatch — not a
        hand-rolled vars_ dict missing intent_block (regression for the
        chain-dispatch intent_block KeyError bug).
        """
        from lapis_pm import pm_core

        mem = _make_mem_store()
        # Set up: leg_b depends on leg_a; leg_a is now landed
        leg_b = self._make_target("leg_b", ["leg_a"], "implement B", "grp")
        leg_a = self._make_target("leg_a", [], None, "grp")
        leg_a.pm_bound = False  # already archived after landing

        with (
            patch("lapis_pm.chain._mem", return_value=mem),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            # Mark leg_a as landed
            mem.set("pm/landed/leg_a",
                    json.dumps({"landed_at": "2026-04-30T00:00:00+00:00"}),
                    tags=["lapis-pm", "landed"])

            # Store for iteration returns leg_b (leg_a unbound)
            mock_store = MagicMock()
            mock_store.load_all.return_value = [leg_b]

            # pm_core.force_dispatch does its own TargetStore().get(target_id)
            # lookup (guard checks) independent of chain.py's iteration store.
            pm_core_store = MagicMock()
            pm_core_store.get.return_value = leg_b

            dispatch_result = MagicMock()
            dispatch_result.task_id = "task-123"
            dispatch_result.spec_id = "spec-abc"
            mock_dispatch = MagicMock(return_value=dispatch_result)

            write_dispatch_mock = MagicMock()

            with (
                patch("lapis_pm.chain.TargetStore", return_value=mock_store),
                patch("lapis_pm.pm_core.TargetStore", return_value=pm_core_store),
                patch.object(pm_core._SHAPER, "dispatch", mock_dispatch),
                patch("lapis_pm.pm_core._now_iso", return_value="2026-04-30T00:00:00+00:00"),
                patch("lapis_pm.episodic.write_dispatch", write_dispatch_mock),
                patch("lapis_pm.episodic.spec_summary", return_value="spec text"),
            ):
                fired = check_chain_advance("leg_a")

                assert "leg_b" in fired

                # No KeyError: force_dispatch supplied intent_block (and everything
                # else the real fixer system_template requires) — prove it against
                # the actual registry contract, not just a mock.
                mock_dispatch.assert_called_once()
                vars_ = mock_dispatch.call_args.kwargs["vars_"]
                assert "intent_block" in vars_
                fixer_agent = pm_core._SHAPER.get_agent("fixer")
                fixer_agent.system_template.format(**vars_)  # must not raise KeyError

                # dispatched >= 1 via the store (not just the chain-advance return value)
                dispatched = pm_core.load_dispatched("leg_b")
                assert len(dispatched) >= 1
                assert dispatched[0]["gpu_id"] == "task-123"

                # Exactly one episodic write, carrying chain-specific tags.
                write_dispatch_mock.assert_called_once()
                _, wkwargs = write_dispatch_mock.call_args
                extra_tags = wkwargs["extra_tags"]
                assert "pm:chain-group=grp" in extra_tags
                assert "pm:chain-triggered-by=leg_a" in extra_tags

    def test_does_not_fire_when_not_all_deps_satisfied(self):
        mem = _make_mem_store()
        # leg_c depends on both leg_a and leg_b; only leg_a landed
        leg_c = self._make_target("leg_c", ["leg_a", "leg_b"], "implement C", "grp")

        with (
            patch("lapis_pm.chain._mem", return_value=mem),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            mem.set("pm/landed/leg_a",
                    json.dumps({"landed_at": "2026-04-30T00:00:00+00:00"}),
                    tags=["lapis-pm", "landed"])
            # leg_b NOT in landed

            mock_store = MagicMock()
            mock_store.load_all.return_value = [leg_c]

            with patch("lapis_pm.chain.TargetStore", return_value=mock_store):
                fired = check_chain_advance("leg_a")

        assert "leg_c" not in fired

    def test_idempotent_does_not_double_fire(self):
        mem = _make_mem_store()
        leg_b = self._make_target("leg_b", ["leg_a"], "implement B", "grp")

        with (
            patch("lapis_pm.chain._mem", return_value=mem),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            mem.set("pm/landed/leg_a",
                    json.dumps({"landed_at": "2026-04-30T00:00:00+00:00"}),
                    tags=["lapis-pm", "landed"])
            # Simulate leg_b already having a pending dispatch
            mem.set("pm/dispatched/leg_b",
                    json.dumps([{"status": "pending", "gpu_id": "existing-task"}]),
                    tags=["lapis-pm"])

            mock_store = MagicMock()
            mock_store.load_all.return_value = [leg_b]

            with patch("lapis_pm.chain.TargetStore", return_value=mock_store):
                fired = check_chain_advance("leg_a")

        # Should NOT fire because leg_b already has a pending dispatch
        assert "leg_b" not in fired

    def test_skips_target_with_no_initial_dispatch(self):
        mem = _make_mem_store()
        leg_b = self._make_target("leg_b", ["leg_a"], None, "grp")  # no initial_dispatch

        with (
            patch("lapis_pm.chain._mem", return_value=mem),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            mem.set("pm/landed/leg_a",
                    json.dumps({"landed_at": "2026-04-30T00:00:00+00:00"}),
                    tags=["lapis-pm", "landed"])

            mock_store = MagicMock()
            mock_store.load_all.return_value = [leg_b]

            with patch("lapis_pm.chain.TargetStore", return_value=mock_store):
                fired = check_chain_advance("leg_a")

        assert "leg_b" not in fired

    def test_skips_unbound_target(self):
        mem = _make_mem_store()
        leg_b = self._make_target("leg_b", ["leg_a"], "do it", "grp")
        leg_b.pm_bound = False  # not bound

        with (
            patch("lapis_pm.chain._mem", return_value=mem),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            mem.set("pm/landed/leg_a",
                    json.dumps({"landed_at": "2026-04-30T00:00:00+00:00"}),
                    tags=["lapis-pm", "landed"])

            mock_store = MagicMock()
            mock_store.load_all.return_value = [leg_b]

            with patch("lapis_pm.chain.TargetStore", return_value=mock_store):
                fired = check_chain_advance("leg_a")

        assert "leg_b" not in fired

    def test_idempotent_when_dispatch_progressed_past_pending(self):
        """A leg whose dispatch moved to 'processed'/'failed' (not 'landed') must
        still be considered fired — chain state is canonical, dispatched-record
        status is not the only signal.
        """
        mem = _make_mem_store()
        leg_b = self._make_target("leg_b", ["leg_a"], "implement B", "grp")

        with (
            patch("lapis_pm.chain._mem", return_value=mem),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            mem.set("pm/landed/leg_a",
                    json.dumps({"landed_at": "2026-04-30T00:00:00+00:00"}),
                    tags=["lapis-pm", "landed"])
            # Dispatch already moved past 'pending' — e.g. GPU finished but PR
            # never opened/merged. Without a chain-state-aware guard, the
            # leg would re-fire on every subsequent sibling land.
            mem.set("pm/dispatched/leg_b",
                    json.dumps([{"status": "processed", "gpu_id": "old-task"}]),
                    tags=["lapis-pm"])
            # Chain state correctly reflects that leg_b was already dispatched.
            update_chain_state("grp", [
                {"tid": "leg_a", "status": "landed"},
                {"tid": "leg_b", "status": "dispatched"},
            ])

            mock_store = MagicMock()
            mock_store.load_all.return_value = [leg_b]

            with patch("lapis_pm.chain.TargetStore", return_value=mock_store):
                fired = check_chain_advance("leg_a")

        assert "leg_b" not in fired


# ---------------------------------------------------------------------------
# Chain bind atomicity (pre-flight + rollback)
# ---------------------------------------------------------------------------

class TestChainBindAtomicity:
    """Bind-mode atomicity:
    - Pre-flight checks every leg's existence + spec status before any mutation.
    - If any pre-flight fails, no targets are created and no specs are written.
    """

    def _write_legs_yaml(self, tmp_path, legs_data: dict) -> str:
        import yaml
        legs_path = tmp_path / "legs.yaml"
        legs_path.write_text(yaml.safe_dump(legs_data))
        return str(legs_path)

    def _write_spec(self, tmp_path) -> str:
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("# spec\nbody")
        return str(spec_path)

    def test_preflight_existing_target_no_force_aborts_before_create(self, tmp_path, monkeypatch):
        """If leg N's target exists and --force is absent, no leg gets created."""
        from lapis_pm import cli as cli_mod
        from agents_core.targets import TargetStore

        monkeypatch.setattr("agents_core.targets.TARGETS_DIR", tmp_path)
        monkeypatch.setattr("lapis_pm.cli.TargetStore",
                            lambda: TargetStore(targets_dir=tmp_path))

        # Pre-create only leg_b — leg_a does NOT exist
        store = TargetStore(targets_dir=tmp_path)
        store.create("leg_b", title="Pre-existing", urgency="medium",
                     description="x", category="active-work")

        legs_path = self._write_legs_yaml(tmp_path, {
            "legs": [
                {"tid": "leg_a", "repo": "r", "authority": "advisory",
                 "intent": "do a", "branch_slug": "implement"},
                {"tid": "leg_b", "repo": "r", "authority": "advisory",
                 "intent": "do b", "depends_on": ["leg_a"], "branch_slug": "implement"},
            ]
        })
        spec_path = self._write_spec(tmp_path)

        parser = cli_mod.build_parser()
        args = parser.parse_args([
            "bind", "grp",
            "--spec-from", spec_path,
            "--legs-from", legs_path,
            "--create",
        ])
        rc = cli_mod.cmd_bind(args)
        assert rc == 2

        # leg_a must NOT have been created (pre-flight aborts before any create)
        assert (tmp_path / "leg_a.yaml").exists() is False

    def test_preflight_missing_target_no_create_aborts(self, tmp_path, monkeypatch):
        """Without --create, leg with no existing target aborts pre-flight."""
        from lapis_pm import cli as cli_mod
        from agents_core.targets import TargetStore

        monkeypatch.setattr("agents_core.targets.TARGETS_DIR", tmp_path)
        monkeypatch.setattr("lapis_pm.cli.TargetStore",
                            lambda: TargetStore(targets_dir=tmp_path))

        legs_path = self._write_legs_yaml(tmp_path, {
            "legs": [
                {"tid": "leg_x", "repo": "r", "authority": "advisory",
                 "intent": "do x", "branch_slug": "implement"},
            ]
        })
        spec_path = self._write_spec(tmp_path)

        parser = cli_mod.build_parser()
        args = parser.parse_args([
            "bind", "grp",
            "--spec-from", spec_path,
            "--legs-from", legs_path,
            # NO --create
        ])
        rc = cli_mod.cmd_bind(args)
        assert rc == 2
        assert (tmp_path / "leg_x.yaml").exists() is False

    def test_no_auto_fire_writes_pending_status_for_no_dep_legs(self, tmp_path, monkeypatch):
        """--no-auto-fire must NOT mark no-dep legs as 'dispatched' in chain state.

        Pre-fix: chain state recorded "dispatched" for no-dep legs based purely on
        depends_on, even when --no-auto-fire suppressed the actual dispatch loop.
        That made chain state lie about what fired. Fix: status is 'dispatched' only
        when we'll actually fire below.
        """
        from lapis_pm import cli as cli_mod
        from agents_core.targets import TargetStore

        monkeypatch.setattr("agents_core.targets.TARGETS_DIR", tmp_path)
        monkeypatch.setattr("lapis_pm.cli.TargetStore",
                            lambda: TargetStore(targets_dir=tmp_path))

        mem = _make_mem_store()
        monkeypatch.setattr("lapis_pm.chain._mem", lambda: mem)
        monkeypatch.setattr("lapis_pm.pm_core._mem", lambda: mem)
        monkeypatch.setattr("lapis_pm.episodic.spec", lambda tid: None)
        monkeypatch.setattr("lapis_pm.episodic.write_spec",
                            lambda tid, body: None)
        monkeypatch.setattr("lapis_pm.pm_core.clear_classified_prs",
                            lambda tid: None)

        legs_path = self._write_legs_yaml(tmp_path, {
            "legs": [
                {"tid": "root", "repo": "r", "authority": "advisory",
                 "intent": "do root", "branch_slug": "implement"},
                {"tid": "child", "repo": "r", "authority": "advisory",
                 "intent": "do child", "depends_on": ["root"], "branch_slug": "implement"},
            ]
        })
        spec_path = self._write_spec(tmp_path)

        parser = cli_mod.build_parser()
        args = parser.parse_args([
            "bind", "grp",
            "--spec-from", spec_path,
            "--legs-from", legs_path,
            "--create",
            "--no-auto-fire",
        ])
        rc = cli_mod.cmd_bind(args)
        assert rc == 0, "bind should succeed"

        state = get_chain_state("grp")
        assert state is not None
        legs_by_tid = {leg["tid"]: leg for leg in state["legs"]}
        # Root has no deps but --no-auto-fire suppressed firing → must be pending.
        assert legs_by_tid["root"]["status"] == "pending", (
            f"--no-auto-fire root should be pending, got "
            f"{legs_by_tid['root']['status']!r}"
        )
        # Child depends on root → pending regardless.
        assert legs_by_tid["child"]["status"] == "pending"

    def test_auto_fire_dispatches_leg1_with_no_keyerror(self, tmp_path, monkeypatch):
        """cmd_bind_chain's initial-fire loop (leg 1, no depends_on) must route
        through pm_core.force_dispatch — not a hand-rolled vars_ dict missing
        intent_block. Regression for the chain-dispatch intent_block KeyError
        bug: pre-fix, this crashed inside _SHAPER.dispatch -> _compose_system
        with KeyError('intent_block'), silently, after the chain legs were
        already written — bind reported success but no fixer ever dispatched.
        """
        from lapis_pm import cli as cli_mod, pm_core
        from agents_core.targets import TargetStore

        monkeypatch.setattr("agents_core.targets.TARGETS_DIR", tmp_path)
        monkeypatch.setattr("lapis_pm.cli.TargetStore",
                            lambda: TargetStore(targets_dir=tmp_path))
        monkeypatch.setattr("lapis_pm.pm_core.TargetStore",
                            lambda: TargetStore(targets_dir=tmp_path))

        mem = _make_mem_store()
        monkeypatch.setattr("lapis_pm.chain._mem", lambda: mem)
        monkeypatch.setattr("lapis_pm.pm_core._mem", lambda: mem)
        monkeypatch.setattr("lapis_pm.episodic.spec", lambda tid: None)
        monkeypatch.setattr("lapis_pm.episodic.write_spec",
                            lambda tid, body: None)
        monkeypatch.setattr("lapis_pm.pm_core.clear_classified_prs",
                            lambda tid: None)
        monkeypatch.setattr("lapis_pm.episodic.spec_summary",
                            lambda tid, max_chars=None: "spec text")

        dispatch_result = MagicMock()
        dispatch_result.task_id = "task-leg1"
        dispatch_result.spec_id = "spec-leg1"
        mock_dispatch = MagicMock(return_value=dispatch_result)
        monkeypatch.setattr(pm_core._SHAPER, "dispatch", mock_dispatch)

        write_dispatch_mock = MagicMock()
        monkeypatch.setattr("lapis_pm.episodic.write_dispatch", write_dispatch_mock)

        legs_path = self._write_legs_yaml(tmp_path, {
            "legs": [
                {"tid": "leg1", "repo": "r", "authority": "advisory",
                 "intent": "do leg1", "branch_slug": "implement"},
            ]
        })
        spec_path = self._write_spec(tmp_path)

        parser = cli_mod.build_parser()
        args = parser.parse_args([
            "bind", "grp3",
            "--spec-from", spec_path,
            "--legs-from", legs_path,
            "--create",
        ])
        rc = cli_mod.cmd_bind(args)
        assert rc == 0, "bind should succeed"

        # No KeyError: force_dispatch supplied intent_block (and everything
        # else the real fixer system_template requires) — prove it against
        # the actual registry contract, not just a mock.
        mock_dispatch.assert_called_once()
        vars_ = mock_dispatch.call_args.kwargs["vars_"]
        assert "intent_block" in vars_
        fixer_agent = pm_core._SHAPER.get_agent("fixer")
        fixer_agent.system_template.format(**vars_)  # must not raise KeyError

        # dispatched >= 1 via the store — leg 1 actually dispatches, not just binds.
        dispatched = pm_core.load_dispatched("leg1")
        assert len(dispatched) >= 1
        assert dispatched[0]["gpu_id"] == "task-leg1"

        # Exactly one episodic write, carrying chain-specific tags.
        write_dispatch_mock.assert_called_once()
        _, wkwargs = write_dispatch_mock.call_args
        extra_tags = wkwargs["extra_tags"]
        assert "pm:chain-group=grp3" in extra_tags
        assert "pm:chain-initial" in extra_tags


# ---------------------------------------------------------------------------
# Branch slug injection tests
# ---------------------------------------------------------------------------

class TestBranchSlugInjection:
    """Test that cmd_bind_chain injects the Branch: line into initial_dispatch."""

    def _write_legs_yaml(self, tmp_path, legs_data: dict) -> str:
        import yaml
        legs_path = tmp_path / "legs.yaml"
        legs_path.write_text(yaml.safe_dump(legs_data))
        return str(legs_path)

    def _write_spec(self, tmp_path) -> str:
        spec_path = tmp_path / "spec.md"
        spec_path.write_text("# spec\nbody")
        return str(spec_path)

    def test_branch_injected_into_initial_dispatch(self, tmp_path, monkeypatch):
        """Each leg's initial_dispatch starts with 'Branch: lapis/<tid>/<slug>'."""
        from lapis_pm import cli as cli_mod
        from agents_core.targets import TargetStore

        monkeypatch.setattr("agents_core.targets.TARGETS_DIR", tmp_path)
        monkeypatch.setattr("lapis_pm.cli.TargetStore",
                            lambda: TargetStore(targets_dir=tmp_path))

        mem = _make_mem_store()
        monkeypatch.setattr("lapis_pm.chain._mem", lambda: mem)
        monkeypatch.setattr("lapis_pm.pm_core._mem", lambda: mem)
        monkeypatch.setattr("lapis_pm.episodic.spec", lambda tid: None)
        monkeypatch.setattr("lapis_pm.episodic.write_spec", lambda tid, body: None)
        monkeypatch.setattr("lapis_pm.pm_core.clear_classified_prs", lambda tid: None)

        legs_path = self._write_legs_yaml(tmp_path, {
            "legs": [
                {"tid": "my-spec-leg1", "repo": "repo1", "authority": "advisory",
                 "intent": "Implement the thing.", "branch_slug": "primitive"},
                {"tid": "my-spec-leg2", "repo": "repo2", "authority": "advisory",
                 "intent": "Implement the other thing.", "branch_slug": "integrate",
                 "depends_on": ["my-spec-leg1"]},
            ]
        })
        spec_path = self._write_spec(tmp_path)

        parser = cli_mod.build_parser()
        args = parser.parse_args([
            "bind", "my-chain",
            "--spec-from", spec_path,
            "--legs-from", legs_path,
            "--create",
            "--no-auto-fire",
        ])
        rc = cli_mod.cmd_bind(args)
        assert rc == 0

        store = TargetStore(targets_dir=tmp_path)
        leg1 = store.get("my-spec-leg1")
        leg2 = store.get("my-spec-leg2")

        # Leg 1: branch must be lapis/my-spec-leg1/primitive
        dispatch1 = leg1.data["initial_dispatch"]
        assert dispatch1.startswith("Branch: lapis/my-spec-leg1/primitive\n\n"), (
            f"leg1 initial_dispatch does not start with Branch: line: {dispatch1!r}"
        )
        assert "Implement the thing." in dispatch1

        # Leg 2: branch must be lapis/my-spec-leg2/integrate
        dispatch2 = leg2.data["initial_dispatch"]
        assert dispatch2.startswith("Branch: lapis/my-spec-leg2/integrate\n\n"), (
            f"leg2 initial_dispatch does not start with Branch: line: {dispatch2!r}"
        )
        assert "Implement the other thing." in dispatch2

    def test_intent_preserved_verbatim_after_branch_line(self, tmp_path, monkeypatch):
        """The original intent body is preserved exactly after the injected Branch: line."""
        from lapis_pm import cli as cli_mod
        from agents_core.targets import TargetStore

        monkeypatch.setattr("agents_core.targets.TARGETS_DIR", tmp_path)
        monkeypatch.setattr("lapis_pm.cli.TargetStore",
                            lambda: TargetStore(targets_dir=tmp_path))

        mem = _make_mem_store()
        monkeypatch.setattr("lapis_pm.chain._mem", lambda: mem)
        monkeypatch.setattr("lapis_pm.pm_core._mem", lambda: mem)
        monkeypatch.setattr("lapis_pm.episodic.spec", lambda tid: None)
        monkeypatch.setattr("lapis_pm.episodic.write_spec", lambda tid, body: None)
        monkeypatch.setattr("lapis_pm.pm_core.clear_classified_prs", lambda tid: None)

        intent_body = "Line one.\nLine two.\nLine three."
        legs_path = self._write_legs_yaml(tmp_path, {
            "legs": [
                {"tid": "leg-a", "repo": "r", "authority": "advisory",
                 "intent": intent_body, "branch_slug": "impl"},
            ]
        })
        spec_path = self._write_spec(tmp_path)

        parser = cli_mod.build_parser()
        args = parser.parse_args([
            "bind", "grp2",
            "--spec-from", spec_path,
            "--legs-from", legs_path,
            "--create",
            "--no-auto-fire",
        ])
        rc = cli_mod.cmd_bind(args)
        assert rc == 0

        store = TargetStore(targets_dir=tmp_path)
        leg = store.get("leg-a")
        dispatch = leg.data["initial_dispatch"]

        expected = f"Branch: lapis/leg-a/impl\n\n{intent_body.strip()}"
        assert dispatch == expected, (
            f"initial_dispatch does not match expected.\n"
            f"Expected: {expected!r}\n"
            f"Got:      {dispatch!r}"
        )

    def test_missing_branch_slug_rejected_before_any_leg_created(self, tmp_path, monkeypatch):
        """A chain.yaml without branch_slug fails bind before any target is written."""
        from lapis_pm import cli as cli_mod
        from agents_core.targets import TargetStore

        monkeypatch.setattr("agents_core.targets.TARGETS_DIR", tmp_path)
        monkeypatch.setattr("lapis_pm.cli.TargetStore",
                            lambda: TargetStore(targets_dir=tmp_path))

        legs_path = self._write_legs_yaml(tmp_path, {
            "legs": [
                {"tid": "leg-x", "repo": "r", "authority": "advisory",
                 "intent": "do x"},  # missing branch_slug
            ]
        })
        spec_path = self._write_spec(tmp_path)

        parser = cli_mod.build_parser()
        args = parser.parse_args([
            "bind", "grp3",
            "--spec-from", spec_path,
            "--legs-from", legs_path,
            "--create",
        ])
        rc = cli_mod.cmd_bind(args)
        assert rc == 2, "bind should fail (rc=2) when branch_slug is missing"
        # No target YAML should have been created
        assert not (tmp_path / "leg-x.yaml").exists(), (
            "leg-x.yaml was created despite missing branch_slug (atomicity violated)"
        )
