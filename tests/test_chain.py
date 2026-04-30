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
        legs = [{"tid": "a", "repo": "r", "authority": "advisory", "intent": "do it"}]
        validate_legs(legs)  # should not raise

    def test_valid_chain_linear(self):
        legs = [
            {"tid": "a", "depends_on": []},
            {"tid": "b", "depends_on": ["a"]},
            {"tid": "c", "depends_on": ["b"]},
        ]
        validate_legs(legs)

    def test_valid_parallel_rejoin(self):
        # legs b and c both depend on a; d depends on b and c
        legs = [
            {"tid": "a", "depends_on": []},
            {"tid": "b", "depends_on": ["a"]},
            {"tid": "c", "depends_on": ["a"]},
            {"tid": "d", "depends_on": ["b", "c"]},
        ]
        validate_legs(legs)

    def test_duplicate_tid(self):
        legs = [{"tid": "a"}, {"tid": "a"}]
        with pytest.raises(ValueError, match="Duplicate tid"):
            validate_legs(legs)

    def test_cyclic_direct(self):
        legs = [
            {"tid": "a", "depends_on": ["b"]},
            {"tid": "b", "depends_on": ["a"]},
        ]
        with pytest.raises(ValueError, match="Cyclic"):
            validate_legs(legs)

    def test_cyclic_indirect(self):
        legs = [
            {"tid": "a", "depends_on": ["c"]},
            {"tid": "b", "depends_on": ["a"]},
            {"tid": "c", "depends_on": ["b"]},
        ]
        with pytest.raises(ValueError, match="Cyclic"):
            validate_legs(legs)

    def test_unknown_dep(self):
        legs = [{"tid": "a", "depends_on": ["nonexistent"]}]
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

            dispatch_result = MagicMock()
            dispatch_result.task_id = "task-123"
            dispatch_result.spec_id = "spec-abc"

            mock_shaper = MagicMock()
            mock_shaper.dispatch.return_value = dispatch_result

            with (
                patch("lapis_pm.chain.TargetStore", return_value=mock_store),
                patch("lapis_pm.pm_core._SHAPER", mock_shaper),
                patch("lapis_pm.pm_core.append_dispatched"),
                patch("lapis_pm.pm_core._now_iso", return_value="2026-04-30T00:00:00+00:00"),
                patch("lapis_pm.episodic.write_dispatch"),
                patch("lapis_pm.episodic.spec_summary", return_value="spec text"),
                patch("lapis_pm.chain.send_auto_dispatch_brief", return_value=True),
            ):
                fired = check_chain_advance("leg_a")

        assert "leg_b" in fired

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
                 "intent": "do a"},
                {"tid": "leg_b", "repo": "r", "authority": "advisory",
                 "intent": "do b", "depends_on": ["leg_a"]},
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
                 "intent": "do x"},
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
                 "intent": "do root"},
                {"tid": "child", "repo": "r", "authority": "advisory",
                 "intent": "do child", "depends_on": ["root"]},
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
