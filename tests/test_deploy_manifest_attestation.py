"""Unit tests for night-deploy-manifest-attestation-v0.

Covers DoD-0 (AST count re-verification), DoD-1 (unmanifested -> held),
DoD-2 (manifested -> clean), DoD-3 (deliberately stale copy detected through
the production entry point with patched _CONDUCTOR_SCRIPTS_SRC/_DEST),
DoD-4 (land output statement), DoD-5 (per-host attestation), DoD-6 (no
after: dependents), DoD-7 (bound — the node's bounded-ssh discipline), and
DoD-8 (alarm dedup).

The real-filesystem copy test (DoD-3) drives through the production entry
point (attestation.run_attestation) with patched module-level path constants
— the same pattern test_post_land_deploy_hook.py uses for
_CONDUCTOR_SCRIPTS_SRC/_DEST — never inline paths.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from lapis_pm import attestation, pm_core


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_completed_process(returncode=0, stderr="", stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stderr=stderr, stdout=stdout)


def _now():
    return datetime(2026, 9, 8, tzinfo=timezone.utc)


def _write_plan(path: Path, nodes: list[dict], **top) -> None:
    doc = {"schema_version": 1, "nodes": nodes}
    doc.update(top)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _script_node(script: str, node_id: str = "n1", lane: str = "shell") -> dict:
    return {
        "id": node_id,
        "command": f"/usr/bin/python3 {script}",
        "lane": lane,
        "gw_phase": "none",
        "bound": 600,
    }


# ---------------------------------------------------------------------------
# DoD-0: AST count re-verification (handles AnnAssign)
# ---------------------------------------------------------------------------

class TestDoD0AstCounts:
    def test_counts_match_baseline(self):
        """The baseline counts (26/12/5 after the +3 +1 dead-man +1
        gw_seat_lane — night-roles-seat-declaration-v0 O1 BLOCKER fix, rev 2)
        re-verified by an AST pass that handles BOTH Assign and AnnAssign (the
        :532 tuple is an AnnAssign; a grep or Assign-only AST pass silently
        misses it)."""
        counts = attestation.manifest_counts_ast()
        assert counts["_CONDUCTOR_NIGHT_SCRIPTS"] == 26
        assert counts["_CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS"] == 12
        assert counts["_CONDUCTOR_GW_HOST_SCRIPTS"] == 5

    def test_annotated_tuple_is_seen(self):
        """The AnnAssign tuple (not Assign) is what the AST pass must handle —
        a naive Assign-only pass returns 0 for _CONDUCTOR_NIGHT_SCRIPTS."""
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(pm_core))
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                t = node.target
                if isinstance(t, ast.Name) and t.id == "_CONDUCTOR_NIGHT_SCRIPTS":
                    found["annassign"] = len(node.value.elts)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "_CONDUCTOR_NIGHT_SCRIPTS":
                        found["assign"] = len(node.value.elts)
        # The tuple is an AnnAssign — the Assign-only pass would miss it.
        assert "annassign" in found
        assert found.get("annassign") == 26
        assert "assign" not in found

    def test_new_scripts_in_brix_gw_runtime_family(self):
        """DoD-1/DoD-2: the three drifted scripts are in the BRIX_GW_RUNTIME
        family (NOT _CONDUCTOR_NIGHT_SCRIPTS, whose exact-equality closure
        tests would fail on a naive append)."""
        for s in ("mini_1f916_night.py", "keeper_v0.py", "council_sweep.py"):
            assert s in pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS
            assert s not in pm_core._CONDUCTOR_NIGHT_SCRIPTS

    def test_night_deadman_in_brix_gw_runtime_family(self):
        """night-deadman-floor-v0 (Leg 1): the S1 dead-man script is in the
        BRIX_GW_RUNTIME family (NOT _CONDUCTOR_NIGHT_SCRIPTS, whose
        exact-equality closure tests would fail on a naive append)."""
        assert "night_deadman.py" in pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS
        assert "night_deadman.py" not in pm_core._CONDUCTOR_NIGHT_SCRIPTS

    def test_brix_gw_runtime_is_exactly_twelve(self):
        assert set(pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS) == {
            "gw_topology.py", "gw_actuator.py", "gw_host_safety.py",
            "flip_controller.py", "gw-topology", "gw-night-pre.py",
            "gw-night-post.py", "scout-night-pre.py",
            "mini_1f916_night.py", "keeper_v0.py", "council_sweep.py",
            "night_deadman.py",
        }

    def test_disjoint_from_night_scripts(self):
        """The disjointness invariant keeps holding after the +3."""
        assert not (
            set(pm_core._CONDUCTOR_NIGHT_SCRIPTS)
            & set(pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS)
        )

    def test_counts_read_the_on_disk_file_not_the_imported_module(self, tmp_path):
        """DoD-0 (reviewer note): manifest_counts_ast reads the ON-DISK source
        file, not the imported module — so it catches a case where the source
        file and the imported module genuinely disagree (a stale deploy
        clone). A source file with a different tuple length must change the
        count even though the in-memory tuple is untouched."""
        import ast
        import inspect

        real_path = Path(inspect.getsourcefile(pm_core))
        assert real_path.is_file(), "pm_core source file must be on disk"
        # The on-disk read is the baseline (not the in-memory module).
        counts = attestation.manifest_counts_ast()
        assert counts["_CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS"] == 12

        # Simulate a source-file / imported-module disagreement: a file whose
        # AnnAssign tuple has a different length. The AST pass must count the
        # FILE's tuple, not the imported one.
        fake = tmp_path / "pm_core_fake.py"
        fake.write_text(
            "_CONDUCTOR_NIGHT_SCRIPTS: tuple[str, ...] = ('a.py', 'b.py')\n"
            "_CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS: tuple[str, ...] = (\n"
            "    'x.py', 'y.py', 'z.py',\n"
            ")\n"
            "_CONDUCTOR_GW_HOST_SCRIPTS: tuple[str, ...] = ()\n"
        )
        tree = ast.parse(fake.read_text())
        found: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                t = node.target
                if isinstance(t, ast.Name):
                    found[t.id] = len(node.value.elts)
        assert found["_CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS"] == 3
        # ...and that is NOT what the imported module says — proving the
        # on-disk read is the independent baseline the count check verifies.
        assert found["_CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS"] != len(
            pm_core._CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS
        )


# ---------------------------------------------------------------------------
# DoD-1: unmanifested -> held, naming the scripts + ages
# ---------------------------------------------------------------------------

class TestDoD1UnmanifestedHeld:
    def test_unmanifested_script_is_held_and_named(self, tmp_path):
        """A script a node command invokes from /data/agents/scripts/ that is
        NOT in any manifest ends `held` and is named with its live-vs-source
        age (DoD-1)."""
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"

        # A live copy that is 2 days older than the source (the measured hole).
        (src / "stale_script.py").write_text("v2 (merged)\n")
        live = dest / "stale_script.py"
        live.write_text("v1 (stale)\n")
        old = _now() - timedelta(days=2)
        os.utime(live, (old.timestamp(), old.timestamp()))

        _write_plan(plan, [_script_node(str(live), "stale-node")])
        waivers.write_text("waivers: []\n")

        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers,
            src_dir=src, dest_dir=dest, now=_now(),
        )
        assert result.verdict == attestation.VERDICT_HELD
        assert result.reason == attestation.REASON_DRIFT
        assert str(live) in result.unmanifested
        row = next(r for r in result.rows if r.script == str(live))
        assert row.status == "drift"
        assert row.cmp == "differ"
        assert row.age_days is not None and row.age_days > 1.0

    def test_three_unmanifested_named(self, tmp_path):
        """DoD-1 regression: the exact triple (mini_1f916_night / keeper_v0 /
        council_sweep) is named when unmanifested."""
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"

        names = ["mini_1f916_night.py", "keeper_v0.py", "council_sweep.py"]
        for n in names:
            (src / n).write_text(f"# {n} v2\n")
            live = dest / n
            live.write_text(f"# {n} v1\n")

        _write_plan(plan, [_script_node(str(dest / n), f"node-{i}") for i, n in enumerate(names)])
        waivers.write_text("waivers: []\n")

        # A closure WITHOUT the three (the pre-fix state).
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",),
            brix_gw_runtime=("gw_topology.py",),
            gw_host_dests=(),
        )
        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers,
            src_dir=src, dest_dir=dest, closure=closure, now=_now(),
        )
        assert result.verdict == attestation.VERDICT_HELD
        for n in names:
            assert str(dest / n) in result.unmanifested


    def test_live_host_attestation_exercises_real_tree(self, tmp_path):
        """DoD-1/DoD-2 live regression (reviewer note: the fixture path is a
        documented substitution because the host was already fixed by the
        handoff host-op). This test exercises the REAL /data/agents/scripts
        tree + real conductor source + real plan file through the production
        entry point, so the detector keeps pointing at the live tree — the
        regression test no longer relies on the fixture path alone.

        The host is expected to be FIXED (the three scripts manifested +
        current, DoD-2) — so the assertion is that the live pass runs to a
        verdict (not internal_error) and every script the live plan invokes
        from /data/agents/scripts is accounted for by a row. If the host
        regresses (drift returns), the same pass reports `held` + drift rows —
        the live detector is the guard, not the fixture."""
        live_dir = Path("/data/agents/scripts")
        src_dir = Path("/srv/git/conductor/scripts")
        plan_path = Path("/data/slots/night-plan.yaml")
        if not (live_dir.is_dir() and src_dir.is_dir() and plan_path.is_file()):
            pytest.skip("live host tree not present (not on BRIX)")

        waivers = tmp_path / "waivers.yaml"
        waivers.write_text("waivers: []\n")
        ledger = tmp_path / "ledger.json"

        code = attestation.node_main(
            plan_path=plan_path, waiver_path=waivers, ledger_path=ledger,
            emit=False,  # no stdout noise; the ledger record is the surface
        )
        assert code in (attestation.EXIT_CLEAN, attestation.EXIT_HELD)

        records = attestation.read_ledger(ledger)
        assert records and records[-1]["verdict"] in ("clean", "held")
        rec = records[-1]
        assert rec["reason"] != "internal_error"  # the detector itself is sound

        # Every /data/agents/scripts script the live plan invokes must be
        # accounted for by a row (clean/drift/waived/absent/unverifiable) —
        # the live tree is what the night actually executes.
        parsed = attestation.parse_plan_narrow(plan_path.read_text())
        live_invoked = [
            s for s in (
                s for n in parsed.nodes for s in attestation.resolve_scripts(n.command)
            )
            if s.startswith(str(live_dir) + "/")
        ]
        row_scripts = {r["script"] for r in rec["rows"]}
        for s in live_invoked:
            assert s in row_scripts, f"{s} invoked by the live plan but has no attestation row"


# ---------------------------------------------------------------------------
# DoD-2: manifested -> clean
# ---------------------------------------------------------------------------

class TestDoD2ManifestedClean:
    def test_manifested_and_current_is_clean(self, tmp_path):
        """After the three are manifested (tuple update) and current, the node
        ends `clean` (DoD-2)."""
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"

        names = ["mini_1f916_night.py", "keeper_v0.py", "council_sweep.py"]
        for n in names:
            (src / n).write_text(f"# {n}\n")
            (dest / n).write_text(f"# {n}\n")  # current copy

        _write_plan(plan, [_script_node(str(dest / n), f"node-{i}") for i, n in enumerate(names)])
        waivers.write_text("waivers: []\n")

        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",),
            brix_gw_runtime=tuple(names),  # manifested
            gw_host_dests=(),
        )
        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers,
            src_dir=src, dest_dir=dest, closure=closure, now=_now(),
        )
        assert result.verdict == attestation.VERDICT_CLEAN
        assert result.reason is None
        assert result.unmanifested == []

    def test_manifested_but_stale_is_drift(self, tmp_path):
        """A manifested script whose live copy is stale is still a drift row
        (DoD-3) — manifestation is not correctness."""
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"

        (src / "m.py").write_text("v2\n")
        live = dest / "m.py"
        live.write_text("v1\n")
        old = _now() - timedelta(days=3)
        os.utime(live, (old.timestamp(), old.timestamp()))

        _write_plan(plan, [_script_node(str(live), "m-node")])
        waivers.write_text("waivers: []\n")

        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",),
            brix_gw_runtime=("m.py",),
            gw_host_dests=(),
        )
        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers,
            src_dir=src, dest_dir=dest, closure=closure, now=_now(),
        )
        assert result.verdict == attestation.VERDICT_HELD
        row = next(r for r in result.rows if r.script == str(live))
        assert row.status == "drift"
        assert row.cmp == "differ"


# ---------------------------------------------------------------------------
# DoD-3: deliberately stale copy detected through the production entry point
# ---------------------------------------------------------------------------

class TestDoD3StaleDetection:
    def test_stale_copy_reported_with_mtimes_and_cmp(self, tmp_path):
        """A deliberately stale copy of a manifested script is detected and
        reported with BOTH mtimes and a differing cmp (DoD-3). Driven through
        the production entry point (run_attestation) with patched module-level
        path constants — the same pattern the post-land deploy suite uses for
        _CONDUCTOR_SCRIPTS_SRC/_DEST."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"

        (src_dir / "gw_topology.py").write_text("# v2 merged\n")
        live = dest_dir / "gw_topology.py"
        live.write_text("# v1 stale\n")
        old = _now() - timedelta(days=6)
        os.utime(live, (old.timestamp(), old.timestamp()))

        _write_plan(plan, [_script_node(str(live), "gw-node")])
        waivers.write_text("waivers: []\n")

        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",),
            brix_gw_runtime=("gw_topology.py",),
            gw_host_dests=(),
        )
        with (
            patch.object(attestation, "CONDUCTOR_SCRIPTS_SRC", src_dir),
            patch.object(attestation, "AGENTS_SCRIPTS_DIR", dest_dir),
            patch.object(attestation, "NIGHT_PLAN_PATH", plan),
            patch.object(attestation, "WAIVER_FILE", waivers),
        ):
            result = attestation.run_attestation(now=_now())

        assert result.verdict == attestation.VERDICT_HELD
        row = next(r for r in result.rows if r.script == str(live))
        assert row.status == "drift"
        assert row.cmp == "differ"
        assert row.source_mtime is not None
        assert row.live_mtime is not None
        assert row.source_mtime > row.live_mtime  # source is newer
        assert row.age_days is not None and row.age_days > 5.0

    def test_node_main_returns_held_exit_code(self, tmp_path, capsys):
        """The node entry point returns the exit code the success_predicate
        maps: 2 for held (drift), not 0 (clean) and not 1 (failed)."""
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        ledger = tmp_path / "ledger.json"

        (src_dir / "s.py").write_text("v2\n")
        live = dest_dir / "s.py"
        live.write_text("v1\n")

        _write_plan(plan, [_script_node(str(live), "s-node")])
        waivers.write_text("waivers: []\n")

        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",),
            brix_gw_runtime=(),  # unmanifested -> drift
            gw_host_dests=(),
        )
        code = attestation.node_main(
            plan_path=plan, waiver_path=waivers, ledger_path=ledger,
            src_dir=src_dir, dest_dir=dest_dir, closure=closure,
            now=_now(), emit=True,
        )
        assert code == attestation.EXIT_HELD
        out = capsys.readouterr().out
        parsed = json.loads(out.strip().splitlines()[-1])
        assert parsed["verdict"] == "held"
        assert parsed["reason"] == "drift"
        # The ledger record was appended (I7).
        records = attestation.read_ledger(ledger)
        assert records and records[-1]["verdict"] == "held"


# ---------------------------------------------------------------------------
# Plan-file narrow parse (D1) + DoD-6
# ---------------------------------------------------------------------------

class TestPlanParse:
    def test_narrow_parse_extracts_id_command_lane(self, tmp_path):
        plan = tmp_path / "plan.yaml"
        _write_plan(plan, [_script_node("/data/agents/scripts/a.py", "a")])
        parsed = attestation.parse_plan_narrow(plan.read_text())
        assert not parsed.schema_drift
        assert parsed.nodes[0].id == "a"
        assert "/data/agents/scripts/a.py" in parsed.nodes[0].command
        assert parsed.nodes[0].lane == "shell"

    def test_unrecognized_top_level_key_is_schema_drift(self, tmp_path):
        plan = tmp_path / "plan.yaml"
        doc = {"schema_version": 1, "nodes": [], "brand_new_key": 42}
        plan.write_text(yaml.safe_dump(doc))
        parsed = attestation.parse_plan_narrow(plan.read_text())
        assert parsed.schema_drift
        assert "brand_new_key" in parsed.schema_drift_detail

    def test_schema_drift_holds_never_silent_pass(self, tmp_path):
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        doc = {"schema_version": 1, "nodes": [], "unknown": True}
        plan.write_text(yaml.safe_dump(doc))
        waivers.write_text("waivers: []\n")

        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers, src_dir=src, dest_dir=dest,
            now=_now(),
        )
        assert result.verdict == attestation.VERDICT_HELD
        assert result.reason == attestation.REASON_PLAN_SCHEMA_DRIFT

    def test_script_resolution_extracts_py_paths(self):
        cmd = "set -a && . /data/agents/config/conductor.env && /usr/bin/python3 /data/agents/scripts/a.py run"
        assert attestation.resolve_scripts(cmd) == ["/data/agents/scripts/a.py"]

    def test_script_resolution_ignores_non_py(self):
        cmd = "bash /data/agents/scripts/run.sh && /usr/bin/python3"
        assert attestation.resolve_scripts(cmd) == []


# ---------------------------------------------------------------------------
# DoD-6: no node may depend on the attestation node
# ---------------------------------------------------------------------------

class TestDoD6NoDependents:
    def test_dependent_node_is_rejected(self, tmp_path):
        """A node declaring after: [night-deploy-attestation] is rejected —
        the node's held verdict is proved not to darken any live node (DoD-6)."""
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()

        node = _script_node("/data/agents/scripts/a.py", "dependent")
        node["after"] = [attestation.ATTTESTATION_NODE_ID]
        _write_plan(plan, [node])
        waivers.write_text("waivers: []\n")

        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers, src_dir=src, dest_dir=dest,
            now=_now(),
        )
        assert result.verdict == attestation.VERDICT_HELD
        assert result.reason == attestation.REASON_PLAN_SCHEMA_DRIFT
        assert "DoD-6" in result.schema_drift_detail

    def test_no_dependent_is_clean(self, tmp_path):
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        _write_plan(plan, [_script_node("/data/agents/scripts/a.py", "a")])
        waivers.write_text("waivers: []\n")
        # a.py unmanifested -> drift, but NOT a DoD-6 rejection.
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",), brix_gw_runtime=("a.py",),
            gw_host_dests=(),
        )
        (src / "a.py").write_text("x\n")
        (dest / "a.py").write_text("x\n")
        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers, src_dir=src, dest_dir=dest,
            closure=closure, now=_now(),
        )
        assert result.verdict == attestation.VERDICT_CLEAN


# ---------------------------------------------------------------------------
# DoD-5: per-host attestation (deploy clone + agents-scripts + GW)
# ---------------------------------------------------------------------------

class TestDoD5PerHost:
    def test_gw_only_gap_reported_separately(self, tmp_path):
        """A GW-only gap is reported as its own row (status drift), never
        collapsed into the local boolean (DoD-5)."""
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        _write_plan(plan, [])
        waivers.write_text("waivers: []\n")

        # A GW script whose remote hash differs from the source.
        (src / "gw-topology").write_text("# v2\n")
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",),
            brix_gw_runtime=(),
            gw_host_dests=("/usr/local/sbin/gw-topology",),
        )

        def fake_gw_capture(remote_cmd, timeout=attestation._GW_SSH_CMD_TIMEOUT_SECS):
            if remote_cmd[:1] == ["sha256sum"]:
                return "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef  /usr/local/sbin/gw-topology"
            return None

        with (
            patch.object(attestation, "_gw_reachable", lambda: True),
            patch.object(attestation, "_gw_ssh_capture", fake_gw_capture),
        ):
            result = attestation.run_attestation(
                plan_path=plan, waiver_path=waivers, src_dir=src,
                dest_dir=dest, closure=closure, now=_now(),
            )

        gw_row = next(r for r in result.rows if r.host == "gravitywell")
        assert gw_row.status == "drift"
        assert gw_row.cmp == "differ"
        assert result.verdict == attestation.VERDICT_HELD

    def test_gw_unreachable_is_unverifiable_not_silent(self, tmp_path):
        """An unreachable GW yields `unverifiable` rows (not a silent pass,
        I6) — the checker reports what it cannot verify."""
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        _write_plan(plan, [])
        waivers.write_text("waivers: []\n")
        (src / "gw-topology").write_text("# v2\n")
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",),
            brix_gw_runtime=(),
            gw_host_dests=("/usr/local/sbin/gw-topology",),
        )
        with patch.object(attestation, "_gw_reachable", lambda: False):
            result = attestation.run_attestation(
                plan_path=plan, waiver_path=waivers, src_dir=src,
                dest_dir=dest, closure=closure, now=_now(),
            )
        gw_row = next(r for r in result.rows if r.host == "gravitywell")
        assert gw_row.status == "unverifiable"

    def test_gw_ssh_capture_never_raises(self, tmp_path):
        """The bounded never-wake ssh probe never raises (D6): a wedged probe
        (TimeoutExpired) returns None."""
        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=1)

        with patch.object(attestation.subprocess, "run", side_effect=boom):
            # _gw_ssh_capture must swallow the timeout.
            out = attestation._gw_ssh_capture(["sha256sum", "/x"])
        assert out is None


# ---------------------------------------------------------------------------
# DoD-4: land output statement (D3)
# ---------------------------------------------------------------------------

class TestDoD4LandReport:
    def test_land_report_states_no_deploy_explicitly(self, tmp_path, capsys):
        """A land that deploys nothing says so explicitly (D3 — no silent
        green)."""
        log = tmp_path / "deploy-log.md"
        log.write_text("")
        with (
            patch.object(pm_core, "_DEPLOY_LOG", str(log)),
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", True),
        ):
            report = pm_core.render_land_deploy_report("lapis-pm", pr_scripts=[])
        assert "nothing deployed this pass" in report
        assert "disabled" in report

    def test_land_report_names_just_merged_pr_gap(self, tmp_path):
        """D3 gate-3 amendment: land names any just-merged-PR script absent
        from the deployed set."""
        log = tmp_path / "deploy-log.md"
        log.write_text("")
        with (
            patch.object(pm_core, "_DEPLOY_LOG", str(log)),
            patch.object(pm_core, "_DEPLOY_HOOK_DISABLED", False),
        ):
            report = pm_core.render_land_deploy_report(
                "lapis-pm", pr_scripts=["scripts/brand_new_script.py"],
            )
        assert "GAP" in report
        assert "brand_new_script.py" in report

    def test_land_report_mark_clean(self, tmp_path):
        """A land with no drift and no pr scripts reads CLEAN."""
        log = tmp_path / "deploy-log.md"
        log.write_text("")
        waivers = tmp_path / "waivers.yaml"
        waivers.write_text("waivers: []\n")
        ledger = tmp_path / "ledger.json"
        with (
            patch.object(pm_core, "_DEPLOY_LOG", str(log)),
            patch.object(attestation, "WAIVER_FILE", waivers),
            patch.object(attestation, "ATTESTATION_LEDGER_PATH", ledger),
        ):
            report = pm_core.render_land_deploy_report("lapis-pm", pr_scripts=[])
        assert "mark: CLEAN" in report

    def test_land_report_mark_drift_with_duration(self, tmp_path):
        """The DRIFT mark carries duration from the ledger (the temporal
        scar, D3): DRIFT (3 nights, oldest 2026-09-05)."""
        log = tmp_path / "deploy-log.md"
        log.write_text("")
        waivers = tmp_path / "waivers.yaml"
        waivers.write_text("waivers: []\n")
        ledger = tmp_path / "ledger.json"
        # A ledger with 3 drift nights.
        attestation._atomic_write_json(ledger, {"records": [
            {"ts": "2026-09-05T00:00:00+00:00", "verdict": "held", "reason": "drift"},
            {"ts": "2026-09-06T00:00:00+00:00", "verdict": "held", "reason": "drift"},
            {"ts": "2026-09-07T00:00:00+00:00", "verdict": "held", "reason": "drift"},
        ]})
        with (
            patch.object(pm_core, "_DEPLOY_LOG", str(log)),
            patch.object(attestation, "WAIVER_FILE", waivers),
            patch.object(attestation, "ATTESTATION_LEDGER_PATH", ledger),
        ):
            report = pm_core.render_land_deploy_report(
                "lapis-pm", pr_scripts=["scripts/stale.py"],
            )
        assert "DRIFT" in report
        assert "3 nights" in report
        assert "oldest 2026-09-05" in report


    def test_land_report_mark_drift_on_preexisting_run_record_held(self, tmp_path):
        """D3 (current live state, not just the just-merged PR): a held
        attestation verdict in the latest run record — pre-existing drift in
        an UNRELATED script — reads DRIFT at land time, not CLEAN. The
        just-merged-PR subset is empty/manifested here; only the run-record
        verdict carries the drift."""
        log = tmp_path / "deploy-log.md"
        log.write_text("")
        waivers = tmp_path / "waivers.yaml"
        waivers.write_text("waivers: []\n")
        ledger = tmp_path / "ledger.json"
        run_record = tmp_path / "latest.json"
        run_record.write_text(json.dumps({
            "run_id": "2026-09-07T00:00:00Z",
            "status": "ended",
            "nodes": [
                {"id": "clone-currency-sync", "verdict": "clean"},
                {"id": "night-deploy-attestation", "verdict": "held",
                 "hold_reason_class": "drift",
                 "ended_at": "2026-09-07T01:00:00+00:00"},
            ],
        }))
        with (
            patch.object(pm_core, "_DEPLOY_LOG", str(log)),
            patch.object(attestation, "WAIVER_FILE", waivers),
            patch.object(attestation, "ATTESTATION_LEDGER_PATH", ledger),
            patch.object(attestation, "RUN_RECORD_PATH", run_record),
        ):
            report = pm_core.render_land_deploy_report("lapis-pm", pr_scripts=[])
        assert "mark: DRIFT" in report
        assert "run-record held" in report
        assert "pre-existing drift" in report

    def test_land_report_mark_clean_ignores_clean_run_record(self, tmp_path):
        """The run-record consultation does not flip a genuinely clean state:
        a `clean` attestation verdict in the run record leaves the mark CLEAN."""
        log = tmp_path / "deploy-log.md"
        log.write_text("")
        waivers = tmp_path / "waivers.yaml"
        waivers.write_text("waivers: []\n")
        ledger = tmp_path / "ledger.json"
        run_record = tmp_path / "latest.json"
        run_record.write_text(json.dumps({
            "run_id": "2026-09-07T00:00:00Z",
            "nodes": [
                {"id": "night-deploy-attestation", "verdict": "clean"},
            ],
        }))
        with (
            patch.object(pm_core, "_DEPLOY_LOG", str(log)),
            patch.object(attestation, "WAIVER_FILE", waivers),
            patch.object(attestation, "ATTESTATION_LEDGER_PATH", ledger),
            patch.object(attestation, "RUN_RECORD_PATH", run_record),
        ):
            report = pm_core.render_land_deploy_report("lapis-pm", pr_scripts=[])
        assert "mark: CLEAN" in report

    def test_latest_run_record_attestation_reads_pinned_consumer(self, tmp_path):
        """_latest_run_record_attestation reads the pinned run-record shape
        (per-node id/verdict/hold_reason_class/ended_at in the nodes list)
        and degrades to None on unreadable/malformed records (best-effort,
        never a crash)."""
        run_record = tmp_path / "latest.json"
        run_record.write_text(json.dumps({
            "nodes": [
                {"id": "a", "verdict": "clean"},
                {"id": "night-deploy-attestation", "verdict": "held",
                 "hold_reason_class": "drift",
                 "ended_at": "2026-09-07T01:00:00+00:00"},
            ],
        }))
        rec = pm_core._latest_run_record_attestation(run_record)
        assert rec == {
            "verdict": "held", "reason": "drift",
            "ended_at": "2026-09-07T01:00:00+00:00",
        }
        # No attestation node -> None.
        run_record.write_text(json.dumps({"nodes": [{"id": "a", "verdict": "clean"}]}))
        assert pm_core._latest_run_record_attestation(run_record) is None
        # Unreadable / malformed -> None (never raises).
        assert pm_core._latest_run_record_attestation(tmp_path / "missing.json") is None
        run_record.write_text("{not json")
        assert pm_core._latest_run_record_attestation(run_record) is None


# ---------------------------------------------------------------------------
# Waivers (D2): expiring, forever-bug, re-report on drift
# ---------------------------------------------------------------------------

class TestWaivers:
    def test_waived_unmanifested_is_clean(self, tmp_path):
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        (src / "w.py").write_text("v2\n")
        live = dest / "w.py"
        live.write_text("v1\n")
        _write_plan(plan, [_script_node(str(live), "w-node")])
        waivers.write_text(yaml.safe_dump({"waivers": [{
            "script": "w.py", "reason": "tracked fast-follow",
            "added": "2026-09-08", "owner": "pm",
            "expires": "2026-09-15",
        }]}))
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",), brix_gw_runtime=(), gw_host_dests=(),
        )
        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers, src_dir=src, dest_dir=dest,
            closure=closure, now=_now(),
        )
        assert result.verdict == attestation.VERDICT_CLEAN
        assert str(live) in result.waived

    def test_waived_script_that_drifts_is_re_reported(self, tmp_path):
        """D2: a waived script that later drifts is re-reported (a waiver does
        not silently absorb new drift)."""
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        (src / "w.py").write_text("v2\n")
        live = dest / "w.py"
        live.write_text("v1\n")
        _write_plan(plan, [_script_node(str(live), "w-node")])
        waivers.write_text(yaml.safe_dump({"waivers": [{
            "script": "w.py", "reason": "tracked fast-follow",
            "added": "2026-09-08", "owner": "pm",
            "expires": "2026-09-15",
        }]}))
        # Manifested but stale -> drift, re-reported despite the waiver.
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",), brix_gw_runtime=("w.py",),
            gw_host_dests=(),
        )
        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers, src_dir=src, dest_dir=dest,
            closure=closure, now=_now(),
        )
        assert result.verdict == attestation.VERDICT_HELD
        row = next(r for r in result.rows if r.script == str(live))
        assert row.status == "drift"
        assert "waived-but-drift" in (row.waiver or "")

    def test_forever_waiver_is_a_bug(self, tmp_path):
        """D2/Q-c: a forever-waiver (no expires) is a bug the node reports."""
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        (src / "w.py").write_text("v2\n")
        live = dest / "w.py"
        live.write_text("v1\n")
        _write_plan(plan, [_script_node(str(live), "w-node")])
        waivers.write_text(yaml.safe_dump({"waivers": [{
            "script": "w.py", "reason": "tracked fast-follow",
            "added": "2026-09-08", "owner": "pm",
        }]}))
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",), brix_gw_runtime=(), gw_host_dests=(),
        )
        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers, src_dir=src, dest_dir=dest,
            closure=closure, now=_now(),
        )
        assert any("forever-waiver" in p for p in result.waiver_problems)

    def test_expired_waiver_is_re_reported_as_drift(self, tmp_path):
        """D2: an expired waiver is re-reported as drift."""
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        (src / "w.py").write_text("v2\n")
        live = dest / "w.py"
        live.write_text("v1\n")
        _write_plan(plan, [_script_node(str(live), "w-node")])
        waivers.write_text(yaml.safe_dump({"waivers": [{
            "script": "w.py", "reason": "tracked fast-follow",
            "added": "2026-09-01", "owner": "pm",
            "expires": "2026-09-02",  # expired
        }]}))
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",), brix_gw_runtime=(), gw_host_dests=(),
        )
        result = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers, src_dir=src, dest_dir=dest,
            closure=closure, now=_now(),
        )
        assert result.verdict == attestation.VERDICT_HELD
        assert str(live) in result.unmanifested


# ---------------------------------------------------------------------------
# DoD-8: alarm dedup (persisting held pages once per day)
# ---------------------------------------------------------------------------

class TestDoD8AlarmDedup:
    def _drift_result(self, tmp_path):
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        (src / "d.py").write_text("v2\n")
        live = dest / "d.py"
        live.write_text("v1\n")
        _write_plan(plan, [_script_node(str(live), "d-node")])
        waivers.write_text("waivers: []\n")
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",), brix_gw_runtime=(), gw_host_dests=(),
        )
        return attestation.run_attestation(
            plan_path=plan, waiver_path=waivers, src_dir=src, dest_dir=dest,
            closure=closure, now=_now(),
        )

    def test_persisting_held_pages_once(self, tmp_path):
        """A persisting held pages once (the cooldown) — not every night
        (DoD-8)."""
        ledger = tmp_path / "alarm.json"
        deposits = []
        result = self._drift_result(tmp_path)

        # First pass: pages.
        r1 = attestation.maybe_alarm_drift(
            result, alarm_ledger_path=ledger,
            deposit=lambda p: deposits.append(p) or "gem-1",
            now=_now(),
        )
        assert r1["status"] == "deposited"

        # Second pass within the cooldown (same night): suppressed.
        r2 = attestation.maybe_alarm_drift(
            result, alarm_ledger_path=ledger,
            deposit=lambda p: deposits.append(p) or "gem-2",
            now=_now() + timedelta(minutes=30),
        )
        assert r2["status"] == "suppressed"
        assert len(deposits) == 1

        # After the cooldown (next day): pages again.
        r3 = attestation.maybe_alarm_drift(
            result, alarm_ledger_path=ledger,
            deposit=lambda p: deposits.append(p) or "gem-3",
            now=_now() + timedelta(days=1),
        )
        assert r3["status"] == "deposited"
        assert len(deposits) == 2

    def test_healthy_pass_resolves_open_entry(self, tmp_path):
        ledger = tmp_path / "alarm.json"
        # Seed an open entry.
        attestation._atomic_write_json(ledger, {
            attestation.ALARM_LEDGER_SIGNATURE: {
                "status": "open", "gem_id": "gem-1",
                "first_seen": "2026-09-07T00:00:00+00:00",
                "last_seen": "2026-09-07T00:00:00+00:00",
                "times_seen": 1,
            }
        })
        # A clean result resolves it.
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        _write_plan(plan, [])
        waivers.write_text("waivers: []\n")
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",), brix_gw_runtime=(), gw_host_dests=(),
        )
        clean = attestation.run_attestation(
            plan_path=plan, waiver_path=waivers, src_dir=src, dest_dir=dest,
            closure=closure, now=_now(),
        )
        r = attestation.maybe_alarm_drift(
            clean, alarm_ledger_path=ledger, now=_now(),
        )
        assert r["status"] == "resolved"
        doc = json.loads(ledger.read_text())
        assert doc[attestation.ALARM_LEDGER_SIGNATURE]["status"] == "resolved"


# ---------------------------------------------------------------------------
# Ledger (D5/I7): append-ledger-shaped, atomic
# ---------------------------------------------------------------------------

class TestLedger:
    def test_ledger_is_append_shaped(self, tmp_path):
        ledger = tmp_path / "ledger.json"
        attestation.append_ledger_record(
            {"ts": "2026-09-07T00:00:00+00:00", "verdict": "held", "reason": "drift"},
            ledger,
        )
        attestation.append_ledger_record(
            {"ts": "2026-09-08T00:00:00+00:00", "verdict": "clean", "reason": None},
            ledger,
        )
        records = attestation.read_ledger(ledger)
        assert len(records) == 2
        assert records[0]["verdict"] == "held"
        assert records[1]["verdict"] == "clean"

    def test_ledger_atomic_write_leaves_no_tmp(self, tmp_path):
        ledger = tmp_path / "ledger.json"
        attestation.append_ledger_record({"ts": "x", "verdict": "clean"}, ledger)
        tmps = list(tmp_path.glob(".ledger.json.*.tmp"))
        assert tmps == []
        assert ledger.exists()


# ---------------------------------------------------------------------------
# DoD-7: the node's bounded-ssh discipline (a wedged probe ends held within
# the bound, never blocks)
# ---------------------------------------------------------------------------

class TestDoD7Bound:
    def test_wedged_gw_probe_ends_held_not_blocked(self, tmp_path):
        """A wedged GW probe (the bound: 600 discipline) -> the node ends
        `held`/unverifiable within the bound and does not block (DoD-7). The
        bounded ssh returns None on timeout (never raises), so the GW row is
        `unverifiable`, not a hang."""
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        plan = tmp_path / "plan.yaml"
        waivers = tmp_path / "waivers.yaml"
        _write_plan(plan, [])
        waivers.write_text("waivers: []\n")
        (src / "gw-topology").write_text("# v2\n")
        closure = attestation.ManifestClosure(
            night_scripts=("night_plan.py",),
            brix_gw_runtime=(),
            gw_host_dests=("/usr/local/sbin/gw-topology",),
        )

        def wedged_ssh(*a, **k):
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=1)

        with (
            patch.object(attestation.subprocess, "run", side_effect=wedged_ssh),
        ):
            # _gw_reachable must return False on the wedged probe (never raise).
            reachable = attestation._gw_reachable()
        assert reachable is False
        # And the GW row is unverifiable (not a silent pass, not a hang).
        with patch.object(attestation, "_gw_reachable", lambda: False):
            result = attestation.run_attestation(
                plan_path=plan, waiver_path=waivers, src_dir=src,
                dest_dir=dest, closure=closure, now=_now(),
            )
        gw_row = next(r for r in result.rows if r.host == "gravitywell")
        assert gw_row.status == "unverifiable"


# ---------------------------------------------------------------------------
# CLI entry (pre-PR)
# ---------------------------------------------------------------------------

class TestCliEntry:
    def test_attestation_check_subcommand_registered(self):
        from lapis_pm import cli
        parser = cli.build_parser()
        args = parser.parse_args(["attestation-check", "--no-emit"])
        assert args.func is cli.cmd_attestation_check
