"""Smoke test for lapis-scout-v0 (Definition of Done §3–6).

DoD coverage:
3.  simulate() writes LapisToolReturn-shaped JSON for each (cell, run).
4.  digest() writes a LapisToolReturn-shaped map with upstream_calls carrying
    each trace's manifest_hash AND input_refs carrying each trace's content_hash.
5.  JSON parse-success floor ≥80% (json_parse_failure events < 20% of runs).
6.  RoomRAG sandbox guard: "scout" NOT in roomrag scan_dirs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SMOKE_YAML = FIXTURE_DIR / "smoke.yaml"

# ---------------------------------------------------------------------------
# Matrix spec: 2 optional × 2 severity × 1 load × 3 runs = 12 traces
# ---------------------------------------------------------------------------
EXPECTED_CELLS = 4   # 2 × 2 × 1
RUNS_PER_CELL = 3
EXPECTED_TRACES = EXPECTED_CELLS * RUNS_PER_CELL  # 12

# ---------------------------------------------------------------------------
# Mock LLM — always returns valid JSON (exercises happy path)
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = json.dumps({
    "scenario_generated": "An entity event arrives at the dispatcher.",
    "execution_trace": (
        "Step 1: classify_event called → type-A. "
        "Step 2: route(handler_a) called → accepted. "
        "Step 3: emit_ack sent."
    ),
    "breaks_observed": [
        {
            "signature": "handler-B-timeout",
            "severity": "high",
            "at_step": "route-to-handler",
            "description": "Handler B did not respond within 2s under degraded state.",
        }
    ],
    "leverage_points": [
        {
            "description": "Cache lookup prevents reprocessing of duplicate events.",
            "at_step": "cache-lookup",
        }
    ],
    "surprises": ["Handler A occasionally returns stale acks under burst load."],
    "drift_signals": [
        {
            "at_step": "emit-ack",
            "invented": "Ack payload format not specified in scaffold.",
        }
    ],
    "tools_used": {"classify_event": 1, "route": 1, "emit_ack": 1},
    "tools_wished_for": [
        {
            "signature": "health_check(handler_id) -> HealthStatus",
            "would_help_at_step": "route-to-handler",
        }
    ],
    "performance_assessment": (
        "Architecture functional but brittle when handler B degrades. "
        "Cache-lookup optional step significantly reduces duplicate processing."
    ),
})


class MockLLM:
    model = "mock-llm"

    def chat(self, system: str, messages: list) -> str:  # type: ignore[override]
        return _MOCK_RESPONSE


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------

def _assert_lapis_tool_return_shape(data: dict, label: str) -> None:
    """Assert the top-level LapisToolReturn envelope shape."""
    assert "payload" in data, f"{label}: missing 'payload'"
    assert "summary" in data, f"{label}: missing 'summary'"
    assert data["summary"], f"{label}: 'summary' is empty"
    assert "provenance" in data, f"{label}: missing 'provenance'"
    prov = data["provenance"]
    assert prov.get("schema_version") == "lapis-provenance-v0", (
        f"{label}: wrong schema_version: {prov.get('schema_version')}"
    )
    assert prov.get("manifest_hash", "").startswith("sha256:"), (
        f"{label}: manifest_hash missing or malformed: {prov.get('manifest_hash')}"
    )
    assert prov.get("timestamp"), f"{label}: provenance.timestamp is empty"


# ---------------------------------------------------------------------------
# Test 1: simulate → LapisToolReturn-shaped traces
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_writes_expected_trace_count(self, tmp_path: Path) -> None:
        from lapis_pm.scout.runner import simulate

        traces_root = tmp_path / "traces"
        written = simulate(str(SMOKE_YAML), traces_root=traces_root, llm=MockLLM())

        assert len(written) == EXPECTED_TRACES, (
            f"Expected {EXPECTED_TRACES} traces, got {len(written)}"
        )

    def test_trace_lapis_tool_return_shape(self, tmp_path: Path) -> None:
        from lapis_pm.scout.runner import simulate

        traces_root = tmp_path / "traces"
        written = simulate(str(SMOKE_YAML), traces_root=traces_root, llm=MockLLM())

        for path in written:
            data = json.loads(path.read_text())
            _assert_lapis_tool_return_shape(data, str(path.name))

    def test_trace_provenance_agent_and_tool(self, tmp_path: Path) -> None:
        from lapis_pm.scout.runner import simulate

        traces_root = tmp_path / "traces"
        written = simulate(str(SMOKE_YAML), traces_root=traces_root, llm=MockLLM())

        for path in written:
            data = json.loads(path.read_text())
            prov = data["provenance"]
            assert prov.get("agent_id") == "lapis-scout/sim", (
                f"{path.name}: wrong agent_id: {prov.get('agent_id')}"
            )
            assert prov.get("tool") == "pseudocode_simulate", (
                f"{path.name}: wrong tool: {prov.get('tool')}"
            )

    def test_trace_payload_fields(self, tmp_path: Path) -> None:
        from lapis_pm.scout.runner import simulate

        traces_root = tmp_path / "traces"
        written = simulate(str(SMOKE_YAML), traces_root=traces_root, llm=MockLLM())

        for path in written:
            data = json.loads(path.read_text())
            pl = data["payload"]
            assert pl.get("scaffold_hash", "").startswith("sha256:"), (
                f"{path.name}: scaffold_hash missing or malformed"
            )
            assert "cell_id" in pl, f"{path.name}: missing cell_id"
            assert "tick_prompts" in pl, f"{path.name}: missing tick_prompts"
            assert "tick_prompt_hashes" in pl, f"{path.name}: missing tick_prompt_hashes"
            assert len(pl["tick_prompts"]) == len(pl["tick_prompt_hashes"]), (
                f"{path.name}: tick_prompts / tick_prompt_hashes length mismatch"
            )

    def test_trace_prompt_hash_populated(self, tmp_path: Path) -> None:
        from lapis_pm.scout.runner import simulate

        traces_root = tmp_path / "traces"
        written = simulate(str(SMOKE_YAML), traces_root=traces_root, llm=MockLLM())

        for path in written:
            data = json.loads(path.read_text())
            prov = data["provenance"]
            pl = data["payload"]
            if pl.get("tick_prompts"):
                assert prov.get("prompt_hash", "").startswith("sha256:"), (
                    f"{path.name}: prompt_hash missing despite tick_prompts"
                )


# ---------------------------------------------------------------------------
# Test 2: digest → LapisToolReturn-shaped map + dual-record discipline
# ---------------------------------------------------------------------------

class TestDigest:
    def _run_simulate(self, tmp_path: Path):
        from lapis_pm.scout.runner import simulate
        traces_root = tmp_path / "traces"
        written = simulate(str(SMOKE_YAML), traces_root=traces_root, llm=MockLLM())
        return traces_root, written

    def test_digest_writes_map(self, tmp_path: Path) -> None:
        from lapis_pm.scout.digest import digest

        traces_root, written = self._run_simulate(tmp_path)
        maps_root = tmp_path / "maps"
        out = digest("smoke", traces_root=traces_root, maps_root=maps_root)
        assert out.exists(), f"Map file not found: {out}"

    def test_map_lapis_tool_return_shape(self, tmp_path: Path) -> None:
        from lapis_pm.scout.digest import digest

        traces_root, _ = self._run_simulate(tmp_path)
        maps_root = tmp_path / "maps"
        out = digest("smoke", traces_root=traces_root, maps_root=maps_root)
        data = json.loads(out.read_text())
        _assert_lapis_tool_return_shape(data, "smoke-map")

    def test_map_provenance_agent_tool_model(self, tmp_path: Path) -> None:
        from lapis_pm.scout.digest import digest

        traces_root, _ = self._run_simulate(tmp_path)
        maps_root = tmp_path / "maps"
        out = digest("smoke", traces_root=traces_root, maps_root=maps_root)
        data = json.loads(out.read_text())
        prov = data["provenance"]
        assert prov.get("agent_id") == "lapis-scout/digest"
        assert prov.get("tool") == "map_digest"
        assert prov.get("model") is None, "digest model must be null (pure-CPU)"

    def test_map_run_count(self, tmp_path: Path) -> None:
        from lapis_pm.scout.digest import digest

        traces_root, written = self._run_simulate(tmp_path)
        maps_root = tmp_path / "maps"
        out = digest("smoke", traces_root=traces_root, maps_root=maps_root)
        data = json.loads(out.read_text())
        assert data["payload"]["runs"] == len(written), (
            f"Map run count {data['payload']['runs']} != traces written {len(written)}"
        )

    def test_dual_record_upstream_calls(self, tmp_path: Path) -> None:
        """Each trace appears in upstream_calls with its manifest_hash."""
        from lapis_pm.scout.digest import digest

        traces_root, written = self._run_simulate(tmp_path)
        maps_root = tmp_path / "maps"
        out = digest("smoke", traces_root=traces_root, maps_root=maps_root)
        data = json.loads(out.read_text())
        prov = data["provenance"]

        upstream_calls = prov.get("upstream_calls", [])
        sim_calls = [uc for uc in upstream_calls if uc.get("agent_id") == "lapis-scout/sim"]
        assert len(sim_calls) == len(written), (
            f"Expected {len(written)} upstream_calls, got {len(sim_calls)}"
        )
        for uc in sim_calls:
            assert uc.get("manifest_hash", "").startswith("sha256:"), (
                f"UpstreamRef missing manifest_hash: {uc}"
            )

    def test_dual_record_input_refs(self, tmp_path: Path) -> None:
        """Each trace appears in input_refs as type='tool_result' with content_hash."""
        from lapis_pm.scout.digest import digest

        traces_root, written = self._run_simulate(tmp_path)
        maps_root = tmp_path / "maps"
        out = digest("smoke", traces_root=traces_root, maps_root=maps_root)
        data = json.loads(out.read_text())
        prov = data["provenance"]

        input_refs = prov.get("input_refs", [])
        tool_result_refs = [ir for ir in input_refs if ir.get("type") == "tool_result"]
        assert len(tool_result_refs) == len(written), (
            f"Expected {len(written)} tool_result input_refs, got {len(tool_result_refs)}"
        )
        for ir in tool_result_refs:
            assert ir.get("content_hash", "").startswith("sha256:"), (
                f"InputRef missing content_hash: {ir}"
            )


# ---------------------------------------------------------------------------
# Test 3: JSON parse-success floor ≥80%
# ---------------------------------------------------------------------------

class TestJsonParseFloor:
    def test_parse_success_floor(self, tmp_path: Path) -> None:
        """≥80% of runs must have zero json_parse_failure events."""
        from lapis_pm.scout.runner import simulate

        traces_root = tmp_path / "traces"
        written = simulate(str(SMOKE_YAML), traces_root=traces_root, llm=MockLLM())
        assert len(written) > 0, "No traces written"

        total_parse_failures = sum(
            json.loads(p.read_text())["payload"].get("parse_failure_count", 0)
            for p in written
        )
        run_count = len(written)
        # Spec: json_parse_failure event count < 20% of total run count.
        # With MockLLM returning valid JSON every time, failures should be 0.
        threshold = int(run_count * 0.2)
        assert total_parse_failures <= threshold, (
            f"Parse failure floor violated: {total_parse_failures} failures "
            f"out of {run_count} runs (threshold {threshold}). "
            f"Smoke ERRORS, not silent degradation."
        )


# ---------------------------------------------------------------------------
# Test 4: RoomRAG sandbox guard
# ---------------------------------------------------------------------------

class TestRoomRAGSandboxGuard:
    def test_scout_not_in_roomrag_scan_dirs(self) -> None:
        """'scout' must NOT be in roomrag.config scan_dirs."""
        try:
            import roomrag.config  # type: ignore[import]
            settings = roomrag.config.get_settings()
            assert "scout" not in settings.scan_dirs, (
                f"SANDBOX VIOLATION: 'scout' found in roomrag scan_dirs: "
                f"{settings.scan_dirs}"
            )
        except ImportError:
            # TODO: roomrag scan_dirs guard — skipped (roomrag not importable in test env)
            pytest.skip("roomrag not importable in test environment")
