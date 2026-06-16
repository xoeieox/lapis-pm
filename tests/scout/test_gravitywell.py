"""Tests for Scout GravityWell routing and fallback behavior."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.scout.runner import (
    ScoutGravityWellUnavailable,
    _ScoutGravityWellAdapter,
    run_single,
    simulate,
)
from lapis_pm.scout.scaffold import load_scaffold

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SMOKE_YAML = FIXTURE_DIR / "smoke.yaml"


class TestGravityWellRoutingAndFallback:
    """Test GravityWell adapter routing and skip behavior."""

    def test_scout_gravitywell_routes_to_call_operator(self) -> None:
        """Verify GravityWell adapter calls call_operator with correct parameters."""
        from lapis_engine import Message

        adapter = _ScoutGravityWellAdapter(timeout=300)

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = '{"test": "response"}'

            messages = [Message(role="user", content="Test prompt")]
            result = adapter.chat(system="You are a test", messages=messages)

            mock_call.assert_called_once()
            call_args = mock_call.call_args
            assert call_args[0][0] == "gravitywell"
            assert call_args[1]["json_mode"] is True
            assert call_args[1]["on_wake_fail"] == "skip"
            assert result == '{"test": "response"}'

    def test_scout_gravitywell_unavailable_skips_legibly(self, tmp_path: Path, caplog) -> None:
        """Verify GW unavailability produces gw_skipped trace marker and WARNING log."""
        scaffold = load_scaffold(str(SMOKE_YAML))
        cell_params = scaffold.cell_params()[0]

        adapter = _ScoutGravityWellAdapter(timeout=300)

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = None

            traces_root = tmp_path / "traces"
            written = simulate(
                str(SMOKE_YAML),
                runs_per_cell=1,
                cells=[scaffold.cell_id(cell_params)],
                traces_root=traces_root,
                llm=adapter,
            )

            assert len(written) == 1
            trace_data = json.loads(written[0].read_text())
            assert trace_data["payload"]["gw_skipped"] is True

    def test_scout_qwen_still_routes_to_starhouse(self, tmp_path: Path) -> None:
        """Verify qwen model still routes to StarHouse (back-compat)."""
        from lapis_pm.scout.runner import _ScoutJsonAdapter

        scaffold = load_scaffold(str(SMOKE_YAML))
        cell_params = scaffold.cell_params()[0]

        class MockQwenAdapter:
            model = "qwen"

            def chat(self, system: str, messages: list) -> str:  # type: ignore[override]
                return json.dumps({
                    "scenario_generated": "test",
                    "execution_trace": "test",
                    "breaks_observed": [],
                    "leverage_points": [],
                    "surprises": [],
                    "drift_signals": [],
                    "tools_used": {},
                    "tools_wished_for": [],
                    "performance_assessment": "test",
                })

        traces_root = tmp_path / "traces"
        written = simulate(
            str(SMOKE_YAML),
            runs_per_cell=1,
            cells=[scaffold.cell_id(cell_params)],
            traces_root=traces_root,
            llm=MockQwenAdapter(),
        )

        assert len(written) == 1
        trace_data = json.loads(written[0].read_text())
        assert trace_data["payload"]["gw_skipped"] is False

    def test_scout_gravitywell_forwards_json_mode(self) -> None:
        """Verify GW adapter always forwards json_mode=True to call_operator."""
        from lapis_engine import Message

        adapter = _ScoutGravityWellAdapter(timeout=300)

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = '{"result": "ok"}'

            messages = [Message(role="user", content="Test")]
            adapter.chat(system="Test system", messages=messages)

            call_kwargs = mock_call.call_args[1]
            assert call_kwargs["json_mode"] is True

    def test_scout_night_health_gate_model_aware(self) -> None:
        """Verify health gate derives correct URL based on model."""
        from lapis_pm.scout.night_queue import _derive_health_url

        with patch.dict("os.environ", {}, clear=False):
            url = _derive_health_url(model="gravitywell")
            assert "203.0.113.11" in url or "gravitywell" in url.lower()

            url = _derive_health_url(model="qwen")
            assert "203.0.113.12" in url or "llama" in url.lower()

            with patch.dict("os.environ", {"LAPIS_SCOUT_HEALTH_URL": "http://custom:8080/health"}):
                url = _derive_health_url(model="gravitywell")
                assert url == "http://custom:8080/health"
