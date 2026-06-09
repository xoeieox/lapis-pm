"""Regression tests for LLM-field coercion in _build_trace_payload.

Covers the live crash: execution_trace returned as list caused TypeError in
"\n---\n".join(execution_traces).  Also covers the mirror risk (_as_list
silently iterating a bare string char-by-char).

Spec: scout-trace-payload-list-coercion-v0
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Minimal mocks — only what _build_trace_payload needs
# ---------------------------------------------------------------------------


class _MockEvent:
    def __init__(self, type_: str) -> None:
        self.type = type_


class _MockStep:
    def __init__(self, response: str, *, events: list | None = None) -> None:
        self.response = response
        self.events: list = events or []


class _MockRunLog:
    def __init__(self, steps: list[_MockStep]) -> None:
        self.steps = steps


class _MockScaffold:
    spec_id = "test-spec-coercion"
    spec_version = "v0"

    def scaffold_hash(self, cell_params: dict) -> str:  # noqa: ARG002
        return "sha256:" + "0" * 64


class _MockDirector:
    def tick_prompts(self) -> list[str]:
        return ["tick-1"]


def _make_step(payload: dict) -> _MockStep:
    return _MockStep(json.dumps(payload))


def _build(steps: list[_MockStep]) -> Any:
    from lapis_pm.scout.runner import _build_trace_payload

    scaffold = _MockScaffold()
    cell_id = "test-cell-abc"
    cell_params: dict = {}
    seed = 7
    run_log = _MockRunLog(steps)
    director = _MockDirector()
    return _build_trace_payload(scaffold, cell_id, cell_params, seed, run_log, director)


# ---------------------------------------------------------------------------
# Shared well-formed baseline payload dict (used in golden test)
# ---------------------------------------------------------------------------

_WELL_FORMED = {
    "scenario_generated": "A well-formed scenario.",
    "execution_trace": "Step 1 → Step 2 → done.",
    "breaks_observed": [{"signature": "b1"}],
    "leverage_points": [{"description": "lp1"}],
    "surprises": ["surprise-one"],
    "drift_signals": [{"at_step": "s1"}],
    "tools_used": {"tool_a": 1},
    "tools_wished_for": [{"signature": "wish_fn()"}],
    "performance_assessment": "Nominal.",
}


# ---------------------------------------------------------------------------
# 1. execution_trace as list → newline-joined string, no TypeError
# ---------------------------------------------------------------------------


class TestExecutionTraceList:
    def test_list_joined_no_typeerror(self) -> None:
        """Live crash shape: execution_trace is a list of per-turn strings."""
        step = _make_step({
            **_WELL_FORMED,
            "execution_trace": ["turn-1 trace", "turn-2 trace", "turn-3 trace"],
        })
        payload = _build([step])
        assert payload.execution_trace == "turn-1 trace\nturn-2 trace\nturn-3 trace"

    def test_list_across_multiple_steps_joined_by_separator(self) -> None:
        """Multi-step: each step's coerced trace is still joined by '\n---\n'."""
        step_a = _make_step({**_WELL_FORMED, "execution_trace": ["a1", "a2"]})
        step_b = _make_step({**_WELL_FORMED, "execution_trace": ["b1", "b2"]})
        payload = _build([step_a, step_b])
        assert payload.execution_trace == "a1\na2\n---\nb1\nb2"

    def test_list_at_tick_index_3_and_4_no_raise(self) -> None:
        """Reproduce the live failing shapes: list at tick index 3 and 4."""
        steps = [
            _make_step({**_WELL_FORMED, "execution_trace": "normal-0"}),
            _make_step({**_WELL_FORMED, "execution_trace": "normal-1"}),
            _make_step({**_WELL_FORMED, "execution_trace": "normal-2"}),
            _make_step({**_WELL_FORMED, "execution_trace": ["list-at-3-part-a", "list-at-3-part-b"]}),
            _make_step({**_WELL_FORMED, "execution_trace": ["list-at-4-part-a", "list-at-4-part-b"]}),
        ]
        # Must not raise TypeError
        payload = _build(steps)
        assert isinstance(payload.execution_trace, str)
        assert "list-at-3-part-a" in payload.execution_trace
        assert "list-at-4-part-a" in payload.execution_trace


# ---------------------------------------------------------------------------
# 2. scenario_generated / performance_assessment as list → coerced to str
# ---------------------------------------------------------------------------


class TestStrFieldsCoercion:
    def test_scenario_generated_list_coerced(self) -> None:
        step = _make_step({**_WELL_FORMED, "scenario_generated": ["part-a", "part-b"]})
        payload = _build([step])
        assert payload.scenario_generated == "part-a\npart-b"

    def test_performance_assessment_list_coerced(self) -> None:
        step = _make_step({**_WELL_FORMED, "performance_assessment": ["assessment-1", "assessment-2"]})
        payload = _build([step])
        assert payload.performance_assessment == "assessment-1\nassessment-2"


# ---------------------------------------------------------------------------
# 3. List-typed field arriving as bare string → [string], not char-by-char
# ---------------------------------------------------------------------------


class TestListFieldsCoercion:
    def test_breaks_observed_string_not_char_split(self) -> None:
        step = _make_step({**_WELL_FORMED, "breaks_observed": "a-single-break"})
        payload = _build([step])
        assert payload.breaks_observed == ["a-single-break"], (
            f"Expected ['a-single-break'], got {payload.breaks_observed!r} — "
            "bare string must NOT be iterated char-by-char"
        )

    def test_leverage_points_string_not_char_split(self) -> None:
        step = _make_step({**_WELL_FORMED, "leverage_points": "lp-string"})
        payload = _build([step])
        assert payload.leverage_points == ["lp-string"]

    def test_surprises_string_not_char_split(self) -> None:
        step = _make_step({**_WELL_FORMED, "surprises": "surprise-str"})
        payload = _build([step])
        assert payload.surprises == ["surprise-str"]

    def test_drift_signals_string_not_char_split(self) -> None:
        step = _make_step({**_WELL_FORMED, "drift_signals": "drift-str"})
        payload = _build([step])
        assert payload.drift_signals == ["drift-str"]

    def test_tools_wished_for_string_not_char_split(self) -> None:
        step = _make_step({**_WELL_FORMED, "tools_wished_for": "wish-string"})
        payload = _build([step])
        assert payload.tools_wished_for == ["wish-string"]


# ---------------------------------------------------------------------------
# 4. All-well-formed (golden) case is unchanged
# ---------------------------------------------------------------------------


class TestWellFormedGolden:
    def test_golden_payload_unchanged(self) -> None:
        step = _make_step(_WELL_FORMED)
        payload = _build([step])
        assert payload.scenario_generated == "A well-formed scenario."
        assert payload.execution_trace == "Step 1 → Step 2 → done."
        assert payload.breaks_observed == [{"signature": "b1"}]
        assert payload.leverage_points == [{"description": "lp1"}]
        assert payload.surprises == ["surprise-one"]
        assert payload.drift_signals == [{"at_step": "s1"}]
        assert payload.tools_used == {"tool_a": 1}
        assert payload.tools_wished_for == [{"signature": "wish_fn()"}]
        assert payload.performance_assessment == "Nominal."

    def test_multi_step_join_separator_unchanged(self) -> None:
        steps = [
            _make_step({**_WELL_FORMED, "execution_trace": "trace-A"}),
            _make_step({**_WELL_FORMED, "execution_trace": "trace-B"}),
        ]
        payload = _build(steps)
        assert payload.execution_trace == "trace-A\n---\ntrace-B"


# ---------------------------------------------------------------------------
# 5. Coercion emits WARNING; well-formed path emits none
# ---------------------------------------------------------------------------


class TestCoercionLogging:
    def test_list_execution_trace_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        step = _make_step({**_WELL_FORMED, "execution_trace": ["x", "y"]})
        with caplog.at_level(logging.WARNING, logger="lapis_pm.scout.runner"):
            _build([step])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "execution_trace" in msg
        assert "list" in msg

    def test_warning_includes_cell_context(self, caplog: pytest.LogCaptureFixture) -> None:
        step = _make_step({**_WELL_FORMED, "execution_trace": ["x"]})
        with caplog.at_level(logging.WARNING, logger="lapis_pm.scout.runner"):
            _build([step])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "test-spec-coercion" in msg
        assert "test-cell-abc" in msg

    def test_str_field_as_list_logs_exactly_one_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        step = _make_step({**_WELL_FORMED, "scenario_generated": ["a", "b"]})
        with caplog.at_level(logging.WARNING, logger="lapis_pm.scout.runner"):
            _build([step])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "scenario_generated" in warnings[0].getMessage()

    def test_list_field_as_string_logs_exactly_one_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        step = _make_step({**_WELL_FORMED, "breaks_observed": "bare-string"})
        with caplog.at_level(logging.WARNING, logger="lapis_pm.scout.runner"):
            _build([step])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "breaks_observed" in warnings[0].getMessage()

    def test_well_formed_emits_no_warnings(self, caplog: pytest.LogCaptureFixture) -> None:
        step = _make_step(_WELL_FORMED)
        with caplog.at_level(logging.WARNING, logger="lapis_pm.scout.runner"):
            _build([step])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [], (
            f"Well-formed input must emit no coercion warnings; got: {[r.getMessage() for r in warnings]}"
        )
