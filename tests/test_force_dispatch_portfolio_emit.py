"""Tests for force_dispatch portfolio emission — pm-wire-disp-640657.

Coverage:
1. emit_decision_dispatch is called on successful force_dispatch
2. fragment_id = "kickoff" for fixer when this is the first dispatch (len==1)
3. fragment_id = "tick" for fixer when dispatched list has >1 entry
4. fragment_id = "review-cycle" for reviewer agent type
5. fragment_id = "human-judgment" for brief agent type
6. fragment_id = agent_type for unknown agent types
7. expert_chosen is mapped via _TIER_MAP (haiku, sonnet, opus, qwen variants)
8. expert_chosen passthrough for unknown models
9. emit failure is non-fatal (exception swallowed; task_id still returned)
10. intent truncated to 200 chars in intent_summary
"""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import MagicMock, call, patch

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(target_id: str = "test-tid", pm_bound: bool = True, pm_repo: str = "myrepo") -> MagicMock:
    t = MagicMock()
    t.pm_bound = pm_bound
    t.pm_repo = pm_repo
    return t


def _make_dispatch_result(task_id: str = "task-abc") -> MagicMock:
    res = MagicMock()
    res.task_id = task_id
    res.spec_id = "spec-abc"
    res.output_path = "/tmp/fake-output"
    return res


def _make_agent(model: str) -> MagicMock:
    a = MagicMock()
    a.model = model
    return a


def _call_force_dispatch(
    target_id: str = "test-tid",
    agent_type: str = "fixer",
    intent: str = "do the thing",
    model: str = "haiku",
    dispatched_list: list | None = None,
) -> tuple[str, MagicMock]:
    """Call force_dispatch with full mocking. Returns (task_id, emit_mock)."""
    if dispatched_list is None:
        dispatched_list = [{"gpu_id": "existing-task"}]  # 1 entry → kickoff

    emit_mock = MagicMock(return_value="router/lapis-pm/decisions/fake-key")

    with (
        patch("lapis_pm.pm_core.TargetStore") as mock_store_cls,
        patch.object(pm_core._SHAPER, "dispatch", return_value=_make_dispatch_result()),
        patch.object(pm_core._SHAPER, "get_agent", return_value=_make_agent(model)),
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec summary"),
        patch("lapis_pm.pm_core.episodic.write_dispatch"),
        patch("lapis_pm.pm_core.append_dispatched"),
        patch("lapis_pm.pm_core.load_dispatched", return_value=dispatched_list),
        patch("lapis_pm.router_portfolio.emit_decision_dispatch", emit_mock),
    ):
        mock_store_cls.return_value.get.return_value = _make_target(target_id)
        task_id = pm_core.force_dispatch(target_id, agent_type, intent)

    return task_id, emit_mock


# ---------------------------------------------------------------------------
# 1. emit_decision_dispatch is called on success
# ---------------------------------------------------------------------------

class TestEmitCalledOnSuccess:
    def test_emit_called_once(self) -> None:
        """force_dispatch calls emit_decision_dispatch exactly once."""
        _, emit_mock = _call_force_dispatch()
        assert emit_mock.call_count == 1

    def test_emit_receives_target_id(self) -> None:
        """emit_decision_dispatch receives the correct target_id."""
        _, emit_mock = _call_force_dispatch(target_id="my-target")
        kwargs = emit_mock.call_args.kwargs
        assert kwargs["target_id"] == "my-target"

    def test_task_id_returned(self) -> None:
        """force_dispatch returns the task_id from the shaper result."""
        task_id, _ = _call_force_dispatch()
        assert task_id == "task-abc"


# ---------------------------------------------------------------------------
# 2-3. fragment_id derivation for fixer
# ---------------------------------------------------------------------------

class TestFragmentIdFixer:
    def test_kickoff_when_first_dispatch(self) -> None:
        """fixer + dispatched list with 1 entry → fragment_id='kickoff'."""
        _, emit_mock = _call_force_dispatch(
            agent_type="fixer",
            dispatched_list=[{"gpu_id": "t1"}],  # len == 1
        )
        assert emit_mock.call_args.kwargs["fragment_id"] == "kickoff"

    def test_tick_when_not_first_dispatch(self) -> None:
        """fixer + dispatched list with 2+ entries → fragment_id='tick'."""
        _, emit_mock = _call_force_dispatch(
            agent_type="fixer",
            dispatched_list=[{"gpu_id": "t1"}, {"gpu_id": "t2"}],  # len == 2
        )
        assert emit_mock.call_args.kwargs["fragment_id"] == "tick"

    def test_tick_when_many_dispatches(self) -> None:
        """fixer + dispatched list with many entries → fragment_id='tick'."""
        many = [{"gpu_id": f"t{i}"} for i in range(5)]
        _, emit_mock = _call_force_dispatch(agent_type="fixer", dispatched_list=many)
        assert emit_mock.call_args.kwargs["fragment_id"] == "tick"


# ---------------------------------------------------------------------------
# 4-6. fragment_id derivation for other agent types
# ---------------------------------------------------------------------------

class TestFragmentIdByAgentType:
    def test_reviewer_gets_review_cycle(self) -> None:
        _, emit_mock = _call_force_dispatch(agent_type="reviewer")
        assert emit_mock.call_args.kwargs["fragment_id"] == "review-cycle"

    def test_brief_gets_human_judgment(self) -> None:
        _, emit_mock = _call_force_dispatch(agent_type="brief")
        assert emit_mock.call_args.kwargs["fragment_id"] == "human-judgment"

    def test_unknown_agent_gets_agent_type_as_fragment(self) -> None:
        _, emit_mock = _call_force_dispatch(agent_type="scout")
        assert emit_mock.call_args.kwargs["fragment_id"] == "scout"


# ---------------------------------------------------------------------------
# 7-8. expert_chosen TIER_MAP mapping
# ---------------------------------------------------------------------------

class TestExpertChosenTierMap:
    @pytest.mark.parametrize("model,expected", [
        ("haiku", "haiku"),
        ("sonnet", "sonnet"),
        ("opus", "opus"),
        ("qwen-3.6-35b-a3b", "qwen-local"),
        ("qwen3.6-35b-a3b", "qwen-local"),
    ])
    def test_known_models_map_correctly(self, model: str, expected: str) -> None:
        _, emit_mock = _call_force_dispatch(model=model)
        assert emit_mock.call_args.kwargs["expert_chosen"] == expected

    def test_unknown_model_passes_through_lowercased(self) -> None:
        """Unknown model strings are lowercased and passed through."""
        _, emit_mock = _call_force_dispatch(model="Claude-Sonnet-4")
        assert emit_mock.call_args.kwargs["expert_chosen"] == "claude-sonnet-4"

    def test_case_insensitive_tier_map_lookup(self) -> None:
        """Model strings are lowercased before TIER_MAP lookup."""
        _, emit_mock = _call_force_dispatch(model="HAIKU")
        assert emit_mock.call_args.kwargs["expert_chosen"] == "haiku"


# ---------------------------------------------------------------------------
# 9. Emit failure is non-fatal
# ---------------------------------------------------------------------------

class TestEmitFailureNonFatal:
    def test_emit_error_does_not_raise(self) -> None:
        """When emit_decision_dispatch raises, force_dispatch still returns task_id."""
        emit_mock = MagicMock(side_effect=RuntimeError("mem unavailable"))

        with (
            patch("lapis_pm.pm_core.TargetStore") as mock_store_cls,
            patch.object(pm_core._SHAPER, "dispatch", return_value=_make_dispatch_result("task-xyz")),
            patch.object(pm_core._SHAPER, "get_agent", return_value=_make_agent("haiku")),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[{"gpu_id": "t1"}]),
            patch("lapis_pm.router_portfolio.emit_decision_dispatch", emit_mock),
        ):
            mock_store_cls.return_value.get.return_value = _make_target()
            task_id = pm_core.force_dispatch("test-tid", "fixer", "intent")

        assert task_id == "task-xyz"

    def test_emit_error_logged_to_stderr(self, capsys) -> None:
        """When emit raises, the failure is logged with [router-portfolio:emit-failed]."""
        emit_mock = MagicMock(side_effect=RuntimeError("boom"))

        with (
            patch("lapis_pm.pm_core.TargetStore") as mock_store_cls,
            patch.object(pm_core._SHAPER, "dispatch", return_value=_make_dispatch_result()),
            patch.object(pm_core._SHAPER, "get_agent", return_value=_make_agent("haiku")),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[{"gpu_id": "t1"}]),
            patch("lapis_pm.router_portfolio.emit_decision_dispatch", emit_mock),
        ):
            mock_store_cls.return_value.get.return_value = _make_target()
            pm_core.force_dispatch("test-tid", "fixer", "intent")

        captured = capsys.readouterr()
        assert "[router-portfolio:emit-failed]" in captured.err
        assert "dispatch" in captured.err


# ---------------------------------------------------------------------------
# 10. Intent truncation
# ---------------------------------------------------------------------------

class TestIntentTruncation:
    def test_intent_truncated_to_200_chars(self) -> None:
        """intent_summary passed to emit is capped at 200 characters."""
        long_intent = "x" * 300
        _, emit_mock = _call_force_dispatch(intent=long_intent)
        summary = emit_mock.call_args.kwargs["intent_summary"]
        assert len(summary) == 200
        assert summary == "x" * 200

    def test_short_intent_passed_unchanged(self) -> None:
        """intent shorter than 200 chars is passed through unchanged."""
        short_intent = "fix the widget"
        _, emit_mock = _call_force_dispatch(intent=short_intent)
        assert emit_mock.call_args.kwargs["intent_summary"] == short_intent
