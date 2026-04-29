"""Tests for shaper's queue-routing decision.

Opus/Sonnet/Haiku → ClaudeQueue (Anthropic API, no local-GPU gating).
Qwen → GPUQueue (local GPU, TOU-paused 4-9 PM).
LAPIS_PM_FORCE_GPU_QUEUE=1 forces everything to GPUQueue (rollback knob).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import shaper
from lapis_pm.shaper import ShapedAgent


def _fake_agent(name: str, model: str) -> ShapedAgent:
    return ShapedAgent(
        name=name,
        chub_bundles=[],
        system_template="ignored",
        model=model,
        timeout_s=60,
    )


@pytest.fixture
def shaper_mocks(tmp_path, monkeypatch):
    monkeypatch.setattr(shaper, "SPEC_DIR", tmp_path)
    monkeypatch.setattr(shaper, "_resolve_repo_cwd", lambda repo: "/tmp/fake-cwd")
    monkeypatch.setattr(shaper, "_compose_system", lambda agent, vars_: "SYSTEM")
    monkeypatch.delenv("LAPIS_PM_FORCE_GPU_QUEUE", raising=False)

    claude_q = MagicMock()
    claude_q._generate_id.return_value = "claude_task_id"
    claude_q.submit.return_value = None
    gpu_q = MagicMock()
    gpu_q.submit.return_value = "gpu_task_id"

    monkeypatch.setattr(shaper, "ClaudeQueue", lambda: claude_q)
    monkeypatch.setattr(shaper, "GPUQueue", lambda: gpu_q)

    return claude_q, gpu_q


def _dispatch(model: str) -> None:
    agent = _fake_agent(name="reviewer" if model == "opus" else "fixer", model=model)
    with patch.object(shaper, "get_agent", return_value=agent):
        shaper.dispatch(
            agent_type=agent.name,
            target_id="t-1",
            user_prompt="do the thing",
            vars_={"repo": "lapis-pm"},
        )


@pytest.mark.parametrize("model", ["opus", "sonnet", "haiku"])
def test_anthropic_models_route_to_claude_queue(shaper_mocks, model):
    claude_q, gpu_q = shaper_mocks
    _dispatch(model)
    assert claude_q.submit.called, f"{model} should route to ClaudeQueue"
    assert not gpu_q.submit.called, f"{model} must not touch GPUQueue"


def test_qwen_routes_to_gpu_queue(shaper_mocks):
    claude_q, gpu_q = shaper_mocks
    _dispatch("qwen3.6-35b-a3b")
    assert gpu_q.submit.called, "qwen should route to GPUQueue"
    assert not claude_q.submit.called, "qwen must not touch ClaudeQueue"


@pytest.mark.parametrize("model", ["opus", "sonnet", "haiku"])
def test_force_gpu_queue_env_overrides(shaper_mocks, monkeypatch, model):
    claude_q, gpu_q = shaper_mocks
    monkeypatch.setenv("LAPIS_PM_FORCE_GPU_QUEUE", "1")
    _dispatch(model)
    assert gpu_q.submit.called, f"{model} should fall back to GPUQueue when forced"
    assert not claude_q.submit.called
