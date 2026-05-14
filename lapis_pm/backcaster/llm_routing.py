"""Backcaster LLM routing — model-aware sync dispatch.

Qwen  → call_operator (local backend, synchronous)
Opus/Sonnet/Haiku → call_claude_cli (claude -p subprocess, synchronous)

Timeout auto-bump: Anthropic-family calls enforce a floor of 300s since
Opus stages can take 60-120s each and the stage default (120s) is too tight.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

ANTHROPIC_MODELS = {"opus", "sonnet", "haiku"}

_MODEL_NAMES = {
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
    "haiku":  "claude-haiku-4-5-20251001",
}

_ANTHROPIC_TIMEOUT_FLOOR = 300  # seconds

try:
    from agents_core.llm import call_operator
except ImportError:
    call_operator = None  # type: ignore

try:
    from agents_core.llm import call_claude_cli
except ImportError:
    call_claude_cli = None  # type: ignore


def call_model_sync(
    model: str,
    prompt: str,
    system: str | None = None,
    json_mode: bool = False,
    timeout: int = 120,
) -> str | None:
    """Route a synchronous LLM call to the appropriate backend.

    Returns the response string, or None if the call could not be made.
    """
    if model == "qwen":
        if call_operator is None:
            log.warning("call_model_sync: call_operator unavailable; returning None")
            return None
        return call_operator(
            "qwen",
            prompt=prompt,
            system=system,
            json_mode=json_mode,
            timeout=timeout,
        )
    elif model in ANTHROPIC_MODELS:
        if call_claude_cli is None:
            log.warning("call_model_sync: call_claude_cli unavailable; returning None")
            return None
        resolved = _MODEL_NAMES[model]
        return call_claude_cli(
            prompt,
            model=resolved,
            system=system or "",
            json_mode=json_mode,
            timeout=max(timeout, _ANTHROPIC_TIMEOUT_FLOOR),
        )
    else:
        log.warning("call_model_sync: unknown model %r; returning None", model)
        return None
