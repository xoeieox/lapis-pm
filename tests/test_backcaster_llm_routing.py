"""Tests: backcaster LLM routing — call_model_sync dispatch."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_qwen_routes_to_call_operator():
    """model='qwen' delegates to call_operator with 'qwen' as first arg."""
    from lapis_pm.backcaster.llm_routing import call_model_sync

    with patch("lapis_pm.backcaster.llm_routing.call_operator", return_value="result") as mock_op:
        result = call_model_sync("qwen", "prompt")

    mock_op.assert_called_once()
    assert mock_op.call_args.args[0] == "qwen"
    assert result == "result"


def test_opus_routes_to_call_claude_cli():
    """model='opus' calls call_claude_cli with model='claude-opus-4-7'."""
    from lapis_pm.backcaster.llm_routing import call_model_sync

    with patch("lapis_pm.backcaster.llm_routing.call_claude_cli", return_value="opus result") as mock_cli:
        result = call_model_sync("opus", "prompt")

    mock_cli.assert_called_once()
    kwargs = mock_cli.call_args.kwargs
    assert kwargs.get("model") == "claude-opus-4-7"
    assert result == "opus result"


def test_sonnet_routes_to_call_claude_cli():
    """model='sonnet' calls call_claude_cli with model='claude-sonnet-4-6'."""
    from lapis_pm.backcaster.llm_routing import call_model_sync

    with patch("lapis_pm.backcaster.llm_routing.call_claude_cli", return_value="sonnet result") as mock_cli:
        result = call_model_sync("sonnet", "prompt")

    mock_cli.assert_called_once()
    kwargs = mock_cli.call_args.kwargs
    assert kwargs.get("model") == "claude-sonnet-4-6"
    assert result == "sonnet result"


def test_opus_timeout_floor():
    """Caller passing timeout=60 with model='opus' gets timeout=300 enforced."""
    from lapis_pm.backcaster.llm_routing import _ANTHROPIC_TIMEOUT_FLOOR, call_model_sync

    with patch("lapis_pm.backcaster.llm_routing.call_claude_cli", return_value="ok") as mock_cli:
        call_model_sync("opus", "prompt", timeout=60)

    kwargs = mock_cli.call_args.kwargs
    assert kwargs.get("timeout") >= _ANTHROPIC_TIMEOUT_FLOOR
    assert kwargs.get("timeout") == 300


def test_gravitywell_routes_to_call_operator():
    """model='gravitywell' delegates to call_operator with 'gravitywell' as first arg and on_wake_fail forwarded."""
    from lapis_pm.backcaster.llm_routing import call_model_sync, _GRAVITYWELL_WAKE_FAIL

    with patch("lapis_pm.backcaster.llm_routing.call_operator", return_value="gravitywell result") as mock_op:
        result = call_model_sync("gravitywell", "prompt")

    mock_op.assert_called_once()
    assert mock_op.call_args.args[0] == "gravitywell"
    assert mock_op.call_args.kwargs.get("on_wake_fail") == _GRAVITYWELL_WAKE_FAIL
    assert result == "gravitywell result"


def test_gravitywell_call_operator_unavailable_returns_none():
    """If call_operator is None, call_model_sync('gravitywell', ...) returns None."""
    from lapis_pm.backcaster import llm_routing

    original_op = llm_routing.call_operator
    try:
        llm_routing.call_operator = None

        from lapis_pm.backcaster.llm_routing import call_model_sync

        result = call_model_sync("gravitywell", "prompt")
    finally:
        llm_routing.call_operator = original_op

    assert result is None


def test_import_error_returns_none():
    """If both call_operator and call_claude_cli are None, call_model_sync returns None."""
    from lapis_pm.backcaster import llm_routing

    original_op = llm_routing.call_operator
    original_cli = llm_routing.call_claude_cli
    try:
        llm_routing.call_operator = None
        llm_routing.call_claude_cli = None

        from lapis_pm.backcaster.llm_routing import call_model_sync

        result_qwen = call_model_sync("qwen", "prompt")
        result_opus = call_model_sync("opus", "prompt")
    finally:
        llm_routing.call_operator = original_op
        llm_routing.call_claude_cli = original_cli

    assert result_qwen is None
    assert result_opus is None
