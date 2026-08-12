"""Tests for lapis_pm.completion_text (lapis-pm-corroboration-thinking-parse-
and-truncation-loudness-v0).

Coverage (DoD #1, #2):
1. The real reproduction: content=None, reasoning populated, finish_reason
   "length" — yields usable text, truncated=True, never raises.
2. Field-precedence fixtures: content-only, reasoning-only, reasoning_content-
   only, none-of-the-three.
3. finish_reason "stop" vs "length".
"""

from __future__ import annotations

from lapis_pm.completion_text import (
    NO_USABLE_TEXT,
    ExtractedText,
    extract_completion_text,
)


# ---------------------------------------------------------------------------
# Test 1: the real reproduction (2026-08-12 13:15 PT, GravityWell :8081)
# ---------------------------------------------------------------------------

class TestRealReproduction:

    def test_none_content_with_reasoning_and_length_does_not_raise(self):
        """content: None, reasoning populated, finish_reason: 'length' — the
        exact shape reproduced live against GravityWell. Pre-fix, this raised
        AttributeError on `.strip()`; must not raise here, and must yield the
        reasoning text with truncated=True."""
        message = {"content": None, "reasoning": "some partial reasoning text..."}
        result = extract_completion_text(message, "length")

        assert result.text == "some partial reasoning text..."
        assert result.truncated is True
        assert result.has_text is True


# ---------------------------------------------------------------------------
# Test 2: field-precedence fixtures
# ---------------------------------------------------------------------------

class TestFieldPrecedence:

    def test_content_only(self):
        result = extract_completion_text({"content": "the answer"}, "stop")
        assert result.text == "the answer"
        assert result.has_text is True

    def test_reasoning_only_vllm_shape(self):
        """content absent/empty, reasoning (vLLM convention) populated."""
        result = extract_completion_text({"content": "", "reasoning": "thinking..."}, "stop")
        assert result.text == "thinking..."
        assert result.has_text is True

    def test_reasoning_content_only_llamacpp_shape(self):
        """content absent/empty, reasoning_content (llama.cpp convention) populated."""
        result = extract_completion_text({"content": None, "reasoning_content": "llamacpp thinking"}, "stop")
        assert result.text == "llamacpp thinking"
        assert result.has_text is True

    def test_content_wins_over_reasoning_when_both_present(self):
        result = extract_completion_text(
            {"content": "final answer", "reasoning": "scratch work"}, "stop",
        )
        assert result.text == "final answer"

    def test_reasoning_wins_over_reasoning_content_when_both_present(self):
        result = extract_completion_text(
            {"content": "", "reasoning": "vllm reasoning", "reasoning_content": "llamacpp reasoning"},
            "stop",
        )
        assert result.text == "vllm reasoning"

    def test_none_of_the_three_is_named_not_exception_not_empty_string_surprise(self):
        """No usable text anywhere is a distinct, named condition (has_text=False),
        not an exception and not silently indistinguishable from '' meaning
        something else."""
        result = extract_completion_text({"content": None, "reasoning": None}, "stop")
        assert isinstance(result, ExtractedText)
        assert result.has_text is False
        assert result.text == ""

    def test_none_message_does_not_raise(self):
        result = extract_completion_text(None, "stop")
        assert result.has_text is False
        assert result.truncated is False

    def test_empty_dict_message_does_not_raise(self):
        result = extract_completion_text({}, None)
        assert result.has_text is False


# ---------------------------------------------------------------------------
# Test 3: finish_reason stop vs length
# ---------------------------------------------------------------------------

class TestFinishReason:

    def test_stop_is_not_truncated(self):
        result = extract_completion_text({"content": "done"}, "stop")
        assert result.truncated is False

    def test_length_is_truncated(self):
        result = extract_completion_text({"content": "cut off"}, "length")
        assert result.truncated is True

    def test_tool_calls_is_not_truncated(self):
        result = extract_completion_text({"content": ""}, "tool_calls")
        assert result.truncated is False

    def test_none_finish_reason_is_not_truncated(self):
        result = extract_completion_text({"content": "x"}, None)
        assert result.truncated is False

    def test_truncated_with_no_usable_text_at_all(self):
        """Truncated AND no text in any field — both facts are independently
        readable off the result (has_text=False, truncated=True)."""
        result = extract_completion_text({"content": None, "reasoning": None}, "length")
        assert result.truncated is True
        assert result.has_text is False


def test_no_usable_text_constant_is_a_plain_string():
    """NO_USABLE_TEXT is a stable, greppable label — not an exception class."""
    assert isinstance(NO_USABLE_TEXT, str)
    assert NO_USABLE_TEXT
