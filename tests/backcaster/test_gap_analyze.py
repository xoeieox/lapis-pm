"""Tests for gap_analyze non-stub probe path.

Verifies that when Synapse and mem.db are reachable, neither appears in
degraded_paths (i.e. the MemoryStore import fix and URL fix both hold).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from lapis_pm.backcaster.schema import Precondition


def _make_precondition(pid: str = "test-01") -> Precondition:
    return Precondition(id=pid, axis="infrastructural", statement="Test precondition.")


def _synapse_content_response() -> MagicMock:
    """Mock httpx.post response with hits containing 'content' field (real Synapse shape)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "payload": {
            "hits": [
                {"id": "doc/synapse-real-id", "content": "Real corpus passage about civic theory."},
            ]
        }
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _healthz_response() -> MagicMock:
    """Mock httpx.get response for the Synapse probe."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    return mock_resp


def test_non_stub_probe_no_degraded_paths(monkeypatch):
    """With mocked Synapse + MemoryStore both healthy, degraded_paths must be empty."""
    monkeypatch.delenv("BACKCASTER_STUB", raising=False)
    monkeypatch.delenv("SYNAPSE_URL", raising=False)

    gap_json = json.dumps({
        "what_exists": "something",
        "what_missing": "nothing",
        "what_miswired": "nothing identified",
    })

    with (
        patch("httpx.get", return_value=_healthz_response()) as mock_get,
        patch("httpx.post", return_value=_synapse_content_response()) as mock_post,
        patch("agents_core.mem.MemoryStore") as mock_memstore,
        patch("lapis_pm.backcaster.llm_routing.call_operator", return_value=gap_json),
    ):
        mock_instance = MagicMock()
        mock_instance.search = MagicMock(return_value=[])
        mock_memstore.return_value = mock_instance

        from lapis_pm.backcaster.gap_analyze import analyze_gaps

        gaps, degraded_paths, prompt_hash = analyze_gaps(
            preconditions=[_make_precondition()],
        )

    assert degraded_paths == [], f"Expected no degraded paths, got: {degraded_paths}"
    mock_get.assert_called_once()
    mock_post.assert_called_once()


def test_retrieve_synapse_content_field_reaches_prompt(monkeypatch):
    """A1: Synapse hits with 'content' field must inject that content into the LLM prompt."""
    monkeypatch.delenv("BACKCASTER_STUB", raising=False)
    monkeypatch.delenv("SYNAPSE_URL", raising=False)

    gap_json = json.dumps({
        "what_exists": "something",
        "what_missing": "nothing",
        "what_miswired": "nothing identified",
    })

    captured_prompts: list[str] = []

    def fake_call_model_sync(model, *, prompt, system, json_mode, timeout):
        captured_prompts.append(prompt)
        return gap_json

    import lapis_pm.backcaster.gap_analyze as mod

    with (
        patch("httpx.get", return_value=_healthz_response()),
        patch("httpx.post", return_value=_synapse_content_response()),
        patch("agents_core.mem.MemoryStore") as mock_memstore,
        patch.object(mod, "call_model_sync", side_effect=fake_call_model_sync),
    ):
        mock_instance = MagicMock()
        mock_instance.search = MagicMock(return_value=[])
        mock_memstore.return_value = mock_instance

        gaps, degraded_paths, _ = mod.analyze_gaps(
            preconditions=[_make_precondition()],
        )

    assert degraded_paths == []
    assert len(captured_prompts) == 1
    # The corpus passage content must appear in the prompt
    assert "Real corpus passage about civic theory." in captured_prompts[0], (
        f"Expected content in prompt, got: {captured_prompts[0][:500]}"
    )


def test_retrieve_mem_uses_search_not_list_by_prefix():
    """A2: _retrieve_mem must call MemoryStore.search and return content from 'content' field.

    list_by_prefix must NOT be called.
    """
    from lapis_pm.backcaster.gap_analyze import _retrieve_mem

    mem_items = [
        {"key": "architecture/foo", "content": "Relevant architecture note."},
        {"key": "project/bar", "content": "Relevant project note."},
    ]

    mock_mem_instance = MagicMock()
    mock_mem_instance.search = MagicMock(return_value=mem_items)
    mock_mem_instance.list_by_prefix = MagicMock()

    with patch("agents_core.mem.MemoryStore", return_value=mock_mem_instance):
        citations, context_text = _retrieve_mem("some precondition query")

    # search must have been called with the query
    mock_mem_instance.search.assert_called_once_with("some precondition query", limit=5)
    # list_by_prefix must NOT be called
    mock_mem_instance.list_by_prefix.assert_not_called()
    # context_text must contain the content from both items
    assert "Relevant architecture note." in context_text
    assert "Relevant project note." in context_text
    # citations must be populated
    assert len(citations) == 2
    assert citations[0].ref == "architecture/foo"
    assert citations[1].ref == "project/bar"
