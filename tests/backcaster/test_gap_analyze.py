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


def _synapse_hits_response() -> MagicMock:
    """Mock httpx.post response with a minimal valid hits payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "payload": {
            "hits": [
                {"ref": "test/doc", "excerpt": "Relevant excerpt."},
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

    # Build a minimal JSON gap response so call_operator returns something parseable
    gap_json = json.dumps({
        "what_exists": "something",
        "what_missing": "nothing",
        "what_miswired": "nothing identified",
    })

    with (
        patch("httpx.get", return_value=_healthz_response()) as mock_get,
        patch("httpx.post", return_value=_synapse_hits_response()) as mock_post,
        patch("agents_core.mem.MemoryStore") as mock_memstore,
        patch("lapis_pm.backcaster.gap_analyze.call_operator", return_value=gap_json),
    ):
        # MemoryStore() instantiation must succeed (mock returns a MagicMock instance)
        mock_memstore.return_value = MagicMock()
        mock_memstore.return_value.list_by_prefix = MagicMock(return_value=[])

        from lapis_pm.backcaster.gap_analyze import analyze_gaps

        gaps, degraded_paths, prompt_hash = analyze_gaps(
            preconditions=[_make_precondition()],
        )

    # Core assertion: neither retrieval path should be degraded
    assert degraded_paths == [], f"Expected no degraded paths, got: {degraded_paths}"

    # Probe and retrieval calls were attempted
    mock_get.assert_called_once()
    mock_post.assert_called_once()
