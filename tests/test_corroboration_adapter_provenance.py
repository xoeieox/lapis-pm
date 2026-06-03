"""Tests for LapisPMReviewerAdapter ProvenanceCapableAdapter conformance.

Coverage (per spec §Tests "Surface 2"):
1. Initial state: freshly-constructed adapter's score_provenance() returns {}.
2. Post-success: score_provenance() returns model + prompt_hash + upstream_calls
   after a successful score() call.
3. Post-failure: score_provenance() returns {} after score() hits the failure path.
4. Response-without-model: LLM response omits "model"; score() still returns
   CorroborationResult; score_provenance() returns dict with model=None.
5. Multiple calls: score_provenance() reflects the most-recent score() call.
6. Protocol conformance: isinstance(adapter, ProvenanceCapableAdapter) is True.
7. Envelope round-trip: corroborate_envelope() returns envelope whose
   provenance.model and prompt_hash come from score_provenance().
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.corroboration_adapter import (
    LapisPMReviewerAdapter,
    _IdentifierSubstrate,
)

# Fixture diff and substrates shared across tests
_DIFF = """\
--- a/lapis_pm/pm_core.py
+++ b/lapis_pm/pm_core.py
@@ -1,3 +1,5 @@
+def tick(target_id: str) -> None:
+    pass
"""

_SUBSTRATES = [
    _IdentifierSubstrate(
        identifier="tick",
        repo_hits=[{"file": "/repo/lapis_pm/pm_core.py", "line": "1", "text": "def tick"}],
        vault_hits=[],
    )
]

_LLM_RESPONSE_WITH_MODEL = {
    "model": "qwen3.6-35b-a3b",
    "choices": [{"message": {"content": json.dumps({
        "verdict": "clean",
        "claims": [{"identifier": "tick", "drift_class": "none", "notes": "ok"}],
        "summary": "All identifiers valid",
    })}}],
}

_LLM_RESPONSE_WITHOUT_MODEL = {
    "choices": [{"message": {"content": json.dumps({
        "verdict": "clean",
        "claims": [{"identifier": "tick", "drift_class": "none", "notes": "ok"}],
        "summary": "All identifiers valid",
    })}}],
    # "model" key intentionally absent
}


def _make_mock_post(resp_dict: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = resp_dict
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Test 1: Initial state
# ---------------------------------------------------------------------------

class TestInitialState:

    def test_score_provenance_empty_before_any_call(self):
        """Freshly-constructed adapter returns {} from score_provenance()."""
        adapter = LapisPMReviewerAdapter()
        assert adapter.score_provenance() == {}

    def test_last_fields_none_at_construction(self):
        """_last_model, _last_prompt_hash, _last_score_at are None at init."""
        adapter = LapisPMReviewerAdapter(repo_path="/repo")
        assert adapter._last_model is None
        assert adapter._last_prompt_hash is None
        assert adapter._last_score_at is None


# ---------------------------------------------------------------------------
# Test 2: Post-success
# ---------------------------------------------------------------------------

class TestPostSuccess:

    def test_score_provenance_populated_after_success(self):
        """score_provenance() returns model + prompt_hash + upstream_calls after success."""
        adapter = LapisPMReviewerAdapter()

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", return_value=_make_mock_post(_LLM_RESPONSE_WITH_MODEL)),
        ):
            adapter.score(_DIFF, _SUBSTRATES, "lapis-pm")

        prov = adapter.score_provenance()
        assert prov["model"] == "qwen3.6-35b-a3b"
        assert prov["prompt_hash"].startswith("sha256:")
        assert prov["upstream_calls"] == []

    def test_prompt_hash_matches_resolved_prompt(self):
        """prompt_hash is sha256 of the resolved prompt sent over the wire."""
        adapter = LapisPMReviewerAdapter()
        captured_prompt: list[str] = []

        original_post = __import__("httpx").post

        def capture_and_mock(url, *, json=None, timeout=None, **kw):
            if json and "messages" in json:
                captured_prompt.append(json["messages"][0]["content"])
            return _make_mock_post(_LLM_RESPONSE_WITH_MODEL)

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=capture_and_mock),
        ):
            adapter.score(_DIFF, _SUBSTRATES, "lapis-pm")

        assert captured_prompt, "Expected httpx.post to be called"
        expected_hash = "sha256:" + hashlib.sha256(
            captured_prompt[0].encode("utf-8")
        ).hexdigest()
        prov = adapter.score_provenance()
        assert prov["prompt_hash"] == expected_hash


# ---------------------------------------------------------------------------
# Test 3: Post-failure
# ---------------------------------------------------------------------------

class TestPostFailure:

    def test_score_provenance_empty_after_llm_failure(self):
        """score_provenance() returns {} after score() hits the failure path."""
        adapter = LapisPMReviewerAdapter()

        with patch("httpx.post", side_effect=ConnectionError("refused")):
            result = adapter.score(_DIFF, _SUBSTRATES, "lapis-pm")

        assert result.verdict == "uncertain"
        assert adapter.score_provenance() == {}

    def test_last_fields_cleared_after_failure(self):
        """_last_model and _last_prompt_hash are None after failure."""
        adapter = LapisPMReviewerAdapter()

        with patch("httpx.post", side_effect=ConnectionError("refused")):
            adapter.score(_DIFF, _SUBSTRATES, "lapis-pm")

        assert adapter._last_model is None
        assert adapter._last_prompt_hash is None


# ---------------------------------------------------------------------------
# Test 4: Response without "model" field
# ---------------------------------------------------------------------------

class TestResponseWithoutModel:

    def test_score_succeeds_when_model_field_absent(self):
        """score() returns CorroborationResult even if LLM response omits 'model'."""
        adapter = LapisPMReviewerAdapter()

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", return_value=_make_mock_post(_LLM_RESPONSE_WITHOUT_MODEL)),
        ):
            result = adapter.score(_DIFF, _SUBSTRATES, "lapis-pm")

        assert result.verdict == "clean"

    def test_score_provenance_model_none_when_field_absent(self):
        """score_provenance() returns model=None when LLM response omits 'model'."""
        adapter = LapisPMReviewerAdapter()

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", return_value=_make_mock_post(_LLM_RESPONSE_WITHOUT_MODEL)),
        ):
            adapter.score(_DIFF, _SUBSTRATES, "lapis-pm")

        prov = adapter.score_provenance()
        assert "model" in prov
        assert prov["model"] is None
        # prompt_hash is still populated (we still sent the prompt)
        assert prov["prompt_hash"] is not None
        assert prov["prompt_hash"].startswith("sha256:")


# ---------------------------------------------------------------------------
# Test 5: Multiple calls — reflects most recent
# ---------------------------------------------------------------------------

class TestMultipleCalls:

    def test_score_provenance_reflects_most_recent_call(self):
        """After two score() calls, score_provenance() reflects the second."""
        adapter = LapisPMReviewerAdapter()

        resp_first = {
            "model": "model-first",
            "choices": [{"message": {"content": json.dumps({
                "verdict": "clean", "claims": [], "summary": "first",
            })}}],
        }
        resp_second = {
            "model": "qwen3.6-35b-a3b",
            "choices": [{"message": {"content": json.dumps({
                "verdict": "clean", "claims": [], "summary": "second",
            })}}],
        }

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", return_value=_make_mock_post(resp_first)),
        ):
            adapter.score(_DIFF, _SUBSTRATES, "lapis-pm")
        prov_after_first = adapter.score_provenance()

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", return_value=_make_mock_post(resp_second)),
        ):
            adapter.score(_DIFF, _SUBSTRATES, "lapis-pm")
        prov_after_second = adapter.score_provenance()

        assert prov_after_first["model"] == "model-first"
        assert prov_after_second["model"] == "qwen3.6-35b-a3b"
        # prompt hashes differ because the substrate text changes each call
        # (they may be equal here since same diff/substrates — that's fine,
        # the important assertion is model reflects the most-recent call)


# ---------------------------------------------------------------------------
# Test 6: Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:

    def test_isinstance_provenance_capable_adapter(self):
        """isinstance(adapter, ProvenanceCapableAdapter) is True."""
        from archetypes_core.corroboration import ProvenanceCapableAdapter

        adapter = LapisPMReviewerAdapter()
        assert isinstance(adapter, ProvenanceCapableAdapter), (
            "LapisPMReviewerAdapter must satisfy the ProvenanceCapableAdapter Protocol"
        )

    def test_score_provenance_is_callable(self):
        """score_provenance is present and callable."""
        adapter = LapisPMReviewerAdapter()
        assert callable(adapter.score_provenance)

    def test_scope_id_property_present(self):
        """scope_id attribute is present (SubstrateAdapter requirement)."""
        adapter = LapisPMReviewerAdapter()
        assert isinstance(adapter.scope_id, str)


# ---------------------------------------------------------------------------
# Test 7: Envelope round-trip
# ---------------------------------------------------------------------------

class TestEnvelopeRoundTrip:

    def test_envelope_provenance_reflects_adapter(self):
        """corroborate_envelope() envelope's model + prompt_hash match score_provenance()."""
        from archetypes_core.corroboration import (
            FreshnessBudget,
            CorroborationResult as CoreResult,
            corroborate_envelope,
        )

        adapter = LapisPMReviewerAdapter(repo_path="/fake/repo")

        # Step 1: run score() to populate provenance state
        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", return_value=_make_mock_post(_LLM_RESPONSE_WITH_MODEL)),
        ):
            adapter.score(_DIFF, _SUBSTRATES, "lapis-pm")

        expected_prov = adapter.score_provenance()
        assert expected_prov["model"] == "qwen3.6-35b-a3b"

        # Step 2: call corroborate_envelope with corroborate mocked so we don't
        # have to satisfy the full SubstrateAdapter.retrieve/score call signature
        dummy_result = CoreResult(
            verdict="clean",
            claim="all identifiers valid",
            citations=[],
            freshness_stamp=datetime.now(timezone.utc),
            scope_id="repo:lapis-pm",
        )

        with patch("archetypes_core.corroboration.corroborate", return_value=dummy_result):
            envelope = corroborate_envelope(
                "some diff claim",
                adapter,
                FreshnessBudget.HOUR,
                agent_id="lapis-pm/reviewer",
            )

        assert envelope.provenance.model == "qwen3.6-35b-a3b"
        assert envelope.provenance.prompt_hash == expected_prov["prompt_hash"]

    def test_envelope_provenance_empty_when_no_prior_score(self):
        """Envelope's model + prompt_hash are None when no score() has run."""
        from archetypes_core.corroboration import (
            FreshnessBudget,
            CorroborationResult as CoreResult,
            corroborate_envelope,
        )

        adapter = LapisPMReviewerAdapter()
        # Do NOT call score() first

        dummy_result = CoreResult(
            verdict="uncertain",
            claim="",
            citations=[],
            freshness_stamp=datetime.now(timezone.utc),
            scope_id="repo:lapis-pm",
        )

        with patch("archetypes_core.corroboration.corroborate", return_value=dummy_result):
            envelope = corroborate_envelope(
                "claim",
                adapter,
                FreshnessBudget.HOUR,
                agent_id="lapis-pm/reviewer",
            )

        assert envelope.provenance.model is None
        assert envelope.provenance.prompt_hash is None
