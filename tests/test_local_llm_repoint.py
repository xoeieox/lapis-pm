"""Tests for lapis-pm-sh-literal-repoint-v0.

Covers the call-time env-resolved GravityWell defaults in
corroboration_adapter.py (D1) and local_reviewer_witness.py (D2), the
provenance stamping in trajectory.py (D3), and the chaos case (D4):
pointing the resolver at an unreachable endpoint degrades loudly, never
silently, and never by falling back to another hardcoded address.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.corroboration_adapter import (
    LapisPMReviewerAdapter,
    _IdentifierSubstrate,
    _llm_url,
)
from lapis_pm.local_reviewer_witness import (
    _default_endpoint,
    _default_model,
    run_local_reviewer_witness,
)
from lapis_pm.trajectory import _qwen_model_provenance

GW_DEFAULT = "http://203.0.113.11:8081/v1/chat/completions"
FAKE_DIFF = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n+x = 1\n"


# ---------------------------------------------------------------------------
# D1: corroboration_adapter._llm_url()
# ---------------------------------------------------------------------------

class TestCorroborationAdapterUrlResolution:

    def test_default_resolves_to_gravitywell(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        assert _llm_url() == GW_DEFAULT

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://example.test:9999/v1/chat/completions")
        assert _llm_url() == "http://example.test:9999/v1/chat/completions"

    def test_no_starhouse_literal_in_module(self):
        import lapis_pm.corroboration_adapter as mod
        import inspect
        src = inspect.getsource(mod)
        # StarHouse literal must not appear anywhere reachable as a request URL
        assert "203.0.113.12" not in src

    def test_score_uses_gw_default_when_env_unset(self, monkeypatch):
        """score() posts to the GravityWell default when LOCAL_LLM_URL is unset."""
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate(identifier="Foo", repo_hits=[], vault_hits=[])]
        llm_resp = {
            "choices": [{"message": {"content": json.dumps({
                "verdict": "clean", "claims": [], "summary": "ok",
            })}}]
        }
        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True) as mock_reach,
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200, json=lambda: llm_resp, raise_for_status=lambda: None,
            )
            adapter.score("content with `Foo`", substrates, "lapis-pm")

        # node_reachable and httpx.post were both called against the GW default
        assert mock_reach.call_args[0][0] == GW_DEFAULT
        assert mock_post.call_args[0][0] == GW_DEFAULT

    def test_score_degrades_loudly_on_unreachable_endpoint(self, monkeypatch):
        """Chaos case: unreachable endpoint -> loud 'uncertain' verdict naming the
        unreachable URL, never a silent failure and never a fallback to another
        hardcoded address."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://192.0.2.1:1/v1/chat/completions")  # TEST-NET-1, unreachable
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate(identifier="Foo", repo_hits=[], vault_hits=[])]

        with patch("lapis_pm.corroboration_adapter.node_reachable", return_value=False):
            result = adapter.score("content with `Foo`", substrates, "lapis-pm")

        assert result.verdict == "uncertain"
        assert "192.0.2.1" in result.notes
        assert "203.0.113.12" not in result.notes
        assert "203.0.113.11" not in result.notes  # no silent fallback to GW either


# ---------------------------------------------------------------------------
# D2: local_reviewer_witness._default_endpoint() / _default_model()
# ---------------------------------------------------------------------------

class TestLocalReviewerWitnessDefaults:

    def test_default_endpoint_resolves_to_gravitywell(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        assert _default_endpoint() == GW_DEFAULT

    def test_default_endpoint_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://example.test:9999/v1/chat/completions")
        assert _default_endpoint() == "http://example.test:9999/v1/chat/completions"

    def test_default_model_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        assert _default_model() is None

    def test_default_model_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_MODEL", "some-model")
        assert _default_model() == "some-model"

    def test_no_starhouse_literal_in_module(self):
        import lapis_pm.local_reviewer_witness as mod
        import inspect
        src = inspect.getsource(mod)
        assert "203.0.113.12" not in src

    def test_unset_model_omits_model_key_from_payload(self, monkeypatch):
        """model unset -> no 'model' key sent in the request body at all."""
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "model": "gravitywell-122b",
            "choices": [{"message": {"content": json.dumps(
                {"verdict": "clean", "issues": [], "confidence": 0.9}
            )}}],
        }
        resp.raise_for_status = MagicMock()

        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", return_value=resp) as mock_post,
        ):
            run_local_reviewer_witness(
                diff_text=FAKE_DIFF,
                repo="lapis-pm",
                pr_number=1,
                spec_summary="spec",
                claude_verdict={"verdict": "clean", "issues": [], "confidence": 0.9},
            )

        sent_body = mock_post.call_args.kwargs["json"]
        assert "model" not in sent_body
        # and the endpoint used was the GW default
        assert mock_post.call_args[0][0] == GW_DEFAULT

    def test_explicit_model_kwarg_still_included(self, monkeypatch):
        """Preserve existing keyword-argument signature: explicit model still works."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "model": "explicit-model",
            "choices": [{"message": {"content": json.dumps(
                {"verdict": "clean", "issues": [], "confidence": 0.9}
            )}}],
        }
        resp.raise_for_status = MagicMock()

        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", return_value=resp) as mock_post,
        ):
            run_local_reviewer_witness(
                diff_text=FAKE_DIFF,
                repo="lapis-pm",
                pr_number=1,
                spec_summary="spec",
                claude_verdict={"verdict": "clean", "issues": [], "confidence": 0.9},
                model="explicit-model",
                endpoint="http://explicit-endpoint.test:8081/v1/chat/completions",
            )

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["model"] == "explicit-model"
        assert mock_post.call_args[0][0] == "http://explicit-endpoint.test:8081/v1/chat/completions"

    def test_chaos_unreachable_endpoint_fails_loudly_not_silently(self, monkeypatch):
        """Chaos case: fully unreachable endpoint -> explicit failure result naming
        the unreachable URL, never a silent success and never a fallback address."""
        unreachable = "http://192.0.2.1:1/v1/chat/completions"
        with patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=False):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF,
                repo="lapis-pm",
                pr_number=1,
                spec_summary="spec",
                claude_verdict={"verdict": "clean", "issues": [], "confidence": 0.9},
                endpoint=unreachable,
            )

        assert result.verdict is None
        assert result.agreement == "local_failed"
        assert result.error is not None
        assert "192.0.2.1" in result.error
        assert "203.0.113.12" not in result.error


# ---------------------------------------------------------------------------
# D3: trajectory._qwen_model_provenance()
# ---------------------------------------------------------------------------

class TestTrajectoryModelProvenance:

    def test_unset_env_stamps_server_default(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        assert _qwen_model_provenance() == "server-default"

    def test_never_stamps_unknown(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        assert _qwen_model_provenance() != "unknown"

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_MODEL", "gravitywell-122b")
        assert _qwen_model_provenance() == "gravitywell-122b"

    def test_no_stale_model_name_default(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        assert _qwen_model_provenance() != "qwen3.6-35b-a3b"
