"""Tests for lapis_pm.local_reviewer_witness (local-reviewer-witness-v0).

Coverage per spec §Tests:
1.  test_run_witness_success_with_grammar
2.  test_run_witness_json_parse_failure_when_grammar_off
3.  test_run_witness_endpoint_unavailable
4.  test_run_witness_timeout
5.  test_run_witness_diff_truncation
6.  test_agreement_classification (parametrized)
7.  test_witness_field_attached_to_reviewer_comment
8.  test_provenance_fields_present
9.  test_graceful_degradation_to_json_object
10. test_divergence_observation_written_on_major_diverge
11. test_no_divergence_observation_on_minor_diverge
12. test_no_divergence_observation_on_agree
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from lapis_pm.local_reviewer_witness import (
    LocalReviewerWitnessResult,
    _format_divergence_note,
    _score_agreement,
    run_local_reviewer_witness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_httpx_response(verdict: str, issues: list, confidence: float, *,
                         model_name: str = "qwen3.6-35b-a3b.gguf",
                         status_code: int = 200) -> MagicMock:
    content = json.dumps({"verdict": verdict, "issues": issues, "confidence": confidence})
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "model": model_name,
        "choices": [{"message": {"content": content}}],
    }
    resp.raise_for_status = MagicMock()
    return resp


_CLAUDE_CLEAN = {"verdict": "clean", "issues": [], "confidence": 0.95}
_CLAUDE_FIXABLE = {"verdict": "fixable",
                   "issues": [{"severity": "med", "path": "foo.py", "note": "x"}],
                   "confidence": 0.80}
_CLAUDE_NEEDS_HUMAN = {"verdict": "needs-human", "issues": [], "confidence": 0.60}

FAKE_DIFF = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n+x = 1\n"


# ---------------------------------------------------------------------------
# Test 1: success with grammar
# ---------------------------------------------------------------------------

class TestRunWitnessSuccessWithGrammar:

    def test_all_fields_populated(self):
        """Mock httpx to return a valid verdict; assert all fields populated + agreement correct."""
        resp = _make_httpx_response("clean", [], 0.95)
        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", return_value=resp),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF,
                repo="lapis-pm",
                pr_number=7,
                spec_summary="spec",
                claude_verdict=_CLAUDE_CLEAN,
            )

        assert result.verdict == "clean"
        assert result.issues == []
        assert result.confidence == pytest.approx(0.95)
        assert result.agreement == "agree"
        assert result.json_valid is True
        assert result.model == "qwen3.6-35b-a3b.gguf"
        assert result.prompt_hash.startswith("sha256:")
        assert result.dispatched_at  # non-empty ISO timestamp
        assert result.error is None
        assert result.latency_ms >= 0

    def test_to_dict_is_json_serialisable(self):
        resp = _make_httpx_response("clean", [], 0.95)
        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", return_value=resp),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
            )
        d = result.to_dict()
        json.dumps(d)  # must not raise
        assert d["verdict"] == "clean"
        assert d["agreement"] == "agree"


# ---------------------------------------------------------------------------
# Test 2: JSON parse failure when grammar off
# ---------------------------------------------------------------------------

class TestRunWitnessJsonParseFailureWhenGrammarOff:

    def test_json_parse_failure_returns_local_failed(self):
        """When grammar disabled and model returns malformed JSON, agreement=local_failed."""
        bad_content = "sorry I cannot assist with that"
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "model": "qwen3.6-35b-a3b.gguf",
            "choices": [{"message": {"content": bad_content}}],
        }
        resp.raise_for_status = MagicMock()

        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", return_value=resp),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
                _use_grammar=False,
            )

        assert result.json_valid is False
        assert result.agreement == "local_failed"
        assert result.error is not None
        assert "JSONDecodeError" in result.error


# ---------------------------------------------------------------------------
# Test 3: endpoint unavailable
# ---------------------------------------------------------------------------

class TestRunWitnessEndpointUnavailable:

    def test_connect_error_returns_failure_dataclass(self):
        """When httpx raises ConnectError on POST, returns valid dataclass with error."""
        import httpx
        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
            )

        assert isinstance(result, LocalReviewerWitnessResult)
        assert result.agreement == "local_failed"
        assert result.error is not None
        assert "ConnectError" in result.error
        assert result.verdict is None


# ---------------------------------------------------------------------------
# Test 4: timeout
# ---------------------------------------------------------------------------

class TestRunWitnessTimeout:

    def test_timeout_returns_failure_dataclass(self):
        """When httpx raises TimeoutException on POST, returns valid dataclass with error."""
        import httpx
        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", side_effect=httpx.TimeoutException("timed out")),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
            )

        assert isinstance(result, LocalReviewerWitnessResult)
        assert result.agreement == "local_failed"
        assert result.error is not None
        assert "TimeoutException" in result.error


# ---------------------------------------------------------------------------
# Test 5: diff truncation
# ---------------------------------------------------------------------------

class TestRunWitnessDiffTruncation:

    def test_oversized_diff_is_truncated(self):
        """Diff exceeding max_diff_chars is truncated; result returns normally."""
        big_diff = "x" * 50000
        resp = _make_httpx_response("clean", [], 0.90)

        captured_prompts: list[str] = []

        def capture_post(url, *, json=None, timeout=None):
            if json and "messages" in json:
                captured_prompts.append(json["messages"][0]["content"])
            return resp

        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", side_effect=capture_post),
        ):
            result = run_local_reviewer_witness(
                diff_text=big_diff, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
                max_diff_chars=1000,
            )

        assert result.agreement == "agree"
        assert captured_prompts
        assert "truncated" in captured_prompts[0]
        # Prompt should not contain the full 50k diff
        assert len(captured_prompts[0]) < 50000


# ---------------------------------------------------------------------------
# Test 6: agreement classification (parametrized)
# ---------------------------------------------------------------------------

class TestAgreementClassification:

    @pytest.mark.parametrize("claude_v,local_v,expected", [
        ("clean", "clean", "agree"),
        ("fixable", "fixable", "agree"),
        ("needs-human", "needs-human", "agree"),
        ("clean", "fixable", "diverge_minor"),
        ("fixable", "clean", "diverge_minor"),
        ("clean", "needs-human", "diverge_major"),
        ("needs-human", "clean", "diverge_major"),
        ("fixable", "needs-human", "diverge_major"),
        ("needs-human", "fixable", "diverge_major"),
    ])
    def test_score_agreement(self, claude_v, local_v, expected):
        claude = {"verdict": claude_v}
        local = {"verdict": local_v}
        assert _score_agreement(claude, local) == expected

    def test_score_agreement_local_none_is_local_failed(self):
        assert _score_agreement({"verdict": "clean"}, None) == "local_failed"


# ---------------------------------------------------------------------------
# Test 7: witness field attached to reviewer comment
# ---------------------------------------------------------------------------

class TestWitnessFieldAttachedToReviewerComment:
    """Pattern off tests/test_corroboration_adapter.py Test 5."""

    def _make_pending_reviewer_record(self, repo: str = "lapis-pm") -> dict:
        return {
            "gpu_id": "gpu-wit-test-001",
            "spec_id": "spec-wit",
            "agent_type": "reviewer",
            "intent": "review PR #9 cycle 1",
            "repo": repo,
            "pr_number": 9,
            "cycle": 1,
            "status": "pending",
            "retry_count": 0,
        }

    def test_witness_field_in_verdict_comment(self):
        """After _encode_gpu_results, the episodic entry JSON contains local_reviewer_witness."""
        from lapis_pm import pm_core

        record = self._make_pending_reviewer_record()
        output = json.dumps({"verdict": "clean", "issues": [], "confidence": 0.95})
        written_contents: list[str] = []

        def capture_write_result(target_id, content, extra_tags=None):
            written_contents.append(content)
            c = MagicMock()
            c.id = "wit-test-comment"
            return c

        mock_path = MagicMock()
        mock_path.read_text.return_value = output
        mock_path.parent = object()

        corr_result = {
            "verdict": "clean",
            "claim": "all identifiers valid",
            "citations": [],
            "freshness_stamp": datetime.now(timezone.utc).isoformat(),
            "scope_id": "repo:lapis-pm",
            "drift_class": None,
            "notes": "clean",
        }

        wit_result = LocalReviewerWitnessResult(
            verdict="clean",
            issues=[],
            confidence=0.92,
            agreement="agree",
            latency_ms=2374,
            json_valid=True,
            model="qwen3.6-35b-a3b.gguf",
            prompt_hash="sha256:abc123",
            dispatched_at="2026-05-19T22:15:00Z",
            error=None,
        )

        written_obs: list[tuple] = []

        def capture_write_obs(target_id, content, extra_tags=None):
            written_obs.append((target_id, content, extra_tags))
            c = MagicMock()
            c.id = "wit-obs"
            return c

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=mock_path),
            patch("lapis_pm.pm_core._read_fixer_meta", return_value=None),
            patch("lapis_pm.pm_core._consume_fixer_meta"),
            patch("lapis_pm.pm_core.episodic.write_result",
                  side_effect=capture_write_result),
            patch("lapis_pm.pm_core.episodic.write_observation",
                  side_effect=capture_write_obs),
            patch("lapis_pm.pm_core.FAILED_DIR", object()),
            patch("lapis_pm.pm_core._diff_text_for_corr", return_value="fake diff"),
            patch("lapis_pm.pm_core._run_corroboration_pass_sync",
                  return_value=corr_result),
            patch("lapis_pm.local_reviewer_witness.run_local_reviewer_witness",
                  return_value=wit_result),
        ):
            encoded, failed = pm_core._encode_gpu_results("my-target")

        assert encoded == 1
        assert not failed
        reviewer_entries = [c for c in written_contents if "Reviewer verdict for PR #9:" in c]
        assert reviewer_entries, f"No reviewer verdict entry in {written_contents}"
        json_part = reviewer_entries[0].split("\n", 1)[-1].strip()
        verdict_data = json.loads(json_part)
        assert "local_reviewer_witness" in verdict_data, (
            f"local_reviewer_witness missing from verdict JSON: {list(verdict_data.keys())}"
        )
        assert verdict_data["local_reviewer_witness"]["agreement"] == "agree"
        assert verdict_data["local_reviewer_witness"]["verdict"] == "clean"


# ---------------------------------------------------------------------------
# Test 8: provenance fields
# ---------------------------------------------------------------------------

class TestProvenanceFieldsPresent:

    def test_prompt_hash_is_sha256_prefixed(self):
        resp = _make_httpx_response("clean", [], 0.95)
        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", return_value=resp),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
            )
        assert result.prompt_hash.startswith("sha256:")
        # hex part is 64 chars
        assert len(result.prompt_hash) == len("sha256:") + 64

    def test_model_populated_from_response(self):
        resp = _make_httpx_response("clean", [], 0.95, model_name="qwen3.6-custom")
        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", return_value=resp),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
            )
        assert result.model == "qwen3.6-custom"

    def test_dispatched_at_is_iso8601(self):
        resp = _make_httpx_response("clean", [], 0.95)
        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", return_value=resp),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
            )
        # Should parse without raising
        datetime.fromisoformat(result.dispatched_at.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Test 9: graceful degradation to json_object
# ---------------------------------------------------------------------------

class TestGracefulDegradationToJsonObject:

    def test_retries_with_json_object_on_422(self):
        """On 4xx for json_schema, retries once with json_object and succeeds."""
        # First call returns 422; second returns valid response
        bad_resp = MagicMock()
        bad_resp.status_code = 422
        bad_resp.raise_for_status = MagicMock()
        bad_resp.json.return_value = {}

        good_resp = _make_httpx_response("clean", [], 0.90)

        call_count = [0]

        def side_effect(url, *, json=None, timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return bad_resp
            return good_resp

        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", side_effect=side_effect),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
            )

        assert call_count[0] == 2, f"Expected exactly 2 calls, got {call_count[0]}"
        assert result.agreement == "agree"
        assert result.json_valid is True
        assert result.error is None

    def test_no_retry_when_grammar_already_off(self):
        """When _use_grammar=False and endpoint returns 422, no retry loop."""
        bad_resp = MagicMock()
        bad_resp.status_code = 422
        bad_resp.raise_for_status = MagicMock()
        bad_resp.json.return_value = {}

        call_count = [0]

        def side_effect(url, *, json=None, timeout=None):
            call_count[0] += 1
            return bad_resp

        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", side_effect=side_effect),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
                _use_grammar=False,
            )

        # Should only call once (no retry when grammar already off)
        assert call_count[0] == 1
        assert result.agreement == "local_failed"


# ---------------------------------------------------------------------------
# Test 10: divergence observation written on major diverge
# ---------------------------------------------------------------------------

class TestDivergenceObservationOnMajorDiverge:

    def _make_pending_reviewer_record(self) -> dict:
        return {
            "gpu_id": "gpu-div-test-001",
            "spec_id": "spec-div",
            "agent_type": "reviewer",
            "intent": "review PR #11 cycle 1",
            "repo": "lapis-pm",
            "pr_number": 11,
            "cycle": 1,
            "status": "pending",
            "retry_count": 0,
        }

    def test_divergence_obs_written_on_major_diverge(self):
        """When local verdict is needs-human and Claude is clean, a pm:reviewer-divergence obs is written."""
        from lapis_pm import pm_core

        record = self._make_pending_reviewer_record()
        output = json.dumps({"verdict": "clean", "issues": [], "confidence": 0.95})

        corr_result = {
            "verdict": "uncertain", "claim": "", "citations": [],
            "freshness_stamp": datetime.now(timezone.utc).isoformat(),
            "scope_id": "repo:lapis-pm", "drift_class": None, "notes": "",
        }

        # Local witness returns needs-human (major diverge from Claude's clean)
        wit_result = LocalReviewerWitnessResult(
            verdict="needs-human",
            issues=[{"severity": "high", "path": "foo.py", "note": "scope question"}],
            confidence=0.70,
            agreement="diverge_major",
            latency_ms=2000,
            json_valid=True,
            model="qwen3.6-35b-a3b.gguf",
            prompt_hash="sha256:abc",
            dispatched_at="2026-05-19T22:15:00Z",
            error=None,
        )

        written_obs: list[tuple] = []

        def capture_write_obs(target_id, content, extra_tags=None):
            written_obs.append((target_id, content, extra_tags or []))
            c = MagicMock()
            c.id = "obs-div"
            return c

        written_contents: list[str] = []

        def capture_write_result(target_id, content, extra_tags=None):
            written_contents.append(content)
            c = MagicMock()
            c.id = "res"
            return c

        mock_path = MagicMock()
        mock_path.read_text.return_value = output
        mock_path.parent = object()

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=mock_path),
            patch("lapis_pm.pm_core._read_fixer_meta", return_value=None),
            patch("lapis_pm.pm_core._consume_fixer_meta"),
            patch("lapis_pm.pm_core.episodic.write_result",
                  side_effect=capture_write_result),
            patch("lapis_pm.pm_core.episodic.write_observation",
                  side_effect=capture_write_obs),
            patch("lapis_pm.pm_core.FAILED_DIR", object()),
            patch("lapis_pm.pm_core._diff_text_for_corr", return_value="fake diff"),
            patch("lapis_pm.pm_core._run_corroboration_pass_sync",
                  return_value=corr_result),
            patch("lapis_pm.local_reviewer_witness.run_local_reviewer_witness",
                  return_value=wit_result),
        ):
            encoded, failed = pm_core._encode_gpu_results("my-target")

        assert encoded == 1
        # Find divergence observation
        div_obs = [
            o for o in written_obs
            if any("pm:reviewer-divergence" in t for t in o[2])
        ]
        assert div_obs, f"Expected pm:reviewer-divergence observation, got obs: {written_obs}"
        tags = div_obs[0][2]
        assert "pm:reviewer-divergence" in tags
        assert "pm:reviewer-divergence:pr=11:type=major" in tags
        assert "pm:pr=11" in tags


# ---------------------------------------------------------------------------
# Test 11: no divergence observation on minor diverge
# ---------------------------------------------------------------------------

class TestNoDivergenceObservationOnMinorDiverge:

    def _make_pending_reviewer_record(self) -> dict:
        return {
            "gpu_id": "gpu-minor-test-001",
            "spec_id": "spec-minor",
            "agent_type": "reviewer",
            "intent": "review PR #12 cycle 1",
            "repo": "lapis-pm",
            "pr_number": 12,
            "cycle": 1,
            "status": "pending",
            "retry_count": 0,
        }

    def test_no_divergence_obs_on_minor_diverge(self):
        """clean vs fixable (minor) — no pm:reviewer-divergence observation written."""
        from lapis_pm import pm_core

        record = self._make_pending_reviewer_record()
        output = json.dumps({"verdict": "clean", "issues": [], "confidence": 0.95})

        corr_result = {
            "verdict": "uncertain", "claim": "", "citations": [],
            "freshness_stamp": datetime.now(timezone.utc).isoformat(),
            "scope_id": "repo:lapis-pm", "drift_class": None, "notes": "",
        }

        # clean (Claude) vs fixable (local) = minor diverge
        wit_result = LocalReviewerWitnessResult(
            verdict="fixable",
            issues=[{"severity": "low", "path": "foo.py", "note": "nit"}],
            confidence=0.80,
            agreement="diverge_minor",
            latency_ms=2000,
            json_valid=True,
            model="qwen3.6-35b-a3b.gguf",
            prompt_hash="sha256:abc",
            dispatched_at="2026-05-19T22:15:00Z",
            error=None,
        )

        written_obs: list[tuple] = []

        def capture_write_obs(target_id, content, extra_tags=None):
            written_obs.append((target_id, content, extra_tags or []))
            c = MagicMock()
            c.id = "obs-minor"
            return c

        written_contents: list[str] = []

        def capture_write_result(target_id, content, extra_tags=None):
            written_contents.append(content)
            c = MagicMock()
            c.id = "res"
            return c

        mock_path = MagicMock()
        mock_path.read_text.return_value = output
        mock_path.parent = object()

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=mock_path),
            patch("lapis_pm.pm_core._read_fixer_meta", return_value=None),
            patch("lapis_pm.pm_core._consume_fixer_meta"),
            patch("lapis_pm.pm_core.episodic.write_result",
                  side_effect=capture_write_result),
            patch("lapis_pm.pm_core.episodic.write_observation",
                  side_effect=capture_write_obs),
            patch("lapis_pm.pm_core.FAILED_DIR", object()),
            patch("lapis_pm.pm_core._diff_text_for_corr", return_value="fake diff"),
            patch("lapis_pm.pm_core._run_corroboration_pass_sync",
                  return_value=corr_result),
            patch("lapis_pm.local_reviewer_witness.run_local_reviewer_witness",
                  return_value=wit_result),
        ):
            encoded, failed = pm_core._encode_gpu_results("my-target")

        assert encoded == 1
        # Verify no pm:reviewer-divergence observation
        div_obs = [
            o for o in written_obs
            if any("pm:reviewer-divergence" in t for t in o[2])
        ]
        assert not div_obs, (
            f"Expected NO pm:reviewer-divergence observation for minor diverge, got: {div_obs}"
        )
        # But witness field should still be in the verdict JSON
        reviewer_entries = [c for c in written_contents if "Reviewer verdict for PR #12:" in c]
        assert reviewer_entries
        verdict_data = json.loads(reviewer_entries[0].split("\n", 1)[-1].strip())
        assert "local_reviewer_witness" in verdict_data
        assert verdict_data["local_reviewer_witness"]["agreement"] == "diverge_minor"


# ---------------------------------------------------------------------------
# Test 12: no divergence observation on agree
# ---------------------------------------------------------------------------

class TestNoDivergenceObservationOnAgree:

    def _make_pending_reviewer_record(self) -> dict:
        return {
            "gpu_id": "gpu-agree-test-001",
            "spec_id": "spec-agree",
            "agent_type": "reviewer",
            "intent": "review PR #13 cycle 1",
            "repo": "lapis-pm",
            "pr_number": 13,
            "cycle": 1,
            "status": "pending",
            "retry_count": 0,
        }

    def test_no_divergence_obs_on_agree(self):
        """Both clean — no pm:reviewer-divergence observation written."""
        from lapis_pm import pm_core

        record = self._make_pending_reviewer_record()
        output = json.dumps({"verdict": "clean", "issues": [], "confidence": 0.95})

        corr_result = {
            "verdict": "uncertain", "claim": "", "citations": [],
            "freshness_stamp": datetime.now(timezone.utc).isoformat(),
            "scope_id": "repo:lapis-pm", "drift_class": None, "notes": "",
        }

        wit_result = LocalReviewerWitnessResult(
            verdict="clean",
            issues=[],
            confidence=0.95,
            agreement="agree",
            latency_ms=2000,
            json_valid=True,
            model="qwen3.6-35b-a3b.gguf",
            prompt_hash="sha256:abc",
            dispatched_at="2026-05-19T22:15:00Z",
            error=None,
        )

        written_obs: list[tuple] = []

        def capture_write_obs(target_id, content, extra_tags=None):
            written_obs.append((target_id, content, extra_tags or []))
            c = MagicMock()
            c.id = "obs-agree"
            return c

        written_contents: list[str] = []

        def capture_write_result(target_id, content, extra_tags=None):
            written_contents.append(content)
            c = MagicMock()
            c.id = "res"
            return c

        mock_path = MagicMock()
        mock_path.read_text.return_value = output
        mock_path.parent = object()

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=mock_path),
            patch("lapis_pm.pm_core._read_fixer_meta", return_value=None),
            patch("lapis_pm.pm_core._consume_fixer_meta"),
            patch("lapis_pm.pm_core.episodic.write_result",
                  side_effect=capture_write_result),
            patch("lapis_pm.pm_core.episodic.write_observation",
                  side_effect=capture_write_obs),
            patch("lapis_pm.pm_core.FAILED_DIR", object()),
            patch("lapis_pm.pm_core._diff_text_for_corr", return_value="fake diff"),
            patch("lapis_pm.pm_core._run_corroboration_pass_sync",
                  return_value=corr_result),
            patch("lapis_pm.local_reviewer_witness.run_local_reviewer_witness",
                  return_value=wit_result),
        ):
            encoded, failed = pm_core._encode_gpu_results("my-target")

        assert encoded == 1
        div_obs = [
            o for o in written_obs
            if any("pm:reviewer-divergence" in t for t in o[2])
        ]
        assert not div_obs, (
            f"Expected NO pm:reviewer-divergence observation on agree, got: {div_obs}"
        )
