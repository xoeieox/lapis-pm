"""Tests for gw_principal generation and threading in spec-review (U3).

AC1: one gw_principal per run_spec_review, same value reaches both destinations.
AC2: _dispatch_gw_reviewer passes principal= to call_gw_agent (direct mock, not stub path).
AC3: DeliberationRequest.gw_principal is set to the generated value.
AC4: GW_REVIEW_STUB=1 path completes and gw_principal is threaded.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.shared_deliberation.envelope import DeliberationEnvelope
from lapis_pm.spec_review import _dispatch_gw_reviewer, run_spec_review


@pytest.fixture
def advisory_spec(tmp_path):
    spec_path = tmp_path / "gw_principal_test_spec.md"
    spec_path.write_text(
        """
# Test Spec: gw-principal threading

**Target ID:** `gw-principal-test`
**Repo:** `lapis-pm`
**Authority:** advisory

## Goal
Verify gw_principal threading.
""",
        encoding="utf-8",
    )
    return spec_path


def _happy_envelope():
    return DeliberationEnvelope(
        deliberation_request_id="test-req-gw-principal",
        triage="full",
        facets_ok=True,
        facets={
            "deliberation_id": "facets-gw-test",
            "synthesis": {
                "escalation_recommendation": "proceed",
                "consensus_level": "strong",
                "confidence": "high",
                "recommendation": "All clear",
            },
            "stances": [],
            "methodology": {"synthesis_operator": "haiku"},
        },
        facets_deliberation_id="facets-gw-test",
        operator_requested="haiku",
        operator_effective="haiku",
        council_ok=True,
        council_run_id="council-gw-test",
        council_status="resolved",
        council_landing="Proceed.",
        council_confidence="converged",
        council_open_questions=[],
        council_positions=[],
        council_voicing_requested="gravitywell",
        council_voicing_effective="gravitywell",
        council_voicing_degraded=False,
        council_voicing_degraded_reason="",
    )


# ---------------------------------------------------------------------------
# AC1 + AC3: same gw_principal reaches both _dispatch_gw_reviewer and DeliberationRequest
# ---------------------------------------------------------------------------

def test_gw_principal_same_value_reaches_both_destinations(advisory_spec, monkeypatch):
    """AC1+AC3: the same gw_principal string reaches both call sites in one run_spec_review call."""
    captured = {}

    def mock_dispatch(
        spec_text,
        synth_target_id,
        parsed_target_id,
        repo,
        run_id,
        gw_principal=None,
        abandoned_event=None,
    ):
        captured["dispatch_gw_principal"] = gw_principal
        return (None, [], 0.0, "")

    def mock_run_deliberation(request):
        captured["delib_gw_principal"] = request.gw_principal
        return _happy_envelope()

    monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
    monkeypatch.setattr("lapis_pm.spec_review._dispatch_gw_reviewer", mock_dispatch)
    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

    run_spec_review(
        advisory_spec,
        council_voicing="gravitywell",
        timeout_s=30,
        dispatch_facets=True,
    )

    assert "dispatch_gw_principal" in captured, "_dispatch_gw_reviewer was not called"
    assert "delib_gw_principal" in captured, "run_deliberation was not called"

    val = captured["dispatch_gw_principal"]
    assert val is not None, "gw_principal was None when reaching _dispatch_gw_reviewer"
    assert val == captured["delib_gw_principal"], (
        f"gw_principal mismatch: reference leg got {val!r}, "
        f"DeliberationRequest got {captured['delib_gw_principal']!r}"
    )


def test_gw_principal_format(advisory_spec, monkeypatch):
    """gw_principal matches 'gw-gate-<12 lowercase hex chars>'."""
    captured = {}

    def mock_dispatch(**kwargs):
        captured["val"] = kwargs.get("gw_principal")
        return (None, [], 0.0)

    monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
    monkeypatch.setattr("lapis_pm.spec_review._dispatch_gw_reviewer", mock_dispatch)
    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", lambda r: _happy_envelope())

    run_spec_review(
        advisory_spec,
        council_voicing="gravitywell",
        timeout_s=30,
        dispatch_facets=True,
    )

    val = captured.get("val")
    assert val is not None
    assert re.fullmatch(r"gw-gate-[0-9a-f]{12}", val), (
        f"gw_principal {val!r} does not match 'gw-gate-<12 hex>'"
    )


# ---------------------------------------------------------------------------
# AC2: _dispatch_gw_reviewer passes principal= to call_gw_agent (direct, not stub path)
# ---------------------------------------------------------------------------

def test_dispatch_gw_reviewer_passes_principal_to_call_gw_agent():
    """AC2: gw_principal='P' passed to _dispatch_gw_reviewer reaches call_gw_agent(principal='P').

    The stub path (GW_REVIEW_STUB=1) returns before call_gw_agent — this test
    explicitly takes the non-stub path by mocking doorman + module.
    """
    mock_call_gw = MagicMock(return_value=("clean", []))
    mock_gw_module = MagicMock()
    mock_gw_module.call_gw_agent = mock_call_gw
    mock_gw_module.DEFAULT_READONLY_TOOLS = []

    with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
        with patch.dict(sys.modules, {
            "agents_core.gw_agent": mock_gw_module,
        }):
            _dispatch_gw_reviewer(
                spec_text="test spec",
                synth_target_id="tid",
                parsed_target_id="id",
                repo="lapis-pm",
                run_id="run-ac2",
                gw_principal="P",
            )

    mock_call_gw.assert_called_once()
    kwargs = mock_call_gw.call_args.kwargs
    assert kwargs.get("principal") == "P", (
        f"call_gw_agent did not receive principal='P'; got principal={kwargs.get('principal')!r}"
    )


def test_dispatch_gw_reviewer_none_principal_is_forwarded():
    """Default gw_principal=None is forwarded as principal=None to call_gw_agent."""
    mock_call_gw = MagicMock(return_value=("clean", []))
    mock_gw_module = MagicMock()
    mock_gw_module.call_gw_agent = mock_call_gw
    mock_gw_module.DEFAULT_READONLY_TOOLS = []

    with patch("lapis_pm.spec_review.swarm_model", return_value="gravitywell-devstral"):
        with patch.dict(sys.modules, {"agents_core.gw_agent": mock_gw_module}):
            _dispatch_gw_reviewer(
                spec_text="spec",
                synth_target_id="t",
                parsed_target_id="t",
                repo="r",
                run_id="run-none",
            )

    kwargs = mock_call_gw.call_args.kwargs
    assert kwargs.get("principal") is None


# ---------------------------------------------------------------------------
# AC4: GW_REVIEW_STUB=1 path completes; gw_principal still threaded to DeliberationRequest
# ---------------------------------------------------------------------------

def test_run_spec_review_stub_completes_and_threads_gw_principal(advisory_spec, monkeypatch):
    """AC4: GW_REVIEW_STUB=1 — run_spec_review completes and DeliberationRequest.gw_principal is set."""
    captured = {}

    def mock_run_deliberation(request):
        captured["gw_principal"] = request.gw_principal
        return _happy_envelope()

    monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
    monkeypatch.setenv("GW_REVIEW_STUB", "1")
    monkeypatch.setenv("GW_REVIEW_STUB_VERDICT", "clean")
    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

    brief = run_spec_review(
        advisory_spec,
        council_voicing="gravitywell",
        timeout_s=30,
        dispatch_facets=True,
    )

    assert brief is not None, "run_spec_review returned None (should return SpecReviewBrief)"
    assert "gw_principal" in captured, "run_deliberation was not called"
    val = captured["gw_principal"]
    assert val is not None, "DeliberationRequest.gw_principal was None in stub mode"
    assert re.fullmatch(r"gw-gate-[0-9a-f]{12}", val), (
        f"gw_principal format wrong in stub mode: {val!r}"
    )
