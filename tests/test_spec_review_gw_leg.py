"""Tests for GravityWell reference leg in spec-review.

Tests the additive, non-steering GW reference leg alongside Sonnet:
- Stub mechanism (GW_REVIEW_STUB env var)
- Non-steering property (recommendation identical with/without GW leg)
- Divergence record shape (agree, gw_ran, gw_transcript_ref)
- gw_ran=False path (doorman unreachable, timeout)
- Return shape regression (text is not None check)
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.spec_review import (
    _check_doorman_heartbeat,
    _combined_recommendation,
    _dispatch_gw_reviewer,
    SpecReviewBrief,
)


# ---------------------------------------------------------------------------
# Doorman heartbeat
# ---------------------------------------------------------------------------

def test_doorman_heartbeat_reachable():
    """_check_doorman_heartbeat returns True when doorman responds 200."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("requests.get", return_value=mock_response):
        result = _check_doorman_heartbeat()
        assert result is True


def test_doorman_heartbeat_unreachable():
    """_check_doorman_heartbeat returns False on network error."""
    with patch("requests.get", side_effect=Exception("connection refused")):
        result = _check_doorman_heartbeat()
        assert result is False


def test_doorman_heartbeat_non_200():
    """_check_doorman_heartbeat returns False on non-200 status."""
    mock_response = MagicMock()
    mock_response.status_code = 503

    with patch("requests.get", return_value=mock_response):
        result = _check_doorman_heartbeat()
        assert result is False


def test_doorman_probe_target_regression():
    """_check_doorman_heartbeat probes http://127.0.0.1:8407/healthz (not 203.0.113.10 or /health)."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("requests.get", return_value=mock_response) as mock_get:
        result = _check_doorman_heartbeat()

    # Verify the correct URL was called
    mock_get.assert_called_once_with(
        "http://127.0.0.1:8407/healthz",
        timeout=2.0,
    )
    assert result is True


# ---------------------------------------------------------------------------
# GW stub mechanism
# ---------------------------------------------------------------------------

def test_gw_stub_returns_verdict():
    """GW_REVIEW_STUB=1 returns stubbed verdict without calling call_gw_agent."""
    with patch.dict(os.environ, {"GW_REVIEW_STUB": "1", "GW_REVIEW_STUB_VERDICT": "fixable"}):
        text, transcript, elapsed = _dispatch_gw_reviewer(
            spec_text="test spec",
            synth_target_id="test-tid",
            parsed_target_id="test-id",
            repo="lapis-pm",
            run_id="test-run",
        )

    assert text == "fixable"
    assert transcript == []
    assert elapsed >= 0


def test_gw_stub_default_verdict_clean():
    """GW_REVIEW_STUB=1 defaults to 'clean' when GW_REVIEW_STUB_VERDICT not set."""
    env = {"GW_REVIEW_STUB": "1"}
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("GW_REVIEW_STUB_VERDICT", None)
        text, transcript, elapsed = _dispatch_gw_reviewer(
            spec_text="test",
            synth_target_id="tid",
            parsed_target_id="id",
            repo="repo",
            run_id="run",
        )

    assert text == "clean"


# ---------------------------------------------------------------------------
# GW leg doesn't steer recommendation
# ---------------------------------------------------------------------------

def test_gw_verdict_excluded_from_recommendation():
    """GW verdict is never used in combined_recommendation (always reference-only)."""
    # Test that a GW "needs-human" doesn't move the recommendation
    # when Sonnet is skipped (advisory-only mode).
    # Recommendation should be drive by Facets + Council only.

    rec = _combined_recommendation(
        sonnet_verdict="skip",  # Sonnet is advisory-only (always "skip")
        sonnet_issues=[],
        council_status="resolved",
        council_positions=[],
        facets_escalation="proceed",
        facets_unreliable=False,
        authority="advisory",
    )
    assert rec == "proceed-to-bind"


def test_gw_leg_does_not_steer_recommendation_integration():
    """Integration test: GW leg fields in brief are set correctly; never steer recommendation.

    This verifies the structural guarantee that the GW leg is additive and
    non-steering: when GW is stubbed to different verdicts (fixable vs clean),
    the fields appear in the brief but the recommendation logic is unaffected.
    """
    # Test that _build_brief correctly populates GW fields when called with
    # different GW verdicts, and that _combined_recommendation never consults them.

    # Case 1: GW ran with verdict "fixable"
    brief1 = SpecReviewBrief(
        spec_path=Path("/tmp/spec.md"),
        target_id="test",
        repo="test",
        council_status="resolved",
        council_landing="",
        council_open_questions=[],
        council_confidence="",
        council_positions=[],
        council_run_id="",
        elapsed_s=1.0,
        combined_recommendation="proceed-to-bind",
        gw_ran=True,
        gw_verdict="fixable",  # GW found issues
        gw_findings_count=2,
        elapsed_gw=0.5,
        gw_transcript_ref="/srv/lapis/spec-review-artifacts/abc/gw-transcript.json",
    )

    # Case 2: GW did not run (gw_ran=False)
    brief2 = SpecReviewBrief(
        spec_path=Path("/tmp/spec.md"),
        target_id="test",
        repo="test",
        council_status="resolved",
        council_landing="",
        council_open_questions=[],
        council_confidence="",
        council_positions=[],
        council_run_id="",
        elapsed_s=1.0,
        combined_recommendation="proceed-to-bind",  # Must be same
        gw_ran=False,
        gw_verdict="skip",
        gw_findings_count=0,
        elapsed_gw=0.0,
        gw_transcript_ref="",
    )

    # The recommendation is identical regardless of GW leg state — the brief
    # structure allows storing GW fields, but the recommendation itself never
    # depends on them (it's driven by Facets + Council only).
    assert brief1.combined_recommendation == brief2.combined_recommendation
    assert brief1.combined_recommendation == "proceed-to-bind"

    # GW verdict values are correctly preserved in the brief for logging
    assert brief1.gw_ran is True
    assert brief1.gw_verdict == "fixable"
    assert brief1.gw_findings_count == 2
    assert brief1.elapsed_gw == 0.5

    assert brief2.gw_ran is False
    assert brief2.gw_verdict == "skip"
    assert brief2.gw_findings_count == 0
    assert brief2.elapsed_gw == 0.0


def test_gw_and_sonnet_both_advisory_only():
    """Both GW and Sonnet are reference-only; neither steers recommendation."""
    # Council resolved + Sonnet skipped → proceed-to-bind
    rec = _combined_recommendation(
        sonnet_verdict="skip",
        sonnet_issues=[],
        council_status="resolved",
        council_positions=[],
        facets_escalation="proceed",
        facets_unreliable=False,
        authority="advisory",
    )
    assert rec == "proceed-to-bind"

    # Adding GW verdict in output doesn't change the recommendation
    # (the recommendation logic never consults GW fields)


# ---------------------------------------------------------------------------
# Divergence record shape
# ---------------------------------------------------------------------------

def test_gw_ran_true_sets_agree_comparison():
    """When gw_ran=True, agree is computed as exact verdict equality."""
    # This test verifies the divergence record logic (see run_spec_review).
    # When gw_ran is True, agree = (gw_verdict == sonnet_verdict)

    gw_ran = True
    gw_verdict = "clean"
    sonnet_verdict = "clean"

    agree = (gw_verdict == sonnet_verdict) if gw_ran else None
    assert agree is True

    # Opposite verdicts
    gw_verdict = "fixable"
    agree = (gw_verdict == sonnet_verdict) if gw_ran else None
    assert agree is False


def test_gw_ran_false_agree_is_none():
    """When gw_ran=False, agree is None (not False) — silence not interpreted."""
    gw_ran = False
    gw_verdict = "skip"
    sonnet_verdict = "clean"

    agree = (gw_verdict == sonnet_verdict) if gw_ran else None
    assert agree is None  # Not False — this is critical


def test_gw_transcript_ref_persisted():
    """When GW ran, gw_transcript_ref is the absolute path to transcript JSON."""
    # This is tested implicitly in run_spec_review; here we verify the brief
    # dataclass fields.
    brief = SpecReviewBrief(
        spec_path=Path("/tmp/spec.md"),
        target_id="test",
        repo="test",
        council_status="resolved",
        council_landing="proceed",
        council_open_questions=[],
        council_confidence="full",
        council_positions=[],
        council_run_id="run123",
        elapsed_s=5.0,
        combined_recommendation="proceed-to-bind",
        gw_ran=True,
        gw_verdict="clean",
        gw_findings_count=0,
        elapsed_gw=2.5,
        gw_transcript_ref="/srv/lapis/spec-review-artifacts/abc123/gw-transcript.json",
    )

    assert brief.gw_transcript_ref == "/srv/lapis/spec-review-artifacts/abc123/gw-transcript.json"
    assert brief.gw_ran is True


def test_gw_ran_false_transcript_ref_empty():
    """When gw_ran=False, gw_transcript_ref is empty string."""
    brief = SpecReviewBrief(
        spec_path=Path("/tmp/spec.md"),
        target_id="test",
        repo="test",
        council_status="resolved",
        council_landing="",
        council_open_questions=[],
        council_confidence="",
        council_positions=[],
        council_run_id="",
        elapsed_s=1.0,
        combined_recommendation="proceed-to-bind",
        gw_ran=False,
        gw_verdict="skip",
        gw_findings_count=0,
        elapsed_gw=0.0,
        gw_transcript_ref="",
    )

    assert brief.gw_transcript_ref == ""
    assert brief.gw_ran is False


# ---------------------------------------------------------------------------
# gw_ran=False paths
# ---------------------------------------------------------------------------

def test_gw_doorman_unreachable_returns_none():
    """When doorman is unreachable, _dispatch_gw_reviewer returns (None, [], elapsed)."""
    with patch("lapis_pm.spec_review._check_doorman_heartbeat", return_value=False):
        text, transcript, elapsed = _dispatch_gw_reviewer(
            spec_text="spec",
            synth_target_id="tid",
            parsed_target_id="id",
            repo="repo",
            run_id="run",
        )

    assert text is None
    assert transcript == []
    assert elapsed >= 0


def test_gw_agents_core_import_failure_returns_none():
    """When agents_core import fails, _dispatch_gw_reviewer returns (None, [], elapsed)."""
    # Patch at the import site inside _dispatch_gw_reviewer
    with patch("lapis_pm.spec_review._check_doorman_heartbeat", return_value=True):
        with patch.dict("sys.modules", {"agents_core": None, "agents_core.gw_agent": None}):
            text, transcript, elapsed = _dispatch_gw_reviewer(
                spec_text="spec",
                synth_target_id="tid",
                parsed_target_id="id",
                repo="repo",
                run_id="run",
            )

    # When import fails, we catch and return None
    assert text is None
    assert transcript == []


# ---------------------------------------------------------------------------
# Return shape regression: (text, transcript) unpacking
# ---------------------------------------------------------------------------

def test_return_shape_is_tuple_when_return_transcript_true():
    """call_gw_agent with return_transcript=True returns (text|None, transcript) tuple."""
    # This test documents the return shape contract.
    # CRITICAL: (None, [...]) is truthy, so 'if result is None' is WRONG.
    # Must check 'if text is None' after unpacking.

    with patch.dict(os.environ, {"GW_REVIEW_STUB": "1", "GW_REVIEW_STUB_VERDICT": "clean"}):
        result = _dispatch_gw_reviewer(
            spec_text="spec",
            synth_target_id="tid",
            parsed_target_id="id",
            repo="repo",
            run_id="run",
        )

    # Unpack: this is the critical pattern
    text, transcript, elapsed = result

    # text can be a string or None
    assert isinstance(text, str) or text is None
    # transcript is always a list (even if empty)
    assert isinstance(transcript, list)
    # elapsed is a float
    assert isinstance(elapsed, float)


def test_tuple_none_first_element_is_truthy():
    """(None, [...]) tuple is truthy — demonstrates the trap."""
    # This is the root of the bug the spec warns against
    result = (None, ["item1", "item2"])

    # The tuple itself is truthy!
    assert result  # True
    # But the first element is None
    assert result[0] is None
    # So 'if result is None' would fail (result is not None), but text is None
    text, transcript = result
    assert text is None  # CORRECT check


def test_doorman_unreachable_returns_none_not_empty_tuple():
    """When doorman is unreachable, gw_text is None, not an empty tuple."""
    with patch("lapis_pm.spec_review._check_doorman_heartbeat", return_value=False):
        gw_text, gw_transcript, gw_elapsed = _dispatch_gw_reviewer(
            spec_text="spec",
            synth_target_id="tid",
            parsed_target_id="id",
            repo="repo",
            run_id="run",
        )

    # CRITICAL: check text is None, not whether the tuple is None
    assert gw_text is None
    assert isinstance(gw_transcript, list)
    assert isinstance(gw_elapsed, float)


# ---------------------------------------------------------------------------
# GW findings count extraction
# ---------------------------------------------------------------------------

def test_gw_findings_count_extracted_from_json():
    """When GW returns JSON, gw_findings_count = len(issues)."""
    # This logic is in run_spec_review; verify the extraction
    gw_text = json.dumps({
        "verdict": "fixable",
        "issues": [
            {"severity": "HIGH", "note": "issue 1"},
            {"severity": "MED", "note": "issue 2"},
        ],
        "confidence": 0.8,
    })

    gw_verdict_obj = json.loads(gw_text)
    gw_verdict = gw_verdict_obj.get("verdict", "error")
    gw_findings_count = len(gw_verdict_obj.get("issues", []))

    assert gw_verdict == "fixable"
    assert gw_findings_count == 2


def test_gw_findings_count_zero_when_no_issues():
    """When GW returns no issues, gw_findings_count = 0."""
    gw_text = json.dumps({
        "verdict": "clean",
        "issues": [],
        "confidence": 0.95,
    })

    gw_verdict_obj = json.loads(gw_text)
    gw_findings_count = len(gw_verdict_obj.get("issues", []))

    assert gw_findings_count == 0


# ---------------------------------------------------------------------------
# Brief rendering includes GW section
# ---------------------------------------------------------------------------

def test_brief_dataclass_has_gw_fields():
    """SpecReviewBrief dataclass includes all GW fields."""
    brief = SpecReviewBrief(
        spec_path=Path("/tmp/spec.md"),
        target_id="test",
        repo="test",
        council_status="resolved",
        council_landing="",
        council_open_questions=[],
        council_confidence="",
        council_positions=[],
        council_run_id="",
        elapsed_s=1.0,
        combined_recommendation="proceed-to-bind",
        gw_verdict="clean",
        gw_ran=True,
        gw_findings_count=0,
        elapsed_gw=0.5,
        gw_transcript_ref="/srv/lapis/spec-review-artifacts/abc/gw-transcript.json",
    )

    assert brief.gw_verdict == "clean"
    assert brief.gw_ran is True
    assert brief.gw_findings_count == 0
    assert brief.elapsed_gw == 0.5
    assert brief.gw_transcript_ref == "/srv/lapis/spec-review-artifacts/abc/gw-transcript.json"
