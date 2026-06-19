"""Test spec_review with stubbed shared deliberation envelope.

Validates the field mapping from DeliberationEnvelope to SpecReviewBrief
and tests degraded paths (council_ok=False, facets_ok=False).
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from agents_core.shared_deliberation.envelope import DeliberationEnvelope
from lapis_pm.spec_review import run_spec_review


@pytest.fixture
def spec_fixture(tmp_path):
    """Create a minimal valid spec for testing."""
    spec_path = tmp_path / "test_spec.md"
    spec_path.write_text(
        """
# Test Spec

**Target ID:** `test-target-id`
**Repo:** `test-repo`
**Authority:** advisory

## Goal
Test the shared-deliberation integration.
""",
        encoding="utf-8",
    )
    return spec_path


def test_run_spec_review_with_stubbed_envelope_happy_path(spec_fixture, monkeypatch, tmp_path):
    """Test that run_spec_review correctly maps a happy-path stubbed envelope to SpecReviewBrief.

    Both Facets and Council succeed; the brief should reflect resolved status
    and proceed-to-bind recommendation.
    """
    # Stub the invariant context loader
    monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")

    # Create a happy-path envelope
    def mock_run_deliberation(request):
        return DeliberationEnvelope(
            deliberation_request_id="test-req-123",
            triage="full",
            facets_ok=True,
            facets={
                "deliberation_id": "facets-123",
                "synthesis": {
                    "escalation_recommendation": "proceed",
                    "consensus_level": "strong",
                    "confidence": "high",
                    "recommendation": "All clear",
                },
                "stances": [],
                "methodology": {"synthesis_operator": "haiku"},
            },
            facets_deliberation_id="facets-123",
            operator_requested="haiku",
            operator_effective="haiku",
            council_ok=True,
            council_run_id="council-456",
            council_status="resolved",
            council_landing="Land this spec.",
            council_confidence="converged",
            council_open_questions=["Q1"],
            council_positions=[{"entity": "member1", "position": "support", "reason": "Looks good"}],
            council_voicing_requested="gravitywell",
            council_voicing_effective="gravitywell",
            council_voicing_degraded=False,
            council_voicing_degraded_reason="",
        )

    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

    # Stub invariant context
    monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")

    # Run spec-review
    brief = run_spec_review(
        spec_fixture,
        council_voicing="gravitywell",
        timeout_s=30,
        dispatch_facets=True,
    )

    # Verify field mapping
    assert brief.target_id == "test-target-id"
    assert brief.council_status == "resolved"
    assert brief.council_landing == "Land this spec."
    assert brief.council_confidence == "converged"
    assert brief.council_open_questions == ["Q1"]
    assert len(brief.council_positions) == 1
    assert brief.council_positions[0]["entity"] == "member1"
    assert brief.council_voicing_effective == "gravitywell"
    assert brief.council_voicing_degraded is False
    assert brief.facets_deliberation is not None
    assert brief.facets_deliberation["deliberation_id"] == "facets-123"
    # With Sonnet disabled, recommendation should be proceed-to-bind (council resolved + no issues)
    assert brief.combined_recommendation == "proceed-to-bind"


def test_run_spec_review_with_council_failure(spec_fixture, monkeypatch, tmp_path):
    """Test that run_spec_review handles council_ok=False gracefully.

    When Council fails, the brief should have council_status reflecting the failure,
    recommendation should be incomplete, and the brief should not crash.
    """
    monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")

    # Create a council-failure envelope
    def mock_run_deliberation(request):
        return DeliberationEnvelope(
            deliberation_request_id="test-req-456",
            triage="full",
            facets_ok=True,
            facets={
                "deliberation_id": "facets-789",
                "synthesis": {"escalation_recommendation": "proceed"},
                "stances": [],
                "methodology": {"synthesis_operator": "haiku"},
            },
            facets_deliberation_id="facets-789",
            operator_requested="haiku",
            operator_effective="haiku",
            council_ok=False,  # Council failed
            council_run_id="council-fail",
            council_status="timeout",  # Council timed out
            council_landing="",
            council_confidence="",
            council_open_questions=[],
            council_positions=[],
            council_voicing_requested="gravitywell",
            council_voicing_effective=None,
            council_voicing_degraded=False,
        )

    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

    # Run spec-review
    brief = run_spec_review(
        spec_fixture,
        council_voicing="gravitywell",
        timeout_s=30,
        dispatch_facets=True,
    )

    # Verify degraded path handling
    assert brief.council_status == "timeout"
    assert brief.council_landing == ""
    assert brief.council_open_questions == []
    assert brief.council_positions == []
    # When Council times out, recommendation should be incomplete
    assert brief.combined_recommendation == "incomplete"
    # Brief should not crash, and facets should still be there
    assert brief.facets_deliberation is not None


def test_run_spec_review_with_facets_and_council_failure(spec_fixture, monkeypatch):
    """Test that run_spec_review handles both facets_ok=False and council_ok=False.

    When both fail, the brief should degrade gracefully and use Council-only logic.
    """
    monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")

    # Create an envelope where both fail
    def mock_run_deliberation(request):
        return DeliberationEnvelope(
            deliberation_request_id="test-req-789",
            triage="full",
            facets_ok=False,  # Facets failed
            facets=None,
            facets_deliberation_id=None,
            operator_requested=None,
            operator_effective=None,
            council_ok=False,  # Council also failed
            council_run_id="council-fail-2",
            council_status="error",
            council_landing="",
            council_confidence="",
            council_open_questions=[],
            council_positions=[],
            council_voicing_requested="gravitywell",
            council_voicing_effective=None,
            council_voicing_degraded=False,
        )

    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

    # Run spec-review
    brief = run_spec_review(
        spec_fixture,
        council_voicing="gravitywell",
        timeout_s=30,
        dispatch_facets=True,
    )

    # Verify both failures are reflected
    assert brief.council_status == "error"
    assert brief.facets_deliberation is None
    # When Council fails, recommendation should be incomplete
    assert brief.combined_recommendation == "incomplete"
