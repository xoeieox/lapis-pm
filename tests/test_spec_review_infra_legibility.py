"""Tests for GW-liveness preflight (D1) and infra-failure legibility (D2).

Covers AC1-AC5 from spec-review-gate-infra-legibility-preflight-v0.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from agents_core.shared_deliberation.envelope import DeliberationEnvelope
from lapis_pm.spec_review import (
    SpecReviewBrief,
    _build_brief,
    _council_infra_guidance,
    format_brief,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spec_fixture(tmp_path):
    spec_path = tmp_path / "test_spec.md"
    spec_path.write_text(
        """
# Test Spec

**Target ID:** `test-target-id`
**Repo:** `test-repo`
**Authority:** advisory

## Goal
Test infra-legibility paths.
""",
        encoding="utf-8",
    )
    return spec_path


def _make_council_raw(status: str = "error", error_reason: str = "", run_id: str = "") -> dict:
    return {
        "status": status,
        "landing": "",
        "open_questions": [],
        "confidence": "",
        "positions": [],
        "run_id": run_id,
        "voicing_effective": None,
        "voicing_degraded": False,
        "voicing_degraded_reason": "",
        "error_reason": error_reason,
    }


# ---------------------------------------------------------------------------
# AC5: reason→guidance mapping — unit tests for _council_infra_guidance
# ---------------------------------------------------------------------------

class TestCouncilInfraGuidance:
    def test_heartbeat_stale(self):
        msg = "council worker died/stalled (run_id=r1, last_heartbeat=None, reason=heartbeat_stale)"
        result = _council_infra_guidance(msg)
        assert "Not a spec objection" in result
        assert "stalled" in result.lower() or "Re-gate SOLO" in result

    def test_no_heartbeat_after_startup(self):
        msg = "council worker died/stalled (run_id=r1, last_heartbeat=None, reason=no_heartbeat_after_startup)"
        result = _council_infra_guidance(msg)
        assert "Not a spec objection" in result
        assert "stalled" in result.lower() or "Re-gate SOLO" in result

    def test_gw_not_serving(self):
        result = _council_infra_guidance("gw_not_serving")
        assert "Not a spec objection" in result
        assert "GW not serving" in result

    def test_connect_timeout(self):
        result = _council_infra_guidance("Council poll error: ConnectTimeout()")
        assert "Not a spec objection" in result
        assert "asleep" in result.lower() or "unreachable" in result.lower()

    def test_council_poll_error_prefix(self):
        result = _council_infra_guidance("Council poll error: some network failure")
        assert "Not a spec objection" in result

    def test_slot_queued_timeout(self):
        result = _council_infra_guidance("slot_queued_timeout after 60s")
        assert "Not a spec objection" in result
        assert "GW_ADMISSION_MODE" in result

    def test_wake_failed(self):
        result = _council_infra_guidance("wake_failed: no slot available")
        assert "Not a spec objection" in result
        assert "GW_ADMISSION_MODE" in result

    def test_unknown_reason_no_run_id(self):
        result = _council_infra_guidance("some completely unknown failure string")
        assert "Not a spec objection" in result
        assert "Re-gate ONCE" in result

    def test_unknown_reason_with_run_id(self):
        result = _council_infra_guidance("some unknown failure", run_id="abc123")
        assert "abc123" in result
        assert "Re-gate ONCE" in result

    def test_most_specific_match_heartbeat_stale_wins_over_unknown(self):
        # The full real string from orchestrator.py — must not fall through to default
        full_str = "council worker died/stalled (run_id=r99, last_heartbeat=2026-06-23T10:00:00, reason=heartbeat_stale)"
        result = _council_infra_guidance(full_str)
        assert "COUNCIL_STALL_S" in result  # heartbeat_stale path mentions this knob


# ---------------------------------------------------------------------------
# AC2: reason capture from envelope.errors (full sentence, not bare keyword)
# ---------------------------------------------------------------------------

class TestErrorReasonCapture:
    def _run_build_brief_with_envelope_errors(self, error_str: str) -> SpecReviewBrief:
        council_raw = _make_council_raw(status="error", error_reason=error_str)
        return _build_brief(
            council_raw=council_raw,
            spec_path=Path("/tmp/test.md"),
            parsed_target_id="test-tid",
            repo="test-repo",
            elapsed_s=1.0,
        )

    def test_full_sentence_no_heartbeat(self):
        full_str = "council worker died/stalled (run_id=r1, last_heartbeat=None, reason=no_heartbeat_after_startup)"
        brief = self._run_build_brief_with_envelope_errors(full_str)
        assert brief.council_error_reason == full_str
        assert brief.combined_recommendation == "incomplete"

    def test_full_sentence_heartbeat_stale(self):
        full_str = "council worker died/stalled (run_id=r2, last_heartbeat=2026-06-23T09:00:00, reason=heartbeat_stale)"
        brief = self._run_build_brief_with_envelope_errors(full_str)
        assert brief.council_error_reason == full_str

    def test_connect_timeout_sentence(self):
        full_str = "Council poll error: ConnectTimeout()"
        brief = self._run_build_brief_with_envelope_errors(full_str)
        assert brief.council_error_reason == full_str

    def test_rendered_brief_contains_raw_reason_and_guidance(self):
        full_str = "council worker died/stalled (run_id=r1, last_heartbeat=None, reason=no_heartbeat_after_startup)"
        brief = self._run_build_brief_with_envelope_errors(full_str)
        rendered = format_brief(brief)
        assert full_str in rendered
        assert "Not a spec objection" in rendered

    def test_rendered_brief_heartbeat_stale_contains_guidance(self):
        full_str = "council worker died/stalled (run_id=r2, last_heartbeat=2026-06-23T09:00:00, reason=heartbeat_stale)"
        brief = self._run_build_brief_with_envelope_errors(full_str)
        rendered = format_brief(brief)
        assert full_str in rendered
        assert "Not a spec objection" in rendered

    def test_rendered_brief_connect_timeout_contains_guidance(self):
        full_str = "Council poll error: ConnectTimeout()"
        brief = self._run_build_brief_with_envelope_errors(full_str)
        rendered = format_brief(brief)
        assert full_str in rendered
        assert "Not a spec objection" in rendered


# ---------------------------------------------------------------------------
# AC3: INFRA banner distinct from content-objection path
# ---------------------------------------------------------------------------

class TestInfraBannerRendering:
    def test_infra_banner_present_for_incomplete_with_reason(self):
        council_raw = _make_council_raw(status="error", error_reason="gw_not_serving")
        brief = _build_brief(
            council_raw=council_raw,
            spec_path=Path("/tmp/test.md"),
            parsed_target_id="test-tid",
            repo="test-repo",
            elapsed_s=1.0,
        )
        rendered = format_brief(brief)
        assert "INFRA" in rendered
        assert "gw_not_serving" in rendered
        assert "Not a spec objection" in rendered

    def test_infra_banner_absent_for_content_objection(self):
        # A genuine laid-down (content objection) must NOT show INFRA banner
        council_raw = {
            "status": "laid-down",
            "landing": "The council objects to this spec.",
            "open_questions": [],
            "confidence": "converged",
            "positions": [{"entity": "member1", "position": "block", "reason": "scope too broad"}],
            "run_id": "council-laid-down",
            "voicing_effective": "gravitywell",
            "voicing_degraded": False,
            "voicing_degraded_reason": "",
            "error_reason": "",
        }
        brief = _build_brief(
            council_raw=council_raw,
            spec_path=Path("/tmp/test.md"),
            parsed_target_id="test-tid",
            repo="test-repo",
            elapsed_s=1.0,
        )
        rendered = format_brief(brief)
        assert "INFRA" not in rendered
        assert brief.combined_recommendation == "shape-with-Erah"

    def test_infra_banner_absent_when_no_error_reason(self):
        # No error_reason → no INFRA banner even when recommendation is incomplete
        council_raw = _make_council_raw(status="error", error_reason="")
        brief = _build_brief(
            council_raw=council_raw,
            spec_path=Path("/tmp/test.md"),
            parsed_target_id="test-tid",
            repo="test-repo",
            elapsed_s=1.0,
        )
        rendered = format_brief(brief)
        assert "INFRA FAILURE" not in rendered

    def test_infra_next_step_says_not_bind_block(self):
        council_raw = _make_council_raw(
            status="error",
            error_reason="council worker died/stalled (run_id=r1, last_heartbeat=None, reason=heartbeat_stale)",
        )
        brief = _build_brief(
            council_raw=council_raw,
            spec_path=Path("/tmp/test.md"),
            parsed_target_id="test-tid",
            repo="test-repo",
            elapsed_s=1.0,
        )
        rendered = format_brief(brief)
        # Must say re-gate, not bind-block
        assert "re-gate" in rendered.lower() or "Re-gate" in rendered
        assert "Not a spec objection" in rendered
        # Must have the INFRA label in next-step
        assert "INFRA" in rendered


# ---------------------------------------------------------------------------
# AC4: regression — genuine content objection and clean resolved unchanged
# ---------------------------------------------------------------------------

class TestRegressionNoChange:
    def test_shape_with_Erah_unchanged_for_laid_down(self):
        council_raw = {
            "status": "laid-down",
            "landing": "Council objects.",
            "open_questions": [],
            "confidence": "converged",
            "positions": [{"entity": "voice1", "position": "block", "reason": "scope creep"}],
            "run_id": "c1",
            "voicing_effective": "gravitywell",
            "voicing_degraded": False,
            "voicing_degraded_reason": "",
            "error_reason": "",
        }
        brief = _build_brief(
            council_raw=council_raw,
            spec_path=Path("/tmp/t.md"),
            parsed_target_id="t-id",
            repo="repo",
            elapsed_s=1.0,
        )
        assert brief.combined_recommendation == "shape-with-Erah"
        rendered = format_brief(brief)
        assert "shape-with-Erah" in rendered
        assert "INFRA" not in rendered

    def test_proceed_to_bind_unchanged_for_clean_resolved(self):
        council_raw = {
            "status": "resolved",
            "landing": "Land it.",
            "open_questions": [],
            "confidence": "converged",
            "positions": [{"entity": "voice1", "position": "support", "reason": "good"}],
            "run_id": "c2",
            "voicing_effective": "gravitywell",
            "voicing_degraded": False,
            "voicing_degraded_reason": "",
            "error_reason": "",
        }
        brief = _build_brief(
            council_raw=council_raw,
            spec_path=Path("/tmp/t.md"),
            parsed_target_id="t-id",
            repo="repo",
            elapsed_s=1.0,
        )
        assert brief.combined_recommendation == "proceed-to-bind"
        rendered = format_brief(brief)
        assert "proceed-to-bind" in rendered
        assert "INFRA" not in rendered


# ---------------------------------------------------------------------------
# AC1: GW preflight — swarm_serving=False skips run_deliberation, no paid Sonnet
# ---------------------------------------------------------------------------

def _stub_invariant_context(monkeypatch):
    """Patch out invariant context loading (requires real repo files to exist)."""
    monkeypatch.setattr("lapis_pm.spec_review._load_invariant_context", lambda repo: "")


class TestGWPreflight:
    def test_preflight_skips_run_deliberation_when_swarm_not_serving(
        self, spec_fixture, monkeypatch
    ):
        monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
        _stub_invariant_context(monkeypatch)

        run_deliberation_called = []

        async def mock_run_deliberation(request):
            run_deliberation_called.append(request)
            return DeliberationEnvelope(
                deliberation_request_id="should-not-be-called",
                triage="full",
                council_ok=True,
                council_status="resolved",
                council_landing="Land it.",
            )

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review.swarm_serving", lambda: False)

        from lapis_pm.spec_review import run_spec_review
        brief = run_spec_review(
            spec_fixture,
            council_voicing="gravitywell",
            timeout_s=30,
            dispatch_facets=True,
        )

        assert not run_deliberation_called, "run_deliberation must NOT be called when swarm not serving"
        assert brief.council_error_reason == "gw_not_serving"
        assert brief.combined_recommendation == "incomplete"

    def test_preflight_no_paid_sonnet_fallback(self, spec_fixture, monkeypatch):
        """When swarm not serving, no paid Sonnet call must be made."""
        monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
        _stub_invariant_context(monkeypatch)
        monkeypatch.setattr("lapis_pm.spec_review.swarm_serving", lambda: False)

        call_operator_calls = []

        def mock_call_operator(*args, **kwargs):
            call_operator_calls.append((args, kwargs))
            return "mocked"

        # Patch at the LLM layer — if this is called with operator=sonnet, it's a violation
        try:
            import agents_core.llm as llm_mod
            monkeypatch.setattr(llm_mod, "call_operator", mock_call_operator)
        except (ImportError, AttributeError):
            pass

        from lapis_pm.spec_review import run_spec_review
        brief = run_spec_review(
            spec_fixture,
            council_voicing="gravitywell",
            timeout_s=30,
            dispatch_facets=True,
        )

        sonnet_calls = [
            c for c in call_operator_calls
            if "sonnet" in str(c).lower()
        ]
        assert not sonnet_calls, f"Paid Sonnet was called on preflight-fail path: {sonnet_calls}"
        assert brief.council_error_reason == "gw_not_serving"

    def test_preflight_not_triggered_for_non_gw_voicing(self, spec_fixture, monkeypatch):
        """Preflight only fires for council_voicing=gravitywell."""
        monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
        _stub_invariant_context(monkeypatch)
        # swarm_serving returns False, but voicing=sonnet so preflight should not block
        monkeypatch.setattr("lapis_pm.spec_review.swarm_serving", lambda: False)

        run_deliberation_called = []

        async def mock_run_deliberation(request):
            run_deliberation_called.append(request)
            return DeliberationEnvelope(
                deliberation_request_id="req-sonnet",
                triage="full",
                council_ok=True,
                council_status="resolved",
                council_landing="Land it.",
                council_confidence="converged",
            )

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        from lapis_pm.spec_review import run_spec_review
        brief = run_spec_review(
            spec_fixture,
            council_voicing="sonnet",
            timeout_s=30,
            dispatch_facets=True,
        )

        assert run_deliberation_called, "run_deliberation SHOULD be called for non-GW voicing"
        assert brief.council_error_reason == ""  # not an infra skip

    def test_preflight_result_in_rendered_brief(self, spec_fixture, monkeypatch):
        """Rendered brief for preflight-skip must have INFRA banner."""
        monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
        _stub_invariant_context(monkeypatch)
        monkeypatch.setattr("lapis_pm.spec_review.swarm_serving", lambda: False)

        from lapis_pm.spec_review import run_spec_review, format_brief
        brief = run_spec_review(
            spec_fixture,
            council_voicing="gravitywell",
            timeout_s=30,
            dispatch_facets=True,
        )
        rendered = format_brief(brief)
        assert "INFRA" in rendered
        assert "gw_not_serving" in rendered
        assert "Not a spec objection" in rendered
        # Must not suggest bind-block
        assert "shape-with-Erah" not in rendered


# ---------------------------------------------------------------------------
# AC2 (integration): envelope.errors["council"] captured in council_raw
# ---------------------------------------------------------------------------

class TestEnvelopeErrorCapture:
    def test_envelope_error_council_string_propagated(self, spec_fixture, monkeypatch):
        monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
        _stub_invariant_context(monkeypatch)

        full_reason = "council worker died/stalled (run_id=r1, last_heartbeat=None, reason=no_heartbeat_after_startup)"

        async def mock_run_deliberation(request):
            return DeliberationEnvelope(
                deliberation_request_id="req-fail",
                triage="full",
                facets_ok=False,
                council_ok=False,
                council_run_id="r1",
                council_status="error",
                errors={"council": full_reason},
            )

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        from lapis_pm.spec_review import run_spec_review
        brief = run_spec_review(
            spec_fixture,
            council_voicing="gravitywell",
            timeout_s=30,
            dispatch_facets=True,
        )

        assert brief.council_error_reason == full_reason
        assert brief.combined_recommendation == "incomplete"

    def test_envelope_error_in_rendered_brief(self, spec_fixture, monkeypatch):
        monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
        _stub_invariant_context(monkeypatch)

        full_reason = "council worker died/stalled (run_id=r2, last_heartbeat=2026-06-23T09:00:00, reason=heartbeat_stale)"

        async def mock_run_deliberation(request):
            return DeliberationEnvelope(
                deliberation_request_id="req-fail2",
                triage="full",
                facets_ok=False,
                council_ok=False,
                council_run_id="r2",
                council_status="error",
                errors={"council": full_reason},
            )

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        from lapis_pm.spec_review import run_spec_review, format_brief
        brief = run_spec_review(
            spec_fixture,
            council_voicing="gravitywell",
            timeout_s=30,
            dispatch_facets=True,
        )
        rendered = format_brief(brief)

        assert full_reason in rendered
        assert "Not a spec objection" in rendered
        assert "INFRA" in rendered

    def test_council_error_reason_distinct_from_voicing_degraded_reason(self, spec_fixture, monkeypatch):
        """council_error_reason and council_voicing_degraded_reason must not collide."""
        monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
        _stub_invariant_context(monkeypatch)

        async def mock_run_deliberation(request):
            return DeliberationEnvelope(
                deliberation_request_id="req-degrade",
                triage="full",
                facets_ok=False,
                council_ok=True,
                council_run_id="r3",
                council_status="resolved",
                council_landing="Land it.",
                council_voicing_degraded=True,
                council_voicing_degraded_reason="gw_not_serving",
                errors={},
            )

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        from lapis_pm.spec_review import run_spec_review
        brief = run_spec_review(
            spec_fixture,
            council_voicing="gravitywell",
            timeout_s=30,
            dispatch_facets=True,
        )

        # voicing_degraded_reason is set but council DID complete (council_ok=True)
        assert brief.council_voicing_degraded_reason == "gw_not_serving"
        # council_error_reason must NOT be set when council completed successfully
        assert brief.council_error_reason == ""
