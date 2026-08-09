"""Tests for _build_brief and combined_recommendation derivation.

Assertion table (8 representative cases; all checked below):

  sonnet_verdict  council_status  has_block  expected
  ---             ---             ---        ---
  timeout         resolved        False      incomplete
  clean           timeout         False      incomplete
  clean           resolved        False      proceed-to-bind
  clean           resolved        True       proceed-to-bind   (blocks on 'resolved' don't trigger shape)
  clean           open            False      amend-spec
  clean           open            True       shape-with-Erah
  fixable         resolved        False      amend-spec
  needs-human     resolved        False      shape-with-Erah
  clean           laid-down       False      shape-with-Erah
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lapis_pm.spec_review import SpecReviewBrief, _build_brief, _combined_recommendation


def _positions(*pos_strings: str) -> list[dict]:
    return [{"position": p, "entity": f"ent-{i}"} for i, p in enumerate(pos_strings)]


def _sonnet_raw(verdict: str, issues: list[dict] | None = None, confidence: float = 0.9) -> dict:
    return {
        "status": "timeout" if verdict == "timeout" else "processed",
        "verdict": verdict,
        "issues": issues or [],
        "confidence": confidence,
        "run_id": f"sonnet-{verdict}",
    }


def _council_raw(status: str, positions: list[dict] | None = None) -> dict:
    return {
        "status": status,
        "landing": "test landing",
        "open_questions": [],
        "confidence": "converged" if status == "resolved" else "partial",
        "positions": positions or [],
        "run_id": f"council-{status}",
    }


# ---------------------------------------------------------------------------
# combined_recommendation unit tests (all 8 assertion table rows)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sonnet_verdict,council_status,positions,expected", [
    # incomplete cases
    ("timeout", "resolved", [], "incomplete"),
    ("clean", "timeout", [], "incomplete"),
    # proceed-to-bind
    ("clean", "resolved", [], "proceed-to-bind"),
    ("clean", "resolved", _positions("block"), "proceed-to-bind"),  # blocks ignored on resolved
    # amend-spec
    ("clean", "open", [], "amend-spec"),
    ("fixable", "resolved", [], "amend-spec"),
    # shape-with-Erah
    ("clean", "open", _positions("block"), "shape-with-Erah"),
    ("needs-human", "resolved", [], "shape-with-Erah"),
    ("clean", "laid-down", [], "shape-with-Erah"),
])
def test_combined_recommendation(sonnet_verdict, council_status, positions, expected):
    result = _combined_recommendation(
        reference_verdict=sonnet_verdict,
        reference_issues=[],
        council_status=council_status,
        council_positions=positions,
    )
    assert result == expected, (
        f"verdict={sonnet_verdict!r} status={council_status!r} "
        f"positions={positions} → got {result!r}, expected {expected!r}"
    )


def test_high_severity_issue_triggers_amend():
    """HIGH sonnet issue → amend-spec even if verdict=clean and council=resolved."""
    result = _combined_recommendation(
        reference_verdict="clean",
        reference_issues=[{"severity": "high", "note": "missing file"}],
        council_status="resolved",
        council_positions=[],
    )
    assert result == "amend-spec"


def test_high_severity_case_insensitive():
    result = _combined_recommendation(
        reference_verdict="clean",
        reference_issues=[{"severity": "HIGH", "note": "something"}],
        council_status="resolved",
        council_positions=[],
    )
    assert result == "amend-spec"


# ---------------------------------------------------------------------------
# _build_brief integration
# ---------------------------------------------------------------------------

def test_build_brief_produces_correct_recommendation(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=42.0,
    )
    assert brief.combined_recommendation == "proceed-to-bind"
    assert brief.target_id == "my-tid"
    assert brief.repo == "lapis-pm"
    assert brief.elapsed_s == 42.0


def test_build_brief_reservation_fields(tmp_path):
    """converged-with-reservation is preserved in the brief."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    council = _council_raw("resolved", positions=_positions("agree", "stand-aside"))
    council["confidence"] = "converged-with-reservation"

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=council,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
    )
    assert brief.combined_recommendation == "proceed-to-bind"
    assert brief.council_confidence == "converged-with-reservation"


def test_build_brief_incomplete_on_timeout(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("timeout"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="x",
        repo="r",
        elapsed_s=5.0,
    )
    assert brief.combined_recommendation == "incomplete"
    assert brief.reference_verdict == "timeout"


# ---------------------------------------------------------------------------
# Effective voicing / operator surface tests
# ---------------------------------------------------------------------------

def test_build_brief_council_voicing_no_degrade(tmp_path):
    """Council ran on GravityWell with no degrade."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    council = _council_raw("resolved")
    council["voicing_effective"] = "gravitywell"
    council["voicing_degraded"] = False

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=council,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        council_voicing_requested="gravitywell",
    )
    assert brief.council_voicing_requested == "gravitywell"
    assert brief.council_voicing_effective == "gravitywell"
    assert brief.council_voicing_degraded is False
    assert brief.council_voicing_degraded_reason == ""


def test_build_brief_council_voicing_degraded(tmp_path):
    """Council degraded from GravityWell to Sonnet due to gw_not_serving."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    council = _council_raw("resolved")
    council["voicing_effective"] = "sonnet"
    council["voicing_degraded"] = True
    council["voicing_degraded_reason"] = "gw_not_serving"

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=council,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        council_voicing_requested="gravitywell",
    )
    assert brief.council_voicing_requested == "gravitywell"
    assert brief.council_voicing_effective == "sonnet"
    assert brief.council_voicing_degraded is True
    assert brief.council_voicing_degraded_reason == "gw_not_serving"


def test_build_brief_council_voicing_missing_provenance(tmp_path):
    """Council effective voicing missing (older run) renders as unknown."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    council = _council_raw("resolved")
    # No voicing_effective field

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=council,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        council_voicing_requested="gravitywell",
    )
    assert brief.council_voicing_effective == "unknown"
    assert brief.council_voicing_degraded is False


def test_build_brief_facets_operator_no_degrade(tmp_path):
    """Facets ran on haiku with no degrade (no operator_requested field)."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    facets = {
        "deliberation_id": "fac-123",
        "synthesis": {
            "escalation_recommendation": "proceed",
            "consensus_level": "strong",
            "confidence": "high",
            "recommendation": "clean",
        },
        "methodology": {
            "operator_requested": None,
            "synthesis_operator": "haiku",
            "persona_operators": {"technical-integrity": "haiku", "trickster": "haiku"},
        },
        "stances": [],
    }

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        facets_deliberation=facets,
        facets_operator="haiku",
    )
    assert brief.facets_operator_requested == ""
    assert brief.facets_operator_effective == "haiku"
    assert brief.facets_operator_degraded is False


def test_build_brief_facets_operator_degraded(tmp_path):
    """Facets degraded from gravitywell to haiku (operator_requested set)."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    facets = {
        "deliberation_id": "fac-456",
        "synthesis": {
            "escalation_recommendation": "proceed",
            "consensus_level": "strong",
            "confidence": "high",
            "recommendation": "clean",
        },
        "methodology": {
            "operator_requested": "gravitywell",
            "synthesis_operator": "haiku",
            "persona_operators": {"technical-integrity": "haiku", "trickster": "haiku"},
        },
        "stances": [],
    }

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        facets_deliberation=facets,
        facets_operator="haiku",
    )
    assert brief.facets_operator_requested == "gravitywell"
    assert brief.facets_operator_effective == "haiku"
    assert brief.facets_operator_degraded is True


def test_build_brief_facets_operator_missing_provenance(tmp_path):
    """Facets missing synthesis_operator (incomplete provenance)."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    facets = {
        "deliberation_id": "fac-789",
        "synthesis": {
            "escalation_recommendation": "proceed",
            "consensus_level": "strong",
            "confidence": "high",
            "recommendation": "clean",
        },
        "methodology": {
            "operator_requested": None,
            # Missing synthesis_operator
            "persona_operators": {"technical-integrity": "haiku", "trickster": "sonnet"},
        },
        "stances": [],
    }

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        facets_deliberation=facets,
        facets_operator="haiku",
    )
    assert brief.facets_operator_effective == "unknown"
    assert brief.facets_operator_degraded is False


def test_format_brief_degradation_summary_council_only(tmp_path):
    """Degradation summary line appears when Council degraded."""
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    council = _council_raw("resolved")
    council["voicing_effective"] = "sonnet"
    council["voicing_degraded"] = True
    council["voicing_degraded_reason"] = "gw_not_serving"

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=council,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        council_voicing_requested="gravitywell",
    )
    output = format_brief(brief)
    assert "⚠️ DEGRADED:" in output
    assert "Council" in output
    assert "gw_not_serving" in output


def test_format_brief_degradation_summary_facets_only(tmp_path):
    """Degradation summary line appears when Facets degraded."""
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    facets = {
        "deliberation_id": "fac-456",
        "synthesis": {
            "escalation_recommendation": "proceed",
            "consensus_level": "strong",
            "confidence": "high",
            "recommendation": "clean",
        },
        "methodology": {
            "operator_requested": "gravitywell",
            "synthesis_operator": "haiku",
            "persona_operators": {"technical-integrity": "haiku", "trickster": "haiku"},
        },
        "stances": [],
    }

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        facets_deliberation=facets,
        facets_operator="haiku",
    )
    output = format_brief(brief)
    assert "⚠️ DEGRADED:" in output
    assert "Facets" in output


def test_format_brief_no_degradation_summary_when_clean(tmp_path):
    """No degradation summary line when both legs are clean."""
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    council = _council_raw("resolved")
    council["voicing_effective"] = "gravitywell"
    council["voicing_degraded"] = False

    facets = {
        "deliberation_id": "fac-123",
        "synthesis": {
            "escalation_recommendation": "proceed",
            "consensus_level": "strong",
            "confidence": "high",
            "recommendation": "clean",
        },
        "methodology": {
            "operator_requested": None,
            "synthesis_operator": "haiku",
            "persona_operators": {"technical-integrity": "haiku", "trickster": "haiku"},
        },
        "stances": [],
    }

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=council,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        council_voicing_requested="gravitywell",
        facets_deliberation=facets,
        facets_operator="haiku",
    )
    output = format_brief(brief)
    assert "⚠️ DEGRADED:" not in output
    assert "(on GW)" in output


def test_format_brief_voicing_lines(tmp_path):
    """Voicing lines are rendered with proper formatting."""
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    council = _council_raw("resolved")
    council["voicing_effective"] = "gravitywell"
    council["voicing_degraded"] = False

    facets = {
        "deliberation_id": "fac-123",
        "synthesis": {
            "escalation_recommendation": "proceed",
            "consensus_level": "strong",
            "confidence": "high",
            "recommendation": "clean",
        },
        "methodology": {
            "operator_requested": None,
            "synthesis_operator": "haiku",
            "persona_operators": {"technical-integrity": "haiku", "trickster": "haiku"},
        },
        "stances": [],
    }

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=council,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        council_voicing_requested="gravitywell",
        facets_deliberation=facets,
        facets_operator="haiku",
    )
    output = format_brief(brief)
    assert "Council voicing:" in output
    assert "Facets operator:" in output
    assert "gravitywell (on GW)" in output
    assert "haiku (no degrade)" in output
    assert "haiku (on GW)" not in output


# ---------------------------------------------------------------------------
# claims_checked plumbing (Empiricist re-character)
# ---------------------------------------------------------------------------

def test_build_brief_claims_checked_present(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    raw = _sonnet_raw("clean")
    raw["claims_checked"] = 34

    brief = _build_brief(
        reference_raw=raw,
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
    )
    assert brief.reference_claims_checked == 34


def test_build_brief_claims_checked_absent_is_none_not_zero(tmp_path):
    """Missing claims_checked must default to None, never 0 — the two mean
    different things: None is 'not reported', 0 is a broken lens."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
    )
    assert brief.reference_claims_checked is None


def test_build_brief_claims_checked_non_numeric_is_none(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    raw = _sonnet_raw("clean")
    raw["claims_checked"] = "not-a-number"

    brief = _build_brief(
        reference_raw=raw,
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
    )
    assert brief.reference_claims_checked is None


def test_format_brief_renders_claims_checked(tmp_path):
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    raw = _sonnet_raw("clean")
    raw["claims_checked"] = 40

    brief = _build_brief(
        reference_raw=raw,
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
    )
    output = format_brief(brief)
    assert "Claims checked:** 40" in output
    assert "## Empiricist" in output
    assert "## Sonnet technical review" not in output


def test_format_brief_renders_claims_checked_not_reported_when_absent(tmp_path):
    """A missing claims_checked field must render as '(not reported)', never
    raise, and never silently print 0."""
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
    )
    output = format_brief(brief)
    assert "Claims checked:** (not reported)" in output
    assert "Claims checked:** 0" not in output


# ---------------------------------------------------------------------------
# lapis-pm-spec-review-grounding-legibility-v0 D2: render "Grounding: <status>
# (sha: <hash>, age: <days>d)" for a verified grounding; source_repo is
# deliberately NOT threaded (deferred at the review gate).
# ---------------------------------------------------------------------------

def test_format_brief_renders_grounding_sha_and_age(tmp_path):
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        grounding_status="verified",
        grounding_reason="",
        grounding_resolved_sha="a" * 40,
        grounding_age_days=3,
    )
    output = format_brief(brief)
    assert f"**Grounding:** verified (sha: {'a' * 40}, age: 3d)" in output
    # age must be a plain machine-parsable integer + "d", never prose
    assert "3 days" not in output


def test_format_brief_grounding_sha_omitted_when_not_verified(tmp_path):
    """A non-verified grounding must never fabricate a sha — the reason string
    renders instead, exactly as before this unit."""
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        grounding_status="silent-denial",
        grounding_reason="sim_data_missing",
    )
    output = format_brief(brief)
    assert "**Grounding:** silent-denial (sim_data_missing)" in output
    assert "sha:" not in output


def test_format_brief_does_not_thread_source_repo(tmp_path):
    """source_repo is DEFERRED (Council unanimous call, 2026-08-03) — the report
    must not depend on it, and SpecReviewBrief must carry no such field."""
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        grounding_status="verified",
        grounding_resolved_sha="b" * 40,
        grounding_age_days=0,
    )
    assert not hasattr(brief, "grounding_source_repo")
    output = format_brief(brief)
    assert output  # renders fine without any source_repo field


def test_format_brief_stale_clone_distinguishable_from_fresh(tmp_path):
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    fresh = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        grounding_status="verified",
        grounding_resolved_sha="c" * 40,
        grounding_age_days=0,
    )
    stale = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        grounding_status="verified",
        grounding_resolved_sha="c" * 40,
        grounding_age_days=14,
    )
    assert "age: 0d" in format_brief(fresh)
    assert "age: 14d" in format_brief(stale)
    assert format_brief(fresh) != format_brief(stale)


# ---------------------------------------------------------------------------
# lapis-pm-gate-brief-tells-the-truth-v0
# D1: escalation_reason rendered unconditionally.
# D2: citation_guard block, including the primary DoD (would_have_downgraded).
# D3: per-stance citation_state.
# ---------------------------------------------------------------------------

def _facets_with_synthesis(synthesis: dict, stances: list[dict] | None = None) -> dict:
    return {
        "deliberation_id": "fac-cg-1",
        "synthesis": synthesis,
        "methodology": {
            "operator_requested": None,
            "synthesis_operator": "haiku",
            "persona_operators": {"technical-integrity": "haiku", "trickster": "haiku"},
        },
        "stances": stances or [],
    }


def _brief_from_facets(tmp_path, facets):
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")
    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        facets_deliberation=facets,
        facets_operator="haiku",
    )
    from lapis_pm.spec_review import format_brief
    return format_brief(brief)


def test_format_brief_renders_escalation_reason(tmp_path):
    facets = _facets_with_synthesis({
        "escalation_recommendation": "proceed",
        "escalation_reason": "citations failed validation; block discarded",
        "consensus_level": "strong",
        "confidence": "high",
        "recommendation": "clean",
    })
    output = _brief_from_facets(tmp_path, facets)
    assert "citations failed validation; block discarded" in output


def test_format_brief_escalation_reason_empty_still_renders_line(tmp_path):
    facets = _facets_with_synthesis({
        "escalation_recommendation": "proceed",
        "consensus_level": "strong",
        "confidence": "high",
        "recommendation": "clean",
    })
    output = _brief_from_facets(tmp_path, facets)
    assert "**Escalation reason:**" in output


def test_format_brief_citation_guard_would_have_downgraded(tmp_path):
    """Primary DoD: the post-#36 shape. discounted_personas must be named and the
    brief must say the block stands despite no validated citation."""
    facets = _facets_with_synthesis({
        "escalation_recommendation": "proceed",
        "consensus_level": "strong",
        "confidence": "high",
        "recommendation": "clean",
        "citation_guard": {
            "checkable": True,
            "downgraded": False,
            "would_have_downgraded": True,
            "blocking_personas": ["technical-integrity", "trickster"],
            "discounted_personas": ["technical-integrity", "trickster"],
            "citation_states": {},
        },
    })
    output = _brief_from_facets(tmp_path, facets)
    assert "technical-integrity" in output
    assert "trickster" in output
    assert "block stands" in output.lower()


def test_format_brief_citation_guard_legacy_downgraded_shape(tmp_path):
    """DoD 1b: the pre-#36 legacy shape (downgraded: True) must also render as an
    action taken, not as a clean proceed."""
    facets = _facets_with_synthesis({
        "escalation_recommendation": "proceed",
        "consensus_level": "strong",
        "confidence": "high",
        "recommendation": "clean",
        "citation_guard": {
            "checkable": True,
            "downgraded": True,
            "blocking_personas": ["technical-integrity"],
            "discounted_personas": ["technical-integrity"],
            "citation_states": {},
        },
    })
    output = _brief_from_facets(tmp_path, facets)
    assert "technical-integrity" in output
    assert "no action taken" not in output.lower()


def test_format_brief_citation_guard_unguarded(tmp_path):
    facets = _facets_with_synthesis({
        "escalation_recommendation": "proceed",
        "consensus_level": "strong",
        "confidence": "high",
        "recommendation": "clean",
        "citation_guard": {
            "checkable": False,
            "downgraded": False,
            "blocking_personas": [],
            "discounted_personas": [],
            "citation_states": {},
            "unguarded": True,
        },
    })
    output = _brief_from_facets(tmp_path, facets)
    assert "unguarded" in output.lower()


def test_format_brief_citation_guard_not_checkable_no_unguarded_key_no_keyerror(tmp_path):
    """DoD 3: checkable:False, downgraded:False, with NO 'unguarded' key at all
    (the not-checkable branch) must render without KeyError."""
    facets = _facets_with_synthesis({
        "escalation_recommendation": "proceed",
        "consensus_level": "strong",
        "confidence": "high",
        "recommendation": "clean",
        "citation_guard": {
            "checkable": False,
            "downgraded": False,
            "blocking_personas": [],
            "discounted_personas": [],
            "citation_states": {},
        },
    })
    output = _brief_from_facets(tmp_path, facets)
    assert output


def test_format_brief_citation_guard_absent_renders_as_before(tmp_path):
    """DoD 4: regression guard — citation_guard absent renders the same as a
    synthesis carrying no citation_guard key at all."""
    facets_without = _facets_with_synthesis({
        "escalation_recommendation": "proceed",
        "consensus_level": "strong",
        "confidence": "high",
        "recommendation": "clean",
    })
    facets_with_none = _facets_with_synthesis({
        "escalation_recommendation": "proceed",
        "consensus_level": "strong",
        "confidence": "high",
        "recommendation": "clean",
        "citation_guard": None,
    })
    out_without = _brief_from_facets(tmp_path, facets_without)
    out_with_none = _brief_from_facets(tmp_path, facets_with_none)
    assert "Citation guard" not in out_without
    assert "Citation guard" not in out_with_none


def test_format_brief_renders_per_stance_citation_state(tmp_path):
    facets = _facets_with_synthesis(
        {
            "escalation_recommendation": "proceed",
            "consensus_level": "strong",
            "confidence": "high",
            "recommendation": "clean",
        },
        stances=[
            {
                "persona": "technical-integrity",
                "confidence": "high",
                "claim": "some claim",
                "citation_state": "unvalidated",
            },
        ],
    )
    output = _brief_from_facets(tmp_path, facets)
    assert "unvalidated" in output
    # The dead `verified` key branch is explicitly out of scope — must remain
    # 'inferred' since StanceRecord carries no 'verified' key.
    assert "inferred" in output


# ---------------------------------------------------------------------------
# C7 consumer criterion wiring (lapis-pm-gate-consumer-criterion-v0)
# ---------------------------------------------------------------------------

def test_build_brief_consumer_criterion_blocking_overrides_proceed_to_bind(tmp_path):
    """A blocking consumer-criterion finding forces the recommendation off
    proceed-to-bind even when Facets/Council both signal clean (DoD-2)."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=1.0,
        consumer_criterion={"status": "blocking", "reason": "missing", "raw_value": None},
    )
    assert brief.combined_recommendation == "amend-spec"
    assert brief.consumer_criterion_status == "blocking"
    assert brief.consumer_criterion_finding
    assert "C7" in brief.consumer_criterion_finding


def test_build_brief_consumer_criterion_ok_does_not_downgrade(tmp_path):
    """A passing consumer criterion never introduces a finding or moves the
    recommendation (DoD-4: the criterion cannot pass by rejecting everything)."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=1.0,
        consumer_criterion={"status": "ok", "raw_value": "the nightly briefing pipeline"},
    )
    assert brief.combined_recommendation == "proceed-to-bind"
    assert brief.consumer_criterion_status == "ok"
    assert brief.consumer_criterion_finding == ""


def test_build_brief_consumer_criterion_defaults_to_ok_when_absent(tmp_path):
    """Callers that don't pass consumer_criterion (legacy call sites) get the
    safe default — no finding, no behavior change."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=1.0,
    )
    assert brief.consumer_criterion_status == "ok"
    assert brief.combined_recommendation == "proceed-to-bind"


def test_format_brief_renders_blocking_consumer_criterion(tmp_path):
    """The blocking finding surfaces in the rendered verdict text (DoD-5 requires
    this in a live gate run; this test asserts the render mechanism itself)."""
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=1.0,
        consumer_criterion={"status": "blocking", "reason": "missing", "raw_value": None},
    )
    output = format_brief(brief)
    assert "C7" in output
    assert "BLOCKING" in output
    assert "amend-spec" in output
