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
        sonnet_verdict=sonnet_verdict,
        sonnet_issues=[],
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
        sonnet_verdict="clean",
        sonnet_issues=[{"severity": "high", "note": "missing file"}],
        council_status="resolved",
        council_positions=[],
    )
    assert result == "amend-spec"


def test_high_severity_case_insensitive():
    result = _combined_recommendation(
        sonnet_verdict="clean",
        sonnet_issues=[{"severity": "HIGH", "note": "something"}],
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("timeout"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="x",
        repo="r",
        elapsed_s=5.0,
    )
    assert brief.combined_recommendation == "incomplete"
    assert brief.sonnet_verdict == "timeout"


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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
        sonnet_raw=_sonnet_raw("clean"),
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
