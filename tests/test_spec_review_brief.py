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
