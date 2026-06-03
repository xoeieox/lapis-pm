"""Tests for compare-opus dual-review mode.

Covers the ratified design (Facets drives; Opus is reference):
  - In compare mode the Opus leg is rendered side-by-side but is reference-only —
    it never moves the combined recommendation.
  - run_spec_review(compare_opus=True) actually dispatches the Opus spec_reviewer
    in parallel and threads its result onto the brief.
"""
from __future__ import annotations

import os
import time
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from lapis_pm.spec_review import (
    _CLAUDE_QUEUE_COMPLETED,
    _build_brief,
    format_brief,
    run_spec_review,
)


def _council_raw(status: str, positions=None) -> dict:
    return {
        "status": status,
        "landing": "test landing",
        "open_questions": [],
        "confidence": "converged" if status == "resolved" else "partial",
        "positions": positions or [],
        "run_id": f"council-{status}",
    }


def _opus_raw_high() -> dict:
    """An Opus result that, if it counted, would force amend-spec (HIGH issue)."""
    return {
        "status": "processed",
        "verdict": "clean",
        "issues": [{"severity": "high", "citation": "x.py:1", "note": "scary"}],
        "confidence": 0.9,
        "run_id": "opus-ref",
    }


# ---------------------------------------------------------------------------
# Opus is reference-only in compare mode
# ---------------------------------------------------------------------------

def test_compare_opus_reference_only_does_not_change_recommendation():
    """A HIGH-severity Opus issue must NOT move the gate when opus_advisory_only=True.

    Council resolved + no Facets + Opus(reference) ⇒ proceed-to-bind, even though the
    Opus leg carries a HIGH issue that would otherwise force amend-spec.
    """
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        opus_raw=_opus_raw_high(),
        facets_deliberation=None,
        authority="advisory",
        opus_advisory_only=True,
    )
    assert brief.combined_recommendation == "proceed-to-bind"
    # Opus fields still populated for rendering
    assert brief.opus_verdict == "clean"
    assert brief.opus_issues and brief.opus_issues[0]["severity"] == "high"
    assert brief.opus_advisory_only is True


def test_same_opus_issue_does_move_recommendation_when_not_advisory_only():
    """Control: the identical HIGH issue forces amend-spec when Opus is authoritative.

    Proves the reference-only behavior above is the flag's doing, not a no-op input.
    """
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        opus_raw=_opus_raw_high(),
        facets_deliberation=None,
        authority="advisory",
        opus_advisory_only=False,
    )
    assert brief.combined_recommendation == "amend-spec"


def test_compare_opus_timeout_does_not_force_incomplete():
    """A timed-out reference Opus leg must not drag the gate to 'incomplete'."""
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        opus_raw={"status": "timeout", "verdict": "timeout", "issues": [],
                  "confidence": 0.0, "run_id": "opus-to"},
        facets_deliberation=None,
        authority="advisory",
        opus_advisory_only=True,
    )
    assert brief.combined_recommendation == "proceed-to-bind"


# ---------------------------------------------------------------------------
# Rendering: both sections + reference label
# ---------------------------------------------------------------------------

def test_format_brief_renders_both_sections_with_reference_label():
    facets = {
        "deliberation_id": "fac-123",
        "synthesis": {
            "consensus_level": "strong",
            "escalation_recommendation": "proceed",
            "confidence": "high",
            "recommendation": "looks good",
        },
        "stances": [{"persona": "technical-integrity", "confidence": "high", "claim": "sound"}],
    }
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        opus_raw=_opus_raw_high(),
        facets_deliberation=facets,
        authority="advisory",
        opus_advisory_only=True,
    )
    out = format_brief(brief)
    assert "## Facets deliberation" in out
    assert "Opus technical review (reference — does not affect recommendation)" in out
    assert "## Mirror Council" in out


def test_format_brief_opus_heading_plain_when_authoritative():
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        opus_raw=_opus_raw_high(),
        facets_deliberation=None,
        authority="advisory",
        opus_advisory_only=False,
    )
    out = format_brief(brief)
    assert "## Opus technical review\n" in out
    assert "reference — does not affect recommendation" not in out


# ---------------------------------------------------------------------------
# Integration: run_spec_review(compare_opus=True) dispatches the Opus leg
# ---------------------------------------------------------------------------

_ADVISORY_SPEC = textwrap.dedent("""\
    # Spec: Compare Opus Target

    **Target ID:** `compare-opus-target`
    **Repo:** `lapis-pm`
    **Authority:** advisory

    ## Goal

    Validate the compare-opus dual-review path.
""")


def _run_compare(tmp_path, compare_opus: bool):
    spec = tmp_path / "spec.md"
    spec.write_text(_ADVISORY_SPEC, encoding="utf-8")

    run_id = f"cmp-council-{int(time.time())}"
    from lapis_pm.spec_review import _COUNCIL_DIR
    import yaml
    _COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    council_path = _COUNCIL_DIR / f"{run_id}.yaml"
    council_path.write_text(yaml.safe_dump({
        "run_id": run_id,
        "status": "resolved",
        "mode": "deliberation",
        "synthesis": {
            "confidence": "converged",
            "landing": "aligns",
            "open_questions": [],
            "positions": [{"entity": "ent-a", "position": "agree"}],
        },
    }))

    try:
        with patch(
            "lapis_pm.spec_review._dispatch_council", return_value=run_id
        ), patch(
            "lapis_pm.spec_review._load_invariant_context", return_value="ctx"
        ), patch.dict(os.environ, {
            "FACETS_DISPATCH_DISABLED": "1",   # exercise dispatch/poll without live Facets
            "SPEC_REVIEWER_STUB": "1",          # Opus leg writes a stub 'clean' verdict
        }):
            brief = run_spec_review(
                spec_path=spec,
                dispatch_facets=True,
                compare_opus=compare_opus,
                timeout_s=60,
            )
        return brief
    finally:
        if council_path.exists():
            council_path.unlink()
        # stub Opus output is named after the (random) synth task id == opus_run_id
        if "brief" in dir() and getattr(brief, "opus_run_id", ""):
            stub_out = _CLAUDE_QUEUE_COMPLETED / f"{brief.opus_run_id}-output.md"
            if stub_out.exists():
                stub_out.unlink()


def test_run_spec_review_compare_opus_dispatches_reference_opus(tmp_path):
    brief = _run_compare(tmp_path, compare_opus=True)
    # Opus leg ran and is on the brief, reference-only
    assert brief.opus_verdict == "clean"
    assert brief.opus_advisory_only is True
    assert brief.opus_run_id.startswith("stub-")
    # Council resolved + Opus reference-only + Facets disabled ⇒ proceed-to-bind
    assert brief.combined_recommendation == "proceed-to-bind"
    out = format_brief(brief)
    assert "Opus technical review (reference — does not affect recommendation)" in out


def test_run_spec_review_default_skips_opus(tmp_path):
    brief = _run_compare(tmp_path, compare_opus=False)
    # No Opus dispatched in default Facets mode
    assert brief.opus_verdict == "skip"
    assert brief.opus_advisory_only is False
    assert "Opus technical review" not in format_brief(brief)
