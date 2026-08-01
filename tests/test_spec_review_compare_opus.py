"""Tests for the reference-leg (the Empiricist) reference mode.

Covers the ratified design (Facets drives; the reference leg is reference-only):
  - The reference leg is rendered side-by-side but is reference-only —
    it never moves the combined recommendation.
  - run_spec_review() dispatches the reference spec_reviewer by default for advisory/hold.
  - The deprecated compare_opus parameter is accepted and emits a deprecation note.
  - The deprecated sonnet_reviewer/--no-sonnet-reviewer/SPEC_REVIEW_SONNET_DISABLED
    aliases still work (sunset 90 days after merge, 2026-09-05).
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


def _reference_raw_high() -> dict:
    """A reference-leg result that, if it counted, would force amend-spec (HIGH issue)."""
    return {
        "status": "processed",
        "verdict": "clean",
        "issues": [{"severity": "high", "citation": "x.py:1", "note": "scary"}],
        "confidence": 0.9,
        "run_id": "reference-ref",
    }


# ---------------------------------------------------------------------------
# Reference leg is reference-only (AC 2: reference-only invariant)
# ---------------------------------------------------------------------------

def test_reference_leg_reference_only_does_not_change_recommendation():
    """A HIGH-severity reference issue must NOT move the gate when reference_advisory_only=True.

    Council resolved + no Facets + reference(reference) => proceed-to-bind, even though
    the reference leg carries a HIGH issue that would otherwise force amend-spec.
    """
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        reference_raw=_reference_raw_high(),
        facets_deliberation=None,
        authority="advisory",
        reference_advisory_only=True,
    )
    assert brief.combined_recommendation == "proceed-to-bind"
    # Reference fields still populated for rendering
    assert brief.reference_verdict == "clean"
    assert brief.reference_issues and brief.reference_issues[0]["severity"] == "high"
    assert brief.reference_advisory_only is True


def test_same_reference_issue_does_move_recommendation_when_not_advisory_only():
    """Control: the identical HIGH issue forces amend-spec when the reference leg is authoritative."""
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        reference_raw=_reference_raw_high(),
        facets_deliberation=None,
        authority="advisory",
        reference_advisory_only=False,
    )
    assert brief.combined_recommendation == "amend-spec"


def test_reference_leg_timeout_does_not_force_incomplete():
    """A timed-out reference leg must not drag the gate to 'incomplete'."""
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        reference_raw={"status": "timeout", "verdict": "timeout", "issues": [],
                       "confidence": 0.0, "run_id": "reference-to"},
        facets_deliberation=None,
        authority="advisory",
        reference_advisory_only=True,
    )
    assert brief.combined_recommendation == "proceed-to-bind"


def test_reference_leg_needs_human_does_not_move_recommendation():
    """AC 2: needs-human reference verdict must NOT move the gate when reference-only."""
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        reference_raw={"status": "processed", "verdict": "needs-human", "issues": [],
                       "confidence": 0.5, "run_id": "reference-nh"},
        facets_deliberation=None,
        authority="advisory",
        reference_advisory_only=True,
    )
    assert brief.combined_recommendation == "proceed-to-bind"


def test_reference_leg_parse_failed_does_not_move_recommendation():
    """AC 2: parse_failed reference result must NOT move the gate when reference-only."""
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        reference_raw={"status": "processed", "verdict": "parse_failed", "issues": [],
                       "confidence": 0.0, "run_id": "reference-pf"},
        facets_deliberation=None,
        authority="advisory",
        reference_advisory_only=True,
    )
    # parse_failed only fires when the reference leg is authoritative (not advisory-only)
    assert brief.combined_recommendation == "proceed-to-bind"


# ---------------------------------------------------------------------------
# Rendering: reference-leg section + correct heading (AC 4)
# ---------------------------------------------------------------------------

def test_format_brief_renders_reference_section_with_reference_heading():
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
        reference_raw=_reference_raw_high(),
        facets_deliberation=facets,
        authority="advisory",
        reference_advisory_only=True,
    )
    out = format_brief(brief)
    assert "## Facets deliberation" in out
    assert "## Empiricist (" in out
    assert "factual-claim verification, reference only (does not affect recommendation)" in out
    assert "## Mirror Council" in out
    # Old heading must not appear
    assert "Opus technical review" not in out
    assert "Sonnet technical review" not in out


def test_format_brief_reference_section_omitted_when_not_dispatched():
    """AC 3: when the reference leg did not run, brief omits the reference section."""
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        reference_raw=None,
        facets_deliberation=None,
        authority="advisory",
        reference_advisory_only=False,
    )
    out = format_brief(brief)
    assert "Sonnet technical review" not in out
    assert "## Empiricist (" not in out
    assert brief.reference_verdict == "skip"


# ---------------------------------------------------------------------------
# Deprecated opus_* aliases still resolve (AC 4)
# ---------------------------------------------------------------------------

def test_deprecated_opus_aliases_return_reference_values():
    """AC 4: deprecated opus_* read-aliases must return the same values as reference_*."""
    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=Path("/tmp/spec.md"),
        parsed_target_id="t",
        repo="lapis-pm",
        elapsed_s=1.0,
        reference_raw=_reference_raw_high(),
        facets_deliberation=None,
        authority="advisory",
        reference_advisory_only=True,
    )
    assert brief.opus_verdict == brief.reference_verdict
    assert brief.opus_issues == brief.reference_issues
    assert brief.opus_confidence == brief.reference_confidence
    assert brief.opus_run_id == brief.reference_run_id
    assert brief.opus_advisory_only == brief.reference_advisory_only


# ---------------------------------------------------------------------------
# Integration: run_spec_review dispatches the reference leg by default (AC 1)
# ---------------------------------------------------------------------------

_ADVISORY_SPEC = textwrap.dedent("""\
    # Spec: Reference Default Target

    **Target ID:** `reference-default-target`
    **Repo:** `lapis-pm`
    **Authority:** advisory

    ## Goal

    Validate the reference-leg default deep-review path.
""")

_AUTOMERGE_SPEC = textwrap.dedent("""\
    # Spec: Auto-merge Target

    **Target ID:** `automerge-target`
    **Repo:** `lapis-pm`
    **Authority:** auto-merge

    ## Goal

    Validate that auto-merge specs do NOT dispatch the reference leg.
""")


def _run_review(tmp_path, spec_text=_ADVISORY_SPEC, reference_reviewer=True,
                env_overrides=None, **kwargs):
    spec = tmp_path / "spec.md"
    spec.write_text(spec_text, encoding="utf-8")

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

    env = {"FACETS_DISPATCH_DISABLED": "1", "SPEC_REVIEWER_STUB": "1"}
    if env_overrides:
        env.update(env_overrides)

    brief = None
    try:
        with patch(
            "lapis_pm.spec_review._dispatch_council", return_value=run_id
        ), patch(
            "lapis_pm.spec_review._load_invariant_context", return_value="ctx"
        ), patch.dict(os.environ, env):
            brief = run_spec_review(
                spec_path=spec,
                dispatch_facets=True,
                reference_reviewer=reference_reviewer,
                timeout_s=60,
                **kwargs,
            )
        return brief
    finally:
        if council_path.exists():
            council_path.unlink()
        if brief is not None and getattr(brief, "reference_run_id", ""):
            stub_out = _CLAUDE_QUEUE_COMPLETED / f"{brief.reference_run_id}-output.md"
            if stub_out.exists():
                stub_out.unlink()


def test_advisory_default_dispatches_reference_leg(tmp_path):
    """AC 1: default advisory run dispatches the reference leg with no flag."""
    brief = _run_review(tmp_path)
    # Reference leg ran and is on the brief, reference-only
    assert brief.reference_verdict == "clean"
    assert brief.reference_advisory_only is True
    assert brief.reference_run_id.startswith("stub-")
    # Council resolved + reference leg reference-only + Facets disabled => proceed-to-bind
    assert brief.combined_recommendation == "proceed-to-bind"


def test_always_render_reference_section_on_advisory(tmp_path):
    """AC 1: brief must contain the reference-leg section on an advisory/hold run by default."""
    brief = _run_review(tmp_path)
    out = format_brief(brief)
    assert "## Empiricist (" in out
    assert "factual-claim verification, reference only (does not affect recommendation)" in out


def test_automerge_spec_does_not_dispatch_reference_leg(tmp_path):
    """AC 1: auto-merge spec must NOT dispatch the reference leg."""
    brief = _run_review(tmp_path, spec_text=_AUTOMERGE_SPEC)
    assert brief.reference_verdict == "skip"
    assert "Sonnet technical review" not in format_brief(brief)


def test_no_reference_reviewer_flag_skips_leg(tmp_path):
    """AC 3: reference_reviewer=False skips the leg; brief omits the reference section."""
    brief = _run_review(tmp_path, reference_reviewer=False)
    assert brief.reference_verdict == "skip"
    assert "## Empiricist (" not in format_brief(brief)


def test_deprecated_sonnet_reviewer_param_still_skips_leg(tmp_path):
    """AC 4: deprecated sonnet_reviewer=False still skips the leg (sunset 90d after merge)."""
    brief = _run_review(tmp_path, reference_reviewer=True, sonnet_reviewer=False)
    assert brief.reference_verdict == "skip"


def test_spec_review_reference_disabled_env_skips_leg(tmp_path):
    """AC 3: SPEC_REVIEW_REFERENCE_DISABLED=1 skips the leg."""
    brief = _run_review(tmp_path, env_overrides={"SPEC_REVIEW_REFERENCE_DISABLED": "1"})
    assert brief.reference_verdict == "skip"
    assert "## Empiricist (" not in format_brief(brief)


def test_spec_review_sonnet_disabled_env_still_skips_leg(tmp_path):
    """AC 4: deprecated SPEC_REVIEW_SONNET_DISABLED=1 still skips the leg."""
    brief = _run_review(tmp_path, env_overrides={"SPEC_REVIEW_SONNET_DISABLED": "1"})
    assert brief.reference_verdict == "skip"
    assert "## Empiricist (" not in format_brief(brief)


def test_compare_opus_deprecated_is_noop(tmp_path):
    """AC 4: compare_opus=True is accepted, emits deprecation note, and is a no-op."""
    import io
    spec = tmp_path / "spec.md"
    spec.write_text(_ADVISORY_SPEC, encoding="utf-8")

    run_id = f"dep-council-{int(time.time())}"
    from lapis_pm.spec_review import _COUNCIL_DIR
    import yaml
    _COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    council_path = _COUNCIL_DIR / f"{run_id}.yaml"
    council_path.write_text(yaml.safe_dump({
        "run_id": run_id, "status": "resolved", "mode": "deliberation",
        "synthesis": {"confidence": "converged", "landing": "aligns",
                      "open_questions": [], "positions": []},
    }))

    brief = None
    try:
        with patch("lapis_pm.spec_review._dispatch_council", return_value=run_id), \
             patch("lapis_pm.spec_review._load_invariant_context", return_value="ctx"), \
             patch.dict(os.environ, {"FACETS_DISPATCH_DISABLED": "1", "SPEC_REVIEWER_STUB": "1"}):
            import sys as _sys
            import io as _io
            captured = _io.StringIO()
            old_stderr = _sys.stderr
            _sys.stderr = captured
            try:
                brief = run_spec_review(
                    spec_path=spec, dispatch_facets=True, compare_opus=True, timeout_s=60
                )
            finally:
                _sys.stderr = old_stderr
            stderr_out = captured.getvalue()
        assert "deprecated" in stderr_out.lower()
        # Reference leg still ran (default-on; compare_opus is a no-op)
        assert brief.reference_verdict == "clean"
        assert brief.reference_advisory_only is True
    finally:
        if council_path.exists():
            council_path.unlink()
        if brief is not None and getattr(brief, "reference_run_id", ""):
            stub_out = _CLAUDE_QUEUE_COMPLETED / f"{brief.reference_run_id}-output.md"
            if stub_out.exists():
                stub_out.unlink()
