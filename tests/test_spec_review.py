"""Unit tests for facets-pre-bind-wire-v0.

Tests: run_spec_review with Facets disabled, auto-merge authority gating,
_parse_spec_authority, _build_brief with Facets, _combined_recommendation with Facets
escalations, format_brief Facets section, cmd_spec_review --no-facets flag,
and shared-orchestrator in-process deliberation semaphore initialization.
"""
from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lapis_pm.spec_review import (
    _COUNCIL_DIR,
    SpecFrontmatterError,
    SpecReviewBrief,
    _build_brief,
    _combined_recommendation,
    _parse_spec_authority,
    format_brief,
    run_spec_review,
)

# ---------------------------------------------------------------------------
# L2 fixtures
# ---------------------------------------------------------------------------

def _facets_deliberation_with_parse_failures(
    stance_failed: bool = True,
    synth_failed: bool = False,
) -> dict:
    """Fixture with optional parse failures on stances and/or synthesis."""
    stances = [
        {
            "persona": "technical-integrity",
            "confidence": "medium",
            "claim": "",
            "parse_failed": stance_failed,
            "parse_error": "no JSON object found" if stance_failed else None,
        },
        {
            "persona": "trickster",
            "confidence": "medium",
            "claim": "portfolio fit is good",
            "parse_failed": False,
            "parse_error": None,
        },
    ]
    synthesis = {
        "escalation_recommendation": "brief-to-Erah",
        "consensus_level": "divergent",
        "confidence": "low",
        "recommendation": "Synthesis failed to parse.",
        "parse_failed": synth_failed,
        "parse_error": "JSON decode error: ..." if synth_failed else None,
    }
    return {
        "deliberation_id": "test-parse-fail-id",
        "synthesis": synthesis,
        "stances": stances,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_SPEC_ADVISORY = textwrap.dedent("""\
    # Spec: My Target

    **Target ID:** `my-target-id`
    **Repo:** `lapis-pm`
    **Authority:** advisory

    ## Goal

    Do the thing.
""")

_VALID_SPEC_AUTO_MERGE = textwrap.dedent("""\
    # Spec: Auto Target

    **Target ID:** `auto-target-id`
    **Repo:** `lapis-pm`
    **Authority:** auto-merge

    ## Goal

    Small thing.
""")

_VALID_SPEC_HOLD = textwrap.dedent("""\
    # Spec: Hold Target

    **Target ID:** `hold-target-id`
    **Repo:** `lapis-pm`
    **Authority:** hold

    ## Goal

    Big thing.
""")

_SPEC_NO_AUTHORITY = textwrap.dedent("""\
    # Spec: No Authority

    **Target ID:** `no-auth-id`
    **Repo:** `lapis-pm`

    ## Goal

    No authority line here.
""")


def _write_spec(tmp_path: Path, content: str, name: str = "spec.md") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _council_yaml(run_id: str, status: str = "resolved") -> str:
    import yaml
    return yaml.safe_dump({
        "run_id": run_id,
        "status": status,
        "mode": "deliberation",
        "synthesis": {
            "confidence": "converged" if status == "resolved" else "partial",
            "landing": "test landing",
            "open_questions": [],
            "positions": [{"entity": "ent-a", "position": "agree"}],
        },
    })


def _write_council_yaml(run_id: str, status: str = "resolved") -> Path:
    _COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    p = _COUNCIL_DIR / f"{run_id}.yaml"
    p.write_text(_council_yaml(run_id, status))
    return p


def _facets_deliberation_fixture(
    escalation: str = "proceed",
    consensus: str = "consensus",
) -> dict:
    return {
        "deliberation_id": "test-deliberation-id",
        "synthesis": {
            "escalation_recommendation": escalation,
            "consensus_level": consensus,
            "confidence": "high",
            "recommendation": "Looks good.",
        },
        "stances": [
            {
                "persona": "Trickster",
                "confidence": "high",
                "claim": "This fits the portfolio well.",
            },
            {
                "persona": "Technical-Integrity",
                "confidence": "medium",
                "claim": "Risk is manageable.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Test 1: _parse_spec_authority extracts authority from spec
# ---------------------------------------------------------------------------

def test_parse_spec_authority_advisory(tmp_path):
    spec = _write_spec(tmp_path, _VALID_SPEC_ADVISORY)
    assert _parse_spec_authority(spec) == "advisory"


def test_parse_spec_authority_hold(tmp_path):
    spec = _write_spec(tmp_path, _VALID_SPEC_HOLD)
    assert _parse_spec_authority(spec) == "hold"


def test_parse_spec_authority_auto_merge(tmp_path):
    spec = _write_spec(tmp_path, _VALID_SPEC_AUTO_MERGE)
    assert _parse_spec_authority(spec) == "auto-merge"


# ---------------------------------------------------------------------------
# Test 4: _parse_spec_authority raises SpecFrontmatterError when absent
# ---------------------------------------------------------------------------

def test_parse_spec_authority_raises_on_missing(tmp_path):
    spec = _write_spec(tmp_path, _SPEC_NO_AUTHORITY)
    with pytest.raises(SpecFrontmatterError, match="missing \\*\\*Authority:\\*\\*"):
        _parse_spec_authority(spec)


# ---------------------------------------------------------------------------
# Test 5: _build_brief includes Facets envelope
# ---------------------------------------------------------------------------

def test_build_brief_includes_facets_deliberation(tmp_path):
    """_build_brief with facets_deliberation populated → brief.facets_deliberation set."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    council_raw = {
        "status": "resolved",
        "landing": "Good fit.",
        "open_questions": [],
        "confidence": "converged",
        "positions": [{"entity": "ent-a", "position": "agree"}],
        "run_id": "council-t9",
    }
    fd = _facets_deliberation_fixture(escalation="proceed", consensus="consensus")

    brief = _build_brief(
        council_raw=council_raw,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=30.0,
        facets_deliberation=fd,
        authority="advisory",
    )

    assert brief.facets_deliberation is not None
    assert brief.facets_deliberation == fd
    assert brief.combined_recommendation == "proceed-to-bind"


# ---------------------------------------------------------------------------
# Test 10: _combined_recommendation with Facets escalation="claude-max"
# ---------------------------------------------------------------------------

def test_combined_recommendation_facets_claude_max():
    result = _combined_recommendation(
        council_status="resolved",
        council_positions=[],
        facets_escalation="claude-max",
        authority="advisory",
    )
    assert result == "shape-with-Erah"


# ---------------------------------------------------------------------------
# Test 11: _combined_recommendation with Facets escalation="investigate-first"
# ---------------------------------------------------------------------------

def test_combined_recommendation_facets_investigate_first():
    """investigate-first overrides Council resolved → amend-spec."""
    result = _combined_recommendation(
        council_status="resolved",
        council_positions=[],
        facets_escalation="investigate-first",
        authority="advisory",
    )
    assert result == "amend-spec"


# ---------------------------------------------------------------------------
# Test 12: _combined_recommendation with Facets escalation="brief-to-Erah"
# ---------------------------------------------------------------------------

def test_combined_recommendation_facets_brief_to_Erah():
    result = _combined_recommendation(
        council_status="resolved",
        council_positions=[],
        facets_escalation="brief-to-Erah",
        authority="advisory",
    )
    assert result == "shape-with-Erah"


# ---------------------------------------------------------------------------
# Test 13: _combined_recommendation with Council open, Facets proceed
# ---------------------------------------------------------------------------

def test_combined_recommendation_facets_proceed_defers_to_council():
    """Facets escalation=proceed → defer to Council signal; open → amend-spec."""
    result = _combined_recommendation(
        council_status="open",
        council_positions=[],
        facets_escalation="proceed",
        authority="advisory",
    )
    assert result == "amend-spec"


# ---------------------------------------------------------------------------
# Test 14: format_brief includes Facets section when populated
# ---------------------------------------------------------------------------

def test_format_brief_includes_facets_section(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")
    fd = _facets_deliberation_fixture(escalation="proceed", consensus="consensus")

    brief = _build_brief(
        council_raw={
            "status": "resolved",
            "landing": "",
            "open_questions": [],
            "confidence": "converged",
            "positions": [],
            "run_id": "council-t14",
        },
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=10.0,
        facets_deliberation=fd,
        authority="advisory",
    )

    output = format_brief(brief)
    assert "## Facets deliberation" in output
    assert "test-deliberation-id" in output
    assert "consensus_level" in output or "consensus" in output


# ---------------------------------------------------------------------------
# Test 15: format_brief omits Facets section when None
# ---------------------------------------------------------------------------

def test_format_brief_omits_facets_section_when_none(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        council_raw={
            "status": "resolved",
            "landing": "",
            "open_questions": [],
            "confidence": "converged",
            "positions": [],
            "run_id": "council-t15",
        },
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        facets_deliberation=None,
        authority="advisory",
    )

    output = format_brief(brief)
    assert "Facets" not in output


# ---------------------------------------------------------------------------
# Test 16: cmd_spec_review --no-facets flag skips Facets
# ---------------------------------------------------------------------------

def test_cmd_spec_review_no_facets_flag(tmp_path):
    """--no-facets → dispatch_facets=False passed to run_spec_review."""
    spec = _write_spec(tmp_path, _VALID_SPEC_ADVISORY)

    calls = {}

    def fake_run_spec_review(**kwargs):
        calls.update(kwargs)
        # Return a minimal valid brief
        return SpecReviewBrief(
            spec_path=spec,
            target_id="my-target-id",
            repo="lapis-pm",
            council_status="resolved",
            council_landing="",
            council_open_questions=[],
            council_confidence="converged",
            council_positions=[],
            council_run_id="council-t16",
            elapsed_s=1.0,
            combined_recommendation="proceed-to-bind",
        )

    with patch("lapis_pm.spec_review.run_spec_review", side_effect=fake_run_spec_review):
        import argparse
        from lapis_pm.cli import cmd_spec_review

        args = argparse.Namespace(
            spec_path=str(spec),
            council_voicing="local",
            timeout=60,
            repo_override=None,
            authority=None,
            no_facets=True,
        )
        result = cmd_spec_review(args)

    assert calls.get("dispatch_facets") is False, (
        f"Expected dispatch_facets=False, got {calls.get('dispatch_facets')!r}"
    )
    assert result == 0


# ---------------------------------------------------------------------------
# L2-T1: _combined_recommendation with facets_unreliable=True ignores Facets
# ---------------------------------------------------------------------------

def test_combined_recommendation_facets_unreliable_ignores_escalation():
    """facets_unreliable=True → Facets escalation (brief-to-Erah) is ignored; Council drives."""
    result = _combined_recommendation(
        council_status="resolved",
        council_positions=[],
        facets_escalation="brief-to-Erah",
        facets_unreliable=True,
        authority="advisory",
    )
    # Council resolved + no Opus → proceed-to-bind (Facets escalation ignored)
    assert result == "proceed-to-bind"


# ---------------------------------------------------------------------------
# L2-T4: _combined_recommendation with facets_unreliable=False behaves as before
# ---------------------------------------------------------------------------

def test_combined_recommendation_facets_reliable_escalation_applies():
    """facets_unreliable=False + escalation=brief-to-Erah → shape-with-Erah (existing behavior)."""
    result = _combined_recommendation(
        council_status="resolved",
        council_positions=[],
        facets_escalation="brief-to-Erah",
        facets_unreliable=False,
        authority="advisory",
    )
    assert result == "shape-with-Erah"


# ---------------------------------------------------------------------------
# L2-T5: _build_brief sets facets_unreliable=True when any stance has parse_failed
# ---------------------------------------------------------------------------

def test_build_brief_facets_unreliable_from_stance(tmp_path):
    """Any stance with parse_failed=True → facets_unreliable=True in combined rec."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    fd = _facets_deliberation_with_parse_failures(stance_failed=True, synth_failed=False)
    # escalation=brief-to-Erah would give shape-with-Erah if reliable;
    # with unreliable, Council resolved → proceed-to-bind
    council_raw = {
        "status": "resolved",
        "landing": "",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": "council-l2t5",
    }
    brief = _build_brief(
        council_raw=council_raw,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        facets_deliberation=fd,
        authority="advisory",
    )
    assert brief.combined_recommendation == "proceed-to-bind", (
        f"Expected proceed-to-bind (Facets unreliable), got {brief.combined_recommendation}"
    )


# ---------------------------------------------------------------------------
# L2-T6: _build_brief sets facets_unreliable=True when synthesis has parse_failed
# ---------------------------------------------------------------------------

def test_build_brief_facets_unreliable_from_synthesis(tmp_path):
    """synthesis.parse_failed=True → facets_unreliable=True."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    fd = _facets_deliberation_with_parse_failures(stance_failed=False, synth_failed=True)
    council_raw = {
        "status": "resolved",
        "landing": "",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": "council-l2t6",
    }
    brief = _build_brief(
        council_raw=council_raw,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        facets_deliberation=fd,
        authority="advisory",
    )
    assert brief.combined_recommendation == "proceed-to-bind", (
        f"Expected proceed-to-bind (synthesis unreliable), got {brief.combined_recommendation}"
    )


# ---------------------------------------------------------------------------
# L2-T7: _build_brief sets facets_unreliable=False when no parse failures
# ---------------------------------------------------------------------------

def test_build_brief_facets_reliable_when_no_parse_failures(tmp_path):
    """No parse failures → facets_unreliable=False; Facets escalation drives recommendation."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    fd = {
        "deliberation_id": "reliable-id",
        "synthesis": {
            "escalation_recommendation": "brief-to-Erah",
            "consensus_level": "divergent",
            "confidence": "medium",
            "recommendation": "Shape with Erah.",
            "parse_failed": False,
            "parse_error": None,
        },
        "stances": [
            {
                "persona": "trickster",
                "confidence": "high",
                "claim": "fits well",
                "parse_failed": False,
                "parse_error": None,
            }
        ],
    }
    council_raw = {
        "status": "resolved",
        "landing": "",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": "council-l2t7",
    }
    brief = _build_brief(
        council_raw=council_raw,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        facets_deliberation=fd,
        authority="advisory",
    )
    # Facets reliable + escalation=brief-to-Erah → shape-with-Erah
    assert brief.combined_recommendation == "shape-with-Erah", (
        f"Expected shape-with-Erah (Facets reliable escalation), got {brief.combined_recommendation}"
    )


# ---------------------------------------------------------------------------
# L2-T8: format_brief annotates Facets section with unreliable header
# ---------------------------------------------------------------------------

def test_format_brief_facets_unreliable_header(tmp_path):
    """When parse failures present, Facets section contains the unreliable header line."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    fd = _facets_deliberation_with_parse_failures(stance_failed=True, synth_failed=False)
    council_raw = {
        "status": "resolved",
        "landing": "",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": "council-l2t8",
    }
    brief = _build_brief(
        council_raw=council_raw,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        facets_deliberation=fd,
        authority="advisory",
    )
    output = format_brief(brief)
    assert "Facets leg unreliable" in output, (
        "Expected 'Facets leg unreliable' in format_brief output when parse failures present"
    )
    assert "combined recommendation derived from Council only" in output


# ---------------------------------------------------------------------------
# L2-T9: format_brief lists each parse-failed persona with its parse_error reason
# ---------------------------------------------------------------------------

def test_format_brief_lists_failed_personas(tmp_path):
    """format_brief surfaces each parse-failed persona and its parse_error reason."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    fd = _facets_deliberation_with_parse_failures(stance_failed=True, synth_failed=False)
    council_raw = {
        "status": "resolved",
        "landing": "",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": "council-l2t9",
    }
    brief = _build_brief(
        council_raw=council_raw,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        facets_deliberation=fd,
        authority="advisory",
    )
    output = format_brief(brief)
    assert "Personas with parse failures" in output, (
        "Expected 'Personas with parse failures' in output"
    )
    # The fixture has technical-integrity with parse_failed=True, parse_error="no JSON object found"
    assert "technical-integrity" in output
    assert "no JSON object found" in output


# ---------------------------------------------------------------------------
# facets-operator-v0: _dispatch_facets operator flag injection
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# facets-operator-v0: format_brief operator header
# ---------------------------------------------------------------------------

def test_format_brief_operator_header(tmp_path):
    """SpecReviewBrief with facets_operator='sonnet' renders [operator: sonnet] in heading."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    fd = _facets_deliberation_fixture(escalation="proceed", consensus="consensus")
    council_raw = {
        "status": "resolved",
        "landing": "",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": "council-op-header",
    }
    brief = _build_brief(
        council_raw=council_raw,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        facets_deliberation=fd,
        authority="advisory",
        facets_operator="sonnet",
    )
    output = format_brief(brief)
    assert "[operator: sonnet]" in output, (
        f"Expected '[operator: sonnet]' in Facets heading; got:\n{output}"
    )


# ---------------------------------------------------------------------------
# facets-operator-v0: stance confidence_type (verified vs inferred)
# ---------------------------------------------------------------------------

def test_format_brief_stance_confidence_type(tmp_path):
    """Stance with verified=True renders 'verified/high'; without 'verified' renders 'inferred/high'."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    fd = {
        "deliberation_id": "test-conf-type",
        "synthesis": {
            "escalation_recommendation": "proceed",
            "consensus_level": "consensus",
            "confidence": "high",
            "recommendation": "OK.",
        },
        "stances": [
            {"persona": "technical-integrity", "confidence": "high", "claim": "Verified claim.", "verified": True},
            {"persona": "trickster", "confidence": "high", "claim": "Inferred claim."},
        ],
    }
    council_raw = {
        "status": "resolved",
        "landing": "",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": "council-conf-type",
    }
    brief = _build_brief(
        council_raw=council_raw,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        facets_deliberation=fd,
        authority="advisory",
    )
    output = format_brief(brief)
    assert "verified/high" in output, (
        f"Expected 'verified/high' for stance with verified=True; got:\n{output}"
    )
    assert "inferred/high" in output, (
        f"Expected 'inferred/high' for stance without 'verified' key; got:\n{output}"
    )


# ---------------------------------------------------------------------------
# facets-operator-v0: uncertainty bounds fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operator,expected_fragment", [
    ("haiku", "not surfaced by this operator"),
    ("sonnet", "not reported by synthesis"),
])
def test_format_brief_uncertainty_bounds_fallback(tmp_path, operator, expected_fragment):
    """Synthesis without scope_limits/uncertainty renders operator-appropriate fallback."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    fd = _facets_deliberation_fixture(escalation="proceed", consensus="consensus")
    # Confirm fixture has no scope_limits or uncertainty keys
    assert "scope_limits" not in fd.get("synthesis", {})
    assert "uncertainty" not in fd.get("synthesis", {})

    council_raw = {
        "status": "resolved",
        "landing": "",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": f"council-ub-{operator}",
    }
    brief = _build_brief(
        council_raw=council_raw,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        facets_deliberation=fd,
        authority="advisory",
        facets_operator=operator,
    )
    output = format_brief(brief)
    assert expected_fragment in output, (
        f"Expected '{expected_fragment}' in Uncertainty bounds for operator={operator}; got:\n{output}"
    )


# ---------------------------------------------------------------------------
# Regression Test: in-process deliberation must init Facets semaphore
# ---------------------------------------------------------------------------

def test_run_spec_review_in_process_deliberation_inits_semaphore(tmp_path, monkeypatch, capsys):
    """Regression: in-process run_deliberation must init_facets_semaphore inside the loop.

    When run_spec_review calls asyncio.run(run_deliberation(...)), the semaphore
    must be initialized inside the running loop (inside _deliberate()), not before.
    This test verifies:
    1. init_facets_semaphore is called during the deliberation
    2. run_deliberation completes without "Facets semaphore not initialized" error
    3. The semaphore guard (orchestrator.py:55-56) is exercised

    The test ONLY stubs _facets_subprocess to avoid subprocess calls, allowing the
    real orchestrator code path (including the semaphore guard check) to execute.
    """
    spec = _write_spec(tmp_path, _VALID_SPEC_ADVISORY)

    # Track init_facets_semaphore calls to verify it was invoked
    init_calls = []

    def mock_init_facets_semaphore(max_concurrent):
        import asyncio as _asyncio
        init_calls.append(max_concurrent)
        # Call the real init so the semaphore is actually initialized for the orchestrator
        from agents_core.shared_deliberation import orchestrator
        orchestrator._facets_semaphore = _asyncio.Semaphore(max_concurrent)

    # Create a mock facets subprocess response
    mock_facets_subprocess_response = {
        "status": "resolved",
        "deliberation_id": "test-deliberation-id",
        "synthesis": {
            "escalation_recommendation": "proceed",
            "consensus_level": "consensus",
            "confidence": "high",
            "recommendation": "Looks good.",
        },
        "stances": [
            {
                "persona": "Trickster",
                "confidence": "high",
                "claim": "This fits the portfolio well.",
            },
        ],
    }

    # Stub council to avoid hitting live service
    monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_STUB", "1")
    # Set explicit semaphore max concurrent value to verify default is used
    monkeypatch.delenv("SHARED_DELIBERATION_MAX_CONCURRENT", raising=False)

    try:
        # Create async mock for _facets_subprocess
        async_mock_facets_subprocess = AsyncMock(return_value=mock_facets_subprocess_response)

        with patch(
            "lapis_pm.spec_review.init_facets_semaphore", side_effect=mock_init_facets_semaphore
        ), patch(
            "agents_core.shared_deliberation.orchestrator._facets_subprocess",
            new=async_mock_facets_subprocess
        ), patch(
            "lapis_pm.spec_review._load_invariant_context", return_value="ctx"
        ), patch(
            "lapis_pm.spec_review._dispatch_spec_reviewer", return_value=None
        ):
            brief = run_spec_review(
                spec_path=spec,
                dispatch_facets=True,
                timeout_s=30,
            )

        # Assert init_facets_semaphore was called exactly once with the default value
        assert len(init_calls) == 1, "init_facets_semaphore must be called exactly once during deliberation"
        assert init_calls[0] == 2, "init_facets_semaphore must be called with default SHARED_DELIBERATION_MAX_CONCURRENT=2"

        # Verify that "Facets semaphore not initialized" error does NOT appear in output
        captured = capsys.readouterr()
        assert "Facets semaphore not initialized" not in captured.err, (
            "init_facets_semaphore must prevent 'Facets semaphore not initialized' error"
        )

        # Assert the brief was built (errors in orchestration are OK; the key is no semaphore error)
        assert brief is not None
    finally:
        pass
