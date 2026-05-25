"""16 unit tests for facets-pre-bind-wire-v0.

Tests: run_spec_review with Facets disabled, auto-merge authority gating,
_parse_spec_authority, _dispatch_facets, _poll_until_terminal (Council-only),
_build_brief with Facets, _combined_recommendation with Facets escalations,
format_brief Facets section, cmd_spec_review --no-facets flag.
"""
from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.spec_review import (
    _CLAUDE_QUEUE_COMPLETED,
    _COUNCIL_DIR,
    SpecFrontmatterError,
    SpecReviewBrief,
    _build_brief,
    _combined_recommendation,
    _dispatch_facets,
    _parse_spec_authority,
    _poll_until_terminal,
    format_brief,
    run_spec_review,
)


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
# Test 1: run_spec_review with Facets disabled
# ---------------------------------------------------------------------------

def test_run_spec_review_facets_disabled(tmp_path):
    """dispatch_facets=False → brief.facets_deliberation is None; recommendation unchanged."""
    spec = _write_spec(tmp_path, _VALID_SPEC_ADVISORY)
    run_id = f"council-t1-{int(time.time())}"
    council_path = _write_council_yaml(run_id, status="resolved")

    try:
        with patch(
            "lapis_pm.spec_review._dispatch_council", return_value=run_id
        ), patch(
            "lapis_pm.spec_review._load_invariant_context", return_value="ctx"
        ):
            brief = run_spec_review(
                spec_path=spec,
                dispatch_facets=False,
                timeout_s=30,
            )

        assert brief.facets_deliberation is None
        assert brief.combined_recommendation == "proceed-to-bind"
    finally:
        if council_path.exists():
            council_path.unlink()


# ---------------------------------------------------------------------------
# Test 2: run_spec_review for auto-merge authority — Facets not dispatched
# ---------------------------------------------------------------------------

def test_run_spec_review_auto_merge_skips_facets(tmp_path):
    """Auto-merge spec: Facets not dispatched even if dispatch_facets=True."""
    spec = _write_spec(tmp_path, _VALID_SPEC_AUTO_MERGE)
    run_id = f"council-t2-{int(time.time())}"
    council_path = _write_council_yaml(run_id, status="resolved")

    try:
        dispatch_calls = []
        original_dispatch = _dispatch_facets

        def mock_facets(*args, **kwargs):
            dispatch_calls.append(args)
            return None

        with patch(
            "lapis_pm.spec_review._dispatch_facets", side_effect=mock_facets
        ), patch(
            "lapis_pm.spec_review._dispatch_council", return_value=run_id
        ), patch(
            "lapis_pm.spec_review._load_invariant_context", return_value="ctx"
        ):
            brief = run_spec_review(
                spec_path=spec,
                dispatch_facets=True,
                timeout_s=30,
            )

        assert dispatch_calls == [], "Facets must not be dispatched for auto-merge"
        assert brief.facets_deliberation is None
    finally:
        if council_path.exists():
            council_path.unlink()


# ---------------------------------------------------------------------------
# Test 3: _parse_spec_authority extracts authority from spec
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
# Test 5: _dispatch_facets returns deliberation_id on success
# ---------------------------------------------------------------------------

def test_dispatch_facets_returns_deliberation_id_on_success():
    """Mock subprocess.run to return valid JSON with deliberation_id."""
    fake_output = json.dumps({"deliberation_id": "test-delib-id"})
    mock_result = CompletedProcess(args=[], returncode=0, stdout=fake_output, stderr="")

    with patch("subprocess.run", return_value=mock_result), patch.dict(
        os.environ, {"FACETS_DISPATCH_DISABLED": ""}, clear=False
    ):
        result = _dispatch_facets(
            spec_text="spec content",
            parsed_target_id="my-target",
            repo="lapis-pm",
            authority="advisory",
            start_time=time.time(),
        )

    assert result == "test-delib-id"


# ---------------------------------------------------------------------------
# Test 6: _dispatch_facets returns None when disabled
# ---------------------------------------------------------------------------

def test_dispatch_facets_returns_none_when_disabled():
    """FACETS_DISPATCH_DISABLED=1 → returns None without touching subprocess."""
    with patch.dict(os.environ, {"FACETS_DISPATCH_DISABLED": "1"}):
        result = _dispatch_facets(
            spec_text="spec",
            parsed_target_id="tid",
            repo="repo",
            authority="advisory",
            start_time=time.time(),
        )
    assert result is None


# ---------------------------------------------------------------------------
# Test 7: _dispatch_facets returns None on subprocess error
# ---------------------------------------------------------------------------

def test_dispatch_facets_returns_none_on_subprocess_error(capsys):
    """Subprocess exits with code 1 → returns None and logs error."""
    mock_result = CompletedProcess(
        args=[], returncode=1, stdout="", stderr="some error"
    )
    with patch("subprocess.run", return_value=mock_result), patch.dict(
        os.environ, {"FACETS_DISPATCH_DISABLED": ""}, clear=False
    ):
        result = _dispatch_facets(
            spec_text="spec",
            parsed_target_id="tid",
            repo="repo",
            authority="advisory",
            start_time=time.time(),
        )

    assert result is None
    captured = capsys.readouterr()
    assert "facets-dispatch-error" in captured.err


# ---------------------------------------------------------------------------
# Test 8: _poll_until_terminal polls Council only (Facets mode)
# ---------------------------------------------------------------------------

def test_poll_until_terminal_council_only():
    """spec_reviewer_task_id=None → only Council polled; opus_result=None returned."""
    run_id = f"council-t8-{int(time.time())}"
    council_path = _write_council_yaml(run_id, status="resolved")
    try:
        opus_raw, council_raw = _poll_until_terminal(
            council_run_id=run_id,
            timeout_s=60,
            start_time=time.time(),
            # spec_reviewer_task_id defaults to None
        )
        assert opus_raw is None
        assert council_raw is not None
        assert council_raw["status"] == "resolved"
        assert council_raw["landing"] == "test landing"
    finally:
        if council_path.exists():
            council_path.unlink()


# ---------------------------------------------------------------------------
# Test 9: _build_brief includes Facets envelope
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
