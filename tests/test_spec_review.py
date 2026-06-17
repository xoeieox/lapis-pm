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
    """spec_reviewer_task_id=None → only Council polled; sonnet_result=None returned."""
    run_id = f"council-t8-{int(time.time())}"
    council_path = _write_council_yaml(run_id, status="resolved")
    try:
        sonnet_raw, council_raw = _poll_until_terminal(
            council_run_id=run_id,
            timeout_s=60,
            start_time=time.time(),
            # spec_reviewer_task_id defaults to None
        )
        assert sonnet_raw is None
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


# ---------------------------------------------------------------------------
# L2-T1: _dispatch_facets passes spec_text via context file, excludes mirror-rep
# ---------------------------------------------------------------------------

def test_dispatch_facets_passes_spec_text_via_context_file_and_excludes_mirror_rep():
    """_dispatch_facets writes spec_text to the context tempfile, does not use stdin,
    and passes --personas technical-integrity,trickster as a single argv pair."""
    fake_output = json.dumps({"deliberation_id": "delib-l2t1"})
    captured_calls = []
    captured_context_files = []

    def mock_run(cmd, **kwargs):
        # Capture context-file contents before lapis-pm unlinks it
        try:
            ctx_path = cmd[cmd.index("--context-file") + 1]
            captured_context_files.append(json.loads(Path(ctx_path).read_text()))
        except (ValueError, IndexError, FileNotFoundError):
            pass
        captured_calls.append({"cmd": cmd, "input": kwargs.get("input")})
        return CompletedProcess(args=cmd, returncode=0, stdout=fake_output, stderr="")

    with patch("subprocess.run", side_effect=mock_run), patch.dict(
        os.environ, {"FACETS_DISPATCH_DISABLED": ""}, clear=False
    ):
        result = _dispatch_facets(
            spec_text="the spec content here",
            parsed_target_id="my-target",
            repo="lapis-pm",
            authority="advisory",
            start_time=time.time(),
        )

    assert result == "delib-l2t1"
    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert "--spec-text-from-stdin" not in call["cmd"]
    assert call["input"] is None or call["input"] == ""
    # --personas is a single argv pair with comma-separated value
    personas_idx = call["cmd"].index("--personas")
    assert call["cmd"][personas_idx + 1] == "technical-integrity,trickster"
    # mirror-rep is not in the personas list anywhere
    assert "mirror-rep" not in call["cmd"]
    # spec_text travels in the context file
    assert len(captured_context_files) == 1
    assert captured_context_files[0].get("spec_text") == "the spec content here"


# ---------------------------------------------------------------------------
# L2-T2: _dispatch_facets respects FACETS_DISPATCH_DISABLED=1
# ---------------------------------------------------------------------------

def test_dispatch_facets_disabled_returns_none_no_subprocess():
    """FACETS_DISPATCH_DISABLED=1 → returns None without calling subprocess."""
    calls = []
    with patch("subprocess.run", side_effect=lambda *a, **k: calls.append(a)), \
         patch.dict(os.environ, {"FACETS_DISPATCH_DISABLED": "1"}):
        result = _dispatch_facets(
            spec_text="spec",
            parsed_target_id="tid",
            repo="repo",
            authority="advisory",
            start_time=time.time(),
        )
    assert result is None
    assert calls == [], "subprocess.run must not be called when FACETS_DISPATCH_DISABLED=1"


# ---------------------------------------------------------------------------
# L2-T3: _combined_recommendation with facets_unreliable=True ignores Facets
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

@pytest.mark.parametrize("operator,expect_flags", [
    ("sonnet", True),
    ("haiku", False),
    ("gravitywell", True),
])
def test_dispatch_facets_operator_argv(monkeypatch, operator, expect_flags):
    """facets_operator='sonnet' adds --persona-operator + --synthesis-operator; 'haiku' adds neither."""
    monkeypatch.delenv("FACETS_DISPATCH_DISABLED", raising=False)
    captured_argv = []

    def mock_run(argv, **kwargs):
        captured_argv.extend(argv)
        output = json.dumps({"deliberation_id": "test-op-id"})
        return CompletedProcess(argv, returncode=0, stdout=output, stderr="")

    with patch("subprocess.run", side_effect=mock_run):
        result = _dispatch_facets(
            spec_text="# Spec\nContent.",
            parsed_target_id="op-test-target",
            repo="lapis-pm",
            authority="advisory",
            start_time=time.time(),
            facets_operator=operator,
        )

    assert result == "test-op-id"
    has_persona_flag = "--persona-operator" in captured_argv
    has_synthesis_flag = "--synthesis-operator" in captured_argv
    assert has_persona_flag == expect_flags, (
        f"--persona-operator present={has_persona_flag}, expected={expect_flags} for operator={operator}"
    )
    assert has_synthesis_flag == expect_flags, (
        f"--synthesis-operator present={has_synthesis_flag}, expected={expect_flags} for operator={operator}"
    )
    if expect_flags:
        idx = captured_argv.index("--persona-operator")
        assert captured_argv[idx + 1] == operator
        idx2 = captured_argv.index("--synthesis-operator")
        assert captured_argv[idx2 + 1] == operator


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
# target-repo-v0: --target-repo Mode-1 grounding flag wiring
# ---------------------------------------------------------------------------

def test_dispatch_facets_target_repo_present_when_dir_exists(monkeypatch):
    """--target-repo /srv/git/myrepo-working appended to argv when resolved dir exists."""
    monkeypatch.delenv("FACETS_DISPATCH_DISABLED", raising=False)
    monkeypatch.delenv("FACETS_GROUNDING_DISABLED", raising=False)

    captured_argv = []

    def mock_run(argv, **kwargs):
        captured_argv.extend(argv)
        return CompletedProcess(argv, returncode=0,
                                stdout=json.dumps({"deliberation_id": "grounding-test"}), stderr="")

    with patch("subprocess.run", side_effect=mock_run), \
         patch("pathlib.Path.is_dir", return_value=True):
        result = _dispatch_facets(
            spec_text="spec content",
            parsed_target_id="my-target",
            repo="myrepo",
            authority="advisory",
            start_time=time.time(),
        )

    assert result == "grounding-test"
    assert "--target-repo" in captured_argv
    idx = captured_argv.index("--target-repo")
    assert captured_argv[idx + 1] == "/srv/git/myrepo-working"


def test_dispatch_facets_target_repo_absent_when_dir_missing(monkeypatch, capsys):
    """--target-repo absent + stderr notice when resolved dir does not exist."""
    monkeypatch.delenv("FACETS_DISPATCH_DISABLED", raising=False)
    monkeypatch.delenv("FACETS_GROUNDING_DISABLED", raising=False)

    captured_argv = []

    def mock_run(argv, **kwargs):
        captured_argv.extend(argv)
        return CompletedProcess(argv, returncode=0,
                                stdout=json.dumps({"deliberation_id": "no-grounding"}), stderr="")

    with patch("subprocess.run", side_effect=mock_run), \
         patch("pathlib.Path.is_dir", return_value=False):
        result = _dispatch_facets(
            spec_text="spec",
            parsed_target_id="absent-target",
            repo="absent-repo",
            authority="advisory",
            start_time=time.time(),
        )

    assert result == "no-grounding"
    assert "--target-repo" not in captured_argv
    captured = capsys.readouterr()
    assert "no working tree for repo 'absent-repo'" in captured.err
    assert "Mode-1 grounding inert" in captured.err


def test_dispatch_facets_target_repo_absent_when_grounding_disabled(monkeypatch, tmp_path):
    """FACETS_GROUNDING_DISABLED=1 → --target-repo absent even when dir exists."""
    monkeypatch.delenv("FACETS_DISPATCH_DISABLED", raising=False)
    monkeypatch.setenv("FACETS_GROUNDING_DISABLED", "1")

    fake_repo_path = tmp_path / "somerepo-working"
    fake_repo_path.mkdir()

    captured_argv = []

    def mock_run(argv, **kwargs):
        captured_argv.extend(argv)
        return CompletedProcess(argv, returncode=0,
                                stdout=json.dumps({"deliberation_id": "disabled-grounding"}), stderr="")

    with patch("subprocess.run", side_effect=mock_run), \
         patch("pathlib.Path.is_dir", return_value=True):
        result = _dispatch_facets(
            spec_text="spec",
            parsed_target_id="some-target",
            repo="somerepo",
            authority="advisory",
            start_time=time.time(),
        )

    assert result == "disabled-grounding"
    assert "--target-repo" not in captured_argv
