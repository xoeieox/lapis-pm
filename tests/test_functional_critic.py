"""Unit tests for functional_critic.py (functional-critic-v0 DoD).

Covers per-DoD requirements:
- AC extraction (valid spec, no-AC spec)
- Verdict schema validation
- unverifiable path (GW unavailable / doorman unreachable)
- teardown-on-exception (worktree removal in finally)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from lapis_pm.functional_critic import (
    extract_acs,
    validate_verdict,
    run_functional_critic,
    write_verdict_artifact,
    unexercised_ac_count,
    verdict_summary_text,
    _plant_sandbox_shims,
    _remove_worktree,
)


# ---------------------------------------------------------------------------
# AC extraction tests
# ---------------------------------------------------------------------------

_SPEC_WITH_ACS = """\
## Acceptance Criteria

**AC1 (worktree isolation):** The critic checks out the PR head SHA into a fresh
ephemeral worktree.

**AC2 (AC extraction):** Given a spec with numbered ACs, the critic extracts them.

AC3 (oracle lens): For each changed test file, the critic assesses whether the
test asserts intent.
"""

_SPEC_NO_ACS = """\
## Description

This spec has no numbered acceptance criteria. Just prose.

- Feature A should work
- Feature B should be tested
"""


def test_extract_acs_valid_spec():
    acs = extract_acs(_SPEC_WITH_ACS)
    assert len(acs) == 3
    assert acs[0]["ac_id"] == "AC1"
    assert acs[1]["ac_id"] == "AC2"
    assert acs[2]["ac_id"] == "AC3"
    assert "worktree" in acs[0]["claim"].lower()


def test_extract_acs_no_acs():
    acs = extract_acs(_SPEC_NO_ACS)
    assert acs == []


def test_extract_acs_empty_string():
    assert extract_acs("") == []


def test_extract_acs_claim_truncated():
    long_claim = "x" * 1000
    spec = f"AC1 (foo): {long_claim}\n\nAC2 (bar): short\n"
    acs = extract_acs(spec)
    assert len(acs) >= 1
    assert len(acs[0]["claim"]) <= 400


# ---------------------------------------------------------------------------
# Verdict schema validation tests
# ---------------------------------------------------------------------------

_VALID_VERDICT = {
    "ac_verdicts": [
        {
            "ac_id": "AC1",
            "claim": "worktree isolation",
            "verdict": "pass",
            "evidence": "found test at tests/test_wt.py line 12",
            "notes": "",
        },
        {
            "ac_id": "AC2",
            "claim": "AC extraction",
            "verdict": "could_not_exercise",
            "evidence": "no CLI entrypoint found in the diff",
            "notes": "surface is internal module only",
        },
    ],
    "oracle_assessment": {
        "summary": "tests assert intent",
        "suspicious_tests": [],
    },
    "overall": "partial",
}


def test_validate_verdict_valid():
    errors = validate_verdict(_VALID_VERDICT)
    assert errors == []


def test_validate_verdict_missing_evidence():
    bad = {
        "ac_verdicts": [
            {"ac_id": "AC1", "claim": "foo", "verdict": "pass", "evidence": ""},
        ],
        "oracle_assessment": {"summary": "", "suspicious_tests": []},
        "overall": "pass",
    }
    errors = validate_verdict(bad)
    assert any("evidence" in e for e in errors)


def test_validate_verdict_invalid_overall():
    bad = dict(_VALID_VERDICT)
    bad = {**_VALID_VERDICT, "overall": "unknown_value"}
    errors = validate_verdict(bad)
    assert any("overall" in e for e in errors)


def test_validate_verdict_invalid_verdict_value():
    bad = {
        "ac_verdicts": [
            {"ac_id": "AC1", "claim": "x", "verdict": "maybe", "evidence": "some evidence"},
        ],
        "oracle_assessment": {"summary": "", "suspicious_tests": []},
        "overall": "pass",
    }
    errors = validate_verdict(bad)
    assert any("invalid verdict" in e for e in errors)


def test_validate_verdict_ac_verdicts_not_list():
    bad = {"ac_verdicts": "not a list", "overall": "pass",
           "oracle_assessment": {"summary": "", "suspicious_tests": []}}
    errors = validate_verdict(bad)
    assert any("list" in e for e in errors)


def test_validate_verdict_missing_oracle():
    bad = {**_VALID_VERDICT, "oracle_assessment": "not a dict"}
    errors = validate_verdict(bad)
    assert any("oracle_assessment" in e for e in errors)


# ---------------------------------------------------------------------------
# unverifiable path (GW unavailable / doorman unreachable)
# ---------------------------------------------------------------------------

def test_run_functional_critic_no_acs_returns_unverifiable():
    """Spec with no ACs → unverifiable without hitting GW."""
    with patch("lapis_pm.functional_critic._git_fetch"), \
         patch("lapis_pm.functional_critic._create_worktree", return_value=False):
        verdict = run_functional_critic(
            target_id="test-target",
            pr_number=99,
            head_sha="abc12345",
            repo="lapis-pm",
            spec_text=_SPEC_NO_ACS,
            run_id="test-run-1",
        )
    assert verdict["overall"] == "unverifiable"
    assert "no numbered ACs" in verdict["oracle_assessment"]["summary"]


def test_run_functional_critic_gw_unavailable_returns_unverifiable(tmp_path):
    """When GW/doorman is unavailable, returns unverifiable without paid fallback."""
    with patch("lapis_pm.functional_critic._git_fetch"), \
         patch("lapis_pm.functional_critic._create_worktree", return_value=True), \
         patch("lapis_pm.functional_critic._plant_sandbox_shims"), \
         patch("lapis_pm.functional_critic._run_pytest",
               return_value={"returncode": 0, "stdout": "", "stderr": "", "elapsed_s": 0.1, "timed_out": False}), \
         patch("lapis_pm.functional_critic._check_doorman", return_value=False), \
         patch("lapis_pm.functional_critic._remove_worktree") as mock_rm:
        verdict = run_functional_critic(
            target_id="test-target",
            pr_number=42,
            head_sha="deadbeef",
            repo="lapis-pm",
            spec_text=_SPEC_WITH_ACS,
            run_id="test-run-2",
        )
    assert verdict["overall"] == "unverifiable"
    assert verdict["oracle_assessment"]["summary"] == "gw_unavailable"
    # Worktree MUST be torn down even when GW is unavailable
    mock_rm.assert_called_once()


def test_run_functional_critic_worktree_creation_fails():
    """Worktree creation failure → unverifiable, no GW call."""
    with patch("lapis_pm.functional_critic._git_fetch"), \
         patch("lapis_pm.functional_critic._create_worktree", return_value=False), \
         patch("lapis_pm.functional_critic._call_gw_critic") as mock_gw:
        verdict = run_functional_critic(
            target_id="test-target",
            pr_number=5,
            head_sha="cafebabe",
            repo="lapis-pm",
            spec_text=_SPEC_WITH_ACS,
            run_id="test-run-3",
        )
    assert verdict["overall"] == "unverifiable"
    assert "worktree creation failed" in verdict["oracle_assessment"]["summary"]
    mock_gw.assert_not_called()


# ---------------------------------------------------------------------------
# Teardown-on-exception test
# ---------------------------------------------------------------------------

def test_worktree_torn_down_on_exception(tmp_path):
    """Worktree MUST be removed even if an exception is raised mid-run."""
    with patch("lapis_pm.functional_critic._git_fetch"), \
         patch("lapis_pm.functional_critic._create_worktree", return_value=True), \
         patch("lapis_pm.functional_critic._plant_sandbox_shims"), \
         patch("lapis_pm.functional_critic._run_pytest",
               side_effect=RuntimeError("unexpected explosion")), \
         patch("lapis_pm.functional_critic._remove_worktree") as mock_rm:
        with pytest.raises(RuntimeError, match="unexpected explosion"):
            run_functional_critic(
                target_id="test-target",
                pr_number=7,
                head_sha="aabbccdd",
                repo="lapis-pm",
                spec_text=_SPEC_WITH_ACS,
                run_id="test-run-4",
            )
    mock_rm.assert_called_once()


# ---------------------------------------------------------------------------
# Sandbox shims test
# ---------------------------------------------------------------------------

def test_plant_sandbox_shims(tmp_path):
    worktree = str(tmp_path)
    _plant_sandbox_shims(worktree)
    shim_dir = tmp_path / ".critic-bin"
    assert shim_dir.exists()
    for cmd in ("pip", "pip3", "curl", "wget", "nc"):
        shim = shim_dir / cmd
        assert shim.exists(), f"shim {cmd} not found"
        assert shim.stat().st_mode & 0o111, f"shim {cmd} not executable"
    # Verify shim actually blocks execution
    result = subprocess.run(
        [str(shim_dir / "pip"), "install", "requests"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "blocked" in result.stderr


# ---------------------------------------------------------------------------
# Artifact writer test
# ---------------------------------------------------------------------------

def test_write_verdict_artifact(tmp_path):
    with patch("lapis_pm.functional_critic.ARTIFACTS_BASE", tmp_path):
        path = write_verdict_artifact("run-123", 42, _VALID_VERDICT)
    assert path.exists()
    written = json.loads(path.read_text())
    assert written["overall"] == "partial"
    assert "ac_verdicts" in written


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_unexercised_ac_count():
    verdict = {
        "ac_verdicts": [
            {"verdict": "pass"},
            {"verdict": "could_not_exercise"},
            {"verdict": "could_not_exercise"},
            {"verdict": "fail"},
        ]
    }
    assert unexercised_ac_count(verdict) == 2


def test_unexercised_ac_count_empty():
    assert unexercised_ac_count({}) == 0


def test_verdict_summary_text_includes_overall():
    summary = verdict_summary_text(_VALID_VERDICT)
    assert "partial" in summary
    assert "AC1" in summary
    assert "AC2" in summary


def test_verdict_summary_text_unverifiable():
    verdict = {
        "ac_verdicts": [],
        "oracle_assessment": {"summary": "gw_unavailable", "suspicious_tests": []},
        "overall": "unverifiable",
        "provenance": {"run_id": "test-run"},
    }
    summary = verdict_summary_text(verdict)
    assert "unverifiable" in summary


# ---------------------------------------------------------------------------
# Provenance is always set
# ---------------------------------------------------------------------------

def test_run_functional_critic_provenance_always_present(tmp_path):
    """Every verdict from run_functional_critic must carry provenance."""
    mock_gw_verdict = {
        "ac_verdicts": [
            {"ac_id": "AC1", "claim": "x", "verdict": "pass", "evidence": "e", "notes": ""},
        ],
        "oracle_assessment": {"summary": "ok", "suspicious_tests": []},
        "overall": "pass",
    }
    with patch("lapis_pm.functional_critic._git_fetch"), \
         patch("lapis_pm.functional_critic._create_worktree", return_value=True), \
         patch("lapis_pm.functional_critic._plant_sandbox_shims"), \
         patch("lapis_pm.functional_critic._run_pytest",
               return_value={"returncode": 0, "stdout": "1 passed", "stderr": "", "elapsed_s": 0.5, "timed_out": False}), \
         patch("lapis_pm.functional_critic._call_gw_critic", return_value=mock_gw_verdict), \
         patch("lapis_pm.functional_critic._remove_worktree"):
        verdict = run_functional_critic(
            target_id="test-target",
            pr_number=1,
            head_sha="12345678",
            repo="lapis-pm",
            spec_text=_SPEC_WITH_ACS,
            run_id="test-run-prov",
        )
    prov = verdict.get("provenance")
    assert prov is not None
    assert prov.get("pr_head_sha") == "12345678"
    assert prov.get("gw_model") == "gravitywell-122b"
    assert prov.get("run_id") == "test-run-prov"
