"""Tests: backcaster grounding pre-flight gate (backcaster-grounding-hard-error-v0).

Acceptance criteria:
  1. Synapse unreachable + no --allow-degraded -> RuntimeError before stage-1 decompose.
  2. mem unreachable + no --allow-degraded -> RuntimeError before stage-1 decompose.
  3. Both unreachable + no --allow-degraded -> RuntimeError naming both deps.
  4. --allow-degraded -> run proceeds; degraded_paths in run.yaml; DEGRADED banner in roadmap.md.
  5. Both grounding deps healthy -> clean run (no banner, no degraded_paths).
  6. stub=True bypasses pre-flight entirely.
  7. No live Synapse/mem calls - all probes mocked.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goal_file(tmp_path: Path) -> Path:
    g = tmp_path / "goal.md"
    g.write_text("# Goal\nTest goal state.", encoding="utf-8")
    return g


def _make_stub_returns():
    """Return (decompose_rv, analyze_gaps_rv, derive_rv) with empty-but-valid shapes."""
    from lapis_pm.backcaster.schema import Precondition, Gap, Component
    preconditions = [
        Precondition(id="p-test", axis="psychological", statement="Test precondition"),
    ]
    gaps = [
        Gap(
            precondition_id="p-test",
            what_exists="nothing",
            what_missing="everything",
            what_miswired="nothing identified",
            unsourced=True,
        ),
    ]
    components = [
        Component(
            id="c-test",
            gap_precondition_id="p-test",
            description="A test component",
            category="software",
            effort_estimate="small",
            reversibility="high",
        ),
    ]
    return (preconditions, "hash-decomp", False), (gaps, [], "hash-gap"), (components, "hash-derive", 0)


# Patch targets
_CHECK_GROUNDING = "lapis_pm.backcaster.runner._check_grounding"
_DECOMPOSE = "lapis_pm.backcaster.runner.decompose"
_ANALYZE_GAPS = "lapis_pm.backcaster.runner.analyze_gaps"
_DERIVE = "lapis_pm.backcaster.runner.derive_components"


# ---------------------------------------------------------------------------
# Test 1: Synapse down -> abort without flag
# ---------------------------------------------------------------------------

def test_synapse_down_aborts_without_flag(tmp_path):
    from lapis_pm.backcaster.runner import run_backcaster

    with patch(_CHECK_GROUNDING, return_value=["synapse"]) as mock_check, \
         patch(_DECOMPOSE) as mock_decompose:
        with pytest.raises(RuntimeError) as exc_info:
            run_backcaster(
                goal_file=_goal_file(tmp_path),
                stub=False,
                allow_degraded=False,
            )

    assert "synapse" in str(exc_info.value).lower()
    assert "--allow-degraded" in str(exc_info.value)
    # Must abort before stage-1 decompose
    mock_decompose.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: mem down -> abort without flag
# ---------------------------------------------------------------------------

def test_mem_down_aborts_without_flag(tmp_path):
    from lapis_pm.backcaster.runner import run_backcaster

    with patch(_CHECK_GROUNDING, return_value=["mem"]), \
         patch(_DECOMPOSE) as mock_decompose:
        with pytest.raises(RuntimeError) as exc_info:
            run_backcaster(
                goal_file=_goal_file(tmp_path),
                stub=False,
                allow_degraded=False,
            )

    assert "mem" in str(exc_info.value).lower()
    assert "--allow-degraded" in str(exc_info.value)
    mock_decompose.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: both down -> abort naming both deps
# ---------------------------------------------------------------------------

def test_both_down_aborts_naming_both(tmp_path):
    from lapis_pm.backcaster.runner import run_backcaster

    with patch(_CHECK_GROUNDING, return_value=["synapse", "mem"]), \
         patch(_DECOMPOSE) as mock_decompose:
        with pytest.raises(RuntimeError) as exc_info:
            run_backcaster(
                goal_file=_goal_file(tmp_path),
                stub=False,
                allow_degraded=False,
            )

    msg = str(exc_info.value)
    assert "synapse" in msg.lower()
    assert "mem" in msg.lower()
    mock_decompose.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: --allow-degraded -> run proceeds, degraded stamped in artifacts
# ---------------------------------------------------------------------------

def test_allow_degraded_proceeds_with_banner(tmp_path):
    from lapis_pm.backcaster.runner import run_backcaster

    decomp_rv, gap_rv, derive_rv = _make_stub_returns()

    with patch(_CHECK_GROUNDING, return_value=["synapse"]), \
         patch(_DECOMPOSE, return_value=decomp_rv), \
         patch(_ANALYZE_GAPS, return_value=gap_rv), \
         patch(_DERIVE, return_value=derive_rv):
        run_dir = run_backcaster(
            goal_file=_goal_file(tmp_path),
            out_dir=tmp_path / "run",
            stub=False,
            allow_degraded=True,
        )

    import yaml

    # run.yaml must list degraded_paths including 'synapse'
    run_meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert "synapse" in run_meta.get("degraded_paths", [])

    # roadmap.md must carry DEGRADED banner at the top (before the first heading)
    roadmap = (run_dir / "roadmap.md").read_text()
    first_heading_pos = roadmap.find("# Backcaster Roadmap")
    banner_region = roadmap[:first_heading_pos]
    assert "DEGRADED" in banner_region
    assert "synapse" in banner_region.lower()


# ---------------------------------------------------------------------------
# Test 5: both grounding deps healthy -> clean run (no banner, no degraded_paths)
# ---------------------------------------------------------------------------

def test_healthy_grounding_clean_run(tmp_path):
    from lapis_pm.backcaster.runner import run_backcaster

    decomp_rv, gap_rv, derive_rv = _make_stub_returns()

    with patch(_CHECK_GROUNDING, return_value=[]), \
         patch(_DECOMPOSE, return_value=decomp_rv), \
         patch(_ANALYZE_GAPS, return_value=gap_rv), \
         patch(_DERIVE, return_value=derive_rv):
        run_dir = run_backcaster(
            goal_file=_goal_file(tmp_path),
            out_dir=tmp_path / "run",
            stub=False,
            allow_degraded=False,
        )

    import yaml
    run_meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert run_meta.get("degraded_paths", []) == []

    roadmap = (run_dir / "roadmap.md").read_text()
    assert "DEGRADED" not in roadmap


# ---------------------------------------------------------------------------
# Test 6: stub=True bypasses pre-flight entirely (no _check_grounding call)
# ---------------------------------------------------------------------------

def test_stub_mode_skips_preflight(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKCASTER_STUB", "1")
    from lapis_pm.backcaster.runner import run_backcaster

    with patch(_CHECK_GROUNDING) as mock_check:
        run_backcaster(
            goal_file=_goal_file(tmp_path),
            out_dir=tmp_path / "run",
            stub=True,
        )

    mock_check.assert_not_called()
