"""Unit tests for the blind-run driver (U3).

All tests are off-lane: call_operator and overlay are mocked.
No GW lane consumption or paid fallbacks.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.jagged_seam.blind_run import (
    run_blind,
    _check_neutrality,
    JaggedSeamGravityWellUnavailable,
)
from lapis_pm.jagged_seam.schema import (
    ConstraintSurface,
    ConstraintSurfaceItem,
    PoleClaim,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def state_doc_clean(temp_dir):
    """Create a clean state doc (no disqualifying terms)."""
    doc = temp_dir / "state.md"
    doc.write_text("""
# Q2 Defederation Design

## Current Situation

Lapis operates an enforcement of last resort. The independent-defederation council has gaps
in authority distribution. There are tensions between witness-independence and
density/irreducibility.

## Design Constraints

- Defederation must work across federated systems.
- Large platforms have exit options.
- Witness agreements are hard to coordinate.

## Open Questions

What mechanisms ensure credible enforcement? What are the failure vectors?
""")
    return doc


@pytest.fixture
def state_doc_biased(temp_dir):
    """Create a state doc with disqualifying terms."""
    doc = temp_dir / "state_biased.md"
    doc.write_text("""
# State with Answer Pre-loaded

The key decision_variable here is about consent and defederation.
The seam involves how to balance both sides.
""")
    return doc


@pytest.fixture
def temp_out_dir(temp_dir):
    """Create a temporary output directory."""
    out_dir = temp_dir / "runs"
    out_dir.mkdir()
    return out_dir


class TestNeutralityCheck:
    """Test the neutrality precondition guard."""

    def test_neutrality_pass(self, state_doc_clean):
        """Clean state doc should pass neutrality check."""
        is_neutral, flagged = _check_neutrality(state_doc_clean)
        assert is_neutral is True
        assert flagged == []

    def test_neutrality_fail_decision_variable(self, state_doc_biased):
        """State doc with 'decision_variable' should fail."""
        is_neutral, flagged = _check_neutrality(state_doc_biased)
        assert is_neutral is False
        assert "decision_variable" in flagged

    def test_neutrality_fail_seam(self, temp_dir):
        """State doc with 'seam' should fail."""
        doc = temp_dir / "state_seam.md"
        doc.write_text("The seam here is about consent.")
        is_neutral, flagged = _check_neutrality(doc)
        assert is_neutral is False
        assert "seam" in flagged

    def test_neutrality_case_insensitive(self, temp_dir):
        """Neutrality check should be case-insensitive."""
        doc = temp_dir / "state_case.md"
        doc.write_text("The SEAM is important. The Decision-Variable is X.")
        is_neutral, flagged = _check_neutrality(doc)
        assert is_neutral is False
        assert len(flagged) > 0


class TestInputVerification:
    """Test input preconditions (file existence, non-empty, etc.)."""

    def test_missing_state_doc(self, temp_out_dir):
        """Missing state doc should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            run_blind(
                state_doc="/nonexistent/path/state.md",
                out_dir=str(temp_out_dir),
            )

    def test_empty_state_doc(self, temp_dir, temp_out_dir):
        """Empty state doc should raise FileNotFoundError."""
        doc = temp_dir / "empty.md"
        doc.write_text("")
        with pytest.raises(FileNotFoundError):
            run_blind(
                state_doc=str(doc),
                out_dir=str(temp_out_dir),
            )


class TestNeutralityPrecondition:
    """Test that neutrality check aborts before any GW call."""

    def test_biased_state_aborts_before_gw(self, state_doc_biased, temp_out_dir):
        """Biased state doc should raise ValueError before any GW call."""
        with pytest.raises(ValueError, match="contains disqualifying terminology"):
            run_blind(
                state_doc=str(state_doc_biased),
                out_dir=str(temp_out_dir),
            )


class TestWithMocking:
    """Test run_blind with mocked call_operator and overlay."""

    def test_full_run_happy_path(self, state_doc_clean, temp_out_dir):
        """Happy path: clean state, GW responds, overlay succeeds."""
        with patch("agents_core.llm.call_operator") as mock_call_op:
            with patch("lapis_pm.jagged_seam.blind_run.overlay") as mock_overlay:
                with patch("lapis_pm.jagged_seam.blind_run.from_prose") as mock_from_prose:
                    # Mock the three GW calls (fear, desire, control).
                    # call_operator returns JSON strings that need to be parseable.
                    fear_resp = json.dumps(["Fear 1", "Fear 2"])
                    desire_resp = json.dumps(["Desire 1", "Desire 2"])
                    control_resp = json.dumps(["Control 1", "Control 2"])
                    verifier_resp = json.dumps({"novel": True, "cited_source": None})

                    mock_call_op.side_effect = [
                        fear_resp,
                        desire_resp,
                        control_resp,
                        verifier_resp,  # Verifier call
                    ]

                    # Mock poles.
                    mock_fear_pole = MagicMock()
                    mock_fear_pole.claims = [
                        PoleClaim("Fear 1", "fear", "gw:fear-pass[0]", "verified"),
                        PoleClaim("Fear 2", "fear", "gw:fear-pass[1]", "verified"),
                    ]

                    mock_desire_pole = MagicMock()
                    mock_desire_pole.claims = [
                        PoleClaim("Desire 1", "desire", "gw:desire-pass[0]", "verified"),
                        PoleClaim("Desire 2", "desire", "gw:desire-pass[1]", "verified"),
                    ]

                    mock_from_prose.side_effect = [
                        mock_fear_pole,
                        mock_desire_pole,
                    ]

                    # Mock surface with one novel item.
                    novel_item = ConstraintSurfaceItem(
                        decision_variable="Novel decision var",
                        fear_source=mock_fear_pole.claims[0],
                        desire_source=mock_desire_pole.claims[0],
                        crossing_type="tension",
                    )

                    mock_overlay.return_value = ConstraintSurface(
                        state_ref="zephyr-defederation-teeth-2026-06-17",
                        items=[novel_item],
                        skipped_no_axis=False,
                        dropped_provenance_count=0,
                    )

                    result = run_blind(
                        state_doc=str(state_doc_clean),
                        out_dir=str(temp_out_dir),
                    )

                    # Verify result.
                    assert result["metric_result"] == "supported"
                    assert result["new_variable_count"] == 1
                    assert result["skipped_no_axis"] is False
                    assert len(result["novelty_analysis"]) == 1
                    assert result["novelty_analysis"][0]["novel"] is True

    def test_skipped_no_axis(self, state_doc_clean, temp_out_dir):
        """Overlay with skipped_no_axis=True should set metric_result='not_supported'."""
        with patch("agents_core.llm.call_operator") as mock_call_op:
            with patch("lapis_pm.jagged_seam.blind_run.overlay") as mock_overlay:
                with patch("lapis_pm.jagged_seam.blind_run.from_prose") as mock_from_prose:
                    # Mock three GW calls.
                    mock_call_op.side_effect = [
                        json.dumps(["Fear 1"]),
                        json.dumps(["Desire 1"]),
                        json.dumps(["Control 1"]),
                    ]

                    # Mock poles with empty claims.
                    mock_fear_pole = MagicMock()
                    mock_fear_pole.claims = []

                    mock_desire_pole = MagicMock()
                    mock_desire_pole.claims = []

                    mock_from_prose.side_effect = [
                        mock_fear_pole,
                        mock_desire_pole,
                    ]

                    # Mock overlay with skipped_no_axis.
                    mock_overlay.return_value = ConstraintSurface(
                        state_ref="zephyr-defederation-teeth-2026-06-17",
                        items=[],
                        skipped_no_axis=True,
                        dropped_provenance_count=0,
                    )

                    result = run_blind(
                        state_doc=str(state_doc_clean),
                        out_dir=str(temp_out_dir),
                    )

                    # Verify result.
                    assert result["metric_result"] == "not_supported"
                    assert result["new_variable_count"] == 0
                    assert result["skipped_no_axis"] is True

    def test_restatement_not_novel(self, state_doc_clean, temp_out_dir):
        """Item marked as restatement should not increment new_variable_count."""
        with patch("agents_core.llm.call_operator") as mock_call_op:
            with patch("lapis_pm.jagged_seam.blind_run.overlay") as mock_overlay:
                with patch("lapis_pm.jagged_seam.blind_run.from_prose") as mock_from_prose:
                    # Mock three GW calls + verifier.
                    mock_call_op.side_effect = [
                        json.dumps(["Fear 1"]),
                        json.dumps(["Desire 1"]),
                        json.dumps(["Control 1"]),
                        json.dumps({"novel": False, "cited_source": "Fear 1"}),  # Restatement.
                    ]

                    # Mock poles.
                    mock_fear_pole = MagicMock()
                    mock_fear_pole.claims = [
                        PoleClaim("Fear 1", "fear", "gw:fear-pass[0]", "verified"),
                    ]

                    mock_desire_pole = MagicMock()
                    mock_desire_pole.claims = [
                        PoleClaim("Desire 1", "desire", "gw:desire-pass[0]", "verified"),
                    ]

                    mock_from_prose.side_effect = [
                        mock_fear_pole,
                        mock_desire_pole,
                    ]

                    # Mock surface with restatement item.
                    item = ConstraintSurfaceItem(
                        decision_variable="Some var echoing Fear 1",
                        fear_source=mock_fear_pole.claims[0],
                        desire_source=mock_desire_pole.claims[0],
                        crossing_type="tension",
                    )

                    mock_overlay.return_value = ConstraintSurface(
                        state_ref="zephyr-defederation-teeth-2026-06-17",
                        items=[item],
                        skipped_no_axis=False,
                        dropped_provenance_count=0,
                    )

                    result = run_blind(
                        state_doc=str(state_doc_clean),
                        out_dir=str(temp_out_dir),
                    )

                    # Verify result.
                    assert result["metric_result"] == "not_supported"
                    assert result["new_variable_count"] == 0
                    assert result["novelty_analysis"][0]["novel"] is False
                    assert result["novelty_analysis"][0]["cited_source"] == "Fear 1"

    def test_verifier_call_failure_conservative_default(self, state_doc_clean, temp_out_dir):
        """Verifier returning None (GW down) should mark item as NOT novel."""
        with patch("agents_core.llm.call_operator") as mock_call_op:
            with patch("lapis_pm.jagged_seam.blind_run.overlay") as mock_overlay:
                with patch("lapis_pm.jagged_seam.blind_run.from_prose") as mock_from_prose:
                    # Verifier returns None (GW unavailable).
                    mock_call_op.side_effect = [
                        json.dumps(["Fear 1"]),
                        json.dumps(["Desire 1"]),
                        json.dumps(["Control 1"]),
                        None,  # Verifier call fails.
                    ]

                    # Mock poles.
                    mock_fear_pole = MagicMock()
                    mock_fear_pole.claims = [
                        PoleClaim("Fear 1", "fear", "gw:fear-pass[0]", "verified"),
                    ]

                    mock_desire_pole = MagicMock()
                    mock_desire_pole.claims = [
                        PoleClaim("Desire 1", "desire", "gw:desire-pass[0]", "verified"),
                    ]

                    mock_from_prose.side_effect = [
                        mock_fear_pole,
                        mock_desire_pole,
                    ]

                    # Mock surface with one item.
                    item = ConstraintSurfaceItem(
                        decision_variable="Some var",
                        fear_source=mock_fear_pole.claims[0],
                        desire_source=mock_desire_pole.claims[0],
                        crossing_type="tension",
                    )

                    mock_overlay.return_value = ConstraintSurface(
                        state_ref="zephyr-defederation-teeth-2026-06-17",
                        items=[item],
                        skipped_no_axis=False,
                        dropped_provenance_count=0,
                    )

                    result = run_blind(
                        state_doc=str(state_doc_clean),
                        out_dir=str(temp_out_dir),
                    )

                    # Conservative default: NOT novel.
                    assert result["new_variable_count"] == 0
                    assert result["novelty_analysis"][0]["novel"] is False

    def test_gw_unavailable_on_fear_pass(self, state_doc_clean, temp_out_dir):
        """GW unavailable on fear pass should raise and write skip.yaml."""
        with patch("agents_core.llm.call_operator") as mock_call_op:
            # Fear pass returns None (GW down).
            mock_call_op.return_value = None

            with pytest.raises(JaggedSeamGravityWellUnavailable):
                run_blind(
                    state_doc=str(state_doc_clean),
                    out_dir=str(temp_out_dir),
                )

            # Check skip.yaml was written.
            skip_file = Path(temp_out_dir) / "skip.yaml"
            assert skip_file.exists()

    def test_artifacts_written(self, state_doc_clean, temp_out_dir):
        """All artifacts should be written to the run directory."""
        with patch("agents_core.llm.call_operator") as mock_call_op:
            with patch("lapis_pm.jagged_seam.blind_run.overlay") as mock_overlay:
                with patch("lapis_pm.jagged_seam.blind_run.from_prose") as mock_from_prose:
                    # Mock three GW calls.
                    mock_call_op.side_effect = [
                        json.dumps(["Fear 1"]),
                        json.dumps(["Desire 1"]),
                        json.dumps(["Control 1"]),
                    ]

                    # Mock poles with empty claims.
                    mock_fear_pole = MagicMock()
                    mock_fear_pole.claims = []

                    mock_desire_pole = MagicMock()
                    mock_desire_pole.claims = []

                    mock_from_prose.side_effect = [
                        mock_fear_pole,
                        mock_desire_pole,
                    ]

                    # Mock overlay with no items (no verification needed).
                    mock_overlay.return_value = ConstraintSurface(
                        state_ref="zephyr-defederation-teeth-2026-06-17",
                        items=[],
                        skipped_no_axis=False,
                        dropped_provenance_count=0,
                    )

                    result = run_blind(
                        state_doc=str(state_doc_clean),
                        out_dir=str(temp_out_dir),
                    )

                    run_id = result["run_id"]
                    run_dir = Path(temp_out_dir) / run_id

                    # Verify all artifacts exist.
                    assert (run_dir / "fear_pass.yaml").exists()
                    assert (run_dir / "desire_pass.yaml").exists()
                    assert (run_dir / "control_pass.yaml").exists()
                    assert (run_dir / "constraint_surface.yaml").exists()
                    assert (run_dir / "novelty_analysis.yaml").exists()
                    assert (run_dir / "run.yaml").exists()

                    # Load and check run.yaml.
                    import yaml
                    with open(run_dir / "run.yaml") as f:
                        run_data = yaml.safe_load(f)

                    assert run_data["state_ref"] == "zephyr-defederation-teeth-2026-06-17"
                    assert "state_doc" in run_data
                    assert "state_doc_sha256" in run_data
                    assert "operator" in run_data
                    assert "neutrality_check" in run_data
                    assert run_data["neutrality_check"]["passed"] is True

    def test_dropped_provenance_count_surfaced(self, state_doc_clean, temp_out_dir):
        """Non-zero dropped_provenance_count should appear in result and artifacts."""
        with patch("agents_core.llm.call_operator") as mock_call_op:
            with patch("lapis_pm.jagged_seam.blind_run.overlay") as mock_overlay:
                with patch("lapis_pm.jagged_seam.blind_run.from_prose") as mock_from_prose:
                    # Mock three GW calls.
                    mock_call_op.side_effect = [
                        json.dumps(["Fear 1"]),
                        json.dumps(["Desire 1"]),
                        json.dumps(["Control 1"]),
                    ]

                    # Mock poles.
                    mock_fear_pole = MagicMock()
                    mock_fear_pole.claims = []

                    mock_desire_pole = MagicMock()
                    mock_desire_pole.claims = []

                    mock_from_prose.side_effect = [
                        mock_fear_pole,
                        mock_desire_pole,
                    ]

                    # Mock overlay with dropped_provenance_count > 0.
                    mock_overlay.return_value = ConstraintSurface(
                        state_ref="zephyr-defederation-teeth-2026-06-17",
                        items=[],
                        skipped_no_axis=False,
                        dropped_provenance_count=2,
                    )

                    result = run_blind(
                        state_doc=str(state_doc_clean),
                        out_dir=str(temp_out_dir),
                    )

                    # Verify dropped_provenance_count is in the run.yaml artifact.
                    run_id = result["run_id"]
                    run_dir = Path(temp_out_dir) / run_id
                    import yaml
                    with open(run_dir / "run.yaml") as f:
                        run_data = yaml.safe_load(f)
                    assert run_data["dropped_provenance_count"] == 2

    def test_operative_shift_capture(self, state_doc_clean, temp_out_dir):
        """Restatement should include the cited source claim text."""
        with patch("agents_core.llm.call_operator") as mock_call_op:
            with patch("lapis_pm.jagged_seam.blind_run.overlay") as mock_overlay:
                with patch("lapis_pm.jagged_seam.blind_run.from_prose") as mock_from_prose:
                    # Mock three GW calls + verifier.
                    mock_call_op.side_effect = [
                        json.dumps(["The enforcement is fragile"]),
                        json.dumps(["Enforcement must be robust"]),
                        json.dumps(["Control 1"]),
                        json.dumps({
                            "novel": False,
                            "cited_source": "gw:fear-pass[0]",  # Verifier cites the fear-source provenance.
                        }),
                    ]

                    # Mock poles with the cited claim.
                    mock_fear_pole = MagicMock()
                    mock_fear_pole.claims = [
                        PoleClaim(
                            "The enforcement is fragile",
                            "fear",
                            "gw:fear-pass[0]",
                            "verified",
                        ),
                    ]

                    mock_desire_pole = MagicMock()
                    mock_desire_pole.claims = [
                        PoleClaim(
                            "Enforcement must be robust",
                            "desire",
                            "gw:desire-pass[0]",
                            "verified",
                        ),
                    ]

                    mock_from_prose.side_effect = [
                        mock_fear_pole,
                        mock_desire_pole,
                    ]

                    # Mock surface with restatement item.
                    item = ConstraintSurfaceItem(
                        decision_variable="Some var about enforcement",
                        fear_source=mock_fear_pole.claims[0],
                        desire_source=mock_desire_pole.claims[0],
                        crossing_type="tension",
                    )

                    mock_overlay.return_value = ConstraintSurface(
                        state_ref="zephyr-defederation-teeth-2026-06-17",
                        items=[item],
                        skipped_no_axis=False,
                        dropped_provenance_count=0,
                    )

                    result = run_blind(
                        state_doc=str(state_doc_clean),
                        out_dir=str(temp_out_dir),
                    )

                    # Verify cited_source_text is captured.
                    assert (
                        result["novelty_analysis"][0]["cited_source_text"]
                        == "The enforcement is fragile"
                    )
