"""Unit tests for lapis_pm.jagged_seam.warmup run-driver.

All tests are off-lane (overlay mocked), covering:
1. Pole construction from archived YAML
2. Artifact writing
3. Reproduction signal (positive and negative)
4. Gap-gate (gap_testimony.yaml on non-reproduction)
5. GW-unavailable handling
6. Library purity (no sys.exit)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lapis_pm.jagged_seam import (
    ConstraintSurface,
    ConstraintSurfaceItem,
    JaggedSeamGravityWellUnavailable,
    PoleClaim,
    PoleOutput,
)
from lapis_pm.jagged_seam.warmup import (
    _compute_reproduction_signal,
    _gen_run_id,
    _pole_to_dict,
    _surface_to_dict,
    run_warmup,
)


@pytest.fixture
def temp_run_dir(tmp_path):
    """Create a temporary archived run directory with fixture YAML files."""
    run_dir = tmp_path / "archived_run"
    run_dir.mkdir()

    # Minimal valid decomposition.yaml
    decomposition = {
        "axes": ["economic", "governance"],
        "preconditions": [
            {
                "id": "governance-02",
                "axis": "governance",
                "statement": "Defederation must be mechanically enforced with publicly testable criteria.",
                "implicit_dependencies": ["Criteria finalized before onboarding"],
            },
            {
                "id": "economic-02",
                "axis": "economic",
                "statement": "Hoarding must yield lower return than active contribution.",
                "implicit_dependencies": ["No supply-capping mechanism"],
            },
        ],
    }
    with open(run_dir / "decomposition.yaml", "w") as f:
        yaml.dump(decomposition, f)

    # Minimal valid gaps.yaml
    gaps = {
        "gaps": [
            {
                "precondition_id": "governance-02",
                "what_exists": "Defederation concept is described.",
                "what_missing": "No mechanically enforced criteria published.",
                "what_miswired": "Discretion not constrained.",
                "citations": [],
                "unsourced": True,
            },
            {
                "precondition_id": "economic-02",
                "what_exists": "Use-tax model described.",
                "what_missing": "No formal holding-cost primitive.",
                "what_miswired": "Consent-term hoarding lever creates unpunished passivity.",
                "citations": [],
                "unsourced": True,
            },
        ],
    }
    with open(run_dir / "gaps.yaml", "w") as f:
        yaml.dump(gaps, f)

    # Minimal valid components.yaml
    components_data = {"components": []}
    with open(run_dir / "components.yaml", "w") as f:
        yaml.dump(components_data, f)

    return run_dir


@pytest.fixture
def temp_out_dir(tmp_path):
    """Create a temporary output directory."""
    out_dir = tmp_path / "run_outputs"
    return out_dir


# ---------------------------------------------------------------------------
# Test 1: Pole construction
# ---------------------------------------------------------------------------


def test_pole_construction_from_archived_yaml(temp_run_dir, temp_out_dir):
    """Test that run_warmup builds desire pole from archived Backcaster YAML."""
    with patch("lapis_pm.jagged_seam.warmup.overlay") as mock_overlay:
        # Mock a successful overlay with no items (thin result).
        mock_overlay.return_value = ConstraintSurface(
            state_ref="zephyr-sustainability-2026-06-08",
            items=[],
            skipped_no_axis=False,
        )

        # run_warmup should load the archived YAML and build the poles.
        with pytest.raises(ValueError, match="Non-reproduction"):
            run_warmup(str(temp_run_dir), str(temp_out_dir))

        # Verify overlay was called (poles were built and passed to overlay).
        mock_overlay.assert_called_once()
        call_args = mock_overlay.call_args
        fear_pole, desire_pole = call_args[0]

        # Verify desire pole has claims from the fixture gaps/preconditions.
        assert desire_pole.kind == "desire"
        assert desire_pole.state_ref == "zephyr-sustainability-2026-06-08"
        assert all(c.fidelity == "verified" for c in desire_pole.claims)
        assert len(desire_pole.claims) >= 3  # At least 2 gaps * 2 fields + 2 preconditions

        # Verify fear pole has the 4 hand-transcribed drift-vectors.
        assert fear_pole.kind == "fear"
        assert fear_pole.state_ref == "zephyr-sustainability-2026-06-08"
        assert len(fear_pole.claims) == 4
        assert all(c.fidelity == "transcribed" for c in fear_pole.claims)

        # Verify fear provenance uses the mem thread key.
        assert all(
            "mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vector"
            in c.provenance
            for c in fear_pole.claims
        )


# ---------------------------------------------------------------------------
# Test 2: Artifact write
# ---------------------------------------------------------------------------


def test_artifact_write(temp_run_dir, temp_out_dir):
    """Test that run_warmup writes fear_pole.yaml, desire_pole.yaml, constraint_surface.yaml, run.yaml."""
    with patch("lapis_pm.jagged_seam.warmup.overlay") as mock_overlay:
        mock_overlay.return_value = ConstraintSurface(
            state_ref="zephyr-sustainability-2026-06-08",
            items=[],
            skipped_no_axis=False,
        )

        with pytest.raises(ValueError, match="Non-reproduction"):
            run_warmup(str(temp_run_dir), str(temp_out_dir))

        # Find the run_id directory created under temp_out_dir.
        run_dirs = list(temp_out_dir.glob("*-warmup"))
        assert len(run_dirs) == 1
        run_path = run_dirs[0]

        # Check for all required artifacts.
        assert (run_path / "fear_pole.yaml").exists()
        assert (run_path / "desire_pole.yaml").exists()
        assert (run_path / "constraint_surface.yaml").exists()
        assert (run_path / "run.yaml").exists()

        # Verify artifacts are valid YAML and round-trippable.
        with open(run_path / "fear_pole.yaml") as f:
            fear_data = yaml.safe_load(f)
            assert fear_data["kind"] == "fear"
            assert len(fear_data["claims"]) == 4

        with open(run_path / "desire_pole.yaml") as f:
            desire_data = yaml.safe_load(f)
            assert desire_data["kind"] == "desire"
            assert len(desire_data["claims"]) >= 3

        with open(run_path / "constraint_surface.yaml") as f:
            surface_data = yaml.safe_load(f)
            assert surface_data["state_ref"] == "zephyr-sustainability-2026-06-08"
            assert "items" in surface_data

        with open(run_path / "run.yaml") as f:
            run_data = yaml.safe_load(f)
            assert "run_id" in run_data
            assert "reproduction_signal" in run_data
            assert "human_curation_confound" in run_data


# ---------------------------------------------------------------------------
# Test 3: Reproduction signal — positive (seam present)
# ---------------------------------------------------------------------------


def test_reproduction_signal_positive_seam_present(temp_run_dir, temp_out_dir):
    """Test that reproduction_signal.seam_item_present=True when consent seam is found."""
    with patch("lapis_pm.jagged_seam.warmup.overlay") as mock_overlay:
        # Mock overlay to return a surface with the consent-hoarding seam.
        mock_overlay.return_value = ConstraintSurface(
            state_ref="zephyr-sustainability-2026-06-08",
            items=[
                ConstraintSurfaceItem(
                    decision_variable="how restrictive consent terms may be given hoarding leverage",
                    fear_source=PoleClaim(
                        claim="Consent terms function as a hoarding lever...",
                        kind="fear",
                        provenance="mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vector[0]",
                        fidelity="transcribed",
                    ),
                    desire_source=PoleClaim(
                        claim="Defederation must be mechanically enforced with governance separation.",
                        kind="desire",
                        provenance="backcaster:precondition_id=governance-02",
                        fidelity="verified",
                    ),
                    crossing_type="tension",
                ),
            ],
            skipped_no_axis=False,
        )

        # run_warmup should succeed.
        result = run_warmup(str(temp_run_dir), str(temp_out_dir))

        # Check reproduction_signal in result.
        assert result["reproduction_signal"]["seam_item_present"] is True
        assert not result["reproduction_signal"]["skipped_no_axis"]
        assert len(result["reproduction_signal"]["candidate_items"]) > 0

        # Check reproduction_signal in run.yaml.
        run_dirs = list(temp_out_dir.glob("*-warmup"))
        run_path = run_dirs[0]
        with open(run_path / "run.yaml") as f:
            run_data = yaml.safe_load(f)
            assert run_data["reproduction_signal"]["seam_item_present"] is True


# ---------------------------------------------------------------------------
# Test 4: Gap-gate — non-reproduction (seam absent)
# ---------------------------------------------------------------------------


def test_gap_gate_seam_absent(temp_run_dir, temp_out_dir):
    """Test that non-reproduction writes gap_testimony.yaml and raises ValueError."""
    with patch("lapis_pm.jagged_seam.warmup.overlay") as mock_overlay:
        # Mock overlay to return empty items (no seam).
        mock_overlay.return_value = ConstraintSurface(
            state_ref="zephyr-sustainability-2026-06-08",
            items=[
                ConstraintSurfaceItem(
                    decision_variable="some other decision variable",
                    fear_source=PoleClaim(
                        claim="Fee concern",
                        kind="fear",
                        provenance="mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vector[1]",
                        fidelity="transcribed",
                    ),
                    desire_source=PoleClaim(
                        claim="Some economic goal",
                        kind="desire",
                        provenance="backcaster:gap[precondition_id=economic-02]/what_missing",
                        fidelity="verified",
                    ),
                    crossing_type="shared_dependency",
                ),
            ],
            skipped_no_axis=False,
        )

        # run_warmup should raise ValueError for non-reproduction.
        with pytest.raises(ValueError, match="Non-reproduction"):
            run_warmup(str(temp_run_dir), str(temp_out_dir))

        # Verify gap_testimony.yaml was written.
        run_dirs = list(temp_out_dir.glob("*-warmup"))
        run_path = run_dirs[0]
        assert (run_path / "gap_testimony.yaml").exists()

        # Verify gap_testimony contains facts only (no diagnosis).
        with open(run_path / "gap_testimony.yaml") as f:
            testimony = yaml.safe_load(f)
            assert testimony["missing_axis"] == "consent-restrictiveness seam absent"
            assert "near_misses" in testimony
            assert "inputs_sha256" in testimony
            assert len(testimony["inputs_sha256"]) == 64  # SHA256 hex length
            # Check that diagnosis/why fields are absent.
            assert "why" not in testimony
            assert "diagnosis" not in testimony


def test_gap_gate_thin_surface(temp_run_dir, temp_out_dir):
    """Test gap-gate when overlay returns skipped_no_axis=True."""
    with patch("lapis_pm.jagged_seam.warmup.overlay") as mock_overlay:
        mock_overlay.return_value = ConstraintSurface(
            state_ref="zephyr-sustainability-2026-06-08",
            items=[],
            skipped_no_axis=True,
        )

        with pytest.raises(ValueError, match="Non-reproduction"):
            run_warmup(str(temp_run_dir), str(temp_out_dir))

        # Verify gap_testimony was written.
        run_dirs = list(temp_out_dir.glob("*-warmup"))
        run_path = run_dirs[0]
        assert (run_path / "gap_testimony.yaml").exists()


# ---------------------------------------------------------------------------
# Test 5: Input-hash stability
# ---------------------------------------------------------------------------


def test_inputs_sha256_stability(temp_run_dir, temp_out_dir):
    """Test that inputs_sha256 is deterministic and changes with pole input changes."""
    with patch("lapis_pm.jagged_seam.warmup.overlay") as mock_overlay:
        mock_overlay.return_value = ConstraintSurface(
            state_ref="zephyr-sustainability-2026-06-08",
            items=[],
            skipped_no_axis=False,
        )

        # First run.
        with pytest.raises(ValueError):
            run_warmup(str(temp_run_dir), str(temp_out_dir))

        run_dirs = sorted(temp_out_dir.glob("*-warmup"))
        first_testimony_path = run_dirs[0] / "gap_testimony.yaml"
        with open(first_testimony_path) as f:
            first_hash = yaml.safe_load(f)["inputs_sha256"]

        # Second run (same inputs).
        with pytest.raises(ValueError):
            run_warmup(str(temp_run_dir), str(temp_out_dir))

        run_dirs = sorted(temp_out_dir.glob("*-warmup"))
        second_testimony_path = run_dirs[-1] / "gap_testimony.yaml"
        with open(second_testimony_path) as f:
            second_hash = yaml.safe_load(f)["inputs_sha256"]

        # Hashes should match (same inputs).
        assert first_hash == second_hash
        assert len(first_hash) == 64


# ---------------------------------------------------------------------------
# Test 6: GW-unavailable
# ---------------------------------------------------------------------------


def test_gw_unavailable(temp_run_dir, temp_out_dir):
    """Test that JaggedSeamGravityWellUnavailable is raised and skip.yaml is written."""
    with patch("lapis_pm.jagged_seam.warmup.overlay") as mock_overlay:
        mock_overlay.side_effect = JaggedSeamGravityWellUnavailable("GW not serving")

        with pytest.raises(JaggedSeamGravityWellUnavailable):
            run_warmup(str(temp_run_dir), str(temp_out_dir))

        # Verify skip.yaml was written.
        run_dirs = list(temp_out_dir.glob("*-warmup"))
        run_path = run_dirs[0]
        assert (run_path / "skip.yaml").exists()

        with open(run_path / "skip.yaml") as f:
            skip_data = yaml.safe_load(f)
            assert skip_data["skipped"] is True
            assert skip_data["reason"] == "gw_unavailable"
            assert "ts" in skip_data


# ---------------------------------------------------------------------------
# Test 7: Library purity (no sys.exit in run_warmup)
# ---------------------------------------------------------------------------


def test_library_purity_no_sys_exit(temp_run_dir, temp_out_dir):
    """Test that run_warmup never calls sys.exit on any path."""
    test_cases = [
        # Case 1: Successful run
        (
            "success",
            lambda m: setattr(
                m,
                "return_value",
                ConstraintSurface(
                    state_ref="zephyr-sustainability-2026-06-08",
                    items=[
                        ConstraintSurfaceItem(
                            decision_variable="how restrictive consent",
                            fear_source=PoleClaim(
                                claim="Hoarding lever",
                                kind="fear",
                                provenance="mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vector[0]",
                                fidelity="transcribed",
                            ),
                            desire_source=PoleClaim(
                                claim="Governance separation required",
                                kind="desire",
                                provenance="backcaster:precondition_id=governance-02",
                                fidelity="verified",
                            ),
                            crossing_type="tension",
                        ),
                    ],
                    skipped_no_axis=False,
                ),
            ),
        ),
        # Case 2: Non-reproduction
        (
            "non-reproduction",
            lambda m: setattr(
                m,
                "return_value",
                ConstraintSurface(
                    state_ref="zephyr-sustainability-2026-06-08",
                    items=[],
                    skipped_no_axis=False,
                ),
            ),
        ),
        # Case 3: GW unavailable
        (
            "gw_unavailable",
            lambda m: setattr(m, "side_effect", JaggedSeamGravityWellUnavailable("test")),
        ),
    ]

    for case_name, setup in test_cases:
        with patch("lapis_pm.jagged_seam.warmup.overlay") as mock_overlay:
            setup(mock_overlay)

            with patch("sys.exit") as mock_exit:
                try:
                    run_warmup(str(temp_run_dir), str(temp_out_dir))
                except (ValueError, JaggedSeamGravityWellUnavailable):
                    pass

                # run_warmup should never call sys.exit (even on error).
                mock_exit.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8: Input verification (missing YAML files)
# ---------------------------------------------------------------------------


def test_input_verification_missing_file(tmp_path):
    """Test that run_warmup raises FileNotFoundError if required YAML files are missing."""
    incomplete_run_dir = tmp_path / "incomplete"
    incomplete_run_dir.mkdir()
    # Missing gaps.yaml and components.yaml
    with open(incomplete_run_dir / "decomposition.yaml", "w") as f:
        yaml.dump({"preconditions": []}, f)

    out_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="decomposition.yaml|gaps.yaml|components.yaml"):
        run_warmup(str(incomplete_run_dir), str(out_dir))


def test_input_verification_empty_file(tmp_path):
    """Test that run_warmup raises FileNotFoundError if required YAML files are empty."""
    incomplete_run_dir = tmp_path / "incomplete_empty"
    incomplete_run_dir.mkdir()

    # Create empty decomposition.yaml
    (incomplete_run_dir / "decomposition.yaml").touch()
    with open(incomplete_run_dir / "gaps.yaml", "w") as f:
        yaml.dump({"gaps": []}, f)
    with open(incomplete_run_dir / "components.yaml", "w") as f:
        yaml.dump({"components": []}, f)

    out_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="empty"):
        run_warmup(str(incomplete_run_dir), str(out_dir))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_gen_run_id():
    """Test that _gen_run_id produces valid run IDs."""
    run_id = _gen_run_id()
    assert "-warmup" in run_id
    assert len(run_id) > 0
    # Should have format like 20260617-151801-warmup
    assert run_id.endswith("-warmup")


def test_pole_to_dict():
    """Test _pole_to_dict serialization."""
    pole = PoleOutput(
        kind="fear",
        state_ref="test-state",
        claims=[
            PoleClaim(
                claim="Test claim",
                kind="fear",
                provenance="test:provenance",
                fidelity="transcribed",
            ),
        ],
        thin=False,
    )

    pole_dict = _pole_to_dict(pole)
    assert pole_dict["kind"] == "fear"
    assert pole_dict["state_ref"] == "test-state"
    assert pole_dict["thin"] is False
    assert len(pole_dict["claims"]) == 1
    assert pole_dict["claims"][0]["claim"] == "Test claim"


def test_surface_to_dict():
    """Test _surface_to_dict serialization."""
    surface = ConstraintSurface(
        state_ref="test-state",
        items=[
            ConstraintSurfaceItem(
                decision_variable="test dv",
                fear_source=PoleClaim(
                    claim="fear",
                    kind="fear",
                    provenance="test:fear",
                    fidelity="verified",
                ),
                desire_source=PoleClaim(
                    claim="desire",
                    kind="desire",
                    provenance="test:desire",
                    fidelity="verified",
                ),
                crossing_type="tension",
            ),
        ],
        skipped_no_axis=False,
    )

    surface_dict = _surface_to_dict(surface)
    assert surface_dict["state_ref"] == "test-state"
    assert surface_dict["skipped_no_axis"] is False
    assert len(surface_dict["items"]) == 1
    assert surface_dict["items"][0]["decision_variable"] == "test dv"


def test_reproduction_signal_heuristic():
    """Test the consent-hoarding seam heuristic."""
    # Case 1: Seam present (fear[0] + governance keyword).
    surface = ConstraintSurface(
        state_ref="test",
        items=[
            ConstraintSurfaceItem(
                decision_variable="test",
                fear_source=PoleClaim(
                    claim="hoarding",
                    kind="fear",
                    provenance="mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vector[0]",
                    fidelity="transcribed",
                ),
                desire_source=PoleClaim(
                    claim="governance must be separated",
                    kind="desire",
                    provenance="test",
                    fidelity="verified",
                ),
                crossing_type="tension",
            ),
        ],
        skipped_no_axis=False,
    )
    fear_pole = PoleOutput(kind="fear", state_ref="test", claims=[])
    desire_pole = PoleOutput(kind="desire", state_ref="test", claims=[])

    signal = _compute_reproduction_signal(surface, fear_pole, desire_pole)
    assert signal["seam_item_present"] is True
    assert len(signal["candidate_items"]) > 0

    # Case 2: Seam absent (no matching fear[0]).
    surface_no_seam = ConstraintSurface(
        state_ref="test",
        items=[
            ConstraintSurfaceItem(
                decision_variable="test",
                fear_source=PoleClaim(
                    claim="other",
                    kind="fear",
                    provenance="mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vector[1]",
                    fidelity="transcribed",
                ),
                desire_source=PoleClaim(
                    claim="governance must be separated",
                    kind="desire",
                    provenance="test",
                    fidelity="verified",
                ),
                crossing_type="tension",
            ),
        ],
        skipped_no_axis=False,
    )

    signal_no_seam = _compute_reproduction_signal(surface_no_seam, fear_pole, desire_pole)
    assert signal_no_seam["seam_item_present"] is False
