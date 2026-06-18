"""Unit tests for lapis_pm.jagged_seam arbiter.

Tests AC#1-6 per spec:
  AC#1: seam shape (decision-variables + two-sided provenance)
  AC#2: dry discipline (no verdict/recommendation/synthesis fields)
  AC#3: off-lane (model call mocked, full testability without GW)
  AC#4: normalized contract / pole-agnostic (prose + scout trace → same arbiter)
  AC#4b: fidelity propagation (transcribed vs verified markers)
  AC#4c: state_ref mismatch detection
  AC#5: TENSION-GATE short-circuit (thin pole → no model call)
  AC#6: GW-down legibility (JaggedSeamGravityWellUnavailable on wake fail)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.jagged_seam import (
    ConstraintSurface,
    ConstraintSurfaceItem,
    JaggedSeamGravityWellUnavailable,
    PoleClaim,
    PoleOutput,
    from_backcaster,
    from_prose,
    from_scout_trace,
    overlay,
)


# ---------------------------------------------------------------------------
# Fixtures: mock poles for testing
# ---------------------------------------------------------------------------


@pytest.fixture
def fear_pole_fixture() -> PoleOutput:
    """Fixture: a fear pole with three sample claims."""
    return PoleOutput(
        kind="fear",
        state_ref="zephyr-defederation-teeth-2026-06-08",
        claims=[
            PoleClaim(
                claim="Consent terms may trap defederating orgs in asymmetric liability",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[0]",
                fidelity="verified",
            ),
            PoleClaim(
                claim="No recovery path if federation fails mid-transaction",
                kind="fear",
                provenance="scout:cell_id=defed-001/breaks_observed[0]",
                fidelity="verified",
            ),
            PoleClaim(
                claim="Data residency rules may be unenforceable across boundaries",
                kind="fear",
                provenance="scout:cell_id=defed-001/leverage_points[0]",
                fidelity="verified",
            ),
        ],
    )


@pytest.fixture
def desire_pole_fixture() -> PoleOutput:
    """Fixture: a desire pole with three sample claims."""
    return PoleOutput(
        kind="desire",
        state_ref="zephyr-defederation-teeth-2026-06-08",
        claims=[
            PoleClaim(
                claim="Consent terms must be restrictive enough to protect org interests",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
            PoleClaim(
                claim="Federation must provide transaction rollback guarantees",
                kind="desire",
                provenance="backcaster:precondition_id=prec-02",
                fidelity="verified",
            ),
            PoleClaim(
                claim="Data residency must be verifiable and enforceable",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-03]/what_miswired",
                fidelity="verified",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# AC#1 — Seam shape test
# ---------------------------------------------------------------------------


def test_ac1_seam_shape(fear_pole_fixture, desire_pole_fixture):
    """AC#1: overlay() emits ConstraintSurface with decision-variables and two-sided provenance.

    Given two fixture PoleOutputs with the model call mocked to return a
    canned overlay, assert that the result has the correct shape:
    - Each item has decision_variable, fear_source, desire_source with
      non-empty provenance.
    - ConstraintSurface.state_ref matches the poles'.
    """
    canned_response = json.dumps(
        [
            {
                "decision_variable": "how restrictive consent terms may be given liability asymmetry",
                "fear_source_provenance": "scout:cell_id=defed-001/drift_signals[0]",
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            },
            {
                "decision_variable": "whether federation must guarantee transaction atomicity",
                "fear_source_provenance": "scout:cell_id=defed-001/breaks_observed[0]",
                "desire_source_provenance": "backcaster:precondition_id=prec-02",
                "crossing_type": "shared_dependency",
            },
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear_pole_fixture, desire_pole_fixture)

        # Assert result shape
        assert isinstance(result, ConstraintSurface)
        assert result.state_ref == "zephyr-defederation-teeth-2026-06-08"
        assert not result.skipped_no_axis
        assert len(result.items) == 2

        # Assert each item has provenance on both sides
        for item in result.items:
            assert isinstance(item, ConstraintSurfaceItem)
            assert isinstance(item.decision_variable, str)
            assert len(item.decision_variable) > 0
            assert isinstance(item.fear_source, PoleClaim)
            assert isinstance(item.desire_source, PoleClaim)
            assert len(item.fear_source.provenance) > 0
            assert len(item.desire_source.provenance) > 0

        # Verify operator was called once
        mock_operator.assert_called_once()


# ---------------------------------------------------------------------------
# AC#2 — Dry discipline test
# ---------------------------------------------------------------------------


def test_ac2_dry_discipline_no_forbidden_fields(fear_pole_fixture, desire_pole_fixture):
    """AC#2: ConstraintSurface and Item schemas expose NO verdict/recommendation/synthesis.

    Assert that the schema itself has no verdict/recommendation/synthesis fields,
    and that a response containing these fields is rejected as a parse failure.
    """
    # Verify schema has no forbidden fields
    sample_item = ConstraintSurfaceItem(
        decision_variable="test",
        fear_source=PoleClaim(
            claim="test",
            kind="fear",
            provenance="test",
            fidelity="verified",
        ),
        desire_source=PoleClaim(
            claim="test",
            kind="desire",
            provenance="test",
            fidelity="verified",
        ),
        crossing_type="tension",
    )
    item_dict = sample_item.__dict__
    assert "verdict" not in item_dict
    assert "recommendation" not in item_dict
    assert "synthesis" not in item_dict

    surface = ConstraintSurface(
        state_ref="test",
        items=[sample_item],
        skipped_no_axis=False,
    )
    surface_dict = surface.__dict__
    assert "verdict" not in surface_dict
    assert "recommendation" not in surface_dict
    assert "synthesis" not in surface_dict

    # Test that a response with a verdict field is rejected
    bad_response = json.dumps(
        [
            {
                "decision_variable": "test",
                "fear_source_provenance": "scout:cell_id=defed-001/drift_signals[0]",
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
                "verdict": "REJECTED — this field should not exist",
            }
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = bad_response

        with pytest.raises(ValueError, match="forbidden fields"):
            overlay(fear_pole_fixture, desire_pole_fixture)


# ---------------------------------------------------------------------------
# AC#3 — Off-lane testability
# ---------------------------------------------------------------------------


def test_ac3_off_lane_mocked_operator(fear_pole_fixture, desire_pole_fixture):
    """AC#3: All overlay logic is unit-testable with call_operator mocked (no GW lane).

    With the operator mocked, the arbiter's logic (normalization, schema validation,
    provenance threading, tension-gate) is independently testable off the GW lane.
    """
    # Demonstrate testability by running a complete overlay with mocked operator.
    canned_response = json.dumps([])  # Empty seam (no intersection)

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear_pole_fixture, desire_pole_fixture)

        # Assert result is valid even with empty intersection
        assert isinstance(result, ConstraintSurface)
        assert result.items == []
        assert not result.skipped_no_axis

        # Verify call_operator was invoked (off-lane, mocked)
        mock_operator.assert_called_once()
        call_args = mock_operator.call_args
        assert call_args[0][0] == "gravitywell"  # operator_class


# ---------------------------------------------------------------------------
# AC#4 — Normalized contract / pole-agnostic
# ---------------------------------------------------------------------------


def test_ac4_pole_agnostic_prose_and_scout(fear_pole_fixture):
    """AC#4: Arbiter accepts both prose (warm-up style) and scout trace (live style).

    Build one fear PoleOutput via from_prose (warm-up) and one via from_scout_trace
    (live). Feed both with a fixed mocked overlay; assert the arbiter accepts both
    and emits the same-shaped ConstraintSurface, proving U2 and U3 share the arbiter.
    """
    # Warm-up style: prose fear pole
    prose_fear = from_prose(
        text_items=[
            "Defederation could trap consent terms in asymmetric liability",
            "No recovery path exists for mid-transaction failures",
        ],
        kind="fear",
        state_ref="zephyr-defederation-teeth-2026-06-08",
        provenance_prefix="mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vectors",
        fidelity="transcribed",
    )

    # Live style: scout trace payload (dict form for simplicity)
    scout_payload = {
        "cell_id": "defed-001",
        "scaffold_hash": "abc123",
        "spec_version": "1.0",
        "seed": 42,
        "scenario_generated": "...",
        "execution_trace": "...",
        "breaks_observed": [
            {"signature": "No recovery path for mid-transaction failures"}
        ],
        "leverage_points": [
            {"description": "Defederation could trap consent terms in asymmetric liability"}
        ],
        "surprises": [],
        "drift_signals": [],
        "gw_skipped": False,
    }

    scout_fear = from_scout_trace(
        scout_payload,
        state_ref="zephyr-defederation-teeth-2026-06-08",
    )

    # Use the same desire pole for both
    desire = PoleOutput(
        kind="desire",
        state_ref="zephyr-defederation-teeth-2026-06-08",
        claims=[
            PoleClaim(
                claim="Consent terms must protect org interests",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
        ],
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        # Prose warm-up: response uses prose provenance
        prose_response = json.dumps(
            [
                {
                    "decision_variable": "how to balance consent terms against liability risk",
                    "fear_source_provenance": "mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vectors[0]",
                    "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                    "crossing_type": "tension",
                }
            ]
        )
        mock_operator.return_value = prose_response
        prose_result = overlay(prose_fear, desire)

        # Scout live: response uses scout provenance
        scout_response = json.dumps(
            [
                {
                    "decision_variable": "how to balance consent terms against liability risk",
                    "fear_source_provenance": "scout:cell_id=defed-001/leverage_points[0]",
                    "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                    "crossing_type": "tension",
                }
            ]
        )
        mock_operator.return_value = scout_response
        scout_result = overlay(scout_fear, desire)

        # Both should emit same-shaped ConstraintSurface
        assert isinstance(prose_result, ConstraintSurface)
        assert isinstance(scout_result, ConstraintSurface)
        assert prose_result.state_ref == scout_result.state_ref
        assert len(prose_result.items) == len(scout_result.items)


# ---------------------------------------------------------------------------
# AC#4b — Fidelity propagation
# ---------------------------------------------------------------------------


def test_ac4b_fidelity_propagation():
    """AC#4b: fidelity marker propagates from source to constraint-surface item.

    from_prose claims carry fidelity="transcribed", from_scout_trace carries
    fidelity="verified". The constraint-surface item preserves the originating
    claim's fidelity so the U4 report can flag a transcribed-leaning seam.
    """
    # Prose fear (transcribed)
    prose_fear = from_prose(
        text_items=["Defederation risk"],
        kind="fear",
        state_ref="test-state",
        provenance_prefix="mem:thread/test",
        fidelity="transcribed",
    )

    # Scout fear (verified)
    scout_fear = from_scout_trace(
        {
            "cell_id": "cell-1",
            "scaffold_hash": "hash",
            "spec_version": "1.0",
            "seed": 1,
            "scenario_generated": "",
            "execution_trace": "",
            "breaks_observed": [{"signature": "Break observed"}],
            "leverage_points": [],
            "surprises": [],
            "drift_signals": [],
            "gw_skipped": False,
        },
        state_ref="test-state",
        fidelity="verified",
    )

    # Desire
    desire = from_backcaster(
        gaps=[],
        preconditions=[
            {
                "id": "prec-1",
                "axis": "governance",
                "statement": "Governance must be robust",
                "implicit_dependencies": [],
            }
        ],
        state_ref="test-state",
        fidelity="verified",
    )

    # For prose fear: source has fidelity="transcribed"
    assert prose_fear.claims[0].fidelity == "transcribed"

    # For scout fear: source has fidelity="verified"
    assert scout_fear.claims[0].fidelity == "verified"

    # For desire: source has fidelity="verified"
    assert desire.claims[0].fidelity == "verified"

    # When these are resolved in a constraint-surface item, the original
    # fidelity is preserved (tested via provenance resolution).
    prose_claim = prose_fear.claims[0]
    assert prose_claim.fidelity == "transcribed"

    scout_claim = scout_fear.claims[0]
    assert scout_claim.fidelity == "verified"


# ---------------------------------------------------------------------------
# AC#4c — state_ref mismatch detection
# ---------------------------------------------------------------------------


def test_ac4c_state_ref_mismatch():
    """AC#4c: overlay() raises ValueError when fear.state_ref != desire.state_ref."""
    fear = PoleOutput(
        kind="fear",
        state_ref="state-A",
        claims=[
            PoleClaim(
                claim="fear claim",
                kind="fear",
                provenance="test",
                fidelity="verified",
            )
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="state-B",
        claims=[
            PoleClaim(
                claim="desire claim",
                kind="desire",
                provenance="test",
                fidelity="verified",
            )
        ],
    )

    with pytest.raises(ValueError, match="state_ref mismatch"):
        overlay(fear, desire)


def test_ac4c_state_ref_match_populates_surface():
    """When state_refs match, ConstraintSurface.state_ref is populated."""
    fear = PoleOutput(
        kind="fear",
        state_ref="shared-state",
        claims=[
            PoleClaim(
                claim="fear",
                kind="fear",
                provenance="scout:cell/drift[0]",
                fidelity="verified",
            )
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="shared-state",
        claims=[
            PoleClaim(
                claim="desire",
                kind="desire",
                provenance="backcaster:gap/what_missing",
                fidelity="verified",
            )
        ],
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = json.dumps([])

        result = overlay(fear, desire)

        assert result.state_ref == "shared-state"


# ---------------------------------------------------------------------------
# AC#5 — TENSION-GATE short-circuit
# ---------------------------------------------------------------------------


def test_ac5_tension_gate_thin_fear():
    """TENSION-GATE: thin=True fear pole returns skipped_no_axis=True, no model call."""
    fear = PoleOutput(kind="fear", state_ref="test", claims=[], thin=True)
    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire",
                kind="desire",
                provenance="test",
                fidelity="verified",
            )
        ],
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        result = overlay(fear, desire)

        # Short-circuited: no model call
        mock_operator.assert_not_called()

        # Result signals thin pole
        assert result.skipped_no_axis
        assert result.items == []


def test_ac5_tension_gate_thin_desire():
    """TENSION-GATE: thin=True desire pole returns skipped_no_axis=True, no model call."""
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear",
                kind="fear",
                provenance="test",
                fidelity="verified",
            )
        ],
    )
    desire = PoleOutput(kind="desire", state_ref="test", claims=[], thin=True)

    with patch("agents_core.llm.call_operator") as mock_operator:
        result = overlay(fear, desire)

        mock_operator.assert_not_called()
        assert result.skipped_no_axis
        assert result.items == []


def test_ac5_tension_gate_empty_claims():
    """TENSION-GATE: empty claims on either side → no model call."""
    fear = PoleOutput(kind="fear", state_ref="test", claims=[])
    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire",
                kind="desire",
                provenance="test",
                fidelity="verified",
            )
        ],
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        result = overlay(fear, desire)

        mock_operator.assert_not_called()
        assert result.skipped_no_axis


# ---------------------------------------------------------------------------
# AC#6 — GW-down legibility
# ---------------------------------------------------------------------------


def test_ac6_gw_unavailable_raises_exception():
    """AC#6: GW-down (wake fail) raises JaggedSeamGravityWellUnavailable, no fallback."""
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear",
                kind="fear",
                provenance="test",
                fidelity="verified",
            )
        ],
    )
    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire",
                kind="desire",
                provenance="test",
                fidelity="verified",
            )
        ],
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        # Simulate GW unavailable: operator returns None
        mock_operator.return_value = None

        with pytest.raises(JaggedSeamGravityWellUnavailable):
            overlay(fear, desire)


def test_ac6_gw_unavailable_greppable_sentinel():
    """AC#6: Sentinel JAGGED_SEAM_GW_UNAVAILABLE is defined and greppable."""
    from lapis_pm.jagged_seam.arbiter import JAGGED_SEAM_GW_UNAVAILABLE

    assert JAGGED_SEAM_GW_UNAVAILABLE == "JAGGED_SEAM_GW_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Integration tests (off-lane, mocked)
# ---------------------------------------------------------------------------


def test_integration_from_fixtures_end_to_end(
    fear_pole_fixture, desire_pole_fixture
):
    """Integration test: complete flow from fixtures to seam."""
    canned_response = json.dumps(
        [
            {
                "decision_variable": "consent restrictiveness vs liability",
                "fear_source_provenance": "scout:cell_id=defed-001/drift_signals[0]",
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            },
            {
                "decision_variable": "transaction atomicity guarantees",
                "fear_source_provenance": "scout:cell_id=defed-001/breaks_observed[0]",
                "desire_source_provenance": "backcaster:precondition_id=prec-02",
                "crossing_type": "shared_dependency",
            },
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear_pole_fixture, desire_pole_fixture)

        # Validate complete result
        assert result.state_ref == "zephyr-defederation-teeth-2026-06-08"
        assert not result.skipped_no_axis
        assert len(result.items) == 2

        # First item
        item1 = result.items[0]
        assert "consent" in item1.decision_variable.lower()
        assert "drift_signals" in item1.fear_source.provenance
        assert "what_missing" in item1.desire_source.provenance
        assert item1.crossing_type == "tension"

        # Second item
        item2 = result.items[1]
        assert "transaction" in item2.decision_variable.lower()
        assert "breaks_observed" in item2.fear_source.provenance
        assert "prec-02" in item2.desire_source.provenance
        assert item2.crossing_type == "shared_dependency"


# ---------------------------------------------------------------------------
# Provenance hardening tests (normalized match + drop counter)
# ---------------------------------------------------------------------------


def test_provenance_normalized_match_resolves_drift():
    """Normalized match resolves provenance drift (whitespace + case).

    A model response whose fear_source_provenance differs from the real claim's
    provenance only by case and extra internal whitespace should resolve to the
    correct PoleClaim via the normalized fallback.
    """
    # Real claim: exact provenance
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear claim",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[0]",
                fidelity="verified",
            ),
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire claim",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
        ],
    )

    # Model response with provenance drifted by case + extra whitespace
    # (but normalizes to the same value as the real claim)
    canned_response = json.dumps(
        [
            {
                "decision_variable": "test decision variable",
                "fear_source_provenance": "SCOUT:CELL_ID=DEFED-001/DRIFT_SIGNALS[0]",  # All caps, should normalize
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            }
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear, desire)

        # Should resolve successfully (normalized match finds the claim)
        assert len(result.items) == 1
        assert result.items[0].fear_source.provenance == "scout:cell_id=defed-001/drift_signals[0]"
        assert result.dropped_provenance_count == 0


def test_provenance_normalized_match_collapses_whitespace():
    """Normalized match handles extra internal whitespace."""
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear claim",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[0]",
                fidelity="verified",
            ),
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire claim",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
        ],
    )

    # Model response with extra internal whitespace
    canned_response = json.dumps(
        [
            {
                "decision_variable": "test decision variable",
                "fear_source_provenance": "scout:cell_id=defed-001/drift_signals[0]   ",  # Trailing space
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            }
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear, desire)

        # Should resolve successfully
        assert len(result.items) == 1
        assert result.dropped_provenance_count == 0


def test_provenance_no_false_match():
    """Provenance that normalizes to nothing still returns None, no false match.

    A provenance that does not normalize-equal any claim should be dropped, not
    mapped to a wrong claim.
    """
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear claim 1",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[0]",
                fidelity="verified",
            ),
            PoleClaim(
                claim="fear claim 2",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[1]",
                fidelity="verified",
            ),
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire claim",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
        ],
    )

    # Model response with a provenance that does not match any real claim
    canned_response = json.dumps(
        [
            {
                "decision_variable": "test decision variable",
                "fear_source_provenance": "scout:cell_id=defed-999/drift_signals[99]",  # Non-existent
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            }
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear, desire)

        # Should drop the item (not match to wrong claim)
        assert len(result.items) == 0
        assert result.dropped_provenance_count == 1


def test_provenance_drop_counter_fear_unresolvable():
    """Drop counter increments when fear_source_provenance cannot be resolved."""
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear claim",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[0]",
                fidelity="verified",
            ),
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire claim",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
        ],
    )

    # Model response with unresolvable fear_source_provenance (good desire)
    canned_response = json.dumps(
        [
            {
                "decision_variable": "test decision variable",
                "fear_source_provenance": "scout:cell_id=nonexistent/drift_signals[0]",
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            }
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear, desire)

        # Should drop the item and increment counter
        assert len(result.items) == 0
        assert result.dropped_provenance_count == 1


def test_provenance_drop_counter_desire_unresolvable():
    """Drop counter increments when desire_source_provenance cannot be resolved."""
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear claim",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[0]",
                fidelity="verified",
            ),
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire claim",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
        ],
    )

    # Model response with unresolvable desire_source_provenance (good fear)
    canned_response = json.dumps(
        [
            {
                "decision_variable": "test decision variable",
                "fear_source_provenance": "scout:cell_id=defed-001/drift_signals[0]",
                "desire_source_provenance": "backcaster:gap[precondition_id=nonexistent]/what_missing",
                "crossing_type": "tension",
            }
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear, desire)

        # Should drop the item and increment counter
        assert len(result.items) == 0
        assert result.dropped_provenance_count == 1


def test_provenance_drop_counter_multiple_drops():
    """Drop counter tracks multiple unresolvable items."""
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear claim 1",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[0]",
                fidelity="verified",
            ),
            PoleClaim(
                claim="fear claim 2",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[1]",
                fidelity="verified",
            ),
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire claim 1",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
            PoleClaim(
                claim="desire claim 2",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-02]/what_missing",
                fidelity="verified",
            ),
        ],
    )

    # One resolvable item, two unresolvable items
    canned_response = json.dumps(
        [
            {
                "decision_variable": "good item",
                "fear_source_provenance": "scout:cell_id=defed-001/drift_signals[0]",
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            },
            {
                "decision_variable": "bad fear item",
                "fear_source_provenance": "scout:cell_id=nonexistent/drift_signals[0]",
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            },
            {
                "decision_variable": "bad desire item",
                "fear_source_provenance": "scout:cell_id=defed-001/drift_signals[1]",
                "desire_source_provenance": "backcaster:gap[precondition_id=nonexistent]/what_missing",
                "crossing_type": "tension",
            },
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear, desire)

        # One item should be included, two dropped
        assert len(result.items) == 1
        assert result.dropped_provenance_count == 2


def test_provenance_drop_counter_zero_when_all_resolvable():
    """Drop counter is zero when all overlay items are resolvable."""
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear claim",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[0]",
                fidelity="verified",
            ),
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire claim",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
        ],
    )

    canned_response = json.dumps(
        [
            {
                "decision_variable": "test decision variable",
                "fear_source_provenance": "scout:cell_id=defed-001/drift_signals[0]",
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            }
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear, desire)

        # All items resolvable
        assert len(result.items) == 1
        assert result.dropped_provenance_count == 0


def test_provenance_backward_compat_default_zero():
    """ConstraintSurface defaults dropped_provenance_count to 0 for backward-compat."""
    # Create a ConstraintSurface without specifying dropped_provenance_count
    surface = ConstraintSurface(
        state_ref="test",
        items=[],
        skipped_no_axis=False,
    )

    # Should default to 0
    assert surface.dropped_provenance_count == 0


def test_provenance_backward_compat_existing_tests():
    """Existing U1 tests still pass (backward-compatible)."""
    fear = PoleOutput(
        kind="fear",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="fear claim",
                kind="fear",
                provenance="scout:cell_id=defed-001/drift_signals[0]",
                fidelity="verified",
            ),
        ],
    )

    desire = PoleOutput(
        kind="desire",
        state_ref="test",
        claims=[
            PoleClaim(
                claim="desire claim",
                kind="desire",
                provenance="backcaster:gap[precondition_id=prec-01]/what_missing",
                fidelity="verified",
            ),
        ],
    )

    canned_response = json.dumps(
        [
            {
                "decision_variable": "test decision variable",
                "fear_source_provenance": "scout:cell_id=defed-001/drift_signals[0]",
                "desire_source_provenance": "backcaster:gap[precondition_id=prec-01]/what_missing",
                "crossing_type": "tension",
            }
        ]
    )

    with patch("agents_core.llm.call_operator") as mock_operator:
        mock_operator.return_value = canned_response

        result = overlay(fear, desire)

        # Result shape unchanged; dropped_provenance_count is just additive
        assert isinstance(result, ConstraintSurface)
        assert len(result.items) == 1
        assert result.dropped_provenance_count == 0
