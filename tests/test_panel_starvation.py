"""Tests for lapis_pm.panel_starvation (lapis-pm-degraded-panel-confidence-and-absence-claims-v0, Leg 1).

Coverage (DoD #8): degradation detection at the any-leg-down threshold (R1),
confidence attenuation with raw+attenuated both retained.
"""

from __future__ import annotations

from lapis_pm.panel_starvation import (
    apply_panel_starvation,
    attenuate_confidence,
    is_panel_starved,
    starved_legs,
)


def _healthy_verdict(confidence=0.9) -> dict:
    return {
        "verdict": "fixable",
        "issues": [],
        "confidence": confidence,
        "local_reviewer_witness": {"agreement": "agree"},
        "corroboration_result": {
            "verdict": "clean",
            "claim": "identifiers checked",
            "cross_node_divergence": "agree",
        },
    }


class TestStarvedLegs:
    def test_fully_corroborated_panel_has_no_legs_down(self):
        assert starved_legs(_healthy_verdict()) == []
        assert is_panel_starved(_healthy_verdict()) is False

    def test_local_witness_local_failed_counts_as_down(self):
        v = _healthy_verdict()
        v["local_reviewer_witness"] = {"agreement": "local_failed", "error": "JSONDecodeError: ..."}
        assert "local_witness" in starved_legs(v)
        assert is_panel_starved(v) is True

    def test_missing_local_witness_block_counts_as_down(self):
        v = _healthy_verdict()
        del v["local_reviewer_witness"]
        assert "local_witness" in starved_legs(v)

    def test_corroboration_substrate_unavailable_counts_as_down(self):
        v = _healthy_verdict()
        v["corroboration_result"] = {
            "verdict": "uncertain",
            "claim": "(substrate unavailable)",
            "notes": "LLM unavailable: AttributeError",
            "cross_node_divergence": "agree",
        }
        assert "corroboration" in starved_legs(v)

    def test_corroboration_no_identifiers_is_not_starvation(self):
        # "(no identifiers extracted)" is a legitimate clean uncertain result,
        # not a substrate-unavailable failure — must not count as a down leg.
        v = _healthy_verdict()
        v["corroboration_result"] = {
            "verdict": "uncertain",
            "claim": "(no identifiers extracted)",
            "notes": "No identifiers found in diff",
            "cross_node_divergence": "agree",
        }
        assert "corroboration" not in starved_legs(v)

    def test_second_node_unavailable_counts_as_down(self):
        v = _healthy_verdict()
        v["corroboration_result"]["cross_node_divergence"] = "node2_unavailable"
        assert "second_node" in starved_legs(v)

    def test_missing_corroboration_block_counts_both_legs_down(self):
        v = _healthy_verdict()
        del v["corroboration_result"]
        legs = starved_legs(v)
        assert "corroboration" in legs
        assert "second_node" in legs

    def test_all_three_legs_down_incident_replay(self):
        # Replay of the PR #220 incident: local witness invalid JSON,
        # corroboration "LLM unavailable", second node unreachable.
        v = {
            "verdict": "fixable",
            "confidence": 0.9,
            "issues": [{"severity": "high", "path": "agents_core/phala_tee.py",
                        "note": "`_check_equal` is not defined anywhere"}],
            "local_reviewer_witness": {"agreement": "local_failed", "error": "JSONDecodeError"},
            "corroboration_result": {
                "verdict": "uncertain",
                "claim": "(substrate unavailable)",
                "notes": "LLM unavailable: AttributeError",
                "cross_node_divergence": "node2_unavailable",
            },
        }
        legs = starved_legs(v)
        assert set(legs) == {"local_witness", "corroboration", "second_node"}
        assert is_panel_starved(v) is True

    def test_single_leg_down_still_starved_r1_no_graded_threshold(self):
        # R1: any leg down drops the verdict — not a graded 0.4/0.8 cap.
        v = _healthy_verdict()
        v["local_reviewer_witness"] = {"agreement": "local_failed"}
        assert is_panel_starved(v) is True
        assert len(starved_legs(v)) == 1


class TestAttenuateConfidence:
    def test_zero_legs_down_leaves_confidence_unchanged(self):
        assert attenuate_confidence(0.9, 0, 3) == 0.9

    def test_all_legs_down_floors_confidence_to_zero(self):
        assert attenuate_confidence(0.9, 3, 3) == 0.0

    def test_one_of_three_legs_down_scales_proportionally(self):
        assert attenuate_confidence(0.9, 1, 3) == round(0.9 * 2 / 3, 4)

    def test_non_numeric_confidence_returns_none(self):
        assert attenuate_confidence("high", 1, 3) is None
        assert attenuate_confidence(None, 1, 3) is None


class TestApplyPanelStarvation:
    def test_attaches_panel_starvation_block_and_rewrites_confidence(self):
        v = _healthy_verdict(confidence=0.9)
        v["local_reviewer_witness"] = {"agreement": "local_failed"}
        result = apply_panel_starvation(v)
        assert result is v  # mutates in place
        assert v["panel_starvation"]["starved"] is True
        assert v["panel_starvation"]["legs_down"] == ["local_witness"]
        # Raw value preserved for audit (DoD #3).
        assert v["panel_starvation"]["confidence_raw"] == 0.9
        # Top-level confidence rewritten to the attenuated value.
        assert v["confidence"] == round(0.9 * 2 / 3, 4)

    def test_healthy_panel_confidence_unchanged_but_block_present(self):
        v = _healthy_verdict(confidence=0.8)
        result = apply_panel_starvation(v)
        assert result["panel_starvation"]["starved"] is False
        assert result["panel_starvation"]["legs_down"] == []
        assert result["confidence"] == 0.8
        assert result["panel_starvation"]["confidence_raw"] == 0.8
