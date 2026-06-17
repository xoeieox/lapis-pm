"""Unit tests for facets_gw_eval harness.

Tests metric extraction, two-arm aggregation, verdict logic, and doorman-lease guarding.
Uses mocked envelopes; does NOT call GW or spawn deliberations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from unittest.mock import patch, MagicMock

from lapis_pm.facets_gw_eval import (
    Envelope,
    ArmMetrics,
    EvalPass,
    _parse_elapsed_from_envelope,
    _parse_consensus_and_confidence,
    _parse_operator_requested,
    _compute_verdict,
    _run_eval_pass,
    run_quality_cross_judge,
)


# ---------------------------------------------------------------------------
# Mock envelope fixtures
# ---------------------------------------------------------------------------

def _make_mock_envelope(
    persona_durations_ms: dict[str, int],
    synthesis_duration_ms: int = 500,
    consensus_level: str = "consensus",
    confidence: str = "high",
    operator_requested: str | None = None,
) -> Envelope:
    """Factory for mock envelopes (mocked JSON structure)."""
    return Envelope(
        deliberation_id="test-deliberation-123",
        methodology={
            "duration_ms": {
                **persona_durations_ms,
                "synthesis": synthesis_duration_ms,
            },
            "operator_requested": operator_requested,
            "persona_operators": {
                "persona_1": "gravitywell" if not operator_requested else "haiku",
                "persona_2": "gravitywell" if not operator_requested else "haiku",
            },
            "persona_failures": [],
        },
        synthesis={
            "consensus_level": consensus_level,
            "confidence": confidence,
            "recommendation": "proceed",
        },
        personas_invoked=list(persona_durations_ms.keys()),
    )


# ---------------------------------------------------------------------------
# Tests: metric extraction
# ---------------------------------------------------------------------------

class TestMetricExtraction:
    """Test envelope metric parsing."""

    def test_parse_elapsed_from_envelope(self):
        """Extract duration_ms and convert to seconds."""
        env = _make_mock_envelope(
            persona_durations_ms={"persona_1": 1200, "persona_2": 1500},
            synthesis_duration_ms=800,
        )
        elapsed = _parse_elapsed_from_envelope(env)

        assert elapsed["persona_1"] == pytest.approx(1.2, abs=0.01)
        assert elapsed["persona_2"] == pytest.approx(1.5, abs=0.01)
        assert elapsed["synthesis"] == pytest.approx(0.8, abs=0.01)
        assert elapsed["total"] == pytest.approx(3.5, abs=0.01)

    def test_parse_consensus_and_confidence(self):
        """Extract consensus_level and confidence from synthesis."""
        env = _make_mock_envelope(
            persona_durations_ms={"persona_1": 1000},
            consensus_level="split",
            confidence="medium",
        )
        consensus, confidence = _parse_consensus_and_confidence(env)

        assert consensus == "split"
        assert confidence == "medium"

    def test_parse_consensus_and_confidence_missing(self):
        """Graceful fallback when synthesis missing."""
        env = _make_mock_envelope(persona_durations_ms={"persona_1": 1000})
        env.synthesis = None

        consensus, confidence = _parse_consensus_and_confidence(env)

        assert consensus == "unknown"
        assert confidence == "unknown"

    def test_parse_operator_requested(self):
        """Check operator_requested fallback flag."""
        env_no_fallback = _make_mock_envelope(
            persona_durations_ms={"persona_1": 1000},
            operator_requested=None,
        )
        assert _parse_operator_requested(env_no_fallback) is None

        env_fallback = _make_mock_envelope(
            persona_durations_ms={"persona_1": 1000},
            operator_requested="gravitywell",
        )
        assert _parse_operator_requested(env_fallback) == "gravitywell"


# ---------------------------------------------------------------------------
# Tests: ArmMetrics aggregation
# ---------------------------------------------------------------------------

class TestArmMetrics:
    """Test per-arm metric aggregation."""

    def test_arm_metrics_p50(self):
        """Median elapsed_s."""
        arm = ArmMetrics("test_arm")
        arm.elapsed_s_per_run = [1.0, 2.0, 3.0]

        assert arm.p50() == 2.0

    def test_arm_metrics_p95(self):
        """95th percentile with enough samples."""
        arm = ArmMetrics("test_arm")
        # 20 samples, quantiles n=20 cuts at 5% intervals
        arm.elapsed_s_per_run = [float(i) for i in range(1, 21)]  # [1.0, 2.0, ..., 20.0]

        p95 = arm.p95()
        assert p95 > 15.0  # Should be near the top

    def test_arm_metrics_p95_insufficient_samples(self):
        """p95 with < 2 samples returns 0."""
        arm = ArmMetrics("test_arm")
        arm.elapsed_s_per_run = [1.5]

        assert arm.p95() == 0.0

    def test_arm_metrics_degrade_rate(self):
        """Degrade-rate % (operator_requested / run_count)."""
        arm = ArmMetrics("test_arm")
        arm.run_count = 10
        arm.operator_requested_count = 2

        assert arm.degrade_rate_pct() == 20.0

    def test_arm_metrics_degrade_rate_zero_runs(self):
        """Degrade-rate with no runs is 0%."""
        arm = ArmMetrics("test_arm")
        arm.run_count = 0
        arm.operator_requested_count = 0

        assert arm.degrade_rate_pct() == 0.0


# ---------------------------------------------------------------------------
# Tests: verdict logic
# ---------------------------------------------------------------------------

class TestVerdictLogic:
    """Test verdict computation against thresholds."""

    def _make_pass(
        self,
        arm_a_times: list[float],
        arm_b_times: list[float],
        arm_b_degrade_count: int = 0,
    ) -> EvalPass:
        """Helper: construct a mock EvalPass."""
        arm_a = ArmMetrics("haiku")
        arm_a.run_count = len(arm_a_times)
        arm_a.elapsed_s_per_run = arm_a_times

        arm_b = ArmMetrics("gravitywell")
        arm_b.run_count = len(arm_b_times)
        arm_b.elapsed_s_per_run = arm_b_times
        arm_b.operator_requested_count = arm_b_degrade_count

        return EvalPass(pass_name="clean", pass_at="2026-06-17T00:00:00Z",
                        arm_a=arm_a, arm_b=arm_b)

    def test_verdict_latency_pass(self):
        """Latency verdict: B-A p95 delta <= 180s."""
        clean = self._make_pass(
            arm_a_times=[10.0, 11.0, 12.0, 13.0, 14.0],
            arm_b_times=[11.0, 12.0, 13.0, 14.0, 15.0],  # B slightly slower
        )
        wild = self._make_pass(
            arm_a_times=[10.5] * 5,
            arm_b_times=[11.5] * 5,
        )

        verdict = _compute_verdict(clean, wild)

        assert verdict["latency_p95_delta_pass"] is True
        assert verdict["latency_p95_delta_s"] <= 180

    def test_verdict_latency_fail(self):
        """Latency verdict: B-A p95 > 180s fails."""
        clean = self._make_pass(
            arm_a_times=[10.0] * 10,
            arm_b_times=[200.0] * 10,  # B is much slower
        )
        wild = self._make_pass(
            arm_a_times=[10.0] * 5,
            arm_b_times=[200.0] * 5,
        )

        verdict = _compute_verdict(clean, wild)

        assert verdict["latency_p95_delta_pass"] is False
        assert verdict["latency_p95_delta_s"] > 180

    def test_verdict_degrade_rate_pass(self):
        """Degrade-rate verdict: <= 5% passes."""
        clean = self._make_pass(
            arm_a_times=[10.0] * 10,
            arm_b_times=[11.0] * 10,
            arm_b_degrade_count=0,  # 0% fallback
        )
        wild = self._make_pass(
            arm_a_times=[10.0] * 5,
            arm_b_times=[11.0] * 5,
        )

        verdict = _compute_verdict(clean, wild)

        assert verdict["degrade_rate_pass"] is True
        assert verdict["arm_b_degrade_rate_pct"] == 0.0

    def test_verdict_degrade_rate_fail(self):
        """Degrade-rate verdict: > 5% fails."""
        clean = self._make_pass(
            arm_a_times=[10.0] * 10,
            arm_b_times=[11.0] * 10,
            arm_b_degrade_count=7,  # 70% fallback
        )
        wild = self._make_pass(
            arm_a_times=[10.0] * 5,
            arm_b_times=[11.0] * 5,
        )

        verdict = _compute_verdict(clean, wild)

        assert verdict["degrade_rate_pass"] is False
        assert verdict["arm_b_degrade_rate_pct"] == 70.0

    def test_verdict_variance_normal(self):
        """Variance: wild < 3x clean is acceptable."""
        clean = self._make_pass(
            arm_a_times=[10.0] * 5,
            arm_b_times=[12.0] * 5,
        )
        wild = self._make_pass(
            arm_a_times=[10.0] * 5,
            arm_b_times=[30.0] * 5,  # 2.5x wild vs clean
        )

        verdict = _compute_verdict(clean, wild)

        assert verdict["variance_fragile"] is False
        assert verdict["variance_ratio"] <= 3.0

    def test_verdict_variance_fragile(self):
        """Variance: wild > 3x clean is FRAGILE."""
        clean = self._make_pass(
            arm_a_times=[10.0] * 5,
            arm_b_times=[10.0] * 5,
        )
        wild = self._make_pass(
            arm_a_times=[10.0] * 5,
            arm_b_times=[35.0] * 5,  # 3.5x wild vs clean
        )

        verdict = _compute_verdict(clean, wild)

        assert verdict["variance_fragile"] is True
        assert verdict["variance_ratio"] > 3.0

    def test_verdict_cold_wake_split(self):
        """Cold-wake vs warm latency split."""
        clean = self._make_pass(
            arm_a_times=[5.0, 11.0, 12.0],
            arm_b_times=[15.0, 11.0, 12.0],  # First call (cold) is slower
        )
        clean.arm_b.cold_wake_elapsed_s = [15.0]
        clean.arm_b.warm_elapsed_s = [11.0, 12.0]

        wild = self._make_pass(
            arm_a_times=[5.0] * 3,
            arm_b_times=[12.0] * 3,
        )

        verdict = _compute_verdict(clean, wild)

        # Cold-wake should be the first cold value
        assert verdict["arm_b_cold_wake_s"] == 15.0
        # Warm p95 is based on overall p95 when warm_elapsed_s is populated
        assert "arm_b_warm_p95_s" in verdict

    def test_verdict_overall_pass(self):
        """Overall PASS when latency and degrade pass; quality pending."""
        clean = self._make_pass(
            arm_a_times=[10.0] * 10,
            arm_b_times=[12.0] * 10,
            arm_b_degrade_count=0,
        )
        wild = self._make_pass(
            arm_a_times=[10.0] * 5,
            arm_b_times=[13.0] * 5,
        )

        verdict = _compute_verdict(clean, wild)

        # Latency and degrade should pass; quality is pending (None) until cross-judge runs
        assert verdict["latency_p95_delta_pass"] is True
        assert verdict["degrade_rate_pass"] is True
        assert verdict["quality_pass"] is None
        # Overall pass treats None as True (pending quality judgment)
        assert verdict["overall_pass"] is True

    def test_verdict_overall_fail_on_any(self):
        """Overall FAIL if any metric fails."""
        clean = self._make_pass(
            arm_a_times=[10.0] * 10,
            arm_b_times=[200.0] * 10,  # Latency fails
            arm_b_degrade_count=0,
        )
        wild = self._make_pass(
            arm_a_times=[10.0] * 5,
            arm_b_times=[200.0] * 5,
        )

        verdict = _compute_verdict(clean, wild)

        assert verdict["latency_p95_delta_pass"] is False
        assert verdict["overall_pass"] is False


# ---------------------------------------------------------------------------
# Tests: harness invariants
# ---------------------------------------------------------------------------

class TestHarnessInvariants:
    """Test harness behavioral invariants."""

    def test_fixture_corpus_exists(self):
        """FIXTURE_SPEC_IDS are defined (unit test does not run live eval)."""
        from lapis_pm.facets_gw_eval import FIXTURE_SPEC_IDS
        assert len(FIXTURE_SPEC_IDS) >= 3
        assert all(isinstance(s, str) for s in FIXTURE_SPEC_IDS)

    def test_envelope_metrics_consistency(self):
        """Envelope duration_ms total == sum of parts."""
        env = _make_mock_envelope(
            persona_durations_ms={"p1": 1000, "p2": 2000, "p3": 1500},
            synthesis_duration_ms=500,
        )
        elapsed = _parse_elapsed_from_envelope(env)

        total_expected = (1000 + 2000 + 1500 + 500) / 1000.0
        assert elapsed["total"] == pytest.approx(total_expected, abs=0.01)

    def test_empty_envelope_handling(self):
        """Envelope with no methodology returns empty dict."""
        env = Envelope(
            deliberation_id="test",
            methodology=None,
        )
        elapsed = _parse_elapsed_from_envelope(env)
        assert elapsed == {}

    def test_verdict_field_completeness(self):
        """Verdict dict has all required fields."""
        clean = EvalPass(
            pass_name="clean",
            pass_at="2026-06-17T00:00:00Z",
            arm_a=ArmMetrics("haiku", run_count=1, elapsed_s_per_run=[10.0]),
            arm_b=ArmMetrics("gravitywell", run_count=1, elapsed_s_per_run=[11.0]),
        )
        wild = EvalPass(
            pass_name="wild",
            pass_at="2026-06-17T01:00:00Z",
            arm_a=ArmMetrics("haiku", run_count=1, elapsed_s_per_run=[10.0]),
            arm_b=ArmMetrics("gravitywell", run_count=1, elapsed_s_per_run=[11.0]),
        )

        verdict = _compute_verdict(clean, wild)

        required_fields = {
            "latency_p95_delta_s", "latency_p95_delta_pass",
            "degrade_rate_pass", "quality_pass", "overall_pass",
            "variance_ratio", "variance_fragile",
        }
        assert all(f in verdict for f in required_fields)

    def test_doorman_lease_failure_raises_error(self):
        """Clean pass raises ValueError if doorman lease acquisition fails."""
        with patch('lapis_pm.facets_gw_eval._acquire_doorman_lease', return_value=None):
            with pytest.raises(ValueError, match="Could not acquire doorman lease"):
                _run_eval_pass("clean", [])


# ---------------------------------------------------------------------------
# Tests: quality cross-judge
# ---------------------------------------------------------------------------

class TestQualityCrossJudge:
    """Test quality cross-judge functionality (mocked LLM calls)."""

    def test_cross_judge_all_equivalent(self):
        """Cross-judge returns 100% when all fixtures are equivalent."""
        clean = EvalPass(
            pass_name="clean",
            pass_at="2026-06-17T00:00:00Z",
            arm_a=ArmMetrics("haiku", run_count=3, elapsed_s_per_run=[10.0] * 3,
                           consensus_levels={"consensus": 3},
                           confidences={"high": 3}),
            arm_b=ArmMetrics("gravitywell", run_count=3, elapsed_s_per_run=[11.0] * 3,
                           consensus_levels={"consensus": 3},
                           confidences={"high": 3}),
        )
        spec_paths = [Path(f"/tmp/fixture{i}.md") for i in range(3)]

        # Mock call_claude_cli to return "equivalent" verdict for all fixtures
        mock_verdict = json.dumps({"judgment": "equivalent", "reasoning": "both make same recommendation"})

        with patch('lapis_pm.facets_gw_eval.call_claude_cli', return_value=mock_verdict):
            result = run_quality_cross_judge(clean, spec_paths)

        assert result == 100.0

    def test_cross_judge_mixed_verdicts(self):
        """Cross-judge aggregates equivalent and stronger, excludes weaker."""
        clean = EvalPass(
            pass_name="clean",
            pass_at="2026-06-17T00:00:00Z",
            arm_a=ArmMetrics("haiku", run_count=3, elapsed_s_per_run=[10.0] * 3,
                           consensus_levels={"consensus": 3},
                           confidences={"high": 3}),
            arm_b=ArmMetrics("gravitywell", run_count=3, elapsed_s_per_run=[11.0] * 3,
                           consensus_levels={"consensus": 2, "split": 1},
                           confidences={"high": 3}),
        )
        spec_paths = [Path(f"/tmp/fixture{i}.md") for i in range(3)]

        verdicts = [
            json.dumps({"judgment": "equivalent", "reasoning": "same"}),
            json.dumps({"judgment": "stronger", "reasoning": "clearer"}),
            json.dumps({"judgment": "weaker", "reasoning": "less confident"}),
        ]

        with patch('lapis_pm.facets_gw_eval.call_claude_cli', side_effect=verdicts):
            result = run_quality_cross_judge(clean, spec_paths)

        # 2 out of 3 are equivalent-or-stronger = 66.7%
        assert result == pytest.approx(66.67, abs=0.1)

    def test_cross_judge_no_runs(self):
        """Cross-judge returns None if no runs in clean pass."""
        clean = EvalPass(
            pass_name="clean",
            pass_at="2026-06-17T00:00:00Z",
            arm_a=ArmMetrics("haiku", run_count=0),
            arm_b=ArmMetrics("gravitywell", run_count=0),
        )
        spec_paths = [Path(f"/tmp/fixture{i}.md") for i in range(2)]

        result = run_quality_cross_judge(clean, spec_paths)

        assert result is None

    def test_cross_judge_unavailable(self):
        """Cross-judge returns None if call_claude_cli is unavailable."""
        clean = EvalPass(
            pass_name="clean",
            pass_at="2026-06-17T00:00:00Z",
            arm_a=ArmMetrics("haiku", run_count=1, elapsed_s_per_run=[10.0]),
            arm_b=ArmMetrics("gravitywell", run_count=1, elapsed_s_per_run=[11.0]),
        )
        spec_paths = [Path(f"/tmp/fixture{i}.md") for i in range(1)]

        with patch('lapis_pm.facets_gw_eval.call_claude_cli', None):
            result = run_quality_cross_judge(clean, spec_paths)

        assert result is None

    def test_cross_judge_malformed_response(self):
        """Cross-judge handles malformed LLM responses gracefully."""
        clean = EvalPass(
            pass_name="clean",
            pass_at="2026-06-17T00:00:00Z",
            arm_a=ArmMetrics("haiku", run_count=2, elapsed_s_per_run=[10.0] * 2),
            arm_b=ArmMetrics("gravitywell", run_count=2, elapsed_s_per_run=[11.0] * 2),
        )
        spec_paths = [Path(f"/tmp/fixture{i}.md") for i in range(2)]

        # First response is malformed JSON, second is valid
        responses = [
            "not valid json",
            json.dumps({"judgment": "equivalent", "reasoning": "ok"}),
        ]

        with patch('lapis_pm.facets_gw_eval.call_claude_cli', side_effect=responses):
            result = run_quality_cross_judge(clean, spec_paths)

        # Only 1 out of 2 parsed correctly and is equivalent = 50%
        assert result == 50.0
