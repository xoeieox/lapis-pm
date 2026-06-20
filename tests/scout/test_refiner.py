"""Tests for scout/backcaster refiner v0.

Acceptance criteria coverage:
  AC1 (integration) — refiner_observe over grants-runway-v0 collapses raw signatures to
      <300 semantic clusters at cos≥0.80.
  AC2 (integration) — validity filter quarantines known degenerate backcaster runs and
      emits two void signals (per-goal degenerate-frequency + systemic-failure-night flag).
  AC3 — cells_observed_in populated per cluster.
  AC4 — structural retire gate: thin-plateau never yields retire proposal.
  AC5 — propose-only: no scaffold YAML mutated; output confined to /room equivalent.
  AC6 — salience map renders for scout + backcaster.
  AC7 — zero paid model: observe is pure-CPU; refine uses gravitywell on_wake_fail='skip'.
  AC8 — LapisToolReturn provenance markers on refine output.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lapis_pm.scout.refiner import (
    BackcasterGapCluster,
    BackcasterObserveResult,
    ClusterRecord,
    Proposal,
    QuarantineRecord,
    RefinerObserveResult,
    RefinerRefineResult,
    SaturationCurve,
    _can_retire,
    _compute_saturation,
    _cosine,
    _embed_texts,
    _greedy_cluster,
    _is_degenerate_backcaster_run,
    _is_degenerate_scout_trace,
    _parse_proposals,
    _render_backcaster_salience,
    _render_scout_salience,
    backcaster_observe,
    refiner_observe,
    refiner_refine,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic trace/run builders
# ---------------------------------------------------------------------------

def _write_trace(
    traces_root: Path,
    spec_id: str,
    cell_id: str,
    run_id: str,
    breaks: list[dict],
    night: str = "2026-06-01",
    parse_failure_count: int = 0,
) -> Path:
    cell_dir = traces_root / spec_id / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    path = cell_dir / f"{run_id}.json"
    path.write_text(json.dumps({
        "payload": {
            "cell_id": cell_id,
            "spec_version": "v0",
            "seed": 1,
            "scenario_generated": "test",
            "execution_trace": "test",
            "breaks_observed": breaks,
            "leverage_points": [],
            "surprises": [],
            "drift_signals": [],
            "tools_used": {},
            "tools_wished_for": [],
            "performance_assessment": "",
            "parse_failure_count": parse_failure_count,
            "total_tick_count": 1,
        },
        "provenance": {
            "timestamp": f"{night}T12:00:00Z",
            "agent_id": "lapis-scout/sim",
        },
        "summary": "test trace",
    }))
    return path


def _write_bc_run(
    runs_root: Path,
    run_name: str,
    histogram: dict | None = None,
    gaps: list[dict] | None = None,
) -> Path:
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if histogram is None:
        histogram = {
            "community-formation": 5,
            "cultural-shift": 2,
            "financial": 3,
            "financial.distributive": 2,
            "financial.extractive": 0,
            "financial.neutral": 1,
            "infrastructure": 1,
            "policy": 8,
            "research": 4,
            "research.fallback": 0,
            "software": 2,
        }
    if gaps is None:
        gaps = [
            {
                "precondition_id": "econ-01",
                "what_exists": "Current funding models are inadequate.",
                "what_missing": "A sustainable revenue model for independent auditing.",
                "what_miswired": "Incentives are misaligned.",
                "unsourced": True,
            }
        ]
    (run_dir / "histogram.yaml").write_text(yaml.dump(histogram))
    (run_dir / "gaps.yaml").write_text(yaml.dump({"gaps": gaps}))
    (run_dir / "run.yaml").write_text(yaml.dump({"run_id": run_name, "model": "gravitywell"}))
    return run_dir


# ---------------------------------------------------------------------------
# Unit tests — validity predicates
# ---------------------------------------------------------------------------

class TestValidityPredicates:

    def test_degenerate_scout_empty_breaks(self) -> None:
        degen, reason = _is_degenerate_scout_trace({"breaks_observed": []})
        assert degen is True
        assert "empty" in reason

    def test_degenerate_scout_missing_breaks_key(self) -> None:
        degen, reason = _is_degenerate_scout_trace({})
        assert degen is True

    def test_degenerate_scout_parse_failure(self) -> None:
        degen, reason = _is_degenerate_scout_trace({
            "breaks_observed": [{"signature": "x"}],
            "parse_failure_count": 2,
        })
        assert degen is True
        assert "parse_failure_count=2" in reason

    def test_valid_scout_trace(self) -> None:
        degen, _ = _is_degenerate_scout_trace({
            "breaks_observed": [{"signature": "x", "severity": "high"}],
            "parse_failure_count": 0,
        })
        assert degen is False

    def test_degenerate_bc_all_zero_histogram(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        hist = {k: 0 for k in [
            "community-formation", "cultural-shift", "financial",
            "financial.distributive", "financial.extractive", "financial.neutral",
            "infrastructure", "policy", "research", "research.fallback", "software",
        ]}
        (run_dir / "histogram.yaml").write_text(yaml.dump(hist))
        (run_dir / "gaps.yaml").write_text(yaml.dump({"gaps": []}))
        degen, reason = _is_degenerate_backcaster_run(run_dir)
        assert degen is True
        assert "zero" in reason

    def test_degenerate_bc_empty_gaps(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "histogram.yaml").write_text(yaml.dump({"policy": 5}))
        (run_dir / "gaps.yaml").write_text(yaml.dump({"gaps": []}))
        degen, reason = _is_degenerate_backcaster_run(run_dir)
        assert degen is True

    def test_valid_bc_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "histogram.yaml").write_text(yaml.dump({"policy": 5}))
        (run_dir / "gaps.yaml").write_text(yaml.dump({"gaps": [
            {"what_missing": "something important", "unsourced": True}
        ]}))
        degen, _ = _is_degenerate_backcaster_run(run_dir)
        assert degen is False


# ---------------------------------------------------------------------------
# Unit tests — semantic clustering
# ---------------------------------------------------------------------------

class TestSemanticClustering:

    def test_cosine_identical(self) -> None:
        v = [1.0, 0.0, 0.5]
        assert abs(_cosine(v, v) - 1.0) < 1e-9

    def test_cosine_orthogonal(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine(a, b)) < 1e-9

    def test_greedy_cluster_identical_texts_collapse(self, tmp_path: Path) -> None:
        texts = ["speculative hoarding", "speculative hoarding", "speculative hoarding"]
        embeddings = _embed_texts(texts)
        clusters = _greedy_cluster(texts, embeddings, threshold=0.80)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_greedy_cluster_distinct_texts_separate(self) -> None:
        texts = ["speculative hoarding", "infrastructure collapse", "security breach"]
        embeddings = _embed_texts(texts)
        clusters = _greedy_cluster(texts, embeddings, threshold=0.80)
        # Should produce at least 2 clusters (they are semantically distinct)
        assert len(clusters) >= 2

    def test_greedy_cluster_paraphrase_collapse(self) -> None:
        # Near-paraphrases with different surface forms collapse at cos≥0.80
        # (string-Jaccard would NOT collapse these; "handler"/"timeout" shares tokens
        #  but the embedding captures the semantic equivalence more robustly)
        texts = [
            "handler timeout under degraded network conditions",
            "handler timed out due to network degradation",
            "network handler timeout in degraded state",
        ]
        embeddings = _embed_texts(texts)
        clusters = _greedy_cluster(texts, embeddings, threshold=0.80)
        assert len(clusters) == 1, (
            f"Expected 3 near-paraphrase texts to collapse to 1 cluster at cos≥0.80; "
            f"got {len(clusters)}"
        )


# ---------------------------------------------------------------------------
# Unit tests — saturation curve
# ---------------------------------------------------------------------------

class TestSaturationCurve:

    def test_insufficient_data_single_night(self) -> None:
        curve = _compute_saturation(
            "test",
            {"2026-06-01": 10},
            {"2026-06-01": 5},
            ["2026-06-01"],
        )
        assert curve.verdict == "insufficient_data"
        assert curve.confidence == "low"

    def test_thin_plateau_no_fat_nights(self) -> None:
        nights = [f"2026-06-0{i}" for i in range(1, 6)]
        night_to_new = {n: 1 for n in nights}
        # All thin (< 10 cells)
        night_to_cells = {n: 3 for n in nights}
        curve = _compute_saturation("test", night_to_new, night_to_cells, nights)
        assert curve.verdict == "thin_plateau"
        assert curve.confidence == "low"

    def test_true_exhaustion_fat_nights_concave(self) -> None:
        # 10 nights: heavy discovery early, then saturates to zero (concave + fat)
        nights = [f"2026-06-{i:02d}" for i in range(1, 11)]
        night_to_new = dict(zip(nights, [50, 20, 10, 5, 2, 0, 0, 0, 0, 0]))
        night_to_cells = {n: 15 for n in nights}  # all fat (≥10)
        curve = _compute_saturation("test", night_to_new, night_to_cells, nights)
        assert curve.verdict == "true_exhaustion"
        assert curve.confidence == "high"

    def test_active_rising_marginals(self) -> None:
        nights = ["2026-06-01", "2026-06-02", "2026-06-03"]
        night_to_new = {"2026-06-01": 10, "2026-06-02": 15, "2026-06-03": 20}
        night_to_cells = {n: 12 for n in nights}
        curve = _compute_saturation("test", night_to_new, night_to_cells, nights)
        assert curve.verdict == "active"


# ---------------------------------------------------------------------------
# AC3 — cells_observed_in populated
# ---------------------------------------------------------------------------

class TestCellsObservedIn:

    def test_cells_observed_in_populated(self, tmp_path: Path) -> None:
        traces_root = tmp_path / "traces"
        _write_trace(
            traces_root, "test-spec", "cell-A", "run-1",
            [{"signature": "handler timeout", "severity": "high"}],
            night="2026-06-01",
        )
        _write_trace(
            traces_root, "test-spec", "cell-B", "run-2",
            [{"signature": "handler timeout under load", "severity": "medium"}],
            night="2026-06-02",
        )

        result = refiner_observe("test-spec", traces_root=traces_root)
        assert result.valid_traces == 2
        # Both break sigs are about "handler timeout" — should cluster together
        all_cells = {cell for c in result.clusters for cell in c.cells_observed_in}
        assert "cell-A" in all_cells
        assert "cell-B" in all_cells
        # Each cluster has cells_observed_in populated (not empty)
        for c in result.clusters:
            assert len(c.cells_observed_in) >= 1


# ---------------------------------------------------------------------------
# AC4 — structural retire gate
# ---------------------------------------------------------------------------

class TestRetireGate:

    def test_can_retire_requires_true_exhaustion_high(self) -> None:
        sat_exhausted = SaturationCurve(
            spec_id="x",
            nights=["n1"],
            verdict="true_exhaustion",
            confidence="high",
        )
        assert _can_retire(sat_exhausted) is True

    def test_cannot_retire_thin_plateau(self) -> None:
        sat_thin = SaturationCurve(
            spec_id="x",
            nights=["n1", "n2"],
            verdict="thin_plateau",
            confidence="low",
        )
        assert _can_retire(sat_thin) is False

    def test_cannot_retire_true_exhaustion_low_confidence(self) -> None:
        sat = SaturationCurve(
            spec_id="x",
            nights=["n1"],
            verdict="true_exhaustion",
            confidence="low",
        )
        assert _can_retire(sat) is False

    def test_cannot_retire_active(self) -> None:
        sat = SaturationCurve(spec_id="x", verdict="active", confidence="high")
        assert _can_retire(sat) is False

    def test_cannot_retire_none_saturation(self) -> None:
        assert _can_retire(None) is False

    def test_retire_rerouted_to_deepen_on_thin_plateau(self) -> None:
        """AC4: thin-plateau input yields deepen, never retire."""
        sat = SaturationCurve(
            spec_id="x",
            nights=["2026-06-01", "2026-06-02"],
            verdict="thin_plateau",
            confidence="low",
        )
        gw_response = json.dumps([
            {"action": "retire", "target": "my-spec", "rationale": "seems done", "rank": 1}
        ])
        proposals, gw_skipped = _parse_proposals(gw_response, "my-spec", sat)

        assert gw_skipped is False
        assert len(proposals) == 1
        assert proposals[0].action == "deepen", (
            f"Expected retire→deepen re-route on thin_plateau, got {proposals[0].action!r}"
        )
        assert "retire re-routed" in proposals[0].rationale

    def test_retire_allowed_on_true_exhaustion_high(self) -> None:
        """retire passes through when saturation is certified."""
        sat = SaturationCurve(
            spec_id="x",
            nights=["2026-06-01", "2026-06-02"],
            verdict="true_exhaustion",
            confidence="high",
        )
        gw_response = json.dumps([
            {"action": "retire", "target": "old-spec", "rationale": "saturated", "rank": 1}
        ])
        proposals, _ = _parse_proposals(gw_response, "old-spec", sat)
        assert proposals[0].action == "retire"

    def test_retire_not_emitted_when_verdict_active(self) -> None:
        """active saturation → retire becomes deepen."""
        sat = SaturationCurve(
            spec_id="x",
            nights=["2026-06-01"],
            verdict="active",
            confidence="high",
        )
        gw_response = json.dumps([
            {"action": "retire", "target": "spec-x", "rationale": "wrong", "rank": 1}
        ])
        proposals, _ = _parse_proposals(gw_response, "spec-x", sat)
        assert proposals[0].action == "deepen"


# ---------------------------------------------------------------------------
# AC5 — propose-only
# ---------------------------------------------------------------------------

class TestProposeOnly:

    def test_no_scaffold_yaml_mutated(self, tmp_path: Path) -> None:
        """refiner_refine writes only to output_root, never touches scaffold paths."""
        output_root = tmp_path / "refiner-out"
        scaffold_dir = tmp_path / "scaffolds"
        scaffold_dir.mkdir()
        scaffold_file = scaffold_dir / "my-spec.yaml"
        scaffold_file.write_text("spec_id: my-spec\nversion: v0\n")
        mtime_before = scaffold_file.stat().st_mtime

        # Build minimal observe result
        observe = RefinerObserveResult(
            spec_id="my-spec",
            total_traces=2,
            valid_traces=2,
            clusters=[
                ClusterRecord(
                    cluster_id=0,
                    centroid_signature="test break",
                    member_signatures=["test break"],
                    instance_count=1,
                    cells_observed_in=["cell-A"],
                    first_seen_night="2026-06-01",
                    frequency=0.5,
                )
            ],
            saturation=SaturationCurve(
                spec_id="my-spec",
                nights=["2026-06-01"],
                verdict="thin_plateau",
                confidence="low",
            ),
        )

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = json.dumps([
                {"action": "deepen", "target": "my-spec", "rationale": "needs more runs", "rank": 1}
            ])
            results = refiner_refine(observe, None, output_root=output_root)

        assert scaffold_file.stat().st_mtime == mtime_before, (
            "refiner_refine must not touch scaffold YAML files"
        )
        assert len(results) == 1
        assert results[0].salience_map_path.startswith(str(output_root))

    def test_output_confined_to_output_root(self, tmp_path: Path) -> None:
        """All refiner output files must be inside output_root."""
        output_root = tmp_path / "refiner-out"
        observe = RefinerObserveResult(
            spec_id="test-spec",
            total_traces=1,
            valid_traces=1,
            clusters=[],
        )
        bc_observe = BackcasterObserveResult(
            total_runs=1,
            valid_runs=1,
        )

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = json.dumps([
                {"action": "sharpen", "target": "goal-x", "rationale": "r", "rank": 1}
            ])
            results = refiner_refine(observe, bc_observe, output_root=output_root)

        for result in results:
            if result.salience_map_path:
                assert result.salience_map_path.startswith(str(output_root)), (
                    f"Output file outside output_root: {result.salience_map_path}"
                )


# ---------------------------------------------------------------------------
# AC6 — salience map renders
# ---------------------------------------------------------------------------

class TestSalienceMapRender:

    def _make_scout_observe(self, spec_id: str = "test-scout") -> RefinerObserveResult:
        return RefinerObserveResult(
            spec_id=spec_id,
            total_traces=10,
            valid_traces=8,
            quarantined=[
                QuarantineRecord(run_path="/run/x.json", reason="empty breaks_observed",
                                 night="2026-06-01", spec_id=spec_id)
            ],
            degenerate_frequency=0.2,
            systemic_failure_nights=[],
            clusters=[
                ClusterRecord(
                    cluster_id=0,
                    centroid_signature="handler timeout under degraded state",
                    member_signatures=["handler timeout under degraded state"],
                    instance_count=5,
                    cells_observed_in=["cell-A", "cell-B"],
                    first_seen_night="2026-06-01",
                    frequency=0.625,
                )
            ],
            cluster_counts_by_threshold={"0.7": 3, "0.8": 4, "0.9": 5},
            saturation=SaturationCurve(
                spec_id=spec_id,
                nights=["2026-06-01", "2026-06-02"],
                cumulative_clusters=[5, 6],
                cells_per_night=[12, 8],
                marginal_per_cell=[0.42, 0.125],
                verdict="thin_plateau",
                confidence="low",
            ),
            optional_step_classification={"cache-lookup": "promote"},
        )

    def _make_bc_observe(self) -> BackcasterObserveResult:
        return BackcasterObserveResult(
            total_runs=10,
            valid_runs=7,
            quarantined=[
                QuarantineRecord(run_path="/runs/x", reason="all-zero histogram",
                                 night="2026-05-23", spec_id="consent-seam-held"),
            ],
            degenerate_frequency=0.3,
            systemic_failure_nights=["2026-05-23"],
            gap_clusters=[
                BackcasterGapCluster(
                    cluster_id=0,
                    centroid_text="sustainable revenue model for independent auditing",
                    goal_slugs=["accountable-router", "consent-seam-held"],
                    run_ids=["2026-05-14-0625-accountable-router"],
                )
            ],
            histogram_drift={"accountable-router": [
                {"run_id": "2026-05-14-0625-accountable-router", "counts": {"policy": 29}}
            ]},
            never_firing_categories=["financial.extractive"],
            unsourced_summary={"accountable-router": 3},
        )

    def test_scout_salience_contains_required_sections(self) -> None:
        observe = self._make_scout_observe()
        refine = RefinerRefineResult(
            spec_id="test-scout",
            proposals=[
                Proposal(action="deepen", target="test-scout", rationale="needs fat batch", rank=1)
            ],
            gw_skipped=False,
            prompt_hash="sha256:abc123",
        )
        text = _render_scout_salience(observe, refine)

        assert "# Scout Salience Map" in text
        assert "Landscape Summary" in text
        assert "Quarantine Log" in text
        assert "Break-Mode Clusters" in text
        assert "Proposals" in text
        assert "deepen" in text
        assert "lapis-refiner/observe" in text
        assert "sha256:abc123" in text

    def test_scout_salience_observe_only_when_gw_skipped(self) -> None:
        observe = self._make_scout_observe()
        refine = RefinerRefineResult(
            spec_id="test-scout", gw_skipped=True, prompt_hash="sha256:x"
        )
        text = _render_scout_salience(observe, refine)
        assert "observe-only mode" in text
        assert "proposals skipped" in text

    def test_backcaster_salience_contains_required_sections(self) -> None:
        observe = self._make_bc_observe()
        refine = RefinerRefineResult(
            spec_id="backcaster",
            proposals=[
                Proposal(action="spawn", target="gap-cluster-0",
                         rationale="recurring across goals", rank=1)
            ],
            gw_skipped=False,
            prompt_hash="sha256:def456",
        )
        text = _render_backcaster_salience(observe, refine)

        assert "# Backcaster Salience Map" in text
        assert "Systemic Failure Nights" in text
        assert "2026-05-23" in text
        assert "Never-Firing Categories" in text
        assert "financial.extractive" in text
        assert "Gap Clusters" in text
        assert "Proposals" in text
        assert "spawn" in text
        assert "lapis-refiner/refine" in text

    def test_salience_maps_written_to_disk(self, tmp_path: Path) -> None:
        """AC6: salience map files are written for both scout and backcaster."""
        observe = self._make_scout_observe("spec-x")
        bc_obs = self._make_bc_observe()
        output_root = tmp_path / "refiner-out"

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = json.dumps([
                {"action": "deepen", "target": "spec-x", "rationale": "r", "rank": 1},
                {"action": "spawn", "target": "gap-0", "rationale": "r", "rank": 2},
            ])
            results = refiner_refine(observe, bc_obs, output_root=output_root)

        assert len(results) == 2
        for result in results:
            assert result.salience_map_path
            assert Path(result.salience_map_path).exists()
            content = Path(result.salience_map_path).read_text()
            assert len(content) > 100


# ---------------------------------------------------------------------------
# AC7 — zero paid model spend
# ---------------------------------------------------------------------------

class TestZeroPaidModel:

    def test_gw_skipped_when_call_operator_returns_none(self, tmp_path: Path) -> None:
        """When GW is down (returns None), gw_skipped=True, no fallback to paid model."""
        output_root = tmp_path / "out"
        observe = RefinerObserveResult(
            spec_id="test", total_traces=1, valid_traces=1, clusters=[]
        )

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = None
            results = refiner_refine(observe, None, output_root=output_root)

        assert len(results) == 1
        assert results[0].gw_skipped is True
        assert results[0].proposals == []

        # Verify on_wake_fail='skip' was passed — no paid-model fallback
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs.get("on_wake_fail") == "skip"

    def test_call_operator_always_uses_gravitywell(self, tmp_path: Path) -> None:
        """observe is pure-CPU; refine always routes to gravitywell, never a paid operator."""
        output_root = tmp_path / "out"
        observe = RefinerObserveResult(
            spec_id="test", total_traces=1, valid_traces=1, clusters=[]
        )

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = "[]"
            refiner_refine(observe, None, output_root=output_root)

        # All calls must be to "gravitywell"
        for call_args in mock_call.call_args_list:
            operator = call_args[0][0]
            assert operator == "gravitywell", (
                f"refiner_refine must only use gravitywell, got {operator!r}"
            )


# ---------------------------------------------------------------------------
# AC8 — LapisToolReturn provenance markers
# ---------------------------------------------------------------------------

class TestProvenance:

    def test_proposals_carry_agent_id_provenance(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        observe = RefinerObserveResult(
            spec_id="my-spec", total_traces=2, valid_traces=2, clusters=[]
        )

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = json.dumps([
                {"action": "deepen", "target": "my-spec", "rationale": "r", "rank": 1}
            ])
            results = refiner_refine(observe, None, output_root=output_root)

        proposals = results[0].proposals
        assert proposals
        prov = proposals[0].provenance
        assert prov.get("agent_id") == "lapis-refiner/refine"
        assert prov.get("source") == "gw-122b"

    def test_prompt_hash_is_sha256(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        observe = RefinerObserveResult(
            spec_id="my-spec", total_traces=1, valid_traces=1, clusters=[]
        )

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = "[]"
            results = refiner_refine(observe, None, output_root=output_root)

        ph = results[0].prompt_hash
        assert ph.startswith("sha256:"), f"prompt_hash should be sha256:... got {ph!r}"
        hex_part = ph[7:]
        assert len(hex_part) == 64
        assert re.match(r"^[0-9a-f]+$", hex_part)

    def test_salience_map_contains_provenance_markers(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        observe = RefinerObserveResult(
            spec_id="prov-spec", total_traces=1, valid_traces=1, clusters=[]
        )

        with patch("agents_core.llm.call_operator") as mock_call:
            mock_call.return_value = "[]"
            results = refiner_refine(observe, None, output_root=output_root)

        content = Path(results[0].salience_map_path).read_text()
        assert "lapis-refiner/observe" in content
        assert "lapis-refiner/refine" in content
        assert "prompt_hash" in content


# ---------------------------------------------------------------------------
# AC1 — integration: semantic collapse on real grants-runway-v0 corpus
# ---------------------------------------------------------------------------

GRANTS_RUNWAY_TRACES = Path("/srv/lapis/scout/traces/grants-runway-v0")


@pytest.mark.integration
def test_ac1_semantic_collapse_grants_runway() -> None:
    """AC1: refiner_observe over grants-runway-v0 collapses raw sigs to <300 at cos≥0.80."""
    if not GRANTS_RUNWAY_TRACES.exists():
        pytest.skip("grants-runway-v0 traces not present")

    result = refiner_observe("grants-runway-v0")
    assert result.valid_traces > 0, "No valid traces found in grants-runway-v0"
    n_clusters = len(result.clusters)

    assert n_clusters < 300, (
        f"Expected <300 semantic clusters at cos≥0.80, got {n_clusters}. "
        f"(valid_traces={result.valid_traces}, total={result.total_traces})"
    )
    # Calibration check: 0.80 should give fewer clusters than 0.70
    c70 = result.cluster_counts_by_threshold.get("0.7", 0)
    c80 = result.cluster_counts_by_threshold.get("0.8", 0)
    assert c80 <= c70, f"Expect cos≥0.80 ≤ cos≥0.70 cluster count; got {c80} vs {c70}"


# ---------------------------------------------------------------------------
# AC2 — integration: validity filter quarantines known degenerate backcaster runs
# ---------------------------------------------------------------------------

BACKCASTER_RUNS = Path("/srv/lapis/backcaster/runs")


@pytest.mark.integration
def test_ac2_validity_filter_quarantine() -> None:
    """AC2: known degenerate runs are quarantined; void signals emitted correctly."""
    if not BACKCASTER_RUNS.exists():
        pytest.skip("Backcaster runs not present")

    result = backcaster_observe()

    # 2026-05-23 was systemic all-zero across all goals
    assert "2026-05-23" in result.systemic_failure_nights, (
        f"Expected 2026-05-23 in systemic_failure_nights; got {result.systemic_failure_nights}"
    )

    # Quarantine log names each degenerate run (not silently dropped)
    quarantine_paths = {q.run_path for q in result.quarantined}
    assert len(quarantine_paths) > 0, "No quarantined runs found"
    # Every quarantined run has a reason
    for q in result.quarantined:
        assert q.reason, f"Quarantine record missing reason: {q}"

    # degenerate_frequency > 0 (we know there are degenerate runs)
    assert result.degenerate_frequency > 0

    # Known degenerate runs: 2026-06-14-1000-* batch (all-zero histograms)
    degen_run_ids = {q.run_path for q in result.quarantined}
    assert any("2026-06-14" in p for p in degen_run_ids), (
        "Expected 2026-06-14 degenerate runs in quarantine"
    )

    # Per-goal degenerate frequency is computable from the quarantine list
    goal_degen: dict[str, int] = {}
    for q in result.quarantined:
        goal_degen[q.spec_id] = goal_degen.get(q.spec_id, 0) + 1
    assert len(goal_degen) > 0, "Cannot compute per-goal degenerate frequency"


@pytest.mark.integration
def test_ac2_cells_observed_in_grants_runway() -> None:
    """AC3 (integration): cells_observed_in populated on real corpus."""
    if not GRANTS_RUNWAY_TRACES.exists():
        pytest.skip("grants-runway-v0 traces not present")

    result = refiner_observe("grants-runway-v0")
    assert result.clusters, "No clusters produced from grants-runway-v0"
    for cluster in result.clusters[:5]:
        assert len(cluster.cells_observed_in) >= 1, (
            f"cells_observed_in empty for cluster {cluster.cluster_id}: "
            f"{cluster.centroid_signature[:40]!r}"
        )
