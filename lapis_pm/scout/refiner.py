"""Scout/Backcaster Refiner v0 — observe→refine loop over sim trace corpora.

observe leg: pure-CPU, zero spend
  - validity-filters traces (quarantine-and-measure, never delete)
  - semantic-clusters break signatures via fastembed nomic-embed-text-v1.5
  - populates cells_observed_in per cluster
  - computes per-scaffold saturation curve (marginal-new-clusters per cell by night)
  - backcaster analog: gap clustering, histogram drift, never-firing categories

refine leg: GW-122B local, zero paid
  - reads landscape, emits ranked proposals with provenance
  - structural retire gate: cannot emit retire without fat-batch-confirmed concave curve
  - on_wake_fail='skip' → degrades to observe-only when GW down
  - writes salience map to /srv/lapis/scout/refiner/

propose-only: no scaffold YAML is mutated, no timer touched.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

SCOUT_TRACES_ROOT = Path("/srv/lapis/scout/traces")
SCOUT_MAPS_ROOT = Path("/srv/lapis/scout/maps")
BACKCASTER_RUNS_ROOT = Path("/srv/lapis/backcaster/runs")
REFINER_OUTPUT_ROOT = Path("/srv/lapis/scout/refiner")

# Cosine thresholds reported for calibration
_COS_THRESHOLDS = (0.70, 0.80, 0.90)
_DEFAULT_COS_THRESHOLD = 0.80

# Fat-batch: night must have at least this many distinct cells to count as fat
_FAT_BATCH_MIN_CELLS = 10

# True-exhaustion: marginal-new-clusters-per-cell below this on a fat night
_SATURATION_THRESHOLD = 0.05

# Retire gate error text (used in re-routed proposal rationale)
_RETIRE_GATE_MSG = (
    "retire re-routed to deepen: retire requires fat-batch-confirmed concave curve "
    "(verdict=true_exhaustion, confidence=high); thin-plateau must route to "
    "deepen/promote/split per invariant."
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QuarantineRecord:
    run_path: str
    reason: str
    night: str
    spec_id: str


@dataclass
class ClusterRecord:
    """One semantic cluster of break-mode signatures."""
    cluster_id: int
    centroid_signature: str
    member_signatures: list[str] = field(default_factory=list)
    instance_count: int = 0
    cells_observed_in: list[str] = field(default_factory=list)
    first_seen_night: str = ""
    frequency: float = 0.0  # instance_count / valid_trace_count


@dataclass
class SaturationCurve:
    spec_id: str
    nights: list[str] = field(default_factory=list)
    cumulative_clusters: list[int] = field(default_factory=list)
    cells_per_night: list[int] = field(default_factory=list)
    marginal_per_cell: list[float] = field(default_factory=list)
    # "true_exhaustion" | "thin_plateau" | "active" | "insufficient_data"
    verdict: str = "insufficient_data"
    # "high" only when fat-batch nights confirm the verdict
    confidence: str = "low"


@dataclass
class RefinerObserveResult:
    spec_id: str
    total_traces: int
    valid_traces: int
    quarantined: list[QuarantineRecord] = field(default_factory=list)
    degenerate_frequency: float = 0.0
    systemic_failure_nights: list[str] = field(default_factory=list)
    clusters: list[ClusterRecord] = field(default_factory=list)
    cluster_counts_by_threshold: dict[str, int] = field(default_factory=dict)
    saturation: SaturationCurve | None = None
    optional_step_classification: dict[str, str] = field(default_factory=dict)


@dataclass
class BackcasterGapCluster:
    cluster_id: int
    centroid_text: str
    goal_slugs: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)


@dataclass
class BackcasterObserveResult:
    total_runs: int
    valid_runs: int
    quarantined: list[QuarantineRecord] = field(default_factory=list)
    degenerate_frequency: float = 0.0
    systemic_failure_nights: list[str] = field(default_factory=list)
    gap_clusters: list[BackcasterGapCluster] = field(default_factory=list)
    histogram_drift: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    never_firing_categories: list[str] = field(default_factory=list)
    unsourced_summary: dict[str, int] = field(default_factory=dict)


@dataclass
class Proposal:
    action: str  # deepen | retire | promote | split | sharpen | spawn
    target: str
    rationale: str
    rank: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinerRefineResult:
    spec_id: str
    proposals: list[Proposal] = field(default_factory=list)
    gw_skipped: bool = False
    prompt_hash: str = ""
    manifest_hash: str = ""
    salience_map_path: str = ""


# ---------------------------------------------------------------------------
# Semantic clustering — fastembed nomic-embed-text-v1.5 (in-process, zero GPU)
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts via fastembed nomic-embed-text-v1.5. Pure-CPU."""
    from fastembed import TextEmbedding  # type: ignore[import]
    model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
    return [list(vec) for vec in model.embed(texts)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _greedy_cluster(
    texts: list[str],
    embeddings: list[list[float]],
    threshold: float = _DEFAULT_COS_THRESHOLD,
) -> list[list[int]]:
    """Greedy cosine clustering over pre-embedded texts.

    Returns list of index-lists (one per cluster, centroid = first element).
    """
    clusters: list[list[int]] = []
    centroids: list[list[float]] = []
    for i, emb in enumerate(embeddings):
        matched = False
        for j, centroid in enumerate(centroids):
            if _cosine(emb, centroid) >= threshold:
                clusters[j].append(i)
                matched = True
                break
        if not matched:
            clusters.append([i])
            centroids.append(emb)
    return clusters


# ---------------------------------------------------------------------------
# Validity predicates
# ---------------------------------------------------------------------------

def _is_degenerate_scout_trace(payload: dict[str, Any]) -> tuple[bool, str]:
    if not payload.get("breaks_observed"):
        return True, "empty breaks_observed"
    if payload.get("parse_failure_count", 0) > 0:
        return True, f"parse_failure_count={payload['parse_failure_count']}"
    return False, ""


def _is_degenerate_backcaster_run(run_dir: Path) -> tuple[bool, str]:
    histogram_path = run_dir / "histogram.yaml"
    gaps_path = run_dir / "gaps.yaml"

    if histogram_path.exists():
        try:
            hist = yaml.safe_load(histogram_path.read_text()) or {}
            if hist and all(v == 0 for v in hist.values() if isinstance(v, int)):
                return True, "all-zero histogram"
        except Exception:
            return True, "histogram parse error"

    if gaps_path.exists():
        try:
            gaps_raw = yaml.safe_load(gaps_path.read_text())
            if gaps_raw is None:
                return True, "empty gaps"
            gaps = gaps_raw if isinstance(gaps_raw, list) else gaps_raw.get("gaps", [])
            if not gaps:
                return True, "empty gaps"
        except Exception:
            return True, "gaps parse error"

    return False, ""


# ---------------------------------------------------------------------------
# Saturation curve
# ---------------------------------------------------------------------------

def _compute_saturation(
    spec_id: str,
    night_to_new: dict[str, int],
    night_to_cells: dict[str, int],
    nights: list[str],
) -> SaturationCurve:
    cumulative = 0
    cum_list: list[int] = []
    marginal_list: list[float] = []
    cells_list: list[int] = []

    for night in nights:
        new = night_to_new.get(night, 0)
        cells = max(night_to_cells.get(night, 1), 1)
        cumulative += new
        cum_list.append(cumulative)
        cells_list.append(cells)
        marginal_list.append(new / cells)

    curve = SaturationCurve(
        spec_id=spec_id,
        nights=nights,
        cumulative_clusters=cum_list,
        cells_per_night=cells_list,
        marginal_per_cell=marginal_list,
    )

    if len(nights) < 2:
        curve.verdict = "insufficient_data"
        curve.confidence = "low"
        return curve

    fat_nights = [n for n in nights if night_to_cells.get(n, 0) >= _FAT_BATCH_MIN_CELLS]

    if not fat_nights:
        # Only thin nights seen — cannot certify exhaustion
        curve.verdict = "thin_plateau"
        curve.confidence = "low"
        return curve

    # Concavity check: marginals are non-increasing (recent ≤ prior)
    is_concave = all(
        marginal_list[i] >= marginal_list[i + 1]
        for i in range(len(marginal_list) - 1)
    )

    recent = marginal_list[-3:] if len(marginal_list) >= 3 else marginal_list
    below_threshold = all(m < _SATURATION_THRESHOLD for m in recent)

    if below_threshold and is_concave and len(fat_nights) >= 2:
        curve.verdict = "true_exhaustion"
        curve.confidence = "high"
    elif below_threshold and is_concave and len(fat_nights) >= 1:
        curve.verdict = "true_exhaustion"
        curve.confidence = "low"
    else:
        # Fat nights exist but not certifiably exhausted → still active.
        # thin_plateau (no fat nights) is handled by the early return above.
        curve.verdict = "active"
        curve.confidence = "high"

    return curve


# ---------------------------------------------------------------------------
# Scout observe
# ---------------------------------------------------------------------------

def refiner_observe(
    spec_id: str,
    *,
    traces_root: Path | None = None,
    maps_root: Path | None = None,
    cos_threshold: float = _DEFAULT_COS_THRESHOLD,
) -> RefinerObserveResult:
    """Semantic observe pass over Scout traces for one spec.

    Extends digest() with:
    - validity filter + quarantine-and-measure (never deletes)
    - fastembed semantic clustering of unique break signatures
    - cells_observed_in population per cluster
    - saturation curve: marginal-new-clusters per cell, per night
    """
    if traces_root is None:
        traces_root = SCOUT_TRACES_ROOT

    spec_dir = traces_root / spec_id
    if not spec_dir.exists():
        raise FileNotFoundError(
            f"No traces directory for spec_id={spec_id!r} at {spec_dir}"
        )

    trace_files = sorted(spec_dir.rglob("*.json"))
    if not trace_files:
        raise FileNotFoundError(f"No trace JSON files under {spec_dir}")

    quarantined: list[QuarantineRecord] = []
    # sig -> {instances: [...], cells: set, first_night: str}
    sig_index: dict[str, dict[str, Any]] = {}
    optional_step_usage: dict[str, int] = defaultdict(int)
    total_traces = 0
    night_to_cells: dict[str, set[str]] = defaultdict(set)

    for tf in trace_files:
        total_traces += 1
        try:
            data = json.loads(tf.read_bytes())
        except (json.JSONDecodeError, OSError):
            quarantined.append(QuarantineRecord(
                run_path=str(tf), reason="json_parse_error",
                night="unknown", spec_id=spec_id,
            ))
            continue

        prov = data.get("provenance", {})
        night = (prov.get("timestamp") or "")[:10] or "unknown"
        payload = data.get("payload", {})

        degen, reason = _is_degenerate_scout_trace(payload)
        if degen:
            quarantined.append(QuarantineRecord(
                run_path=str(tf), reason=reason,
                night=night, spec_id=spec_id,
            ))
            continue

        cell_id = payload.get("cell_id") or tf.parent.name
        if night != "unknown":
            night_to_cells[night].add(cell_id)

        for break_item in payload.get("breaks_observed", []):
            sig = break_item.get("signature", "")
            if sig not in sig_index:
                sig_index[sig] = {
                    "instances": [],
                    "cells": set(),
                    "first_night": night if night != "unknown" else "",
                }
            entry = sig_index[sig]
            entry["instances"].append({**break_item, "_cell_id": cell_id, "_night": night})
            entry["cells"].add(cell_id)
            # Track earliest night
            if night != "unknown":
                if not entry["first_night"] or night < entry["first_night"]:
                    entry["first_night"] = night

        for tool_id in payload.get("tools_used", {}):
            optional_step_usage[tool_id] += 1

    valid_count = total_traces - len(quarantined)
    degen_freq = len(quarantined) / max(total_traces, 1)

    # Systemic failure nights: nights where valid cell count is 0 but quarantine > 0
    all_nights_seen: set[str] = set(night_to_cells.keys()) | {
        q.night for q in quarantined if q.night != "unknown"
    }
    night_to_degen: dict[str, int] = defaultdict(int)
    for q in quarantined:
        if q.night != "unknown":
            night_to_degen[q.night] += 1

    systemic_nights: list[str] = [
        n for n in sorted(all_nights_seen)
        if len(night_to_cells.get(n, set())) == 0 and night_to_degen.get(n, 0) > 0
    ]

    # Short-circuit: no valid breaks
    if not sig_index:
        return RefinerObserveResult(
            spec_id=spec_id,
            total_traces=total_traces,
            valid_traces=valid_count,
            quarantined=quarantined,
            degenerate_frequency=degen_freq,
            systemic_failure_nights=systemic_nights,
        )

    unique_sigs = list(sig_index.keys())
    embeddings = _embed_texts(unique_sigs)

    # Cluster at multiple thresholds for calibration report
    threshold_counts: dict[str, int] = {}
    for thresh in _COS_THRESHOLDS:
        raw = _greedy_cluster(unique_sigs, embeddings, threshold=thresh)
        threshold_counts[str(thresh)] = len(raw)

    raw_clusters = _greedy_cluster(unique_sigs, embeddings, threshold=cos_threshold)

    # Build ClusterRecord objects (sorted by total instance count descending)
    cluster_records: list[ClusterRecord] = []
    for ci, sig_idxs in enumerate(sorted(raw_clusters, key=lambda idxs: -sum(
        len(sig_index[unique_sigs[i]]["instances"]) for i in idxs
    ))):
        sigs_in_cluster = [unique_sigs[i] for i in sig_idxs]
        centroid = sigs_in_cluster[0]

        all_instances = [
            inst
            for sig in sigs_in_cluster
            for inst in sig_index[sig]["instances"]
        ]
        cells = sorted({inst["_cell_id"] for inst in all_instances})
        first_night = min(
            (sig_index[sig]["first_night"] for sig in sigs_in_cluster
             if sig_index[sig]["first_night"]),
            default="",
        )

        cluster_records.append(ClusterRecord(
            cluster_id=ci,
            centroid_signature=centroid,
            member_signatures=sigs_in_cluster,
            instance_count=len(all_instances),
            cells_observed_in=cells,
            first_seen_night=first_night,
            frequency=len(all_instances) / max(valid_count, 1),
        ))

    # Saturation curve: per-night new-cluster count
    nights_sorted = sorted(all_nights_seen - {"unknown"})
    night_to_new: dict[str, int] = defaultdict(int)
    seen_cluster_ids: set[int] = set()

    for night in nights_sorted:
        for cr in cluster_records:
            if cr.first_seen_night == night and cr.cluster_id not in seen_cluster_ids:
                seen_cluster_ids.add(cr.cluster_id)
                night_to_new[night] += 1

    night_to_cell_count: dict[str, int] = {
        n: len(cells) for n, cells in night_to_cells.items()
    }
    sat = _compute_saturation(spec_id, night_to_new, night_to_cell_count, nights_sorted)

    # Optional step classification
    opt_class: dict[str, str] = {}
    for tool_id, count in optional_step_usage.items():
        freq = count / max(valid_count, 1)
        if freq > 0.7:
            opt_class[tool_id] = "promote"
        elif freq < 0.3:
            opt_class[tool_id] = "confirm-optional"
        else:
            opt_class[tool_id] = "inconclusive"

    return RefinerObserveResult(
        spec_id=spec_id,
        total_traces=total_traces,
        valid_traces=valid_count,
        quarantined=quarantined,
        degenerate_frequency=degen_freq,
        systemic_failure_nights=systemic_nights,
        clusters=cluster_records,
        cluster_counts_by_threshold=threshold_counts,
        saturation=sat,
        optional_step_classification=opt_class,
    )


# ---------------------------------------------------------------------------
# Backcaster observe
# ---------------------------------------------------------------------------

def backcaster_observe(
    *,
    runs_root: Path | None = None,
    goal_filter: str | None = None,
    cos_threshold: float = _DEFAULT_COS_THRESHOLD,
) -> BackcasterObserveResult:
    """Validity-filtered landscape over backcaster runs.

    Covers:
    - validity filter + quarantine (all-zero histograms, empty gaps)
    - semantic gap clustering (what_missing texts)
    - histogram drift per goal
    - never-firing category detection
    - unsourced precondition counts
    """
    if runs_root is None:
        runs_root = BACKCASTER_RUNS_ROOT

    if not runs_root.exists():
        raise FileNotFoundError(f"Backcaster runs root not found: {runs_root}")

    run_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())
    if goal_filter:
        run_dirs = [d for d in run_dirs if goal_filter in d.name]

    quarantined: list[QuarantineRecord] = []
    valid_gaps: list[dict[str, Any]] = []
    histogram_drift: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unsourced_summary: dict[str, int] = defaultdict(int)
    total_runs = 0

    # night -> {valid_count, degen_count}
    night_valid: dict[str, int] = defaultdict(int)
    night_degen: dict[str, int] = defaultdict(int)

    for run_dir in run_dirs:
        total_runs += 1
        name = run_dir.name
        m = re.match(r"^(\d{4}-\d{2}-\d{2})-\d{4}-(.+)$", name)
        night = m.group(1) if m else "unknown"
        goal_slug = m.group(2) if m else name

        degen, reason = _is_degenerate_backcaster_run(run_dir)
        if degen:
            quarantined.append(QuarantineRecord(
                run_path=str(run_dir), reason=reason,
                night=night, spec_id=goal_slug,
            ))
            if night != "unknown":
                night_degen[night] += 1
            continue

        if night != "unknown":
            night_valid[night] += 1

        gaps_path = run_dir / "gaps.yaml"
        if gaps_path.exists():
            try:
                gaps_raw = yaml.safe_load(gaps_path.read_text())
                gaps = gaps_raw if isinstance(gaps_raw, list) else (
                    gaps_raw.get("gaps", []) if isinstance(gaps_raw, dict) else []
                )
                for gap in gaps:
                    if not isinstance(gap, dict):
                        continue
                    what_missing = gap.get("what_missing", "")
                    if what_missing:
                        valid_gaps.append({
                            "what_missing": what_missing,
                            "goal_slug": goal_slug,
                            "run_id": name,
                        })
                    if gap.get("unsourced"):
                        unsourced_summary[goal_slug] = (
                            unsourced_summary.get(goal_slug, 0) + 1
                        )
            except Exception:
                pass

        hist_path = run_dir / "histogram.yaml"
        if hist_path.exists():
            try:
                hist = yaml.safe_load(hist_path.read_text()) or {}
                histogram_drift[goal_slug].append({"run_id": name, "counts": dict(hist)})
            except Exception:
                pass

    valid_count = total_runs - len(quarantined)
    degen_freq = len(quarantined) / max(total_runs, 1)

    # Systemic failure nights: all runs that night were degenerate
    all_nights: set[str] = set(night_valid.keys()) | set(night_degen.keys())
    systemic_nights: list[str] = [
        n for n in sorted(all_nights - {"unknown"})
        if night_valid.get(n, 0) == 0 and night_degen.get(n, 0) > 0
    ]

    # Gap semantic clustering
    gap_clusters: list[BackcasterGapCluster] = []
    if valid_gaps:
        gap_texts = [g["what_missing"] for g in valid_gaps]
        gap_embeddings = _embed_texts(gap_texts)
        raw_clusters = _greedy_cluster(gap_texts, gap_embeddings, threshold=cos_threshold)
        for ci, member_idxs in enumerate(sorted(raw_clusters, key=lambda x: -len(x))):
            members = [valid_gaps[i] for i in member_idxs]
            centroid = gap_texts[member_idxs[0]]
            goal_slugs = sorted({m["goal_slug"] for m in members})
            run_ids = [m["run_id"] for m in members]
            gap_clusters.append(BackcasterGapCluster(
                cluster_id=ci,
                centroid_text=centroid,
                goal_slugs=goal_slugs,
                run_ids=run_ids,
            ))

    # Never-firing categories: always 0 across all valid runs
    cat_ever_nonzero: dict[str, bool] = {}
    for runs_hist in histogram_drift.values():
        for run_h in runs_hist:
            for cat, cnt in run_h["counts"].items():
                if cat not in cat_ever_nonzero:
                    cat_ever_nonzero[cat] = False
                if isinstance(cnt, int) and cnt > 0:
                    cat_ever_nonzero[cat] = True

    never_firing = sorted(cat for cat, ever in cat_ever_nonzero.items() if not ever)

    return BackcasterObserveResult(
        total_runs=total_runs,
        valid_runs=valid_count,
        quarantined=quarantined,
        degenerate_frequency=degen_freq,
        systemic_failure_nights=systemic_nights,
        gap_clusters=gap_clusters,
        histogram_drift=dict(histogram_drift),
        never_firing_categories=never_firing,
        unsourced_summary=dict(unsourced_summary),
    )


# ---------------------------------------------------------------------------
# Retire gate (structural invariant)
# ---------------------------------------------------------------------------

def _can_retire(saturation: SaturationCurve | None) -> bool:
    """Structural gate: retire is only valid on fat-batch-confirmed true exhaustion.

    Thin-plateau must route to deepen/promote/split, never retire.
    """
    if saturation is None:
        return False
    return (
        saturation.verdict == "true_exhaustion"
        and saturation.confidence == "high"
    )


# ---------------------------------------------------------------------------
# Refine: proposals via GW-122B
# ---------------------------------------------------------------------------

def _build_scout_prompt(observe: RefinerObserveResult) -> str:
    top_clusters = observe.clusters[:10]
    cluster_lines = "\n".join(
        f"  cluster_{c.cluster_id}: \"{c.centroid_signature[:80]}\" "
        f"(freq={c.frequency:.2f}, cells={c.cells_observed_in[:3]}, "
        f"first={c.first_seen_night})"
        for c in top_clusters
    ) or "  (none)"

    sat = observe.saturation
    sat_line = (
        f"verdict={sat.verdict}, confidence={sat.confidence}, "
        f"nights={len(sat.nights)}, "
        f"recent_marginal={sat.marginal_per_cell[-1]:.3f}"
        if sat and sat.nights and sat.marginal_per_cell else "insufficient_data"
    )
    promotes = [k for k, v in observe.optional_step_classification.items() if v == "promote"]

    return f"""You are the Lapis Refiner analyzing a Scout failure-mode landscape.

SPEC: {observe.spec_id}
VALID TRACES: {observe.valid_traces} / {observe.total_traces}
QUARANTINED: {len(observe.quarantined)} degenerate runs
SYSTEMIC FAILURE NIGHTS: {observe.systemic_failure_nights or "none"}
SATURATION: {sat_line}
SEMANTIC CLUSTERS ({len(observe.clusters)} total at cos≥0.80):
{cluster_lines}
OPTIONAL STEPS TO PROMOTE: {promotes or "none"}

Emit 3-7 ranked proposals as a JSON array. Each entry:
{{
  "action": "deepen|retire|promote|split",
  "target": "<scaffold_id or cluster_id or step_name>",
  "rationale": "<1-2 sentences citing specific cluster/cell evidence>",
  "rank": <integer, 1=highest priority>
}}

Vocabulary:
- deepen: schedule a fatter matrix run (widen an axis or bump runs_per_cell)
- retire: park scaffold (ONLY valid when saturation=true_exhaustion AND confidence=high)
- promote: a recurring tools_wished_for / drift_signal → new optional_step or scaffold
- split: high-frequency cluster confined to one cell-region → focused sub-scaffold

HARD CONSTRAINT: "retire" is FORBIDDEN unless saturation verdict="true_exhaustion"
AND confidence="high". If saturation is thin_plateau or active, use "deepen" or "split".

Return ONLY the JSON array, no prose before or after."""


def _build_backcaster_prompt(observe: BackcasterObserveResult) -> str:
    top_clusters = observe.gap_clusters[:8]
    cluster_lines = "\n".join(
        f"  cluster_{c.cluster_id}: \"{c.centroid_text[:80]}\" "
        f"(goals={c.goal_slugs})"
        for c in top_clusters
    ) or "  (none)"

    return f"""You are the Lapis Refiner analyzing a Backcaster gap landscape.

VALID RUNS: {observe.valid_runs} / {observe.total_runs}
QUARANTINED: {len(observe.quarantined)} degenerate runs
SYSTEMIC FAILURE NIGHTS: {observe.systemic_failure_nights or "none"}
NEVER-FIRING CATEGORIES: {observe.never_firing_categories or "none"}
UNSOURCED BY GOAL: {observe.unsourced_summary}
TOP GAP CLUSTERS ({len(observe.gap_clusters)} total):
{cluster_lines}

Emit 3-7 ranked proposals as a JSON array. Each entry:
{{
  "action": "sharpen|spawn",
  "target": "<goal_slug or gap_cluster_id or category>",
  "rationale": "<1-2 sentences citing specific gap cluster or histogram evidence>",
  "rank": <integer, 1=highest priority>
}}

Vocabulary:
- sharpen: tighten an existing held-goal based on a recurring gap pattern
- spawn: a recurring cross-goal what_missing → new held-goal

Return ONLY the JSON array, no prose before or after."""


def _parse_proposals(
    text: str | None,
    spec_id: str,
    saturation: SaturationCurve | None,
) -> tuple[list[Proposal], bool]:
    """Parse GW response and enforce the structural retire gate.

    Returns (proposals, gw_skipped).
    """
    if text is None:
        return [], True

    try:
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if not m:
            return [], False
        raw = json.loads(m.group(0))
    except (json.JSONDecodeError, AttributeError):
        return [], False

    proposals: list[Proposal] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).lower().strip()
        target = str(item.get("target", ""))
        rationale = str(item.get("rationale", ""))
        rank = int(item.get("rank", 99))

        # Structural retire gate — reroute to deepen if not certifiable
        if action == "retire" and not _can_retire(saturation):
            action = "deepen"
            rationale = f"[{_RETIRE_GATE_MSG}] {rationale}"

        proposals.append(Proposal(
            action=action,
            target=target,
            rationale=rationale,
            rank=rank,
            provenance={
                "spec_id": spec_id,
                "agent_id": "lapis-refiner/refine",
                "source": "gw-122b",
            },
        ))

    proposals.sort(key=lambda p: p.rank)
    return proposals, False


def refiner_refine(
    scout_observe: RefinerObserveResult | None,
    backcaster_result: BackcasterObserveResult | None,
    *,
    output_root: Path | None = None,
) -> list[RefinerRefineResult]:
    """Emit ranked proposals via GW-122B.

    propose-only: no scaffold YAML is mutated, no timer touched.
    Degrades to observe-only (gw_skipped=True) when GW is unreachable.
    """
    try:
        from agents_core.llm import call_operator  # type: ignore[import]
    except ImportError:
        log.warning("agents_core.llm unavailable — refine leg will be skipped")
        call_operator = None  # type: ignore[assignment]

    if output_root is None:
        output_root = REFINER_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[RefinerRefineResult] = []

    def _call_gw(prompt: str) -> tuple[str | None, list]:
        if call_operator is None:
            return None, []
        prov_out: list[Any] = []
        text = call_operator(
            "gravitywell",
            prompt=prompt,
            on_wake_fail="skip",
            _provenance_out=prov_out,
        )
        return text, prov_out

    def _manifest_hash_for(
        proposals: list[Proposal],
        prompt_hash: str,
        effective_operator: str,
        spec_id: str,
    ) -> str:
        payload = json.dumps({
            "agent_id": "lapis-refiner/refine",
            "effective_operator": effective_operator,
            "prompt_hash": prompt_hash,
            "proposals": [
                {"action": p.action, "target": p.target, "rank": p.rank}
                for p in proposals
            ],
            "spec_id": spec_id,
        }, sort_keys=True)
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    # Scout refine
    if scout_observe is not None:
        prompt = _build_scout_prompt(scout_observe)
        prompt_hash = "sha256:" + hashlib.sha256(prompt.encode()).hexdigest()
        response, prov_out = _call_gw(prompt)
        effective_operator = prov_out[-1][1] if prov_out else "gravitywell"
        proposals, gw_skipped = _parse_proposals(
            response, scout_observe.spec_id, scout_observe.saturation
        )
        for p in proposals:
            p.provenance["effective_operator"] = effective_operator
        manifest_hash = _manifest_hash_for(
            proposals, prompt_hash, effective_operator, scout_observe.spec_id
        )
        result = RefinerRefineResult(
            spec_id=scout_observe.spec_id,
            proposals=proposals,
            gw_skipped=gw_skipped,
            prompt_hash=prompt_hash,
            manifest_hash=manifest_hash,
        )
        salience_path = output_root / f"{scout_observe.spec_id}_salience.md"
        salience_path.write_text(_render_scout_salience(scout_observe, result))
        result.salience_map_path = str(salience_path)
        results.append(result)

    # Backcaster refine
    if backcaster_result is not None:
        prompt = _build_backcaster_prompt(backcaster_result)
        prompt_hash = "sha256:" + hashlib.sha256(prompt.encode()).hexdigest()
        response, prov_out = _call_gw(prompt)
        effective_operator = prov_out[-1][1] if prov_out else "gravitywell"
        proposals, gw_skipped = _parse_proposals(response, "backcaster", None)
        for p in proposals:
            p.provenance["effective_operator"] = effective_operator
        manifest_hash = _manifest_hash_for(
            proposals, prompt_hash, effective_operator, "backcaster"
        )
        result = RefinerRefineResult(
            spec_id="backcaster",
            proposals=proposals,
            gw_skipped=gw_skipped,
            prompt_hash=prompt_hash,
            manifest_hash=manifest_hash,
        )
        salience_path = output_root / "backcaster_salience.md"
        salience_path.write_text(_render_backcaster_salience(backcaster_result, result))
        result.salience_map_path = str(salience_path)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Salience map rendering
# ---------------------------------------------------------------------------

def _render_scout_salience(
    observe: RefinerObserveResult,
    refine: RefinerRefineResult,
) -> str:
    sat = observe.saturation
    sat_desc = (
        f"**{sat.verdict}** (confidence: {sat.confidence}, "
        f"nights: {len(sat.nights)}, "
        f"latest marginal: {sat.marginal_per_cell[-1]:.3f})"
        if sat and sat.nights and sat.marginal_per_cell else "insufficient data"
    )
    lines = [
        f"# Scout Salience Map — {observe.spec_id}",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        "## Landscape Summary",
        f"- Valid traces: {observe.valid_traces} / {observe.total_traces}",
        f"- Semantic clusters: {len(observe.clusters)} "
        f"(cos≥{_DEFAULT_COS_THRESHOLD}; "
        + ", ".join(
            f"≥{t}: {n}"
            for t, n in observe.cluster_counts_by_threshold.items()
        ) + ")" if observe.cluster_counts_by_threshold else
        f"- Semantic clusters: {len(observe.clusters)} (cos≥{_DEFAULT_COS_THRESHOLD})",
        f"- Quarantined: {len(observe.quarantined)} runs (retained, measured)",
        f"- Saturation: {sat_desc}",
        "",
    ]

    if observe.systemic_failure_nights:
        lines += [
            "## Systemic Failure Nights",
            *(f"- {n}" for n in observe.systemic_failure_nights),
            "",
        ]

    if observe.quarantined:
        lines += [
            "## Quarantine Log",
            f"{len(observe.quarantined)} runs excluded from living analysis:",
            *(
                f"- `{q.run_path}` — {q.reason} (night: {q.night})"
                for q in observe.quarantined[:15]
            ),
        ]
        if len(observe.quarantined) > 15:
            lines.append(f"- ... and {len(observe.quarantined) - 15} more")
        lines.append("")

    lines += [
        "## Top Break-Mode Clusters",
        *(
            f"{i+1}. `{c.centroid_signature[:70]}` "
            f"— freq={c.frequency:.0%}, cells={len(c.cells_observed_in)}, "
            f"first={c.first_seen_night}"
            for i, c in enumerate(observe.clusters[:10])
        ),
        "",
    ]

    promotes = [k for k, v in observe.optional_step_classification.items() if v == "promote"]
    if promotes:
        lines += ["## Optional Steps to Promote", *(f"- {s}" for s in promotes), ""]

    if refine.gw_skipped:
        lines += [
            "## Proposals",
            "_GW unavailable at observe time — proposals skipped (observe-only mode)._",
            "",
        ]
    elif refine.proposals:
        lines += [
            "## Proposals (thumb to arm)",
            *(
                f"{p.rank}. **{p.action}** `{p.target}` — {p.rationale}"
                for p in refine.proposals
            ),
            "",
        ]

    lines += [
        "---",
        f"*agent_id: lapis-refiner/observe + lapis-refiner/refine*",
        f"*prompt_hash: {refine.prompt_hash}*",
        f"*manifest_hash: {refine.manifest_hash}*",
    ]
    return "\n".join(lines)


def _render_backcaster_salience(
    observe: BackcasterObserveResult,
    refine: RefinerRefineResult,
) -> str:
    lines = [
        "# Backcaster Salience Map",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        "## Landscape Summary",
        f"- Valid runs: {observe.valid_runs} / {observe.total_runs}",
        f"- Gap clusters: {len(observe.gap_clusters)} (cos≥{_DEFAULT_COS_THRESHOLD})",
        f"- Quarantined: {len(observe.quarantined)} runs (retained, measured)",
        "",
    ]

    if observe.systemic_failure_nights:
        lines += [
            "## Systemic Failure Nights",
            *(f"- {n}" for n in observe.systemic_failure_nights),
            "",
        ]

    if observe.never_firing_categories:
        lines += [
            "## Never-Firing Categories",
            *(f"- `{c}`" for c in observe.never_firing_categories),
            "",
        ]

    if observe.quarantined:
        lines += [
            "## Quarantine Log",
            f"{len(observe.quarantined)} runs excluded from living analysis:",
            *(
                f"- `{q.run_path}` — {q.reason} (night: {q.night})"
                for q in observe.quarantined[:15]
            ),
        ]
        if len(observe.quarantined) > 15:
            lines.append(f"- ... and {len(observe.quarantined) - 15} more")
        lines.append("")

    if observe.gap_clusters:
        lines += [
            "## Top Gap Clusters",
            *(
                f"{i+1}. `{c.centroid_text[:80]}` — goals={c.goal_slugs}"
                for i, c in enumerate(observe.gap_clusters[:10])
            ),
            "",
        ]

    if observe.unsourced_summary:
        lines += [
            "## Unsourced Preconditions by Goal",
            *(f"- {goal}: {count}" for goal, count in sorted(observe.unsourced_summary.items())),
            "",
        ]

    if observe.histogram_drift:
        lines += ["## Histogram Drift (recent runs per goal)", ""]
        for goal_slug in sorted(observe.histogram_drift.keys()):
            runs_hist = observe.histogram_drift[goal_slug]
            lines.append(f"**{goal_slug}** ({len(runs_hist)} runs):")
            for hr in runs_hist[-3:]:
                lines.append(f"  - {hr['run_id']}: {hr['counts']}")
        lines.append("")

    if refine.gw_skipped:
        lines += ["## Proposals", "_GW unavailable — proposals skipped._", ""]
    elif refine.proposals:
        lines += [
            "## Proposals (thumb to arm)",
            *(
                f"{p.rank}. **{p.action}** `{p.target}` — {p.rationale}"
                for p in refine.proposals
            ),
            "",
        ]

    lines += [
        "---",
        f"*agent_id: lapis-refiner/observe + lapis-refiner/refine*",
        f"*prompt_hash: {refine.prompt_hash}*",
        f"*manifest_hash: {refine.manifest_hash}*",
    ]
    return "\n".join(lines)
