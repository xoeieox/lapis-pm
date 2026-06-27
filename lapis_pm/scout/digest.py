"""Pure-CPU map digestion — reads traces, builds failure-mode landscape.

Storage layout:
  Input:  /srv/lapis/scout/traces/<spec-id>/<cell-id>/<run-id>.json
  Output: /srv/lapis/scout/maps/<spec-id>.yaml   (LapisToolReturn-conformant JSON)

No LLM calls at v0.  Clustering uses exact-string + Jaccard token-overlap.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents_core.room_paths import room_path, room_str
from archetypes_core.provenance import InputRef, UpstreamRef, to_lapis_return

from .scaffold import load_scaffold
from .schema import BreakModeAggregate, ScoutMapPayload

SCOUT_TRACES_ROOT = room_path('scout.traces')
SCOUT_MAPS_ROOT = room_path('scout.maps')

NOVELTY_PARROT_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "in", "at", "of", "and", "or", "to",
    "it", "by", "on", "as", "be", "do", "no", "not", "if",
})


def _tokenize(s: str) -> frozenset[str]:
    tokens = re.findall(r"[a-z0-9]+", s.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS)


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Break-mode clustering
# ---------------------------------------------------------------------------


def _cluster_breaks(all_breaks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster break-mode signatures by exact-string or Jaccard > 0.5.

    Returns list of cluster dicts: {signature, instances: [...]}.
    """
    clusters: list[dict[str, Any]] = []
    for break_item in all_breaks:
        sig = break_item.get("signature", "")
        matched = False
        for cluster in clusters:
            if cluster["signature"] == sig or _jaccard(cluster["signature"], sig) > 0.5:
                cluster["instances"].append(break_item)
                matched = True
                break
        if not matched:
            clusters.append({"signature": sig, "instances": [break_item]})
    return clusters


# ---------------------------------------------------------------------------
# Novelty filter — distinct from _cluster_breaks
#
# _cluster_breaks: Jaccard between *two signature strings* to group similar
#                  break modes together.
# _score_novelty:  Jaccard between each cluster's *signature* and the *union
#                  of tokens injected via scaffold context* to flag parroting.
# ---------------------------------------------------------------------------


def _score_novelty(
    break_modes: list[BreakModeAggregate],
    context_token_union: frozenset[str],
) -> None:
    """Set parroted_likely=True on any cluster whose signature overlaps the
    injected context token union by more than NOVELTY_PARROT_THRESHOLD (strict >).

    Mutates break_modes in-place; does not reorder or drop entries.
    """
    for cluster in break_modes:
        sig_tokens = _tokenize(cluster.signature)
        if not sig_tokens or not context_token_union:
            cluster.parroted_likely = False
            continue
        overlap = len(sig_tokens & context_token_union) / len(sig_tokens | context_token_union)
        cluster.parroted_likely = overlap > NOVELTY_PARROT_THRESHOLD


# ---------------------------------------------------------------------------
# Main digest function
# ---------------------------------------------------------------------------


def digest(
    spec_id: str,
    *,
    traces_root: Path | None = None,
    maps_root: Path | None = None,
) -> Path:
    """Read all traces for a spec and write a LapisToolReturn-conformant map.

    Parameters
    ----------
    spec_id:
        The spec identifier (matches the sub-directory under traces_root).
    traces_root:
        Override default ``/srv/lapis/scout/traces`` (useful for tests).
    maps_root:
        Override default ``/srv/lapis/scout/maps`` (useful for tests).

    Returns
    -------
    Path
        Path to the written map JSON file.
    """
    if traces_root is None:
        traces_root = SCOUT_TRACES_ROOT
    if maps_root is None:
        maps_root = SCOUT_MAPS_ROOT

    # Load scaffold to build context token union for novelty filter.
    scaffold_path = room_path('scout.sims', f'{spec_id}.yaml')
    scaffold = load_scaffold(scaffold_path) if scaffold_path.exists() else None

    spec_traces_dir = traces_root / spec_id
    if not spec_traces_dir.exists():
        raise FileNotFoundError(
            f"No traces directory for spec_id={spec_id!r} at {spec_traces_dir}"
        )

    trace_files = sorted(spec_traces_dir.rglob("*.json"))
    if not trace_files:
        raise FileNotFoundError(
            f"No trace JSON files found under {spec_traces_dir}"
        )

    # Accumulators
    all_breaks: list[dict[str, Any]] = []
    all_leverage: list[dict[str, Any]] = []
    all_drift: list[dict[str, Any]] = []
    optional_step_usage: dict[str, int] = defaultdict(int)
    run_count = 0
    last_run = ""
    spec_version = "v0"

    # Provenance records — dual-record discipline:
    # traces appear in BOTH input_refs (type="tool_result" with content_hash)
    # AND upstream_calls (manifest_hash composition pointer).
    input_refs: list[InputRef] = [
        InputRef(
            ref=room_str('scout.sims', f'{spec_id}.yaml'),
            content_hash=None,
            type="file",
        )
    ]
    upstream_calls: list[UpstreamRef] = []

    for trace_file in trace_files:
        raw_bytes = trace_file.read_bytes()
        content_hash = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()

        try:
            data = json.loads(raw_bytes)
        except json.JSONDecodeError:
            continue  # skip corrupt trace files

        prov = data.get("provenance", {})
        manifest_hash = prov.get("manifest_hash")
        agent_id = prov.get("agent_id", "lapis-scout/sim")

        # input_refs: type="tool_result" + content_hash (data lineage)
        input_refs.append(InputRef(
            ref=str(trace_file),
            content_hash=content_hash,
            type="tool_result",
        ))

        # upstream_calls: manifest_hash composition pointer
        if manifest_hash:
            upstream_calls.append(UpstreamRef(
                agent_id=agent_id,
                manifest_hash=manifest_hash,
            ))

        payload = data.get("payload", {})
        spec_version = payload.get("spec_version", spec_version)

        # Track most-recent run date from provenance timestamp
        ts = prov.get("timestamp", "")
        if ts:
            run_date = ts[:10]
            if run_date > last_run:
                last_run = run_date

        run_count += 1
        all_breaks.extend(payload.get("breaks_observed", []))
        all_leverage.extend(payload.get("leverage_points", []))
        all_drift.extend(payload.get("drift_signals", []))

        # Accumulate optional step (tool) usage for promotion signal
        for tool_id in payload.get("tools_used", {}):
            optional_step_usage[tool_id] += 1

    if run_count == 0:
        raise ValueError(f"No valid trace JSON files found under {spec_traces_dir}")

    # ------------------------------------------------------------------
    # Cluster break modes
    # ------------------------------------------------------------------
    clusters = _cluster_breaks(all_breaks)
    break_modes: list[BreakModeAggregate] = []
    for cluster in sorted(clusters, key=lambda c: -len(c["instances"])):
        instances = cluster["instances"]
        frequency = len(instances) / run_count
        sev_dist: dict[str, int] = defaultdict(int)
        steps: list[str] = []
        for inst in instances:
            sev = inst.get("severity", "unknown")
            sev_dist[sev] += 1
            step = inst.get("at_step", "")
            if step and step not in steps:
                steps.append(step)
        break_modes.append(BreakModeAggregate(
            signature=cluster["signature"],
            frequency=frequency,
            severity_distribution=dict(sev_dist),
            associated_steps=steps,
            cells_observed_in=[],  # cell tracking reserved for v1
        ))

    # ------------------------------------------------------------------
    # Novelty filter — score each cluster against injected context tokens
    # ------------------------------------------------------------------
    if scaffold is not None:
        ctx_tokens: frozenset[str] = frozenset()
        for chub_id in scaffold.context.chubs:
            ctx_tokens = ctx_tokens | _tokenize(chub_id)
        for vs in scaffold.context.vault_sections:
            ctx_tokens = ctx_tokens | _tokenize(vs.path)
            for section in vs.sections:
                ctx_tokens = ctx_tokens | _tokenize(section)
        ctx_tokens = ctx_tokens | _tokenize(scaffold.static_scaffold.architecture_sketch)
        ctx_tokens = ctx_tokens | _tokenize(scaffold.static_scaffold.objective)
        _score_novelty(break_modes, ctx_tokens)

    # ------------------------------------------------------------------
    # Optional step promotion signal
    # Thresholds: >0.7 → promote; <0.3 → confirm-optional; else inconclusive
    # ------------------------------------------------------------------
    optional_step_promotion_signal: dict[str, float] = {}
    optional_step_classification: dict[str, str] = {}
    for tool_id, count in optional_step_usage.items():
        freq = count / run_count
        optional_step_promotion_signal[tool_id] = freq
        if freq > 0.7:
            optional_step_classification[tool_id] = "promote"
        elif freq < 0.3:
            optional_step_classification[tool_id] = "confirm-optional"
        else:
            optional_step_classification[tool_id] = "inconclusive"

    # ------------------------------------------------------------------
    # Top drift signals
    # ------------------------------------------------------------------
    drift_counter: dict[str, int] = defaultdict(int)
    for ds in all_drift:
        key = ds.get("invented") or ds.get("at_step") or "unknown"
        drift_counter[key] += 1
    drift_signals_top = [
        {"signal": k, "count": v}
        for k, v in sorted(drift_counter.items(), key=lambda x: -x[1])[:5]
    ]

    # ------------------------------------------------------------------
    # Assemble map payload + summary
    # ------------------------------------------------------------------
    map_payload = ScoutMapPayload(
        spec_id=spec_id,
        spec_version=spec_version,
        runs=run_count,
        break_modes=break_modes,
        leverage_points=all_leverage[:20],
        optional_step_promotion_signal=optional_step_promotion_signal,
        optional_step_classification=optional_step_classification,
        drift_signals_top=drift_signals_top,
        last_run=last_run or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        stale_after_days=30,
    )

    break_desc = ", ".join(
        f"{b.signature[:40]} ({b.frequency:.0%})" for b in break_modes[:3]
    ) or "none"
    promote_desc = ", ".join(
        k for k, v in optional_step_classification.items() if v == "promote"
    ) or "none"
    summary = (
        f"{spec_id} ({spec_version}, {run_count} runs): "
        f"{len(break_modes)} break mode(s) ({break_desc}); "
        f"optional steps to promote: {promote_desc}; "
        f"{len(drift_signals_top)} top drift signal(s)"
    )

    ltr = to_lapis_return(
        payload=map_payload,
        agent_id="lapis-scout/digest",
        tool="map_digest",
        summary=summary,
        model=None,          # pure-CPU digest, no model
        prompt_hash=None,
        input_refs=input_refs,
        upstream_calls=upstream_calls,
        scope_id=None,
    )

    maps_root.mkdir(parents=True, exist_ok=True)
    out_path = maps_root / f"{spec_id}.yaml"
    out_path.write_text(ltr.to_json())
    return out_path
