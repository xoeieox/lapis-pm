"""Scout payload dataclasses — LapisToolReturn payload shapes.

Move 2 has landed: archetypes_core.provenance ships the full envelope.
Import everything from there; nothing is duplicated here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Provenance envelope — Move 2 landed, import directly.
from archetypes_core.provenance import (  # noqa: F401 (re-exported for callers)
    SCHEMA_VERSION,
    InputRef,
    LapisToolReturn,
    Provenance,
    UpstreamRef,
    to_lapis_return,
)
from archetypes_core.corroboration import Citation  # noqa: F401


# ---------------------------------------------------------------------------
# BreakModeAggregate — per-spec digest element
# ---------------------------------------------------------------------------


@dataclass
class BreakModeAggregate:
    """Aggregated break mode over many simulation runs."""

    signature: str
    frequency: float                                  # fraction of runs that exhibited this break
    severity_distribution: dict[str, int] = field(default_factory=dict)
    associated_steps: list[str] = field(default_factory=list)
    cells_observed_in: list[str] = field(default_factory=list)
    parroted_likely: bool = False                     # True iff Jaccard(signature, context_union) > 0.30

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "frequency": self.frequency,
            "severity_distribution": self.severity_distribution,
            "associated_steps": self.associated_steps,
            "cells_observed_in": self.cells_observed_in,
            "parroted_likely": self.parroted_likely,
        }


# ---------------------------------------------------------------------------
# ScoutTracePayload — per-run simulation output
# ---------------------------------------------------------------------------


@dataclass
class ScoutTracePayload:
    """Payload shape for a single pseudocode_simulate run.

    Stored as the ``payload`` field of a LapisToolReturn.
    """

    # Cell / run identity
    scaffold_hash: str      # sha256(canonical_json(static_scaffold + cell-row))
    spec_version: str
    cell_id: str
    seed: int

    # Simulation content
    scenario_generated: str
    execution_trace: str    # Qwen's narrative (the husk)
    breaks_observed: list[dict[str, Any]] = field(default_factory=list)
    leverage_points: list[dict[str, Any]] = field(default_factory=list)
    surprises: list[str] = field(default_factory=list)
    drift_signals: list[dict[str, Any]] = field(default_factory=list)
    tools_used: dict[str, int] = field(default_factory=dict)
    tools_wished_for: list[dict[str, Any]] = field(default_factory=list)
    performance_assessment: str = ""

    # Multi-tick provenance (null-byte-delimited prompt_hash at envelope level;
    # per-tick detail stored here for traceability)
    tick_prompts: list[str] = field(default_factory=list)
    tick_prompt_hashes: list[str] = field(default_factory=list)

    # Parse telemetry (for smoke floor assertion)
    parse_failure_count: int = 0
    total_tick_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scaffold_hash": self.scaffold_hash,
            "spec_version": self.spec_version,
            "cell_id": self.cell_id,
            "seed": self.seed,
            "scenario_generated": self.scenario_generated,
            "execution_trace": self.execution_trace,
            "breaks_observed": self.breaks_observed,
            "leverage_points": self.leverage_points,
            "surprises": self.surprises,
            "drift_signals": self.drift_signals,
            "tools_used": self.tools_used,
            "tools_wished_for": self.tools_wished_for,
            "performance_assessment": self.performance_assessment,
            "tick_prompts": self.tick_prompts,
            "tick_prompt_hashes": self.tick_prompt_hashes,
            "parse_failure_count": self.parse_failure_count,
            "total_tick_count": self.total_tick_count,
        }


# ---------------------------------------------------------------------------
# ScoutMapPayload — per-spec digest output
# ---------------------------------------------------------------------------


@dataclass
class ScoutMapPayload:
    """Payload shape for a map_digest output (per-spec failure-mode landscape).

    Stored as the ``payload`` field of a LapisToolReturn.
    """

    spec_id: str
    spec_version: str
    runs: int
    break_modes: list[BreakModeAggregate] = field(default_factory=list)
    leverage_points: list[dict[str, Any]] = field(default_factory=list)

    # Optional-step promotion signal.
    # Thresholds: >0.7 → promote; <0.3 → confirm-optional; 0.3–0.7 → inconclusive.
    optional_step_promotion_signal: dict[str, float] = field(default_factory=dict)
    optional_step_classification: dict[str, str] = field(default_factory=dict)

    drift_signals_top: list[dict[str, Any]] = field(default_factory=list)
    last_run: str = ""          # ISO date (YYYY-MM-DD)
    stale_after_days: int = 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "spec_version": self.spec_version,
            "runs": self.runs,
            "break_modes": [b.to_dict() for b in self.break_modes],
            "leverage_points": self.leverage_points,
            "optional_step_promotion_signal": self.optional_step_promotion_signal,
            "optional_step_classification": self.optional_step_classification,
            "drift_signals_top": self.drift_signals_top,
            "last_run": self.last_run,
            "stale_after_days": self.stale_after_days,
        }
