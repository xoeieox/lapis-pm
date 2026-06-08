"""ScoutScaffold: YAML schema loader and dataclasses.

Scaffold YAML lives at /srv/lapis/scout/sims/<spec-id>.yaml.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StepSpec:
    id: str
    description: str


@dataclass
class ToolSpec:
    id: str
    description: str


@dataclass
class VaultSectionRef:
    path: str
    sections: list[str] = field(default_factory=list)


@dataclass
class ContextConfig:
    chubs: list[str] = field(default_factory=list)
    vault_sections: list[VaultSectionRef] = field(default_factory=list)


@dataclass
class Scenario:
    conditions: list[str] = field(default_factory=list)
    # Ordered dict: tick-label → description.  Insertion order preserved (Python 3.7+).
    time_progression: dict[str, str] = field(default_factory=dict)
    external_state: dict[str, Any] = field(default_factory=dict)
    utilization_pattern: str = ""


@dataclass
class StaticScaffold:
    objective: str
    architecture_sketch: str
    necessary_steps: list[StepSpec] = field(default_factory=list)
    optional_steps: list[StepSpec] = field(default_factory=list)
    available_tools: list[ToolSpec] = field(default_factory=list)
    scenario: Scenario = field(default_factory=Scenario)


@dataclass
class MatrixConfig:
    optional_steps_included: list[list[str]] = field(default_factory=lambda: [[]])
    external_state_severity: list[str] = field(default_factory=lambda: ["healthy"])
    concurrent_load: list[int] = field(default_factory=lambda: [1])
    runs_per_cell: int = 1


# ---------------------------------------------------------------------------
# Top-level ScoutScaffold
# ---------------------------------------------------------------------------


_ALLOWED_PRIORITY_PROFILES = frozenset({"full-pass-once", "variance-resolution", "continuous-baseline"})


@dataclass
class ScoutScaffold:
    spec_id: str
    spec_version: str
    description: str
    static_scaffold: StaticScaffold
    generation_directive: str
    context: ContextConfig = field(default_factory=ContextConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    # Scheduling profile for the night queue.
    # Lives here (ScoutScaffold), NOT on StaticScaffold — adding it to
    # StaticScaffold would invalidate scaffold_hash for every existing scaffold.
    priority_profile: str = "full-pass-once"
    # Design-phase relevance — typed refs this scaffold probes.
    # Outside static_scaffold so scaffold_hash is NOT invalidated.
    covers: list[str] = field(default_factory=list)
    # Provenance block (author id, how-derived); carried through untouched.
    # Outside static_scaffold so scaffold_hash is NOT invalidated.
    provenance: dict | None = None

    # ------------------------------------------------------------------
    # Parameter matrix helpers
    # ------------------------------------------------------------------

    def cell_params(self) -> list[dict[str, Any]]:
        """Enumerate all parameter matrix cells (full factorial)."""
        cells = []
        for opt_steps in self.matrix.optional_steps_included:
            for severity in self.matrix.external_state_severity:
                for load in self.matrix.concurrent_load:
                    cells.append({
                        "optional_steps_included": list(opt_steps),
                        "external_state_severity": severity,
                        "concurrent_load": load,
                    })
        return cells

    @staticmethod
    def cell_id(params: dict[str, Any]) -> str:
        """Stable string key for a cell's parameter row.

        Format: ``opt=<step1>+<step2>,load=<n>,severity=<s>``
        Empty optional_steps_included → ``opt=none``.
        """
        opt_steps: list[str] = params.get("optional_steps_included", [])
        opt_str = "+".join(sorted(opt_steps)) if opt_steps else "none"
        return (
            f"opt={opt_str}"
            f",load={params['concurrent_load']}"
            f",severity={params['external_state_severity']}"
        )

    def scaffold_hash(self, cell_params: dict[str, Any]) -> str:
        """sha256(canonical_json(static_scaffold + cell-resolved-matrix-row)).

        Cosmetic edits to top-level ``description`` do not invalidate prior runs;
        substantive scaffold edits or matrix changes do.
        """
        data = {
            "static_scaffold": _static_scaffold_to_dict(self.static_scaffold),
            "cell": {
                "optional_steps_included": sorted(
                    cell_params.get("optional_steps_included", [])
                ),
                "external_state_severity": cell_params["external_state_severity"],
                "concurrent_load": cell_params["concurrent_load"],
            },
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _static_scaffold_to_dict(s: StaticScaffold) -> dict[str, Any]:
    return {
        "objective": s.objective,
        "architecture_sketch": s.architecture_sketch,
        "necessary_steps": [
            {"id": st.id, "description": st.description} for st in s.necessary_steps
        ],
        "optional_steps": [
            {"id": st.id, "description": st.description} for st in s.optional_steps
        ],
        "available_tools": [
            {"id": t.id, "description": t.description} for t in s.available_tools
        ],
        "scenario": {
            "conditions": s.scenario.conditions,
            "time_progression": s.scenario.time_progression,
            "external_state": s.scenario.external_state,
            "utilization_pattern": s.scenario.utilization_pattern,
        },
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_scaffold(path: str | Path) -> ScoutScaffold:
    """Parse a scaffold YAML file and return a ScoutScaffold."""
    data = yaml.safe_load(Path(path).read_text())

    raw_static = data.get("static_scaffold", {})
    scenario_raw = raw_static.get("scenario", {})
    scenario = Scenario(
        conditions=scenario_raw.get("conditions", []),
        time_progression=dict(scenario_raw.get("time_progression", {})),
        external_state=scenario_raw.get("external_state", {}),
        utilization_pattern=scenario_raw.get("utilization_pattern", ""),
    )
    static = StaticScaffold(
        objective=raw_static.get("objective", ""),
        architecture_sketch=raw_static.get("architecture_sketch", ""),
        necessary_steps=[
            StepSpec(id=s["id"], description=s["description"])
            for s in raw_static.get("necessary_steps", [])
        ],
        optional_steps=[
            StepSpec(id=s["id"], description=s["description"])
            for s in raw_static.get("optional_steps", [])
        ],
        available_tools=[
            ToolSpec(id=t["id"], description=t["description"])
            for t in raw_static.get("available_tools", [])
        ],
        scenario=scenario,
    )

    raw_ctx = data.get("context", {})
    context = ContextConfig(
        chubs=raw_ctx.get("chubs", []),
        vault_sections=[
            VaultSectionRef(path=vs["path"], sections=vs.get("sections", []))
            for vs in raw_ctx.get("vault_sections", [])
        ],
    )

    raw_matrix = data.get("matrix", {})
    matrix = MatrixConfig(
        optional_steps_included=raw_matrix.get("optional_steps_included", [[]]),
        external_state_severity=raw_matrix.get("external_state_severity", ["healthy"]),
        concurrent_load=raw_matrix.get("concurrent_load", [1]),
        runs_per_cell=int(raw_matrix.get("runs_per_cell", 1)),
    )

    priority_profile = data.get("priority_profile", "full-pass-once")
    if priority_profile not in _ALLOWED_PRIORITY_PROFILES:
        allowed = ", ".join(sorted(_ALLOWED_PRIORITY_PROFILES))
        raise ValueError(
            f"Unknown priority_profile {priority_profile!r} in {path}. "
            f"Allowed values: {allowed}"
        )

    covers = data.get("covers", []) or []
    if not isinstance(covers, list):
        covers = []
    provenance_raw = data.get("provenance", None)
    provenance = dict(provenance_raw) if isinstance(provenance_raw, dict) else None

    return ScoutScaffold(
        spec_id=data["spec_id"],
        spec_version=data.get("spec_version", "v0"),
        description=data.get("description", ""),
        static_scaffold=static,
        generation_directive=data.get("generation_directive", ""),
        context=context,
        matrix=matrix,
        priority_profile=priority_profile,
        covers=covers,
        provenance=provenance,
    )
