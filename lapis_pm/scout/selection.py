"""Scout value-based sim selection.

select_worklist(...) -> SelectionPlan

Pure, side-effect-free. Importable by Unit 2 (night coordinator) without change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# Saturation: one full matrix pass. Tunable in v1; no CLI surface.
SATURATION_PCT = 1.0

# Gap alerting: refs older than this many days trigger the alerting flag.
GAP_ALERT_DAYS = 14


# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------


@dataclass
class SelectedEntry:
    """One entry from selected.yaml."""
    sim: str
    target_pct: float = 1.0


@dataclass
class PlanEntry:
    """One sim in the priority or relevance lane."""
    spec_id: str
    scaffold_path: Path
    completion: float        # current clean-trace completion [0, 1]
    lane: str                # "priority" | "relevance"
    target_pct: float        # for priority: declared target; relevance: SATURATION_PCT
    # Units needed to go from current completion to target_pct (0 if already there)
    units_needed: int
    summary: str             # templated natural-language summary for pre-run review


@dataclass
class ParkedEntry:
    """One sim excluded from this run."""
    spec_id: str
    scaffold_path: Path
    # "saturated" | "covers-all-spent" | "quarantined"
    reason: str
    summary: str


@dataclass
class GapEntry:
    """A molten ref with no covering sim (set complement)."""
    ref: str
    age_days: float | None   # days since the ref key was recorded (None if unknown)
    alerting: bool           # True if age_days > GAP_ALERT_DAYS or age unknown


@dataclass
class SelectionPlan:
    """Canonical selection plan for one Scout night run."""
    priority_lane: list[PlanEntry] = field(default_factory=list)
    relevance_lane: list[PlanEntry] = field(default_factory=list)
    parked: list[ParkedEntry] = field(default_factory=list)
    gap_list: list[GapEntry] = field(default_factory=list)
    budget_deadline: float = 0.0      # Unix epoch
    budget_max_units: int | None = None


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


def completion(spec_id: str, scaffold, traces_root: Path) -> float:
    """Fraction of matrix completed: clean_trace_count / matrix_size.

    clean_trace_count = number of .json files under traces_root/<spec_id>/
    (all traces are clean — error runs raise and write nothing, per the runner contract).
    """
    from .scaffold import ScoutScaffold
    cells = scaffold.cell_params()
    runs_per_cell = scaffold.matrix.runs_per_cell
    matrix_size = len(cells) * runs_per_cell
    if matrix_size == 0:
        return 1.0

    clean_count = 0
    spec_dir = traces_root / spec_id
    if spec_dir.exists():
        for cell_params in cells:
            from .scaffold import ScoutScaffold as _SS
            cell_id = _SS.cell_id(cell_params)
            cell_dir = spec_dir / cell_id
            if cell_dir.exists():
                clean_count += sum(1 for f in cell_dir.glob("*.json") if f.is_file())

    return min(clean_count / matrix_size, 1.0)


def _units_needed(current_completion: float, target_pct: float, scaffold) -> int:
    """How many additional units to reach target_pct from current_completion."""
    cells = scaffold.cell_params()
    runs_per_cell = scaffold.matrix.runs_per_cell
    matrix_size = len(cells) * runs_per_cell
    if matrix_size == 0:
        return 0
    already = int(current_completion * matrix_size)
    target = int(target_pct * matrix_size)
    return max(0, target - already)


# ---------------------------------------------------------------------------
# Natural-language summary templates (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _priority_summary(entry: SelectedEntry, comp: float, scaffold) -> str:
    desc = (scaffold.description or "").split("\n")[0].strip()[:80]
    covers_str = ""
    if scaffold.covers:
        covers_str = f" — covers {', '.join(scaffold.covers[:2])}"
    return (
        f"Priority run: {entry.sim} (target {entry.target_pct*100:.0f}%, "
        f"currently {comp*100:.0f}% sampled){covers_str}. {desc}"
    ).strip()


def _relevance_summary(spec_id: str, comp: float, scaffold, molten_refs: list[str]) -> str:
    desc = (scaffold.description or "").split("\n")[0].strip()[:80]
    if molten_refs:
        refs_str = ", ".join(f"`{r}`" for r in molten_refs[:2])
        return (
            f"Probing {refs_str} — {spec_id}, {comp*100:.0f}% sampled. {desc}"
        ).strip()
    return f"Relevance run: {spec_id}, {comp*100:.0f}% sampled. {desc}".strip()


def _parked_summary(spec_id: str, reason: str, scaffold) -> str:
    if reason == "saturated":
        matrix_size = len(scaffold.cell_params()) * scaffold.matrix.runs_per_cell
        return (
            f"Parked (saturated): {spec_id} has completed one full matrix pass "
            f"({matrix_size} cells × runs); no new coverage possible this cycle."
        )
    if reason == "covers-all-spent":
        covers_str = ", ".join(f"`{r}`" for r in (scaffold.covers or [])[:3])
        return (
            f"Parked: the questions this sim probed ({covers_str}) were bound "
            f"2026-06-08 or earlier, so its probing window has closed "
            f"(administratively closed, not resolved)."
        )
    if reason == "quarantined":
        return (
            f"Parked (quarantined): {spec_id} has been excluded due to repeated "
            f"errors this run."
        )
    return f"Parked ({reason}): {spec_id}."


def _gap_summary(ref: str, age_days: float | None, alerting: bool) -> str:
    age_str = f"{age_days:.0f}d" if age_days is not None else "age unknown"
    alert_str = " *** ALERT: no sim coverage" if alerting else ""
    return f"Gap: {ref} is molten with no covering sim ({age_str} old){alert_str}."


# ---------------------------------------------------------------------------
# Core selection logic
# ---------------------------------------------------------------------------


def select_worklist(
    scaffolds: list,
    selected: list[SelectedEntry],
    traces_root: Path,
    liveness_fn: Callable[[str], str],
    budget=None,
    quarantined_ids: set[str] | None = None,
) -> SelectionPlan:
    """Build a SelectionPlan from scaffolds, selected entries, and liveness.

    Pure: reads traces_root (read-only), calls liveness_fn, writes nothing.

    Parameters
    ----------
    scaffolds:
        List of ScoutScaffold objects to consider.
    selected:
        Priority-lane entries from selected.yaml (may be empty).
    traces_root:
        Root directory for clean traces (read-only).
    liveness_fn:
        Callable(ref: str) -> "molten" | "spent". Injected for testability.
    budget:
        NightBudget (optional; stored in the plan for rendering only).
    quarantined_ids:
        Set of spec_ids quarantined this run (excluded from all lanes).
    """
    from .budget import NightBudget

    quarantined_ids = quarantined_ids or set()
    selected_map = {e.sim: e for e in selected}
    scaffold_map = {s.spec_id: s for s in scaffolds}

    plan = SelectionPlan()
    if budget is not None:
        plan.budget_deadline = budget.deadline
        plan.budget_max_units = budget.max_units

    # Track all molten refs across scaffolds (for gap computation)
    all_molten_refs: set[str] = set()

    # Classify each scaffold
    priority_entries: list[PlanEntry] = []
    relevance_candidates: list[tuple[float, bool, str, object]] = []  # (comp, has_molten, spec_id, scaffold)
    parked: list[ParkedEntry] = []

    # Pre-compute liveness for all unique covers refs
    all_refs: set[str] = set()
    for s in scaffolds:
        all_refs.update(s.covers or [])
    ref_liveness: dict[str, str] = {}
    for ref in all_refs:
        ref_liveness[ref] = liveness_fn(ref)
        if ref_liveness[ref] == "molten":
            all_molten_refs.add(ref)

    for scaffold in scaffolds:
        spec_id = scaffold.spec_id

        # Quarantined sims → parked
        if spec_id in quarantined_ids:
            parked.append(ParkedEntry(
                spec_id=spec_id,
                scaffold_path=_scaffold_path(scaffold),
                reason="quarantined",
                summary=_parked_summary(spec_id, "quarantined", scaffold),
            ))
            continue

        comp = completion(spec_id, scaffold, traces_root)

        # Priority lane: explicit selection supersedes other park rules
        if spec_id in selected_map:
            entry = selected_map[spec_id]
            needed = _units_needed(comp, entry.target_pct, scaffold)
            priority_entries.append(PlanEntry(
                spec_id=spec_id,
                scaffold_path=_scaffold_path(scaffold),
                completion=comp,
                lane="priority",
                target_pct=entry.target_pct,
                units_needed=needed,
                summary=_priority_summary(entry, comp, scaffold),
            ))
            continue

        # Check covers liveness
        covers = scaffold.covers or []
        if covers:
            cover_states = [ref_liveness.get(r, "molten") for r in covers]
            if all(s == "spent" for s in cover_states):
                parked.append(ParkedEntry(
                    spec_id=spec_id,
                    scaffold_path=_scaffold_path(scaffold),
                    reason="covers-all-spent",
                    summary=_parked_summary(spec_id, "covers-all-spent", scaffold),
                ))
                continue

        # Check saturation
        if comp >= SATURATION_PCT:
            parked.append(ParkedEntry(
                spec_id=spec_id,
                scaffold_path=_scaffold_path(scaffold),
                reason="saturated",
                summary=_parked_summary(spec_id, "saturated", scaffold),
            ))
            continue

        # Relevance lane candidate
        molten_refs = [r for r in covers if ref_liveness.get(r) == "molten"]
        has_molten = bool(molten_refs)
        relevance_candidates.append((comp, has_molten, spec_id, scaffold, molten_refs))

    # Order relevance lane: molten-covers sims first (least-sampled), then undeclared (least-sampled)
    with_molten = [(comp, sid, sc, refs) for comp, has_m, sid, sc, refs in relevance_candidates if has_m]
    without_molten = [(comp, sid, sc, refs) for comp, has_m, sid, sc, refs in relevance_candidates if not has_m]
    with_molten.sort(key=lambda x: x[0])       # least-sampled first
    without_molten.sort(key=lambda x: x[0])    # least-sampled first

    for comp, spec_id, scaffold, molten_refs in with_molten + without_molten:
        needed = _units_needed(comp, SATURATION_PCT, scaffold)
        plan.relevance_lane.append(PlanEntry(
            spec_id=spec_id,
            scaffold_path=_scaffold_path(scaffold),
            completion=comp,
            lane="relevance",
            target_pct=SATURATION_PCT,
            units_needed=needed,
            summary=_relevance_summary(spec_id, comp, scaffold, molten_refs),
        ))

    # Priority lane: preserve declared order
    for e in selected:
        if e.sim not in scaffold_map:
            log.warning("select_worklist: selected sim %r not found in sims_dir — skipping", e.sim)
            continue
        # Find in priority_entries (already built)
    plan.priority_lane = priority_entries

    plan.parked = parked

    # Gap list: molten refs with no covering sim
    covered_molten = set()
    for s in scaffolds:
        for r in (s.covers or []):
            if ref_liveness.get(r) == "molten":
                covered_molten.add(r)

    # All molten refs that appear in any covers declaration
    uncovered = all_molten_refs - covered_molten
    gap_entries = []
    for ref in sorted(uncovered):
        age = _ref_age_days(ref)
        alerting = age is None or age > GAP_ALERT_DAYS
        gap_entries.append(GapEntry(ref=ref, age_days=age, alerting=alerting))
    # Age-sort: oldest (largest age) first; unknowns last
    gap_entries.sort(key=lambda g: (g.age_days is None, -(g.age_days or 0)))
    plan.gap_list = gap_entries

    return plan


def _scaffold_path(scaffold) -> Path:
    """Extract or derive path from scaffold object."""
    return getattr(scaffold, "_path", Path(f"/srv/lapis/scout/sims/{scaffold.spec_id}.yaml"))


def _ref_age_days(ref: str) -> float | None:
    """Estimate age of a ref in days using mem.db if available; None if unknown."""
    try:
        import subprocess, json as _json
        result = subprocess.run(
            ["mem", "get", ref],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # mem output may include a created/updated timestamp
            # Try parsing as JSON to find a date field
            try:
                data = _json.loads(result.stdout)
                for key in ("created", "updated", "timestamp"):
                    ts = data.get(key)
                    if ts:
                        from datetime import datetime
                        import time
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        age = (time.time() - dt.timestamp()) / 86400
                        return max(0.0, age)
            except Exception:
                pass
    except Exception:
        pass
    return None


def render_plan(plan: SelectionPlan) -> str:
    """Render a SelectionPlan as human-readable text for the night-run log."""
    from datetime import datetime
    lines = ["=== Scout Selection Plan ==="]

    if plan.budget_deadline and plan.budget_deadline != float("inf"):
        try:
            dl = datetime.fromtimestamp(plan.budget_deadline).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, OverflowError):
            dl = str(plan.budget_deadline)
        max_str = f", max {plan.budget_max_units} units" if plan.budget_max_units is not None else ""
        lines.append(f"Budget: deadline {dl}{max_str}")
    elif plan.budget_max_units is not None:
        lines.append(f"Budget: max {plan.budget_max_units} units (no deadline)")

    lines.append(f"\nPriority lane ({len(plan.priority_lane)} sims):")
    for e in plan.priority_lane:
        lines.append(f"  [{e.completion*100:.0f}%→{e.target_pct*100:.0f}%, {e.units_needed} units] {e.summary}")

    lines.append(f"\nRelevance lane ({len(plan.relevance_lane)} sims):")
    for e in plan.relevance_lane:
        lines.append(f"  [{e.completion*100:.0f}%, {e.units_needed} units needed] {e.summary}")

    lines.append(f"\nParked ({len(plan.parked)} sims):")
    for p in plan.parked:
        lines.append(f"  [{p.reason}] {p.summary}")

    if plan.gap_list:
        lines.append(f"\nGap list ({len(plan.gap_list)} molten refs with no covering sim):")
        for g in plan.gap_list:
            alert = " [ALERT]" if g.alerting else ""
            age = f"{g.age_days:.0f}d" if g.age_days is not None else "?"
            lines.append(f"  {g.ref} ({age} old){alert}")

    return "\n".join(lines)
