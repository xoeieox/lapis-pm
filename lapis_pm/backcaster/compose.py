"""Backcaster Stage 4 — Composer.

Synthesizes preconditions + gaps + components into:
  - roadmap.md  (human-readable, per-axis sections + histogram)
  - histogram   (category distribution, financial subtypes, research.fallback breakout)
  - run.yaml    (run metadata including epistemic_caution + concentration_warning)

Also computes:
  - epistemic_caution: low | medium | high (from unsourced-ratio across gaps)
  - concentration_warning: list of "<category>-dominant" strings
"""
from __future__ import annotations

from typing import Any

from .schema import (
    AXES,
    CATEGORIES,
    FINANCIAL_SUBTYPES,
    Component,
    Gap,
    Histogram,
    Precondition,
    RoadmapRun,
)


# ---------------------------------------------------------------------------
# Epistemic caution
# ---------------------------------------------------------------------------

def compute_epistemic_caution(gaps: list[Gap]) -> str:
    """Compute epistemic_caution from unsourced-ratio across gaps.

    low    < 20% unsourced
    medium 20-50% unsourced
    high   > 50% unsourced
    """
    if not gaps:
        return "high"
    unsourced_count = sum(1 for g in gaps if g.unsourced)
    ratio = unsourced_count / len(gaps)
    if ratio < 0.20:
        return "low"
    elif ratio <= 0.50:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Concentration warning
# ---------------------------------------------------------------------------

def compute_concentration_warning(components: list[Component]) -> list[str]:
    """Compute concentration_warning list.

    Evaluated at two levels:
      1. Top-level categories — > 70% triggers "<category>-dominant"
      2. Financial subtypes   — > 70% triggers "financial.<subtype>-dominant"

    A roll-up financial total > 70% triggers "financial-dominant" PLUS any
    subtype also > 70%.
    """
    if not components:
        return []

    total = len(components)
    warnings: list[str] = []

    # Top-level category counts
    cat_counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    subtype_counts: dict[str, int] = {st: 0 for st in FINANCIAL_SUBTYPES}

    for comp in components:
        cat_counts[comp.category] = cat_counts.get(comp.category, 0) + 1
        if comp.category == "financial" and comp.subtype in FINANCIAL_SUBTYPES:
            subtype_counts[comp.subtype] = subtype_counts.get(comp.subtype, 0) + 1

    for cat, count in cat_counts.items():
        if count / total > 0.70:
            warnings.append(f"{cat}-dominant")

    # Financial subtype check
    for st, count in subtype_counts.items():
        if count / total > 0.70:
            warnings.append(f"financial.{st}-dominant")

    return sorted(warnings)


# ---------------------------------------------------------------------------
# Markdown roadmap renderer
# ---------------------------------------------------------------------------

def _render_roadmap(
    goal_text: str,
    preconditions: list[Precondition],
    gaps: list[Gap],
    components: list[Component],
    histogram: Histogram,
    run_id: str,
) -> str:
    """Render the human-readable roadmap markdown."""
    lines: list[str] = []

    lines.append(f"# Backcaster Roadmap")
    lines.append(f"\n**Run:** `{run_id}`\n")
    lines.append("## Goal-State\n")
    lines.append(goal_text.strip())
    lines.append("")

    # Category histogram (load-bearing for software-bias inspection)
    lines.append("## Category Histogram\n")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for cat in CATEGORIES:
        count = histogram.counts.get(cat, 0)
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    # Financial subtype breakout
    financial_total = histogram.counts.get("financial", 0)
    if financial_total > 0:
        lines.append("**Financial subtypes:**")
        for st in FINANCIAL_SUBTYPES:
            count = histogram.counts.get(f"financial.{st}", 0)
            lines.append(f"- `financial.{st}`: {count}")
        lines.append("")

    # Research fallback note
    fallback_count = histogram.counts.get("research.fallback", 0)
    if fallback_count > 0:
        lines.append(f"> **Note:** {fallback_count} component(s) landed via fallback (category=research). See `run.yaml:derive_fallback_count`.\n")

    # Per-axis sections
    gap_by_pid = {g.precondition_id: g for g in gaps}
    comps_by_pid: dict[str, list[Component]] = {}
    for comp in components:
        comps_by_pid.setdefault(comp.gap_precondition_id, []).append(comp)

    for axis in AXES:
        axis_precs = [p for p in preconditions if p.axis == axis]
        if not axis_precs:
            continue

        lines.append(f"## {axis.replace('-', ' ').title()} Axis\n")

        for prec in axis_precs:
            lines.append(f"### Precondition: {prec.statement}\n")

            gap = gap_by_pid.get(prec.id)
            if gap:
                lines.append("**Gap analysis:**")
                lines.append(f"- *Exists:* {gap.what_exists}")
                lines.append(f"- *Missing:* {gap.what_missing}")
                if gap.what_miswired and gap.what_miswired != "nothing identified":
                    lines.append(f"- *Miswired:* {gap.what_miswired}")
                if gap.unsourced:
                    lines.append("\n> *Unsourced: no citation-grounded evidence available.*")
                lines.append("")

            comps = comps_by_pid.get(prec.id, [])
            if comps:
                lines.append("**Required components:**")
                lines.append("")
                for comp in comps:
                    fallback_tag = " *(fallback)*" if comp.fallback else ""
                    subtype_tag = f" [{comp.subtype}]" if comp.category == "financial" and comp.subtype else ""
                    lines.append(
                        f"- **[{comp.category}{subtype_tag}]**{fallback_tag} "
                        f"{comp.description}  "
                    )
                    lines.append(
                        f"  *Effort: {comp.effort_estimate} | Reversibility: {comp.reversibility}*"
                    )
                    if comp.dependencies:
                        lines.append(f"  *Depends on: {', '.join(comp.dependencies)}*")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compose entry point
# ---------------------------------------------------------------------------

def compose(
    goal_text: str,
    preconditions: list[Precondition],
    gaps: list[Gap],
    components: list[Component],
    run_id: str,
) -> tuple[str, Histogram, str, list[str]]:
    """Compose roadmap markdown and compute histogram + metadata.

    Returns (roadmap_md, histogram, epistemic_caution, concentration_warning).
    """
    histogram = Histogram.from_components(components)
    epistemic_caution = compute_epistemic_caution(gaps)
    concentration_warning = compute_concentration_warning(components)

    roadmap_md = _render_roadmap(
        goal_text=goal_text,
        preconditions=preconditions,
        gaps=gaps,
        components=components,
        histogram=histogram,
        run_id=run_id,
    )

    return roadmap_md, histogram, epistemic_caution, concentration_warning
