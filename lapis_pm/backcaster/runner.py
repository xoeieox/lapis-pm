"""Backcaster runner — orchestration + run-dir writes.

Wires together the four pipeline stages and writes all 7 run artifacts:
  goal.md, decomposition.yaml, gaps.yaml, components.yaml,
  roadmap.md, histogram.yaml, run.yaml

Run-dir slug derived from goal-file basename, kebab-cased, max 32 chars.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .compose import compose
from .decompose import decompose
from .derive import derive_components
from .gap_analyze import analyze_gaps
from .schema import AXES, Component, Gap, Precondition, RoadmapRun

log = logging.getLogger(__name__)

BACKCASTER_RUNS_ROOT = Path("/srv/lapis/backcaster/runs")


# ---------------------------------------------------------------------------
# Goal-state file parsing
# ---------------------------------------------------------------------------

def _parse_goal_file(path: Path) -> str:
    """Extract goal text from a goal-state file.

    Tolerates:
      - heading-prefixed (# Goal\n...)
      - front-matter (---\n...\n---\n...)
      - bare markdown body
    """
    text = path.read_text(encoding="utf-8").strip()

    # Strip YAML front-matter if present
    if text.startswith("---"):
        match = re.match(r"^---\n.*?\n---\n(.+)$", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # Malformed front-matter — use text as-is after the first ---
            text = re.sub(r"^---[^\n]*\n", "", text, count=1).strip()

    # Strip leading heading if present (# Goal, ## Goal, etc.)
    text = re.sub(r"^#{1,6}\s+[Gg]oal[^\n]*\n+", "", text).strip()

    return text


# ---------------------------------------------------------------------------
# Run-dir slug derivation
# ---------------------------------------------------------------------------

def _derive_slug(goal_path: Path, max_len: int = 32) -> str:
    """Derive a kebab-case slug from the goal file basename, max 32 chars.

    Truncates at word boundary.
    """
    stem = goal_path.stem  # filename without extension
    # Convert to kebab-case
    slug = stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")

    if len(slug) <= max_len:
        return slug

    # Truncate at word boundary
    truncated = slug[:max_len]
    last_dash = truncated.rfind("-")
    if last_dash > 0:
        truncated = truncated[:last_dash]
    return truncated


# ---------------------------------------------------------------------------
# YAML serialization helpers
# ---------------------------------------------------------------------------

def _sorted_yaml(obj: Any) -> str:
    """Dump obj as YAML with sorted keys for deterministic diffs."""
    return yaml.dump(obj, default_flow_style=False, sort_keys=True, allow_unicode=True)


# ---------------------------------------------------------------------------
# Main run entry point
# ---------------------------------------------------------------------------

def run_backcaster(
    goal_file: str | Path,
    axes: list[str] | None = None,
    corpus_paths: list[str] | None = None,
    scenario_ids: list[str] | None = None,
    model: str = "qwen",
    out_dir: str | Path | None = None,
    stub: bool = False,
) -> Path:
    """Run the full Backcaster pipeline and write all artifacts.

    Returns the run-dir Path.
    """
    if os.environ.get("BACKCASTER_STUB") == "1":
        stub = True

    goal_path = Path(goal_file)
    if not goal_path.exists():
        raise FileNotFoundError(f"Goal-state file not found: {goal_path}")

    goal_text = _parse_goal_file(goal_path)

    if axes is None:
        axes = list(AXES)

    if scenario_ids:
        log.info(
            "backcaster: --scenarios flag is a v0 no-op stub. "
            "Scenario ids received (%s) but not used.", scenario_ids
        )

    # Determine run-dir
    if out_dir is not None:
        run_dir = Path(out_dir)
    else:
        slug = _derive_slug(goal_path)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
        run_dir = BACKCASTER_RUNS_ROOT / f"{ts}-{slug}"

    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    log.info("backcaster: run_id=%s run_dir=%s", run_id, run_dir)

    t_start = time.monotonic()
    started_at_ts = datetime.now(timezone.utc)
    prompt_hashes: dict[str, str] = {}
    degraded_paths: list[str] = []

    # ------------------------------------------------------------------
    # Stage 1: Decompose
    # ------------------------------------------------------------------
    log.info("backcaster: stage 1 — decompose")
    t0 = time.monotonic()
    preconditions, decomp_hash = decompose(goal_text, axes=axes, model=model, stub=stub)
    prompt_hashes["decompose"] = decomp_hash
    log.info("backcaster: decompose produced %d preconditions (%.1fs)", len(preconditions), time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Stage 2: Gap analysis
    # ------------------------------------------------------------------
    log.info("backcaster: stage 2 — gap_analyze")
    t0 = time.monotonic()
    gaps, degraded, gap_hash = analyze_gaps(
        preconditions, corpus_paths=corpus_paths, model=model, stub=stub
    )
    degraded_paths.extend(degraded)
    prompt_hashes["gap_analyze"] = gap_hash
    log.info("backcaster: gap_analyze produced %d gaps, degraded=%s (%.1fs)", len(gaps), degraded, time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Stage 3: Component derivation
    # ------------------------------------------------------------------
    log.info("backcaster: stage 3 — derive")
    t0 = time.monotonic()
    components, derive_hash, fallback_count = derive_components(gaps, model=model, stub=stub)
    prompt_hashes["derive"] = derive_hash
    log.info("backcaster: derive produced %d components, fallbacks=%d (%.1fs)", len(components), fallback_count, time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Stage 4: Compose
    # ------------------------------------------------------------------
    log.info("backcaster: stage 4 — compose")
    roadmap_md, histogram, epistemic_caution, concentration_warning = compose(
        goal_text=goal_text,
        preconditions=preconditions,
        gaps=gaps,
        components=components,
        run_id=run_id,
    )

    t_total = time.monotonic() - t_start

    # ------------------------------------------------------------------
    # Build RoadmapRun metadata
    # ------------------------------------------------------------------
    roadmap_run = RoadmapRun(
        run_id=run_id,
        goal_file=str(goal_path),
        goal_text=goal_text,
        model=model,
        corpus_used=corpus_paths or [],
        scenario_ids=scenario_ids or [],
        timing={
            "total_seconds": round(t_total, 2),
            "started_at": started_at_ts.isoformat(),
        },
        prompt_hashes=prompt_hashes,
        epistemic_caution=epistemic_caution,
        concentration_warning=concentration_warning,
        degraded_paths=degraded_paths,
        derive_fallback_count=fallback_count,
    )

    # ------------------------------------------------------------------
    # Write all 7 artifacts
    # ------------------------------------------------------------------

    # 1. goal.md
    (run_dir / "goal.md").write_text(goal_text + "\n", encoding="utf-8")

    # 2. decomposition.yaml
    decomp_data = {
        "axes": sorted(axes),
        "preconditions": sorted(
            [p.to_dict() for p in preconditions],
            key=lambda x: x["id"],
        ),
    }
    (run_dir / "decomposition.yaml").write_text(_sorted_yaml(decomp_data), encoding="utf-8")

    # 3. gaps.yaml
    gaps_data = {
        "gaps": sorted(
            [g.to_dict() for g in gaps],
            key=lambda x: x["precondition_id"],
        )
    }
    (run_dir / "gaps.yaml").write_text(_sorted_yaml(gaps_data), encoding="utf-8")

    # 4. components.yaml
    comps_data = {
        "components": sorted(
            [c.to_dict() for c in components],
            key=lambda x: x["id"],
        )
    }
    (run_dir / "components.yaml").write_text(_sorted_yaml(comps_data), encoding="utf-8")

    # 5. roadmap.md
    (run_dir / "roadmap.md").write_text(roadmap_md + "\n", encoding="utf-8")

    # 6. histogram.yaml
    (run_dir / "histogram.yaml").write_text(_sorted_yaml(histogram.to_dict()), encoding="utf-8")

    # 7. run.yaml
    (run_dir / "run.yaml").write_text(_sorted_yaml(roadmap_run.to_dict()), encoding="utf-8")

    log.info("backcaster: all 7 artifacts written to %s", run_dir)
    return run_dir
