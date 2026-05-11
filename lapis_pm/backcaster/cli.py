"""Backcaster CLI command handlers.

Wired into the main lapis-pm CLI as:
  lapis-pm backcaster <goal-file> [--axes ax1,ax2,...] [--corpus PATH]
                       [--scenarios <run-ids>] [--model opus|sonnet|qwen]
                       [--out /srv/lapis/backcaster/runs/<run-id>/]
"""
from __future__ import annotations

import sys


def cmd_backcaster(args) -> int:
    """Run the Backcaster pipeline on a goal-state file."""
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    goal_file = args.goal_file

    # Validate goal file exists early for clear exit code 1
    if not Path(goal_file).exists():
        print(f"error: goal file not found: {goal_file}", file=sys.stderr)
        return 1

    # Parse axes
    axes_raw: str | None = getattr(args, "axes", None)
    axes = [a.strip() for a in axes_raw.split(",")] if axes_raw else None

    # Parse corpus
    corpus_raw: str | None = getattr(args, "corpus", None)
    corpus_paths = [corpus_raw] if corpus_raw else None

    # Parse scenarios (v0 no-op)
    scenarios_raw: str | None = getattr(args, "scenarios", None)
    scenario_ids = [s.strip() for s in scenarios_raw.split(",")] if scenarios_raw else None

    # Spec default is opus; v0 interim default is qwen because the ClaudeQueue
    # synchronous-call surface for Anthropic-family models is not yet implemented
    # (see agents-core-claude-queue-sync-surface-v0). Passing --model opus will
    # raise a clear RuntimeError rather than silently downgrading.
    model: str = getattr(args, "model", "qwen") or "qwen"
    out_dir: str | None = getattr(args, "out", None)

    try:
        from .runner import run_backcaster
        run_dir = run_backcaster(
            goal_file=goal_file,
            axes=axes,
            corpus_paths=corpus_paths,
            scenario_ids=scenario_ids,
            model=model,
            out_dir=out_dir,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2

    print(f"backcaster: run complete")
    print(f"  run_dir: {run_dir}")
    print(f"  roadmap: {run_dir}/roadmap.md")
    print(f"  histogram: {run_dir}/histogram.yaml")
    return 0
