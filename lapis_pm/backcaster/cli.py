"""Backcaster CLI command handlers.

Wired into the main lapis-pm CLI as:
  lapis-pm backcaster <goal-file> [--axes ax1,ax2,...] [--corpus PATH]
                       [--scenarios <run-ids>] [--model gravitywell|qwen|sonnet|opus]
                       [--out /srv/lapis/backcaster/runs/<run-id>/]
  lapis-pm backcaster-quest <run-id> [--gap <id>...] [--all-unsourced] [--escalate]
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

    # Spec default is opus; current default is gravitywell (owned 122B local).
    # Fallback to gravitywell if for any reason the argparse default is not applied.
    model: str = getattr(args, "model", "gravitywell") or "gravitywell"
    out_dir: str | None = getattr(args, "out", None)
    allow_degraded: bool = bool(getattr(args, "allow_degraded", False))

    try:
        from .runner import run_backcaster
        run_dir = run_backcaster(
            goal_file=goal_file,
            axes=axes,
            corpus_paths=corpus_paths,
            scenario_ids=scenario_ids,
            model=model,
            out_dir=out_dir,
            allow_degraded=allow_degraded,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
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


def cmd_backcaster_quest(args) -> int:
    """Source unsourced gaps in a completed Backcaster run via Dowser."""
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_id: str = args.run_id
    gap_ids: list[str] | None = getattr(args, "gap", None) or None
    all_unsourced: bool = bool(getattr(args, "all_unsourced", False))
    escalate: bool = bool(getattr(args, "escalate", False))

    if not gap_ids and not all_unsourced:
        print("error: must specify --gap <id>... or --all-unsourced", file=sys.stderr)
        return 1

    # Validate run dir
    try:
        from agents_core.room_paths import room_path
        run_dir = Path(room_path("backcaster.runs")) / run_id
    except Exception as exc:
        print(f"error: could not resolve run dir: {exc}", file=sys.stderr)
        return 1

    if not run_dir.exists():
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 1

    try:
        from .quest_leg import quest_source_run
        summary = quest_source_run(
            run_id,
            gap_ids=gap_ids,
            all_unsourced=all_unsourced,
            escalate=escalate,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2

    print(f"backcaster-quest: run complete")
    print(f"  run_id:          {run_id}")
    print(f"  gaps_targeted:   {summary['gaps_targeted']}")
    print(f"  gaps_sourced:    {summary['gaps_sourced']}")
    print(f"  gaps_honest_null:{summary['gaps_honest_null']}")
    print(f"  citations_added: {summary['citations_added']}")
    print(f"  caution_before:  {summary['caution_before']}")
    print(f"  caution_after:   {summary['caution_after']}")
    print(f"  retries_used:    {summary['retries_used']}")
    nulls = summary.get("null_outcomes", {})
    if any(nulls.values()):
        print(f"  null_outcomes:")
        for k, v in nulls.items():
            if v:
                print(f"    {k}: {v}")
    note = summary.get("note")
    if note:
        print(f"  note: {note}")
    escalate_note = summary.get("escalate_note")
    if escalate_note:
        print(f"  escalate_note: {escalate_note}")
    return 0
