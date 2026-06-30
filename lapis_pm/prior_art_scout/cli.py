"""CLI command handlers for `lapis-pm prior-art-scout`."""
from __future__ import annotations

import sys


def cmd_prior_art_scout_run(args) -> int:
    """Run the prior-art scout."""
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    escalate: bool = bool(getattr(args, "escalate", False))
    dry_run: bool = bool(getattr(args, "dry_run", False))
    wall_budget: int | None = getattr(args, "wall_budget", None)

    try:
        from .runner import scout_run
        run = scout_run(escalate=escalate, dry_run=dry_run, wall_budget=wall_budget)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"prior-art-scout: run complete")
    print(f"  run_date:         {run.run_date}")
    print(f"  items_targeted:   {run.items_targeted}")
    print(f"  items_sourced:    {run.items_sourced}")
    print(f"  items_honest_null:{run.items_honest_null}")
    print(f"  caution:          {run.caution}")
    print(f"  model_policy:     {run.model_policy}")
    if run.saturated_namespaces:
        print(f"  SATURATED:        {', '.join(run.saturated_namespaces)} (snapshot cap hit)")
    if dry_run:
        print("  (dry-run: no LLM calls made)")
    return 0


def cmd_prior_art_scout_status(args) -> int:
    """Print last run summary from run.yaml."""
    import logging
    from pathlib import Path

    import yaml

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    runs_root = Path("/srv/lapis/prior-art-scout/runs")
    if not runs_root.exists():
        print("no runs found", file=sys.stderr)
        return 1

    run_dirs = sorted(runs_root.iterdir(), key=lambda p: p.name, reverse=True)
    for d in run_dirs:
        run_yaml = d / "run.yaml"
        if run_yaml.exists():
            data = yaml.safe_load(run_yaml.read_text()) or {}
            print(f"Last run: {d.name}")
            for k, v in sorted(data.items()):
                print(f"  {k}: {v}")
            return 0

    print("no run.yaml found in any run dir", file=sys.stderr)
    return 1


def cmd_prior_art_scout_reset_hopeless(args) -> int:
    """Clear the known-hopeless sidecar for a given key."""
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    key: str = args.key
    try:
        from .sidecar import load_sidecar, sidecar_path
        path = sidecar_path(key)
        if not path.exists():
            print(f"no sidecar found for key: {key}")
            return 0
        sidecar = load_sidecar(key) or {}
        if sidecar.get("verdict") != "insufficient-sources":
            print(f"sidecar verdict is {sidecar.get('verdict')!r}, not 'insufficient-sources'; no-op")
            return 0
        path.unlink()
        print(f"cleared known-hopeless sidecar for {key}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
