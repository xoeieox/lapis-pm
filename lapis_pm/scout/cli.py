"""Scout CLI command handlers.

Wired into the main lapis-pm CLI as:
  lapis-pm scout simulate <scaffold_path> [--cell CELL_ID] [--runs N]
  lapis-pm scout digest <spec_id>
"""
from __future__ import annotations

import sys


def cmd_scout_simulate(args) -> int:
    """Run a scaffold simulation (full matrix or a specific cell)."""
    from .runner import simulate

    scaffold_path = args.scaffold
    runs: int | None = getattr(args, "runs", None)
    cell_arg: str | None = getattr(args, "cell", None)
    cells = [cell_arg] if cell_arg else None

    try:
        written = simulate(scaffold_path, runs_per_cell=runs, cells=cells)
        for p in written:
            print(f"wrote: {p}")
        print(f"done: {len(written)} trace(s) written")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_scout_digest(args) -> int:
    """Digest all traces for a spec into a failure-mode map."""
    from .digest import digest

    spec_id = args.spec_id

    try:
        out_path = digest(spec_id)
        print(f"wrote: {out_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_scout(args) -> int:
    """Dispatcher for `lapis-pm scout <subcommand>`."""
    sub = getattr(args, "scout_sub", None)
    if sub == "simulate":
        return cmd_scout_simulate(args)
    if sub == "digest":
        return cmd_scout_digest(args)
    print("usage: lapis-pm scout {simulate,digest}", file=sys.stderr)
    return 2
