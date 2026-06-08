"""Scout CLI command handlers.

Wired into the main lapis-pm CLI as:
  lapis-pm scout simulate <scaffold_path> [--cell CELL_ID] [--runs N]
  lapis-pm scout digest <spec_id>
  lapis-pm scout night run [--until HHMM] [--once] [--sims-dir DIR]
                           [--profile-override SPEC_ID:PROFILE]
  lapis-pm scout night status [--log-root DIR] [--json]
  lapis-pm scout night quarantine-clear <spec_id> [--log-root DIR]
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


def cmd_scout_night_run(args) -> int:
    """Start a night run."""
    import logging
    from pathlib import Path
    from .night_queue import run_night
    from .budget import NightBudget

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sims_dir_arg: str | None = getattr(args, "sims_dir", None)
    sims_dir = Path(sims_dir_arg) if sims_dir_arg else Path("/srv/lapis/scout/sims")

    once: bool = getattr(args, "once", False)
    until_str: str | None = getattr(args, "until", None)
    deadline_in_str: str | None = getattr(args, "deadline_in", None)
    max_units_arg: int | None = getattr(args, "max_units", None)

    selected_arg: str | None = getattr(args, "selected", None)
    selected_path = Path(selected_arg) if selected_arg else None

    log_root_arg: str | None = getattr(args, "log_root", None)
    log_root = Path(log_root_arg) if log_root_arg else None

    profiles_override: dict[str, str] | None = None
    overrides_raw: list[str] = getattr(args, "profile_override", []) or []
    if overrides_raw:
        profiles_override = {}
        for item in overrides_raw:
            if ":" not in item:
                print(f"error: --profile-override must be SPEC_ID:PROFILE, got {item!r}", file=sys.stderr)
                return 2
            spec_id, profile = item.split(":", 1)
            profiles_override[spec_id.strip()] = profile.strip()

    # Build budget: --deadline-in > --until (today-only, no roll) > default 4h
    # --once mode bypasses the budget deadline (drains worklist; budget still applies
    # for max_units if set)
    budget: NightBudget | None = None
    if not once:
        try:
            budget = NightBudget.from_cli(
                until_hhmm=until_str,
                deadline_in=deadline_in_str,
                max_units=max_units_arg,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif max_units_arg is not None:
        # --once with --max-units: respect the ceiling
        budget = NightBudget(deadline=float("inf"), max_units=max_units_arg)

    try:
        result = run_night(
            sims_dir=sims_dir,
            once=once,
            log_root=log_root,
            profiles_override=profiles_override,
            budget=budget,
            selected_path=selected_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Night run complete.")
    print(f"  Manifest: {result.manifest_path}")
    print(f"  Total: {result.total_units} units | OK: {result.completed_ok} | Err: {result.errored}")
    print(f"  Quarantine-skip: {result.skipped_quarantine} | Health-skip: {result.skipped_health}")
    if result.aborted:
        print(f"  ABORTED: {result.abort_reason}", file=sys.stderr)
        return 1
    return 0


def cmd_scout_night_status(args) -> int:
    """Show status of the most recent (or specified) night run."""
    from pathlib import Path
    from .night_queue import read_status

    log_root_arg: str | None = getattr(args, "log_root", None)
    as_json: bool = getattr(args, "json", False)

    if log_root_arg:
        log_root = Path(log_root_arg)
    else:
        import glob
        dirs = sorted(glob.glob("/tmp/scout-night-*"), reverse=True)
        if not dirs:
            print("No night-run log roots found under /tmp/scout-night-*", file=sys.stderr)
            return 1
        log_root = Path(dirs[0])

    try:
        text = read_status(log_root, as_json=as_json)
        print(text)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_scout_night_quarantine_clear(args) -> int:
    """Clear a scaffold's quarantine entry."""
    from pathlib import Path
    from .night_queue import Quarantine

    spec_id: str = args.spec_id
    log_root_arg: str | None = getattr(args, "log_root", None)

    if log_root_arg:
        log_root = Path(log_root_arg)
    else:
        import glob
        dirs = sorted(glob.glob("/tmp/scout-night-*"), reverse=True)
        if not dirs:
            print("No night-run log roots found under /tmp/scout-night-*", file=sys.stderr)
            return 1
        log_root = Path(dirs[0])

    quarantine = Quarantine(log_root / "quarantine.json")
    removed = quarantine.clear(spec_id)
    if removed:
        print(f"quarantine cleared: {spec_id}")
    else:
        print(f"not quarantined (no-op): {spec_id}")
    return 0


def cmd_scout_night(args) -> int:
    """Dispatcher for `lapis-pm scout night <subcommand>`."""
    sub = getattr(args, "night_sub", None)
    if sub == "run":
        return cmd_scout_night_run(args)
    if sub == "status":
        return cmd_scout_night_status(args)
    if sub == "quarantine-clear":
        return cmd_scout_night_quarantine_clear(args)
    print("usage: lapis-pm scout night {run,status,quarantine-clear}", file=sys.stderr)
    return 2


def cmd_scout(args) -> int:
    """Dispatcher for `lapis-pm scout <subcommand>`."""
    sub = getattr(args, "scout_sub", None)
    if sub == "simulate":
        return cmd_scout_simulate(args)
    if sub == "digest":
        return cmd_scout_digest(args)
    if sub == "night":
        return cmd_scout_night(args)
    print("usage: lapis-pm scout {simulate,digest,night}", file=sys.stderr)
    return 2
