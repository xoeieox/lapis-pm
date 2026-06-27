"""Scout CLI command handlers.

Wired into the main lapis-pm CLI as:
  lapis-pm scout simulate <scaffold_path> [--cell CELL_ID] [--runs N]
  lapis-pm scout digest <spec_id>
  lapis-pm scout refiner [--observe-only] [--spec ID] [--cos-threshold F]
  lapis-pm scout night run [--until HHMM] [--once] [--sims-dir DIR]
                           [--profile-override SPEC_ID:PROFILE]
  lapis-pm scout night status [--log-root DIR] [--json]
  lapis-pm scout night quarantine-clear <spec_id> [--log-root DIR]
"""
from __future__ import annotations

import re
import sys

from agents_core.room_paths import room_path


def cmd_scout_simulate(args) -> int:
    """Run a scaffold simulation (full matrix or a specific cell)."""
    from .runner import simulate, _ScoutJsonAdapter, _ScoutGravityWellAdapter

    scaffold_path = args.scaffold
    runs: int | None = getattr(args, "runs", None)
    cell_arg: str | None = getattr(args, "cell", None)
    cells = [cell_arg] if cell_arg else None
    model: str = getattr(args, "model", "gravitywell")

    llm = None
    if model == "qwen":
        llm = _ScoutJsonAdapter(max_tokens=2048)
    elif model == "gravitywell":
        llm = _ScoutGravityWellAdapter(timeout=300)

    try:
        written = simulate(scaffold_path, runs_per_cell=runs, cells=cells, llm=llm)
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
    sims_dir = Path(sims_dir_arg) if sims_dir_arg else room_path('scout.sims')

    once: bool = getattr(args, "once", False)
    until_str: str | None = getattr(args, "until", None)
    deadline_in_str: str | None = getattr(args, "deadline_in", None)
    max_units_arg: int | None = getattr(args, "max_units", None)

    selected_arg: str | None = getattr(args, "selected", None)
    selected_path = Path(selected_arg) if selected_arg else None

    log_root_arg: str | None = getattr(args, "log_root", None)
    log_root = Path(log_root_arg) if log_root_arg else None

    model: str = getattr(args, "model", "gravitywell")

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
            model=model,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Night run complete.")
    print(f"  Manifest: {result.manifest_path}")
    print(f"  Total: {result.total_units} units | OK: {result.completed_ok} | Err: {result.errored}")
    print(f"  Quarantine-skip: {result.skipped_quarantine} | Health-skip: {result.skipped_health} | GW-skip: {result.skipped_gw}")
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


def cmd_scout_refiner(args) -> int:
    """Run observe→refine loop over active Scout scaffolds + Backcaster goals.

    Full pass: refiner_observe all active specs + backcaster_observe all goals,
    then refiner_refine (GW-122B), render salience maps, write dated output,
    inject summary into Active Work.md.

    --observe-only: pure-CPU, no GW call.
    --spec <id>: restrict to one scaffold spec.
    --cos-threshold <f>: override default 0.80 Scout clustering threshold.
    """
    import json as _json
    import logging
    from datetime import datetime, timezone
    from pathlib import Path

    from .refiner import (
        REFINER_OUTPUT_ROOT,
        SCOUT_TRACES_ROOT,
        BackcasterObserveResult,
        backcaster_observe,
        refiner_observe,
        refiner_refine,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    observe_only: bool = getattr(args, "observe_only", False)
    spec_filter: str | None = getattr(args, "spec", None)
    cos_threshold: float = float(getattr(args, "cos_threshold", 0.80))

    # Discover active scaffold specs that have trace data
    sims_dir = room_path('scout.sims')
    if spec_filter:
        spec_ids = [spec_filter]
    else:
        if sims_dir.exists():
            spec_ids = [
                f.stem for f in sorted(sims_dir.glob("*.yaml"))
                if (SCOUT_TRACES_ROOT / f.stem).exists()
            ]
        else:
            spec_ids = [
                d.name for d in sorted(SCOUT_TRACES_ROOT.iterdir())
                if d.is_dir()
            ] if SCOUT_TRACES_ROOT.exists() else []

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dated_output_root = REFINER_OUTPUT_ROOT / date_str
    dated_output_root.mkdir(parents=True, exist_ok=True)

    # --- Observe ---
    observe_results = []
    for sid in spec_ids:
        try:
            obs = refiner_observe(sid, cos_threshold=cos_threshold)
            observe_results.append(obs)
            collapse = (
                f", collapse {obs.raw_unique_count}→{len(obs.clusters)} "
                f"({obs.raw_unique_count / max(len(obs.clusters), 1):.1f}×)"
                if obs.raw_unique_count else ""
            )
            print(
                f"observe {sid}: {obs.valid_traces}/{obs.total_traces} valid, "
                f"{len(obs.clusters)} clusters{collapse}"
            )
            if obs.systemic_failure_nights:
                print(f"  systemic nights: {obs.systemic_failure_nights}")
            if obs.thin_nights:
                print(f"  thin nights (not systemic): {obs.thin_nights}")
        except Exception as exc:
            print(f"observe error {sid}: {exc}", file=sys.stderr)

    bc_result: BackcasterObserveResult | None = None
    try:
        bc_result = backcaster_observe()
        print(
            f"backcaster observe: {bc_result.valid_runs}/{bc_result.total_runs} valid, "
            f"{bc_result.gap_raw_count} raw gaps → {len(bc_result.gap_clusters)} clusters "
            f"(ratio={bc_result.gap_collapse_ratio:.1f}×, "
            f"cos≥{bc_result.gap_threshold_used})"
        )
        if bc_result.gap_sanity_warning:
            print(f"  WARNING: {bc_result.gap_sanity_warning}", file=sys.stderr)
        if bc_result.systemic_failure_nights:
            print(f"  systemic nights: {bc_result.systemic_failure_nights}")
    except Exception as exc:
        print(f"backcaster observe error: {exc}", file=sys.stderr)

    if not observe_results and bc_result is None:
        print("no observe results; nothing to refine", file=sys.stderr)
        return 1

    # --- Refine (one call per spec + one for backcaster) ---
    all_refine_results = []
    for obs in observe_results:
        try:
            results = refiner_refine(
                obs, None,
                output_root=dated_output_root,
                observe_only=observe_only,
            )
            all_refine_results.extend(results)
            for r in results:
                status = "observe-only (no GW call)" if r.gw_skipped else f"{len(r.proposals)} proposals"
                print(f"refine {r.spec_id}: {status}")
                if r.salience_map_path:
                    print(f"  salience: {r.salience_map_path}")
        except Exception as exc:
            print(f"refine error {obs.spec_id}: {exc}", file=sys.stderr)

    if bc_result is not None:
        try:
            bc_refine = refiner_refine(
                None, bc_result,
                output_root=dated_output_root,
                observe_only=observe_only,
            )
            all_refine_results.extend(bc_refine)
            for r in bc_refine:
                status = "observe-only (no GW call)" if r.gw_skipped else f"{len(r.proposals)} proposals"
                print(f"refine backcaster: {status}")
                if r.salience_map_path:
                    print(f"  salience: {r.salience_map_path}")
        except Exception as exc:
            print(f"backcaster refine error: {exc}", file=sys.stderr)

    # --- Update latest.md pointer (A3) ---
    latest_path = REFINER_OUTPUT_ROOT / "latest.md"
    n_proposals = sum(len(r.proposals) for r in all_refine_results)
    n_retire = sum(
        1 for r in all_refine_results
        for p in r.proposals if p.action == "retire"
    )
    n_active = len(observe_results)
    n_saturated = sum(
        1 for obs in observe_results
        if obs.saturation and obs.saturation.verdict == "true_exhaustion"
    )
    latest_content = (
        f"# Scout Refiner — Latest Run\n\n"
        f"Date: {date_str}\n"
        f"Scaffolds observed: {n_active} active, {n_saturated} saturated\n"
        f"Proposals: {n_proposals} total, {n_retire} retire candidates\n"
        f"Run output: {dated_output_root}/\n\n"
        f"See salience maps in the dated directory above.\n"
    )
    latest_path.write_text(latest_content)
    print(f"latest.md: {latest_path}")

    # --- Inject summary into Active Work.md (A3 forced visibility) ---
    vault_active_work = Path("/srv/git/inertia-vault-working/Active Work.md")
    if vault_active_work.exists():
        _inject_refiner_digest_line(
            vault_active_work,
            date_str=date_str,
            n_active=n_active,
            n_saturated=n_saturated,
            n_proposals=n_proposals,
            n_retire=n_retire,
            latest_path=str(latest_path),
        )
        print(f"digest line injected: {vault_active_work}")
    else:
        print(f"Active Work.md not found at {vault_active_work} — skipping digest injection", file=sys.stderr)

    # Emit LapisToolReturn-style JSON summary
    summary = {
        "agent_id": "lapis-refiner/observe+refine",
        "date": date_str,
        "specs_observed": [r.spec_id for r in observe_results],
        "n_proposals": n_proposals,
        "n_retire_candidates": n_retire,
        "salience_maps": [r.salience_map_path for r in all_refine_results if r.salience_map_path],
        "gw_skipped": any(r.gw_skipped for r in all_refine_results),
        "observe_only": observe_only,
        "latest_md": str(latest_path),
    }
    print("\n--- LapisToolReturn ---")
    print(_json.dumps(summary, indent=2))
    return 0


def _inject_refiner_digest_line(
    vault_path: Path,
    *,
    date_str: str,
    n_active: int,
    n_saturated: int,
    n_proposals: int,
    n_retire: int,
    latest_path: str,
) -> None:
    """Upsert a high-contrast Scout Refiner summary section in Active Work.md."""
    try:
        content = vault_path.read_text(encoding="utf-8")
    except OSError:
        return

    summary_line = (
        f"**Scout Refiner — {date_str}:** "
        f"{n_active} scaffolds active ({n_saturated} saturated), "
        f"**{n_proposals} proposals waiting** ({n_retire} retire candidates). "
        f"→ `{latest_path}`"
    )

    section_header = "### Scout Refiner — Last Run"
    new_section = f"{section_header}\n\n{summary_line}\n"

    if section_header in content:
        # Replace existing section: from the header to the next ### or ##
        pattern = re.compile(
            r"(### Scout Refiner — Last Run\n).*?(?=\n###|\n##|\Z)",
            re.DOTALL,
        )
        updated = pattern.sub(new_section, content)
    else:
        # Insert after "## Lapis PM — Live Status" header if present, else append
        insert_marker = "## Lapis PM — Live Status"
        if insert_marker in content:
            insert_pos = content.index(insert_marker) + len(insert_marker)
            # Find end of that line
            nl = content.find("\n", insert_pos)
            if nl == -1:
                nl = len(content)
            updated = content[:nl + 1] + "\n" + new_section + "\n" + content[nl + 1:]
        else:
            updated = content.rstrip() + "\n\n" + new_section

    try:
        vault_path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        import sys as _sys
        print(f"warning: could not write Active Work.md: {exc}", file=_sys.stderr)


def cmd_scout(args) -> int:
    """Dispatcher for `lapis-pm scout <subcommand>`."""
    sub = getattr(args, "scout_sub", None)
    if sub == "simulate":
        return cmd_scout_simulate(args)
    if sub == "digest":
        return cmd_scout_digest(args)
    if sub == "refiner":
        return cmd_scout_refiner(args)
    if sub == "night":
        return cmd_scout_night(args)
    print("usage: lapis-pm scout {simulate,digest,refiner,night}", file=sys.stderr)
    return 2
