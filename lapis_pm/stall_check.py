"""Dedicated tick-coverage + directive-outcome stall checker
(attestation-contract-v0 leg 2, D6a).

A DEDICATED LIGHTWEIGHT CHECKER UNIT, NOT a detector embedded in tick_all
(rev-1 placement was structurally wrong: the killed pass is the thing being
detected — an end-of-tick_all detector never fires for the starvation it
detects, and a hung pass blocks it indefinitely).

Run as `python3 -m lapis_pm.cli stall-check` on an independent 10-minute
timer (systemd/lapis-pm-stall-check.{service,timer}). The same checks also
run from the tick pass (pm_core._check_tick_stalls /
pm_core._check_directive_stalls) so a healthy tick pass pages within one
pass; this unit catches the case where the tick pass itself is dead or
hung.

Contract (spec D6):
  * stdlib + mem reads only, NO LLM calls (the checker drives no LLM call
    path and imports none — asserted by tests), completes in seconds;
  * for each pm-bound ACTIVE target: page HIGH once per episode when the
    pm/cursor/<tid> watermark is older than TICK_COVERAGE_STALL_S;
  * PAUSED targets are EXCLUDED by design (a paused target is intentionally
    not being ticked; the first tick after un-pause advances its cursor);
  * REV 2 pins: (i) a target with no cursor record uses its bound ts (the
    spec:bound comment ts) as the watermark; (ii) while the Forgejo
    consecutive-fail counter is at or above the page threshold the checker
    skips paging (the existing Forgejo 3-strike page owns that episode —
    I5, no duplicate page); (iii) the page carries the target's last
    decision tag (pm/last-decision/<tid>, line omitted when absent);
  * I5 page hygiene: one HIGH page per target per episode (dedup mem key
    pm/tick-stall/<tid> stamped with the last-paged ts; a cursor advance
    clears it);
  * the unit file MUST carry no EnvironmentFile lines (the stdlib+mem-only
    checker reads no credentials — it must not inherit conductor.env /
    phala.env from lapis-pm.service).
"""

from __future__ import annotations

import sys

from zoneinfo import ZoneInfo

from . import pm_core

PACIFIC = ZoneInfo("America/Los_Angeles")


def run_checks() -> int:
    """Run the tick-coverage + directive-outcome stall checks.

    Returns the number of pages emitted (0 = nothing to page). Never
    raises — a checker failure is logged, not paged (the checker's own
    death is caught by the next healthy tick pass's detectors).
    """
    from datetime import datetime
    from agents_core.targets import TargetStore

    store = TargetStore()
    now = datetime.now(PACIFIC)
    pages = 0
    try:
        pages += pm_core._check_tick_stalls(store, now)
    except Exception as e:
        print(f"[stall-check] tick-coverage check failed: {e}", file=sys.stderr)
    try:
        pages += pm_core._check_directive_stalls(store, now)
    except Exception as e:
        print(f"[stall-check] directive-outcome check failed: {e}", file=sys.stderr)
    if pages:
        print(f"[stall-check] pages={pages}", flush=True)
    else:
        print("[stall-check] no pages", flush=True)
    return pages


def main() -> int:
    run_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
