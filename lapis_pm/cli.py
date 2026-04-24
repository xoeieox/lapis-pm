#!/usr/bin/env python3
"""lapis-pm CLI.

Commands:
    bind <target_id> --spec-from PATH|- --repo REPO [--authority advisory|auto]
               [--create --title TITLE [--description TEXT] [--urgency medium]
                [--tag TAG ...] [--product NAME]] [--force]
    unbind <target_id>
    tick [--target ID | --all] [--force-brief] [--force-dispatch AGENT:INTENT]
    status [target_id] [--explain]
    pause <target_id> [--reason TEXT]
    resume <target_id>
    list
    land <target_id> [--dry-run]
    review-gate {status,resume}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents_core.targets import TargetStore

# Package imports work because the CLI is launched via `python -m lapis_pm.cli`.
from . import episodic, shaper, brief, pm_core, land

# Four-stage PM lifecycle template used for all lapis-pm-monitored targets.
PM_LIFECYCLE_STAGES = [
    {"name": "Bind to lapis-pm daemon with spec + advisory authority", "status": "active"},
    {"name": "Shaped agent dispatch produces PR on lapis/<target_id>/<slug>", "status": "pending"},
    {"name": "PM perceives PR, digest screen classifies, brief posted", "status": "pending"},
    {"name": "Accept or reject based on brief; land thread via `lapis-pm land`", "status": "pending"},
]


def _read_spec(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    return Path(source).read_text()


def _derive_tags(repo: str, target_id: str, extra_tags: list[str]) -> list[str]:
    """Build tag list: repo name first, then extra --tag flags. Deduped, insertion order."""
    seen: dict[str, None] = {}
    for t in ([repo] if repo else []) + list(extra_tags):
        if t and t not in seen:
            seen[t] = None
    return list(seen)


def cmd_bind(args) -> int:
    store = TargetStore()

    if args.create:
        if not args.title:
            print("ERROR: --title is required with --create", file=sys.stderr)
            return 2

        target = store.get(args.target_id)
        if target is not None and not args.force:
            print(
                f"ERROR: target {args.target_id!r} already exists. "
                "Use --force to replace spec binding.",
                file=sys.stderr,
            )
            return 2

        if target is None:
            tags = _derive_tags(args.repo, args.target_id, args.tag)
            target = store.create(
                args.target_id,
                title=args.title,
                urgency=args.urgency,
                work_mode=args.work_mode,
                description=args.description,
                stages=list(PM_LIFECYCLE_STAGES),
                category="active-work",
            )
            if tags:
                target.data["tags"] = tags
            if args.product:
                target.data["product"] = args.product
            target.save()
    else:
        target = store.get(args.target_id)
        if target is None:
            print(f"ERROR: target not found: {args.target_id}", file=sys.stderr)
            return 2

    existing_spec = episodic.spec(args.target_id)
    if existing_spec and not args.force:
        print("ERROR: target already has a spec bound. Use --force to replace.",
              file=sys.stderr)
        return 2

    spec_body = _read_spec(args.spec_from).strip()
    if not spec_body:
        print("ERROR: empty spec body", file=sys.stderr)
        return 2

    target.bind_pm(repo=args.repo, authority=args.authority)
    target.save()

    episodic.write_spec(args.target_id, spec_body)
    pm_core.clear_classified_prs(args.target_id)
    print(f"Bound {args.target_id} → repo={args.repo}, authority={args.authority}")
    print(f"Spec: {len(spec_body)} chars")
    return 0


def cmd_unbind(args) -> int:
    store = TargetStore()
    target = store.get(args.target_id)
    if target is None:
        print(f"ERROR: target not found: {args.target_id}", file=sys.stderr)
        return 2
    target.unbind_pm()
    target.save()
    pm_core.clear_classified_prs(args.target_id)
    print(f"Unbound {args.target_id}")
    return 0


def cmd_pause(args) -> int:
    store = TargetStore()
    target = store.get(args.target_id)
    if target is None:
        print(f"ERROR: target not found: {args.target_id}", file=sys.stderr)
        return 2
    target.set_paused(True, reason=args.reason)
    target.save()
    print(f"Paused {args.target_id}" + (f" (reason: {args.reason})" if args.reason else ""))
    return 0


def cmd_resume(args) -> int:
    store = TargetStore()
    target = store.get(args.target_id)
    if target is None:
        print(f"ERROR: target not found: {args.target_id}", file=sys.stderr)
        return 2
    target.set_paused(False)
    target.save()
    print(f"Resumed {args.target_id}")
    return 0


def cmd_tick(args) -> int:
    # Force-dispatch: bypass the normal decide path for smoke testing.
    if args.force_dispatch:
        if ":" not in args.force_dispatch:
            print("ERROR: --force-dispatch format is AGENT:INTENT", file=sys.stderr)
            return 2
        agent_type, intent = args.force_dispatch.split(":", 1)
        if not args.target:
            print("ERROR: --force-dispatch requires --target", file=sys.stderr)
            return 2
        target = TargetStore().get(args.target)
        if target is None or not target.pm_bound:
            print(f"ERROR: target {args.target} not bound", file=sys.stderr)
            return 2
        spec_sum = episodic.spec_summary(args.target)
        vars_ = {
            "target_id": args.target,
            "spec_summary": spec_sum,
            "repo": target.pm_repo or "",
            "question": intent,
            "pr_number": "",
            "slug": "forced",
        }
        res = shaper.dispatch(agent_type, args.target, intent, vars_=vars_)
        pm_core.append_dispatched(args.target, {
            "gpu_id": res.task_id,
            "spec_id": res.spec_id,
            "agent_type": agent_type,
            "intent": intent,
            "repo": target.pm_repo or "",
            "ts": pm_core._now_iso(),
            "status": "pending",
            "retry_count": 0,
        })
        episodic.write_dispatch(
            args.target,
            f"Forced dispatch: {agent_type} → {res.task_id}\nIntent: {intent}",
            extra_tags=[f"pm:gpu={res.task_id}", f"pm:agent={agent_type}"],
        )
        print(f"Dispatched: {agent_type} task_id={res.task_id} output={res.output_path}")
        return 0

    if args.force_brief:
        if not args.target:
            print("ERROR: --force-brief requires --target", file=sys.stderr)
            return 2
        b = brief.synthesize(args.target, trigger="manual force-brief")
        pm_core.set_outstanding_brief(args.target, b.comment_id)
        print(f"Brief posted: comment={b.comment_id} pushed={b.pushed}")
        return 0

    if args.target:
        results = [pm_core.tick(args.target)]
    else:
        results = pm_core.tick_all()

    for r in results:
        print(f"[{r.target_id}] skipped={r.skipped} reason={r.reason} "
              f"encoded={r.encoded} decision={r.decision}")
    return 0


def cmd_status(args) -> int:
    store = TargetStore()
    if args.target_id:
        t = store.get(args.target_id)
        if t is None:
            print(f"ERROR: target not found: {args.target_id}", file=sys.stderr)
            return 2
        _print_target_status(t, explain=args.explain)
        return 0
    for t in store.load_all():
        if t.pm_bound:
            _print_target_status(t, explain=args.explain)
            print()
    return 0


def _print_target_status(t, explain: bool = False):
    print(f"=== {t.id} ===")
    print(f"  title:         {t.title}")
    print(f"  pm_bound:      {t.pm_bound}")
    print(f"  pm_repo:       {t.pm_repo}")
    print(f"  pm_authority:  {t.pm_authority}")
    print(f"  paused:        {t.paused}"
          + (f" (reason: {t.paused_reason})" if t.paused and t.paused_reason else ""))
    cursor = pm_core.get_cursor(t.id)
    print(f"  cursor:        {cursor}")
    dispatched = pm_core.load_dispatched(t.id)
    pending = [d for d in dispatched if d.get("status") == "pending"]
    print(f"  dispatched:    {len(dispatched)} total, {len(pending)} pending")
    outstanding = pm_core.get_outstanding_brief(t.id)
    print(f"  outstanding:   {outstanding or '(none)'}")

    # Review-gate loop context (shown when a review loop is active)
    try:
        from agents_core.forgejo import get_open_prs
        open_prs = get_open_prs(t.pm_repo) if t.pm_repo else []
        from . import pm_core as _pm
        review_state = _pm._active_review_state(t.id, open_prs)
        if review_state:
            budget = _pm._REVIEW_CYCLE_BUDGETS.get(t.pm_authority, 2)
            mode = "fresh-reviewer" if t.pm_authority == "hold" else "same-reviewer"
            print(f"  reviewing:     PR #{review_state['pr_number']}, "
                  f"cycle {review_state['cycle']}/{budget} (opus, {mode})")
            verdict = review_state.get("verdict", "pending")
            issues = review_state.get("issues", 0)
            if verdict != "pending":
                print(f"  last-verdict:  {verdict}"
                      + (f", {issues} issue(s)" if verdict == "fixable" else ""))
    except Exception:
        pass  # status display is best-effort

    if explain:
        print("  --- recent PM episodes ---")
        for c in episodic.all_comments(t.id)[-10:]:
            if set(c.tags) & episodic.PM_TAGS:
                preview = c.content.replace("\n", " ")
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                print(f"  [{c.ts}] {','.join(c.tags):40s} {preview}")


def cmd_list(args) -> int:
    store = TargetStore()
    bound = [t for t in store.load_all() if t.pm_bound]
    if not bound:
        print("(no pm-bound targets)")
        return 0
    print(f"{'TARGET':30s} {'REPO':20s} {'AUTH':10s} {'PAUSED':8s} {'OUTSTANDING':12s}")
    for t in bound:
        outstanding = pm_core.get_outstanding_brief(t.id)
        print(f"{t.id:30s} {t.pm_repo or '-':20s} {t.pm_authority:10s} "
              f"{'yes' if t.paused else 'no':8s} "
              f"{outstanding[:10] if outstanding else '-':12s}")
    return 0


def cmd_land(args) -> int:
    store = TargetStore()
    target = store.get(args.target_id)
    if target is None:
        print(f"ERROR: target not found: {args.target_id}", file=sys.stderr)
        return 2
    if not target.pm_bound:
        print(f"ERROR: target not pm_bound: {args.target_id}", file=sys.stderr)
        return 2

    arc = land.generate_arc_doc(args.target_id)
    if args.dry_run:
        sys.stdout.write(arc.body)
        return 0

    path = land.write_arc_doc(arc)
    print(f"Arc doc written: {path}")
    print(f"Length: {len(arc.body)} chars")

    # Hygiene: archive the target, unbind from PM, and clear mem state so
    # the landed target stops appearing in tick loops and `lapis-pm status`.
    # Previously land stopped after writing the arc doc, leaving
    # pm_bound=True + stale cursor/dispatched/brief entries indefinitely.
    store.archive(args.target_id)          # sets status=archived + saves
    target = store.get(args.target_id)     # re-read after archive saved
    target.unbind_pm()
    target.save()
    cleared = pm_core.clear_landed_state(args.target_id)
    cleared_summary = ", ".join(f"{k}={v}" for k, v in cleared.items() if v)
    print(
        f"Archived + unbound {args.target_id}"
        + (f"; cleared: {cleared_summary}" if cleared_summary else "; mem state already clean")
    )
    return 0


def cmd_review_gate(args) -> int:
    sub = args.review_gate_sub
    if sub == "status":
        state = pm_core.review_gate_status()
        print(f"review-gate counter:   {state['counter']} / {state['threshold']}")
        print(f"review-gate paused:    {state['paused']}")
        if state["paused"]:
            print("  → Resume with: lapis-pm review-gate resume")
        return 0
    if sub == "resume":
        prev = pm_core.review_gate_resume()
        print(f"Review-gate counter reset (was {prev}). Opus reviewer active again.")
        return 0
    print(f"ERROR: unknown review-gate subcommand: {sub}", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lapis-pm", description="Lapis PM agent CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bind", help="Bind a target to the PM with a spec.")
    b.add_argument("target_id")
    b.add_argument("--spec-from", required=True, help="Path to spec, or '-' for stdin")
    b.add_argument("--repo", required=True, help="Forgejo repo name (Erah/<name>)")
    b.add_argument("--authority", default="advisory", choices=["advisory", "auto"])
    b.add_argument("--force", action="store_true", help="Replace existing spec binding")
    # --create flags: create target YAML if it doesn't exist
    b.add_argument("--create", action="store_true",
                   help="Create target YAML if it doesn't exist (errors if exists without --force)")
    b.add_argument("--title", default="", help="Target title (required with --create)")
    b.add_argument("--description", default="", help="Target description")
    b.add_argument("--urgency", default="medium", choices=["low", "medium", "high"],
                   help="Target urgency (default: medium)")
    b.add_argument("--work-mode", default="anywhere", dest="work_mode",
                   help="Target work mode (default: anywhere)")
    b.add_argument("--tag", action="append", default=[], metavar="TAG",
                   help="Add a tag (repeatable; repo name always included)")
    b.add_argument("--product", default="", help="Product name (e.g. 'Archetypal Intelligence')")
    b.set_defaults(func=cmd_bind)

    u = sub.add_parser("unbind", help="Remove PM binding from a target.")
    u.add_argument("target_id")
    u.set_defaults(func=cmd_unbind)

    t = sub.add_parser("tick", help="Run one tick of the PM.")
    t.add_argument("--target", help="Single target id. Default: all bound.")
    t.add_argument("--all", action="store_true", help="Tick all bound targets")
    t.add_argument("--force-brief", action="store_true")
    t.add_argument("--force-dispatch", help="AGENT:INTENT — bypass decide, smoke test dispatch")
    t.set_defaults(func=cmd_tick)

    s = sub.add_parser("status", help="Show PM state for bound target(s).")
    s.add_argument("target_id", nargs="?")
    s.add_argument("--explain", action="store_true")
    s.set_defaults(func=cmd_status)

    pa = sub.add_parser("pause", help="Pause a target.")
    pa.add_argument("target_id")
    pa.add_argument("--reason")
    pa.set_defaults(func=cmd_pause)

    re_ = sub.add_parser("resume", help="Resume a paused target.")
    re_.add_argument("target_id")
    re_.set_defaults(func=cmd_resume)

    ls = sub.add_parser("list", help="List all pm_bound targets.")
    ls.set_defaults(func=cmd_list)

    ld = sub.add_parser("land", help="Produce an arc doc for a landed thread.")
    ld.add_argument("target_id")
    ld.add_argument("--dry-run", action="store_true",
                    help="Print arc doc to stdout instead of writing to /srv/lapis/lapis-state/")
    ld.set_defaults(func=cmd_land)

    rg = sub.add_parser("review-gate",
                        help="Manage the Opus reviewer kill-switch.")
    rg_sub = rg.add_subparsers(dest="review_gate_sub", required=True)
    rg_sub.add_parser("status", help="Print current counter + threshold.")
    rg_sub.add_parser("resume", help="Reset counter; re-enable Opus reviewer.")
    rg.set_defaults(func=cmd_review_gate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
