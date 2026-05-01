#!/usr/bin/env python3
"""lapis-pm CLI.

Commands:
    bind <target_id> --spec-from PATH|- --repo REPO [--authority advisory|auto|hold]
               [--create --title TITLE [--description TEXT] [--urgency medium]
                [--tag TAG ...] [--product NAME]] [--force]
    bind <chain-group-id> --spec-from PATH|- --legs-from YAML-PATH
               [--create --title TITLE ...] [--no-auto-fire]
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

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore

from agents_core.targets import TargetStore

# Package imports work because the CLI is launched via `python -m lapis_pm.cli`.
from . import episodic, brief, pm_core, land, chain as chain_mod
from .router_portfolio import emit_decision_kickoff, emit_decision_dispatch, emit_decision_land

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


def _normalize_authority(auth: str | None) -> str:
    """Normalize authority string; accept 'auto-merge' as alias for 'auto'."""
    if auth is None:
        return "advisory"
    if auth == "auto-merge":
        return "auto"
    return auth


def cmd_bind(args) -> int:
    # Route to chain bind if --legs-from is present
    if getattr(args, "legs_from", None):
        if getattr(args, "repo", None):
            print(
                "ERROR: --legs-from and --repo are mutually exclusive; "
                "use --legs-from for chain-mode or --repo for single-target mode",
                file=sys.stderr,
            )
            return 2
        return cmd_bind_chain(args)

    # Single-target mode: --repo is required
    if not args.repo:
        print("ERROR: --repo is required for single-target bind", file=sys.stderr)
        return 2

    if args.authority is None:
        args.authority = "advisory"

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

    # Resolve pr_count: explicit flag > auto-detect (on --create) > preserve existing
    pr_count = getattr(args, "pr_count", None)
    if pr_count is not None:
        if pr_count < 1:
            print("ERROR: --pr-count must be >= 1", file=sys.stderr)
            return 2
        if not args.create and target.data.get("pr_count", 1) != pr_count and not args.force:
            print(
                f"ERROR: --pr-count {pr_count} differs from existing pr_count "
                f"{target.data.get('pr_count', 1)}. Use --force to override.",
                file=sys.stderr,
            )
            return 2
        target.data["pr_count"] = pr_count
    elif args.create:
        detected = pm_core._detect_pr_count_from_spec(spec_body)
        target.data["pr_count"] = detected

    target.bind_pm(repo=args.repo, authority=args.authority)
    target.save()

    episodic.write_spec(args.target_id, spec_body)
    pm_core.clear_classified_prs(args.target_id)
    try:
        _spec_title = spec_body.split("\n")[0].lstrip("# ").strip()[:200]
        emit_decision_kickoff(
            target_id=args.target_id,
            intent_summary=_spec_title,
            expert_chosen=None,
        )
    except Exception as _e:
        print(f"[router-portfolio:emit-failed] kickoff: {_e}", file=sys.stderr)
    print(f"Bound {args.target_id} → repo={args.repo}, authority={args.authority}")
    print(f"Spec: {len(spec_body)} chars")
    return 0


# ---------------------------------------------------------------------------
# Chain-mode bind
# ---------------------------------------------------------------------------

def cmd_bind_chain(args) -> int:
    """Bind a multi-leg chain from a legs YAML file.

    Expects args.target_id = chain-group-id, args.legs_from = path to legs YAML.
    Reads the spec from args.spec_from (applied to all legs).

    Legs YAML shape:
      legs:
        - tid: <target-id>
          repo: <repo-name>
          authority: advisory | auto | hold
          intent: |
            Multi-line fixer intent...
          # depends_on optional; absent = fires immediately on bind
          depends_on:
            - <prior-tid>
          title: <optional per-leg title>
    """
    if _yaml is None:
        print("ERROR: PyYAML not installed; cannot parse --legs-from YAML", file=sys.stderr)
        return 2

    chain_group = args.target_id
    legs_path = Path(args.legs_from)
    if not legs_path.exists():
        print(f"ERROR: legs file not found: {legs_path}", file=sys.stderr)
        return 2

    try:
        legs_raw = _yaml.safe_load(legs_path.read_text())
    except Exception as e:
        print(f"ERROR: failed to parse legs YAML: {e}", file=sys.stderr)
        return 2

    if not isinstance(legs_raw, dict) or "legs" not in legs_raw:
        print("ERROR: legs YAML must have a top-level 'legs' key", file=sys.stderr)
        return 2

    legs: list[dict] = legs_raw["legs"]
    if not legs:
        print("ERROR: legs list is empty", file=sys.stderr)
        return 2

    # Validate required fields per leg
    for i, leg in enumerate(legs):
        for field in ("tid", "repo", "authority", "intent", "branch_slug"):
            if not leg.get(field):
                print(f"ERROR: leg[{i}] missing required field {field!r}", file=sys.stderr)
                return 2
        leg["authority"] = _normalize_authority(leg["authority"])
        if leg["authority"] not in ("advisory", "auto", "hold"):
            print(
                f"ERROR: leg[{i}] authority {leg['authority']!r} invalid; "
                "must be advisory | auto | hold",
                file=sys.stderr,
            )
            return 2

    # Chain validation (cycle detection, duplicate tids, unknown deps)
    try:
        chain_mod.validate_legs(legs)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    spec_body = _read_spec(args.spec_from).strip()
    if not spec_body:
        print("ERROR: empty spec body", file=sys.stderr)
        return 2

    store = TargetStore()
    force = bool(getattr(args, "force", False))
    create = bool(getattr(args, "create", False))
    no_auto_fire = bool(getattr(args, "no_auto_fire", False))

    # Pre-flight: validate every leg upfront before mutating anything.
    # Catches existing-target / missing-target / existing-spec failures
    # before we partially bind a chain.
    for leg in legs:
        tid = leg["tid"]
        existing = store.get(tid)
        if existing is not None and not force:
            print(
                f"ERROR: target {tid!r} already exists. Use --force to replace.",
                file=sys.stderr,
            )
            return 2
        if existing is None and not create:
            print(
                f"ERROR: target {tid!r} not found. Use --create to create it.",
                file=sys.stderr,
            )
            return 2
        if episodic.spec(tid) and not force:
            print(
                f"ERROR: target {tid!r} already has a spec bound. Use --force.",
                file=sys.stderr,
            )
            return 2

    # Track new YAMLs we create so we can roll them back if a later
    # leg's mutation step raises unexpectedly.
    created_tids: list[str] = []

    def _rollback() -> None:
        for created_tid in created_tids:
            try:
                t = store.get(created_tid)
                if t is not None and t.path.exists():
                    t.path.unlink()
            except Exception:
                pass

    # Create/bind all legs
    initial_state_legs: list[dict] = []
    try:
        for leg in legs:
            tid = leg["tid"]
            leg_depends_on: list[str] = leg.get("depends_on") or []
            leg_intent: str = leg["intent"].strip()
            has_deps = bool(leg_depends_on)

            target = store.get(tid)
            newly_created = target is None
            if target is None:
                title = leg.get("title") or f"{chain_group}: {tid}"
                tags = _derive_tags(leg["repo"], tid, getattr(args, "tag", []) or [])
                target = store.create(
                    tid,
                    title=title,
                    urgency=getattr(args, "urgency", "medium"),
                    work_mode=getattr(args, "work_mode", "anywhere"),
                    description=leg.get("description", getattr(args, "description", "") or ""),
                    stages=list(PM_LIFECYCLE_STAGES),
                    category="active-work",
                )
                created_tids.append(tid)
                if tags:
                    target.data["tags"] = tags
                if getattr(args, "product", None):
                    target.data["product"] = args.product

            # Construct canonical branch and inject as first line of initial_dispatch.
            # Fixer sees the branch first; it cannot miss it regardless of spec wording.
            branch_slug = leg["branch_slug"]
            canonical_branch = f"lapis/{tid}/{branch_slug}"
            injected_dispatch = f"Branch: {canonical_branch}\n\n{leg_intent}"

            # Set chain fields on the target
            target.data["chain_group"] = chain_group
            if leg_depends_on:
                target.data["depends_on"] = leg_depends_on
            target.data["initial_dispatch"] = injected_dispatch

            target.bind_pm(repo=leg["repo"], authority=leg["authority"])
            target.save()

            episodic.write_spec(tid, spec_body)
            pm_core.clear_classified_prs(tid)
            try:
                _kickoff_summary = (leg_intent[:200] if leg_intent
                                    else spec_body.split("\n")[0].lstrip("# ").strip()[:200])
                emit_decision_kickoff(
                    target_id=tid,
                    intent_summary=_kickoff_summary,
                    expert_chosen=None,
                )
            except Exception as _e:
                print(f"[router-portfolio:emit-failed] kickoff leg {tid}: {_e}", file=sys.stderr)

            # No-dep legs are 'dispatched' only if we'll actually fire them below.
            # With --no-auto-fire the dispatch loop is skipped, so they stay 'pending'
            # until manually dispatched — keeps chain state honest about what fired.
            leg_status = "dispatched" if (not has_deps and not no_auto_fire) else "pending"
            initial_state_legs.append({"tid": tid, "status": leg_status})
            print(f"  Bound leg {tid} → repo={leg['repo']}, authority={leg['authority']}, "
                  f"depends_on={leg_depends_on or '(none)'}")
    except Exception as e:
        _rollback()
        print(
            f"ERROR: chain bind failed mid-loop: {e}\n"
            f"Rolled back {len(created_tids)} newly-created leg target(s); "
            "pre-existing targets bound earlier in the loop may need manual unbind.",
            file=sys.stderr,
        )
        return 1

    # Write initial chain state + emit bind event
    chain_mod.update_chain_state(chain_group, initial_state_legs)
    chain_mod.emit_chain_event(
        chain_group, "bind", chain_group,
        details={"legs": [leg["tid"] for leg in legs]},
    )
    print(f"Chain {chain_group!r} bound: {len(legs)} leg(s)")

    # Fire initial_dispatch for legs with no depends_on (unless --no-auto-fire)
    if not no_auto_fire:
        for leg in legs:
            if not (leg.get("depends_on") or []):
                tid = leg["tid"]
                target = store.get(tid)
                if target is None:
                    continue
                intent = leg["intent"].strip()
                spec_sum = episodic.spec_summary(tid)
                vars_ = {
                    "target_id": tid,
                    "spec_summary": spec_sum,
                    "repo": target.pm_repo or "",
                    "question": intent,
                    "pr_number": "",
                    "slug": "chain-init",
                }
                res = pm_core._SHAPER.dispatch("fixer", tid, intent, vars_=vars_)
                pm_core.append_dispatched(tid, {
                    "gpu_id": res.task_id,
                    "spec_id": res.spec_id,
                    "agent_type": "fixer",
                    "intent": intent,
                    "repo": target.pm_repo or "",
                    "ts": pm_core._now_iso(),
                    "status": "pending",
                    "retry_count": 0,
                })
                episodic.write_dispatch(
                    tid,
                    f"Chain initial dispatch: fixer → {res.task_id}\nIntent: {intent}",
                    extra_tags=[
                        f"pm:gpu={res.task_id}",
                        "pm:agent=fixer",
                        f"pm:chain-group={chain_group}",
                        "pm:chain-initial",
                    ],
                )
                try:
                    _model = "unknown"
                    try:
                        _model = pm_core._SHAPER.get_agent("fixer").model
                    except Exception:
                        pass
                    _TIER_MAP = {
                        "haiku": "haiku", "sonnet": "sonnet", "opus": "opus",
                        "qwen-3.6-35b-a3b": "qwen-local", "qwen3.6-35b-a3b": "qwen-local",
                    }
                    emit_decision_dispatch(
                        target_id=tid,
                        fragment_id="kickoff",
                        expert_chosen=_TIER_MAP.get(_model.lower(), _model.lower()),
                        intent_summary=intent[:200],
                    )
                except Exception as _e:
                    print(f"[router-portfolio:emit-failed] chain dispatch {tid}: {_e}",
                          file=sys.stderr)
                chain_mod.emit_chain_event(
                    chain_group, "dispatch", tid,
                    details={"task_id": res.task_id, "initial": True},
                )
                print(f"  Dispatched leg {tid}: fixer task_id={res.task_id}")

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
        res = pm_core._SHAPER.dispatch(agent_type, args.target, intent, vars_=vars_)
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
        try:
            _dispatched = pm_core.load_dispatched(args.target)
            if agent_type == "fixer":
                _frag = "kickoff" if len(_dispatched) == 1 else "tick"
            elif agent_type == "reviewer":
                _frag = "review-cycle"
            elif agent_type == "brief":
                _frag = "human-judgment"
            else:
                _frag = agent_type
            _model = "unknown"
            try:
                _model = pm_core._SHAPER.get_agent(agent_type).model
            except Exception:
                pass
            _TIER_MAP = {
                "haiku": "haiku", "sonnet": "sonnet", "opus": "opus",
                "qwen-3.6-35b-a3b": "qwen-local", "qwen3.6-35b-a3b": "qwen-local",
            }
            emit_decision_dispatch(
                target_id=args.target,
                fragment_id=_frag,
                expert_chosen=_TIER_MAP.get(_model.lower(), _model.lower()),
                intent_summary=intent[:200],
            )
        except Exception as _e:
            print(f"[router-portfolio:emit-failed] dispatch: {_e}", file=sys.stderr)
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
        rec_part = f" reconciled={r.reconciled}" if r.reconciled else ""
        print(f"[{r.target_id}] skipped={r.skipped} reason={r.reason}"
              f"{rec_part} encoded={r.encoded} decision={r.decision}")
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


def _target_to_json_dict(t) -> dict:
    dispatched = pm_core.load_dispatched(t.id)
    pending = [d for d in dispatched if d.get("status") == "pending"]
    return {
        "target_id": t.id,
        "title": t.title,
        "pm_repo": t.pm_repo,
        "pm_authority": t.pm_authority,
        "paused": t.paused,
        "cursor": pm_core.get_cursor(t.id),
        "dispatched_total": len(dispatched),
        "dispatched_pending": len(pending),
        "outstanding_brief_id": pm_core.get_outstanding_brief(t.id),
        "tags": t.data.get("tags", []),
        "urgency": t.urgency,
        "category": t.category,
    }


def cmd_list(args) -> int:
    store = TargetStore()
    bound = [t for t in store.load_all() if t.pm_bound]
    if getattr(args, "json", False):
        print(json.dumps([_target_to_json_dict(t) for t in bound]))
        return 0
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
    # Read chain_group before archive — `archive()` reloads the target and
    # subsequent state changes can leave the in-memory copy stale.
    chain_group = target.data.get("chain_group") or ""

    try:
        emit_decision_land(
            target_id=args.target_id,
            intent_summary=str(path),
        )
    except Exception as _e:
        print(f"[router-portfolio:emit-failed] land: {_e}", file=sys.stderr)

    pm_core._mem().set(
        pm_core._landed_key(args.target_id),
        json.dumps({"manual": True, "ts": pm_core._now_iso(),
                    "arc_path": str(path)}),
        tags=["lapis-pm", "landed"],
    )

    # Chain advance: auto-fire dependent legs before archiving this target.
    # Wrap both calls — if either raises, we still want to archive/unbind so
    # the target doesn't get stuck in a half-landed state.
    try:
        chain_mod.on_leg_landed(args.target_id, chain_group)
        chain_mod.check_chain_advance(args.target_id)
    except Exception as e:
        print(f"Warning: chain advance failed: {e}", file=sys.stderr)

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

    b = sub.add_parser("bind", help="Bind a target (or chain) to the PM with a spec.")
    b.add_argument("target_id",
                   help="Target ID (single-target) or chain-group-id (with --legs-from)")
    b.add_argument("--spec-from", required=True, help="Path to spec, or '-' for stdin")
    b.add_argument("--repo", default=None, help="Forgejo repo name (required for single-target)")
    b.add_argument("--authority", default=None, choices=["advisory", "auto", "hold"],
                   help="Authority level (default: advisory; hold = fresh-reviewer + 4-cycle budget)")
    b.add_argument("--legs-from", dest="legs_from", default=None, metavar="PATH",
                   help="Path to chain legs YAML; switches to chain-mode (mutex with --repo)")
    b.add_argument("--no-auto-fire", dest="no_auto_fire", action="store_true",
                   help="In chain mode: do not fire initial leg dispatch at bind time")
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
    b.add_argument("--pr-count", dest="pr_count", type=int, default=None, metavar="N",
                   help="Number of PRs that must merge before auto-land fires (>= 1). "
                        "With --create and no --pr-count: auto-detected from '### PR N' headers.")
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
    ls.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
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
