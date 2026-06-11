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
    brief --period {morning,afternoon,weekly,live} [--week YYYY-Www]
    brief-resolve <target_id> <option_id>
    review-gate {status,resume}
    trajectory-rollup --rebuild-index
    trajectory-rollup --period per-target [--target TID | --all]
    trajectory-rollup --period weekly [--week YYYY-Www]
    trajectory-rollup --period monthly [--month YYYY-MM]
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
from .backcaster.cli import cmd_backcaster
from .scout.cli import cmd_scout

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


def _humanize_chain_group(group: str) -> str:
    """Humanize a chain-group id for display. e.g. 'lapis-cockpit-loom-v0' -> 'Lapis cockpit loom v0'."""
    if not group:
        return ""
    words = group.replace("-", " ").replace("_", " ").split()
    if not words:
        return group
    return words[0].capitalize() + (" " + " ".join(words[1:]) if len(words) > 1 else "")


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

    # Validate --adopt-pr early so we fail before creating or mutating state.
    _adopted_pr_number: int | None = getattr(args, "adopt_pr", None)
    _adopted_head_branch: str = ""
    if _adopted_pr_number is not None:
        try:
            from agents_core.forgejo import get_pr as _get_pr
            _adopted_pr_data = _get_pr(args.repo, _adopted_pr_number)
        except Exception as e:
            print(
                f"ERROR: failed to fetch PR #{_adopted_pr_number} from {args.repo!r}: {e}",
                file=sys.stderr,
            )
            return 2
        _pr_state = _adopted_pr_data.get("state", "")
        _pr_merged = _adopted_pr_data.get("merged", False)
        if _pr_state != "open" or _pr_merged:
            print(
                f"ERROR: PR #{_adopted_pr_number} is not open "
                f"(state={_pr_state!r}, merged={_pr_merged}); only open PRs can be adopted",
                file=sys.stderr,
            )
            return 2
        _pr_base_repo = (
            (_adopted_pr_data.get("base") or {}).get("repo") or {}
        ).get("name", "")
        if _pr_base_repo and _pr_base_repo != args.repo:
            print(
                f"ERROR: PR #{_adopted_pr_number} base repo is {_pr_base_repo!r} "
                f"but --repo is {args.repo!r}",
                file=sys.stderr,
            )
            return 2
        _adopted_head_branch = (
            (_adopted_pr_data.get("head") or {}).get("ref") or ""
        )
        if not _adopted_head_branch:
            print(
                f"ERROR: PR #{_adopted_pr_number} has no head branch ref",
                file=sys.stderr,
            )
            return 2

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

    # Destination flags: validate + write if set. Omitting flags preserves existing value.
    dest_slug = getattr(args, "destination_slug", None)
    dest_name = getattr(args, "destination_name", None)
    dest_when = getattr(args, "destination_when", None)

    if dest_when is not None and dest_slug is None:
        print("ERROR: --destination-when requires --destination-slug", file=sys.stderr)
        return 2
    slug_set = dest_slug is not None
    name_set = dest_name is not None
    if slug_set != name_set:
        print(
            "ERROR: --destination-slug and --destination-name must both be set or both be omitted",
            file=sys.stderr,
        )
        return 2
    if slug_set:
        if not chain_mod._BRANCH_SLUG_RE.match(dest_slug):
            print(
                f"ERROR: --destination-slug must be kebab-case (lowercase alphanumeric, "
                f"hyphens allowed in the middle, no leading/trailing hyphens) "
                f"— got: {dest_slug}",
                file=sys.stderr,
            )
            return 2
        if not dest_name.strip():
            print("ERROR: --destination-name must be non-empty", file=sys.stderr)
            return 2
        target.data["destination"] = {
            "slug": dest_slug,
            "name": dest_name,
            "when": dest_when,
        }

    # loom_visibility: write if set, preserve existing value if omitted.
    loom_vis = getattr(args, "loom_visibility", None)
    if loom_vis is not None:
        target.data["loom_visibility"] = loom_vis

    if _adopted_pr_number is not None:
        target.data["adopted_pr_number"] = _adopted_pr_number
        target.data["adopted_head_branch"] = _adopted_head_branch

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
    if _adopted_pr_number is not None:
        print(f"Adopted PR #{_adopted_pr_number} on branch {_adopted_head_branch!r}")
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

    # Destination flags are not allowed in chain mode.
    if any(
        getattr(args, f, None) is not None
        for f in ("destination_slug", "destination_name", "destination_when")
    ):
        print(
            "ERROR: destination flags are not allowed in chain mode — "
            "chain group id becomes the destination automatically",
            file=sys.stderr,
        )
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

            # Destination defaulting: guard is load-bearing (preserves explicit destination
            # set by a prior single-target bind through chain rebind via --force).
            if "destination" not in target.data:
                target.data["destination"] = {
                    "slug": chain_group,
                    "name": _humanize_chain_group(chain_group),
                    "when": None,
                }

            # loom_visibility: write if flag set, preserve existing if omitted.
            chain_loom_vis = getattr(args, "loom_visibility", None)
            if chain_loom_vis is not None:
                target.data["loom_visibility"] = chain_loom_vis

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
        task_id = pm_core.force_dispatch(args.target, agent_type, intent)
        print(f"Dispatched: {agent_type} task_id={task_id}")
        return 0

    if args.force_brief:
        if not args.target:
            print("ERROR: --force-brief requires --target", file=sys.stderr)
            return 2
        b = brief.synthesize(args.target, trigger="manual force-brief", notify=None)
        pm_core.set_outstanding_brief(args.target, b.comment_id)
        print(f"Brief posted: comment={b.comment_id}")
        return 0

    if args.target:
        results = [pm_core.tick(args.target)]
    else:
        results = pm_core.tick_all()

    for r in results:
        if r.decision == "skipped:forgejo_unreachable":
            continue  # tick_all() already printed this line
        rec_part = f" reconciled={r.reconciled}" if r.reconciled else ""
        print(f"[{r.target_id}] skipped={r.skipped} reason={r.reason}"
              f"{rec_part} encoded={r.encoded} decision={r.decision}")
    return 0


def cmd_status(args) -> int:
    # Global review-gate kill-switch section (separate from per-target loop state).
    rg = pm_core.review_gate_status()
    print("=== review-gate ===")
    print(f"  counter:       {rg['counter']} / {rg['threshold']}")
    print(f"  paused:        {rg['paused']}")
    if rg["paused"]:
        print('  → Resume with: lapis-pm review-gate resume --reason "..."')
    print()

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
        if t.pm_repo:
            repo_name, owner = pm_core._repo_owner(t.pm_repo)
            open_prs = get_open_prs(repo_name, owner=owner)
        else:
            open_prs = []
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
        "destination": t.data.get("destination"),
        "loom_visibility": t.data.get("loom_visibility"),
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

    pm_core._post_land_deploy_hook(target.pm_repo)

    # Chain advance: auto-fire dependent legs before archiving this target.
    # Wrap both calls — if either raises, we still want to archive/unbind so
    # the target doesn't get stuck in a half-landed state.
    try:
        chain_mod.on_leg_landed(args.target_id, chain_group)
        chain_mod.check_chain_advance(args.target_id)
    except Exception as e:
        print(f"Warning: chain advance failed: {e}", file=sys.stderr)

    # Trajectory rollup hook: fire per-target + rebuild-index AFTER chain advance.
    # Non-fatal — Qwen can take 30-120s; failures are logged at WARNING only.
    # The nightly timer will catch any missed rollup on the next tick.
    try:
        from . import trajectory as _traj
        _traj.on_land(args.target_id)
    except Exception as e:
        import warnings
        warnings.warn(f"trajectory on_land failed (non-fatal): {e}", stacklevel=2)

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

    # Opportunistic compaction of old landed targets (non-fatal).
    try:
        compaction_results = pm_core.compact_eligible_targets()
        newly_compacted = [tid for tid, outcome in compaction_results if outcome == "compacted"]
        if newly_compacted:
            print(f"Opportunistic compaction: {len(newly_compacted)} old targets compacted.")
    except Exception as _ce:
        import warnings
        warnings.warn(f"compact_eligible_targets failed (non-fatal): {_ce}", stacklevel=2)

    return 0


def cmd_ratify(args) -> int:
    """Principal-feedback surface: record ratification of a Router decision.

    lapis-pm ratify <target_id> <confirm|correct|override|redirect>
                    [--intent SUMMARY] [--prior EVENT_ID] [--json]
    """
    from .router_portfolio import (
        find_latest_decision,
        emit_ratify_confirm,
        emit_ratify_correct,
        emit_ratify_override,
        emit_ratify_redirect,
    )

    target_id = args.target_id
    outcome = args.outcome
    intent = args.intent
    prior_raw = args.prior
    as_json = getattr(args, "json", False)

    # --intent is required for non-confirm outcomes
    if outcome != "confirm" and not intent:
        print(f"ERROR: --intent is required for outcome {outcome!r}", file=sys.stderr)
        return 2

    # Normalize --prior: strip namespace prefix if a full mem key was given
    _DECISIONS_NS = "router/lapis-pm/decisions/"
    if prior_raw is not None:
        prior_event_id: str | None = (
            prior_raw[len(_DECISIONS_NS):]
            if prior_raw.startswith(_DECISIONS_NS)
            else prior_raw
        )
    else:
        # Auto-prior resolution: find most recent decision for this target
        decision = find_latest_decision(target_id)
        if decision is not None:
            mem_key = decision.get("_mem_key", "")
            prior_event_id = (
                mem_key[len(_DECISIONS_NS):]
                if mem_key.startswith(_DECISIONS_NS)
                else mem_key
            )
        else:
            if outcome == "confirm":
                print(
                    f"no prior decision found for {target_id}; cannot confirm — "
                    f"pass --prior <event_id> explicitly",
                    file=sys.stderr,
                )
                return 3
            else:
                print(
                    f"[ratify] no prior decision found for {target_id}; writing with empty citations",
                    file=sys.stderr,
                )
                prior_event_id = None

    # Emit ratification
    try:
        if outcome == "confirm":
            key = emit_ratify_confirm(
                target_id=target_id,
                prior_decision_event_id=prior_event_id,
            )
        elif outcome == "correct":
            key = emit_ratify_correct(
                target_id=target_id,
                intent_summary=intent,
                prior_decision_event_id=prior_event_id,
            )
        elif outcome == "override":
            key = emit_ratify_override(
                target_id=target_id,
                intent_summary=intent,
                prior_decision_event_id=prior_event_id,
            )
        else:  # redirect
            key = emit_ratify_redirect(
                target_id=target_id,
                intent_summary=intent,
                prior_decision_event_id=prior_event_id,
            )
    except Exception as e:
        print(f"ERROR: ratification failed: {e}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps({"key": key}))
    else:
        print(f"Ratified {target_id} ({outcome}): {key}")
    return 0


def cmd_brief_resolve(args) -> int:
    """Resolve the current outstanding brief for target_id by choosing option_id.

    Reads the outstanding brief id from mem, then calls brief.apply_decision().
    Exits 0 on success (including noop_already_applied); non-zero on error.
    """
    target_id = args.target_id
    option_id = args.option_id

    outstanding = pm_core.get_outstanding_brief(target_id)
    if outstanding is None:
        print(f"ERROR: no outstanding brief for {target_id}", file=sys.stderr)
        return 2

    result = brief.apply_decision(target_id, outstanding, option_id)
    if result.get("ok"):
        print(f"Resolved: target={target_id} brief={outstanding} "
              f"option={option_id} action={result.get('action_kind')} "
              f"detail={result.get('detail')}")
        return 0
    else:
        print(f"ERROR: {result.get('error', 'unknown error')}", file=sys.stderr)
        return 1


def cmd_brief(args) -> int:
    from . import state_brief
    period = args.period
    out = state_brief.generate_brief(period)
    if out is not None:
        print(f"Brief written: {out}")
        print(f"Symlink updated: /srv/lapis/briefs/latest-{period}.md")
    return 0


def cmd_decisions_export(args) -> int:
    from lapis_pm.decisions_export import run
    return run(
        since_spec=args.since,
        include=args.include,
        exclude=args.exclude,
        tag=args.tag,
        out=args.out,
    )


def cmd_compact(args) -> int:
    """Handle `lapis-pm compact [--days N] [--dry-run]`."""
    import json as _json
    from datetime import datetime as _dt, timezone, timedelta

    days = args.days
    if days < 1:
        print("ERROR: --days must be >= 1", file=sys.stderr)
        return 2

    dry_run = args.dry_run
    mem = pm_core._mem()
    cutoff = _dt.now(timezone.utc) - timedelta(days=days)

    landed_entries = mem.list_by_prefix("pm/landed/", limit=500)

    eligible: list[tuple[str, list]] = []  # (target_id, dispatched_records)
    skipped_already = 0
    skipped_no_key = 0

    for entry in landed_entries:
        key = entry["key"]
        target_id = key[len("pm/landed/"):]

        try:
            landed_rec = _json.loads(entry["content"])
        except Exception:
            continue

        age_ts_str = landed_rec.get("landed_at") or landed_rec.get("ts")
        if not age_ts_str:
            continue

        try:
            age_dt = _dt.fromisoformat(age_ts_str)
            if age_dt.tzinfo is None:
                age_dt = age_dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if age_dt > cutoff:
            continue

        dispatched_rec = mem.get(pm_core._dispatched_key(target_id))
        if not dispatched_rec:
            skipped_no_key += 1
            continue

        try:
            dispatched = _json.loads(dispatched_rec["content"])
        except Exception:
            dispatched = []

        if dispatched and dispatched[0].get("compacted"):
            skipped_already += 1
            continue

        eligible.append((target_id, dispatched))

    if dry_run:
        print(f"Would compact {len(eligible)} targets ({days}+ days since landing):")
        for tid, records in eligible:
            size_bytes = len(_json.dumps(records).encode())
            print(f"  {tid}: {len(records)} records, ~{size_bytes / 1024:.1f} KB")
        return 0

    # Live run - compact each eligible target.
    compacted_tids = []
    for tid, _records in eligible:
        pm_core.compact_dispatched(tid)
        compacted_tids.append(tid)

    skip_note = ""
    if skipped_already or skipped_no_key:
        parts = []
        if skipped_already:
            parts.append(f"{skipped_already} already compacted")
        if skipped_no_key:
            parts.append(f"{skipped_no_key} no dispatched key")
        skip_note = f" (skipped {', '.join(parts)})"
    print(f"Compacted {len(compacted_tids)} targets{skip_note}.")

    for tid in compacted_tids:
        stub_rec = mem.get(pm_core._dispatched_key(tid))
        try:
            stub = _json.loads(stub_rec["content"]) if stub_rec else [{}]
        except Exception:
            stub = [{}]
        count = stub[0].get("dispatch_count", "?") if stub else "?"
        print(f"  {tid}: {count} records \u2192 stub")

    return 0


def cmd_local_witness(args) -> int:
    """Handle `lapis-pm local-witness stats` subcommand."""
    sub = getattr(args, "local_witness_sub", None)
    if sub != "stats":
        print(f"ERROR: unknown local-witness subcommand: {sub}", file=sys.stderr)
        return 2

    import json as _json
    from lapis_pm import episodic as _episodic
    from agents_core.mem import Mem

    repo_filter = getattr(args, "repo", None)
    since_filter = getattr(args, "since", None)

    since_dt = None
    if since_filter:
        from datetime import datetime as _dt
        try:
            since_dt = _dt.strptime(since_filter, "%Y-%m-%d")
        except ValueError:
            print(f"ERROR: --since must be YYYY-MM-DD, got: {since_filter}", file=sys.stderr)
            return 2

    # Collect all targets that have reviewer comments
    mem = Mem()
    all_targets: list[str] = []
    try:
        dispatched_keys = mem.search("pm/dispatched")
        for k in dispatched_keys:
            tid = k.split("pm/dispatched/", 1)[-1]
            if tid and tid not in all_targets:
                all_targets.append(tid)
    except Exception:
        pass

    # Fall back to scanning episodic store if mem lookup is sparse
    # (The store path is the canonical substrate)
    from lapis_pm.episodic import _store as _ep_store
    try:
        store = _ep_store()
        for tid in store.list_targets():
            if tid not in all_targets:
                all_targets.append(tid)
    except Exception:
        pass

    # Agreement matrix accumulator
    agree_counts = {"agree": 0, "diverge_minor": 0, "diverge_major": 0, "local_failed": 0}
    # Confusion matrix: (claude_verdict, local_verdict) -> count
    confusion: dict[tuple[str, str], int] = {}
    latencies: list[int] = []
    json_valid_count = 0
    total = 0

    for tid in all_targets:
        try:
            for comment in _episodic.all_comments(tid):
                # Only look at reviewer verdict comments
                is_reviewer = any(
                    t.startswith("pm:reviewer:pr=") and ":cycle=" in t and ":verdict=" in t
                    for t in comment.tags
                )
                if not is_reviewer:
                    continue
                content = comment.content or ""
                json_part = content.split("\n", 1)[-1].strip()
                try:
                    body = _json.loads(json_part)
                except Exception:
                    continue
                if "local_reviewer_witness" not in body:
                    continue

                # Apply date filter
                if since_dt:
                    # Comments have a timestamp field via the store
                    ts = getattr(comment, "created_at", None) or getattr(comment, "timestamp", None)
                    if ts:
                        from datetime import datetime as _dt2
                        try:
                            if isinstance(ts, str):
                                cdt = _dt2.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                            else:
                                cdt = ts.replace(tzinfo=None) if hasattr(ts, 'replace') else ts
                            if cdt < since_dt:
                                continue
                        except Exception:
                            pass

                # Apply repo filter
                if repo_filter:
                    has_repo_tag = any(f"pm:repo={repo_filter}" in t for t in comment.tags)
                    if not has_repo_tag:
                        continue

                wit = body["local_reviewer_witness"]
                agreement = wit.get("agreement", "local_failed")
                if agreement in agree_counts:
                    agree_counts[agreement] += 1

                cv = body.get("verdict", "?")
                lv = wit.get("verdict") or "?"
                key = (cv, lv)
                confusion[key] = confusion.get(key, 0) + 1

                lat = wit.get("latency_ms")
                if isinstance(lat, int):
                    latencies.append(lat)

                if wit.get("json_valid"):
                    json_valid_count += 1

                total += 1
        except Exception:
            continue

    if total == 0:
        print("No local_reviewer_witness observations found.")
        if since_filter:
            print(f"  (filter: --since {since_filter})")
        if repo_filter:
            print(f"  (filter: --repo {repo_filter})")
        return 0

    print(f"Local-reviewer witness stats  (N={total})")
    print()
    print("Agreement breakdown:")
    for k, v in agree_counts.items():
        pct = 100 * v / total if total else 0
        print(f"  {k:<18} {v:>4}  ({pct:.1f}%)")
    print()

    verdicts_order = ["clean", "fixable", "needs-human", "?"]
    print("Confusion matrix (claude_verdict × local_verdict):")
    header_cols = [lv for _, lv in confusion.keys()]
    all_lv = sorted(set(header_cols), key=lambda x: verdicts_order.index(x) if x in verdicts_order else 99)
    all_cv = sorted(set(cv for cv, _ in confusion.keys()), key=lambda x: verdicts_order.index(x) if x in verdicts_order else 99)
    col_w = max(len(v) for v in all_lv + ["local →"]) + 2
    cv_w = max(len(v) for v in all_cv + ["claude ↓"]) + 2
    header = f"{'claude ↓ / local →':<{cv_w}}" + "".join(f"{lv:>{col_w}}" for lv in all_lv)
    print("  " + header)
    for cv in all_cv:
        row = f"  {cv:<{cv_w}}"
        for lv in all_lv:
            cnt = confusion.get((cv, lv), 0)
            row += f"{cnt:>{col_w}}"
        print(row)
    print()

    # Precision / recall on fixable
    tp_fixable = confusion.get(("fixable", "fixable"), 0)
    fp_fixable = sum(confusion.get((cv, "fixable"), 0) for cv in all_cv if cv != "fixable")
    fn_fixable = sum(confusion.get(("fixable", lv), 0) for lv in all_lv if lv != "fixable")
    prec = tp_fixable / (tp_fixable + fp_fixable) if (tp_fixable + fp_fixable) > 0 else float("nan")
    rec = tp_fixable / (tp_fixable + fn_fixable) if (tp_fixable + fn_fixable) > 0 else float("nan")
    print(f"Fixable precision: {prec:.2f}   recall: {rec:.2f}")
    print()

    # Latency percentiles
    if latencies:
        latencies_sorted = sorted(latencies)
        n_lat = len(latencies_sorted)
        def pct(p: float) -> int:
            idx = max(0, int(n_lat * p / 100) - 1)
            return latencies_sorted[idx]
        print(f"Latency (ms)  p50={pct(50)}  p90={pct(90)}  p99={pct(99)}")
    else:
        print("Latency data: none recorded")

    # JSON validity
    print(f"JSON validity: {json_valid_count}/{total} ({100*json_valid_count/total:.1f}%)")
    return 0


def cmd_review_gate(args) -> int:
    sub = args.review_gate_sub
    if sub == "status":
        state = pm_core.review_gate_status()
        print(f"review-gate counter:   {state['counter']} / {state['threshold']}")
        print(f"review-gate paused:    {state['paused']}")
        if state["paused"]:
            print('  → Resume with: lapis-pm review-gate resume --reason "..."')
        return 0
    if sub == "resume":
        try:
            prev = pm_core.review_gate_resume(reason=args.reason)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(f"Review-gate counter reset (was {prev}). Opus reviewer active again.")
        print(f"Reason recorded: {args.reason}")
        return 0
    print(f"ERROR: unknown review-gate subcommand: {sub}", file=sys.stderr)
    return 2


def cmd_spec_review(args) -> int:
    """Handle `lapis-pm spec-review` subcommand."""
    from pathlib import Path
    from .spec_review import (
        SpecFrontmatterError,
        InvariantContextError,
        format_brief,
        run_spec_review,
    )

    spec_path = Path(args.spec_path)
    # Emit deprecation note when caller passes the legacy --compare-opus flag
    if getattr(args, "compare_opus", False):
        print(
            "[spec-review] WARNING: --compare-opus is deprecated and a no-op; "
            "the Sonnet reference leg now runs by default (sunset 90 days after merge).",
            file=sys.stderr,
        )
    try:
        brief = run_spec_review(
            spec_path=spec_path,
            council_voicing=args.council_voicing,
            timeout_s=args.timeout,
            repo_override=args.repo_override,
            authority=getattr(args, "authority", None),
            dispatch_facets=not getattr(args, "no_facets", False),
            sonnet_reviewer=not getattr(args, "no_sonnet_reviewer", False),
            facets_operator=getattr(args, "facets_operator", "haiku"),
        )
    except (SpecFrontmatterError, InvariantContextError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: dispatch failure — {e}", file=sys.stderr)
        return 2

    print(format_brief(brief))
    return 0 if brief.combined_recommendation not in {"parse_failed", "incomplete"} else 1


def cmd_eval_gate(args) -> int:
    """Handle `lapis-pm eval-gate` subcommand."""
    from . import eval_gate as eg

    sub = args.eval_gate_sub

    if sub == "status":
        # Show last eval run for each open Synapse PR
        try:
            from agents_core.forgejo import get_open_prs
            open_prs = get_open_prs("synapse") or []
        except Exception as exc:
            print(f"WARNING: could not fetch open synapse PRs: {exc}", file=sys.stderr)
            open_prs = []

        statuses = eg.eval_gate_status(open_prs, repo="synapse")
        if args.json:
            print(json.dumps(statuses, ensure_ascii=False, indent=2))
        else:
            if not statuses:
                print("No open Synapse PRs.")
            for s in statuses:
                actioned = " [actioned]" if s["actioned"] else ""
                regressed = ""
                if s["regressed_metrics"]:
                    regressed = f" regressed={s['regressed_metrics']}"
                print(
                    f"PR #{s['pr_number']} sha={s['head_sha']} "
                    f"status={s['status']}{regressed}{actioned}"
                )
        return 0

    if sub == "run":
        pr_number = args.pr
        # Force re-eval by removing cache entry
        from agents_core.forgejo import get_open_prs
        try:
            open_prs = get_open_prs("synapse") or []
        except Exception as exc:
            print(f"ERROR: cannot fetch open Synapse PRs: {exc}", file=sys.stderr)
            return 1

        pr = next((p for p in open_prs if p.get("number") == pr_number), None)
        if pr is None:
            print(f"ERROR: PR #{pr_number} not found in open Synapse PRs", file=sys.stderr)
            return 1

        head_sha = (pr.get("head") or {}).get("sha", "")
        if head_sha:
            cache_file = eg.RUNS_DIR / f"{head_sha}.json"
            if cache_file.exists():
                cache_file.unlink()
                print(f"Cache cleared for sha={head_sha[:8]}")

        result = eg.evaluate_pr("eval-gate-cli", pr, "synapse")
        if result is None:
            print("Eval gate returned no result (path-filter miss, fixture missing, or eval error).")
            return 1

        if args.json:
            from dataclasses import asdict
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print(f"PR #{result.pr_number} sha={result.head_sha[:8]}")
            print(f"Status: {result.status}")
            print(f"Regressed: {result.regressed}")
            if result.regressed_metrics:
                print(f"Regressed metrics: {result.regressed_metrics}")
            print()
            print(result.summary_text)
        return 0

    if sub == "baseline":
        baseline_sub = args.baseline_sub

        if baseline_sub == "show":
            baseline = eg.load_baseline()
            if baseline is None:
                print("No baseline found at", eg.BASELINE_PATH)
                return 1
            if args.json:
                print(json.dumps(baseline, ensure_ascii=False, indent=2))
            else:
                sha = baseline.get("sha", "?")[:12]
                gen = baseline.get("generated_at", "?")
                metrics = baseline.get("metrics", {})
                print(f"Baseline sha={sha} generated_at={gen}")
                for m, v in metrics.items():
                    print(f"  {m}: {v}")
            return 0

        if baseline_sub == "regenerate":
            print("Regenerating baseline (this may take up to 240s)...")
            result = eg.regenerate_baseline()
            if result is None:
                print("ERROR: baseline regeneration failed. Check logs.", file=sys.stderr)
                return 1
            sha = result.get("sha", "?")[:12]
            print(f"Baseline regenerated: sha={sha}")
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        print(f"ERROR: unknown baseline subcommand: {baseline_sub}", file=sys.stderr)
        return 2

    print(f"ERROR: unknown eval-gate subcommand: {sub}", file=sys.stderr)
    return 2


def cmd_trajectory_rollup(args) -> int:
    """Handle `lapis-pm trajectory-rollup` subcommand."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from . import trajectory as _traj

    if args.rebuild_index:
        try:
            path = _traj.rebuild_index()
            print(f"index.json written: {path}")
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 1
        return 0

    period = args.period
    if not period:
        print("ERROR: --period or --rebuild-index is required", file=sys.stderr)
        return 2

    if period == "per-target":
        if not args.target and not args.all:
            print("ERROR: --period per-target requires --target TID or --all",
                  file=sys.stderr)
            return 2
        paths = _traj.rollup_per_target(
            tid=args.target or None,
            all_targets=bool(args.all),
        )
        print(f"per-target: wrote {len(paths)} file(s)")
        return 0

    if period == "weekly":
        path = _traj.rollup_weekly(week=args.week or None)
        print(f"weekly digest written: {path}")
        return 0

    if period == "monthly":
        path = _traj.rollup_monthly(month=args.month or None)
        print(f"monthly digest written: {path}")
        return 0

    print(f"ERROR: unknown --period {period!r}", file=sys.stderr)
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
    b.add_argument("--destination-slug", default=None, dest="destination_slug", metavar="SLUG",
                   help="Destination slug (kebab-case). Optional. Single-target bind only.")
    b.add_argument("--destination-name", default=None, dest="destination_name", metavar="NAME",
                   help="Destination display name. Required if --destination-slug is set.")
    b.add_argument("--destination-when", default=None, dest="destination_when", metavar="WHEN",
                   help="Optional schedule label (e.g. 'tomorrow', 'this week').")
    b.add_argument("--loom-visibility", default=None, dest="loom_visibility",
                   choices=["pinned", "default", "hidden"],
                   help="Per-target Loom visibility. 'pinned' always shows in Loom; 'default' "
                        "shows when active in window (implicit when field absent); 'hidden' never shows.")
    b.add_argument("--adopt-pr", dest="adopt_pr", type=int, default=None, metavar="N",
                   help="Adopt an existing open PR by number. Records its head branch so "
                        "perception honors it and the first tick dispatches a reviewer "
                        "instead of the initial fixer. The PR must be open and its base "
                        "repo must match --repo.")
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

    rat = sub.add_parser(
        "ratify",
        help="Record principal ratification of a Router decision.",
    )
    rat.add_argument("target_id", help="Target ID being ratified")
    rat.add_argument(
        "outcome",
        choices=["confirm", "correct", "override", "redirect"],
        help="Ratification outcome",
    )
    rat.add_argument(
        "--intent",
        default=None,
        metavar="SUMMARY",
        help=(
            "One-line summary (required for correct/override/redirect; "
            "optional for confirm)"
        ),
    )
    rat.add_argument(
        "--prior",
        default=None,
        metavar="EVENT_ID",
        help=(
            "Prior decision event_id or full mem key "
            "(auto-resolved from target if omitted)"
        ),
    )
    rat.add_argument(
        "--json",
        action="store_true",
        help="Print written mem key as JSON {\"key\": ...}",
    )
    rat.set_defaults(func=cmd_ratify)

    br = sub.add_parser("brief", help="Generate a state-of-work brief.")
    br.add_argument(
        "--period",
        required=True,
        choices=["morning", "afternoon", "weekly", "live"],
        help="Brief cadence: morning/afternoon/live use Qwen; weekly uses Sonnet via Claude CLI.",
    )
    br.set_defaults(func=cmd_brief)

    brs = sub.add_parser(
        "brief-resolve",
        help="Resolve the current outstanding brief by choosing a closed-form option.",
    )
    brs.add_argument("target_id", help="Target ID with an outstanding brief.")
    brs.add_argument("option_id", help="Option ID to apply (e.g. A, B, C).")
    brs.set_defaults(func=cmd_brief_resolve)

    rg = sub.add_parser("review-gate",
                        help="Manage the Opus reviewer kill-switch.")
    rg_sub = rg.add_subparsers(dest="review_gate_sub", required=True)
    rg_sub.add_parser("status", help="Print current counter + threshold.")
    rg_resume = rg_sub.add_parser("resume", help="Reset counter; re-enable Opus reviewer.")
    rg_resume.add_argument(
        "--reason",
        required=True,
        help="Why is the gate being resumed? Captured in mem audit trail.",
    )
    rg.set_defaults(func=cmd_review_gate)

    tr = sub.add_parser(
        "trajectory-rollup",
        help="Generate trajectory reports: DAG index, per-target one-liners, "
             "weekly and monthly digests.",
    )
    tr.add_argument(
        "--rebuild-index",
        dest="rebuild_index",
        action="store_true",
        help="Regenerate /srv/lapis/trajectory/index.json (deterministic, no LLM). "
             "Errors on depends_on cycles.",
    )
    tr.add_argument(
        "--period",
        choices=["per-target", "weekly", "monthly"],
        default=None,
        help="Report period to generate.",
    )
    tr.add_argument(
        "--target",
        default=None,
        metavar="TID",
        help="Target ID (with --period per-target).",
    )
    tr.add_argument(
        "--all",
        action="store_true",
        dest="all",
        help="Process all landed targets (with --period per-target).",
    )
    tr.add_argument(
        "--week",
        default=None,
        metavar="YYYY-Www",
        help="ISO week to generate (default: current week). With --period weekly.",
    )
    tr.add_argument(
        "--month",
        default=None,
        metavar="YYYY-MM",
        help="Month to generate (default: current month). With --period monthly.",
    )
    tr.set_defaults(func=cmd_trajectory_rollup)

    # ------------------------------------------------------------------
    # scout — pseudocode-simulate v0
    # ------------------------------------------------------------------
    sc = sub.add_parser(
        "scout",
        help="Lapis Scout v0: architectural fuzzing via pseudocode simulation.",
    )
    sc_sub = sc.add_subparsers(dest="scout_sub", required=True)

    sc_sim = sc_sub.add_parser(
        "simulate",
        help="Run a scaffold against the parameter matrix and persist traces.",
    )
    sc_sim.add_argument(
        "scaffold",
        metavar="SCAFFOLD_PATH",
        help="Path to the scaffold YAML file.",
    )
    sc_sim.add_argument(
        "--cell",
        default=None,
        metavar="CELL_ID",
        help="Run only this cell (e.g. opt=weight-by-recency,load=5,severity=degraded).",
    )
    sc_sim.add_argument(
        "--runs",
        type=int,
        default=None,
        metavar="N",
        help="Override runs_per_cell from the scaffold.",
    )
    sc_sim.set_defaults(func=cmd_scout)

    sc_dig = sc_sub.add_parser(
        "digest",
        help="Digest persisted traces into a failure-mode map.",
    )
    sc_dig.add_argument(
        "spec_id",
        help="Spec ID to digest (must have traces under /srv/lapis/scout/traces/<spec_id>/).",
    )
    sc_dig.set_defaults(func=cmd_scout)

    # ------------------------------------------------------------------
    # scout night — night-queue orchestrator
    # ------------------------------------------------------------------
    sc_night = sc_sub.add_parser(
        "night",
        help="Night-queue orchestrator: health-gated, quarantine-aware, round-robin scheduler.",
    )
    sc_night_sub = sc_night.add_subparsers(dest="night_sub", required=True)

    sc_nr = sc_night_sub.add_parser("run", help="Start a night run.")
    sc_nr.add_argument(
        "--until",
        default=None,
        metavar="HHMM",
        help=(
            "Stop at this local time today (e.g. 0400). "
            "If already past, run is a no-op (no next-day roll). "
            "Superseded by --deadline-in. Default: 4h from now."
        ),
    )
    sc_nr.add_argument(
        "--deadline-in",
        default=None,
        metavar="DURATION",
        help="Stop after this duration from now (e.g. '4h', '30m', '1h30m'). Supersedes --until.",
    )
    sc_nr.add_argument(
        "--max-units",
        default=None,
        type=int,
        metavar="N",
        help="Hard ceiling on units executed regardless of time.",
    )
    sc_nr.add_argument(
        "--selected",
        default=None,
        metavar="FILE",
        help=(
            "Priority-lane YAML (list of {sim, target_pct}). "
            "Default: <sims-dir>/selected.yaml if it exists; else no priority lane."
        ),
    )
    sc_nr.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run each scaffold once then exit (drains worklist; budget still applies).",
    )
    sc_nr.add_argument(
        "--sims-dir",
        default=None,
        metavar="DIR",
        help="Directory containing scaffold YAMLs (default: /srv/lapis/scout/sims).",
    )
    sc_nr.add_argument(
        "--log-root",
        default=None,
        metavar="DIR",
        help="Directory for manifest.tsv and quarantine.json.",
    )
    sc_nr.add_argument(
        "--profile-override",
        action="append",
        default=[],
        metavar="SPEC_ID:PROFILE",
        help="Override a scaffold's priority profile (may be repeated).",
    )
    sc_nr.set_defaults(func=cmd_scout)

    sc_ns = sc_night_sub.add_parser("status", help="Show night-run status.")
    sc_ns.add_argument(
        "--log-root",
        default=None,
        metavar="DIR",
        help="Log root of the run to inspect (default: most recent).",
    )
    sc_ns.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit structured JSON (for claude-view/Librarian consumption).",
    )
    sc_ns.set_defaults(func=cmd_scout)

    sc_nq = sc_night_sub.add_parser(
        "quarantine-clear",
        help="Clear a scaffold's quarantine entry (idempotent).",
    )
    sc_nq.add_argument("spec_id", help="Spec ID to un-quarantine.")
    sc_nq.add_argument(
        "--log-root",
        default=None,
        metavar="DIR",
        help="Log root of the run (default: most recent).",
    )
    sc_nq.set_defaults(func=cmd_scout)

    # ------------------------------------------------------------------
    # eval-gate — synapse retrieval-quality gate
    # ------------------------------------------------------------------
    eg = sub.add_parser(
        "eval-gate",
        help="Synapse eval-gate: retrieval-quality check on PR branches.",
    )
    eg_sub = eg.add_subparsers(dest="eval_gate_sub", required=True)

    eg_status = eg_sub.add_parser("status", help="Show last eval run for each open Synapse PR.")
    eg_status.add_argument("--json", action="store_true", default=False,
                           help="Emit structured JSON output.")
    eg_status.set_defaults(func=cmd_eval_gate)

    eg_run = eg_sub.add_parser("run", help="Force a re-eval for a specific PR (bypasses cache).")
    eg_run.add_argument("--pr", type=int, required=True, metavar="N", help="PR number to evaluate.")
    eg_run.add_argument("--json", action="store_true", default=False,
                        help="Emit structured JSON output.")
    eg_run.set_defaults(func=cmd_eval_gate)

    eg_baseline = eg_sub.add_parser("baseline", help="Manage the eval baseline.")
    eg_baseline_sub = eg_baseline.add_subparsers(dest="baseline_sub", required=True)

    eg_bl_regen = eg_baseline_sub.add_parser(
        "regenerate", help="Regenerate baseline against current origin/main."
    )
    eg_bl_regen.add_argument("--json", action="store_true", default=False,
                             help="Emit JSON on success.")
    eg_bl_regen.set_defaults(func=cmd_eval_gate)

    eg_bl_show = eg_baseline_sub.add_parser("show", help="Print current baseline JSON.")
    eg_bl_show.add_argument("--json", action="store_true", default=False,
                            help="Emit raw JSON (default: human-readable).")
    eg_bl_show.set_defaults(func=cmd_eval_gate)

    sr = sub.add_parser(
        "spec-review",
        help="Pre-bind Facets + Council review of a spec document (Sonnet deep-reviewer runs by default).",
    )
    sr.add_argument("spec_path", help="Path to the spec markdown file.")
    sr.add_argument(
        "--council-voicing",
        default="gravitywell",
        choices=["local", "gravitywell", "haiku", "sonnet", "opus"],
        dest="council_voicing",
        help="Voicing for Mirror Council deliberation (default: gravitywell — owned 122B, zero paid spend; Sonnet fallback if GW is down).",
    )
    sr.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Total timeout in seconds (default: 1800).",
    )
    sr.add_argument(
        "--repo",
        default=None,
        dest="repo_override",
        help="Override repo parsed from spec frontmatter.",
    )
    sr.add_argument(
        "--no-facets",
        action="store_true",
        dest="no_facets",
        help="Skip Facets deliberation dispatch (for smoke or testing).",
    )
    sr.add_argument(
        "--authority",
        choices=["auto-merge", "advisory", "hold"],
        default=None,
        help="Override spec's stated authority for Facets dispatch gating.",
    )
    sr.add_argument(
        "--no-sonnet-reviewer",
        action="store_true",
        dest="no_sonnet_reviewer",
        help=(
            "Skip the reference-only Sonnet deep-review leg (on by default for "
            "advisory/hold; opt out for faster/cheaper runs)."
        ),
    )
    sr.add_argument(
        "--compare-opus",
        action="store_true",
        dest="compare_opus",
        help=(
            "Deprecated/no-op: the Sonnet reference leg now runs by default; "
            "this flag is kept for back-compat (sunset 90 days after merge)."
        ),
    )
    sr.add_argument(
        "--facets-operator",
        choices=["haiku", "sonnet", "opus", "qwen"],
        default="haiku",
        dest="facets_operator",
        help=(
            "Model operator for Facets personas and synthesis (default: haiku). "
            "Pass sonnet for a more thorough technical pass."
        ),
    )
    sr.set_defaults(func=cmd_spec_review)

    # ------------------------------------------------------------------
    # backcaster — inverse goal-state decomposition
    # ------------------------------------------------------------------
    bc = sub.add_parser(
        "backcaster",
        help="Backcaster v0: decompose a goal-state into required components.",
    )
    bc.add_argument(
        "goal_file",
        metavar="GOAL_FILE",
        help="Path to the goal-state markdown file.",
    )
    bc.add_argument(
        "--axes",
        default=None,
        metavar="ax1,ax2,...",
        help=(
            "Comma-separated axes to decompose (default: all 6). "
            "Values: psychological,material,infrastructural,governance,social-norm,economic"
        ),
    )
    bc.add_argument(
        "--corpus",
        default=None,
        metavar="PATH",
        help="Path to corpus directory for Synapse retrieval (default: library/civic-theory).",
    )
    bc.add_argument(
        "--scenarios",
        default=None,
        metavar="RUN_IDS",
        help="Comma-separated civic-sim run IDs (v0 no-op stub).",
    )
    bc.add_argument(
        "--model",
        default="qwen",
        choices=["qwen", "sonnet", "opus"],
        help="LLM model to use (default: qwen).",
    )
    bc.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="Override run output directory (default: /srv/lapis/backcaster/runs/<timestamp>-<slug>/).",
    )
    bc.add_argument(
        "--allow-degraded",
        action="store_true",
        default=False,
        help=(
            "Proceed even when grounding services (Synapse, mem) are unreachable. "
            "Gap analysis will be model-reasoned rather than corpus-grounded; "
            "a DEGRADED banner is stamped at the top of roadmap.md."
        ),
    )
    bc.set_defaults(func=cmd_backcaster)

    # ------------------------------------------------------------------
    # decisions-export — export recent mem entries as a markdown artifact
    # ------------------------------------------------------------------
    de = sub.add_parser(
        "decisions-export",
        help="Export recent decision/feedback/pattern mem entries as a single markdown artifact",
    )
    de.add_argument("--since", default="7d", help="Window spec: `Nd` or `YYYY-MM-DD` (default: 7d)")
    de.add_argument(
        "--include", default="decision,feedback,pattern",
        help="Comma-separated types to include (default: decision,feedback,pattern)",
    )
    de.add_argument("--exclude", default="", help="Comma-separated types to exclude (subtracted from --include)")
    de.add_argument("--tag", default="", help="Optional tag filter (entries must contain this tag)")
    de.add_argument("--out", default=None, help="Write to file (default: stdout)")
    de.set_defaults(func=cmd_decisions_export)

    lw = sub.add_parser(
        "local-witness",
        help="Local-reviewer witness tools (agreement stats, divergence history).",
    )
    lw_sub = lw.add_subparsers(dest="local_witness_sub")
    lw_sub.required = True
    lw_stats = lw_sub.add_parser(
        "stats",
        help="Print agreement matrix + latency percentiles for local-reviewer-witness observations.",
    )
    lw_stats.add_argument("--repo", default=None, help="Filter to a specific repo (default: all)")
    lw_stats.add_argument("--since", default=None, help="Filter to observations since YYYY-MM-DD")
    lw.set_defaults(func=cmd_local_witness)

    # ------------------------------------------------------------------
    # compact — compact dispatched records for old landed targets
    # ------------------------------------------------------------------
    cmp = sub.add_parser(
        "compact",
        help="Compact dispatched records for old landed targets.",
    )
    cmp.add_argument(
        "--days", type=int, default=30,
        help="Eligibility threshold: targets landed >= N days ago (default: 30, min: 1)",
    )
    cmp.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print eligible targets and their current dispatch record size without writing.",
    )
    cmp.set_defaults(func=cmd_compact)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
