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
    deploy-status [--json]
    pause <target_id> [--reason TEXT]
    resume <target_id>
    list
    land <target_id> [--dry-run]
    friction list [--target ID] [--repo REPO] [--limit N] [--json] [--exclude-suspect]
    friction backfill-provenance [--apply]
    spec-id-backfill [--apply]
    brief --period {morning,afternoon,weekly,live} [--week YYYY-Www]
    brief-resolve <target_id> <option_id>
    trajectory-rollup --rebuild-index
    trajectory-rollup --period per-target [--target TID | --all]
    trajectory-rollup --period weekly [--week YYYY-Www]
    trajectory-rollup --period monthly [--month YYYY-MM]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore

from agents_core.targets import TargetStore

# Package imports work because the CLI is launched via `python -m lapis_pm.cli`.
from . import episodic, brief, pm_core, land, chain as chain_mod, steer as steer_mod
from . import node_identity
from . import signed_directive
from .spec_review import (
    _parse_spec_authority_text,
    _parse_spec_verification_text_guarded as _parse_spec_verification_text,
)
from .router_portfolio import emit_decision_kickoff, emit_decision_land
from .backcaster.cli import cmd_backcaster, cmd_backcaster_quest
from .research_quest import cmd_research_quest
from .scout.cli import cmd_scout
from .prior_art_scout.cli import (
    cmd_prior_art_scout_run,
    cmd_prior_art_scout_status,
    cmd_prior_art_scout_reset_hopeless,
)

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

    # Validate --adopt-pr early so we fail before creating or mutating state.
    _adopted_pr_number: int | None = getattr(args, "adopt_pr", None)
    _adopted_head_branch: str = ""
    if _adopted_pr_number is not None:
        _adopt_repo_name, _adopt_owner = pm_core._repo_owner(args.repo)
        try:
            from agents_core.forgejo import get_pr as _get_pr
            _adopted_pr_data = _get_pr(_adopt_repo_name, _adopted_pr_number, owner=_adopt_owner)
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
        if _pr_base_repo and _pr_base_repo != _adopt_repo_name:
            print(
                f"ERROR: PR #{_adopted_pr_number} base repo is {_pr_base_repo!r} "
                f"but --repo resolves to {_adopt_repo_name!r}",
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

    # Probe that the repo resolves under its owner before mutating any state (Defect B guard).
    _bind_repo_name, _bind_owner = pm_core._repo_owner(args.repo)
    try:
        from agents_core.forgejo import get_open_prs as _forgejo_probe
        _forgejo_probe(_bind_repo_name, owner=_bind_owner)
    except Exception as _probe_err:
        import httpx as _httpx
        if isinstance(_probe_err, _httpx.HTTPStatusError) and _probe_err.response.status_code == 404:
            _resolved = f"{_bind_owner or 'Erah'}/{_bind_repo_name}"
            print(
                f"ERROR: repo '{_resolved}' not found on Forgejo. "
                f"If this repo lives under an org, bind with the full "
                f"'<org>/{_bind_repo_name}' (e.g. 'lapis/{_bind_repo_name}'), "
                f"not the bare name. Bind aborted; nothing was written.",
                file=sys.stderr,
            )
            return 2
        # Connectivity error (not 404) — warn and proceed so offline BRIX does not block binds.
        print(
            f"WARNING: could not verify repo existence (Forgejo unreachable): "
            f"{_probe_err}; proceeding",
            file=sys.stderr,
        )

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

    _spec_authority = _parse_spec_authority_text(spec_body)
    if args.authority is not None and _spec_authority is not None and args.authority != _spec_authority:
        print(
            f"ERROR: spec declares **Authority:** {_spec_authority} but --authority "
            f"{args.authority} was passed. Pass --authority {_spec_authority} to match "
            "the spec, or fix the spec's Authority header if the spec is wrong. "
            "Bind aborted; nothing was written.",
            file=sys.stderr,
        )
        return 2
    if args.authority is None:
        args.authority = _spec_authority if _spec_authority is not None else "advisory"

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

    # Resolve pm_verification: --verification flag overrides spec field; absent => pm-live-test
    _spec_verif = _parse_spec_verification_text(spec_body)
    _verif_flag = getattr(args, "verification", None)
    effective_verification = _verif_flag if _verif_flag is not None else _spec_verif
    target.data["pm_verification"] = effective_verification

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
    print(f"Bound {args.target_id} → repo={args.repo}, authority={args.authority}, "
          f"verification={effective_verification}")
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
                task_id = pm_core.force_dispatch(
                    tid, "fixer", intent,
                    extra_episodic_tags=[
                        f"pm:chain-group={chain_group}",
                        "pm:chain-initial",
                    ],
                    episodic_message=f"Chain initial dispatch: fixer\nIntent: {intent}",
                )
                chain_mod.emit_chain_event(
                    chain_group, "dispatch", tid,
                    details={"task_id": task_id, "initial": True},
                )
                print(f"  Dispatched leg {tid}: fixer task_id={task_id}")

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
    pm_core.clear_outstanding_brief(args.target_id, reason="unbind")
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


def cmd_clear_reviewer_attempts(args) -> int:
    """Sole escape hatch for a reviewer-attempt-ceiling pause. Deliberately
    cheap: target_id only, no confirmation prompt, no metadata (Erah's
    ruling — ceremony here inverts the point of a cheap, obvious clear).
    Does NOT resume the target; follow with `lapis-pm resume <target_id>`.
    """
    cleared = pm_core.clear_reviewer_attempts(args.target_id)
    print(f"Cleared {cleared} reviewer-attempt counter(s) for {args.target_id}")
    print(f"If paused, resume separately with: lapis-pm resume {args.target_id}")
    return 0


def cmd_clear_dispatch(args) -> int:
    """Leg 3 escape hatch for a stale-pending dispatch record. Deliberately
    cheap: target_id only, no confirmation prompt. Flips every pending
    dispatch record to failed; never touches the intention registry, which
    is a separate wedge — see the reminder printed below unconditionally.
    """
    cleared = pm_core.clear_dispatch(args.target_id)
    print(f"Cleared {cleared} pending dispatch record(s) for {args.target_id}")
    # Unconditional, never suppressed — clearing a dispatch record and
    # composting an intention are two independent pieces of state; both
    # must be cleared or an operator can conclude the target is free when
    # it is not (intention-registry-orphan-wedge).
    print(
        "Reminder: this does not touch the intention registry. If this "
        "target also has a stuck intention, clear it separately."
    )
    return 0


def cmd_stall_check(args) -> int:
    """Dedicated stall checker (attestation-contract-v0 leg 2, D6a)."""
    from . import stall_check
    return stall_check.main()


def cmd_autopilot(args) -> int:
    """The pipeline autopilot (the prodder) — lapis-pm-pipeline-autopilot-v0.

    ``lapis-pm autopilot sweep``: run one per-tick sweep (PERCEIVE -> classify
    -> UNBLOCK / ADJUDICATE / RE-FIRE -> ESCALATE). The daemon tick is the
    sole dispatch/merge actor; the sweep only writes state + emits dossiers +
    escalates. Shadow-mode-first (``LAPIS_PM_AUTOPILOT=shadow`` default):
    proposals are observed only. ``off`` halts the sweep.

    ``lapis-pm autopilot report``: the shadow-review surface (the Nudge Log —
    the narrative render of the last N shadow proposals + reversals).

    ``lapis-pm autopilot liveness``: the BRIX-side liveness backstop (D5) —
    check the heartbeat freshness and page/record when stale (catches a dead
    prodder while GW sleeps).
    """
    from . import autopilot
    from . import pm_core

    if args.autopilot_cmd == "sweep":
        if args.target:
            target_ids = [args.target]
        else:
            target_ids = autopilot._bounded_target_ids()
        result = autopilot.run_sweep(
            target_ids, now=None, tick_id=args.tick_id or "")
        if args.json:
            import json as _json
            print(_json.dumps(result, ensure_ascii=False, default=str))
        else:
            print(f"[autopilot] mode={result['mode']} "
                  f"halted={result['halted']} "
                  f"dossiers_this_tick={result.get('dossiers_this_tick', 0)}")
            for t in result["targets"]:
                print(f"  {t.get('target_id')}: {t.get('action', t.get('skipped'))} "
                      f"(state={t.get('state_code', '-')})")
        return 0

    if args.autopilot_cmd == "report":
        mem = pm_core._mem()
        print(autopilot.report(
            mem, target_id=args.target, limit=args.limit))
        return 0

    if args.autopilot_cmd == "liveness":
        mem = pm_core._mem()
        result = autopilot.check_prodder_liveness(mem)
        if args.json:
            import json as _json
            print(_json.dumps(result, ensure_ascii=False, default=str))
        else:
            print(f"[autopilot liveness] {result.get('action')} "
                  f"(age={result.get('age_s', 'n/a')}s)")
        return 0

    # Unknown subcommand (argparse enforces choices, so this is defensive).
    print(f"ERROR: unknown autopilot subcommand: {args.autopilot_cmd}",
          file=sys.stderr)
    return 2


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
    print(f"  pm_verification: {t.data.get('pm_verification', 'pm-live-test (default)')}")
    print(f"  paused:        {t.paused}"
          + (f" (reason: {t.paused_reason})" if t.paused and t.paused_reason else ""))
    cursor = pm_core.get_cursor(t.id)
    print(f"  cursor:        {cursor}")
    dispatched = pm_core.load_dispatched(t.id)
    pending = [d for d in dispatched if d.get("status") == "pending"]
    print(f"  dispatched:    {len(dispatched)} total, {len(pending)} pending")
    for d in pending:
        age = pm_core._dispatch_age(d.get("ts"))
        age_str = pm_core._format_age(age) if age is not None else "?"
        print(f"    pending: gpu_id={d.get('gpu_id', 'unknown')} "
              f"agent_type={d.get('agent_type', 'unknown')} age={age_str}")
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
                  f"cycle {review_state['cycle']}/{budget} (local reviewer, {mode})")
            verdict = review_state.get("verdict", "pending")
            issues = review_state.get("issues", 0)
            if verdict != "pending":
                print(f"  last-verdict:  {verdict}"
                      + (f", {issues} issue(s)" if verdict == "fixable" else ""))

            # Panel-health line (lapis-pm-panel-leg-survival-v0, Change 4):
            # starved count over the last 5 reviewer verdicts for the active
            # PR, with the union of legs_down. Renders only when the
            # active-review block renders (no open PR / Forgejo down -> no
            # line); a PR with no completed verdicts -> no line.
            panel = _pm._panel_health_summary(t.id, review_state["pr_number"])
            if panel:
                legs_text = ", ".join(panel["legs"]) or "-"
                print(f"  panel:         {panel['starved']}/{panel['total']} starved (legs: {legs_text})")

        # R3 (lapis-pm-reviewer-defer-backoff-v0): rendered independently of
        # review_state above — during an active backoff the failed attempt
        # has no pending reviewer and no verdict yet, exactly the shape
        # _active_review_state skips (see its docstring). Prefixed
        # "reviewer-backoff:" (not "paused:") — a temporary retreat must not
        # read as a defeat.
        backoff = _pm._active_reviewer_backoff(t.id, open_prs)
        if backoff:
            print(f"  reviewer-backoff: attempt {backoff['infra_count']}/{backoff['infra_budget']} "
                  f"reason={backoff['last_infra_reason']} next_retry_at={backoff['next_retry_at']}")
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


def cmd_deploy_status(args) -> int:
    """Render the deploy-inventory reconciler's last status snapshot.

    (lapis-pm-deploy-inventory-reconciler-v0) — the human-visible surface for
    the derived deploy-graph reconciliation; written by the tick loop.
    """
    from . import deploy_inventory
    try:
        status = json.loads(deploy_inventory._STATUS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        print(
            "No deploy-inventory status yet — run a tick or wait for the next "
            "reconcile pass.",
            file=sys.stderr,
        )
        return 1
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    print(deploy_inventory.render_deploy_status(status))
    return 0


def _target_to_json_dict(t) -> dict:
    dispatched = pm_core.load_dispatched(t.id)
    pending = [d for d in dispatched if d.get("status") == "pending"]
    pending_detail = []
    for d in pending:
        age = pm_core._dispatch_age(d.get("ts"))
        pending_detail.append({
            "gpu_id": d.get("gpu_id"),
            "agent_type": d.get("agent_type"),
            "age_s": int(age.total_seconds()) if age is not None else None,
        })
    return {
        "target_id": t.id,
        "title": t.title,
        "pm_repo": t.pm_repo,
        "pm_authority": t.pm_authority,
        "pm_verification": t.data.get("pm_verification", "pm-live-test"),
        "paused": t.paused,
        "cursor": pm_core.get_cursor(t.id),
        "dispatched_total": len(dispatched),
        "dispatched_pending": len(pending),
        "dispatched_pending_detail": pending_detail,
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


def cmd_friction_list(args) -> int:
    """Operator/debug read of the friction capture queue — not the human-tongue
    ratify surface (that's the Loupe Desk gem, Unit 3). Silent-gap records are
    rendered as a visibly distinct class, never interleaved as reported friction.
    provenance_suspect records (D3 backfill) render with a visible marker and
    are excluded entirely when --exclude-suspect is passed.
    """
    records = pm_core.read_friction_records(
        target_id=getattr(args, "target", None),
        repo=getattr(args, "repo", None),
        limit=getattr(args, "limit", None),
    )
    if getattr(args, "exclude_suspect", False):
        records = [r for r in records if not r.get("provenance_suspect")]
    if getattr(args, "json", False):
        print(json.dumps(records))
        return 0
    if not records:
        print("(no friction records)")
        return 0
    for r in records:
        prov = r.get("provenance") or {}
        ts = prov.get("captured_at", "-")
        tgt = prov.get("target_id", "-")
        repo = prov.get("repo", "-")
        pr = prov.get("pr", "-")
        header = f"{ts}  target={tgt} repo={repo} pr={pr}"
        if r.get("record_source") == "silent-gap":
            print(f"{header}  [SILENT-GAP] derived_confidence={r.get('derived_confidence')}")
            continue
        if r.get("provenance_suspect"):
            header += f"  [PROVENANCE-SUSPECT canonical_task_id={r.get('canonical_task_id')}]"
        print(header)
        print(f"    obstacle:    {r.get('obstacle', '')}")
        if r.get("path_taken"):
            print(f"    path_taken:  {r['path_taken']}")
        if r.get("artifact"):
            print(f"    artifact:    {r['artifact']}")
        if r.get("cost_hint") or r.get("confidence"):
            print(f"    cost_hint={r.get('cost_hint')}  confidence={r.get('confidence')}")
    return 0


def cmd_friction_backfill(args) -> int:
    """One-shot D3 backfill: tag (never delete) queue records whose content
    matches an earlier record under a different task_id — the mechanism by
    which a fixer's inherited friction.json entries picked up false
    provenance before D1+D2 closed the hole. Manual-only: invoked by hand,
    once, post-deploy — no scheduled artifact calls this.
    """
    apply = getattr(args, "apply", False)
    try:
        result = pm_core.friction_backfill_provenance(apply=apply)
    except pm_core._FrictionLockTimeout as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    mode = "APPLIED" if apply else "DRY-RUN"
    print(
        f"[{mode}] total_records={result['total_records']} "
        f"cross_attribution_groups={result['groups']} tagged={result['tagged']}"
    )
    if not apply and result["tagged"]:
        print("Re-run with --apply to write tags to the queue.")
    return 0


def cmd_spec_id_backfill(args) -> int:
    """spec-corpus-id-backfill-v0: backfill `spec_id: <stem>` into
    /srv/lapis/planning/specs/*.md frontmatter for specs that lack it. Dry-run by
    default; --apply writes. Report always written; see D4."""
    import datetime as _datetime

    from . import spec_id_backfill

    apply = getattr(args, "apply", False)
    date_str = _datetime.date.today().isoformat()
    try:
        report = spec_id_backfill.run_backfill(apply=apply, date_str=date_str)
    except (spec_id_backfill.DuplicateStemError, spec_id_backfill.DuplicateSpecIdValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    mode = "APPLIED" if apply else "DRY-RUN"
    print(
        f"[{mode}] scanned={report.scanned} "
        f"modified={len(report.modified)} "
        f"already_carrying={len(report.already_carrying)} "
        f"skipped={len(report.skipped)} "
        f"mismatches={len(report.mismatches)}"
    )
    print(f"report: {report.report_path}")
    if not apply and report.modified:
        print("Re-run with --apply to write spec_id: into the listed candidates.")
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
    merged = pm_core._merged_pr_numbers_observed(args.target_id)
    pm_core.close_resolved_debt_for_target(
        args.target_id,
        resolved_by_pr=max(merged) if merged else None,
        ground="manual land",
    )
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


def cmd_attestation_check(args) -> int:
    """night-deploy-manifest-attestation-v0: the pre-PR / manual attestation
    entry. Runs the deploy-manifest completeness checker (lapis_pm/attestation)
    and returns the exit code the DAG node's success_predicate maps (0 clean /
    2 held / 3 failed). For pre-PR use — the DAG node itself invokes this
    module as a module from the /srv/git/lapis-pm deploy clone."""
    from . import attestation
    return attestation.main([
        *(["--plan", args.plan] if args.plan else []),
        *(["--waivers", args.waivers] if args.waivers else []),
        *(["--ledger", args.ledger] if args.ledger else []),
        *(["--src", args.src] if args.src else []),
        *(["--dest", args.dest] if args.dest else []),
        *([] if not args.no_emit else ["--no-emit"]),
    ])


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

    # Adjudication record (lapis-pm-autonomy-actuator-v0, Design 5): written
    # alongside the ratification-outcome entry in the SAME call, idempotent per
    # (tid, pr, outcome). The fork block is copied from the pm:brief-options
    # sibling when present, else null (auditable, never matches). The
    # ghost-ratify marker (rev 4) records whether this CLI ran interactively.
    adjudication_key = None
    try:
        from . import precedent as _pc

        # Resolve the PR number: --pr flag > adopted_pr_number > prior key.
        pr_number = getattr(args, "pr", None)
        if pr_number is None:
            try:
                store = pm_core.TargetStore()
                t = store.get(target_id)
                if t is not None:
                    pr_number = t.data.get("adopted_pr_number")
            except Exception:
                pr_number = None
        if pr_number is None and prior_event_id:
            m = re.search(r"(?:^|/)(\d+)$", prior_event_id or "")
            if m:
                pr_number = int(m.group(1))
        if pr_number is None:
            pr_number = 0

        repo = ""
        try:
            store = pm_core.TargetStore()
            t = store.get(target_id)
            if t is not None:
                repo = t.pm_repo or ""
        except Exception:
            repo = ""

        # fork copied from the pm:brief-options sibling (Design 5); null when
        # missing/unparseable (briefs predating this spec, non-PR briefs).
        mem = pm_core._mem()
        fork = _pc._fork_from_options_sibling(mem, target_id)

        adjudication_key = _pc.write_adjudication(
            mem,
            target_id=target_id,
            pr_number=pr_number,
            outcome=outcome,
            repo=repo,
            fork=fork,
            intent=intent or "",
            citations=[key],
            invoked_interactive=sys.stdin.isatty(),
        )
    except Exception as e:
        # Never block the ratify itself on an adjudication write failure.
        print(f"WARNING: adjudication record write failed: {e}", file=sys.stderr)

    if as_json:
        print(json.dumps({"key": key, "adjudication_key": adjudication_key}))
    else:
        print(f"Ratified {target_id} ({outcome}): {key}")
        if adjudication_key:
            print(f"Adjudication record: {adjudication_key}")
    return 0


def cmd_backstop_sweep(args) -> int:
    """Run one repo's autonomy-backstop sweep step (U3.4).

    D4 (lapis-pm-reviewer-leg-repair-v0): this is the entry point the
    dedicated `lapis-pm-backstop.timer` (15m cycle, `TimeoutStartSec=1200s`)
    invokes. The sweep is serialized by the persistent `sweep_started_ts`
    stamp (written at sweep START, 15m minimum interval) + the dedicated
    oneshot unit (one at a time host-wide) — NOT by the process-local
    in-flight guard. Class-dependent eligibility (docs/test-only 24h,
    running-path on new demotion) is checked post-selection. Output
    discipline mirrors `lapis-pm land --dry-run`: human-readable,
    machine-greppable.
    """
    from . import autonomy_backstop as _bs

    mem = pm_core._mem()
    now = _now_iso()
    outcome = _bs.sweep(mem, now=now, repo_cursor=getattr(args, "repo", None))

    print(
        f"backstop-sweep: repo={outcome.get('repo')} "
        f"status={outcome.get('status')} "
        f"suspect_prs={outcome.get('suspect_prs')} "
        f"demoted={len(outcome.get('demoted_classes') or [])} "
        f"change_class={outcome.get('change_class')} "
        f"sweep_count={outcome.get('sweep_count')} "
        f"ts={outcome.get('ts')}"
    )
    for pr in outcome.get("suspect_prs") or []:
        print(f"  SUSPECT: PR #{pr} (suspect, not proven — coarse attribution)")
    for key in outcome.get("demoted_classes") or []:
        print(f"  DEMOTED: {key}")
    for key in outcome.get("findings") or []:
        print(f"  FINDING: {key}")
    if outcome.get("status") == "red":
        # U3.2: open the HIGH regression brief (suspect, not proven).
        # D4: brief cooldown — one HIGH per (repo, suspect PR) per day.
        suspect_pr = (outcome.get("suspect_prs") or [None])[0]
        _bs.open_regression_brief(
            outcome.get("repo") or "?", suspect_pr,
            outcome.get("main_failures") or [],
        )
    return 0


def cmd_autonomy_promote(args) -> int:
    """Manual promotion path for a fork class (U3.3, Design 7/8).

    Demotion-cooldown guard: refuses a class with regression_count > 0 and
    last_regression_ts within 14 days without --force; with regression_count
    > 2 within 14 days, even --force requires a logged --justification.
    """
    from . import autonomy_backstop as _bs

    change_classes = [c.strip() for c in (args.change_classes or "").split(",") if c.strip()]
    mem = pm_core._mem()
    result = _bs.promote(
        mem,
        repo=args.repo,
        verdict_class=args.verdict_class,
        change_classes=change_classes,
        loc_bucket=args.loc_bucket,
        force=args.force,
        justification=args.justification,
    )
    if result.get("promoted"):
        print(
            f"promoted: {result.get('key')} by={result.get('promoted_by')} "
            f"ts={result.get('ts')}"
        )
        return 0
    reason = result.get("reason", "unknown")
    print(f"promotion refused: {reason} (key={result.get('key')})", file=sys.stderr)
    return 1


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_directive(args) -> int:
    """Emit a signed human:directive comment under the pm-review agent key
    (spec lapis-pm-signed-directive-acceptance-v0, D2). Replaces hand-writing
    raw JSONL under an unbacked author:"Erah" claim — capture is always
    permitted; acceptance is verified by the daemon at read time.

    lapis-pm directive <target_id> --content TEXT [--json]
    """
    target_id = args.target_id
    content = args.content
    as_json = getattr(args, "json", False)

    try:
        result = signed_directive.emit_signed_directive(target_id, content)
    except Exception as e:
        print(f"ERROR: emit_signed_directive failed: {e}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps({
            "manifest_hash": result["manifest_hash"],
            "pubkey_id": result.get("pubkey_id"),
            "comment_id": result["comment"].id,
        }))
    else:
        print(f"Directive recorded for {target_id}: deposit={result['manifest_hash']}")
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


def cmd_tts_episode(args) -> int:
    """Handle `lapis-pm tts-episode [--period morning] [--voice both|piper|kokoro]`.

    Renders the latest existing brief for `--period` to spoken WAV via the
    Piper/Kokoro bakeoff (gardener-tts-episode-v0, Unit 3a). Delivery is
    vault-drop + one Pushover ping; see lapis_pm.tts_episode for the
    per-engine fallback and Piper latency soft-fail rules.
    """
    from agents_core.room_paths import room_path
    from . import tts_episode

    period = args.period
    voices = list(tts_episode.VOICES) if args.voice == "both" else [args.voice]

    briefs_root = room_path("briefs")
    brief_path = briefs_root / f"latest-{period}.md"
    body = brief_path.read_text(encoding="utf-8") if brief_path.exists() else ""
    if not body.strip():
        print(
            f"no {period} brief to synthesize — run `lapis-pm brief --period {period}` first",
            file=sys.stderr,
        )
        return 1

    out_dir = briefs_root / "audio"
    outcome = tts_episode.run_episode_bakeoff(body, voices=voices, out_dir=out_dir, period=period)

    for e in outcome.engines:
        if e.path is not None:
            print(f"{e.voice}: {e.path} ({e.elapsed_sec:.1f}s)")
        elif e.soft_failed:
            print(
                f"{e.voice}: soft-failed — render exceeded "
                f"{tts_episode.PIPER_LATENCY_SOFT_FAIL_SEC:.0f}s ({e.elapsed_sec:.1f}s), withheld"
            )
        else:
            print(f"{e.voice}: failed — {e.error}")

    if not outcome.ok:
        print("tts-episode: all requested engines failed — no episode delivered", file=sys.stderr)
        return 1

    print(f"Pushover sent: {outcome.notified}")
    return 0


def cmd_tts_episode_publish(args) -> int:
    """Handle `lapis-pm tts-episode-publish [--period morning]`.

    Durable private podcast feed publish (Unit 3b, gardener-tts-feed-v0):
    Kokoro-primary / Piper-automatic-fallback render (reusing
    tts_episode.run_episode_bakeoff read-only), MP3 transcode, stateless
    feed.xml regeneration, and retention prune. See lapis_pm.tts_feed.
    """
    from . import tts_feed

    result = tts_feed.publish_episode(period=args.period)

    if not result.ok:
        print(f"tts-episode-publish: FAILED — {result.error}", file=sys.stderr)
        return 1

    note = " (idempotent no-op — episode already published)" if result.skipped_idempotent else ""
    print(f"tts-episode-publish: {result.engine} -> {result.mp3_path}{note}")
    print(f"feed: {result.feed_url}")
    if result.pruned:
        print(f"pruned {len(result.pruned)} old file(s)")
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
            "the reference leg is opt-in now, use --with-reference-reviewer "
            "(sunset 90 days after merge).",
            file=sys.stderr,
        )
    # --no-reference-reviewer / --no-sonnet-reviewer are harmless no-ops now that the
    # Empiricist leg is off by default (U3a) — they must not error and must not invert
    # to mean "turn it on". Emit a deprecation note but otherwise ignore them; only
    # --with-reference-reviewer can enable the leg.
    if getattr(args, "no_reference_reviewer", False):
        print(
            "[spec-review] WARNING: --no-reference-reviewer is deprecated and a no-op; "
            "the Empiricist leg is off by default now (sunset 90 days after merge, "
            "2026-09-05).",
            file=sys.stderr,
        )
    if getattr(args, "no_sonnet_reviewer", False):
        print(
            "[spec-review] WARNING: --no-sonnet-reviewer is deprecated and a no-op; "
            "the Empiricist leg is off by default now (sunset 90 days after merge, "
            "2026-09-05).",
            file=sys.stderr,
        )
    no_reference_reviewer = (
        getattr(args, "no_reference_reviewer", False)
        or getattr(args, "no_sonnet_reviewer", False)
    )
    reference_reviewer = getattr(args, "with_reference_reviewer", False) and not no_reference_reviewer
    try:
        brief = run_spec_review(
            spec_path=spec_path,
            council_voicing=args.council_voicing,
            timeout_s=args.timeout,
            repo_override=args.repo_override,
            authority=getattr(args, "authority", None),
            dispatch_facets=not getattr(args, "no_facets", False),
            reference_reviewer=reference_reviewer,
            facets_operator=getattr(args, "facets_operator", "haiku"),
            with_gw=getattr(args, "with_gw", False),
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


def cmd_facets_gw_eval(args) -> int:
    """Handle `lapis-pm facets-gw-eval` subcommand."""
    from . import facets_gw_eval as fge

    try:
        result = fge.run_eval()
        if result is None:
            print("ERROR: eval failed (check logs for details)", file=sys.stderr)
            return 1

        # Print summary
        print("\n" + "=" * 70)
        print("Facets-on-GW Load Eval - Summary")
        print("=" * 70)
        verdict = result.verdict
        print(
            f"Overall: {'🟢 PASS' if verdict.get('overall_pass') else '🔴 FAIL'}"
        )
        print(f"  Latency p95 delta (B-A): {verdict.get('latency_p95_delta_s', 0):.2f}s "
              f"({'✓' if verdict.get('latency_p95_delta_pass') else '✗'})")
        print(f"  Degrade-rate: {verdict.get('arm_b_degrade_rate_pct', 0):.1f}% "
              f"({'✓' if verdict.get('degrade_rate_pass') else '✗'})")
        print(f"  Quality equivalent: {verdict.get('quality_equivalent_pct', 0):.0f}% "
              f"({'✓' if verdict.get('quality_pass') else '✗'})")
        if verdict.get('variance_fragile'):
            print(f"  ⚠️  Variance FRAGILE (wild {verdict.get('variance_ratio', 1.0):.2f}x clean p95)")
        print()
        print(f"Report: {result.report_path}")
        print("=" * 70)

        if args.json:
            print(json.dumps({
                "verdict": verdict,
                "report_path": result.report_path,
            }, ensure_ascii=False, indent=2))

        return 0 if verdict.get('overall_pass') else 1

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_functional_critic(args) -> int:
    """Handle `lapis-pm functional-critic` subcommand (DoD integration smoke).

    Usage: python3 -m lapis_pm.cli functional-critic <repo> <pr_number> <spec_path>

    Checks out PR head, runs pytest, calls GW-122B for AC coverage assessment,
    and prints the verdict JSON to stdout.
    """
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from . import functional_critic as fc

    repo = args.repo
    pr_number = args.pr_number
    spec_path = args.spec_path

    # Load spec text
    try:
        spec_text = open(spec_path).read()
    except OSError as exc:
        print(f"ERROR: cannot read spec file {spec_path}: {exc}", file=sys.stderr)
        return 1

    # Fetch head SHA from Forgejo
    try:
        from agents_core.forgejo import get_open_prs as _get_open_prs
        repo_name = repo.split("/")[-1]
        open_prs = _get_open_prs(repo_name) or []
        pr = next((p for p in open_prs if p.get("number") == pr_number), None)
        if pr is None:
            print(f"ERROR: PR #{pr_number} not found in open PRs for {repo_name}", file=sys.stderr)
            return 1
        head_sha = (pr.get("head") or {}).get("sha", "")
        if not head_sha:
            print(f"ERROR: no head SHA for PR #{pr_number}", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"ERROR: cannot fetch PR info: {exc}", file=sys.stderr)
        return 1

    run_id = f"cli-fc-{repo_name}-pr{pr_number}"
    print(f"Running functional critic: repo={repo_name} pr={pr_number} sha={head_sha[:8]} run_id={run_id}", file=sys.stderr)

    verdict = fc.run_functional_critic(
        target_id=f"cli-{repo_name}",
        pr_number=pr_number,
        head_sha=head_sha,
        repo=repo_name,
        spec_text=spec_text,
        run_id=run_id,
    )

    print(json.dumps(verdict, ensure_ascii=False, indent=2))

    if verdict.get("overall") == "pass":
        return 0
    elif verdict.get("overall") == "unverifiable":
        return 2
    else:
        return 1


def cmd_batched_fixer_eval(args) -> int:
    """Handle `lapis-pm batched-fixer-eval` subcommand."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from . import batched_fixer_eval as bfe
    from dataclasses import asdict

    phase = args.phase
    run_id = args.run_id
    mock_mode = args.mock
    target_sha = getattr(args, "target_sha", None)
    repo_only = getattr(args, "repo_only", None)

    try:
        result = bfe.run_eval(
            phase=phase, run_id=run_id, mock_mode=mock_mode,
            target_sha=target_sha, repo_only=repo_only,
        )
    except bfe.TargetShaNotResolvedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except bfe.CorpusPowerFloorUnmetError as e:
        # AC5: manifest is already written with the achieved counts — this reserved exit
        # code means "built, but under floor", never shared with any other failure.
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except bfe.NonDiscriminatesHarvestError as e:
        print(f"warning: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    try:
        if result is None:
            if phase == "build-corpus":
                print("Corpus built and frozen. PM must ratify before run phase.", file=sys.stderr)
                return 0
            else:
                print("ERROR: eval failed (check logs for details)", file=sys.stderr)
                return 1

        # Print summary
        print("\n" + "=" * 70)
        print("Batched-Fixer Patch-Quality Eval - Summary")
        print("=" * 70)
        print(f"Run ID: {result.run_id}")
        print(f"Generated: {result.generated_at}")
        print(f"Served model: {result.served_model_id} ({result.served_model_backend})")
        print()
        if result.tier_metrics:
            for tier in ["T1", "T2", "T3"]:
                if tier in result.tier_metrics:
                    print(f"{tier}:")
                    for n, metrics in result.tier_metrics[tier].items():
                        print(f"  N={n}: success={metrics.execute_success_rate:.1%} "
                              f"({metrics.execute_success_ci[0]:.1%}-{metrics.execute_success_ci[1]:.1%}), "
                              f"→ {metrics.routing_recommendation}")
        print()
        if result.report_path:
            print(f"Report: {result.report_path}")
        print("=" * 70)

        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_code_oracle_run(args) -> int:
    """Handle `lapis-pm code-oracle-run` subcommand.

    Thin wiring: loads one fixture from the corpus by id and delegates to
    the existing `run_code_oracle_experiment` (real writeable agent iterates
    against a failing test in an isolated throwaway clone, judged by the
    independent oracle). This command performs no filesystem writes of its
    own - isolation is entirely `run_code_oracle_experiment`'s existing
    dedicated_clone/detached_worktree responsibility.
    """
    import subprocess

    from . import batched_fixer_eval as bfe
    from .code_oracle_run import run_code_oracle_experiment

    corpus = bfe.load_corpus()
    available_ids = [f"{f.repo}-{f.sha[:8]}" for f in corpus]
    fixture = next(
        (f for f, fid in zip(corpus, available_ids) if fid == args.fixture_id), None
    )
    if fixture is None:
        print(
            f"ERROR: unknown --fixture-id {args.fixture_id!r}. "
            f"Available ids: {available_ids}",
            file=sys.stderr,
        )
        return 1

    repo_path = Path(args.repo_path)
    if not (repo_path / ".git").exists():
        print(
            f"ERROR: --repo-path {repo_path} is not a valid git repo (no .git found).",
            file=sys.stderr,
        )
        return 1

    verify = subprocess.run(
        ["git", "cat-file", "-e", fixture.parent_sha],
        cwd=repo_path, capture_output=True,
    )
    if verify.returncode != 0:
        print(
            f"ERROR: fixture parent_sha {fixture.parent_sha} not found in "
            f"{repo_path} history.",
            file=sys.stderr,
        )
        return 1

    print(
        "code-oracle-run: this command performs no writes under "
        f"{repo_path}; isolation is delegated entirely to "
        "run_code_oracle_experiment's dedicated_clone/detached_worktree.",
        file=sys.stderr,
    )

    try:
        result = run_code_oracle_experiment(
            fixture=fixture,
            agent_backend_url=args.agent_backend_url,
            repo_path=repo_path,
            out_dir=Path(args.out_dir),
            run_id=args.run_id,
            max_steps=args.max_steps,
            timeout_s=args.timeout,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    result_path = Path(args.out_dir) / f"{result['run_id']}.json"
    print("\n" + "=" * 70)
    print("Code-Oracle Run - Summary")
    print("=" * 70)
    print(f"Run ID: {result['run_id']}")
    print(f"Fixture ID: {result['fixture_id']}")
    print(f"Agent backend URL: {result['agent_backend_url']}")
    print(f"Oracle outcome: {result['oracle_outcome']}")
    print(f"Apply status: {result['apply_status']}")
    print(f"Agent self-report: {result['agent_self_report']}")
    print(f"Result file: {result_path}")
    print("=" * 70)

    return 0


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


def cmd_bundle_autodispatch(args) -> int:
    """Handle `lapis-pm bundle-autodispatch [--dry-run] [--spec-dir PATH]`."""
    import logging as _logging
    from datetime import datetime, timezone
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from pathlib import Path as _Path
    from .bundle_autodispatch import reconcile

    # Timestamp injected here at the CLI edge — never generated inside the pure reconcile logic.
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    spec_dir = _Path(args.spec_dir) if args.spec_dir else None
    results = reconcile(
        spec_dir=spec_dir,
        dry_run=args.dry_run,
        gate_timeout_s=args.gate_timeout,
        run_ts=run_ts,
    )

    print(
        f"bundle-autodispatch: "
        f"bound={len(results['bound'])} "
        f"deferred={len(results['deferred'])} "
        f"skipped={len(results['skipped'])} "
        f"failed={len(results['failed'])} "
        f"faulted={len(results.get('faulted', []))} "
        f"salvaged={len(results.get('salvaged', []))}"
    )
    for entry in results["bound"]:
        prefix = "[dry-run] " if entry.get("dry_run") else ""
        print(f"  {prefix}BOUND: {entry['spec']} → repo={entry.get('repo', '?')}")
    for entry in results["deferred"]:
        print(f"  DEFERRED: {entry['spec']} ({entry.get('reason', '?')})")
    for entry in results["skipped"]:
        print(f"  SKIPPED: {entry['spec']} ({entry.get('reason', '?')})")
    for entry in results["failed"]:
        print(f"  FAILED: {entry['spec']} ({entry.get('reason', '?')})")
    for entry in results.get("faulted", []):
        print(f"  FAULTED: {entry['spec']} ({entry.get('ground', '?')[:120]}, retried={entry.get('retried')})")
    for entry in results.get("salvaged", []):
        print(
            f"  SALVAGED: {entry['spec']} "
            f"(dropped={len(entry.get('dropped_items', []))}, outcome={entry.get('outcome', '?')})"
        )
    for entry in results.get("triage", []):
        if entry.get("class") == "mechanical":
            print(
                f"  TRIAGE-MECHANICAL: {entry['spec']}/{entry['debt_id']} → "
                f"{entry.get('target_id', '?')} (bound={entry.get('bound')})"
            )
        else:
            print(f"  TRIAGE-FORK: {entry['spec']}/{entry['debt_id']} ({entry.get('reason', '?')})")

    # Exit 1 if any failed (bound-but-dead state), 0 otherwise. Faults are
    # expected transients (infra retried once then recorded) — never gate
    # the exit code (lapis-pm-bundle-autodispatch-enforce-v0 Design 1).
    return 1 if results["failed"] else 0


def cmd_hold_shadow_summary(args) -> int:
    """Handle `lapis-pm hold-shadow-summary`.

    lapis-pm-hold-shadow-observer-v0's Thursday summary pass: groups the
    accumulated /srv/lapis/hold-shadow/*.jsonl records by proposed classification
    and deposits exactly one Desk gem. Runs OUT of tick, off its own systemd
    timer (systemd/lapis-hold-shadow-summary.timer, ~06:45 America/Los_Angeles)
    — never called from the tick loop itself.
    """
    from .hold_shadow_summary import run_thursday_summary

    gem_id = run_thursday_summary()
    if gem_id:
        print(f"hold-shadow-summary: deposited gem {gem_id}")
        return 0
    print("hold-shadow-summary: gem deposit failed (weaver unreachable or error)", file=sys.stderr)
    return 1


def cmd_steer(args) -> int:
    """File a typed steer message for mid-run PM course-correction."""
    type_ = args.type
    if type_ not in steer_mod.VALID_TYPES:
        print(
            f"ERROR: invalid steer type {type_!r}; must be one of {steer_mod.VALID_TYPES}",
            file=sys.stderr,
        )
        return 2
    try:
        path = steer_mod.file_steer(
            target_id=args.target_id,
            type_=type_,
            message=args.message,
            filed_by="pm",
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    # Read back to confirm write
    import json as _json
    payload = _json.loads(path.read_text())
    print(f"Steer filed: {path}")
    print(f"  type:    {payload['type']}")
    print(f"  target:  {payload['target_id']}")
    print(f"  message: {payload['message']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lapis-pm", description="Lapis PM agent CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bind", help="Bind a target (or chain) to the PM with a spec.")
    b.add_argument("target_id",
                   help="Target ID (single-target) or chain-group-id (with --legs-from)")
    b.add_argument("--spec-from", required=True, help="Path to spec, or '-' for stdin")
    b.add_argument("--repo", default=None, help="Forgejo repo name (required for single-target)")
    b.add_argument("--authority", default=None, choices=["advisory", "auto", "hold"],
                   help="Authority level. Precedence: --authority flag > spec's **Authority:** "
                        "header > advisory (fail-safe default). If both are given and disagree, "
                        "bind errors (exit 2, nothing written) naming both values. "
                        "hold = fresh-reviewer + 4-cycle budget.")
    b.add_argument("--verification", default=None, choices=["machine", "pm-live-test"],
                   help="Verification mode override (default: parsed from spec, else pm-live-test). "
                        "'machine' = auto-merge eligible; 'pm-live-test' = always routes to PM.")
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

    sc = sub.add_parser(
        "stall-check",
        help="Dedicated tick-coverage + directive-outcome stall checker "
             "(attestation-contract-v0 leg 2, D6a). stdlib + mem reads only, "
             "no LLM calls; run by lapis-pm-stall-check.timer on a 10m cadence.",
    )
    sc.set_defaults(func=cmd_stall_check)

    ap = sub.add_parser(
        "autopilot",
        help="The pipeline autopilot (the prodder) — "
             "lapis-pm-pipeline-autopilot-v0. PERCEIVE -> classify -> UNBLOCK / "
             "ADJUDICATE / RE-FIRE -> ESCALATE. The daemon tick is the sole "
             "dispatch/merge actor; the sweep only writes state + emits "
             "dossiers + escalates. Shadow-mode-first (default).",
    )
    ap_sub = ap.add_subparsers(dest="autopilot_cmd", required=True)
    ap_sweep = ap_sub.add_parser(
        "sweep",
        help="Run one per-tick sweep (the prodder loop).",
    )
    ap_sweep.add_argument("target", nargs="?", default=None,
                          help="Target id (default: all pm-bound targets)")
    ap_sweep.add_argument("--tick-id", default=None,
                          help="Explicit tick id (default: a UTC ts slug)")
    ap_sweep.add_argument("--json", action="store_true",
                          help="Emit machine-readable JSON")
    ap_sweep.set_defaults(func=cmd_autopilot)
    ap_report = ap_sub.add_parser(
        "report",
        help="The shadow-review surface (the Nudge Log — the narrative render "
             "of the last N shadow proposals + reversals).",
    )
    ap_report.add_argument("target", nargs="?", default=None,
                           help="Target id (default: all targets)")
    ap_report.add_argument("--limit", type=int, default=20,
                           help="Max proposals to show (default: 20)")
    ap_report.set_defaults(func=cmd_autopilot)
    ap_live = ap_sub.add_parser(
        "liveness",
        help="The BRIX-side liveness backstop (D5) — check the heartbeat "
             "freshness and page/record when stale (catches a dead prodder "
             "while GW sleeps).",
    )
    ap_live.add_argument("--json", action="store_true",
                         help="Emit machine-readable JSON")
    ap_live.set_defaults(func=cmd_autopilot)

    s = sub.add_parser("status", help="Show PM state for bound target(s).")
    s.add_argument("target_id", nargs="?")
    s.add_argument("--explain", action="store_true")
    s.set_defaults(func=cmd_status)

    dsv = sub.add_parser(
        "deploy-status",
        help="Render the deploy-inventory reconciler's status (unmapped clones, currency, drift).",
    )
    dsv.add_argument("--json", action="store_true", help="Emit raw status JSON instead of the rendered table")
    dsv.set_defaults(func=cmd_deploy_status)

    pa = sub.add_parser("pause", help="Pause a target.")
    pa.add_argument("target_id")
    pa.add_argument("--reason")
    pa.set_defaults(func=cmd_pause)

    re_ = sub.add_parser("resume", help="Resume a paused target.")
    re_.add_argument("target_id")
    re_.set_defaults(func=cmd_resume)

    cra = sub.add_parser(
        "clear-reviewer-attempts",
        help="Clear reviewer-attempt-ceiling counters for a target (sole escape hatch "
             "after a ceiling auto-pause). Does not resume the target.",
    )
    cra.add_argument("target_id")
    cra.set_defaults(func=cmd_clear_reviewer_attempts)

    cld = sub.add_parser(
        "clear-dispatch",
        help="Clear pending dispatch record(s) for a target (escape hatch for a "
             "stale-pending fixer/reviewer dispatch the reaper couldn't prove dead). "
             "Also prints a reminder about the intention-registry sibling wedge.",
    )
    cld.add_argument("target_id")
    cld.set_defaults(func=cmd_clear_dispatch)

    ls = sub.add_parser("list", help="List all pm_bound targets.")
    ls.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    ls.set_defaults(func=cmd_list)

    ld = sub.add_parser("land", help="Produce an arc doc for a landed thread.")
    ld.add_argument("target_id")
    ld.add_argument("--dry-run", action="store_true",
                    help="Print arc doc to stdout instead of writing to /srv/lapis/lapis-state/")
    ld.set_defaults(func=cmd_land)

    # night-deploy-manifest-attestation-v0: the deploy-manifest completeness
    # checker (pre-PR / manual entry). The DAG node invokes the module directly;
    # this is the CLI surface for operators + the pre-PR proof.
    ac = sub.add_parser(
        "attestation-check",
        help="Run the deploy-manifest completeness attestation (0 clean / 2 held / 3 failed).",
    )
    ac.add_argument("--plan", default=None, help="Night plan path (default /data/slots/night-plan.yaml)")
    ac.add_argument("--waivers", default=None, help="Waiver file path")
    ac.add_argument("--ledger", default=None, help="Attestation ledger path")
    ac.add_argument("--src", default=None, help="Conductor scripts source dir")
    ac.add_argument("--dest", default=None, help="BRIX host-op scripts dir")
    ac.add_argument("--no-emit", action="store_true", help="Do not print the JSON line")
    ac.set_defaults(func=cmd_attestation_check)

    fr = sub.add_parser("friction", help="Friction capture queue operations (operator/debug surface).")
    fr_sub = fr.add_subparsers(dest="friction_cmd", required=True)
    frl = fr_sub.add_parser("list", help="List recent friction records, newest first.")
    frl.add_argument("--target", default=None, help="Filter by target id")
    frl.add_argument("--repo", default=None, help="Filter by repo")
    frl.add_argument("--limit", type=int, default=20, help="Max records to show (default 20)")
    frl.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    frl.add_argument("--exclude-suspect", action="store_true",
                      help="Omit records tagged provenance_suspect by the D3 backfill (default: shown)")
    frl.set_defaults(func=cmd_friction_list)

    frb = fr_sub.add_parser(
        "backfill-provenance",
        help="One-shot: tag (never delete) queue records inherited under false cross-target "
             "provenance. Manual-only, dry-run by default.",
    )
    frb.add_argument("--apply", action="store_true",
                      help="Write tags to the queue (default: dry-run, prints summary only).")
    frb.set_defaults(func=cmd_friction_backfill)

    sib = sub.add_parser(
        "spec-id-backfill",
        help="One-shot: backfill spec_id: <stem> into /srv/lapis/planning/specs/*.md "
             "frontmatter for specs that lack it. Dry-run by default.",
    )
    sib.add_argument("--apply", action="store_true",
                      help="Write spec_id: into candidate files (default: dry-run, report only).")
    sib.set_defaults(func=cmd_spec_id_backfill)

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
        "--pr",
        type=int,
        default=None,
        metavar="N",
        help=(
            "PR number the ratification resolves (lapis-pm-autonomy-actuator-v0). "
            "When omitted, auto-resolved from the target's adopted_pr_number or "
            "the PR number embedded in the prior decision's mem key."
        ),
    )
    rat.add_argument(
        "--json",
        action="store_true",
        help="Print written mem key as JSON {\"key\": ...}",
    )
    rat.set_defaults(func=cmd_ratify)

    dirp = sub.add_parser(
        "directive",
        help="Emit a signed human:directive comment under the pm-review agent key.",
    )
    dirp.add_argument("target_id", help="Target ID to post the directive on")
    dirp.add_argument(
        "--content",
        required=True,
        metavar="TEXT",
        help="One-paragraph directive: what to change, why",
    )
    dirp.add_argument(
        "--json",
        action="store_true",
        help="Print {\"manifest_hash\", \"pubkey_id\", \"comment_id\"} as JSON",
    )
    dirp.set_defaults(func=cmd_directive)

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

    te = sub.add_parser(
        "tts-episode",
        help="Render the latest brief to spoken WAV via a Piper/Kokoro bakeoff (Unit 3a).",
    )
    te.add_argument(
        "--period",
        default="morning",
        help="Brief cadence to synthesize (default: morning — v0's only exercised path).",
    )
    te.add_argument(
        "--voice",
        default="both",
        choices=["both", "piper", "kokoro"],
        help="Which engine(s) to render (default: both, for the bakeoff).",
    )
    te.set_defaults(func=cmd_tts_episode)

    tep = sub.add_parser(
        "tts-episode-publish",
        help="Publish the latest brief to the durable private podcast RSS feed (Unit 3b).",
    )
    tep.add_argument(
        "--period",
        default="morning",
        choices=["morning"],
        help="Brief cadence to publish (default: morning — v0's only supported feed).",
    )
    tep.set_defaults(func=cmd_tts_episode_publish)

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
    sc_sim.add_argument(
        "--model",
        default="gravitywell",
        choices=["gravitywell", "qwen"],
        help="Voicing model: gravitywell (owned 122B local, default) | qwen (StarHouse 35B-A3B).",
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
    # scout refiner — autonomy loop (observe→refine→salience)
    # ------------------------------------------------------------------
    sc_ref = sc_sub.add_parser(
        "refiner",
        help=(
            "Semantic observe→refine loop over active Scout scaffolds + Backcaster goals. "
            "Writes dated salience maps and injects a summary into Active Work.md."
        ),
    )
    sc_ref.add_argument(
        "--observe-only",
        dest="observe_only",
        action="store_true",
        default=False,
        help="Pure-CPU pass — skip the GW refine leg (no proposals, just landscape).",
    )
    sc_ref.add_argument(
        "--spec",
        default=None,
        metavar="ID",
        help="Restrict to one scaffold spec ID (matches digest's --spec shape).",
    )
    sc_ref.add_argument(
        "--cos-threshold",
        dest="cos_threshold",
        type=float,
        default=0.80,
        metavar="F",
        help="Override default 0.80 cosine threshold for Scout signature clustering.",
    )
    sc_ref.set_defaults(func=cmd_scout)

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
    sc_nr.add_argument(
        "--model",
        default="gravitywell",
        choices=["gravitywell", "qwen"],
        help="Voicing model: gravitywell (owned 122B local, default) | qwen (StarHouse 35B-A3B).",
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

    fge = sub.add_parser(
        "facets-gw-eval",
        help="Facets-on-GW load eval: two-arm latency + quality test (haiku vs gravitywell).",
    )
    fge.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit verdict as JSON (in addition to markdown report).",
    )
    fge.set_defaults(func=cmd_facets_gw_eval)

    # ------------------------------------------------------------------
    # functional-critic — AC-anchored oracle verification (functional-critic-v0)
    # ------------------------------------------------------------------
    fc_p = sub.add_parser(
        "functional-critic",
        help=(
            "Run the functional critic against an open PR. "
            "Checks out PR head, runs pytest, calls GW-122B for AC coverage, "
            "prints verdict JSON to stdout."
        ),
    )
    fc_p.add_argument("repo", help="Repository name (e.g. 'lapis-pm').")
    fc_p.add_argument("pr_number", type=int, help="PR number to evaluate.")
    fc_p.add_argument("spec_path", help="Path to the spec markdown file.")
    fc_p.set_defaults(func=cmd_functional_critic)

    # ------------------------------------------------------------------
    # batched-fixer-eval — patch-quality eval harness (H5/U4)
    # ------------------------------------------------------------------
    bfe = sub.add_parser(
        "batched-fixer-eval",
        help="Batched-fixer patch-quality eval: best-of-N Coder-Next vs paid claude-p on real fixes.",
    )
    bfe.add_argument(
        "phase",
        choices=["build-corpus", "run", "report"],
        help="Eval phase: build-corpus (extract + freeze fixtures), run (generate + select candidates), report (aggregate).",
    )
    bfe.add_argument(
        "--run-id",
        default=None,
        help="Run ID (default: YYYYMMDD-HHMMSS).",
    )
    bfe.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Mock mode: use canned swarm completions, no GW dependency.",
    )
    bfe.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit result as JSON (in addition to markdown report).",
    )
    bfe.add_argument(
        "--target-sha",
        default=None,
        help="build-corpus only: harvest exactly this one commit (via harvest_one) instead of "
             "the full corpus sweep; bypasses the '^fix' subject filter (structural filters "
             "still apply).",
    )
    bfe.add_argument(
        "--repo-only",
        default="lapis-pm",
        help="Repo label to harvest --target-sha from (default: lapis-pm). Must be a "
             "label discover_repos() resolves (any /srv/git/*-working repo, e.g. "
             "'conductor') — run_eval validates it against that same discovery, so "
             "single-commit and batch harvest can't disagree about which repos exist.",
    )
    bfe.set_defaults(func=cmd_batched_fixer_eval)

    from .batched_fixer_eval import LAPIS_PM_REPO as _LAPIS_PM_REPO

    cor = sub.add_parser(
        "code-oracle-run",
        help="Run one self-correction agent against one corpus fixture, judged by the "
             "independent oracle (thin wiring to run_code_oracle_experiment).",
    )
    cor.add_argument(
        "--fixture-id", required=True,
        help="Corpus fixture id, e.g. lapis-pm-66a7d2ea.",
    )
    cor.add_argument(
        "--agent-backend-url", required=True,
        help="Endpoint the writeable agent is served from (model selection is by endpoint).",
    )
    cor.add_argument(
        "--repo-path", default=_LAPIS_PM_REPO, type=Path,
        help="Git repo the fixture's clone/worktree is created from (default: "
             f"{_LAPIS_PM_REPO}).",
    )
    cor.add_argument(
        "--out-dir", default="/tmp/code-oracle-run", type=Path,
        help="Directory the structured result + transcript are written to.",
    )
    cor.add_argument(
        "--run-id", default=None,
        help="Run id (default: derived by the harness from the fixture identity).",
    )
    cor.add_argument(
        "--max-steps", default=60, type=int,
        help="Max agent tool-call steps before giving up (default: 60).",
    )
    cor.add_argument(
        "--timeout", default=1800, type=int,
        help="Agent wall-clock timeout in seconds; maps to timeout_s (default: 1800).",
    )
    cor.set_defaults(func=cmd_code_oracle_run)

    sr = sub.add_parser(
        "spec-review",
        help="Pre-bind Facets + Council review of a spec document (fast path by default; --with-reference-reviewer / --with-gw opt in to the reference legs).",
    )
    sr.add_argument("spec_path", help="Path to the spec markdown file.")
    sr.add_argument(
        "--council-voicing",
        default="gravitywell",
        choices=["local", "gravitywell", "flashnext"],
        dest="council_voicing",
        help=(
            "Voicing for Mirror Council deliberation (default: gravitywell — owned 122B, "
            "zero paid spend). flashnext = the flash-next seat (GW :30000, "
            "Qwen3.8-Flash-Next-NVFP4-SSD-Stream), resolved through the gw-seats "
            "registry (gate-lanes-registry-driven-flashnext-v0 S3/S4): a requested-but-"
            "inactive flashnext lane is an honest leg_down, never a silent gravitywell "
            "fallback. Paid-model voicing removed; entity selection retains its own "
            "on_wake_fail fallback."
        ),
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
        "--with-reference-reviewer",
        action="store_true",
        dest="with_reference_reviewer",
        help=(
            "Opt in to the reference-only Empiricist claim-verification leg (off by "
            "default — it is advisory-only and never moves the recommendation, so the "
            "default fast path skips its ~18min cost)."
        ),
    )
    sr.add_argument(
        "--no-reference-reviewer",
        action="store_true",
        dest="no_reference_reviewer",
        help=(
            "Deprecated no-op: the Empiricist leg is off by default now, so this flag "
            "has nothing left to disable. Kept so existing invocations keep working "
            "unchanged (sunset 90 days after merge, 2026-09-05)."
        ),
    )
    sr.add_argument(
        "--no-sonnet-reviewer",
        action="store_true",
        dest="no_sonnet_reviewer",
        help=(
            "Deprecated no-op alias for --no-reference-reviewer; still honored "
            "(sunset 90 days after merge, 2026-09-05)."
        ),
    )
    sr.add_argument(
        "--with-gw",
        action="store_true",
        dest="with_gw",
        help=(
            "Opt in to the GravityWell reference leg (off by default — U3b). When set, "
            "the gate submits the GW job and blocks on its result before rendering; a "
            "timeout still renders the brief, with an explicit failed-grounding marker "
            "rather than a bare error."
        ),
    )
    sr.add_argument(
        "--compare-opus",
        action="store_true",
        dest="compare_opus",
        help=(
            "Deprecated/no-op: the reference leg is opt-in now (--with-reference-reviewer); "
            "this flag is kept for back-compat (sunset 90 days after merge)."
        ),
    )
    sr.add_argument(
        "--facets-operator",
        choices=["haiku", "sonnet", "opus", "qwen", "gravitywell", "flashnext"],
        default="gravitywell",
        dest="facets_operator",
        help=(
            "Model operator for Facets personas and synthesis (default: gravitywell). gravitywell = owned 122B "
            "local, zero paid spend; degrades to haiku when GW is unavailable, and the degrade is reported in "
            "the gate output (⚠️ DEGRADED banner + requested → effective voicing line). Pass sonnet or haiku "
            "to deliberately spend money on a paid pass instead."
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
        default="gravitywell",
        choices=["gravitywell", "qwen", "sonnet", "opus"],
        help="LLM model to use (default: gravitywell - owned 122B local).",
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
    # backcaster-quest — post-hoc web sourcing for unsourced gaps
    # Sibling subcommand (not nested under backcaster) to avoid positional
    # goal_file collision.
    # ------------------------------------------------------------------
    bq = sub.add_parser(
        "backcaster-quest",
        help=(
            "Source unsourced gaps in a completed Backcaster run via Dowser "
            "(web read+critique, leg-driven GW flips)."
        ),
    )
    bq.add_argument(
        "run_id",
        metavar="RUN_ID",
        help="Run directory name under /srv/lapis/backcaster/runs/ (e.g. 2026-06-28-2004-zephyr-...).",
    )
    bq.add_argument(
        "--gap",
        action="append",
        default=None,
        metavar="PRECONDITION_ID",
        help="Target a specific gap by precondition ID (repeatable). Mutually exclusive with --all-unsourced.",
    )
    bq.add_argument(
        "--all-unsourced",
        dest="all_unsourced",
        action="store_true",
        default=False,
        help="Target all unsourced gaps in the run.",
    )
    bq.add_argument(
        "--escalate",
        action="store_true",
        default=False,
        help=(
            "Use read_operator=sonnet (paid) instead of quest (local-only). "
            "No swarm flip. Morning human-targeted re-dispatch path only."
        ),
    )
    bq.set_defaults(func=cmd_backcaster_quest)

    # ------------------------------------------------------------------
    # research-quest — nightly producer sourcing the standing Dowser backlog
    # ------------------------------------------------------------------
    rq = sub.add_parser(
        "research-quest",
        help=(
            "Source pending entries in the Dowser research backlog "
            "(/srv/lapis/research/queue/dowser-backlog.yaml) into /srv/lapis/research/."
        ),
    )
    rq.add_argument(
        "--limit",
        type=int,
        default=5,
        metavar="N",
        help="Max backlog entries to process this run (default: 5).",
    )
    rq.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Select and log pending entries without touching GW/Dowser or the backlog.",
    )
    rq.set_defaults(func=cmd_research_quest)

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

    # ------------------------------------------------------------------
    # bundle-autodispatch — reconcile unbound cr-bundle-* specs
    # ------------------------------------------------------------------
    bad = sub.add_parser(
        "bundle-autodispatch",
        help=(
            "Reconcile unbound cr-bundle-* specs: run the GW spec-review gate and "
            "bind+tick qualifying ones with advisory authority."
        ),
    )
    bad.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Print what would be bound without binding or writing markers.",
    )
    bad.add_argument(
        "--spec-dir",
        default=None,
        dest="spec_dir",
        metavar="PATH",
        help="Spec directory to scan (default: /srv/lapis/planning/specs).",
    )
    bad.add_argument(
        "--gate-timeout",
        default=1800,
        type=int,
        dest="gate_timeout",
        metavar="SECONDS",
        help="Hard timeout for the spec-review gate in seconds (default: 1800).",
    )
    bad.set_defaults(func=cmd_bundle_autodispatch)

    # ------------------------------------------------------------------
    # hold-shadow-summary — Thursday grouped-by-classification Desk gem
    # ------------------------------------------------------------------
    hss = sub.add_parser(
        "hold-shadow-summary",
        help=(
            "Group /srv/lapis/hold-shadow/*.jsonl records by proposed classification "
            "and deposit one Desk gem (lapis-pm-hold-shadow-observer-v0)."
        ),
    )
    hss.set_defaults(func=cmd_hold_shadow_summary)

    # ------------------------------------------------------------------
    # backstop-sweep — one repo's autonomy-backstop sweep step (U3.4)
    # ------------------------------------------------------------------
    bsw = sub.add_parser(
        "backstop-sweep",
        help=(
            "Run one repo's autonomy-backstop sweep step: re-run the main "
            "baseline, attribute red suites to recently resolved PRs, demote "
            "suspect fork classes (lapis-pm-autonomy-actuator-v0, U3)."
        ),
    )
    bsw.add_argument(
        "--repo",
        default=None,
        metavar="REPO",
        help="Force the round-robin cursor to this repo (default: oldest last_sweep_ts).",
    )
    bsw.set_defaults(func=cmd_backstop_sweep)

    # ------------------------------------------------------------------
    # autonomy-promote — manual promotion of a fork class (U3.3)
    # ------------------------------------------------------------------
    ap = sub.add_parser(
        "autonomy-promote",
        help=(
            "Promote a fork class to auto-resolve (manual path; demotion is "
            "automatic, promotion is Erah's from the Thursday reading). "
            "Refuses within the 14-day regression cooldown without --force; "
            "with regression_count > 2, --force requires --justification."
        ),
    )
    ap.add_argument("repo", help="Repo name (e.g. 'lapis-pm').")
    ap.add_argument(
        "verdict_class",
        choices=["clean", "fixable-low", "fixable-nonlow", "needs-human", "needs-review"],
        help="Verdict class of the fork.",
    )
    ap.add_argument(
        "change_classes",
        help="Comma-separated change classes (e.g. 'code,test').",
    )
    ap.add_argument(
        "loc_bucket",
        choices=["lt100", "100_400", "gt400"],
        help="LOC bucket of the fork.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Promote despite the regression cooldown (recorded as Erah-force).",
    )
    ap.add_argument(
        "--justification",
        default=None,
        metavar="TEXT",
        help="Required with --force when regression_count > 2 within 14 days; logged on the class record.",
    )
    ap.set_defaults(func=cmd_autonomy_promote)

    # ------------------------------------------------------------------
    # steer — file a typed mid-run steer message
    # ------------------------------------------------------------------
    st = sub.add_parser(
        "steer",
        help="File a typed steer message for mid-run PM course-correction.",
    )
    st.add_argument("target_id", help="Target ID to steer.")
    st.add_argument(
        "type",
        choices=list(steer_mod.VALID_TYPES),
        help="Steer type: context (informational), directive (reprioritize/constrain), emergency (halt).",
    )
    st.add_argument("message", help="Steer message text.")
    st.set_defaults(func=cmd_steer)

    # ------------------------------------------------------------------
    # prior-art-scout — nightly lineage sweep for prior art via Dowser
    # ------------------------------------------------------------------
    pas = sub.add_parser(
        "prior-art-scout",
        help="Prior-Art Scout: find prior art for committed roadmap items via Dowser.",
    )
    pas_sub = pas.add_subparsers(dest="prior_art_scout_sub", required=True)

    pas_run = pas_sub.add_parser("run", help="Execute a scout run.")
    pas_run.add_argument(
        "--escalate",
        action="store_true",
        default=False,
        help=(
            "Use read_operator=sonnet (paid, morning targeted path). "
            "No swarm flip. Only for named-item re-runs at Erah's request."
        ),
    )
    pas_run.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Select items and write cursor but skip all LLM calls.",
    )
    pas_run.add_argument(
        "--wall-budget",
        dest="wall_budget",
        type=int,
        default=None,
        metavar="N",
        help="Cap total items processed this run (smoke/testing).",
    )
    pas_run.set_defaults(func=cmd_prior_art_scout_run)

    pas_status = pas_sub.add_parser("status", help="Print last run summary from run.yaml.")
    pas_status.set_defaults(func=cmd_prior_art_scout_status)

    pas_rh = pas_sub.add_parser(
        "reset-hopeless",
        help="Clear the known-hopeless sidecar for a given item key.",
    )
    pas_rh.add_argument(
        "key",
        metavar="KEY",
        help="Roadmap item key (e.g. architecture/prior-art-scout-v0).",
    )
    pas_rh.set_defaults(func=cmd_prior_art_scout_reset_hopeless)

    return p


def main(argv: list[str] | None = None) -> int:
    # Resolve node identity once, before any command dispatch. Fail-closed:
    # NodeIdentityViolation / NodeConfigError propagate and exit the process
    # (spec: lapis-pm-node-write-ownership-v0, Design §1).
    node_identity.resolve_node_identity()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
