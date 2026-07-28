"""Brief-gem write-back: deposit a brief as a decision-gem on raise; reconcile decided gems.

Unit 3 of the gem arc. Every brief raised by the PM flows to the weaver Desk as a
decision-gem (Erah decides on the calm Desk rather than through pm-pr-review).
When Erah decides, the reconciler detects it and calls brief.apply_decision().

Mapping in mem.db (two keys per brief-gem pair):
  pm/brief-gem/map/<gem_id>              -> {target_id, brief_comment_id, pr_number,
                                             deposited_ts, actioned_ts|null,
                                             status:"open"|"actioned"|"superseded",
                                             annotation|null}
  pm/brief-gem/by-brief/<tid>/<cid>      -> gem_id  (reverse index for idempotency)

The forward key is the reconciler's lookup (by gem_id from weaver).
The reverse key is the deposit-side idempotency guard.

pm_core is imported inside functions only (circular import avoidance).
All other lapis_pm sub-modules are imported at module level (no circular).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

from agents_core.notify import send_notification, Priority as _NotifyPriority

from . import episodic
from . import authority
from . import brief as _brief

logger = logging.getLogger(__name__)

_BRIX_DEFAULT = "http://203.0.113.10:8403"  # prod-only fallback; pytest guard below prevents tests from reaching it


def _weaver_base_url() -> str:
    """Resolve weaver base URL: WEAVER_BASE_URL > WEAVER_BIND_PORT > BRIX default."""
    base = os.environ.get("WEAVER_BASE_URL")
    if base:
        return base.rstrip("/")
    port = os.environ.get("WEAVER_BIND_PORT")
    if port:
        return f"http://127.0.0.1:{port}"
    return _BRIX_DEFAULT


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _forward_key(gem_id: str) -> str:
    return f"pm/brief-gem/map/{gem_id}"


def _reverse_key(target_id: str, brief_comment_id: str) -> str:
    return f"pm/brief-gem/by-brief/{target_id}/{brief_comment_id}"


_PR_NUMBER_RE = re.compile(r"PR\s*#\s*(\d+)", re.IGNORECASE)


def _extract_pr_number(text: str) -> int | None:
    """Best-effort scan for a 'PR #<N>' mention in brief text. None if not found."""
    m = _PR_NUMBER_RE.search(text or "")
    return int(m.group(1)) if m else None


def _parse_brief_title_and_ask(body: str) -> tuple[str, str]:
    """Extract the first line of '## State' and '## Decision needed' from brief markdown."""
    state_m = re.search(r"^## State\s*\n(.+?)(?=\n##|\Z)", body, re.DOTALL | re.MULTILINE)
    ask_m   = re.search(r"^## Decision needed\s*\n(.+?)(?=\n##|\Z)", body, re.DOTALL | re.MULTILINE)

    title = ""
    if state_m:
        lines = state_m.group(1).strip().splitlines()
        title = lines[0].strip() if lines else ""
    if not title:
        title = "PM brief"

    ask = ""
    if ask_m:
        lines = ask_m.group(1).strip().splitlines()
        ask = lines[0].strip() if lines else ""
    if not ask or ask.lower() == "none":
        ask = "Awaiting decision"

    return title[:200], ask[:500]


def deposit_brief_gem(
    target_id: str,
    b: "lapis_pm.brief.Brief",  # type: ignore[name-defined]
) -> str | None:
    """Deposit b as a decision-gem on the weaver Desk. Returns gem_id or None.

    Idempotent: if (target_id, b.comment_id) already has a live gem, returns
    the existing gem_id without re-depositing.
    Fail-soft: weaver unreachable or any error -> logs warning, returns None.
    The brief must already be set outstanding before this is called.
    """
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("WEAVER_BASE_URL"):
        # under pytest with no explicit (mock/test) weaver target → do NOT hit live prod
        return None   # benign no-op

    from . import pm_core as _pm

    mem = _pm._mem()

    # Idempotency: check reverse index before depositing
    rkey = _reverse_key(target_id, b.comment_id)
    existing = mem.get(rkey)
    if existing:
        return existing["content"]

    # Read brief options to map to gem options (may be empty for hold/non-closed-form briefs)
    opts = _brief.read_options(target_id, b.comment_id)
    gem_options: list[dict] = []
    extracted_pr: int | None = None
    if opts and isinstance(opts.get("options"), list):
        for o in opts["options"]:
            gem_options.append({
                "key": o.get("id", "?"),
                "title": o.get("label", "?"),
                "sub": "",
            })
            if o.get("action", {}).get("kind") == "merge_pr" and extracted_pr is None:
                extracted_pr = o["action"].get("pr")

    if not gem_options:
        # No pm:brief-options comment backs this ask (or its options list was
        # empty) — synthesize a default set so the gem is never a dead end.
        if extracted_pr is None:
            extracted_pr = _extract_pr_number(b.body)
        if extracted_pr is not None:
            gem_options.append({
                "key": "merge_pr",
                "title": f"Merge PR #{extracted_pr}",
                "sub": "",
            })
        gem_options.append({"key": "ack", "title": "Acknowledge", "sub": ""})

    title, ask = _parse_brief_title_and_ask(b.body)
    why = f"Brief for target {target_id}" + (f" PR #{extracted_pr}" if extracted_pr else "")

    payload = {
        "title": title,
        "ask": ask,
        "why": why,
        "origin": f"PM brief · {target_id}",
        "agent": "PM",
        "deposited_by": "lapis-pm:brief",
        "source_thread_id": target_id,
        "options": gem_options,
        "state": "needs",
    }

    try:
        import httpx
        base = _weaver_base_url()
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{base}/v0/decision-gems", json=payload)
            resp.raise_for_status()
            gem_id: str = resp.json()["gem_id"]
    except Exception as exc:
        logger.warning(
            "brief-gem: deposit failed for %s/%s (weaver unreachable or error): %s",
            target_id, b.comment_id, exc,
        )
        return None

    # Write forward + reverse mapping keys
    deposited_ts = _now_iso()
    forward_rec = {
        "target_id": target_id,
        "brief_comment_id": b.comment_id,
        "pr_number": extracted_pr,
        "deposited_ts": deposited_ts,
        "actioned_ts": None,
        "status": "open",
        "annotation": None,
    }
    mem.set(_forward_key(gem_id), json.dumps(forward_rec, ensure_ascii=False),
            tags=["lapis-pm", "brief-gem"])
    mem.set(rkey, gem_id, tags=["lapis-pm", "brief-gem"])

    logger.info("brief-gem: deposited gem %s for %s/%s", gem_id, target_id, b.comment_id)
    return gem_id


def _update_map_status(
    gem_id: str,
    status: str,
    annotation: str | None = None,
    actioned_ts: str | None = None,
) -> None:
    """Update status (and optionally annotation/actioned_ts) of the forward mapping."""
    from . import pm_core as _pm

    mem = _pm._mem()
    fkey = _forward_key(gem_id)
    rec = mem.get(fkey)
    if not rec:
        return
    try:
        data = json.loads(rec["content"])
    except (json.JSONDecodeError, TypeError):
        return
    data["status"] = status
    if annotation is not None:
        data["annotation"] = annotation
    if actioned_ts is not None:
        data["actioned_ts"] = actioned_ts
    mem.set(fkey, json.dumps(data, ensure_ascii=False), tags=["lapis-pm", "brief-gem"])


def _fetch_decided_gems() -> list[dict]:
    """GET /v0/decision-gems?state=decided. Returns empty list on any failure (fail-soft)."""
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("WEAVER_BASE_URL"):
        # under pytest with no explicit (mock/test) weaver target → do NOT hit live prod
        return []   # benign no-op

    try:
        import httpx
        base = _weaver_base_url()
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{base}/v0/decision-gems",
                params={"state": "decided", "limit": 100},
            )
            resp.raise_for_status()
            return resp.json().get("gems", [])
    except Exception as exc:
        logger.warning(
            "brief-gem: decided-gem fetch failed (weaver unreachable or error): %s", exc,
        )
        return []


def _call_supersede_endpoint(gem_id: str, reason: str, by: str) -> bool | None:
    """POST /v0/decision-gems/{gem_id}/supersede.

    Returns True on success, False on 409 (already terminal), None on 404 or any error.
    Fail-soft: logs warnings on unexpected failures.
    """
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("WEAVER_BASE_URL"):
        return None  # benign no-op under pytest without mock weaver

    try:
        import httpx
        base = _weaver_base_url()
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{base}/v0/decision-gems/{gem_id}/supersede",
                json={"reason": reason, "by": by},
            )
            if resp.status_code == 409:
                return False  # already decided or superseded
            if resp.status_code == 404:
                return None  # Leg 1 not yet deployed; swallow
            resp.raise_for_status()
            return True
    except Exception as exc:
        logger.warning(
            "brief-gem: supersede call failed for gem %s: %s", gem_id, exc,
        )
        return None


def _held_path_check_live(target_id: str, pr_number: int, repo: str) -> tuple[bool, str]:
    """Re-fetch live PR diff against LIVE HEAD and check for held paths.

    Returns (blocked, reason). blocked=True on any held path, fetch error,
    branch gone, or unparseable diff — fail-closed on any ambiguity.
    """
    if not repo:
        return True, "no repo configured"

    if "/" in repo:
        _owner, repo_name = repo.split("/", 1)
        owner: str | None = _owner
    else:
        repo_name, owner = repo, None

    try:
        from agents_core.forgejo import get_pr_diff, get_pr
        pr_data = get_pr(repo_name, pr_number, owner=owner)
        if not pr_data:
            return True, f"PR #{pr_number} not found in {repo}"
        diff_text = get_pr_diff(repo_name, pr_number, owner=owner)
        if diff_text is None:
            return True, "PR diff returned None"
        paths = authority.changed_paths(diff_text)
        held_hits = [p for p in paths if authority.is_held_path(p)]
        if held_hits:
            return True, f"held path(s): {', '.join(held_hits[:5])}"
        return False, ""
    except Exception as exc:
        return True, f"diff fetch error: {exc}"


def _notify_and_observe_manual_required(
    target_id: str,
    pr_number: int | None,
    gem_id: str,
    reason: str,
) -> None:
    """Send HIGH-priority Pushover + write observation when a manual merge is required."""
    pr_str = f" PR #{pr_number}" if pr_number else ""
    msg = f"Brief-gem decided but needs your manual merge - {reason}"
    title_str = f"lapis-pm: brief-gem manual merge - {target_id}{pr_str}"
    try:
        send_notification(msg, title=title_str, priority=_NotifyPriority.HIGH)
    except Exception as exc:
        logger.warning("brief-gem: notify failed: %s", exc)
    try:
        episodic.write_observation(
            target_id,
            f"Brief-gem {gem_id} decided but manual merge required: {reason}",
            extra_tags=["pm:brief-gem-manual-required", f"pm:gem={gem_id}"],
        )
    except Exception as exc:
        logger.warning("brief-gem: observation write failed: %s", exc)


def reconcile_decided_gems() -> list[str]:
    """Process decided gems and execute approved brief actions.

    Called once per tick_all() (cross-target). Returns action strings for logging.
    Fail-soft: never raises. Each gem processed independently; errors on one skip the rest.
    """
    from . import pm_core as _pm
    from agents_core.targets import TargetStore

    # Permanent errors warrant supersession rather than transient retry
    _PERMANENT_ERRORS = frozenset({"options_not_found", "stale_brief"})

    actions: list[str] = []
    gems = _fetch_decided_gems()
    mem = _pm._mem()

    if gems:
        store = TargetStore()

    for gem in gems:
        gem_id = gem.get("gem_id", "")
        if not gem_id:
            continue

        fkey = _forward_key(gem_id)
        rec = mem.get(fkey)
        if not rec:
            continue  # not a brief-gem we deposited

        try:
            mapping = json.loads(rec["content"])
        except (json.JSONDecodeError, TypeError):
            continue

        if mapping.get("status") != "open":
            continue  # already actioned or superseded

        target_id = mapping.get("target_id", "")
        brief_comment_id = mapping.get("brief_comment_id", "")
        pr_number: int | None = mapping.get("pr_number")

        if not target_id or not brief_comment_id:
            continue

        decision_json = gem.get("decision_json") or {}
        option_key = decision_json.get("option_key")
        if not option_key:
            logger.warning("brief-gem: gem %s has decided state but no option_key", gem_id)
            continue

        try:
            # Supersession check: the same brief must still be outstanding
            current_brief = _pm.get_outstanding_brief(target_id)
            if current_brief != brief_comment_id:
                _update_map_status(gem_id, "superseded",
                                   annotation="brief already resolved via other path")
                actions.append(f"brief-gem:superseded:{gem_id}")
                continue

            # Determine action_kind from the stored brief options
            opts = _brief.read_options(target_id, brief_comment_id)
            action_kind: str | None = None
            if opts and isinstance(opts.get("options"), list):
                for o in opts["options"]:
                    if o.get("id") == option_key:
                        action_kind = o.get("action", {}).get("kind")
                        break

            # Safety gates — merge_pr only
            if action_kind == "merge_pr":
                target = store.get(target_id)
                repo = (target.pm_repo if target else None) or ""

                # Hold-authority gate: hold = human-in-loop on every merge
                if target and getattr(target, "pm_authority", None) == "hold":
                    reason = "target has hold authority (manual merge required)"
                    _notify_and_observe_manual_required(target_id, pr_number, gem_id, reason)
                    _update_map_status(gem_id, "open",
                                       annotation="manual merge required (hold-authority)")
                    actions.append(f"brief-gem:manual-required:hold-authority:{gem_id}")
                    continue

                # Held-path gate against LIVE PR HEAD (fail-closed)
                if pr_number:
                    blocked, reason = _held_path_check_live(target_id, pr_number, repo)
                    if blocked:
                        full_reason = f"held path on {target_id} PR #{pr_number}: {reason}"
                        _notify_and_observe_manual_required(
                            target_id, pr_number, gem_id, full_reason,
                        )
                        _update_map_status(
                            gem_id, "open",
                            annotation=f"manual merge required (held: {reason})",
                        )
                        actions.append(f"brief-gem:manual-required:held-path:{gem_id}")
                        continue
                else:
                    # No PR number in mapping — cannot safely auto-merge; fail closed
                    reason = f"merge_pr action but no PR number in mapping for {target_id}"
                    _notify_and_observe_manual_required(target_id, pr_number, gem_id, reason)
                    _update_map_status(gem_id, "open",
                                       annotation="manual merge required (no PR number)")
                    actions.append(f"brief-gem:manual-required:no-pr:{gem_id}")
                    continue

            # Execute — reuse apply_decision (handles brief clear + audit + observation)
            result = _brief.apply_decision(target_id, brief_comment_id, option_key)

            if result.get("ok"):
                _update_map_status(gem_id, "actioned", actioned_ts=_now_iso())
                actions.append(
                    f"brief-gem:actioned:{gem_id}:{result.get('action_kind', '?')}"
                )
            else:
                err = result.get("error", "unknown")
                # Permanent errors: bad data that won't improve on retry — supersede
                is_permanent = (
                    err in _PERMANENT_ERRORS
                    or err.startswith("unknown_option_id")
                )
                if is_permanent:
                    _update_map_status(gem_id, "superseded",
                                       annotation=f"permanent error: {err}")
                    actions.append(f"brief-gem:superseded:permanent-error:{gem_id}")
                else:
                    # Transient: leave open, next tick retries
                    logger.warning(
                        "brief-gem: transient error for gem %s: %s (will retry)", gem_id, err,
                    )
                    actions.append(f"brief-gem:retry:{gem_id}:{err}")

        except Exception as exc:
            logger.warning(
                "brief-gem: unexpected error processing gem %s: %s", gem_id, exc,
            )
            actions.append(f"brief-gem:error:{gem_id}:{exc}")

    # --- orphan open-gem sweep ---
    # Supersede open gems whose underlying brief is no longer outstanding.
    # Silent: no notification, no Desk state. 409 (gem decided on same tick) → leave open;
    # decided-gem path self-heals on the next tick.
    map_entries = mem.list_by_prefix("pm/brief-gem/map/", limit=500)
    for entry in map_entries:
        key = entry.get("key", "")
        if not key.startswith("pm/brief-gem/map/"):
            continue
        orphan_gem_id = key[len("pm/brief-gem/map/"):]
        if not orphan_gem_id:
            continue
        try:
            orphan_data = json.loads(entry.get("content", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if orphan_data.get("status") != "open":
            continue
        orphan_target_id = orphan_data.get("target_id", "")
        orphan_brief_cid = orphan_data.get("brief_comment_id", "")
        if not orphan_target_id or not orphan_brief_cid:
            continue
        try:
            current_brief = _pm.get_outstanding_brief(orphan_target_id)
            if current_brief == orphan_brief_cid:
                continue  # brief still outstanding — leave untouched
            supersede_result = _call_supersede_endpoint(
                orphan_gem_id,
                reason="brief resolved via other path (orphan reconcile)",
                by="lapis-pm:brief-gem-reconciler",
            )
            if supersede_result is True:
                _update_map_status(orphan_gem_id, "superseded",
                                   annotation="orphan: brief resolved via other path")
                actions.append(f"brief-gem:orphan-superseded:{orphan_gem_id}")
            elif supersede_result is False:
                # 409: gem already decided/superseded by Erah on same tick; leave open
                logger.debug(
                    "brief-gem: orphan gem %s already terminal (409), leaving open",
                    orphan_gem_id,
                )
            # None: 404 (Leg 1 not deployed) or network error — swallowed silently
        except Exception as exc:
            logger.warning(
                "brief-gem: unexpected error in orphan sweep for gem %s: %s",
                orphan_gem_id, exc,
            )

    return actions
