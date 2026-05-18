"""Lapis PM tick orchestration.

One tick per bound target: perceive → score → encode → decide → act.

State across ticks lives in mem.db:
  pm/cursor/<target_id>            ISO timestamp; comments at or before this
                                   were already considered.
  pm/dispatched/<target_id>        JSON list of dispatch records.
  pm/pause-state/<target_id>       "paused" | "active" — transition detector.
  pm/outstanding-brief/<target_id> Comment id of the most recent unanswered brief.

Single-action discipline: at most one decision-action per tick. Encoding
percepts as comments is bookkeeping, not action.

Decision taxonomy (see README.md § "Tick Decision Taxonomy"):
  Noop:   noop:no_change | noop:paused | noop:reviewer_in_flight:pr=N:cycle=K
          | noop:fixer_in_flight:dispatch=ID | noop:awaiting_chain_dependency:waiting_on=TID
  Action: action:auto_merge:pr=N | action:auto_land:pr=N:arc=PATH
          | action:reviewer_dispatched:pr=N:cycle=K
          | action:fixer_dispatched:source=(init|retry):...
          | action:brief_emitted:kind=KIND:cid=CID | action:brief_decision_applied:BID:OID
          | action:directive_brief:cid=CID | action:abandon_brief:cid=CID | ...
  Skip:   skipped=True reason=(target not found|target not pm_bound|paused
          |forgejo_unreachable|ratelimit|cursor_locked)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

from agents_core.targets import TargetStore
from agents_core.mem import MemoryStore
from agents_core.notify import Priority as NotifyPriority

try:
    from agents_core.forgejo import get_open_prs, merge_pr, get_pr as _forgejo_get_pr, get_branch as _forgejo_get_branch
except Exception:
    get_open_prs = None  # type: ignore
    merge_pr = None  # type: ignore
    _forgejo_get_pr = None  # type: ignore
    _forgejo_get_branch = None  # type: ignore

try:
    from agents_core.claude_queue import ClaudeQueue as _ClaudeQueue
except Exception:
    _ClaudeQueue = None  # type: ignore

from agents_core.shaper import Shaper, DispatchResult as _DispatchResult  # noqa: F401
from . import episodic, brief, authority

try:
    from . import eval_gate as _eval_gate
except Exception:
    _eval_gate = None  # type: ignore

# Module-level singleton — constructed at import time so a malformed
# registry.yaml crashes the daemon immediately, not at first dispatch.
_SHAPER = Shaper(Path(__file__).parent / "registry.yaml")


PACIFIC = ZoneInfo("America/Los_Angeles")
COMPLETED_DIR = Path("/srv/lapis/gpu-queue/completed")
FAILED_DIR = Path("/srv/lapis/gpu-queue/failed")
SHAPED_DIR = Path("/srv/lapis/gpu-queue/shaped")  # shaped_runner meta sidecars
CLAUDE_QUEUE_COMPLETED_DIR = Path("/srv/lapis/claude-queue/completed")
CLAUDE_QUEUE_FAILED_DIR = Path("/srv/lapis/claude-queue/failed")
MAX_DISPATCH_RETRIES = 2

# Review-gate loop constants
REVIEW_GATE_THRESHOLD = 40          # Opus reviewer calls before soft-pause
REVIEW_GATE_COUNTER_KEY = "pm/review-gate/cycles-this-window"
REVIEW_GATE_PAUSED_KEY = "pm/review-gate/paused"
REVIEW_GATE_PAUSE_BRIEF_KEY = "pm/review-gate/pause-brief-posted"

# Cycle budgets per authority level (number of reviewer dispatches before exhausted)
_REVIEW_CYCLE_BUDGETS: dict[str, int] = {
    "advisory": 2,
    "hold": 4,
}

# Reviewer mode per authority level (fresh-reviewer = cold full-diff each cycle)
_REVIEWER_MODES: dict[str, str] = {
    "advisory": "same-reviewer",
    "hold": "fresh-reviewer",
}

# Forgejo health gate constants
FORGEJO_CONSECUTIVE_FAILS_KEY = "pm/forgejo_consecutive_fails"
FORGEJO_UNREACHABLE_THRESHOLD = 3  # consecutive failed probes before Pushover

# Tick-local corroboration cache: (target_id, pr_number) -> corr_result dict.
# Populated by _encode_gpu_results; consumed by _persist_review_state_cache in
# the same tick to avoid re-scanning episodic for data already in hand.
_tick_corr_cache: dict[tuple[str, int], dict] = {}

# ---------------------------------------------------------------------------
# Post-land deploy hook
# ---------------------------------------------------------------------------

_POST_LAND_RESTART: dict[str, tuple[str, ...]] = {
    "agents-core": ("claude-queue-runner.service", "gpu-queue-runner.service"),
}

_POST_LAND_PULL: dict[str, list[str]] = {
    "lapis-pm":    ["/srv/lapis/lapis-pm"],
    "agents-core": ["/srv/git/agents-core-working"],
}

# Checked once at module load so tests can patch the env before import.
_DEPLOY_HOOK_DISABLED = os.environ.get("LAPIS_PM_DEPLOY_HOOK_DISABLE") == "1"


def _post_land_git_pull(repo: str | None) -> None:
    """Git-pull each working clone mapped to `repo`.

    Best-effort. Never raises. Failures are logged to stderr. No-ops on
    None or unmapped repos. Uses --ff-only so a diverged clone fails
    loudly rather than silently creating a merge commit.
    """
    if repo is None:
        return
    paths = _POST_LAND_PULL.get(repo)
    if not paths:
        return
    for path in paths:
        try:
            result = subprocess.run(
                ["git", "-C", path, "pull", "--ff-only", "origin", "main"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                print(
                    f"[post-land-pull] pull {path} failed rc={result.returncode}: "
                    f"{result.stderr[:200]}",
                    file=sys.stderr,
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"[post-land-pull] pull {path} errored: {e}", file=sys.stderr)


def _post_land_deploy_hook(repo: str | None) -> None:
    """Pull working clones then restart long-running services for `repo`.

    Best-effort. Failures (sudo unavailable, unit missing, restart timeout)
    are logged to stderr. Never raises — landing must complete even if the
    pull or restart fails.

    Idempotent at the systemd level: `systemctl restart` of an
    already-running unit is a clean SIGTERM + restart; the runner drains
    in-flight tasks per its existing shutdown handler (claude_queue_runner.py
    lines 520-541), bounded by `TimeoutStopSec=900`.
    """
    if _DEPLOY_HOOK_DISABLED:
        print(
            "[post-land-deploy:disabled] skipping restart "
            "(LAPIS_PM_DEPLOY_HOOK_DISABLE=1)",
            file=sys.stderr,
        )
        return
    _post_land_git_pull(repo)
    if not repo:
        return
    units = _POST_LAND_RESTART.get(repo)
    if not units:
        return
    for unit in units:
        try:
            result = subprocess.run(
                ["sudo", "-n", "systemctl", "restart", unit],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                print(
                    f"[post-land-deploy] restart {unit} failed "
                    f"rc={result.returncode}: {result.stderr.strip()[:200]}",
                    file=sys.stderr,
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            print(
                f"[post-land-deploy] restart {unit} errored: {e}",
                file=sys.stderr,
            )


def _read_fixer_meta(spec_id: str) -> dict | None:
    """Read the {spec_id}-meta.json sidecar written by _runner.py for fixers."""
    p = SHAPED_DIR / f"{spec_id}-meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _consume_fixer_meta(spec_id: str) -> None:
    """Delete the meta sidecar after pm_core has consumed it. Best-effort."""
    p = SHAPED_DIR / f"{spec_id}-meta.json"
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def _read_fixer_verdict(spec_id: str) -> dict | None:
    """Read the {spec_id}-verdict.json sidecar copied from worktree by shaped_runner.

    Returns the parsed dict if present and valid JSON, None otherwise.
    Logs a WARN if the file exists but is unreadable or not a dict.
    """
    import sys
    p = SHAPED_DIR / f"{spec_id}-verdict.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            print(f"WARN: verdict sidecar {spec_id} is not a dict: {type(data).__name__}",
                  file=sys.stderr)
            return None
        return data
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: verdict sidecar {spec_id} unreadable: {exc}", file=sys.stderr)
        return None


def _consume_fixer_verdict(spec_id: str) -> None:
    """Delete the verdict sidecar after pm_core has consumed it. Best-effort."""
    p = SHAPED_DIR / f"{spec_id}-verdict.json"
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def _validate_already_satisfied_pr(repo: str, pr_num: int) -> bool:
    """Return True if pr_num exists in repo and is merged+closed in Forgejo."""
    if not repo or not _forgejo_get_pr:
        return False
    try:
        pr_data = _forgejo_get_pr(repo, pr_num)
        return bool(pr_data.get("merged") and pr_data.get("state") == "closed")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _mem() -> MemoryStore:
    return MemoryStore()


def _cursor_key(target_id: str) -> str:
    return f"pm/cursor/{target_id}"


def _dispatched_key(target_id: str) -> str:
    return f"pm/dispatched/{target_id}"


def _pause_key(target_id: str) -> str:
    return f"pm/pause-state/{target_id}"


def _brief_key(target_id: str) -> str:
    return f"pm/outstanding-brief/{target_id}"


def _classified_prs_key(target_id: str) -> str:
    return f"pm/classified-prs/{target_id}"


REVIEW_STATE_KEY_PREFIX = "pm/review-state/"


def _review_state_key(target_id: str) -> str:
    return f"{REVIEW_STATE_KEY_PREFIX}{target_id}"


# ---------------------------------------------------------------------------
# Corroboration follow-up pass helpers (cross-node-corroboration-v0, PR 3)
# ---------------------------------------------------------------------------

def _diff_text_for_corr(repo: str, pr_num: int | str) -> str:
    """Fetch PR diff for the corroboration pass. Returns '' on any failure."""
    try:
        from agents_core.forgejo import get_pr_diff as _get_diff
        return _get_diff(repo, int(pr_num))
    except Exception:
        return ""


def _run_corroboration_pass_sync(diff_text: str, repo: str) -> dict:
    """Run corroboration follow-up pass synchronously.

    Returns an uncertain-shaped dict on any error so the caller never
    needs to handle exceptions (Compost invariant: all outputs are nutrients).
    This function is a named wrapper so tests can patch it directly.
    """
    try:
        from .corroboration_adapter import run_corroboration_pass
        return run_corroboration_pass(diff_text, repo)
    except Exception as exc:
        return {
            "verdict": "uncertain",
            "claim": "",
            "citations": [],
            "freshness_stamp": _now_iso(),
            "scope_id": f"repo:{repo}",
            "drift_class": None,
            "notes": f"corroboration pass unavailable: {type(exc).__name__}",
            "primitive_decomposition": None,
        }


def _persist_review_state_cache(target_id: str, target, open_prs: list[dict]) -> None:
    """Write or delete pm/review-state/<tid> from current tick state.

    Called once per tick after decide-step. Side-effect only; never raises
    (failures logged via episodic.write_observation with pm:error tag,
    but never abort the tick — cache is best-effort visibility).
    """
    state = _active_review_state(target_id, open_prs)
    key = _review_state_key(target_id)
    if state is None:
        # No active review loop — delete stale cache entry if present.
        _mem().delete(key)
        return

    authority_level = target.pm_authority or "advisory"
    budget = _REVIEW_CYCLE_BUDGETS.get(authority_level, 2)
    mode = _REVIEWER_MODES.get(authority_level, "same-reviewer")
    verdict = state.get("verdict")
    has_real_verdict = verdict and verdict != "pending"

    # Compact corroboration summary for quick display in claude-view.
    # Full evidence-packet lives in the episodic entry; cache stores verdict+drift only.
    last_corroboration: dict | None = None
    if has_real_verdict:
        pr_number = state.get("pr_number")
        if pr_number is not None:
            # Prefer tick-local cache (populated by _encode_gpu_results this same tick)
            # to avoid re-scanning episodic for data already in hand.
            corr = _tick_corr_cache.get((target_id, pr_number))
            if corr is None:
                verdict_info = _last_review_verdict(target_id, pr_number)
                if verdict_info:
                    corr = verdict_info.get("corroboration_result")
            if corr:
                last_corroboration = {
                    "verdict": corr.get("verdict"),
                    "drift_class": corr.get("drift_class"),
                }

    payload = {
        "pr_number": state["pr_number"],
        "cycle": state["cycle"],
        "budget": budget,
        "mode": mode,
        "last_verdict": verdict if has_real_verdict else None,
        "last_issues": state.get("issues") if (has_real_verdict and verdict == "fixable") else None,
        "paused": _review_gate_paused(),
        "updated_at": _now_iso(),
        "last_corroboration": last_corroboration,
    }
    _mem().set(key, json.dumps(payload),
               tags=["lapis-pm", "review-state"])


def _now_iso() -> str:
    """Microsecond-precision so it interleaves cleanly with comment timestamps."""
    return datetime.now(PACIFIC).isoformat(timespec="microseconds")


def get_cursor(target_id: str) -> str | None:
    rec = _mem().get(_cursor_key(target_id))
    return rec["content"] if rec else None


def set_cursor(target_id: str, ts: str):
    _mem().set(_cursor_key(target_id), ts, tags=["lapis-pm", "cursor"])


def load_dispatched(target_id: str) -> list[dict]:
    rec = _mem().get(_dispatched_key(target_id))
    if not rec:
        return []
    try:
        data = json.loads(rec["content"])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_dispatched(target_id: str, records: list[dict]):
    _mem().set(_dispatched_key(target_id),
               json.dumps(records, ensure_ascii=False),
               tags=["lapis-pm", "dispatched"])


def append_dispatched(target_id: str, record: dict):
    records = load_dispatched(target_id)
    records.append(record)
    save_dispatched(target_id, records)


def force_dispatch(target_id: str, agent_type: str, intent: str) -> str:
    """Dispatch a shaped agent, record it, and emit a router-portfolio event.

    Extracted from cmd_tick's force-dispatch block so that both the CLI and
    _act_force_dispatch_retry can call the same path.  Returns task_id.
    """
    target = TargetStore().get(target_id)
    if target is None:
        raise ValueError(f"target not found: {target_id}")
    spec_sum = episodic.spec_summary(target_id)
    existing_branch = f"lapis/{target_id}/forced"
    base_branch = "main"
    if agent_type == "fixer_retry" and target.pm_repo:
        try:
            from agents_core.forgejo import get_open_prs as _get_open_prs
            for pr in _get_open_prs(target.pm_repo):
                pr_ref = (pr.get("head") or {}).get("ref", "")
                if pr_ref.startswith(f"lapis/{target_id}/"):
                    existing_branch = pr_ref
                    base_branch = (pr.get("base") or {}).get("ref", "main")
                    break
        except Exception:
            pass
    vars_ = {
        "target_id": target_id,
        "spec_summary": spec_sum,
        "repo": target.pm_repo or "",
        "question": intent,
        "pr_number": "",
        "slug": "forced",
        "existing_branch": existing_branch,
        "base_branch": base_branch,
    }
    res = _SHAPER.dispatch(agent_type, target_id, intent, vars_=vars_)
    append_dispatched(target_id, {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": agent_type,
        "intent": intent,
        "repo": target.pm_repo or "",
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": 0,
    })
    episodic.write_dispatch(
        target_id,
        f"Forced dispatch: {agent_type} → {res.task_id}\nIntent: {intent}",
        extra_tags=[f"pm:gpu={res.task_id}", f"pm:agent={agent_type}"],
    )
    try:
        from .router_portfolio import emit_decision_dispatch as _emit_dispatch
        dispatched = load_dispatched(target_id)
        if agent_type == "fixer":
            frag = "kickoff" if len(dispatched) == 1 else "tick"
        elif agent_type == "reviewer":
            frag = "review-cycle"
        elif agent_type == "brief":
            frag = "human-judgment"
        else:
            frag = agent_type
        model = "unknown"
        try:
            model = _SHAPER.get_agent(agent_type).model
        except Exception:
            pass
        _TIER_MAP = {
            "haiku": "haiku", "sonnet": "sonnet", "opus": "opus",
            "qwen-3.6-35b-a3b": "qwen-local", "qwen3.6-35b-a3b": "qwen-local",
        }
        _emit_dispatch(
            target_id=target_id,
            fragment_id=frag,
            expert_chosen=_TIER_MAP.get(model.lower(), model.lower()),
            intent_summary=intent[:200],
        )
    except Exception as _e:
        import sys
        print(f"[router-portfolio:emit-failed] dispatch: {_e}", file=sys.stderr)
    return res.task_id


def get_pause_state(target_id: str) -> str | None:
    rec = _mem().get(_pause_key(target_id))
    return rec["content"] if rec else None


def set_pause_state(target_id: str, state: str):
    _mem().set(_pause_key(target_id), state, tags=["lapis-pm", "pause-state"])


def set_outstanding_brief(target_id: str, comment_id: str):
    _mem().set(_brief_key(target_id), comment_id, tags=["lapis-pm", "outstanding-brief"])


def clear_outstanding_brief(target_id: str, reason: str = "unspecified") -> None:
    import sys
    # Read the current value for the audit line. A read failure must not
    # block the delete — that would regress idempotency.
    try:
        observed = get_outstanding_brief(target_id)
    except Exception as exc:
        print(
            f"[outstanding-brief:clearing:read-error] tid={target_id} "
            f"reason={reason} err={exc!r}",
            file=sys.stderr,
        )
        observed = None
    if observed is not None:
        print(
            f"[outstanding-brief:clearing] tid={target_id} cid={observed} "
            f"reason={reason}",
            file=sys.stderr,
        )
    else:
        print(
            f"[outstanding-brief:clearing:already-absent] tid={target_id} "
            f"reason={reason}",
            file=sys.stderr,
        )
    _mem().delete(_brief_key(target_id))


def get_outstanding_brief(target_id: str) -> str | None:
    rec = _mem().get(_brief_key(target_id))
    return rec["content"] if rec else None


class OutstandingBriefWriteError(RuntimeError):
    """set_outstanding_brief failed to persist after retry."""


def set_outstanding_brief_verified(target_id: str, comment_id: str) -> None:
    """Set the outstanding-brief mem key with a read-back verify.

    Writes the key, immediately reads it back, and on mismatch:
    1. Logs a structured WARN line to stderr (so journalctl picks it
       up — see `[router-portfolio:emit-failed]` precedent at
       pm_core.py:384) with target_id, comment_id, and the observed
       value (or 'missing').
    2. Retries the write exactly once.
    3. Re-reads. If still missing or mismatched after retry, raises
       OutstandingBriefWriteError so the caller surfaces the failure
       rather than logging a phantom action.

    The action-string returned by _act_brief is the operator's contract
    for "the brief is now resolvable" — silent persistence failure breaks
    that contract.
    """
    import sys
    set_outstanding_brief(target_id, comment_id)
    for attempt in range(1, 3):
        observed = get_outstanding_brief(target_id)
        if observed == comment_id:
            return
        observed_repr = observed if observed is not None else "missing"
        print(
            f"[outstanding-brief:write-mismatch] tid={target_id} cid={comment_id}"
            f" observed={observed_repr} attempt={attempt}",
            file=sys.stderr,
        )
        if attempt == 1:
            set_outstanding_brief(target_id, comment_id)
    raise OutstandingBriefWriteError(
        f"set_outstanding_brief failed to persist after retry: "
        f"tid={target_id} cid={comment_id}"
    )


def _post_write_sweep_brief(target_id: str, comment_id: str) -> None:
    """Read the brief key once more after _mark_pr_classified ran.

    This catches the external-deleter hypothesis the verify-and-retry
    cannot defend against: if some sibling process (sweeper, concurrent
    PM session, mem CLI invocation) deletes the key in the window
    between verify and the next status read, this sweep is the only
    on-tick surface that records the disappearance.

    Logs to stderr only; never raises. Sweep is observability, not
    guarantee — the verify-and-retry IS the guarantee.
    """
    import sys
    observed = get_outstanding_brief(target_id)
    if observed != comment_id:
        print(
            f"[outstanding-brief:disappeared-post-write] tid={target_id} cid={comment_id}",
            file=sys.stderr,
        )


def _landed_key(target_id: str) -> str:
    return f"pm/landed/{target_id}"


# ---------------------------------------------------------------------------
# Forgejo health gate
# ---------------------------------------------------------------------------

def probe_forgejo_health() -> tuple[bool, str]:
    """Probe Forgejo reachability with GET /api/v1/version (3-second timeout).

    Sends the same Authorization header as all other agents_core.forgejo calls
    (Forgejo returns 403 to anonymous requests on this instance).

    Called once per tick from tick_all() only; single-target tick() bypasses
    the probe entirely.

    Returns (True, "") on success.
    Returns (False, reason) on failure where reason is one of:
      "connect_error", "timeout", "http=<code>"
    """
    try:
        from agents_core.forgejo import FORGEJO_URL, FORGEJO_TOKEN
    except Exception:
        return True, ""  # agents_core unavailable — assume reachable
    try:
        import httpx
    except ImportError:
        return True, ""  # httpx unavailable — assume reachable
    try:
        r = httpx.get(
            f"{FORGEJO_URL}/api/v1/version",
            headers={"Authorization": f"token {FORGEJO_TOKEN}"},
            timeout=3.0,
        )
        if not (200 <= r.status_code < 300):
            return False, f"http={r.status_code}"
        return True, ""
    except httpx.TimeoutException:
        return False, "timeout"
    except Exception:
        return False, "connect_error"


def _get_forgejo_consecutive_fails() -> int:
    rec = _mem().get(FORGEJO_CONSECUTIVE_FAILS_KEY)
    if not rec:
        return 0
    try:
        return int(rec["content"])
    except (ValueError, KeyError, TypeError):
        return 0


def _set_forgejo_consecutive_fails(n: int) -> None:
    _mem().set(FORGEJO_CONSECUTIVE_FAILS_KEY, str(n), tags=["lapis-pm", "forgejo-health"])


def _notify_forgejo_unreachable() -> None:
    """Emit one HIGH-priority Pushover when Forgejo has been unreachable for ≥3 ticks."""
    try:
        from agents_core.notify import send_notification, Priority as _P
        send_notification(
            "Forgejo health probe has failed for 3 or more consecutive ticks. "
            "Daemon is skipping all target processing until Forgejo recovers.",
            title="lapis-pm: Forgejo unreachable for >3 ticks",
            priority=_P.HIGH,
        )
    except Exception:
        pass


def _classified_pr_ids(target_id: str) -> set[int]:
    """PR numbers that have already been classified (hold/advisory/merge) this target."""
    rec = _mem().get(_classified_prs_key(target_id))
    if not rec:
        return set()
    try:
        data = json.loads(rec["content"])
        return {int(x) for x in (data if isinstance(data, list) else [])}
    except (json.JSONDecodeError, ValueError):
        return set()


def _mark_pr_classified(target_id: str, pr_number: int):
    ids = _classified_pr_ids(target_id)
    ids.add(pr_number)
    _mem().set(_classified_prs_key(target_id),
               json.dumps(sorted(ids)),
               tags=["lapis-pm", "classified-prs"])


def clear_classified_prs(target_id: str):
    """Reset classified-PR state. Called on unbind/rebind so stale PR numbers don't block re-evaluation."""
    _mem().delete(_classified_prs_key(target_id))


def clear_landed_state(target_id: str) -> dict[str, int]:
    """Clear all PM mem state for a landed target.

    Called from `lapis-pm land` so archived targets stop polluting
    tick loops and `lapis-pm status`. Pending dispatched records are
    marked `status: "landed"` (not deleted) so the history stays
    queryable from mem for future arc-doc regeneration or audit.

    Clears: outstanding_brief, classified_prs, cursor, pause_state,
    dispatched_pending (marked landed), review_state cache.

    Returns a dict describing what was non-empty at clear time —
    for a caller (the `land` command) to print a useful summary.
    Idempotent: safe to call on a target that has no state.
    """
    mem = _mem()
    summary = {
        "outstanding_brief": 0,
        "classified_prs": 0,
        "cursor": 0,
        "pause_state": 0,
        "dispatched_pending": 0,
        "review_state": 0,
    }

    if get_outstanding_brief(target_id) is not None:
        summary["outstanding_brief"] = 1
    clear_outstanding_brief(target_id, reason="auto_land")

    summary["classified_prs"] = len(_classified_pr_ids(target_id))
    clear_classified_prs(target_id)

    if get_cursor(target_id) is not None:
        summary["cursor"] = 1
    mem.delete(_cursor_key(target_id))

    if get_pause_state(target_id) is not None:
        summary["pause_state"] = 1
    mem.delete(_pause_key(target_id))

    records = load_dispatched(target_id)
    pending = [r for r in records if r.get("status") == "pending"]
    if pending:
        for r in records:
            if r.get("status") == "pending":
                r["status"] = "landed"
        save_dispatched(target_id, records)
    summary["dispatched_pending"] = len(pending)

    if mem.get(_review_state_key(target_id)) is not None:
        summary["review_state"] = 1
    mem.delete(_review_state_key(target_id))

    return summary


# ---------------------------------------------------------------------------
# Already-satisfied verdict helpers
# ---------------------------------------------------------------------------

def _handle_already_satisfied_verdict(target_id: str, rec: dict, verdict_raw: dict) -> bool:
    """Evaluate an already_satisfied verdict sidecar. Returns True if handled.

    When True is returned, the caller should mark the dispatch processed and
    continue (skip confabulation check and normal write_result). Two sub-cases:

    - valid verdict + valid merged PR → write pm:already-satisfied observation
    - valid verdict + invalid/missing PR → write pm:already-satisfied:invalid
      observation (decide phase will brief the human)

    Malformed verdicts (non-dict, unrecognised verdict type, missing pr_num)
    return False so the caller falls through to the normal confabulation path.
    """
    import sys

    verdict_type = verdict_raw.get("verdict")
    if verdict_type != "already_satisfied":
        print(f"WARN: unrecognised verdict type in sidecar: {verdict_type!r}", file=sys.stderr)
        return False

    pr_num = verdict_raw.get("satisfied_by_pr")
    if not isinstance(pr_num, int) or pr_num <= 0:
        print(f"WARN: malformed satisfied_by_pr in verdict sidecar: {pr_num!r}", file=sys.stderr)
        return False

    evidence = str(verdict_raw.get("evidence", ""))[:500]
    repo = rec.get("repo", "")

    if _validate_already_satisfied_pr(repo, pr_num):
        episodic.write_observation(
            target_id,
            f"Fixer verdict: already_satisfied by PR #{pr_num}\nEvidence: {evidence}",
            extra_tags=["pm:already-satisfied", f"pm:already-satisfied:pr={pr_num}"],
        )
        return True

    # Forgejo validation failed — warn but still mark as handled so the fixer
    # is not re-retried (it intentionally produced no PR).
    print(
        f"WARN: already_satisfied PR #{pr_num} not found or not merged in {repo!r}",
        file=sys.stderr,
    )
    episodic.write_observation(
        target_id,
        f"Fixer claimed already_satisfied (PR #{pr_num}) but Forgejo validation failed "
        f"— will brief human instead of auto-landing. Evidence: {evidence}",
        extra_tags=["pm:already-satisfied:invalid", f"pm:already-satisfied:invalid:pr={pr_num}"],
    )
    return True


def _already_satisfied_pending(target_id: str) -> tuple[int, str, str] | None:
    """Return (pr_num, evidence, ts) if a valid already_satisfied verdict awaits action.

    Conditions: pm:already-satisfied:pr=N observation exists, target not yet
    landed, no pending dispatches.
    """
    if _mem().get(_landed_key(target_id)):
        return None
    if _has_pending_dispatch(target_id):
        return None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:already-satisfied:pr=") and "invalid" not in t:
                try:
                    pr_num = int(t.split("=", 1)[1])
                    content = c.content
                    evidence = ""
                    if "Evidence:" in content:
                        evidence = content.split("Evidence:", 1)[1].strip()
                    return (pr_num, evidence, c.ts)
                except (ValueError, IndexError):
                    pass
    return None


def _already_satisfied_invalid_pending(target_id: str) -> tuple[int, str] | None:
    """Return (pr_num, content) if an invalid already_satisfied verdict needs a brief.

    Returns None if already briefed (de-dup via pm:already-satisfied:invalid-briefed tag).
    """
    briefed = any(
        "pm:already-satisfied:invalid-briefed" in c.tags
        for c in episodic.all_comments(target_id)
    )
    if briefed:
        return None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:already-satisfied:invalid:pr="):
                try:
                    pr_num = int(t.split("=", 1)[1])
                    return (pr_num, c.content)
                except (ValueError, IndexError):
                    pass
    return None


# ---------------------------------------------------------------------------
# Auto-land helpers
# ---------------------------------------------------------------------------

def _merged_pr_numbers_observed(target_id: str) -> set[int]:
    """PR numbers for which a pm:pr-merged observation has been recorded in episodic."""
    out: set[int] = set()
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:pr-merged:"):
                try:
                    out.add(int(t.split(":", 2)[2]))
                except (ValueError, IndexError):
                    pass
    return out


def _merged_at_for_pr(target_id: str, pr_num: int) -> str:
    """Return the actual Forgejo merged_at timestamp for *pr_num* from episodic.

    Falls back to _now_iso() if the observation is missing (shouldn't happen
    in practice, but keeps _act_auto_land safe on retry after partial failure).
    """
    tag = f"pm:pr-merged:{pr_num}"
    for c in episodic.all_comments(target_id):
        if tag in c.tags:
            m = re.search(r"merged at (\S+),", c.content)
            if m:
                return m.group(1)
    return _now_iso()


def _encode_merged_prs(target_id: str, repo: str) -> int:
    """Check Forgejo for newly merged lapis PRs and write pm:pr-merged observations.

    Only calls get_pr for PR numbers not already noted. Returns count of new
    observations written. Gracefully degrades when Forgejo is unavailable.
    """
    if not repo or not _forgejo_get_pr:
        return 0
    seen = _seen_pr_ids(target_id)
    already_noted = _merged_pr_numbers_observed(target_id)
    new_obs = 0
    for pr_num in sorted(seen - already_noted):
        try:
            pr_data = _forgejo_get_pr(repo, pr_num)
        except Exception:
            continue
        if pr_data.get("merged") and pr_data.get("state") == "closed":
            merged_at = pr_data.get("merged_at") or _now_iso()
            created_at = pr_data.get("created_at") or ""
            head_sha = (pr_data.get("head") or {}).get("sha", "")
            episodic.write_observation(
                target_id,
                f"PR #{pr_num} merged at {merged_at}, created_at={created_at}, head_sha={head_sha}",
                extra_tags=[f"pm:pr-merged:{pr_num}", f"pm:pr={pr_num}"],
            )
            new_obs += 1
    return new_obs


def _detect_pr_count_from_spec(spec_text: str) -> int:
    """Count '^### PR \\d+' headers in spec; return max(count, 1). Regex-only, LLM-free."""
    matches = re.findall(r"^### PR \d+(?=\s|$)", spec_text, re.MULTILINE)
    return max(len(matches), 1)


def _has_auto_land_waiting_comment(target_id: str, pr_count: int) -> bool:
    """Return True if a pm:auto-land:waiting comment for this pr_count already exists."""
    tag = f"pm:auto-land:waiting:count={pr_count}"
    for c in episodic.all_comments(target_id):
        if tag in c.tags:
            return True
    return False


def _is_auto_land_eligible(target_id: str) -> bool:
    """Return True if this target meets all auto-land conditions.

    Conditions (all must hold):
      - No pm/landed/<tid> entry already exists
      - No pending dispatches
      - At least one merged PR is observed
      - The most recently seen PR is the merged one (no newer open PR)
      - All declared PRs have merged (len(merged) >= target.pr_count)
      - The PR's head branch is deleted (or Forgejo unavailable, per proxy)
    Paused check is handled in tick() before this is reached.
    """
    if _mem().get(_landed_key(target_id)):
        return False
    if _has_pending_dispatch(target_id):
        return False
    seen = _seen_pr_ids(target_id)
    merged = _merged_pr_numbers_observed(target_id)
    if not merged or not seen:
        return False
    # Guard: a newer (higher-numbered) open PR must not exist
    if max(seen) != max(merged):
        return False
    # Guard: all declared PRs must have merged
    target = TargetStore().get(target_id)
    required = target.data.get("pr_count", 1) if target else 1
    if len(merged) < required:
        # Emit one audit comment per (target_id, pr_count) tuple — de-dup by tag
        if not _has_auto_land_waiting_comment(target_id, required):
            episodic.write_observation(
                target_id,
                f"auto-land deferred: {len(merged)}/{required} PRs merged on lapis/{target_id}/*",
                extra_tags=["pm:auto-land:waiting", f"pm:auto-land:waiting:count={required}"],
            )
        return False
    # Branch deletion check: confirm the PR's head branch is gone
    if _forgejo_get_pr is not None and _forgejo_get_branch is not None:
        if target and target.pm_repo:
            try:
                pr_data = _forgejo_get_pr(target.pm_repo, max(merged))
                head_ref = (pr_data.get("head") or {}).get("ref", "")
                if head_ref:
                    try:
                        _forgejo_get_branch(target.pm_repo, head_ref)
                        # Branch still exists — not yet eligible
                        return False
                    except Exception:
                        pass  # 404 or network error → treat as deleted
            except Exception:
                pass  # Forgejo unavailable → fall through to merge-status proxy
    return True


def _spec_bound_ts(target_id: str) -> str:
    """Return ts of the spec:bound comment (proxy for bind time); used for sorting."""
    for c in episodic.all_comments(target_id):
        if episodic.TAG_SPEC in c.tags:
            return c.ts
    return "9999-99-99"  # unbound targets sort last


def _act_auto_land(target_id: str) -> str:
    """Perform automatic landing: arc doc + archive + unbind + mem cleanup.

    Writes pm/landed/<tid> BEFORE unbinding so any partial failure between
    the mem write and unbind is observable and the target can be manually
    completed. The arc doc write is idempotent (overwrites on retry).
    """
    from . import land as land_module

    merged = _merged_pr_numbers_observed(target_id)
    pr_num = max(merged) if merged else 0
    merged_at = _merged_at_for_pr(target_id, pr_num)
    landed_at = _now_iso()

    # 1. Generate and write arc doc (idempotent on retry)
    arc = land_module.generate_arc_doc(target_id)
    path = land_module.write_arc_doc(arc)

    # 2. Write pm/landed/<tid> — commit marker (must succeed before unbind)
    _mem().set(
        _landed_key(target_id),
        json.dumps({"pr_num": pr_num, "merged_at": merged_at,
                    "landed_at": landed_at, "arc_path": str(path)}),
        tags=["lapis-pm", "landed"],
    )

    _deploy_target = TargetStore().get(target_id)
    _post_land_deploy_hook(_deploy_target.pm_repo if _deploy_target else None)

    # 3. Audit comment in the target JSONL (spec-required format)
    episodic.write(
        target_id,
        f"auto-landed: PR #{pr_num} merged at {merged_at}, "
        f"branch deleted, no pending dispatches. arc={path}",
        tags=["pm:auto-land"],
    )

    # 3a. Chain advance: update chain state + auto-fire dependent legs.
    # Read chain_group before archive — `archive()` reloads the target and
    # subsequent state changes can leave the in-memory copy stale.
    store = TargetStore()
    _target_for_chain = store.get(target_id)
    _chain_group = _target_for_chain.data.get("chain_group") or "" if _target_for_chain else ""
    try:
        from . import chain as _chain
        _chain.on_leg_landed(target_id, _chain_group)
        _chain.check_chain_advance(target_id)
    except Exception as _chain_err:
        episodic.write_observation(
            target_id,
            f"chain-advance error (non-fatal): {_chain_err}",
            extra_tags=["pm:chain-error"],
        )

    # 4. Archive + unbind + clear mem state
    store.archive(target_id)
    target = store.get(target_id)
    target.unbind_pm()
    target.save()
    clear_landed_state(target_id)

    return f"action:auto_land:pr={pr_num}:arc={path}"


def _act_auto_land_already_satisfied(target_id: str) -> str:
    """Auto-land via an already_satisfied verdict: arc doc + archive + unbind.

    Mirrors _act_auto_land but uses the verdict's PR number (not a freshly
    merged lapis/* PR) and appends the spec-required "already satisfied" note
    to the arc doc's Origin section.
    """
    from . import land as land_module

    result = _already_satisfied_pending(target_id)
    if result is None:
        return "noop"
    pr_num, evidence, satisfied_ts = result
    landed_at = _now_iso()

    # Generate arc doc with the already-satisfied addendum to Origin.
    already_sat_note = (
        f"_Spec verified already satisfied by PR #{pr_num} at {satisfied_ts}; "
        "landed without new dispatch._"
    )
    arc = land_module.generate_arc_doc(target_id, extra_origin_note=already_sat_note)
    path = land_module.write_arc_doc(arc)

    # Commit marker (same structure as regular auto-land for symmetry)
    _mem().set(
        _landed_key(target_id),
        json.dumps({
            "pr_num": pr_num,
            "merged_at": satisfied_ts,
            "landed_at": landed_at,
            "arc_path": str(path),
            "via": "already_satisfied",
        }),
        tags=["lapis-pm", "landed"],
    )

    _deploy_target = TargetStore().get(target_id)
    _post_land_deploy_hook(_deploy_target.pm_repo if _deploy_target else None)

    # Audit comment (spec-required tag)
    episodic.write(
        target_id,
        f"auto-landed via already_satisfied verdict: PR #{pr_num} was merged before "
        f"dispatch ran. arc={path}",
        tags=["pm:auto-land", "pm:auto-land:already-satisfied"],
    )

    # Chain advance (same as regular auto-land)
    store = TargetStore()
    _target_for_chain = store.get(target_id)
    _chain_group = (
        _target_for_chain.data.get("chain_group") or ""
        if _target_for_chain else ""
    )
    try:
        from . import chain as _chain
        _chain.on_leg_landed(target_id, _chain_group)
        _chain.check_chain_advance(target_id)
    except Exception as _chain_err:
        episodic.write_observation(
            target_id,
            f"chain-advance error (non-fatal): {_chain_err}",
            extra_tags=["pm:chain-error"],
        )

    # Archive + unbind + clear mem state
    store.archive(target_id)
    target = store.get(target_id)
    target.unbind_pm()
    target.save()
    clear_landed_state(target_id)

    return f"auto_land:already_satisfied:pr={pr_num}:arc={path}"


def _act_brief_already_satisfied_invalid(target_id: str) -> str:
    """Post a brief for an already_satisfied verdict where the cited PR failed validation."""
    result = _already_satisfied_invalid_pending(target_id)
    if result is None:
        return "noop"
    pr_num, content = result

    b = brief.synthesize(
        target_id,
        trigger=f"fixer claimed already_satisfied but PR #{pr_num} not found or not merged",
        query=f"Fixer already-satisfied verdict: invalid PR #{pr_num}",
        notify=NotifyPriority.NORMAL,
    )
    set_outstanding_brief(target_id, b.comment_id)
    episodic.write_observation(
        target_id,
        f"Brief posted for invalid already_satisfied verdict (PR #{pr_num}): {b.comment_id}",
        extra_tags=["pm:already-satisfied:invalid-briefed"],
    )
    return f"already_satisfied_invalid_brief:{b.comment_id}"


# ---------------------------------------------------------------------------
# Review-gate kill-switch helpers
# ---------------------------------------------------------------------------

def _review_gate_counter() -> int:
    rec = _mem().get(REVIEW_GATE_COUNTER_KEY)
    if not rec:
        return 0
    try:
        return int(rec["content"])
    except (ValueError, TypeError):
        return 0


def _increment_review_gate_counter() -> int:
    count = _review_gate_counter() + 1
    _mem().set(REVIEW_GATE_COUNTER_KEY, str(count), tags=["lapis-pm", "review-gate"])
    return count


def _review_gate_paused() -> bool:
    rec = _mem().get(REVIEW_GATE_PAUSED_KEY)
    return bool(rec and rec["content"] == "1")


def _set_review_gate_paused(paused: bool) -> None:
    _mem().set(REVIEW_GATE_PAUSED_KEY, "1" if paused else "0",
               tags=["lapis-pm", "review-gate"])


def review_gate_resume() -> int:
    """Reset kill-switch counter. Returns previous count.

    Called by `lapis-pm review-gate resume`.
    """
    count = _review_gate_counter()
    _mem().set(REVIEW_GATE_COUNTER_KEY, "0", tags=["lapis-pm", "review-gate"])
    _set_review_gate_paused(False)
    _mem().delete(REVIEW_GATE_PAUSE_BRIEF_KEY)
    return count


def review_gate_status() -> dict:
    """Return kill-switch state for `lapis-pm review-gate status`."""
    return {
        "counter": _review_gate_counter(),
        "threshold": REVIEW_GATE_THRESHOLD,
        "paused": _review_gate_paused(),
    }


# ---------------------------------------------------------------------------
# Review-gate loop helpers
# ---------------------------------------------------------------------------

def _reviewer_cycle_count(target_id: str, pr_number: int) -> int:
    """Count completed reviewer cycles for this PR via episodic tags."""
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    count = 0
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                if verdict_val != "pending":
                    count += 1
    return count


def _last_review_verdict(target_id: str, pr_number: int) -> dict | None:
    """Return the most recent completed reviewer verdict dict for this PR, or None."""
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    last_comment = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                if verdict_val != "pending":
                    last_comment = c
    if last_comment is None:
        return None
    # Content format: "Reviewer verdict for PR #N:\n<json>"
    content = last_comment.content
    try:
        json_part = content.split("\n", 1)[-1].strip()
        return json.loads(json_part)
    except (json.JSONDecodeError, IndexError):
        # Fall back to extracting verdict from tag
        for t in last_comment.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                return {"verdict": verdict_val, "issues": [], "confidence": 0.0}
        return None


def _review_verdict_for_cycle(target_id: str, pr_number: int, cycle: int) -> dict | None:
    """Return the verdict dict for a specific reviewer cycle, or None if absent.

    Mirrors _last_review_verdict's tag-and-content parsing but filters by cycle.
    """
    prefix = f"pm:reviewer:pr={pr_number}:cycle={cycle}:verdict="
    target_comment = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix):
                verdict_val = t[len(prefix):]
                if verdict_val != "pending":
                    target_comment = c
    if target_comment is None:
        return None
    content = target_comment.content
    try:
        json_part = content.split("\n", 1)[-1].strip()
        return json.loads(json_part)
    except (json.JSONDecodeError, IndexError):
        for t in target_comment.tags:
            if t.startswith(prefix):
                verdict_val = t[len(prefix):]
                return {"verdict": verdict_val, "issues": [], "confidence": 0.0}
        return None


def _fixer_retry_count(target_id: str, pr_number: int) -> int:
    """Count fixer_retry dispatches that actually advanced the PR for this PR.

    Only `status == "processed"` records count — those are the dispatches the
    SHA-advance perceiver (~pm_core.py _perceive_pr_sha_advance) has confirmed
    pushed code. Failed dispatches did not push, so they do not advance cycle
    accounting.
    """
    return sum(
        1 for r in load_dispatched(target_id)
        if r.get("agent_type") == "fixer_retry"
        and r.get("pr_number") == pr_number
        and r.get("status") == "processed"
    )


def _has_pending_reviewer_for_pr(target_id: str, pr_number: int) -> bool:
    return any(
        r.get("status") == "pending"
        and r.get("agent_type") in ("reviewer", "reviewer_fresh")
        and r.get("pr_number") == pr_number
        for r in load_dispatched(target_id)
    )


def _has_pending_fixer_for_pr(target_id: str, pr_number: int) -> bool:
    return any(
        r.get("status") == "pending"
        and r.get("agent_type") == "fixer_retry"
        and r.get("pr_number") == pr_number
        for r in load_dispatched(target_id)
    )


def _collect_review_history(target_id: str, pr_number: int) -> list[dict]:
    """Collect all reviewer verdicts for a PR from episodic, in order."""
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    history: list[dict] = []
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                if verdict_val == "pending":
                    continue
                cycle_part = t.split(":cycle=")[1].split(":")[0]
                try:
                    cycle_num = int(cycle_part)
                except ValueError:
                    cycle_num = 0
                entry: dict = {"cycle": cycle_num, "verdict": verdict_val, "issues": []}
                try:
                    json_part = c.content.split("\n", 1)[-1].strip()
                    data = json.loads(json_part)
                    entry["issues"] = data.get("issues", [])
                except (json.JSONDecodeError, IndexError):
                    pass
                history.append(entry)
    history.sort(key=lambda h: h["cycle"])
    return history


def _last_observed_pr_sha(target_id: str, pr_number: int) -> str | None:
    """Return the most recently observed head SHA for this PR from episodic, or None."""
    prefix = f"pm:pr={pr_number}:sha="
    last: str | None = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix):
                last = t[len(prefix):]
    return last


def _fixer_sha_completion_ts(target_id: str, pr_number: int, dispatch_ts: str) -> str | None:
    """Return ts of first SHA-advance observation for PR N after dispatch_ts, or None."""
    prefix = f"pm:pr={pr_number}:sha="
    for c in episodic.all_comments(target_id):
        if c.ts > dispatch_ts:
            for t in c.tags:
                if t.startswith(prefix):
                    return c.ts
    return None


def _reviewer_dispatch_ts(target_id: str, pr_number: int, cycle: int) -> str | None:
    """Return dispatch ts of reviewer at cycle K for PR N, or None."""
    for r in load_dispatched(target_id):
        if (r.get("agent_type") in ("reviewer", "reviewer_fresh")
                and r.get("pr_number") == pr_number
                and r.get("cycle") == cycle):
            return r.get("ts")
    return None


def _pr_sha_advanced_since(target_id: str, pr_number: int, since_ts: str) -> bool:
    """Return True if any SHA-advance observation for PR N exists after since_ts."""
    prefix = f"pm:pr={pr_number}:sha="
    for c in episodic.all_comments(target_id):
        if c.ts > since_ts:
            for t in c.tags:
                if t.startswith(prefix):
                    return True
    return False


def _active_review_state(target_id: str, open_prs: list[dict]) -> dict | None:
    """Return active review-loop state for status display, or None."""
    classified_ids = _classified_pr_ids(target_id)
    for pr in open_prs:
        pr_number = pr.get("number")
        if pr_number in classified_ids:
            continue
        cycle = _reviewer_cycle_count(target_id, pr_number)
        if cycle == 0 and not _has_pending_reviewer_for_pr(target_id, pr_number):
            continue  # not started yet
        verdict_info = _last_review_verdict(target_id, pr_number)
        return {
            "pr_number": pr_number,
            "cycle": cycle,
            "verdict": verdict_info.get("verdict") if verdict_info else "pending",
            "issues": len(verdict_info.get("issues", [])) if verdict_info else 0,
        }
    return None


# ---------------------------------------------------------------------------
# Perceive helpers
# ---------------------------------------------------------------------------

def _gpu_output_path(task_id: str) -> Path | None:
    for completed_dir, failed_dir in [
        (COMPLETED_DIR, FAILED_DIR),
        (CLAUDE_QUEUE_COMPLETED_DIR, CLAUDE_QUEUE_FAILED_DIR),
    ]:
        p = completed_dir / f"{task_id}-output.md"
        if p.exists():
            return p
        p = failed_dir / f"{task_id}-output.md"
        if p.exists():
            return p
    return None


def _branch_belongs(target_id: str, branch: str) -> bool:
    return branch.startswith(f"lapis/{target_id}/")


def _perceive_prs(target_id: str, repo: str) -> tuple[list[dict], bool]:
    """Return (open_prs, forgejo_ok).  forgejo_ok=False means Forgejo was unreachable."""
    if not get_open_prs:
        return [], False
    try:
        prs = get_open_prs(repo)
    except Exception as e:
        episodic.write_observation(
            target_id, f"PR fetch failed for {repo}: {e}",
            extra_tags=["pm:error"],
        )
        return [], False
    out = []
    for pr in prs:
        head = (pr.get("head") or {}).get("ref") or ""
        if _branch_belongs(target_id, head):
            out.append(pr)
    return out, True


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    kind: str   # "merge" | "advisory_brief" | "hold_brief" | "retry" | "abandon_brief" | "directive_ack"
               #  | "noop_no_change" | "noop_reviewer_in_flight" | "noop_fixer_in_flight"
    payload: dict


def _decide_for_pr(target_id: str, repo: str, pr: dict, pm_authority: str) -> Decision:
    spec_summary = episodic.spec_summary(target_id)
    cls = authority.classify(repo, pr["number"], spec_summary, pm_authority=pm_authority)
    payload = {"classification": cls, "pr": pr}

    # Static hold: held paths always surface immediately (no reviewer needed)
    if cls.static_outcome == authority.StaticOutcome.auto_hold_path:
        return Decision("hold_brief", payload)

    # Auto-merge path: inline Sonnet screen result drives action (unchanged)
    if pm_authority == "auto":
        if cls.verdict == "auto":
            return Decision("merge", payload)
        if cls.verdict == "hold":
            return Decision("hold_brief", payload)
        return Decision("advisory_brief", payload)

    # Advisory / hold path — review-gate loop
    pr_number = pr["number"]

    # Kill-switch: if paused, fall back to inline Sonnet
    if _review_gate_paused():
        return _decide_review_gate_fallback(target_id, cls, pr, pm_authority)

    # Don't double-dispatch while reviewer or fixer is pending for this PR
    if _has_pending_reviewer_for_pr(target_id, pr_number):
        cycle = _reviewer_cycle_count(target_id, pr_number)
        return Decision("noop_reviewer_in_flight",
                        {"pr_number": pr_number, "cycle": cycle})
    if _has_pending_fixer_for_pr(target_id, pr_number):
        dispatch_id = next(
            (r.get("gpu_id", "unknown") for r in load_dispatched(target_id)
             if r.get("status") == "pending"
             and r.get("agent_type") == "fixer_retry"
             and r.get("pr_number") == pr_number),
            "unknown",
        )
        return Decision("noop_fixer_in_flight",
                        {"pr_number": pr_number, "dispatch_id": dispatch_id})

    reviewer_count = _reviewer_cycle_count(target_id, pr_number)
    fixer_count = _fixer_retry_count(target_id, pr_number)
    budget = _REVIEW_CYCLE_BUDGETS.get(pm_authority, 2)
    mode = "fresh" if pm_authority == "hold" else "same"

    if reviewer_count == fixer_count:
        # Either initial dispatch (both==0) or post-fixer dispatch (both==N).
        # For post-fixer (N>0): guard against dispatching reviewer K+1 when the
        # fixer didn't push any code (SHA unchanged since cycle K dispatch).
        # If reviewer_ts is unavailable (e.g. record cleared), default to proceed.
        if reviewer_count > 0:
            reviewer_ts = _reviewer_dispatch_ts(target_id, pr_number, reviewer_count)
            if reviewer_ts and not _pr_sha_advanced_since(target_id, pr_number, reviewer_ts):
                return Decision("noop_no_change", {
                    "reason": (
                        f"PR #{pr_number} head SHA unchanged since reviewer "
                        f"cycle {reviewer_count} dispatch — waiting for fixer commit"
                    )
                })
        if reviewer_count >= budget:
            # Both sides exhausted budget
            history = _collect_review_history(target_id, pr_number)
            return Decision("review_exhausted_brief", {
                "pr": pr, "cls": cls, "history": history,
            })
        # Check kill-switch threshold before dispatching Opus reviewer
        if _review_gate_counter() >= REVIEW_GATE_THRESHOLD:
            _set_review_gate_paused(True)
            return Decision("review_gate_pause", {"pr": pr, "cls": cls})
        next_cycle = reviewer_count + 1
        return Decision("dispatch_reviewer", {
            "pr": pr, "cls": cls, "mode": mode, "cycle": next_cycle,
        })

    # reviewer_count > fixer_count: reviewer has returned a verdict
    verdict_info = _last_review_verdict(target_id, pr_number)
    if verdict_info is None:
        # Shouldn't happen; defensively noop
        return Decision("noop_no_change", {"reason": "reviewer_count > fixer_count but no verdict found"})

    verdict = verdict_info.get("verdict", "needs-human")
    issues = verdict_info.get("issues", [])

    # ---------------------------------------------------------------------------
    # §4 Audit gate — drop unsubstantiated still_present regurgitation
    # Runs only when there are prior issues to audit (cycle ≥ 2, same mode).
    # ---------------------------------------------------------------------------
    prior_issues_for_gate: list = []
    if mode == "same" and reviewer_count >= 2:
        prior_verdict_data = _review_verdict_for_cycle(target_id, pr_number, reviewer_count - 1)
        if prior_verdict_data is not None:
            prior_issues_for_gate = prior_verdict_data.get("issues") or []

    if prior_issues_for_gate:
        prior_resolution_raw: list = verdict_info.get("prior_resolution") or []

        # Validate resolution entries; drop malformed
        valid_resolutions: list = []
        for entry in prior_resolution_raw:
            idx = entry.get("prior_index")
            status = entry.get("status")
            evidence = entry.get("evidence", "")
            if not isinstance(idx, int) or not (0 <= idx < len(prior_issues_for_gate)):
                logger.warning(
                    "dropped malformed prior_resolution entry: prior_index=%r out of range "
                    "(prior_issues len=%d)",
                    idx, len(prior_issues_for_gate),
                )
                continue
            if status not in ("addressed", "still_present", "not_applicable"):
                logger.warning(
                    "dropped malformed prior_resolution entry: unknown status=%r prior_index=%d",
                    status, idx,
                )
                continue
            if not evidence:
                logger.warning(
                    "dropped malformed prior_resolution entry: empty evidence "
                    "prior_index=%d status=%r",
                    idx, status,
                )
                continue
            valid_resolutions.append({"prior_index": idx, "status": status, "evidence": evidence})

        # Build still_present_set from validated resolutions
        still_present_set: set = {
            r["prior_index"] for r in valid_resolutions if r["status"] == "still_present"
        }

        # Also build a set of prior paths that are still_present (for defense-in-depth)
        still_present_paths: set = {
            prior_issues_for_gate[i].get("path", "")
            for i in still_present_set
        }

        # Filter issues
        filtered_issues: list = []
        for iss in issues:
            pi = iss.get("prior_index")
            if pi is not None:
                # Issue cites a prior index — keep only if still_present
                if pi in still_present_set:
                    filtered_issues.append(iss)
                else:
                    logger.warning(
                        "dropped unsubstantiated regurgitated prior issue: prior_index=%d path=%r",
                        pi, iss.get("path"),
                    )
            else:
                # No prior_index — check path-only defense-in-depth
                iss_path = iss.get("path", "")
                prior_paths_all = {p.get("path", "") for p in prior_issues_for_gate}
                if iss_path in prior_paths_all and iss_path not in still_present_paths:
                    logger.warning(
                        "dropped uncited prior-path issue: path=%r", iss_path,
                    )
                else:
                    filtered_issues.append(iss)

        old_verdict = verdict
        issues = filtered_issues

        # Verdict downgrade after filtering
        if old_verdict == "fixable" and not issues:
            verdict = "clean"
            logger.info("verdict downgraded fixable→clean: no substantiated issues")
        elif old_verdict == "needs-human" and not issues and prior_issues_for_gate:
            # All priors addressed/not_applicable, no new issues
            all_resolved = all(
                r["status"] in ("addressed", "not_applicable") for r in valid_resolutions
            )
            if all_resolved and len(valid_resolutions) == len(prior_issues_for_gate):
                verdict = "clean"
                logger.info("verdict downgraded needs-human→clean: all priors resolved, no new issues")

        # §5 Telemetry — per-resolution observations
        for res_entry in valid_resolutions:
            episodic.write_observation(
                target_id,
                f"reviewer cycle {reviewer_count} prior_resolution: "
                f"index={res_entry['prior_index']} "
                f"status={res_entry['status']} "
                f"evidence={res_entry['evidence'][:120]}",
                extra_tags=[
                    "pm:reviewer-prior-resolution",
                    f"pm:cycle={reviewer_count}",
                    f"pm:status={res_entry['status']}",
                ],
            )

        # §5 Telemetry — rollup observation
        kept_count = len([i for i in issues if i.get("prior_index") in still_present_set])
        dropped_count = sum(
            1 for iss in verdict_info.get("issues", [])
            if iss.get("prior_index") is not None and iss.get("prior_index") not in still_present_set
        )
        episodic.write_observation(
            target_id,
            f"audit-gate: cycle={reviewer_count} priors={len(prior_issues_for_gate)} "
            f"still_present_kept={kept_count} "
            f"dropped_unsubstantiated={dropped_count} "
            f"verdict_downgrade={old_verdict}->{verdict}",
            extra_tags=["pm:reviewer-audit-gate", f"pm:cycle={reviewer_count}"],
        )
    # ---------------------------------------------------------------------------
    # End audit gate
    # ---------------------------------------------------------------------------

    if verdict == "clean":
        # Reviewer approved — surface per authority level
        cls.screen_verdict = "clean"
        cls.issues = issues
        if pm_authority == "hold":
            return Decision("hold_brief", payload)
        return Decision("advisory_brief", payload)

    if verdict == "fixable":
        if reviewer_count < budget:
            return Decision("dispatch_fixer_retry", {
                "pr": pr, "cls": cls, "issues": issues,
                "cycle": reviewer_count, "budget": budget,
            })
        history = _collect_review_history(target_id, pr_number)
        return Decision("review_exhausted_brief", {
            "pr": pr, "cls": cls, "history": history,
        })

    # needs-human (or unknown verdict)
    cls.screen_verdict = verdict
    cls.issues = issues
    return Decision("hold_brief", payload)


def _decide_review_gate_fallback(target_id: str, cls: authority.PRClassification,
                                  pr: dict, pm_authority: str) -> Decision:
    """Inline Sonnet fallback when review-gate kill-switch is active."""
    spec_summary = episodic.spec_summary(target_id)
    screen_result = authority.screen(cls.repo, cls.pr_number, spec_summary, cls.diff)
    sv = screen_result.get("verdict", "needs-human")
    cls.screen_verdict = sv
    cls.issues = list(screen_result.get("issues") or [])
    payload = {"classification": cls, "pr": pr}
    if sv == "needs-human":
        cls.verdict = "hold"
        return Decision("hold_brief", payload)
    if sv == "clean" and pm_authority == "advisory":
        return Decision("advisory_brief", payload)
    return Decision("advisory_brief", payload)


def _has_pending_dispatch(target_id: str) -> bool:
    return any(d.get("status") == "pending" for d in load_dispatched(target_id))


# ---------------------------------------------------------------------------
# Act
# ---------------------------------------------------------------------------

def _act_merge(target_id: str, payload: dict) -> str:
    cls: authority.PRClassification = payload["classification"]
    # merge_pr uses OWNER='Erah' hardcoded in forgejo_api.py. Auto-merge only
    # works on Erah/* repos; conductor/* repos will fail here and fall through
    # to the except branch, logging a pm:hold instead of merging.
    try:
        merge_pr(cls.repo, cls.pr_number)
    except Exception as e:
        episodic.write_hold(
            target_id,
            f"Auto-merge attempted for PR #{cls.pr_number} but failed: {e}",
            extra_tags=[f"pm:repo={cls.repo}"],
        )
        return f"action:merge_failed:{e}"
    episodic.write_merge(
        target_id,
        f"Auto-merged PR #{cls.pr_number} ({cls.title}) — "
        f"{cls.diff_loc} LOC, screen={cls.screen_verdict}\n{cls.html_url}",
        extra_tags=[f"pm:repo={cls.repo}", f"pm:pr={cls.pr_number}"],
    )
    _mark_pr_classified(target_id, cls.pr_number)
    return f"action:auto_merge:pr={cls.pr_number}"


def _act_brief(target_id: str, trigger: str, hold: bool, payload: dict) -> str:
    cls: authority.PRClassification = payload["classification"]
    if hold:
        episodic.write_hold(
            target_id,
            f"PR #{cls.pr_number} held: {'; '.join(cls.reasons)}\n"
            f"Title: {cls.title}\n{cls.html_url}",
            extra_tags=[f"pm:repo={cls.repo}", f"pm:pr={cls.pr_number}"],
        )
    # For advisory (non-hold) briefs, derive a closed-form trigger string so
    # the synthesizer emits a pm:brief-options sibling with resolution buttons.
    effective_trigger = trigger
    if not hold:
        if cls.issues:
            effective_trigger = "advisory-screen-issue"
        else:
            effective_trigger = "advisory-clean"
    reviewer_verdict_text: str | None = None
    if effective_trigger == "advisory-clean":
        verdict_info = _last_review_verdict(target_id, cls.pr_number)
        if verdict_info:
            v = verdict_info.get("verdict", "?")
            conf = verdict_info.get("confidence", "?")
            n_issues = len(verdict_info.get("issues") or [])
            corr = verdict_info.get("corroboration_result") or {}
            corr_v = corr.get("verdict")
            parts = [f"Opus reviewer: verdict={v}", f"confidence={conf}", f"issues={n_issues}"]
            if corr_v:
                parts.append(f"corroboration={corr_v}")
            reviewer_verdict_text = "; ".join(parts)
    b = brief.synthesize(
        target_id,
        trigger=effective_trigger,
        query=cls.title,
        diff_snippet=cls.diff or None,
        screen_issues=cls.issues or None,
        pr_number=cls.pr_number if not hold else None,
        notify=NotifyPriority.NORMAL if hold else None,
        reviewer_verdict_text=reviewer_verdict_text,
    )
    _mark_pr_classified(target_id, cls.pr_number)
    set_outstanding_brief_verified(target_id, b.comment_id)
    _post_write_sweep_brief(target_id, b.comment_id)
    if hold:
        kind = "hold"
    elif cls.issues:
        kind = "advisory_screen_issue"
    else:
        kind = "advisory_clean"
    return f"action:brief_emitted:kind={kind}:cid={b.comment_id}"


def _act_retry(target_id: str, dispatch_record: dict) -> str:
    """Redispatch a failed shaped agent (≤MAX_DISPATCH_RETRIES)."""
    agent_type = dispatch_record.get("agent_type", "scout")
    intent = dispatch_record.get("intent", "(no intent)")
    spec_summary = episodic.spec_summary(target_id)
    user_prompt = (
        f"Previous attempt failed. Retry intent: {intent}\n"
        f"Note: previous failure observation is in the thread comments."
    )
    vars_ = {
        "target_id": target_id,
        "spec_summary": spec_summary,
        "repo": dispatch_record.get("repo", ""),
        "question": intent,
        "pr_number": dispatch_record.get("pr_number", ""),
        "slug": dispatch_record.get("slug", "retry"),
    }
    res = _SHAPER.dispatch(agent_type, target_id, user_prompt, vars_=vars_)
    new_record = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": agent_type,
        "intent": intent,
        "repo": dispatch_record.get("repo", ""),
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": dispatch_record.get("retry_count", 0) + 1,
        "parent_gpu_id": dispatch_record.get("gpu_id"),
    }
    append_dispatched(target_id, new_record)
    episodic.write_retry(
        target_id,
        f"Retry #{new_record['retry_count']} dispatched: {agent_type} → {res.task_id}\n"
        f"Intent: {intent}",
        extra_tags=[f"pm:gpu={res.task_id}", f"pm:agent={agent_type}"],
    )
    return f"action:fixer_dispatched:source=init:dispatch={res.task_id}"


def _act_dispatch_reviewer(target_id: str, pr: dict, cls: authority.PRClassification,
                            mode: str = "same", cycle: int = 1) -> str:
    """Dispatch a reviewer agent for the given PR."""
    pr_number = pr["number"]
    repo = cls.repo
    spec_summary = episodic.spec_summary(target_id)

    agent_type = "reviewer_fresh" if mode == "fresh" else "reviewer"

    # Build prior_review context for same-reviewer mode
    prior_review_text = ""
    if mode == "same" and cycle > 1:
        prior = _review_verdict_for_cycle(target_id, pr_number, cycle - 1)
        if prior:
            prior_issues = prior.get("issues", [])
            indexed_issues = "\n".join(
                f"  [{i}] {iss.get('severity', '?').upper()} {iss.get('path', '?')} — {iss.get('note', '')}"
                for i, iss in enumerate(prior_issues)
            ) if prior_issues else "  (none)"
            prior_review_text = (
                f"\n## Prior review context (cycle {cycle - 1})\n\n"
                f"The prior reviewer cycle returned this verdict on an EARLIER state of this branch:\n\n"
                f"Verdict: {prior.get('verdict')}\n"
                f"Prior issues (indexed):\n{indexed_issues}\n\n"
                f"The diff in the user prompt is the CURRENT state. Your task is to classify\n"
                f"EACH prior issue against the current diff. Then return your own fresh\n"
                f"verdict on the current diff.\n\n"
                f"For each prior issue, you MUST emit a `prior_resolution` entry with:\n"
                f'  - "prior_index": the index above\n'
                f'  - "status": "addressed" | "still_present" | "not_applicable"\n'
                f'  - "evidence": for "still_present", a current-diff line/path citation;\n'
                f'                for "addressed", the line/path that fixes it;\n'
                f'                for "not_applicable", a one-sentence reason\n'
                f'                (empty string is NOT acceptable for any status)\n\n'
                f"Then your `issues` array must contain ONLY:\n"
                f'  - prior issues you classified as "still_present" (re-stated, with the\n'
                f"    same path/severity, but `note` updated to reference the current-diff\n"
                f"    evidence), AND\n"
                f"  - any new issues you find in the current diff that were not in the\n"
                f"    prior set.\n\n"
                f'Issues you classified as "addressed" or "not_applicable" must NOT appear\n'
                f"in `issues`. Reviewer cycles are explicit deltas, not stateless re-reads.\n\n"
                f'When re-stating a prior issue in `issues`, include `prior_index: <i>`\n'
                f"pointing to the prior set; new issues omit `prior_index`.\n"
            )

    existing_branch = (pr.get("head") or {}).get("ref") or f"lapis/{target_id}/pr{pr_number}"
    base_branch = (pr.get("base") or {}).get("ref") or "main"

    vars_: dict = {
        "target_id": target_id,
        "spec_summary": spec_summary,
        "repo": repo,
        "repo_cwd": Shaper.resolve_repo_cwd(repo),
        "pr_number": pr_number,
        "slug": f"pr{pr_number}-review-c{cycle}",
        "question": f"review PR #{pr_number}",
        "prior_review": prior_review_text,
        "existing_branch": existing_branch,
        "base_branch": base_branch,
    }

    # Get the diff for the reviewer prompt
    try:
        from agents_core.forgejo import get_pr_diff as _get_diff
        diff_text = _get_diff(repo, pr_number)
        if len(diff_text) > authority.DIFF_INLINE_CAP:
            diff_text = diff_text[:authority.DIFF_INLINE_CAP] + "\n\n... (diff truncated)"
    except Exception:
        diff_text = "(diff unavailable)"

    if mode == "same" and cycle >= 2:
        schema_extra = (
            ', "prior_resolution": ['
            '{"prior_index": <int>, "status": "addressed"|"still_present"|"not_applicable", '
            '"evidence": "<non-empty string>"}]'
            " (required when prior issues were provided); "
            'issues that re-state a prior issue carry "prior_index": <int>; '
            "new issues omit prior_index"
        )
    else:
        schema_extra = ""
    user_prompt = (
        f"Review PR #{pr_number} in {repo}. This is reviewer cycle {cycle}.\n\n"
        f"```diff\n{diff_text}\n```\n\n"
        f'Return JSON: {{"verdict": "clean" | "fixable" | "needs-human", '
        f'"issues": [{{"severity": "high"|"med"|"low", "path": "...", "note": "..."'
        f'{"," if schema_extra else ""}{"prior_index?: <int>" if schema_extra else ""}'
        f'}}]{schema_extra}, "confidence": 0.0-1.0}}'
    )

    # Increment kill-switch counter before dispatch
    _increment_review_gate_counter()

    res = _SHAPER.dispatch(agent_type, target_id, user_prompt, vars_=vars_)

    record = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": agent_type,
        "intent": f"review PR #{pr_number} cycle {cycle}",
        "repo": repo,
        "pr_number": pr_number,
        "cycle": cycle,
        "mode": mode,
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": 0,
    }
    append_dispatched(target_id, record)

    episodic.write_dispatch(
        target_id,
        f"Reviewer dispatched (cycle {cycle}, mode={mode}): {agent_type} → {res.task_id}\n"
        f"PR #{pr_number}: {pr.get('title', '')}",
        extra_tags=[
            f"pm:repo={repo}",
            f"pm:pr={pr_number}",
            f"pm:reviewer:pr={pr_number}:cycle={cycle}:verdict=pending",
        ],
    )
    return f"action:reviewer_dispatched:pr={pr_number}:cycle={cycle}"


def _act_dispatch_fixer_retry(target_id: str, payload: dict) -> str:
    """Dispatch a fixer_retry agent to fix reviewer-flagged issues on an existing PR."""
    pr = payload["pr"]
    cls = payload["cls"]
    issues = payload["issues"]
    cycle = payload.get("cycle", 1)  # reviewer cycle that returned fixable

    pr_number = pr["number"]
    pr_branch = (pr.get("head") or {}).get("ref") or f"lapis/{target_id}/pr{pr_number}"

    issues_text = "\n".join(
        f"- [{i.get('severity', '?')}] {i.get('path', '?')}: {i.get('note', '')}"
        for i in issues
    ) or "(no issues listed)"

    spec_summary = episodic.spec_summary(target_id)

    vars_: dict = {
        "target_id": target_id,
        "spec_summary": spec_summary,
        "repo": cls.repo,
        "repo_cwd": Shaper.resolve_repo_cwd(cls.repo),
        "pr_number": pr_number,
        "slug": f"pr{pr_number}-fix-c{cycle}",
        "existing_branch": pr_branch,
        "question": f"fix reviewer issues on PR #{pr_number}",
    }

    user_prompt = (
        f"PR #{pr_number} reviewer returned `fixable` on cycle {cycle}. "
        f"Fix the listed issues on existing branch `{pr_branch}`.\n\n"
        f"Issues:\n{issues_text}\n\n"
        f"Push to the existing branch — do NOT create a new branch or new PR."
    )

    res = _SHAPER.dispatch("fixer_retry", target_id, user_prompt, vars_=vars_)

    record = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": "fixer_retry",
        "intent": f"fix PR #{pr_number} after reviewer cycle {cycle}",
        "repo": cls.repo,
        "pr_number": pr_number,
        "cycle": cycle,
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": 0,
    }
    append_dispatched(target_id, record)

    episodic.write_dispatch(
        target_id,
        f"Fixer retry dispatched (reviewer cycle {cycle}): fixer_retry → {res.task_id}\n"
        f"PR #{pr_number} issues:\n{issues_text}",
        extra_tags=[
            f"pm:repo={cls.repo}",
            f"pm:pr={pr_number}",
            "pm:fixer-retry",
        ],
    )
    return f"action:fixer_dispatched:source=retry:pr={pr_number}:cycle={cycle}"


def _act_brief_review_exhausted(target_id: str, payload: dict) -> str:
    """Human brief when review cycle budget is exhausted."""
    pr = payload["pr"]
    cls = payload["cls"]
    history = payload.get("history", [])

    history_text = "\n\n".join(
        f"Cycle {h.get('cycle')}: verdict={h.get('verdict')}\n"
        f"Issues: {json.dumps(h.get('issues', []))}"
        for h in history
    ) or "(no history)"

    episodic.write_hold(
        target_id,
        f"PR #{cls.pr_number} review budget exhausted after {len(history)} cycle(s).\n"
        f"Title: {cls.title}\n{cls.html_url}\n\nHistory:\n{history_text}",
        extra_tags=[
            f"pm:repo={cls.repo}",
            f"pm:pr={cls.pr_number}",
            "pm:review-exhausted",
        ],
    )
    b = brief.synthesize(
        target_id,
        trigger=f"Review budget exhausted for PR #{cls.pr_number} — human judgment needed",
        query=f"PR #{cls.pr_number} review exhausted: {cls.title}",
        diff_snippet=cls.diff or None,
        screen_issues=None,
        notify=NotifyPriority.HIGH,
    )
    set_outstanding_brief(target_id, b.comment_id)
    _mark_pr_classified(target_id, cls.pr_number)
    return f"action:review_exhausted_brief:cid={b.comment_id}"


def _act_review_gate_pause(target_id: str, payload: dict) -> str:
    """Emit a single pause brief when the kill-switch threshold is exceeded."""
    # Only post the pause brief once (idempotent)
    rec = _mem().get(REVIEW_GATE_PAUSE_BRIEF_KEY)
    if rec:
        return "action:review_gate_paused:already_briefed"

    count = _review_gate_counter()
    episodic.write_observation(
        target_id,
        f"Review-gate loop soft-paused after {count} Opus reviewer calls in the past 7d. "
        f"Falling back to inline-Sonnet behavior for new PRs. "
        f"Resume with `lapis-pm review-gate resume`.",
        extra_tags=["pm:review-gate-paused"],
    )
    b = brief.synthesize(
        target_id,
        trigger=f"Review-gate loop soft-paused after {count} Opus reviewer calls",
        query="review-gate pause — token budget exceeded",
        notify=NotifyPriority.HIGH,
    )
    set_outstanding_brief(target_id, b.comment_id)
    _mem().set(REVIEW_GATE_PAUSE_BRIEF_KEY, b.comment_id,
               tags=["lapis-pm", "review-gate"])
    return f"action:review_gate_paused:cid={b.comment_id}"


# ---------------------------------------------------------------------------
# Encode percepts
# ---------------------------------------------------------------------------

def _encode_user_comments(target_id: str, comments: list) -> list:
    """Return list of comments that triggered any state change worth acting on.

    User comments themselves are already in the JSONL — we don't re-encode
    them as observations. We return the directive comments for `decide` use.
    """
    directives = [c for c in comments if episodic.TAG_HUMAN_DIRECTIVE in c.tags]
    acks = [c for c in comments if episodic.TAG_HUMAN_ACK in c.tags]
    if acks:
        clear_outstanding_brief(target_id, reason="user_ack")
        episodic.write_observation(
            target_id, f"Brief acknowledged by user ({len(acks)} ack(s)).",
        )
    return directives


_GPU_FAIL_PREFIXES = ("ERROR", "EXIT ", "TIMEOUT")


def _encode_pr_sha_updates(target_id: str, open_prs: list[dict]) -> int:
    """Write SHA-advance observation when a PR's head SHA changes. Returns count written."""
    written = 0
    for pr in open_prs:
        pr_num = pr.get("number")
        sha = (pr.get("head") or {}).get("sha")
        if not sha or not pr_num:
            continue
        if sha != _last_observed_pr_sha(target_id, pr_num):
            episodic.write_observation(
                target_id,
                f"PR #{pr_num} head SHA: {sha}",
                extra_tags=[f"pm:pr={pr_num}", f"pm:pr={pr_num}:sha={sha}"],
            )
            # SHA changed → invalidate classification so the next decide loop
            # re-screens the new code instead of noop'ing on a stale brief.
            ids = _classified_pr_ids(target_id)
            if pr_num in ids:
                ids.discard(pr_num)
                _mem().set(
                    _classified_prs_key(target_id),
                    json.dumps(sorted(ids)),
                    tags=["lapis-pm", "classified-prs"],
                )
            written += 1
    return written


def _recover_reviewer_verdict(raw: str) -> dict | None:
    """Attempt to recover a reviewer verdict dict from malformed JSON.

    Recovery strategies tried in order:
    1. Strip code fences (``` or ```json) and re-parse. (Mostly redundant with
       the eager fence-strip in _encode_gpu_results' live path; retained so
       the helper is also useful when called directly.)
    2. Extract outermost {...} via string-aware balanced-brace scan and
       re-parse. Braces inside JSON string literals are ignored so prose like
       `"note": "} oops"` doesn't close the object early.
    3. Regex extraction of verdict literal as last resort — returns
       issues=[], confidence=0.0 if the structure is otherwise unparseable.

    Returns a parsed dict on success, None if all strategies fail.
    """
    # Strategy 1: strip code fences
    stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    stripped = re.sub(r"\s*```\s*$", "", stripped.strip())
    if stripped != raw.strip():
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Strategy 2: extract outermost {...} via string-aware balanced-brace scan
    start = raw.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(raw[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # Strategy 3: regex extraction of verdict field (verdict-only recovery).
    # confidence=0.0 signals "structure was lost, only the verdict survived"
    # so any downstream consumer can treat it as low-trust if needed.
    m = re.search(r'"verdict"\s*:\s*"(clean|fixable|needs-human|blocked)"', raw)
    if m:
        return {"verdict": m.group(1), "issues": [], "confidence": 0.0}

    return None


def _encode_gpu_results(target_id: str) -> tuple[int, list[dict]]:
    """For each pending dispatch, check for completion and encode result.

    Returns (total_encoded, failed_for_retry).
    """
    # Clear any stale tick-local corroboration entries for this target so
    # _persist_review_state_cache always sees fresh data from this tick.
    for k in list(_tick_corr_cache):
        if k[0] == target_id:
            del _tick_corr_cache[k]
    records = load_dispatched(target_id)
    failed_for_retry: list[dict] = []
    total_encoded = 0
    changed = False
    for rec in records:
        if rec.get("status") != "pending":
            continue

        # fixer_retry uses PR SHA advancement as completion signal (preferred per spec)
        # rather than GPU output file, so we can detect completion even when the
        # fixer pushed code but the queue output path differs (e.g. claude-queue).
        if rec.get("agent_type") == "fixer_retry":
            pr_num = rec.get("pr_number")
            dispatch_ts = rec.get("ts", "")
            if pr_num is not None:
                completion_ts = _fixer_sha_completion_ts(target_id, pr_num, dispatch_ts)
                if completion_ts:
                    rec["status"] = "processed"
                    rec["completed_at"] = completion_ts
                    changed = True
                    total_encoded += 1
                    episodic.write_result(
                        target_id,
                        f"Fixer retry for PR #{pr_num} completed: head SHA advanced after dispatch",
                        extra_tags=[
                            f"pm:gpu={rec['gpu_id']}",
                            "pm:agent=fixer_retry",
                            f"pm:pr={pr_num}",
                        ],
                    )
                continue  # fixer_retry uses SHA-advance signal, not GPU output file
            # fixer_retry without pr_number: fall through to GPU output file path
            # as defensive fallback (shouldn't happen in practice).

        out_path = _gpu_output_path(rec["gpu_id"])
        if not out_path:
            continue
        try:
            text = out_path.read_text(encoding="utf-8")
        except OSError:
            continue

        is_failure = (
            out_path.parent in (FAILED_DIR, CLAUDE_QUEUE_FAILED_DIR)
            or any(text.lstrip().startswith(p) for p in _GPU_FAIL_PREFIXES)
        )

        # Already-satisfied verdict check: fixer may write a machine-readable
        # verdict sidecar instead of opening a PR (lapis-pm-fixer-already-done-verdict).
        # Check before confabulation so a valid verdict is never marked confabulated.
        _fixer_spec_id = rec.get("spec_id") if rec.get("agent_type") == "fixer" else None
        if _fixer_spec_id and not is_failure:
            _verdict_raw = _read_fixer_verdict(_fixer_spec_id)
            _consume_fixer_verdict(_fixer_spec_id)
            if _verdict_raw is not None:
                _verdict_handled = _handle_already_satisfied_verdict(
                    target_id, rec, _verdict_raw
                )
                if _verdict_handled:
                    # Also consume meta sidecar (verdict takes priority)
                    _consume_fixer_meta(_fixer_spec_id)
                    rec["status"] = "processed"
                    rec["completed_at"] = _now_iso()
                    changed = True
                    total_encoded += 1
                    continue  # Skip confabulation + normal result encoding

        # Confabulation check: fixer agents that produced substantial prose
        # without using tools are stochastic failures and should be retried.
        # Meta sidecar is only written for fixers (capture_meta=true in registry.yaml).
        confabulation_note = ""
        spec_id = rec.get("spec_id") if rec.get("agent_type") == "fixer" else None
        if spec_id and not is_failure:
            meta = _read_fixer_meta(spec_id)
            if meta and meta.get("confabulated"):
                is_failure = True
                confabulation_note = (
                    f"CONFABULATED — fixer produced "
                    f"{meta.get('char_count', 0)} chars with no PR URL "
                    f"or git evidence (num_turns={meta.get('num_turns')}, "
                    f"basis={meta.get('decision_basis', '?')})\n\n"
                )
        # Whether we read the meta or it was missing/clean, drop the sidecar
        # now that the dispatch is no longer pending. Avoids /srv/lapis/gpu-queue/
        # shaped/ accumulating stale meta files indefinitely.
        if spec_id:
            _consume_fixer_meta(spec_id)

        rec["status"] = "failed" if is_failure else "processed"
        rec["completed_at"] = _now_iso()
        changed = True

        snippet = text.strip()
        if len(snippet) > 1500:
            snippet = snippet[:1500] + "\n…[truncated, full output: " + str(out_path) + "]"
        snippet = confabulation_note + snippet
        tags = [
            f"pm:gpu={rec['gpu_id']}",
            f"pm:agent={rec.get('agent_type', 'unknown')}",
        ]
        if is_failure:
            episodic.write_result(
                target_id,
                f"FAILED — {rec.get('agent_type')} task {rec['gpu_id']}\n"
                f"Intent: {rec.get('intent')}\n\n{snippet}",
                extra_tags=tags + ["pm:failure"],
            )
            failed_for_retry.append(rec)
        else:
            # Reviewer agents: parse JSON output and write reviewer-tagged episodic entry
            if rec.get("agent_type") in ("reviewer", "reviewer_fresh"):
                pr_num = rec.get("pr_number", "?")
                cycle_num = rec.get("cycle", 1)
                raw_output = text.strip()
                if raw_output.startswith("```"):
                    raw_output = re.sub(r"^```(?:json)?\s*", "", raw_output)
                    raw_output = re.sub(r"\s*```$", "", raw_output.strip())
                parse_recovered = False
                try:
                    verdict_data = json.loads(raw_output)
                    verdict_val = verdict_data.get("verdict", "needs-human")
                    stored_json = json.dumps(verdict_data)
                except json.JSONDecodeError:
                    recovered = _recover_reviewer_verdict(raw_output)
                    if recovered is not None:
                        verdict_val = recovered.get("verdict", "needs-human")
                        stored_json = json.dumps(recovered)
                        parse_recovered = True
                    else:
                        verdict_val = "needs-human"
                        stored_json = json.dumps({"verdict": "needs-human", "issues": [],
                                                  "confidence": 0.0,
                                                  "_parse_error": raw_output[:200]})
                # Corroboration follow-up pass (q4 resolution: two LLM calls per
                # review; prompt-cached overlap; reviewer read-only invariant preserved
                # — corroboration_result is additive only, never mutates mainline fields).
                try:
                    _corr_diff = _diff_text_for_corr(rec.get("repo", ""), pr_num)
                    _corr_result = _run_corroboration_pass_sync(_corr_diff, rec.get("repo", ""))
                    _stored_dict = json.loads(stored_json)
                    _stored_dict["corroboration_result"] = _corr_result
                    stored_json = json.dumps(_stored_dict)
                    if isinstance(pr_num, int):
                        _tick_corr_cache[(target_id, pr_num)] = _corr_result
                except Exception as _corr_exc:
                    episodic.write_observation(
                        target_id,
                        f"corroboration pass skipped for PR #{pr_num}: {type(_corr_exc).__name__}",
                        extra_tags=["pm:corroboration-skipped"],
                    )  # best-effort; never fail verdict encoding
                result_tags = tags + [
                    f"pm:reviewer:pr={pr_num}:cycle={cycle_num}:verdict={verdict_val}",
                    f"pm:pr={pr_num}",
                ]
                if parse_recovered:
                    result_tags.append("pm:reviewer:parse-recovered")
                episodic.write_result(
                    target_id,
                    f"Reviewer verdict for PR #{pr_num}:\n{stored_json}",
                    extra_tags=result_tags,
                )
            else:
                episodic.write_result(
                    target_id,
                    f"Completed — {rec.get('agent_type')} task {rec['gpu_id']}\n"
                    f"Intent: {rec.get('intent')}\n\n{snippet}",
                    extra_tags=tags,
                )
        total_encoded += 1
    if changed:
        save_dispatched(target_id, records)
    return total_encoded, failed_for_retry


def _encode_new_prs(target_id: str, repo: str, prs: list[dict],
                    seen_ids: set[int]) -> list[dict]:
    """Write pm:observation for newly-seen PRs. Returns the new ones."""
    new_prs = [pr for pr in prs if pr.get("number") not in seen_ids]
    for pr in new_prs:
        episodic.write_observation(
            target_id,
            f"PR #{pr.get('number')} opened in {repo}: {pr.get('title')}\n"
            f"{pr.get('html_url', '')}",
            extra_tags=[f"pm:repo={repo}", f"pm:pr={pr.get('number')}"],
        )
    return new_prs


def _seen_pr_ids(target_id: str) -> set[int]:
    """Recover already-encoded PRs by scanning prior comment tags."""
    out: set[int] = set()
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:pr="):
                try:
                    out.add(int(t.split("=", 1)[1]))
                except (ValueError, IndexError):
                    pass
    return out


# ---------------------------------------------------------------------------
# Dispatch–queue reconciliation
# ---------------------------------------------------------------------------

def _reconcile_dispatched_with_queue(target_id: str) -> int:
    """Flip pending dispatch records to terminal state based on ClaudeQueue.

    Returns count of records flipped. Idempotent — calling twice in a row
    with no new queue activity is a no-op (returns 0 the second time).

    Reads get_recent_failed(limit=50) and get_recent_completed(limit=50)
    once per call (not once per record). Matches on gpu_id (the queue's
    `id` field) — covers all agent types naturally without agent-specific
    logic. Only `pending` records are eligible; already-terminal records
    are never re-flipped.

    Limit=50: large enough that a 10-min tick interval plus typical queue
    throughput cannot push a terminal record past the window before we
    observe it. Raise to 200 if get_recent_* proves cheap and throughput
    grows significantly.

    Carve-outs (two):
    - fixer_retry + completed: left pending so the SHA-advance perceiver in
      _encode_gpu_results owns the terminal flip.
    - reviewer/reviewer_fresh + completed: left pending so the output-file
      verdict-encoder in _encode_gpu_results owns the terminal flip (reads
      the output file, parses JSON, writes "Reviewer verdict for PR #N:").
    Failed flips for both carve-out types are still permitted — a crashed
    job produces no output file, so the record must fail rather than wait.
    """
    if _ClaudeQueue is None:
        return 0

    try:
        cq = _ClaudeQueue()
        failed_entries = cq.get_recent_failed(limit=50)
        completed_entries = cq.get_recent_completed(limit=50)
    except Exception:
        return 0

    # Build gpu_id → terminal-state index from both lists.
    # failed wins if a task somehow appears in both (shouldn't happen).
    terminal: dict[str, dict] = {}
    for entry in completed_entries:
        task_id = entry.get("id")
        if task_id:
            terminal[task_id] = {
                "state": "processed",
                "error": None,
                "completed_at": entry.get("completed_at"),
            }
    for entry in failed_entries:
        task_id = entry.get("id")
        if task_id:
            terminal[task_id] = {
                "state": "failed",
                "error": entry.get("error"),
                "completed_at": entry.get("completed_at"),
            }

    records = load_dispatched(target_id)
    flipped = 0
    changed = False

    # Pre-load existing reconcile tags for de-dup: one audit comment per
    # gpu_id, never written again if the tag already exists in the JSONL.
    existing_dedup_tags: set[str] = set()
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:dispatch-reconciled:gpu="):
                existing_dedup_tags.add(t)

    for rec in records:
        if rec.get("status") != "pending":
            # Only pending records are touched. Terminal records are left alone.
            continue
        gpu_id = rec.get("gpu_id")
        if not gpu_id:
            continue
        t = terminal.get(gpu_id)
        if t is None:
            continue

        # fixer_retry carve-out: SHA-advance perception is the sole authority
        # for fixer_retry → processed. If the queue says "completed" for a
        # fixer_retry, we leave the record as pending — the SHA-advance
        # perceiver will flip it when it confirms the PR head advanced.
        # Failed flips for fixer_retry are still permitted (the job crashed or
        # was rejected; that doesn't advance the cycle regardless).
        if rec.get("agent_type") == "fixer_retry" and t["state"] == "processed":
            continue  # SHA-advance perceiver owns fixer_retry → processed

        # reviewer/reviewer_fresh carve-out: output-file verdict-encoder in
        # _encode_gpu_results is the sole authority for reviewer → processed.
        # If the queue says "completed" for a reviewer/reviewer_fresh, leave
        # the record as pending — _encode_gpu_results reads the output file,
        # parses the JSON verdict, writes the "Reviewer verdict for PR #N:"
        # episodic entry, and then flips to processed.
        # Failed flips for reviewers are still permitted (crashed job → no
        # output file coming; must fail fast rather than wait forever).
        if rec.get("agent_type") in ("reviewer", "reviewer_fresh") and t["state"] == "processed":
            continue  # output-file verdict-encoder owns reviewer → processed

        # Flip the record to the terminal state.
        rec["status"] = t["state"]
        if t["error"] is not None:
            rec["error"] = t["error"]
        if t["completed_at"] is not None:
            rec["completed_at"] = t["completed_at"]
        changed = True
        flipped += 1

        # Audit comment — one per flip, de-duped by tag so a second call
        # with the same gpu_id in a later tick writes nothing.
        dedup_tag = f"pm:dispatch-reconciled:gpu={gpu_id}"
        if dedup_tag not in existing_dedup_tags:
            error_note = f" ({t['error']})" if t["error"] else ""
            episodic.write_observation(
                target_id,
                f"Reconciled dispatch {gpu_id}: pending → {t['state']}{error_note}",
                extra_tags=["pm:dispatch-reconciled", dedup_tag],
            )
            existing_dedup_tags.add(dedup_tag)

    if changed:
        save_dispatched(target_id, records)

    return flipped


# ---------------------------------------------------------------------------
# Lost-dispatch detection and handling
# ---------------------------------------------------------------------------

def _collect_merged_pr_created_ats(target_id: str) -> list[str]:
    """Return PR creation timestamps from pm:pr-merged episodic observations.

    Parses 'created_at=<ts>' from observation content (written by
    _encode_merged_prs since the lost-dispatch feature landed). Falls back to
    the observation's own timestamp for older records that pre-date this field.
    """
    result: list[str] = []
    for c in episodic.all_comments(target_id):
        if not any(t.startswith("pm:pr-merged:") for t in c.tags):
            continue
        m = re.search(r"created_at=(\S+)", c.content)
        result.append(m.group(1) if m else c.ts)
    return result


def _find_lost_fixer_dispatches(
    target_id: str,
    records: list[dict],
    open_prs: list[dict],
    forgejo_ok: bool,
) -> tuple[list[dict], list[tuple[dict, dict | None]]]:
    """Classify terminal fixer dispatches with no corresponding PR as lost.

    Returns (needs_retry, needs_brief):
      needs_retry: original fixer records where lost_retry_count == 0 and no
                   pending retry child exists (never preempt a live fixer).
      needs_brief: (original, retry_child_or_None) pairs where lost_retry_count >= 1
                   and the retry child is absent or also terminal.

    A `lost` classification requires forgejo_ok=True — stale empty PR list is
    indistinguishable from real empty; returns ([], []) if Forgejo was unreachable.

    Only original dispatches (no parent_gpu_id) are classified; retry children
    are located by parent pointer and returned as the second tuple element for
    the brief path (so both dispatch IDs can appear in the brief body).
    """
    if not forgejo_ok:
        return [], []

    # Lazily loaded on first miss — avoids scanning episodic on ticks where no
    # terminal fixer dispatch exists or all are covered by open_prs.
    merged_pr_created_ats: list[str] | None = None
    # New: lazy caches for merge-state and head-SHA-advance gates (Change 1).
    seen_prs: set[int] | None = None
    merged_prs_for_target: set[int] | None = None

    needs_retry: list[dict] = []
    needs_brief: list[tuple[dict, dict | None]] = []

    for rec in records:
        if rec.get("agent_type") != "fixer":
            continue
        if rec.get("parent_gpu_id"):
            continue  # Only classify originals, not retry children
        if rec.get("status") not in ("processed", "failed"):
            continue  # Not terminal — job may still open a PR

        dispatch_ts = rec.get("ts", "")

        # Does a matching PR exist (created at or after this dispatch)?
        has_pr = any(pr.get("created_at", "") >= dispatch_ts for pr in open_prs)
        if not has_pr:
            if merged_pr_created_ats is None:
                merged_pr_created_ats = _collect_merged_pr_created_ats(target_id)
            has_pr = any(ts >= dispatch_ts for ts in merged_pr_created_ats)

        # Gate 1 (new): a merged PR's merged_at >= dispatch_ts, restricted to
        # _seen_pr_ids(target_id).  Covers the foyer-shaped case where the fixer
        # pushed onto an *existing* branch rather than opening a new PR.
        if not has_pr:
            if seen_prs is None:
                seen_prs = _seen_pr_ids(target_id)
            if merged_prs_for_target is None:
                merged_prs_for_target = _merged_pr_numbers_observed(target_id)
            for n in merged_prs_for_target & seen_prs:
                if _merged_at_for_pr(target_id, n) >= dispatch_ts:
                    has_pr = True
                    break

        # Gate 2 (new): a head-SHA-advance observation (pm:pr=<n>:sha=<sha>) at
        # ts >= dispatch_ts, restricted to _seen_pr_ids(target_id).  Covers the
        # case where the fixer pushed a commit and the PR is still open.
        if not has_pr:
            if seen_prs is None:
                seen_prs = _seen_pr_ids(target_id)
            for c in episodic.all_comments(target_id):
                if c.ts >= dispatch_ts:
                    for t in c.tags:
                        if t.startswith("pm:pr=") and ":sha=" in t:
                            try:
                                n = int(t[len("pm:pr="):].split(":sha=")[0])
                                if n in seen_prs:
                                    has_pr = True
                                    break
                            except (ValueError, IndexError):
                                pass
                if has_pr:
                    break

        if has_pr:
            continue  # PR exists; not lost

        # Locate the youngest fixer retry child (if any)
        orig_gpu_id = rec.get("gpu_id", "")
        retry_child: dict | None = next(
            (r for r in records
             if r.get("parent_gpu_id") == orig_gpu_id
             and r.get("agent_type") == "fixer"),
            None,
        )
        # Never preempt a live fixer (child pending = retry in flight)
        if retry_child is not None and retry_child.get("status") == "pending":
            continue

        lost_retry_count = rec.get("lost_retry_count", 0)
        if lost_retry_count == 0:
            needs_retry.append(rec)
        else:
            needs_brief.append((rec, retry_child))

    return needs_retry, needs_brief


def _act_lost_fixer_retry(target_id: str, rec: dict) -> str:
    """Re-dispatch a lost fixer with the same intent. Retry budget: exactly 1.

    Increments lost_retry_count on the original record and appends a child
    dispatch record with parent_gpu_id pointing back to the original.
    Logs decision=fixer_lost:retrying:dispatch=<id>.
    """
    agent_type = rec.get("agent_type", "fixer")
    intent = rec.get("intent", "(no intent)")
    spec_summary = episodic.spec_summary(target_id)
    vars_ = {
        "target_id": target_id,
        "spec_summary": spec_summary,
        "repo": rec.get("repo", ""),
        "question": intent,
        "pr_number": "",
        "slug": rec.get("slug", "forced"),
    }
    res = _SHAPER.dispatch(agent_type, target_id, intent, vars_=vars_)

    orig_gpu_id = rec.get("gpu_id", "?")
    new_record = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": agent_type,
        "intent": intent,
        "repo": rec.get("repo", ""),
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": 0,
        "lost_retry_count": 0,
        "parent_gpu_id": orig_gpu_id,
    }

    # Increment lost_retry_count on original and append child in one save
    records = load_dispatched(target_id)
    for r in records:
        if r.get("gpu_id") == orig_gpu_id:
            r["lost_retry_count"] = 1
            break
    records.append(new_record)
    save_dispatched(target_id, records)

    episodic.write_observation(
        target_id,
        f"Lost dispatch {orig_gpu_id}: retrying → {res.task_id}\nIntent: {intent}",
        extra_tags=["pm:lost-dispatch-retry", f"pm:gpu={res.task_id}",
                    f"pm:agent={agent_type}"],
    )
    return f"fixer_lost:retrying:dispatch={orig_gpu_id}"


def _act_lost_brief(
    target_id: str, original_rec: dict, retry_rec: dict | None
) -> str:
    """Emit a lost-dispatch brief (both attempts terminated, no PR). NORMAL priority.

    Brief body includes both dispatch IDs, error strings, and spec reference.
    Options: retry-again, amend-spec-and-retry, unbind.
    Logs decision=fixer_lost:briefing:dispatches=<id1>,<id2>.

    Idempotency guard: if a pm:brief-options comment tagged
    pm:lost-original-gpu=<orig_gpu_id> already exists for this dispatch,
    and the outstanding-brief mem key still matches, the brief is not
    re-composed.  Returns noop:lost-brief-suppressed:gpu=<id> in that case.
    """
    orig_id = original_rec.get("gpu_id", "unknown")
    orig_error = original_rec.get("error") or "no error recorded"
    retry_id = retry_rec.get("gpu_id", "none") if retry_rec else "none"
    retry_error = (retry_rec.get("error") or "no error recorded") if retry_rec else ""

    # --- Idempotency guard (Change 2) ---
    gpu_tag = f"pm:lost-original-gpu={orig_id}"
    existing_brief_id: str | None = None
    for c in episodic.all_comments(target_id):
        if "pm:brief-options" not in c.tags:
            continue
        if gpu_tag not in c.tags:
            continue
        try:
            data = json.loads(c.content)
        except (json.JSONDecodeError, AttributeError):
            continue
        if data.get("trigger") == "lost-dispatch":
            existing_brief_id = data.get("brief_id")
            break

    if existing_brief_id is not None:
        current_outstanding = get_outstanding_brief(target_id)
        if current_outstanding == existing_brief_id:
            # Brief is outstanding and matches — suppress entirely (noop).
            return f"noop:lost-brief-suppressed:gpu={orig_id}"
        else:
            # Mem key is stale (lapis-pm-outstanding-brief-write-verify-v0
            # addresses this path separately).  Still suppress composition but
            # write a single observation so the mismatch is visible in episodic.
            episodic.write_observation(
                target_id,
                f"Lost-brief suppressed (mem-stale): orig_gpu={orig_id} "
                f"existing_brief={existing_brief_id} "
                f"mem_key={current_outstanding!r}",
                extra_tags=["pm:lost-brief-suppressed", gpu_tag],
            )
            return f"noop:lost-brief-suppressed:gpu={orig_id}"
    # --- End idempotency guard ---

    spec_ref = (episodic.spec(target_id) or "")[:80] or "(spec not found)"
    spec_path = f"/srv/lapis/planning/specs/{target_id}.md"

    query = (
        f"Two fixer dispatches for {target_id} terminated without opening a PR.\n"
        f"Original dispatch: {orig_id} — error: {orig_error}\n"
        f"Retry dispatch: {retry_id}"
        + (f" — error: {retry_error}" if retry_error else "")
        + f"\nSpec: {spec_path} — {spec_ref}"
    )

    b = brief.synthesize(
        target_id,
        trigger="lost-dispatch",
        query=query,
        notify=NotifyPriority.NORMAL,
        options_extra_tags=[gpu_tag],
    )
    set_outstanding_brief(target_id, b.comment_id)

    dispatch_ids = f"{orig_id},{retry_id}" if retry_rec else orig_id
    return f"fixer_lost:briefing:dispatches={dispatch_ids}"


# ---------------------------------------------------------------------------
# Brief-decision directive consumer
# ---------------------------------------------------------------------------

_DIRECTIVES_BASE = Path("/srv/lapis/directives/brief-decisions")


def _consume_brief_decisions(target_id: str) -> str | None:
    """Consume at most one brief-decision directive for target_id this tick.

    Glob: /srv/lapis/directives/brief-decisions/<target_id>__*.json  (top-level = pending).
    Also recovers any processing/<target_id>__*.json left by a prior crashed tick.

    Returns decision_str "brief_decision_applied:<brief_id>:<option_id>" if a
    directive was consumed, else None.
    """
    base = _DIRECTIVES_BASE
    pending_dir = base
    processing_dir = base / "processing"
    applied_dir = base / "applied"
    failed_dir = base / "failed"

    for d in (processing_dir, applied_dir, failed_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _load_candidates(directory: Path, glob_pattern: str) -> list[Path]:
        try:
            return sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
        except OSError:
            return []

    # Recover any file stuck in processing/ from a prior crashed tick.
    processing_candidates = _load_candidates(processing_dir, f"{target_id}__*.json")
    # Also pick up top-level pending files.
    pending_candidates = _load_candidates(pending_dir, f"{target_id}__*.json")

    # Process recovery files first (they're already renamed); then pending.
    candidates = [(f, True) for f in processing_candidates] + \
                 [(f, False) for f in pending_candidates]

    for src_path, already_processing in candidates:
        stem = src_path.stem  # <target_id>__<brief_id>
        processing_path = processing_dir / src_path.name
        applied_path = applied_dir / src_path.name
        failed_path = failed_dir / src_path.name

        # Load directive JSON.
        try:
            directive = json.loads(src_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            episodic.write_observation(
                target_id,
                f"brief-decision directive unreadable ({src_path.name}): {exc}",
                extra_tags=["pm:error"],
            )
            continue

        brief_id = directive.get("brief_id", "")
        option_id = directive.get("option_id", "")

        if not already_processing:
            # Atomic rename to processing/ before calling apply_decision.
            try:
                os.rename(src_path, processing_path)
            except OSError as exc:
                episodic.write_observation(
                    target_id,
                    f"brief-decision rename-to-processing failed ({src_path.name}): {exc}",
                    extra_tags=["pm:error"],
                )
                continue

        # Apply the decision (idempotent).
        result = brief.apply_decision(target_id, brief_id, option_id)

        ts = _now_iso()
        if result.get("ok"):
            # Write augmented payload (directive + result + applied_ts) into
            # applied/ then unlink processing/.  Earlier code wrote the augmented
            # payload then os.rename'd processing_path over it, which silently
            # clobbered the result/applied_ts fields.
            result_payload = json.dumps({**directive, "result": result, "applied_ts": ts},
                                        ensure_ascii=False)
            try:
                applied_path.write_text(result_payload)
                processing_path.unlink(missing_ok=True)
            except OSError:
                # best-effort: leave in processing if write/unlink fails
                pass
            return f"action:brief_decision_applied:{brief_id}:{option_id}"
        else:
            err = result.get("error", "unknown")
            failed_payload = json.dumps({**directive, "error": err, "failed_ts": ts},
                                        ensure_ascii=False)
            try:
                failed_path.write_text(failed_payload)
                processing_path.unlink(missing_ok=True)
            except OSError:
                pass
            episodic.write_observation(
                target_id,
                f"brief-decision directive failed ({brief_id}/{option_id}): {err}",
                extra_tags=["pm:error"],
            )
            # A failed directive is still "consumed" this tick — return None
            # so the rest of the decide chain can run.
            return None

    return None


# ---------------------------------------------------------------------------
# Eval-gate helpers (synapse-eval-gate-v1)
# ---------------------------------------------------------------------------

def _act_eval_gate_brief(
    target_id: str, result, repo: str
) -> str:
    """Emit an advisory brief for a regressed or unverified eval-gate result.

    Calls brief.synthesize() with the documented parameter mapping:
      trigger  = "synapse_eval_regressed" | "synapse_eval_unverified"
      query    = "PR #N retrieval-quality check"
      diff_snippet = result.summary_text   (delta table)
      screen_issues = []
      pr_number = result.pr_number
      notify    = None    (advisory only; no Pushover per invariant)

    Marks the result actioned in the cache and sets the outstanding brief.
    Does NOT mark the PR classified so the reviewer/fixer can proceed on the
    next tick.
    """
    trigger = (
        "synapse_eval_regressed"
        if result.status == "regressed"
        else "synapse_eval_unverified"
    )
    b = brief.synthesize(
        target_id,
        trigger=trigger,
        query=f"PR #{result.pr_number} retrieval-quality check",
        diff_snippet=result.summary_text,
        screen_issues=[],
        pr_number=result.pr_number,
        notify=None,
    )
    if _eval_gate:
        _eval_gate.mark_eval_actioned(result.head_sha)
    episodic.write_observation(
        target_id,
        f"Eval-gate brief emitted: PR #{result.pr_number} "
        f"sha={result.head_sha[:8]} status={result.status} cid={b.comment_id}",
        extra_tags=[
            f"pm:synapse-eval:pr={result.pr_number}",
            f"pm:synapse-eval:sha={result.head_sha[:8]}",
            f"pm:synapse-eval:status={result.status}",
        ],
    )
    set_outstanding_brief_verified(target_id, b.comment_id)
    _post_write_sweep_brief(target_id, b.comment_id)
    return (
        f"action:eval_gate_brief:pr={result.pr_number}"
        f":status={result.status}:cid={b.comment_id}"
    )


def _act_regenerate_synapse_baseline(target_id: str) -> str:
    """Regenerate the synapse eval baseline at origin/main HEAD.

    Blocking operation with REGEN_TIMEOUT_S ceiling (240s).  Tears down
    worktree + ephemeral process unconditionally.  On success, writes
    /data/synapse/eval/baselines/main.json and logs a journal observation.
    On failure, leaves the prior baseline in place.

    Rate-limited: at most one retry per 30 minutes after a failure (tracked
    via pm/synapse-eval/regen-last-attempt in mem.db).
    """
    if not _eval_gate:
        return "noop:eval_gate_unavailable"

    episodic.write_observation(
        target_id,
        "Baseline regeneration started (synapse-eval-gate-v1)",
        extra_tags=["pm:synapse-eval:regen-started"],
    )
    result = _eval_gate.regenerate_baseline()
    if result is not None:
        _eval_gate.record_regen_attempt(success=True)
        sha = result.get("sha", "?")[:12]
        episodic.write_observation(
            target_id,
            f"Baseline regeneration succeeded: sha={sha} "
            f"metrics={list(result.get('metrics', {}).keys())}",
            extra_tags=["pm:synapse-eval:regen-succeeded", f"pm:synapse-eval:sha={sha}"],
        )
        return f"action:regenerate_synapse_baseline:succeeded:sha={sha}"
    else:
        _eval_gate.record_regen_attempt(success=False)
        episodic.write_observation(
            target_id,
            "Baseline regeneration failed - prior baseline preserved; "
            "will retry after rate-limit window (30 min)",
            extra_tags=["pm:synapse-eval:regen-failed", "pm:error"],
        )
        return "action:regenerate_synapse_baseline:failed"


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    target_id: str
    skipped: bool
    reason: str
    encoded: int          # number of percepts encoded as PM comments
    decision: str         # the action taken or noop:<reason> — see README.md § "Tick Decision Taxonomy"
    reconciled: int = 0   # number of dispatch records flipped this tick


def tick(target_id: str, allow_auto_land: bool = True) -> TickResult:
    store = TargetStore()
    target = store.get(target_id)
    if target is None:
        return TickResult(target_id, True, "target not found", 0, "noop:no_change")
    if not target.pm_bound:
        return TickResult(target_id, True, "target not pm_bound", 0, "noop:no_change")

    # 1. Pause guard with transition detection
    prev_state = get_pause_state(target_id) or "active"
    cur_state = "paused" if target.paused else "active"
    if cur_state != prev_state:
        episodic.write_observation(
            target_id,
            f"Tick state transition: {prev_state} → {cur_state}"
            + (f" (reason: {target.paused_reason})" if target.paused and target.paused_reason else ""),
        )
        set_pause_state(target_id, cur_state)
    if cur_state == "paused":
        return TickResult(target_id, True, "paused", 0, "noop:paused")

    # 2. Cursor + perceive
    # Reconcile pending dispatch records against ClaudeQueue terminal state
    # before any encode pass that reads dispatched (prevents stale-pending wedge).
    reconciled = _reconcile_dispatched_with_queue(target_id)

    cursor = get_cursor(target_id)
    new_comments = [
        c for c in episodic.since(target_id, cursor)
        if c.author != episodic.PM_AUTHOR
    ]

    repo = target.pm_repo or ""
    open_prs, forgejo_ok = _perceive_prs(target_id, repo) if repo else ([], False)

    # 3. Encode
    encoded = 0
    directives = _encode_user_comments(target_id, new_comments)
    encoded += len(directives)  # user-authored already in JSONL; count them

    seen_pr_ids = _seen_pr_ids(target_id)
    new_prs = _encode_new_prs(target_id, repo, open_prs, seen_pr_ids)
    encoded += len(new_prs)

    # Track PR head SHA advances (must precede _encode_gpu_results so the SHA
    # observation is visible when fixer_retry completion is checked below).
    encoded += _encode_pr_sha_updates(target_id, open_prs)

    gpu_encoded, failed_dispatches = _encode_gpu_results(target_id)
    encoded += gpu_encoded

    # Check Forgejo for newly merged PRs and write pm:pr-merged observations.
    # This is bookkeeping (encoding), not action — safe to do before decide.
    # Capture the count so the decide phase knows whether a merge just happened
    # (used by the baseline-regeneration trigger, which requires condition (a)
    # "a Synapse PR was just encoded as merged" per spec §Deliverables 2).
    _merged_this_tick = _encode_merged_prs(target_id, repo)
    encoded += _merged_this_tick

    # Classify lost fixer dispatches (terminal job, no PR produced).
    # Must run after encode so freshly-flipped records are visible.
    _lost_all_records = load_dispatched(target_id)
    _lost_needs_retry, _lost_needs_brief = _find_lost_fixer_dispatches(
        target_id, _lost_all_records, open_prs, forgejo_ok
    )

    # 4. Decide (priority order, single action)
    decision_str = "noop:no_change"
    pm_authority = target.pm_authority

    # 4.0 Brief-decision directive consumer — highest-priority decide branch.
    # If a directive file exists for this target, apply it and skip the rest.
    brief_decision_str = _consume_brief_decisions(target_id)
    if brief_decision_str is not None:
        set_cursor(target_id, _now_iso())
        return TickResult(target_id, False, "ok", encoded, brief_decision_str,
                          reconciled=reconciled)

    if directives:
        # v1: surface directives as a brief if any are recent and we don't
        # already have an outstanding brief; never auto-execute imperatives.
        d = directives[-1]
        episodic.write_observation(
            target_id,
            f"Directive received from {d.author} at {d.ts}:\n{d.content[:400]}",
            extra_tags=["pm:directive-seen", f"pm:directive-id={d.id}"],
        )
        encoded += 1
        b = brief.synthesize(target_id, trigger=f"user directive: {d.content[:80]}",
                             query=d.content, notify=None)
        set_outstanding_brief(target_id, b.comment_id)
        decision_str = f"action:directive_brief:cid={b.comment_id}"

    elif open_prs:
        # Skip PRs already classified this binding — prevents re-screening a
        # held PR every 10 min. Classification is reset when the PR is closed
        # or the target is rebound with a fresh spec.
        classified_ids = _classified_pr_ids(target_id)
        actionable_prs = [p for p in open_prs if p.get("number") not in classified_ids]
        if not actionable_prs:
            decision_str = "noop:no_change"
        else:
            # Pick the lowest-numbered PR (FIFO) so the same one drives action
            # until resolved.
            pr = min(actionable_prs, key=lambda p: p.get("number", 1 << 30))
            pr_number_sel = pr.get("number", 0)

            # Eval-gate: for synapse PRs, run quality check before reviewer dispatch.
            # Skip if reviewer is already pending (avoid race with in-flight review).
            _eval_gate_handled = False
            if _eval_gate and repo == "synapse":
                if not _has_pending_reviewer_for_pr(target_id, pr_number_sel):
                    try:
                        _eg_result = _eval_gate.evaluate_pr(target_id, pr, repo)
                    except Exception as _eg_exc:
                        logger.warning("eval_gate.evaluate_pr raised: %s", _eg_exc)
                        _eg_result = None
                    if _eg_result is not None:
                        _already_actioned = _eval_gate.is_eval_actioned(_eg_result.head_sha)
                        if not _already_actioned:
                            if _eg_result.status == "clean":
                                # Percept tag only; fall through to normal dispatch
                                episodic.write_observation(
                                    target_id,
                                    f"Eval-gate: PR #{pr_number_sel} "
                                    f"sha={_eg_result.head_sha[:8]} status=clean",
                                    extra_tags=[
                                        f"pm:synapse-eval:pr={pr_number_sel}",
                                        f"pm:synapse-eval:sha={_eg_result.head_sha[:8]}",
                                        "pm:synapse-eval:status=clean",
                                    ],
                                )
                                _eval_gate.mark_eval_actioned(_eg_result.head_sha)
                                # Fall through to _decide_for_pr below
                            elif _eg_result.status in ("regressed", "unverified"):
                                # Advisory brief — consumes single-action slot
                                decision_str = _act_eval_gate_brief(
                                    target_id, _eg_result, repo
                                )
                                _eval_gate_handled = True

            if not _eval_gate_handled:
                decision = _decide_for_pr(target_id, repo, pr, pm_authority)
                if decision.kind == "merge":
                    decision_str = _act_merge(target_id, decision.payload)
                elif decision.kind == "hold_brief":
                    decision_str = _act_brief(target_id, trigger="held PR", hold=True,
                                              payload=decision.payload)
                elif decision.kind == "advisory_brief":
                    decision_str = _act_brief(target_id, trigger="advisory PR", hold=False,
                                              payload=decision.payload)
                elif decision.kind == "dispatch_reviewer":
                    p = decision.payload
                    decision_str = _act_dispatch_reviewer(
                        target_id, p["pr"], p["cls"], p["mode"], p["cycle"],
                    )
                elif decision.kind == "dispatch_fixer_retry":
                    decision_str = _act_dispatch_fixer_retry(target_id, decision.payload)
                elif decision.kind == "review_exhausted_brief":
                    decision_str = _act_brief_review_exhausted(target_id, decision.payload)
                elif decision.kind == "review_gate_pause":
                    decision_str = _act_review_gate_pause(target_id, decision.payload)
                elif decision.kind == "noop_reviewer_in_flight":
                    p = decision.payload
                    decision_str = f"noop:reviewer_in_flight:pr={p['pr_number']}:cycle={p['cycle']}"
                elif decision.kind == "noop_fixer_in_flight":
                    p = decision.payload
                    decision_str = f"noop:fixer_in_flight:dispatch={p['dispatch_id']}"
                else:
                    # noop_no_change or unknown — single-action discipline: do nothing
                    decision_str = "noop:no_change"

    elif failed_dispatches:
        rec = failed_dispatches[-1]
        if rec.get("retry_count", 0) < MAX_DISPATCH_RETRIES:
            decision_str = _act_retry(target_id, rec)
        else:
            b = brief.synthesize(
                target_id,
                trigger=f"shaped agent {rec.get('agent_type')} failed after {rec.get('retry_count')} retries",
                query=rec.get("intent", ""),
                notify=NotifyPriority.HIGH,
            )
            set_outstanding_brief(target_id, b.comment_id)
            decision_str = f"action:abandon_brief:cid={b.comment_id}"

    elif _already_satisfied_pending(target_id) is not None:
        decision_str = _act_auto_land_already_satisfied(target_id)

    elif _already_satisfied_invalid_pending(target_id) is not None:
        decision_str = _act_brief_already_satisfied_invalid(target_id)

    elif _lost_needs_retry:
        # Lost dispatch: terminal fixer job, no PR produced, first loss → retry once.
        decision_str = _act_lost_fixer_retry(target_id, _lost_needs_retry[0])

    elif _lost_needs_brief:
        # Lost dispatch: retry also produced no PR → surface to human.
        _orig, _retry_rec = _lost_needs_brief[0]
        decision_str = _act_lost_brief(target_id, _orig, _retry_rec)

    elif (
        _eval_gate
        and _merged_this_tick > 0
        and _eval_gate.should_regenerate_baseline(repo)
    ):
        # Baseline regeneration: fires post-merge when baseline is stale/missing.
        # Spec §Deliverables 2: requires BOTH (a) a Synapse PR just encoded as
        # merged this tick AND (b) the baseline is stale or missing.
        # Mirrors auto-land shape — consumes the single-action slot.
        decision_str = _act_regenerate_synapse_baseline(target_id)

    elif allow_auto_land and _is_auto_land_eligible(target_id):
        decision_str = _act_auto_land(target_id)

    # 4.5 Refine noop decision for chain-mode legs waiting on dependencies.
    if decision_str == "noop:no_change":
        _deps = target.data.get("depends_on") or []
        if _deps:
            from . import chain as _chain_mod
            _landed = _chain_mod.landed_tids()
            _unsatisfied = [d for d in _deps if d not in _landed]
            if _unsatisfied:
                decision_str = f"noop:awaiting_chain_dependency:waiting_on={_unsatisfied[0]}"

    # 4.6. Persist review-state cache (best-effort visibility for claude-view).
    try:
        _persist_review_state_cache(target_id, target, open_prs)
    except Exception as e:
        episodic.write_observation(
            target_id, f"review-state cache write failed: {e}",
            extra_tags=["pm:error"],
        )

    # 5. Advance cursor to now (we've considered everything as of this tick).
    set_cursor(target_id, _now_iso())
    return TickResult(target_id, False, "ok", encoded, decision_str, reconciled=reconciled)


def tick_all() -> list[TickResult]:
    """Tick every pm_bound target. Returns one TickResult per target.

    At most one auto-land fires per tick_all call. When multiple targets are
    land-eligible, the oldest-bound (earliest spec:bound comment) is chosen;
    others have auto-land suppressed and become eligible next tick.
    """
    # Pre-perceive health probe — one probe per tick_all call, not per target.
    reachable, probe_reason = probe_forgejo_health()
    if not reachable:
        print(f"[forgejo:unreachable] reason={probe_reason}", flush=True)
        fails = _get_forgejo_consecutive_fails() + 1
        _set_forgejo_consecutive_fails(fails)
        if fails == FORGEJO_UNREACHABLE_THRESHOLD:
            _notify_forgejo_unreachable()
        store = TargetStore()
        bound = [t for t in store.load_all() if t.pm_bound]
        # Cursors are NOT advanced — next healthy tick re-perceives from the same point.
        skipped = [
            TickResult(t.id, True, "forgejo_unreachable", 0, "skipped:forgejo_unreachable")
            for t in bound
        ]
        for r in skipped:
            print(
                f"[{r.target_id}] skipped={r.skipped} reason={r.reason}"
                f" encoded={r.encoded} decision={r.decision}",
                flush=True,
            )
        return skipped
    # Probe succeeded — reset consecutive-fail counter.
    _set_forgejo_consecutive_fails(0)

    store = TargetStore()
    bound = [t for t in store.load_all() if t.pm_bound]

    # Pre-select which target (if any) gets the auto-land slot this tick.
    # This check uses observations from previous ticks; targets that become
    # eligible for the first time this tick are deferred to the next tick.
    eligible_ids = sorted(
        [t.id for t in bound if _is_auto_land_eligible(t.id)],
        key=_spec_bound_ts,
    )
    auto_land_chosen = eligible_ids[0] if eligible_ids else None

    results: list[TickResult] = []
    for t in bound:
        try:
            results.append(tick(t.id, allow_auto_land=(t.id == auto_land_chosen)))
        except Exception as e:
            # Advance cursor even on exception so the same percepts don't get
            # re-encoded as duplicate observations on the next tick.
            try:
                set_cursor(t.id, _now_iso())
            except Exception:
                pass
            results.append(TickResult(t.id, True, f"exception: {e}", 0, "noop:no_change"))
    return results
