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
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agents_core.targets import TargetStore
from agents_core.mem import MemoryStore

try:
    from agents_core.forgejo import get_open_prs, merge_pr
except Exception:
    get_open_prs = None  # type: ignore
    merge_pr = None  # type: ignore

from . import episodic, shaper, brief, authority


PACIFIC = ZoneInfo("America/Los_Angeles")
COMPLETED_DIR = Path("/srv/lapis/gpu-queue/completed")
FAILED_DIR = Path("/srv/lapis/gpu-queue/failed")
SHAPED_DIR = Path("/srv/lapis/gpu-queue/shaped")  # _runner.py meta sidecars
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


def get_pause_state(target_id: str) -> str | None:
    rec = _mem().get(_pause_key(target_id))
    return rec["content"] if rec else None


def set_pause_state(target_id: str, state: str):
    _mem().set(_pause_key(target_id), state, tags=["lapis-pm", "pause-state"])


def set_outstanding_brief(target_id: str, comment_id: str):
    _mem().set(_brief_key(target_id), comment_id, tags=["lapis-pm", "outstanding-brief"])


def clear_outstanding_brief(target_id: str):
    _mem().delete(_brief_key(target_id))


def get_outstanding_brief(target_id: str) -> str | None:
    rec = _mem().get(_brief_key(target_id))
    return rec["content"] if rec else None


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
    }

    if get_outstanding_brief(target_id) is not None:
        summary["outstanding_brief"] = 1
    clear_outstanding_brief(target_id)

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

    return summary


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


def _fixer_retry_count(target_id: str, pr_number: int) -> int:
    """Count completed fixer_retry dispatches for this PR via dispatch records."""
    return sum(
        1 for r in load_dispatched(target_id)
        if r.get("agent_type") == "fixer_retry"
        and r.get("pr_number") == pr_number
        and r.get("status") in ("processed", "failed")
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


def _perceive_prs(target_id: str, repo: str) -> list[dict]:
    if not get_open_prs:
        return []
    try:
        prs = get_open_prs(repo)
    except Exception as e:
        episodic.write_observation(
            target_id, f"PR fetch failed for {repo}: {e}",
            extra_tags=["pm:error"],
        )
        return []
    out = []
    for pr in prs:
        head = (pr.get("head") or {}).get("ref") or ""
        if _branch_belongs(target_id, head):
            out.append(pr)
    return out


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    kind: str   # "noop" | "merge" | "advisory_brief" | "hold_brief" | "retry" | "abandon_brief" | "directive_ack"
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
        return Decision("noop", {"reason": f"reviewer pending for PR #{pr_number}"})
    if _has_pending_fixer_for_pr(target_id, pr_number):
        return Decision("noop", {"reason": f"fixer_retry pending for PR #{pr_number}"})

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
                return Decision("noop", {
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
        return Decision("noop", {"reason": "reviewer_count > fixer_count but no verdict found"})

    verdict = verdict_info.get("verdict", "needs-human")
    issues = verdict_info.get("issues", [])

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
        return f"merge_failed:{e}"
    episodic.write_merge(
        target_id,
        f"Auto-merged PR #{cls.pr_number} ({cls.title}) — "
        f"{cls.diff_loc} LOC, screen={cls.screen_verdict}\n{cls.html_url}",
        extra_tags=[f"pm:repo={cls.repo}", f"pm:pr={cls.pr_number}"],
    )
    _mark_pr_classified(target_id, cls.pr_number)
    return f"merged:{cls.pr_number}"


def _act_brief(target_id: str, trigger: str, hold: bool, payload: dict) -> str:
    cls: authority.PRClassification = payload["classification"]
    if hold:
        episodic.write_hold(
            target_id,
            f"PR #{cls.pr_number} held: {'; '.join(cls.reasons)}\n"
            f"Title: {cls.title}\n{cls.html_url}",
            extra_tags=[f"pm:repo={cls.repo}", f"pm:pr={cls.pr_number}"],
        )
    b = brief.synthesize(
        target_id,
        trigger=trigger,
        query=cls.title,
        diff_snippet=cls.diff or None,
        screen_issues=cls.issues or None,
    )
    set_outstanding_brief(target_id, b.comment_id)
    _mark_pr_classified(target_id, cls.pr_number)
    return f"brief:{b.comment_id} pushed={b.pushed}"


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
    res = shaper.dispatch(agent_type, target_id, user_prompt, vars_=vars_)
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
    return f"retry:{res.task_id}"


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
        prior = _last_review_verdict(target_id, pr_number)
        if prior:
            issues_text = json.dumps(prior.get("issues", []), indent=2)
            prior_review_text = (
                f"\n## Prior review context (cycle {cycle - 1})\n\n"
                f"Verdict: {prior.get('verdict')}\n"
                f"Issues:\n{issues_text}\n\n"
                f"Review the updated diff in light of these prior concerns.\n"
            )

    vars_: dict = {
        "target_id": target_id,
        "spec_summary": spec_summary,
        "repo": repo,
        "repo_cwd": shaper._resolve_repo_cwd(repo),
        "pr_number": pr_number,
        "slug": f"pr{pr_number}-review-c{cycle}",
        "question": f"review PR #{pr_number}",
        "prior_review": prior_review_text,
    }

    # Get the diff for the reviewer prompt
    try:
        from agents_core.forgejo import get_pr_diff as _get_diff
        diff_text = _get_diff(repo, pr_number)
        if len(diff_text) > 60000:
            diff_text = diff_text[:60000] + "\n\n... (diff truncated)"
    except Exception:
        diff_text = "(diff unavailable)"

    user_prompt = (
        f"Review PR #{pr_number} in {repo}. This is reviewer cycle {cycle}.\n\n"
        f"```diff\n{diff_text}\n```\n\n"
        f"Return JSON: {{\"verdict\": \"clean\" | \"fixable\" | \"needs-human\", "
        f"\"issues\": [...], \"confidence\": 0.0-1.0}}"
    )

    # Increment kill-switch counter before dispatch
    _increment_review_gate_counter()

    res = shaper.dispatch(agent_type, target_id, user_prompt, vars_=vars_)

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
    return f"reviewer_dispatched:pr={pr_number}:cycle={cycle}"


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
        "repo_cwd": shaper._resolve_repo_cwd(cls.repo),
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

    res = shaper.dispatch("fixer_retry", target_id, user_prompt, vars_=vars_)

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
    return f"fixer_retry_dispatched:pr={pr_number}:cycle={cycle}"


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
    )
    set_outstanding_brief(target_id, b.comment_id)
    _mark_pr_classified(target_id, cls.pr_number)
    return f"review_exhausted_brief:{b.comment_id}"


def _act_review_gate_pause(target_id: str, payload: dict) -> str:
    """Emit a single pause brief when the kill-switch threshold is exceeded."""
    # Only post the pause brief once (idempotent)
    rec = _mem().get(REVIEW_GATE_PAUSE_BRIEF_KEY)
    if rec:
        return "review_gate_pause:already_briefed"

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
    )
    set_outstanding_brief(target_id, b.comment_id)
    _mem().set(REVIEW_GATE_PAUSE_BRIEF_KEY, b.comment_id,
               tags=["lapis-pm", "review-gate"])
    return f"review_gate_paused:{b.comment_id}"


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
        clear_outstanding_brief(target_id)
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
            written += 1
    return written


def _encode_gpu_results(target_id: str) -> tuple[int, list[dict]]:
    """For each pending dispatch, check for completion and encode result.

    Returns (total_encoded, failed_for_retry).
    """
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

        # Confabulation check: fixer agents that produced substantial prose
        # without using tools are stochastic failures and should be retried.
        # Meta sidecar is only written for fixers (capture_meta in shaper.py).
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
                try:
                    verdict_data = json.loads(raw_output)
                    verdict_val = verdict_data.get("verdict", "needs-human")
                    stored_json = json.dumps(verdict_data)
                except json.JSONDecodeError:
                    verdict_val = "needs-human"
                    stored_json = json.dumps({"verdict": "needs-human", "issues": [],
                                              "confidence": 0.0,
                                              "_parse_error": raw_output[:200]})
                episodic.write_result(
                    target_id,
                    f"Reviewer verdict for PR #{pr_num}:\n{stored_json}",
                    extra_tags=tags + [
                        f"pm:reviewer:pr={pr_num}:cycle={cycle_num}:verdict={verdict_val}",
                        f"pm:pr={pr_num}",
                    ],
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
# Tick
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    target_id: str
    skipped: bool
    reason: str
    encoded: int          # number of percepts encoded as PM comments
    decision: str         # the action taken or "noop"


def tick(target_id: str) -> TickResult:
    store = TargetStore()
    target = store.get(target_id)
    if target is None:
        return TickResult(target_id, True, "target not found", 0, "noop")
    if not target.pm_bound:
        return TickResult(target_id, True, "target not pm_bound", 0, "noop")

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
        return TickResult(target_id, True, "paused", 0, "noop")

    # 2. Cursor + perceive
    cursor = get_cursor(target_id)
    new_comments = [
        c for c in episodic.since(target_id, cursor)
        if c.author != episodic.PM_AUTHOR
    ]

    repo = target.pm_repo or ""
    open_prs = _perceive_prs(target_id, repo) if repo else []

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

    # 4. Decide (priority order, single action)
    decision_str = "noop"
    pm_authority = target.pm_authority

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
                             query=d.content)
        set_outstanding_brief(target_id, b.comment_id)
        decision_str = f"directive_brief:{b.comment_id}"

    elif open_prs:
        # Skip PRs already classified this binding — prevents re-screening a
        # held PR every 10 min. Classification is reset when the PR is closed
        # or the target is rebound with a fresh spec.
        classified_ids = _classified_pr_ids(target_id)
        actionable_prs = [p for p in open_prs if p.get("number") not in classified_ids]
        if not actionable_prs:
            decision_str = "noop"
        else:
            # Pick the lowest-numbered PR (FIFO) so the same one drives action
            # until resolved.
            pr = min(actionable_prs, key=lambda p: p.get("number", 1 << 30))
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
            else:
                # noop or unknown — single-action discipline: do nothing
                decision_str = "noop"

    elif failed_dispatches:
        rec = failed_dispatches[-1]
        if rec.get("retry_count", 0) < MAX_DISPATCH_RETRIES:
            decision_str = _act_retry(target_id, rec)
        else:
            b = brief.synthesize(
                target_id,
                trigger=f"shaped agent {rec.get('agent_type')} failed after {rec.get('retry_count')} retries",
                query=rec.get("intent", ""),
            )
            set_outstanding_brief(target_id, b.comment_id)
            decision_str = f"abandon_brief:{b.comment_id}"

    # 5. Advance cursor to now (we've considered everything as of this tick).
    set_cursor(target_id, _now_iso())
    return TickResult(target_id, False, "ok", encoded, decision_str)


def tick_all() -> list[TickResult]:
    """Tick every pm_bound target. Returns one TickResult per target."""
    store = TargetStore()
    results: list[TickResult] = []
    for t in store.load_all():
        if not t.pm_bound:
            continue
        try:
            results.append(tick(t.id))
        except Exception as e:
            # Advance cursor even on exception so the same percepts don't get
            # re-encoded as duplicate observations on the next tick.
            try:
                set_cursor(t.id, _now_iso())
            except Exception:
                pass
            results.append(TickResult(t.id, True, f"exception: {e}", 0, "noop"))
    return results
