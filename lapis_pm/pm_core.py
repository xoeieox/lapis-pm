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
MAX_DISPATCH_RETRIES = 2


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
# Perceive helpers
# ---------------------------------------------------------------------------

def _gpu_output_path(task_id: str) -> Path | None:
    p = COMPLETED_DIR / f"{task_id}-output.md"
    if p.exists():
        return p
    p = FAILED_DIR / f"{task_id}-output.md"
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
    if cls.verdict == "auto":
        return Decision("merge", payload)
    if cls.verdict == "hold":
        return Decision("hold_brief", payload)
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
        out_path = _gpu_output_path(rec["gpu_id"])
        if not out_path:
            continue
        try:
            text = out_path.read_text(encoding="utf-8")
        except OSError:
            continue

        is_failure = (
            out_path.parent == FAILED_DIR
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
            else:
                decision_str = _act_brief(target_id, trigger="advisory PR", hold=False,
                                          payload=decision.payload)

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
