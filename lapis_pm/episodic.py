"""Episodic memory layer for Lapis PM.

Comments == episodes. The dashboard's CommentStore (JSONL per thread) is the
substrate. This module adds:

- spec(target_id): retrieve the spec body from the one-shot `spec:bound` comment
- recall(target_id, query, k): top-k PM comments scored by 5-factor formula
- classify(event): map a perceived event to encoding-trigger categories
- write helpers (observation/dispatch/result/brief/hold/merge/retry) that just
  delegate to CommentStore with the right tags

We deliberately do NOT reconsolidate (drift) on read in v1 — comments are
authoritative. Reconsolidation is deferred to v2.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

from agents_core.comments import Comment, CommentStore

PM_AUTHOR = "lapis-pm"

# Tag taxonomy — keep in sync with the plan.
TAG_OBSERVATION = "pm:observation"
TAG_DISPATCH = "pm:dispatch"
TAG_RESULT = "pm:result"
TAG_BRIEF = "pm:brief"
TAG_BRIEF_OPTIONS = "pm:brief-options"
TAG_HOLD = "pm:hold"
TAG_MERGE = "pm:merge"
TAG_RETRY = "pm:retry"
TAG_SPEC = "spec:bound"
TAG_HUMAN_DIRECTIVE = "human:directive"
TAG_HUMAN_ACK = "human:ack"

PM_TAGS = {
    TAG_OBSERVATION, TAG_DISPATCH, TAG_RESULT,
    TAG_BRIEF, TAG_HOLD, TAG_MERGE, TAG_RETRY,
}


@dataclass
class ScoredComment:
    comment: Comment
    score: float


def _store() -> CommentStore:
    return CommentStore()


# --- Spec accessor ---------------------------------------------------------

def spec(target_id: str) -> str | None:
    """Return the spec body from the most recent `spec:bound` comment, or None."""
    result = None
    for c in _store().list(target_id):
        if TAG_SPEC in c.tags:
            result = c.content
    return result


def spec_summary(target_id: str, max_chars: int | None = None) -> str:
    """Return the spec, optionally capped for prompt injection. None = no cap."""
    body = spec(target_id) or "(no spec bound)"
    if max_chars is None or len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip() + "\n…[truncated]"


# --- Write helpers ---------------------------------------------------------

def write(target_id: str, content: str, tags: list[str], author_type: str = "agent") -> Comment:
    """Append a PM-authored comment with tags."""
    return _store().append(
        target_id, content,
        author=PM_AUTHOR, author_type=author_type, tags=tags,
    )


def write_observation(target_id: str, content: str, extra_tags: list[str] | None = None) -> Comment:
    tags = [TAG_OBSERVATION] + list(extra_tags or [])
    return write(target_id, content, tags)


def write_dispatch(target_id: str, content: str, extra_tags: list[str] | None = None) -> Comment:
    tags = [TAG_DISPATCH] + list(extra_tags or [])
    return write(target_id, content, tags)


def write_result(target_id: str, content: str, extra_tags: list[str] | None = None) -> Comment:
    tags = [TAG_RESULT] + list(extra_tags or [])
    return write(target_id, content, tags)


def write_brief(target_id: str, content: str, extra_tags: list[str] | None = None) -> Comment:
    tags = [TAG_BRIEF] + list(extra_tags or [])
    return write(target_id, content, tags)


def write_brief_options(target_id: str, content: str, extra_tags: list[str] | None = None) -> Comment:
    """Write a pm:brief-options sibling comment (JSON document)."""
    tags = [TAG_BRIEF_OPTIONS] + list(extra_tags or [])
    return write(target_id, content, tags)


def write_hold(target_id: str, content: str, extra_tags: list[str] | None = None) -> Comment:
    tags = [TAG_HOLD] + list(extra_tags or [])
    return write(target_id, content, tags)


def write_merge(target_id: str, content: str, extra_tags: list[str] | None = None) -> Comment:
    tags = [TAG_MERGE] + list(extra_tags or [])
    return write(target_id, content, tags)


def write_retry(target_id: str, content: str, extra_tags: list[str] | None = None) -> Comment:
    tags = [TAG_RETRY] + list(extra_tags or [])
    return write(target_id, content, tags)


def write_spec(target_id: str, body: str) -> Comment:
    """Append a spec:bound comment. Callers may call multiple times (amendment flow);
    episodic.spec() returns the last entry, so the most recent write wins."""
    return _store().append(
        target_id, body,
        author="lapis-pm-bind", author_type="system", tags=[TAG_SPEC],
    )


# --- Read helpers ----------------------------------------------------------

def all_comments(target_id: str) -> list[Comment]:
    return _store().list(target_id)


def since(target_id: str, cursor_ts: str | None) -> list[Comment]:
    """Comments with ts > cursor_ts (string compare on ISO works)."""
    if cursor_ts is None:
        return all_comments(target_id)
    return [c for c in all_comments(target_id) if c.ts > cursor_ts]


def latest_ts(target_id: str) -> str | None:
    cs = all_comments(target_id)
    return cs[-1].ts if cs else None


def has_outstanding_directive(target_id: str, since_ts: str | None) -> Comment | None:
    """Return latest unprocessed human:directive comment (after since_ts)."""
    out = None
    for c in since(target_id, since_ts):
        if TAG_HUMAN_DIRECTIVE in c.tags and c.author_type == "user":
            out = c
    return out


def has_recent_ack(target_id: str, since_ts: str | None) -> bool:
    for c in since(target_id, since_ts):
        if TAG_HUMAN_ACK in c.tags and c.author_type == "user":
            return True
    return False


# --- Scoring (5-factor) ----------------------------------------------------
# Modeled on archetypes/engine/episodic_memory.py. Without an embedding
# server we use lexical Jaccard for "semantic similarity"; that is good
# enough for v1 since comments are short and topical.

WEIGHTS = {
    "relevance": 0.30,      # query overlap
    "goal": 0.20,           # spec overlap
    "blocker": 0.25,        # tag-derived weight (hold/retry score higher)
    "affinity": 0.15,       # PM-tag presence (signal vs noise)
    "recency": 0.10,        # exponential decay
}


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _recency(ts: str, now: datetime | None = None) -> float:
    """Exponential decay over days. Half-life ~7 days."""
    if not ts:
        return 0.0
    try:
        when = datetime.fromisoformat(ts)
    except ValueError:
        return 0.0
    now = now or datetime.now(when.tzinfo) if when.tzinfo else datetime.now()
    age_days = max(0.0, (now - when).total_seconds() / 86400)
    return math.exp(-age_days * (math.log(2) / 7.0))


def _blocker_weight(tags: Iterable[str]) -> float:
    tset = set(tags)
    if TAG_HOLD in tset:
        return 1.0
    if TAG_RETRY in tset:
        return 0.7
    if TAG_BRIEF in tset:
        return 0.6
    if TAG_DISPATCH in tset or TAG_RESULT in tset:
        return 0.4
    return 0.2


def _affinity(tags: Iterable[str]) -> float:
    tset = set(tags)
    if tset & PM_TAGS:
        return 1.0
    return 0.3


def score_comment(c: Comment, query_tokens: set[str], spec_tokens: set[str],
                  now: datetime | None = None) -> float:
    ctoks = _tokens(c.content)
    rel = _jaccard(query_tokens, ctoks)
    goal = _jaccard(spec_tokens, ctoks)
    blocker = _blocker_weight(c.tags)
    affinity = _affinity(c.tags)
    recency = _recency(c.ts, now=now)
    return (
        WEIGHTS["relevance"] * rel
        + WEIGHTS["goal"] * goal
        + WEIGHTS["blocker"] * blocker
        + WEIGHTS["affinity"] * affinity
        + WEIGHTS["recency"] * recency
    )


def recall(target_id: str, query: str, k: int = 5) -> list[ScoredComment]:
    """Top-k PM comments scored by the 5-factor formula."""
    comments = [c for c in all_comments(target_id) if set(c.tags) & PM_TAGS]
    if not comments:
        return []
    qtok = _tokens(query)
    stok = _tokens(spec(target_id) or "")
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    scored = [ScoredComment(c, score_comment(c, qtok, stok, now=now)) for c in comments]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:k]


# --- Encoding-trigger classification --------------------------------------

@dataclass
class Trigger:
    kind: str   # "wrongness" | "shadow" | "relational" | "directive" | "neutral"
    note: str


def classify_gpu_completion(output_text: str, exit_marker_failed: bool) -> Trigger:
    """Classify a completed GPU task output."""
    if exit_marker_failed or output_text.startswith("ERROR") or output_text.startswith("EXIT "):
        return Trigger("wrongness", "subprocess failed or non-zero exit")
    if output_text.startswith("TIMEOUT"):
        return Trigger("wrongness", "subprocess timed out")
    return Trigger("neutral", "completed")


def classify_pr_event(pr: dict) -> Trigger:
    """Classify a Forgejo PR snapshot dict."""
    state = (pr.get("state") or "").lower()
    if state == "closed" and not pr.get("merged"):
        return Trigger("wrongness", "PR closed without merge")
    mergeable = pr.get("mergeable")
    if mergeable is False:
        return Trigger("wrongness", "PR has merge conflicts")
    return Trigger("neutral", f"PR {pr.get('number')} state={state}")


def classify_user_comment(c: Comment) -> Trigger:
    if TAG_HUMAN_DIRECTIVE in c.tags:
        return Trigger("directive", "user directive")
    if TAG_HUMAN_ACK in c.tags:
        return Trigger("neutral", "user ack")
    return Trigger("neutral", "user comment")
