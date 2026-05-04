"""Router Portfolio Persistence — router-portfolio-persistence-v0.

Defines RouterPortfolioEntry (CorroborationResult-shaped with Router extras)
and emit_* helpers for the eight event catalog entries.

Storage backend: mem.db via agents_core.mem.MemoryStore.
Namespace: router/lapis-pm/{decisions,ratification-outcomes,expert-experience,sessions}

Event catalog (eight events):
  router.decision.kickoff   → decisions/        verdict=proposed
  router.decision.dispatch  → decisions/        verdict=proposed
  router.decision.land      → decisions/        verdict=proposed
  router.ratify.confirm     → ratification-outcomes/  verdict=ratified
  router.ratify.correct     → ratification-outcomes/  verdict=corrected
  router.ratify.override    → ratification-outcomes/  verdict=overridden
  router.ratify.redirect    → ratification-outcomes/  verdict=redirected
  router.expert.outcome     → expert-experience/<tier>/<shape>/  verdict=landed|reverted|flagged|orphaned

Schema mirrors archetypes_core.corroboration.CorroborationResult; when that
import is stable, M0.6 reorganizes to the canonical type. This module defines
it inline so v0 has zero cross-module dependency.

Writes are idempotent: re-firing the same event_id with the same payload is
a no-op (the mem key is stable across re-fires).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

from archetypes_core.corroboration import Citation
from archetypes_core.provenance import InputRef, LapisToolReturn, UpstreamRef, to_lapis_return

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCOPE_ID = "router/lapis-pm"

VerdictT = Literal[
    "proposed",
    "ratified",
    "corrected",
    "overridden",
    "redirected",
    "landed",
    "reverted",
    "flagged",
    "orphaned",
]


@dataclass
class RouterPortfolioEntry:
    """CorroborationResult-aligned portfolio entry for the Lapis PM Router.

    Base fields mirror archetypes_core.corroboration.CorroborationResult.
    Router-specific extras extend the shape for PM-specific data.
    M0.6 reorganizes to the canonical import when it is stable.
    """

    # CorroborationResult base fields:
    verdict: VerdictT
    claim: str
    citations: list[str]          # mem keys, target_ids, PR refs, prior event_ids
    freshness_stamp: str          # ISO-8601 UTC
    scope_id: str = _SCOPE_ID    # always "router/lapis-pm" for this Router
    drift_class: str | None = None              # tonal | thematic | isomorphic | factual | role | none
    primitive_decomposition: list[dict] | None = None  # M0 fragment_id + intent primitives

    # Router-specific extras:
    fragment_id: str | None = None              # M0 vocabulary (kickoff / tick / review-cycle / ...)
    expert_chosen: str | None = None            # qwen-local | haiku | sonnet | opus
    ratification_outcome: str | None = None     # confirm | correct | override | redirect
    target_id: str | None = None                # which target this decision was about
    intent_summary: str | None = None           # short prose for human readability

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_envelope(
        self,
        *,
        agent_id: str = "lapis-pm/router",
        tool: str | None = None,
        input_refs: list[InputRef] | None = None,
        upstream_calls: list[UpstreamRef] | None = None,
        timestamp: str | None = None,
    ) -> LapisToolReturn:
        """Wrap this portfolio entry as a v0-conformant LapisToolReturn.

        The entry itself is the payload; provenance is composed from Router fields
        per the Lapis Provenance Schema v0 mapping. Backward-compatible: this method
        is opt-in; persistent storage and existing emit_* functions are unchanged.

        Args:
            agent_id: Routing identifier; defaults to "lapis-pm/router".
            tool: Tool name within agent; defaults to self.fragment_id (or
                "router_emit" if fragment_id is None).
            input_refs: Optional caller-supplied input refs.
            upstream_calls: Optional caller-supplied upstream call list.
            timestamp: Optional explicit ISO-8601 UTC timestamp; defaults to now.
                (Note: distinct from self.freshness_stamp, which describes the
                recorded event's substrate freshness, not envelope return time.)

        Returns:
            LapisToolReturn with payload=self, summary=self.intent_summary or
            a fallback, and provenance populated per the schema doc mapping.

        Note on prompt_hash:
            prompt_hash is always None for Router emissions. Router entries are
            transcribed records of expert decisions, not single LLM calls, so
            there is no canonical resolved prompt to hash. model set + prompt_hash
            None is a recognized pattern for transcribed/aggregated provenance
            records (Lapis Provenance Schema v0, §"Mapping from existing shapes").
        """
        # 1. Derive summary
        if self.intent_summary:
            summary = self.intent_summary
        else:
            summary = f"{self.verdict}: {self.fragment_id or 'router-event'}"

        # 2. Derive tool
        resolved_tool = tool or self.fragment_id or "router_emit"

        # 3. Wrap citations: list[str] → list[Citation]
        wrapped_citations = [
            Citation(
                source_id=s,
                excerpt="",
                content_hash=None,
                provenance_method="router_referenced",
            )
            for s in self.citations
        ]

        # 4 & 5. Compose provenance and build envelope
        return to_lapis_return(
            payload=self,
            agent_id=agent_id,
            tool=resolved_tool,
            summary=summary,
            model=self.expert_chosen,
            prompt_hash=None,  # transcribed-provenance pattern: no canonical resolved prompt
            citations=wrapped_citations,
            scope_id=self.scope_id,
            primitive_decomposition=None,  # Router stores list[dict] in payload; envelope stays None at v0
            input_refs=input_refs or [],
            upstream_calls=upstream_calls or [],
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Mem access
# ---------------------------------------------------------------------------

_mem_store = None


def _mem():
    """Return the shared module-level MemoryStore instance (lazy init)."""
    global _mem_store
    if _mem_store is None:
        from agents_core.mem import MemoryStore
        _mem_store = MemoryStore()
    return _mem_store


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _event_id(prefix: str = "") -> str:
    """Generate a sortable, unique event_id: <YYYY-MM-DDTHH:MM:SSZ>-<short-hash>."""
    ts = _now_iso()
    # Use a partial hash of ts + monotonic time for uniqueness within the same second
    raw = f"{ts}-{time.monotonic_ns()}-{prefix}"
    short = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{ts}-{short}"


def _write_entry(mem_key: str, entry: RouterPortfolioEntry, tags: list[str]) -> str:
    """Write entry to mem.db. Returns the mem_key written.

    Idempotent: if the key already exists with identical content, the write is
    a no-op (mem.set overwrites but content is stable for same event_id payload).

    Set ROUTER_PORTFOLIO_FAIL_WRITES=1 in the environment to force all writes to
    raise RuntimeError — used by smoke tests to exercise the try/except guard in callers.
    """
    if os.environ.get("ROUTER_PORTFOLIO_FAIL_WRITES"):
        raise RuntimeError("ROUTER_PORTFOLIO_FAIL_WRITES is set (test mode)")
    _mem().set(mem_key, entry.to_json(), tags=tags)
    return mem_key


# ---------------------------------------------------------------------------
# Event emitters — one per event catalog entry
# ---------------------------------------------------------------------------

def emit_decision_kickoff(
    target_id: str,
    expert_chosen: str,
    intent_summary: str,
    citations: list[str] | None = None,
    primitive_decomposition: list[dict] | None = None,
    event_id: str | None = None,
) -> str:
    """router.decision.kickoff — Router has chosen a kickoff fragment+expert for a target.

    verdict=proposed; namespace=router/lapis-pm/decisions/<event_id>
    Returns the mem key written.
    """
    eid = event_id or _event_id("kickoff")
    entry = RouterPortfolioEntry(
        verdict="proposed",
        claim=f"this kickoff fragment+expert composition is the right kickoff for `{target_id}`",
        citations=citations or [],
        freshness_stamp=_now_iso(),
        scope_id=_SCOPE_ID,
        fragment_id="kickoff",
        expert_chosen=expert_chosen,
        target_id=target_id,
        intent_summary=intent_summary,
        primitive_decomposition=primitive_decomposition,
    )
    key = f"router/lapis-pm/decisions/{eid}"
    return _write_entry(key, entry, tags=["lapis-pm", "router-portfolio", f"target:{target_id}"])


def emit_decision_dispatch(
    target_id: str,
    fragment_id: str,
    expert_chosen: str,
    intent_summary: str,
    citations: list[str] | None = None,
    primitive_decomposition: list[dict] | None = None,
    event_id: str | None = None,
) -> str:
    """router.decision.dispatch — Router has chosen a fragment+expert to advance a target.

    verdict=proposed; namespace=router/lapis-pm/decisions/<event_id>
    Returns the mem key written.
    """
    eid = event_id or _event_id("dispatch")
    entry = RouterPortfolioEntry(
        verdict="proposed",
        claim=f"this fragment+expert composition advances `{target_id}`",
        citations=citations or [],
        freshness_stamp=_now_iso(),
        scope_id=_SCOPE_ID,
        fragment_id=fragment_id,
        expert_chosen=expert_chosen,
        target_id=target_id,
        intent_summary=intent_summary,
        primitive_decomposition=primitive_decomposition,
    )
    key = f"router/lapis-pm/decisions/{eid}"
    return _write_entry(key, entry, tags=["lapis-pm", "router-portfolio", f"target:{target_id}"])


def emit_decision_land(
    target_id: str,
    intent_summary: str,
    citations: list[str] | None = None,
    primitive_decomposition: list[dict] | None = None,
    event_id: str | None = None,
) -> str:
    """router.decision.land — Router judges target is ready to land.

    verdict=proposed; namespace=router/lapis-pm/decisions/<event_id>
    Returns the mem key written.
    """
    eid = event_id or _event_id("land")
    entry = RouterPortfolioEntry(
        verdict="proposed",
        claim=f"`{target_id}` is ready to land",
        citations=citations or [],
        freshness_stamp=_now_iso(),
        scope_id=_SCOPE_ID,
        fragment_id="land",
        target_id=target_id,
        intent_summary=intent_summary,
        primitive_decomposition=primitive_decomposition,
    )
    key = f"router/lapis-pm/decisions/{eid}"
    return _write_entry(key, entry, tags=["lapis-pm", "router-portfolio", f"target:{target_id}"])


def emit_ratify_confirm(
    target_id: str,
    prior_decision_event_id: str,
    citations: list[str] | None = None,
    event_id: str | None = None,
) -> str:
    """router.ratify.confirm — Principal confirmed the Router's prior decision.

    verdict=ratified; namespace=router/lapis-pm/ratification-outcomes/<event_id>
    Returns the mem key written.
    """
    eid = event_id or _event_id("confirm")
    prior_key = f"router/lapis-pm/decisions/{prior_decision_event_id}"
    all_citations = [prior_key] + (citations or [])
    entry = RouterPortfolioEntry(
        verdict="ratified",
        claim=f"principal confirmed Router decision for `{target_id}`",
        citations=all_citations,
        freshness_stamp=_now_iso(),
        scope_id=_SCOPE_ID,
        ratification_outcome="confirm",
        target_id=target_id,
    )
    key = f"router/lapis-pm/ratification-outcomes/{eid}"
    return _write_entry(key, entry, tags=["lapis-pm", "router-portfolio", f"target:{target_id}", "ratify:confirm"])


def emit_ratify_correct(
    target_id: str,
    intent_summary: str,
    prior_decision_event_id: str | None = None,
    citations: list[str] | None = None,
    event_id: str | None = None,
) -> str:
    """router.ratify.correct — Principal corrected the Router's prior decision.

    verdict=corrected; namespace=router/lapis-pm/ratification-outcomes/<event_id>
    Returns the mem key written.
    """
    eid = event_id or _event_id("correct")
    all_citations = []
    if prior_decision_event_id:
        all_citations.append(f"router/lapis-pm/decisions/{prior_decision_event_id}")
    all_citations.extend(citations or [])
    entry = RouterPortfolioEntry(
        verdict="corrected",
        claim=f"principal corrected Router decision for `{target_id}`",
        citations=all_citations,
        freshness_stamp=_now_iso(),
        scope_id=_SCOPE_ID,
        ratification_outcome="correct",
        target_id=target_id,
        intent_summary=intent_summary,
    )
    key = f"router/lapis-pm/ratification-outcomes/{eid}"
    return _write_entry(key, entry, tags=["lapis-pm", "router-portfolio", f"target:{target_id}", "ratify:correct"])


def emit_ratify_override(
    target_id: str,
    intent_summary: str,
    prior_decision_event_id: str | None = None,
    citations: list[str] | None = None,
    event_id: str | None = None,
) -> str:
    """router.ratify.override — Principal overrode the Router's prior decision (high signal).

    verdict=overridden; namespace=router/lapis-pm/ratification-outcomes/<event_id>
    Returns the mem key written.
    """
    eid = event_id or _event_id("override")
    all_citations = []
    if prior_decision_event_id:
        all_citations.append(f"router/lapis-pm/decisions/{prior_decision_event_id}")
    all_citations.extend(citations or [])
    entry = RouterPortfolioEntry(
        verdict="overridden",
        claim=f"principal overrode Router decision for `{target_id}`",
        citations=all_citations,
        freshness_stamp=_now_iso(),
        scope_id=_SCOPE_ID,
        ratification_outcome="override",
        target_id=target_id,
        intent_summary=intent_summary,
    )
    key = f"router/lapis-pm/ratification-outcomes/{eid}"
    return _write_entry(key, entry, tags=["lapis-pm", "router-portfolio", f"target:{target_id}", "ratify:override"])


def emit_ratify_redirect(
    target_id: str,
    intent_summary: str,
    prior_decision_event_id: str | None = None,
    citations: list[str] | None = None,
    event_id: str | None = None,
) -> str:
    """router.ratify.redirect — Principal redirected scope mid-flow.

    verdict=redirected; namespace=router/lapis-pm/ratification-outcomes/<event_id>
    Returns the mem key written.
    """
    eid = event_id or _event_id("redirect")
    all_citations = []
    if prior_decision_event_id:
        all_citations.append(f"router/lapis-pm/decisions/{prior_decision_event_id}")
    all_citations.extend(citations or [])
    entry = RouterPortfolioEntry(
        verdict="redirected",
        claim=f"principal redirected scope for `{target_id}`",
        citations=all_citations,
        freshness_stamp=_now_iso(),
        scope_id=_SCOPE_ID,
        ratification_outcome="redirect",
        target_id=target_id,
        intent_summary=intent_summary,
    )
    key = f"router/lapis-pm/ratification-outcomes/{eid}"
    return _write_entry(key, entry, tags=["lapis-pm", "router-portfolio", f"target:{target_id}", "ratify:redirect"])


def emit_expert_outcome(
    target_id: str,
    expert_chosen: str,
    outcome: Literal["landed", "reverted", "flagged", "orphaned"],
    fragment_id: str | None = None,
    drift_class: str | None = None,
    citations: list[str] | None = None,
    primitive_decomposition: list[dict] | None = None,
    event_id: str | None = None,
) -> str:
    """router.expert.outcome — Record an Expert's outcome for a target.

    verdict = outcome (landed | reverted | flagged | orphaned)
    namespace = router/lapis-pm/expert-experience/<expert_tier>/<intent_shape>/<event_id>
    Returns the mem key written.
    """
    eid = event_id or _event_id("expert")
    # intent_shape derived from primitive_decomposition when present, else "unstructured"
    if primitive_decomposition:
        shape_parts = [
            p.get("fragment_id", p.get("intent", ""))
            for p in primitive_decomposition[:2]
        ]
        intent_shape = "-".join(s for s in shape_parts if s) or "unstructured"
    else:
        intent_shape = "unstructured"

    entry = RouterPortfolioEntry(
        verdict=outcome,
        claim=f"expert `{expert_chosen}` outcome for `{target_id}`",
        citations=citations or [],
        freshness_stamp=_now_iso(),
        scope_id=_SCOPE_ID,
        fragment_id=fragment_id,
        expert_chosen=expert_chosen,
        target_id=target_id,
        drift_class=drift_class,
        primitive_decomposition=primitive_decomposition,
    )
    key = f"router/lapis-pm/expert-experience/{expert_chosen}/{intent_shape}/{eid}"
    return _write_entry(
        key, entry,
        tags=["lapis-pm", "router-portfolio", f"target:{target_id}", f"expert:{expert_chosen}", f"outcome:{outcome}"],
    )


# ---------------------------------------------------------------------------
# Session-level helpers
# ---------------------------------------------------------------------------

def read_session_entries(since_iso: str | None = None) -> list[dict]:
    """Return all router/lapis-pm/* entries, optionally filtered by freshness_stamp >= since_iso.

    Used by /router-checkpoint to consolidate a session's writes.
    """
    mem = _mem()
    raw = mem.list_all(tag="router-portfolio", limit=500)
    results = []
    for row in raw:
        try:
            data = json.loads(row["content"])
        except (json.JSONDecodeError, KeyError):
            continue
        if since_iso and data.get("freshness_stamp", "") < since_iso:
            continue
        data["_mem_key"] = row.get("key", "")
        results.append(data)
    return results


def write_session_summary(
    slug: str,
    summary_markdown: str,
    entries: list[dict],
    session_start_iso: str | None = None,
) -> str:
    """Write a session summary to router/lapis-pm/sessions/<YYYY-MM-DD-HHMM>-<slug>.

    Returns the mem key written.
    """
    now = datetime.now(timezone.utc)
    ts_prefix = now.strftime("%Y-%m-%d-%H%M")
    key = f"router/lapis-pm/sessions/{ts_prefix}-{slug}"

    payload = {
        "summary": summary_markdown,
        "session_start": session_start_iso or "",
        "written_at": _now_iso(),
        "event_count": len(entries),
        "event_keys": [e.get("_mem_key", "") for e in entries],
    }
    _mem().set(key, json.dumps(payload, ensure_ascii=False), tags=["lapis-pm", "router-portfolio", "router-session"])
    return key


def find_latest_decision(target_id: str) -> dict | None:
    """Return the most recent router/lapis-pm/decisions/* entry for target_id.

    Searches mem by the target:<target_id> tag, filters to the decisions/
    namespace only, and returns the entry with the latest freshness_stamp.
    Returns None if no matching decision exists.

    Only router/lapis-pm/decisions/* keys are eligible; ratification-outcomes
    and other namespaces are ignored.
    """
    rows = _mem().list_all(tag=f"target:{target_id}", limit=500)
    best: dict | None = None
    best_stamp = ""
    for row in rows:
        key = row.get("key", "")
        if not key.startswith("router/lapis-pm/decisions/"):
            continue
        try:
            data = json.loads(row["content"])
        except (json.JSONDecodeError, KeyError):
            continue
        stamp = data.get("freshness_stamp", "")
        if stamp > best_stamp:
            best_stamp = stamp
            best = dict(data)
            best["_mem_key"] = key
    return best


def session_checkpoint_exists(since_iso: str) -> bool:
    """Return True if a session summary exists with written_at >= since_iso.

    Used by the Stop hook to determine whether /router-checkpoint ran.
    """
    mem = _mem()
    rows = mem.list_all(tag="router-session", limit=100)
    for row in rows:
        try:
            data = json.loads(row["content"])
            if data.get("written_at", "") >= since_iso:
                return True
        except (json.JSONDecodeError, KeyError):
            continue
    return False
