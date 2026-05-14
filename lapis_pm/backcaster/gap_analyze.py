"""Backcaster Stage 2 — Gap Analyzer.

For each precondition, asks:
  - What exists today?
  - What's missing?
  - What's miswired?

Retrieval sources:
  1. Synapse service at http://203.0.113.12:8401/serve (fail-soft)
  2. mem.db architecture/* + project/* keys (fail-soft)

Returns a list of Gap objects with citation-anchored statements.

Stub mode (BACKCASTER_STUB=1): returns canned gaps.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from .schema import BackcasterCitation, Gap, Precondition

# Module-level import so tests can patch lapis_pm.backcaster.llm_routing.call_model_sync
from .llm_routing import call_model_sync

log = logging.getLogger(__name__)

_SYNAPSE_TIMEOUT = 10  # seconds; fail-soft if slow

_SYSTEM_PROMPT = """\
You are Backcaster, analyzing gaps between the current state of the world and \
a required precondition for a goal-state.

You will be given:
  1. A precondition (what must be true for the goal to be reachable)
  2. Optional context passages from a civic-theory corpus and ecosystem state

Your task: write a gap analysis with three parts:
  - what_exists: what currently exists that is relevant (be concrete; if nothing, say so)
  - what_missing: what is absent that would need to be created or established
  - what_miswired: what exists but points in the wrong direction or creates friction

Return ONLY a JSON object:
{
  "what_exists": "<concrete statement>",
  "what_missing": "<concrete statement>",
  "what_miswired": "<concrete statement or 'nothing identified'>"
}

Be honest about uncertainty. Do not invent facts.
"""

# Canned stub gaps — one per stub precondition id
_STUB_GAPS: dict[str, dict[str, str]] = {
    "psychological-01": {
        "what_exists": "Some small-group creative tools (Google Docs, Notion) offer invite-only spaces.",
        "what_missing": "A cultural frame that names small-group creative work as legitimate without growth pressure.",
        "what_miswired": "Existing platforms optimize for public sharing and growth metrics, creating implicit pressure to expand.",
    },
    "psychological-02": {
        "what_exists": "Collaborative norms exist in some open-source communities.",
        "what_missing": "Scaffolding that helps friend-groups define their own autonomy/interdependence contract explicitly.",
        "what_miswired": "Most platforms impose contribution models (owner/contributor) that don't match friendship dynamics.",
    },
    "material-01": {
        "what_exists": "Smartphones and laptops are widely available in the target demographic.",
        "what_missing": "Nothing material is missing for baseline cases.",
        "what_miswired": "nothing identified",
    },
    "infrastructural-01": {
        "what_exists": "Notion, Obsidian Publish, and similar tools offer small-group spaces.",
        "what_missing": "A platform with friend-group-shaped permissions that doesn't require an enterprise admin model.",
        "what_miswired": "Current platforms assume either fully public or org-hierarchy access models.",
    },
    "infrastructural-02": {
        "what_exists": "Some tools (Notion, Obsidian) offer export.",
        "what_missing": "Standardized portable formats for creative collaboration outputs.",
        "what_miswired": "Export features exist but are rarely discoverable or usable by non-technical users.",
    },
    "governance-01": {
        "what_exists": "Informal norms emerge organically in long-standing friend groups.",
        "what_missing": "Lightweight tooling for groups to make implicit norms explicit without bureaucratic overhead.",
        "what_miswired": "Most governance tooling (parliamentary procedure, RACI charts) is enterprise-shaped.",
    },
    "social-norm-01": {
        "what_exists": "Some creative subcultures (zine communities, small music scenes) normalize intentional smallness.",
        "what_missing": "Broader cultural legitimacy for 'good enough for us' as a valid project scope.",
        "what_miswired": "Startup culture and social media metrics conflate success with scale.",
    },
    "economic-01": {
        "what_exists": "Free tiers exist on many platforms.",
        "what_missing": "A pricing model that scales with person-count rather than org-tier features.",
        "what_miswired": "Enterprise pricing bundles collaboration features that individuals need with org-management features they don't.",
    },
}


def _retrieve_synapse(precondition: Precondition, session_id: str) -> tuple[list[BackcasterCitation], str]:
    """Query Synapse for context relevant to this precondition.

    Returns (citations, context_text). Fail-soft: returns ([], "") on error.
    """
    try:
        import httpx
        synapse_url = os.environ.get("SYNAPSE_URL", "http://203.0.113.12:8401")
        resp = httpx.post(
            f"{synapse_url}/serve",
            json={"session_id": session_id, "prompt": precondition.statement},
            timeout=_SYNAPSE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract hit refs from payload.hits
        payload = data.get("payload", {})
        hits = payload.get("hits", [])
        citations = []
        context_parts = []
        for hit in hits[:3]:  # cap at 3 context passages
            ref = hit.get("id") or hit.get("ref") or hit.get("path") or ""
            excerpt = hit.get("content") or hit.get("excerpt") or hit.get("text") or ""
            if ref:
                citations.append(BackcasterCitation(type="corpus", ref=ref))
            if excerpt:
                context_parts.append(f"[{ref}]: {excerpt[:300]}")
        return citations, "\n\n".join(context_parts)
    except Exception as exc:  # noqa: BLE001
        log.debug("gap_analyze: Synapse unreachable (%s)", exc)
        return [], ""


def _retrieve_mem(query: str) -> tuple[list[BackcasterCitation], str]:
    """Query mem.db via FTS for context relevant to query.

    Returns (citations, context_text). Fail-soft: returns ([], "") on error.
    """
    try:
        from agents_core.mem import MemoryStore
        mem = MemoryStore()
        items = mem.search(query, limit=5)
        citations = []
        context_parts = []
        for item in items:
            key = item.get("key", "")
            content = item.get("content", "")
            if key:
                citations.append(BackcasterCitation(type="mem", ref=key))
            if content:
                context_parts.append(f"[mem:{key}]: {str(content)[:200]}")
        return citations, "\n\n".join(context_parts)
    except Exception as exc:  # noqa: BLE001
        log.debug("gap_analyze: mem.db unreachable (%s)", exc)
        return [], ""


def _parse_gap_output(raw: str, precondition_id: str, citations: list[BackcasterCitation]) -> Gap:
    """Parse LLM JSON into a Gap."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(text)
    gap = Gap(
        precondition_id=precondition_id,
        what_exists=data.get("what_exists", ""),
        what_missing=data.get("what_missing", ""),
        what_miswired=data.get("what_miswired", "nothing identified"),
        citations=citations,
        unsourced=len(citations) == 0,
    )
    return gap


def analyze_gaps(
    preconditions: list[Precondition],
    corpus_paths: list[str] | None = None,
    model: str = "qwen",
    stub: bool = False,
) -> tuple[list[Gap], list[str], str]:
    """Analyze gaps for all preconditions.

    Returns (gaps, degraded_paths, prompt_hash).
    degraded_paths is populated when retrieval layers are unavailable.
    """
    if stub or os.environ.get("BACKCASTER_STUB") == "1":
        gaps = _gaps_from_stub(preconditions)
        return gaps, [], "stub:gap_analyze"

    degraded_paths: list[str] = []
    gaps: list[Gap] = []
    prompt_hashes: list[str] = []

    # Test retrieval availability
    session_id = f"backcaster-{id(preconditions)}"
    synapse_ok = True
    mem_ok = True

    # Quick probe
    synapse_url = os.environ.get("SYNAPSE_URL", "http://203.0.113.12:8401")
    try:
        import httpx
        httpx.get(f"{synapse_url}/healthz", timeout=3)
    except Exception:  # noqa: BLE001
        synapse_ok = False
        degraded_paths.append("synapse")
        log.warning("gap_analyze: Synapse unreachable - running in LLM-only mode")

    try:
        from agents_core.mem import MemoryStore
        MemoryStore()
    except Exception:  # noqa: BLE001
        mem_ok = False
        degraded_paths.append("mem")
        log.warning("gap_analyze: mem.db unreachable - running without mem context")

    for prec in preconditions:
        # Gather context
        corpus_citations: list[BackcasterCitation] = []
        corpus_context = ""
        mem_citations: list[BackcasterCitation] = []
        mem_context = ""

        if synapse_ok:
            corpus_citations, corpus_context = _retrieve_synapse(prec, session_id)
        if mem_ok:
            mem_citations, mem_context = _retrieve_mem(prec.statement)

        all_citations = corpus_citations + mem_citations

        # Build context section
        context_section = ""
        if corpus_context:
            context_section += f"\n\nCorpus context:\n{corpus_context}"
        if mem_context:
            context_section += f"\n\nEcosystem state context:\n{mem_context}"

        prompt = (
            f"Precondition (axis: {prec.axis}):\n{prec.statement}"
            f"{context_section}\n\nAnalyze the gap."
        )
        prompt_hash = "sha256:" + hashlib.sha256(prompt.encode()).hexdigest()
        prompt_hashes.append(prompt_hash)

        raw = None
        try:
            raw = call_model_sync(model, prompt=prompt, system=_SYSTEM_PROMPT, json_mode=True, timeout=120)
        except Exception as exc:  # noqa: BLE001
            log.warning("gap_analyze: LLM call failed for %s (%s)", prec.id, exc)
        if raw is None:
            gaps.append(Gap(
                precondition_id=prec.id,
                what_exists="",
                what_missing="LLM call failed",
                what_miswired="nothing identified",
                citations=[],
                unsourced=True,
            ))
            continue

        try:
            gap = _parse_gap_output(raw, prec.id, all_citations)
        except Exception as exc:  # noqa: BLE001
            log.warning("gap_analyze: parse failed for %s (%s)", prec.id, exc)
            gap = Gap(
                precondition_id=prec.id,
                what_exists="",
                what_missing="parse failure",
                what_miswired="nothing identified",
                citations=[],
                unsourced=True,
            )

        gaps.append(gap)

    combined_hash = "sha256:" + hashlib.sha256(
        b"\x00".join(h.encode() for h in prompt_hashes)
    ).hexdigest() if prompt_hashes else "none"

    return gaps, degraded_paths, combined_hash


def _gaps_from_stub(preconditions: list[Precondition]) -> list[Gap]:
    gaps = []
    for prec in preconditions:
        stub = _STUB_GAPS.get(prec.id, {
            "what_exists": "No stub entry for this precondition.",
            "what_missing": "Unknown.",
            "what_miswired": "nothing identified",
        })
        gaps.append(Gap(
            precondition_id=prec.id,
            what_exists=stub["what_exists"],
            what_missing=stub["what_missing"],
            what_miswired=stub["what_miswired"],
            citations=[],
            unsourced=True,
        ))
    return gaps
