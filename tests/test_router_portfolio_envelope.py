"""Tests for RouterPortfolioEntry.to_envelope() — Move 3 Surface 3.

Verifies that to_envelope() produces v0-conformant LapisToolReturn envelopes
per the Lapis Provenance Schema v0 field mapping.
"""
from __future__ import annotations

import pytest

from archetypes_core.corroboration import Citation
from archetypes_core.provenance import InputRef, LapisToolReturn, UpstreamRef, Provenance

from lapis_pm.router_portfolio import RouterPortfolioEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _full_entry() -> RouterPortfolioEntry:
    """A RouterPortfolioEntry with all fields populated."""
    return RouterPortfolioEntry(
        verdict="proposed",
        claim="kickoff fragment+expert composition for target X",
        citations=["pm/dispatched/foo", "router/lapis-pm/decisions/2026-05-03-abc"],
        freshness_stamp="2026-05-03T12:00:00Z",
        scope_id="router/lapis-pm",
        drift_class="thematic",
        primitive_decomposition=[{"fragment_id": "kickoff"}, {"fragment_id": "tick"}],
        fragment_id="kickoff",
        expert_chosen="qwen-local",
        ratification_outcome="confirm",
        target_id="example-target",
        intent_summary="Router proposes kickoff with qwen-local for example-target.",
    )


# ---------------------------------------------------------------------------
# Test 1: Basic wrap
# ---------------------------------------------------------------------------

def test_basic_wrap():
    """to_envelope() returns a LapisToolReturn; payload is the entry; schema_version correct."""
    entry = _full_entry()
    envelope = entry.to_envelope()

    assert isinstance(envelope, LapisToolReturn)
    assert envelope.payload is entry
    assert envelope.summary == entry.intent_summary
    assert envelope.provenance.schema_version == "lapis-provenance-v0"


# ---------------------------------------------------------------------------
# Test 2: Default agent_id and tool fallback chain
# ---------------------------------------------------------------------------

def test_default_agent_id_and_tool():
    """Default agent_id == 'lapis-pm/router'; tool falls back fragment_id → 'router_emit'."""
    entry = _full_entry()
    envelope = entry.to_envelope()
    assert envelope.provenance.agent_id == "lapis-pm/router"
    assert envelope.provenance.tool == "kickoff"  # fragment_id

    # fragment_id=None → "router_emit"
    entry_no_frag = RouterPortfolioEntry(
        verdict="ratified",
        claim="some claim",
        citations=[],
        freshness_stamp="2026-05-03T12:00:00Z",
        fragment_id=None,
    )
    envelope2 = entry_no_frag.to_envelope()
    assert envelope2.provenance.tool == "router_emit"

    # explicit tool overrides fragment_id
    envelope3 = entry.to_envelope(tool="custom-tool")
    assert envelope3.provenance.tool == "custom-tool"


# ---------------------------------------------------------------------------
# Test 3: expert_chosen → provenance.model
# ---------------------------------------------------------------------------

def test_expert_chosen_to_model():
    """expert_chosen flows to provenance.model for both Qwen and Opus values."""
    for model_val in ("qwen-local", "claude-opus-4-6"):
        entry = RouterPortfolioEntry(
            verdict="proposed",
            claim="c",
            citations=[],
            freshness_stamp="2026-05-03T12:00:00Z",
            expert_chosen=model_val,
        )
        envelope = entry.to_envelope()
        assert envelope.provenance.model == model_val

    # None expert_chosen → None model
    entry_no_model = RouterPortfolioEntry(
        verdict="proposed", claim="c", citations=[], freshness_stamp="2026-05-03T12:00:00Z"
    )
    assert entry_no_model.to_envelope().provenance.model is None


# ---------------------------------------------------------------------------
# Test 4: intent_summary → summary; fallback when None
# ---------------------------------------------------------------------------

def test_intent_summary_to_summary_and_fallback():
    """intent_summary flows to top-level summary; fallback used when intent_summary is None."""
    entry = _full_entry()
    envelope = entry.to_envelope()
    assert envelope.summary == "Router proposes kickoff with qwen-local for example-target."

    # fallback: "<verdict>: <fragment_id or 'router-event'>"
    entry_no_summary = RouterPortfolioEntry(
        verdict="ratified",
        claim="c",
        citations=[],
        freshness_stamp="2026-05-03T12:00:00Z",
        fragment_id="tick",
        intent_summary=None,
    )
    envelope2 = entry_no_summary.to_envelope()
    assert envelope2.summary == "ratified: tick"

    # fallback when both intent_summary and fragment_id are None
    entry_minimal = RouterPortfolioEntry(
        verdict="flagged",
        claim="c",
        citations=[],
        freshness_stamp="2026-05-03T12:00:00Z",
        fragment_id=None,
        intent_summary=None,
    )
    assert entry_minimal.to_envelope().summary == "flagged: router-event"


# ---------------------------------------------------------------------------
# Test 5: citations list[str] → list[Citation]
# ---------------------------------------------------------------------------

def test_citations_wrap():
    """Three string citations wrap to three Citation objects; empty list stays empty."""
    sources = ["src/a", "src/b", "src/c"]
    entry = RouterPortfolioEntry(
        verdict="proposed",
        claim="c",
        citations=sources,
        freshness_stamp="2026-05-03T12:00:00Z",
    )
    envelope = entry.to_envelope()
    cits = envelope.provenance.citations
    assert len(cits) == 3
    for i, s in enumerate(sources):
        c = cits[i]
        assert isinstance(c, Citation)
        assert c.source_id == s
        assert c.excerpt == ""
        assert c.content_hash is None
        assert c.provenance_method == "router_referenced"
        # confidence is not a v0 field on Citation; defensively verify not set
        assert getattr(c, "confidence", None) is None

    # empty citations
    entry_empty = RouterPortfolioEntry(
        verdict="proposed", claim="c", citations=[], freshness_stamp="2026-05-03T12:00:00Z"
    )
    assert entry_empty.to_envelope().provenance.citations == []


# ---------------------------------------------------------------------------
# Test 6: primitive_decomposition stays in payload; provenance stays None
# ---------------------------------------------------------------------------

def test_primitive_decomposition_stays_in_payload():
    """primitive_decomposition remains in payload; envelope provenance.primitive_decomposition is None."""
    prims = [{"fragment_id": "kickoff"}, {"fragment_id": "review"}]
    entry = RouterPortfolioEntry(
        verdict="proposed",
        claim="c",
        citations=[],
        freshness_stamp="2026-05-03T12:00:00Z",
        primitive_decomposition=prims,
    )
    envelope = entry.to_envelope()
    assert envelope.payload.primitive_decomposition == prims
    assert envelope.provenance.primitive_decomposition is None


# ---------------------------------------------------------------------------
# Test 7: scope_id flows to provenance
# ---------------------------------------------------------------------------

def test_scope_id_flows_to_provenance():
    """scope_id is duplicated into envelope.provenance.scope_id."""
    entry = RouterPortfolioEntry(
        verdict="proposed",
        claim="c",
        citations=[],
        freshness_stamp="2026-05-03T12:00:00Z",
        scope_id="router/lapis-pm",
    )
    envelope = entry.to_envelope()
    assert envelope.provenance.scope_id == "router/lapis-pm"
    assert envelope.payload.scope_id == "router/lapis-pm"


# ---------------------------------------------------------------------------
# Test 8: fragment_id, target_id, ratification_outcome stay in payload
# ---------------------------------------------------------------------------

def test_router_fields_stay_in_payload():
    """Router-specific fields are accessible via payload and not in provenance dict."""
    entry = _full_entry()
    envelope = entry.to_envelope()

    assert envelope.payload.fragment_id == "kickoff"
    assert envelope.payload.target_id == "example-target"
    assert envelope.payload.ratification_outcome == "confirm"

    prov_dict = envelope.provenance.to_dict()
    assert "fragment_id" not in prov_dict
    assert "target_id" not in prov_dict
    assert "ratification_outcome" not in prov_dict


# ---------------------------------------------------------------------------
# Test 9: prompt_hash is None even when model is set
# ---------------------------------------------------------------------------

def test_prompt_hash_none_when_model_set():
    """prompt_hash is always None; transcribed-provenance pattern; no exception raised."""
    entry = RouterPortfolioEntry(
        verdict="proposed",
        claim="c",
        citations=[],
        freshness_stamp="2026-05-03T12:00:00Z",
        expert_chosen="qwen-local",
    )
    envelope = entry.to_envelope()
    assert envelope.provenance.model == "qwen-local"
    assert envelope.provenance.prompt_hash is None


# ---------------------------------------------------------------------------
# Test 10: caller-supplied input_refs and upstream_calls preserved
# ---------------------------------------------------------------------------

def test_caller_supplied_refs_preserved():
    """input_refs and upstream_calls passed by caller appear in provenance."""
    ir = InputRef(ref="some/file.py", content_hash=None, type="file")
    ur = UpstreamRef(agent_id="lapis-pm/reviewer", manifest_hash="sha256:abc123")
    entry = RouterPortfolioEntry(
        verdict="proposed", claim="c", citations=[], freshness_stamp="2026-05-03T12:00:00Z"
    )
    envelope = entry.to_envelope(input_refs=[ir], upstream_calls=[ur])
    assert envelope.provenance.input_refs == [ir]
    assert envelope.provenance.upstream_calls == [ur]


# ---------------------------------------------------------------------------
# Test 11: to_envelope() does not mutate the entry
# ---------------------------------------------------------------------------

def test_to_envelope_does_not_mutate_entry():
    """Calling to_envelope() leaves the entry's to_dict() output byte-identical."""
    entry = _full_entry()
    before = entry.to_dict()
    entry.to_envelope()
    after = entry.to_dict()
    assert before == after


# ---------------------------------------------------------------------------
# Test 12: Round-trip via to_dict / from_dict
# ---------------------------------------------------------------------------

def test_round_trip_via_dict():
    """LapisToolReturn.from_dict(envelope.to_dict()) reconstructs the envelope.

    v0 limitation: payload comes back as a dict, not the original RouterPortfolioEntry
    dataclass. Reconstructing the dataclass requires knowing its type, which the
    envelope doesn't carry at v0. v1 may add a payload_type discriminator.
    """
    entry = _full_entry()
    envelope = entry.to_envelope()

    d = envelope.to_dict()
    restored = LapisToolReturn.from_dict(d)

    # Payload is a dict after round-trip (v0 limitation — documented)
    assert isinstance(restored.payload, dict)
    assert restored.payload == entry.to_dict()

    assert restored.summary == envelope.summary
    assert restored.provenance.schema_version == envelope.provenance.schema_version
    assert restored.provenance.manifest_hash == envelope.provenance.manifest_hash
    assert restored.provenance.agent_id == envelope.provenance.agent_id
    assert restored.provenance.tool == envelope.provenance.tool
    assert restored.provenance.model == envelope.provenance.model
    assert restored.provenance.scope_id == envelope.provenance.scope_id
    assert len(restored.provenance.citations) == len(envelope.provenance.citations)
    for orig_c, rest_c in zip(envelope.provenance.citations, restored.provenance.citations):
        assert orig_c.source_id == rest_c.source_id
        assert orig_c.provenance_method == rest_c.provenance_method
