"""Tests: backcaster-quest-leg-v0 acceptance criteria.

All tests use stubbed Dowser client, flip-controller, and doorman.
No live GW / StarHouse / SearXNG / network dependency.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lapis_pm.backcaster.schema import BackcasterCitation, Gap
from lapis_pm.backcaster.quest_leg import (
    _audit_citation,
    _gate_doorman_serving,
    _is_known_hopeless,
    _write_sidecar,
    quest_source_run,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path: Path, gaps: list[dict], epistemic_caution: str = "high") -> tuple[Path, str]:
    """Create a minimal run dir with gaps.yaml, run.yaml, decomposition.yaml."""
    run_id = "test-run-2026-06-28"
    run_dir = tmp_path / run_id
    run_dir.mkdir()

    # gaps.yaml
    (run_dir / "gaps.yaml").write_text(
        yaml.dump({"gaps": gaps}, default_flow_style=False, allow_unicode=True)
    )

    # run.yaml
    (run_dir / "run.yaml").write_text(
        yaml.dump({
            "run_id": run_id,
            "goal_file": "/srv/lapis/backcaster/goals/test.md",
            "model": "gravitywell",
            "epistemic_caution": epistemic_caution,
            "corpus_used": [],
            "scenario_ids": [],
            "timing": {},
            "prompt_hashes": {},
            "concentration_warning": [],
            "degraded_paths": [],
            "derive_fallback_count": 0,
        }, default_flow_style=False)
    )

    # decomposition.yaml (with precondition statements)
    (run_dir / "decomposition.yaml").write_text(
        yaml.dump({
            "axes": ["economic"],
            "preconditions": [
                {"id": g["precondition_id"], "axis": "economic",
                 "statement": f"Statement for {g['precondition_id']}",
                 "implicit_dependencies": []}
                for g in gaps
            ],
        }, default_flow_style=False)
    )

    # components.yaml (minimal)
    (run_dir / "components.yaml").write_text(
        yaml.dump({"components": []}, default_flow_style=False)
    )

    return run_dir, run_id


def _stub_dowser(
    read_outcome: str = "sources-found",
    read_citations: list[dict] | None = None,
    critique_status: str = "pass",
    critique_diagnosis: str = "",
    read_findings: str = "Relevant findings from the web.",
) -> MagicMock:
    """Return a minimal Dowser stub."""
    if read_citations is None:
        read_citations = [
            {"url": "https://arxiv.org/abs/1234.5678", "title": "Test Paper",
             "excerpt": "Direct excerpt from source.", "credibility": "high"},
        ]

    dowser = MagicMock()

    def _read_batch(requests_list, read_operator="quest", budget=None):
        drafts = []
        for req in requests_list:
            drafts.append({
                "intent": req["intent"],
                "findings": read_findings if read_outcome == "sources-found" else "",
                "citations": read_citations if read_outcome == "sources-found" else [],
                "outcome": read_outcome,
                "provenance": {
                    "search_strings": [req["intent"][:50]],
                    "hits_count": 5 if read_outcome != "infra-unavailable" else 0,
                    "triaged_urls": [c["url"] for c in read_citations] if read_citations else [],
                    "read_urls": [c["url"] for c in read_citations] if read_citations else [],
                    "friction_ratio": 0.1,
                    "rewrote_from_diagnosis": bool(req.get("prior_diagnosis")),
                    "notes": [],
                },
            })
        return {"drafts": drafts}

    def _critique_batch(drafts, critic_operator="gravitywell"):
        verdicts = []
        for draft in drafts:
            verdicts.append({
                "intent": draft["intent"],
                "status": critique_status,
                "verdict": {"relevance": 8, "credibility": 8, "faithfulness": 8, "confidence": "high"},
                "diagnosis": critique_diagnosis,
            })
        return {"verdicts": verdicts}

    dowser.read_batch = MagicMock(side_effect=_read_batch)
    dowser.critique_batch = MagicMock(side_effect=_critique_batch)
    return dowser


def _always_ok(mode: str) -> bool:
    return True


def _always_serving(timeout_s: int = 240) -> bool:
    return True


def _with_runs_root(tmp_path: Path):
    """Context manager patcher: redirect room_path('backcaster.runs') to tmp_path."""
    return patch(
        "lapis_pm.backcaster.quest_leg.Path",
        side_effect=lambda *a, **kw: Path(*a, **kw),
    )


# ---------------------------------------------------------------------------
# AC2: CitationType accepts "web"; attach-then-clear ordering
# ---------------------------------------------------------------------------

def test_web_citation_type_accepted():
    """CitationType includes 'web'; a Gap with a web citation serializes with unsourced=false."""
    c = BackcasterCitation(type="web", ref="https://arxiv.org/abs/test")
    assert c.type == "web"
    assert c.to_dict() == {"type": "web", "ref": "https://arxiv.org/abs/test"}


def test_gap_attach_web_citation_clears_unsourced():
    """Attach web citation FIRST, then set unsourced=False — validator must not trip."""
    g = Gap(
        precondition_id="p-01",
        what_exists="x", what_missing="y", what_miswired="z",
        citations=[],
        unsourced=True,
    )
    assert g.unsourced is True
    # Attach first
    g.citations.append(BackcasterCitation(type="web", ref="https://arxiv.org/abs/1"))
    # Then clear
    g.unsourced = False
    assert g.unsourced is False
    d = g.to_dict()
    assert d["unsourced"] is False
    assert d["citations"][0]["type"] == "web"


def test_existing_schema_tests_still_pass():
    """Existing corpus/mem/scenario types still accepted."""
    from pydantic import ValidationError
    for t in ("corpus", "mem", "scenario", "web"):
        c = BackcasterCitation(type=t, ref="some/ref")  # type: ignore[arg-type]
        assert c.type == t
    with pytest.raises(ValidationError):
        BackcasterCitation(type="unknown", ref="foo")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC7: llm_routing "quest" route
# ---------------------------------------------------------------------------

def test_llm_routing_quest_calls_operator():
    """call_model_sync('quest', ...) routes via call_operator('quest', on_wake_fail='skip')."""
    with patch("lapis_pm.backcaster.llm_routing.call_operator") as mock_op:
        mock_op.return_value = "some response"
        from lapis_pm.backcaster.llm_routing import call_model_sync
        result = call_model_sync("quest", "test prompt")
        assert result == "some response"
        mock_op.assert_called_once()
        call_kwargs = mock_op.call_args
        assert call_kwargs[0][0] == "quest"  # first positional arg = operator name
        assert call_kwargs[1].get("on_wake_fail") == "skip"


def test_llm_routing_quest_returns_none_when_operator_unavailable():
    """If call_operator is None (unavailable), quest route returns None without raising."""
    import lapis_pm.backcaster.llm_routing as routing
    original = routing.call_operator
    try:
        routing.call_operator = None
        from lapis_pm.backcaster.llm_routing import call_model_sync
        result = call_model_sync("quest", "test prompt")
        assert result is None
    finally:
        routing.call_operator = original


# ---------------------------------------------------------------------------
# AC1: End-to-end write-back + re-derive on pass
# ---------------------------------------------------------------------------

def test_quest_source_run_pass_writes_back(tmp_path):
    """AC1: pass gap gets web citation, unsourced=False, quest sidecar, re-derive called."""
    gaps = [{
        "precondition_id": "economic-01",
        "what_exists": "fee structures",
        "what_missing": "sub-linear fee scaling",
        "what_miswired": "incentives favor high-value transactions",
        "citations": [],
        "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    dowser = _stub_dowser()

    # Patch the room_path import inside quest_leg
    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch("lapis_pm.backcaster.quest_leg._rederive_gap") as mock_rederive:
            summary = quest_source_run(
                run_id,
                all_unsourced=True,
                _dowser=dowser,
                _flip_fn=_always_ok,
                _gate_fn=_always_serving,
            )

    assert summary["gaps_sourced"] == 1
    assert summary["citations_added"] == 1
    assert summary["caution_after"] in ("low", "medium")  # improved from high

    # Check gaps.yaml updated
    gaps_data = yaml.safe_load((run_dir / "gaps.yaml").read_text())
    g = gaps_data["gaps"][0]
    assert g["unsourced"] is False
    assert any(c["type"] == "web" for c in g["citations"])

    # Check quest sidecar written
    sidecar = yaml.safe_load((run_dir / "quest" / "economic-01.yaml").read_text())
    assert sidecar["verdict"] == "sourced"
    assert len(sidecar["citations"]) >= 1

    # Check re-derive was called
    mock_rederive.assert_called_once()


# ---------------------------------------------------------------------------
# AC3: Typed honest-null routing
# ---------------------------------------------------------------------------

def test_honest_null_no_credible_sources_marks_insufficient(tmp_path):
    """AC3: no-credible-sources -> adversarial counter-query, then insufficient-sources marker."""
    gaps = [{
        "precondition_id": "gap-01",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    # Stub: no-credible-sources outcome, subpar critique
    dowser = _stub_dowser(
        read_outcome="no-credible-sources",
        read_citations=[],
        critique_status="subpar",
        critique_diagnosis="no credible sources found",
    )

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = quest_source_run(
            run_id, all_unsourced=True,
            _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    assert summary["gaps_sourced"] == 0
    sidecar = yaml.safe_load((run_dir / "quest" / "gap-01.yaml").read_text())
    assert sidecar["verdict"] == "insufficient-sources"
    # Next run must skip this gap
    assert _is_known_hopeless(run_dir, "gap-01")


def test_honest_null_high_friction_marks_retry_candidate(tmp_path):
    """AC3: high-friction -> high-friction-retry-candidate, NOT skipped next run."""
    gaps = [{
        "precondition_id": "gap-hf",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    dowser = _stub_dowser(
        read_outcome="high-friction",
        read_citations=[],
        critique_status="subpar",
        critique_diagnosis="high friction",
    )

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = quest_source_run(
            run_id, all_unsourced=True,
            _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    sidecar = yaml.safe_load((run_dir / "quest" / "gap-hf.yaml").read_text())
    assert sidecar["verdict"] == "high-friction-retry-candidate"
    # Must NOT be skipped next run (not known-hopeless)
    assert not _is_known_hopeless(run_dir, "gap-hf")
    assert "escalate_candidates" in summary


def test_honest_null_infra_unavailable_is_transient(tmp_path):
    """AC3: infra-unavailable -> transient marker, not known-hopeless, retried next run."""
    gaps = [{
        "precondition_id": "gap-infra",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    dowser = _stub_dowser(read_outcome="infra-unavailable", read_citations=[], critique_status="subpar")

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = quest_source_run(
            run_id, all_unsourced=True,
            _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    sidecar = yaml.safe_load((run_dir / "quest" / "gap-infra.yaml").read_text())
    assert sidecar["verdict"] == "infra-unavailable"
    # Must NOT be marked known-hopeless
    assert not _is_known_hopeless(run_dir, "gap-infra")


def test_known_hopeless_gap_is_skipped(tmp_path):
    """AC3: gap with insufficient-sources sidecar is skipped on next run."""
    gaps = [{
        "precondition_id": "gap-skip",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    # Pre-write the hopeless sidecar
    _write_sidecar(run_dir, "gap-skip", {"quest_attempted": "2026-06-27", "verdict": "insufficient-sources"})

    dowser = _stub_dowser()
    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = quest_source_run(
            run_id, all_unsourced=True,
            _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    assert summary["gaps_targeted"] == 0
    # Dowser should not have been called
    dowser.read_batch.assert_not_called()


def test_no_fabricated_citation_on_subpar(tmp_path):
    """AC3: subpar critique -> gap stays unsourced, no web citation added."""
    gaps = [{
        "precondition_id": "gap-subpar",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    dowser = _stub_dowser(
        read_outcome="no-credible-sources",
        read_citations=[],
        critique_status="subpar",
        critique_diagnosis="thin results",
    )

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        quest_source_run(
            run_id, all_unsourced=True,
            _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    gaps_data = yaml.safe_load((run_dir / "gaps.yaml").read_text())
    g = gaps_data["gaps"][0]
    assert g["unsourced"] is True
    assert not any(c.get("type") == "web" for c in g.get("citations", []))


# ---------------------------------------------------------------------------
# AC3b: Provenance audit drops junk citations
# ---------------------------------------------------------------------------

def test_provenance_audit_drops_junk_domain(tmp_path):
    """AC3b: junk-domain citation is dropped; if none survive, unsourced stays True."""
    gaps = [{
        "precondition_id": "gap-junk",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)

    # Stub: pass verdict but junk-domain citation
    junk_cit = [
        {"url": "https://reddit.com/r/something/junk", "title": "Junk",
         "excerpt": "Some excerpt from reddit.", "credibility": "low"},
    ]
    dowser = _stub_dowser(
        read_outcome="sources-found",
        read_citations=junk_cit,
        critique_status="pass",
        read_findings="Findings from junk source.",
    )

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = quest_source_run(
            run_id, all_unsourced=True,
            _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    # Gap must remain unsourced - no valid citations survived audit
    gaps_data = yaml.safe_load((run_dir / "gaps.yaml").read_text())
    g = gaps_data["gaps"][0]
    assert g["unsourced"] is True
    assert summary["citations_added"] == 0


def test_provenance_audit_passes_credible_domain(tmp_path):
    """AC3b: credible-domain citation passes audit; unsourced flips to False."""
    gaps = [{
        "precondition_id": "gap-credible",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    good_cit = [
        {"url": "https://arxiv.org/abs/9999.0001", "title": "Paper",
         "excerpt": "Direct excerpt supporting the claim.", "credibility": "high"},
    ]
    dowser = _stub_dowser(read_outcome="sources-found", read_citations=good_cit, critique_status="pass")

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch("lapis_pm.backcaster.quest_leg._rederive_gap"):
            summary = quest_source_run(
                run_id, all_unsourced=True,
                _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
            )

    gaps_data = yaml.safe_load((run_dir / "gaps.yaml").read_text())
    g = gaps_data["gaps"][0]
    assert g["unsourced"] is False
    assert summary["citations_added"] == 1


# ---------------------------------------------------------------------------
# AC4: Caution recomputed and run.yaml rewritten
# ---------------------------------------------------------------------------

def test_caution_recomputed_after_writeback(tmp_path):
    """AC4: after sourcing all gaps, caution recomputes and run.yaml is rewritten."""
    # All 6 gaps unsourced -> caution=high; source all -> caution=low
    gap_ids = [f"econ-{i:02d}" for i in range(6)]
    gaps = [
        {"precondition_id": pid, "what_exists": "x", "what_missing": "y",
         "what_miswired": "z", "citations": [], "unsourced": True}
        for pid in gap_ids
    ]
    run_dir, run_id = _make_run_dir(tmp_path, gaps, epistemic_caution="high")
    dowser = _stub_dowser()

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch("lapis_pm.backcaster.quest_leg._rederive_gap"):
            summary = quest_source_run(
                run_id, all_unsourced=True,
                _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
            )

    assert summary["caution_before"] == "high"
    assert summary["caution_after"] == "low"  # all sourced -> < 20% unsourced
    run_data = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert run_data["epistemic_caution"] == "low"


# ---------------------------------------------------------------------------
# AC5: One diagnosis-driven retry, capped
# ---------------------------------------------------------------------------

def test_retry_capped_at_one(tmp_path):
    """AC5: subpar-then-pass gap re-calls read_batch once with prior_diagnosis; still-subpar -> null."""
    gaps = [{
        "precondition_id": "gap-retry",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)

    call_count = {"read": 0}
    def _read_batch(requests_list, read_operator="quest", budget=None):
        call_count["read"] += 1
        # First call: no-credible-sources (subpar); second call (retry): still subpar
        outcome = "no-credible-sources"
        return {"drafts": [{
            "intent": requests_list[0]["intent"],
            "findings": "",
            "citations": [],
            "outcome": outcome,
            "provenance": {"search_strings": [], "hits_count": 0, "triaged_urls": [],
                           "read_urls": [], "friction_ratio": 0.0,
                           "rewrote_from_diagnosis": call_count["read"] > 1, "notes": []},
        }]}

    dowser = MagicMock()
    dowser.read_batch = MagicMock(side_effect=_read_batch)
    dowser.critique_batch = MagicMock(return_value={"verdicts": [{
        "intent": "y",
        "status": "subpar",
        "verdict": {"relevance": 2, "credibility": 2, "faithfulness": 2, "confidence": "high"},
        "diagnosis": "no relevant hits",
    }]})

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = quest_source_run(
            run_id, all_unsourced=True,
            _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    # read_batch called at most 3 times: initial + 1 retry + 1 adversarial counter-query
    assert dowser.read_batch.call_count <= 3
    # Gap stays unsourced after capped retry
    assert summary["gaps_sourced"] == 0
    sidecar = yaml.safe_load((run_dir / "quest" / "gap-retry.yaml").read_text())
    assert sidecar["verdict"] == "insufficient-sources"


def test_retry_prior_diagnosis_passed(tmp_path):
    """AC5: subpar-then-pass: retry carries prior_diagnosis; on pass, gap is sourced."""
    gaps = [{
        "precondition_id": "gap-retry-pass",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)

    call_count = {"read": 0}
    good_cit = [{"url": "https://arxiv.org/abs/retry-test", "title": "T",
                 "excerpt": "Excerpt.", "credibility": "high"}]

    def _read_batch(requests_list, read_operator="quest", budget=None):
        call_count["read"] += 1
        # First call fails; second call (retry) succeeds
        if call_count["read"] == 1:
            return {"drafts": [{"intent": requests_list[0]["intent"], "findings": "",
                "citations": [], "outcome": "no-credible-sources",
                "provenance": {"search_strings": [], "hits_count": 0, "triaged_urls": [],
                               "read_urls": [], "friction_ratio": 0.0,
                               "rewrote_from_diagnosis": False, "notes": []}}]}
        # Retry: check prior_diagnosis was passed
        assert requests_list[0].get("prior_diagnosis"), "prior_diagnosis must be set on retry"
        return {"drafts": [{"intent": requests_list[0]["intent"],
            "findings": "Found relevant info.",
            "citations": good_cit, "outcome": "sources-found",
            "provenance": {"search_strings": [], "hits_count": 3, "triaged_urls": [],
                           "read_urls": [], "friction_ratio": 0.1,
                           "rewrote_from_diagnosis": True, "notes": []}}]}

    critique_calls = {"n": 0}
    def _critique_batch(drafts, critic_operator="gravitywell"):
        critique_calls["n"] += 1
        if critique_calls["n"] == 1:
            return {"verdicts": [{"intent": drafts[0]["intent"], "status": "subpar",
                "verdict": {"relevance": 2, "credibility": 2, "faithfulness": 2, "confidence": "high"},
                "diagnosis": "thin results"}]}
        return {"verdicts": [{"intent": drafts[0]["intent"], "status": "pass",
            "verdict": {"relevance": 9, "credibility": 9, "faithfulness": 9, "confidence": "high"},
            "diagnosis": ""}]}

    dowser = MagicMock()
    dowser.read_batch = MagicMock(side_effect=_read_batch)
    dowser.critique_batch = MagicMock(side_effect=_critique_batch)

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch("lapis_pm.backcaster.quest_leg._rederive_gap"):
            summary = quest_source_run(
                run_id, all_unsourced=True,
                _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
            )

    assert summary["gaps_sourced"] == 1
    assert summary["retries_used"] == 1


# ---------------------------------------------------------------------------
# AC6: Flip sequencing (local-only)
# ---------------------------------------------------------------------------

def test_flip_sequence_local_only(tmp_path):
    """AC6: local-only path flips swarm->big; each phase gated on doorman serving."""
    gaps = [{
        "precondition_id": "gap-flip",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    dowser = _stub_dowser()

    flip_calls: list[str] = []
    gate_calls: list[int] = []

    def _mock_flip(mode: str) -> bool:
        flip_calls.append(mode)
        return True

    def _mock_gate(timeout_s: int = 240) -> bool:
        gate_calls.append(timeout_s)
        return True

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch("lapis_pm.backcaster.quest_leg._rederive_gap"):
            quest_source_run(
                run_id, all_unsourced=True, escalate=False,
                _dowser=dowser, _flip_fn=_mock_flip, _gate_fn=_mock_gate,
            )

    # Must flip: swarm (read phase), big (critique phase), big (end invariant)
    assert "swarm" in flip_calls
    assert "big" in flip_calls
    # Swarm must come before big
    swarm_idx = flip_calls.index("swarm")
    big_idx = flip_calls.index("big")
    assert swarm_idx < big_idx
    # Gate must be called at least twice (after swarm flip, after big flip)
    assert len(gate_calls) >= 2


def test_escalate_no_swarm_flip(tmp_path):
    """AC6: --escalate uses read_operator=sonnet, no swarm flip."""
    gaps = [{
        "precondition_id": "gap-esc",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    dowser = _stub_dowser()

    flip_calls: list[str] = []

    def _mock_flip(mode: str) -> bool:
        flip_calls.append(mode)
        return True

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch("lapis_pm.backcaster.quest_leg._rederive_gap"):
            quest_source_run(
                run_id, all_unsourced=True, escalate=True,
                _dowser=dowser, _flip_fn=_mock_flip, _gate_fn=_always_serving,
            )

    assert "swarm" not in flip_calls
    # Escalate path: read uses sonnet, not quest
    read_calls = dowser.read_batch.call_args_list
    assert any("sonnet" in str(c) for c in read_calls)


# ---------------------------------------------------------------------------
# AC6b: Anti-deadlock — serving-gate timeout
# ---------------------------------------------------------------------------

def test_flip_serve_timeout_aborts_read_phase(tmp_path):
    """AC6b: gate timeout after swarm flip -> gaps infra-unavailable, GW left on big."""
    gaps = [{
        "precondition_id": "gap-timeout",
        "what_exists": "x", "what_missing": "y", "what_miswired": "z",
        "citations": [], "unsourced": True,
    }]
    run_dir, run_id = _make_run_dir(tmp_path, gaps)
    dowser = _stub_dowser()

    flip_calls: list[str] = []

    def _mock_flip(mode: str) -> bool:
        flip_calls.append(mode)
        return True

    def _gate_never_serves(timeout_s: int = 240) -> bool:
        return False  # always times out

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = quest_source_run(
            run_id, all_unsourced=True, escalate=False,
            _dowser=dowser, _flip_fn=_mock_flip, _gate_fn=_gate_never_serves,
        )

    # Run must exit cleanly (no exception)
    assert summary["gaps_sourced"] == 0
    assert summary["gaps_honest_null"] == 1

    # Gaps must be marked infra-unavailable (transient, not hopeless)
    sidecar = yaml.safe_load((run_dir / "quest" / "gap-timeout.yaml").read_text())
    assert sidecar["verdict"] == "infra-unavailable"
    assert not _is_known_hopeless(run_dir, "gap-timeout")

    # GW must be left on big (fail toward big invariant)
    # After swarm flip + gate timeout, the leg must flip back to big
    assert flip_calls[-1] == "big"
    assert "swarm" in flip_calls


# ---------------------------------------------------------------------------
# AC8: Full end-to-end with stubbed Dowser (all_unsourced, round-trip)
# ---------------------------------------------------------------------------

def test_end_to_end_all_unsourced(tmp_path):
    """AC8: --all-unsourced against seeded run dir with stubbed Dowser round-trips end to end."""
    gaps = [
        {"precondition_id": "econ-01", "what_exists": "ex", "what_missing": "mi",
         "what_miswired": "mw", "citations": [], "unsourced": True},
        {"precondition_id": "econ-02", "what_exists": "ex2", "what_missing": "mi2",
         "what_miswired": "mw2", "citations": [], "unsourced": True},
        # Already sourced — should not be touched
        {"precondition_id": "econ-03", "what_exists": "ex3", "what_missing": "mi3",
         "what_miswired": "mw3",
         "citations": [{"type": "corpus", "ref": "library/test.md"}], "unsourced": False},
    ]
    run_dir, run_id = _make_run_dir(tmp_path, gaps, epistemic_caution="high")
    dowser = _stub_dowser()

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch("lapis_pm.backcaster.quest_leg._rederive_gap"):
            summary = quest_source_run(
                run_id, all_unsourced=True,
                _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
            )

    assert summary["gaps_targeted"] == 2  # only unsourced
    assert summary["gaps_sourced"] == 2
    assert summary["citations_added"] == 2

    # econ-03 was already sourced; after sourcing econ-01 + econ-02, 0/3 unsourced -> low
    assert summary["caution_after"] == "low"

    # run.yaml updated
    run_data = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert run_data["epistemic_caution"] == "low"
