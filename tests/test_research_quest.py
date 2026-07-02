"""Tests: research-quest-nightly-producer-v0 acceptance criteria (AC5).

All tests mock the AC1 shared helpers (lapis_pm.gw_flip_gate) and the Dowser
client. No live GW / StarHouse / SearXNG / network dependency.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lapis_pm import research_quest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _write_backlog(tmp_path: Path, entries: list[dict]) -> Path:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    path = queue_dir / "dowser-backlog.yaml"
    path.write_text(yaml.dump({"entries": entries}, default_flow_style=False, sort_keys=False))
    return path


def _entry(id_: str, status: str = "pending", attempts: int = 0, intent: str | None = None) -> dict:
    return {
        "id": id_,
        "intent": intent or f"What is the state of {id_}?",
        "status": status,
        "attempts": attempts,
        "added": "2026-07-01",
    }


def _stub_dowser(
    read_outcome: str = "sources-found",
    read_citations: list[dict] | None = None,
    critique_status: str = "pass",
) -> MagicMock:
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
                "findings": "Relevant findings from the web." if read_outcome == "sources-found" else "",
                "citations": read_citations if read_outcome == "sources-found" else [],
                "outcome": read_outcome,
                "provenance": {"search_strings": [req["intent"][:50]], "hits_count": 5},
            })
        return {"drafts": drafts}

    def _critique_batch(drafts, critic_operator="gravitywell"):
        verdicts = []
        for draft in drafts:
            verdicts.append({
                "intent": draft["intent"],
                "status": critique_status,
                "verdict": {"relevance": 8, "credibility": 8, "faithfulness": 8, "confidence": "high"},
                "diagnosis": "",
            })
        return {"verdicts": verdicts}

    dowser.read_batch = MagicMock(side_effect=_read_batch)
    dowser.critique_batch = MagicMock(side_effect=_critique_batch)
    return dowser


def _always_ok(mode: str) -> bool:
    return True


def _always_serving(timeout_s: int = 240) -> bool:
    return True


# ---------------------------------------------------------------------------
# Backlog selection: --limit and status: pending filtering
# ---------------------------------------------------------------------------

def test_backlog_pop_honors_limit_and_pending_filter(tmp_path):
    entries = [
        _entry("a"), _entry("b"), _entry("c"),
        _entry("d", status="done"), _entry("e", status="no-credible-sources"),
    ]
    _write_backlog(tmp_path, entries)
    dowser = _stub_dowser()

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = research_quest.run_research_quest(
            limit=2, _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    assert summary["entries_targeted"] == 2
    # Only the first 2 pending entries (a, b) were read; c/d/e untouched
    read_intents = [c.args[0][0]["intent"] for c in dowser.read_batch.call_args_list]
    assert read_intents == [entries[0]["intent"], entries[1]["intent"]]


# ---------------------------------------------------------------------------
# sources-found writes the dated file and flips the entry to done
# ---------------------------------------------------------------------------

def test_sources_found_writes_file_and_marks_done(tmp_path):
    entries = [_entry("gap-01", intent="What is the sub-linear fee scaling landscape?")]
    backlog_path = _write_backlog(tmp_path, entries)
    dowser = _stub_dowser()

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = research_quest.run_research_quest(
            limit=5, _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    assert summary["entries_sourced"] == 1
    assert summary["entries_honest_null"] == 0

    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text()
    assert "What is the sub-linear fee scaling landscape?" in content
    assert "## Sources" in content
    assert "arxiv.org" in content

    updated = yaml.safe_load(backlog_path.read_text())
    assert updated["entries"][0]["status"] == "done"


# ---------------------------------------------------------------------------
# Attempts / retry-cap-then-permanent-null: 0->1->2, pending->pending->no-credible-sources
# ---------------------------------------------------------------------------

def test_attempts_retry_cap_then_permanent_null(tmp_path):
    entries = [_entry("gap-fail")]
    backlog_path = _write_backlog(tmp_path, entries)
    dowser = _stub_dowser(read_outcome="no-credible-sources", read_citations=[], critique_status="subpar")

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        # Run 1: attempts 0 -> 1, stays pending
        research_quest.run_research_quest(
            limit=5, _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )
    updated = yaml.safe_load(backlog_path.read_text())
    assert updated["entries"][0]["attempts"] == 1
    assert updated["entries"][0]["status"] == "pending"

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        # Run 2: attempts 1 -> 2, becomes permanent null
        research_quest.run_research_quest(
            limit=5, _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )
    updated = yaml.safe_load(backlog_path.read_text())
    assert updated["entries"][0]["attempts"] == 2
    assert updated["entries"][0]["status"] == "no-credible-sources"

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        # Run 3: entry no longer pending, must not be targeted
        summary = research_quest.run_research_quest(
            limit=5, _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )
    assert summary["entries_targeted"] == 0


# ---------------------------------------------------------------------------
# One entry's exception doesn't block the next
# ---------------------------------------------------------------------------

def test_one_entry_exception_does_not_block_next(tmp_path):
    entries = [_entry("gap-boom"), _entry("gap-ok")]
    backlog_path = _write_backlog(tmp_path, entries)

    dowser = MagicMock()

    def _read_batch(requests_list, read_operator="quest", budget=None):
        if "gap-boom" in requests_list[0]["intent"]:
            raise RuntimeError("boom")
        return {"drafts": [{
            "intent": requests_list[0]["intent"],
            "findings": "ok findings",
            "citations": [{"url": "https://arxiv.org/abs/9", "title": "T",
                            "excerpt": "e", "credibility": "high"}],
            "outcome": "sources-found",
            "provenance": {},
        }]}

    dowser.read_batch = MagicMock(side_effect=_read_batch)
    dowser.critique_batch = MagicMock(return_value={"verdicts": [{
        "intent": "x", "status": "pass",
        "verdict": {"relevance": 9, "credibility": 9, "faithfulness": 9, "confidence": "high"},
        "diagnosis": "",
    }]})

    entries[0]["intent"] = "gap-boom intent"
    entries[1]["intent"] = "gap-ok intent"
    _write_backlog(tmp_path, entries)

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = research_quest.run_research_quest(
            limit=5, _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
        )

    assert summary["entries_targeted"] == 2
    assert summary["entries_sourced"] == 1
    assert summary["entries_honest_null"] == 1

    updated = yaml.safe_load(backlog_path.read_text())
    by_id = {e["id"]: e for e in updated["entries"]}
    assert by_id["gap-boom"]["attempts"] == 1
    assert by_id["gap-boom"]["status"] == "pending"
    assert by_id["gap-ok"]["status"] == "done"


# ---------------------------------------------------------------------------
# Empty/all-non-pending backlog: WARN + exit 0 + zero flip calls
# ---------------------------------------------------------------------------

def test_empty_backlog_warns_and_makes_zero_flip_calls(tmp_path):
    _write_backlog(tmp_path, [])
    flip_fn = MagicMock(return_value=True)
    gate_fn = MagicMock(return_value=True)
    dowser = _stub_dowser()

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch.object(research_quest.log, "warning") as mock_warn:
            summary = research_quest.run_research_quest(
                limit=5, _dowser=dowser, _flip_fn=flip_fn, _gate_fn=gate_fn,
            )

    assert summary["entries_targeted"] == 0
    assert any("no pending backlog entries" in str(c.args) for c in mock_warn.call_args_list)
    flip_fn.assert_not_called()
    gate_fn.assert_not_called()
    dowser.read_batch.assert_not_called()
    dowser.critique_batch.assert_not_called()


def test_all_non_pending_backlog_warns_and_makes_zero_flip_calls(tmp_path):
    _write_backlog(tmp_path, [_entry("a", status="done"), _entry("b", status="no-credible-sources")])
    flip_fn = MagicMock(return_value=True)
    gate_fn = MagicMock(return_value=True)
    dowser = _stub_dowser()

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch.object(research_quest.log, "warning") as mock_warn:
            summary = research_quest.run_research_quest(
                limit=5, _dowser=dowser, _flip_fn=flip_fn, _gate_fn=gate_fn,
            )

    assert summary["entries_targeted"] == 0
    assert any("no pending backlog entries" in str(c.args) for c in mock_warn.call_args_list)
    flip_fn.assert_not_called()
    gate_fn.assert_not_called()


# ---------------------------------------------------------------------------
# Missing backlog file entirely (never seeded) behaves like empty
# ---------------------------------------------------------------------------

def test_missing_backlog_file_behaves_like_empty(tmp_path):
    flip_fn = MagicMock(return_value=True)
    dowser = _stub_dowser()

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = research_quest.run_research_quest(
            limit=5, _dowser=dowser, _flip_fn=flip_fn, _gate_fn=_always_serving,
        )

    assert summary["entries_targeted"] == 0
    flip_fn.assert_not_called()


# ---------------------------------------------------------------------------
# --dry-run: selects but does not touch GW/Dowser or mutate the backlog
# ---------------------------------------------------------------------------

def test_dry_run_does_not_touch_gw_or_backlog(tmp_path):
    entries = [_entry("gap-dry")]
    backlog_path = _write_backlog(tmp_path, entries)
    flip_fn = MagicMock(return_value=True)
    gate_fn = MagicMock(return_value=True)
    dowser = _stub_dowser()

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        summary = research_quest.run_research_quest(
            limit=5, dry_run=True, _dowser=dowser, _flip_fn=flip_fn, _gate_fn=gate_fn,
        )

    assert summary["entries_targeted"] == 1
    flip_fn.assert_not_called()
    dowser.read_batch.assert_not_called()

    # Backlog untouched
    unchanged = yaml.safe_load(backlog_path.read_text())
    assert unchanged["entries"][0]["status"] == "pending"
    assert unchanged["entries"][0]["attempts"] == 0


# ---------------------------------------------------------------------------
# AC3 calls into AC1's extracted functions, not new flip/gate code
# ---------------------------------------------------------------------------

def test_uses_shared_gw_flip_gate_phase_functions(tmp_path):
    entries = [_entry("gap-shared")]
    _write_backlog(tmp_path, entries)
    dowser = _stub_dowser()

    with patch("agents_core.room_paths.room_path", return_value=str(tmp_path)):
        with patch(
            "lapis_pm.research_quest.gw_flip_gate.phase_swarm_read",
            wraps=research_quest.gw_flip_gate.phase_swarm_read,
        ) as mock_swarm_read, patch(
            "lapis_pm.research_quest.gw_flip_gate.phase_big_critique",
            wraps=research_quest.gw_flip_gate.phase_big_critique,
        ) as mock_big_critique:
            research_quest.run_research_quest(
                limit=5, _dowser=dowser, _flip_fn=_always_ok, _gate_fn=_always_serving,
            )

    mock_swarm_read.assert_called_once()
    mock_big_critique.assert_called_once()
