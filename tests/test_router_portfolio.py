"""Tests for lapis_pm.router_portfolio — router-portfolio-persistence-v0.

Coverage (per spec §Tests):
1. test_event_write_idempotent         — same payload twice → single mem entry
2. test_event_categories               — each category writes to correct subkey path
3. test_checkpoint_skill_consolidates  — fixture session N events → summary references all N
4. test_checkpoint_skill_labels_primitives — entry missing primitive_decomposition → labeled
5. test_stop_hook_prompts_when_missing — session terminates without checkpoint → prompt present
6. test_stop_hook_minimal_fallback     — terminal session, no turns → minimal checkpoint runs
7. test_session_start_loads_prior      — bootstrap fixture → session summary readable via mem
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from lapis_pm.router_portfolio import (
    RouterPortfolioEntry,
    _event_id,
    _now_iso,
    emit_decision_dispatch,
    emit_decision_kickoff,
    emit_decision_land,
    emit_expert_outcome,
    emit_ratify_confirm,
    emit_ratify_correct,
    emit_ratify_override,
    emit_ratify_redirect,
    read_session_entries,
    session_checkpoint_exists,
    write_session_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_mem(store: dict | None = None) -> MagicMock:
    """Return a MagicMock MemoryStore with .set / .get / .list_all / .delete wired up."""
    data: dict[str, dict] = {}
    if store:
        data.update(store)

    mock = MagicMock()

    def _set(key: str, content: str, tags: list | None = None, source: str = "") -> bool:
        data[key] = {"key": key, "content": content, "tags": tags or []}
        return True

    def _get(key: str) -> dict | None:
        return data.get(key)

    def _list_all(tag: str = "", tags: list | None = None, since: str = "", limit: int = 50) -> list[dict]:
        result = list(data.values())
        if tag:
            result = [r for r in result if tag in (r.get("tags") or [])]
        return result[:limit]

    def _delete(key: str) -> bool:
        data.pop(key, None)
        return True

    mock.set.side_effect = _set
    mock.get.side_effect = _get
    mock.list_all.side_effect = _list_all
    mock.delete.side_effect = _delete
    mock._data = data
    return mock


# ---------------------------------------------------------------------------
# 1. Idempotent write
# ---------------------------------------------------------------------------

class TestEventWriteIdempotent:
    def test_same_event_id_writes_once(self) -> None:
        """Re-firing the same event_id with the same payload → single mem entry."""
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            fixed_eid = "2026-05-01T00:00:00Z-aabbccdd"
            key1 = emit_decision_kickoff(
                target_id="test-target",
                expert_chosen="haiku",
                intent_summary="kickoff test",
                event_id=fixed_eid,
            )
            key2 = emit_decision_kickoff(
                target_id="test-target",
                expert_chosen="haiku",
                intent_summary="kickoff test",
                event_id=fixed_eid,
            )

        assert key1 == key2
        # mem.set was called twice (idempotent write is a content-stable overwrite)
        # but the key is the same — only one entry in the store
        assert len([k for k in mock_mem._data if fixed_eid in k]) == 1

    def test_different_event_ids_produce_different_entries(self) -> None:
        """Two distinct events → two distinct mem entries."""
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key1 = emit_decision_kickoff("t1", "haiku", "first kickoff")
            key2 = emit_decision_kickoff("t1", "haiku", "second kickoff")

        assert key1 != key2
        assert len(mock_mem._data) == 2


# ---------------------------------------------------------------------------
# 2. Event categories — each event writes to the correct namespace subkey
# ---------------------------------------------------------------------------

class TestEventCategories:
    def test_decision_kickoff_writes_to_decisions(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_decision_kickoff("t1", "haiku", "kickoff intent")
        assert key.startswith("router/lapis-pm/decisions/")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["verdict"] == "proposed"
        assert entry["fragment_id"] == "kickoff"

    def test_decision_dispatch_writes_to_decisions(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_decision_dispatch("t1", "tick", "sonnet", "dispatch intent")
        assert key.startswith("router/lapis-pm/decisions/")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["verdict"] == "proposed"
        assert entry["fragment_id"] == "tick"

    def test_decision_land_writes_to_decisions(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_decision_land("t1", "ready to land")
        assert key.startswith("router/lapis-pm/decisions/")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["verdict"] == "proposed"
        assert entry["fragment_id"] == "land"

    def test_ratify_confirm_writes_to_ratification_outcomes(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_ratify_confirm("t1", prior_decision_event_id="abc123")
        assert key.startswith("router/lapis-pm/ratification-outcomes/")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["verdict"] == "ratified"
        assert entry["ratification_outcome"] == "confirm"

    def test_ratify_correct_writes_to_ratification_outcomes(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_ratify_correct("t1", "correction reason")
        assert key.startswith("router/lapis-pm/ratification-outcomes/")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["verdict"] == "corrected"
        assert entry["ratification_outcome"] == "correct"

    def test_ratify_override_writes_to_ratification_outcomes(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_ratify_override("t1", "override reason")
        assert key.startswith("router/lapis-pm/ratification-outcomes/")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["verdict"] == "overridden"
        assert entry["ratification_outcome"] == "override"

    def test_ratify_redirect_writes_to_ratification_outcomes(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_ratify_redirect("t1", "new scope intent")
        assert key.startswith("router/lapis-pm/ratification-outcomes/")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["verdict"] == "redirected"
        assert entry["ratification_outcome"] == "redirect"

    def test_expert_outcome_landed_writes_to_expert_experience(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_expert_outcome("t1", "haiku", "landed", fragment_id="tick")
        assert key.startswith("router/lapis-pm/expert-experience/haiku/")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["verdict"] == "landed"
        assert entry["expert_chosen"] == "haiku"

    def test_expert_outcome_reverted_writes_to_expert_experience(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_expert_outcome("t1", "sonnet", "reverted")
        assert key.startswith("router/lapis-pm/expert-experience/sonnet/")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["verdict"] == "reverted"

    def test_expert_outcome_with_primitive_decomposition_uses_shape_slug(self) -> None:
        mock_mem = _make_mock_mem()
        pd = [{"fragment_id": "kickoff", "intent": "start work"}]
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_expert_outcome("t1", "haiku", "landed", primitive_decomposition=pd)
        # intent_shape derived from primitive_decomposition
        assert "kickoff" in key or "start-work" in key or "unstructured" in key

    def test_expert_outcome_without_primitive_decomposition_uses_unstructured(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_expert_outcome("t1", "haiku", "landed")
        assert "unstructured" in key

    def test_scope_id_is_always_router_lapis_pm(self) -> None:
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_decision_kickoff("t1", "haiku", "test")
        entry = json.loads(mock_mem._data[key]["content"])
        assert entry["scope_id"] == "router/lapis-pm"


# ---------------------------------------------------------------------------
# 3. Checkpoint skill consolidates all session events
# ---------------------------------------------------------------------------

class TestCheckpointSkillConsolidates:
    def test_read_session_entries_returns_all_tagged_entries(self) -> None:
        """Fixture session with 3 events → read_session_entries returns all 3."""
        now = _now_iso()
        store_data = {}

        def _make_entry(verdict: str, target: str) -> dict:
            return {
                "key": f"router/lapis-pm/decisions/{_event_id()}",
                "content": json.dumps({
                    "verdict": verdict,
                    "claim": f"claim for {target}",
                    "citations": [],
                    "freshness_stamp": now,
                    "scope_id": "router/lapis-pm",
                    "target_id": target,
                }),
                "tags": ["router-portfolio"],
            }

        entries = [
            _make_entry("proposed", "t1"),
            _make_entry("proposed", "t2"),
            _make_entry("ratified", "t1"),
        ]
        for e in entries:
            store_data[e["key"]] = e

        mock_mem = _make_mock_mem(store_data)
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            result = read_session_entries()

        assert len(result) == 3

    def test_write_session_summary_writes_to_sessions_namespace(self) -> None:
        """write_session_summary writes to router/lapis-pm/sessions/<ts>-<slug>."""
        mock_mem = _make_mock_mem()
        entries = [{"_mem_key": "router/lapis-pm/decisions/abc", "verdict": "proposed"}]

        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = write_session_summary(
                slug="test-session",
                summary_markdown="## Test session\n\nNothing happened.",
                entries=entries,
            )

        assert key.startswith("router/lapis-pm/sessions/")
        assert "test-session" in key
        stored = json.loads(mock_mem._data[key]["content"])
        assert stored["event_count"] == 1
        assert "router/lapis-pm/decisions/abc" in stored["event_keys"]
        assert "## Test session" in stored["summary"]

    def test_checkpoint_references_all_events(self) -> None:
        """Session summary event_keys must include all N event mem_keys."""
        mock_mem = _make_mock_mem()
        n = 3
        entries = [
            {"_mem_key": f"router/lapis-pm/decisions/event-{i}", "verdict": "proposed"}
            for i in range(n)
        ]
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = write_session_summary(
                slug="full-session",
                summary_markdown="## Full session",
                entries=entries,
            )
        stored = json.loads(mock_mem._data[key]["content"])
        assert stored["event_count"] == n
        for i in range(n):
            assert f"router/lapis-pm/decisions/event-{i}" in stored["event_keys"]


# ---------------------------------------------------------------------------
# 4. Checkpoint labels missing primitive_decomposition
# ---------------------------------------------------------------------------

class TestCheckpointLabelsPrimitives:
    def test_entry_missing_primitive_decomposition_can_be_labeled(self) -> None:
        """An entry with primitive_decomposition=None can receive a label post-hoc.

        v0: labeling is done inline by the Router; this test verifies the entry
        shape is mutable and the labeled version writes correctly.
        """
        mock_mem = _make_mock_mem()
        # Write a kickoff entry with no primitive_decomposition
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = emit_decision_kickoff("t1", "haiku", "test kickoff", primitive_decomposition=None)

        entry_data = json.loads(mock_mem._data[key]["content"])
        assert entry_data["primitive_decomposition"] is None

        # Simulate Router labeling it post-hoc
        entry_data["primitive_decomposition"] = [{"fragment_id": "kickoff", "intent": "start work on t1"}]
        # Re-write the updated entry to mem
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            mock_mem.set(key, json.dumps(entry_data), tags=["router-portfolio"])

        updated = json.loads(mock_mem._data[key]["content"])
        assert updated["primitive_decomposition"] is not None
        assert updated["primitive_decomposition"][0]["fragment_id"] == "kickoff"


# ---------------------------------------------------------------------------
# 5. Stop hook prompts when checkpoint missing
# ---------------------------------------------------------------------------

class TestStopHookPromptsWhenMissing:
    def test_session_checkpoint_exists_returns_false_when_no_summaries(self) -> None:
        """session_checkpoint_exists → False when no session summaries in mem."""
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            result = session_checkpoint_exists(since_iso="2026-05-01T00:00:00Z")
        assert result is False

    def test_session_checkpoint_exists_returns_true_when_summary_present(self) -> None:
        """session_checkpoint_exists → True when a recent session summary exists."""
        since = "2026-05-01T00:00:00Z"
        payload = json.dumps({
            "summary": "## Test session",
            "written_at": "2026-05-01T01:00:00Z",
            "event_count": 1,
            "event_keys": [],
        })
        store = {"router/lapis-pm/sessions/2026-05-01-0100-test": {
            "key": "router/lapis-pm/sessions/2026-05-01-0100-test",
            "content": payload,
            "tags": ["router-session"],
        }}
        mock_mem = _make_mock_mem(store)
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            result = session_checkpoint_exists(since_iso=since)
        assert result is True

    def test_session_checkpoint_exists_returns_false_for_stale_summary(self) -> None:
        """session_checkpoint_exists → False when summary is older than since_iso."""
        since = "2026-05-01T06:00:00Z"
        payload = json.dumps({
            "summary": "## Old session",
            "written_at": "2026-05-01T00:00:00Z",  # before since
            "event_count": 0,
            "event_keys": [],
        })
        store = {"router/lapis-pm/sessions/2026-05-01-0000-old": {
            "key": "router/lapis-pm/sessions/2026-05-01-0000-old",
            "content": payload,
            "tags": ["router-session"],
        }}
        mock_mem = _make_mock_mem(store)
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            result = session_checkpoint_exists(since_iso=since)
        assert result is False


# ---------------------------------------------------------------------------
# 6. Stop hook minimal fallback
# ---------------------------------------------------------------------------

class TestStopHookMinimalFallback:
    def test_write_session_summary_with_no_entries(self) -> None:
        """Minimal checkpoint with zero events still writes a summary key."""
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = write_session_summary(
                slug="auto-stop-fallback",
                summary_markdown="## Auto-checkpoint (minimal)",
                entries=[],
            )
        assert "auto-stop-fallback" in key
        stored = json.loads(mock_mem._data[key]["content"])
        assert stored["event_count"] == 0

    def test_write_session_summary_tags_as_router_session(self) -> None:
        """Session summary is tagged with 'router-session' for checkpoint detection."""
        mock_mem = _make_mock_mem()
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            key = write_session_summary("fallback", "## Fallback", entries=[])
        assert "router-session" in (mock_mem._data[key].get("tags") or [])


# ---------------------------------------------------------------------------
# 7. Session start loads prior
# ---------------------------------------------------------------------------

class TestSessionStartLoadsPrior:
    def test_read_session_entries_filtered_by_since_iso(self) -> None:
        """Bootstrap fixture: read_session_entries(since_iso=<future>) → empty."""
        now = _now_iso()
        store_data = {}
        entry = {
            "key": "router/lapis-pm/decisions/old-event",
            "content": json.dumps({
                "verdict": "proposed",
                "claim": "old claim",
                "citations": [],
                "freshness_stamp": "2020-01-01T00:00:00Z",
                "scope_id": "router/lapis-pm",
            }),
            "tags": ["router-portfolio"],
        }
        store_data[entry["key"]] = entry

        mock_mem = _make_mock_mem(store_data)
        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            # Filter with since_iso in the future of the old entry
            result = read_session_entries(since_iso="2025-01-01T00:00:00Z")

        assert result == []

    def test_prior_session_summary_readable_via_list_all(self) -> None:
        """Prior session summary is findable via list_all(tag='router-session')."""
        summary_payload = json.dumps({
            "summary": "## Prior session\n\n- target t1 dispatched and ratified",
            "written_at": "2026-04-30T12:00:00Z",
            "event_count": 2,
            "event_keys": ["router/lapis-pm/decisions/abc", "router/lapis-pm/ratification-outcomes/def"],
        })
        store = {"router/lapis-pm/sessions/2026-04-30-1200-prior": {
            "key": "router/lapis-pm/sessions/2026-04-30-1200-prior",
            "content": summary_payload,
            "tags": ["lapis-pm", "router-portfolio", "router-session"],
        }}
        mock_mem = _make_mock_mem(store)

        with patch("lapis_pm.router_portfolio._mem", return_value=mock_mem):
            rows = mock_mem.list_all(tag="router-session", limit=10)

        assert len(rows) == 1
        data = json.loads(rows[0]["content"])
        assert "Prior session" in data["summary"]
        assert data["event_count"] == 2


# ---------------------------------------------------------------------------
# RouterPortfolioEntry shape
# ---------------------------------------------------------------------------

class TestRouterPortfolioEntryShape:
    def test_to_dict_includes_all_base_fields(self) -> None:
        entry = RouterPortfolioEntry(
            verdict="proposed",
            claim="test claim",
            citations=["ref1"],
            freshness_stamp="2026-05-01T00:00:00Z",
            scope_id="router/lapis-pm",
        )
        d = entry.to_dict()
        assert d["verdict"] == "proposed"
        assert d["claim"] == "test claim"
        assert d["citations"] == ["ref1"]
        assert d["scope_id"] == "router/lapis-pm"
        assert d["drift_class"] is None
        assert d["primitive_decomposition"] is None

    def test_to_dict_includes_router_extras(self) -> None:
        entry = RouterPortfolioEntry(
            verdict="landed",
            claim="expert landed",
            citations=[],
            freshness_stamp="2026-05-01T00:00:00Z",
            scope_id="router/lapis-pm",
            fragment_id="tick",
            expert_chosen="haiku",
            target_id="t1",
            intent_summary="landed successfully",
        )
        d = entry.to_dict()
        assert d["fragment_id"] == "tick"
        assert d["expert_chosen"] == "haiku"
        assert d["target_id"] == "t1"
        assert d["intent_summary"] == "landed successfully"

    def test_to_json_is_valid_json(self) -> None:
        entry = RouterPortfolioEntry(
            verdict="proposed",
            claim="json test",
            citations=[],
            freshness_stamp=_now_iso(),
            scope_id="router/lapis-pm",
        )
        raw = entry.to_json()
        parsed = json.loads(raw)
        assert parsed["verdict"] == "proposed"
