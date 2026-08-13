"""Tests for lapis-pm-bundle-autodispatch-enforce-v0's morning-brief section
(Design 3 — "the surface Erah's ruling names").

Coverage:
  1. TestReadAutodispatchRendering: reads enforce-outcomes.jsonl, renders
     bound/salvaged/faulted/deferred lines, honors the time window, degrades
     to [] on any read failure or absent file, weekly no-op.
  2. TestFormatBucketSectionsAutodispatch: "Bundle autodispatch enforce"
     bucket special-casing in format_bucket_sections() (daily-only,
     omit-when-empty) — the mirror image of Climate/Locality's weekly-only.
  3. TestReadBucketsIntegration: _read_buckets wiring — a reader exception
     never breaks the overall bucket read.

Patch targets:
  - ROOM_ROOT env var -> hold_shadow.enforce_outcomes_path() resolution
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from lapis_pm import state_brief, state_brief_prompts


def _enforce_record(
    spec="cr-bundle-myrepo-2026-08-12",
    enforce_class="defer",
    action="deferred",
    ground="council blocked: genuine design fork",
    ts_utc=None,
    salvage_dropped_items=None,
):
    return {
        "schema_version": "enforce-outcome/v1",
        "record_id": "test-record-id",
        "ts_utc": ts_utc or "2026-08-13T12:16:00+00:00",
        "run_id": "run-test",
        "spec": spec,
        "enforce_class": enforce_class,
        "enforce_action": action,
        "salvage_dropped_items": salvage_dropped_items or [],
        "verbatim_grounds": ground,
    }


def _write_enforce_outcomes(room_root, records):
    hs_dir = room_root / "hold-shadow"
    hs_dir.mkdir(parents=True, exist_ok=True)
    path = hs_dir / "enforce-outcomes.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n" if records else "",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# _read_autodispatch
# ---------------------------------------------------------------------------

class TestReadAutodispatchRendering:
    def test_no_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        start_ts = datetime(2026, 8, 13, tzinfo=timezone.utc) - timedelta(hours=24)
        assert state_brief._read_autodispatch(start_ts) == []

    def test_weekly_is_always_a_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        _write_enforce_outcomes(tmp_path, [_enforce_record()])
        start_ts = datetime(2026, 8, 6, tzinfo=timezone.utc)
        assert state_brief._read_autodispatch(start_ts, period="weekly") == []

    def test_deferred_record_renders_verbatim_ground(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        _write_enforce_outcomes(tmp_path, [
            _enforce_record(
                spec="cr-bundle-myrepo-2026-08-12",
                enforce_class="defer",
                action="deferred",
                ground="council blocked: genuine design fork",
            ),
        ])
        start_ts = datetime(2026, 8, 6, tzinfo=timezone.utc)
        lines = state_brief._read_autodispatch(start_ts)
        assert len(lines) == 1
        assert "DEFERRED" in lines[0]
        assert "cr-bundle-myrepo-2026-08-12" in lines[0]
        assert "council blocked: genuine design fork" in lines[0]

    def test_infra_fault_renders_faulted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        _write_enforce_outcomes(tmp_path, [
            _enforce_record(
                enforce_class="infra", action="faulted",
                ground="Synthesis error: did not complete within 210s",
            ),
        ])
        start_ts = datetime(2026, 8, 6, tzinfo=timezone.utc)
        lines = state_brief._read_autodispatch(start_ts)
        assert len(lines) == 1
        assert "FAULTED" in lines[0]
        assert "did not complete within 210s" in lines[0]

    def test_salvage_bound_renders_bound_line(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        _write_enforce_outcomes(tmp_path, [
            _enforce_record(enforce_class="salvage", action="salvage_bound"),
        ])
        start_ts = datetime(2026, 8, 6, tzinfo=timezone.utc)
        lines = state_brief._read_autodispatch(start_ts)
        assert len(lines) == 1
        assert "BOUND" in lines[0]
        assert "salvage" in lines[0]

    def test_infra_retry_bound_renders_bound_line(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        _write_enforce_outcomes(tmp_path, [
            _enforce_record(enforce_class="infra", action="retried_then_bound"),
        ])
        start_ts = datetime(2026, 8, 6, tzinfo=timezone.utc)
        lines = state_brief._read_autodispatch(start_ts)
        assert len(lines) == 1
        assert "BOUND" in lines[0]

    def test_salvage_deferred_names_dropped_items_and_pr_comment_owed(self, tmp_path, monkeypatch):
        """Design 2c: the bundle text's source-PR comment obligation is
        surfaced here, not silently dropped."""
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        _write_enforce_outcomes(tmp_path, [
            _enforce_record(
                enforce_class="salvage", action="salvage_regate_deferred",
                salvage_dropped_items=[
                    {"debt_id": "debt-abc123", "confidence": "high",
                     "source": "escalation_reason", "source_pr": 219},
                ],
            ),
        ])
        start_ts = datetime(2026, 8, 6, tzinfo=timezone.utc)
        lines = state_brief._read_autodispatch(start_ts)
        assert len(lines) == 1
        assert "SALVAGED" in lines[0]
        assert "debt-abc123" in lines[0]
        assert "confidence=high" in lines[0]
        assert "source_pr=#219" in lines[0]
        assert "PR comment owed" in lines[0]

    def test_records_outside_window_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        _write_enforce_outcomes(tmp_path, [
            _enforce_record(ts_utc="2026-08-01T00:00:00+00:00"),  # too old
        ])
        start_ts = datetime(2026, 8, 12, tzinfo=timezone.utc)
        assert state_brief._read_autodispatch(start_ts) == []

    def test_torn_line_never_aborts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        hs_dir = tmp_path / "hold-shadow"
        hs_dir.mkdir(parents=True)
        path = hs_dir / "enforce-outcomes.jsonl"
        path.write_text(
            json.dumps(_enforce_record()) + "\n" + "not-json\n", encoding="utf-8",
        )
        start_ts = datetime(2026, 8, 6, tzinfo=timezone.utc)
        lines = state_brief._read_autodispatch(start_ts)
        assert len(lines) == 1

    def test_import_error_degrades_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        with patch.dict("sys.modules", {"lapis_pm.hold_shadow": None}):
            start_ts = datetime(2026, 8, 6, tzinfo=timezone.utc)
            assert state_brief._read_autodispatch(start_ts) == []


# ---------------------------------------------------------------------------
# format_bucket_sections — daily-only, mirroring Climate/Locality in reverse
# ---------------------------------------------------------------------------

class TestFormatBucketSectionsAutodispatch:
    def _buckets(self, items=None):
        return {
            "Built": [],
            "Notable ratifications": [],
            "In flight": [],
            "Captured — not yet built": [],
            "Awaiting your call": [],
            "Gardener Cross-Cutting Observations": [],
            "Bundle autodispatch enforce": items or [],
        }

    def test_daily_renders_when_present(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["DEFERRED: cr-bundle-x — council blocked"]),
            "2026-08-13 08:00 PT", period="daily",
        )
        assert "## Bundle autodispatch enforce" in sections
        assert "council blocked" in sections

    def test_daily_omits_when_empty(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets([]), "2026-08-13 08:00 PT", period="daily",
        )
        assert "Bundle autodispatch enforce" not in sections

    def test_weekly_omits_entirely_even_when_present(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["DEFERRED: cr-bundle-x — council blocked"]),
            "2026-08-13 08:00 PT", period="weekly",
        )
        assert "Bundle autodispatch enforce" not in sections


# ---------------------------------------------------------------------------
# _read_buckets integration — a reader exception never breaks the brief
# ---------------------------------------------------------------------------

class TestReadBucketsIntegration:
    def test_autodispatch_reader_exception_does_not_break_read_buckets(self, monkeypatch):
        with (
            patch.object(state_brief, "_mem") as mock_mem,
            patch.object(state_brief, "_arc_docs_since", return_value=[]),
            patch.object(state_brief, "_read_gardener_observations", return_value=[]),
            patch.object(state_brief, "_read_arc_climate", return_value=[]),
            patch.object(state_brief, "_read_locality", return_value=[]),
            patch.object(state_brief, "_read_autodispatch", side_effect=RuntimeError("boom")),
        ):
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.return_value = []
            start_ts = datetime(2026, 8, 12, tzinfo=timezone.utc)
            buckets = state_brief._read_buckets(start_ts, period="morning")

        assert buckets[state_brief.B_AUTODISPATCH] == []
