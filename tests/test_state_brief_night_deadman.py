"""Tests for night-deadman-floor-v0's morning-brief section (Leg 1).

Coverage:
  1. TestReadNightDeadman: reads /data/slots/night-deadman/*.json, renders
     ok-artifact / finding lines, records ok-artifact absence (I3 dead-
     deadman liveness proof), honors the time window, degrades to [] on
     any read failure or absent directory, weekly no-op.
  2. TestFormatBucketSectionsNightDeadman: "Night dead-man" bucket
     special-casing in format_bucket_sections() (daily-only,
     omit-when-empty) — the mirror image of Climate/Locality's weekly-only.
  3. TestReadBucketsIntegration: _read_buckets wiring — a reader exception
     never breaks the overall bucket read.

Patch targets:
  - _NIGHT_DEADMAN_DIR monkeypatched to a tmp_path
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from lapis_pm import state_brief, state_brief_prompts


def _write_deadman_artifact(dir_path: Path, name: str, data: dict) -> Path:
    """Write a dead-man artifact JSON file to the given directory."""
    path = dir_path / name
    dir_path.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _ok_artifact(ts_utc: str | None = None) -> dict:
    return {
        "verdict": "ok",
        "ts_utc": ts_utc or "2026-09-15T08:15:00+00:00",
    }


def _finding_artifact(ts_utc: str | None = None, summary: str = "never-started") -> dict:
    return {
        "verdict": "finding",
        "severity": "HIGH",
        "ts_utc": ts_utc or "2026-09-15T08:15:00+00:00",
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# _read_night_deadman
# ---------------------------------------------------------------------------

class TestReadNightDeadman:
    def test_no_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state_brief, "_NIGHT_DEADMAN_DIR", tmp_path / "nonexistent")
        start_ts = datetime(2026, 9, 15, tzinfo=timezone.utc) - timedelta(hours=24)
        assert state_brief._read_night_deadman(start_ts) == []

    def test_weekly_is_always_a_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state_brief, "_NIGHT_DEADMAN_DIR", tmp_path)
        _write_deadman_artifact(tmp_path, "ok-20260915T081500Z.json", _ok_artifact())
        start_ts = datetime(2026, 9, 8, tzinfo=timezone.utc)
        assert state_brief._read_night_deadman(start_ts, period="weekly") == []

    def test_ok_artifact_renders_ok_line(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state_brief, "_NIGHT_DEADMAN_DIR", tmp_path)
        _write_deadman_artifact(tmp_path, "ok-20260915T081500Z.json", _ok_artifact())
        start_ts = datetime(2026, 9, 14, tzinfo=timezone.utc)
        lines = state_brief._read_night_deadman(start_ts)
        assert len(lines) == 1
        assert "[OK]" in lines[0]
        assert "dead-man all-clear" in lines[0]

    def test_finding_artifact_renders_finding_line(self, tmp_path, monkeypatch):
        """A finding without an ok-artifact also triggers the CRITICAL line
        (I3: the ok-artifact absence is the timer-death / service-crash
        detector). The finding line is still present."""
        monkeypatch.setattr(state_brief, "_NIGHT_DEADMAN_DIR", tmp_path)
        _write_deadman_artifact(tmp_path, "20260915T081500Z-findings.json", _finding_artifact())
        start_ts = datetime(2026, 9, 14, tzinfo=timezone.utc)
        lines = state_brief._read_night_deadman(start_ts)
        assert len(lines) == 2
        assert "[CRITICAL]" in lines[0]
        assert "[FINDING]" in lines[1]
        assert "HIGH" in lines[1]
        assert "never-started" in lines[1]

    def test_no_ok_artifact_renders_critical(self, tmp_path, monkeypatch):
        """I3: the ok-artifact absence is the timer-death / service-crash
        detector. If no ok-artifact was found in the window, the dead-man
        itself may be dead."""
        monkeypatch.setattr(state_brief, "_NIGHT_DEADMAN_DIR", tmp_path)
        # Only a finding, no ok-artifact
        _write_deadman_artifact(tmp_path, "20260915T081500Z-findings.json", _finding_artifact())
        start_ts = datetime(2026, 9, 14, tzinfo=timezone.utc)
        lines = state_brief._read_night_deadman(start_ts)
        assert len(lines) == 2
        assert "[CRITICAL]" in lines[0]
        assert "no ok-artifact" in lines[0]
        assert "[FINDING]" in lines[1]

    def test_empty_dir_renders_critical(self, tmp_path, monkeypatch):
        """An empty directory (no artifacts at all) means the dead-man never
        ran — the ok-artifact absence is the timer-death detector."""
        monkeypatch.setattr(state_brief, "_NIGHT_DEADMAN_DIR", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        start_ts = datetime(2026, 9, 14, tzinfo=timezone.utc)
        lines = state_brief._read_night_deadman(start_ts)
        assert len(lines) == 1
        assert "[CRITICAL]" in lines[0]
        assert "no ok-artifact" in lines[0]

    def test_records_outside_window_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state_brief, "_NIGHT_DEADMAN_DIR", tmp_path)
        # Write an artifact with an old timestamp
        old_artifact = _ok_artifact(ts_utc="2026-09-01T00:00:00+00:00")
        _write_deadman_artifact(tmp_path, "ok-20260901T000000Z.json", old_artifact)
        # Set the file mtime to be old
        import os
        old_time = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()
        os.utime(tmp_path / "ok-20260901T000000Z.json", (old_time, old_time))
        start_ts = datetime(2026, 9, 14, tzinfo=timezone.utc)
        lines = state_brief._read_night_deadman(start_ts)
        # The old artifact is excluded; no ok-artifact in window -> CRITICAL
        assert len(lines) == 1
        assert "[CRITICAL]" in lines[0]

    def test_torn_json_never_aborts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state_brief, "_NIGHT_DEADMAN_DIR", tmp_path)
        (tmp_path / "torn.json").write_text("not-json", encoding="utf-8")
        _write_deadman_artifact(tmp_path, "ok-20260915T081500Z.json", _ok_artifact())
        start_ts = datetime(2026, 9, 14, tzinfo=timezone.utc)
        lines = state_brief._read_night_deadman(start_ts)
        assert len(lines) == 1
        assert "[OK]" in lines[0]

    def test_mixed_ok_and_findings(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state_brief, "_NIGHT_DEADMAN_DIR", tmp_path)
        _write_deadman_artifact(tmp_path, "ok-20260915T081500Z.json", _ok_artifact())
        _write_deadman_artifact(tmp_path, "20260915T011500Z-findings.json", _finding_artifact())
        start_ts = datetime(2026, 9, 14, tzinfo=timezone.utc)
        lines = state_brief._read_night_deadman(start_ts)
        assert len(lines) == 2
        # ok-artifact present -> no CRITICAL
        assert not any("[CRITICAL]" in line for line in lines)
        assert any("[OK]" in line for line in lines)
        assert any("[FINDING]" in line for line in lines)


# ---------------------------------------------------------------------------
# format_bucket_sections — daily-only, mirroring Climate/Locality in reverse
# ---------------------------------------------------------------------------

class TestFormatBucketSectionsNightDeadman:
    def _buckets(self, items=None):
        return {
            "Built": [],
            "Notable ratifications": [],
            "In flight": [],
            "Captured — not yet built": [],
            "Awaiting your call": [],
            "Gardener Cross-Cutting Observations": [],
            "Night dead-man": items or [],
        }

    def test_daily_renders_when_present(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["[OK] dead-man all-clear 2026-09-15T08:15:00+00:00 — ok"]),
            "2026-09-15 08:00 PT", period="daily",
        )
        assert "## Night dead-man" in sections
        assert "dead-man all-clear" in sections

    def test_daily_omits_when_empty(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets([]), "2026-09-15 08:00 PT", period="daily",
        )
        assert "Night dead-man" not in sections

    def test_weekly_omits_entirely_even_when_present(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["[OK] dead-man all-clear"]),
            "2026-09-15 08:00 PT", period="weekly",
        )
        assert "Night dead-man" not in sections


# ---------------------------------------------------------------------------
# _read_buckets integration — a reader exception never breaks the brief
# ---------------------------------------------------------------------------

class TestReadBucketsIntegration:
    def test_night_deadman_reader_exception_does_not_break_read_buckets(self, monkeypatch):
        with (
            patch.object(state_brief, "_mem") as mock_mem,
            patch.object(state_brief, "_arc_docs_since", return_value=[]),
            patch.object(state_brief, "_read_gardener_observations", return_value=[]),
            patch.object(state_brief, "_read_arc_climate", return_value=[]),
            patch.object(state_brief, "_read_locality", return_value=[]),
            patch.object(state_brief, "_read_autodispatch", return_value=[]),
            patch.object(state_brief, "_read_night_deadman", side_effect=RuntimeError("boom")),
        ):
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.return_value = []
            start_ts = datetime(2026, 9, 14, tzinfo=timezone.utc)
            buckets = state_brief._read_buckets(start_ts, period="morning")

        assert buckets[state_brief.B_NIGHT_DEADMAN] == []
