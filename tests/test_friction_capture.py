"""Tests for operational-learning-friction-capture-v0 (Unit 1a).

Covers: zero-trust sanitization of agent-authored fields, silent-gap capture,
idempotent append-only queue writes, and the `lapis-pm friction list` CLI
read path.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from lapis_pm import pm_core, cli


# ---------------------------------------------------------------------------
# Zero-trust sanitization
# ---------------------------------------------------------------------------

class TestSanitizeFrictionRecord:
    def test_valid_record_passthrough(self):
        raw = {
            "obstacle": "could not find the config file",
            "path_taken": "grepped for the key instead",
            "artifact": "lapis_pm/pm_core.py",
            "cost_hint": "minor",
            "confidence": "likely",
        }
        out = pm_core._sanitize_friction_record(raw)
        assert out == raw

    def test_drops_empty_obstacle(self):
        assert pm_core._sanitize_friction_record({"obstacle": ""}) is None
        assert pm_core._sanitize_friction_record({"obstacle": "   "}) is None
        assert pm_core._sanitize_friction_record({}) is None

    def test_drops_non_string_obstacle(self):
        assert pm_core._sanitize_friction_record({"obstacle": 12345}) is None
        assert pm_core._sanitize_friction_record({"obstacle": None}) is None

    def test_non_dict_record_dropped(self):
        assert pm_core._sanitize_friction_record("not a dict") is None
        assert pm_core._sanitize_friction_record(["also", "not"]) is None

    def test_strips_control_chars(self):
        out = pm_core._sanitize_friction_record({"obstacle": "bad\x00obstacle\x1f here"})
        assert out["obstacle"] == "badobstacle here"

    def test_truncates_over_long_fields(self):
        long_obstacle = "x" * 5000
        out = pm_core._sanitize_friction_record({"obstacle": long_obstacle})
        assert len(out["obstacle"]) == pm_core._FRICTION_MAX_OBSTACLE_LEN

    def test_rejects_unknown_keys(self):
        raw = {
            "obstacle": "something blocked me",
            "malicious_gate_field": "certain",
            "verdict": "auto-merge",
        }
        out = pm_core._sanitize_friction_record(raw)
        assert set(out.keys()) <= {"obstacle", "path_taken", "artifact", "cost_hint", "confidence"}
        assert "malicious_gate_field" not in out
        assert "verdict" not in out

    def test_out_of_range_enum_dropped_not_defaulted(self):
        raw = {"obstacle": "x", "cost_hint": "catastrophic", "confidence": "absolutely certain"}
        out = pm_core._sanitize_friction_record(raw)
        assert "cost_hint" not in out
        assert "confidence" not in out

    def test_missing_path_taken_defaults_empty_string(self):
        out = pm_core._sanitize_friction_record({"obstacle": "x"})
        assert out["path_taken"] == ""

    def test_optional_artifact_omitted_when_absent(self):
        out = pm_core._sanitize_friction_record({"obstacle": "x"})
        assert "artifact" not in out


class TestSanitizeFrictionArray:
    def test_non_list_payload_returns_empty(self):
        assert pm_core._sanitize_friction_array({"obstacle": "x"}) == []
        assert pm_core._sanitize_friction_array("not a list") == []
        assert pm_core._sanitize_friction_array(None) == []

    def test_caps_at_max_records(self):
        raw = [{"obstacle": f"obstacle {i}"} for i in range(20)]
        out = pm_core._sanitize_friction_array(raw)
        assert len(out) == pm_core.FRICTION_MAX_RECORDS
        assert [r["index"] for r in out] == list(range(pm_core.FRICTION_MAX_RECORDS))

    def test_drops_malformed_entries_preserves_index_of_survivors(self):
        raw = [
            {"obstacle": "first"},
            {"obstacle": ""},       # dropped: empty
            "not a dict",           # dropped: malformed
            {"obstacle": "fourth"},
        ]
        out = pm_core._sanitize_friction_array(raw)
        assert len(out) == 2
        assert out[0]["obstacle"] == "first"
        assert out[0]["index"] == 0
        assert out[1]["obstacle"] == "fourth"
        assert out[1]["index"] == 3

    def test_all_dropped_returns_empty(self):
        raw = [{"obstacle": ""}, {"not_obstacle": "x"}]
        assert pm_core._sanitize_friction_array(raw) == []


# ---------------------------------------------------------------------------
# Harvest — silent-gap + agent-deposit, idempotency
# ---------------------------------------------------------------------------

DISPATCH_RECORD = {
    "gpu_id": "gpu-task-001",
    "spec_id": "spec-abc",
    "agent_type": "fixer",
    "repo": "lapis-pm",
    "ts": "2026-07-20T00:00:00+00:00",
    "status": "pending",
    "retry_count": 0,
}

PR = {
    "number": 42,
    "body": "Implements the thing.\n\n<!-- lapis-gpu-id: gpu-task-001 -->\n<!-- lapis-tid: my-target -->",
}


@pytest.fixture
def queue_path(tmp_path, monkeypatch):
    path = tmp_path / "friction" / "queue.jsonl"
    monkeypatch.setattr(pm_core, "_friction_queue_path", lambda: path)
    return path


class TestHarvestFrictionForPr:
    def test_no_matching_dispatch_record_skips(self, queue_path):
        pr = {"number": 99, "body": "no markers here"}
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]):
            written = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", pr)
        assert written == 0
        assert not queue_path.exists()

    def test_silent_gap_when_no_sidecar(self, queue_path):
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=None):
            written = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        assert written == 1
        lines = queue_path.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["record_source"] == "silent-gap"
        assert rec["obstacle"] == "no_friction_reported"
        assert rec["derived_confidence"] == "zero"
        assert rec["environment"] == {
            "latency_ms": None, "step_depth": None, "load_pct": None, "pr_diff_lines": None,
        }
        assert rec["provenance"]["task_id"] == "gpu-task-001"
        assert rec["provenance"]["spec_id"] == "spec-abc"
        assert rec["provenance"]["target_id"] == "my-target"
        assert rec["provenance"]["repo"] == "lapis-pm"
        assert rec["provenance"]["pr"] == 42
        assert rec["provenance"]["agent_type"] == "fixer"

    def test_silent_gap_when_sidecar_has_only_dropped_records(self, queue_path):
        raw = json.dumps([{"obstacle": ""}, {"not_obstacle": "whatever"}])
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=raw):
            written = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        assert written == 1
        rec = json.loads(queue_path.read_text().strip())
        assert rec["record_source"] == "silent-gap"

    def test_malformed_json_sidecar_treated_as_silent_gap(self, queue_path):
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value="{not valid json"):
            written = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        assert written == 1
        rec = json.loads(queue_path.read_text().strip())
        assert rec["record_source"] == "silent-gap"

    def test_agent_deposit_records_written(self, queue_path):
        raw = json.dumps([
            {"obstacle": "hit a rate limit", "path_taken": "backed off and retried",
             "cost_hint": "minor", "confidence": "certain"},
            {"obstacle": "second obstacle"},
        ])
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=raw):
            written = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        assert written == 2
        lines = [json.loads(l) for l in queue_path.read_text().strip().splitlines()]
        assert all(r["record_source"] == "agent-deposit" for r in lines)
        assert lines[0]["obstacle"] == "hit a rate limit"
        assert lines[0]["cost_hint"] == "minor"
        assert lines[1]["obstacle"] == "second obstacle"
        assert lines[1]["path_taken"] == ""

    def test_no_agent_field_ever_reaches_provenance_or_environment(self, queue_path):
        raw = json.dumps([{
            "obstacle": "x", "provenance": {"task_id": "forged"}, "environment": {"load_pct": 999},
        }])
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=raw):
            pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        rec = json.loads(queue_path.read_text().strip())
        assert rec["provenance"]["task_id"] == "gpu-task-001"  # machine-stamped, not the forged value
        assert rec["environment"]["load_pct"] is None


class TestIdempotency:
    def test_harvest_twice_same_pr_no_duplicate(self, queue_path):
        raw = json.dumps([{"obstacle": "one obstacle"}])
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=raw):
            first = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
            second = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        assert first == 1
        assert second == 0
        lines = queue_path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_silent_gap_then_later_agent_deposit_does_not_collide(self, queue_path):
        # First observe: no sidecar yet → silent-gap at reserved index -1.
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=None):
            pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        # A later commit adds friction.json (e.g. via fixer_retry) → real record at index 0.
        raw = json.dumps([{"obstacle": "found it later"}])
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=raw):
            written = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        assert written == 1
        lines = [json.loads(l) for l in queue_path.read_text().strip().splitlines()]
        assert len(lines) == 2
        sources = {r["record_source"] for r in lines}
        assert sources == {"silent-gap", "agent-deposit"}


# ---------------------------------------------------------------------------
# read_friction_records
# ---------------------------------------------------------------------------

class TestReadFrictionRecords:
    def _write(self, queue_path, records):
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_missing_queue_returns_empty(self, queue_path):
        assert pm_core.read_friction_records() == []

    def test_newest_first_and_filters(self, queue_path):
        records = [
            {"provenance": {"target_id": "t1", "repo": "lapis-pm"}, "index": 0, "obstacle": "a"},
            {"provenance": {"target_id": "t2", "repo": "other-repo"}, "index": 0, "obstacle": "b"},
            {"provenance": {"target_id": "t1", "repo": "lapis-pm"}, "index": 1, "obstacle": "c"},
        ]
        self._write(queue_path, records)

        all_recs = pm_core.read_friction_records()
        assert [r["obstacle"] for r in all_recs] == ["c", "b", "a"]  # newest (last-written) first

        t1_only = pm_core.read_friction_records(target_id="t1")
        assert [r["obstacle"] for r in t1_only] == ["c", "a"]

        repo_only = pm_core.read_friction_records(repo="other-repo")
        assert [r["obstacle"] for r in repo_only] == ["b"]

        limited = pm_core.read_friction_records(limit=1)
        assert [r["obstacle"] for r in limited] == ["c"]

    def test_skips_malformed_lines(self, queue_path):
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text('{"obstacle": "good"}\nnot json at all\n')
        recs = pm_core.read_friction_records()
        assert len(recs) == 1
        assert recs[0]["obstacle"] == "good"


# ---------------------------------------------------------------------------
# Queue path fallback (classmap key not yet landed in agents-core)
# ---------------------------------------------------------------------------

class TestFrictionQueuePathFallback:
    def test_falls_back_to_room_root_when_classmap_key_missing(self, monkeypatch, tmp_path):
        def _raise_keyerror(key, *parts, **kwargs):
            raise KeyError(key)
        monkeypatch.setattr(pm_core, "room_path", _raise_keyerror)
        monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
        path = pm_core._friction_queue_path()
        assert path == tmp_path / "friction" / "queue.jsonl"


# ---------------------------------------------------------------------------
# CLI — silent-gap rendered as a visibly distinct class
# ---------------------------------------------------------------------------

class TestCmdFrictionList:
    def test_empty_queue(self, capsys):
        args = type("A", (), {"target": None, "repo": None, "limit": 20, "json": False})()
        with patch("lapis_pm.pm_core.read_friction_records", return_value=[]):
            rc = cli.cmd_friction_list(args)
        assert rc == 0
        assert "(no friction records)" in capsys.readouterr().out

    def test_silent_gap_visibly_distinct(self, capsys):
        records = [
            {
                "record_source": "silent-gap",
                "obstacle": "no_friction_reported",
                "derived_confidence": "zero",
                "provenance": {"captured_at": "t", "target_id": "tid", "repo": "r", "pr": 1},
            },
            {
                "record_source": "agent-deposit",
                "obstacle": "real obstacle",
                "path_taken": "real path",
                "provenance": {"captured_at": "t", "target_id": "tid", "repo": "r", "pr": 1},
            },
        ]
        args = type("A", (), {"target": None, "repo": None, "limit": 20, "json": False})()
        with patch("lapis_pm.pm_core.read_friction_records", return_value=records):
            rc = cli.cmd_friction_list(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "[SILENT-GAP]" in out
        assert "real obstacle" in out

    def test_json_output(self, capsys):
        records = [{"record_source": "agent-deposit", "obstacle": "x", "provenance": {}}]
        args = type("A", (), {"target": None, "repo": None, "limit": 20, "json": True})()
        with patch("lapis_pm.pm_core.read_friction_records", return_value=records):
            rc = cli.cmd_friction_list(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert json.loads(out) == records
