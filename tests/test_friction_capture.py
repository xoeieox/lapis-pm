"""Tests for operational-learning-friction-capture-v0 (Unit 1a).

Covers: zero-trust sanitization of agent-authored fields, silent-gap capture,
idempotent append-only queue writes, and the `lapis-pm friction list` CLI
read path.
"""

from __future__ import annotations

import fcntl
import json
import threading
import time
from pathlib import Path
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
# D2 — content-level dedup suppressing cross-target inherited entries
# ---------------------------------------------------------------------------

def _agent_deposit_record(task_id, target_id, obstacle, path_taken, captured_at, index=0, pr=1):
    return {
        "record_source": "agent-deposit",
        "obstacle": obstacle,
        "path_taken": path_taken,
        "index": index,
        "environment": dict(pm_core._FRICTION_NULL_ENVIRONMENT),
        "provenance": {
            "task_id": task_id, "target_id": target_id, "repo": "lapis-pm",
            "pr": pr, "agent_type": "fixer", "captured_at": captured_at,
        },
    }


class TestContentDedupSuppression:
    def test_inherited_entry_suppressed_across_tasks(self, queue_path, caplog):
        existing_rec = _agent_deposit_record(
            "gpu-task-A", "target-a", "Hit a rate limit on the API",
            "Backed off and retried", "2026-07-20T00:00:00Z",
        )
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text(json.dumps(existing_rec) + "\n")

        dispatch_b = {**DISPATCH_RECORD, "gpu_id": "gpu-task-B"}
        pr_b = {"number": 77, "body": "<!-- lapis-gpu-id: gpu-task-B -->\n<!-- lapis-tid: target-b -->"}
        raw = json.dumps([{"obstacle": "Hit a rate limit on the API", "path_taken": "Backed off and retried"}])
        with caplog.at_level("WARNING"), \
             patch("lapis_pm.pm_core.load_dispatched", return_value=[dispatch_b]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=raw):
            written = pm_core._harvest_friction_for_pr("target-b", "lapis-pm", pr_b)

        assert written == 0
        lines = queue_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == existing_rec  # original untouched

        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "suppressed inherited entry" in m and "task=gpu-task-B" in m and "task=gpu-task-A" in m
            for m in warnings
        )

        marker_path = queue_path.parent / "inherited-entries-gpu-task-B.json"
        assert marker_path.exists()
        marker = json.loads(marker_path.read_text())
        assert len(marker) == 1
        assert marker[0]["incoming"]["task_id"] == "gpu-task-B"
        assert marker[0]["incoming"]["target_id"] == "target-b"
        assert marker[0]["existing"]["task_id"] == "gpu-task-A"
        assert marker[0]["existing"]["target_id"] == "target-a"

    def test_distinct_obstacle_from_same_task_appends_normally(self, queue_path):
        raw = json.dumps([
            {"obstacle": "first distinct obstacle", "path_taken": "resolved one way"},
            {"obstacle": "second distinct obstacle", "path_taken": "resolved another way"},
        ])
        with patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=raw):
            written = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        assert written == 2
        lines = [json.loads(l) for l in queue_path.read_text().strip().splitlines()]
        assert {r["obstacle"] for r in lines} == {"first distinct obstacle", "second distinct obstacle"}

    def test_same_task_repeat_handled_by_identity_key_not_content_suppression(self, queue_path, caplog):
        raw = json.dumps([{"obstacle": "flaky test", "path_taken": "reran it"}])
        with caplog.at_level("WARNING"), \
             patch("lapis_pm.pm_core.load_dispatched", return_value=[DISPATCH_RECORD]), \
             patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=raw):
            first = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
            second = pm_core._harvest_friction_for_pr("my-target", "lapis-pm", PR)
        assert first == 1
        assert second == 0
        assert not any("suppressed inherited entry" in r.message for r in caplog.records)
        assert len(queue_path.read_text().strip().splitlines()) == 1

    def test_silent_gap_records_never_content_suppressed_across_targets(self, queue_path, caplog):
        with caplog.at_level("WARNING"):
            for i in range(3):
                dispatch = {**DISPATCH_RECORD, "gpu_id": f"gpu-task-{i}"}
                pr = {"number": 100 + i,
                      "body": f"<!-- lapis-gpu-id: gpu-task-{i} -->\n<!-- lapis-tid: target-{i} -->"}
                with patch("lapis_pm.pm_core.load_dispatched", return_value=[dispatch]), \
                     patch("lapis_pm.pm_core._fetch_friction_sidecar", return_value=None):
                    written = pm_core._harvest_friction_for_pr(f"target-{i}", "lapis-pm", pr)
                assert written == 1
        assert not any("suppressed inherited entry" in r.message for r in caplog.records)
        lines = queue_path.read_text().strip().splitlines()
        assert len(lines) == 3
        assert all(json.loads(l)["record_source"] == "silent-gap" for l in lines)
        # Clean path: no marker files written at all.
        assert list(queue_path.parent.glob("inherited-entries-*.json")) == []


# ---------------------------------------------------------------------------
# D2a — serialized queue writers (flock)
# ---------------------------------------------------------------------------

class TestQueueLockSerialization:
    def test_append_blocks_then_succeeds_once_lock_released(self, queue_path):
        lock_path = pm_core._friction_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        held_fd = open(lock_path, "w")
        fcntl.flock(held_fd, fcntl.LOCK_EX)

        def release_after_delay():
            time.sleep(0.3)
            fcntl.flock(held_fd, fcntl.LOCK_UN)
            held_fd.close()

        t = threading.Thread(target=release_after_delay)
        t.start()
        try:
            rec = _agent_deposit_record("t1", "tgt", "x", "y", "2026-07-27T00:00:00Z")
            written = pm_core._append_friction_records([rec])
        finally:
            t.join()
        assert written == 1
        assert len(queue_path.read_text().strip().splitlines()) == 1

    def test_append_lock_timeout_warns_returns_zero_never_raises_then_retry_succeeds(
        self, queue_path, caplog,
    ):
        lock_path = pm_core._friction_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        held_fd = open(lock_path, "w")
        fcntl.flock(held_fd, fcntl.LOCK_EX)
        try:
            rec = _agent_deposit_record("t1", "tgt", "x", "y", "2026-07-27T00:00:00Z")
            with caplog.at_level("WARNING"), \
                 patch.object(pm_core, "_FRICTION_LOCK_TIMEOUT_DEFAULT", 0.2):
                written = pm_core._append_friction_records([rec])
            assert written == 0
            assert any("lock timed out" in r.message for r in caplog.records)
            assert not queue_path.exists()
        finally:
            fcntl.flock(held_fd, fcntl.LOCK_UN)
            held_fd.close()

        # Lock released — the same record retries successfully (idempotence preserved).
        written2 = pm_core._append_friction_records([rec])
        assert written2 == 1

    def test_backfill_lock_timeout_aborts_and_changes_zero_bytes(self, queue_path):
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        seed = _agent_deposit_record("t1", "tgt", "x", "y", "2026-07-27T00:00:00Z")
        queue_path.write_text(json.dumps(seed) + "\n")
        before = queue_path.read_text()

        lock_path = pm_core._friction_lock_path()
        held_fd = open(lock_path, "w")
        fcntl.flock(held_fd, fcntl.LOCK_EX)
        try:
            with patch.object(pm_core, "_FRICTION_LOCK_TIMEOUT_DEFAULT", 0.2):
                with pytest.raises(pm_core._FrictionLockTimeout):
                    pm_core.friction_backfill_provenance(apply=True)
        finally:
            fcntl.flock(held_fd, fcntl.LOCK_UN)
            held_fd.close()
        assert queue_path.read_text() == before

    def test_cli_backfill_exits_nonzero_on_lock_timeout(self, capsys):
        args = type("A", (), {"apply": True})()
        with patch(
            "lapis_pm.pm_core.friction_backfill_provenance",
            side_effect=pm_core._FrictionLockTimeout("friction queue lock timed out after 10s: /tmp/x"),
            ):
            rc = cli.cmd_friction_backfill(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# D3 — one-shot backfill: tag, never delete
# ---------------------------------------------------------------------------

class TestFrictionGroupKeyForBackfill:
    def test_agent_deposit_gets_a_key(self):
        rec = _agent_deposit_record("t1", "tgt", "obstacle text", "path text", "2026-07-27T00:00:00Z")
        assert pm_core._friction_group_key_for_backfill(rec) is not None

    def test_silent_gap_excluded_entirely(self):
        rec = {
            "record_source": "silent-gap", "obstacle": "no_friction_reported",
            "derived_confidence": "zero", "index": pm_core._FRICTION_SILENT_GAP_INDEX,
            "provenance": {"task_id": "t1", "target_id": "tgt", "captured_at": "2026-07-27T00:00:00Z"},
        }
        assert pm_core._friction_group_key_for_backfill(rec) is None

    def test_malformed_line_excluded(self):
        assert pm_core._friction_group_key_for_backfill(None) is None


class TestComputeFrictionBackfillTags:
    def test_silent_gap_corpus_produces_zero_groups(self):
        records = [
            {
                "record_source": "silent-gap", "obstacle": "no_friction_reported",
                "derived_confidence": "zero", "index": pm_core._FRICTION_SILENT_GAP_INDEX,
                "provenance": {"task_id": f"t{i}", "target_id": f"tgt{i}", "captured_at": "2026-07-27T00:00:00Z"},
            }
            for i in range(5)
        ]
        tags, group_count = pm_core._compute_friction_backfill_tags(records)
        assert tags == {}
        assert group_count == 0


class TestBackfillNonDestructive:
    def _seed(self, queue_path, records):
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_tags_only_non_canonical_cross_target_members(self, queue_path):
        records = [
            _agent_deposit_record("task-A", "target-a", "shared obstacle text", "shared path", "2026-07-01T00:00:00Z"),
            _agent_deposit_record("task-B", "target-b", "unrelated obstacle", "unrelated path", "2026-07-02T00:00:00Z"),
            _agent_deposit_record("task-C", "target-c", "Shared Obstacle Text", "Shared  Path", "2026-07-03T00:00:00Z"),
        ]
        self._seed(queue_path, records)
        result = pm_core.friction_backfill_provenance(apply=True)
        assert result["total_records"] == 3
        assert result["groups"] == 1
        assert result["tagged"] == 1
        assert result["applied"] is True

        lines = [json.loads(l) for l in queue_path.read_text().strip().splitlines()]
        assert len(lines) == 3  # line count preserved
        assert [l["provenance"]["task_id"] for l in lines] == ["task-A", "task-B", "task-C"]  # order preserved
        assert "provenance_suspect" not in lines[0]  # canonical (earliest captured_at)
        assert "provenance_suspect" not in lines[1]  # unrelated content, untouched
        assert lines[2]["provenance_suspect"] is True
        assert lines[2]["canonical_task_id"] == "task-A"

    def test_same_task_repeats_not_grouped_as_cross_attribution(self, queue_path):
        records = [
            _agent_deposit_record("task-A", "target-a", "same task repeat", "path", "2026-07-01T00:00:00Z", index=0),
            _agent_deposit_record("task-A", "target-a", "same task repeat", "path", "2026-07-01T00:00:01Z", index=1),
        ]
        self._seed(queue_path, records)
        result = pm_core.friction_backfill_provenance(apply=True)
        assert result["tagged"] == 0
        assert result["groups"] == 0

    def test_idempotent_across_two_runs(self, queue_path):
        records = [
            _agent_deposit_record("task-A", "target-a", "dup obstacle", "dup path", "2026-07-01T00:00:00Z"),
            _agent_deposit_record("task-B", "target-b", "dup obstacle", "dup path", "2026-07-02T00:00:00Z"),
        ]
        self._seed(queue_path, records)
        first = pm_core.friction_backfill_provenance(apply=True)
        content_after_first = queue_path.read_text()
        second = pm_core.friction_backfill_provenance(apply=True)
        content_after_second = queue_path.read_text()
        assert first == second
        assert content_after_first == content_after_second

    def test_dry_run_writes_nothing(self, queue_path):
        records = [
            _agent_deposit_record("task-A", "target-a", "dup obstacle", "dup path", "2026-07-01T00:00:00Z"),
            _agent_deposit_record("task-B", "target-b", "dup obstacle", "dup path", "2026-07-02T00:00:00Z"),
        ]
        self._seed(queue_path, records)
        before = queue_path.read_text()
        result = pm_core.friction_backfill_provenance(apply=False)
        after = queue_path.read_text()
        assert before == after
        assert result["tagged"] == 1
        assert result["applied"] is False

    def test_missing_queue_is_a_no_op(self, queue_path):
        result = pm_core.friction_backfill_provenance(apply=True)
        assert result == {"total_records": 0, "tagged": 0, "groups": 0, "applied": False}


class TestBackfillManualOnly:
    def test_no_systemd_unit_references_backfill(self):
        repo_root = Path(__file__).resolve().parents[1]
        systemd_dir = repo_root / "systemd"
        hits = []
        for path in sorted(systemd_dir.glob("*")):
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            if "backfill" in content.lower():
                hits.append(str(path.relative_to(repo_root)))
        assert hits == []

    def test_no_daemon_or_night_pass_call_site(self):
        repo_root = Path(__file__).resolve().parents[1]
        lapis_pm_dir = repo_root / "lapis_pm"
        offenders = []
        for path in sorted(lapis_pm_dir.glob("*.py")):
            if path.name in {"cli.py", "pm_core.py"}:
                continue  # the CLI subcommand wiring and the implementation itself
            content = path.read_text(encoding="utf-8", errors="ignore")
            if "friction_backfill_provenance" in content or "backfill-provenance" in content:
                offenders.append(str(path.relative_to(repo_root)))
        assert offenders == []


class TestFrictionListExcludeSuspect:
    def test_exclude_suspect_flag_filters(self, capsys):
        records = [
            {"record_source": "agent-deposit", "obstacle": "clean one", "provenance": {}},
            {"record_source": "agent-deposit", "obstacle": "suspect one", "provenance": {},
             "provenance_suspect": True, "canonical_task_id": "task-A"},
        ]
        args = type("A", (), {
            "target": None, "repo": None, "limit": 20, "json": False, "exclude_suspect": True,
        })()
        with patch("lapis_pm.pm_core.read_friction_records", return_value=records):
            rc = cli.cmd_friction_list(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "clean one" in out
        assert "suspect one" not in out

    def test_suspect_shown_with_marker_by_default(self, capsys):
        records = [
            {"record_source": "agent-deposit", "obstacle": "suspect one", "provenance": {},
             "provenance_suspect": True, "canonical_task_id": "task-A"},
        ]
        args = type("A", (), {
            "target": None, "repo": None, "limit": 20, "json": False, "exclude_suspect": False,
        })()
        with patch("lapis_pm.pm_core.read_friction_records", return_value=records):
            rc = cli.cmd_friction_list(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "[PROVENANCE-SUSPECT canonical_task_id=task-A]" in out


# ---------------------------------------------------------------------------
# D1 — registry.yaml instruction conformance
# ---------------------------------------------------------------------------

class TestRegistryInstructionConformance:
    def test_replace_dont_append_sentence_present(self):
        registry_path = Path(__file__).resolve().parents[1] / "lapis_pm" / "registry.yaml"
        import re as _re
        content = _re.sub(r"\s+", " ", registry_path.read_text(encoding="utf-8"))
        assert "per-PR sidecar, not a running log" in content
        assert "replace its entire contents with your own entries" in content


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
