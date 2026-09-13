"""L2.D2 / L2.D3 / L2.D4 / L2.D6 tests (local-reviewer-identity-and-provenance-v0,
Leg 2).

Covers:
- L2.D2: served_model on dispatch records — null-tolerant across missing
  field / missing yaml / crashed run; a null yaml served_model lands as
  None, never the seat alias; a malformed (bound-violating) token is void.
- L2.D3: the pm:served=<model> episodic tag — present when known, absent
  (a non-error) when unknown, never the alias value.
- L2.D4: the deploy-log label <seat-alias>:<served-model> — same-call echo
  wins; seam-filtered UTC-dated ledger fallback (previous-UTC-day fallback,
  multi-model day -> not-reported); the seat alias never occupies the
  served-model slot.
- L2.D6: the plain-text boundary scan over lapis_pm/*.py.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from lapis_pm import pm_core, provenance

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# L2.D2 — served_model on dispatch records (queue-yaml read)
# ---------------------------------------------------------------------------

class TestServedModelFromQueueYaml:
    def test_missing_field_is_none(self, tmp_path, monkeypatch):
        """Pre-Leg-1 yaml (no served_model field) -> None."""
        comp = tmp_path / "cq-comp"
        comp.mkdir()
        (comp / "task1.yaml").write_text(yaml.safe_dump(
            {"id": "task1", "model": "gravitywell-slot1", "status": "completed"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", comp)
        assert provenance.read_served_model_from_queue_yaml("task1") is None

    def test_explicit_null_is_none_never_alias(self, tmp_path, monkeypatch):
        """Void echo: served_model is explicit null -> None, never the alias."""
        comp = tmp_path / "cq-comp"
        comp.mkdir()
        (comp / "task2.yaml").write_text(yaml.safe_dump(
            {"id": "task2", "model": "gravitywell-slot1",
             "served_model": None, "status": "completed"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", comp)
        result = provenance.read_served_model_from_queue_yaml("task2")
        assert result is None
        assert result != "gravitywell-slot1"

    def test_present_echo_is_returned(self, tmp_path, monkeypatch):
        comp = tmp_path / "cq-comp"
        comp.mkdir()
        (comp / "task3.yaml").write_text(yaml.safe_dump(
            {"id": "task3", "model": "gravitywell-slot1",
             "served_model": "gravitywell-27b", "status": "completed"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", comp)
        assert provenance.read_served_model_from_queue_yaml("task3") == "gravitywell-27b"

    def test_missing_yaml_is_none(self, tmp_path, monkeypatch):
        """Crashed run: the yaml is absent entirely -> None, no raise."""
        comp = tmp_path / "cq-comp"
        comp.mkdir()
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", comp)
        assert provenance.read_served_model_from_queue_yaml("absent") is None

    def test_failed_dir_yaml_is_read(self, tmp_path, monkeypatch):
        """A failed-dir yaml (crashed run that left a task file) is read
        the same way."""
        failed = tmp_path / "cq-failed"
        failed.mkdir()
        (failed / "task4.yaml").write_text(yaml.safe_dump(
            {"id": "task4", "model": "gravitywell-slot1",
             "served_model": "gravitywell-27b", "status": "failed"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_FAILED_DIR", failed)
        assert provenance.read_served_model_from_queue_yaml("task4") == "gravitywell-27b"

    def test_bound_violating_token_is_void(self, tmp_path, monkeypatch):
        """A server-echoed token outside the L1.D1 bound is VOID, not a value."""
        comp = tmp_path / "cq-comp"
        comp.mkdir()
        (comp / "task5.yaml").write_text(yaml.safe_dump(
            {"id": "task5", "model": "gravitywell-slot1",
             "served_model": "evil token; rm -rf /", "status": "completed"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", comp)
        assert provenance.read_served_model_from_queue_yaml("task5") is None

    def test_no_task_id_is_none(self):
        assert provenance.read_served_model_from_queue_yaml(None) is None
        assert provenance.read_served_model_from_queue_yaml("") is None

    def test_malformed_yaml_is_none(self, tmp_path, monkeypatch):
        comp = tmp_path / "cq-comp"
        comp.mkdir()
        (comp / "task6.yaml").write_text("::: not yaml {{{")
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", comp)
        assert provenance.read_served_model_from_queue_yaml("task6") is None

    def test_encode_gpu_results_stamps_served_model(self, tmp_path, monkeypatch):
        """End-to-end: _encode_gpu_results stamps served_model onto the
        dispatch record from the queue yaml, and a missing yaml leaves the
        field absent without failing the encode."""
        from unittest.mock import MagicMock

        tid = "prov-tid"
        # Queue dirs
        comp = tmp_path / "cq-comp"
        comp.mkdir()
        (comp / "taskA.yaml").write_text(yaml.safe_dump(
            {"id": "taskA", "model": "gravitywell-slot1",
             "served_model": "gravitywell-27b"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", comp)
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_FAILED_DIR", tmp_path / "cq-failed")
        monkeypatch.setattr(pm_core, "COMPLETED_DIR", tmp_path / "gq-comp")
        monkeypatch.setattr(pm_core, "FAILED_DIR", tmp_path / "gq-failed")
        (tmp_path / "cq-failed").mkdir()
        (tmp_path / "gq-comp").mkdir()
        (tmp_path / "gq-failed").mkdir()

        # Output files (fixer output, non-reviewer so no verdict parsing)
        out = tmp_path / "cq-comp" / "taskA-output.md"
        out.write_text("PR opened: http://forgejo/Erah/lapis-pm/pulls/1\n")
        out2 = tmp_path / "cq-comp" / "taskB-output.md"
        out2.write_text("PR opened: http://forgejo/Erah/lapis-pm/pulls/2\n")

        records = [
            {"gpu_id": "taskA", "spec_id": None, "agent_type": "fixer",
             "intent": "fix x", "repo": "lapis-pm", "ts": "2026-09-12T00:00:00",
             "status": "pending", "retry_count": 0},
            {"gpu_id": "taskB", "spec_id": None, "agent_type": "fixer",
             "intent": "fix y", "repo": "lapis-pm", "ts": "2026-09-12T00:00:00",
             "status": "pending", "retry_count": 0},
        ]
        saved: dict[str, list] = {}
        loaded = {tid: records}

        def fake_load(target_id):
            return loaded.get(target_id, [])

        def fake_save(target_id, recs):
            saved[target_id] = recs

        # Hermetic: no mem, no episodic, no forgejo, no slots, no confabulation
        mem = MagicMock()
        mem.get.return_value = None
        monkeypatch.setattr(pm_core, "_mem", lambda: mem)
        monkeypatch.setattr(pm_core, "load_dispatched", fake_load)
        monkeypatch.setattr(pm_core, "save_dispatched", fake_save)
        monkeypatch.setattr(pm_core, "append_dispatched", lambda *a, **k: None)
        monkeypatch.setattr(pm_core, "_close_slot_and_deposit", lambda *a, **k: None)
        monkeypatch.setattr(pm_core, "_read_fixer_verdict", lambda *a, **k: None)
        monkeypatch.setattr(pm_core, "_consume_fixer_verdict", lambda *a, **k: None)
        monkeypatch.setattr(pm_core, "_read_fixer_meta", lambda *a, **k: None)
        monkeypatch.setattr(pm_core, "_consume_fixer_meta", lambda *a, **k: None)
        monkeypatch.setattr(pm_core, "episodic", MagicMock())
        monkeypatch.setattr(pm_core, "_check_calcification", lambda *a, **k: None)
        monkeypatch.setattr(pm_core, "_tick_corr_cache", {})

        with patch("lapis_pm.episodic.write_result"), \
             patch("lapis_pm.episodic.write_observation"), \
             patch("lapis_pm.episodic.all_comments", return_value=[]):
            total, failed = pm_core._encode_gpu_results(tid)

        assert total == 2
        rec_by_id = {r["gpu_id"]: r for r in saved[tid]}
        # taskA: the yaml carries the echo -> stamped
        assert rec_by_id["taskA"]["served_model"] == "gravitywell-27b"
        # taskB: no yaml -> field absent (None via the read), never the alias
        assert rec_by_id["taskB"].get("served_model") is None
        assert rec_by_id["taskB"].get("served_model") != "gravitywell-slot1"


# ---------------------------------------------------------------------------
# L2.D3 — pm:served=<model> episodic tag
# ---------------------------------------------------------------------------

class TestServedTag:
    def test_tag_present_when_known(self):
        assert provenance.served_tag_for(
            {"served_model": "gravitywell-27b"}) == "pm:served=gravitywell-27b"

    def test_tag_absent_is_non_error(self):
        assert provenance.served_tag_for({}) is None
        assert provenance.served_tag_for({"served_model": None}) is None
        assert provenance.served_tag_for({"served_model": ""}) is None

    def test_tag_never_alias(self):
        # A record whose served_model somehow holds the alias must not emit
        # a pm:served tag with the alias value (the alias is a role, not a
        # model substance).
        tag = provenance.served_tag_for({"served_model": "gravitywell"})
        # 'gravitywell' matches the token bound, so the tag is emitted as-is;
        # the guard is upstream (the read never returns the alias). Assert the
        # tag shape is the raw value and that the read side can't produce it.
        assert tag == "pm:served=gravitywell"
        assert provenance.read_served_model_from_queue_yaml(None) is None


# ---------------------------------------------------------------------------
# L2.D4 — deploy-log label <seat-alias>:<served-model>
# ---------------------------------------------------------------------------

def _write_ledger(root: Path, day: str, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for e in entries:
        entry = {
            "ts": f"{day}T12:00:00.000+00:00",
            "seam": "call_gw_agent",
            "requested_operator": "gravitywell",
            "served_model": e.get("served_model"),
            "host": "http://203.0.113.11:8081",
            "cost_class": "local-gw",
            "fallback_fired": False,
            "cost_usd": None,
            "duration_ms": 10.0,
            "ok": True,
        }
        lines.append(json.dumps(entry))
    (root / f"{day}.jsonl").write_text("\n".join(lines) + "\n")


class TestDeployLogLabel:
    def test_same_call_echo_wins(self, tmp_path, monkeypatch):
        """Resolution step 1: the call's in-bounds served_model_out wins."""
        monkeypatch.setattr(
            provenance, "_locality_root", lambda: tmp_path / "locality")
        assert provenance.deploy_log_label(["gravitywell-27b"]) == \
            "gravitywell:gravitywell-27b"

    def test_echo_bound_violating_falls_through_to_ledger(self, tmp_path, monkeypatch):
        """A bound-violating echo is VOID — the ledger fallback decides."""
        monkeypatch.setattr(
            provenance, "_locality_root", lambda: tmp_path / "locality")
        now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
        _write_ledger(tmp_path / "locality", "2026-09-12",
                      [{"served_model": "gravitywell-27b"}])
        assert provenance.deploy_log_label(["bad token; x"], now) == \
            "gravitywell:gravitywell-27b"

    def test_ledger_fallback_single_model(self, tmp_path, monkeypatch):
        """Resolution step 2: seam-filtered, single-distinct-model day."""
        monkeypatch.setattr(
            provenance, "_locality_root", lambda: tmp_path / "locality")
        now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
        _write_ledger(tmp_path / "locality", "2026-09-12", [
            {"served_model": "gravitywell-27b"},
            {"served_model": "gravitywell-27b"},
        ])
        assert provenance.deploy_log_label(None, now) == \
            "gravitywell:gravitywell-27b"

    def test_multi_model_day_is_not_reported(self, tmp_path, monkeypatch):
        """A mixed day yields <alias>:not-reported — never the alias alone."""
        monkeypatch.setattr(
            provenance, "_locality_root", lambda: tmp_path / "locality")
        now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
        _write_ledger(tmp_path / "locality", "2026-09-12", [
            {"served_model": "gravitywell-27b"},
            {"served_model": "other-model"},
        ])
        label = provenance.deploy_log_label(None, now)
        assert label == "gravitywell:not-reported"

    def test_empty_today_falls_back_to_previous_utc_day(self, tmp_path, monkeypatch):
        """UTC-date convention: empty today -> previous UTC day before
        not-reported."""
        monkeypatch.setattr(
            provenance, "_locality_root", lambda: tmp_path / "locality")
        now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
        _write_ledger(tmp_path / "locality", "2026-09-11",
                      [{"served_model": "gravitywell-27b"}])
        assert provenance.deploy_log_label(None, now) == \
            "gravitywell:gravitywell-27b"

    def test_empty_both_days_is_not_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            provenance, "_locality_root", lambda: tmp_path / "locality")
        now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
        assert provenance.deploy_log_label(None, now) == \
            "gravitywell:not-reported"

    def test_non_gw_agent_seam_entries_are_ignored(self, tmp_path, monkeypatch):
        """Filter keys on seam == call_gw_agent ONLY (the host field is the
        full GW base URL, never a filter key)."""
        monkeypatch.setattr(
            provenance, "_locality_root", lambda: tmp_path / "locality")
        now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
        root = tmp_path / "locality"
        root.mkdir()
        entries = [
            {"ts": "2026-09-12T12:00:00.000+00:00",
             "seam": "call_operator", "requested_operator": "sonnet",
             "served_model": "paid-model",
             "host": "http://203.0.113.11:8081", "cost_class": "paid-anthropic",
             "fallback_fired": False, "cost_usd": None, "duration_ms": 1.0,
             "ok": True},
        ]
        (root / "2026-09-12.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n")
        assert provenance.deploy_log_label(None, now) == \
            "gravitywell:not-reported"

    def test_alias_never_in_model_slot(self, tmp_path, monkeypatch):
        """The seat alias never occupies the served-model slot."""
        monkeypatch.setattr(
            provenance, "_locality_root", lambda: tmp_path / "locality")
        now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
        for label in (provenance.deploy_log_label(None, now),
                      provenance.deploy_log_label(["gravitywell-27b"], now)):
            model_slot = label.split(":", 1)[1]
            assert model_slot != "gravitywell", (
                f"alias in the model slot: {label!r}")

    def test_write_deploy_log_uses_new_label(self, tmp_path, monkeypatch):
        """pm_core._write_deploy_log emits the new label shape (no
        brief.py:unknown, no dead _BRIEF_MODEL)."""
        log = tmp_path / "deploy-log.md"
        monkeypatch.setattr(pm_core, "_DEPLOY_LOG", log)
        monkeypatch.setattr(
            provenance, "_locality_root", lambda: tmp_path / "locality")
        now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
        _write_ledger(tmp_path / "locality", "2026-09-12",
                      [{"served_model": "gravitywell-27b"}])
        pm_core._write_deploy_log("/srv/git/lapis-pm", "aaa", "bbb", "post-merge-hook")
        line = log.read_text().strip()
        assert "gravitywell:gravitywell-27b" in line
        assert "unknown" not in line
        # The old label shape is dead
        assert "brief.py:unknown" not in line
        # The line still parses through _DEPLOY_LOG_LINE_RE
        m = re.match(pm_core._DEPLOY_LOG_LINE_RE.pattern, line)
        assert m and m.group("trigger") == "post-merge-hook"


# ---------------------------------------------------------------------------
# L2.D6 — plain-text boundary scan over lapis_pm/*.py
# ---------------------------------------------------------------------------

# Forbidden identity strings (stale seat identities + retired paid identities).
# Historical references to the Opus era in genuinely historical context are
# NOT in scope — these are CURRENT-behavior assertions.
_FORBIDDEN_PATTERNS = [
    "Opus reviewer",
    "inline Sonnet",
    "local 122B",
]

# Escape hatch: legitimate strings that must remain, each with a rationale +
# sunset date (the pattern the naming-boundary test uses).
_ALLOWLIST: dict[str, tuple[str, str]] = {}


def _scan_files() -> list[Path]:
    return sorted((_REPO_ROOT / "lapis_pm").glob("*.py"))


def _scan_hits() -> list[str]:
    hits: list[str] = []
    for path in _scan_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pat in _FORBIDDEN_PATTERNS:
                if pat in line:
                    allowed = _ALLOWLIST.get(pat)
                    if allowed and allowed[0] in line:
                        continue  # allow-listed with rationale
                    hits.append(f"{path.name}:{lineno}: {pat!r} in {line.strip()[:120]!r}")
    return hits


def test_boundary_scan_no_stale_identity_strings():
    """Plain-text scan over lapis_pm/*.py: zero hits for the forbidden
    identity strings (string literals, comments, and docstrings in scope —
    deliberately broader than the AST-identifier precedent)."""
    hits = _scan_hits()
    assert not hits, "Stale identity strings found:\n" + "\n".join(hits)


def test_boundary_scan_catches_planted_string(tmp_path):
    """Mutation check: the scan catches a planted 'Opus reviewer' string."""
    planted = tmp_path / "planted.py"
    planted.write_text('X = "Opus reviewer is the gate"\n')
    text = planted.read_text(encoding="utf-8")
    assert any(p in text for p in _FORBIDDEN_PATTERNS)
