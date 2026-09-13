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
        (comp / "claude_20260912_120000_3000_fixer.yaml").write_text(
            yaml.safe_dump({"id": "claude_20260912_120000_3000_fixer",
                            "model": "gravitywell-slot1",
                            "served_model": "gravitywell-27b",
                            "status": "completed"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", comp)
        assert provenance.read_served_model_from_queue_yaml(
            "claude_20260912_120000_3000_fixer") == "gravitywell-27b"

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
        (failed / "claude_20260912_120000_4000_fixer.yaml").write_text(
            yaml.safe_dump({"id": "claude_20260912_120000_4000_fixer",
                            "model": "gravitywell-slot1",
                            "served_model": "gravitywell-27b",
                            "status": "failed"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_FAILED_DIR", failed)
        assert provenance.read_served_model_from_queue_yaml(
            "claude_20260912_120000_4000_fixer") == "gravitywell-27b"

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

    def test_namespace_guard_prefers_claude_queue_yaml(self, tmp_path, monkeypatch):
        """Collision case (cycle-1 review): a legacy gpu_queue yaml that
        shares a claude_-prefixed task id must NOT shadow the
        claude_queue record — the claude_ completed/failed yamls are
        matched first, and the legacy names are consulted only when
        absent."""
        cq_comp = tmp_path / "cq-comp"
        cq_comp.mkdir()
        (cq_comp / "claude_20260912_120000_1000_fixer.yaml").write_text(
            yaml.safe_dump({"id": "claude_20260912_120000_1000_fixer",
                            "model": "gravitywell-slot1",
                            "served_model": "gravitywell-27b"}))
        gq_comp = tmp_path / "gq-comp"
        gq_comp.mkdir()
        (gq_comp / "claude_20260912_120000_1000_fixer.yaml").write_text(
            yaml.safe_dump({"id": "claude_20260912_120000_1000_fixer",
                            "model": "legacy-alias",
                            "served_model": None}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", cq_comp)
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_FAILED_DIR", tmp_path / "cq-failed")
        monkeypatch.setattr(pm_core, "COMPLETED_DIR", gq_comp)
        monkeypatch.setattr(pm_core, "FAILED_DIR", tmp_path / "gq-failed")
        (tmp_path / "cq-failed").mkdir()
        (tmp_path / "gq-failed").mkdir()
        assert provenance.read_served_model_from_queue_yaml(
            "claude_20260912_120000_1000_fixer") == "gravitywell-27b"

    def test_namespace_guard_legacy_fallback_when_claude_absent(
            self, tmp_path, monkeypatch):
        """A legacy (non-claude_-prefixed) task id with only a gpu_queue
        yaml still resolves through the legacy names."""
        gq_comp = tmp_path / "gq-comp"
        gq_comp.mkdir()
        (gq_comp / "gpu_legacy_1.yaml").write_text(
            yaml.safe_dump({"id": "gpu_legacy_1",
                            "served_model": "gravitywell-27b"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", tmp_path / "cq-comp")
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_FAILED_DIR", tmp_path / "cq-failed")
        monkeypatch.setattr(pm_core, "COMPLETED_DIR", gq_comp)
        monkeypatch.setattr(pm_core, "FAILED_DIR", tmp_path / "gq-failed")
        (tmp_path / "cq-comp").mkdir()
        (tmp_path / "cq-failed").mkdir()
        (tmp_path / "gq-failed").mkdir()
        assert provenance.read_served_model_from_queue_yaml(
            "gpu_legacy_1") == "gravitywell-27b"

    def test_encode_gpu_results_stamps_served_model(self, tmp_path, monkeypatch):
        """End-to-end: _encode_gpu_results stamps served_model onto the
        dispatch record from the queue yaml, and a missing yaml leaves the
        field absent without failing the encode."""
        from unittest.mock import MagicMock

        tid = "prov-tid"
        # Queue dirs
        comp = tmp_path / "cq-comp"
        comp.mkdir()
        (comp / "claude_20260912_120000_1000_fixer.yaml").write_text(
            yaml.safe_dump({"id": "claude_20260912_120000_1000_fixer",
                            "model": "gravitywell-slot1",
                            "served_model": "gravitywell-27b"}))
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", comp)
        monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_FAILED_DIR", tmp_path / "cq-failed")
        monkeypatch.setattr(pm_core, "COMPLETED_DIR", tmp_path / "gq-comp")
        monkeypatch.setattr(pm_core, "FAILED_DIR", tmp_path / "gq-failed")
        (tmp_path / "cq-failed").mkdir()
        (tmp_path / "gq-comp").mkdir()
        (tmp_path / "gq-failed").mkdir()

        # Output files (fixer output, non-reviewer so no verdict parsing)
        out = tmp_path / "cq-comp" / "claude_20260912_120000_1000_fixer-output.md"
        out.write_text("PR opened: http://forgejo/Erah/lapis-pm/pulls/1\n")
        out2 = tmp_path / "cq-comp" / "claude_20260912_120000_2000_fixer-output.md"
        out2.write_text("PR opened: http://forgejo/Erah/lapis-pm/pulls/2\n")
        out3 = tmp_path / "cq-comp" / "claude_20260912_120000_3000_scout-output.md"
        out3.write_text("scout result\n")

        records = [
            {"gpu_id": "claude_20260912_120000_1000_fixer",
             "spec_id": None, "agent_type": "fixer",
             "intent": "fix x", "repo": "lapis-pm", "ts": "2026-09-12T00:00:00",
             "status": "pending", "retry_count": 0},
            {"gpu_id": "claude_20260912_120000_2000_fixer",
             "spec_id": None, "agent_type": "fixer",
             "intent": "fix y", "repo": "lapis-pm", "ts": "2026-09-12T00:00:00",
             "status": "pending", "retry_count": 0},
            # Cycle-1 review: the stamp is scoped to reviewer/fixer/
            # fixer_retry — a scout record reaching the same loop must NOT
            # gain the field.
            {"gpu_id": "claude_20260912_120000_3000_scout",
             "spec_id": None, "agent_type": "scout",
             "intent": "scout z", "repo": "lapis-pm", "ts": "2026-09-12T00:00:00",
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

        assert total == 3
        rec_by_id = {r["gpu_id"]: r for r in saved[tid]}
        # fixer #1: the yaml carries the echo -> stamped
        assert rec_by_id["claude_20260912_120000_1000_fixer"][
            "served_model"] == "gravitywell-27b"
        # fixer #2: no yaml -> field absent (None via the read), never the
        # alias
        assert rec_by_id["claude_20260912_120000_2000_fixer"].get(
            "served_model") is None
        assert rec_by_id["claude_20260912_120000_2000_fixer"].get(
            "served_model") != "gravitywell-slot1"
        # scout: outside the stamp scope — the field must not be added even
        # though the record reaches the same encode loop.
        assert "served_model" not in rec_by_id["claude_20260912_120000_3000_scout"]

    def test_stamp_scoped_to_dispatch_verified_agent_types(self, monkeypatch):
        """Cycle-1 review: the stamp scope is the dispatch-verified agent
        types (reviewer/reviewer_fresh/fixer/fixer_retry) — other agent
        types never gain the field at any record-close site."""
        assert pm_core._SERVED_MODEL_STAMP_AGENT_TYPES == frozenset(
            {"reviewer", "reviewer_fresh", "reviewer_fresh_contractor",
             "fixer", "fixer_retry"})
        rec = {"gpu_id": "claude_x", "agent_type": "scout"}
        pm_core._stamp_served_model(rec, "tid")
        assert "served_model" not in rec
        rec2 = {"gpu_id": None, "agent_type": "fixer"}
        pm_core._stamp_served_model(rec2, "tid")
        assert rec2.get("served_model") is None


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
        # A record whose served_model somehow holds the seat alias must NOT
        # emit a pm:served tag (the alias is a role, not a model substance —
        # Erah 2026-09-06 ruling). The re-validation guard in
        # served_tag_for catches this even for a hand-edited/corrupted
        # record.
        assert provenance.served_tag_for({"served_model": "gravitywell"}) is None
        assert provenance.read_served_model_from_queue_yaml(None) is None

    def test_tag_never_bound_violating(self):
        # A bound-violating value on the record (corrupted write) must not
        # leak into the episodic stream — the re-validation guard voids it.
        assert provenance.served_tag_for(
            {"served_model": "evil token; rm -rf /"}) is None


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


def test_boundary_scan_catches_planted_string(tmp_path, monkeypatch):
    """Mutation check (DoD #7): plant a forbidden string in a file the
    scanner globs and run the REAL scanner — a scanner regression (broken
    glob, broken pattern match, or a vacuous allow-list) fails this test.

    The planted file lives under a temp ``lapis_pm`` package dir that
    replaces the scanner's scan root for the duration of the test; the
    planted string is never committed to the real tree.
    """
    planted_pkg = tmp_path / "lapis_pm"
    planted_pkg.mkdir()
    planted = planted_pkg / "planted.py"
    planted.write_text('X = "Opus reviewer is the gate"\n')
    monkeypatch.setattr(
        "tests.test_provenance_identity._scan_files",
        lambda: sorted(planted_pkg.glob("*.py")),
    )
    hits = _scan_hits()
    assert any("Opus reviewer" in h for h in hits), (
        "boundary scan missed a planted forbidden string — scanner "
        f"regression; hits={hits!r}")
