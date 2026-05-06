"""CLI smoke tests for lapis-pm scout night commands.

Covers the three cases from the spec Definition of Done §4:
  1. run --once --sims-dir DIR returns 0 and produces a manifest file.
  2. status --json parses as JSON with expected keys.
  3. quarantine-clear removes a quarantined entry idempotently.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from lapis_pm.scout.cli import (
    cmd_scout_night_quarantine_clear,
    cmd_scout_night_run,
    cmd_scout_night_status,
)
from lapis_pm.scout.night_queue import QUARANTINE_THRESHOLD, Quarantine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fixture_scaffold(sims_dir: Path, spec_id: str) -> Path:
    """Write a minimal single-cell scaffold YAML for CLI testing."""
    data = {
        "spec_id": spec_id,
        "spec_version": "v0",
        "description": f"CLI fixture {spec_id}",
        "priority_profile": "full-pass-once",
        "static_scaffold": {
            "objective": "test",
            "architecture_sketch": "test",
            "scenario": {
                "conditions": [],
                "time_progression": {},
                "external_state": {},
                "utilization_pattern": "",
            },
        },
        "generation_directive": "test",
        "matrix": {
            "optional_steps_included": [[]],
            "external_state_severity": ["healthy"],
            "concurrent_load": [1],
            "runs_per_cell": 1,
        },
    }
    path = sims_dir / f"{spec_id}.yaml"
    path.write_text(yaml.dump(data))
    return path


def _make_args(**kwargs):
    """Create a SimpleNamespace args object with defaults matching the CLI."""
    defaults = {
        "sims_dir": None,
        "once": False,
        "until": None,
        "log_root": None,
        "profile_override": [],
        "json": False,
        "spec_id": None,
        "night_sub": None,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Smoke 1: run --once --sims-dir DIR returns 0 + manifest written
# ---------------------------------------------------------------------------

def test_cli_night_run_once_produces_manifest(tmp_path, monkeypatch):
    """cmd_scout_night_run with --once returns 0 and writes manifest.tsv."""
    sims_dir = tmp_path / "sims"
    sims_dir.mkdir()
    _make_fixture_scaffold(sims_dir, "spec_cli_run")
    log_root = tmp_path / "log"

    # Health URL override so _derive_health_url() doesn't need lapis_engine
    monkeypatch.setenv("LAPIS_SCOUT_HEALTH_URL", "http://test-server/health")
    monkeypatch.setattr("httpx.get", lambda url, timeout: MagicMock(status_code=200))
    monkeypatch.setattr(
        "lapis_pm.scout.runner.simulate",
        lambda path, *, runs_per_cell, cells, **kw: [],
    )

    args = _make_args(sims_dir=str(sims_dir), once=True, log_root=str(log_root))
    rc = cmd_scout_night_run(args)

    assert rc == 0, "Expected exit code 0 from cmd_scout_night_run --once"
    manifest = log_root / "manifest.tsv"
    assert manifest.exists(), "manifest.tsv was not written"
    content = manifest.read_text()
    assert "started_at_local" in content, "manifest header row missing"


# ---------------------------------------------------------------------------
# Smoke 2: status --json parses as JSON with expected stable keys
# ---------------------------------------------------------------------------

def test_cli_night_status_json(tmp_path, monkeypatch, capsys):
    """cmd_scout_night_status --json outputs parseable JSON with expected keys."""
    sims_dir = tmp_path / "sims"
    sims_dir.mkdir()
    _make_fixture_scaffold(sims_dir, "spec_cli_status")
    log_root = tmp_path / "log"

    # Bootstrap: produce a real manifest via run --once
    monkeypatch.setenv("LAPIS_SCOUT_HEALTH_URL", "http://test-server/health")
    monkeypatch.setattr("httpx.get", lambda url, timeout: MagicMock(status_code=200))
    monkeypatch.setattr(
        "lapis_pm.scout.runner.simulate",
        lambda path, *, runs_per_cell, cells, **kw: [],
    )
    run_args = _make_args(sims_dir=str(sims_dir), once=True, log_root=str(log_root))
    assert cmd_scout_night_run(run_args) == 0
    capsys.readouterr()  # discard run output before capturing status output

    # Now test status --json
    status_args = _make_args(log_root=str(log_root), json=True)
    rc = cmd_scout_night_status(status_args)
    assert rc == 0, "Expected exit code 0 from cmd_scout_night_status --json"

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)

    expected_keys = [
        "run_tag", "manifest_path", "shuffle_seed", "started_at", "last_row_at",
        "units_total", "units_completed", "units_errored",
        "units_skipped_quarantine", "units_skipped_health", "units_skipped_contention",
        "scaffolds", "quarantined",
    ]
    for key in expected_keys:
        assert key in parsed, f"Expected JSON key {key!r} missing from status output"


# ---------------------------------------------------------------------------
# Smoke 3: quarantine-clear removes entry idempotently
# ---------------------------------------------------------------------------

def test_cli_night_quarantine_clear_idempotent(tmp_path):
    """quarantine-clear removes a quarantined spec; second call is a no-op (exit 0)."""
    log_root = tmp_path / "log"
    log_root.mkdir()

    # Seed a quarantine entry
    q = Quarantine(log_root / "quarantine.json")
    for _ in range(QUARANTINE_THRESHOLD):
        q.record_zero_parse("spec_qc")
    assert q.is_quarantined("spec_qc")

    # First clear — should remove the entry
    args = _make_args(spec_id="spec_qc", log_root=str(log_root))
    rc = cmd_scout_night_quarantine_clear(args)
    assert rc == 0

    # Verify the entry is gone
    q2 = Quarantine(log_root / "quarantine.json")
    assert not q2.is_quarantined("spec_qc")

    # Second clear — idempotent, still exit 0
    rc2 = cmd_scout_night_quarantine_clear(args)
    assert rc2 == 0
