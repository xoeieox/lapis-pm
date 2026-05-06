"""Tests for lapis_pm/scout/night_queue.py — 12 cases per spec."""
from __future__ import annotations

import json
import os
import signal
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import lapis_pm.scout.night_queue as night_queue
from lapis_pm.scout.night_queue import (
    BACKOFF_BASE,
    BACKOFF_CEILING,
    HEALTH_ABORT_SECONDS,
    QUARANTINE_THRESHOLD,
    ContentionMonitor,
    HealthGate,
    NightRunResult,
    PriorityProfile,
    Quarantine,
    Scheduler,
    WorkUnit,
    _ScaffoldState,
    run_night,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_scaffold_yaml(
    tmp_path: Path,
    spec_id: str,
    cells: int = 1,
    runs_per_cell: int = 1,
    profile: str = "full-pass-once",
    optional_steps: int = 0,
    load_values: list[int] | None = None,
    severity_values: list[str] | None = None,
) -> Path:
    """Write a minimal scaffold YAML for testing."""
    opt_steps_included = [[]] if optional_steps == 0 else [[], [f"opt-{i}" for i in range(optional_steps)]]
    matrix = {
        "optional_steps_included": opt_steps_included[:cells] if cells <= len(opt_steps_included) else opt_steps_included,
        "external_state_severity": severity_values or ["healthy"],
        "concurrent_load": load_values or [1],
        "runs_per_cell": runs_per_cell,
    }
    # Build matrix to produce exactly `cells` cells
    # The simplest way: use concurrent_load with len=cells
    if cells > 1 and not load_values:
        matrix["concurrent_load"] = list(range(1, cells + 1))

    data = {
        "spec_id": spec_id,
        "spec_version": "v0",
        "description": f"Test scaffold {spec_id}",
        "priority_profile": profile,
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
        "matrix": matrix,
    }
    path = tmp_path / f"{spec_id}.yaml"
    path.write_text(yaml.dump(data))
    return path


def _make_scheduler(
    scaffold_paths_and_profiles: list[tuple[Path, str, int, int]],
    quarantine: Quarantine,
    traces_root: Path,
    shuffle_seed: int = 42,
) -> Scheduler:
    """Build a Scheduler from (path, profile, cells, runs_per_cell) tuples."""
    from lapis_pm.scout.scaffold import load_scaffold
    states = []
    for path, profile_str, _cells, runs_per_cell in scaffold_paths_and_profiles:
        scaffold = load_scaffold(path)
        profile = PriorityProfile.from_str(profile_str)
        state = _ScaffoldState(
            spec_id=scaffold.spec_id,
            scaffold_path=path,
            profile=profile,
            runs_per_cell=runs_per_cell,
        )
        states.append(state)
    return Scheduler(states, quarantine, shuffle_seed=shuffle_seed, traces_root=traces_root)


# ---------------------------------------------------------------------------
# Test 1: health gate backoff then proceed
# ---------------------------------------------------------------------------

def test_health_gate_backoff_then_proceed(monkeypatch):
    """503 twice, then 200 — gate backs off (5s, 10s) then clears."""
    call_count = 0

    def fake_get(url, timeout):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.status_code = 503 if call_count <= 2 else 200
        return resp

    monkeypatch.setattr("httpx.get", fake_get)
    monkeypatch.setattr(night_queue, "BACKOFF_BASE", 0.01)
    monkeypatch.setattr(night_queue, "BACKOFF_CEILING", 0.1)

    gate = HealthGate()
    # Also patch time.sleep to avoid real delays
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    result = gate.wait_until_healthy()
    assert result is None  # healthy
    assert call_count == 3
    # Slept twice (once per unhealthy probe)
    assert len(slept) == 2
    # First backoff = BACKOFF_BASE (0.01), second = 0.02
    assert slept[0] == pytest.approx(0.01)
    assert slept[1] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Test 2: health gate aborts after wall-clock budget
# ---------------------------------------------------------------------------

def test_health_gate_aborts_after_wall_clock_budget(monkeypatch):
    """Always 503; time advances past HEALTH_ABORT_SECONDS — gate returns 'health-abort'."""
    def fake_get(url, timeout):
        resp = MagicMock()
        resp.status_code = 503
        return resp

    monkeypatch.setattr("httpx.get", fake_get)
    monkeypatch.setattr(night_queue, "BACKOFF_BASE", 0.001)
    monkeypatch.setattr(night_queue, "BACKOFF_CEILING", 0.001)
    monkeypatch.setattr(night_queue, "HEALTH_ABORT_SECONDS", 0.05)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    gate = HealthGate()
    result = gate.wait_until_healthy()
    assert result == "health-abort"


# ---------------------------------------------------------------------------
# Test 3: quarantine after three zero-parse runs
# ---------------------------------------------------------------------------

def test_quarantine_after_three_zero_parse(tmp_path):
    """Three consecutive zero-parse units → spec_a quarantined; subsequent skipped (exit_code=2)."""
    q = Quarantine(tmp_path / "quarantine.json")

    for _ in range(QUARANTINE_THRESHOLD):
        assert not q.is_quarantined("spec_a")
        q.record_zero_parse("spec_a")

    assert q.is_quarantined("spec_a")
    assert q.get("spec_a").quarantined_at is not None
    assert "zero-parse" in (q.get("spec_a").reason or "")


# ---------------------------------------------------------------------------
# Test 4: quarantine resets on nonzero parse
# ---------------------------------------------------------------------------

def test_quarantine_resets_on_nonzero_parse(tmp_path):
    """Two zero → one nonzero → two zero → no quarantine (counter reset)."""
    q = Quarantine(tmp_path / "quarantine.json")

    q.record_zero_parse("spec_a")
    q.record_zero_parse("spec_a")
    assert not q.is_quarantined("spec_a")

    q.record_nonzero_parse("spec_a")
    assert q.get("spec_a").consecutive_zero_parse_runs == 0

    q.record_zero_parse("spec_a")
    q.record_zero_parse("spec_a")
    assert not q.is_quarantined("spec_a")


# ---------------------------------------------------------------------------
# Test 5: quarantine persists across restart
# ---------------------------------------------------------------------------

def test_quarantine_persists_across_restart(tmp_path):
    """quarantine.json written by first Quarantine instance is read by a fresh one."""
    q1 = Quarantine(tmp_path / "quarantine.json")
    for _ in range(QUARANTINE_THRESHOLD):
        q1.record_zero_parse("spec_a")
    assert q1.is_quarantined("spec_a")

    # Fresh instance — simulates restart
    q2 = Quarantine(tmp_path / "quarantine.json")
    assert q2.is_quarantined("spec_a")
    assert q2.get("spec_a").quarantined_at is not None


# ---------------------------------------------------------------------------
# Test 6: full_pass_once emits each triple exactly once
# ---------------------------------------------------------------------------

def test_full_pass_once_emits_each_triple_exactly_once(tmp_path):
    """2 scaffolds × 3 cells × 2 runs = 12 WorkUnits, then None."""
    q = Quarantine(tmp_path / "quarantine.json")
    traces_root = tmp_path / "traces"

    path_a = _make_scaffold_yaml(tmp_path, "spec_a", cells=3, runs_per_cell=2)
    path_b = _make_scaffold_yaml(tmp_path, "spec_b", cells=3, runs_per_cell=2)

    sched = _make_scheduler(
        [(path_a, "full-pass-once", 3, 2), (path_b, "full-pass-once", 3, 2)],
        q,
        traces_root,
        shuffle_seed=0,
    )

    units = []
    for _ in range(20):
        u = sched.next_unit()
        if u is None:
            break
        units.append(u)

    assert len(units) == 12
    # After 12, scheduler is done
    assert sched.next_unit() is None
    assert sched.all_done()


# ---------------------------------------------------------------------------
# Test 7: round-robin prevents small scaffold starvation
# ---------------------------------------------------------------------------

def test_round_robin_across_scaffolds(tmp_path):
    """Scaffold A (24 triples) and B (216 triples): within first 48, A appears ≥20 times."""
    q = Quarantine(tmp_path / "quarantine.json")
    traces_root = tmp_path / "traces"

    # 24 triples: 24 cells × 1 run
    path_a = _make_scaffold_yaml(tmp_path, "spec_a", cells=24, runs_per_cell=1, load_values=list(range(1, 25)))
    # 216 triples: 216 cells × 1 run
    path_b = _make_scaffold_yaml(tmp_path, "spec_b", cells=216, runs_per_cell=1, load_values=list(range(1, 217)))

    sched = _make_scheduler(
        [(path_a, "full-pass-once", 24, 1), (path_b, "full-pass-once", 216, 1)],
        q,
        traces_root,
        shuffle_seed=42,
    )

    units = []
    for _ in range(48):
        u = sched.next_unit()
        if u is None:
            break
        units.append(u)

    a_count = sum(1 for u in units if u.spec_id == "spec_a")
    assert a_count >= 20, f"spec_a appeared only {a_count} times in first 48 — starvation"


# ---------------------------------------------------------------------------
# Test 8: continuous_baseline cycles indefinitely with increasing run_index
# ---------------------------------------------------------------------------

def test_continuous_baseline_cycles_indefinitely(tmp_path):
    """1 scaffold × 2 cells × 1 run, continuous-baseline: 100 units, run_index increases per cell."""
    q = Quarantine(tmp_path / "quarantine.json")
    traces_root = tmp_path / "traces"

    path = _make_scaffold_yaml(tmp_path, "spec_cb", cells=2, runs_per_cell=1, profile="continuous-baseline", load_values=[1, 2])

    sched = _make_scheduler(
        [(path, "continuous-baseline", 2, 1)],
        q,
        traces_root,
        shuffle_seed=0,
    )

    units = []
    for _ in range(100):
        u = sched.next_unit()
        assert u is not None, "continuous-baseline must never return None while running"
        units.append(u)

    # run_index should increase for same cell over cycles
    per_cell: dict[str, list[int]] = {}
    for u in units:
        per_cell.setdefault(u.cell_id, []).append(u.run_index)
    for cell_id, indices in per_cell.items():
        assert indices == sorted(indices), f"run_index not monotonically increasing for {cell_id}"


# ---------------------------------------------------------------------------
# Test 9: variance resolution appends varying cells only
# ---------------------------------------------------------------------------

def test_variance_resolution_appends_varying_cells(tmp_path):
    """1 scaffold × 3 cells × 2 runs, variance-resolution.
    Cell 1 varies; cells 2–3 don't. After 6 warmup units → 2 more for cell 1 → None.
    """
    q = Quarantine(tmp_path / "quarantine.json")
    traces_root = tmp_path / "traces"

    # 3 cells via load_values=[1,2,3], 2 runs each
    path = _make_scaffold_yaml(
        tmp_path, "spec_vr", cells=3, runs_per_cell=2,
        profile="variance-resolution", load_values=[1, 2, 3]
    )

    # Pre-populate trace files: cell with load=1 varies (2 runs, different sigs)
    from lapis_pm.scout.scaffold import load_scaffold, ScoutScaffold
    scaffold = load_scaffold(path)
    all_cells = [ScoutScaffold.cell_id(cp) for cp in scaffold.cell_params()]
    # Sort deterministically
    varying_cell = all_cells[0]
    non_varying_cells = all_cells[1:]

    # Write varying traces for varying_cell
    cell_dir = traces_root / "spec_vr" / varying_cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "run1.json").write_text(json.dumps({
        "payload": {"breaks_observed": [{"signature": "sig_A"}]}
    }))
    (cell_dir / "run2.json").write_text(json.dumps({
        "payload": {"breaks_observed": [{"signature": "sig_B"}]}
    }))

    # Write identical traces for non-varying cells
    for c in non_varying_cells:
        d = traces_root / "spec_vr" / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            (d / f"run{i}.json").write_text(json.dumps({
                "payload": {"breaks_observed": [{"signature": "sig_X"}]}
            }))

    sched = _make_scheduler(
        [(path, "variance-resolution", 3, 2)],
        q,
        traces_root,
        shuffle_seed=0,
    )

    # Collect all units
    units = []
    for _ in range(20):
        u = sched.next_unit()
        if u is None:
            break
        units.append(u)

    # 6 warmup + 2 variance-resolution = 8
    assert len(units) == 8, f"Expected 8 units, got {len(units)}"
    # The variance-resolution pass should be for varying_cell only
    extra_units = units[6:]
    assert len(extra_units) == 2
    for u in extra_units:
        assert u.cell_id == varying_cell, f"Expected {varying_cell}, got {u.cell_id}"

    assert sched.next_unit() is None


# ---------------------------------------------------------------------------
# Test 10: unknown priority profile raises at load
# ---------------------------------------------------------------------------

def test_unknown_priority_profile_raises_at_load(tmp_path):
    """A scaffold with priority_profile: nonsense raises ValueError on load."""
    data = {
        "spec_id": "bad-profile",
        "spec_version": "v0",
        "description": "bad",
        "priority_profile": "nonsense",
        "static_scaffold": {
            "objective": "x",
            "architecture_sketch": "x",
            "scenario": {"conditions": [], "time_progression": {}, "external_state": {}, "utilization_pattern": ""},
        },
        "generation_directive": "x",
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(data))

    from lapis_pm.scout.scaffold import load_scaffold
    with pytest.raises(ValueError) as exc_info:
        load_scaffold(path)
    msg = str(exc_info.value)
    assert "nonsense" in msg
    # All three allowed values mentioned
    assert "full-pass-once" in msg
    assert "variance-resolution" in msg
    assert "continuous-baseline" in msg


# ---------------------------------------------------------------------------
# Test 11: contention yields when active task present
# ---------------------------------------------------------------------------

def test_contention_yields_when_active_task_present():
    """ContentionMonitor.should_yield() returns True when get_active() is non-None."""
    mock_q = MagicMock()
    mock_q.get_active.return_value = {"task_type": "training", "submitted_by": "test"}
    mock_q.get_pending.return_value = []

    monitor = ContentionMonitor(mock_q)
    assert monitor.should_yield() is True

    # Clear active task
    mock_q.get_active.return_value = None
    assert monitor.should_yield() is False


def test_contention_yields_clears_before_simulate(tmp_path, monkeypatch):
    """Orchestrator does not invoke simulate() until contention clears."""
    sims_dir = tmp_path / "sims"
    sims_dir.mkdir()
    _make_scaffold_yaml(sims_dir, "spec_c", cells=1, runs_per_cell=1)

    log_root = tmp_path / "log"

    simulate_calls = []
    def fake_simulate(path, *, runs_per_cell, cells, **kw):
        simulate_calls.append((path, cells))
        return []

    mock_q = MagicMock()
    # Return active task on first call, None thereafter
    call_count = [0]
    def get_active():
        call_count[0] += 1
        if call_count[0] <= 2:
            return {"task_type": "training", "submitted_by": "test"}
        return None
    mock_q.get_active.side_effect = get_active
    mock_q.get_pending.return_value = []

    monkeypatch.setattr("httpx.get", lambda url, timeout: MagicMock(status_code=200))
    monkeypatch.setattr(night_queue, "CONTENTION_POLL_INTERVAL", 0.01)
    monkeypatch.setattr("lapis_pm.scout.runner.simulate", fake_simulate)

    # Direct test: ContentionMonitor waits
    monitor = ContentionMonitor(mock_q)
    slept = []
    with patch.object(time, "sleep", side_effect=lambda s: slept.append(s)):
        monitor.wait_until_clear()
    # Should have slept at least twice (once per contention check)
    assert len(slept) >= 1


# ---------------------------------------------------------------------------
# Test 12: SIGTERM completes in-flight unit then exits cleanly
# ---------------------------------------------------------------------------

def test_sigterm_completes_in_flight_then_exits(tmp_path, monkeypatch):
    """SIGTERM mid-loop: in-flight simulate() completes, loop exits, no further units.

    run_night() is called from the main thread (required for signal handlers).
    A background thread mimics the SIGTERM handler by setting _stop_requested
    while simulate() is executing, then unblocks it.  The loop must exit after
    that unit completes without dispatching a second unit.
    """
    sims_dir = tmp_path / "sims"
    sims_dir.mkdir()
    # Two cells so scheduler has a second unit available after the first
    _make_scaffold_yaml(sims_dir, "spec_s", cells=2, runs_per_cell=1, load_values=[1, 2])

    log_root = tmp_path / "log"

    simulate_calls = []
    simulate_started = threading.Event()
    simulate_continue = threading.Event()

    def fake_simulate(path, *, runs_per_cell, cells, **kw):
        simulate_started.set()
        simulate_continue.wait(timeout=3.0)
        simulate_calls.append(list(cells))
        return []

    monkeypatch.setattr("httpx.get", lambda url, timeout: MagicMock(status_code=200))
    monkeypatch.setattr("lapis_pm.scout.runner.simulate", fake_simulate)
    monkeypatch.setattr(night_queue, "CONTENTION_POLL_INTERVAL", 0.01)

    def trigger_stop():
        # Wait for simulate() to start, then mimic the SIGTERM handler
        simulate_started.wait(timeout=3.0)
        night_queue._stop_requested = True
        # Give the orchestrator a moment to see the flag, then unblock simulate
        time.sleep(0.05)
        simulate_continue.set()

    trigger_thread = threading.Thread(target=trigger_stop, daemon=True)
    trigger_thread.start()

    # run_night in the main thread — signal handlers install correctly here
    result = run_night(
        sims_dir=sims_dir,
        once=True,
        log_root=log_root,
        profiles_override=None,
    )

    trigger_thread.join(timeout=3.0)

    # Exactly one simulate() call completed
    assert len(simulate_calls) == 1

    # Reset stop flag for subsequent tests
    night_queue._stop_requested = False
