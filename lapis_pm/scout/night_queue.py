"""Lapis Scout — night-queue orchestrator.

Replaces the sequential bash loop (scripts/scout_night.sh) with a Python
orchestrator that gates on llama-server health, quarantines bad scaffolds,
schedules cells in round-robin across scaffolds, and yields to GPU contention.

Public surface::

    run_night(sims_dir, until_epoch=None, once=False, log_root=None,
              profiles_override=None) -> NightRunResult

See the spec at /srv/lapis/planning/specs/lapis-scout-night-queue-v0.md.
"""
from __future__ import annotations

import enum
import json
import logging
import os
import random
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (tunable in v1 if needed; NOT config in v0)
# ---------------------------------------------------------------------------

QUARANTINE_THRESHOLD = 3          # consecutive zero-parse runs → quarantine
HEALTH_ABORT_SECONDS = 3600       # 1 hour continuous unhealthy → abort night run
BACKOFF_BASE = 5.0                # seconds; base for exponential health backoff
BACKOFF_CEILING = 120.0           # seconds; max backoff interval
CONTENTION_POLL_INTERVAL = 5      # seconds between contention re-checks

# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_stop_requested: bool = False


def _install_signal_handlers() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        global _stop_requested
        _stop_requested = True
        log.info("signal %s received — will exit after current WorkUnit completes", signum)

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        # signal.signal() can only be called from the main thread; skip gracefully
        # when run_night is invoked from a non-main thread (e.g. tests).
        log.debug("_install_signal_handlers: not in main thread, skipping signal installation")


# ---------------------------------------------------------------------------
# PriorityProfile
# ---------------------------------------------------------------------------

class PriorityProfile(enum.Enum):
    FULL_PASS_ONCE = "full-pass-once"
    VARIANCE_RESOLUTION = "variance-resolution"
    CONTINUOUS_BASELINE = "continuous-baseline"

    @classmethod
    def from_str(cls, value: str) -> "PriorityProfile":
        for member in cls:
            if member.value == value:
                return member
        allowed = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown priority_profile {value!r}. Allowed: {allowed}"
        )


# ---------------------------------------------------------------------------
# WorkUnit
# ---------------------------------------------------------------------------

@dataclass
class WorkUnit:
    spec_id: str
    cell_id: str
    run_index: int
    scaffold_path: Path


# ---------------------------------------------------------------------------
# QuarantineEntry + Quarantine state
# ---------------------------------------------------------------------------

@dataclass
class QuarantineEntry:
    consecutive_zero_parse_runs: int = 0
    quarantined_at: float | None = None
    reason: str | None = None

    def is_quarantined(self) -> bool:
        return self.quarantined_at is not None


class Quarantine:
    """In-memory quarantine state, persisted to quarantine.json on every change."""

    def __init__(self, quarantine_path: Path) -> None:
        self._path = quarantine_path
        self._entries: dict[str, QuarantineEntry] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                for spec_id, entry in raw.items():
                    self._entries[spec_id] = QuarantineEntry(
                        consecutive_zero_parse_runs=entry.get("consecutive_zero_parse_runs", 0),
                        quarantined_at=entry.get("quarantined_at"),
                        reason=entry.get("reason"),
                    )
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Failed to load quarantine.json: %s", exc)

    def _persist(self) -> None:
        data = {
            spec_id: {
                "consecutive_zero_parse_runs": e.consecutive_zero_parse_runs,
                "quarantined_at": e.quarantined_at,
                "reason": e.reason,
            }
            for spec_id, e in self._entries.items()
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._path)

    def get(self, spec_id: str) -> QuarantineEntry:
        if spec_id not in self._entries:
            self._entries[spec_id] = QuarantineEntry()
        return self._entries[spec_id]

    def is_quarantined(self, spec_id: str) -> bool:
        return self.get(spec_id).is_quarantined()

    def record_zero_parse(self, spec_id: str) -> None:
        entry = self.get(spec_id)
        entry.consecutive_zero_parse_runs += 1
        if (
            not entry.is_quarantined()
            and entry.consecutive_zero_parse_runs >= QUARANTINE_THRESHOLD
        ):
            entry.quarantined_at = time.time()
            entry.reason = f"{QUARANTINE_THRESHOLD} consecutive zero-parse runs"
            log.warning(
                "QUARANTINE: spec_id=%s quarantined after %d consecutive zero-parse runs",
                spec_id,
                entry.consecutive_zero_parse_runs,
            )
        self._persist()

    def record_nonzero_parse(self, spec_id: str) -> None:
        entry = self.get(spec_id)
        if entry.consecutive_zero_parse_runs > 0:
            entry.consecutive_zero_parse_runs = 0
            self._persist()

    def clear(self, spec_id: str) -> bool:
        """Remove quarantine entry; returns True if it existed."""
        if spec_id in self._entries:
            del self._entries[spec_id]
            self._persist()
            return True
        return False

    def all_entries(self) -> dict[str, QuarantineEntry]:
        return dict(self._entries)


# ---------------------------------------------------------------------------
# HealthGate
# ---------------------------------------------------------------------------

def _derive_health_url() -> str:
    """Derive health URL from lapis_engine.adapters.LLAMA_SERVER_URL at runtime.

    The URL is the source of truth — do not duplicate the literal here.
    An env-var override (LAPIS_SCOUT_HEALTH_URL) wins when set.
    """
    override = os.environ.get("LAPIS_SCOUT_HEALTH_URL", "")
    if override:
        return override
    try:
        from lapis_engine.adapters import LLAMA_SERVER_URL  # type: ignore[import]
        base = LLAMA_SERVER_URL
    except ImportError as exc:
        raise RuntimeError(
            "lapis_engine is unavailable — cannot derive health URL. "
            "Set the LAPIS_SCOUT_HEALTH_URL env var to override."
        ) from exc
    # Strip known suffix if present; fall back to simple rstrip approach.
    SUFFIX = "/v1/chat/completions"
    if base.endswith(SUFFIX):
        base = base[: -len(SUFFIX)]
    return base.rstrip("/") + "/health"


class HealthGate:
    """Checks llama-server health before each WorkUnit.

    Implements exponential backoff (base BACKOFF_BASE, ceiling BACKOFF_CEILING)
    and a wall-clock abort budget (HEALTH_ABORT_SECONDS).
    """

    def __init__(self) -> None:
        self._health_url = _derive_health_url()
        self._backoff_interval = BACKOFF_BASE
        self._first_unhealthy_at: float | None = None

    def _probe(self) -> bool:
        try:
            r = httpx.get(self._health_url, timeout=5)
            return r.status_code == 200
        except Exception:  # connection refused, timeout, etc.
            return False

    def wait_until_healthy(self) -> str | None:
        """Block until healthy or abort budget exceeded.

        Returns None when healthy (caller may proceed).
        Returns an error string when the wall-clock budget expires.
        """
        while True:
            if _stop_requested:
                return "stop_requested"
            healthy = self._probe()
            if healthy:
                if self._first_unhealthy_at is not None:
                    log.info("llama-server healthy again (was down for %.0fs)", time.time() - self._first_unhealthy_at)
                self._first_unhealthy_at = None
                self._backoff_interval = BACKOFF_BASE
                return None

            # Unhealthy path
            now = time.time()
            if self._first_unhealthy_at is None:
                self._first_unhealthy_at = now
                log.warning("llama-server unhealthy at %s — backing off %.0fs", self._health_url, self._backoff_interval)
            else:
                elapsed = now - self._first_unhealthy_at
                if elapsed >= HEALTH_ABORT_SECONDS:
                    log.error(
                        "llama-server has been down for %.0fs (>= %ds) — aborting night run",
                        elapsed,
                        HEALTH_ABORT_SECONDS,
                    )
                    return "health-abort"
                log.warning("llama-server still unhealthy (%.0fs elapsed) — backing off %.0fs", elapsed, self._backoff_interval)

            time.sleep(self._backoff_interval)
            self._backoff_interval = min(self._backoff_interval * 2, BACKOFF_CEILING)


# ---------------------------------------------------------------------------
# ContentionMonitor
# ---------------------------------------------------------------------------

class ContentionMonitor:
    """Read-only consumer of GPUQueue state for contention detection.

    The night queue does NOT submit tasks to GPUQueue.
    """

    def __init__(self, gpu_queue: Any) -> None:
        self._q = gpu_queue

    def should_yield(self) -> bool:
        """Return True if the GPU is busy with non-Scout work."""
        try:
            active = self._q.get_active()
            if active is not None:
                return True
            pending = self._q.get_pending()
            # Yield if any pending task has priority < IDLE (i.e. more urgent)
            from agents_core.gpu import Priority  # type: ignore[import]
            for task in pending:
                if task.get("priority", Priority.IDLE) < Priority.IDLE:
                    return True
        except Exception as exc:
            log.debug("ContentionMonitor probe failed (ignoring): %s", exc)
        return False

    def wait_until_clear(self) -> None:
        """Block until GPU is idle or stop is requested."""
        first_yield = True
        while self.should_yield() and not _stop_requested:
            if first_yield:
                try:
                    active = self._q.get_active()
                    task_type = active.get("task_type", "?") if active else "?"
                    submitted_by = active.get("submitted_by", "?") if active else "?"
                    log.info("GPU contention: yielding (active task_type=%s submitted_by=%s)", task_type, submitted_by)
                except Exception:
                    log.info("GPU contention: yielding")
                first_yield = False
            time.sleep(CONTENTION_POLL_INTERVAL)
        if not first_yield:
            log.info("GPU contention cleared — resuming night run")


# ---------------------------------------------------------------------------
# Variance computation helpers
# ---------------------------------------------------------------------------

def _cell_varies(spec_id: str, cell_id: str, traces_root: Path) -> bool:
    """Return True if the cell produced ≥2 runs with different break signature sets.

    Reads trace JSON files from traces_root/<spec_id>/<cell_id>/*.json.
    Malformed files are skipped (best-effort; non-varying on error).
    Variance is a literal frozenset comparison — no _tokenize import.
    """
    cell_dir = traces_root / spec_id / cell_id
    if not cell_dir.exists():
        return False

    run_sig_sets: list[frozenset] = []
    for trace_file in sorted(cell_dir.glob("*.json")):
        try:
            data = json.loads(trace_file.read_text())
            payload = data.get("payload", {})
            breaks = payload.get("breaks_observed", [])
            sigs = frozenset(b["signature"] for b in breaks if "signature" in b)
            run_sig_sets.append(sigs)
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    if len(run_sig_sets) < 2:
        return False
    return len(set(run_sig_sets)) >= 2


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class _ScaffoldState:
    spec_id: str
    scaffold_path: Path
    profile: PriorityProfile
    runs_per_cell: int
    # Ordered list of (cell_id, run_index) pairs for the current pass
    worklist: list[tuple[str, int]] = field(default_factory=list)
    # Pointer into worklist for round-robin
    pointer: int = 0
    # Whether we've completed the warmup pass (for variance-resolution)
    warmup_done: bool = False
    # For continuous-baseline: cycle counter per cell
    cycle_counters: dict[str, int] = field(default_factory=dict)
    done: bool = False


class Scheduler:
    """Owns the worklist; emits WorkUnits in round-robin across scaffolds.

    Profile-aware: full-pass-once, variance-resolution, continuous-baseline.
    """

    def __init__(
        self,
        scaffold_states: list[_ScaffoldState],
        quarantine: Quarantine,
        shuffle_seed: int | None = None,
        traces_root: Path | None = None,
    ) -> None:
        self._states = scaffold_states
        self._quarantine = quarantine
        self._rng = random.Random(shuffle_seed if shuffle_seed is not None else int(time.time()))
        self._traces_root = traces_root or Path("/srv/lapis/scout/traces")
        self._scaffold_index = 0  # round-robin cursor

        # Initialize worklists
        for state in self._states:
            self._build_initial_worklist(state)

    def _build_initial_worklist(self, state: _ScaffoldState) -> None:
        """Build the worklist for a scaffold's first pass."""
        from .scaffold import load_scaffold, ScoutScaffold
        scaffold = load_scaffold(state.scaffold_path)
        all_cells = [ScoutScaffold.cell_id(cp) for cp in scaffold.cell_params()]
        self._rng.shuffle(all_cells)
        entries: list[tuple[str, int]] = []
        if state.profile == PriorityProfile.CONTINUOUS_BASELINE:
            # Build initial cycle
            for cell_id in all_cells:
                for run_i in range(state.runs_per_cell):
                    entries.append((cell_id, run_i))
        else:
            # full-pass-once and variance-resolution: one pass
            for cell_id in all_cells:
                for run_i in range(state.runs_per_cell):
                    entries.append((cell_id, run_i))
        state.worklist = entries
        state.pointer = 0

    def _extend_variance_resolution(self, state: _ScaffoldState) -> None:
        """After warmup pass completes, append variance-resolution cells."""
        from .scaffold import load_scaffold, ScoutScaffold
        scaffold = load_scaffold(state.scaffold_path)
        all_cells = [ScoutScaffold.cell_id(cp) for cp in scaffold.cell_params()]
        varying_cells = [c for c in all_cells if _cell_varies(state.spec_id, c, self._traces_root)]
        self._rng.shuffle(varying_cells)
        additional: list[tuple[str, int]] = []
        for cell_id in varying_cells:
            for run_i in range(state.runs_per_cell):
                additional.append((cell_id, run_i))
        if additional:
            log.info(
                "variance-resolution: %s — appending %d units for %d varying cell(s)",
                state.spec_id,
                len(additional),
                len(varying_cells),
            )
        state.worklist.extend(additional)
        state.warmup_done = True

    def _next_for_state(self, state: _ScaffoldState) -> WorkUnit | None:
        """Pull next unit from this scaffold's worklist."""
        if state.done:
            return None
        if self._quarantine.is_quarantined(state.spec_id):
            return None

        if state.pointer >= len(state.worklist):
            # Worklist exhausted
            if state.profile == PriorityProfile.CONTINUOUS_BASELINE:
                # Replenish worklist for next cycle
                from .scaffold import load_scaffold, ScoutScaffold
                scaffold = load_scaffold(state.scaffold_path)
                all_cells = [ScoutScaffold.cell_id(cp) for cp in scaffold.cell_params()]
                self._rng.shuffle(all_cells)
                new_entries: list[tuple[str, int]] = []
                for cell_id in all_cells:
                    cycle = state.cycle_counters.get(cell_id, 0) + 1
                    state.cycle_counters[cell_id] = cycle
                    for run_i in range(state.runs_per_cell):
                        run_index = cycle * state.runs_per_cell + run_i
                        new_entries.append((cell_id, run_index))
                state.worklist.extend(new_entries)
                # Don't reset pointer — just keep advancing through extended list
            elif state.profile == PriorityProfile.VARIANCE_RESOLUTION and not state.warmup_done:
                # Warmup pass done; trigger variance extension
                self._extend_variance_resolution(state)
                if state.pointer >= len(state.worklist):
                    state.done = True
                    return None
            else:
                state.done = True
                return None

        cell_id, run_index = state.worklist[state.pointer]
        state.pointer += 1
        return WorkUnit(
            spec_id=state.spec_id,
            cell_id=cell_id,
            run_index=run_index,
            scaffold_path=state.scaffold_path,
        )

    def next_unit(self, now: int | None = None) -> WorkUnit | None:  # noqa: ARG002
        """Return next WorkUnit in round-robin across non-done scaffolds, or None."""
        non_done = [s for s in self._states if not s.done and not self._quarantine.is_quarantined(s.spec_id)]
        if not non_done:
            return None

        # Round-robin: try each scaffold up to len(non_done) times
        for _ in range(len(self._states)):
            if not self._states:
                break
            self._scaffold_index = self._scaffold_index % len(self._states)
            state = self._states[self._scaffold_index]
            self._scaffold_index += 1

            if state.done or self._quarantine.is_quarantined(state.spec_id):
                continue

            unit = self._next_for_state(state)
            if unit is not None:
                return unit

        # All non-quarantined non-done scaffolds produced None (edge case)
        all_done = all(
            s.done or self._quarantine.is_quarantined(s.spec_id)
            for s in self._states
        )
        return None if all_done else None

    def all_done(self) -> bool:
        return all(
            s.done or self._quarantine.is_quarantined(s.spec_id)
            for s in self._states
        )


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

class ManifestWriter:
    """Writes TSV manifest rows.

    Header: started_at_local  spec_id  cell_id  run_index  duration_s
            exit_code  successful_tick_count  total_tick_count  log_path
    """

    HEADER = "\t".join([
        "started_at_local", "spec_id", "cell_id", "run_index",
        "duration_s", "exit_code", "successful_tick_count",
        "total_tick_count", "log_path",
    ])

    def __init__(self, path: Path, run_tag: str, shuffle_seed: int) -> None:
        self._path = path
        self._file = open(path, "w", buffering=1)  # line-buffered
        # Write header + metadata comment
        self._file.write(f"# night-run run_tag={run_tag} shuffle_seed={shuffle_seed}\n")
        self._file.write(self.HEADER + "\n")
        self._file.flush()

    def write_row(
        self,
        *,
        started_at_local: str,
        spec_id: str,
        cell_id: str,
        run_index: int,
        duration_s: float,
        exit_code: int,
        successful_tick_count: int,
        total_tick_count: int,
        log_path: str,
    ) -> None:
        row = "\t".join([
            started_at_local, spec_id, cell_id, str(run_index),
            f"{duration_s:.1f}", str(exit_code),
            str(successful_tick_count), str(total_tick_count), log_path,
        ])
        self._file.write(row + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# NightRunResult
# ---------------------------------------------------------------------------

@dataclass
class NightRunResult:
    total_units: int = 0
    completed_ok: int = 0
    skipped_quarantine: int = 0
    skipped_health: int = 0
    skipped_contention: int = 0
    errored: int = 0
    aborted: bool = False
    abort_reason: str = ""
    log_root: Path | None = None
    manifest_path: Path | None = None


# ---------------------------------------------------------------------------
# Trace inspection
# ---------------------------------------------------------------------------

def _read_trace_counts(trace_paths: list[Path]) -> tuple[int, int]:
    """Return (successful_tick_count, total_tick_count) from trace files.

    successful_tick_count = total_tick_count - parse_failure_count.
    Returns (0, 0) if no traces readable.
    """
    total_ticks = 0
    total_parse_failures = 0
    for p in trace_paths:
        try:
            data = json.loads(p.read_text())
            payload = data.get("payload", {})
            t = int(payload.get("total_tick_count", 0))
            pf = int(payload.get("parse_failure_count", 0))
            total_ticks += t
            total_parse_failures += pf
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    successful = total_ticks - total_parse_failures
    return successful, total_ticks


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_night(
    sims_dir: Path,
    until_epoch: int | None = None,
    once: bool = False,
    log_root: Path | None = None,
    profiles_override: dict[str, str] | None = None,
) -> NightRunResult:
    """Run the night-queue orchestrator.

    Parameters
    ----------
    sims_dir:
        Directory containing scaffold YAML files.
    until_epoch:
        Stop after this Unix timestamp. None = run --once mode.
    once:
        If True, run each non-continuous-baseline scaffold once then exit.
    log_root:
        Directory for manifest.tsv and quarantine.json.
        Defaults to /tmp/scout-night-<run_tag>/.
    profiles_override:
        Map spec_id → profile name, overriding the scaffold's declared profile.
        Intended for testing/smoke.
    """
    global _stop_requested
    _stop_requested = False
    _install_signal_handlers()

    result = NightRunResult()

    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    if log_root is None:
        log_root = Path(f"/tmp/scout-night-{run_tag}")
    log_root.mkdir(parents=True, exist_ok=True)
    result.log_root = log_root

    quarantine = Quarantine(log_root / "quarantine.json")

    shuffle_seed = int(time.time())
    manifest_path = log_root / "manifest.tsv"
    manifest = ManifestWriter(manifest_path, run_tag, shuffle_seed)
    result.manifest_path = manifest_path

    # Enumerate scaffolds
    sims_dir = Path(sims_dir)
    scaffold_paths = sorted(sims_dir.glob("*.yaml"))
    if not scaffold_paths:
        log.warning("No scaffold YAMLs found in %s", sims_dir)
        manifest.close()
        return result

    # Build scaffold states
    from .scaffold import load_scaffold
    scaffold_states: list[_ScaffoldState] = []
    for path in scaffold_paths:
        try:
            scaffold = load_scaffold(path)
        except Exception as exc:
            log.error("Failed to load scaffold %s: %s — skipping", path, exc)
            continue
        spec_id = scaffold.spec_id

        # Determine profile
        if profiles_override and spec_id in profiles_override:
            profile_str = profiles_override[spec_id]
        else:
            profile_str = scaffold.priority_profile
        try:
            profile = PriorityProfile.from_str(profile_str)
        except ValueError as exc:
            log.error("%s — skipping scaffold %s", exc, spec_id)
            continue

        state = _ScaffoldState(
            spec_id=spec_id,
            scaffold_path=path,
            profile=profile,
            runs_per_cell=scaffold.matrix.runs_per_cell,
        )
        scaffold_states.append(state)

    if not scaffold_states:
        log.warning("No valid scaffolds loaded.")
        manifest.close()
        return result

    scheduler = Scheduler(scaffold_states, quarantine, shuffle_seed=shuffle_seed)

    # Instantiate infrastructure
    health_gate = HealthGate()
    try:
        from agents_core.gpu import GPUQueue  # type: ignore[import]
        contention_monitor = ContentionMonitor(GPUQueue())
    except ImportError:
        log.warning("agents_core.gpu not available — contention monitoring disabled")
        contention_monitor = None

    # Deferred import of runner (avoid circular at module load; keep reference on
    # the module so tests can monkeypatch lapis_pm.scout.runner.simulate)
    from . import runner as _runner

    _quarantine_logged: set[str] = set()

    while not _stop_requested:
        # Check time budget
        if until_epoch is not None and time.time() >= until_epoch:
            log.info("Reached until_epoch — night run complete.")
            break
        if scheduler.all_done():
            if once:
                log.info("--once: all scaffolds done — night run complete.")
            elif until_epoch is not None:
                log.info("All scaffolds done — exiting before until_epoch.")
            break

        unit = scheduler.next_unit(now=int(time.time()))
        if unit is None:
            if once:
                break
            if until_epoch is None:
                break
            # Continuous-baseline scaffolds or waiting — sleep briefly
            time.sleep(1)
            continue

        spec_id = unit.spec_id

        # Quarantine skip
        if quarantine.is_quarantined(spec_id):
            if spec_id not in _quarantine_logged:
                log.info("quarantine-skip: spec_id=%s", spec_id)
                _quarantine_logged.add(spec_id)
            started_at = datetime.now().strftime("%Y%m%dT%H%M%S")
            manifest.write_row(
                started_at_local=started_at,
                spec_id=spec_id,
                cell_id=unit.cell_id,
                run_index=unit.run_index,
                duration_s=0.0,
                exit_code=2,
                successful_tick_count=0,
                total_tick_count=0,
                log_path="",
            )
            result.skipped_quarantine += 1
            result.total_units += 1
            continue

        # Health gate
        abort_reason = health_gate.wait_until_healthy()
        if abort_reason == "stop_requested":
            break
        if abort_reason == "health-abort":
            started_at = datetime.now().strftime("%Y%m%dT%H%M%S")
            manifest.write_row(
                started_at_local=started_at,
                spec_id="health-abort",
                cell_id="",
                run_index=0,
                duration_s=0.0,
                exit_code=3,
                successful_tick_count=0,
                total_tick_count=0,
                log_path="",
            )
            result.aborted = True
            result.abort_reason = "health-abort: llama-server down for 1 hour"
            break

        # Contention yield
        if contention_monitor is not None:
            contention_monitor.wait_until_clear()
        if _stop_requested:
            break

        # Execute WorkUnit
        started_at_local = datetime.now().strftime("%Y%m%dT%H%M%S")
        t0 = time.time()
        exit_code = 0
        successful_tick_count = 0
        total_tick_count = 0
        written_paths: list[Path] = []

        try:
            written_paths = _runner.simulate(
                str(unit.scaffold_path),
                runs_per_cell=1,
                cells=[unit.cell_id],
            )
        except Exception as exc:
            log.error(
                "simulate raised for spec_id=%s cell_id=%s: %s",
                spec_id,
                unit.cell_id,
                exc,
                exc_info=True,
            )
            exit_code = 1
            # Do NOT count toward quarantine — can't distinguish infra vs structural

        duration_s = time.time() - t0

        if exit_code == 0 and written_paths:
            successful_tick_count, total_tick_count = _read_trace_counts(written_paths)
            # Quality gate
            if total_tick_count > 0 and successful_tick_count == 0:
                quarantine.record_zero_parse(spec_id)
            elif total_tick_count > 0:
                quarantine.record_nonzero_parse(spec_id)

        log_path_str = str(written_paths[0]) if written_paths else ""
        manifest.write_row(
            started_at_local=started_at_local,
            spec_id=spec_id,
            cell_id=unit.cell_id,
            run_index=unit.run_index,
            duration_s=duration_s,
            exit_code=exit_code,
            successful_tick_count=successful_tick_count,
            total_tick_count=total_tick_count,
            log_path=log_path_str,
        )

        result.total_units += 1
        if exit_code == 0:
            result.completed_ok += 1
        else:
            result.errored += 1

    manifest.close()
    return result


# ---------------------------------------------------------------------------
# Status helpers (used by CLI)
# ---------------------------------------------------------------------------

def read_status(log_root: Path, as_json: bool = False) -> str:
    """Read the most recent night-run state from manifest + quarantine.json.

    Returns human-readable text or JSON string depending on as_json.

    JSON key set (stable, documented for claude-view/Librarian consumption):
        run_tag, manifest_path, shuffle_seed, started_at, last_row_at,
        units_total, units_completed, units_errored,
        units_skipped_quarantine, units_skipped_health, units_skipped_contention,
        scaffolds: [{spec_id, completed, errored, skipped_quarantine}],
        quarantined: [{spec_id, quarantined_at, reason}]
    """
    manifest_path = log_root / "manifest.tsv"
    quarantine_path = log_root / "quarantine.json"

    if not manifest_path.exists():
        if as_json:
            return json.dumps({"error": f"manifest not found at {manifest_path}"})
        return f"No manifest found at {manifest_path}"

    rows = []
    run_tag = ""
    shuffle_seed = ""
    with open(manifest_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# night-run"):
                for part in line.split():
                    if part.startswith("run_tag="):
                        run_tag = part[8:]
                    if part.startswith("shuffle_seed="):
                        shuffle_seed = part[13:]
            elif line.startswith("started_at_local\t"):
                continue  # header
            elif line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 9:
                    rows.append({
                        "started_at_local": parts[0],
                        "spec_id": parts[1],
                        "cell_id": parts[2],
                        "run_index": parts[3],
                        "duration_s": parts[4],
                        "exit_code": int(parts[5]),
                        "successful_tick_count": int(parts[6]),
                        "total_tick_count": int(parts[7]),
                        "log_path": parts[8],
                    })

    # Aggregate per-scaffold counts
    from collections import defaultdict
    per_scaffold: dict[str, dict] = defaultdict(lambda: {"completed": 0, "errored": 0, "skipped_quarantine": 0})
    total = units_completed = units_errored = units_skipped_q = units_skipped_h = units_skipped_c = 0
    for row in rows:
        ec = row["exit_code"]
        sid = row["spec_id"]
        total += 1
        if ec == 0:
            units_completed += 1
            per_scaffold[sid]["completed"] += 1
        elif ec == 1:
            units_errored += 1
            per_scaffold[sid]["errored"] += 1
        elif ec == 2:
            units_skipped_q += 1
            per_scaffold[sid]["skipped_quarantine"] += 1
        elif ec == 3:
            units_skipped_h += 1
        elif ec == 4:
            units_skipped_c += 1

    # Quarantine state
    quarantined_entries = []
    if quarantine_path.exists():
        try:
            q_data = json.loads(quarantine_path.read_text())
            for spec_id, entry in q_data.items():
                if entry.get("quarantined_at") is not None:
                    quarantined_entries.append({
                        "spec_id": spec_id,
                        "quarantined_at": entry.get("quarantined_at"),
                        "reason": entry.get("reason"),
                    })
        except (json.JSONDecodeError, OSError):
            pass

    # Elapsed
    started_at = rows[0]["started_at_local"] if rows else ""
    last_at = rows[-1]["started_at_local"] if rows else ""

    data = {
        "run_tag": run_tag,
        "manifest_path": str(manifest_path),
        "shuffle_seed": shuffle_seed,
        "started_at": started_at,
        "last_row_at": last_at,
        "units_total": total,
        "units_completed": units_completed,
        "units_errored": units_errored,
        "units_skipped_quarantine": units_skipped_q,
        "units_skipped_health": units_skipped_h,
        "units_skipped_contention": units_skipped_c,
        "scaffolds": [
            {"spec_id": sid, **counts}
            for sid, counts in sorted(per_scaffold.items())
        ],
        "quarantined": quarantined_entries,
    }

    if as_json:
        return json.dumps(data, indent=2)

    # Human-readable text
    lines = [
        f"Night run: {run_tag}",
        f"Manifest:  {manifest_path}",
        f"Started:   {started_at}  Last row: {last_at}",
        f"Units:     {total} total | {units_completed} ok | {units_errored} err "
        f"| {units_skipped_q} quarantined | {units_skipped_h} health | {units_skipped_c} contention",
        "",
        "Per-scaffold:",
    ]
    for sid, counts in sorted(per_scaffold.items()):
        lines.append(
            f"  {sid}: {counts['completed']} ok, {counts['errored']} err, "
            f"{counts['skipped_quarantine']} quar-skip"
        )
    if quarantined_entries:
        lines.append("")
        lines.append("Quarantined:")
        for q in quarantined_entries:
            lines.append(f"  {q['spec_id']}: {q['reason']}")
    return "\n".join(lines)
