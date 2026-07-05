"""Fixture stub of conductor's night_coordinator.py.

Reproduces the real file's lane-registry shape, including the try/except-guarded
`_register_gpu_lane()` call at module load (real file :300-310) — the exact trap
where a bare `import night_coordinator` succeeds even when gpu_lane.py is absent,
silently leaving the "gpu" lane unregistered. The R4 load-test asserts "gpu" IS in
_LANE_HANDLERS after import, which only holds when gpu_lane.py is present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agents_core.notify import Priority, send_notification  # noqa: F401
from agents_core.slots import IS_MASTER, TERMINAL_STATUSES, SlotStore  # noqa: F401

log = logging.getLogger("night-coordinator")

_LANE_HANDLERS: dict[str, Any] = {}


def register_lane_handler(lane: str, handler) -> None:
    _LANE_HANDLERS[lane] = handler


@dataclass
class NightBudget:
    deadline: Any
    per_producer_cap: dict = field(default_factory=dict)

    def cap_for(self, name: str) -> int:
        return self.per_producer_cap.get(name, 1)


@dataclass
class ExecResult:
    health: str = "ok"
    served_operator: str = ""


class NoLaneHandler(RuntimeError):
    pass


@dataclass
class WorkItem:
    project_id: str
    horizon: dict
    lane: str


def _deposit(*args, **kwargs) -> None:
    pass


def _register_gpu_lane() -> None:
    try:
        from gpu_lane import gpu_lane_handler
        register_lane_handler("gpu", gpu_lane_handler)
    except ImportError as exc:
        log.warning("[night-coordinator] gpu_lane module not available: %s", exc)


_register_gpu_lane()
