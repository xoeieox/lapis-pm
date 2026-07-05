"""Fixture stub of conductor's night_plan.py entrypoint.

Reproduces the real file's load-bearing import shape (verified against conductor
origin/main = bede656, night-plan-conductor-deploy-sync-v0 spec): a module-level
import of agents_core.slots, plus deferred (function-local) imports of
night_coordinator symbols and importlib.import_module of producer modules. Business
logic is intentionally NOT reproduced — this fixture exists solely to exercise the
scripts/-local import graph the R4 closure test walks.
"""

from __future__ import annotations

import importlib

from agents_core.slots import IS_MASTER, TERMINAL_STATUSES, SlotStore  # noqa: F401


def _dispatch_lane(lane: str, node_id: str):
    from night_coordinator import _LANE_HANDLERS

    if lane not in _LANE_HANDLERS:
        raise RuntimeError(
            f"NoLaneHandler: node '{node_id}' lane={lane!r} has no registered handler "
            f"(registered: {sorted(_LANE_HANDLERS)})"
        )
    return _LANE_HANDLERS[lane]


def _run_producer(module_name: str):
    mod = importlib.import_module(module_name)
    return mod


def _run_night(deadline):
    from night_coordinator import NightBudget, _deposit, NoLaneHandler  # noqa: F401

    budget = NightBudget(deadline=deadline, per_producer_cap={})
    return budget
