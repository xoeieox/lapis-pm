"""lapis_pm.bench — Synapse fresh-instance benchmark harness.

Spawns deliberately-stripped Claude Code instances against a pre-registered
loud/quiet topic battery and captures responses in a stable JSON format.

At v1 this produces baseline runs only.  The Synapse-enabled run path is
reserved for Phase 2 (suggested: pass ``synapse=True`` to ``run_battery`` +
``spawn_stripped`` and populate a ``synapse_state`` key in the capture).

Usage::

    python -m lapis_pm.bench --battery loud_quiet --out runs/baseline-<ts>.json
    python -m lapis_pm.bench.compare baseline.json labeled.json
"""

from lapis_pm.bench.fresh_instance import spawn_stripped
from lapis_pm.bench.battery import load_battery, run_battery
from lapis_pm.bench.compare import compare

__all__ = ["spawn_stripped", "load_battery", "run_battery", "compare"]
