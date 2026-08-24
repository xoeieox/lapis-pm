"""Registry-shape pin: the berth flip (gw-gpu1-berth-standing-seat-v0, leg 3)
is HELD - the pair runs the stopgap.

The flip (PR #302, merged 2026-08-24) was reverted the same day: the first
real berth dispatch failed, exposing two defects (mem finding/
gpu1-berth-first-berth-dispatch-failed-2026-08-24):

  1. SHAPER ENCODE GAP (proven): agents_core/shaper.py never carries the
     swarm_payload registry key into the encoded job spec, so the keep-both
     berth shape collapses at runtime - _is_swarm computes False -> full
     payload -> NInfer 400 chat_template_option_not_supported (reproduced
     by direct POST to :8082).
  2. ADMISSION DEFER (root cause open): the job's per-run deferrable lease
     was starved in the doorman wait-list for the full ~58s retry budget on
     both runs.

Until the shaper fix (agents-core) deploys AND the admission root cause
closes, fixer + fixer_retry run the STOPGAP shape - the fully-proven
full-payload path on the vLLM seat:

  engine: local-fixer
  model: gravitywell-slot1   (the :8081 seat)
  NO backend_url             (routes via the seat alias)
  NO swarm_payload           (the full payload is the proven shape here)

This file pins the stopgap shape so a premature re-flip (re-pointing either
entry to the berth before the encode link is fixed) fails loudly here. The
re-flip PR restores the berth assertions on this same file: model ninfer-27b
+ backend_url http://203.0.113.11:8082 + swarm_payload: true
(keep-both: acquire_lease stays at the default true - no explicit key).
The engine+model parity between the pair is test_registry_agent_tiers.py's
job; this file pins the shape keys on each entry individually so a
half-revert (one entry re-pointed to the berth, the other not) is caught by
BOTH tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_STOPGAP_MODEL = "gravitywell-slot1"


@pytest.fixture(scope="module")
def agents():
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    data = yaml.safe_load(registry_path.read_text())
    return data["agents"]


@pytest.mark.parametrize("name", ["fixer", "fixer_retry"])
def test_stopgap_shape_held_pending_shaper_fix(name, agents):
    """The pair is on the stopgap: gravitywell-slot1, no berth keys.

    Re-pointing either entry to the berth (ninfer-27b / :8082 /
    swarm_payload) before the agents-core shaper fix deploys re-introduces
    the proven 400 on every dispatch - see the module docstring. The
    re-flip PR updates this test to the berth shape."""
    entry = agents[name]
    assert entry["engine"] == "local-fixer", (
        f"{name}.engine={entry.get('engine')!r} - the pair is the "
        "local-fixer engine"
    )
    assert entry["model"] == _STOPGAP_MODEL, (
        f"{name}.model={entry.get('model')!r} - the berth is HELD (shaper "
        "encode gap + admission defer open); the pair must stay on the "
        f"stopgap seat {_STOPGAP_MODEL!r} until the re-flip PR lands"
    )
    assert entry.get("backend_url") is None, (
        f"{name}.backend_url={entry.get('backend_url')!r} - the stopgap "
        "lane routes via the seat alias; a berth backend_url is only "
        "correct once the shaper carries swarm_payload (held re-flip)"
    )
    assert entry.get("swarm_payload") is None, (
        f"{name}.swarm_payload={entry.get('swarm_payload')!r} - the "
        "stopgap lane runs the full payload; swarm_payload is only correct "
        "on the berth (held re-flip)"
    )