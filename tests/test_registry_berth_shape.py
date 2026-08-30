"""Registry-shape pin: fixer + fixer_retry run the STOPGAP shape.

The pair is on the stopgap - the fully-proven full-payload path on the
vLLM seat at :8081:

  engine: local-fixer
  model: gravitywell-slot1   (the :8081 seat; the slot alias survives model
                              swaps)
  NO backend_url             (routes via the seat alias / GW_URL default)
  NO swarm_payload           (the full payload is the proven shape here)

Provenance (2026-08-29 stopgap pause, Erah-directed): the berth - the
standing NInfer 27B seat on GPU 1 (3090 Ti) at host port :8082 (spec
gw-gpu1-berth-standing-seat-v0, leg 3) - is DOWN for the 2026-08-29 night
and GPU 1 may be repurposed to ComfyUI. This is a deliberate operational
pause, NOT a defect hold: the agents-core shaper encode fix (PR #256 /
5dc2216) is deployed and the berth encode link was proven end-to-end before
the re-flip (PR #305 / ff23355). GPU 0 / :8081 is NOT the new default - the
stopgap is in effect for the current dispatches only.

This file pins the stopgap shape so a premature re-flip (re-pointing either
entry to the berth: ninfer-27b / :8082 / swarm_payload) fails loudly here.
The re-flip PR restores the berth assertions on this same file. The
engine+model parity between the pair is test_registry_agent_tiers.py's job;
this file pins the shape keys on each entry individually so a half-revert
(one entry re-pointed to the berth, the other not) is caught by BOTH tests.
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
def test_stopgap_shape_keys(name, agents):
    """The pair is on the stopgap: gravitywell-slot1, no berth keys.

    Re-pointing either entry to the berth (ninfer-27b / :8082 /
    swarm_payload) before the berth is back up routes dispatches at the
    dead seat - see the module docstring. The re-flip PR updates this test
    to the berth shape."""
    entry = agents[name]
    assert entry["engine"] == "local-fixer", (
        f"{name}.engine={entry.get('engine')!r} - the pair is the "
        "local-fixer engine"
    )
    assert entry["model"] == _STOPGAP_MODEL, (
        f"{name}.model={entry.get('model')!r} - the berth is down for the "
        "2026-08-29 night (GPU 1 may be repurposed); the pair must stay on "
        f"the stopgap seat {_STOPGAP_MODEL!r} until the re-flip PR lands"
    )
    assert entry.get("backend_url") is None, (
        f"{name}.backend_url={entry.get('backend_url')!r} - the stopgap "
        "lane routes via the seat alias; a berth backend_url is only "
        "correct once the berth is re-flipped"
    )
    assert entry.get("swarm_payload") is None, (
        f"{name}.swarm_payload={entry.get('swarm_payload')!r} - the "
        "stopgap lane runs the full payload; swarm_payload is only correct "
        "on the berth"
    )


@pytest.mark.parametrize("name", ["fixer", "fixer_retry"])
def test_acquire_lease_stays_default_true(name, agents):
    """keep-both: the entry must NOT set acquire_lease explicitly - the
    default (true) means the job holds the per-run job lease AND the
    supervisor lease. An explicit acquire_lease (true or false) changes the
    lease topology and must be a deliberate, reviewed change."""
    entry = agents[name]
    assert "acquire_lease" not in entry, (
        f"{name} must not set acquire_lease explicitly - the default (true) "
        "is the keep-both posture (spec Design / keep-both)"
    )
