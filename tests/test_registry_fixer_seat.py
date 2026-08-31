"""Registry-shape pin: fixer + fixer_retry run the GPU0 DURABLE DEFAULT shape.

The pair is on the durable default (ratified 2026-08-31, berth dormant
until on-demand) - the fully-proven full-payload path on the vLLM seat at
:8081:

  engine: local-fixer
  model: gravitywell-slot1   (the :8081 seat; the slot alias survives model
                              swaps)
  NO backend_url             (routes via the seat alias / GW_URL default)
  NO swarm_payload           (the full payload is the proven shape here)

Provenance: the 2026-08-24 berth flip (PR #305, spec
gw-gpu1-berth-standing-seat-v0, leg 3) was reversed 2026-08-29 by the
stopgap pause (PR #308, Erah-directed: the berth - the NInfer 27B seat on
GPU 1 / 3090 Ti at host port :8082 - down, GPU 1 may be repurposed to
ComfyUI). The 2026-08-31 ratification made the stopgap shape the DURABLE
default: the berth is dormant by default (unit disabled, no auto-start)
and is the ON-DEMAND OVERFLOW seat only, activated by re-pointing the DEPLOY
CLONE registry (never this repo file) while the unit is started manually.

This file pins the GPU0-default shape so a repo re-point of either entry
to the berth (ninfer-27b / :8082 / swarm_payload) fails loudly here: it is
only correct as a deliberate, reviewed on-demand overflow edit of the
DEPLOY CLONE, never as a repo commit. (The file formerly named
test_registry_berth_shape.py is now a zero-test deprecation stub - the
fixer toolset has no file-delete primitive.) The engine+model parity
between the pair is test_registry_agent_tiers.py's job; this file pins the
shape keys on each entry individually so a half-re-point (one entry to the
berth, the other not) is caught by BOTH tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_GPU0_DEFAULT_MODEL = "gravitywell-slot1"


@pytest.fixture(scope="module")
def agents():
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    data = yaml.safe_load(registry_path.read_text())
    return data["agents"]


@pytest.mark.parametrize("name", ["fixer", "fixer_retry"])
def test_gpu0_default_shape_keys(name, agents):
    """The pair is on the GPU0 durable default: gravitywell-slot1, no berth
    keys. A repo re-point of either entry to the berth (ninfer-27b / :8082
    / swarm_payload) routes dispatches at a seat that is dormant by
    default - see the module docstring. On-demand overflow re-points the
    DEPLOY CLONE only."""
    entry = agents[name]
    assert entry["engine"] == "local-fixer", (
        f"{name}.engine={entry.get('engine')!r} - the pair is the "
        "local-fixer engine"
    )
    assert entry["model"] == _GPU0_DEFAULT_MODEL, (
        f"{name}.model={entry.get('model')!r} - the durable default is the "
        f"GPU0 seat {_GPU0_DEFAULT_MODEL!r}; a repo re-point to the berth "
        "(ninfer-27b) is a deliberate, reviewed overflow change, never a "
        "default"
    )
    assert entry.get("backend_url") is None, (
        f"{name}.backend_url={entry.get('backend_url')!r} - the durable "
        "default routes via the seat alias; a berth backend_url is only "
        "correct in an on-demand overflow edit of the DEPLOY CLONE"
    )
    assert entry.get("swarm_payload") is None, (
        f"{name}.swarm_payload={entry.get('swarm_payload')!r} - the "
        "durable default runs the full payload; swarm_payload is only "
        "correct on the berth (overflow, deploy clone)"
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
