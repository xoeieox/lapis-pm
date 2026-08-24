"""Registry-shape pin: the berth flip (gw-gpu1-berth-standing-seat-v0, leg 3).

fixer + fixer_retry carry the berth shape - the standing NInfer 27B seat on
GPU 1 (3090 Ti) at host port :8082:

  engine: local-fixer
  model: ninfer-27b            (the NInfer --model-id; leg-1 DoD confirms
                                 /v1/models serves it)
  backend_url: http://203.0.113.11:8082
  swarm_payload: true          (the NInfer engine rejects the full payload
                                 with HTTP 400 chat_template_option_not_
                                 supported - G2 gate record 2026-08-19 - so
                                 the berth seat MUST run the swarm shape)
  acquire_lease: default true  (keep-both: the job holds the per-run job
                                 lease AND the supervisor lease; the spec
                                 Design / keep-both section)

The flip is test-pinned, not comment-pinned: a silent re-point of either
entry back to the stopgap seat (gravitywell-slot1) fails loudly here.
The engine+model parity between the pair is test_registry_agent_tiers.py's
job; this file pins the berth shape keys on each entry individually so a
half-flip (one entry re-pointed, the other not) is caught by BOTH tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_BERTH_BACKEND_URL = "http://203.0.113.11:8082"
_BERTH_MODEL = "ninfer-27b"


@pytest.fixture(scope="module")
def agents():
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    data = yaml.safe_load(registry_path.read_text())
    return data["agents"]


@pytest.mark.parametrize("name", ["fixer", "fixer_retry"])
def test_berth_shape_keys(name, agents):
    entry = agents[name]
    assert entry["engine"] == "local-fixer", (
        f"{name}.engine={entry.get('engine')!r} - the berth pair is the "
        "local-fixer engine"
    )
    assert entry["model"] == _BERTH_MODEL, (
        f"{name}.model={entry.get('model')!r} - the berth seat serves "
        f"{_BERTH_MODEL!r} (NInfer --model-id); a re-point back to the "
        "stopgap seat (gravitywell-slot1) must be a deliberate, reviewed "
        "change (spec stopgap fallback)"
    )
    assert entry.get("backend_url") == _BERTH_BACKEND_URL, (
        f"{name}.backend_url={entry.get('backend_url')!r} - the berth seat "
        "is at host port :8082 (Erah port settlement 2026-08-21)"
    )
    assert entry.get("swarm_payload") is True, (
        f"{name}.swarm_payload={entry.get('swarm_payload')!r} - the NInfer "
        "engine rejects the full payload (HTTP 400 "
        "chat_template_option_not_supported); the berth seat MUST run the "
        "swarm shape"
    )


@pytest.mark.parametrize("name", ["fixer", "fixer_retry"])
def test_acquire_lease_stays_default_true(name, agents):
    """keep-both: the berth entry must NOT set acquire_lease explicitly -
    the default (true) means the job holds the per-run job lease AND the
    supervisor lease. An explicit acquire_lease (true or false) changes the
    lease topology and must be a deliberate, reviewed change."""
    entry = agents[name]
    assert "acquire_lease" not in entry, (
        f"{name} must not set acquire_lease explicitly - the default (true) "
        "is the keep-both posture (spec Design / keep-both)"
    )