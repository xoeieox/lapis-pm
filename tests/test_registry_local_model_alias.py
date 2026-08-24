"""Registry regression test: local-reviewer / local-fixer agents must pin
`model` to a vLLM slot alias, never a concrete model id, and any
`acquire_lease: false` must carry an explicit `backend_url` alongside it.

Spec: lapis-pm-spec-review-leg-hotfix-reconcile-v0

This is the test the original defect would have caught. `spec_reviewer`,
`reviewer`, and `reviewer_fresh` all carried `model: gravitywell-122b` - a
concrete model id that stopped resolving when GravityWell moved to the dual
a3b pair. vLLM rejects an unknown model id, `call_gw_agent` returns None,
and `result or ""` in shaped_runner converts that failed inference into a
recorded successful empty string - a 1s "completed" no-op with no verdict
and no error. A concrete model id is silently accepted by YAML and only
fails at request time against a live server, so nothing short of a registry
assertion catches a drift like this before it reaches production.

`fixer_local` and `scout` carried the identical dead pin and were left
unfixed by lapis-pm-spec-review-leg-hotfix-reconcile-v0 (neither was on
the spec-review or merge path at the time). lapis-pm-stale-122b-model-pins-v0
repinned both to `gravitywell-slot1`, so every local-reviewer / local-fixer
entry now passes this assertion outright - no xfail marks remain.

One deliberate exemption: the GPU1 berth pair (fixer + fixer_retry,
gpu1-berth-reflip-v0 / gw-gpu1-berth-standing-seat-v0 leg 3) is a
GravityWell backend, but on the NInfer engine at host port :8082 - not a
vLLM slot. The slot-alias defect this file guards is vLLM slot-alias drift;
the berth is a non-vLLM GravityWell engine whose model id is engine-defined
(the NInfer --model-id `ninfer-27b`, served verbatim by its /v1/models per
the leg-1 DoD capture), so no slot alias exists to pin to and the
slot-alias invariant does not apply. Its berth SHAPE (model + backend_url +
swarm_payload) is pinned by tests/test_registry_berth_shape.py; the
NINFER_BERTH_AGENTS set below exempts ONLY the alias check.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REGISTRY_PATH = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"

# Every vLLM slot advertises both its model id (e.g. gravitywell-a3b-nvfp4)
# and a stable alias (gravitywell-slot1/slot2). The alias survives model
# swaps; a concrete model id does not.
VALID_SLOT_ALIASES = {"gravitywell-slot1", "gravitywell-slot2"}

# Non-GravityWell backends (third-party swarm/contractor seats) are exempt:
# the alias/concrete-id defect this file guards against is specific to
# vLLM's slot-serving model on GravityWell. A contractor seat's model field
# is a literal id its own API expects (e.g. Phala's "deepseek/deepseek-v4-
# flash") - there is no alias to pin to, and pinning one would just be
# wrong. lapis-pm-reviewer-peak-contractor-route-v0 adds the first such
# entry, reviewer_fresh_contractor, routed to the BRIX phala-test-key seat.
NON_GRAVITYWELL_CONTRACTOR_AGENTS = {"reviewer_fresh_contractor"}

# The GPU1 berth pair (gpu1-berth-reflip-v0 / gw-gpu1-berth-standing-seat-v0
# leg 3): a GravityWell backend, but the NInfer engine at host port :8082 -
# not a vLLM slot. Its model is the NInfer --model-id (ninfer-27b) because
# the NInfer engine's /v1/models serves that exact id (leg-1 DoD capture);
# no slot alias exists to pin to, so the slot-alias invariant does not
# apply. The berth SHAPE (model + backend_url + swarm_payload) is pinned by
# tests/test_registry_berth_shape.py; this set only exempts the alias check.
NINFER_BERTH_AGENTS = {"fixer", "fixer_retry"}

_REGISTRY = yaml.safe_load(REGISTRY_PATH.read_text())
_AGENTS = _REGISTRY["agents"]

LOCAL_AGENT_NAMES = sorted(
    name
    for name, cfg in _AGENTS.items()
    if cfg.get("engine") in ("local-reviewer", "local-fixer")
)

GRAVITYWELL_AGENT_NAMES = sorted(
    name
    for name in LOCAL_AGENT_NAMES
    if name not in NON_GRAVITYWELL_CONTRACTOR_AGENTS and name not in NINFER_BERTH_AGENTS
)


@pytest.mark.parametrize("name", GRAVITYWELL_AGENT_NAMES)
def test_local_agent_model_is_slot_alias_not_concrete_id(name):
    model = _AGENTS[name].get("model")
    assert model in VALID_SLOT_ALIASES, (
        f"{name}.model={model!r} is not a slot alias (gravitywell-slot1 / "
        f"gravitywell-slot2). A concrete model id is rejected by the model "
        f"server at request time - vLLM 404s, call_gw_agent returns None, "
        f"and shaped_runner's `result or \"\"` converts that failed "
        f"inference into a recorded successful empty string (a silent "
        f"1s 'completed' no-op with no verdict, no error). This is the "
        f"exact defect lapis-pm-spec-review-leg-hotfix-reconcile-v0 fixed - "
        f"pin `model` to the slot alias, never a concrete model id."
    )


@pytest.mark.parametrize("name", LOCAL_AGENT_NAMES)
def test_local_agent_acquire_lease_false_requires_backend_url(name):
    cfg = _AGENTS[name]
    if cfg.get("acquire_lease") is False:
        assert cfg.get("backend_url"), (
            f"{name} sets acquire_lease: false without an explicit "
            f"backend_url. The shaper's phantom-swarm guardrail rejects "
            f"this combination at registry load: acquire_lease: false "
            f"without backend_url would skip the doorman lease AND fall "
            f"back to the default GW_URL endpoint, which is not what a "
            f"leaseless dispatch means. acquire_lease: false requires an "
            f"explicit backend_url alongside it."
        )
