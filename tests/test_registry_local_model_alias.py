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

`fixer_local` and `scout` carry the identical dead pin and are deliberately
left unfixed by lapis-pm-spec-review-leg-hotfix-reconcile-v0 (neither is on
the spec-review or merge path; `fixer_local`'s DEPRECATED status needs a
decision that isn't this target's to make). They are marked xfail(strict)
rather than excluded, per the spec-review gate's explicit mandate to
preserve signal integrity and prevent silent reversion by future workers.
A follow-up target must fix the pin (or delete the entry) for `fixer_local`
and unpin `scout`; when that lands, these xfails will start XPASSing and
CI will force the marks to be removed.
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

# Found while applying this fix, deliberately raised rather than fixed here -
# see the spec's "Two more entries carry the same dead pin" section.
KNOWN_UNFIXED_DEAD_PIN = {"fixer_local", "scout"}

_REGISTRY = yaml.safe_load(REGISTRY_PATH.read_text())
_AGENTS = _REGISTRY["agents"]

LOCAL_AGENT_NAMES = sorted(
    name
    for name, cfg in _AGENTS.items()
    if cfg.get("engine") in ("local-reviewer", "local-fixer")
)


def _model_alias_param(name):
    marks = []
    if name in KNOWN_UNFIXED_DEAD_PIN:
        marks.append(
            pytest.mark.xfail(
                reason=(
                    f"{name} still carries the dead gravitywell-122b pin - "
                    "raised, not fixed, by lapis-pm-spec-review-leg-hotfix-"
                    "reconcile-v0 (not on the spec-review or merge path). A "
                    "follow-up target must decide fixer_local's fate and "
                    "unpin scout."
                ),
                strict=True,
            )
        )
    return pytest.param(name, id=name, marks=marks)


@pytest.mark.parametrize("name", [_model_alias_param(n) for n in LOCAL_AGENT_NAMES])
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
