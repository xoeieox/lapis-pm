"""Registry-shape pin: the fixer_flash trial tier (flashnext-fixer-trial-v0,
leg 1, D1).

fixer_flash is the flash-next TRIAL fixer tier - an ADDITIVE registry row
cloned from the berth pattern (the NInfer 27B seat on :8082 was the
precedent for a non-vLLM seat whose model field is the CONCRETE served id,
not a vLLM slot alias):

  engine: local-fixer
  model: Qwen3.8-Flash-Next-NVFP4-SSD-Stream   (the EXACT served id on the
                                                sglang flash-next seat)
  backend_url: http://203.0.113.11:30000    (the sglang seat, HARD-CAPPED
                                                at 2 concurrent requests
                                                engine-side via
                                                --max-running-requests 2 /
                                                --max-mamba-cache-size 8)
  swarm_payload: true                          (berth precedent; the ONLY
                                                lever that turns on the
                                                swarm body while
                                                acquire_lease stays at the
                                                default true)
  NO explicit acquire_lease                    (keep-both default true; the
                                                leg-2 mandatory-lease leg
                                                enforces defer-not-proceed
                                                on acquire failure)

This file reads the real lapis_pm/registry.yaml (not a fixture) so a future
edit to the block is checked against the value that actually ships.

Scope guard: the fixer / fixer_retry parity rule (test_registry_agent_tiers.py)
and the GPU0-default shape pin (test_registry_fixer_seat.py) stay scoped to
the 27B pair - fixer_flash is the trial arm, a SEPARATE tier, and must not
be pulled into either guardrail. Conversely the fixer pair must NOT drift
onto the trial seat: the baseline arm of the trial is the 27B pair on
:8081, and a re-point of either pair member to the flash-next seat would
corrupt the arms.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# The EXACT served id on the sglang flash-next seat (verified via
# :30000/v1/models data[0].id at 09:25 on 2026-09-24; spec section 2).
# On this seat the served id IS the concrete model id - the slot-alias
# pitfall (registry :74-79) is a vLLM-seat property and does not apply.
SERVED_ID = "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
BACKEND_URL = "http://203.0.113.11:30000"


@pytest.fixture(scope="module")
def agents():
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    data = yaml.safe_load(registry_path.read_text())
    return data["agents"]


def test_fixer_flash_entry_exists(agents):
    assert "fixer_flash" in agents, (
        "the fixer_flash trial tier (flashnext-fixer-trial-v0, leg 1, D1) "
        "must be present in the registry"
    )


def test_fixer_flash_engine_is_local_fixer(agents):
    """The trial arm runs the SAME engine as the fixer pair
    (_run_local_fixer) - not the staged harness. The worktree_required
    exemption applies because the engine is local-* (shaper.py:416-419)."""
    assert agents["fixer_flash"]["engine"] == "local-fixer", (
        f"fixer_flash.engine={agents['fixer_flash'].get('engine')!r} - the "
        "trial arm must run the local-fixer engine (the _run_local_fixer "
        "path), not the staged harness"
    )


def test_fixer_flash_model_equals_served_id(agents):
    """The model field MUST equal the exact served id on the sglang seat.

    A wrong id is the slot-alias-pitfall defect (registry :74-79) in its
    sglang form: call_gw_agent returns None and shaped_runner's
    `result or ""` turns that into a silent empty-string success. On the
    vLLM seats the model must be the slot alias; on this sglang seat the
    served id IS the concrete model id, so the concrete id is correct and
    a slot alias would be the defect.
    """
    model = agents["fixer_flash"].get("model")
    assert model == SERVED_ID, (
        f"fixer_flash.model={model!r} != the exact served id "
        f"{SERVED_ID!r} (verified via :30000/v1/models data[0].id). A "
        "wrong id is the silent-empty-string defect: call_gw_agent "
        "returns None and shaped_runner's `result or \"\"` converts that "
        "failed inference into a recorded successful empty string."
    )


def test_fixer_flash_backend_url_is_sglang_seat(agents):
    """backend_url MUST be the sglang flash-next seat at :30000 and MUST
    stay present and non-empty.

    backend_url presence is an active behavioral constraint, not legacy
    residue: it disarms nothing while acquire_lease stays at the default
    true, and it feeds the queue-side seat match (claude_queue.py) and the
    lease-scope gate (the leg-2 mandatory-lease leg acquires against the
    :30000 anchor). The ONLY sanctioned payload fallback is
    swarm_payload: false - NEVER dropping backend_url.
    """
    backend_url = agents["fixer_flash"].get("backend_url")
    assert backend_url == BACKEND_URL, (
        f"fixer_flash.backend_url={backend_url!r} != {BACKEND_URL!r} - the "
        "trial seat is the sglang flash-next seat at :30000"
    )


def test_fixer_flash_swarm_payload_true(agents):
    """swarm_payload MUST be true (berth precedent).

    _is_swarm = swarm_payload or ((backend_url is not None) and (not
    acquire_lease)) - with acquire_lease at the default true, backend_url
    presence alone does NOT flip _is_swarm; the lever is the registry
    swarm_payload key. The smoke-test swarm premise (enable_thinking +
    grammar response_format) rides swarm_payload. The ONLY sanctioned
    fallback is swarm_payload: false - never dropping backend_url.
    """
    assert agents["fixer_flash"].get("swarm_payload") is True, (
        f"fixer_flash.swarm_payload={agents['fixer_flash'].get('swarm_payload')!r} - "
        "the berth precedent is swarm_payload: true (the only lever that "
        "turns on the swarm body while acquire_lease stays at the default "
        "true). The only sanctioned fallback is false - never dropping "
        "backend_url."
    )


def test_fixer_flash_no_explicit_acquire_lease(agents):
    """keep-both: the entry must NOT set acquire_lease explicitly - the
    default (true) means the job holds the per-run job lease AND the
    supervisor lease. The leg-2 mandatory-lease leg enforces
    defer-not-proceed on acquire failure at the :30000 anchor; an
    explicit acquire_lease (true or false) changes the lease topology and
    must be a deliberate, reviewed change.
    """
    assert "acquire_lease" not in agents["fixer_flash"], (
        "fixer_flash must not set acquire_lease explicitly - the default "
        "(true) is the keep-both posture (spec D1 / keep-both)"
    )


def test_fixer_flash_not_member_of_fixer_pair(agents):
    """fixer_flash is the trial arm - a SEPARATE tier, not a third member
    of the fixer/fixer_retry pair. The parity rule
    (test_registry_agent_tiers.py) and the GPU0-default shape pin
    (test_registry_fixer_seat.py) stay scoped to the 27B pair; pulling
    fixer_flash into either would break the guardrail the moment the
    trial seat (concrete id, :30000, swarm_payload) is present.
    """
    pair = (agents["fixer"], agents["fixer_retry"])
    for entry in pair:
        assert entry.get("model") != SERVED_ID, (
            f"a fixer-pair member carries the flash-next served id "
            f"{SERVED_ID!r} - the baseline arm of the trial is the 27B "
            "pair on :8081 (gravitywell-slot1); a re-point of either pair "
            "member to the flash-next seat corrupts the arms"
        )
        assert entry.get("backend_url") != BACKEND_URL, (
            f"a fixer-pair member carries the flash-next backend_url "
            f"{BACKEND_URL!r} - the baseline arm stays on :8081"
        )


def test_fixer_flash_system_template_renders(agents):
    """The system_template must render without a KeyError under the full
    dispatch var set (the fixer template's var surface + the steer
    overlay). The template is str.format-interpolated by the shaper:
    literal braces in the friction.json / verdict JSON examples are
    DOUBLED ({{ }}) - a single brace would raise KeyError at dispatch.
    """
    entry = agents["fixer_flash"]
    template = entry["system_template"]
    # The template carries the steer directive channel (single-shot
    # directive; inject_overlay always sets the key, so the placeholder
    # must be present to avoid a KeyError on a render that lacks it).
    assert "{steer_directive_block}" in template
    vars_ = {
        "target_id": "flashnext-fixer-trial-v0",
        "spec_summary": "a spec",
        "repo": "lapis-pm",
        "question": "implement X",
        "pr_number": "",
        "slug": "forced",
        "existing_branch": "lapis/flashnext-fixer-trial-v0/forced",
        "base_branch": "main",
        "intent_block": "## Intent\nDo the thing.",
        "steer_directive_block": "## Active PM directive\nfocus on X",
    }
    rendered = template.format(**vars_)
    assert "flashnext-fixer-trial-v0" in rendered
    assert "lapis/flashnext-fixer-trial-v0/forced" in rendered
