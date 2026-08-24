"""Parity guardrail: fixer and fixer_retry must share one capability tier.

Coverage (spec: restore-fixer-retry-sonnet-registry-v0; berth flip
gw-gpu1-berth-standing-seat-v0 leg 3, reverted the same day):

  fixer and fixer_retry are a pair by design - initial attempt and correction
  attempt at the same capability tier. The pair is the local stopgap pair
  (gravitywell-slot1 on :8081). The 2026-08-24 berth flip (ninfer-27b on
  :8082, GPU 1 / 3090 Ti) was REVERTED the same day - first-berth-dispatch
  failure: shaper encode gap (swarm_payload never carried registry->spec)
  + admission defer (mem finding/
  gpu1-berth-first-berth-dispatch-failed-2026-08-24); the re-flip re-points
  both entries to the berth and updates this pin with it. The 2026-07-17
  migration
  (restore-fixer-sonnet-default-registry-v0) moved `fixer` to claude/sonnet on
  a stated premise that fixer_retry was "already sonnet" - a premise that was
  never checked and turned out to be false, leaving fixer_retry on
  local-fixer/gravitywell-122b for four days while every fixer_retry dispatch
  (the amend-spec-and-redispatch and target-directive redispatch paths) ran on
  a tier that reliably no-progress-aborts on anything past clean-MEDIUM.

  This test reads the real lapis_pm/registry.yaml (not a fixture) so a future
  edit to either block is checked against the value that actually ships, not
  a snapshot that can drift out of sync with it.

  fixer_local is deliberately excluded: it is the intentional opt-in local
  tier (see its own DEPRECATED-comment block in registry.yaml), not a second
  member of the fixer/fixer_retry pair. Dragging it into this parity check
  would break the guardrail every time someone legitimately used the local
  tier on purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def agents():
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    data = yaml.safe_load(registry_path.read_text())
    return data["agents"]


def test_fixer_and_fixer_retry_share_engine_and_model(agents):
    fixer = agents["fixer"]
    fixer_retry = agents["fixer_retry"]

    assert fixer["engine"] == fixer_retry["engine"], (
        f"fixer.engine={fixer['engine']!r} != fixer_retry.engine={fixer_retry['engine']!r} - "
        "fixer and fixer_retry must stay on the same engine tier "
        "(see restore-fixer-retry-sonnet-registry-v0: a migration that touches "
        "one half of this pair without the other silently strands retries on "
        "the wrong tier)"
    )
    assert fixer["model"] == fixer_retry["model"], (
        f"fixer.model={fixer['model']!r} != fixer_retry.model={fixer_retry['model']!r} - "
        "fixer and fixer_retry must stay on the same model tier "
        "(see restore-fixer-retry-sonnet-registry-v0: a migration that touches "
        "one half of this pair without the other silently strands retries on "
        "the wrong tier)"
    )


def test_fixer_and_fixer_retry_currently_on_local_27b(agents):
    """Pins the expected tier so a silent change of BOTH blocks together
    (which the parity test alone would not catch) still fails loudly.

    Flipped from claude/sonnet to local-fixer/gravitywell-slot1 2026-08-21
    as the deadline stopgap for the Claude Max end (spec:
    lapis-pm-fixer-local-stopgap-flip-v0). The berth flip 2026-08-24 (spec:
    gw-gpu1-berth-standing-seat-v0, leg 3) was REVERTED the same day -
    first-berth-dispatch failure (shaper encode gap + admission defer; mem
    finding/gpu1-berth-first-berth-dispatch-failed-2026-08-24) - so the pair
    is back on the stopgap until the re-flip PR lands, which updates this
    pin to ninfer-27b with it. The stopgap/berth shape keys are pinned by
    tests/test_registry_berth_shape.py; this test pins engine + model only."""
    assert agents["fixer"]["engine"] == "local-fixer"
    assert agents["fixer"]["model"] == "gravitywell-slot1"
    assert agents["fixer_retry"]["engine"] == "local-fixer"
    assert agents["fixer_retry"]["model"] == "gravitywell-slot1"


def test_fixer_local_excluded_from_parity_by_design(agents):
    """fixer_local is the intentional opt-in local tier, not part of the
    fixer/fixer_retry pair - it must stay on local-fixer/gravitywell-slot1
    and must NOT be pulled into the parity assertion above."""
    assert agents["fixer_local"]["engine"] == "local-fixer"
    assert agents["fixer_local"]["model"] == "gravitywell-slot1"
