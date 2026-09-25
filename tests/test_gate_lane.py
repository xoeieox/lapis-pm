"""Tests for lapis_pm.gate_lane — the registry-resolved gate-lane resolver
(gate-lanes-registry-driven-flashnext-v0, S1 vendored shim).

Acceptance (machine, this PR's CI — stubbed registry, never a live call):
  1. With a stubbed registry flashnext row, --facets-operator flashnext
     resolves base_url :30000 + correct served-id; with an empty registry,
     byte-identical to today (the GW_URL fallback).
  3. Council voicing flashnext builds against the resolved lane; preflight
     probes the right port.
  S8. The local-voicing lease-free guard pins the doorman /lease/acquire
      call site unreachable on the local-voicing code path (the 409
      contention shape is pinned unreachable, not merely a zero count);
      blind never refuses; a held gravitywell-seat lease refuses with an
      honest leg_down (never a fallback-run).
"""
from __future__ import annotations

import pytest

from lapis_pm import gate_lane
from lapis_pm.gate_lane import (
    FLASHNEXT_SERVED_ID,
    FLASHNEXT_LANE_NAME,
    SLOT1_LANE_NAME,
    GateLane,
    gate_lane_serving,
    resolve_gate_lane,
)


# ---------------------------------------------------------------------------
# Stubbed registry payloads (the f0fb039 row shape; verified live 2026-09-25
# against the :8408 status: reality_view + seats[30000] with state/model/
# model_root/bind)
# ---------------------------------------------------------------------------

def _flashnext_solo_payload():
    return {
        "host": "gravitywell",
        "service": "gw-seats",
        "reality_view": {
            "reality": "flashnext-solo",
            "primary": {"port": 30000, "model_root": "/models/models/Qwen3.8-Flash-Next-NVFP4-SSD-Stream"},
            "anchor": "/models/models/Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
        },
        "seats": [
            {"port": 8081, "state": "down", "model": None, "model_root": None, "bind": None},
            {"port": 8082, "state": "down", "model": None, "model_root": None, "bind": None},
            {"port": 30000, "state": "serving", "engine": "sglang",
             "model": FLASHNEXT_SERVED_ID,
             "model_root": "/models/models/Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
             "bind": "203.0.113.11"},
        ],
    }


def _slot1_solo_payload():
    return {
        "host": "gravitywell",
        "service": "gw-seats",
        "reality_view": {
            "reality": "slot1-solo",
            "primary": {"port": 8081, "model_root": "/models/models/Qwen3.8-27B-NVFP4"},
            "anchor": "/models/models/Qwen3.8-27B-NVFP4",
        },
        "seats": [
            {"port": 8081, "state": "serving", "engine": "vllm",
             "model": "gravitywell-27b",
             "model_root": "/models/models/Qwen3.8-27B-NVFP4",
             "bind": "203.0.113.11"},
            {"port": 30000, "state": "down", "model": None, "model_root": None, "bind": None},
        ],
    }


def _flashnext_down_payload():
    """Readable registry, flashnext seat registered but NOT serving (the
    27B is up). The requested-but-dead-lane case: an honest leg_down, never
    a masked legacy fallback."""
    return {
        "host": "gravitywell",
        "service": "gw-seats",
        "reality_view": {
            "reality": "slot1-solo",
            "primary": {"port": 8081, "model_root": "/models/models/Qwen3.8-27B-NVFP4"},
            "anchor": "/models/models/Qwen3.8-27B-NVFP4",
        },
        "seats": [
            {"port": 8081, "state": "serving", "model": "gravitywell-27b",
             "model_root": "/models/models/Qwen3.8-27B-NVFP4", "bind": "203.0.113.11"},
            {"port": 30000, "state": "down", "model": None, "model_root": None, "bind": None},
        ],
    }


# ---------------------------------------------------------------------------
# Acceptance 1: flashnext row resolution + byte-identical empty-registry
# fallback
# ---------------------------------------------------------------------------

class TestResolveGateLaneFlashnext:
    def test_flashnext_solo_resolves_30000_and_served_id(self):
        """AC1: with a stubbed registry flashnext row, the flashnext lane
        resolves base_url :30000 + the correct served-id."""
        lane = resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=_flashnext_solo_payload)
        assert lane is not None
        assert lane.name == FLASHNEXT_LANE_NAME
        assert lane.base_url == "http://203.0.113.11:30000"
        assert lane.served_model == FLASHNEXT_SERVED_ID

    def test_live_lane_under_flashnext_solo_is_flashnext(self):
        """lane=None (the live lane) resolves to flashnext when reality is
        flashnext-solo."""
        lane = resolve_gate_lane(lane=None, fetcher=_flashnext_solo_payload)
        assert lane is not None
        assert lane.name == FLASHNEXT_LANE_NAME
        assert lane.base_url == "http://203.0.113.11:30000"

    def test_live_lane_under_slot1_solo_is_slot1(self):
        lane = resolve_gate_lane(lane=None, fetcher=_slot1_solo_payload)
        assert lane is not None
        assert lane.name == SLOT1_LANE_NAME
        assert lane.base_url == "http://203.0.113.11:8081"

    def test_empty_registry_is_blind_none(self):
        """AC1: with an empty registry (blind), resolution is None — the
        ONLY case in which callers fall back to the GW_URL behavior
        byte-identically."""
        assert resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=lambda: {}) is None
        assert resolve_gate_lane(lane=None, fetcher=lambda: {}) is None

    def test_malformed_payload_is_blind_none(self):
        assert resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=lambda: "not-a-dict") is None
        assert resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=lambda: None) is None

    def test_requested_but_dead_lane_is_none(self):
        """The caller contract: a readable registry with a requested-but-
        inactive lane resolves to None — the caller records an honest
        leg_down, NEVER a silent legacy fallback."""
        assert resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=_flashnext_down_payload) is None

    def test_unknown_lane_is_none(self):
        assert resolve_gate_lane(lane="bogus", fetcher=_flashnext_solo_payload) is None


class TestResolveGateLaneGwUrlFallback:
    """AC1 byte-identical fallback: a blind registry degrades to the exact
    GW_URL value (env-var driven), never a guess."""

    def test_blind_falls_back_to_gw_url_default(self, monkeypatch):
        monkeypatch.delenv("GW_URL", raising=False)
        monkeypatch.delenv("GW_SEATS_URL", raising=False)
        from lapis_pm.spec_review import _gw_primary_url
        # lane=None (the default gravitywell legs) is byte-identical to
        # today: the GW_URL default, never registry-probing.
        assert _gw_primary_url() == "http://203.0.113.11:8081"

    def test_blind_falls_back_to_gw_url_env(self, monkeypatch):
        monkeypatch.setenv("GW_URL", "http://10.0.0.9:9999")
        from lapis_pm.spec_review import _gw_primary_url
        assert _gw_primary_url() == "http://10.0.0.9:9999"

    def test_explicit_lane_blind_degrades_to_gw_url(self, monkeypatch):
        """An explicit lane under a blind registry degrades to the GW_URL
        value byte-identically (the ONLY fallback case)."""
        monkeypatch.setenv("GW_URL", "http://10.0.0.9:9999")
        monkeypatch.setattr(gate_lane, "resolve_gate_lane", lambda *a, **k: None)
        from lapis_pm.spec_review import _gw_primary_url
        assert _gw_primary_url(lane="flashnext") == "http://10.0.0.9:9999"

    def test_explicit_lane_resolves_through_registry(self, monkeypatch):
        """An explicit lane under a readable registry resolves to the
        lane's registry base_url — never the GW_URL value."""
        monkeypatch.setenv("GW_URL", "http://10.0.0.9:9999")

        def _fake_resolve(lane=None, fetcher=None):
            if lane == "flashnext":
                return GateLane(name="flashnext", base_url="http://203.0.113.11:30000",
                                served_model=FLASHNEXT_SERVED_ID)
            return None

        monkeypatch.setattr(gate_lane, "resolve_gate_lane", _fake_resolve)
        from lapis_pm.spec_review import _gw_primary_url
        assert _gw_primary_url(lane="flashnext") == "http://203.0.113.11:30000"
        # lane=None stays byte-identical (never registry-probing)
        assert _gw_primary_url() == "http://100.0.9.9:9999".replace("100.0.9.9", "10.0.0.9")


# ---------------------------------------------------------------------------
# Acceptance 3: preflight probes the right port (S5)
# ---------------------------------------------------------------------------

class TestGateLaneServing:
    def test_flashnext_serving_probes_30000(self):
        """AC3: preflight probes the flashnext lane's actual /v1/models
        (the :30000 base_url), for the registry-pinned served-model-name."""
        probed: list[str] = []

        def probe(base_url: str):
            probed.append(base_url)
            return FLASHNEXT_SERVED_ID

        ok, reason = gate_lane_serving(lane="flashnext", fetcher=_flashnext_solo_payload, probe=probe)
        assert ok is True
        assert reason == ""
        assert probed == ["http://203.0.113.11:30000"], (
            "the preflight must probe the resolved :30000 lane, never the "
            "hardcoded :8081 SWARM_URL"
        )

    def test_flashnext_down_is_honest_leg_down(self):
        """A readable registry with a dead flashnext seat is an honest
        leg_down — never a silent mis-skip, never a masked legacy
        fallback."""
        ok, reason = gate_lane_serving(lane="flashnext", fetcher=_flashnext_down_payload, probe=lambda u: "x")
        assert ok is False
        assert reason == "flashnext_not_serving"

    def test_flashnext_probe_failure_is_honest_leg_down(self):
        """The registry says serving but the live /v1/models probe fails ->
        honest leg_down (the lane is not actually serving)."""
        ok, reason = gate_lane_serving(lane="flashnext", fetcher=_flashnext_solo_payload, probe=lambda u: None)
        assert ok is False
        assert reason == "flashnext_not_serving"

    def test_flashnext_served_id_mismatch_is_leg_down(self):
        """The lane serves a DIFFERENT model than the registry pinned -> not
        the requested lane (served-model-name pins are the contract)."""
        ok, reason = gate_lane_serving(
            lane="flashnext", fetcher=_flashnext_solo_payload, probe=lambda u: "some-other-model"
        )
        assert ok is False
        assert reason == "flashnext_not_serving"

    def test_blind_registry_reports_registry_blind(self):
        """The ONLY case the caller may fall back to legacy behavior: the
        registry is unreachable/malformed."""
        ok, reason = gate_lane_serving(lane="flashnext", fetcher=lambda: {}, probe=lambda u: "x")
        assert ok is False
        assert reason == "registry_blind"


# ---------------------------------------------------------------------------
# S8: the local-voicing lease-free guard (hardened acceptance)
# ---------------------------------------------------------------------------

class TestLocalVoicingLeaseFree:
    """S8 acceptance: (a) zero lease calls when voicing=local AND (b) the
    doorman /lease/acquire call site is UNREACHABLE on the local-voicing
    code path — the 409 contention shape is pinned unreachable, not merely
    that a lease-call count is zero. A zero-count assertion on a path the
    acquire never reaches is vacuous; the live 409 fired before the
    voicing branch was evaluated. The guard is a fail-closed REFUSE
    (leg_down), never a fallback-run; blind never refuses."""

    def _doorman(self, status_result, acquire_raises=None):
        from unittest.mock import MagicMock

        client = MagicMock()
        client.status.return_value = status_result
        if acquire_raises is not None:
            client.acquire.side_effect = acquire_raises
        else:
            client.acquire.side_effect = AssertionError(
                "doorman /lease/acquire MUST be unreachable on the "
                "local-voicing code path (S8: the 409 contention shape is "
                "pinned unreachable, not merely a zero count)"
            )
        return client

    def test_local_voicing_never_acquires_lease(self, monkeypatch):
        """(b) the doorman /lease/acquire call site is unreachable on the
        local-voicing code path: any acquire call raises (the mock's
        side_effect), and the guard completes without touching it. The
        guard also makes zero lease calls (a) — assert both: the acquire
        was never called AND the guard returned the lease-free verdict."""
        from unittest.mock import MagicMock
        client = self._doorman({"lease_count": 0, "leases": []})
        monkeypatch.setattr(
            "agents_core.doorman_client.DoormanClient", lambda *a, **k: client
        )
        ok, reason = gate_lane_serving(lane="local")
        assert ok is True
        assert reason == ""
        # (a) zero lease calls: acquire was never invoked.
        client.acquire.assert_not_called()
        # (b) the acquire call site is unreachable: the guard read the
        # seat state (status) and never fell into an acquire.
        client.status.assert_called_once()

    def test_local_voicing_refuses_when_lease_held(self, monkeypatch):
        """A held gravitywell-seat lease -> honest leg_down (refuse), never
        a fallback-run. The 409 contention shape (a flashnext window with
        a held lease) is refused BEFORE the leg would acquire."""
        from unittest.mock import MagicMock
        client = self._doorman({"lease_count": 1, "leases": [{"work_id": "council-delib-x"}]})
        monkeypatch.setattr(
            "agents_core.doorman_client.DoormanClient", lambda *a, **k: client
        )
        ok, reason = gate_lane_serving(lane="local")
        assert ok is False
        assert reason == "local_seat_lease_refused"
        # The refusal is a leg_down, never a fallback-run: the guard never
        # acquired a lease itself (the acquire call site is unreachable on
        # this path — a fallback-run would have acquired on the
        # gravitywell lane).
        client.acquire.assert_not_called()

    def test_local_voicing_blind_doorman_never_refuses(self, monkeypatch):
        """Blind never refuses: the doorman is unreachable -> the local
        voicing runs lease-free exactly as today (the 409 shape requires a
        reachable doorman with a held lease, which this path never
        produces because it never acquires)."""
        from unittest.mock import MagicMock
        client = MagicMock()
        client.status.side_effect = Exception("doorman unreachable")
        client.acquire.side_effect = AssertionError(
            "acquire MUST be unreachable on the local-voicing path"
        )
        monkeypatch.setattr(
            "agents_core.doorman_client.DoormanClient", lambda *a, **k: client
        )
        ok, reason = gate_lane_serving(lane="local")
        assert ok is True
        assert reason == ""
        client.acquire.assert_not_called()

    def test_local_voicing_blind_doorman_import_never_refuses(self, monkeypatch):
        """A missing agents_core.doorman_client (test isolation) is blind ->
        never refuses."""
        import sys
        import importlib

        real = sys.modules.get("agents_core.doorman_client")
        monkeypatch.setitem(sys.modules, "agents_core.doorman_client", None)
        try:
            ok, reason = gate_lane_serving(lane="local")
        finally:
            monkeypatch.undo()
        assert ok is True
        assert reason == ""
