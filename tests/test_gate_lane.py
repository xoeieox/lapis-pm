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
# Stubbed registry payloads — row shape matches the LIVE :8408 payload as
# re-attested 2026-09-25 for the review HIGH-1 fold: the serving :30000 row
# carries bind "0.0.0.0" (the SERVING bind, not a client host — client
# dialing must resolve through _registry_host()/GW_SEATS_URL, pinned by
# TestClientHostDerivation below); non-serving rows carry bind null.
# (The prior stubs fabricated bind "203.0.113.11" under a "verified
# live" header — the exact case the live registry never sends.)
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
             "bind": "0.0.0.0"},
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

    def test_wildcard_bind_uses_registry_origin_host(self, monkeypatch):
        """HIGH-1 fold (independent review 2026-09-25): the LIVE :8408
        serving row carries bind "0.0.0.0" (the serving-bind). The client
        host must be the registry's own origin host (GW_SEATS_URL), never
        the wildcard — otherwise every resolved lane is a phantom
        http://0.0.0.0:<port> that cannot be reached from the PM host."""
        monkeypatch.setenv("GW_SEATS_URL", "http://199.9.9.9:8408")
        lane = resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=_flashnext_solo_payload)
        assert lane is not None
        assert lane.base_url == "http://199.9.9.9:30000"

    def test_absent_bind_uses_registry_origin_host(self, monkeypatch):
        monkeypatch.setenv("GW_SEATS_URL", "http://199.9.9.9:8408")
        payload = _flashnext_solo_payload()
        for s in payload["seats"]:
            if s["port"] == 30000:
                s["bind"] = None
        lane = resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=lambda: payload)
        assert lane is not None
        assert lane.base_url == "http://199.9.9.9:30000"

    def test_ipv6_wildcard_bind_also_defers_to_registry_host(self, monkeypatch):
        monkeypatch.setenv("GW_SEATS_URL", "http://199.9.9.9:8408")
        payload = _flashnext_solo_payload()
        for s in payload["seats"]:
            if s["port"] == 30000:
                s["bind"] = "::"
        lane = resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=lambda: payload)
        assert lane is not None
        assert lane.base_url == "http://199.9.9.9:30000"

    def test_concrete_bind_still_honored(self):
        """A concrete bind (registry-forwarded rows) remains authoritative —
        the wildcard handling must not discard real hosts."""
        payload = _flashnext_solo_payload()
        for s in payload["seats"]:
            if s["port"] == 30000:
                s["bind"] = "10.0.0.5"
        lane = resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=lambda: payload)
        assert lane is not None
        assert lane.base_url == "http://10.0.0.5:30000"

    def test_default_env_resolves_tailscale_ip(self, monkeypatch):
        """The f0fb039 precedent: with no GW_SEATS_URL override, lanes dial
        the GW tailscale IP (NOT 0.0.0.0)."""
        monkeypatch.delenv("GW_SEATS_URL", raising=False)
        lane = resolve_gate_lane(lane=FLASHNEXT_LANE_NAME, fetcher=_flashnext_solo_payload)
        assert lane is not None
        assert lane.base_url == "http://203.0.113.11:30000"
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
        assert _gw_primary_url() == "http://10.0.0.9:9999"


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
    (leg_down), never a fallback-run; blind never refuses.

    HIGH-2 fold (independent review 2026-09-25): the OBSERVED live 409
    fired with lease_count 0 (doorman /status: GW seat serving=false,
    flashnext sub-view seat_state up_registered, window active). A guard
    that keys only on held leases pins the wrong shape — the live-shape
    stub below is the 409 as it actually happened."""

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

    def _live_409_shape_status(self):
        """The doorman /status snapshot captured 2026-09-25 while the
        acquire path was answering 409 (finding/council-lease-409-under-
        flashnext-window-2026-09-25): nodes.gravitywell.serving == false,
        flashnext sub-view up_registered with window active, lease_count 0
        and an EMPTY lease list. A guard that returns (True, "") on THIS
        shape sends the local leg into the acquiring council submit path
        and reproduces the live 409 — exactly the vacuity the spec text
        pre-judges."""
        return {
            "nodes": {
                "gravitywell": {
                    "serving": False,
                    "lease_count": 0,
                    "leases": [],
                    "flashnext": {
                        "seat_health": True,
                        "seat_state": "up_registered",
                        "served_id": FLASHNEXT_SERVED_ID,
                        "registered": True,
                        "window": "active",
                    },
                }
            }
        }

    def test_local_voicing_refuses_the_live_409_shape(self, monkeypatch):
        """The live-409 shape (seat down + flashnext window up, lease_count
        0, empty lease list) MUST be refused BEFORE the leg would acquire:
        the guard returns the honest leg_down reason, and the acquire call
        site stays unreachable (the mock's side_effect would raise)."""
        client = self._doorman(self._live_409_shape_status())
        monkeypatch.setattr(
            "agents_core.doorman_client.DoormanClient", lambda *a, **k: client
        )
        ok, reason = gate_lane_serving(lane="local")
        assert ok is False
        assert reason == "local_voicing_flashnext_window"
        client.acquire.assert_not_called()
        client.status.assert_called_once()

    def test_local_voicing_runs_lease_free_on_a_clean_seat(self, monkeypatch):
        """Lease-free shape: the GW seat itself is not asserting a flashnext
        window conflict and no lease is held -> the local voicing runs,
        making ZERO lease calls (the acquire call site is unreachable)."""
        status = {
            "nodes": {
                "gravitywell": {
                    "serving": True,
                    "lease_count": 0,
                    "leases": [],
                    "flashnext": {"seat_health": False, "seat_state": "down", "window": "closed"},
                }
            }
        }
        client = self._doorman(status)
        monkeypatch.setattr(
            "agents_core.doorman_client.DoormanClient", lambda *a, **k: client
        )
        ok, reason = gate_lane_serving(lane="local")
        assert ok is True
        assert reason == ""
        client.acquire.assert_not_called()
        client.status.assert_called_once()

    def test_local_voicing_legacy_status_shape_falls_back_conservatively(self, monkeypatch):
        """A /status payload WITHOUT the nodes block (a client whose shape
        lacks per-node leases) falls back to the GLOBAL lease_count — the
        honest fallback the guard's docstring documents; zero-count ->
        allow, zero acquire calls."""
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


# ---------------------------------------------------------------------------
# S7: the many-eyes pin is a MARKED DEBT, not a silent skip (tracked
# follow-up, NOT bundled in this PR — 2026-09-25 re-gate fold)
# ---------------------------------------------------------------------------

class TestS7MarkedDebt:
    def test_s7_follow_up_is_marked_not_implemented(self):
        """S7 (the :8408 reality_view.subagent_pin + registry.yaml gate_lane
        entries for the opencode lens legs) has NO in-repo code point in this
        tree — it is its own bind after this PR merges. This test pins the
        debt as MARKED (the spec_review module docstring names it) rather
        than a silent skip: the opencode-side pin is a marked debt rather
        than a silent skip."""
        import lapis_pm.spec_review as sr
        doc = sr.__doc__ or ""
        assert "MARKED DEBT" in doc
        assert "S7" in doc
        assert "subagent_pin" in doc
        # The debt is tracked (not bundled): the gate_lane module is the
        # protocol boundary the follow-up pins against, and it carries the
        # follow-up reference in its docstring.
        doc_lane = gate_lane.__doc__ or ""
        assert "gate-lanes-registry-driven-flashnext-v0" in doc_lane
