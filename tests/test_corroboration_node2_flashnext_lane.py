"""Tests for the corroboration node2 flashnext lane
(gate-lanes-registry-driven-flashnext-v0, S6).

Acceptance 2 (machine, this PR's CI — stubbed registry):
  - The node2 leg builds against the flashnext lane (registry-served model
    name on :30000) under a flashnext-solo stub, via the existing node_url
    injection point.
  - node2_unavailable fires ONLY when :30000 is actually unreachable, with
    an honest panel_starvation row (leg_status != "ok" -> the
    cross_node_divergence="node2_unavailable" row).
  - The legacy Phala/`local` paths are unchanged: a blind registry (the
    ONLY fallback case) falls back to the sanctioned PhalaTeeClient path
    byte-identically.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import gate_lane
from lapis_pm.corroboration_adapter import (
    _flashnext_lane_url,
    run_corroboration_pass,
)

FLASH_SERVED_ID = "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
FLASH_URL = "http://203.0.113.11:30000/v1/chat/completions"


def _flashnext_solo_payload():
    return {
        "reality_view": {
            "reality": "flashnext-solo",
            "primary": {"port": 30000, "model_root": "/models/models/Qwen3.8-Flash-Next-NVFP4-SSD-Stream"},
        },
        "seats": [
            {"port": 8081, "state": "down", "model": None, "model_root": None, "bind": None},
            {"port": 30000, "state": "serving", "engine": "sglang",
             "model": FLASH_SERVED_ID,
             "model_root": "/models/models/Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
             "bind": "203.0.113.11"},
        ],
    }


def _ok_completion(model: str):
    """A minimal OpenAI-shaped chat completion body."""
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": json.dumps({
                "verdict": "clean",
                "claims": [],
                "summary": "all identifiers present",
            })},
            "finish_reason": "stop",
        }],
    }


# ---------------------------------------------------------------------------
# _flashnext_lane_url seam
# ---------------------------------------------------------------------------

class TestFlashnextLaneUrl:
    def test_resolves_flashnext_lane(self, monkeypatch):
        """Under a flashnext-solo stub, the lane seam resolves the
        :30000 base_url + the registry-served model name."""
        monkeypatch.setattr(
            gate_lane, "resolve_gate_lane",
            lambda lane=None, fetcher=None: (
                gate_lane.GateLane(name="flashnext", base_url="http://203.0.113.11:30000",
                                   served_model=FLASH_SERVED_ID)
                if lane == "flashnext" else None
            ),
        )
        base_url, model = _flashnext_lane_url()
        assert base_url == "http://203.0.113.11:30000"
        assert model == FLASH_SERVED_ID

    def test_blind_registry_is_none(self, monkeypatch):
        """A blind registry (the ONLY fallback case) -> (None, None): the
        caller falls back to the legacy Phala path byte-identically."""
        monkeypatch.setattr(gate_lane, "resolve_gate_lane", lambda *a, **k: None)
        assert _flashnext_lane_url() == (None, None)


# ---------------------------------------------------------------------------
# node2 leg builds against the flashnext lane (acceptance 2)
# ---------------------------------------------------------------------------

class TestNode2FlashnextLane:
    def _adapter(self):
        from lapis_pm.corroboration_adapter import LapisPMReviewerAdapter
        return LapisPMReviewerAdapter()

    def test_node2_builds_against_flashnext_lane(self, monkeypatch):
        """AC2: under a flashnext-solo stub, the node2 leg builds against
        the flashnext lane — node_url :30000 + the registry-served model
        name — via the existing node_url injection point (the raw-POST
        path, not the PhalaTeeClient path)."""
        adapter = self._adapter()
        substrates = [
            __import__("lapis_pm.corroboration_adapter", fromlist=["_IdentifierSubstrate"])
            ._IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], []),
        ]

        captured: dict = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["model"] = json.get("model")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = _ok_completion(FLASH_SERVED_ID)
            resp.raise_for_status.return_value = None
            return resp

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=fake_post),
        ):
            result = adapter.score(
                "some diff", substrates, "lapis-pm",
                node_url="http://203.0.113.11:30000",
                node_model=FLASH_SERVED_ID,
                include_vault=False,
            )

        assert result.leg_status == "ok"
        assert result.verdict == "clean"
        assert captured["url"] == "http://203.0.113.11:30000"
        assert captured["model"] == FLASH_SERVED_ID

    def test_node2_unavailable_only_when_30000_unreachable(self, monkeypatch):
        """AC2: node2_unavailable fires ONLY when :30000 is actually
        unreachable — with an honest panel_starvation row (leg_status
        != "ok")."""
        adapter = self._adapter()
        substrates = [
            __import__("lapis_pm.corroboration_adapter", fromlist=["_IdentifierSubstrate"])
            ._IdentifierSubstrate("tick", [], []),
        ]

        with patch("lapis_pm.corroboration_adapter.node_reachable", return_value=False):
            result = adapter.score(
                "some diff", substrates, "lapis-pm",
                node_url="http://203.0.113.11:30000",
                node_model=FLASH_SERVED_ID,
                include_vault=False,
            )

        # The honest panel_starvation row: leg_status != "ok" (the field
        # consumers read instead of string-matching notes prose).
        assert result.leg_status == "substrate_unavailable"
        assert result.verdict == "uncertain"
        assert "node unreachable" in (result.notes or "")

    def test_run_corroboration_pass_node2_rides_flashnext_lane(self, monkeypatch):
        """The run_corroboration_pass node2 leg, under a flashnext-solo
        stub, builds against the flashnext lane via the node_url injection
        point — and the honest node2_unavailable row (cross_node_divergence)
        is set when :30000 is actually unreachable."""
        # Stub the lane seam: flashnext lane is registered + serving.
        monkeypatch.setattr(
            "lapis_pm.corroboration_adapter._flashnext_lane_url",
            lambda: ("http://203.0.113.11:30000", FLASH_SERVED_ID),
        )
        # :30000 is actually unreachable -> honest node2_unavailable.
        monkeypatch.setattr(
            "lapis_pm.corroboration_adapter.node_reachable", lambda *a, **k: False
        )
        # node1 (the local GW seat) is fine.
        monkeypatch.setattr(
            "lapis_pm.corroboration_adapter._llm_url",
            lambda: "http://203.0.113.11:8081/v1/chat/completions",
        )

        def fake_post(url, json=None, headers=None, timeout=None):
            if url.startswith("http://203.0.113.11:30000"):
                raise ConnectionError("refused")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = _ok_completion("gravitywell-27b")
            resp.raise_for_status.return_value = None
            return resp

        with patch("httpx.post", side_effect=fake_post):
            combined = run_corroboration_pass(
                "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+def tick(): pass\n",
                "lapis-pm",
            )

        # The honest panel_starvation row: node2_unavailable.
        assert combined["cross_node_divergence"] == "node2_unavailable"
        n2 = combined["node2_corroboration"]
        assert n2["leg_status"] != "ok"

    def test_run_corroboration_pass_node2_ok_on_flashnext_lane(self, monkeypatch):
        """The node2 leg succeeds on the flashnext lane when :30000 is
        serving — eliminating the node2_unavailable starvation class under
        flashnext-solo."""
        monkeypatch.setattr(
            "lapis_pm.corroboration_adapter._flashnext_lane_url",
            lambda: ("http://203.0.113.11:30000", FLASH_SERVED_ID),
        )
        monkeypatch.setattr(
            "lapis_pm.corroboration_adapter.node_reachable", lambda *a, **k: True
        )
        monkeypatch.setattr(
            "lapis_pm.corroboration_adapter._llm_url",
            lambda: "http://203.0.113.11:8081/v1/chat/completions",
        )

        def fake_post(url, json=None, headers=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = _ok_completion(
                json.get("model") if json else "unknown"
            )
            resp.raise_for_status.return_value = None
            return resp

        with patch("httpx.post", side_effect=fake_post):
            combined = run_corroboration_pass(
                "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+def tick(): pass\n",
                "lapis-pm",
            )

        # No starvation: node2 ran on the flashnext lane and agreed.
        assert combined["cross_node_divergence"] in ("agree", "diverge")
        n2 = combined["node2_corroboration"]
        assert n2["leg_status"] == "ok"

    def test_blind_registry_falls_back_to_phala_path(self, monkeypatch):
        """A blind registry (the ONLY fallback case) -> the node2 leg takes
        the legacy sanctioned PhalaTeeClient path byte-identically (via
        via_node2_client=True)."""
        monkeypatch.setattr(
            "lapis_pm.corroboration_adapter._flashnext_lane_url",
            lambda: (None, None),
        )
        # The Phala client import is a wiring failure in this test -> the
        # named node2_client_import_error class (the legacy path was taken,
        # not the flashnext lane path).
        with patch("lapis_pm.corroboration_adapter._build_node2_client",
                   side_effect=ImportError("no phala client in test")):
            combined = run_corroboration_pass(
                "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+def tick(): pass\n",
                "lapis-pm",
            )
        # The legacy path was taken (the named client-import class is the
        # PhalaTeeClient path's wiring-defect marker — the flashnext lane
        # path never calls _build_node2_client).
        assert combined["cross_node_divergence"] == "node2_unavailable"
        n2 = combined["node2_corroboration"]
        assert "node2_client_import_error" in (n2["notes"] or "")
