"""Tests for the lane-aware council preflight
(gate-lanes-registry-driven-flashnext-v0, S5/S8).

Acceptance 3 (machine, this PR's CI — stubbed registry):
  - Council voicing flashnext builds against the resolved lane; preflight
    probes the right port (:30000, not the hardcoded :8081 SWARM_URL).
  - A readable registry with a dead flashnext lane is an honest leg_down
    (flashnext_not_serving), never a silent mis-skip, never a masked
    legacy fallback.
  - A registry-blind resolution degrades to the legacy probe
    byte-identically (the ONLY fallback case).
  - S8: a local-voiced council leg is lease-free — the doorman
    /lease/acquire call site is unreachable on the local-voicing code
    path; a held gravitywell-seat lease is an honest leg_down (refuse),
    never a fallback-run; blind never refuses.
"""
from __future__ import annotations

import textwrap
from unittest.mock import MagicMock

import pytest

from agents_core.shared_deliberation.envelope import DeliberationEnvelope
from lapis_pm import spec_review
from lapis_pm.spec_review import run_spec_review


def _happy_envelope():
    return DeliberationEnvelope(
        deliberation_request_id="test-req-flashnext",
        triage="full",
        facets_ok=True,
        facets={
            "deliberation_id": "facets-flashnext",
            "synthesis": {
                "escalation_recommendation": "proceed",
                "consensus_level": "strong",
                "confidence": "high",
                "recommendation": "All clear",
            },
            "stances": [],
            "methodology": {"synthesis_operator": "flashnext"},
        },
        facets_deliberation_id="facets-flashnext",
        operator_requested="flashnext",
        operator_effective="flashnext",
        council_ok=True,
        council_run_id="council-flashnext",
        council_status="resolved",
        council_landing="aligns",
        council_confidence="converged",
        council_open_questions=[],
        council_positions=[{"entity": "ent-a", "position": "agree"}],
        council_voicing_requested="flashnext",
        council_voicing_effective="flashnext",
        council_voicing_degraded=False,
    )


@pytest.fixture
def advisory_spec(tmp_path):
    spec_path = tmp_path / "flashnext_preflight_spec.md"
    spec_path.write_text(
        textwrap.dedent("""\
            # Test Spec: flashnext preflight

            **Target ID:** `flashnext-preflight-test`
            **Repo:** `lapis-pm`
            **Authority:** advisory

            ## Goal
            Verify the lane-aware council preflight.
        """),
        encoding="utf-8",
    )
    return spec_path


@pytest.fixture(autouse=True)
def _isolated_lock(tmp_path_factory, monkeypatch):
    lock_dir = tmp_path_factory.mktemp("flashnext-preflight-lock")
    lock_file = lock_dir / "lapis-pm-spec-review.lock"
    monkeypatch.setattr("lapis_pm.spec_review._spec_review_lock_path", lambda: lock_file)
    return lock_file


@pytest.fixture(autouse=True)
def _no_live_registry(monkeypatch):
    """Never a live call from a test: the registry read is injected. The
    default is a BLIND registry (the byte-identical fallback case); tests
    that need a readable registry monkeypatch gate_lane_serving /
    resolve_gate_lane themselves (their patch wins, applied after this)."""
    monkeypatch.setattr("lapis_pm.gate_lane.gate_lane_serving", lambda *a, **k: (False, "registry_blind"))
    monkeypatch.setattr("lapis_pm.gate_lane.resolve_gate_lane", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Acceptance 3: flashnext preflight probes the right port
# ---------------------------------------------------------------------------

class TestFlashnextPreflight:
    def test_flashnext_serving_runs_deliberation(self, advisory_spec, monkeypatch):
        """AC3: with the flashnext lane serving (the preflight probes the
        resolved :30000 lane, not :8081), the deliberation leg runs."""
        monkeypatch.setattr(
            "lapis_pm.gate_lane.gate_lane_serving", lambda *a, **k: (True, "")
        )
        called = []

        async def mock_run_deliberation(request):
            called.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        brief = run_spec_review(
            advisory_spec, council_voicing="flashnext", timeout_s=30, dispatch_facets=True,
        )
        assert len(called) == 1, "a serving flashnext lane must run the deliberation leg"
        assert called[0].council_voicing == "flashnext"
        assert brief.council_status == "resolved"

    def test_flashnext_operator_resolves_30000_lane(self, advisory_spec, monkeypatch):
        """S3 (gate-lanes-registry-driven-flashnext-v0): the ENDPOINT-
        CONSTRUCTION seam — with a stubbed registry flashnext row, the
        flashnext operator/voicing threads lane="flashnext" into
        _gw_primary_url, so the leg endpoints are BUILT against the
        :30000 registry lane, never the :8081 GW_URL default.

        Honest scope (review MED-3 fold, 2026-09-25): this pins that the
        construction call sites pass the lane — the in-leg HTTP routing
        itself resolves inside agents-core (facets S2 operator registry /
        council S4 _build_adapter) and lands with the companion bind
        gate-lanes-registry-driven-flashnext-v0-agents-core. This test is
        the lapis-pm-side seam pin, not a claim that the legs already dial
        the resolved endpoint."""
        monkeypatch.setattr(
            "lapis_pm.gate_lane.gate_lane_serving", lambda *a, **k: (True, "")
        )
        from lapis_pm.gate_lane import FLASHNEXT_SERVED_ID, GateLane
        monkeypatch.setattr(
            "lapis_pm.gate_lane.resolve_gate_lane",
            lambda *a, **k: GateLane(
                name="flashnext",
                base_url="http://203.0.113.11:30000",
                served_model=FLASHNEXT_SERVED_ID,
            ),
        )
        endpoints = []

        def spy_primary_url(lane=None):
            endpoints.append(lane)
            return "http://203.0.113.11:30000"

        monkeypatch.setattr("lapis_pm.spec_review._gw_primary_url", spy_primary_url)
        called = []

        async def mock_run_deliberation(request):
            called.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        brief = run_spec_review(
            advisory_spec,
            council_voicing="flashnext",
            facets_operator="flashnext",
            timeout_s=30,
            dispatch_facets=True,
        )
        assert len(called) == 1, "a serving flashnext lane must run the deliberation leg"
        # The flashnext operator/voicing resolved the :30000 registry lane
        # through the shim — the endpoint-construction call sites passed
        # lane="flashnext" (never lane=None, which would silently build the
        # :8081 GW_URL default).
        assert "flashnext" in endpoints, (
            "the flashnext facets-leg endpoint must resolve the :30000 lane "
            f"through the gate_lane shim; got {endpoints}"
        )
        assert brief.council_status == "resolved"

    def test_flashnext_down_is_honest_leg_down(self, advisory_spec, monkeypatch):
        """A readable registry with a dead flashnext lane is an honest
        leg_down (flashnext_not_serving) — the deliberation leg is skipped,
        the reason is recorded, and it is NEVER a silent mis-skip nor a
        masked legacy fallback."""
        monkeypatch.setattr(
            "lapis_pm.gate_lane.gate_lane_serving",
            lambda *a, **k: (False, "flashnext_not_serving"),
        )
        called = []

        async def mock_run_deliberation(request):
            called.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        brief = run_spec_review(
            advisory_spec, council_voicing="flashnext", timeout_s=30, dispatch_facets=True,
        )
        assert called == [], (
            "a requested-but-dead flashnext lane must skip the deliberation leg "
            "(honest leg_down), never a silent mis-skip or a masked gravitywell "
            "fallback"
        )
        assert brief.council_status == "error"
        assert "flashnext_not_serving" in (brief.council_error_reason or "")

    def test_flashnext_blind_degrades_to_legacy_probe(self, advisory_spec, monkeypatch):
        """The ONLY fallback case: a registry-blind resolution degrades to
        the legacy probe byte-identically. Here the legacy probe
        (swarm_serving on :8081) is False -> the legacy gw_not_serving
        reason (the exact today behavior for a blind gate leg)."""
        # The autouse _no_live_registry stub already returns
        # (False, "registry_blind") for gate_lane_serving.
        monkeypatch.setattr("lapis_pm.spec_review.swarm_serving", lambda: False)
        called = []

        async def mock_run_deliberation(request):
            called.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        brief = run_spec_review(
            advisory_spec, council_voicing="flashnext", timeout_s=30, dispatch_facets=True,
        )
        assert called == []
        assert brief.council_status == "error"
        # registry_blind degrades to the legacy gw_not_serving reason
        # byte-identically (the ONLY fallback case).
        assert brief.council_error_reason == "gw_not_serving"

    def test_flashnext_blind_with_serving_gw_runs_legacy_path(self, advisory_spec, monkeypatch):
        """LOW-6 fold (independent review 2026-09-25): the S1/S2 caller
        contract says blind = "caller falls back to the gravitywell path
        UNCHANGED". With the registry blind AND the legacy GW probe
        SERVING, the deliberation leg must RUN on the gravitywell path
        (reported degrade) — the prior code killed the leg with an
        unprobed gw_not_serving reason without ever consulting
        swarm_serving()."""
        # The autouse _no_live_registry stub returns (False,
        # "registry_blind") for gate_lane_serving.
        monkeypatch.setattr("lapis_pm.spec_review.swarm_serving", lambda: True)
        called = []

        async def mock_run_deliberation(request):
            called.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        brief = run_spec_review(
            advisory_spec, council_voicing="flashnext", timeout_s=30, dispatch_facets=True,
        )
        assert len(called) == 1, (
            "blind + serving GW must fall back to the gravitywell path "
            "UNCHANGED (S1 caller contract), not skip the leg on an "
            "unprobed reason"
        )
        # The fallback runs on the legacy voicing (byte-identical
        # gravitywell path), not an un-honored flashnext voicing.
        assert called[0].council_voicing == "gravitywell"
        assert brief.council_status == "resolved"

    def test_gravitywell_preflight_unchanged(self, advisory_spec, monkeypatch):
        """The gravitywell leg is byte-identical to today: swarm_serving()
        on :8081, env-var driven, never registry-probing."""
        monkeypatch.setattr("lapis_pm.spec_review.swarm_serving", lambda: False)
        called = []

        async def mock_run_deliberation(request):
            called.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        brief = run_spec_review(
            advisory_spec, council_voicing="gravitywell", timeout_s=30, dispatch_facets=True,
        )
        assert called == []
        assert brief.council_error_reason == "gw_not_serving"


# ---------------------------------------------------------------------------
# S8: local voicing is lease-free
# ---------------------------------------------------------------------------

class TestLocalVoicingLeaseFree:
    """S8 acceptance: (a) zero lease calls when voicing=local AND (b) the
    doorman /lease/acquire call site is unreachable on the local-voicing
    code path — the 409 contention shape is pinned unreachable, not merely
    that a lease-call count is zero."""

    def test_local_voicing_blind_doorman_runs_deliberation(self, advisory_spec, monkeypatch):
        """Blind never refuses: the doorman is unreachable -> the local
        voicing runs the deliberation leg lease-free (the 409 shape
        requires a reachable doorman with a held lease)."""
        # The autouse _no_live_registry stub returns (True, "") for the
        # local lane (blind doorman -> never refuses) — re-stub explicitly.
        monkeypatch.setattr(
            "lapis_pm.gate_lane.gate_lane_serving", lambda *a, **k: (True, "")
        )
        called = []

        async def mock_run_deliberation(request):
            called.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        brief = run_spec_review(
            advisory_spec, council_voicing="local", timeout_s=30, dispatch_facets=True,
        )
        assert len(called) == 1, (
            "a blind doorman never refuses: the local voicing runs lease-free"
        )
        assert called[0].council_voicing == "local"

    def test_local_voicing_refused_when_lease_held(self, advisory_spec, monkeypatch):
        """A held gravitywell-seat lease -> honest leg_down (refuse), never
        a fallback-run on the gravitywell lane. The doorman /lease/acquire
        call site is unreachable on the local-voicing code path: the guard
        refuses BEFORE the leg would acquire (the 409 contention shape is
        pinned unreachable)."""
        monkeypatch.setattr(
            "lapis_pm.gate_lane.gate_lane_serving",
            lambda *a, **k: (False, "local_seat_lease_refused"),
        )
        called = []

        async def mock_run_deliberation(request):
            called.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        brief = run_spec_review(
            advisory_spec, council_voicing="local", timeout_s=30, dispatch_facets=True,
        )
        assert called == [], (
            "a refused local voicing must skip the deliberation leg (honest "
            "leg_down), never a fallback-run on the gravitywell lane"
        )
        assert brief.council_status == "error"
        assert "local_seat_lease_refused" in (brief.council_error_reason or "")

    def test_local_voicing_refused_on_the_live_409_shape(self, advisory_spec, monkeypatch):
        """HIGH-2 fold (independent review 2026-09-25): the OBSERVED live
        409 shape is the GW seat not serving + the flashnext window up,
        with lease_count 0 — the guard (pinned in tests/test_gate_lane.py
        against the exact /status snapshot) refuses with
        local_voicing_flashnext_window and the preflight must skip the
        deliberation leg on that reason (honest leg_down), NOT run it into
        the acquiring council submit path that died 409 live."""
        monkeypatch.setattr(
            "lapis_pm.gate_lane.gate_lane_serving",
            lambda *a, **k: (False, "local_voicing_flashnext_window"),
        )
        called = []

        async def mock_run_deliberation(request):
            called.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)

        brief = run_spec_review(
            advisory_spec, council_voicing="local", timeout_s=30, dispatch_facets=True,
        )
        assert called == [], (
            "the live-409 seat state must be refused BEFORE the leg would "
            "acquire — the 409 contention shape pinned unreachable"
        )
        assert brief.council_status == "error"
        assert "local_voicing_flashnext_window" in (brief.council_error_reason or "")
