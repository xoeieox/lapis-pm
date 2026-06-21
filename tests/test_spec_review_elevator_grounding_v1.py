"""Tests for elevator-based grounding producer in run_spec_review.

Covers AC1-AC8 for lapis-pm-spec-review-grounding-on-swarm-v1.
Implementation: ELEVATOR_ACTIVE-gated pre-grounding step in run_spec_review,
before DeliberationRequest construction. Sets grounding_result_file= on the
request (not argv injection — H4 migration removed _dispatch_facets).
"""
import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest
from agents_core.shared_deliberation.envelope import DeliberationEnvelope

from lapis_pm import spec_review


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spec_fixture(tmp_path):
    """Minimal valid advisory spec file."""
    spec_path = tmp_path / "test_spec.md"
    spec_path.write_text(
        """# Test Spec

**Target ID:** `test-grounding-v1`
**Repo:** `test-repo`
**Authority:** advisory

## Goal
Test elevator grounding producer v1.
""",
        encoding="utf-8",
    )
    return spec_path


def _happy_envelope():
    return DeliberationEnvelope(
        deliberation_request_id="req-123",
        triage="full",
        facets_ok=True,
        facets={
            "deliberation_id": "facets-123",
            "synthesis": {
                "escalation_recommendation": "proceed",
                "consensus_level": "strong",
                "confidence": "high",
                "recommendation": "All clear",
            },
            "stances": [],
            "methodology": {"synthesis_operator": "haiku"},
        },
        facets_deliberation_id="facets-123",
        operator_requested="haiku",
        operator_effective="haiku",
        council_ok=True,
        council_run_id="council-456",
        council_status="resolved",
        council_landing="Land this spec.",
        council_confidence="converged",
        council_open_questions=[],
        council_positions=[],
        council_voicing_requested="gravitywell",
        council_voicing_effective="gravitywell",
        council_voicing_degraded=False,
        council_voicing_degraded_reason="",
    )


@pytest.fixture(autouse=True)
def mock_infra(monkeypatch):
    """Stub out infrastructure that requires live repos/services."""
    monkeypatch.setenv("SPEC_REVIEW_SONNET_DISABLED", "1")
    monkeypatch.setattr(
        "lapis_pm.spec_review._load_invariant_context",
        lambda repo: "mock-invariant-context",
    )
    monkeypatch.setattr(
        "lapis_pm.spec_review._dispatch_gw_reviewer",
        mock.Mock(side_effect=Exception("gw-disabled-in-test")),
    )
    monkeypatch.setattr(
        "lapis_pm.spec_review._dispatch_spec_reviewer",
        mock.Mock(return_value=None),
    )


# ---------------------------------------------------------------------------
# _swarm_serving unit tests
# ---------------------------------------------------------------------------

class TestSwarmServing:
    def test_both_endpoints_ok(self):
        with mock.patch("requests.get") as mock_get:
            health = mock.Mock()
            health.status_code = 200
            models = mock.Mock()
            models.status_code = 200
            models.json.return_value = {"data": [{"id": "m1"}]}
            mock_get.side_effect = [health, models]
            assert spec_review._swarm_serving() is True

    def test_health_fails(self):
        with mock.patch("requests.get") as mock_get:
            health = mock.Mock()
            health.status_code = 503
            mock_get.return_value = health
            assert spec_review._swarm_serving() is False

    def test_models_empty(self):
        with mock.patch("requests.get") as mock_get:
            health = mock.Mock()
            health.status_code = 200
            models = mock.Mock()
            models.status_code = 200
            models.json.return_value = {"data": []}
            mock_get.side_effect = [health, models]
            assert spec_review._swarm_serving() is False

    def test_connection_error(self):
        with mock.patch("requests.get", side_effect=ConnectionError()):
            assert spec_review._swarm_serving() is False


# ---------------------------------------------------------------------------
# Integration: elevator grounding via run_spec_review
# ---------------------------------------------------------------------------

class TestElevatorGroundingV1:
    """Tests that the pre-grounding step in run_spec_review sets
    grounding_result_file correctly on the DeliberationRequest."""

    def _run(self, spec_fixture, monkeypatch, captured,
             swarm_serving=True, enqueue_resp=None, get_responses=None):
        """Helper: run run_spec_review, capturing the DeliberationRequest."""
        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: swarm_serving)

        post_mock = mock.Mock(return_value=enqueue_resp) if enqueue_resp else mock.Mock()

        if get_responses is not None:
            resp_iter = iter(get_responses)
            def get_side(*a, **kw):
                try:
                    return next(resp_iter)
                except StopIteration:
                    r = mock.Mock()
                    r.status_code = 200
                    r.json.return_value = {"status": "pending"}
                    return r
            get_mock = mock.Mock(side_effect=get_side)
        else:
            get_mock = mock.Mock()

        with mock.patch("requests.post", post_mock), \
             mock.patch("requests.get", get_mock), \
             mock.patch("time.sleep"):
            brief = spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )
        return brief, post_mock, get_mock

    def test_ac1_held_path_parity(self, spec_fixture, monkeypatch):
        """AC1: ELEVATOR_ACTIVE unset → grounding_result_file=None, no POST."""
        monkeypatch.delenv("ELEVATOR_ACTIVE", raising=False)
        captured = []
        brief, post_mock, _ = self._run(spec_fixture, monkeypatch, captured)
        assert len(captured) == 1
        assert captured[0].grounding_result_file is None
        post_mock.assert_not_called()

    def test_ac1_held_path_false_string(self, spec_fixture, monkeypatch):
        """AC1: ELEVATOR_ACTIVE='false' → held path, no enqueue."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "false")
        captured = []
        _, post_mock, _ = self._run(spec_fixture, monkeypatch, captured)
        assert captured[0].grounding_result_file is None
        post_mock.assert_not_called()

    def test_ac2_active_enqueue_well_formed(self, spec_fixture, monkeypatch):
        """AC2: ELEVATOR_ACTIVE=true + swarm serving → POSTs well-formed grounding item (no depends_on)."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")
        # timeout=0 → poll loop exits immediately after the enqueue (elapsed >= 0 always True)
        monkeypatch.setenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "0")

        captured = []

        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        enqueue_resp = mock.Mock()
        enqueue_resp.status_code = 201
        enqueue_resp.json.return_value = {"item_id": "grnd-ac2"}

        post_mock = mock.Mock(return_value=enqueue_resp)

        with mock.patch("requests.post", post_mock), \
             mock.patch("requests.get", mock.Mock()):
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        assert post_mock.called
        body = post_mock.call_args[1]["json"]
        assert body["lane"] == "execution"
        assert body["kind"] == "grounding"
        assert body["payload"]["backend"] == "swarm"
        assert body["payload"]["target_repo"] == "/srv/git/test-repo-working"
        assert "spec_text" in body["payload"]
        assert "depends_on" not in body

    def test_ac3_served_handoff_sets_field(self, spec_fixture, monkeypatch):
        """AC3: served result with content → grounding_result_file set on DeliberationRequest."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")
        monkeypatch.setenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "60")

        captured = []

        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        enqueue_resp = mock.Mock()
        enqueue_resp.status_code = 201
        enqueue_resp.json.return_value = {"item_id": "grnd-ac3"}

        poll_resp = mock.Mock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {
            "status": "served",
            "result": {"grounding": "data", "model": "swarm"},
        }

        with mock.patch("requests.post", mock.Mock(return_value=enqueue_resp)), \
             mock.patch("requests.get", mock.Mock(return_value=poll_resp)), \
             mock.patch("time.sleep"):
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        assert len(captured) == 1
        req = captured[0]
        # grounding_result_file was set (file is cleaned up after deliberation)
        assert req.grounding_result_file is not None
        assert isinstance(req.grounding_result_file, str)

    def test_ac3b_empty_result_degrades_to_none(self, spec_fixture, monkeypatch):
        """AC3b: served but result=None → grounding_result_file=None, never a bad path."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")
        monkeypatch.setenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "60")

        captured = []

        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        enqueue_resp = mock.Mock()
        enqueue_resp.status_code = 201
        enqueue_resp.json.return_value = {"item_id": "grnd-empty"}

        poll_resp = mock.Mock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "served", "result": None}

        with mock.patch("requests.post", mock.Mock(return_value=enqueue_resp)), \
             mock.patch("requests.get", mock.Mock(return_value=poll_resp)), \
             mock.patch("time.sleep"):
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        assert captured[0].grounding_result_file is None

    def test_ac3b_empty_string_result_degrades(self, spec_fixture, monkeypatch):
        """AC3b: served with empty string result → grounding_result_file=None."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")
        monkeypatch.setenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "60")

        captured = []

        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        enqueue_resp = mock.Mock()
        enqueue_resp.status_code = 201
        enqueue_resp.json.return_value = {"item_id": "grnd-empty-str"}

        poll_resp = mock.Mock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "served", "result": ""}

        with mock.patch("requests.post", mock.Mock(return_value=enqueue_resp)), \
             mock.patch("requests.get", mock.Mock(return_value=poll_resp)), \
             mock.patch("time.sleep"):
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        assert captured[0].grounding_result_file is None

    def test_ac4_fallback_on_item_failed(self, spec_fixture, monkeypatch):
        """AC4: item status=failed → grounding_result_file=None, gate still produces verdict."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")
        monkeypatch.setenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "60")

        captured = []

        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        enqueue_resp = mock.Mock()
        enqueue_resp.status_code = 201
        enqueue_resp.json.return_value = {"item_id": "grnd-failed"}

        poll_resp = mock.Mock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "failed"}

        with mock.patch("requests.post", mock.Mock(return_value=enqueue_resp)), \
             mock.patch("requests.get", mock.Mock(return_value=poll_resp)), \
             mock.patch("time.sleep"):
            brief = spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        assert captured[0].grounding_result_file is None
        assert brief is not None

    def test_ac4_fallback_on_expired(self, spec_fixture, monkeypatch):
        """AC4: item status=expired → grounding_result_file=None."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")
        monkeypatch.setenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "60")

        captured = []

        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        enqueue_resp = mock.Mock()
        enqueue_resp.status_code = 201
        enqueue_resp.json.return_value = {"item_id": "grnd-expired"}

        poll_resp = mock.Mock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "expired"}

        with mock.patch("requests.post", mock.Mock(return_value=enqueue_resp)), \
             mock.patch("requests.get", mock.Mock(return_value=poll_resp)), \
             mock.patch("time.sleep"):
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        assert captured[0].grounding_result_file is None

    def test_ac5_bounded_poll(self, spec_fixture, monkeypatch):
        """AC5: poll loop terminates within ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")
        monkeypatch.setenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "1")

        def mock_deliberation(request):
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        enqueue_resp = mock.Mock()
        enqueue_resp.status_code = 201
        enqueue_resp.json.return_value = {"item_id": "bounded"}

        poll_resp = mock.Mock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "pending"}

        wall_start = time.time()
        with mock.patch("requests.post", mock.Mock(return_value=enqueue_resp)), \
             mock.patch("requests.get", mock.Mock(return_value=poll_resp)):
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )
        elapsed = time.time() - wall_start
        # Should complete well under 30s (the 1s timeout allows only 1 real sleep)
        assert elapsed < 30, f"Poll took {elapsed}s, expected < 30s"

    def test_ac5b_claim_stall_fast_fallback(self, spec_fixture, monkeypatch, capsys):
        """AC5b: item stuck pending > claim deadline → fall back before full poll timeout; logs swarm-stall-unclaimed."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")
        monkeypatch.setenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "180")
        monkeypatch.setenv("ELEVATOR_GROUNDING_CLAIM_DEADLINE_SEC", "75")

        captured = []

        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        enqueue_resp = mock.Mock()
        enqueue_resp.status_code = 201
        enqueue_resp.json.return_value = {"item_id": "stall-item"}

        poll_resp = mock.Mock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "pending"}

        with mock.patch("requests.post", mock.Mock(return_value=enqueue_resp)), \
             mock.patch("requests.get", mock.Mock(return_value=poll_resp)), \
             mock.patch("time.sleep"), \
             mock.patch("time.time") as mock_time:
            # time.time() calls in order: start_time(1282), poll_start(1446),
            # first loop elapsed(1449) → 80s > 75s claim deadline fires,
            # final elapsed(1689) → any value
            mock_time.side_effect = [1000.0, 1000.0, 1080.0, 1001.0]
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        assert captured[0].grounding_result_file is None
        err = capsys.readouterr().err
        assert "swarm-stall-unclaimed" in err

    def test_ac6_provenance_swarm_served(self, spec_fixture, monkeypatch, capsys):
        """AC6: On swarm-served, logs grounding_path=swarm:<item_id>."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")
        monkeypatch.setenv("ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC", "60")

        def mock_deliberation(request):
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        enqueue_resp = mock.Mock()
        enqueue_resp.status_code = 201
        enqueue_resp.json.return_value = {"item_id": "prov-item-xyz"}

        poll_resp = mock.Mock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {"status": "served", "result": {"g": "data"}}

        with mock.patch("requests.post", mock.Mock(return_value=enqueue_resp)), \
             mock.patch("requests.get", mock.Mock(return_value=poll_resp)), \
             mock.patch("time.sleep"):
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        err = capsys.readouterr().err
        assert "grounding_path=swarm:prov-item-xyz" in err

    def test_ac6_provenance_inline_fallback(self, spec_fixture, monkeypatch, capsys):
        """AC6: On fallback (swarm not serving), logs grounding_path=inline:<reason>."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")

        def mock_deliberation(request):
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: False)

        with mock.patch("requests.post", mock.Mock()):
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        err = capsys.readouterr().err
        assert "grounding_path=inline:" in err

    def test_ac7_readiness_precheck_no_enqueue(self, spec_fixture, monkeypatch, capsys):
        """AC7: swarm_serving()=False → no POST, grounding_result_file=None, logs swarm-not-serving."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")

        captured = []

        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: False)

        post_mock = mock.Mock()
        with mock.patch("requests.post", post_mock):
            spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        post_mock.assert_not_called()
        assert captured[0].grounding_result_file is None
        err = capsys.readouterr().err
        assert "swarm-not-serving" in err

    def test_ac4_store_unreachable_fallback(self, spec_fixture, monkeypatch):
        """AC4: store unreachable (enqueue raises) → grounding_result_file=None, no crash."""
        monkeypatch.setenv("ELEVATOR_ACTIVE", "true")
        monkeypatch.setenv("ELEVATOR_STORE_URL", "http://test-store:8405")

        captured = []

        def mock_deliberation(request):
            captured.append(request)
            return _happy_envelope()

        monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_deliberation)
        monkeypatch.setattr("lapis_pm.spec_review._swarm_serving", lambda: True)

        with mock.patch("requests.post", side_effect=ConnectionError("refused")):
            brief = spec_review.run_spec_review(
                spec_path=spec_fixture,
                dispatch_facets=True,
                authority="advisory",
            )

        assert captured[0].grounding_result_file is None
        assert brief is not None
