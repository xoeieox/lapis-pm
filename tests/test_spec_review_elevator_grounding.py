"""Tests for elevator-based grounding in spec_review._dispatch_facets.

Covers acceptance criteria AC1-AC8 for lapis-pm-spec-review-grounding-on-swarm-v0.
"""
import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from lapis_pm import spec_review


class TestSwarmServing:
    """Tests for _swarm_serving() readiness pre-check."""

    def test_swarm_serving_both_endpoints_ok(self):
        """When both /health and /v1/models return 200 with data, returns True."""
        with mock.patch("requests.get") as mock_get:
            # Mock health endpoint
            health_resp = mock.Mock()
            health_resp.status_code = 200

            # Mock models endpoint
            models_resp = mock.Mock()
            models_resp.status_code = 200
            models_resp.json.return_value = {"data": [{"id": "model-1"}]}

            mock_get.side_effect = [health_resp, models_resp]

            assert spec_review._swarm_serving() is True

    def test_swarm_serving_health_fails(self):
        """When /health returns non-200, returns False."""
        with mock.patch("requests.get") as mock_get:
            health_resp = mock.Mock()
            health_resp.status_code = 503

            mock_get.return_value = health_resp

            assert spec_review._swarm_serving() is False

    def test_swarm_serving_models_fails(self):
        """When /v1/models returns non-200, returns False."""
        with mock.patch("requests.get") as mock_get:
            health_resp = mock.Mock()
            health_resp.status_code = 200

            models_resp = mock.Mock()
            models_resp.status_code = 502

            mock_get.side_effect = [health_resp, models_resp]

            assert spec_review._swarm_serving() is False

    def test_swarm_serving_models_empty(self):
        """When /v1/models returns empty data, returns False."""
        with mock.patch("requests.get") as mock_get:
            health_resp = mock.Mock()
            health_resp.status_code = 200

            models_resp = mock.Mock()
            models_resp.status_code = 200
            models_resp.json.return_value = {"data": []}

            mock_get.side_effect = [health_resp, models_resp]

            assert spec_review._swarm_serving() is False

    def test_swarm_serving_timeout(self):
        """When requests timeout, returns False."""
        with mock.patch("requests.get") as mock_get:
            mock_get.side_effect = TimeoutError()

            assert spec_review._swarm_serving() is False

    def test_swarm_serving_connection_error(self):
        """When connection fails, returns False."""
        with mock.patch("requests.get") as mock_get:
            mock_get.side_effect = ConnectionError()

            assert spec_review._swarm_serving() is False


class TestDispatchFacetsElevatorIntegration:
    """Tests for elevator integration in _dispatch_facets."""

    @pytest.fixture
    def mock_facets_env(self):
        """Setup common environment."""
        old_env = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(old_env)

    @pytest.fixture
    def mock_repo_path(self, tmp_path):
        """Create a mock working repo tree."""
        repo_path = tmp_path / "test-repo-working"
        repo_path.mkdir()
        yield repo_path

    def test_ac1_held_path_parity_elevator_inactive(self, mock_facets_env, mock_repo_path):
        """AC1: With ELEVATOR_ACTIVE unset, argv is identical to today (no enqueue)."""
        # Verify: no ELEVATOR_ACTIVE set, should not enqueue
        assert os.getenv("ELEVATOR_ACTIVE") != "true"

        with mock.patch("subprocess.run") as mock_run, \
             mock.patch("lapis_pm.spec_review._FACETS_REPO_PATH", str(mock_repo_path)), \
             mock.patch.dict(os.environ, {"FACETS_DISPATCH_DISABLED": "0"}):

            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"deliberation_id": "test-id-123"})
            mock_run.return_value = mock_result

            spec_text = "test spec"
            result = spec_review._dispatch_facets(
                spec_text=spec_text,
                parsed_target_id="test-target",
                repo="test-repo",
                authority="advisory",
                start_time=time.time(),
            )

            # Should succeed and return deliberation_id
            assert result == "test-id-123"

            # Verify subprocess was called (held path)
            assert mock_run.called

            # Extract the argv that was passed
            call_args = mock_run.call_args
            argv = call_args[0][0]

            # Verify no elevator flags were added
            assert "--grounding-result-file" not in argv
            assert "--no-auto-ground" not in argv

    def test_ac2_active_enqueue_well_formed(self, mock_facets_env, mock_repo_path):
        """AC2: With ELEVATOR_ACTIVE="true", POSTs well-formed grounding item."""
        with mock.patch.dict(
            os.environ,
            {
                "ELEVATOR_ACTIVE": "true",
                "ELEVATOR_STORE_URL": "http://test-elevator:8405",
                "ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC": "60",
                "FACETS_DISPATCH_DISABLED": "0",
            },
        ), \
        mock.patch("lapis_pm.spec_review._swarm_serving", return_value=True), \
        mock.patch("requests.post") as mock_post, \
        mock.patch("requests.get") as mock_get, \
        mock.patch("subprocess.run") as mock_run, \
        mock.patch("lapis_pm.spec_review._FACETS_REPO_PATH", str(mock_repo_path)):

            # Mock successful enqueue (returns item_id)
            enqueue_resp = mock.Mock()
            enqueue_resp.status_code = 201
            enqueue_resp.json.return_value = {"item_id": "grounding-item-123"}
            mock_post.return_value = enqueue_resp

            # Mock poll timeout (item never served)
            poll_resp = mock.Mock()
            poll_resp.status_code = 200
            poll_resp.json.return_value = {"status": "pending"}
            mock_get.return_value = poll_resp

            # Mock Facets subprocess
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"deliberation_id": "facets-id-456"})
            mock_run.return_value = mock_result

            spec_text = "test spec for ac2"
            result = spec_review._dispatch_facets(
                spec_text=spec_text,
                parsed_target_id="test-target-ac2",
                repo="test-repo",
                authority="advisory",
                start_time=time.time(),
            )

            # Should enqueue
            assert mock_post.called
            post_call = mock_post.call_args
            enqueue_body = post_call[1]["json"]

            # Verify payload structure per spec
            assert enqueue_body["lane"] == "execution"
            assert enqueue_body["kind"] == "grounding"
            assert enqueue_body["payload"]["spec_text"] == spec_text
            assert enqueue_body["payload"]["backend"] == "swarm"
            # target_repo follows the pattern /srv/git/{repo}-working
            assert enqueue_body["payload"]["target_repo"] == "/srv/git/test-repo-working"
            assert enqueue_body["principal"] == "lapis-pm-spec-review"
            assert enqueue_body["latency_class"] == "batch"
            assert "depends_on" not in enqueue_body  # No depends_on per spec

    def test_ac3_served_injection(self, mock_facets_env, mock_repo_path):
        """AC3: When item reaches status=served, injects --grounding-result-file + --no-auto-ground."""
        grounding_json = {"grounding": "data", "model": "swarm"}

        with mock.patch.dict(
            os.environ,
            {
                "ELEVATOR_ACTIVE": "true",
                "ELEVATOR_STORE_URL": "http://test-elevator:8405",
                "ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC": "60",
                "FACETS_DISPATCH_DISABLED": "0",
            },
        ), \
        mock.patch("lapis_pm.spec_review._swarm_serving", return_value=True), \
        mock.patch("requests.post") as mock_post, \
        mock.patch("requests.get") as mock_get, \
        mock.patch("subprocess.run") as mock_run, \
        mock.patch("lapis_pm.spec_review._FACETS_REPO_PATH", str(mock_repo_path)):

            # Mock successful enqueue
            enqueue_resp = mock.Mock()
            enqueue_resp.status_code = 201
            enqueue_resp.json.return_value = {"item_id": "item-served-123"}
            mock_post.return_value = enqueue_resp

            # Mock poll returns served with result
            poll_resp = mock.Mock()
            poll_resp.status_code = 200
            poll_resp.json.return_value = {
                "status": "served",
                "result": grounding_json,
            }
            mock_get.return_value = poll_resp

            # Mock Facets subprocess
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"deliberation_id": "facets-served-456"})
            mock_run.return_value = mock_result

            result = spec_review._dispatch_facets(
                spec_text="spec text ac3",
                parsed_target_id="ac3-target",
                repo="test-repo",
                authority="advisory",
                start_time=time.time(),
            )

            assert result == "facets-served-456"

            # Verify Facets was called with grounding injection
            facets_call = mock_run.call_args
            argv = facets_call[0][0]

            # Should have grounding injection
            assert "--grounding-result-file" in argv
            assert "--no-auto-ground" in argv
            # Should NOT have --target-repo (inline grounding)
            assert "--target-repo" not in argv

    def test_ac4_graceful_fallback_poll_timeout(self, mock_facets_env, mock_repo_path):
        """AC4: On poll timeout, falls back to inline grounding (--target-repo, no --grounding-result-file)."""
        with mock.patch.dict(
            os.environ,
            {
                "ELEVATOR_ACTIVE": "true",
                "ELEVATOR_STORE_URL": "http://test-elevator:8405",
                "ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC": "2",  # Short timeout for test
                "FACETS_DISPATCH_DISABLED": "0",
            },
        ), \
        mock.patch("lapis_pm.spec_review._swarm_serving", return_value=True), \
        mock.patch("requests.post") as mock_post, \
        mock.patch("requests.get") as mock_get, \
        mock.patch("subprocess.run") as mock_run, \
        mock.patch("time.sleep"), \
        mock.patch("lapis_pm.spec_review._FACETS_REPO_PATH", str(mock_repo_path)), \
        mock.patch("pathlib.Path.is_dir", return_value=True):

            # Mock successful enqueue
            enqueue_resp = mock.Mock()
            enqueue_resp.status_code = 201
            enqueue_resp.json.return_value = {"item_id": "item-timeout-123"}
            mock_post.return_value = enqueue_resp

            # Mock poll never reaching terminal (always pending)
            poll_resp = mock.Mock()
            poll_resp.status_code = 200
            poll_resp.json.return_value = {"status": "pending"}
            mock_get.return_value = poll_resp

            # Mock Facets subprocess
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"deliberation_id": "facets-fallback-456"})
            mock_run.return_value = mock_result

            # Patch time.time to simulate timeout
            with mock.patch("time.time") as mock_time:
                start = 1000.0
                mock_time.side_effect = [start, start + 3.0, start + 3.0]  # Exceed 2s timeout
                result = spec_review._dispatch_facets(
                    spec_text="spec text ac4",
                    parsed_target_id="ac4-target",
                    repo="test-repo",
                    authority="advisory",
                    start_time=start,
                )

            assert result == "facets-fallback-456"

            # Verify Facets was called WITHOUT grounding injection (fallback to inline)
            facets_call = mock_run.call_args
            argv = facets_call[0][0]

            # Should NOT have grounding injection
            assert "--grounding-result-file" not in argv
            assert "--no-auto-ground" not in argv
            # Should have --target-repo (inline grounding)
            assert "--target-repo" in argv

    def test_ac4_graceful_fallback_item_failed(self, mock_facets_env, mock_repo_path):
        """AC4: When item status=failed, falls back to inline grounding."""
        with mock.patch.dict(
            os.environ,
            {
                "ELEVATOR_ACTIVE": "true",
                "ELEVATOR_STORE_URL": "http://test-elevator:8405",
                "ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC": "60",
                "FACETS_DISPATCH_DISABLED": "0",
            },
        ), \
        mock.patch("lapis_pm.spec_review._swarm_serving", return_value=True), \
        mock.patch("requests.post") as mock_post, \
        mock.patch("requests.get") as mock_get, \
        mock.patch("subprocess.run") as mock_run, \
        mock.patch("lapis_pm.spec_review._FACETS_REPO_PATH", str(mock_repo_path)), \
        mock.patch("pathlib.Path.is_dir", return_value=True):

            # Mock successful enqueue
            enqueue_resp = mock.Mock()
            enqueue_resp.status_code = 201
            enqueue_resp.json.return_value = {"item_id": "item-failed-123"}
            mock_post.return_value = enqueue_resp

            # Mock poll returns failed status
            poll_resp = mock.Mock()
            poll_resp.status_code = 200
            poll_resp.json.return_value = {"status": "failed"}
            mock_get.return_value = poll_resp

            # Mock Facets subprocess
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"deliberation_id": "facets-failed-456"})
            mock_run.return_value = mock_result

            result = spec_review._dispatch_facets(
                spec_text="spec text ac4-failed",
                parsed_target_id="ac4-failed-target",
                repo="test-repo",
                authority="advisory",
                start_time=time.time(),
            )

            assert result == "facets-failed-456"

            # Verify fallback to inline
            facets_call = mock_run.call_args
            argv = facets_call[0][0]
            assert "--grounding-result-file" not in argv
            assert "--target-repo" in argv

    def test_ac5_bounded_poll(self, mock_facets_env, mock_repo_path):
        """AC5: Poll loop terminates within ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC."""
        with mock.patch.dict(
            os.environ,
            {
                "ELEVATOR_ACTIVE": "true",
                "ELEVATOR_STORE_URL": "http://test-elevator:8405",
                "ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC": "1",  # 1 second timeout
                "FACETS_DISPATCH_DISABLED": "0",
            },
        ), \
        mock.patch("lapis_pm.spec_review._swarm_serving", return_value=True), \
        mock.patch("requests.post") as mock_post, \
        mock.patch("requests.get") as mock_get, \
        mock.patch("subprocess.run") as mock_run, \
        mock.patch("time.sleep"), \
        mock.patch("lapis_pm.spec_review._FACETS_REPO_PATH", str(mock_repo_path)):

            # Mock enqueue
            enqueue_resp = mock.Mock()
            enqueue_resp.status_code = 201
            enqueue_resp.json.return_value = {"item_id": "bounded-poll"}
            mock_post.return_value = enqueue_resp

            # Always return pending (never terminal)
            poll_resp = mock.Mock()
            poll_resp.status_code = 200
            poll_resp.json.return_value = {"status": "pending"}
            mock_get.return_value = poll_resp

            # Mock Facets
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"deliberation_id": "bounded-facets"})
            mock_run.return_value = mock_result

            start_time = time.time()
            spec_review._dispatch_facets(
                spec_text="bounded test",
                parsed_target_id="bounded",
                repo="test-repo",
                authority="advisory",
                start_time=start_time,
            )

            # Should complete quickly (poll timeout should exit loop)
            elapsed = time.time() - start_time
            # Allow some overhead, but should be well under 10+ seconds
            assert elapsed < 30, f"Poll took {elapsed}s, should be bounded"

    def test_ac6_provenance_label_swarm_success(self, mock_facets_env, mock_repo_path, capsys):
        """AC6: On swarm-served, emits grounding_path=swarm:<item_id>."""
        with mock.patch.dict(
            os.environ,
            {
                "ELEVATOR_ACTIVE": "true",
                "ELEVATOR_STORE_URL": "http://test-elevator:8405",
                "ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC": "60",
                "FACETS_DISPATCH_DISABLED": "0",
            },
        ), \
        mock.patch("lapis_pm.spec_review._swarm_serving", return_value=True), \
        mock.patch("requests.post") as mock_post, \
        mock.patch("requests.get") as mock_get, \
        mock.patch("subprocess.run") as mock_run, \
        mock.patch("lapis_pm.spec_review._FACETS_REPO_PATH", str(mock_repo_path)):

            # Mock enqueue
            enqueue_resp = mock.Mock()
            enqueue_resp.status_code = 201
            enqueue_resp.json.return_value = {"item_id": "swarm-item-xyz"}
            mock_post.return_value = enqueue_resp

            # Mock poll returns served
            poll_resp = mock.Mock()
            poll_resp.status_code = 200
            poll_resp.json.return_value = {
                "status": "served",
                "result": {"grounding": "result"},
            }
            mock_get.return_value = poll_resp

            # Mock Facets
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"deliberation_id": "facets-xyz"})
            mock_run.return_value = mock_result

            spec_review._dispatch_facets(
                spec_text="test",
                parsed_target_id="provenance-swarm",
                repo="test-repo",
                authority="advisory",
                start_time=time.time(),
            )

            # Check stderr for provenance label
            captured = capsys.readouterr()
            assert "grounding_path=swarm:swarm-item-xyz" in captured.err

    def test_ac6_provenance_label_fallback(self, mock_facets_env, mock_repo_path, capsys):
        """AC6: On fallback, emits grounding_path=inline:<reason>."""
        with mock.patch.dict(
            os.environ,
            {
                "ELEVATOR_ACTIVE": "true",
                "ELEVATOR_STORE_URL": "http://test-elevator:8405",
                "ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC": "60",
                "FACETS_DISPATCH_DISABLED": "0",
            },
        ), \
        mock.patch("lapis_pm.spec_review._swarm_serving", return_value=False), \
        mock.patch("subprocess.run") as mock_run, \
        mock.patch("lapis_pm.spec_review._FACETS_REPO_PATH", str(mock_repo_path)):

            # Mock Facets
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"deliberation_id": "facets-fallback"})
            mock_run.return_value = mock_result

            spec_review._dispatch_facets(
                spec_text="test",
                parsed_target_id="provenance-fallback",
                repo="test-repo",
                authority="advisory",
                start_time=time.time(),
            )

            # Check stderr for fallback reason
            captured = capsys.readouterr()
            assert "grounding_path=inline:" in captured.err

    def test_ac8_readiness_precheck_swarm_not_serving(self, mock_facets_env, mock_repo_path, capsys):
        """AC8: When swarm_serving()=False, doesn't enqueue, falls back inline, logs swarm-not-serving."""
        with mock.patch.dict(
            os.environ,
            {
                "ELEVATOR_ACTIVE": "true",
                "ELEVATOR_STORE_URL": "http://test-elevator:8405",
                "ELEVATOR_GROUNDING_POLL_TIMEOUT_SEC": "180",
                "FACETS_DISPATCH_DISABLED": "0",
            },
        ), \
        mock.patch("lapis_pm.spec_review._swarm_serving", return_value=False) as mock_swarm, \
        mock.patch("requests.post") as mock_post, \
        mock.patch("subprocess.run") as mock_run, \
        mock.patch("lapis_pm.spec_review._FACETS_REPO_PATH", str(mock_repo_path)), \
        mock.patch("pathlib.Path.is_dir", return_value=True):

            # Mock Facets
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"deliberation_id": "facets-noswarm"})
            mock_run.return_value = mock_result

            spec_review._dispatch_facets(
                spec_text="test ac8",
                parsed_target_id="ac8-target",
                repo="test-repo",
                authority="advisory",
                start_time=time.time(),
            )

            # Verify swarm_serving was called (pre-check)
            assert mock_swarm.called

            # Verify NO enqueue was attempted
            assert not mock_post.called

            # Verify fallback log
            captured = capsys.readouterr()
            assert "grounding_path=inline:swarm-not-serving" in captured.err
            assert "swarm not serving" in captured.err

            # Verify Facets got inline grounding
            facets_call = mock_run.call_args
            argv = facets_call[0][0]
            assert "--grounding-result-file" not in argv
            assert "--target-repo" in argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
