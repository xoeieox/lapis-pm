"""Tests for lapis_pm.node_probe (lapis-pm-gpu-probe-tick-hang-v0).

Coverage:
1. Returns True when httpx.get returns a 200 response.
2. Returns True when httpx.get returns a 404 response (any status = reachable).
3. Returns False on httpx.ConnectError.
4. Returns False on httpx.TimeoutException.
5. TTL memoization: second call within TTL does not re-invoke the transport.
6. Corroboration: with :8081 blocked by autouse guard, score() returns uncertain
   without calling httpx.post.
7. Witness: with :8081 blocked by autouse guard, run_local_reviewer_witness()
   returns local_failed without calling httpx.post.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_200() -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    return r


def _mock_404() -> MagicMock:
    r = MagicMock()
    r.status_code = 404
    return r


_TEST_URL = "http://192.0.2.55:9877/v1/chat/completions"  # RFC5737, not in blocked list


# ---------------------------------------------------------------------------
# Test 1: True on 200
# ---------------------------------------------------------------------------

class TestNodeReachableTrue:

    def test_returns_true_on_200(self):
        from lapis_pm.node_probe import node_reachable
        with patch("httpx.get", return_value=_mock_200()) as mock_get:
            result = node_reachable(_TEST_URL)
        assert result is True
        assert mock_get.call_count == 1
        # Probe URL is origin + /health
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://192.0.2.55:9877/health"

    def test_returns_true_on_404(self):
        """Any HTTP response (including 404) counts as reachable."""
        from lapis_pm.node_probe import node_reachable
        with patch("httpx.get", return_value=_mock_404()):
            result = node_reachable(_TEST_URL)
        assert result is True


# ---------------------------------------------------------------------------
# Test 2: False on connection-level failures
# ---------------------------------------------------------------------------

class TestNodeReachableFalse:

    def test_returns_false_on_connect_error(self):
        import httpx
        from lapis_pm.node_probe import node_reachable
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = node_reachable(_TEST_URL)
        assert result is False

    def test_returns_false_on_timeout_exception(self):
        import httpx
        from lapis_pm.node_probe import node_reachable
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            result = node_reachable(_TEST_URL)
        assert result is False

    def test_returns_false_on_connect_timeout(self):
        import httpx
        from lapis_pm.node_probe import node_reachable
        with patch("httpx.get", side_effect=httpx.ConnectTimeout("connect timed out")):
            result = node_reachable(_TEST_URL)
        assert result is False

    def test_never_raises_on_unexpected_exception(self):
        """Unexpected exceptions must not propagate — degrade to False."""
        from lapis_pm.node_probe import node_reachable
        with patch("httpx.get", side_effect=RuntimeError("unexpected")):
            result = node_reachable(_TEST_URL)
        assert result is False


# ---------------------------------------------------------------------------
# Test 3: TTL memoization
# ---------------------------------------------------------------------------

class TestTTLMemoization:

    def test_second_call_within_ttl_uses_cache(self):
        """Second call within TTL must not re-invoke the transport."""
        from lapis_pm.node_probe import node_reachable

        with patch("httpx.get", return_value=_mock_200()) as mock_get:
            r1 = node_reachable(_TEST_URL)
            r2 = node_reachable(_TEST_URL)

        assert r1 is True
        assert r2 is True
        assert mock_get.call_count == 1, (
            f"Expected 1 httpx.get call (TTL cache should suppress second), got {mock_get.call_count}"
        )

    def test_different_ports_probed_independently(self):
        """Calls to different ports are cached independently."""
        from lapis_pm.node_probe import node_reachable

        url_a = "http://192.0.2.55:9877/v1"
        url_b = "http://192.0.2.55:9878/v1"  # different port

        with patch("httpx.get", return_value=_mock_200()) as mock_get:
            node_reachable(url_a)
            node_reachable(url_b)

        assert mock_get.call_count == 2, (
            f"Different ports must each be probed; expected 2 calls, got {mock_get.call_count}"
        )

    def test_cache_keyed_by_host_and_port(self):
        """Same host+port from different URL paths shares the cache."""
        from lapis_pm.node_probe import node_reachable

        url1 = "http://192.0.2.55:9877/path/one"
        url2 = "http://192.0.2.55:9877/path/two"

        with patch("httpx.get", return_value=_mock_200()) as mock_get:
            node_reachable(url1)
            node_reachable(url2)

        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# Test 4: Corroboration gate — :8081 blocked → score() skips POST
# ---------------------------------------------------------------------------

class TestCorroborationProbeGate:
    """With :8081 blocked by the autouse _block_sleeping_node_http fixture,
    node_reachable returns False → score() must return uncertain without
    calling httpx.post."""

    def _make_substrates(self):
        from lapis_pm.corroboration_adapter import _IdentifierSubstrate
        return [_IdentifierSubstrate(
            identifier="tick",
            repo_hits=[{"file": "f.py", "line": "1", "text": "def tick"}],
            vault_hits=[],
        )]

    def test_score_returns_uncertain_when_node_unreachable(self):
        """score() short-circuits to uncertain when probe fails, no POST attempted."""
        from lapis_pm.corroboration_adapter import LapisPMReviewerAdapter

        adapter = LapisPMReviewerAdapter()
        substrates = self._make_substrates()

        # httpx.post patched to raise loudly if called — probe must short-circuit
        with patch("httpx.post", side_effect=AssertionError("POST must not be called when node unreachable")):
            result = adapter.score("some diff", substrates, "lapis-pm")

        assert result.verdict == "uncertain"
        assert result.notes is not None
        assert "unreachable" in result.notes

    def test_score_post_not_called_when_node_unreachable(self):
        """Assert httpx.post call count is 0 when probe fails."""
        from lapis_pm.corroboration_adapter import LapisPMReviewerAdapter

        adapter = LapisPMReviewerAdapter()
        substrates = self._make_substrates()

        with patch("httpx.post") as mock_post:
            adapter.score("some diff", substrates, "lapis-pm")

        assert mock_post.call_count == 0, (
            f"httpx.post must not be called when node is unreachable (probe blocked by conftest); "
            f"got {mock_post.call_count} calls"
        )


# ---------------------------------------------------------------------------
# Test 5: Witness gate — :8081 blocked → run_local_reviewer_witness() skips POST
# ---------------------------------------------------------------------------

class TestWitnessProbeGate:
    """With :8081 blocked by the autouse _block_sleeping_node_http fixture,
    node_reachable returns False → run_local_reviewer_witness() must return
    local_failed without calling httpx.post."""

    _FAKE_DIFF = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n+x = 1\n"
    _CLAUDE_CLEAN = {"verdict": "clean", "issues": [], "confidence": 0.95}

    def test_returns_local_failed_when_endpoint_unreachable(self):
        """run_local_reviewer_witness() returns local_failed fast when probe fails."""
        from lapis_pm.local_reviewer_witness import run_local_reviewer_witness

        with patch("httpx.post", side_effect=AssertionError("POST must not be called when probe fails")):
            result = run_local_reviewer_witness(
                diff_text=self._FAKE_DIFF,
                repo="lapis-pm",
                pr_number=7,
                spec_summary="spec",
                claude_verdict=self._CLAUDE_CLEAN,
            )

        assert result.agreement == "local_failed"
        assert result.error is not None
        assert "unreachable" in result.error

    def test_post_not_called_when_endpoint_unreachable(self):
        """Assert httpx.post call count is 0 when probe fails."""
        from lapis_pm.local_reviewer_witness import run_local_reviewer_witness

        with patch("httpx.post") as mock_post:
            result = run_local_reviewer_witness(
                diff_text=self._FAKE_DIFF,
                repo="lapis-pm",
                pr_number=7,
                spec_summary="spec",
                claude_verdict=self._CLAUDE_CLEAN,
            )

        assert mock_post.call_count == 0, (
            f"httpx.post must not be called when node is unreachable; "
            f"got {mock_post.call_count} calls"
        )
        assert result.agreement == "local_failed"
