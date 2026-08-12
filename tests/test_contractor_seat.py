"""Tests for lapis_pm.contractor_seat — the D7 out-of-band availability
canary for the Phala contractor seat.

Covers the two verified traps the design explicitly must not fall into:
  - `GET /v1/models` is a static catalog and must never be used as a health
    check (this module never calls it).
  - `x-ratelimit-limit: 0` on Phala is transient throttling, not "no
    allowance" — a 429 must be classified transient/retryable, never
    terminal.

Also covers: RemoteDisconnected mid-call, empty content, proxy-down
(connection refused), and that a 429 does NOT terminate on the first
attempt (bounded retry, honouring retry-after).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lapis_pm import contractor_seat as cs


class FakeResponse:
    def __init__(self, status_code, json_body=None, headers=None):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.headers = headers or {}

    def json(self):
        return self._json_body


class TestClassifyResponse:

    def test_200_with_content_is_healthy(self):
        state, _ = cs.classify_response(200, {"choices": [{"message": {"content": "ok"}}]}, None)
        assert state == cs.STATE_HEALTHY

    def test_200_with_empty_content_is_named_empty_content(self):
        state, _ = cs.classify_response(200, {"choices": [{"message": {"content": ""}}]}, None)
        assert state == cs.STATE_EMPTY_CONTENT

    def test_200_with_reasoning_only_is_healthy_not_empty(self):
        """lapis-pm-corroboration-thinking-parse-and-truncation-loudness-v0
        Gate outcome §4: this site is a health classifier, not a parse site —
        a thinking seat that reasons but never emits `content` was being
        misclassified as empty (same false-dead-seat class the corroboration
        leg had). content=None + reasoning populated must now read healthy."""
        state, detail = cs.classify_response(
            200,
            {"choices": [{"message": {"content": None, "reasoning": "pong (reasoned)"}}]},
            None,
        )
        assert state == cs.STATE_HEALTHY
        assert detail == "200, non-empty content"

    def test_200_with_reasoning_content_only_is_healthy(self):
        """llama.cpp's reasoning_content convention is honoured too."""
        state, _ = cs.classify_response(
            200,
            {"choices": [{"message": {"content": "", "reasoning_content": "pong"}}]},
            None,
        )
        assert state == cs.STATE_HEALTHY

    def test_200_with_none_of_the_three_fields_is_still_empty_content(self):
        """The fix widens the fallback; it does not stop detecting a genuinely
        empty response."""
        state, _ = cs.classify_response(
            200,
            {"choices": [{"message": {"content": None, "reasoning": None}}]},
            None,
        )
        assert state == cs.STATE_EMPTY_CONTENT

    def test_429_is_transient_throttled_never_terminal(self):
        """The core correction: x-ratelimit-limit: 0 does NOT mean dead
        account. A 429 must classify as upstream_throttled (retryable),
        never as a terminal/dead state."""
        state, detail = cs.classify_response(429, {}, None)
        assert state == cs.STATE_UPSTREAM_THROTTLED
        assert "transient" in detail.lower()

    def test_5xx_is_upstream_5xx(self):
        state, _ = cs.classify_response(503, {}, None)
        assert state == cs.STATE_UPSTREAM_5XX

    def test_connection_refused_is_proxy_down(self):
        exc = ConnectionError("Connection refused")
        state, _ = cs.classify_response(None, None, exc)
        assert state == cs.STATE_PROXY_DOWN

    def test_remote_disconnected_is_upstream_disconnected(self):
        class RemoteDisconnected(Exception):
            pass
        exc = RemoteDisconnected("Remote end closed connection without response")
        state, _ = cs.classify_response(None, None, exc)
        assert state == cs.STATE_UPSTREAM_DISCONNECTED


class TestProbeSeatBoundedRetry:

    def test_429_does_not_terminate_on_first_refusal(self):
        """DoD 7/re-gate: the 429 test must assert the route does not park
        on the first refusal — bounded retry, not immediate terminal."""
        call_log = []

        def fake_probe_once(*a, **kw):
            call_log.append(1)
            if len(call_log) < 3:
                return cs.STATE_UPSTREAM_THROTTLED, "429", {"retry-after": "0"}
            return cs.STATE_HEALTHY, "200", {}

        with patch.object(cs, "probe_seat_once", side_effect=fake_probe_once):
            health = cs.probe_seat(max_retries=5, sleep_fn=lambda s: None)

        assert len(call_log) == 3, "must have retried past the first 429"
        assert health.state == cs.STATE_HEALTHY
        assert health.retries_used == 2

    def test_proxy_down_does_not_retry(self):
        """Terminal states are returned immediately — retrying a dead proxy
        wastes the same way retrying a genuinely-exhausted account would."""
        call_log = []

        def fake_probe_once(*a, **kw):
            call_log.append(1)
            return cs.STATE_PROXY_DOWN, "connection refused", None

        with patch.object(cs, "probe_seat_once", side_effect=fake_probe_once):
            health = cs.probe_seat(max_retries=5, sleep_fn=lambda s: None)

        assert len(call_log) == 1, "proxy_down must not be retried"
        assert health.state == cs.STATE_PROXY_DOWN

    def test_empty_content_does_not_retry(self):
        call_log = []

        def fake_probe_once(*a, **kw):
            call_log.append(1)
            return cs.STATE_EMPTY_CONTENT, "200 empty", {}

        with patch.object(cs, "probe_seat_once", side_effect=fake_probe_once):
            health = cs.probe_seat(max_retries=5, sleep_fn=lambda s: None)

        assert len(call_log) == 1
        assert health.state == cs.STATE_EMPTY_CONTENT

    def test_exhausts_retries_and_reports_last_transient_state(self):
        def fake_probe_once(*a, **kw):
            return cs.STATE_UPSTREAM_THROTTLED, "429", {"retry-after": "0"}

        with patch.object(cs, "probe_seat_once", side_effect=fake_probe_once):
            health = cs.probe_seat(max_retries=2, sleep_fn=lambda s: None)

        assert health.state == cs.STATE_UPSTREAM_THROTTLED
        assert health.retries_used == 2


class TestNotAHealthCheckTraps:

    def test_probe_never_calls_v1_models(self):
        """GET /v1/models is a static catalog and answers 200 regardless of
        upstream state — it must never be treated as a health check."""
        with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
            mock_post.return_value = FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})
            cs.probe_seat_once()
            mock_get.assert_not_called()
            assert mock_post.called
            called_url = mock_post.call_args[0][0]
            assert "/v1/models" not in called_url
            assert "/v1/chat/completions" in called_url


class TestCachedHealth:

    def test_cached_health_returns_none_when_absent(self):
        mem_store = MagicMock()
        mem_store.get.return_value = None
        assert cs.cached_health(mem_store) is None

    def test_refresh_writes_to_mem(self):
        mem_store = MagicMock()
        with patch.object(cs, "probe_seat", return_value=cs.SeatHealth(
            state=cs.STATE_HEALTHY, checked_at="2026-08-04T00:00:00+00:00",
        )):
            health = cs.refresh_cached_health(mem_store)
        assert health.state == cs.STATE_HEALTHY
        assert mem_store.set.called
        key = mem_store.set.call_args[0][0]
        assert key == cs.CACHE_MEM_KEY

    def test_cached_health_roundtrips_dispatch_reads_a_dict(self):
        mem_store = MagicMock()
        stored = {"state": cs.STATE_HEALTHY, "checked_at": "2026-08-04T00:00:00+00:00",
                  "detail": "ok", "retries_used": 0}
        mem_store.get.return_value = {"content": str(stored)}
        result = cs.cached_health(mem_store)
        assert result["state"] == cs.STATE_HEALTHY
