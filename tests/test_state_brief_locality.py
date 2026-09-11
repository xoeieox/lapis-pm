"""Tests for lapis-pm-locality-brief-block-v0 (Leg 2 of locality-ledger-chain-v0).

Coverage:
  1. TestReadLocalityRendering: weekly render of a healthy/seeded ledger, no
     ledger (ImportError / never-written), stale ledger, daily/monthly no-op,
     and the "0% local must render loudly" case.
  2. TestFormatBucketSections: Locality bucket special-casing in
     format_bucket_sections() (weekly-only, omit-when-empty), mirroring the
     existing TestClimateBucketRendering coverage.
  3. TestLocalityTransition: the ratified OK->BAD->escalate->OK state
     machine, proven against the exact table in the spec.
  4. TestEvaluateLocalityGem: end-to-end firing behaviour via
     _evaluate_locality_gem — deposit payload shape, threshold values in
     context, supersession of a prior open gem, and a Weaver outage during
     deposit logging a WARNING without raising.
  5. TestGemTextPlainTone: no accusatory/escalating phrasing in gem text.
  6. TestReadBucketsIntegration: _read_buckets wiring — a Locality reader
     exception never breaks the overall bucket read.

Patch targets:
  - lapis_pm.state_brief._mem                      -> MemoryStore factory
  - agents_core.locality.summarize / is_ledger_healthy (lazy-imported)
  - httpx.Client                                    -> weaver POSTs
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import state_brief, state_brief_prompts


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _summary(
    total=100,
    pct_local=80.0,
    by_cost_class=None,
    fallback_count=5,
    by_fallback_reason=None,
    paid_cost_usd=None,
):
    return {
        "total": total,
        "pct_local": pct_local,
        "by_cost_class": by_cost_class or {"local-gw": 80, "paid-anthropic": 20},
        "fallback_count": fallback_count,
        "by_fallback_reason": by_fallback_reason or {"gw_not_serving": 3, "fallback": 2},
        "paid_cost_usd": paid_cost_usd,
        "by_requested_operator": {},
        "window": {"since": None, "until": None},
    }


def _make_mock_mem(state: dict | None = None):
    """mem mock whose .get(pm/locality/gem-state) returns a seeded state and
    whose .set() calls are captured for inspection."""
    mem = MagicMock()
    set_calls: list[tuple] = []

    def _get(key):
        if key == state_brief._LOCALITY_STATE_KEY and state is not None:
            return {"content": json.dumps(state)}
        return None

    def _set(key, value, **kwargs):
        set_calls.append((key, value))

    mem.get.side_effect = _get
    mem.set.side_effect = _set
    mem._set_calls = set_calls
    return mem


def _saved_state(mem) -> dict:
    rec = next(v for k, v in mem._set_calls if k == state_brief._LOCALITY_STATE_KEY)
    return json.loads(rec)


# ---------------------------------------------------------------------------
# 1. _read_locality rendering
# ---------------------------------------------------------------------------

class TestReadLocalityRendering:
    def test_daily_period_returns_empty_without_touching_ledger(self):
        with patch("agents_core.locality.summarize") as mock_summarize:
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="daily")
        assert result == []
        mock_summarize.assert_not_called()

    def test_monthly_period_returns_empty(self):
        result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="monthly")
        assert result == []

    def test_import_error_returns_empty_and_brief_generation_unaffected(self):
        """agents_core.locality unimportable -> no section, no raise."""
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "agents_core.locality" or name.startswith("agents_core.locality"):
                raise ImportError("agents_core.locality not available")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="weekly")
        assert result == []

    def test_never_written_ledger_returns_empty_no_section(self):
        with (
            patch("agents_core.locality.is_ledger_healthy",
                  return_value=(False, "no ledger directory found — no records ever written")),
            patch("agents_core.locality.summarize") as mock_summarize,
        ):
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="weekly")
        assert result == []
        mock_summarize.assert_not_called()

    def test_stale_ledger_renders_warning_no_percentage(self):
        with patch(
            "agents_core.locality.is_ledger_healthy",
            return_value=(False, "no ledger record in 48.0h (threshold 24.0h) — last record at ..."),
        ):
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="weekly")
        assert len(result) == 1
        assert "silent" in result[0].lower()
        assert "%" not in result[0]  # no percentage rendered, even though the
        # reason text itself may carry a duration in hours

    def test_healthy_ledger_renders_percentage_and_breakdown(self):
        mock_mem = _make_mock_mem(state=None)
        with (
            patch("agents_core.locality.is_ledger_healthy", return_value=(True, "last record 0.1h ago")),
            patch("agents_core.locality.summarize", return_value=_summary(pct_local=80.0)),
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            patch.dict(os.environ, {}, clear=False),
        ):
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="weekly")
        assert any("80 percent" in line for line in result)
        assert any("local-gw" in line for line in result)
        assert any("fell back" in line for line in result)
        assert any("bypass this ledger" in line for line in result)

    def test_zero_percent_local_renders_loudly_not_omitted(self):
        """A healthy ledger with real data reporting 0% local must still render."""
        mock_mem = _make_mock_mem(state=None)
        with (
            patch("agents_core.locality.is_ledger_healthy", return_value=(True, "last record 0.1h ago")),
            patch("agents_core.locality.summarize",
                  return_value=_summary(total=50, pct_local=0.0, by_cost_class={"paid-anthropic": 50})),
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            ):
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="weekly")
        assert result != []
        assert any("0 percent" in line for line in result)

    def test_healthy_but_zero_calls_recorded_renders_explicit_line(self):
        mock_mem = _make_mock_mem(state=None)
        with (
            patch("agents_core.locality.is_ledger_healthy", return_value=(True, "last record 0.1h ago")),
            patch("agents_core.locality.summarize", return_value=_summary(total=0, pct_local=0.0)),
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            ):
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="weekly")
        assert len(result) == 1
        assert "no calls" in result[0].lower()

    def test_summarize_exception_degrades_to_empty(self):
        with (
            patch("agents_core.locality.is_ledger_healthy", return_value=(True, "ok")),
            patch("agents_core.locality.summarize", side_effect=RuntimeError("boom")),
        ):
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="weekly")
        assert result == []

    def test_is_ledger_healthy_exception_degrades_to_empty(self):
        with patch("agents_core.locality.is_ledger_healthy", side_effect=RuntimeError("boom")):
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="weekly")
        assert result == []


# ---------------------------------------------------------------------------
# 2. format_bucket_sections — Locality special-casing (mirrors Climate)
# ---------------------------------------------------------------------------

class TestFormatBucketSectionsLocality:
    def _buckets(self, locality_items=None):
        return {
            "Built": [],
            "Notable ratifications": [],
            "In flight": [],
            "Captured — not yet built": [],
            "Awaiting your call": [],
            "Gardener Cross-Cutting Observations": [],
            "Climate": [],
            "Locality": locality_items or [],
        }

    def test_daily_omits_locality_entirely(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["80 percent of calls this week were served locally."]),
            "2026-07-06 08:00 PT", period="daily",
        )
        assert "Locality" not in sections

    def test_weekly_renders_locality_when_present(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["80 percent of calls this week were served locally."]),
            "2026-07-06 08:00 PT", period="weekly",
        )
        assert "## Locality" in sections
        assert "80 percent of calls this week were served locally." in sections

    def test_weekly_omits_locality_when_empty(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets([]), "2026-07-06 08:00 PT", period="weekly",
        )
        assert "Locality" not in sections

    def test_locality_appears_after_climate(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["80 percent of calls this week were served locally."]),
            "2026-07-06 08:00 PT", period="weekly",
        )
        # Climate section absent (empty) in this fixture, but Locality must still
        # follow Awaiting your call in ordering.
        assert sections.index("## Locality") > sections.index("Awaiting your call")

    def test_bucket_order_has_locality_after_climate(self):
        assert state_brief_prompts.BUCKET_ORDER.index("Locality") == \
            state_brief_prompts.BUCKET_ORDER.index("Climate") + 1


# ---------------------------------------------------------------------------
# 3. The ratified firing-rule state machine, proven against the spec table
# ---------------------------------------------------------------------------

class TestLocalityTransition:
    """week | reading | action
        1   |  OK     | silent
        2   |  BAD    | GEM — crossing
        3   |  BAD    | silent
        4   |  BAD    | silent
        5   |  BAD    | GEM — escalation (N=4)
        6   |  BAD    | silent
        7   |  OK     | silent, re-armed
        8   |  BAD    | GEM — new crossing
    """

    def test_full_ratified_table(self):
        state = state_brief._default_locality_state()
        expected = [
            (False, None),   # week 1: OK -> silent
            (True, "crossing"),   # week 2: OK->BAD -> GEM
            (True, None),     # week 3: BAD -> silent
            (True, None),     # week 4: BAD -> silent
            (True, "escalation"),  # week 5: 4th consecutive BAD -> GEM once
            (True, None),     # week 6: BAD -> silent
            (False, None),    # week 7: BAD->OK -> silent, re-armed
            (True, "crossing"),   # week 8: OK->BAD -> GEM (new crossing)
        ]
        for week, (is_bad, expected_fire) in enumerate(expected, start=1):
            state, fire = state_brief._locality_transition(state, is_bad, escalation_weeks=4)
            assert fire == expected_fire, f"week {week}: expected fire={expected_fire!r}, got {fire!r}"

    def test_ok_to_ok_never_fires(self):
        state = state_brief._default_locality_state()
        for _ in range(5):
            state, fire = state_brief._locality_transition(state, False, escalation_weeks=4)
            assert fire is None

    def test_escalation_does_not_refire_weekly_after_first(self):
        """BAD persists for 10 straight weeks with N=4: only 2 gems total
        (crossing at week 1, escalation at week 4), never a 3rd."""
        state = state_brief._default_locality_state()
        fires = []
        for _ in range(10):
            state, fire = state_brief._locality_transition(state, True, escalation_weeks=4)
            fires.append(fire)
        assert fires.count("crossing") == 1
        assert fires.count("escalation") == 1
        assert fires[0] == "crossing"
        assert fires[3] == "escalation"
        assert all(f is None for i, f in enumerate(fires) if i not in (0, 3))

    def test_escalation_weeks_env_overridable_via_thresholds(self, monkeypatch):
        monkeypatch.setenv("LOCALITY_ESCALATION_WEEKS", "2")
        thresholds = state_brief._locality_thresholds()
        assert thresholds["escalation_weeks"] == 2

    def test_default_escalation_weeks_is_four(self):
        thresholds = state_brief._locality_thresholds()
        assert thresholds["escalation_weeks"] == 4


class TestLocalityIsBad:
    def test_below_floor_is_bad(self):
        thresholds = {"pct_local_floor": 50.0, "fallback_ceiling": 100}
        assert state_brief._locality_is_bad(_summary(pct_local=10.0, fallback_count=0), thresholds)

    def test_above_floor_and_under_ceiling_is_ok(self):
        thresholds = {"pct_local_floor": 50.0, "fallback_ceiling": 100}
        assert not state_brief._locality_is_bad(_summary(pct_local=90.0, fallback_count=5), thresholds)

    def test_fallback_count_over_ceiling_is_bad(self):
        thresholds = {"pct_local_floor": 50.0, "fallback_ceiling": 10}
        assert state_brief._locality_is_bad(_summary(pct_local=90.0, fallback_count=11), thresholds)


# ---------------------------------------------------------------------------
# 4. _evaluate_locality_gem — deposit shape, thresholds in context,
#    supersession, Weaver-outage fail-soft
# ---------------------------------------------------------------------------

class TestEvaluateLocalityGem:
    def test_crossing_deposits_gem_with_thresholds_in_context(self):
        mock_mem = _make_mock_mem(state=None)  # fresh -> ok
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"gem_id": "gem-001"}
        fake_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        deposited = []
        mock_client.post.side_effect = lambda url, json=None, **kw: deposited.append((url, json)) or fake_resp

        with (
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            patch("httpx.Client", return_value=mock_client),
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            state_brief._evaluate_locality_gem(_summary(pct_local=10.0, fallback_count=0))

        assert len(deposited) == 1
        url, payload = deposited[0]
        assert url.endswith("/v0/decision-gems")
        assert payload["title"].strip()
        assert payload["ask"].strip()
        assert payload["why"].strip()
        labels = [c["label"] for c in payload["context"]]
        assert "Active thresholds" in labels
        thresholds_block = next(c for c in payload["context"] if c["label"] == "Active thresholds")
        assert any("Floor" in line for line in thresholds_block["lines"])
        assert any("ceiling" in line.lower() for line in thresholds_block["lines"])
        assert any("Escalation" in line for line in thresholds_block["lines"])
        assert payload["agent"] == "locality-ledger"

        saved = _saved_state(mock_mem)
        assert saved["state"] == "bad"
        assert saved["consecutive_bad"] == 1
        assert saved["open_gem_id"] == "gem-001"

    def test_bad_to_bad_does_not_deposit(self):
        prior = {
            "state": "bad", "consecutive_bad": 1, "escalated": False,
            "open_gem_id": "gem-001", "last_pct_local": 10.0, "last_fallback_count": 0,
        }
        mock_mem = _make_mock_mem(state=prior)
        with (
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            patch("httpx.Client") as mock_httpx,
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            state_brief._evaluate_locality_gem(_summary(pct_local=10.0, fallback_count=0))
        mock_httpx.assert_not_called()
        saved = _saved_state(mock_mem)
        assert saved["consecutive_bad"] == 2
        assert saved["open_gem_id"] == "gem-001"  # carried over, untouched

    def test_new_crossing_supersedes_prior_open_gem(self):
        prior = {
            "state": "ok", "consecutive_bad": 0, "escalated": False,
            "open_gem_id": "gem-old", "last_pct_local": 90.0, "last_fallback_count": 1,
        }
        mock_mem = _make_mock_mem(state=prior)
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"gem_id": "gem-new"}
        fake_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        posted_urls = []
        mock_client.post.side_effect = lambda url, json=None, **kw: posted_urls.append(url) or fake_resp

        with (
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            patch("httpx.Client", return_value=mock_client),
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            state_brief._evaluate_locality_gem(_summary(pct_local=10.0, fallback_count=0))

        assert any("gem-old/supersede" in u for u in posted_urls)
        assert any(u.endswith("/v0/decision-gems") for u in posted_urls)
        saved = _saved_state(mock_mem)
        assert saved["open_gem_id"] == "gem-new"

    def test_recovery_to_ok_re_arms_and_does_not_deposit(self):
        prior = {
            "state": "bad", "consecutive_bad": 5, "escalated": True,
            "open_gem_id": "gem-001", "last_pct_local": 10.0, "last_fallback_count": 0,
        }
        mock_mem = _make_mock_mem(state=prior)
        with (
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            patch("httpx.Client") as mock_httpx,
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            state_brief._evaluate_locality_gem(_summary(pct_local=90.0, fallback_count=1))
        mock_httpx.assert_not_called()
        saved = _saved_state(mock_mem)
        assert saved["state"] == "ok"
        assert saved["consecutive_bad"] == 0
        assert saved["escalated"] is False

    def test_weaver_outage_during_deposit_logs_warning_and_does_not_raise(self, caplog):
        import httpx as _httpx
        mock_mem = _make_mock_mem(state=None)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = _httpx.ConnectError("weaver down")

        with (
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            patch("httpx.Client", return_value=mock_client),
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
            caplog.at_level("WARNING"),
        ):
            # Must not raise.
            state_brief._evaluate_locality_gem(_summary(pct_local=10.0, fallback_count=0))

        assert any("deposit failed" in rec.message for rec in caplog.records)
        # State is still saved even though the deposit failed.
        saved = _saved_state(mock_mem)
        assert saved["state"] == "bad"

    def test_weaver_outage_during_read_locality_does_not_break_brief(self, caplog):
        """Full path: _read_locality still returns its rendered lines even
        when the gem deposit fails underneath it."""
        import httpx as _httpx
        mock_mem = _make_mock_mem(state=None)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = _httpx.ConnectError("weaver down")

        with (
            patch("agents_core.locality.is_ledger_healthy", return_value=(True, "ok")),
            patch("agents_core.locality.summarize",
                  return_value=_summary(pct_local=10.0, fallback_count=0)),
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            patch("httpx.Client", return_value=mock_client),
            patch.dict(os.environ, {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
            caplog.at_level("WARNING"),
        ):
            result = state_brief._read_locality(datetime.now(tz=timezone.utc), period="weekly")

        assert result != []  # rendered lines still returned
        assert any("deposit failed" in rec.message for rec in caplog.records)

    def test_pytest_guard_skips_live_post_without_weaver_base_url(self):
        """Under pytest with no WEAVER_BASE_URL set, no live prod POST is attempted."""
        mock_mem = _make_mock_mem(state=None)
        env = {k: v for k, v in os.environ.items() if k != "WEAVER_BASE_URL"}
        with (
            patch("lapis_pm.state_brief._mem", return_value=mock_mem),
            patch("httpx.Client") as mock_httpx,
            patch.dict(os.environ, env, clear=True),
        ):
            state_brief._evaluate_locality_gem(_summary(pct_local=10.0, fallback_count=0))
        mock_httpx.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Gem text tone — ratified plain, not escalating/accusatory
# ---------------------------------------------------------------------------

class TestGemTextPlainTone:
    _BANNED_SUBSTRINGS = (
        "cloud-dependent", "cloud dependent", "burden", "alarm", "unacceptable",
        "failed", "failing", "blame", "shame", "should have", "you broke",
        "critical failure", "!",
    )

    def _assert_plain(self, payload):
        haystack = f"{payload['title']} {payload['ask']} {payload['why']}".lower()
        for banned in self._BANNED_SUBSTRINGS:
            assert banned not in haystack, f"banned phrase {banned!r} found in gem text: {haystack}"

    def test_crossing_gem_text_is_plain(self):
        thresholds = state_brief._locality_thresholds()
        prior = state_brief._default_locality_state()
        payload = state_brief._locality_gem_text(
            "crossing", _summary(pct_local=10.0, fallback_count=20), thresholds, prior,
        )
        self._assert_plain(payload)
        assert "10.0" in payload["why"]
        assert "?" in payload["ask"]

    def test_escalation_gem_text_is_plain(self):
        thresholds = state_brief._locality_thresholds()
        prior = {
            "state": "bad", "consecutive_bad": 3, "escalated": False,
            "open_gem_id": "gem-001", "last_pct_local": 12.0, "last_fallback_count": 18,
        }
        payload = state_brief._locality_gem_text(
            "escalation", _summary(pct_local=9.0, fallback_count=22), thresholds, prior,
        )
        self._assert_plain(payload)
        assert "4 weeks" in payload["title"]

    def test_gem_text_includes_delta_from_prior_reading(self):
        thresholds = state_brief._locality_thresholds()
        prior = {
            "state": "bad", "consecutive_bad": 1, "escalated": False,
            "open_gem_id": None, "last_pct_local": 40.0, "last_fallback_count": 5,
        }
        payload = state_brief._locality_gem_text(
            "crossing", _summary(pct_local=30.0, fallback_count=5), thresholds, prior,
        )
        assert "40.0" in payload["why"]

    def test_gem_text_no_prior_reading_says_so(self):
        thresholds = state_brief._locality_thresholds()
        prior = state_brief._default_locality_state()
        payload = state_brief._locality_gem_text(
            "crossing", _summary(pct_local=30.0, fallback_count=5), thresholds, prior,
        )
        assert "no prior reading" in payload["why"].lower()


# ---------------------------------------------------------------------------
# 6. _read_buckets integration — Locality reader failure never breaks the brief
# ---------------------------------------------------------------------------

class TestReadBucketsIntegration:
    def test_locality_reader_exception_does_not_break_bucket_read(self):
        with (
            patch("lapis_pm.state_brief._mem") as mock_mem,
            patch("lapis_pm.state_brief._read_arc_climate", return_value=[]),
            patch("lapis_pm.state_brief._read_locality", side_effect=RuntimeError("boom")),
        ):
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.return_value = []
            buckets = state_brief._read_buckets(datetime.now(tz=timezone.utc), period="weekly")
        assert buckets[state_brief.B_LOCALITY] == []

    def test_locality_key_present_in_bucket_dict(self):
        with (
            patch("lapis_pm.state_brief._mem") as mock_mem,
            patch("lapis_pm.state_brief._read_arc_climate", return_value=[]),
            patch("lapis_pm.state_brief._read_locality", return_value=["a line"]),
        ):
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.return_value = []
            buckets = state_brief._read_buckets(datetime.now(tz=timezone.utc), period="weekly")
        assert buckets[state_brief.B_LOCALITY] == ["a line"]
