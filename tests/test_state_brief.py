"""Tests for state brief degradation on StarHouse timeout.

Tests verify:
1. Brief completes within wall-clock budget when qwen/StarHouse is unreachable
2. Degraded brief contains all five deterministic local sections
3. Degraded brief has visible (DEGRADED) marker
4. All three periods (morning, afternoon, live) are covered by the timeout guard
5. DASHBOARD_BASE points to BRIX (not stale StarHouse)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pytest

from lapis_pm import state_brief, state_brief_prompts, brief


class TestDegradedBriefTimeout:
    """Tests for the fail-fast timeout guard on qwen LLM calls."""

    def test_call_llm_with_timeout_returns_result_when_fast(self):
        """call_llm_with_timeout returns the result when call completes quickly."""
        def fast_call_llm(**kwargs):
            return "test response"

        with patch("agents_core.llm.call_llm", side_effect=fast_call_llm):
            result = state_brief._call_llm_with_timeout(
                prompt="test",
                system="test system",
                timeout_sec=30,
            )

        assert result == "test response", "Expected fast call to return result"

    def test_call_llm_with_timeout_handles_exception(self):
        """call_llm_with_timeout returns None when qwen raises an exception."""
        def failing_call_llm(**kwargs):
            raise OSError("Connection refused")

        with patch("agents_core.llm.call_llm", side_effect=failing_call_llm):
            result = state_brief._call_llm_with_timeout(
                prompt="test",
                system="test system",
                timeout_sec=30,
            )

        assert result is None, "Expected None when LLM raises exception"

    def test_degraded_brief_on_llm_failure(self):
        """On LLM failure (exception), brief degrades to placeholder with all sections."""
        def failing_call_llm(**kwargs):
            raise OSError("Connection refused")

        buckets = {
            "Built": ["arc-doc: tid1.md"],
            "Notable ratifications": ["feature: important"],
            "In flight": ["dispatched-tid: tid2"],
            "Captured — not yet built": ["thread1"],
            "Awaiting your call": ["brief1"],
        }

        with patch("agents_core.llm.call_llm", side_effect=failing_call_llm):
            result = state_brief._generate_prose(
                period="live",
                buckets=buckets,
                start_label="2026-06-16 12:00 PT",
            )

        # Degraded brief should contain all five sections
        assert result is not None
        assert "Built" in result
        assert "Notable ratifications" in result
        assert "In flight" in result
        assert "Captured" in result
        assert "Awaiting your call" in result
        assert "*(DEGRADED — StarHouse unreachable)*" in result

    def test_dry_run_mode_never_calls_llm(self):
        """LAPIS_BRIEF_DRY_RUN=1 skips the LLM call entirely."""
        call_llm_mock = MagicMock()

        with (
            patch.dict("os.environ", {"LAPIS_BRIEF_DRY_RUN": "1"}),
            patch("agents_core.llm.call_llm", side_effect=call_llm_mock),
        ):
            result = state_brief._generate_prose(
                period="live",
                buckets={
                    "Built": [],
                    "Notable ratifications": [],
                    "In flight": [],
                    "Captured — not yet built": [],
                    "Awaiting your call": [],
                },
                start_label="2026-06-16 12:00 PT",
            )

        assert result is not None
        call_llm_mock.assert_not_called()


class TestDashboardBaseRepoint:
    """Tests for DASHBOARD_BASE repoint to BRIX."""

    def test_dashboard_base_is_brix(self):
        """DASHBOARD_BASE points to BRIX (not stale StarHouse)."""
        assert brief.DASHBOARD_BASE == "http://203.0.113.10:8400", (
            f"Expected BRIX dashboard, got {brief.DASHBOARD_BASE}"
        )

    def test_dashboard_base_is_not_starhouse_ip(self):
        """DASHBOARD_BASE must not reference the old StarHouse IP."""
        assert "203.0.113.12" not in brief.DASHBOARD_BASE, (
            "DASHBOARD_BASE still references stale StarHouse IP"
        )


class TestGenerateBriefWallClockBudget:
    """Tests for the 30s wall-clock budget on brief generation."""

    def test_generate_brief_morning_returns_within_budget(self):
        """Morning brief completes within 30s even with qwen timeout."""
        def slow_call_llm(**kwargs):
            time.sleep(45)
            return "should not reach"

        with (
            patch("agents_core.llm.call_llm", side_effect=slow_call_llm),
            patch("lapis_pm.state_brief._mem") as mock_mem,
        ):
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.return_value = []

            start = time.time()
            try:
                state_brief.generate_brief(
                    period="morning",
                    start_ts=datetime.now(tz=timezone.utc),
                    stdout=True,
                )
            except SystemExit:
                pass  # stdout mode doesn't write file
            elapsed = time.time() - start

        assert elapsed < 32, f"Morning brief took {elapsed:.1f}s (budget: 30s + overhead)"

    def test_generate_brief_afternoon_returns_within_budget(self):
        """Afternoon brief completes within 30s even with qwen timeout."""
        def slow_call_llm(**kwargs):
            time.sleep(45)
            return "should not reach"

        with (
            patch("agents_core.llm.call_llm", side_effect=slow_call_llm),
            patch("lapis_pm.state_brief._mem") as mock_mem,
        ):
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.return_value = []

            start = time.time()
            try:
                state_brief.generate_brief(
                    period="afternoon",
                    start_ts=datetime.now(tz=timezone.utc),
                    stdout=True,
                )
            except SystemExit:
                pass  # stdout mode doesn't write file
            elapsed = time.time() - start

        assert elapsed < 32, f"Afternoon brief took {elapsed:.1f}s (budget: 30s + overhead)"

    def test_generate_brief_live_returns_within_budget(self):
        """Live brief completes within 30s even with qwen timeout."""
        def slow_call_llm(**kwargs):
            time.sleep(45)
            return "should not reach"

        with (
            patch("agents_core.llm.call_llm", side_effect=slow_call_llm),
            patch("lapis_pm.state_brief._mem") as mock_mem,
        ):
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.return_value = []

            start = time.time()
            try:
                state_brief.generate_brief(
                    period="live",
                    start_ts=datetime.now(tz=timezone.utc),
                    stdout=True,
                )
            except SystemExit:
                pass  # stdout mode doesn't write file
            elapsed = time.time() - start

        assert elapsed < 32, f"Live brief took {elapsed:.1f}s (budget: 30s + overhead)"


class TestReadGardenerObservations:
    """Tests for _read_gardener_observations()."""

    def test_parses_critical_and_warning_bullets(self):
        """Critical/Warning bullets are parsed with urgency label preserved."""
        entry = {
            "key": "gardener/derived/2026-07-05-0800",
            "value": (
                "# Gardener derived context — 2026-07-05T08:00:00Z\n\n"
                "- [Critical] A systemic deadlock is confirmed.  (evidence: mem:foo, mem:bar)\n"
                "- [Warning] Stale cursors detected.  (evidence: mem:baz)\n"
            ),
        }
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [
                {"key": entry["key"]}
            ]
            mock_mem.return_value.get.return_value = entry

            result = state_brief._read_gardener_observations()

        assert result == [
            "[Critical] A systemic deadlock is confirmed.",
            "[Warning] Stale cursors detected.",
        ]

    def test_filters_narration_keys(self):
        """narration-* keys are excluded from consideration, even if newer."""
        narration_entry = {
            "key": "gardener/derived/narration-2026-07-06-0900",
            "value": "- [Critical] Should not appear.  (evidence: mem:x)",
        }
        real_entry = {
            "key": "gardener/derived/2026-07-05-0800",
            "value": "- [Warning] Should appear.  (evidence: mem:y)",
        }
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [
                {"key": narration_entry["key"]},
                {"key": real_entry["key"]},
            ]
            mock_mem.return_value.get.return_value = real_entry

            result = state_brief._read_gardener_observations()

        assert result == ["[Warning] Should appear."]

    def test_no_entries_returns_empty_list(self):
        """No gardener/derived entries at all → []."""
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = []

            result = state_brief._read_gardener_observations()

        assert result == []

    def test_only_narration_entries_returns_empty_list(self):
        """If every entry is narration-*, result is []."""
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [
                {"key": "gardener/derived/narration-2026-07-06-0900"}
            ]

            result = state_brief._read_gardener_observations()

        assert result == []

    def test_info_and_unclassified_are_skipped(self):
        """Only Critical/Warning bullets are captured; other lines are ignored."""
        entry = {
            "key": "gardener/derived/2026-07-05-0800",
            "value": (
                "# Gardener derived context — 2026-07-05T08:00:00Z\n\n"
                "- No Critical/Warning cross-cutting observations this pass.\n"
                "- [Critical] Real one.  (evidence: mem:z)\n"
            ),
        }
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [{"key": entry["key"]}]
            mock_mem.return_value.get.return_value = entry

            result = state_brief._read_gardener_observations()

        assert result == ["[Critical] Real one."]

    def test_descending_key_sort_picks_latest(self):
        """Multiple entries → the lexicographically-latest key is used."""
        older = {"key": "gardener/derived/2026-07-01-0800"}
        newer = {"key": "gardener/derived/2026-07-05-0800"}
        newer_full = {**newer, "value": "- [Warning] Newest.  (evidence: mem:w)"}

        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [older, newer]
            mock_mem.return_value.get.return_value = newer_full

            result = state_brief._read_gardener_observations()

        mock_mem.return_value.get.assert_called_once_with(newer["key"])
        assert result == ["[Warning] Newest."]

    def test_read_buckets_degrades_on_mem_failure(self):
        """_read_buckets() swallows exceptions from the gardener read and shows []."""
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.side_effect = OSError("mem unreachable")

            buckets = state_brief._read_buckets(datetime.now(tz=timezone.utc))

        assert buckets[state_brief.B_GARDENER] == []


class TestFormatBucketSectionsPeriod:
    """Tests for period-conditional Gardener bucket exclusion."""

    def _buckets(self):
        return {
            "Built": [],
            "Notable ratifications": [],
            "In flight": [],
            "Captured — not yet built": [],
            "Awaiting your call": [],
            "Gardener Cross-Cutting Observations": ["[Critical] test observation"],
        }

    def test_daily_includes_gardener_section(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(), "2026-07-06 08:00 PT", period="daily",
        )
        assert "Gardener Cross-Cutting Observations" in sections
        assert "[Critical] test observation" in sections

    def test_weekly_omits_gardener_section(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(), "2026-06-29 08:00 PT", period="weekly",
        )
        assert "Gardener Cross-Cutting Observations" not in sections

    def test_default_period_is_daily(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(), "2026-07-06 08:00 PT",
        )
        assert "Gardener Cross-Cutting Observations" in sections

    def test_build_daily_prompt_includes_gardener(self):
        prompt = state_brief_prompts.build_daily_prompt(self._buckets(), "2026-07-06 08:00 PT")
        assert "Gardener Cross-Cutting Observations" in prompt

    def test_build_weekly_prompt_omits_gardener(self):
        prompt = state_brief_prompts.build_weekly_prompt(self._buckets(), "2026-06-29 08:00 PT")
        assert "Gardener Cross-Cutting Observations" not in prompt


class TestWeeklyDryRunAndDegraded:
    """Tests for weekly-period dry-run and degraded-fallback paths."""

    def test_weekly_dry_run_omits_gardener_bucket(self):
        """Weekly dry-run placeholder omits the Gardener section."""
        buckets = {
            "Built": [],
            "Notable ratifications": [],
            "In flight": [],
            "Captured — not yet built": [],
            "Awaiting your call": [],
            "Gardener Cross-Cutting Observations": ["[Critical] should not appear"],
        }
        with patch.dict("os.environ", {"LAPIS_BRIEF_DRY_RUN": "1"}):
            result = state_brief._generate_prose(
                period="weekly",
                buckets=buckets,
                start_label="2026-06-29 08:00 PT",
            )

        assert "Gardener Cross-Cutting Observations" not in result
        assert "should not appear" not in result

    def test_weekly_degraded_fallback_omits_gardener_bucket(self):
        """Weekly call_claude_cli failure degrades to a placeholder without Gardener."""
        buckets = {
            "Built": [],
            "Notable ratifications": [],
            "In flight": [],
            "Captured — not yet built": [],
            "Awaiting your call": [],
            "Gardener Cross-Cutting Observations": ["[Warning] should not appear"],
        }
        with patch("agents_core.llm.call_claude_cli", return_value=None):
            result = state_brief._generate_prose(
                period="weekly",
                buckets=buckets,
                start_label="2026-06-29 08:00 PT",
            )

        assert "*(DEGRADED — StarHouse unreachable)*" in result
        assert "Gardener Cross-Cutting Observations" not in result
        assert "should not appear" not in result
