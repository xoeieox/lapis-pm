"""Tests for state brief degradation on StarHouse timeout.

Tests verify:
1. Brief completes within wall-clock budget when qwen/StarHouse is unreachable
2. Degraded brief contains all five deterministic local sections
3. Degraded brief has visible (DEGRADED) marker
4. All three periods (morning, afternoon, live) are covered by the timeout guard
5. DASHBOARD_BASE points to BRIX (not stale StarHouse)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
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
    """Tests for _read_gardener_observations().

    All entries come back from list_by_prefix() carrying their content
    directly (as the real mem client does) — there is no per-entry
    mem.get() call, so mock_mem.return_value.get is intentionally left
    unset/unasserted throughout this class.
    """

    def test_parses_critical_and_warning_bullets_with_evidence(self):
        """Critical/Warning bullets are parsed with urgency label and evidence preserved."""
        entry = {
            "key": "gardener/derived/2026-07-05-0800",
            "value": (
                "# Gardener derived context — 2026-07-05T08:00:00Z\n\n"
                "- [Critical] A systemic deadlock is confirmed.  (evidence: mem:foo, mem:bar)\n"
                "- [Warning] Stale cursors detected.  (evidence: mem:baz)\n"
            ),
        }
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [entry]

            result = state_brief._read_gardener_observations()

        assert result == [
            "[Critical] A systemic deadlock is confirmed.  (evidence: mem:foo, mem:bar)",
            "[Warning] Stale cursors detected.  (evidence: mem:baz)",
        ]
        mock_mem.return_value.get.assert_not_called()

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
                narration_entry,
                real_entry,
            ]

            result = state_brief._read_gardener_observations()

        assert result == ["[Warning] Should appear.  (evidence: mem:y)"]

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
            mock_mem.return_value.list_by_prefix.return_value = [entry]

            result = state_brief._read_gardener_observations()

        assert result == ["[Critical] Real one.  (evidence: mem:z)"]

    def test_descending_key_sort_picks_latest_non_weekly(self):
        """Non-weekly: multiple entries → only the lexicographically-latest key is used."""
        older = {
            "key": "gardener/derived/2026-07-01-0800",
            "value": "- [Warning] Older.  (evidence: mem:o)",
        }
        newer = {
            "key": "gardener/derived/2026-07-05-0800",
            "value": "- [Warning] Newest.  (evidence: mem:w)",
        }

        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [older, newer]

            result = state_brief._read_gardener_observations(period="daily")

        assert result == ["[Warning] Newest.  (evidence: mem:w)"]

    def test_non_weekly_caps_at_ten_observations(self):
        """Non-weekly (daily/morning/afternoon/live) caps output at 10 observations."""
        bullets = "\n".join(f"- [Warning] Item {i}." for i in range(15))
        entry = {"key": "gardener/derived/2026-07-05-0800", "value": bullets}
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [entry]

            result = state_brief._read_gardener_observations(period="morning")

        assert len(result) == 10

    def test_full_observation_text_survives_no_mid_word_truncation(self):
        """Observations longer than 200 chars must not be sliced mid-word.

        brief-gardener-obs-truncation-v0: the old `text[:200]` hard slice cut
        bullets mid-word. The count cap (_GARDENER_DAILY_CAP) is the sole
        prompt-size guard; per-item text is no longer truncated.
        """
        long_text = (
            "This observation deliberately runs well past the two hundred "
            "character mark that the old hard slice used to cut off mid-word, "
            "so that the assertion below can confirm the full sentence, "
            "including this final word, survives intact"
        )
        assert len(long_text) > 200
        entry = {
            "key": "gardener/derived/2026-07-24-1431",
            "value": f"- [Critical] {long_text}  (evidence: mem:foo, mem:bar)\n",
        }
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [entry]

            result = state_brief._read_gardener_observations()

        assert result == [
            f"[Critical] {long_text}  (evidence: mem:foo, mem:bar)"
        ]
        assert result[0].endswith("intact  (evidence: mem:foo, mem:bar)")

    def test_gardener_section_bounded_by_count_cap_not_char_slice(self):
        """D5: even with long observations, the section is bounded by the
        count cap (_GARDENER_DAILY_CAP), not by per-item character slicing.
        """
        long_text = "Word " * 150  # well over the old 200-char slice threshold
        bullets = "\n".join(
            f"- [Warning] {long_text.strip()} item{i}." for i in range(15)
        )
        entry = {"key": "gardener/derived/2026-07-24-1431", "value": bullets}
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_by_prefix.return_value = [entry]

            result = state_brief._read_gardener_observations(period="morning")

        assert len(result) == state_brief._GARDENER_DAILY_CAP
        for obs in result:
            assert len(obs) > 200

    def test_weekly_window_includes_all_entries_within_7_days_uncapped(self):
        """Weekly period pulls every entry in the trailing 7 days, uncapped."""
        entries = [
            {
                "key": f"gardener/derived/2026-07-0{i}-0800",
                "value": f"- [Warning] Day {i} observation.  (evidence: mem:day{i})",
            }
            for i in range(1, 8)
        ]
        # 15 bullets total (well above the daily cap of 10) across 7 days.
        with patch("lapis_pm.state_brief._mem") as mock_mem, \
             patch("lapis_pm.state_brief.datetime") as mock_dt:
            mock_mem.return_value.list_by_prefix.return_value = entries
            mock_dt.now.return_value = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
            mock_dt.strptime = datetime.strptime

            result = state_brief._read_gardener_observations(period="weekly")

        assert len(result) == 7
        assert all("evidence:" in item for item in result)

    def test_weekly_window_excludes_entries_older_than_7_days(self):
        """Weekly period excludes entries whose key-date falls outside the 7-day window."""
        in_window = {
            "key": "gardener/derived/2026-07-06-0800",
            "value": "- [Critical] In window.  (evidence: mem:in)",
        }
        out_of_window = {
            "key": "gardener/derived/2026-06-01-0800",
            "value": "- [Critical] Out of window.  (evidence: mem:out)",
        }
        with patch("lapis_pm.state_brief._mem") as mock_mem, \
             patch("lapis_pm.state_brief.datetime") as mock_dt:
            mock_mem.return_value.list_by_prefix.return_value = [in_window, out_of_window]
            mock_dt.now.return_value = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
            mock_dt.strptime = datetime.strptime

            result = state_brief._read_gardener_observations(period="weekly")

        assert result == ["[Critical] In window.  (evidence: mem:in)"]

    def test_read_buckets_degrades_on_mem_failure(self):
        """_read_buckets() swallows exceptions from the gardener read and shows []."""
        with patch("lapis_pm.state_brief._mem") as mock_mem:
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.side_effect = OSError("mem unreachable")

            buckets = state_brief._read_buckets(datetime.now(tz=timezone.utc))

        assert buckets[state_brief.B_GARDENER] == []

    def test_read_buckets_passes_period_to_gardener_read(self):
        """_read_buckets(period=...) forwards the period to _read_gardener_observations."""
        with patch("lapis_pm.state_brief._mem") as mock_mem, \
             patch(
                 "lapis_pm.state_brief._read_gardener_observations",
                 return_value=["[Critical] weekly synth"],
             ) as mock_read:
            mock_mem.return_value.list_all.return_value = []

            buckets = state_brief._read_buckets(
                datetime.now(tz=timezone.utc), period="weekly",
            )

        mock_read.assert_called_once_with(period="weekly")
        assert buckets[state_brief.B_GARDENER] == ["[Critical] weekly synth"]


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

    def test_weekly_omits_gardener_bucket_header(self):
        """Weekly omits the 'Gardener Cross-Cutting Observations' bucket header,
        but still appends the data as a raw synthesis block."""
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(), "2026-06-29 08:00 PT", period="weekly",
        )
        assert "## Gardener Cross-Cutting Observations" not in sections
        assert "Gardener Observations (7-day window for synthesis)" in sections
        assert "[Critical] test observation" in sections

    def test_weekly_omits_data_block_when_no_gardener_items(self):
        """Weekly appends nothing Gardener-related when the bucket is empty."""
        buckets = self._buckets()
        buckets["Gardener Cross-Cutting Observations"] = []
        sections = state_brief_prompts.format_bucket_sections(
            buckets, "2026-06-29 08:00 PT", period="weekly",
        )
        assert "Gardener" not in sections

    def test_default_period_is_daily(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(), "2026-07-06 08:00 PT",
        )
        assert "Gardener Cross-Cutting Observations" in sections

    def test_build_daily_prompt_includes_gardener(self):
        prompt = state_brief_prompts.build_daily_prompt(self._buckets(), "2026-07-06 08:00 PT")
        assert "Gardener Cross-Cutting Observations" in prompt

    def test_build_weekly_prompt_omits_gardener_header_but_includes_data(self):
        prompt = state_brief_prompts.build_weekly_prompt(self._buckets(), "2026-06-29 08:00 PT")
        assert "## Gardener Cross-Cutting Observations" not in prompt
        assert "Gardener Observations (7-day window for synthesis)" in prompt
        assert "[Critical] test observation" in prompt


class TestWeeklyDryRunAndDegraded:
    """Tests for weekly-period dry-run and degraded-fallback paths."""

    def test_weekly_dry_run_includes_gardener_data_block_not_bucket_header(self):
        """Weekly dry-run placeholder omits the Gardener bucket header but
        includes the raw 7-day data block (consumed by the Weekly Arc synthesis
        in the real LLM path; still rendered here since dry-run reuses the
        same format_bucket_sections call)."""
        buckets = {
            "Built": [],
            "Notable ratifications": [],
            "In flight": [],
            "Captured — not yet built": [],
            "Awaiting your call": [],
            "Gardener Cross-Cutting Observations": ["[Critical] weekly pattern item"],
        }
        with patch.dict("os.environ", {"LAPIS_BRIEF_DRY_RUN": "1"}):
            result = state_brief._generate_prose(
                period="weekly",
                buckets=buckets,
                start_label="2026-06-29 08:00 PT",
            )

        assert "## Gardener Cross-Cutting Observations" not in result
        assert "Gardener Observations (7-day window for synthesis)" in result
        assert "weekly pattern item" in result

    def test_weekly_degraded_fallback_includes_gardener_data_block_not_bucket_header(self):
        """Weekly call_claude_cli failure degrades to a placeholder that still
        omits the Gardener bucket header but includes the raw data block."""
        buckets = {
            "Built": [],
            "Notable ratifications": [],
            "In flight": [],
            "Captured — not yet built": [],
            "Awaiting your call": [],
            "Gardener Cross-Cutting Observations": ["[Warning] weekly pattern item"],
        }
        with patch("agents_core.llm.call_claude_cli", return_value=None):
            result = state_brief._generate_prose(
                period="weekly",
                buckets=buckets,
                start_label="2026-06-29 08:00 PT",
            )

        assert "*(DEGRADED — StarHouse unreachable)*" in result
        assert "## Gardener Cross-Cutting Observations" not in result
        assert "Gardener Observations (7-day window for synthesis)" in result
        assert "weekly pattern item" in result


class TestReadArcClimate:
    """Tests for _read_arc_climate() — the arc-reconciler (gardener-arc-climate-v0)."""

    def _write_arc_doc(self, tmp_path, slug: str, body: str, *, age_days: int = 0):
        lapis_state = tmp_path / "lapis_state"
        lapis_state.mkdir(parents=True, exist_ok=True)
        doc = lapis_state / f"{slug}.md"
        doc.write_text(body, encoding="utf-8")
        if age_days:
            mtime = (datetime.now(tz=timezone.utc) - timedelta(days=age_days)).timestamp()
            os.utime(doc, (mtime, mtime))
        return doc

    def test_non_weekly_period_short_circuits(self, tmp_path):
        """Non-weekly periods return [] immediately without touching disk (v0 is weekly-only)."""
        self._write_arc_doc(tmp_path, "some-arc", "NEXT: do the thing", age_days=30)
        room_path_mock = MagicMock()
        with patch("lapis_pm.state_brief.room_path", room_path_mock):
            result = state_brief._read_arc_climate(datetime.now(tz=timezone.utc), period="daily")

        assert result == []
        room_path_mock.assert_not_called()

    def test_gone_quiet_classification(self, tmp_path):
        """A stale arc with a queryable (but timestamp-less) timer anchor classifies gone-quiet."""
        self._write_arc_doc(
            tmp_path, "gone-quiet-arc",
            "# Gone Quiet Arc\n\nNEXT: write the follow-up docs\n\nAnchors: foo.service\n",
            age_days=30,
        )
        with (
            patch("lapis_pm.state_brief.room_path", return_value=tmp_path / "lapis_state"),
            patch("lapis_pm.deploy_inventory._show_unit", return_value={"ActiveState": "active"}),
            patch("lapis_pm.deploy_inventory.read_status_json", return_value=None),
            patch("lapis_pm.state_brief._arc_weaver_signal", return_value=None),
        ):
            result = state_brief._read_arc_climate(datetime.now(tz=timezone.utc), period="weekly")

        assert len(result) == 1
        assert result[0]["classification"] == "gone-quiet"
        assert result[0]["slug"] == "gone-quiet-arc"
        assert "gone-quiet-arc" in result[0]["text"]
        assert "30d of silence" in result[0]["text"]
        assert "write the follow-up docs" in result[0]["text"]

    def test_silently_advanced_fires_regardless_of_age(self, tmp_path):
        """A merged PR whose declared NEXT still reads as pending action fires immediately,
        even though the doc was touched moments ago (independent of _ARC_STALE_DAYS)."""
        self._write_arc_doc(
            tmp_path, "realized-arc",
            "# Realized Arc\n\nNEXT: review and merge PR #219 (lapis-pm)\n",
            age_days=0,
        )
        merged_at = (datetime.now(tz=timezone.utc) - timedelta(days=5)).isoformat()
        fake_pr = {"state": "closed", "merged": True, "merged_at": merged_at, "updated_at": merged_at}
        with (
            patch("lapis_pm.state_brief.room_path", return_value=tmp_path / "lapis_state"),
            patch("agents_core.forgejo.get_pr", return_value=fake_pr) as mock_get_pr,
            patch("lapis_pm.deploy_inventory.read_status_json", return_value=None),
            patch("lapis_pm.state_brief._arc_weaver_signal", return_value=None),
        ):
            result = state_brief._read_arc_climate(datetime.now(tz=timezone.utc), period="weekly")

        # "merge" (the prose word immediately before "PR #219") must never be
        # mistaken for the repo — only a hyphenated token or the "(repo)"
        # parenthetical form counts. This doc uses the parenthetical form.
        mock_get_pr.assert_called_once_with("lapis-pm", 219)
        assert len(result) == 1
        assert result[0]["classification"] == "silently-advanced"
        assert "PR #219 merged 5d ago" in result[0]["text"]
        assert "[CONTEXT DRIFT: declared NEXT still says" in result[0]["text"]

    def test_unresolvable_classification(self, tmp_path):
        """A stale arc with no PR/timer/weaver anchor at all classifies unresolvable."""
        self._write_arc_doc(
            tmp_path, "orphan-arc",
            "# Orphan Arc\n\nNEXT: think about this more\n",
            age_days=25,
        )
        with (
            patch("lapis_pm.state_brief.room_path", return_value=tmp_path / "lapis_state"),
            patch("lapis_pm.deploy_inventory.read_status_json", return_value=None),
            patch("lapis_pm.state_brief._arc_weaver_signal", return_value=None),
        ):
            result = state_brief._read_arc_climate(datetime.now(tz=timezone.utc), period="weekly")

        assert len(result) == 1
        assert result[0]["classification"] == "unresolvable"
        assert "cannot verify" in result[0]["text"]
        assert "25d since the arc-doc was touched" in result[0]["text"]

    def test_moving_arc_is_omitted(self, tmp_path):
        """An arc touched within _ARC_STALE_DAYS is omitted entirely — silence is not churn."""
        self._write_arc_doc(
            tmp_path, "moving-arc",
            "# Moving Arc\n\nNEXT: keep iterating\n",
            age_days=3,
        )
        with (
            patch("lapis_pm.state_brief.room_path", return_value=tmp_path / "lapis_state"),
            patch("lapis_pm.deploy_inventory.read_status_json", return_value=None),
            patch("lapis_pm.state_brief._arc_weaver_signal", return_value=None),
        ):
            result = state_brief._read_arc_climate(datetime.now(tz=timezone.utc), period="weekly")

        assert result == []

    def test_partial_failure_forgejo_up_weaver_down_still_classifies(self, tmp_path):
        """DoD #4: one ground-truth source failing (weaver) drops only that signal —
        the arc is still classified using the Forgejo signal that succeeded."""
        self._write_arc_doc(
            tmp_path, "partial-failure-arc",
            "# Partial Failure Arc\n\nNEXT: schema design discussion\n\nlapis-pm PR #300\n",
            age_days=30,
        )
        old_ts = (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat()
        fake_pr = {"state": "open", "merged": False, "updated_at": old_ts}
        with (
            patch("lapis_pm.state_brief.room_path", return_value=tmp_path / "lapis_state"),
            patch("agents_core.forgejo.get_pr", return_value=fake_pr),
            patch("lapis_pm.deploy_inventory.read_status_json", return_value=None),
            patch("lapis_pm.state_brief._arc_weaver_signal", side_effect=OSError("weaver down")),
        ):
            result = state_brief._read_arc_climate(datetime.now(tz=timezone.utc), period="weekly")

        assert len(result) == 1
        assert result[0]["classification"] == "gone-quiet"
        assert result[0]["slug"] == "partial-failure-arc"

    def test_pr_ground_truth_retries_lapis_org_on_404(self, tmp_path):
        """A repo living under the lapis org 404s on the default (Erah) owner lookup;
        _arc_pr_ground_truth must retry with owner=LAPIS_ORG rather than silently
        dropping the signal (PR #225 review finding)."""
        self._write_arc_doc(
            tmp_path, "cross-repo-arc",
            "# Cross Repo Arc\n\nNEXT: schema design discussion\n\nagents-core PR #42\n",
            age_days=30,
        )
        old_ts = (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat()
        fake_pr = {"state": "open", "merged": False, "updated_at": old_ts}

        import httpx
        req = httpx.Request("GET", "http://forgejo/api/v1/repos/Erah/agents-core/pulls/42")
        resp = httpx.Response(404, request=req)
        not_found = httpx.HTTPStatusError("404 Not Found", request=req, response=resp)

        def fake_get_pr(repo, pr_number, owner=None):
            if owner is None:
                raise not_found
            assert owner == "lapis"
            return fake_pr

        with (
            patch("lapis_pm.state_brief.room_path", return_value=tmp_path / "lapis_state"),
            patch("agents_core.forgejo.get_pr", side_effect=fake_get_pr),
            patch("lapis_pm.deploy_inventory.read_status_json", return_value=None),
            patch("lapis_pm.state_brief._arc_weaver_signal", return_value=None),
        ):
            result = state_brief._read_arc_climate(datetime.now(tz=timezone.utc), period="weekly")

        assert len(result) == 1
        assert result[0]["classification"] == "gone-quiet"
        assert result[0]["slug"] == "cross-repo-arc"

    def test_zero_available_signals_is_unresolvable_not_catastrophic(self, tmp_path):
        """DoD #4: an arc with zero available signals classifies unresolvable, not []."""
        self._write_arc_doc(
            tmp_path, "no-signal-arc",
            "# No Signal Arc\n\nNEXT: figure this out\n",
            age_days=40,
        )
        with (
            patch("lapis_pm.state_brief.room_path", return_value=tmp_path / "lapis_state"),
            patch("lapis_pm.deploy_inventory.read_status_json", side_effect=OSError("unreachable")),
            patch("lapis_pm.state_brief._arc_weaver_signal", side_effect=Exception("weaver down")),
        ):
            result = state_brief._read_arc_climate(datetime.now(tz=timezone.utc), period="weekly")

        assert len(result) == 1
        assert result[0]["classification"] == "unresolvable"

    def test_no_lapis_state_dir_returns_empty(self, tmp_path):
        """Missing arc-doc directory degrades to [] (best-effort, matches _arc_docs_since)."""
        missing = tmp_path / "does-not-exist"
        with patch("lapis_pm.state_brief.room_path", return_value=missing):
            result = state_brief._read_arc_climate(datetime.now(tz=timezone.utc), period="weekly")

        assert result == []

    def test_read_buckets_wires_climate_text_only(self, tmp_path):
        """_read_buckets() threads _read_arc_climate's bullet text into B_CLIMATE."""
        with (
            patch("lapis_pm.state_brief._mem") as mock_mem,
            patch(
                "lapis_pm.state_brief._read_arc_climate",
                return_value=[{"slug": "x", "classification": "gone-quiet", "text": "x: stale"}],
            ) as mock_climate,
        ):
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.return_value = []

            buckets = state_brief._read_buckets(datetime.now(tz=timezone.utc), period="weekly")

        mock_climate.assert_called_once()
        assert buckets[state_brief.B_CLIMATE] == ["x: stale"]

    def test_read_buckets_degrades_climate_on_catastrophic_failure(self, tmp_path):
        """_read_buckets() swallows exceptions from _read_arc_climate and shows []."""
        with (
            patch("lapis_pm.state_brief._mem") as mock_mem,
            patch(
                "lapis_pm.state_brief._read_arc_climate",
                side_effect=OSError("catastrophic"),
            ),
        ):
            mock_mem.return_value.list_all.return_value = []
            mock_mem.return_value.list_by_prefix.return_value = []

            buckets = state_brief._read_buckets(datetime.now(tz=timezone.utc), period="weekly")

        assert buckets[state_brief.B_CLIMATE] == []


class TestClimateBucketRendering:
    """Tests for Climate bucket special-casing in format_bucket_sections()."""

    def _buckets(self, climate_items=None):
        return {
            "Built": [],
            "Notable ratifications": [],
            "In flight": [],
            "Captured — not yet built": [],
            "Awaiting your call": [],
            "Gardener Cross-Cutting Observations": [],
            "Climate": climate_items or [],
        }

    def test_daily_omits_climate_entirely(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["some-arc: 30d of silence"]), "2026-07-06 08:00 PT", period="daily",
        )
        assert "Climate" not in sections

    def test_weekly_renders_climate_when_present(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["some-arc: 30d of silence"]), "2026-07-06 08:00 PT", period="weekly",
        )
        assert "## Climate" in sections
        assert "some-arc: 30d of silence" in sections

    def test_weekly_omits_climate_when_empty(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets([]), "2026-07-06 08:00 PT", period="weekly",
        )
        assert "Climate" not in sections

    def test_climate_appears_after_gardener_data_block(self):
        sections = state_brief_prompts.format_bucket_sections(
            self._buckets(["some-arc: 30d of silence"]), "2026-07-06 08:00 PT", period="weekly",
        )
        assert sections.index("## Climate") > sections.index("Awaiting your call")
