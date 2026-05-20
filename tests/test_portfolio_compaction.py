"""Tests for portfolio compaction (lapis-pm-portfolio-compaction-v0 spec).

Coverage:
  - test_compact_dispatched_basic: multi-record key → stub shape verified
  - test_compact_dispatched_idempotent: second call is a no-op (mem written once)
  - test_compact_eligible_targets_age_filter: only 40-day-old target compacted
  - test_compact_eligible_targets_manual_land_shape: ts field used for age
  - test_compact_eligible_targets_skips_missing_dispatched: no_dispatched_key outcome
  - test_compact_eligible_targets_skips_unparseable_age: target skipped entirely
  - test_cmd_compact_dry_run: no mem writes in dry-run mode
  - test_cmd_compact_live: dispatched key replaced with stub after live run
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_days_ago(days: int) -> str:
    """Return an ISO timestamp exactly `days` days in the past (UTC, tz-aware)."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _make_records(n: int = 4) -> list[dict]:
    """Build n synthetic dispatch records."""
    return [
        {
            "agent_type": ["fixer", "reviewer", "fixer", "fixer"][i % 4],
            "ts": _ts_days_ago(50 - i),
            "status": "landed",
            "completed_at": _ts_days_ago(49 - i) if i < n - 1 else None,
        }
        for i in range(n)
    ]


def _make_mem_with_store(store: dict | None = None) -> MagicMock:
    """Return a mem mock backed by a dict `store` so set/get/list_by_prefix interact."""
    if store is None:
        store = {}

    mem = MagicMock()

    def _get(key):
        val = store.get(key)
        if val is None:
            return None
        return {"key": key, "content": val}

    def _set(key, content, tags=None, **kwargs):
        store[key] = content

    def _list_by_prefix(prefix, limit=500):
        return [
            {"key": k, "content": v}
            for k, v in store.items()
            if k.startswith(prefix)
        ][:limit]

    mem.get.side_effect = _get
    mem.set.side_effect = _set
    mem.list_by_prefix.side_effect = _list_by_prefix
    return mem


# ---------------------------------------------------------------------------
# test_compact_dispatched_basic
# ---------------------------------------------------------------------------

class TestCompactDispatchedBasic:

    def test_stub_shape(self):
        """compact_dispatched() replaces records with a correctly shaped stub."""
        records = _make_records(4)
        store = {
            "pm/dispatched/my-target": json.dumps(records),
        }
        mem = _make_mem_with_store(store)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            pm_core.compact_dispatched("my-target")

        saved = json.loads(store["pm/dispatched/my-target"])
        assert isinstance(saved, list)
        assert len(saved) == 1

        stub = saved[0]
        assert stub["compacted"] is True
        assert stub["dispatch_count"] == 4
        assert stub["agent_types"] == [r["agent_type"] for r in records]
        assert stub["first_dispatch"] == records[0]["ts"]
        # last_completed: last record with non-None completed_at
        expected_last = max(
            r["completed_at"] for r in records if r.get("completed_at")
        )
        assert stub["last_completed"] == expected_last
        assert "compacted_at" in stub

    def test_no_completed_at_omits_field(self):
        """If no record has completed_at, stub omits last_completed entirely."""
        records = [
            {"agent_type": "fixer", "ts": _ts_days_ago(40), "status": "landed"},
        ]
        store = {"pm/dispatched/t": json.dumps(records)}
        mem = _make_mem_with_store(store)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            pm_core.compact_dispatched("t")

        stub = json.loads(store["pm/dispatched/t"])[0]
        assert "last_completed" not in stub


# ---------------------------------------------------------------------------
# test_compact_dispatched_idempotent
# ---------------------------------------------------------------------------

class TestCompactDispatchedIdempotent:

    def test_second_call_no_write(self):
        """If the key is already a compacted stub, compact_dispatched() does not write."""
        stub = [{"compacted": True, "compacted_at": "2026-01-01T00:00:00+00:00",
                 "dispatch_count": 3, "agent_types": ["fixer"], "first_dispatch": "x"}]
        store = {"pm/dispatched/t": json.dumps(stub)}
        mem = _make_mem_with_store(store)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            pm_core.compact_dispatched("t")  # should be a no-op
            pm_core.compact_dispatched("t")  # definitely a no-op

        # mem.set was never called (store was pre-populated but via dict, not mem.set)
        mem.set.assert_not_called()

    def test_written_exactly_once_on_first_call(self):
        """First call writes; second call (key now compacted) does not write again."""
        records = _make_records(2)
        store = {"pm/dispatched/t": json.dumps(records)}
        mem = _make_mem_with_store(store)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            pm_core.compact_dispatched("t")
            write_count_after_first = mem.set.call_count
            pm_core.compact_dispatched("t")
            write_count_after_second = mem.set.call_count

        assert write_count_after_first == 1
        assert write_count_after_second == 1  # no additional write


# ---------------------------------------------------------------------------
# test_compact_eligible_targets_age_filter
# ---------------------------------------------------------------------------

class TestCompactEligibleTargetsAgeFilter:

    def test_only_old_target_compacted(self):
        """Only the 40-day-old target is compacted; the 10-day-old is skipped."""
        old_records = _make_records(3)
        store = {
            "pm/landed/old-target": json.dumps({"landed_at": _ts_days_ago(40), "arc_path": "/srv/lapis/x"}),
            "pm/dispatched/old-target": json.dumps(old_records),
            "pm/landed/new-target": json.dumps({"landed_at": _ts_days_ago(10), "arc_path": "/srv/lapis/y"}),
            "pm/dispatched/new-target": json.dumps(_make_records(2)),
        }
        mem = _make_mem_with_store(store)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            results = pm_core.compact_eligible_targets(min_age_days=30)

        outcomes = dict(results)
        assert outcomes.get("old-target") == "compacted"
        assert "new-target" not in outcomes

        # Verify old-target stub was written.
        saved = json.loads(store["pm/dispatched/old-target"])
        assert saved[0]["compacted"] is True

        # new-target dispatched key unchanged.
        saved_new = json.loads(store["pm/dispatched/new-target"])
        assert not saved_new[0].get("compacted")


# ---------------------------------------------------------------------------
# test_compact_eligible_targets_manual_land_shape
# ---------------------------------------------------------------------------

class TestCompactEligibleTargetsManualLandShape:

    def test_ts_field_used_for_age(self):
        """Manual-land shape uses `ts` not `landed_at`; age computed correctly."""
        records = _make_records(2)
        store = {
            "pm/landed/manual-target": json.dumps(
                {"manual": True, "ts": _ts_days_ago(35), "arc_path": "/srv/lapis/m"}
            ),
            "pm/dispatched/manual-target": json.dumps(records),
        }
        mem = _make_mem_with_store(store)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            results = pm_core.compact_eligible_targets(min_age_days=30)

        outcomes = dict(results)
        assert outcomes.get("manual-target") == "compacted"


# ---------------------------------------------------------------------------
# test_compact_eligible_targets_skips_missing_dispatched
# ---------------------------------------------------------------------------

class TestCompactEligibleTargetsSkipsMissingDispatched:

    def test_no_dispatched_key_returns_outcome(self):
        """landed key exists but no dispatched key → no_dispatched_key outcome, no crash."""
        store = {
            "pm/landed/ghost-target": json.dumps({"landed_at": _ts_days_ago(40), "arc_path": "/srv/lapis/g"}),
            # No pm/dispatched/ghost-target entry.
        }
        mem = _make_mem_with_store(store)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            results = pm_core.compact_eligible_targets(min_age_days=30)

        outcomes = dict(results)
        assert outcomes.get("ghost-target") == "no_dispatched_key"


# ---------------------------------------------------------------------------
# test_compact_eligible_targets_skips_unparseable_age
# ---------------------------------------------------------------------------

class TestCompactEligibleTargetsSkipsUnparseableAge:

    def test_skips_target_with_no_age_fields(self):
        """landed record with neither landed_at nor ts → target skipped entirely."""
        store = {
            "pm/landed/bad-target": json.dumps({"pr_num": 99, "arc_path": "/srv/lapis/b"}),
            "pm/dispatched/bad-target": json.dumps(_make_records(2)),
        }
        mem = _make_mem_with_store(store)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            results = pm_core.compact_eligible_targets(min_age_days=30)

        outcomes = dict(results)
        assert "bad-target" not in outcomes
        # Dispatched key untouched.
        saved = json.loads(store["pm/dispatched/bad-target"])
        assert not saved[0].get("compacted")


# ---------------------------------------------------------------------------
# test_cmd_compact_dry_run
# ---------------------------------------------------------------------------

class TestCmdCompactDryRun:

    def test_dry_run_no_writes(self, capsys):
        """cmd_compact with --dry-run prints eligible targets but writes nothing."""
        from lapis_pm.cli import cmd_compact

        records = _make_records(4)
        store = {
            "pm/landed/old-target": json.dumps({"landed_at": _ts_days_ago(40), "arc_path": "/srv/lapis/x"}),
            "pm/dispatched/old-target": json.dumps(records),
        }
        mem = _make_mem_with_store(store)

        args = SimpleNamespace(days=30, dry_run=True)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            rc = cmd_compact(args)

        assert rc == 0
        # No writes occurred.
        mem.set.assert_not_called()

        out = capsys.readouterr().out
        assert "Would compact" in out
        assert "old-target" in out

    def test_dry_run_days_validation(self, capsys):
        """cmd_compact rejects --days 0."""
        from lapis_pm.cli import cmd_compact

        args = SimpleNamespace(days=0, dry_run=True)
        store = {}
        mem = _make_mem_with_store(store)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            rc = cmd_compact(args)

        assert rc == 2


# ---------------------------------------------------------------------------
# test_cmd_compact_live
# ---------------------------------------------------------------------------

class TestCmdCompactLive:

    def test_live_compacts_eligible(self, capsys):
        """cmd_compact without --dry-run compacts eligible targets and prints summary."""
        from lapis_pm.cli import cmd_compact

        records = _make_records(3)
        store = {
            "pm/landed/old-target": json.dumps({"landed_at": _ts_days_ago(40), "arc_path": "/srv/lapis/x"}),
            "pm/dispatched/old-target": json.dumps(records),
        }
        mem = _make_mem_with_store(store)

        args = SimpleNamespace(days=30, dry_run=False)

        with patch("lapis_pm.pm_core._mem", return_value=mem):
            rc = cmd_compact(args)

        assert rc == 0

        # Dispatched key should now be a stub.
        saved = json.loads(store["pm/dispatched/old-target"])
        assert saved[0]["compacted"] is True
        assert saved[0]["dispatch_count"] == 3

        out = capsys.readouterr().out
        assert "Compacted 1 targets" in out
        assert "old-target" in out
        assert "stub" in out
