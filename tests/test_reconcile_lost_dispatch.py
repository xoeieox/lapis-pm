"""Unit tests for lost-dispatch detection and handling.

Coverage:
  (a) terminal-queue-job-no-PR → classified as lost (needs_retry)
  (b) first-loss retry → _act_lost_fixer_retry dispatches child, increments lost_retry_count
  (c) second-loss brief → _act_lost_brief called after retry child also terminal
  (d) merged-PR-no-misclassification → dispatch with a post-dispatch merged PR is NOT lost
  (e) open-PR-no-misclassification → dispatch with a post-dispatch open PR is NOT lost
  (f) forgejo_ok=False → classification deferred (returns empty)
  (g) pending-child → classification deferred (never preempt live fixer)
  (h) non-fixer agent types → not classified as lost
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import MagicMock, patch, call

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixer_record(
    gpu_id: str = "gpu-original-001",
    ts: str = "2026-05-01T10:00:00-07:00",
    status: str = "failed",
    lost_retry_count: int = 0,
    parent_gpu_id: str | None = None,
    agent_type: str = "fixer",
    error: str | None = None,
) -> dict:
    rec = {
        "gpu_id": gpu_id,
        "spec_id": f"spec-{gpu_id}",
        "agent_type": agent_type,
        "intent": "implement the spec",
        "repo": "lapis-pm",
        "ts": ts,
        "status": status,
        "retry_count": 0,
        "lost_retry_count": lost_retry_count,
    }
    if parent_gpu_id is not None:
        rec["parent_gpu_id"] = parent_gpu_id
    if error is not None:
        rec["error"] = error
    return rec


def _open_pr(created_at: str, number: int = 1) -> dict:
    return {
        "number": number,
        "created_at": created_at,
        "head": {"ref": "lapis/my-target/forced"},
        "title": "test PR",
    }


def _comment_with_tag(tag: str, ts: str = "2026-05-01T11:00:00-07:00") -> MagicMock:
    c = MagicMock()
    c.tags = [tag, "pm:observation"]
    c.ts = ts
    return c


# ---------------------------------------------------------------------------
# (a) Classification: terminal, no PR → needs_retry
# ---------------------------------------------------------------------------

class TestClassifyLostFixer:

    def test_terminal_failed_no_pr_classified_as_lost(self):
        """Terminal failed fixer with no PR → appears in needs_retry."""
        rec = _fixer_record(status="failed")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1
        assert retry[0]["gpu_id"] == rec["gpu_id"]
        assert brief == []

    def test_terminal_processed_no_pr_classified_as_lost(self):
        """Terminal processed fixer (ran to completion, no PR) → needs_retry."""
        rec = _fixer_record(status="processed")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1

    def test_pending_fixer_not_classified(self):
        """Pending fixer (job still running) → NOT lost; never preempt live fixer."""
        rec = _fixer_record(status="pending")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert brief == []

    def test_forgejo_unreachable_defers_classification(self):
        """forgejo_ok=False → classification deferred; return ([], [])."""
        rec = _fixer_record(status="failed")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=False
            )
        assert retry == []
        assert brief == []

    def test_non_fixer_not_classified(self):
        """reviewer and brief agent types are NOT classified as lost."""
        for atype in ("reviewer", "reviewer_fresh", "brief", "fixer_retry"):
            rec = _fixer_record(status="failed", agent_type=atype)
            with patch("lapis_pm.episodic.all_comments", return_value=[]):
                retry, brief = pm_core._find_lost_fixer_dispatches(
                    "my-target", [rec], open_prs=[], forgejo_ok=True
                )
            assert retry == [], f"agent_type={atype} should not be in needs_retry"
            assert brief == [], f"agent_type={atype} should not be in needs_brief"

    def test_retry_child_not_classified_directly(self):
        """Dispatch with parent_gpu_id (retry child) is skipped; only originals."""
        child = _fixer_record(
            gpu_id="gpu-child-001",
            status="failed",
            parent_gpu_id="gpu-original-001",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [child], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert brief == []


# ---------------------------------------------------------------------------
# (e) Open PR prevents lost classification
# ---------------------------------------------------------------------------

class TestOpenPrPreventsLost:

    def test_open_pr_after_dispatch_prevents_lost(self):
        """Open PR created after dispatch ts → NOT lost."""
        rec = _fixer_record(
            ts="2026-05-01T10:00:00-07:00",
            status="failed",
        )
        pr = _open_pr(created_at="2026-05-01T10:30:00-07:00")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[pr], forgejo_ok=True
            )
        assert retry == []
        assert brief == []

    def test_open_pr_before_dispatch_does_not_prevent_lost(self):
        """Open PR created BEFORE dispatch ts → does NOT prevent lost classification."""
        rec = _fixer_record(
            ts="2026-05-01T10:00:00-07:00",
            status="failed",
        )
        pr = _open_pr(created_at="2026-04-30T10:00:00-07:00")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[pr], forgejo_ok=True
            )
        assert len(retry) == 1


# ---------------------------------------------------------------------------
# (d) Merged PR prevents lost classification
# ---------------------------------------------------------------------------

class TestMergedPrPreventsLost:

    def test_merged_pr_observation_after_dispatch_prevents_lost(self):
        """Merged PR observation ts >= dispatch ts → NOT lost."""
        rec = _fixer_record(
            ts="2026-05-01T10:00:00-07:00",
            status="failed",
        )
        # Episodic has a pm:pr-merged observation AFTER the dispatch
        merged_obs = _comment_with_tag(
            "pm:pr-merged:5", ts="2026-05-01T11:00:00-07:00"
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[merged_obs]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert brief == []

    def test_merged_pr_observation_before_dispatch_does_not_prevent_lost(self):
        """Merged PR observation ts < dispatch ts → does NOT prevent lost."""
        rec = _fixer_record(
            ts="2026-05-01T10:00:00-07:00",
            status="failed",
        )
        merged_obs = _comment_with_tag(
            "pm:pr-merged:3", ts="2026-04-30T09:00:00-07:00"
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[merged_obs]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1


# ---------------------------------------------------------------------------
# (g) Pending child → deferred (never preempt live fixer)
# ---------------------------------------------------------------------------

class TestPendingChildDeferred:

    def test_pending_retry_child_defers_classification(self):
        """Original with lost_retry_count=0 but pending retry child → skipped."""
        original = _fixer_record(gpu_id="gpu-orig", status="failed")
        child = _fixer_record(
            gpu_id="gpu-child",
            status="pending",
            parent_gpu_id="gpu-orig",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [original, child], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert brief == []

    def test_pending_retry_child_defers_brief_too(self):
        """Original with lost_retry_count=1 but pending retry child → brief deferred."""
        original = _fixer_record(
            gpu_id="gpu-orig", status="failed", lost_retry_count=1
        )
        child = _fixer_record(
            gpu_id="gpu-child",
            status="pending",
            parent_gpu_id="gpu-orig",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [original, child], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert brief == []


# ---------------------------------------------------------------------------
# (b) First-loss retry: _act_lost_fixer_retry
# ---------------------------------------------------------------------------

class TestActLostFixerRetry:

    def test_dispatches_child_and_increments_lost_retry_count(self):
        """_act_lost_fixer_retry creates child record and sets lost_retry_count=1."""
        orig = _fixer_record(gpu_id="gpu-orig-001", status="failed")
        existing_records = [orig]

        fake_res = MagicMock()
        fake_res.task_id = "gpu-retry-001"
        fake_res.spec_id = "spec-retry-001"

        written_obs = []

        def capture_obs(tid, content, extra_tags=None):
            c = MagicMock()
            c.tags = extra_tags or []
            written_obs.append(content)
            return c

        with (
            patch("lapis_pm.pm_core._SHAPER") as mock_shaper,
            patch("lapis_pm.pm_core.load_dispatched", return_value=existing_records),
            patch("lapis_pm.pm_core.save_dispatched") as mock_save,
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.episodic.write_observation", side_effect=capture_obs),
        ):
            mock_shaper.dispatch.return_value = fake_res
            result = pm_core._act_lost_fixer_retry("my-target", orig)

        assert result == "fixer_lost:retrying:dispatch=gpu-orig-001"

        # save_dispatched called once with updated records
        mock_save.assert_called_once()
        saved = mock_save.call_args[0][1]  # second positional arg = records list

        # Original record has lost_retry_count=1
        orig_saved = next(r for r in saved if r["gpu_id"] == "gpu-orig-001")
        assert orig_saved["lost_retry_count"] == 1

        # Child record appended with correct fields
        child_saved = next(r for r in saved if r["gpu_id"] == "gpu-retry-001")
        assert child_saved["parent_gpu_id"] == "gpu-orig-001"
        assert child_saved["status"] == "pending"
        assert child_saved["agent_type"] == "fixer"

        # Observation written with lost-dispatch-retry tag
        assert any("Lost dispatch" in obs and "gpu-orig-001" in obs
                   for obs in written_obs)

    def test_decision_string_format(self):
        """Decision string matches expected log format."""
        orig = _fixer_record(gpu_id="gpu-X", status="failed")

        fake_res = MagicMock()
        fake_res.task_id = "gpu-Y"
        fake_res.spec_id = "spec-Y"

        with (
            patch("lapis_pm.pm_core._SHAPER") as mock_shaper,
            patch("lapis_pm.pm_core.load_dispatched", return_value=[orig]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.episodic.write_observation"),
        ):
            mock_shaper.dispatch.return_value = fake_res
            result = pm_core._act_lost_fixer_retry("my-target", orig)

        assert result.startswith("fixer_lost:retrying:dispatch=")


# ---------------------------------------------------------------------------
# (c) Second-loss brief: _act_lost_brief + classify path
# ---------------------------------------------------------------------------

class TestSecondLossBrief:

    def test_terminal_retry_child_triggers_brief_classification(self):
        """lost_retry_count=1 + terminal retry child → needs_brief with both records."""
        original = _fixer_record(
            gpu_id="gpu-orig",
            status="failed",
            lost_retry_count=1,
            error="ERROR: SEGV",
        )
        child = _fixer_record(
            gpu_id="gpu-child",
            status="failed",
            parent_gpu_id="gpu-orig",
            error="ERROR: GP fault",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [original, child], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert len(brief) == 1
        orig_r, child_r = brief[0]
        assert orig_r["gpu_id"] == "gpu-orig"
        assert child_r is not None
        assert child_r["gpu_id"] == "gpu-child"

    def test_act_lost_brief_decision_string(self):
        """_act_lost_brief returns fixer_lost:briefing:dispatches=<id1>,<id2>."""
        original = _fixer_record(
            gpu_id="gpu-orig", status="failed", error="ERROR: SEGV"
        )
        child = _fixer_record(
            gpu_id="gpu-child",
            status="failed",
            parent_gpu_id="gpu-orig",
            error="ERROR: GP fault",
        )

        fake_brief = MagicMock()
        fake_brief.comment_id = "brief-abc"
        fake_brief.pushed = False
        fake_brief.body = "brief body"
        fake_brief.target_id = "my-target"

        with (
            patch("lapis_pm.pm_core.brief.synthesize", return_value=fake_brief),
            patch("lapis_pm.pm_core.set_outstanding_brief"),
            patch("lapis_pm.pm_core.episodic.spec", return_value="spec summary"),
        ):
            result = pm_core._act_lost_brief("my-target", original, child)

        assert result == "fixer_lost:briefing:dispatches=gpu-orig,gpu-child"

    def test_act_lost_brief_no_retry_child(self):
        """_act_lost_brief with no retry child still emits brief."""
        original = _fixer_record(gpu_id="gpu-solo", status="failed")

        fake_brief = MagicMock()
        fake_brief.comment_id = "brief-xyz"
        fake_brief.pushed = False
        fake_brief.body = "brief body"
        fake_brief.target_id = "my-target"

        with (
            patch("lapis_pm.pm_core.brief.synthesize", return_value=fake_brief),
            patch("lapis_pm.pm_core.set_outstanding_brief"),
            patch("lapis_pm.pm_core.episodic.spec", return_value="spec summary"),
        ):
            result = pm_core._act_lost_brief("my-target", original, None)

        # No retry child → dispatch_ids is just orig_id
        assert result == "fixer_lost:briefing:dispatches=gpu-solo"

    def test_act_lost_brief_uses_normal_priority(self):
        """_act_lost_brief uses NORMAL (not HIGH) Pushover priority."""
        from agents_core.notify import Priority as NotifyPriority

        original = _fixer_record(gpu_id="gpu-brief", status="failed")

        fake_brief = MagicMock()
        fake_brief.comment_id = "brief-000"
        fake_brief.pushed = False
        fake_brief.body = "b"
        fake_brief.target_id = "my-target"

        captured_notify = []

        def capture_synthesize(tid, trigger, query="", notify=None, **kw):
            captured_notify.append(notify)
            return fake_brief

        with (
            patch("lapis_pm.pm_core.brief.synthesize", side_effect=capture_synthesize),
            patch("lapis_pm.pm_core.set_outstanding_brief"),
            patch("lapis_pm.pm_core.episodic.spec", return_value="spec"),
        ):
            pm_core._act_lost_brief("my-target", original, None)

        assert captured_notify == [NotifyPriority.NORMAL]


# ---------------------------------------------------------------------------
# Tick-integration: lost dispatch path fires in decide
# ---------------------------------------------------------------------------

def _make_lost_target(pm_repo: str = "lapis-pm"):
    from agents_core.targets import Target
    t = MagicMock(spec=Target)
    t.pm_bound = True
    t.paused = False
    t.pm_repo = pm_repo
    t.pm_authority = "advisory"
    t.data = {}
    return t


_TICK_BASE_PATCHES = [
    ("lapis_pm.pm_core._reconcile_dispatched_with_queue", dict(return_value=0)),
    ("lapis_pm.pm_core.get_cursor", dict(return_value=None)),
    ("lapis_pm.pm_core.set_cursor", {}),
    ("lapis_pm.pm_core.get_pause_state", dict(return_value=None)),
    ("lapis_pm.pm_core.set_pause_state", {}),
    ("lapis_pm.episodic.since", dict(return_value=[])),
    ("lapis_pm.pm_core._encode_new_prs", dict(return_value=[])),
    ("lapis_pm.pm_core._encode_pr_sha_updates", dict(return_value=0)),
    ("lapis_pm.pm_core._encode_gpu_results", dict(return_value=(0, []))),
    ("lapis_pm.pm_core._encode_merged_prs", dict(return_value=0)),
    ("lapis_pm.pm_core._encode_user_comments", dict(return_value=[])),
    ("lapis_pm.pm_core._consume_brief_decisions", dict(return_value=None)),
    ("lapis_pm.pm_core._is_auto_land_eligible", dict(return_value=False)),
    ("lapis_pm.pm_core._persist_review_state_cache", {}),
    ("lapis_pm.pm_core.episodic.spec_summary", dict(return_value="spec")),
    ("lapis_pm.pm_core.episodic.spec", dict(return_value="spec summary")),
    ("lapis_pm.episodic.all_comments", dict(return_value=[])),
    ("lapis_pm.episodic.write_observation", {}),
    ("lapis_pm.pm_core.save_dispatched", {}),
]


@contextlib.contextmanager
def _tick_ctx(records, perceive_result=([], True), extra_patches=None):
    """Context manager that patches tick() internals for lost-dispatch tests."""
    stack = contextlib.ExitStack()
    mocks = {}
    try:
        store_mock = stack.enter_context(patch("lapis_pm.pm_core.TargetStore"))
        store_mock.return_value.get.return_value = _make_lost_target()
        mocks["TargetStore"] = store_mock

        load_mock = stack.enter_context(
            patch("lapis_pm.pm_core.load_dispatched", return_value=records)
        )
        mocks["load_dispatched"] = load_mock

        perceive_mock = stack.enter_context(
            patch("lapis_pm.pm_core._perceive_prs", return_value=perceive_result)
        )
        mocks["_perceive_prs"] = perceive_mock

        for target, kwargs in _TICK_BASE_PATCHES:
            mocks[target] = stack.enter_context(patch(target, **kwargs))

        for target, kwargs in (extra_patches or []):
            mocks[target] = stack.enter_context(patch(target, **kwargs))

        yield mocks
    finally:
        stack.close()


class TestTickLostDispatchDecide:

    def test_tick_retries_lost_dispatch_on_first_loss(self):
        """tick() returns fixer_lost:retrying when a lost dispatch has lost_retry_count=0."""
        orig = _fixer_record(
            gpu_id="gpu-lost-001",
            ts="2026-05-01T10:00:00-07:00",
            status="failed",
        )
        fake_res = MagicMock()
        fake_res.task_id = "gpu-lost-retry-001"
        fake_res.spec_id = "spec-retry"

        extra = [("lapis_pm.pm_core._SHAPER", {})]
        with _tick_ctx([orig], extra_patches=extra) as mocks:
            mocks["lapis_pm.pm_core._SHAPER"].dispatch.return_value = fake_res
            result = pm_core.tick("my-target")

        assert result.decision.startswith("fixer_lost:retrying:dispatch=")

    def test_tick_briefs_on_second_loss(self):
        """tick() returns fixer_lost:briefing when lost_retry_count=1 and child terminal."""
        original = _fixer_record(
            gpu_id="gpu-lost-A",
            ts="2026-05-01T10:00:00-07:00",
            status="failed",
            lost_retry_count=1,
            error="ERROR: SEGV",
        )
        child = _fixer_record(
            gpu_id="gpu-lost-B",
            ts="2026-05-01T10:15:00-07:00",
            status="failed",
            parent_gpu_id="gpu-lost-A",
            error="ERROR: GP fault",
        )
        fake_brief = MagicMock()
        fake_brief.comment_id = "brief-lost-001"
        fake_brief.pushed = True
        fake_brief.body = "brief body"
        fake_brief.target_id = "my-target"

        extra = [
            ("lapis_pm.pm_core.brief.synthesize", dict(return_value=fake_brief)),
            ("lapis_pm.pm_core.set_outstanding_brief", {}),
        ]
        with _tick_ctx([original, child], extra_patches=extra):
            result = pm_core.tick("my-target")

        assert result.decision.startswith("fixer_lost:briefing:dispatches=")
        assert "gpu-lost-A" in result.decision
        assert "gpu-lost-B" in result.decision

    def test_tick_noop_when_forgejo_unreachable(self):
        """tick() returns noop (not retry) when Forgejo was unreachable this tick."""
        orig = _fixer_record(
            gpu_id="gpu-lost-stale",
            ts="2026-05-01T10:00:00-07:00",
            status="failed",
        )
        # forgejo_ok=False: Forgejo unreachable this tick
        with _tick_ctx([orig], perceive_result=([], False)):
            result = pm_core.tick("my-target")

        assert not result.decision.startswith("fixer_lost:")
