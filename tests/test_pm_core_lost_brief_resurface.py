"""Unit tests for the lost-dispatch brief resurface
(lapis-pm-lost-brief-must-surface-v0).

The defect: _act_lost_brief's idempotency guard found a prior
pm:brief-options comment (trigger=lost-dispatch) and, when the
outstanding-brief mem key was absent or stale, suppressed the brief
FOREVER — even when the brief had never been delivered to the operator
(2026-08-28 case: the comment was written but the key was never set).

The fix: that branch now re-emits the brief through the normal compose/
surface path (bounded to one resurface attempt per 24h per orig_gpu_id,
the attempt ts recorded in an episodic observation), instead of writing a
pm:lost-brief-suppressed observation and suppressing.

Test contract (spec §Tests):
  (a) regression: brief-options present + key matches -> suppress,
      no re-composition, no new write.
  (b) the fix: brief-options present + key absent (None) -> resurface
      (brief composed + surfaced + key-set called with the brief id).
  (c) regression: the pm:lost-brief-suppressed observation is written at
      most once per dispatch (sibling unbounded-write fix — must not
      regress).
  (d) the bound: two ticks < 24h apart with a failing key-set -> exactly
      one resurface attempt; the attempt is recorded.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixer_record(gpu_id: str = "gpu-resurf-001", status: str = "failed") -> dict:
    return {
        "gpu_id": gpu_id,
        "spec_id": f"spec-{gpu_id}",
        "agent_type": "fixer",
        "intent": "implement the spec",
        "repo": "lapis-pm",
        "ts": "2026-09-07T10:00:00-07:00",
        "status": status,
        "retry_count": 0,
        "lost_retry_count": 1,
    }


def _options_comment(brief_id: str, orig_gpu_id: str, ts: str = "2026-09-07T11:00:00-07:00") -> MagicMock:
    """A pm:brief-options comment as brief.synthesize would write it."""
    payload = {
        "brief_id": brief_id,
        "trigger": "lost-dispatch",
        "options": [
            {"id": "A", "label": "Retry again",
             "action": {"kind": "force_dispatch_retry"}},
        ],
    }
    c = MagicMock()
    c.tags = ["pm:brief-options", f"pm:lost-original-gpu={orig_gpu_id}"]
    c.ts = ts
    c.content = json.dumps(payload)
    return c


def _resurface_attempt_comment(
    orig_gpu_id: str, ts: str, failed: bool = False
) -> MagicMock:
    """The episodic attempt record the resurface path writes."""
    tag = "pm:lost-brief-resurface-failed" if failed else "pm:lost-brief-resurface"
    c = MagicMock()
    c.tags = ["pm:observation", tag, f"pm:lost-original-gpu={orig_gpu_id}"]
    c.ts = ts
    c.content = f"Lost-brief resurface attempt: orig_gpu={orig_gpu_id}"
    return c


def _fake_brief(comment_id: str = "brief-resurf-001") -> MagicMock:
    b = MagicMock()
    b.comment_id = comment_id
    b.pushed = True
    b.synthesis_failed = False
    b.body = "lost dispatch brief body"
    b.target_id = "my-target"
    return b


# ---------------------------------------------------------------------------
# (a) Regression: brief-options present + key matches -> suppress
# ---------------------------------------------------------------------------

class TestMatchBranchSuppress:

    def test_matching_key_suppresses_no_recomposition_no_write(self):
        """brief-options present + outstanding key matches the brief id:
        suppress entirely — no brief.synthesize, no observation, no key-set."""
        orig = _fixer_record(gpu_id="gpu-match-001")
        brief_id = "brief-match-001"
        options = _options_comment(brief_id, "gpu-match-001")

        with (
            patch("lapis_pm.episodic.all_comments", return_value=[options]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=brief_id),
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
            patch("lapis_pm.episodic.write_observation") as mock_obs,
            patch("lapis_pm.pm_core.set_outstanding_brief") as mock_set,
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
        ):
            result = pm_core._act_lost_brief("my-target", orig, None)

        assert result == "noop:lost-brief-suppressed:gpu=gpu-match-001"
        mock_synth.assert_not_called()
        mock_obs.assert_not_called()
        mock_set.assert_not_called()


# ---------------------------------------------------------------------------
# (b) The fix: brief-options present + key absent (None) -> resurface
# ---------------------------------------------------------------------------

class TestResurfaceOnAbsentKey:

    def test_absent_key_resurfaces_through_normal_path(self):
        """The 2026-08-28 case: pm:brief-options comment exists but the
        outstanding-brief key was never set.  The guard must NOT suppress —
        it re-emits the brief through the normal compose/surface path
        (brief.synthesize with trigger=lost-dispatch + the gpu tag, then
        key-set called with the NEW brief id)."""
        orig = _fixer_record(gpu_id="gpu-absent-001")
        old_brief_id = "brief-0828-001"
        options = _options_comment(old_brief_id, "gpu-absent-001")

        synth_kwargs = {}

        def capture_synth(tid, trigger, query="", notify=None, **kw):
            synth_kwargs.update(trigger=trigger, query=query, notify=notify, **kw)
            return _fake_brief("brief-resurf-new-001")

        key_set_calls: list = []

        def capture_set(tid, b):
            key_set_calls.append((tid, b.comment_id))
            return True

        written_obs: list = []

        def capture_obs(tid, content, extra_tags=None):
            written_obs.append((content, extra_tags or []))
            return MagicMock()

        with (
            patch("lapis_pm.episodic.all_comments", return_value=[options]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core.brief.synthesize", side_effect=capture_synth),
            patch("lapis_pm.pm_core._set_brief_outstanding", side_effect=capture_set),
            patch("lapis_pm.episodic.write_observation", side_effect=capture_obs),
            patch("lapis_pm.pm_core.episodic.spec", return_value="spec body"),
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
        ):
            result = pm_core._act_lost_brief("my-target", orig, None)

        # Brief was composed and surfaced (not suppressed).
        assert result == "fixer_lost:briefing:dispatches=gpu-absent-001"
        assert synth_kwargs["trigger"] == "lost-dispatch"
        assert synth_kwargs["options_extra_tags"] == ["pm:lost-original-gpu=gpu-absent-001"]
        # Key-set called with the NEW brief id.
        assert key_set_calls == [("my-target", "brief-resurf-new-001")]
        # The resurface attempt was recorded in an episodic observation
        # (the 24h bound record), tagged with the gpu id.
        assert len(written_obs) == 1
        content, tags = written_obs[0]
        assert "pm:lost-brief-resurface" in tags
        assert "pm:lost-original-gpu=gpu-absent-001" in tags
        assert "gpu-absent-001" in content
        assert "attempt_ts=" in content
        # The old suppression observation is NOT written on this path.
        assert "pm:lost-brief-suppressed" not in tags

    def test_stale_key_also_resurfaces(self):
        """Key present but pointing at a different brief id (stale) is the
        same indistinguishable case — resurface, not suppress."""
        orig = _fixer_record(gpu_id="gpu-stale-001")
        options = _options_comment("brief-stale-old", "gpu-stale-001")

        with (
            patch("lapis_pm.episodic.all_comments", return_value=[options]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value="brief-someone-else"),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=_fake_brief("brief-stale-new")),
            patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set,
            patch("lapis_pm.episodic.write_observation"),
            patch("lapis_pm.pm_core.episodic.spec", return_value="spec body"),
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
        ):
            result = pm_core._act_lost_brief("my-target", orig, None)

        assert result == "fixer_lost:briefing:dispatches=gpu-stale-001"
        assert mock_set.call_count == 1
        assert mock_set.call_args[0][1].comment_id == "brief-stale-new"


# ---------------------------------------------------------------------------
# (c) Regression: pm:lost-brief-suppressed observation written at most once
#     per dispatch (sibling unbounded-write fix must not regress)
# ---------------------------------------------------------------------------

class TestSuppressedObservationWrittenOnce:

    def test_prior_suppressed_observation_still_suppresses(self):
        """A prior pm:lost-brief-suppressed observation for the same
        orig_gpu short-circuits the guard: no resurface, no new write."""
        orig = _fixer_record(gpu_id="gpu-once-001")
        options = _options_comment("brief-once-001", "gpu-once-001")
        prior_obs = MagicMock()
        prior_obs.tags = ["pm:observation", "pm:lost-brief-suppressed",
                          "pm:lost-original-gpu=gpu-once-001"]
        prior_obs.ts = "2026-09-07T11:05:00-07:00"
        prior_obs.content = "Lost-brief suppressed (mem-stale): ..."

        with (
            patch("lapis_pm.episodic.all_comments", return_value=[options, prior_obs]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
            patch("lapis_pm.episodic.write_observation") as mock_obs,
            patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set,
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
        ):
            result = pm_core._act_lost_brief("my-target", orig, None)

        assert result == "noop:lost-brief-suppressed:gpu=gpu-once-001"
        mock_synth.assert_not_called()
        mock_obs.assert_not_called()  # no second observation
        mock_set.assert_not_called()

    def test_second_tick_after_successful_resurface_suppresses_no_new_write(self):
        """End-to-end shape of the fix: tick 1 resurfaces (key set to the new
        brief id); tick 2 sees the matching key -> match-branch suppress,
        no re-composition, no new write."""
        orig = _fixer_record(gpu_id="gpu-twotick-001")
        options = _options_comment("brief-twotick-001", "gpu-twotick-001")

        # Tick 1: key absent -> resurface.
        with (
            patch("lapis_pm.episodic.all_comments", return_value=[options]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core.brief.synthesize",
                  return_value=_fake_brief("brief-twotick-001")),
            patch("lapis_pm.pm_core._set_brief_outstanding"),
            patch("lapis_pm.episodic.write_observation"),
            patch("lapis_pm.pm_core.episodic.spec", return_value="spec"),
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
        ):
            r1 = pm_core._act_lost_brief("my-target", orig, None)
        assert r1 == "fixer_lost:briefing:dispatches=gpu-twotick-001"

        # Tick 2: key now matches the brief id in the comment -> suppress.
        with (
            patch("lapis_pm.episodic.all_comments", return_value=[options]),
            patch("lapis_pm.pm_core.get_outstanding_brief",
                  return_value="brief-twotick-001"),
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
            patch("lapis_pm.episodic.write_observation") as mock_obs,
            patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set,
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
        ):
            r2 = pm_core._act_lost_brief("my-target", orig, None)

        assert r2 == "noop:lost-brief-suppressed:gpu=gpu-twotick-001"
        mock_synth.assert_not_called()
        mock_obs.assert_not_called()
        mock_set.assert_not_called()


# ---------------------------------------------------------------------------
# (d) The bound: two ticks < 24h apart with a failing key-set -> exactly one
#     resurface attempt; the attempt is recorded
# ---------------------------------------------------------------------------

class TestResurfaceBound:

    def test_failing_key_set_bounds_to_one_attempt_per_24h(self):
        """Tick 1: resurface attempt recorded, then key-set raises.
        Tick 2 (same tick, key still absent): the recorded attempt ts is
        < 24h old -> suppress, NO second attempt, NO second synthesize."""
        orig = _fixer_record(gpu_id="gpu-bound-001")
        options = _options_comment("brief-bound-001", "gpu-bound-001")
        tick_ts = "2026-09-07T11:00:00-07:00"

        synth_calls = [0]

        def capture_synth(tid, trigger, query="", notify=None, **kw):
            synth_calls[0] += 1
            return _fake_brief("brief-bound-new")

        def failing_set(tid, b):
            raise RuntimeError("mem write failed")

        # --- Tick 1 ---
        with (
            patch("lapis_pm.episodic.all_comments", return_value=[options]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core.brief.synthesize", side_effect=capture_synth),
            patch("lapis_pm.pm_core._set_brief_outstanding",
                  side_effect=failing_set),
            patch("lapis_pm.episodic.write_observation") as mock_obs1,
            patch("lapis_pm.pm_core.episodic.spec", return_value="spec"),
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
        ):
            with pytest.raises(RuntimeError, match="mem write failed"):
                pm_core._act_lost_brief("my-target", orig, None)

        # Tick 1 recorded the attempt observation (tag carries the gpu id),
        # and the attempt ts embedded in its content is what the 24h bound
        # reads back on the next tick.
        attempt_calls = [
            args for args in mock_obs1.call_args_list
            if "pm:lost-brief-resurface" in (args.kwargs.get("extra_tags") or [])
            and "pm:lost-brief-resurface-failed" not in (args.kwargs.get("extra_tags") or [])
        ]
        assert len(attempt_calls) == 1, "exactly one attempt observation on tick 1"
        attempt_content = attempt_calls[0].args[1]
        assert "pm:lost-original-gpu=gpu-bound-001" in attempt_calls[0].kwargs["extra_tags"]
        assert "attempt_ts=" in attempt_content
        # The key-set failure also produced a loud (visible) observation.
        failed_calls = [
            args for args in mock_obs1.call_args_list
            if "pm:lost-brief-resurface-failed" in (args.kwargs.get("extra_tags") or [])
        ]
        assert len(failed_calls) == 1
        assert synth_calls[0] == 1

        # --- Tick 2: same tick, key still absent, attempt ts < 24h old ---
        # Reconstruct the attempt comment the episodic store would now hold,
        # using the ts the code actually embedded in the attempt content.
        embedded_ts = attempt_content.split("attempt_ts=")[1].strip()
        attempt_obs = _resurface_attempt_comment("gpu-bound-001", embedded_ts)
        with (
            patch("lapis_pm.episodic.all_comments",
                  return_value=[options, attempt_obs]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core.brief.synthesize", side_effect=capture_synth),
            patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set2,
            patch("lapis_pm.episodic.write_observation") as mock_obs2,
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
        ):
            result = pm_core._act_lost_brief("my-target", orig, None)

        # Bounded: suppressed, no second resurface attempt.
        assert result == "noop:lost-brief-suppressed:gpu=gpu-bound-001"
        assert synth_calls[0] == 1, "no second composition within the 24h window"
        mock_set2.assert_not_called()
        mock_obs2.assert_not_called()  # no new write on the bounded tick

    def test_attempt_older_than_24h_allows_second_resurface(self):
        """The bound is per-24h, not one-forever: an attempt ts > 24h old
        allows a fresh resurface attempt."""
        orig = _fixer_record(gpu_id="gpu-bound2-001")
        options = _options_comment("brief-bound2-001", "gpu-bound2-001")
        old_attempt_ts = "2026-09-06T10:00:00-07:00"  # > 24h before now
        attempt_obs = _resurface_attempt_comment("gpu-bound2-001", old_attempt_ts)

        with (
            patch("lapis_pm.episodic.all_comments",
                  return_value=[options, attempt_obs]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core.brief.synthesize",
                  return_value=_fake_brief("brief-bound2-new")) as mock_synth,
            patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set,
            patch("lapis_pm.episodic.write_observation"),
            patch("lapis_pm.pm_core.episodic.spec", return_value="spec"),
            patch("lapis_pm.pm_core._close_slot_and_deposit"),
        ):
            result = pm_core._act_lost_brief("my-target", orig, None)

        assert result == "fixer_lost:briefing:dispatches=gpu-bound2-001"
        mock_synth.assert_called_once()
        assert mock_set.call_count == 1

    def test_helper_window_math(self):
        """Direct check of the window predicate (naive `now` is the
        production shape — the predicate normalizes a naive attempt ts to
        UTC so naive/aware subtraction never raises)."""
        now = datetime(2026, 9, 7, 11, 0, 0)
        recent = now - timedelta(hours=1)
        old = now - timedelta(hours=25)
        with (
            patch("lapis_pm.episodic.all_comments", return_value=[]),
        ):
            assert pm_core._lost_brief_resurface_allowed("t", "g", now) is True
        with (
            patch("lapis_pm.episodic.all_comments",
                  return_value=[_resurface_attempt_comment("g", recent.isoformat())]),
        ):
            assert pm_core._lost_brief_resurface_allowed("t", "g", now) is False
        with (
            patch("lapis_pm.episodic.all_comments",
                  return_value=[_resurface_attempt_comment("g", old.isoformat())]),
        ):
            assert pm_core._lost_brief_resurface_allowed("t", "g", now) is True
        # A different orig_gpu on the same target does not block this one.
        with (
            patch("lapis_pm.episodic.all_comments",
                  return_value=[_resurface_attempt_comment("other-gpu", recent.isoformat())]),
        ):
            assert pm_core._lost_brief_resurface_allowed("t", "g", now) is True
