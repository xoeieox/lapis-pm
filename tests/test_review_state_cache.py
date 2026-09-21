"""Tests for the review-state cache (lapis-pm-review-state-cache spec).

Coverage:
  - Writes cache when _active_review_state returns a dict.
  - mode derivation: hold → fresh-reviewer, otherwise → same-reviewer.
  - budget derivation: advisory→2, hold→4, unknown→2.
  - Real verdict surfaces (fixable/clean issue handling).
  - Deletes cache when _active_review_state returns None.
  - Idempotent delete: key already absent + state None → no error, no key.
  - Tag set is exactly {"lapis-pm", "review-state"}.
  - Payload size < 500 bytes.
  - clear_landed_state deletes cache and reports it in summary.
  - tick() integration: key written when active state, deleted when None.
  - Tick swallows persist errors: tick returns success, writes pm:error observation.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mem():
    """Return a fresh MagicMock acting as MemoryStore."""
    mem = MagicMock()
    mem.get.return_value = None
    return mem


def _make_target(pm_authority: str = "advisory"):
    t = MagicMock()
    t.pm_authority = pm_authority
    t.pm_bound = True
    t.paused = False
    t.pm_repo = ""
    return t


ACTIVE_STATE = {
    "pr_number": 18,
    "cycle": 1,
    "verdict": "pending",
    "issues": 0,
}


# ---------------------------------------------------------------------------
# _persist_review_state_cache: basic write
# ---------------------------------------------------------------------------

class TestPersistReviewStateCacheWrite:

    def test_writes_cache_when_active_state(self):
        """Cache is written when _active_review_state returns a dict."""
        mem = _make_mem()
        target = _make_target("advisory")

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=ACTIVE_STATE),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("my-target", target, [])

        mem.set.assert_called_once()
        key, raw, *_ = mem.set.call_args[0]
        assert key == "pm/review-state/my-target"
        payload = json.loads(raw)
        assert payload["pr_number"] == 18
        assert payload["cycle"] == 1
        assert payload["budget"] == 3
        assert payload["mode"] == "same-reviewer"
        assert payload["last_verdict"] is None   # verdict=pending → None
        assert payload["last_issues"] is None
        assert "updated_at" in payload

    def test_all_nine_fields_present(self):
        """Payload must have exactly the nine schema fields (seven base +
        last_corroboration + panel_starvation). The former kill-switch
        ``paused`` field was removed with the review-gate counter
        (lapis-pm-review-gate-counter-removal-v0, spec D2/D4 rev 3)."""
        mem = _make_mem()
        target = _make_target("advisory")
        expected_fields = {
            "pr_number", "cycle", "budget", "mode",
            "last_verdict", "last_issues", "updated_at",
            "last_corroboration", "panel_starvation",
        }

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=ACTIVE_STATE),
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=None),
            ):
            pm_core._persist_review_state_cache("my-target", target, [])

        raw = mem.set.call_args[0][1]
        payload = json.loads(raw)
        assert set(payload.keys()) == expected_fields


# ---------------------------------------------------------------------------
# mode derivation
# ---------------------------------------------------------------------------

class TestModeDerivation:

    def test_hold_authority_yields_fresh_reviewer(self):
        mem = _make_mem()
        target = _make_target("hold")

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=ACTIVE_STATE),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("tid", target, [])

        payload = json.loads(mem.set.call_args[0][1])
        assert payload["mode"] == "fresh-reviewer"

    def test_advisory_authority_yields_same_reviewer(self):
        mem = _make_mem()
        target = _make_target("advisory")

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=ACTIVE_STATE),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("tid", target, [])

        payload = json.loads(mem.set.call_args[0][1])
        assert payload["mode"] == "same-reviewer"


# ---------------------------------------------------------------------------
# budget derivation
# ---------------------------------------------------------------------------

class TestBudgetDerivation:

    @pytest.mark.parametrize("authority,expected_budget", [
        ("advisory", 3),   # cr-bundle item df1ac58425: 2 -> 3 (loud brief on exhaustion)
        ("hold", 4),
        ("unknown-authority", 3),  # falls back to default 3
    ])
    def test_budget_per_authority(self, authority, expected_budget):
        mem = _make_mem()
        target = _make_target(authority)

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=ACTIVE_STATE),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("tid", target, [])

        payload = json.loads(mem.set.call_args[0][1])
        assert payload["budget"] == expected_budget


# ---------------------------------------------------------------------------
# Real verdict surfacing
# ---------------------------------------------------------------------------

class TestVerdictSurfacing:

    def test_fixable_verdict_surfaces_verdict_and_issues(self):
        """verdict=fixable, issues=3 → last_verdict=fixable, last_issues=3."""
        mem = _make_mem()
        target = _make_target("advisory")
        state = {"pr_number": 18, "cycle": 2, "verdict": "fixable", "issues": 3}

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=state),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("tid", target, [])

        payload = json.loads(mem.set.call_args[0][1])
        assert payload["last_verdict"] == "fixable"
        assert payload["last_issues"] == 3

    def test_clean_verdict_last_issues_is_none(self):
        """verdict=clean, issues=0 → last_verdict=clean, last_issues=None."""
        mem = _make_mem()
        target = _make_target("advisory")
        state = {"pr_number": 18, "cycle": 1, "verdict": "clean", "issues": 0}

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=state),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("tid", target, [])

        payload = json.loads(mem.set.call_args[0][1])
        assert payload["last_verdict"] == "clean"
        assert payload["last_issues"] is None

    def test_pending_verdict_yields_null_last_verdict(self):
        """verdict=pending → last_verdict=None, last_issues=None."""
        mem = _make_mem()
        target = _make_target("advisory")
        state = {"pr_number": 18, "cycle": 0, "verdict": "pending", "issues": 0}

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=state),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("tid", target, [])

        payload = json.loads(mem.set.call_args[0][1])
        assert payload["last_verdict"] is None
        assert payload["last_issues"] is None


# ---------------------------------------------------------------------------
# Delete when no active review
# ---------------------------------------------------------------------------

class TestDeleteOnNoActiveReview:

    def test_deletes_cache_when_state_is_none(self):
        """When _active_review_state returns None, key is deleted."""
        mem = _make_mem()
        mem.get.return_value = {"content": '{"pr_number": 18}'}  # pre-seeded
        target = _make_target("advisory")

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=None),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("tid", target, [])

        mem.delete.assert_called_with("pm/review-state/tid")
        mem.set.assert_not_called()

    def test_idempotent_delete_no_error_when_key_absent(self):
        """Key already absent + state None → no error, no crash."""
        mem = _make_mem()
        mem.get.return_value = None
        target = _make_target("advisory")

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=None),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            # Must not raise
            pm_core._persist_review_state_cache("tid", target, [])

        mem.delete.assert_called_with("pm/review-state/tid")


# ---------------------------------------------------------------------------
# Tag set
# ---------------------------------------------------------------------------

class TestTagSet:

    def test_tag_set_is_exactly_lapis_pm_and_review_state(self):
        """Tags must be exactly ["lapis-pm", "review-state"]."""
        mem = _make_mem()
        target = _make_target("advisory")

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=ACTIVE_STATE),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("tid", target, [])

        # mem.set called as mem.set(key, value, tags=[...])
        call_kwargs = mem.set.call_args
        tags = call_kwargs[1].get("tags") or call_kwargs[0][2]
        assert set(tags) == {"lapis-pm", "review-state"}


# ---------------------------------------------------------------------------
# Payload size
# ---------------------------------------------------------------------------

class TestPayloadSize:

    def test_payload_under_500_bytes(self):
        """Payload must be < 500 bytes serialized."""
        mem = _make_mem()
        target = _make_target("advisory")
        captured = {}

        def capture_set(key, value, tags=None):
            captured["value"] = value

        mem.set.side_effect = capture_set

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=ACTIVE_STATE),
            patch("lapis_pm.pm_core._mem", return_value=mem),
        ):
            pm_core._persist_review_state_cache("tid", target, [])

        assert len(captured["value"]) < 500


# ---------------------------------------------------------------------------
# clear_landed_state integration
# ---------------------------------------------------------------------------

class TestClearLandedState:

    def test_clears_review_state_cache_and_reports_1(self):
        """Pre-seed key → clear_landed_state deletes it and reports review_state==1."""
        mem = _make_mem()
        # First call to mem.get for review_state key returns content (pre-seeded)
        # We need to simulate mem.get returning a record for _review_state_key

        def selective_get(key):
            if key == pm_core._review_state_key("my-target"):
                return {"content": '{"pr_number": 18}'}
            return None

        mem.get.side_effect = selective_get

        with (
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core.clear_outstanding_brief"),
            patch("lapis_pm.pm_core._classified_pr_ids", return_value=[]),
            patch("lapis_pm.pm_core.clear_classified_prs"),
            patch("lapis_pm.pm_core.get_cursor", return_value=None),
            patch("lapis_pm.pm_core.get_pause_state", return_value=None),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("lapis_pm.pm_core.save_dispatched"),
            ):
            summary = pm_core.clear_landed_state("my-target")

        assert summary["review_state"] == 1
        mem.delete.assert_any_call(pm_core._review_state_key("my-target"))

    def test_idempotent_reports_0_when_key_absent(self):
        """No pre-seeded key → summary["review_state"] == 0, no error."""
        mem = _make_mem()
        mem.get.return_value = None  # all keys absent

        with (
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core.clear_outstanding_brief"),
            patch("lapis_pm.pm_core._classified_pr_ids", return_value=[]),
            patch("lapis_pm.pm_core.clear_classified_prs"),
            patch("lapis_pm.pm_core.get_cursor", return_value=None),
            patch("lapis_pm.pm_core.get_pause_state", return_value=None),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("lapis_pm.pm_core.save_dispatched"),
            ):
            summary = pm_core.clear_landed_state("my-target")

        assert summary["review_state"] == 0


# ---------------------------------------------------------------------------
# tick() integration
# ---------------------------------------------------------------------------

class TestTickIntegration:

    def _make_mock_target(self, pm_authority="advisory"):
        from agents_core.targets import Target
        t = MagicMock(spec=Target)
        t.pm_bound = True
        t.paused = False
        t.pm_repo = ""
        t.pm_authority = pm_authority
        t.data = {}
        return t

    def test_tick_writes_key_when_active_state(self):
        """tick() writes pm/review-state/<tid> when _active_review_state returns non-None."""
        mem = _make_mem()
        mock_target = self._make_mock_target()

        with (
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core._active_review_state", return_value=ACTIVE_STATE),            patch("lapis_pm.pm_core._perceive_prs", return_value=([], True)),
            patch("lapis_pm.pm_core.get_cursor", return_value=None),
            patch("lapis_pm.pm_core.set_cursor"),
            patch("lapis_pm.pm_core.get_pause_state", return_value=None),
            patch("lapis_pm.pm_core.set_pause_state"),
            patch("lapis_pm.pm_core._reconcile_dispatched_with_queue", return_value=0),
            patch("lapis_pm.episodic.since", return_value=[]),
            patch("lapis_pm.pm_core._encode_new_prs", return_value=[]),
            patch("lapis_pm.pm_core._encode_pr_sha_updates", return_value=0),
            patch("lapis_pm.pm_core._encode_pr_body_updates", return_value=0),
            patch("lapis_pm.pm_core._encode_gpu_results", return_value=(0, [])),
            patch("lapis_pm.pm_core._encode_merged_prs", return_value=0),
            patch("lapis_pm.pm_core._encode_user_comments", return_value=[]),
            patch("lapis_pm.pm_core._is_auto_land_eligible", return_value=False),
            ):
            MockStore.return_value.get.return_value = mock_target
            result = pm_core.tick("my-target")

        assert result.skipped is False
        # mem.set should have been called for the review-state key
        set_keys = [c[0][0] for c in mem.set.call_args_list]
        assert "pm/review-state/my-target" in set_keys

    def test_tick_deletes_key_when_no_active_state(self):
        """tick() deletes pm/review-state/<tid> when _active_review_state returns None."""
        mem = _make_mem()
        mock_target = self._make_mock_target()

        with (
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core._active_review_state", return_value=None),
            patch("lapis_pm.pm_core._perceive_prs", return_value=([], True)),
            patch("lapis_pm.pm_core.get_cursor", return_value=None),
            patch("lapis_pm.pm_core.set_cursor"),
            patch("lapis_pm.pm_core.get_pause_state", return_value=None),
            patch("lapis_pm.pm_core.set_pause_state"),
            patch("lapis_pm.pm_core._reconcile_dispatched_with_queue", return_value=0),
            patch("lapis_pm.episodic.since", return_value=[]),
            patch("lapis_pm.pm_core._encode_new_prs", return_value=[]),
            patch("lapis_pm.pm_core._encode_pr_sha_updates", return_value=0),
            patch("lapis_pm.pm_core._encode_pr_body_updates", return_value=0),
            patch("lapis_pm.pm_core._encode_gpu_results", return_value=(0, [])),
            patch("lapis_pm.pm_core._encode_merged_prs", return_value=0),
            patch("lapis_pm.pm_core._encode_user_comments", return_value=[]),
            patch("lapis_pm.pm_core._is_auto_land_eligible", return_value=False),
            ):
            MockStore.return_value.get.return_value = mock_target
            result = pm_core.tick("my-target")

        assert result.skipped is False
        delete_keys = [c[0][0] for c in mem.delete.call_args_list]
        assert "pm/review-state/my-target" in delete_keys


# ---------------------------------------------------------------------------
# Tick swallows persist errors
# ---------------------------------------------------------------------------

class TestTickSwallowsPersistErrors:

    def test_tick_continues_when_persist_raises(self):
        """tick() must not fail when _persist_review_state_cache raises."""
        from agents_core.targets import Target
        mock_target = MagicMock(spec=Target)
        mock_target.pm_bound = True
        mock_target.paused = False
        mock_target.pm_repo = ""
        mock_target.pm_authority = "advisory"
        mock_target.data = {}

        observations = []

        def capture_obs(tid, content, extra_tags=None):
            observations.append((content, extra_tags or []))

        with (
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core._persist_review_state_cache",
                  side_effect=RuntimeError("boom")),
            patch("lapis_pm.pm_core._perceive_prs", return_value=([], True)),
            patch("lapis_pm.pm_core.get_cursor", return_value=None),
            patch("lapis_pm.pm_core.set_cursor"),
            patch("lapis_pm.pm_core.get_pause_state", return_value=None),
            patch("lapis_pm.pm_core.set_pause_state"),
            patch("lapis_pm.pm_core._reconcile_dispatched_with_queue", return_value=0),
            patch("lapis_pm.episodic.since", return_value=[]),
            patch("lapis_pm.pm_core._encode_new_prs", return_value=[]),
            patch("lapis_pm.pm_core._encode_pr_sha_updates", return_value=0),
            patch("lapis_pm.pm_core._encode_pr_body_updates", return_value=0),
            patch("lapis_pm.pm_core._encode_gpu_results", return_value=(0, [])),
            patch("lapis_pm.pm_core._encode_merged_prs", return_value=0),
            patch("lapis_pm.pm_core._encode_user_comments", return_value=[]),
            patch("lapis_pm.pm_core._is_auto_land_eligible", return_value=False),
            patch("lapis_pm.episodic.write_observation", side_effect=capture_obs),
        ):
            MockStore.return_value.get.return_value = mock_target
            result = pm_core.tick("my-target")

        # Tick must succeed
        assert result.skipped is False
        assert result.decision == "noop:no_change"

        # An pm:error observation must have been written
        error_obs = [(c, t) for c, t in observations if "pm:error" in t]
        assert error_obs, "Expected a pm:error observation to be written"
        assert any("review-state cache write failed" in c for c, _ in error_obs)
