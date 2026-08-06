"""Tests for the GW reference leg retirement under the single-slot posture.

lapis-pm-empiricist-repin-slot1-gw-leg-retire-v0: both spec-review reference
legs were pinned to a dead slot 2. The Empiricist (spec_reviewer) is repinned
to slot 1 elsewhere; this file covers the companion change — the GW leg
(_dispatch_gw_reviewer) is declared retired by default rather than left to
skip with "slot2_unavailable" against a `failed` systemd unit.

SPEC_REVIEW_GW_LEG_RETIRED=0 un-retires the leg for a run (e.g. once a live
Slot-2 exists again); the un-retired path itself is exercised by
tests/test_spec_review_gw_leg.py, which sets that override via an autouse
fixture so its pre-existing assertions keep passing unchanged.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from lapis_pm.spec_review import _dispatch_gw_reviewer, _gw_leg_retired


def _dispatch(**kwargs):
    defaults = dict(
        spec_text="# spec",
        synth_target_id="t",
        parsed_target_id="t",
        repo="lapis-pm",
        run_id="run-1",
    )
    defaults.update(kwargs)
    return _dispatch_gw_reviewer(**defaults)


class TestGwLegRetiredHelper:
    def test_defaults_to_retired_when_unset(self, monkeypatch):
        monkeypatch.delenv("SPEC_REVIEW_GW_LEG_RETIRED", raising=False)
        assert _gw_leg_retired() is True

    def test_un_retired_when_set_to_0(self, monkeypatch):
        monkeypatch.setenv("SPEC_REVIEW_GW_LEG_RETIRED", "0")
        assert _gw_leg_retired() is False

    def test_any_other_value_stays_retired(self, monkeypatch):
        monkeypatch.setenv("SPEC_REVIEW_GW_LEG_RETIRED", "false")
        assert _gw_leg_retired() is True


class TestDispatchGwReviewerRetiredByDefault:
    def test_skip_reason_and_tuple_shape(self, monkeypatch):
        monkeypatch.delenv("SPEC_REVIEW_GW_LEG_RETIRED", raising=False)
        monkeypatch.delenv("GW_REVIEW_STUB", raising=False)
        text, transcript, elapsed, skip_reason, provenance = _dispatch()
        assert skip_reason == "gw_leg_retired"
        assert text is None
        assert provenance is None
        assert transcript == []
        assert isinstance(elapsed, float)

    def test_no_probe_issued(self, monkeypatch):
        """swarm_model must never be called — the short-circuit precedes every
        probe and every worktree operation."""
        monkeypatch.delenv("SPEC_REVIEW_GW_LEG_RETIRED", raising=False)
        monkeypatch.delenv("GW_REVIEW_STUB", raising=False)
        with patch("lapis_pm.spec_review.swarm_model") as mock_probe:
            _dispatch()
            mock_probe.assert_not_called()

    def test_no_worktree_command_issued(self, monkeypatch):
        monkeypatch.delenv("SPEC_REVIEW_GW_LEG_RETIRED", raising=False)
        monkeypatch.delenv("GW_REVIEW_STUB", raising=False)
        with patch("lapis_pm.spec_review._create_gw_worktree") as mock_wt:
            _dispatch()
            mock_wt.assert_not_called()


class TestUnRetiredReachesExistingProbePath:
    def test_env_0_reaches_slot2_unavailable(self, monkeypatch):
        monkeypatch.setenv("SPEC_REVIEW_GW_LEG_RETIRED", "0")
        monkeypatch.delenv("GW_REVIEW_STUB", raising=False)
        with patch("lapis_pm.spec_review.swarm_model", return_value=None) as mock_probe:
            text, transcript, elapsed, skip_reason, provenance = _dispatch()
            mock_probe.assert_called_once()
        assert skip_reason == "slot2_unavailable"
        assert text is None
        assert provenance is None


class TestStubTakesPrecedenceOverRetirement:
    def test_stub_wins_even_when_retired(self, monkeypatch):
        monkeypatch.delenv("SPEC_REVIEW_GW_LEG_RETIRED", raising=False)
        monkeypatch.setenv("GW_REVIEW_STUB", "1")
        monkeypatch.setenv("GW_REVIEW_STUB_VERDICT", "clean")
        text, transcript, elapsed, skip_reason, provenance = _dispatch()
        assert text == "clean"
        assert skip_reason == ""
