"""Tests for lapis-pm-auto-brief-resolve-v0 — conservative auto-resolve.

Coverage:
  1. test_autoresolve_happy: advisory + clean + no-issues + pr-number + no-held +
     dial=conservative => merge executed; no outstanding brief set; no gem deposited;
     pm:auto-resolved observation written.
  2. test_autoresolve_hold_authority_gems: pm_authority="hold" => predicate False;
     normal gem/brief path.
  3. test_autoresolve_reviewer_not_clean_gems: screen_verdict="fixable" => predicate
     False; normal gem/brief path.
  4. test_autoresolve_held_path_gems: _held_path_check_live blocked => predicate False;
     live gate consulted, not a cached one.
  5. test_autoresolve_no_merge_option_gems: advisory-clean template patched to have no
     merge_pr option => predicate False.
  6. test_autoresolve_dial_off: PM_AUTO_RESOLVE=off => always predicate False; unknown
     value also treated as off.
  7. test_autoresolve_apply_decision_failure_falls_through: merge raises => brief/gem
     path still runs; no auto-resolved observation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest

from lapis_pm import auto_resolve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cls(
    screen_verdict: str = "clean",
    issues: list | None = None,
    pr_number: int = 42,
    repo: str = "lapis-pm",
) -> MagicMock:
    cls = MagicMock()
    cls.screen_verdict = screen_verdict
    cls.issues = issues if issues is not None else []
    cls.pr_number = pr_number
    cls.repo = repo
    cls.title = "feat: add thing"
    cls.html_url = "http://forgejo/pr/42"
    cls.diff = ""
    cls.diff_loc = 10
    cls.reasons = []
    return cls


def _make_target(
    target_id: str = "my-target",
    pm_authority: str = "advisory",
    pm_repo: str = "lapis-pm",
    pm_verification: str = "machine",
) -> MagicMock:
    t = MagicMock()
    t.id = target_id
    t.pm_authority = pm_authority
    t.pm_repo = pm_repo
    t.data = {"pm_verification": pm_verification}
    return t


# ---------------------------------------------------------------------------
# 1. Happy path — full predicate satisfied
# ---------------------------------------------------------------------------

class TestAutoResolvePredicate:

    def test_happy_all_clauses(self):
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="advisory")

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.auto_resolve.brief_gem") as mock_bg,
        ):
            mock_bg._held_path_check_live.return_value = (False, "")
            ok, opt = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is True
        assert opt == "A"  # advisory-clean template merge_pr option id

    def test_hold_authority_returns_false(self):
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="hold")

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "pm_authority" in reason

    def test_screen_verdict_fixable_returns_false(self):
        cls = _make_cls(screen_verdict="fixable", issues=[])
        target = _make_target(pm_authority="advisory")

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "screen_verdict" in reason

    def test_screen_verdict_none_returns_false(self):
        """Absent/None screen_verdict must fail-safe to gem."""
        cls = _make_cls(screen_verdict=None, issues=[])
        target = _make_target(pm_authority="advisory")

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "screen_verdict" in reason

    def test_issues_present_returns_false(self):
        cls = _make_cls(screen_verdict="clean", issues=[{"severity": "MEDIUM", "path": "x.py", "note": "foo"}])
        target = _make_target(pm_authority="advisory")

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "issues" in reason

    def test_held_path_blocks(self):
        """_held_path_check_live blocking returns False (live gate consulted)."""
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="advisory")

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.auto_resolve.brief_gem") as mock_bg,
        ):
            mock_bg._held_path_check_live.return_value = (True, "held path: room/targets/x.yaml")
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "held_path" in reason
        # Confirm the live gate was actually called
        mock_bg._held_path_check_live.assert_called_once_with("my-target", 42, "lapis-pm")

    def test_dial_off_returns_false(self):
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="advisory")

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "off"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "dial" in reason

    def test_dial_unknown_treated_as_off(self):
        """Unknown dial values fail-safe to off."""
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="advisory")

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "moderate"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "dial" in reason

    def test_no_merge_pr_option_in_template(self):
        """If advisory-clean template has no merge_pr option, predicate fails."""
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="advisory")

        patched_triggers = {
            "advisory-clean": [
                {"id": "B", "label": "Acknowledge", "action": {"kind": "acknowledge_and_clear"}},
            ]
        }

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.auto_resolve.brief_gem") as mock_bg,
            patch("lapis_pm.auto_resolve._CLOSED_FORM_TRIGGERS", patched_triggers),
        ):
            mock_bg._held_path_check_live.return_value = (False, "")
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "no merge_pr option" in reason


# ---------------------------------------------------------------------------
# 2. _act_brief integration — hook in pm_core
# ---------------------------------------------------------------------------

class TestActBriefAutoResolveHook:
    """Test that _act_brief invokes the auto-resolve path or falls through."""

    _BASE_PATCHES = [
        ("lapis_pm.pm_core.episodic.write_observation", MagicMock()),
        ("lapis_pm.pm_core.episodic.write_hold", MagicMock()),
        ("lapis_pm.pm_core.episodic.write_merge", MagicMock()),
        ("lapis_pm.pm_core._mark_pr_classified", MagicMock()),
    ]

    def _make_payload(self, cls):
        return {"classification": cls}

    def test_happy_auto_resolves_no_brief_no_gem(self):
        """advisory + clean + conservative => merge called; no brief synthesized; no gem."""
        from lapis_pm import pm_core

        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target()
        payload = self._make_payload(cls)

        mock_mem = MagicMock()
        mock_mem.get.return_value = None
        mock_merge = MagicMock()
        mock_obs = MagicMock()
        mock_synthesize = MagicMock()
        mock_set_outstanding = MagicMock()

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.pm_core.TargetStore") as mock_ts,
            patch("lapis_pm.auto_resolve.brief_gem") as mock_bg,
            patch("lapis_pm.pm_core.brief._act_merge_pr", mock_merge),
            patch("lapis_pm.pm_core.episodic.write_observation", mock_obs),
            patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.brief.synthesize", mock_synthesize),
            patch("lapis_pm.pm_core._set_brief_outstanding", mock_set_outstanding),
            patch("lapis_pm.pm_core._mark_pr_classified"),
        ):
            mock_ts.return_value.get.return_value = target
            mock_bg._held_path_check_live.return_value = (False, "")

            result = pm_core._act_brief("my-target", "advisory-clean", False, payload)

        assert result == "action:auto_resolved:advisory_clean:pr=42"
        mock_merge.assert_called_once_with("my-target", 42)
        mock_synthesize.assert_not_called()
        mock_set_outstanding.assert_not_called()
        # Observation with pm:auto-resolved tag was written
        obs_calls = [str(c) for c in mock_obs.call_args_list]
        assert any("auto-resolved" in s.lower() or "pm:auto-resolved" in s for s in obs_calls)

    def test_hold_authority_produces_brief_not_auto_resolve(self):
        """pm_authority=hold => skip auto-resolve; normal brief/gem path."""
        from lapis_pm import pm_core

        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="hold")
        payload = self._make_payload(cls)

        mock_b = MagicMock()
        mock_b.comment_id = "brief-001"
        mock_b.synthesis_failed = False
        mock_merge = MagicMock()
        mock_set_outstanding = MagicMock()

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.pm_core.TargetStore") as mock_ts,
            patch("lapis_pm.pm_core.brief._act_merge_pr", mock_merge),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_b),
            patch("lapis_pm.pm_core._set_brief_outstanding", mock_set_outstanding),
            patch("lapis_pm.pm_core.episodic.write_hold"),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core._mem", return_value=MagicMock()),
            patch("lapis_pm.pm_core._mark_pr_classified"),
        ):
            mock_ts.return_value.get.return_value = target

            result = pm_core._act_brief("my-target", "advisory", True, payload)

        assert "brief_emitted" in result
        mock_merge.assert_not_called()
        mock_set_outstanding.assert_called_once()

    def test_not_clean_produces_brief_not_auto_resolve(self):
        """screen_verdict=fixable => no auto-resolve; normal brief/gem path."""
        from lapis_pm import pm_core

        cls = _make_cls(screen_verdict="fixable", issues=[{"severity": "HIGH", "path": "x.py", "note": "bad"}])
        target = _make_target(pm_authority="advisory")
        payload = self._make_payload(cls)

        mock_b = MagicMock()
        mock_b.comment_id = "brief-001"
        mock_b.synthesis_failed = False
        mock_merge = MagicMock()
        mock_set_outstanding = MagicMock()

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.pm_core.TargetStore") as mock_ts,
            patch("lapis_pm.pm_core.brief._act_merge_pr", mock_merge),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_b),
            patch("lapis_pm.pm_core._set_brief_outstanding", mock_set_outstanding),
            patch("lapis_pm.pm_core.episodic.write_hold"),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core._mem", return_value=MagicMock()),
            patch("lapis_pm.pm_core._mark_pr_classified"),
        ):
            mock_ts.return_value.get.return_value = target

            result = pm_core._act_brief("my-target", "advisory-screen-issue", False, payload)

        assert "brief_emitted" in result
        assert "advisory_screen_issue" in result
        mock_merge.assert_not_called()
        mock_set_outstanding.assert_called_once()

    def test_held_path_produces_brief_not_auto_resolve(self):
        """held-path blocked => auto-resolve skipped; normal brief/gem path."""
        from lapis_pm import pm_core

        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="advisory")
        payload = self._make_payload(cls)

        mock_b = MagicMock()
        mock_b.comment_id = "brief-001"
        mock_b.synthesis_failed = False
        mock_merge = MagicMock()
        mock_set_outstanding = MagicMock()

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.pm_core.TargetStore") as mock_ts,
            patch("lapis_pm.auto_resolve.brief_gem") as mock_bg,
            patch("lapis_pm.pm_core.brief._act_merge_pr", mock_merge),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_b),
            patch("lapis_pm.pm_core._set_brief_outstanding", mock_set_outstanding),
            patch("lapis_pm.pm_core.episodic.write_hold"),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core._mem", return_value=MagicMock()),
            patch("lapis_pm.pm_core._mark_pr_classified"),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=None),
        ):
            mock_ts.return_value.get.return_value = target
            mock_bg._held_path_check_live.return_value = (True, "held: room/targets/x.yaml")

            result = pm_core._act_brief("my-target", "advisory-clean", False, payload)

        assert "brief_emitted" in result
        mock_merge.assert_not_called()
        mock_set_outstanding.assert_called_once()
        # Verify the live gate was actually called (not a cached check)
        mock_bg._held_path_check_live.assert_called_once()

    def test_dial_off_produces_brief_not_auto_resolve(self):
        """PM_AUTO_RESOLVE=off => skip auto-resolve; normal brief/gem path."""
        from lapis_pm import pm_core

        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="advisory")
        payload = self._make_payload(cls)

        mock_b = MagicMock()
        mock_b.comment_id = "brief-001"
        mock_b.synthesis_failed = False
        mock_merge = MagicMock()
        mock_set_outstanding = MagicMock()

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "off"}),
            patch("lapis_pm.pm_core.TargetStore") as mock_ts,
            patch("lapis_pm.pm_core.brief._act_merge_pr", mock_merge),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_b),
            patch("lapis_pm.pm_core._set_brief_outstanding", mock_set_outstanding),
            patch("lapis_pm.pm_core.episodic.write_hold"),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core._mem", return_value=MagicMock()),
            patch("lapis_pm.pm_core._mark_pr_classified"),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=None),
        ):
            mock_ts.return_value.get.return_value = target

            result = pm_core._act_brief("my-target", "advisory-clean", False, payload)

        assert "brief_emitted" in result
        mock_merge.assert_not_called()
        mock_set_outstanding.assert_called_once()

    def test_merge_failure_falls_through_to_brief(self):
        """If merge raises, fall through to brief/gem path — brief is never lost."""
        from lapis_pm import pm_core

        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_authority="advisory")
        payload = self._make_payload(cls)

        mock_b = MagicMock()
        mock_b.comment_id = "brief-001"
        mock_b.synthesis_failed = False
        mock_set_outstanding = MagicMock()
        mock_obs = MagicMock()

        def _raise_merge(target_id, pr_number):
            raise RuntimeError("forgejo merge API error")

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.pm_core.TargetStore") as mock_ts,
            patch("lapis_pm.auto_resolve.brief_gem") as mock_bg,
            patch("lapis_pm.pm_core.brief._act_merge_pr", side_effect=_raise_merge),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_b),
            patch("lapis_pm.pm_core._set_brief_outstanding", mock_set_outstanding),
            patch("lapis_pm.pm_core.episodic.write_hold"),
            patch("lapis_pm.pm_core.episodic.write_observation", mock_obs),
            patch("lapis_pm.pm_core._mem", return_value=MagicMock()),
            patch("lapis_pm.pm_core._mark_pr_classified"),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=None),
        ):
            mock_ts.return_value.get.return_value = target
            mock_bg._held_path_check_live.return_value = (False, "")

            result = pm_core._act_brief("my-target", "advisory-clean", False, payload)

        # Brief still emitted — not lost
        assert "brief_emitted" in result
        mock_set_outstanding.assert_called_once()
        # No pm:auto-resolved observation (merge did not succeed)
        obs_calls = [str(c) for c in mock_obs.call_args_list]
        assert not any("pm:auto-resolved" in s for s in obs_calls)
