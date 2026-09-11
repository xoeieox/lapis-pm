"""Tests for lapis-pm-auto-merge-eligibility-verification-v0.

Coverage (acceptance criteria 1-6):
  1. spec_verification_parser: machine/pm-live-test/absent/alias/unrecognized values
  2. auto_resolve_clause6: verification gates should_auto_resolve (eligible, not-eligible,
     absent-default, alias resolution)
  3. end_to_end_advisory: advisory + not-machine => routes to brief, not auto-merge
  4. end_to_end_machine: advisory + machine => auto-merges (all other clauses holding)
  5. auto_tier_downgrade: auto-authority + not-machine => downgraded to advisory_brief in classify()
  6. back_compat: existing targets with no pm_verification field treated as pm-live-test
  7. bind_flag: --verification bind flag overrides spec value
  8. status_legibility: pm_verification shown in status output
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from lapis_pm.spec_review import _parse_spec_verification_text, _parse_spec_verification
from lapis_pm import auto_resolve
from lapis_pm import authority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cls(
    screen_verdict: str = "clean",
    issues: list | None = None,
    pr_number: int = 42,
) -> MagicMock:
    cls = MagicMock()
    cls.screen_verdict = screen_verdict
    cls.issues = issues if issues is not None else []
    cls.pr_number = pr_number
    cls.repo = "lapis-pm"
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
    pm_verification: str | None = None,
) -> MagicMock:
    t = MagicMock()
    t.id = target_id
    t.pm_authority = pm_authority
    t.pm_repo = pm_repo
    data = {"pm_authority": pm_authority, "pm_repo": pm_repo}
    if pm_verification is not None:
        data["pm_verification"] = pm_verification
    t.data = data
    return t


# ---------------------------------------------------------------------------
# 1. Spec verification parser
# ---------------------------------------------------------------------------

class TestParseSpecVerificationText:
    def test_machine_returns_machine(self):
        text = "# Spec\n**Verification:** machine\n**Authority:** hold\n"
        assert _parse_spec_verification_text(text) == "machine"

    def test_pm_live_test_returns_pm_live_test(self):
        text = "# Spec\n**Verification:** pm-live-test\n"
        assert _parse_spec_verification_text(text) == "pm-live-test"

    def test_live_test_alias_returns_pm_live_test(self):
        text = "# Spec\n**Verification:** live-test\n"
        assert _parse_spec_verification_text(text) == "pm-live-test"

    def test_human_alias_returns_pm_live_test(self):
        text = "# Spec\n**Verification:** human\n"
        assert _parse_spec_verification_text(text) == "pm-live-test"

    def test_absent_returns_pm_live_test(self):
        text = "# Spec\n**Authority:** advisory\n"
        assert _parse_spec_verification_text(text) == "pm-live-test"

    def test_unrecognized_returns_pm_live_test(self):
        text = "# Spec\n**Verification:** both\n"
        assert _parse_spec_verification_text(text) == "pm-live-test"

    def test_only_first_50_lines_scanned(self):
        """Verification field beyond line 50 must be ignored (fail-safe)."""
        header = "\n".join(f"line {i}" for i in range(55))
        text = header + "\n**Verification:** machine\n"
        assert _parse_spec_verification_text(text) == "pm-live-test"

    def test_machine_inline_with_extra_text(self):
        """machine followed by prose description must still parse."""
        text = "**Verification:** machine — acceptance is fully covered by tests\n"
        assert _parse_spec_verification_text(text) == "machine"

    def test_empty_text_returns_pm_live_test(self):
        assert _parse_spec_verification_text("") == "pm-live-test"


class TestParseSpecVerificationFile:
    def test_reads_machine_from_file(self, tmp_path):
        f = tmp_path / "spec.md"
        f.write_text("**Verification:** machine\n**Authority:** hold\n")
        assert _parse_spec_verification(f) == "machine"

    def test_missing_file_returns_pm_live_test(self, tmp_path):
        f = tmp_path / "nonexistent.md"
        assert _parse_spec_verification(f) == "pm-live-test"

    def test_absent_field_returns_pm_live_test(self, tmp_path):
        f = tmp_path / "spec.md"
        f.write_text("# Spec\n**Authority:** advisory\n")
        assert _parse_spec_verification(f) == "pm-live-test"


# ---------------------------------------------------------------------------
# 2. auto_resolve Clause 6
# ---------------------------------------------------------------------------

class TestAutoResolveClause6:
    def test_machine_verification_passes(self):
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_verification="machine")

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.auto_resolve.brief_gem") as mock_bg,
        ):
            mock_bg._held_path_check_live.return_value = (False, "")
            ok, opt = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is True
        assert opt == "A"

    def test_pm_live_test_blocks(self):
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_verification="pm-live-test")

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "verification=" in reason
        assert "needs-pm-touch" in reason

    def test_absent_verification_defaults_to_pm_live_test(self):
        """Back-compat: no pm_verification field => pm-live-test => not eligible."""
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target()  # no pm_verification key in data

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "needs-pm-touch" in reason

    def test_verification_clause_short_circuits_before_held_path_network_call(self):
        """When verification blocks, the live network gate must NOT be called."""
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_verification="pm-live-test")

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.auto_resolve.brief_gem") as mock_bg,
        ):
            mock_bg._held_path_check_live.return_value = (False, "")
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        mock_bg._held_path_check_live.assert_not_called()

    def test_reason_format(self):
        """Reason must include both verification value and needs-pm-touch signal."""
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_verification="pm-live-test")

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "verification='pm-live-test'" in reason
        assert "needs-pm-touch" in reason


# ---------------------------------------------------------------------------
# 3 & 4. End-to-end advisory: not-machine routes to brief; machine auto-merges
# ---------------------------------------------------------------------------

class TestActBriefVerificationGate:
    """_act_brief skips auto-resolve when pm_verification != machine."""

    def test_pm_live_test_routes_to_brief_not_auto_merge(self):
        from lapis_pm import pm_core

        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_verification="pm-live-test")
        payload = {"classification": cls}

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
            mock_bg._held_path_check_live.return_value = (False, "")

            result = pm_core._act_brief("my-target", "advisory-clean", False, payload)

        assert "brief_emitted" in result
        mock_merge.assert_not_called()
        mock_set_outstanding.assert_called_once()

    def test_machine_verification_auto_resolves(self):
        """advisory + machine + clean + conservative => merge, no brief."""
        from lapis_pm import pm_core

        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_verification="machine")
        payload = {"classification": cls}

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
            patch("lapis_pm.pm_core._mem", return_value=MagicMock()),
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

    def test_verification_skip_writes_observation(self):
        """When auto-resolve is skipped due to verification, an observation is recorded."""
        from lapis_pm import pm_core

        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target(pm_verification="pm-live-test")
        payload = {"classification": cls}

        mock_b = MagicMock()
        mock_b.comment_id = "brief-001"
        mock_b.synthesis_failed = False
        mock_obs = MagicMock()

        with (
            patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}),
            patch("lapis_pm.pm_core.TargetStore") as mock_ts,
            patch("lapis_pm.auto_resolve.brief_gem") as mock_bg,
            patch("lapis_pm.pm_core.brief._act_merge_pr"),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_b),
            patch("lapis_pm.pm_core._set_brief_outstanding"),
            patch("lapis_pm.pm_core.episodic.write_hold"),
            patch("lapis_pm.pm_core.episodic.write_observation", mock_obs),
            patch("lapis_pm.pm_core._mem", return_value=MagicMock()),
            patch("lapis_pm.pm_core._mark_pr_classified"),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=None),
        ):
            mock_ts.return_value.get.return_value = target
            mock_bg._held_path_check_live.return_value = (False, "")

            pm_core._act_brief("my-target", "advisory-clean", False, payload)

        obs_calls = [str(c) for c in mock_obs.call_args_list]
        assert any("needs-pm-touch" in s or "pm-live-test" in s for s in obs_calls), (
            f"Expected verification-skip observation; got: {obs_calls}"
        )


# ---------------------------------------------------------------------------
# 5. auto-tier downgrade via authority.classify()
# ---------------------------------------------------------------------------

class TestAuthorityClassifyVerificationDowngrade:
    def test_auto_tier_not_machine_downgrades_to_advisory(self):
        """auto-authority + verification != machine => classify returns advisory verdict."""
        mock_pr = {"title": "feat: thing", "html_url": "http://x/1", "mergeable": True}
        mock_diff = ""

        with (
            patch("lapis_pm.authority.get_pr", return_value=mock_pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=mock_diff),
        ):
            cls = authority.classify(
                "lapis-pm", 1, "summary",
                pm_authority="auto",
                verification="pm-live-test",
            )

        assert cls.verdict == "advisory"
        assert any("downgraded" in r for r in cls.reasons)
        assert any("verification" in r for r in cls.reasons)

    def test_auto_tier_machine_proceeds_to_screen(self):
        """auto-authority + machine => runs inline screen (screen called)."""
        mock_pr = {"title": "feat: thing", "html_url": "http://x/1", "mergeable": True}
        mock_diff = ""
        mock_screen = {"verdict": "clean", "issues": [], "confidence": 0.95}

        with (
            patch("lapis_pm.authority.get_pr", return_value=mock_pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=mock_diff),
            patch("lapis_pm.authority.screen", return_value=mock_screen) as mock_sc,
        ):
            cls = authority.classify(
                "lapis-pm", 1, "summary",
                pm_authority="auto",
                verification="machine",
            )

        mock_sc.assert_called_once()
        assert cls.screen_verdict == "clean"

    def test_advisory_path_ignores_verification(self):
        """advisory authority is unaffected by verification (no screen called)."""
        mock_pr = {"title": "feat: thing", "html_url": "http://x/1", "mergeable": True}
        mock_diff = ""

        with (
            patch("lapis_pm.authority.get_pr", return_value=mock_pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=mock_diff),
            patch("lapis_pm.authority.screen") as mock_sc,
        ):
            cls = authority.classify(
                "lapis-pm", 1, "summary",
                pm_authority="advisory",
                verification="pm-live-test",
            )

        # Advisory path never runs inline screen
        mock_sc.assert_not_called()
        assert cls.verdict == "advisory"

    def test_absent_verification_defaults_pm_live_test_for_auto_tier(self):
        """auto-authority with no verification arg => pm-live-test default => downgraded."""
        mock_pr = {"title": "feat: thing", "html_url": "http://x/1", "mergeable": True}
        mock_diff = ""

        with (
            patch("lapis_pm.authority.get_pr", return_value=mock_pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=mock_diff),
        ):
            cls = authority.classify("lapis-pm", 1, "summary", pm_authority="auto")

        assert cls.verdict == "advisory"


# ---------------------------------------------------------------------------
# 6. Back-compat: existing targets without pm_verification
# ---------------------------------------------------------------------------

class TestBackCompat:
    def test_no_pm_verification_field_treated_as_pm_live_test(self):
        """Targets bound before this change have no pm_verification — must default to
        pm-live-test (not eligible), not crash."""
        cls = _make_cls(screen_verdict="clean", issues=[])
        target = _make_target()  # data has no "pm_verification" key

        assert "pm_verification" not in target.data

        with patch.dict("os.environ", {"PM_AUTO_RESOLVE": "conservative"}):
            ok, reason = auto_resolve.should_auto_resolve(cls, target, "lapis-pm")

        assert ok is False
        assert "pm-live-test" in reason


# ---------------------------------------------------------------------------
# 7. --verification bind flag
# ---------------------------------------------------------------------------

class TestBindVerificationFlag:
    def test_verification_flag_accepted_by_parser(self):
        from lapis_pm.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "bind", "mytarget",
            "--spec-from", "-",
            "--repo", "myrepo",
            "--verification", "machine",
        ])
        assert args.verification == "machine"

    def test_pm_live_test_accepted_by_parser(self):
        from lapis_pm.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "bind", "mytarget",
            "--spec-from", "-",
            "--repo", "myrepo",
            "--verification", "pm-live-test",
        ])
        assert args.verification == "pm-live-test"

    def test_invalid_verification_rejected(self):
        from lapis_pm.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "bind", "mytarget",
                "--spec-from", "-",
                "--repo", "myrepo",
                "--verification", "invalid",
            ])

    def test_verification_absent_defaults_to_none(self):
        """Without --verification, arg is None (spec value used at runtime)."""
        from lapis_pm.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "bind", "mytarget",
            "--spec-from", "-",
            "--repo", "myrepo",
        ])
        assert args.verification is None


# ---------------------------------------------------------------------------
# 8. Status legibility
# ---------------------------------------------------------------------------

class TestStatusLegibility:
    def test_pm_verification_shown_in_status_output(self, capsys):
        from lapis_pm.cli import _print_target_status

        t = MagicMock()
        t.id = "my-target"
        t.title = "My Target"
        t.pm_bound = True
        t.pm_repo = "lapis-pm"
        t.pm_authority = "advisory"
        t.paused = False
        t.paused_reason = None
        t.data = {"pm_verification": "machine"}

        with (
            patch("lapis_pm.cli.pm_core.get_cursor", return_value="2026-06-28T00:00:00"),
            patch("lapis_pm.cli.pm_core.load_dispatched", return_value=[]),
            patch("lapis_pm.cli.pm_core.get_outstanding_brief", return_value=None),
            ):
            _print_target_status(t, explain=False)

        captured = capsys.readouterr()
        assert "pm_verification" in captured.out
        assert "machine" in captured.out

    def test_absent_pm_verification_shows_default_in_status(self, capsys):
        from lapis_pm.cli import _print_target_status

        t = MagicMock()
        t.id = "my-target"
        t.title = "My Target"
        t.pm_bound = True
        t.pm_repo = "lapis-pm"
        t.pm_authority = "advisory"
        t.paused = False
        t.paused_reason = None
        t.data = {}  # no pm_verification key

        with (
            patch("lapis_pm.cli.pm_core.get_cursor", return_value="2026-06-28T00:00:00"),
            patch("lapis_pm.cli.pm_core.load_dispatched", return_value=[]),
            patch("lapis_pm.cli.pm_core.get_outstanding_brief", return_value=None),
            ):
            _print_target_status(t, explain=False)

        captured = capsys.readouterr()
        assert "pm_verification" in captured.out
        assert "pm-live-test" in captured.out
