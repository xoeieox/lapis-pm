"""Tests for bind --authority choices including new 'hold' tier.

Coverage:
  - CLI parser accepts hold as valid --authority choice
  - Single-target bind with hold authority stores pm_authority=hold
  - pm_core._REVIEW_CYCLE_BUDGETS has hold → 4 entries
  - Fresh-reviewer mode selected for hold authority (not same-reviewer)
  - hold authority accepted in cli.build_parser()
  - _parse_spec_authority_text: parses **Authority:** header from spec text,
    including auto-merge/auto alias normalization, absent/unrecognized => None,
    only first 50 lines scanned (lapis-pm-bind-authority-header-parity-v0)
  - cmd_bind resolves --authority as: flag > spec header > advisory fallback,
    and errors (exit 2, nothing written) on flag/header mismatch
"""

from __future__ import annotations

import io
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agents_core.targets import TargetStore
from lapis_pm.cli import build_parser, main
from lapis_pm.spec_review import _parse_spec_authority_text
from lapis_pm import pm_core


class TestBindAuthorityChoices:
    def test_advisory_accepted(self):
        parser = build_parser()
        args = parser.parse_args([
            "bind", "mytarget",
            "--spec-from", "-",
            "--repo", "myrepo",
            "--authority", "advisory",
        ])
        assert args.authority == "advisory"

    def test_auto_accepted(self):
        parser = build_parser()
        args = parser.parse_args([
            "bind", "mytarget",
            "--spec-from", "-",
            "--repo", "myrepo",
            "--authority", "auto",
        ])
        assert args.authority == "auto"

    def test_hold_accepted(self):
        """hold must be a valid --authority choice."""
        parser = build_parser()
        args = parser.parse_args([
            "bind", "mytarget",
            "--spec-from", "-",
            "--repo", "myrepo",
            "--authority", "hold",
        ])
        assert args.authority == "hold"

    def test_invalid_authority_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "bind", "mytarget",
                "--spec-from", "-",
                "--repo", "myrepo",
                "--authority", "invalid",
            ])

    def test_hold_not_in_default_when_legs_from_absent(self):
        """Without --legs-from, authority defaults to None (normalized to advisory at runtime)."""
        parser = build_parser()
        args = parser.parse_args([
            "bind", "mytarget",
            "--spec-from", "-",
            "--repo", "myrepo",
        ])
        assert args.authority is None  # normalized to "advisory" in cmd_bind

    def test_legs_from_present(self):
        """--legs-from flag is accepted by the parser."""
        parser = build_parser()
        args = parser.parse_args([
            "bind", "my-chain",
            "--spec-from", "-",
            "--legs-from", "/tmp/legs.yaml",
        ])
        assert args.legs_from == "/tmp/legs.yaml"
        assert args.repo is None

    def test_no_auto_fire_flag(self):
        """--no-auto-fire is accepted by the parser."""
        parser = build_parser()
        args = parser.parse_args([
            "bind", "my-chain",
            "--spec-from", "-",
            "--legs-from", "/tmp/legs.yaml",
            "--no-auto-fire",
        ])
        assert args.no_auto_fire is True


class TestHoldReviewBehavior:
    def test_hold_cycle_budget_is_4(self):
        """hold authority must have a 4-cycle budget."""
        assert pm_core._REVIEW_CYCLE_BUDGETS.get("hold") == 4

    def test_advisory_cycle_budget_is_2(self):
        assert pm_core._REVIEW_CYCLE_BUDGETS.get("advisory") == 2

    def test_hold_uses_fresh_reviewer_mode(self):
        """hold authority must map to fresh-reviewer mode in pm_core._REVIEWER_MODES."""
        assert pm_core._REVIEWER_MODES.get("hold") == "fresh-reviewer"

    def test_non_hold_uses_same_reviewer_mode(self):
        """advisory authority must map to same-reviewer mode in pm_core._REVIEWER_MODES."""
        assert pm_core._REVIEWER_MODES.get("advisory") == "same-reviewer"


class TestBindPmAcceptsHoldAuthority:
    """End-to-end check that target.bind_pm() accepts 'hold'.

    The new --authority hold CLI choice is only useful if the underlying
    Target.bind_pm() method (in agents-core) doesn't reject 'hold'. Without
    this regression test, the CLI surface and the binding contract can drift
    and a hold-authority bind will crash at runtime instead of in CI.

    Skipped automatically if the installed agents-core predates the
    bind_pm-widening change; the test re-engages once agents-core ships
    the matching PR.
    """

    def test_target_bind_pm_accepts_hold(self, tmp_path):
        from agents_core.targets import Target
        path = tmp_path / "h.yaml"
        path.write_text("id: h\ntitle: H\n")
        t = Target({"id": "h", "title": "H"}, path)
        try:
            t.bind_pm(repo="r", authority="hold")
        except ValueError as e:
            pytest.skip(
                f"installed agents-core rejects authority='hold' ({e}); "
                "merge agents-core PR widening bind_pm first"
            )
        assert t.pm_authority == "hold"


# ---------------------------------------------------------------------------
# _parse_spec_authority_text unit tests (mirrors _parse_spec_verification_text
# test shape in tests/test_verification_eligibility.py)
# ---------------------------------------------------------------------------


class TestParseSpecAuthorityText:
    def test_hold_returns_hold(self):
        text = "# Spec\n**Authority:** hold\n"
        assert _parse_spec_authority_text(text) == "hold"

    def test_advisory_returns_advisory(self):
        text = "# Spec\n**Authority:** advisory\n"
        assert _parse_spec_authority_text(text) == "advisory"

    def test_auto_returns_auto(self):
        text = "# Spec\n**Authority:** auto\n"
        assert _parse_spec_authority_text(text) == "auto"

    def test_auto_merge_alias_returns_auto(self):
        text = "# Spec\n**Authority:** auto-merge\n"
        assert _parse_spec_authority_text(text) == "auto"

    def test_absent_returns_none(self):
        text = "# Spec\n**Verification:** machine\n"
        assert _parse_spec_authority_text(text) is None

    def test_unrecognized_returns_none(self):
        text = "# Spec\n**Authority:** maybe\n"
        assert _parse_spec_authority_text(text) is None

    def test_only_first_50_lines_scanned(self):
        """Authority field beyond line 50 must be ignored, same as verification."""
        header = "\n".join(f"line {i}" for i in range(55))
        text = header + "\n**Authority:** hold\n"
        assert _parse_spec_authority_text(text) is None

    def test_hold_inline_with_extra_text(self):
        """hold followed by prose description must still parse."""
        text = "**Authority:** hold — this touches the secure spine\n"
        assert _parse_spec_authority_text(text) == "hold"

    def test_empty_text_returns_none(self):
        assert _parse_spec_authority_text("") is None


# ---------------------------------------------------------------------------
# cmd_bind authority resolution: flag > spec header > advisory fallback
# ---------------------------------------------------------------------------

SPEC_BODY_HOLD = textwrap.dedent("""\
    # Test spec

    **Authority:** hold

    A minimal spec body declaring hold authority.
""")

SPEC_BODY_ADVISORY = textwrap.dedent("""\
    # Test spec

    **Authority:** advisory

    A minimal spec body declaring advisory authority.
""")

SPEC_BODY_AUTO_MERGE_ALIAS = textwrap.dedent("""\
    # Test spec

    **Authority:** auto-merge

    A minimal spec body declaring the auto-merge alias.
""")

SPEC_BODY_NO_HEADER = textwrap.dedent("""\
    # Test spec

    A minimal spec body with no Authority header at all.
""")


def _write_spec(tmp_path: Path, body: str, name: str = "spec.md") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def _run(argv: list[str], targets_dir: Path) -> tuple[int, str, str]:
    """Run main() with a patched TargetStore and stubbed episodic/pm_core."""
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    with (
        patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
        patch("lapis_pm.cli.episodic.spec", return_value=None),
            patch("lapis_pm.cli.episodic.write_spec", return_value=None),
        patch("lapis_pm.cli.pm_core.clear_classified_prs", return_value=None),
            patch("agents_core.forgejo.get_open_prs", return_value=[]),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        rc = main(argv)

    return rc, out_buf.getvalue(), err_buf.getvalue()


class TestCmdBindAuthorityResolution:
    def test_hold_header_no_flag_resolves_hold(self, tmp_path):
        """Regression test for both secure-spine incidents: spec says hold, no
        --authority flag passed, bound authority must be hold, not advisory."""
        spec_path = _write_spec(tmp_path, SPEC_BODY_HOLD)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "my-target.yaml").read_text())
        assert data["pm_authority"] == "hold"

    def test_advisory_header_no_flag_stays_advisory(self, tmp_path):
        spec_path = _write_spec(tmp_path, SPEC_BODY_ADVISORY)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "my-target.yaml").read_text())
        assert data["pm_authority"] == "advisory"

    def test_no_header_no_flag_defaults_advisory(self, tmp_path):
        """Fail-safe default preserved for specs that predate this convention."""
        spec_path = _write_spec(tmp_path, SPEC_BODY_NO_HEADER)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "my-target.yaml").read_text())
        assert data["pm_authority"] == "advisory"

    def test_hold_header_advisory_flag_mismatch_errors(self, tmp_path):
        """Explicit --authority disagreeing with the spec header must abort the
        bind before anything is written, naming both values."""
        spec_path = _write_spec(tmp_path, SPEC_BODY_HOLD)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        with patch("agents_core.targets.Target.bind_pm") as mock_bind_pm:
            rc, out, err = _run([
                "bind", "my-target",
                "--spec-from", str(spec_path),
                "--repo", "lapis-pm",
                "--authority", "advisory",
                "--force",
            ], tmp_path)

        assert rc == 2
        assert "hold" in err
        assert "advisory" in err
        mock_bind_pm.assert_not_called()

    def test_hold_header_hold_flag_matches_no_error(self, tmp_path):
        spec_path = _write_spec(tmp_path, SPEC_BODY_HOLD)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--authority", "hold",
            "--force",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "my-target.yaml").read_text())
        assert data["pm_authority"] == "hold"

    def test_auto_merge_header_auto_flag_alias_matches_no_error(self, tmp_path):
        """auto-merge header + --authority auto is a match via alias normalization."""
        spec_path = _write_spec(tmp_path, SPEC_BODY_AUTO_MERGE_ALIAS)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--authority", "auto",
            "--force",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "my-target.yaml").read_text())
        assert data["pm_authority"] == "auto"
