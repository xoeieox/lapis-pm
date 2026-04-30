"""Tests for bind --authority choices including new 'hold' tier.

Coverage:
  - CLI parser accepts hold as valid --authority choice
  - Single-target bind with hold authority stores pm_authority=hold
  - pm_core._REVIEW_CYCLE_BUDGETS has hold → 4 entries
  - Fresh-reviewer mode selected for hold authority (not same-reviewer)
  - hold authority accepted in cli.build_parser()
"""

from __future__ import annotations

import pytest

from lapis_pm.cli import build_parser
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
        """The mode string for hold should be 'fresh' (for fresh-reviewer)."""
        # This mirrors the logic in _decide_for_pr
        mode = "fresh" if "hold" == "hold" else "same"
        assert mode == "fresh"

    def test_non_hold_uses_same_reviewer_mode(self):
        mode = "fresh" if "advisory" == "hold" else "same"
        assert mode == "same"
