"""Smoke test: `lapis-pm clear-reviewer-attempts <target_id>` is registered,
requires only the target id, and clears counters via pm_core.clear_reviewer_attempts
(DoD 4b: single documented command, no required flags beyond target,
no confirmation prompt)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from agents_core.mem import MemoryStore
from lapis_pm import cli, pm_core

TID = "test-cli-clear-reviewer-attempts-fixture"


def test_subcommand_parses_with_target_id_only():
    parser = cli.build_parser()
    args = parser.parse_args(["clear-reviewer-attempts", TID])
    assert args.target_id == TID
    assert args.func is cli.cmd_clear_reviewer_attempts


def test_cmd_clear_reviewer_attempts_clears_and_returns_zero(capsys):
    store = MemoryStore(db_path=Path(tempfile.mktemp(suffix=".db")))
    with patch("lapis_pm.pm_core._mem", return_value=store):
        pm_core._increment_reviewer_attempt(TID, 33, 1)
        pm_core._increment_reviewer_attempt(TID, 33, 1)
        assert pm_core._reviewer_attempt_count(TID, 33, 1) == 2

        parser = cli.build_parser()
        args = parser.parse_args(["clear-reviewer-attempts", TID])
        rc = args.func(args)

        assert rc == 0
        assert pm_core._reviewer_attempt_count(TID, 33, 1) == 0

    out = capsys.readouterr().out
    assert "Cleared" in out
