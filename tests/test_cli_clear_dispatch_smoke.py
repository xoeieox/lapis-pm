"""Smoke test: `lapis-pm clear-dispatch <target_id>` is registered, requires
only the target id, flips pending dispatch records to failed via
pm_core.clear_dispatch, leaves non-pending records untouched, and always
prints the intention-registry reminder (Leg 3 of
lapis-pm-stale-pending-dispatch-reaper-v0 — modeled on
test_cli_clear_reviewer_attempts_smoke.py)."""

from __future__ import annotations

from unittest.mock import patch

from lapis_pm import cli, pm_core

TID = "test-cli-clear-dispatch-fixture"


def test_subcommand_parses_with_target_id_only():
    parser = cli.build_parser()
    args = parser.parse_args(["clear-dispatch", TID])
    assert args.target_id == TID
    assert args.func is cli.cmd_clear_dispatch


def test_cmd_clear_dispatch_clears_pending_and_leaves_others(capsys):
    records = [
        {"gpu_id": "gpu-pending-1", "agent_type": "fixer", "status": "pending", "ts": "2026-08-01T00:00:00-07:00"},
        {"gpu_id": "gpu-processed-1", "agent_type": "fixer", "status": "processed", "ts": "2026-08-01T00:00:00-07:00"},
        {"gpu_id": "gpu-pending-2", "agent_type": "reviewer", "status": "pending", "ts": "2026-08-02T00:00:00-07:00"},
    ]
    saved = {}

    def fake_save(target_id, recs):
        saved["records"] = recs

    with patch("lapis_pm.pm_core.load_dispatched", return_value=records), \
         patch("lapis_pm.pm_core.save_dispatched", side_effect=fake_save), \
         patch("lapis_pm.episodic.write_observation") as mock_obs:
        parser = cli.build_parser()
        args = parser.parse_args(["clear-dispatch", TID])
        rc = args.func(args)

    assert rc == 0
    assert saved["records"][0]["status"] == "failed"
    assert saved["records"][0]["failure_reason"] == "cleared:operator"
    assert saved["records"][0]["completed_at"]
    assert saved["records"][1]["status"] == "processed"  # untouched
    assert saved["records"][2]["status"] == "failed"
    assert saved["records"][2]["failure_reason"] == "cleared:operator"
    mock_obs.assert_called_once()

    out = capsys.readouterr().out
    assert "Cleared 2 pending dispatch record(s)" in out
    assert "intention registry" in out.lower()


def test_cmd_clear_dispatch_zero_records_still_prints_reminder(capsys):
    with patch("lapis_pm.pm_core.load_dispatched", return_value=[]), \
         patch("lapis_pm.pm_core.save_dispatched") as mock_save, \
         patch("lapis_pm.episodic.write_observation") as mock_obs:
        parser = cli.build_parser()
        args = parser.parse_args(["clear-dispatch", TID])
        rc = args.func(args)

    assert rc == 0
    mock_save.assert_not_called()
    mock_obs.assert_not_called()

    out = capsys.readouterr().out
    assert "Cleared 0 pending dispatch record(s)" in out
    assert "intention registry" in out.lower()
