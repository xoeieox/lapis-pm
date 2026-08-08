"""Hermetic tests for lapis_pm.deploy_inventory_repair (deploy-inventory-repair-proposer-v0).

No live weaver/GW dependency: httpx.Client and call_gw_agent are stubbed at
every call site. Mirrors the DoD-1..9c shape from the bound spec.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from lapis_pm import deploy_inventory_repair as dir_mod


def _finding(kind="stale_behind_origin", severity="HIGH", detail="3 commits behind"):
    return {"kind": kind, "severity": severity, "detail": detail}


def _clone(path="/srv/git/foo-working", mapped=True, branch="main", commits_behind=3,
           findings=None, backing_units=None):
    return {
        "path": path,
        "mapped": mapped,
        "branch": branch,
        "commits_behind": commits_behind,
        "backing_units": backing_units or [],
        "findings": findings if findings is not None else [_finding()],
    }


def _status(clones):
    return {"generated": "2026-08-07T00:00:00+00:00", "clones": clones}


@pytest.fixture
def ledger_file(tmp_path):
    return tmp_path / "deploy-inventory-repair-ledger.json"


def _degrade_diagnosis(*args, **kwargs):
    return dir_mod._degrade(kwargs.get("finding") or args[1]), "mechanical"


class TestSignature:
    def test_signature_deterministic_and_excludes_commits_behind(self):
        sig1 = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        sig2 = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        assert sig1 == sig2
        assert len(sig1) == 16

    def test_signature_differs_by_kind_or_path(self):
        a = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        b = dir_mod.compute_signature("/srv/git/foo", "stray_branch")
        c = dir_mod.compute_signature("/srv/git/bar", "stale_behind_origin")
        assert len({a, b, c}) == 3


class TestLedgerAtomicWrite:
    def test_write_then_read_roundtrip(self, ledger_file):
        ledger = {"abc123": {"gem_id": "g1", "status": "open"}}
        dir_mod.write_ledger(ledger, path=ledger_file)
        assert dir_mod.read_ledger(path=ledger_file) == ledger

    def test_read_missing_file_fails_open_empty(self, tmp_path):
        assert dir_mod.read_ledger(path=tmp_path / "nope.json") == {}

    def test_read_corrupt_file_fails_open_empty(self, ledger_file):
        ledger_file.write_text("{not valid json")
        assert dir_mod.read_ledger(path=ledger_file) == {}

    def test_write_is_atomic_temp_then_replace(self, ledger_file, tmp_path):
        """A write failure mid-flight (os.replace raises) must leave the
        prior ledger content intact — never a torn/partial file (DoD-9a).
        """
        dir_mod.write_ledger({"orig": {"status": "open"}}, path=ledger_file)
        prior_content = ledger_file.read_text()

        with patch("os.replace", side_effect=OSError("disk full")):
            dir_mod.write_ledger({"new": {"status": "open"}}, path=ledger_file)

        # prior content untouched — no partial/torn write landed
        assert ledger_file.read_text() == prior_content


class TestDedupAndSelection:
    def test_no_entry_deposits_and_records(self, ledger_file):
        ledger = {}
        with (
            patch.object(dir_mod, "diagnose_finding", return_value=({"root_cause": "x", "suggested_direction": "", "affected_files": [], "confidence": None, "uncertain": False}, "qwen3.6-35b-a3b")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-1") as mock_deposit,
            patch.object(dir_mod, "emit_provenance") as mock_prov,
        ):
            action = dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=3, backing_units=[],
            )
        assert action == "deposited:new"
        assert mock_deposit.called
        sig = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        assert ledger[sig]["gem_id"] == "gem-1"
        assert ledger[sig]["status"] == "open"
        mock_prov.assert_called_once()
        assert mock_prov.call_args.kwargs["model"] == "qwen3.6-35b-a3b"

    def test_still_open_suppresses_and_bumps(self):
        sig = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        ledger = {sig: {"gem_id": "gem-1", "first_seen": "2026-08-01T00:00:00+00:00",
                        "last_seen": "2026-08-01T00:00:00+00:00", "times_seen": 1,
                        "status": "open", "staled_at": None,
                        "clone_path": "/srv/git/foo", "finding_kind": "stale_behind_origin"}}
        with patch.object(dir_mod, "deposit_gem") as mock_deposit:
            action = dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=5, backing_units=[],
            )
        assert action == "skip:still-open"
        assert not mock_deposit.called
        assert ledger[sig]["times_seen"] == 2

    def test_dismissed_fp_suppresses_and_bumps(self):
        sig = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        ledger = {sig: {"gem_id": "gem-1", "first_seen": "2026-08-01T00:00:00+00:00",
                        "last_seen": "2026-08-01T00:00:00+00:00", "times_seen": 1,
                        "status": "dismissed_fp", "staled_at": None,
                        "clone_path": "/srv/git/foo", "finding_kind": "stale_behind_origin"}}
        with patch.object(dir_mod, "deposit_gem") as mock_deposit:
            action = dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=5, backing_units=[],
            )
        assert action == "skip:dismissed-fp"
        assert not mock_deposit.called
        assert ledger[sig]["times_seen"] == 2

    def test_acked_reproposes_with_recurrence_note(self):
        sig = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        ledger = {sig: {"gem_id": "gem-1", "first_seen": "2026-08-01T00:00:00+00:00",
                        "last_seen": "2026-08-01T00:00:00+00:00", "times_seen": 1,
                        "status": "acked", "staled_at": None,
                        "clone_path": "/srv/git/foo", "finding_kind": "stale_behind_origin"}}
        with (
            patch.object(dir_mod, "diagnose_finding", return_value=(dir_mod._degrade(_finding()), "mechanical")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-2") as mock_deposit,
            patch.object(dir_mod, "emit_provenance"),
        ):
            action = dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=5, backing_units=[],
            )
        assert action == "deposited:recurred-after-ack"
        assert mock_deposit.called
        payload = mock_deposit.call_args[0][0]
        stakes_lines = payload["context"][3]["lines"]
        assert any("recurred-after-ack" in line for line in stakes_lines)
        assert ledger[sig]["gem_id"] == "gem-2"
        assert ledger[sig]["first_seen"] == "2026-08-01T00:00:00+00:00"  # preserved

    def test_deposit_failure_leaves_ledger_unrecorded(self):
        ledger = {}
        with (
            patch.object(dir_mod, "diagnose_finding", return_value=(dir_mod._degrade(_finding()), "mechanical")),
            patch.object(dir_mod, "deposit_gem", return_value=None),
        ):
            action = dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=3, backing_units=[],
            )
        assert action == "skip:deposit-failed"
        assert ledger == {}  # retries next pass


class TestDiagnosisGracefulDegrade:
    def test_valid_json_populates_diagnosis(self):
        served = []

        def fake_call(*args, served_model_out=None, **kwargs):
            served_model_out.append("qwen3.6-35b-a3b")
            return json.dumps({
                "root_cause": "clone is behind", "suggested_direction": "pull it",
                "affected_files": [], "confidence": "high", "uncertain": False,
            })

        with patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call):
            diag, model = dir_mod.diagnose_finding(
                "/srv/git/foo", _finding(), mapped=True, commits_behind=3,
                branch="main", backing_units=[],
            )
        assert diag["root_cause"] == "clone is behind"
        assert diag["uncertain"] is False
        assert model == "qwen3.6-35b-a3b"

    def test_malformed_result_degrades_to_detail(self):
        with patch("agents_core.gw_agent.call_gw_agent", return_value="not json at all"):
            diag, model = dir_mod.diagnose_finding(
                "/srv/git/foo", _finding(detail="3 behind"), mapped=True,
                commits_behind=3, branch="main", backing_units=[],
            )
        assert diag["root_cause"] == "3 behind"
        assert diag["uncertain"] is True
        assert model == "mechanical"

    def test_none_result_degrades(self):
        with patch("agents_core.gw_agent.call_gw_agent", return_value=None):
            diag, model = dir_mod.diagnose_finding(
                "/srv/git/foo", _finding(detail="3 behind"), mapped=True,
                commits_behind=3, branch="main", backing_units=[],
            )
        assert diag["uncertain"] is True
        assert model == "mechanical"

    def test_exception_degrades(self):
        with patch("agents_core.gw_agent.call_gw_agent", side_effect=RuntimeError("boom")):
            diag, model = dir_mod.diagnose_finding(
                "/srv/git/foo", _finding(detail="3 behind"), mapped=True,
                commits_behind=3, branch="main", backing_units=[],
            )
        assert diag["uncertain"] is True
        assert model == "mechanical"


class TestDepositContract:
    def test_deposit_returns_gem_id_on_201(self):
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {"gem_id": "gem-99"}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        with (
            patch("httpx.Client", return_value=mock_client),
            patch.dict("os.environ", {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            gem_id = dir_mod.deposit_gem({"title": "t"})
        assert gem_id == "gem-99"

    def test_deposit_non_201_returns_none(self):
        mock_resp = MagicMock(status_code=422, text="bad payload")
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        with (
            patch("httpx.Client", return_value=mock_client),
            patch.dict("os.environ", {"WEAVER_BASE_URL": "http://mock-weaver:9999"}),
        ):
            gem_id = dir_mod.deposit_gem({"title": "t"})
        assert gem_id is None

    def test_pytest_guard_blocks_without_weaver_base_url(self):
        import os
        env = {k: v for k, v in os.environ.items() if k != "WEAVER_BASE_URL"}
        env["PYTEST_CURRENT_TEST"] = "test_pytest_guard_blocks_without_weaver_base_url"
        with (
            patch.dict("os.environ", env, clear=True),
            patch("httpx.Client") as mock_httpx,
        ):
            gem_id = dir_mod.deposit_gem({"title": "t"})
        assert gem_id is None
        mock_httpx.assert_not_called()

    def test_payload_has_two_options_no_dead_button(self):
        payload = dir_mod.build_gem_payload(
            clone_path="/srv/git/foo", finding=_finding(), mapped=True, branch="main",
            commits_behind=3, diagnosis=dir_mod._degrade(_finding()),
            stakes_line="not currently backing a live unit", times_seen=1,
        )
        keys = {o["key"] for o in payload["options"]}
        assert keys == {"ack_watch", "not_real"}
        assert payload["state"] == "needs"
        assert "title" in payload and "ask" in payload and "why" in payload


class TestCapAndNoSilentTruncation:
    def test_cap_limits_diagnosed_count_and_logs_dropped(self, caplog):
        clones = [
            _clone(path=f"/srv/git/repo{i}", findings=[_finding(kind="stale_behind_origin")])
            for i in range(dir_mod.DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN + 3)
        ]
        status = _status(clones)
        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "diagnose_finding", return_value=(dir_mod._degrade(_finding()), "mechanical")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-x"),
            patch.object(dir_mod, "emit_provenance"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value={}),
            patch.object(dir_mod, "escalate_stale_gems"),
            caplog.at_level("WARNING"),
        ):
            actions = dir_mod.run_repair_pass(status, prior_high_keys=set(), corrupt=False)
        deposited = [a for a in actions if a.startswith("deposited:")]
        assert len(deposited) == dir_mod.DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN
        assert any(a.startswith("dropped:3") for a in actions)
        assert any("dropped" in rec.message for rec in caplog.records)


class TestFaultIsolation:
    def test_one_finding_exception_does_not_abort_others(self):
        clones = [
            _clone(path="/srv/git/bad", findings=[_finding(kind="stale_behind_origin")]),
            _clone(path="/srv/git/good", findings=[_finding(kind="stray_branch")]),
        ]
        status = _status(clones)

        def fake_propose(ledger, *, clone_path, finding, **kwargs):
            if clone_path == "/srv/git/bad":
                raise RuntimeError("boom")
            return "deposited:new"

        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "propose_repair", side_effect=fake_propose),
            patch.object(dir_mod, "write_ledger") as mock_write,
            patch.object(dir_mod, "read_ledger", return_value={}),
            patch.object(dir_mod, "escalate_stale_gems"),
        ):
            actions = dir_mod.run_repair_pass(status, prior_high_keys=set(), corrupt=False)

        assert any(a.startswith("error:/srv/git/bad") for a in actions)
        assert any(a.startswith("deposited:new:/srv/git/good") for a in actions)
        mock_write.assert_called_once()  # ledger still persisted despite the one failure


class TestSelectionGate:
    def test_only_fresh_high_findings_proposed(self):
        clones = [
            _clone(path="/srv/git/foo", findings=[
                _finding(kind="stale_behind_origin"),  # already in prior_high_keys
                _finding(kind="stray_branch"),          # fresh
            ]),
        ]
        status = _status(clones)
        prior_high_keys = {("/srv/git/foo", "stale_behind_origin")}
        seen_findings = []

        def fake_propose(ledger, *, clone_path, finding, **kwargs):
            seen_findings.append(finding["kind"])
            return "deposited:new"

        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "propose_repair", side_effect=fake_propose),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value={}),
            patch.object(dir_mod, "escalate_stale_gems"),
        ):
            dir_mod.run_repair_pass(status, prior_high_keys=prior_high_keys, corrupt=False)

        assert seen_findings == ["stray_branch"]

    def test_auto_recovery_restart_failed_exempt_from_freshness_gate(self):
        clones = [
            _clone(path="/srv/git/foo", findings=[
                _finding(kind="auto_recovery_restart_failed"),
            ]),
        ]
        status = _status(clones)
        prior_high_keys = {("/srv/git/foo", "auto_recovery_restart_failed")}
        seen_findings = []

        def fake_propose(ledger, *, clone_path, finding, **kwargs):
            seen_findings.append(finding["kind"])
            return "deposited:new"

        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "propose_repair", side_effect=fake_propose),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value={}),
            patch.object(dir_mod, "escalate_stale_gems"),
        ):
            dir_mod.run_repair_pass(status, prior_high_keys=prior_high_keys, corrupt=False)

        assert seen_findings == ["auto_recovery_restart_failed"]

    def test_normal_and_info_findings_never_proposed(self):
        clones = [
            _clone(path="/srv/git/foo", findings=[
                _finding(kind="unmapped_live_clone", severity="NORMAL"),
                _finding(kind="unmapped_live_clone", severity="INFO"),
            ]),
        ]
        status = _status(clones)
        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "propose_repair") as mock_propose,
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value={}),
            patch.object(dir_mod, "escalate_stale_gems"),
        ):
            dir_mod.run_repair_pass(status, prior_high_keys=set(), corrupt=False)
        assert not mock_propose.called


class TestCorruptStatusPath:
    def test_corrupt_status_deposits_once_via_fixed_signature(self, ledger_file):
        with (
            patch.object(dir_mod, "_LEDGER_FILE", ledger_file),
            patch.object(dir_mod, "deposit_gem", return_value="gem-corrupt") as mock_deposit,
            patch.object(dir_mod, "emit_provenance"),
        ):
            action1 = dir_mod.run_repair_pass({"clones": []}, prior_high_keys=None, corrupt=True)
            action2 = dir_mod.run_repair_pass({"clones": []}, prior_high_keys=None, corrupt=True)
        assert any(a == "deposited:corrupt-status" for a in action1)
        assert any(a == "skip:still-open" for a in action2)
        assert mock_deposit.call_count == 1


class TestTerminalGemReconciliation:
    def test_decided_ack_watch_sets_acked(self):
        ledger = {"sig1": {"gem_id": "gem-1", "status": "open"}}
        with patch.object(dir_mod, "_fetch_gems_by_state", side_effect=lambda s: (
            [{"gem_id": "gem-1", "decision_json": {"option_key": "ack_watch"}}] if s == "decided" else []
        )):
            dir_mod.reconcile_terminal_gems(ledger)
        assert ledger["sig1"]["status"] == "acked"

    def test_decided_not_real_sets_dismissed_fp(self):
        ledger = {"sig1": {"gem_id": "gem-1", "status": "open"}}
        with patch.object(dir_mod, "_fetch_gems_by_state", side_effect=lambda s: (
            [{"gem_id": "gem-1", "decision_json": {"option_key": "not_real"}}] if s == "decided" else []
        )):
            dir_mod.reconcile_terminal_gems(ledger)
        assert ledger["sig1"]["status"] == "dismissed_fp"

    def test_superseded_gem_sets_dismissed_fp(self):
        ledger = {"sig1": {"gem_id": "gem-1", "status": "open"}}
        with patch.object(dir_mod, "_fetch_gems_by_state", side_effect=lambda s: (
            [{"gem_id": "gem-1"}] if s == "superseded" else []
        )):
            dir_mod.reconcile_terminal_gems(ledger)
        assert ledger["sig1"]["status"] == "dismissed_fp"


class TestProvenance:
    def test_provenance_carries_actual_model(self):
        mock_recorder = MagicMock()
        mock_get_recorder = MagicMock(return_value=mock_recorder)
        fake_module = MagicMock(get_recorder=mock_get_recorder)
        with patch("importlib.import_module", return_value=fake_module):
            dir_mod.emit_provenance(
                gem_id="gem-1", signature="sig1", clone_path="/srv/git/foo", model="qwen3.6-35b-a3b",
            )
        recorded = mock_recorder.record.call_args[0][0]
        assert recorded["model"] == "qwen3.6-35b-a3b"

    def test_provenance_mechanical_on_degrade(self):
        mock_recorder = MagicMock()
        mock_get_recorder = MagicMock(return_value=mock_recorder)
        fake_module = MagicMock(get_recorder=mock_get_recorder)
        with patch("importlib.import_module", return_value=fake_module):
            dir_mod.emit_provenance(
                gem_id="gem-1", signature="sig1", clone_path="/srv/git/foo", model="mechanical",
            )
        recorded = mock_recorder.record.call_args[0][0]
        assert recorded["model"] == "mechanical"

    def test_provenance_never_raises_and_is_non_blocking(self):
        with patch("importlib.import_module", side_effect=RuntimeError("zephyr down")):
            # must not raise
            dir_mod.emit_provenance(
                gem_id="gem-1", signature="sig1", clone_path="/srv/git/foo", model="mechanical",
            )

    def test_provenance_failure_does_not_block_deposit_success(self, ledger_file):
        """A recorder whose .record() raises must not prevent the gem deposit
        from being recorded as successful in the ledger, nor propagate out of
        propose_repair (DoD-9b) — emit_provenance's own try/except absorbs it."""
        ledger = {}
        mock_recorder = MagicMock()
        mock_recorder.record.side_effect = RuntimeError("zephyr unreachable")
        fake_module = MagicMock(get_recorder=MagicMock(return_value=mock_recorder))
        with (
            patch.object(dir_mod, "diagnose_finding", return_value=(dir_mod._degrade(_finding()), "mechanical")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-1"),
            patch("importlib.import_module", return_value=fake_module),
        ):
            action = dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=3, backing_units=[],
            )
        assert action == "deposited:new"
        sig = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        assert ledger[sig]["gem_id"] == "gem-1"


class TestStaleGemEscalation:
    def test_stale_open_entry_fires_once(self):
        import datetime as _dt
        old_ts = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=dir_mod.DEPLOY_INVENTORY_STALE_GEM_SECS + 3600)).isoformat()
        ledger = {"sig1": {"gem_id": "gem-1", "status": "open", "first_seen": old_ts,
                           "last_seen": old_ts, "times_seen": 3, "staled_at": None,
                           "clone_path": "/srv/git/foo", "finding_kind": "stale_behind_origin"}}
        with patch.object(dir_mod, "deposit_gem", return_value="gem-stale") as mock_deposit:
            fired = dir_mod.escalate_stale_gems(ledger)
        assert fired == ["sig1"]
        assert ledger["sig1"]["staled_at"] is not None
        assert mock_deposit.call_count == 1

    def test_second_pass_past_threshold_does_not_refire(self):
        import datetime as _dt
        old_ts = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=dir_mod.DEPLOY_INVENTORY_STALE_GEM_SECS + 3600)).isoformat()
        ledger = {"sig1": {"gem_id": "gem-1", "status": "open", "first_seen": old_ts,
                           "last_seen": old_ts, "times_seen": 3, "staled_at": "2026-08-01T00:00:00+00:00",
                           "clone_path": "/srv/git/foo", "finding_kind": "stale_behind_origin"}}
        with patch.object(dir_mod, "deposit_gem", return_value="gem-stale") as mock_deposit:
            fired = dir_mod.escalate_stale_gems(ledger)
        assert fired == []
        assert not mock_deposit.called

    def test_terminal_gem_never_triggers_staleness(self):
        import datetime as _dt
        old_ts = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=dir_mod.DEPLOY_INVENTORY_STALE_GEM_SECS + 3600)).isoformat()
        ledger = {"sig1": {"gem_id": "gem-1", "status": "acked", "first_seen": old_ts,
                           "last_seen": old_ts, "times_seen": 3, "staled_at": None,
                           "clone_path": "/srv/git/foo", "finding_kind": "stale_behind_origin"}}
        with patch.object(dir_mod, "deposit_gem") as mock_deposit:
            fired = dir_mod.escalate_stale_gems(ledger)
        assert fired == []
        assert not mock_deposit.called

    def test_fresh_open_entry_does_not_fire(self):
        ledger = {"sig1": {"gem_id": "gem-1", "status": "open", "first_seen": dir_mod._now_iso(),
                           "last_seen": dir_mod._now_iso(), "times_seen": 1, "staled_at": None,
                           "clone_path": "/srv/git/foo", "finding_kind": "stale_behind_origin"}}
        with patch.object(dir_mod, "deposit_gem") as mock_deposit:
            fired = dir_mod.escalate_stale_gems(ledger)
        assert fired == []
        assert not mock_deposit.called


class TestNoPushover:
    def test_module_never_calls_send_notification(self):
        """Grep-based assertion: the repair-proposer module contains no
        reference to agents_core.notify.send_notification anywhere (DoD-8)."""
        src = Path(dir_mod.__file__).read_text()
        assert "send_notification" not in src

    def test_pm_core_deploy_inventory_path_has_no_send_notification(self):
        import lapis_pm.pm_core as pm_core
        src = Path(pm_core.__file__).read_text()
        # Isolate the _reconcile_deploy_inventory function body
        start = src.index("def _reconcile_deploy_inventory()")
        end = src.index("\ndef _post_land_deploy_hook(")
        body = src[start:end]
        assert "send_notification" not in body
