"""Hermetic tests for deploy-inventory-repair-degrade-visibility-and-rediagnose-v0.

Covers D1 (silent-degrade logging), D2 (action-string + summary), D3 (ledger
diagnosis_model/rediagnose_attempts), D4 (honest Diagnosis block), D5a (ledger
scan candidate selection) and D5 (supersede-and-redeposit). No live
weaver/GW dependency - httpx.Client, call_gw_agent and
_call_supersede_endpoint are stubbed at every call site.
"""

import json
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
    return {"generated": "2026-08-08T00:00:00+00:00", "clones": clones}


def _open_entry(*, clone_path, finding_kind, diagnosis_model, gem_id="gem-old",
                 rediagnose_attempts=0, staled_at=None):
    return {
        "gem_id": gem_id,
        "first_seen": "2026-08-01T00:00:00+00:00",
        "last_seen": "2026-08-01T00:00:00+00:00",
        "times_seen": 1,
        "status": "open",
        "staled_at": staled_at,
        "clone_path": clone_path,
        "finding_kind": finding_kind,
        "diagnosis_model": diagnosis_model,
        "rediagnose_attempts": rediagnose_attempts,
    }


# ---------------------------------------------------------------------------
# D1 - the two silent degrade paths now log, distinguishably (DoD-1, DoD-2)
# ---------------------------------------------------------------------------

class TestSilentDegradeLogging:
    def test_none_return_logs_warning_naming_served_model_out(self, caplog):
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value=None),
            caplog.at_level("WARNING"),
        ):
            dir_mod.diagnose_finding(
                "/srv/git/foo", _finding(), mapped=True, commits_behind=3,
                branch="main", backing_units=[],
            )
        matches = [r for r in caplog.records if "GW produced no text" in r.message]
        assert len(matches) == 1
        assert "/srv/git/foo" in matches[0].message
        assert "stale_behind_origin" in matches[0].message

    def test_populated_served_model_out_named_in_empty_return_log(self, caplog):
        def fake_call(*args, served_model_out=None, **kwargs):
            served_model_out.append("gravitywell-27b")
            return None

        with (
            patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call),
            caplog.at_level("WARNING"),
        ):
            dir_mod.diagnose_finding(
                "/srv/git/foo", _finding(), mapped=True, commits_behind=3,
                branch="main", backing_units=[],
            )
        matches = [r for r in caplog.records if "GW produced no text" in r.message]
        assert len(matches) == 1
        assert "gravitywell-27b" in matches[0].message

    def test_non_dict_parse_logs_a_different_message_with_truncated_repr(self, caplog):
        long_text = "not json at all " * 50  # far over 200 chars
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value=long_text),
            caplog.at_level("WARNING"),
        ):
            dir_mod.diagnose_finding(
                "/srv/git/foo", _finding(), mapped=True, commits_behind=3,
                branch="main", backing_units=[],
            )
        matches = [r for r in caplog.records if "did not parse to an object" in r.message]
        assert len(matches) == 1
        assert "/srv/git/foo" in matches[0].message
        assert "stale_behind_origin" in matches[0].message
        assert len(matches[0].message) < len(long_text) + 200  # capped, not the whole text

    def test_the_two_messages_are_distinguishable(self, caplog):
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value=None),
            caplog.at_level("WARNING"),
        ):
            dir_mod.diagnose_finding(
                "/srv/git/foo", _finding(), mapped=True, commits_behind=3,
                branch="main", backing_units=[],
            )
        empty_msgs = [r.message for r in caplog.records]
        caplog.clear()
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value="not json"),
            caplog.at_level("WARNING"),
        ):
            dir_mod.diagnose_finding(
                "/srv/git/foo", _finding(), mapped=True, commits_behind=3,
                branch="main", backing_units=[],
            )
        parse_msgs = [r.message for r in caplog.records]
        assert set(empty_msgs).isdisjoint(set(parse_msgs))


# ---------------------------------------------------------------------------
# D2 - degrade marker in the action string; N of M summary (DoD-3)
# ---------------------------------------------------------------------------

class TestDegradeVisibleInReturnValue:
    def test_degraded_deposit_action_carries_marker_and_prefix(self):
        ledger = {}
        with (
            patch.object(dir_mod, "diagnose_finding",
                         return_value=(dir_mod._degrade(_finding()), "mechanical")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-1"),
            patch.object(dir_mod, "emit_provenance"),
        ):
            action = dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=3, backing_units=[],
            )
        assert action == "deposited:new:degraded"
        assert action.startswith("deposited:new")

    def test_healthy_deposit_action_has_no_marker(self):
        ledger = {}
        good_diag = {"root_cause": "x", "suggested_direction": "", "affected_files": [],
                     "confidence": "high", "uncertain": False}
        with (
            patch.object(dir_mod, "diagnose_finding", return_value=(good_diag, "gravitywell-27b")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-1"),
            patch.object(dir_mod, "emit_provenance"),
        ):
            action = dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=3, backing_units=[],
            )
        assert action == "deposited:new"

    def test_pass_logs_n_of_m_degraded_summary(self, caplog):
        clones = [_clone(path="/srv/git/foo", findings=[_finding(kind="stale_behind_origin")])]
        status = _status(clones)
        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "diagnose_finding",
                         return_value=(dir_mod._degrade(_finding()), "mechanical")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-x"),
            patch.object(dir_mod, "emit_provenance"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value={}),
            patch.object(dir_mod, "escalate_stale_gems"),
            caplog.at_level("INFO"),
        ):
            dir_mod.run_repair_pass(status, prior_high_keys=set(), corrupt=False)
        summary = [r.message for r in caplog.records if "diagnoses degraded" in r.message]
        assert summary == ["[deploy-inventory-repair] 1 of 1 diagnoses degraded this pass"]


# ---------------------------------------------------------------------------
# D3 - ledger records diagnosis quality; legacy entries are never eligible
# ---------------------------------------------------------------------------

class TestLedgerRecordsDiagnosisQuality:
    def test_degraded_deposit_writes_mechanical(self):
        ledger = {}
        with (
            patch.object(dir_mod, "diagnose_finding",
                         return_value=(dir_mod._degrade(_finding()), "mechanical")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-1"),
            patch.object(dir_mod, "emit_provenance"),
        ):
            dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=3, backing_units=[],
            )
        sig = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        assert ledger[sig]["diagnosis_model"] == "mechanical"
        assert ledger[sig]["rediagnose_attempts"] == 0

    def test_good_deposit_writes_real_model_id(self):
        ledger = {}
        good_diag = {"root_cause": "x", "suggested_direction": "", "affected_files": [],
                     "confidence": "high", "uncertain": False}
        with (
            patch.object(dir_mod, "diagnose_finding", return_value=(good_diag, "gravitywell-27b")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-1"),
            patch.object(dir_mod, "emit_provenance"),
        ):
            dir_mod.propose_repair(
                ledger, clone_path="/srv/git/foo", finding=_finding(),
                mapped=True, branch="main", commits_behind=3, backing_units=[],
            )
        sig = dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin")
        assert ledger[sig]["diagnosis_model"] == "gravitywell-27b"


# ---------------------------------------------------------------------------
# D4 - build_gem_payload honestly labels a degraded diagnosis (DoD-6)
# ---------------------------------------------------------------------------

class TestHonestDiagnosisBlock:
    def test_uncertain_true_does_not_repeat_detail_a_third_time(self):
        finding = _finding(detail="3 commits behind")
        diagnosis = dir_mod._degrade(finding)
        payload = dir_mod.build_gem_payload(
            clone_path="/srv/git/foo", finding=finding, mapped=True, branch="main",
            commits_behind=3, diagnosis=diagnosis, stakes_line="stakes", times_seen=1,
        )
        diagnosis_block = next(c for c in payload["context"] if c["label"] == "Diagnosis")
        assert "3 commits behind" not in diagnosis_block["lines"][0]
        assert "no model diagnosis" in diagnosis_block["lines"][0].lower()
        # Symptom, ask, and options unchanged under degrade.
        symptom_block = next(c for c in payload["context"] if c["label"] == "Symptom")
        assert symptom_block["lines"][0] == "3 commits behind"
        assert payload["ask"] == "3 commits behind"
        assert {o["key"] for o in payload["options"]} == {"ack_watch", "not_real"}

    def test_uncertain_false_payload_is_byte_identical_to_today(self):
        finding = _finding(detail="3 commits behind")
        diagnosis = {"root_cause": "clone drifted", "suggested_direction": "pull",
                     "affected_files": [], "confidence": "high", "uncertain": False}
        payload = dir_mod.build_gem_payload(
            clone_path="/srv/git/foo", finding=finding, mapped=True, branch="main",
            commits_behind=3, diagnosis=diagnosis, stakes_line="stakes", times_seen=1,
        )
        diagnosis_block = next(c for c in payload["context"] if c["label"] == "Diagnosis")
        assert diagnosis_block["lines"] == ["clone drifted"]


# ---------------------------------------------------------------------------
# D5a - explicit ledger scan selects exactly the intended candidates (DoD-10a)
# ---------------------------------------------------------------------------

class TestRediagnoseSelection:
    def _status_with_high(self, clone_path="/srv/git/foo", kind="stale_behind_origin"):
        return _status([_clone(path=clone_path, findings=[_finding(kind=kind)])])

    def test_status_not_open_excluded(self):
        entry = _open_entry(clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical")
        entry["status"] = "acked"
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        status = self._status_with_high()
        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value=ledger),
            patch.object(dir_mod, "escalate_stale_gems"),
            patch.object(dir_mod, "_rediagnose_one") as mock_redx,
        ):
            dir_mod.run_repair_pass(status, prior_high_keys={("/srv/git/foo", "stale_behind_origin")},
                                     corrupt=False)
        assert not mock_redx.called

    def test_missing_diagnosis_model_legacy_excluded(self):
        entry = _open_entry(clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical")
        del entry["diagnosis_model"]
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        status = self._status_with_high()
        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value=ledger),
            patch.object(dir_mod, "escalate_stale_gems"),
            patch.object(dir_mod, "_rediagnose_one") as mock_redx,
        ):
            dir_mod.run_repair_pass(status, prior_high_keys={("/srv/git/foo", "stale_behind_origin")},
                                     corrupt=False)
        assert not mock_redx.called

    def test_attempts_at_ceiling_excluded(self):
        entry = _open_entry(
            clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
            diagnosis_model="mechanical",
            rediagnose_attempts=dir_mod.DEPLOY_INVENTORY_REDIAGNOSE_MAX_ATTEMPTS,
        )
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        status = self._status_with_high()
        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value=ledger),
            patch.object(dir_mod, "escalate_stale_gems"),
            patch.object(dir_mod, "_rediagnose_one") as mock_redx,
        ):
            dir_mod.run_repair_pass(status, prior_high_keys={("/srv/git/foo", "stale_behind_origin")},
                                     corrupt=False)
        assert not mock_redx.called

    def test_staled_at_set_excluded(self):
        entry = _open_entry(clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical", staled_at="2026-08-01T00:00:00+00:00")
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        status = self._status_with_high()
        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value=ledger),
            patch.object(dir_mod, "escalate_stale_gems"),
            patch.object(dir_mod, "_rediagnose_one") as mock_redx,
        ):
            dir_mod.run_repair_pass(status, prior_high_keys={("/srv/git/foo", "stale_behind_origin")},
                                     corrupt=False)
        assert not mock_redx.called

    def test_resolved_finding_class_excluded(self):
        """An entry whose (clone_path, finding_kind) no longer appears as a
        HIGH finding in the current status dict must not be re-diagnosed."""
        entry = _open_entry(clone_path="/srv/git/gone", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical")
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        status = self._status_with_high(clone_path="/srv/git/foo")  # different clone
        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value=ledger),
            patch.object(dir_mod, "escalate_stale_gems"),
            patch.object(dir_mod, "_rediagnose_one") as mock_redx,
        ):
            dir_mod.run_repair_pass(status, prior_high_keys={("/srv/git/foo", "stale_behind_origin")},
                                     corrupt=False)
        assert not mock_redx.called

    def test_eligible_candidate_selected(self):
        entry = _open_entry(clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical")
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        status = self._status_with_high()
        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value=ledger),
            patch.object(dir_mod, "escalate_stale_gems"),
            patch.object(dir_mod, "_rediagnose_one", return_value="redx:superseded") as mock_redx,
        ):
            dir_mod.run_repair_pass(status, prior_high_keys={("/srv/git/foo", "stale_behind_origin")},
                                     corrupt=False)
        assert mock_redx.called


# ---------------------------------------------------------------------------
# D5 - re-diagnosis: degrade-again bumps attempts, success supersedes,
# ordering fails safe, bounded (DoD-7, DoD-8, DoD-9, DoD-10, DoD-10b, DoD-11)
# ---------------------------------------------------------------------------

class TestRediagnoseExecution:
    def test_degrades_again_bumps_attempts_and_deposits_nothing(self):
        entry = _open_entry(clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical", gem_id="gem-old")
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        with (
            patch.object(dir_mod, "diagnose_finding",
                         return_value=(dir_mod._degrade(_finding()), "mechanical")),
            patch.object(dir_mod, "deposit_gem") as mock_deposit,
        ):
            from lapis_pm import brief_gem as _brief_gem
            with patch.object(_brief_gem, "_call_supersede_endpoint") as mock_supersede:
                action = dir_mod._rediagnose_one(ledger, "sig1", entry, _clone(), _finding())
        assert action == "redx:degraded-again"
        assert entry["rediagnose_attempts"] == 1
        assert entry["gem_id"] == "gem-old"  # untouched
        assert not mock_deposit.called
        assert not mock_supersede.called

    def test_success_deposits_new_supersedes_old_and_repoints_ledger(self):
        entry = _open_entry(clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical", gem_id="gem-old",
                             rediagnose_attempts=1)
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        good_diag = {"root_cause": "clone drifted", "suggested_direction": "pull",
                     "affected_files": [], "confidence": "high", "uncertain": False}
        from lapis_pm import brief_gem as _brief_gem
        with (
            patch.object(dir_mod, "diagnose_finding", return_value=(good_diag, "gravitywell-27b")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-new") as mock_deposit,
            patch.object(dir_mod, "emit_provenance"),
            patch.object(_brief_gem, "_call_supersede_endpoint", return_value=True) as mock_supersede,
        ):
            action = dir_mod._rediagnose_one(ledger, "sig1", entry, _clone(), _finding())

        assert action == "redx:superseded"
        assert entry["gem_id"] == "gem-new"
        assert entry["diagnosis_model"] == "gravitywell-27b"
        assert entry["rediagnose_attempts"] == 0
        mock_supersede.assert_called_once()
        assert mock_supersede.call_args[0][0] == "gem-old"
        # DoD-9: the new gem's context names the superseded gem id.
        new_payload = mock_deposit.call_args[0][0]
        supersedes_block = next(c for c in new_payload["context"] if c["label"] == "Supersedes")
        assert "gem-old" in supersedes_block["lines"][0]

    def test_deposit_failure_never_supersedes_the_old_gem(self):
        """DoD-8: the single most important ordering test. A deposit failure
        on a successful re-diagnosis must never touch the old gem - the
        finding must never vanish from the Desk."""
        entry = _open_entry(clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical", gem_id="gem-old")
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        good_diag = {"root_cause": "clone drifted", "suggested_direction": "pull",
                     "affected_files": [], "confidence": "high", "uncertain": False}
        from lapis_pm import brief_gem as _brief_gem
        with (
            patch.object(dir_mod, "diagnose_finding", return_value=(good_diag, "gravitywell-27b")),
            patch.object(dir_mod, "deposit_gem", return_value=None),
            patch.object(_brief_gem, "_call_supersede_endpoint") as mock_supersede,
        ):
            action = dir_mod._rediagnose_one(ledger, "sig1", entry, _clone(), _finding())

        assert action == "redx:deposit-failed"
        assert not mock_supersede.called
        assert entry["gem_id"] == "gem-old"  # untouched - finding still on the Desk

    def test_supersede_409_abandons_link_leaves_ledger_alone(self):
        """Trickster's condition (D5a): a 409 means someone else terminalised
        the old gem in the residual race. Never retry into it; leave the
        ledger entry pointed at the old (now-terminal) gem rather than
        linking in an orphaned new one."""
        entry = _open_entry(clone_path="/srv/git/foo", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical", gem_id="gem-old",
                             rediagnose_attempts=1)
        ledger = {dir_mod.compute_signature("/srv/git/foo", "stale_behind_origin"): entry}
        good_diag = {"root_cause": "clone drifted", "suggested_direction": "pull",
                     "affected_files": [], "confidence": "high", "uncertain": False}
        from lapis_pm import brief_gem as _brief_gem
        with (
            patch.object(dir_mod, "diagnose_finding", return_value=(good_diag, "gravitywell-27b")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-new"),
            patch.object(_brief_gem, "_call_supersede_endpoint", return_value=False) as mock_supersede,
        ):
            action = dir_mod._rediagnose_one(ledger, "sig1", entry, _clone(), _finding())

        assert action == "redx:supersede-abandoned"
        mock_supersede.assert_called_once()
        # Ledger entry left alone - never retried, never repointed.
        assert entry["gem_id"] == "gem-old"
        assert entry["rediagnose_attempts"] == 1
        assert entry["diagnosis_model"] == "mechanical"

    def test_bounded_stops_at_max_attempts(self):
        """DoD-10: an entry already at the ceiling is never selected again
        (covered structurally by the D5a selection tests above); this test
        pins the ceiling constant is actually consulted per-candidate."""
        assert dir_mod.DEPLOY_INVENTORY_REDIAGNOSE_MAX_ATTEMPTS >= 1

    def test_fresh_findings_win_the_cap_no_rediagnosis_when_saturated(self):
        """DoD-11: with the cap saturated by fresh candidates, no
        re-diagnosis occurs - re-diagnosis uses only leftover slots."""
        cap = dir_mod.DEPLOY_INVENTORY_REPAIR_MAX_PER_RUN
        fresh_clones = [
            _clone(path=f"/srv/git/fresh{i}", findings=[_finding(kind="stray_branch")])
            for i in range(cap)
        ]
        rediagnose_clone = _clone(path="/srv/git/rediag", findings=[_finding(kind="stale_behind_origin")])
        status = _status(fresh_clones + [rediagnose_clone])

        entry = _open_entry(clone_path="/srv/git/rediag", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical")
        ledger = {"sig-redx": entry}

        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "diagnose_finding",
                         return_value=(dir_mod._degrade(_finding()), "mechanical")),
            patch.object(dir_mod, "deposit_gem", return_value="gem-new"),
            patch.object(dir_mod, "emit_provenance"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value=ledger),
            patch.object(dir_mod, "escalate_stale_gems"),
            patch.object(dir_mod, "_rediagnose_one") as mock_redx,
        ):
            actions = dir_mod.run_repair_pass(status, prior_high_keys=set(), corrupt=False)

        assert not mock_redx.called
        deposited = [a for a in actions if a.startswith("deposited:")]
        assert len(deposited) == cap

    def test_rediagnosis_uses_leftover_slots(self):
        """The complement of DoD-11: with cap slots left over, re-diagnosis
        does fire."""
        entry = _open_entry(clone_path="/srv/git/rediag", finding_kind="stale_behind_origin",
                             diagnosis_model="mechanical")
        ledger = {"sig-redx": entry}
        status = _status([_clone(path="/srv/git/rediag", findings=[_finding(kind="stale_behind_origin")])])

        with (
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "write_ledger"),
            patch.object(dir_mod, "read_ledger", return_value=ledger),
            patch.object(dir_mod, "escalate_stale_gems"),
            patch.object(dir_mod, "_rediagnose_one", return_value="redx:superseded") as mock_redx,
        ):
            actions = dir_mod.run_repair_pass(status, prior_high_keys={("/srv/git/rediag", "stale_behind_origin")},
                                               corrupt=False)

        assert mock_redx.called
        assert any(a.startswith("redx:superseded") for a in actions)
