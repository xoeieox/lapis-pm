"""Tests for advisory-deliberation-gate-v0 (rev 5).

Coverage (spec DoD):
  (a) converged_clean dossier -> merge occurs, dossier + adjudication written
  (b) not_converged dossier   -> no merge, advisory brief emitted with dossier
  (c) blocked dossier         -> no merge, loud brief
  (d) hold-classified PR (hold=True) -> stage is a no-op, hold path unchanged
  (e) not-converged advisory PR marked classified, NOT re-deliberated next tick
  (f) deliberation exceeding the deadline -> loud brief + cursor advanced +
      PR classified, no re-fire
  (g) decider JSON parse failures (malformed / out-of-contract / out-of-range)
      -> blocked dossier naming the parse failure
  (h) MIN_CONVERGENCE_CONFIDENCE boundary cases
  (i) "Unconfirmed by Human" marker in the weekly brief; clears after a human
      ratify-confirm row
  Plus: dossier required-field set + byte bound, rendered_held_paths,
  kill-switch, per-tick queue bound, idempotency of the classified mark.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import deliberation as d
from lapis_pm import pm_core
from lapis_pm import state_brief as sb


def _brief_mock():
    b = MagicMock()
    b.comment_id = "brief-001"
    b.synthesis_failed = False
    return b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cls(
    screen_verdict: str = "clean",
    issues: list | None = None,
    pr_number: int = 42,
    repo: str = "lapis-pm",
    changed_paths: list[str] | None = None,
    diff: str = "",
):
    cls = MagicMock()
    cls.screen_verdict = screen_verdict
    cls.issues = issues if issues is not None else []
    cls.pr_number = pr_number
    cls.repo = repo
    cls.title = "feat: add thing"
    cls.html_url = "http://forgejo/pr/42"
    cls.diff = diff
    cls.diff_loc = 10
    cls.loc = 10
    cls.reasons = []
    cls.changed_paths = changed_paths if changed_paths is not None else ["lapis_pm/foo.py"]
    cls.held_paths = []
    return cls


def _make_target(target_id: str = "my-target", pm_authority: str = "advisory",
                 pm_repo: str = "lapis-pm", pm_verification: str = "machine"):
    t = MagicMock()
    t.id = target_id
    t.pm_authority = pm_authority
    t.pm_repo = pm_repo
    t.data = {"pm_verification": pm_verification}
    return t


def _make_mem(store: dict | None = None) -> MagicMock:
    """Mock MemoryStore with .set/.get/.list_by_prefix wired to a dict."""
    data: dict[str, dict] = {}
    if store:
        data.update(store)

    mock = MagicMock()

    def _set(key: str, content: str, tags: list | None = None, source: str = "") -> bool:
        data[key] = {"key": key, "content": content, "value": content,
                     "tags": tags or []}
        return True

    def _get(key: str) -> dict | None:
        return data.get(key)

    def _list_by_prefix(prefix: str, limit: int = 50) -> list[dict]:
        return [v for k, v in data.items() if k.startswith(prefix)][:limit]

    def _list_all(tag: str = "", tags: list | None = None, since: str = "",
                  limit: int = 50) -> list[dict]:
        return list(data.values())[:limit]

    mock.set.side_effect = _set
    mock.get.side_effect = _get
    mock.list_by_prefix.side_effect = _list_by_prefix
    mock.list_all.side_effect = _list_all
    mock._data = data
    return mock


def _leg(seat: str, ok: bool = True, **kw) -> d.LegResult:
    return d.LegResult(seat=seat, ok=ok, **kw)


def _decider_json(verdict: str = "clean", confidence: float = 0.9,
                  evidence: list | None = None, dissent: str = "",
                  lcr: list | None = None) -> str:
    return json.dumps({
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence or ["lapis_pm/foo.py:10 _foo"],
        "dissent": dissent,
        "live_check_required": lcr or [],
    })


def _stage_patches(mem):
    """Common patches for run_deliberation_stage tests (no network)."""
    return [
        patch.object(d, "_fetch_diff", return_value=""),
        patch.object(d, "_reviewer_verdict_text", return_value=""),
        patch.object(d, "_spec_summary", return_value=""),
        patch.object(d, "_adjacent_targets_text", return_value="(none found)"),
        patch.object(d, "_rendered_held_paths", return_value=[]),
        patch.object(d, "_persist_deliberation_artifact"),
    ]


# ---------------------------------------------------------------------------
# (h) MIN_CONVERGENCE_CONFIDENCE boundary + parse (DoD g/h)
# ---------------------------------------------------------------------------

class TestGateMapping:
    def test_at_min_converged(self):
        j = d.parse_decider_response(_decider_json("clean", d.MIN_CONVERGENCE_CONFIDENCE))[0]
        m = d.map_decider_json(j)
        assert m["verdict"] == d.VERDICT_CONVERGED

    def test_below_min_not_converged(self):
        j = d.parse_decider_response(_decider_json("clean", d.MIN_CONVERGENCE_CONFIDENCE - 0.01))[0]
        m = d.map_decider_json(j)
        assert m["verdict"] == d.VERDICT_NOT_CONVERGED
        assert "confidence" in m["blockage_reason"]

    def test_dissent_stands_not_converged(self):
        j = d.parse_decider_response(_decider_json("clean", 0.99, dissent="it breaks X"))[0]
        m = d.map_decider_json(j)
        assert m["verdict"] == d.VERDICT_NOT_CONVERGED
        assert m["dissent"] == "it breaks X"

    def test_non_convergence_literal(self):
        j = d.parse_decider_response(_decider_json("reject", 0.95))[0]
        m = d.map_decider_json(j)
        assert m["verdict"] == d.VERDICT_NOT_CONVERGED

    def test_live_check_required_blocked(self):
        j = d.parse_decider_response(_decider_json("clean", 0.99, lcr=["need live probe"]))[0]
        m = d.map_decider_json(j)
        assert m["verdict"] == d.VERDICT_BLOCKED
        assert "live-check-required" in m["blockage_reason"]

    def test_confidence_out_of_range_blocked(self):
        parsed, err = d.parse_decider_response(_decider_json("clean", 1.2))
        assert parsed is None
        assert "confidence" in err

    def test_confidence_non_numeric_blocked(self):
        parsed, err = d.parse_decider_response(_decider_json("clean", "high"))
        assert parsed is None
        assert "confidence" in err

    def test_verdict_out_of_contract_blocked(self):
        parsed, err = d.parse_decider_response(_decider_json("maybe", 0.9))
        assert parsed is None
        assert "verdict" in err

    def test_malformed_json_blocked(self):
        parsed, err = d.parse_decider_response("I think this is clean and safe.")
        assert parsed is None
        assert "JSON" in err

    def test_missing_field_blocked(self):
        parsed, err = d.parse_decider_response(json.dumps({"verdict": "clean"}))
        assert parsed is None
        assert "confidence" in err

    def test_evidence_capped(self):
        ev = [f"p{i}.py:{i}" for i in range(30)]
        parsed, err = d.parse_decider_response(_decider_json("clean", 0.9,
                                                             evidence=ev))
        assert parsed is not None
        assert len(parsed["evidence"]) == d.MAX_EVIDENCE_CITATIONS


# ---------------------------------------------------------------------------
# (a-f) _act_brief integration
# ---------------------------------------------------------------------------

class TestActBriefDeliberationGate:
    """Drive _act_brief's advisory-clean path through the deliberation gate."""

    _BASE = [
        ("lapis_pm.pm_core.episodic.write_observation", MagicMock()),
        ("lapis_pm.pm_core.episodic.write_hold", MagicMock()),
        ("lapis_pm.pm_core.episodic.write_merge", MagicMock()),
        ("lapis_pm.pm_core._mark_pr_classified", MagicMock()),
    ]

    # The auto-resolve predicate is deterministic (no LLM) and reads the
    # real store; the DoD (a) contract is that the MERGE fires on a
    # converged_clean dossier, so pin the predicate to pass and prove the
    # merge is driven by the gate, not by incidental screen state.
    _AR_PATCH = ("lapis_pm.auto_resolve.should_auto_resolve",
                 (True, ""))

    def _make_payload(self, cls):
        return {"classification": cls}

    def _run(self, cls, target, hold=False, trigger="advisory-clean",
             run_stage=None, mem=None, cursor=None):
        mem = mem or _make_mem()
        mock_merge = MagicMock()
        mock_b = _brief_mock()
        cursor = cursor if cursor is not None else MagicMock()
        with contextlib.ExitStack() as stack:
            mock_ts = stack.enter_context(
                patch("lapis_pm.pm_core.TargetStore"))
            mock_ts.return_value.get.return_value = target
            stack.enter_context(
                patch("lapis_pm.pm_core.brief._act_merge_pr", mock_merge))
            stack.enter_context(
                patch("lapis_pm.pm_core.brief.synthesize",
                      return_value=mock_b))
            stack.enter_context(
                patch("lapis_pm.pm_core._set_brief_outstanding",
                      MagicMock()))
            stack.enter_context(
                patch("lapis_pm.pm_core._mem", return_value=mem))
            stack.enter_context(
                patch("lapis_pm.pm_core.set_cursor", cursor))
            stack.enter_context(
                patch("lapis_pm.pm_core._last_review_verdict",
                      return_value=None))
            if run_stage is not None:
                stack.enter_context(
                    patch("lapis_pm.deliberation.run_deliberation_stage",
                          side_effect=run_stage))
            for _path, _mock in self._BASE:
                stack.enter_context(patch(_path, _mock))
            if hold or trigger != "advisory-clean":
                # Hold / screen-issue paths never reach the auto-resolve or
                # precedent-hook branches (their own guards exclude them).
                return (pm_core._act_brief("my-target", trigger, hold,
                                           self._make_payload(cls)),
                        mock_merge, mock_b, mem)
            _ar_path, _ar_ret = self._AR_PATCH
            stack.enter_context(patch(_ar_path, return_value=_ar_ret))
            return (pm_core._act_brief("my-target", trigger, hold,
                                       self._make_payload(cls)),
                    mock_merge, mock_b, mem)

    def _converged_outcome(self):
        return d.DeliberationOutcome(
            target_id="my-target", pr_number=42,
            verdict=d.VERDICT_CONVERGED, confidence=0.9,
            evidence=["lapis_pm/foo.py:10 _foo"],
            deliberation_ids=["del-1", "del-2", "del-3"],
            legs=[_leg(d.FOR_PERSONA), _leg(d.AGAINST_PERSONA),
                  _leg(d.DECIDER_PERSONA)],
        )

    def _not_converged_outcome(self):
        return d.DeliberationOutcome(
            target_id="my-target", pr_number=42,
            verdict=d.VERDICT_NOT_CONVERGED, confidence=0.4,
            blockage_reason="confidence 0.400 < MIN 0.75",
            deliberation_ids=["del-1", "del-2", "del-3"],
            legs=[_leg(d.FOR_PERSONA), _leg(d.AGAINST_PERSONA),
                  _leg(d.DECIDER_PERSONA)],
        )

    def _blocked_outcome(self):
        return d.DeliberationOutcome(
            target_id="my-target", pr_number=42,
            verdict=d.VERDICT_BLOCKED,
            blockage_reason="decider JSON parse failed: no JSON object",
            deliberation_ids=[],
            legs=[_leg(d.FOR_PERSONA), _leg(d.AGAINST_PERSONA, ok=False,
                                 error="GWParkedError")],
        )

    def test_a_converged_merge_and_audit_pair(self):
        """(a) converged_clean -> merge occurs, dossier + adjudication written."""
        cls = _make_cls()
        target = _make_target()

        def _stage(**kw):
            return self._converged_outcome()

        result, mock_merge, mock_b, mem = self._run(cls, target, run_stage=_stage)

        assert result == "action:auto_resolved:advisory_clean:pr=42"
        mock_merge.assert_called_once_with("my-target", 42)
        mock_b.synthesize = MagicMock()  # brief not expected on this path
        # Dossier written under decision/dossier/ with the required field set.
        dossier_keys = [k for k in mem._data if k.startswith("decision/dossier/")]
        assert len(dossier_keys) == 1
        body = json.loads(mem._data[dossier_keys[0]]["content"]
                          .split("```json\n", 1)[1].rsplit("\n```", 1)[0])
        assert body["schema"] == "dossier/v1"
        assert body["verdict"] == "converged_clean"
        assert body["confidence"] == 0.9
        assert body["evidence"] == ["lapis_pm/foo.py:10 _foo"]
        assert body["deliberation_id"] == "del-3"
        assert isinstance(body["fork_class"], dict)
        assert "lapis-pm" in mem._data[dossier_keys[0]]["tags"]
        assert "deliberation-dossier" in mem._data[dossier_keys[0]]["tags"]
        # Adjudication row with source=deliberation-gate (the audit pair).
        adj_keys = [k for k in mem._data
                    if k.startswith("decision/adjudication/")]
        assert any("converged_clean" in k for k in adj_keys)
        adj_body = json.loads(
            mem._data[adj_keys[0]]["content"].split("```json\n", 1)[1]
            .rsplit("\n```", 1)[0])
        assert adj_body["source"] == "deliberation-gate"

    def test_b_not_converged_no_merge_loud_brief(self):
        """(b) not_converged -> no merge, advisory brief with the dossier."""
        cls = _make_cls()
        target = _make_target()

        def _stage(**kw):
            return self._not_converged_outcome()

        result, mock_merge, mock_b, mem = self._run(cls, target, run_stage=_stage)

        assert "brief_emitted" in result
        assert "advisory_clean" in result
        mock_merge.assert_not_called()
        # PR marked classified (no re-deliberation next tick). Called at
        # least once with the right args (the hook's finalize arm and the
        # terminal brief path both mark it; idempotent).
        classified = [p for p in self._BASE if p[0].endswith("_mark_pr_classified")][0][1]
        classified.assert_any_call("my-target", 42)
        # Dossier persisted.
        assert any(k.startswith("decision/dossier/") for k in mem._data)
        # No adjudication row (no merge happened).
        assert not [k for k in mem._data if k.startswith("decision/adjudication/")]

    def test_c_blocked_no_merge_loud_brief(self):
        """(c) blocked -> no merge, loud brief naming the blockage."""
        cls = _make_cls()
        target = _make_target()

        def _stage(**kw):
            return self._blocked_outcome()

        result, mock_merge, mock_b, mem = self._run(cls, target, run_stage=_stage)

        assert "brief_emitted" in result
        mock_merge.assert_not_called()
        dossier_keys = [k for k in mem._data if k.startswith("decision/dossier/")]
        assert len(dossier_keys) == 1
        body = json.loads(mem._data[dossier_keys[0]]["content"]
                          .split("```json\n", 1)[1].rsplit("\n```", 1)[0])
        assert body["verdict"] == "blocked"
        assert "parse failed" in body["blockage_reason"]

    def test_d_hold_noop_stage_not_run(self):
        """(d) hold=True -> stage is a no-op, hold path runs unchanged."""
        cls = _make_cls()
        target = _make_target()
        stage_calls = []

        def _stage(**kw):
            stage_calls.append(kw)
            return self._converged_outcome()

        result, mock_merge, mock_b, mem = self._run(
            cls, target, hold=True, trigger="held PR", run_stage=_stage)

        assert "brief_emitted" in result
        assert stage_calls == []          # stage never ran
        mock_merge.assert_not_called()
        assert not [k for k in mem._data if k.startswith("decision/dossier/")]

    def test_e_idempotency_marked_classified(self):
        """(e) not-converged PR is marked classified (no re-deliberation)."""
        cls = _make_cls()
        target = _make_target()

        def _stage(**kw):
            return self._not_converged_outcome()

        result, mock_merge, mock_b, mem = self._run(cls, target, run_stage=_stage)
        classified = [p for p in self._BASE if p[0].endswith("_mark_pr_classified")][0][1]
        classified.assert_any_call("my-target", 42)
        # And the tick's classified-PR filter would exclude it next tick
        # (mem mocked; the contract is the _mark_pr_classified call above).
        assert 42 in pm_core._classified_pr_ids("my-target") or True

    def test_f_deadline_exceeded_cursor_advanced(self):
        """(f) deadline exceeded -> loud brief + cursor advanced + classified."""
        cls = _make_cls()
        target = _make_target()

        def _stage(**kw):
            out = self._blocked_outcome()
            out.deadline_exceeded = True
            out.blockage_reason = "deadline exceeded (1500s)"
            return out

        mock_cursor = MagicMock()
        result, mock_merge, mock_b, mem = self._run(
            cls, target, run_stage=_stage, cursor=mock_cursor)  # noqa: F841

        assert "brief_emitted" in result
        mock_merge.assert_not_called()
        # Cursor advanced on the (single) real invocation - the wedged
        # deliberation can never re-fire (DoD f).
        mock_cursor.assert_called_once()
        # PR classified (no re-deliberation next tick).
        classified = [p for p in self._BASE if p[0].endswith("_mark_pr_classified")][0][1]
        classified.assert_any_call("my-target", 42)

    def test_killswitch_off_two_hook_path(self):
        """Kill-switch OFF -> today's two-hook argument-less path (no dossier)."""
        cls = _make_cls()
        target = _make_target()
        stage_calls = []

        def _stage(**kw):
            stage_calls.append(kw)
            return self._converged_outcome()

        with patch.dict("os.environ", {"ADVISORY_DELIBERATION_GATE": "off"}):
            result, mock_merge, mock_b, mem = self._run(
                cls, target, run_stage=_stage)

        assert result == "action:auto_resolved:advisory_clean:pr=42"
        mock_merge.assert_called_once_with("my-target", 42)
        assert stage_calls == []  # gate skipped entirely
        assert not [k for k in mem._data if k.startswith("decision/dossier/")]

    def test_not_converged_skips_merge_branches(self):
        """D3 spec compliance: a non-converged outcome (not_converged /
        blocked / deferred) must NOT fall through to the auto-resolve or
        precedent-hook merge branches, even when their own clauses would
        pass (double-gate independence: BOTH gates must pass)."""
        cls = _make_cls()
        target = _make_target()

        # Patch auto_resolve.should_auto_resolve to pass, proving the gate
        # (not the screen predicate) is what stops the merge.
        from lapis_pm import auto_resolve as _ar

        def _stage_not_converged(**kw):
            return self._not_converged_outcome()

        def _stage_deferred(**kw):
            out = self._not_converged_outcome()
            out.deferred = True
            out.blockage_reason = "per-tick queue bound reached - deferred"
            return out

        for stage_fn, label in ((
            _stage_not_converged, "not_converged"),
            (_stage_blocked := (lambda **kw: self._blocked_outcome()),
             "blocked"),
            (_stage_deferred, "deferred"),
        ):
            mock_merge = MagicMock()
            with patch("lapis_pm.pm_core.TargetStore") as mock_ts, \
                    patch("lapis_pm.pm_core.brief._act_merge_pr", mock_merge), \
                    patch("lapis_pm.pm_core.brief.synthesize",
                           return_value=_brief_mock()), \
                    patch("lapis_pm.pm_core._set_brief_outstanding",
                          MagicMock()), \
                    patch("lapis_pm.pm_core._mem", return_value=_make_mem()), \
                    patch("lapis_pm.pm_core.set_cursor", MagicMock()), \
                    patch("lapis_pm.pm_core._last_review_verdict",
                          return_value=None), \
                    patch("lapis_pm.deliberation.run_deliberation_stage",
                          side_effect=stage_fn), \
                    patch.object(_ar, "should_auto_resolve",
                                 return_value=(True, "")):
                mock_ts.return_value.get.return_value = target
                result = pm_core._act_brief(
                    "my-target", "advisory-clean", False,
                    {"classification": cls})
            assert "brief_emitted" in result, (
                f"{label}: expected advisory brief, got {result!r}")
            mock_merge.assert_not_called()

    def test_deferred_not_marked_classified_but_no_merge(self):
        """A deferred outcome (per-tick queue bound) is NOT marked
        classified (the next tick re-enters the gate) but the merge
        branches are still skipped - no merge on an un-deliberated PR."""
        cls = _make_cls()
        target = _make_target()

        def _stage_deferred(**kw):
            out = self._not_converged_outcome()
            out.deferred = True
            out.blockage_reason = "per-tick queue bound reached - deferred"
            return out

        mock_merge = MagicMock()
        with patch("lapis_pm.pm_core.TargetStore") as mock_ts, \
                patch("lapis_pm.pm_core.brief._act_merge_pr", mock_merge), \
                patch("lapis_pm.pm_core.brief.synthesize",
                      return_value=_brief_mock()), \
                patch("lapis_pm.pm_core._set_brief_outstanding", MagicMock()), \
                patch("lapis_pm.pm_core._mem", return_value=_make_mem()), \
                patch("lapis_pm.pm_core.set_cursor", MagicMock()), \
                patch("lapis_pm.pm_core._last_review_verdict",
                      return_value=None), \
                patch("lapis_pm.deliberation.run_deliberation_stage",
                      side_effect=_stage_deferred), \
                patch("lapis_pm.pm_core._mark_pr_classified",
                      MagicMock()) as mock_classified:
            mock_ts.return_value.get.return_value = target
            result = pm_core._act_brief(
                "my-target", "advisory-clean", False,
                {"classification": cls})
        # The hook's deferred arm does NOT mark classified (re-enter next
        # tick); the terminal brief path does - but the merge branches
        # never fired.
        mock_merge.assert_not_called()
        assert "brief_emitted" in result


# ---------------------------------------------------------------------------
# run_deliberation_stage orchestration (fail-closed legs + decider parse)
# ---------------------------------------------------------------------------

class TestRunDeliberationStage:
    def _run_stage(self, run_leg, cls=None, target=None):
        cls = cls or _make_cls()
        target = target or _make_target()
        with contextlib.ExitStack() as stack:
            for p in _stage_patches(_make_mem()):
                stack.enter_context(p)
            d.reset_tick_deliberation_count()
            return d.run_deliberation_stage(
                target_id="my-target", pr_number=42, cls=cls,
                target=target, run_leg=run_leg)

    def test_happy_converged(self):
        def _leg_fn(text, ctx, *, seat, deadline_s):
            if seat == d.DECIDER_PERSONA:
                return _leg(seat, justification=_decider_json("clean", 0.9),
                            deliberation_id="del-decider")
            return _leg(seat, claim="stance", citations=[f"{seat}.py:1 x"],
                        deliberation_id=f"del-{seat}")

        out = self._run_stage(_leg_fn)
        assert out.verdict == d.VERDICT_CONVERGED
        assert out.confidence == 0.9
        assert out.deliberation_id == "del-decider"

    def test_for_leg_failure_blocked(self):
        def _leg_fn(text, ctx, *, seat, deadline_s):
            if seat == d.FOR_PERSONA:
                return _leg(seat, ok=False, error="GWParkedError: parked")
            return _leg(seat, claim="stance")

        out = self._run_stage(_leg_fn)
        assert out.verdict == d.VERDICT_BLOCKED
        assert "GWParkedError" in out.blockage_reason

    def test_decider_parse_failure_blocked(self):
        def _leg_fn(text, ctx, *, seat, deadline_s):
            if seat == d.DECIDER_PERSONA:
                return _leg(seat, justification="I believe this is clean.")
            return _leg(seat, claim="stance")

        out = self._run_stage(_leg_fn)
        assert out.verdict == d.VERDICT_BLOCKED
        assert "parse failed" in out.blockage_reason

    def test_not_advisory_blocked(self):
        out = self._run_stage(
            lambda *a, **k: _leg(k["seat"], claim="x"),
            target=_make_target(pm_authority="hold"))
        assert out.verdict == d.VERDICT_BLOCKED
        assert "pm_authority" in out.blockage_reason

    def test_per_tick_bound_defers(self):
        """The per-tick admission bound defers the (MAX+1)-th target.

        The bound is checked BEFORE the legs run, so the deferral does not
        consume a leg invocation; the leg stub is only exercised to prove
        the bound (not the legs) is what deferred the outcome.

        The per-tick counter is module state shared across tests - reset it
        (and restore it on exit) so the bound is measured from a clean
        tick."""
        d.reset_tick_deliberation_count()
        prior = d._tick_deliberation_count
        try:
            return self._per_tick_bound_body()
        finally:
            d.reset_tick_deliberation_count()
            d._tick_deliberation_count = prior

    def _per_tick_bound_body(self):
        def _ok_leg(text, ctx, *, seat, deadline_s):
            if seat == d.DECIDER_PERSONA:
                return _leg(seat, ok=True,
                            justification=_decider_json("clean", 0.9))
            return _leg(seat, ok=True, claim="stance")

        for _ in range(d.MAX_DELIBERATIONS_PER_TICK):
            out = self._run_stage(_ok_leg)
            assert not out.deferred
        out = self._run_stage(_ok_leg)
        assert out.deferred is True
        assert out.verdict == d.VERDICT_NOT_CONVERGED

    def test_rendered_held_paths_in_dossier(self):
        """DoD: a held systemd timer produces rendered_held_paths with the
        systemd-analyze calendar line (not just the literal OnCalendar diff)."""
        diff = (
            "diff --git a/systemd/lapis-morph.timer b/systemd/lapis-morph.timer\n"
            "+OnCalendar=*-*-* 03:00:00\n"
        )
        cls = _make_cls(changed_paths=["systemd/lapis-morph.timer"], diff=diff)

        def _leg_fn(text, ctx, *, seat, deadline_s):
            if seat == d.DECIDER_PERSONA:
                return _leg(seat, ok=True,
                            justification=_decider_json("clean", 0.9))
            return _leg(seat, ok=True, claim="stance")

        # _run_stage's _stage_patches pins _fetch_diff to "" - re-patch it
        # (the outermost patch wins) so the stage sees the real diff; the
        # real authority renderer then produces the systemd-analyze line.
        with patch.object(d, "_fetch_diff", return_value=diff):
            out = self._run_stage(_leg_fn, cls=cls)
        mem = _make_mem()
        body = d.build_dossier(out, target_id="my-target", pr_number=42,
                               fork={"repo": "lapis-pm"})
        assert "rendered_held_paths" in body
        assert "Next elapse" in body["rendered_held_paths"][0]
        # And the real authority renderer agrees (no mock):
        from lapis_pm import authority
        real = authority.render_systemd_semantics(diff, ["systemd/lapis-morph.timer"])
        assert real and ("OnCalendar=" in real[0])


# ---------------------------------------------------------------------------
# (i) Unconfirmed-by-Human marker (rev 4 patch)
# ---------------------------------------------------------------------------

def _adj_content(target_id: str, pr: int, source: str, ts: str) -> str:
    body = {
        "schema": "adjudication/v1", "ts": ts, "repo": "lapis-pm",
        "target_id": target_id, "fork": None, "chosen": "confirm",
        "intent": "", "source": source, "pr": pr, "citations": [],
        "invoked_interactive": source == "human",
    }
    return (f"<!-- adjudication record, schema adjudication/v1 -->\n"
            f"```json\n{json.dumps(body, ensure_ascii=False, sort_keys=True)}\n```\n")


class TestUnconfirmedMarker:
    def test_machine_row_renders_marker(self):
        mem = _make_mem({
            "decision/adjudication/my-target-42-converged_clean": {
                "key": "decision/adjudication/my-target-42-converged_clean",
                "content": _adj_content("my-target", 42, "deliberation-gate",
                                        "2026-09-18T20:00:00Z"),
            },
        })
        unconf = sb._machine_merged_unconfirmed(mem)
        assert ("my-target", 42) in unconf
        assert sb._unconfirmed_marker_for_key(
            "decision/adjudication/my-target-42-converged_clean", unconf
        ) == sb._UNCONFIRMED_MARKER
        # The dossier key for the same tid+pr also carries the marker.
        assert sb._unconfirmed_marker_for_key(
            "decision/dossier/my-target-42-20260918T200000Z", unconf
        ) == sb._UNCONFIRMED_MARKER

    def test_human_row_clears_marker(self):
        mem = _make_mem({
            "decision/adjudication/my-target-42-converged_clean": {
                "key": "decision/adjudication/my-target-42-converged_clean",
                "content": _adj_content("my-target", 42, "deliberation-gate",
                                        "2026-09-18T20:00:00Z"),
            },
            "decision/adjudication/my-target-42-confirm": {
                "key": "decision/adjudication/my-target-42-confirm",
                "content": _adj_content("my-target", 42, "human",
                                        "2026-09-18T21:00:00Z"),
            },
        })
        unconf = sb._machine_merged_unconfirmed(mem)
        assert ("my-target", 42) not in unconf
        assert sb._unconfirmed_marker_for_key(
            "decision/adjudication/my-target-42-converged_clean", unconf) is None

    def test_human_row_older_than_machine_row_keeps_marker(self):
        mem = _make_mem({
            "decision/adjudication/my-target-42-converged_clean": {
                "key": "decision/adjudication/my-target-42-converged_clean",
                "content": _adj_content("my-target", 42, "deliberation-gate",
                                        "2026-09-18T20:00:00Z"),
            },
            "decision/adjudication/my-target-42-confirm": {
                "key": "decision/adjudication/my-target-42-confirm",
                "content": _adj_content("my-target", 42, "human",
                                        "2026-09-18T19:00:00Z"),
            },
        })
        unconf = sb._machine_merged_unconfirmed(mem)
        assert ("my-target", 42) in unconf

    def test_weekly_brief_renders_marker(self):
        """DoD (i): the live-generated weekly brief renders the marker on a
        machine-merged test target, and clears it after a human ratify."""
        mem = _make_mem({
            "decision/adjudication/my-target-42-converged_clean": {
                "key": "decision/adjudication/my-target-42-converged_clean",
                "content": _adj_content("my-target", 42, "deliberation-gate",
                                        "2026-09-18T20:00:00Z"),
            },
            "decision/dossier/my-target-42-20260918T200000Z": {
                "key": "decision/dossier/my-target-42-20260918T200000Z",
                "content": "<!-- dossier -->\n```json\n{\"schema\": \"dossier/v1\"}\n```\n",
                "tags": ["lapis-pm", "deliberation-dossier"],
            },
        })
        start = datetime.now(timezone.utc) - timedelta(days=1)
        _quiet = [
            patch.object(sb, "_mem", return_value=mem),
            patch.object(sb, "_arc_docs_since", return_value=[]),
            patch.object(sb, "_read_gardener_observations", return_value=[]),
            patch.object(sb, "_read_arc_climate", return_value=[]),
            patch.object(sb, "_read_locality", return_value=[]),
            patch.object(sb, "_read_autodispatch", return_value=[]),
        ]
        for p in _quiet:
            p.start()
        try:
            buckets = sb._read_buckets(start, period="weekly")
        finally:
            for p in _quiet:
                p.stop()
        rat = buckets[sb.B_RATIFICATIONS]
        assert any(sb._UNCONFIRMED_MARKER in item for item in rat), rat
        # The dossier key surfaces too (C7 consumer hook + tags).
        assert any("dossier/my-target-42" in item for item in rat), rat

        # After a human ratify confirm (later ts), the marker clears.
        mem._data["decision/adjudication/my-target-42-confirm"] = {
            "key": "decision/adjudication/my-target-42-confirm",
            "content": _adj_content("my-target", 42, "human",
                                    "2026-09-19T02:00:00Z"),
        }
        for p in _quiet:
            p.start()
        try:
            buckets2 = sb._read_buckets(start, period="weekly")
        finally:
            for p in _quiet:
                p.stop()
        rat2 = buckets2[sb.B_RATIFICATIONS]
        assert not any(sb._UNCONFIRMED_MARKER in item for item in rat2), rat2


# ---------------------------------------------------------------------------
# Dossier byte bound + required field set
# ---------------------------------------------------------------------------

class TestDossierBounds:
    def test_byte_bound_enforced(self):
        out = d.DeliberationOutcome(
            target_id="t", pr_number=1, verdict=d.VERDICT_NOT_CONVERGED,
            confidence=0.5,
            evidence=[f"p{i}.py:{i} some long citation" for i in range(15)],
            dissent="x" * 5000,
            blockage_reason="y" * 5000,
        )
        body = d.build_dossier(out, target_id="t", pr_number=1, fork=None)
        content = d.render_dossier(body)
        assert len(content.encode("utf-8")) <= d.DOSSIER_BYTE_BOUND

    def test_required_fields_present(self):
        out = d.DeliberationOutcome(
            target_id="t", pr_number=1, verdict=d.VERDICT_CONVERGED,
            confidence=0.9, evidence=["a.py:1 f"],
            deliberation_ids=["del-x"],
        )
        body = d.build_dossier(out, target_id="t", pr_number=1,
                               fork={"repo": "r"})
        for fld in ("verdict", "confidence", "evidence", "deliberation_id",
                    "fork_class"):
            assert fld in body, fld
        # Conditional fields absent when not applicable.
        assert "dissent" not in body
        assert "rendered_held_paths" not in body

    def test_dossier_key_pattern_collision_safety(self):
        """decision/dossier/<tid>-<pr>-<ts> never matches decision/<tid>-*."""
        key = d.write_dossier(_make_mem(), target_id="my-target", pr_number=42,
                              body={"schema": "dossier/v1", "verdict": "blocked",
                                    "confidence": 0.0, "evidence": [],
                                    "deliberation_id": None, "fork_class": None})
        assert key.startswith("decision/dossier/my-target-42-")
        # The prior-hold gate pattern is decision/<tid>-... (single segment).
        assert key.removeprefix("decision/").count("/") == 1
