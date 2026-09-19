"""D6 (fixer-reception-v0, leg 2): the auditor triggers + encode.

Coverage (spec AC4, daemon half):
  - D6a salvage trigger: a NEW salvage-shaped PR passes the three gates
    (marker / pending / budget) and dispatches exactly ONE auditor; a
    same-head re-fire is blocked by the gate-(i) marker (value = head
    sha); a head advance re-triggers.
  - the per-target 2-per-12h budget (the env-var-resolved constant
    GW_AGENT_AUDIT_BUDGET_PER_12H, default 2 — the doctrine of
    Invariant 2 applies symmetrically to loosening AND tightening);
  - the pending-auditor gate (in-flight guard; a pending audit defers to
    the next tick);
  - D6b no-op trigger: healthy-verdict partition (starved/refuted keep
    the existing escalation path unchanged — spec Invariant 8), once-per-
    pr+cycle marker;
  - D6c encode: a successful audit JSON writes the pm:auditor:pr=N:sha=
    comment + the gate-(i) marker (value = head sha) + the brief JSON;
    a parse-fail writes the failed observation and does NOT set the
    marker (a failed audit is retryable);
  - the reconcile failed-flip clears the marker (a hard-killed/timeout
    audit is lost but the head is still un-audited).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents_core.comments import Comment
from agents_core.mem import MemoryStore

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TID = "aud-test-target"
PR = 945
HEAD_SHA = "19585ea01234567890abcdef1234567890abcdef"
PREV_SHA = "abcdef1234567890abcdef1234567890abcdef12"
MAIN_SHA = "0123456789abcdef0123456789abcdef01234567"
SALVAGE_REF = f"lapis/{TID}/local-salvage-19585ea0"


@pytest.fixture
def mem_store(tmp_path):
    store = MemoryStore(tmp_path / "mem.db")
    with patch("lapis_pm.pm_core._mem", return_value=store):
        yield store


@pytest.fixture
def episodic_store(tmp_path):
    """An in-memory episodic store: all_comments() returns the fixture's
    comment list; write_* appends Comment objects into it (tags + content
    are what the tests assert on)."""
    state: dict = {"comments": []}

    def _append(content: str, tags: list[str]) -> Comment:
        c = Comment(id=f"cm-{len(state['comments'])}",
                    ts=f"2026-09-08T13:00:00.{len(state['comments']):06d}",
                    author="lapis-pm", author_type="agent",
                    content=content, tags=list(tags))
        state["comments"].append(c)
        return c

    with patch("lapis_pm.pm_core.episodic") as ep:
        ep.all_comments.side_effect = lambda tid: list(state["comments"])
        ep.spec.return_value = "/srv/lapis/planning/specs/fixer-reception-v0.md"
        ep.spec_summary.return_value = "(spec summary)"
        ep.write_observation.side_effect = (
            lambda tid, content, extra_tags=None:
            _append(content, ["pm:observation"] + list(extra_tags or []))
        )
        ep.write_result.side_effect = (
            lambda tid, content, extra_tags=None:
            _append(content, ["pm:result"] + list(extra_tags or []))
        )
        ep.write_dispatch.side_effect = (
            lambda tid, content, extra_tags=None:
            _append(content, ["pm:dispatch"] + list(extra_tags or []))
        )
        ep.write_brief.side_effect = (
            lambda tid, content, extra_tags=None:
            _append(content, ["pm:brief"] + list(extra_tags or []))
        )
        yield ep


def _comment(ts: str, tags: list[str], content: str = "c") -> Comment:
    return Comment(id=f"cm-{ts}", ts=ts, author="lapis-pm",
                   author_type="agent", content=content, tags=list(tags))


def _salvage_pr() -> dict:
    return {
        "number": PR,
        "repo": "conductor",
        "title": f"[SALVAGE] fixer-reception test (PR #{PR})",
        "head": {"ref": SALVAGE_REF, "sha": HEAD_SHA},
        "body": (
            "Salvage of a gate-rejected fixer run.\n"
            "<!-- lapis-gpu-id: gpu-1 -->\n"
            "<!-- lapis-tid: aud-test-target -->\n"
            "<!-- lapis-salvage: true -->\n"
        ),
    }


def _dispatch_result(task_id: str = "gpu-aud-1") -> SimpleNamespace:
    return SimpleNamespace(task_id=task_id, spec_id="spec-aud-1")


def _dispatched_record(**over) -> dict:
    rec = {
        "gpu_id": "gpu-aud-1",
        "spec_id": "spec-aud-1",
        "agent_type": "auditor",
        "intent": "audit",
        "repo": "conductor",
        "pr_number": PR,
        "mode": "salvage",
        "head_sha": HEAD_SHA,
        "existing_branch": SALVAGE_REF,
        "ts": "2026-09-08T12:00:00-07:00",
        "status": "pending",
        "retry_count": 0,
    }
    rec.update(over)
    return rec


def _patch_dispatch(res=None):
    """Patch the shaper dispatch + the dispatch-side side effects so
    _act_dispatch_auditor runs end-to-end against the mem-backed gates."""
    res = res or _dispatch_result()
    return (
        patch("lapis_pm.pm_core._SHAPER.dispatch", return_value=res),
            patch("lapis_pm.pm_core.steer.inject_overlay"),
        patch("lapis_pm.pm_core._ensure_dispatch_owned"),
            patch("lapis_pm.pm_core._check_calcification"),
        patch("lapis_pm.pm_core._baseline_main_sha", return_value=MAIN_SHA),
    )


def _seed_budget(store: MemoryStore, n: int, age_hours: float = 0.0) -> None:
    ts = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    store.set(
        f"pm/audit/{TID}/budget",
        json.dumps({"entries": [ts.isoformat()] * n}),
        tags=["lapis-pm", "pm:audit-budget"],
    )


# ---------------------------------------------------------------------------
# Budget constant (D6a gate iii — env-var-resolved, default 2)
# ---------------------------------------------------------------------------

def test_audit_budget_default_is_two(mem_store):
    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop(pm_core.AUDIT_BUDGET_PER_12H_ENV, None)
        assert pm_core._audit_budget_per_12h() == 2


def test_audit_budget_env_overrides(mem_store):
    with patch.dict("os.environ", {pm_core.AUDIT_BUDGET_PER_12H_ENV: "5"}):
        assert pm_core._audit_budget_per_12h() == 5


def test_audit_budget_env_invalid_falls_back_to_default(mem_store):
    with patch.dict("os.environ", {pm_core.AUDIT_BUDGET_PER_12H_ENV: "not-an-int"}):
        assert pm_core._audit_budget_per_12h() == 2
    with patch.dict("os.environ", {pm_core.AUDIT_BUDGET_PER_12H_ENV: "0"}):
        assert pm_core._audit_budget_per_12h() == 2


def test_audit_budget_window_prunes_old_entries(mem_store):
    # Two entries inside the window + one 13h old: only two count.
    _seed_budget(mem_store, 2)
    _seed_budget(mem_store, 1, age_hours=13)
    # _seed_budget overwrites; rebuild with mixed ages explicitly.
    now = datetime.now(timezone.utc)
    mem_store.set(
        f"pm/audit/{TID}/budget",
        json.dumps({"entries": [
            (now - timedelta(hours=1)).isoformat(),
            (now - timedelta(hours=11)).isoformat(),
            (now - timedelta(hours=13)).isoformat(),
        ]}),
        tags=["lapis-pm", "pm:audit-budget"],
    )
    assert pm_core._audit_budget_used(TID) == 2
    assert pm_core._audit_budget_has_room(TID) is False  # 2/2 exhausted
    # Drop one entry → room.
    mem_store.set(
        f"pm/audit/{TID}/budget",
        json.dumps({"entries": [
            (now - timedelta(hours=1)).isoformat(),
            (now - timedelta(hours=13)).isoformat(),
        ]}),
        tags=["lapis-pm", "pm:audit-budget"],
    )
    assert pm_core._audit_budget_used(TID) == 1
    assert pm_core._audit_budget_has_room(TID) is True


# ---------------------------------------------------------------------------
# Gate (i): the dispatched marker (value = head sha)
# ---------------------------------------------------------------------------

def test_gate_i_marker_blocks_same_head(mem_store, episodic_store):
    """A fresh salvage PR dispatches; a same-head re-fire is blocked by
    the marker (AC4: exactly ONE dispatch per head)."""
    pr = _salvage_pr()
    with _patch_dispatch()[0] as disp:
        action = pm_core._maybe_dispatch_auditor_salvage(TID, "conductor", pr)
    assert action == f"action:auditor_dispatched:pr={PR}:mode=salvage"
    assert disp.call_count == 1

    # The gate-(i) marker is set at ENCODE time (D6c) — not at dispatch
    # time — so a failed/timed-out audit leaves no marker. Simulate the
    # encode (D6c success path) setting it with the head sha as value.
    pm_core._set_audit_dispatched(TID, PR, HEAD_SHA)
    rec = mem_store.get(pm_core._audit_dispatched_key(TID, PR))
    assert rec is not None
    assert json.loads(rec["content"])["head_sha"] == HEAD_SHA

    # Same head, next tick: gate (i) holds — no second dispatch.
    with _patch_dispatch()[0] as disp2:
        action2 = pm_core._maybe_dispatch_auditor_salvage(TID, "conductor", pr)
    assert action2 is None
    assert disp2.call_count == 0


def test_gate_i_marker_retriggers_on_head_advance(mem_store, episodic_store):
    """A head advance re-triggers (AC4): the marker's value is the
    audited head sha; a new sha differs → dispatch again. (The pending
    record from the first dispatch is flipped to processed first — the
    encode's successful audit does that — so gate (ii) releases.)"""
    pr = _salvage_pr()
    with _patch_dispatch()[0]:
        assert pm_core._maybe_dispatch_auditor_salvage(TID, "conductor", pr) is not None
    # The successful encode flips the record to processed and sets the
    # marker (D6c) — simulate both, then advance the head.
    records = pm_core.load_dispatched(TID)
    records[-1]["status"] = "processed"
    pm_core.save_dispatched(TID, records)
    pm_core._set_audit_dispatched(TID, PR, HEAD_SHA)
    pr2 = dict(pr)
    pr2["head"] = {"ref": SALVAGE_REF, "sha": PREV_SHA}
    with _patch_dispatch()[0] as disp:
        action = pm_core._maybe_dispatch_auditor_salvage(TID, "conductor", pr2)
    assert action == f"action:auditor_dispatched:pr={PR}:mode=salvage"
    assert disp.call_count == 1


def test_gate_i_marker_clear_retriggers(mem_store, episodic_store):
    """A failed/timed-out audit clears the marker (D6c) → the un-audited
    head re-dispatches next tick."""
    pm_core._set_audit_dispatched(TID, PR, HEAD_SHA)
    pm_core._clear_audit_dispatched(TID, PR)
    assert mem_store.get(pm_core._audit_dispatched_key(TID, PR)) is None
    with _patch_dispatch()[0] as disp:
        action = pm_core._maybe_dispatch_auditor_salvage(
            TID, "conductor", _salvage_pr()
        )
    assert action == f"action:auditor_dispatched:pr={PR}:mode=salvage"
    assert disp.call_count == 1


# ---------------------------------------------------------------------------
# Gate (ii): pending-auditor in-flight guard
# ---------------------------------------------------------------------------

def test_gate_ii_pending_auditor_defers(mem_store, episodic_store):
    """A pending auditor record for the PR defers the dispatch to the
    next tick (the in-flight guard; the council's trigger-density
    finding)."""
    mem_store.set(
        pm_core._dispatched_key(TID),
        json.dumps([_dispatched_record(status="pending")]),
        tags=["lapis-pm"],
    )
    with _patch_dispatch()[0] as disp:
        action = pm_core._maybe_dispatch_auditor_salvage(
            TID, "conductor", _salvage_pr()
        )
    assert action is None
    assert disp.call_count == 0


def test_has_pending_auditor_ignores_other_prs_and_types(mem_store):
    mem_store.set(
        pm_core._dispatched_key(TID),
        json.dumps([
            _dispatched_record(status="pending", pr_number=999),
            _dispatched_record(status="processed"),
            _dispatched_record(status="pending", agent_type="reviewer"),
        ]),
        tags=["lapis-pm"],
    )
    assert pm_core._has_pending_auditor_for_pr(TID, PR) is False
    mem_store.set(
        pm_core._dispatched_key(TID),
        json.dumps([_dispatched_record(status="pending")]),
        tags=["lapis-pm"],
    )
    assert pm_core._has_pending_auditor_for_pr(TID, PR) is True


# ---------------------------------------------------------------------------
# Gate (iii): the per-target budget
# ---------------------------------------------------------------------------

def test_gate_iii_budget_exhausted_defers(mem_store, episodic_store):
    """At most N per 12h (default 2): a third dispatch in the window is
    deferred with the pm:auditor-budget-exhausted observation."""
    _seed_budget(mem_store, 2)
    with _patch_dispatch()[0] as disp:
        action = pm_core._maybe_dispatch_auditor_salvage(
            TID, "conductor", _salvage_pr()
        )
    assert action == f"noop:auditor_budget_exhausted:pr={PR}"
    assert disp.call_count == 0
    # The observation carries the budget tag.
    obs = [c for c in episodic_store.all_comments(TID)
           if "pm:auditor-budget-exhausted" in c.tags]
    assert len(obs) == 1


def test_dispatch_records_budget_ledger(mem_store, episodic_store):
    """The budget ledger is recorded at dispatch time — a lost audit
    still cost a dispatch."""
    pr = _salvage_pr()
    with _patch_dispatch()[0]:
        pm_core._maybe_dispatch_auditor_salvage(TID, "conductor", pr)
    assert pm_core._audit_budget_used(TID) == 1
    # The successful encode flips the record to processed (gate (ii)
    # releases) and sets the marker — simulate both. A head advance is
    # what re-triggers (the marker's value is the audited head sha).
    records = pm_core.load_dispatched(TID)
    records[-1]["status"] = "processed"
    pm_core.save_dispatched(TID, records)
    pm_core._set_audit_dispatched(TID, PR, HEAD_SHA)
    pr2 = dict(pr)
    pr2["head"] = {"ref": SALVAGE_REF, "sha": PREV_SHA}
    with _patch_dispatch()[0]:
        pm_core._maybe_dispatch_auditor_salvage(TID, "conductor", pr2)
    assert pm_core._audit_budget_used(TID) == 2
    assert pm_core._audit_budget_has_room(TID) is False


# ---------------------------------------------------------------------------
# Salvage-shape recognition (Invariant 3: consumed, not reshaped)
# ---------------------------------------------------------------------------

def test_is_salvage_pr_requires_ref_and_marker():
    pr = _salvage_pr()
    assert pm_core._is_salvage_pr(pr) is True
    # Missing the body marker → not salvage-shaped.
    no_marker = dict(pr, body="no marker here")
    assert pm_core._is_salvage_pr(no_marker) is False
    # Missing the -salvage- ref → not salvage-shaped.
    no_ref = dict(pr, head={"ref": "lapis/aud-test-target/local", "sha": HEAD_SHA})
    assert pm_core._is_salvage_pr(no_ref) is False


def test_non_salvage_pr_does_not_trigger(mem_store, episodic_store):
    pr = dict(_salvage_pr(),
              head={"ref": "lapis/aud-test-target/local", "sha": HEAD_SHA})
    with _patch_dispatch()[0] as disp:
        assert pm_core._maybe_dispatch_auditor_salvage(TID, "conductor", pr) is None
    assert disp.call_count == 0


# ---------------------------------------------------------------------------
# D5 input resolution: the previous head sha
# ---------------------------------------------------------------------------

def test_previous_head_sha_from_observations(mem_store, episodic_store):
    """The latest observed sha DISTINCT from the current head; falls back
    to the head only when nothing distinct exists; None when nothing has
    been observed."""
    with patch("lapis_pm.pm_core.episodic") as ep:
        ep.all_comments.return_value = [
            _comment("2026-09-08T10:00:00", [f"pm:pr={PR}:sha={PREV_SHA}"]),
            _comment("2026-09-08T11:00:00", [f"pm:pr={PR}:sha={HEAD_SHA}"]),
        ]
        # Current head is the latest observation → the previous head is
        # the one before it.
        assert pm_core._previous_head_sha(TID, PR, HEAD_SHA) == PREV_SHA
        # No prior head (fresh salvage, only one observation) → the head
        # itself (the fallback).
        ep.all_comments.return_value = [
            _comment("2026-09-08T10:00:00", [f"pm:pr={PR}:sha={HEAD_SHA}"]),
        ]
        assert pm_core._previous_head_sha(TID, PR, HEAD_SHA) == HEAD_SHA
        # Nothing observed → None (the caller falls back to the current head).
        ep.all_comments.return_value = []
        assert pm_core._previous_head_sha(TID, PR, HEAD_SHA) is None


def test_previous_head_sha_latest_distinct_matrix(mem_store, episodic_store):
    """auditor-diagnosability-v0 D-L3: the resolver returns the latest
    DISTINCT sha, not merely the second-to-last observation. [A, H, H]
    with current=H returns A (today returns H - the degenerate case the
    2026-09-08 live run hit); [A, B] with current=H returns B (latest
    distinct, not latest overall)."""
    with patch("lapis_pm.pm_core.episodic") as ep:
        # [A, H, H] current=H -> A (the live degenerate case).
        ep.all_comments.return_value = [
            _comment("2026-09-08T10:00:00", [f"pm:pr={PR}:sha={PREV_SHA}"]),
            _comment("2026-09-08T11:00:00", [f"pm:pr={PR}:sha={HEAD_SHA}"]),
            _comment("2026-09-08T12:00:00", [f"pm:pr={PR}:sha={HEAD_SHA}"]),
        ]
        assert pm_core._previous_head_sha(TID, PR, HEAD_SHA) == PREV_SHA
        # [A, B] current=H -> B (latest distinct, not latest overall).
        ep.all_comments.return_value = [
            _comment("2026-09-08T10:00:00", [f"pm:pr={PR}:sha={PREV_SHA}"]),
            _comment("2026-09-08T11:00:00", [f"pm:pr={PR}:sha={MAIN_SHA}"]),
        ]
        assert pm_core._previous_head_sha(TID, PR, HEAD_SHA) == MAIN_SHA


# ---------------------------------------------------------------------------
# D6b: the no-op trigger (healthy-verdict partition)
# ---------------------------------------------------------------------------

def _noop_rec(**over) -> dict:
    rec = {
        "gpu_id": "gpu-fix-retry-1",
        "agent_type": "fixer_retry",
        "pr_number": PR,
        "cycle": 1,
        "repo": "conductor",
        "head_sha": HEAD_SHA,
        "existing_branch": SALVAGE_REF,
        "ts": "2026-09-08T12:00:00-07:00",
        "status": "processed",
    }
    rec.update(over)
    return rec


def test_noop_trigger_healthy_verdict_dispatches(mem_store, episodic_store):
    """A fixer-retry no-op against a HEALTHY verdict dispatches the
    auditor (audit why the retry produced no change)."""
    rec = _noop_rec()
    with patch("lapis_pm.pm_core._review_verdict_for_cycle",
               return_value={"verdict": "fixable", "issues": [],
                             "panel_starvation": {"starved": False,
                                                  "legs_down": []}}), \
         patch("lapis_pm.pm_core.get_open_prs", return_value=[]), \
         _patch_dispatch()[0] as disp:
        action = pm_core._maybe_dispatch_auditor_noop(TID, rec, PR)
    assert action == f"action:auditor_dispatched:pr={PR}:mode=noop"
    assert disp.call_count == 1
    # The once-per-pr+cycle marker is set.
    marker = mem_store.get(
        f"pm/noop-audit/{TID}/pr={PR}/cycle=1/recorded"
    )
    assert marker is not None


def test_noop_trigger_starved_verdict_keeps_existing_path(mem_store, episodic_store):
    """Invariant 8: the starved/refuted case keeps its existing
    _escalate_noop_retry_if_degraded path UNCHANGED — the no-op auditor
    trigger fires only for the healthy-verdict case."""
    rec = _noop_rec()
    with patch("lapis_pm.pm_core._review_verdict_for_cycle",
               return_value={"verdict": "needs-human", "issues": [],
                             "legs_down": ["local_witness"]}), \
         patch("lapis_pm.panel_starvation.verdict_is_starved",
               return_value=True), \
         _patch_dispatch()[0] as disp:
        action = pm_core._maybe_dispatch_auditor_noop(TID, rec, PR)
    assert action is None
    assert disp.call_count == 0


def test_noop_trigger_refuted_verdict_keeps_existing_path(mem_store, episodic_store):
    rec = _noop_rec()
    with patch("lapis_pm.pm_core._review_verdict_for_cycle",
               return_value={"verdict": "needs-human", "issues": [],
                             "refuted_absence_findings": ["f1"]}), \
         patch("lapis_pm.panel_starvation.verdict_is_starved",
               return_value=False), \
         _patch_dispatch()[0] as disp:
        action = pm_core._maybe_dispatch_auditor_noop(TID, rec, PR)
    assert action is None
    assert disp.call_count == 0


def test_noop_trigger_once_per_pr_cycle(mem_store, episodic_store):
    """The once-per-pr+cycle marker: a second no-op in the same cycle
    does not re-dispatch."""
    rec = _noop_rec()
    with patch("lapis_pm.pm_core._review_verdict_for_cycle",
               return_value={"verdict": "fixable", "issues": [],
                             "panel_starvation": {"starved": False,
                                                  "legs_down": []}}), \
         patch("lapis_pm.pm_core.get_open_prs", return_value=[]):
        with _patch_dispatch()[0] as disp1:
            assert pm_core._maybe_dispatch_auditor_noop(TID, rec, PR) is not None
        assert disp1.call_count == 1
        # Same cycle again: marker holds.
        with _patch_dispatch()[0] as disp2:
            assert pm_core._maybe_dispatch_auditor_noop(TID, rec, PR) is None
        assert disp2.call_count == 0
    # A new cycle dispatches again — a head advance is what re-triggers
    # (the marker's value is the audited head sha; the once-per-pr+cycle
    # marker is per-cycle, and gate (i) is per-head). The successful
    # encode flips the record to processed and sets the marker (D6c) —
    # simulate both.
    records = pm_core.load_dispatched(TID)
    records[-1]["status"] = "processed"
    pm_core.save_dispatched(TID, records)
    pm_core._set_audit_dispatched(TID, PR, HEAD_SHA)
    rec2 = _noop_rec(cycle=2, head_sha=PREV_SHA)
    with patch("lapis_pm.pm_core._review_verdict_for_cycle",
               return_value={"verdict": "fixable", "issues": [],
                             "confidence": 0.9}), \
         patch("lapis_pm.panel_starvation.verdict_is_starved",
               return_value=False), \
         patch("lapis_pm.pm_core.get_open_prs", return_value=[]), \
         _patch_dispatch()[0] as disp3:
        assert pm_core._maybe_dispatch_auditor_noop(TID, rec2, PR) is not None
    assert disp3.call_count == 1


# ---------------------------------------------------------------------------
# D6c: the auditor result encode
# ---------------------------------------------------------------------------

AUDIT_JSON = {
    "suite_states": {
        "main": {"sha": MAIN_SHA, "passed": 375, "failed": 0, "errors": 0},
        "previous_head": {"sha": PREV_SHA, "passed": 344, "failed": 31, "errors": 0},
        "salvage_head": {"sha": HEAD_SHA, "passed": 317, "failed": 58, "errors": 0},
    },
    "failure_set_delta": {
        "new_failures": ["tests/test_runner.py::test_auto_pick"],
        "fixed_failures": [],
        "unchanged_count": 31,
        "classification": "interaction",
    },
    "root_cause": [
        {"failure_class": "auto_pick_ambiguous",
         "claim": "DoD-1 auto-pick raises ValueError('ambiguous') on the 122B collision",
         "evidence": "conductor/models.py:453", "verified": True},
    ],
    "delta_to_green": [
        {"unit": "stub _converge_probe_seat in the runner tests",
         "gate": "the 31 runner tests pass"},
    ],
    "deliverables_evaluation": [
        {"deliverable": "D1 slug fix", "status": "met",
         "evidence": "tests/test_d1.py::test_push_ref"},
    ],
}


def test_encode_audit_brief_renders_five_sections():
    """The five-section brief (the advisory brief surface)."""
    brief = pm_core._render_audit_brief(AUDIT_JSON, PR, HEAD_SHA)
    assert "### 1. Suite states" in brief
    assert "### 2. Failure-set delta" in brief
    assert "### 3. Root causes" in brief
    assert "### 4. Delta-to-green" in brief
    assert "### 5. Deliverables evaluation" in brief
    assert "classification: interaction" in brief
    assert "conductor/models.py:453" in brief
    assert "stub _converge_probe_seat" in brief
    assert "D1 slug fix: met" in brief


def test_encode_audit_brief_unverified_root_cause_is_honest():
    """verified: false renders as an honest UNVERIFIED marker (the
    Council's clarification — not a failed audit)."""
    aud = json.loads(json.dumps(AUDIT_JSON))
    aud["root_cause"][0]["verified"] = False
    brief = pm_core._render_audit_brief(aud, PR, HEAD_SHA)
    assert "UNVERIFIED" in brief


@pytest.fixture
def gpu_dirs(tmp_path, monkeypatch):
    """Redirect the gpu-queue output dirs to a tmp root so the
    _gpu_output_path lookup (the real COMPLETED/FAILED dirs) finds the
    test's output file. Yields (completed_dir, failed_dir)."""
    completed = tmp_path / "gpu_queue.completed"
    failed = tmp_path / "gpu_queue.failed"
    completed.mkdir(parents=True)
    failed.mkdir(parents=True)
    monkeypatch.setattr(pm_core, "COMPLETED_DIR", completed)
    monkeypatch.setattr(pm_core, "FAILED_DIR", failed)
    monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_COMPLETED_DIR", completed)
    monkeypatch.setattr(pm_core, "CLAUDE_QUEUE_FAILED_DIR", failed)
    return completed, failed


def _auditor_record(**over) -> dict:
    rec = {
        "gpu_id": "gpu-aud-1",
        "agent_type": "auditor",
        "pr_number": PR,
        "head_sha": HEAD_SHA,
        "existing_branch": SALVAGE_REF,
        "repo": "conductor",
        "ts": "2026-09-08T12:00:00-07:00",
        "status": "pending",
    }
    rec.update(over)
    return rec


def test_encode_successful_audit_sets_marker_and_brief(
        mem_store, episodic_store, gpu_dirs):
    """D6c success path: the pm:auditor:pr=N:sha= tagged comment + the
    gate-(i) marker (value = head sha) + the brief JSON mem key."""
    completed, _ = gpu_dirs
    out_file = completed / f"{_auditor_record()['gpu_id']}-output.md"
    out_file.write_text("```json\n" + json.dumps(AUDIT_JSON) + "\n```")
    rec = _auditor_record()
    mem_store.set(pm_core._dispatched_key(TID), json.dumps([rec]),
                  tags=["lapis-pm"])
    with patch("lapis_pm.pm_core._reap_verdict", return_value=None):
        encoded, _ = pm_core._encode_gpu_results(TID)
    assert encoded >= 1
    # The brief comment carries the auditor + pr + sha tags.
    comments = episodic_store.all_comments(TID)
    brief_comments = [
        c for c in comments
        if f"pm:auditor:pr={PR}:sha={HEAD_SHA}" in c.tags
    ]
    assert len(brief_comments) == 1
    assert "### 1. Suite states" in brief_comments[0].content
    # Gate-(i) marker set with the head sha as value.
    marker = mem_store.get(pm_core._audit_dispatched_key(TID, PR))
    assert marker is not None
    assert json.loads(marker["content"])["head_sha"] == HEAD_SHA
    # The brief JSON is stored.
    stored = mem_store.get(pm_core._audit_brief_key(TID, PR))
    assert stored is not None
    assert json.loads(stored["content"])["failure_set_delta"]["classification"] == "interaction"


def test_encode_failed_audit_writes_observation_not_marker(
        mem_store, episodic_store, gpu_dirs):
    """D6c parse-fail: the pm:auditor:pr=N:failed observation is written
    and the dispatched marker is NOT set (a failed audit is retryable —
    the next tick may re-dispatch the un-audited head)."""
    completed, _ = gpu_dirs
    out_file = completed / f"{_auditor_record()['gpu_id']}-output.md"
    out_file.write_text("I could not complete the audit — the suite hung.")
    rec = _auditor_record()
    mem_store.set(pm_core._dispatched_key(TID), json.dumps([rec]),
                  tags=["lapis-pm"])
    with patch("lapis_pm.pm_core._reap_verdict", return_value=None):
        pm_core._encode_gpu_results(TID)
    comments = episodic_store.all_comments(TID)
    failed = [c for c in comments if "pm:failure" in c.tags
              and f"pm:auditor:pr={PR}:failed" in c.tags]
    assert len(failed) == 1
    # No marker — the head is still un-audited.
    assert mem_store.get(pm_core._audit_dispatched_key(TID, PR)) is None


def test_encode_failed_audit_clears_stale_marker(
        mem_store, episodic_store, gpu_dirs):
    """A parse-fail also CLEARS a marker a prior partial state left, so
    the retry is not blocked by gate (i)."""
    pm_core._set_audit_dispatched(TID, PR, HEAD_SHA)
    assert mem_store.get(pm_core._audit_dispatched_key(TID, PR)) is not None
    completed, _ = gpu_dirs
    out_file = completed / f"{_auditor_record()['gpu_id']}-output.md"
    out_file.write_text("not json at all")
    rec = _auditor_record()
    mem_store.set(pm_core._dispatched_key(TID), json.dumps([rec]),
                  tags=["lapis-pm"])
    with patch("lapis_pm.pm_core._reap_verdict", return_value=None):
        pm_core._encode_gpu_results(TID)
    assert mem_store.get(pm_core._audit_dispatched_key(TID, PR)) is None


# ---------------------------------------------------------------------------
# D-L1 (auditor-diagnosability-v0): parse robustness - fence-anywhere +
# first-JSON-object salvage with the suite_states shape gate
# ---------------------------------------------------------------------------

LIVE_TRAILER = (
    "Running as unit: lapis-fixer-gpu-aud-1.scope; "
    "invocation ID: 1727\n"
    "WARN: worktree_setup: .claude/settings.json missing at "
    "/tmp/lapis-pm-worktrees/gpu-aud-1\n"
)


def _encode_audit_output(mem_store, episodic_store, gpu_dirs, content: str):
    """Write the auditor output file + pending record, run the encode,
    and return (encoded, comments, mem_store)."""
    completed, _ = gpu_dirs
    out_file = completed / f"{_auditor_record()['gpu_id']}-output.md"
    out_file.write_text(content)
    rec = _auditor_record()
    mem_store.set(pm_core._dispatched_key(TID), json.dumps([rec]),
                  tags=["lapis-pm"])
    with patch("lapis_pm.pm_core._reap_verdict", return_value=None):
        encoded, _ = pm_core._encode_gpu_results(TID)
    return encoded, episodic_store.all_comments(TID), out_file


def _assert_success_encode(encoded, comments, mem_store):
    assert encoded >= 1
    brief_comments = [
        c for c in comments
        if f"pm:auditor:pr={PR}:sha={HEAD_SHA}" in c.tags
    ]
    assert len(brief_comments) == 1
    assert "### 1. Suite states" in brief_comments[0].content
    marker = mem_store.get(pm_core._audit_dispatched_key(TID, PR))
    assert marker is not None
    assert json.loads(marker["content"])["head_sha"] == HEAD_SHA
    stored = mem_store.get(pm_core._audit_brief_key(TID, PR))
    assert stored is not None
    assert json.loads(stored["content"])["suite_states"]["main"]["passed"] == 375


def _assert_failed_encode(comments, mem_store):
    failed = [c for c in comments if "pm:failure" in c.tags
              and f"pm:auditor:pr={PR}:failed" in c.tags]
    assert len(failed) == 1
    assert mem_store.get(pm_core._audit_dispatched_key(TID, PR)) is None


def test_salvage_audit_json_clean_and_fenced():
    """(1) the clean whole-text parse is untouched; a leading fence is
    still handled (regression)."""
    assert pm_core._salvage_audit_json(json.dumps(AUDIT_JSON)) == AUDIT_JSON
    assert (pm_core._salvage_audit_json(
        "```json\n" + json.dumps(AUDIT_JSON) + "\n```") == AUDIT_JSON)


def test_salvage_audit_json_trailing_metadata():
    """The LIVE shape: complete JSON + the runner's trailing metadata
    lines (no fence)."""
    text = json.dumps(AUDIT_JSON) + "\n" + LIVE_TRAILER
    assert pm_core._salvage_audit_json(text) == AUDIT_JSON


def test_salvage_audit_json_prose_wrapped_and_braces_in_strings():
    """Leading prose + fenced JSON; the JSON carries braces inside
    string values (the string-state tracking is pinned)."""
    aud = json.loads(json.dumps(AUDIT_JSON))
    aud["root_cause"][0]["evidence"] = (
        'tool output: {"error": "unknown tool: run_tests"} on every '
        "call, see conductor/models.py:453"
    )
    text = (
        "Here is the audit result:\n"
        "```json\n" + json.dumps(aud, indent=2) + "\n```\n"
        + LIVE_TRAILER
    )
    assert pm_core._salvage_audit_json(text) == aud


def test_salvage_audit_json_skips_decoy_leader():
    """A leading NON-AUDIT dict (the live decoy shape: run 1 quotes
    {\"error\": \"unknown tool: run_tests\"} as tool output) is SKIPPED
    (no suite_states); the real audit JSON wins."""
    text = (
        '{"error": "unknown tool: run_tests"}\n'
        + json.dumps(AUDIT_JSON) + "\n" + LIVE_TRAILER
    )
    assert pm_core._salvage_audit_json(text) == AUDIT_JSON


def test_salvage_audit_json_bom_prefixed():
    """A BOM-prefixed valid audit JSON now salvages (BOM tolerance)."""
    text = "\ufeff" + json.dumps(AUDIT_JSON) + "\n" + LIVE_TRAILER
    assert pm_core._salvage_audit_json(text) == AUDIT_JSON


def test_salvage_audit_json_truncated_and_prose_only():
    """A truncated (unbalanced) JSON is NOT salvaged (no fabrication);
    prose-only -> None."""
    full = json.dumps(AUDIT_JSON)
    truncated = full[: len(full) // 2]  # cut mid-object
    assert pm_core._salvage_audit_json(truncated) is None
    assert pm_core._salvage_audit_json(
        "I could not complete the audit - the suite hung.") is None
    assert pm_core._salvage_audit_json("") is None


def test_encode_audit_live_shape_encodes_success(
        mem_store, episodic_store, gpu_dirs):
    """D6c + D-L1: the LIVE output shape (five-section JSON + trailing
    metadata lines) encodes to the success path - brief comment rendered,
    gate-(i) marker set, brief mem key written."""
    encoded, comments, _ = _encode_audit_output(
        mem_store, episodic_store, gpu_dirs,
        json.dumps(AUDIT_JSON) + "\n" + LIVE_TRAILER,
    )
    _assert_success_encode(encoded, comments, mem_store)


def test_encode_audit_decoy_leader_encodes_success(
        mem_store, episodic_store, gpu_dirs):
    """D6c + D-L1: the live decoy shape (a leading non-audit dict
    followed by the real audit JSON + trailer) encodes to the success
    path - the decoy is skipped, the real JSON wins."""
    encoded, comments, _ = _encode_audit_output(
        mem_store, episodic_store, gpu_dirs,
        '{"error": "unknown tool: run_tests"}\n'
        + json.dumps(AUDIT_JSON) + "\n" + LIVE_TRAILER,
    )
    _assert_success_encode(encoded, comments, mem_store)


def test_encode_audit_truncated_output_takes_failed_path(
        mem_store, episodic_store, gpu_dirs):
    """D6c + D-L1: a truncated (unbalanced) JSON still takes the failed
    path (NOT salvaged - no fabrication)."""
    full = json.dumps(AUDIT_JSON)
    encoded, comments, _ = _encode_audit_output(
        mem_store, episodic_store, gpu_dirs, full[: len(full) // 2],
    )
    _assert_failed_encode(comments, mem_store)


# ---------------------------------------------------------------------------
# D-L2 (auditor-diagnosability-v0): retain the full failed output
# ---------------------------------------------------------------------------

def test_encode_failed_audit_retains_full_output_and_names_path(
        mem_store, episodic_store, gpu_dirs):
    """D6c + D-L2: a failed encode with a >1500-char output retains the
    FULL raw text at the failed-output mem key and names the output file
    path in the failed comment; the failed tags and the cleared-marker
    (retryable) semantics are unchanged."""
    prose = ("I could not complete the audit - the suite hung. " * 100)
    assert len(prose) > 1500
    encoded, comments, out_file = _encode_audit_output(
        mem_store, episodic_store, gpu_dirs, prose,
    )
    failed = [c for c in comments if "pm:failure" in c.tags
              and f"pm:auditor:pr={PR}:failed" in c.tags]
    assert len(failed) == 1
    # The full raw text is retained (not the 400-char snippet).
    stored = mem_store.get(pm_core._audit_failed_output_key(TID, PR))
    assert stored is not None
    assert stored["content"] == prose
    assert "pm:audit-failed-output" in stored["tags"]
    # The failed comment names the output file path (and keeps the
    # inline snippet for brief-surface readability).
    assert f"full output: {out_file}" in failed[0].content
    assert "(output: " in failed[0].content
    # Retryable: the marker is NOT set.
    assert mem_store.get(pm_core._audit_dispatched_key(TID, PR)) is None


def test_encode_failed_audit_clears_stale_marker_and_retains_output(
        mem_store, episodic_store, gpu_dirs):
    """A parse-fail still CLEARS a marker a prior partial state left
    (gate (i) stays released for the retry) AND retains the full output."""
    pm_core._set_audit_dispatched(TID, PR, HEAD_SHA)
    assert mem_store.get(pm_core._audit_dispatched_key(TID, PR)) is not None
    prose = "not json at all " + ("x" * 1600)
    encoded, comments, _ = _encode_audit_output(
        mem_store, episodic_store, gpu_dirs, prose,
    )
    assert mem_store.get(pm_core._audit_dispatched_key(TID, PR)) is None
    assert mem_store.get(pm_core._audit_failed_output_key(TID, PR)) is not None


def test_reconcile_failed_flip_clears_marker(mem_store, episodic_store,
                                             monkeypatch):
    """D6c (reconcile): on a hard-kill/timeout flip the audit is lost
    but the head is still un-audited — the marker is cleared so the next
    tick re-dispatches."""
    pm_core._set_audit_dispatched(TID, PR, HEAD_SHA)
    rec = _auditor_record(status="pending")
    mem_store.set(pm_core._dispatched_key(TID), json.dumps([rec]),
                  tags=["lapis-pm"])

    class _FakeQueue:
        def get_recent_failed(self, limit=50):
            return [{"id": rec["gpu_id"], "error": "TIMEOUT: exceeded",
                     "completed_at": None}]

        def get_recent_completed(self, limit=50):
            return []

        def get_active(self):
            return []

        def get_pending(self):
            return []

    monkeypatch.setattr(pm_core, "_ClaudeQueue", _FakeQueue)
    flipped, ok = pm_core._reconcile_dispatched_with_queue_ex(TID)
    assert ok is True
    assert flipped == 1
    assert mem_store.get(pm_core._audit_dispatched_key(TID, PR)) is None
    # The record flipped to failed.
    records = pm_core.load_dispatched(TID)
    assert records[0]["status"] == "failed"
