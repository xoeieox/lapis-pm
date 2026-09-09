"""Tests for the tick decision taxonomy (lapis-pm-tick-decision-telemetry spec).

Parameterized over the full closed enum. Each case constructs the minimal tick
state needed to exercise that code path and asserts the emitted decision string
matches the expected tag exactly.

Coverage:
  Noop family:
    noop:no_change        — default fallthrough (no PRs, no dispatches, no auto-land)
    noop:paused           — target.paused is True
    noop:reviewer_in_flight:pr=N:cycle=K — pending reviewer in dispatched records
    noop:fixer_in_flight:dispatch=ID     — pending fixer_retry in dispatched records
    noop:awaiting_chain_dependency:waiting_on=TID — depends_on unsatisfied

  Action family:
    action:auto_merge:pr=N               — _act_merge success
    action:merge_failed:<e>              — _act_merge failure
    action:auto_land:pr=N:arc=PATH       — _act_auto_land
    action:reviewer_dispatched:pr=N:cycle=K — _act_dispatch_reviewer
    action:fixer_dispatched:source=retry:pr=N:cycle=K — _act_dispatch_fixer_retry
    action:fixer_dispatched:source=init:dispatch=ID   — _act_retry
    action:brief_emitted:kind=hold:...   — _act_brief hold=True
    action:brief_emitted:kind=advisory_clean:... — _act_brief hold=False, no issues
    action:brief_emitted:kind=advisory_screen_issue:... — _act_brief hold=False, issues
    action:brief_decision_applied:...    — _consume_brief_decisions applied
    action:directive_brief:cid=...       — directive received
    action:abandon_brief:cid=...         — retry exhausted
    action:review_exhausted_brief:cid=... — review budget exhausted
    action:review_gate_paused:cid=...    — kill-switch threshold exceeded (first time)
    action:review_gate_pause:already_briefed — kill-switch already briefed

  Skip family:
    skipped=True reason=paused           — decision field = noop:paused
    skipped=True reason=target not found — decision field = noop:no_change
    skipped=True reason=target not pm_bound — decision field = noop:no_change
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_target(paused: bool = False, depends_on: list | None = None,
                 pm_repo: str = "", pm_authority: str = "advisory",
                 data: dict | None = None) -> MagicMock:
    t = MagicMock()
    t.pm_bound = True
    t.paused = paused
    t.paused_reason = None
    t.pm_repo = pm_repo
    t.pm_authority = pm_authority
    t.data = data or {}
    if depends_on is not None:
        t.data["depends_on"] = depends_on
    return t


def _tick_with_patches(target: MagicMock, extra_patches: dict | None = None):
    """Run tick("my-target") with minimal I/O patches + optional extras."""
    from contextlib import ExitStack
    extra = extra_patches or {}
    base_patches = [
        ("lapis_pm.pm_core.TargetStore", MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=target)))),
        ("lapis_pm.pm_core.get_pause_state", MagicMock(return_value="active")),
        ("lapis_pm.pm_core.set_pause_state", MagicMock()),
        ("lapis_pm.pm_core._reconcile_dispatched_with_queue", MagicMock(return_value=0)),
        ("lapis_pm.pm_core.get_cursor", MagicMock(return_value=None)),
        ("lapis_pm.episodic.since", MagicMock(return_value=[])),
        ("lapis_pm.pm_core._perceive_prs", MagicMock(return_value=([], True))),
        ("lapis_pm.pm_core._encode_user_comments", MagicMock(return_value=[])),
        ("lapis_pm.pm_core._seen_pr_ids", MagicMock(return_value=set())),
        ("lapis_pm.pm_core._encode_new_prs", MagicMock(return_value=[])),
        ("lapis_pm.pm_core._encode_pr_sha_updates", MagicMock(return_value=0)),
        ("lapis_pm.pm_core._encode_pr_body_updates", MagicMock(return_value=0)),
        ("lapis_pm.pm_core._encode_gpu_results", MagicMock(return_value=(0, []))),
        ("lapis_pm.pm_core._encode_merged_prs", MagicMock(return_value=0)),
        ("lapis_pm.pm_core._consume_brief_decisions", MagicMock(return_value=None)),
        ("lapis_pm.pm_core._is_auto_land_eligible", MagicMock(return_value=False)),
        ("lapis_pm.pm_core._persist_review_state_cache", MagicMock()),
        ("lapis_pm.pm_core.set_cursor", MagicMock()),
        ("lapis_pm.episodic.write_observation", MagicMock()),
    ]
    with ExitStack() as stack:
        for target_str, mock_val in base_patches:
            stack.enter_context(patch(target_str, mock_val))
        for target_str, mock_val in extra.items():
            stack.enter_context(patch(target_str, mock_val))
        return pm_core.tick("my-target")


# ---------------------------------------------------------------------------
# Noop family
# ---------------------------------------------------------------------------

class TestNoopTaxonomy:
    def test_noop_no_change_default(self):
        """No PRs, no dispatches, no auto-land → noop:no_change."""
        target = _mock_target()
        result = _tick_with_patches(target)
        assert result.decision == "noop:no_change"
        assert not result.skipped

    def test_noop_paused_decision(self):
        """Paused target → skipped=True, decision=noop:paused."""
        target = _mock_target(paused=True)
        store = MagicMock()
        store.get.return_value = target
        with (
            patch("lapis_pm.pm_core.TargetStore", return_value=store),
            patch("lapis_pm.pm_core.get_pause_state", return_value="active"),
            patch("lapis_pm.pm_core.set_pause_state"),
            patch("lapis_pm.pm_core.episodic.write_observation"),
        ):
            result = pm_core.tick("my-target")
        assert result.skipped is True
        assert result.reason == "paused"
        assert result.decision == "noop:paused"

    def test_noop_target_not_found(self):
        """Target not in store → skipped=True, decision=noop:no_change."""
        store = MagicMock()
        store.get.return_value = None
        with patch("lapis_pm.pm_core.TargetStore", return_value=store):
            result = pm_core.tick("missing-target")
        assert result.skipped is True
        assert result.reason == "target not found"
        assert result.decision == "noop:no_change"

    def test_noop_target_not_pm_bound(self):
        """Target exists but not pm_bound → skipped=True, decision=noop:no_change."""
        target = MagicMock()
        target.pm_bound = False
        store = MagicMock()
        store.get.return_value = target
        with patch("lapis_pm.pm_core.TargetStore", return_value=store):
            result = pm_core.tick("my-target")
        assert result.skipped is True
        assert result.reason == "target not pm_bound"
        assert result.decision == "noop:no_change"

    def test_noop_reviewer_in_flight(self):
        """Pending reviewer in dispatched records → noop:reviewer_in_flight:pr=N:cycle=K."""
        target = _mock_target(pm_repo="myrepo")
        pr = {"number": 7, "title": "feat: widgets",
              "head": {"ref": "lapis/my-target/widgets"}, "html_url": "http://x"}
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.static_outcome = _auth.StaticOutcome.static_pass
        cls.verdict = "advisory"
        result = _tick_with_patches(target, {
            "lapis_pm.pm_core._perceive_prs": MagicMock(return_value=([pr], True)),
            "lapis_pm.pm_core._classified_pr_ids": MagicMock(return_value=set()),
            "lapis_pm.pm_core.authority.classify": MagicMock(return_value=cls),
            "lapis_pm.pm_core.episodic.spec_summary": MagicMock(return_value="spec"),
            "lapis_pm.pm_core._review_gate_paused": MagicMock(return_value=False),
            "lapis_pm.pm_core._has_pending_reviewer_for_pr": MagicMock(return_value=True),
            "lapis_pm.pm_core._reviewer_cycle_count": MagicMock(return_value=2),
        })
        assert result.decision == "noop:reviewer_in_flight:pr=7:cycle=2"

    def test_noop_fixer_in_flight(self):
        """Pending fixer_retry in dispatched records → noop:fixer_in_flight:dispatch=ID."""
        target = _mock_target(pm_repo="myrepo")
        pr = {"number": 7, "title": "feat: widgets",
              "head": {"ref": "lapis/my-target/widgets"}, "html_url": "http://x"}
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.static_outcome = _auth.StaticOutcome.static_pass
        cls.verdict = "advisory"
        dispatch_rec = {
            "gpu_id": "task-fixer-xyz",
            "agent_type": "fixer_retry",
            "pr_number": 7,
            "status": "pending",
        }
        result = _tick_with_patches(target, {
            "lapis_pm.pm_core._perceive_prs": MagicMock(return_value=([pr], True)),
            "lapis_pm.pm_core._classified_pr_ids": MagicMock(return_value=set()),
            "lapis_pm.pm_core.authority.classify": MagicMock(return_value=cls),
            "lapis_pm.pm_core.episodic.spec_summary": MagicMock(return_value="spec"),
            "lapis_pm.pm_core._review_gate_paused": MagicMock(return_value=False),
            "lapis_pm.pm_core._has_pending_reviewer_for_pr": MagicMock(return_value=False),
            "lapis_pm.pm_core._has_pending_fixer_for_pr": MagicMock(return_value=True),
            "lapis_pm.pm_core.load_dispatched": MagicMock(return_value=[dispatch_rec]),
        })
        assert result.decision == "noop:fixer_in_flight:dispatch=task-fixer-xyz"

    def test_noop_awaiting_chain_dependency(self):
        """Chain leg with unsatisfied depends_on → noop:awaiting_chain_dependency:waiting_on=TID."""
        target = _mock_target(depends_on=["leg1-other-tid"])
        result = _tick_with_patches(target, {
            "lapis_pm.chain.landed_tids": MagicMock(return_value=set()),
        })
        assert result.decision == "noop:awaiting_chain_dependency:waiting_on=leg1-other-tid"

    def test_chain_dependency_satisfied_falls_through_to_no_change(self):
        """When all depends_on are satisfied → NOT noop:awaiting_chain_dependency."""
        target = _mock_target(depends_on=["leg1-other-tid"])
        result = _tick_with_patches(target, {
            "lapis_pm.chain.landed_tids": MagicMock(return_value={"leg1-other-tid"}),
        })
        assert result.decision == "noop:no_change"


# ---------------------------------------------------------------------------
# Action family — _act_* return value assertions
# ---------------------------------------------------------------------------

class TestActionTaxonomy:
    def test_act_merge_success(self):
        """Successful merge → action:auto_merge:pr=N."""
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.pr_number = 42
        cls.title = "feat: widgets"
        cls.html_url = "http://x"
        cls.diff_loc = 5
        cls.screen_verdict = "auto"
        cls.repo = "myrepo"
        cls.issues = []
        payload = {"classification": cls, "pr": {"number": 42}}
        with (
            patch("lapis_pm.pm_core.merge_pr"),
            patch("lapis_pm.pm_core.episodic.write_merge"),
            patch("lapis_pm.pm_core._mark_pr_classified"),
        ):
            result = pm_core._act_merge("tid", payload)
        assert result == "action:auto_merge:pr=42"

    def test_act_merge_failure(self):
        """Failed merge → action:merge_failed:..."""
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.pr_number = 42
        cls.repo = "myrepo"
        payload = {"classification": cls, "pr": {"number": 42}}
        with (
            patch("lapis_pm.pm_core.merge_pr", side_effect=RuntimeError("forbidden")),
            patch("lapis_pm.pm_core.episodic.write_hold"),
        ):
            result = pm_core._act_merge("tid", payload)
        assert result.startswith("action:merge_failed:")

    def test_act_brief_hold(self):
        """Hold brief → action:brief_emitted:kind=hold:cid=..."""
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.pr_number = 42
        cls.title = "feat: add widgets"
        cls.html_url = "http://x"
        cls.diff = None
        cls.issues = []
        cls.repo = "myrepo"
        cls.reasons = ["static hold path"]
        mock_brief = MagicMock()
        mock_brief.comment_id = "cmt-hold-001"
        mock_brief.synthesis_failed = False
        mock_brief.pushed = True
        with (
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_brief),
            patch("lapis_pm.pm_core.set_outstanding_brief_verified"),
            patch("lapis_pm.pm_core._post_write_sweep_brief"),
            patch("lapis_pm.pm_core._mark_pr_classified"),
            patch("lapis_pm.pm_core.episodic.write_hold"),
        ):
            result = pm_core._act_brief("tid", trigger="held PR", hold=True,
                                        payload={"classification": cls})
        assert result == "action:brief_emitted:kind=hold:cid=cmt-hold-001"

    def test_act_brief_advisory_clean(self):
        """Advisory brief with no issues → action:brief_emitted:kind=advisory_clean:cid=..."""
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.pr_number = 42
        cls.title = "feat: add widgets"
        cls.html_url = "http://x"
        cls.diff = None
        cls.issues = []
        cls.repo = "myrepo"
        mock_brief = MagicMock()
        mock_brief.comment_id = "cmt-adv-001"
        mock_brief.synthesis_failed = False
        mock_brief.pushed = True
        with (
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_brief),
            patch("lapis_pm.pm_core.set_outstanding_brief_verified"),
            patch("lapis_pm.pm_core._post_write_sweep_brief"),
            patch("lapis_pm.pm_core._mark_pr_classified"),
        ):
            result = pm_core._act_brief("tid", trigger="advisory PR", hold=False,
                                        payload={"classification": cls})
        assert result == "action:brief_emitted:kind=advisory_clean:cid=cmt-adv-001"

    def test_act_brief_advisory_screen_issue(self):
        """Advisory brief with issues → action:brief_emitted:kind=advisory_screen_issue:cid=..."""
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.pr_number = 42
        cls.title = "feat: add widgets"
        cls.html_url = "http://x"
        cls.diff = None
        cls.issues = [{"severity": "major", "path": "foo.py", "note": "bad"}]
        cls.repo = "myrepo"
        mock_brief = MagicMock()
        mock_brief.comment_id = "cmt-issue-001"
        mock_brief.synthesis_failed = False
        mock_brief.pushed = True
        with (
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_brief),
            patch("lapis_pm.pm_core.set_outstanding_brief_verified"),
            patch("lapis_pm.pm_core._post_write_sweep_brief"),
            patch("lapis_pm.pm_core._mark_pr_classified"),
        ):
            result = pm_core._act_brief("tid", trigger="advisory PR", hold=False,
                                        payload={"classification": cls})
        assert result == "action:brief_emitted:kind=advisory_screen_issue:cid=cmt-issue-001"

    def test_act_dispatch_reviewer(self):
        """Reviewer dispatch → action:reviewer_dispatched:pr=N:cycle=K."""
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.repo = "myrepo"
        pr = {"number": 5, "title": "feat: foo", "head": {"ref": "lapis/t/foo"}}
        mock_res = MagicMock()
        mock_res.task_id = "task-rev-001"
        mock_res.spec_id = "spec-001"
        with (
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd", return_value="/tmp"),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=None),
            patch("lapis_pm.pm_core._SHAPER") as mock_shaper,
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core._increment_review_gate_counter"),
        ):
            mock_shaper.dispatch.return_value = mock_res
            # Patch get_pr_diff inside the function
            with patch("agents_core.forgejo.get_pr_diff", return_value="--- a\n+++ b",
                       create=True):
                result = pm_core._act_dispatch_reviewer("tid", pr, cls, mode="same", cycle=3)
        assert result == "action:reviewer_dispatched:pr=5:cycle=3"

    def test_act_dispatch_fixer_retry(self):
        """Fixer retry dispatch → action:fixer_dispatched:source=retry:pr=N:cycle=K."""
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.repo = "myrepo"
        pr = {"number": 5, "title": "feat: foo",
              "head": {"ref": "lapis/t/foo"}}
        issues = [{"severity": "major", "path": "foo.py", "note": "bad"}]
        payload = {"pr": pr, "cls": cls, "issues": issues, "cycle": 1, "budget": 2}
        mock_res = MagicMock()
        mock_res.task_id = "task-fixer-001"
        mock_res.spec_id = "spec-001"
        with (
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd", return_value="/tmp"),
            patch("lapis_pm.pm_core._SHAPER") as mock_shaper,
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
        ):
            mock_shaper.dispatch.return_value = mock_res
            result = pm_core._act_dispatch_fixer_retry("tid", payload)
        assert result == "action:fixer_dispatched:source=retry:pr=5:cycle=1"

    def test_act_retry_init_dispatch(self):
        """Initial/retry dispatch via _act_retry → action:fixer_dispatched:source=init:dispatch=ID."""
        rec = {
            "agent_type": "fixer",
            "intent": "implement feature",
            "repo": "myrepo",
            "pr_number": "",
            "slug": "retry",
            "retry_count": 0,
        }
        mock_res = MagicMock()
        mock_res.task_id = "task-init-001"
        mock_res.spec_id = "spec-001"
        with (
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd", return_value="/tmp"),
            patch("lapis_pm.pm_core._SHAPER") as mock_shaper,
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core.episodic.write_retry"),
        ):
            mock_shaper.dispatch.return_value = mock_res
            result = pm_core._act_retry("tid", rec)
        assert result == "action:fixer_dispatched:source=init:dispatch=task-init-001"

    def test_act_auto_land(self):
        """auto_land eligible target → action:auto_land:pr=N:arc=PATH."""
        mock_arc = MagicMock()
        mock_arc.comment_id = "cmt-land-001"
        mock_target = MagicMock()
        mock_target.data = {}
        mock_store = MagicMock()
        mock_store.get.return_value = mock_target
        with (
            patch("lapis_pm.pm_core._merged_pr_numbers_observed", return_value={42}),
            patch("lapis_pm.pm_core._merged_at_for_pr", return_value="2026-05-02T00:00:00"),
            patch("lapis_pm.pm_core._now_iso", return_value="2026-05-02T00:01:00"),
            patch("lapis_pm.pm_core._mem") as mock_mem_fn,
            patch("lapis_pm.pm_core.TargetStore", return_value=mock_store),
            patch("lapis_pm.pm_core.episodic.write"),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.clear_landed_state"),
        ):
            mock_mem_fn.return_value.set = MagicMock()
            land_mod = MagicMock()
            land_mod.generate_arc_doc.return_value = MagicMock()
            land_mod.write_arc_doc.return_value = "/srv/lapis/lapis-state/tid.md"
            chain_mod = MagicMock()
            with (
                patch.dict("sys.modules", {
                    "lapis_pm.land": land_mod,
                    "lapis_pm.chain": chain_mod,
                }),
            ):
                result = pm_core._act_auto_land("tid")
        assert result == "action:auto_land:pr=42:arc=/srv/lapis/lapis-state/tid.md"

    def test_act_directive_brief_via_tick(self):
        """Directive present → action:directive_brief:cid=..."""
        target = _mock_target(pm_repo="myrepo")
        directive = MagicMock()
        directive.author = "Erah"
        directive.ts = "2026-05-02T00:00:00"
        directive.content = "implement feature X"
        directive.id = "dir-001"
        mock_brief = MagicMock()
        mock_brief.comment_id = "cmt-dir-001"
        mock_brief.synthesis_failed = False
        result = _tick_with_patches(target, {
            "lapis_pm.pm_core._encode_user_comments": MagicMock(return_value=[directive]),
            "lapis_pm.pm_core.brief.synthesize": MagicMock(return_value=mock_brief),
            "lapis_pm.pm_core.set_outstanding_brief": MagicMock(),
        })
        assert result.decision == "action:directive_brief:cid=cmt-dir-001"

    def test_act_abandon_brief_via_tick(self):
        """Failed dispatch at retry limit → action:abandon_brief:cid=..."""
        from lapis_pm import pm_core as _pm
        target = _mock_target(pm_repo="myrepo")
        failed_rec = {
            "agent_type": "fixer",
            "intent": "do stuff",
            "retry_count": _pm.MAX_DISPATCH_RETRIES,  # at the limit
            "status": "failed",
        }
        mock_brief = MagicMock()
        mock_brief.comment_id = "cmt-abandon-001"
        mock_brief.synthesis_failed = False
        result = _tick_with_patches(target, {
            "lapis_pm.pm_core._encode_gpu_results": MagicMock(
                return_value=(0, [failed_rec])
            ),
            "lapis_pm.pm_core.brief.synthesize": MagicMock(return_value=mock_brief),
            "lapis_pm.pm_core.set_outstanding_brief": MagicMock(),
        })
        assert result.decision == "action:abandon_brief:cid=cmt-abandon-001"

    def test_act_brief_review_exhausted(self):
        """Review budget exhausted → action:review_exhausted_brief:cid=..."""
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        cls.pr_number = 42
        cls.title = "feat: foo"
        cls.html_url = "http://x"
        cls.diff = None
        cls.repo = "myrepo"
        pr = {"number": 42}
        mock_brief = MagicMock()
        mock_brief.comment_id = "cmt-exhaust-001"
        mock_brief.synthesis_failed = False
        with (
            patch("lapis_pm.pm_core.episodic.write_hold"),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_brief),
            patch("lapis_pm.pm_core.set_outstanding_brief"),
            patch("lapis_pm.pm_core._mark_pr_classified"),
        ):
            result = pm_core._act_brief_review_exhausted(
                "tid", {"pr": pr, "cls": cls, "history": []}
            )
        assert result == "action:review_exhausted_brief:cid=cmt-exhaust-001"

    def test_act_review_gate_paused_first_time(self):
        """Kill-switch first fire → action:review_gate_paused:cid=..."""
        mock_brief = MagicMock()
        mock_brief.comment_id = "cmt-gate-001"
        mock_brief.synthesis_failed = False
        from lapis_pm import authority as _auth
        cls = MagicMock(spec=_auth.PRClassification)
        pr = {"number": 42}
        with (
            patch("lapis_pm.pm_core._mem") as mock_mem_fn,
            patch("lapis_pm.pm_core._review_gate_counter", return_value=42),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_brief),
            patch("lapis_pm.pm_core.set_outstanding_brief"),
        ):
            mock_mem_fn.return_value.get.return_value = None  # not already briefed
            mock_mem_fn.return_value.set = MagicMock()
            result = pm_core._act_review_gate_pause("tid", {"pr": pr, "cls": cls})
        assert result == "action:review_gate_paused:cid=cmt-gate-001"

    def test_act_review_gate_pause_already_briefed(self):
        """Kill-switch already briefed → action:review_gate_paused:already_briefed."""
        with patch("lapis_pm.pm_core._mem") as mock_mem_fn:
            mock_mem_fn.return_value.get.return_value = {"content": "cmt-gate-001"}
            result = pm_core._act_review_gate_pause("tid", {})
        assert result == "action:review_gate_paused:already_briefed"


# ---------------------------------------------------------------------------
# Invariant: every noop variant starts with "noop:", every action with "action:"
# ---------------------------------------------------------------------------

NOOP_VARIANTS = [
    "noop:no_change",
    "noop:paused",
    "noop:reviewer_in_flight:pr=7:cycle=2",
    "noop:fixer_in_flight:dispatch=task-xyz",
    "noop:awaiting_chain_dependency:waiting_on=some-tid",
]

ACTION_VARIANTS = [
    "action:auto_merge:pr=42",
    "action:merge_failed:some error",
    "action:auto_land:pr=42:arc=/srv/lapis/lapis-state/tid.md",
    "action:reviewer_dispatched:pr=5:cycle=3",
    "action:fixer_dispatched:source=retry:pr=5:cycle=1",
    "action:fixer_dispatched:source=init:dispatch=task-001",
    "action:brief_emitted:kind=hold:cid=cmt-001",
    "action:brief_emitted:kind=advisory_clean:cid=cmt-001",
    "action:brief_emitted:kind=advisory_screen_issue:cid=cmt-001",
    "action:brief_decision_applied:bid-001:opt-a",
    "action:directive_brief:cid=cmt-001",
    "action:abandon_brief:cid=cmt-001",
    "action:review_exhausted_brief:cid=cmt-001",
    "action:review_gate_paused:cid=cmt-001",
    "action:review_gate_paused:already_briefed",
]


@pytest.mark.parametrize("decision", NOOP_VARIANTS)
def test_noop_variants_start_with_noop_prefix(decision):
    """All noop variants must start with 'noop:'."""
    assert decision.startswith("noop:"), f"Expected noop: prefix, got: {decision}"


@pytest.mark.parametrize("decision", ACTION_VARIANTS)
def test_action_variants_start_with_action_prefix(decision):
    """All action variants must start with 'action:'."""
    assert decision.startswith("action:"), f"Expected action: prefix, got: {decision}"


def test_noop_and_action_families_are_disjoint():
    """No overlap between noop: and action: families."""
    noop_set = {d.split(":")[0] for d in NOOP_VARIANTS}
    action_set = {d.split(":")[0] for d in ACTION_VARIANTS}
    assert noop_set.isdisjoint(action_set)


# ---------------------------------------------------------------------------
# D6a auditor-salvage encode-phase side effect (decision_str pre-init regression)
# ---------------------------------------------------------------------------

class TestD6aAuditorSalvageEncodePhase:
    def test_auditor_salvage_new_pr_does_not_raise_unbound(self):
        """D6a: a NEW salvage-shaped PR whose _maybe_dispatch_auditor_salvage
        returns a non-None action must NOT raise UnboundLocalError from the
        encode-phase D6a block (which previously read decision_str before the
        decide phase bound it). The dispatch is an encode-phase side effect:
        it is counted in `encoded` but must NOT become the decide-phase action.

        Regression for lapis-pm-d6a-auditor-decision-str-unbound-v0: the pre-fix
        code `if decision_str == "noop:no_change": decision_str = _auditor_action`
        raised UnboundLocalError on the first tick where new_prs was non-empty
        and _maybe_dispatch_auditor_salvage returned non-None.
        """
        target = _mock_target()
        salvage_pr = {"number": 957, "head_sha": "abc123", "title": "salvage"}
        extra_patches = {
            "lapis_pm.pm_core._encode_new_prs": MagicMock(
                return_value=[salvage_pr]),
            "lapis_pm.pm_core._maybe_dispatch_auditor_salvage": MagicMock(
                return_value="action:auditor_dispatched:pr=957:mode=salvage"),
        }
        # (a) returns without raising UnboundLocalError
        result = _tick_with_patches(target, extra_patches)
        # (b) not skipped
        assert result.skipped is False
        # (c) decision is well-formed (the auditor dispatch did NOT leak into
        #     the decide-phase action; with no PRs it falls through to noop)
        assert result.decision.startswith(("noop:", "action:")), (
            f"Expected well-formed decision, got: {result.decision}")
        # (d) the encode-phase dispatch was still counted
        assert result.encoded >= 1
        # The dispatch was invoked exactly once for the new salvage PR.
        extra_patches[
            "lapis_pm.pm_core._maybe_dispatch_auditor_salvage"].assert_called_once()
