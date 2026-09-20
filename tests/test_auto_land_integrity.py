"""Tests for lapis-pm-auto-land-integrity-v0 (AC1-AC6).

Auto-land eligibility integrity: slot pre-selection, PR-observation
ownership, merged-state predicate.

Coverage (spec AC6, T1-T12):
  T1  poisoned stream (legacy pm:orphan-untraceable + pm:pr=N + pm:pr-merged:N)
      with an open own PR -> _is_auto_land_eligible False (AC3).
  T2  same target minus the poison, own PR open -> False (AC3).
  T3  clean target with post-merge head-ref degradation -> True (AC3).
  T4  multi-PR target with one orphan-provenanced PR -> reduced set, True on
      the clean PR (AC3).
  T5  pre-selection: eligible A with an open own PR vs clean B, A bound-older
      -> B wins (AC4(a)).
  T6  pre-selection: lost-wedged A vs clean B -> B wins (AC4(b)).
  T7  untraceable-orphan path emits pm:orphan-pr=N and NOT pm:pr=N; the
      auto-adopted path still emits pm:pr=N (AC1).
  T8  _encode_merged_prs refuses orphan-provenanced N, encodes the rest,
      skip observation emitted exactly once per (target, N) (AC2).
  T9  _is_pr_merged merged_at semantics (AC5).
  T10 fail-soft: Forgejo unreachable during pre-selection -> tick_all
      completes (AC4 fail-soft).
  T11 reconcile-verify-failure poison: helper returns {N}, eligibility False
      (AC3 clause ii).
  T12 (i) new-shape exclusion (pm:orphan-pr=N only) still excludes;
      (ii) ownership transfer (adopted_pr_number) beats poison (AC3 carve-out).

Cage-only: no network, no GPU, no real Forgejo — all Forgejo and episodic
surfaces are monkeypatched.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core
from agents_core.comments import Comment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comment(tags: list[str], ts: str = "2026-09-12T10:00:00", content: str = "") -> Comment:
    return Comment(
        id="c-" + ts,
        ts=ts,
        author="lapis-pm",
        author_type="agent",
        content=content or "obs",
        tags=tags,
    )


def _target(tid: str, pm_repo: str = "lapis-test", paused: bool = False,
            data: dict | None = None) -> MagicMock:
    t = MagicMock()
    t.id = tid
    t.pm_bound = True
    t.pm_repo = pm_repo
    t.pm_authority = "advisory"
    t.paused = paused
    t.paused_reason = None
    t.data = data if data is not None else {}
    return t


def _pr(number: int, head_ref: str, created_at: str = "2026-09-11T09:00:00Z") -> dict:
    return {"number": number, "head": {"ref": head_ref, "sha": "a" * 40},
            "created_at": created_at, "state": "open"}


def _patch_target_store(store_get: dict[str, MagicMock] | None = None):
    """Patch pm_core.TargetStore so .get() returns the given targets."""
    mock_store = MagicMock()
    if store_get:
        mock_store.get.side_effect = lambda tid: store_get.get(tid)
    else:
        mock_store.get.return_value = None
    return patch("lapis_pm.pm_core.TargetStore", return_value=mock_store)


def _patch_mem():
    """Patch _mem() at the level production actually gates on.

    _is_auto_land_eligible's first gate is ``_mem().get(_landed_key(tid))``
    (pm_core.py:5426). Patching ``pm_core._mem`` and setting
    ``mem.get.return_value`` is the wrong MagicMock level: the patched
    function's return value is a fresh MagicMock on each call whose
    ``.get`` is not the mock's ``.get`` attribute. Patch ``_mem`` so it
    RETURNS a mock store whose ``.get`` returns None (no pm/landed entry).
    """
    mock_store = MagicMock()
    mock_store.get.return_value = None
    return patch("lapis_pm.pm_core._mem", return_value=mock_store)


# ---------------------------------------------------------------------------
# T1-T4, T11, T12 — _is_auto_land_eligible (AC3)
# ---------------------------------------------------------------------------

class TestEligibilityPoisonedStreams:

    def test_t1_poisoned_stream_open_own_pr_not_eligible(self):
        """T1 (AC3): the exact 2026-09-12 kami-rag shape.

        Stream carries pm:orphan-untraceable + pm:pr=963 (legacy poison) +
        pm:pr-merged:963, AND an open own PR #933. Pre-fix, seen={933,963},
        merged={963}, max(seen)==max(merged) and len(merged)>=1 both passed
        -> falsely eligible. Post-fix the provenance set {963} reduces both
        sets; merged becomes empty -> False via the existing gate.
        """
        tid = "kami-rag-down-attribution-v0"
        comments = [
            _comment(["pm:observation", "pm:pr=933"], ts="2026-09-11T09:00:00"),
            _comment(["pm:observation", "pm:orphan-untraceable", "pm:pr=963",
                      "pm:brief=cid-1"], ts="2026-09-12T09:00:00"),
            _comment(["pm:observation", "pm:pr-merged:963", "pm:pr=963"],
                     ts="2026-09-12T10:00:00"),
        ]
        open_prs = [_pr(933, f"lapis/{tid}/fix")]
        with (
            patch("lapis_pm.pm_core.episodic") as ep,
            _patch_mem(),
            patch("lapis_pm.pm_core._has_pending_dispatch", return_value=False),
            patch("lapis_pm.pm_core.get_open_prs", return_value=open_prs),
            _patch_target_store({tid: _target(tid, pm_repo="lapis-test")}),
        ):
            ep.all_comments.return_value = comments
            assert pm_core._orphan_provenance_pr_ids(tid) == {963}
            assert pm_core._is_auto_land_eligible(tid) is False

    def test_t2_clean_stream_open_own_pr_not_eligible(self):
        """T2 (AC3): same target minus the poison; own PR #933 open.

        seen={933}, merged={} -> the existing `if not merged or not seen`
        gate returns False. An open PR must not count as merged.
        """
        tid = "kami-rag-down-attribution-v0"
        comments = [
            _comment(["pm:observation", "pm:pr=933"], ts="2026-09-11T09:00:00"),
        ]
        with (
            patch("lapis_pm.pm_core.episodic") as ep,
            _patch_mem(),
            patch("lapis_pm.pm_core._has_pending_dispatch", return_value=False),
            _patch_target_store({tid: _target(tid, pm_repo="lapis-test")}),
        ):
            ep.all_comments.return_value = comments
            assert pm_core._orphan_provenance_pr_ids(tid) == set()
            assert pm_core._is_auto_land_eligible(tid) is False

    def test_t3_clean_target_post_merge_head_ref_eligible(self):
        """T3 (AC3): the cockpit worked case.

        seen={24}, merged={24}, head ref degraded to refs/pull/24/head
        (post-merge), branch gone (404), pr_count=1, no pending dispatch,
        no pm/landed entry -> True. AC3 must NOT branch-check the ref
        (ownership of merged PRs cannot be re-derived live).
        """
        tid = "cockpit-gw-kv-telemetry-v0"
        comments = [
            _comment(["pm:observation", "pm:pr=24"], ts="2026-09-10T09:00:00"),
            _comment(["pm:observation", "pm:pr-merged:24", "pm:pr=24"],
                     ts="2026-09-11T09:00:00"),
        ]
        with (
            patch("lapis_pm.pm_core.episodic") as ep,
            _patch_mem(),
            patch("lapis_pm.pm_core._has_pending_dispatch", return_value=False),
            patch("lapis_pm.pm_core._repo_owner", return_value=("coderag", "lapis")),
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"head": {"ref": "refs/pull/24/head"},
                                "merged_at": "2026-09-11T09:00:00Z",
                                "state": "closed"}),
            patch("lapis_pm.pm_core._forgejo_get_branch",
                  side_effect=Exception("404 not found")),
            _patch_target_store({tid: _target(tid, pm_repo="lapis/coderag",
                                              data={"pr_count": 1})}),
        ):
            ep.all_comments.return_value = comments
            assert pm_core._is_auto_land_eligible(tid) is True

    def test_t4_multi_pr_one_poisoned_reduces_set(self):
        """T4 (AC3): the cr-bundle worked case.

        seen={314,317}, merged={314,317}, 317 orphan-provenanced -> reduced
        merged={314}; pr_count=1 -> True on #314.
        """
        tid = "cr-bundle-item-agents-core-7ad2a96154"
        comments = [
            _comment(["pm:observation", "pm:pr=314"], ts="2026-09-10T09:00:00"),
            _comment(["pm:observation", "pm:pr=317"], ts="2026-09-10T10:00:00"),
            _comment(["pm:observation", "pm:orphan-untraceable", "pm:pr=317",
                      "pm:brief=cid-9"], ts="2026-09-11T08:00:00"),
            _comment(["pm:observation", "pm:pr-merged:314", "pm:pr=314"],
                     ts="2026-09-11T09:00:00"),
            _comment(["pm:observation", "pm:pr-merged:317", "pm:pr=317"],
                     ts="2026-09-11T10:00:00"),
        ]
        with (
            patch("lapis_pm.pm_core.episodic") as ep,
            _patch_mem(),
            patch("lapis_pm.pm_core._has_pending_dispatch", return_value=False),
            patch("lapis_pm.pm_core._repo_owner", return_value=("agents-core", "lapis")),
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"head": {"ref": "refs/pull/314/head"},
                                "merged_at": "2026-09-11T09:00:00Z",
                                "state": "closed"}),
            patch("lapis_pm.pm_core._forgejo_get_branch",
                  side_effect=Exception("404 not found")),
            _patch_target_store({tid: _target(tid, pm_repo="lapis/agents-core",
                                              data={"pr_count": 1})}),
        ):
            ep.all_comments.return_value = comments
            assert pm_core._orphan_provenance_pr_ids(tid) == {317}
            assert pm_core._is_auto_land_eligible(tid) is True

    def test_t11_reconcile_verify_failure_poison(self):
        """T11 (AC3 clause ii): the second not-owned write site.

        Stream carries pm:reconcile-verify-failure + pm:pr=99 +
        pm:pr-merged:99 and no other own PRs -> helper returns {99},
        eligibility False.
        """
        tid = "service-health-responder-brix-vault-rag-v0"
        comments = [
            _comment(["pm:observation", "pm:reconcile-verify-failure",
                      "pm:pr=99"], ts="2026-09-12T09:00:00"),
            _comment(["pm:observation", "pm:pr-merged:99", "pm:pr=99"],
                     ts="2026-09-12T10:00:00"),
        ]
        with (
            patch("lapis_pm.pm_core.episodic") as ep,
            _patch_mem(),
            patch("lapis_pm.pm_core._has_pending_dispatch", return_value=False),
            _patch_target_store({tid: _target(tid, pm_repo="lapis-test")}),
        ):
            ep.all_comments.return_value = comments
            assert pm_core._orphan_provenance_pr_ids(tid) == {99}
            assert pm_core._is_auto_land_eligible(tid) is False

    def test_t12_i_new_shape_exclusion(self):
        """T12(i) (AC3): post-fix streams stay excluded.

        A comment carrying ONLY pm:orphan-untraceable + pm:orphan-pr=963
        (no legacy pm:pr=963) plus pm:pr-merged:963 -> helper returns {963},
        eligibility False. An implementer matching only the legacy clause
        silently loses all future exclusion.
        """
        tid = "post-fix-poisoned-target"
        comments = [
            _comment(["pm:observation", "pm:orphan-untraceable",
                      "pm:orphan-pr=963", "pm:brief=cid-2"],
                     ts="2026-09-12T09:00:00"),
            _comment(["pm:observation", "pm:pr-merged:963"],
                     ts="2026-09-12T10:00:00"),
        ]
        with (
            patch("lapis_pm.pm_core.episodic") as ep,
            _patch_mem(),
            patch("lapis_pm.pm_core._has_pending_dispatch", return_value=False),
            _patch_target_store({tid: _target(tid, pm_repo="lapis-test")}),
        ):
            ep.all_comments.return_value = comments
            assert pm_core._orphan_provenance_pr_ids(tid) == {963}
            assert pm_core._is_auto_land_eligible(tid) is False

    def test_t12_ii_ownership_transfer_beats_poison(self):
        """T12(ii) (AC3 carve-out): ownership transfer beats poison.

        Same stream as T12(i) PLUS adopted_pr_number == 963 -> helper
        returns {} and the target is eligible on 963 (the episodic store is
        append-only; a ratified adoption must un-wedge the target).
        """
        tid = "adopted-after-poison-target"
        comments = [
            _comment(["pm:observation", "pm:orphan-untraceable",
                      "pm:orphan-pr=963", "pm:pr=963", "pm:brief=cid-2"],
                     ts="2026-09-12T09:00:00"),
            _comment(["pm:observation", "pm:pr-merged:963"],
                     ts="2026-09-12T10:00:00"),
        ]
        with (
            patch("lapis_pm.pm_core.episodic") as ep,
            _patch_mem(),
            patch("lapis_pm.pm_core._has_pending_dispatch", return_value=False),
            patch("lapis_pm.pm_core._repo_owner", return_value=("coderag", "lapis")),
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"head": {"ref": "refs/pull/963/head"},
                                "merged_at": "2026-09-12T10:00:00Z",
                                "state": "closed"}),
            patch("lapis_pm.pm_core._forgejo_get_branch",
                  side_effect=Exception("404 not found")),
            _patch_target_store({tid: _target(tid, pm_repo="lapis/coderag",
                                              data={"adopted_pr_number": 963,
                                                    "pr_count": 1})}),
        ):
            ep.all_comments.return_value = comments
            assert pm_core._orphan_provenance_pr_ids(tid) == set()
            assert pm_core._is_auto_land_eligible(tid) is True

    def test_t12_ii_late_adoption_comment_beats_poison(self):
        """T12(ii) variant: a later pm:orphan-adopted + pm:pr=N comment
        (ts after the poison) drops N from the provenance set."""
        tid = "adopted-via-comment-target"
        comments = [
            _comment(["pm:observation", "pm:orphan-untraceable",
                      "pm:pr=963", "pm:brief=cid-2"],
                     ts="2026-09-12T09:00:00"),
            _comment(["pm:observation", "pm:orphan-adopted", "pm:pr=963"],
                     ts="2026-09-12T12:00:00"),
            _comment(["pm:observation", "pm:pr-merged:963", "pm:pr=963"],
                     ts="2026-09-12T13:00:00"),
        ]
        with (
            patch("lapis_pm.pm_core.episodic") as ep,
            _patch_target_store({tid: _target(tid, pm_repo="lapis-test")}),
        ):
            ep.all_comments.return_value = comments
            assert pm_core._orphan_provenance_pr_ids(tid) == set()

    def test_early_adoption_does_not_beat_later_poison(self):
        """Carve-out is timestamped: an adoption EARLIER than the poison
        does not drop N (the poison came later; ownership was not ratified
        after the poison)."""
        tid = "early-adopt-late-poison"
        comments = [
            _comment(["pm:observation", "pm:orphan-adopted", "pm:pr=963"],
                     ts="2026-09-10T09:00:00"),
            _comment(["pm:observation", "pm:orphan-untraceable",
                      "pm:pr=963", "pm:brief=cid-3"],
                     ts="2026-09-12T09:00:00"),
        ]
        with (
            patch("lapis_pm.pm_core.episodic") as ep,
            _patch_target_store({tid: _target(tid, pm_repo="lapis-test")}),
        ):
            ep.all_comments.return_value = comments
            assert pm_core._orphan_provenance_pr_ids(tid) == {963}


# ---------------------------------------------------------------------------
# T5, T6, T10 — tick_all pre-selection (AC4)
# ---------------------------------------------------------------------------

def _run_tick_all(targets, eligible_ids, bound_ts_by_id, open_prs_by_repo=None,
                  get_open_prs_side_effect=None, load_dispatched=None):
    """Run tick_all() with Forgejo health/deploy/gem plumbing bypassed.

    Returns (results, allow_auto_land_by_id).
    """
    allow_auto_land_by_id: dict[str, bool] = {}

    def fake_tick(target_id, allow_auto_land=True):
        allow_auto_land_by_id[target_id] = allow_auto_land
        return pm_core.TickResult(target_id, False, "ok", 0, "noop:no_change")

    def fake_eligible(tid):
        return tid in eligible_ids

    def fake_bound_ts(tid):
        return bound_ts_by_id.get(tid, "9999-99-99")

    def fake_get_open_prs(repo_name, owner=None):
        if get_open_prs_side_effect is not None:
            return get_open_prs_side_effect(repo_name)
        if open_prs_by_repo is not None:
            return open_prs_by_repo.get(repo_name, [])
        return []

    def fake_load_dispatched(tid):
        if load_dispatched is not None:
            return load_dispatched.get(tid, [])
        return []

    with (
        patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(True, "")),
        patch("lapis_pm.pm_core._set_forgejo_consecutive_fails"),
        patch("lapis_pm.pm_core._check_deploy_currency"),
        patch("lapis_pm.pm_core._reconcile_deploy_inventory"),
        patch("lapis_pm.pm_core._check_tick_stalls"),
        patch("lapis_pm.pm_core._check_directive_stalls"),
        patch("lapis_pm.pm_core.TargetStore") as MockStore,
        patch("lapis_pm.pm_core._is_auto_land_eligible", side_effect=fake_eligible),
        patch("lapis_pm.pm_core._spec_bound_ts", side_effect=fake_bound_ts),
        patch("lapis_pm.pm_core.tick", side_effect=fake_tick),
        patch("lapis_pm.pm_core.get_open_prs", side_effect=fake_get_open_prs),
        patch("lapis_pm.pm_core.load_dispatched", side_effect=fake_load_dispatched),
        patch("lapis_pm.pm_core._repo_owner", return_value=("testrepo", "lapis")),
    ):
        MockStore.return_value.load_all.return_value = targets
        results = pm_core.tick_all()

    return results, allow_auto_land_by_id


class TestPreSelectionDivert:

    def test_t5_eligible_with_open_own_pr_diverts(self):
        """T5 (AC4(a)): A eligible + open own PR, B clean; A bound-older.

        A would win by age but tick() would divert it at `elif open_prs:` —
        the pre-selection must skip A and grant the slot to B.
        """
        tid_a, tid_b = "target-a-open-pr", "target-b-clean"
        a = _target(tid_a, pm_repo="lapis-test")
        b = _target(tid_b, pm_repo="lapis-test")
        bound_ts = {tid_a: "2026-01-01T00:00:00", tid_b: "2026-06-01T00:00:00"}
        open_prs_by_repo = {
            "testrepo": [_pr(30, f"lapis/{tid_a}/fix")],
        }
        _results, allow = _run_tick_all(
            [a, b],
            eligible_ids={tid_a, tid_b},
            bound_ts_by_id=bound_ts,
            open_prs_by_repo=open_prs_by_repo,
        )
        assert allow.get(tid_b) is True
        assert allow.get(tid_a) is False

    def test_t5b_adopted_pr_number_diverts(self):
        """T5 variant (AC4(a) superset): A's open PR is matched by NUMBER
        (adopted_pr_number), not by branch prefix."""
        tid_a, tid_b = "target-a-adopted", "target-b-clean2"
        a = _target(tid_a, pm_repo="lapis-test",
                    data={"adopted_pr_number": 55})
        b = _target(tid_b, pm_repo="lapis-test")
        bound_ts = {tid_a: "2026-01-01T00:00:00", tid_b: "2026-06-01T00:00:00"}
        open_prs_by_repo = {
            "testrepo": [_pr(55, "backstop/some-deviant-branch")],
        }
        _results, allow = _run_tick_all(
            [a, b],
            eligible_ids={tid_a, tid_b},
            bound_ts_by_id=bound_ts,
            open_prs_by_repo=open_prs_by_repo,
        )
        assert allow.get(tid_b) is True
        assert allow.get(tid_a) is False

    def test_t6_lost_wedged_diverts(self):
        """T6 (AC4(b)): the 2026-09-01 regression.

        A lost-wedged (terminal fixer record, no open PR, eligible-shaped)
        vs B clean -> B wins. The classifier (not raw record membership)
        decides: a terminal fixer record with no PR and forgejo_ok=True
        classifies as lost -> needs_retry.
        """
        tid_a, tid_b = "target-a-lost", "target-b-clean3"
        a = _target(tid_a, pm_repo="lapis-test")
        b = _target(tid_b, pm_repo="lapis-test")
        bound_ts = {tid_a: "2026-01-01T00:00:00", tid_b: "2026-06-01T00:00:00"}
        lost_record = {
            "gpu_id": "gpu-lost-001",
            "spec_id": "spec-lost",
            "agent_type": "fixer",
            "intent": "implement",
            "repo": "lapis-pm",
            "ts": "2026-09-01T08:00:00-07:00",
            "status": "processed",
            "retry_count": 0,
            "lost_retry_count": 0,
        }
        _results, allow = _run_tick_all(
            [a, b],
            eligible_ids={tid_a, tid_b},
            bound_ts_by_id=bound_ts,
            load_dispatched={tid_a: [lost_record]},
        )
        assert allow.get(tid_b) is True
        assert allow.get(tid_a) is False

    def test_t10_forgejo_unreachable_fail_soft(self):
        """T10 (AC4 fail-soft): get_open_prs raises during pre-selection.

        tick_all completes (no new exception class), and today's behavior is
        kept for the affected target: the oldest eligible wins (no divert
        info available).
        """
        tid_a, tid_b = "target-a-fs", "target-b-fs"
        a = _target(tid_a, pm_repo="lapis-test")
        b = _target(tid_b, pm_repo="lapis-test")
        bound_ts = {tid_a: "2026-01-01T00:00:00", tid_b: "2026-06-01T00:00:00"}

        def boom(repo_name):
            raise Exception("connection refused")

        _results, allow = _run_tick_all(
            [a, b],
            eligible_ids={tid_a, tid_b},
            bound_ts_by_id=bound_ts,
            get_open_prs_side_effect=boom,
        )
        # Fail-soft: no divert info -> today's behavior (oldest eligible wins).
        assert allow.get(tid_a) is True
        assert allow.get(tid_b) is False
        assert len(_results) == 2

    def test_preselection_no_eligible_no_fetch(self):
        """No eligible targets -> no open-PR fetches at all (the fetch is
        only additive when there is a slot to assign)."""
        a = _target("target-none", pm_repo="lapis-test")
        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(True, "")),
            patch("lapis_pm.pm_core._set_forgejo_consecutive_fails"),
            patch("lapis_pm.pm_core._check_deploy_currency"),
            patch("lapis_pm.pm_core._reconcile_deploy_inventory"),
            patch("lapis_pm.pm_core._check_tick_stalls"),
            patch("lapis_pm.pm_core._check_directive_stalls"),
            patch("lapis_pm.pm_core.TargetStore") as MockStore,
            patch("lapis_pm.pm_core._is_auto_land_eligible", return_value=False),
            patch("lapis_pm.pm_core.tick",
                  return_value=pm_core.TickResult("target-none", False, "ok", 0,
                                                  "noop:no_change")),
            patch("lapis_pm.pm_core.get_open_prs") as mock_goprs,
        ):
            MockStore.return_value.load_all.return_value = [a]
            pm_core.tick_all()
        mock_goprs.assert_not_called()


# ---------------------------------------------------------------------------
# T7 — AC1 observation tags
# ---------------------------------------------------------------------------

class TestUntraceableOrphanObservationTags:

    @contextlib.contextmanager
    def _patched(self, ep, brief_mod, tid="orphan-target",
                 pr_number=963, head="backstop/deviant"):
        """Common patch set for _reconcile_orphan_prs with one untraceable PR."""
        pr = {"number": pr_number, "head": {"ref": head},
              "body": "no markers here", "created_at": "2026-09-12T09:00:00Z"}
        b = MagicMock()
        b.comment_id = "cid-963"
        brief_mod.synthesize.return_value = b
        with (
            patch("lapis_pm.pm_core.brief", brief_mod),
            patch("lapis_pm.pm_core.episodic", ep),
            patch("lapis_pm.pm_core._is_pr_traceable_to_target", return_value=False),
            patch("lapis_pm.pm_core._pr_owned_by_bound_sibling", return_value=False),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core._set_brief_outstanding"),
        ):
            yield pr, b

    def test_t7_untraceable_writes_orphan_pr_not_pr(self):
        """T7 (AC1): the untraceable-orphan observation carries
        pm:orphan-pr=N and NOT pm:pr=N; the brief is still raised with the
        same options and notify priority."""
        tid = "orphan-target"
        target = _target(tid, pm_repo="lapis-test")
        ep = MagicMock()
        brief_mod = MagicMock()
        with self._patched(ep, brief_mod, tid=tid) as (pr, b):
            pm_core._reconcile_orphan_prs(tid, target, "lapis-test", [pr])

        # The brief was still raised with the same trigger/notify.
        brief_mod.synthesize.assert_called_once()
        kwargs = brief_mod.synthesize.call_args.kwargs
        assert kwargs["trigger"] == "orphan-pr-untraceable"
        assert kwargs["pr_number"] == 963
        assert kwargs["notify"] is pm_core.NotifyPriority.NORMAL

        # Exactly one observation written; tags carry pm:orphan-pr, not pm:pr.
        assert ep.write_observation.call_count == 1
        call_kwargs = ep.write_observation.call_args.kwargs
        tags = call_kwargs["extra_tags"]
        assert "pm:orphan-untraceable" in tags
        assert "pm:orphan-pr=963" in tags
        assert "pm:brief=cid-963" in tags
        assert not any(t.startswith("pm:pr=") for t in tags)

    def test_t7_adopted_path_still_writes_pr(self):
        """T7 (AC1 keep-site): the in-tick auto-adopted path STILL carries
        pm:pr=N — adopted PRs are owned by the target."""
        tid = "adopt-target"
        target = _target(tid, pm_repo="lapis-test")
        ep = MagicMock()
        brief_mod = MagicMock()
        pr = {"number": 970, "head": {"ref": "backstop/deviant2"},
              "body": "<!-- lapis-gpu-id: g1 -->\n<!-- lapis-tid: adopt-target -->",
              "created_at": "2026-09-12T09:00:00Z"}
        with (
            patch("lapis_pm.pm_core.brief", brief_mod),
            patch("lapis_pm.pm_core.episodic", ep),
            patch("lapis_pm.pm_core._is_pr_traceable_to_target", return_value=True),
        ):
            pm_core._reconcile_orphan_prs(tid, target, "lapis-test", [pr])

        assert ep.write_observation.call_count == 1
        call_kwargs = ep.write_observation.call_args.kwargs
        tags = call_kwargs["extra_tags"]
        assert "pm:orphan-adopted" in tags
        assert "pm:pr=970" in tags
        # Adopted targets record ownership in their data.
        assert target.data["adopted_pr_number"] == 970
        assert target.data["adopted_head_branch"] == "backstop/deviant2"


# ---------------------------------------------------------------------------
# T8 — AC2 _encode_merged_prs gate
# ---------------------------------------------------------------------------

class TestEncodeMergedPrsGate:

    def _stream(self, tid):
        """Stream: seen {963 (poisoned), 314 (clean)}, neither noted merged."""
        return [
            _comment(["pm:observation", "pm:pr=963", "pm:orphan-untraceable",
                      "pm:brief=cid-1"], ts="2026-09-12T09:00:00"),
            _comment(["pm:observation", "pm:pr=314"], ts="2026-09-12T08:00:00"),
        ]

    def test_t8_poisoned_n_not_encoded_clean_n_encoded(self):
        """T8 (AC2): orphan-provenanced N in seen + Forgejo says merged ->
        no pm:pr-merged:N written; non-provenanced N -> written as today;
        skip observation emitted exactly once per (target, N)."""
        tid = "encode-gate-target"
        ep = MagicMock()
        forgejo = {
            963: {"merged": True, "state": "closed",
                  "merged_at": "2026-09-12T10:00:00Z",
                  "created_at": "2026-09-11T00:00:00Z",
                  "head": {"sha": "b" * 40}},
            314: {"merged": True, "state": "closed",
                  "merged_at": "2026-09-12T11:00:00Z",
                  "created_at": "2026-09-10T00:00:00Z",
                  "head": {"sha": "c" * 40}},
        }
        with (
            patch("lapis_pm.pm_core.episodic", ep),
            patch("lapis_pm.pm_core._repo_owner", return_value=("testrepo", "lapis")),
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  side_effect=lambda repo, n, owner=None: forgejo[n]),
            _patch_target_store({tid: _target(tid, pm_repo="lapis/testrepo")}),
        ):
            ep.all_comments.return_value = self._stream(tid)
            new_obs = pm_core._encode_merged_prs(tid, "lapis/testrepo")

        assert new_obs == 1  # only 314 encoded
        written_tags = [
            c.kwargs.get("extra_tags") or c.args[2]
            for c in ep.write_observation.call_args_list
        ]
        merged_314 = [t for t in written_tags if t and "pm:pr-merged:314" in t]
        merged_963 = [t for t in written_tags if t and "pm:pr-merged:963" in t]
        assert len(merged_314) == 1
        assert len(merged_963) == 0
        skipped = [t for t in written_tags
                   if t and "pm:encode-skipped:orphan-provenance:pr=963" in t]
        assert len(skipped) == 1

    def test_t8_skip_observation_emitted_once_per_n(self):
        """T8 (AC2 de-dup): a second call with the skip tag already in the
        stream does NOT re-emit the skip observation."""
        tid = "encode-dedupe-target"
        ep = MagicMock()
        base_stream = self._stream(tid)
        skip_comment = _comment(
            ["pm:observation", "pm:encode-skipped",
             "pm:encode-skipped:orphan-provenance:pr=963"],
            ts="2026-09-12T12:00:00",
        )
        with (
            patch("lapis_pm.pm_core.episodic", ep),
            patch("lapis_pm.pm_core._repo_owner", return_value=("testrepo", "lapis")),
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  side_effect=lambda repo, n, owner=None: {
                      "merged": True, "state": "closed",
                      "merged_at": "2026-09-12T10:00:00Z",
                      "created_at": "2026-09-11T00:00:00Z",
                      "head": {"sha": "b" * 40}}),
            _patch_target_store({tid: _target(tid, pm_repo="lapis/testrepo")}),
        ):
            ep.all_comments.return_value = base_stream + [skip_comment]
            new_obs = pm_core._encode_merged_prs(tid, "lapis/testrepo")

        # new_obs counts only newly written pm:pr-merged observations: the
        # skip observation is NOT a merge observation, so even though one
        # observation (314's) was written, new_obs == 1 and the skip count
        # below is what the de-dup contract pins.
        assert new_obs == 1
        # Only the clean PR (314) was encoded; the skip was NOT re-emitted.
        written_tags = [
            c.kwargs.get("extra_tags") or c.args[2]
            for c in ep.write_observation.call_args_list
        ]
        assert sum(1 for t in written_tags
                   if t and "pm:encode-skipped:orphan-provenance:pr=963" in t) == 0
        assert sum(1 for t in written_tags if t and "pm:pr-merged:314" in t) == 1


# ---------------------------------------------------------------------------
# T9 — AC5 _is_pr_merged merged_at semantics
# ---------------------------------------------------------------------------

class TestIsPrMergedMergedAt:

    def _run(self, pr_data):
        target = _target("merged-pred-target", pm_repo="lapis/coderag")
        with (
            patch("lapis_pm.pm_core._merged_pr_numbers_observed",
                  return_value=set()),
            patch("lapis_pm.pm_core._forgejo_get_pr", return_value=pr_data),
            patch("lapis_pm.pm_core.TargetStore") as mock_store,
            patch("lapis_pm.pm_core._repo_owner", return_value=("coderag", "lapis")),
        ):
            mock_store.return_value.get.return_value = target
            return pm_core._is_pr_merged("merged-pred-target", 42)

    def test_t9_closed_with_merged_at_true(self):
        """T9 (AC5): state closed + merged_at set -> True."""
        assert self._run({"state": "closed",
                          "merged_at": "2026-09-11T12:00:00Z"}) is True

    def test_t9_closed_without_merged_at_false(self):
        """T9 (AC5): state closed + merged_at None -> False."""
        assert self._run({"state": "closed", "merged_at": None}) is False

    def test_t9_open_state_false(self):
        """T9 (AC5): state open -> False."""
        assert self._run({"state": "open"}) is False

    def test_t9_observation_short_circuit_no_forgejo_call(self):
        """T9 (AC5): the observation short-circuit still wins without a
        Forgejo call."""
        with (
            patch("lapis_pm.pm_core._merged_pr_numbers_observed",
                  return_value={42}),
            patch("lapis_pm.pm_core._forgejo_get_pr") as mock_get_pr,
        ):
            assert pm_core._is_pr_merged("merged-pred-target", 42) is True
            mock_get_pr.assert_not_called()
