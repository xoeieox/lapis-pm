"""Tests for no-op fixer_retry auto-land unblock (lapis-pm-failer-retry-noop-blocks-auto-land-v0).

Deliverable coverage (spec §Deliverable 4):
  (a) no-op fixer_retry where a prior sibling in the retry chain pushed the PR
      head -> tick() returns action:auto_land (not skipped:exception)
  (b) adversarial chain-scoping case, boundary-pinned: an advance observation
      with ts strictly EARLIER than the chain start does NOT mask a genuinely
      lost retry
  (c) merged-check net: no-op fixer_retry with NO advance observations but a
      live-merged PR -> not classified lost (classifier-level)
  (d) negative: no sibling advance AND PR not merged -> still classified lost;
      _act_lost_fixer_retry loud-skips (returns, does not raise), one-shot
      observation
  (e) _is_pr_merged fail-conservative semantics: Forgejo exception / 404 ->
      False, no exception propagates
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers (copied verbatim from tests/test_reconcile_lost_dispatch.py)
# ---------------------------------------------------------------------------

def _fixer_record(
    gpu_id: str = "gpu-original-001",
    ts: str = "2026-05-01T10:00:00-07:00",
    status: str = "failed",
    lost_retry_count: int = 0,
    parent_gpu_id: str | None = None,
    agent_type: str = "fixer",
    error: str | None = None,
    pr_number: int | None = None,
) -> dict:
    rec = {
        "gpu_id": gpu_id,
        "spec_id": f"spec-{gpu_id}",
        "agent_type": agent_type,
        "intent": "implement the spec",
        "repo": "lapis-pm",
        "ts": ts,
        "status": status,
        "retry_count": 0,
        "lost_retry_count": lost_retry_count,
    }
    if parent_gpu_id is not None:
        rec["parent_gpu_id"] = parent_gpu_id
    if error is not None:
        rec["error"] = error
    if pr_number is not None:
        rec["pr_number"] = pr_number
    return rec


def _comment(tags: list[str], ts: str, content: str = "") -> MagicMock:
    c = MagicMock()
    c.tags = tags
    c.ts = ts
    c.content = content
    return c


def _make_target(pm_repo: str = "lapis-pm"):
    from agents_core.targets import Target
    t = MagicMock(spec=Target)
    t.pm_bound = True
    t.paused = False
    t.pm_repo = pm_repo
    t.pm_authority = "advisory"
    t.data = {}
    return t


_TICK_BASE_PATCHES = [
    ("lapis_pm.pm_core._reconcile_dispatched_with_queue", dict(return_value=0)),
    ("lapis_pm.pm_core.get_cursor", dict(return_value=None)),
    ("lapis_pm.pm_core.set_cursor", {}),
    ("lapis_pm.pm_core.get_pause_state", dict(return_value=None)),
    ("lapis_pm.pm_core.set_pause_state", {}),
    ("lapis_pm.episodic.since", dict(return_value=[])),
    ("lapis_pm.pm_core._encode_new_prs", dict(return_value=[])),
    ("lapis_pm.pm_core._encode_pr_sha_updates", dict(return_value=0)),
    ("lapis_pm.pm_core._encode_pr_body_updates", dict(return_value=0)),
    ("lapis_pm.pm_core._encode_gpu_results", dict(return_value=(0, []))),
    ("lapis_pm.pm_core._encode_merged_prs", dict(return_value=0)),
    ("lapis_pm.pm_core._encode_user_comments", dict(return_value=[])),
    ("lapis_pm.pm_core._consume_brief_decisions", dict(return_value=None)),
    ("lapis_pm.pm_core._persist_review_state_cache", {}),
    ("lapis_pm.pm_core.episodic.spec_summary", dict(return_value="spec")),
    ("lapis_pm.pm_core.episodic.spec", dict(return_value="spec summary")),
    ("lapis_pm.episodic.all_comments", dict(return_value=[])),
    ("lapis_pm.episodic.write_observation", {}),
    ("lapis_pm.pm_core.save_dispatched", {}),
]


@contextlib.contextmanager
def _tick_ctx(records, perceive_result=([], True), extra_patches=None):
    """Context manager that patches tick() internals for lost-dispatch tests."""
    stack = contextlib.ExitStack()
    mocks = {}
    try:
        store_mock = stack.enter_context(patch("lapis_pm.pm_core.TargetStore"))
        store_mock.return_value.get.return_value = _make_target()
        mocks["TargetStore"] = store_mock

        load_mock = stack.enter_context(
            patch("lapis_pm.pm_core.load_dispatched", return_value=records)
        )
        mocks["load_dispatched"] = load_mock

        perceive_mock = stack.enter_context(
            patch("lapis_pm.pm_core._perceive_prs", return_value=perceive_result)
        )
        mocks["_perceive_prs"] = perceive_mock

        for target, kwargs in _TICK_BASE_PATCHES:
            mocks[target] = stack.enter_context(patch(target, **kwargs))

        for target, kwargs in (extra_patches or []):
            mocks[target] = stack.enter_context(patch(target, **kwargs))

        yield mocks
    finally:
        stack.close()


# ---------------------------------------------------------------------------
# (a) No-op fixer_retry whose sibling advanced the PR -> auto-land
# ---------------------------------------------------------------------------

class TestNoopRetryAutoLand:

    def test_noop_retry_sibling_advance_reaches_auto_land(self):
        """A terminal no-op fixer_retry (pr 106) whose prior sibling in the
        retry chain advanced the PR head is NOT classified lost, so the
        if/elif chain reaches the auto-land branch and fires."""
        dispatch_ts = "2026-06-01T12:00:00Z"
        rec = _fixer_record(
            gpu_id="gpu-noop-1",
            ts=dispatch_ts,
            status="processed",
            agent_type="fixer_retry",
            pr_number=106,
        )
        # Sibling-in-chain advance: a pm:pr=106:sha observation written AFTER
        # the earliest fixer_retry dispatch on PR 106 (the chain start).
        obs = _comment(
            tags=["pm:observation", "pm:pr=106", "pm:pr=106:sha=abc123"],
            ts="2026-06-01T12:05:00Z",
            content="PR #106 head SHA: abc123",
        )

        extra = [
            ("lapis_pm.episodic.all_comments", dict(return_value=[obs])),
            ("lapis_pm.pm_core._is_auto_land_eligible", dict(return_value=True)),
            ("lapis_pm.pm_core._act_auto_land",
             dict(return_value="action:auto_land:pr=106:arc=/tmp/x")),
        ]
        with _tick_ctx([rec], extra_patches=extra):
            result = pm_core.tick("my-target")

        assert result.decision.startswith("action:auto_land"), (
            f"Expected auto-land for no-op retry with sibling advance, got {result.decision}"
        )
        assert not result.decision.startswith("fixer_lost:")
        assert not result.skipped


# ---------------------------------------------------------------------------
# (b) Adversarial chain-scoping, boundary-pinned
# ---------------------------------------------------------------------------

class TestBoundaryPinnedAdversarial:

    def test_advance_before_chain_start_does_not_count(self):
        """(b1) An advance observation with ts STRICTLY EARLIER than the chain
        start ts does not satisfy _pr_advanced_since (strict > boundary)."""
        chain_start = "2026-06-01T12:00:00Z"
        obs = _comment(
            tags=["pm:observation", "pm:pr=106", "pm:pr=106:sha=oldsha"],
            ts="2026-06-01T11:00:00Z",  # strictly before chain start
            content="PR #106 head SHA: oldsha",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[obs]):
            assert pm_core._pr_advanced_since("t", 106, chain_start) is False

    def test_advance_before_chain_start_does_not_mask_lost_retry(self):
        """(b2) Classifier-level: a terminal no-op fixer_retry with only an
        EARLIER (pre-chain) advance observation is still classified lost —
        a pre-chain advance must not mask a genuinely lost retry."""
        chain_start = "2026-06-01T12:00:00Z"
        rec = _fixer_record(
            gpu_id="gpu-noop-2",
            ts=chain_start,
            status="processed",
            agent_type="fixer_retry",
            pr_number=106,
        )
        obs = _comment(
            tags=["pm:observation", "pm:pr=106", "pm:pr=106:sha=oldsha"],
            ts="2026-06-01T11:00:00Z",  # strictly before chain start
            content="PR #106 head SHA: oldsha",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[obs]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1
        assert retry[0]["gpu_id"] == "gpu-noop-2"
        assert brief == []


# ---------------------------------------------------------------------------
# (c) Merged-check net (Part C)
# ---------------------------------------------------------------------------

class TestMergedPrNet:

    def test_live_merged_pr_with_no_observations_not_lost(self):
        """A terminal no-op fixer_retry with NO advance observations at all is
        NOT classified lost when merged_pr_nums contains its PR number."""
        rec = _fixer_record(
            gpu_id="gpu-noop-3",
            ts="2026-06-01T12:00:00Z",
            status="processed",
            agent_type="fixer_retry",
            pr_number=106,
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True,
                merged_pr_nums={106},
            )
        assert retry == []
        assert brief == []

    def test_net_disabled_still_lost(self):
        """With merged_pr_nums=None (net disabled) the same record IS lost."""
        rec = _fixer_record(
            gpu_id="gpu-noop-4",
            ts="2026-06-01T12:00:00Z",
            status="processed",
            agent_type="fixer_retry",
            pr_number=106,
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True,
                merged_pr_nums=None,
            )
        assert len(retry) == 1

    def test_empty_merged_set_still_lost(self):
        """With an empty merged_pr_nums set the same record IS lost."""
        rec = _fixer_record(
            gpu_id="gpu-noop-5",
            ts="2026-06-01T12:00:00Z",
            status="processed",
            agent_type="fixer_retry",
            pr_number=106,
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True,
                merged_pr_nums=set(),
            )
        assert len(retry) == 1


# ---------------------------------------------------------------------------
# (d) Genuinely lost fixer_retry -> loud skip, no raise
# ---------------------------------------------------------------------------

class TestGenuinelyLostLoudSkip:

    def test_no_advance_no_merge_classified_lost(self):
        """(d1) A no-op fixer_retry with no advance observation and no
        merged_pr_nums is classified lost."""
        rec = _fixer_record(
            gpu_id="gpu-lost-x",
            ts="2026-06-01T12:00:00Z",
            status="processed",
            agent_type="fixer_retry",
            pr_number=106,
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1
        assert retry[0]["gpu_id"] == "gpu-lost-x"

    def test_act_lost_fixer_retry_loud_skip_returns_not_raises(self):
        """(d2) _act_lost_fixer_retry on a fixer_retry record RETURNS a
        skip decision string (does not raise) and writes exactly one
        observation carrying the skip tag. A second call with the
        observation already present writes NO second observation."""
        rec = _fixer_record(
            gpu_id="gpu-lost-x",
            ts="2026-06-01T12:00:00Z",
            status="processed",
            agent_type="fixer_retry",
            pr_number=106,
        )
        rec["intent"] = "x"

        written_obs = []

        def capture_obs(tid, content, extra_tags=None):
            written_obs.append((content, extra_tags or []))
            return MagicMock()

        # First call: no prior observation -> writes the one-shot observation.
        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.episodic.all_comments", return_value=[]),
            patch("lapis_pm.episodic.write_observation", side_effect=capture_obs),
            ):
            result = pm_core._act_lost_fixer_retry("my-target", rec)

        assert result.startswith("skip:lost_fixer_retry_undispatchable:pr=106"), (
            f"Expected loud-skip decision, got {result!r}"
        )
        assert len(written_obs) == 1, "Expected exactly one observation on first call"
        content, tags = written_obs[0]
        assert "pm:lost:fixer_retry:skipped:pr=106" in tags
        assert "gpu-lost-x" in content

        # Second call: the observation is now present in the comment stream.
        seeded = _comment(
            tags=["pm:observation", "pm:lost:fixer_retry:skipped:pr=106"],
            ts="2026-06-01T12:10:00Z",
            content=content,
        )
        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.episodic.all_comments", return_value=[seeded]),
            patch("lapis_pm.episodic.write_observation", side_effect=capture_obs),
            ):
            result2 = pm_core._act_lost_fixer_retry("my-target", rec)

        assert result2.startswith("skip:lost_fixer_retry_undispatchable:pr=106")
        assert len(written_obs) == 1, "One-shot: no second observation written"


# ---------------------------------------------------------------------------
# (e) _is_pr_merged fail-conservative semantics
# ---------------------------------------------------------------------------

class TestIsPrMergedFailConservative:

    def test_forgejo_exception_returns_false(self):
        """(e1) Forgejo raising in the live check -> False, no exception escapes."""
        with (
            patch("lapis_pm.pm_core._merged_pr_numbers_observed", return_value=set()),
            patch("lapis_pm.pm_core._forgejo_get_pr", side_effect=Exception("boom")),
        ):
            assert pm_core._is_pr_merged("t", 106) is False

    def test_forgejo_merged_state_returns_true(self):
        """(e2) Live Forgejo merged_at set -> True.

        AC5 (lapis-pm-auto-land-integrity-v0): the Forgejo API NEVER returns
        state == "merged" — merged PRs are state "closed" with merged_at set.
        The old fixture ({"state": "merged"}, no merged_at) pinned the exact
        broken predicate; the merged_at semantics are the fix.
        """
        target = MagicMock()
        target.pm_repo = "owner/repo"
        with (
            patch("lapis_pm.pm_core._merged_pr_numbers_observed", return_value=set()),
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"state": "closed",
                                "merged_at": "2026-09-11T12:00:00Z"}),
            patch("lapis_pm.pm_core.TargetStore") as mock_store,
            patch("lapis_pm.pm_core._repo_owner", return_value=("repo", "owner")),
        ):
            mock_store.return_value.get.return_value = target
            assert pm_core._is_pr_merged("t", 106) is True

    def test_forgejo_closed_without_merged_at_returns_false(self):
        """(e2b, AC5) state closed but merged_at None -> False (not merged)."""
        target = MagicMock()
        target.pm_repo = "owner/repo"
        with (
            patch("lapis_pm.pm_core._merged_pr_numbers_observed", return_value=set()),
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"state": "closed", "merged_at": None}),
            patch("lapis_pm.pm_core.TargetStore") as mock_store,
            patch("lapis_pm.pm_core._repo_owner", return_value=("repo", "owner")),
        ):
            mock_store.return_value.get.return_value = target
            assert pm_core._is_pr_merged("t", 106) is False

    def test_forgejo_open_state_returns_false(self):
        """(e3) Live Forgejo state == 'open' -> False."""
        target = MagicMock()
        target.pm_repo = "owner/repo"
        with (
            patch("lapis_pm.pm_core._merged_pr_numbers_observed", return_value=set()),
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"state": "open"}),
            patch("lapis_pm.pm_core.TargetStore") as mock_store,
            patch("lapis_pm.pm_core._repo_owner", return_value=("repo", "owner")),
        ):
            mock_store.return_value.get.return_value = target
            assert pm_core._is_pr_merged("t", 106) is False

    def test_observed_merged_short_circuits_true(self):
        """An observed merged PR short-circuits to True without a live check."""
        with (
            patch("lapis_pm.pm_core._merged_pr_numbers_observed",
                  return_value={106}),
            patch("lapis_pm.pm_core._forgejo_get_pr") as mock_get_pr,
        ):
            assert pm_core._is_pr_merged("t", 106) is True
            mock_get_pr.assert_not_called()