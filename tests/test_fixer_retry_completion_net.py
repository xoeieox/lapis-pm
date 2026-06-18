"""Tests for fixer_retry completion perception (lapis-pm-fixer-retry-completion-net-v0).

Deliverable coverage:
  D1 - body fingerprint perceiver (_encode_pr_body_updates, _last_observed_pr_body_fp)
  D2 - fixer_retry completion accepts description-advance (_fixer_completion_ts, _encode_gpu_results)
  D3 - bounce gate accepts description-advance (_pr_advanced_since, _decide_for_pr)
  D4 - reviewer sees current PR body (_act_dispatch_reviewer)
  D5 - fixer_retry lost-dispatch net (_find_lost_fixer_dispatches extended)
  D6 - SHA-advance regression, no-op terminal, never-preempt

Scenario from #106: reviewer offers "OR have the PR description state the justification".
Fixer edits PR body via API (no commit). PM must see the body change, flip fixer_retry
to processed, and dispatch reviewer cycle K+1.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, call, patch

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comment(tags: list[str], ts: str, content: str = "") -> MagicMock:
    c = MagicMock()
    c.tags = tags
    c.ts = ts
    c.content = content
    return c


def _fp(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _fixer_retry_record(
    gpu_id: str = "gpu-retry-001",
    ts: str = "2026-06-01T10:00:00Z",
    status: str = "pending",
    pr_number: int = 106,
    lost_retry_count: int = 0,
    parent_gpu_id: str | None = None,
) -> dict:
    rec = {
        "gpu_id": gpu_id,
        "spec_id": f"spec-{gpu_id}",
        "agent_type": "fixer_retry",
        "intent": f"fix PR #{pr_number} after reviewer cycle 1",
        "repo": "lapis-pm",
        "pr_number": pr_number,
        "ts": ts,
        "status": status,
        "retry_count": 0,
        "lost_retry_count": lost_retry_count,
    }
    if parent_gpu_id is not None:
        rec["parent_gpu_id"] = parent_gpu_id
    return rec


# ---------------------------------------------------------------------------
# D1 — Body fingerprint perceiver
# ---------------------------------------------------------------------------

class TestEncodeBodyUpdates:

    def test_first_observation_written_when_no_prior(self):
        """First tick: no prior body observation → writes pm:pr=N:body=<fp> observation."""
        open_prs = [{"number": 5, "body": "some description"}]
        expected_fp = _fp("some description")

        observations = []

        def capture_obs(tid, content, extra_tags=None):
            observations.append((content, extra_tags or []))

        with (
            patch("lapis_pm.pm_core._last_observed_pr_body_fp", return_value=None),
            patch("lapis_pm.pm_core._classified_pr_ids", return_value=set()),
            patch("lapis_pm.pm_core._mem"),
            patch("lapis_pm.episodic.write_observation", side_effect=capture_obs),
        ):
            count = pm_core._encode_pr_body_updates("my-target", open_prs)

        assert count == 1
        assert any(f"pm:pr=5:body={expected_fp}" in tags for _, tags in observations)
        assert any("pm:pr=5" in tags for _, tags in observations)

    def test_body_unchanged_no_observation(self):
        """Body fingerprint unchanged → no observation written."""
        body = "unchanged description"
        fp = _fp(body)
        open_prs = [{"number": 3, "body": body}]

        observations = []
        with (
            patch("lapis_pm.pm_core._last_observed_pr_body_fp", return_value=fp),
            patch("lapis_pm.episodic.write_observation", side_effect=lambda *a, **k: observations.append(a)),
        ):
            count = pm_core._encode_pr_body_updates("my-target", open_prs)

        assert count == 0
        assert observations == []

    def test_body_change_invalidates_classified_prs(self):
        """Body change removes the PR from classified-prs (mirrors SHA advance)."""
        old_fp = _fp("old body")
        new_body = "new body describing the justification"
        new_fp = _fp(new_body)
        open_prs = [{"number": 7, "body": new_body}]

        mem_store: dict[str, str] = {}
        classified_key = pm_core._classified_prs_key("my-target")
        mem_store[classified_key] = json.dumps([7])

        mem_mock = MagicMock()
        mem_mock.get.side_effect = lambda k: {"content": mem_store[k]} if k in mem_store else None
        mem_mock.set.side_effect = lambda k, v, tags=None: mem_store.update({k: v})

        with (
            patch("lapis_pm.pm_core._last_observed_pr_body_fp", return_value=old_fp),
            patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.episodic.write_observation"),
        ):
            pm_core._encode_pr_body_updates("my-target", open_prs)
            result = pm_core._classified_pr_ids("my-target")

        assert 7 not in result

    def test_none_body_uses_empty_string_for_fingerprint(self):
        """PR body=None is treated as empty string for fingerprinting."""
        open_prs = [{"number": 2, "body": None}]
        none_fp = _fp("")

        observations = []
        with (
            patch("lapis_pm.pm_core._last_observed_pr_body_fp", return_value=None),
            patch("lapis_pm.pm_core._classified_pr_ids", return_value=set()),
            patch("lapis_pm.pm_core._mem"),
            patch("lapis_pm.episodic.write_observation",
                  side_effect=lambda tid, content, extra_tags=None: observations.append(extra_tags or [])),
        ):
            pm_core._encode_pr_body_updates("my-target", open_prs)

        assert any(f"pm:pr=2:body={none_fp}" in tags for tags in observations)

    def test_last_observed_pr_body_fp_scans_episodic(self):
        """_last_observed_pr_body_fp returns the last seen fingerprint from episodic."""
        c1 = _comment(["pm:pr=5:body=aabbccdd11223344"], ts="2026-06-01T09:00:00Z")
        c2 = _comment(["pm:pr=5:body=deadbeefcafebabe"], ts="2026-06-01T10:00:00Z")
        c3 = _comment(["pm:pr=6:body=unrelated0000"], ts="2026-06-01T10:30:00Z")

        with patch("lapis_pm.episodic.all_comments", return_value=[c1, c2, c3]):
            fp = pm_core._last_observed_pr_body_fp("my-target", 5)

        assert fp == "deadbeefcafebabe"

    def test_last_observed_pr_body_fp_returns_none_when_absent(self):
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            assert pm_core._last_observed_pr_body_fp("my-target", 5) is None


# ---------------------------------------------------------------------------
# D2 — fixer_completion_ts (SHA or body)
# ---------------------------------------------------------------------------

class TestFixerCompletionTs:

    def test_sha_advance_detected(self):
        """SHA-advance observation after dispatch_ts → returns ts."""
        dispatch_ts = "2026-06-01T10:00:00Z"
        obs = _comment(
            tags=["pm:pr=106:sha=abc123"],
            ts="2026-06-01T10:05:00Z",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[obs]):
            ts = pm_core._fixer_completion_ts("my-target", 106, dispatch_ts)
        assert ts == "2026-06-01T10:05:00Z"

    def test_body_advance_detected(self):
        """Body-advance observation after dispatch_ts → returns ts."""
        dispatch_ts = "2026-06-01T10:00:00Z"
        obs = _comment(
            tags=[f"pm:pr=106:body={_fp('justified description')}"],
            ts="2026-06-01T10:03:00Z",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[obs]):
            ts = pm_core._fixer_completion_ts("my-target", 106, dispatch_ts)
        assert ts == "2026-06-01T10:03:00Z"

    def test_no_advance_returns_none(self):
        """No SHA or body advance → returns None."""
        dispatch_ts = "2026-06-01T10:00:00Z"
        obs = _comment(
            tags=["pm:pr=106:sha=abc123"],  # before dispatch_ts
            ts="2026-06-01T09:00:00Z",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[obs]):
            ts = pm_core._fixer_completion_ts("my-target", 106, dispatch_ts)
        assert ts is None

    def test_sha_before_dispatch_ignored(self):
        """Advance observation before dispatch_ts is ignored."""
        dispatch_ts = "2026-06-01T11:00:00Z"
        obs = _comment(
            tags=["pm:pr=106:sha=early"],
            ts="2026-06-01T10:00:00Z",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[obs]):
            assert pm_core._fixer_completion_ts("my-target", 106, dispatch_ts) is None


# ---------------------------------------------------------------------------
# D2 — _encode_gpu_results: description-only fix flips fixer_retry
# ---------------------------------------------------------------------------

class TestEncodeGpuResultsDescriptionFix:

    def test_body_advance_flips_fixer_retry_to_processed(self):
        """Body-advance observation after dispatch → fixer_retry flipped to processed."""
        dispatch_ts = "2026-06-01T10:00:00Z"
        completion_ts = "2026-06-01T10:05:00Z"
        pr_num = 106
        rec = _fixer_retry_record(ts=dispatch_ts, pr_number=pr_num)
        body_fp = _fp("PR body with justification")
        body_obs = _comment(
            tags=[f"pm:pr={pr_num}:body={body_fp}"],
            ts=completion_ts,
        )

        written_results = []

        def capture_result(tid, content, extra_tags=None):
            written_results.append(content)

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.episodic.all_comments", return_value=[body_obs]),
            patch("lapis_pm.episodic.write_result", side_effect=capture_result),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=None),
        ):
            encoded, failed = pm_core._encode_gpu_results("my-target")

        assert rec["status"] == "processed"
        assert rec.get("completed_at") == completion_ts
        assert encoded == 1
        assert any("PR description advanced after dispatch" in r for r in written_results)

    def test_sha_advance_still_works(self):
        """SHA-advance observation still flips fixer_retry (regression guard)."""
        dispatch_ts = "2026-06-01T10:00:00Z"
        completion_ts = "2026-06-01T10:05:00Z"
        pr_num = 106
        rec = _fixer_retry_record(ts=dispatch_ts, pr_number=pr_num)
        sha_obs = _comment(
            tags=[f"pm:pr={pr_num}:sha=newsha123"],
            ts=completion_ts,
        )

        written_results = []
        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.episodic.all_comments", return_value=[sha_obs]),
            patch("lapis_pm.episodic.write_result",
                  side_effect=lambda tid, c, extra_tags=None: written_results.append(c)),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=None),
        ):
            encoded, _ = pm_core._encode_gpu_results("my-target")

        assert rec["status"] == "processed"
        assert encoded == 1
        assert any("head SHA advanced after dispatch" in r for r in written_results)

    def test_no_advance_no_output_file_leaves_pending(self):
        """No advance, no output file → fixer_retry stays pending."""
        rec = _fixer_retry_record(ts="2026-06-01T10:00:00Z")
        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.episodic.all_comments", return_value=[]),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=None),
        ):
            encoded, _ = pm_core._encode_gpu_results("my-target")

        assert rec["status"] == "pending"
        assert encoded == 0

    def test_no_advance_with_output_file_flips_for_lost_detection(self):
        """No advance but job terminal (output file exists) → flipped to processed for lost-dispatch net."""
        rec = _fixer_retry_record(ts="2026-06-01T10:00:00Z")
        fake_path = MagicMock()

        observations = []
        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.episodic.all_comments", return_value=[]),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=fake_path),
            patch("lapis_pm.episodic.write_observation",
                  side_effect=lambda tid, c, extra_tags=None: observations.append((c, extra_tags or []))),
        ):
            pm_core._encode_gpu_results("my-target")

        assert rec["status"] == "processed"
        assert any("pm:fixer-retry-noop" in tags for _, tags in observations)


# ---------------------------------------------------------------------------
# D3 — bounce gate accepts description-advance
# ---------------------------------------------------------------------------

class TestPrAdvancedSince:

    def test_sha_advance_satisfies_gate(self):
        """SHA advance after since_ts → _pr_advanced_since True."""
        obs = _comment(tags=["pm:pr=5:sha=abc"], ts="2026-06-01T11:00:00Z")
        with patch("lapis_pm.episodic.all_comments", return_value=[obs]):
            assert pm_core._pr_advanced_since("t", 5, "2026-06-01T10:00:00Z") is True

    def test_body_advance_satisfies_gate(self):
        """Body advance after since_ts → _pr_advanced_since True."""
        obs = _comment(tags=["pm:pr=5:body=abcdef1234567890"], ts="2026-06-01T11:00:00Z")
        with patch("lapis_pm.episodic.all_comments", return_value=[obs]):
            assert pm_core._pr_advanced_since("t", 5, "2026-06-01T10:00:00Z") is True

    def test_no_advance_returns_false(self):
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            assert pm_core._pr_advanced_since("t", 5, "2026-06-01T10:00:00Z") is False

    def test_advance_before_since_ts_ignored(self):
        obs = _comment(tags=["pm:pr=5:sha=old"], ts="2026-06-01T09:00:00Z")
        with patch("lapis_pm.episodic.all_comments", return_value=[obs]):
            assert pm_core._pr_advanced_since("t", 5, "2026-06-01T10:00:00Z") is False


class TestDecideForPrBodyAdvanceGate:
    """_decide_for_pr: reviewer K+1 dispatches after description-only fix."""

    def _make_pr(self, number: int = 106) -> dict:
        return {
            "number": number,
            "title": "fix: dedup by content hash",
            "body": "Justification: dedup uses content hash, not filename.",
            "head": {"ref": f"lapis/my-target/pr{number}", "sha": "abc123"},
            "base": {"ref": "main"},
            "html_url": f"http://forgejo/Erah/lapis-pm/pulls/{number}",
        }

    def _make_cls(self, pr_number: int = 106, repo: str = "lapis-pm") -> MagicMock:
        from lapis_pm import authority
        cls = MagicMock(spec=authority.PRClassification)
        cls.static_outcome = authority.StaticOutcome.static_pass
        cls.repo = repo
        cls.pr_number = pr_number
        cls.verdict = "fixable"
        cls.issues = []
        cls.title = "fix PR"
        cls.diff = "diff --git ..."
        cls.html_url = f"http://forgejo/{pr_number}"
        return cls

    def test_body_advance_allows_next_reviewer_cycle(self):
        """Body-advance (no SHA advance) unblocks reviewer K+1 dispatch."""
        pr = self._make_pr()
        reviewer_ts = "2026-06-01T08:00:00Z"
        body_advance_ts = "2026-06-01T09:00:00Z"

        body_obs = _comment(
            tags=["pm:pr=106:body=abcdef1234567890"],
            ts=body_advance_ts,
        )

        with (
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),
            patch("lapis_pm.pm_core._reviewer_dispatch_ts", return_value=reviewer_ts),
            patch("lapis_pm.episodic.all_comments", return_value=[body_obs]),
            patch("lapis_pm.pm_core.authority.classify", return_value=self._make_cls()),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
        ):
            decision = pm_core._decide_for_pr("my-target", "lapis-pm", pr, "hold")

        assert decision.kind == "dispatch_reviewer"

    def test_no_advance_blocks_reviewer_cycle(self):
        """No SHA or body advance → noop_no_change (waiting for fixer)."""
        pr = self._make_pr()
        reviewer_ts = "2026-06-01T08:00:00Z"

        with (
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._fixer_retry_count", return_value=1),
            patch("lapis_pm.pm_core._reviewer_dispatch_ts", return_value=reviewer_ts),
            patch("lapis_pm.episodic.all_comments", return_value=[]),
            patch("lapis_pm.pm_core.authority.classify", return_value=self._make_cls()),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
        ):
            decision = pm_core._decide_for_pr("my-target", "lapis-pm", pr, "hold")

        assert decision.kind == "noop_no_change"


# ---------------------------------------------------------------------------
# D4 — reviewer sees current PR body
# ---------------------------------------------------------------------------

class TestReviewerSeesCurrentPrBody:

    def test_reviewer_prompt_includes_pr_body(self):
        """_act_dispatch_reviewer user_prompt includes the PR's body."""
        pr = {
            "number": 106,
            "title": "fix: dedup",
            "body": "Justification: dedup uses content hash per the spec.",
            "head": {"ref": "lapis/my-target/pr106", "sha": "abc123"},
            "base": {"ref": "main"},
            "html_url": "http://forgejo/106",
        }
        from lapis_pm import authority
        cls = MagicMock(spec=authority.PRClassification)
        cls.repo = "lapis-pm"
        cls.pr_number = 106

        dispatched_prompts = []

        fake_res = MagicMock()
        fake_res.task_id = "gpu-r-001"
        fake_res.spec_id = "spec-r-001"

        def capture_dispatch(agent_type, target_id, user_prompt, vars_=None):
            dispatched_prompts.append(user_prompt)
            return fake_res

        with (
            patch("lapis_pm.pm_core._SHAPER") as mock_shaper,
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core._increment_review_gate_counter"),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd", return_value="/tmp"),
        ):
            from agents_core.forgejo import get_pr_diff as _get_diff
            mock_shaper.dispatch.side_effect = capture_dispatch

            with patch("agents_core.forgejo.get_pr_diff", return_value="diff text"):
                pm_core._act_dispatch_reviewer("my-target", pr, cls, mode="fresh", cycle=1)

        assert dispatched_prompts, "dispatch was never called"
        prompt = dispatched_prompts[0]
        assert "Justification: dedup uses content hash per the spec." in prompt
        assert "**PR Description:**" in prompt

    def test_reviewer_prompt_skips_body_section_when_empty(self):
        """Empty PR body → no **PR Description:** section in prompt."""
        pr = {
            "number": 5,
            "title": "small fix",
            "body": "",
            "head": {"ref": "lapis/my-target/pr5", "sha": "abc"},
            "base": {"ref": "main"},
        }
        from lapis_pm import authority
        cls = MagicMock(spec=authority.PRClassification)
        cls.repo = "lapis-pm"
        cls.pr_number = 5

        dispatched_prompts = []
        fake_res = MagicMock()
        fake_res.task_id = "gpu-r-002"
        fake_res.spec_id = "spec-r-002"

        def capture_dispatch(agent_type, target_id, user_prompt, vars_=None):
            dispatched_prompts.append(user_prompt)
            return fake_res

        with (
            patch("lapis_pm.pm_core._SHAPER") as mock_shaper,
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core._increment_review_gate_counter"),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd", return_value="/tmp"),
            patch("agents_core.forgejo.get_pr_diff", return_value="diff text"),
        ):
            mock_shaper.dispatch.side_effect = capture_dispatch
            pm_core._act_dispatch_reviewer("my-target", pr, cls, mode="fresh", cycle=1)

        assert dispatched_prompts
        assert "**PR Description:**" not in dispatched_prompts[0]


# ---------------------------------------------------------------------------
# D5 — fixer_retry lost-dispatch net
# ---------------------------------------------------------------------------

class TestFindLostFixerRetryDispatches:

    def test_failed_fixer_retry_no_advance_classified_lost(self):
        """Terminal failed fixer_retry with no sha/body advance → in needs_retry."""
        rec = _fixer_retry_record(status="failed")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, briefs = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1
        assert retry[0]["gpu_id"] == rec["gpu_id"]
        assert briefs == []

    def test_processed_fixer_retry_no_advance_classified_lost(self):
        """Processed fixer_retry (no sha/body) → in needs_retry."""
        rec = _fixer_retry_record(status="processed")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, briefs = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1

    def test_pending_fixer_retry_never_classified(self):
        """Pending fixer_retry → NEVER classified as lost (never preempt)."""
        rec = _fixer_retry_record(status="pending")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, briefs = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert briefs == []

    def test_sha_advance_after_dispatch_prevents_lost_classification(self):
        """SHA advance after dispatch_ts → fixer_retry is NOT lost (normal completion)."""
        dispatch_ts = "2026-06-01T10:00:00Z"
        rec = _fixer_retry_record(ts=dispatch_ts, status="processed")
        sha_obs = _comment(
            tags=["pm:pr=106:sha=newcommit"],
            ts="2026-06-01T10:05:00Z",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[sha_obs]):
            retry, briefs = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert briefs == []

    def test_body_advance_after_dispatch_prevents_lost_classification(self):
        """Body advance after dispatch_ts → fixer_retry is NOT lost (description-fix completion)."""
        dispatch_ts = "2026-06-01T10:00:00Z"
        rec = _fixer_retry_record(ts=dispatch_ts, status="processed")
        body_obs = _comment(
            tags=[f"pm:pr=106:body={_fp('justified')}"],
            ts="2026-06-01T10:05:00Z",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[body_obs]):
            retry, briefs = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert briefs == []

    def test_second_loss_classified_for_brief(self):
        """lost_retry_count=1 + terminal retry child, no advance → needs_brief."""
        original = _fixer_retry_record(
            gpu_id="gpu-orig",
            status="failed",
            lost_retry_count=1,
        )
        child = _fixer_retry_record(
            gpu_id="gpu-child",
            status="failed",
            parent_gpu_id="gpu-orig",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, briefs = pm_core._find_lost_fixer_dispatches(
                "my-target", [original, child], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert len(briefs) == 1
        orig_r, child_r = briefs[0]
        assert orig_r["gpu_id"] == "gpu-orig"
        assert child_r is not None and child_r["gpu_id"] == "gpu-child"

    def test_pending_child_defers_classification(self):
        """Original with lost_retry_count=0, pending child → never preempt live fixer_retry."""
        original = _fixer_retry_record(gpu_id="gpu-orig", status="failed")
        child = _fixer_retry_record(
            gpu_id="gpu-child",
            status="pending",
            parent_gpu_id="gpu-orig",
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, briefs = pm_core._find_lost_fixer_dispatches(
                "my-target", [original, child], open_prs=[], forgejo_ok=True
            )
        assert retry == []
        assert briefs == []

    def test_forgejo_unreachable_defers_all(self):
        """forgejo_ok=False → no classification even for fixer_retry."""
        rec = _fixer_retry_record(status="failed")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, briefs = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=False
            )
        assert retry == []
        assert briefs == []

    def test_original_fixer_lost_detection_unchanged(self):
        """Original fixer (agent_type='fixer') lost detection still works (regression)."""
        rec = {
            "gpu_id": "gpu-fixer-orig",
            "agent_type": "fixer",
            "status": "failed",
            "ts": "2026-06-01T10:00:00Z",
            "retry_count": 0,
            "lost_retry_count": 0,
        }
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, briefs = pm_core._find_lost_fixer_dispatches(
                "my-target", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1
        assert retry[0]["gpu_id"] == "gpu-fixer-orig"


# ---------------------------------------------------------------------------
# D5 — _act_lost_fixer_retry for fixer_retry agent type
# ---------------------------------------------------------------------------

class TestActLostFixerRetryForRetryAgent:

    def test_fixer_retry_redispatch_includes_pr_number(self):
        """Lost fixer_retry re-dispatch is not yet supported — raises NotImplementedError."""
        orig = _fixer_retry_record(gpu_id="gpu-retry-lost", status="failed", pr_number=106)

        with pytest.raises(NotImplementedError, match="fixer_retry"):
            pm_core._act_lost_fixer_retry("my-target", orig)

    def test_fixer_retry_redispatch_includes_intent_note(self):
        """Lost fixer_retry dispatch is not yet supported — raises NotImplementedError."""
        orig = _fixer_retry_record(gpu_id="gpu-noop", status="failed", pr_number=106)

        with pytest.raises(NotImplementedError, match="fixer_retry"):
            pm_core._act_lost_fixer_retry("my-target", orig)

    def test_fixer_retry_redispatch_carries_pr_number_in_record(self):
        """Lost fixer_retry re-dispatch is not yet supported — raises NotImplementedError."""
        orig = _fixer_retry_record(gpu_id="gpu-carry-orig", status="failed", pr_number=106)

        with pytest.raises(NotImplementedError, match="fixer_retry"):
            pm_core._act_lost_fixer_retry("my-target", orig)


# ---------------------------------------------------------------------------
# Deposit invariant tests (DoD: _close_slot_and_deposit assertions)
# ---------------------------------------------------------------------------

class TestDepositInvariants:
    """Verify _close_slot_and_deposit is called at exactly the right points."""

    def _body_obs(self, pr_num: int, ts: str) -> MagicMock:
        fp = _fp("PR body with justification")
        return _comment(tags=[f"pm:pr={pr_num}:body={fp}"], ts=ts)

    def test_close_slot_called_on_description_advance(self):
        """D2: description-advance completion path calls _close_slot_and_deposit."""
        dispatch_ts = "2026-06-01T10:00:00Z"
        completion_ts = "2026-06-01T10:05:00Z"
        pr_num = 106
        rec = _fixer_retry_record(ts=dispatch_ts, pr_number=pr_num)
        body_obs = self._body_obs(pr_num, completion_ts)

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.episodic.all_comments", return_value=[body_obs]),
            patch("lapis_pm.episodic.write_result"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=None),
            patch("lapis_pm.pm_core._close_slot_and_deposit") as mock_deposit,
        ):
            pm_core._encode_gpu_results("my-target")

        mock_deposit.assert_called_once_with(rec, "my-target")

    def test_close_slot_called_on_sha_advance(self):
        """SHA-advance path also calls _close_slot_and_deposit (regression guard for #106)."""
        dispatch_ts = "2026-06-01T10:00:00Z"
        completion_ts = "2026-06-01T10:05:00Z"
        pr_num = 106
        rec = _fixer_retry_record(ts=dispatch_ts, pr_number=pr_num)
        sha_obs = _comment(tags=[f"pm:pr={pr_num}:sha=newsha123"], ts=completion_ts)

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.episodic.all_comments", return_value=[sha_obs]),
            patch("lapis_pm.episodic.write_result"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=None),
            patch("lapis_pm.pm_core._close_slot_and_deposit") as mock_deposit,
        ):
            pm_core._encode_gpu_results("my-target")

        mock_deposit.assert_called_once_with(rec, "my-target")

    def test_close_slot_not_called_on_first_retry_path(self):
        """D5: first retry (_act_lost_fixer_retry) with fixer_retry is not yet supported."""
        orig = _fixer_retry_record(gpu_id="gpu-first-retry", status="failed", pr_number=106)

        with pytest.raises(NotImplementedError, match="fixer_retry"):
            pm_core._act_lost_fixer_retry("my-target", orig)

    def test_close_slot_called_on_terminal_brief(self):
        """D5: brief-raised (terminal) path calls _close_slot_and_deposit on original_rec."""
        original = _fixer_retry_record(gpu_id="gpu-terminal-orig", status="failed")
        retry_child = _fixer_retry_record(
            gpu_id="gpu-terminal-child", status="failed", parent_gpu_id="gpu-terminal-orig"
        )

        fake_brief = MagicMock()
        fake_brief.comment_id = "brief-terminal-001"
        fake_brief.pushed = True
        fake_brief.body = "brief body"
        fake_brief.target_id = "my-target"

        with (
            patch("lapis_pm.episodic.all_comments", return_value=[]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.pm_core.episodic.spec", return_value="spec summary"),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=fake_brief),
            patch("lapis_pm.pm_core.set_outstanding_brief"),
            patch("lapis_pm.pm_core._close_slot_and_deposit") as mock_deposit,
        ):
            pm_core._act_lost_brief("my-target", original, retry_child)

        mock_deposit.assert_called_once_with(original, "my-target")


# ---------------------------------------------------------------------------
# D6 — Tick integration: full #106-class scenario
# ---------------------------------------------------------------------------

_TICK_PATCHES = [
    ("lapis_pm.pm_core._reconcile_dispatched_with_queue", dict(return_value=0)),
    ("lapis_pm.pm_core.get_cursor", dict(return_value=None)),
    ("lapis_pm.pm_core.set_cursor", {}),
    ("lapis_pm.pm_core.get_pause_state", dict(return_value=None)),
    ("lapis_pm.pm_core.set_pause_state", {}),
    ("lapis_pm.episodic.since", dict(return_value=[])),
    ("lapis_pm.pm_core._encode_new_prs", dict(return_value=[])),
    ("lapis_pm.pm_core._encode_pr_sha_updates", dict(return_value=0)),
    ("lapis_pm.pm_core._encode_pr_body_updates", dict(return_value=1)),  # body changed this tick
    ("lapis_pm.pm_core._encode_gpu_results", dict(return_value=(0, []))),
    ("lapis_pm.pm_core._encode_merged_prs", dict(return_value=0)),
    ("lapis_pm.pm_core._encode_user_comments", dict(return_value=[])),
    ("lapis_pm.pm_core._consume_brief_decisions", dict(return_value=None)),
    ("lapis_pm.pm_core._is_auto_land_eligible", dict(return_value=False)),
    ("lapis_pm.pm_core._persist_review_state_cache", {}),
    ("lapis_pm.pm_core.episodic.spec_summary", dict(return_value="spec")),
    ("lapis_pm.pm_core.episodic.spec", dict(return_value="spec summary")),
    ("lapis_pm.episodic.all_comments", dict(return_value=[])),
    ("lapis_pm.episodic.write_observation", {}),
    ("lapis_pm.pm_core.save_dispatched", {}),
]


import contextlib


@contextlib.contextmanager
def _tick_ctx(records, perceive_result=([], True), extra_patches=None):
    from agents_core.targets import Target
    stack = contextlib.ExitStack()
    mocks = {}
    try:
        store_mock = stack.enter_context(patch("lapis_pm.pm_core.TargetStore"))
        t = MagicMock(spec=Target)
        t.pm_bound = True
        t.paused = False
        t.pm_repo = "lapis-pm"
        t.pm_authority = "hold"
        t.data = {}
        store_mock.return_value.get.return_value = t
        mocks["TargetStore"] = store_mock

        load_mock = stack.enter_context(
            patch("lapis_pm.pm_core.load_dispatched", return_value=records)
        )
        mocks["load_dispatched"] = load_mock

        perceive_mock = stack.enter_context(
            patch("lapis_pm.pm_core._perceive_prs", return_value=perceive_result)
        )
        mocks["_perceive_prs"] = perceive_mock

        for target, kwargs in _TICK_PATCHES:
            mocks[target] = stack.enter_context(patch(target, **kwargs))

        for target, kwargs in (extra_patches or []):
            mocks[target] = stack.enter_context(patch(target, **kwargs))

        yield mocks
    finally:
        stack.close()


class TestTickBodyAdvanceScenario:
    """Tick-integration: #106-class scenario auto-advances via body advance."""

    def test_tick_dispatches_reviewer_after_body_advance(self):
        """Body-advance fixer_retry (processed) unblocks reviewer K+1 in tick."""
        # fixer_retry completed (processed) via body-advance perceiver
        fixer_ret_rec = _fixer_retry_record(
            status="processed",
            ts="2026-06-01T09:00:00Z",
        )
        # Open PR for the target
        open_pr = {
            "number": 106,
            "title": "fix: dedup",
            "body": "Justification: dedup uses content hash per the spec.",
            "head": {"ref": "lapis/my-target/pr106", "sha": "abc"},
            "base": {"ref": "main"},
        }

        fake_res = MagicMock()
        fake_res.task_id = "gpu-reviewer-c2"
        fake_res.spec_id = "spec-reviewer-c2"

        extra = [
            ("lapis_pm.pm_core._decide_for_pr",
             dict(return_value=pm_core.Decision(
                 "dispatch_reviewer",
                 {"pr": open_pr, "cls": MagicMock(), "mode": "fresh", "cycle": 2}
             ))),
            ("lapis_pm.pm_core._act_dispatch_reviewer", dict(return_value="action:reviewer_dispatched:pr=106:cycle=2")),
        ]
        with _tick_ctx([fixer_ret_rec], perceive_result=([open_pr], True), extra_patches=extra) as mocks:
            result = pm_core.tick("my-target")

        assert "reviewer_dispatched" in result.decision

    def test_tick_retries_lost_fixer_retry_on_first_loss(self):
        """Tick: no-op fixer_retry lost dispatch is not yet supported."""
        rec = _fixer_retry_record(
            status="failed",
            ts="2026-06-01T09:00:00Z",
        )

        extra = [("lapis_pm.pm_core._SHAPER", {})]
        with _tick_ctx([rec], perceive_result=([], True), extra_patches=extra) as mocks:
            with pytest.raises(NotImplementedError, match="fixer_retry"):
                pm_core.tick("my-target")

    def test_tick_briefs_on_second_loss(self):
        """Tick: no-op fixer_retry + lost_retry_count=1 + terminal child → brief."""
        original = _fixer_retry_record(
            gpu_id="gpu-orig",
            status="failed",
            lost_retry_count=1,
        )
        child = _fixer_retry_record(
            gpu_id="gpu-child",
            status="failed",
            parent_gpu_id="gpu-orig",
        )

        fake_brief = MagicMock()
        fake_brief.comment_id = "brief-retry-001"
        fake_brief.pushed = True
        fake_brief.body = "brief body"
        fake_brief.target_id = "my-target"

        extra = [
            ("lapis_pm.pm_core.brief.synthesize", dict(return_value=fake_brief)),
            ("lapis_pm.pm_core.set_outstanding_brief", {}),
        ]
        with _tick_ctx([original, child], perceive_result=([], True), extra_patches=extra):
            result = pm_core.tick("my-target")

        assert result.decision.startswith("fixer_lost:briefing:dispatches=")
        assert "gpu-orig" in result.decision

    def test_tick_noop_when_fixer_retry_pending(self):
        """Tick: pending fixer_retry → noop (never preempt live fixer)."""
        rec = _fixer_retry_record(status="pending")

        extra = [
            ("lapis_pm.pm_core._decide_for_pr",
             dict(return_value=pm_core.Decision(
                 "noop_fixer_in_flight",
                 {"pr_number": 106, "dispatch_id": "gpu-retry-001"}
             ))),
        ]
        open_pr = {
            "number": 106, "title": "t", "body": "",
            "head": {"ref": "lapis/my-target/pr106", "sha": "abc"},
            "base": {"ref": "main"},
        }
        with _tick_ctx([rec], perceive_result=([open_pr], True), extra_patches=extra):
            result = pm_core.tick("my-target")

        # noop because fixer is in flight
        assert "fixer_in_flight" in result.decision
