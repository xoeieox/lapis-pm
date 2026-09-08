"""D7 (fixer-reception-v0, leg 2): the stale-verdict supersede.

Verdicts are keyed by PR number, not head SHA — the R2 wedge: a
needs-human verdict rendered against an OLD head re-briefs forever
(live-observed 2026-09-07 20:41 PT). The fix: each verdict is tagged
with the head sha it was rendered against (the `pm:reviewer-sha=`
render-time tag, falling back to the latest `pm:pr=N:sha=` observation
at or before the verdict's ts), and the decide phase treats a verdict
whose rendered-against sha differs from the current head as STALE —
it neither blocks nor briefs from a stale verdict.

Coverage (spec AC5):
  - _verdict_rendered_sha's preference order: the verdict comment's own
    pm:reviewer-sha= tag; else the latest pm:pr=N:sha= observation at or
    before the verdict ts; else the latest overall; else None (fresh —
    the `sha=unknown` perceive-miss rule);
  - the D7 sha partition in _decide_for_pr: a stale verdict does NOT
    hold-brief from the stale verdict (the stale note is attached and
    the decision is a fresh re-review / exhausted path, per the
    budget-exhausted partition), and a matching-sha verdict behaves
    EXACTLY as today (the existing needs-human → hold_brief path).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents_core.comments import Comment

from lapis_pm import pm_core
from lapis_pm import authority


TID = "stale-test-target"
PR = 42
OLD_SHA = "old0000000000000000000000000000000000000"
NEW_SHA = "new0000000000000000000000000000000000000"


def _comment(ts: str, tags: list[str], content: str = "c") -> Comment:
    return Comment(id=f"cm-{ts}", ts=ts, author="lapis-pm",
                   author_type="agent", content=content, tags=list(tags))


def _verdict_comment(ts: str, verdict: str, sha_tag: str | None = None,
                     content: str | None = None) -> Comment:
    tags = [f"pm:reviewer:pr={PR}:cycle=1:verdict={verdict}", f"pm:pr={PR}"]
    if sha_tag:
        tags.append(f"pm:reviewer-sha={sha_tag}")
    content = content or f"Reviewer verdict for PR #{PR}:\n" \
        f'{{"verdict": "{verdict}", "issues": []}}'
    return _comment(ts, tags, content)


# ---------------------------------------------------------------------------
# _verdict_rendered_sha: the preference order
# ---------------------------------------------------------------------------

def test_rendered_sha_prefers_verdict_own_tag():
    """(1) The verdict comment's own pm:reviewer-sha= tag wins — the
    exact sha the verdict was rendered against."""
    comments = [
        _comment("2026-09-08T10:00:00", [f"pm:pr={PR}:sha={OLD_SHA}"]),
        _verdict_comment("2026-09-08T10:05:00", "needs-human",
                         sha_tag=OLD_SHA),
        # A LATER sha observation (the head advanced after the verdict)
        # must not shadow the verdict's own tag.
        _comment("2026-09-08T11:00:00", [f"pm:pr={PR}:sha={NEW_SHA}"]),
    ]
    with patch("lapis_pm.pm_core.episodic") as ep:
        ep.all_comments.return_value = comments
        assert pm_core._verdict_rendered_sha(TID, PR, "2026-09-08T10:05:00") == OLD_SHA


def test_rendered_sha_falls_back_to_sha_observation_at_or_before_ts():
    """(2) No pm:reviewer-sha= tag (a pre-tag verdict): the latest
    pm:pr=N:sha= observation at or before the verdict's ts."""
    comments = [
        _comment("2026-09-08T10:00:00", [f"pm:pr={PR}:sha={OLD_SHA}"]),
        _verdict_comment("2026-09-08T10:05:00", "needs-human"),
        _comment("2026-09-08T11:00:00", [f"pm:pr={PR}:sha={NEW_SHA}"]),
    ]
    with patch("lapis_pm.pm_core.episodic") as ep:
        ep.all_comments.return_value = comments
        assert pm_core._verdict_rendered_sha(TID, PR, "2026-09-08T10:05:00") == OLD_SHA


def test_rendered_sha_falls_back_to_latest_overall_when_no_ts():
    """(3) comment_ts unavailable (None): the latest pm:pr=N:sha=
    observation overall (pre-tag verdicts; treated as fresh when it does
    not differ from the current head)."""
    comments = [
        _comment("2026-09-08T10:00:00", [f"pm:pr={PR}:sha={OLD_SHA}"]),
        _verdict_comment("2026-09-08T10:05:00", "needs-human"),
        _comment("2026-09-08T11:00:00", [f"pm:pr={PR}:sha={NEW_SHA}"]),
    ]
    with patch("lapis_pm.pm_core.episodic") as ep:
        ep.all_comments.return_value = comments
        assert pm_core._verdict_rendered_sha(TID, PR, None) == NEW_SHA


def test_rendered_sha_none_when_nothing_observed():
    """(4) Nothing available → None: the caller treats that as FRESH
    (the `sha=unknown` perceive-miss rule — effectively unreachable)."""
    with patch("lapis_pm.pm_core.episodic") as ep:
        ep.all_comments.return_value = [
            _verdict_comment("2026-09-08T10:05:00", "needs-human"),
        ]
        assert pm_core._verdict_rendered_sha(TID, PR, "2026-09-08T10:05:00") is None


def test_rendered_sha_ignores_other_prs():
    """The sha observation must be for THIS PR."""
    comments = [
        _comment("2026-09-08T10:00:00", ["pm:pr=999:sha=other"]),
        _verdict_comment("2026-09-08T10:05:00", "needs-human"),
    ]
    with patch("lapis_pm.pm_core.episodic") as ep:
        ep.all_comments.return_value = comments
        assert pm_core._verdict_rendered_sha(TID, PR, "2026-09-08T10:05:00") is None


def test_last_review_verdict_comment_returns_verdict_comment():
    """The factored-out helper: the most recent COMPLETED verdict comment
    (pending verdicts excluded), so the D7 lookup has the verdict's ts."""
    pending = _comment("2026-09-08T09:00:00",
                       [f"pm:reviewer:pr={PR}:cycle=1:verdict=pending",
                        f"pm:pr={PR}"])
    done = _verdict_comment("2026-09-08T10:05:00", "needs-human")
    with patch("lapis_pm.pm_core.episodic") as ep:
        ep.all_comments.return_value = [pending, done]
        c = pm_core._last_review_verdict_comment(TID, PR)
        assert c is done
        ep.all_comments.return_value = [pending]
        assert pm_core._last_review_verdict_comment(TID, PR) is None


# ---------------------------------------------------------------------------
# The D7 sha partition in _decide_for_pr
# ---------------------------------------------------------------------------

def _make_cls(pm_authority: str = "advisory") -> authority.PRClassification:
    """A real PRClassification (not a MagicMock): the classify patch
    returns this instance, and _decide_for_pr reads cls.static_outcome /
    cls.verdict directly — a bare MagicMock would compare equal to
    authority.StaticOutcome.auto_hold_path (MagicMock == anything) and
    take the static-hold path instead of the review-gate loop."""
    return authority.PRClassification(
        verdict="advisory",
        screen_verdict="unknown",
        static_outcome=authority.StaticOutcome.static_pass,
        reasons=[],
        issues=[],
        pr_number=PR,
        repo="lapis-pm",
        title="test PR",
        html_url="http://forgejo/pr/42",
        changed_paths=[],
        diff_loc=10,
        diff="",
    )


def _decide_patches(comments: list, verdict: dict, *,
                    reviewer_count: int = 1, fixer_count: int = 0):
    """The standard _decide_for_pr patch set for the verdict fork:
    reviewer_count > fixer_count (the reviewer has returned a verdict).

    Returns (patches, ep_mock): enter every patch, then set
    ``ep_mock.all_comments.return_value = comments``."""
    ep = patch("lapis_pm.pm_core.episodic").start()
    patches = [
        patch("lapis_pm.authority.classify", return_value=_make_cls()),
        patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
        patch("lapis_pm.pm_core._has_pending_fixer_for_pr", return_value=False),
        patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=reviewer_count),
        patch("lapis_pm.pm_core._fixer_retry_count", return_value=fixer_count),
        patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict),
        patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
        patch("lapis_pm.pm_core._review_gate_counter", return_value=0),
        patch("lapis_pm.panel_starvation.verdict_is_starved", return_value=False),
        patch("lapis_pm.pm_core._collect_review_history", return_value=[]),
        patch("lapis_pm.pm_core._reviewer_attempt_ceiling_check",
              return_value=None),
        patch("lapis_pm.pm_core._reviewer_infra_backoff_check",
              return_value=None),
    ]
    for p in patches:
        p.start()
    ep.all_comments.return_value = comments
    return patches, ep


def test_decide_stale_verdict_does_not_hold_brief_from_stale():
    """AC5: a stored needs-human verdict whose rendered-against sha
    differs from the current head sha → the stale verdict does NOT
    drive the hold brief. The decision is not hold_brief (the stale
    note is attached to whatever path the budget partition takes); the
    pm:verdict-stale-superseded observation is written with both shas."""
    verdict = {"verdict": "needs-human", "issues": [{"path": "a.py"}]}
    comments = [
        _comment("2026-09-08T10:00:00", [f"pm:pr={PR}:sha={OLD_SHA}"]),
        _verdict_comment("2026-09-08T10:05:00", "needs-human", sha_tag=OLD_SHA),
        _comment("2026-09-08T11:00:00", [f"pm:pr={PR}:sha={NEW_SHA}"]),
    ]
    pr = {"number": PR, "head": {"sha": NEW_SHA}}
    patches, ep = _decide_patches(comments, verdict)
    try:
        decision = pm_core._decide_for_pr(TID, "lapis-pm", pr,
                                          pm_authority="advisory")
    finally:
        for p in reversed(patches):
            p.stop()
        ep.stop()
    # The stale verdict must not hold-brief from itself.
    assert decision.kind != "hold_brief", (
        f"stale verdict drove hold_brief — the R2 wedge. got {decision.kind!r}"
    )
    # The supersede observation names both shas.
    stale_obs = [c for c in ep.all_comments.return_value
                 if "pm:verdict-stale-superseded" in c.tags]
    assert stale_obs, "pm:verdict-stale-superseded observation not written"
    obs = stale_obs[0]
    assert OLD_SHA in obs.tags or OLD_SHA in obs.content
    assert NEW_SHA in obs.tags or NEW_SHA in obs.content


def test_decide_fresh_verdict_behaves_exactly_as_today():
    """AC5: a verdict whose rendered-against sha MATCHES the current head
    behaves EXACTLY as today (needs-human → hold_brief, no supersede
    observation)."""
    verdict = {"verdict": "needs-human", "issues": [{"path": "a.py"}]}
    comments = [
        _comment("2026-09-08T10:00:00", [f"pm:pr={PR}:sha={OLD_SHA}"]),
        _verdict_comment("2026-09-08T10:05:00", "needs-human", sha_tag=OLD_SHA),
    ]
    pr = {"number": PR, "head": {"sha": OLD_SHA}}
    patches, ep = _decide_patches(comments, verdict)
    try:
        decision = pm_core._decide_for_pr(TID, "lapis-pm", pr,
                                          pm_authority="advisory")
    finally:
        for p in reversed(patches):
            p.stop()
        ep.stop()
    assert decision.kind == "hold_brief"
    # No supersede observation — the sha matches.
    stale_obs = [c for c in ep.all_comments.return_value
                 if "pm:verdict-stale-superseded" in c.tags]
    assert not stale_obs


def test_decide_no_sha_observed_treated_as_fresh():
    """The `sha=unknown` rule: no rendered-against sha available (a
    Forgejo perceive miss) → treated as FRESH, behaves exactly as today."""
    verdict = {"verdict": "needs-human", "issues": [{"path": "a.py"}]}
    comments = [
        # A verdict with no sha tag and NO pm:pr=N:sha= observation.
        _verdict_comment("2026-09-08T10:05:00", "needs-human"),
    ]
    pr = {"number": PR, "head": {"sha": NEW_SHA}}
    patches, ep = _decide_patches(comments, verdict)
    try:
        decision = pm_core._decide_for_pr(TID, "lapis-pm", pr,
                                          pm_authority="advisory")
    finally:
        for p in reversed(patches):
            p.stop()
        ep.stop()
    assert decision.kind == "hold_brief"
    stale_obs = [c for c in ep.all_comments.return_value
                 if "pm:verdict-stale-superseded" in c.tags]
    assert not stale_obs
