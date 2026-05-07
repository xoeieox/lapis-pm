"""Tests for lapis-pm-brief-composer-anchor-reviewer-verdict-v0.

Deliverable 5 unit tests (6 total):
  1. Verdict block rendering: synthesize() with non-empty reviewer_verdict_text produces
     a prompt containing the 'Reviewer verdict:' block.
  2. Verdict block omission: synthesize() with reviewer_verdict_text=None produces no block.
  3. Empty-string is treated as None: reviewer_verdict_text="" produces no block.
  4. pm_core integration — verdict present: _act_brief on advisory-clean PR calls
     brief.synthesize with reviewer_verdict_text containing verdict=clean + confidence.
  5. pm_core integration — verdict missing: _act_brief when _last_review_verdict returns None
     passes reviewer_verdict_text=None (no crash).
  6. System-prompt rule presence: BRIEF_SYSTEM contains 'Reviewer-anchor rule' and
     the literal string 'MUST output `none`'.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import brief, pm_core
from lapis_pm.brief import BRIEF_SYSTEM
from lapis_pm import authority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_comment(cid: str = "brief-cid-anchor") -> MagicMock:
    c = MagicMock()
    c.id = cid
    c.tags = ["pm:brief"]
    return c


def _make_cls(
    pr_number: int = 99,
    issues: list | None = None,
    diff: str = "diff --git a/foo.py b/foo.py\n+pass\n",
) -> authority.PRClassification:
    return authority.PRClassification(
        verdict="advisory",
        screen_verdict="clean",
        static_outcome=authority.StaticOutcome.static_pass,
        reasons=[],
        issues=issues or [],
        pr_number=pr_number,
        repo="lapis-pm",
        title="Test PR title",
        html_url="http://forgejo/pr/99",
        changed_paths=["foo.py"],
        diff_loc=1,
        diff=diff,
    )


# ---------------------------------------------------------------------------
# 1. Verdict block rendering
# ---------------------------------------------------------------------------

def test_synthesize_includes_verdict_block_when_provided():
    """synthesize() with reviewer_verdict_text='verdict=clean; confidence=0.93'
    produces a user prompt containing 'Reviewer verdict:\\nverdict=clean; confidence=0.93\\n'.
    """
    captured = {}

    def fake_llm(prompt, system, model, timeout):
        captured["prompt"] = prompt
        return "## State\nok\n## Recent activity\n- x\n## Risk / spec deviation\nnone\n## Decision needed\nnone\n"

    with (
        patch("lapis_pm.brief.call_claude_cli", side_effect=fake_llm),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment()),
        patch("lapis_pm.brief.episodic.write_brief_options"),
    ):
        brief.synthesize(
            target_id="anchor-test-tid",
            trigger="advisory-clean",
            reviewer_verdict_text="verdict=clean; confidence=0.93",
            notify=None,
        )

    assert captured, "call_claude_cli was not called"
    assert "Reviewer verdict:\nverdict=clean; confidence=0.93\n" in captured["prompt"], (
        f"Expected 'Reviewer verdict:' block in prompt; got:\n{captured['prompt']}"
    )


# ---------------------------------------------------------------------------
# 2. Verdict block omission (None)
# ---------------------------------------------------------------------------

def test_synthesize_omits_verdict_block_when_none():
    """synthesize() with reviewer_verdict_text=None produces no 'Reviewer verdict:' block."""
    captured = {}

    def fake_llm(prompt, system, model, timeout):
        captured["prompt"] = prompt
        return "## State\nok\n## Recent activity\n- x\n## Risk / spec deviation\nnone\n## Decision needed\nnone\n"

    with (
        patch("lapis_pm.brief.call_claude_cli", side_effect=fake_llm),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment()),
        patch("lapis_pm.brief.episodic.write_brief_options"),
    ):
        brief.synthesize(
            target_id="anchor-test-tid",
            trigger="advisory-clean",
            reviewer_verdict_text=None,
            notify=None,
        )

    assert captured, "call_claude_cli was not called"
    assert "Reviewer verdict:" not in captured["prompt"], (
        f"Unexpected 'Reviewer verdict:' block in prompt when kwarg is None; got:\n{captured['prompt']}"
    )


# ---------------------------------------------------------------------------
# 3. Empty-string treated as None
# ---------------------------------------------------------------------------

def test_synthesize_omits_verdict_block_when_empty_string():
    """synthesize() with reviewer_verdict_text='' produces no 'Reviewer verdict:' block."""
    captured = {}

    def fake_llm(prompt, system, model, timeout):
        captured["prompt"] = prompt
        return "## State\nok\n## Recent activity\n- x\n## Risk / spec deviation\nnone\n## Decision needed\nnone\n"

    with (
        patch("lapis_pm.brief.call_claude_cli", side_effect=fake_llm),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment()),
        patch("lapis_pm.brief.episodic.write_brief_options"),
    ):
        brief.synthesize(
            target_id="anchor-test-tid",
            trigger="advisory-clean",
            reviewer_verdict_text="",
            notify=None,
        )

    assert captured, "call_claude_cli was not called"
    assert "Reviewer verdict:" not in captured["prompt"], (
        f"Unexpected 'Reviewer verdict:' block in prompt when kwarg is empty string; got:\n{captured['prompt']}"
    )


# ---------------------------------------------------------------------------
# 4. pm_core integration — verdict present
# ---------------------------------------------------------------------------

def test_act_brief_advisory_clean_passes_verdict_text():
    """_act_brief on advisory-clean PR with _last_review_verdict returning a clean
    verdict calls brief.synthesize with reviewer_verdict_text containing
    'verdict=clean' and 'confidence=0.93'.
    """
    cls = _make_cls(pr_number=99, issues=[])

    verdict_info = {
        "verdict": "clean",
        "confidence": 0.93,
        "issues": [],
    }

    captured_kwargs: dict = {}
    mock_brief_result = MagicMock()
    mock_brief_result.comment_id = "brief-anchor-cid"
    mock_brief_result.pushed = False

    def fake_synthesize(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_brief_result

    with (
        patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict_info),
        patch("lapis_pm.pm_core.brief.synthesize", side_effect=fake_synthesize),
        patch("lapis_pm.pm_core._mark_pr_classified"),
        patch("lapis_pm.pm_core.set_outstanding_brief_verified"),
        patch("lapis_pm.pm_core._post_write_sweep_brief"),
        patch("lapis_pm.pm_core.episodic.write_hold"),
    ):
        result = pm_core._act_brief(
            "anchor-integration-tid",
            trigger="advisory",
            hold=False,
            payload={"classification": cls},
        )

    assert "reviewer_verdict_text" in captured_kwargs, (
        "brief.synthesize was not called with reviewer_verdict_text kwarg"
    )
    rvt = captured_kwargs["reviewer_verdict_text"]
    assert rvt is not None, "reviewer_verdict_text should not be None when verdict_info present"
    assert "verdict=clean" in rvt, f"Expected 'verdict=clean' in reviewer_verdict_text; got: {rvt!r}"
    assert "confidence=0.93" in rvt, f"Expected 'confidence=0.93' in reviewer_verdict_text; got: {rvt!r}"


# ---------------------------------------------------------------------------
# 5. pm_core integration — verdict missing
# ---------------------------------------------------------------------------

def test_act_brief_advisory_clean_no_verdict_passes_none():
    """_act_brief on advisory-clean PR when _last_review_verdict returns None
    passes reviewer_verdict_text=None — no crash, graceful absence.
    """
    cls = _make_cls(pr_number=99, issues=[])

    captured_kwargs: dict = {}
    mock_brief_result = MagicMock()
    mock_brief_result.comment_id = "brief-anchor-cid-none"
    mock_brief_result.pushed = False

    def fake_synthesize(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_brief_result

    with (
        patch("lapis_pm.pm_core._last_review_verdict", return_value=None),
        patch("lapis_pm.pm_core.brief.synthesize", side_effect=fake_synthesize),
        patch("lapis_pm.pm_core._mark_pr_classified"),
        patch("lapis_pm.pm_core.set_outstanding_brief_verified"),
        patch("lapis_pm.pm_core._post_write_sweep_brief"),
        patch("lapis_pm.pm_core.episodic.write_hold"),
    ):
        result = pm_core._act_brief(
            "anchor-integration-tid",
            trigger="advisory",
            hold=False,
            payload={"classification": cls},
        )

    assert "reviewer_verdict_text" in captured_kwargs, (
        "brief.synthesize was not called with reviewer_verdict_text kwarg"
    )
    assert captured_kwargs["reviewer_verdict_text"] is None, (
        f"Expected reviewer_verdict_text=None when verdict absent; got: {captured_kwargs['reviewer_verdict_text']!r}"
    )


# ---------------------------------------------------------------------------
# 6. System-prompt rule presence
# ---------------------------------------------------------------------------

def test_brief_system_contains_reviewer_anchor_rule():
    """BRIEF_SYSTEM contains the Reviewer-anchor rule literal strings."""
    assert "Reviewer-anchor rule" in BRIEF_SYSTEM, (
        "BRIEF_SYSTEM missing 'Reviewer-anchor rule'"
    )
    assert "MUST output `none`" in BRIEF_SYSTEM, (
        "BRIEF_SYSTEM missing 'MUST output `none`'"
    )
