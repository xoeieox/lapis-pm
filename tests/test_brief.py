"""Prompt-content contract tests for lapis-pm-brief-composer-tighten-v0.

These tests pin the load-bearing BRIEF_SYSTEM instructions and trigger_block
prompt assembly against future drift. They do NOT make live LLM calls.

Required contract test (Deliverable 5):
  - system= arg contains the grounding rule sentence
  - system= arg contains the no-absence-from-truncation rule sentence
  - prompt= arg contains "Inline screen: did not run" when screen_issues=[]
  - prompt= arg contains the Trigger: line

Optional round-trip smoke test:
  - synthesize() returns a Brief and writes a pm:brief comment when
    call_claude_cli is mocked to return a valid body.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from lapis_pm import brief
from lapis_pm.brief import BRIEF_SYSTEM


# ---------------------------------------------------------------------------
# Required: prompt-content contract test
# ---------------------------------------------------------------------------

def test_brief_prompt_contract_advisory_screen_no_issues():
    """Contract test: synthesize() with advisory-screen-issue + empty screen_issues.

    Asserts:
      (a) BRIEF_SYSTEM contains the grounding-rule sentence verbatim
      (b) BRIEF_SYSTEM contains the no-absence-from-truncation sentence verbatim
      (c) captured prompt contains "Inline screen: did not run"
      (d) captured prompt contains the Trigger: line
    """
    # (a) Grounding rule — verbatim from spec
    assert (
        "Every claim in 'Risk / spec deviation' MUST cite either "
        "(a) a `screen_issues` entry by its severity+path, or "
        "(b) a specific line or symbol from the diff section. "
        "If neither is available, output `none` for that section."
    ) in BRIEF_SYSTEM, "BRIEF_SYSTEM missing grounding rule"

    # (b) No-absence-from-truncation rule — verbatim from spec
    assert (
        "You cannot conclude something is missing or absent from a truncated diff. "
        "If the diff section ends with `(diff truncated)`, do not make absence-claims "
        "(e.g. 'missing dependency X', 'no test for Y'); consult the spec_summary for "
        "declared dependencies/structure instead, and only claim absence when the spec "
        "affirmatively says X should exist and the diff section is complete."
    ) in BRIEF_SYSTEM, "BRIEF_SYSTEM missing no-absence-from-truncation rule"

    # Capture call_claude_cli arguments
    captured = {}

    def fake_call_claude_cli(prompt, system, model, timeout):
        captured["prompt"] = prompt
        captured["system"] = system
        return "## State\nok\n## Recent activity\n- thing\n## Risk / spec deviation\nnone\n## Decision needed\nnone\n"

    fake_comment = MagicMock()
    fake_comment.id = "brief-cid-contract"
    fake_comment.tags = ["pm:brief"]

    with (
        patch("lapis_pm.brief.call_claude_cli", side_effect=fake_call_claude_cli),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=fake_comment),
        patch("lapis_pm.brief.episodic.write_brief_options"),
    ):
        brief.synthesize(
            target_id="contract-test-tid",
            trigger="advisory-screen-issue",
            screen_issues=[],
            notify=None,
        )

    assert captured, "call_claude_cli was not called"

    # (c) Prompt contains the no-screen indicator
    assert "Inline screen: did not run" in captured["prompt"], (
        f"prompt missing 'Inline screen: did not run'; got:\n{captured['prompt']}"
    )

    # (d) Prompt contains the trigger line
    assert "Trigger: advisory-screen-issue" in captured["prompt"], (
        f"prompt missing trigger line; got:\n{captured['prompt']}"
    )


def test_brief_prompt_inline_screen_ran_when_issues_present():
    """When screen_issues is non-empty, prompt says 'Inline screen: ran'."""
    captured = {}

    def fake_call_claude_cli(prompt, system, model, timeout):
        captured["prompt"] = prompt
        return "## State\nok\n## Recent activity\n- x\n## Risk / spec deviation\nnone\n## Decision needed\nnone\n"

    fake_comment = MagicMock()
    fake_comment.id = "brief-cid-ran"
    fake_comment.tags = ["pm:brief"]

    with (
        patch("lapis_pm.brief.call_claude_cli", side_effect=fake_call_claude_cli),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=fake_comment),
        patch("lapis_pm.brief.episodic.write_brief_options"),
    ):
        brief.synthesize(
            target_id="contract-test-tid",
            trigger="advisory-screen-issue",
            screen_issues=[{"severity": "high", "path": "foo.py", "note": "bad thing"}],
            notify=None,
        )

    assert "Inline screen: ran" in captured["prompt"]


# ---------------------------------------------------------------------------
# Optional: round-trip smoke test
# ---------------------------------------------------------------------------

def test_synthesize_round_trip_returns_brief():
    """synthesize() returns a Brief with comment_id when LLM call succeeds."""
    fake_body = (
        "## State\nAll good.\n"
        "## Recent activity\n- Fixer opened PR\n"
        "## Risk / spec deviation\nnone\n"
        "## Decision needed\nnone\n"
    )
    fake_comment = MagicMock()
    fake_comment.id = "smoke-brief-cid"
    fake_comment.tags = ["pm:brief"]

    with (
        patch("lapis_pm.brief.call_claude_cli", return_value=fake_body),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=fake_comment),
        patch("lapis_pm.brief.episodic.write_brief_options"),
    ):
        result = brief.synthesize(
            target_id="smoke-tid",
            trigger="advisory-screen-issue",
            screen_issues=[],
            notify=None,
        )

    assert isinstance(result, brief.Brief)
    assert result.target_id == "smoke-tid"
    assert result.comment_id == "smoke-brief-cid"
    assert result.body == fake_body
    assert result.pushed is False
