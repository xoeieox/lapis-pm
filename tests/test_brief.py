"""Prompt-content contract tests for lapis-pm-brief-composer-tighten-v0.

These tests pin the load-bearing BRIEF_SYSTEM instructions and trigger_block
prompt assembly against future drift. They do NOT make live LLM calls.

Required contract test (Deliverable 5):
  - system= arg contains the grounding rule sentence
  - system= arg contains the no-absence-from-truncation rule sentence
  - prompt= arg contains "Inline screen: did not run" when screen_issues=[]
  - prompt= arg contains the Trigger: line

Routing contract (lapis-pm-prose-synthesis-local-repoint-v0):
  - synthesize() routes to call_gw_agent (the local seat), NOT call_claude_cli
  - the provenance line records the actual served_model_out (or an explicit
    'local-seat-unavailable' marker when the seat produced no text)

Optional round-trip smoke test:
  - synthesize() returns a Brief and writes a pm:brief comment when
    call_gw_agent is mocked to return a valid body.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import brief
from lapis_pm.brief import BRIEF_SYSTEM


def _fake_comment(cid: str) -> MagicMock:
    c = MagicMock()
    c.id = cid
    c.tags = ["pm:brief"]
    return c


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
      (e) call_gw_agent is invoked with json_mode=False (NOT call_claude_cli)
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

    # Capture call_gw_agent arguments
    captured = {}

    def fake_call(prompt, system, **kwargs):
        captured["prompt"] = prompt
        captured["system"] = system
        captured["kwargs"] = kwargs
        kwargs["served_model_out"].append("gravitywell-slot1")
        return "## State\nok\n## Recent activity\n- thing\n## Risk / spec deviation\nnone\n## Decision needed\nnone\n"

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("brief-cid-contract")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
    ):
        brief.synthesize(
            target_id="contract-test-tid",
            trigger="advisory-screen-issue",
            screen_issues=[],
            notify=None,
        )

    assert captured, "call_gw_agent was not called"

    # (c) Prompt contains the no-screen indicator
    assert "Inline screen: did not run" in captured["prompt"], (
        f"prompt missing 'Inline screen: did not run'; got:\n{captured['prompt']}"
    )

    # (d) Prompt contains the trigger line
    assert "Trigger: advisory-screen-issue" in captured["prompt"], (
        f"prompt missing trigger line; got:\n{captured['prompt']}"
    )

    # (e) Local seat was invoked with json_mode=False (prose brief)
    assert captured["kwargs"].get("json_mode") is False, (
        "briefs are prose; json_mode must be False"
    )
    assert captured["kwargs"].get("writeable") is False


def test_brief_prompt_inline_screen_ran_when_issues_present():
    """When screen_issues is non-empty, prompt says 'Inline screen: ran'."""
    captured = {}

    def fake_call(prompt, system, **kwargs):
        captured["prompt"] = prompt
        kwargs["served_model_out"].append("gravitywell-slot1")
        return "## State\nok\n## Recent activity\n- x\n## Risk / spec deviation\nnone\n## Decision needed\nnone\n"

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("brief-cid-ran")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
    ):
        brief.synthesize(
            target_id="contract-test-tid",
            trigger="advisory-screen-issue",
            screen_issues=[{"severity": "high", "path": "foo.py", "note": "bad thing"}],
            notify=None,
        )

    assert "Inline screen: ran" in captured["prompt"]


# ---------------------------------------------------------------------------
# Routing contract: local seat is invoked, not call_claude_cli
# ---------------------------------------------------------------------------

def test_synthesize_routes_to_local_seat_not_claude():
    """synthesize() must route to call_gw_agent, NOT call_claude_cli."""
    gw_calls = []

    def fake_gw(prompt, system, **kwargs):
        gw_calls.append(kwargs)
        kwargs["served_model_out"].append("gravitywell-slot1")
        return "## State\nok\n## Recent activity\n- x\n## Risk / spec deviation\nnone\n## Decision needed\nnone\n"

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("routing-cid")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
    ):
        result = brief.synthesize(
            target_id="routing-tid",
            trigger="advisory-clean",
            screen_issues=[],
            notify=None,
        )

    assert gw_calls, "call_gw_agent was not called"
    # json_mode must be False for prose briefs
    assert gw_calls[0].get("json_mode") is False, "briefs are prose; json_mode must be False"
    assert result.synthesis_failed is False


def test_synthesize_fallback_when_local_seat_returns_no_text():
    """When the local seat returns no text, synthesize() sets synthesis_failed=True."""
    def fake_gw(prompt, system, **kwargs):
        # Seat unreachable: served_model_out stays empty, returns None
        return None

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("fallback-cid")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
    ):
        result = brief.synthesize(
            target_id="fallback-tid",
            trigger="advisory-clean",
            screen_issues=[],
            notify=None,
        )

    assert result.synthesis_failed is True
    assert "local seat returned no text" in result.body


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

    def fake_gw(prompt, system, **kwargs):
        kwargs["served_model_out"].append("gravitywell-slot1")
        return fake_body

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("smoke-brief-cid")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
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


# ---------------------------------------------------------------------------
# Version line tests (deliverable 5 — running-version provenance)
# ---------------------------------------------------------------------------

def test_brief_body_contains_version_line():
    """Every emitted brief body carries the running-version provenance line,
    stamped with the ACTUAL seat that produced the body (served_model_out)."""
    fake_body = (
        "## State\nRunning.\n"
        "## Recent activity\n- tick\n"
        "## Risk / spec deviation\nnone\n"
        "## Decision needed\nnone\n"
    )

    written_body = {}

    def capture_write_brief(target_id, body):
        written_body["body"] = body
        return _fake_comment("version-line-cid")

    def fake_gw(prompt, system, **kwargs):
        kwargs["served_model_out"].append("gravitywell-slot1")
        return fake_body

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", side_effect=capture_write_brief),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        # Stub the git call so the test doesn't depend on the git tree
        patch("lapis_pm.brief.subprocess.run",
              return_value=subprocess.CompletedProcess([], 0, stdout="abc1234\n", stderr="")),
    ):
        result = brief.synthesize(
            target_id="version-tid",
            trigger="advisory-clean",
            notify=None,
        )

    body = written_body.get("body", result.body)
    assert "lapis_pm @" in body, f"Version line missing from brief body:\n{body}"
    assert "brief.py:" in body
    # Provenance truth-integrity: the label is the ACTUAL seat that produced the text
    assert "brief.py:gravitywell-slot1" in body, (
        f"Provenance line must record the actual seat; got:\n{body}"
    )
    # Starts with the synthesized content (version line is additive, not replacing)
    assert body.startswith(fake_body)


def test_brief_fallback_body_records_seat_unavailable():
    """The failure-fallback brief body carries the version line AND records the
    explicit 'local-seat-unavailable' marker (never a model name that did not run)."""
    written_body = {}

    def capture_write_brief(target_id, body):
        written_body["body"] = body
        return _fake_comment("fallback-version-cid")

    def fake_gw(prompt, system, **kwargs):
        # Seat down: no text, served_model_out stays empty
        return None

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", side_effect=capture_write_brief),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief.subprocess.run",
              return_value=subprocess.CompletedProcess([], 0, stdout="abc1234\n", stderr="")),
    ):
        brief.synthesize(
            target_id="fallback-version-tid",
            trigger="advisory-clean",
            notify=None,
        )

    body = written_body.get("body", "")
    assert "lapis_pm @" in body, "Version line must appear even in fallback brief"
    assert "local seat returned no text" in body
    # Provenance truth-integrity: explicit unavailable marker, not a model name
    assert "brief.py:local-seat-unavailable" in body, (
        f"Provenance line must record 'local-seat-unavailable' when the seat produced no text; got:\n{body}"
    )


def test_brief_synthesis_retries_once_on_empty():
    """First call returns empty; second call returns valid body — synthesize() uses it."""
    valid_body = (
        "## State\nRecovered.\n"
        "## Recent activity\n- retry worked\n"
        "## Risk / spec deviation\nnone\n"
        "## Decision needed\nnone\n"
    )
    call_count = {"n": 0}

    def fake_call(prompt, system, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None  # first attempt: seat down
        kwargs["served_model_out"].append("gravitywell-slot1")
        return valid_body

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("retry-brief-cid")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
    ):
        result = brief.synthesize(
            target_id="retry-tid",
            trigger="advisory-clean",
            notify=None,
        )

    assert result.body == valid_body, "Expected retry body, got placeholder"
    assert call_count["n"] == 2, f"Expected 2 calls, got {call_count['n']}"


def test_brief_synthesis_fallback_after_two_empties():
    """Both calls return empty — placeholder is returned and does not mention 'Sonnet'."""
    def fake_gw(prompt, system, **kwargs):
        return None  # both attempts: no text

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("fallback-brief-cid")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
    ):
        result = brief.synthesize(
            target_id="fallback-tid",
            trigger="advisory-clean",
            notify=None,
        )

    assert "Sonnet" not in result.body, f"Placeholder must not mention 'Sonnet'; got:\n{result.body}"
    assert "local seat returned no text" in result.body


# ---------------------------------------------------------------------------
# Synthesis-failed flag tests (acceptance criteria 1, 5)
# ---------------------------------------------------------------------------

def test_synthesize_returns_synthesis_failed_when_fallback_used():
    """When _synthesize_body() returns None, synthesize() sets synthesis_failed=True."""
    def fake_gw(prompt, system, **kwargs):
        return None

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("synthesis-failed-cid")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
    ):
        result = brief.synthesize(
            target_id="synthesis-failed-tid",
            trigger="advisory-clean",
            notify=None,
        )

    assert result.synthesis_failed is True


def test_synthesize_returns_synthesis_false_when_body_produced():
    """When _synthesize_body() returns a body, synthesis_failed=False."""
    def fake_gw(prompt, system, **kwargs):
        kwargs["served_model_out"].append("gravitywell-slot1")
        return "## State\nok\n## Recent activity\n- x\n## Risk / spec deviation\nnone\n## Decision needed\nnone\n"

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
        patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("synthesis-ok-cid")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
    ):
        result = brief.synthesize(
            target_id="synthesis-ok-tid",
            trigger="advisory-clean",
            notify=None,
        )

    assert result.synthesis_failed is False


def test_pushover_sent_when_synthesis_succeeds():
    """Pushover IS pushed when b.synthesis_failed=False."""
    valid_body = (
        "## State\nGood.\n"
        "## Recent activity\n- ok\n"
        "## Risk / spec deviation\nnone\n"
        "## Decision needed\nnone\n"
    )

    send_notification = MagicMock(return_value=True)

    def fake_gw(prompt, system, **kwargs):
        kwargs["served_model_out"].append("gravitywell-slot1")
        return valid_body

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
        patch("lapis_pm.brief.send_notification", send_notification),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
        patch("lapis_pm.brief.episodic.spec_summary", return_value="stub spec"),
        patch("lapis_pm.brief.episodic.write_brief", return_value=_fake_comment("pushover-success-cid")),
        patch("lapis_pm.brief.episodic.write_brief_options"),
        patch("lapis_pm.brief._brief_version_line", return_value=""),
    ):
        result = brief.synthesize(
            target_id="pushover-success-tid",
            trigger="advisory-clean",
            notify=brief.NotifyPriority.NORMAL,
        )

    assert result.synthesis_failed is False
    assert result.pushed is True, "Expected pushed=True when synthesis_failed=False"
    send_notification.assert_called_once()