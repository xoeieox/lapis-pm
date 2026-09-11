"""Tests for lapis-pm-brief-click-resolve-v0 — Leg 1.

Coverage:
  - synthesize() emits sibling pm:brief-options for each closed-form trigger
  - synthesize() does NOT emit sibling for amendment-shape triggers
  - read_options() finds the sibling correctly; returns None when missing or malformed
  - _act_merge_pr, _act_force_dispatch_retry, _act_pause_target,
    _act_acknowledge_and_clear with mocked deps
  - apply_decision() idempotency probe on each action kind
  - apply_decision() returns stale_brief when outstanding brief has moved on
  - Integration: directive file → _consume_brief_decisions → file moves
    pending → processing → applied; mem reflects resolution
  - Crash-recovery: processing/ file present at tick start recovers successfully
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from lapis_pm import brief, episodic


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_fake_comment(cid: str, tags: list[str], content: str) -> MagicMock:
    c = MagicMock()
    c.id = cid
    c.tags = tags
    c.content = content
    return c


def _fake_write_brief_factory(cid: str):
    def _fake_write_brief(target_id, body, extra_tags=None):
        return _make_fake_comment(cid, ["pm:brief"] + list(extra_tags or []), body)
    return _fake_write_brief


def _fake_write_brief_options_factory(captured: list):
    def _fake(target_id, content, extra_tags=None):
        c = MagicMock()
        c.id = "brief-options-cid"
        c.tags = ["pm:brief-options"] + list(extra_tags or [])
        c.content = content
        captured.append((target_id, content))
        return c
    return _fake


# ---------------------------------------------------------------------------
# Synthesizer — sibling emission
# ---------------------------------------------------------------------------

_COMMON_SYNTH_PATCHES = [
    ("agents_core.gw_agent.call_gw_agent", "## State\nok\n## Decision needed\nnone"),
    ("lapis_pm.brief.send_notification", None),
    ("lapis_pm.brief.episodic.recall", []),
    ("lapis_pm.brief.episodic.spec_summary", "spec"),
]


def _start_synth_patches(extra=None):
    patches = {}
    for target, retval in _COMMON_SYNTH_PATCHES:
        m = patch(target, return_value=retval)
        patches[target] = m.start()
    if extra:
        for target, retval in extra:
            m = patch(target, return_value=retval)
            patches[target] = m.start()
    return patches


def _stop_synth_patches(patches):
    for m in patches.values():
        try:
            m._mock_methods  # noqa: just trigger any cleanup
        except Exception:
            pass
    for key in list(patches.keys()):
        try:
            patch(key).stop()
        except RuntimeError:
            pass


@pytest.mark.parametrize("trigger,expected_option_ids,expected_action_kinds", [
    (
        "advisory-screen-issue",
        ["A", "B", "C"],
        ["merge_pr", "force_dispatch_retry", "pause_target"],
    ),
    (
        "advisory-clean",
        ["A", "B"],
        ["merge_pr", "acknowledge_and_clear"],
    ),
])
def test_synthesize_emits_sibling_for_closed_form_triggers(
    trigger, expected_option_ids, expected_action_kinds
):
    """Closed-form triggers emit pm:brief-options sibling with correct options."""
    brief_cid = "test-brief-cid-001"
    captured_options: list = []

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value="## State\nok\n## Decision needed\nnone"),
            patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
            patch("lapis_pm.brief.episodic.spec_summary", return_value="spec"),
        patch("lapis_pm.brief.episodic.write_brief",
              side_effect=_fake_write_brief_factory(brief_cid)),
        patch("lapis_pm.brief.episodic.write_brief_options",
              side_effect=_fake_write_brief_options_factory(captured_options)),
    ):
        b = brief.synthesize("test-tid", trigger=trigger, notify=None)

    assert b.comment_id == brief_cid
    assert len(captured_options) == 1, "exactly one sibling comment written"

    target_id_written, content_written = captured_options[0]
    assert target_id_written == "test-tid"

    data = json.loads(content_written)
    assert data["brief_id"] == brief_cid
    assert data["trigger"] == trigger
    assert [o["id"] for o in data["options"]] == expected_option_ids
    assert [o["action"]["kind"] for o in data["options"]] == expected_action_kinds


def test_synthesize_emits_sibling_with_pr_number():
    """merge_pr option includes pr_number when provided to synthesize()."""
    brief_cid = "test-brief-cid-002"
    captured_options: list = []

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value="## State\nok\n## Decision needed\nnone"),
            patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
            patch("lapis_pm.brief.episodic.spec_summary", return_value="spec"),
        patch("lapis_pm.brief.episodic.write_brief",
              side_effect=_fake_write_brief_factory(brief_cid)),
        patch("lapis_pm.brief.episodic.write_brief_options",
              side_effect=_fake_write_brief_options_factory(captured_options)),
    ):
        brief.synthesize("test-tid", trigger="advisory-screen-issue", pr_number=42, notify=None)

    data = json.loads(captured_options[0][1])
    merge_opt = next(o for o in data["options"] if o["action"]["kind"] == "merge_pr")
    assert merge_opt["action"]["pr"] == 42


@pytest.mark.parametrize("trigger", [
    "stuck-daemon",
    "manual force-brief",
    "advisory PR",
    "held PR",
    "user directive: foo",
    "unknown-trigger",
    "",
])
def test_synthesize_no_sibling_for_amendment_triggers(trigger):
    """Amendment-shape triggers never emit pm:brief-options sibling."""
    captured_options: list = []

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value="## State\nok\n## Decision needed\nnone"),
            patch("lapis_pm.brief.send_notification", return_value=False),
        patch("lapis_pm.brief.episodic.recall", return_value=[]),
            patch("lapis_pm.brief.episodic.spec_summary", return_value="spec"),
        patch("lapis_pm.brief.episodic.write_brief",
              side_effect=_fake_write_brief_factory("cid-x")),
        patch("lapis_pm.brief.episodic.write_brief_options",
              side_effect=_fake_write_brief_options_factory(captured_options)),
    ):
        brief.synthesize("test-tid", trigger=trigger, notify=None)

    assert len(captured_options) == 0, f"should not emit sibling for trigger={trigger!r}"


# ---------------------------------------------------------------------------
# read_options
# ---------------------------------------------------------------------------

def test_read_options_finds_matching_sibling():
    brief_id = "brief-abc-123"
    options_payload = {
        "brief_id": brief_id,
        "trigger": "advisory-screen-issue",
        "options": [
            {"id": "A", "label": "Merge", "action": {"kind": "merge_pr", "pr": 7}},
            {"id": "B", "label": "Retry", "action": {"kind": "force_dispatch_retry"}},
        ],
    }
    options_comment = _make_fake_comment(
        "opts-cid", ["pm:brief-options"], json.dumps(options_payload)
    )

    with patch("lapis_pm.brief.episodic.all_comments", return_value=[options_comment]):
        result = brief.read_options("test-tid", brief_id)

    assert result is not None
    assert result["brief_id"] == brief_id
    assert len(result["options"]) == 2


def test_read_options_returns_none_when_missing():
    with patch("lapis_pm.brief.episodic.all_comments", return_value=[]):
        result = brief.read_options("test-tid", "nonexistent-brief-id")
    assert result is None


def test_read_options_returns_none_on_malformed_json():
    bad_comment = _make_fake_comment("opts-cid", ["pm:brief-options"], "{not valid json}")
    with patch("lapis_pm.brief.episodic.all_comments", return_value=[bad_comment]):
        result = brief.read_options("test-tid", "any-brief-id")
    assert result is None


def test_read_options_returns_none_when_options_list_missing():
    payload = {"brief_id": "b1", "trigger": "advisory-clean"}  # no "options" key
    bad_comment = _make_fake_comment("opts-cid", ["pm:brief-options"], json.dumps(payload))
    with patch("lapis_pm.brief.episodic.all_comments", return_value=[bad_comment]):
        result = brief.read_options("test-tid", "b1")
    assert result is None


def test_read_options_never_raises_on_exception():
    with patch("lapis_pm.brief.episodic.all_comments", side_effect=RuntimeError("db gone")):
        result = brief.read_options("test-tid", "any")
    assert result is None


# ---------------------------------------------------------------------------
# _act_* functions (unit, mocked deps)
# ---------------------------------------------------------------------------

def _make_options_comment(brief_id, options):
    return _make_fake_comment(
        "opts-cid", ["pm:brief-options"],
        json.dumps({"brief_id": brief_id, "trigger": "advisory-screen-issue", "options": options})
    )


def test_act_merge_pr_calls_forgejo():
    """_act_merge_pr calls agents_core.forgejo.merge_pr with the right args."""
    from lapis_pm.brief import _act_merge_pr
    mock_target = MagicMock()
    mock_target.pm_repo = "test-repo"
    with (
        patch("agents_core.targets.TargetStore") as MockStore,
        patch("agents_core.forgejo.merge_pr") as mock_merge,
    ):
        MockStore.return_value.get.return_value = mock_target
        result = _act_merge_pr("test-tid", 17)
    mock_merge.assert_called_once_with("test-repo", 17, owner=None)
    assert "17" in result


def test_act_merge_pr_raises_without_pr_number():
    from lapis_pm.brief import _act_merge_pr
    mock_target = MagicMock()
    mock_target.pm_repo = "repo"
    with patch("agents_core.targets.TargetStore") as MockStore:
        MockStore.return_value.get.return_value = mock_target
        with pytest.raises(ValueError, match="pr_number"):
            _act_merge_pr("test-tid", None)


def test_act_pause_target_sets_pause_state():
    from lapis_pm.brief import _act_pause_target
    mock_target = MagicMock()
    with (
        patch("lapis_pm.pm_core.set_pause_state") as mock_set_pause,
        patch("agents_core.targets.TargetStore") as MockStore,
    ):
        MockStore.return_value.get.return_value = mock_target
        result = _act_pause_target("test-tid")
    mock_set_pause.assert_called_once_with("test-tid", "paused")
    mock_target.set_paused.assert_called_once_with(True, reason="brief option: pause_target")
    assert "paused" in result


def test_act_acknowledge_and_clear_returns_detail():
    from lapis_pm.brief import _act_acknowledge_and_clear
    result = _act_acknowledge_and_clear("test-tid")
    assert "clear" in result.lower()


def test_act_force_dispatch_retry_calls_force_dispatch():
    from lapis_pm.brief import _act_force_dispatch_retry
    with patch("lapis_pm.pm_core.force_dispatch", return_value="task-xyz") as mock_fd:
        result = _act_force_dispatch_retry("test-tid", "brief-cid-001")
    mock_fd.assert_called_once()
    assert mock_fd.call_args[0][0] == "test-tid"
    assert mock_fd.call_args[0][1] == "fixer"
    assert "brief-cid-001" in mock_fd.call_args[0][2]
    assert "Spec is source of truth" in mock_fd.call_args[0][2]
    assert "task-xyz" in result


# ---------------------------------------------------------------------------
# apply_decision — idempotency + stale brief
# ---------------------------------------------------------------------------

def _make_full_options_comment(brief_id, action_kind, pr=None):
    action = {"kind": action_kind}
    if pr is not None:
        action["pr"] = pr
    return _make_options_comment(brief_id, [
        {"id": "A", "label": "Do it", "action": action}
    ])


def _patch_pm_core_for_apply_decision(brief_id, outstanding_id, audit_content=None):
    """Return a list of patches needed to mock pm_core in apply_decision."""
    mock_mem = MagicMock()
    if audit_content is not None:
        mock_mem.get.return_value = {"content": audit_content}
    else:
        mock_mem.get.return_value = None

    return [
        patch("lapis_pm.pm_core._mem", return_value=mock_mem),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=outstanding_id),
        patch("lapis_pm.pm_core.clear_outstanding_brief"),
    ], mock_mem


@pytest.mark.parametrize("action_kind", [
    "acknowledge_and_clear",
    "pause_target",
    "merge_pr",
    "force_dispatch_retry",
])
def test_apply_decision_idempotency(action_kind):
    """Re-applying the same directive returns noop_already_applied."""
    brief_id = "idempotency-brief-001"
    audit_content = json.dumps({
        "target_id": "test-tid",
        "option_id": "A",
        "action_kind": action_kind,
        "ts": "2026-05-02T00:00:00+00:00",
        "result": "ok",
    })
    patches, mock_mem = _patch_pm_core_for_apply_decision(
        brief_id, brief_id, audit_content=audit_content
    )
    for p in patches:
        p.start()
    try:
        result = brief.apply_decision("test-tid", brief_id, "A")
    finally:
        for p in reversed(patches):
            try:
                p.stop()
            except RuntimeError:
                pass

    assert result["ok"] is True
    assert result["action_kind"] == "noop_already_applied"


def test_apply_decision_stale_brief():
    """Returns stale_brief when outstanding brief doesn't match."""
    patches, _ = _patch_pm_core_for_apply_decision("old-brief-id", "different-brief-id")
    for p in patches:
        p.start()
    try:
        result = brief.apply_decision("test-tid", "old-brief-id", "A")
    finally:
        for p in reversed(patches):
            try:
                p.stop()
            except RuntimeError:
                pass

    assert result["ok"] is False
    assert result["error"] == "stale_brief"


def test_apply_decision_options_not_found():
    patches, _ = _patch_pm_core_for_apply_decision("b1", "b1")
    for p in patches:
        p.start()
    try:
        with patch("lapis_pm.brief.episodic.all_comments", return_value=[]):
            result = brief.apply_decision("test-tid", "b1", "A")
    finally:
        for p in reversed(patches):
            try:
                p.stop()
            except RuntimeError:
                pass

    assert result["ok"] is False
    assert result["error"] == "options_not_found"


def test_apply_decision_unknown_option_id():
    brief_id = "b-unknown-opt"
    options_comment = _make_full_options_comment(brief_id, "acknowledge_and_clear")
    patches, _ = _patch_pm_core_for_apply_decision(brief_id, brief_id)
    for p in patches:
        p.start()
    try:
        with (
            patch("lapis_pm.brief.episodic.all_comments", return_value=[options_comment]),
            patch("lapis_pm.brief.episodic.write_observation"),
        ):
            result = brief.apply_decision("test-tid", brief_id, "Z")  # Z not in options
    finally:
        for p in reversed(patches):
            try:
                p.stop()
            except RuntimeError:
                pass

    assert result["ok"] is False
    assert "unknown_option_id" in result["error"]


def test_apply_decision_acknowledge_and_clear_succeeds():
    brief_id = "b-ack-001"
    options_comment = _make_full_options_comment(brief_id, "acknowledge_and_clear")
    patches, mock_mem = _patch_pm_core_for_apply_decision(brief_id, brief_id)
    clear_patch = patch("lapis_pm.pm_core.clear_outstanding_brief")
    for p in patches:
        p.start()
    mock_clear = clear_patch.start()
    try:
        with (
            patch("lapis_pm.brief.episodic.all_comments", return_value=[options_comment]),
            patch("lapis_pm.brief.episodic.write_observation"),
        ):
            result = brief.apply_decision("test-tid", brief_id, "A")
    finally:
        for p in reversed(patches):
            try:
                p.stop()
            except RuntimeError:
                pass
        try:
            clear_patch.stop()
        except RuntimeError:
            pass

    assert result["ok"] is True
    assert result["action_kind"] == "acknowledge_and_clear"
    mock_clear.assert_called_once_with("test-tid", reason="principal_decision")
    mock_mem.set.assert_called_once()


def test_apply_decision_merge_pr_succeeds():
    brief_id = "b-merge-001"
    options_comment = _make_full_options_comment(brief_id, "merge_pr", pr=99)
    mock_target = MagicMock()
    mock_target.pm_repo = "test-repo"
    patches, _ = _patch_pm_core_for_apply_decision(brief_id, brief_id)
    clear_patch = patch("lapis_pm.pm_core.clear_outstanding_brief")
    for p in patches:
        p.start()
    mock_clear = clear_patch.start()
    try:
        with (
            patch("lapis_pm.brief.episodic.all_comments", return_value=[options_comment]),
            patch("lapis_pm.brief.episodic.write_observation"),
            patch("agents_core.targets.TargetStore") as MockStore,
            patch("agents_core.forgejo.merge_pr") as mock_merge,
        ):
            MockStore.return_value.get.return_value = mock_target
            result = brief.apply_decision("test-tid", brief_id, "A")
    finally:
        for p in reversed(patches):
            try:
                p.stop()
            except RuntimeError:
                pass
        try:
            clear_patch.stop()
        except RuntimeError:
            pass

    assert result["ok"] is True
    assert result["action_kind"] == "merge_pr"
    mock_merge.assert_called_once_with("test-repo", 99, owner=None)


# ---------------------------------------------------------------------------
# Integration: directive consumer (_consume_brief_decisions via pm_core)
# ---------------------------------------------------------------------------

def test_consume_brief_decisions_pending_to_applied(tmp_path):
    """Directive file in pending → processing → applied after successful apply_decision."""
    from lapis_pm import pm_core as _pm

    brief_id = "integration-brief-001"
    target_id = "integration-target"
    option_id = "A"

    # Set up directive file at top-level (pending)
    directive = {
        "brief_id": brief_id,
        "target_id": target_id,
        "option_id": option_id,
        "ts": "2026-05-02T00:00:00-07:00",
        "submitter": "claude-view",
    }
    pending_file = tmp_path / f"{target_id}__{brief_id}.json"
    pending_file.write_text(json.dumps(directive))

    apply_result = {"ok": True, "action_kind": "acknowledge_and_clear", "detail": "cleared"}

    with (
        patch("lapis_pm.pm_core._DIRECTIVES_BASE", new=tmp_path),
            patch("lapis_pm.brief.apply_decision", return_value=apply_result),
        patch("lapis_pm.pm_core.episodic.write_observation"),
    ):
        result = _pm._consume_brief_decisions(target_id)

    assert result == f"action:brief_decision_applied:{brief_id}:{option_id}"
    assert not pending_file.exists(), "pending file should be gone"
    applied = tmp_path / "applied" / f"{target_id}__{brief_id}.json"
    assert applied.exists(), "applied file should exist"
    # Audit-trail invariant: applied file contains the augmented payload
    # (original directive + result + applied_ts), not just the original directive.
    applied_payload = json.loads(applied.read_text())
    assert applied_payload["brief_id"] == brief_id
    assert applied_payload["result"] == apply_result
    assert "applied_ts" in applied_payload


def test_consume_brief_decisions_failed_directive(tmp_path):
    """Failed apply_decision → file moves to failed/, returns None."""
    from lapis_pm import pm_core as _pm

    brief_id = "fail-brief-001"
    target_id = "fail-target"
    option_id = "B"

    directive = {"brief_id": brief_id, "target_id": target_id,
                 "option_id": option_id, "ts": "2026-05-02T00:00:00Z", "submitter": "test"}
    pending_file = tmp_path / f"{target_id}__{brief_id}.json"
    pending_file.write_text(json.dumps(directive))

    apply_result = {"ok": False, "error": "stale_brief"}

    with (
        patch("lapis_pm.pm_core._DIRECTIVES_BASE", new=tmp_path),
            patch("lapis_pm.brief.apply_decision", return_value=apply_result),
        patch("lapis_pm.pm_core.episodic.write_observation"),
    ):
        result = _pm._consume_brief_decisions(target_id)

    assert result is None
    assert not pending_file.exists()
    failed = tmp_path / "failed" / f"{target_id}__{brief_id}.json"
    assert failed.exists()
    failed_payload = json.loads(failed.read_text())
    assert failed_payload["brief_id"] == brief_id
    assert failed_payload["error"] == "stale_brief"
    assert "failed_ts" in failed_payload


def test_consume_brief_decisions_crash_recovery(tmp_path):
    """processing/ file present at tick start → re-run apply_decision and resolve."""
    from lapis_pm import pm_core as _pm

    brief_id = "crash-brief-001"
    target_id = "crash-target"
    option_id = "A"

    # File stuck in processing/ (simulates prior tick crash)
    processing_dir = tmp_path / "processing"
    processing_dir.mkdir()
    directive = {"brief_id": brief_id, "target_id": target_id, "option_id": option_id,
                 "ts": "2026-05-02T00:00:00Z", "submitter": "test"}
    processing_file = processing_dir / f"{target_id}__{brief_id}.json"
    processing_file.write_text(json.dumps(directive))

    apply_result = {"ok": True, "action_kind": "acknowledge_and_clear", "detail": "cleared"}

    with (
        patch("lapis_pm.pm_core._DIRECTIVES_BASE", new=tmp_path),
            patch("lapis_pm.brief.apply_decision", return_value=apply_result),
        patch("lapis_pm.pm_core.episodic.write_observation"),
    ):
        result = _pm._consume_brief_decisions(target_id)

    assert result == f"action:brief_decision_applied:{brief_id}:{option_id}"
    assert not processing_file.exists()
    applied = tmp_path / "applied" / f"{target_id}__{brief_id}.json"
    assert applied.exists()


def test_consume_brief_decisions_no_files(tmp_path):
    """No directive files → returns None."""
    from lapis_pm import pm_core as _pm

    with patch("lapis_pm.pm_core._DIRECTIVES_BASE", new=tmp_path):
        result = _pm._consume_brief_decisions("any-target")

    assert result is None


def test_consume_brief_decisions_only_processes_target_namespace(tmp_path):
    """Directive for a different target is not consumed."""
    from lapis_pm import pm_core as _pm

    other_target_file = tmp_path / "other-target__brief-001.json"
    other_target_file.write_text(json.dumps({
        "brief_id": "brief-001", "target_id": "other-target",
        "option_id": "A", "ts": "2026-05-02T00:00:00Z", "submitter": "test",
    }))

    with (
        patch("lapis_pm.pm_core._DIRECTIVES_BASE", new=tmp_path),
            patch("lapis_pm.pm_core.episodic.write_observation"),
    ):
        result = _pm._consume_brief_decisions("my-target")

    assert result is None
    assert other_target_file.exists(), "other target file untouched"
