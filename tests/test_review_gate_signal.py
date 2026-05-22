"""Tests for lapis-pm-review-gate-paused-persistent-signal-v0.

Coverage:
  D1 — cmd_status shows global review-gate section (paused + unpaused), does NOT
       touch _print_target_status per-target output.
  D2 — tick_all() emits exactly one [review-gate-paused] log line when paused,
       silent when not.
  D3 — review_gate_resume() raises ValueError on empty reason, writes audit entry,
       preserves all existing behavior (counter clear, paused flag, brief key).
  D4 covered implicitly by D3 integration (stale text replaced in source).

Mock pattern for tick_all isolation
-------------------------------------
- patch `lapis_pm.pm_core.TargetStore` so `.load_all()` returns [].
- patch `lapis_pm.pm_core.probe_forgejo_health` to return (True, "ok").
- patch `lapis_pm.pm_core._set_forgejo_consecutive_fails` to no-op.
- This gives a healthy, zero-target tick_all() that exercises only the
  preamble (probe + review-gate log) without touching per-target logic or
  the Forgejo API.
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core
from lapis_pm.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_status(extra_argv=None) -> tuple[int, str, str]:
    """Run `lapis-pm status` with a minimal patched environment."""
    argv = ["status"] + (extra_argv or [])
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    mock_store = MagicMock()
    mock_store.load_all.return_value = []

    with (
        patch("lapis_pm.cli.TargetStore", return_value=mock_store),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        rc = main(argv)

    return rc, out_buf.getvalue(), err_buf.getvalue()


def _run_review_gate(sub_argv: list[str]) -> tuple[int, str, str]:
    """Run `lapis-pm review-gate <sub_argv>` and capture output."""
    argv = ["review-gate"] + sub_argv
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    with (
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        try:
            rc = main(argv)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1

    return rc, out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# D1 — cmd_status global review-gate section
# ---------------------------------------------------------------------------

def test_status_shows_global_paused_section_when_paused():
    """Global review-gate section appears with counter, threshold, and resume hint."""
    with (
        patch("lapis_pm.pm_core._review_gate_paused", return_value=True),
        patch("lapis_pm.pm_core._review_gate_counter", return_value=42),
    ):
        rc, out, _ = _run_status()

    assert rc == 0
    assert "=== review-gate ===" in out
    assert "42 / 40" in out
    assert "True" in out
    assert "--reason" in out
    # Must appear before any target section (no targets in this test so just
    # check it appears at all in output).
    assert out.index("=== review-gate ===") < len(out)


def test_status_shows_global_unpaused_section_when_healthy():
    """Global review-gate section appears without resume hint when not paused."""
    with (
        patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
        patch("lapis_pm.pm_core._review_gate_counter", return_value=12),
    ):
        rc, out, _ = _run_status()

    assert rc == 0
    assert "=== review-gate ===" in out
    assert "12 / 40" in out
    assert "False" in out
    assert "--reason" not in out


def test_status_does_not_modify_per_target_review_display():
    """_print_target_status output is byte-identical regardless of paused state.

    We compare the per-target portion of status output with paused=True vs
    paused=False. The global header changes, but the per-target section must
    not. This test verifies D1 did NOT accidentally touch _print_target_status.
    """
    from lapis_pm import cli as cli_mod
    import io

    # Build a minimal mock target that _print_target_status can render.
    mock_target = MagicMock()
    mock_target.id = "test-target-x"
    mock_target.title = "Test Target"
    mock_target.pm_bound = True
    mock_target.pm_repo = None
    mock_target.pm_authority = "advisory"
    mock_target.paused = False
    mock_target.paused_reason = None

    def _capture_per_target(paused: bool) -> str:
        out_buf = io.StringIO()
        with (
            patch("lapis_pm.pm_core._review_gate_paused", return_value=paused),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=5),
            patch("lapis_pm.pm_core.get_cursor", return_value="2026-05-22T00:00:00Z"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("agents_core.forgejo.get_open_prs", return_value=[]),
            redirect_stdout(out_buf),
        ):
            cli_mod._print_target_status(mock_target)
        return out_buf.getvalue()

    output_paused = _capture_per_target(True)
    output_unpaused = _capture_per_target(False)
    assert output_paused == output_unpaused, (
        "_print_target_status output differs when review-gate paused state changes — "
        "D1 may have accidentally touched per-target display."
    )


# ---------------------------------------------------------------------------
# D2 — tick_all paused-tick log
# ---------------------------------------------------------------------------

def _run_tick_all_capture_logs(paused: bool, target_count: int = 3) -> list[str]:
    """Run tick_all() and return captured log messages from pm_core logger."""
    mock_targets = []
    for i in range(target_count):
        t = MagicMock()
        t.id = f"target-{i}"
        t.pm_bound = True
        mock_targets.append(t)

    captured_msgs = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            captured_msgs.append(record.getMessage())

    handler = CapturingHandler()
    pm_logger = logging.getLogger("lapis_pm.pm_core")
    pm_logger.addHandler(handler)
    orig_level = pm_logger.level
    pm_logger.setLevel(logging.DEBUG)

    try:
        with (
            patch("lapis_pm.pm_core.probe_forgejo_health", return_value=(True, "ok")),
            patch("lapis_pm.pm_core._set_forgejo_consecutive_fails"),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=paused),
            patch("lapis_pm.pm_core._review_gate_counter", return_value=42),
            patch("lapis_pm.pm_core.TargetStore") as mock_store_cls,
        ):
            mock_store = MagicMock()
            mock_store.load_all.return_value = mock_targets
            mock_store_cls.return_value = mock_store

            # Patch tick() to no-op so per-target logic doesn't run.
            with patch("lapis_pm.pm_core.tick") as mock_tick:
                mock_tick.return_value = pm_core.TickResult(
                    "target-0", False, "ok", 0, "noop"
                )
                pm_core.tick_all()
    finally:
        pm_logger.removeHandler(handler)
        pm_logger.setLevel(orig_level)

    return captured_msgs


def test_tick_log_emitted_once_when_paused():
    """Exactly one [review-gate-paused] log line emitted regardless of target count."""
    msgs = _run_tick_all_capture_logs(paused=True, target_count=5)
    paused_lines = [m for m in msgs if "[review-gate-paused]" in m]
    assert len(paused_lines) == 1, (
        f"Expected exactly 1 [review-gate-paused] log line, got {len(paused_lines)}: {paused_lines}"
    )


def test_tick_log_silent_when_unpaused():
    """No [review-gate-paused] log line emitted when gate is healthy."""
    msgs = _run_tick_all_capture_logs(paused=False, target_count=3)
    paused_lines = [m for m in msgs if "[review-gate-paused]" in m]
    assert len(paused_lines) == 0, (
        f"Expected no [review-gate-paused] log lines when unpaused, got: {paused_lines}"
    )


# ---------------------------------------------------------------------------
# D3 — review_gate_resume(reason) behavior
# ---------------------------------------------------------------------------

def test_review_gate_resume_requires_reason():
    """Empty or whitespace-only reason raises ValueError."""
    mock_mem = MagicMock()
    mock_mem.get.return_value = {"content": "5"}

    with patch("lapis_pm.pm_core._mem", return_value=mock_mem):
        with pytest.raises(ValueError, match="non-empty reason"):
            pm_core.review_gate_resume("")
        with pytest.raises(ValueError, match="non-empty reason"):
            pm_core.review_gate_resume("   ")


def test_review_gate_resume_writes_audit_entry():
    """Successful resume writes a decision/review-gate-resume/<ts> mem key with correct content and tags."""
    mem_calls = {}

    mock_mem = MagicMock()
    mock_mem.get.return_value = {"content": "15"}

    def capture_set(key, value, tags=None):
        mem_calls[key] = {"value": value, "tags": tags}

    mock_mem.set.side_effect = capture_set

    with patch("lapis_pm.pm_core._mem", return_value=mock_mem):
        prev = pm_core.review_gate_resume("test reason for audit")

    assert prev == 15
    audit_keys = [k for k in mem_calls if k.startswith("decision/review-gate-resume/")]
    assert len(audit_keys) == 1, f"Expected one audit key, got: {list(mem_calls.keys())}"
    audit_key = audit_keys[0]
    entry = mem_calls[audit_key]
    assert "test reason for audit" in entry["value"]
    assert "15" in entry["value"]
    assert entry["tags"] == ["lapis-pm", "review-gate", "resume-audit"]


def test_review_gate_resume_preserves_existing_behavior():
    """Counter is reset, paused flag flipped, and REVIEW_GATE_PAUSE_BRIEF_KEY deleted."""
    set_calls = {}
    delete_calls = []

    mock_mem = MagicMock()
    mock_mem.get.return_value = {"content": "7"}

    def capture_set(key, value, tags=None):
        set_calls[key] = value

    def capture_delete(key):
        delete_calls.append(key)

    mock_mem.set.side_effect = capture_set
    mock_mem.delete.side_effect = capture_delete

    with patch("lapis_pm.pm_core._mem", return_value=mock_mem):
        prev = pm_core.review_gate_resume("preserve test")

    assert prev == 7
    # Counter reset
    assert set_calls.get(pm_core.REVIEW_GATE_COUNTER_KEY) == "0"
    # Paused flag flipped to False (stored as "0")
    assert set_calls.get(pm_core.REVIEW_GATE_PAUSED_KEY) == "0"
    # Pause-brief idempotency key deleted
    assert pm_core.REVIEW_GATE_PAUSE_BRIEF_KEY in delete_calls


# ---------------------------------------------------------------------------
# D3 CLI — --reason flag requirement
# ---------------------------------------------------------------------------

def test_cli_resume_rejects_missing_reason():
    """Bare `lapis-pm review-gate resume` (no --reason) exits with non-zero code."""
    rc, out, err = _run_review_gate(["resume"])
    assert rc != 0


def test_cli_resume_passes_reason_through():
    """--reason value is forwarded to pm_core.review_gate_resume."""
    called_with = {}

    def mock_resume(reason):
        called_with["reason"] = reason
        return 5

    with patch("lapis_pm.pm_core.review_gate_resume", side_effect=mock_resume):
        rc, out, err = _run_review_gate(["resume", "--reason", "smoke test"])

    assert rc == 0
    assert called_with.get("reason") == "smoke test"
    assert "smoke test" in out
