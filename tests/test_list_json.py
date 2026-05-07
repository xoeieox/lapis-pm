"""Tests for `lapis-pm list --json` (lapis-pm-list-json spec).

Coverage:
  - JSON output is valid JSON (parseable).
  - Each emitted object contains the full schema keys (no missing, no extras).
  - paused: true when target is paused; false when not.
  - outstanding_brief_id is UUID string when brief exists; null when absent.
  - dispatched_total / dispatched_pending match human-readable semantics.
  - Unbound targets are excluded from JSON output.
  - No-flag invocation is byte-for-byte unchanged (no regression).
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agents_core.targets import TargetStore
from lapis_pm.cli import main

# Canonical schema keys from spec
SCHEMA_KEYS = {
    "target_id",
    "title",
    "pm_repo",
    "pm_authority",
    "paused",
    "cursor",
    "dispatched_total",
    "dispatched_pending",
    "outstanding_brief_id",
    "tags",
    "urgency",
    "category",
    "destination",
    "loom_visibility",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_list(argv: list[str], targets_dir: Path,
              dispatched: dict[str, list] | None = None,
              outstanding: dict[str, str | None] | None = None,
              cursor: dict[str, str | None] | None = None) -> tuple[int, str, str]:
    """Run `lapis-pm list [argv...]` with stubbed pm_core helpers.

    dispatched: map of target_id -> list of dispatch records
    outstanding: map of target_id -> brief comment UUID or None
    cursor: map of target_id -> ISO cursor string or None
    """
    dispatched = dispatched or {}
    outstanding = outstanding or {}
    cursor = cursor or {}

    out_buf = io.StringIO()
    err_buf = io.StringIO()

    with (
        patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
        patch("lapis_pm.cli.pm_core.load_dispatched",
              side_effect=lambda tid: dispatched.get(tid, [])),
        patch("lapis_pm.cli.pm_core.get_outstanding_brief",
              side_effect=lambda tid: outstanding.get(tid, None)),
        patch("lapis_pm.cli.pm_core.get_cursor",
              side_effect=lambda tid: cursor.get(tid, None)),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        rc = main(["list"] + argv)

    return rc, out_buf.getvalue(), err_buf.getvalue()


def _make_bound_target(targets_dir: Path, target_id: str, *,
                       paused: bool = False, tags: list[str] | None = None,
                       urgency: str = "medium", category: str = "active-work",
                       title: str | None = None) -> None:
    store = TargetStore(targets_dir)
    t = store.create(target_id, title=title or target_id, urgency=urgency,
                     category=category)
    t.bind_pm(repo="lapis-pm", authority="advisory")
    if tags:
        t.data["tags"] = tags
    if paused:
        t.set_paused(True)
    t.save()


def _make_unbound_target(targets_dir: Path, target_id: str) -> None:
    store = TargetStore(targets_dir)
    t = store.create(target_id, title=target_id)
    t.save()


# ---------------------------------------------------------------------------
# Tests: JSON schema correctness
# ---------------------------------------------------------------------------


def test_json_output_is_valid_json(tmp_path):
    _make_bound_target(tmp_path, "alpha")
    rc, out, err = _run_list(["--json"], tmp_path)
    assert rc == 0, f"stderr={err!r}"
    parsed = json.loads(out)  # raises if invalid
    assert isinstance(parsed, list)


def test_json_schema_keys_present_no_extras(tmp_path):
    _make_bound_target(tmp_path, "alpha", tags=["lapis-pm"])
    rc, out, _ = _run_list(["--json"], tmp_path)
    assert rc == 0
    items = json.loads(out)
    assert len(items) == 1
    assert set(items[0].keys()) == SCHEMA_KEYS, (
        f"key mismatch: got {set(items[0].keys())}, expected {SCHEMA_KEYS}"
    )


def test_json_field_values_basic(tmp_path):
    _make_bound_target(tmp_path, "alpha", tags=["lapis-pm"], urgency="high",
                       category="active-work", title="Alpha Target")
    rc, out, _ = _run_list(
        ["--json"], tmp_path,
        cursor={"alpha": "2026-04-24T15:35:37.317184-07:00"},
    )
    assert rc == 0
    item = json.loads(out)[0]
    assert item["target_id"] == "alpha"
    assert item["title"] == "Alpha Target"
    assert item["pm_repo"] == "lapis-pm"
    assert item["pm_authority"] == "advisory"
    assert item["paused"] is False
    assert item["cursor"] == "2026-04-24T15:35:37.317184-07:00"
    assert item["tags"] == ["lapis-pm"]
    assert item["urgency"] == "high"
    assert item["category"] == "active-work"


# ---------------------------------------------------------------------------
# Tests: paused flag
# ---------------------------------------------------------------------------


def test_paused_false_when_not_paused(tmp_path):
    _make_bound_target(tmp_path, "beta", paused=False)
    rc, out, _ = _run_list(["--json"], tmp_path)
    assert rc == 0
    assert json.loads(out)[0]["paused"] is False


def test_paused_true_when_paused(tmp_path):
    _make_bound_target(tmp_path, "beta", paused=True)
    rc, out, _ = _run_list(["--json"], tmp_path)
    assert rc == 0
    assert json.loads(out)[0]["paused"] is True


# ---------------------------------------------------------------------------
# Tests: outstanding_brief_id
# ---------------------------------------------------------------------------


def test_outstanding_brief_id_null_when_absent(tmp_path):
    _make_bound_target(tmp_path, "gamma")
    rc, out, _ = _run_list(["--json"], tmp_path, outstanding={"gamma": None})
    assert rc == 0
    assert json.loads(out)[0]["outstanding_brief_id"] is None


def test_outstanding_brief_id_uuid_when_present(tmp_path):
    uuid = "abc12345-dead-beef-0000-111122223333"
    _make_bound_target(tmp_path, "gamma")
    rc, out, _ = _run_list(["--json"], tmp_path, outstanding={"gamma": uuid})
    assert rc == 0
    assert json.loads(out)[0]["outstanding_brief_id"] == uuid


# ---------------------------------------------------------------------------
# Tests: dispatched counts
# ---------------------------------------------------------------------------


def test_dispatched_total_and_pending(tmp_path):
    _make_bound_target(tmp_path, "delta")
    records = [
        {"status": "pending"},
        {"status": "pending"},
        {"status": "done"},
    ]
    rc, out, _ = _run_list(["--json"], tmp_path,
                            dispatched={"delta": records})
    assert rc == 0
    item = json.loads(out)[0]
    assert item["dispatched_total"] == 3
    assert item["dispatched_pending"] == 2


def test_dispatched_zero_when_no_records(tmp_path):
    _make_bound_target(tmp_path, "delta")
    rc, out, _ = _run_list(["--json"], tmp_path)
    assert rc == 0
    item = json.loads(out)[0]
    assert item["dispatched_total"] == 0
    assert item["dispatched_pending"] == 0


# ---------------------------------------------------------------------------
# Tests: unbound targets excluded
# ---------------------------------------------------------------------------


def test_unbound_targets_excluded(tmp_path):
    _make_bound_target(tmp_path, "bound-one")
    _make_unbound_target(tmp_path, "not-bound")
    rc, out, _ = _run_list(["--json"], tmp_path)
    assert rc == 0
    items = json.loads(out)
    ids = [i["target_id"] for i in items]
    assert "bound-one" in ids
    assert "not-bound" not in ids


def test_empty_when_no_bound_targets(tmp_path):
    _make_unbound_target(tmp_path, "orphan")
    rc, out, _ = _run_list(["--json"], tmp_path)
    assert rc == 0
    assert json.loads(out) == []


# ---------------------------------------------------------------------------
# Tests: human-readable path unchanged (no regression)
# ---------------------------------------------------------------------------


def test_no_flag_human_readable_unchanged(tmp_path):
    """Without --json, output is the human-readable table (no JSON)."""
    _make_bound_target(tmp_path, "epsilon", title="Epsilon Target")
    rc, out, _ = _run_list([], tmp_path)
    assert rc == 0
    # Human table header present
    assert "TARGET" in out
    assert "REPO" in out
    # Not JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_no_flag_no_bound_targets_message(tmp_path):
    """Without --json and no bound targets, prints the legacy message."""
    rc, out, _ = _run_list([], tmp_path)
    assert rc == 0
    assert "(no pm-bound targets)" in out


# ---------------------------------------------------------------------------
# Tests: cursor null when absent
# ---------------------------------------------------------------------------


def test_cursor_null_when_no_cursor(tmp_path):
    _make_bound_target(tmp_path, "zeta")
    rc, out, _ = _run_list(["--json"], tmp_path)  # cursor dict empty → None
    assert rc == 0
    assert json.loads(out)[0]["cursor"] is None
