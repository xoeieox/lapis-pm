"""tests/test_hold_shadow_production_isolation.py — production-log isolation,
byte-proven (spec Scope item 8 / Definition of Done #7, PM review finding on
PR #285, 2026-08-12 ~12:40, reproduced live).

hold_shadow.py's two hooks live inside run_spec_review's single return point
and pm_core._act_brief's hold branch — call sites the rest of this suite
drives constantly with no awareness of hold_shadow. Before the conftest.py
session-wide ROOM_ROOT pin, every one of those pre-existing tests silently
appended real records into the production /srv/lapis/hold-shadow/*.jsonl files:
98 fictional gate-outcome records were already present before merge, and a
single 26-test re-run of tests/test_spec_review_gate_defaults.py appended six
more. This file proves the fix rather than assuming it:

  1. Under the suite's ambient environment (no per-test ROOM_ROOT override —
     the whole point is to test what the session-wide pin alone gives you),
     every hold-shadow path resolves outside /room.
  2. The fault writer shares that same resolution (spec Scope item 8(c): it
     currently shares _room_root() with the record writers — verify, don't
     assume) by reproducing the exact PR #285 failure mode (a MagicMock
     landing where a string is expected) and checking both that the fault
     line lands under the pinned tmp dir and that production
     /srv/lapis/hold-shadow/faults.jsonl gains zero new lines.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from lapis_pm import hold_shadow

_PRODUCTION_ROOM = Path("/room")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.read_text(encoding="utf-8").splitlines())


def test_hold_shadow_paths_resolve_outside_room_under_suite_ambient_env():
    """No monkeypatch here — that's the point. If conftest's session-wide
    ROOM_ROOT pin ever regresses, this is the test that catches it, because
    it deliberately relies on nothing but the ambient test environment
    every other test in the suite also runs under."""
    for path in (
        hold_shadow.hold_shadow_dir(),
        hold_shadow.gate_outcomes_path(),
        hold_shadow.hold_facts_path(),
        hold_shadow.faults_path(),
    ):
        resolved = path.resolve()
        assert resolved != _PRODUCTION_ROOM
        assert _PRODUCTION_ROOM not in resolved.parents, (
            f"{path} resolves under production /room — the suite would write "
            "into the exact artifact Thursday's ruling reads."
        )


def test_fault_writer_honours_the_same_room_root_override_as_the_record_writers():
    """Spec Scope item 8(c). Reproduce the PR #285 failure shape — a
    MagicMock reaching a code path that expects a string — and confirm the
    resulting fault line is written under the pinned tmp dir, never
    production /room, and that production faults.jsonl gains no new lines
    from this test's action."""
    production_faults = _PRODUCTION_ROOM / "hold-shadow" / "faults.jsonl"
    before = _line_count(production_faults)

    hold_shadow.observe_hold_fact(
        target_id="tid-fault-writer-test",
        pr_number=1,
        repo="lapis-pm",
        hold_reasons=[MagicMock()],  # not a str — the PR #285 bug shape
        hold_comment_id="h-fault",
        brief_comment_id="b-fault",
        pm_authority="hold",
        spec_bound_ts="ts",
    )

    fault_path = hold_shadow.faults_path()
    resolved = fault_path.resolve()
    assert resolved != production_faults
    assert _PRODUCTION_ROOM not in resolved.parents

    assert fault_path.exists(), "observe_hold_fact must never-raise but still log the fault"
    last_fault = json.loads(fault_path.read_text(encoding="utf-8").splitlines()[-1])
    assert last_fault["schema_version"] == hold_shadow.FAULT_SCHEMA
    assert last_fault["hook"] == "hold-fact"

    after = _line_count(production_faults)
    assert after == before, (
        "the fault writer wrote into production /srv/lapis/hold-shadow/faults.jsonl "
        f"({before} lines before, {after} after)"
    )
