"""Tests for lapis-pm-phala-test-key D2/D5.1: the TOU-peak
contractor route is removed. A fresh-mode reviewer dispatch now selects the
local `reviewer_fresh` seat in every window — the clock no longer changes
the outcome.

Formerly (lapis-pm-reviewer-peak-contractor-route-v0) this file asserted the
opposite: that a dispatch inside the 16:00-21:00 PT window redirected to
`reviewer_fresh_contractor`. That route, and the seat backing it, are gone
(Erah, 2026-08-19: "Drop Phala as contractor, the rotating setup breaks
whatever we have often enough that it shouldn't be relied upon for
Machinery."). This file is rewritten, not deleted, per spec D5.6 — it still
covers the invariant that mattered (same-mode is never redirected), now
updated to also cover the fresh-mode invariant post-removal.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from lapis_pm import authority, pm_core
from lapis_pm.tou_window import PACIFIC

PR = {
    "number": 42,
    "title": "feat: add widgets",
    "html_url": "https://forgejo/Erah/myrepo/pulls/42",
    "head": {"ref": "lapis/my-target/widgets"},
    "base": {"ref": "main"},
    "mergeable": True,
}

CLS = authority.PRClassification(
    verdict="advisory",
    screen_verdict="unknown",
    static_outcome=authority.StaticOutcome.static_pass,
    reasons=["static checks passed"],
    issues=[],
    pr_number=42,
    repo="myrepo",
    title="feat: add widgets",
    html_url="https://forgejo/Erah/myrepo/pulls/42",
    changed_paths=["src/foo.py"],
    diff_loc=30,
    diff="",
)

PEAK_TIME = datetime(2026, 8, 4, 18, 0, tzinfo=PACIFIC)     # 18:00 PT — formerly 16:00-21:00 peak
OFFPEAK_TIME = datetime(2026, 8, 4, 10, 0, tzinfo=PACIFIC)  # 10:00 PT — formerly outside


def _make_dispatch_result():
    result = MagicMock()
    result.task_id = "task-abc"
    result.spec_id = "spec-abc"
    return result


def _call_reviewer(mode, now, records=None):
    captured_agent_type = {}

    def capture_dispatch(agent_type, target_id, user_prompt, vars_=None, **kw):
        captured_agent_type["value"] = agent_type
        return _make_dispatch_result()

    dispatched_records = records if records is not None else []

    def fake_append(target_id, record):
        dispatched_records.append(record)

    with (
        patch.object(pm_core._SHAPER, "dispatch", side_effect=capture_dispatch),
        patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd", return_value="/srv/git/myrepo-working"),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
        patch("lapis_pm.pm_core.episodic.write_dispatch"),
            patch("lapis_pm.pm_core.append_dispatched", side_effect=fake_append),        patch("lapis_pm.pm_core._increment_reviewer_attempt"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("agents_core.forgejo.get_pr_diff", return_value="diff text"),
            patch("lapis_pm.tou_window.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = now
        pm_core._act_dispatch_reviewer("tid", PR, CLS, mode=mode, cycle=1)

    return captured_agent_type.get("value"), dispatched_records


class TestNoMoreContractorRoute:
    """D5.1: the clock no longer decides the seat. Fresh mode always
    dispatches to the local reviewer_fresh seat, peak window or not."""

    def test_fresh_mode_in_former_peak_window_selects_reviewer_fresh(self):
        agent_type, records = _call_reviewer("fresh", PEAK_TIME)
        assert agent_type == "reviewer_fresh"
        assert records[0]["agent_type"] == "reviewer_fresh"

    def test_fresh_mode_offpeak_selects_reviewer_fresh(self):
        agent_type, records = _call_reviewer("fresh", OFFPEAK_TIME)
        assert agent_type == "reviewer_fresh"
        assert records[0]["agent_type"] == "reviewer_fresh"

    def test_same_mode_never_redirected_in_former_peak_window(self):
        agent_type, _ = _call_reviewer("same", PEAK_TIME)
        assert agent_type == "reviewer"

    def test_same_mode_never_redirected_offpeak(self):
        agent_type, _ = _call_reviewer("same", OFFPEAK_TIME)
        assert agent_type == "reviewer"

    def test_no_dispatch_record_ever_carries_contractor_agent_type(self):
        _, records_peak = _call_reviewer("fresh", PEAK_TIME)
        _, records_offpeak = _call_reviewer("fresh", OFFPEAK_TIME)
        for records in (records_peak, records_offpeak):
            assert len(records) == 1
            assert records[0]["agent_type"] != "reviewer_fresh_contractor"
            assert "route" not in records[0]

    def test_dispatch_record_carries_local_seat_model(self):
        _, records = _call_reviewer("fresh", PEAK_TIME)
        assert records[0]["model"] == "gravitywell-slot1"


class TestNoLiveContractorAssignmentSite:
    """Structural invariant: no code path in pm_core.py assigns agent_type
    to "reviewer_fresh_contractor" anymore — the only surviving mentions of
    the name are the legacy compat tuple/family mapping and their comments
    (D4)."""

    def test_no_assignment_of_agent_type_to_contractor(self):
        src = Path(pm_core.__file__).read_text()
        assignment_lines = [
            i for i, line in enumerate(src.splitlines())
            if re.search(r'agent_type\s*=\s*"reviewer_fresh_contractor"', line)
        ]
        assert assignment_lines == [], (
            f"expected zero live assignments of agent_type to "
            f"reviewer_fresh_contractor post-removal, found at lines "
            f"{[i + 1 for i in assignment_lines]}"
        )

    def test_no_except_block_mentions_contractor_seat(self):
        """No `except` clause body (up to the next top-level statement) may
        reference reviewer_fresh_contractor."""
        src = Path(pm_core.__file__).read_text()
        lines = src.splitlines()
        in_except = False
        except_indent = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("except") and stripped.endswith(":"):
                in_except = True
                except_indent = len(line) - len(line.lstrip())
                continue
            if in_except:
                if line.strip() == "":
                    continue
                indent = len(line) - len(line.lstrip())
                if indent <= except_indent:
                    in_except = False
                    continue
                assert "reviewer_fresh_contractor" not in line, (
                    f"line {i + 1} inside an except block references the "
                    f"contractor seat: {line!r}"
                )
