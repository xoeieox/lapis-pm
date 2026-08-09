"""Tests for lapis-pm-reviewer-peak-contractor-route-v0 D2: dispatch-time-only
TOU-peak routing of reviewer_fresh -> reviewer_fresh_contractor.

Invariants under test:
  - Inside peak, a fresh-mode reviewer dispatch selects reviewer_fresh_contractor.
  - Outside peak, a fresh-mode reviewer dispatch selects reviewer_fresh.
  - same-mode ("reviewer") dispatch is never redirected, peak or not.
  - The dispatch record carries the model actually used (auditability, D2).
  - No exception handler anywhere may switch seats — grepped structurally
    below, since the invariant is about the SHAPE of the code, not a single
    runtime behavior.
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

PEAK_TIME = datetime(2026, 8, 4, 18, 0, tzinfo=PACIFIC)     # 18:00 PT — inside 16:00-21:00
OFFPEAK_TIME = datetime(2026, 8, 4, 10, 0, tzinfo=PACIFIC)  # 10:00 PT — outside


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
        patch("lapis_pm.pm_core.append_dispatched", side_effect=fake_append),
        patch("lapis_pm.pm_core._increment_review_gate_counter", return_value=1),
        patch("lapis_pm.pm_core._increment_reviewer_attempt"),
        patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        patch("agents_core.forgejo.get_pr_diff", return_value="diff text"),
        patch("lapis_pm.tou_window.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = now
        pm_core._act_dispatch_reviewer("tid", PR, CLS, mode=mode, cycle=1)

    return captured_agent_type.get("value"), dispatched_records


class TestPeakRouting:

    def test_fresh_mode_in_peak_selects_contractor(self):
        agent_type, _ = _call_reviewer("fresh", PEAK_TIME)
        assert agent_type == "reviewer_fresh_contractor"

    def test_fresh_mode_offpeak_selects_reviewer_fresh(self):
        agent_type, _ = _call_reviewer("fresh", OFFPEAK_TIME)
        assert agent_type == "reviewer_fresh"

    def test_same_mode_never_redirected_in_peak(self):
        agent_type, _ = _call_reviewer("same", PEAK_TIME)
        assert agent_type == "reviewer"

    def test_same_mode_never_redirected_offpeak(self):
        agent_type, _ = _call_reviewer("same", OFFPEAK_TIME)
        assert agent_type == "reviewer"

    def test_dispatch_record_carries_model_in_peak(self):
        _, records = _call_reviewer("fresh", PEAK_TIME)
        assert len(records) == 1
        assert records[0]["agent_type"] == "reviewer_fresh_contractor"
        assert records[0]["model"] == "deepseek/deepseek-v4-flash"
        assert records[0]["route"] == "peak-window-policy"

    def test_dispatch_record_carries_model_offpeak(self):
        _, records = _call_reviewer("fresh", OFFPEAK_TIME)
        assert len(records) == 1
        assert records[0]["agent_type"] == "reviewer_fresh"
        assert records[0]["model"] == "gravitywell-slot1"
        assert "route" not in records[0]


class TestNoExceptionHandlerSwitchesSeats:
    """Structural invariant (D2): the routing decision belongs at dispatch
    time only. Assert by source inspection that no `except` block anywhere
    in pm_core.py assigns agent_type to "reviewer_fresh_contractor" — the
    only assignment site must be the dispatch-time clock check in
    _act_dispatch_reviewer.
    """

    def test_contractor_assignment_appears_at_known_dispatch_time_sites(self):
        """Two legitimate assignment sites are expected: the TOU-peak check
        (D2) and the dead-seat gap-spanner check
        (lapis-pm-reviewer-seat-phala-test-key) — both dispatch-time
        routing inputs in _act_dispatch_reviewer, neither exception-driven.
        A third site would indicate an undocumented new routing input."""
        src = Path(pm_core.__file__).read_text()
        assignment_lines = [
            i for i, line in enumerate(src.splitlines())
            if re.search(r'agent_type\s*=\s*"reviewer_fresh_contractor"', line)
        ]
        assert len(assignment_lines) == 2, (
            f"expected exactly two assignments of agent_type to "
            f"reviewer_fresh_contractor (TOU-peak + gap-spanner), "
            f"found {len(assignment_lines)}"
        )

    def test_no_except_block_mentions_contractor_seat(self):
        """No `except` clause body (up to the next top-level statement) may
        reference reviewer_fresh_contractor — that would be exactly the
        "call the contractor on failure" pattern D2 forbids."""
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
