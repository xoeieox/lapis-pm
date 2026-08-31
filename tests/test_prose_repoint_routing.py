"""Routing-integrity tests for lapis-pm-prose-synthesis-local-repoint-v0.

One group per deliverable D2-D6 (D1 lives in tests/test_brief.py): the local
client (agents_core.gw_agent.call_gw_agent) is invoked, call_claude_cli is NOT,
and each site's return/fallback contract is preserved. The last class is the
full-surface regression guard (spec DoD).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import lapis_pm
from lapis_pm import authority, facets_gw_eval, land, state_brief, trajectory
from lapis_pm.facets_gw_eval import EvalPass, ArmMetrics, run_quality_cross_judge

CLEAN_VERDICT = json.dumps({"verdict": "clean", "issues": [], "confidence": 0.9})

_WEEKLY_BUCKETS = {
    "Built": [],
    "Notable ratifications": [],
    "In flight": [],
    "Captured — not yet built": [],
    "Awaiting your call": [],
    "Gardener Cross-Cutting Observations": ["obs item"],
}


def _claude_guard():
    return patch(
        "agents_core.llm.call_claude_cli",
        side_effect=AssertionError("call_claude_cli must not be called"),
    )


class TestAuthorityScreenRouting:
    def test_screen_routes_to_local_seat(self):
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value=CLEAN_VERDICT) as mock_gw,
            _claude_guard(),
        ):
            result = authority.screen("lapis-pm", 301, "spec summary", "diff text")

        assert isinstance(result, dict)
        assert result["verdict"] == "clean"
        assert mock_gw.call_count == 1
        assert mock_gw.call_args.kwargs["json_mode"] is True

    def test_screen_needs_human_fallback_on_no_text(self):
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value=None),
            _claude_guard(),
        ):
            result = authority.screen("lapis-pm", 301, "spec summary", "diff text")

        assert result["verdict"] == "needs-human"


class TestLandRouting:
    def _ctx(self):
        return (
            patch("lapis_pm.land.episodic.spec", return_value="stub spec"),
            patch("lapis_pm.land.episodic.all_comments", return_value=[]),
            _claude_guard(),
        )

    def test_land_records_actual_seat_in_provenance(self):
        body = (
            "# t - Arc Doc\n\n"
            "## Origin\norigin text\n\n"
            "## Landing summary\nsummary\n"
        )

        def fake_gw(prompt, system, **kwargs):
            kwargs["served_model_out"].append("test-seat")
            return body

        p = self._ctx()
        with (
            patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw),
            p[0], p[1], p[2],
        ):
            doc = land.generate_arc_doc("routing-test-tid")

        assert "model: test-seat" in doc.body

    def test_land_local_seat_unavailable_marker(self):
        body = (
            "# t - Arc Doc\n\n"
            "## Origin\norigin text\n\n"
            "## Landing summary\nsummary\n"
        )

        p = self._ctx()
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value=body),
            p[0], p[1], p[2],
        ):
            doc = land.generate_arc_doc("routing-test-tid")

        assert "model: local-seat-unavailable" in doc.body

    def test_land_no_text_fallback(self):
        p = self._ctx()
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value=None),
            p[0], p[1], p[2],
        ):
            doc = land.generate_arc_doc("routing-test-tid")

        assert "local seat returned no text" in doc.body
        assert "model: local-seat-unavailable" in doc.body


class TestStateBriefWeeklyRouting:
    def test_weekly_routes_to_local_seat(self):
        buckets = _WEEKLY_BUCKETS
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value="weekly prose text") as mock_gw,
            _claude_guard(),
        ):
            prose = state_brief._generate_prose(
                period="weekly", buckets=buckets, start_label="2026-08-17 08:00 PT",
            )

        assert "weekly prose text" in prose
        assert mock_gw.call_count == 1


class TestTrajectoryRouting:
    def test_call_haiku_routes_to_local_seat(self):
        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value="rollup one-liner") as mock_gw,
            _claude_guard(),
        ):
            out = trajectory._call_haiku("prompt", "system")

        assert out == "rollup one-liner"
        assert mock_gw.call_count == 1


class TestFacetsRouting:
    def test_cross_judge_routes_to_local_seat(self):
        clean = EvalPass(
            pass_name="clean",
            pass_at="2026-06-17T00:00:00Z",
            arm_a=ArmMetrics("haiku", run_count=1, elapsed_s_per_run=[10.0]),
            arm_b=ArmMetrics("gravitywell", run_count=1, elapsed_s_per_run=[11.0]),
        )
        verdict = json.dumps({"judgment": "equivalent", "reasoning": "ok"})
        with (
            patch("lapis_pm.facets_gw_eval.call_gw_agent", return_value=verdict) as mock_gw,
            _claude_guard(),
        ):
            result = run_quality_cross_judge(clean, [Path("/tmp/fixture0.md")])

        assert result == 100.0
        assert mock_gw.call_count == 1

    def test_module_exposes_no_claude_symbol(self):
        assert not hasattr(facets_gw_eval, "call_claude_cli")


class TestNoClaudeCliInProseLayer:
    """Full-surface regression guard (spec DoD).

    Scans every non-test .py file under lapis_pm/ for a call_claude_cli(
    call and asserts zero, with an explicit exclusion list of exactly one
    entry (the sanctioned backcaster routing-plumbing file).
    """

    EXCLUSIONS = {"lapis_pm/backcaster/llm_routing.py"}

    def test_no_call_claude_cli_in_non_test_surface(self):
        package_root = Path(lapis_pm.__file__).parent
        offenders = []
        for path in sorted(package_root.rglob("*.py")):
            rel = path.relative_to(package_root.parent).as_posix()
            if rel in self.EXCLUSIONS:
                continue
            if "call_claude_cli(" in path.read_text():
                offenders.append(rel)
        assert not offenders, (
            "call_claude_cli( present in prose layer: " + repr(offenders)
        )
