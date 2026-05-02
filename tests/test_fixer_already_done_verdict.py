"""Tests for the fixer already-done verdict protocol (lapis-pm-fixer-already-done-verdict).

Coverage:
  (a) verdict file present + valid merged PR → daemon auto-lands the target
  (b) verdict file absent → existing behavior unchanged (confabulation check runs)
  (c) malformed verdict file → ignored with WARN log, confabulation check runs
  (d) satisfied_by_pr references non-existent/unmerged PR → invalid verdict → brief
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_comment(tags: list[str], content: str = "", ts: str = "2026-04-30T00:00:00") -> MagicMock:
    c = MagicMock()
    c.tags = tags
    c.content = content
    c.ts = ts
    return c


def _make_mem(landed: bool = False, dispatched: list | None = None):
    """Return a mock MemoryStore."""
    mem = MagicMock()

    def _get(key):
        if "landed" in key and landed:
            return {"content": "{}"}
        if "dispatched" in key:
            payload = dispatched if dispatched is not None else []
            return {"content": json.dumps(payload)}
        return None

    mem.get.side_effect = _get
    return mem


def _dispatch_record(
    agent_type: str = "fixer",
    spec_id: str = "spec-abc",
    status: str = "pending",
    repo: str = "lapis-test",
    gpu_id: str = "gpu-test-1",
) -> dict:
    return {
        "gpu_id": gpu_id,
        "spec_id": spec_id,
        "agent_type": agent_type,
        "intent": "implement the thing",
        "repo": repo,
        "ts": "2026-04-30T00:00:00",
        "status": status,
        "retry_count": 0,
    }


# ---------------------------------------------------------------------------
# (a) Verdict present + valid PR → auto-land fires
# ---------------------------------------------------------------------------

class TestVerdictPresentAutoLand:
    """Verdict file present, PR is merged → decide phase triggers auto-land."""

    def test_already_satisfied_pending_returns_pr_info(self):
        """_already_satisfied_pending finds the observation and returns pr_num."""
        comment = _make_comment(
            tags=["pm:observation", "pm:already-satisfied", "pm:already-satisfied:pr=42"],
            content="Fixer verdict: already_satisfied by PR #42\nEvidence: files match.",
            ts="2026-04-30T10:00:00",
        )
        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[comment]),
            patch("lapis_pm.pm_core._mem", return_value=_make_mem(landed=False)),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        ):
            result = pm_core._already_satisfied_pending("mytid")

        assert result is not None
        pr_num, evidence, ts = result
        assert pr_num == 42
        assert "files match" in evidence
        assert ts == "2026-04-30T10:00:00"

    def test_already_satisfied_pending_none_when_landed(self):
        """_already_satisfied_pending returns None when pm/landed already set."""
        comment = _make_comment(
            tags=["pm:already-satisfied", "pm:already-satisfied:pr=42"],
        )
        mem = _make_mem(landed=True)
        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[comment]),
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
        ):
            assert pm_core._already_satisfied_pending("mytid") is None

    def test_already_satisfied_pending_none_when_dispatch_pending(self):
        """_already_satisfied_pending returns None when a dispatch is still pending."""
        comment = _make_comment(
            tags=["pm:already-satisfied", "pm:already-satisfied:pr=42"],
        )
        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[comment]),
            patch("lapis_pm.pm_core._mem", return_value=_make_mem()),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[
                {"status": "pending", "gpu_id": "x"}
            ]),
        ):
            assert pm_core._already_satisfied_pending("mytid") is None

    def test_act_auto_land_already_satisfied_full_flow(self):
        """_act_auto_land_already_satisfied writes arc doc, lands, unbinds."""
        comment = _make_comment(
            tags=["pm:already-satisfied", "pm:already-satisfied:pr=42"],
            content="Fixer verdict: already_satisfied by PR #42\nEvidence: confirmed.",
            ts="2026-04-30T10:00:00",
        )
        mock_arc = MagicMock()
        mock_arc.path = Path("/srv/lapis/lapis-state/mytid.md")

        mock_target = MagicMock()
        mock_target.data = {}

        mock_store = MagicMock()
        mock_store.get.return_value = mock_target

        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[comment]),
            patch("lapis_pm.pm_core._mem", return_value=_make_mem()),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("lapis_pm.pm_core.TargetStore", return_value=mock_store),
            patch("lapis_pm.pm_core.episodic.write"),
            patch("lapis_pm.pm_core.episodic.write_observation"),
            patch("lapis_pm.pm_core.clear_landed_state"),
            patch("lapis_pm.land.generate_arc_doc", return_value=mock_arc) as mock_gen,
            patch("lapis_pm.land.write_arc_doc", return_value=mock_arc.path),
        ):
            result = pm_core._act_auto_land_already_satisfied("mytid")

        assert result.startswith("auto_land:already_satisfied:pr=42:")
        # Arc doc called with extra_origin_note containing pr#
        mock_gen.assert_called_once()
        note = mock_gen.call_args[1]["extra_origin_note"]
        assert "PR #42" in note
        assert "landed without new dispatch" in note
        # Target archived and unbound
        mock_store.archive.assert_called_once_with("mytid")
        mock_target.unbind_pm.assert_called_once()


# ---------------------------------------------------------------------------
# (b) Verdict absent → existing behavior unchanged
# ---------------------------------------------------------------------------

class TestVerdictAbsentNoChange:
    """No verdict sidecar → confabulation path runs normally."""

    def test_handle_already_satisfied_verdict_skipped_when_no_sidecar(self, tmp_path):
        """If verdict sidecar doesn't exist, _read_fixer_verdict returns None."""
        # Point SHAPED_DIR at tmp_path (no sidecar written)
        with patch("lapis_pm.pm_core.SHAPED_DIR", tmp_path):
            result = pm_core._read_fixer_verdict("spec-missing")
        assert result is None

    def test_verdict_sidecar_absent_encode_falls_through(self, tmp_path):
        """In _encode_gpu_results, absent verdict → confabulation check runs."""
        rec = _dispatch_record(spec_id="spec-absent")
        output_file = tmp_path / f"{rec['gpu_id']}-output.md"
        output_file.write_text("Implemented the thing and opened PR #5 at http://1.2.3.4:3000/x/y/pulls/5")

        with (
            patch("lapis_pm.pm_core.SHAPED_DIR", tmp_path),
            patch("lapis_pm.pm_core.COMPLETED_DIR", tmp_path),
            patch("lapis_pm.pm_core.FAILED_DIR", tmp_path / "failed"),
            patch("lapis_pm.pm_core.CLAUDE_QUEUE_COMPLETED_DIR", tmp_path / "cq-comp"),
            patch("lapis_pm.pm_core.CLAUDE_QUEUE_FAILED_DIR", tmp_path / "cq-fail"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core.episodic.write_result"),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]),
        ):
            encoded, failed = pm_core._encode_gpu_results("mytid")

        # Result encoded normally (no verdict sidecar), not failed
        assert encoded == 1
        assert failed == []
        assert rec["status"] == "processed"


# ---------------------------------------------------------------------------
# (c) Malformed verdict → WARN logged, falls through
# ---------------------------------------------------------------------------

class TestMalformedVerdict:
    """Malformed verdict sidecar → WARN + ignored, no auto-land."""

    def test_read_verdict_warns_on_bad_json(self, tmp_path, capsys):
        """_read_fixer_verdict logs WARN and returns None for bad JSON."""
        sidecar = tmp_path / "spec-bad-verdict.json"
        sidecar.write_text("this is not json {{{{")
        with patch("lapis_pm.pm_core.SHAPED_DIR", tmp_path):
            result = pm_core._read_fixer_verdict("spec-bad")
        assert result is None
        captured = capsys.readouterr()
        assert "WARN" in captured.err

    def test_read_verdict_warns_on_non_dict(self, tmp_path, capsys):
        """_read_fixer_verdict logs WARN and returns None when JSON is not a dict."""
        sidecar = tmp_path / "spec-list-verdict.json"
        sidecar.write_text("[1, 2, 3]")
        with patch("lapis_pm.pm_core.SHAPED_DIR", tmp_path):
            result = pm_core._read_fixer_verdict("spec-list")
        assert result is None
        captured = capsys.readouterr()
        assert "WARN" in captured.err

    def test_handle_verdict_warns_on_unrecognised_type(self, capsys):
        """_handle_already_satisfied_verdict logs WARN for unknown verdict type."""
        rec = _dispatch_record()
        verdict = {"verdict": "needs_human_judgment"}
        result = pm_core._handle_already_satisfied_verdict("mytid", rec, verdict)
        assert result is False
        assert "WARN" in capsys.readouterr().err

    def test_handle_verdict_warns_on_missing_pr_num(self, capsys):
        """_handle_already_satisfied_verdict logs WARN when satisfied_by_pr missing."""
        rec = _dispatch_record()
        verdict = {"verdict": "already_satisfied", "evidence": "files match"}
        result = pm_core._handle_already_satisfied_verdict("mytid", rec, verdict)
        assert result is False
        assert "WARN" in capsys.readouterr().err

    def test_handle_verdict_warns_on_negative_pr_num(self, capsys):
        """_handle_already_satisfied_verdict logs WARN when satisfied_by_pr <= 0."""
        rec = _dispatch_record()
        verdict = {"verdict": "already_satisfied", "satisfied_by_pr": -1, "evidence": "x"}
        result = pm_core._handle_already_satisfied_verdict("mytid", rec, verdict)
        assert result is False
        assert "WARN" in capsys.readouterr().err

    def test_malformed_verdict_does_not_prevent_normal_encode(self, tmp_path):
        """Malformed sidecar (bad JSON) → _encode_gpu_results falls through to confabulation."""
        rec = _dispatch_record(spec_id="spec-malformed")
        output_file = tmp_path / f"{rec['gpu_id']}-output.md"
        output_file.write_text(
            "Implemented the thing and opened PR #7 at http://1.2.3.4:3000/x/y/pulls/7"
        )
        sidecar = tmp_path / "spec-malformed-verdict.json"
        sidecar.write_text("{bad json")

        with (
            patch("lapis_pm.pm_core.SHAPED_DIR", tmp_path),
            patch("lapis_pm.pm_core.COMPLETED_DIR", tmp_path),
            patch("lapis_pm.pm_core.FAILED_DIR", tmp_path / "failed"),
            patch("lapis_pm.pm_core.CLAUDE_QUEUE_COMPLETED_DIR", tmp_path / "cq-comp"),
            patch("lapis_pm.pm_core.CLAUDE_QUEUE_FAILED_DIR", tmp_path / "cq-fail"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[rec]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core.episodic.write_result"),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]),
        ):
            encoded, failed = pm_core._encode_gpu_results("mytid")

        # Sidecar was malformed → verdict ignored, normal encode path ran
        assert encoded == 1
        assert rec["status"] == "processed"
        # Sidecar consumed (file should no longer exist)
        assert not sidecar.exists()


# ---------------------------------------------------------------------------
# (d) satisfied_by_pr references non-existent/unmerged PR → brief not auto-land
# ---------------------------------------------------------------------------

class TestInvalidPrFallback:
    """PR validation fails → invalid observation written → brief synthesized."""

    def test_validate_pr_returns_false_on_not_merged(self):
        """_validate_already_satisfied_pr returns False when PR not merged."""
        mock_pr = {"merged": False, "state": "open"}
        with patch("lapis_pm.pm_core._forgejo_get_pr", return_value=mock_pr):
            assert pm_core._validate_already_satisfied_pr("repo", 99) is False

    def test_validate_pr_returns_false_on_exception(self):
        """_validate_already_satisfied_pr returns False when Forgejo raises."""
        with patch("lapis_pm.pm_core._forgejo_get_pr", side_effect=Exception("404")):
            assert pm_core._validate_already_satisfied_pr("repo", 99) is False

    def test_validate_pr_returns_true_when_merged(self):
        """_validate_already_satisfied_pr returns True for a merged+closed PR."""
        mock_pr = {"merged": True, "state": "closed"}
        with patch("lapis_pm.pm_core._forgejo_get_pr", return_value=mock_pr):
            assert pm_core._validate_already_satisfied_pr("lapis-pm", 42) is True

    def test_handle_verdict_invalid_pr_writes_invalid_tag(self, capsys):
        """_handle_already_satisfied_verdict writes pm:already-satisfied:invalid when PR not merged."""
        rec = _dispatch_record(repo="lapis-test")
        verdict = {
            "verdict": "already_satisfied",
            "satisfied_by_pr": 99,
            "evidence": "I think it was merged",
        }
        written_obs = {}

        def _write_obs(tid, content, extra_tags=None):
            written_obs["content"] = content
            written_obs["tags"] = extra_tags or []

        with (
            patch("lapis_pm.pm_core._validate_already_satisfied_pr", return_value=False),
            patch("lapis_pm.pm_core.episodic.write_observation", side_effect=_write_obs),
        ):
            result = pm_core._handle_already_satisfied_verdict("mytid", rec, verdict)

        assert result is True  # Handled (don't retry the fixer)
        assert "pm:already-satisfied:invalid" in written_obs["tags"]
        assert "pm:already-satisfied:invalid:pr=99" in written_obs["tags"]
        assert "WARN" in capsys.readouterr().err

    def test_already_satisfied_invalid_pending_found(self):
        """_already_satisfied_invalid_pending returns (pr_num, content) when unacted."""
        comment = _make_comment(
            tags=["pm:observation", "pm:already-satisfied:invalid",
                  "pm:already-satisfied:invalid:pr=99"],
            content="PR #99 validation failed",
        )
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[comment]):
            result = pm_core._already_satisfied_invalid_pending("mytid")

        assert result is not None
        pr_num, content = result
        assert pr_num == 99

    def test_already_satisfied_invalid_pending_deduped_after_brief(self):
        """_already_satisfied_invalid_pending returns None if brief already posted."""
        invalid_comment = _make_comment(
            tags=["pm:already-satisfied:invalid", "pm:already-satisfied:invalid:pr=99"],
        )
        briefed_comment = _make_comment(
            tags=["pm:already-satisfied:invalid-briefed"],
        )
        with patch(
            "lapis_pm.pm_core.episodic.all_comments",
            return_value=[invalid_comment, briefed_comment],
        ):
            assert pm_core._already_satisfied_invalid_pending("mytid") is None

    def test_act_brief_already_satisfied_invalid_posts_brief(self):
        """_act_brief_already_satisfied_invalid synthesizes a brief for the human."""
        invalid_comment = _make_comment(
            tags=["pm:already-satisfied:invalid", "pm:already-satisfied:invalid:pr=99"],
            content="PR #99 validation failed",
        )
        mock_brief = MagicMock()
        mock_brief.comment_id = "brief-xyz"
        mock_brief.pushed = True

        with (
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[invalid_comment]),
            patch("lapis_pm.pm_core.brief.synthesize", return_value=mock_brief),
            patch("lapis_pm.pm_core.set_outstanding_brief"),
            patch("lapis_pm.pm_core.episodic.write_observation"),
        ):
            result = pm_core._act_brief_already_satisfied_invalid("mytid")

        assert result.startswith("already_satisfied_invalid_brief:")
