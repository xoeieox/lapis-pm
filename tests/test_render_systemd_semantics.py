"""Tests for systemd timer/service semantics rendering in held-path holds.

Coverage:
  - OnCalendar=*-*-* *:*:00/5 (every 5 seconds) yields cadence line with ~5s
  - OnCalendar=*:00/5:00 (every 5 minutes) yields cadence line with ~5min
  - Non-systemd held diff yields no extra lines (unchanged behavior)
  - Renderer failure path (missing binary, timeout, parse error) returns
    classification + unavailable note, does not raise
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import subprocess

from lapis_pm import authority


class TestRenderSystemdSemantics:
    """Test the render_systemd_semantics helper."""

    def test_no_systemd_files_yields_empty(self):
        """Non-systemd held paths yield no rendered lines."""
        diff = "+++ b/infra/deploy.sh\n+echo hi"
        held_hits = ["infra/deploy.sh"]
        result = authority.render_systemd_semantics(diff, held_hits)
        assert result == []

    def test_no_oncalendar_in_diff_yields_empty(self):
        """Systemd files without OnCalendar changes yield no rendered lines."""
        diff = "+++ b/systemd/app.timer\n+[Unit]\n+Description=app"
        held_hits = ["systemd/app.timer"]
        result = authority.render_systemd_semantics(diff, held_hits)
        assert result == []

    def test_oncalendar_fast_cadence(self):
        """OnCalendar=*-*-* *:*:00/5 should yield a ~5s cadence line."""
        diff = "+++ b/systemd/app.timer\n+OnCalendar=*-*-* *:*:00/5"
        held_hits = ["systemd/app.timer"]

        mock_output = """\
  Original form: *-*-* *:*:00/5
  Normalized form: *-*-* *:*:00/5
       Next elapse: Mon 2026-06-15 11:01:00 PDT
          From now: 4s
"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=mock_output,
                stderr="",
            )
            result = authority.render_systemd_semantics(diff, held_hits)

        assert len(result) > 0
        # Must contain the cadence (From now or Next elapse substring), not the limited-detail fallback
        assert any("From now" in line or "Next elapse" in line for line in result)
        assert not any("limited detail" in line for line in result)

    def test_oncalendar_5min_cadence(self):
        """OnCalendar=*:00/5:00 should yield a ~5min cadence line."""
        diff = "+++ b/systemd/app.timer\n+OnCalendar=*:00/5:00"
        held_hits = ["systemd/app.timer"]

        mock_output = """\
  Original form: *:00/5:00
  Normalized form: *:00/5:00
       Next elapse: Mon 2026-06-15 11:05:00 PDT
          From now: 4min 23s
"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=mock_output,
                stderr="",
            )
            result = authority.render_systemd_semantics(diff, held_hits)

        assert len(result) > 0
        # Must contain the cadence (From now or Next elapse substring), not the limited-detail fallback
        assert any("From now" in line or "Next elapse" in line for line in result)
        assert not any("limited detail" in line for line in result)

    def test_systemd_analyze_missing(self):
        """Missing systemd-analyze binary yields unavailable note."""
        diff = "+++ b/systemd/app.timer\n+OnCalendar=*:*:*"
        held_hits = ["systemd/app.timer"]

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("systemd-analyze not found")
            result = authority.render_systemd_semantics(diff, held_hits)

        assert len(result) > 0
        assert any("unavailable" in line.lower() for line in result)

    def test_systemd_analyze_timeout(self):
        """systemd-analyze timeout yields unavailable note."""
        diff = "+++ b/systemd/app.timer\n+OnCalendar=*:*:*"
        held_hits = ["systemd/app.timer"]

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 5)
            result = authority.render_systemd_semantics(diff, held_hits)

        assert len(result) > 0
        assert any("timeout" in line.lower() for line in result)

    def test_systemd_analyze_nonzero_exit(self):
        """systemd-analyze non-zero exit yields unavailable note."""
        diff = "+++ b/systemd/app.timer\n+OnCalendar=invalid!@#"
        held_hits = ["systemd/app.timer"]

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Invalid calendar",
            )
            result = authority.render_systemd_semantics(diff, held_hits)

        assert len(result) > 0
        assert any("unavailable" in line.lower() and "1" in line for line in result)

    def test_oncalendar_no_cadence_fallback(self):
        """When systemd-analyze has no 'From now' line, fall back to limited-detail note."""
        diff = "+++ b/systemd/app.timer\n+OnCalendar=something"
        held_hits = ["systemd/app.timer"]

        # Mock output with no "From now" or "Next elapse" line
        mock_output = """\
  Original form: something
  Normalized form: something
"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=mock_output,
                stderr="",
            )
            result = authority.render_systemd_semantics(diff, held_hits)

        assert len(result) > 0
        # Should fall back to limited-detail note
        assert any("limited detail" in line for line in result)


class TestClassifyWithSystemdRendering:
    """Test that classify() correctly wires in systemd rendering."""

    def test_classify_held_path_with_timer_renders_semantics(self):
        """classify() with held timer path appends rendered semantics to reasons."""
        pr = {"title": "t", "html_url": "http://example.com", "mergeable": True}
        diff = "+++ b/systemd/app.timer\n+OnCalendar=*:*:00/5"

        mock_output = """\
  Original form: *:*:00/5
  Normalized form: *:*:00/5
       Next elapse: Mon 2026-06-15 11:01:00 PDT
          From now: 4s
"""
        with (
            patch("lapis_pm.authority.get_pr", return_value=pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=diff),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=mock_output,
                stderr="",
            )
            cls = authority.classify("myrepo", 1, "spec", pm_authority="advisory")

        assert cls.verdict == "hold"
        assert cls.static_outcome == authority.StaticOutcome.auto_hold_path
        # Should have both the base "held path(s) touched" line and a rendered line with cadence
        assert len(cls.reasons) >= 2
        assert "held path(s) touched" in cls.reasons[0]
        # The rendered line must contain the cadence (From now / Next elapse), not limited detail
        assert any(("From now" in r or "Next elapse" in r) for r in cls.reasons[1:])
        assert not any("limited detail" in r for r in cls.reasons)

    def test_classify_held_path_non_systemd_no_rendering(self):
        """classify() with non-systemd held path yields no extra lines."""
        pr = {"title": "t", "html_url": "http://example.com", "mergeable": True}
        diff = "+++ b/infra/deploy.sh\n+echo hi"

        with (
            patch("lapis_pm.authority.get_pr", return_value=pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=diff),
        ):
            cls = authority.classify("myrepo", 1, "spec", pm_authority="advisory")

        assert cls.verdict == "hold"
        assert cls.static_outcome == authority.StaticOutcome.auto_hold_path
        # Should have exactly 1 reason line (the base "held path(s) touched")
        assert len(cls.reasons) == 1
        assert "held path(s) touched" in cls.reasons[0]

    def test_classify_held_path_systemd_render_unavailable(self):
        """classify() handles render failures gracefully."""
        pr = {"title": "t", "html_url": "http://example.com", "mergeable": True}
        diff = "+++ b/systemd/app.timer\n+OnCalendar=*:*:*"

        with (
            patch("lapis_pm.authority.get_pr", return_value=pr),
            patch("lapis_pm.authority.get_pr_diff", return_value=diff),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = FileNotFoundError("not found")
            cls = authority.classify("myrepo", 1, "spec", pm_authority="advisory")

        # Must not raise; classification still returned
        assert cls.verdict == "hold"
        # Should have the base line + unavailable note
        assert len(cls.reasons) >= 2
        assert any("unavailable" in r.lower() for r in cls.reasons)
