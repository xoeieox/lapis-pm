"""Tests for lapis_pm.bundle_autodispatch.

Coverage:
- discovery skips already-bound specs, superseded/, parses frontmatter from YAML+markdown
- proceed verdict → bind + tick; hold/incomplete → defer (no bind, no marker)
- GW-unreachable → defer (zero binds, no paid path, no marker)
- idempotency: .autodispatch marker skips; partial state (YAML exists, no marker) → failed tombstone
- --dry-run: no binds, no markers written
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from lapis_pm import bundle_autodispatch as bad


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUNDLE_SPEC = textwrap.dedent("""\
    ---
    spec_id: cr-bundle-myrepo-2026-06-26
    status: draft
    created: 2026-06-26
    source: code-reviewer-debt-bundle-v0 nightly sweep
    ---

    # Spec: cr-bundle-myrepo-2026-06-26 — agentic debt fix-bundle

    **Repo:** `myrepo`
    **Authority:** advisory

    ## Goal

    Resolve 3 open MED-severity debt items in `myrepo`.

    **Suggested bind:** `lapis-pm bind cr-bundle-myrepo-2026-06-26 --repo myrepo --authority advisory --create --title "Resolve 3 aged MED debt items in myrepo"`
""")


def _write_spec(tmp_path: Path, name: str = "cr-bundle-myrepo-2026-06-26.md") -> Path:
    p = tmp_path / name
    p.write_text(BUNDLE_SPEC, encoding="utf-8")
    return p


def _make_brief(recommendation: str = "proceed-to-bind"):
    """Return a MagicMock SpecReviewBrief with the given recommendation."""
    brief = MagicMock()
    brief.combined_recommendation = recommendation
    return brief


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

class TestParseBundleFrontmatter:
    def test_parse_valid(self, tmp_path):
        p = _write_spec(tmp_path)
        spec_id, repo = bad._parse_bundle_frontmatter(p)
        assert spec_id == "cr-bundle-myrepo-2026-06-26"
        assert repo == "myrepo"

    def test_missing_spec_id_raises(self, tmp_path):
        p = tmp_path / "cr-bundle-x.md"
        p.write_text("**Repo:** `myrepo`\n", encoding="utf-8")
        with pytest.raises(ValueError, match="spec_id"):
            bad._parse_bundle_frontmatter(p)

    def test_missing_repo_raises(self, tmp_path):
        p = tmp_path / "cr-bundle-x.md"
        p.write_text("spec_id: cr-bundle-x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Repo"):
            bad._parse_bundle_frontmatter(p)

    def test_missing_file_raises(self, tmp_path):
        p = tmp_path / "nonexistent.md"
        with pytest.raises(ValueError, match="cannot read spec"):
            bad._parse_bundle_frontmatter(p)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscoverSpecs:
    def test_finds_cr_bundle_files(self, tmp_path):
        (tmp_path / "cr-bundle-a-2026-06-01.md").write_text("x")
        (tmp_path / "cr-bundle-b-2026-06-02.md").write_text("x")
        (tmp_path / "unrelated.md").write_text("x")
        specs = bad._discover_specs(tmp_path)
        names = [p.name for p in specs]
        assert "cr-bundle-a-2026-06-01.md" in names
        assert "cr-bundle-b-2026-06-02.md" in names
        assert "unrelated.md" not in names

    def test_excludes_superseded_subdir(self, tmp_path):
        superseded = tmp_path / "superseded"
        superseded.mkdir()
        (superseded / "cr-bundle-old-2026-01-01.md").write_text("x")
        (tmp_path / "cr-bundle-live-2026-06-26.md").write_text("x")
        specs = bad._discover_specs(tmp_path)
        names = [p.name for p in specs]
        assert "cr-bundle-live-2026-06-26.md" in names
        assert "cr-bundle-old-2026-01-01.md" not in names

    def test_empty_dir_returns_empty(self, tmp_path):
        assert bad._discover_specs(tmp_path) == []


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------

class TestMarkers:
    def test_autodispatch_marker_path(self, tmp_path):
        p = tmp_path / "cr-bundle-x-2026-06-26.md"
        assert bad._autodispatch_marker(p) == Path(str(p) + ".autodispatch")

    def test_failed_marker_path(self, tmp_path):
        p = tmp_path / "cr-bundle-x-2026-06-26.md"
        assert bad._failed_marker(p) == Path(str(p) + ".autodispatch-failed")


# ---------------------------------------------------------------------------
# proceed-to-bind → bind + tick
# ---------------------------------------------------------------------------

class TestProceedToBind:
    def test_proceed_calls_bind_and_tick_once(self, tmp_path):
        spec_path = _write_spec(tmp_path)

        mock_brief = _make_brief("proceed-to-bind")
        mock_dispatch = MagicMock(return_value=[{"status": "pending"}])

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True) as mock_bind,
            patch.object(bad, "_tick", return_value=True) as mock_tick,
            patch.object(bad, "_verify_dispatched", return_value=True),
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        mock_bind.assert_called_once_with(
            "cr-bundle-myrepo-2026-06-26", "myrepo", spec_path
        )
        mock_tick.assert_called_once_with("cr-bundle-myrepo-2026-06-26", "myrepo")
        assert len(results["bound"]) == 1
        assert results["bound"][0]["spec"] == "cr-bundle-myrepo-2026-06-26"

    def test_proceed_writes_autodispatch_marker(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief("proceed-to-bind")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True),
            patch.object(bad, "_tick", return_value=True),
            patch.object(bad, "_verify_dispatched", return_value=True),
        ):
            bad.reconcile(spec_dir=tmp_path)

        marker = bad._autodispatch_marker(spec_path)
        assert marker.exists(), ".autodispatch marker must be written after verified bind+tick"

    def test_proceed_no_failed_marker_on_success(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief("proceed-to-bind")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True),
            patch.object(bad, "_tick", return_value=True),
            patch.object(bad, "_verify_dispatched", return_value=True),
        ):
            bad.reconcile(spec_dir=tmp_path)

        assert not bad._failed_marker(spec_path).exists()


# ---------------------------------------------------------------------------
# hold / non-proceed → defer (no bind, no marker)
# ---------------------------------------------------------------------------

class TestHoldDefer:
    @pytest.mark.parametrize("recommendation", [
        "amend-spec", "shape-with-Erah", "incomplete", "parse_failed",
    ])
    def test_non_proceed_defers_no_bind(self, tmp_path, recommendation):
        _write_spec(tmp_path)
        mock_brief = _make_brief(recommendation)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True) as mock_bind,
            patch.object(bad, "_tick", return_value=True) as mock_tick,
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        mock_bind.assert_not_called()
        mock_tick.assert_not_called()
        assert len(results["deferred"]) == 1
        assert results["deferred"][0]["reason"] == f"gate:{recommendation}"

    @pytest.mark.parametrize("recommendation", [
        "amend-spec", "shape-with-Erah", "incomplete",
    ])
    def test_non_proceed_writes_no_marker(self, tmp_path, recommendation):
        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief(recommendation)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True),
            patch.object(bad, "_tick", return_value=True),
        ):
            bad.reconcile(spec_dir=tmp_path)

        assert not bad._autodispatch_marker(spec_path).exists()
        assert not bad._failed_marker(spec_path).exists()


# ---------------------------------------------------------------------------
# GW-unreachable → defer (no bind, no paid path)
# ---------------------------------------------------------------------------

class TestGWUnreachable:
    def test_gw_not_serving_defers(self, tmp_path):
        _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=False),
            patch.object(bad, "_run_gate") as mock_gate,
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        mock_gate.assert_not_called()
        mock_bind.assert_not_called()
        assert len(results["deferred"]) == 1
        assert results["deferred"][0]["reason"] == "gw_not_serving"

    def test_gw_not_serving_no_marker(self, tmp_path):
        spec_path = _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=False),
            patch.object(bad, "_run_gate"),
        ):
            bad.reconcile(spec_dir=tmp_path)

        assert not bad._autodispatch_marker(spec_path).exists()
        assert not bad._failed_marker(spec_path).exists()

    def test_gate_none_return_defers(self, tmp_path):
        """_run_gate returning None (timeout/error) defers without bind."""
        _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=None),
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        mock_bind.assert_not_called()
        assert results["deferred"][0]["reason"] == "gate_timeout_or_error"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_autodispatch_marker_skips_spec(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        bad._autodispatch_marker(spec_path).write_text("")

        with (
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate") as mock_gate,
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        mock_gate.assert_not_called()
        mock_bind.assert_not_called()
        assert results["skipped"][0]["reason"] == "already_dispatched"

    def test_failed_tombstone_skips_spec(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        bad._failed_marker(spec_path).write_text("failed")

        with (
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate") as mock_gate,
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        mock_gate.assert_not_called()
        mock_bind.assert_not_called()
        assert results["skipped"][0]["reason"] == "failed_tombstone"

    def test_partial_state_yaml_exists_no_marker_writes_failed(self, tmp_path):
        """Target YAML exists but no .autodispatch marker → partial bind → tombstone."""
        spec_path = _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=True),
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        mock_bind.assert_not_called()
        assert results["failed"][0]["reason"] == "partial_bind_no_marker"
        assert bad._failed_marker(spec_path).exists()

    def test_partial_state_no_double_bind(self, tmp_path):
        """A spec in partial state must not trigger a bind."""
        _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=True),
            patch.object(bad, "_bind") as mock_bind,
            patch.object(bad, "_tick") as mock_tick,
        ):
            bad.reconcile(spec_dir=tmp_path)

        mock_bind.assert_not_called()
        mock_tick.assert_not_called()

    def test_rerun_after_autodispatch_marker_is_noop(self, tmp_path):
        """Second run with existing marker: no gate, no bind, no tick."""
        spec_path = _write_spec(tmp_path)
        bad._autodispatch_marker(spec_path).write_text("done")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=True),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate") as mock_gate,
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        mock_gate.assert_not_called()
        mock_bind.assert_not_called()
        assert results["skipped"]


# ---------------------------------------------------------------------------
# Tick failure → .autodispatch-failed tombstone
# ---------------------------------------------------------------------------

class TestTickFailure:
    def test_tick_failure_writes_failed_marker(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief("proceed-to-bind")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True),
            patch.object(bad, "_tick", return_value=False),
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        assert results["failed"][0]["reason"] == "tick_failed"
        assert bad._failed_marker(spec_path).exists()
        assert not bad._autodispatch_marker(spec_path).exists()

    def test_dispatched_not_verified_writes_failed_marker(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief("proceed-to-bind")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True),
            patch.object(bad, "_tick", return_value=True),
            patch.object(bad, "_verify_dispatched", return_value=False),
        ):
            results = bad.reconcile(spec_dir=tmp_path)

        assert results["failed"][0]["reason"] == "dispatch_not_verified"
        assert bad._failed_marker(spec_path).exists()
        assert not bad._autodispatch_marker(spec_path).exists()


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_no_bind_no_marker(self, tmp_path):
        spec_path = _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_bind") as mock_bind,
            patch.object(bad, "_tick") as mock_tick,
        ):
            results = bad.reconcile(spec_dir=tmp_path, dry_run=True)

        mock_bind.assert_not_called()
        mock_tick.assert_not_called()
        assert not bad._autodispatch_marker(spec_path).exists()
        assert not bad._failed_marker(spec_path).exists()
        # dry-run still counts as "would-bind"
        assert results["bound"][0]["dry_run"] is True

    def test_dry_run_gw_not_serving_defers(self, tmp_path):
        _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=False),
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = bad.reconcile(spec_dir=tmp_path, dry_run=True)

        mock_bind.assert_not_called()
        assert results["deferred"][0]["reason"] == "gw_not_serving"

    def test_dry_run_partial_state_no_marker_write(self, tmp_path):
        """In dry-run, partial state is detected but no tombstone is written."""
        spec_path = _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=True),
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = bad.reconcile(spec_dir=tmp_path, dry_run=True)

        mock_bind.assert_not_called()
        assert results["failed"][0]["reason"] == "partial_bind_no_marker"
        assert not bad._failed_marker(spec_path).exists()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLI:
    def test_bundle_autodispatch_subcommand_registered(self):
        from lapis_pm.cli import build_parser
        p = build_parser()
        # Should not raise
        args = p.parse_args(["bundle-autodispatch", "--dry-run"])
        assert args.dry_run is True

    def test_bundle_autodispatch_spec_dir_arg(self, tmp_path):
        from lapis_pm.cli import build_parser
        p = build_parser()
        args = p.parse_args(["bundle-autodispatch", "--spec-dir", str(tmp_path)])
        assert args.spec_dir == str(tmp_path)

    def test_bundle_autodispatch_gate_timeout_arg(self):
        from lapis_pm.cli import build_parser
        p = build_parser()
        args = p.parse_args(["bundle-autodispatch", "--gate-timeout", "600"])
        assert args.gate_timeout == 600

    def test_cmd_returns_0_on_no_failures(self, tmp_path):
        from lapis_pm.cli import cmd_bundle_autodispatch
        import argparse

        args = argparse.Namespace(
            dry_run=True,
            spec_dir=str(tmp_path),
            gate_timeout=1800,
        )
        # Empty dir → no failures
        rc = cmd_bundle_autodispatch(args)
        assert rc == 0

    def test_cmd_returns_1_on_failures(self, tmp_path):
        from lapis_pm.cli import cmd_bundle_autodispatch
        import argparse

        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief("proceed-to-bind")

        args = argparse.Namespace(
            dry_run=False,
            spec_dir=str(tmp_path),
            gate_timeout=1800,
        )

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True),
            patch.object(bad, "_tick", return_value=False),  # tick fails → failed list
        ):
            rc = cmd_bundle_autodispatch(args)

        assert rc == 1
