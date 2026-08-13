"""Tests for lapis_pm.bundle_autodispatch.

Coverage:
- discovery: debt-bundle filename regex + source frontmatter gating
- discovery skips non-debt-bundle specs (self-collision fix, Defect A)
- proceed verdict → bind + tick; hold/incomplete → defer (no bind, no marker)
- GW-unreachable → defer (zero binds, no paid path, no marker)
- write-ahead pending marker lifecycle (atomic rename, never write-then-delete)
- crash detection: .autodispatch-pending present on later run → tombstone (Defect B fix)
- external bind: target YAML + no marker → SKIP + audit log, NEVER tombstone (Defect B regression)
- --dry-run: no writes/renames of any marker kind; logs "would tombstone" for stale pending
- atomicity: pending→success and pending→failed are renames, not write+delete
"""
from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import bundle_autodispatch as bad

_RUN_TS = "2026-06-27T00:00:00Z"


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

# A spec that looks like a bundle filename but has the wrong source frontmatter
WRONG_SOURCE_SPEC = textwrap.dedent("""\
    ---
    spec_id: cr-bundle-foo-2026-06-27
    status: draft
    created: 2026-06-27
    source: some-other-source
    ---

    **Repo:** `foo`
""")

# The autodispatch reconciler's own spec (should be excluded by filename - no ISO date)
AUTODISPATCH_SPEC = textwrap.dedent("""\
    ---
    spec_id: cr-bundle-autodispatch-v0
    status: draft
    created: 2026-06-27
    source: some-other-source
    ---

    **Repo:** `lapis-pm`
""")


def _write_spec(tmp_path: Path, name: str = "cr-bundle-myrepo-2026-06-26.md") -> Path:
    p = tmp_path / name
    p.write_text(BUNDLE_SPEC, encoding="utf-8")
    return p


def _make_brief(recommendation: str = "proceed-to-bind", **overrides):
    """A safely-defaulted brief mock. lapis-pm-bundle-autodispatch-enforce-v0's
    three-way classifier reads several fields beyond combined_recommendation
    (facets_deliberation, council_open_questions, council_status, ...) — a bare
    MagicMock() auto-creates those as truthy sub-mocks, which crashes the
    classifier's `list(...)` calls and would silently attempt a real GW call.
    Explicit safe defaults here keep every pre-existing test hermetic; pass
    overrides= for tests that need a specific classifier-relevant field."""
    brief = MagicMock()
    brief.combined_recommendation = recommendation
    brief.facets_deliberation = overrides.pop("facets_deliberation", None)
    brief.council_open_questions = overrides.pop("council_open_questions", [])
    brief.council_status = overrides.pop("council_status", "resolved")
    brief.council_positions = overrides.pop("council_positions", [])
    brief.parse_error = overrides.pop("parse_error", None)
    brief.council_error_reason = overrides.pop("council_error_reason", "")
    brief.council_run_id = overrides.pop("council_run_id", "run-test")
    for k, v in overrides.items():
        setattr(brief, k, v)
    return brief


def _reconcile(tmp_path, dry_run=False, **kw):
    return bad.reconcile(spec_dir=tmp_path, dry_run=dry_run, run_ts=_RUN_TS, **kw)


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
# Discovery — filename regex + source frontmatter gating (Defect A fix)
# ---------------------------------------------------------------------------

class TestDiscoverSpecs:
    def test_finds_valid_debt_bundle(self, tmp_path):
        _write_spec(tmp_path, "cr-bundle-myrepo-2026-06-26.md")
        specs = bad._discover_specs(tmp_path)
        assert len(specs) == 1
        assert specs[0].name == "cr-bundle-myrepo-2026-06-26.md"

    def test_excludes_no_date_filename(self, tmp_path):
        """cr-bundle-autodispatch-v0.md has no ISO date suffix — excluded by filename."""
        p = tmp_path / "cr-bundle-autodispatch-v0.md"
        p.write_text(AUTODISPATCH_SPEC, encoding="utf-8")
        specs = bad._discover_specs(tmp_path)
        assert specs == []

    def test_excludes_notadate_filename(self, tmp_path):
        """cr-bundle-notadate.md has no ISO date — excluded by filename."""
        p = tmp_path / "cr-bundle-notadate.md"
        p.write_text(BUNDLE_SPEC, encoding="utf-8")
        specs = bad._discover_specs(tmp_path)
        assert specs == []

    def test_excludes_wrong_source_frontmatter(self, tmp_path):
        """Correct filename but wrong source → excluded."""
        p = tmp_path / "cr-bundle-foo-2026-06-27.md"
        p.write_text(WRONG_SOURCE_SPEC, encoding="utf-8")
        specs = bad._discover_specs(tmp_path)
        assert specs == []

    def test_includes_correct_filename_and_source(self, tmp_path):
        """cr-bundle-foo-2026-06-27.md with debt-bundle source → included."""
        p = tmp_path / "cr-bundle-foo-2026-06-27.md"
        p.write_text(BUNDLE_SPEC, encoding="utf-8")
        specs = bad._discover_specs(tmp_path)
        assert len(specs) == 1
        assert specs[0].name == "cr-bundle-foo-2026-06-27.md"

    def test_excludes_superseded_subdir(self, tmp_path):
        superseded = tmp_path / "superseded"
        superseded.mkdir()
        (superseded / "cr-bundle-old-2026-01-01.md").write_text(BUNDLE_SPEC)
        _write_spec(tmp_path, "cr-bundle-live-2026-06-26.md")
        specs = bad._discover_specs(tmp_path)
        names = [p.name for p in specs]
        assert "cr-bundle-live-2026-06-26.md" in names
        assert "cr-bundle-old-2026-01-01.md" not in names

    def test_excludes_non_cr_bundle_files(self, tmp_path):
        (tmp_path / "unrelated.md").write_text("x")
        (tmp_path / "cr-other-2026-06-26.md").write_text("x")
        specs = bad._discover_specs(tmp_path)
        assert specs == []

    def test_empty_dir_returns_empty(self, tmp_path):
        assert bad._discover_specs(tmp_path) == []

    def test_self_exclusion_this_fix_spec(self, tmp_path):
        """cr-bundle-autodispatch-crashguard-fix-v0.md is excluded by filename (no ISO date)."""
        p = tmp_path / "cr-bundle-autodispatch-crashguard-fix-v0.md"
        p.write_text(BUNDLE_SPEC, encoding="utf-8")
        specs = bad._discover_specs(tmp_path)
        assert specs == []

    def test_hyphenated_repo_name_included(self, tmp_path):
        """cr-bundle-my-repo-2026-06-27.md has a hyphenated repo — still matched."""
        p = tmp_path / "cr-bundle-my-repo-2026-06-27.md"
        p.write_text(BUNDLE_SPEC, encoding="utf-8")
        specs = bad._discover_specs(tmp_path)
        assert len(specs) == 1


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------

class TestMarkers:
    def test_autodispatch_marker_path(self, tmp_path):
        p = tmp_path / "cr-bundle-x-2026-06-26.md"
        assert bad._autodispatch_marker(p) == Path(str(p) + ".autodispatch")

    def test_pending_marker_path(self, tmp_path):
        p = tmp_path / "cr-bundle-x-2026-06-26.md"
        assert bad._pending_marker(p) == Path(str(p) + ".autodispatch-pending")

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

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True) as mock_bind,
            patch.object(bad, "_tick", return_value=True) as mock_tick,
            patch.object(bad, "_verify_dispatched", return_value=True),
        ):
            results = _reconcile(tmp_path)

        mock_bind.assert_called_once_with(
            "cr-bundle-myrepo-2026-06-26", "myrepo", spec_path
        )
        mock_tick.assert_called_once_with("cr-bundle-myrepo-2026-06-26", "myrepo")
        assert len(results["bound"]) == 1
        assert results["bound"][0]["spec"] == "cr-bundle-myrepo-2026-06-26"

    def test_proceed_writes_pending_then_renames_to_autodispatch(self, tmp_path):
        """Pending marker written before bind, then atomically renamed to .autodispatch on success."""
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
            _reconcile(tmp_path)

        # .autodispatch-pending must be gone (renamed to .autodispatch)
        assert not bad._pending_marker(spec_path).exists(), \
            ".autodispatch-pending must be renamed away on success"
        # .autodispatch must now exist
        assert bad._autodispatch_marker(spec_path).exists(), \
            ".autodispatch must exist after successful bind+tick"

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
            _reconcile(tmp_path)

        assert not bad._failed_marker(spec_path).exists()

    def test_pending_written_before_bind(self, tmp_path):
        """The .autodispatch-pending marker must be written before bind is called."""
        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief("proceed-to-bind")
        pending_written_before_bind = []

        def check_pending_on_bind(*args, **kwargs):
            pending_written_before_bind.append(bad._pending_marker(spec_path).exists())
            return True

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", side_effect=check_pending_on_bind),
            patch.object(bad, "_tick", return_value=True),
            patch.object(bad, "_verify_dispatched", return_value=True),
        ):
            _reconcile(tmp_path)

        assert pending_written_before_bind == [True], \
            ".autodispatch-pending must exist when bind() is called"

    def test_pending_marker_content_includes_ts(self, tmp_path):
        """Pending marker must record the injected run_ts."""
        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief("proceed-to-bind")
        captured = []

        original_write = Path.write_text
        def capture_write(self, text, *a, **kw):
            if ".autodispatch-pending" in str(self):
                captured.append(text)
            return original_write(self, text, *a, **kw)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True),
            patch.object(bad, "_tick", return_value=True),
            patch.object(bad, "_verify_dispatched", return_value=True),
            patch.object(Path, "write_text", capture_write),
        ):
            _reconcile(tmp_path)

        assert any(_RUN_TS in c for c in captured), \
            "Pending marker must embed the injected run_ts"


# ---------------------------------------------------------------------------
# hold / non-proceed → defer (no bind, no marker)
# ---------------------------------------------------------------------------

class TestHoldDefer:
    @pytest.mark.parametrize("recommendation", [
        "amend-spec", "shape-with-Erah",
    ])
    def test_non_proceed_defers_no_bind(self, tmp_path, recommendation):
        """amend-spec / shape-with-Erah with no facets/council grounds present
        (the default _make_brief) route through the salvage classifier, find
        empty grounds, and fail-closed to DEFER — same observable outcome as
        the pre-enforce behavior. See TestThreeWayClassification for the
        infra/salvage split with real grounds present."""
        _write_spec(tmp_path)
        mock_brief = _make_brief(recommendation)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True) as mock_bind,
            patch.object(bad, "_tick", return_value=True) as mock_tick,
        ):
            results = _reconcile(tmp_path)

        mock_bind.assert_not_called()
        mock_tick.assert_not_called()
        assert len(results["deferred"]) == 1
        assert results["deferred"][0]["reason"] == f"gate:{recommendation}"

    @pytest.mark.parametrize("recommendation", [
        "incomplete", "parse_failed",
    ])
    def test_incomplete_and_parse_failed_are_infra_not_defer(self, tmp_path, recommendation):
        """lapis-pm-bundle-autodispatch-enforce-v0: incomplete/parse_failed are
        now INFRA triggers, not blanket defer — never bound, never deferred to
        human, counted in the new `faulted` bucket."""
        _write_spec(tmp_path)
        mock_brief = _make_brief(recommendation)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True) as mock_bind,
            patch.object(bad, "_tick", return_value=True) as mock_tick,
        ):
            results = _reconcile(tmp_path)

        mock_bind.assert_not_called()
        mock_tick.assert_not_called()
        assert results["deferred"] == []
        assert len(results["faulted"]) == 1
        assert recommendation in results["faulted"][0]["ground"]

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
            _reconcile(tmp_path)

        assert not bad._autodispatch_marker(spec_path).exists()
        assert not bad._pending_marker(spec_path).exists()
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
            results = _reconcile(tmp_path)

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
            _reconcile(tmp_path)

        assert not bad._autodispatch_marker(spec_path).exists()
        assert not bad._pending_marker(spec_path).exists()
        assert not bad._failed_marker(spec_path).exists()

    def test_gate_none_return_is_infra_not_defer(self, tmp_path):
        """lapis-pm-bundle-autodispatch-enforce-v0: _run_gate returning None
        (timeout/error) is now an INFRA trigger — retried once (its own text
        names a timeout), never bound, never deferred to a human."""
        _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=None) as mock_gate,
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = _reconcile(tmp_path)

        mock_bind.assert_not_called()
        assert results["deferred"] == []
        assert len(results["faulted"]) == 1
        assert results["faulted"][0]["retried"] is True
        # retried once: initial gate call + the one infra retry
        assert mock_gate.call_count == 2


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
            results = _reconcile(tmp_path)

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
            results = _reconcile(tmp_path)

        mock_gate.assert_not_called()
        mock_bind.assert_not_called()
        assert results["skipped"][0]["reason"] == "failed_tombstone"

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
            results = _reconcile(tmp_path)

        mock_gate.assert_not_called()
        mock_bind.assert_not_called()
        assert results["skipped"]


# ---------------------------------------------------------------------------
# Crash detection: .autodispatch-pending on later run → tombstone (Defect B fix)
# ---------------------------------------------------------------------------

class TestCrashDetection:
    def test_stale_pending_tombstones_via_rename(self, tmp_path):
        """Stale .autodispatch-pending → rename to .autodispatch-failed + loud log."""
        spec_path = _write_spec(tmp_path)
        bad._pending_marker(spec_path).write_text("stale pending from crashed run")

        results = _reconcile(tmp_path)

        # pending must be gone
        assert not bad._pending_marker(spec_path).exists(), \
            ".autodispatch-pending must be renamed away on crash detection"
        # failed must exist (renamed from pending)
        assert bad._failed_marker(spec_path).exists(), \
            ".autodispatch-failed must be created from pending on crash detection"
        assert results["failed"][0]["reason"] == "crash_pending_marker"

    def test_stale_pending_tombstone_is_idempotent(self, tmp_path):
        """After the crash tombstone is written, a second run SKIPs (failed_tombstone)."""
        spec_path = _write_spec(tmp_path)
        bad._pending_marker(spec_path).write_text("stale")

        # First run: crash detected, pending → failed
        r1 = _reconcile(tmp_path)
        assert r1["failed"][0]["reason"] == "crash_pending_marker"
        assert bad._failed_marker(spec_path).exists()

        # Second run: failed tombstone → SKIP
        with patch.object(bad, "_bind") as mock_bind:
            r2 = _reconcile(tmp_path)

        mock_bind.assert_not_called()
        assert r2["skipped"][0]["reason"] == "failed_tombstone"

    def test_stale_pending_logs_error(self, tmp_path, caplog):
        """Stale pending detection must emit an ERROR-level log."""
        spec_path = _write_spec(tmp_path)
        bad._pending_marker(spec_path).write_text("stale")

        with caplog.at_level(logging.ERROR, logger="lapis_pm.bundle_autodispatch"):
            _reconcile(tmp_path)

        assert any("stale .autodispatch-pending" in r.message for r in caplog.records), \
            "Must emit ERROR-level log on crash detection"

    def test_pending_and_success_never_coexist(self, tmp_path):
        """After a successful run, both .pending and .success cannot coexist."""
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
            _reconcile(tmp_path)

        pending = bad._pending_marker(spec_path).exists()
        success = bad._autodispatch_marker(spec_path).exists()
        assert not pending, ".autodispatch-pending must not exist after success"
        assert success, ".autodispatch must exist after success"
        # They never coexist
        assert not (pending and success)


# ---------------------------------------------------------------------------
# External bind: target YAML + no marker → SKIP, NEVER tombstone (Defect B regression)
# ---------------------------------------------------------------------------

class TestExternalBindSkip:
    def test_external_bind_is_skip_not_tombstone(self, tmp_path):
        """Regression: cr-bundle-conductor-2026-06-23 shape — target YAML + no markers → SKIP.

        This was the false-positive tombstone path in PR #191 (Defect B).
        The fix: absence of a success marker is NOT evidence of a crash.
        Only an unresolved .autodispatch-pending is evidence of a crash.
        """
        _write_spec(tmp_path, "cr-bundle-conductor-2026-06-23.md")

        with patch.object(bad, "_target_yaml_exists", return_value=True):
            results = _reconcile(tmp_path)

        assert results["skipped"][0]["reason"] == "externally_bound"
        assert results["failed"] == [], "Must NEVER tombstone an externally-bound spec"

    def test_external_bind_does_not_write_failed_marker(self, tmp_path):
        """No .autodispatch-failed marker must be written for externally-bound specs."""
        spec_path = _write_spec(tmp_path, "cr-bundle-conductor-2026-06-23.md")

        with patch.object(bad, "_target_yaml_exists", return_value=True):
            _reconcile(tmp_path)

        assert not bad._failed_marker(spec_path).exists()

    def test_external_bind_emits_audit_log(self, tmp_path, caplog):
        """External bind skip must emit a high-priority (WARNING) audit log line."""
        _write_spec(tmp_path, "cr-bundle-conductor-2026-06-23.md")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=True),
            caplog.at_level(logging.WARNING, logger="lapis_pm.bundle_autodispatch"),
        ):
            _reconcile(tmp_path)

        assert any("externally" in r.message.lower() or "manually bound" in r.message.lower()
                   for r in caplog.records), \
            "Must emit audit-level log for externally-bound skip"

    def test_external_bind_does_not_call_bind(self, tmp_path):
        spec_path = _write_spec(tmp_path, "cr-bundle-conductor-2026-06-23.md")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=True),
            patch.object(bad, "_bind") as mock_bind,
            patch.object(bad, "_tick") as mock_tick,
        ):
            _reconcile(tmp_path)

        mock_bind.assert_not_called()
        mock_tick.assert_not_called()

    def test_external_bind_also_cr_bundle_autodispatch_v0_shape(self, tmp_path):
        """cr-bundle-autodispatch-v0.md is excluded by filename before state check."""
        p = tmp_path / "cr-bundle-autodispatch-v0.md"
        p.write_text(AUTODISPATCH_SPEC, encoding="utf-8")

        with patch.object(bad, "_target_yaml_exists", return_value=True):
            results = _reconcile(tmp_path)

        # Excluded by filename — never reaches state-check
        assert results == {
            "bound": [], "deferred": [], "skipped": [], "failed": [],
            "faulted": [], "salvaged": [],
        }


# ---------------------------------------------------------------------------
# Tick failure → rename pending → .autodispatch-failed (atomic)
# ---------------------------------------------------------------------------

class TestTickFailure:
    def test_tick_failure_renames_pending_to_failed(self, tmp_path):
        """On tick failure, .autodispatch-pending is atomically renamed to .autodispatch-failed."""
        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief("proceed-to-bind")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=True),
            patch.object(bad, "_tick", return_value=False),
        ):
            results = _reconcile(tmp_path)

        assert results["failed"][0]["reason"] == "tick_failed"
        assert bad._failed_marker(spec_path).exists()
        assert not bad._autodispatch_marker(spec_path).exists()
        assert not bad._pending_marker(spec_path).exists(), \
            ".autodispatch-pending must be renamed away on tick failure"

    def test_dispatched_not_verified_renames_pending_to_failed(self, tmp_path):
        """On dispatched<1 after tick, pending is atomically renamed to failed."""
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
            results = _reconcile(tmp_path)

        assert results["failed"][0]["reason"] == "dispatch_not_verified"
        assert bad._failed_marker(spec_path).exists()
        assert not bad._autodispatch_marker(spec_path).exists()
        assert not bad._pending_marker(spec_path).exists()

    def test_bind_failure_renames_pending_to_failed(self, tmp_path):
        """On bind failure, .autodispatch-pending is atomically renamed to .autodispatch-failed.

        Regression for Defect B fix: bind failure must not leave .autodispatch-pending in place,
        which would be misread as a crash by the next run's state-3 crash detection.
        """
        spec_path = _write_spec(tmp_path)
        mock_brief = _make_brief("proceed-to-bind")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_run_gate", return_value=mock_brief),
            patch.object(bad, "_bind", return_value=False),
        ):
            results = _reconcile(tmp_path)

        assert results["failed"][0]["reason"] == "bind_failed"
        assert bad._failed_marker(spec_path).exists(), \
            ".autodispatch-failed must exist after bind failure"
        assert not bad._pending_marker(spec_path).exists(), \
            ".autodispatch-pending must be renamed away on bind failure"
        assert not bad._autodispatch_marker(spec_path).exists()

    def test_bind_failure_pending_left_then_next_run_tombstones(self, tmp_path):
        """A stale .autodispatch-pending from a bind failure tombstones on the next reconcile run.

        This validates the state-3 crash detection path: if a previous run wrote the pending
        marker but then crashed before bind completed (leaving pending in place), the next run
        correctly tombstones it.
        """
        spec_path = _write_spec(tmp_path)
        # Simulate state: target YAML exists, pending marker left by a crashed prior run
        bad._pending_marker(spec_path).write_text("autodispatch-pending: spec_id=cr-bundle-foo-2026-01-01 ts=2026-01-01T00:00:00\n")

        with (
            patch.object(bad, "_target_yaml_exists", return_value=True),
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = _reconcile(tmp_path)

        mock_bind.assert_not_called()
        assert bad._failed_marker(spec_path).exists(), \
            "state-3: stale pending must be tombstoned as .autodispatch-failed"
        assert not bad._pending_marker(spec_path).exists()
        assert results["failed"][0]["reason"] == "crash_pending_marker"


# ---------------------------------------------------------------------------
# --dry-run: fully side-effect-free for ALL marker states
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_fresh_spec_no_bind_no_marker(self, tmp_path):
        spec_path = _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_bind") as mock_bind,
            patch.object(bad, "_tick") as mock_tick,
        ):
            results = _reconcile(tmp_path, dry_run=True)

        mock_bind.assert_not_called()
        mock_tick.assert_not_called()
        assert not bad._autodispatch_marker(spec_path).exists()
        assert not bad._pending_marker(spec_path).exists()
        assert not bad._failed_marker(spec_path).exists()
        # dry-run still counts as "would-bind"
        assert results["bound"][0]["dry_run"] is True

    def test_dry_run_stale_pending_logs_but_no_rename(self, tmp_path, caplog):
        """Dry-run with stale .autodispatch-pending: logs 'would tombstone', touches nothing."""
        spec_path = _write_spec(tmp_path)
        bad._pending_marker(spec_path).write_text("stale from crashed run")

        with caplog.at_level(logging.ERROR, logger="lapis_pm.bundle_autodispatch"):
            results = _reconcile(tmp_path, dry_run=True)

        # Pending marker must still be there (dry-run never renames)
        assert bad._pending_marker(spec_path).exists(), \
            "dry-run must not rename .autodispatch-pending"
        # Failed marker must NOT appear
        assert not bad._failed_marker(spec_path).exists(), \
            "dry-run must not create .autodispatch-failed"
        # Must log "would tombstone" or equivalent
        assert any("would" in r.message.lower() for r in caplog.records), \
            "dry-run must log that it would tombstone"
        # Result still reported in failed for observability
        assert results["failed"][0]["reason"] == "crash_pending_marker"

    def test_dry_run_gw_not_serving_defers(self, tmp_path):
        _write_spec(tmp_path)

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=False),
            patch.object(bad, "_bind") as mock_bind,
        ):
            results = _reconcile(tmp_path, dry_run=True)

        mock_bind.assert_not_called()
        assert results["deferred"][0]["reason"] == "gw_not_serving"

    def test_dry_run_external_bind_no_tombstone(self, tmp_path):
        """Dry-run over externally-bound spec: SKIP, no marker write."""
        spec_path = _write_spec(tmp_path)

        with patch.object(bad, "_target_yaml_exists", return_value=True):
            results = _reconcile(tmp_path, dry_run=True)

        assert results["skipped"][0]["reason"] == "externally_bound"
        assert not bad._failed_marker(spec_path).exists()
        assert not bad._pending_marker(spec_path).exists()

    def test_dry_run_filesystem_identical_over_all_shapes(self, tmp_path):
        """Dry-run over all marker shapes leaves the filesystem byte-identical."""
        # Set up one spec in each state
        fresh_spec = _write_spec(tmp_path, "cr-bundle-fresh-2026-06-27.md")

        done_spec = _write_spec(tmp_path, "cr-bundle-done-2026-06-27.md")
        bad._autodispatch_marker(done_spec).write_text("done")

        failed_spec = _write_spec(tmp_path, "cr-bundle-failed-2026-06-27.md")
        bad._failed_marker(failed_spec).write_text("failed tombstone")

        stale_spec = _write_spec(tmp_path, "cr-bundle-stale-2026-06-27.md")
        bad._pending_marker(stale_spec).write_text("stale pending")

        # Snapshot filesystem state before dry-run
        def fs_snapshot():
            return {p.name: p.read_text() for p in tmp_path.iterdir() if p.is_file()}

        before = fs_snapshot()

        with (
            patch.object(bad, "_target_yaml_exists", return_value=False),
            patch.object(bad, "_gw_serving", return_value=True),
            patch.object(bad, "_bind") as mock_bind,
            patch.object(bad, "_tick") as mock_tick,
        ):
            _reconcile(tmp_path, dry_run=True)

        after = fs_snapshot()

        mock_bind.assert_not_called()
        mock_tick.assert_not_called()
        assert before == after, \
            "dry-run must leave the filesystem byte-identical (no writes/renames of any marker)"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLI:
    def test_bundle_autodispatch_subcommand_registered(self):
        from lapis_pm.cli import build_parser
        p = build_parser()
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


# ---------------------------------------------------------------------------
# Journal-line contract sentinel (lapis-pm-bundle-autodispatch-enforce-v0,
# gate-amended 2026-08-13, Council — DoD 7a).
#
# loupe's navigator (loupe/navigator/subjects.py) greps the service journal
# for the LITERAL "run complete: bound=" substring and regexes bound=(\d+)
# to judge subject health. This is enforced law, not convention: the
# existing bound/deferred/skipped/failed buckets must never reorder or
# rename, and "bound=" must never be preceded by anything else. New buckets
# (faulted, salvaged) are APPENDED only. This test fails loudly on any
# rearrangement disguised as improvement — including a well-intentioned
# alphabetical resort or a rename for "clarity".
# ---------------------------------------------------------------------------

class TestJournalLineSentinel:
    _EXPECTED_LINE_RE = re.compile(
        r"run complete: "
        r"bound=(\d+) deferred=(\d+) skipped=(\d+) failed=(\d+) "
        r"faulted=(\d+) salvaged=(\d+)$"
    )

    def test_journal_line_bound_is_first_and_buckets_are_not_reordered(self, tmp_path, caplog):
        """The literal contract loupe's navigator parses: 'bound=' must be
        the first bucket in the line, immediately after 'run complete: ',
        and the six buckets must appear in exactly this order with exactly
        these names. Any reorder, rename, or insertion before bound= is a
        contract break for loupe, silently, in production."""
        caplog.set_level(logging.INFO, logger="lapis_pm.bundle_autodispatch")

        with (
            patch.object(bad, "_discover_specs", return_value=[]),
        ):
            bad.reconcile(spec_dir=tmp_path, dry_run=True, run_ts=_RUN_TS)

        complete_lines = [
            r.getMessage() for r in caplog.records if "run complete:" in r.getMessage()
        ]
        assert len(complete_lines) == 1, (
            f"expected exactly one 'run complete:' journal line, got {complete_lines!r}"
        )
        line = complete_lines[0]

        # bound= must be the literal substring immediately following
        # "run complete: " — nothing else may precede it.
        idx = line.index("run complete: ")
        after = line[idx + len("run complete: "):]
        assert after.startswith("bound="), (
            f"'bound=' must be the first bucket after 'run complete: ', got: {after!r}"
        )

        m = self._EXPECTED_LINE_RE.search(line)
        assert m is not None, (
            f"journal line does not match the pinned bound/deferred/skipped/failed/"
            f"faulted/salvaged contract (order + names): {line!r}"
        )

    def test_journal_line_buckets_reflect_actual_counts(self, tmp_path, caplog):
        """Sanity companion to the sentinel above: the six counts in the
        journal line must reflect the actual results dict, not just match
        the regex shape."""
        caplog.set_level(logging.INFO, logger="lapis_pm.bundle_autodispatch")

        with patch.object(bad, "_discover_specs", return_value=[]):
            results = bad.reconcile(spec_dir=tmp_path, dry_run=True, run_ts=_RUN_TS)

        line = next(
            r.getMessage() for r in caplog.records if "run complete:" in r.getMessage()
        )
        m = self._EXPECTED_LINE_RE.search(line)
        assert m is not None
        assert [int(g) for g in m.groups()] == [
            len(results["bound"]), len(results["deferred"]), len(results["skipped"]),
            len(results["failed"]), len(results["faulted"]), len(results["salvaged"]),
        ]
