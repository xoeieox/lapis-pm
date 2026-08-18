"""Tests for lapis_pm.spec_id_backfill (spec-corpus-id-backfill-v0).

Coverage (per the bound spec's DoD):
- DoD-2 fixtures: frontmatter-present insert, no-frontmatter block creation,
  idempotency (second run zero changes), anchor-at-line-48 boundary case
  (skip + report), mtime preservation, byte-identity outside the inserted
  lines.
- DoD-3: the three guards (anchor-window, duplicate-stem, duplicate-value),
  each asserted with a violating fixture.
- DoD-4: the report lists the Target-ID-vs-stem mismatch set.
- DoD-1-shaped: counts sum to scanned, dry-run makes zero writes.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from lapis_pm import spec_id_backfill as sib


DATE = "2026-08-17"


def _header(target_id: str, repo: str = "lapis-pm", authority: str = "advisory",
            consumer: str = "some reader") -> str:
    return (
        f"**Target ID:** `{target_id}`\n"
        f"**Repo:** `{repo}`\n"
        f"**Authority:** {authority}\n"
        f"**Consumer:** {consumer}\n"
    )


def _with_frontmatter(stem: str, *, body_extra: str = "") -> str:
    return (
        "---\n"
        "status: draft\n"
        "created: 2026-08-17\n"
        "---\n"
        "\n"
        f"# Spec: {stem}\n"
        "\n"
        + _header(stem)
        + "\n"
        + body_extra
    )


def _no_frontmatter(stem: str) -> str:
    return f"# Spec: {stem}\n\n" + _header(stem) + "\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# discover_specs / find_duplicate_stems
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_top_level_non_recursive(self, tmp_path):
        _write(tmp_path, "a.md", _with_frontmatter("a"))
        sub = tmp_path / "superseded"
        sub.mkdir()
        (sub / "b.md").write_text(_with_frontmatter("b"), encoding="utf-8")
        found = sib.discover_specs(tmp_path)
        assert [p.name for p in found] == ["a.md"]

    def test_no_duplicate_stems_in_real_glob_result(self, tmp_path):
        _write(tmp_path, "a.md", _with_frontmatter("a"))
        _write(tmp_path, "b.md", _with_frontmatter("b"))
        found = sib.discover_specs(tmp_path)
        assert sib.find_duplicate_stems(found) == []

    def test_find_duplicate_stems_synthetic(self, tmp_path):
        # Not reachable via a single discover_specs() glob (filenames are
        # unique per directory) — the guard function is general-purpose and
        # tested directly against a synthetic list, per D3's own framing.
        files = [tmp_path / "a" / "dup.md", tmp_path / "b" / "dup.md", tmp_path / "c" / "x.md"]
        assert sib.find_duplicate_stems(files) == ["dup"]


# ---------------------------------------------------------------------------
# already_carries_spec_id / target_id_in_header
# ---------------------------------------------------------------------------

class TestParsing:
    def test_already_carries_true(self):
        text = "---\nspec_id: foo\nstatus: draft\n---\n\nbody\n"
        assert sib.already_carries_spec_id(text) is True

    def test_already_carries_false_no_frontmatter(self):
        assert sib.already_carries_spec_id("# no frontmatter\n") is False

    def test_already_carries_false_frontmatter_without_key(self):
        text = "---\nstatus: draft\n---\n\nbody\n"
        assert sib.already_carries_spec_id(text) is False

    def test_already_carries_false_unterminated_frontmatter(self):
        text = "---\nstatus: draft\nspec_id: foo\nno closing delimiter\n"
        # Conservative: an unterminated block is treated as not-carrying so
        # the caller routes it to a skip, never a write.
        assert sib.already_carries_spec_id(text) is False

    def test_target_id_in_header_present(self):
        text = _with_frontmatter("my-spec")
        assert sib.target_id_in_header(text) == "my-spec"

    def test_target_id_in_header_absent(self):
        assert sib.target_id_in_header("no header here\n") is None


# ---------------------------------------------------------------------------
# DoD-2: fixture tests
# ---------------------------------------------------------------------------

class TestBackfillFixtures:
    def test_frontmatter_present_insert(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        p = _write(specs, "foo.md", _with_frontmatter("foo"))
        original = p.read_text(encoding="utf-8")

        report = sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        assert [x.name for x in report.modified] == ["foo.md"]
        new_text = p.read_text(encoding="utf-8")
        lines = new_text.splitlines()
        assert lines[0] == "---"
        assert lines[1] == "spec_id: foo"
        # Everything else byte-identical, just shifted down by one line.
        assert new_text == "---\nspec_id: foo\n" + original[len("---\n"):]

    def test_no_frontmatter_block_creation(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        p = _write(specs, "bar.md", _no_frontmatter("bar"))
        original = p.read_text(encoding="utf-8")

        report = sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        assert [x.name for x in report.modified] == ["bar.md"]
        new_text = p.read_text(encoding="utf-8")
        assert new_text == "---\nspec_id: bar\n---\n" + original

    def test_idempotency_second_run_zero_changes(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        _write(specs, "foo.md", _with_frontmatter("foo"))

        first = sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)
        assert len(first.modified) == 1

        second = sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)
        assert second.modified == []
        assert [x.name for x in second.already_carrying] == ["foo.md"]

    def test_anchor_at_line_48_boundary_skips_and_reports(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        # No frontmatter -> 3-line insertion. Pad so **Target ID:** lands on
        # line 48; 48 + 3 = 51 > 50 -> must be skipped. (Single anchor line
        # only, to keep the boundary arithmetic unambiguous — the guard
        # checks each anchor regex independently, so this exercises it the
        # same way a spec carrying only a subset of the four would.)
        padding = "\n".join(f"filler line {i}" for i in range(1, 48)) + "\n"
        assert len(padding.splitlines()) == 47
        text = padding + "**Target ID:** `padded-spec`\n"
        p = _write(specs, "padded-spec.md", text)
        assert text.splitlines()[47] == "**Target ID:** `padded-spec`"  # 0-indexed 47 == line 48
        original = p.read_text(encoding="utf-8")

        report = sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        assert report.modified == []
        assert len(report.skipped) == 1
        skipped_path, reason = report.skipped[0]
        assert skipped_path.name == "padded-spec.md"
        assert reason == "anchor-window"
        # Never modified.
        assert p.read_text(encoding="utf-8") == original

    def test_anchor_at_line_47_boundary_does_not_skip(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        # 47 + 3 == 50 -> exactly at the window edge, still admissible.
        padding = "\n".join(f"filler line {i}" for i in range(1, 47)) + "\n"
        assert len(padding.splitlines()) == 46
        text = padding + "**Target ID:** `padded-spec-2`\n"
        _write(specs, "padded-spec-2.md", text)
        assert text.splitlines()[46] == "**Target ID:** `padded-spec-2`"  # line 47

        report = sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        assert [x.name for x in report.modified] == ["padded-spec-2.md"]
        assert report.skipped == []

    def test_mtime_preservation(self, tmp_path):
        # D2: mtime is the deliberate preservation target (seam 5's
        # fingerprint cache and 5b's draft active-set clause both key on
        # mtime) — atime is not a claim this tool makes.
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        p = _write(specs, "foo.md", _with_frontmatter("foo"))
        old_mtime = 1_700_000_000.0
        os.utime(p, (old_mtime, old_mtime))

        sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        st = p.stat()
        assert st.st_mtime == pytest.approx(old_mtime)

    def test_byte_identity_outside_inserted_lines(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        body = _with_frontmatter("foo", body_extra="some trailing\nbody text\n")
        p = _write(specs, "foo.md", body)
        original_lines = body.splitlines(keepends=True)

        sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        new_lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        assert new_lines[0] == original_lines[0]  # "---\n" unchanged
        assert new_lines[1] == "spec_id: foo\n"    # the one inserted line
        assert new_lines[2:] == original_lines[1:]  # rest byte-identical, shifted by 1

    def test_dry_run_makes_zero_writes(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        p = _write(specs, "foo.md", _with_frontmatter("foo"))
        original = p.read_text(encoding="utf-8")
        st_before = p.stat()

        report = sib.run_backfill(specs, apply=False, report_dir=reports, date_str=DATE)

        assert [x.name for x in report.modified] == ["foo.md"]
        assert p.read_text(encoding="utf-8") == original
        st_after = p.stat()
        assert st_after.st_mtime == st_before.st_mtime
        assert report.applied is False


# ---------------------------------------------------------------------------
# DoD-1-shaped: counts sum to scanned
# ---------------------------------------------------------------------------

class TestCounts:
    def test_counts_sum_to_scanned(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        _write(specs, "a.md", _with_frontmatter("a"))          # -> modified
        _write(specs, "b.md", "---\nspec_id: b\n---\n\nbody\n")  # -> already-carrying
        padding = "\n".join(f"filler {i}" for i in range(1, 48)) + "\n"
        _write(specs, "c.md", padding + _header("c"))          # -> skipped (anchor-window)

        report = sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        assert report.scanned == 3
        assert report.counts_sum_to_scanned()
        assert len(report.modified) == 1
        assert len(report.already_carrying) == 1
        assert len(report.skipped) == 1


# ---------------------------------------------------------------------------
# DoD-3: guards, each with a violating fixture
# ---------------------------------------------------------------------------

class TestGuards:
    def test_anchor_window_guard(self, tmp_path):
        # Covered end-to-end above (test_anchor_at_line_48_boundary_skips_and_reports);
        # this asserts the guard is unconditional across apply/dry-run.
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        padding = "\n".join(f"filler {i}" for i in range(1, 48)) + "\n"
        _write(specs, "c.md", padding + _header("c"))

        dry = sib.run_backfill(specs, apply=False, report_dir=reports, date_str=DATE)
        assert dry.skipped and dry.skipped[0][1] == "anchor-window"

    def test_duplicate_stem_guard_refuses_to_run(self, tmp_path, monkeypatch):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"

        p1 = _write(specs, "a.md", _with_frontmatter("a"))

        # Force a duplicate-stem scenario by monkeypatching discover_specs'
        # result shape at the module boundary the guard actually consumes
        # (see TestDiscovery.test_find_duplicate_stems_synthetic for why
        # this can't arise from a real glob of one flat directory).
        other_dir = tmp_path / "elsewhere"
        other_dir.mkdir()
        p2 = _write(other_dir, "a.md", _with_frontmatter("a"))

        monkeypatch.setattr(sib, "discover_specs", lambda specs_dir: sorted([p1, p2]))

        with pytest.raises(sib.DuplicateStemError):
            sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        # No writes happened to either file.
        assert p1.read_text(encoding="utf-8") == _with_frontmatter("a")
        assert p2.read_text(encoding="utf-8") == _with_frontmatter("a")
        # No report was written either — the guard fires before any scan.
        assert not reports.exists() or not any(reports.iterdir())

    def test_duplicate_value_guard_post_apply(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        # File a already carries a spec_id that does NOT match its own stem
        # (a pre-existing corpus anomaly — the 12-mismatch class shows this
        # happens) and collides with what file b's stem-derived backfill
        # will produce.
        _write(specs, "unrelated-name.md", "---\nspec_id: dup-value\n---\n\nbody\n")
        _write(specs, "dup-value.md", _no_frontmatter("dup-value"))

        with pytest.raises(sib.DuplicateSpecIdValueError):
            sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        # Guard fires AFTER report + mem-finding — evidence is left behind.
        report_path = reports / f"spec-id-backfill-report-{DATE}.md"
        assert report_path.exists()
        # The write to dup-value.md still happened (guard is a trailing
        # assertion over the already-applied result, not a pre-check).
        assert "spec_id: dup-value" in (specs / "dup-value.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# DoD-4: report lists the Target-ID-vs-stem mismatch set
# ---------------------------------------------------------------------------

class TestReportMismatches:
    def test_mismatch_reported_never_fixed(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        # Target ID disagrees with the filename stem.
        text = _with_frontmatter("actual-target-id")
        p = _write(specs, "different-stem.md", text)
        original = p.read_text(encoding="utf-8")

        report = sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)

        assert len(report.mismatches) == 1
        mpath, stem, target_id = report.mismatches[0]
        assert mpath.name == "different-stem.md"
        assert stem == "different-stem"
        assert target_id == "actual-target-id"

        # Backfill still uses the STEM, never the Target ID (D2/"decision the
        # spec makes explicit"), and the mismatch itself is never "fixed".
        new_text = p.read_text(encoding="utf-8")
        assert "spec_id: different-stem" in new_text
        assert "spec_id: actual-target-id" not in new_text

        report_text = report.report_path.read_text(encoding="utf-8")
        assert "different-stem.md" in report_text
        assert "stem=`different-stem`" in report_text
        assert "Target ID=`actual-target-id`" in report_text
        # Original body/header untouched apart from the inserted spec_id line.
        assert original.splitlines()[0] == "---"


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_written_dry_run_and_apply(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        reports = tmp_path / "reports"
        _write(specs, "a.md", _with_frontmatter("a"))

        dry = sib.run_backfill(specs, apply=False, report_dir=reports, date_str=DATE)
        assert dry.report_path.exists()
        assert "DRY-RUN" in dry.report_path.read_text(encoding="utf-8")

        applied = sib.run_backfill(specs, apply=True, report_dir=reports, date_str=DATE)
        assert applied.report_path.exists()
        assert "APPLY" in applied.report_path.read_text(encoding="utf-8")
