"""lapis_pm.spec_id_backfill — one-shot corpus migration (spec-corpus-id-backfill-v0).

Backfills `spec_id: <filename-stem>` into the YAML frontmatter of every
`/srv/lapis/planning/specs/*.md` spec that lacks it — top-level, non-recursive,
exactly the population seam 5's `discover_specs()` scans (`glob("*.md")`).

The migration is mechanically settled: in 100% of specs that already carry
`spec_id:`, it equals the filename stem (both come from the same variable in
`bundle_triage.cook_item_to_spec()`). Backfilling `spec_id: <stem>` therefore
changes no key value anywhere — it upgrades `key_source` from
`filename_stem` to `spec_id` for seam 5's `discover_specs()`
(`scripts/spec_edges.py:141-148` on conductor PR #877) and satisfies
`bundle_autodispatch._parse_bundle_frontmatter()`
(`lapis_pm/bundle_autodispatch.py:79-81`), which raises on a missing
`spec_id`.

Dry-run by default; `apply=True` writes. Byte-conservative: the only bytes
that ever change are the inserted `spec_id: <stem>` line (plus, for specs
with no frontmatter at all, two `---` delimiter lines around it) — no Target
ID edit, no body edit, no rename, no reformat. mtime is restored after write
(D2 — deliberate: bumping mtimes corpus-wide would flood seam 5b's draft
mtime-window active-set clause; the cache staleness this trades for is
cosmetic and self-healing on each file's next natural edit).

Guards (checked before any write — D3):
  - Anchor-window: any spec whose `**Target ID:**` / `**Repo:**` /
    `**Authority:**` / `**Consumer:**` header lines would be pushed past line
    50 by the insertion (the window `_consumer_criterion_check` and seam 2's
    firm shape-check both read) is skipped and reported, never modified.
  - Duplicate-stem: refuse to run if the discovered file list has two paths
    sharing a stem (impossible today via a single flat glob — defensive).
  - Duplicate-value: post-apply, assert `spec_id` values in scope are unique.

Report: both dry-run and apply write
`/srv/lapis/planning/spec-id-backfill-report-<date>.md` — the migration's real
product alongside the edits (D4) — listing modified / already-carrying /
skipped files, and the Target-ID-vs-stem mismatch set (a pre-existing defect
class, reported for Erah's ruling, never silently cemented or "fixed" here —
touching them would touch bound targets).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from agents_core.room_paths import room_path
from lapis_pm.spec_review import _TID_RE, _REPO_RE, _AUTHORITY_TEXT_RE, _CONSUMER_RE

# Same window the C7 admission check and seam 2's firm shape-check read
# (spec_review.py's `first_50 = "\n".join(text.splitlines()[:50])` pattern).
_ANCHOR_WINDOW = 50

_FRONTMATTER_DELIM = "---"
_SPEC_ID_KEY_RE = re.compile(r"^spec_id:\s*\S+\s*$", re.MULTILINE)

_ANCHOR_RES = (_TID_RE, _REPO_RE, _AUTHORITY_TEXT_RE, _CONSUMER_RE)


class DuplicateStemError(ValueError):
    """Raised by the duplicate-stem guard — refuses to run, no writes made."""


class DuplicateSpecIdValueError(ValueError):
    """Raised by the post-apply duplicate-value guard."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_specs(specs_dir: Path) -> list[Path]:
    """Top-level, non-recursive discovery — mirrors seam 5's `glob("*.md")`.

    Excludes superseded/, cooked/, archived/ (all subdirectories) by
    construction: `Path.glob("*.md")` never descends.
    """
    return sorted(specs_dir.glob("*.md"))


def find_duplicate_stems(files: list[Path]) -> list[str]:
    """Return stems that appear more than once in `files`.

    Unreachable via `discover_specs()` on a single flat directory (a
    filename bijects with its stem within one directory listing) — this
    guards the general case, in case scope ever widens.
    """
    seen: set[str] = set()
    dupes: set[str] = set()
    for f in files:
        stem = f.stem
        if stem in seen:
            dupes.add(stem)
        seen.add(stem)
    return sorted(dupes)


# ---------------------------------------------------------------------------
# Frontmatter parsing — byte-conservative
# ---------------------------------------------------------------------------

def _detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _frontmatter_close_index(lines: list[str]) -> int | None:
    """`lines[0]` must already be confirmed to be the opening `---`. Return
    the index of the closing `---` line, or None if unterminated."""
    for i in range(1, len(lines)):
        if lines[i].rstrip("\r\n") == _FRONTMATTER_DELIM:
            return i
    return None


def _frontmatter_bounds(text: str) -> tuple[bool, list[str], int | None]:
    """Return (has_frontmatter, lines-with-line-endings, close_index).

    has_frontmatter is True iff line 1 is exactly `---`. close_index is the
    index of the matching closing `---`, or None if line 1 opens a block
    that never closes (malformed — treated conservatively, see
    `already_carries_spec_id`)."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != _FRONTMATTER_DELIM:
        return False, lines, None
    return True, lines, _frontmatter_close_index(lines)


def already_carries_spec_id(text: str) -> bool:
    """True iff a `spec_id:` key already exists inside the file's YAML
    frontmatter block. A malformed (unterminated) block is treated
    conservatively as NOT carrying it — the caller's anchor/frontmatter
    handling then routes it to a skip, never a write."""
    has_fm, lines, close_idx = _frontmatter_bounds(text)
    if not has_fm or close_idx is None:
        return False
    block = "".join(lines[1:close_idx])
    return bool(_SPEC_ID_KEY_RE.search(block))


def target_id_in_header(text: str) -> str | None:
    """Return the `**Target ID:**` value from the first 50 lines, or None."""
    first_window = "\n".join(text.splitlines()[:_ANCHOR_WINDOW])
    m = _TID_RE.search(first_window)
    return m.group(1) if m else None


def _anchor_line_numbers(text: str) -> list[int]:
    """1-indexed line numbers (within the first `_ANCHOR_WINDOW` lines) of
    whichever of the four anchored header lines are present."""
    lines = text.splitlines()[:_ANCHOR_WINDOW]
    numbers = []
    for idx, line in enumerate(lines, start=1):
        if any(rx.match(line) for rx in _ANCHOR_RES):
            numbers.append(idx)
    return numbers


def _insert_spec_id(text: str, stem: str, *, has_fm: bool, lines: list[str]) -> str:
    """Insert `spec_id: <stem>` as the frontmatter's first key (has_fm=True)
    or create a minimal 3-line block above line 1 (has_fm=False). Everything
    else is byte-identical to the input."""
    if has_fm:
        opening = lines[0]
        nl = "\r\n" if opening.endswith("\r\n") else "\n"
        return opening + f"spec_id: {stem}{nl}" + "".join(lines[1:])
    nl = _detect_newline(text)
    return f"---{nl}spec_id: {stem}{nl}---{nl}" + text


# ---------------------------------------------------------------------------
# Per-file outcome + report
# ---------------------------------------------------------------------------

@dataclass
class BackfillReport:
    modified: list[Path] = field(default_factory=list)
    already_carrying: list[Path] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)  # (path, reason)
    mismatches: list[tuple[Path, str, str]] = field(default_factory=list)  # (path, stem, target_id)
    applied: bool = False
    scanned: int = 0
    report_path: Path | None = None

    def counts_sum_to_scanned(self) -> bool:
        return (
            len(self.modified) + len(self.already_carrying) + len(self.skipped)
            == self.scanned
        )


def _render_report(report: BackfillReport, specs_dir: Path, date_str: str) -> str:
    mode = "APPLY" if report.applied else "DRY-RUN"
    lines = [
        f"# spec-id-backfill report — {date_str} ({mode})",
        "",
        f"Scope: `{specs_dir}` (top-level `*.md`, non-recursive).",
        "",
        f"- scanned: {report.scanned}",
        f"- modified: {len(report.modified)}"
        + (" (candidates — dry-run, zero writes)" if not report.applied else ""),
        f"- already-carrying spec_id: {len(report.already_carrying)}",
        f"- skipped: {len(report.skipped)}",
        f"- Target-ID-vs-stem mismatches: {len(report.mismatches)}",
        "",
    ]

    lines.append("## Skipped" + (" (none)" if not report.skipped else ""))
    for path, reason in report.skipped:
        lines.append(f"- `{path.name}` — {reason}")
    lines.append("")

    lines.append(
        "## Target-ID-vs-stem mismatches"
        + (" (none)" if not report.mismatches else " — reported for a ruling, not fixed here")
    )
    for path, stem, target_id in report.mismatches:
        lines.append(f"- `{path.name}`: stem=`{stem}` Target ID=`{target_id}`")
    lines.append("")

    lines.append(f"## {'Modified' if report.applied else 'Candidates for modification'}")
    for path in report.modified:
        lines.append(f"- `{path.name}`")
    lines.append("")

    lines.append("## Already carrying spec_id")
    for path in report.already_carrying:
        lines.append(f"- `{path.name}`")
    lines.append("")

    return "\n".join(lines) + "\n"


def _write_report(report: BackfillReport, report_dir: Path, date_str: str, specs_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"spec-id-backfill-report-{date_str}.md"
    report_path.write_text(_render_report(report, specs_dir, date_str), encoding="utf-8")
    return report_path


def _write_mem_finding(report: BackfillReport, date_str: str) -> None:
    """Apply-only: record the counts in mem under `finding/`."""
    from lapis_pm import node_identity

    content = (
        f"spec-id-backfill apply run {date_str}: "
        f"scanned={report.scanned} modified={len(report.modified)} "
        f"already_carrying={len(report.already_carrying)} "
        f"skipped={len(report.skipped)} mismatches={len(report.mismatches)}"
    )
    node_identity.writable_store().set(
        f"finding/spec-corpus-id-backfill-v0-apply-{date_str}",
        content,
        tags=["lapis-pm", "spec-corpus-id-backfill-v0"],
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_backfill(
    specs_dir: Path | None = None,
    *,
    apply: bool = False,
    report_dir: Path | None = None,
    date_str: str,
) -> BackfillReport:
    """Run the migration over `specs_dir` (default: `room_path("planning.specs")`).

    Dry-run (apply=False, the default): computes and reports the same
    candidate/skip/mismatch sets but writes nothing to any spec file.
    Idempotent: a file already carrying `spec_id:` is left untouched, so a
    second `apply=True` run reports zero modified candidates.
    """
    specs_dir = specs_dir if specs_dir is not None else room_path("planning.specs")
    report_dir = report_dir if report_dir is not None else room_path("planning")

    files = discover_specs(specs_dir)

    dup_stems = find_duplicate_stems(files)
    if dup_stems:
        raise DuplicateStemError(
            f"refusing to run: duplicate stem(s) in scope: {dup_stems}"
        )

    report = BackfillReport(scanned=len(files))

    for path in files:
        stem = path.stem
        try:
            # Snapshot pre-read mtime so the read below never leaks a
            # touched mtime into what os.utime() restores after write.
            pre_stat = path.stat()
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            report.skipped.append((path, f"read-error: {e}"))
            continue

        target_id = target_id_in_header(text)
        if target_id is not None and target_id != stem:
            report.mismatches.append((path, stem, target_id))

        if already_carries_spec_id(text):
            report.already_carrying.append(path)
            continue

        has_fm, lines, close_idx = _frontmatter_bounds(text)
        if has_fm and close_idx is None:
            report.skipped.append((path, "unterminated-frontmatter"))
            continue

        insert_lines = 1 if has_fm else 3
        anchor_lines = _anchor_line_numbers(text)
        if any((n + insert_lines) > _ANCHOR_WINDOW for n in anchor_lines):
            report.skipped.append((path, "anchor-window"))
            continue

        new_text = _insert_spec_id(text, stem, has_fm=has_fm, lines=lines)

        if apply:
            path.write_text(new_text, encoding="utf-8")
            # D2: mtime restored, deliberately — seam 5's fingerprint cache
            # and 5b's draft mtime-window active-set clause both key on
            # mtime, not atime (a later scan of `files` for the post-apply
            # duplicate-value guard reads every file again regardless, so
            # atime is not a preservable quantity here even in principle).
            os.utime(path, (path.stat().st_atime, pre_stat.st_mtime))

        report.modified.append(path)

    report.applied = apply
    report.report_path = _write_report(report, report_dir, date_str, specs_dir)

    if apply and report.modified:
        _write_mem_finding(report, date_str)

    if apply:
        # Trailing safety assertion, checked AFTER the report/mem-finding are
        # already written — a guard failure should leave evidence behind for
        # triage, not swallow the run's output (D3: "asserted anyway").
        dup_values = _duplicate_spec_id_values(files)
        if dup_values:
            raise DuplicateSpecIdValueError(
                f"post-apply guard failed: duplicate spec_id value(s): {dup_values}"
            )

    return report


def _duplicate_spec_id_values(files: list[Path]) -> list[str]:
    """Post-apply guard: re-read `spec_id:` values across `files` and return
    any that appear more than once. Follows from the stem guard (values are
    stems, stems are unique per directory) — asserted anyway (D3)."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        has_fm, lines, close_idx = _frontmatter_bounds(text)
        if not has_fm or close_idx is None:
            continue
        block = "".join(lines[1:close_idx])
        m = re.search(r"^spec_id:\s*(\S+)\s*$", block, re.MULTILINE)
        if not m:
            continue
        value = m.group(1)
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return sorted(dupes)
