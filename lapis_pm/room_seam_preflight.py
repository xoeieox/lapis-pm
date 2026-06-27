"""DoD-9/10: machine-readable shell-seam ledger for room-paths-sweep-lapis-pm-v0.

This module serves two purposes:
1. Documents the true-shell /room literals that are deliberately NOT swept by this
   python-only unit (DoD-9 — visible CI warning, non-fatal in this PR).
2. Provides the preflight-contract function that the Phase-3 ROOM_ROOT flip preflight
   consumes (DoD-10 — teeth at the flip, not in this PR).

Asymmetry rationale: shell files are correctly still hardcoded /room during the
python-sweep phase because ROOM_ROOT is unset → /room, so the divergent state is
safe ONLY until the flip. The flip preflight MUST call verify_shell_seams_clean()
and gate on its return value — at that point, any remaining /room literal in these
files is a hard blocker.

Anti-ledger-fallacy: verify_shell_seams_clean() does a LIVE GREP of the shell
artifacts, not just a documentation check. The preflight cannot pass if the files
still contain /room path literals, even if this ledger entry says they were deferred.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# DoD-9 ledger — true-shell items deferred to the cutover shell-seam sub-unit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShellSeamEntry:
    """One shell-seam site deferred to the cutover sub-unit."""
    file: str        # repo-relative path
    line: int        # line number at time of this PR
    literal: str     # the hardcoded /room literal
    reason: str      # why it is non-scope for this PR


DEFERRED_SHELL_SEAMS: list[ShellSeamEntry] = [
    ShellSeamEntry(
        file="deploy/sync.sh",
        line=196,
        literal='DEPLOY_LOG="/srv/lapis/lapis-state/lapis-pm-deploy-log.md"',
        reason="true-shell; covered by room_paths.sh sourcing in the cutover shell-seam sub-unit",
    ),
    ShellSeamEntry(
        file="deploy/daemon-sync.sh",
        line=15,
        literal='DEPLOY_LOG="/srv/lapis/lapis-state/lapis-pm-deploy-log.md"',
        reason="true-shell; covered by room_paths.sh sourcing in the cutover shell-seam sub-unit",
    ),
    ShellSeamEntry(
        file="lapis_pm/smoke.sh",
        line=0,  # 45 hits across the file; line=0 means whole-file
        literal="/srv/lapis/... (45 occurrences)",
        reason="true-shell smoke test harness; same cutover sub-unit",
    ),
    ShellSeamEntry(
        file="lapis_pm/smoke_fixtures/spec_review/facets_dispatch_smoke.sh",
        line=0,
        literal="/srv/lapis/... (1 occurrence)",
        reason="true-shell smoke fixture; same cutover sub-unit",
    ),
]


def emit_dod9_warning() -> str:
    """Return the DoD-9 CI warning string (machine-readable, non-fatal)."""
    lines = [
        "[room-paths-sweep-lapis-pm-v0] DoD-9: deferred true-shell seams (NON-FATAL in this PR):",
    ]
    for entry in DEFERRED_SHELL_SEAMS:
        lines.append(
            f"  SHELL_SEAM_DEFERRED file={entry.file} line={entry.line} "
            f'literal="{entry.literal}" reason="{entry.reason}"'
        )
    lines.append(
        "[room-paths-sweep-lapis-pm-v0] DoD-10: these entries are registered in the "
        "Phase-3 ROOM_ROOT flip preflight. The flip CANNOT proceed while any of "
        "these files contain /room path literals (live-grep verified, not just ledger check)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DoD-10 preflight contract — teeth at the flip
# ---------------------------------------------------------------------------

def verify_shell_seams_clean(repo_root: Path) -> tuple[bool, list[str]]:
    """Live-grep the deferred shell files for remaining /room path literals.

    Called by the Phase-3 ROOM_ROOT flip preflight. Returns (clean, violations).
    clean=True means the flip may proceed for the shell seams.
    clean=False means at least one deferred file still has /room literals — BLOCK THE FLIP.

    This is a LIVE GREP, not a documentation check. The ledger cannot drift from reality.
    """
    path_literal_pattern = re.compile(r'["\']\/room\/')
    violations: list[str] = []

    whole_file_entries = {e.file for e in DEFERRED_SHELL_SEAMS if e.line == 0}
    single_line_entries = {e.file: e.line for e in DEFERRED_SHELL_SEAMS if e.line != 0}

    all_files = whole_file_entries | set(single_line_entries.keys())

    for rel_path in sorted(all_files):
        sh_file = repo_root / rel_path
        if not sh_file.exists():
            continue
        for lineno, line in enumerate(sh_file.read_text(encoding="utf-8").splitlines(), 1):
            if path_literal_pattern.search(line):
                violations.append(f"{rel_path}:{lineno}: {line.strip()}")

    return (len(violations) == 0), violations


def preflight_assert_shell_seams_clean(repo_root: Path) -> None:
    """Assert all deferred shell seams are clean. Raises RuntimeError if not.

    This is the BLOCKING GATE the Phase-3 flip preflight calls.
    If it raises, the ROOM_ROOT flip must not proceed.
    """
    clean, violations = verify_shell_seams_clean(repo_root)
    if not clean:
        raise RuntimeError(
            "ROOM_ROOT flip blocked: lapis-pm shell seams still contain /room literals. "
            "Complete the cutover shell-seam sub-unit first.\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )
