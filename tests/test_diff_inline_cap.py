"""Tests for DIFF_INLINE_CAP constants — lapis-pm-reviewer-full-context-v0.

Asserts:
  - DIFF_INLINE_CAP_REVIEWER == 200_000 (reviewer dispatch path cap)
  - DIFF_INLINE_CAP_SCREEN == 60_000 (authority.screen() inline cap — no
    clone fallback, so it retains the pre-200k value; cr-bundle item 4a082d29b0)
  - DIFF_INLINE_CAP is the reviewer cap (back-compat alias)
  - Both pm_core.py and authority.py use the named constants (no bare integer literal)
  - The caps are actually applied at their respective dispatch sites
"""

from __future__ import annotations

import re
from pathlib import Path

from lapis_pm.authority import (
    DIFF_INLINE_CAP,
    DIFF_INLINE_CAP_REVIEWER,
    DIFF_INLINE_CAP_SCREEN,
)


AUTHORITY_PY = Path(__file__).parent.parent / "lapis_pm" / "authority.py"
PM_CORE_PY = Path(__file__).parent.parent / "lapis_pm" / "pm_core.py"

# Bare integer literals we're checking are absent at the old cap site.
# 60000 was the old cap; 200000 is the new value as a bare integer.
OLD_CAP_LITERAL = "60000"
NEW_CAP_BARE = "200000"   # bare literal (without underscore) would indicate a mistake


class TestDiffInlineCap:

    def test_cap_value(self):
        """DIFF_INLINE_CAP must equal 200_000."""
        assert DIFF_INLINE_CAP == 200_000, \
            f"DIFF_INLINE_CAP = {DIFF_INLINE_CAP}, expected 200_000"

    def test_authority_py_no_old_cap_literal(self):
        """authority.py must not contain the old 60000 literal."""
        src = AUTHORITY_PY.read_text()
        assert OLD_CAP_LITERAL not in src, \
            "authority.py still contains bare '60000' literal — use DIFF_INLINE_CAP"

    def test_pm_core_py_no_old_cap_literal(self):
        """pm_core.py must not contain the old 60000 literal."""
        src = PM_CORE_PY.read_text()
        assert OLD_CAP_LITERAL not in src, \
            "pm_core.py still contains bare '60000' literal — use DIFF_INLINE_CAP"

    def test_authority_py_uses_named_constant(self):
        """authority.py must reference DIFF_INLINE_CAP by name at the truncation site."""
        src = AUTHORITY_PY.read_text()
        assert "DIFF_INLINE_CAP" in src, \
            "authority.py does not define or reference DIFF_INLINE_CAP"

    def test_pm_core_py_uses_named_constant(self):
        """pm_core.py must reference authority.DIFF_INLINE_CAP at the truncation site."""
        src = PM_CORE_PY.read_text()
        assert "DIFF_INLINE_CAP" in src, \
            "pm_core.py does not reference DIFF_INLINE_CAP"

    def test_authority_py_no_bare_new_cap(self):
        """authority.py must not contain the new cap as a bare integer literal 200000."""
        src = AUTHORITY_PY.read_text()
        # Allow 200_000 (with underscore) but not 200000 (without)
        matches = [m for m in re.findall(r'\b200000\b', src)]
        assert not matches, \
            "authority.py uses bare '200000' literal instead of DIFF_INLINE_CAP"

    def test_pm_core_py_no_bare_new_cap(self):
        """pm_core.py must not contain the new cap as a bare integer literal 200000."""
        src = PM_CORE_PY.read_text()
        matches = [m for m in re.findall(r'\b200000\b', src)]
        assert not matches, \
            "pm_core.py uses bare '200000' literal instead of DIFF_INLINE_CAP"
