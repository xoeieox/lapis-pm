"""Integrity assertions for the lapis-pm-tick-loop-v0 calibration scaffold YAML.

Validates the sycophant-trap guard conditions from the bound spec:
- context.chubs is explicitly empty
- no vault_section path references StarHouse, incidents, or forbidden topics
- architecture_sketch contains none of the forbidden words
- scaffold parses cleanly via load_scaffold()
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

SCAFFOLD_PATH = Path("/srv/lapis/scout/sims/lapis-pm-tick-loop-v0.yaml")

# ---------------------------------------------------------------------------
# Forbidden words in architecture_sketch (case-insensitive)
# ---------------------------------------------------------------------------
_FORBIDDEN_WORDS = [
    "SEGV",
    "segfault",
    "drop_caches",
    "Forgejo silent",
    "Forgejo silently",
    "unless-stopped",
    "lost dispatch",
    "lost-dispatch",
    "RAM-marginal",
]

# ---------------------------------------------------------------------------
# Forbidden patterns in vault section paths / section names
# ---------------------------------------------------------------------------
_FORBIDDEN_PATH_PATTERNS = [
    "Lapis/StarHouse/*",
    "*incident_*",
    "*lost-dispatch*",
    "*resilience-quartet*",
]

_FORBIDDEN_PATH_KEYWORDS = [
    "incident",
    "segv",
    "forgejo silent",
    "lost-dispatch",
    "drop_caches",
]


@pytest.fixture(scope="module")
def scaffold():
    """Load and return the calibration scaffold once per module."""
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Calibration scaffold not found: {SCAFFOLD_PATH}")
    from lapis_pm.scout.scaffold import load_scaffold
    return load_scaffold(SCAFFOLD_PATH)


# ---------------------------------------------------------------------------
# Test: parses cleanly via load_scaffold
# ---------------------------------------------------------------------------

def test_scaffold_loads(scaffold) -> None:
    assert scaffold.spec_id == "lapis-pm-tick-loop-v0"
    assert scaffold.spec_version == "v0"
    assert scaffold.description


# ---------------------------------------------------------------------------
# Test: chubs explicitly empty
# ---------------------------------------------------------------------------

def test_chubs_empty(scaffold) -> None:
    assert scaffold.context.chubs == [], (
        f"context.chubs must be [] for calibration integrity; got: {scaffold.context.chubs}"
    )


# ---------------------------------------------------------------------------
# Test: no forbidden vault paths or section names
# ---------------------------------------------------------------------------

def test_vault_sections_no_forbidden_paths(scaffold) -> None:
    for vs in scaffold.context.vault_sections:
        path_lower = vs.path.lower()

        # Glob pattern match
        for pattern in _FORBIDDEN_PATH_PATTERNS:
            assert not fnmatch.fnmatch(vs.path, pattern), (
                f"Vault path {vs.path!r} matches forbidden pattern {pattern!r}"
            )

        # Keyword match (case-insensitive)
        for kw in _FORBIDDEN_PATH_KEYWORDS:
            assert kw.lower() not in path_lower, (
                f"Vault path {vs.path!r} contains forbidden keyword {kw!r}"
            )

        # Also check section names
        for section in vs.sections:
            section_lower = section.lower()
            for kw in _FORBIDDEN_PATH_KEYWORDS:
                assert kw.lower() not in section_lower, (
                    f"Vault section {section!r} (in {vs.path!r}) contains forbidden keyword {kw!r}"
                )


# ---------------------------------------------------------------------------
# Test: architecture_sketch contains no forbidden words
# ---------------------------------------------------------------------------

def test_architecture_sketch_no_forbidden_words(scaffold) -> None:
    sketch = scaffold.static_scaffold.architecture_sketch
    for word in _FORBIDDEN_WORDS:
        assert word.lower() not in sketch.lower(), (
            f"architecture_sketch contains forbidden word: {word!r}"
        )


# ---------------------------------------------------------------------------
# Test: matrix produces expected 72-run count
# ---------------------------------------------------------------------------

def test_matrix_run_count(scaffold) -> None:
    """4 optional × 2 severity × 3 load × 3 runs_per_cell = 72."""
    cells = scaffold.cell_params()
    total_runs = len(cells) * scaffold.matrix.runs_per_cell
    assert total_runs == 72, (
        f"Expected 72 total runs (4×2×3×3), got {total_runs} "
        f"({len(cells)} cells × {scaffold.matrix.runs_per_cell} runs/cell)"
    )
