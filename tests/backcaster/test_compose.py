"""Tests: compose stage — epistemic_caution, concentration_warning (Tests 9, 10, 11)."""
from __future__ import annotations

from lapis_pm.backcaster.compose import (
    compute_concentration_warning,
    compute_epistemic_caution,
)
from lapis_pm.backcaster.schema import BackcasterCitation, Component, Gap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gap(pid: str, unsourced: bool) -> Gap:
    return Gap(
        precondition_id=pid,
        what_exists="x",
        what_missing="y",
        what_miswired="nothing identified",
        citations=[] if unsourced else [BackcasterCitation(type="corpus", ref="foo.md")],
        unsourced=unsourced,
    )


def _make_comp(i: int, cat: str, subtype: str | None = None) -> Component:
    return Component(
        id=f"c-{i}",
        gap_precondition_id="p-01",
        description="test",
        category=cat,
        subtype=subtype,
        effort_estimate="medium",
        reversibility="medium",
    )


# ---------------------------------------------------------------------------
# Epistemic caution (Test 9)
# ---------------------------------------------------------------------------

def test_epistemic_caution_high():
    """60% unsourced → high."""
    gaps = [_make_gap(f"p-{i}", unsourced=(i < 6)) for i in range(10)]
    assert compute_epistemic_caution(gaps) == "high"


def test_epistemic_caution_medium():
    """30% unsourced → medium."""
    gaps = [_make_gap(f"p-{i}", unsourced=(i < 3)) for i in range(10)]
    assert compute_epistemic_caution(gaps) == "medium"


def test_epistemic_caution_low():
    """10% unsourced → low."""
    gaps = [_make_gap(f"p-{i}", unsourced=(i == 0)) for i in range(10)]
    assert compute_epistemic_caution(gaps) == "low"


def test_epistemic_caution_all_sourced():
    gaps = [_make_gap(f"p-{i}", unsourced=False) for i in range(5)]
    assert compute_epistemic_caution(gaps) == "low"


def test_epistemic_caution_empty_gaps():
    assert compute_epistemic_caution([]) == "high"


# ---------------------------------------------------------------------------
# Concentration warning top-level (Test 10)
# ---------------------------------------------------------------------------

def test_concentration_warning_software_dominant():
    """8/10 software → software-dominant."""
    comps = [_make_comp(i, "software") for i in range(8)]
    comps += [_make_comp(8, "research"), _make_comp(9, "policy")]
    warnings = compute_concentration_warning(comps)
    assert "software-dominant" in warnings


def test_concentration_warning_no_concentration():
    """4/10 software + 4/10 financial.extractive + 2 others → no warning."""
    comps = (
        [_make_comp(i, "software") for i in range(4)]
        + [_make_comp(i + 4, "financial", subtype="extractive") for i in range(4)]
        + [_make_comp(8, "research"), _make_comp(9, "policy")]
    )
    warnings = compute_concentration_warning(comps)
    assert warnings == []


# ---------------------------------------------------------------------------
# Concentration warning financial subtype (Test 11)
# ---------------------------------------------------------------------------

def test_concentration_warning_financial_extractive():
    """8/10 financial.extractive → both financial-dominant and financial.extractive-dominant."""
    comps = [_make_comp(i, "financial", subtype="extractive") for i in range(8)]
    comps += [_make_comp(8, "research"), _make_comp(9, "policy")]
    warnings = compute_concentration_warning(comps)
    assert "financial-dominant" in warnings
    assert "financial.extractive-dominant" in warnings


def test_concentration_warning_is_list():
    """concentration_warning is always a list."""
    warnings = compute_concentration_warning([])
    assert isinstance(warnings, list)
