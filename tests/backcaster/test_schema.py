"""Tests: Pydantic schema validation (Tests 6, 8 from spec)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from lapis_pm.backcaster.schema import (
    CATEGORIES,
    FINANCIAL_SUBTYPES,
    BackcasterCitation,
    Component,
    Gap,
    Histogram,
    Precondition,
)


# ---------------------------------------------------------------------------
# BackcasterCitation
# ---------------------------------------------------------------------------

def test_citation_valid():
    c = BackcasterCitation(type="corpus", ref="library/civic-theory/ostrom.md")
    assert c.type == "corpus"
    assert c.to_dict() == {"type": "corpus", "ref": "library/civic-theory/ostrom.md"}


def test_citation_invalid_type():
    with pytest.raises(ValidationError):
        BackcasterCitation(type="unknown", ref="foo")


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------

def test_precondition_valid():
    p = Precondition(
        id="psychological-01",
        axis="psychological",
        statement="People feel safe.",
        implicit_dependencies=["trust exists"],
    )
    assert p.axis == "psychological"


def test_precondition_invalid_axis():
    with pytest.raises(ValidationError):
        Precondition(id="x-01", axis="spiritual", statement="foo")


# ---------------------------------------------------------------------------
# Gap
# ---------------------------------------------------------------------------

def test_gap_empty_citations_sets_unsourced():
    g = Gap(
        precondition_id="p-01",
        what_exists="nothing",
        what_missing="a lot",
        what_miswired="nothing identified",
        citations=[],
        unsourced=False,  # should be auto-corrected to True
    )
    assert g.unsourced is True


def test_gap_with_citations_not_forced_unsourced():
    g = Gap(
        precondition_id="p-01",
        what_exists="something",
        what_missing="something else",
        what_miswired="nothing identified",
        citations=[BackcasterCitation(type="corpus", ref="foo.md")],
        unsourced=False,
    )
    assert g.unsourced is False


def test_gap_to_dict_shape():
    g = Gap(
        precondition_id="p-01",
        what_exists="x",
        what_missing="y",
        what_miswired="z",
        citations=[BackcasterCitation(type="mem", ref="architecture/foo")],
    )
    d = g.to_dict()
    assert d["citations"][0] == {"type": "mem", "ref": "architecture/foo"}
    assert "unsourced" in d


# ---------------------------------------------------------------------------
# Component — category cardinality (Test 8)
# ---------------------------------------------------------------------------

def test_component_valid_all_categories():
    """Exactly 7 categories must be accepted."""
    valid_cats = [
        ("software", None),
        ("policy", None),
        ("community-formation", None),
        ("research", None),
        ("infrastructure", None),
        ("cultural-shift", None),
        ("financial", "extractive"),
    ]
    for cat, subtype in valid_cats:
        c = Component(
            id="c-01",
            gap_precondition_id="p-01",
            description="test",
            category=cat,
            subtype=subtype,
            effort_estimate="small",
            reversibility="high",
        )
        assert c.category == cat


def test_component_invalid_category_rejected():
    """Any category outside the 7 must be rejected."""
    with pytest.raises(ValidationError):
        Component(
            id="c-01",
            gap_precondition_id="p-01",
            description="test",
            category="relational",
            effort_estimate="small",
            reversibility="high",
        )


def test_component_financial_requires_valid_subtype():
    """Financial without valid subtype is a schema error (Test 8)."""
    with pytest.raises(ValidationError):
        Component(
            id="c-01",
            gap_precondition_id="p-01",
            description="test",
            category="financial",
            subtype=None,  # missing
            effort_estimate="small",
            reversibility="high",
        )


def test_component_financial_invalid_subtype():
    with pytest.raises(ValidationError):
        Component(
            id="c-01",
            gap_precondition_id="p-01",
            description="test",
            category="financial",
            subtype="vampire",  # not in FINANCIAL_SUBTYPES
            effort_estimate="small",
            reversibility="high",
        )


def test_component_financial_all_three_subtypes():
    """Exactly 3 financial subtypes (Test 8)."""
    for st in FINANCIAL_SUBTYPES:
        c = Component(
            id=f"c-{st}",
            gap_precondition_id="p-01",
            description="test",
            category="financial",
            subtype=st,
            effort_estimate="medium",
            reversibility="medium",
        )
        assert c.subtype == st


def test_component_fallback_fields():
    c = Component(
        id="c-fallback",
        gap_precondition_id="p-01",
        description="fallback component",
        category="research",
        effort_estimate="medium",
        reversibility="medium",
        unsourced=True,
        fallback=True,
    )
    d = c.to_dict()
    assert d["fallback"] is True
    assert d["unsourced"] is True


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def test_histogram_always_emits_all_categories():
    """Histogram must emit zero-count entries for all 7 categories (Invariant 4)."""
    hist = Histogram.from_components([])
    d = hist.to_dict()
    from lapis_pm.backcaster.schema import CATEGORIES, FINANCIAL_SUBTYPES
    for cat in CATEGORIES:
        assert cat in d
    for st in FINANCIAL_SUBTYPES:
        assert f"financial.{st}" in d
    assert "research.fallback" in d


def test_histogram_counts_correct():
    comps = [
        Component(
            id=f"c-{i}", gap_precondition_id="p-01", description="test",
            category="software", effort_estimate="small", reversibility="high",
        )
        for i in range(3)
    ] + [
        Component(
            id="c-fin", gap_precondition_id="p-01", description="test",
            category="financial", subtype="extractive",
            effort_estimate="medium", reversibility="medium",
        ),
        Component(
            id="c-res-fb", gap_precondition_id="p-01", description="fallback",
            category="research", effort_estimate="medium", reversibility="medium",
            fallback=True, unsourced=True,
        ),
    ]
    hist = Histogram.from_components(comps)
    d = hist.to_dict()
    assert d["software"] == 3
    assert d["financial"] == 1
    assert d["financial.extractive"] == 1
    assert d["financial.distributive"] == 0
    assert d["research"] == 1
    assert d["research.fallback"] == 1
