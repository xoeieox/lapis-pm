"""Backcaster Pydantic models.

Precondition, Gap, Component, RoadmapRun — type-safe shapes for pipeline I/O
and downstream consumption.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Axis enum
# ---------------------------------------------------------------------------

AXES = (
    "psychological",
    "material",
    "infrastructural",
    "governance",
    "social-norm",
    "economic",
)

# ---------------------------------------------------------------------------
# Category enum
# ---------------------------------------------------------------------------

CATEGORIES = (
    "software",
    "policy",
    "community-formation",
    "research",
    "infrastructure",
    "cultural-shift",
    "financial",
)

FINANCIAL_SUBTYPES = ("extractive", "distributive", "neutral")


# ---------------------------------------------------------------------------
# Citation shape (parallel to archetypes_core.corroboration.Citation but
# simpler — backcaster uses {type, ref} dicts per spec)
# ---------------------------------------------------------------------------

CitationType = Literal["corpus", "mem", "scenario"]


class BackcasterCitation(BaseModel):
    type: CitationType
    ref: str

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "ref": self.ref}


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------

class Precondition(BaseModel):
    id: str
    axis: str
    statement: str
    implicit_dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_axis(self) -> "Precondition":
        if self.axis not in AXES:
            raise ValueError(f"axis {self.axis!r} not in {AXES}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "axis": self.axis,
            "statement": self.statement,
            "implicit_dependencies": self.implicit_dependencies,
        }


# ---------------------------------------------------------------------------
# Gap
# ---------------------------------------------------------------------------

class Gap(BaseModel):
    precondition_id: str
    what_exists: str
    what_missing: str
    what_miswired: str
    citations: list[BackcasterCitation] = Field(default_factory=list)
    unsourced: bool = False

    @model_validator(mode="after")
    def _check_unsourced(self) -> "Gap":
        # Empty citations MUST have unsourced=True
        if not self.citations and not self.unsourced:
            self.unsourced = True
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "precondition_id": self.precondition_id,
            "what_exists": self.what_exists,
            "what_missing": self.what_missing,
            "what_miswired": self.what_miswired,
            "citations": [c.to_dict() for c in self.citations],
            "unsourced": self.unsourced,
        }


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class Component(BaseModel):
    id: str
    gap_precondition_id: str
    description: str
    category: str  # validated below
    subtype: Optional[str] = None  # required when category == "financial"
    effort_estimate: Literal["small", "medium", "large"]
    reversibility: Literal["high", "medium", "low"]
    dependencies: list[str] = Field(default_factory=list)
    unsourced: bool = False
    fallback: bool = False

    @model_validator(mode="after")
    def _check_category(self) -> "Component":
        if self.category not in CATEGORIES:
            raise ValueError(
                f"category {self.category!r} not in {CATEGORIES}. "
                "Exactly 7 categories are supported in v0."
            )
        if self.category == "financial":
            if self.subtype not in FINANCIAL_SUBTYPES:
                raise ValueError(
                    f"financial component requires subtype in {FINANCIAL_SUBTYPES}, "
                    f"got {self.subtype!r}"
                )
        else:
            if self.subtype is not None:
                # Non-financial components don't use subtype
                pass
        return self

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "gap_precondition_id": self.gap_precondition_id,
            "description": self.description,
            "category": self.category,
            "effort_estimate": self.effort_estimate,
            "reversibility": self.reversibility,
            "dependencies": self.dependencies,
        }
        if self.category == "financial":
            d["subtype"] = self.subtype
        if self.unsourced:
            d["unsourced"] = True
        if self.fallback:
            d["fallback"] = True
        return d


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

class Histogram(BaseModel):
    """Category distribution for a Backcaster run.

    Always emitted — zero-count entries included for all 7 categories.
    Financial subtypes broken out separately plus a roll-up total.
    Research fallbacks broken out as research.fallback.
    """
    counts: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_components(cls, components: list[Component]) -> "Histogram":
        counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
        # financial subtype breakouts
        for st in FINANCIAL_SUBTYPES:
            counts[f"financial.{st}"] = 0
        counts["research.fallback"] = 0

        for comp in components:
            counts[comp.category] = counts.get(comp.category, 0) + 1
            if comp.category == "financial" and comp.subtype:
                key = f"financial.{comp.subtype}"
                counts[key] = counts.get(key, 0) + 1
            if comp.category == "research" and comp.fallback:
                counts["research.fallback"] = counts.get("research.fallback", 0) + 1

        return cls(counts=counts)

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


# ---------------------------------------------------------------------------
# RoadmapRun
# ---------------------------------------------------------------------------

class RoadmapRun(BaseModel):
    run_id: str
    goal_file: str
    goal_text: str
    model: str
    corpus_used: list[str] = Field(default_factory=list)
    scenario_ids: list[str] = Field(default_factory=list)
    timing: dict[str, Any] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    epistemic_caution: Literal["low", "medium", "high"] = "high"
    concentration_warning: list[str] = Field(default_factory=list)
    degraded_paths: list[str] = Field(default_factory=list)
    derive_fallback_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal_file": self.goal_file,
            "model": self.model,
            "corpus_used": self.corpus_used,
            "scenario_ids": self.scenario_ids,
            "timing": self.timing,
            "prompt_hashes": self.prompt_hashes,
            "epistemic_caution": self.epistemic_caution,
            "concentration_warning": self.concentration_warning,
            "degraded_paths": self.degraded_paths,
            "derive_fallback_count": self.derive_fallback_count,
        }
