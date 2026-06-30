"""Prior-Art Scout Pydantic models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScoutItem(BaseModel):
    key: str
    namespace: str
    summary: str
    status: str
    findings: str = ""
    citations: list[dict] = Field(default_factory=list)
    outcome: str = ""  # sources-found | no-credible-sources | high-friction | infra-unavailable
    lean: str = ""     # adopt-pattern | adopt-tool | not-relevant | pending


class ScoutRun(BaseModel):
    run_date: str
    priority1_count: int
    priority2_batch: str  # e.g. "30/250 (cursor @decision/some-key)"
    skipped_hopeless: int
    items_targeted: int
    items_sourced: int
    items_honest_null: int
    caution: str          # low | medium | high
    model_policy: str
    saturated_namespaces: list[str] = Field(default_factory=list)
    wall_budget_applied: int | None = None
