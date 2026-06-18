"""The dry overlay arbiter: forced-extremal-articulation at the intersection of two grounded poles."""
from __future__ import annotations

import json
import logging
from typing import Any

from .schema import ConstraintSurface, ConstraintSurfaceItem, PoleClaim, PoleOutput

log = logging.getLogger(__name__)

JAGGED_SEAM_GW_UNAVAILABLE = "JAGGED_SEAM_GW_UNAVAILABLE"


def _norm(s: str) -> str:
    """Normalize a provenance string for matching: collapse whitespace, case-fold.

    Used by _resolve_claim_by_provenance as a fallback when exact match fails.
    Handles trivial drift (extra whitespace, case differences) without fuzzy matching.
    """
    return " ".join(s.split()).casefold()


class JaggedSeamGravityWellUnavailable(Exception):
    """GravityWell is unavailable (not serving, doorman unreachable, or wake failed)."""

    pass


def overlay(
    fear: PoleOutput,
    desire: PoleOutput,
    *,
    operator_class: str = "gravitywell",
) -> ConstraintSurface:
    """Overlay two grounded poles (fear and desire) to extract decision-variables.

    The arbiter is a DRY intersection-extractor: it takes two ALREADY-grounded poles'
    outputs over the same catalogued state and emits named decision-variables at their
    intersection. No verdict, no recommendation, no narrative — only provenance-tagged
    decision-variables.

    Args:
        fear: PoleOutput from the fear pole (Scout: drift/exploit/failure finder).
        desire: PoleOutput from the desire pole (Backcaster: works-back-from-goal).
        operator_class: LLM operator class to use ("gravitywell" default, "qwen" alt).

    Returns:
        ConstraintSurface with decision-variables named at the intersection, each
        carrying provenance to both the fear-side and desire-side claim it draws on.
        If either pole is thin or empty, returns skipped_no_axis=True without calling
        the model (TENSION-GATE short-circuit).

    Raises:
        ValueError: If fear.state_ref != desire.state_ref (poles must be over the
            same catalogued state).
        JaggedSeamGravityWellUnavailable: If the operator is unavailable and
            on_wake_fail="skip" is set (no paid fallback).
    """
    # Verify state_ref match (Sonnet #3).
    if fear.state_ref != desire.state_ref:
        raise ValueError(
            f"state_ref mismatch: fear={fear.state_ref!r} != desire={desire.state_ref!r}. "
            "Poles MUST be over the same catalogued state."
        )

    # TENSION-GATE: short-circuit if either pole is thin or has no claims (Amendment B).
    # An honestly-thin pole is a legal answer; the arbiter must not fabricate a seam.
    if fear.thin or desire.thin or not fear.claims or not desire.claims:
        return ConstraintSurface(
            state_ref=fear.state_ref,
            items=[],
            skipped_no_axis=True,
            dropped_provenance_count=0,
        )

    # The overlay call routes through call_operator with on_wake_fail="skip".
    try:
        from agents_core.llm import call_operator  # type: ignore[import]
    except ImportError:
        log.error("agents_core.llm not available — cannot route to operator")
        raise JaggedSeamGravityWellUnavailable(
            "agents_core.llm not available"
        ) from None

    # Serialize both poles as JSON for the model.
    fear_json = _serialize_pole_output(fear)
    desire_json = _serialize_pole_output(desire)

    user_message = f"""You are a dry overlay arbiter. Your job is to extract decision-variables from the intersection of two grounded poles (fear and desire) over the same catalogued state.

**Poles:**

FEAR POLE (constraint/failure focus):
{fear_json}

DESIRE POLE (goal/capability focus):
{desire_json}

**Your task:**

For each place a fear-claim and a desire-claim are about the same object from opposite sides, name the decision-variable that falls out and cite both source claims.

**Output constraints (STRICT):**
- Output ONLY a JSON list of constraint-surface items.
- Each item MUST have: decision_variable (string naming the variable, NOT a recommendation), fear_source_provenance (string, the provenance of the fear claim), desire_source_provenance (string, the provenance of the desire claim), crossing_type (one of "tension", "shared_dependency", or "reinforcement").
- Do NOT include a verdict, recommendation, or synthesis field.
- Do NOT narrate or explain. Output is ONLY the JSON array, nothing else.

Example output format:
[
  {{"decision_variable": "how restrictive consent terms may be given X", "fear_source_provenance": "scout:cell_id=.../drift_signals[0]", "desire_source_provenance": "backcaster:gap[precondition_id=...]/what_missing", "crossing_type": "tension"}},
  {{"decision_variable": "...", "fear_source_provenance": "...", "desire_source_provenance": "...", "crossing_type": "..."}}
]

If there is no intersection (no places where fear and desire claims are about the same object), return an empty array: []
"""

    system_message = """You are a dry overlay arbiter. Your role is to extract named decision-variables from the intersection of two grounded poles' claims over the same catalogued state. You do NOT issue verdicts, recommendations, or synthesize narratives. You ONLY extract and name the intersection."""

    result = call_operator(
        operator_class,
        prompt=user_message,
        system=system_message,
        json_mode=True,
        timeout=300,
        on_wake_fail="skip",
    )

    # Handle operator unavailable (Sonnet #6: GW-down bridge).
    if result is None:
        raise JaggedSeamGravityWellUnavailable(
            f"{operator_class} unavailable (wake failed, not serving, or doorman unreachable)"
        )

    # Parse the result as a JSON list and validate against schema.
    try:
        items_json = json.loads(str(result))
        if not isinstance(items_json, list):
            raise ValueError("Expected JSON array, got non-list")

        # Build ConstraintSurfaceItem objects with full provenance.
        items: list[ConstraintSurfaceItem] = []
        dropped_provenance_count = 0
        for item_dict in items_json:
            # Validate schema: no verdict, recommendation, or synthesis fields (AC#2).
            if any(
                k in item_dict
                for k in ["verdict", "recommendation", "synthesis", "analysis", "note"]
            ):
                raise ValueError(
                    f"Response contains forbidden fields: {[k for k in item_dict.keys() if k in ['verdict', 'recommendation', 'synthesis', 'analysis', 'note']]}"
                )

            fear_prov = item_dict.get("fear_source_provenance", "")
            desire_prov = item_dict.get("desire_source_provenance", "")

            # Resolve provenance strings back to PoleClaim objects.
            fear_claim = _resolve_claim_by_provenance(fear, fear_prov)
            desire_claim = _resolve_claim_by_provenance(desire, desire_prov)

            if fear_claim is None:
                log.warning(
                    f"Could not resolve fear provenance {fear_prov!r}; skipping item"
                )
                dropped_provenance_count += 1
                continue
            if desire_claim is None:
                log.warning(
                    f"Could not resolve desire provenance {desire_prov!r}; skipping item"
                )
                dropped_provenance_count += 1
                continue

            items.append(
                ConstraintSurfaceItem(
                    decision_variable=item_dict.get("decision_variable", ""),
                    fear_source=fear_claim,
                    desire_source=desire_claim,
                    crossing_type=item_dict.get(
                        "crossing_type", "tension"
                    ),  # type: ignore[arg-type]
                )
            )

        return ConstraintSurface(
            state_ref=fear.state_ref,
            items=items,
            skipped_no_axis=False,
            dropped_provenance_count=dropped_provenance_count,
        )

    except (json.JSONDecodeError, ValueError) as e:
        log.error(f"Failed to parse or validate operator response: {e}")
        raise ValueError(f"Invalid operator response: {e}") from e


def _serialize_pole_output(pole: PoleOutput) -> str:
    """Serialize a PoleOutput to JSON for the model."""
    claims_data = [
        {
            "claim": c.claim,
            "kind": c.kind,
            "provenance": c.provenance,
            "fidelity": c.fidelity,
        }
        for c in pole.claims
    ]
    pole_data = {
        "kind": pole.kind,
        "state_ref": pole.state_ref,
        "claims": claims_data,
        "thin": pole.thin,
    }
    return json.dumps(pole_data, indent=2)


def _resolve_claim_by_provenance(pole: PoleOutput, provenance: str) -> PoleClaim | None:
    """Resolve a provenance string back to its PoleClaim in the pole.

    Tries exact match first, then normalized (whitespace/case-insensitive) match.
    Returns the PoleClaim or None if no match is found.

    Simple linear search (O(n) where n is the number of claims).
    """
    # Try exact match first (unchanged).
    for claim in pole.claims:
        if claim.provenance == provenance:
            return claim

    # On miss, try normalized match: collapse whitespace, case-fold.
    normalized_prov = _norm(provenance)
    for claim in pole.claims:
        if _norm(claim.provenance) == normalized_prov:
            return claim

    return None
