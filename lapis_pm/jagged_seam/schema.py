"""Normalized pole-input contract and constraint-surface output for the jagged-seam arbiter.

The jagged-seam arbiter consumes a common normalized shape so asymmetric warm-up
(prose fear side) and live run (ScoutTracePayload fear side) feed the same arbiter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PoleClaim:
    """A single claim from one pole (fear or desire).

    Attributes:
        claim: The text of the claim.
        kind: Either "fear" (constraint/failure) or "desire" (goal/capability).
        provenance: Free pointer back to the source for traceability.
            Examples: "scout:cell_id=…/drift_signals[2]",
            "backcaster:gap[precondition_id=…]",
            "mem:thread/zephyr-…#drift-vectors"
        fidelity: "verified" = drawn from live/archived structured pole output;
            "transcribed" = hand-transcribed prose (warm-up fear side).
    """

    claim: str
    kind: Literal["fear", "desire"]
    provenance: str
    fidelity: Literal["verified", "transcribed"]


@dataclass
class PoleOutput:
    """Normalized output from one pole (fear or desire).

    Attributes:
        kind: Either "fear" (Scout) or "desire" (Backcaster).
        state_ref: External semantic label for the catalogued state both poles
            react to. NOT derivable from a payload; the caller must pass it.
            Examples: "zephyr-defederation-teeth-2026-06-08"
        claims: List of PoleClaim items from this pole.
        thin: True if this pole found no extremal axis (e.g., Scout had no breaks,
            Backcaster found no gaps). A thin pole is honest — the arbiter must not
            fabricate a seam from nothing.
    """

    kind: Literal["fear", "desire"]
    state_ref: str
    claims: list[PoleClaim] = field(default_factory=list)
    thin: bool = False


@dataclass
class ConstraintSurfaceItem:
    """A single decision-variable revealed by the overlay.

    Attributes:
        decision_variable: Names the what-matters the overlay reveals
            (a variable, NOT a recommendation). Example: "how restrictive
            consent terms may be given X".
        fear_source: The PoleClaim from the fear pole this item draws on.
        desire_source: The PoleClaim from the desire pole this item draws on.
        crossing_type: How the two poles intersect: "tension" (direct conflict),
            "shared_dependency" (both need the same thing), or "reinforcement"
            (both point the same way).
    """

    decision_variable: str
    fear_source: PoleClaim
    desire_source: PoleClaim
    crossing_type: Literal["tension", "shared_dependency", "reinforcement"]


@dataclass
class ConstraintSurface:
    """The seam output from the arbiter.

    A dry overlay: no verdict, no recommendation, no free-prose synthesis.
    Named decision-variables with two-sided, fidelity-tagged provenance.

    Attributes:
        state_ref: Shared label both poles reacted to (from PoleOutput.state_ref).
        items: List of ConstraintSurfaceItem revealing decision-variables.
        skipped_no_axis: True if the tension-gate short-circuited (one or both
            poles were thin or had no claims). This is a flag-to-human,
            not a silent success — the consumers (U2/U3 drivers + U4 report)
            MUST render this as a loud flag.
    """

    state_ref: str
    items: list[ConstraintSurfaceItem] = field(default_factory=list)
    skipped_no_axis: bool = False
