"""Per-pole normalizers: native output → PoleOutput.

Maps Scout and Backcaster outputs (and hand-transcribed prose) to the
normalized PoleClaim/PoleOutput contract so all three sources feed the
same arbiter.
"""
from __future__ import annotations

from typing import Any, Literal

from .schema import PoleClaim, PoleOutput


def from_scout_trace(
    payload: dict[str, Any] | Any,
    state_ref: str,
    *,
    fidelity: Literal["verified", "transcribed"] = "verified",
) -> PoleOutput:
    """Normalize a ScoutTracePayload into PoleOutput (fear pole).

    Args:
        payload: Either a ScoutTracePayload dataclass or a deserialized dict
            (from JSON archive).
        state_ref: External semantic label for the catalogued state.
        fidelity: Mark claims as "verified" (default, for live/archived payloads)
            or "transcribed" (for hand-transcribed prose warm-ups).

    Returns:
        PoleOutput with kind="fear", claims flattened from breaks_observed,
        leverage_points, surprises, drift_signals. If gw_skipped is set,
        returns thin=True with empty claims (skipped Scout run has no fear
        material — do not fabricate).
    """
    # Normalize payload to dict if it's a dataclass.
    if hasattr(payload, "to_dict"):
        payload_dict = payload.to_dict()
    else:
        payload_dict = payload

    # Check if Scout was skipped (no material to work with).
    if payload_dict.get("gw_skipped", False):
        return PoleOutput(kind="fear", state_ref=state_ref, claims=[], thin=True)

    claims: list[PoleClaim] = []
    cell_id = payload_dict.get("cell_id", "unknown")

    # Flatten breaks_observed.
    for i, break_item in enumerate(payload_dict.get("breaks_observed", [])):
        sig = break_item.get("signature", "")
        claims.append(
            PoleClaim(
                claim=sig,
                kind="fear",
                provenance=f"scout:cell_id={cell_id}/breaks_observed[{i}]",
                fidelity=fidelity,
            )
        )

    # Flatten leverage_points.
    for i, lp in enumerate(payload_dict.get("leverage_points", [])):
        description = lp.get("description", "") or str(lp)
        claims.append(
            PoleClaim(
                claim=description,
                kind="fear",
                provenance=f"scout:cell_id={cell_id}/leverage_points[{i}]",
                fidelity=fidelity,
            )
        )

    # Flatten surprises.
    for i, surprise in enumerate(payload_dict.get("surprises", [])):
        claims.append(
            PoleClaim(
                claim=surprise,
                kind="fear",
                provenance=f"scout:cell_id={cell_id}/surprises[{i}]",
                fidelity=fidelity,
            )
        )

    # Flatten drift_signals.
    for i, signal in enumerate(payload_dict.get("drift_signals", [])):
        description = signal.get("description", "") or str(signal)
        claims.append(
            PoleClaim(
                claim=description,
                kind="fear",
                provenance=f"scout:cell_id={cell_id}/drift_signals[{i}]",
                fidelity=fidelity,
            )
        )

    return PoleOutput(kind="fear", state_ref=state_ref, claims=claims)


def from_backcaster(
    gaps: list[Any],
    preconditions: list[Any],
    state_ref: str,
    components: list[Any] | None = None,
    *,
    fidelity: Literal["verified", "transcribed"] = "verified",
) -> PoleOutput:
    """Normalize Backcaster gaps and preconditions into PoleOutput (desire pole).

    Args:
        gaps: List of Gap objects or deserialized dicts with what_missing / what_miswired.
        preconditions: List of Precondition objects or deserialized dicts with statement.
        state_ref: External semantic label for the catalogued state.
        components: Optional list of Component objects (metadata only in this version;
            generates no claims).
        fidelity: Mark claims as "verified" (default) or "transcribed".

    Returns:
        PoleOutput with kind="desire", claims from Gap.what_missing + Gap.what_miswired
        + Precondition.statement.
    """
    claims: list[PoleClaim] = []

    # Map Precondition.statement to claims.
    for prec in preconditions:
        prec_dict = prec.to_dict() if hasattr(prec, "to_dict") else prec
        prec_id = prec_dict.get("id", "unknown")
        statement = prec_dict.get("statement", "")
        if statement:
            claims.append(
                PoleClaim(
                    claim=statement,
                    kind="desire",
                    provenance=f"backcaster:precondition_id={prec_id}",
                    fidelity=fidelity,
                )
            )

    # Map Gap.what_missing and Gap.what_miswired to claims.
    for gap in gaps:
        gap_dict = gap.to_dict() if hasattr(gap, "to_dict") else gap
        prec_id = gap_dict.get("precondition_id", "unknown")

        what_missing = gap_dict.get("what_missing", "")
        if what_missing:
            claims.append(
                PoleClaim(
                    claim=what_missing,
                    kind="desire",
                    provenance=f"backcaster:gap[precondition_id={prec_id}]/what_missing",
                    fidelity=fidelity,
                )
            )

        what_miswired = gap_dict.get("what_miswired", "")
        if what_miswired:
            claims.append(
                PoleClaim(
                    claim=what_miswired,
                    kind="desire",
                    provenance=f"backcaster:gap[precondition_id={prec_id}]/what_miswired",
                    fidelity=fidelity,
                )
            )

    # Note: Gap.what_exists is current-state context, not a desire-pole claim.
    # components are accepted for provenance metadata only (v0); no claims generated.

    return PoleOutput(kind="desire", state_ref=state_ref, claims=claims)


def from_prose(
    text_items: list[str],
    kind: Literal["fear", "desire"],
    state_ref: str,
    provenance_prefix: str,
    *,
    fidelity: Literal["verified", "transcribed"] = "transcribed",
) -> PoleOutput:
    """Build PoleOutput from hand-transcribed prose (warm-up fear side).

    Args:
        text_items: List of prose strings, each becomes one PoleClaim.
        kind: Either "fear" or "desire".
        state_ref: External semantic label for the catalogued state.
        provenance_prefix: Free string prefix for provenance (e.g.,
            "mem:thread/zephyr-…#drift-vectors").
        fidelity: Mark claims as "transcribed" (default for prose) or "verified".

    Returns:
        PoleOutput with kind=kind, claims built from text_items.
    """
    claims: list[PoleClaim] = []
    for i, text in enumerate(text_items):
        claims.append(
            PoleClaim(
                claim=text,
                kind=kind,
                provenance=f"{provenance_prefix}[{i}]",
                fidelity=fidelity,
            )
        )

    return PoleOutput(kind=kind, state_ref=state_ref, claims=claims)
