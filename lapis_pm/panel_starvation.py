"""Panel starvation detection and confidence attenuation
(lapis-pm-degraded-panel-confidence-and-absence-claims-v0, Leg 1).

Incident: PR #220 shipped a reviewer verdict at confidence=0.9 with all three
corroboration legs down — the local witness returned invalid JSON, the
corroboration pass reported "LLM unavailable", and the second node was
unreachable. Confidence was not attenuated by any of it, and the false HIGH
finding it carried ("`_check_equal` is not defined anywhere") auto-dispatched
a fixer retry against code that was fine.

Per R1 (ratified 2026-08-08, closed — not open for relitigation): the
threshold is ALL legs surviving, not a graded count. Any leg down drops the
verdict to advisory — it may inform, it may never gate or auto-dispatch on
its own authority. This module only computes the condition and the
attenuated confidence; the gate/advisory routing lives in pm_core.py.

AMENDED (2026-09-10, decision/reviewer-single-leg-local-phala-test-key):
the GATING legs are the two local legs (local_witness + corroboration, both
through the local GW :8081). The second_node (Phala TEE) leg is optional for
gating: its absence attenuates confidence and flags loudly, but does NOT by
itself drop the verdict to advisory. The PR #220 guard above stands: a
verdict with BOTH local legs down still never gates or auto-dispatches (in
that incident all three legs were down, so the amended rule does not reopen
the failure class).
"""

from __future__ import annotations

# The three corroboration legs this seat's verdict depends on. Order is
# stable — it is used both for the starved-legs list and the attenuation
# denominator.
_LEGS = ("local_witness", "corroboration", "second_node")

# Gating legs (AMENDED 2026-09-10,
# decision/reviewer-single-leg-local-phala-test-key): the reviewer
# is single-leg LOCAL — a verdict gates on the two local legs surviving.
# second_node (Phala TEE) absence attenuates + flags loudly but does not
# starve. PR #220 guard: BOTH local legs are mandatory — in that incident
# (all three legs down) the verdict still must not gate.
_GATE_LEGS = ("local_witness", "corroboration")


def _local_witness_down(verdict: dict) -> bool:
    """True if the local-reviewer-witness pass failed or returned unparseable JSON.

    `local_reviewer_witness.LocalReviewerWitnessResult.agreement` is
    "local_failed" on every failure path (unreachable endpoint, HTTP error,
    JSONDecodeError, empty response) — see local_reviewer_witness.py
    `_score_agreement` / `_make_failure`. Absence of the witness block
    entirely (e.g. the pass was skipped before it could run) counts as down
    too — a missing witness cannot corroborate anything.
    """
    witness = verdict.get("local_reviewer_witness")
    if not witness:
        return True
    return witness.get("agreement") == "local_failed"


def _corroboration_down(verdict: dict) -> bool:
    """True if the corroboration (node1) pass could not reach substrate/LLM,
    or reached it but was cut off before producing a usable answer.

    `corroboration_adapter.LapisPMReviewerAdapter.score()` sets `leg_status`
    to something other than "ok" on every failure path — "substrate_unavailable"
    (probed node unreachable, or any exception from the LLM call) and
    "truncated" (HTTP 200 but finish_reason=="length" cut the model off; see
    lapis-pm-corroboration-thinking-parse-and-truncation-loudness-v0) both
    count as down. leg_status is the field consumers must read instead of
    string-matching `claim`/`notes` prose (see CorroborationResult's own
    docstring) — a verdict of "uncertain" because there were simply no
    identifiers to check ("(no identifiers extracted)") is leg_status="ok", a
    legitimate clean result, not starvation. Absence of the block entirely
    (pass never attached) also counts as down.

    Falls back to the pre-leg_status claim-marker check for verdicts written
    before that field existed (no `leg_status` key at all).
    """
    corr = verdict.get("corroboration_result")
    if not corr:
        return True
    leg_status = corr.get("leg_status")
    if leg_status is not None:
        return leg_status != "ok"
    return corr.get("claim") == "(substrate unavailable)"


def _second_node_down(verdict: dict) -> bool:
    """True if the second-node (cross-node) witness was unreachable.

    `corroboration_adapter.run_corroboration_pass()` sets
    `cross_node_divergence="node2_unavailable"` when node2's score() call
    itself degraded to substrate-unavailable/error — see
    corroboration_adapter.py :543-549. Absence of the field (e.g. an older
    verdict written before node2 corroboration existed) also counts as down —
    an absent leg cannot corroborate.
    """
    corr = verdict.get("corroboration_result")
    if not corr:
        return True
    return corr.get("cross_node_divergence") == "node2_unavailable"


_LEG_CHECKS = {
    "local_witness": _local_witness_down,
    "corroboration": _corroboration_down,
    "second_node": _second_node_down,
}


def starved_legs(verdict: dict) -> list[str]:
    """Return the names of every corroboration leg that is down for this verdict.

    Empty list means fully corroborated (all three legs survived).
    """
    return [leg for leg in _LEGS if _LEG_CHECKS[leg](verdict)]


def is_panel_starved(verdict: dict) -> bool:
    """AMENDED (2026-09-10): a GATE leg down means the panel is starved.

    The gate legs are the two local legs (`_GATE_LEGS`). A second_node-only
    absence is NOT starved — it attenuates confidence and flags loudly but
    the verdict stands on local legs. The PR #220 guard holds: both local
    legs down (with or without second_node) is starved.
    """
    down = set(starved_legs(verdict))
    return bool(down & set(_GATE_LEGS))


def attenuate_confidence(raw_confidence, legs_down: int, legs_total: int = len(_LEGS)):
    """Attenuate a reported confidence value proportional to legs down.

    Simple, auditable, linear formula: confidence scales by the fraction of
    legs that survived. Zero legs down leaves confidence unchanged; all legs
    down floors it at 0.0. This is a legibility signal only — it does NOT
    carry gate/advisory authority (that split is R1's any-leg-down rule,
    computed separately by `is_panel_starved`); the two must never disagree,
    so this function never itself decides gate vs. advisory.

    Returns None if raw_confidence is not a number (defensive — reviewer
    output is untrusted input).
    """
    try:
        raw = float(raw_confidence)
    except (TypeError, ValueError):
        return None
    if legs_total <= 0:
        return raw
    legs_down = max(0, min(legs_down, legs_total))
    survival_fraction = (legs_total - legs_down) / legs_total
    return round(raw * survival_fraction, 4)


def verdict_is_starved(verdict: dict) -> bool:
    """Authoritative starved check for a stored verdict dict.

    Prefers the precomputed `panel_starvation.starved` annotation (attached
    once, at encode time, by `apply_panel_starvation`) over recomputing from
    the raw `local_reviewer_witness`/`corroboration_result` blocks — a
    verdict that has already been annotated should not silently re-derive a
    different answer at gate time just because a caller trimmed the raw
    sub-blocks before storing/passing it around (e.g. test fixtures, or a
    future compaction pass). Falls back to a fresh `is_panel_starved` compute
    only when the verdict predates this feature entirely (no
    `panel_starvation` key at all).
    """
    pstarv = verdict.get("panel_starvation")
    if pstarv is not None:
        return bool(pstarv.get("starved"))
    return is_panel_starved(verdict)


def apply_panel_starvation(verdict: dict) -> dict:
    """Compute starvation + attenuated confidence and attach it to `verdict`.

    Mutates and returns `verdict` in place (mirrors the additive-only
    convention `corroboration_result`/`local_reviewer_witness` already use —
    see pm_core.py :6528-6533). Adds a `panel_starvation` block:

      {
        "legs_down": [...],       # subset of ("local_witness", "corroboration", "second_node")
        "starved": bool,          # True iff a GATE leg (local_witness /
                                  # corroboration) is down — AMENDED 2026-09-10;
                                  # legs_down may be non-empty (second_node)
                                  # with starved False
        "confidence_raw": <original "confidence" value, unmodified>,
      }

    Also rewrites the top-level `confidence` field to the attenuated value so
    every downstream reader (claude-view, briefs, `_act_brief`'s reviewer
    verdict summary) sees the attenuated number by default, while the raw
    value stays available in `panel_starvation.confidence_raw` for audit —
    Definition of Done #3: "record both the raw and the attenuated value so
    the attenuation is auditable rather than a silent rewrite."
    """
    legs_down = starved_legs(verdict)
    raw_confidence = verdict.get("confidence")
    attenuated = attenuate_confidence(raw_confidence, len(legs_down))

    verdict["panel_starvation"] = {
        "legs_down": legs_down,
        "starved": is_panel_starved(verdict),
        "confidence_raw": raw_confidence,
    }
    if attenuated is not None:
        verdict["confidence"] = attenuated
    return verdict
