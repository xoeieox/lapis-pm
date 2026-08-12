"""hold_shadow_rules.py — deterministic classification rules for the
hold-shadow observer (lapis-pm-hold-shadow-observer-v0).

HELD (see lapis_pm/authority.py HELD_PATTERNS): this is the rules table an
enforce-mode successor unit would eventually act on. Held for the same
reason authority.py holds itself (authority.py:38, "the fixer must not
autonomously edit its own cage") — the fixer must not autonomously edit the
boundary Erah's Thursday reading is measuring.

No LLM, no I/O, no imports beyond stdlib. Every function here is a pure
function over already-structured inputs (PR-hold reasons, authority level,
issue presence) fed in by lapis_pm/hold_shadow.py's two observation hooks —
see that module's docstring for how they're wired to run_spec_review and
_act_brief.
"""
from __future__ import annotations

# Closed enum: the hold_class values a PR-hold classifies into. "other" is
# the residual bucket for reasons text the table below doesn't recognize —
# never silently mapped to a more specific class.
HOLD_CLASSES = (
    "held_path",
    "needs_human_screen",
    "hold_authority_default",
    "review_exhausted",
    "lost_dispatch",
    "other",
)

# Closed enum: the REAL action vocabulary (brief.py:110's _ACTION_KINDS) plus
# amend_spec_redispatch (an enforce-mode-only action, named here so
# Thursday's boundary is checkable against the actual action space even
# though nothing in this unit can take it) and none (the Erah-domain shapes
# where no mechanical action would have applied).
WOULD_HAVE_ACTIONS = (
    "merge_pr",
    "force_dispatch_retry",
    "pause_target",
    "unbind_target",
    "acknowledge_and_clear",
    "adopt_pr",
    "amend_spec_redispatch",
    "none",
)

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"


def classify_hold(
    hold_reasons: list[str],
    *,
    authority_level: str = "hold",
    has_issues: bool = False,
) -> tuple[str, str, dict, str]:
    """Deterministic rules table: PR-hold reasons -> (hold_class,
    would_have_action, would_have_params, confidence).

    No model call, no randomness — the same input always yields the same
    output (DoD-1's fixture-determinism requirement). Erah-domain shapes
    (held paths, needs-human screens — the shapes this Thursday reading
    exists to rule on) derive would_have_action="none" per the spec's
    ratified scope; this deterministic core is not meant to pre-empt them.
    """
    reasons_text = " ".join(hold_reasons).lower()

    if "held path" in reasons_text:
        return "held_path", "none", {}, CONFIDENCE_HIGH

    if "needs-human" in reasons_text:
        return "needs_human_screen", "none", {}, CONFIDENCE_HIGH

    if "budget exhausted" in reasons_text or "cycle budget" in reasons_text:
        return "review_exhausted", "force_dispatch_retry", {}, CONFIDENCE_MEDIUM

    if "lost" in reasons_text and "dispatch" in reasons_text:
        return "lost_dispatch", "force_dispatch_retry", {}, CONFIDENCE_MEDIUM

    if authority_level == "hold":
        # The hold-authority default: every PR to a hold-authority target
        # lands here when no more specific reason matched above.
        # has_issues distinguishes a clean static pass (would likely have
        # merged) from one carrying flagged issues (would likely have gone
        # back to the fixer) — still a default, hence lower confidence than
        # the two structural matches above.
        if has_issues:
            return "hold_authority_default", "force_dispatch_retry", {}, CONFIDENCE_MEDIUM
        return "hold_authority_default", "merge_pr", {}, CONFIDENCE_LOW

    return "other", "none", {}, CONFIDENCE_LOW
