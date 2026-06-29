"""Conservative auto-resolve predicate for PM_AUTO_RESOLVE=conservative.

No LLM calls here — the predicate is fully deterministic (rules + live gates).
Returns (True, merge_option_id) or (False, reason). Unknown dial values or any
uncertainty fail-safe to (False, reason), routing to the normal gem/brief path.
"""

from __future__ import annotations

import logging
import os

from . import brief_gem
from .brief import _CLOSED_FORM_TRIGGERS

logger = logging.getLogger(__name__)


def _get_dial() -> str:
    """Read PM_AUTO_RESOLVE. Unknown values treated as 'off' (fail-safe)."""
    val = os.environ.get("PM_AUTO_RESOLVE", "conservative").lower().strip()
    if val in ("conservative", "off"):
        return val
    logger.warning("PM_AUTO_RESOLVE=%r is unknown; treating as 'off'", val)
    return "off"


def should_auto_resolve(
    cls,     # authority.PRClassification
    target,  # agents_core.targets.Target
    repo: str,
) -> tuple[bool, str | None]:
    """Conservative auto-resolve predicate — all six clauses must hold.

    Returns (True, merge_option_id) or (False, reason).
    """
    # Clause 1: dial is 'conservative'
    dial = _get_dial()
    if dial != "conservative":
        return False, f"dial={dial!r}"

    # Clause 2: advisory authority only (hold always escalates; None => fail-safe to gem)
    pm_authority = getattr(target, "pm_authority", None)
    if pm_authority != "advisory":
        return False, f"pm_authority={pm_authority!r}"

    # Clause 3: screen_verdict must be exactly "clean" (None/absent => fail-safe)
    sv = getattr(cls, "screen_verdict", None)
    if sv != "clean":
        return False, f"screen_verdict={sv!r}"

    # Clause 4: a merge_pr option exists with a concrete PR number.
    # advisory-clean trigger (no issues + concrete pr_number) maps to the
    # advisory-clean closed-form template, which carries a merge_pr option.
    if not cls.pr_number:
        return False, "no pr_number"
    if cls.issues:
        return False, "issues present (advisory-screen-issue path)"

    merge_opt = next(
        (o for o in _CLOSED_FORM_TRIGGERS.get("advisory-clean", [])
         if o.get("action", {}).get("kind") == "merge_pr"),
        None,
    )
    if merge_opt is None:
        return False, "advisory-clean template has no merge_pr option"
    merge_option_id: str = merge_opt["id"]

    # Clause 6: verification must be 'machine' — target's acceptance must be
    # fully establishable by the automated pipeline. Absent field => 'pm-live-test'
    # (fail-safe: unmarked targets stop auto-resolving).
    pm_verification = getattr(target, "data", {}).get("pm_verification", "pm-live-test")
    if pm_verification != "machine":
        return False, f"verification={pm_verification!r}:needs-pm-touch"

    # Clause 5: live held-path check via the same gate brief_gem uses (fail-closed)
    # Placed last because it requires a network call; short-circuit above avoids it.
    blocked, reason = brief_gem._held_path_check_live(target.id, cls.pr_number, repo)
    if blocked:
        return False, f"held_path blocked: {reason}"

    return True, merge_option_id
