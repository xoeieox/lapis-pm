"""Intent artifact support — read /srv/lapis/intent/<target_id>.md and build
dispatch-injection blocks.

/srv/lapis/intent/ is the deterministic storage path for per-target human intent
artifacts (spec: intent-layer-harvest-v0). Files are authored by the
harvest-assist skill, keyed by target_id — out of the fuzzy-recall index,
zero BRIX cost, append-only history.

VOID fields (<!-- VOID: <gap> --> markers) denote fields the harvest skill
could not ground in Erah's verbatim quotes or explicit agreements. Workers
receiving a VOID MUST NOT fill or work around it — surface it and flag to Erah.
"""
from __future__ import annotations

import re
from pathlib import Path

from agents_core.room_paths import room_path, room_str

INTENT_DIR = room_path('intent')
VOID_RE = re.compile(r"<!--\s*VOID[:\s]", re.IGNORECASE)

# Mem key prefix for the consecutive-void-dispatch counter (calcification monitor).
VOID_DISPATCH_KEY_PREFIX = "intent/void-dispatch-count"

# Consecutive dispatches with unresolved voids before the calcification advisory fires.
CALCIFICATION_THRESHOLD = 3


def _path(target_id: str) -> Path:
    return INTENT_DIR / f"{target_id}.md"


def _strip_frontmatter(content: str) -> str:
    """Strip YAML frontmatter (--- ... ---) from content, returning only the body."""
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return content
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return "".join(lines[i + 1:]).lstrip("\n")
    return content


def load(target_id: str) -> str | None:
    """Return the intent artifact content, or None if absent."""
    try:
        return _path(target_id).read_text()
    except (FileNotFoundError, OSError):
        return None


def has_voids(content: str) -> bool:
    """True if the artifact contains at least one VOID marker."""
    return bool(VOID_RE.search(content))


def void_field_count(content: str) -> int:
    """Count distinct VOID markers in the artifact."""
    return len(VOID_RE.findall(content))


def dispatch_block(target_id: str) -> str:
    """Return the intent section for injection into a dispatch's vars_.

    Returns an empty string when no artifact exists — transparent no-op for
    targets that have not yet had their intent harvested.

    When VOIDs are present, appends a mandatory advisory for the worker:
    do not fill, surface and flag to Erah.
    """
    content = load(target_id)
    if not content:
        return ""
    n = void_field_count(content)
    void_note = ""
    if n:
        void_note = (
            f"\n\n> **ADVISORY — {n} VOID field(s) present** "
            f"(see `<!-- VOID: ... -->` markers above).\n"
            "> You MUST NOT fill, resolve, or work around any VOID.\n"
            "> Surface each VOID explicitly (in your PR description or as an advisory issue)\n"
            "> and FLAG it back to Erah. VOIDs are advisory-only and never block\n"
            "> implementation work."
        )
    body = _strip_frontmatter(content)
    return f"## Target intent\n\n{body.strip()}{void_note}"
