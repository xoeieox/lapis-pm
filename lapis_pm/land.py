"""Arc doc synthesis — the `land` action.

Produces `/srv/lapis/lapis-state/<target_id>.md`: a narrative arc doc for a thread
that has landed (or is being archived). Mirrors `brief.synthesize()` but feeds
the FULL chronology (not top-k recall) to Haiku and writes to a file instead
of a PM comment.

No TargetStore mutation in this slice. No chub update in this slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agents_core.llm import call_claude_cli

from . import episodic


ARC_DOC_DIR = Path("/srv/lapis/lapis-state")
PACIFIC = ZoneInfo("America/Los_Angeles")

ARC_DOC_SYSTEM = """You are the Lapis PM producing an *arc doc* for a work thread that has landed.

The arc doc is a narrative snapshot that future PM instances (and Erah) will read to
understand what this thread was, what it produced, and what it left open. RoomRAG will
index it; it becomes queryable context for future work.

Output Markdown in this exact structure (no preamble, no closing, no code fences):

# <Target ID> — Arc Doc

## Origin
<2-4 sentences: what spawned this thread, what problem it set out to address, and the
spec's core claim. Use the Spec section of the input.>

## Key decisions
- <bullet: decision and why, terse>
- <bullet: another decision, terse>
- <as many as are actually load-bearing; skip if there were none>

## Dispatches and results
<1-3 short paragraphs summarizing what shaped agents were dispatched, what they produced,
what was accepted/rejected. If no dispatches occurred, say so in one sentence.>

## Landing summary
<2-3 sentences: what shipped, what was abandoned, what the final state is. Include
PR numbers if visible in the episodes.>

## Open threads
- <bullet: what's still unresolved, if anything>
- <skip section entirely if nothing remains open>

Rules:
- Be specific. Cite dates, PR numbers, component names when the episodes mention them.
- Never invent facts. If the episodes don't say it, omit it.
- The reader is a future PM or Erah reopening this later — write so they can pick up cold.
"""

# Budget guard — episodes block fed to Haiku
MAX_EPISODES_CHARS = 12_000


@dataclass
class ArcDoc:
    target_id: str
    body: str
    path: Path


def _format_all_episodes(comments: list, limit_chars: int = 400) -> str:
    """Chronological rendering of every comment on the thread."""
    if not comments:
        return "(no episodes)"
    lines = []
    for c in comments:
        snippet = c.content.replace("\n", " ")
        if len(snippet) > limit_chars:
            snippet = snippet[:limit_chars] + "…"
        tags = ",".join(c.tags) if c.tags else "-"
        lines.append(f"- [{c.ts}] {c.author} ({tags}) {snippet}")
    joined = "\n".join(lines)
    if len(joined) > MAX_EPISODES_CHARS:
        # Keep earliest + most recent, drop the middle.
        half = MAX_EPISODES_CHARS // 2
        head = joined[:half]
        tail = joined[-half:]
        joined = head + "\n\n…[middle elided for budget]\n\n" + tail
    return joined


def generate_arc_doc(target_id: str) -> ArcDoc:
    """Synthesize the arc doc body. Does NOT write to disk — caller decides."""
    spec_body = episodic.spec(target_id) or "(no spec bound)"
    comments = episodic.all_comments(target_id)
    episodes_block = _format_all_episodes(comments)
    now = datetime.now(PACIFIC).isoformat(timespec="seconds")

    user = (
        f"Target ID: {target_id}\n"
        f"Generated: {now}\n"
        f"Comment count: {len(comments)}\n\n"
        f"Spec:\n{spec_body}\n\n"
        f"Full thread chronology (oldest first):\n{episodes_block}\n"
    )

    body = call_claude_cli(
        prompt=user, system=ARC_DOC_SYSTEM,
        model="haiku", timeout=180,
    )
    if not body:
        body = (
            f"# {target_id} — Arc Doc\n\n"
            "## Origin\nArc doc synthesis failed — Haiku call returned empty.\n\n"
            f"## Landing summary\nFallback dump of thread chronology:\n\n{episodes_block}\n"
        )

    # Prepend a tiny YAML frontmatter so RoomRAG / Kami can tag arc docs cleanly.
    frontmatter = (
        "---\n"
        f"target_id: {target_id}\n"
        f"generated: {now}\n"
        f"comment_count: {len(comments)}\n"
        "kind: arc-doc\n"
        "---\n\n"
    )
    full = frontmatter + body.strip() + "\n"

    path = ARC_DOC_DIR / f"{target_id}.md"
    return ArcDoc(target_id=target_id, body=full, path=path)


def write_arc_doc(arc: ArcDoc) -> Path:
    """Write to /srv/lapis/lapis-state/<target_id>.md. Creates parent if missing."""
    arc.path.parent.mkdir(parents=True, exist_ok=True)
    arc.path.write_text(arc.body, encoding="utf-8")
    return arc.path
