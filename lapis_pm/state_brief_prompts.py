"""Prompt templates for lapis-pm state briefs.

One template per cadence. The data-read layer formats bucket data into labelled
sections before the LLM call; the template instructs the model to write prose
under those headers exactly as given.

Bucket ordering (load-bearing — all cadences):
  Built (since <start>)
  Notable ratifications (since <start>)
  In flight
  Captured — not yet built
  Awaiting your call
"""

from __future__ import annotations

BUCKET_ORDER = [
    "Built",
    "Notable ratifications",
    "In flight",
    "Captured — not yet built",
    "Awaiting your call",
]

DAILY_SYSTEM = """You are the Lapis PM state-brief narrator. Your job is to write
clear, terse prose for Erah — the principal engineer — summarising the current
state of in-flight Lapis work. You receive structured data already sorted into
named buckets. Write prose under each bucket header as given. Do not add headers
of your own and do not move items between headers."""

DAILY_TEMPLATE = """\
Write a short prose paragraph under each header exactly as given below.
Do not move items between headers.
If a section has no items, write "Nothing to report."
Be terse — Erah is context-switching; each paragraph should be 2–4 sentences.

{bucket_sections}
"""

WEEKLY_SYSTEM = """You are the Lapis PM state-brief narrator. Your job is to write
a weekly synthesis for Erah — the principal engineer — covering the trailing 7 days
of Lapis work. You receive structured data already sorted into named buckets.
Write prose under each bucket header as given. Do not add headers of your own and
do not move items between headers. After the five buckets, write a "Weekly arc"
section as instructed."""

WEEKLY_TEMPLATE = """\
Write a short prose paragraph under each header exactly as given below.
Do not move items between headers.
If a section has no items, write "Nothing to report."
Be terse — each paragraph should be 2–4 sentences.

{bucket_sections}

After the five buckets, add a "## Weekly arc" section:
identify the dominant theme across the landed work, describe where the
Distributed Grounding spine stands, and name what is newly unblocked in active
chains (note: unblockings outside of explicit chain depends_on are model-inferred
and should be labelled "speculative" in your output).
"""


def format_bucket_sections(buckets: dict[str, list[str]], start_label: str) -> str:
    """Render the five standard buckets into labelled markdown sections.

    Args:
        buckets: mapping of bucket name → list of item strings.
            Keys must match BUCKET_ORDER entries (with or without the
            "(since <start>)" suffix — this function adds that suffix).
        start_label: human-readable time label, e.g. "2026-05-01 08:00 PT" or
            "2026-04-24 (7 days ago)".
    """
    sections: list[str] = []
    for name in BUCKET_ORDER:
        if name in ("Built", "Notable ratifications"):
            header = f"## {name} (since {start_label})"
        else:
            header = f"## {name}"
        items = buckets.get(name, [])
        if items:
            body = "\n".join(f"- {item}" for item in items)
        else:
            body = "(none)"
        sections.append(f"{header}\n{body}")
    return "\n\n".join(sections)


def build_daily_prompt(buckets: dict[str, list[str]], start_label: str) -> str:
    """Return the user-turn prompt for morning/afternoon/live briefs."""
    return DAILY_TEMPLATE.format(
        bucket_sections=format_bucket_sections(buckets, start_label),
    )


def build_weekly_prompt(buckets: dict[str, list[str]], start_label: str) -> str:
    """Return the user-turn prompt for weekly briefs."""
    return WEEKLY_TEMPLATE.format(
        bucket_sections=format_bucket_sections(buckets, start_label),
    )
