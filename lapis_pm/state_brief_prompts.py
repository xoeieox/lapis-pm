"""Prompt templates for lapis-pm state briefs.

One template per cadence. The data-read layer formats bucket data into labelled
sections before the LLM call; the template instructs the model to write prose
under those headers exactly as given.

Bucket ordering (load-bearing — daily cadences: morning/afternoon/live):
  Built (since <start>)
  Notable ratifications (since <start>)
  In flight
  Captured — not yet built
  Awaiting your call
  Gardener Cross-Cutting Observations

Weekly briefs omit the Gardener bucket *header* (six-bucket ordering above
is daily-cadence only) but receive a 7-day Gardener observation data block
appended below the five rendered buckets, for the Weekly Arc synthesis to
consume (see format_bucket_sections `period` arg).
"""

from __future__ import annotations

BUCKET_ORDER = [
    "Built",
    "Notable ratifications",
    "In flight",
    "Captured — not yet built",
    "Awaiting your call",
    "Gardener Cross-Cutting Observations",
]

_GARDENER_BUCKET = "Gardener Cross-Cutting Observations"

DAILY_SYSTEM = """You are the Lapis PM state-brief narrator. Your job is to write
clear, terse prose for Erah — the principal engineer — summarising the current
state of in-flight Lapis work. You receive structured data already sorted into
named buckets. Write prose under each bucket header as given. Do not add headers
of your own and do not move items between headers.

For the Gardener Cross-Cutting Observations bucket, preserve the urgency labels
([Critical], [Warning]) and evidence links exactly as given. These are systemic
insights from Gardener's overnight synthesis — treat them as high-priority.
Gardener is a standing observer, not an enforcement mechanism: Critical signals
indicate systemic issues that may need human attention, but do NOT automatically
block or invalidate claims in any other bucket."""

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
section as instructed.

NOTE: Omit the Gardener Cross-Cutting Observations bucket header entirely for
weekly briefs. However, you MUST synthesize the Gardener data block (7-day
window, appended below the five buckets) into the "Weekly arc" section —
transform daily Critical signals into a "changing climate" narrative, not
just a list of isolated storms.

TONE MANDATE: the Weekly arc's Gardener synthesis is a narrative mirror —
reveal patterns without accusation. Weave the story of decay, but do not
name the rotter. The Gardener shows the cliff; the human chooses to turn.
Avoid language that triggers defensiveness or assigns blame. Present the
pattern as a story that invites reflection, not an indictment."""

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

If a "Gardener Observations (7-day window for synthesis)" data block is present
above, weave it into the Weekly arc as a narrative mirror: what systemic pattern
emerged across the week's Critical/Warning signals? Don't list isolated
incidents — tell the story of the pattern, in a tone that invites reflection
rather than assigning blame. If the data block is absent or empty, omit any
Gardener mention from the Weekly arc.
"""


def format_bucket_sections(buckets: dict[str, list[str]], start_label: str, *, period: str = "daily") -> str:
    """Render the standard buckets into labelled markdown sections.

    Args:
        buckets: mapping of bucket name → list of item strings.
            Keys must match BUCKET_ORDER entries (with or without the
            "(since <start>)" suffix — this function adds that suffix).
        start_label: human-readable time label, e.g. "2026-05-01 08:00 PT" or
            "2026-04-24 (7 days ago)".
        period: "weekly" or anything else — normalized to weekly vs.
            non-weekly. Weekly omits the Gardener bucket *header* and instead
            appends its data (the 7-day window) as a raw block below the
            five rendered buckets, for the Weekly Arc synthesis to consume.
    """
    sections: list[str] = []
    gardener_data: list[str] = []
    for name in BUCKET_ORDER:
        if name == _GARDENER_BUCKET and period == "weekly":
            gardener_data = buckets.get(name, [])
            continue
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

    if period == "weekly" and gardener_data:
        gardener_body = "\n".join(f"- {item}" for item in gardener_data)
        sections.append(
            f"## Gardener Observations (7-day window for synthesis)\n{gardener_body}"
        )

    return "\n\n".join(sections)


def build_daily_prompt(buckets: dict[str, list[str]], start_label: str) -> str:
    """Return the user-turn prompt for morning/afternoon/live briefs."""
    return DAILY_TEMPLATE.format(
        bucket_sections=format_bucket_sections(buckets, start_label, period="daily"),
    )


def build_weekly_prompt(buckets: dict[str, list[str]], start_label: str) -> str:
    """Return the user-turn prompt for weekly briefs."""
    return WEEKLY_TEMPLATE.format(
        bucket_sections=format_bucket_sections(buckets, start_label, period="weekly"),
    )
