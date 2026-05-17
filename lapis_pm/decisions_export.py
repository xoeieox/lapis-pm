"""decisions_export — export recent mem entries as a single markdown artifact.

Subcommand: lapis-pm decisions-export
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from agents_core.mem import MemoryStore

VALID_TYPES = ("decision", "feedback", "pattern")


def parse_since(spec: str) -> datetime:
    """Parse `Nd` or `YYYY-MM-DD` into a UTC datetime. Raises ValueError on bad input."""
    spec = spec.strip()
    if spec.endswith("d") and spec[:-1].isdigit():
        days = int(spec[:-1])
        return datetime.now(timezone.utc) - timedelta(days=days)
    return datetime.strptime(spec, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_types(include: str, exclude: str) -> list[str]:
    """Compute effective type list. Raises ValueError on unknown tokens or empty result."""
    inc = [t.strip() for t in include.split(",") if t.strip()]
    exc = [t.strip() for t in exclude.split(",") if t.strip()]
    for t in inc + exc:
        if t not in VALID_TYPES:
            raise ValueError(f"unknown type: {t} (valid: {', '.join(VALID_TYPES)})")
    effective = [t for t in inc if t not in exc]
    if not effective:
        raise ValueError("no types selected after exclude")
    return effective


# Far above current corpus size (~12 decisions/week, ~2k decisions all-time).
# If a future corpus exceeds this, raise it — but log a warning when result count == cap.
_LIST_CAP = 100_000


def collect(types: list[str], since: datetime, tag: str = "") -> dict[str, list[dict]]:
    """Pull all entries of each type since the cutoff. Filter by tag if provided.

    Sort: newest-first by `created_at`, tie-broken by `key` ascending (Invariant 2).
    Implementation uses two stable sorts (key ascending, then created_at descending).
    """
    store = MemoryStore()
    result: dict[str, list[dict]] = {}
    for t in types:
        rows = store.list_by_prefix(f"{t}/", limit=_LIST_CAP)
        if len(rows) == _LIST_CAP:
            print(
                f"[decisions-export] warning: hit list cap ({_LIST_CAP}) for {t}/ — "
                "results may be truncated",
                file=sys.stderr,
            )
        kept = []
        for r in rows:
            created = r.get("created_at", "")
            if not created or created < since.isoformat():
                continue
            if tag:
                row_tags = [s.strip() for s in (r.get("tags") or "").split(",")]
                if tag not in row_tags:
                    continue
            kept.append(r)
        # Two stable sorts: secondary key ascending first, then primary created_at descending.
        kept.sort(key=lambda r: r.get("key", ""))
        kept.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        result[t] = kept
    return result


def render(grouped: dict[str, list[dict]], since: datetime, tag: str = "") -> str:
    """Emit the markdown artifact."""
    now = datetime.now(timezone.utc)
    total = sum(len(v) for v in grouped.values())
    counts = {t: len(v) for t, v in grouped.items()}

    lines: list[str] = []
    window = f"since {since.strftime('%Y-%m-%d')} ({now.strftime('%Y-%m-%d %H:%M UTC')})"
    if tag:
        window += f" — tag={tag}"
    lines.append(f"# Mem export — {window}")
    lines.append("")
    lines.append(f"**Window:** {since.isoformat()} → {now.isoformat()}")
    lines.append(f"**Total entries:** {total}")
    # Display labels avoid naive pluralization ("feedbacks" wrong).
    _DISPLAY_PLURAL = {"decision": "decisions", "feedback": "feedback", "pattern": "patterns"}
    type_summary = ", ".join(f"{_DISPLAY_PLURAL[t]} {counts[t]}" for t in grouped)
    lines.append(f"**By type:** {type_summary}")
    # Excluded-types disclaimer — surfaces selection bias to downstream consumers.
    _ALL_KNOWN_TYPES = {"decision", "feedback", "pattern", "project", "architecture",
                        "incident", "reference"}
    excluded = sorted(_ALL_KNOWN_TYPES - set(grouped.keys()))
    if excluded:
        lines.append(f"**Excluded types:** {', '.join(excluded)} (not exported by this tool)")
    lines.append(
        "**Note:** This is a curated trace — entries outside the window or "
        "of an excluded type are not present. Do not treat as an exhaustive corpus dump."
    )
    lines.append("")

    if total == 0:
        lines.append("No entries in window.")
        return "\n".join(lines) + "\n"

    type_headers = {"decision": "Decisions", "feedback": "Feedback", "pattern": "Patterns"}
    for t, entries in grouped.items():
        if not entries:
            continue
        lines.append(f"## {type_headers[t]} ({len(entries)})")
        lines.append("")
        for e in entries:
            lines.append(f"### {e['key']}")
            lines.append("")
            tags = e.get("tags") or ""
            lines.append(f"**Tags:** {tags}")
            created = (e.get("created_at") or "")[:10]
            updated = (e.get("updated_at") or "")[:10]
            lines.append(f"**Created:** {created}")
            if updated and updated != created:
                lines.append(f"**Updated:** {updated}")
            lines.append("")
            lines.append((e.get("content") or "").rstrip())
            lines.append("")
            lines.append("---")
            lines.append("")
    return "\n".join(lines) + "\n"


def run(since_spec: str, include: str, exclude: str, tag: str, out: str | None) -> int:
    """Entrypoint. Returns shell exit code."""
    try:
        since = parse_since(since_spec)
        types = parse_types(include, exclude)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    grouped = collect(types, since, tag)
    artifact = render(grouped, since, tag)
    total = sum(len(v) for v in grouped.values())
    if out:
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(artifact)
        except OSError as e:
            print(f"write failed: {out}: {e}", file=sys.stderr)
            return 1
        print(f"wrote {out} ({total} entries)")
    else:
        sys.stdout.write(artifact)
    return 0
