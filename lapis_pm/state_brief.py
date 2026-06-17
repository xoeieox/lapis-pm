"""State brief generator for lapis-pm.

Generates scheduled state-of-work briefs (morning, afternoon, weekly) and
supports a --live debug mode. Writes output to /srv/lapis/briefs/ and updates
latest-<period>.md symlinks.

IMPORTANT: Do NOT import from lapis_pm.brief — that module is a different
abstraction (PM authority-gate brief synthesiser) and must not be modified.

Bucket ordering (all cadences):
  Built (since <start>)
  Notable ratifications (since <start>)
  In flight
  Captured — not yet built
  Awaiting your call

Models:
  morning / afternoon / live → call_llm (qwen3.6-35b-a3b, local GPU)
  weekly                     → call_claude_cli(model="sonnet") via Max sub

Dry-run gate:
  LAPIS_BRIEF_DRY_RUN=1 skips the LLM call and writes a placeholder that
  still includes all five bucket headings. Used by smoke.sh.

Pushover:
  Not fired from state briefs. Outstanding briefs are notified at creation
  time via brief.synthesize(). Scheduled brief generation is informational
  only — visible in claude-view, no phone interrupt.
"""

from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
BRIEFS_ROOT = Path("/srv/lapis/briefs")

# Bucket names (canonical — must match state_brief_prompts.BUCKET_ORDER)
B_BUILT = "Built"
B_RATIFICATIONS = "Notable ratifications"
B_IN_FLIGHT = "In flight"
B_CAPTURED = "Captured — not yet built"
B_AWAITING = "Awaiting your call"


# ---------------------------------------------------------------------------
# Data read helpers
# ---------------------------------------------------------------------------

def _mem():
    """Return a fresh MemoryStore instance."""
    from agents_core.mem import MemoryStore
    return MemoryStore()


def _read_buckets(start_ts: datetime) -> dict[str, list[str]]:
    """Read all data sources and return buckets dict.

    All reads are deterministic; performed before any LLM call.
    Uses list_all(tag="lapis-pm", since=...) + client-side prefix filter.
    Does NOT use mem.search() (no since= support).

    Returns:
        buckets: dict mapping bucket name → list of item strings
    """
    mem = _mem()
    since_str = start_ts.isoformat()

    # --- Built: recently landed targets ---
    landed_keys_raw = mem.list_all(tag="lapis-pm", since=since_str, limit=500)
    built_items: list[str] = []
    for entry in landed_keys_raw:
        k = entry.get("key", "")
        if k.startswith("pm/landed/"):
            tid = k.removeprefix("pm/landed/")
            rec = mem.get(k)
            val = rec.get("value", "") if rec else ""
            built_items.append(f"{tid}: {val}" if val else tid)

    # Arc docs as corroboration — list modified files (best-effort, non-blocking)
    arc_items = _arc_docs_since(start_ts)
    for arc in arc_items:
        label = f"arc-doc: {arc}"
        if label not in built_items:
            built_items.append(label)

    # --- Notable ratifications: decision/* keys since start ---
    decision_keys_raw = mem.list_all(tag="lapis-pm", since=since_str, limit=500)
    ratification_items: list[str] = []
    for entry in decision_keys_raw:
        k = entry.get("key", "")
        if k.startswith("decision/"):
            rec = mem.get(k)
            val = rec.get("value", "") if rec else ""
            label = k.removeprefix("decision/")
            ratification_items.append(f"{label}: {val[:120]}" if val else label)

    # --- In flight: dispatched + chain keys (all time — not time-windowed) ---
    all_keys = mem.list_all(tag="lapis-pm", limit=1000)
    in_flight_items: list[str] = []
    for entry in all_keys:
        k = entry.get("key", "")
        if k.startswith("pm/dispatched/"):
            tid = k.removeprefix("pm/dispatched/")
            rec = mem.get(k)
            val = rec.get("value", "") if rec else ""
            in_flight_items.append(f"{tid}: {val[:120]}" if val else tid)
        elif k.startswith("chain/"):
            chain_id = k.removeprefix("chain/")
            in_flight_items.append(f"chain:{chain_id}")

    # --- Captured — not yet built: thread/* keys ---
    captured_items: list[str] = []
    for entry in all_keys:
        k = entry.get("key", "")
        if k.startswith("thread/"):
            rec = mem.get(k)
            val = rec.get("value", "") if rec else ""
            label = k.removeprefix("thread/")
            captured_items.append(f"{label}: {val[:120]}" if val else label)

    # --- Awaiting your call: pm/outstanding-brief/* keys ---
    awaiting_items: list[str] = []
    for entry in all_keys:
        k = entry.get("key", "")
        if k.startswith("pm/outstanding-brief/"):
            brief_id = k.removeprefix("pm/outstanding-brief/")
            rec = mem.get(k)
            val = rec.get("value", "") if rec else ""
            awaiting_items.append(f"{brief_id}: {val[:120]}" if val else brief_id)

    return {
        B_BUILT: built_items,
        B_RATIFICATIONS: ratification_items,
        B_IN_FLIGHT: in_flight_items,
        B_CAPTURED: captured_items,
        B_AWAITING: awaiting_items,
    }


def _arc_docs_since(start_ts: datetime) -> list[str]:
    """Return arc doc filenames modified since start_ts. Best-effort."""
    arc_dir = Path("/srv/lapis/lapis-state")
    if not arc_dir.exists():
        return []
    results: list[str] = []
    start_mtime = start_ts.timestamp()
    for p in arc_dir.glob("*.md"):
        try:
            if p.stat().st_mtime >= start_mtime:
                results.append(p.name)
        except OSError:
            pass
    return sorted(results)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _generate_prose(period: str, buckets: dict[str, list[str]], start_label: str) -> str:
    """Call the appropriate LLM and return the generated brief body.

    Respects LAPIS_BRIEF_DRY_RUN=1 — returns a placeholder with all five
    bucket headings if set.

    For daily periods (morning, afternoon, live), wraps call_llm in a fail-fast
    guard with ~30s wall-clock timeout. On timeout/unreachable, returns atomic
    degraded brief (local sections + marker) instead of hanging.
    """
    from .state_brief_prompts import (
        DAILY_SYSTEM, WEEKLY_SYSTEM,
        build_daily_prompt, build_weekly_prompt,
    )

    if os.environ.get("LAPIS_BRIEF_DRY_RUN") == "1":
        return _dry_run_placeholder(buckets, start_label)

    if period == "weekly":
        prompt = build_weekly_prompt(buckets, start_label)
        from agents_core.llm import call_claude_cli
        result = call_claude_cli(
            prompt=prompt,
            system=WEEKLY_SYSTEM,
            model="sonnet",
            timeout=300,
        )
    else:
        prompt = build_daily_prompt(buckets, start_label)
        from agents_core.llm import call_llm
        result = _call_llm_with_timeout(prompt, DAILY_SYSTEM, timeout_sec=30)

    if not result:
        # Fallback: placeholder so the file is always structurally valid
        return _dry_run_placeholder(buckets, start_label, tag="(LLM returned empty)")

    return result


def _call_llm_with_timeout(prompt: str, system: str, timeout_sec: int = 30) -> str | None:
    """Call qwen LLM with fail-fast timeout guard.

    Wraps call_llm in a ThreadPoolExecutor to enforce a hard wall-clock limit.
    If the call times out or raises an exception, returns None so the brief
    can degrade gracefully to deterministic local sections.

    Args:
        prompt: The prompt to send to the LLM.
        system: The system message.
        timeout_sec: Wall-clock timeout in seconds (default 30s per spec).

    Returns:
        The LLM result string, or None if timeout/error occurs.
    """
    from agents_core.llm import call_llm

    def _do_call():
        return call_llm(prompt=prompt, system=system, timeout=300)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_call)
            result = future.result(timeout=timeout_sec)
            return result if result else None
    except FuturesTimeoutError:
        return None
    except Exception:
        return None


def _dry_run_placeholder(
    buckets: dict[str, list[str]],
    start_label: str,
    tag: str = "(dry run — bucket headings included)",
) -> str:
    """Generate a structurally complete placeholder with all five bucket headings."""
    from .state_brief_prompts import format_bucket_sections
    sections = format_bucket_sections(buckets, start_label)
    return f"{tag}\n\n{sections}\n"


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _ensure_dirs(period: str) -> Path:
    """Create output directory and return it."""
    if period == "weekly":
        d = BRIEFS_ROOT / "weekly"
    else:
        d = BRIEFS_ROOT / "daily"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _output_path(period: str, now: datetime) -> Path:
    """Compute the output file path for a given period."""
    d = _ensure_dirs(period)
    if period == "weekly":
        # ISO week: YYYY-Www
        week_label = now.strftime("%G-W%V")
        return d / f"{week_label}.md"
    else:
        ts = now.strftime("%Y-%m-%d-%H%M")
        return d / f"{ts}-{period}.md"


def _update_symlink(period: str, target: Path) -> None:
    """Update /srv/lapis/briefs/latest-<period>.md to point at target."""
    link = BRIEFS_ROOT / f"latest-{period}.md"
    # Compute relative path from BRIEFS_ROOT to target for a tidy symlink
    try:
        rel = target.relative_to(BRIEFS_ROOT)
    except ValueError:
        rel = target  # absolute fallback
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(rel)


def _write_brief(path: Path, header: str, body: str) -> None:
    """Write the brief file with a markdown title header."""
    content = f"# {header}\n\n{body}\n"
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_brief(
    period: str,
    start_ts: datetime | None = None,
    *,
    stdout: bool = False,
) -> Path | None:
    """Generate a state brief for the given period.

    Args:
        period: one of "morning", "afternoon", "weekly", "live"
        start_ts: start of the look-back window. Defaults to 24h ago (7d for weekly).
        stdout: if True, print the brief body to stdout instead of writing a file
                (used for --live mode).

    Returns:
        Path to the written brief file, or None for stdout mode.
    """
    now = datetime.now(tz=PACIFIC)

    if start_ts is None:
        if period == "weekly":
            start_ts = datetime.now(tz=timezone.utc) - timedelta(days=7)
        else:
            start_ts = datetime.now(tz=timezone.utc) - timedelta(hours=24)

    # start_label for display in headings
    start_label = start_ts.astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M PT")

    # 1. Read data — all deterministic, before LLM call
    buckets = _read_buckets(start_ts)

    # 2. Call LLM (or dry-run placeholder)
    body = _generate_prose(period, buckets, start_label)

    # 3. Write output
    if stdout or period == "live":
        sys.stdout.write(body)
        if not body.endswith("\n"):
            sys.stdout.write("\n")
        return None

    now_label = now.strftime("%Y-%m-%d %H:%M PT")
    header = f"Lapis {period.capitalize()} Brief — {now_label}"
    out_path = _output_path(period, now)
    _write_brief(out_path, header, body)
    _update_symlink(period, out_path)

    return out_path
