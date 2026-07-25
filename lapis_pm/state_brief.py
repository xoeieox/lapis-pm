"""State brief generator for lapis-pm.

Generates scheduled state-of-work briefs (morning, afternoon, weekly) and
supports a --live debug mode. Writes output to /srv/lapis/briefs/ and updates
latest-<period>.md symlinks.

IMPORTANT: Do NOT import from lapis_pm.brief — that module is a different
abstraction (PM authority-gate brief synthesiser) and must not be modified.

Bucket ordering (daily cadences — morning/afternoon/live):
  Built (since <start>)
  Notable ratifications (since <start>)
  In flight
  Captured — not yet built
  Awaiting your call
  Gardener Cross-Cutting Observations

Weekly briefs omit the Gardener bucket header, but a 7-day Gardener
observation window is appended as a raw data block for Weekly Arc
synthesis (see state_brief_prompts.py).

Weekly briefs additionally render a "Climate" bucket (gardener-arc-climate-v0,
Unit 1) — a deterministic arc-reconciler that joins each on-disk lapis_state
arc-doc's declared NEXT against ground truth (Forgejo PRs, deploy-inventory,
systemd timers, weaver digests) and classifies it as silently-advanced,
gone-quiet, or unresolvable. Omitted entirely for daily cadences (v0 is
weekly-only) and omitted from weekly output when every arc is moving.

Temporal compression hierarchy (Gardener observations only):
  daily cadences (morning/afternoon/live) → single latest gardener/derived
  entry, capped at 10 observations ("weather today")
  weekly                                 → all entries from the trailing
  7 days, uncapped, fed to the Weekly Arc synthesis ("weather pattern")

Models:
  morning / afternoon / live → call_llm (qwen3.6-35b-a3b, local GPU)
  weekly                     → call_claude_cli(model="sonnet") via Max sub

Dry-run gate:
  LAPIS_BRIEF_DRY_RUN=1 skips the LLM call and writes a placeholder.
  Daily cadences get all six bucket headings; weekly gets the five
  bucket headings plus an appended Gardener 7-day data block. Used by
  smoke.sh.

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

from agents_core.room_paths import room_path, room_str

PACIFIC = ZoneInfo("America/Los_Angeles")
BRIEFS_ROOT = room_path('briefs')

# Bucket names (canonical — must match state_brief_prompts.BUCKET_ORDER)
B_BUILT = "Built"
B_RATIFICATIONS = "Notable ratifications"
B_IN_FLIGHT = "In flight"
B_CAPTURED = "Captured — not yet built"
B_AWAITING = "Awaiting your call"
B_GARDENER = "Gardener Cross-Cutting Observations"
B_CLIMATE = "Climate"

# Parses flat markdown bullets from gardener/writeback.py:derive_context output, e.g.
# "- [Critical] <text>  (evidence: ...)". Info/Unclassified are filtered out upstream
# and never appear in these mem entries. Evidence is captured (not discarded) so
# it can be preserved in the rendered observation for human investigation.
_GARDENER_BULLET_RE = re.compile(r'^- \[(Critical|Warning)\]\s+(.+?)(?:\s+\(evidence:\s*(.+?)\))?$')
_GARDENER_DERIVED_PREFIX = "gardener/derived/"
_GARDENER_DAILY_CAP = 10

# In-flight landed-join scan limit (brief-inflight-landed-join-v0). Well above
# the live 626 dual dispatched+landed keys so the landed set is never
# truncated and a landed target never mis-counts as in-flight.
_LANDED_PREFIX_LIMIT = 10_000

# --- Arc-Climate reconciler (gardener-arc-climate-v0, Unit 1) ---
# Global staleness threshold — a single named constant, trivially tunable
# (OQ-2 RESOLVED: Facets-endorsed 21 days; per-arc importance-scaling deferred
# to Unit 1.5's mem.db arc-registry schema).
_ARC_STALE_DAYS = 21

_NEXT_LINE_RE = re.compile(r'^\s*(?:\*\*)?NEXT(?:\*\*)?[:=]\s*(.+?)\s*$', re.IGNORECASE)
_NEXT_HEADER_RE = re.compile(r'^\s*(?:#+\s*Next\b|\*\*NEXT\*\*)\s*$', re.IGNORECASE)
_BULLET_LINE_RE = re.compile(r'^\s*[-*]\s+(.+?)\s*$')
# Repo token must contain a hyphen (Lapis repos are consistently
# hyphenated: lapis-pm, agents-core, lapis-engine, ...) so ordinary prose
# words immediately preceding "PR #n" (e.g. "merge PR #219") never get
# mistaken for a repo name; the parenthetical form is the fallback anchor.
_PR_ANCHOR_RE = re.compile(
    r'([A-Za-z][\w]*-[\w.-]*)\s+PR\s*#(\d+)|PR\s*#(\d+)\s*\(([A-Za-z][\w.-]*)\)',
    re.IGNORECASE,
)
_UNIT_ANCHOR_RE = re.compile(r'\b([\w-]+\.(?:timer|service))\b')
_TARGET_ID_RE = re.compile(r'\b([a-z][a-z0-9]*(?:-[a-z0-9]+){2,})\b')
_SILENTLY_ADVANCED_NEXT_RE = re.compile(r'review|merge|land|pm-pr-review', re.IGNORECASE)
_WEAVER_BIN = "/home/user/.local/bin/weaver"


# ---------------------------------------------------------------------------
# Data read helpers
# ---------------------------------------------------------------------------

def _mem():
    """Return a fresh writable MemoryStore instance bound to owned_mem_store."""
    from . import node_identity
    return node_identity.writable_store()


def _read_buckets(start_ts: datetime, *, period: str = "daily") -> dict[str, list[str]]:
    """Read all data sources and return buckets dict.

    All reads are deterministic; performed before any LLM call.
    Uses list_all(tag="lapis-pm", since=...) + client-side prefix filter.
    Does NOT use mem.search() (no since= support).

    Args:
        start_ts: time window start for Built/Ratifications buckets.
        period: the brief cadence (morning/afternoon/live/weekly). Only
            "weekly" is treated specially (7-day Gardener window); every
            other value gets the daily-cadence (single latest entry) window.

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
    # Landed-beats-dispatched join: a pm/dispatched/<tid> is in-flight only if
    # no pm/landed/<tid> exists. Landing is asymmetric — clear_landed_state
    # keeps the dispatched key for audit-query — so key presence alone
    # over-reports. Same precedence rule as the reference oracle,
    # trajectory._derive_node_state ("landed if pm/landed/<tid> exists" wins
    # over any dispatched signal). Built from a dedicated pm/landed/ prefix
    # scan (limit well above the current 626 dual-key count) rather than a
    # per-target mem round-trip inside the loop below. Degrades to an empty
    # landed set on failure (matches the Gardener/Climate degrade pattern
    # below) rather than blocking the whole brief on a mem outage.
    try:
        landed_entries = mem.list_by_prefix("pm/landed/", limit=_LANDED_PREFIX_LIMIT)
    except Exception:
        landed_entries = []
    landed_set = {
        e.get("key", "").removeprefix("pm/landed/")
        for e in landed_entries
        if e.get("key", "").startswith("pm/landed/")
    }

    all_keys = mem.list_all(tag="lapis-pm", limit=1000)
    in_flight_items: list[str] = []
    for entry in all_keys:
        k = entry.get("key", "")
        if k.startswith("pm/dispatched/"):
            tid = k.removeprefix("pm/dispatched/")
            if tid in landed_set:
                continue
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

    # --- Gardener cross-cutting observations ---
    # Daily cadences: latest gardener/derived entry only. Weekly: 7-day window
    # (temporal compression hierarchy — see module docstring). Degrades to []
    # on any mem query failure so a Gardener outage never blocks the brief.
    try:
        gardener_items = _read_gardener_observations(period=period)
    except Exception:
        gardener_items = []

    # --- Climate: arc-reconciler bullets (weekly cadence only — v0) ---
    # Degrades to [] on any catastrophic failure, matching Gardener's guard.
    # Per-source degrade for individual ground-truth signals happens inside
    # _read_arc_climate itself (each source wrapped in its own try/except).
    try:
        climate_entries = _read_arc_climate(start_ts, period=period)
    except Exception:
        climate_entries = []
    climate_items = [entry["text"] for entry in climate_entries]

    return {
        B_BUILT: built_items,
        B_RATIFICATIONS: ratification_items,
        B_IN_FLIGHT: in_flight_items,
        B_CAPTURED: captured_items,
        B_AWAITING: awaiting_items,
        B_GARDENER: gardener_items,
        B_CLIMATE: climate_items,
    }


def _read_gardener_observations(period: str = "daily") -> list[str]:
    """Read Critical/Warning observations from gardener/derived/* mem entries.

    Reads gardener/derived/* mem entries (the structured digest gardener.writeback
    produces), not the on-disk synthesis artifacts. Narration entries
    (gardener/derived/narration-*) are excluded — see gardener/server.py:_is_narration_key —
    since they hold prose, not the flat bullet digest this parses.

    Temporal compression hierarchy: period is normalized to weekly vs.
    non-weekly (every daily cadence — morning/afternoon/live — collapses to
    the same non-weekly behavior; there is no third branch).
      "weekly"    → all entries whose key-embedded date falls in the
                    trailing 7 days, uncapped (Weekly Arc synthesizes the
                    pattern from raw data).
      non-weekly  → the single latest entry, capped at 10 observations.

    Uses entry["value"]/entry["content"] directly from the list_by_prefix()
    result — no redundant mem.get() call per entry.

    Returns [] if no gardener entries exist, all are narration/out-of-window,
    or parsing fails.
    """
    mem = _mem()
    entries = mem.list_by_prefix(_GARDENER_DERIVED_PREFIX, limit=10000)
    entries = [
        e for e in entries
        if not e.get("key", "").removeprefix(_GARDENER_DERIVED_PREFIX).startswith("narration-")
    ]
    if not entries:
        return []

    if period == "weekly":
        # 7-day window: parse the date embedded in the key
        # (gardener/derived/YYYY-MM-DD-HHMM); skip anything malformed.
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)
        relevant = []
        for e in entries:
            date_part = e.get("key", "").removeprefix(_GARDENER_DERIVED_PREFIX)[:10]
            try:
                entry_date = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if entry_date >= cutoff:
                relevant.append(e)
    else:
        # Non-weekly (morning/afternoon/live/daily): single latest entry.
        # Keys are ISO-ish timestamp slugs; descending sort gets the newest.
        relevant = sorted(entries, key=lambda e: e.get("key", ""), reverse=True)[:1]

    if not relevant:
        return []

    relevant = sorted(relevant, key=lambda e: e.get("key", ""), reverse=True)

    observations: list[str] = []
    for entry in relevant:
        content = entry.get("value") or entry.get("content") or ""
        if not content:
            continue
        for line in content.split("\n"):
            match = _GARDENER_BULLET_RE.match(line.strip())
            if not match:
                continue
            urgency, text, evidence = match.group(1), match.group(2).strip(), match.group(3)
            if not text:
                continue
            if evidence:
                observations.append(f"[{urgency}] {text}  (evidence: {evidence})")
            else:
                observations.append(f"[{urgency}] {text}")

    # Weekly: uncapped — the LLM synthesizes patterns from the full window.
    # Non-weekly: cap to avoid prompt bloat.
    return observations if period == "weekly" else observations[:_GARDENER_DAILY_CAP]


# --- Arc-Climate reconciler helpers (gardener-arc-climate-v0, Unit 1) ------


def _parse_declared_next(text: str) -> str | None:
    """First line matching NEXT[:=]..., else the first bullet (or line) under
    a '## Next' / '**NEXT**' marker. None if neither is present."""
    lines = text.splitlines()
    for line in lines:
        m = _NEXT_LINE_RE.match(line)
        if m:
            return m.group(1).strip()
    for i, line in enumerate(lines):
        if _NEXT_HEADER_RE.match(line):
            for follow in lines[i + 1:]:
                if not follow.strip():
                    continue
                bm = _BULLET_LINE_RE.match(follow)
                return (bm.group(1) if bm else follow).strip()
    return None


def _parse_arc_anchors(text: str) -> dict:
    """Regex-extract PR anchors, *.timer/*.service names, and target-id-shaped
    slugs mentioned in an arc-doc. Best-effort, never raises."""
    prs: list[tuple[str | None, int]] = []
    for m in _PR_ANCHOR_RE.finditer(text):
        if m.group(2):
            prs.append((m.group(1), int(m.group(2))))
        else:
            prs.append((m.group(4), int(m.group(3))))
    units = sorted(set(_UNIT_ANCHOR_RE.findall(text)))
    target_ids = sorted(set(_TARGET_ID_RE.findall(text)))
    return {"prs": prs, "units": units, "target_ids": target_ids}


def _parse_iso_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _arc_pr_ground_truth(repo: str, pr_number: int) -> dict | None:
    """get_pr(repo, pr_number) wrapped for per-source degrade. None on any failure.

    Arc-doc anchors carry a bare repo token (never an "owner/repo" prefix — see
    _PR_ANCHOR_RE), so the owner namespace is unknown up front. get_pr defaults
    to the Erah namespace when owner is omitted; agent-managed repos that
    instead live under the lapis org (per feedback/merge-and-deploy-owner-kwarg)
    404 on that first attempt. Retry once against the lapis org before giving
    up, rather than silently dropping the signal for every non-Erah repo."""
    import httpx
    from agents_core.forgejo import LAPIS_ORG, get_pr
    try:
        return get_pr(repo, pr_number)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            try:
                return get_pr(repo, pr_number, owner=LAPIS_ORG)
            except Exception:
                return None
        return None
    except Exception:
        return None


def _arc_deploy_signal(anchors: dict) -> datetime | None:
    """Best-effort: the deploy-inventory status timestamp, if a HIGH finding's
    clone path intersects one of this arc's anchor tokens (target-id or repo).
    None if no match, or the status has no parseable 'generated' timestamp —
    never fabricates a timestamp to avoid a spurious freshness signal."""
    from . import deploy_inventory
    anchor_tokens = {tid for tid in anchors.get("target_ids", [])} | {
        repo for repo, _num in anchors.get("prs", []) if repo
    }
    if not anchor_tokens:
        return None
    try:
        status = deploy_inventory.read_status_json()
        if not status:
            return None
        keys = deploy_inventory.high_finding_keys(status)
    except Exception:
        return None
    for clone_path, _kind in keys:
        if clone_path and any(tok in clone_path for tok in anchor_tokens):
            return _parse_iso_ts(status.get("generated"))
    return None


def _arc_timer_resolves(units: list[str]) -> bool:
    """Best-effort: True if any *.timer/*.service anchor successfully answers
    `systemctl show` (system or user scope) — evidence the anchor is real and
    queryable. The reused property set carries no last-triggered timestamp,
    so a resolving timer only proves ground-truth *availability* (keeps the
    arc out of 'unresolvable') — it never contributes a fabricated freshness
    timestamp, since timer/service units are typically active/waiting at
    nearly all times regardless of whether the arc itself has moved."""
    from . import deploy_inventory
    for unit in units:
        for scope in ("system", "user"):
            try:
                props = deploy_inventory._show_unit(unit, scope)
            except Exception:
                props = None
            if props:
                return True
    return False


def _arc_weaver_signal(slug: str) -> datetime | None:
    """Best-effort thread match: `weaver get <slug> --json` -> latest digest ts.
    Advisory-only per spec — never cited as ratified. None on any failure
    (binary missing, no matching thread, malformed JSON)."""
    import json
    import subprocess
    try:
        result = subprocess.run(
            [_WEAVER_BIN, "get", slug, "--json"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("updated_at", "last_updated", "updated", "timestamp", "digest_ts"):
        ts = _parse_iso_ts(data.get(key))
        if ts:
            return ts
    return None


def _read_arc_climate(start_ts: datetime, *, period: str = "daily") -> list[dict]:
    """Reconcile each on-disk arc-doc's declared NEXT against ground truth and
    classify it. Deterministic reader — no LLM call; the weekly prose stage
    narrates the returned bullets in narrative-mirror / debt-of-time voice.

    Weekly cadence only for v0 (Design decisions): non-weekly periods return
    [] immediately without touching Forgejo/deploy-inventory/systemd/weaver,
    so daily/morning/afternoon/live briefs never pay this network cost.

    Each ground-truth source (Forgejo PR, deploy-inventory, systemd timer,
    weaver digest) is wrapped in its own try/except — a failing source drops
    only that signal; an arc with zero available external signals classifies
    as unresolvable (once past _ARC_STALE_DAYS) rather than aborting the pass.
    Catastrophic failure (e.g. the arc-doc directory itself is unreadable)
    degrades to [] via the caller's try/except in _read_buckets.

    Returns a list of {"slug", "classification", "text"} dicts — "moving"
    arcs (age_days <= _ARC_STALE_DAYS) are omitted entirely (silence is not
    churn), matching the "omit empty section" behaviour DoD requires.
    """
    if period != "weekly":
        return []

    arc_dir = room_path('lapis_state')
    if not arc_dir.exists():
        return []

    now = datetime.now(tz=timezone.utc)
    results: list[dict] = []

    for doc_path in sorted(arc_dir.glob("*.md")):
        try:
            text = doc_path.read_text(encoding="utf-8")
        except OSError:
            continue

        slug = doc_path.stem
        declared_next = _parse_declared_next(text)
        anchors = _parse_arc_anchors(text)

        try:
            doc_mtime = datetime.fromtimestamp(doc_path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            doc_mtime = now

        # external_signals feeds the freshness/age calculation (real
        # timestamps only). has_ground_truth tracks whether *any* source
        # resolved at all — including timer/service anchors, which prove
        # queryability but carry no usable timestamp (see _arc_timer_resolves).
        external_signals: list[datetime] = []
        has_ground_truth = False
        pr_hits: list[tuple[str, int, dict]] = []

        for repo, num in anchors["prs"]:
            if not repo:
                continue
            pr = _arc_pr_ground_truth(repo, num)
            if pr:
                pr_hits.append((repo, num, pr))
                has_ground_truth = True
                ts = _parse_iso_ts(pr.get("merged_at")) or _parse_iso_ts(pr.get("updated_at"))
                if ts:
                    external_signals.append(ts)

        try:
            deploy_ts = _arc_deploy_signal(anchors)
        except Exception:
            deploy_ts = None
        if deploy_ts:
            external_signals.append(deploy_ts)
            has_ground_truth = True

        try:
            timer_resolved = _arc_timer_resolves(anchors["units"])
        except Exception:
            timer_resolved = False
        if timer_resolved:
            has_ground_truth = True

        try:
            weaver_ts = _arc_weaver_signal(slug)
        except Exception:
            weaver_ts = None
        if weaver_ts:
            external_signals.append(weaver_ts)
            has_ground_truth = True

        last_activity = max([doc_mtime, *external_signals])
        age_days = max((now - last_activity).days, 0)

        # --- silently-advanced (Realized Intent): fires on detection,
        # independent of _ARC_STALE_DAYS (Council round-2). ---
        advanced: dict | None = None
        if declared_next and _SILENTLY_ADVANCED_NEXT_RE.search(declared_next):
            for repo, num, pr in pr_hits:
                state = (pr.get("state") or "").lower()
                if pr.get("merged") or state in ("closed", "merged"):
                    merged_at = _parse_iso_ts(pr.get("merged_at")) or _parse_iso_ts(pr.get("updated_at"))
                    delta_repr = str((now - merged_at).days) if merged_at else "?"
                    advanced = {
                        "slug": slug,
                        "classification": "silently-advanced",
                        "text": (
                            f"{slug}: PR #{num} merged {delta_repr}d ago; the arc reads "
                            f"complete. Update the record? [CONTEXT DRIFT: declared NEXT "
                            f"still says '{declared_next}']"
                        ),
                    }
                    break
        if advanced:
            results.append(advanced)
            continue

        if age_days <= _ARC_STALE_DAYS:
            continue  # moving — signal is silence, not churn

        if has_ground_truth:
            results.append({
                "slug": slug,
                "classification": "gone-quiet",
                "text": (
                    f"{slug}: {age_days}d of silence since "
                    f"'{declared_next or 'no declared NEXT'}'. The thread has gone slack "
                    f"— shall we pull it taut again?"
                ),
            })
        else:
            results.append({
                "slug": slug,
                "classification": "unresolvable",
                "text": (
                    f"{slug}: cannot verify — {age_days}d since the arc-doc was touched, "
                    f"no PR/timer/thread anchor."
                ),
            })

    return results


def _arc_docs_since(start_ts: datetime) -> list[str]:
    """Return arc doc filenames modified since start_ts. Best-effort."""
    arc_dir = room_path('lapis_state')
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

    Respects LAPIS_BRIEF_DRY_RUN=1 — returns a placeholder with all six
    bucket headings for daily cadences (five bucket headings plus an
    appended Gardener 7-day data block for weekly) if set.

    For daily periods (morning, afternoon, live), wraps call_llm in a fail-fast
    guard with ~30s wall-clock timeout. On timeout/unreachable, returns atomic
    degraded brief (local sections + marker) instead of hanging.
    """
    from .state_brief_prompts import (
        DAILY_SYSTEM, WEEKLY_SYSTEM,
        build_daily_prompt, build_weekly_prompt,
    )

    if os.environ.get("LAPIS_BRIEF_DRY_RUN") == "1":
        return _dry_run_placeholder(buckets, start_label, period=period)

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
        return _dry_run_placeholder(
            buckets, start_label, period=period,
            tag="*(DEGRADED — StarHouse unreachable)*",
        )

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
        return call_llm(prompt=prompt, system=system, timeout=timeout_sec)

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_do_call)
        result = future.result(timeout=timeout_sec)
        executor.shutdown(wait=False)
        return result if result else None
    except FuturesTimeoutError:
        executor.shutdown(wait=False)
        return None
    except Exception:
        executor.shutdown(wait=False)
        return None


def _dry_run_placeholder(
    buckets: dict[str, list[str]],
    start_label: str,
    period: str = "daily",
    tag: str = "(dry run — bucket headings included)",
) -> str:
    """Generate a structurally complete placeholder with all bucket headings for `period`."""
    from .state_brief_prompts import format_bucket_sections
    sections = format_bucket_sections(buckets, start_label, period=period)
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
    buckets = _read_buckets(start_ts, period=period)

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
