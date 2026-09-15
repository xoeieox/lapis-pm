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

Weekly briefs additionally render a "Locality" bucket (locality-ledger-chain-v0,
Leg 2) — % local, cost-class split, and fallback breakdown read from
agents_core.locality.summarize() (Leg 1's per-call ledger). Lazily imported;
degrades to no section if Leg 1 is unavailable. A stale ledger (no record in
24h) renders a health warning instead of a percentage — silence must never
read as a good week. A reading below threshold deposits a Desk gem per the
ratified OK->BAD state machine (fire on crossing, silent while BAD, escalate
once per streak, re-arm on recovery) — see _evaluate_locality_gem.

Daily briefs additionally render a "Bundle autodispatch enforce" bucket
(lapis-pm-bundle-autodispatch-enforce-v0, Design 3) — what bound (via infra
retry or salvage), what salvaged (dropped items + PR comment owed), what
faulted, and what deferred and why, verbatim, since start_ts. Reads
hold_shadow's enforce-outcomes.jsonl directly; degrades to no section on any
read failure or when the file has no records in-window. Weekly-omitted
(mirroring Climate/Locality's daily-omitted symmetry in reverse) — see
_read_autodispatch.

Daily briefs additionally render a "Night dead-man" bucket
(night-deadman-floor-v0, Leg 1) — the S1 dead-man's artifact output
(findings + ok-artifacts) from /data/slots/night-deadman/*.json. The
ok-artifact absence is the I3 dead-deadman liveness proof (the F4 fix —
the morning-plate reader detects a missing ok-artifact). Weekly-omitted
(mirroring Climate/Locality's daily-omitted symmetry in reverse) — see
_read_night_deadman.

Temporal compression hierarchy (Gardener observations only):
  daily cadences (morning/afternoon/live) → single latest gardener/derived
  entry, capped at 10 observations ("weather today")
  weekly                                 → all entries from the trailing
  7 days, uncapped, fed to the Weekly Arc synthesis ("weather pattern")

Models:
  morning / afternoon / live → call_llm (qwen3.6-35b-a3b, local GPU)
  weekly                     → call_gw_agent (local seat, gravitywell-slot1)

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

import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from agents_core.room_paths import room_path, room_str

logger = logging.getLogger(__name__)

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
B_LOCALITY = "Locality"
B_AUTODISPATCH = "Bundle autodispatch enforce"
B_NIGHT_DEADMAN = "Night dead-man"

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

# Per-family prefix scans replace the tag-wide scan (state-brief-inflight-window-v0).
# A tag-wide limit truncates the OLDEST in-flight targets first, i.e. hides the work that
# has been waiting longest. Each limit is >= 5x the live count measured 2026-08-17
# (dispatched 967, chain 194, thread 54, outstanding-brief 0) and mirrors the
# _LANDED_PREFIX_LIMIT pattern above.
_INFLIGHT_DISPATCHED_LIMIT = 5_000
_INFLIGHT_CHAIN_LIMIT = 5_000
_CAPTURED_THREAD_LIMIT = 5_000
_AWAITING_OUTSTANDING_LIMIT = 5_000

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
    Built/Notable ratifications use list_all(tag="lapis-pm", since=...) +
    client-side prefix filter. In flight/Captured/Awaiting your call use one
    list_by_prefix() scan per family (state-brief-inflight-window-v0) +
    client-side tag filter, since list_by_prefix has no tag parameter.
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
    # scan (limit well above the current 851 dual-key count) rather than a
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

    # In-flight/Captured/Awaiting all used to come from one tag-wide
    # list_all(tag="lapis-pm", limit=1_000) scan (state-brief-inflight-window-v0).
    # That scan is updated_at-DESC, so a 1,000-row cap silently truncates the
    # OLDEST in-flight targets first — the direction a decision surface must
    # never move, and one that shrinks further as the machinery gets busier.
    # Replaced with one list_by_prefix() per family (each with its own named
    # limit, see above) so no family's window can shrink because another
    # family got noisy. None of these four prefixes overlap EXHAUST_PREFIXES
    # today (elevator/, weather/, router/gw-review-divergence/ —
    # mem_exhaust.py:72-76), so list_by_prefix's sibling-exhaust merge is inert
    # here; routing any of them to exhaust later would change brief semantics.
    # Each family scan degrades to empty on its own failure (matches the
    # landed-scan degrade pattern immediately above) so one family's mem
    # trouble never blocks the other three or the rest of the brief. A family
    # that comes back at exactly its limit logs a warning naming the family
    # and the limit — a future truncation must be attributable, not silent.
    all_keys: list[dict] = []
    for _prefix, _limit in (
        ("pm/dispatched/", _INFLIGHT_DISPATCHED_LIMIT),
        ("chain/", _INFLIGHT_CHAIN_LIMIT),
        ("thread/", _CAPTURED_THREAD_LIMIT),
        ("pm/outstanding-brief/", _AWAITING_OUTSTANDING_LIMIT),
    ):
        try:
            _entries = mem.list_by_prefix(_prefix, limit=_limit)
        except Exception:
            logger.warning(
                "_read_buckets: %s scan failed; degrading to empty for this family",
                _prefix,
            )
            continue
        if len(_entries) == _limit:
            logger.warning(
                "_read_buckets: %s scan hit its limit (%d) — results may be truncated",
                _prefix, _limit,
            )
        all_keys.extend(_entries)

    # list_by_prefix has no tag parameter (key LIKE only — mem.py:233,253); the
    # old list_all(tag="lapis-pm") call did this filtering, so restore it here
    # or unrelated rows leak in (measured 2026-08-17: 25 of 54 thread/ rows are
    # Zephyrium/research threads, not lapis-pm; 1 of 968 pm/dispatched/ rows).
    all_keys = [
        e for e in all_keys
        if "lapis-pm" in [t.strip() for t in e.get("tags", "").split(",")]
    ]

    # list_by_prefix returns key-ascending (mem.py:234); list_all returned
    # updated_at DESC (mem.py:226). Re-sort the merged, heterogeneous set to
    # (updated_at DESC, key ASC) — a stable ascending-key sort followed by a
    # stable descending-updated_at sort — so keys that were visible under the
    # old window render in the same relative order, and newly-visible keys
    # (the ones the old window dropped) interleave at their true updated_at
    # position. That is the point of the fix, not a regression.
    all_keys.sort(key=lambda e: e.get("key", ""))
    all_keys.sort(key=lambda e: e.get("updated_at", ""), reverse=True)

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

    # --- Locality: % local / cost-class / fallback bucket (weekly cadence only) ---
    # A brief must never fail because the ledger's source is unavailable.
    try:
        locality_items = _read_locality(start_ts, period=period)
    except Exception:
        locality_items = []

    # --- Bundle autodispatch enforce (daily cadence only) ---
    # A brief must never fail because hold-shadow is unavailable.
    try:
        autodispatch_items = _read_autodispatch(start_ts, period=period)
    except Exception:
        autodispatch_items = []

    # --- Night dead-man (daily cadence only) ---
    # A brief must never fail because the dead-man artifacts are unavailable.
    try:
        night_deadman_items = _read_night_deadman(start_ts, period=period)
    except Exception:
        night_deadman_items = []

    return {
        B_BUILT: built_items,
        B_RATIFICATIONS: ratification_items,
        B_IN_FLIGHT: in_flight_items,
        B_CAPTURED: captured_items,
        B_AWAITING: awaiting_items,
        B_GARDENER: gardener_items,
        B_CLIMATE: climate_items,
        B_LOCALITY: locality_items,
        B_AUTODISPATCH: autodispatch_items,
        B_NIGHT_DEADMAN: night_deadman_items,
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


def _read_arc_climate(
    start_ts: datetime, *, period: str = "daily",
    arc_source: str = "prose",
    prefix: str = "arc/",
) -> list[dict]:
    """Reconcile each arc's declared NEXT against ground truth and classify
    it. `arc_source` selects where the declared/observed state comes from --
    "prose" (default, on-disk arc-doc parsing, unchanged this unit) or
    "registry" (arc_registry.py's structured mem.db rows). Additive per D4:
    the reconciler keeps prose as the behavior under test; registry is off
    by default and does not change prose output.

    `prefix` selects which mem-key prefix the "registry" source reads from
    (e.g. a scratch prefix for a DoD 14-15 verification run). Ignored by the
    "prose" source. Defaults to arc_registry.DEFAULT_PREFIX's value ("arc/").
    """
    if period != "weekly":
        return []
    if arc_source == "registry":
        from . import arc_registry
        return arc_registry.read_arc_climate_from_registry(prefix)
    return _read_arc_climate_from_prose(start_ts)


def _read_arc_climate_from_prose(start_ts: datetime) -> list[dict]:
    """Reconcile each on-disk arc-doc's declared NEXT against ground truth and
    classify it. Deterministic reader — no LLM call; the weekly prose stage
    narrates the returned bullets in narrative-mirror / debt-of-time voice.

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
# Locality bucket (locality-ledger-chain-v0, Leg 2)
# ---------------------------------------------------------------------------
#
# Reads agents_core.locality.summarize() / is_ledger_healthy() (Leg 1, landed
# 2026-07-28 as agents-core-locality-ledger-v0). agents-core is NOT a declared
# dependency of lapis-pm -- always import it lazily and degrade to [] on
# ImportError, exactly like every other bucket source degrades on failure.

_LOCALITY_PCT_FLOOR_DEFAULT = 50.0
_LOCALITY_FALLBACK_CEILING_DEFAULT = 25
_LOCALITY_ESCALATION_WEEKS_DEFAULT = 4
_LOCALITY_STATE_KEY = "pm/locality/gem-state"
_LOCALITY_NEVER_WRITTEN_MARKER = "no records ever written"
_WEAVER_BASE_DEFAULT = "http://203.0.113.10:8403"


def _weaver_base_url() -> str:
    """Resolve weaver base URL: WEAVER_BASE_URL > WEAVER_BIND_PORT > BRIX default.
    Mirrors brief_gem._weaver_base_url's resolution order (kept local rather
    than imported -- locality gems are a distinct, non-reconciled write path)."""
    base = os.environ.get("WEAVER_BASE_URL")
    if base:
        return base.rstrip("/")
    port = os.environ.get("WEAVER_BIND_PORT")
    if port:
        return f"http://127.0.0.1:{port}"
    return _WEAVER_BASE_DEFAULT


def _locality_thresholds() -> dict:
    """Active threshold values, env-overridable. Conservative defaults --
    there is no prior locality data at ship time, so the first
    LOCALITY_ESCALATION_WEEKS (default 4) BAD readings are calibration, not
    judgment (see PR body)."""
    return {
        "pct_local_floor": float(os.environ.get(
            "LOCALITY_PCT_LOCAL_FLOOR", _LOCALITY_PCT_FLOOR_DEFAULT)),
        "fallback_ceiling": int(os.environ.get(
            "LOCALITY_FALLBACK_CEILING", _LOCALITY_FALLBACK_CEILING_DEFAULT)),
        "escalation_weeks": int(os.environ.get(
            "LOCALITY_ESCALATION_WEEKS", _LOCALITY_ESCALATION_WEEKS_DEFAULT)),
    }


def _locality_is_bad(summary: dict, thresholds: dict) -> bool:
    pct = summary.get("pct_local")
    fallback_count = summary.get("fallback_count", 0) or 0
    if pct is not None and pct < thresholds["pct_local_floor"]:
        return True
    if fallback_count > thresholds["fallback_ceiling"]:
        return True
    return False


def _format_locality_lines(summary: dict) -> list[str]:
    """Spoken-aloud lines for the Locality bucket -- no tables, spelled-out
    numbers (this block lands in the weekly TTS episode via tts_episode.py).
    A ledger that is healthy but reports 0% local still renders loudly here
    (total > 0 is the only gate); only a genuinely empty ledger is omitted
    upstream in _read_locality."""
    total = summary.get("total", 0) or 0
    if total == 0:
        return ["The locality ledger is healthy but recorded no calls this week."]

    pct = summary.get("pct_local") or 0.0
    lines = [f"{pct:.0f} percent of calls this week were served locally."]

    by_cost_class = summary.get("by_cost_class") or {}
    if by_cost_class:
        parts = ", ".join(f"{count} {cls}" for cls, count in sorted(by_cost_class.items()))
        lines.append(f"Breakdown by cost class: {parts}.")

    fallback_count = summary.get("fallback_count", 0) or 0
    lines.append(f"{fallback_count} calls fell back to a paid model this week.")

    by_reason = summary.get("by_fallback_reason") or {}
    if by_reason:
        top = sorted(by_reason.items(), key=lambda kv: -kv[1])[:3]
        reasons = ", ".join(f"{reason} ({count})" for reason, count in top)
        lines.append(f"Top fallback reasons: {reasons}.")

    paid_cost_usd = summary.get("paid_cost_usd")
    if paid_cost_usd is not None:
        lines.append(f"Paid spend recorded this week: {paid_cost_usd:.2f} dollars.")

    lines.append(
        "Note: roughly twenty call sites bypass this ledger's seams entirely "
        "and are all local, so the true percent local is at least the figure "
        "reported here."
    )
    return lines


def _default_locality_state() -> dict:
    return {
        "state": "ok",
        "consecutive_bad": 0,
        "escalated": False,
        "open_gem_id": None,
        "last_pct_local": None,
        "last_fallback_count": None,
    }


def _load_locality_state(mem) -> dict:
    """Persisted in mem under pm/locality/gem-state (the brief already reads
    mem via _mem() elsewhere in this module). Defaults to a fresh 'ok' state
    on any read/parse failure -- see _evaluate_locality_gem's docstring for
    what that means for the firing rule if state is ever lost."""
    rec = mem.get(_LOCALITY_STATE_KEY)
    content = rec.get("content") if rec else None
    if not content:
        return _default_locality_state()
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return _default_locality_state()
    state = _default_locality_state()
    if isinstance(data, dict):
        state.update({k: v for k, v in data.items() if k in state})
    return state


def _save_locality_state(mem, state: dict) -> None:
    mem.set(_LOCALITY_STATE_KEY, json.dumps(state, ensure_ascii=False),
             tags=["lapis-pm", "locality"])


def _locality_transition(prior: dict, is_bad: bool, escalation_weeks: int) -> tuple[dict, str | None]:
    """Ratified firing rule (locality-ledger-chain-v0, Erah 2026-07-28) --
    NOT a content dedup, a state machine:

      fire on the OK->BAD transition ("crossing")
      stay silent while it remains BAD
      escalate exactly once per BAD streak, at the Nth consecutive BAD
      reading ("escalation")
      re-arm on recovery to OK

    Returns (new_state, fire) where fire is None, "crossing", or "escalation".
    """
    new_state = dict(prior)
    fire: str | None = None
    if is_bad:
        if prior.get("state") != "bad":
            new_state["state"] = "bad"
            new_state["consecutive_bad"] = 1
            new_state["escalated"] = False
            fire = "crossing"
        else:
            new_state["consecutive_bad"] = prior.get("consecutive_bad", 0) + 1
            if not prior.get("escalated") and new_state["consecutive_bad"] >= escalation_weeks:
                new_state["escalated"] = True
                fire = "escalation"
    else:
        new_state["state"] = "ok"
        new_state["consecutive_bad"] = 0
        new_state["escalated"] = False
    return new_state, fire


def _locality_gem_text(kind: str, summary: dict, thresholds: dict, prior_state: dict) -> dict:
    """Plain, factual gem payload -- ratified 2026-07-28: Erah sided with the
    Mirror Council stand-aside against the majority's escalating/accusatory
    narrative. No surprise->accusation->verdict progression, no 'Cloud-
    Dependent by default' burden shift. Just the number, the delta from the
    prior reading, the top fallback reasons, and one direct question. This
    text is spoken aloud in the weekly TTS episode.

    Active threshold values are recorded in the context block on every fire
    (provenance, not prevention -- a later reading of the gem shows whether
    the line moved, rather than the code trying to guard against retuning)."""
    pct = summary.get("pct_local") or 0.0
    fallback_count = summary.get("fallback_count", 0) or 0
    prior_pct = prior_state.get("last_pct_local")
    if prior_pct is None:
        delta_str = "no prior reading to compare against"
    else:
        delta_str = f"{pct - prior_pct:+.1f} points from last week's {prior_pct:.1f} percent"

    by_reason = summary.get("by_fallback_reason") or {}
    top_reasons = sorted(by_reason.items(), key=lambda kv: -kv[1])[:3]
    reasons_str = ", ".join(f"{k} ({v})" for k, v in top_reasons) if top_reasons else "none recorded"

    if kind == "escalation":
        weeks = prior_state.get("consecutive_bad", thresholds["escalation_weeks"]) + 1
        title = f"Locality has stayed below the floor for {weeks} weeks running"
        ask = "Fix the fallback defaults, or adjust the threshold?"
    else:
        title = "Locality reading crossed below the floor this week"
        ask = "Is this expected, or should the fallback defaults change?"

    why = (
        f"{pct:.1f} percent local this week, {delta_str}. "
        f"Fallback count: {fallback_count}. Top fallback reasons: {reasons_str}."
    )

    context = [
        {
            "label": "This week",
            "lines": [
                f"% local: {pct:.1f}",
                f"Fallback count: {fallback_count}",
                f"Top fallback reasons: {reasons_str}",
            ],
        },
        {
            "label": "Active thresholds",
            "lines": [
                f"Floor: {thresholds['pct_local_floor']}% local",
                f"Fallback ceiling: {thresholds['fallback_ceiling']} per week",
                f"Escalation: {thresholds['escalation_weeks']} consecutive weeks",
            ],
        },
    ]

    return {
        "title": title[:200],
        "ask": ask[:500],
        "why": why,
        "context": context,
        "options": [],
        "agent": "locality-ledger",
        "origin": "from · weekly brief",
    }


def _post_locality_gem(payload: dict) -> str | None:
    """POST /v0/decision-gems. Fail-soft: weaver unreachable or any error ->
    logs a WARNING, returns None. Never raises -- a Weaver outage must not
    stop the weekly brief from being written."""
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("WEAVER_BASE_URL"):
        return None  # under pytest with no explicit (mock/test) weaver target -> do NOT hit live prod
    try:
        import httpx
        base = _weaver_base_url()
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{base}/v0/decision-gems", json=payload)
            resp.raise_for_status()
            return resp.json().get("gem_id")
    except Exception as exc:
        logger.warning("locality: gem deposit failed (weaver unreachable or error): %s", exc)
        return None


def _supersede_locality_gem(gem_id: str, reason: str) -> None:
    """POST /v0/decision-gems/{gem_id}/supersede. Best-effort -- swallows any
    failure (404/409/network) with a WARNING; superseding is housekeeping,
    not load-bearing for the new gem's deposit."""
    if not gem_id:
        return
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("WEAVER_BASE_URL"):
        return
    try:
        import httpx
        base = _weaver_base_url()
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{base}/v0/decision-gems/{gem_id}/supersede",
                json={"reason": reason, "by": "lapis-pm:locality-ledger"},
            )
    except Exception as exc:
        logger.warning("locality: gem supersede failed for %s: %s", gem_id, exc)


def _evaluate_locality_gem(summary: dict) -> None:
    """Decide whether this week's reading fires a Desk gem per the ratified
    state machine, and deposit it if so. State (current OK/BAD, consecutive-
    BAD count, escalation flag, open gem id, last reading) is persisted in
    mem at pm/locality/gem-state.

    If that state is lost (mem entry missing/corrupt), it re-derives from a
    fresh 'ok' baseline: a genuinely-OK week stays silent as normal, but a
    week that is still BAD will look like a fresh OK->BAD crossing and fire
    again -- and since the old open_gem_id is lost too, any still-open prior
    gem cannot be superseded (a possible duplicate open gem, self-healing on
    the next transition). This is a deliberate simplicity tradeoff: a small
    mem key beside the brief output is judged reliable enough not to warrant
    a second, file-based durability path for a weekly-cadence signal.

    Never raises -- wrapped by the caller (_read_locality) with a WARNING.
    """
    thresholds = _locality_thresholds()
    is_bad = _locality_is_bad(summary, thresholds)

    mem = _mem()
    prior = _load_locality_state(mem)
    new_state, fire = _locality_transition(prior, is_bad, thresholds["escalation_weeks"])

    logger.info(
        "locality: pct_local=%.1f fallback_count=%s is_bad=%s fire=%s",
        summary.get("pct_local") or 0.0, summary.get("fallback_count", 0), is_bad, fire,
    )

    if fire:
        payload = _locality_gem_text(fire, summary, thresholds, prior)
        prior_gem_id = prior.get("open_gem_id")
        if prior_gem_id:
            _supersede_locality_gem(prior_gem_id, reason=f"superseded by new {fire} reading")
        gem_id = _post_locality_gem(payload)
        new_state["open_gem_id"] = gem_id or prior_gem_id

    new_state["last_pct_local"] = summary.get("pct_local")
    new_state["last_fallback_count"] = summary.get("fallback_count", 0)
    _save_locality_state(mem, new_state)


def _read_locality(start_ts: datetime, *, period: str = "weekly") -> list[str]:
    """Locality bucket (locality-ledger-chain-v0, Leg 2): % local, cost-class
    split, and fallback breakdown from agents_core.locality's per-call ledger
    (Leg 1, landed). Weekly-only, mirroring Climate.

    Silence and zero are different findings and must render differently:
      - agents_core.locality unimportable, or is_ledger_healthy()/summarize()
        raise, or the ledger has literally never been written -> [] (no
        section at all -- same as an absent Leg 1 install; the brief still
        generates).
      - the ledger exists but has gone stale (no record in 24h) -> a health
        warning line, no percentage. An empty week must never read as a
        good week.
      - the ledger is healthy and reports 0% local -> that 0% renders,
        loudly (see _format_locality_lines).

    A healthy reading also runs the threshold/gem evaluation as a side
    effect (weekly-cadence only, matching the ratified DoD) -- wrapped in
    its own try/except so a Weaver outage or gem-logic bug never blocks the
    brief.
    """
    if period != "weekly":
        return []

    try:
        from agents_core.locality import summarize, is_ledger_healthy
    except ImportError:
        return []

    try:
        healthy, health_reason = is_ledger_healthy()
    except Exception:
        return []

    if not healthy:
        if _LOCALITY_NEVER_WRITTEN_MARKER in health_reason:
            return []  # never written at all -- treat as no data, not silence
        return [
            f"The locality ledger has gone silent: {health_reason}. "
            "Treat this as missing data, not a good week -- no percentage below."
        ]

    try:
        summary = summarize(since=start_ts)
    except Exception:
        return []

    lines = _format_locality_lines(summary)

    try:
        _evaluate_locality_gem(summary)
    except Exception as exc:
        logger.warning("locality: gem evaluation failed: %s", exc)

    return lines


# --- Bundle autodispatch enforce (lapis-pm-bundle-autodispatch-enforce-v0,
# Design 3): what bound-via-enforce, salvaged, faulted, and deferred, since
# start_ts. Daily-cadence only (morning/afternoon/live) — the autodispatch
# timer fires nightly at 05:15 LA and Erah's ruling names the morning brief
# specifically as the surface ("it binds for real, and reports in the
# brief"). Reads directly from hold_shadow's enforce-outcomes.jsonl
# (Design 3's new record type); a plain proceed-to-bind bundle bind that
# never touched the classifier emits no enforce record and is already
# visible via the existing Built/In flight buckets — this section adds
# exactly the enforce-classification information that was invisible before.
# ---------------------------------------------------------------------------

def _read_autodispatch(start_ts: datetime, *, period: str = "daily") -> list[str]:
    """Autodispatch bucket: reads enforce-outcomes.jsonl since start_ts and
    renders what bound (via infra retry or salvage), what salvaged (items
    dropped + PR comment owed), what faulted, and what deferred and why —
    verbatim, per Design 3. Never raises; degrades to [] on any failure so
    a hold-shadow outage never blocks the brief. Daily-cadence only,
    mirroring Climate/Locality's weekly-only symmetry in reverse."""
    if period == "weekly":
        return []

    try:
        from . import hold_shadow as _hold_shadow
    except ImportError:
        return []

    try:
        path = _hold_shadow.enforce_outcomes_path()
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn/partial line never aborts the brief
    except Exception:
        return []

    recent = [
        rec for rec in records
        if (ts := _parse_iso_ts(rec.get("ts_utc"))) is not None and ts >= start_ts
    ]
    if not recent:
        return []

    lines: list[str] = []
    for rec in recent:
        spec = rec.get("spec", "?")
        enforce_class = rec.get("enforce_class", "?")
        action = rec.get("enforce_action", "?")
        ground = (rec.get("verbatim_grounds") or "").strip()
        ground_snippet = ground[:200] + ("…" if len(ground) > 200 else "")

        if action in ("retried_then_bound", "salvage_bound"):
            lines.append(f"BOUND (via {enforce_class}): {spec} — {action}")
        elif enforce_class == "salvage":
            dropped = rec.get("salvage_dropped_items") or []
            dropped_desc = ", ".join(
                f"{d.get('debt_id', '?')} (confidence={d.get('confidence', '?')}, "
                f"source_pr=#{d.get('source_pr', '?')}, PR comment owed)"
                for d in dropped
            ) or "(no items recorded)"
            lines.append(
                f"SALVAGED: {spec} — dropped [{dropped_desc}] — outcome={action} "
                f"— ground: {ground_snippet}"
            )
        elif enforce_class == "infra":
            lines.append(f"FAULTED: {spec} — {ground_snippet}")
        else:
            lines.append(f"DEFERRED: {spec} — {action} — ground: {ground_snippet}")

    return lines


# ---------------------------------------------------------------------------
# Night dead-man bucket (night-deadman-floor-v0, Leg 1)
# ---------------------------------------------------------------------------
#
# Reads /data/slots/night-deadman/*.json — the S1 dead-man's artifact output
# (findings + ok-artifacts). The I3 invariant: a dead dead-man must be the
# loudest failure class. The ok-artifact is the dead-man's own liveness proof;
# its absence is the timer-death / service-crash detector (the F4 fix — the
# morning-plate reader detects a missing ok-artifact). Daily-cadence only,
# mirroring Climate/Locality's weekly-only symmetry in reverse.

_NIGHT_DEADMAN_DIR = Path("/data/slots/night-deadman")


def _read_night_deadman(start_ts: datetime, *, period: str = "daily") -> list[str]:
    """Night dead-man bucket: reads /data/slots/night-deadman/*.json since
    start_ts and renders the dead-man's findings + ok-artifact status.

    I3 dead-deadman liveness proof: the ok-artifact (ok-*.json) is written
    by night_deadman.py when all checks pass. Its absence is the timer-death
    / service-crash detector — the F4 fix names this reader as the mechanism
    that detects a missing ok-artifact. A missing ok-artifact in the window
    is rendered as a [CRITICAL] line; a present ok-artifact is rendered as a
    [OK] line. Findings (violation pages) are rendered as [FINDING] lines.

    Never raises; degrades to [] on any failure so a dead-man artifact
    outage never blocks the brief. Daily-cadence only, mirroring
    Climate/Locality's weekly-only symmetry in reverse.
    """
    if period == "weekly":
        return []

    try:
        if not _NIGHT_DEADMAN_DIR.is_dir():
            return []
    except Exception:
        return []

    lines: list[str] = []
    ok_artifact_found = False

    try:
        for path in sorted(_NIGHT_DEADMAN_DIR.glob("*.json")):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if mtime < start_ts:
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(data, dict):
                continue

            # ok-artifact: the dead-man's own liveness proof
            if path.name.startswith("ok-"):
                ok_artifact_found = True
                verdict = data.get("verdict", "ok")
                ts = data.get("ts_utc", mtime.isoformat())
                lines.append(f"[OK] dead-man all-clear {ts} — {verdict}")
            else:
                # findings: violation pages
                verdict = data.get("verdict", "finding")
                ts = data.get("ts_utc", mtime.isoformat())
                severity = data.get("severity", "HIGH")
                summary = data.get("summary", path.name)
                lines.append(f"[FINDING] {severity} {ts} — {summary}")
    except Exception:
        return []

    # I3: the ok-artifact absence is the timer-death / service-crash detector.
    # If no ok-artifact was found in the window, the dead-man itself may be
    # dead (timer-death or service-crash). This is the F4 fix — the
    # morning-plate reader detects a missing ok-artifact.
    if not ok_artifact_found:
        lines.insert(0, "[CRITICAL] no ok-artifact in window — dead-man may be dead (timer-death or service-crash)")

    return lines


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
        try:
            from agents_core.gw_agent import call_gw_agent
            result = call_gw_agent(
                prompt=prompt,
                system=WEEKLY_SYSTEM,
                writeable=False,
                json_mode=False,
                max_steps=1,
                timeout=300,
                on_wake_fail="skip",
            )
        except Exception as exc:
            logger.warning("state_brief: call_gw_agent error: %s", exc)
            result = None
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
