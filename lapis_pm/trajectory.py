"""Trajectory rollup generator — DAG-anchored arc-doc rollups with Qwen narration.

Output tree under /srv/lapis/trajectory/:
  index.json               — deterministic unlock DAG (no LLM)
  per-target/<tid>.json    — Qwen-generated one-liner per landed target
  weekly/<YYYY-Www>.md     — Qwen-generated weekly digest
  monthly/<YYYY-MM>.md     — monthly digest via per-week extractive→synthesize

Storage invariant: /srv/lapis/trajectory/ NOT /srv/lapis/lapis-state/.
Qwen-narrated derivative content stays out of the RoomRAG-indexed corpus.

LLM calls go through agents_core.llm / agents_core.gw_agent:
  call_llm()        — Qwen 3.6 (default)
  call_gw_agent()   — local seat (when LAPIS_TRAJECTORY_MONTHLY_MODEL=haiku;
                      the local route takes precedence per the re-point spec)

Dry-run gate: LAPIS_TRAJECTORY_DRY_RUN=1 skips LLM calls and writes placeholder
one_liner: "(dry run)". Read inside this module at LLM call site.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agents_core.room_paths import room_path, room_str

logger = logging.getLogger(__name__)

PACIFIC = ZoneInfo("America/Los_Angeles")
TRAJECTORY_ROOT = room_path('trajectory')
TARGETS_DIR = room_path('targets')
ARC_DOCS_DIR = room_path('lapis_state')


def _qwen_model_provenance() -> str:
    """Model name to stamp into rollup provenance ('model: ...' fields).

    This is provenance/metadata text written into artifacts, not a request
    payload field - call_llm() itself resolves its own endpoint/model. Report
    what was actually used: LOCAL_LLM_MODEL when set, else the literal
    "server-default" - stamping a hardcoded name here would be false the
    moment the served model changes underneath. Never "unknown" (that reads
    as ignorance; "server-default" truthfully records that model choice was
    delegated to the server). Gate requirement, Mirror Council run
    2026-08-05-101841-86891c.
    """
    return os.environ.get("LOCAL_LLM_MODEL", "server-default")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mem():
    from . import node_identity
    return node_identity.writable_store()


def _now_iso() -> str:
    return datetime.now(PACIFIC).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dry_run() -> bool:
    return os.environ.get("LAPIS_TRAJECTORY_DRY_RUN", "").strip() == "1"


def _use_haiku_for_monthly() -> bool:
    return os.environ.get("LAPIS_TRAJECTORY_MONTHLY_MODEL", "").strip() == "haiku"


def _call_qwen(prompt: str, system: str) -> str | None:
    """Call Qwen via agents_core.llm.call_llm. Returns text or None."""
    if _dry_run():
        return "(dry run)"
    from agents_core.llm import call_llm
    return call_llm(prompt, system=system, timeout=600, temperature=0.5)


def _call_haiku(prompt: str, system: str = "") -> str | None:
    """Call the local seat via call_gw_agent. Returns text or None.

    Legacy name kept for the LAPIS_TRAJECTORY_MONTHLY_MODEL=haiku gate; the
    local route takes precedence per lapis-pm-prose-synthesis-local-repoint-v0.
    """
    if _dry_run():
        return "(dry run)"
    try:
        from agents_core.gw_agent import call_gw_agent
        served_model_out: list = []
        return call_gw_agent(
            prompt=prompt,
            system=system,
            writeable=False,
            json_mode=False,
            max_steps=1,
            timeout=300,
            on_wake_fail="skip",
            served_model_out=served_model_out,
        )
    except Exception as exc:
        logger.warning("trajectory: call_gw_agent error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Target YAML loading
# ---------------------------------------------------------------------------

def _load_all_target_yamls() -> list[dict]:
    """Load all target YAMLs from TARGETS_DIR. Skip non-yaml files and subdirs."""
    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None

    if _yaml is None:
        logger.warning("PyYAML not installed; cannot load target YAMLs")
        return []

    targets: list[dict] = []
    for p in sorted(TARGETS_DIR.glob("*.yaml")):
        try:
            data = _yaml.safe_load(p.read_text())
            if isinstance(data, dict):
                targets.append(data)
        except Exception as e:
            logger.warning("Failed to load target YAML %s: %s", p, e)
    return targets


def _load_landed_mem_entries() -> dict[str, dict]:
    """Return {tid: payload_dict} for all pm/landed/<tid> mem entries."""
    mem = _mem()
    results = mem.list_all(tag="landed", limit=100_000)
    landed: dict[str, dict] = {}
    for r in results:
        key = r.get("key", "")
        if not key.startswith("pm/landed/"):
            continue
        tid = key[len("pm/landed/"):]
        try:
            payload = json.loads(r["content"])
        except Exception:
            payload = {"arc_path": ""}
        landed[tid] = payload
    return landed


# ---------------------------------------------------------------------------
# State flattening (spec table)
# ---------------------------------------------------------------------------

def _derive_node_state(tid: str, target_data: dict, landed: dict[str, dict]) -> str:
    """Flatten lifecycle into one of: landed | dispatched | bound.

    Rule per spec:
      landed     — pm/landed/<tid> mem key exists
      dispatched — not landed AND has outstanding_brief_id OR dispatched_pending > 0
                   OR classified_prs.length > 0 OR review_state != null
                   OR dispatched_history.length > 0
      bound      — not landed AND target YAML exists AND none of the above
    """
    if tid in landed:
        return "landed"

    mem = _mem()

    # Check outstanding brief
    brief_rec = mem.get(f"pm/outstanding-brief/{tid}")
    if brief_rec:
        return "dispatched"

    # Check dispatched records
    dispatched_rec = mem.get(f"pm/dispatched/{tid}")
    if dispatched_rec:
        try:
            dispatched_list = json.loads(dispatched_rec["content"])
            if isinstance(dispatched_list, list) and len(dispatched_list) > 0:
                return "dispatched"
        except Exception:
            pass

    # Check classified PRs
    classified_rec = mem.get(f"pm/classified-prs/{tid}")
    if classified_rec:
        try:
            prs = json.loads(classified_rec["content"])
            if isinstance(prs, list) and len(prs) > 0:
                return "dispatched"
        except Exception:
            pass

    # Check review state
    review_rec = mem.get(f"pm/review-state/{tid}")
    if review_rec:
        return "dispatched"

    return "bound"


# ---------------------------------------------------------------------------
# Cycle detection for depends_on graphs
# ---------------------------------------------------------------------------

def _detect_cycle(nodes: list[str], edges: dict[str, list[str]]) -> list[str] | None:
    """Detect a cycle in the dependency graph. Returns cycle member list or None."""
    state: dict[str, int] = {n: 0 for n in nodes}
    # 0=unvisited, 1=in-progress, 2=done
    parent: dict[str, str | None] = {n: None for n in nodes}

    def dfs(start: str) -> list[str] | None:
        stack = [(start, iter(edges.get(start, [])))]
        state[start] = 1
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child not in state:
                    # Cross-target dep — skip (allowed in v0)
                    continue
                if state[child] == 1:
                    # Found cycle — reconstruct path
                    cycle = [child]
                    cur = node
                    while cur != child:
                        cycle.append(cur)
                        cur = parent[cur] or ""
                        if not cur or cur not in state:
                            break
                    cycle.append(child)
                    return list(reversed(cycle))
                if state[child] == 0:
                    state[child] = 1
                    parent[child] = node
                    stack.append((child, iter(edges.get(child, []))))
            except StopIteration:
                state[node] = 2
                stack.pop()
        return None

    for n in nodes:
        if state[n] == 0:
            cycle = dfs(n)
            if cycle:
                return cycle
    return None


# ---------------------------------------------------------------------------
# index.json — deterministic, no LLM
# ---------------------------------------------------------------------------

def rebuild_index() -> Path:
    """Regenerate /srv/lapis/trajectory/index.json from target YAMLs + landed mem.

    Pure-deterministic; no LLM. Errors loudly on depends_on cycle.
    """
    targets = _load_all_target_yamls()
    landed = _load_landed_mem_entries()

    # Build nodes
    nodes: list[dict] = []
    edges: list[dict] = []
    dep_graph: dict[str, list[str]] = {}

    # Build per-tid lookup for cross-referencing
    tid_to_data: dict[str, dict] = {t["id"]: t for t in targets if "id" in t}

    for target in targets:
        tid = target.get("id", "")
        if not tid:
            continue

        state = _derive_node_state(tid, target, landed)
        landed_payload = landed.get(tid, {})

        # landed_at / pr_num — best-effort from mem payload
        landed_at: str | None = landed_payload.get("landed_at") or landed_payload.get("merged_at") or landed_payload.get("ts")
        pr_num: int | None = landed_payload.get("pr_num")

        # arc_path
        arc_path: str | None = landed_payload.get("arc_path")
        if not arc_path:
            candidate = ARC_DOCS_DIR / f"{tid}.md"
            if candidate.exists():
                arc_path = str(candidate)

        # one_liner_path
        one_liner_path = str(TRAJECTORY_ROOT / "per-target" / f"{tid}.json")

        node: dict[str, Any] = {
            "tid": tid,
            "title": target.get("title", tid),
            "repo": target.get("pm_repo") or target.get("tags", [None])[0],
            "authority": target.get("pm_authority"),
            "chain_group": target.get("chain_group"),
            "tags": target.get("tags", []),
            "state": state,
            "arc_path": arc_path,
            "one_liner_path": one_liner_path,
        }
        if landed_at:
            node["landed_at"] = landed_at
        if pr_num is not None:
            node["pr_num"] = pr_num

        nodes.append(node)

        # Edges from depends_on
        dep_graph[tid] = []
        for dep in target.get("depends_on") or []:
            edges.append({
                "from": dep,
                "to": tid,
                "kind": "depends_on",
                "source": "target-yaml",
            })
            dep_graph[tid].append(dep)

    # Cycle detection
    all_tids = list(dep_graph.keys())
    cycle = _detect_cycle(all_tids, dep_graph)
    if cycle:
        cycle_str = " → ".join(cycle)
        raise SystemExit(
            f"ERROR: depends_on cycle detected in target YAMLs: {cycle_str}\n"
            "Fix the cycle before rebuilding the index."
        )

    output: dict = {
        "generated_at": _now_iso(),
        "nodes": nodes,
        "edges": edges,
    }

    out_path = TRAJECTORY_ROOT / "index.json"
    TRAJECTORY_ROOT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Wrote index.json: %d nodes, %d edges", len(nodes), len(edges))
    return out_path


# ---------------------------------------------------------------------------
# per-target/<tid>.json — Qwen one-liner
# ---------------------------------------------------------------------------

_PER_TARGET_SYSTEM = """You are a technical writer summarising landed software work for a product owner.
You produce extremely concise, factual summaries anchored to the provided arc document.
Respond with valid JSON only — no markdown fences, no prose outside the JSON object."""

_PER_TARGET_PROMPT_TMPL = """Arc document for target '{tid}':

---
{arc_body}
---

Produce a JSON object with exactly these fields:
{{
  "one_liner": "<one sentence, max 200 chars, describing what this work was and what it delivered>",
  "what_it_enables": ["<item 1>", "<item 2>"]
}}

Rules:
- one_liner: concrete, factual, 1 sentence. Name the key deliverable and its purpose.
- what_it_enables: 0-4 short bullets drawn from the arc doc's Open threads / out-of-scope framing.
  These are speculative — things this work might unlock. Omit if none are clear.
- Do not invent facts. Anchor to the arc document only.
- Respond with valid JSON only."""


def rollup_per_target(tid: str | None = None, all_targets: bool = False) -> list[Path]:
    """Generate per-target JSON files for landed targets.

    If tid is given, process just that one target.
    If all_targets is True, process all landed targets.
    Idempotent: skips if arc_doc_sha256 matches stored hash.
    """
    landed = _load_landed_mem_entries()

    if all_targets:
        tids_to_process = list(landed.keys())
    elif tid:
        if tid not in landed:
            logger.warning("Target %s not found in pm/landed mem; skipping", tid)
            return []
        tids_to_process = [tid]
    else:
        logger.warning("rollup_per_target: no --target or --all specified")
        return []

    out_dir = TRAJECTORY_ROOT / "per-target"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for t in tids_to_process:
        path = _rollup_one_target(t, landed[t], out_dir)
        if path:
            written.append(path)
    return written


def _rollup_one_target(tid: str, payload: dict, out_dir: Path) -> Path | None:
    """Generate per-target JSON for one landed target. Returns path or None on skip/error."""
    # Find arc doc
    arc_path_str = payload.get("arc_path", "")
    arc_path = Path(arc_path_str) if arc_path_str else ARC_DOCS_DIR / f"{tid}.md"
    if not arc_path.exists():
        logger.warning("Arc doc not found for %s at %s; skipping", tid, arc_path)
        return None

    arc_body = arc_path.read_text()
    current_sha = _sha256(arc_body)

    out_path = out_dir / f"{tid}.json"

    # Idempotency check: skip if arc_doc_sha256 matches
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("arc_doc_sha256") == current_sha:
                logger.debug("Skipping %s — arc doc unchanged", tid)
                return out_path
        except Exception:
            pass  # re-generate if existing JSON is malformed

    # Call Qwen (or dry-run)
    prompt = _PER_TARGET_PROMPT_TMPL.format(tid=tid, arc_body=arc_body[:12000])
    raw = _call_qwen(prompt, _PER_TARGET_SYSTEM)

    if raw is None:
        logger.warning("Qwen returned None for per-target %s; skipping", tid)
        return None

    if _dry_run():
        one_liner = "(dry run)"
        what_it_enables: list[str] = []
    else:
        # Parse JSON from Qwen response
        parsed = _parse_json_from_llm(raw)
        if parsed is None:
            logger.warning("Failed to parse Qwen JSON for %s; raw=%s", tid, raw[:200])
            return None
        one_liner = str(parsed.get("one_liner", "")).strip()
        what_it_enables = parsed.get("what_it_enables", [])
        if not isinstance(what_it_enables, list):
            what_it_enables = []
        # Truncate to max 4 items
        what_it_enables = what_it_enables[:4]

    result = {
        "tid": tid,
        "arc_doc_sha256": current_sha,
        "generated_at": _now_iso(),
        "model": "dry-run" if _dry_run() else _qwen_model_provenance(),
        "one_liner": one_liner,
        "what_it_enables": what_it_enables,
    }

    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote per-target %s.json", tid)
    return out_path


def _parse_json_from_llm(text: str) -> dict | None:
    """Extract JSON object from LLM response text."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Fall back to finding first JSON object
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    return None


# ---------------------------------------------------------------------------
# ISO week helpers
# ---------------------------------------------------------------------------

def _current_iso_week() -> str:
    """Return current ISO week string like '2026-W18'."""
    today = date.today()
    iso = today.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _parse_iso_week(week_str: str) -> tuple[date, date]:
    """Parse 'YYYY-Www' → (range_start, range_end) as date objects."""
    m = re.match(r"^(\d{4})-W(\d{2})$", week_str)
    if not m:
        raise ValueError(f"Invalid week format: {week_str!r} (expected YYYY-Www)")
    year, week = int(m.group(1)), int(m.group(2))
    # ISO week starts Monday
    jan4 = date(year, 1, 4)
    week1_monday = jan4 - timedelta(days=jan4.weekday())
    start = week1_monday + timedelta(weeks=week - 1)
    end = start + timedelta(days=6)
    return start, end


def _current_month() -> str:
    """Return current month string like '2026-05'."""
    today = date.today()
    return today.strftime("%Y-%m")


def _parse_month(month_str: str) -> tuple[date, date]:
    """Parse 'YYYY-MM' → (range_start, range_end)."""
    m = re.match(r"^(\d{4})-(\d{2})$", month_str)
    if not m:
        raise ValueError(f"Invalid month format: {month_str!r} (expected YYYY-MM)")
    year, month = int(m.group(1)), int(m.group(2))
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return start, end


# ---------------------------------------------------------------------------
# Weekly digest
# ---------------------------------------------------------------------------

_WEEKLY_SYSTEM = """You are a technical product writer producing a weekly trajectory digest
for a product owner. You write factually, concisely, and in plain English.
Ground everything in the provided data — do not hallucinate targets or events."""

_WEEKLY_PROMPT_TMPL = """Produce a weekly Lapis trajectory digest covering {range_start} through {range_end}.

LANDED TARGETS ({landed_count} targets):
{landed_section}

IN-FLIGHT TARGETS AT WEEK END ({in_flight_count} targets):
{in_flight_section}

WHAT_IT_ENABLES aggregations (from per-target data):
{enables_section}

Write a Markdown digest with these sections (no frontmatter — the caller adds it):

## Headline
One paragraph, 3-5 sentences. Name the largest-arc movement of the week.

## Landed
One bullet per landed target. Format: **<tid>** — <one_liner>  *(<repo>, <authority>, PR #<pr_num>)*
Use the provided one-liners. If a one-liner is missing, write a brief factual summary from the target title.

## In flight at week's end
One bullet per in-flight target. Format: **<tid>** — <status description>

## What this week opens up
2-4 bullets drawn from the what_it_enables aggregations. Deduplicate and cluster related items.
Label speculative items with "(speculative)".

Keep bullets tight. No hallucination."""


def rollup_weekly(week: str | None = None) -> Path:
    """Generate /srv/lapis/trajectory/weekly/<YYYY-Www>.md for the given ISO week."""
    if week is None:
        week = _current_iso_week()

    range_start, range_end = _parse_iso_week(week)

    # Gather per-target data for landed targets in this week
    landed = _load_landed_mem_entries()
    targets = _load_all_target_yamls()
    tid_to_yaml: dict[str, dict] = {t["id"]: t for t in targets if "id" in t}

    per_target_dir = TRAJECTORY_ROOT / "per-target"

    landed_in_week: list[dict] = []
    in_flight: list[dict] = []
    enables_agg: list[str] = []

    for t in targets:
        tid = t.get("id", "")
        if not tid:
            continue

        state = _derive_node_state(tid, t, landed)

        if state == "landed":
            payload = landed.get(tid, {})
            landed_at_str = payload.get("landed_at") or payload.get("ts") or ""
            # Check if landed within this week
            in_week = False
            if landed_at_str:
                try:
                    # Parse ISO timestamp
                    ts = datetime.fromisoformat(landed_at_str.replace("Z", "+00:00"))
                    landed_date = ts.date()
                    in_week = range_start <= landed_date <= range_end
                except Exception:
                    pass

            if in_week:
                # Load one_liner from per-target JSON
                one_liner = ""
                what_enables: list[str] = []
                per_target_file = per_target_dir / f"{tid}.json"
                if per_target_file.exists():
                    try:
                        pt_data = json.loads(per_target_file.read_text())
                        one_liner = pt_data.get("one_liner", "")
                        what_enables = pt_data.get("what_it_enables", [])
                    except Exception:
                        pass

                landed_in_week.append({
                    "tid": tid,
                    "title": t.get("title", tid),
                    "repo": t.get("pm_repo") or "",
                    "authority": t.get("pm_authority") or "",
                    "pr_num": payload.get("pr_num"),
                    "one_liner": one_liner or t.get("title", tid),
                })
                enables_agg.extend(what_enables)

        elif state in ("dispatched", "bound"):
            in_flight.append({
                "tid": tid,
                "title": t.get("title", tid),
                "state": state,
                "repo": t.get("pm_repo") or "",
            })

    # Build prompt sections
    def _landed_line(item: dict) -> str:
        pr = f"PR #{item['pr_num']}" if item.get("pr_num") else "no PR"
        return (f"- tid={item['tid']} | title={item['title']} | "
                f"repo={item['repo']} | authority={item['authority']} | {pr}\n"
                f"  one_liner: {item['one_liner']}")

    def _flight_line(item: dict) -> str:
        return f"- tid={item['tid']} | title={item['title']} | state={item['state']} | repo={item['repo']}"

    landed_section = "\n".join(_landed_line(i) for i in landed_in_week) or "(none)"
    in_flight_section = "\n".join(_flight_line(i) for i in in_flight) or "(none)"
    enables_section = "\n".join(f"- {e}" for e in enables_agg) or "(none)"

    prompt = _WEEKLY_PROMPT_TMPL.format(
        range_start=range_start,
        range_end=range_end,
        landed_count=len(landed_in_week),
        in_flight_count=len(in_flight),
        landed_section=landed_section,
        in_flight_section=in_flight_section,
        enables_section=enables_section,
    )

    raw = _call_qwen(prompt, _WEEKLY_SYSTEM)
    if raw is None:
        logger.warning("Qwen returned None for weekly digest %s; skipping", week)
        return TRAJECTORY_ROOT / "weekly" / f"{week}.md"

    body = raw.strip()

    # Build frontmatter
    frontmatter = (
        f"---\n"
        f"period: weekly\n"
        f"range_start: {range_start}\n"
        f"range_end: {range_end}\n"
        f"generated_at: {_now_iso()}\n"
        f"model: {'dry-run' if _dry_run() else _qwen_model_provenance()}\n"
        f"landed_count: {len(landed_in_week)}\n"
        f"in_flight_count: {len(in_flight)}\n"
        f"---\n\n"
        f"# Lapis trajectory — week of {range_start}\n\n"
    )

    out_dir = TRAJECTORY_ROOT / "weekly"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{week}.md"
    out_path.write_text(frontmatter + body)
    logger.info("Wrote weekly digest: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Monthly digest — per-week extractive → synthesize
# ---------------------------------------------------------------------------

_MONTHLY_EXTRACT_SYSTEM = """You are a technical summariser. Extract 3-5 concise bullet points
from the provided weekly digest. Focus on the most significant technical deliverables and themes.
Respond with a plain bulleted list only (- bullet text). No preamble."""

_MONTHLY_SYNTH_SYSTEM = """You are a technical product writer producing a monthly Lapis trajectory digest.
Write factually and concisely from the provided per-week bullet extractions."""

_MONTHLY_SYNTH_PROMPT_TMPL = """Produce a monthly Lapis trajectory digest for {month}.

Per-week extracted bullets:

{weekly_bullets}

Write a Markdown digest with these sections (no frontmatter — the caller adds it):

## Headline
One paragraph, 3-5 sentences. Name the largest-arc movements of the month.

## Landed
Group significant landed targets by theme/repo cluster. 2-4 bullets per group, grouped under sub-headers.

## In flight at month end
2-4 bullets on the most significant in-flight work.

## What this month opens up
2-4 bullets on what the month's work collectively unlocks.

Keep it tight. No hallucination. Ground in the provided bullets only."""


def _weeks_in_month(month_str: str) -> list[str]:
    """Return list of ISO week strings that overlap with the given month."""
    range_start, range_end = _parse_month(month_str)
    weeks: list[str] = []
    cur = range_start
    # Walk day by day to catch all weeks
    seen: set[str] = set()
    while cur <= range_end:
        iso = cur.isocalendar()
        week_str = f"{iso.year}-W{iso.week:02d}"
        if week_str not in seen:
            seen.add(week_str)
            weeks.append(week_str)
        cur += timedelta(days=7)
        cur = cur - timedelta(days=cur.weekday())  # snap to Monday
    return weeks


def rollup_monthly(month: str | None = None) -> Path:
    """Generate /srv/lapis/trajectory/monthly/<YYYY-MM>.md for the given month.

    Strategy: per-week extractive→synthesize chunking (always).
    1. For each constituent weekly digest, ask Qwen to extract 3-5 bullets.
    2. Synthesize all bullets into the monthly headline + sections.
    """
    if month is None:
        month = _current_month()

    _parse_month(month)  # validate format

    weekly_dir = TRAJECTORY_ROOT / "weekly"
    weeks = _weeks_in_month(month)

    # Step 1: extractive pass per available weekly digest
    weekly_bullets: list[str] = []
    for week in weeks:
        week_file = weekly_dir / f"{week}.md"
        if not week_file.exists():
            logger.debug("Weekly digest %s not found for monthly rollup; skipping", week)
            continue
        week_body = week_file.read_text()

        extract_prompt = (
            f"Weekly digest for {week}:\n\n{week_body[:6000]}\n\n"
            "Extract 3-5 concise bullet points capturing the key technical deliverables "
            "and themes. Respond with a plain bulleted list only (- bullet text)."
        )
        extract_raw = _call_qwen(extract_prompt, _MONTHLY_EXTRACT_SYSTEM)
        if extract_raw is None:
            logger.warning("Qwen returned None for weekly extract %s; skipping", week)
            continue

        if _dry_run():
            weekly_bullets.append(f"### {week}\n- (dry run)")
        else:
            weekly_bullets.append(f"### {week}\n{extract_raw.strip()}")

    if not weekly_bullets:
        logger.warning("No weekly digests found for month %s; writing empty monthly", month)
        out_dir = TRAJECTORY_ROOT / "monthly"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{month}.md"
        range_start, range_end = _parse_month(month)
        out_path.write_text(
            f"---\nperiod: monthly\nrange_start: {range_start}\nrange_end: {range_end}\n"
            f"generated_at: {_now_iso()}\nmodel: {'dry-run' if _dry_run() else _qwen_model_provenance()}\n"
            f"landed_count: 0\nin_flight_count: 0\n---\n\n"
            f"# Lapis trajectory — {month}\n\n*(No weekly digests available for this month.)*\n"
        )
        return out_path

    # Step 2: synthesis pass
    all_bullets = "\n\n".join(weekly_bullets)

    # Haiku fallback if env flag set
    if _use_haiku_for_monthly():
        synth_prompt = _MONTHLY_SYNTH_PROMPT_TMPL.format(
            month=month,
            weekly_bullets=all_bullets,
        )
        raw = _call_haiku(synth_prompt, _MONTHLY_SYNTH_SYSTEM)
        model_used = "claude-haiku-4-5-20251001"
    else:
        synth_prompt = _MONTHLY_SYNTH_PROMPT_TMPL.format(
            month=month,
            weekly_bullets=all_bullets,
        )
        raw = _call_qwen(synth_prompt, _MONTHLY_SYNTH_SYSTEM)
        model_used = _qwen_model_provenance()

    if raw is None:
        logger.warning("LLM returned None for monthly synthesis %s; skipping", month)
        return TRAJECTORY_ROOT / "monthly" / f"{month}.md"

    body = raw.strip()
    range_start, range_end = _parse_month(month)

    # Count landed/in-flight from most recent weekly digest (best effort)
    landed_count = 0
    in_flight_count = 0
    if weeks:
        last_week_file = weekly_dir / f"{weeks[-1]}.md"
        if last_week_file.exists():
            content = last_week_file.read_text()
            m1 = re.search(r"^landed_count:\s*(\d+)", content, re.MULTILINE)
            m2 = re.search(r"^in_flight_count:\s*(\d+)", content, re.MULTILINE)
            if m1:
                landed_count = int(m1.group(1))
            if m2:
                in_flight_count = int(m2.group(1))

    frontmatter = (
        f"---\n"
        f"period: monthly\n"
        f"range_start: {range_start}\n"
        f"range_end: {range_end}\n"
        f"generated_at: {_now_iso()}\n"
        f"model: {'dry-run' if _dry_run() else model_used}\n"
        f"landed_count: {landed_count}\n"
        f"in_flight_count: {in_flight_count}\n"
        f"---\n\n"
        f"# Lapis trajectory — {month}\n\n"
    )

    out_dir = TRAJECTORY_ROOT / "monthly"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{month}.md"
    out_path.write_text(frontmatter + body)
    logger.info("Wrote monthly digest: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Land-time hook — called from cmd_land after check_chain_advance
# ---------------------------------------------------------------------------

def on_land(target_id: str) -> None:
    """Fire per-target rollup + rebuild-index for a newly landed target.

    Non-fatal: all errors are logged at WARNING and suppressed.
    Called from cmd_land AFTER check_chain_advance.
    """
    try:
        rollup_per_target(tid=target_id)
    except Exception as e:
        logger.warning("trajectory per-target rollup failed for %s: %s", target_id, e)

    try:
        rebuild_index()
    except SystemExit as e:
        # SystemExit is from cycle detection — log but don't re-raise
        logger.warning("trajectory rebuild-index failed for %s: %s", target_id, e)
    except Exception as e:
        logger.warning("trajectory rebuild-index failed for %s: %s", target_id, e)
