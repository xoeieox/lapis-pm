"""mem.db arc registry: structured arc state, replacing prose-parsing.

D2 (arc-registry-contract-schema-v0): mem key `arc/<slug>` carries
`{slug, declared_next, anchors[], last_touched, importance, contract,
derived_ts, derived_from}`. NEXT IS COMPUTED, not stored: nothing here
persists a hand-maintained NEXT -- the reconciler still computes it as a
nightly delta (D4's `_read_arc_climate`); this module only supplies
structured rows as an alternative to prose-parsing on-disk arc-docs.

D2a (no heuristic authority): the populator never manufactures a value it
could not derive. `importance` is int | None -- null, never a default, when
it cannot be derived from non-degenerate exhaust. `derived_from` is a
runtime assertion: a row with an empty `derived_from` is never written.

Sources (D3/D3a), each contributing named fields only -- nothing is
inferred across sources:
  arc-docs  -- room_path('lapis_state')/<slug>.md -> slug, declared_next,
               last_touched (generated: frontmatter, else file mtime).
  targets   -- room_path('targets')/<id>.yaml -> anchors (target ids), by
               filename stem appearing in the arc-doc body.
  landed    -- mem keys pm/landed/<target> -> anchors (PR/repo), last_touched
               candidate (merged_at).

Weaver is NOT a source in v0 (dropped at gate round 2 -- no thread-digest
read path exists to build on). MEMORY.md is NOT a source in v0 (OQ-1, a
hard scope boundary, not an open question).

The three NEXT-extraction regexes are imported from state_brief.py, not
redefined here -- a second, divergent NEXT parser is the most likely way
this unit silently breaks byte-identical prose-path output (DoD 11).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .contract import Contract, Observation, SubjectRef, compute_delta
from .state_brief import (
    _ARC_STALE_DAYS,
    _BULLET_LINE_RE,
    _NEXT_HEADER_RE,
    _NEXT_LINE_RE,
    _parse_iso_ts,
    _SILENTLY_ADVANCED_NEXT_RE,
)

_GENERATED_RE = re.compile(r'^generated:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)

DEFAULT_PREFIX = "arc/"


# ---------------------------------------------------------------------------
# Per-source derivation (D3a)
# ---------------------------------------------------------------------------

def _extract_declared_next(text: str) -> str | None:
    """Same algorithm as state_brief._parse_declared_next, built on the same
    (imported, not redefined) regex objects -- see module docstring."""
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


def _extract_generated_ts(text: str) -> datetime | None:
    """`generated:` frontmatter timestamp, else None (caller falls back to mtime)."""
    m = _GENERATED_RE.search(text)
    if not m:
        return None
    return _parse_iso_ts(m.group(1))


def _derive_row(
    slug: str,
    doc_path: Path,
    targets_dir: Path,
    mem,
    *,
    now: datetime,
) -> dict:
    """Derive one arc/<slug> row from exhaust. Never fabricates a value it
    cannot derive (D2a) -- unresolved fields stay None, not a filled-in guess."""
    derived_from: list[str] = ["arc-docs"]

    text = doc_path.read_text(encoding="utf-8")
    declared_next = _extract_declared_next(text)

    generated_ts = _extract_generated_ts(text)
    if generated_ts is not None:
        last_touched_candidates: list[datetime] = [generated_ts]
    else:
        try:
            last_touched_candidates = [datetime.fromtimestamp(doc_path.stat().st_mtime, tz=timezone.utc)]
        except OSError:
            last_touched_candidates = []

    # anchor -> best-known timestamp (None if the anchor carries no usable ts)
    anchor_ts: dict[str, datetime | None] = {}

    # --- targets source: degrade to no anchors on any failure, never abort ---
    try:
        if targets_dir.exists():
            for target_path in sorted(targets_dir.glob("*.yaml")):
                target_id = target_path.stem
                if target_id and target_id in text:
                    if "targets" not in derived_from:
                        derived_from.append("targets")
                    try:
                        anchor_ts[target_id] = datetime.fromtimestamp(
                            target_path.stat().st_mtime, tz=timezone.utc,
                        )
                    except OSError:
                        anchor_ts[target_id] = None
    except OSError:
        pass  # targets source unreachable -- contributes nothing, row still written

    # --- landed source: degrade to no anchors/timestamp on any failure ---
    landed_signal: str | None = None
    landed_obs_ts: str | None = None
    target_anchors = [a for a in anchor_ts]
    for target_id in target_anchors:
        try:
            rec = mem.get(f"pm/landed/{target_id}")
        except Exception:
            continue
        if rec is None:
            continue
        if "landed" not in derived_from:
            derived_from.append("landed")
        try:
            payload = json.loads(rec.get("content", "") or "")
        except (ValueError, TypeError):
            payload = None
        merged_ts = None
        pr_repr = None
        if isinstance(payload, dict):
            merged_ts = _parse_iso_ts(payload.get("merged_at"))
            pr_num = payload.get("pr") or payload.get("pr_number")
            repo = payload.get("repo")
            if pr_num and repo:
                pr_repr = f"{repo}#{pr_num}"
                anchor_ts[pr_repr] = merged_ts
        if merged_ts is not None:
            last_touched_candidates.append(merged_ts)
            landed_obs_ts = merged_ts.isoformat()
            landed_signal = f"PR {pr_repr} merged" if pr_repr else None

    last_touched = max(last_touched_candidates).isoformat() if last_touched_candidates else None
    anchors = sorted(anchor_ts.keys())
    importance = _compute_importance(anchor_ts, now=now)
    contract = _build_contract(slug, declared_next, landed_signal, landed_obs_ts)

    return {
        "slug": slug,
        "declared_next": declared_next,
        "anchors": anchors,
        "last_touched": last_touched,
        "importance": importance,
        "contract": contract.to_dict() if contract is not None else None,
        "derived_from": sorted(set(derived_from)),
    }


def _compute_importance(anchor_ts: dict[str, datetime | None], *, now: datetime) -> int | None:
    """D2b, concretely and deterministically. `None` only for zero-anchor
    arcs -- never 0, never a default, never a midpoint."""
    anchors_n = len(anchor_ts)
    if anchors_n == 0:
        return None

    timestamps = [ts for ts in anchor_ts.values() if ts is not None]
    if timestamps:
        newest = max(timestamps)
        age = max((now - newest).days, 0)
        if age < 7:
            recency_bonus = 2
        elif age < _ARC_STALE_DAYS:
            recency_bonus = 1
        else:
            recency_bonus = 0
    else:
        # No anchor carries a derivable timestamp: honest default is "no
        # freshness signal", never a fabricated bonus.
        recency_bonus = 0

    return min(anchors_n + recency_bonus, 10)


def _build_contract(
    slug: str,
    declared_next: str | None,
    landed_signal: str | None,
    landed_obs_ts: str | None,
) -> Contract | None:
    """Contract is null unless there is a declaration to check. If the
    landed source was queried (landed_obs_ts set) but yielded no signal,
    an Observation(value=None) is recorded -- the "lie of omission" fold
    (D1a) -- rather than being silently dropped as unobserved."""
    if declared_next is None:
        return None
    observed: tuple[Observation, ...] = ()
    if landed_obs_ts is not None:
        observed = (Observation(source="pm/landed", key="next", value=landed_signal, ts=landed_obs_ts),)
    return Contract(
        subject=SubjectRef(kind="arc", id=slug),
        declaration=declared_next,
        elaboration={"next": declared_next},
        observed=observed,
    )


# ---------------------------------------------------------------------------
# Write path (through the existing provenance-emitting mem deposit path)
# ---------------------------------------------------------------------------

def _write_row(mem, slug: str, row: dict, *, prefix: str = DEFAULT_PREFIX) -> None:
    """Raises if `derived_from` is empty -- provenance is a runtime
    assertion, not documentation. A row without it has negative value."""
    if not row.get("derived_from"):
        raise ValueError(
            f"arc_registry: refusing to write {prefix}{slug} with empty derived_from"
        )
    key = f"{prefix}{slug}"
    mem.set(key, json.dumps(row, sort_keys=True, ensure_ascii=False), tags=["lapis-pm", "arc-registry"])


# ---------------------------------------------------------------------------
# Populator (D3/D3a/D3b)
# ---------------------------------------------------------------------------

def populate(
    *,
    prefix: str = DEFAULT_PREFIX,
    dry_run: bool = False,
    limit: int | None = None,
    arc_dir: Path | None = None,
    targets_dir: Path | None = None,
    mem=None,
    now: datetime | None = None,
) -> dict:
    """Derive arc/<slug> rows from exhaust only and (unless dry_run) write
    them through the mem deposit path. Idempotent: re-running over
    unchanged exhaust produces byte-identical rows apart from derived_ts.

    Degenerate-output guard (D2a): a full-corpus run (limit is None) whose
    derived `importance` values have zero variance across >= 2 rows raises
    rather than writing -- a systemic derivation failure, not a result.
    A zero-arc corpus is vacuously non-degenerate and succeeds silently;
    `--limit N` sampling runs skip the guard entirely.
    """
    if arc_dir is None or targets_dir is None or mem is None:
        from . import node_identity
        from agents_core.room_paths import room_path
        if arc_dir is None:
            arc_dir = room_path('lapis_state')
        if targets_dir is None:
            targets_dir = room_path('targets')
        if mem is None:
            mem = node_identity.writable_store()
    if now is None:
        now = datetime.now(tz=timezone.utc)

    if not arc_dir.exists():
        return {"written": 0, "skipped": 0, "rows": {}}

    doc_paths = sorted(arc_dir.glob("*.md"))
    if limit is not None:
        doc_paths = doc_paths[:limit]

    rows: dict[str, dict] = {}
    skipped = 0
    for doc_path in doc_paths:
        try:
            row = _derive_row(doc_path.stem, doc_path, targets_dir, mem, now=now)
        except OSError:
            skipped += 1
            continue
        rows[row["slug"]] = row

    if limit is None and len(rows) >= 2:
        importances = [r["importance"] for r in rows.values()]
        if len(set(importances)) <= 1:
            raise RuntimeError(
                f"arc_registry populate: zero variance in importance across "
                f"{len(importances)} arcs -- systemic derivation failure, "
                "not writing a degenerate corpus"
            )

    written = 0
    if not dry_run:
        for slug, row in rows.items():
            out = dict(row)
            out["derived_ts"] = now.isoformat()
            _write_row(mem, slug, out, prefix=prefix)
            written += 1

    return {"written": written, "skipped": skipped, "rows": rows}


# ---------------------------------------------------------------------------
# D4 read path: registry-sourced classification for _read_arc_climate
# ---------------------------------------------------------------------------

def read_registry_rows(prefix: str = DEFAULT_PREFIX, *, mem=None) -> dict[str, dict]:
    """Read all arc/<slug> rows under `prefix`, keyed by slug."""
    if mem is None:
        from . import node_identity
        mem = node_identity.writable_store()
    entries = mem.list_by_prefix(prefix, limit=10_000)
    rows: dict[str, dict] = {}
    for entry in entries:
        try:
            row = json.loads(entry.get("content", "") or "")
        except (ValueError, TypeError):
            continue
        if not isinstance(row, dict):
            continue
        slug = row.get("slug") or entry.get("key", "").removeprefix(prefix)
        rows[slug] = row
    return rows


def _row_deltas(row: dict):
    contract_dict = row.get("contract")
    if not contract_dict:
        return ()
    return compute_delta(Contract.from_dict(contract_dict))


def classify_registry_row(row: dict, *, now: datetime) -> dict | None:
    """Classify a single registry row the same way _read_arc_climate
    classifies a prose-parsed arc-doc, sourced from structured fields
    instead of live re-parsing. Additive -- never called by the default
    ("prose") path."""
    slug = row.get("slug", "?")
    declared_next = row.get("declared_next")
    last_touched_raw = row.get("last_touched")
    last_touched = _parse_iso_ts(last_touched_raw) if last_touched_raw else None
    age_days = max((now - last_touched).days, 0) if last_touched is not None else None

    deltas = _row_deltas(row)
    unobserved = [d for d in deltas if d.kind == "unobserved"]
    contradicted = [d for d in deltas if d.kind == "contradicted"]

    if declared_next and _SILENTLY_ADVANCED_NEXT_RE.search(declared_next) and contradicted:
        return {
            "slug": slug,
            "classification": "silently-advanced",
            "text": (
                f"{slug}: registry shows declared NEXT '{declared_next}' contradicted "
                "by an observed value -- the arc reads complete."
            ),
        }

    if age_days is not None and age_days <= _ARC_STALE_DAYS and not unobserved:
        return None  # moving -- silence is not churn

    if unobserved:
        return {
            "slug": slug,
            "classification": "unresolvable",
            "text": (
                f"{slug}: registry has {len(unobserved)} unobserved delta(s) against "
                f"'{declared_next or 'no declared NEXT'}'."
            ),
        }

    return {
        "slug": slug,
        "classification": "gone-quiet",
        "text": (
            f"{slug}: {age_days if age_days is not None else '?'}d of silence since "
            f"'{declared_next or 'no declared NEXT'}'."
        ),
    }


def read_arc_climate_from_registry(prefix: str = DEFAULT_PREFIX, *, mem=None, now: datetime | None = None) -> list[dict]:
    """Registry-sourced counterpart to state_brief's prose-parsing reconciler.
    Additive read path (D4) -- off by default, never called unless
    arc_source="registry" is passed explicitly."""
    if now is None:
        now = datetime.now(tz=timezone.utc)
    rows = read_registry_rows(prefix, mem=mem)
    results = []
    for slug in sorted(rows):
        classified = classify_registry_row(rows[slug], now=now)
        if classified is not None:
            results.append(classified)
    return results


# ---------------------------------------------------------------------------
# D5/D5a: automated drift cross-check
# ---------------------------------------------------------------------------

def compare_sources(start_ts: datetime, period: str = "weekly", *, prefix: str = DEFAULT_PREFIX) -> dict:
    """Run _read_arc_climate with arc_source="prose" and "registry" over the
    same inputs and return a structural diff. Reports only -- never raises,
    never mutates, never auto-fails a night pass.

    `prefix` selects which mem-key prefix the registry side reads from, so a
    PM can run this against a scratch prefix (DoD 14-15) before ever
    comparing against the live `arc/` prefix. Defaults to DEFAULT_PREFIX.

    D5a: `blind_spots` names slugs where the prose path produced a definite
    classification while the registry either has no row at all for that slug
    or produced >= 1 "unobserved" delta for it -- the mechanical fingerprint
    of "the declaration lives somewhere the registry cannot see."
    """
    from . import state_brief

    prose = state_brief._read_arc_climate(start_ts, period=period, arc_source="prose")
    registry = state_brief._read_arc_climate(start_ts, period=period, arc_source="registry", prefix=prefix)

    prose_by_slug = {e["slug"]: e for e in prose}
    registry_by_slug = {e["slug"]: e for e in registry}

    counts_by_classification: dict[str, dict[str, int]] = {"prose": {}, "registry": {}}
    for e in prose:
        counts_by_classification["prose"][e["classification"]] = (
            counts_by_classification["prose"].get(e["classification"], 0) + 1
        )
    for e in registry:
        counts_by_classification["registry"][e["classification"]] = (
            counts_by_classification["registry"].get(e["classification"], 0) + 1
        )

    all_slugs = sorted(set(prose_by_slug) | set(registry_by_slug))
    agreements: list[str] = []
    disagreements: list[dict] = []
    only_in_prose: list[str] = []
    only_in_registry: list[str] = []

    for slug in all_slugs:
        p = prose_by_slug.get(slug)
        r = registry_by_slug.get(slug)
        if p is not None and r is None:
            only_in_prose.append(slug)
        elif r is not None and p is None:
            only_in_registry.append(slug)
        elif p["classification"] == r["classification"]:
            agreements.append(slug)
        else:
            disagreements.append({"slug": slug, "prose": p["classification"], "registry": r["classification"]})

    raw_rows = read_registry_rows(prefix)
    blind_spots: list[str] = []
    for slug in prose_by_slug:
        row = raw_rows.get(slug)
        if row is None or any(d.kind == "unobserved" for d in _row_deltas(row)):
            blind_spots.append(slug)
    blind_spots.sort()

    return {
        "agreements": agreements,
        "disagreements": disagreements,
        "only_in_prose": only_in_prose,
        "only_in_registry": only_in_registry,
        "counts_by_classification": counts_by_classification,
        "blind_spots": blind_spots,
    }


# ---------------------------------------------------------------------------
# D3b CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lapis_pm.arc_registry")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("populate", help="Derive and write arc/<slug> registry rows from exhaust.")
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "populate":
        result = populate(prefix=args.prefix, dry_run=args.dry_run, limit=args.limit)
        print(json.dumps({
            "written": result["written"],
            "skipped": result["skipped"],
            "slugs": sorted(result["rows"].keys()),
        }, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
