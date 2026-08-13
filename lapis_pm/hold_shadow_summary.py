"""hold_shadow_summary.py — Thursday-morning summary pass
(lapis-pm-hold-shadow-observer-v0, Scope item 4).

Runs OUT of tick, on its own systemd timer (~06:45 Pacific,
systemd/lapis-hold-shadow-summary.{timer,service}). Reads the accumulated
gate-outcome/v1 and hold-fact/v1 records, groups them by proposed
classification per the ratified Thursday process change (Erah, 2026-08-12
seam directive: "present the shadow log grouped by proposed
classification"), and deposits exactly ONE Desk gem via weaver's
decision-gems endpoint, deposited_by="hold-shadow-observer-v0".

Isolation (same invariants as hold_shadow.py — see that module's
docstring): reads only its own JSONL files under /srv/lapis/hold-shadow/, writes
nothing to mem, and critically never writes a pm/brief-gem/map/<gem_id> key
— so the gem this deposits is structurally unactionable. brief_gem.py's
reconcile_decided_gems() skips any gem with no forward-mapping key ("not a
brief-gem we deposited", brief_gem.py's reconcile loop) — the gem reconciler
can never mistake this for a real decision to execute. The only network
call this module makes is the single gem POST.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

DEPOSITED_BY = "hold-shadow-observer-v0"
_BRIX_DEFAULT = "http://203.0.113.10:8403"  # prod-only fallback, same default as brief_gem.py


def _room_root() -> Path:
    return Path(os.environ.get("ROOM_ROOT", "/room"))


def hold_shadow_dir() -> Path:
    return _room_root() / "hold-shadow"


def _weaver_base_url() -> str:
    """Resolve weaver base URL: WEAVER_BASE_URL > WEAVER_BIND_PORT > BRIX
    default. Mirrors brief_gem.py's _weaver_base_url — duplicated rather
    than imported so this module stays free of any lapis_pm import (the
    isolation boundary test asserts zero act-surface imports; not importing
    brief_gem.py at all keeps that assertion trivially true)."""
    base = os.environ.get("WEAVER_BASE_URL")
    if base:
        return base.rstrip("/")
    port = os.environ.get("WEAVER_BIND_PORT")
    if port:
        return f"http://127.0.0.1:{port}"
    return _BRIX_DEFAULT


def _read_jsonl(path: Path) -> list[dict]:
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
            continue  # a torn/partial line never aborts the summary
    return records


def build_summary(
    gate_outcomes: list[dict],
    hold_facts: list[dict],
    enforce_outcomes: list[dict] | None = None,
) -> dict:
    """Pure function: group records by proposed classification. No I/O —
    the caller (run_thursday_summary) does all reading/writing.

    enforce_outcomes (lapis-pm-bundle-autodispatch-enforce-v0, Design 3):
    the bundle-autodispatch three-way classifier's own record type,
    grouped by enforce_class (infra/salvage/defer) alongside the existing
    hold-fact and gate-outcome groupings. Optional + defaulted so every
    pre-existing caller (this module's own tests included) keeps working
    unchanged."""
    enforce_outcomes = enforce_outcomes or []

    by_hold_class: dict[str, list[dict]] = defaultdict(list)
    for rec in hold_facts:
        by_hold_class[rec.get("hold_class", "other")].append(rec)

    by_recommendation: dict[str, list[dict]] = defaultdict(list)
    for rec in gate_outcomes:
        by_recommendation[rec.get("combined_recommendation", "unknown")].append(rec)

    by_enforce_class: dict[str, list[dict]] = defaultdict(list)
    for rec in enforce_outcomes:
        by_enforce_class[rec.get("enforce_class", "unknown")].append(rec)

    return {
        "hold_facts_total": len(hold_facts),
        "gate_outcomes_total": len(gate_outcomes),
        "enforce_outcomes_total": len(enforce_outcomes),
        "by_hold_class": {k: len(v) for k, v in by_hold_class.items()},
        "by_combined_recommendation": {k: len(v) for k, v in by_recommendation.items()},
        "by_enforce_class": {k: len(v) for k, v in by_enforce_class.items()},
        "hold_facts_grouped": dict(by_hold_class),
        "gate_outcomes_grouped": dict(by_recommendation),
        "enforce_outcomes_grouped": dict(by_enforce_class),
    }


def _render_gem_payload(summary: dict) -> dict:
    enforce_total = summary.get("enforce_outcomes_total", 0)
    total = summary["hold_facts_total"] + summary["gate_outcomes_total"] + enforce_total
    lines = [
        f"Hold-shadow overnight log: {summary['hold_facts_total']} hold-fact "
        f"record(s), {summary['gate_outcomes_total']} gate-outcome record(s), "
        f"{enforce_total} enforce-outcome record(s).",
        "",
        "By proposed classification (hold-fact):",
    ]
    if summary["by_hold_class"]:
        for hold_class, n in sorted(summary["by_hold_class"].items()):
            lines.append(f"  - {hold_class}: {n}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("By gate recommendation (gate-outcome):")
    if summary["by_combined_recommendation"]:
        for rec, n in sorted(summary["by_combined_recommendation"].items()):
            lines.append(f"  - {rec}: {n}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(
        "By enforce classification (enforce-outcome, bundle_autodispatch's "
        "three-way infra/salvage/defer classifier):"
    )
    if summary.get("by_enforce_class"):
        for enforce_class, n in sorted(summary["by_enforce_class"].items()):
            lines.append(f"  - {enforce_class}: {n}")
    else:
        lines.append("  (none)")

    ask = (
        "Read the grouped log and rule on where the enforce boundary should sit."
        if total else
        "No records overnight — the log is empty; a small first-night log is "
        "still a valid boundary input (ratified, Erah option A)."
    )

    return {
        "title": "Hold-shadow overnight log — draw the enforce boundary",
        "ask": ask,
        "why": "\n".join(lines),
        "origin": "hold-shadow-observer-v0",
        "agent": "PM",
        "deposited_by": DEPOSITED_BY,
        "source_thread_id": "hold-shadow-observer-v0",
        "options": [{"key": "ack", "title": "Acknowledge", "sub": ""}],
        "state": "needs",
    }


def run_thursday_summary() -> str | None:
    """Deposit exactly one Desk gem for this morning's read. Returns the
    gem_id on success, or None on any failure (fail-soft: never raises out
    of this function; the caller's exit code carries the failure signal
    instead — see cli.py's hold-shadow-summary subcommand)."""
    gate_outcomes = _read_jsonl(hold_shadow_dir() / "gate-outcomes.jsonl")
    hold_facts = _read_jsonl(hold_shadow_dir() / "hold-facts.jsonl")
    enforce_outcomes = _read_jsonl(hold_shadow_dir() / "enforce-outcomes.jsonl")
    summary = build_summary(gate_outcomes, hold_facts, enforce_outcomes)
    payload = _render_gem_payload(summary)

    try:
        import httpx
        base = _weaver_base_url()
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{base}/v0/decision-gems", json=payload)
            resp.raise_for_status()
            return resp.json().get("gem_id")
    except Exception:
        return None
