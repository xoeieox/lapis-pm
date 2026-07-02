"""Research-quest nightly producer — sources a standing backlog via Dowser.

Dowser's second, general-purpose consumer (research-quest-nightly-producer-v0):
works a hand-curated backlog of research intents independent of any Backcaster
run, writing cited findings into /srv/lapis/research/ as fresh corpus content.

Reuses the swarm<->big flip choreography, doorman serving-gate, phase helpers,
and any-exit flip guard from lapis_pm.gw_flip_gate (extracted from
backcaster/quest_leg.py, AC1) rather than reimplementing GW's safety-critical
mode handling. Per backlog entry, runs one swarm-read -> big-critique cycle —
the same phase sequence backcaster-quest already proves.
"""
from __future__ import annotations

import functools
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import gw_flip_gate

log = logging.getLogger(__name__)

_BACKLOG_HEADER = """\
# Dowser research backlog — hand-curated standing queue for research-quest.
#
# Schema (list under `entries`):
#   - id: <str>                    unique backlog entry id
#     intent: <str>                the research question / intent (required)
#     context: <str>               optional extra context passed to Dowser
#     sub_intents: [<str>, ...]    optional list of sub-questions
#     status: pending|done|no-credible-sources
#     attempts: <int>              retry counter, default 0
#     added: <date YYYY-MM-DD>     when the entry was seeded
#
# Entries are seeded by hand (Erah/PM) or a future producer — this file does
# not fabricate research questions. research-quest-nightly-producer-v0.
"""

_ATTEMPTS_CAP = 2
_MAX_SLUG_LEN = 60


# ---------------------------------------------------------------------------
# Backlog IO
# ---------------------------------------------------------------------------

def _research_root() -> Path:
    from agents_core.room_paths import room_path
    return Path(room_path("research"))


def _backlog_path() -> Path:
    return _research_root() / "queue" / "dowser-backlog.yaml"


def _load_backlog(path: Path) -> list[dict]:
    """Load backlog entries, bootstrapping the documented empty-schema file
    on first run if it doesn't exist yet (AC2: ship with the schema
    documented and an empty entry list — entries are seeded by hand)."""
    if not path.exists():
        _write_backlog(path, [])
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    return raw.get("entries", []) or []


def _write_backlog(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump(
        {"entries": entries}, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
    path.write_text(_BACKLOG_HEADER + "\n" + body)


def _record_attempt(entry: dict) -> None:
    """Increment attempts; cap-then-permanent-null (mirrors Dowser's own retry-cap-1)."""
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    if entry["attempts"] >= _ATTEMPTS_CAP:
        entry["status"] = "no-credible-sources"
    else:
        entry["status"] = "pending"


# ---------------------------------------------------------------------------
# Findings write-back (loose essay shape, matches existing /srv/lapis/research/*.md)
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:_MAX_SLUG_LEN].rstrip("-") or "untitled"


def _write_finding_file(root: Path, entry: dict, draft: dict) -> Path:
    today = datetime.now(timezone.utc).date().isoformat()
    slug = _slugify(entry["intent"])
    path = root / f"{today}-{slug}.md"

    lines = [f"# {entry['intent']}", "", draft.get("findings", "").strip(), "", "## Sources", ""]
    for c in draft.get("citations", []):
        title = c.get("title") or c.get("url", "")
        excerpt = c.get("excerpt", "")
        credibility = c.get("credibility", "")
        lines.append(f"- [{title}]({c.get('url', '')}) (credibility: {credibility}) - {excerpt}")

    root.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_research_quest(
    limit: int = 5,
    dry_run: bool = False,
    *,
    _dowser: Any = None,
    _flip_fn: Any = None,
    _gate_fn: Any = None,
) -> dict:
    """Source up to `limit` pending backlog entries via Dowser.

    Parameters
    ----------
    limit   : max backlog entries to process this run
    dry_run : select + log pending entries without touching GW/Dowser or
              mutating the backlog
    _dowser, _flip_fn, _gate_fn : injectable (tests); default to
              agents_core.dowser and the shared gw_flip_gate functions.

    Returns a human-readable summary dict.
    """
    backlog_path = _backlog_path()
    entries = _load_backlog(backlog_path)
    pending = [e for e in entries if e.get("status") == "pending"][:limit]

    if not pending:
        log.warning("research-quest: no pending backlog entries — idle this cycle")
        return {
            "entries_targeted": 0,
            "entries_sourced": 0,
            "entries_honest_null": 0,
            "note": "no pending backlog entries",
        }

    if dry_run:
        for entry in pending:
            log.info(
                "research-quest: [dry-run] would process %s: %s",
                entry.get("id"), entry.get("intent"),
            )
        return {
            "entries_targeted": len(pending),
            "entries_sourced": 0,
            "entries_honest_null": 0,
            "note": "dry-run",
        }

    if _dowser is None:
        import agents_core.dowser as _dowser
    if _flip_fn is None:
        _flip_fn = functools.partial(gw_flip_gate.flip_gw, source="research-quest-night")
    if _gate_fn is None:
        _gate_fn = gw_flip_gate.gate_doorman_serving

    read_operator = "quest"
    critic_operator = "gravitywell"
    research_root = _research_root()

    entries_sourced = 0
    entries_honest_null = 0

    with gw_flip_gate.any_exit_flip_guard(_flip_fn, active=True):
        for entry in pending:
            try:
                request: dict[str, Any] = {"intent": entry["intent"]}
                if entry.get("context"):
                    request["context"] = entry["context"]
                if entry.get("sub_intents"):
                    request["sub_intents"] = entry["sub_intents"]

                drafts, _read_timed_out = gw_flip_gate.phase_swarm_read(
                    [request], read_operator, _dowser, _flip_fn, _gate_fn
                )
                if not drafts:
                    log.warning("research-quest: entry %s read phase failed", entry.get("id"))
                    _record_attempt(entry)
                    entries_honest_null += 1
                    continue

                draft = drafts[0]
                verdicts, _critique_timed_out = gw_flip_gate.phase_big_critique(
                    drafts, critic_operator, _dowser, _flip_fn, _gate_fn, True
                )
                if not verdicts:
                    log.warning("research-quest: entry %s critique phase failed", entry.get("id"))
                    _record_attempt(entry)
                    entries_honest_null += 1
                    continue

                verdict = verdicts[0]
                outcome = draft.get("outcome", "no-credible-sources")

                if verdict.get("status") == "pass" and outcome == "sources-found" and draft.get("citations"):
                    path = _write_finding_file(research_root, entry, draft)
                    entry["status"] = "done"
                    entries_sourced += 1
                    log.info("research-quest: sourced %s -> %s", entry.get("id"), path)
                else:
                    _record_attempt(entry)
                    entries_honest_null += 1

            except Exception as exc:  # noqa: BLE001 — one entry's failure must not abort the batch
                log.warning("research-quest: entry %s failed: %s", entry.get("id"), exc)
                _record_attempt(entry)
                entries_honest_null += 1
                continue

    _write_backlog(backlog_path, entries)

    return {
        "entries_targeted": len(pending),
        "entries_sourced": entries_sourced,
        "entries_honest_null": entries_honest_null,
    }


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_research_quest(args) -> int:
    """Source pending entries in the Dowser research backlog."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    limit: int = getattr(args, "limit", 5) or 5
    dry_run: bool = bool(getattr(args, "dry_run", False))

    try:
        summary = run_research_quest(limit=limit, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2

    print("research-quest: run complete")
    print(f"  entries_targeted:   {summary['entries_targeted']}")
    print(f"  entries_sourced:    {summary['entries_sourced']}")
    print(f"  entries_honest_null:{summary['entries_honest_null']}")
    note = summary.get("note")
    if note:
        print(f"  note: {note}")
    return 0
