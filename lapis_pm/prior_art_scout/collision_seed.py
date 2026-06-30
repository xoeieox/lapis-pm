"""Load unseen kami-collision survivors as Dowser seeds.

Reads COLLIDER_ROOT/<run_id>/_manifest.json files (newest-first), filters to
challenges + novel-seed verdicts, skips already-sidecarred hashes, and returns
up to batch_cap unseen records.

Graceful decay: if COLLIDER_ROOT is absent, not a dir, or empty, logs a warning
and returns []. Never crashes - absent collider infra is a valid system state.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .sidecar import load_sidecar

log = logging.getLogger(__name__)

_QUALIFYING_VERDICTS = {"challenges", "novel-seed"}
_IDEA_SECTION = "## Emergent Idea"


def _parse_emergent_idea(md_text: str) -> str | None:
    """Extract first paragraph after '## Emergent Idea'. Returns None on parse failure."""
    idx = md_text.find(_IDEA_SECTION)
    if idx == -1:
        return None
    rest = md_text[idx + len(_IDEA_SECTION):].lstrip("\n")
    lines = []
    for line in rest.splitlines():
        if not line.strip() and lines:
            break
        if line.strip():
            lines.append(line.strip())
    return " ".join(lines) if lines else None


def load_collision_seeds(
    collider_root: Path,
    batch_cap: int,
) -> list[dict]:
    """Return up to batch_cap unseen collision survivors (challenges + novel-seed).

    Each dict: {"hash": str, "idea": str, "verdict": str, "value_score": float,
                "atom_titles": list[str], "run_id": str}

    Reads COLLIDER_ROOT/<run_id>/_manifest.json for all run_ids (newest-first).
    For each unseen hash (no sidecar at collision/<hash>):
      - reads <run_id>/<hash>.md for emergent idea text
      - returns up to batch_cap records
    """
    if batch_cap <= 0:
        return []

    if not collider_root.exists() or not collider_root.is_dir():
        log.warning(
            "[collision_seed] COLLIDER_ROOT %s absent or not a directory; skipping collision intake",
            collider_root,
        )
        return []

    run_dirs = sorted(
        [d for d in collider_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,  # newest run_id first
    )
    if not run_dirs:
        log.warning("[collision_seed] COLLIDER_ROOT %s has no run subdirectories", collider_root)
        return []

    results: list[dict] = []
    for run_dir in run_dirs:
        if len(results) >= batch_cap:
            break

        manifest_path = run_dir / "_manifest.json"
        if not manifest_path.exists():
            continue

        try:
            entries = json.loads(manifest_path.read_text())
        except Exception as exc:
            log.warning("[collision_seed] failed to read manifest %s: %s", manifest_path, exc)
            continue

        for entry in entries:
            if len(results) >= batch_cap:
                break

            verdict = entry.get("verdict", "")
            col_hash = entry.get("hash", "")

            if verdict not in _QUALIFYING_VERDICTS:
                continue
            if not col_hash:
                continue
            # Dedup: skip if any sidecar already exists for this hash
            if load_sidecar("collision/" + col_hash) is not None:
                continue

            md_path = run_dir / f"{col_hash}.md"
            if not md_path.exists():
                log.warning(
                    "[collision_seed] missing md for hash %s in %s; skipping",
                    col_hash, run_dir.name,
                )
                continue

            try:
                md_text = md_path.read_text()
            except Exception as exc:
                log.warning("[collision_seed] failed to read %s: %s", md_path, exc)
                continue

            idea = _parse_emergent_idea(md_text)
            if not idea:
                log.warning(
                    "[collision_seed] could not parse emergent idea from %s; skipping", md_path
                )
                continue

            results.append({
                "hash": col_hash,
                "idea": idea,
                "verdict": verdict,
                "value_score": entry.get("value_score", 0.0),
                "atom_titles": entry.get("atom_titles", []),
                "run_id": run_dir.name,
            })

    return results
