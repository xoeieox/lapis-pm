"""Per-topic diff between two bench run captures.

At v1 the delta is a human-readable text summary — the reviewer reads the pair
and makes the correctness call.  A future extension could add LLM-judge
scoring; do not implement that now.

Phase 2 extension point: when Synapse-enabled runs land, call this module with
a baseline capture and a Synapse capture to measure what retrieval contributes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def compare(baseline_path: Path, labeled_path: Path) -> dict:
    """Return per-pair diff between two run captures.

    Args:
        baseline_path: Path to the baseline capture JSON.
        labeled_path: Path to the labeled (e.g. Synapse-enabled) capture JSON.

    Returns:
        ``{pair_id: {"loud_delta": str, "quiet_delta": str}}``

        An empty string delta means the responses are identical.
        A non-empty string delta is a short human-readable summary.
        Pairs present in one capture but not the other are noted explicitly.
    """
    baseline = _load(baseline_path)
    labeled = _load(labeled_path)

    baseline_by_id = {r["pair_id"]: r for r in baseline["results"]}
    labeled_by_id = {r["pair_id"]: r for r in labeled["results"]}

    all_ids = sorted(set(baseline_by_id) | set(labeled_by_id))
    return {
        pair_id: _diff_pair(baseline_by_id.get(pair_id), labeled_by_id.get(pair_id))
        for pair_id in all_ids
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _diff_pair(baseline: dict | None, labeled: dict | None) -> dict:
    if baseline is None:
        return {
            "loud_delta": "<missing in baseline>",
            "quiet_delta": "<missing in baseline>",
        }
    if labeled is None:
        return {
            "loud_delta": "<missing in labeled>",
            "quiet_delta": "<missing in labeled>",
        }
    return {
        "loud_delta": _text_delta(baseline["loud"]["response"], labeled["loud"]["response"]),
        "quiet_delta": _text_delta(baseline["quiet"]["response"], labeled["quiet"]["response"]),
    }


def _text_delta(a: str, b: str) -> str:
    """Return a human-readable summary of the difference between *a* and *b*."""
    if a == b:
        return ""
    if not a:
        return f"[baseline empty] labeled={b[:200]!r}"
    if not b:
        return f"[labeled empty] baseline={a[:200]!r}"
    # Truncate to 200 chars each side to keep the diff scannable
    return f"baseline={a[:200]!r} | labeled={b[:200]!r}"


# ---------------------------------------------------------------------------
# Module entry point: python -m lapis_pm.bench.compare <baseline> <labeled>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "usage: python -m lapis_pm.bench.compare <baseline.json> <labeled.json>",
            file=sys.stderr,
        )
        sys.exit(2)

    deltas = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(deltas, indent=2))
    if all(not d["loud_delta"] and not d["quiet_delta"] for d in deltas.values()):
        print("(no deltas — captures are identical)")
