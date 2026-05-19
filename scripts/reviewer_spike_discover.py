#!/usr/bin/env python3
"""List recently-landed lapis-pm targets with their reviewer'd PR numbers.

Reads mem keys pm/landed/lapis-pm-* and emits target_id:pr_number pairs that have
recorded Claude reviewer verdicts in episodic. Output suitable to feed into
reviewer_spike.py --pairs.

Usage:
    python3 /srv/lapis/lapis-pm/scripts/reviewer_spike_discover.py [--limit 10] [--mixed]

--mixed biases selection toward a mix of clean / fixable / needs-human verdicts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/srv/lapis/lapis-pm")
sys.path.insert(0, "/srv/git/agents-core-working")

from lapis_pm import episodic  # noqa: E402


def _mem_keys_prefix(prefix: str) -> list[str]:
    """List mem keys matching a prefix via the mem CLI."""
    import subprocess
    proc = subprocess.run(
        ["mem", "search", prefix],
        capture_output=True, text=True, timeout=30,
    )
    keys: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            # mem search lines start with the key; sometimes followed by [tags]
            key = line.split()[0]
            keys.append(key)
    return keys


def _mem_get(key: str) -> str:
    import subprocess
    proc = subprocess.run(
        ["mem", "get", key],
        capture_output=True, text=True, timeout=10,
    )
    return proc.stdout


def _get_claude_verdict(target_id: str, pr_number: int) -> dict | None:
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    last_comment = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                if verdict_val != "pending":
                    last_comment = c
    if last_comment is None:
        return None
    content = last_comment.content
    try:
        json_part = content.split("\n", 1)[-1].strip()
        return json.loads(json_part)
    except (json.JSONDecodeError, IndexError):
        for t in last_comment.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                return {"verdict": t.split(":verdict=")[-1], "issues": [], "confidence": 0.0}
        return None


def _scan_comments_for_verdicts(
    comments_dir: Path,
    target_prefix: str | None,
    include_parse_errors: bool = False,
) -> list[tuple[str, int, str]]:
    """Walk comments jsonl files and pull (target_id, pr_number, verdict) for every clean reviewer verdict found.

    Filters out non-ground-truth records:
    - `_parse_error` content: Claude reviewer JSON parse failed; verdict was set to needs-human as fallback
    - `prior_index` / `status: addressed` content: post-fixer-retry tracking records that inherit verdict tag
    - records whose verdict field is missing or empty

    The verdict is read from CONTENT not from the tag; the tag can be stale on follow-up records.
    Most-recent verdict per (target_id, pr_number) wins.
    """
    import re as _re
    tag_re = _re.compile(r"^pm:reviewer:pr=(\d+):cycle=\d+:verdict=([a-z\-]+)$")
    # (target, pr) -> (verdict, ts, is_parse_error)
    latest: dict[tuple[str, int], tuple[str, float, bool]] = {}

    for path in sorted(comments_dir.glob("*.jsonl")):
        target_id = path.stem
        if target_prefix and not target_id.startswith(target_prefix):
            continue
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tags = rec.get("tags") or []
                    ts_raw = rec.get("created_at") or rec.get("ts") or ""
                    try:
                        import datetime as _dt
                        ts = _dt.datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        ts = 0.0

                    # Need a verdict tag to identify this as a reviewer record
                    pr_number = None
                    for t in tags:
                        m = tag_re.match(t)
                        if m:
                            pr_number = int(m.group(1))
                            if m.group(2) == "pending":
                                pr_number = None
                            break
                    if pr_number is None:
                        continue

                    # Parse content JSON; reviewer verdicts have shape
                    # "<header>\n{...verdict json...}" so split at first newline.
                    content = rec.get("content") or ""
                    json_part = content.split("\n", 1)[-1].strip()
                    try:
                        body = json.loads(json_part)
                    except (json.JSONDecodeError, ValueError):
                        # Cannot parse content as JSON — skip; ground truth unknown.
                        continue
                    if not isinstance(body, dict):
                        continue

                    # Filter post-fixer-retry tracking records (have prior_index/status keys
                    # but represent a follow-up note, not a fresh verdict).
                    if "prior_index" in body or body.get("status") == "addressed":
                        continue

                    is_parse_error = "_parse_error" in body
                    if is_parse_error and not include_parse_errors:
                        continue

                    verdict_val = (body.get("verdict") or "").strip()
                    if verdict_val not in ("clean", "fixable", "needs-human"):
                        continue

                    key = (target_id, pr_number)
                    prior = latest.get(key)
                    if prior is None or ts > prior[1]:
                        latest[key] = (verdict_val, ts, is_parse_error)
        except OSError:
            continue

    return [(t, p, v) for (t, p), (v, _ts, _err) in latest.items()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10, help="Max pairs to emit")
    ap.add_argument("--mixed", action="store_true",
                    help="Bias selection toward a mix of clean/fixable/needs-human verdicts")
    ap.add_argument("--target-prefix", default="lapis-pm-",
                    help="Only consider targets whose id starts with this prefix (default lapis-pm-)")
    ap.add_argument("--comments-dir", default="/srv/lapis/targets/comments",
                    help="Episodic comments directory")
    args = ap.parse_args()

    comments_dir = Path(args.comments_dir)
    candidates = _scan_comments_for_verdicts(comments_dir, args.target_prefix or None)
    print(f"# Scanned {comments_dir}, found {len(candidates)} targets/PRs with non-pending verdicts (prefix={args.target_prefix!r})",
          file=sys.stderr)

    if args.mixed:
        by_verdict: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        for c in candidates:
            by_verdict[c[2]].append(c)
        # Round-robin across verdict buckets
        picked: list[tuple[str, int, str]] = []
        buckets = list(by_verdict.values())
        while len(picked) < args.limit and any(buckets):
            for b in buckets:
                if b and len(picked) < args.limit:
                    picked.append(b.pop(0))
        candidates = picked
    else:
        candidates = candidates[: args.limit]

    pairs = ",".join(f"{t}:{p}" for t, p, _v in candidates)
    print("# Verdict breakdown:", file=sys.stderr)
    breakdown: dict[str, int] = defaultdict(int)
    for _, _, v in candidates:
        breakdown[v] += 1
    for v, n in breakdown.items():
        print(f"#   {v}: {n}", file=sys.stderr)
    print("\n# Pairs (target_id:pr_number,...):", file=sys.stderr)
    for t, p, v in candidates:
        print(f"#   {t} PR #{p}  ({v})", file=sys.stderr)
    print(pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
