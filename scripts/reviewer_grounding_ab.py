#!/usr/bin/env python3
"""A2/A4 grounding-experiment runner — lapis-pm-reviewer-absence-grounding-rule-v0.

Dispatches a `reviewer_fresh` (or `reviewer_fresh_contractor`) seat twice against
the same PR head — once with `absence_grounding` off, once on — via the existing
forced-dispatch path (`pm_core.force_dispatch`), and writes both raw verdicts
plus a comparison row to a CSV.

This is the runner for experiment arm A4 (see the spec's "experiment this
unlocks" table). Arm A2 (opencode harness, off) is manual and outside this
script's scope.

Poison-pill check (DoD 6): for every issue carrying an `evidence` value, this
script independently re-runs the quoted command against the PR's head branch
(in an isolated worktree — never the shared working clone) and classifies the
result:
  - corroborated  — quoted output is a substring of the re-run's stdout.
  - fabricated    — it is not. This is the state the check exists to catch.
  - search_failed — the re-run exited non-zero, timed out, or the command
                     could not be parsed into something re-runnable. A failed
                     search is NOT evidence of absence and is never scored as
                     corroborated.
  - no_evidence   — the issue carried no `evidence` value. Expected in the
                     flag-off arm; itself a finding about instruction-following
                     in the flag-on arm.

Only a conservative allowlist of read-only search binaries (grep, rg, git
grep/log/show/diff/status/ls-files, find, ls, cat, head, tail) is ever
re-executed. Anything else — or anything that fails to shlex.split — is
`search_failed`, per the "could not be parsed into something re-runnable"
clause of DoD 6, not silently skipped or run anyway.

Usage:
    # From /data/agents (so FORGEJO_TOKEN is on PATH):
    export $(grep FORGEJO_TOKEN /data/agents/config/conductor.env | xargs)
    python3 /srv/lapis/lapis-pm/scripts/reviewer_grounding_ab.py \\
        --target my-target --pr 42 --repo lapis-pm --seat reviewer_fresh \\
        --out /tmp/grounding_ab_results.csv

Run once per seat (reviewer_fresh, reviewer_fresh_contractor) per DoD "seat
coverage" — the CSV appends across runs so both seats land in one file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/srv/lapis/lapis-pm")
sys.path.insert(0, "/srv/git/agents-core-working")

from lapis_pm import pm_core  # noqa: E402
from agents_core.shaper import Shaper  # noqa: E402
from agents_core.forgejo import get_open_prs  # noqa: E402
from agents_core.worktree import setup_worktree, teardown_worktree  # noqa: E402


# Read-only search binaries this script will ever re-execute. Anything else
# quoted in an `evidence` string is search_failed, not run — this is a
# poison-pill checker, not a general command runner.
ALLOWED_BINARIES = {"grep", "rg", "git", "find", "ls", "cat", "head", "tail"}
ALLOWED_GIT_SUBCOMMANDS = {"grep", "log", "show", "diff", "status", "ls-files"}

# Named phrases from the spec's absence-grounding block (Change 1) — an issue
# whose note matches one of these is an "absence-style" finding and its
# evidence (or lack of it) is the thing DoD 6 exists to check.
ABSENCE_PHRASES = (
    "is not defined", "is never called", "is not wired in",
    "could not confirm", "there is no handler for",
)

CSV_FIELDNAMES = [
    "row_type", "target_id", "pr_number", "seat", "grounding",
    "task_id", "queue_state", "verdict", "issue_count", "absence_issue_count",
    "corroborated", "fabricated", "search_failed", "no_evidence",
    "error", "raw_path",
]


def _parse_command(first_line: str) -> list[str] | None:
    line = first_line.strip()
    for prefix in ("$", ">"):
        if line.startswith(prefix):
            line = line[1:].strip()
    if not line:
        return None
    try:
        argv = shlex.split(line)
    except ValueError:
        return None
    if not argv or argv[0] not in ALLOWED_BINARIES:
        return None
    if argv[0] == "git" and (len(argv) < 2 or argv[1] not in ALLOWED_GIT_SUBCOMMANDS):
        return None
    return argv


def classify_evidence(evidence: str | None, cwd: str) -> tuple[str, str]:
    """Classify one issue's `evidence` field. Returns (state, detail).

    `evidence` is expected to carry the exact command on its first line and
    the claimed output on the rest, per the reference wording in the spec's
    appendix. The exact command form isn't mandated further than that, so
    the first line is treated as the command and everything after as the
    claimed output — deliberately NOT an exact line-count match (DoD 6:
    "brittle... would misclassify honest evidence as fabricated"); a
    substring check is the load-bearing one.
    """
    if not evidence or not evidence.strip():
        return "no_evidence", ""
    lines = evidence.strip().splitlines()
    argv = _parse_command(lines[0])
    if argv is None:
        return "search_failed", "command could not be parsed into something re-runnable"
    claimed_output = "\n".join(lines[1:]).strip()
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return "search_failed", f"re-run error: {exc}"
    # grep/rg/git-grep exit 1 for "ran clean, no matches" — not an error.
    # Exit >=2 is a real failure for those tools; any nonzero is a failure
    # for everything else in the allowlist.
    is_grep_like = argv[0] in ("grep", "rg") or (argv[0] == "git" and argv[1] == "grep")
    ok = proc.returncode in (0, 1) if is_grep_like else proc.returncode == 0
    if not ok:
        return "search_failed", f"re-run exited {proc.returncode}: {proc.stderr[:200]}"
    actual = proc.stdout
    if claimed_output:
        if claimed_output in actual:
            return "corroborated", ""
        return "fabricated", "quoted output is not a substring of the re-run's stdout"
    # No claimed output text quoted — a plausible "ran clean, no hits" claim.
    # Score it against whether the re-run actually produced nothing either.
    if actual.strip() == "":
        return "corroborated", ""
    return "fabricated", "claimed no output but the re-run produced output"


def _resolve_open_pr(repo: str, pr_number: int) -> dict:
    for pr in get_open_prs(repo):
        if pr.get("number") == pr_number:
            return pr
    raise RuntimeError(f"PR #{pr_number} not found among open PRs for {repo}")


def _dispatch_arm(target_id: str, seat: str, grounding_on: bool, intent: str) -> str:
    """Force-dispatch one arm with absence_grounding forced on/off.

    Reuses pm_core.force_dispatch — the existing forced-dispatch path — rather
    than hand-rolling a second reviewer invocation (per spec §Change 3).
    Monkeypatches pm_core._absence_grounding_enabled for the duration of this
    one call so both arms can run against the same target/PR without
    round-tripping registry.yaml edits between dispatches.
    """
    orig = pm_core._absence_grounding_enabled
    pm_core._absence_grounding_enabled = lambda _agent_type: grounding_on
    try:
        return pm_core.force_dispatch(target_id, seat, intent)
    finally:
        pm_core._absence_grounding_enabled = orig


def _wait_for_output(task_id: str, timeout_s: int, poll_interval_s: int) -> tuple[str | None, str]:
    """Poll for the dispatch's output file. Returns (raw_text_or_None, state)
    where state is "completed" | "failed" | "timeout"."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        path = pm_core._gpu_output_path(task_id)
        if path is not None:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            failed = path.parent in (pm_core.FAILED_DIR, pm_core.CLAUDE_QUEUE_FAILED_DIR)
            return text, ("failed" if failed else "completed")
        time.sleep(poll_interval_s)
    return None, "timeout"


def _extract_verdict(raw_text: str) -> dict | None:
    raw_output = raw_text.strip()
    if raw_output.startswith("```"):
        raw_output = re.sub(r"^```(?:json)?\s*", "", raw_output)
        raw_output = re.sub(r"\s*```$", "", raw_output.strip())
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return pm_core._recover_reviewer_verdict(raw_output)


def _dump_raw(target_id: str, task_id: str, raw_text: str, verdict: dict | None = None) -> str:
    out_dir = Path("/tmp/reviewer_grounding_ab_raw") / target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out_dir / f"{task_id}.txt"
    body = f"# task_id: {task_id}\n# --- raw ---\n{raw_text}\n"
    if verdict is not None:
        body += f"# --- parsed (with evidence-check annotations) ---\n{json.dumps(verdict, indent=2)}\n"
    sidecar.write_text(body, encoding="utf-8")
    return str(sidecar)


def _run_arm(target_id: str, repo: str, pr_number: int, seat: str, grounding_on: bool,
             timeout_s: int, poll_interval_s: int) -> dict:
    arm_label = "on" if grounding_on else "off"
    print(f"=== arm {seat}:{arm_label} ===", file=sys.stderr)
    intent = f"reviewer_grounding_ab probe PR #{pr_number} (absence_grounding={grounding_on})"
    task_id = _dispatch_arm(target_id, seat, grounding_on, intent)
    print(f"  dispatched task {task_id}", file=sys.stderr)

    row = {
        "row_type": "arm",
        "target_id": target_id,
        "pr_number": pr_number,
        "seat": seat,
        "grounding": arm_label,
        "task_id": task_id,
        "queue_state": "",
        "verdict": "",
        "issue_count": 0,
        "absence_issue_count": 0,
        "corroborated": 0,
        "fabricated": 0,
        "search_failed": 0,
        "no_evidence": 0,
        "error": "",
        "raw_path": "",
    }

    raw_text, queue_state = _wait_for_output(task_id, timeout_s, poll_interval_s)
    row["queue_state"] = queue_state
    if raw_text is None:
        row["error"] = f"queue {queue_state} — no output after {timeout_s}s"
        return row

    verdict = _extract_verdict(raw_text)
    if verdict is None:
        row["error"] = "verdict JSON unparseable (even after recovery)"
        row["raw_path"] = _dump_raw(target_id, task_id, raw_text)
        return row

    row["verdict"] = verdict.get("verdict", "")
    issues = verdict.get("issues") or []
    row["issue_count"] = len(issues)

    repo_cwd = Shaper.resolve_repo_cwd(repo)
    pr = _resolve_open_pr(repo, pr_number)
    existing_branch = (pr.get("head") or {}).get("ref") or ""
    wt_task_id = f"grounding-ab-verify-{task_id}"
    wt_path = None
    if existing_branch:
        try:
            handle = setup_worktree(wt_task_id, repo_cwd, existing_branch)
            wt_path = handle.path
        except Exception as exc:
            print(f"  WARN: worktree setup for evidence re-run failed: {exc}", file=sys.stderr)

    try:
        verify_cwd = str(wt_path) if wt_path else repo_cwd
        for iss in issues:
            note = (iss.get("note") or "").lower()
            is_absence_style = any(p in note for p in ABSENCE_PHRASES)
            evidence = iss.get("evidence")
            if not is_absence_style and evidence is None:
                continue  # nothing to check on an ordinary issue with no evidence
            state, detail = classify_evidence(evidence, verify_cwd)
            iss["_evidence_state"] = state
            iss["_evidence_detail"] = detail
            row[state] = row.get(state, 0) + 1
            if is_absence_style:
                row["absence_issue_count"] += 1
    finally:
        if wt_path is not None:
            try:
                teardown_worktree(wt_task_id, repo_cwd)
            except Exception:
                pass

    row["raw_path"] = _dump_raw(target_id, task_id, raw_text, verdict)
    return row


def _comparison_row(target_id: str, pr_number: int, seat: str, off_row: dict, on_row: dict) -> dict:
    """A single row summarizing off-vs-on, per spec §Change 3 ("a comparison row")."""
    return {
        "row_type": "comparison",
        "target_id": target_id,
        "pr_number": pr_number,
        "seat": seat,
        "grounding": "off_vs_on",
        "task_id": f"{off_row['task_id']}->{on_row['task_id']}",
        "queue_state": f"{off_row['queue_state']}->{on_row['queue_state']}",
        "verdict": f"{off_row['verdict']}->{on_row['verdict']}",
        "issue_count": on_row["issue_count"] - off_row["issue_count"],
        "absence_issue_count": on_row["absence_issue_count"] - off_row["absence_issue_count"],
        # Total fabrications across BOTH arms — the headline poison-pill
        # number. Any nonzero value here means the CSV rows above must be
        # read by a human, not treated as a clean pass.
        "corroborated": off_row["corroborated"] + on_row["corroborated"],
        "fabricated": off_row["fabricated"] + on_row["fabricated"],
        "search_failed": off_row["search_failed"] + on_row["search_failed"],
        "no_evidence": off_row["no_evidence"] + on_row["no_evidence"],
        "error": "; ".join(e for e in (off_row["error"], on_row["error"]) if e),
        "raw_path": "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, help="Bound lapis-pm target id")
    ap.add_argument("--pr", type=int, required=True, help="PR number to dispatch both arms against")
    ap.add_argument("--repo", required=True, help="Repo name for the PR (e.g. lapis-pm)")
    ap.add_argument("--seat", choices=["reviewer_fresh", "reviewer_fresh_contractor"],
                     default="reviewer_fresh",
                     help="Which reviewer seat to run both arms through "
                          "(reviewer_fresh=27B/gravitywell, reviewer_fresh_contractor=DeepSeek). "
                          "Run this script once per seat for full seat coverage.")
    ap.add_argument("--out", required=True, help="CSV output path (appends if exists)")
    ap.add_argument("--timeout", type=int, default=900, help="Max seconds to wait per arm")
    ap.add_argument("--poll-interval", type=int, default=15, help="Seconds between completion polls")
    args = ap.parse_args()

    off_row = _run_arm(args.target, args.repo, args.pr, args.seat, False, args.timeout, args.poll_interval)
    on_row = _run_arm(args.target, args.repo, args.pr, args.seat, True, args.timeout, args.poll_interval)

    for row in (off_row, on_row):
        print(
            f"  [{row['grounding']}] verdict={row['verdict']!r} issues={row['issue_count']} "
            f"absence_issues={row['absence_issue_count']} corroborated={row['corroborated']} "
            f"fabricated={row['fabricated']} search_failed={row['search_failed']} "
            f"no_evidence={row['no_evidence']} error={row['error']!r}",
            file=sys.stderr,
        )

    comparison = _comparison_row(args.target, args.pr, args.seat, off_row, on_row)
    if comparison["fabricated"] > 0:
        print(
            f"  POISON PILL: {comparison['fabricated']} fabricated evidence string(s) "
            "across both arms — do not read either arm as a clean pass.",
            file=sys.stderr,
        )

    rows = [off_row, on_row, comparison]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
