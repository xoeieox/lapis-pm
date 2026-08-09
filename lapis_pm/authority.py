"""Authority gate for PR auto-merge / advisory / hold classification.

Static checks (held paths, size cap) determine whether the reviewer agent
should be dispatched. For auto-merge targets, an inline Sonnet screen still
runs as before. For advisory/hold targets the LLM verdict is a dispatched
Opus reviewer (see pm_core.py review-gate loop).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from agents_core.llm import call_claude_cli
from agents_core.forgejo import get_pr, get_pr_diff

logger = logging.getLogger(__name__)


# Paths whose touch forces a hold.
HELD_PATTERNS = [
    re.compile(r"^infra/"),
    re.compile(r"(^|/)systemd/"),
    re.compile(r"\.sql$"),
    re.compile(r"(^|/)schema/"),
    re.compile(r"(^|/)ARCHITECTURE\.md$"),
    re.compile(r"(^|/)SPEC\.md$"),
    re.compile(r"\.service$"),
    re.compile(r"\.timer$"),
    # --- lapis-pm-containment-held-paths-v0, Leg 1 (anchored, each with a rationale) ---
    # The hold list itself, and the classifier that reads it. Remove only if held-path
    # enforcement moves to a different module.
    re.compile(r"^lapis_pm/authority\.py$"),
    # The six-clause autonomous-merge predicate. Remove only if auto-resolve is retired.
    re.compile(r"^lapis_pm/auto_resolve\.py$"),
    # The unattended binder: the one path that binds+ticks with no human at 01:30 nightly.
    re.compile(r"^lapis_pm/bundle_autodispatch\.py$"),
    # Session hooks the PM executes on its own runs (router-portfolio-stop.py).
    re.compile(r"^lapis_pm/hooks/"),
    # Deploy manifest + sync scripts: how merged code reaches the running daemon.
    re.compile(r"^deploy/"),
    # Registers the hooks above. Held as a pair with lapis_pm/hooks/ — holding either alone
    # leaves an escape.
    re.compile(r"^\.claude/settings(\.local)?\.json$"),
    # The audit instrument: the checklist Erah follows when reviewing the machine's work.
    # Both copies — the flat legacy file and the live SKILL.md — or the sibling is the escape.
    re.compile(r"^\.claude/skills/pm-pr-review(\.md|/SKILL\.md)$"),
]

MAX_AUTO_LOC = 400
DIFF_INLINE_CAP = 200_000
DIFF_PATH_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
DIFF_PATH_RE_OLD = re.compile(r"^--- a/(.+)$", re.MULTILINE)
DIFF_HUNK_LINE_RE = re.compile(r"^[+-](?![+-])", re.MULTILINE)


class StaticOutcome(str, Enum):
    auto_hold_path = "auto_hold_path"   # touched a held path — skip reviewer
    size_exceeded = "size_exceeded"     # LOC exceeds auto-merge cap (auto only)
    static_pass = "static_pass"         # no static blockers


@dataclass
class PRClassification:
    verdict: str               # "auto" | "advisory" | "hold"
    screen_verdict: str        # "clean" | "fixable" | "needs-human" | "unknown"
    static_outcome: str        # StaticOutcome value
    reasons: list[str]
    issues: list[dict]
    pr_number: int
    repo: str
    title: str
    html_url: str
    changed_paths: list[str]
    diff_loc: int
    diff: str = ""             # raw diff text (used by brief for risk callouts)


def changed_paths(diff_text: str) -> list[str]:
    """Union the added ('+++ b/') and removed ('--- a/') sides of the diff.

    A pure deletion emits '+++ /dev/null' — the added-side capture alone misses it
    entirely, so a held file could be deleted without ever tripping the gate. The
    removed side ('--- a/<path>') catches deletions and renames-away; '/dev/null'
    is excluded from both sides (it appears on '--- a/' for pure additions).
    """
    added = set(DIFF_PATH_RE.findall(diff_text))
    removed = set(DIFF_PATH_RE_OLD.findall(diff_text))
    return sorted((added | removed) - {"/dev/null"})


def diff_loc(diff_text: str) -> int:
    return len(DIFF_HUNK_LINE_RE.findall(diff_text))


def is_held_path(path: str) -> bool:
    return any(p.search(path) for p in HELD_PATTERNS)


def render_systemd_semantics(diff_text: str, held_hits: list[str]) -> list[str]:
    """Extract OnCalendar semantics from held systemd timer/service diffs.

    Parses ADDED OnCalendar= lines from changed .timer/.service files in the diff,
    runs systemd-analyze calendar on each, and returns human-readable cadence lines.
    Failures degrade gracefully — no exception raised, just append unavailable note.
    """
    result = []

    # Find all .timer/.service files in held_hits
    systemd_files = [p for p in held_hits if p.endswith(".timer") or p.endswith(".service")]
    if not systemd_files:
        return result

    # Extract ADDED OnCalendar= lines from the diff
    oncalendar_pattern = re.compile(r"^\+OnCalendar=(.+)$", re.MULTILINE)
    matches = oncalendar_pattern.findall(diff_text)
    if not matches:
        return result

    # For each OnCalendar value, run systemd-analyze and extract the cadence
    for value in matches:
        value = value.strip()
        if not value:
            continue
        try:
            output = subprocess.run(
                ["/usr/bin/systemd-analyze", "calendar", value],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if output.returncode != 0:
                result.append(f"(semantics render unavailable: systemd-analyze returned {output.returncode})")
                continue

            # Parse the output: look for cadence line (Next elapse / From now)
            # The cadence line is what matters most for the human reading the brief
            lines = output.stdout.strip().split("\n")
            cadence = None
            for line in lines:
                if "Next elapse" in line or "From now" in line:
                    cadence = line.strip()
                    break

            if cadence:
                # Emit the cadence line (this is the critical info for the human)
                result.append(f"OnCalendar={value} -> {cadence}")
            else:
                # Only fall back to limited-detail note if we truly found no cadence
                result.append(f"OnCalendar={value} (parsed OK, limited detail)")

        except FileNotFoundError:
            result.append("(semantics render unavailable: systemd-analyze not found)")
            break
        except subprocess.TimeoutExpired:
            result.append("(semantics render unavailable: systemd-analyze timeout)")
            break
        except Exception as e:
            result.append(f"(semantics render unavailable: {type(e).__name__})")
            break

    return result


SCREEN_SYSTEM = """You are a strict PR screener. Read the diff and return JSON:
{"verdict": "clean" | "fixable" | "needs-human",
 "issues": [{"severity": "low|med|high", "path": "...", "note": "..."}],
 "confidence": 0.0-1.0}

"clean" = ready to merge. No bugs, no missing tests for behavior change, no obvious risks.
"fixable" = small concrete issues that an automated fixer can address. List them.
"needs-human" = architectural concerns, ambiguous intent, or risk requiring judgment.

Be terse. Only return the JSON object.
"""


def screen(repo: str, pr_number: int, spec_summary: str, diff_text: str) -> dict:
    """Inline structured screen via Sonnet.

    Used only for auto-merge authority targets and as the kill-switch fallback.
    Returns parsed JSON or fallback dict.
    """
    if len(diff_text) > DIFF_INLINE_CAP:
        diff_text = diff_text[:DIFF_INLINE_CAP] + "\n\n... (diff truncated)"
    user = (
        f"PR #{pr_number} in {repo}.\n\n"
        f"Spec context:\n{spec_summary}\n\n"
        f"Diff:\n```diff\n{diff_text}\n```"
    )
    raw = call_claude_cli(
        prompt=user, system=SCREEN_SYSTEM,
        model="sonnet", timeout=300, json_mode=True, log=logger.warning,
    )
    if not raw:
        return {"verdict": "needs-human", "issues": [], "confidence": 0.0,
                "reason": "screen call returned empty"}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"verdict": "needs-human", "issues": [], "confidence": 0.0,
                "reason": f"unparseable screen output: {raw[:200]}"}


def classify(repo: str, pr_number: int, spec_summary: str,
             pm_authority: str = "advisory",
             verification: str = "pm-live-test") -> PRClassification:
    """Classify a PR based on static checks.

    For auto-merge targets: also runs inline Sonnet screen. If verification is not
    'machine', the auto path downgrades to advisory (raises a brief) rather than merging.
    For advisory/hold targets: static checks only; Opus reviewer is dispatched
    by pm_core's review-gate loop (not here).
    """
    if "/" in repo:
        owner, repo_name = repo.split("/", 1)
    else:
        repo_name, owner = repo, None
    pr = get_pr(repo_name, pr_number, owner=owner)
    diff_text = get_pr_diff(repo_name, pr_number, owner=owner)

    paths = changed_paths(diff_text)
    loc = diff_loc(diff_text)

    held_hits = [p for p in paths if is_held_path(p)]

    # --- Static: held paths (applies to all authority levels) ---
    if held_hits:
        reasons = [f"held path(s) touched: {', '.join(held_hits[:5])}"]
        # Enrich with systemd semantics if applicable
        rendered = render_systemd_semantics(diff_text, held_hits)
        reasons.extend(rendered)
        return PRClassification(
            verdict="hold",
            screen_verdict="unknown",
            static_outcome=StaticOutcome.auto_hold_path,
            reasons=reasons,
            issues=[],
            pr_number=pr_number,
            repo=repo,
            title=pr.get("title", ""),
            html_url=pr.get("html_url", ""),
            changed_paths=paths,
            diff_loc=loc,
            diff=diff_text,
        )

    # --- Auto-merge path: inline Sonnet screen ---
    if pm_authority == "auto":
        # Gate: verification must be 'machine'. Not machine-verified => downgrade to advisory.
        if verification != "machine":
            return PRClassification(
                verdict="advisory",
                screen_verdict="unknown",
                static_outcome=StaticOutcome.static_pass,
                reasons=[
                    f"auto-tier downgraded: verification={verification!r} (needs PM touch)"
                ],
                issues=[],
                pr_number=pr_number,
                repo=repo,
                title=pr.get("title", ""),
                html_url=pr.get("html_url", ""),
                changed_paths=paths,
                diff_loc=loc,
                diff=diff_text,
            )
        screen_result = screen(repo, pr_number, spec_summary, diff_text)
        sv = screen_result.get("verdict", "needs-human")
        reasons: list[str] = []
        if sv == "needs-human":
            reasons.append("screen verdict: needs-human")
            verdict = "hold"
        elif sv == "clean" and loc < MAX_AUTO_LOC \
                and pr.get("mergeable") is not False:
            verdict = "auto"
            reasons.append(f"clean screen, {loc} LOC, mergeable")
        elif sv == "fixable":
            verdict = "advisory"
            reasons.append("fixable issues found")
        else:
            verdict = "advisory"
            reasons.append(f"defaulting to advisory (screen={sv}, loc={loc})")
        return PRClassification(
            verdict=verdict,
            screen_verdict=sv,
            static_outcome=StaticOutcome.static_pass,
            reasons=reasons,
            issues=list(screen_result.get("issues") or []),
            pr_number=pr_number,
            repo=repo,
            title=pr.get("title", ""),
            html_url=pr.get("html_url", ""),
            changed_paths=paths,
            diff_loc=loc,
            diff=diff_text,
        )

    # --- Advisory / hold path: static pass only ---
    # Opus reviewer will be dispatched by pm_core review-gate loop.
    reasons = [f"static checks passed, {loc} LOC — Opus reviewer will be dispatched"]
    return PRClassification(
        verdict="advisory",        # tentative; reviewer verdict drives final action
        screen_verdict="unknown",
        static_outcome=StaticOutcome.static_pass,
        reasons=reasons,
        issues=[],
        pr_number=pr_number,
        repo=repo,
        title=pr.get("title", ""),
        html_url=pr.get("html_url", ""),
        changed_paths=paths,
        diff_loc=loc,
        diff=diff_text,
    )
