"""Authority gate for PR auto-merge / advisory / hold classification.

We do not call `digest.py screen` as a subprocess (it posts a PR comment
rather than emitting structured JSON). Instead, we run an inline Sonnet
verdict via call_claude_cli with json_mode, then apply path/size/CI rules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from agents_core.llm import call_claude_cli
from agents_core.forgejo import get_pr, get_pr_diff


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
]

MAX_AUTO_LOC = 400
DIFF_PATH_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
DIFF_HUNK_LINE_RE = re.compile(r"^[+-](?![+-])", re.MULTILINE)


@dataclass
class PRClassification:
    verdict: str               # "auto" | "advisory" | "hold"
    screen_verdict: str        # "clean" | "fixable" | "needs-human" | "unknown"
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
    return sorted(set(DIFF_PATH_RE.findall(diff_text)))


def diff_loc(diff_text: str) -> int:
    return len(DIFF_HUNK_LINE_RE.findall(diff_text))


def is_held_path(path: str) -> bool:
    return any(p.search(path) for p in HELD_PATTERNS)


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
    """Inline structured screen via Sonnet. Returns parsed JSON or fallback dict."""
    if len(diff_text) > 60000:
        diff_text = diff_text[:60000] + "\n\n... (diff truncated)"
    user = (
        f"PR #{pr_number} in {repo}.\n\n"
        f"Spec context:\n{spec_summary}\n\n"
        f"Diff:\n```diff\n{diff_text}\n```"
    )
    raw = call_claude_cli(
        prompt=user, system=SCREEN_SYSTEM,
        model="sonnet", timeout=300, json_mode=True,
    )
    if not raw:
        return {"verdict": "needs-human", "issues": [], "confidence": 0.0,
                "reason": "screen call returned empty"}
    # Strip code fences just in case
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
             pm_authority: str = "advisory") -> PRClassification:
    """Decide auto/advisory/hold for a PR.

    pm_authority is the target's setting. "advisory" never auto-merges.
    """
    pr = get_pr(repo, pr_number)
    diff_text = get_pr_diff(repo, pr_number)

    paths = changed_paths(diff_text)
    loc = diff_loc(diff_text)

    held_hits = [p for p in paths if is_held_path(p)]
    reasons: list[str] = []

    screen_result = screen(repo, pr_number, spec_summary, diff_text)
    sv = screen_result.get("verdict", "needs-human")

    # Derive verdict
    if held_hits:
        reasons.append(f"held path(s) touched: {', '.join(held_hits[:5])}")
        verdict = "hold"
    elif sv == "needs-human":
        reasons.append("screen verdict: needs-human")
        verdict = "hold"
    elif sv == "clean" and pm_authority == "auto" and loc < MAX_AUTO_LOC \
            and pr.get("mergeable") is not False:
        verdict = "auto"
        reasons.append(f"clean screen, {loc} LOC, mergeable")
    elif sv == "clean" and pm_authority == "advisory":
        verdict = "advisory"
        reasons.append("clean screen but target authority is advisory")
    elif sv == "fixable":
        verdict = "advisory"
        reasons.append("fixable issues found")
    else:
        verdict = "advisory"
        reasons.append(f"defaulting to advisory (screen={sv}, loc={loc})")

    return PRClassification(
        verdict=verdict,
        screen_verdict=sv,
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
