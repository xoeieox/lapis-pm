"""Absence-claim grounding (lapis-pm-degraded-panel-confidence-and-absence-claims-v0, Leg 2).

Incident: the reviewer flagged `_check_equal` as "not defined anywhere in the
diff or in the existing code I read" — HIGH severity, confidence 0.9. False:
`_check_equal` is defined at agents_core/phala_tee.py:427 and called at :658
on the PR head. The reviewer reasoned over diff hunks only; the definition
sat untouched 230 lines outside the diff, so absence-from-view read as
absence-from-codebase.

This module applies `feedback/sweeps-find-candidates-not-absence` to code
review: a finding that asserts something does NOT exist is the class most
likely to be wrong for exactly this reason, so it gets checked against the
full file at the PR head — not the diff — before it is allowed to ship as
actionable.

Per R2 (ratified 2026-08-08, closed — not open for relitigation): an
ungroundable claim (file unreachable, symbol ambiguous, etc.) stays
actionable and is marked unverified. It is never treated as a refutation
and never as a confirmation, and it is never routed anywhere else on its
own — a real missing-symbol bug is exactly the class this seat exists to
catch, and on a healthy panel the other legs are corroborating it. The
starved-panel wedge risk is Leg 1's problem, not this one's.
"""

from __future__ import annotations

import re

# Conservative absence-claim phrasing. A missed absence-claim just leaves
# today's (unverified) behaviour — acceptable. A wrongly-matched claim risks
# grounding text that was never asserting non-existence, so this stays a
# tight, literal phrase list rather than a broad heuristic.
_ABSENCE_PHRASES = (
    "is not defined",
    "is not called",
    "is never defined",
    "is never called",
    "is never implemented",
    "is not implemented",
    "does not exist",
    "doesn't exist",
    "not defined anywhere",
    "not found in",
    "was removed",
    "has been removed",
    "no definition of",
    "no such function",
    "no such method",
    "missing function definition",
    "missing definition",
)

# Backtick-quoted identifier, e.g. `_check_equal` or `agents_core.phala_tee`.
# Conservative on purpose (DoD: "keep symbol extraction conservative") — only
# an explicitly-quoted identifier is extracted; bare prose names are not
# guessed at, per "when the pattern is ambiguous, let the finding through".
_BACKTICK_IDENT_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]{1,80})`")


def is_absence_claim(note: str) -> bool:
    """True if `note` asserts non-existence of something (missing / not
    defined / removed / never implemented — see `feedback/sweeps-find-candidates-not-absence`).
    """
    if not note:
        return False
    lowered = note.lower()
    return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


def extract_symbol(note: str) -> str | None:
    """Extract the backtick-quoted symbol name an absence claim is about.

    Returns the first backtick-quoted identifier, or None if none is present
    (ambiguous — the caller should treat that as unrunnable, never as a
    confirmation or refutation).
    """
    if not note:
        return None
    m = _BACKTICK_IDENT_RE.search(note)
    if not m:
        return None
    ident = m.group(1)
    # A dotted module path like `agents_core.phala_tee` — the grounding check
    # is symbol-in-file, so use the last segment (the actual name).
    return ident.rsplit(".", 1)[-1]


def fetch_file_at_ref(repo: str, path: str, ref: str, owner: str | None = None) -> str | None:
    """Fetch a file's full content at `ref` from Forgejo. Returns None on any
    failure (repo/path/ref not found, network error, etc.) — the caller
    treats None as "check cannot run" (R2: stays actionable, marked
    unverified), never as a refutation.
    """
    try:
        import httpx
        from agents_core.forgejo import API, _headers, _owner

        r = httpx.get(
            f"{API}/repos/{_owner(owner)}/{repo}/raw/{path}",
            headers=_headers(),
            params={"ref": ref},
            timeout=15,
        )
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def symbol_in_file(symbol: str, file_text: str) -> bool:
    """True if `symbol` appears as a whole identifier anywhere in `file_text`
    (definition or reference — grounding only needs "does it exist", not
    "is it defined here").
    """
    return re.search(rf"\b{re.escape(symbol)}\b", file_text) is not None


def ground_issue(
    issue: dict,
    repo: str,
    pr_number: int,
    pr_head_sha: str | None,
    *,
    fetch_file=fetch_file_at_ref,
) -> tuple[str, dict]:
    """Ground a single reviewer issue if it makes an absence claim.

    Returns (outcome, annotated_issue) where outcome is one of:
      "not_absence_claim" — issue text doesn't assert non-existence; passed through unchanged.
      "refuted"            — symbol found in the full file at PR head; drop from actionable set.
      "actionable"          — symbol genuinely not found; issue survives unchanged.
      "unverified"          — check could not run (no symbol, no path, fetch failed,
                               no head sha); issue survives, marked unverified (R2).
    """
    note = issue.get("note", "") or ""
    if not is_absence_claim(note):
        return "not_absence_claim", issue

    symbol = extract_symbol(note)
    path = issue.get("path")

    if not symbol or not path or path in ("?", "") or not pr_head_sha:
        annotated = dict(issue)
        annotated["absence_check"] = "unverified"
        return "unverified", annotated

    try:
        file_text = fetch_file(repo, path, pr_head_sha)
    except Exception:
        # Any failure in the fetch (including a caller-supplied `fetch_file`
        # that raises rather than returning None) is "check cannot run" — R2:
        # never a refutation, never a confirmation.
        file_text = None
    if file_text is None:
        annotated = dict(issue)
        annotated["absence_check"] = "unverified"
        return "unverified", annotated

    if symbol_in_file(symbol, file_text):
        annotated = dict(issue)
        annotated["absence_check"] = "refuted"
        annotated["refuted_location"] = f"{path}@{pr_head_sha}"
        return "refuted", annotated

    annotated = dict(issue)
    annotated["absence_check"] = "actionable"
    return "actionable", annotated


def ground_verdict_issues(
    verdict: dict,
    repo: str,
    pr_number: int,
    pr_head_sha: str | None,
    *,
    fetch_file=fetch_file_at_ref,
) -> dict:
    """Post-process a verdict's `issues` list, grounding absence claims
    against the full file at the PR head (DoD #5, #6).

    Mutates and returns `verdict`. `issues` keeps every non-refuted finding
    (unchanged issues, unverified absence claims, and confirmed-absent
    findings all stay actionable). Refuted findings are dropped from
    `issues` and moved to `refuted_absence_findings`, annotated with where
    the symbol was found, so the record shows the reviewer was wrong rather
    than showing nothing.
    """
    issues = verdict.get("issues") or []
    kept: list[dict] = []
    refuted: list[dict] = []
    for issue in issues:
        outcome, annotated = ground_issue(
            issue, repo, pr_number, pr_head_sha, fetch_file=fetch_file,
        )
        if outcome == "refuted":
            refuted.append(annotated)
        else:
            kept.append(annotated)

    verdict["issues"] = kept
    if refuted:
        verdict["refuted_absence_findings"] = refuted
    return verdict
