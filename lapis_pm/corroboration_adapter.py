"""Lapis-PM reviewer corroboration adapter.

LapisPMReviewerAdapter implements the SubstrateAdapter protocol shape
from the cross-node-corroboration-v0 spec.

- retrieve: extracts doc-mentioned-identifier claims from a PR diff;
  greps repo to check existence; also greps vault when available.
- score: single constrained-JSON LLM call against the retrieved substrate;
  returns CorroborationResult with drift_class in:
  {"missing_referent", "stale_referent", "renamed_referent", "none"}.

Invariants (from spec §Invariants):
- Read-only: never mutates reviewer verdict mainline fields.
- Flag-shaped, not block-shaped: findings are advisory at v0.
- Gracefully degrades to verdict="uncertain" on substrate-unavailable or
  LLM-unavailable.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Data shapes (CorroborationResult is evidence-packet-shaped from day 1)
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """SCP-shaped attribution: source path + relevant text snippet."""
    source: str   # e.g. "repo:lapis-pm:lapis_pm/pm_core.py:42"
    text: str     # the relevant text (truncated to 200 chars)


@dataclass
class CorroborationResult:
    """Evidence-packet result from a corroboration pass.

    Shape is stable from v0; downstream consumers (claude-view Atmosphere
    panel, mem.db evidence keys, future federation) depend on this shape.
    """
    verdict: Literal["clean", "flagged", "uncertain"]
    claim: str
    citations: list[Citation]
    freshness_stamp: str       # ISO timestamp (UTC)
    scope_id: str              # e.g. "repo:lapis-pm"
    drift_class: str | None = None   # "missing_referent" | "stale_referent" | "renamed_referent" | "none"
    notes: str | None = None
    primitive_decomposition: dict | None = None  # entry-time decomposition; compost-routing handle (spec Compost invariant)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "claim": self.claim,
            "citations": [{"source": c.source, "text": c.text} for c in self.citations],
            "freshness_stamp": self.freshness_stamp,
            "scope_id": self.scope_id,
            "drift_class": self.drift_class,
            "notes": self.notes,
            "primitive_decomposition": self.primitive_decomposition,
        }


# ---------------------------------------------------------------------------
# Identifier extraction from diff text
# ---------------------------------------------------------------------------

_IDENTIFIER_PATTERNS = [
    # Python / JS function and class definitions on added/context lines
    r'\bdef\s+([A-Za-z_][A-Za-z0-9_]{2,})\b',
    r'\bclass\s+([A-Za-z_][A-Za-z0-9_]{2,})\b',
    r'\bfunc\s+([A-Za-z_][A-Za-z0-9_]{2,})\b',
    # Backtick-quoted identifiers in prose / commit messages
    r'`([A-Za-z_][A-Za-z0-9_.]{2,})`',
    # Dotted module paths (e.g. lapis_pm.corroboration_adapter)
    r'\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]+){1,3})\b',
]

_MAX_IDENTIFIERS = 15  # cap to bound LLM prompt size


def _extract_identifiers(diff_text: str) -> list[str]:
    """Extract doc-mentioned identifiers from a PR diff.

    Pulls:
    - Filenames from diff headers (--- a/... / +++ b/...)
    - Function/class names from code lines
    - Backtick-quoted identifiers from prose lines
    - Dotted module paths

    Returns deduplicated list, longest-first (module paths before segments).
    Capped at _MAX_IDENTIFIERS to bound downstream processing.
    """
    identifiers: set[str] = set()

    # File paths from unified diff headers
    for m in re.finditer(r'^(?:---|\+\+\+)\s+[ab]/(.+)$', diff_text, re.MULTILINE):
        path = m.group(1).strip()
        if path and path != '/dev/null':
            # Include both full path and basename without extension
            identifiers.add(path)
            stem = Path(path).stem
            if len(stem) >= 3:
                identifiers.add(stem)

    # Named identifiers from code and prose
    for pattern in _IDENTIFIER_PATTERNS:
        for m in re.finditer(pattern, diff_text):
            name = m.group(1).strip()
            # Skip very short names and private/dunder identifiers
            if len(name) >= 3 and not name.startswith('__'):
                identifiers.add(name)

    # Sort longest-first so module paths (most specific) come before segments
    return sorted(identifiers, key=len, reverse=True)[:_MAX_IDENTIFIERS]


# ---------------------------------------------------------------------------
# Substrate retrieval helpers
# ---------------------------------------------------------------------------

_VAULT_PATH = Path("/srv/git/inertia-vault-working")
_GREP_TIMEOUT = 8  # seconds; grep over large trees can be slow


def _grep_repo(identifier: str, repo_path: str) -> list[dict]:
    """Grep repo for identifier. Returns list of {file, line, text} dicts."""
    try:
        proc = subprocess.run(
            [
                "grep", "-rn",
                "--include=*.py", "--include=*.md", "--include=*.yaml",
                "-l",            # list files first to avoid huge output
                "--", identifier, repo_path,
            ],
            capture_output=True, text=True, timeout=_GREP_TIMEOUT,
        )
        # Get actual lines for the first few matching files
        matching_files = proc.stdout.splitlines()[:3]
        results = []
        for fpath in matching_files:
            try:
                line_proc = subprocess.run(
                    ["grep", "-n", "-m", "2", "--", identifier, fpath],
                    capture_output=True, text=True, timeout=5,
                )
                for line in line_proc.stdout.splitlines():
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        results.append({
                            "file": fpath,
                            "line": parts[0],
                            "text": parts[1][:200],
                        })
            except (subprocess.TimeoutExpired, OSError):
                pass
        return results[:6]  # cap total hits
    except (subprocess.TimeoutExpired, OSError):
        return []


def _vault_grep(identifier: str) -> list[dict]:
    """Grep vault for identifier. Returns [] if vault not present."""
    if not _VAULT_PATH.exists():
        return []
    try:
        proc = subprocess.run(
            [
                "grep", "-rn", "--include=*.md",
                "-l", "--", identifier, str(_VAULT_PATH),
            ],
            capture_output=True, text=True, timeout=_GREP_TIMEOUT,
        )
        matching_files = proc.stdout.splitlines()[:2]
        results = []
        for fpath in matching_files:
            try:
                lp = subprocess.run(
                    ["grep", "-n", "-m", "1", "--", identifier, fpath],
                    capture_output=True, text=True, timeout=5,
                )
                for line in lp.stdout.splitlines():
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        results.append({
                            "file": fpath,
                            "line": parts[0],
                            "text": parts[1][:200],
                        })
            except (subprocess.TimeoutExpired, OSError):
                pass
        return results[:4]
    except (subprocess.TimeoutExpired, OSError):
        return []


@dataclass
class _IdentifierSubstrate:
    identifier: str
    repo_hits: list[dict]
    vault_hits: list[dict]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

_LLM_URL = "http://203.0.113.12:8081/v1/chat/completions"
_LLM_TIMEOUT = 45  # seconds; Haiku-scale call, should be fast


class LapisPMReviewerAdapter:
    """SubstrateAdapter for Lapis-PM reviewer corroboration.

    retrieve: extracts doc-mentioned identifiers from diff; greps repo + vault.
    score: single constrained-JSON LLM call to check claim accuracy.

    Invariant: read-only. Never touches reviewer verdict mainline fields.
    """

    def __init__(self, repo_path: str | None = None):
        self._repo_path = repo_path

    def retrieve(
        self,
        diff_text: str,
        repo: str,
        repo_path: str | None = None,
    ) -> list[_IdentifierSubstrate]:
        """Extract identifiers from diff and grep repo + vault for each."""
        rpath = repo_path or self._repo_path or f"/srv/git/{repo}-working"
        identifiers = _extract_identifiers(diff_text)
        substrates = []
        for ident in identifiers[:_MAX_IDENTIFIERS]:
            substrates.append(_IdentifierSubstrate(
                identifier=ident,
                repo_hits=_grep_repo(ident, rpath),
                vault_hits=_vault_grep(ident),
            ))
        return substrates

    def score(
        self,
        diff_text: str,
        substrates: list[_IdentifierSubstrate],
        repo: str,
    ) -> CorroborationResult:
        """Single LLM call to check identifier claims against substrate.

        Returns uncertain if no identifiers or LLM unavailable.
        """
        scope_id = f"repo:{repo}"
        now = datetime.now(timezone.utc).isoformat()

        if not substrates:
            return CorroborationResult(
                verdict="uncertain",
                claim="(no identifiers extracted)",
                citations=[],
                freshness_stamp=now,
                scope_id=scope_id,
                drift_class=None,
                notes="No identifiers found in diff",
            )

        # Build substrate summary for the LLM
        substrate_lines: list[str] = []
        for s in substrates[:10]:
            substrate_lines.append(f"### `{s.identifier}`")
            if s.repo_hits:
                for h in s.repo_hits[:3]:
                    substrate_lines.append(
                        f"  repo:{h['file']}:{h['line']}: {h['text']}"
                    )
            else:
                substrate_lines.append("  (not found in repo)")
            if s.vault_hits:
                for h in s.vault_hits[:2]:
                    substrate_lines.append(
                        f"  vault:{h['file']}:{h['line']}: {h['text']}"
                    )
        substrate_text = "\n".join(substrate_lines)

        prompt = (
            f"You are checking doc-mentioned-identifier claims in a PR diff "
            f"for repo `{repo}`. Identifiers are names mentioned in prose, "
            f"commit messages, or spec text.\n\n"
            f"## Diff (excerpt)\n```diff\n{diff_text[:2500]}\n```\n\n"
            f"## Substrate (grep results)\n{substrate_text}\n\n"
            f"For each identifier listed in the substrate, determine whether "
            f"it exists in the repo as claimed. Classify each as:\n"
            f"- `missing_referent`: identifier mentioned in diff prose but not found in repo\n"
            f"- `renamed_referent`: identifier mentioned but found only under a different name\n"
            f"- `stale_referent`: identifier found but its state doesn't match the claim\n"
            f"- `none`: identifier exists as claimed (no drift)\n\n"
            f"Return ONLY valid JSON with this exact schema:\n"
            f'{{"verdict": "clean"|"flagged"|"uncertain", '
            f'"claims": [{{"identifier": str, "drift_class": "missing_referent"|"stale_referent"|"renamed_referent"|"none", "notes": str}}], '
            f'"summary": str}}\n'
            f"verdict=flagged if any claim has non-none drift_class. "
            f"verdict=clean if all none. verdict=uncertain if substrate insufficient.\n"
            f"No prose outside the JSON."
        )

        try:
            import httpx
            resp = httpx.post(
                _LLM_URL,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 512,
                },
                timeout=_LLM_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content.strip())
            # Some models wrap with <think>...</think>; strip thinking
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            result_data = json.loads(content)
        except Exception as exc:
            return CorroborationResult(
                verdict="uncertain",
                claim="(substrate unavailable)",
                citations=[],
                freshness_stamp=datetime.now(timezone.utc).isoformat(),
                scope_id=scope_id,
                drift_class=None,
                notes=f"LLM unavailable: {type(exc).__name__}",
            )

        verdict = result_data.get("verdict", "uncertain")
        if verdict not in ("clean", "flagged", "uncertain"):
            verdict = "uncertain"

        claims = result_data.get("claims", [])

        # Build citations from substrate grep results for flagged identifiers
        citations: list[Citation] = []
        for claim_item in claims:
            dc = claim_item.get("drift_class", "none")
            if dc == "none":
                continue
            ident = claim_item.get("identifier", "")
            for s in substrates:
                if s.identifier == ident:
                    for h in s.repo_hits[:1]:
                        citations.append(Citation(
                            source=f"repo:{repo}:{h['file']}:{h['line']}",
                            text=h["text"],
                        ))
                    break

        # Worst drift_class across all claims
        _drift_rank = {
            "missing_referent": 3,
            "renamed_referent": 2,
            "stale_referent": 1,
            "none": 0,
        }
        worst_dc: str | None = None
        for claim_item in claims:
            dc = claim_item.get("drift_class", "none")
            if _drift_rank.get(dc, 0) > _drift_rank.get(worst_dc or "none", 0):
                worst_dc = dc
        if worst_dc == "none":
            worst_dc = None

        return CorroborationResult(
            verdict=verdict,
            claim=result_data.get("summary", ""),
            citations=citations,
            freshness_stamp=datetime.now(timezone.utc).isoformat(),
            scope_id=scope_id,
            drift_class=worst_dc,
            notes=result_data.get("summary"),
        )


# ---------------------------------------------------------------------------
# Public entry point for pm_core integration
# ---------------------------------------------------------------------------

def run_corroboration_pass(
    diff_text: str,
    repo: str,
    repo_path: str | None = None,
) -> dict:
    """Run the corroboration follow-up pass on a PR diff.

    Returns a dict suitable for attaching as `corroboration_result` to
    the reviewer verdict JSON. Never raises — returns uncertain-shaped
    dict on any error (Compost invariant: all outputs are nutrients).
    """
    adapter = LapisPMReviewerAdapter(repo_path=repo_path)
    try:
        substrates = adapter.retrieve(diff_text, repo, repo_path)
        result = adapter.score(diff_text, substrates, repo)
        return result.to_dict()
    except Exception as exc:
        return CorroborationResult(
            verdict="uncertain",
            claim="(corroboration pass failed)",
            citations=[],
            freshness_stamp=datetime.now(timezone.utc).isoformat(),
            scope_id=f"repo:{repo}",
            drift_class=None,
            notes=f"Adapter error: {type(exc).__name__}: {exc}",
        ).to_dict()
