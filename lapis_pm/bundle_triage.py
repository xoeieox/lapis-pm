"""bundle_triage — per-item mechanical/fork classification for debt-bundle
items, run BEFORE the full spec-review gate (spec bundle-item-level-triage-v0).

Why: the nightly debt-bundle gate has bound zero items across 15 consecutive
runs — every bundle item, however mechanical, rides the same philosophical
council as a genuine design fork, and the council escalates everything. This
module splits each bundle item into:

  - "mechanical" — code-grounded, concrete fix, small blast radius, and an
    EXTRACTED (never formulated) Verification Contract. Cooked into its own
    fix spec and bound directly, advisory, no council.
  - "fork" — anything that fails a tier. Stays in the bundle and reaches the
    full gate unchanged, so genuine invariant/ethics/UX/vision-fork items
    still route to Erah exactly as before.

Three-tier check, binary, fail-closed to "fork" on any tier miss:
  1. Static check       — code-grounded fix, no denylisted path touched.
  2. Blast-radius check — small named-file count, no denylisted path.
  3. Behavioral-invariance check — a POST-COOK BIND-GATE VERIFIER, NOT a
     classification-time tier. `classify_item` / `is_mechanical` cover tiers
     1-2 only. Tier 3 is `verify_behavioral_invariance`, called by
     `bundle_autodispatch._reconcile_one` AFTER `cook_item_to_spec` and
     BEFORE the mechanical bind fires. A tier-3 miss re-routes the item to
     fork with reason "invariance" — never a silent bind.

The Verification Contract is EXTRACTION-ONLY: `extract_verification_contract`
locates existing proof (a named test that already pins the touched surface's
behavior) — it never generates a new assertion or interprets meaning to
synthesize one. No extractable contract means the item can never classify
mechanical, however simple the fix looks (DoD-1, fail-closed).

The tier-1/2 interface-and-invariant denylist SOURCES from the existing
`lapis_pm.authority.HELD_PATTERNS` rather than inventing a parallel list.

`cook_item_to_spec` is IN-REPO: it renders a spec from a plain dict via an
f-string template — no import of, or call into, conductor or any other repo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from lapis_pm.authority import is_held_path

# ---------------------------------------------------------------------------
# Tier thresholds / reject lists
# ---------------------------------------------------------------------------

# Tier 2: "a small named-file count" (v0 stand-in for the Council's fuller
# dependency-topology read — see spec Design §3).
MAX_BLAST_RADIUS_FILES = 3

# Case-insensitive exact-match placeholder tokens for fix_description — a
# non-responsive value is not a concrete fix, mirroring the C7 Consumer-line
# reject-token convention in spec_review.py.
_FIX_DESCRIPTION_REJECT_TOKENS: frozenset[str] = frozenset({
    "tbd", "n/a", "na", "none", "unknown", "todo", "",
})

_FORK_REASONS = ("static", "blast-radius", "invariance")


# ---------------------------------------------------------------------------
# Tier 1 + 2: the classification-time predicate
# ---------------------------------------------------------------------------

def _looks_placeholder(fix_description: str) -> bool:
    normalized = re.sub(r"\s+", " ", fix_description.strip().lower())
    return normalized in _FIX_DESCRIPTION_REJECT_TOKENS


def _static_check(fix_description: str, touched_surfaces: list[str]) -> bool:
    """Tier 1: the fix is code-grounded (a concrete, non-placeholder fix
    description naming concrete touched files) and touches no denylisted
    path."""
    if not fix_description or not fix_description.strip():
        return False
    if _looks_placeholder(fix_description):
        return False
    if not touched_surfaces:
        return False
    if any(is_held_path(p) for p in touched_surfaces):
        return False
    return True


def _blast_radius_check(touched_surfaces: list[str]) -> bool:
    """Tier 2: a small named-file count, no denylisted path. Denylist is
    re-checked defensively — this function must stand alone as a correct
    tier-2 predicate even if a future caller runs it without tier 1."""
    if not touched_surfaces:
        return False
    if len(touched_surfaces) > MAX_BLAST_RADIUS_FILES:
        return False
    if any(is_held_path(p) for p in touched_surfaces):
        return False
    return True


def is_mechanical(fix_description: str, touched_surfaces: list[str]) -> bool:
    """The mechanical predicate, tiers 1+2 only — separately callable so a
    second call site (seam 4, `adjudication-boundary-four-trigger-alignment-v0`
    DoD-3) can share this exact definition rather than forking it.

    Pure, deterministic, no I/O. Tier 3 (behavioral-invariance) is
    deliberately NOT part of this predicate — see module docstring and
    `verify_behavioral_invariance`.
    """
    return (
        _static_check(fix_description, touched_surfaces)
        and _blast_radius_check(touched_surfaces)
    )


def classify_item(item: dict) -> tuple[str, str | None]:
    """Thin wrapper over `is_mechanical` — tiers 1+2 only.

    Returns (class_, fork_reason):
      - ("mechanical", None) — both tiers pass.
      - ("fork", "static")       — tier 1 (code-grounded / denylist) failed.
      - ("fork", "blast-radius") — tier 1 passed, tier 2 (file count) failed.

    Never returns reason "invariance" — that categorized reason belongs only
    to the post-cook tier-3 verifier in bundle_autodispatch._reconcile_one,
    which re-routes an already-"mechanical" item to fork when no
    Verification Contract can be extracted.
    """
    fix_description = item.get("fix_description") or ""
    touched_surfaces = item.get("touched_surfaces") or []

    if not _static_check(fix_description, touched_surfaces):
        return "fork", "static"
    if not _blast_radius_check(touched_surfaces):
        return "fork", "blast-radius"
    return "mechanical", None


# ---------------------------------------------------------------------------
# Tier 3: the post-cook bind-gate verifier (Verification Contract)
# ---------------------------------------------------------------------------

def _repo_root(repo: str) -> Path:
    """Convention path for a repo's local working clone (CLAUDE.md 'Repos:'
    section: /srv/git/<repo>-working). A thin, monkeypatchable indirection —
    tests override this rather than touching the real filesystem."""
    return Path(f"/srv/git/{repo}-working")


def _locate_existing_test(root: Path, surface: str) -> str | None:
    """Static predicate over the tree: does a conventionally-named existing
    test already cover *surface*? Checked locations, in order:
      - <root>/tests/test_<stem>.py           (repo-root tests/ convention)
      - <surface's dir>/test_<stem>.py         (co-located test)
      - <surface's dir>/tests/test_<stem>.py   (package-local tests/)
    Returns the path (relative to root when possible) of the first hit, or
    None. Never writes, never runs anything — presence on disk only."""
    stem = Path(surface).stem
    surface_dir = Path(surface).parent
    candidates = [
        root / "tests" / f"test_{stem}.py",
        root / surface_dir / f"test_{stem}.py",
        root / surface_dir / "tests" / f"test_{stem}.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.relative_to(root))
            except ValueError:
                return str(candidate)
    return None


def extract_verification_contract(item: dict) -> str | None:
    """EXTRACTION-ONLY Verification Contract: locate existing proof for one
    of the item's touched surfaces — never formulate or generate a new
    assertion. Returns a deterministic, static verification statement (e.g.
    "existing test `tests/test_x.py` pins behavior for `x.py` and passes
    unchanged"), or None when no touched surface has a locatable test —
    fail-closed (DoD-1): no extractable contract, no matter how simple the
    fix looks.
    """
    repo = item.get("repo")
    touched_surfaces = item.get("touched_surfaces") or []
    if not repo or not touched_surfaces:
        return None
    root = _repo_root(repo)
    for surface in touched_surfaces:
        located = _locate_existing_test(root, surface)
        if located is not None:
            return (
                f"existing test `{located}` pins behavior for `{surface}` "
                f"and passes unchanged"
            )
    return None


def verify_behavioral_invariance(item: dict) -> tuple[bool, str | None]:
    """Tier 3 — the POST-COOK BIND-GATE VERIFIER (not a classification-time
    tier; see module docstring). Called by
    `bundle_autodispatch._reconcile_one` AFTER `cook_item_to_spec` renders
    the fix spec and BEFORE the mechanical bind fires. Re-confirms the
    item's Verification Contract still holds (extraction-only, static).

    Returns (holds, contract). holds=False means: re-route the item to fork
    with reason "invariance" — never a silent bind.
    """
    contract = extract_verification_contract(item)
    return contract is not None, contract


# ---------------------------------------------------------------------------
# The in-repo cook engine
# ---------------------------------------------------------------------------

def mechanical_item_target_id(repo: str, debt_id: str) -> str:
    """Deterministic kebab-case target id for a cooked mechanical-item spec
    — matches the **Target ID:**/**Repo:** anchored-header regex
    (spec_review.py's _TID_RE / _REPO_RE)."""
    slug = re.sub(r"[^a-z0-9]+", "-", debt_id.lower()).strip("-")
    return f"cr-bundle-item-{repo}-{slug}"


def cook_item_to_spec(item: dict, out_dir: Path | None = None) -> Path:
    """IN-REPO cook engine (bundle-item-level-triage-v0, 2026-08-17 gate
    amendment): renders a minimal, gate-admissible fix spec for one
    mechanical bundle item — four C7 header lines in the first 50 lines
    (**Target ID:**, **Repo:**, **Authority:**, **Consumer:**), the item's
    concrete fix as the sole deliverable, and the Verification Contract
    copied VERBATIM into the DoD.

    Rendering, not analysis: this function never extracts or re-derives the
    contract itself — it renders whatever `item["verification_contract"]`
    already holds (set by the caller from `extract_verification_contract` /
    `verify_behavioral_invariance` before cooking). No cross-repo import, no
    cross-repo call — conductor's `_run_cook_pass` is prior art for the
    pattern only, never a dependency.
    """
    from agents_core.room_paths import room_path

    repo = item["repo"]
    debt_id = item["debt_id"]
    target_id = mechanical_item_target_id(repo, debt_id)
    spec_dir = out_dir if out_dir is not None else room_path("planning.specs")
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{target_id}.md"

    surfaces = item.get("touched_surfaces") or []
    files_line = ", ".join(f"`{s}`" for s in surfaces) if surfaces else "(none named)"
    fix_description = item.get("fix_description") or "(no fix description captured)"
    contract = item.get("verification_contract") or "(no Verification Contract recorded)"
    source_pr = item.get("source_pr")

    text = (
        "---\n"
        f"spec_id: {target_id}\n"
        "status: draft\n"
        f"source: bundle-item-level-triage-v0 mechanical cook (debt_id={debt_id})\n"
        "---\n"
        "\n"
        f"# Spec: {target_id} — mechanical debt-item fix (cooked, no council)\n"
        "\n"
        f"**Target ID:** `{target_id}`\n"
        f"**Repo:** `{repo}`\n"
        "**Authority:** advisory\n"
        "**Consumer:** the night code-review debt-bundle pipeline "
        "(`lapis_pm/bundle_autodispatch.py` `reconcile()`) — the reader of "
        "each bundle item's triage class and the auto-bound fix it produces\n"
        "\n"
        "## Origin\n"
        "\n"
        f"Cooked by `lapis_pm.bundle_triage.cook_item_to_spec` "
        f"(bundle-item-level-triage-v0) from debt item `{debt_id}`"
        f"{f' (source PR #{source_pr})' if source_pr else ''} in the nightly "
        f"debt-bundle sweep for `{repo}`. Classified `mechanical` by the "
        "three-tier triage check (static + blast-radius both pass; "
        "behavioral-invariance verified) — bound directly, advisory, no "
        "philosophical council.\n"
        "\n"
        "## Deliverable\n"
        "\n"
        f"{fix_description}\n"
        "\n"
        f"**Touched surfaces:** {files_line}\n"
        "\n"
        "## Definition of Done\n"
        "\n"
        "- The fix above is applied.\n"
        "- **Verification Contract** (extraction-only, copied verbatim from "
        f"the triage record): {contract}\n"
    )
    spec_path.write_text(text, encoding="utf-8")
    return spec_path


# ---------------------------------------------------------------------------
# Per-item routing record (deliverable (c) — the sidecar the conductor-side
# day-surface renderer can read without importing lapis-pm)
# ---------------------------------------------------------------------------

def write_route_record(spec_path: Path, repo: str, triage_records: list[dict]) -> Path:
    """Write the per-item machine-readable route record adjacent to the
    bundle spec under /srv/lapis/planning/specs/ — deliverable (c). This is NOT
    the day-surface rendering (conductor-side, out of scope for this unit;
    see spec Deliverables)."""
    record_path = Path(str(spec_path) + ".route.json")
    payload = {"spec": spec_path.name, "repo": repo, "items": triage_records}
    record_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return record_path
