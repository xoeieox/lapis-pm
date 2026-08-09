"""AC7 (lapis-pm-reference-leg-provider-neutral-naming-v0): enforce the FALSE/TRUE
classification boundary as a real assertion, not a one-time act of review attention.

Why AST instead of grep (Erah's H2 ruling, gate pass 1, 2026-07-31): introspecting
every dataclass field, function signature, and module variable across three files by
hand is itself a complex grep, and a plain-text grep would false-positive on comments
and docstrings that correctly still say "Sonnet" or "Opus" in prose (e.g. explaining
why a rename happened, or documenting a genuinely-paid call). Walking the module AST
and collecting only real Python identifiers — Name/Attribute/arg/function-and-class
names — ignores comments and string literals entirely, so a docstring mentioning
Sonnet does not fail this test; only a live sonnet_*/opus_*-spelled identifier does.

Two directions, both required (a test that only catches one is half a guard):
  1. FALSE direction: no sonnet_/opus_-derived identifier is reachable on the
     spec-review reference-leg path in spec_review.py / cli.py / pm_core.py, except
     the explicit AC4/AC5 back-compat shims named below.
  2. TRUE direction: the genuinely-paid Sonnet call sites (authority.screen's real
     model="sonnet" inline auto-merge screen, and cli.py's --facets-operator /
     --model choices, which are real model selectors a caller may pass) are still
     present — this unit must not have accidentally swept up real paid-model
     references in the rename.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SONNET_OPUS_RE = re.compile(r"(sonnet|opus)", re.IGNORECASE)

# Explicit back-compat shims permitted to carry a sonnet_/opus_-spelled identifier
# (lapis-pm-reference-leg-provider-neutral-naming-v0 AC4/AC5, sunset 90 days after
# merge — 2026-09-05 for the opus_* dataclass aliases; ~2026-10-30 for the CLI/env
# aliases and the divergence-record dual-emit fields dated in-record).
_ALLOWED_IDENTIFIERS = {
    # SpecReviewBrief deprecated read-aliases (OQ1 ruling: point at reference_*)
    "opus_verdict",
    "opus_issues",
    "opus_confidence",
    "opus_run_id",
    "opus_advisory_only",
    # _combined_recommendation / _build_brief deprecated parameter aliases
    "opus_raw",
    # run_spec_review's deprecated parameter alias for reference_reviewer
    "sonnet_reviewer",
    # --compare-opus: deprecated/no-op flag, kept for back-compat
    "compare_opus",
}


def _collect_identifiers(path: Path) -> set[str]:
    """Real Python identifiers only — no comments, no string literals (docstrings,
    log messages, dict/env-var string keys are deliberately excluded; see module
    docstring for why)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


_SCANNED_FILES = [
    _REPO_ROOT / "lapis_pm" / "spec_review.py",
    _REPO_ROOT / "lapis_pm" / "cli.py",
    _REPO_ROOT / "lapis_pm" / "pm_core.py",
]


def test_no_stray_sonnet_or_opus_identifiers_on_reference_leg_path():
    """FALSE direction: every sonnet_/opus_-spelled identifier in these three modules
    is one of the named AC4/AC5 back-compat shims — nothing else survived the rename."""
    for path in _SCANNED_FILES:
        assert path.is_file(), f"expected file missing: {path}"
        hits = {n for n in _collect_identifiers(path) if _SONNET_OPUS_RE.search(n)}
        stray = hits - _ALLOWED_IDENTIFIERS
        assert not stray, (
            f"{path.relative_to(_REPO_ROOT)} has un-allow-listed sonnet_/opus_ "
            f"identifiers: {sorted(stray)} — either rename them to reference_*, or "
            f"if they are a genuine new back-compat shim, add them to "
            f"_ALLOWED_IDENTIFIERS with a reason."
        )


def test_allowed_identifiers_are_all_actually_present():
    """Guards the allow-list itself from rotting: if a shim is deleted (e.g. at its
    90-day sunset), its entry must be removed from _ALLOWED_IDENTIFIERS too, or this
    test stops meaning anything for that name."""
    all_hits: set[str] = set()
    for path in _SCANNED_FILES:
        all_hits |= _collect_identifiers(path)
    missing = _ALLOWED_IDENTIFIERS - all_hits
    assert not missing, (
        f"allow-listed shim identifiers no longer appear in any scanned file: "
        f"{sorted(missing)} — remove them from _ALLOWED_IDENTIFIERS."
    )


def test_authority_inline_sonnet_screen_still_real():
    """TRUE direction: authority.py's real, paid model="sonnet" inline auto-merge
    screen must survive this unit untouched — the inversion risk (AC2) is a real
    paid call quietly starting to look local, and this is the one call this whole
    unit must not touch."""
    path = _REPO_ROOT / "lapis_pm" / "authority.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "model":
            value = node.value
            if isinstance(value, ast.Constant) and value.value == "sonnet":
                found = True
                break
    assert found, (
        "authority.py no longer contains a model=\"sonnet\" call — this is the real "
        "paid inline screen (AC2 of lapis-pm-reference-leg-provider-neutral-naming-v0); "
        "it must never be renamed away by a reference-leg naming pass."
    )


def test_cli_facets_operator_and_model_choices_still_offer_real_models():
    """TRUE direction: --facets-operator and --model style choices=[...] lists are
    genuine model selectors (a caller may deliberately spend money on sonnet/opus) —
    these string-literal choices must survive, distinct from the FALSE-table rename."""
    path = _REPO_ROOT / "lapis_pm" / "cli.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    real_model_choice_lists = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "choices":
            value = node.value
            if isinstance(value, ast.List):
                elts = [
                    e.value for e in value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                if "sonnet" in elts and "opus" in elts:
                    real_model_choice_lists += 1
    assert real_model_choice_lists >= 2, (
        "expected at least two choices=[...] lists in cli.py still offering real "
        "'sonnet'/'opus' model values (--facets-operator, --model) — found "
        f"{real_model_choice_lists}. If one was legitimately removed, update this "
        "count; if it was accidentally renamed away, that's the AC2 inversion this "
        "test exists to catch."
    )
