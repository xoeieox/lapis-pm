"""Isolation-boundary tests for lapis-pm-hold-shadow-observer-v0 (spec Scope
item 6 + Definition of Done #2/#3).

Two directions:
  1. Import/AST boundary — the shadow modules (hold_shadow.py,
     hold_shadow_rules.py, hold_shadow_summary.py) import none of the act
     surfaces named in Scope 6 (tick/tick_all, force_dispatch,
     merge_and_deploy, any _act_*, apply_decision, bind,
     reconcile_decided_gems, steer.* mutators, episodic.write_*).
     AST-based (Erah's H2 ruling precedent, lapis-pm-reference-leg-provider-
     neutral-naming-v0 AC7 / tests/test_reference_leg_naming_boundary.py):
     walking real Import/ImportFrom nodes ignores comments and docstrings
     entirely, so prose mentioning "episodic" in a module docstring does not
     false-positive this test; only a live import does.
  2. Negative leakage — /srv/lapis/hold-shadow/ is invisible to every existing
     consumer glob (directives, steers, comments, state-brief buckets, the
     room_paths registry); no record shape collides with the executable
     {brief_id, option_id} directive shape; the rules-config path is present
     in HELD_PATTERNS.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

from lapis_pm import authority
from lapis_pm import hold_shadow

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SHADOW_MODULE_PATHS = [
    _REPO_ROOT / "lapis_pm" / "hold_shadow.py",
    _REPO_ROOT / "lapis_pm" / "hold_shadow_rules.py",
    _REPO_ROOT / "lapis_pm" / "hold_shadow_summary.py",
]

# Modules that host an act surface named in Scope 6. Forbidding the module
# itself is stricter than forbidding only the specific symbol — it also
# closes off any future addition to that module reaching the shadow code
# unnoticed.
_FORBIDDEN_MODULE_NAMES = {
    "lapis_pm.pm_core", "pm_core",             # tick, tick_all, force_dispatch,
                                                # merge_and_deploy, any _act_*
    "lapis_pm.cli", "cli",                     # bind (cmd_bind)
    "lapis_pm.brief", "brief",                 # apply_decision
    "lapis_pm.brief_gem", "brief_gem",         # reconcile_decided_gems
    "lapis_pm.steer", "steer",                 # steer.* mutators
    "lapis_pm.episodic", "episodic",           # write_*
    "lapis_pm.spec_review", "spec_review",     # run_spec_review / gate lock
}


def _imported_names(path: Path) -> set[str]:
    """Real Import/ImportFrom targets only — walking the AST means comments
    and string literals (docstrings, log messages) can never trip this."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            if node.level and node.level > 0:
                # from . import X / from .. import X — X itself may be the
                # forbidden module name (e.g. "from . import episodic").
                for alias in node.names:
                    names.add(alias.name)
    return names


def test_shadow_modules_import_no_act_surface():
    for path in _SHADOW_MODULE_PATHS:
        hit = _imported_names(path) & _FORBIDDEN_MODULE_NAMES
        assert not hit, f"{path.name} imports forbidden act-surface module(s): {hit}"


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _non_docstring_code_strings(path: Path) -> list[str]:
    """Every string literal in the module EXCEPT module/function/class
    docstrings. Comments are never in the AST at all, so they're excluded
    for free — this is the same "real code, not prose" idiom as
    _imported_names above (Erah's H2 ruling), applied to string literals
    instead of identifiers."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skip = _docstring_node_ids(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
    ]


def test_shadow_modules_reference_no_forbidden_room_path_fragment():
    """Belt-and-suspenders check: outside of explanatory prose (docstrings),
    the shadow modules never even construct the paths owned by other
    consumers (directives, steers, comments, the brief-gem map key), so a
    future edit can't quietly widen scope without also tripping this test.
    AST-based, not a raw-text grep — see _non_docstring_code_strings — so a
    docstring explaining what this module deliberately does NOT touch
    (necessarily naming the forbidden path to explain the guarantee) can't
    false-positive this the way a naive grep would."""
    forbidden_fragments = [
        "room/directives", "directives.brief_decisions",
        "room/steers", "steers",
        "targets/comments", "targets.comments",
        "pm/brief-gem/map", "_forward_key",
        "pm/outstanding-brief",
    ]
    for path in _SHADOW_MODULE_PATHS:
        strings = _non_docstring_code_strings(path)
        for frag in forbidden_fragments:
            hits = [s for s in strings if frag in s]
            assert not hits, f"{path.name} constructs forbidden path fragment {frag!r} in: {hits}"


def test_shadow_modules_never_call_run_spec_review_or_gate_lock():
    forbidden = ["run_spec_review(", "_spec_review_lock", "spec-review-lock", "spec_review.lock"]
    for path in _SHADOW_MODULE_PATHS:
        text = path.read_text(encoding="utf-8")
        for frag in forbidden:
            assert frag not in text, f"{path.name} references the spec-review gate/lock: {frag!r}"


# ---------------------------------------------------------------------------
# Negative leakage (DoD #3)
# ---------------------------------------------------------------------------

def test_room_paths_registry_does_not_know_hold_shadow():
    """/srv/lapis/hold-shadow/ must stay unregistered — the whole point is that
    no existing consumer (which all read via room_paths or a hardcoded
    literal keyed off a registered subpath) can ever glob into it."""
    from agents_core import room_paths

    for key, entry in room_paths._KEYS.items():
        assert "hold-shadow" not in key
        assert "hold_shadow" not in key
        subpath = entry.get("subpath", "")
        assert "hold-shadow" not in subpath
        assert "hold_shadow" not in subpath


def test_hold_shadow_dir_disjoint_from_known_consumer_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    hs_dir = hold_shadow.hold_shadow_dir().resolve()

    consumer_roots = [
        tmp_path / "directives",
        tmp_path / "steers",
        tmp_path / "targets" / "comments",
        tmp_path / "briefs",
        tmp_path / "lapis-state",
    ]
    for root in consumer_roots:
        root_r = root.resolve()
        assert hs_dir != root_r
        assert root_r not in hs_dir.parents
        assert hs_dir not in root_r.parents


def test_hold_fact_record_never_carries_executable_directive_shape(monkeypatch, tmp_path):
    """The one JSON shape that gets machine-executed by the next tick is
    {brief_id, option_id} under /srv/lapis/directives/brief-decisions/ (brief.py's
    apply_decision path). A hold-fact record must never carry both keys."""
    import json

    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    hold_shadow.observe_hold_fact(
        target_id="tid", pr_number=1, repo="lapis-pm",
        hold_reasons=["held path(s) touched: infra/x"],
        hold_comment_id="h1", brief_comment_id="b1",
        pm_authority="hold", spec_bound_ts="ts",
    )
    record = json.loads(hold_shadow.hold_facts_path().read_text(encoding="utf-8").splitlines()[0])
    assert not ({"brief_id", "option_id"} <= record.keys())
    hold_shadow.validate_hold_fact_record(record)  # also asserts this structurally


def test_hold_shadow_rules_path_is_held():
    assert authority.is_held_path("lapis_pm/hold_shadow_rules.py")


def test_hold_shadow_wiring_modules_are_not_held():
    """Sanity: only the rules table is held (Scope item 5's precedent is
    authority.py holding only itself, not every file that reads it). The
    hook-wiring module and the summary pass are plain PR-review surfaces."""
    assert not authority.is_held_path("lapis_pm/hold_shadow.py")
    assert not authority.is_held_path("lapis_pm/hold_shadow_summary.py")
