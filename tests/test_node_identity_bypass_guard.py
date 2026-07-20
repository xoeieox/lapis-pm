"""CI bypass-guard (I7, AC10): no raw `MemoryStore(` construction anywhere in
lapis_pm/ outside the sanctioned factory module.

This is a build-time perimeter, not an absolute one. Erah's ruling
(2026-07-19, lapis-pm-node-write-ownership-v0 v2): a static AST/grep sweep is
the enforcement mechanism; a runtime import-hook was considered and declined
as disproportionate/fragile. Runtime monkey-patching, dynamic imports of a
rogue module, and a compromised agents-core are explicitly OUT of this
target's threat model — this test cannot and does not claim to close that
gap, only the build-time one.

Two files are allowlisted besides node_identity.py itself:
  - lapis_pm/node_identity.py   — the factory module itself (writable_store /
    ReadOnlyPeerStore construct MemoryStore directly by design).
  - lapis_pm/router_portfolio.py — already ownership-aware before this target
    (routes through MemClient off the mem master); spec §3a explicitly says
    leave it as-is rather than rewire it through writable_store().
"""
from __future__ import annotations

import ast
from pathlib import Path

LAPIS_PM_ROOT = Path(__file__).resolve().parent.parent / "lapis_pm"

_ALLOWLISTED_RELATIVE_PATHS = {
    "node_identity.py",
    "router_portfolio.py",
}


def _find_memorystore_constructions(path: Path) -> list[int]:
    """Return line numbers where `MemoryStore(...)` is called in this file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "MemoryStore":
                hits.append(node.lineno)
    return hits


def test_no_raw_memorystore_construction_outside_factory():
    offenders: dict[str, list[int]] = {}
    for path in LAPIS_PM_ROOT.rglob("*.py"):
        rel = path.relative_to(LAPIS_PM_ROOT)
        if str(rel) in _ALLOWLISTED_RELATIVE_PATHS:
            continue
        hits = _find_memorystore_constructions(path)
        if hits:
            offenders[str(rel)] = hits

    assert not offenders, (
        "raw MemoryStore(...) construction found outside lapis_pm/node_identity.py "
        f"(the only sanctioned factory): {offenders}. Use node_identity.writable_store() "
        "or node_identity.peer_store() instead."
    )
