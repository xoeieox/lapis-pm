#!/usr/bin/env python3
"""import_closure.py — compute transitive import closure of local modules.

Usage:
  python3 deploy/import_closure.py <copy_from_dir> <seed>...

Prints one relative path per line (relative to <copy_from_dir>) for every file
in the transitive closure. Exits non-zero with an error message on stderr if
any seed cannot be resolved to a file in <copy_from_dir>.

Seeds can be:
  - Bare module names (e.g., 'enlightenment_producer') → resolves to enlightenment_producer.py
  - Filenames (e.g., 'night_coordinator.py') → used as-is

AST walks descend into function/method bodies to catch lazy imports.
"""

import ast
import sys
from pathlib import Path


def resolve_seed(copy_from: Path, seed: str) -> Path:
    """Resolve a seed to a .py file in copy_from.

    Seed can be:
      - 'foo.py' → copy_from/foo.py
      - 'foo' → copy_from/foo.py

    Raises ValueError if the file does not exist.
    """
    if seed.endswith('.py'):
        candidate = copy_from / seed
    else:
        candidate = copy_from / f"{seed}.py"

    if not candidate.exists():
        raise ValueError(f"Cannot resolve seed '{seed}' to a file in {copy_from}")

    return candidate


def extract_imports(code_str: str) -> set:
    """Extract all imported names from a code string.

    Returns a set of module names imported (e.g., {'foo', 'bar', 'baz.qux'}).
    Walks the entire AST including function and class bodies.
    """
    imported_names = set()
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return imported_names

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module.split('.')[0])

    return imported_names


def compute_closure(copy_from: Path, seed_files: list) -> set:
    """Compute transitive closure of imports.

    Args:
        copy_from: Root directory containing .py modules
        seed_files: List of relative paths (e.g., ['night_coordinator.py', 'enlightenment_reader.py'])

    Returns:
        Set of relative paths (as strings) in the closure, including seeds.
    """
    closure = set()
    to_process = set(seed_files)
    processed = set()

    while to_process:
        rel_path = to_process.pop()
        if rel_path in processed:
            continue
        processed.add(rel_path)
        closure.add(rel_path)

        full_path = copy_from / rel_path
        if not full_path.exists():
            continue

        try:
            code = full_path.read_text(encoding='utf-8', errors='ignore')
        except (IOError, OSError):
            continue

        imported_names = extract_imports(code)

        for name in imported_names:
            candidate_py = f"{name}.py"
            candidate_path = copy_from / candidate_py
            if candidate_path.exists() and candidate_py not in processed:
                to_process.add(candidate_py)

    return closure


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 import_closure.py <copy_from_dir> <seed>...", file=sys.stderr)
        sys.exit(1)

    copy_from_str = sys.argv[1]
    seeds = sys.argv[2:] if len(sys.argv) > 2 else []

    copy_from = Path(copy_from_str)
    if not copy_from.is_dir():
        print(f"Error: {copy_from_str} is not a directory", file=sys.stderr)
        sys.exit(1)

    if not seeds:
        print("Error: at least one seed must be provided", file=sys.stderr)
        sys.exit(1)

    # Resolve seeds to file paths
    seed_files = []
    for seed in seeds:
        try:
            resolved = resolve_seed(copy_from, seed)
            seed_files.append(resolved.relative_to(copy_from).as_posix())
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    closure = compute_closure(copy_from, seed_files)

    # Print in sorted order for determinism
    for path in sorted(closure):
        print(path)


if __name__ == '__main__':
    main()
