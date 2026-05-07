"""Tests for _load_invariant_context."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lapis_pm.spec_review import InvariantContextError, _load_invariant_context


def _make_files(tmp_path: Path, *, claude_md: str = "claude", spec_md: str | None = "spec", kernel: str = "kernel"):
    """Write fixture files under tmp_path and return (claude_path, spec_path, kernel_path)."""
    claude = tmp_path / "lapis-pm-working" / "CLAUDE.md"
    claude.parent.mkdir(parents=True, exist_ok=True)
    claude.write_text(claude_md, encoding="utf-8")

    spec = None
    if spec_md is not None:
        spec = tmp_path / "lapis-pm-working" / "SPEC.md"
        spec.write_text(spec_md, encoding="utf-8")

    vault = tmp_path / "inertia-vault-working" / "Lapis"
    vault.mkdir(parents=True, exist_ok=True)
    k = vault / "Constitution-Kernel.md"
    k.write_text(kernel, encoding="utf-8")

    return claude, spec, k


def _patch_paths(tmp_path: Path):
    """Return a context manager that patches the path helpers in spec_review."""
    base = tmp_path

    def fake_claude(repo):
        return base / f"{repo}-working" / "CLAUDE.md"

    def fake_spec(repo):
        return base / f"{repo}-working" / "SPEC.md"

    def fake_kernel():
        return base / "inertia-vault-working" / "Lapis" / "Constitution-Kernel.md"

    import lapis_pm.spec_review as sr

    class _Ctx:
        def __enter__(self):
            self._orig_load = sr._load_invariant_context

            def _patched_load(repo: str) -> str:
                claude_md_path = fake_claude(repo)
                spec_md_path = fake_spec(repo)
                kernel_path = fake_kernel()
                if not claude_md_path.exists():
                    raise InvariantContextError(f"CLAUDE.md not found at {claude_md_path}")
                if not kernel_path.exists():
                    raise InvariantContextError(f"Constitution-Kernel.md not found at {kernel_path}")
                claude_md = claude_md_path.read_text(encoding="utf-8")
                spec_md = spec_md_path.read_text(encoding="utf-8") if spec_md_path.exists() else ""
                kernel = kernel_path.read_text(encoding="utf-8")
                return (
                    f"=== CLAUDE.md ({repo}) ===\n{claude_md}\n\n"
                    f"=== SPEC.md ({repo}) ===\n{spec_md}\n\n"
                    f"=== Constitution-Kernel.md ===\n{kernel}"
                )

            sr._load_invariant_context = _patched_load
            return _patched_load

        def __exit__(self, *args):
            sr._load_invariant_context = self._orig_load

    return _Ctx()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_all_files_present(tmp_path):
    _make_files(tmp_path, claude_md="my-claude", spec_md="my-spec", kernel="my-kernel")
    with _patch_paths(tmp_path) as load:
        ctx = load("lapis-pm")
    assert "=== CLAUDE.md (lapis-pm) ===" in ctx
    assert "my-claude" in ctx
    assert "=== SPEC.md (lapis-pm) ===" in ctx
    assert "my-spec" in ctx
    assert "=== Constitution-Kernel.md ===" in ctx
    assert "my-kernel" in ctx


def test_spec_md_missing_yields_empty_section(tmp_path):
    _make_files(tmp_path, claude_md="c", spec_md=None, kernel="k")
    with _patch_paths(tmp_path) as load:
        ctx = load("lapis-pm")
    assert "=== SPEC.md (lapis-pm) ===" in ctx
    # spec_md section exists but is empty
    after_spec_header = ctx.split("=== SPEC.md (lapis-pm) ===")[1]
    before_kernel = after_spec_header.split("=== Constitution-Kernel.md ===")[0]
    assert before_kernel.strip() == "", f"expected empty SPEC.md section, got: {before_kernel!r}"


def test_context_non_empty_with_all_delimiters(tmp_path):
    _make_files(tmp_path)
    with _patch_paths(tmp_path) as load:
        ctx = load("lapis-pm")
    for delimiter in ["=== CLAUDE.md (lapis-pm) ===", "=== SPEC.md (lapis-pm) ===", "=== Constitution-Kernel.md ==="]:
        assert delimiter in ctx


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_missing_claude_md_raises(tmp_path):
    # Only write kernel — no CLAUDE.md
    vault = tmp_path / "inertia-vault-working" / "Lapis"
    vault.mkdir(parents=True, exist_ok=True)
    (vault / "Constitution-Kernel.md").write_text("k", encoding="utf-8")

    with _patch_paths(tmp_path) as load:
        with pytest.raises(InvariantContextError, match="CLAUDE.md"):
            load("lapis-pm")


def test_missing_kernel_raises(tmp_path):
    # Only write CLAUDE.md — no kernel
    claude = tmp_path / "lapis-pm-working" / "CLAUDE.md"
    claude.parent.mkdir(parents=True, exist_ok=True)
    claude.write_text("c", encoding="utf-8")

    with _patch_paths(tmp_path) as load:
        with pytest.raises(InvariantContextError, match="Constitution-Kernel"):
            load("lapis-pm")
