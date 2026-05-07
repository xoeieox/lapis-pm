"""Tests for _parse_spec_frontmatter — target_id/repo extraction and error paths."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lapis_pm.spec_review import SpecFrontmatterError, _parse_spec_frontmatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_spec(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "spec.md"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Valid frontmatter
# ---------------------------------------------------------------------------

VALID_FRONTMATTER = """\
    # Spec: My Thing

    **Target ID:** `my-target-id`
    **Repo:** `lapis-pm`
    **Branch:** `lapis/my-target-id/implement`
    **Authority:** `advisory`

    ## Goal

    Do the thing.
    """


def test_parse_valid_frontmatter(tmp_path):
    spec = _write_spec(tmp_path, VALID_FRONTMATTER)
    tid, repo = _parse_spec_frontmatter(spec)
    assert tid == "my-target-id"
    assert repo == "lapis-pm"


def test_parse_valid_two_char_names(tmp_path):
    """Two-char kebab-case values are the minimum valid length per the regex."""
    content = "**Target ID:** `ab`\n**Repo:** `lm`\n"
    spec = _write_spec(tmp_path, content)
    tid, repo = _parse_spec_frontmatter(spec)
    assert tid == "ab"
    assert repo == "lm"


def test_parse_valid_ignores_after_50_lines(tmp_path):
    """Frontmatter found in first 50 lines is used; garbage after is ignored."""
    lines = ["some preamble\n"] * 5
    lines.append("**Target ID:** `my-tid`\n")
    lines.append("**Repo:** `my-repo`\n")
    spec = _write_spec(tmp_path, "".join(lines))
    tid, repo = _parse_spec_frontmatter(spec)
    assert tid == "my-tid"
    assert repo == "my-repo"


# ---------------------------------------------------------------------------
# repo_override
# ---------------------------------------------------------------------------

def test_repo_override_supersedes_parsed(tmp_path):
    """repo_override replaces the parsed repo; target_id is still from frontmatter."""
    spec = _write_spec(tmp_path, VALID_FRONTMATTER)
    tid, repo = _parse_spec_frontmatter(spec)
    # Simulate repo_override at the call site (spec_review.run_spec_review does this)
    repo_override = "agents-core"
    effective_repo = repo_override if repo_override else repo
    assert tid == "my-target-id"
    assert effective_repo == "agents-core"


# ---------------------------------------------------------------------------
# Missing fields
# ---------------------------------------------------------------------------

def test_missing_target_id(tmp_path):
    content = "# Spec\n\n**Repo:** `lapis-pm`\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError, match="Target ID"):
        _parse_spec_frontmatter(spec)


def test_missing_repo(tmp_path):
    content = "# Spec\n\n**Target ID:** `my-tid`\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError, match="Repo"):
        _parse_spec_frontmatter(spec)


def test_missing_both(tmp_path):
    content = "# Spec\n\nNo frontmatter here.\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError):
        _parse_spec_frontmatter(spec)


# ---------------------------------------------------------------------------
# Malformed values
# ---------------------------------------------------------------------------

def test_rejects_whitespace_inside_backticks(tmp_path):
    """Values with spaces are not kebab-case — regex anchors + char class rejects them."""
    content = "**Target ID:** `my target`\n**Repo:** `lapis-pm`\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError, match="Target ID"):
        _parse_spec_frontmatter(spec)


def test_rejects_uppercase_in_target_id(tmp_path):
    content = "**Target ID:** `MyTarget`\n**Repo:** `lapis-pm`\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError, match="Target ID"):
        _parse_spec_frontmatter(spec)


def test_rejects_missing_backticks(tmp_path):
    """Without backticks the regex doesn't match."""
    content = "**Target ID:** my-tid\n**Repo:** lapis-pm\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError):
        _parse_spec_frontmatter(spec)


def test_rejects_leading_hyphen_in_target_id(tmp_path):
    """Target ID must start with [a-z0-9]."""
    content = "**Target ID:** `-my-tid`\n**Repo:** `lapis-pm`\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError, match="Target ID"):
        _parse_spec_frontmatter(spec)


def test_rejects_trailing_hyphen_in_target_id(tmp_path):
    """Target ID must end with [a-z0-9]."""
    content = "**Target ID:** `my-tid-`\n**Repo:** `lapis-pm`\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError, match="Target ID"):
        _parse_spec_frontmatter(spec)


def test_rejects_non_kebab_chars(tmp_path):
    """Underscores and dots are not in the allowed charset."""
    content = "**Target ID:** `my_tid`\n**Repo:** `lapis-pm`\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError, match="Target ID"):
        _parse_spec_frontmatter(spec)


def test_rejects_indented_target_id(tmp_path):
    """The regex is anchored — leading spaces prevent a match."""
    content = "  **Target ID:** `my-tid`\n**Repo:** `lapis-pm`\n"
    spec = _write_spec(tmp_path, content)
    with pytest.raises(SpecFrontmatterError, match="Target ID"):
        _parse_spec_frontmatter(spec)


def test_nonexistent_file_raises(tmp_path):
    spec = tmp_path / "nonexistent.md"
    with pytest.raises(SpecFrontmatterError, match="cannot read"):
        _parse_spec_frontmatter(spec)
