"""Tests for reviewer template content — lapis-pm-reviewer-full-context-v0.

Asserts that the reviewer and reviewer_fresh system_template blocks in
registry.yaml contain the required strings for the full-context fallback:
  - existing_branch  (template variable placeholder)
  - base_branch      (template variable placeholder)
  - git fetch origin (fallback command)
  - git diff origin/ (fallback command prefix)

The template format is the contract; this test pins it.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REGISTRY_PATH = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"

REQUIRED_STRINGS = [
    "existing_branch",
    "base_branch",
    "git fetch origin",
    "git diff origin/",
]


def _load_templates() -> dict[str, str]:
    data = yaml.safe_load(REGISTRY_PATH.read_text())
    agents = data.get("agents", {})
    return {
        "reviewer": agents["reviewer"]["system_template"],
        "reviewer_fresh": agents["reviewer_fresh"]["system_template"],
    }


class TestReviewerTemplateContent:

    def test_reviewer_template_has_existing_branch(self):
        tpl = _load_templates()["reviewer"]
        assert "existing_branch" in tpl, \
            "reviewer template missing 'existing_branch' placeholder"

    def test_reviewer_template_has_base_branch(self):
        tpl = _load_templates()["reviewer"]
        assert "base_branch" in tpl, \
            "reviewer template missing 'base_branch' placeholder"

    def test_reviewer_template_has_git_fetch_origin(self):
        tpl = _load_templates()["reviewer"]
        assert "git fetch origin" in tpl, \
            "reviewer template missing 'git fetch origin' fallback recipe"

    def test_reviewer_template_has_git_diff_origin(self):
        tpl = _load_templates()["reviewer"]
        assert "git diff origin/" in tpl, \
            "reviewer template missing 'git diff origin/' fallback command"

    def test_reviewer_fresh_template_has_existing_branch(self):
        tpl = _load_templates()["reviewer_fresh"]
        assert "existing_branch" in tpl, \
            "reviewer_fresh template missing 'existing_branch' placeholder"

    def test_reviewer_fresh_template_has_base_branch(self):
        tpl = _load_templates()["reviewer_fresh"]
        assert "base_branch" in tpl, \
            "reviewer_fresh template missing 'base_branch' placeholder"

    def test_reviewer_fresh_template_has_git_fetch_origin(self):
        tpl = _load_templates()["reviewer_fresh"]
        assert "git fetch origin" in tpl, \
            "reviewer_fresh template missing 'git fetch origin' fallback recipe"

    def test_reviewer_fresh_template_has_git_diff_origin(self):
        tpl = _load_templates()["reviewer_fresh"]
        assert "git diff origin/" in tpl, \
            "reviewer_fresh template missing 'git diff origin/' fallback command"

    def test_reviewer_template_all_required_strings(self):
        """Single sweep: all required strings present in reviewer template."""
        tpl = _load_templates()["reviewer"]
        missing = [s for s in REQUIRED_STRINGS if s not in tpl]
        assert not missing, f"reviewer template missing: {missing}"

    def test_reviewer_fresh_template_all_required_strings(self):
        """Single sweep: all required strings present in reviewer_fresh template."""
        tpl = _load_templates()["reviewer_fresh"]
        missing = [s for s in REQUIRED_STRINGS if s not in tpl]
        assert not missing, f"reviewer_fresh template missing: {missing}"

    def test_reviewer_template_retains_role_invariant(self):
        """Role invariant ('PURE JUDGE') stays in reviewer template."""
        tpl = _load_templates()["reviewer"]
        assert "PURE JUDGE" in tpl, \
            "reviewer template lost the 'PURE JUDGE' role invariant"

    def test_reviewer_fresh_template_retains_role_invariant(self):
        """Role invariant ('PURE JUDGE') stays in reviewer_fresh template."""
        tpl = _load_templates()["reviewer_fresh"]
        assert "PURE JUDGE" in tpl, \
            "reviewer_fresh template lost the 'PURE JUDGE' role invariant"
