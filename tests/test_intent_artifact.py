"""Tests for intent_artifact module (spec: intent-layer-harvest-v0)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from lapis_pm import intent_artifact


CLEAN_CONTENT = """\
---
target_id: my-target
author: Erah
ratified: 2026-06-25
supersedes: none
---

**problem:** We have no per-target intent artifact.[^p1]

**purpose:** Give workers a drift-reference to validate against.[^p2]

**on-track:** Worker flags divergence from intent, not just spec violations.[^p3]

**not-this:** No auto-blocking; advisory only.[^p4]

[^p1]: Session 2026-06-25, turn 1: "We have no per-target intent artifact."
[^p2]: Session 2026-06-25, turn 2: "Give workers a drift-reference to validate against."
[^p3]: Session 2026-06-25, Erah agreed to: "Worker flags divergence from intent."
[^p4]: Session 2026-06-25, turn 3: "No auto-blocking; advisory only."
"""

VOID_CONTENT = """\
---
target_id: my-target
author: Erah
ratified: 2026-06-25
supersedes: none
---

**problem:** We have no per-target intent artifact.[^p1]

**purpose:** <!-- VOID: no verbatim quote found for this field — need Erah's statement of purpose -->

**on-track:** <!-- VOID: need explicit on-track statement from Erah -->

**not-this:** No auto-blocking; advisory only.[^p4]

[^p1]: Session 2026-06-25, turn 1: "We have no per-target intent artifact."
[^p4]: Session 2026-06-25, turn 3: "No auto-blocking; advisory only."
"""


# --- has_voids / void_field_count ---

def test_has_voids_clean():
    assert not intent_artifact.has_voids(CLEAN_CONTENT)


def test_has_voids_with_voids():
    assert intent_artifact.has_voids(VOID_CONTENT)


def test_void_field_count_clean():
    assert intent_artifact.void_field_count(CLEAN_CONTENT) == 0


def test_void_field_count_with_voids():
    assert intent_artifact.void_field_count(VOID_CONTENT) == 2


def test_void_re_case_insensitive():
    assert intent_artifact.has_voids("<!-- void: something -->")
    assert intent_artifact.has_voids("<!-- VOID: something -->")
    assert intent_artifact.has_voids("<!-- Void: something -->")


# --- load ---

def test_load_missing_returns_none():
    with patch.object(intent_artifact, "INTENT_DIR", Path("/nonexistent/path")):
        assert intent_artifact.load("any-target") is None


def test_load_returns_content(tmp_path):
    intent_dir = tmp_path / "intent"
    intent_dir.mkdir()
    (intent_dir / "my-target.md").write_text(CLEAN_CONTENT)
    with patch.object(intent_artifact, "INTENT_DIR", intent_dir):
        result = intent_artifact.load("my-target")
    assert result == CLEAN_CONTENT


# --- dispatch_block ---

def test_dispatch_block_no_artifact():
    with patch.object(intent_artifact, "INTENT_DIR", Path("/nonexistent")):
        assert intent_artifact.dispatch_block("any-target") == ""


def test_dispatch_block_clean_artifact(tmp_path):
    intent_dir = tmp_path / "intent"
    intent_dir.mkdir()
    (intent_dir / "my-target.md").write_text(CLEAN_CONTENT)
    with patch.object(intent_artifact, "INTENT_DIR", intent_dir):
        block = intent_artifact.dispatch_block("my-target")
    assert block.startswith("## Target intent\n")
    assert "VOID" not in block
    assert "ADVISORY" not in block
    assert "We have no per-target intent artifact" in block


def test_dispatch_block_with_voids_includes_advisory(tmp_path):
    intent_dir = tmp_path / "intent"
    intent_dir.mkdir()
    (intent_dir / "my-target.md").write_text(VOID_CONTENT)
    with patch.object(intent_artifact, "INTENT_DIR", intent_dir):
        block = intent_artifact.dispatch_block("my-target")
    assert "## Target intent" in block
    assert "ADVISORY" in block
    assert "2 VOID field(s)" in block
    assert "MUST NOT fill" in block
    assert "FLAG it back to Erah" in block


def test_dispatch_block_void_count_accurate(tmp_path):
    intent_dir = tmp_path / "intent"
    intent_dir.mkdir()
    content = "<!-- VOID: one --> <!-- VOID: two --> <!-- VOID: three -->"
    (intent_dir / "t.md").write_text(content)
    with patch.object(intent_artifact, "INTENT_DIR", intent_dir):
        block = intent_artifact.dispatch_block("t")
    assert "3 VOID field(s)" in block


# --- registry template render (integration: vars_ injection) ---

def test_registry_templates_accept_intent_block():
    """Smoke: all worker templates render without KeyError when intent_block is empty."""
    import yaml
    from pathlib import Path as P

    registry_path = P(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    raw = yaml.safe_load(registry_path.read_text())
    worker_agents = ["fixer", "fixer_retry", "reviewer", "reviewer_fresh", "fixer_local"]
    base_vars = {
        "repo": "lapis-pm",
        "target_id": "test-target",
        "spec_summary": "test spec",
        "intent_block": "",
        "pr_number": "1",
        "slug": "forced",
        "existing_branch": "lapis/test-target/forced",
        "base_branch": "main",
        "question": "test question",
        "prior_review": "",
        "repo_cwd": "/tmp",
        "invariant_context": "",
    }
    for name in worker_agents:
        agent = raw["agents"].get(name)
        if agent is None:
            continue
        tmpl = agent["system_template"]
        try:
            tmpl.format(**base_vars)
        except KeyError as e:
            pytest.fail(f"template '{name}' missing key: {e}")
