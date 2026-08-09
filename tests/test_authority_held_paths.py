"""Tests for HELD_PATTERNS anchoring and diff-path capture (lapis-pm-containment-held-paths-v0).

Coverage:
  - AC1.2: positive matches — each of the seven newly-added patterns matches its
    real path.
  - AC1.3: negative/over-match tests — one per addition, proving the patterns are
    left-anchored and do not bleed into unrelated paths.
  - AC2: changed_paths() unions the '+++ b/' and '--- a/' sides so deletions and
    renames-away are captured, not just additions.
"""

from __future__ import annotations

from lapis_pm import authority


# ---------------------------------------------------------------------------
# Leg 1 — positive matches for the seven newly-anchored patterns
# ---------------------------------------------------------------------------

def test_held_positive_authority_py():
    assert authority.is_held_path("lapis_pm/authority.py")


def test_held_positive_auto_resolve_py():
    assert authority.is_held_path("lapis_pm/auto_resolve.py")


def test_held_positive_bundle_autodispatch_py():
    assert authority.is_held_path("lapis_pm/bundle_autodispatch.py")


def test_held_positive_hooks_dir():
    assert authority.is_held_path("lapis_pm/hooks/router-portfolio-stop.py")


def test_held_positive_deploy_dir():
    assert authority.is_held_path("deploy/deploy.sh")


def test_held_positive_claude_settings_json():
    assert authority.is_held_path(".claude/settings.json")


def test_held_positive_claude_settings_local_json():
    assert authority.is_held_path(".claude/settings.local.json")


def test_held_positive_pm_pr_review_flat():
    assert authority.is_held_path(".claude/skills/pm-pr-review.md")


def test_held_positive_pm_pr_review_skill_md():
    assert authority.is_held_path(".claude/skills/pm-pr-review/SKILL.md")


# ---------------------------------------------------------------------------
# Leg 1 — negative / over-match tests, one per addition
# ---------------------------------------------------------------------------

def test_held_negative_authority_py_not_anchored_elsewhere():
    # unanchored `authority\.py$` would match this — must not, now that it's
    # left-anchored to the repo root path `lapis_pm/authority.py`.
    assert not authority.is_held_path("foo/authority.py")


def test_held_negative_authority_py_substring_module():
    assert not authority.is_held_path("agents_core/my_authority.py")


def test_held_negative_auto_resolve_elsewhere():
    assert not authority.is_held_path("other_pkg/auto_resolve.py")


def test_held_negative_bundle_autodispatch_elsewhere():
    assert not authority.is_held_path("other_pkg/bundle_autodispatch.py")


def test_held_negative_hooks_outside_lapis_pm():
    assert not authority.is_held_path("scripts/hooks/y")


def test_held_negative_deploy_not_at_root():
    assert not authority.is_held_path("src/deploy/x")


def test_held_negative_claude_skills_other():
    assert not authority.is_held_path(".claude/skills/other.md")


def test_held_negative_claude_skills_pm_pr_review_sibling_file():
    # A sibling file inside the pm-pr-review dir that isn't SKILL.md must not match —
    # only the two canonical copies are held.
    assert not authority.is_held_path(".claude/skills/pm-pr-review/notes.md")


# ---------------------------------------------------------------------------
# Leg 2 — deletion / rename capture in changed_paths()
# ---------------------------------------------------------------------------

DELETION_DIFF = """diff --git a/infra/deploy.sh b/infra/deploy.sh
deleted file mode 100644
index abc1234..0000000
--- a/infra/deploy.sh
+++ /dev/null
@@ -1,3 +0,0 @@
-#!/bin/bash
-echo hello
-exit 0
"""

RENAME_DIFF = """diff --git a/lapis_pm/authority.py b/scripts/authority_moved.py
similarity index 96%
rename from lapis_pm/authority.py
rename to scripts/authority_moved.py
index abc1234..def5678 100644
--- a/lapis_pm/authority.py
+++ b/scripts/authority_moved.py
@@ -1,1 +1,1 @@
-old
+new
"""

ADDITION_DIFF = """diff --git a/foo/new_file.py b/foo/new_file.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/foo/new_file.py
@@ -0,0 +1,2 @@
+print("hi")
+
"""


def test_changed_paths_captures_pure_deletion():
    paths = authority.changed_paths(DELETION_DIFF)
    assert "infra/deploy.sh" in paths
    assert "/dev/null" not in paths


def test_changed_paths_deletion_of_held_path_classifies_hold():
    assert any(authority.is_held_path(p) for p in authority.changed_paths(DELETION_DIFF))


def test_changed_paths_captures_rename_away_from_held_path():
    # The old (held) path is caught via '--- a/', even though the new path is unheld.
    paths = authority.changed_paths(RENAME_DIFF)
    assert "lapis_pm/authority.py" in paths
    assert any(authority.is_held_path(p) for p in paths)


def test_changed_paths_ordinary_addition_no_dev_null_no_dupes():
    paths = authority.changed_paths(ADDITION_DIFF)
    assert paths == ["foo/new_file.py"]
    assert "/dev/null" not in paths
