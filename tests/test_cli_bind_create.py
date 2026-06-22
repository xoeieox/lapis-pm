"""Tests for `lapis-pm bind --create` (lapis-pm-bind-ergonomics spec).

Coverage:
  - bind --create on fresh target_id → creates YAML + writes spec + updates pm_bound
  - bind --create when target exists → exit 2 with --force hint
  - bind --create --force when target exists → replaces spec, updates pm_bound, preserves fields
  - bind without --create still errors on missing target (unchanged behaviour)
  - missing --title with --create → exit 2
  - tag derivation: --repo + --tag flags, deduped
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents_core.targets import TargetStore
from lapis_pm.cli import PM_LIFECYCLE_STAGES, _derive_tags, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPEC_BODY = textwrap.dedent("""\
    # Test spec

    A minimal spec body for unit testing.
""")


def _write_spec_file(tmp_path: Path, name: str = "spec.md") -> Path:
    p = tmp_path / name
    p.write_text(SPEC_BODY)
    return p


def _run(argv: list[str], targets_dir: Path) -> tuple[int, str, str]:
    """Run main() with a patched TargetStore and episodic/pm_core stubs.

    Returns (exit_code, stdout_written_via_print, stderr_written_via_print).
    We patch the module-level imports inside lapis_pm.cli.
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr

    out_buf = io.StringIO()
    err_buf = io.StringIO()

    with (
        patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
        patch("lapis_pm.cli.episodic.spec", return_value=None),
        patch("lapis_pm.cli.episodic.write_spec", return_value=MagicMock()),
        patch("lapis_pm.cli.pm_core.clear_classified_prs", return_value=None),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        rc = main(argv)

    return rc, out_buf.getvalue(), err_buf.getvalue()


def _run_with_existing_spec(argv: list[str], targets_dir: Path) -> tuple[int, str, str]:
    """Same as _run but episodic.spec returns a non-empty body (target has spec)."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    out_buf = io.StringIO()
    err_buf = io.StringIO()

    with (
        patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
        patch("lapis_pm.cli.episodic.spec", return_value="existing spec"),
        patch("lapis_pm.cli.episodic.write_spec", return_value=MagicMock()),
        patch("lapis_pm.cli.pm_core.clear_classified_prs", return_value=None),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        rc = main(argv)

    return rc, out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# Tests: bind --create happy path
# ---------------------------------------------------------------------------


def test_bind_create_fresh_target(tmp_path):
    """bind --create on a fresh target_id: creates YAML, writes spec, updates pm_bound."""
    spec_path = _write_spec_file(tmp_path)

    rc, out, err = _run(
        [
            "bind", "my-fresh-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-engine",
            "--authority", "advisory",
            "--create",
            "--title", "My Fresh Target",
            "--description", "A test target.",
        ],
        targets_dir=tmp_path,
    )

    assert rc == 0, f"expected exit 0; stderr={err!r}"

    # YAML created
    yaml_path = tmp_path / "my-fresh-target.yaml"
    assert yaml_path.exists(), "target YAML was not created"

    data = yaml.safe_load(yaml_path.read_text())
    assert data["id"] == "my-fresh-target"
    assert data["title"] == "My Fresh Target"
    assert data["description"] == "A test target."
    assert data["category"] == "active-work"
    assert data["urgency"] == "medium"
    assert data["work_mode"] == "anywhere"
    assert data["pm_bound"] is True
    assert data["pm_repo"] == "lapis-engine"
    assert data["pm_authority"] == "advisory"
    assert len(data["stages"]) == 4

    # repo tag derived
    assert "lapis-engine" in data.get("tags", [])

    assert "Bound my-fresh-target" in out


def test_bind_create_with_tags_and_product(tmp_path):
    """Tags from --repo + --tag are deduped; --product written to YAML."""
    spec_path = _write_spec_file(tmp_path)

    rc, out, err = _run(
        [
            "bind", "lapis-engine-primitive-card-component",
            "--spec-from", str(spec_path),
            "--repo", "lapis-engine",
            "--authority", "advisory",
            "--create",
            "--title", "Primitive Card Component",
            "--tag", "lapis-engine",   # duplicate of repo — should be deduped
            "--tag", "phase-2",
            "--product", "Archetypal Intelligence",
        ],
        targets_dir=tmp_path,
    )

    assert rc == 0, f"stderr={err!r}"

    data = yaml.safe_load((tmp_path / "lapis-engine-primitive-card-component.yaml").read_text())
    tags = data.get("tags", [])

    # lapis-engine present exactly once
    assert tags.count("lapis-engine") == 1, f"expected lapis-engine once, got {tags}"
    assert "phase-2" in tags
    assert data["product"] == "Archetypal Intelligence"


def test_bind_create_urgency_override(tmp_path):
    """--urgency high propagates to YAML."""
    spec_path = _write_spec_file(tmp_path)

    rc, out, err = _run(
        [
            "bind", "urgent-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--create",
            "--title", "Urgent Thing",
            "--urgency", "high",
        ],
        targets_dir=tmp_path,
    )

    assert rc == 0, f"stderr={err!r}"
    data = yaml.safe_load((tmp_path / "urgent-target.yaml").read_text())
    assert data["urgency"] == "high"


# ---------------------------------------------------------------------------
# Tests: bind --create error cases
# ---------------------------------------------------------------------------


def test_bind_create_missing_title_exits_2(tmp_path):
    """--create without --title → exit 2."""
    spec_path = _write_spec_file(tmp_path)

    rc, out, err = _run(
        [
            "bind", "no-title-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-engine",
            "--create",
            # no --title
        ],
        targets_dir=tmp_path,
    )

    assert rc == 2, f"expected exit 2; got {rc}"
    assert "title" in err.lower()


def test_bind_create_target_exists_no_force(tmp_path):
    """bind --create when target YAML already exists → exit 2 with --force hint."""
    # Pre-create the target
    store = TargetStore(tmp_path)
    store.create("existing-target", title="Existing")

    spec_path = _write_spec_file(tmp_path)

    rc, out, err = _run(
        [
            "bind", "existing-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-engine",
            "--create",
            "--title", "Existing Target",
        ],
        targets_dir=tmp_path,
    )

    assert rc == 2, f"expected exit 2; got {rc}"
    assert "--force" in err, f"expected --force hint in stderr: {err!r}"


def test_bind_create_force_when_target_exists(tmp_path):
    """bind --create --force when target exists → overwrites spec, updates pm_bound, preserves other fields."""
    store = TargetStore(tmp_path)
    t = store.create("existing-target", title="Existing Title")
    # Manually record a history field we want to survive
    t.data["touched"] = "2026-01-01"
    t.save()

    spec_path = _write_spec_file(tmp_path)

    rc, out, err = _run_with_existing_spec(
        [
            "bind", "existing-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-engine",
            "--create",
            "--title", "Existing Target",
            "--force",
        ],
        targets_dir=tmp_path,
    )

    assert rc == 0, f"stderr={err!r}"

    data = yaml.safe_load((tmp_path / "existing-target.yaml").read_text())
    assert data["pm_bound"] is True
    assert data["pm_repo"] == "lapis-engine"
    # Preserved field not stomped
    assert data["touched"] == "2026-01-01"
    # Title unchanged (--force doesn't re-apply create flags to existing target)
    assert data["title"] == "Existing Title"


# ---------------------------------------------------------------------------
# Tests: bind without --create (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_bind_no_create_missing_target_exits_2(tmp_path):
    """bind without --create on a missing target → exit 2 (unchanged behaviour)."""
    spec_path = _write_spec_file(tmp_path)

    rc, out, err = _run(
        [
            "bind", "nonexistent-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-engine",
            # no --create
        ],
        targets_dir=tmp_path,
    )

    assert rc == 2, f"expected exit 2; got {rc}"
    assert "not found" in err


def test_bind_no_create_existing_target_succeeds(tmp_path):
    """bind without --create on an existing target → succeeds (unchanged behaviour)."""
    store = TargetStore(tmp_path)
    store.create("my-target", title="My Target")

    spec_path = _write_spec_file(tmp_path)

    rc, out, err = _run(
        [
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-engine",
        ],
        targets_dir=tmp_path,
    )

    assert rc == 0, f"stderr={err!r}"
    assert "Bound my-target" in out


# ---------------------------------------------------------------------------
# Tests: _derive_tags unit tests
# ---------------------------------------------------------------------------


def test_derive_tags_repo_included():
    assert "lapis-engine" in _derive_tags("lapis-engine", "lapis-engine-scene-director", [])


def test_derive_tags_dedup():
    tags = _derive_tags("lapis-engine", "lapis-engine-x", ["lapis-engine", "phase-2"])
    assert tags.count("lapis-engine") == 1
    assert "phase-2" in tags


def test_derive_tags_repo_first():
    tags = _derive_tags("lapis-engine", "some-target", ["phase-2", "extra"])
    assert tags[0] == "lapis-engine"


def test_derive_tags_empty_extra():
    tags = _derive_tags("my-repo", "my-repo-feature", [])
    assert tags == ["my-repo"]


# ---------------------------------------------------------------------------
# Tests: PM_LIFECYCLE_STAGES structure
# ---------------------------------------------------------------------------


def test_pm_lifecycle_stages_four_stages():
    assert len(PM_LIFECYCLE_STAGES) == 4


def test_pm_lifecycle_stages_first_active():
    assert PM_LIFECYCLE_STAGES[0]["status"] == "active"


def test_pm_lifecycle_stages_rest_pending():
    for stage in PM_LIFECYCLE_STAGES[1:]:
        assert stage["status"] == "pending"
