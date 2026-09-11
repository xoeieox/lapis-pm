"""Tests for `lapis-pm bind --adopt-pr` (lapis-pm-bind-adopt-pr-v0 spec).

Coverage:
  - bind --adopt-pr records adopted_pr_number + adopted_head_branch on target
  - bind --adopt-pr with closed/merged PR → exit 2
  - bind --adopt-pr with repo mismatch → exit 2
  - _perceive_prs includes adopted PR even when branch does not start with lapis/<tid>/
  - tick decides reviewer-first (not initial fixer) for adopted target
  - force_dispatch raises for initial fixer on adopted target
"""

from __future__ import annotations

import io
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents_core.targets import TargetStore
from lapis_pm import pm_core, authority
from lapis_pm.cli import main


SPEC_BODY = textwrap.dedent("""\
    # Adopt-PR test spec

    Minimal spec body for --adopt-pr unit tests.
""")


def _write_spec_file(tmp_path: Path) -> Path:
    p = tmp_path / "spec.md"
    p.write_text(SPEC_BODY)
    return p


def _fake_pr(
    number: int = 42,
    state: str = "open",
    merged: bool = False,
    head_ref: str = "feat/my-feature",
    base_repo: str = "lapis-test",
) -> dict:
    return {
        "number": number,
        "state": state,
        "merged": merged,
        "head": {"ref": head_ref, "repo": {"name": "lapis-test"}},
        "base": {"ref": "main", "repo": {"name": base_repo}},
        "html_url": f"http://forgejo/Erah/lapis-test/pulls/{number}",
    }


def _run_bind(
    argv: list[str],
    targets_dir: Path,
    fake_pr_data: dict | None = None,
) -> tuple[int, str, str]:
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    ctx_patches = [
        patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
        patch("lapis_pm.cli.episodic.spec", return_value=None),
            patch("lapis_pm.cli.episodic.write_spec", return_value=MagicMock()),
        patch("lapis_pm.cli.pm_core.clear_classified_prs", return_value=None),
            patch("lapis_pm.cli.emit_decision_kickoff", return_value=None),
        # Mock existence probe so tests don't depend on Forgejo connectivity.
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
    ]
    if fake_pr_data is not None:
        ctx_patches.append(patch("agents_core.forgejo.get_pr", return_value=fake_pr_data))

    import contextlib
    with contextlib.ExitStack() as stack:
        for p in ctx_patches:
            stack.enter_context(p)
        stack.enter_context(redirect_stdout(out_buf))
        stack.enter_context(redirect_stderr(err_buf))
        rc = main(argv)

    return rc, out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# bind --adopt-pr happy path
# ---------------------------------------------------------------------------


def test_bind_adopt_pr_records_fields(tmp_path):
    """bind --adopt-pr records adopted_pr_number + adopted_head_branch on the target YAML."""
    spec_path = _write_spec_file(tmp_path)
    pr = _fake_pr(number=42, head_ref="feat/my-feature")

    rc, out, err = _run_bind(
        [
            "bind", "my-adopt-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-test",
            "--authority", "advisory",
            "--create",
            "--title", "Adopt-PR Target",
            "--adopt-pr", "42",
        ],
        targets_dir=tmp_path,
        fake_pr_data=pr,
    )

    assert rc == 0, f"expected exit 0; stderr={err!r}"

    data = yaml.safe_load((tmp_path / "my-adopt-target.yaml").read_text())
    assert data.get("adopted_pr_number") == 42, f"adopted_pr_number not set: {data}"
    assert data.get("adopted_head_branch") == "feat/my-feature", (
        f"adopted_head_branch wrong: {data}"
    )
    assert data["pm_bound"] is True
    assert "Adopted PR #42" in out, f"adoption confirmation missing from stdout: {out!r}"


def test_bind_adopt_pr_non_adopted_path_unchanged(tmp_path):
    """bind without --adopt-pr does not write adopted fields (non-adopted path byte-identical)."""
    spec_path = _write_spec_file(tmp_path)

    rc, out, err = _run_bind(
        [
            "bind", "plain-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-test",
            "--create",
            "--title", "Plain Target",
        ],
        targets_dir=tmp_path,
    )

    assert rc == 0, f"stderr={err!r}"
    data = yaml.safe_load((tmp_path / "plain-target.yaml").read_text())
    assert "adopted_pr_number" not in data, "adopted_pr_number should be absent for non-adopted target"
    assert "adopted_head_branch" not in data, "adopted_head_branch should be absent for non-adopted target"


# ---------------------------------------------------------------------------
# bind --adopt-pr error cases
# ---------------------------------------------------------------------------


def test_bind_adopt_pr_closed_pr_rejected(tmp_path):
    """bind --adopt-pr with a closed PR → exit 2."""
    spec_path = _write_spec_file(tmp_path)
    pr = _fake_pr(number=7, state="closed", merged=False)

    rc, out, err = _run_bind(
        [
            "bind", "adopt-closed",
            "--spec-from", str(spec_path),
            "--repo", "lapis-test",
            "--create",
            "--title", "Adopt Closed",
            "--adopt-pr", "7",
        ],
        targets_dir=tmp_path,
        fake_pr_data=pr,
    )

    assert rc == 2, f"expected exit 2; got {rc}"
    assert "not open" in err.lower() or "closed" in err.lower(), f"stderr: {err!r}"


def test_bind_adopt_pr_merged_pr_rejected(tmp_path):
    """bind --adopt-pr with a merged PR → exit 2."""
    spec_path = _write_spec_file(tmp_path)
    pr = _fake_pr(number=8, state="closed", merged=True)

    rc, out, err = _run_bind(
        [
            "bind", "adopt-merged",
            "--spec-from", str(spec_path),
            "--repo", "lapis-test",
            "--create",
            "--title", "Adopt Merged",
            "--adopt-pr", "8",
        ],
        targets_dir=tmp_path,
        fake_pr_data=pr,
    )

    assert rc == 2, f"expected exit 2; got {rc}"
    assert "not open" in err.lower() or "merged" in err.lower(), f"stderr: {err!r}"


def test_bind_adopt_pr_repo_mismatch_rejected(tmp_path):
    """bind --adopt-pr when PR base repo mismatches --repo → exit 2."""
    spec_path = _write_spec_file(tmp_path)
    pr = _fake_pr(number=9, base_repo="other-repo")

    rc, out, err = _run_bind(
        [
            "bind", "adopt-mismatch",
            "--spec-from", str(spec_path),
            "--repo", "lapis-test",
            "--create",
            "--title", "Adopt Mismatch",
            "--adopt-pr", "9",
        ],
        targets_dir=tmp_path,
        fake_pr_data=pr,
    )

    assert rc == 2, f"expected exit 2; got {rc}"
    assert "repo" in err.lower(), f"stderr: {err!r}"


# ---------------------------------------------------------------------------
# Perception: _perceive_prs honors adopted PR
# ---------------------------------------------------------------------------


def test_perceive_prs_includes_adopted_pr(tmp_path):
    """_perceive_prs returns the adopted PR even when its branch is feat/* (not lapis/<tid>/)."""
    target_id = "my-adopt-target"
    adopted_pr = {
        "number": 42,
        "state": "open",
        "head": {"ref": "feat/my-feature"},
        "base": {"ref": "main"},
    }

    with patch("lapis_pm.pm_core.get_open_prs", return_value=[adopted_pr]):
        prs, ok = pm_core._perceive_prs(target_id, "lapis-test", adopted_pr_number=42)

    assert ok
    assert any(p["number"] == 42 for p in prs), (
        f"adopted PR #42 not in perceived PRs: {prs}"
    )


def test_perceive_prs_excludes_non_adopted_foreign_branch(tmp_path):
    """_perceive_prs excludes PRs on branches that are neither lapis/<tid>/ nor adopted."""
    target_id = "my-adopt-target"
    unrelated_pr = {
        "number": 99,
        "state": "open",
        "head": {"ref": "feat/unrelated"},
        "base": {"ref": "main"},
    }

    with patch("lapis_pm.pm_core.get_open_prs", return_value=[unrelated_pr]):
        prs, ok = pm_core._perceive_prs(target_id, "lapis-test", adopted_pr_number=42)

    assert ok
    assert not any(p["number"] == 99 for p in prs), "unrelated PR should not be perceived"


def test_perceive_prs_non_adopted_path_unchanged():
    """_perceive_prs without adopted_pr_number behaves exactly as before (byte-identical)."""
    target_id = "plain-target"
    lapis_pr = {
        "number": 55,
        "state": "open",
        "head": {"ref": "lapis/plain-target/my-branch"},
        "base": {"ref": "main"},
    }
    other_pr = {
        "number": 66,
        "state": "open",
        "head": {"ref": "feat/unrelated"},
        "base": {"ref": "main"},
    }

    with patch("lapis_pm.pm_core.get_open_prs", return_value=[lapis_pr, other_pr]):
        prs, ok = pm_core._perceive_prs(target_id, "lapis-test")  # no adopted_pr_number

    assert ok
    assert len(prs) == 1
    assert prs[0]["number"] == 55


# ---------------------------------------------------------------------------
# Decide: reviewer dispatched first for adopted target
# ---------------------------------------------------------------------------


def test_tick_adopted_target_dispatches_reviewer_first(tmp_path):
    """tick() on an adopted target with the adopted PR visible dispatches a reviewer, not fixer."""
    # Create a target YAML with adopted fields
    store = TargetStore(tmp_path)
    t = store.create("adopted-tid", title="Adopted Target")
    t.bind_pm(repo="lapis-test", authority="advisory")
    t.data["adopted_pr_number"] = 42
    t.data["adopted_head_branch"] = "feat/my-feature"
    t.save()

    adopted_pr = {
        "number": 42,
        "title": "feat: my feature",
        "html_url": "http://forgejo/Erah/lapis-test/pulls/42",
        "head": {"ref": "feat/my-feature"},
        "base": {"ref": "main"},
        "mergeable": True,
    }

    mock_cls = authority.PRClassification(
        verdict="advisory",
        screen_verdict="unknown",
        static_outcome=authority.StaticOutcome.static_pass,
        reasons=[],
        issues=[],
        pr_number=42,
        repo="lapis-test",
        title="feat: my feature",
        html_url="http://forgejo/Erah/lapis-test/pulls/42",
        changed_paths=[],
        diff_loc=0,
        diff="",
    )

    dispatched_agent_types: list[str] = []

    def capture_dispatch(agent_type, tid, user_prompt, vars_=None, **kw):
        dispatched_agent_types.append(agent_type)
        r = MagicMock()
        r.task_id = "smoke-reviewer-task"
        r.spec_id = "smoke-spec"
        return r

    with (
        patch("lapis_pm.pm_core.TargetStore", lambda: store),
            patch("lapis_pm.pm_core._perceive_prs", return_value=([adopted_pr], True)),
        patch("lapis_pm.pm_core.authority.classify", return_value=mock_cls),
            patch.object(pm_core._SHAPER, "dispatch", side_effect=capture_dispatch),
        patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec summary"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
        patch("lapis_pm.pm_core.append_dispatched"),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("agents_core.forgejo.get_pr_diff", return_value="diff here"),
            patch("lapis_pm.pm_core.Shaper.resolve_repo_cwd", return_value="/tmp/smoke"),
        ):
        result = pm_core.tick("adopted-tid")

    assert any("reviewer" in a for a in dispatched_agent_types), (
        f"expected reviewer dispatch; got: {dispatched_agent_types}"
    )
    assert not any(a == "fixer" for a in dispatched_agent_types), (
        f"initial fixer was dispatched for adopted target (must not be): {dispatched_agent_types}"
    )
    assert "reviewer_dispatched" in result.decision, (
        f"decision should contain reviewer_dispatched: {result.decision}"
    )


# ---------------------------------------------------------------------------
# force_dispatch guard: initial fixer blocked for adopted targets
# ---------------------------------------------------------------------------


def test_force_dispatch_blocks_initial_fixer_for_adopted_target(tmp_path):
    """force_dispatch raises ValueError when asked to fire initial fixer for an adopted target."""
    store = TargetStore(tmp_path)
    t = store.create("adopted-fixer-guard", title="Guard Test")
    t.bind_pm(repo="lapis-test", authority="advisory")
    t.data["adopted_pr_number"] = 42
    t.data["adopted_head_branch"] = "feat/guarded"
    t.save()

    with (
        patch("lapis_pm.pm_core.TargetStore", lambda: store),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            ):
        with pytest.raises(ValueError, match="adopted PR"):
            pm_core.force_dispatch("adopted-fixer-guard", "fixer", "implement from scratch")


def test_force_dispatch_allows_fixer_after_prior_fixer_dispatch(tmp_path):
    """force_dispatch allows fixer if a prior fixer dispatch already exists (not the initial fixer)."""
    store = TargetStore(tmp_path)
    t = store.create("adopted-fixer-ok", title="Fixer OK Test")
    t.bind_pm(repo="lapis-test", authority="advisory")
    t.data["adopted_pr_number"] = 42
    t.data["adopted_head_branch"] = "feat/guarded"
    t.save()

    prior_fixer_record = {
        "gpu_id": "prior-fixer-001",
        "agent_type": "fixer",
        "status": "processed",
        "intent": "prior fixer",
        "repo": "lapis-test",
        "ts": "2026-01-01T00:00:00Z",
        "retry_count": 0,
    }

    captured: dict = {}

    def fake_dispatch(agent_type, tid, user_prompt, vars_=None, **kw):
        captured["agent_type"] = agent_type
        r = MagicMock()
        r.task_id = "fixer-task-ok"
        r.spec_id = "spec-ok"
        return r

    with (
        patch("lapis_pm.pm_core.TargetStore", lambda: store),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[prior_fixer_record]),
            patch("lapis_pm.pm_core.episodic.spec_summary", return_value="spec"),
            patch("lapis_pm.pm_core.episodic.write_dispatch"),
        patch("lapis_pm.pm_core.append_dispatched"),
            patch.object(pm_core._SHAPER, "dispatch", side_effect=fake_dispatch),
    ):
        pm_core.force_dispatch("adopted-fixer-ok", "fixer", "re-implement after review")

    assert captured.get("agent_type") == "fixer", "fixer should be dispatched after prior fixer"
