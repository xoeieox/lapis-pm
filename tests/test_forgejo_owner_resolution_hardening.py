"""Tests for lapis-pm-forgejo-owner-resolution-hardening-v0.

AC1 — adopt-pr preflight is owner-aware (Defect A)
AC2 — bind aborts loud on unresolvable repo (Defect B)
AC3 — classify distinguishes config-404 from transient outage (Defect C)
"""

from __future__ import annotations

import io
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import httpx

from agents_core.targets import TargetStore
from lapis_pm import pm_core, episodic
from lapis_pm.cli import main


SPEC_BODY = textwrap.dedent("""\
    # Hardening test spec

    Minimal spec for owner-resolution hardening unit tests.
""")


def _write_spec(tmp_path: Path) -> Path:
    p = tmp_path / "spec.md"
    p.write_text(SPEC_BODY)
    return p


def _run_bind(argv: list[str], targets_dir: Path, extra_patches=()) -> tuple[int, str, str]:
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    import contextlib
    base_patches = [
        patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
        patch("lapis_pm.cli.episodic.spec", return_value=None),
        patch("lapis_pm.cli.episodic.write_spec", return_value=MagicMock()),
        patch("lapis_pm.cli.pm_core.clear_classified_prs", return_value=None),
        patch("lapis_pm.cli.emit_decision_kickoff", return_value=None),
    ]
    with contextlib.ExitStack() as stack:
        for p in base_patches + list(extra_patches):
            stack.enter_context(p)
        stack.enter_context(redirect_stdout(out_buf))
        stack.enter_context(redirect_stderr(err_buf))
        rc = main(argv)

    return rc, out_buf.getvalue(), err_buf.getvalue()


def _make_404_error() -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "http://forgejo/api/v1/repos/Erah/foo/pulls")
    resp = httpx.Response(404, request=req)
    return httpx.HTTPStatusError("404 Not Found", request=req, response=resp)


def _make_connection_error() -> httpx.ConnectError:
    return httpx.ConnectError("Connection refused")


# ---------------------------------------------------------------------------
# AC1 — adopt-pr preflight is owner-aware (Defect A)
# ---------------------------------------------------------------------------


def test_adopt_pr_preflight_uses_resolved_owner_for_slashed_repo(tmp_path):
    """With --adopt-pr N --repo lapis/foo, _get_pr is called with ('foo', N, owner='lapis')."""
    spec_path = _write_spec(tmp_path)

    captured_calls: list = []

    def fake_get_pr(repo, pr_number, owner=None):
        captured_calls.append((repo, pr_number, owner))
        return {
            "number": 1,
            "state": "open",
            "merged": False,
            "head": {"ref": "lapis/my-target/forced", "repo": {"name": "foo"}},
            "base": {"ref": "main", "repo": {"name": "foo"}},
        }

    rc, out, err = _run_bind(
        [
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis/foo",
            "--authority", "advisory",
            "--create",
            "--title", "Owner Test",
            "--adopt-pr", "1",
        ],
        targets_dir=tmp_path,
        extra_patches=[
            patch("agents_core.forgejo.get_pr", side_effect=fake_get_pr),
            # Probe succeeds (empty list = repo exists)
            patch("agents_core.forgejo.get_open_prs", return_value=[]),
        ],
    )

    assert len(captured_calls) >= 1, f"get_pr was never called; stderr={err!r}"
    repo_arg, pr_num_arg, owner_kwarg = captured_calls[0]
    assert repo_arg == "foo", f"expected repo='foo', got {repo_arg!r}"
    assert pr_num_arg == 1, f"expected pr_number=1, got {pr_num_arg!r}"
    assert owner_kwarg == "lapis", f"expected owner='lapis', got {owner_kwarg!r}"
    assert rc == 0, f"expected exit 0; stderr={err!r}"


def test_adopt_pr_preflight_bare_repo_passes_owner_none(tmp_path):
    """With --adopt-pr N --repo foo (bare), _get_pr is called with ('foo', N, owner=None)."""
    spec_path = _write_spec(tmp_path)

    captured_calls: list = []

    def fake_get_pr(repo, pr_number, owner=None):
        captured_calls.append((repo, pr_number, owner))
        return {
            "number": 5,
            "state": "open",
            "merged": False,
            "head": {"ref": "lapis/bare-target/forced", "repo": {"name": "foo"}},
            "base": {"ref": "main", "repo": {"name": "foo"}},
        }

    rc, out, err = _run_bind(
        [
            "bind", "bare-target",
            "--spec-from", str(spec_path),
            "--repo", "foo",
            "--authority", "advisory",
            "--create",
            "--title", "Bare Repo Test",
            "--adopt-pr", "5",
        ],
        targets_dir=tmp_path,
        extra_patches=[
            patch("agents_core.forgejo.get_pr", side_effect=fake_get_pr),
            patch("agents_core.forgejo.get_open_prs", return_value=[]),
        ],
    )

    assert len(captured_calls) >= 1, "get_pr was never called"
    _repo, _num, _owner = captured_calls[0]
    assert _repo == "foo"
    assert _num == 5
    assert _owner is None, f"expected owner=None for bare repo, got {_owner!r}"


# ---------------------------------------------------------------------------
# AC2 — bind aborts loud on unresolvable repo (Defect B)
# ---------------------------------------------------------------------------


def test_bind_aborts_on_404_repo(tmp_path):
    """bind --repo <unresolvable> 404s → exit 2, no bind_pm call, remedy in stderr."""
    spec_path = _write_spec(tmp_path)

    bind_pm_calls: list = []

    rc, out, err = _run_bind(
        [
            "bind", "orphan-target",
            "--spec-from", str(spec_path),
            "--repo", "rag-ops",
            "--authority", "advisory",
            "--create",
            "--title", "Orphan Target",
        ],
        targets_dir=tmp_path,
        extra_patches=[
            patch("agents_core.forgejo.get_open_prs", side_effect=_make_404_error()),
        ],
    )

    assert rc == 2, f"expected exit 2; got {rc}; stderr={err!r}"
    assert "not found" in err.lower() or "404" in err.lower(), f"remedy text missing: {err!r}"
    assert "lapis/rag-ops" in err or "rag-ops" in err, f"repo name missing from message: {err!r}"
    assert "nothing was written" in err.lower(), f"'nothing was written' missing: {err!r}"
    # Target YAML must NOT exist (no state persisted)
    assert not (tmp_path / "orphan-target.yaml").exists(), "target YAML should not be created on 404"


def test_bind_proceeds_on_200(tmp_path):
    """bind --repo <valid-repo> → existence probe 200 → exit 0, bind proceeds normally."""
    spec_path = _write_spec(tmp_path)

    rc, out, err = _run_bind(
        [
            "bind", "valid-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis/myrepo",
            "--authority", "advisory",
            "--create",
            "--title", "Valid Target",
        ],
        targets_dir=tmp_path,
        extra_patches=[
            patch("agents_core.forgejo.get_open_prs", return_value=[]),
        ],
    )

    assert rc == 0, f"expected exit 0; stderr={err!r}"
    assert (tmp_path / "valid-target.yaml").exists(), "target YAML should be created on 200"


def test_bind_warns_on_connection_error_and_proceeds(tmp_path):
    """bind with Forgejo unreachable (connection error) → prints WARNING but proceeds (exit 0)."""
    spec_path = _write_spec(tmp_path)

    rc, out, err = _run_bind(
        [
            "bind", "offline-target",
            "--spec-from", str(spec_path),
            "--repo", "myrepo",
            "--authority", "advisory",
            "--create",
            "--title", "Offline Target",
        ],
        targets_dir=tmp_path,
        extra_patches=[
            patch("agents_core.forgejo.get_open_prs", side_effect=_make_connection_error()),
        ],
    )

    assert rc == 0, f"expected exit 0 on connectivity error; got {rc}; stderr={err!r}"
    assert "warning" in err.lower(), f"expected WARNING in stderr: {err!r}"
    assert (tmp_path / "offline-target.yaml").exists(), "target YAML should be created despite warning"


def test_force_rebind_to_404_repo_aborts_without_clobbering(tmp_path):
    """--force rebind to a 404 repo exits 2 without overwriting the existing binding."""
    import yaml

    spec_path = _write_spec(tmp_path)

    # Create a valid binding first
    _run_bind(
        [
            "bind", "existing-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis/good-repo",
            "--authority", "advisory",
            "--create",
            "--title", "Existing Target",
        ],
        targets_dir=tmp_path,
        extra_patches=[
            patch("agents_core.forgejo.get_open_prs", return_value=[]),
        ],
    )
    original_data = yaml.safe_load((tmp_path / "existing-target.yaml").read_text())

    # Now try to force-rebind to a 404 repo
    rc, out, err = _run_bind(
        [
            "bind", "existing-target",
            "--spec-from", str(spec_path),
            "--repo", "bad-repo",
            "--authority", "advisory",
            "--force",
        ],
        targets_dir=tmp_path,
        extra_patches=[
            patch("agents_core.forgejo.get_open_prs", side_effect=_make_404_error()),
        ],
    )

    assert rc == 2, f"expected exit 2 on force rebind to 404 repo; got {rc}"
    current_data = yaml.safe_load((tmp_path / "existing-target.yaml").read_text())
    assert current_data.get("pm_repo") == original_data.get("pm_repo"), (
        "pm_repo was clobbered by failed force rebind"
    )


# ---------------------------------------------------------------------------
# AC3 — classify distinguishes config-404 from transient outage (Defect C)
# ---------------------------------------------------------------------------


def test_perceive_prs_404_emits_repo_unresolved_tag(tmp_path):
    """_perceive_prs with a 404 HTTPStatusError emits pm:repo-unresolved once."""
    pm_core._repo_unresolved_signalled.discard("unresolved-tid")

    observations: list[dict] = []

    def fake_write_obs(tid, content, extra_tags=None):
        observations.append({"tid": tid, "content": content, "tags": extra_tags or []})
        c = MagicMock()
        return c

    with (
        patch("lapis_pm.pm_core.get_open_prs", side_effect=_make_404_error()),
        patch("lapis_pm.pm_core.episodic.write_observation", side_effect=fake_write_obs),
    ):
        prs, ok = pm_core._perceive_prs("unresolved-tid", "rag-ops")

    assert not ok or prs == [], f"expected empty PRs on 404; got prs={prs}, ok={ok}"
    assert any("pm:repo-unresolved" in o["tags"] for o in observations), (
        f"pm:repo-unresolved tag missing from observations: {observations}"
    )
    assert not any("pm:error" in o["tags"] and "pm:repo-unresolved" not in o["tags"]
                   for o in observations), (
        "404 should NOT emit a plain pm:error observation without pm:repo-unresolved"
    )


def test_perceive_prs_404_signal_deduped_on_second_tick(tmp_path):
    """_perceive_prs 404 emits the loud signal exactly once across repeated calls."""
    pm_core._repo_unresolved_signalled.discard("dedup-tid")

    observations: list[dict] = []

    def fake_write_obs(tid, content, extra_tags=None):
        observations.append({"tid": tid, "content": content, "tags": extra_tags or []})
        return MagicMock()

    with (
        patch("lapis_pm.pm_core.get_open_prs", side_effect=_make_404_error()),
        patch("lapis_pm.pm_core.episodic.write_observation", side_effect=fake_write_obs),
    ):
        pm_core._perceive_prs("dedup-tid", "rag-ops")

    # Reset side_effect for second call
    with (
        patch("lapis_pm.pm_core.get_open_prs", side_effect=_make_404_error()),
        patch("lapis_pm.pm_core.episodic.write_observation", side_effect=fake_write_obs),
    ):
        pm_core._perceive_prs("dedup-tid", "rag-ops")

    repo_unresolved_obs = [o for o in observations if "pm:repo-unresolved" in o["tags"]]
    assert len(repo_unresolved_obs) == 1, (
        f"expected exactly one pm:repo-unresolved signal, got {len(repo_unresolved_obs)}: {observations}"
    )

    # Cleanup
    pm_core._repo_unresolved_signalled.discard("dedup-tid")


def test_perceive_prs_connection_error_uses_quiet_pm_error_path(tmp_path):
    """_perceive_prs with a connection error (not 404) uses the existing quiet pm:error path."""
    pm_core._repo_unresolved_signalled.discard("conn-err-tid")

    observations: list[dict] = []

    def fake_write_obs(tid, content, extra_tags=None):
        observations.append({"tid": tid, "content": content, "tags": extra_tags or []})
        return MagicMock()

    with (
        patch("lapis_pm.pm_core.get_open_prs", side_effect=_make_connection_error()),
        patch("lapis_pm.pm_core.episodic.write_observation", side_effect=fake_write_obs),
    ):
        prs, ok = pm_core._perceive_prs("conn-err-tid", "some-repo")

    assert prs == [] and not ok, f"expected empty/false on connection error; got {prs}, {ok}"
    assert any("pm:error" in o["tags"] for o in observations), (
        "connection error should emit pm:error observation"
    )
    assert not any("pm:repo-unresolved" in o["tags"] for o in observations), (
        "connection error must NOT emit pm:repo-unresolved"
    )
