"""Tests for U1 - pipeline-attested machine verification
(lapis-pm-autonomy-actuator-v0).

Coverage (per spec AC list):
  AC4  containment pin intact      -> _bind still pins pm-live-test
  AC5  runner-control intersection -> inconclusive (fail-safe)
  AC5  attestation timeout         -> inconclusive (fail-safe)
  AC5  unattested / inconclusive   -> never flip pm_verification
  U1.1 attest() result shapes      -> attested / unattested / inconclusive
  U1.2 cache discipline            -> one run per head SHA (force-push re-runs)
  U1.3 containment invariant        -> the hook never consults the spec-prose
                                        path (--verification / spec field)
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import verification_attestation as va


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mem(store: dict | None = None) -> MagicMock:
    """Mock MemoryStore with .set / .get wired up."""
    data: dict[str, dict] = {}
    if store:
        data.update(store)

    mock = MagicMock()

    def _set(key: str, content: str, tags: list | None = None, source: str = "") -> bool:
        data[key] = {"key": key, "content": content, "tags": tags or []}
        return True

    def _get(key: str) -> dict | None:
        return data.get(key)

    mock.set.side_effect = _set
    mock.get.side_effect = _get
    mock._data = data
    return mock


def _make_cls(changed_paths: list[str] | None = None) -> MagicMock:
    cls = MagicMock()
    cls.screen_verdict = "clean"
    cls.issues = []
    cls.pr_number = 42
    cls.repo = "lapis-pm"
    cls.title = "feat: add thing"
    cls.html_url = "http://forgejo/pr/42"
    cls.diff = ""
    cls.diff_loc = 10
    cls.reasons = []
    cls.changed_paths = changed_paths if changed_paths is not None else ["lapis_pm/foo.py"]
    return cls


def _make_target(pm_verification: str = "pm-live-test") -> MagicMock:
    t = MagicMock()
    t.id = "my-target"
    t.pm_authority = "advisory"
    t.pm_repo = "lapis-pm"
    t.data = {"pm_verification": pm_verification}
    return t


def _att(result: str, **kw) -> dict:
    base = {"result": result, "pr_failures": [], "preexisting": [],
            "suite": "python3 -m pytest -q", "ts": "2026-09-06T00:00:00Z"}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# U1.1 - attest() result shapes
# ---------------------------------------------------------------------------

class TestAttestResults:

    def test_missing_head_sha_inconclusive(self):
        res = va.attest("lapis-pm", 42, "")
        assert res["result"] == "inconclusive"
        assert res["reason"] == "missing_head_sha"

    def test_runner_control_smoke_sh_inconclusive(self):
        res = va.attest("lapis-pm", 42, "abc123",
                        changed_paths=["smoke.sh"])
        assert res["result"] == "inconclusive"
        assert res["reason"] == "runner_control_intersection"

    def test_runner_control_pyproject_inconclusive(self):
        res = va.attest("lapis-pm", 42, "abc123",
                        changed_paths=["pyproject.toml"])
        assert res["result"] == "inconclusive"
        assert res["reason"] == "runner_control_intersection"

    def test_runner_control_conftest_any_depth_inconclusive(self):
        res = va.attest("lapis-pm", 42, "abc123",
                        changed_paths=["tests/sub/conftest.py"])
        assert res["result"] == "inconclusive"
        assert res["reason"] == "runner_control_intersection"

    def test_runner_control_tests_ini_inconclusive(self):
        res = va.attest("lapis-pm", 42, "abc123",
                        changed_paths=["tests/pytest.ini"])
        assert res["result"] == "inconclusive"
        assert res["reason"] == "runner_control_intersection"

    def test_no_runner_control_hit_degrades_to_clone_missing(self):
        # A normal code PR does not hit the runner-control set; with no local
        # clone it degrades to inconclusive (clone_missing), NOT runner-control.
        res = va.attest("lapis-pm", 42, "abc123",
                        changed_paths=["lapis_pm/foo.py"])
        assert res["result"] == "inconclusive"
        assert res["reason"] != "runner_control_intersection"

    def test_attested_no_new_failures(self):
        # PR head and main have the same failures -> attested.
        with (
            patch.object(va, "_repo_clone_path", return_value="/tmp/fake-clone"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch.object(va, "_create_worktree"),
            patch.object(va, "_remove_worktree"),
            patch.object(va, "_run_suite_in_worktree",
                         side_effect=[("python3 -m pytest -q", ["tests/test_x.py::test_a"], 0),
                                       ("python3 -m pytest -q", ["tests/test_x.py::test_a"], 0),
                         ]),
        ):
            res = va.attest("lapis-pm", 42, "abc123",
                            changed_paths=["lapis_pm/foo.py"])
        assert res["result"] == "attested"
        assert res["pr_failures"] == ["tests/test_x.py::test_a"]
        assert res["preexisting"] == ["tests/test_x.py::test_a"]

    def test_unattested_new_failure_on_pr_head(self):
        # PR head introduces a failure not on main -> unattested.
        with (
            patch.object(va, "_repo_clone_path", return_value="/tmp/fake-clone"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch.object(va, "_create_worktree"),
            patch.object(va, "_remove_worktree"),
            patch.object(va, "_run_suite_in_worktree",
                         side_effect=[("python3 -m pytest -q", ["tests/test_x.py::test_new"], 1),
                                       ("python3 -m pytest -q", [], 0),
                         ]),
        ):
            res = va.attest("lapis-pm", 42, "abc123",
                            changed_paths=["lapis_pm/foo.py"])
        assert res["result"] == "unattested"
        assert res["pr_failures"] == ["tests/test_x.py::test_new"]

    def test_timeout_inconclusive(self):
        # AC5: attestation timeout -> inconclusive (fail-safe).
        with (
            patch.object(va, "_repo_clone_path", return_value="/tmp/fake-clone"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch.object(va, "_create_worktree"),
            patch.object(va, "_remove_worktree"),
            patch.object(va, "_run_suite_in_worktree",
                         side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=900)),
        ):
            res = va.attest("lapis-pm", 42, "abc123",
                            changed_paths=["lapis_pm/foo.py"])
        assert res["result"] == "inconclusive"
        assert res["reason"] == "wall_budget_exceeded"

    def test_red_suite_no_parseable_failures_inconclusive(self):
        # Hard invariant (fail-safe): a suite that exits non-zero (red) with
        # NO parseable FAILED lines must yield inconclusive, NOT attested —
        # e.g. a repo whose entry point is `bash smoke.sh` and whose red run
        # prints no FAILED lines at all.
        with (
            patch.object(va, "_repo_clone_path", return_value="/tmp/fake-clone"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch.object(va, "_create_worktree"),
            patch.object(va, "_remove_worktree"),
            patch.object(va, "_run_suite_in_worktree",
                         side_effect=[("bash smoke.sh", [], 1),
                                       ("bash smoke.sh", [], 0),
                         ]),
        ):
            res = va.attest("lapis-pm", 42, "abc123",
                            changed_paths=["lapis_pm/foo.py"])
        assert res["result"] == "inconclusive"
        assert res["reason"] == "red_suite_unparseable_failures"

    def test_green_suite_no_failures_attested(self):
        # Zero exit, no failures on either head -> attested.
        with (
            patch.object(va, "_repo_clone_path", return_value="/tmp/fake-clone"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch.object(va, "_create_worktree"),
            patch.object(va, "_remove_worktree"),
            patch.object(va, "_run_suite_in_worktree",
                         side_effect=[("bash smoke.sh", [], 0),
                                       ("bash smoke.sh", [], 0),
                         ]),
        ):
            res = va.attest("lapis-pm", 42, "abc123",
                            changed_paths=["lapis_pm/foo.py"])
        assert res["result"] == "attested"
        assert res["pr_failures"] == []
        assert res["preexisting"] == []

    def test_parser_recognizes_failed_lines_and_summary(self):
        # The parser recognizes BOTH pytest -q failure shapes: short-summary
        # "FAILED path::test" lines and the "= N failed" summary line.
        out = "FAILED tests/test_x.py::test_a - AssertionError\n"
        failures = va._pr_failures_from_output(out)
        assert "tests/test_x.py::test_a" in failures

        summary = "= 3 failed, 1 passed in 1.23s =\n"
        failures = va._pr_failures_from_output(summary)
        assert failures  # failure evidence present (the summary marker)

        # No failures at all -> empty set.
        assert va._pr_failures_from_output("4 passed in 0.1s\n") == []

    def test_attest_main_baseline_green(self):
        # Dedicated main-baseline run: one detached worktree at origin/main,
        # green suite -> attested with an empty failure set.
        with (
            patch.object(va, "_repo_clone_path", return_value="/tmp/fake-clone"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch.object(va, "_create_worktree") as mock_create,
            patch.object(va, "_remove_worktree"),
            patch.object(va, "_run_suite_in_worktree",
                         return_value=("bash smoke.sh", [], 0)),
        ):
            res = va.attest_main_baseline("lapis-pm", base_branch="main")
        assert res["result"] == "attested"
        assert res["pr_failures"] == []
        assert res["suite"] == "bash smoke.sh"
        # ONE worktree, detached at origin/main (no _head_branch_for).
        assert mock_create.call_count == 1
        assert mock_create.call_args.args[1] == "origin/main"

    def test_attest_main_baseline_red_with_failures(self):
        # Red main baseline with parseable failures -> unattested, failures
        # carried in pr_failures (the backstop reads them as main_failures).
        with (
            patch.object(va, "_repo_clone_path", return_value="/tmp/fake-clone"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch.object(va, "_create_worktree"),
            patch.object(va, "_remove_worktree"),
            patch.object(va, "_run_suite_in_worktree",
                         return_value=("python3 -m pytest -q",
                                       ["tests/test_x.py::test_a"], 1)),
        ):
            res = va.attest_main_baseline("lapis-pm")
        assert res["result"] == "unattested"
        assert res["pr_failures"] == ["tests/test_x.py::test_a"]

    def test_attest_main_baseline_red_no_failures_inconclusive(self):
        # Hard invariant: red main baseline with no parseable failures is
        # inconclusive (fail-safe), never attested.
        with (
            patch.object(va, "_repo_clone_path", return_value="/tmp/fake-clone"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch.object(va, "_create_worktree"),
            patch.object(va, "_remove_worktree"),
            patch.object(va, "_run_suite_in_worktree",
                         return_value=("bash smoke.sh", [], 1)),
        ):
            res = va.attest_main_baseline("lapis-pm")
        assert res["result"] == "inconclusive"
        assert res["reason"] == "red_suite_unparseable_failures"

    def test_attest_main_baseline_timeout_inconclusive(self):
        # Timeout in the main-baseline run -> inconclusive (wall budget).
        with (
            patch.object(va, "_repo_clone_path", return_value="/tmp/fake-clone"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch.object(va, "_create_worktree"),
            patch.object(va, "_remove_worktree"),
            patch.object(va, "_run_suite_in_worktree",
                         side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=900)),
        ):
            res = va.attest_main_baseline("lapis-pm")
        assert res["result"] == "inconclusive"
        assert res["reason"] == "wall_budget_exceeded"


# ---------------------------------------------------------------------------
# U1.2 hook slug fix: the attestation hook derives the PR head slug from the
# payload's pr.head.ref and passes it to attest(); a missing/mismatched ref
# skips the attestation entirely (never call attest with a wrong slug).
# ---------------------------------------------------------------------------

class TestHookSlug:

    def _run_hook(self, payload: dict, att: dict):
        from lapis_pm import pm_core
        cls = _make_cls()
        target = _make_target(pm_verification="pm-live-test")
        with (
            patch.object(pm_core, "TargetStore") as mock_store,
            patch.object(pm_core, "_mem", return_value=_make_mem()),
            patch.object(va, "attest", return_value=att) as mock_attest,
            patch.object(va, "read_cache", return_value=None),
            patch.object(va, "write_cache"),
            patch("lapis_pm.episodic.write_observation"),
            patch.dict("os.environ", {"PM_AUTONOMY_ACTUATOR": "shadow"}),
        ):
            mock_store.return_value.get.return_value = target
            pm_core._attestation_hook("my-target", cls, payload)
        return mock_attest

    def test_hook_passes_slug_from_head_ref(self):
        # pr.head.ref "lapis/my-target/local" -> attest receives slug="local".
        payload = {"pr": {"head": {"sha": "abc123",
                                   "ref": "lapis/my-target/local"}},
                   "classification": None}
        mock_attest = self._run_hook(payload, _att("attested"))
        assert mock_attest.called
        assert mock_attest.call_args.kwargs.get("slug") == "local"

    def test_hook_not_called_when_ref_absent(self):
        # No pr.head.ref -> attest is NOT called (skip; inconclusive by
        # omission is fail-safe).
        payload = {"pr": {"head": {"sha": "abc123"}}, "classification": None}
        mock_attest = self._run_hook(payload, _att("attested"))
        assert not mock_attest.called

    def test_hook_not_called_when_ref_prefix_mismatch(self):
        # A ref that does not start with "lapis/<target_id>/" -> attest is
        # NOT called (never call attest with a wrong slug).
        payload = {"pr": {"head": {"sha": "abc123",
                                   "ref": "lapis/other-target/local"}},
                   "classification": None}
        mock_attest = self._run_hook(payload, _att("attested"))
        assert not mock_attest.called


# ---------------------------------------------------------------------------
# U1.2 - cache discipline (one run per head SHA)
# ---------------------------------------------------------------------------

class TestCacheDiscipline:

    def test_cache_reused_for_same_head_sha(self):
        mem = _make_mem()
        va.write_cache(mem, "my-target", 42, "abc123",
                       _att("attested"), shadow=False)
        cached = va.read_cache(mem, "my-target", 42, "abc123")
        assert cached is not None
        assert cached["result"] == "attested"
        assert cached["head_sha"] == "abc123"

    def test_cache_invalidated_on_force_push(self):
        mem = _make_mem()
        va.write_cache(mem, "my-target", 42, "abc123", _att("attested"))
        # A force-push (new head SHA) invalidates the cache -> re-run.
        cached = va.read_cache(mem, "my-target", 42, "def456")
        assert cached is None

    def test_cache_shadow_tag(self):
        mem = _make_mem()
        va.write_cache(mem, "my-target", 42, "abc123", _att("attested"),
                       shadow=True)
        rec = mem._data[va.cache_key("my-target", 42)]
        assert "shadow" in rec["tags"]
        import json
        assert json.loads(rec["content"])["shadow"] is True


# ---------------------------------------------------------------------------
# U1.3 - containment invariant: the hook never consults the spec-prose path
# ---------------------------------------------------------------------------

class TestContainmentInvariant:

    def _run_hook(self, dial: str, att: dict):
        """Run the attestation hook with the given dial and attestation result.
        Returns (target, mock_attest)."""
        from lapis_pm import pm_core
        cls = _make_cls()
        # The payload carries the PR head ref (the hook derives the branch
        # slug from it); without a matching "lapis/<tid>/<slug>" ref the hook
        # skips the attestation entirely (never call attest with a wrong
        # slug).
        payload = {"pr": {"head": {"sha": "abc123",
                                   "ref": "lapis/my-target/local"}},
                   "classification": cls}
        target = _make_target(pm_verification="pm-live-test")
        with (
            patch.object(pm_core, "TargetStore") as mock_store,
            patch.object(pm_core, "_mem", return_value=_make_mem()),
            patch.object(va, "attest", return_value=att) as mock_attest,
            patch.object(va, "read_cache", return_value=None),
            patch.object(va, "write_cache"),
            patch("lapis_pm.episodic.write_observation"),
            patch.dict("os.environ", {"PM_AUTONOMY_ACTUATOR": dial}),
        ):
            mock_store.return_value.get.return_value = target
            pm_core._attestation_hook("my-target", cls, payload)
        return target, mock_attest

    def test_hook_enforce_attested_flips_from_observation(self):
        """The hook's grant source is the observed run (attest()), not the
        spec-prose path (--verification / spec field). Enforce + attested
        flips the field from the observation."""
        target, mock_attest = self._run_hook("enforce", _att("attested"))
        # attest() was consulted (the observation path).
        assert mock_attest.called
        # The grant came from the observed run: enforce + attested flips the field.
        assert target.data["pm_verification"] == "machine"
        target.save.assert_called()

    def test_hook_shadow_does_not_flip(self):
        target, _ = self._run_hook("shadow", _att("attested"))
        # Shadow mode: the field is NOT touched (no grant), the human brief is
        # intact.
        assert target.data["pm_verification"] == "pm-live-test"
        target.save.assert_not_called()

    def test_hook_unattested_never_flips(self):
        target, _ = self._run_hook("enforce", _att("unattested",
                                                   pr_failures=["tests/test_x.py::test_new"]))
        # unattested NEVER flips the field (fail-safe to the human brief).
        assert target.data["pm_verification"] == "pm-live-test"
        target.save.assert_not_called()

    def test_hook_inconclusive_never_flips(self):
        target, _ = self._run_hook("enforce", _att("inconclusive",
                                                   reason="runner_control_intersection"))
        # inconclusive NEVER flips the field (fail-safe to the human brief).
        assert target.data["pm_verification"] == "pm-live-test"
        target.save.assert_not_called()


# ---------------------------------------------------------------------------
# AC4 - containment pin intact (the file is held and unchanged)
# ---------------------------------------------------------------------------

class TestContainmentPinIntact:

    def test_bundle_bind_still_pins_pm_live_test(self):
        """A test asserting bundle_autodispatch._bind still passes
        verification='pm-live-test' to cmd_bind (the file is held and
        unchanged - this test pins that the fix does NOT route around the pin
        via the spec-prose path)."""
        import inspect
        from lapis_pm import bundle_autodispatch

        src = inspect.getsource(bundle_autodispatch._bind)
        assert 'verification="pm-live-test"' in src or "verification='pm-live-test'" in src
