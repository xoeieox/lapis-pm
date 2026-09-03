"""Tests for the open-PR clobber guard in _ensure_head_branch_deleted.

Incident (finding/lapis-pm-branch-reconcile-clobbers-open-pr-same-branch-
2026-08-31): every fixer for a target pushes to the SAME branch name
lapis/<target_id>/local, so when PR-A merges while PR-B is in flight on that
branch, the per-tick head-branch reconcile (targeting the merged PR-A) deleted
the shared branch and Forgejo auto-closed PR-B, silently destroying the work.

The durable fix probes open PRs before the explicit DELETE and refuses to
delete a branch an open PR still holds. The probe is best-effort: a probe
failure logs a warning and falls through to the DELETE, exactly as today's
behavior on a partial Forgejo outage.

Coverage:
  AC1 — an open PR on the branch -> the DELETE is skipped and a warning
        naming the holding PR number(s) and the branch is logged.
  AC2 — no open PR on the branch -> the explicit DELETE proceeds
        (today's behavior, unchanged).
  AC3 — the open-PR probe raises -> the function does not raise and the
        DELETE proceeds (today's behavior on a partial Forgejo outage).
  AC4 — the probe surface is unavailable (import fallback None) -> the
        DELETE proceeds (today's behavior, unchanged).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from lapis_pm import pm_core

# The shared-branch shape from the incident: one branch, two PRs.
REF = "lapis/tgt/local"


def _open_pr(number: int, ref: str) -> dict:
    return {"number": number, "head": {"ref": ref}}


def _fake_delete_factory(calls: list):
    def fake_delete(url, **kwargs):
        calls.append(url)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    return fake_delete


# ---------------------------------------------------------------------------
# AC1: an open PR holds the branch -> DELETE skipped, warning logged
# ---------------------------------------------------------------------------

class TestOpenPRHoldsBranch:
    """The guard refuses to delete a branch an open PR still uses."""

    def test_delete_skipped_when_open_pr_holds_branch(self, caplog):
        delete_calls = []
        open_prs = [
            _open_pr(55, REF),
            _open_pr(56, "lapis/other-target/local"),  # different branch: must NOT hold
        ]

        caplog.set_level(logging.WARNING, logger="lapis_pm.pm_core")
        with (
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"head": {"ref": REF}}),
            patch("lapis_pm.pm_core._forgejo_get_branch", return_value={}),
            patch("lapis_pm.pm_core.get_open_prs", return_value=open_prs) as mock_prs,
            patch.dict("os.environ", {"FORGEJO_TOKEN": "fake-token"}),
        ):
            import httpx
            with patch.object(httpx, "delete", side_effect=_fake_delete_factory(delete_calls)):
                pm_core._ensure_head_branch_deleted("lapis-pm", 77, owner="Erah")

        assert delete_calls == [], (
            "DELETE must not be issued while an open PR holds the branch"
        )
        # owner passed the same way the function's other forgejo calls pass it
        mock_prs.assert_called_once_with("lapis-pm", owner="Erah")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(REF in r.message and "55" in r.message for r in warnings), (
            "warning must name the holding PR number and the branch; got:\n"
            + "\n".join(r.message for r in warnings)
        )
        assert not any("56" in r.message for r in warnings), (
            "a PR on a DIFFERENT branch must not hold the delete"
        )

    def test_multiple_holding_prs_all_named(self, caplog):
        delete_calls = []
        open_prs = [_open_pr(55, REF), _open_pr(71, REF)]

        caplog.set_level(logging.WARNING, logger="lapis_pm.pm_core")
        with (
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"head": {"ref": REF}}),
            patch("lapis_pm.pm_core._forgejo_get_branch", return_value={}),
            patch("lapis_pm.pm_core.get_open_prs", return_value=open_prs),
            patch.dict("os.environ", {"FORGEJO_TOKEN": "fake-token"}),
        ):
            import httpx
            with patch.object(httpx, "delete", side_effect=_fake_delete_factory(delete_calls)):
                pm_core._ensure_head_branch_deleted("lapis-pm", 77, owner="Erah")

        assert delete_calls == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("55" in r.message and "71" in r.message for r in warnings), (
            "both holding PR numbers must be named in the warning"
        )


# ---------------------------------------------------------------------------
# AC2: no open PR on the branch -> DELETE proceeds (today's behavior)
# ---------------------------------------------------------------------------

class TestNoOpenPRonBranch:
    """The guard is transparent when nothing holds the branch."""

    def test_delete_proceeds_with_no_open_prs(self):
        delete_calls = []

        with (
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"head": {"ref": REF}}),
            patch("lapis_pm.pm_core._forgejo_get_branch", return_value={}),
            patch("lapis_pm.pm_core.get_open_prs", return_value=[]),
            patch.dict("os.environ", {"FORGEJO_TOKEN": "fake-token"}),
        ):
            import httpx
            with patch.object(httpx, "delete", side_effect=_fake_delete_factory(delete_calls)):
                pm_core._ensure_head_branch_deleted("lapis-pm", 77, owner="Erah")

        assert len(delete_calls) == 1
        assert REF in delete_calls[0]

    def test_delete_proceeds_when_open_prs_use_other_branches(self):
        delete_calls = []
        open_prs = [
            _open_pr(56, "lapis/other-target/local"),
            _open_pr(57, "feat/direct-human-work"),
        ]

        with (
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"head": {"ref": REF}}),
            patch("lapis_pm.pm_core._forgejo_get_branch", return_value={}),
            patch("lapis_pm.pm_core.get_open_prs", return_value=open_prs),
            patch.dict("os.environ", {"FORGEJO_TOKEN": "fake-token"}),
        ):
            import httpx
            with patch.object(httpx, "delete", side_effect=_fake_delete_factory(delete_calls)):
                pm_core._ensure_head_branch_deleted("lapis-pm", 77, owner="Erah")

        assert len(delete_calls) == 1
        assert REF in delete_calls[0]


# ---------------------------------------------------------------------------
# AC3: the open-PR probe raises -> non-raising, DELETE proceeds
# ---------------------------------------------------------------------------

class TestProbeFailure:
    """A probe failure must not change today's behavior on a partial outage."""

    def test_probe_exception_does_not_raise_and_delete_proceeds(self, caplog):
        delete_calls = []

        caplog.set_level(logging.WARNING, logger="lapis_pm.pm_core")
        with (
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"head": {"ref": REF}}),
            patch("lapis_pm.pm_core._forgejo_get_branch", return_value={}),
            patch("lapis_pm.pm_core.get_open_prs",
                  side_effect=Exception("forgejo 500")),
            patch.dict("os.environ", {"FORGEJO_TOKEN": "fake-token"}),
        ):
            import httpx
            with patch.object(httpx, "delete", side_effect=_fake_delete_factory(delete_calls)):
                pm_core._ensure_head_branch_deleted("lapis-pm", 77, owner="Erah")
                # must not raise

        assert len(delete_calls) == 1, (
            "a failed open-PR probe must not change the DELETE outcome"
        )
        assert REF in delete_calls[0]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("open-PR probe" in r.message for r in warnings), (
            "the probe failure must be logged as a warning; got:\n"
            + "\n".join(r.message for r in warnings)
        )


# ---------------------------------------------------------------------------
# AC4: probe surface unavailable (import fallback None) -> today's behavior
# ---------------------------------------------------------------------------

class TestProbeSurfaceUnavailable:
    """If the get_open_prs import fell back to None, the guard is a no-op."""

    def test_probe_unavailable_delete_proceeds(self):
        delete_calls = []

        with (
            patch("lapis_pm.pm_core._forgejo_get_pr",
                  return_value={"head": {"ref": REF}}),
            patch("lapis_pm.pm_core._forgejo_get_branch", return_value={}),
            patch("lapis_pm.pm_core.get_open_prs", None),
            patch.dict("os.environ", {"FORGEJO_TOKEN": "fake-token"}),
        ):
            import httpx
            with patch.object(httpx, "delete", side_effect=_fake_delete_factory(delete_calls)):
                pm_core._ensure_head_branch_deleted("lapis-pm", 77, owner="Erah")
                # must not raise

        assert len(delete_calls) == 1
        assert REF in delete_calls[0]
