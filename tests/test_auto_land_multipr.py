"""Tests for the multi-PR auto-land guard (lapis-pm-auto-land-multipr-guard spec).

Coverage:
  - _detect_pr_count_from_spec: 0 / 1 / 3 headers
  - _is_auto_land_eligible: pr_count=2 with 1 merged PR → False (suppressed)
  - _is_auto_land_eligible: pr_count=2 with 2 merged PRs (and other guards pass) → True
  - _is_auto_land_eligible: pr_count=1 default → behaves identically to current logic
  - pm:auto-land:waiting comment emitted once per N; de-duped on second call
  - --pr-count flag on bind: persisted to target YAML
  - --create without --pr-count: auto-detect from spec headers
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from lapis_pm import pm_core
from lapis_pm.pm_core import _detect_pr_count_from_spec, _is_auto_land_eligible


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_comment(tags: list[str], content: str = "") -> MagicMock:
    c = MagicMock()
    c.tags = tags
    c.content = content
    return c


def _target_with_pr_count(pr_count: int) -> MagicMock:
    """Return a mock Target with the given pr_count and pm_repo set."""
    t = MagicMock()
    t.data = {"pr_count": pr_count, "pm_repo": "lapis-test"}
    t.pm_repo = "lapis-test"
    return t


def _make_mem(landed: bool = False, dispatched: list | None = None):
    """Return a mock MemoryStore.

    landed=True → pm/landed/<tid> exists.
    dispatched → list of dispatch records (empty by default).
    """
    import json
    mem = MagicMock()

    def _get(key):
        if "landed" in key and landed:
            return {"content": "{}"}
        if "dispatched" in key:
            payload = dispatched if dispatched is not None else []
            return {"content": json.dumps(payload)}
        return None

    mem.get.side_effect = _get
    return mem


# Build synthetic episodic comment lists for a target that has seen + merged pr_num(s).
def _comments_for_prs(seen: list[int], merged: list[int], waiting_count: int | None = None) -> list:
    """Build comment list with pm:pr=N (seen) and pm:pr-merged:N (merged) tags.

    waiting_count: if set, also include an existing pm:auto-land:waiting:count=<N> comment.
    """
    comments = []
    for pr in seen:
        comments.append(_make_comment([f"pm:pr={pr}", "pm:observation"]))
    for pr in merged:
        comments.append(_make_comment([f"pm:pr-merged:{pr}", f"pm:pr={pr}", "pm:observation"]))
    if waiting_count is not None:
        comments.append(_make_comment(
            ["pm:auto-land:waiting", f"pm:auto-land:waiting:count={waiting_count}"],
            content=f"auto-land deferred: 1/{waiting_count} PRs merged on lapis/test/*",
        ))
    return comments


# ---------------------------------------------------------------------------
# _detect_pr_count_from_spec
# ---------------------------------------------------------------------------

class TestDetectPrCountFromSpec:
    def test_no_headers_returns_1(self):
        spec = "# Title\n\nSome content with no PR sections."
        assert _detect_pr_count_from_spec(spec) == 1

    def test_one_header_returns_1(self):
        spec = textwrap.dedent("""\
            # Spec

            ### PR 1
            First PR details.
        """)
        assert _detect_pr_count_from_spec(spec) == 1

    def test_three_headers_returns_3(self):
        spec = textwrap.dedent("""\
            # Spec

            ### PR 1
            First.

            ### PR 2 — Hook integration
            Second.

            ### PR 3
            Third.
        """)
        assert _detect_pr_count_from_spec(spec) == 3

    def test_level2_header_not_counted(self):
        """## PR 1 (level-2) must NOT match."""
        spec = "## PR 1\n## PR 2\n### PR 3"
        assert _detect_pr_count_from_spec(spec) == 1

    def test_decimal_not_counted(self):
        """### PR 1.5 is not a valid match (decimal after digits excluded by lookahead)."""
        # "### PR 1.5" should not match because '.' after the digit is not \s or $
        # "### PR 1" on the second line should match
        spec = "### PR 1.5\n### PR 1"
        assert _detect_pr_count_from_spec(spec) == 1

    def test_empty_spec_returns_1(self):
        assert _detect_pr_count_from_spec("") == 1


# ---------------------------------------------------------------------------
# _is_auto_land_eligible — multi-PR guard
# ---------------------------------------------------------------------------

class TestIsAutoLandEligibleMultiPr:
    """Unit tests for the pr_count guard in _is_auto_land_eligible."""

    def _run(
        self,
        pr_count: int,
        seen: list[int],
        merged: list[int],
        waiting_count: int | None = None,
        write_obs_mock: MagicMock | None = None,
    ) -> bool:
        """Call _is_auto_land_eligible with controlled state."""
        comments = _comments_for_prs(seen, merged, waiting_count)
        mem = _make_mem(landed=False, dispatched=[])
        target = _target_with_pr_count(pr_count)

        write_obs = write_obs_mock or MagicMock()

        with (
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
            patch("lapis_pm.pm_core.TargetStore") as mock_ts,
            patch("lapis_pm.pm_core.episodic.write_observation", write_obs),
            # Disable Forgejo branch-deletion check
            patch("lapis_pm.pm_core._forgejo_get_pr", None),
            patch("lapis_pm.pm_core._forgejo_get_branch", None),
        ):
            mock_ts.return_value.get.return_value = target
            return _is_auto_land_eligible("test-target")

    def test_partial_merge_suppresses_land(self):
        """pr_count=3, 2 of 3 PRs merged → False.

        seen=[1,2] merged=[1,2] means max(seen)==max(merged), so the existing
        open-PR guard passes. The pr_count guard then fires (2 < 3).
        """
        result = self._run(pr_count=3, seen=[1, 2], merged=[1, 2])
        assert result is False

    def test_full_merge_allows_land(self):
        """pr_count=3, all 3 PRs merged → True (branch-deletion check skipped)."""
        result = self._run(pr_count=3, seen=[1, 2, 3], merged=[1, 2, 3])
        assert result is True

    def test_single_pr_default_allows_land(self):
        """pr_count=1 (default), one merged PR → True (regression guard)."""
        result = self._run(pr_count=1, seen=[1], merged=[1])
        assert result is True

    def test_single_pr_default_no_merge_suppresses(self):
        """pr_count=1, no merged PR → False (existing behavior unchanged)."""
        result = self._run(pr_count=1, seen=[1], merged=[])
        assert result is False

    def test_waiting_comment_emitted_when_suppressed(self):
        """When suppressed by pr_count guard, write_observation is called once."""
        write_obs = MagicMock()
        # seen=[1,2] merged=[1,2] → max check passes; pr_count=3 guard fires (2<3)
        result = self._run(pr_count=3, seen=[1, 2], merged=[1, 2], write_obs_mock=write_obs)
        assert result is False
        write_obs.assert_called_once()
        args, kwargs = write_obs.call_args
        assert "auto-land deferred" in args[1]
        assert "2/3" in args[1]
        extra_tags = kwargs.get("extra_tags", [])
        assert "pm:auto-land:waiting" in extra_tags
        assert "pm:auto-land:waiting:count=3" in extra_tags

    def test_waiting_comment_deduped(self):
        """If pm:auto-land:waiting:count=N already in comments, do NOT emit again."""
        write_obs = MagicMock()
        # seen=[1,2] merged=[1,2] → max check passes; pr_count guard fires (2<3)
        # waiting_count=3 means a prior waiting comment for N=3 already exists
        result = self._run(
            pr_count=3, seen=[1, 2], merged=[1, 2],
            waiting_count=3, write_obs_mock=write_obs,
        )
        assert result is False
        write_obs.assert_not_called()

    def test_waiting_comment_not_emitted_when_eligible(self):
        """When all PRs merged and eligible, no waiting comment is written."""
        write_obs = MagicMock()
        result = self._run(pr_count=2, seen=[1, 2], merged=[1, 2], write_obs_mock=write_obs)
        assert result is True
        write_obs.assert_not_called()


# ---------------------------------------------------------------------------
# CLI: --pr-count flag on bind
# ---------------------------------------------------------------------------

class TestBindPrCount:
    """Verify --pr-count is persisted to target YAML on bind."""

    SPEC_BODY = textwrap.dedent("""\
        # Test spec

        A minimal spec body for unit testing.
    """)

    def _write_spec(self, tmp_path: Path) -> Path:
        p = tmp_path / "spec.md"
        p.write_text(self.SPEC_BODY)
        return p

    def _run(self, argv: list[str], targets_dir: Path) -> tuple[int, str, str]:
        import io
        from contextlib import redirect_stdout, redirect_stderr

        from agents_core.targets import TargetStore
        from lapis_pm.cli import main

        out_buf = io.StringIO()
        err_buf = io.StringIO()

        with (
            patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
            patch("lapis_pm.cli.episodic.spec", return_value=None),
            patch("lapis_pm.cli.episodic.write_spec", return_value=MagicMock()),
            patch("lapis_pm.cli.pm_core.clear_classified_prs", return_value=None),
            redirect_stdout(out_buf),
            redirect_stderr(err_buf),
        ):
            rc = main(argv)

        return rc, out_buf.getvalue(), err_buf.getvalue()

    def test_explicit_pr_count_written_to_yaml(self, tmp_path):
        """--pr-count 3 is persisted to target YAML."""
        import yaml
        spec_path = self._write_spec(tmp_path)

        rc, out, err = self._run(
            [
                "bind", "multipr-target",
                "--spec-from", str(spec_path),
                "--repo", "lapis-pm",
                "--authority", "advisory",
                "--create",
                "--title", "Multi-PR Target",
                "--pr-count", "3",
            ],
            targets_dir=tmp_path,
        )

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "multipr-target.yaml").read_text())
        assert data["pr_count"] == 3

    def test_auto_detect_from_spec_headers(self, tmp_path):
        """--create without --pr-count: auto-detect from ### PR N headers."""
        import yaml

        spec_with_prs = textwrap.dedent("""\
            # Spec

            ### PR 1
            First PR.

            ### PR 2
            Second PR.
        """)
        spec_path = tmp_path / "spec.md"
        spec_path.write_text(spec_with_prs)

        rc, out, err = self._run(
            [
                "bind", "autodetect-target",
                "--spec-from", str(spec_path),
                "--repo", "lapis-pm",
                "--authority", "advisory",
                "--create",
                "--title", "Auto-detect Target",
                # no --pr-count
            ],
            targets_dir=tmp_path,
        )

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "autodetect-target.yaml").read_text())
        assert data["pr_count"] == 2

    def test_auto_detect_no_headers_defaults_to_1(self, tmp_path):
        """--create without --pr-count and no headers → pr_count=1."""
        import yaml
        spec_path = self._write_spec(tmp_path)

        rc, out, err = self._run(
            [
                "bind", "no-headers-target",
                "--spec-from", str(spec_path),
                "--repo", "lapis-pm",
                "--authority", "advisory",
                "--create",
                "--title", "No Headers Target",
            ],
            targets_dir=tmp_path,
        )

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "no-headers-target.yaml").read_text())
        assert data.get("pr_count", 1) == 1
