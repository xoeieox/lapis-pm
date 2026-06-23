"""Regression tests for prod-store isolation.

AC1: _set_brief_outstanding with synthesis_failed=True writes to tmp stores only;
     prod /srv/lapis/targets/comments/my-target.jsonl and prod mem.db untouched.

AC2: The three named leaker tests run under the autouse isolation (verified by
     conftest _isolate_comment_store fixture; no additional assertions here).

AC3: cmd_unbind clears the outstanding brief key so get_outstanding_brief returns None.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROD_COMMENTS = Path("/srv/lapis/targets/comments/my-target.jsonl")


# ---------------------------------------------------------------------------
# AC1 — lost-brief failure path writes to tmp stores only
# ---------------------------------------------------------------------------


def test_set_brief_outstanding_synth_fail_writes_to_isolated_store(tmp_path):
    """_set_brief_outstanding with synthesis_failed=True must NOT touch prod stores.

    The conftest autouse _isolate_comment_store fixture redirects episodic._store
    to a per-test CommentStore(root=tmp_path). The MEM_DB_PATH env var (set at
    module load in conftest) redirects MemoryStore() to a tmp db. Together they
    prevent any write from reaching /srv/lapis/targets/comments/ or /data/memory/mem.db.
    """
    from lapis_pm import pm_core

    # Record prod state before the call.
    prod_existed_before = PROD_COMMENTS.exists()
    prod_mtime_before = PROD_COMMENTS.stat().st_mtime if prod_existed_before else None

    fake_brief = MagicMock()
    fake_brief.synthesis_failed = True   # truthy → triggers write_observation path

    # _set_brief_outstanding with fail_n < threshold (starts at 0) takes the
    # write_observation branch (pm_core.py:1431) and writes a pm:synthesis-failed
    # observation to the comment store.
    pm_core._set_brief_outstanding("my-target", fake_brief)

    # Prod comment file must be untouched.
    if prod_existed_before:
        assert PROD_COMMENTS.stat().st_mtime == prod_mtime_before, (
            "prod /srv/lapis/targets/comments/my-target.jsonl was modified — store isolation failed"
        )
    else:
        assert not PROD_COMMENTS.exists(), (
            "prod /srv/lapis/targets/comments/my-target.jsonl was created — store isolation failed"
        )

    # The tmp comment store (rooted at tmp_path by the autouse fixture) must have
    # received the observation.
    isolated_jsonl = tmp_path / "my-target.jsonl"
    assert isolated_jsonl.exists(), (
        "tmp comment store did not receive the observation — isolation fixture may be broken"
    )


# ---------------------------------------------------------------------------
# AC3 — cmd_unbind clears the outstanding brief key
# ---------------------------------------------------------------------------


def test_cmd_unbind_clears_outstanding_brief():
    """Unbinding a target must clear any live outstanding-brief key.

    Prior to the fix, cmd_unbind called clear_classified_prs but not
    clear_outstanding_brief, leaving pm/outstanding-brief/<tid> live in mem.db
    so /pm-pr-review kept re-surfacing the ghost.
    """
    from lapis_pm import pm_core
    from lapis_pm.cli import cmd_unbind

    tid = "ac3-unbind-test-target"

    # Inject an outstanding brief into the isolated (tmp) mem store.
    pm_core.set_outstanding_brief(tid, "cid-ac3-test")
    assert pm_core.get_outstanding_brief(tid) == "cid-ac3-test"

    # Mock the TargetStore so cmd_unbind doesn't need a real target YAML.
    mock_target = MagicMock()
    mock_store = MagicMock()
    mock_store.get.return_value = mock_target

    args = MagicMock()
    args.target_id = tid

    with (
        patch("lapis_pm.cli.TargetStore", return_value=mock_store),
        patch("lapis_pm.pm_core.clear_classified_prs"),
    ):
        rc = cmd_unbind(args)

    assert rc == 0
    assert pm_core.get_outstanding_brief(tid) is None, (
        "cmd_unbind did not clear the outstanding brief — ghost brief regression"
    )
