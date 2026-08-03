"""tests/conftest.py — spec-review GW leg worktree isolation.

_dispatch_gw_reviewer (lapis_pm.spec_review) pins a detached origin/main
worktree of /srv/git/<repo>-working before running the GW leg (see
lapis-pm-gw-leg-ground-pinned-main-v0). Real git worktree creation depends on
a live local clone existing on disk; most spec-review unit tests exercise the
function directly with fictional repo names ("repo", "r", ...) that have no
such clone, so an unmocked run would hit real `git worktree add` failures and
turn every one of those tests non-hermetic and dependent on host filesystem
state — exactly what the existing httpx-block fixture in the root conftest.py
already guards against for network calls.

This default fakes worktree creation/removal to a hermetic success so tests
that don't care about worktree mechanics keep exercising call_gw_agent as
before. Tests that DO care about worktree behavior (success/failure paths,
concurrency, cleanup-on-failure) monkeypatch
lapis_pm.spec_review._create_gw_worktree / _remove_gw_worktree themselves
within the test body — that patch, applied after this fixture runs, wins.
"""
from __future__ import annotations

import sys
import uuid

import pytest


@pytest.fixture(autouse=True)
def _stub_gw_leg_worktree_by_default(monkeypatch):
    if "lapis_pm.spec_review" not in sys.modules:
        yield
        return

    def _fake_create(repo, run_id):
        return f"/tmp/fake-gw-worktree-{run_id}", f"fake{uuid.uuid4().hex[:8]}", ""

    def _fake_remove(repo, worktree_path):
        return None

    monkeypatch.setattr("lapis_pm.spec_review._create_gw_worktree", _fake_create, raising=False)
    monkeypatch.setattr("lapis_pm.spec_review._remove_gw_worktree", _fake_remove, raising=False)
    yield
