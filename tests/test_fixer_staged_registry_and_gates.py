"""Tests for the fixer_staged agent_type — registry entry + the two gate
extensions (lapis-pm-fixers-harness-registry-v0, Leg 2).

Coverage:
  - Registry shape: fixer_staged is modeled on fixer_retry (engine
    local-fixer-staged, model gravitywell-slot1, capture_meta/notify off,
    timeout_s 2700) and its system_template carries {steer_directive_block}.
  - Gate 1 (steer.py): the single-shot directive overlay is consumed for a
    fixer_staged target (pinned in tests/test_steer.py; re-asserted here for
    the registry/template side).
  - Gate 2 (pm_core.py): the L1.D2 open-PR scan fires for fixer_staged and
    resolves the parked PR's ref into vars_["existing_branch"].
  - Gate 2 negative (correctness F3): the open-PR ValueError guard does NOT
    fire for fixer_staged — it is a BARE scan-tuple entry, NOT a member of
    _INITIAL_FIXER_TYPES (which would block the staged dispatch against a
    parked PR).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lapis_pm import pm_core


REGISTRY_PATH = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"


def _load_agents() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text())["agents"]


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

class TestRegistryFixerStaged:
    """fixer_staged is modeled on fixer_retry (Leg 2)."""

    @pytest.fixture(scope="class")
    def entry(self):
        return _load_agents()["fixer_staged"]

    def test_engine(self, entry):
        assert entry["engine"] == "local-fixer-staged"

    def test_model(self, entry):
        assert entry["model"] == "gravitywell-slot1"

    def test_capture_meta_false(self, entry):
        assert entry["capture_meta"] is False

    def test_notify_false(self, entry):
        assert entry["notify"] is False

    def test_timeout_2700(self, entry):
        assert entry["timeout_s"] == 2700

    def test_system_template_carries_steer_directive_block(self, entry):
        # The mission fence rides the single-shot steer directive channel.
        assert "{steer_directive_block}" in entry["system_template"]

    def test_system_template_renders_without_overlay(self, entry):
        # inject_overlay always sets the key; the template must not KeyError.
        from lapis_pm import steer
        mem_mock = MagicMock()
        mem_mock.get.return_value = None
        mem_mock.delete.return_value = True
        vars_ = {
            "target_id": "staged-target-v0",
            "spec_summary": "a spec",
            "repo": "lapis-pm",
            "question": "implement X",
            "pr_number": "",
            "slug": "forced",
            "existing_branch": "lapis/staged-target-v0/forced",
            "base_branch": "main",
            "intent_block": "## Intent\nDo the thing.",
        }
        with patch("lapis_pm.steer._mem", return_value=mem_mock), \
             patch("lapis_pm.episodic.write_observation"):
            steer.inject_overlay("staged-target-v0", vars_, "fixer_staged")
        rendered = entry["system_template"].format(**vars_)
        assert "staged-target-v0" in rendered


# ---------------------------------------------------------------------------
# Gate 2: the L1.D2 open-PR scan fires for fixer_staged
# ---------------------------------------------------------------------------

def _make_target(pm_repo):
    t = MagicMock()
    t.pm_repo = pm_repo
    t.data = {}
    return t


def _live_queue(gpu_id):
    """A queue double proving the (absent) pending records are not live, so
    the L1.D3 reconcile pass never re-raises."""
    cq = MagicMock()
    cq.get_recent_failed.return_value = []
    cq.get_recent_completed.return_value = []
    cq.get_pending.return_value = []
    cq.get_active.return_value = []
    return cq


class TestFixerStagedOpenPrScan:
    """fixer_staged must resolve the parked PR's ref via the L1.D2 scan so
    the staged engine checks out the PR head (not main)."""

    def test_scan_resolves_parked_pr_ref(self):
        target = _make_target("lapis/coderag")
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = target
            with patch("lapis_pm.pm_core.episodic.spec_summary", return_value="stub"), \
                 patch("lapis_pm.pm_core.load_dispatched", return_value=[]), \
                 patch.object(pm_core, "_ClaudeQueue", return_value=_live_queue("gpu-x")), \
                 patch("lapis_pm.pm_core._SHAPER.dispatch") as shaper_m, \
                 patch("lapis_pm.pm_core._mem") as mem_m, \
                 patch("lapis_pm.pm_core.episodic.write_dispatch"), \
                 patch("lapis_pm.pm_core._check_calcification"):
                res = MagicMock()
                res.task_id = "task-staged"
                res.spec_id = "spec-staged"
                shaper_m.return_value = res
                mem_m.return_value = MagicMock()
                open_prs = [
                    {
                        "number": 913,
                        "head": {"ref": "lapis/agora-town-hall-v0/d2a"},
                        "base": {"ref": "main"},
                    }
                ]
                with patch("agents_core.forgejo.get_open_prs", return_value=open_prs) as mock_scan:
                    pm_core.force_dispatch("agora-town-hall-v0", "fixer_staged", "mission")
                # The scan fired for fixer_staged.
                assert mock_scan.called
                # The staged dispatch resolved the PR ref (not the /forced default).
                call_kwargs = shaper_m.call_args.kwargs
                assert call_kwargs["vars_"]["existing_branch"] == "lapis/agora-town-hall-v0/d2a"
                assert call_kwargs["vars_"]["pr_number"] == "913"


# ---------------------------------------------------------------------------
# Gate 2 negative: the open-PR ValueError guard does NOT fire for fixer_staged
# ---------------------------------------------------------------------------

class TestFixerStagedNotInitialFixerGuard:
    """correctness F3: fixer_staged is a BARE scan-tuple entry, NOT a member
    of _INITIAL_FIXER_TYPES. The open-PR ValueError guard (and the adopted-PR
    guard) must NOT fire for fixer_staged — extending _INITIAL_FIXER_TYPES
    would block the acceptance dispatch against parked PR #913."""

    def test_not_a_member_of_initial_fixer_types(self):
        assert "fixer_staged" not in pm_core._INITIAL_FIXER_TYPES

    def test_open_pr_guard_does_not_fire_for_fixer_staged(self):
        """A parked open PR on the canonical branch must NOT raise for
        fixer_staged (it raises for a plain `fixer` — the contrast that
        proves the guard is scoped to _INITIAL_FIXER_TYPES)."""
        target = _make_target("lapis/coderag")
        open_prs = [
            {
                "number": 913,
                "head": {"ref": "lapis/agora-town-hall-v0/d2a"},
                "base": {"ref": "main"},
            }
        ]
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = target
            with patch("lapis_pm.pm_core.episodic.spec_summary", return_value="stub"), \
                 patch("lapis_pm.pm_core.load_dispatched", return_value=[]), \
                 patch.object(pm_core, "_ClaudeQueue", return_value=_live_queue("gpu-x")), \
                 patch("lapis_pm.pm_core._SHAPER.dispatch") as shaper_m, \
                 patch("lapis_pm.pm_core._mem") as mem_m, \
                 patch("lapis_pm.pm_core.episodic.write_dispatch"), \
                 patch("lapis_pm.pm_core._check_calcification"):
                res = MagicMock()
                res.task_id = "task-staged"
                res.spec_id = "spec-staged"
                shaper_m.return_value = res
                mem_m.return_value = MagicMock()
                with patch("agents_core.forgejo.get_open_prs", return_value=open_prs):
                    # No ValueError: the guard is scoped to _INITIAL_FIXER_TYPES.
                    task_id = pm_core.force_dispatch("agora-town-hall-v0", "fixer_staged", "mission")
        assert task_id == "task-staged"

    def test_open_pr_guard_still_fires_for_plain_fixer(self):
        """Contrast pin: the SAME parked PR raises for a plain `fixer`
        (initial fixer), confirming the guard still works where it should."""
        target = _make_target("lapis/coderag")
        open_prs = [
            {
                "number": 913,
                "head": {"ref": "lapis/agora-town-hall-v0/d2a"},
                "base": {"ref": "main"},
            }
        ]
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = target
            with patch("lapis_pm.pm_core.episodic.spec_summary", return_value="stub"), \
                 patch("lapis_pm.pm_core.load_dispatched", return_value=[]), \
                 patch.object(pm_core, "_ClaudeQueue", return_value=_live_queue("gpu-x")), \
                 patch("agents_core.forgejo.get_open_prs", return_value=open_prs):
                with pytest.raises(ValueError, match="already has open PR #913"):
                    pm_core.force_dispatch("agora-town-hall-v0", "fixer", "intent")
