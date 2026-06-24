"""Tests for fixer_local agent_type — registry entry + initial-fixer guard extension.

Coverage (spec: local-fixer-registry-entry-v0):

  AC1 — Raw YAML parse: fixer_local has engine=local-fixer, model=gravitywell-122b,
         and system_template contains none of the forbidden git/PR verbs.

  AC2 — force_dispatch guard scenarios:
         - fixer_local blocked by pending fixer_local (L1.D3 inner fix — was hard-coded)
         - fixer_local blocked by pending fixer (L1.D3 cross-block)
         - fixer blocked by pending fixer_local (L1.D3 outer + inner)
         - fixer_local blocked by pending fixer_retry (L1.D3 still includes fixer_retry)
         - Adopted-PR target: fixer_local with no prior history raises
         - Adopted-PR target: fixer_local with prior fixer_local history proceeds
         - L1.D1 open-PR scan fires for fixer_local (same as fixer)

  AC3 — Lost-fixer detection:
         - fixer_local with no resulting PR is found by _find_lost_fixer_dispatches
         - fixer_local with live pending retry child is NOT preempted

  AC4 — Regression: fixer and fixer_retry behavior unchanged.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers shared by AC2/AC3/AC4
# ---------------------------------------------------------------------------

def _make_target(adopted_pr_number=None, adopted_head_branch=None, pm_repo=None):
    t = MagicMock()
    t.pm_repo = pm_repo
    t.data = {}
    if adopted_pr_number is not None:
        t.data["adopted_pr_number"] = adopted_pr_number
    if adopted_head_branch is not None:
        t.data["adopted_head_branch"] = adopted_head_branch
    return t


def _dispatch_record(gpu_id, agent_type, status="pending"):
    return {"gpu_id": gpu_id, "agent_type": agent_type, "status": status}


def _fixer_local_rec(gpu_id="gpu-001", status="failed", parent_gpu_id=None):
    rec = {
        "gpu_id": gpu_id,
        "spec_id": f"spec-{gpu_id}",
        "agent_type": "fixer_local",
        "intent": "fix the spec",
        "repo": "lapis-pm",
        "ts": "2026-06-23T22:00:00-07:00",
        "status": status,
        "retry_count": 0,
        "lost_retry_count": 0,
    }
    if parent_gpu_id is not None:
        rec["parent_gpu_id"] = parent_gpu_id
    return rec


# ---------------------------------------------------------------------------
# AC1: raw YAML parse
# ---------------------------------------------------------------------------

class TestRegistryYamlFixerLocal:

    @pytest.fixture(scope="class")
    def entry(self):
        registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
        data = yaml.safe_load(registry_path.read_text())
        return data["agents"]["fixer_local"]

    def test_engine_field(self, entry):
        assert entry.get("engine") == "local-fixer"

    def test_model_field(self, entry):
        assert entry.get("model") == "gravitywell-122b"

    def test_system_template_no_forbidden_verbs(self, entry):
        template = entry.get("system_template", "")
        forbidden = ("git", "commit", "push", "branch", "create_pr", " PR")
        for verb in forbidden:
            assert verb not in template, (
                f"fixer_local system_template must not contain {verb!r} "
                f"(model has no git tools; naming them causes hallucination)"
            )

    def test_capture_meta_true(self, entry):
        assert entry.get("capture_meta") is True

    def test_notify_infra_only(self, entry):
        assert entry.get("notify_policy") == "infra-only"

    def test_timeout_above_fixer(self, entry):
        # fixer is 2700; fixer_local needs headroom for the 122B lane
        assert entry.get("timeout_s", 0) > 2700


# ---------------------------------------------------------------------------
# AC2: force_dispatch guards treat fixer_local as an initial fixer
# ---------------------------------------------------------------------------

@pytest.fixture
def target_no_repo():
    with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
        ts_mock.return_value.get.return_value = _make_target()
        yield


@pytest.fixture
def episodic_stub():
    with patch("lapis_pm.pm_core.episodic.spec_summary", return_value="stub"):
        yield


@pytest.fixture
def shaper_stub():
    with patch("lapis_pm.pm_core._SHAPER.dispatch") as m:
        result = MagicMock()
        result.task_id = "task-stub"
        result.spec_id = "spec-stub"
        m.return_value = result
        yield m


class TestL1D3FixerLocalCrossBlock:
    """L1.D3 inner check now uses _INITIAL_FIXER_TYPES + (fixer_retry,)."""

    def test_fixer_local_blocked_by_pending_fixer_local(self, episodic_stub):
        """fixer_local must not fire when another fixer_local is pending (self-block)."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target()
            pending = _dispatch_record("gpu-A", "fixer_local", "pending")
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[pending]):
                with pytest.raises(ValueError, match="pending fixer_local"):
                    pm_core.force_dispatch("tgt", "fixer_local", "intent")

    def test_fixer_local_blocked_by_pending_fixer(self, episodic_stub):
        """fixer_local must not fire when a fixer is pending (cross-block)."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target()
            pending = _dispatch_record("gpu-B", "fixer", "pending")
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[pending]):
                with pytest.raises(ValueError, match="pending fixer"):
                    pm_core.force_dispatch("tgt", "fixer_local", "intent")

    def test_fixer_blocked_by_pending_fixer_local(self, episodic_stub):
        """fixer must not fire when a fixer_local is pending (cross-block reverse)."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target()
            pending = _dispatch_record("gpu-C", "fixer_local", "pending")
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[pending]):
                with pytest.raises(ValueError, match="pending fixer_local"):
                    pm_core.force_dispatch("tgt", "fixer", "intent")

    def test_fixer_local_blocked_by_pending_fixer_retry(self, episodic_stub):
        """fixer_local must not fire when a fixer_retry is pending."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target()
            pending = _dispatch_record("gpu-D", "fixer_retry", "pending")
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[pending]):
                with pytest.raises(ValueError, match="pending fixer_retry"):
                    pm_core.force_dispatch("tgt", "fixer_local", "intent")

    def test_fixer_local_proceeds_when_only_completed_records(
        self, episodic_stub, shaper_stub
    ):
        """fixer_local fires when prior records are all completed (not pending)."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target()
            completed = _dispatch_record("gpu-E", "fixer_local", "completed")
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[completed]):
                task_id = pm_core.force_dispatch("tgt", "fixer_local", "intent")
                assert task_id == "task-stub"


class TestAdoptedPRGuardFixerLocal:
    """Adopted-PR guard outer + inner history check extend to fixer_local."""

    def test_fixer_local_raises_on_adopted_pr_no_history(self, episodic_stub):
        """fixer_local raises for adopted target with no prior fixer history."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target(adopted_pr_number=77)
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[]):
                with pytest.raises(ValueError, match="has an adopted PR"):
                    pm_core.force_dispatch("tgt", "fixer_local", "intent")

    def test_fixer_local_proceeds_on_adopted_pr_with_prior_fixer_local_history(
        self, episodic_stub, shaper_stub
    ):
        """fixer_local proceeds when a prior fixer_local is in history (inner check)."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target(adopted_pr_number=77)
            prior = _dispatch_record("gpu-F", "fixer_local", "completed")
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[prior]):
                task_id = pm_core.force_dispatch("tgt", "fixer_local", "intent")
                assert task_id == "task-stub"

    def test_fixer_local_proceeds_on_adopted_pr_with_prior_fixer_history(
        self, episodic_stub, shaper_stub
    ):
        """fixer_local proceeds when a prior fixer (not fixer_local) is in history."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target(adopted_pr_number=77)
            prior = _dispatch_record("gpu-G", "fixer", "completed")
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[prior]):
                task_id = pm_core.force_dispatch("tgt", "fixer_local", "intent")
                assert task_id == "task-stub"


class TestL1D1OpenPRScanFixerLocal:
    """L1.D1 open-PR guard fires for fixer_local as well as fixer."""

    def test_fixer_local_raises_when_open_pr_exists(self, episodic_stub):
        """fixer_local raises when an open lapis/<tid>/ PR already exists."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target(pm_repo="lapis-pm")
            open_prs = [{"number": 55, "head": {"ref": "lapis/tgt/forced"}}]
            with patch("agents_core.forgejo.get_open_prs", return_value=open_prs):
                with pytest.raises(ValueError, match="already has open PR #55"):
                    pm_core.force_dispatch("tgt", "fixer_local", "intent")


# ---------------------------------------------------------------------------
# AC3: _find_lost_fixer_dispatches includes fixer_local
# ---------------------------------------------------------------------------

class TestLostFixerDetectionFixerLocal:
    """fixer_local dispatches that produce no PR are detected as lost."""

    def test_fixer_local_no_pr_classified_as_lost(self):
        """A failed fixer_local with no PR is included in needs_retry."""
        rec = _fixer_local_rec(status="failed")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "tgt", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1
        assert retry[0]["gpu_id"] == rec["gpu_id"]
        assert brief == []

    def test_fixer_local_processed_no_pr_classified_as_lost(self):
        """A processed fixer_local with no PR (no evidence) is included in needs_retry."""
        rec = _fixer_local_rec(status="processed")
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "tgt", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1

    def test_fixer_local_with_pending_retry_child_not_preempted(self):
        """A fixer_local with a live (pending) fixer_local retry child is skipped."""
        orig = _fixer_local_rec(gpu_id="gpu-orig", status="failed")
        child = _fixer_local_rec(
            gpu_id="gpu-child", status="pending", parent_gpu_id="gpu-orig"
        )
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "tgt", [orig, child], open_prs=[], forgejo_ok=True
            )
        assert retry == [], "live retry child must prevent preemption"
        assert brief == []


# ---------------------------------------------------------------------------
# AC4: Regression — fixer and fixer_retry unchanged
# ---------------------------------------------------------------------------

class TestRegressionFixerBehaviorUnchanged:
    """Existing fixer and fixer_retry classification is byte-identical."""

    def test_fixer_no_pr_still_classified_as_lost(self):
        """A failed fixer with no PR is still included in needs_retry."""
        rec = {
            "gpu_id": "gpu-fixer-001",
            "spec_id": "spec-x",
            "agent_type": "fixer",
            "intent": "fix",
            "repo": "lapis-pm",
            "ts": "2026-06-23T22:00:00-07:00",
            "status": "failed",
            "retry_count": 0,
            "lost_retry_count": 0,
        }
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "tgt", [rec], open_prs=[], forgejo_ok=True
            )
        assert len(retry) == 1
        assert retry[0]["agent_type"] == "fixer"

    def test_fixer_retry_not_classified_as_lost(self):
        """fixer_retry is excluded from lost-fixer detection (unchanged)."""
        rec = {
            "gpu_id": "gpu-retry-001",
            "spec_id": "spec-x",
            "agent_type": "fixer_retry",
            "intent": "retry",
            "repo": "lapis-pm",
            "ts": "2026-06-23T22:00:00-07:00",
            "status": "failed",
            "retry_count": 0,
            "lost_retry_count": 0,
        }
        with patch("lapis_pm.episodic.all_comments", return_value=[]):
            retry, brief = pm_core._find_lost_fixer_dispatches(
                "tgt", [rec], open_prs=[], forgejo_ok=True
            )
        assert retry == [], "fixer_retry must not appear in needs_retry"
        assert brief == [], "fixer_retry must not appear in needs_brief"

    def test_fixer_blocked_by_pending_fixer_guard_unchanged(self, episodic_stub):
        """fixer still raises when a pending fixer record exists (unchanged behavior)."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target()
            pending = _dispatch_record("gpu-H", "fixer", "pending")
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[pending]):
                with pytest.raises(ValueError, match="pending fixer"):
                    pm_core.force_dispatch("tgt", "fixer", "intent")

    def test_fixer_blocked_by_pending_fixer_retry_guard_unchanged(self, episodic_stub):
        """fixer still raises when a pending fixer_retry exists (unchanged behavior)."""
        with patch("lapis_pm.pm_core.TargetStore") as ts_mock:
            ts_mock.return_value.get.return_value = _make_target()
            pending = _dispatch_record("gpu-I", "fixer_retry", "pending")
            with patch("lapis_pm.pm_core.load_dispatched", return_value=[pending]):
                with pytest.raises(ValueError, match="pending fixer_retry"):
                    pm_core.force_dispatch("tgt", "fixer", "intent")
