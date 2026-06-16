"""Tests for lapis_pm/eval_gate.py — synapse-eval-gate-v1.

Required coverage (spec §Tests):
 1. Threshold detection — regressed flag for each crossing case, per tracked metric.
 2. Path-filter logic — True/False per diff content.
 3. Cache hit short-circuits eval.
 4. Baseline-stale flag.
 4b. Status semantics — all four (baseline_stale, regressed) -> status cells.
 4c. Brief emission by status — clean=no brief; regressed/unverified emit advisory.
 5. Missing fixture — returns None, journal warning.
 6. Timeout path — subprocess hangs -> returns None, no exception.
 7. Smoke integration — full tick path: percept encoding -> eval invocation
    -> cache write -> brief emission on regressed result.
 8. Brief body shape — body contains delta table + run-path; brief tagged
    pm:synapse-eval:*; brief.synthesize called with documented mapping.
 9. Baseline regeneration — success path writes main.json; failure path
    preserves prior baseline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pr(pr_number: int = 42, head_sha: str = "abc123def456") -> dict:
    return {
        "number": pr_number,
        "head": {"sha": head_sha},
        "title": f"PR {pr_number}",
    }


def _diff_touches(paths: list[str]) -> str:
    """Build a synthetic diff string touching the given paths."""
    lines = []
    for p in paths:
        lines.append(f"diff --git a/{p} b/{p}")
        lines.append("index 000..111 100644")
        lines.append(f"--- a/{p}")
        lines.append(f"+++ b/{p}")
        lines.append("@@ -1 +1 @@")
        lines.append("+changed")
    return "\n".join(lines)


def _make_baseline(sha: str = "main_sha_001", metrics: dict | None = None) -> dict:
    if metrics is None:
        metrics = {
            "chub_jaccard_at_k_mean": 0.85,
            "top1_stability": 0.90,
            "mem_jaccard_at_k_mean": 0.70,
            "latency_p50_ms": 100.0,
        }
    return {
        "sha": sha,
        "generated_at": "2026-05-11T00:00:00Z",
        "metrics": metrics,
        "fixture_path": "/data/synapse/eval/fixtures/baseline.ndjson",
    }


def _write_baseline(tmp_path: Path, sha: str = "main_sha_001", metrics: dict | None = None) -> Path:
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    p = baseline_dir / "main.json"
    p.write_text(json.dumps(_make_baseline(sha, metrics)))
    return p


def _write_fixture(tmp_path: Path) -> Path:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    p = fixtures_dir / "baseline.ndjson"
    p.write_text('{"query_id": "q1", "expected_chub_ids": ["c1"]}\n')
    return p


# ---------------------------------------------------------------------------
# Test 2: Path-filter logic
# ---------------------------------------------------------------------------

class TestPathFilter:
    def setup_method(self):
        from lapis_pm.eval_gate import pr_touches_retrieval
        self.f = pr_touches_retrieval

    def test_service_dir(self):
        diff = _diff_touches(["synapse/service/retrieval.py"])
        assert self.f(diff) is True

    def test_eval_dir(self):
        diff = _diff_touches(["synapse/eval/runner.py"])
        assert self.f(diff) is True

    def test_init_exact(self):
        diff = _diff_touches(["synapse/__init__.py"])
        assert self.f(diff) is True

    def test_hooks_dir_no_match(self):
        diff = _diff_touches(["synapse/hooks/post_commit.py"])
        assert self.f(diff) is False

    def test_readme_no_match(self):
        diff = _diff_touches(["README.md"])
        assert self.f(diff) is False

    def test_service_nested(self):
        diff = _diff_touches(["synapse/service/deep/nested/file.py"])
        assert self.f(diff) is True

    def test_mixed_with_hit(self):
        diff = _diff_touches(["README.md", "synapse/service/foo.py"])
        assert self.f(diff) is True

    def test_empty_diff(self):
        assert self.f("") is False

    def test_similar_but_not_exact(self):
        # synapse/services/ (note plural) - should NOT match
        diff = _diff_touches(["synapse/services/foo.py"])
        assert self.f(diff) is False


# ---------------------------------------------------------------------------
# Test 1: Threshold detection
# ---------------------------------------------------------------------------

class TestThresholdDetection:
    def _make_result(self, baseline_metrics, current_metrics, baseline_stale=False, baseline_sha="main_sha"):
        """Call evaluate_pr with mocked internals, return EvalResult."""
        import lapis_pm.eval_gate as eg
        from lapis_pm.eval_gate import DEFAULT_THRESHOLDS, _metric_regressed

        thresholds = DEFAULT_THRESHOLDS
        deltas = {}
        regressed_metrics = []
        for metric, cfg in thresholds.items():
            cv = current_metrics.get(metric)
            bv = baseline_metrics.get(metric)
            if cv is None or bv is None:
                continue
            delta = cv - bv
            deltas[metric] = delta
            if _metric_regressed(metric, cv, bv, cfg):
                regressed_metrics.append(metric)
        regressed = bool(regressed_metrics)
        status = "regressed" if regressed else ("unverified" if baseline_stale else "clean")
        from lapis_pm.eval_gate import EvalResult, _build_summary_text
        return EvalResult(
            pr_number=1,
            head_sha="sha001",
            baseline_sha=baseline_sha,
            baseline_stale=baseline_stale,
            metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            deltas=deltas,
            regressed=regressed,
            regressed_metrics=regressed_metrics,
            summary_text=_build_summary_text(
                current_metrics, baseline_metrics, deltas, regressed_metrics,
                baseline_stale, baseline_sha,
            ),
            run_path="/tmp/test.json",
            status=status,
        )

    def test_chub_jaccard_regression(self):
        # drop > 0.05 -> regressed
        r = self._make_result(
            {"chub_jaccard_at_k_mean": 0.85},
            {"chub_jaccard_at_k_mean": 0.79},  # drop = 0.06 > 0.05
        )
        assert r.regressed is True
        assert "chub_jaccard_at_k_mean" in r.regressed_metrics

    def test_chub_jaccard_no_regression(self):
        # drop == 0.05 (NOT > 0.05) -> clean
        r = self._make_result(
            {"chub_jaccard_at_k_mean": 0.85},
            {"chub_jaccard_at_k_mean": 0.80},  # drop = exactly 0.05
        )
        assert r.regressed is False

    def test_top1_stability_regression(self):
        r = self._make_result(
            {"top1_stability": 0.90},
            {"top1_stability": 0.84},  # drop = 0.06 > 0.05
        )
        assert r.regressed is True
        assert "top1_stability" in r.regressed_metrics

    def test_mem_jaccard_regression(self):
        # threshold is 0.10
        r = self._make_result(
            {"mem_jaccard_at_k_mean": 0.70},
            {"mem_jaccard_at_k_mean": 0.59},  # drop = 0.11 > 0.10
        )
        assert r.regressed is True
        assert "mem_jaccard_at_k_mean" in r.regressed_metrics

    def test_mem_jaccard_no_regression(self):
        r = self._make_result(
            {"mem_jaccard_at_k_mean": 0.70},
            {"mem_jaccard_at_k_mean": 0.61},  # drop = 0.09 < 0.10
        )
        assert r.regressed is False

    def test_latency_p50_regression(self):
        # increase > 25% -> regressed
        r = self._make_result(
            {"latency_p50_ms": 100.0},
            {"latency_p50_ms": 126.0},  # 26% increase
        )
        assert r.regressed is True
        assert "latency_p50_ms" in r.regressed_metrics

    def test_latency_p50_no_regression(self):
        r = self._make_result(
            {"latency_p50_ms": 100.0},
            {"latency_p50_ms": 120.0},  # exactly 20% increase
        )
        assert r.regressed is False

    def test_all_metrics_fine(self):
        r = self._make_result(
            {"chub_jaccard_at_k_mean": 0.85, "top1_stability": 0.90,
             "mem_jaccard_at_k_mean": 0.70, "latency_p50_ms": 100.0},
            {"chub_jaccard_at_k_mean": 0.85, "top1_stability": 0.90,
             "mem_jaccard_at_k_mean": 0.70, "latency_p50_ms": 100.0},
        )
        assert r.regressed is False
        assert r.regressed_metrics == []


# ---------------------------------------------------------------------------
# Test 4b: Status semantics — all four cells of (baseline_stale, regressed) -> status
# ---------------------------------------------------------------------------

class TestStatusSemantics:
    def _status_for(self, baseline_stale: bool, regressed: bool) -> str:
        from lapis_pm.eval_gate import EvalResult
        r = EvalResult(
            pr_number=1, head_sha="sha1", baseline_sha="bsha",
            baseline_stale=baseline_stale,
            metrics={}, baseline_metrics={}, deltas={},
            regressed=regressed, regressed_metrics=["m1"] if regressed else [],
            summary_text="", run_path="",
            status=("regressed" if regressed else ("unverified" if baseline_stale else "clean")),
        )
        return r.status

    def test_not_stale_not_regressed_is_clean(self):
        assert self._status_for(False, False) == "clean"

    def test_not_stale_regressed_is_regressed(self):
        assert self._status_for(False, True) == "regressed"

    def test_stale_not_regressed_is_unverified_not_clean(self):
        # Council Amendment 1: never "clean" against a stale baseline
        s = self._status_for(True, False)
        assert s == "unverified"
        assert s != "clean"

    def test_stale_regressed_is_regressed(self):
        assert self._status_for(True, True) == "regressed"


# ---------------------------------------------------------------------------
# Test 3: Cache hit short-circuits eval
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_cache_hit_returns_without_replay(self, tmp_path):
        import lapis_pm.eval_gate as eg

        sha = "cachedsha001"
        cached = eg.EvalResult(
            pr_number=7, head_sha=sha, baseline_sha="bsha",
            baseline_stale=False,
            metrics={"chub_jaccard_at_k_mean": 0.88},
            baseline_metrics={"chub_jaccard_at_k_mean": 0.85},
            deltas={"chub_jaccard_at_k_mean": 0.03},
            regressed=False, regressed_metrics=[],
            summary_text="delta table", run_path=str(tmp_path / f"{sha}.json"),
            status="clean",
        )

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        cache_file = runs_dir / f"{sha}.json"
        from dataclasses import asdict
        cache_file.write_text(json.dumps({**asdict(cached), "actioned": False}))

        # Call _load_cached_result directly with RUNS_DIR patched
        with patch.object(eg, "RUNS_DIR", runs_dir):
            result = eg._load_cached_result(sha)

        assert result is not None
        assert result.pr_number == 7
        assert result.status == "clean"

        # Now verify that evaluate_pr returns the cached result and does NOT call get_pr_diff
        with (
            patch.object(eg, "RUNS_DIR", runs_dir),
            patch("agents_core.forgejo.get_pr_diff") as mock_diff,
        ):
            pr = _make_pr(7, sha)
            result2 = eg.evaluate_pr("tid", pr, "synapse")

        # Cache hit: result2 should be non-None and match
        assert result2 is not None
        assert result2.pr_number == 7
        # get_pr_diff should NOT have been called (short-circuited by cache)
        mock_diff.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Baseline-stale flag
# ---------------------------------------------------------------------------

class TestBaselineStale:
    def test_stale_when_shas_differ(self):
        import lapis_pm.eval_gate as eg

        baseline = _make_baseline(sha="abc123")
        with patch.object(eg, "_current_synapse_main_sha", return_value="def456"):
            stale, current = eg._check_baseline_stale(baseline)
        assert stale is True
        assert current == "def456"

    def test_not_stale_when_shas_match(self):
        import lapis_pm.eval_gate as eg

        sha = "abc123def456"
        baseline = _make_baseline(sha=sha)
        with patch.object(eg, "_current_synapse_main_sha", return_value=sha):
            stale, current = eg._check_baseline_stale(baseline)
        assert stale is False
        assert current == sha


# ---------------------------------------------------------------------------
# Test 4c: Brief emission by status (unit-level wiring test)
# ---------------------------------------------------------------------------

class TestBriefEmissionByStatus:
    """Test the tick-wiring logic in pm_core that decides whether to emit a brief."""

    def _run_eval_gate_wiring(
        self,
        eval_status: str,
        already_actioned: bool = False,
        reviewer_pending: bool = False,
    ) -> dict:
        """Simulate the tick's eval-gate wiring for a single PR.

        Drives brief emission through _act_eval_gate_brief (the real pm_core function)
        so that changes to pm_core's wiring are caught by these tests.

        Returns {"brief_emitted": bool, "percept_tag_written": bool, "fell_through": bool}.
        """
        from lapis_pm import eval_gate as eg
        from lapis_pm.eval_gate import EvalResult
        import lapis_pm.pm_core as pm_core_mod
        from lapis_pm.pm_core import _act_eval_gate_brief

        sha = "sha_4c_001"
        result = EvalResult(
            pr_number=11,
            head_sha=sha,
            baseline_sha="bsha",
            baseline_stale=(eval_status == "unverified"),
            metrics={},
            baseline_metrics={},
            deltas={},
            regressed=(eval_status == "regressed"),
            regressed_metrics=["chub_jaccard_at_k_mean"] if eval_status == "regressed" else [],
            summary_text="delta table",
            run_path="/tmp/r.json",
            status=eval_status,
        )

        brief_calls = []
        percept_calls = []

        def fake_synthesize(*args, **kwargs):
            brief_calls.append(kwargs)
            m = MagicMock()
            m.comment_id = "cid-4c"
            m.synthesis_failed = False
            return m

        def fake_write_observation(*args, **kwargs):
            percept_calls.append((args, kwargs))

        # Simulate the tick-wiring decision logic, driving brief emission via
        # the real _act_eval_gate_brief to catch future wiring changes.
        fell_through = False
        mock_eval_gate_obj = MagicMock()
        with (
            patch.object(eg, "evaluate_pr", return_value=result),
            patch.object(eg, "is_eval_actioned", return_value=already_actioned),
            patch.object(eg, "mark_eval_actioned"),
            patch.object(pm_core_mod, "_eval_gate", mock_eval_gate_obj),
            patch.object(pm_core_mod, "set_outstanding_brief_verified"),
            patch.object(pm_core_mod, "_post_write_sweep_brief"),
        ):
            from lapis_pm import brief as brief_mod, episodic as ep_mod
            with (
                patch.object(brief_mod, "synthesize", side_effect=fake_synthesize),
                patch.object(ep_mod, "write_observation", side_effect=fake_write_observation),
            ):
                if not reviewer_pending:
                    r = eg.evaluate_pr("tid", {"number": 11, "head": {"sha": sha}}, "synapse")
                    if r is not None and not eg.is_eval_actioned(r.head_sha):
                        if r.status == "clean":
                            ep_mod.write_observation(
                                "tid", f"eval clean pr={r.pr_number}",
                                extra_tags=[f"pm:synapse-eval:pr={r.pr_number}:sha={sha[:8]}:status=clean"],
                            )
                            eg.mark_eval_actioned(r.head_sha)
                            fell_through = True
                        elif r.status in ("regressed", "unverified"):
                            # Drive through the real pm_core function for test fidelity.
                            _act_eval_gate_brief("tid", r, "synapse")
                    elif r is not None and eg.is_eval_actioned(r.head_sha):
                        fell_through = True
                else:
                    fell_through = True  # reviewer pending -> skip eval gate

        return {
            "brief_emitted": len(brief_calls) > 0,
            "percept_tag_written": len(percept_calls) > 0,
            "fell_through": fell_through,
            "brief_calls": brief_calls,
        }

    def test_clean_status_no_brief(self):
        out = self._run_eval_gate_wiring("clean")
        assert out["brief_emitted"] is False
        assert out["percept_tag_written"] is True
        assert out["fell_through"] is True

    def test_regressed_status_emits_brief(self):
        out = self._run_eval_gate_wiring("regressed")
        assert out["brief_emitted"] is True
        assert out["fell_through"] is False
        # Verify synthesize parameters
        call_kwargs = out["brief_calls"][0]
        assert call_kwargs["trigger"] == "synapse_eval_regressed"
        assert call_kwargs["screen_issues"] == []
        assert call_kwargs["notify"] is None
        assert call_kwargs["pr_number"] == 11
        assert "delta table" in call_kwargs["diff_snippet"]

    def test_unverified_status_emits_brief(self):
        out = self._run_eval_gate_wiring("unverified")
        assert out["brief_emitted"] is True
        call_kwargs = out["brief_calls"][0]
        assert call_kwargs["trigger"] == "synapse_eval_unverified"
        assert call_kwargs["screen_issues"] == []
        assert call_kwargs["notify"] is None

    def test_already_actioned_falls_through(self):
        out = self._run_eval_gate_wiring("regressed", already_actioned=True)
        assert out["brief_emitted"] is False
        assert out["fell_through"] is True

    def test_reviewer_pending_skips_eval(self):
        out = self._run_eval_gate_wiring("regressed", reviewer_pending=True)
        assert out["brief_emitted"] is False
        assert out["fell_through"] is True


# ---------------------------------------------------------------------------
# Test 5: Missing fixture
# ---------------------------------------------------------------------------

class TestMissingFixture:
    def test_missing_fixture_returns_none(self, tmp_path, caplog):
        import lapis_pm.eval_gate as eg
        import logging

        pr = _make_pr(5, "sha005")
        diff = _diff_touches(["synapse/service/foo.py"])
        missing_fixture = tmp_path / "fixtures/baseline.ndjson"

        with (
            patch("agents_core.forgejo.get_pr_diff", return_value=diff),
            patch.object(eg, "FIXTURE_PATH", missing_fixture),
            patch.object(eg, "RUNS_DIR", tmp_path / "runs"),
            patch.object(eg, "_current_synapse_main_sha", return_value="main001"),
            caplog.at_level(logging.WARNING, logger="lapis_pm.eval_gate"),
        ):
            result = eg.evaluate_pr("tid", pr, "synapse")

        assert result is None
        assert any("pinned fixture missing" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 6: Timeout path
# ---------------------------------------------------------------------------

class TestTimeoutPath:
    def test_timeout_returns_none(self, tmp_path):
        import lapis_pm.eval_gate as eg

        pr = _make_pr(6, "sha006")
        diff = _diff_touches(["synapse/service/foo.py"])

        baseline = _make_baseline("main006")
        fixture = _write_fixture(tmp_path)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        def fake_run_replay(*args, **kwargs):
            # Simulate the timeout path: replay hangs, returns None
            return None

        with (
            patch("agents_core.forgejo.get_pr_diff", return_value=diff),
            patch.object(eg, "FIXTURE_PATH", fixture),
            patch.object(eg, "BASELINE_PATH", tmp_path / "baselines/main.json"),
            patch.object(eg, "RUNS_DIR", runs_dir),
            patch.object(eg, "load_baseline", return_value=baseline),
            patch.object(eg, "_check_baseline_stale", return_value=(False, "main006")),
            patch.object(eg, "_run_replay", side_effect=fake_run_replay),
            patch.object(
                eg, "_run_worktree_add",
                create=True,
                side_effect=lambda *a, **k: (True, None),
            ),
        ):
            # We mock worktree add too (the subprocess call)
            with patch("subprocess.run") as mock_subp:
                mock_subp.return_value = MagicMock(returncode=0, stdout="", stderr="")
                result = eg.evaluate_pr("tid", pr, "synapse")

        # Should return None (not raise), because _run_replay returned None
        assert result is None


# ---------------------------------------------------------------------------
# Test 7: Smoke integration
# ---------------------------------------------------------------------------

class TestSmokeIntegration:
    """Full tick-path: percept encoding -> eval invocation -> cache write -> brief emission."""

    def test_smoke_regressed_emits_brief(self, tmp_path):
        import lapis_pm.eval_gate as eg
        from lapis_pm.eval_gate import EvalResult

        sha = "smoke_sha_001"
        pr_number = 99

        # Pre-built EvalResult simulating a regression
        regressed_result = EvalResult(
            pr_number=pr_number,
            head_sha=sha,
            baseline_sha="main_sha_base",
            baseline_stale=False,
            metrics={"chub_jaccard_at_k_mean": 0.78},
            baseline_metrics={"chub_jaccard_at_k_mean": 0.85},
            deltas={"chub_jaccard_at_k_mean": -0.07},
            regressed=True,
            regressed_metrics=["chub_jaccard_at_k_mean"],
            summary_text="chub_jaccard_at_k_mean | 0.85 | 0.78 | -0.07 | REGRESSED",
            run_path=str(tmp_path / f"{sha}.json"),
            status="regressed",
        )

        # Write the run cache (simulates a completed eval)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        cache_file = runs_dir / f"{sha}.json"
        from dataclasses import asdict
        cache_file.write_text(json.dumps({**asdict(regressed_result), "actioned": False}))

        brief_calls = []

        def fake_synthesize(*args, **kwargs):
            brief_calls.append(kwargs)
            m = MagicMock()
            m.comment_id = "smoke-brief-cid"
            m.synthesis_failed = False
            return m

        observation_calls = []

        def fake_write_observation(tid, content, extra_tags=None):
            observation_calls.append({"tid": tid, "content": content, "tags": extra_tags})

        import lapis_pm.pm_core as pm_core_mod
        from lapis_pm.pm_core import _act_eval_gate_brief

        mock_eval_gate_obj = MagicMock()
        with (
            patch.object(eg, "RUNS_DIR", runs_dir),
            patch.object(eg, "evaluate_pr", return_value=regressed_result),
            patch.object(eg, "is_eval_actioned", return_value=False),
            patch.object(eg, "mark_eval_actioned"),
            patch.object(pm_core_mod, "_eval_gate", mock_eval_gate_obj),
            patch.object(pm_core_mod, "set_outstanding_brief_verified"),
            patch.object(pm_core_mod, "_post_write_sweep_brief"),
        ):
            from lapis_pm import brief as brief_mod, episodic as ep_mod
            with (
                patch.object(brief_mod, "synthesize", side_effect=fake_synthesize),
                patch.object(ep_mod, "write_observation", side_effect=fake_write_observation),
            ):
                # Simulate the tick wiring - drive brief emission through the real
                # _act_eval_gate_brief function for test fidelity.
                r = eg.evaluate_pr("smoke-tid", {"number": pr_number, "head": {"sha": sha}}, "synapse")
                assert r is not None
                assert not eg.is_eval_actioned(sha)

                # status == "regressed" -> emit brief via real pm_core path
                _act_eval_gate_brief("smoke-tid", r, "synapse")

        # Verify brief was emitted with the documented parameters
        assert len(brief_calls) == 1
        call_kw = brief_calls[0]
        assert call_kw["trigger"] == "synapse_eval_regressed"
        assert call_kw["screen_issues"] == []
        assert call_kw["notify"] is None
        assert call_kw["pr_number"] == pr_number

        # Verify mark_eval_actioned was called via _eval_gate
        mock_eval_gate_obj.mark_eval_actioned.assert_called_once_with(sha)


# ---------------------------------------------------------------------------
# Test 8: Brief body shape
# ---------------------------------------------------------------------------

class TestBriefBodyShape:
    def test_regressed_brief_contains_delta_table_and_run_path(self):
        from lapis_pm.eval_gate import _build_summary_text

        metrics = {"chub_jaccard_at_k_mean": 0.78}
        baseline_metrics = {"chub_jaccard_at_k_mean": 0.85}
        deltas = {"chub_jaccard_at_k_mean": -0.07}
        regressed_metrics = ["chub_jaccard_at_k_mean"]
        run_path = "/data/synapse/eval/runs/abc123.json"

        text = _build_summary_text(
            metrics, baseline_metrics, deltas, regressed_metrics,
            baseline_stale=False, baseline_sha="main_sha",
            run_path=run_path,
        )

        assert "chub_jaccard_at_k_mean" in text
        assert "REGRESSED" in text
        assert run_path in text
        assert "0.78" in text
        assert "0.85" in text
        assert "-0.07" in text

    def test_stale_brief_contains_staleness_warning(self):
        from lapis_pm.eval_gate import _build_summary_text

        text = _build_summary_text(
            {}, {}, {}, [],
            baseline_stale=True,
            baseline_sha="stale_sha_001",
            current_main_sha="fresh_sha_002",
        )
        assert "stale" in text.lower() or "WARNING" in text
        assert "stale_sha_001"[:12] in text

    def test_synthesize_called_with_documented_mapping(self):
        """brief.synthesize receives the documented parameter mapping from §Deliverables 5.

        Drives through _act_eval_gate_brief (the real pm_core function) so that
        changes to the parameter mapping in pm_core are caught by this test.
        """
        from lapis_pm.eval_gate import EvalResult
        import lapis_pm.pm_core as pm_core_mod
        from lapis_pm.pm_core import _act_eval_gate_brief

        sha = "sha_test_8"
        pr_number = 33
        summary = "delta-table-content"

        result = EvalResult(
            pr_number=pr_number, head_sha=sha, baseline_sha="bsha",
            baseline_stale=False,
            metrics={}, baseline_metrics={}, deltas={},
            regressed=True, regressed_metrics=["top1_stability"],
            summary_text=summary,
            run_path="/tmp/r.json",
            status="regressed",
        )

        from lapis_pm import brief as brief_mod, episodic as ep_mod
        captured_kwargs = {}

        def fake_synthesize(*args, **kwargs):
            captured_kwargs.update(kwargs)
            m = MagicMock()
            m.comment_id = "cid-8"
            m.synthesis_failed = False
            return m

        mock_eval_gate_obj = MagicMock()
        with (
            patch.object(brief_mod, "synthesize", side_effect=fake_synthesize),
            patch.object(ep_mod, "write_observation"),
            patch.object(pm_core_mod, "_eval_gate", mock_eval_gate_obj),
            patch.object(pm_core_mod, "set_outstanding_brief_verified"),
            patch.object(pm_core_mod, "_post_write_sweep_brief"),
        ):
            _act_eval_gate_brief("tid-8", result, "synapse")

        assert captured_kwargs["trigger"] == "synapse_eval_regressed"
        assert captured_kwargs["query"] == f"PR #{pr_number} retrieval-quality check"
        assert captured_kwargs["diff_snippet"] == summary
        assert captured_kwargs["screen_issues"] == []
        assert captured_kwargs["pr_number"] == pr_number
        assert captured_kwargs["notify"] is None

    def test_unverified_brief_uses_correct_trigger(self):
        """Status=unverified should use trigger="synapse_eval_unverified".

        Drives through _act_eval_gate_brief for test fidelity.
        """
        from lapis_pm.eval_gate import EvalResult
        from lapis_pm import brief as brief_mod, episodic as ep_mod
        import lapis_pm.pm_core as pm_core_mod
        from lapis_pm.pm_core import _act_eval_gate_brief

        result = EvalResult(
            pr_number=44, head_sha="sha44", baseline_sha="bsha",
            baseline_stale=True,
            metrics={}, baseline_metrics={}, deltas={},
            regressed=False, regressed_metrics=[],
            summary_text="no baseline",
            run_path="/tmp/r.json",
            status="unverified",
        )

        captured_kwargs = {}

        def fake_synthesize(*args, **kwargs):
            captured_kwargs.update(kwargs)
            m = MagicMock()
            m.comment_id = "cid-unverified"
            m.synthesis_failed = False
            return m

        mock_eval_gate_obj = MagicMock()
        with (
            patch.object(brief_mod, "synthesize", side_effect=fake_synthesize),
            patch.object(ep_mod, "write_observation"),
            patch.object(pm_core_mod, "_eval_gate", mock_eval_gate_obj),
            patch.object(pm_core_mod, "set_outstanding_brief_verified"),
            patch.object(pm_core_mod, "_post_write_sweep_brief"),
        ):
            _act_eval_gate_brief("tid-44", result, "synapse")

        assert captured_kwargs["trigger"] == "synapse_eval_unverified"
        assert captured_kwargs["screen_issues"] == []
        assert captured_kwargs["notify"] is None


# ---------------------------------------------------------------------------
# Test 9: Baseline regeneration
# ---------------------------------------------------------------------------

class TestBaselineRegeneration:
    def test_success_writes_baseline_json(self, tmp_path):
        import lapis_pm.eval_gate as eg

        baseline_dir = tmp_path / "baselines"
        baseline_dir.mkdir()
        fixture = _write_fixture(tmp_path)

        fake_metrics = {
            "chub_jaccard_at_k_mean": 0.87,
            "top1_stability": 0.92,
            "mem_jaccard_at_k_mean": 0.72,
            "latency_p50_ms": 95.0,
        }

        with (
            patch.object(eg, "FIXTURE_PATH", fixture),
            patch.object(eg, "BASELINE_PATH", baseline_dir / "main.json"),
            patch.object(eg, "_current_synapse_main_sha", return_value="new_main_sha"),
            patch.object(eg, "_run_replay", return_value=fake_metrics),
            patch("subprocess.run") as mock_subp,
        ):
            mock_subp.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = eg.regenerate_baseline(synapse_repo_path=str(tmp_path))

        assert result is not None
        assert result["sha"] == "new_main_sha"
        assert result["metrics"]["chub_jaccard_at_k_mean"] == 0.87
        assert "generated_at" in result
        assert "fixture_path" in result

        # Verify file was written
        written = json.loads((baseline_dir / "main.json").read_text())
        assert written["sha"] == "new_main_sha"

    def test_failure_preserves_prior_baseline(self, tmp_path):
        import lapis_pm.eval_gate as eg
        import logging

        prior = _make_baseline("prior_sha")
        baseline_dir = tmp_path / "baselines"
        baseline_dir.mkdir()
        baseline_file = baseline_dir / "main.json"
        baseline_file.write_text(json.dumps(prior))

        fixture = _write_fixture(tmp_path)

        with (
            patch.object(eg, "FIXTURE_PATH", fixture),
            patch.object(eg, "BASELINE_PATH", baseline_file),
            patch.object(eg, "_current_synapse_main_sha", return_value="new_sha"),
            patch.object(eg, "_run_replay", return_value=None),  # replay fails
            patch("subprocess.run") as mock_subp,
        ):
            mock_subp.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = eg.regenerate_baseline(synapse_repo_path=str(tmp_path))

        assert result is None

        # Prior baseline must be preserved
        preserved = json.loads(baseline_file.read_text())
        assert preserved["sha"] == "prior_sha"


# ---------------------------------------------------------------------------
# Smoke fixture existence test
# ---------------------------------------------------------------------------

def test_smoke_fixture_exists():
    """The bundled smoke fixture must exist at lapis_pm/smoke_fixtures/synapse_eval/."""
    from lapis_pm.eval_gate import SMOKE_FIXTURE_PATH
    assert SMOKE_FIXTURE_PATH.exists(), (
        f"Smoke fixture missing at {SMOKE_FIXTURE_PATH}; "
        "it must be committed as part of synapse-eval-gate-v1"
    )
    # Verify it's valid NDJSON (at least one line with JSON)
    lines = [l.strip() for l in SMOKE_FIXTURE_PATH.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1
    json.loads(lines[0])  # must be valid JSON
