"""Tests for lapis_pm.corroboration_adapter (cross-node-corroboration-v0 PR 3).

Coverage (per spec §Tests "PR 3"):
1. retrieve extracts doc-mentioned-identifier claims from a fixture diff
2. score flags a missing-referent claim
3. score flags a renamed-referent claim
4. score returns clean for accurate doc-mentioned-identifier references
5. Reviewer integration: verdict carries corroboration_result;
   review-state-cache stores it; resume reads it back unchanged
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.corroboration_adapter import (
    Citation,
    CorroborationResult,
    LapisPMReviewerAdapter,
    _extract_identifiers,
    _IdentifierSubstrate,
    run_corroboration_pass,
)


# ---------------------------------------------------------------------------
# Fixture diffs
# ---------------------------------------------------------------------------

FIXTURE_DIFF = """\
--- a/lapis_pm/corroboration_adapter.py
+++ b/lapis_pm/corroboration_adapter.py
@@ -0,0 +1,20 @@
+class LapisPMReviewerAdapter:
+    def retrieve(self, diff_text: str, repo: str) -> list:
+        pass
+    def score(self, claim: str, substrate: list) -> CorroborationResult:
+        pass
+
+def run_corroboration_pass(diff_text: str, repo: str) -> dict:
+    \"\"\"Entry point for `corroboration_result` wiring.\"\"\"
+    pass
+
commit message: wire `LapisPMReviewerAdapter` into reviewer verdict path
"""

FIXTURE_DIFF_PROSE_ONLY = """\
--- a/README.md
+++ b/README.md
@@ -1,3 +1,5 @@
+The `NonExistentHelper` function is referenced here for clarity.
+See `lapis_pm.pm_core` for the main tick loop.
"""

FIXTURE_DIFF_CLEAN = """\
--- a/lapis_pm/pm_core.py
+++ b/lapis_pm/pm_core.py
@@ -1,3 +1,5 @@
+def tick(target_id: str) -> TickResult:
+    pass
"""


# ---------------------------------------------------------------------------
# Test 1: retrieve extracts doc-mentioned-identifier claims from fixture diff
# ---------------------------------------------------------------------------

class TestRetrieveExtractsIdentifiers:

    def test_extract_identifiers_finds_file_paths(self):
        """File paths from --- a/... / +++ b/... headers are extracted."""
        identifiers = _extract_identifiers(FIXTURE_DIFF)
        # Should include the file path and/or its stem
        assert any("corroboration_adapter" in i for i in identifiers), (
            f"Expected corroboration_adapter in identifiers, got: {identifiers}"
        )

    def test_extract_identifiers_finds_class_names(self):
        """Class definitions on added lines are extracted."""
        identifiers = _extract_identifiers(FIXTURE_DIFF)
        assert "LapisPMReviewerAdapter" in identifiers, (
            f"Expected LapisPMReviewerAdapter in identifiers, got: {identifiers}"
        )

    def test_extract_identifiers_finds_backtick_quoted(self):
        """Backtick-quoted identifiers in prose are extracted."""
        identifiers = _extract_identifiers(FIXTURE_DIFF_PROSE_ONLY)
        assert any("lapis_pm.pm_core" in i or "lapis_pm" in i for i in identifiers), (
            f"Expected lapis_pm module path in identifiers, got: {identifiers}"
        )

    def test_extract_identifiers_capped(self):
        """Output is capped at _MAX_IDENTIFIERS."""
        from lapis_pm.corroboration_adapter import _MAX_IDENTIFIERS
        # Generate a diff with many unique identifiers
        big_diff = "\n".join(
            f"`identifier_name_{i}`" for i in range(100)
        )
        identifiers = _extract_identifiers(big_diff)
        assert len(identifiers) <= _MAX_IDENTIFIERS

    def test_retrieve_calls_grep_for_each_identifier(self):
        """retrieve() calls _grep_repo for each extracted identifier."""
        adapter = LapisPMReviewerAdapter(repo_path="/fake/repo")

        with (
            patch("lapis_pm.corroboration_adapter._grep_repo", return_value=[]) as mock_grep,
            patch("lapis_pm.corroboration_adapter._vault_grep", return_value=[]),
        ):
            substrates = adapter.retrieve(FIXTURE_DIFF, "lapis-pm", "/fake/repo")

        assert mock_grep.called, "Expected _grep_repo to be called"
        called_idents = [call[0][0] for call in mock_grep.call_args_list]
        assert len(called_idents) > 0, "Expected at least one grep call"


# ---------------------------------------------------------------------------
# Test 2: score flags a missing-referent claim
# ---------------------------------------------------------------------------

class TestScoreFlagsMissingReferent:

    def _make_substrate_no_hits(self, identifier: str) -> _IdentifierSubstrate:
        return _IdentifierSubstrate(
            identifier=identifier,
            repo_hits=[],   # not found in repo
            vault_hits=[],
        )

    def _mock_llm_response(self, verdict: str, claims: list[dict], summary: str) -> dict:
        return {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "verdict": verdict,
                        "claims": claims,
                        "summary": summary,
                    })
                }
            }]
        }

    def test_score_flags_missing_referent(self):
        """score returns flagged + drift_class=missing_referent when identifier absent."""
        adapter = LapisPMReviewerAdapter()
        substrates = [self._make_substrate_no_hits("NonExistentHelper")]

        llm_resp = self._mock_llm_response(
            verdict="flagged",
            claims=[{
                "identifier": "NonExistentHelper",
                "drift_class": "missing_referent",
                "notes": "not found in repo",
            }],
            summary="NonExistentHelper referenced in prose but not found in repo",
        )

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: llm_resp,
                raise_for_status=lambda: None,
            )
            result = adapter.score(FIXTURE_DIFF_PROSE_ONLY, substrates, "lapis-pm")

        assert result.verdict == "flagged"
        assert result.drift_class == "missing_referent"
        assert result.scope_id == "repo:lapis-pm"

    def test_score_missing_referent_no_citations_when_no_hits(self):
        """When identifier has no repo_hits, citations list is empty."""
        adapter = LapisPMReviewerAdapter()
        substrates = [self._make_substrate_no_hits("PhantomClass")]

        llm_resp = self._mock_llm_response(
            verdict="flagged",
            claims=[{
                "identifier": "PhantomClass",
                "drift_class": "missing_referent",
                "notes": "not found",
            }],
            summary="PhantomClass not found",
        )

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: llm_resp,
                raise_for_status=lambda: None,
            )
            result = adapter.score("content with `PhantomClass`", substrates, "lapis-pm")

        assert result.citations == []  # no hits → no citations


# ---------------------------------------------------------------------------
# Test 3: score flags a renamed-referent claim
# ---------------------------------------------------------------------------

class TestScoreFlagsRenamedReferent:

    def _make_substrate_with_hits(self, identifier: str) -> _IdentifierSubstrate:
        return _IdentifierSubstrate(
            identifier=identifier,
            repo_hits=[{
                "file": "/repo/lapis_pm/pm_core.py",
                "line": "42",
                "text": "def tick_old(target_id: str) -> TickResult:",
            }],
            vault_hits=[],
        )

    def test_score_flags_renamed_referent(self):
        """score returns flagged + drift_class=renamed_referent."""
        adapter = LapisPMReviewerAdapter()
        substrates = [self._make_substrate_with_hits("tick")]

        llm_resp = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "verdict": "flagged",
                        "claims": [{
                            "identifier": "tick",
                            "drift_class": "renamed_referent",
                            "notes": "function was renamed to tick_old",
                        }],
                        "summary": "tick renamed to tick_old",
                    })
                }
            }]
        }

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: llm_resp,
                raise_for_status=lambda: None,
            )
            result = adapter.score("diff mentioning `tick`", substrates, "lapis-pm")

        assert result.verdict == "flagged"
        assert result.drift_class == "renamed_referent"
        # Citation should reference the repo hit
        assert len(result.citations) == 1
        assert result.citations[0].source_id.startswith("repo:lapis-pm:")


# ---------------------------------------------------------------------------
# Test 4: score returns clean for accurate doc-mentioned-identifier references
# ---------------------------------------------------------------------------

class TestScoreClean:

    def _make_substrate_found(self, identifier: str) -> _IdentifierSubstrate:
        return _IdentifierSubstrate(
            identifier=identifier,
            repo_hits=[{
                "file": "/repo/lapis_pm/pm_core.py",
                "line": "1604",
                "text": "def tick(target_id: str, allow_auto_land: bool = True) -> TickResult:",
            }],
            vault_hits=[],
        )

    def test_score_clean_when_identifiers_exist(self):
        """score returns clean when all identifiers found as claimed."""
        adapter = LapisPMReviewerAdapter()
        substrates = [self._make_substrate_found("tick")]

        llm_resp = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "verdict": "clean",
                        "claims": [{
                            "identifier": "tick",
                            "drift_class": "none",
                            "notes": "exists as expected",
                        }],
                        "summary": "All identifiers valid",
                    })
                }
            }]
        }

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: llm_resp,
                raise_for_status=lambda: None,
            )
            result = adapter.score(FIXTURE_DIFF_CLEAN, substrates, "lapis-pm")

        assert result.verdict == "clean"
        assert result.drift_class is None  # "none" → None in output

    def test_score_uncertain_when_no_substrates(self):
        """score returns uncertain immediately when substrate list is empty."""
        adapter = LapisPMReviewerAdapter()
        result = adapter.score("", [], "lapis-pm")
        assert result.verdict == "uncertain"
        assert result.notes == "No identifiers found in diff"

    def test_score_uncertain_on_llm_failure(self):
        """score returns uncertain when LLM call raises."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=ConnectionError("refused")),
        ):
            result = adapter.score("some diff", substrates, "lapis-pm")

        assert result.verdict == "uncertain"
        assert result.drift_class is None
        assert "LLM unavailable" in (result.notes or "")


# ---------------------------------------------------------------------------
# leg_status field (lapis-pm-corroboration-producer-leg-status-v1)
# ---------------------------------------------------------------------------

class TestLegStatus:
    """leg_status is set correctly at every CorroborationResult construction
    site and is emitted by to_dict(). This is the field consumers must read
    instead of string-matching `claim`/`notes` prose."""

    def test_to_dict_emits_leg_status(self):
        """to_dict() includes leg_status; default is 'ok'."""
        result = CorroborationResult(
            verdict="clean",
            claim="x",
            citations=[],
            freshness_stamp="2026-08-10T00:00:00+00:00",
            scope_id="repo:lapis-pm",
        )
        assert result.leg_status == "ok"
        assert result.to_dict()["leg_status"] == "ok"

    def test_success_path_leg_status_ok_even_when_summary_missing(self):
        """D3 guard: a healthy pass whose response omits `summary` still
        reports leg_status == 'ok', even though claim == '' — the field this
        unit exists for. claim-string matching cannot distinguish this case
        from a failure marker; leg_status can."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        llm_resp = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "verdict": "clean",
                        "claims": [{"identifier": "tick", "drift_class": "none", "notes": "ok"}],
                        # "summary" deliberately omitted
                    })
                }
            }]
        }

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: llm_resp,
                raise_for_status=lambda: None,
            )
            result = adapter.score("some diff", substrates, "lapis-pm")

        assert result.claim == "", "expected empty claim when summary is omitted"
        assert result.verdict == "clean"
        assert result.leg_status == "ok"

    def test_probe_unreachable_leg_status_substrate_unavailable(self):
        """Probe-unreachable path sets leg_status='substrate_unavailable'."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [], [])]

        with patch("lapis_pm.corroboration_adapter.node_reachable", return_value=False):
            result = adapter.score("some diff", substrates, "lapis-pm")

        assert result.verdict == "uncertain"
        assert result.leg_status == "substrate_unavailable"

    def test_llm_exception_leg_status_substrate_unavailable(self):
        """LLM-call-exception path sets leg_status='substrate_unavailable'."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=ConnectionError("refused")),
        ):
            result = adapter.score("some diff", substrates, "lapis-pm")

        assert result.leg_status == "substrate_unavailable"

    def test_no_identifiers_leg_status_ok(self):
        """(no identifiers extracted) is a legitimate clean result, not
        starvation — it must stay leg_status='ok'."""
        adapter = LapisPMReviewerAdapter()
        result = adapter.score("", [], "lapis-pm")
        assert result.claim == "(no identifiers extracted)"
        assert result.leg_status == "ok"

    def test_make_uncertain_returns_fresh_instances(self):
        """_make_uncertain returns a brand-new object on every call — never
        the same shared instance (the D2 root cause)."""
        from lapis_pm.corroboration_adapter import _make_uncertain

        r1 = _make_uncertain("lapis-pm", "first failure")
        r2 = _make_uncertain("lapis-pm", "second failure")

        assert r1 is not r2
        assert r1.notes == "first failure"
        assert r2.notes == "second failure"
        assert r1.leg_status == "pass_failed"
        assert r2.leg_status == "pass_failed"

    def test_pm_core_outer_except_leg_status_pass_failed(self):
        """pm_core._run_corroboration_pass_sync's outer-except dict carries
        leg_status='pass_failed' so nothing downstream mistakes it for a
        healthy result (D1c)."""
        from lapis_pm import pm_core

        with patch(
            "lapis_pm.corroboration_adapter.run_corroboration_pass",
            side_effect=RuntimeError("boom"),
        ):
            result = pm_core._run_corroboration_pass_sync("diff", "lapis-pm")

        assert result["leg_status"] == "pass_failed"
        assert result["verdict"] == "uncertain"


# ---------------------------------------------------------------------------
# Test 5: Reviewer integration — verdict carries corroboration_result;
#         review-state-cache stores it; resume reads it back unchanged
# ---------------------------------------------------------------------------

class TestReviewerIntegration:

    def _make_pending_reviewer_record(self, repo: str = "lapis-pm") -> dict:
        return {
            "gpu_id": "gpu-corr-test-001",
            "spec_id": "spec-corr",
            "agent_type": "reviewer",
            "intent": "review PR #7 cycle 1",
            "repo": repo,
            "pr_number": 7,
            "cycle": 1,
            "status": "pending",
            "retry_count": 0,
        }

    def test_verdict_carries_corroboration_result(self):
        """After _encode_gpu_results, the episodic entry JSON contains
        corroboration_result with the expected shape."""
        from lapis_pm import pm_core
        from unittest.mock import MagicMock, patch

        record = self._make_pending_reviewer_record()
        output = json.dumps({"verdict": "clean", "issues": [], "confidence": 0.95})
        written_contents: list[str] = []

        def capture_write_result(target_id, content, extra_tags=None):
            written_contents.append(content)
            c = MagicMock()
            c.id = "corr-test-comment"
            return c

        mock_path = MagicMock()
        mock_path.read_text.return_value = output
        mock_path.parent = object()

        corr_result = {
            "verdict": "clean",
            "claim": "all identifiers valid",
            "citations": [],
            "freshness_stamp": datetime.now(timezone.utc).isoformat(),
            "scope_id": "repo:lapis-pm",
            "drift_class": None,
            "notes": "clean",
        }

        with (
            patch("lapis_pm.pm_core.load_dispatched", return_value=[record]),
            patch("lapis_pm.pm_core.save_dispatched"),
            patch("lapis_pm.pm_core._gpu_output_path", return_value=mock_path),
            patch("lapis_pm.pm_core._read_fixer_meta", return_value=None),
            patch("lapis_pm.pm_core._consume_fixer_meta"),
            patch("lapis_pm.pm_core.episodic.write_result",
                  side_effect=capture_write_result),
            patch("lapis_pm.pm_core.FAILED_DIR", object()),
            patch("lapis_pm.pm_core._diff_text_for_corr", return_value="fake diff"),
            patch("lapis_pm.pm_core._run_corroboration_pass_sync",
                  return_value=corr_result),
        ):
            encoded, failed = pm_core._encode_gpu_results("my-target")

        assert encoded == 1
        assert not failed
        # Find the reviewer verdict content entry
        reviewer_entries = [c for c in written_contents if "Reviewer verdict for PR #7:" in c]
        assert reviewer_entries, f"No reviewer verdict entry in {written_contents}"
        json_part = reviewer_entries[0].split("\n", 1)[-1].strip()
        verdict_data = json.loads(json_part)
        assert "corroboration_result" in verdict_data, (
            f"corroboration_result missing from verdict JSON: {verdict_data}"
        )
        assert verdict_data["corroboration_result"]["verdict"] == "clean"

    def test_last_review_verdict_preserves_corroboration_result(self):
        """_last_review_verdict returns corroboration_result unchanged (resume path)."""
        from lapis_pm import pm_core

        corr_result = {
            "verdict": "flagged",
            "claim": "missing referent",
            "citations": [{"source": "repo:lapis-pm:foo.py:1", "text": "def foo"}],
            "freshness_stamp": "2026-04-30T12:00:00+00:00",
            "scope_id": "repo:lapis-pm",
            "drift_class": "missing_referent",
            "notes": "foo missing",
        }
        verdict_json = json.dumps({
            "verdict": "fixable",
            "issues": [{"severity": "med", "path": "foo.py", "note": "missing"}],
            "confidence": 0.8,
            "corroboration_result": corr_result,
        })

        # Simulate an episodic comment with this verdict
        comment = MagicMock()
        comment.tags = ["pm:reviewer:pr=7:cycle=1:verdict=fixable", "pm:pr=7"]
        comment.content = f"Reviewer verdict for PR #7:\n{verdict_json}"

        with patch("lapis_pm.episodic.all_comments", return_value=[comment]):
            result = pm_core._last_review_verdict("my-target", 7)

        assert result is not None
        assert result["verdict"] == "fixable"
        assert "corroboration_result" in result, (
            "corroboration_result must be preserved by _last_review_verdict"
        )
        assert result["corroboration_result"]["drift_class"] == "missing_referent"

    def test_review_state_cache_stores_last_corroboration(self):
        """_persist_review_state_cache stores last_corroboration when verdict has one."""
        from lapis_pm import pm_core

        mem = MagicMock()
        mem.get.return_value = None
        target = MagicMock()
        target.pm_authority = "advisory"

        corr_result = {
            "verdict": "flagged",
            "drift_class": "missing_referent",
        }
        verdict_with_corr = {
            "verdict": "fixable",
            "issues": [{"severity": "med", "path": "foo.py", "note": "x"}],
            "confidence": 0.8,
            "corroboration_result": corr_result,
        }

        active_state = {
            "pr_number": 7,
            "cycle": 1,
            "verdict": "fixable",
            "issues": 1,
        }

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=active_state),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict_with_corr),
        ):
            pm_core._persist_review_state_cache("my-target", target, [])

        mem.set.assert_called_once()
        payload = json.loads(mem.set.call_args[0][1])
        assert "last_corroboration" in payload, (
            f"last_corroboration missing from cache payload: {payload}"
        )
        assert payload["last_corroboration"]["verdict"] == "flagged"
        assert payload["last_corroboration"]["drift_class"] == "missing_referent"

    def test_review_state_cache_last_corroboration_null_when_no_corr(self):
        """last_corroboration is null in cache when verdict has no corroboration_result."""
        from lapis_pm import pm_core

        mem = MagicMock()
        mem.get.return_value = None
        target = MagicMock()
        target.pm_authority = "advisory"

        verdict_no_corr = {
            "verdict": "clean",
            "issues": [],
            "confidence": 0.95,
            # no corroboration_result key
        }

        active_state = {
            "pr_number": 7,
            "cycle": 1,
            "verdict": "clean",
            "issues": 0,
        }

        with (
            patch("lapis_pm.pm_core._active_review_state", return_value=active_state),
            patch("lapis_pm.pm_core._review_gate_paused", return_value=False),
            patch("lapis_pm.pm_core._mem", return_value=mem),
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict_no_corr),
        ):
            pm_core._persist_review_state_cache("my-target", target, [])

        payload = json.loads(mem.set.call_args[0][1])
        assert payload["last_corroboration"] is None


# ---------------------------------------------------------------------------
# run_corroboration_pass: public entry point
# ---------------------------------------------------------------------------

class TestRunCorroborationPass:

    def test_returns_uncertain_on_adapter_error(self):
        """run_corroboration_pass returns uncertain dict when adapter fails."""
        with patch(
            "lapis_pm.corroboration_adapter.LapisPMReviewerAdapter.retrieve",
            side_effect=RuntimeError("boom"),
        ):
            result = run_corroboration_pass("diff text", "lapis-pm")

        assert result["verdict"] == "uncertain"
        assert "scope_id" in result
        assert "freshness_stamp" in result
        assert "citations" in result

    def test_returns_dict_not_dataclass(self):
        """run_corroboration_pass always returns a plain dict (JSON-serialisable)."""
        corr_result = CorroborationResult(
            verdict="clean",
            claim="test",
            citations=[],
            freshness_stamp="2026-04-30T12:00:00+00:00",
            scope_id="repo:test",
        )
        with (
            patch("lapis_pm.corroboration_adapter.LapisPMReviewerAdapter.retrieve",
                  return_value=[]),
            patch("lapis_pm.corroboration_adapter.LapisPMReviewerAdapter.score",
                  return_value=corr_result),
        ):
            result = run_corroboration_pass("diff", "test")

        assert isinstance(result, dict)
        # Must be JSON-serialisable
        json.dumps(result)

    def test_both_nodes_failing_yields_node2_unavailable_never_agree(self):
        """D2 regression — the single most important test in this unit.

        When both node1 and node2 scoring fail, cross_node_divergence must be
        'node2_unavailable', never 'agree'. Before the fix, both failure paths
        aliased one shared `_uncertain` instance, so result_n1 is result_n2 and
        the divergence test (verdict == verdict) was trivially true — two dead
        nodes recorded as agreeing with each other.
        """
        from lapis_pm.corroboration_adapter import _make_uncertain as real_make_uncertain

        captured: list[CorroborationResult] = []

        def spy_make_uncertain(repo, notes):
            r = real_make_uncertain(repo, notes)
            captured.append(r)
            return r

        with (
            patch("lapis_pm.corroboration_adapter.LapisPMReviewerAdapter.retrieve",
                  return_value=[_IdentifierSubstrate("tick", [], [])]),
            patch("lapis_pm.corroboration_adapter.LapisPMReviewerAdapter.score",
                  side_effect=RuntimeError("both nodes dead")),
            patch("lapis_pm.corroboration_adapter._make_uncertain",
                  side_effect=spy_make_uncertain),
        ):
            result = run_corroboration_pass(FIXTURE_DIFF_CLEAN, "lapis-pm")

        # Object identity, not equality: two independent failure results.
        assert len(captured) == 2, f"expected 2 fresh uncertain results, got {len(captured)}"
        result_n1, result_n2 = captured
        assert result_n1 is not result_n2, "result_n1 and result_n2 must not be the same object"

        # The regression: this must never read "agree".
        assert result["cross_node_divergence"] == "node2_unavailable"
        assert result["cross_node_divergence"] != "agree"

        # Both legs report themselves failed, independently.
        assert result["leg_status"] == "pass_failed"
        assert result["node2_corroboration"]["leg_status"] == "pass_failed"


# ---------------------------------------------------------------------------
# Thinking-model parse + truncation loudness
# (lapis-pm-corroboration-thinking-parse-and-truncation-loudness-v0)
# ---------------------------------------------------------------------------

class TestThinkingModelParseAndTruncation:
    """Regression coverage for the real reproduction (2026-08-12 13:15 PT,
    GravityWell :8081): content=None, reasoning populated, finish_reason
    "length", HTTP 200. Pre-fix this raised AttributeError on `.strip()`
    inside the bare `except Exception`, which reported leg_status=
    "substrate_unavailable" — indistinguishable from a genuinely dead node."""

    def _make_thinking_response(self, *, content, reasoning, finish_reason, model="gravitywell-27b"):
        return {
            "model": model,
            "choices": [{
                "message": {"content": content, "reasoning": reasoning},
                "finish_reason": finish_reason,
            }],
        }

    def test_none_content_with_reasoning_and_length_does_not_raise_and_is_truncated(self):
        """DoD #1: the real reproduction — usable text, truncated=True, no
        AttributeError."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        llm_resp = self._make_thinking_response(
            content=None,
            reasoning="the model reasons here but never emits content...",
            finish_reason="length",
        )

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: llm_resp,
                raise_for_status=lambda: None,
            )
            result = adapter.score("some diff", substrates, "lapis-pm")

        # Must not raise (the pre-fix AttributeError path is gone) and must
        # report the distinct truncated classification, not substrate_unavailable.
        assert result.verdict == "uncertain"
        assert result.leg_status == "truncated"
        assert result.leg_status != "substrate_unavailable"
        assert "truncat" in (result.notes or "").lower()

    def test_truncated_is_visibly_different_from_substrate_unavailable(self):
        """DoD #4: a reader must be able to tell 'cut off' from 'unreachable'
        without reading code — assert the claim/leg_status pair differs."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [], [])]

        truncated_resp = self._make_thinking_response(
            content=None, reasoning="partial...", finish_reason="length",
        )
        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200, json=lambda: truncated_resp, raise_for_status=lambda: None,
            )
            truncated_result = adapter.score("diff", substrates, "lapis-pm")

        with patch("lapis_pm.corroboration_adapter.node_reachable", return_value=False):
            unavailable_result = adapter.score("diff", substrates, "lapis-pm")

        assert truncated_result.leg_status == "truncated"
        assert unavailable_result.leg_status == "substrate_unavailable"
        assert truncated_result.leg_status != unavailable_result.leg_status
        assert truncated_result.claim != unavailable_result.claim

    def test_content_only_stop_still_parses_normally(self):
        """finish_reason=stop, content populated — ordinary success path unaffected."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        llm_resp = {
            "model": "gravitywell-27b",
            "choices": [{
                "message": {"content": json.dumps({
                    "verdict": "clean",
                    "claims": [{"identifier": "tick", "drift_class": "none", "notes": "ok"}],
                    "summary": "all clear",
                })},
                "finish_reason": "stop",
            }],
        }

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200, json=lambda: llm_resp, raise_for_status=lambda: None,
            )
            result = adapter.score("diff", substrates, "lapis-pm")

        assert result.verdict == "clean"
        assert result.leg_status == "ok"

    def test_reasoning_content_llamacpp_shape_also_parses(self):
        """llama.cpp's reasoning_content field is honoured as a fallback too."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        payload = json.dumps({
            "verdict": "clean",
            "claims": [{"identifier": "tick", "drift_class": "none", "notes": "ok"}],
            "summary": "all clear",
        })
        llm_resp = {
            "model": "local-llamacpp",
            "choices": [{
                "message": {"content": "", "reasoning_content": payload},
                "finish_reason": "stop",
            }],
        }

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200, json=lambda: llm_resp, raise_for_status=lambda: None,
            )
            result = adapter.score("diff", substrates, "lapis-pm")

        assert result.verdict == "clean"
        assert result.leg_status == "ok"

    def test_no_usable_text_anywhere_is_substrate_unavailable_not_crash(self):
        """content, reasoning, and reasoning_content all empty, finish_reason
        stop — a distinct condition, reported as substrate_unavailable, never
        an unhandled exception."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [], [])]

        llm_resp = {
            "model": "gravitywell-27b",
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
        }

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post") as mock_post,
        ):
            mock_post.return_value = MagicMock(
                status_code=200, json=lambda: llm_resp, raise_for_status=lambda: None,
            )
            result = adapter.score("diff", substrates, "lapis-pm")

        assert result.leg_status == "substrate_unavailable"

    def test_exception_message_included_not_just_class_name(self):
        """Honest diagnostics: the except path carries the exception message,
        not just type(exc).__name__ — a bare class name is what hid this
        defect for over a week."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=ConnectionError("connection refused by peer")),
        ):
            result = adapter.score("diff", substrates, "lapis-pm")

        assert result.leg_status == "substrate_unavailable"
        assert "ConnectionError" in result.notes
        assert "connection refused by peer" in result.notes

    def test_max_tokens_raised_and_grammar_constraint_present(self):
        """Scope 5: max_tokens raised well above 512, json_schema/strict
        grammar adopted — copied from local_reviewer_witness.py's proven shape."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        llm_resp = {
            "model": "gravitywell-27b",
            "choices": [{
                "message": {"content": json.dumps({
                    "verdict": "clean",
                    "claims": [{"identifier": "tick", "drift_class": "none", "notes": "ok"}],
                    "summary": "ok",
                })},
                "finish_reason": "stop",
            }],
        }

        captured_bodies: list[dict] = []

        def capture_post(url, *, json=None, timeout=None):
            captured_bodies.append(json)
            return MagicMock(status_code=200, json=lambda: llm_resp, raise_for_status=lambda: None)

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=capture_post),
        ):
            adapter.score("diff", substrates, "lapis-pm")

        assert captured_bodies, "expected httpx.post to be called"
        body = captured_bodies[0]
        assert body["max_tokens"] > 512
        assert body["max_tokens"] == 4096
        assert body.get("response_format", {}).get("type") == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True

    def test_grammar_rejected_degrades_to_plain_call_never_hard_fails(self):
        """Risk: a seat that rejects json_schema (HTTP 4xx on the grammar
        attempt) must degrade to a plain call, never hard-fail — mirrors
        local_reviewer_witness.py's HTTP-4xx retry precedent."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        llm_resp = {
            "model": "node2-mlx",
            "choices": [{
                "message": {"content": json.dumps({
                    "verdict": "clean",
                    "claims": [{"identifier": "tick", "drift_class": "none", "notes": "ok"}],
                    "summary": "ok",
                })},
                "finish_reason": "stop",
            }],
        }

        bad_resp = MagicMock(status_code=422, raise_for_status=lambda: None, json=lambda: {})
        good_resp = MagicMock(status_code=200, json=lambda: llm_resp, raise_for_status=lambda: None)

        call_count = [0]

        def side_effect(url, *, json=None, timeout=None):
            call_count[0] += 1
            return bad_resp if call_count[0] == 1 else good_resp

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=side_effect),
        ):
            result = adapter.score("diff", substrates, "lapis-pm")

        assert call_count[0] == 2, "expected a retry without grammar after the 4xx"
        assert result.verdict == "clean"
        assert result.leg_status == "ok"
