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

        with patch("httpx.post") as mock_post:
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

        with patch("httpx.post") as mock_post:
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

        with patch("httpx.post") as mock_post:
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

        with patch("httpx.post") as mock_post:
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

        with patch("httpx.post", side_effect=ConnectionError("refused")):
            result = adapter.score("some diff", substrates, "lapis-pm")

        assert result.verdict == "uncertain"
        assert result.drift_class is None
        assert "LLM unavailable" in (result.notes or "")


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
