"""Tests for lapis_pm.absence_ground (lapis-pm-degraded-panel-confidence-and-absence-claims-v0, Leg 2).

Coverage (DoD #8): a refuted absence claim, an unrunnable absence check
staying actionable, plus the underlying phrase/symbol extraction.
"""

from __future__ import annotations

from lapis_pm.absence_ground import (
    extract_symbol,
    ground_issue,
    ground_verdict_issues,
    is_absence_claim,
    symbol_in_file,
)


class TestIsAbsenceClaim:
    def test_matches_not_defined_phrasing(self):
        assert is_absence_claim(
            "`_check_equal` is not defined anywhere in the diff or in the "
            "existing code I read."
        ) is True

    def test_matches_missing_function_definition(self):
        assert is_absence_claim("This is a missing function definition.") is True

    def test_does_not_match_ordinary_finding(self):
        assert is_absence_claim("The error handling here swallows exceptions silently.") is False

    def test_empty_note_is_not_an_absence_claim(self):
        assert is_absence_claim("") is False
        assert is_absence_claim(None) is False

    def test_self_refuting_no_issue_text_not_absence_claim(self):
        assert is_absence_claim("no issue here") is False


class TestExtractSymbol:
    def test_extracts_backtick_quoted_identifier(self):
        assert extract_symbol("`_check_equal` is not defined anywhere") == "_check_equal"

    def test_extracts_last_segment_of_dotted_path(self):
        assert extract_symbol("`agents_core.phala_tee._check_equal` is missing") == "_check_equal"

    def test_no_backtick_identifier_returns_none(self):
        assert extract_symbol("the equal check function is missing") is None

    def test_empty_note_returns_none(self):
        assert extract_symbol("") is None


class TestSymbolInFile:
    def test_finds_whole_word_match(self):
        text = "def _check_equal(a, b):\n    return a == b\n"
        assert symbol_in_file("_check_equal", text) is True

    def test_does_not_match_substring_of_longer_identifier(self):
        text = "def _check_equal_strict(a, b):\n    pass\n"
        assert symbol_in_file("_check_equal", text) is False

    def test_absent_symbol_not_found(self):
        assert symbol_in_file("_check_equal", "def other_fn(): pass\n") is False


class TestGroundIssue:
    def _issue(self, note, path="agents_core/phala_tee.py"):
        return {"severity": "high", "path": path, "note": note}

    def test_non_absence_issue_passes_through_unchanged(self):
        issue = self._issue("The retry loop has an off-by-one error.")
        outcome, annotated = ground_issue(
            issue, "agents-core", 220, "deadbeef",
            fetch_file=lambda repo, path, ref: "irrelevant",
        )
        assert outcome == "not_absence_claim"
        assert annotated == issue

    def test_refuted_when_symbol_found_in_full_file_at_head(self):
        # PR #220 incident replay: `_check_equal` IS defined at
        # agents_core/phala_tee.py:427 on the PR head; the diff just didn't
        # show it.
        issue = self._issue(
            "`_check_equal` is not defined anywhere in the diff or in the "
            "existing code I read. This is a missing function definition."
        )
        file_text = "...\ndef _check_equal(a, b):\n    return a == b\n...\n"
        outcome, annotated = ground_issue(
            issue, "agents-core", 220, "deadbeef",
            fetch_file=lambda repo, path, ref: file_text,
        )
        assert outcome == "refuted"
        assert annotated["absence_check"] == "refuted"
        assert "agents_core/phala_tee.py" in annotated["refuted_location"]
        assert "deadbeef" in annotated["refuted_location"]

    def test_actionable_when_symbol_genuinely_absent(self):
        issue = self._issue("`_never_defined_anywhere` is not defined anywhere in the code.")
        outcome, annotated = ground_issue(
            issue, "agents-core", 220, "deadbeef",
            fetch_file=lambda repo, path, ref: "def something_else(): pass\n",
        )
        assert outcome == "actionable"
        assert annotated["absence_check"] == "actionable"

    def test_unverified_when_fetch_fails_r2(self):
        # R2: an unrunnable check stays actionable, marked unverified — never
        # a refutation and never a confirmation.
        issue = self._issue("`_check_equal` is not defined anywhere in the diff.")
        outcome, annotated = ground_issue(
            issue, "agents-core", 220, "deadbeef",
            fetch_file=lambda repo, path, ref: None,
        )
        assert outcome == "unverified"
        assert annotated["absence_check"] == "unverified"
        assert "refuted_location" not in annotated

    def test_unverified_when_no_symbol_extractable(self):
        issue = self._issue("The equal-check helper is not defined anywhere.")
        outcome, annotated = ground_issue(
            issue, "agents-core", 220, "deadbeef",
            fetch_file=lambda repo, path, ref: "irrelevant",
        )
        assert outcome == "unverified"

    def test_unverified_when_no_head_sha(self):
        issue = self._issue("`_check_equal` is not defined anywhere.")
        outcome, annotated = ground_issue(
            issue, "agents-core", 220, None,
            fetch_file=lambda repo, path, ref: "irrelevant",
        )
        assert outcome == "unverified"

    def test_unverified_never_confirms_or_refutes(self):
        # Belt-and-suspenders on the R2 language: "never treat an unrunnable
        # check as a refutation, and never as a confirmation."
        issue = self._issue("`_check_equal` is not defined anywhere.")
        outcome, _ = ground_issue(
            issue, "agents-core", 220, "deadbeef",
            fetch_file=lambda repo, path, ref: (_ for _ in ()).throw(Exception("boom")),
        )
        assert outcome not in ("refuted", "actionable")
        assert outcome == "unverified"


class TestGroundVerdictIssues:
    def test_refuted_finding_dropped_from_issues_and_moved_to_refuted_list(self):
        verdict = {
            "verdict": "fixable",
            "confidence": 0.9,
            "issues": [
                {
                    "severity": "high",
                    "path": "agents_core/phala_tee.py",
                    "note": "`_check_equal` is not defined anywhere in the diff "
                            "or in the existing code I read.",
                },
                {"severity": "low", "path": "foo.py", "note": "minor style nit"},
            ],
        }
        file_text = "def _check_equal(a, b):\n    return a == b\n"
        ground_verdict_issues(
            verdict, "agents-core", 220, "deadbeef",
            fetch_file=lambda repo, path, ref: file_text,
        )
        assert len(verdict["issues"]) == 1
        assert verdict["issues"][0]["note"] == "minor style nit"
        assert len(verdict["refuted_absence_findings"]) == 1
        assert verdict["refuted_absence_findings"][0]["absence_check"] == "refuted"

    def test_unverified_absence_claim_stays_in_issues_r2(self):
        verdict = {
            "verdict": "fixable",
            "confidence": 0.9,
            "issues": [
                {"severity": "high", "path": "foo.py", "note": "`bar` is not defined anywhere."},
            ],
        }
        ground_verdict_issues(
            verdict, "agents-core", 220, "deadbeef",
            fetch_file=lambda repo, path, ref: None,  # fetch fails
        )
        assert len(verdict["issues"]) == 1
        assert verdict["issues"][0]["absence_check"] == "unverified"
        assert "refuted_absence_findings" not in verdict

    def test_no_issues_is_a_noop(self):
        verdict = {"verdict": "clean", "confidence": 0.9, "issues": []}
        ground_verdict_issues(verdict, "agents-core", 220, "deadbeef")
        assert verdict["issues"] == []
        assert "refuted_absence_findings" not in verdict
