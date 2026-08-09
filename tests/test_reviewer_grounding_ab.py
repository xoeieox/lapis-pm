"""Tests for scripts/reviewer_grounding_ab.py — lapis-pm-reviewer-absence-grounding-rule-v0.

Covers DoD 6 (the evidence-fabrication poison-pill check) at the unit level:
classify_evidence's four states, the read-only command allowlist, and the
comparison-row synthesis. Does not exercise the live-dispatch path (_run_arm)
— that needs a real queue + real PR and is covered by the "pm-live-test" DoD
item as a manual run, not a pytest.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "reviewer_grounding_ab.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reviewer_grounding_ab", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    # The script inserts hardcoded sys.path entries and imports lapis_pm/agents_core
    # at module scope; both are already importable in this test environment.
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


class TestParseCommand:

    def test_allows_grep(self, mod):
        assert mod._parse_command("grep -rn foo .") == ["grep", "-rn", "foo", "."]

    def test_allows_git_grep(self, mod):
        assert mod._parse_command("git grep -n foo") == ["git", "grep", "-n", "foo"]

    def test_rejects_disallowed_git_subcommand(self, mod):
        """git checkout/push/commit etc. must never be re-runnable."""
        assert mod._parse_command("git push origin main") is None
        assert mod._parse_command("git commit -am pwned") is None

    def test_rejects_disallowed_binary(self, mod):
        assert mod._parse_command("rm -rf /") is None
        assert mod._parse_command("curl http://evil") is None

    def test_strips_dollar_prefix(self, mod):
        assert mod._parse_command("$ grep foo bar.py") == ["grep", "foo", "bar.py"]

    def test_unparseable_shlex_returns_none(self, mod):
        assert mod._parse_command('grep "unterminated') is None

    def test_empty_line_returns_none(self, mod):
        assert mod._parse_command("   ") is None


class TestClassifyEvidence:

    def test_no_evidence_when_missing(self, mod, tmp_path):
        state, _ = mod.classify_evidence(None, str(tmp_path))
        assert state == "no_evidence"

    def test_no_evidence_when_blank(self, mod, tmp_path):
        state, _ = mod.classify_evidence("   ", str(tmp_path))
        assert state == "no_evidence"

    def test_search_failed_when_command_unparseable(self, mod, tmp_path):
        state, detail = mod.classify_evidence("rm -rf /\nsome claimed output", str(tmp_path))
        assert state == "search_failed"
        assert "could not be parsed" in detail

    def test_search_failed_when_rerun_exits_nonzero(self, mod, tmp_path):
        # grep with no matches on a nonexistent file path exits 2.
        state, _ = mod.classify_evidence("grep -rn nosuchthing /nonexistent/path", str(tmp_path))
        assert state == "search_failed"

    def test_corroborated_when_output_matches(self, mod, tmp_path):
        (tmp_path / "foo.py").write_text("def handle_foo():\n    pass\n")
        evidence = "grep -rn handle_foo .\nfoo.py:1:def handle_foo():"
        state, _ = mod.classify_evidence(evidence, str(tmp_path))
        assert state == "corroborated"

    def test_fabricated_when_output_does_not_match(self, mod, tmp_path):
        (tmp_path / "foo.py").write_text("def handle_foo():\n    pass\n")
        evidence = "grep -rn handle_foo .\nfoo.py:99:this line does not exist"
        state, detail = mod.classify_evidence(evidence, str(tmp_path))
        assert state == "fabricated"
        assert "substring" in detail

    def test_fabricated_when_clean_claim_but_rerun_has_hits(self, mod, tmp_path):
        (tmp_path / "foo.py").write_text("def handle_foo():\n    pass\n")
        # Command line only, no claimed-output text — implicitly claims "no hits".
        evidence = "grep -rn handle_foo ."
        state, _ = mod.classify_evidence(evidence, str(tmp_path))
        assert state == "fabricated"

    def test_corroborated_when_clean_claim_and_rerun_empty(self, mod, tmp_path):
        (tmp_path / "foo.py").write_text("def unrelated():\n    pass\n")
        evidence = "grep -rn nosuchsymbol foo.py"
        state, _ = mod.classify_evidence(evidence, str(tmp_path))
        assert state == "corroborated"

    def test_not_an_exact_line_count_match(self, mod, tmp_path):
        """DoD 6: substring check, not exact match — extra real output around
        the quoted line must not be scored as fabricated."""
        (tmp_path / "foo.py").write_text("def handle_foo():\n    pass\n")
        (tmp_path / "bar.py").write_text("def handle_foo():\n    pass\n")
        evidence = "grep -rn handle_foo .\nfoo.py:1:def handle_foo():"
        state, _ = mod.classify_evidence(evidence, str(tmp_path))
        assert state == "corroborated"


class TestComparisonRow:

    def _row(self, **overrides):
        base = {
            "task_id": "t1", "queue_state": "completed", "verdict": "clean",
            "issue_count": 0, "absence_issue_count": 0,
            "corroborated": 0, "fabricated": 0, "search_failed": 0, "no_evidence": 0,
            "error": "",
        }
        base.update(overrides)
        return base

    def test_sums_fabricated_across_both_arms(self, mod):
        off_row = self._row(fabricated=1)
        on_row = self._row(fabricated=2)
        cmp_row = mod._comparison_row("tid", 42, "reviewer_fresh", off_row, on_row)
        assert cmp_row["fabricated"] == 3
        assert cmp_row["row_type"] == "comparison"

    def test_zero_fabricated_is_clean(self, mod):
        off_row = self._row()
        on_row = self._row()
        cmp_row = mod._comparison_row("tid", 42, "reviewer_fresh", off_row, on_row)
        assert cmp_row["fabricated"] == 0

    def test_deltas_computed(self, mod):
        off_row = self._row(issue_count=3, absence_issue_count=1)
        on_row = self._row(issue_count=1, absence_issue_count=0)
        cmp_row = mod._comparison_row("tid", 42, "reviewer_fresh", off_row, on_row)
        assert cmp_row["issue_count"] == -2
        assert cmp_row["absence_issue_count"] == -1


class TestCsvFieldnamesCoverAllRowTypes(object):

    def test_arm_row_keys_subset_of_fieldnames(self, mod):
        assert set(mod.CSV_FIELDNAMES) >= {
            "row_type", "target_id", "pr_number", "seat", "grounding",
            "task_id", "queue_state", "verdict", "issue_count",
            "absence_issue_count", "corroborated", "fabricated",
            "search_failed", "no_evidence", "error", "raw_path",
        }
