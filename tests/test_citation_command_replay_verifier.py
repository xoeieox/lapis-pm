"""Tests for the deterministic command-replay citation verifier
(lapis-pm-citation-command-replay-verifier-v0).

Covers: allowlist accept/reject (incl. path-containment), the per-program target
extraction table, the Relevance Gap cross-check, the Coincidence Trap line-granularity
check, the command_args-shape validation gate, grep's nonzero-exit-is-not-failure
handling, target_repo_path unavailability, aggregate wall-clock bounding, brief
rendering of the three command_verified states, and non-interference with
combined_recommendation.
"""
from __future__ import annotations

import subprocess
import time

import pytest

from lapis_pm.spec_review import (
    _citation_args_shape_valid,
    _citation_command_allowlisted,
    _citation_extract_targets,
    _verify_citation_commands,
    _format_issue_line,
    format_brief,
    SpecReviewBrief,
)


# ---------------------------------------------------------------------------
# Fixtures — a small real repo tree to run real commands against
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path):
    (tmp_path / "bar.py").write_text("line1\nline2\nFOO_BAR marker\n" + ("pad\n" * 40), encoding="utf-8")
    (tmp_path / "foo.py").write_text(
        "\n".join(f"line{n}" for n in range(1, 42)) + "\nFOO_BAR = 1\n", encoding="utf-8"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "nested.py").write_text("nested content\n", encoding="utf-8")
    agents_core = tmp_path / "agents_core"
    agents_core.mkdir()
    (agents_core / "llm.py").write_text("def call():\n    pass\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# 2. Allowlist check — accept / reject
# ---------------------------------------------------------------------------

def test_allowlist_accepts_grep_with_flags(repo):
    assert _citation_command_allowlisted("grep", ["-n", "foo", "bar.py"], str(repo))


def test_allowlist_accepts_git_grep(repo):
    assert _citation_command_allowlisted("git", ["grep", "foo"], str(repo))


def test_allowlist_accepts_cat(repo):
    assert _citation_command_allowlisted("cat", ["bar.py"], str(repo))


def test_allowlist_accepts_nested_relative_path(repo):
    assert _citation_command_allowlisted(
        "grep", ["-n", "foo", "agents_core/llm.py"], str(repo)
    )


def test_allowlist_rejects_non_whitelisted_program(repo):
    assert not _citation_command_allowlisted("sed", ["-n", "1p", "bar.py"], str(repo))


def test_allowlist_rejects_git_log(repo):
    assert not _citation_command_allowlisted("git", ["log", "-1"], str(repo))


def test_allowlist_rejects_git_bare(repo):
    assert not _citation_command_allowlisted("git", [], str(repo))


def test_allowlist_rejects_dotdot_escape(repo):
    assert not _citation_command_allowlisted("cat", ["../../etc/passwd"], str(repo))


def test_allowlist_rejects_absolute_path_escape(repo):
    assert not _citation_command_allowlisted("cat", ["/etc/passwd"], str(repo))


def test_allowlist_rejects_symlink_escape(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.txt"
    secret.write_text("nope\n", encoding="utf-8")
    link = repo / "escape_link"
    link.symlink_to(secret)
    assert not _citation_command_allowlisted("cat", ["escape_link"], str(repo))


def test_command_args_never_rejoined_and_reparsed():
    """Regression stand-in for the shell-metacharacter-injection case: since the
    schema is structured (program, args as list[str]) there is no tokenization
    step anywhere in the verifier, so command_args entries are never joined into
    a single string and re-split."""
    import inspect
    from lapis_pm import spec_review

    src = inspect.getsource(spec_review._verify_citation_commands)
    assert ".join(command_args" not in src
    assert "shlex" not in src
    assert " + \" \" + " not in src


# ---------------------------------------------------------------------------
# 2c. command_args shape validation — schema-validation time, before allowlist
# ---------------------------------------------------------------------------

def test_command_args_bare_string_invalid():
    assert not _citation_args_shape_valid("grep foo bar.py")


def test_command_args_nested_list_invalid():
    assert not _citation_args_shape_valid([["grep"], "foo"])


def test_command_args_non_string_element_invalid():
    assert not _citation_args_shape_valid(["foo", 42])


def test_command_args_plain_list_valid():
    assert _citation_args_shape_valid(["-n", "foo", "bar.py"])


def test_verify_bad_command_args_shape_not_checked(repo):
    issues = [{
        "severity": "high", "citation": "bar.py",
        "command_program": "grep", "command_args": "grep foo bar.py",
        "command_output": "whatever",
    }]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["command_verified"] == "not_checked"


# ---------------------------------------------------------------------------
# 2b. Per-program extraction table
# ---------------------------------------------------------------------------

def test_extract_cat_multiple_files():
    assert _citation_extract_targets("cat", ["file1", "file2"]) == ["file1", "file2"]


def test_extract_grep_directory_candidate():
    assert _citation_extract_targets("grep", ["-r", "-n", "pattern", "./src"]) == ["./src"]


def test_extract_git_show_colon_split():
    assert _citation_extract_targets("git", ["show", "v1.0:foo.py"]) == ["foo.py"]


def test_extract_git_show_no_colon():
    assert _citation_extract_targets("git", ["show", "foo.py"]) == ["foo.py"]


def test_extract_find_search_root():
    assert _citation_extract_targets("find", ["./src", "-name", "*.py"]) == ["./src"]


# ---------------------------------------------------------------------------
# 2a. Relevance Gap — target mismatch fails closed even on byte-match
# ---------------------------------------------------------------------------

def test_relevance_gap_target_mismatch_fails_even_on_byte_match(repo):
    real_output = subprocess.run(
        ["cat", "bar.py"], cwd=str(repo), capture_output=True, text=True
    ).stdout
    issues = [{
        "severity": "high",
        "citation": "foo.py:42",
        "command_program": "cat",
        "command_args": ["bar.py"],
        "command_output": real_output,
    }]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["command_verified"] is False
    assert "bar.py" in out[0].get("command_verify_diff", "")


# ---------------------------------------------------------------------------
# 2d. Coincidence Trap — line-granularity check
# ---------------------------------------------------------------------------

def test_line_granularity_grep_confirms_exact_line(repo):
    real = subprocess.run(
        ["grep", "-n", "FOO_BAR", "foo.py"], cwd=str(repo), capture_output=True, text=True
    )
    line_no = int(real.stdout.split(":", 1)[0])
    issues = [{
        "severity": "high",
        "citation": f"foo.py:{line_no}",
        "command_program": "grep",
        "command_args": ["-n", "FOO_BAR", "foo.py"],
        "command_output": real.stdout,
    }]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["command_verified"] is True


def test_line_granularity_cat_full_file_not_checked_even_if_substring_present(repo):
    real = subprocess.run(
        ["cat", "foo.py"], cwd=str(repo), capture_output=True, text=True
    )
    issues = [{
        "severity": "high",
        "citation": "foo.py:42",
        "command_program": "cat",
        "command_args": ["foo.py"],
        "command_output": real.stdout,
    }]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["command_verified"] == "not_checked"


def test_line_granularity_skipped_when_citation_has_no_line(repo):
    real = subprocess.run(
        ["cat", "foo.py"], cwd=str(repo), capture_output=True, text=True
    )
    issues = [{
        "severity": "high",
        "citation": "foo.py",
        "command_program": "cat",
        "command_args": ["foo.py"],
        "command_output": real.stdout,
    }]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["command_verified"] is True


# ---------------------------------------------------------------------------
# 2e. grep exit code 1 (no match) is not an execution failure
# ---------------------------------------------------------------------------

def test_grep_no_match_empty_claim_verifies_true(repo):
    issues = [{
        "severity": "med",
        "citation": "bar.py",
        "command_program": "grep",
        "command_args": ["-n", "NOPE_NOT_PRESENT", "bar.py"],
        "command_output": "",
    }]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["command_verified"] is True


# ---------------------------------------------------------------------------
# 2f. target_repo_path unavailable -> every issue not_checked, never raises
# ---------------------------------------------------------------------------

def test_target_repo_path_none_all_not_checked():
    issues = [
        {"severity": "high", "citation": "a.py:1", "command_program": "cat",
         "command_args": ["a.py"], "command_output": "x"},
        {"severity": "med", "citation": "b.py:2", "command_program": "grep",
         "command_args": ["-n", "x", "b.py"], "command_output": "x"},
    ]
    out = _verify_citation_commands(issues, None)
    assert all(i["command_verified"] == "not_checked" for i in out)


def test_target_repo_path_nonexistent_all_not_checked():
    issues = [{"severity": "high", "citation": "a.py:1", "command_program": "cat",
               "command_args": ["a.py"], "command_output": "x"}]
    out = _verify_citation_commands(issues, "/no/such/dir/at/all")
    assert out[0]["command_verified"] == "not_checked"


# ---------------------------------------------------------------------------
# 3/4/5. true / false / missing-fields outcomes
# ---------------------------------------------------------------------------

def test_matching_command_verifies_true(repo):
    real = subprocess.run(["cat", "bar.py"], cwd=str(repo), capture_output=True, text=True)
    issues = [{
        "severity": "high", "citation": "bar.py",
        "command_program": "cat", "command_args": ["bar.py"],
        "command_output": real.stdout,
    }]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["command_verified"] is True


def test_mismatching_command_verifies_false_and_issue_kept(repo):
    issues = [{
        "severity": "high", "citation": "bar.py", "note": "original note",
        "command_program": "cat", "command_args": ["bar.py"],
        "command_output": "this is definitely not the real content",
    }]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["command_verified"] is False
    assert out[0]["note"] == "original note"
    assert "command_verify_diff" in out[0]


def test_missing_command_fields_not_checked_no_exception(repo):
    issues = [{"severity": "high", "citation": "bar.py", "note": "no command fields"}]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["command_verified"] == "not_checked"


def test_low_severity_issue_untouched(repo):
    issues = [{"severity": "low", "citation": "bar.py", "note": "weak locator"}]
    out = _verify_citation_commands(issues, str(repo))
    assert "command_verified" not in out[0]


# ---------------------------------------------------------------------------
# 6. Never mutates other issue fields / never touches recommendation inputs
# ---------------------------------------------------------------------------

def test_verification_never_mutates_severity_or_note(repo):
    issues = [{
        "severity": "high", "citation": "bar.py", "note": "keep me",
        "command_program": "cat", "command_args": ["bar.py"], "command_output": "wrong",
    }]
    out = _verify_citation_commands(issues, str(repo))
    assert out[0]["severity"] == "high"
    assert out[0]["note"] == "keep me"


# ---------------------------------------------------------------------------
# 7. Aggregate wall-clock bound
# ---------------------------------------------------------------------------

def test_aggregate_budget_bounds_total_time(repo, monkeypatch):
    from lapis_pm import spec_review

    monkeypatch.setattr(spec_review, "_CITATION_VERIFY_AGGREGATE_BUDGET_S", 0.05)

    real_run = subprocess.run

    def _slow_run(*args, **kwargs):
        time.sleep(0.03)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(spec_review.subprocess, "run", _slow_run)

    issues = [
        {"severity": "high", "citation": f"bar.py:{n}", "command_program": "cat",
         "command_args": ["bar.py"], "command_output": "x"}
        for n in range(30)
    ]
    start = time.monotonic()
    out = spec_review._verify_citation_commands(issues, str(repo))
    elapsed = time.monotonic() - start

    assert elapsed < 5.0
    assert any(i["command_verified"] == "not_checked" for i in out)


# ---------------------------------------------------------------------------
# Brief rendering — three distinct states
# ---------------------------------------------------------------------------

def test_render_confirmed_neutral_weight():
    line = _format_issue_line({"severity": "high", "citation": "a.py", "note": "n",
                                "command_verified": True})
    assert "confirmed" in line
    assert "CONTRADICTED" not in line


def test_render_contradicted_high_contrast_with_diff():
    line = _format_issue_line({
        "severity": "high", "citation": "a.py", "note": "n",
        "command_verified": False, "command_verify_diff": "claimed=X actual=Y",
    })
    assert "CONTRADICTED" in line
    assert "claimed=X actual=Y" in line


def test_render_not_checked_distinct_neutral_state():
    line = _format_issue_line({"severity": "high", "citation": "a.py", "note": "n",
                                "command_verified": "not_checked"})
    assert "not checked" in line
    assert "CONTRADICTED" not in line
    assert "confirmed" not in line


def test_render_absent_field_unchanged():
    line = _format_issue_line({"severity": "low", "citation": "a.py", "note": "n"})
    assert "command-verified" not in line


def test_format_brief_renders_command_verified_states(tmp_path):
    brief = SpecReviewBrief(
        spec_path=tmp_path / "spec.md",
        target_id="tid",
        repo="lapis-pm",
        council_status="resolved",
        council_landing="landing",
        council_open_questions=[],
        council_confidence="converged",
        council_positions=[],
        council_run_id="run-1",
        elapsed_s=1.0,
        combined_recommendation="proceed-to-bind",
        reference_verdict="fixable",
        reference_issues=[
            {"severity": "high", "citation": "a.py:1", "note": "n1", "command_verified": True},
            {"severity": "med", "citation": "b.py:2", "note": "n2", "command_verified": False,
             "command_verify_diff": "claimed=X actual=Y"},
            {"severity": "med", "citation": "c.py:3", "note": "n3", "command_verified": "not_checked"},
        ],
    )
    out = format_brief(brief)
    assert "confirmed" in out
    assert "CONTRADICTED" in out
    assert "claimed=X actual=Y" in out
    assert "not checked" in out
