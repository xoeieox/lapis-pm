"""Tests for the hardened _read_verdict_from_output pipeline and related surfaces.

Covers:
- truncated-head fixture  → parse_failed envelope (head genuinely missing)
- prose-preamble fixture  → real clean verdict recovered via fenced extraction
- strict JSON-only        → direct json.loads path
- fenced JSON no preamble → fenced extraction path
- embedded code-span trap → fence wins over bracket-counter false-positive
- two competing candidates → fenced JSON wins over small bracket-counter hit
- _combined_recommendation with reference_verdict="parse_failed" → "parse_failed"
- _build_brief with parse_failed → rendered brief contains ## Parse error section
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lapis_pm.spec_review import (
    SpecReviewBrief,
    _build_brief,
    _combined_recommendation,
    _read_verdict_from_output,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "spec_review"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_output(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "output.md"
    p.write_text(content, encoding="utf-8")
    return p


def _sonnet_raw(verdict: str, issues: list | None = None, confidence: float = 0.9, **kw) -> dict:
    d = {
        "status": "processed",
        "verdict": verdict,
        "issues": issues or [],
        "confidence": confidence,
        "run_id": f"sonnet-{verdict}",
    }
    d.update(kw)
    return d


def _council_raw(status: str = "resolved") -> dict:
    return {
        "status": status,
        "landing": "test landing",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": f"council-{status}",
    }


# ---------------------------------------------------------------------------
# 1. Truncated-head fixture → parse_failed envelope
# ---------------------------------------------------------------------------

def test_truncated_head_fixture_returns_parse_failed(tmp_path):
    """Head is genuinely missing (outermost { was sliced off) — must return parse_failed envelope."""
    fixture = _FIXTURES / "truncated_head_20260511_100406.md"
    content = fixture.read_text(encoding="utf-8")
    p = _write_output(tmp_path, content)

    result = _read_verdict_from_output(p)

    assert result["verdict"] == "parse_failed"
    assert "parse_error" in result
    pe = result["parse_error"]
    assert pe["file_size"] > 0
    assert pe["head"].startswith("rief emission")


# ---------------------------------------------------------------------------
# 2. Prose-preamble fixture → real Opus verdict recovered
# ---------------------------------------------------------------------------

def test_prose_preamble_fixture_recovers_clean_verdict(tmp_path):
    """Prose before the ```json fence must not prevent extraction of the real verdict."""
    fixture = _FIXTURES / "prose_preamble_20260511_104429.md"
    content = fixture.read_text(encoding="utf-8")
    p = _write_output(tmp_path, content)

    result = _read_verdict_from_output(p)

    assert result["verdict"] == "clean"
    assert "parse_error" not in result


# ---------------------------------------------------------------------------
# 3. Strict JSON-only (no fence, no prose)
# ---------------------------------------------------------------------------

def test_strict_json_only(tmp_path):
    payload = {"verdict": "fixable", "issues": [{"severity": "low", "note": "x"}], "confidence": 0.7}
    p = _write_output(tmp_path, json.dumps(payload))

    result = _read_verdict_from_output(p)

    assert result["verdict"] == "fixable"
    assert result["confidence"] == 0.7


# ---------------------------------------------------------------------------
# 4. Fenced JSON, no preamble
# ---------------------------------------------------------------------------

def test_fenced_json_no_preamble(tmp_path):
    payload = {"verdict": "clean", "issues": [], "confidence": 0.95}
    p = _write_output(tmp_path, f"```json\n{json.dumps(payload)}\n```")

    result = _read_verdict_from_output(p)

    assert result["verdict"] == "clean"
    assert result["confidence"] == 0.95


# ---------------------------------------------------------------------------
# 5. Embedded code-span trap — bracket-counter must not grab {invalid} fragment
# ---------------------------------------------------------------------------

def test_embedded_code_span_trap(tmp_path):
    """Prose with `{invalid: dict[key]={x: y}}` must not fool the bracket-counter."""
    payload = {"verdict": "clean", "issues": [], "confidence": 0.9}
    content = (
        "Here is the result: `{invalid: dict[key]={x: y}}` and then proper:\n"
        f"```json\n{json.dumps(payload)}\n```"
    )
    p = _write_output(tmp_path, content)

    result = _read_verdict_from_output(p)

    assert result["verdict"] == "clean", (
        f"got verdict={result['verdict']!r}; parser grabbed the code-span fragment instead of the fence"
    )


# ---------------------------------------------------------------------------
# 6. Two competing JSON candidates — fenced wins over bracket-counter hit
# ---------------------------------------------------------------------------

def test_two_competing_candidates_fence_wins(tmp_path):
    """A small nested JSON object before the fenced verdict — fenced match must win."""
    inner = json.dumps({"type": "context", "value": 42})
    verdict = {"verdict": "needs-human", "issues": [], "confidence": 0.6}
    content = (
        f"Preamble context: {inner}\n"
        f"Actual verdict:\n```json\n{json.dumps(verdict)}\n```"
    )
    p = _write_output(tmp_path, content)

    result = _read_verdict_from_output(p)

    assert result["verdict"] == "needs-human", (
        f"got verdict={result['verdict']!r}; parser picked the small inner object"
    )


# ---------------------------------------------------------------------------
# 7. _combined_recommendation: parse_failed → "parse_failed" (not "incomplete")
# ---------------------------------------------------------------------------

def test_combined_recommendation_parse_failed():
    result = _combined_recommendation(
        reference_verdict="parse_failed",
        reference_issues=[],
        council_status="resolved",
        council_positions=[],
    )
    assert result == "parse_failed", f"expected 'parse_failed', got {result!r}"


# ---------------------------------------------------------------------------
# 8. _build_brief: parse_failed envelope → rendered brief has ## Parse error
# ---------------------------------------------------------------------------

def test_build_brief_parse_failed_renders_parse_error_section(tmp_path):
    from lapis_pm.spec_review import format_brief

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    sonnet = _sonnet_raw(
        "parse_failed",
        confidence=0.0,
        parse_error={
            "file_size": 1976,
            "head": "Confirmed: for a generic failure...",
            "tail": "...WARN: worktree_setup: missing settings.json",
        },
    )

    brief = _build_brief(
        reference_raw=sonnet,
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="test-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
    )

    assert brief.combined_recommendation == "parse_failed"
    assert brief.parse_error is not None
    assert brief.parse_error["file_size"] == 1976

    rendered = format_brief(brief)
    assert "## Parse error" in rendered
    assert "1976 bytes" in rendered
