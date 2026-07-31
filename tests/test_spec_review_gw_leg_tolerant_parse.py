"""Tests for lapis-pm-gw-leg-tolerant-verdict-parse-v0.

The GW reference leg used to parse its verdict with a single bare json.loads()
call while the local leg ran a four-strategy tolerant pipeline. This meant a
model that narrated before emitting its fenced JSON verdict -- the single most
common LLM output shape -- was recorded as verdict=error, findings=0, and the
raw text was never persisted anywhere, so the failure was undiagnosable.

Covers:
- DoD 1/1a/1b: the extracted _extract_verdict_from_text pipeline preserves the
  local leg's behaviour byte-for-byte (pure move), including the head/tail
  200-char caps and errors="replace" encoding tolerance.
- DoD 2: narrate-then-emit fenced JSON is recovered, not read as a failure.
- DoD 3/4: the GW leg parses tolerantly via the same pipeline, and
  distinguishes parse_failed (orchestration couldn't extract a verdict) from
  error (the model itself reported failure).
- DoD 5: gw-raw-output.txt is persisted verbatim on every run; a write
  failure is best-effort and never raises or alters the verdict.
- DoD 6: the rendered brief surfaces parse_failed diagnostics and omits them
  otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lapis_pm.spec_review import (
    SpecReviewBrief,
    _build_brief,
    _extract_verdict_from_text,
    _persist_gw_artifacts,
    _process_gw_verdict,
    _read_verdict_from_output,
    format_brief,
)


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
# DoD 1 — refactor is a pure move; local leg behaviour is unchanged
# ---------------------------------------------------------------------------

def test_read_verdict_matches_extract_text_bare_json(tmp_path):
    payload = {"verdict": "clean", "issues": [], "confidence": 0.9}
    content = json.dumps(payload)
    p = tmp_path / "output.md"
    p.write_text(content, encoding="utf-8")

    assert _read_verdict_from_output(p) == _extract_verdict_from_text(content)


def test_read_verdict_matches_extract_text_fenced_with_narration(tmp_path):
    payload = {"verdict": "fixable", "issues": [{"severity": "low"}], "confidence": 0.8}
    content = f"Let me review this spec.\n\nDone.\n```json\n{json.dumps(payload)}\n```"
    p = tmp_path / "output.md"
    p.write_text(content, encoding="utf-8")

    assert _read_verdict_from_output(p) == _extract_verdict_from_text(content)
    assert _extract_verdict_from_text(content)["verdict"] == "fixable"


def test_read_verdict_matches_extract_text_fenceless_embedded_in_prose(tmp_path):
    payload = {"verdict": "clean", "issues": [], "confidence": 0.95}
    content = f"Here's my verdict: {json.dumps(payload)} — hope that helps."
    p = tmp_path / "output.md"
    p.write_text(content, encoding="utf-8")

    assert _read_verdict_from_output(p) == _extract_verdict_from_text(content)
    assert _extract_verdict_from_text(content)["verdict"] == "clean"


def test_read_verdict_matches_extract_text_unparseable_junk(tmp_path):
    content = "This is not JSON at all, just narration with no verdict anywhere."
    p = tmp_path / "output.md"
    p.write_text(content, encoding="utf-8")

    file_result = _read_verdict_from_output(p)
    text_result = _extract_verdict_from_text(content)

    assert file_result == text_result
    assert file_result["verdict"] == "parse_failed"
    assert "parse_error" in file_result
    assert file_result["parse_error"]["file_size"] == len(content)


# ---------------------------------------------------------------------------
# DoD 1a — head/tail 200-char caps survive the extraction, built at
# envelope-construction time
# ---------------------------------------------------------------------------

def test_head_tail_caps_survive_extraction():
    content = ("x" * 500) + "not json" + ("y" * 500)
    result = _extract_verdict_from_text(content)

    assert result["verdict"] == "parse_failed"
    pe = result["parse_error"]
    assert len(pe["head"]) == 200
    assert len(pe["tail"]) == 200
    assert pe["head"] == content[:200]
    assert pe["tail"] == content[-200:]


def test_head_tail_caps_built_at_construction_not_render_time():
    """The envelope dict already carries the sliced head/tail; nothing at
    render time re-slices from a longer string, so mutating parse_error after
    construction has no effect on a value that was already computed."""
    content = ("a" * 1000) + "still not json"
    result = _extract_verdict_from_text(content)
    pe = result["parse_error"]

    assert pe["head"] == content[:200]
    assert pe["tail"] == content[-200:]
    # The stored strings are already the sliced values, not the full content.
    assert len(pe["head"]) < len(content)
    assert len(pe["tail"]) < len(content)


# ---------------------------------------------------------------------------
# DoD 1b — malformed encoding does not escape the pipeline
# ---------------------------------------------------------------------------

def test_malformed_encoding_yields_parse_failed_not_unicode_decode_error(tmp_path):
    p = tmp_path / "output.md"
    # 0xff is not valid standalone UTF-8; a bare read_text(encoding="utf-8")
    # would raise UnicodeDecodeError here, which is not a JSONDecodeError and
    # would therefore bypass the strategy pipeline entirely.
    p.write_bytes(b'{"verdict": "clean"\xff\xfe garbage')

    result = _read_verdict_from_output(p)

    assert result["verdict"] == "parse_failed"
    assert "parse_error" in result


# ---------------------------------------------------------------------------
# DoD 2 — narrate-then-emit is the exact shape that produced verdict=error
# in gate run 04cb9e1d
# ---------------------------------------------------------------------------

def test_extract_verdict_handles_narrate_then_emit():
    content = (
        "I reviewed the spec in detail, checking the invariants and the "
        "proposed interface changes against the current codebase.\n\n"
        "```json\n"
        '{"verdict": "clean", "issues": [], "confidence": 0.9}\n'
        "```"
    )
    result = _extract_verdict_from_text(content)

    assert result == {"verdict": "clean", "issues": [], "confidence": 0.9}


# ---------------------------------------------------------------------------
# DoD 3 — the GW leg parses tolerantly, verified against the constructed
# SpecReviewBrief (not just the parser's return value)
# ---------------------------------------------------------------------------

def test_gw_leg_parses_narrate_then_emit_into_brief(tmp_path):
    gw_text = (
        "Reviewing the spec now... looks reasonable overall.\n\n"
        "```json\n"
        '{"verdict": "fixable", "issues": [{"severity": "low", "note": "x"}, '
        '{"severity": "low", "note": "y"}], "confidence": 0.85}\n'
        "```"
    )
    gw_verdict, gw_findings_count, gw_parse_error = _process_gw_verdict(gw_text)

    assert gw_verdict == "fixable"
    assert gw_findings_count == 2
    assert gw_parse_error is None

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="test-tid",
        repo="lapis-pm",
        elapsed_s=5.0,
        gw_verdict=gw_verdict,
        gw_ran=True,
        gw_findings_count=gw_findings_count,
        elapsed_gw=12.3,
        gw_transcript_ref="/srv/lapis/spec-review-artifacts/run/gw-transcript.json",
        gw_parse_error=gw_parse_error,
        gw_raw_output_ref="/srv/lapis/spec-review-artifacts/run/gw-raw-output.txt",
    )

    assert brief.gw_verdict == "fixable"
    assert brief.gw_findings_count == 2
    assert brief.gw_verdict != "error"


# ---------------------------------------------------------------------------
# DoD 4 — parse_failed is distinguished from error
# ---------------------------------------------------------------------------

def test_process_gw_verdict_unparseable_text_is_parse_failed():
    gw_text = "The spec looks fine to me, no JSON here though."
    gw_verdict, gw_findings_count, gw_parse_error = _process_gw_verdict(gw_text)

    assert gw_verdict == "parse_failed"
    assert gw_findings_count == 0
    assert gw_parse_error is not None
    assert "head" in gw_parse_error and "tail" in gw_parse_error


def test_process_gw_verdict_model_reported_failure_is_error():
    gw_text = json.dumps({"verdict": "error", "issues": [], "confidence": 0.0})
    gw_verdict, gw_findings_count, gw_parse_error = _process_gw_verdict(gw_text)

    assert gw_verdict == "error"
    assert gw_findings_count == 0
    assert gw_parse_error is None


# ---------------------------------------------------------------------------
# DoD 5 — gw-raw-output.txt is persisted verbatim on every run; a write
# failure is best-effort and never raises or alters the verdict
# ---------------------------------------------------------------------------

def test_persist_gw_artifacts_writes_raw_output_verbatim(tmp_path):
    gw_text = "narration\n```json\n{\"verdict\": \"clean\", \"issues\": []}\n```"
    artifacts_dir = tmp_path / "run-id"

    raw_ref, transcript_ref = _persist_gw_artifacts(artifacts_dir, gw_text, [{"tool": "grep"}])

    raw_path = artifacts_dir / "gw-raw-output.txt"
    assert raw_path.exists()
    assert raw_path.read_text(encoding="utf-8") == gw_text
    assert raw_ref == str(raw_path.resolve())

    transcript_path = artifacts_dir / "gw-transcript.json"
    assert transcript_path.exists()
    assert json.loads(transcript_path.read_text(encoding="utf-8")) == [{"tool": "grep"}]
    assert transcript_ref == str(transcript_path.resolve())


def test_persist_gw_artifacts_writes_raw_output_even_on_successful_parse(tmp_path):
    """Change 3 is explicit: persist on every run, not only on parse failure."""
    gw_text = json.dumps({"verdict": "clean", "issues": [], "confidence": 0.95})
    artifacts_dir = tmp_path / "run-id"

    raw_ref, _ = _persist_gw_artifacts(artifacts_dir, gw_text, [])

    assert raw_ref != ""
    assert (artifacts_dir / "gw-raw-output.txt").read_text(encoding="utf-8") == gw_text


def test_persist_gw_artifacts_write_failure_is_best_effort(tmp_path, capsys):
    artifacts_dir = tmp_path / "run-id"

    with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
        raw_ref, transcript_ref = _persist_gw_artifacts(artifacts_dir, "some text", [])

    # Must not raise, and must not fabricate a ref for a write that failed.
    assert raw_ref == ""
    assert transcript_ref == ""
    captured = capsys.readouterr()
    assert "gw-raw-output-write-error" in captured.err
    assert "gw-transcript-write-error" in captured.err


# ---------------------------------------------------------------------------
# DoD 6 — the brief renders parse_failed diagnostics, omits them otherwise
# ---------------------------------------------------------------------------

def test_brief_renders_gw_parse_failed_diagnostics():
    brief = SpecReviewBrief(
        spec_path=Path("/tmp/spec.md"),
        target_id="test",
        repo="test",
        council_status="resolved",
        council_landing="",
        council_open_questions=[],
        council_confidence="",
        council_positions=[],
        council_run_id="",
        elapsed_s=1.0,
        combined_recommendation="proceed-to-bind",
        gw_ran=True,
        gw_verdict="parse_failed",
        gw_findings_count=0,
        elapsed_gw=193.6,
        gw_transcript_ref="/srv/lapis/spec-review-artifacts/run/gw-transcript.json",
        gw_raw_output_ref="/srv/lapis/spec-review-artifacts/run/gw-raw-output.txt",
        gw_parse_error={
            "file_size": 68021,
            "head": "Investigating the spec now" + " x" * 50,
            "tail": "that concludes my review" + " y" * 50,
        },
    )

    rendered = format_brief(brief)

    assert "parse_failed" in rendered
    assert "68021" in rendered
    assert "Investigating the spec now" in rendered
    assert "that concludes my review" in rendered
    assert "gw-raw-output.txt" in rendered


def test_brief_omits_parse_failed_block_when_verdict_is_not_parse_failed():
    brief = SpecReviewBrief(
        spec_path=Path("/tmp/spec.md"),
        target_id="test",
        repo="test",
        council_status="resolved",
        council_landing="",
        council_open_questions=[],
        council_confidence="",
        council_positions=[],
        council_run_id="",
        elapsed_s=1.0,
        combined_recommendation="proceed-to-bind",
        gw_ran=True,
        gw_verdict="clean",
        gw_findings_count=0,
        elapsed_gw=10.0,
        gw_transcript_ref="/srv/lapis/spec-review-artifacts/run/gw-transcript.json",
        gw_raw_output_ref="/srv/lapis/spec-review-artifacts/run/gw-raw-output.txt",
        gw_parse_error=None,
    )

    rendered = format_brief(brief)

    assert "Verdict:** clean" in rendered
    assert "Head (first 200 chars)" not in rendered
    assert "Tail (last 200 chars)" not in rendered
