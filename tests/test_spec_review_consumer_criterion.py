"""Tests for the C7 consumer admission criterion (lapis-pm-gate-consumer-criterion-v0).

C7 MUST NOT - A new autonomous producer be built without an identified consumer.

Covers DoD-1 (extraction), DoD-2 (missing => blocking), DoD-3 (reject-list table),
DoD-3b (self-referential rejection), DoD-4 (substantive value passes), and DoD-6
(this spec's own Consumer line survives the criterion it introduces).
"""
from __future__ import annotations

import pytest

from lapis_pm.spec_review import (
    _consumer_criterion_check,
    _consumer_criterion_finding_text,
    _CONSUMER_REJECT_TOKENS,
    C7_TEXT,
)
from pathlib import Path


TARGET_ID = "lapis-pm-gate-consumer-criterion-v0"


def _spec(consumer_line: str | None) -> str:
    header = f"**Target ID:** `{TARGET_ID}`\n**Repo:** `lapis-pm`\n**Authority:** hold\n"
    if consumer_line is not None:
        header += consumer_line + "\n"
    return header + "\n## Why\n\nBody text.\n"


# ---------------------------------------------------------------------------
# DoD-1: extraction from a well-formed header
# ---------------------------------------------------------------------------

def test_dod1_extracts_substantive_consumer():
    text = _spec("**Consumer:** downstream service X reads the emitted report nightly")
    result = _consumer_criterion_check(text, TARGET_ID)
    assert result["status"] == "ok"
    assert "downstream service X" in result["raw_value"]


# ---------------------------------------------------------------------------
# DoD-2: no **Consumer:** line at all => blocking
# ---------------------------------------------------------------------------

def test_dod2_missing_line_is_blocking():
    text = _spec(None)
    result = _consumer_criterion_check(text, TARGET_ID)
    assert result["status"] == "blocking"
    assert result["reason"] == "missing"


def test_dod2_finding_cites_c7_verbatim_and_spec_path():
    text = _spec(None)
    result = _consumer_criterion_check(text, TARGET_ID)
    finding = _consumer_criterion_finding_text(Path("/srv/lapis/planning/specs/foo.md"), result)
    assert C7_TEXT in finding
    assert "/srv/lapis/planning/specs/foo.md" in finding


# ---------------------------------------------------------------------------
# DoD-3: reject-list — table-driven, case-insensitive exact match
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token", sorted(_CONSUMER_REJECT_TOKENS - {""}))
def test_dod3_reject_list_tokens_are_blocking(token):
    for variant in (token, token.upper(), token.title(), f"  {token}  ", f"`{token}`"):
        text = _spec(f"**Consumer:** {variant}")
        result = _consumer_criterion_check(text, TARGET_ID)
        assert result["status"] == "blocking", f"expected blocking for {variant!r}"
        assert result["reason"] in ("trivial-non-responsive", "empty")


def test_dod3_empty_value_is_blocking():
    text = _spec("**Consumer:**")
    result = _consumer_criterion_check(text, TARGET_ID)
    assert result["status"] == "blocking"
    assert result["reason"] == "empty"


# ---------------------------------------------------------------------------
# DoD-3b: self-referential value (names only the producer itself) => blocking
# ---------------------------------------------------------------------------

def test_dod3b_self_referential_value_is_blocking():
    text = _spec(f"**Consumer:** `{TARGET_ID}`")
    result = _consumer_criterion_check(text, TARGET_ID)
    assert result["status"] == "blocking"
    assert result["reason"] == "self-referential"


def test_dod3b_self_referential_value_ignoring_formatting_is_blocking():
    # Same target_id, different casing/punctuation — must still be caught.
    mangled = TARGET_ID.replace("-", " ").upper()
    text = _spec(f"**Consumer:** {mangled}")
    result = _consumer_criterion_check(text, TARGET_ID)
    assert result["status"] == "blocking"
    assert result["reason"] == "self-referential"


# ---------------------------------------------------------------------------
# DoD-4: substantive consumer passes — the criterion cannot pass by rejecting
# everything
# ---------------------------------------------------------------------------

def test_dod4_substantive_consumer_passes_no_finding():
    text = _spec("**Consumer:** the nightly briefing pipeline, which reads this report")
    result = _consumer_criterion_check(text, TARGET_ID)
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# DoD-6 (HARD BLOCKER): this spec's own Consumer line survives the criterion,
# including DoD-3 and DoD-3b.
# ---------------------------------------------------------------------------

def test_dod6_this_specs_own_consumer_line_survives():
    own_consumer = (
        "the spec-review gate's own admission path (`spec_review.py`), "
        "and Erah reading the gate verdict. See DoD-5 for the end-to-end demonstration."
    )
    text = _spec(f"**Consumer:** {own_consumer}")
    result = _consumer_criterion_check(text, TARGET_ID)
    assert result["status"] == "ok", (
        "This spec's own Consumer line must never be rejected by the criterion "
        "it introduces (DoD-6, hard blocker)."
    )
