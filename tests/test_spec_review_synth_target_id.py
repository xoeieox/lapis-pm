"""Tests for _synth_target_id."""
from __future__ import annotations

import re

from lapis_pm.spec_review import _synth_target_id


def test_shape():
    tid = _synth_target_id("my-spec")
    assert re.match(
        r"^spec-review-my-spec-\d{10,}-[0-9a-f]{6}$", tid
    ), f"unexpected shape: {tid!r}"


def test_includes_parsed_target_id():
    tid = _synth_target_id("agents-core-v0")
    assert tid.startswith("spec-review-agents-core-v0-")


def test_distinct_consecutive_calls():
    """uuid suffix must prevent collision even within the same second."""
    ids = {_synth_target_id("my-spec") for _ in range(20)}
    assert len(ids) == 20, "expected all 20 IDs to be distinct"


def test_two_char_parsed_tid():
    tid = _synth_target_id("ab")
    assert tid.startswith("spec-review-ab-")
