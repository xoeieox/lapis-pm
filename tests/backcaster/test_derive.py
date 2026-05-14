"""Tests: derive stage — fallback path, retry behavior (Test 3, 4)."""
from __future__ import annotations

import json
import unittest.mock as mock

import pytest
from pydantic import ValidationError

from lapis_pm.backcaster.schema import Gap


def _make_gap(pid: str = "p-01") -> Gap:
    return Gap(
        precondition_id=pid,
        what_exists="something",
        what_missing="something else",
        what_miswired="nothing identified",
        citations=[],
        unsourced=True,
    )


# ---------------------------------------------------------------------------
# Test 3: Fallback path — malformed output → retry → fallback component
# ---------------------------------------------------------------------------

def test_derive_fallback_on_both_attempts_fail(tmp_path):
    """Both LLM attempts produce malformed JSON → fallback component written."""
    from lapis_pm.backcaster.derive import derive_components

    gaps = [_make_gap("p-01")]

    # Both calls return invalid JSON
    with mock.patch("lapis_pm.backcaster.llm_routing.call_operator", return_value="not json"):
        components, _, fallback_count = derive_components(gaps, stub=False)

    assert fallback_count == 1
    assert len(components) == 1
    c = components[0]
    assert c.category == "research"
    assert c.unsourced is True
    assert c.fallback is True


def test_derive_fallback_on_invalid_category(tmp_path):
    """Invalid category in LLM output → retry → fallback."""
    from lapis_pm.backcaster.derive import derive_components

    gaps = [_make_gap("p-01")]

    bad_output = json.dumps({"components": [{"description": "x", "category": "magic",
                                              "effort_estimate": "small", "reversibility": "high",
                                              "dependencies": []}]})

    with mock.patch("lapis_pm.backcaster.llm_routing.call_operator", return_value=bad_output):
        components, _, fallback_count = derive_components(gaps, stub=False)

    assert fallback_count == 1
    assert components[0].fallback is True


def test_derive_retry_succeeds_second_attempt():
    """First call fails, second call returns valid output → component accepted, no fallback."""
    from lapis_pm.backcaster.derive import derive_components

    gaps = [_make_gap("p-01")]

    good_output = json.dumps({
        "components": [{
            "description": "A real component",
            "category": "policy",
            "effort_estimate": "small",
            "reversibility": "high",
            "dependencies": [],
        }]
    })

    call_count = [0]

    def mock_call(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return "not valid json"  # first call fails parse
        return good_output  # second call succeeds

    with mock.patch("lapis_pm.backcaster.llm_routing.call_operator", side_effect=mock_call):
        components, _, fallback_count = derive_components(gaps, stub=False)

    assert fallback_count == 0
    assert len(components) == 1
    assert components[0].category == "policy"
    assert components[0].fallback is False


def test_derive_fallback_count_two_gaps(tmp_path):
    """2 out of 10 gaps produce malformed output → derive_fallback_count == 2."""
    from lapis_pm.backcaster.derive import derive_components

    good_output = json.dumps({
        "components": [{
            "description": "A component",
            "category": "software",
            "effort_estimate": "small",
            "reversibility": "high",
            "dependencies": [],
        }]
    })

    gaps = [_make_gap(f"p-{i:02d}") for i in range(10)]
    fail_pids = {"p-03", "p-07"}

    call_map: dict[str, int] = {}

    def mock_call(*args, **kwargs):
        # Figure out which gap we're on from prompt content
        prompt = kwargs.get("prompt") or (args[0] if args else "")
        for pid in fail_pids:
            if pid in prompt:
                call_map[pid] = call_map.get(pid, 0) + 1
                return "not json"  # always fail for these two
        return good_output

    with mock.patch("lapis_pm.backcaster.llm_routing.call_operator", side_effect=mock_call):
        components, _, fallback_count = derive_components(gaps, stub=False)

    assert fallback_count == 2, f"Expected fallback_count=2, got {fallback_count}"

    # histogram check: research.fallback breakout
    from lapis_pm.backcaster.schema import Histogram
    hist = Histogram.from_components(components)
    d = hist.to_dict()
    assert d["research.fallback"] == 2, f"research.fallback should be 2, got {d['research.fallback']}"

    # Organic research components (non-fallback) should not count in research.fallback
    organic_research = sum(1 for c in components if c.category == "research" and not c.fallback)
    assert d["research"] == fallback_count + organic_research
