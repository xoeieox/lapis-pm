"""Unit tests for lapis-pm-fixer-candidate-format-retry-v0 (bounded corrective
format retry in generate_candidates).

All tests are fully offline: swarm mocked via monkeypatch, no GW, no live network.
"""

from dataclasses import asdict
import json

import pytest

from lapis_pm.batched_fixer_eval import (
    FixtureRecord,
    _FORMAT_RETRY_SUFFIX,
    apply_search_replace,
    generate_candidates,
    run_fixture,
)

VALID_BLOCK = (
    "<<<<<<< SEARCH\n"
    "def foo():\n    return 'old'\n"
    "=======\n"
    "def foo():\n    return 'new'\n"
    ">>>>>>> REPLACE\n"
)

# Well-formed markers, but a SEARCH body that won't match any real file - this is
# an APPLY-level failure, not a PARSE-level one, and must NOT trigger retry (AC9).
NON_APPLYING_VALID_BLOCK = (
    "<<<<<<< SEARCH\n"
    "this text will never match anything\n"
    "=======\n"
    "def foo():\n    return 'new'\n"
    ">>>>>>> REPLACE\n"
)

MALFORMED_JSON_BLOCK = '{"SEARCH": "def foo():", "REPLACE": "def bar():"}'


def _fixture(**overrides):
    defaults = dict(
        repo="test", sha="sha123", parent_sha="parent",
        pr_number=1, path="test.py", file_loc="unknown", changed_lines=1, tier="T1",
        task_intent_paraphrased="Fix the bug",
        pre_state_slice="def foo():\n    return 'old'\n",
    )
    defaults.update(overrides)
    return FixtureRecord(**defaults)


def _is_corrective(prompt: str) -> bool:
    return _FORMAT_RETRY_SUFFIX in prompt


def test_ac1_first_shot_compliant(monkeypatch):
    calls = []

    def fake_call_swarm(prompts, **kwargs):
        calls.append(list(prompts))
        return [VALID_BLOCK for _ in prompts]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    fixture = _fixture()
    candidates = generate_candidates(fixture, n=1, mock_mode=False)
    assert len(candidates) == 1
    text, retries = candidates[0]
    assert text is not None
    assert retries == 0
    assert len(calls) == 1
    assert len(calls[0]) == 1


def test_ac2_fails_then_succeeds(monkeypatch):
    calls = []

    def fake_call_swarm(prompts, **kwargs):
        calls.append(list(prompts))
        if _is_corrective(prompts[0]):
            return [VALID_BLOCK for _ in prompts]
        return [MALFORMED_JSON_BLOCK for _ in prompts]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    fixture = _fixture()
    candidates = generate_candidates(fixture, n=1, mock_mode=False, max_format_retries=2)
    assert len(candidates) == 1
    text, retries = candidates[0]
    assert retries == 1
    assert text == VALID_BLOCK
    assert len(calls) == 2


def test_ac3_never_complies(monkeypatch):
    calls = []

    def fake_call_swarm(prompts, **kwargs):
        calls.append(list(prompts))
        return [MALFORMED_JSON_BLOCK for _ in prompts]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    fixture = _fixture()
    candidates = generate_candidates(fixture, n=1, mock_mode=False, max_format_retries=2)
    assert len(candidates) == 1
    text, retries = candidates[0]
    assert retries == 2
    assert text == MALFORMED_JSON_BLOCK
    assert len(calls) == 3

    # downstream: candidate_text never applies -> apply_status "failed", never None
    new_text, reason = apply_search_replace(fixture.pre_state_slice, text)
    assert new_text is None
    assert reason in ("no-blocks-parsed", "malformed-block")


def test_ac4_backward_compatible_zero_retries(monkeypatch):
    calls = []

    def fake_call_swarm(prompts, **kwargs):
        calls.append(list(prompts))
        return [MALFORMED_JSON_BLOCK, VALID_BLOCK]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    fixture = _fixture()
    candidates = generate_candidates(fixture, n=2, mock_mode=False, max_format_retries=0)
    assert len(candidates) == 2
    assert all(retries == 0 for _, retries in candidates)
    assert candidates[0][0] == MALFORMED_JSON_BLOCK
    assert candidates[1][0] == VALID_BLOCK
    assert len(calls) == 1


def test_ac5_leak_guard_on_retry_prompts(monkeypatch):
    seen_prompts = []

    def fake_call_swarm(prompts, **kwargs):
        seen_prompts.extend(prompts)
        return [MALFORMED_JSON_BLOCK for _ in prompts]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    fixture = _fixture(
        golden_test_diff="",
        golden_test_ids=["tests/test_m.py::test_secret_oracle"],
        checker_class="DISCRIMINATES",
        fail_first_confirmed=True,
    )
    candidates = generate_candidates(fixture, n=1, mock_mode=False, max_format_retries=2)
    assert len(candidates) == 1
    # No leak guard raised across initial + 2 retry rounds.
    assert len(seen_prompts) == 3
    for tid in fixture.golden_test_ids:
        assert tid not in _FORMAT_RETRY_SUFFIX
    assert "golden_test" not in _FORMAT_RETRY_SUFFIX


def test_ac5_leak_guard_still_raises_on_base_prompt(monkeypatch):
    fixture = _fixture(
        pre_state_slice=(
            "diff --git a/tests/test_m.py b/tests/test_m.py\n"
            "--- /dev/null\n+++ b/tests/test_m.py\n"
            "+def test_secret_oracle():\n+    assert True\n"
        ),
        golden_test_diff=(
            "diff --git a/tests/test_m.py b/tests/test_m.py\n"
            "--- /dev/null\n+++ b/tests/test_m.py\n"
            "+def test_secret_oracle():\n+    assert True\n"
        ),
        golden_test_ids=["tests/test_m.py::test_secret_oracle"],
        checker_class="DISCRIMINATES",
        fail_first_confirmed=True,
    )
    with pytest.raises(RuntimeError):
        generate_candidates(fixture, n=1, mock_mode=False)


def test_ac6_bounded_total_calls(monkeypatch):
    total_items = []

    def fake_call_swarm(prompts, **kwargs):
        total_items.extend(prompts)
        return [MALFORMED_JSON_BLOCK for _ in prompts]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    n = 3
    max_format_retries = 2
    fixture = _fixture()
    candidates = generate_candidates(
        fixture, n=n, mock_mode=False, max_format_retries=max_format_retries
    )
    assert len(candidates) == n
    assert all(retries == max_format_retries for _, retries in candidates)
    assert len(total_items) == n * (1 + max_format_retries)


def test_ac7_serialized_distinct_format_retries(monkeypatch):
    call_round = {"n": 0}

    def fake_call_swarm(prompts, **kwargs):
        call_round["n"] += 1
        if call_round["n"] == 1:
            # slot 0 malformed, slot 1 valid first-shot
            return [MALFORMED_JSON_BLOCK, VALID_BLOCK]
        return [VALID_BLOCK for _ in prompts]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    fixture = _fixture()
    result = run_fixture(fixture, n_range=[2], mock_mode=False)
    format_retries = [c.format_retries for c in result.candidates]
    assert len(set(format_retries)) > 1, f"expected distinct retry counts, got {format_retries}"

    # Confirm the field survives the same serialization the runner uses.
    for candidate in result.candidates:
        record = json.loads(json.dumps(asdict(candidate), default=str))
        assert "format_retries" in record
        assert record["format_retries"] == candidate.format_retries


def test_ac8_mock_path_unchanged(monkeypatch):
    called = {"n": 0}

    def fake_call_swarm(prompts, **kwargs):
        called["n"] += 1
        return [VALID_BLOCK for _ in prompts]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    fixture = _fixture()
    candidates = generate_candidates(fixture, n=3, mock_mode=True)
    assert len(candidates) == 3
    assert all(retries == 0 for _, retries in candidates)
    assert all(text is not None for text, _ in candidates)
    assert called["n"] == 0


def test_ac9_parse_level_only_boundary(monkeypatch):
    calls = []

    def fake_call_swarm(prompts, **kwargs):
        calls.append(list(prompts))
        return [NON_APPLYING_VALID_BLOCK for _ in prompts]

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    fixture = _fixture()
    candidates = generate_candidates(fixture, n=1, mock_mode=False, max_format_retries=2)
    assert len(candidates) == 1
    text, retries = candidates[0]
    assert retries == 0
    assert text == NON_APPLYING_VALID_BLOCK
    assert len(calls) == 1

    # Confirm it really is an apply-level failure, not a parse-level one.
    new_text, reason = apply_search_replace(fixture.pre_state_slice, text)
    assert new_text is None
    assert reason == "search-not-found"


def test_retry_round_exception_preserves_prior_success(monkeypatch):
    """A call_swarm exception on a later retry round must not discard slots that
    already parsed successfully in an earlier round (spec item 7)."""
    call_round = {"n": 0}

    def fake_call_swarm(prompts, **kwargs):
        call_round["n"] += 1
        if call_round["n"] == 1:
            # slot 0 valid first-shot, slot 1 malformed
            return [VALID_BLOCK, MALFORMED_JSON_BLOCK]
        raise RuntimeError("simulated swarm outage on retry round")

    monkeypatch.setattr("agents_core.llm.call_swarm", fake_call_swarm)

    fixture = _fixture()
    candidates = generate_candidates(fixture, n=2, mock_mode=False, max_format_retries=2)
    assert len(candidates) == 2
    text0, retries0 = candidates[0]
    text1, retries1 = candidates[1]
    assert text0 == VALID_BLOCK
    assert retries0 == 0
    # Slot 1 never got a successful retry; it keeps its last (malformed) text, not None.
    assert text1 == MALFORMED_JSON_BLOCK
