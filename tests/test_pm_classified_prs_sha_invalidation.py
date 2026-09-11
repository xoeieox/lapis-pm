"""Tests for classified-prs SHA-advance invalidation (pm-classified-prs-sha-invalidation).

Coverage:
  - SHA advance removes the PR from classified-prs.
  - Identical SHA (no advance) leaves classified-prs unchanged.
  - Advancing one PR's SHA preserves other classified PRs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_pr(number: int, sha: str) -> dict:
    return {"number": number, "head": {"sha": sha}}


def _patch_encode(target_id: str, open_prs: list[dict], classified: set[int], last_sha: str | None):
    """Run _encode_pr_sha_updates with mocked episodic and mem."""
    mem_store: dict[str, str] = {}

    import json as _json

    mem_mock = MagicMock()

    def _mem_get(key):
        val = mem_store.get(key)
        if val is None:
            return None
        return {"content": val}

    def _mem_set(key, value, tags=None):
        mem_store[key] = value

    mem_mock.get.side_effect = _mem_get
    mem_mock.set.side_effect = _mem_set

    # Pre-populate classified-prs
    import json
    mem_store[pm_core._classified_prs_key(target_id)] = json.dumps(sorted(classified))

    with (
        patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core._last_observed_pr_sha", return_value=last_sha),
        patch("lapis_pm.pm_core.episodic") as episodic_mock,
    ):
        episodic_mock.write_observation = MagicMock()
        pm_core._encode_pr_sha_updates(target_id, open_prs)
        result_ids = pm_core._classified_pr_ids(target_id)

    return result_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sha_advance_invalidates_classified_prs():
    """SHA advance for PR #13 removes it from classified-prs."""
    mem_store: dict[str, str] = {}
    import json

    mem_mock = MagicMock()

    def _mem_get(key):
        val = mem_store.get(key)
        return {"content": val} if val is not None else None

    def _mem_set(key, value, tags=None):
        mem_store[key] = value

    mem_mock.get.side_effect = _mem_get
    mem_mock.set.side_effect = _mem_set

    target_id = "test-target"
    mem_store[pm_core._classified_prs_key(target_id)] = json.dumps([13])

    open_prs = [_fake_pr(13, "newsha123")]

    with (
        patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core._last_observed_pr_sha", return_value="oldsha456"),
        patch("lapis_pm.pm_core.episodic") as episodic_mock,
    ):
        episodic_mock.write_observation = MagicMock()
        pm_core._encode_pr_sha_updates(target_id, open_prs)
        result = pm_core._classified_pr_ids(target_id)

    assert 13 not in result


def test_sha_unchanged_preserves_classified_prs():
    """Same SHA (no advance) leaves classified-prs unchanged."""
    mem_store: dict[str, str] = {}
    import json

    mem_mock = MagicMock()

    def _mem_get(key):
        val = mem_store.get(key)
        return {"content": val} if val is not None else None

    def _mem_set(key, value, tags=None):
        mem_store[key] = value

    mem_mock.get.side_effect = _mem_get
    mem_mock.set.side_effect = _mem_set

    target_id = "test-target"
    mem_store[pm_core._classified_prs_key(target_id)] = json.dumps([13])

    open_prs = [_fake_pr(13, "samesha")]

    with (
        patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core._last_observed_pr_sha", return_value="samesha"),
        patch("lapis_pm.pm_core.episodic") as episodic_mock,
    ):
        episodic_mock.write_observation = MagicMock()
        pm_core._encode_pr_sha_updates(target_id, open_prs)
        result = pm_core._classified_pr_ids(target_id)

    assert 13 in result


def test_sha_advance_unrelated_pr_preserves_classification():
    """Advancing PR #13's SHA leaves PR #14 in classified-prs."""
    mem_store: dict[str, str] = {}
    import json

    mem_mock = MagicMock()

    def _mem_get(key):
        val = mem_store.get(key)
        return {"content": val} if val is not None else None

    def _mem_set(key, value, tags=None):
        mem_store[key] = value

    mem_mock.get.side_effect = _mem_get
    mem_mock.set.side_effect = _mem_set

    target_id = "test-target"
    mem_store[pm_core._classified_prs_key(target_id)] = json.dumps([13, 14])

    open_prs = [_fake_pr(13, "newsha789")]

    def _last_sha(tid, pr_num):
        if pr_num == 13:
            return "oldsha000"
        return "samesha"

    with (
        patch("lapis_pm.pm_core._mem", return_value=mem_mock),
            patch("lapis_pm.pm_core._last_observed_pr_sha", side_effect=_last_sha),
        patch("lapis_pm.pm_core.episodic") as episodic_mock,
    ):
        episodic_mock.write_observation = MagicMock()
        pm_core._encode_pr_sha_updates(target_id, open_prs)
        result = pm_core._classified_pr_ids(target_id)

    assert 13 not in result
    assert 14 in result
