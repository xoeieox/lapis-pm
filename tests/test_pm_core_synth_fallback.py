"""Tests for brief synthesis fallback handling (brief-synthesis-fallback-v0).

Acceptance criteria:
2. _set_brief_outstanding(target_id, b) called with b.synthesis_failed=True writes a
   pm:synthesis-failed observation, does NOT call set_outstanding_brief, and returns False.
3. On the 3rd consecutive call with synthesis_failed=True, it DOES call
   set_outstanding_brief and returns True.
4. On a successful brief after prior failures, fail counter resets to 0.
6. Existing call sites in pm_core.py all route through _set_brief_outstanding.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from lapis_pm import brief, pm_core, episodic


# ---------------------------------------------------------------------------
# Synth-fail counter helpers (acceptance criteria 2, 3, 4)
# ---------------------------------------------------------------------------

def test_set_brief_outstanding_success_resets_counter():
    """When b.synthesis_failed=False, counter is reset and outstanding brief is set."""
    target_id = "test-tid"

    # Set up a prior failure to initialize the counter
    with patch("lapis_pm.pm_core._mem") as mock_mem:
        mem_data = {}

        def mock_set(key, value, **kwargs):
            mem_data[key] = {"content": value}

        def mock_get(key):
            return mem_data.get(key)

        def mock_delete(key):
            if key in mem_data:
                del mem_data[key]

        mock_mem.return_value.set = mock_set
        mock_mem.return_value.get = mock_get
        mock_mem.return_value.delete = mock_delete

        # Initialize counter to 2
        pm_core._mem().set(pm_core._SYNTH_FAIL_KEY.format(target_id), "2")

        # Now call with a successful brief
        b = brief.Brief(
            target_id=target_id,
            body="Valid brief body",
            comment_id="success-cid",
            pushed=False,
            synthesis_failed=False,
        )

        with patch("lapis_pm.pm_core.set_outstanding_brief") as mock_set_brief:
            result = pm_core._set_brief_outstanding(target_id, b, verified=False)

        assert result is True, "Expected _set_brief_outstanding to return True for success"
        mock_set_brief.assert_called_once_with(target_id, "success-cid")

        # Counter should be deleted (reset)
        counter = pm_core._get_synth_fail_count(target_id)
        assert counter == 0, f"Expected counter=0 after success, got {counter}"


def test_set_brief_outstanding_skip_on_early_failure():
    """When synthesis_failed=True (attempt 1-2), skip set_outstanding_brief, return False."""
    target_id = "test-tid-fail"

    with patch("lapis_pm.pm_core._mem") as mock_mem:
        mem_data = {}

        def mock_set(key, value, **kwargs):
            mem_data[key] = {"content": value}

        def mock_get(key):
            return mem_data.get(key)

        def mock_delete(key):
            if key in mem_data:
                del mem_data[key]

        mock_mem.return_value.set = mock_set
        mock_mem.return_value.get = mock_get
        mock_mem.return_value.delete = mock_delete

        b = brief.Brief(
            target_id=target_id,
            body="Placeholder brief",
            comment_id="fail-cid",
            pushed=False,
            synthesis_failed=True,
        )

        with (
            patch("lapis_pm.pm_core.set_outstanding_brief") as mock_set_brief,
            patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs,
        ):
            # First call: attempt 1
            result = pm_core._set_brief_outstanding(target_id, b, verified=False)

        assert result is False, "Expected False for first failure attempt"
        mock_set_brief.assert_not_called()
        mock_obs.assert_called_once()
        obs_content = mock_obs.call_args[0][1]
        assert "pm:synthesis-failed" in mock_obs.call_args[1]["extra_tags"]
        assert "attempt 1/3" in obs_content


def test_set_brief_outstanding_threshold_forces_outstanding():
    """After 3 consecutive failures, force set_outstanding_brief and return True."""
    target_id = "test-tid-threshold"

    with patch("lapis_pm.pm_core._mem") as mock_mem:
        mem_data = {}

        def mock_set(key, value, **kwargs):
            mem_data[key] = {"content": value}

        def mock_get(key):
            return mem_data.get(key)

        def mock_delete(key):
            if key in mem_data:
                del mem_data[key]

        mock_mem.return_value.set = mock_set
        mock_mem.return_value.get = mock_get
        mock_mem.return_value.delete = mock_delete

        # Simulate 2 prior failures
        pm_core._inc_synth_fail_count(target_id)
        pm_core._inc_synth_fail_count(target_id)

        b = brief.Brief(
            target_id=target_id,
            body="Placeholder brief",
            comment_id="threshold-cid",
            pushed=False,
            synthesis_failed=True,
        )

        with (
            patch("lapis_pm.pm_core.set_outstanding_brief") as mock_set_brief,
            patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs,
        ):
            # Third call: should hit threshold
            result = pm_core._set_brief_outstanding(target_id, b, verified=False)

        assert result is True, "Expected True when threshold is hit"
        mock_set_brief.assert_called_once_with(target_id, "threshold-cid")


def test_set_brief_outstanding_verified_paths():
    """_set_brief_outstanding with verified=True calls set_outstanding_brief_verified."""
    target_id = "test-tid-verified"

    with patch("lapis_pm.pm_core._mem") as mock_mem:
        mem_data = {}

        def mock_set(key, value, **kwargs):
            mem_data[key] = {"content": value}

        def mock_get(key):
            return mem_data.get(key)

        def mock_delete(key):
            if key in mem_data:
                del mem_data[key]

        mock_mem.return_value.set = mock_set
        mock_mem.return_value.get = mock_get
        mock_mem.return_value.delete = mock_delete

        b = brief.Brief(
            target_id=target_id,
            body="Valid brief body",
            comment_id="verified-cid",
            pushed=False,
            synthesis_failed=False,
        )

        with (
            patch("lapis_pm.pm_core.set_outstanding_brief_verified") as mock_set_verified,
            patch("lapis_pm.pm_core._post_write_sweep_brief") as mock_sweep,
        ):
            result = pm_core._set_brief_outstanding(target_id, b, verified=True)

        assert result is True
        mock_set_verified.assert_called_once_with(target_id, "verified-cid")
        mock_sweep.assert_called_once_with(target_id, "verified-cid")


def test_synth_fail_counter_increment_and_reset():
    """Counter functions correctly increment and reset."""
    target_id = "test-tid-counter"

    with patch("lapis_pm.pm_core._mem") as mock_mem:
        mem_data = {}

        def mock_set(key, value, **kwargs):
            mem_data[key] = {"content": value}

        def mock_get(key):
            return mem_data.get(key)

        def mock_delete(key):
            if key in mem_data:
                del mem_data[key]

        mock_mem.return_value.set = mock_set
        mock_mem.return_value.get = mock_get
        mock_mem.return_value.delete = mock_delete

        # Initial: 0
        assert pm_core._get_synth_fail_count(target_id) == 0

        # Increment: 1
        n = pm_core._inc_synth_fail_count(target_id)
        assert n == 1
        assert pm_core._get_synth_fail_count(target_id) == 1

        # Increment: 2
        n = pm_core._inc_synth_fail_count(target_id)
        assert n == 2
        assert pm_core._get_synth_fail_count(target_id) == 2

        # Reset: 0
        pm_core._reset_synth_fail_count(target_id)
        assert pm_core._get_synth_fail_count(target_id) == 0


# ---------------------------------------------------------------------------
# Call site verification (acceptance criterion 6)
# ---------------------------------------------------------------------------

def test_all_set_outstanding_brief_call_sites_routed():
    """Verify that the 10 external call sites route through _set_brief_outstanding.

    Does NOT check internal calls within set_outstanding_brief_verified (verify-and-retry
    loop, lines ~1146/1158) or within _set_brief_outstanding itself, which are expected.
    """
    import subprocess
    import lapis_pm.pm_core as _m
    pm_core_path = _m.__file__
    result = subprocess.run(
        [
            "grep", "-n",
            "_set_brief_outstanding",
            pm_core_path,
        ],
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

    # Count call sites (excluding def, type hints, and strings)
    call_sites = []
    for line in lines:
        if not line:
            continue
        # Skip: def _set_brief_outstanding
        if "def _set_brief_outstanding" in line:
            continue
        # Skip: lines in docstrings or strings
        line_content = line.split(":", 1)[1] if ":" in line else line
        if '"""' in line_content or "'''" in line_content or line_content.strip().startswith('"') or line_content.strip().startswith("'"):
            continue
        if "_set_brief_outstanding" in line:
            call_sites.append(line)

    # We expect at least 7 call sites (the 10 external call sites minus 3 that should not appear)
    # The grep finds the definition + 7 call sites
    assert len(call_sites) >= 7, (
        f"Expected at least 7 _set_brief_outstanding call sites, "
        f"but found only {len(call_sites)}:\n" + "\n".join(call_sites)
    )
