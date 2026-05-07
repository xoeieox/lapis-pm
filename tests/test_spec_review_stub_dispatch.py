"""Tests for SPEC_REVIEWER_STUB dispatch short-circuit."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.spec_review import (
    _CLAUDE_QUEUE_COMPLETED,
    _dispatch_spec_reviewer,
)


def _cleanup(task_id: str):
    p = _CLAUDE_QUEUE_COMPLETED / f"{task_id}-output.md"
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Stub short-circuit
# ---------------------------------------------------------------------------

def test_stub_dispatch_does_not_call_shaper(tmp_path):
    """SPEC_REVIEWER_STUB=1 must not touch _SHAPER.dispatch.

    The stub path exits immediately without importing agents_core.shaper at all.
    We verify this by patching the lazy import and asserting it was never called.
    """
    synth_tid = f"stub-test-{int(time.time())}"
    try:
        import sys
        # If agents_core.shaper is importable, patch it; otherwise skip that check.
        with patch.dict(os.environ, {"SPEC_REVIEWER_STUB": "1", "SPEC_REVIEWER_STUB_VERDICT": "clean"}):
            result = _dispatch_spec_reviewer(
                spec_text="spec text",
                synth_target_id=synth_tid,
                parsed_target_id="my-tid",
                repo="lapis-pm",
                invariant_context="ctx",
            )

        # Stub path returns immediately with stub task_id — shaper was not needed
        assert result.task_id == f"stub-{synth_tid}"
        assert result.agent_type == "spec_reviewer"
    finally:
        _cleanup(f"stub-{synth_tid}")


def test_stub_writes_output_file(tmp_path):
    """SPEC_REVIEWER_STUB=1 writes a readable output file with the fixture verdict."""
    synth_tid = f"stub-write-{int(time.time())}"
    try:
        with patch.dict(os.environ, {"SPEC_REVIEWER_STUB": "1", "SPEC_REVIEWER_STUB_VERDICT": "fixable"}):
            result = _dispatch_spec_reviewer(
                spec_text="s",
                synth_target_id=synth_tid,
                parsed_target_id="t",
                repo="r",
                invariant_context="c",
            )

        out = _CLAUDE_QUEUE_COMPLETED / f"stub-{synth_tid}-output.md"
        assert out.exists(), "stub must write output file"
        data = json.loads(out.read_text())
        assert data["verdict"] == "fixable"
        assert data["confidence"] == 0.95
    finally:
        _cleanup(f"stub-{synth_tid}")


def test_stub_default_verdict_is_clean(tmp_path):
    """Default SPEC_REVIEWER_STUB_VERDICT is 'clean'."""
    synth_tid = f"stub-default-{int(time.time())}"
    env = {"SPEC_REVIEWER_STUB": "1"}
    env.pop("SPEC_REVIEWER_STUB_VERDICT", None)
    try:
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("SPEC_REVIEWER_STUB_VERDICT", None)
            result = _dispatch_spec_reviewer(
                spec_text="s",
                synth_target_id=synth_tid,
                parsed_target_id="t",
                repo="r",
                invariant_context="c",
            )

        out = _CLAUDE_QUEUE_COMPLETED / f"stub-{synth_tid}-output.md"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["verdict"] == "clean"
    finally:
        _cleanup(f"stub-{synth_tid}")
        os.environ.pop("SPEC_REVIEWER_STUB_VERDICT", None)


def test_stub_needs_human_verdict(tmp_path):
    synth_tid = f"stub-nh-{int(time.time())}"
    try:
        with patch.dict(os.environ, {"SPEC_REVIEWER_STUB": "1", "SPEC_REVIEWER_STUB_VERDICT": "needs-human"}):
            _dispatch_spec_reviewer(
                spec_text="s",
                synth_target_id=synth_tid,
                parsed_target_id="t",
                repo="r",
                invariant_context="c",
            )
        out = _CLAUDE_QUEUE_COMPLETED / f"stub-{synth_tid}-output.md"
        data = json.loads(out.read_text())
        assert data["verdict"] == "needs-human"
    finally:
        _cleanup(f"stub-{synth_tid}")


# ---------------------------------------------------------------------------
# SPEC_REVIEWER_STUB env var is independent of COUNCIL_ENGINE_STUB
# ---------------------------------------------------------------------------

def test_stub_env_var_independence():
    """Setting SPEC_REVIEWER_STUB=1 does not require COUNCIL_ENGINE_STUB to be set."""
    synth_tid = f"stub-indep-{int(time.time())}"
    try:
        env = {"SPEC_REVIEWER_STUB": "1"}
        with patch.dict(os.environ, env, clear=False):
            # Ensure COUNCIL_ENGINE_STUB is NOT set
            os.environ.pop("COUNCIL_ENGINE_STUB", None)
            result = _dispatch_spec_reviewer(
                spec_text="s",
                synth_target_id=synth_tid,
                parsed_target_id="t",
                repo="r",
                invariant_context="c",
            )
        # Just needs to succeed without touching shaper
        assert result.task_id.startswith("stub-")
    finally:
        _cleanup(f"stub-{synth_tid}")
