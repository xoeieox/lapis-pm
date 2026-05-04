"""Unit tests for multi-tick prompt_hash rule (DoD §8).

Rule: envelope-level prompt_hash = sha256(null-byte-delimited concat of tick
prompts in tick order).  Per-tick hashes live in payload.tick_prompt_hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SMOKE_YAML = FIXTURE_DIR / "smoke.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_prompt_hash(prompts: list[str]) -> str:
    combined = b"\x00".join(p.encode("utf-8") for p in prompts)
    return "sha256:" + hashlib.sha256(combined).hexdigest()


def _expected_tick_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class MockLLM:
    model = "mock"

    def chat(self, system: str, messages: list) -> str:  # type: ignore[override]
        return json.dumps({
            "scenario_generated": "test scenario",
            "execution_trace": "executed",
            "breaks_observed": [],
            "leverage_points": [],
            "surprises": [],
            "drift_signals": [],
            "tools_used": {},
            "tools_wished_for": [],
            "performance_assessment": "nominal",
        })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPromptHash:
    def test_prompt_hash_is_null_byte_delimited(self, tmp_path: Path) -> None:
        """Envelope prompt_hash == sha256(null-byte-delimited tick prompts)."""
        from lapis_pm.scout.runner import run_single
        from lapis_pm.scout.scaffold import load_scaffold

        scaffold = load_scaffold(SMOKE_YAML)
        cell_params = scaffold.cell_params()[0]
        ltr, _ = run_single(scaffold, cell_params, seed=0, llm=MockLLM())

        tick_prompts = ltr.payload.tick_prompts
        assert tick_prompts, "Expected at least one tick prompt"

        expected = _expected_prompt_hash(tick_prompts)
        actual = ltr.provenance.prompt_hash
        assert actual == expected, (
            f"prompt_hash mismatch:\n  expected: {expected}\n  actual:   {actual}"
        )

    def test_per_tick_hashes_match_individual_prompts(self, tmp_path: Path) -> None:
        """Each tick_prompt_hash[i] == sha256(tick_prompts[i])."""
        from lapis_pm.scout.runner import run_single
        from lapis_pm.scout.scaffold import load_scaffold

        scaffold = load_scaffold(SMOKE_YAML)
        cell_params = scaffold.cell_params()[0]
        ltr, _ = run_single(scaffold, cell_params, seed=0, llm=MockLLM())

        prompts = ltr.payload.tick_prompts
        hashes = ltr.payload.tick_prompt_hashes
        assert len(prompts) == len(hashes), (
            f"tick_prompts length {len(prompts)} != tick_prompt_hashes {len(hashes)}"
        )
        for i, (prompt, h) in enumerate(zip(prompts, hashes)):
            expected = _expected_tick_hash(prompt)
            assert h == expected, (
                f"tick_prompt_hashes[{i}] mismatch:\n  expected: {expected}\n  actual: {h}"
            )

    def test_prompt_hash_is_deterministic(self, tmp_path: Path) -> None:
        """Same ordered list of prompts → same hash (pure function of content)."""
        prompts = ["first tick prompt", "second tick prompt"]
        h1 = _expected_prompt_hash(prompts)
        h2 = _expected_prompt_hash(prompts)
        assert h1 == h2

    def test_prompt_hash_order_sensitive(self) -> None:
        """Reversing prompt order changes the hash (order matters)."""
        prompts = ["alpha", "beta"]
        h_forward = _expected_prompt_hash(prompts)
        h_reversed = _expected_prompt_hash(list(reversed(prompts)))
        assert h_forward != h_reversed, (
            "prompt_hash must be order-sensitive (null-byte delimited)"
        )

    def test_prompt_hash_single_tick(self, tmp_path: Path) -> None:
        """Single-tick run: prompt_hash == sha256 of that single prompt."""
        from lapis_pm.scout.runner import run_single
        from lapis_pm.scout.scaffold import load_scaffold

        scaffold = load_scaffold(SMOKE_YAML)
        cell_params = scaffold.cell_params()[0]
        ltr, _ = run_single(scaffold, cell_params, seed=99, llm=MockLLM())

        prompts = ltr.payload.tick_prompts
        if len(prompts) == 1:
            # Single tick: null-byte-delimited of one item = just that item's bytes
            expected = _expected_prompt_hash(prompts)
            assert ltr.provenance.prompt_hash == expected

    def test_tick_count_matches_scaffold_time_progression(self) -> None:
        """Number of ticks should equal the number of time_progression entries."""
        from lapis_pm.scout.scaffold import load_scaffold

        scaffold = load_scaffold(SMOKE_YAML)
        n_ticks = len(scaffold.static_scaffold.scenario.time_progression)
        assert n_ticks >= 1, "Smoke fixture must have at least 1 time_progression entry"

        from lapis_pm.scout.runner import run_single
        cell_params = scaffold.cell_params()[0]
        ltr, _ = run_single(scaffold, cell_params, seed=0, llm=MockLLM())

        assert len(ltr.payload.tick_prompts) == n_ticks, (
            f"Expected {n_ticks} ticks (from time_progression), "
            f"got {len(ltr.payload.tick_prompts)}"
        )
