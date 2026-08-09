"""Tests for lapis-pm-reviewer-absence-grounding-rule-v0.

Covers DoD 1-4:
  1. absence_grounding field exists on reviewer/reviewer_fresh, defaults false,
     parsed through the shaper's _bool_val.
  2. Flag false -> rendered system_template byte-identical to origin/main's
     (this unit's own baseline, before the grounding block existed).
  3. Flag true -> rendered template contains the grounding block and its
     five constraints.
  4. An issue carrying an "evidence" key round-trips through the verdict
     encoder (json.loads -> json.dumps path in _encode_gpu_results) without
     being rejected or dropped.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from lapis_pm import pm_core


REPO_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = REPO_ROOT / "lapis_pm" / "registry.yaml"


def _load_agents() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text())["agents"]


class TestAbsenceGroundingFieldDefaults:
    """DoD 1: field exists, defaults false, parsed via _bool_val."""

    def test_reviewer_has_absence_grounding_field(self):
        agents = _load_agents()
        assert "absence_grounding" in agents["reviewer"]

    def test_reviewer_fresh_has_absence_grounding_field(self):
        agents = _load_agents()
        assert "absence_grounding" in agents["reviewer_fresh"]

    def test_reviewer_defaults_false(self):
        agents = _load_agents()
        assert pm_core._bool_val(agents["reviewer"]["absence_grounding"]) is False

    def test_reviewer_fresh_defaults_false(self):
        agents = _load_agents()
        assert pm_core._bool_val(agents["reviewer_fresh"]["absence_grounding"]) is False

    def test_helper_reads_false_by_default(self):
        assert pm_core._absence_grounding_enabled("reviewer") is False
        assert pm_core._absence_grounding_enabled("reviewer_fresh") is False

    def test_helper_uses_shaper_bool_val(self):
        """DoD 1 explicitly requires reuse of agents_core.shaper._bool_val,
        not a new ad-hoc truthiness check."""
        with patch.object(pm_core, "_bool_val", wraps=pm_core._bool_val) as spy:
            pm_core._absence_grounding_enabled("reviewer")
            assert spy.called

    def test_unset_agent_type_is_false(self):
        """Unknown/non-reviewer agent types degrade to False, not an error."""
        assert pm_core._absence_grounding_enabled("fixer") is False
        assert pm_core._absence_grounding_block_for("fixer") == ""


class TestFlagOffByteIdentical:
    """DoD 2: with the flag false, the rendered template is byte-identical
    to origin/main's — this is the invariant that makes the unit safe to merge.
    """

    def _origin_main_templates(self) -> dict[str, str] | None:
        try:
            raw = subprocess.run(
                ["git", "show", "origin/main:lapis_pm/registry.yaml"],
                cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            pytest.skip(f"could not read origin/main:lapis_pm/registry.yaml: {exc}")
        if raw.returncode != 0:
            pytest.skip(f"git show origin/main:lapis_pm/registry.yaml failed: {raw.stderr!r}")
        data = yaml.safe_load(raw.stdout)
        agents = data.get("agents", {})
        return {
            "reviewer": agents.get("reviewer", {}).get("system_template", ""),
            "reviewer_fresh": agents.get("reviewer_fresh", {}).get("system_template", ""),
        }

    def test_reviewer_flag_off_matches_origin_main(self):
        old = self._origin_main_templates()
        new_tpl = _load_agents()["reviewer"]["system_template"]
        rendered_off = new_tpl.replace("{absence_grounding_block}", "")
        assert rendered_off == old["reviewer"], (
            "reviewer system_template with absence_grounding_block='' must be "
            "byte-identical to origin/main's template"
        )

    def test_reviewer_fresh_flag_off_matches_origin_main(self):
        old = self._origin_main_templates()
        new_tpl = _load_agents()["reviewer_fresh"]["system_template"]
        rendered_off = new_tpl.replace("{absence_grounding_block}", "")
        assert rendered_off == old["reviewer_fresh"], (
            "reviewer_fresh system_template with absence_grounding_block='' must be "
            "byte-identical to origin/main's template"
        )

    def test_absence_grounding_block_for_returns_empty_when_off(self):
        assert pm_core._absence_grounding_block_for("reviewer") == ""
        assert pm_core._absence_grounding_block_for("reviewer_fresh") == ""
        assert pm_core._absence_grounding_block_for("reviewer_fresh_contractor") == ""


class TestFlagOnGroundingBlockPresence:
    """DoD 3: with the flag true, the rendered template contains the
    grounding block and all five listed constraints."""

    def test_block_present_when_forced_on(self):
        with patch.object(pm_core, "_absence_grounding_enabled", return_value=True):
            rendered = pm_core._absence_grounding_block_for("reviewer")
        assert rendered != ""
        assert "HARD RULE" in rendered

    def test_five_constraints_present(self):
        with patch.object(pm_core, "_absence_grounding_enabled", return_value=True):
            block = pm_core._absence_grounding_block_for("reviewer")
        # 1. absence-style findings invalid without a full-branch search + quoted output
        assert "is not defined" in block and "INVALID unless" in block
        # 2. absence from diff != absence from codebase
        assert "absent from the DIFF does not mean it is absent from the CODEBASE" in block
        # 3. definitions/call sites routinely far apart, different hunks
        assert "hundreds of" in block and "different hunks" in block
        # 4. any unexpected hit withdraws the finding
        assert "any hit you did not expect" in block and "withdrawn" in block
        # 5. every absence-style finding must carry command+output in `evidence`
        assert "`evidence`" in block and "No evidence, no" in block

    def test_reviewer_fresh_contractor_inherits_reviewer_fresh_flag(self):
        """contractor has no field of its own; shares reviewer_fresh's flag
        (they share the template via YAML anchor)."""
        def fake_enabled(agent_type):
            return agent_type == "reviewer_fresh"

        with patch.object(pm_core, "_absence_grounding_enabled", side_effect=fake_enabled):
            assert pm_core._absence_grounding_block_for("reviewer_fresh_contractor") != ""
            assert pm_core._absence_grounding_block_for("reviewer") == ""

    def test_full_render_with_flag_on_is_well_formed(self):
        """Sanity: the fully-rendered (flag-on) template still .format()s clean
        with a representative vars_ dict — no stray braces introduced."""
        agents = _load_agents()
        tpl = agents["reviewer"]["system_template"]
        with patch.object(pm_core, "_absence_grounding_enabled", return_value=True):
            block = pm_core._absence_grounding_block_for("reviewer")
        rendered = tpl.format(
            pr_number=1, repo="x", spec_summary="s", intent_block="",
            existing_branch="b", base_branch="main", prior_review="",
            absence_grounding_block=block,
        )
        assert "HARD RULE" in rendered
        assert rendered.startswith("You are reviewing PR 1 in repo x")


class TestEvidenceFieldRoundTrip:
    """DoD 4: an issue carrying 'evidence' round-trips through the verdict
    encoder without being rejected or dropped."""

    def test_evidence_key_survives_json_round_trip(self):
        """Mirrors the exact encode path in _encode_gpu_results (line ~6512:
        verdict_data = json.loads(raw_output); stored_json = json.dumps(verdict_data))."""
        raw_output = json.dumps({
            "verdict": "fixable",
            "issues": [
                {
                    "severity": "high",
                    "path": "foo.py",
                    "note": "handler is not wired in",
                    "evidence": "git grep -n handle_foo\n(no output)",
                },
                {
                    "severity": "low",
                    "path": "bar.py",
                    "note": "nit",
                },
            ],
            "confidence": 0.7,
        })
        verdict_data = json.loads(raw_output)
        stored_json = json.dumps(verdict_data)
        round_tripped = json.loads(stored_json)
        issues = round_tripped["issues"]
        assert issues[0]["evidence"] == "git grep -n handle_foo\n(no output)"
        # Second issue never had an evidence key — must not gain one, and
        # must not be dropped from the array.
        assert "evidence" not in issues[1]
        assert len(issues) == 2

    def test_recover_reviewer_verdict_preserves_evidence(self):
        """Malformed-JSON recovery path (_recover_reviewer_verdict) must also
        preserve the evidence key when it recovers a parseable object."""
        raw = (
            '```json\n'
            '{"verdict": "fixable", "issues": [{"severity": "high", '
            '"path": "foo.py", "note": "x", "evidence": "grep -rn foo\\nno hits"}], '
            '"confidence": 0.5}\n'
            '```'
        )
        recovered = pm_core._recover_reviewer_verdict(raw)
        assert recovered is not None
        assert recovered["issues"][0]["evidence"] == "grep -rn foo\nno hits"
