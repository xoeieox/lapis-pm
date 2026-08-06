"""Tests for lapis-pm-empiricist-reference-seat-v0.

Covers:
  1. spec_reviewer.system_template no longer solicits design judgment.
  2. spec_reviewer is pinned to slot 1 (:8081), lease unchanged.
  3. The citation-not-existence discipline is present in the template.

DoD item 3 (the planted-false-claims regression corpus against a live model
call) is a separate, model-in-the-loop concern and is exercised by the live
gate run (DoD item 5), not here.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"

# Exact headings/phrases from the old five-job design-judge template. Matched
# in this precise form (not loose substrings like "premature" or
# "implementability" alone) because the new Empiricist template legitimately
# uses some of those bare words in its own hard-invariant prohibition text
# (e.g. "...the abstraction premature...") and reuses the unchanged category
# enum (which still lists "implementability" as a category tag, per the
# spec's non-goal of not touching the existing JSON schema). What must be
# gone is the old job-description headings that solicited that judgment.
FORBIDDEN_DESIGN_JUDGMENT_STRINGS = [
    "Edge cases:",
    "over-engineering paths",
    "premature generalizations",
    "DoD coverage:",
    "Implementability:",
]


def _load_spec_reviewer() -> dict:
    data = yaml.safe_load(REGISTRY_PATH.read_text())
    return data["agents"]["spec_reviewer"]


def _template() -> str:
    return _load_spec_reviewer()["system_template"]


class TestNoDesignJudgment:

    def test_template_has_no_forbidden_design_judgment_strings(self):
        tpl = _template()
        present = [s for s in FORBIDDEN_DESIGN_JUDGMENT_STRINGS if s in tpl]
        assert not present, (
            "spec_reviewer.system_template still solicits design judgment "
            f"via: {present} — this seat must be re-charactered as the "
            "Empiricist (lapis-pm-empiricist-reference-seat-v0)"
        )

    def test_template_states_explicit_prohibition_on_design_commentary(self):
        tpl = _template()
        assert "no design judgment" in tpl.lower(), (
            "spec_reviewer.system_template must state an explicit hard "
            "invariant prohibiting design commentary"
        )


class TestClaimVerificationCharacter:

    def test_template_names_empiricist(self):
        assert "EMPIRICIST" in _template()

    def test_template_has_claim_classes(self):
        tpl = _template()
        for claim_class in ["file path", "line citation", "symbol name", "config"]:
            assert claim_class in tpl, f"missing claim class: {claim_class!r}"

    def test_template_has_repo_vs_host_discipline(self):
        tpl = _template()
        assert "HOST" in tpl and "FALSE POSITIVE" in tpl

    def test_template_has_limit_of_verification_discipline(self):
        tpl = _template()
        assert "does not exceed it" in tpl.lower() or "limit of your verification" in tpl.lower()

    def test_template_has_citation_not_existence_rule(self):
        """Gate pass 2 amendment: a wrong citation must not be escalated to
        'the symbol does not exist' without positively established absence."""
        tpl = _template()
        assert "does not mean the symbol does not exist" in tpl or (
            "citation" in tpl.lower() and "positively established absence" in tpl.lower()
        )

    def test_template_requires_claims_checked_output_field(self):
        tpl = _template()
        assert "claims_checked" in tpl

    def test_template_claims_checked_counts_extractions_not_failures(self):
        tpl = _template()
        assert "EXTRACTED AND CHECKED" in tpl or "extractions, not failures" in tpl.lower()

    def test_template_no_prescriptive_rerun_instruction(self):
        """Facets condition from gate pass 2: the prohibition on prescriptive
        're-run' instructions extends to the new template."""
        tpl = _template()
        assert "re-run spec-review" not in tpl.lower()


class TestSlot1Pin:

    def test_model_is_slot1(self):
        assert _load_spec_reviewer()["model"] == "gravitywell-slot1"

    def test_backend_url_is_slot1_port(self):
        assert _load_spec_reviewer()["backend_url"].endswith(":8081")

    def test_acquire_lease_still_false(self):
        assert _load_spec_reviewer()["acquire_lease"] is False

    def test_backend_url_present_and_nonempty(self):
        # Regression test for the _is_swarm coupling (agents_core/gw_agent.py:1404):
        # backend_url must stay present and non-empty even though its value now
        # equals the GW_URL default, or _is_swarm silently flips and changes this
        # seat's inference payload. See registry.yaml's spec_reviewer comment block.
        backend_url = _load_spec_reviewer()["backend_url"]
        assert backend_url is not None
        assert backend_url != ""
