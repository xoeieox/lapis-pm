"""D5 (fixer-reception-v0, leg 2): the `auditor` registry entry.

The auditor is a first-class LLM role (Erah 2026-09-07, ratified) — the
LLM-powered receiving role for non-passing fixer terminations. Its entry
lives in the REPO copy of the registry (lapis_pm/registry.yaml) and is
carried into the deploy clone by the 10-min ff-only sync — these tests pin
the entry's shape so a registry edit that drops or reshapes it fails here
instead of at the first live audit.

Coverage (spec AC4, registry half):
  - the entry exists under `agents:` with engine `local-auditor`
  - the local-lane seat (model gravitywell-slot1 — all-local lanes,
    2026-08-22 stopgap flip, durable 2026-08-31)
  - max_steps 100 + timeout_s 1800 (plumbed to the engine per D4c; the
    reviewer engine's 24-step default would kill a 35-70-step audit)
  - the notify policy (infra-only — the audit brief is advisory, not a
    wake-up)
  - the system_template brace-doubling contract: the JSON contract section
    carries {{ }} so str.format interpolates the dispatch vars and leaves
    the JSON braces literal (the render site processes only the template;
    substituted values render literally — no format-injection surface)
  - the template carries the mandate's named clauses (three-head in-place
    checkout, the repeated-run_tests prohibition, the honest
    verified:false contract, the five output fields)
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"

# The D5 dispatch vars the daemon resolves at dispatch time (pm_core
# _act_dispatch_auditor) — every one must be interpolable from the
# template via str.format.
D5_DISPATCH_VARS = {
    "target_id": "tid-test",
    "repo": "conductor",
    "repo_cwd": "/srv/git/conductor-working",
    "pr_number": 945,
    "existing_branch": "lapis/tid-test/local-salvage-19585ea0",
    "salvage_head_sha": "19585ea01234567890abcdef1234567890abcdef",
    "previous_head_sha": "abcdef1234567890abcdef1234567890abcdef12",
    "baseline_main_sha": "0123456789abcdef0123456789abcdef01234567",
    "bound_spec_path": "/srv/lapis/planning/specs/fixer-reception-v0.md",
}


def _auditor_entry() -> dict:
    data = yaml.safe_load(REGISTRY_PATH.read_text())
    agents = data.get("agents")
    assert isinstance(agents, dict), "registry must carry an `agents:` mapping"
    assert "auditor" in agents, (
        "the `auditor` entry is missing from lapis_pm/registry.yaml "
        "(fixer-reception-v0 leg 2, D5)"
    )
    return agents["auditor"]


def test_auditor_entry_engine_is_local_auditor():
    """D4: the primary local-auditor engine (rev-2 redesign)."""
    assert _auditor_entry()["engine"] == "local-auditor"


def test_auditor_entry_local_lane_seat():
    """All-local lanes (2026-08-22 stopgap flip, durable 2026-08-31): the
    auditor joins the local tier — no new seat, no new model."""
    assert _auditor_entry()["model"] == "gravitywell-slot1"


def test_auditor_entry_max_steps_and_timeout():
    """AC4: the resolved auditor spec carries max_steps 100 +
    timeout_s 1800 (the existing shaped call-site default, not the
    rev-1 900s). The reviewer engine's 24-step default would kill a
    35-70-step audit at step 24 — the registry values are the
    D4c plumbing source."""
    entry = _auditor_entry()
    assert entry["max_steps"] == 100
    assert entry["timeout_s"] == 1800


def test_auditor_entry_notify_policy():
    """The audit brief is advisory (consumed at dispatch decisions); the
    notify policy is infra-only — a completed audit does not wake a human,
    a failed/infra one does."""
    entry = _auditor_entry()
    assert entry["notify"] is True
    assert entry["notify_policy"] == "infra-only"


def test_auditor_template_brace_doubling_contract():
    """The brace-doubling contract (spec D5 + the panel's clean-area
    finding): the template's JSON contract section uses {{ }} so
    str.format leaves the JSON braces literal. Two assertions:

    1. every single brace in the template is part of a {var} dispatch
       field (no stray single braces that would KeyError at render time);
    2. after str.format with the D5 dispatch vars, the rendered template
       still carries the doubled-brace JSON skeleton (the literal braces
       of the five-field contract survive rendering).
    """
    template = _auditor_entry()["system_template"]
    # (1) No stray single braces: re-find all { ... } groups that are NOT
    # doubled and NOT a doubled-brace escape. str.format itself is the
    # oracle — it raises on a malformed single brace.
    rendered = template.format(**D5_DISPATCH_VARS)
    # (2) The JSON contract's braces survived rendering as literal braces.
    assert '"suite_states": {' in rendered
    assert '"failure_set_delta": {' in rendered
    assert '"root_cause": [' in rendered
    assert '"delta_to_green": [' in rendered
    assert '"deliverables_evaluation": [' in rendered
    # The doubled-brace source is present in the RAW template (the
    # contract itself — a template edit that "un-doubles" the JSON
    # braces breaks str.format and is caught by (1) above).
    assert '{{' in template


def test_auditor_template_interpolates_all_d5_dispatch_vars():
    """Every D5 dispatch var the template references is a named field
    (a template reference the daemon never supplies KeyErrors at render
    — pinned by str.format with exactly the D5 dispatch vars), and the
    core dispatch vars are actually interpolated (a var the template
    never references is dead plumbing)."""
    template = _auditor_entry()["system_template"]
    # The core dispatch vars must be named fields.
    for var in ("target_id", "repo", "pr_number", "existing_branch",
                "salvage_head_sha", "previous_head_sha", "baseline_main_sha",
                "bound_spec_path"):
        assert re.search(rf"(?<!\{{)\{{{var}\}}(?!\}})", template), (
            f"template must reference {{{var}}} (D5 dispatch var)"
        )
    # And it renders cleanly with exactly the D5 dispatch vars (no
    # reference to a var the daemon does not supply).
    rendered = template.format(**D5_DISPATCH_VARS)
    # Interpolation actually happened (not just a clean no-op render).
    assert "tid-test" in rendered
    assert str(D5_DISPATCH_VARS["pr_number"]) in rendered
    assert D5_DISPATCH_VARS["salvage_head_sha"] in rendered


def test_auditor_template_mandate_clauses():
    """The mandate's named clauses (spec D5 + the gate's Mirror Council
    clarifications, run 2026-09-07-224357):

    - strict read-only observer of a single ephemeral worktree (the
      Council's landing — the secondary-worktree / second-front shape is
      explicitly rejected);
    - the three-head suite runs via in-place git checkout;
    - repeated identical run_tests calls are FORBIDDEN (the no-progress
      guard would kill the auditor for spinning);
    - verified: false is a valid, honest output (code illegible / cause
      non-deterministic), NOT a failed audit;
    - the five output fields the daemon encodes (D5 output contract).
    """
    template = _auditor_entry()["system_template"]
    assert "STRICT READ-ONLY OBSERVER" in template
    assert "single ephemeral worktree" in template
    assert "secondary-worktree" in template  # the rejected shape, named
    assert "in-place `git checkout`" in template
    assert "REPEATED IDENTICAL `run_tests` CALLS ARE FORBIDDEN" in template
    assert "verified: false" in template
    for field in (
        "suite_states",
        "failure_set_delta",
        "root_cause",
        "delta_to_green",
        "deliverables_evaluation",
    ):
        assert f'"{field}"' in template
