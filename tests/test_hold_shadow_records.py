"""Tests for lapis-pm-hold-shadow-observer-v0's record schema, deterministic
classification, dedupe, and never-raise fault isolation.

Coverage (spec Definition of Done #1):
  - gate-outcome/v1 and hold-fact/v1 records both validate against their schema.
  - classify_hold's hold_class + would_have_action derivations are deterministic
    on fixture inputs covering every HOLD_CLASSES enum value.
  - The full WOULD_HAVE_ACTIONS closed enum (including values classify_hold
    itself never emits, e.g. pause_target/amend_spec_redispatch) is accepted by
    schema validation — the enum is closed for the real action space, not just
    for what v0's rules table currently produces.
  - dedupe: one record per target_id:brief_comment_id.
  - both hooks are never-raise under fault injection (disk-full/permission mocks).
"""
from __future__ import annotations

import json

import pytest

from lapis_pm import hold_shadow
from lapis_pm import hold_shadow_rules as rules
from lapis_pm.spec_review import _build_brief


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_spec_review_brief.py's local fixtures)
# ---------------------------------------------------------------------------

def _council_raw(status: str = "resolved") -> dict:
    return {
        "status": status,
        "landing": "test landing",
        "open_questions": ["q1"],
        "confidence": "converged",
        "positions": [{"position": "agree", "entity": "ent-0"}],
        "run_id": f"council-{status}",
    }


def _sonnet_raw(verdict: str = "clean") -> dict:
    return {"status": "processed", "verdict": verdict, "issues": [], "confidence": 0.9,
            "run_id": f"sonnet-{verdict}"}


def _facets_deliberation() -> dict:
    return {
        "synthesis": {
            "escalation_recommendation": "proceed",
            "consensus_level": "strong",
            "escalation_reason": "",
        },
        "methodology": {},
        "stances": [],
    }


# ---------------------------------------------------------------------------
# gate-outcome/v1 — schema + hook
# ---------------------------------------------------------------------------

def test_gate_outcome_record_validates_and_matches_brief(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=_council_raw("resolved"),
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=1.0,
        facets_deliberation=_facets_deliberation(),
        grounding_status="verified",
        grounding_resolved_sha="abc1234",
    )

    hold_shadow.observe_gate_outcome(brief, authority="hold")

    lines = hold_shadow.gate_outcomes_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    hold_shadow.validate_gate_outcome_record(record)  # must not raise

    assert record["schema_version"] == "gate-outcome/v1"
    assert record["spec_path"] == str(spec)
    assert record["repo"] == "lapis-pm"
    assert record["authority"] == "hold"
    assert record["combined_recommendation"] == brief.combined_recommendation
    assert record["council"]["status"] == "resolved"
    assert record["council"]["open_questions_n"] == 1
    assert record["council"]["blocks_n"] == 0
    assert record["facets"]["consensus_level"] == "strong"
    assert record["facets"]["escalation"] == "proceed"
    assert record["grounding"]["status"] == "verified"
    assert record["grounding"]["sha"] == "abc1234"
    # never fabricated: comment ts is Pacific microseconds, this record's ts_utc
    # is a fresh UTC stamp, never copied from anywhere else.
    assert record["ts_utc"].endswith("+00:00")


def test_gate_outcome_blocks_n_counts_block_positions(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")

    council = _council_raw("open")
    council["positions"] = [
        {"position": "block", "entity": "a"},
        {"position": "agree", "entity": "b"},
        {"position": "block", "entity": "c"},
    ]
    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"),
        council_raw=council,
        spec_path=spec,
        parsed_target_id="my-tid",
        repo="lapis-pm",
        elapsed_s=1.0,
    )
    hold_shadow.observe_gate_outcome(brief, authority="advisory")
    record = json.loads(hold_shadow.gate_outcomes_path().read_text(encoding="utf-8").splitlines()[0])
    assert record["council"]["blocks_n"] == 2


def test_observe_gate_outcome_never_raises_when_writes_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))

    def _boom(*a, **kw):
        raise PermissionError("simulated disk-full")

    monkeypatch.setattr(hold_shadow, "_append_jsonl", _boom)

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")
    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"), council_raw=_council_raw("resolved"),
        spec_path=spec, parsed_target_id="tid", repo="lapis-pm", elapsed_s=1.0,
    )
    # Must not raise even though every write (including the fault log) fails.
    hold_shadow.observe_gate_outcome(brief, authority="hold")


def test_observe_gate_outcome_logs_fault_when_primary_write_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    real_append = hold_shadow._append_jsonl

    def _flaky(path, record):
        if path == hold_shadow.gate_outcomes_path():
            raise OSError("simulated permission denied")
        return real_append(path, record)

    monkeypatch.setattr(hold_shadow, "_append_jsonl", _flaky)

    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\n", encoding="utf-8")
    brief = _build_brief(
        reference_raw=_sonnet_raw("clean"), council_raw=_council_raw("resolved"),
        spec_path=spec, parsed_target_id="tid", repo="lapis-pm", elapsed_s=1.0,
    )
    hold_shadow.observe_gate_outcome(brief, authority="hold")

    assert not hold_shadow.gate_outcomes_path().exists()
    faults = [json.loads(l) for l in hold_shadow.faults_path().read_text(encoding="utf-8").splitlines()]
    assert len(faults) == 1
    assert faults[0]["hook"] == "gate-outcome"
    assert faults[0]["schema_version"] == "hold-shadow-fault/v1"


# ---------------------------------------------------------------------------
# classify_hold — deterministic, every HOLD_CLASSES enum value reachable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reasons,kwargs,expected_class,expected_action", [
    (["held path(s) touched: infra/foo.sh"], {}, "held_path", "none"),
    (["screen verdict: needs-human"], {}, "needs_human_screen", "none"),
    (["review cycle budget exhausted (4/4)"], {}, "review_exhausted", "force_dispatch_retry"),
    (["fixer dispatch lost — worker never returned"], {}, "lost_dispatch", "force_dispatch_retry"),
    (["static checks passed, 12 LOC"], {"authority_level": "hold", "has_issues": False},
     "hold_authority_default", "merge_pr"),
    (["static checks passed, 12 LOC"], {"authority_level": "hold", "has_issues": True},
     "hold_authority_default", "force_dispatch_retry"),
    (["some unrecognized reason"], {"authority_level": "advisory"}, "other", "none"),
])
def test_classify_hold_deterministic(reasons, kwargs, expected_class, expected_action):
    hold_class, would_have_action, params, confidence = rules.classify_hold(reasons, **kwargs)
    assert hold_class == expected_class
    assert would_have_action == expected_action
    assert isinstance(params, dict)
    assert confidence in {rules.CONFIDENCE_HIGH, rules.CONFIDENCE_MEDIUM, rules.CONFIDENCE_LOW}

    # Same input, called again, yields the exact same output — no randomness,
    # no model call, no clock dependency.
    again = rules.classify_hold(reasons, **kwargs)
    assert again == (hold_class, would_have_action, params, confidence)


def test_classify_hold_fixtures_cover_every_hold_class():
    fixtures = [
        (["held path(s) touched: infra/foo.sh"], {}),
        (["screen verdict: needs-human"], {}),
        (["review cycle budget exhausted"], {}),
        (["fixer dispatch lost"], {}),
        (["static checks passed"], {"authority_level": "hold"}),
        (["unrecognized"], {"authority_level": "advisory"}),
    ]
    seen = {rules.classify_hold(r, **kw)[0] for r, kw in fixtures}
    assert seen == set(rules.HOLD_CLASSES)


@pytest.mark.parametrize("action", rules.WOULD_HAVE_ACTIONS)
def test_would_have_action_full_enum_accepted_by_schema(action):
    """The closed enum includes values classify_hold itself never emits in v0
    (pause_target, unbind_target, acknowledge_and_clear, adopt_pr,
    amend_spec_redispatch) — it's closed against the real action space
    (brief.py's _ACTION_KINDS + the two named additions), not just against
    what today's rules table produces. Schema validation must accept all of
    them so Thursday's boundary stays checkable against the actual action
    space even as the rules table grows."""
    record = _minimal_hold_fact_record(would_have_action=action)
    hold_shadow.validate_hold_fact_record(record)  # must not raise


def _minimal_hold_fact_record(**overrides) -> dict:
    record = {
        "schema_version": "hold-fact/v1",
        "record_id": "r1",
        "observed_at_utc": "2026-08-13T13:45:00+00:00",
        "target_id": "tid",
        "brief_comment_id": "b1",
        "hold_comment_id": "h1",
        "pr_number": 1,
        "repo": "lapis-pm",
        "pm_authority": "hold",
        "spec_bound_ts": "2026-08-01T00:00:00-07:00",
        "hold_class": "held_path",
        "hold_reasons_verbatim": ["held path(s) touched: infra/foo.sh"],
        "would_have_action": "none",
        "would_have_params": {},
        "confidence": "high",
        "dedupe_key": "tid:b1",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# hold-fact/v1 — schema + hook + dedupe + never-raise
# ---------------------------------------------------------------------------

def test_hold_fact_record_validates(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    hold_shadow.observe_hold_fact(
        target_id="tid", pr_number=42, repo="lapis-pm",
        hold_reasons=["held path(s) touched: infra/foo.sh"],
        hold_comment_id="hold-cid-1", brief_comment_id="brief-cid-1",
        pm_authority="hold", spec_bound_ts="2026-08-01T00:00:00-07:00",
    )
    lines = hold_shadow.hold_facts_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    hold_shadow.validate_hold_fact_record(record)  # must not raise

    assert record["target_id"] == "tid"
    assert record["pr_number"] == 42
    assert isinstance(record["pr_number"], int)
    assert record["repo"] == "lapis-pm"
    assert record["hold_comment_id"] == "hold-cid-1"
    assert record["brief_comment_id"] == "brief-cid-1"
    assert record["hold_class"] == "held_path"
    assert record["would_have_action"] == "none"
    assert record["dedupe_key"] == "tid:brief-cid-1"
    assert record["observed_at_utc"].endswith("+00:00")


def test_hold_fact_merge_pr_params_carry_pr_number(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    hold_shadow.observe_hold_fact(
        target_id="tid", pr_number=7, repo="lapis-pm",
        hold_reasons=["static checks passed, 5 LOC"],
        hold_comment_id="h1", brief_comment_id="b1",
        pm_authority="hold", spec_bound_ts="ts", has_issues=False,
    )
    record = json.loads(hold_shadow.hold_facts_path().read_text(encoding="utf-8").splitlines()[0])
    assert record["would_have_action"] == "merge_pr"
    assert record["would_have_params"] == {"pr": 7}


def test_hold_fact_dedupe_same_key_writes_once(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    kwargs = dict(
        target_id="tid", pr_number=1, repo="lapis-pm",
        hold_reasons=["held path(s) touched: infra/x"],
        hold_comment_id="h1", brief_comment_id="b1",
        pm_authority="hold", spec_bound_ts="ts",
    )
    hold_shadow.observe_hold_fact(**kwargs)
    hold_shadow.observe_hold_fact(**kwargs)
    lines = hold_shadow.hold_facts_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_hold_fact_dedupe_different_brief_comment_writes_twice(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    base = dict(
        target_id="tid", pr_number=1, repo="lapis-pm",
        hold_reasons=["held path(s) touched: infra/x"],
        hold_comment_id="h1", pm_authority="hold", spec_bound_ts="ts",
    )
    hold_shadow.observe_hold_fact(brief_comment_id="b1", **base)
    hold_shadow.observe_hold_fact(brief_comment_id="b2", **base)
    lines = hold_shadow.hold_facts_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_observe_hold_fact_never_raises_when_writes_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))

    def _boom(*a, **kw):
        raise OSError("simulated disk-full")

    monkeypatch.setattr(hold_shadow, "_append_jsonl", _boom)
    # Must not raise even though every write (including the fault log) fails.
    hold_shadow.observe_hold_fact(
        target_id="tid", pr_number=1, repo="lapis-pm",
        hold_reasons=["held path(s) touched: infra/x"],
        hold_comment_id="h1", brief_comment_id="b1",
        pm_authority="hold", spec_bound_ts="ts",
    )


def test_observe_hold_fact_logs_fault_when_primary_write_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    real_append = hold_shadow._append_jsonl

    def _flaky(path, record):
        if path == hold_shadow.hold_facts_path():
            raise PermissionError("simulated permission denied")
        return real_append(path, record)

    monkeypatch.setattr(hold_shadow, "_append_jsonl", _flaky)
    hold_shadow.observe_hold_fact(
        target_id="tid", pr_number=1, repo="lapis-pm",
        hold_reasons=["held path(s) touched: infra/x"],
        hold_comment_id="h1", brief_comment_id="b1",
        pm_authority="hold", spec_bound_ts="ts",
    )

    assert not hold_shadow.hold_facts_path().exists()
    faults = [json.loads(l) for l in hold_shadow.faults_path().read_text(encoding="utf-8").splitlines()]
    assert len(faults) == 1
    assert faults[0]["hook"] == "hold-fact"
