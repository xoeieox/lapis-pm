"""Tests for lapis_pm/contract.py (arc-registry-contract-schema-v0, D1/D1a).

Covers the hermetic fixer DoD items 1-6a: subject-agnosticism (inspection
test over the module source, not prose), the Eigen-compatibility round-trip,
"delta is computed, never persisted", delta correctness, "unobserved is a
delta, not a silence", and the empty-but-observed vs missing-observation
distinction (the round-2 Council "lie of omission" fold).
"""

from __future__ import annotations

import ast
import inspect

import pytest

from lapis_pm import contract as contract_mod
from lapis_pm.contract import Contract, Delta, Observation, SubjectRef, compute_delta


class TestSubjectAgnostic:
    """DoD 1: contract.py contains no import of state_brief, and no branch
    on subject.kind. Asserted by an inspection test over the module source."""

    def test_no_state_brief_import(self):
        source = inspect.getsource(contract_mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any("state_brief" in alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or "state_brief" not in node.module

    def test_no_branch_on_subject_kind(self):
        source = inspect.getsource(contract_mod)
        assert ".kind ==" not in source
        assert ".kind !=" not in source
        assert "subject.kind" not in source


class TestEigenCompatibilityRoundTrip:
    """DoD 2: the same construct/serialize/deserialize path works for
    kind="arc" and a second kind, with no code change."""

    @pytest.mark.parametrize("kind,subj_id", [
        ("arc", "my-arc-slug"),
        ("simulation-parameters", "eigen-run-42"),
    ])
    def test_round_trip_identical_for_any_kind(self, kind, subj_id):
        c = Contract(
            subject=SubjectRef(kind=kind, id=subj_id),
            declaration="should do X",
            elaboration={"k1": "X happened"},
            observed=(Observation(source="pm/landed", key="k1", value="X happened", ts="2026-07-27T00:00:00+00:00"),),
        )
        d = c.to_dict()
        restored = Contract.from_dict(d)
        assert restored == c
        assert restored.subject.kind == kind
        assert compute_delta(restored) == compute_delta(c)


class TestDeltaComputedNeverPersisted:
    """DoD 3: serialization omits `delta`; deserializing then recomputing
    yields the same deltas."""

    def test_to_dict_omits_delta_key(self):
        c = Contract(
            subject=SubjectRef(kind="arc", id="s"),
            declaration="should do X",
            elaboration={"k1": "X happened"},
            observed=(),
        )
        d = c.to_dict()
        assert set(d.keys()) == {"subject", "declaration", "elaboration", "observed"}
        assert "delta" not in d

    def test_round_trip_then_recompute_matches(self):
        c = Contract(
            subject=SubjectRef(kind="arc", id="s"),
            declaration="should do X",
            elaboration={"k1": "X happened", "k2": "Y happened"},
            observed=(Observation(source="pm/landed", key="k1", value="X happened", ts="t"),),
        )
        before = compute_delta(c)
        restored = Contract.from_dict(c.to_dict())
        after = compute_delta(restored)
        assert before == after

    def test_delta_never_serialized_even_if_field_existed_transiently(self):
        c = Contract(
            subject=SubjectRef(kind="arc", id="s"),
            declaration="should do X",
            elaboration={"k1": "X happened"},
        )
        assert not hasattr(c, "delta")


class TestDeltaCorrectness:
    """DoD 4: satisfied elaboration yields no delta; contradicted elaboration
    yields exactly one delta naming the failing elaboration."""

    def test_satisfied_elaboration_yields_no_delta(self):
        c = Contract(
            subject=SubjectRef(kind="arc", id="s"),
            declaration="should ship the feature",
            elaboration={"shipped": "feature shipped"},
            observed=(Observation(source="pm/landed", key="shipped", value="feature shipped", ts="t"),),
        )
        assert compute_delta(c) == ()

    def test_contradicted_elaboration_yields_exactly_one_delta(self):
        c = Contract(
            subject=SubjectRef(kind="arc", id="s"),
            declaration="should ship the feature",
            elaboration={"shipped": "feature shipped"},
            observed=(Observation(source="pm/landed", key="shipped", value="feature NOT shipped, blocked", ts="t"),),
        )
        deltas = compute_delta(c)
        assert len(deltas) == 1
        assert deltas[0].kind == "contradicted"
        assert deltas[0].key == "shipped"


class TestUnobservedIsADeltaNotASilence:
    """DoD 5: a contract whose elaborations have zero matching observations
    returns N kind="unobserved" deltas -- never an empty tuple."""

    def test_zero_observations_yields_n_unobserved_deltas(self):
        c = Contract(
            subject=SubjectRef(kind="arc", id="s"),
            declaration="should do X and Y",
            elaboration={"x": "X happened", "y": "Y happened"},
            observed=(),
        )
        deltas = compute_delta(c)
        assert len(deltas) == 2
        assert all(d.kind == "unobserved" for d in deltas)
        assert deltas != ()


class TestEmptyButObservedVsUnobserved:
    """DoD 6a: an Observation present with value=None against a positive
    elaboration yields Delta(kind="contradicted"); a *missing* observation
    for that key yields Delta(kind="unobserved"). Both in one test so the
    distinction cannot rot."""

    def test_empty_but_observed_vs_never_observed(self):
        c = Contract(
            subject=SubjectRef(kind="arc", id="s"),
            declaration="should do X and Y",
            elaboration={"x": "X happened", "y": "Y happened"},
            observed=(Observation(source="pm/landed", key="x", value=None, ts="t"),),
        )
        deltas = {d.key: d for d in compute_delta(c)}
        assert len(deltas) == 2
        assert deltas["x"].kind == "contradicted"  # looked, found nothing: lie of omission
        assert deltas["y"].kind == "unobserved"    # never looked: silence
