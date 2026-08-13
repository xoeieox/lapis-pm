"""Tests for lapis-pm-hold-shadow-observer-v0's Thursday summary pass
(hold_shadow_summary.py — spec Scope item 4, DoD #6).

Coverage:
  - build_summary groups by hold_class / combined_recommendation (the
    ratified "grouped by proposed classification" process change).
  - the deposited gem payload never writes a pm/brief-gem/map/<gem_id> key
    (this module doesn't even import anything that could write mem) — so
    reconcile_decided_gems structurally cannot execute it.
  - an empty log (small-first-night risk, accepted/ratified) still produces
    a valid gem payload rather than erroring.
  - the network POST is fail-soft: any failure returns None, never raises.
"""
from __future__ import annotations

import inspect
import json

import pytest

from lapis_pm import hold_shadow_summary as summary


def _gate_outcome(recommendation: str) -> dict:
    return {
        "schema_version": "gate-outcome/v1",
        "combined_recommendation": recommendation,
    }


def _hold_fact(hold_class: str) -> dict:
    return {
        "schema_version": "hold-fact/v1",
        "hold_class": hold_class,
    }


def _enforce_outcome(enforce_class: str) -> dict:
    return {
        "schema_version": "enforce-outcome/v1",
        "enforce_class": enforce_class,
    }


def test_build_summary_groups_by_classification():
    gate_outcomes = [
        _gate_outcome("proceed-to-bind"),
        _gate_outcome("proceed-to-bind"),
        _gate_outcome("amend-spec"),
    ]
    hold_facts = [
        _hold_fact("held_path"),
        _hold_fact("hold_authority_default"),
        _hold_fact("held_path"),
    ]
    result = summary.build_summary(gate_outcomes, hold_facts)

    assert result["gate_outcomes_total"] == 3
    assert result["hold_facts_total"] == 3
    assert result["by_combined_recommendation"] == {"proceed-to-bind": 2, "amend-spec": 1}
    assert result["by_hold_class"] == {"held_path": 2, "hold_authority_default": 1}
    assert len(result["hold_facts_grouped"]["held_path"]) == 2


def test_build_summary_empty_inputs_do_not_error():
    result = summary.build_summary([], [])
    assert result["gate_outcomes_total"] == 0
    assert result["hold_facts_total"] == 0
    assert result["by_hold_class"] == {}
    assert result["by_combined_recommendation"] == {}
    # enforce_outcomes is optional/defaulted — omitting it entirely (the
    # pre-enforce two-arg call shape) must not error and must still
    # produce a well-formed (empty) enforce grouping.
    assert result["enforce_outcomes_total"] == 0
    assert result["by_enforce_class"] == {}


def test_build_summary_groups_enforce_outcomes_by_class():
    """lapis-pm-bundle-autodispatch-enforce-v0: enforce-outcomes group by
    enforce_class (infra/salvage/defer) alongside the existing groupings."""
    enforce_outcomes = [
        _enforce_outcome("infra"),
        _enforce_outcome("salvage"),
        _enforce_outcome("salvage"),
        _enforce_outcome("defer"),
    ]
    result = summary.build_summary([], [], enforce_outcomes)

    assert result["enforce_outcomes_total"] == 4
    assert result["by_enforce_class"] == {"infra": 1, "salvage": 2, "defer": 1}
    assert len(result["enforce_outcomes_grouped"]["salvage"]) == 2


def test_render_gem_payload_shape_and_deposited_by():
    result = summary.build_summary([_gate_outcome("proceed-to-bind")], [_hold_fact("held_path")])
    payload = summary._render_gem_payload(result)

    assert payload["deposited_by"] == "hold-shadow-observer-v0"
    assert payload["source_thread_id"] == "hold-shadow-observer-v0"
    assert "held_path" in payload["why"]
    assert "proceed-to-bind" in payload["why"]
    # No directive-shaped or brief-gem-mappable fields anywhere in the payload.
    assert "brief_id" not in payload
    assert "option_id" not in payload
    assert "gem_id" not in payload


def test_render_gem_payload_includes_enforce_grouping():
    result = summary.build_summary([], [], [_enforce_outcome("salvage")])
    payload = summary._render_gem_payload(result)
    assert "enforce" in payload["why"].lower()
    assert "salvage: 1" in payload["why"]


def test_render_gem_payload_empty_log_still_valid():
    result = summary.build_summary([], [])
    payload = summary._render_gem_payload(result)
    assert payload["title"]
    assert payload["ask"]
    assert "No records overnight" in payload["ask"]


def test_run_thursday_summary_reads_jsonl_files(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    hs_dir = tmp_path / "hold-shadow"
    hs_dir.mkdir(parents=True)
    (hs_dir / "gate-outcomes.jsonl").write_text(
        json.dumps(_gate_outcome("proceed-to-bind")) + "\n", encoding="utf-8",
    )
    (hs_dir / "hold-facts.jsonl").write_text(
        json.dumps(_hold_fact("held_path")) + "\n" + "not-json\n",
        encoding="utf-8",
    )
    (hs_dir / "enforce-outcomes.jsonl").write_text(
        json.dumps(_enforce_outcome("infra")) + "\n", encoding="utf-8",
    )

    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"gem_id": "gem-123"}

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)

    gem_id = summary.run_thursday_summary()

    assert gem_id == "gem-123"
    assert captured["url"].endswith("/v0/decision-gems")
    assert captured["payload"]["deposited_by"] == "hold-shadow-observer-v0"
    # A torn line ("not-json") never aborts the pass — the one valid hold-fact
    # record still made it into the summary.
    assert "held_path" in captured["payload"]["why"]
    # enforce-outcomes.jsonl is read alongside the two existing files.
    assert "infra: 1" in captured["payload"]["why"]


def test_run_thursday_summary_fail_soft_on_network_error(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            raise ConnectionError("weaver unreachable")

    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)

    assert summary.run_thursday_summary() is None


def test_summary_module_never_writes_brief_gem_map_key():
    """This module must not even be able to write pm/brief-gem/map/ — it
    imports no mem client at all (see test_hold_shadow_boundary.py for the
    full import-boundary assertion); this test pins the specific claim the
    DoD makes: the deposited gem carries zero machine-executable linkage.

    Checks real code strings only (AST, excluding docstrings) — the module's
    own docstring names "pm/brief-gem/map/" in prose to explain the
    guarantee, which a raw substring-in-source check would misfire on.
    """
    import ast

    tree = ast.parse(inspect.getsource(summary), filename=summary.__file__)
    docstring_ids = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    code_strings = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstring_ids
    ]
    joined = "\n".join(code_strings)
    assert "mem.set" not in joined
    assert "_forward_key" not in joined
    assert "brief-gem/map" not in joined
