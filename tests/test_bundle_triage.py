"""Tests for lapis_pm.bundle_triage (bundle-item-level-triage-v0).

Coverage:
- is_mechanical / classify_item: three-tier check (tiers 1-2), binary,
  fail-closed to fork on any ambiguity.
- extract_verification_contract / verify_behavioral_invariance: extraction-
  only, static, fail-closed to no-contract when no existing test is located.
- cook_item_to_spec: renders all four C7 anchored header lines within the
  first 50 lines, copies the contract verbatim (rendering, not analysis),
  and makes no cross-repo import (verified structurally by this file's own
  module-level import list, which names only lapis_pm.authority / stdlib /
  agents_core.room_paths).
- write_route_record: sidecar shape.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lapis_pm import bundle_triage as bt


# ---------------------------------------------------------------------------
# is_mechanical / classify_item
# ---------------------------------------------------------------------------

class TestIsMechanical:
    def test_true_for_concrete_fix_small_blast_radius(self):
        assert bt.is_mechanical(
            "Add `--glob path_glob` to the rg invocation.",
            ["agents_core/gw_agent.py"],
        ) is True

    def test_false_for_empty_fix_description(self):
        assert bt.is_mechanical("", ["agents_core/gw_agent.py"]) is False

    def test_false_for_placeholder_fix_description(self):
        for placeholder in ("TBD", "n/a", "  none  ", "unknown"):
            assert bt.is_mechanical(placeholder, ["a.py"]) is False

    def test_false_for_no_touched_surfaces(self):
        assert bt.is_mechanical("Fix the thing.", []) is False

    def test_false_for_held_path(self):
        assert bt.is_mechanical(
            "Split the constant.", ["lapis_pm/authority.py"],
        ) is False

    def test_false_for_blast_radius_exceeded(self):
        surfaces = [f"pkg/mod_{i}.py" for i in range(bt.MAX_BLAST_RADIUS_FILES + 1)]
        assert bt.is_mechanical("Rename the function everywhere.", surfaces) is False

    def test_true_at_blast_radius_boundary(self):
        surfaces = [f"pkg/mod_{i}.py" for i in range(bt.MAX_BLAST_RADIUS_FILES)]
        assert bt.is_mechanical("Rename the function everywhere.", surfaces) is True


class TestClassifyItem:
    def test_concrete_fix_classifies_mechanical(self):
        item = {
            "debt_id": "abc123",
            "fix_description": "Add `--glob path_glob` to the rg invocation.",
            "touched_surfaces": ["agents_core/gw_agent.py"],
        }
        cls, reason = bt.classify_item(item)
        assert cls == "mechanical"
        assert reason is None

    def test_no_fix_description_classifies_fork_static(self):
        item = {"debt_id": "x", "fix_description": None, "touched_surfaces": ["a.py"]}
        assert bt.classify_item(item) == ("fork", "static")

    def test_held_path_classifies_fork_static(self):
        item = {
            "debt_id": "x",
            "fix_description": "Change the interface.",
            "touched_surfaces": ["lapis_pm/authority.py"],
        }
        assert bt.classify_item(item) == ("fork", "static")

    def test_too_many_files_classifies_fork_blast_radius(self):
        item = {
            "debt_id": "x",
            "fix_description": "Rename across the tree.",
            "touched_surfaces": [f"m{i}.py" for i in range(bt.MAX_BLAST_RADIUS_FILES + 1)],
        }
        assert bt.classify_item(item) == ("fork", "blast-radius")

    def test_missing_fields_defaults_to_fork(self):
        # A vision/interface item with no fix_description/touched_surfaces
        # keys at all (real shape of items code_reviewer emits without a
        # [suggestion: ...] bracket) — fail-closed, never mechanical.
        item = {"debt_id": "x"}
        cls, reason = bt.classify_item(item)
        assert cls == "fork"
        assert reason == "static"


# ---------------------------------------------------------------------------
# extract_verification_contract / verify_behavioral_invariance
# ---------------------------------------------------------------------------

class TestVerificationContract:
    def test_none_when_no_repo(self):
        item = {"touched_surfaces": ["a.py"]}
        assert bt.extract_verification_contract(item) is None

    def test_none_when_no_touched_surfaces(self):
        item = {"repo": "myrepo", "touched_surfaces": []}
        assert bt.extract_verification_contract(item) is None

    def test_none_when_no_existing_test_located(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        item = {"repo": "myrepo", "touched_surfaces": ["pkg/mod.py"]}
        assert bt.extract_verification_contract(item) is None

    def test_found_via_top_level_tests_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_mod.py").write_text("def test_x(): pass\n")
        item = {"repo": "myrepo", "touched_surfaces": ["pkg/mod.py"]}
        contract = bt.extract_verification_contract(item)
        assert contract is not None
        assert "tests/test_mod.py" in contract
        assert "pkg/mod.py" in contract

    def test_found_via_colocated_test(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "test_mod.py").write_text("def test_x(): pass\n")
        item = {"repo": "myrepo", "touched_surfaces": ["pkg/mod.py"]}
        contract = bt.extract_verification_contract(item)
        assert contract is not None
        assert "pkg/test_mod.py" in contract

    def test_verify_behavioral_invariance_holds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_mod.py").write_text("def test_x(): pass\n")
        item = {"repo": "myrepo", "touched_surfaces": ["pkg/mod.py"]}
        holds, contract = bt.verify_behavioral_invariance(item)
        assert holds is True
        assert contract is not None

    def test_verify_behavioral_invariance_fails_closed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        item = {"repo": "myrepo", "touched_surfaces": ["pkg/mod.py"]}
        holds, contract = bt.verify_behavioral_invariance(item)
        assert holds is False
        assert contract is None


# ---------------------------------------------------------------------------
# cook_item_to_spec
# ---------------------------------------------------------------------------

class TestCookItemToSpec:
    ITEM = {
        "debt_id": "abc123",
        "repo": "myrepo",
        "fix_description": "Add `--glob path_glob` to the rg invocation.",
        "touched_surfaces": ["pkg/mod.py"],
        "verification_contract": "existing test `tests/test_mod.py` pins behavior for `pkg/mod.py` and passes unchanged",
        "source_pr": 42,
    }

    def test_target_id_deterministic(self):
        assert bt.mechanical_item_target_id("myrepo", "abc123") == "cr-bundle-item-myrepo-abc123"

    def test_renders_all_four_c7_anchors_in_first_50_lines(self, tmp_path):
        spec_path = bt.cook_item_to_spec(self.ITEM, out_dir=tmp_path)
        assert spec_path.exists()
        first_50 = "\n".join(spec_path.read_text().splitlines()[:50])
        assert re.search(r"^\*\*Target ID:\*\*\s+`cr-bundle-item-myrepo-abc123`\s*$", first_50, re.MULTILINE)
        assert re.search(r"^\*\*Repo:\*\*\s+`myrepo`\s*$", first_50, re.MULTILINE)
        assert re.search(r"^\*\*Authority:\*\*\s+advisory\b", first_50, re.MULTILINE)
        assert re.search(r"^\*\*Consumer:\*\*\s+\S", first_50, re.MULTILINE)

    def test_consumer_line_is_not_self_referential(self, tmp_path):
        spec_path = bt.cook_item_to_spec(self.ITEM, out_dir=tmp_path)
        text = spec_path.read_text()
        m = re.search(r"^\*\*Consumer:\*\*\s+(.+)$", text, re.MULTILINE)
        assert m is not None
        consumer = m.group(1).lower()
        assert "cr-bundle-item-myrepo-abc123" not in consumer

    def test_contract_copied_verbatim_into_dod(self, tmp_path):
        spec_path = bt.cook_item_to_spec(self.ITEM, out_dir=tmp_path)
        text = spec_path.read_text()
        assert self.ITEM["verification_contract"] in text
        # Lands in the DoD section, not just anywhere.
        dod_idx = text.index("## Definition of Done")
        assert text.index(self.ITEM["verification_contract"]) > dod_idx

    def test_deliverable_names_the_fix(self, tmp_path):
        spec_path = bt.cook_item_to_spec(self.ITEM, out_dir=tmp_path)
        text = spec_path.read_text()
        assert self.ITEM["fix_description"] in text

    def test_no_contract_renders_explicit_placeholder_never_silent(self, tmp_path):
        item = {**self.ITEM}
        del item["verification_contract"]
        spec_path = bt.cook_item_to_spec(item, out_dir=tmp_path)
        text = spec_path.read_text()
        assert "no Verification Contract recorded" in text

    def test_makes_no_cross_repo_import(self):
        """Structural check: the module never imports anything from another
        repo's package (e.g. conductor) — only lapis_pm/stdlib/agents_core."""
        source = Path(bt.__file__).read_text()
        assert "import conductor" not in source
        assert "from conductor" not in source
        assert "night_plan_manager" not in source


# ---------------------------------------------------------------------------
# write_route_record
# ---------------------------------------------------------------------------

class TestWriteRouteRecord:
    def test_sidecar_shape(self, tmp_path):
        spec_path = tmp_path / "cr-bundle-myrepo-2026-08-17.md"
        spec_path.write_text("dummy")
        records = [
            {"debt_id": "a", "class": "mechanical", "reason": None},
            {"debt_id": "b", "class": "fork", "reason": "static"},
        ]
        record_path = bt.write_route_record(spec_path, "myrepo", records)
        assert record_path == Path(str(spec_path) + ".route.json")
        payload = json.loads(record_path.read_text())
        assert payload["repo"] == "myrepo"
        assert payload["spec"] == spec_path.name
        assert payload["items"] == records

    def test_sidecar_excluded_from_bundle_discovery_glob(self, tmp_path):
        """The sidecar must never match the cr-bundle-*.md discovery glob —
        it would otherwise be (mis)treated as a new bundle spec."""
        spec_path = tmp_path / "cr-bundle-myrepo-2026-08-17.md"
        spec_path.write_text("dummy")
        bt.write_route_record(spec_path, "myrepo", [])
        assert list(tmp_path.glob("cr-bundle-*.md")) == [spec_path]
