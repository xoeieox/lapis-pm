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
import time
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

def _green_baseline(test_rel: str, surface: str, main_sha: str = "abc1234def5678") -> "bt.Baseline":
    return bt.Baseline(
        main_sha=main_sha, test_rel=test_rel, state="green", red=[],
        n_tests=1, duration_s=0.4, ts="2026-08-18T00:00:00+00:00", surface=surface,
    )


class TestVerificationContract:
    """extract_verification_contract now runs baseline_suite_for_item (spec
    verification-contract-main-baseline-v0) — these tests monkeypatch that
    single seam wholesale to stay hermetic (no real git/subprocess calls)
    while exercising the surrounding extraction logic. Coverage of
    baseline_suite_for_item's own internals lives in TestBaselineSuiteForItem
    below."""

    def test_none_when_no_repo(self):
        item = {"touched_surfaces": ["a.py"]}
        assert bt.extract_verification_contract(item) is None

    def test_none_when_no_touched_surfaces(self):
        item = {"repo": "myrepo", "touched_surfaces": []}
        assert bt.extract_verification_contract(item) is None

    def test_none_when_no_existing_test_located(self, tmp_path, monkeypatch):
        # No test anywhere under root -> baseline_suite_for_item's own
        # locate scan misses (reason "not_located") before any git call is
        # attempted — no need to monkeypatch baseline_suite_for_item itself.
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        item = {"repo": "myrepo", "touched_surfaces": ["pkg/mod.py"]}
        assert bt.extract_verification_contract(item) is None
        assert item["verification_baseline"]["reason"] == "not_located"

    def test_found_via_top_level_tests_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_mod.py").write_text("def test_x(): pass\n")
        monkeypatch.setattr(
            bt, "baseline_suite_for_item",
            lambda item, deadline_monotonic=None: _green_baseline("tests/test_mod.py", "pkg/mod.py"),
        )
        item = {"repo": "myrepo", "touched_surfaces": ["pkg/mod.py"]}
        contract = bt.extract_verification_contract(item)
        assert contract is not None
        assert "tests/test_mod.py" in contract
        assert "pkg/mod.py" in contract

    def test_found_via_colocated_test(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "test_mod.py").write_text("def test_x(): pass\n")
        monkeypatch.setattr(
            bt, "baseline_suite_for_item",
            lambda item, deadline_monotonic=None: _green_baseline("pkg/test_mod.py", "pkg/mod.py"),
        )
        item = {"repo": "myrepo", "touched_surfaces": ["pkg/mod.py"]}
        contract = bt.extract_verification_contract(item)
        assert contract is not None
        assert "pkg/test_mod.py" in contract

    def test_verify_behavioral_invariance_holds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_mod.py").write_text("def test_x(): pass\n")
        monkeypatch.setattr(
            bt, "baseline_suite_for_item",
            lambda item, deadline_monotonic=None: _green_baseline("tests/test_mod.py", "pkg/mod.py"),
        )
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


# ---------------------------------------------------------------------------
# baseline_suite_for_item (verification-contract-main-baseline-v0)
# ---------------------------------------------------------------------------

def _item_with_locatable_test(tmp_path, monkeypatch, surface="pkg/mod.py", test_rel="tests/test_mod.py"):
    """Set up a hermetic -working tree with a real, locatable test file for
    `surface`, and point bt._repo_root at it. Returns the item dict."""
    monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_mod.py").write_text("def test_x(): pass\n")
    return {"repo": "myrepo", "touched_surfaces": [surface]}


class TestBaselineSuiteForItem:
    def test_no_repo_or_surfaces(self):
        baseline = bt.baseline_suite_for_item({"touched_surfaces": ["a.py"]})
        assert baseline.state == "unverified"
        assert baseline.reason == "no_repo_or_surfaces"

    def test_not_located(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path)
        item = {"repo": "myrepo", "touched_surfaces": ["pkg/mod.py"]}
        baseline = bt.baseline_suite_for_item(item)
        assert baseline.state == "unverified"
        assert baseline.reason == "not_located"

    def test_fetch_failed(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        monkeypatch.setattr(bt, "_resolve_main_sha", lambda root: None)
        baseline = bt.baseline_suite_for_item(item)
        assert baseline.state == "unverified"
        assert baseline.reason == "fetch_failed"
        assert baseline.test_rel == "tests/test_mod.py"

    def test_worktree_failed(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        monkeypatch.setattr(bt, "_resolve_main_sha", lambda root: "abc1234def5678")
        monkeypatch.setattr(bt, "_ensure_baseline_worktree", lambda repo, root, sha: None)
        baseline = bt.baseline_suite_for_item(item)
        assert baseline.state == "unverified"
        assert baseline.reason == "worktree_failed"
        assert baseline.main_sha == "abc1234def5678"

    def test_not_on_main(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        empty_worktree = tmp_path / "_empty_worktree"
        empty_worktree.mkdir()
        monkeypatch.setattr(bt, "_resolve_main_sha", lambda root: "abc1234def5678")
        monkeypatch.setattr(bt, "_ensure_baseline_worktree", lambda repo, root, sha: empty_worktree)
        baseline = bt.baseline_suite_for_item(item)
        assert baseline.state == "unverified"
        assert baseline.reason == "not_on_main"

    def _with_worktree(self, tmp_path, monkeypatch, item):
        worktree = tmp_path / "_worktree"
        (worktree / "tests").mkdir(parents=True)
        (worktree / "tests" / "test_mod.py").write_text("def test_x(): pass\n")
        monkeypatch.setattr(bt, "_resolve_main_sha", lambda root: "abc1234def5678")
        monkeypatch.setattr(bt, "_ensure_baseline_worktree", lambda repo, root, sha: worktree)
        return worktree

    def test_timeout(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        self._with_worktree(tmp_path, monkeypatch, item)
        monkeypatch.setattr(
            bt, "_run_baseline_pytest", lambda wt, rel, timeout_s: ("timeout", "", 180.0),
        )
        baseline = bt.baseline_suite_for_item(item)
        assert baseline.state == "unverified"
        assert baseline.reason == "timeout"
        assert baseline.duration_s == 180.0

    def test_collection_error(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        self._with_worktree(tmp_path, monkeypatch, item)
        monkeypatch.setattr(
            bt, "_run_baseline_pytest",
            lambda wt, rel, timeout_s: ("collection-error", "ImportError\n", 0.2),
        )
        baseline = bt.baseline_suite_for_item(item)
        assert baseline.state == "unverified"
        assert baseline.reason == "collection_error"

    def test_budget_exhausted_no_io_attempted(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)

        def _boom(*a, **kw):
            raise AssertionError("no I/O should be attempted once the budget is exhausted")

        monkeypatch.setattr(bt, "_resolve_main_sha", _boom)
        monkeypatch.setattr(bt, "_ensure_baseline_worktree", _boom)
        monkeypatch.setattr(bt, "_run_baseline_pytest", _boom)
        exhausted_deadline = time.monotonic() - 1
        baseline = bt.baseline_suite_for_item(item, deadline_monotonic=exhausted_deadline)
        assert baseline.state == "unverified"
        assert baseline.reason == "budget_exhausted"

    def test_green_suite(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        self._with_worktree(tmp_path, monkeypatch, item)
        monkeypatch.setattr(
            bt, "_run_baseline_pytest",
            lambda wt, rel, timeout_s: ("ran", "50 passed in 0.40s\n", 0.4),
        )
        baseline = bt.baseline_suite_for_item(item)
        assert baseline.state == "green"
        assert baseline.red == []
        assert baseline.n_tests == 50
        assert baseline.main_sha == "abc1234def5678"
        assert baseline.surface == "pkg/mod.py"

    def test_annotated_suite_full_red_list_in_record(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        self._with_worktree(tmp_path, monkeypatch, item)
        # A file-level run with more than 5 reds — the record must carry
        # the FULL list even though the rendered contract string caps at 5.
        red_ids = [f"tests/test_mod.py::test_{i}" for i in range(7)]
        output = "".join(f"FAILED {rid} - AssertionError\n" for rid in red_ids)
        output += "43 passed, 7 failed in 1.10s\n"
        monkeypatch.setattr(
            bt, "_run_baseline_pytest", lambda wt, rel, timeout_s: ("ran", output, 1.1),
        )
        baseline = bt.baseline_suite_for_item(item)
        assert baseline.state == "annotated"
        assert baseline.red == red_ids
        assert baseline.n_tests == 50


# ---------------------------------------------------------------------------
# extract_verification_contract — three-state render (green / annotated /
# unverified), verification-contract-main-baseline-v0
# ---------------------------------------------------------------------------

class TestExtractVerificationContractThreeStates:
    def test_green_contract_string(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        monkeypatch.setattr(
            bt, "baseline_suite_for_item",
            lambda item, deadline_monotonic=None: bt.Baseline(
                main_sha="abc1234def5678", test_rel="tests/test_mod.py", state="green",
                red=[], n_tests=50, duration_s=1.9, ts="2026-08-18T00:00:00+00:00",
                surface="pkg/mod.py",
            ),
        )
        contract = bt.extract_verification_contract(item)
        assert contract is not None
        assert "main@abc1234" in contract
        assert "suite green at baseline" in contract
        assert "50 tests in 1.9s" in contract
        assert "2026-08-18T00:00:00+00:00" in contract

    def test_annotated_contract_string_scoped_disambiguation(self, tmp_path, monkeypatch):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        red_ids = [f"tests/test_mod.py::test_{i}" for i in range(7)]
        monkeypatch.setattr(
            bt, "baseline_suite_for_item",
            lambda item, deadline_monotonic=None: bt.Baseline(
                main_sha="abc1234def5678", test_rel="tests/test_mod.py", state="annotated",
                red=red_ids, n_tests=50, duration_s=1.1, ts="2026-08-18T00:00:00+00:00",
                surface="pkg/mod.py",
            ),
        )
        contract = bt.extract_verification_contract(item)
        assert contract is not None
        assert "main@abc1234" in contract
        assert "7 pre-existing red in this file" in contract
        # Contract string caps the preview at 5 ids...
        for rid in red_ids[:5]:
            assert rid in contract
        assert red_ids[5] not in contract
        assert red_ids[6] not in contract
        assert "full list in route record" in contract
        assert "scoped to this FILE-LEVEL fingerprint, not to any single test" in contract
        # ...but the record itself carries the FULL list.
        assert item["verification_baseline"]["red"] == red_ids

    @pytest.mark.parametrize("reason", [
        "fetch_failed", "not_on_main", "timeout", "collection_error", "budget_exhausted",
    ])
    def test_unverified_reasons_yield_none_with_categorized_reason(self, tmp_path, monkeypatch, reason):
        item = _item_with_locatable_test(tmp_path, monkeypatch)
        monkeypatch.setattr(
            bt, "baseline_suite_for_item",
            lambda item, deadline_monotonic=None: bt.Baseline(
                main_sha="abc1234def5678", test_rel="tests/test_mod.py", state="unverified",
                red=[], n_tests=0, duration_s=0.0, ts="2026-08-18T00:00:00+00:00",
                reason=reason, surface="pkg/mod.py",
            ),
        )
        contract = bt.extract_verification_contract(item)
        assert contract is None
        assert item["verification_baseline"]["reason"] == reason


# ---------------------------------------------------------------------------
# verify_behavioral_invariance — tier-3 sha-compare fast path + re-run
# ---------------------------------------------------------------------------

class TestTier3ShaCompare:
    def _bound_item(self, main_sha="abc1234def5678"):
        return {
            "repo": "myrepo",
            "touched_surfaces": ["pkg/mod.py"],
            "verification_contract": "existing test `tests/test_mod.py` pins behavior "
                                      "for `pkg/mod.py` and passes unchanged (main@abc1234: "
                                      "suite green at baseline, 50 tests in 0.4s, ts)",
            "verification_baseline": {
                "main_sha": main_sha, "test_rel": "tests/test_mod.py", "state": "green",
                "red": [], "n_tests": 50, "duration_s": 0.4, "ts": "ts",
                "reason": None, "surface": "pkg/mod.py",
            },
        }

    def test_same_sha_holds_without_rerunning_suite(self, monkeypatch):
        item = self._bound_item()
        monkeypatch.setattr(bt, "_repo_root", lambda repo: Path("/nonexistent"))
        monkeypatch.setattr(bt, "_resolve_main_sha", lambda root: "abc1234def5678")

        def _boom(*a, **kw):
            raise AssertionError("the suite must not be re-run when main sha is unchanged")

        monkeypatch.setattr(bt, "baseline_suite_for_item", _boom)
        holds, contract = bt.verify_behavioral_invariance(item)
        assert holds is True
        assert contract == item["verification_contract"]

    def test_advanced_sha_reruns(self, monkeypatch):
        item = self._bound_item(main_sha="abc1234def5678")
        monkeypatch.setattr(bt, "_repo_root", lambda repo: Path("/nonexistent"))
        # Tier 3's own sha-compare check reports the NEW sha — advanced.
        monkeypatch.setattr(bt, "_resolve_main_sha", lambda root: "9999999def5678")

        calls = []

        def _rerun(item, deadline_monotonic=None):
            calls.append(1)
            return bt.Baseline(
                main_sha="9999999def5678", test_rel="tests/test_mod.py", state="green",
                red=[], n_tests=50, duration_s=0.4, ts="ts2", surface="pkg/mod.py",
            )

        monkeypatch.setattr(bt, "baseline_suite_for_item", _rerun)
        holds, contract = bt.verify_behavioral_invariance(item)
        assert len(calls) == 1
        assert holds is True
        assert "9999999" in contract

    def test_advanced_sha_timeout_holds_false(self, monkeypatch):
        item = self._bound_item(main_sha="abc1234def5678")
        monkeypatch.setattr(bt, "_repo_root", lambda repo: Path("/nonexistent"))
        monkeypatch.setattr(bt, "_resolve_main_sha", lambda root: "9999999def5678")
        monkeypatch.setattr(
            bt, "baseline_suite_for_item",
            lambda item, deadline_monotonic=None: bt.Baseline(
                main_sha="9999999def5678", test_rel="tests/test_mod.py", state="unverified",
                red=[], n_tests=0, duration_s=180.0, ts="ts2", reason="timeout",
                surface="pkg/mod.py",
            ),
        )
        holds, contract = bt.verify_behavioral_invariance(item)
        assert holds is False
        assert contract is None


# ---------------------------------------------------------------------------
# Worktree reuse (gate finding: worktree creation is the slow part)
# ---------------------------------------------------------------------------

class TestWorktreeReuse:
    def test_second_call_at_same_sha_does_not_recreate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_BASELINE_WORKTREE_ROOT", tmp_path / "baselines")
        calls = []

        def _fake_add(repo_root, wt_path, sha):
            calls.append(sha)
            wt_path.mkdir(parents=True, exist_ok=True)
            return True

        monkeypatch.setattr(bt, "_git_worktree_add", _fake_add)
        root = tmp_path / "_repo"
        root.mkdir()

        wt1 = bt._ensure_baseline_worktree("myrepo", root, "abc1234def5678")
        wt2 = bt._ensure_baseline_worktree("myrepo", root, "abc1234def5678")

        assert wt1 == wt2
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Run-start prune pass (gate hardening fold: 87%-full `/` at spec time)
# ---------------------------------------------------------------------------

class TestPruneStaleBaselineWorktrees:
    def test_stale_worktree_removed_current_untouched(self, tmp_path, monkeypatch):
        root = tmp_path / "baselines"
        root.mkdir()
        stale = root / "myrepo-0000000"
        current = root / "myrepo-abc1234"
        stale.mkdir()
        current.mkdir()

        monkeypatch.setattr(bt, "_BASELINE_WORKTREE_ROOT", root)
        monkeypatch.setattr(bt, "_repo_root", lambda repo: tmp_path / "_repo")
        monkeypatch.setattr(bt, "_resolve_main_sha", lambda repo_root: "abc1234def5678")

        removed_paths = []
        monkeypatch.setattr(
            bt, "_remove_baseline_worktree",
            lambda repo, wt_path: removed_paths.append(wt_path.name),
        )

        removed = bt.prune_stale_baseline_worktrees()

        assert removed == ["myrepo-0000000"]
        assert removed_paths == ["myrepo-0000000"]
        assert current.exists()

    def test_no_worktree_root_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bt, "_BASELINE_WORKTREE_ROOT", tmp_path / "does-not-exist")
        assert bt.prune_stale_baseline_worktrees() == []


# ---------------------------------------------------------------------------
# Parser + invocation parity pin (gate hardening fold: a future flag/parser
# drift must fail loudly, not silently turn a real red into a false green)
# ---------------------------------------------------------------------------

class TestParserInvocationParityPin:
    def test_run_baseline_pytest_invocation_shape_is_pinned(self, tmp_path, monkeypatch):
        """Pins the exact command the baseline runner emits — the format
        batched_fixer_eval.parse_pytest_failures must stay compatible with."""
        captured = {}

        class _FakeResult:
            returncode = 1
            stdout = "FAILED tests/test_mod.py::test_x - AssertionError\n1 failed in 0.10s\n"
            stderr = ""

        def _fake_run(cmd, cwd, capture_output, text, timeout, env):
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            captured["env"] = env
            return _FakeResult()

        monkeypatch.setattr(bt.subprocess, "run", _fake_run)
        outcome, output, duration_s = bt._run_baseline_pytest(tmp_path, "tests/test_mod.py", 180)

        assert outcome == "ran"
        assert captured["cmd"] == [
            "python3", "-m", "pytest", "tests/test_mod.py",
            "-q", "--no-header", "-p", "no:cacheprovider",
        ]
        assert captured["env"]["PYTHONUSERBASE"] == "/home/user/.local"

    def test_parse_pytest_failures_matches_baseline_output_format(self):
        """batched_fixer_eval.parse_pytest_failures must extract exactly the
        failing node ids from the short-summary `FAILED <node id>` lines the
        baseline invocation emits (-q --no-header -p no:cacheprovider) —
        verified compatible at spec time (batched_fixer_eval.py:461-474);
        this test pins that compatibility as a regression, not a one-time
        reading."""
        from lapis_pm.batched_fixer_eval import parse_pytest_failures

        sample_output = (
            "FAILED tests/test_mod.py::test_a - AssertionError: boom\n"
            "FAILED tests/test_mod.py::TestX::test_b - ValueError\n"
            "48 passed, 2 failed in 0.87s\n"
        )
        failures = parse_pytest_failures(sample_output)
        assert failures == [
            "tests/test_mod.py::test_a",
            "tests/test_mod.py::TestX::test_b",
        ]
