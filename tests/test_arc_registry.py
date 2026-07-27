"""Tests for lapis_pm/arc_registry.py (arc-registry-contract-schema-v0).

Covers hermetic fixer DoD items 6, 6a(shared with contract), 7, 7a, 8, 9,
10, 12, 12a, 13.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import datetime, timedelta, timezone

import pytest

from lapis_pm import arc_registry, state_brief
from lapis_pm.contract import Contract, Observation, SubjectRef


class FakeMem:
    """Minimal in-memory double for agents_core.mem.MemoryStore's used surface."""

    def __init__(self, initial: dict[str, str] | None = None, *, raise_on_get: set[str] = frozenset()):
        self._store: dict[str, str] = dict(initial or {})
        self._raise_on_get = raise_on_get

    def get(self, key: str):
        if key in self._raise_on_get:
            raise OSError(f"simulated unreadable: {key}")
        if key in self._store:
            return {"key": key, "content": self._store[key]}
        return None

    def set(self, key: str, content: str, tags=None, source: str = "") -> bool:
        created = key not in self._store
        self._store[key] = content
        return created

    def list_by_prefix(self, prefix: str, limit: int = 50):
        items = [
            {"key": k, "content": v}
            for k, v in sorted(self._store.items())
            if k.startswith(prefix)
        ]
        return items[:limit]


def _write_arc_doc(arc_dir, slug: str, body: str) -> None:
    arc_dir.mkdir(parents=True, exist_ok=True)
    (arc_dir / f"{slug}.md").write_text(body, encoding="utf-8")


def _write_target(targets_dir, target_id: str) -> None:
    targets_dir.mkdir(parents=True, exist_ok=True)
    (targets_dir / f"{target_id}.yaml").write_text("id: " + target_id + "\n", encoding="utf-8")


class TestRegistryRowShape:
    """DoD 6: arc/<slug> rows validate against D2, including importance and derived_from."""

    def test_row_has_d2_shape(self, tmp_path):
        arc_dir = tmp_path / "lapis_state"
        targets_dir = tmp_path / "targets"
        _write_arc_doc(arc_dir, "my-arc", "# My Arc\n\nNEXT: ship the thing\n\nmy-target\n")
        _write_target(targets_dir, "my-target")
        mem = FakeMem()
        now = datetime(2026, 7, 27, tzinfo=timezone.utc)

        row = arc_registry._derive_row("my-arc", arc_dir / "my-arc.md", targets_dir, mem, now=now)

        assert row["slug"] == "my-arc"
        assert row["declared_next"] == "ship the thing"
        assert row["anchors"] == ["my-target"]
        assert row["last_touched"] is not None
        assert isinstance(row["importance"], int)
        assert row["contract"]["subject"] == {"kind": "arc", "id": "my-arc"}
        assert row["derived_from"] == ["arc-docs", "targets"]

    def test_zero_anchor_arc_importance_is_null(self, tmp_path):
        arc_dir = tmp_path / "lapis_state"
        targets_dir = tmp_path / "targets"
        _write_arc_doc(arc_dir, "lonely-arc", "# Lonely Arc\n\nNEXT: nothing pins this\n")
        mem = FakeMem()
        now = datetime(2026, 7, 27, tzinfo=timezone.utc)

        row = arc_registry._derive_row("lonely-arc", arc_dir / "lonely-arc.md", targets_dir, mem, now=now)

        assert row["anchors"] == []
        assert row["importance"] is None  # never 0, never a default
        assert row["derived_from"] == ["arc-docs"]


class TestImportanceFormula:
    """DoD 7: D2b formula asserted directly on fixtures at each recency
    boundary (<7d, <21d, older) and at the cap."""

    def test_zero_anchors_is_none(self):
        assert arc_registry._compute_importance({}, now=datetime.now(tz=timezone.utc)) is None

    def test_fresh_anchor_recency_bonus_2(self):
        now = datetime(2026, 7, 27, tzinfo=timezone.utc)
        anchor_ts = {"a": now - timedelta(days=3)}
        assert arc_registry._compute_importance(anchor_ts, now=now) == 1 + 2

    def test_medium_anchor_recency_bonus_1(self):
        now = datetime(2026, 7, 27, tzinfo=timezone.utc)
        anchor_ts = {"a": now - timedelta(days=10)}
        assert arc_registry._compute_importance(anchor_ts, now=now) == 1 + 1

    def test_old_anchor_recency_bonus_0(self):
        now = datetime(2026, 7, 27, tzinfo=timezone.utc)
        anchor_ts = {"a": now - timedelta(days=30)}
        assert arc_registry._compute_importance(anchor_ts, now=now) == 1 + 0

    def test_capped_at_ten(self):
        now = datetime(2026, 7, 27, tzinfo=timezone.utc)
        anchor_ts = {f"a{i}": now - timedelta(days=1) for i in range(20)}
        assert arc_registry._compute_importance(anchor_ts, now=now) == 10

    def test_no_derivable_timestamp_defaults_bonus_zero(self):
        now = datetime(2026, 7, 27, tzinfo=timezone.utc)
        anchor_ts = {"a": None, "b": None}
        assert arc_registry._compute_importance(anchor_ts, now=now) == 2  # 2 anchors + 0 bonus


class TestNoHeuristicAuthority:
    """DoD 7: a write attempted with empty derived_from raises and writes nothing."""

    def test_write_with_empty_derived_from_raises(self):
        mem = FakeMem()
        row = {"slug": "x", "derived_from": []}
        with pytest.raises(ValueError):
            arc_registry._write_row(mem, "x", row)
        assert "arc/x" not in mem._store


class TestDegenerateCorpusGuard:
    """DoD 7a: full-corpus populate whose importance has zero variance
    raises; the same fixture under --limit 3 does not. DoD 2a empty-corpus
    edge case: zero arcs is vacuously non-degenerate and succeeds."""

    def _uniform_fixture(self, tmp_path, n: int):
        arc_dir = tmp_path / "lapis_state"
        targets_dir = tmp_path / "targets"
        for i in range(n):
            _write_arc_doc(arc_dir, f"arc-{i}", f"# Arc {i}\n\nNEXT: do thing {i}\n")
        targets_dir.mkdir(parents=True, exist_ok=True)
        return arc_dir, targets_dir

    def test_full_corpus_zero_variance_raises(self, tmp_path):
        arc_dir, targets_dir = self._uniform_fixture(tmp_path, 5)
        mem = FakeMem()
        with pytest.raises(RuntimeError):
            arc_registry.populate(arc_dir=arc_dir, targets_dir=targets_dir, mem=mem,
                                   now=datetime.now(tz=timezone.utc))
        assert mem._store == {}

    def test_limit_sampling_skips_guard(self, tmp_path):
        arc_dir, targets_dir = self._uniform_fixture(tmp_path, 5)
        mem = FakeMem()
        result = arc_registry.populate(arc_dir=arc_dir, targets_dir=targets_dir, mem=mem,
                                        limit=3, now=datetime.now(tz=timezone.utc))
        assert result["written"] == 3

    def test_empty_corpus_does_not_raise(self, tmp_path):
        arc_dir = tmp_path / "lapis_state"
        targets_dir = tmp_path / "targets"
        arc_dir.mkdir(parents=True, exist_ok=True)
        mem = FakeMem()
        result = arc_registry.populate(arc_dir=arc_dir, targets_dir=targets_dir, mem=mem,
                                        now=datetime.now(tz=timezone.utc))
        assert result == {"written": 0, "skipped": 0, "rows": {}}

    def test_missing_arc_dir_does_not_raise(self, tmp_path):
        mem = FakeMem()
        result = arc_registry.populate(
            arc_dir=tmp_path / "does-not-exist", targets_dir=tmp_path / "targets", mem=mem,
        )
        assert result == {"written": 0, "skipped": 0, "rows": {}}


class TestPopulatorIdempotent:
    """DoD 8: two consecutive runs over identical fixture exhaust produce
    byte-identical rows apart from derived_ts."""

    def test_idempotent_across_two_runs(self, tmp_path):
        arc_dir = tmp_path / "lapis_state"
        targets_dir = tmp_path / "targets"
        _write_arc_doc(arc_dir, "my-arc", "# My Arc\n\nNEXT: review and merge PR #10 (lapis-pm)\n\nmy-target\n")
        _write_target(targets_dir, "my-target")
        merged_at = (datetime(2026, 7, 27, tzinfo=timezone.utc) - timedelta(days=2)).isoformat()
        mem = FakeMem({
            "pm/landed/my-target": json.dumps({"merged_at": merged_at, "pr": 10, "repo": "lapis-pm"}),
        })
        now = datetime(2026, 7, 27, tzinfo=timezone.utc)

        arc_registry.populate(arc_dir=arc_dir, targets_dir=targets_dir, mem=mem, now=now)
        first = json.loads(mem._store["arc/my-arc"])
        arc_registry.populate(arc_dir=arc_dir, targets_dir=targets_dir, mem=mem, now=now)
        second = json.loads(mem._store["arc/my-arc"])

        assert first.pop("derived_ts") is not None
        assert second.pop("derived_ts") is not None
        assert first == second


class TestPerSourceDegrade:
    """DoD 9: each source made unreadable in turn -> populator completes,
    affected fields marked unresolved, row is not written as if complete,
    no exception escapes."""

    def test_targets_source_unreadable_degrades_not_aborts(self, tmp_path):
        arc_dir = tmp_path / "lapis_state"
        _write_arc_doc(arc_dir, "my-arc", "# My Arc\n\nNEXT: ship it\n")
        # targets_dir points at a plain file -> .glob() raises NotADirectoryError (an OSError)
        not_a_dir = tmp_path / "targets-is-a-file"
        not_a_dir.write_text("not a directory", encoding="utf-8")
        mem = FakeMem()

        result = arc_registry.populate(arc_dir=arc_dir, targets_dir=not_a_dir, mem=mem,
                                        now=datetime.now(tz=timezone.utc))

        assert result["written"] == 1
        row = json.loads(mem._store["arc/my-arc"])
        assert row["anchors"] == []
        assert row["importance"] is None
        assert row["derived_from"] == ["arc-docs"]

    def test_landed_source_unreadable_degrades_not_aborts(self, tmp_path):
        arc_dir = tmp_path / "lapis_state"
        targets_dir = tmp_path / "targets"
        _write_arc_doc(arc_dir, "my-arc", "# My Arc\n\nNEXT: ship it\n\nmy-target\n")
        _write_target(targets_dir, "my-target")
        mem = FakeMem(raise_on_get={"pm/landed/my-target"})

        result = arc_registry.populate(arc_dir=arc_dir, targets_dir=targets_dir, mem=mem,
                                        now=datetime.now(tz=timezone.utc))

        assert result["written"] == 1
        row = json.loads(mem._store["arc/my-arc"])
        assert row["anchors"] == ["my-target"]
        assert "landed" not in row["derived_from"]


class TestSingleNextParser:
    """DoD 10: arc_registry.py imports _NEXT_LINE_RE / _NEXT_HEADER_RE /
    _BULLET_LINE_RE from state_brief.py rather than defining its own.
    Inspection-test method: ast, not string-grep -- an import cannot be
    confused with a redefinition this way."""

    _NAMES = {"_NEXT_LINE_RE", "_NEXT_HEADER_RE", "_BULLET_LINE_RE"}

    def test_imports_next_regexes_from_state_brief(self):
        source = inspect.getsource(arc_registry)
        tree = ast.parse(source)
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("state_brief"):
                imported_names.update(alias.name for alias in node.names)
        assert self._NAMES <= imported_names

    def test_no_local_redefinition_of_next_regexes(self):
        source = inspect.getsource(arc_registry)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and getattr(node, "col_offset", None) == 0:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assert target.id not in self._NAMES


class TestCompareSources:
    """DoD 12: compare_sources works -- zero disagreements when sources
    agree, names the divergent slug on a seeded divergence. Never raises,
    never mutates. DoD 12a: blind_spots fires only when prose is confident
    and the registry shows an unobserved delta for that slug."""

    def test_agreement_yields_no_disagreements_or_blind_spots(self, monkeypatch):
        def fake_read(start_ts, *, period="weekly", arc_source="prose"):
            return [{"slug": "a", "classification": "gone-quiet", "text": "a"}]

        monkeypatch.setattr(state_brief, "_read_arc_climate", fake_read)
        monkeypatch.setattr(arc_registry, "read_registry_rows", lambda *a, **kw: {})

        result = arc_registry.compare_sources(datetime.now(tz=timezone.utc))

        assert result["disagreements"] == []
        assert result["agreements"] == ["a"]
        assert result["only_in_prose"] == []
        assert result["only_in_registry"] == []
        assert result["blind_spots"] == []

    def test_seeded_divergence_names_the_slug_and_fires_blind_spot(self, monkeypatch):
        def fake_read(start_ts, *, period="weekly", arc_source="prose"):
            if arc_source == "prose":
                return [
                    {"slug": "a", "classification": "gone-quiet", "text": "a"},
                    {"slug": "b", "classification": "unresolvable", "text": "b"},
                ]
            return [{"slug": "a", "classification": "silently-advanced", "text": "a2"}]

        monkeypatch.setattr(state_brief, "_read_arc_climate", fake_read)

        blind_contract = Contract(
            subject=SubjectRef(kind="arc", id="b"),
            declaration="do X",
            elaboration={"next": "do X"},
            observed=(),  # never observed -> unobserved delta
        )
        monkeypatch.setattr(
            arc_registry, "read_registry_rows",
            lambda *a, **kw: {"b": {"slug": "b", "contract": blind_contract.to_dict()}},
        )

        result = arc_registry.compare_sources(datetime.now(tz=timezone.utc))

        assert {"slug": "a", "prose": "gone-quiet", "registry": "silently-advanced"} in result["disagreements"]
        assert result["only_in_prose"] == ["b"]
        assert result["blind_spots"] == ["b"]

    def test_never_raises_on_empty_sources(self, monkeypatch):
        monkeypatch.setattr(state_brief, "_read_arc_climate", lambda *a, **kw: [])
        monkeypatch.setattr(arc_registry, "read_registry_rows", lambda *a, **kw: {})
        result = arc_registry.compare_sources(datetime.now(tz=timezone.utc))
        assert result["agreements"] == []
        assert result["blind_spots"] == []


class TestNoNewMutableGlobalOrEnvVar:
    """DoD 13: no new module-level mutable global, no new env var."""

    def test_no_module_level_mutable_literal_assignment(self):
        source = inspect.getsource(arc_registry)
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                pytest.fail(f"module-level mutable literal assignment found: {ast.dump(node)}")

    def test_no_os_environ_reference(self):
        source = inspect.getsource(arc_registry)
        assert "os.environ" not in source


class TestCLI:
    """D3b: python -m lapis_pm.arc_registry populate [--prefix] [--dry-run] [--limit N]."""

    def test_populate_dry_run_via_cli(self, tmp_path, capsys):
        arc_dir = tmp_path / "lapis_state"
        targets_dir = tmp_path / "targets"
        _write_arc_doc(arc_dir, "my-arc", "# My Arc\n\nNEXT: ship it\n")

        def fake_populate(*, prefix, dry_run, limit):
            assert dry_run is True
            assert prefix == "scratch/"
            assert limit == 5
            return {"written": 0, "skipped": 0, "rows": {"my-arc": {}}}

        import unittest.mock as mock
        with mock.patch.object(arc_registry, "populate", fake_populate):
            rc = arc_registry.main(["populate", "--prefix", "scratch/", "--dry-run", "--limit", "5"])

        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["slugs"] == ["my-arc"]

    def test_default_prefix_is_arc(self):
        parser = arc_registry._build_arg_parser()
        args = parser.parse_args(["populate"])
        assert args.prefix == "arc/"
        assert args.dry_run is False
        assert args.limit is None
