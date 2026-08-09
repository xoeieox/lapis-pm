"""Unit tests for lapis_pm/spec_attestation.py (system-spec-drift-attestation-v0)."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from lapis_pm import spec_attestation as sa


def _write_spec(root: Path, slug: str, *, drift="green", code_paths=None, extra=None) -> Path:
    fm = {
        "system": slug,
        "status": "living",
        "last-verified": "2026-08-01",
        "verified-by": "test",
        "verified-against": "deadbeef",
        "drift": drift,
        "code-paths": code_paths or [],
        "related-mem": [],
    }
    if extra:
        fm.update(extra)
    text = "---\n" + yaml.safe_dump(fm, sort_keys=False) + "---\n\n# " + slug + "\n\nBody text.\n"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{slug}.md"
    path.write_text(text)
    return path


def _init_git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()
    return sha


def _write_repowise_state(path: Path, last_sync_commit: str) -> None:
    d = path / ".repowise"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({"last_sync_commit": last_sync_commit}))


def _write_knowledge_graph(path: Path, nodes, edges) -> None:
    d = path / ".repowise"
    d.mkdir(parents=True, exist_ok=True)
    (d / "knowledge-graph.json").write_text(json.dumps({"nodes": nodes, "edges": edges}))


# --------------------------------------------------------------------------
# classify_code_path / split_repo_relative
# --------------------------------------------------------------------------

def test_classify_repo_relative():
    assert sa.classify_code_path("gardener/gardener/inputs.py") == "repo_relative"


def test_classify_absolute_home():
    assert sa.classify_code_path("~/.claude/hooks/synapse-inject.py") == "absolute"


def test_classify_host_scoped_colon_prefix():
    assert sa.classify_code_path("gravitywell:/usr/local/bin/gw-serve") == "host_scoped"


def test_classify_host_scoped_systemd():
    assert sa.classify_code_path("systemd: weaver-watcher.service") == "host_scoped"


def test_split_repo_relative():
    assert sa.split_repo_relative("gardener/gardener/inputs.py") == ("gardener", "gardener/inputs.py")


# --------------------------------------------------------------------------
# frontmatter parsing
# --------------------------------------------------------------------------

def test_load_spec_roundtrip(tmp_path):
    p = _write_spec(tmp_path, "myspec", code_paths=["repo1/a.py"])
    spec = sa.load_spec(p)
    assert spec is not None
    assert spec.slug == "myspec"
    assert spec.drift == "green"
    assert spec.code_paths_for_repo("repo1") == ["a.py"]


def test_iter_spec_paths_excludes_underscore_files(tmp_path):
    _write_spec(tmp_path, "real-spec")
    (tmp_path / "_manifest.md").write_text("manifest")
    (tmp_path / "_format.md").write_text("format")
    paths = sa.iter_spec_paths(tmp_path)
    names = [p.name for p in paths]
    assert "real-spec.md" in names
    assert "_manifest.md" not in names
    assert "_format.md" not in names


# --------------------------------------------------------------------------
# DoD: a land touching a declared code-path degrades exactly the right spec row
# --------------------------------------------------------------------------

def test_named_match_degrades_drift(tmp_path):
    repo_clone = tmp_path / "clones" / "repo1"
    sha = _init_git_repo(repo_clone)
    _write_repowise_state(repo_clone, sha)
    _write_knowledge_graph(repo_clone, [], [])

    spec_path = _write_spec(tmp_path / "systems", "spec-a", code_paths=["repo1/pm_core.py"])
    spec = sa.load_spec(spec_path)
    clone = sa.CloneInfo(repo="repo1", path=repo_clone, kind="deploy", sha=sha, dirty_count=None)

    result = sa.evaluate_spec_for_land(spec, "repo1", ["pm_core.py"], clone)
    assert result is not None
    assert result.matched is True
    assert result.match_kind == "named"

    sa.apply_land_result(result, target_id="some-target-v0", trigger="post-merge-hook")
    reloaded = sa.load_spec(spec_path)
    assert reloaded.drift == "amber"
    assert reloaded.frontmatter["drift-last-target"] == "some-target-v0"


def test_unrelated_spec_not_touched(tmp_path):
    spec_path = _write_spec(tmp_path / "systems", "spec-b", code_paths=["other-repo/foo.py"])
    spec = sa.load_spec(spec_path)
    result = sa.evaluate_spec_for_land(spec, "repo1", ["pm_core.py"], clone=None)
    assert result is None  # not relevant — no code-paths in this repo


# --------------------------------------------------------------------------
# DoD: a land touching a file only *reachable* from a declared path also flags
# --------------------------------------------------------------------------

def test_reachable_but_not_named_match_flags(tmp_path):
    """The blast-radius case naive path-matching misses: the spec declares
    `declared.py`; the land changes `leaf_util.py`, which `declared.py`
    depends on (directly or transitively) but never names. `declared.py`
    is the graph's *source* node — walking forward from it (what it
    depends on) reaches `leaf_util.py`, so the spec must flag even though
    `leaf_util.py` is never named in `code-paths`.
    """
    repo_clone = tmp_path / "clones" / "repo2"
    sha = _init_git_repo(repo_clone)
    _write_repowise_state(repo_clone, sha)
    _write_knowledge_graph(
        repo_clone,
        nodes=[],
        edges=[
            {"source": "file:declared.py", "target": "file:helper.py",
             "type": "imports", "direction": "forward", "weight": 1.0},
            {"source": "file:helper.py", "target": "file:leaf_util.py",
             "type": "imports", "direction": "forward", "weight": 1.0},
        ],
    )
    spec_path = _write_spec(
        tmp_path / "systems", "spec-c", code_paths=["repo2/declared.py"]
    )
    spec = sa.load_spec(spec_path)
    clone = sa.CloneInfo(repo="repo2", path=repo_clone, kind="deploy", sha=sha, dirty_count=None)

    reachable = sa.reachable_from(repo_clone, ["declared.py"])
    assert reachable == {"helper.py", "leaf_util.py"}

    # Land changes leaf_util.py — never named by the spec, but transitively
    # depended on by declared.py.
    result = sa.evaluate_spec_for_land(spec, "repo2", ["leaf_util.py"], clone)
    assert result.matched is True
    assert result.match_kind == "reachable"
    assert result.matched_paths == ["leaf_util.py"]

    # A truly unrelated changed file must NOT flag.
    result2 = sa.evaluate_spec_for_land(spec, "repo2", ["totally_unrelated.py"], clone)
    assert result2.matched is False


def test_reachable_from_transitive(tmp_path):
    repo_clone = tmp_path / "clones" / "repo3"
    _write_knowledge_graph(
        repo_clone,
        nodes=[],
        edges=[
            {"source": "file:top.py", "target": "file:mid.py", "type": "imports"},
            {"source": "file:mid.py", "target": "file:leaf.py", "type": "imports"},
        ],
    )
    reach = sa.reachable_from(repo_clone, ["top.py"])
    assert reach == {"mid.py", "leaf.py"}


# --------------------------------------------------------------------------
# DoD: specs with host-scoped paths show partial-coverage
# --------------------------------------------------------------------------

def test_partial_coverage_flagged(tmp_path):
    repo_clone = tmp_path / "clones" / "repo4"
    sha = _init_git_repo(repo_clone)
    _write_repowise_state(repo_clone, sha)
    _write_knowledge_graph(repo_clone, [], [])

    manifest_dir = tmp_path / "systems"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "_manifest.md"
    manifest.write_text("| System | Spec |\n|---|---|\n| Spec D | [spec-d.md](spec-d.md) |\n")

    spec_path = _write_spec(
        manifest_dir, "spec-d",
        code_paths=["repo4/x.py", "~/.claude/hooks/thing.py", "systemd: unit.service"],
    )
    spec = sa.load_spec(spec_path)
    assert spec.has_non_repo_relative_paths() is True

    clone = sa.CloneInfo(repo="repo4", path=repo_clone, kind="deploy", sha=sha, dirty_count=None)
    result = sa.evaluate_spec_for_land(spec, "repo4", ["x.py"], clone)
    assert result.partial_coverage is True

    sa._update_manifest_row("spec-d.md", f"`{sa.PARTIAL_COVERAGE}`", manifest=manifest)
    assert sa.PARTIAL_COVERAGE in manifest.read_text()


# --------------------------------------------------------------------------
# DoD (invariant 6): a stale index reads STALE-GRAPH, never a clean row
# --------------------------------------------------------------------------

def test_stale_graph_renders_not_clean(tmp_path):
    repo_clone = tmp_path / "clones" / "repo5"
    sha = _init_git_repo(repo_clone)
    # index was built against a stale sha, unrelated to current HEAD
    _write_repowise_state(repo_clone, "0000000000000000000000000000000000000000")
    _write_knowledge_graph(repo_clone, [], [])

    spec_path = _write_spec(tmp_path / "systems", "spec-e", code_paths=["repo5/y.py"])
    spec = sa.load_spec(spec_path)
    clone = sa.CloneInfo(repo="repo5", path=repo_clone, kind="deploy", sha=sha, dirty_count=None)

    # No named match — the only way this could stay green is if the graph
    # reachability check silently reports "no drift".
    result = sa.evaluate_spec_for_land(spec, "repo5", ["unrelated-file.py"], clone)
    assert result.graph_status == sa.STALE_GRAPH
    assert result.matched is False

    sa.apply_land_result(result, target_id="t1", trigger="post-merge-hook")
    reloaded = sa.load_spec(spec_path)
    assert reloaded.frontmatter.get("graph-status") == sa.STALE_GRAPH
    # drift must NOT have been silently cleared/left green-and-trusted without
    # the STALE-GRAPH marker being visible somewhere machine-readable.
    manifest = tmp_path / "systems" / "_manifest.md"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text("| System | Spec |\n|---|---|\n| Spec E | [spec-e.md](spec-e.md) |\n")
        sa._update_manifest_row("spec-e.md", f"`{sa.STALE_GRAPH}`", manifest=manifest)
    assert sa.STALE_GRAPH in manifest.read_text()


def test_graph_status_no_index_when_repowise_absent(tmp_path):
    repo_clone = tmp_path / "clones" / "repo6"
    sha = _init_git_repo(repo_clone)
    # No .repowise/state.json at all.
    clone = sa.CloneInfo(repo="repo6", path=repo_clone, kind="deploy", sha=sha, dirty_count=None)
    status, index_sha, head_sha = sa.graph_status(clone)
    assert status == sa.NO_INDEX


def test_graph_status_none_clone_is_no_index():
    status, index_sha, head_sha = sa.graph_status(None)
    assert status == sa.NO_INDEX


def test_graph_status_ok_when_current(tmp_path):
    repo_clone = tmp_path / "clones" / "repo7"
    sha = _init_git_repo(repo_clone)
    _write_repowise_state(repo_clone, sha)
    clone = sa.CloneInfo(repo="repo7", path=repo_clone, kind="deploy", sha=sha, dirty_count=None)
    status, _, _ = sa.graph_status(clone)
    assert status == sa.STATUS_OK


# --------------------------------------------------------------------------
# drift degrade-only (invariant 1)
# --------------------------------------------------------------------------

def test_drift_never_upgrades(tmp_path):
    assert sa._degrade("green") == "amber"
    assert sa._degrade("amber") == "red"
    assert sa._degrade("red") == "red"


def test_apply_land_result_does_not_downgrade_already_red(tmp_path):
    repo_clone = tmp_path / "clones" / "repo8"
    sha = _init_git_repo(repo_clone)
    _write_repowise_state(repo_clone, sha)
    _write_knowledge_graph(repo_clone, [], [])

    spec_path = _write_spec(tmp_path / "systems", "spec-f", drift="red", code_paths=["repo8/z.py"])
    spec = sa.load_spec(spec_path)
    clone = sa.CloneInfo(repo="repo8", path=repo_clone, kind="deploy", sha=sha, dirty_count=None)
    result = sa.evaluate_spec_for_land(spec, "repo8", ["z.py"], clone)
    sa.apply_land_result(result, target_id="t2", trigger="post-merge-hook")
    reloaded = sa.load_spec(spec_path)
    assert reloaded.drift == "red"


# --------------------------------------------------------------------------
# resolve_repo_clone: deploy clone preferred, working-tree fallback records dirty count
# --------------------------------------------------------------------------

def test_resolve_repo_clone_prefers_deploy(tmp_path):
    _init_git_repo(tmp_path / "myrepo")
    _init_git_repo(tmp_path / "myrepo-working")
    info = sa.resolve_repo_clone("myrepo", clone_root=tmp_path)
    assert info.kind == "deploy"
    assert info.path == tmp_path / "myrepo"


def test_resolve_repo_clone_falls_back_to_working(tmp_path):
    _init_git_repo(tmp_path / "onlyworking-working")
    info = sa.resolve_repo_clone("onlyworking", clone_root=tmp_path)
    assert info.kind == "working"
    assert info.dirty_count is not None


def test_resolve_repo_clone_missing_returns_none(tmp_path):
    info = sa.resolve_repo_clone("nope", clone_root=tmp_path)
    assert info is None


# --------------------------------------------------------------------------
# run_post_land_attestation: best-effort, never raises
# --------------------------------------------------------------------------

def test_run_post_land_attestation_never_raises_on_garbage(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "load_all_specs", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    # Must not raise.
    sa.run_post_land_attestation("repo1", ["a.py"], target_id="t", trigger="x")


def test_run_post_land_attestation_noop_on_empty_changed_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(sa, "load_all_specs", lambda: calls.append("called") or [])
    sa.run_post_land_attestation("repo1", [], target_id="t", trigger="x")
    assert calls == []


# --------------------------------------------------------------------------
# Direction B: mem key culling (deterministic, testable gate)
# --------------------------------------------------------------------------

def test_cull_mem_keys_keeps_only_allowed_prefixes():
    keys = [
        "decision/foo-2026-08-01",
        "finding/bar",
        "correction/baz",
        "reference/qux",
        "router/lapis-pm/sessions/abc",
        "pm/dispatched/xyz",
        "pm/cursor/1",
        "pm/brief-gem/2",
        "arc-scratch-123",
    ]
    culled = sa.cull_mem_keys_for_decisions(keys)
    assert culled == [
        "decision/foo-2026-08-01", "finding/bar", "correction/baz", "reference/qux",
    ]


def test_repo_for_mem_key_uses_related_spec(tmp_path):
    _write_spec(tmp_path / "systems", "widget-system", code_paths=["widget-repo/core.py"])
    specs = sa.load_all_specs(tmp_path / "systems")
    repo = sa.repo_for_mem_key("decision/widget-system-thing-2026-08-01", specs=specs)
    assert repo == "widget-repo"


def test_repo_for_mem_key_no_match_returns_none(tmp_path):
    specs = sa.load_all_specs(tmp_path / "systems")
    assert sa.repo_for_mem_key("decision/totally-unrelated", specs=specs) is None
