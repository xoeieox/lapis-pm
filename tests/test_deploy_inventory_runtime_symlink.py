"""Tests for lapis_pm.deploy_inventory's runtime_symlink_into_dev_tree finding
(lapis-pm-deploy-inventory-runtime-symlink-into-dev-tree-v0).

No live-host dependency for the synthetic fixtures below — every test builds
its own fake `/srv/git`(-shaped) tree under `tmp_path` and patches
`_DEV_TREE_ROOT` / `_CLONE_ROOT_PREFIXES` to point at it. The one exception
(`TestLiveHostRegression`) deliberately runs against the real host paths —
see its docstring.
"""

from unittest.mock import patch

import pytest

from lapis_pm import deploy_inventory as di


# ---------------------------------------------------------------------------
# D1/D2: a symlink resolving into an unmapped `-working` tree is HIGH
# ---------------------------------------------------------------------------

class TestBasicDetection:
    def test_symlink_into_unmapped_working_tree_yields_one_high_finding(self, tmp_path):
        dev_root = tmp_path / "git"
        deploy_clone = dev_root / "myrepo"
        working_tree = dev_root / "myrepo-working" / "config"
        working_tree.mkdir(parents=True)
        deploy_clone.mkdir(parents=True)

        target = working_tree / "topology.yaml"
        target.write_text("topology: real\n")
        symlink = deploy_clone / "topology.yaml"
        symlink.symlink_to(target)

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)), \
             patch.object(di, "_resolve_git_root", return_value=str(deploy_clone)):
            findings = di.find_runtime_symlink_findings({})

        assert str(deploy_clone) in findings
        clone_findings = findings[str(deploy_clone)]
        assert len(clone_findings) == 1
        f = clone_findings[0]
        assert f.kind == "runtime_symlink_into_dev_tree"
        assert f.severity == "HIGH"
        assert str(symlink) in f.detail
        assert str(target) in f.detail
        assert str(dev_root / "myrepo-working") in f.detail

    def test_symlink_into_mapped_working_tree_yields_no_finding(self, tmp_path):
        dev_root = tmp_path / "git"
        deploy_clone = dev_root / "myrepo"
        working_tree = dev_root / "myrepo-working" / "config"
        working_tree.mkdir(parents=True)
        deploy_clone.mkdir(parents=True)

        target = working_tree / "topology.yaml"
        target.write_text("topology: real\n")
        symlink = deploy_clone / "topology.yaml"
        symlink.symlink_to(target)

        # The tree IS in _POST_LAND_PULL's flattened path set this time.
        post_land_pull = {"myrepo": [str(dev_root / "myrepo-working")]}

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)), \
             patch.object(di, "_resolve_git_root", return_value=str(deploy_clone)):
            findings = di.find_runtime_symlink_findings(post_land_pull)

        assert findings == {}

    def test_symlink_outside_dev_tree_entirely_yields_no_finding(self, tmp_path):
        dev_root = tmp_path / "git"
        deploy_clone = dev_root / "myrepo"
        deploy_clone.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()

        target = outside / "config.yaml"
        target.write_text("x")
        symlink = deploy_clone / "config.yaml"
        symlink.symlink_to(target)

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)), \
             patch.object(di, "_resolve_git_root", return_value=str(deploy_clone)):
            findings = di.find_runtime_symlink_findings({})

        assert findings == {}

    def test_symlink_inside_its_own_working_tree_is_not_flagged(self, tmp_path):
        """A `-working` tree's own internal tooling symlinks (venv lib64,
        node_modules/.bin shims) are not a runtime-path-into-a-foreign-tree
        hazard — excluding them is what keeps DoD item 8 (zero findings on
        the real, current host) true; the real host has ~25 of these.
        """
        dev_root = tmp_path / "git"
        working_tree = dev_root / "myrepo-working"
        venv_lib = working_tree / ".venv" / "lib"
        venv_lib.mkdir(parents=True)
        symlink = working_tree / ".venv" / "lib64"
        symlink.symlink_to(venv_lib)

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)):
            findings = di.find_runtime_symlink_findings({})

        assert findings == {}


# ---------------------------------------------------------------------------
# D4: kind-scoped ack, and the collision guard vs. a bare legacy ack
# ---------------------------------------------------------------------------

class TestKindScopedAck:
    def test_kind_scoped_ack_downgrades_to_info(self, tmp_path):
        dev_root = tmp_path / "git"
        deploy_clone = dev_root / "myrepo"
        working_tree = dev_root / "myrepo-working" / "config"
        working_tree.mkdir(parents=True)
        deploy_clone.mkdir(parents=True)

        target = working_tree / "topology.yaml"
        target.write_text("x")
        symlink = deploy_clone / "topology.yaml"
        symlink.symlink_to(target)

        (deploy_clone / di._ACK_MARKER).write_text("runtime_symlink_into_dev_tree\n")

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)), \
             patch.object(di, "_resolve_git_root", return_value=str(deploy_clone)):
            findings = di.find_runtime_symlink_findings({})

        assert findings[str(deploy_clone)][0].severity == "INFO"

    def test_bare_legacy_ack_leaves_symlink_finding_high_but_still_acks_unmapped_live_clone(self, tmp_path):
        """The regression D4 exists to prevent: a bare `.deploy-unmapped-ack`
        (no content, dropped for an unrelated unmapped_live_clone finding)
        must NOT silence a HIGH runtime_symlink_into_dev_tree finding on the
        same clone — while it must still silence the legacy finding it was
        actually meant for.
        """
        dev_root = tmp_path / "git"
        deploy_clone = dev_root / "myrepo"
        working_tree = dev_root / "myrepo-working" / "config"
        working_tree.mkdir(parents=True)
        deploy_clone.mkdir(parents=True)

        target = working_tree / "topology.yaml"
        target.write_text("x")
        symlink = deploy_clone / "topology.yaml"
        symlink.symlink_to(target)

        # Bare legacy marker — empty file, no kind scoping.
        (deploy_clone / di._ACK_MARKER).write_text("")

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)), \
             patch.object(di, "_resolve_git_root", return_value=str(deploy_clone)):
            symlink_findings = di.find_runtime_symlink_findings({})

        assert symlink_findings[str(deploy_clone)][0].severity == "HIGH"

        # The SAME bare marker still does its original job on the legacy kind.
        inventory = {
            str(deploy_clone): di.CloneEntry(
                path=str(deploy_clone), remote=None,
                backing_units=[di.BackingUnit("x.service", "WorkingDirectory", "system", True, "simple")],
            ),
        }
        legacy_findings = di.reconcile_inventory(inventory, {}, {}, {})
        assert legacy_findings[str(deploy_clone)][0].kind == "unmapped_live_clone"
        assert legacy_findings[str(deploy_clone)][0].severity == "INFO"

    def test_ack_kinds_empty_when_no_marker(self, tmp_path):
        assert di._ack_kinds(str(tmp_path / "no-such-clone")) == frozenset()

    def test_ack_kinds_legacy_only_for_empty_marker(self, tmp_path):
        clone = tmp_path / "clone"
        clone.mkdir()
        (clone / di._ACK_MARKER).write_text("")
        assert di._ack_kinds(str(clone)) == di._LEGACY_ACK_KINDS
        assert "runtime_symlink_into_dev_tree" not in di._ack_kinds(str(clone))


# ---------------------------------------------------------------------------
# D5 / fail-open: broken symlinks, resolution loops
# ---------------------------------------------------------------------------

class TestFailOpen:
    def test_broken_symlink_yields_no_finding_and_does_not_raise(self, tmp_path):
        dev_root = tmp_path / "git"
        deploy_clone = dev_root / "myrepo"
        deploy_clone.mkdir(parents=True)

        symlink = deploy_clone / "broken.yaml"
        symlink.symlink_to(deploy_clone / "nonexistent-target.yaml")

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)), \
             patch.object(di, "_resolve_git_root", return_value=str(deploy_clone)):
            findings = di.find_runtime_symlink_findings({})  # must not raise

        assert findings == {}

    def test_resolution_loop_yields_no_finding_and_does_not_raise(self, tmp_path):
        dev_root = tmp_path / "git"
        deploy_clone = dev_root / "myrepo"
        deploy_clone.mkdir(parents=True)

        a = deploy_clone / "a.yaml"
        b = deploy_clone / "b.yaml"
        a.symlink_to(b)
        b.symlink_to(a)

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)), \
             patch.object(di, "_resolve_git_root", return_value=str(deploy_clone)):
            findings = di.find_runtime_symlink_findings({})  # must not raise

        assert findings == {}

    def test_missing_root_yields_no_finding(self, tmp_path):
        with patch.object(di, "_DEV_TREE_ROOT", str(tmp_path / "git")), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(tmp_path / "nonexistent-root"),)):
            findings = di.find_runtime_symlink_findings({})
        assert findings == {}


# ---------------------------------------------------------------------------
# D1a: TRANSITIVE resolution — chained hops, scope stop, resolution loop
# ---------------------------------------------------------------------------

class TestTransitiveResolution:
    def test_chain_through_innocent_intermediate_is_caught_and_hops_recorded(self, tmp_path):
        dev_root = tmp_path / "git"
        deploy_clone = dev_root / "myrepo"
        intermediate_dir = dev_root / "some-other-clone"
        working_tree = dev_root / "myrepo-working" / "config"
        for d in (deploy_clone, intermediate_dir, working_tree):
            d.mkdir(parents=True)

        real_target = working_tree / "topology.yaml"
        real_target.write_text("x")

        b = intermediate_dir / "b.yaml"          # innocent intermediate hop
        b.symlink_to(real_target)
        a = deploy_clone / "a.yaml"               # the runtime-facing symlink
        a.symlink_to(b)

        # `b` is itself a real symlink under a walked root and — taken on its
        # own — also resolves into the same unmapped tree, so it legitimately
        # earns its own independent finding too (D1: "one finding per
        # symlink"). Give each starting point its own owning clone so the
        # two don't collapse into one list, and assert on `a`'s specifically.
        def fake_git_root(path_str):
            return {str(deploy_clone): str(deploy_clone), str(intermediate_dir): str(intermediate_dir)}.get(path_str)

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)), \
             patch.object(di, "_resolve_git_root", side_effect=fake_git_root):
            findings = di.find_runtime_symlink_findings({})

        assert str(deploy_clone) in findings
        clone_findings = [f for f in findings[str(deploy_clone)] if f.kind == "runtime_symlink_into_dev_tree"]
        assert len(clone_findings) == 1
        detail = clone_findings[0].detail
        # Every hop in the chain is legible in the detail string (D1a).
        assert str(a) in detail
        assert str(b) in detail
        assert str(real_target) in detail

    def test_chain_terminating_outside_dev_tree_yields_no_finding(self, tmp_path):
        dev_root = tmp_path / "git"
        deploy_clone = dev_root / "myrepo"
        deploy_clone.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()

        real_target = outside / "real.yaml"
        real_target.write_text("x")
        b = outside / "b.yaml"
        b.symlink_to(real_target)
        a = deploy_clone / "a.yaml"
        a.symlink_to(b)

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root),)), \
             patch.object(di, "_resolve_git_root", return_value=str(deploy_clone)):
            findings = di.find_runtime_symlink_findings({})

        assert findings == {}

    def test_resolve_symlink_chain_records_every_hop(self, tmp_path):
        """Direct unit coverage of the resolver itself — proves transitivity
        is a property of this function's own loop, not an accident of
        Path.resolve(), so a future one-hop regression (e.g. a refactor to
        bare os.readlink()) fails this assertion directly.
        """
        dev_root = tmp_path / "git"
        working_tree = dev_root / "myrepo-working"
        working_tree.mkdir(parents=True)
        real_target = working_tree / "x.yaml"
        real_target.write_text("x")
        b = dev_root / "b.yaml"
        b.symlink_to(real_target)
        a = dev_root / "a.yaml"
        a.symlink_to(b)

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)):
            result = di._resolve_symlink_chain(a)

        assert result is not None
        hops, terminus = result
        assert hops == [str(a), str(b), str(real_target)]
        assert terminus == str(real_target)


# ---------------------------------------------------------------------------
# D7: rendered on the operator board
# ---------------------------------------------------------------------------

class TestRenderedOnOperatorBoard:
    def test_render_deploy_status_names_the_finding_kind_on_owning_row(self):
        status = {
            "generated": "now",
            "clones": [
                {
                    "path": "/data/agents", "backing_units": [], "commits_behind": 0,
                    "branch": "main", "tracked_dirty": False, "untracked_present": False,
                    "mapped": True,
                    "findings": [
                        {"kind": "runtime_symlink_into_dev_tree", "severity": "HIGH", "detail": "x"},
                    ],
                },
                {
                    "path": "/srv/git/other-clone", "backing_units": [], "commits_behind": 0,
                    "branch": "main", "tracked_dirty": False, "untracked_present": False,
                    "mapped": True, "findings": [],
                },
            ],
        }
        rendered = di.render_deploy_status(status)
        lines = rendered.splitlines()
        agents_line = next(l for l in lines if l.startswith("agents "))
        assert "runtime_symlink_into_dev_tree" in agents_line
        other_line = next(l for l in lines if l.startswith("other-clone "))
        assert "runtime_symlink_into_dev_tree" not in other_line


# ---------------------------------------------------------------------------
# DoD item 7: the two 2026-08-13 instances, as fixtures
# ---------------------------------------------------------------------------

class TestAugust13Regression:
    def test_both_2026_08_13_instances_would_have_been_caught(self, tmp_path):
        """/data/agents/config/gw-topology.yaml and .../podcasts.yaml, both
        symlinking into /srv/git/conductor-working (absent from
        _POST_LAND_PULL, per pm_core.py's own comment that this is
        deliberate) — reconstructed as fixtures, pre-remediation shape.
        """
        dev_root = tmp_path / "git"
        agents_root = tmp_path / "agents"
        conductor_working_config = dev_root / "conductor-working" / "config"
        agents_config = agents_root / "config"
        conductor_working_config.mkdir(parents=True)
        agents_config.mkdir(parents=True)

        gw_target = conductor_working_config / "gw-topology.yaml"
        gw_target.write_text("topology: old\n")
        podcasts_target = conductor_working_config / "podcasts.yaml"
        podcasts_target.write_text("podcasts: old\n")

        (agents_config / "gw-topology.yaml").symlink_to(gw_target)
        (agents_config / "podcasts.yaml").symlink_to(podcasts_target)

        # _POST_LAND_PULL maps only the deploy clone, never -working (matches
        # pm_core.py's real map shape).
        post_land_pull = {"conductor": [str(dev_root / "conductor")]}

        with patch.object(di, "_DEV_TREE_ROOT", str(dev_root)), \
             patch.object(di, "_CLONE_ROOT_PREFIXES", (str(dev_root), str(agents_root))), \
             patch.object(di, "_resolve_git_root", return_value=str(agents_root)):
            findings = di.find_runtime_symlink_findings(post_land_pull)

        assert str(agents_root) in findings
        kinds = [f.kind for f in findings[str(agents_root)]]
        assert kinds.count("runtime_symlink_into_dev_tree") == 2
        assert all(f.severity == "HIGH" for f in findings[str(agents_root)])


# ---------------------------------------------------------------------------
# DoD item 8: live host produces zero findings (both instances remediated
# 2026-08-13). Runs against the REAL /srv/git + /data/agents — a vacuous
# pass in an environment without those paths, a genuine regression check on
# the actual host.
# ---------------------------------------------------------------------------

class TestLiveHostRegression:
    def test_live_host_has_zero_runtime_symlink_findings(self):
        from lapis_pm import pm_core

        findings_by_clone = di.find_runtime_symlink_findings(pm_core._POST_LAND_PULL)
        offenders = {
            clone: [f for f in fs if f.kind == "runtime_symlink_into_dev_tree"]
            for clone, fs in findings_by_clone.items()
        }
        offenders = {k: v for k, v in offenders.items() if v}
        assert offenders == {}, (
            f"runtime_symlink_into_dev_tree finding(s) on the live host — a third, "
            f"undiscovered instance: {offenders}"
        )
