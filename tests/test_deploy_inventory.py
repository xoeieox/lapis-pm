"""Fixture-based tests for lapis_pm.deploy_inventory (lapis-pm-deploy-inventory-reconciler-v0).

No live-host dependency: every systemctl/git call is intercepted through
deploy_inventory._run, and clone roots are real tmp_path directories with
_CLONE_ROOT_PREFIXES patched to accept them.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from lapis_pm import deploy_inventory as di


def _cp(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _write_git_config(clone_root: Path, remote_url: str | None = "git@example.com:org/repo.git"):
    git_dir = clone_root / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    if remote_url is not None:
        (git_dir / "config").write_text(
            f'[core]\n\trepositoryformatversion = 0\n[remote "origin"]\n\turl = {remote_url}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
        )
    else:
        (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n")


# ---------------------------------------------------------------------------
# 1. derive_deploy_inventory() — fixture unit text + fake clone tree
# ---------------------------------------------------------------------------

class TestDeriveDeployInventory:
    def test_resolves_clone_paths_and_backing_units(self, tmp_path):
        clone_a = tmp_path / "repoA-working"
        clone_a.mkdir()
        _write_git_config(clone_a)

        clone_b = tmp_path / "repoB-working"
        clone_b.mkdir()
        _write_git_config(clone_b, remote_url="git@example.com:org/repoB.git")

        list_units_system = (
            "foo.service      loaded active running Foo daemon (WorkingDirectory)\n"
            "bar.service      loaded active running Bar daemon (ExecStart)\n"
            "baz.service      loaded inactive dead   Baz oneshot (timer-driven)\n"
            "baz.timer        loaded active waiting  Baz timer\n"
        )
        list_units_user = ""

        unit_props = {
            ("system", "foo.service"): {
                "ActiveState": "active", "SubState": "running", "Type": "simple",
                "WorkingDirectory": str(clone_a), "ExecStart": "", "Environment": "",
            },
            ("system", "bar.service"): {
                "ActiveState": "active", "SubState": "running", "Type": "simple",
                "WorkingDirectory": "",
                "ExecStart": (
                    "{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 "
                    f"{clone_b}/bin/run.py ; ignore_errors=no ; start_time=[n/a] }}"
                ),
                "Environment": "",
            },
            ("system", "baz.service"): {
                "ActiveState": "inactive", "SubState": "dead", "Type": "oneshot",
                "WorkingDirectory": str(clone_a), "ExecStart": "", "Environment": "",
            },
            ("system", "baz.timer"): {
                "ActiveState": "active", "SubState": "waiting", "Type": "",
                "WorkingDirectory": "", "ExecStart": "", "Environment": "",
            },
        }

        git_toplevel = {
            str(clone_a): str(clone_a),
            str(clone_b): str(clone_b),
            str(clone_b / "bin" / "run.py"): str(clone_b),
        }

        def fake_run(cmd, timeout=None):
            if cmd[0] == "systemctl":
                if "list-units" in cmd:
                    out = list_units_user if "--user" in cmd else list_units_system
                    return _cp(0, out)
                if "show" in cmd:
                    unit = cmd[cmd.index("show") + 1]
                    scope = "user" if "--user" in cmd else "system"
                    props = unit_props.get((scope, unit))
                    if props is None:
                        return _cp(1, "")
                    text = "\n".join(f"{k}={v}" for k, v in props.items())
                    return _cp(0, text)
            if cmd[0] == "git" and "rev-parse" in cmd and "--show-toplevel" in cmd:
                path = cmd[cmd.index("-C") + 1]
                root = git_toplevel.get(path)
                if root is None:
                    return _cp(1, "")
                return _cp(0, root + "\n")
            return _cp(1, "")

        with patch.object(di, "_run", side_effect=fake_run):
            with patch.object(di, "_CLONE_ROOT_PREFIXES", (str(tmp_path),)):
                with patch.object(di, "_discover_editable_installs", return_value={}):
                    inventory = di.derive_deploy_inventory()

        assert str(clone_a) in inventory
        assert str(clone_b) in inventory

        a_units = {bu.unit: bu for bu in inventory[str(clone_a)].backing_units}
        assert "foo.service" in a_units
        assert a_units["foo.service"].field == "WorkingDirectory"
        assert a_units["foo.service"].scope == "system"
        assert a_units["foo.service"].live is True
        assert a_units["foo.service"].systemd_type == "simple"

        # oneshot unit is live because its companion timer is active/waiting.
        assert "baz.service" in a_units
        assert a_units["baz.service"].live is True
        assert a_units["baz.service"].systemd_type == "oneshot"

        b_units = {bu.unit: bu for bu in inventory[str(clone_b)].backing_units}
        assert "bar.service" in b_units
        assert b_units["bar.service"].field == "ExecStart"

        assert inventory[str(clone_a)].remote == "git@example.com:org/repo.git"

    def test_stopped_unit_is_not_live_and_creates_no_finding_path(self, tmp_path):
        clone = tmp_path / "repoC-working"
        clone.mkdir()
        _write_git_config(clone)

        list_units_system = "stopped.service  loaded inactive dead Stopped daemon\n"

        def fake_run(cmd, timeout=None):
            if cmd[0] == "systemctl":
                if "list-units" in cmd:
                    return _cp(0, "" if "--user" in cmd else list_units_system)
                if "show" in cmd:
                    return _cp(0, (
                        "ActiveState=inactive\nSubState=dead\nType=simple\n"
                        f"WorkingDirectory={clone}\nExecStart=\nEnvironment=\n"
                    ))
            if cmd[0] == "git" and "rev-parse" in cmd:
                return _cp(0, str(clone) + "\n")
            return _cp(1, "")

        with patch.object(di, "_run", side_effect=fake_run):
            with patch.object(di, "_CLONE_ROOT_PREFIXES", (str(tmp_path),)):
                with patch.object(di, "_discover_editable_installs", return_value={}):
                    inventory = di.derive_deploy_inventory()

        assert str(clone) in inventory
        bu = inventory[str(clone)].backing_units[0]
        assert bu.live is False

    def test_malformed_show_output_is_skipped_not_raised(self, tmp_path):
        def fake_run(cmd, timeout=None):
            if cmd[0] == "systemctl" and "list-units" in cmd:
                return _cp(0, "" if "--user" in cmd else "ghost.service loaded active running Ghost\n")
            if cmd[0] == "systemctl" and "show" in cmd:
                return _cp(1, "")  # simulates a truncated/failed probe
            return _cp(1, "")

        with patch.object(di, "_run", side_effect=fake_run):
            with patch.object(di, "_discover_editable_installs", return_value={}):
                inventory = di.derive_deploy_inventory()

        assert inventory == {}

    def test_editable_install_discovered_and_filtered_by_prefix(self, tmp_path):
        clone = tmp_path / "editable-repo"
        clone.mkdir()
        _write_git_config(clone)

        class FakeDist:
            def __init__(self, name, direct_url):
                self.metadata = {"Name": name}
                self._direct_url = direct_url

            def read_text(self, filename):
                if filename == "direct_url.json" and self._direct_url is not None:
                    return json.dumps(self._direct_url)
                return None

        in_scope = FakeDist("myeditable", {"url": f"file://{clone}", "dir_info": {"editable": True}})
        out_of_scope = FakeDist("otherlib", {"url": "file:///opt/otherlib", "dir_info": {"editable": True}})
        non_editable = FakeDist("regularpkg", None)

        with patch.object(di, "_CLONE_ROOT_PREFIXES", (str(tmp_path),)):
            result = di._discover_editable_installs([in_scope, out_of_scope, non_editable])

        assert result == {"myeditable": str(clone)}


# ---------------------------------------------------------------------------
# 2. Reconciler: unmapped_live_clone / pull_without_restart
# ---------------------------------------------------------------------------

def _entry(path, units, editable=False):
    return di.CloneEntry(path=path, remote=None, backing_units=list(units), is_editable_install=editable)


class TestReconcileInventory:
    def test_unmapped_live_clone_finding(self, tmp_path):
        clone = str(tmp_path / "unmapped-repo")
        inventory = {
            clone: _entry(clone, [di.BackingUnit("x.service", "WorkingDirectory", "system", True, "simple")]),
        }
        findings = di.reconcile_inventory(inventory, post_land_pull={}, post_land_restart={}, post_land_restart_user={})
        assert clone in findings
        kinds = [f.kind for f in findings[clone]]
        assert "unmapped_live_clone" in kinds
        sev = [f.severity for f in findings[clone] if f.kind == "unmapped_live_clone"][0]
        assert sev == "NORMAL"

    def test_mapped_clone_produces_no_unmapped_finding(self, tmp_path):
        clone = str(tmp_path / "mapped-repo")
        inventory = {
            clone: _entry(clone, [di.BackingUnit("x.service", "WorkingDirectory", "system", True, "simple")]),
        }
        post_land_pull = {"myrepo": [clone]}
        post_land_restart = {"myrepo": ("x.service",)}
        findings = di.reconcile_inventory(inventory, post_land_pull, post_land_restart, {})
        assert clone not in findings

    def test_pull_without_restart_finding_for_type_simple(self, tmp_path):
        clone = str(tmp_path / "pull-only-repo")
        inventory = {
            clone: _entry(clone, [di.BackingUnit("daemon.service", "WorkingDirectory", "system", True, "simple")]),
        }
        post_land_pull = {"myrepo": [clone]}
        findings = di.reconcile_inventory(inventory, post_land_pull, {}, {})
        assert clone in findings
        kinds = [f.kind for f in findings[clone]]
        assert kinds == ["pull_without_restart"]
        assert findings[clone][0].severity == "NORMAL"

    def test_oneshot_timer_unit_does_not_trigger_pull_without_restart(self, tmp_path):
        clone = str(tmp_path / "oneshot-repo")
        inventory = {
            clone: _entry(clone, [di.BackingUnit("nightly.service", "WorkingDirectory", "system", True, "oneshot")]),
        }
        post_land_pull = {"myrepo": [clone]}
        findings = di.reconcile_inventory(inventory, post_land_pull, {}, {})
        assert clone not in findings

    def test_ack_marker_downgrades_unmapped_to_info(self, tmp_path):
        clone = tmp_path / "acked-repo"
        clone.mkdir()
        (clone / di._ACK_MARKER).write_text("")
        inventory = {
            str(clone): _entry(str(clone), [di.BackingUnit("x.service", "WorkingDirectory", "system", True, "simple")]),
        }
        findings = di.reconcile_inventory(inventory, {}, {}, {})
        assert findings[str(clone)][0].severity == "INFO"

    def test_no_ack_marker_stays_normal(self, tmp_path):
        clone = tmp_path / "unacked-repo"
        clone.mkdir()
        inventory = {
            str(clone): _entry(str(clone), [di.BackingUnit("x.service", "WorkingDirectory", "system", True, "simple")]),
        }
        findings = di.reconcile_inventory(inventory, {}, {}, {})
        assert findings[str(clone)][0].severity == "NORMAL"


# ---------------------------------------------------------------------------
# 3. Currency assertions — three independent checks
# ---------------------------------------------------------------------------

class TestCheckCurrency:
    def _fake_run_factory(self, *, head="abc123", branch_rc=0, branch="main",
                           behind=0, status_lines=()):
        def fake_run(cmd, timeout=None):
            if cmd[0] == "git":
                if "fetch" in cmd:
                    return _cp(0, "")
                if "rev-parse" in cmd and "HEAD" in cmd:
                    return _cp(0, head + "\n")
                if "symbolic-ref" in cmd:
                    return _cp(branch_rc, (branch + "\n") if branch_rc == 0 else "")
                if "rev-list" in cmd:
                    return _cp(0, str(behind) + "\n")
                if "status" in cmd:
                    return _cp(0, "\n".join(status_lines) + ("\n" if status_lines else ""))
            return _cp(1, "")
        return fake_run

    def test_clean_current_on_main_emits_no_findings(self):
        fake_run = self._fake_run_factory()
        with patch.object(di, "_run", side_effect=fake_run):
            result = di.check_currency("/srv/git/some-repo")
        assert result.findings == []
        assert result.commits_behind == 0
        assert result.branch == "main"
        assert result.tracked_dirty is False

    def test_behind_count_emits_stale_behind_origin_high(self):
        fake_run = self._fake_run_factory(behind=3)
        with patch.object(di, "_run", side_effect=fake_run):
            result = di.check_currency("/srv/git/some-repo")
        kinds = {f.kind: f.severity for f in result.findings}
        assert kinds.get("stale_behind_origin") == "HIGH"

    def test_off_main_branch_emits_stray_branch_high(self):
        fake_run = self._fake_run_factory(branch="fix/some-thing")
        with patch.object(di, "_run", side_effect=fake_run):
            result = di.check_currency("/srv/git/some-repo")
        kinds = {f.kind: f.severity for f in result.findings}
        assert kinds.get("stray_branch") == "HIGH"
        assert result.branch == "fix/some-thing"

    def test_detached_head_treated_as_stray_branch(self):
        fake_run = self._fake_run_factory(branch_rc=1)
        with patch.object(di, "_run", side_effect=fake_run):
            result = di.check_currency("/srv/git/some-repo")
        assert result.branch == "DETACHED"
        kinds = {f.kind: f.severity for f in result.findings}
        assert kinds.get("stray_branch") == "HIGH"

    def test_tracked_dirty_emits_high(self):
        fake_run = self._fake_run_factory(status_lines=(" M some_file.py",))
        with patch.object(di, "_run", side_effect=fake_run):
            result = di.check_currency("/srv/git/some-repo")
        kinds = {f.kind: f.severity for f in result.findings}
        assert kinds.get("tracked_dirty_tree") == "HIGH"
        assert result.tracked_dirty is True

    def test_untracked_only_does_not_trip_clean_tree_failure(self):
        fake_run = self._fake_run_factory(status_lines=("?? some_runtime.db", "?? cache/"))
        with patch.object(di, "_run", side_effect=fake_run):
            result = di.check_currency("/srv/git/some-repo")
        assert result.tracked_dirty is False
        assert result.untracked_present is True
        assert not any(f.kind == "tracked_dirty_tree" for f in result.findings)

    def test_indeterminate_probe_never_emits_finding(self):
        def fake_run(cmd, timeout=None):
            return None  # simulates a hard probe failure (timeout/OSError) — truly indeterminate
        with patch.object(di, "_run", side_effect=fake_run):
            result = di.check_currency("/srv/git/some-repo")
        assert result.findings == []
        assert result.head is None
        assert result.branch is None
        assert result.commits_behind is None


# ---------------------------------------------------------------------------
# 4. Stale-lock escalation
# ---------------------------------------------------------------------------

class TestStaleLockEscalation:
    def test_old_lock_escalates_to_high(self, tmp_path):
        clone = "/srv/git/locked-repo"
        with patch.object(di, "_LOCK_DIR", tmp_path):
            lock_file = di._lock_path(clone)
            lock_file.write_text(json.dumps({
                "clone": clone,
                "reason": "genuine_divergence",
                "commits_behind": 5,
                "locked_at": "2026-07-01T00:00:00+00:00",
            }))
            finding, age = di.check_stale_lock(clone, now=__import__("datetime").datetime(
                2026, 7, 22, 0, 0, 0, tzinfo=__import__("datetime").timezone.utc))
        assert finding is not None
        assert finding.severity == "HIGH"
        assert finding.kind == "stale_deploy_pull_lock"
        assert age > 6 * 3600

    def test_fresh_lock_stays_normal(self, tmp_path):
        import datetime as _dt
        clone = "/srv/git/locked-repo-2"
        now = _dt.datetime(2026, 7, 22, 12, 0, 0, tzinfo=_dt.timezone.utc)
        with patch.object(di, "_LOCK_DIR", tmp_path):
            lock_file = di._lock_path(clone)
            lock_file.write_text(json.dumps({
                "clone": clone,
                "reason": "genuine_divergence",
                "commits_behind": 1,
                "locked_at": (now - _dt.timedelta(minutes=30)).isoformat(),
            }))
            finding, age = di.check_stale_lock(clone, now=now)
        assert finding is not None
        assert finding.severity == "NORMAL"
        assert age < 3600

    def test_no_lock_file_returns_none(self, tmp_path):
        with patch.object(di, "_LOCK_DIR", tmp_path):
            finding, age = di.check_stale_lock("/srv/git/never-locked")
        assert finding is None
        assert age is None


# ---------------------------------------------------------------------------
# Severity mapping (explicit, cross-cutting assertion)
# ---------------------------------------------------------------------------

class TestSeverityMapping:
    def test_currency_failures_are_high(self):
        fake_run = TestCheckCurrency()._fake_run_factory(behind=1)
        with patch.object(di, "_run", side_effect=fake_run):
            result = di.check_currency("/srv/git/x")
        assert all(f.severity == "HIGH" for f in result.findings)

    def test_unmapped_and_pull_without_restart_are_normal(self, tmp_path):
        unmapped = str(tmp_path / "u")
        pull_only = str(tmp_path / "p")
        inventory = {
            unmapped: _entry(unmapped, [di.BackingUnit("a.service", "WorkingDirectory", "system", True, "simple")]),
            pull_only: _entry(pull_only, [di.BackingUnit("b.service", "WorkingDirectory", "system", True, "simple")]),
        }
        findings = di.reconcile_inventory(inventory, {"repo": [pull_only]}, {}, {})
        assert findings[unmapped][0].severity == "NORMAL"
        assert findings[pull_only][0].severity == "NORMAL"

    def test_acked_is_info(self, tmp_path):
        clone = tmp_path / "acked"
        clone.mkdir()
        (clone / di._ACK_MARKER).write_text("")
        inventory = {
            str(clone): _entry(str(clone), [di.BackingUnit("a.service", "WorkingDirectory", "system", True, "simple")]),
        }
        findings = di.reconcile_inventory(inventory, {}, {}, {})
        assert findings[str(clone)][0].severity == "INFO"


# ---------------------------------------------------------------------------
# Detection-only guard: no mutating git/systemctl subcommand is ever issued
# ---------------------------------------------------------------------------

_ALLOWED_GIT_VERBS = {"fetch", "rev-parse", "rev-list", "status", "symbolic-ref", "log", "config"}
_ALLOWED_SYSTEMCTL_VERBS = {"show", "list-units"}
_FORBIDDEN_VERBS = {
    "pull", "checkout", "reset", "restart", "stop", "start", "kill",
    "mask", "enable", "disable", "rm", "push", "commit", "add", "branch",
}


class TestDetectionOnlyGuard:
    def test_full_pass_issues_only_read_only_subcommands(self, tmp_path):
        clone = tmp_path / "guard-repo"
        clone.mkdir()
        _write_git_config(clone)

        calls = []

        def fake_run(cmd, timeout=None):
            calls.append(cmd)
            if cmd[0] == "systemctl":
                if "list-units" in cmd:
                    return _cp(0, "" if "--user" in cmd else "guard.service loaded active running Guard\n")
                if "show" in cmd:
                    return _cp(0, (
                        "ActiveState=active\nSubState=running\nType=simple\n"
                        f"WorkingDirectory={clone}\nExecStart=\nEnvironment=\n"
                    ))
            if cmd[0] == "git":
                if "rev-parse" in cmd and "--show-toplevel" in cmd:
                    return _cp(0, str(clone) + "\n")
                if "rev-parse" in cmd:
                    return _cp(0, "deadbeef\n")
                if "symbolic-ref" in cmd:
                    return _cp(0, "main\n")
                if "rev-list" in cmd:
                    return _cp(0, "0\n")
                if "status" in cmd:
                    return _cp(0, "")
                if "fetch" in cmd:
                    return _cp(0, "")
            return _cp(1, "")

        with patch.object(di, "_run", side_effect=fake_run):
            with patch.object(di, "_CLONE_ROOT_PREFIXES", (str(tmp_path),)):
                with patch.object(di, "_discover_editable_installs", return_value={}):
                    with patch.object(di, "_LOCK_DIR", tmp_path / "locks"):
                        status = di.run_reconcile_pass({}, {}, {})

        assert status["clones"]  # sanity: the pass actually produced output

        for cmd in calls:
            if cmd[0] == "git":
                verb = cmd[3] if len(cmd) > 3 and cmd[1] == "-C" else cmd[1]
                assert verb not in _FORBIDDEN_VERBS, f"mutating git verb issued: {cmd}"
                assert verb in _ALLOWED_GIT_VERBS, f"unexpected git verb: {cmd}"
            elif cmd[0] == "systemctl":
                # verb is the first non-flag token
                verb = next((t for t in cmd[1:] if not t.startswith("--")), None)
                assert verb not in _FORBIDDEN_VERBS, f"mutating systemctl verb issued: {cmd}"
                assert verb in _ALLOWED_SYSTEMCTL_VERBS, f"unexpected systemctl verb: {cmd}"


# ---------------------------------------------------------------------------
# write_status_json + render_deploy_status smoke
# ---------------------------------------------------------------------------

class TestStatusOutput:
    def test_write_status_json_roundtrip(self, tmp_path):
        status = {"generated": "2026-07-22T00:00:00+00:00", "clones": []}
        target = tmp_path / "status.json"
        di.write_status_json(status, path=target)
        assert json.loads(target.read_text()) == status

    def test_read_status_json_returns_none_when_absent(self, tmp_path):
        assert di.read_status_json(path=tmp_path / "missing.json") is None

    def test_read_status_json_roundtrips_write(self, tmp_path):
        status = {"generated": "2026-07-22T00:00:00+00:00", "clones": []}
        target = tmp_path / "status.json"
        di.write_status_json(status, path=target)
        assert di.read_status_json(path=target) == status

    def test_read_status_json_raises_on_corrupt_content(self, tmp_path):
        target = tmp_path / "status.json"
        target.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            di.read_status_json(path=target)

    def test_high_finding_keys_extracts_only_high_severity(self):
        status = {
            "clones": [
                {"path": "/srv/git/a", "findings": [
                    {"kind": "tracked_dirty_tree", "severity": "HIGH", "detail": "x"},
                    {"kind": "unmapped_live_clone", "severity": "NORMAL", "detail": "y"},
                ]},
                {"path": "/srv/git/b", "findings": [
                    {"kind": "stray_branch", "severity": "HIGH", "detail": "z"},
                ]},
            ],
        }
        assert di.high_finding_keys(status) == {
            ("/srv/git/a", "tracked_dirty_tree"),
            ("/srv/git/b", "stray_branch"),
        }

    def test_render_deploy_status_sorts_high_first(self):
        status = {
            "generated": "now",
            "clones": [
                {"path": "/srv/git/b", "backing_units": [], "commits_behind": 0,
                 "branch": "main", "tracked_dirty": False, "untracked_present": False,
                 "mapped": True, "findings": []},
                {"path": "/srv/git/a", "backing_units": [], "commits_behind": 5,
                 "branch": "fix/x", "tracked_dirty": True, "untracked_present": False,
                 "mapped": True, "findings": [{"kind": "stale_behind_origin", "severity": "HIGH", "detail": "x"}]},
            ],
        }
        rendered = di.render_deploy_status(status)
        lines = [l for l in rendered.splitlines() if l.startswith(("a", "b")) or "/a " in l or "/b " in l]
        idx_a = rendered.index("\na ") if "\na " in rendered else rendered.index(" a ")
        idx_b = rendered.index("\nb ") if "\nb " in rendered else rendered.index(" b ")
        assert idx_a < idx_b
