"""Tests for `lapis-pm bind` destination + loom_visibility fields (lapis-cockpit-loom-schema-v0 leg 1).

Coverage:
  Destination (single-target):
  - bind with all three destination flags writes expected YAML block
  - bind without destination flags writes target without the field
  - --destination-slug without --destination-name errors
  - --destination-name without --destination-slug errors
  - --destination-when without --destination-slug errors
  - slug regex enforcement: invalid patterns rejected, valid patterns accepted
  - _humanize_chain_group produces sentence-case output

  Destination (chain-mode):
  - chain bind with three legs writes per-leg destination from chain group
  - chain bind with destination flags errors (any of the three)
  - chain rebind via --force preserves prior destination (load-bearing guard)
  - chain rebind on target with explicit single-target destination preserves it

  loom_visibility (single-target):
  - bind with --loom-visibility pinned/default/hidden writes field
  - bind without --loom-visibility: field absent on resulting YAML
  - --loom-visibility unknown-value errors at argparse time (choices enforcement)
  - bind --force --loom-visibility default overwrites prior pinned value

  loom_visibility (chain-mode):
  - chain bind with --loom-visibility writes value on every leg
  - chain bind without --loom-visibility: no field on any leg
  - chain rebind without --loom-visibility preserves prior value on legs that had one

  Round-trip:
  - bind with destination + visibility → list --json includes both fields with bound values
  - bind without either → both keys present with None value in JSON
"""

from __future__ import annotations

import io
import json
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import pytest
import yaml

from agents_core.targets import TargetStore
from lapis_pm.cli import _humanize_chain_group, build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPEC_BODY = textwrap.dedent("""\
    # Test spec

    A minimal spec body for unit testing.
""")


def _write_spec(tmp_path: Path, name: str = "spec.md") -> Path:
    p = tmp_path / name
    p.write_text(SPEC_BODY)
    return p


def _make_mem_store():
    from agents_core.mem import MemoryStore
    tmp = tempfile.mktemp(suffix=".db")
    return MemoryStore(db_path=Path(tmp))


def _run(argv: list[str], targets_dir: Path) -> tuple[int, str, str]:
    """Run main() with a patched TargetStore and stubbed episodic/pm_core."""
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    with (
        patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
        patch("lapis_pm.cli.episodic.spec", return_value=None),
        patch("lapis_pm.cli.episodic.write_spec", return_value=None),
        patch("lapis_pm.cli.pm_core.clear_classified_prs", return_value=None),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        rc = main(argv)

    return rc, out_buf.getvalue(), err_buf.getvalue()


def _run_chain(argv: list[str], targets_dir: Path, monkeypatch) -> tuple[int, str, str]:
    """Run main() in chain mode with fully stubbed mem + chain/episodic/pm_core."""
    mem = _make_mem_store()
    monkeypatch.setattr("lapis_pm.chain._mem", lambda: mem)
    monkeypatch.setattr("lapis_pm.pm_core._mem", lambda: mem)
    monkeypatch.setattr("lapis_pm.episodic.spec", lambda tid: None)
    monkeypatch.setattr("lapis_pm.episodic.write_spec", lambda tid, body: None)
    monkeypatch.setattr("lapis_pm.pm_core.clear_classified_prs", lambda tid: None)

    out_buf = io.StringIO()
    err_buf = io.StringIO()

    with (
        patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        rc = main(argv)

    return rc, out_buf.getvalue(), err_buf.getvalue()


def _write_legs_yaml(tmp_path: Path, tids: list[str], chain_group: str) -> Path:
    legs = []
    for i, tid in enumerate(tids):
        leg: dict = {
            "tid": tid,
            "repo": "lapis-pm",
            "authority": "advisory",
            "intent": f"Implement {tid}.",
            "branch_slug": "implement",
        }
        if i > 0:
            leg["depends_on"] = [tids[i - 1]]
        legs.append(leg)
    legs_path = tmp_path / "legs.yaml"
    legs_path.write_text(yaml.safe_dump({"legs": legs}))
    return legs_path


def _run_list_json(targets_dir: Path) -> tuple[int, list]:
    """Run `lapis-pm list --json` with stubbed pm_core, return (rc, parsed_list)."""
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    with (
        patch("lapis_pm.cli.TargetStore", lambda: TargetStore(targets_dir)),
        patch("lapis_pm.cli.pm_core.load_dispatched", return_value=[]),
        patch("lapis_pm.cli.pm_core.get_outstanding_brief", return_value=None),
        patch("lapis_pm.cli.pm_core.get_cursor", return_value=None),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        rc = main(["list", "--json"])

    return rc, json.loads(out_buf.getvalue())


# ---------------------------------------------------------------------------
# _humanize_chain_group unit tests
# ---------------------------------------------------------------------------


class TestHumanizeChainGroup:
    def test_basic_sentence_case(self):
        assert _humanize_chain_group("lapis-cockpit-loom-v0") == "Lapis cockpit loom v0"

    def test_single_word(self):
        assert _humanize_chain_group("foo") == "Foo"

    def test_single_char(self):
        assert _humanize_chain_group("x") == "X"

    def test_empty_string(self):
        assert _humanize_chain_group("") == ""

    def test_underscores_treated_as_spaces(self):
        assert _humanize_chain_group("my_chain_v1") == "My chain v1"

    def test_not_title_case(self):
        # Sentence case only — first word capitalized, rest lower
        result = _humanize_chain_group("lapis-cockpit-loom-v0")
        assert result == "Lapis cockpit loom v0"
        assert "Cockpit" not in result  # not title case


# ---------------------------------------------------------------------------
# Destination: single-target bind
# ---------------------------------------------------------------------------


class TestBindDestinationSingleTarget:
    def test_bind_with_all_destination_flags_writes_yaml_block(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-slug", "cockpit-loom",
            "--destination-name", "Lapis Cockpit Loom",
            "--destination-when", "this week",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "my-target.yaml").read_text())
        assert "destination" in data
        dest = data["destination"]
        assert dest["slug"] == "cockpit-loom"
        assert dest["name"] == "Lapis Cockpit Loom"
        assert dest["when"] == "this week"

    def test_bind_with_slug_and_name_no_when(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-slug", "my-dest",
            "--destination-name", "My Dest",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "my-target.yaml").read_text())
        dest = data["destination"]
        assert dest["slug"] == "my-dest"
        assert dest["name"] == "My Dest"
        assert dest["when"] is None

    def test_bind_without_destination_flags_no_field(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "my-target.yaml").read_text())
        assert "destination" not in data

    def test_slug_without_name_errors(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-slug", "my-dest",
        ], tmp_path)

        assert rc == 2
        assert "must both be set or both be omitted" in err

    def test_name_without_slug_errors(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-name", "My Dest",
        ], tmp_path)

        assert rc == 2
        assert "must both be set or both be omitted" in err

    def test_when_without_slug_errors(self, tmp_path):
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-when", "tomorrow",
        ], tmp_path)

        assert rc == 2
        assert "--destination-when requires --destination-slug" in err

    def test_rebind_force_preserves_prior_destination(self, tmp_path):
        """bind --force without destination flags preserves existing destination."""
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")

        # First bind: set destination
        _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-slug", "original-dest",
            "--destination-name", "Original Dest",
        ], tmp_path)

        # Second bind: no destination flags → preserve
        rc, out, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "my-target.yaml").read_text())
        assert data["destination"]["slug"] == "original-dest"


# ---------------------------------------------------------------------------
# Destination: slug regex enforcement
# ---------------------------------------------------------------------------


class TestDestinationSlugRegex:
    """Slug regex enforcement via _BRANCH_SLUG_RE (same object as chain.py:51)."""

    def _bind(self, tmp_path, slug):
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("slug-test", title="Slug Test")
        return _run([
            "bind", "slug-test",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-slug", slug,
            "--destination-name", "Name",
        ], tmp_path)

    def test_leading_slash_invalid(self, tmp_path):
        rc, _, err = self._bind(tmp_path, "lapis/foo")
        assert rc == 2
        assert "kebab-case" in err

    def test_uppercase_invalid(self, tmp_path):
        rc, _, err = self._bind(tmp_path, "Foo")
        assert rc == 2
        assert "kebab-case" in err

    def test_leading_dash_invalid(self, tmp_path):
        # Pass via = form so argparse doesn't interpret -bad as a flag name
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("slug-test", title="Slug Test")
        rc, _, err = _run([
            "bind", "slug-test",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-slug=-bad",
            "--destination-name", "Name",
        ], tmp_path)
        assert rc == 2
        assert "kebab-case" in err

    def test_trailing_dash_invalid(self, tmp_path):
        rc, _, err = self._bind(tmp_path, "bad-")
        assert rc == 2
        assert "kebab-case" in err

    def test_underscore_invalid(self, tmp_path):
        rc, _, err = self._bind(tmp_path, "_underscore")
        assert rc == 2
        assert "kebab-case" in err

    def test_single_char_valid(self, tmp_path):
        rc, _, err = self._bind(tmp_path, "f")
        assert rc == 0, f"single char should be valid; err={err!r}"

    def test_simple_slug_valid(self, tmp_path):
        rc, _, err = self._bind(tmp_path, "foo")
        assert rc == 0, f"'foo' should be valid; err={err!r}"

    def test_chain_id_style_valid(self, tmp_path):
        rc, _, err = self._bind(tmp_path, "lapis-cockpit-loom-v0")
        assert rc == 0, f"chain-id-style slug should be valid; err={err!r}"

    def test_empty_name_errors(self, tmp_path):
        """--destination-name with empty string after .strip() errors."""
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("my-target", title="My Target")
        rc, _, err = _run([
            "bind", "my-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-slug", "my-dest",
            "--destination-name", "   ",  # whitespace only → empty after strip
        ], tmp_path)
        assert rc == 2
        assert "non-empty" in err


# ---------------------------------------------------------------------------
# Destination: chain-mode
# ---------------------------------------------------------------------------


class TestBindDestinationChainMode:
    def test_chain_bind_populates_per_leg_destination(self, tmp_path, monkeypatch):
        legs_path = _write_legs_yaml(tmp_path, ["leg-a", "leg-b", "leg-c"], "my-chain-group")
        spec_path = _write_spec(tmp_path)

        rc, out, err = _run_chain([
            "bind", "my-chain-group",
            "--spec-from", str(spec_path),
            "--legs-from", str(legs_path),
            "--create",
            "--no-auto-fire",
        ], tmp_path, monkeypatch)

        assert rc == 0, f"stderr={err!r}"

        store = TargetStore(tmp_path)
        for tid in ("leg-a", "leg-b", "leg-c"):
            data = yaml.safe_load((tmp_path / f"{tid}.yaml").read_text())
            assert "destination" in data, f"{tid} missing destination"
            dest = data["destination"]
            assert dest["slug"] == "my-chain-group"
            assert dest["name"] == "My chain group"
            assert dest["when"] is None

    def test_chain_bind_with_destination_slug_errors(self, tmp_path, monkeypatch):
        legs_path = _write_legs_yaml(tmp_path, ["leg-a"], "grp")
        spec_path = _write_spec(tmp_path)

        rc, out, err = _run_chain([
            "bind", "grp",
            "--spec-from", str(spec_path),
            "--legs-from", str(legs_path),
            "--create",
            "--no-auto-fire",
            "--destination-slug", "my-dest",
        ], tmp_path, monkeypatch)

        assert rc == 2
        assert "destination flags are not allowed in chain mode" in err

    def test_chain_bind_with_destination_name_errors(self, tmp_path, monkeypatch):
        legs_path = _write_legs_yaml(tmp_path, ["leg-a"], "grp")
        spec_path = _write_spec(tmp_path)

        rc, out, err = _run_chain([
            "bind", "grp",
            "--spec-from", str(spec_path),
            "--legs-from", str(legs_path),
            "--create",
            "--no-auto-fire",
            "--destination-name", "My Dest",
        ], tmp_path, monkeypatch)

        assert rc == 2
        assert "destination flags are not allowed in chain mode" in err

    def test_chain_bind_with_destination_when_errors(self, tmp_path, monkeypatch):
        legs_path = _write_legs_yaml(tmp_path, ["leg-a"], "grp")
        spec_path = _write_spec(tmp_path)

        rc, out, err = _run_chain([
            "bind", "grp",
            "--spec-from", str(spec_path),
            "--legs-from", str(legs_path),
            "--create",
            "--no-auto-fire",
            "--destination-when", "tomorrow",
        ], tmp_path, monkeypatch)

        assert rc == 2
        assert "destination flags are not allowed in chain mode" in err

    def test_chain_rebind_preserves_explicit_single_target_destination(
        self, tmp_path, monkeypatch
    ):
        """Chain rebind on a target with prior explicit destination keeps it (guard is load-bearing)."""
        spec_path = _write_spec(tmp_path)

        # Step 1: single-target bind with explicit destination
        store = TargetStore(tmp_path)
        store.create("leg-x", title="Leg X")
        _run([
            "bind", "leg-x",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-slug", "explicit-dest",
            "--destination-name", "Explicit Dest",
        ], tmp_path)

        # Step 2: chain rebind via --force → guard fires, chain-group default NOT applied
        legs = [{"tid": "leg-x", "repo": "lapis-pm", "authority": "advisory",
                 "intent": "do x", "branch_slug": "implement"}]
        legs_path = tmp_path / "legs.yaml"
        legs_path.write_text(yaml.safe_dump({"legs": legs}))

        rc, out, err = _run_chain([
            "bind", "chain-rebind-group",
            "--spec-from", str(spec_path),
            "--legs-from", str(legs_path),
            "--force",
            "--no-auto-fire",
        ], tmp_path, monkeypatch)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "leg-x.yaml").read_text())
        assert data["destination"]["slug"] == "explicit-dest", (
            "chain-group default overwrote explicit destination (guard failed)"
        )


# ---------------------------------------------------------------------------
# loom_visibility: single-target
# ---------------------------------------------------------------------------


class TestLoomVisibilitySingleTarget:
    def _bind_with_visibility(self, tmp_path, visibility_arg=None):
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("vis-target", title="Vis Target")
        argv = [
            "bind", "vis-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
        ]
        if visibility_arg is not None:
            argv += ["--loom-visibility", visibility_arg]
        return _run(argv, tmp_path)

    def test_pinned_writes_field(self, tmp_path):
        rc, _, err = self._bind_with_visibility(tmp_path, "pinned")
        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "vis-target.yaml").read_text())
        assert data["loom_visibility"] == "pinned"

    def test_default_writes_field(self, tmp_path):
        rc, _, err = self._bind_with_visibility(tmp_path, "default")
        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "vis-target.yaml").read_text())
        assert data["loom_visibility"] == "default"

    def test_hidden_writes_field(self, tmp_path):
        rc, _, err = self._bind_with_visibility(tmp_path, "hidden")
        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "vis-target.yaml").read_text())
        assert data["loom_visibility"] == "hidden"

    def test_omitted_flag_no_field(self, tmp_path):
        rc, _, err = self._bind_with_visibility(tmp_path, None)
        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "vis-target.yaml").read_text())
        assert "loom_visibility" not in data

    def test_invalid_value_errors_at_argparse(self, tmp_path):
        """Argparse choices enforcement fires before cmd_bind runs."""
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("vis-target", title="Vis Target")
        with pytest.raises(SystemExit) as exc_info:
            # argparse raises SystemExit(2) for invalid choices
            main([
                "bind", "vis-target",
                "--spec-from", str(spec_path),
                "--repo", "lapis-pm",
                "--loom-visibility", "unknown-value",
            ])
        assert exc_info.value.code == 2

    def test_force_rebind_overwrites_prior_visibility(self, tmp_path):
        """bind --force --loom-visibility default over prior pinned writes default."""
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("vis-target", title="Vis Target")

        # First bind: pinned
        _run([
            "bind", "vis-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--loom-visibility", "pinned",
        ], tmp_path)

        # Second bind: change to default
        rc, _, err = _run([
            "bind", "vis-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--loom-visibility", "default",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "vis-target.yaml").read_text())
        assert data["loom_visibility"] == "default"

    def test_force_rebind_without_flag_preserves_prior_visibility(self, tmp_path):
        """bind --force without --loom-visibility preserves existing value."""
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("vis-target", title="Vis Target")

        # First bind: pinned
        _run([
            "bind", "vis-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--loom-visibility", "pinned",
        ], tmp_path)

        # Second bind: no flag → preserve
        rc, _, err = _run([
            "bind", "vis-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
        ], tmp_path)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "vis-target.yaml").read_text())
        assert data["loom_visibility"] == "pinned"


# ---------------------------------------------------------------------------
# loom_visibility: chain-mode
# ---------------------------------------------------------------------------


class TestLoomVisibilityChainMode:
    def test_chain_bind_with_visibility_writes_every_leg(self, tmp_path, monkeypatch):
        legs_path = _write_legs_yaml(tmp_path, ["leg-a", "leg-b"], "my-chain")
        spec_path = _write_spec(tmp_path)

        rc, out, err = _run_chain([
            "bind", "my-chain",
            "--spec-from", str(spec_path),
            "--legs-from", str(legs_path),
            "--create",
            "--no-auto-fire",
            "--loom-visibility", "pinned",
        ], tmp_path, monkeypatch)

        assert rc == 0, f"stderr={err!r}"
        for tid in ("leg-a", "leg-b"):
            data = yaml.safe_load((tmp_path / f"{tid}.yaml").read_text())
            assert data.get("loom_visibility") == "pinned", (
                f"{tid} expected loom_visibility=pinned, got {data.get('loom_visibility')!r}"
            )

    def test_chain_bind_without_visibility_no_field_on_any_leg(self, tmp_path, monkeypatch):
        legs_path = _write_legs_yaml(tmp_path, ["leg-a", "leg-b"], "my-chain")
        spec_path = _write_spec(tmp_path)

        rc, out, err = _run_chain([
            "bind", "my-chain",
            "--spec-from", str(spec_path),
            "--legs-from", str(legs_path),
            "--create",
            "--no-auto-fire",
        ], tmp_path, monkeypatch)

        assert rc == 0, f"stderr={err!r}"
        for tid in ("leg-a", "leg-b"):
            data = yaml.safe_load((tmp_path / f"{tid}.yaml").read_text())
            assert "loom_visibility" not in data, (
                f"{tid} should not have loom_visibility when flag omitted"
            )

    def test_chain_rebind_without_flag_preserves_prior_visibility(self, tmp_path, monkeypatch):
        """Chain rebind without --loom-visibility preserves each leg's prior value."""
        spec_path = _write_spec(tmp_path)

        # First bind: pinned
        legs_path = _write_legs_yaml(tmp_path, ["leg-a"], "my-chain")
        _run_chain([
            "bind", "my-chain",
            "--spec-from", str(spec_path),
            "--legs-from", str(legs_path),
            "--create",
            "--no-auto-fire",
            "--loom-visibility", "pinned",
        ], tmp_path, monkeypatch)

        # Second bind: no flag
        rc, out, err = _run_chain([
            "bind", "my-chain",
            "--spec-from", str(spec_path),
            "--legs-from", str(legs_path),
            "--force",
            "--no-auto-fire",
        ], tmp_path, monkeypatch)

        assert rc == 0, f"stderr={err!r}"
        data = yaml.safe_load((tmp_path / "leg-a.yaml").read_text())
        assert data.get("loom_visibility") == "pinned", (
            "chain rebind without --loom-visibility should preserve prior value"
        )


# ---------------------------------------------------------------------------
# Round-trip: list --json
# ---------------------------------------------------------------------------


class TestListJsonRoundTrip:
    def test_list_json_includes_destination_and_visibility(self, tmp_path):
        """bind with destination + visibility → list --json includes both fields."""
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("round-trip-target", title="Round Trip")

        rc, _, err = _run([
            "bind", "round-trip-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
            "--destination-slug", "my-dest",
            "--destination-name", "My Dest",
            "--destination-when", "this week",
            "--loom-visibility", "pinned",
        ], tmp_path)
        assert rc == 0, f"bind failed: {err!r}"

        _, items = _run_list_json(tmp_path)
        item = next(i for i in items if i["target_id"] == "round-trip-target")

        assert item["destination"] == {
            "slug": "my-dest",
            "name": "My Dest",
            "when": "this week",
        }
        assert item["loom_visibility"] == "pinned"

    def test_list_json_both_keys_present_with_none_when_absent(self, tmp_path):
        """bind without destination/visibility → both keys in JSON with None value."""
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("empty-target", title="Empty")

        rc, _, err = _run([
            "bind", "empty-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
        ], tmp_path)
        assert rc == 0, f"bind failed: {err!r}"

        _, items = _run_list_json(tmp_path)
        item = next(i for i in items if i["target_id"] == "empty-target")

        assert "destination" in item
        assert item["destination"] is None
        assert "loom_visibility" in item
        assert item["loom_visibility"] is None

    def test_list_json_schema_includes_new_keys(self, tmp_path):
        """The new keys are always present regardless of value (stable consumer shape)."""
        spec_path = _write_spec(tmp_path)
        store = TargetStore(tmp_path)
        store.create("schema-target", title="Schema")

        _run([
            "bind", "schema-target",
            "--spec-from", str(spec_path),
            "--repo", "lapis-pm",
            "--force",
        ], tmp_path)

        _, items = _run_list_json(tmp_path)
        item = items[0]
        assert "destination" in item
        assert "loom_visibility" in item
