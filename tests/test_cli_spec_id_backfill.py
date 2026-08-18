"""CLI wiring smoke test for `lapis-pm spec-id-backfill` (spec-corpus-id-backfill-v0).

The module-level behavior (guards, fixtures, report shape) is covered in
tests/test_spec_id_backfill.py — this just proves argparse wiring reaches
lapis_pm.spec_id_backfill.run_backfill with the room-root the conftest
session fixture already pins to a throwaway tmp dir (never production /room).
"""
from __future__ import annotations

import os

from lapis_pm.cli import main


def _write_spec(specs_dir, stem: str) -> None:
    specs_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "---\n"
        "status: draft\n"
        "---\n\n"
        f"**Target ID:** `{stem}`\n"
        "**Repo:** `lapis-pm`\n"
        "**Authority:** advisory\n"
        "**Consumer:** some reader\n"
    )
    (specs_dir / f"{stem}.md").write_text(text, encoding="utf-8")


class TestCliSpecIdBackfill:
    def test_dry_run_default_no_writes(self, capsys):
        room_root = os.environ["ROOM_ROOT"]  # session-pinned throwaway tmp dir
        specs_dir = __import__("pathlib").Path(room_root) / "planning" / "specs"
        _write_spec(specs_dir, "cli-test-spec")
        original = (specs_dir / "cli-test-spec.md").read_text(encoding="utf-8")

        rc = main(["spec-id-backfill"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "DRY-RUN" in out
        assert "modified=1" in out
        assert (specs_dir / "cli-test-spec.md").read_text(encoding="utf-8") == original

    def test_apply_writes(self, capsys):
        room_root = os.environ["ROOM_ROOT"]
        specs_dir = __import__("pathlib").Path(room_root) / "planning" / "specs"
        _write_spec(specs_dir, "cli-test-spec-2")

        rc = main(["spec-id-backfill", "--apply"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "APPLIED" in out
        new_text = (specs_dir / "cli-test-spec-2.md").read_text(encoding="utf-8")
        assert "spec_id: cli-test-spec-2" in new_text
