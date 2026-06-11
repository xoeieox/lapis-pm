"""Tests for spec-review --council-voicing argparse choices.

Verifies:
  - gravitywell is accepted as a valid choice
  - existing choices (local, haiku, sonnet, opus) still work
  - unknown values are rejected
"""

import argparse
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from lapis_pm.cli import main


def test_spec_review_council_voicing_gravitywell_accepted(tmp_path):
    """--council-voicing gravitywell is accepted by argparse."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Spec: Test\n**Target ID:** `test-id`\n**Repo:** `lapis-pm`\n**Authority:** advisory\n")

    # Mock cmd_spec_review to avoid actually running the spec review
    with patch("lapis_pm.cli.cmd_spec_review", return_value=0):
        rc, _, _ = _run_main(
            ["spec-review", str(spec_file), "--council-voicing", "gravitywell"],
            tmp_path,
        )
    assert rc == 0, "gravitywell should be accepted"


def test_spec_review_council_voicing_local_accepted(tmp_path):
    """--council-voicing local (default) is accepted."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Spec: Test\n**Target ID:** `test-id`\n**Repo:** `lapis-pm`\n**Authority:** advisory\n")

    with patch("lapis_pm.cli.cmd_spec_review", return_value=0):
        rc, _, _ = _run_main(
            ["spec-review", str(spec_file), "--council-voicing", "local"],
            tmp_path,
        )
    assert rc == 0, "local should be accepted"


def test_spec_review_council_voicing_haiku_accepted(tmp_path):
    """--council-voicing haiku is accepted."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Spec: Test\n**Target ID:** `test-id`\n**Repo:** `lapis-pm`\n**Authority:** advisory\n")

    with patch("lapis_pm.cli.cmd_spec_review", return_value=0):
        rc, _, _ = _run_main(
            ["spec-review", str(spec_file), "--council-voicing", "haiku"],
            tmp_path,
        )
    assert rc == 0, "haiku should be accepted"


def test_spec_review_council_voicing_sonnet_accepted(tmp_path):
    """--council-voicing sonnet is accepted."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Spec: Test\n**Target ID:** `test-id`\n**Repo:** `lapis-pm`\n**Authority:** advisory\n")

    with patch("lapis_pm.cli.cmd_spec_review", return_value=0):
        rc, _, _ = _run_main(
            ["spec-review", str(spec_file), "--council-voicing", "sonnet"],
            tmp_path,
        )
    assert rc == 0, "sonnet should be accepted"


def test_spec_review_council_voicing_opus_accepted(tmp_path):
    """--council-voicing opus is accepted."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Spec: Test\n**Target ID:** `test-id`\n**Repo:** `lapis-pm`\n**Authority:** advisory\n")

    with patch("lapis_pm.cli.cmd_spec_review", return_value=0):
        rc, _, _ = _run_main(
            ["spec-review", str(spec_file), "--council-voicing", "opus"],
            tmp_path,
        )
    assert rc == 0, "opus should be accepted"


def test_spec_review_council_voicing_invalid_rejected(tmp_path):
    """--council-voicing with invalid value is rejected."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Spec: Test\n**Target ID:** `test-id`\n**Repo:** `lapis-pm`\n**Authority:** advisory\n")

    rc, _, err = _run_main(
        ["spec-review", str(spec_file), "--council-voicing", "invalid-voicing"],
        tmp_path,
    )
    assert rc != 0, "invalid voicing should be rejected"
    assert "invalid choice" in err.lower(), "error should mention invalid choice"


def test_spec_review_council_voicing_default_local(tmp_path):
    """spec-review without --council-voicing defaults to local."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Spec: Test\n**Target ID:** `test-id`\n**Repo:** `lapis-pm`\n**Authority:** advisory\n")

    captured_args = {}

    def mock_cmd_spec_review(args):
        captured_args.update(vars(args))
        return 0

    with patch("lapis_pm.cli.cmd_spec_review", side_effect=mock_cmd_spec_review):
        rc, _, _ = _run_main(
            ["spec-review", str(spec_file)],
            tmp_path,
        )

    assert rc == 0
    assert captured_args.get("council_voicing") == "local", "default should be local"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_main(argv, tmp_path):
    """Run main() with mocked dependencies and return (rc, stdout, stderr)."""
    from contextlib import redirect_stdout, redirect_stderr
    from unittest.mock import MagicMock

    out_buf = StringIO()
    err_buf = StringIO()

    with (
        patch("lapis_pm.cli.TargetStore", return_value=MagicMock()),
        patch("lapis_pm.cli.episodic.spec", return_value=None),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        try:
            rc = main(argv)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1

    return rc, out_buf.getvalue(), err_buf.getvalue()
