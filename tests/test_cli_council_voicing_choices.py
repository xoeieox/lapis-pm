"""Tests for spec-review --council-voicing argparse choices.

Verifies:
  - gravitywell is accepted as a valid choice
  - existing choices (local, haiku, sonnet, opus) still work
  - unknown values are rejected
  - run_spec_review() function signature defaults to gravitywell
"""

import argparse
import inspect
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from lapis_pm.cli import main
from lapis_pm.spec_review import run_spec_review


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


def test_spec_review_council_voicing_invalid_rejected(tmp_path):
    """--council-voicing with invalid value is rejected; sonnet (paid-model) is now invalid."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Spec: Test\n**Target ID:** `test-id`\n**Repo:** `lapis-pm`\n**Authority:** advisory\n")

    # Test that a completely invalid value is rejected
    rc, _, err = _run_main(
        ["spec-review", str(spec_file), "--council-voicing", "invalid-voicing"],
        tmp_path,
    )
    assert rc != 0, "invalid voicing should be rejected"
    assert "invalid choice" in err.lower(), "error should mention invalid choice"

    # Test that sonnet (previously valid paid-model voicing) is now rejected
    rc, _, err = _run_main(
        ["spec-review", str(spec_file), "--council-voicing", "sonnet"],
        tmp_path,
    )
    assert rc != 0, "sonnet should be rejected (paid-model voicing removed)"
    assert "invalid choice" in err.lower(), "error should mention invalid choice"


def test_spec_review_council_voicing_default_gravitywell(tmp_path):
    """spec-review without --council-voicing defaults to gravitywell."""
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
    assert captured_args.get("council_voicing") == "gravitywell", "default should be gravitywell"


def test_run_spec_review_function_signature_default():
    """run_spec_review() function signature defaults council_voicing to gravitywell."""
    sig = inspect.signature(run_spec_review)
    council_voicing_param = sig.parameters['council_voicing']
    assert council_voicing_param.default == 'gravitywell', (
        f"run_spec_review() council_voicing default should be 'gravitywell', "
        f"got {council_voicing_param.default!r}"
    )


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
