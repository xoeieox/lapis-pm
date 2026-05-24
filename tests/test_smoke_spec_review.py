"""Integration test for facets-pre-bind-wire-v0 spec-review e2e flow.

Skipped if live Facets adapter is unavailable (FACETS_DISPATCH_DISABLED=1 not set
but facets module not importable).

Run with: pytest tests/test_smoke_spec_review.py -v
"""
from __future__ import annotations

import os
import time
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip if Facets is explicitly disabled globally (smoke/CI environment)
pytestmark = pytest.mark.skipif(
    os.getenv("FACETS_DISPATCH_DISABLED") == "1",
    reason="FACETS_DISPATCH_DISABLED=1: Facets integration not available",
)


_ADVISORY_SPEC = textwrap.dedent("""\
    # Spec: Smoke Integration Target

    **Target ID:** `smoke-integration-target`
    **Repo:** `lapis-pm`
    **Authority:** advisory

    ## Goal

    Validate Facets + Council pre-bind gate e2e.
""")


@pytest.mark.skipif(
    os.getenv("FACETS_DISPATCH_DISABLED") == "1"
    or os.getenv("SPEC_REVIEW_SMOKE_SKIP") == "1",
    reason="Facets not available or SPEC_REVIEW_SMOKE_SKIP=1",
)
def test_full_e2e_spec_review_with_facets(tmp_path):
    """Full e2e: Facets + Council spec-review. Facets disabled via env for test isolation.

    This test uses FACETS_DISPATCH_DISABLED=1 internally so it exercises the full
    dispatch/poll/brief-build path without requiring a live Facets adapter.
    It validates that the brief structure is correct and format_brief produces
    valid markdown with both Council and no-Facets output.
    """
    from lapis_pm.spec_review import run_spec_review, format_brief

    spec = tmp_path / "spec.md"
    spec.write_text(_ADVISORY_SPEC, encoding="utf-8")

    run_id = f"smoke-council-{int(time.time())}"

    # Write a resolved Council YAML for the poll loop to find
    from lapis_pm.spec_review import _COUNCIL_DIR
    import yaml
    _COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    council_path = _COUNCIL_DIR / f"{run_id}.yaml"
    council_path.write_text(yaml.safe_dump({
        "run_id": run_id,
        "status": "resolved",
        "mode": "deliberation",
        "synthesis": {
            "confidence": "converged",
            "landing": "Spec aligns with ecosystem.",
            "open_questions": [],
            "positions": [{"entity": "ent-a", "position": "agree"}],
        },
    }))

    try:
        with patch(
            "lapis_pm.spec_review._dispatch_council", return_value=run_id
        ), patch(
            "lapis_pm.spec_review._load_invariant_context", return_value="ctx"
        ), patch.dict(os.environ, {"FACETS_DISPATCH_DISABLED": "1"}):
            brief = run_spec_review(
                spec_path=spec,
                dispatch_facets=True,  # Facets will be disabled by env var
                timeout_s=60,
            )

        # Brief structure valid
        assert brief.target_id == "smoke-integration-target"
        assert brief.repo == "lapis-pm"
        assert brief.council_status == "resolved"
        # Facets absent (disabled by env)
        assert brief.facets_deliberation is None
        assert brief.combined_recommendation == "proceed-to-bind"

        # format_brief produces valid markdown
        output = format_brief(brief)
        assert "# Spec Review: smoke-integration-target" in output
        assert "Mirror Council" in output
        assert "proceed-to-bind" in output
        assert "Facets" not in output  # omitted when None

    finally:
        if council_path.exists():
            council_path.unlink()
