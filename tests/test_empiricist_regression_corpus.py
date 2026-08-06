"""Regression corpus for the Empiricist reference seat (DoD item 3 of
lapis-pm-empiricist-reference-seat-v0).

The fixture spec (tests/fixtures/spec_review/empiricist_regression_spec.md)
plants one false claim from each of four historical miss classes plus two
true claims (one purely in-repo, one a host fact absent from the repo — the
miss-#5 shape). The recorded response
(tests/fixtures/spec_review/empiricist_regression_response.json) is what a
correctly-behaving Empiricist should produce against that fixture.

A live model call is impractical in unit tests (per the spec), so this test
is driven by the recorded/stubbed response rather than a live GravityWell
call — the live run happens as DoD item 5, a real gate run.
"""
from __future__ import annotations

import json
from pathlib import Path

from lapis_pm.spec_review import _build_brief

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "spec_review"
SPEC_PATH = FIXTURES_DIR / "empiricist_regression_spec.md"
RESPONSE_PATH = FIXTURES_DIR / "empiricist_regression_response.json"

# Planted false claims, identified by a substring that must appear somewhere
# in the flagged issue for that claim (citation or note).
PLANTED_FALSE_CLAIMS = [
    "nonexistent_reconcile_module.py",
    "spec_review.py:1",
    "timeout_s",
    "empiricist-seat-does-not-exist-fabricated-key",
]

# Planted true claims — must NOT appear as a flagged issue.
PLANTED_TRUE_CLAIMS = [
    "spec_review.py:457",
    "gravitywell-slot1",
]

TOTAL_PLANTED_CLAIMS = len(PLANTED_FALSE_CLAIMS) + len(PLANTED_TRUE_CLAIMS)


def _load_response() -> dict:
    return json.loads(RESPONSE_PATH.read_text(encoding="utf-8"))


def _council_raw() -> dict:
    return {
        "status": "resolved",
        "landing": "test landing",
        "open_questions": [],
        "confidence": "converged",
        "positions": [],
        "run_id": "council-fixture",
    }


def test_fixture_spec_exists_and_has_planted_claims():
    text = SPEC_PATH.read_text(encoding="utf-8")
    for needle in PLANTED_FALSE_CLAIMS + PLANTED_TRUE_CLAIMS:
        assert needle in text, f"fixture spec missing planted claim marker: {needle!r}"


def test_recorded_response_flags_every_false_claim():
    response = _load_response()
    issue_text = "\n".join(
        f"{i.get('citation', '')} {i.get('note', '')}" for i in response["issues"]
    )
    missing = [c for c in PLANTED_FALSE_CLAIMS if c not in issue_text]
    assert not missing, f"recorded Empiricist response failed to flag: {missing}"


def test_recorded_response_does_not_flag_true_claims():
    response = _load_response()
    issue_text = "\n".join(
        f"{i.get('citation', '')} {i.get('note', '')}" for i in response["issues"]
    )
    falsely_flagged = [c for c in PLANTED_TRUE_CLAIMS if c in issue_text]
    assert not falsely_flagged, (
        f"recorded Empiricist response incorrectly flagged true claims: {falsely_flagged} "
        "— a seat that reports everything as unverifiable is as useless as one that "
        "reports nothing"
    )


def test_recorded_response_claims_checked_covers_all_planted_claims():
    response = _load_response()
    assert response["claims_checked"] >= TOTAL_PLANTED_CLAIMS, (
        f"claims_checked={response['claims_checked']} is less than the "
        f"{TOTAL_PLANTED_CLAIMS} planted claims in the fixture — claims_checked: 0 "
        "(or any undercount) on a spec making factual claims is a broken lens, "
        "never a pass (gate pass 1 resolution)."
    )


def test_recorded_response_high_severity_only_for_proven_false():
    """severity=high must mean 'checked and proven false', not merely 'unconfirmed'."""
    response = _load_response()
    for issue in response["issues"]:
        if issue["severity"] == "high":
            note = issue["note"].lower()
            assert "does not exist" in note or "not " in note or "actual value" in note, (
                f"issue marked high severity without stating a proven-false finding: {issue}"
            )


def test_response_plumbs_through_build_brief_end_to_end(tmp_path):
    """The recorded response, fed through _build_brief exactly as the live
    poller would, must surface claims_checked and all four false-claim issues
    on the resulting SpecReviewBrief."""
    response = _load_response()

    brief = _build_brief(
        sonnet_raw=response,
        council_raw=_council_raw(),
        spec_path=SPEC_PATH,
        parsed_target_id="empiricist-regression-fixture-v0",
        repo="lapis-pm",
        elapsed_s=5.0,
    )

    assert brief.sonnet_claims_checked == 6
    assert len(brief.sonnet_issues) == len(PLANTED_FALSE_CLAIMS)
    assert brief.sonnet_verdict == "fixable"
