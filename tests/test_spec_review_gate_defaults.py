"""Tests for lapis-pm-gate-defaults-and-dead-poll-v0.

Covers:
  U3a: reference (Empiricist) leg off by default; --no-reference-reviewer is a
       harmless no-op (never errors, never inverts); default poll timeout >= 1200;
       GW_REVIEWER_TIMEOUT_SEC is not CLI-settable (Transmuter fold).
  U3b: GW leg never submitted on the default path; --with-gw opts in and blocks;
       a --with-gw timeout still renders the brief with an explicit
       failed-grounding marker.
  U3c: SpecReviewBrief.grounding_status / grounding_reason.
  U3d: the cross-session lock is released before the Empiricist poll and held
       across the GW join and across Facets+Council on the default path.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import inspect
import os
import textwrap
import threading
from unittest.mock import MagicMock, patch

import pytest

from agents_core.shared_deliberation.envelope import DeliberationEnvelope
from lapis_pm.cli import main
from lapis_pm.spec_review import (
    _detect_grounding_status,
    _resolve_grounding_sha_and_age,
    run_spec_review,
)


def _happy_envelope():
    return DeliberationEnvelope(
        deliberation_request_id="test-req-gate-defaults",
        triage="full",
        facets_ok=True,
        facets={
            "deliberation_id": "facets-gate-defaults",
            "synthesis": {
                "escalation_recommendation": "proceed",
                "consensus_level": "strong",
                "confidence": "high",
                "recommendation": "All clear",
            },
            "stances": [],
            "methodology": {"synthesis_operator": "haiku"},
        },
        facets_deliberation_id="facets-gate-defaults",
        operator_requested="haiku",
        operator_effective="haiku",
        council_ok=True,
        council_run_id="council-gate-defaults",
        council_status="resolved",
        council_landing="aligns",
        council_confidence="converged",
        council_open_questions=[],
        council_positions=[{"entity": "ent-a", "position": "agree"}],
        council_voicing_requested="gravitywell",
        council_voicing_effective="gravitywell",
        council_voicing_degraded=False,
    )


@pytest.fixture
def advisory_spec(tmp_path):
    spec_path = tmp_path / "gate_defaults_test_spec.md"
    spec_path.write_text(
        textwrap.dedent("""\
            # Test Spec: gate defaults

            **Target ID:** `gate-defaults-test`
            **Repo:** `lapis-pm`
            **Authority:** advisory

            ## Goal
            Verify default-off legs and grounding status.
        """),
        encoding="utf-8",
    )
    return spec_path


@pytest.fixture(autouse=True)
def _isolated_lock(tmp_path_factory, monkeypatch):
    """Every test in this file uses a tmp-dir lock file, never the real
    cross-session /run/user/<uid> lock — this suite runs on a shared host where a
    live, legitimate spec-review gate may hold that lock for up to 30 minutes."""
    lock_dir = tmp_path_factory.mktemp("gate-defaults-lock")
    lock_file = lock_dir / "lapis-pm-spec-review.lock"
    monkeypatch.setattr("lapis_pm.spec_review._spec_review_lock_path", lambda: lock_file)
    return lock_file


# ---------------------------------------------------------------------------
# DoD 1 / DoD 4: default path submits neither leg
# ---------------------------------------------------------------------------

def test_default_path_dispatches_neither_reference_nor_gw_leg(advisory_spec, monkeypatch):
    """A bare run (no with_gw, no reference_reviewer) must not touch the
    ThreadPoolExecutor (GW) or _dispatch_spec_reviewer (Empiricist) at all."""
    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", lambda r: _happy_envelope())

    executor_calls = []
    dispatch_calls = []
    monkeypatch.setattr(
        "lapis_pm.spec_review.ThreadPoolExecutor",
        lambda *a, **k: executor_calls.append((a, k)) or MagicMock(),
    )
    monkeypatch.setattr(
        "lapis_pm.spec_review._dispatch_spec_reviewer",
        lambda **k: dispatch_calls.append(k) or (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    brief = run_spec_review(advisory_spec, council_voicing="gravitywell", timeout_s=1800, dispatch_facets=True)

    assert executor_calls == [], "default path must not construct a ThreadPoolExecutor (no GW future)"
    assert dispatch_calls == [], "default path must not dispatch the Empiricist leg"
    assert brief.gw_ran is False
    assert brief.reference_verdict == "skip"


# ---------------------------------------------------------------------------
# DoD 2: --no-reference-reviewer is a harmless no-op
# ---------------------------------------------------------------------------

def test_no_reference_reviewer_flag_does_not_error(advisory_spec):
    """--no-reference-reviewer must still parse cleanly (never errors)."""
    from lapis_pm.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["spec-review", str(advisory_spec), "--no-reference-reviewer"])
    assert args.no_reference_reviewer is True


def test_no_reference_reviewer_flag_never_inverts_to_enable(advisory_spec):
    """Passing --no-reference-reviewer alone must NOT enable the leg (default is off;
    the flag must never invert to mean 'turn it on')."""
    captured = {}

    def fake_run_spec_review(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch("lapis_pm.spec_review.run_spec_review", side_effect=fake_run_spec_review), \
         patch("lapis_pm.spec_review.format_brief", return_value=""):
        rc = main(["spec-review", str(advisory_spec), "--no-reference-reviewer"])

    assert rc == 0
    assert captured.get("reference_reviewer") is False, (
        "the deprecated --no-reference-reviewer no-op must never result in "
        "reference_reviewer=True"
    )


def test_with_reference_reviewer_is_the_only_way_to_enable():
    """--with-reference-reviewer (and only it) enables the leg."""
    from lapis_pm.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["spec-review", "spec.md", "--with-reference-reviewer"])
    assert args.with_reference_reviewer is True

    args2 = parser.parse_args(["spec-review", "spec.md"])
    assert getattr(args2, "with_reference_reviewer", False) is False


# ---------------------------------------------------------------------------
# DoD 3: default poll timeout >= 1200
# ---------------------------------------------------------------------------

def test_run_spec_review_default_timeout_at_least_1200():
    sig = inspect.signature(run_spec_review)
    default = sig.parameters["timeout_s"].default
    assert default >= 1200, f"run_spec_review timeout_s default must be >=1200, got {default}"


def test_cli_spec_review_default_timeout_at_least_1200():
    from lapis_pm.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["spec-review", "spec.md"])
    assert args.timeout >= 1200, f"--timeout default must be >=1200, got {args.timeout}"


# ---------------------------------------------------------------------------
# Transmuter fold: GW_REVIEWER_TIMEOUT_SEC must not be CLI-settable (DoD 12)
# ---------------------------------------------------------------------------

def test_gw_reviewer_timeout_not_settable_via_cli():
    from lapis_pm.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["spec-review", "spec.md", "--gw-reviewer-timeout-sec", "60"])
    with pytest.raises(SystemExit):
        parser.parse_args(["spec-review", "spec.md", "--gw-timeout", "60"])


# ---------------------------------------------------------------------------
# DoD 10 / DoD 13: grounding_status + grounding_reason
# ---------------------------------------------------------------------------

def test_detect_grounding_status_verified():
    facets_dict = {
        "rounds": [
            {"round_num": 0, "sim_failures": {}, "sim_results": {"codebase:foo.py": "ok"}},
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
        ]
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "verified"
    assert reason == ""


def test_detect_grounding_status_failed_recorded_denial():
    """Case (a): a non-final round records a codebase sim_failure."""
    facets_dict = {
        "rounds": [
            {
                "round_num": 0,
                "sim_failures": {"codebase:foo.py": "no target_repo in context"},
                "sim_results": {},
            },
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
        ]
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "failed"
    assert reason and reason != ""


def test_detect_grounding_status_silent_denial():
    """Case (b): empty sim_failures AND empty sim_results — silent denial, not logged
    as a failure. Must still be detected and must still emit (LOW tier)."""
    facets_dict = {
        "rounds": [
            {"round_num": 0, "sim_failures": {}, "sim_results": {}},
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
        ]
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "silent-denial"
    assert reason != "", "silent-denial must carry a non-empty reason"


def test_detect_grounding_status_facets_never_ran():
    status, reason = _detect_grounding_status(None)
    assert status == "failed"
    assert reason != ""


# ---------------------------------------------------------------------------
# lapis-pm-spec-review-grounding-legibility-v0 D1: settled taxonomy split.
# no_repo_in_context is reserved for the genuine repo-absent case resolved in
# agents_core's _resolve_grounding_target; _detect_grounding_status must never
# emit it, since it never observes that signal directly.
# ---------------------------------------------------------------------------

def test_detect_grounding_status_sim_data_missing():
    """Case (b): no sim data recorded at all in a non-final round → sim_data_missing,
    never no_repo_in_context (that name is reserved for the genuine repo-absent case)."""
    facets_dict = {
        "rounds": [
            {"round_num": 0, "sim_failures": {}, "sim_results": {}},
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
        ]
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "silent-denial"
    assert reason == "sim_data_missing"
    assert reason != "no_repo_in_context"


def test_detect_grounding_status_sim_data_malformed():
    """Sim data was recorded in a non-final round, but none of it is codebase-shaped
    → sim_data_malformed, distinct from sim_data_missing and from no_repo_in_context."""
    facets_dict = {
        "rounds": [
            {
                "round_num": 0,
                "sim_failures": {},
                "sim_results": {"other-surface:foo": "some result"},
            },
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
        ]
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "silent-denial"
    assert reason == "sim_data_malformed"
    assert reason != "no_repo_in_context"
    assert reason != "sim_data_missing"


@pytest.mark.parametrize("facets_dict", [
    {"rounds": [{"round_num": 0, "sim_failures": {}, "sim_results": {}},
                {"round_num": 1, "sim_failures": {}, "sim_results": {}}]},
    {"rounds": [{"round_num": 0, "sim_failures": {}, "sim_results": {"x:y": "z"}},
                {"round_num": 1, "sim_failures": {}, "sim_results": {}}]},
])
def test_detect_grounding_status_never_returns_no_repo_in_context(facets_dict):
    """_detect_grounding_status has no visibility into whether the repo was in
    context at all — that check happens upstream in orchestrator.py's
    _resolve_grounding_target. It must never emit no_repo_in_context itself."""
    _status, reason = _detect_grounding_status(facets_dict)
    assert reason != "no_repo_in_context"


def test_detect_grounding_status_verified_unchanged_regression_fence():
    facets_dict = {
        "rounds": [
            {"round_num": 0, "sim_failures": {}, "sim_results": {"codebase:foo.py": "ok"}},
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
        ]
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "verified"
    assert reason == ""


def test_detect_grounding_status_failed_unchanged_regression_fence():
    facets_dict = {
        "rounds": [
            {
                "round_num": 0,
                "sim_failures": {"codebase:foo.py": "no target_repo in context"},
                "sim_results": {},
            },
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
        ]
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "failed"
    assert reason == "codebase_surface_denied"



# ---------------------------------------------------------------------------
# lapis-pm-gate-brief-tells-the-truth-v0 D4: context.target_repo false negative.
# Mode-1 auto-grounding threads its report into working_context and
# context.target_repo (facets/adapter.py:947-949) but never into any
# RoundRecord, so a fully-grounded run previously fell through to
# silent-denial/sim_data_missing despite having a real grounding target.
# ---------------------------------------------------------------------------

def test_detect_grounding_status_auto_grounded_via_target_repo():
    """Real shape from /srv/lapis/facets/deliberations/2026-08-06-112305-8465de.json:
    empty sim_results/sim_failures in every round, but context.target_repo set
    to a resolved auto-grounding worktree. Must report verified, not
    silent-denial."""
    facets_dict = {
        "context": {
            "spec_path": "/srv/lapis/planning/specs/agents-core-shaper-capture-submit-taskid-v0.md",
            "target_repo": "/tmp/grounding-agents-core-kl4x7962",
        },
        "rounds": [
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
            {"round_num": 2, "sim_failures": {}, "sim_results": {}},
        ],
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "verified"
    assert reason == "auto-grounded"


def test_detect_grounding_status_codebase_denial_still_wins_over_target_repo():
    """Precedence order: an explicit codebase denial in a non-final round must
    still return failed/codebase_surface_denied even if context.target_repo
    happens to be set."""
    facets_dict = {
        "context": {"target_repo": "/tmp/some-worktree"},
        "rounds": [
            {
                "round_num": 0,
                "sim_failures": {"codebase:foo.py": "no target_repo in context"},
                "sim_results": {},
            },
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
        ],
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "failed"
    assert reason == "codebase_surface_denied"


def test_detect_grounding_status_sim_data_missing_when_no_target_repo():
    """Regression fence: with no context.target_repo and no sim data, the
    existing sim_data_missing fallback is unchanged."""
    facets_dict = {
        "context": {},
        "rounds": [
            {"round_num": 0, "sim_failures": {}, "sim_results": {}},
            {"round_num": 1, "sim_failures": {}, "sim_results": {}},
        ],
    }
    status, reason = _detect_grounding_status(facets_dict)
    assert status == "silent-denial"
    assert reason == "sim_data_missing"


def test_resolve_grounding_sha_and_age_never_fabricates_on_missing_clone():
    """A nonexistent local clone must degrade to ("", None), never raise, never
    invent a sha."""
    sha, age = _resolve_grounding_sha_and_age("definitely-not-a-real-repo-xyz")
    assert sha == ""
    assert age is None


def test_grounding_status_not_applicable_when_facets_not_dispatched(advisory_spec, monkeypatch):
    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", lambda r: _happy_envelope())
    brief = run_spec_review(
        advisory_spec, council_voicing="gravitywell", timeout_s=30, dispatch_facets=False,
    )
    assert brief.grounding_status == "not-applicable"
    assert brief.grounding_reason == ""


# ---------------------------------------------------------------------------
# DoD 11: --with-gw timeout renders the brief with grounding_status=failed
# ---------------------------------------------------------------------------

def test_with_gw_timeout_renders_brief_with_failed_grounding(advisory_spec, monkeypatch):
    monkeypatch.setenv("GW_REVIEWER_TIMEOUT_SEC", "1")
    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", lambda r: _happy_envelope())
    monkeypatch.setattr(
        "lapis_pm.node_identity.resolve_node_identity",
        lambda force=False: MagicMock(node_role="master"),
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **k: MagicMock(returncode=0))

    fake_future = MagicMock()
    fake_future.result.side_effect = concurrent.futures.TimeoutError()
    mock_executor = MagicMock()
    mock_executor.submit.return_value = fake_future
    monkeypatch.setattr("lapis_pm.spec_review.ThreadPoolExecutor", lambda *a, **k: mock_executor)

    brief = run_spec_review(
        advisory_spec, council_voicing="gravitywell", timeout_s=30, dispatch_facets=True,
        with_gw=True,
    )

    assert brief is not None, "a --with-gw timeout must still render a brief, never withhold it"
    assert brief.gw_abandoned is True
    assert brief.grounding_status == "failed"
    assert brief.grounding_reason == "gw_timeout"
    assert brief.elapsed_gw >= 0.0


# ---------------------------------------------------------------------------
# DoD 8 / DoD 9: lock boundary
# ---------------------------------------------------------------------------

def test_lock_released_before_reference_poll(advisory_spec, monkeypatch, _isolated_lock):
    """U3d: the lock must be released before _poll_reference_until_terminal runs —
    a concurrent, independent acquire attempt from another thread must succeed
    while the (stubbed) poll is in flight."""
    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", lambda r: _happy_envelope())

    contended = threading.Event()
    lock_free_during_poll = {"value": None}

    def fake_poll(**kwargs):
        # Try a second, independent, non-blocking flock on the same lock file.
        fd = os.open(str(_isolated_lock), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_free_during_poll["value"] = True
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            lock_free_during_poll["value"] = False
        finally:
            os.close(fd)
        contended.set()
        return None

    monkeypatch.setattr("lapis_pm.spec_review._poll_reference_until_terminal", fake_poll)

    run_spec_review(advisory_spec, council_voicing="gravitywell", timeout_s=30, dispatch_facets=True)

    assert contended.is_set()
    assert lock_free_during_poll["value"] is True, (
        "the lock must be released before _poll_reference_until_terminal runs"
    )


def test_lock_held_across_gw_join(advisory_spec, monkeypatch, _isolated_lock):
    """U3d: the lock must still be held while the GW future is being joined."""
    
    monkeypatch.setenv("GW_REVIEWER_TIMEOUT_SEC", "30")
    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", lambda r: _happy_envelope())
    monkeypatch.setattr(
        "lapis_pm.node_identity.resolve_node_identity",
        lambda force=False: MagicMock(node_role="master"),
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **k: MagicMock(returncode=0))

    lock_state_during_join = {"value": None}

    class FakeFuture:
        def result(self, timeout=None):
            fd = os.open(str(_isolated_lock), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_state_during_join["value"] = "free"
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                lock_state_during_join["value"] = "held"
            finally:
                os.close(fd)
            return ('{"verdict": "clean", "issues": [], "confidence": 0.9}', [], 1.0, "")

        def cancel(self):
            return False

    mock_executor = MagicMock()
    mock_executor.submit.return_value = FakeFuture()
    monkeypatch.setattr("lapis_pm.spec_review.ThreadPoolExecutor", lambda *a, **k: mock_executor)

    run_spec_review(
        advisory_spec, council_voicing="gravitywell", timeout_s=30, dispatch_facets=True,
        with_gw=True,
    )

    assert lock_state_during_join["value"] == "held", (
        "the lock must still be held while the GW future is being joined"
    )


def test_lock_held_across_facets_and_council_default_path(advisory_spec, monkeypatch, _isolated_lock):
    """DoD 9: on the default path (no with_gw, no reference opt-in), the lock is
    still held across Facets+Council — only the (now-skipped) polling phase moves
    outside it."""
    
    lock_state_during_deliberation = {"value": None}

    def fake_run_deliberation(request):
        fd = os.open(str(_isolated_lock), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_state_during_deliberation["value"] = "free"
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            lock_state_during_deliberation["value"] = "held"
        finally:
            os.close(fd)
        return _happy_envelope()

    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", fake_run_deliberation)

    run_spec_review(advisory_spec, council_voicing="gravitywell", timeout_s=30, dispatch_facets=True)

    assert lock_state_during_deliberation["value"] == "held", (
        "the lock must still be held across Facets+Council on the default path"
    )


# ---------------------------------------------------------------------------
# lapis-pm-bundle-gate-facets-local-default-v0 (2026-09-16 Claude-gone re-point)
#
# The run_spec_review signature default was the ONE remaining paid default in
# the lapis-pm surface (facets_operator="haiku" -> claude-haiku-4-5 -> `claude
# -p` on an expired Max OAuth -> call_claude_cli returned None), and it is the
# default the cr-bundle autodispatch gate inherits (bundle_autodispatch._run_gate
# does not pass facets_operator explicitly). It faulted on every nightly run
# since 2026-08-26. All four facets-operator surfaces now agree on
# "gravitywell": this signature, the SpecReviewBrief dataclass, the _build_brief
# param, and the manual CLI --facets-operator default.
# ---------------------------------------------------------------------------

def test_run_spec_review_facets_operator_default_is_gravitywell():
    """DoD test (a): the signature default is the local seat, not a paid tier."""
    default = inspect.signature(run_spec_review).parameters["facets_operator"].default
    assert default == "gravitywell"


def test_bundle_gate_path_resolves_to_gravitywell(advisory_spec, monkeypatch):
    """DoD test (b): the bundle path resolves to "gravitywell" at the
    run_spec_review boundary. Spy the DeliberationRequest construction when
    run_spec_review is invoked the way bundle_autodispatch._run_gate invokes it
    (no explicit facets_operator — it inherits the signature default)."""
    captured = []

    def fake_run_deliberation(request):
        captured.append(request)
        return _happy_envelope()

    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", fake_run_deliberation)
    monkeypatch.setattr(
        "lapis_pm.spec_review.swarm_serving", lambda: True
    )

    # Exactly the kwargs bundle_autodispatch._run_gate passes (bundle_autodispatch.py:
    # _run_gate) — no facets_operator, so the default under test is what flows.
    brief = run_spec_review(
        advisory_spec,
        council_voicing="gravitywell",
        timeout_s=1800,
        authority="advisory",
        dispatch_facets=True,
        sonnet_reviewer=False,
        invoked_by="bundle_autodispatch",
    )

    assert len(captured) == 1, "the deliberation leg must have run exactly once"
    assert captured[0].facets_operator == "gravitywell", (
        "the bundle path must resolve the facets operator to the local seat, "
        "never a paid tier"
    )
    assert brief.facets_operator == "gravitywell"


def test_bundle_gate_path_gw_down_defers_not_degrades(advisory_spec, monkeypatch):
    """Change item 4: the bundle path DEFERS when GW is not serving — it never
    launches the facets leg that would degrade to the dead haiku tier. The
    defer site is bundle_autodispatch's GW-liveness pre-check (it fires before
    _run_gate, which is pinned by TestGWUnreachable in
    tests/test_bundle_autodispatch.py). This test pins the gate-side half of the
    contract: with swarm_serving()==False and council_voicing="gravitywell"
    (the bundle path's voicing), run_spec_review skips run_deliberation
    entirely — no doomed deliberation, no paid fallback."""
    run_deliberation_called = []

    async def mock_run_deliberation(request):
        run_deliberation_called.append(request)
        return _happy_envelope()

    monkeypatch.setattr("lapis_pm.spec_review.run_deliberation", mock_run_deliberation)
    monkeypatch.setattr("lapis_pm.spec_review.swarm_serving", lambda: False)

    brief = run_spec_review(
        advisory_spec,
        council_voicing="gravitywell",
        timeout_s=30,
        dispatch_facets=True,
    )

    assert run_deliberation_called == [], (
        "GW-down must skip the deliberation leg entirely (defer), never launch a "
        "doomed deliberation that could fall back to a paid tier"
    )
    assert brief.council_status == "error"
    assert brief.council_error_reason == "gw_not_serving"
    assert brief.facets_deliberation is None
