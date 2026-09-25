"""Tests for lapis-pm-panel-leg-survival-v0 (the five changes, rev 4).

Coverage (spec DoD #1-#6, #8; rev 4 amendment supersedes the rev-3
thinking-disable pins):
  #1 payload shape pinned (rev 4): witness + node1 bodies carry
     max_tokens=65536 and carry NO chat_template_kwargs (thinking stays
     ON); the timeout constants are 300s (witness), 120s (node1),
     120s (node2).
  #2 node2 re-point pinned: new _NODE2_URL/_NODE2_MODEL constants; node2
     prompt carries no vault: lines while node1's does; score() leaves the
     passed substrates list unmutated.
  #3 missing PHALA_API_KEY fails closed (node2 substrate_unavailable, note
     names the literal), node1 still ok on the same pass, no exception;
     sentinel: a fake key value never leaks into notes/error/claim.
  #4 (existing tests unchanged) fail-closed semantics preserved.
  #5 Leg 3 all branches (a-f): persistent / intermittent / simultaneous
     blip / short-window / pre-annotation recompute / single-cycle void
     framing; no "verdict was false" phrasing anywhere.
  #6 status line: `panel: K/N starved (legs: ...)` renders with stored
     starved verdicts and renders nothing new without verdicts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import pm_core
from lapis_pm.corroboration_adapter import (
    LapisPMReviewerAdapter,
    _IdentifierSubstrate,
    _NODE2_MODEL,
    _NODE2_TIMEOUT,
    _MAX_TOKENS,
    _LLM_TIMEOUT,
    run_corroboration_pass,
)
from lapis_pm.local_reviewer_witness import run_local_reviewer_witness

FAKE_DIFF = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n+x = 1\n"
_CLAUDE_CLEAN = {"verdict": "clean", "issues": [], "confidence": 0.95}

VALID_CORR_RESP = {
    "model": "test-model",
    "choices": [{
        "message": {"content": json.dumps({
            "verdict": "clean",
            "claims": [{"identifier": "tick", "drift_class": "none", "notes": "ok"}],
            "summary": "all clear",
        })},
        "finish_reason": "stop",
    }],
}

VALID_WITNESS_RESP = {
    "model": "test-model",
    "choices": [{
        "message": {"content": json.dumps({
            "verdict": "clean", "issues": [], "confidence": 0.9,
        })},
        "finish_reason": "stop",
    }],
}


# ---------------------------------------------------------------------------
# DoD #1 (rev 4): payload shape pinned — thinking ON, budget raised
# ---------------------------------------------------------------------------

class TestThinkingOnPayload:

    def test_witness_body_carries_65536_and_no_thinking_field(self):
        """The witness request body carries max_tokens=65536 and NO
        chat_template_kwargs (rev 4: thinking stays ON), with temperature
        unchanged from baseline."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = VALID_WITNESS_RESP
        resp.raise_for_status = MagicMock()

        captured: list[dict] = []

        def capture_post(url, *, json=None, timeout=None, **kw):
            captured.append(json)
            return resp

        with (
            patch("lapis_pm.local_reviewer_witness.node_reachable", return_value=True),
            patch("httpx.post", side_effect=capture_post),
        ):
            result = run_local_reviewer_witness(
                diff_text=FAKE_DIFF, repo="lapis-pm", pr_number=7,
                spec_summary="spec", claude_verdict=_CLAUDE_CLEAN,
            )

        assert result.agreement != "local_failed"
        assert captured, "no POST captured"
        body = captured[0]
        assert body["max_tokens"] == 65536
        assert "chat_template_kwargs" not in body
        assert body["temperature"] == 0.1

    def test_node1_body_carries_65536_and_no_thinking_field(self):
        """The node1 corroboration body (shared _build_body) carries
        max_tokens=65536 and NO chat_template_kwargs (rev 4: thinking stays
        ON), with temperature unchanged."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        captured: list[dict] = []

        def capture_post(url, *, json=None, timeout=None, **kw):
            captured.append(json)
            return MagicMock(status_code=200, json=lambda: VALID_CORR_RESP, raise_for_status=lambda: None)

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=capture_post),
        ):
            result = adapter.score(FAKE_DIFF, substrates, "lapis-pm")

        assert result.leg_status == "ok"
        assert captured, "no POST captured"
        body = captured[0]
        assert body["max_tokens"] == _MAX_TOKENS == 65536
        assert "chat_template_kwargs" not in body
        assert body["temperature"] == 0.1

    def test_timeouts_sized_by_measurement(self):
        """Rev 4: witness read timeout 300s (arm F1 = 137.7s measured),
        node1 _LLM_TIMEOUT 120s (arm C = 9.5s measured), _NODE2_TIMEOUT
        stays 120s (arm E = 15.3s measured)."""
        assert _LLM_TIMEOUT == 120
        assert _NODE2_TIMEOUT == 120
        from lapis_pm.local_reviewer_witness import _PROBE_TIMEOUT as _w_probe
        assert _w_probe == 3
        import inspect
        sig = inspect.signature(run_local_reviewer_witness)
        assert sig.parameters["timeout"].default == 300


# ---------------------------------------------------------------------------
# DoD #2: node2 re-point pinned — Phala TEE, code-only prompt, no mutation
# ---------------------------------------------------------------------------

class TestNode2Repoint:
    """D2 (lapis-pm-reviewer-leg-repair-v0): node2 rides the sanctioned
    PhalaTeeClient verifying path (attestation + ACI verify hop on every
    call), not a raw Bearer POST. The raw POST + hardcoded _NODE2_URL are
    gone; the four named error classes (node2_unavailable /
    node2_aci_unverified / node2_cap_refused / node2_client_import_error)
    surface on the DEGRADED PANEL line and the pm/review-state mem summary.
    The code-only tenancy constraint (no vault lines in the node2 prompt) is
    preserved."""

    def test_node2_model_constant_is_phala(self):
        assert _NODE2_MODEL == "deepseek/deepseek-v4-flash-0731"

    def test_node2_prompt_excludes_vault_and_node1_keeps_it(self):
        """A substrate with a vault hit: the node2 prompt carries no
        `vault:` lines, the node1 prompt does, and score() leaves the passed
        substrates list unmutated (a node1 re-run still sees the vault
        lines)."""
        adapter = LapisPMReviewerAdapter()
        substrates = [_IdentifierSubstrate(
            "tick",
            repo_hits=[{"file": "f", "line": "1", "text": "def tick"}],
            vault_hits=[{"file": "vault/notes.md", "line": "9", "text": "vault secret line"}],
        )]

        def capture_posts(captured):
            def _post(url, *, json=None, timeout=None, **kw):
                captured.append(json["messages"][0]["content"])
                return MagicMock(status_code=200, json=lambda: VALID_CORR_RESP, raise_for_status=lambda: None)
            return _post

        # node1 (default include_vault=True) — vault lines present
        n1_prompts: list[str] = []
        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=capture_posts(n1_prompts)),
        ):
            r1 = adapter.score(FAKE_DIFF, substrates, "lapis-pm")
        assert r1.leg_status == "ok"
        assert any("vault:" in p for p in n1_prompts), "node1 prompt must carry vault lines"

        # node2 (include_vault=False, via_node2_client=True) — no vault
        # lines. The sanctioned client reads PHALA_API_KEY from the env at
        # call time, so the test supplies one.
        n2_prompts: list[str] = []

        def _fake_chat_completion(*, messages, model, nonce=None, extra_body=None):
            n2_prompts.append(messages[0]["content"])
            return VALID_CORR_RESP

        with (
            patch("lapis_pm.corroboration_adapter._build_node2_client") as mock_client,
            patch.dict("os.environ", {"PHALA_API_KEY": "phala-test-key"}),
        ):
            mock_client.return_value.chat_completion.side_effect = _fake_chat_completion
            r2 = adapter.score(
                FAKE_DIFF, substrates, "lapis-pm",
                node_model=_NODE2_MODEL,
                node_timeout=_NODE2_TIMEOUT, include_vault=False,
                via_node2_client=True,
            )
        assert r2.leg_status == "ok"
        assert n2_prompts, "no node2 prompt captured"
        assert not any("vault:" in p for p in n2_prompts), "node2 prompt must be code-only"

        # No mutation: the substrates list still carries the vault hit, so a
        # node1 re-run still sees the vault lines.
        assert substrates[0].vault_hits == [
            {"file": "vault/notes.md", "line": "9", "text": "vault secret line"}
        ]
        n1_again: list[str] = []
        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=capture_posts(n1_again)),
        ):
            r3 = adapter.score(FAKE_DIFF, substrates, "lapis-pm")
        assert r3.leg_status == "ok"
        assert any("vault:" in p for p in n1_again), "node1 re-run must still see vault lines"

    def test_run_corroboration_pass_node2_is_code_only_and_client_gated(self):
        """End-to-end: with the key set, node2's captured prompt is code-only
        while node1's carries the vault lines; node2 rides the sanctioned
        PhalaTeeClient (not a raw Bearer POST)."""
        substrates = [_IdentifierSubstrate(
            "tick",
            repo_hits=[{"file": "f", "line": "1", "text": "def tick"}],
            vault_hits=[{"file": "vault/notes.md", "line": "9", "text": "vault secret line"}],
        )]

        n1_prompts: list[str] = []
        n2_prompts: list[str] = []

        def _post(url, *, json=None, timeout=None, **kw):
            n1_prompts.append(json["messages"][0]["content"])
            return MagicMock(status_code=200, json=lambda: VALID_CORR_RESP, raise_for_status=lambda: None)

        def _fake_chat_completion(*, messages, model, nonce=None, extra_body=None):
            n2_prompts.append(messages[0]["content"])
            return VALID_CORR_RESP

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=_post),
            patch("lapis_pm.corroboration_adapter._build_node2_client") as mock_client,
            # These D2-era tests pin the LEGACY node2 contract (the
            # sanctioned PhalaTeeClient path), which the gate-lanes PR
            # scopes to the blind/non-solo lane shape (S1 caller contract).
            # Stub the seam to that shape so the assertions hold
            # host-independently; the flashnext-solo lane shape is pinned by
            # tests/test_corroboration_node2_flashnext_lane.py.
            patch("lapis_pm.corroboration_adapter._flashnext_lane_url",
                  return_value=(None, None, None)),
            patch.dict("os.environ", {"PHALA_API_KEY": "phala-test-key"}),
        ):
            mock_client.return_value.chat_completion.side_effect = _fake_chat_completion
            result = run_corroboration_pass(FAKE_DIFF, "lapis-pm")

        assert result["cross_node_divergence"] == "agree"
        assert result["leg_status"] == "ok"
        assert result["node2_corroboration"]["leg_status"] == "ok"
        assert n2_prompts and not any("vault:" in p for p in n2_prompts)
        assert n1_prompts and any("vault:" in p for p in n1_prompts)
        # node2 rode the sanctioned client (not a raw Bearer POST).
        mock_client.assert_called_once()


# ---------------------------------------------------------------------------
# DoD #3: missing key fails closed, local legs untouched; key-value sentinel
# ---------------------------------------------------------------------------

class TestPhalaKeyFailClosed:

    def test_missing_key_node2_fails_closed_node1_ok(self):
        """With PHALA_API_KEY unset: the sanctioned client raises (no key ->
        no auth), node2 returns leg_status='substrate_unavailable' with a
        named error class; node1 on the same pass is still 'ok'; no
        exception escapes."""
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        class _NoKeyError(Exception):
            pass

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", return_value=MagicMock(
                status_code=200, json=lambda: VALID_CORR_RESP, raise_for_status=lambda: None,
            )),
            patch("lapis_pm.corroboration_adapter._build_node2_client") as mock_client,
            # Legacy (blind-lane) shape, host-independent (see the
            # TestNode2Repoint sibling).
            patch("lapis_pm.corroboration_adapter._flashnext_lane_url",
                  return_value=(None, None, None)),
            patch.dict("os.environ", {}, clear=False),
        ):
            mock_client.return_value.chat_completion.side_effect = _NoKeyError("no key")
            # Ensure the key is absent for this test.
            import os as _os
            _os.environ.pop("PHALA_API_KEY", None)
            result = run_corroboration_pass(FAKE_DIFF, "lapis-pm")

        assert result["leg_status"] == "ok", "node1 (local leg) must stay healthy"
        n2 = result["node2_corroboration"]
        assert n2["leg_status"] == "substrate_unavailable"
        # Named error class (node2_unavailable for a genuine transport/key
        # failure — not masked as a generic unavailability).
        assert "node2_" in (n2["notes"] or "")
        assert result["cross_node_divergence"] == "node2_unavailable"

    def test_missing_key_no_raw_post_attempted(self):
        """No half-call: with the key absent, node2 never reaches httpx.post
        (it rides the sanctioned client, not a raw Bearer POST)."""
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]
        posted_urls: list[str] = []

        def _post(url, *a, **kw):
            posted_urls.append(str(url))
            return MagicMock(status_code=200, json=lambda: VALID_CORR_RESP, raise_for_status=lambda: None)

        class _NoKeyError(Exception):
            pass

        import os as _os
        _os.environ.pop("PHALA_API_KEY", None)
        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("httpx.post", side_effect=_post),
            patch("lapis_pm.corroboration_adapter._build_node2_client") as mock_client,
        ):
            mock_client.return_value.chat_completion.side_effect = _NoKeyError("no key")
            adapter = LapisPMReviewerAdapter()
            result = adapter.score(
                FAKE_DIFF, substrates, "lapis-pm",
                node_model=_NODE2_MODEL,
                node_timeout=_NODE2_TIMEOUT, include_vault=False,
                via_node2_client=True,
            )
        assert result.leg_status == "substrate_unavailable"
        assert "node2_" in (result.notes or "")
        # No raw POST to Phala (the client path, not a raw Bearer POST).
        assert posted_urls == [], "no raw httpx POST on the node2 path"

    def test_key_value_never_appears_in_result_strings(self):
        """Sentinel (extended, D2): a fake key value forced through a failing
        call appears in no notes/error/claim string of the result (the env
        var name may be named; the value must never leak). The note echoes
        only the exception CLASS NAME, never the message, for every named
        client exception class — so a future exception string carrying a key
        cannot leak through this path."""
        fake_key = "phala-test-key"
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]

        class _KeyLeakingError(Exception):
            def __str__(self):
                return f"upstream rejected Bearer {fake_key}"

        with (
            patch("lapis_pm.corroboration_adapter.node_reachable", return_value=True),
            patch("lapis_pm.corroboration_adapter._build_node2_client") as mock_client,
            patch.dict("os.environ", {"PHALA_API_KEY": fake_key}),
        ):
            mock_client.return_value.chat_completion.side_effect = _KeyLeakingError()
            adapter = LapisPMReviewerAdapter()
            result = adapter.score(
                FAKE_DIFF, substrates, "lapis-pm",
                node_model=_NODE2_MODEL,
                node_timeout=_NODE2_TIMEOUT, include_vault=False,
                via_node2_client=True,
            )

        assert result.leg_status == "substrate_unavailable"
        for field in ("notes", "claim"):
            value = getattr(result, field)
            if value:
                assert fake_key not in value, f"key value leaked into {field}: {value}"
        d = result.to_dict()
        for field in ("notes", "claim"):
            if d.get(field):
                assert fake_key not in d[field]


# ---------------------------------------------------------------------------
# D2 (lapis-pm-reviewer-leg-repair-v0): four named node2 error classes
# ---------------------------------------------------------------------------

class TestNode2ErrorClasses:
    """DoD-1: a genuine node2 unavailability surfaces a NAMED error class
    (probe / ACI / cap-refused / client-import) on a PM surface — the
    DEGRADED PANEL line and the pm/review-state mem summary carry the
    error-class tail of node2_corroboration.notes. The raw Bearer POST and
    hardcoded URL are gone; the 400/422 grammar-degrade retry is exercised."""

    def test_node2_error_class_constants(self):
        """The four named error classes exist and are distinct."""
        from lapis_pm.corroboration_adapter import (
            NODE2_ERROR_UNAVAILABLE, NODE2_ERROR_ACI_UNVERIFIED,
            NODE2_ERROR_CAP_REFUSED, NODE2_ERROR_CLIENT_IMPORT,
            NODE2_ERROR_CLASSES,
        )
        assert NODE2_ERROR_UNAVAILABLE == "node2_unavailable"
        assert NODE2_ERROR_ACI_UNVERIFIED == "node2_aci_unverified"
        assert NODE2_ERROR_CAP_REFUSED == "node2_cap_refused"
        assert NODE2_ERROR_CLIENT_IMPORT == "node2_client_import_error"
        assert len(NODE2_ERROR_CLASSES) == 4
        assert len(set(NODE2_ERROR_CLASSES)) == 4  # all distinct

    def test_node2_error_class_mapping(self):
        """_node2_error_class maps the named client exceptions to the four
        classes: spend-cap -> node2_cap_refused, ACI -> node2_aci_unverified,
        a genuine transport failure -> node2_unavailable."""
        from lapis_pm.corroboration_adapter import _node2_error_class
        import agents_core.phala_tee as pt

        # Spend cap -> node2_cap_refused.
        cap_exc = pt.PhalaSpendCapExceededError(2.0, 2.0, 0)
        assert _node2_error_class(cap_exc) == "node2_cap_refused"

        # ACI verify hop failure -> node2_aci_unverified.
        aci_exc = pt.ReportVerificationError.__new__(pt.ReportVerificationError)
        assert _node2_error_class(aci_exc) == "node2_aci_unverified"

        # A genuine transport failure (ConnectionError) -> node2_unavailable.
        assert _node2_error_class(ConnectionError("refused")) == "node2_unavailable"

    def test_node2_cap_refused_note(self):
        """A spend-cap refusal surfaces `node2_cap_refused` in the note (a
        cost decision, not an outage) — distinct from a genuine TEE outage."""
        substrates = [_IdentifierSubstrate("tick", [], [])]
        import agents_core.phala_tee as pt

        with (
            patch("lapis_pm.corroboration_adapter._build_node2_client") as mock_client,
        ):
            mock_client.return_value.chat_completion.side_effect = (
                pt.PhalaSpendCapExceededError(2.0, 2.0, 0)
            )
            adapter = LapisPMReviewerAdapter()
            result = adapter.score(
                FAKE_DIFF, substrates, "lapis-pm",
                node_model=_NODE2_MODEL, node_timeout=_NODE2_TIMEOUT,
                include_vault=False, via_node2_client=True,
            )
        assert result.leg_status == "substrate_unavailable"
        assert "node2_cap_refused" in (result.notes or "")

    def test_node2_client_import_error_note(self):
        """A cross-repo PhalaTeeClient import failure surfaces
        `node2_client_import_error` (a WIRING defect, not a TEE outage) —
        the gate-4 amendment."""
        substrates = [_IdentifierSubstrate("tick", [], [])]
        with (
            patch("lapis_pm.corroboration_adapter._build_node2_client",
                  side_effect=ImportError("agents_core.phala_tee not found")),
        ):
            adapter = LapisPMReviewerAdapter()
            result = adapter.score(
                FAKE_DIFF, substrates, "lapis-pm",
                node_model=_NODE2_MODEL, node_timeout=_NODE2_TIMEOUT,
                include_vault=False, via_node2_client=True,
            )
        assert result.leg_status == "substrate_unavailable"
        assert "node2_client_import_error" in (result.notes or "")

    def test_node2_grammar_degrade_retry(self):
        """DoD-1: the 400/422 grammar-degrade retry is exercised — a
        grammar-rejecting hop (400) is retried ONCE without response_format,
        and the retry's success produces a healthy leg (not a 100% fail)."""
        substrates = [_IdentifierSubstrate("tick", [{"file": "f", "line": "1", "text": "def tick"}], [])]
        call_count = [0]

        def _chat_completion(*, messages, model, nonce=None, extra_body=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # First attempt (grammar-carrying) is rejected with a 400.
                raise Exception("400 Client Error: grammar not supported")
            # Retry (no grammar) succeeds.
            return VALID_CORR_RESP

        with (
            patch("lapis_pm.corroboration_adapter._build_node2_client") as mock_client,
        ):
            mock_client.return_value.chat_completion.side_effect = _chat_completion
            adapter = LapisPMReviewerAdapter()
            result = adapter.score(
                FAKE_DIFF, substrates, "lapis-pm",
                node_model=_NODE2_MODEL, node_timeout=_NODE2_TIMEOUT,
                include_vault=False, via_node2_client=True,
            )
        # Two calls: the grammar-carrying attempt + the grammar-degrade retry.
        assert call_count[0] == 2, "expected a grammar-degrade retry after the 400"
        assert result.leg_status == "ok"
        assert result.verdict == "clean"

    def test_node2_grammar_degrade_retry_both_fail(self):
        """DoD-1: if the grammar-degrade retry ALSO fails, the leg fails
        closed with a named error class (no double retry)."""
        substrates = [_IdentifierSubstrate("tick", [], [])]
        call_count = [0]

        def _chat_completion(*, messages, model, nonce=None, extra_body=None):
            call_count[0] += 1
            raise Exception("400 Client Error: grammar not supported")

        with (
            patch("lapis_pm.corroboration_adapter._build_node2_client") as mock_client,
        ):
            mock_client.return_value.chat_completion.side_effect = _chat_completion
            adapter = LapisPMReviewerAdapter()
            result = adapter.score(
                FAKE_DIFF, substrates, "lapis-pm",
                node_model=_NODE2_MODEL, node_timeout=_NODE2_TIMEOUT,
                include_vault=False, via_node2_client=True,
            )
        # Two calls max (no double retry): the grammar-carrying attempt + the
        # grammar-degrade retry.
        assert call_count[0] == 2
        assert result.leg_status == "substrate_unavailable"
        assert "node2_" in (result.notes or "")

    def test_node2_payload_carries_all_three_fields(self):
        """DoD-1: the node2 payload (extra_body) carries ALL THREE of
        temperature 0.1 / max_tokens 65536 / the json_schema grammar (the
        client merges verbatim)."""
        substrates = [_IdentifierSubstrate("tick", [], [])]
        captured_extra: list[dict] = []

        def _chat_completion(*, messages, model, nonce=None, extra_body=None):
            captured_extra.append(extra_body or {})
            return VALID_CORR_RESP

        with (
            patch("lapis_pm.corroboration_adapter._build_node2_client") as mock_client,
        ):
            mock_client.return_value.chat_completion.side_effect = _chat_completion
            adapter = LapisPMReviewerAdapter()
            result = adapter.score(
                FAKE_DIFF, substrates, "lapis-pm",
                node_model=_NODE2_MODEL, node_timeout=_NODE2_TIMEOUT,
                include_vault=False, via_node2_client=True,
            )
        assert result.leg_status == "ok"
        assert captured_extra, "no extra_body captured"
        body = captured_extra[0]
        # All three payload fields ride extra_body (the client merges
        # verbatim into the POST body).
        assert body["temperature"] == 0.1
        assert body["max_tokens"] == _MAX_TOKENS == 65536
        assert "response_format" in body  # the json_schema grammar
        # model is NOT in extra_body (the client sets it on the body itself).
        assert "model" not in body


# ---------------------------------------------------------------------------
# DoD #5: Leg 3 all branches (a-f)
# ---------------------------------------------------------------------------

def _starved_verdict(legs=("local_witness", "corroboration", "second_node")):
    # Carries the raw witness/corroboration blocks (all legs down) so the
    # recompute fallback paths agree with the annotation.
    return {
        "verdict": "fixable",
        "issues": [{"severity": "high", "path": "foo.py", "note": "x"}],
        "confidence": 0.0,
        "local_reviewer_witness": {"agreement": "local_failed"},
        "corroboration_result": {
            "verdict": "uncertain",
            "claim": "(substrate unavailable)",
            "leg_status": "substrate_unavailable",
            "cross_node_divergence": "node2_unavailable",
        },
        "panel_starvation": {
            "legs_down": list(legs),
            "starved": True,
            "confidence_raw": 0.9,
        },
    }


def _healthy_verdict():
    # Carries the raw witness/corroboration blocks (all legs up) so the
    # recompute fallback paths agree with the annotation.
    return {
        "verdict": "fixable",
        "issues": [{"severity": "low", "path": "foo.py", "note": "nit"}],
        "confidence": 0.8,
        "local_reviewer_witness": {"agreement": "agree"},
        "corroboration_result": {
            "verdict": "clean",
            "claim": "identifiers checked",
            "leg_status": "ok",
            "cross_node_divergence": "agree",
        },
        "panel_starvation": {"legs_down": [], "starved": False, "confidence_raw": 0.8},
    }


def _second_node_only_verdict():
    """AMENDED 2026-09-10 shape: second_node down but NOT starved (both local
    gate legs up) — the verdict stands on local legs with attenuated
    confidence. Carries the raw witness/corroboration blocks (both local
    legs up) so the recompute fallback paths agree with the annotation."""
    return {
        "verdict": "fixable",
        "issues": [{"severity": "low", "path": "foo.py", "note": "nit"}],
        "confidence": 0.5333,
        "local_reviewer_witness": {"agreement": "agree"},
        "corroboration_result": {
            "verdict": "clean",
            "claim": "identifiers checked",
            "leg_status": "ok",
            "cross_node_divergence": "node2_unavailable",
        },
        "panel_starvation": {"legs_down": ["second_node"], "starved": False,
                             "confidence_raw": 0.8},
    }


class _Comment:
    def __init__(self, ts, tags, content):
        self.ts = ts
        self.tags = tags
        self.content = content


def _verdict_comment(ts, pr, cycle, verdict_dict):
    return _Comment(
        ts,
        [f"pm:reviewer:pr={pr}:cycle={cycle}:verdict={verdict_dict.get('verdict', 'fixable')}"],
        "Reviewer verdict for PR #%d:\n%s" % (pr, json.dumps(verdict_dict)),
    )


def _run_escalation(comments, rec=None, pr=7, cycle=3, current=None):
    """Drive _escalate_noop_retry_if_degraded with a fake episodic source."""
    rec = rec or {"gpu_id": "task-1", "agent_type": "fixer_retry", "pr_number": pr, "cycle": cycle}
    current = current if current is not None else _starved_verdict()

    def fake_for_cycle(target_id, pr_number, cyc):
        for c in comments:
            for t in c.tags:
                if t == f"pm:reviewer:pr={pr_number}:cycle={cyc}:verdict=fixable":
                    return json.loads(c.content.split("\n", 1)[-1].strip())
        return None

    with (
        patch("lapis_pm.pm_core._review_verdict_for_cycle", side_effect=fake_for_cycle),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
            patch("lapis_pm.pm_core._mem") as mock_mem,
        patch("lapis_pm.pm_core.episodic.write_hold") as mock_hold,
        patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
        patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set_outstanding,
    ):
        mock_mem.return_value.get.return_value = None
        mock_brief = MagicMock()
        mock_brief.comment_id = "cid-test"
        mock_synth.return_value = mock_brief

        action = pm_core._escalate_noop_retry_if_degraded("tid", rec, pr)

    hold_text = mock_hold.call_args.kwargs["content"] if mock_hold.call_args else ""
    return action, hold_text, mock_synth, mock_set_outstanding


class TestLeg3DegradedPredicate:

    def test_a_three_starved_same_leg_persistent(self):
        """(a) three starved verdicts, same leg in >= 2 of them -> degraded
        brief labeled 'persistent panel degradation', naming the starved
        count, the cycle numbers, and the union of legs_down; suspend-trust /
        do-not-gate; NOT 'verdict was false'."""
        comments = [
            _verdict_comment("2026-09-01T10:00:00+00:00", 7, 1, _starved_verdict(("local_witness",))),
            _verdict_comment("2026-09-01T11:00:00+00:00", 7, 2, _starved_verdict(("local_witness", "corroboration"))),
            _verdict_comment("2026-09-01T12:00:00+00:00", 7, 3, _starved_verdict(("local_witness",))),
        ]
        action, hold_text, mock_synth, mock_set = _run_escalation(comments, cycle=3)
        assert action is not None and "noop_retry_degraded_verdict_brief" in action
        assert "persistent panel degradation" in hold_text
        assert "3 of last 3" in hold_text
        assert "cycles 1-3" in hold_text
        assert "local_witness" in hold_text
        assert "do not gate on it" in hold_text
        assert "suspend trust" in hold_text.lower()
        assert "verdict was false" not in hold_text
        assert "verdict was false" not in mock_synth.call_args[1]["trigger"]
        assert "verdict was false" not in mock_synth.call_args[1]["query"]
        mock_set.assert_called_once()

    def test_b_two_starved_disjoint_separated_intermittent(self):
        """(b) two starved verdicts with disjoint legs, temporally separated
        (beyond the blip window) -> 'intermittent panel instability'."""
        comments = [
            _verdict_comment("2026-09-01T10:00:00+00:00", 7, 1, _starved_verdict(("local_witness",))),
            _verdict_comment("2026-09-01T12:00:00+00:00", 7, 2, _starved_verdict(("second_node",))),
        ]
        action, hold_text, _, _ = _run_escalation(comments, cycle=2)
        assert action is not None
        assert "intermittent panel instability" in hold_text
        assert "2 of last 2" in hold_text
        assert "cycles 1-2" in hold_text
        assert "verdict was false" not in hold_text

    def test_c_two_starved_disjoint_within_blip_simultaneous(self):
        """(c) two starved verdicts with disjoint legs within
        PANEL_BLIP_WINDOW_SECONDS -> 'transient panel degradation
        (simultaneous blip)' with the retry suggestion."""
        comments = [
            _verdict_comment("2026-09-01T10:00:00+00:00", 7, 1, _starved_verdict(("local_witness",))),
            _verdict_comment("2026-09-01T10:00:30+00:00", 7, 2, _starved_verdict(("second_node",))),
        ]
        action, hold_text, _, _ = _run_escalation(comments, cycle=2)
        assert action is not None
        assert "transient panel degradation (simultaneous blip)" in hold_text
        assert "retry" in hold_text
        assert "verdict was false" not in hold_text

    def test_d_two_cycle_window_same_leg_persistent(self):
        """(d) a two-cycle window (window < 3) with both starved on the same
        leg -> degraded / persistent."""
        comments = [
            _verdict_comment("2026-09-01T10:00:00+00:00", 7, 1, _starved_verdict(("corroboration",))),
            _verdict_comment("2026-09-01T11:00:00+00:00", 7, 2, _starved_verdict(("corroboration",))),
        ]
        action, hold_text, _, _ = _run_escalation(comments, cycle=2)
        assert action is not None
        assert "persistent panel degradation" in hold_text
        assert "2 of last 2" in hold_text

    def test_e_pre_annotation_recompute_overlaps(self):
        """(e) a pre-annotation verdict (no `panel_starvation` block; raw
        blocks show the same leg down) + one annotated starved verdict on the
        same leg -> the starved_legs() recompute path treats them as
        overlapping -> persistent."""
        legacy = {
            "verdict": "fixable",
            "issues": [{"severity": "high", "path": "foo.py", "note": "x"}],
            "confidence": 0.9,
            # Raw blocks from before the annotation existed: local_witness down.
            "local_reviewer_witness": {"agreement": "local_failed"},
            "corroboration_result": {
                "verdict": "clean",
                "claim": "ok",
                "leg_status": "ok",
                "cross_node_divergence": "agree",
            },
        }
        comments = [
            _verdict_comment("2026-09-01T10:00:00+00:00", 7, 1, legacy),
            _verdict_comment("2026-09-01T11:00:00+00:00", 7, 2, _starved_verdict(("local_witness",))),
        ]
        action, hold_text, _, _ = _run_escalation(comments, cycle=2)
        assert action is not None
        assert "persistent panel degradation" in hold_text
        assert "local_witness" in hold_text
        assert "2 of last 2" in hold_text

    def test_f_single_cycle_starved_void_framing(self):
        """(f) single-cycle starved (one starved verdict, none before) -> the
        reworded void-framing single-cycle brief ('uncorroborated - do not
        gate on it'), NOT the old 'verdict was false' text, and no degraded
        label."""
        comments = [
            _verdict_comment("2026-09-01T12:00:00+00:00", 7, 3, _starved_verdict()),
        ]
        action, hold_text, _, _ = _run_escalation(comments, cycle=3)
        assert action is not None
        assert "uncorroborated" in hold_text
        assert "do not gate on it" in hold_text
        assert "stands as advisory" in hold_text
        assert "persistent panel degradation" not in hold_text
        assert "intermittent panel instability" not in hold_text
        assert "simultaneous blip" not in hold_text
        assert "verdict was false" not in hold_text

    def test_window_truncates_to_last_three(self):
        """The window is the last min(3, available): 4 starved verdicts where
        only the last 2 share a leg must not read persistent from the
        truncated-out third one."""
        comments = [
            _verdict_comment("2026-09-01T08:00:00+00:00", 7, 1, _starved_verdict(("local_witness",))),
            _verdict_comment("2026-09-01T09:00:00+00:00", 7, 2, _starved_verdict(("second_node",))),
            _verdict_comment("2026-09-01T10:00:00+00:00", 7, 3, _starved_verdict(("second_node",))),
            _verdict_comment("2026-09-01T11:00:00+00:00", 7, 4, _starved_verdict(("corroboration",))),
        ]
        action, hold_text, _, _ = _run_escalation(comments, cycle=4)
        assert action is not None
        # Window = cycles 2,3,4: second_node in 2 of them -> persistent.
        assert "persistent panel degradation" in hold_text
        assert "3 of last 3" in hold_text
        assert "cycles 2-4" in hold_text


# ---------------------------------------------------------------------------
# DoD #5 (Leg 2 branch): refuted-only verdicts still escalate, void-framed
# ---------------------------------------------------------------------------

class TestVoidFraming:

    def test_refuted_only_branch_still_escalates_without_falsehood(self):
        """A refuted-only verdict (not starved) still escalates a no-op
        retry, worded VOID not 'verdict was false' — no falsehood phrasing
        anywhere (Council reclassification: the claim is about the process,
        never 'the verdict is wrong')."""
        refuted = {
            "verdict": "fixable",
            "issues": [{"severity": "high", "path": "foo.py", "note": "x"}],
            "confidence": 0.9,
            "refuted_absence_findings": ["_check_equal is not defined anywhere"],
            "panel_starvation": {"legs_down": [], "starved": False, "confidence_raw": 0.9},
        }
        comments = [
            _verdict_comment("2026-09-01T12:00:00+00:00", 7, 3, refuted),
        ]

        def fake_for_cycle(target_id, pr_number, cyc):
            for c in comments:
                for t in c.tags:
                    if t == f"pm:reviewer:pr={pr_number}:cycle={cyc}:verdict=fixable":
                        return json.loads(c.content.split("\n", 1)[-1].strip())
            return None

        with (
            patch("lapis_pm.pm_core._review_verdict_for_cycle", side_effect=fake_for_cycle),
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
            patch("lapis_pm.pm_core._mem") as mock_mem,
            patch("lapis_pm.pm_core.episodic.write_hold") as mock_hold,
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
            patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set_outstanding,
        ):
            mock_mem.return_value.get.return_value = None
            mock_brief = MagicMock()
            mock_brief.comment_id = "cid-test"
            mock_synth.return_value = mock_brief
            action = pm_core._escalate_noop_retry_if_degraded("tid", {
                "gpu_id": "task-1", "agent_type": "fixer_retry", "pr_number": 7, "cycle": 3,
            }, 7)

        assert action is not None and "noop_retry_degraded_verdict_brief" in action
        hold_text = mock_hold.call_args.kwargs["content"] if mock_hold.call_args else ""
        assert hold_text, "the refuted-only branch must still write a hold"
        assert "verdict was false" not in hold_text
        assert "verdict was false" not in mock_synth.call_args[1]["trigger"]
        assert "verdict was false" not in mock_synth.call_args[1]["query"]
        mock_set_outstanding.assert_called_once()


# ---------------------------------------------------------------------------
# AMENDED 2026-09-10 (lapis-pm-reviewer-single-leg-local-v0, D2 consumer
# census site 2/3): the escalation-partition test. A no-op fixer_retry
# against a saved second_node-only NOT-starved verdict is HEALTHY-partition:
# it reaches _maybe_dispatch_auditor_noop (D6b) and does NOT raise the
# _escalate_noop_retry_if_degraded HIGH brief. Both sides are asserted
# explicitly so a future 'fix' that reverts the predicate change (a silent
# Phala outage looking like a regression) fails the suite instead of
# landing.
# ---------------------------------------------------------------------------

class TestEscalationPartitionSecondNodeOnly:

    _rec = {"gpu_id": "task-1", "agent_type": "fixer_retry", "pr_number": 7,
            "cycle": 1, "repo": "lapis-pm", "head_sha": "abc123"}

    def _second_node_only_verdict(self):
        return {
            "verdict": "fixable",
            "issues": [{"severity": "low", "path": "foo.py", "note": "nit"}],
            "confidence": 0.5667,
            "panel_starvation": {"legs_down": ["second_node"], "starved": False,
                                 "confidence_raw": 0.85},
        }

    def test_noop_retry_second_node_only_reaches_auditor_noop(self):
        """(a) healthy partition: the D6b no-op auditor dispatch fires for a
        second_node-only NOT-starved verdict (a local-legs verdict's no-op
        is auditable, not a panel-degradation episode)."""
        verdict = self._second_node_only_verdict()
        with (
            patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=verdict),
            patch("lapis_pm.pm_core._mem") as mock_mem,
            patch("lapis_pm.pm_core._audit_dispatched_head", return_value=None),
            patch("lapis_pm.pm_core._has_pending_auditor_for_pr", return_value=False),
            patch("lapis_pm.pm_core._act_dispatch_auditor",
                  return_value="action:auditor_dispatched:pr=7:mode=noop"),
            patch("lapis_pm.pm_core._repo_owner", return_value=("lapis-pm", "Erah")),
            patch("lapis_pm.pm_core.get_open_prs", return_value=[]),
        ):
            mock_mem.return_value.get.return_value = None
            action = pm_core._maybe_dispatch_auditor_noop("tid", self._rec, 7)
        assert action is not None
        assert "auditor_dispatched" in action

    def test_noop_retry_second_node_only_does_not_raise_high_escalation(self):
        """(b) the _escalate_noop_retry_if_degraded HIGH brief is ABSENT for
        the same second_node-only NOT-starved verdict (a persistent Phala
        outage is no longer loud on THAT surface — by design, per the
        2026-09-10 ruling; it is loud on the brief loud line + the mem
        review-state node2_error_class + the status legs union)."""
        verdict = self._second_node_only_verdict()
        with (
            patch("lapis_pm.pm_core._review_verdict_for_cycle", return_value=verdict),
            patch("lapis_pm.pm_core._mem") as mock_mem,
            patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]),
            patch("lapis_pm.pm_core.episodic.write_hold") as mock_hold,
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
            patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set_outstanding,
        ):
            mock_mem.return_value.get.return_value = None
            action = pm_core._escalate_noop_retry_if_degraded("tid", self._rec, 7)
        assert action is None
        mock_hold.assert_not_called()
        mock_synth.assert_not_called()
        mock_set_outstanding.assert_not_called()


# ---------------------------------------------------------------------------
# DoD #6: status panel-health line
# ---------------------------------------------------------------------------

class TestPanelHealthLine:

    def test_summary_counts_starved_and_legs_union(self):
        # AMENDED 2026-09-10: the starved count counts starved-only verdicts
        # (a GATE leg down), while the legs union spans ALL verdicts'
        # legs_down — the second_node-only (not-starved) verdict still shows
        # up in the union.
        comments = [
            _verdict_comment("2026-09-01T09:00:00+00:00", 7, 1, _healthy_verdict()),
            _verdict_comment("2026-09-01T10:00:00+00:00", 7, 2, _starved_verdict(("local_witness",))),
            _verdict_comment("2026-09-01T11:00:00+00:00", 7, 3, _second_node_only_verdict()),
        ]
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments):
            s = pm_core._panel_health_summary("tid", 7)
        assert s == {"starved": 1, "total": 3, "legs": ["local_witness", "second_node"]}

    def test_summary_second_node_only_stretch_zero_starved_legs_visible(self):
        """AMENDED 2026-09-10: a sustained Phala outage (second_node-only
        verdicts, never starved) renders `panel: 0/N starved (legs:
        second_node)` — absent-but-visible on the at-a-glance surface."""
        comments = [
            _verdict_comment("2026-09-01T10:00:00+00:00", 7, 1, _second_node_only_verdict()),
            _verdict_comment("2026-09-01T11:00:00+00:00", 7, 2, _second_node_only_verdict()),
            _verdict_comment("2026-09-01T12:00:00+00:00", 7, 3, _healthy_verdict()),
        ]
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments):
            s = pm_core._panel_health_summary("tid", 7)
        assert s == {"starved": 0, "total": 3, "legs": ["second_node"]}

    def test_summary_window_is_last_five(self):
        # 6 verdicts: the display window is the last 5 (total==5, the first
        # verdict truncated out), while the starved count and legs union cover
        # the full stored history — 4 of 6 are starved, 2 healthy.
        # 6 verdicts: the display window is the last 5 (total==5, the first
        # verdict truncated out). AMENDED 2026-09-10: the starved count is
        # starved-only (GATE legs down) — 3 of 6; the legs union spans ALL
        # verdicts, so the second_node-only (not-starved) verdict's leg still
        # shows.
        comments = []
        for i, legs in enumerate([("local_witness",), None, ("second_node",), None, ("corroboration",), ("local_witness",)], start=1):
            v = (_second_node_only_verdict() if legs == ("second_node",)
                 else _starved_verdict(legs) if legs else _healthy_verdict())
            comments.append(_verdict_comment(f"2026-09-01T{10 + i:02d}:00:00+00:00", 7, i, v))
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments):
            s = pm_core._panel_health_summary("tid", 7)
        assert s["total"] == 5
        assert s["starved"] == 3
        assert s["legs"] == ["corroboration", "local_witness", "second_node"]

    def test_summary_none_when_no_verdicts(self):
        with patch("lapis_pm.pm_core.episodic.all_comments", return_value=[]):
            assert pm_core._panel_health_summary("tid", 7) is None

    def test_cli_renders_panel_line(self, capsys):
        """`lapis-pm status <target>` shows `panel: K/N starved (legs: ...)`
        when the target has stored starved verdicts."""
        from lapis_pm import cli
        from agents_core.targets import TargetStore

        comments = [
            _verdict_comment("2026-09-01T10:00:00+00:00", 7, 1, _starved_verdict(("local_witness",))),
            _verdict_comment("2026-09-01T11:00:00+00:00", 7, 2, _second_node_only_verdict()),
        ]

        class _FakeTarget:
            id = "tid-panel"
            title = "panel target"
            pm_bound = True
            pm_repo = "lapis-pm"
            pm_authority = "advisory"
            paused = False
            paused_reason = None
            data = {}

        fake_pr = {"number": 7, "title": "t", "html_url": "u",
                   "head": {"ref": "b"}, "mergeable": True}

        with (
            patch.object(TargetStore, "get", return_value=_FakeTarget()),
            patch("lapis_pm.pm_core.get_cursor", return_value=""),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.cli.episodic.all_comments", return_value=[]),
            patch("agents_core.forgejo.get_open_prs", return_value=[fake_pr]),
            patch("lapis_pm.pm_core._classified_pr_ids", return_value=set()),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=2),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._last_review_verdict",
                  return_value={"verdict": "fixable", "issues": [1]}),
            patch("lapis_pm.pm_core._active_reviewer_backoff", return_value=None),
            patch("lapis_pm.pm_core._panel_health_summary",
                  return_value={"starved": 1, "total": 2,
                                "legs": ["local_witness", "second_node"]}),
        ):
            args = MagicMock()
            args.target_id = "tid-panel"
            args.explain = False
            rc = cli.cmd_status(args)

        out = capsys.readouterr().out
        assert rc == 0
        # AMENDED 2026-09-10: the union carries the second_node-only
        # (not-starved) verdict's leg too.
        assert "panel:         1/2 starved (legs: local_witness, second_node)" in out

    def test_cli_no_panel_line_without_verdicts(self, capsys):
        """A target with no verdicts shows nothing new."""
        from lapis_pm import cli
        from agents_core.targets import TargetStore

        class _FakeTarget:
            id = "tid-none"
            title = "no verdicts"
            pm_bound = True
            pm_repo = "lapis-pm"
            pm_authority = "advisory"
            paused = False
            paused_reason = None
            data = {}

        fake_pr = {"number": 7, "title": "t", "html_url": "u",
                   "head": {"ref": "b"}, "mergeable": True}

        with (
            patch.object(TargetStore, "get", return_value=_FakeTarget()),
            patch("lapis_pm.pm_core.get_cursor", return_value=""),
            patch("lapis_pm.pm_core.load_dispatched", return_value=[]),
            patch("lapis_pm.pm_core.get_outstanding_brief", return_value=None),
            patch("lapis_pm.cli.episodic.all_comments", return_value=[]),
            patch("agents_core.forgejo.get_open_prs", return_value=[fake_pr]),
            patch("lapis_pm.pm_core._classified_pr_ids", return_value=set()),
            patch("lapis_pm.pm_core._reviewer_cycle_count", return_value=1),
            patch("lapis_pm.pm_core._has_pending_reviewer_for_pr", return_value=False),
            patch("lapis_pm.pm_core._last_review_verdict",
                  return_value={"verdict": "fixable", "issues": [1]}),
            patch("lapis_pm.pm_core._active_reviewer_backoff", return_value=None),
            patch("lapis_pm.pm_core._panel_health_summary", return_value=None),
            ):
            args = MagicMock()
            args.target_id = "tid-none"
            args.explain = False
            rc = cli.cmd_status(args)

        out = capsys.readouterr().out
        assert rc == 0
        assert "panel:" not in out


# ---------------------------------------------------------------------------
# DoD-1/5 (lapis-pm-reviewer-leg-repair-v0): the brief carries drift_class
# verbatim + the node2 error-class tail on the DEGRADED PANEL line + the
# positive "panel: 3/3 legs reporting" line when not starved.
# ---------------------------------------------------------------------------

class TestBriefSurfaces:
    """DoD-1/5: the brief composition (reviewer_verdict_text) carries
    drift_class verbatim, the node2 error-class tail on the DEGRADED PANEL
    line, and a positive "panel: 3/3 legs reporting" line when not starved.
    The trigger gating is stated: the line fires for advisory-clean/hold
    triggers (the reviewer_verdict_text block)."""

    def _render_reviewer_verdict_text(self, verdict_info: dict) -> str:
        """Drive the reviewer_verdict_text composition in _act_brief with a
        fake verdict. Returns the composed text (or '' if the block did not
        render)."""
        from lapis_pm import pm_core
        from lapis_pm import authority

        cls = authority.PRClassification(
            verdict="advisory",
            screen_verdict="clean",
            static_outcome="advisory",
            reasons=[],
            issues=[],
            pr_number=7,
            repo="lapis-pm",
            title="t",
            html_url="u",
            changed_paths=[],
            diff_loc=10,
            diff="diff",
        )

        class _FakeTarget:
            id = "tid-brief"
            pm_repo = "lapis-pm"
            data = {"pm_verification": "pm-live-test"}

        with (
            patch("lapis_pm.pm_core._last_review_verdict", return_value=verdict_info),
            patch("lapis_pm.pm_core.TargetStore") as mock_ts,
            patch("lapis_pm.pm_core._attestation_hook"),
            patch("lapis_pm.pm_core._precedent_hook", return_value=None),
            patch("lapis_pm.pm_core._mark_pr_classified"),
            patch("lapis_pm.auto_resolve.should_auto_resolve",
                  return_value=(False, None)),
            patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
            patch("lapis_pm.pm_core._set_brief_outstanding"),
            ):
            mock_ts.return_value.get.return_value = _FakeTarget()
            mock_synth.return_value = MagicMock(comment_id="cid")
            try:
                pm_core._act_brief(
                    "tid-brief", "advisory-clean", False,
                    {"classification": cls, "pr": {}},
                )
            except Exception:
                pass
            # reviewer_verdict_text is passed to brief.synthesize.
            if mock_synth.call_args:
                return mock_synth.call_args.kwargs.get("reviewer_verdict_text") or ""
        return ""

    def test_brief_carries_drift_class_verbatim(self):
        """DoD-5: the brief carries drift_class verbatim (the rev-1 brief
        rendered only corroboration={verdict})."""
        verdict_info = {
            "verdict": "clean",
            "issues": [],
            "confidence": 0.9,
            "corroboration_result": {
                "verdict": "clean",
                "drift_class": "new",
                "node2_corroboration": {"leg_status": "ok", "notes": "ok"},
            },
            "panel_starvation": {"legs_down": [], "starved": False,
                                 "confidence_raw": 0.9},
        }
        text = self._render_reviewer_verdict_text(verdict_info)
        assert "drift_class=new" in text, f"drift_class not verbatim: {text}"
        # Not starved -> the positive line renders.
        assert "panel: 3/3 legs reporting" in text

    def test_brief_degraded_panel_carries_node2_error_class(self):
        """DoD-1: the DEGRADED PANEL line carries the node2 error-class tail
        (a genuine node2 down is distinguishable from a wiring bug on a PM
        surface). AMENDED 2026-09-10: the advisory DEGRADED wording applies
        to a genuinely starved verdict (a local gate leg down) — the fixture
        is re-pointed at that shape; a second_node-only-down verdict renders
        the loud line instead (see the tests below)."""
        verdict_info = {
            "verdict": "fixable",
            "issues": [{"severity": "high", "path": "foo.py", "note": "x"}],
            "confidence": 0.0,
            "corroboration_result": {
                "verdict": "uncertain",
                "drift_class": None,
                "node2_corroboration": {
                    "leg_status": "substrate_unavailable",
                    "notes": "node2_aci_unverified: ReportVerificationError — node2 failed closed",
                },
            },
            "panel_starvation": {"legs_down": ["local_witness", "second_node"],
                                 "starved": True, "confidence_raw": 0.9},
        }
        text = self._render_reviewer_verdict_text(verdict_info)
        assert "DEGRADED PANEL" in text
        assert "node2=node2_aci_unverified" in text, (
            f"node2 error-class tail missing: {text}"
        )

    def test_brief_client_import_error_distinguishable(self):
        """DoD-1: a cross-repo import failure (node2_client_import_error) is
        distinguishable from a genuine TEE outage on the DEGRADED PANEL
        line (a wiring defect, not a TEE outage)."""
        verdict_info = {
            "verdict": "fixable",
            "issues": [{"severity": "high", "path": "foo.py", "note": "x"}],
            "confidence": 0.0,
            "corroboration_result": {
                "verdict": "uncertain",
                "drift_class": None,
                "node2_corroboration": {
                    "leg_status": "substrate_unavailable",
                    "notes": "node2_client_import_error: ImportError — node2 wiring defect",
                },
            },
            "panel_starvation": {"legs_down": ["corroboration", "second_node"],
                                 "starved": True, "confidence_raw": 0.9},
        }
        text = self._render_reviewer_verdict_text(verdict_info)
        assert "DEGRADED PANEL" in text
        assert "node2=node2_client_import_error" in text

    def test_brief_positive_line_when_not_starved(self):
        """DoD-1: a positive "panel: 3/3 legs reporting" line renders when
        not starved (a healthy panel is distinguishable from a legacy
        unannotated verdict)."""
        verdict_info = {
            "verdict": "clean",
            "issues": [],
            "confidence": 0.95,
            "corroboration_result": {
                "verdict": "clean",
                "drift_class": None,
                "node2_corroboration": {"leg_status": "ok", "notes": "ok"},
            },
            "panel_starvation": {"legs_down": [], "starved": False,
                                 "confidence_raw": 0.95},
        }
        text = self._render_reviewer_verdict_text(verdict_info)
        assert "panel: 3/3 legs reporting" in text
        assert "DEGRADED PANEL" not in text

    def test_brief_second_node_only_loud_line(self):
        """AMENDED 2026-09-10 (D3 brief-surface): a second_node-only-down,
        NOT-starved verdict renders the loud line with the error class +
        raw/attenuated values, does NOT render "3/3 legs reporting", and
        does NOT render the advisory DEGRADED line."""
        verdict_info = {
            "verdict": "fixable",
            "issues": [{"severity": "low", "path": "foo.py", "note": "nit"}],
            "confidence": 0.5667,
            "corroboration_result": {
                "verdict": "clean",
                "drift_class": None,
                "node2_corroboration": {
                    "leg_status": "substrate_unavailable",
                    "notes": "node2_unavailable: ConnectionError — Phala TEE unreachable",
                },
            },
            "panel_starvation": {"legs_down": ["second_node"], "starved": False,
                                 "confidence_raw": 0.85},
        }
        text = self._render_reviewer_verdict_text(verdict_info)
        assert (
            "second_node (Phala TEE) ABSENT (node2_unavailable); raw "
            "confidence 0.85 attenuated to 0.5667 — verdict stands on "
            "local legs only"
        ) in text
        assert "panel: 3/3 legs reporting" not in text
        assert "DEGRADED PANEL" not in text

    def test_brief_second_node_only_loud_line_unknown_fallback(self):
        """AMENDED 2026-09-10 (D3, many-eyes M3): when no allowlisted
        NODE2_ERROR_CLASSES constant matches the node2 notes, the loud line
        renders the literal token `unknown` — never empty parens, never
        free-form notes text."""
        verdict_info = {
            "verdict": "fixable",
            "issues": [{"severity": "low", "path": "foo.py", "note": "nit"}],
            "confidence": 0.5333,
            "corroboration_result": {
                "verdict": "clean",
                "drift_class": None,
                "node2_corroboration": {
                    "leg_status": "substrate_unavailable",
                    "notes": "something else entirely happened",
                },
            },
            "panel_starvation": {"legs_down": ["second_node"], "starved": False,
                                 "confidence_raw": 0.8},
        }
        text = self._render_reviewer_verdict_text(verdict_info)
        assert (
            "second_node (Phala TEE) ABSENT (unknown); raw "
            "confidence 0.8 attenuated to 0.5333 — verdict stands on "
            "local legs only"
        ) in text
        assert "panel: 3/3 legs reporting" not in text
        assert "DEGRADED PANEL" not in text
        # The notes text itself must never leak into the render.
        assert "something else entirely happened" not in text
