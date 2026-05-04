"""Unit tests for the novelty filter (_score_novelty) in digest.py.

Three cases per spec:
1. test_novelty_filter_flags_overlap     — overlap > 0.30 → parroted_likely=True
2. test_novelty_filter_passes_clean      — overlap ≤ 0.30 → parroted_likely=False
3. test_novelty_threshold_boundary       — overlap exactly 0.30 → parroted_likely=False (strict >)
"""
from __future__ import annotations

from lapis_pm.scout.digest import NOVELTY_PARROT_THRESHOLD, _score_novelty, _tokenize
from lapis_pm.scout.schema import BreakModeAggregate


def _make_cluster(signature: str) -> BreakModeAggregate:
    return BreakModeAggregate(signature=signature, frequency=0.5)


# ---------------------------------------------------------------------------
# Test 1: overlap exceeds threshold → parroted_likely=True
# ---------------------------------------------------------------------------

def test_novelty_filter_flags_overlap() -> None:
    """Scaffold context contains tokens matching the cluster signature.

    Context tokens: {"starhouse", "segv", "ram", "marginal"}
    Signature: "RAM-marginal SEGV under sustained load"
    Signature tokens: {"ram", "marginal", "segv", "under", "sustained", "load"}
    Intersection: {"ram", "marginal", "segv"} = 3
    Union: {"starhouse", "ram", "marginal", "segv", "under", "sustained", "load"} = 7
    Jaccard = 3/7 ≈ 0.428 > 0.30 → parroted_likely=True
    """
    context_token_union = frozenset({"starhouse", "segv", "ram", "marginal"})
    cluster = _make_cluster("RAM-marginal SEGV under sustained load")

    _score_novelty([cluster], context_token_union)

    assert cluster.parroted_likely is True, (
        f"Expected parroted_likely=True for high-overlap signature; "
        f"sig_tokens={_tokenize(cluster.signature)}, context={context_token_union}"
    )


# ---------------------------------------------------------------------------
# Test 2: generic signature, minimal overlap → parroted_likely=False
# ---------------------------------------------------------------------------

def test_novelty_filter_passes_clean() -> None:
    """Architecture-sketch context has only generic tokens; signature is novel.

    Context tokens derived from something like:
      "dispatcher receives entity events classifies routes handler emits"
    Signature: "reviewer cycle exhaustion when both agents bounce changes"
    These share few/no tokens → Jaccard ≤ 0.30 → parroted_likely=False
    """
    context_token_union = _tokenize(
        "dispatcher receives entity events classifies routes handler emits acknowledgement"
    )
    cluster = _make_cluster("reviewer cycle exhaustion when both agents bounce changes")

    _score_novelty([cluster], context_token_union)

    assert cluster.parroted_likely is False, (
        f"Expected parroted_likely=False for clean signature; "
        f"sig_tokens={_tokenize(cluster.signature)}, context={context_token_union}"
    )


# ---------------------------------------------------------------------------
# Test 3: boundary — Jaccard exactly 0.30 → parroted_likely=False (strict >)
# ---------------------------------------------------------------------------

def test_novelty_threshold_boundary() -> None:
    """Engineer Jaccard overlap = exactly 0.30 and verify strict > threshold.

    Construction for Jaccard = 3/10 = 0.30:
    |A & B| = 3, |A | B| = 10
    Simplest: sig_tokens has no tokens outside the union (|A - B| = 0, |B - A| = 7)
    -> sig_tokens = {alpha, beta, gamma}    (3 tokens, all shared)
    -> ctx_tokens = {alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa}
    Intersection = 3, Union = 10, Jaccard = 3/10 = 0.30 exactly.
    """
    assert NOVELTY_PARROT_THRESHOLD == 0.30, "Threshold must be 0.30 for this boundary test"

    sig_tokens = frozenset({"alpha", "beta", "gamma"})
    ctx_tokens = frozenset({"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"})
    assert len(sig_tokens & ctx_tokens) == 3
    assert len(sig_tokens | ctx_tokens) == 10
    jaccard = len(sig_tokens & ctx_tokens) / len(sig_tokens | ctx_tokens)
    assert jaccard == 0.30, f"Test setup error: Jaccard={jaccard}, expected 0.30"

    # Build a cluster whose _tokenize(signature) yields exactly sig_tokens.
    # Use a signature where the three words tokenize to the expected set.
    cluster = _make_cluster("alpha beta gamma")
    # Verify tokenization matches our intent (no stop-word stripping issues)
    assert _tokenize(cluster.signature) == sig_tokens, (
        f"Tokenization mismatch: {_tokenize(cluster.signature)} != {sig_tokens}"
    )

    _score_novelty([cluster], ctx_tokens)

    assert cluster.parroted_likely is False, (
        f"Expected parroted_likely=False at exactly Jaccard=0.30 (strict >); "
        f"got parroted_likely={cluster.parroted_likely}"
    )
