"""Lapis-PM reviewer corroboration adapter.

LapisPMReviewerAdapter implements the SubstrateAdapter protocol shape
from the cross-node-corroboration-v0 spec.

- retrieve: extracts doc-mentioned-identifier claims from a PR diff;
  greps repo to check existence; also greps vault when available.
  Identifiers the diff itself introduces (present in the diff's added
  lines) are marked `diff_introduced` so the score pass classifies them
  `new`, never `missing_referent` (lapis-pm-reviewer-leg-repair-v0, D1).
- score: single constrained-JSON LLM call against the retrieved substrate;
  returns CorroborationResult with drift_class in:
  {"missing_referent", "stale_referent", "renamed_referent", "new", "none"}.
  The node2 (Phala TEE) leg runs through the sanctioned
  agents_core.phala_tee.PhalaTeeClient verifying path (attestation + ACI
  verify hop on every call) — no raw Bearer POST (lapis-pm-reviewer-leg-
  repair-v0, D2 / I2).

Invariants (from spec §Invariants):
- Read-only: never mutates reviewer verdict mainline fields.
- Flag-shaped, not block-shaped: findings are advisory at v0.
- Gracefully degrades to verdict="uncertain" on substrate-unavailable or
  LLM-unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from archetypes_core.corroboration import Citation
from lapis_pm.completion_text import NO_USABLE_TEXT, extract_completion_text
from lapis_pm.node_probe import node_reachable


# ---------------------------------------------------------------------------
# Data shapes (CorroborationResult is evidence-packet-shaped from day 1)
# ---------------------------------------------------------------------------

@dataclass
class CorroborationResult:
    """Evidence-packet result from a corroboration pass.

    Shape is stable from v0; downstream consumers (claude-view Atmosphere
    panel, mem.db evidence keys, future federation) depend on this shape.
    """
    verdict: Literal["clean", "flagged", "uncertain"]
    claim: str
    citations: list[Citation]
    freshness_stamp: str       # ISO timestamp (UTC)
    scope_id: str              # e.g. "repo:lapis-pm"
    drift_class: str | None = None   # "missing_referent" | "stale_referent" | "renamed_referent" | "new" | "none"
    notes: str | None = None
    primitive_decomposition: dict | None = None  # entry-time decomposition; compost-routing handle (spec Compost invariant)
    # Structured leg health, independent of `claim`/`notes` prose. "claim" can be
    # empty on a healthy pass (LLM omitted `summary`) and non-empty on a dead leg
    # (failure-marker prose) — leg_status is the field consumers must read instead
    # of string-matching either one. See lapis-pm-corroboration-producer-leg-status-v1.
    # "truncated" (lapis-pm-corroboration-thinking-parse-and-truncation-loudness-v0):
    # the call reached the substrate (HTTP 200) but finish_reason=="length" cut the
    # model off — a distinct condition from "substrate_unavailable" (never reached
    # the substrate at all). Both are down legs for panel_starvation purposes.
    leg_status: Literal["ok", "substrate_unavailable", "pass_failed", "truncated"] = "ok"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "claim": self.claim,
            "citations": [{"source_id": c.source_id, "excerpt": c.excerpt} for c in self.citations],
            "freshness_stamp": self.freshness_stamp,
            "scope_id": self.scope_id,
            "drift_class": self.drift_class,
            "notes": self.notes,
            "primitive_decomposition": self.primitive_decomposition,
            "leg_status": self.leg_status,
        }


# ---------------------------------------------------------------------------
# Identifier extraction from diff text
# ---------------------------------------------------------------------------

_IDENTIFIER_PATTERNS = [
    # Python / JS function and class definitions on added/context lines
    r'\bdef\s+([A-Za-z_][A-Za-z0-9_]{2,})\b',
    r'\bclass\s+([A-Za-z_][A-Za-z0-9_]{2,})\b',
    r'\bfunc\s+([A-Za-z_][A-Za-z0-9_]{2,})\b',
    # Backtick-quoted identifiers in prose / commit messages
    r'`([A-Za-z_][A-Za-z0-9_.]{2,})`',
    # Dotted module paths (e.g. lapis_pm.corroboration_adapter)
    r'\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]+){1,3})\b',
]

_MAX_IDENTIFIERS = 15  # cap to bound LLM prompt size


def _extract_identifiers(diff_text: str) -> list[str]:
    """Extract doc-mentioned identifiers from a PR diff.

    Pulls:
    - Filenames from diff headers (--- a/... / +++ b/...)
    - Function/class names from code lines
    - Backtick-quoted identifiers from prose lines
    - Dotted module paths

    Returns deduplicated list, longest-first (module paths before segments).
    Capped at _MAX_IDENTIFIERS to bound downstream processing.
    """
    identifiers: set[str] = set()

    # File paths from unified diff headers
    for m in re.finditer(r'^(?:---|\+\+\+)\s+[ab]/(.+)$', diff_text, re.MULTILINE):
        path = m.group(1).strip()
        if path and path != '/dev/null':
            # Include both full path and basename without extension
            identifiers.add(path)
            stem = Path(path).stem
            if len(stem) >= 3:
                identifiers.add(stem)

    # Named identifiers from code and prose
    for pattern in _IDENTIFIER_PATTERNS:
        for m in re.finditer(pattern, diff_text):
            name = m.group(1).strip()
            # Skip very short names and private/dunder identifiers
            if len(name) >= 3 and not name.startswith('__'):
                identifiers.add(name)

    # Sort longest-first so module paths (most specific) come before segments
    return sorted(identifiers, key=len, reverse=True)[:_MAX_IDENTIFIERS]


# ---------------------------------------------------------------------------
# Substrate retrieval helpers
# ---------------------------------------------------------------------------

_VAULT_PATH = Path("/srv/git/inertia-vault-working")
_GREP_TIMEOUT = 8  # seconds; grep over large trees can be slow


def _grep_repo(identifier: str, repo_path: str) -> list[dict]:
    """Grep repo for identifier. Returns list of {file, line, text} dicts."""
    try:
        proc = subprocess.run(
            [
                "grep", "-rn",
                "--include=*.py", "--include=*.md", "--include=*.yaml",
                "-l",            # list files first to avoid huge output
                "--", identifier, repo_path,
            ],
            capture_output=True, text=True, timeout=_GREP_TIMEOUT,
        )
        # Get actual lines for the first few matching files
        matching_files = proc.stdout.splitlines()[:3]
        results = []
        for fpath in matching_files:
            try:
                line_proc = subprocess.run(
                    ["grep", "-n", "-m", "2", "--", identifier, fpath],
                    capture_output=True, text=True, timeout=5,
                )
                for line in line_proc.stdout.splitlines():
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        results.append({
                            "file": fpath,
                            "line": parts[0],
                            "text": parts[1][:200],
                        })
            except (subprocess.TimeoutExpired, OSError):
                pass
        return results[:6]  # cap total hits
    except (subprocess.TimeoutExpired, OSError):
        return []


def _added_lines(diff_text: str) -> str:
    """Return the diff's added lines (a unified-diff `+` line, excluding the
    `+++ b/` file header) joined as one string.

    D1 (lapis-pm-reviewer-leg-repair-v0): the fallback classifier for
    diff-introduced identifiers. `_extract_identifiers` is pure regex and
    text-agnostic, so it can be run over this projection — `+++ b/` headers
    still yield new-file paths, `+def foo` / `+class Bar` yield new symbols.
    """
    lines = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
    return "\n".join(lines)


def _vault_grep(identifier: str) -> list[dict]:
    """Grep vault for identifier. Returns [] if vault not present."""
    if not _VAULT_PATH.exists():
        return []
    try:
        proc = subprocess.run(
            [
                "grep", "-rn", "--include=*.md",
                "-l", "--", identifier, str(_VAULT_PATH),
            ],
            capture_output=True, text=True, timeout=_GREP_TIMEOUT,
        )
        matching_files = proc.stdout.splitlines()[:2]
        results = []
        for fpath in matching_files:
            try:
                lp = subprocess.run(
                    ["grep", "-n", "-m", "1", "--", identifier, fpath],
                    capture_output=True, text=True, timeout=5,
                )
                for line in lp.stdout.splitlines():
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        results.append({
                            "file": fpath,
                            "line": parts[0],
                            "text": parts[1][:200],
                        })
            except (subprocess.TimeoutExpired, OSError):
                pass
        return results[:4]
    except (subprocess.TimeoutExpired, OSError):
        return []


@dataclass
class _IdentifierSubstrate:
    identifier: str
    repo_hits: list[dict]
    vault_hits: list[dict]
    # D1 (lapis-pm-reviewer-leg-repair-v0): True when this identifier appears
    # in the diff's ADDED lines. Such an identifier is, by construction,
    # present at the PR head, so an absent-in-base-tree grep is NOT a
    # missing referent — it is a `new` (diff-introduced) identifier. The
    # base-tree grep still runs first and keeps its citations; this flag only
    # reclassifies what the grep misses (the +lines fallback classifier).
    diff_introduced: bool = False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

def _llm_url() -> str:
    """Resolve the qwen-operator local-LLM endpoint.

    Read at call time (not module load) so tests can monkeypatch.setenv - same
    call-time-vs-import-time discipline as agents_core.llm._llamacpp_url().
    Defaults to GravityWell; LOCAL_LLM_URL wins when set. StarHouse (the old
    bare-literal default) kernel-panicked 2026-08-03 and is being held off
    deliberately - see agents-core-local-llm-gw-repoint-v0.
    """
    return os.environ.get("LOCAL_LLM_URL", "http://203.0.113.11:8081/v1/chat/completions")


def _flashnext_lane_url(
    fetcher: "Callable[[], dict] | None" = None,
) -> tuple[str | None, str | None, str | None]:
    """S6 (gate-lanes-registry-driven-flashnext-v0): resolve the flashnext
    node2 lane through the gw-seats registry.

    Returns (base_url, served_model, blocked_reason):

    * (base_url, served_model, None) — the registry is readable, the 27B
      is NOT serving (flashnext-solo, the spec's S6 trigger: "reality_view
      says the 27B is down and a flashnext row is registered"), and the
      flashnext seat (:30000) is serving. The node2 leg builds against that
      lane (registry-served model name on :30000) so corroboration does not
      silently attenuate panels under flashnext-solo.
    * (None, None, None) — a LEGACY-shape case: the registry is blind
      (unreachable/malformed, the S1 contract's ONLY fallback case) OR the
      27B IS serving (dual/slot1 posture — the spec keeps the phala path
      unchanged there; MED-5 fold: the switch is conditioned on
      flashnext-solo, not merely a registered flashnext row, so node2 never
      jumps onto a seat the gate legs run on while the 27B lane is alive).
      The caller rides the sanctioned PhalaTeeClient path byte-identically.
    * (None, None, reason) — readable registry, 27B down, flashnext lane
      absent/not serving: an honest node2_unavailable (the panel_starvation
      row stays truthful) — NEVER a silent fallback to the Phala/legacy
      path that would mask the seat's absence (the S1 caller contract,
      "no lying leg"; MED-4 fold: the code now matches this docstring —
      resolve_gate_lane collapses blind and dead-lane to the same None, so
      the distinction is made by re-reading the payload, the same pattern
      gate_lane_serving uses).

    Single registry read (one _fetch_payload, no double-fetch); reads the
    seat rows directly from the payload via the gate_lane helpers (shim-
    internal: the companion lane_registry promotion keeps the same shape).
    """
    try:
        from lapis_pm import gate_lane as _gate_lane
    except Exception:
        return (None, None, None)  # module unavailable -> today's behavior
    try:
        payload = (fetcher or _gate_lane._fetch_payload)()
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or not _gate_lane._seat_rows(payload):
        return (None, None, None)  # blind — the ONLY legacy-fallback case
    if _gate_lane._resolve_from_payload(payload, _gate_lane.SLOT1_LANE_NAME) is not None:
        return (None, None, None)  # 27B serving: not flashnext-solo; phala unchanged
    lane_obj = _gate_lane._resolve_from_payload(payload, _gate_lane.FLASHNEXT_LANE_NAME)
    if lane_obj is None:
        return (None, None, "flashnext_not_serving")  # readable + dead: honest
    return (lane_obj.base_url, lane_obj.served_model, None)


# _LLM_TIMEOUT 120s (lapis-pm-panel-leg-survival-v0 rev 4, Erah ruling
# 2026-09-01): sized by measurement, not precaution — the runaway-guard
# principle (Erah, 2026-08-12) applied to the new measurement. Arm C measured
# 9.5s thinking ON on a 3.7k-char corroboration-shaped prompt (84% of tokens in
# the reasoning channel); 120s carries >=10x even at ~10x the prompt scale.
# The old 45s guard was sized from a no-thinking arm and tripped under seat
# load on production-sized prompts (the ReadTimeout lines in the verdict
# JSONLs).
_LLM_TIMEOUT = 120  # seconds; measured 9.5s live thinking ON, 120s guard
_PROBE_TIMEOUT = 3   # seconds for connect probe before POST
_CONNECT_TIMEOUT = 5  # seconds connect cap on POST (read budget preserved at _LLM_TIMEOUT)

# A runaway guard, not a size estimate: too-small silently corrupts every call by
# starving a thinking model's reasoning channel before it reaches an answer (the
# root cause this unit repairs), too-large only costs when a model actually reaches
# it. 65,536 (Erah ruling 2026-09-19, tunable via LAPIS_REVIEWER_MAX_TOKENS):
# thinking stays ON; the 27B deliberates longer than the prior 16,384 cap on
# hard reviews — arm F1 (2026-09-01) measured 12,847 reasoning tokens on a
# SIMPLE production shape at the 16,384 budget, barely finishing; a hard
# review (e.g. a 13-file PR) exceeds it -> finish_reason=length -> failed leg.
# 1/4 of the seat's 262k context — still a guard, now sized for real
# deliberation. Shared body: node1 rides it and node2/Phala rides it too (the
# budget is within the TEE substrate's capacity — arm E completed at 125
# tokens). Read at import time (module constant); the daemon is timer-fired
# (fresh process per tick) so a config change + env is picked up each tick.
_MAX_TOKENS = int(os.environ.get("LAPIS_REVIEWER_MAX_TOKENS", "65536"))

# Node 2 — Phala TEE through the SANCTIONED verifying client
# (lapis-pm-reviewer-leg-repair-v0, D2; I2: no direct LLM API calls).
#
# Re-pointed to Phala 2026-09-01 (lapis-pm-panel-leg-survival-v0, D3; tenancy
# ratified by Erah 2026-09-01): the previous substrate — a MacBook Pro (MLX,
# 100.124.203.15:8080, the May 2026 mesh-milestone witness) — sleeps and drops
# off the tailnet when asleep. Phala is a different host, model family and
# engine, always-on. Tenancy basis (code review only): the node2 prompt is
# CODE-ONLY — the diff excerpt + repo grep hits, never vault content
# (`include_vault=False`, pinned by test); local-first is preserved (two of
# three legs stay local).
#
# The 2026-09-07 defect this unit removes: the re-point shipped a hardcoded
# _NODE2_URL + raw Bearer POST that BYPASSED the sanctioned PhalaTeeClient
# verifying path — no attestation fetch, no ACI verify hop, so 100% of node2
# calls failed with HTTPStatusError and the panel ran 2-of-3 legs on every
# verdict. The call now goes through agents_core.phala_tee.PhalaTeeClient.
# chat_completion(): attestation fetch + ACI verify hop (the two-leg `aci`
# CLI) on EVERY call, loopback-refusal policy, $2/day spend cap (fail-closed
# before I/O), one locality-ledger row per call. The daemon service already
# sources /data/agents/config/phala.env (PHALA_BASE_URL=loopback verifying
# hop, ACI_VERIFIER_BIN, PHALA_API_KEY) — no new env, no plaintext key path.
_NODE2_MODEL = "deepseek/deepseek-v4-flash-0731"
# 120s stays a runaway guard, not a size estimate: arm E measured 15.3s on a
# 1,838-char prompt (7.8x margin); the production node2 prompt is code-bounded
# (~15-16k chars worst case: diff[:2500] + 10 identifiers x 3 repo hits, 200
# chars each, vault excluded) — >=4x margin at worst case. Pinned explicitly
# on the client (whose default is 300s): the node2 leg must not out-run the
# tick's budget.
_NODE2_TIMEOUT = 120
# ACI-hop budget (D2): the two-leg `aci` CLI verify runs synchronously on the
# tick path (via _run_corroboration_pass_sync) and is never cached. The
# client's per-leg subprocess timeout is pinned here so a hung verify cannot
# block the tick unboundedly — 30s per leg x 2 legs + attestation fetch stays
# inside the 120s node2 guard.
_NODE2_ACI_LEG_TIMEOUT_S = 30.0

# Named node2 error classes (D2 / DoD-1): a genuine unavailability is one of
# these four, surfaced on the DEGRADED PANEL line and the pm/review-state mem
# summary — never collapsed into a generic "unavailable" (the operational
# boundary stated, not masked).
#   node2_unavailable          — genuine probe/transport failure (the TEE hop
#                                is down / unreachable).
#   node2_aci_unverified       — the attestation failed the ACI verify hop
#                                (toolchain fault or report failure).
#   node2_cap_refused          — the $2/day spend cap refused the call BEFORE
#                                any I/O (a cost decision, not an outage).
#   node2_client_import_error  — the cross-repo PhalaTeeClient import/resolve
#                                failed (a WIRING defect, not a TEE outage).
NODE2_ERROR_UNAVAILABLE = "node2_unavailable"
NODE2_ERROR_ACI_UNVERIFIED = "node2_aci_unverified"
NODE2_ERROR_CAP_REFUSED = "node2_cap_refused"
NODE2_ERROR_CLIENT_IMPORT = "node2_client_import_error"
NODE2_ERROR_CLASSES = (
    NODE2_ERROR_UNAVAILABLE,
    NODE2_ERROR_ACI_UNVERIFIED,
    NODE2_ERROR_CAP_REFUSED,
    NODE2_ERROR_CLIENT_IMPORT,
)


def _phala_api_key() -> str | None:
    """Read the Phala API key at call time (node2 path only).

    Same call-time-vs-import-time discipline as _llm_url/LOCAL_LLM_URL. The
    value lives only in /data/agents/config/phala.env (600, daemon user); this
    function references the env var name, never a key value. Returns None when
    unset — the caller fails the node2 leg closed (substrate_unavailable with
    a note naming PHALA_API_KEY) without touching the local legs.
    """
    return os.environ.get("PHALA_API_KEY")


def _node2_error_class(exc: BaseException) -> str:
    """Map a node2-path exception to one of the four named error classes.

    Best-effort cross-repo typing: agents_core may be unavailable in test
    isolation, so the exception classes are resolved lazily by name match.
    A genuine transport failure (requests.ConnectionError / httpx /
    ConnectionError) is `node2_unavailable`; the ACI verify hop failures
    (AciError subclasses incl. ReportVerificationError) are
    `node2_aci_unverified`; the spend-cap refusal is `node2_cap_refused`.
    Anything else (e.g. a 5xx HTTPStatusError from the hop) stays
    `node2_unavailable` — a genuine unavailability.
    """
    try:
        from agents_core import phala_tee as _pt
        if isinstance(exc, _pt.PhalaSpendCapExceededError):
            return NODE2_ERROR_CAP_REFUSED
        if isinstance(exc, _pt.AciError):
            return NODE2_ERROR_ACI_UNVERIFIED
    except Exception:
        pass
    return NODE2_ERROR_UNAVAILABLE


def _build_node2_client() -> Any:
    """Construct the sanctioned PhalaTeeClient (node2 path only).

    Raises ImportError (or an agents_core resolve error) when the cross-repo
    import fails — the caller maps that to `node2_client_import_error` (a
    wiring defect, NOT a TEE outage). The client reads PHALA_API_KEY /
    PHALA_BASE_URL / ACI_VERIFIER_BIN from the daemon env (phala.env, already
    in the service EnvironmentFile); the per-leg ACI subprocess timeout is
    pinned to _NODE2_ACI_LEG_TIMEOUT_S and the call timeout to
    _NODE2_TIMEOUT.
    """
    from agents_core import phala_tee as _pt

    # Pin the ACI-hop budget: patch the module-level default used by the
    # two-leg `aci` CLI subprocess calls (audit + verify) so a hung verify
    # cannot block the tick past the node2 guard. The wrap is applied ONCE
    # per process (guarded by a sentinel attribute) — re-wrapping on every
    # call would accumulate wrappers on the shared module-level function.
    if hasattr(_pt, "_run_aci_json") and not getattr(
            _pt._run_aci_json, "_lapis_aci_budget_patched", False):
        _orig = _pt._run_aci_json

        def _aci_with_budget(*args, **kwargs):
            kwargs.setdefault("timeout", _NODE2_ACI_LEG_TIMEOUT_S)
            return _orig(*args, **kwargs)

        _aci_with_budget._lapis_aci_budget_patched = True
        _pt._run_aci_json = _aci_with_budget

    return _pt.PhalaTeeClient(timeout=float(_NODE2_TIMEOUT))

# Grammar constraint, copied from local_reviewer_witness.py's REVIEWER_JSON_SCHEMA
# shape — proven on this exact GravityWell seat 2026-08-12 (measured FASTER than no
# grammar: 11.0s vs 15.6s, ~30% less reasoning). Stops the model reasoning freely
# into an unusable response at any budget; more important than the budget raise.
CORROBORATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["clean", "flagged", "uncertain"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string"},
                    "drift_class": {
                        "type": "string",
                        # `new` (lapis-pm-reviewer-leg-repair-v0, D1):
                        # diff-introduced identifiers — benign, never flags.
                        "enum": ["missing_referent", "stale_referent",
                                 "renamed_referent", "new", "none"],
                    },
                    "notes": {"type": "string"},
                },
                "required": ["identifier", "drift_class", "notes"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["verdict", "claims", "summary"],
}


class LapisPMReviewerAdapter:
    """SubstrateAdapter for Lapis-PM reviewer corroboration.

    retrieve: extracts doc-mentioned identifiers from diff; greps repo + vault.
    score: single constrained-JSON LLM call to check claim accuracy.

    Invariant: read-only. Never touches reviewer verdict mainline fields.
    """

    def __init__(self, repo_path: str | None = None):
        self._repo_path = repo_path
        self._last_model: str | None = None
        self._last_prompt_hash: str | None = None
        self._last_score_at: datetime | None = None

    @property
    def scope_id(self) -> str:
        """Stable identifier for this substrate (SubstrateAdapter contract)."""
        return "repo:lapis-pm"

    def retrieve(
        self,
        diff_text: str,
        repo: str,
        repo_path: str | None = None,
    ) -> list[_IdentifierSubstrate]:
        """Extract identifiers from diff and grep repo + vault for each.

        D1 (lapis-pm-reviewer-leg-repair-v0): the base-tree grep runs FIRST
        (existing behavior — citations preserved for identifiers found in
        base). Identifiers that appear in the diff's ADDED lines are marked
        `diff_introduced`: they are present at the PR head by construction,
        so the score prompt classifies them `new` (not `missing_referent`)
        when the base-tree grep misses them. An identifier in neither the
        base tree nor the added lines stays a true `missing_referent`.
        """
        rpath = repo_path or self._repo_path or f"/srv/git/{repo}-working"
        identifiers = _extract_identifiers(diff_text)
        # +lines fallback classifier: which extracted identifiers does the
        # diff itself introduce? (Grep-first: this only reclassifies what the
        # base-tree grep misses — an identifier present in base AND added
        # lines classifies `none` with citations, not `new`.)
        # Grep-first (D1): the base-tree grep runs FIRST (above). The +lines
        # fallback classifier runs over the diff's added lines PLUS the
        # `+++ b/` file headers — the headers yield new-file paths, which the
        # added-lines projection alone drops (a `+++ b/` line is a file
        # header, not an added code line). `_extract_identifiers` is pure
        # regex and text-agnostic, so it picks up both.
        added_lines = _added_lines(diff_text)
        # Reconstruct a diff-shaped projection: the `+++ b/` headers (new-file
        # paths) + the added lines. The `--- a/` headers are dropped (they
        # yield the OLD path, which is present in base and not diff-introduced).
        plus_headers = [
            line for line in diff_text.splitlines()
            if line.startswith("+++ ")
        ]
        projection = "\n".join(plus_headers + added_lines.splitlines())
        added_idents = set(_extract_identifiers(projection)) if projection else set()
        substrates = []
        for ident in identifiers[:_MAX_IDENTIFIERS]:
            substrates.append(_IdentifierSubstrate(
                identifier=ident,
                repo_hits=_grep_repo(ident, rpath),
                vault_hits=_vault_grep(ident),
                diff_introduced=ident in added_idents,
            ))
        return substrates

    def score(
        self,
        diff_text: str,
        substrates: list[_IdentifierSubstrate],
        repo: str,
        *,
        node_url: str | None = None,
        node_model: str | None = None,
        node_timeout: int | None = None,
        include_vault: bool = True,
        via_node2_client: bool = False,
    ) -> CorroborationResult:
        """Single LLM call to check identifier claims against substrate.

        Returns uncertain if no identifiers or LLM unavailable.

        include_vault=False omits vault grep hits from the prompt (the node2
        tenancy constraint — code-only content, never vault docs). This is a
        build-time guard in the prompt loop below: the passed `substrates` list
        is NEVER mutated (it is built once and read by both concurrent scoring
        threads — a mutation would non-deterministically strip vault lines from
        node1's prompt).
        """
        scope_id = f"repo:{repo}"
        now = datetime.now(timezone.utc).isoformat()

        if not substrates:
            return CorroborationResult(
                verdict="uncertain",
                claim="(no identifiers extracted)",
                citations=[],
                freshness_stamp=now,
                scope_id=scope_id,
                drift_class=None,
                notes="No identifiers found in diff",
                leg_status="ok",  # legitimate clean result, not starvation
            )

        # Build substrate summary for the LLM
        substrate_lines: list[str] = []
        for s in substrates[:10]:
            substrate_lines.append(f"### `{s.identifier}`")
            if s.repo_hits:
                for h in s.repo_hits[:3]:
                    substrate_lines.append(
                        f"  repo:{h['file']}:{h['line']}: {h['text']}"
                    )
            elif s.diff_introduced:
                # D1 (lapis-pm-reviewer-leg-repair-v0): absent in the base
                # tree but present in the diff's added lines — by construction
                # present at the PR head. Classify `new`, never
                # `missing_referent` (I1).
                substrate_lines.append(
                    "  (not found in base tree; introduced by this diff — "
                    "classify as `new`)"
                )
            else:
                substrate_lines.append("  (not found in repo)")
            if include_vault and s.vault_hits:
                for h in s.vault_hits[:2]:
                    substrate_lines.append(
                        f"  vault:{h['file']}:{h['line']}: {h['text']}"
                    )
        substrate_text = "\n".join(substrate_lines)

        prompt = (
            f"You are checking doc-mentioned-identifier claims in a PR diff "
            f"for repo `{repo}`. Identifiers are names mentioned in prose, "
            f"commit messages, or spec text.\n\n"
            f"## Diff (excerpt)\n```diff\n{diff_text[:2500]}\n```\n\n"
            f"## Substrate (grep results)\n{substrate_text}\n\n"
            f"For each identifier listed in the substrate, determine whether "
            f"it exists in the repo as claimed. Classify each as:\n"
            f"- `missing_referent`: identifier mentioned in diff prose but not found in repo "
            f"AND not introduced by this diff\n"
            f"- `renamed_referent`: identifier mentioned but found only under a different name\n"
            f"- `stale_referent`: identifier found but its state doesn't match the claim\n"
            f"- `new`: identifier not found in the base tree but introduced by this diff's "
            f"added lines (present at the PR head by construction) — benign, not drift\n"
            f"- `none`: identifier exists as claimed (no drift)\n\n"
            f"Return ONLY valid JSON with this exact schema:\n"
            f'{{"verdict": "clean"|"flagged"|"uncertain", '
            f'"claims": [{{"identifier": str, "drift_class": "missing_referent"|"stale_referent"|"renamed_referent"|"new"|"none", "notes": str}}], '
            f'"summary": str}}\n'
            f"verdict=flagged if any claim has drift_class in "
            f"(missing_referent, renamed_referent, stale_referent) — `new` and `none` "
            f"never flag. "
            f"verdict=clean if all claims are `new` or `none`. "
            f"verdict=uncertain if substrate insufficient.\n"
            f"No prose outside the JSON."
        )

        def _build_body(use_grammar: bool) -> dict:
            body: dict = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                # max_tokens: _MAX_TOKENS (65536 default, tunable via
                # LAPIS_REVIEWER_MAX_TOKENS; Erah ruling 2026-09-19): thinking
                # stays ON — the deliberation is the function the reviewer legs
                # exist for. No chat_template_kwargs field: the rev-3
                # thinking-disable design is void; no leg sends it.
                "max_tokens": _MAX_TOKENS,
            }
            if node_model:
                body["model"] = node_model
            if use_grammar:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CorroborationVerdict",
                        "strict": True,
                        "schema": CORROBORATION_JSON_SCHEMA,
                    },
                }
            return body

        def _node2_extra_body(use_grammar: bool) -> dict:
            """D2 (lapis-pm-reviewer-leg-repair-v0): the node2 payload fields
            that ride `extra_body` on the sanctioned client. ALL THREE —
            temperature, max_tokens, and the json_schema grammar — must ride
            through: PhalaTeeClient merges extra_body verbatim into the POST
            body, and dropping max_tokens would apply TEE defaults ->
            truncated reasoning -> failed leg. `model` is passed as the
            client's own kwarg (it is set on the body by the client)."""
            body = _build_body(use_grammar)
            body.pop("messages", None)
            body.pop("model", None)
            return body

        try:
            import httpx
            _url = node_url or _llm_url()
            _timeout = node_timeout or _LLM_TIMEOUT

            if via_node2_client:
                # D2 (lapis-pm-reviewer-leg-repair-v0): the node2 leg goes
                # through the sanctioned PhalaTeeClient verifying path —
                # attestation fetch + ACI verify hop on every call,
                # loopback-refusal policy, $2/day spend cap, locality-ledger
                # row. No raw Bearer POST, no hardcoded URL (I2).
                try:
                    _client = _build_node2_client()
                except Exception as _imp_exc:
                    # Cross-repo import/resolve failure: a WIRING defect, not
                    # a TEE outage. Named class, never masked as generic
                    # unavailability (gate-4 amendment).
                    self._last_model = None
                    self._last_prompt_hash = None
                    return CorroborationResult(
                        verdict="uncertain",
                        claim="(substrate unavailable)",
                        citations=[],
                        freshness_stamp=datetime.now(timezone.utc).isoformat(),
                        scope_id=scope_id,
                        drift_class=None,
                        notes=(
                            f"{NODE2_ERROR_CLIENT_IMPORT}: "
                            f"{type(_imp_exc).__name__} — node2 wiring "
                            f"defect (cross-repo PhalaTeeClient import "
                            f"failed); local legs unaffected"
                        ),
                        leg_status="substrate_unavailable",
                    )
                try:
                    resp_json = _client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=node_model or _NODE2_MODEL,
                        extra_body=_node2_extra_body(True),
                    )
                except Exception as _n2_exc:
                    # Named error classes (D2 / DoD-1): map the exception to
                    # node2_unavailable / node2_aci_unverified /
                    # node2_cap_refused; the note carries the class so the
                    # DEGRADED PANEL line and the pm/review-state mem summary
                    # can surface it.
                    #
                    # Grammar-degrade retry (D2): the client raises on 4xx
                    # (the raw POST path used to inspect the status code
                    # directly), so a 400/422 grammar rejection is caught
                    # here and retried ONCE without response_format —
                    # otherwise a grammar-rejecting hop would fail the leg
                    # 100%, the exact failure D2 exists to kill.
                    #
                    # Key-redaction sentinel (extended, D2): the key value
                    # lives only in the client's auth headers; the named
                    # client exceptions carry spend/URL, never the key. The
                    # note echoes only the exception CLASS NAME, never the
                    # message, for every named client exception class — so a
                    # future exception string carrying a key cannot leak
                    # through this path.
                    _n2_class = _node2_error_class(_n2_exc)
                    _n2_note = (
                        f"{_n2_class}: {type(_n2_exc).__name__} — node2 "
                        f"(PhalaTeeClient) failed closed; local legs "
                        f"unaffected"
                    )
                    if "400" in str(_n2_exc) or "422" in str(_n2_exc):
                        try:
                            resp_json = _client.chat_completion(
                                messages=[{"role": "user", "content": prompt}],
                                model=node_model or _NODE2_MODEL,
                                extra_body=_node2_extra_body(False),
                            )
                        except Exception as _n2_exc2:
                            _n2_class = _node2_error_class(_n2_exc2)
                            _n2_note = (
                                f"{_n2_class}: {type(_n2_exc2).__name__} — "
                                f"node2 (PhalaTeeClient) grammar-degrade "
                                f"retry failed; local legs unaffected"
                            )
                            self._last_model = None
                            self._last_prompt_hash = None
                            return CorroborationResult(
                                verdict="uncertain",
                                claim="(substrate unavailable)",
                                citations=[],
                                freshness_stamp=datetime.now(timezone.utc).isoformat(),
                                scope_id=scope_id,
                                drift_class=None,
                                notes=_n2_note,
                                leg_status="substrate_unavailable",
                            )
                    else:
                        self._last_model = None
                        self._last_prompt_hash = None
                        return CorroborationResult(
                            verdict="uncertain",
                            claim="(substrate unavailable)",
                            citations=[],
                            freshness_stamp=datetime.now(timezone.utc).isoformat(),
                            scope_id=scope_id,
                            drift_class=None,
                            notes=_n2_note,
                            leg_status="substrate_unavailable",
                        )
            else:
                # Phala key gate (lapis-pm-panel-leg-survival-v0, Change 2):
                # the PHALA_API_KEY read and the Authorization header live on
                # the raw-POST path only. Node1 and the witness never read the
                # key, so a daemon env missing it takes down node2 only, never
                # the local legs. Absent key fails closed with a greppable
                # note naming the env var — never the value.
                _headers: dict[str, str] = {}
                if _url.startswith("https://inference.phala.com"):
                    _phala_key = _phala_api_key()
                    if not _phala_key:
                        self._last_model = None
                        self._last_prompt_hash = None
                        return CorroborationResult(
                            verdict="uncertain",
                            claim="(substrate unavailable)",
                            citations=[],
                            freshness_stamp=datetime.now(timezone.utc).isoformat(),
                            scope_id=scope_id,
                            drift_class=None,
                            notes=(
                                f"PHALA_API_KEY not set in daemon env — node2 "
                                f"({_url}) failed closed without a call; local "
                                f"legs unaffected"
                            ),
                            leg_status="substrate_unavailable",
                        )
                    _headers["Authorization"] = f"Bearer {_phala_key}"

                if not node_reachable(_url, timeout=_PROBE_TIMEOUT):
                    self._last_model = None
                    self._last_prompt_hash = None
                    return CorroborationResult(
                        verdict="uncertain",
                        claim="(substrate unavailable)",
                        citations=[],
                        freshness_stamp=datetime.now(timezone.utc).isoformat(),
                        scope_id=scope_id,
                        drift_class=None,
                        notes=f"node unreachable (probe {_PROBE_TIMEOUT}s): {_url}",
                        leg_status="substrate_unavailable",
                    )

                resp = httpx.post(_url, json=_build_body(True), headers=_headers, timeout=httpx.Timeout(_timeout, connect=_CONNECT_TIMEOUT))
                if resp.status_code in (400, 422):
                    # Grammar-not-supported vs validation-failed, per the
                    # precedent at local_reviewer_witness.py:330-345: an HTTP
                    # 4xx on the first (grammar-carrying) attempt means the
                    # endpoint rejects json_schema — degrade to a plain call
                    # rather than hard-failing a seat that can't take the
                    # grammar. One retry, never a hard fail.
                    resp = httpx.post(_url, json=_build_body(False), headers=_headers, timeout=httpx.Timeout(_timeout, connect=_CONNECT_TIMEOUT))
                resp.raise_for_status()
                resp_json = resp.json()

            choice0 = resp_json["choices"][0]
            extracted = extract_completion_text(choice0.get("message"), choice0.get("finish_reason"))

            if extracted.truncated:
                # Truncation is a named failure, not a substrate claim: the call
                # reached the substrate (HTTP 200) but finish_reason=="length" cut
                # the model off before it produced a usable answer — visibly
                # different from "the machine was unreachable".
                self._last_model = None
                self._last_prompt_hash = None
                return CorroborationResult(
                    verdict="uncertain",
                    claim="(response truncated)",
                    citations=[],
                    freshness_stamp=datetime.now(timezone.utc).isoformat(),
                    scope_id=scope_id,
                    drift_class=None,
                    notes=(
                        f"LLM response truncated (finish_reason=length): model "
                        f"exhausted its {_MAX_TOKENS}-token budget before producing "
                        f"a complete answer"
                    ),
                    leg_status="truncated",
                )

            if not extracted.has_text:
                self._last_model = None
                self._last_prompt_hash = None
                return CorroborationResult(
                    verdict="uncertain",
                    claim="(substrate unavailable)",
                    citations=[],
                    freshness_stamp=datetime.now(timezone.utc).isoformat(),
                    scope_id=scope_id,
                    drift_class=None,
                    notes=f"LLM unavailable: {NO_USABLE_TEXT}",
                    leg_status="substrate_unavailable",
                )

            content = extracted.text.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content.strip())
            # Some models wrap with <think>...</think>; strip thinking
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            result_data = json.loads(content)
            # Capture provenance metadata from the successful LLM call
            self._last_model = resp_json.get("model")
            self._last_prompt_hash = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            self._last_score_at = datetime.now(timezone.utc)
        except Exception as exc:
            # Clear provenance — most-recent score() attempt did not succeed;
            # a stale model from a prior call should not claim credit.
            self._last_model = None
            self._last_prompt_hash = None
            # The exception message is echoed for legibility, EXCEPT on the
            # Phala path where a key-bearing error string must never reach a
            # notes/error/claim field (lapis-pm-panel-leg-survival-v0 DoD #3
            # sentinel: the env var name may be named, the value never).
            _phala_path = _url.startswith("https://inference.phala.com")
            if _phala_path:
                _note = (
                    f"LLM unavailable: {type(exc).__name__} on node2 ({_url}); "
                    f"see daemon log for details"
                )
            else:
                # Message included, not just the exception class name — a bare
                # class name is what hid this defect for over a week.
                _note = f"LLM unavailable: {type(exc).__name__}: {exc}"
            return CorroborationResult(
                verdict="uncertain",
                claim="(substrate unavailable)",
                citations=[],
                freshness_stamp=datetime.now(timezone.utc).isoformat(),
                scope_id=scope_id,
                drift_class=None,
                notes=_note,
                leg_status="substrate_unavailable",
            )

        verdict = result_data.get("verdict", "uncertain")
        if verdict not in ("clean", "flagged", "uncertain"):
            verdict = "uncertain"

        claims = result_data.get("claims", [])

        # Build citations from substrate grep results for flagged identifiers.
        # D1 (lapis-pm-reviewer-leg-repair-v0): `new` is benign — no
        # citations, same treatment as `none`.
        citations: list[Citation] = []
        for claim_item in claims:
            dc = claim_item.get("drift_class", "none")
            if dc in ("none", "new"):
                continue
            ident = claim_item.get("identifier", "")
            for s in substrates:
                if s.identifier == ident:
                    for h in s.repo_hits[:1]:
                        citations.append(Citation(
                            source_id=f"repo:{repo}:{h['file']}:{h['line']}",
                            excerpt=h["text"],
                            provenance_method="diff_grep",
                        ))
                    break

        # Worst drift_class across all claims.
        # D1 (lapis-pm-reviewer-leg-repair-v0): `new` is benign (rank 0, same
        # as `none`). Hard backstop: an UNKNOWN class also ranks 0 via
        # `_drift_rank.get(dc, 0)`, so even a prompt-level miss that emits an
        # unrecognized class cannot flag on its own.
        _drift_rank = {
            "missing_referent": 3,
            "renamed_referent": 2,
            "stale_referent": 1,
            "new": 0,
            "none": 0,
        }
        worst_dc: str | None = None
        for claim_item in claims:
            dc = claim_item.get("drift_class", "none")
            if _drift_rank.get(dc, 0) > _drift_rank.get(worst_dc or "none", 0):
                worst_dc = dc
        if worst_dc == "none":
            worst_dc = None

        return CorroborationResult(
            verdict=verdict,
            claim=result_data.get("summary", ""),
            citations=citations,
            freshness_stamp=datetime.now(timezone.utc).isoformat(),
            scope_id=scope_id,
            drift_class=worst_dc,
            notes=result_data.get("summary"),
            # A successful pass whose response omits `summary` produces claim=="" —
            # indistinguishable from a failure marker by string content alone. This
            # branch reached a real LLM response and parsed it; the leg is healthy
            # regardless of what `claim` says.
            leg_status="ok",
        )

    def score_provenance(self) -> dict[str, Any]:
        """Return adapter-level provenance metadata for the most-recent score() call.

        Returns {"model", "prompt_hash", "upstream_calls"} — keys are populated only
        if a successful score() has run since construction. Empty dict if no LLM
        call has succeeded.

        Recognized by archetypes_core.corroboration.corroborate_envelope() when
        this adapter is passed in; the returned dict's keys flow into the envelope's
        provenance (model, prompt_hash, upstream_calls).

        Concurrency note (v0): _last_* state is a single-call-at-a-time abstraction.
        If two score() calls on the same adapter instance interleave (threads or
        asyncio), score_provenance() returns the metadata of whichever call wrote
        last. This matches the schema doc's framing: provenance describes the
        most-recent call. v1 may switch to a context-local store or per-call return
        value if concurrent usage shows up.
        """
        if self._last_model is None and self._last_prompt_hash is None:
            return {}
        return {
            "model": self._last_model,
            "prompt_hash": self._last_prompt_hash,
            "upstream_calls": [],  # leaf adapter — no nested LLM calls
        }


# ---------------------------------------------------------------------------
# Public entry point for pm_core integration
# ---------------------------------------------------------------------------

def _make_uncertain(repo: str, notes: str) -> CorroborationResult:
    """Fresh CorroborationResult for a failed corroboration pass.

    Each call returns a brand-new instance — never share one across the
    retrieve-failure path and the two node-scoring threads. Those run
    concurrently (ThreadPoolExecutor(max_workers=2)); a single shared
    instance means both nodes failing collapses to `result_n1 is result_n2`,
    which makes the divergence test trivially "agree" and turns `.notes`
    writes into a race between the two threads.
    """
    return CorroborationResult(
        verdict="uncertain",
        claim="(corroboration pass failed)",
        citations=[],
        freshness_stamp=datetime.now(timezone.utc).isoformat(),
        scope_id=f"repo:{repo}",
        drift_class=None,
        notes=notes,
        leg_status="pass_failed",
    )


def run_corroboration_pass(
    diff_text: str,
    repo: str,
    repo_path: str | None = None,
) -> dict:
    """Run the corroboration follow-up pass on a PR diff.

    Dispatches to the local GW seat (primary) and the Phala TEE (node2,
    re-pointed 2026-09-01 — see the _NODE2_URL block) in parallel.
    Returns a dict suitable for attaching as `corroboration_result` to the
    reviewer verdict JSON. Never raises — returns uncertain-shaped dict on any
    error (Compost invariant: all outputs are nutrients).

    Added fields beyond v0 shape:
      node2_corroboration: CorroborationResult dict from the Phala TEE, or None
      cross_node_divergence: "agree" | "diverge" | "node2_unavailable"
    """
    import concurrent.futures

    # Two separate adapter instances to avoid provenance-tracking races.
    adapter_n1 = LapisPMReviewerAdapter(repo_path=repo_path)
    adapter_n2 = LapisPMReviewerAdapter(repo_path=repo_path)

    try:
        substrates = adapter_n1.retrieve(diff_text, repo, repo_path)
    except Exception as exc:
        return _make_uncertain(repo, f"Retrieve error: {type(exc).__name__}: {exc}").to_dict()

    def _score_n1() -> CorroborationResult:
        try:
            return adapter_n1.score(diff_text, substrates, repo)
        except Exception as exc:
            return _make_uncertain(repo, f"Node1 error: {type(exc).__name__}: {exc}")

    def _score_n2() -> CorroborationResult:
        try:
            # S6 (gate-lanes-registry-driven-flashnext-v0): under
            # flashnext-solo (the 27B seat down, the flashnext seat :30000
            # registered + serving), the node2 leg builds against the
            # flashnext lane via the existing node_url injection point —
            # the registry-served model name on :30000, never the dead
            # Phala/legacy path. node2_unavailable then fires ONLY when
            # :30000 is actually unreachable (honest panel_starvation row),
            # eliminating the node2_unavailable starvation class under
            # flashnext-solo.
            #
            # Caller contract (S1): (None, None, None) = registry blind OR
            # a non-solo posture (27B up) — the ONLY shapes that fall back
            # to the legacy PhalaTeeClient path byte-identically. A readable
            # registry, 27B down, with a dead flashnext lane is an honest
            # node2_unavailable, never a masked legacy fallback (MED-4
            # fold: the blocked reason now flows here as the third value).
            _flash_url, _flash_model, _flash_blocked = _flashnext_lane_url()
            if _flash_blocked is not None:
                return _make_uncertain(
                    repo,
                    f"node2_unavailable: gate-lanes registry readable, 27B "
                    f"down, flashnext lane not serving ({_flash_blocked}) — "
                    f"honest leg_down, no masked legacy fallback",
                )
            if _flash_url is not None:
                return adapter_n2.score(
                    diff_text, substrates, repo,
                    node_url=_flash_url,
                    node_model=_flash_model,
                    node_timeout=_NODE2_TIMEOUT,
                    # Code-only tenancy constraint (Erah, 2026-09-01): the
                    # outside caller never sees vault content — diff excerpt
                    # + repo grep hits only. Build-time guard; `substrates`
                    # is not mutated.
                    include_vault=False,
                )
            # D2 (lapis-pm-reviewer-leg-repair-v0): node2 rides the sanctioned
            # PhalaTeeClient verifying path (attestation + ACI verify hop on
            # every call), not a raw Bearer POST. The client reads
            # PHALA_API_KEY / PHALA_BASE_URL / ACI_VERIFIER_BIN from the
            # daemon env (phala.env, already in the service EnvironmentFile).
            return adapter_n2.score(
                diff_text, substrates, repo,
                node_model=_NODE2_MODEL,
                node_timeout=_NODE2_TIMEOUT,
                # Code-only tenancy constraint (Erah, 2026-09-01): the outside
                # caller never sees vault content — diff excerpt + repo grep
                # hits only. Build-time guard; `substrates` is not mutated.
                include_vault=False,
                via_node2_client=True,
            )
        except Exception as exc:
            return _make_uncertain(repo, f"Node2 error: {type(exc).__name__}: {exc}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_n1 = pool.submit(_score_n1)
        fut_n2 = pool.submit(_score_n2)
        result_n1 = fut_n1.result()
        result_n2 = fut_n2.result()

    combined = result_n1.to_dict()
    n2_dict = result_n2.to_dict()
    combined["node2_corroboration"] = n2_dict

    # Structured status, not prose. The old substring match on `notes` raced
    # with the shared-`_uncertain` alias above (D2): whichever thread wrote
    # last owned `.notes`, so the node-2 check could silently miss a real
    # node-2 failure. leg_status is set correctly by each node's own
    # CorroborationResult regardless of write order.
    n2_unavailable = result_n2.leg_status != "ok"
    if n2_unavailable:
        combined["cross_node_divergence"] = "node2_unavailable"
    elif result_n1.verdict == result_n2.verdict:
        combined["cross_node_divergence"] = "agree"
    else:
        combined["cross_node_divergence"] = "diverge"

    return combined
