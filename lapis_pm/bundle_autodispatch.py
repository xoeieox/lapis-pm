"""bundle_autodispatch — reconcile unbound cr-bundle-* specs and bind+tick qualifying ones.

Runs nightly after code-reviewer's debt-bundle sweep. Discovers
cr-bundle-<repo>-<YYYY-MM-DD>.md specs in /srv/lapis/planning/specs (excluding superseded/)
whose frontmatter source contains code-reviewer-debt-bundle-v0, runs the GW spec-review
gate on each unbound one, and binds+ticks the ones that pass with advisory authority.

Idempotency / marker states (sibling to spec, same directory — POSIX rename(2) atomic):
  .autodispatch          - success: dispatched:1 verified in same execution window.
  .autodispatch-pending  - write-ahead intent: written before bind; transitioned atomically
                           on resolve (→ .autodispatch) or confirmed crash (→ .autodispatch-failed).
  .autodispatch-failed   - tombstone: transitioned from .autodispatch-pending only when a
                           genuine crash is detected (pending present on a later run means
                           the writer is gone — singleton lock is the primary guarantee).

State machine per spec:
  .autodispatch present        → SKIP (already completed).
  .autodispatch-failed present → SKIP (human triage).
  .autodispatch-pending present → crash mid-flight → rename pending→failed + loud log.
  target YAML present, no marker → externally/manually bound → SKIP + audit log. NEVER tombstone.
  no target YAML, no pending    → fresh → gate + bind.

Tombstone invariant: .autodispatch-failed is written ONLY from the presence of an unresolved
.autodispatch-pending intent — never from the mere absence of a success marker.

Timestamp source of truth: run_ts is injected by the CLI entry point, not generated inside
this module. This keeps core logic deterministic and avoids clock-skew issues.

GW-serving:
  - swarm_serving() is checked BEFORE calling run_spec_review.
  - If GW is not serving: defer (no bind, no marker, loud log). Never fall back to paid.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeoutError
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SPEC_DIR = Path("/srv/lapis/planning/specs")
_DEFAULT_GATE_TIMEOUT_S = 1800

# Patterns for parsing bundle spec frontmatter
_YAML_SPEC_ID_RE = re.compile(r"^spec_id:\s*(\S+)\s*$", re.MULTILINE)
_MD_REPO_RE = re.compile(
    r"^\*\*Repo:\*\*\s+`([a-z0-9][a-z0-9-]*[a-z0-9])`\s*$",
    re.MULTILINE,
)

# Debt-bundle filename: cr-bundle-<repo>-<YYYY-MM-DD>.md
# Excludes: cr-bundle-autodispatch-v0.md, cr-bundle-notadate.md, this fix spec, etc.
_DEBT_BUNDLE_FILENAME_RE = re.compile(r"^cr-bundle-.+-\d{4}-\d{2}-\d{2}\.md$")

# Source frontmatter must contain code-reviewer-debt-bundle-v0
_DEBT_BUNDLE_SOURCE_RE = re.compile(r"^source:.*code-reviewer-debt-bundle-v0", re.MULTILINE)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

def _parse_bundle_frontmatter(spec_path: Path) -> tuple[str, str]:
    """Parse (spec_id, repo) from a cr-bundle spec.

    spec_id comes from YAML frontmatter block (`spec_id: ...`).
    repo comes from the markdown `**Repo:** `...`` line.
    Raises ValueError on parse failure.
    """
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"cannot read spec: {e}") from e

    head = text[:2000]  # only scan the preamble

    sid_m = _YAML_SPEC_ID_RE.search(head)
    if not sid_m:
        raise ValueError(f"missing 'spec_id:' in YAML frontmatter of {spec_path.name}")

    repo_m = _MD_REPO_RE.search(head)
    if not repo_m:
        raise ValueError(f"missing '**Repo:** `...`' in {spec_path.name}")

    return sid_m.group(1), repo_m.group(1)


def _has_debt_bundle_source(spec_path: Path) -> bool:
    """Return True if the spec frontmatter source contains code-reviewer-debt-bundle-v0."""
    try:
        head = spec_path.read_text(encoding="utf-8")[:2000]
    except OSError:
        return False
    return bool(_DEBT_BUNDLE_SOURCE_RE.search(head))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_specs(spec_dir: Path) -> list[Path]:
    """Return sorted debt-bundle spec paths, excluding superseded/ and non-bundle specs.

    A spec qualifies only if BOTH hold:
    - filename matches cr-bundle-<repo>-<YYYY-MM-DD>.md (ISO-date suffix)
    - frontmatter source: contains code-reviewer-debt-bundle-v0

    Specs failing either check are silently skipped (debug-log only) — they are not
    bundles, not errors. This excludes cr-bundle-autodispatch-v0.md and any future
    non-debt-bundle specs whose names start with cr-bundle-.
    """
    superseded = spec_dir / "superseded"
    result = []
    for p in sorted(spec_dir.glob("cr-bundle-*.md")):
        if not p.is_file():
            continue
        try:
            p.relative_to(superseded)
            continue  # inside superseded/ — skip
        except ValueError:
            pass
        if not _DEBT_BUNDLE_FILENAME_RE.match(p.name):
            logger.debug(
                "[bundle-autodispatch] skipping %s: filename does not match debt-bundle pattern",
                p.name,
            )
            continue
        if not _has_debt_bundle_source(p):
            logger.debug(
                "[bundle-autodispatch] skipping %s: source frontmatter is not code-reviewer-debt-bundle-v0",
                p.name,
            )
            continue
        result.append(p)
    return result


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------

def _autodispatch_marker(spec_path: Path) -> Path:
    return Path(str(spec_path) + ".autodispatch")


def _pending_marker(spec_path: Path) -> Path:
    return Path(str(spec_path) + ".autodispatch-pending")


def _failed_marker(spec_path: Path) -> Path:
    return Path(str(spec_path) + ".autodispatch-failed")


def _salvage_attempted_marker(spec_path: Path) -> Path:
    """HARD BOUND: one salvage attempt per spec per night. Sibling marker,
    same directory as the .autodispatch* family. Tolerant of the 04:30
    supersede rename in the same sense the existing .autodispatch* markers
    already are: debt_bundle.py's _move_to_superseded moves only the spec
    .md file (not sibling markers) — an accepted, pre-existing property this
    unit does not change. By the time a spec would be superseded, autodispatch
    has already run against it for the night that matters (and post the
    Design 5 timer-sequencing fix, autodispatch runs AFTER the bundler, so
    it always sees the live, not-yet-superseded set)."""
    return Path(str(spec_path) + ".autodispatch-salvage-attempted")


# ---------------------------------------------------------------------------
# State checks
# ---------------------------------------------------------------------------

def _target_yaml_exists(spec_id: str) -> bool:
    """Return True if /srv/lapis/targets/<spec_id>.yaml exists."""
    from agents_core.room_paths import room_path
    return (room_path("targets") / f"{spec_id}.yaml").exists()


def _gw_serving() -> bool:
    """Return True if GravityWell swarm is serving."""
    try:
        from agents_core.llm import swarm_serving
        return swarm_serving()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def _run_gate(spec_path: Path, timeout_s: int) -> "SpecReviewBrief | None":
    """Run spec-review gate with a hard outer timeout.

    Returns the brief on success, or None if the gate timed out or errored.
    None always means defer-to-human (no bind, no marker).
    """
    from lapis_pm.spec_review import run_spec_review

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        run_spec_review,
        spec_path=spec_path,
        council_voicing="gravitywell",
        timeout_s=timeout_s,
        authority="advisory",
        dispatch_facets=True,
        sonnet_reviewer=False,
        # lapis-pm-bundle-autodispatch-enforce-v0: pass the invoker explicitly
        # rather than relying on hold_shadow's stack-walk, which never sees
        # this frame — the gate runs on this ThreadPoolExecutor worker, not
        # on the caller's own stack.
        invoked_by="bundle_autodispatch",
    )
    grace = 120  # seconds above the inner timeout before we give up
    try:
        return future.result(timeout=timeout_s + grace)
    except _FuturesTimeoutError:
        logger.warning(
            "[bundle-autodispatch] gate hard-timeout (%ds) for %s — deferring",
            timeout_s, spec_path.name,
        )
        return None
    except Exception as exc:
        logger.warning(
            "[bundle-autodispatch] gate error for %s: %s — deferring",
            spec_path.name, exc,
        )
        return None
    finally:
        # Don't block on the thread finishing — let it run to its own internal timeout
        # in the background. Using the context-manager form would call shutdown(wait=True)
        # here, defeating the hard-timeout guarantee.
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Bind + tick + verify
# ---------------------------------------------------------------------------

def _bind(spec_id: str, repo: str, spec_path: Path) -> bool:
    """Bind the spec advisory. Returns True on success."""
    from lapis_pm.cli import cmd_bind

    args = argparse.Namespace(
        target_id=spec_id,
        spec_from=str(spec_path),
        repo=repo,
        authority="advisory",
        # lapis-pm-containment-held-paths-v0 Leg 4: pin verification explicitly.
        # Without this, cli.py's getattr(args, "verification", None) falls through
        # to None and the spec's OWN prose ("**Verification:**") decides — and this
        # bind path runs unattended nightly on interpolated debt-bundle content, so
        # an injected "**Verification:** machine" line would self-grant the
        # machine-merge clause of auto_resolve.py. An auto-bound spec can never
        # carry machine verification.
        verification="pm-live-test",
        legs_from=None,
        no_auto_fire=False,
        force=False,
        create=True,
        title=f"Resolve aged MED debt items in {repo}",
        description="",
        urgency="medium",
        work_mode="anywhere",
        tag=[],
        product="",
        pr_count=None,
        destination_slug=None,
        destination_name=None,
        destination_when=None,
        loom_visibility=None,
        adopt_pr=None,
    )
    try:
        rc = cmd_bind(args)
        return rc == 0
    except Exception as exc:
        logger.error("[bundle-autodispatch] bind raised for %s: %s", spec_id, exc)
        return False


def _tick(spec_id: str, repo: str) -> bool:
    """Fire the initial fixer dispatch. Returns True on success."""
    from lapis_pm import pm_core

    intent = (
        f"Resolve all bundled MED debt items in {repo} per the bound spec. "
        f"Follow Deliverables, Interface, Tests, Invariants, and Definition-of-Done exactly. "
        f"Each item ends as either resolved-in-this-PR or explicitly marked invalid with reasoning."
    )
    try:
        pm_core.force_dispatch(spec_id, "fixer", intent)
        return True
    except Exception as exc:
        logger.error("[bundle-autodispatch] tick raised for %s: %s", spec_id, exc)
        return False


def _verify_dispatched(spec_id: str) -> bool:
    """Return True if dispatched >= 1 for the target."""
    from lapis_pm import pm_core
    dispatched = pm_core.load_dispatched(spec_id)
    return len(dispatched) >= 1


def _do_bind_tick_verify(
    spec_id: str, repo: str, spec_path: Path, run_ts: str, results: dict,
) -> None:
    """Write-ahead pending marker, bind, tick, verify — the crash-guarded
    bind path (unchanged from the pre-enforce flow). Shared by the direct
    proceed-to-bind case and the post-salvage clean-re-gate case, so both
    get the identical crash-guard discipline."""
    _pending_marker(spec_path).write_text(
        f"autodispatch-pending: spec_id={spec_id} repo={repo} ts={run_ts}\n",
        encoding="utf-8",
    )
    logger.info(
        "[bundle-autodispatch] wrote .autodispatch-pending for %s (ts=%s)", spec_id, run_ts,
    )

    logger.info(
        "[bundle-autodispatch] binding %s → repo=%s authority=advisory", spec_id, repo,
    )
    if not _bind(spec_id, repo, spec_path):
        logger.error(
            "[bundle-autodispatch] !!! %s bind failed — "
            "renaming .autodispatch-pending → .autodispatch-failed",
            spec_id,
        )
        _pending_marker(spec_path).rename(_failed_marker(spec_path))
        results["failed"].append({"spec": spec_id, "reason": "bind_failed"})
        return

    logger.info("[bundle-autodispatch] firing initial tick for %s", spec_id)
    if not _tick(spec_id, repo):
        logger.error(
            "[bundle-autodispatch] !!! %s tick failed after bind — "
            "renaming .autodispatch-pending → .autodispatch-failed",
            spec_id,
        )
        _pending_marker(spec_path).rename(_failed_marker(spec_path))
        results["failed"].append({"spec": spec_id, "reason": "tick_failed"})
        return

    if not _verify_dispatched(spec_id):
        logger.error(
            "[bundle-autodispatch] !!! %s dispatched<1 after tick — "
            "renaming .autodispatch-pending → .autodispatch-failed",
            spec_id,
        )
        _pending_marker(spec_path).rename(_failed_marker(spec_path))
        results["failed"].append({"spec": spec_id, "reason": "dispatch_not_verified"})
        return

    _pending_marker(spec_path).rename(_autodispatch_marker(spec_path))
    logger.info(
        "[bundle-autodispatch] successfully bound+ticked+verified %s — "
        ".autodispatch-pending renamed to .autodispatch",
        spec_id,
    )
    results["bound"].append({"spec": spec_id, "repo": repo})


# ---------------------------------------------------------------------------
# Three-class enforce classifier (lapis-pm-bundle-autodispatch-enforce-v0)
#
# Erah ruled 2026-08-13 (AskUserQuestion): infra-faulted gates retry once
# then record (never bind, never escalate); an auto-generated debt bundle
# whose items are partly invalid may be salvaged in place (drop the invalid
# items, record them to the debt ledger, re-gate once); genuine council
# blocks / forks always defer to Erah. INFRA wins over SALVAGE on a dual
# match (run 9b54d3f6: shape-with-Erah label, but the ground is a synthesis
# timeout — retry before salvaging). Classifier faults / unparseable output
# fail closed to DEFER, never bind.
# ---------------------------------------------------------------------------

# Markers naming a genuine infra fault whose OWN text leaves room for
# transient recovery — retry once. Absence of a transient marker, or
# presence of a permanent one, means zero retries (gate-amended
# 2026-08-13: fault-text retry test, DoD 7c).
_TRANSIENT_FAULT_MARKERS = (
    "timeout", "timed out", "time out", "queue", "unreachable",
    "connection reset", "connection refused", "temporarily unavailable",
    "did not complete within",
)
_PERMANENT_FAULT_MARKERS = (
    "malformed", "invalid syntax", "not recoverable", "permanently",
    "unsupported", "parse error", "syntax error",
)
_SYNTHESIS_ERROR_MARKERS = (
    "synthesis error", "did not complete within", "queue-side", "queue timeout",
)


def _is_transient_fault_text(text: str) -> bool:
    """True only when the fault's own text names a transient-recovery
    marker and no permanent marker. A permanent marker always wins (a fault
    that says both "queue" and "malformed" is not a transient queue hiccup)."""
    if not text:
        return False
    low = text.lower()
    if any(m in low for m in _PERMANENT_FAULT_MARKERS):
        return False
    return any(m in low for m in _TRANSIENT_FAULT_MARKERS)


def _infra_fault_text(brief) -> str | None:
    """Return the verbatim fault text if *brief* matches an INFRA trigger,
    else None. Checked BEFORE salvage/defer so a dual-match (a synthesis
    fault rendered as shape-with-Erah) always classifies infra."""
    if brief is None:
        return "gate returned no brief (timeout or error)"

    rec = getattr(brief, "combined_recommendation", "")
    if rec in ("incomplete", "parse_failed"):
        pe = getattr(brief, "parse_error", None)
        if isinstance(pe, dict) and pe.get("detail"):
            return str(pe["detail"])
        return f"combined_recommendation={rec}"

    status = getattr(brief, "council_status", "")
    if status in ("failed", "error"):
        reason = getattr(brief, "council_error_reason", "") or ""
        return reason or f"council_status={status}"

    fd = getattr(brief, "facets_deliberation", None)
    synthesis = fd.get("synthesis") if isinstance(fd, dict) else None
    if isinstance(synthesis, dict):
        escalation_reason = synthesis.get("escalation_reason") or ""
        if escalation_reason and any(
            m in escalation_reason.lower() for m in _SYNTHESIS_ERROR_MARKERS
        ):
            return escalation_reason

    return None


def _defer_ground(brief) -> str:
    """Verbatim reasons for a DEFER, preserved as stored — never paraphrased."""
    if brief is None:
        return "gate returned no brief (timeout or error)"
    rec = getattr(brief, "combined_recommendation", "unknown")
    open_questions = list(getattr(brief, "council_open_questions", None) or [])
    parts = [f"combined_recommendation={rec}"]
    if open_questions:
        parts.append("open_questions: " + " | ".join(open_questions))
    return "; ".join(parts)


def _gw_grounds_text(brief) -> str:
    """The byte-identical stored grounds — escalation_reason + council open
    questions, AS STORED in the gate record — pinned verbatim (gate-amended
    2026-08-13, Facets). Never a re-summarized or LLM-paraphrased version."""
    fd = getattr(brief, "facets_deliberation", None)
    synthesis = fd.get("synthesis") if isinstance(fd, dict) else None
    escalation_reason = ""
    if isinstance(synthesis, dict):
        escalation_reason = synthesis.get("escalation_reason") or ""
    open_questions = list(getattr(brief, "council_open_questions", None) or [])
    parts = []
    if escalation_reason:
        parts.append(escalation_reason)
    if open_questions:
        parts.append("\n".join(open_questions))
    return "\n".join(parts)


def _extract_json_object(text: str) -> dict | None:
    """Tolerate a fenced code block around the classifier's JSON. Returns
    None (never raises) on anything that doesn't parse as a JSON object."""
    import json as _json

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        parsed = _json.loads(stripped)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _classify_salvage_via_gw(brief) -> dict | None:
    """GW-voiced classification: are the gate's stated grounds about invalid
    or stale bundle ITEMS (salvage) or a genuine design fork (defer)?
    Precedent-blind (gate-amended, Council): no prior night's outcomes are
    ever fed in. Returns None on any fault, missing grounds, or unparseable
    output — callers must fail-closed to DEFER on None."""
    grounds_text = _gw_grounds_text(brief)
    if not grounds_text.strip():
        return None

    prompt = (
        "You are classifying why an automated debt-bundle spec-review gate "
        "did not proceed to bind. Read the verbatim grounds below and decide "
        "whether the gate's concern is that some bundle ITEMS are invalid or "
        "stale (SALVAGE — the bundle can be amended by dropping those items "
        "and re-gated), or whether the concern is a genuine design fork that "
        "needs a human decision (DEFER). Judge this night's grounds alone — "
        "you have no memory of prior nights.\n\n"
        f"Verbatim grounds (byte-identical, as stored):\n{grounds_text}\n\n"
        "Respond with STRICT JSON only, no prose: "
        '{"classification": "salvage" | "defer"}'
    )
    try:
        from agents_core.llm import call_operator
        result = call_operator("gravitywell", prompt)
        if not result:
            return None
        parsed = _extract_json_object(result)
        if parsed is None:
            return None
        cls = parsed.get("classification")
        if cls not in ("salvage", "defer"):
            return None
        return {"is_salvage": cls == "salvage", "ground": grounds_text}
    except Exception as exc:
        logger.warning(
            "[bundle-autodispatch] salvage classifier fault: %s — fail-closed to DEFER", exc,
        )
        return None


def _classify_enforce(brief) -> tuple[str, str]:
    """Classify a non-bind gate outcome into infra | salvage | defer.
    Returns (enforce_class, verbatim_ground)."""
    infra_text = _infra_fault_text(brief)
    if infra_text is not None:
        return "infra", infra_text

    rec = getattr(brief, "combined_recommendation", "")
    if rec in ("shape-with-Erah", "amend-spec"):
        verdict = _classify_salvage_via_gw(brief)
        if verdict is None:
            return "defer", _defer_ground(brief)
        if verdict["is_salvage"]:
            return "salvage", verdict["ground"]
        return "defer", _defer_ground(brief)

    return "defer", _defer_ground(brief)


def _write_enforce_record(
    spec_id: str,
    brief,
    enforce_class: str,
    ground: str,
    action: str,
    salvage_dropped_items: list[dict] | None = None,
) -> None:
    from . import hold_shadow as _hold_shadow

    run_id = getattr(brief, "council_run_id", "") if brief is not None else ""
    _hold_shadow.observe_enforce_outcome(
        run_id=run_id,
        spec=spec_id,
        enforce_class=enforce_class,
        enforce_action=action,
        verbatim_grounds=ground,
        salvage_dropped_items=salvage_dropped_items,
    )


# ---------------------------------------------------------------------------
# Salvage: amend the bundle spec in place, ledger write-back, one re-gate
# (Design 2 + 2c). Machine amendment of spec bodies is barred elsewhere in
# this codebase (spec_attestation.py is frontmatter-only by invariant) —
# the bundle corpus is exempt because these specs are machine-generated
# from the debt ledger with no human author (Erah's salvage ruling, and no
# other corpus).
# ---------------------------------------------------------------------------

_SPEC_DEBT_ID_RE = re.compile(r"\*\*Debt ID:\*\*\s+`([^`]+)`")
_SPEC_SOURCE_PR_RE = re.compile(r"in PR #`(\d+)`")
_SPEC_ITEMS_SECTION_RE = re.compile(
    r"(^## Items\n\n)(.*?)(\n\n## )", re.DOTALL | re.MULTILINE,
)

# bundle-item-level-triage-v0: item schema extension. fix_description comes
# from the debt-bundle renderer's own "[suggestion: ...]" bracket (the
# concrete, actionable fix — code-reviewer-debt-bundle-v0's own shape, not
# invented here); touched_surfaces from the "**Files:**" line (comma-
# separated, backticked paths). Both are None/[] when absent — the triage
# classifier fails closed to "fork" on either, never invents a fix.
_SPEC_ITEM_SUGGESTION_RE = re.compile(r"\[suggestion:\s*(.*?)\]", re.DOTALL)
_SPEC_ITEM_FILES_RE = re.compile(r"^\s*\*\*Files:\*\*\s*(.+)$", re.MULTILINE)


def _parse_item_fix_description(entry: str) -> str | None:
    m = _SPEC_ITEM_SUGGESTION_RE.search(entry)
    if not m:
        return None
    text = re.sub(r"\s+", " ", m.group(1)).strip()
    return text or None


def _parse_item_touched_surfaces(entry: str) -> list[str]:
    m = _SPEC_ITEM_FILES_RE.search(entry)
    if not m:
        return []
    paths = [p.strip().strip("`").strip() for p in m.group(1).split(",")]
    return [p for p in paths if p]


def _parse_bundle_items(spec_text: str) -> list[dict]:
    """Parse each item block out of the spec's '## Items' section. Returns
    [{"debt_id", "block" (verbatim text, re-joined unchanged for kept
    items), "source_pr", "fix_description", "touched_surfaces"}]. Empty list
    if the section isn't found or has no parseable items — callers must
    fail-closed to DEFER (or, for triage, to "fork") on empty."""
    m = _SPEC_ITEMS_SECTION_RE.search(spec_text)
    if not m:
        return []
    items_block = m.group(2)
    entries = re.split(r"\n\n(?=\d+\.\s+\*\*Debt ID:\*\*)", items_block)
    items = []
    for entry in entries:
        idm = _SPEC_DEBT_ID_RE.search(entry)
        if not idm:
            continue
        prm = _SPEC_SOURCE_PR_RE.search(entry)
        items.append({
            "debt_id": idm.group(1),
            "block": entry,
            "source_pr": int(prm.group(1)) if prm else None,
            "fix_description": _parse_item_fix_description(entry),
            "touched_surfaces": _parse_item_touched_surfaces(entry),
        })
    return items


def _classify_invalid_items_via_gw(grounds_text: str, items: list[dict]) -> list[dict] | None:
    """Ask GW exactly which Debt IDs (from *items*) the verbatim grounds
    name as invalid/stale. Returns None (fail-closed to DEFER) on any
    fault, unparseable output, or an ID/confidence the model invented.
    Empty list means the classifier found nothing concretely invalid — also
    treated as fail-closed by the caller (a salvage classification with no
    droppable item is not actionable)."""
    id_list = "\n".join(f"- {it['debt_id']}" for it in items)
    prompt = (
        "A bundle-spec gate review concluded some items in the list below "
        "are invalid or stale and should be dropped from the bundle (the "
        "rest bind as-is). The verbatim grounds are given below; decide "
        "EXACTLY which Debt IDs from the list are invalid, and name which "
        "stored field grounded each verdict.\n\n"
        f"Debt IDs in this bundle:\n{id_list}\n\n"
        f"Verbatim grounds (byte-identical, as stored):\n{grounds_text}\n\n"
        "Respond with STRICT JSON only, no prose: "
        '{"invalid_items": [{"debt_id": "<id>", '
        '"confidence": "high" | "medium" | "low", '
        '"source": "<which stored field grounded this, e.g. escalation_reason, open_question>"}]}'
        ' If no item is clearly invalid, return {"invalid_items": []}.'
    )
    try:
        from agents_core.llm import call_operator
        result = call_operator("gravitywell", prompt)
        if not result:
            return None
        parsed = _extract_json_object(result)
        if parsed is None:
            return None
        raw_items = parsed.get("invalid_items")
        if not isinstance(raw_items, list):
            return None
        valid_ids = {it["debt_id"] for it in items}
        out = []
        for ri in raw_items:
            if not isinstance(ri, dict):
                return None
            did = ri.get("debt_id")
            conf = ri.get("confidence")
            if did not in valid_ids or conf not in ("high", "medium", "low"):
                return None
            out.append({
                "debt_id": did,
                "confidence": conf,
                "source": str(ri.get("source", ""))[:200] or "escalation_reason",
            })
        return out
    except Exception as exc:
        logger.warning(
            "[bundle-autodispatch] invalid-item classifier fault: %s — fail-closed", exc,
        )
        return None


def _amend_spec_salvage(
    spec_path: Path,
    items: list[dict],
    drop_items: list[dict],
    ground_text: str,
    run_ts: str,
) -> list[dict]:
    """Rewrite *spec_path* in place: drop the items named in drop_items from
    '## Items', append a '## Salvage record' section naming, per dropped
    item, its Debt ID, the verbatim ground, confidence, and inference
    source. Returns the dropped-item detail list (debt_id, confidence,
    source, source_pr) for the enforce record + ledger write-back."""
    text = spec_path.read_text(encoding="utf-8")
    dropped_meta = {d["debt_id"]: d for d in drop_items}
    keep_blocks = []
    dropped_details = []
    for it in items:
        meta = dropped_meta.get(it["debt_id"])
        if meta is not None:
            dropped_details.append({**it, **meta})
            continue
        keep_blocks.append(it["block"])
    new_items_block = "\n\n".join(keep_blocks)

    new_text = _SPEC_ITEMS_SECTION_RE.sub(
        lambda m: m.group(1) + new_items_block + m.group(3), text, count=1,
    )

    salvage_lines = [
        "",
        "## Salvage record",
        "",
        f"Salvaged by bundle_autodispatch enforce (gate-amended 2026-08-13, "
        f"Erah's salvage ruling) at {run_ts}. The items below were dropped as "
        f"invalid/stale per the gate's verbatim ground, and the bundle was "
        f"re-gated once. No item disappears silently: ledger status flips to "
        f"`invalid` and a source-PR comment is owed (not yet posted by this "
        f"unit — see the enforce record).",
        "",
    ]
    for d in dropped_details:
        salvage_lines.append(
            f"- **Debt ID:** `{d['debt_id']}` — **Ground:** {ground_text.strip()} "
            f"— **Confidence:** {d['confidence']} — **Inference source:** {d['source']} "
            f"— **PR comment owed:** source PR #{d.get('source_pr', '?')}."
        )
    new_text = new_text.rstrip("\n") + "\n" + "\n".join(salvage_lines) + "\n"

    spec_path.write_text(new_text, encoding="utf-8")
    return dropped_details


# ---------------------------------------------------------------------------
# Item-level triage (bundle-item-level-triage-v0): classify each bundle item
# mechanical | fork BEFORE the full spec-review gate runs. Mechanical items
# are cooked into their own fix spec and bound directly (advisory, no
# council); fork items are left in the bundle — its '## Items' section is
# amended to drop the mechanical ones — so they reach the existing full gate
# exactly as before. Fully deterministic (no model calls in v0), so it runs
# ahead of the GW-liveness check: mechanical items bind even when GW is down,
# which is exactly the defer-everything path this unit fixes.
# ---------------------------------------------------------------------------

def _amend_spec_drop_mechanical_items(
    spec_path: Path,
    items: list[dict],
    dropped_debt_ids: list[str],
    triage_by_debt_id: dict[str, dict],
    run_ts: str,
) -> None:
    """Rewrite spec_path in place: drop items bound mechanically from
    '## Items', append a '## Triage record' section naming each mechanical
    item's cooked spec, target id, and bind outcome. Mirrors
    _amend_spec_salvage's minimal-touch style but never touches the debt
    ledger — a mechanical item is being FIXED, not invalidated."""
    text = spec_path.read_text(encoding="utf-8")
    dropped = set(dropped_debt_ids)
    keep_blocks = [it["block"] for it in items if it["debt_id"] not in dropped]
    new_items_block = "\n\n".join(keep_blocks)

    new_text = _SPEC_ITEMS_SECTION_RE.sub(
        lambda m: m.group(1) + new_items_block + m.group(3), text, count=1,
    )

    lines = [
        "",
        "## Triage record",
        "",
        f"bundle-item-level-triage-v0 classified the items below `mechanical` "
        f"at {run_ts} and bound each directly (advisory, no council) via its "
        f"own cooked fix spec. See each cooked spec's Definition of Done for "
        f"the Verification Contract.",
        "",
    ]
    for debt_id in dropped_debt_ids:
        rec = triage_by_debt_id.get(debt_id, {})
        lines.append(
            f"- **Debt ID:** `{debt_id}` — **Cooked spec:** "
            f"`{rec.get('cooked_spec', '?')}` — **Target:** "
            f"`{rec.get('target_id', '?')}` — **Bound:** {rec.get('bound')}"
        )
    new_text = new_text.rstrip("\n") + "\n" + "\n".join(lines) + "\n"

    spec_path.write_text(new_text, encoding="utf-8")


def _triage_bundle_items(
    spec_id: str, repo: str, spec_path: Path, run_ts: str, results: dict,
    deadline_monotonic: float | None = None,
) -> None:
    """Classify every bundle item mechanical | fork, cook + bind the
    mechanical ones directly, and amend the spec to drop them so the
    existing full-gate path below only ever sees fork items. A spec with no
    parseable '## Items' section is a no-op — the pre-this-unit gate path
    runs completely unchanged.

    `deadline_monotonic` is the per-reconcile-run baseline budget deadline
    (verification-contract-main-baseline-v0) — threaded into every
    extract_verification_contract / verify_behavioral_invariance call below
    so a run that has exhausted BASELINE_RUN_BUDGET_S fails closed
    (unverified/budget_exhausted -> fork) rather than running indefinitely."""
    from lapis_pm import bundle_triage

    text = spec_path.read_text(encoding="utf-8")
    items = _parse_bundle_items(text)
    if not items:
        return

    route_records: list[dict] = []
    triage_by_debt_id: dict[str, dict] = {}
    mechanical_bound_ids: list[str] = []

    for item in items:
        item["repo"] = repo
        debt_id = item["debt_id"]
        cls, reason = bundle_triage.classify_item(item)

        if cls == "mechanical":
            # Extraction-only Verification Contract: lives in the triage
            # record and is rendered VERBATIM into the cooked spec's DoD by
            # cook_item_to_spec (rendering, not analysis — see its docstring).
            # Sets item["verification_baseline"] as a side effect (additive
            # baseline record — verification-contract-main-baseline-v0),
            # whether or not a contract was extracted, so a tier-3 fork
            # re-route below still carries its categorized baseline reason.
            contract = bundle_triage.extract_verification_contract(
                item, deadline_monotonic=deadline_monotonic,
            )
            if contract is None:
                cls, reason = "fork", "invariance"
            else:
                item["verification_contract"] = contract
                target_id = bundle_triage.mechanical_item_target_id(repo, debt_id)
                cooked_path = bundle_triage.cook_item_to_spec(item)

                # Tier 3 — the POST-COOK BIND-GATE VERIFIER (not a
                # classification-time tier): re-confirms the contract still
                # holds now that the spec is cooked, and BEFORE the
                # mechanical bind below fires. A miss here re-routes to
                # fork/invariance — never a silent bind.
                holds, _ = bundle_triage.verify_behavioral_invariance(
                    item, deadline_monotonic=deadline_monotonic,
                )
                if not holds:
                    cls, reason = "fork", "invariance"
                else:
                    bound_ok = _bind(target_id, repo, cooked_path)
                    ticked_ok = bound_ok and _tick(target_id, repo)
                    verified_ok = ticked_ok and _verify_dispatched(target_id)

                    rec = {
                        "debt_id": debt_id, "class": "mechanical", "reason": None,
                        "target_id": target_id, "cooked_spec": str(cooked_path),
                        "contract": contract, "bound": verified_ok,
                        "baseline": item.get("verification_baseline"),
                    }
                    route_records.append(rec)
                    triage_by_debt_id[debt_id] = rec
                    results["triage"].append({"spec": spec_id, **rec})
                    logger.info(
                        "[bundle-autodispatch] %s item %s: mechanical → cooked %s, bound=%s",
                        spec_id, debt_id, cooked_path.name, verified_ok,
                    )
                    if verified_ok:
                        mechanical_bound_ids.append(debt_id)
                        results["bound"].append({
                            "spec": target_id, "repo": repo, "mechanical": True,
                            "source_bundle": spec_id, "debt_id": debt_id,
                        })
                    else:
                        results["failed"].append({
                            "spec": target_id, "reason": "mechanical_bind_failed",
                            "source_bundle": spec_id, "debt_id": debt_id,
                        })
                    continue

        # fork — either from classify_item directly, or a tier-3 re-route.
        # `baseline` is None for a static/blast-radius fork (no baseline was
        # ever attempted) and populated for an invariance re-route (Erah's
        # audit of every fork carries its categorized baseline reason).
        rec = {
            "debt_id": debt_id, "class": "fork", "reason": reason,
            "baseline": item.get("verification_baseline"),
        }
        route_records.append(rec)
        triage_by_debt_id[debt_id] = rec
        results["triage"].append({"spec": spec_id, **rec})
        logger.info(
            "[bundle-autodispatch] %s item %s: fork (%s) — routed to Erah via the full gate",
            spec_id, debt_id, reason,
        )

    bundle_triage.write_route_record(spec_path, repo, route_records)

    if mechanical_bound_ids:
        _amend_spec_drop_mechanical_items(
            spec_path, items, mechanical_bound_ids, triage_by_debt_id, run_ts,
        )


def _mark_debt_invalid(repo: str, debt_id: str, ground_text: str) -> bool:
    """Flip review/debt/<repo>/<debt_id>'s status to invalid via direct mem
    keys (code_reviewer is NOT importable from the deploy python — the
    write-back must go through agents_core.mem.MemoryStore keys directly,
    mirroring the tag convention code_reviewer/memory.py's
    update_debt_status already uses). Never raises — a ledger write-back
    failure must not abort the salvage that already amended the spec; it is
    logged loudly instead."""
    try:
        import yaml as _yaml
        from . import node_identity

        store = node_identity.writable_store()
        key = f"review/debt/{repo}/{debt_id}"
        rec = store.get(key)
        if not rec:
            logger.warning(
                "[bundle-autodispatch] salvage: debt ledger key %s not found for write-back",
                key,
            )
            return False
        body = _yaml.safe_load(rec.get("content", "")) or {}
        body["status"] = "invalid"
        body["invalid_ground"] = ground_text[:500]
        store.set(
            key,
            _yaml.safe_dump(body, sort_keys=False),
            tags=["review-debt", f"repo-{repo}", "debt-invalid"],
            source="bundle-autodispatch-enforce",
        )
        return True
    except Exception as exc:
        logger.warning(
            "[bundle-autodispatch] salvage: ledger write-back failed for %s/%s: %s",
            repo, debt_id, exc,
        )
        return False


def _handle_infra(
    spec_id: str,
    repo: str,
    spec_path: Path,
    brief,
    ground: str,
    gate_timeout_s: int,
    run_ts: str,
    results: dict,
) -> None:
    """INFRA: retry the gate ONCE, and ONLY when the fault's own text leaves
    room for transient recovery. Otherwise zero retries, straight to
    record. Never 'deferring to human', never bind — a faulted spec lands
    in the `faulted` bucket, not `deferred`.

    A mechanically-successful retry (a brief that is neither proceed-to-bind
    nor itself infra-faulted) is a genuine gate outcome, not a fault — it is
    re-classified via _classify_enforce and dispatched to salvage/defer
    (PR #287: falling through to `faulted` here buried genuine council
    blocks in a bucket the morning brief renders as a machine hiccup,
    contradicting Ruling 1(c) — genuine council blocks are always Erah's).
    This never issues a second gate retry from this function; the one
    _run_gate call above is the hard cap."""
    retried = False
    final_ground = ground
    final_brief = brief

    if _is_transient_fault_text(ground):
        retried = True
        logger.warning(
            "[bundle-autodispatch] %s: infra fault looks transient (%s) — retrying gate ONCE",
            spec_id, ground,
        )
        retry_brief = _run_gate(spec_path, gate_timeout_s)
        if retry_brief is not None and retry_brief.combined_recommendation == "proceed-to-bind":
            logger.info(
                "[bundle-autodispatch] %s: infra retry recovered — proceeding to bind", spec_id,
            )
            _write_enforce_record(spec_id, retry_brief, "infra", ground, action="retried_then_bound")
            _do_bind_tick_verify(spec_id, repo, spec_path, run_ts, results)
            return
        final_brief = retry_brief
        retry_infra_text = _infra_fault_text(retry_brief)
        if retry_infra_text is not None:
            final_ground = retry_infra_text
        else:
            # retry_brief is guaranteed not None here — _infra_fault_text(None)
            # always returns text, so a None retry_infra_text means the retry
            # produced a real brief with no infra fault. Re-enter classification
            # on the retry's own brief/ground; do NOT record faulted.
            reclass_class, reclass_ground = _classify_enforce(retry_brief)
            logger.info(
                "[bundle-autodispatch] %s: infra retry mechanically healthy — "
                "reclassified as %s, dispatching (no further gate retry)",
                spec_id, reclass_class,
            )
            if reclass_class == "salvage":
                _handle_salvage(
                    spec_id, repo, spec_path, retry_brief, reclass_ground,
                    gate_timeout_s, run_ts, results,
                )
            else:
                _handle_defer(spec_id, retry_brief, reclass_ground, results)
            return
    else:
        logger.warning(
            "[bundle-autodispatch] %s: infra fault looks permanent (%s) — "
            "zero retries, straight to record",
            spec_id, ground,
        )

    logger.warning(
        "[bundle-autodispatch] %s: infra fault recorded (retried=%s) — "
        "never deferred to human, never bound",
        spec_id, retried,
    )
    _write_enforce_record(spec_id, final_brief, "infra", final_ground, action="faulted")
    results["faulted"].append({"spec": spec_id, "ground": final_ground[:300], "retried": retried})


def _handle_salvage(
    spec_id: str,
    repo: str,
    spec_path: Path,
    brief,
    ground: str,
    gate_timeout_s: int,
    run_ts: str,
    results: dict,
) -> None:
    """SALVAGE: amend the bundle in place (drop invalid items, ledger
    write-back), re-gate ONCE. proceed-to-bind -> bind/tick/verify.
    Anything else -> DEFER (fail-closed toward the human)."""
    marker = _salvage_attempted_marker(spec_path)
    if marker.exists():
        logger.warning(
            "[bundle-autodispatch] %s: salvage already attempted tonight — refusing a second attempt",
            spec_id,
        )
        _write_enforce_record(spec_id, brief, "salvage", ground, action="refused_second_attempt")
        results["deferred"].append({"spec": spec_id, "reason": "salvage_already_attempted"})
        return

    text = spec_path.read_text(encoding="utf-8")
    items = _parse_bundle_items(text)
    if not items:
        logger.warning(
            "[bundle-autodispatch] %s: salvage classified but no items parsed — fail-closed DEFER",
            spec_id,
        )
        _write_enforce_record(spec_id, brief, "defer", ground, action="deferred")
        results["deferred"].append({"spec": spec_id, "reason": "gate:salvage_no_items_parsed"})
        return

    drop_items = _classify_invalid_items_via_gw(ground, items)
    if not drop_items:
        reason = (
            "gate:salvage_classifier_fault" if drop_items is None
            else "gate:salvage_no_invalid_items_identified"
        )
        logger.warning(
            "[bundle-autodispatch] %s: %s — fail-closed DEFER", spec_id, reason,
        )
        _write_enforce_record(spec_id, brief, "defer", ground, action="deferred")
        results["deferred"].append({"spec": spec_id, "reason": reason})
        return

    # Claim the one salvage attempt BEFORE mutating the spec.
    marker.write_text(f"salvage-attempted: spec_id={spec_id} ts={run_ts}\n", encoding="utf-8")

    dropped_details = _amend_spec_salvage(spec_path, items, drop_items, ground, run_ts)
    logger.warning(
        "[bundle-autodispatch] %s: salvage amended spec in place, dropped %d item(s): %s",
        spec_id, len(dropped_details), [d["debt_id"] for d in dropped_details],
    )
    for d in dropped_details:
        _mark_debt_invalid(repo, d["debt_id"], ground)

    logger.info("[bundle-autodispatch] %s: re-gating amended spec (salvage, one attempt)", spec_id)
    regate_brief = _run_gate(spec_path, gate_timeout_s)

    salvage_record_items = [
        {
            "debt_id": d["debt_id"],
            "confidence": d["confidence"],
            "source": d["source"],
            "source_pr": d.get("source_pr"),
        }
        for d in dropped_details
    ]

    if regate_brief is not None and regate_brief.combined_recommendation == "proceed-to-bind":
        _write_enforce_record(
            spec_id, regate_brief, "salvage", ground, action="salvage_bound",
            salvage_dropped_items=salvage_record_items,
        )
        results["salvaged"].append({
            "spec": spec_id, "dropped_items": salvage_record_items, "outcome": "bound",
        })
        _do_bind_tick_verify(spec_id, repo, spec_path, run_ts, results)
        return

    logger.warning(
        "[bundle-autodispatch] %s: salvage re-gate did not proceed-to-bind — DEFER (fail-closed)",
        spec_id,
    )
    _write_enforce_record(
        spec_id, regate_brief, "salvage", ground, action="salvage_regate_deferred",
        salvage_dropped_items=salvage_record_items,
    )
    results["salvaged"].append({
        "spec": spec_id, "dropped_items": salvage_record_items, "outcome": "deferred",
    })
    results["deferred"].append({"spec": spec_id, "reason": "gate:salvage_regate_not_proceed_to_bind"})


def _handle_defer(spec_id: str, brief, ground: str, results: dict) -> None:
    """DEFER: council blocks, laid-down, or a genuine fork — today's defer
    path, verbatim reasons preserved."""
    rec = getattr(brief, "combined_recommendation", None) if brief is not None else None
    reason = f"gate:{rec}" if rec is not None else "gate_timeout_or_error"
    logger.warning(
        "[bundle-autodispatch] %s gate defers to human (%s) — %s", spec_id, reason, ground,
    )
    _write_enforce_record(spec_id, brief, "defer", ground, action="deferred")
    results["deferred"].append({"spec": spec_id, "reason": reason})


# ---------------------------------------------------------------------------
# Per-spec reconciler
# ---------------------------------------------------------------------------

def _reconcile_one(
    spec_path: Path,
    dry_run: bool,
    gate_timeout_s: int,
    run_ts: str,
    results: dict,
    deadline_monotonic: float | None = None,
) -> None:
    """Reconcile one spec file. Mutates results dict in-place.

    State machine (checked in order):
      1. .autodispatch present        → SKIP (completed)
      2. .autodispatch-failed present → SKIP (human triage)
      3. .autodispatch-pending present → crash detected → rename pending→failed + loud log
      4. target YAML present, no marker → externally/manually bound → SKIP + audit log. NEVER tombstone.
      5. no target YAML, no pending   → fresh → gate + bind
    """
    # Parse frontmatter
    try:
        spec_id, repo = _parse_bundle_frontmatter(spec_path)
    except ValueError as exc:
        logger.warning("[bundle-autodispatch] skipping %s: %s", spec_path.name, exc)
        results["skipped"].append({"spec": spec_path.name, "reason": str(exc)})
        return

    # 1. .autodispatch marker → already done
    if _autodispatch_marker(spec_path).exists():
        logger.info("[bundle-autodispatch] %s already has .autodispatch marker — skip", spec_id)
        results["skipped"].append({"spec": spec_id, "reason": "already_dispatched"})
        return

    # 2. .autodispatch-failed tombstone → human triage
    if _failed_marker(spec_path).exists():
        logger.warning(
            "[bundle-autodispatch] %s has .autodispatch-failed tombstone — human triage, skip",
            spec_id,
        )
        results["skipped"].append({"spec": spec_id, "reason": "failed_tombstone"})
        return

    # 3. .autodispatch-pending present → writer is gone (singleton lock guarantees this) → crash
    if _pending_marker(spec_path).exists():
        logger.error(
            "[bundle-autodispatch] !!! %s: stale .autodispatch-pending detected — "
            "prior run crashed between bind and verify; "
            "%s .autodispatch-failed tombstone for human triage",
            spec_id,
            "would write" if dry_run else "writing",
        )
        if not dry_run:
            _pending_marker(spec_path).rename(_failed_marker(spec_path))
        results["failed"].append({"spec": spec_id, "reason": "crash_pending_marker"})
        return

    # 4. Target YAML exists, no marker of any kind → externally/manually bound
    if _target_yaml_exists(spec_id):
        logger.warning(
            "[bundle-autodispatch] AUDIT: %s is externally/manually bound "
            "(target YAML exists, no autodispatch marker) — skipping, not tombstoning. "
            "This path must be observable to prevent mass-suppression of the crash guard.",
            spec_id,
        )
        results["skipped"].append({"spec": spec_id, "reason": "externally_bound"})
        return

    # 5. Fresh spec → per-item triage (bundle-item-level-triage-v0), then
    #    gate + bind over whatever fork items remain.
    logger.info("[bundle-autodispatch] processing %s (repo=%s)", spec_id, repo)

    if dry_run:
        gw_ok = _gw_serving()
        if not gw_ok:
            logger.info("[bundle-autodispatch] [dry-run] GW not serving — would defer %s", spec_id)
            results["deferred"].append({"spec": spec_id, "reason": "gw_not_serving"})
        else:
            logger.info("[bundle-autodispatch] [dry-run] would run gate + bind+tick for %s", spec_id)
            results["bound"].append({"spec": spec_id, "repo": repo, "dry_run": True})
        return

    # Item-level triage: fully deterministic (no model calls in v0), so it
    # runs ahead of the GW-liveness check below — mechanical items bind even
    # when GW is down. Amends spec_path in place to drop mechanical items
    # before the gate (and every downstream read of spec_path) ever sees it.
    _triage_bundle_items(spec_id, repo, spec_path, run_ts, results, deadline_monotonic)

    # GW liveness pre-check
    #
    # This is the bundle path's GW-down DEFER site (unit contract: "GW-local
    # only. If GW is not serving, each spec is deferred (no paid fallback)").
    # It fires BEFORE _run_gate, so the facets leg (run_spec_review,
    # facets_operator default "gravitywell" since the 2026-09-16 Claude-gone
    # re-point, lapis-pm-bundle-gate-facets-local-default-v0) can never start
    # when GW is down — a gravitywell operator that would otherwise have to
    # degrade to the dead haiku tier when GW is not serving never runs. No
    # code change needed here; the pre-check IS the defer.
    if not _gw_serving():
        logger.warning(
            "[bundle-autodispatch] GW not serving for %s — deferring to human "
            "(Pushover already fired from debt-bundle sweep)",
            spec_id,
        )
        results["deferred"].append({"spec": spec_id, "reason": "gw_not_serving"})
        return

    # Spec-review gate
    logger.info(
        "[bundle-autodispatch] running spec-review gate for %s (timeout=%ds)",
        spec_id, gate_timeout_s,
    )
    brief = _run_gate(spec_path, gate_timeout_s)

    if brief is not None:
        logger.info(
            "[bundle-autodispatch] gate recommendation for %s: %s",
            spec_id, brief.combined_recommendation,
        )

    # Direct proceed-to-bind — unchanged crash-guarded bind/tick/verify path.
    if brief is not None and brief.combined_recommendation == "proceed-to-bind":
        _do_bind_tick_verify(spec_id, repo, spec_path, run_ts, results)
        return

    # Three-class enforce (lapis-pm-bundle-autodispatch-enforce-v0): everything
    # that is not a clean proceed-to-bind is now infra / salvage / defer,
    # replacing the old blanket "defer to human" for both brief-is-None and
    # rec != proceed-to-bind.
    enforce_class, ground = _classify_enforce(brief)
    logger.info(
        "[bundle-autodispatch] %s classified enforce_class=%s", spec_id, enforce_class,
    )

    if enforce_class == "infra":
        _handle_infra(spec_id, repo, spec_path, brief, ground, gate_timeout_s, run_ts, results)
        return

    if enforce_class == "salvage":
        _handle_salvage(spec_id, repo, spec_path, brief, ground, gate_timeout_s, run_ts, results)
        return

    _handle_defer(spec_id, brief, ground, results)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def reconcile(
    spec_dir: Path | None = None,
    dry_run: bool = False,
    gate_timeout_s: int = _DEFAULT_GATE_TIMEOUT_S,
    run_ts: str = "",
) -> dict:
    """Reconcile unbound cr-bundle-* specs.

    run_ts is the caller-supplied ISO timestamp for pending marker provenance.
    It must be supplied by the CLI entry point, never generated inside this function.

    Returns a results dict with keys: bound, deferred, skipped, failed,
    faulted, salvaged, triage. Each value is a list of dicts describing the
    outcome for each spec. `faulted` (infra) and `salvaged` are cross-cutting
    descriptors, not exclusive partitions — a salvaged spec that goes on to
    bind appears in both `bound` and `salvaged`; one that fails its re-gate
    appears in both `deferred` and `salvaged` (see _handle_salvage).
    `triage` (bundle-item-level-triage-v0) is per-ITEM, not per-spec: one
    entry per bundle item classified mechanical|fork before the gate ran.
    A mechanically-bound item's own bind/tick/verify outcome ALSO appears in
    `bound`/`failed` (item-level triage-v0 DoD-2: the run summary's `bound`
    count must rise with mechanical binds, same as any other bind).
    """
    if spec_dir is None:
        spec_dir = _DEFAULT_SPEC_DIR

    results: dict = {
        "bound": [], "deferred": [], "skipped": [], "failed": [],
        "faulted": [], "salvaged": [], "triage": [],
    }

    # verification-contract-main-baseline-v0: the stale-worktree prune pass
    # runs ONCE here, at run start, before any item-level baseline — never
    # at individual worktree-creation time (gate hardening fold: unpruned
    # stale worktrees under /tmp/lapis-pm-baselines/ are a disk-exhaustion
    # cascade into `unverified` for every item in the run). The per-run
    # baseline budget deadline is computed once here too and threaded
    # through every spec's item-level triage below.
    from lapis_pm import bundle_triage
    pruned = bundle_triage.prune_stale_baseline_worktrees()
    if pruned:
        logger.info(
            "[bundle-autodispatch] pruned %d stale baseline worktree(s): %s",
            len(pruned), ", ".join(pruned),
        )
    deadline_monotonic = time.monotonic() + bundle_triage.BASELINE_RUN_BUDGET_S

    specs = _discover_specs(spec_dir)
    logger.info(
        "[bundle-autodispatch] discovered %d cr-bundle debt-bundle spec(s) in %s", len(specs), spec_dir,
    )

    for spec_path in specs:
        _reconcile_one(spec_path, dry_run, gate_timeout_s, run_ts, results, deadline_monotonic)

    # loupe's navigator (loupe/navigator/subjects.py) greps the service
    # journal for the literal "run complete: bound=" substring and regexes
    # bound=(\d+) to judge subject health — bound= MUST stay first and the
    # existing buckets MUST never reorder/rename. New buckets are APPENDED
    # only (see tests/test_bundle_autodispatch_enforce.py's sentinel test).
    logger.info(
        "[bundle-autodispatch] run complete: bound=%d deferred=%d skipped=%d failed=%d faulted=%d salvaged=%d",
        len(results["bound"]),
        len(results["deferred"]),
        len(results["skipped"]),
        len(results["failed"]),
        len(results["faulted"]),
        len(results["salvaged"]),
    )

    # bundle-item-level-triage-v0 deliverable (b): per-item triage counts in
    # the run summary. A SEPARATE line — never appended to the pinned
    # "run complete: bound=..." sentinel line above (loupe's navigator
    # regexes that line verbatim; see the comment on it).
    #
    # verification-contract-main-baseline-v0: baseline state counts are
    # APPENDED to this same free-form line only — the pinned sentinel line
    # above stays byte-identical.
    mechanical = [t for t in results["triage"] if t["class"] == "mechanical"]
    fork = [t for t in results["triage"] if t["class"] == "fork"]
    baseline_counts = {"green": 0, "annotated": 0, "unverified": 0}
    for t in results["triage"]:
        b = t.get("baseline")
        if b and b.get("state") in baseline_counts:
            baseline_counts[b["state"]] += 1
    logger.info(
        "[bundle-autodispatch] triage complete: mechanical=%d (bound=%d) fork=%d "
        "baseline=green:%d annotated:%d unverified:%d",
        len(mechanical),
        sum(1 for t in mechanical if t.get("bound")),
        len(fork),
        baseline_counts["green"],
        baseline_counts["annotated"],
        baseline_counts["unverified"],
    )
    return results
