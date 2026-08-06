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


# ---------------------------------------------------------------------------
# Per-spec reconciler
# ---------------------------------------------------------------------------

def _reconcile_one(
    spec_path: Path,
    dry_run: bool,
    gate_timeout_s: int,
    run_ts: str,
    results: dict,
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

    # 5. Fresh spec → gate + bind
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

    # GW liveness pre-check
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

    if brief is None:
        logger.warning(
            "[bundle-autodispatch] gate timed out or errored for %s — deferring to human",
            spec_id,
        )
        results["deferred"].append({"spec": spec_id, "reason": "gate_timeout_or_error"})
        return

    rec = brief.combined_recommendation
    logger.info("[bundle-autodispatch] gate recommendation for %s: %s", spec_id, rec)

    if rec != "proceed-to-bind":
        logger.warning(
            "[bundle-autodispatch] %s gate returned %r — deferring to human",
            spec_id, rec,
        )
        results["deferred"].append({"spec": spec_id, "reason": f"gate:{rec}"})
        return

    # Write-ahead intent marker before bind (atomic resolve: pending→success or pending→failed)
    _pending_marker(spec_path).write_text(
        f"autodispatch-pending: spec_id={spec_id} repo={repo} ts={run_ts}\n",
        encoding="utf-8",
    )
    logger.info(
        "[bundle-autodispatch] wrote .autodispatch-pending for %s (ts=%s)", spec_id, run_ts,
    )

    # Bind
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

    # Tick (initial fixer dispatch)
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

    # Verify dispatched:1
    if not _verify_dispatched(spec_id):
        logger.error(
            "[bundle-autodispatch] !!! %s dispatched<1 after tick — "
            "renaming .autodispatch-pending → .autodispatch-failed",
            spec_id,
        )
        _pending_marker(spec_path).rename(_failed_marker(spec_path))
        results["failed"].append({"spec": spec_id, "reason": "dispatch_not_verified"})
        return

    # Success: atomically rename .autodispatch-pending → .autodispatch
    _pending_marker(spec_path).rename(_autodispatch_marker(spec_path))
    logger.info(
        "[bundle-autodispatch] successfully bound+ticked+verified %s — "
        ".autodispatch-pending renamed to .autodispatch",
        spec_id,
    )
    results["bound"].append({"spec": spec_id, "repo": repo})


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

    Returns a results dict with keys: bound, deferred, skipped, failed.
    Each value is a list of dicts describing the outcome for each spec.
    """
    if spec_dir is None:
        spec_dir = _DEFAULT_SPEC_DIR

    results: dict = {"bound": [], "deferred": [], "skipped": [], "failed": []}

    specs = _discover_specs(spec_dir)
    logger.info(
        "[bundle-autodispatch] discovered %d cr-bundle debt-bundle spec(s) in %s", len(specs), spec_dir,
    )

    for spec_path in specs:
        _reconcile_one(spec_path, dry_run, gate_timeout_s, run_ts, results)

    logger.info(
        "[bundle-autodispatch] run complete: bound=%d deferred=%d skipped=%d failed=%d",
        len(results["bound"]),
        len(results["deferred"]),
        len(results["skipped"]),
        len(results["failed"]),
    )
    return results
