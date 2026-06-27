"""bundle_autodispatch — reconcile unbound cr-bundle-* specs and bind+tick qualifying ones.

Runs nightly after code-reviewer's debt-bundle sweep. Discovers cr-bundle-*.md specs in
/srv/lapis/planning/specs (excluding superseded/), runs the GW spec-review gate on each unbound
one, and binds+ticks the ones that pass with advisory authority.

Idempotency:
  - .autodispatch marker (sibling to spec): written only after dispatched:1 is verified.
  - .autodispatch-failed tombstone: written when bind succeeds but tick does not verify.
  - Specs with either marker are skipped on re-run.
  - A spec whose target YAML exists but whose .autodispatch marker is missing is the
    partial/crashed state → writes .autodispatch-failed for human triage (no double-bind).

GW-serving:
  - swarm_serving() is checked BEFORE calling run_spec_review.
  - If GW is not serving: defer (no bind, no marker, loud log). Never fall back to paid.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
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


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_specs(spec_dir: Path) -> list[Path]:
    """Return sorted cr-bundle-*.md spec paths, excluding superseded/."""
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
        result.append(p)
    return result


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------

def _autodispatch_marker(spec_path: Path) -> Path:
    return Path(str(spec_path) + ".autodispatch")


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

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_spec_review,
            spec_path=spec_path,
            council_voicing="gravitywell",
            timeout_s=timeout_s,
            authority="advisory",
            dispatch_facets=True,
            sonnet_reviewer=True,
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
    results: dict,
) -> None:
    """Reconcile one spec file. Mutates results dict in-place."""
    # Parse frontmatter
    try:
        spec_id, repo = _parse_bundle_frontmatter(spec_path)
    except ValueError as exc:
        logger.warning("[bundle-autodispatch] skipping %s: %s", spec_path.name, exc)
        results["skipped"].append({"spec": spec_path.name, "reason": str(exc)})
        return

    # .autodispatch marker → already done
    if _autodispatch_marker(spec_path).exists():
        logger.info("[bundle-autodispatch] %s already has .autodispatch marker — skip", spec_id)
        results["skipped"].append({"spec": spec_id, "reason": "already_dispatched"})
        return

    # .autodispatch-failed tombstone → human triage
    if _failed_marker(spec_path).exists():
        logger.warning(
            "[bundle-autodispatch] %s has .autodispatch-failed tombstone — human triage, skip",
            spec_id,
        )
        results["skipped"].append({"spec": spec_id, "reason": "failed_tombstone"})
        return

    # Target YAML exists but no .autodispatch marker → partial bind (crash between bind + verify)
    if _target_yaml_exists(spec_id):
        logger.error(
            "[bundle-autodispatch] !!! %s: target YAML exists but .autodispatch marker is missing "
            "— crash after bind detected; writing .autodispatch-failed tombstone for human triage",
            spec_id,
        )
        if not dry_run:
            _failed_marker(spec_path).write_text(
                "autodispatch: target YAML exists but .autodispatch marker was never written "
                "(likely crashed between bind and verify)\n",
                encoding="utf-8",
            )
        results["failed"].append({"spec": spec_id, "reason": "partial_bind_no_marker"})
        return

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

    # Bind
    logger.info(
        "[bundle-autodispatch] binding %s → repo=%s authority=advisory", spec_id, repo,
    )
    if not _bind(spec_id, repo, spec_path):
        logger.error("[bundle-autodispatch] bind failed for %s — deferring", spec_id)
        results["deferred"].append({"spec": spec_id, "reason": "bind_failed"})
        return

    # Tick (initial fixer dispatch)
    logger.info("[bundle-autodispatch] firing initial tick for %s", spec_id)
    if not _tick(spec_id, repo):
        logger.error(
            "[bundle-autodispatch] !!! %s tick failed after bind — writing .autodispatch-failed tombstone",
            spec_id,
        )
        _failed_marker(spec_path).write_text(
            "autodispatch: bind succeeded but force_dispatch raised\n",
            encoding="utf-8",
        )
        results["failed"].append({"spec": spec_id, "reason": "tick_failed"})
        return

    # Verify dispatched:1
    if not _verify_dispatched(spec_id):
        logger.error(
            "[bundle-autodispatch] !!! %s dispatched<1 after tick — writing .autodispatch-failed tombstone",
            spec_id,
        )
        _failed_marker(spec_path).write_text(
            "autodispatch: bind+tick succeeded but dispatched<1 at verify\n",
            encoding="utf-8",
        )
        results["failed"].append({"spec": spec_id, "reason": "dispatch_not_verified"})
        return

    # Success: write .autodispatch marker
    _autodispatch_marker(spec_path).write_text(
        f"autodispatched: spec_id={spec_id} repo={repo}\n",
        encoding="utf-8",
    )
    logger.info(
        "[bundle-autodispatch] successfully bound+ticked+verified %s — .autodispatch marker written",
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
) -> dict:
    """Reconcile unbound cr-bundle-* specs.

    Returns a results dict with keys: bound, deferred, skipped, failed.
    Each value is a list of dicts describing the outcome for each spec.
    """
    if spec_dir is None:
        spec_dir = _DEFAULT_SPEC_DIR

    results: dict = {"bound": [], "deferred": [], "skipped": [], "failed": []}

    specs = _discover_specs(spec_dir)
    logger.info(
        "[bundle-autodispatch] discovered %d cr-bundle spec(s) in %s", len(specs), spec_dir,
    )

    for spec_path in specs:
        _reconcile_one(spec_path, dry_run, gate_timeout_s, results)

    logger.info(
        "[bundle-autodispatch] run complete: bound=%d deferred=%d skipped=%d failed=%d",
        len(results["bound"]),
        len(results["deferred"]),
        len(results["skipped"]),
        len(results["failed"]),
    )
    return results
