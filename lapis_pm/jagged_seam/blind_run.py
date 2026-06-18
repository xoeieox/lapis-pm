"""Blind-run driver for the jagged-seam arbiter: U3 falsification test.

U3: forced-extremal-articulation on a fresh Q2 state. Three extrospective GW passes
(fear / desire / balanced control) over the SAME state, overlay, adversarial novelty
verification, and complete artifact generation for PM hand-off.

The driver is a library function (run_blind) that never calls sys.exit, paired with
a thin runnable entrypoint that owns exit semantics.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from . import (
    from_prose,
    overlay,
    PoleClaim,
    PoleOutput,
    JaggedSeamGravityWellUnavailable,
)

log = logging.getLogger(__name__)

STATE_REF = "zephyr-defederation-teeth-2026-06-17"


def _check_neutrality(state_doc_path: Path) -> tuple[bool, list[str]]:
    """Check state doc for pre-loaded answer terminology (neutrality guard).

    Returns:
        (is_neutral, flagged_tokens) where is_neutral=True if doc passes,
        flagged_tokens is a list of disqualifying terms found (case-insensitive).
    """
    try:
        content = state_doc_path.read_text()
    except Exception as e:
        log.error(f"Could not read state doc for neutrality check: {e}")
        raise

    # Disqualifying terms (case-insensitive).
    disqualifying = [
        "decision_variable",
        "decision-variable",
        "seam",
    ]

    content_lower = content.lower()
    flagged = []
    for term in disqualifying:
        if term in content_lower:
            flagged.append(term)

    is_neutral = len(flagged) == 0
    return is_neutral, flagged


def run_blind(
    state_doc: str = "/srv/lapis/jagged-seam/q2-defederation-teeth-state.md",
    out_dir: str = "/srv/lapis/jagged-seam/runs/",
    *,
    operator_class: str = "gravitywell",
) -> dict[str, Any]:
    """Run the U3 blind test: three extrospective GW passes, overlay, novelty verification.

    Args:
        state_doc: Path to PM-authored Q2 state document (no pre-stated seam).
        out_dir: Path to write run artifacts. Created if missing.
        operator_class: LLM operator to use (default "gravitywell").

    Returns:
        Dict with keys: run_id, new_variable_count, metric_result, novelty_analysis,
        and per-pass details.

    Raises:
        FileNotFoundError: If state_doc does not exist or is empty.
        ValueError: If state_doc fails neutrality check.
        JaggedSeamGravityWellUnavailable: If GW is unavailable (no paid fallback).
    """
    state_doc_path = Path(state_doc).expanduser().resolve()
    out_dir_path = Path(out_dir).expanduser().resolve()

    start_time = time.time()

    # Input verification: state_doc exists and is non-empty.
    if not state_doc_path.exists():
        raise FileNotFoundError(f"State doc not found: {state_doc_path}")
    if state_doc_path.stat().st_size == 0:
        raise FileNotFoundError(f"State doc is empty: {state_doc_path}")

    log.info(f"Loading state doc from {state_doc_path}")
    state_doc_content = state_doc_path.read_text()
    state_doc_sha = hashlib.sha256(state_doc_content.encode()).hexdigest()

    # Neutrality check (precondition, loud abort).
    log.info("Running neutrality check on state doc...")
    is_neutral, flagged_tokens = _check_neutrality(state_doc_path)

    neutrality_check_result = {
        "passed": is_neutral,
        "flagged_tokens": flagged_tokens,
    }

    if not is_neutral:
        log.error(f"Neutrality check FAILED. Flagged tokens: {flagged_tokens}")
        raise ValueError(
            f"State doc contains disqualifying terminology: {flagged_tokens}. "
            "Run aborts before any GW call."
        )

    log.info("Neutrality check PASSED")

    # Import call_operator here (lazy load, like arbiter).
    try:
        from agents_core.llm import call_operator  # type: ignore[import]
    except ImportError:
        log.error("agents_core.llm not available — cannot route to operator")
        raise JaggedSeamGravityWellUnavailable(
            "agents_core.llm not available"
        ) from None

    # THREE GW PASSES over the SAME state doc (extrospective).
    log.info("Running fear pass (Scout-role constraint)...")
    fear_response = _run_fear_pass(
        call_operator, state_doc_content, operator_class
    )
    if fear_response is None:
        # GW unavailable
        _write_skip_artifact(out_dir_path)
        raise JaggedSeamGravityWellUnavailable("GW unavailable on fear pass")

    log.info("Running desire pass (Backcaster-role constraint)...")
    desire_response = _run_desire_pass(
        call_operator, state_doc_content, operator_class
    )
    if desire_response is None:
        # GW unavailable
        _write_skip_artifact(out_dir_path)
        raise JaggedSeamGravityWellUnavailable("GW unavailable on desire pass")

    log.info("Running control pass (balanced)...")
    control_response = _run_control_pass(
        call_operator, state_doc_content, operator_class
    )
    if control_response is None:
        # GW unavailable
        _write_skip_artifact(out_dir_path)
        raise JaggedSeamGravityWellUnavailable("GW unavailable on control pass")

    log.info(f"Fear pass returned {len(fear_response)} claims")
    log.info(f"Desire pass returned {len(desire_response)} claims")
    log.info(f"Control pass returned {len(control_response)} variables")

    # Build poles from responses (fidelity="verified" for live GW outputs).
    fear_pole = from_prose(
        text_items=fear_response,
        kind="fear",
        state_ref=STATE_REF,
        provenance_prefix="gw:fear-pass",
        fidelity="verified",
    )

    desire_pole = from_prose(
        text_items=desire_response,
        kind="desire",
        state_ref=STATE_REF,
        provenance_prefix="gw:desire-pass",
        fidelity="verified",
    )

    # Run overlay (ONE GW call).
    log.info("Running overlay...")
    try:
        surface = overlay(fear_pole, desire_pole, operator_class=operator_class)
    except JaggedSeamGravityWellUnavailable as e:
        log.error(f"GW unavailable on overlay: {e}")
        _write_skip_artifact(out_dir_path)
        raise

    # Check skipped_no_axis flag.
    if surface.skipped_no_axis:
        log.warning("Overlay returned skipped_no_axis=True; no seam formed.")
        # Create output directory and write artifacts anyway.
        out_dir_path.mkdir(parents=True, exist_ok=True)
        run_id = _gen_run_id()
        run_path = out_dir_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        # Write pass artifacts.
        _save_yaml(run_path / "fear_pass.yaml", {
            "claims": fear_response,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        _save_yaml(run_path / "desire_pass.yaml", {
            "claims": desire_response,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        _save_yaml(run_path / "control_pass.yaml", {
            "claims": control_response,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        _save_yaml(run_path / "constraint_surface.yaml", {
            "state_ref": surface.state_ref,
            "skipped_no_axis": True,
            "items": [],
            "dropped_provenance_count": 0,
        })

        # Write run metadata.
        elapsed = time.time() - start_time
        arbiter_sha = _get_arbiter_git_sha()
        run_metadata = {
            "state_ref": STATE_REF,
            "state_doc": str(state_doc_path),
            "state_doc_sha256": state_doc_sha,
            "operator": operator_class,
            "arbiter_git_sha": arbiter_sha,
            "elapsed_seconds": elapsed,
            "neutrality_check": neutrality_check_result,
            "skipped_no_axis": True,
            "dropped_provenance_count": 0,
            "new_variable_count": 0,
            "metric_result": "not_supported",
        }
        _save_yaml(run_path / "run.yaml", run_metadata)
        _save_yaml(run_path / "novelty_analysis.yaml", {
            "note": "skipped_no_axis=True; no seam formed",
            "items": [],
        })

        log.info(f"Run ID: {run_id} (skipped_no_axis)")
        return {
            "run_id": run_id,
            "metric_result": "not_supported",
            "new_variable_count": 0,
            "skipped_no_axis": True,
            "novelty_analysis": [],
        }

    # ADVERSARIAL VERIFICATION: for each surface item, ask if the decision_variable is novel.
    log.info(f"Running adversarial novelty verification on {len(surface.items)} items...")
    novelty_analysis = []
    new_variable_count = 0

    for i, item in enumerate(surface.items):
        log.info(f"Verifying item {i+1}/{len(surface.items)}: {item.decision_variable[:60]}...")

        verification = _run_novelty_verification(
            call_operator,
            item,
            fear_response,
            desire_response,
            control_response,
            operator_class,
        )

        # Conservative default: treat call failure or uncertain response as NOT novel.
        is_novel = verification.get("novel", False)
        if is_novel:
            new_variable_count += 1

        # Record per-item analysis including the cited source claim text (operative-shift capture).
        cited_source_text = ""
        if verification.get("cited_source"):
            cited_source = verification["cited_source"]
            # Find the source claim in fear/desire responses.
            if cited_source in [c.provenance for c in fear_pole.claims]:
                claim_obj = next(c for c in fear_pole.claims if c.provenance == cited_source)
                cited_source_text = claim_obj.claim
            elif cited_source in [c.provenance for c in desire_pole.claims]:
                claim_obj = next(c for c in desire_pole.claims if c.provenance == cited_source)
                cited_source_text = claim_obj.claim

        novelty_analysis.append({
            "decision_variable": item.decision_variable,
            "novel": is_novel,
            "cited_source": verification.get("cited_source"),
            "cited_source_text": cited_source_text,
            "fear_source_provenance": item.fear_source.provenance,
            "desire_source_provenance": item.desire_source.provenance,
        })

    log.info(f"Novelty verification complete: {new_variable_count} novel variables")

    # Determine metric result.
    metric_result = "supported" if new_variable_count >= 1 else "not_supported"

    # Create output directory.
    out_dir_path.mkdir(parents=True, exist_ok=True)
    run_id = _gen_run_id()
    run_path = out_dir_path / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    # Write pass artifacts.
    _save_yaml(run_path / "fear_pass.yaml", {
        "claims": fear_response,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    _save_yaml(run_path / "desire_pass.yaml", {
        "claims": desire_response,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    _save_yaml(run_path / "control_pass.yaml", {
        "claims": control_response,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    # Write constraint surface.
    _save_yaml(run_path / "constraint_surface.yaml", {
        "state_ref": surface.state_ref,
        "skipped_no_axis": False,
        "dropped_provenance_count": surface.dropped_provenance_count,
        "items": [
            {
                "decision_variable": item.decision_variable,
                "fear_source_provenance": item.fear_source.provenance,
                "desire_source_provenance": item.desire_source.provenance,
                "crossing_type": item.crossing_type,
            }
            for item in surface.items
        ],
    })

    # Write novelty analysis (per-item with operative-shift capture).
    _save_yaml(run_path / "novelty_analysis.yaml", {
        "items": novelty_analysis,
    })

    # Write run metadata.
    elapsed = time.time() - start_time
    arbiter_sha = _get_arbiter_git_sha()
    run_metadata = {
        "state_ref": STATE_REF,
        "state_doc": str(state_doc_path),
        "state_doc_sha256": state_doc_sha,
        "operator": operator_class,
        "arbiter_git_sha": arbiter_sha,
        "elapsed_seconds": elapsed,
        "neutrality_check": neutrality_check_result,
        "skipped_no_axis": False,
        "dropped_provenance_count": surface.dropped_provenance_count,
        "new_variable_count": new_variable_count,
        "metric_result": metric_result,
    }
    _save_yaml(run_path / "run.yaml", run_metadata)

    log.info(f"Wrote run artifacts to {run_path}")
    log.info(f"Blind-run completed. Run ID: {run_id}, metric_result: {metric_result}")

    return {
        "run_id": run_id,
        "metric_result": metric_result,
        "new_variable_count": new_variable_count,
        "skipped_no_axis": False,
        "novelty_analysis": novelty_analysis,
        "output_dir": str(run_path),
    }


def _run_fear_pass(
    call_operator: Any,
    state_doc_content: str,
    operator_class: str,
) -> list[str] | None:
    """Run the fear pass (Scout-role constraint).

    Returns:
        List of failure claims (strings), or None if GW unavailable.
    """
    user_message = f"""You are the FEAR pole (Scout-role) for a forced-extremal-articulation analysis.

**SITUATION (the catalogued state):**

{state_doc_content}

**Your constraint:** Defederation may NOT bite — the enforcement of last resort is off the table.

**Your task:** Enumerate exactly what makes the defederation threat empty / what lets a large platform ignore it with impunity, given this situation.

**Output:** Return ONLY a JSON array of plain strings (failure claims / vulnerability vectors). Each string is one failure claim. Do not narrate or explain; output is ONLY the JSON array, nothing else.

Example:
["The enforcement mechanism requires unanimous coordination, which is unstable", "Platforms can simply fork and ignore the federation", ...]
"""

    system_message = "You are a fear-pole analyst running an extremal-articulation pass. Your role is to identify what makes a governance mechanism fail under the constraint given."

    result = call_operator(
        operator_class,
        prompt=user_message,
        system=system_message,
        json_mode=True,
        timeout=300,
        on_wake_fail="skip",
    )

    if result is None:
        return None

    try:
        claims = json.loads(str(result))
        if not isinstance(claims, list):
            raise ValueError("Expected JSON array")
        # Ensure all items are strings.
        claims = [str(c) for c in claims]
        return claims
    except (json.JSONDecodeError, ValueError) as e:
        log.error(f"Failed to parse fear pass response: {e}")
        raise


def _run_desire_pass(
    call_operator: Any,
    state_doc_content: str,
    operator_class: str,
) -> list[str] | None:
    """Run the desire pass (Backcaster-role constraint).

    Returns:
        List of success-mechanic claims (strings), or None if GW unavailable.
    """
    user_message = f"""You are the DESIRE pole (Backcaster-role) for a forced-extremal-articulation analysis.

**SITUATION (the catalogued state):**

{state_doc_content}

**Your constraint:** Defederation MUST be a credible deterrent large platforms genuinely fear.

**Your task:** Derive exactly the mechanics that give it real teeth — the success preconditions, given this situation.

**Output:** Return ONLY a JSON array of plain strings (success-mechanic claims). Each string is one success precondition or mechanic. Do not narrate or explain; output is ONLY the JSON array, nothing else.

Example:
["Credible enforcement requires distributed decision-making that no single platform can veto", "The costs of defederation must exceed the benefits of non-compliance", ...]
"""

    system_message = "You are a desire-pole analyst running an extremal-articulation pass. Your role is to derive the precise mechanics that make a governance system work under the constraint given."

    result = call_operator(
        operator_class,
        prompt=user_message,
        system=system_message,
        json_mode=True,
        timeout=300,
        on_wake_fail="skip",
    )

    if result is None:
        return None

    try:
        claims = json.loads(str(result))
        if not isinstance(claims, list):
            raise ValueError("Expected JSON array")
        # Ensure all items are strings.
        claims = [str(c) for c in claims]
        return claims
    except (json.JSONDecodeError, ValueError) as e:
        log.error(f"Failed to parse desire pass response: {e}")
        raise


def _run_control_pass(
    call_operator: Any,
    state_doc_content: str,
    operator_class: str,
) -> list[str] | None:
    """Run the balanced control pass (no constraint, both sides).

    Returns:
        List of decision-variables (strings), or None if GW unavailable.
    """
    user_message = f"""You are running a balanced analysis (both fear and desire) on a governance question.

**SITUATION (the catalogued state):**

{state_doc_content}

**Your task:** Analyze this situation and surface the decision-variables that matter for whether defederation has real teeth. Be balanced and comprehensive. Think about both what could make it fail AND what would make it succeed.

**Output:** Return ONLY a JSON array of plain strings (decision-variables). Each string names one variable that affects the outcome. Do not narrate or explain; output is ONLY the JSON array, nothing else.

Example:
["Degree of platform-enforcement coordination required", "Economic incentives for platforms to accept defederation", ...]
"""

    system_message = "You are a balanced analyst. Your role is to identify the key decision-variables that determine outcome, without extremal constraint."

    result = call_operator(
        operator_class,
        prompt=user_message,
        system=system_message,
        json_mode=True,
        timeout=300,
        on_wake_fail="skip",
    )

    if result is None:
        return None

    try:
        variables = json.loads(str(result))
        if not isinstance(variables, list):
            raise ValueError("Expected JSON array")
        # Ensure all items are strings.
        variables = [str(v) for v in variables]
        return variables
    except (json.JSONDecodeError, ValueError) as e:
        log.error(f"Failed to parse control pass response: {e}")
        raise


def _run_novelty_verification(
    call_operator: Any,
    item: Any,
    fear_claims: list[str],
    desire_claims: list[str],
    control_variables: list[str],
    operator_class: str,
) -> dict[str, Any]:
    """Run adversarial novelty verification on a single surface item.

    Conservative default: on call failure or uncertain response, novel=False.

    Returns:
        Dict with keys: novel (bool), cited_source (str | None).
    """
    # Build the full lists as context for the verifier.
    fear_text = "\n".join(f"- {c}" for c in fear_claims)
    desire_text = "\n".join(f"- {c}" for c in desire_claims)
    control_text = "\n".join(f"- {v}" for v in control_variables)

    user_message = f"""You are verifying whether a proposed decision-variable is genuinely NEW or merely echoes an existing claim.

**Decision-variable to verify:**
"{item.decision_variable}"

**Fear-pole claims (what makes it fail):**
{fear_text}

**Desire-pole claims (what makes it work):**
{desire_text}

**Control-pole variables (balanced analysis):**
{control_text}

**Your task:** Is this decision-variable genuinely NEW (it falls out of the fear×desire crossing and is NOT stated in any fear claim, any desire claim, or any control variable)? Or is it a restatement of one of them?

Default to RESTATEMENT unless it is clearly novel. If a restatement, cite which source it echoes.

**Output:** Return ONLY a JSON object with two fields:
- "novel": boolean (true if genuinely new, false if restatement or uncertain)
- "cited_source": null if novel, or the TEXT of the source claim/variable it echoes if restatement

Example:
{{"novel": false, "cited_source": "Defederation MUST be a credible deterrent large platforms genuinely fear"}}

or

{{"novel": true, "cited_source": null}}
"""

    system_message = "You are a conservative verifier. Default to RESTATEMENT (novel=false) unless the variable is clearly novel and distinct from all listed sources."

    try:
        result = call_operator(
            operator_class,
            prompt=user_message,
            system=system_message,
            json_mode=True,
            timeout=300,
            on_wake_fail="skip",
        )

        # Conservative default on call failure: NOT novel.
        if result is None:
            return {"novel": False, "cited_source": None}

        verdict = json.loads(str(result))
        if not isinstance(verdict, dict):
            return {"novel": False, "cited_source": None}

        return {
            "novel": verdict.get("novel", False),
            "cited_source": verdict.get("cited_source"),
        }
    except Exception as e:
        # Conservative default: NOT novel.
        log.warning(f"Novelty verification call failed: {e}")
        return {"novel": False, "cited_source": None}


def _write_skip_artifact(out_dir_path: Path) -> None:
    """Write skip.yaml on GW unavailable (no retry)."""
    out_dir_path.mkdir(parents=True, exist_ok=True)
    skip_file = out_dir_path / "skip.yaml"
    skip_payload = {
        "skipped": True,
        "reason": "gw_unavailable",
        "ts": int(time.time()),
    }
    _save_yaml(skip_file, skip_payload)
    log.info(f"Wrote skip marker to {skip_file}")


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Save a dict to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _gen_run_id() -> str:
    """Generate a run ID from ISO timestamp with -blind suffix."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-blind")


def _get_arbiter_git_sha() -> str:
    """Get the current git SHA of the arbiter repo."""
    try:
        import os
        repo_dir = os.environ.get(
            "LAPIS_PM_REPO_DIR",
            "/srv/lapis/lapis-pm",
        )
        sha = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception as e:
        log.warning(f"Could not get arbiter git SHA: {e}")
        return "unknown"


def main() -> int:
    """Runnable entrypoint. Calls sys.exit with appropriate code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        result = run_blind()
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (FileNotFoundError, ValueError) as e:
        log.error(f"Blind run failed: {e}")
        return 1
    except JaggedSeamGravityWellUnavailable as e:
        log.error(f"GW unavailable: {e}")
        return 1
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
