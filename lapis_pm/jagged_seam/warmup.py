"""Warm-up reproduction run-driver for the jagged-seam arbiter.

U2: calibration / known-positive run. Loads the archived 06-08 Backcaster outputs
and hand-transcribed fear drift-vectors, runs the arbiter, and emits the reproduction
signal and gap-gate response.

The driver is a library function (run_warmup) that never calls sys.exit, paired with
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
from typing import Any

import yaml

from . import (
    from_backcaster,
    from_prose,
    overlay,
    JaggedSeamGravityWellUnavailable,
)

log = logging.getLogger(__name__)

STATE_REF = "zephyr-sustainability-2026-06-08"

DRIFT_VECTORS = [
    (
        "consent-hoarding-lever",
        "Consent terms function as a hoarding lever: deposit with "
        "maximally-restrictive consent to accrue provenance weight while suppressing downstream use/cascade.",
    ),
    (
        "resale-fee-leak",
        "Fees passively collect on resale if attribution tags become tradeable on "
        "borrowed ATProto rails — there is no primary-vs-transfer event distinction.",
    ),
    (
        "hold-is-neutral",
        "Deposit-vs-hold is economically neutral: there is no holding cost, so hoarding "
        "is not penalized.",
    ),
    (
        "crypto-gravity",
        "EIP-2981 / NFT-royalty framing creates crypto-gravity, pulling the system "
        "toward speculative settlement.",
    ),
]


def run_warmup(
    run_dir: str = "/srv/lapis/backcaster/runs/2026-06-08-0351-zephyr-sustainability-goal/",
    out_dir: str = "/srv/lapis/jagged-seam/runs/",
    *,
    operator_class: str = "gravitywell",
) -> dict[str, Any]:
    """Run the warm-up reproduction: load archived desire pole, hand-transcribed fear pole, overlay.

    Args:
        run_dir: Path to archived 06-08 Backcaster run with decomposition.yaml, gaps.yaml, components.yaml.
        out_dir: Path to write run artifacts. Created if missing.
        operator_class: LLM operator to use (default "gravitywell").

    Returns:
        Dict with keys: run_id, reproduction_signal (dict), and any error flags.

    Raises:
        FileNotFoundError: If required input YAML files are missing (input verification).
        JaggedSeamGravityWellUnavailable: If overlay() cannot reach the operator.
        ValueError: On other validation failures.

    Side effects:
        - Writes run artifacts under <out_dir>/<run_id>/: fear_pole.yaml, desire_pole.yaml,
          constraint_surface.yaml, run.yaml.
        - On non-reproduction: writes gap_testimony.yaml and raises ValueError.
        - On GW unavailable: writes skip.yaml and raises JaggedSeamGravityWellUnavailable.
    """
    run_dir_path = Path(run_dir).expanduser().resolve()
    out_dir_path = Path(out_dir).expanduser().resolve()

    log.info(f"Loading archived inputs from {run_dir_path}")

    # Input verification: assert required files exist and are non-empty.
    required_files = ["decomposition.yaml", "gaps.yaml", "components.yaml"]
    for fname in required_files:
        fpath = run_dir_path / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Required input file missing: {fpath}")
        if fpath.stat().st_size == 0:
            raise FileNotFoundError(f"Required input file is empty: {fpath}")

    # Load archived YAML.
    decomposition = _load_yaml(run_dir_path / "decomposition.yaml")
    gaps_data = _load_yaml(run_dir_path / "gaps.yaml")
    components_data = _load_yaml(run_dir_path / "components.yaml")

    preconditions = decomposition.get("preconditions", [])
    gaps = gaps_data.get("gaps", [])
    components = components_data.get("components", [])

    log.info(f"Loaded {len(preconditions)} preconditions, {len(gaps)} gaps, {len(components)} components")

    # Build DESIRE pole (verified, from archived Backcaster).
    log.info("Building desire pole from archived Backcaster outputs")
    desire_pole = from_backcaster(
        gaps=gaps,
        preconditions=preconditions,
        state_ref=STATE_REF,
        components=components,
        fidelity="verified",
    )
    log.info(f"Desire pole: {len(desire_pole.claims)} claims")

    # Build FEAR pole (transcribed, from drift-vectors).
    log.info("Building fear pole from hand-transcribed drift-vectors")
    fear_vectors = [vec[1] for vec in DRIFT_VECTORS]
    fear_pole = from_prose(
        text_items=fear_vectors,
        kind="fear",
        state_ref=STATE_REF,
        provenance_prefix="mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vector",
        fidelity="transcribed",
    )
    log.info(f"Fear pole: {len(fear_pole.claims)} claims")

    # Run overlay (ONE GW call).
    log.info("Running overlay...")
    try:
        surface = overlay(fear_pole, desire_pole, operator_class=operator_class)
    except JaggedSeamGravityWellUnavailable as e:
        # GW unavailable: write skip.yaml and re-raise.
        log.error(f"GW unavailable: {e}")
        out_dir_path.mkdir(parents=True, exist_ok=True)
        run_id = _gen_run_id()
        skip_file = out_dir_path / run_id / "skip.yaml"
        skip_file.parent.mkdir(parents=True, exist_ok=True)
        skip_payload = {
            "skipped": True,
            "reason": "gw_unavailable",
            "ts": int(time.time()),
        }
        _save_yaml(skip_file, skip_payload)
        log.info(f"Wrote skip marker to {skip_file}")
        raise

    # Compute reproduction signal.
    log.info("Computing reproduction signal...")
    reproduction_signal = _compute_reproduction_signal(surface, fear_pole, desire_pole)

    # Create output directory.
    out_dir_path.mkdir(parents=True, exist_ok=True)
    run_id = _gen_run_id()
    run_path = out_dir_path / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    # Write pole artifacts.
    _save_yaml(run_path / "fear_pole.yaml", _pole_to_dict(fear_pole))
    _save_yaml(run_path / "desire_pole.yaml", _pole_to_dict(desire_pole))
    _save_yaml(run_path / "constraint_surface.yaml", _surface_to_dict(surface))

    log.info(f"Wrote pole artifacts to {run_path}")

    # Write run metadata.
    arbiter_sha = _get_arbiter_git_sha()
    run_metadata = {
        "run_id": run_id,
        "state_ref": STATE_REF,
        "operator_class": operator_class,
        "elapsed_seconds": 0,  # Placeholder; could track timing.
        "arbiter_git_sha": arbiter_sha,
        "source_run_dir": str(run_dir_path),
        "source_mem_key": "thread/zephyr-backcaster-sustainability-run-2026-06-08",
        "human_curation_confound": (
            "U2 is a calibration / known-positive, not a falsification. "
            "The fear pole is four drift-vectors a human selected as 'Scout fodder' from mem prose. "
            "The seam that forms is real (falls out of the overlay), but the primitive-surface coverage "
            "is limited to these human-selected inputs. U2 reproduces the seam; U3 (with live Scout) "
            "tests whether Scout's own signal detection finds comparable extremes. "
            "fidelity='transcribed' on fear claims carries this mechanically."
        ),
        "reproduction_signal": reproduction_signal,
    }
    _save_yaml(run_path / "run.yaml", run_metadata)
    log.info(f"Wrote run metadata to {run_path / 'run.yaml'}")

    # Check for non-reproduction (gap-gate).
    if not reproduction_signal.get("seam_item_present", False) or surface.skipped_no_axis:
        log.warning("Non-reproduction detected; writing gap_testimony.yaml and raising")
        _write_gap_testimony(run_path, surface, fear_pole, desire_pole)
        raise ValueError(
            "Non-reproduction: seam did not form. See gap_testimony.yaml for near-misses and inputs hash."
        )

    log.info(f"Warm-up run completed successfully. Run ID: {run_id}")
    return {
        "run_id": run_id,
        "reproduction_signal": reproduction_signal,
        "output_dir": str(run_path),
    }


def _compute_reproduction_signal(surface: Any, fear_pole: Any, desire_pole: Any) -> dict[str, Any]:
    """Compute the reproduction signal: check for the consent-hoarding seam.

    The seam is present if:
    1. surface has an item whose fear_source.provenance == <prefix>[0] (drift-vector-a).
    2. That item's desire_source claim mentions governance/consent/defederation/separation.

    Returns a dict with keys: seam_item_present, consent_decision_variable_present,
    candidate_items, skipped_no_axis.
    """
    # Exact provenance string for drift-vector-a (index 0).
    drift_a_provenance = "mem:thread/zephyr-backcaster-sustainability-run-2026-06-08#drift-vector[0]"

    seam_item_present = False
    consent_dv_present = False
    candidate_items = []

    if surface.skipped_no_axis:
        return {
            "seam_item_present": False,
            "consent_decision_variable_present": False,
            "candidate_items": [],
            "skipped_no_axis": True,
        }

    # Check each item.
    for item in surface.items:
        # Check if fear_source matches drift-vector-a.
        if item.fear_source.provenance == drift_a_provenance:
            # Check if desire_source mentions governance keywords.
            desire_text = item.desire_source.claim.lower()
            governance_keywords = ["governance", "consent", "defederation", "separation"]
            if any(kw in desire_text for kw in governance_keywords):
                seam_item_present = True
                candidate_items.append(
                    {
                        "decision_variable": item.decision_variable,
                        "fear_source": {
                            "claim": item.fear_source.claim,
                            "provenance": item.fear_source.provenance,
                            "fidelity": item.fear_source.fidelity,
                        },
                        "desire_source": {
                            "claim": item.desire_source.claim,
                            "provenance": item.desire_source.provenance,
                            "fidelity": item.desire_source.fidelity,
                        },
                        "crossing_type": item.crossing_type,
                    }
                )

        # Check for any item mentioning consent in its decision_variable.
        if "consent" in item.decision_variable.lower():
            consent_dv_present = True
            candidate_items.append(
                {
                    "decision_variable": item.decision_variable,
                    "fear_source": {
                        "claim": item.fear_source.claim,
                        "provenance": item.fear_source.provenance,
                        "fidelity": item.fear_source.fidelity,
                    },
                    "desire_source": {
                        "claim": item.desire_source.claim,
                        "provenance": item.desire_source.provenance,
                        "fidelity": item.desire_source.fidelity,
                    },
                    "crossing_type": item.crossing_type,
                }
            )

    return {
        "seam_item_present": seam_item_present,
        "consent_decision_variable_present": consent_dv_present,
        "candidate_items": candidate_items,
        "skipped_no_axis": False,
    }


def _write_gap_testimony(run_path: Path, surface: Any, fear_pole: Any, desire_pole: Any) -> None:
    """Write gap_testimony.yaml on non-reproduction.

    States facts only: the missing axis, near-misses (highest-overlap items), and
    inputs_sha256 (hash of serialized poles). Does NOT speculate on why.
    """
    # Compute inputs hash.
    fear_json = json.dumps(_pole_to_dict(fear_pole), sort_keys=True)
    desire_json = json.dumps(_pole_to_dict(desire_pole), sort_keys=True)
    inputs_combined = fear_json + desire_json
    inputs_sha = hashlib.sha256(inputs_combined.encode()).hexdigest()

    # Near-misses: items that formed in the surface (if any).
    near_misses = []
    for item in surface.items:
        near_misses.append(
            {
                "decision_variable": item.decision_variable,
                "fear_source_provenance": item.fear_source.provenance,
                "desire_source_provenance": item.desire_source.provenance,
                "crossing_type": item.crossing_type,
            }
        )

    testimony = {
        "missing_axis": "consent-restrictiveness seam absent",
        "near_misses": near_misses,
        "inputs_sha256": inputs_sha,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    _save_yaml(run_path / "gap_testimony.yaml", testimony)
    log.info(f"Wrote gap_testimony.yaml to {run_path / 'gap_testimony.yaml'}")


def _pole_to_dict(pole: Any) -> dict[str, Any]:
    """Serialize a PoleOutput to a dict."""
    return {
        "kind": pole.kind,
        "state_ref": pole.state_ref,
        "thin": pole.thin,
        "claims": [
            {
                "claim": c.claim,
                "kind": c.kind,
                "provenance": c.provenance,
                "fidelity": c.fidelity,
            }
            for c in pole.claims
        ],
    }


def _surface_to_dict(surface: Any) -> dict[str, Any]:
    """Serialize a ConstraintSurface to a dict."""
    return {
        "state_ref": surface.state_ref,
        "skipped_no_axis": surface.skipped_no_axis,
        "items": [
            {
                "decision_variable": item.decision_variable,
                "fear_source": {
                    "claim": item.fear_source.claim,
                    "provenance": item.fear_source.provenance,
                    "fidelity": item.fear_source.fidelity,
                },
                "desire_source": {
                    "claim": item.desire_source.claim,
                    "provenance": item.desire_source.provenance,
                    "fidelity": item.desire_source.fidelity,
                },
                "crossing_type": item.crossing_type,
            }
            for item in surface.items
        ],
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Save a dict to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _gen_run_id() -> str:
    """Generate a run ID from ISO timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-warmup")


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
        result = run_warmup()
        print(json.dumps(result, indent=2))
        return 0
    except (FileNotFoundError, ValueError) as e:
        log.error(f"Warm-up failed: {e}")
        return 1
    except JaggedSeamGravityWellUnavailable as e:
        log.error(f"GW unavailable: {e}")
        return 1
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
