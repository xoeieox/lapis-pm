"""Battery YAML loader and runner.

A battery is a YAML file in ``lapis_pm/bench/batteries/`` that defines a set
of (loud, quiet) prompt pairs.  The loader validates structure; the runner
iterates pairs, calls :func:`~lapis_pm.bench.fresh_instance.spawn_stripped`
for each side, and returns a stable capture dict ready for JSON serialisation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from lapis_pm.bench.fresh_instance import spawn_stripped

BATTERIES_DIR = Path(__file__).parent / "batteries"
CAPTURE_SCHEMA_VERSION = 1


class BatteryError(ValueError):
    """Raised when a battery YAML fails validation."""


def load_battery(name: str) -> dict:
    """Load and validate a battery by name from the batteries/ directory.

    Args:
        name: Battery filename without extension, e.g. ``"loud_quiet"``.

    Raises:
        FileNotFoundError: If no matching YAML exists.
        BatteryError: If the YAML is structurally invalid.
    """
    path = BATTERIES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Battery not found: {path}")
    return parse_battery(path.read_text(), source=path)


def parse_battery(yaml_text: str, source: Path | None = None) -> dict:
    """Parse and validate battery YAML text.

    Args:
        yaml_text: Raw YAML string.
        source: Optional path for error messages.

    Raises:
        BatteryError: If the battery is structurally invalid.
    """
    data = yaml.safe_load(yaml_text)
    _validate(data, source)
    return data


def run_battery(battery: dict, timeout: int = 180, synapse: bool = False) -> dict:
    """Run all pairs in *battery*, capturing stripped-instance responses.

    Each pair side (loud + quiet) is run through
    :func:`~lapis_pm.bench.fresh_instance.spawn_stripped`.

    Args:
        battery: Validated battery dict (from :func:`load_battery` or
            :func:`parse_battery`).
        timeout: Per-side timeout in seconds forwarded to ``spawn_stripped``.
        synapse: When True, the temp HOME's settings.json is populated with
            ONLY the Synapse ``UserPromptSubmit`` hook entry (chub-inject
            suppressed, MCP servers empty).  Requires the Synapse service to
            be running.

    Returns:
        Capture dict::

            {
                "schema_version": int,
                "battery_name":   str,
                "timestamp":      str,   # ISO 8601 UTC
                "synapse":        bool,
                "results": [
                    {
                        "pair_id": str,
                        "loud":    <spawn_stripped return>,
                        "quiet":   <spawn_stripped return>,
                    },
                    ...
                ],
            }

        The ``stripped_state`` field inside each side entry lets a reviewer
        reproduce the exact env the child process saw.  When synapse=True,
        each side also has a ``synapse_state`` key.
    """
    ts = datetime.now(timezone.utc).isoformat()
    results = []
    for pair in battery["pairs"]:
        loud = spawn_stripped(pair["loud"]["topic"], timeout=timeout, synapse=synapse)
        quiet = spawn_stripped(pair["quiet"]["topic"], timeout=timeout, synapse=synapse)
        results.append({
            "pair_id": pair["id"],
            "loud": loud,
            "quiet": quiet,
        })
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "battery_name": battery["name"],
        "timestamp": ts,
        "synapse": synapse,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate(data: Any, source: Path | None) -> None:
    loc = str(source) if source else "<string>"
    if not isinstance(data, dict):
        raise BatteryError(f"{loc}: battery must be a YAML mapping, got {type(data).__name__}")
    _require(data, "name", loc)
    _require(data, "pairs", loc)
    if not isinstance(data["pairs"], list) or not data["pairs"]:
        raise BatteryError(f"{loc}: 'pairs' must be a non-empty list")
    for i, pair in enumerate(data["pairs"]):
        _validate_pair(pair, i, loc)


def _require(data: dict, key: str, loc: str) -> None:
    if key not in data:
        raise BatteryError(f"{loc}: missing required field '{key}'")


def _validate_pair(pair: Any, idx: int, loc: str) -> None:
    if not isinstance(pair, dict):
        raise BatteryError(f"{loc}: pair[{idx}] must be a mapping, got {type(pair).__name__}")
    if "id" not in pair:
        raise BatteryError(f"{loc}: pair[{idx}] missing required field 'id'")
    for side in ("loud", "quiet"):
        if side not in pair:
            raise BatteryError(f"{loc}: pair[{idx}] (id={pair.get('id', '?')!r}) missing '{side}'")
        if not isinstance(pair[side], dict):
            raise BatteryError(
                f"{loc}: pair[{idx}][{side!r}] must be a mapping, got {type(pair[side]).__name__}"
            )
        if "topic" not in pair[side]:
            raise BatteryError(
                f"{loc}: pair[{idx}][{side!r}] missing required field 'topic'"
            )
