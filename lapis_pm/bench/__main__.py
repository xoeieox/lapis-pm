"""CLI entry point for the Synapse fresh-instance benchmark harness.

Usage::

    # Run a battery (top-level flags):
    python -m lapis_pm.bench --battery loud_quiet --out runs/baseline-<ts>.json

    # Run a battery (explicit subcommand):
    python -m lapis_pm.bench run --battery loud_quiet --out runs/baseline-<ts>.json [--timeout 120]

    # Compare two captures:
    python -m lapis_pm.bench compare baseline.json labeled.json

    # Or use the module directly:
    python -m lapis_pm.bench.compare baseline.json labeled.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lapis_pm.bench.battery import load_battery, run_battery
from lapis_pm.bench.compare import compare as do_compare


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_usage()
        return 2

    # Detect subcommand vs legacy top-level --battery/--out flags
    if argv[0] in ("run", "compare"):
        subcommand, rest = argv[0], argv[1:]
    else:
        subcommand, rest = "run", argv

    if subcommand == "run":
        return _cmd_run(rest)
    if subcommand == "compare":
        return _cmd_compare(rest)

    _print_usage()
    return 2


def _cmd_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lapis_pm.bench",
        description="Run a battery of stripped-instance prompts and capture output.",
        add_help=True,
    )
    parser.add_argument("--battery", required=True, metavar="NAME",
                        help="Battery name (file in bench/batteries/, e.g. loud_quiet)")
    parser.add_argument("--out", required=True, metavar="PATH",
                        help="Output JSON capture path")
    parser.add_argument("--timeout", type=int, default=180, metavar="SECS",
                        help="Per-side timeout in seconds (default: 180)")
    args = parser.parse_args(argv)

    print(f"Loading battery: {args.battery}")
    battery = load_battery(args.battery)
    n = len(battery["pairs"])
    print(f"Running {n} pair(s) × 2 sides (timeout={args.timeout}s each) …")

    capture = run_battery(battery, timeout=args.timeout)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(capture, indent=2))
    print(f"Wrote {out_path}  ({len(capture['results'])} pairs captured)")
    return 0


def _cmd_compare(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lapis_pm.bench compare",
        description="Compare two bench run captures and print per-pair deltas.",
    )
    parser.add_argument("baseline", metavar="BASELINE_JSON")
    parser.add_argument("labeled", metavar="LABELED_JSON")
    args = parser.parse_args(argv)

    deltas = do_compare(Path(args.baseline), Path(args.labeled))
    print(json.dumps(deltas, indent=2))
    if all(not d["loud_delta"] and not d["quiet_delta"] for d in deltas.values()):
        print("(no deltas — captures are identical)")
    return 0


def _print_usage() -> None:
    print(
        "usage: python -m lapis_pm.bench --battery NAME --out PATH [--timeout SECS]\n"
        "       python -m lapis_pm.bench compare BASELINE LABELED",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
