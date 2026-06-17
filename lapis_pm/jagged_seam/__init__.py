"""Jagged-seam overlay-arbiter: minimal apparatus to run forced-extremal-articulation experiment.

The jagged-seam primitive claims that forced-extremal-articulation — a fear-pole and a
desire-pole pressing on the SAME catalogued state from opposite sides, overlaid by a
DRY arbiter — surfaces decision-variables a single balanced pass misses.

This package provides the arbiter and adapters to normalize Scout (fear) and Backcaster
(desire) outputs into a common contract and extract named decision-variables at their
intersection. The warmup module provides a thin run-driver for U2 calibration.
"""
from __future__ import annotations

from .adapters import from_backcaster, from_prose, from_scout_trace
from .arbiter import (
    JaggedSeamGravityWellUnavailable,
    overlay,
)
from .schema import (
    ConstraintSurface,
    ConstraintSurfaceItem,
    PoleClaim,
    PoleOutput,
)

__all__ = [
    "PoleClaim",
    "PoleOutput",
    "ConstraintSurfaceItem",
    "ConstraintSurface",
    "from_scout_trace",
    "from_backcaster",
    "from_prose",
    "overlay",
    "JaggedSeamGravityWellUnavailable",
]
