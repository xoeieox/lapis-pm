"""Fixture stub of conductor's gw_seat_lane.py (night-roles-seat-declaration-v0 D2).

The real module is stdlib-only (urllib/subprocess/seat-check CLI) — no
scripts/-local siblings — so this stub needs no import edges; its presence is
what lets the R4 closure test walk the night_plan_manager -> gw_seat_lane
edge (the O1 BLOCKER fix: an unmanifested helper never deploys to
/data/agents/scripts and the role-seat lane gate silently does not exist).
"""

from __future__ import annotations

# Intentionally no scripts/-local imports — a true leaf (like gw_phase_models).
