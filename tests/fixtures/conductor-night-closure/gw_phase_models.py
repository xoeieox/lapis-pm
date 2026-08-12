"""Fixture stub of conductor's gw_phase_models.py — the top-level import
night_plan.py has carried since 2026-08-01 (real file night_plan.py:53, c35e8f2),
added to _CONDUCTOR_NIGHT_SCRIPTS by
lapis-pm-conductor-brix-gw-runtime-deploy-manifest-v0.

Stdlib-only leaf: imports nothing scripts/-local, matching the real module's own
contract (a true leaf importable by both night_plan.py and podcast_ingest_night.py
without either depending on the other). Business logic is intentionally NOT
reproduced — this fixture exists solely to exercise the scripts/-local import
graph the R4 closure test walks.
"""

from __future__ import annotations

GW_SERVED_MODEL_BY_PHASE: dict[str, str] = {}
