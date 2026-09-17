"""Fixture stub of conductor's night_plan_manager.py (Rung C.0 manager).

Reproduces the real file's four scripts/-local import edges (three verified
against conductor origin/main = 6679fd2, PR #797; the fourth — gw_seat_lane —
added by night-roles-seat-declaration-v0 O1, rev 2) so the R4 closure test
walks them. The real module also imports agents_core.llm.call_operator and
agents_core.slots.SlotStore — both external (non-edge), so they are omitted
here.
"""

from __future__ import annotations

import night_task_menu  # noqa: F401
import idea_collider_night_batch  # noqa: F401
import night_coordinator  # noqa: F401
import gw_seat_lane  # noqa: F401
