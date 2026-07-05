"""Fixture stub of conductor's idea_collider.py.

Reproduces the real file's UNGUARDED module-level `from podcast_engine import
call_seam_model, parse_jsonl` (real file :42) — the deferred-siblings edge R4's
load-test exercises by importing idea_collider directly (idea_collider_night_batch_
producer's own import of it is itself deferred, so a bare top-level import of the
producer module would not touch this)."""

from __future__ import annotations

import logging

from agents_core.room_paths import room_path, room_str  # noqa: F401

from podcast_engine import call_seam_model, parse_jsonl  # noqa: E402, F401

from agents_core.gw_agent import call_gw_agent  # noqa: E402, F401
from agents_core.roadmap import (  # noqa: E402, F401
    ROADMAP_MIRROR_PREAMBLE,
    ground_roadmap,
    materialize_committed_plan,
    walk_lineage,
)
from agents_core.mem import MemoryStore  # noqa: E402, F401

log = logging.getLogger("idea-collider")
