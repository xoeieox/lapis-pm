"""Fixture stub of conductor's kami_sweep.py.

Reproduces the real file's deferred `from research_headings import active_headings,
propose_heading` (real file :560)."""

from __future__ import annotations

from agents_core.room_paths import room_path  # noqa: F401
from agents_core.llm import call_llm, parse_json_object  # noqa: F401


def execute_kami_sweep(*args, **kwargs):
    from research_headings import active_headings, propose_heading  # noqa: F401
    return None
