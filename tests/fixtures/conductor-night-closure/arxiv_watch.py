"""Fixture stub of conductor's arxiv_watch.py.

Reproduces the real file's UNGUARDED module-level `from rsi_ingest import (...)`
(real file :40) — the exact deferred-at-producer-level-but-unguarded-at-module-level
edge R4's load-test must exercise: a bare `import arxiv_producer` does not touch this
(it's inside execute()), so the test imports arxiv_watch directly.
"""

from __future__ import annotations

from agents_core.room_paths import room_path  # noqa: F401

from rsi_ingest import (  # noqa: E402, F401
    _probe_qwen,
    _call_qwen,
    _fetch_text,
    _distill,
    paper_filename,
)

DEFAULT_CATEGORIES = ["cs.AI"]
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_MAX_PER_CATEGORY = 5


def run(categories, score_threshold, max_per_category):
    return []
