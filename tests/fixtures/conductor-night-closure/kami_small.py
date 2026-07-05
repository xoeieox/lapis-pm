"""Fixture stub of conductor's kami_small.py — the shared leaf sibling kami_batch.py
and kami_selector.py both import unguarded at module level (real files :33, :23)."""

from __future__ import annotations

from agents_core.room_paths import room_path  # noqa: F401
from agents_core.llm import call_operator  # noqa: F401

RAG_SEARCH_URL = "http://localhost:0000/search"
MAX_RETRIES = 3
RETRIEVAL_KEY = "kami-retrieval"


class KamiRAGUnavailable(RuntimeError):
    pass


class GravityWellUnavailable(RuntimeError):
    pass
