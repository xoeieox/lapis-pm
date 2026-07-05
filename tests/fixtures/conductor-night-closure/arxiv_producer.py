"""Fixture stub of conductor's arxiv_producer.py (ArxivWatchProducer wrapping
arxiv_watch.run(), real file :37 `import arxiv_watch` deferred inside execute())."""

from __future__ import annotations

import logging
import socket
from pathlib import Path

from agents_core.room_paths import room_path  # noqa: F401

log = logging.getLogger("arxiv-producer")


class ArxivWatchProducer:
    name = "arxiv-watch"
    lane = "gpu"

    def execute(self, slot):
        import arxiv_watch

        return arxiv_watch.run(
            categories=arxiv_watch.DEFAULT_CATEGORIES,
            score_threshold=arxiv_watch.DEFAULT_SCORE_THRESHOLD,
            max_per_category=arxiv_watch.DEFAULT_MAX_PER_CATEGORY,
        )
