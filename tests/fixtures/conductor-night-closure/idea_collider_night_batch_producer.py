"""Fixture stub of conductor's idea_collider_night_batch_producer.py.

Reproduces the real file's deferred `import idea_collider as ic` /
`import idea_collider_night_batch as nb` (real file :109, :156-157)."""

from __future__ import annotations

import logging
import socket
from pathlib import Path

log = logging.getLogger("idea-collider-night-batch-producer")


class IdeaColliderNightBatchProducer:
    name = "idea-collider-night-batch"
    lane = "gpu"

    def enumerate_work(self, budget):
        import idea_collider as ic  # noqa: F401
        return []

    def execute(self, slot):
        import idea_collider as ic  # noqa: F401
        import idea_collider_night_batch as nb  # noqa: F401
        return None
