"""Fixture stub of conductor's kami_producer.py.

Reproduces the real file's deferred imports of night_coordinator, kami_selector,
kami_batch, kami_sweep, kami_adjudicator (real file :52-53, 100, 168, 180, 189).
kami is inactive in night_producers.yaml but its full closure is manifested so
re-activating it is a pure conductor/yaml change (see spec)."""

from __future__ import annotations

import logging
import socket
from pathlib import Path

log = logging.getLogger("kami-producer")


class KamiProducer:
    name = "kami"
    lane = "gpu"

    def enumerate_work(self, budget):
        from night_coordinator import WorkItem  # noqa: F401
        from kami_selector import select_kami_docs  # noqa: F401
        return []

    def execute(self, slot):
        from night_coordinator import ExecResult  # noqa: F401
        from kami_batch import execute_kami_batch  # noqa: F401
        from kami_sweep import execute_kami_sweep  # noqa: F401
        from kami_adjudicator import execute_kami_adjudicate_manifest  # noqa: F401
        return None
