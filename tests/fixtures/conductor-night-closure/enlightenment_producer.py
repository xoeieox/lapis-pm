"""Fixture stub of conductor's enlightenment_producer.py.

Reproduces the real file's deferred imports of night_coordinator + enlightenment_reader
(real file :44, :103-104). enlightenment is inactive in night_producers.yaml but its
full closure is manifested so re-activating it is a pure conductor/yaml change."""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

from agents_core.room_paths import room_path  # noqa: F401

log = logging.getLogger("enlightenment-producer")


class EnlightenmentProducer:
    name = "enlightenment"
    lane = "gpu"

    def enumerate_work(self, budget):
        from night_coordinator import WorkItem  # noqa: F401
        return []

    def execute(self, slot):
        from enlightenment_reader import (  # noqa: F401
            execute_enlightenment_read,
            execute_enlightenment_synthesis,
        )
        from night_coordinator import ExecResult  # noqa: F401
        return None
