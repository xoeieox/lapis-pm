"""Fixture stub of conductor's scout_producer.py.

Its cross-package deps (lapis_pm.scout.*, lapis_engine.adapters) are function-deferred
(real file :47-50, 212-213, 273) — NOT scripts/-local siblings, so they are deliberately
NOT part of the R4 closure test's import graph. They are the documented "honest limit"
covered instead by the separate skipif scout-precondition test.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path

from agents_core.room_paths import room_path  # noqa: F401

log = logging.getLogger("scout-producer")


class ScoutProducer:
    name = "scout"
    lane = "gpu"

    def enumerate_work(self, budget):
        from lapis_pm.scout.selection import select_worklist, SelectedEntry  # noqa: F401
        from lapis_pm.scout.liveness import liveness as liveness_fn  # noqa: F401
        from lapis_pm.scout.scaffold import ScoutScaffold, load_scaffold  # noqa: F401
        from lapis_pm.scout.night_queue import Quarantine  # noqa: F401
        return []

    def _model(self):
        from lapis_engine.adapters import LLAMA_SERVER_DEFAULT_MODEL  # noqa: F401
        return LLAMA_SERVER_DEFAULT_MODEL
