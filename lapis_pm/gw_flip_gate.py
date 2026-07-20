"""Shared GravityWell flip + doorman serving-gate primitives.

Extracted from lapis_pm/backcaster/quest_leg.py (research-quest-nightly-producer-v0,
AC1) so that any Dowser-consuming producer can reuse the proven swarm<->big flip
choreography, serving-gate polling, phase helpers, and any-exit flip guard without
reimplementing GW's safety-critical mode handling. This is a behavior-preserving
extraction — quest_leg.py's own tests (72/72) must remain unaffected.

Flip sequencing (local-only path):
  swarm flip -> doorman serving-gate -> read_batch
  -> big flip -> doorman serving-gate -> critique_batch

Anti-deadlock: every serving-gate has a hard timeout (FLIP_SERVE_TIMEOUT,
default 240s). On timeout: abort phase, fail toward big (never strand GW mid-flip
in swarm).
"""
from __future__ import annotations

import contextlib
import logging
import os
import signal as _sig
import time
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FLIP_CONTROLLER_URL = os.environ.get(
    "FLIP_CONTROLLER_URL", "http://203.0.113.10:8408"
)
FLIP_CONTROLLER_TOKEN = os.environ.get("FLIP_CONTROLLER_TOKEN", "")
DOORMAN_URL = os.environ.get("DOORMAN_URL", "http://127.0.0.1:8407")
FLIP_SERVE_TIMEOUT = int(os.environ.get("FLIP_SERVE_TIMEOUT", "240"))


# ---------------------------------------------------------------------------
# GW flip + doorman serving-gate
# ---------------------------------------------------------------------------

def flip_gw(mode: str, *, source: str = "gw-flip-gate") -> bool:
    """POST to flip-controller; return True on success.

    Raises NodeIdentityViolation if this node is not authorized to flip the
    physical GravityWell controller (Design §3b, I2).
    """
    from . import node_identity
    node_identity.ensure_gw_flip_authorized(FLIP_CONTROLLER_URL)
    try:
        import httpx
        url = f"{FLIP_CONTROLLER_URL}/v0/nodes/gravitywell/flip"
        headers: dict[str, str] = {}
        if FLIP_CONTROLLER_TOKEN:
            headers["Authorization"] = f"Bearer {FLIP_CONTROLLER_TOKEN}"
        resp = httpx.post(
            url,
            json={"mode": mode, "source": source},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        log.info("[gw_flip_gate] GW flip -> %s OK (source=%s)", mode, source)
        return True
    except Exception as exc:
        log.warning("[gw_flip_gate] GW flip -> %s failed (source=%s): %s", mode, source, exc)
        return False


def gate_doorman_serving(timeout_s: int = FLIP_SERVE_TIMEOUT) -> bool:
    """Poll doorman /status until nodes.gravitywell.serving == True.

    Returns False only on hard timeout (never raises).
    """
    try:
        import httpx as _httpx
    except ImportError:
        log.warning("[gw_flip_gate] httpx not available; cannot gate on doorman")
        return False

    deadline = time.monotonic() + timeout_s
    poll_interval = 5
    while time.monotonic() < deadline:
        try:
            resp = _httpx.get(f"{DOORMAN_URL}/status", timeout=5)
            if resp.status_code == 200:
                gw = resp.json().get("nodes", {}).get("gravitywell", {})
                if gw.get("serving") or gw.get("serving_mode") == "deferred":
                    return True
        except Exception as exc:
            log.debug("[gw_flip_gate] doorman poll error: %s", exc)
        time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
    log.warning("[gw_flip_gate] doorman serving-gate timed out after %ds", timeout_s)
    return False


# ---------------------------------------------------------------------------
# Core flip+gate phase helpers (return False on timeout/error)
# ---------------------------------------------------------------------------

def phase_swarm_read(
    clean_requests: list[dict],
    read_operator: str,
    dowser: Any,
    flip_fn: Any,
    gate_fn: Any,
) -> tuple[list[dict] | None, bool]:
    """Flip to swarm, gate, read. Returns (drafts, timed_out).

    On timeout: flips back to big before returning.
    """
    if not flip_fn("swarm"):
        log.warning("[gw_flip_gate] swarm flip failed")
        return None, False
    if not gate_fn(FLIP_SERVE_TIMEOUT):
        log.warning("[gw_flip_gate] serving-gate timed out after swarm flip; failing toward big")
        flip_fn("big")
        return None, True
    try:
        result = dowser.read_batch(clean_requests, read_operator=read_operator)
        return result.get("drafts", []), False
    except Exception as exc:
        log.warning("[gw_flip_gate] read_batch failed: %s", exc)
        flip_fn("big")
        return None, False


def phase_big_critique(
    drafts: list[dict],
    critic_operator: str,
    dowser: Any,
    flip_fn: Any,
    gate_fn: Any,
    is_local: bool,
) -> tuple[list[dict] | None, bool]:
    """Flip to big (if local path), gate, critique. Returns (verdicts, timed_out)."""
    if is_local:
        if not flip_fn("big"):
            log.warning("[gw_flip_gate] big flip failed before critique")
            return None, False
        if not gate_fn(FLIP_SERVE_TIMEOUT):
            log.warning("[gw_flip_gate] serving-gate timed out after big flip")
            return None, True
    try:
        result = dowser.critique_batch(drafts, critic_operator=critic_operator)
        return result.get("verdicts", []), False
    except Exception as exc:
        log.warning("[gw_flip_gate] critique_batch failed: %s", exc)
        return None, False


# ---------------------------------------------------------------------------
# Any-exit flip guard (criterion 6c): SIGTERM handler + try/finally ensure
# GW is never stranded on swarm on any exit path (normal, exception, SIGTERM).
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def any_exit_flip_guard(flip_fn: Any, *, active: bool):
    """Install a SIGTERM handler and guarantee flip_fn("big") runs on any exit.

    When active=False this is a no-op passthrough (matches the escalation path,
    which owns no swarm-mode state and must not install a signal handler or
    force a flip on exit).
    """
    if not active:
        yield
        return

    prev_sigterm = _sig.getsignal(_sig.SIGTERM)

    def _sigterm_handler(signum, frame):
        raise SystemExit(128 + signum)

    try:
        _sig.signal(_sig.SIGTERM, _sigterm_handler)
    except (OSError, ValueError):
        pass  # not in main thread; cannot install signal handler

    try:
        yield
    finally:
        flip_fn("big")  # best-effort; idempotent; covers exception + SIGTERM exit paths
        try:
            _sig.signal(_sig.SIGTERM, prev_sigterm)
        except (OSError, ValueError):
            pass
