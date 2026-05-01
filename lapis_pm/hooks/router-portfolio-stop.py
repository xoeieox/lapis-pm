#!/usr/bin/env python3
"""Claude Code Stop hook — router portfolio checkpoint guard.

Fires when the session terminates. If /router-checkpoint was not run during
this session, prompts the Router to run it (or writes a minimal auto-checkpoint
when no Router turns are possible).

Part of: router-portfolio-persistence-v0
Spec: /srv/lapis/planning/specs/router-portfolio-persistence-v0.md §Stop hook fallback

Vendored into lapis_pm/hooks/ so the PR is self-contained (DoD item 4).
.claude/settings.json references this file directly.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Session window: how far back to look for a session summary (hours)
_SESSION_WINDOW_HOURS = 12

# Derive repo root from this file's location: lapis_pm/hooks/ → repo root
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def _session_start_iso(window_hours: int = _SESSION_WINDOW_HOURS) -> str:
    """Return an ISO timestamp for `window_hours` ago as the session start boundary."""
    from datetime import timedelta
    return (datetime.now(timezone.utc) - timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_agents_core() -> bool:
    """Load agents_core via editable finder. Returns True on success."""
    try:
        finder_path = "/home/user/.local/lib/python3.12/site-packages/__editable___agents_core_0_3_0_finder.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("_ac_finder", finder_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            mod.install()
        return True
    except Exception:
        return False


def _check_checkpoint_exists(since_iso: str) -> bool:
    """Return True if a router session summary exists since_iso."""
    try:
        from lapis_pm.router_portfolio import session_checkpoint_exists
        return session_checkpoint_exists(since_iso)
    except Exception:
        return False


def _write_minimal_checkpoint(since_iso: str) -> str | None:
    """Write a minimal auto-checkpoint (no LLM labeling, no prose summary).

    Called when no Router turns remain and /router-checkpoint was not run.
    Returns the mem key written, or None on failure.
    """
    try:
        from lapis_pm.router_portfolio import (
            read_session_entries,
            write_session_summary,
        )
        entries = read_session_entries(since_iso=since_iso)
        if not entries:
            return None
        summary = (
            "## Auto-checkpoint (minimal — /router-checkpoint not invoked)\n\n"
            f"Session ended without /router-checkpoint. "
            f"Captured {len(entries)} portfolio event(s). "
            "No primitive labeling or prose summary generated. "
            "Run /router-checkpoint at the start of the next session to review and label."
        )
        key = write_session_summary(
            slug="auto-stop-fallback",
            summary_markdown=summary,
            entries=entries,
            session_start_iso=since_iso,
        )
        return key
    except Exception:
        return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return

    # Only fire for lapis-pm sessions (check cwd or repo)
    cwd = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR", "")
    if "lapis-pm" not in cwd and "lapis_pm" not in cwd:
        return

    # Load agents_core
    if not _load_agents_core():
        return

    # Add repo root to path if needed (derived from __file__, not hardcoded)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    since_iso = _session_start_iso()

    if _check_checkpoint_exists(since_iso):
        # /router-checkpoint already ran — nothing to do
        return

    # Determine if Router turns are still possible.
    # stop_hook_active=True means the hook is blocking the stop; Claude can respond.
    stop_active = bool(payload.get("stop_hook_active"))

    if stop_active:
        # Turns are possible — prompt the Router to run /router-checkpoint
        reminder = (
            "\n[router-portfolio-stop] This PM session has portfolio events that were not checkpointed. "
            "Please run /router-checkpoint now to consolidate them into a session summary before the session ends. "
            "The next session will inherit this summary as its dynamic prior."
        )
        print(reminder, flush=True)
    else:
        # No turns remain — write minimal auto-checkpoint
        key = _write_minimal_checkpoint(since_iso)
        if key:
            # Log to stderr (not shown to user but captured in hook logs)
            print(f"[router-portfolio-stop] minimal auto-checkpoint written → {key}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
