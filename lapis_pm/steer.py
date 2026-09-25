"""Typed steer channel for mid-run PM course-correction.

Three steer types (lapis-pm-typed-steer-channel-v0):
  context   — informational; folded into episodic, does not consume the action slot
  directive — mid-run constraint; registered as a mem overlay consumed by the
              next fixer/fixer_retry dispatch via inject_overlay()
  emergency — halt; pauses the target, file stays pending until pause commits

Storage: /srv/lapis/steers/<tid>__<ts>-<type>.json (pending)
         /srv/lapis/steers/applied/<tid>__<ts>-<type>.json (consumed)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


VALID_TYPES = ("context", "directive", "emergency")

_OVERLAY_PREFIX = "pm/steer-overlay"

_steers_dir_warned = False


def _steers_dir() -> Path:
    """Prefer the typed room_paths seam; fall back to literal /srv/lapis/steers with a
    one-time warning until the companion agents-core PR registers the 'steers' key."""
    try:
        from agents_core.room_paths import room_path
        return room_path("steers")
    except KeyError:
        global _steers_dir_warned
        if not _steers_dir_warned:
            sys.stderr.write(
                "[steer:room_path-fallback] 'steers' key not yet registered in "
                "agents_core room_paths; using literal /srv/lapis/steers. Land the companion "
                "agents-core PR to remove this fallback.\n"
            )
            _steers_dir_warned = True
        root = os.environ.get("ROOM_ROOT", "/room")
        return Path(root) / "steers"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _applied_dir() -> Path:
    return _steers_dir() / "applied"


def file_steer(
    target_id: str,
    type_: str,
    message: str,
    filed_by: str = "pm",
) -> Path:
    """Write a pending steer file and return its path. Raises ValueError on bad type."""
    if type_ not in VALID_TYPES:
        raise ValueError(f"Invalid steer type {type_!r}; must be one of {VALID_TYPES}")
    ts = _now_iso()
    safe_ts = ts.replace(":", "-").replace("+", "p")
    filename = f"{target_id}__{safe_ts}-{type_}.json"
    steers = _steers_dir()
    steers.mkdir(parents=True, exist_ok=True)
    path = steers / filename
    payload = {
        "type": type_,
        "message": message,
        "ts": ts,
        "filed_by": filed_by,
        "target_id": target_id,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _parse_steer_file(path: Path) -> dict | None:
    """Parse a steer file; return None on error."""
    try:
        data = json.loads(path.read_text())
        data["_path"] = path
        return data
    except Exception as e:
        sys.stderr.write(f"[steer:parse-error] {path}: {e}\n")
        return None


def _pending_steer_paths(target_id: str) -> list[Path]:
    """All pending steer files for target_id, sorted by filename (chrono)."""
    steers = _steers_dir()
    if not steers.exists():
        return []
    return sorted(p for p in steers.glob(f"{target_id}__*.json") if p.is_file())


def mark_applied(path: Path, disposition: str = "applied") -> None:
    """Atomically move a pending steer file to applied/, stamping applied_ts.

    Idempotent: a missing source (already moved) is a no-op."""
    if not path.exists():
        return
    applied = _applied_dir()
    applied.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text())
    except Exception:
        data = {}
    data["applied_ts"] = _now_iso()
    data["disposition"] = disposition
    dest = applied / path.name
    dest.write_text(json.dumps(data, indent=2))
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def consume_pending_steers(
    target_id: str,
    types: tuple[str, ...] = ("context", "directive"),
) -> list[dict]:
    """Glob pending steer files for target_id whose type is in `types`.

    Atomically moves each to applied/ (stamping applied_ts) and returns the
    parsed steer dicts. Emergency is excluded from the default so it is not
    marked applied until the pause commits (see peek_emergencies / mark_applied).
    Best-effort: unreadable files are logged and skipped.
    """
    results = []
    for path in _pending_steer_paths(target_id):
        s = _parse_steer_file(path)
        if s is None:
            continue
        if s.get("type") not in types:
            continue
        mark_applied(path, disposition="consumed")
        results.append(s)
    return results


def peek_emergencies(target_id: str) -> list[dict]:
    """Return pending emergency steer dicts WITHOUT moving their files.

    Each dict carries an internal `_path` so the caller can mark_applied()
    after the pause commits."""
    results = []
    for path in _pending_steer_paths(target_id):
        s = _parse_steer_file(path)
        if s is None:
            continue
        if s.get("type") == "emergency":
            results.append(s)
    return results


def _mem():
    from . import node_identity
    return node_identity.writable_store()


def _overlay_key(target_id: str) -> str:
    return f"{_OVERLAY_PREFIX}/{target_id}"


def set_directive_overlay(target_id: str, message: str) -> None:
    """Store a directive overlay in mem. Last directive wins.

    If an UNCONSUMED overlay already exists, writes a pm:steer:directive:superseded
    observation so the audit log surfaces every directive that never reached a fixer."""
    from . import episodic
    existing = get_directive_overlay(target_id)
    if existing is not None:
        episodic.write_observation(
            target_id,
            f"[steer:directive:superseded] Prior unconsumed directive replaced: {existing[:200]}",
            extra_tags=["pm:steer:directive:superseded"],
        )
    _mem().set(
        _overlay_key(target_id),
        message,
        tags=["lapis-pm", "steer-overlay"],
    )


def get_directive_overlay(target_id: str) -> str | None:
    """Return the pending directive overlay text, or None. Non-destructive read."""
    rec = _mem().get(_overlay_key(target_id))
    return rec["content"] if rec else None


def consume_directive_overlay(target_id: str) -> str | None:
    """Return AND delete the directive overlay. Returns None if absent."""
    rec = _mem().get(_overlay_key(target_id))
    if rec is None:
        return None
    message = rec["content"]
    _mem().delete(_overlay_key(target_id))
    return message


def inject_overlay(target_id: str, vars_: dict, agent_type: str) -> None:
    """The ONE place that touches vars_['steer_directive_block'].

    ALWAYS sets the key so str.format(**vars_) never KeyErrors.
    - For fixer / fixer_retry / fixer_staged / fixer_flash: consume the
      overlay (if any) and build the directive block; write
      pm:steer:directive:consumed observation. (fixer_flash is the
      flashnext-fixer-trial-v0 trial tier - a mid-run PM directive filed
      against a trial target must reach the trial dispatch, not silently
      persist past it.)
    - For all other agent types: set "" without consuming — the directive
      persists until a fixer dispatch picks it up.
    """
    from . import episodic
    if agent_type in ("fixer", "fixer_retry", "fixer_staged", "fixer_flash"):
        overlay = consume_directive_overlay(target_id)
    else:
        overlay = None

    if overlay:
        vars_["steer_directive_block"] = (
            "## Active PM directive (mid-run steer)\n\n"
            f"{overlay}\n\n"
            "This directive was filed after initial dispatch. Treat it as a binding "
            "additional constraint on the spec. If it conflicts with the spec, flag "
            "the conflict explicitly in the PR description rather than silently "
            "resolving it one way."
        )
        episodic.write_observation(
            target_id,
            f"[steer:directive:consumed] Injected into dispatch vars_: {overlay[:200]}",
            extra_tags=["pm:steer:directive:consumed"],
        )
    else:
        vars_["steer_directive_block"] = ""
