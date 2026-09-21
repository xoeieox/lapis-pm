"""Liveness classification for scaffold covers refs.

liveness(ref) -> "molten" | "spent"

Dispatches by ref type prefix (e.g. "spec:foo-v0", "decision:bar", "open:q1").
Fail-open: any unreachable source classifies the ref as molten.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents_core.room_paths import room_path, room_str

log = logging.getLogger(__name__)

_SPECS_DIR = room_path('planning.specs')
_TARGETS_DIR = room_path('targets')


def liveness(ref: str) -> str:
    """Classify ref as "molten" or "spent". Fail-open on any exception."""
    try:
        if ":" not in ref:
            log.warning("liveness: malformed ref (no type prefix) %r — treating as molten", ref)
            return "molten"
        ref_type, ref_key = ref.split(":", 1)
        ref_type = ref_type.strip().lower()
        ref_key = ref_key.strip()

        if ref_type == "spec":
            return _liveness_spec(ref_key)
        if ref_type == "decision":
            return _liveness_decision(ref_key)
        if ref_type == "open":
            return "molten"
        log.warning("liveness: unknown ref type %r in %r — treating as molten", ref_type, ref)
        return "molten"
    except Exception as exc:
        log.warning("liveness: exception classifying %r: %s — treating as molten", ref, exc)
        return "molten"


def _liveness_spec(slug: str) -> str:
    """Molten if spec file exists and no bound/non-abandoned target for it; spent otherwise."""
    spec_path = _SPECS_DIR / f"{slug}.md"
    if not spec_path.exists():
        log.debug("liveness: spec %r not found at %s — molten (fail-open)", slug, spec_path)
        return "molten"
    if _any_bound_target_for_spec(slug):
        return "spent"
    return "molten"


def _liveness_decision(key: str) -> str:
    """Spent only if a bound/landed target references the decision key; else molten."""
    if _any_target_references_key(key):
        return "spent"
    return "molten"


def _any_bound_target_for_spec(slug: str) -> bool:
    """Return True if any active/non-abandoned target is bound to spec slug."""
    try:
        import yaml  # type: ignore[import]
        for yaml_path in _TARGETS_DIR.glob("*.yaml"):
            try:
                data: dict[str, Any] = yaml.safe_load(yaml_path.read_text()) or {}
                if _target_bound_to_spec(data, slug):
                    return True
            except Exception as exc:
                log.debug("liveness: error reading %s: %s", yaml_path, exc)
    except Exception as exc:
        log.debug("liveness: error scanning targets for spec %r: %s", slug, exc)
    return False


def _target_bound_to_spec(data: dict[str, Any], slug: str) -> bool:
    status = data.get("status", "active") or "active"
    if status in ("abandoned", "cancelled"):
        return False
    target_id = data.get("id", "") or ""
    # Target ID matching the spec slug is the primary signal (lapis-pm bind names them alike)
    if target_id == slug:
        return True
    # Also check spec_from path if present
    spec_from = data.get("spec_from", "") or ""
    if spec_from:
        # spec_from is a path like /srv/lapis/planning/specs/<slug>.md
        if slug in str(spec_from):
            return True
    return False


def _any_target_references_key(key: str) -> bool:
    """Return True if any non-abandoned active target references the decision key."""
    try:
        import yaml  # type: ignore[import]
        for yaml_path in _TARGETS_DIR.glob("*.yaml"):
            try:
                data: dict[str, Any] = yaml.safe_load(yaml_path.read_text()) or {}
                status = data.get("status", "active") or "active"
                if status in ("abandoned", "cancelled"):
                    continue
                if _data_references_key(data, key):
                    return True
            except Exception as exc:
                log.debug("liveness: error reading %s: %s", yaml_path, exc)
    except Exception as exc:
        log.debug("liveness: error scanning targets for decision %r: %s", key, exc)
    return False


def _data_references_key(data: dict[str, Any], key: str) -> bool:
    """Return True only if a STRUCTURED field names the decision key exactly.

    cr-bundle-lapis-pm-2026-09-21 item 4fd83ffdba: the previous version
    substring-matched the bare key against `description`, `tags`, `covers`,
    and `derived_from` (and `spec_from`), so a short key like "auth" or "v0"
    false-positived against prose and permanently parked scaffolds whose
    underlying work was never implemented. The match is now narrowed to the
    structured list fields `covers` and `derived_from` with exact element
    equality (also accepting the full canonical form `decision:<key>`, since
    a ref written in canonical form is an unambiguous reference). Free-text
    fields (`description`, `tags`, `spec_from`) are deliberately NOT scanned.
    """
    for field_name in ("covers", "derived_from"):
        val = data.get(field_name)
        if isinstance(val, str):
            val = [val]
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item in (key, f"decision:{key}"):
                    return True
    return False
