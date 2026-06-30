"""Per-item sidecar persistence for the Prior-Art Scout."""
from __future__ import annotations

from pathlib import Path

import yaml

_SIDECAR_ROOT = Path("/srv/lapis/prior-art-scout/sidecars")


def _sorted_yaml(obj) -> str:
    return yaml.dump(obj, default_flow_style=False, sort_keys=True, allow_unicode=True)


def sidecar_path(key: str) -> Path:
    safe = key.replace("/", "__") + ".yaml"
    return _SIDECAR_ROOT / safe


def load_sidecar(key: str) -> dict | None:
    path = sidecar_path(key)
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return None


def write_sidecar(key: str, data: dict) -> None:
    _SIDECAR_ROOT.mkdir(parents=True, exist_ok=True)
    sidecar_path(key).write_text(_sorted_yaml(data))


def is_known_hopeless(key: str) -> bool:
    s = load_sidecar(key)
    return bool(s and s.get("verdict") == "insufficient-sources")
