"""Fixture stub of conductor's rsi_ingest.py — the unguarded sibling arxiv_watch.py
imports at module level (real file :40)."""

from __future__ import annotations

from agents_core.room_paths import room_path  # noqa: F401


def _probe_qwen(*args, **kwargs):
    return None


def _call_qwen(*args, **kwargs):
    return None


def _fetch_text(*args, **kwargs):
    return ""


def _distill(*args, **kwargs):
    return ""


def paper_filename(*args, **kwargs) -> str:
    return "paper.txt"
