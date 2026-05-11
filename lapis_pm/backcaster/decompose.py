"""Backcaster Stage 1 — Decomposer.

Takes a goal-state text, decomposes it into preconditions across 6 axes:
  psychological, material, infrastructural, governance, social-norm, economic

Returns a list of Precondition objects.

Stub mode (BACKCASTER_STUB=1): returns canned preconditions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from .schema import AXES, Precondition

# Module-level import so tests can patch lapis_pm.backcaster.decompose.call_operator
try:
    from agents_core.llm import call_operator
except ImportError:
    call_operator = None  # type: ignore

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are Backcaster, an inverse-planning analyst. Your task is to decompose a \
goal-state into the preconditions that must be true for that goal-state to be \
reachable and livable.

For each axis provided, produce a list of 2-5 concrete preconditions. Each \
precondition is a short sentence (one claim, active voice). Also surface any \
implicit dependencies — things a precondition assumes that aren't stated.

Return ONLY a JSON object. Schema:
{
  "<axis>": [
    {
      "statement": "<one-sentence precondition>",
      "implicit_dependencies": ["<dep1>", "<dep2>"]
    },
    ...
  ],
  ...
}

Axes you must cover: psychological, material, infrastructural, governance, \
social-norm, economic.

Do not add prose outside the JSON object.
"""

# Canned stub output for BACKCASTER_STUB=1
_STUB_PRECONDITIONS: dict[str, list[dict[str, Any]]] = {
    "psychological": [
        {
            "statement": "Individuals feel safe enough to share creative work in small-trust contexts without fear of exploitation.",
            "implicit_dependencies": ["trust is earned, not assumed"],
        },
        {
            "statement": "Participants can maintain autonomy while being interdependent with a friend-group.",
            "implicit_dependencies": ["autonomy and interdependence are not in permanent tension"],
        },
    ],
    "material": [
        {
            "statement": "Participants have access to devices and reliable connectivity sufficient for asynchronous collaboration.",
            "implicit_dependencies": ["baseline digital access"],
        },
    ],
    "infrastructural": [
        {
            "statement": "A shared digital space exists that supports small-group creative collaboration without requiring enterprise account management.",
            "implicit_dependencies": ["no IT department or admin overhead"],
        },
        {
            "statement": "Identity and data portability are possible so members can leave without losing their contributions.",
            "implicit_dependencies": ["open data formats", "no lock-in"],
        },
    ],
    "governance": [
        {
            "statement": "Group norms around contribution, credit, and conflict resolution are established by the group itself, not imposed.",
            "implicit_dependencies": ["self-governance capacity exists"],
        },
    ],
    "social-norm": [
        {
            "statement": "It is socially acceptable to limit a collaboration to a bounded friend group rather than scaling to public.",
            "implicit_dependencies": ["smallness is not failure"],
        },
    ],
    "economic": [
        {
            "statement": "Participating in the collaboration does not require paying enterprise-tier subscription fees.",
            "implicit_dependencies": ["pricing is person-scale, not org-scale"],
        },
    ],
}


def _parse_decomposition(raw: str, axes: list[str]) -> list[Precondition]:
    """Parse LLM JSON output into Precondition objects."""
    text = raw.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(text)
    preconditions: list[Precondition] = []
    counter: dict[str, int] = {}

    for axis in axes:
        items = data.get(axis, [])
        if not isinstance(items, list):
            items = []
        for item in items:
            if isinstance(item, str):
                item = {"statement": item, "implicit_dependencies": []}
            counter[axis] = counter.get(axis, 0) + 1
            pid = f"{axis}-{counter[axis]:02d}"
            preconditions.append(Precondition(
                id=pid,
                axis=axis,
                statement=item.get("statement", ""),
                implicit_dependencies=item.get("implicit_dependencies", []),
            ))

    return preconditions


def decompose(
    goal_text: str,
    axes: list[str] | None = None,
    model: str = "qwen",
    stub: bool = False,
) -> tuple[list[Precondition], str]:
    """Decompose goal_text into preconditions.

    Returns (preconditions, prompt_hash).
    """
    if axes is None:
        axes = list(AXES)

    if stub or os.environ.get("BACKCASTER_STUB") == "1":
        preconditions = _preconditions_from_stub(axes)
        return preconditions, "stub:decompose"

    prompt = (
        f"Goal-state to decompose:\n\n{goal_text}\n\n"
        f"Axes to cover: {', '.join(axes)}\n\n"
        "Return the JSON decomposition."
    )
    prompt_hash = "sha256:" + hashlib.sha256(prompt.encode()).hexdigest()

    _llm = call_operator
    if _llm is None:
        log.warning("decompose: call_operator unavailable; returning empty preconditions")
        return [], prompt_hash
    try:
        raw = _llm(model, prompt=prompt, system=_SYSTEM_PROMPT, json_mode=True, timeout=120)
    except NotImplementedError as exc:
        raise RuntimeError(
            f"decompose: model={model!r} requires ClaudeQueue sync surface not yet "
            f"implemented. Use --model qwen for v0. Detail: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.warning("decompose: LLM call failed (%s); returning empty preconditions", exc)
        return [], prompt_hash

    if raw is None:
        log.warning("decompose: LLM returned None; returning empty preconditions")
        return [], prompt_hash

    try:
        preconditions = _parse_decomposition(raw, axes)
    except Exception as exc:  # noqa: BLE001
        log.warning("decompose: parse failed (%s); returning empty preconditions", exc)
        return [], prompt_hash

    return preconditions, prompt_hash


def _preconditions_from_stub(axes: list[str]) -> list[Precondition]:
    preconditions: list[Precondition] = []
    for axis in axes:
        items = _STUB_PRECONDITIONS.get(axis, [])
        for i, item in enumerate(items, 1):
            preconditions.append(Precondition(
                id=f"{axis}-{i:02d}",
                axis=axis,
                statement=item["statement"],
                implicit_dependencies=item.get("implicit_dependencies", []),
            ))
    return preconditions
