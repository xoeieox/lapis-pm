"""Backcaster Stage 3 — Component Deriver.

For each gap, derives components that could close the gap.
Each component carries a mandatory category tag (one of 7 kinds).
Financial components carry a mandatory subtype (extractive/distributive/neutral).

Invariant 3: retry-then-fallback on malformed LLM output.
  - Retry once on malformed output.
  - If retry also fails: write component with category=research, unsourced=True,
    fallback=True. Never hard-reject.

Stub mode (BACKCASTER_STUB=1): returns canned components.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from .schema import CATEGORIES, FINANCIAL_SUBTYPES, Component, Gap

# Module-level import so tests can patch lapis_pm.backcaster.derive.call_operator
try:
    from agents_core.llm import call_operator
except ImportError:  # agents_core unavailable in isolated test env
    call_operator = None  # type: ignore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deriver system prompt — must include software-bias resistance language
# (Invariant 8; reviewer checks for this language)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Backcaster, a roadmap component deriver for Lapis.

Lapis is a human-wisdom company that uses AI as mirror — not an AI company. \
Resist the reflex to make every component a software item. If the answer is a \
community practice, name it as such. If the answer is a policy change, name it \
as such. If the answer is a funding source or income structure, name it \
`financial` and surface whether the value-flow is extractive (concentrates \
value out of communities), distributive (circulates value through communities \
without concentration), or neutral (one-time grant, savings runway, fixed-fee \
services).

You will be given a gap analysis for one precondition. Derive 1-3 components \
that could close the gap. Each component must carry EXACTLY ONE category tag \
from this fixed list:

  software         — a tool, system, code change
  policy           — a rule, agreement, contract
  community-formation — a group, network, practice
  research         — knowledge to be gathered or generated
  infrastructure   — physical capacity (compute, space, energy)
  cultural-shift   — a normative change in practice or expectation
  financial        — a value-flow component (funding source, income structure,
                     capital requirement, runway extension, fiscal sponsorship,
                     MRR target, debt instrument)

For `financial` components, you MUST also include a `subtype` field:
  extractive   — captures value out of communities to concentrate it
                 (VC funding, debt-financed growth, advertising-monetized scale)
  distributive — circulates value through communities without concentration
                 (mutual aid pool, gift economy, member-supported cooperative)
  neutral      — value-flow without clear extractive/distributive shape
                 (one-time grant, personal savings runway, fixed-fee services)

Components also carry:
  effort_estimate: small | medium | large
  reversibility:   high | medium | low
  dependencies:    [] or list of other component ids

Return ONLY a JSON object:
{
  "components": [
    {
      "description": "<what this component is>",
      "category": "<one of the 7 values above>",
      "subtype": "<only for financial; omit for other categories>",
      "effort_estimate": "small|medium|large",
      "reversibility": "high|medium|low",
      "dependencies": []
    },
    ...
  ]
}

Do not add prose outside the JSON object.
Do not extend the category list — exactly 7 categories, no others.
"""

# Canned stub components per gap precondition_id
_STUB_COMPONENTS: dict[str, list[dict[str, Any]]] = {
    "psychological-01": [
        {
            "description": "Cultural narrative artifacts (essays, zines, talks) that legitimize small-group creative work as valid and complete.",
            "category": "cultural-shift",
            "effort_estimate": "medium",
            "reversibility": "low",
            "dependencies": [],
        },
        {
            "description": "Community practice of intentional-scope declarations — groups publicly name their intended size.",
            "category": "community-formation",
            "effort_estimate": "small",
            "reversibility": "high",
            "dependencies": [],
        },
    ],
    "psychological-02": [
        {
            "description": "Lightweight autonomy-contract templates for friend-groups to make their interdependence terms explicit.",
            "category": "policy",
            "effort_estimate": "small",
            "reversibility": "high",
            "dependencies": [],
        },
    ],
    "material-01": [
        {
            "description": "Research into offline-capable collaboration modes for low-connectivity contexts.",
            "category": "research",
            "effort_estimate": "medium",
            "reversibility": "high",
            "dependencies": [],
        },
    ],
    "infrastructural-01": [
        {
            "description": "Zephyrium friend-group collaboration space — small-group permission model without enterprise admin.",
            "category": "software",
            "effort_estimate": "large",
            "reversibility": "medium",
            "dependencies": [],
        },
    ],
    "infrastructural-02": [
        {
            "description": "Open portable format specification for friend-group creative outputs.",
            "category": "policy",
            "effort_estimate": "medium",
            "reversibility": "high",
            "dependencies": [],
        },
    ],
    "governance-01": [
        {
            "description": "Norm-setting toolkit: guided questions that help friend-groups surface and codify implicit expectations.",
            "category": "community-formation",
            "effort_estimate": "small",
            "reversibility": "high",
            "dependencies": [],
        },
    ],
    "social-norm-01": [
        {
            "description": "Public discourse campaign reframing 'staying small' as a legitimate creative choice.",
            "category": "cultural-shift",
            "effort_estimate": "large",
            "reversibility": "low",
            "dependencies": [],
        },
    ],
    "economic-01": [
        {
            "description": "Person-scale pricing tier for Zephyrium — no org-management features bundled.",
            "category": "financial",
            "subtype": "neutral",
            "effort_estimate": "medium",
            "reversibility": "medium",
            "dependencies": ["infrastructural-01"],
        },
        {
            "description": "Member-supported cooperative funding model for Zephyrium infrastructure costs.",
            "category": "financial",
            "subtype": "distributive",
            "effort_estimate": "large",
            "reversibility": "low",
            "dependencies": [],
        },
    ],
}


def _parse_components(
    raw: str,
    precondition_id: str,
    base_id: str,
) -> list[Component]:
    """Parse LLM JSON into Component objects.

    Raises ValueError or ValidationError on malformed output.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(text)
    items = data.get("components", [])
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("No 'components' list in LLM output")

    components = []
    for i, item in enumerate(items, 1):
        cid = f"{base_id}-{i:02d}"
        # Validate category
        cat = item.get("category", "")
        if cat not in CATEGORIES:
            raise ValueError(f"Invalid category {cat!r}")
        subtype = item.get("subtype")
        if cat == "financial" and subtype not in FINANCIAL_SUBTYPES:
            raise ValueError(f"Financial component missing valid subtype: {subtype!r}")

        comp = Component(
            id=cid,
            gap_precondition_id=precondition_id,
            description=item.get("description", ""),
            category=cat,
            subtype=subtype if cat == "financial" else None,
            effort_estimate=item.get("effort_estimate", "medium"),
            reversibility=item.get("reversibility", "medium"),
            dependencies=item.get("dependencies", []),
        )
        components.append(comp)

    return components


def derive_components(
    gaps: list[Gap],
    model: str = "qwen",
    stub: bool = False,
) -> tuple[list[Component], str, int]:
    """Derive components for all gaps.

    Returns (components, prompt_hash, fallback_count).
    """
    if stub or os.environ.get("BACKCASTER_STUB") == "1":
        components = _components_from_stub(gaps)
        return components, "stub:derive", 0

    all_components: list[Component] = []
    prompt_hashes: list[str] = []
    fallback_count = 0

    for gap in gaps:
        pid = gap.precondition_id
        base_id = f"comp-{pid}"

        prompt = (
            f"Gap for precondition [{pid}]:\n"
            f"  What exists: {gap.what_exists}\n"
            f"  What missing: {gap.what_missing}\n"
            f"  What miswired: {gap.what_miswired}\n\n"
            "Derive components that could close this gap."
        )
        prompt_hash = "sha256:" + hashlib.sha256(prompt.encode()).hexdigest()
        prompt_hashes.append(prompt_hash)

        raw = None
        _llm = call_operator
        if _llm is None:
            log.warning("derive: call_operator unavailable for gap %s", pid)
        else:
            try:
                raw = _llm(model, prompt=prompt, system=_SYSTEM_PROMPT, json_mode=True, timeout=120)
            except NotImplementedError as exc:
                raise RuntimeError(
                    f"derive: model={model!r} requires ClaudeQueue sync surface not yet "
                    f"implemented. Use --model qwen for v0. Detail: {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                log.warning("derive: LLM call failed for gap %s (%s)", pid, exc)

        parsed = False
        if raw is not None:
            # First attempt
            try:
                comps = _parse_components(raw, pid, base_id)
                all_components.extend(comps)
                parsed = True
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                log.warning("derive: parse failed for %s (attempt 1: %s); retrying", pid, exc)

        if not parsed and raw is not None:
            # Retry once — re-call LLM with explicit remediation instruction
            retry_prompt = (
                f"{prompt}\n\n"
                "IMPORTANT: Your previous response was malformed. "
                "Return ONLY valid JSON matching the schema. "
                "category MUST be one of: software, policy, community-formation, "
                "research, infrastructure, cultural-shift, financial. "
                "financial MUST include subtype (extractive, distributive, neutral)."
            )
            retry_hash = "sha256:" + hashlib.sha256(retry_prompt.encode()).hexdigest()
            prompt_hashes.append(retry_hash)

            retry_raw = None
            if _llm is not None:
                try:
                    retry_raw = _llm(
                        model, prompt=retry_prompt, system=_SYSTEM_PROMPT, json_mode=True, timeout=120
                    )
                except NotImplementedError as exc:
                    raise RuntimeError(
                        f"derive: model={model!r} requires ClaudeQueue sync surface not yet "
                        f"implemented. Use --model qwen for v0. Detail: {exc}"
                    ) from exc
                except Exception as exc:  # noqa: BLE001
                    log.warning("derive: retry LLM call failed for %s (%s)", pid, exc)

            if retry_raw is not None:
                try:
                    comps = _parse_components(retry_raw, pid, base_id)
                    all_components.extend(comps)
                    parsed = True
                except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                    log.warning("derive: parse failed for %s (attempt 2: %s); using fallback", pid, exc)

        if not parsed:
            # Fallback — emit research component with fallback=True (Invariant 3)
            cid = f"{base_id}-fallback"
            fallback_comp = Component(
                id=cid,
                gap_precondition_id=pid,
                description=f"Fallback: could not derive component for gap {pid}",
                category="research",
                subtype=None,
                effort_estimate="medium",
                reversibility="medium",
                dependencies=[],
                unsourced=True,
                fallback=True,
            )
            all_components.append(fallback_comp)
            fallback_count += 1
            log.info("derive: fallback component written for gap %s", pid)

    combined_hash = "sha256:" + hashlib.sha256(
        b"\x00".join(h.encode() for h in prompt_hashes)
    ).hexdigest() if prompt_hashes else "none"

    return all_components, combined_hash, fallback_count


def _components_from_stub(gaps: list[Gap]) -> list[Component]:
    components: list[Component] = []
    for gap in gaps:
        pid = gap.precondition_id
        items = _STUB_COMPONENTS.get(pid, [])
        if not items:
            # Default stub fallback
            items = [{
                "description": f"Research: understand what closes gap for {pid}.",
                "category": "research",
                "effort_estimate": "medium",
                "reversibility": "high",
                "dependencies": [],
            }]
        for i, item in enumerate(items, 1):
            cid = f"comp-{pid}-{i:02d}"
            cat = item["category"]
            components.append(Component(
                id=cid,
                gap_precondition_id=pid,
                description=item["description"],
                category=cat,
                subtype=item.get("subtype") if cat == "financial" else None,
                effort_estimate=item.get("effort_estimate", "medium"),
                reversibility=item.get("reversibility", "medium"),
                dependencies=item.get("dependencies", []),
            ))
    return components
