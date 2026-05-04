"""ScoutDirector — single-entity scripted-GM director.

Director IS the GM at v0: scripted perturbation (time/external-state progression)
lives here, NOT in the entity.  One tick = one Qwen call.

Engine contract: Engine.run(director, entities=[entity]) passes the non-director
entity list to next_acting().  The director's own Entity methods (system_prompt,
act, observe) exist to satisfy the Director(Entity) Protocol; they are never
invoked by the engine loop for v0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from lapis_engine import Entity, Event, RunContext

from .scaffold import ScoutScaffold


def _strip_fence(text: str) -> str:
    """Strip markdown code-fence wrapper if present."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    # First line is the fence (possibly with language tag); last closing fence
    start = 1
    end = len(lines)
    for i in range(len(lines) - 1, 0, -1):
        if lines[i].strip().startswith("```"):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


@dataclass
class ScoutDirector:
    """Single-entity scripted-GM director for pseudocode-simulate runs.

    Iterates over time_progression ticks defined in the scaffold.  At each tick:
    1. make_prompt() renders the current tick's world-state context.
    2. resolve() parses the entity's JSON response, emits structured events.
    3. Advances the tick cursor; sets terminal when all ticks exhausted.
    """

    id: str
    scaffold: ScoutScaffold
    cell_id: str
    _tick_cursor: int = field(default=0, init=False)
    _terminal: bool = field(default=False, init=False)
    _tick_prompts: list[str] = field(default_factory=list, init=False)
    _director_events: list[Event] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------
    # Entity methods (Director IS an Entity; these are stubs for v0)
    # ------------------------------------------------------------------

    def system_prompt(self, ctx: RunContext) -> str:
        return ""

    def act(self, prompt: str, ctx: RunContext) -> str:
        return ""

    def observe(self, event: Event) -> None:
        self._director_events.append(event)

    # ------------------------------------------------------------------
    # Director methods
    # ------------------------------------------------------------------

    def next_acting(self, entities: list[Entity]) -> Entity:
        """Single-entity v0: always return entities[0]."""
        return entities[0]

    def make_prompt(self, entity: Entity, ctx: RunContext) -> str:
        """Render the GM prompt for the current tick.

        Perturbation logic lives here (Director-is-GM seam).
        """
        ticks = list(self.scaffold.static_scaffold.scenario.time_progression.items())
        if self._tick_cursor < len(ticks):
            tick_key, tick_desc = ticks[self._tick_cursor]
        else:
            tick_key, tick_desc = "t-final", "End of scenario timeframe."

        lines = [f"At tick={tick_key}: {tick_desc}"]

        ext_state = self.scaffold.static_scaffold.scenario.external_state
        if ext_state:
            lines.append("Current external state:")
            for sys_name, attrs in ext_state.items():
                if isinstance(attrs, dict):
                    attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
                else:
                    attr_str = str(attrs)
                lines.append(f"  {sys_name}: {attr_str}")

        lines.append(
            "\nContinue executing the architecture against this scenario. "
            "Output strict JSON."
        )

        prompt = "\n".join(lines)
        self._tick_prompts.append(prompt)
        return prompt

    def resolve(self, entity: Entity, response: str, ctx: RunContext) -> list[Event]:
        """Parse JSON response, emit structured events, advance tick cursor."""
        events: list[Event] = []

        text = _strip_fence(response)
        try:
            parsed = json.loads(text)

            # Break events
            for b in parsed.get("breaks_observed", []):
                events.append(Event(
                    entity_id=entity.id,
                    type="break_observed",
                    content=b.get("signature", ""),
                    metadata=dict(b),
                ))

            # Leverage events
            for lp in parsed.get("leverage_points", []):
                events.append(Event(
                    entity_id=entity.id,
                    type="leverage_point",
                    content=lp.get("description", ""),
                    metadata=dict(lp),
                ))

            # Drift events
            for ds in parsed.get("drift_signals", []):
                events.append(Event(
                    entity_id=entity.id,
                    type="drift_signal",
                    content=ds.get("invented", ""),
                    metadata=dict(ds),
                ))

            # Tool-used events (one event per use, so toolbox counts correctly)
            for tool_id, count in parsed.get("tools_used", {}).items():
                for _ in range(max(0, int(count))):
                    events.append(Event(
                        entity_id=entity.id,
                        type="tool_used",
                        content=tool_id,
                        metadata={"tool_id": tool_id},
                    ))

            events.append(Event(
                entity_id=entity.id,
                type="tick_result",
                content="ok",
                metadata={"tick": self._tick_cursor, "parsed": True},
            ))

        except (json.JSONDecodeError, ValueError, KeyError):
            events.append(Event(
                entity_id=entity.id,
                type="json_parse_failure",
                content=response[:500],
                metadata={"tick": self._tick_cursor},
            ))

        # Advance tick cursor
        ticks = list(self.scaffold.static_scaffold.scenario.time_progression.items())
        self._tick_cursor += 1
        events.append(Event(
            entity_id=entity.id,
            type="tick_advance",
            content="",
            metadata={"new_tick": self._tick_cursor},
        ))
        if self._tick_cursor >= len(ticks):
            self._terminal = True

        return events

    def is_terminal(self, ctx: RunContext) -> bool:
        return self._terminal

    # ------------------------------------------------------------------
    # Accessors for runner (post-run)
    # ------------------------------------------------------------------

    def tick_prompts(self) -> list[str]:
        return list(self._tick_prompts)
