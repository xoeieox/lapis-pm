"""Scout components: WorldStateComponent and ToolboxComponent.

Both satisfy the lapis_engine.Component protocol:
  contribute(ctx: RunContext) -> str | None
  observe(event: Event) -> None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lapis_engine import Event, RunContext

if TYPE_CHECKING:
    from .scaffold import ScoutScaffold, ToolSpec


@dataclass
class WorldStateComponent:
    """Tracks the current external-state snapshot at the active tick.

    Contributes a "## Current world state" section to the entity's context.
    Advances its internal tick cursor when the director emits tick_advance events.
    """

    scaffold: Any  # ScoutScaffold — Any to avoid circular import at runtime
    _tick_cursor: int = field(default=0, init=False)

    def contribute(self, ctx: RunContext) -> str | None:
        ext = self.scaffold.static_scaffold.scenario.external_state
        if not ext:
            return None
        lines = ["## Current world state"]
        for system_name, attrs in ext.items():
            if isinstance(attrs, dict):
                attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
            else:
                attr_str = str(attrs)
            lines.append(f"- {system_name}: {attr_str}")
        return "\n".join(lines)

    def observe(self, event: Event) -> None:
        if event.type == "tick_advance":
            self._tick_cursor = event.metadata.get("new_tick", self._tick_cursor)


@dataclass
class ToolboxComponent:
    """Declares available_tools and accumulates usage telemetry.

    Contributes a "## Available tools" section to the entity's context.
    Increments usage_counts when tool_used events are observed.
    """

    available_tools: list[Any]  # list[ToolSpec]
    usage_counts: dict[str, int] = field(default_factory=dict)

    def contribute(self, ctx: RunContext) -> str | None:
        if not self.available_tools:
            return None
        lines = ["## Available tools"]
        for tool in self.available_tools:
            lines.append(f"- {tool.id}: {tool.description}")
        return "\n".join(lines)

    def observe(self, event: Event) -> None:
        if event.type == "tool_used":
            tool_id = event.metadata.get("tool_id")
            if tool_id:
                self.usage_counts[tool_id] = self.usage_counts.get(tool_id, 0) + 1
