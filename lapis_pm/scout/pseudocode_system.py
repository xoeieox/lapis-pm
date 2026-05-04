"""PseudocodeSystemEntity — Entity backed by a ScoutScaffold YAML card.

Sibling to archetypes.engine.CharacterEntity — same Entity-backed-by-YAML-card
pattern, different card schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lapis_engine import Event, LanguageModel, Message, RunContext

from .scaffold import ScoutScaffold, load_scaffold
from .components import ToolboxComponent, WorldStateComponent


@dataclass
class PseudocodeSystemEntity:
    """Entity that voices a pseudocode-system architecture from a scaffold YAML.

    The entity renders its system prompt from the scaffold at each tick (so
    WorldStateComponent can inject the current tick's external state).  Each
    act() call is one Qwen call; ScoutDirector parses the JSON result in
    resolve().
    """

    scaffold: ScoutScaffold
    cell_id: str
    seed: int
    llm: LanguageModel
    _components: list[Any] = field(default_factory=list, init=False, repr=False)
    _observed: list[Event] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._components = [
            WorldStateComponent(scaffold=self.scaffold),
            ToolboxComponent(
                available_tools=self.scaffold.static_scaffold.available_tools
            ),
        ]

    # ------------------------------------------------------------------
    # Entity.id
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return f"{self.scaffold.spec_id}/{self.cell_id}/{self.seed}"

    # ------------------------------------------------------------------
    # Classmethod constructor (Entity-backed-by-YAML-card pattern)
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        scaffold_path: str | Path,
        cell_id: str,
        seed: int,
        llm: LanguageModel,
    ) -> "PseudocodeSystemEntity":
        scaffold = load_scaffold(scaffold_path)
        return cls(scaffold=scaffold, cell_id=cell_id, seed=seed, llm=llm)

    # ------------------------------------------------------------------
    # Entity.system_prompt
    # ------------------------------------------------------------------

    def system_prompt(self, ctx: RunContext) -> str:
        """Render the full system prompt for this tick.

        Contains: objective + sketch + all steps (necessary/optional label
        SUPPRESSED) + component contributions (world state, available tools)
        + scenario conditions + utilization pattern + generation directive
        + JSON output schema description.
        """
        s = self.scaffold.static_scaffold
        parts: list[str] = []

        parts.append(f"# Objective\n{s.objective}")
        parts.append(f"# Architecture\n{s.architecture_sketch}")

        # Steps — necessary/optional label SUPPRESSED per spec
        all_steps = s.necessary_steps + s.optional_steps
        if all_steps:
            step_lines = ["# Steps"]
            for step in all_steps:
                step_lines.append(f"- {step.id}: {step.description}")
            parts.append("\n".join(step_lines))

        # Component contributions (world state, toolbox)
        for comp in self._components:
            contribution = comp.contribute(ctx)
            if contribution:
                parts.append(contribution)

        # Scenario conditions
        if s.scenario.conditions:
            cond_lines = ["# Scenario conditions"]
            for cond in s.scenario.conditions:
                cond_lines.append(f"- {cond}")
            parts.append("\n".join(cond_lines))

        # Utilization pattern
        if s.scenario.utilization_pattern:
            parts.append(
                f"# Utilization pattern\n{s.scenario.utilization_pattern}"
            )

        # Context references (chub bundles + vault sections declared in scaffold)
        # At v0 the actual bundle/vault content is not loaded — the references
        # are rendered so Qwen knows what context is relevant to this simulation.
        ctx = self.scaffold.context
        if ctx.chubs or ctx.vault_sections:
            ctx_lines = ["# Context references"]
            for chub_id in ctx.chubs:
                ctx_lines.append(f"- chub: {chub_id}")
            for vs in ctx.vault_sections:
                sections_str = ", ".join(vs.sections) if vs.sections else "all"
                ctx_lines.append(f"- vault: {vs.path} (sections: {sections_str})")
            parts.append("\n".join(ctx_lines))

        # Generation directive
        if self.scaffold.generation_directive:
            parts.append(
                f"# Generation directive\n{self.scaffold.generation_directive}"
            )

        # JSON output schema instruction
        parts.append(
            "# Output format\n"
            "Respond with strict JSON matching this schema (no markdown fences):\n"
            '{"scenario_generated": "<brief description of the scenario you\'re simulating>",\n'
            ' "execution_trace": "<narrative of the architecture executing step by step>",\n'
            ' "breaks_observed": [{"signature": "...", "severity": "high|medium|low",'
            ' "at_step": "...", "description": "..."}],\n'
            ' "leverage_points": [{"description": "...", "at_step": "..."}],\n'
            ' "surprises": ["..."],\n'
            ' "drift_signals": [{"at_step": "...", "invented": "<what context you had to invent>"}],\n'
            ' "tools_used": {"<tool_id>": <count>},\n'
            ' "tools_wished_for": [{"signature": "...", "would_help_at_step": "..."}],\n'
            ' "performance_assessment": "<one paragraph assessment>"}'
        )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Entity.act
    # ------------------------------------------------------------------

    def act(self, prompt: str, ctx: RunContext) -> str:
        """Single Qwen call. Returns raw response string; ScoutDirector parses JSON."""
        return self.llm.chat(
            self.system_prompt(ctx),
            [Message(role="user", content=prompt)],
        )

    # ------------------------------------------------------------------
    # Entity.observe
    # ------------------------------------------------------------------

    def observe(self, event: Event) -> None:
        # _observed is reserved for future episodic-memory-like behavior;
        # v0 stores but does not read.
        self._observed.append(event)
        for comp in self._components:
            comp.observe(event)
