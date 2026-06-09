"""Scout runner — parameter-matrix wrapper and trace persistence.

Each (cell, seed) pair:
  1. Build PseudocodeSystemEntity + ScoutDirector.
  2. Engine().run(director, [entity], on_step=trace_collector).
  3. Wrap final trace as LapisToolReturn.
  4. Write to /srv/lapis/scout/traces/<spec-id>/<cell-id>/<run-id>.json.

Storage layout:
  /srv/lapis/scout/traces/<spec-id>/<cell-id>/<run-id>.json
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

import httpx as _httpx

from lapis_engine import Engine, LlamaAdapter, Message

from archetypes_core.corroboration import Citation
from archetypes_core.provenance import InputRef, to_lapis_return

from .director import ScoutDirector
from .pseudocode_system import PseudocodeSystemEntity
from .scaffold import ScoutScaffold, load_scaffold
from .schema import ScoutTracePayload

SCOUT_TRACES_ROOT = Path("/srv/lapis/scout/traces")


# ---------------------------------------------------------------------------
# JSON-mode adapter (Scout-local, not exported)
# ---------------------------------------------------------------------------


class _ScoutJsonAdapter(LlamaAdapter):
    """LlamaAdapter subclass that forces JSON-object output mode.

    Injects ``response_format: {"type": "json_object"}`` into every request
    so Qwen returns structured JSON rather than prose narration.
    """

    def chat(self, system: str, messages: list[Message]) -> str:  # type: ignore[override]
        payload = {
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        client = self.client or _httpx
        resp = client.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _prompt_hash(prompts: list[str]) -> str:
    """sha256 of null-byte-delimited concatenation of tick prompts (in order)."""
    combined = b"\x00".join(p.encode("utf-8") for p in prompts)
    return "sha256:" + hashlib.sha256(combined).hexdigest()


def _tick_prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Field-coercion helpers (LLM output may drift from declared schema types)
# ---------------------------------------------------------------------------


def _as_str(
    v: Any,
    *,
    field: str = "",
    spec_id: str = "",
    cell_id: str = "",
    seed: int = 0,
) -> str:
    """Coerce v to str. list -> newline-joined; None -> ""; other -> str(v)."""
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    if field:
        log.warning(
            "[scout] coerced field=%s expected=str got=%s spec=%s cell=%s seed=%s",
            field, type(v).__name__, spec_id, cell_id, seed,
        )
    if isinstance(v, list):
        return "\n".join(_as_str(x) for x in v)
    return str(v)


def _as_list(
    v: Any,
    *,
    field: str = "",
    spec_id: str = "",
    cell_id: str = "",
    seed: int = 0,
) -> list:
    """Coerce v to list. list passes through; None -> []; other -> [v]."""
    if isinstance(v, list):
        return v
    if v is None:
        return []
    if field:
        log.warning(
            "[scout] coerced field=%s expected=list got=%s spec=%s cell=%s seed=%s",
            field, type(v).__name__, spec_id, cell_id, seed,
        )
    return [v]


# ---------------------------------------------------------------------------
# Trace payload builder
# ---------------------------------------------------------------------------


def _build_trace_payload(
    scaffold: ScoutScaffold,
    cell_id: str,
    cell_params: dict[str, Any],
    seed: int,
    run_log: Any,  # lapis_engine.RunLog
    director: ScoutDirector,
) -> ScoutTracePayload:
    """Assemble ScoutTracePayload from a completed Engine RunLog."""
    breaks_observed: list[dict] = []
    leverage_points: list[dict] = []
    surprises: list[str] = []
    drift_signals: list[dict] = []
    tools_used: dict[str, int] = {}
    tools_wished_for: list[dict] = []
    execution_traces: list[str] = []
    scenario_generated = ""
    performance_assessment = ""
    parse_failure_count = 0

    for step in run_log.steps:
        # Count parse failures from events
        has_failure = any(e.type == "json_parse_failure" for e in step.events)
        if has_failure:
            parse_failure_count += 1

        # Re-parse the raw response to extract structured fields.
        # (Director already parsed and emitted events; we re-parse for payload fields
        # not captured in events: scenario_generated, execution_trace, etc.)
        from .director import _strip_fence
        text = _strip_fence(step.response)
        try:
            parsed = json.loads(text)
            _lctx = dict(spec_id=scaffold.spec_id, cell_id=cell_id, seed=seed)
            if not scenario_generated:
                scenario_generated = _as_str(parsed.get("scenario_generated", ""), field="scenario_generated", **_lctx)
            trace = _as_str(parsed.get("execution_trace", ""), field="execution_trace", **_lctx)
            if trace:
                execution_traces.append(trace)
            breaks_observed.extend(_as_list(parsed.get("breaks_observed", []), field="breaks_observed", **_lctx))
            leverage_points.extend(_as_list(parsed.get("leverage_points", []), field="leverage_points", **_lctx))
            surprises.extend(_as_list(parsed.get("surprises", []), field="surprises", **_lctx))
            drift_signals.extend(_as_list(parsed.get("drift_signals", []), field="drift_signals", **_lctx))
            tools_used_raw = parsed.get("tools_used", {})
            if isinstance(tools_used_raw, dict):
                for tool_id, count in tools_used_raw.items():
                    tools_used[tool_id] = tools_used.get(tool_id, 0) + int(count)
            tools_wished_for.extend(_as_list(parsed.get("tools_wished_for", []), field="tools_wished_for", **_lctx))
            if not performance_assessment:
                performance_assessment = _as_str(parsed.get("performance_assessment", ""), field="performance_assessment", **_lctx)
        except (json.JSONDecodeError, ValueError):
            pass  # parse_failure already counted above

    tick_prompts = director.tick_prompts()
    tick_prompt_hashes = [_tick_prompt_hash(p) for p in tick_prompts]

    return ScoutTracePayload(
        scaffold_hash=scaffold.scaffold_hash(cell_params),
        spec_version=scaffold.spec_version,
        cell_id=cell_id,
        seed=seed,
        scenario_generated=scenario_generated,
        execution_trace="\n---\n".join(execution_traces),
        breaks_observed=breaks_observed,
        leverage_points=leverage_points,
        surprises=surprises,
        drift_signals=drift_signals,
        tools_used=tools_used,
        tools_wished_for=tools_wished_for,
        performance_assessment=performance_assessment,
        tick_prompts=tick_prompts,
        tick_prompt_hashes=tick_prompt_hashes,
        parse_failure_count=parse_failure_count,
        total_tick_count=len(run_log.steps),
    )


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------


def run_single(
    scaffold: ScoutScaffold,
    cell_params: dict[str, Any],
    seed: int,
    llm: Any | None = None,
) -> tuple[Any, str]:  # (LapisToolReturn, run_id)
    """Run a single (cell, seed) and return (LapisToolReturn, run_id)."""
    if llm is None:
        llm = _ScoutJsonAdapter(max_tokens=2048)

    cell_id = ScoutScaffold.cell_id(cell_params)
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{seed:04d}"
    )

    entity = PseudocodeSystemEntity(
        scaffold=scaffold,
        cell_id=cell_id,
        seed=seed,
        llm=llm,
    )
    director = ScoutDirector(
        id=f"scout-director/{scaffold.spec_id}/{cell_id}/{seed}",
        scaffold=scaffold,
        cell_id=cell_id,
    )

    engine = Engine()
    run_log = engine.run(director=director, entities=[entity])

    payload = _build_trace_payload(scaffold, cell_id, cell_params, seed, run_log, director)

    tick_prompts = director.tick_prompts()
    ph = _prompt_hash(tick_prompts) if tick_prompts else None

    # Build summary (parse-free; Router reads this)
    n_breaks = len(payload.breaks_observed)
    break_sigs = ", ".join(
        b.get("signature", "?") for b in payload.breaks_observed[:3]
    ) or "none"
    summary = (
        f"Run {run_id} spec={scaffold.spec_id} cell={cell_id}: "
        f"{n_breaks} break(s) ({break_sigs}), "
        f"{len(payload.leverage_points)} leverage point(s), "
        f"{len(payload.drift_signals)} drift signal(s), "
        f"tools_used={list(payload.tools_used.keys()) or 'none'}"
    )

    input_refs = [
        InputRef(ref=f"/srv/lapis/scout/sims/{scaffold.spec_id}.yaml", content_hash=None, type="file"),
        InputRef(ref=f"cell:{cell_id}", content_hash=None, type="claim"),
    ]

    # Citations: one per chub bundle + one per vault section declared in scaffold.context.
    # At v0 the actual bundle/vault content is not loaded (content_hash=None, excerpt="").
    # The citations record deterministic context-injection events per the spec.
    citations: list[Citation] = []
    ctx = scaffold.context
    for chub_id in ctx.chubs:
        citations.append(Citation(
            source_id=chub_id,
            excerpt="",
            content_hash=None,
            provenance_method="chub-injection",
        ))
    for vs in ctx.vault_sections:
        for section in vs.sections or [""]:
            src = f"{vs.path}#{section}" if section else vs.path
            citations.append(Citation(
                source_id=src,
                excerpt="",
                content_hash=None,
                provenance_method="vault-section-reference",
            ))

    ltr = to_lapis_return(
        payload=payload,
        agent_id="lapis-scout/sim",
        tool="pseudocode_simulate",
        summary=summary,
        model=getattr(llm, "model", None),
        prompt_hash=ph,
        input_refs=input_refs,
        citations=citations,
        upstream_calls=[],
        scope_id=None,
    )

    return ltr, run_id


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------


def simulate(
    scaffold_path: str,
    *,
    runs_per_cell: int | None = None,
    cells: list[str] | None = None,
    llm: Any | None = None,
    traces_root: Path | None = None,
) -> list[Path]:
    """Run the full parameter matrix (or a subset) and persist traces.

    Parameters
    ----------
    scaffold_path:
        Path to the scaffold YAML file.
    runs_per_cell:
        Override the scaffold's ``matrix.runs_per_cell``.
    cells:
        If given, only run the listed cell IDs (e.g. from ``--cell`` CLI flag).
    llm:
        LanguageModel to use.  Defaults to LlamaAdapter pointing at llama-server.
    traces_root:
        Override the default ``/srv/lapis/scout/traces`` root (useful for tests).

    Returns
    -------
    list[Path]
        Paths of written trace files.
    """
    scaffold = load_scaffold(scaffold_path)
    if traces_root is None:
        traces_root = SCOUT_TRACES_ROOT

    all_cell_params = scaffold.cell_params()
    if cells is not None:
        all_cell_params = [
            cp for cp in all_cell_params
            if ScoutScaffold.cell_id(cp) in cells
        ]

    n_runs = runs_per_cell if runs_per_cell is not None else scaffold.matrix.runs_per_cell

    written: list[Path] = []
    for cell_params in all_cell_params:
        cid = ScoutScaffold.cell_id(cell_params)
        # Sanitize cell_id for use as a directory name (= and , are fine on Linux)
        cell_dir = traces_root / scaffold.spec_id / cid
        cell_dir.mkdir(parents=True, exist_ok=True)

        for seed in range(n_runs):
            ltr, run_id = run_single(scaffold, cell_params, seed, llm=llm)
            out_path = cell_dir / f"{run_id}.json"
            out_path.write_text(ltr.to_json())
            written.append(out_path)

    return written
