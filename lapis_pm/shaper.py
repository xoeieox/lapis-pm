"""Shaped-agent registry + dispatch.

Loads the sibling `registry.yaml` at import time, composes chub bundles +
system templates, writes a JSON spec to /srv/lapis/gpu-queue/shaped/, and submits
a `subprocess` task whose command runs `lapis_pm._runner` against that spec.

The GPU queue runner captures stdout (the Claude response) to
/srv/lapis/gpu-queue/completed/<task_id>-output.md.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from agents_core.claude_queue import ClaudeQueue
from agents_core.gpu import GPUQueue, Priority

# chub_broker still lives in /data/agents/scripts/ (deferred from the agents-core
# day-one scope). Keep this last sys.path shim until chub_broker moves into
# agents-core, at which point this block + the import can be deleted.
if "/data/agents/scripts" not in sys.path:
    sys.path.insert(0, "/data/agents/scripts")

try:
    from chub_broker import select_bundles_by_ids, compose_with_system  # noqa: E402
except Exception:
    select_bundles_by_ids = None  # type: ignore
    compose_with_system = None  # type: ignore


REGISTRY_PATH = Path(__file__).parent / "registry.yaml"
SPEC_DIR = Path("/srv/lapis/gpu-queue/shaped")
RUNNER = str(Path(__file__).parent / "_runner.py")

# Per-repo working-clone convention. A repo name like "lapis-engine" maps to
# /srv/git/lapis-engine-working/. Shaped agents dispatch with this as cwd so
# chub-inject.py (SessionStart hook) finds the repo CLAUDE.md and injects its
# @chub: bundles + per-project auto-memory into the subprocess context.
REPO_CWD_TEMPLATE = "/srv/git/{repo}-working"
# Fallback when the working clone doesn't exist (test runs, unknown repo). The
# subprocess still works — just without repo-specific hook injection.
DEFAULT_CWD = "/data/agents"


@dataclass
class ShapedAgent:
    name: str
    chub_bundles: list[str]
    system_template: str
    model: str
    timeout_s: int


def _load_registry() -> tuple[str, dict[str, ShapedAgent]]:
    """Return (shared_preamble, agents). Both are required fields; a missing
    shared_preamble is treated as empty string for backward compatibility with
    older registry.yaml files that predate the preamble."""
    if not REGISTRY_PATH.exists():
        raise RuntimeError(f"Lapis PM registry missing: {REGISTRY_PATH}")
    raw = yaml.safe_load(REGISTRY_PATH.read_text()) or {}
    preamble = raw.get("shared_preamble") or ""
    out: dict[str, ShapedAgent] = {}
    for name, body in (raw.get("agents") or {}).items():
        out[name] = ShapedAgent(
            name=name,
            chub_bundles=list(body.get("chub_bundles") or []),
            system_template=body.get("system_template", ""),
            model=body.get("model", "haiku"),
            timeout_s=int(body.get("timeout_s", 300)),
        )
    return preamble, out


# Loaded at import time so agent names are available as soon as the module is
# imported. Missing registry.yaml raises RuntimeError immediately — intentional
# fail-fast: the lapis-pm CLI won't start if the registry is absent.
_SHARED_PREAMBLE, _REGISTRY = _load_registry()


def list_agents() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_agent(name: str) -> ShapedAgent:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown shaped agent: {name}. Known: {list_agents()}")
    return _REGISTRY[name]


def reload_registry():
    global _REGISTRY, _SHARED_PREAMBLE
    _SHARED_PREAMBLE, _REGISTRY = _load_registry()


# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 32) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s or "task")[:max_len]


def _resolve_repo_cwd(repo: str) -> str:
    """Map a repo name to its working-clone path, falling back to DEFAULT_CWD
    when the clone doesn't exist. Returned path is what the shaped-agent
    subprocess uses as cwd — determining which CLAUDE.md + hooks fire."""
    if not repo:
        return DEFAULT_CWD
    # Accept bare name ("lapis-engine") or owner-prefixed ("Erah/lapis-engine").
    bare = repo.rsplit("/", 1)[-1]
    candidate = REPO_CWD_TEMPLATE.format(repo=bare)
    return candidate if Path(candidate).is_dir() else DEFAULT_CWD


def _compose_system(agent: ShapedAgent, vars_: dict) -> str:
    # Preamble is prepended to every agent's composed prompt. Its format
    # vars (repo, repo_cwd, target_id) are supplied by dispatch(). Missing
    # vars in the preamble are a programmer error — fail loud at format time
    # rather than silently producing a half-rendered preamble.
    preamble = _SHARED_PREAMBLE.format(**vars_) if _SHARED_PREAMBLE else ""
    base = agent.system_template.format(**vars_)
    composed = f"{preamble}\n\n---\n\n{base}" if preamble else base
    if agent.chub_bundles and select_bundles_by_ids and compose_with_system:
        try:
            sel = select_bundles_by_ids(agent.chub_bundles)
            return compose_with_system(sel, composed)
        except Exception:
            return composed
    return composed


@dataclass
class DispatchResult:
    task_id: str
    agent_type: str
    spec_path: str
    spec_id: str        # short uuid embedded in spec filename (used to find meta sidecar)
    output_path: str    # where the GPU runner will write the result


def dispatch(
    agent_type: str,
    target_id: str,
    user_prompt: str,
    vars_: dict | None = None,
    priority: int = Priority.HIGH,
    submitted_by: str = "lapis-pm",
) -> DispatchResult:
    """Submit a shaped-agent invocation to the GPU queue.

    vars_ are interpolated into the agent's system_template via str.format.
    user_prompt is passed straight through to Claude as the user message.
    """
    agent = get_agent(agent_type)
    vars_ = dict(vars_ or {})

    # Derive the subprocess cwd from the repo. The shaped agent's
    # SessionStart hooks (chub-inject.py, per-project auto-memory) key off
    # this — see REPO_CWD_TEMPLATE docstring. Always populate repo_cwd in
    # vars so the shared_preamble can reference it.
    repo_cwd = _resolve_repo_cwd(vars_.get("repo") or "")
    vars_.setdefault("repo_cwd", repo_cwd)

    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    system = _compose_system(agent, vars_)

    spec = {
        "agent_type": agent.name,
        "target_id": target_id,
        "model": agent.model,
        "timeout_s": agent.timeout_s,
        "system": system,
        "prompt": user_prompt,
        # cwd that _runner.py passes to call_claude_cli. Hooks fire against
        # the CLAUDE.md at this path — that's what pulls chubs + auto-memory
        # into the shaped agent's context.
        "cwd": repo_cwd,
        # Shaped agents run headless with no human to answer Claude Code's
        # workspace-trust prompt. `claude -p` skips that prompt's dialog and
        # returns "please allow writes to <path>" when the workspace isn't
        # cached-trusted. bypassPermissions sidesteps that — safe because
        # the authority gate upstream (spec + --authority flag) already
        # bounds what the shaped agent is allowed to do.
        "permission_mode": "bypassPermissions",
        # Fixer agents get a meta sidecar so pm_core can detect confabulation
        # (substantial prose with no tool use). Other agents skip the overhead.
        "capture_meta": agent.name == "fixer",
    }

    spec_id = uuid.uuid4().hex[:12]
    spec_path = SPEC_DIR / f"{target_id}-{agent.name}-{spec_id}.json"

    cmd = f"python3 {shlex.quote(RUNNER)} {shlex.quote(str(spec_path))}"

    # Route by agent.model. Sonnet/Haiku → ClaudeQueue (API-backed, parallel,
    # per-task worktree). Qwen → GPUQueue (GPU-serialized, no worktree).
    # LAPIS_PM_FORCE_GPU_QUEUE=1 is the emergency rollback knob that sends
    # everything back to GPUQueue.
    route_to_claude = (
        agent.model in {"sonnet", "haiku"}
        and os.getenv("LAPIS_PM_FORCE_GPU_QUEUE") != "1"
    )

    if route_to_claude:
        # Generate task_id BEFORE the single spec write so the runner cannot
        # claim a spec that's missing task_id or worktree_required. See spec
        # §Shaper routing ("Why generate before write, not after submit").
        queue = ClaudeQueue()
        task_id = queue._generate_id(slug=f"{agent.name}-{target_id}")
        spec["task_id"] = task_id
        spec["base_branch"] = "main"
        spec["worktree_required"] = True
        spec_path.write_text(json.dumps(spec, ensure_ascii=False))
        queue.submit({
            "task_type": "subprocess",
            "priority": priority,
            "timeout_seconds": agent.timeout_s + 60,
            "submitted_by": submitted_by,
            "model": agent.model,
            "description": f"{agent.name}:{target_id}",
            "notify": agent.name in {"fixer", "reviewer"},
            "payload": {"command": cmd, "spec_path": str(spec_path)},
        }, task_id=task_id)
        output_path = f"/srv/lapis/claude-queue/completed/{task_id}-output.md"
    else:
        spec_path.write_text(json.dumps(spec, ensure_ascii=False))
        queue = GPUQueue()
        task_id = queue.submit({
            "task_type": "subprocess",
            "priority": priority,
            "timeout_seconds": agent.timeout_s + 60,
            "submitted_by": submitted_by,
            "model": agent.model,
            "payload": {"command": cmd},
        })
        output_path = f"/srv/lapis/gpu-queue/completed/{task_id}-output.md"

    return DispatchResult(
        task_id=task_id,
        agent_type=agent.name,
        spec_path=str(spec_path),
        spec_id=spec_id,
        output_path=output_path,
    )


def select(event_kind: str) -> str:
    """Rule-based shape selection. v2 will learn from pm/shape-stats."""
    if event_kind == "pr_opened":
        return "reviewer"
    if event_kind == "ci_failing" or event_kind == "fixable":
        return "fixer"
    if event_kind == "spec_question" or event_kind == "explore":
        return "scout"
    return "scout"
