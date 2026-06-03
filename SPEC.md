# SPEC — lapis-pm

## Identity

**lapis-pm** is the per-thread project manager agent for Lapis work threads.
It runs the **perceive → score → encode → decide → act** loop for each bound
target: watching associated Forgejo PRs + `/srv/lapis/targets/comments/<tid>.jsonl`
episodes, dispatching shaped sub-agents (fixer / reviewer / scout) through
the GPU queue, classifying PRs against an authority gate (auto-merge /
advisory / hold), and synthesizing briefs into `pm:brief` comments with
Pushover delivery.

A PM-shaped human Claude Code session opens in a product repo (e.g. `lapis-engine`)
via the thin-`CLAUDE.md` + chub-inject pattern, shapes the spec + authority
in conversation with Erah, then hands off to this daemon via `lapis-pm bind`.
The daemon takes over monitoring. When the thread lands, `lapis-pm land`
synthesizes an arc doc at `/srv/lapis/lapis-state/<tid>.md` (RoomRAG-indexed).

## What lapis-pm owns

- **The PM loop** — `pm_core.tick()`: perceive (episodic recall + PR
  perception), score, encode (pm:observation / pm:dispatch / pm:result /
  pm:brief comments), decide (at most one action per tick), act.
- **Episodic memory adapter** — `episodic.py`: wraps the shared CommentStore
  with PM-specific tagging, 5-factor recall, and classification of events
  into encoding-trigger categories.
- **Shaped-agent dispatch** — `agents_core.shaper.Shaper` + `registry.yaml`:
  composes a chub-enriched system prompt + user prompt, writes a spec JSON,
  submits a queue subprocess task whose command runs
  `python3 -m agents_core.shaped_runner` against the spec.
  `agents_core.shaped_runner` invokes `agents_core.llm.call_claude_cli` and
  emits output + confabulation meta-sidecar. (Moved from `lapis_pm/shaper.py`
  + `_runner.py` → `agents_core` 2026-04-28.)
- **Authority gate** — `authority.py`: classifies PRs against held paths,
  size, CI gates, and a Sonnet verdict (`call_claude_cli` with `json_mode`).
- **Briefs** — `brief.py`: Haiku-based synthesis from spec + recent PM
  episodes + triggering event, written as a `pm:brief` comment with
  Pushover delivery.
- **Arc-doc synthesis** — `land.py`: Haiku over the full episodic chronology
  of a landed thread, written to `/srv/lapis/lapis-state/<tid>.md`.
- **CLI + systemd** — `cli.py` exposes `bind/unbind/tick/status/pause/resume/
  list/land`. `bind` accepts `--create` to create the target YAML and bind in one
  step (eliminates the historic two-step `TargetStore.create()` + `bind` flow).
  `bind --adopt-pr <num>` adopts an already-open PR on any branch: the next tick
  perceives that PR (via `adopted_pr_number` / `adopted_head_branch` on the target
  state) and dispatches a reviewer instead of the initial fixer. Use this for
  freehand `feat/*` PRs created outside the PM loop during BRIX cutover — no
  manual branch re-homing required. See README "Adopting a freehand PR".
  `systemd/lapis-pm.service` + `.timer` runs `tick --all` every 10 minutes.
- **Router portfolio auto-fill** — `router_portfolio.py` (`router-portfolio-persistence-v0`,
  M0.5) provides `emit_*` helpers that are wired into the three main CLI paths:
  `bind` emits a `kickoff` decision entry (one per target / chain leg),
  `tick --force-dispatch` emits a `dispatch` entry (fragment_id derived from agent_type
  and dispatch context), and `land` emits a `land` entry. All emits are best-effort
  (wrapped in try/except; failures log `[router-portfolio:emit-failed]` to stderr and
  do not block the primary command). Portfolio entries accumulate in mem.db under
  `router/lapis-pm/decisions/` and are queryable via `mem search "router/lapis-pm/decisions"`.
  Daemon-side wires (tick-loop dispatches, chain auto-advance) are deferred to
  `router-portfolio-wire-emitters-v1`.

## What lapis-pm does NOT own

- **Primitives** — `llm.call_claude_cli`, `forgejo.*`, `notify.*`, `targets.
  TargetStore`, `comments.CommentStore`, `gpu.GPUQueue`, `mem.MemoryStore`
  all come from `agents-core`.
- **Chub bundles** — loaded from `/data/agents/chub-registry/` by
  `chub_broker` (deferred). See "Known couplings".
- **Intention registry / ops-layer** — ops_primitives, ops_weather, and
  intention_registry live in the ops-layer repo (or still under
  `/data/agents/scripts/` pre-extraction). lapis-pm is a *consumer* of
  the GPU queue's intention coordination; it doesn't implement intentions.
- **Fixer / reviewer / scout personalities** — those are shaped via
  `registry.yaml` system templates + chub bundles, not by custom code.
  lapis-pm is the dispatcher, not the fixer.

## Known couplings (tracked as follow-ups)

1. **chub_broker** still lives at `/data/agents/scripts/chub_broker.py`.
   `agents_core.shaper` adds `/data/agents/scripts` to `sys.path` at module
   top to find it. Removing this shim requires either (a) moving `chub_broker`
   into `agents-core`, or (b) publishing a `chub_broker`-compatible
   package in the chub repo. Tracked in the agents-core migration plan.

2. **agents_core.shaper writes to `/srv/lapis/gpu-queue/shaped/`** — the GPU queue
   runner (in conductor) reads from this path to execute
   `python3 -m agents_core.shaped_runner`. This is shared on-disk state, not
   a Python coupling, and is intentional (GPU queue tasks are
   filesystem-orchestrated).

3. **Ops-layer integration** — `agents_core.gpu.GPUQueue.submit` probes
   `intention_registry` via `sys.path` when available. lapis-pm benefits
   from intention coordination (dedup-via-reinforce) but doesn't configure
   it. When ops-layer moves to its own repo, this coupling gets inverted
   (agents-core exposes a hook, ops-layer registers).

## Install

Editable install alongside `agents-core`:

```bash
pip install -e /srv/lapis/lapis-pm --user --break-system-packages
```

After install: `python3 -m lapis_pm.cli --help` works from any cwd. The
CLI wrapper at `bin/lapis-pm` adds env-loading; symlink it to
`/usr/local/bin/lapis-pm` for manual use.

## Governance

- PRs required to merge into `main` (branch protection).
- Schema changes to `pm/cursor/` / `pm/dispatched/` / `pm/pause-state/` /
  `pm/outstanding-brief/` mem.db keys or to PM comment tag taxonomy
  (`pm:observation`, `pm:dispatch`, `pm:result`, `pm:brief`, `pm:hold`,
  `pm:merge`, `pm:retry`, `spec:bound`, `human:directive`) are breaking
  changes. Prefix PR titles with `schema:` so reviewers notice.
- The `registry.yaml` agent definitions are the PM's contract with the
  shaped sub-agents. Changes there may shift fixer/reviewer/scout
  behavior at runtime — prefer additive changes.
- **Reviewer is read-only; fixer is write-only; never merge the roles.**
  The reviewer agent returns a verdict JSON and nothing else. Any reviewer
  template change that adds tool instructions beyond read/report is a
  violation of the role-separation invariant.
- **classified-prs is invalidated by SHA advance.** When a PR's head SHA advances, its entry is removed from `pm/classified-prs/<tid>` so the next decide loop re-screens the new code instead of noop'ing on a stale brief.

## Versioning

No PyPI. The git SHA is the version. `pyproject.toml` version stays at
`0.1.0` for the v0 extraction; bump on meaningful schema or API change.
