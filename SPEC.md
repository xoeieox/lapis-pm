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
- **Shaped-agent dispatch** — `shaper.py` + `registry.yaml`: composes a
  chub-enriched system prompt + user prompt, writes a spec JSON, submits
  a GPU queue subprocess task whose command runs `lapis_pm._runner` against
  the spec. `_runner.py` invokes `agents_core.llm.call_claude_cli` and
  emits output + confabulation meta-sidecar.
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
  `systemd/lapis-pm.service` + `.timer` runs `tick --all` every 10 minutes.

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
   `shaper.py` adds `/data/agents/scripts` to `sys.path` at module top to
   find it. Removing this shim requires either (a) moving `chub_broker`
   into `agents-core`, or (b) publishing a `chub_broker`-compatible
   package in the chub repo. Tracked in the agents-core migration plan.

2. **lapis_pm.shaper writes to `/srv/lapis/gpu-queue/shaped/`** — the GPU queue
   runner (in conductor) reads from this path to execute `_runner.py`.
   This is shared on-disk state, not a Python coupling, and is intentional
   (GPU queue tasks are filesystem-orchestrated).

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

## Versioning

No PyPI. The git SHA is the version. `pyproject.toml` version stays at
`0.1.0` for the v0 extraction; bump on meaningful schema or API change.
