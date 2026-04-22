# lapis-pm

Per-thread project manager daemon for Lapis work threads. Runs a
**perceive → score → encode → decide → act** loop over bound targets,
dispatches shaped sub-agents (fixer / reviewer / scout) through the GPU
queue, classifies PRs against an authority gate, synthesizes briefs, and
writes arc docs for landed threads.

Extracted from `conductor/scripts/lapis_pm/` — this is the agent's own repo
now that `agents-core` exists to carry the shared primitives.

## Install

```bash
pip install -e /srv/lapis/lapis-pm --user --break-system-packages
```

After install, `python3 -m lapis_pm.cli --help` works from any cwd.

## CLI

```bash
lapis-pm bind <target_id> --spec-from PATH|- --repo REPO [--authority advisory|auto]
lapis-pm unbind <target_id>
lapis-pm tick [--target ID | --all] [--force-brief] [--force-dispatch AGENT:INTENT]
lapis-pm status [target_id] [--explain]
lapis-pm pause <target_id> [--reason TEXT]
lapis-pm resume <target_id>
lapis-pm list
lapis-pm land <target_id> [--dry-run]
```

The `lapis-pm` shell wrapper at `bin/lapis-pm` sources `conductor.env`
(FORGEJO_TOKEN, PUSHOVER_\*) and execs `python3 -m lapis_pm.cli`. Symlink
it to `/usr/local/bin/lapis-pm` for manual use.

## Service

Runs every 10 minutes via `systemd/lapis-pm.service` + `.timer`:

```bash
sudo cp systemd/lapis-pm.service systemd/lapis-pm.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lapis-pm.timer
```

## Smoke test

```bash
bash lapis_pm/smoke.sh
```

Exercises bind → dispatch → encode → pause/resume → directive → brief → status
end-to-end. No real PRs or GPU calls; queue-paused sandbox.

## Structure

```
lapis_pm/
├── __init__.py
├── cli.py              CLI dispatch (bind, tick, land, status, ...)
├── pm_core.py          perceive → score → encode → decide → act loop
├── episodic.py         CommentStore adapter (5-factor recall, tag taxonomy)
├── shaper.py           shaped-agent dispatch via GPU queue + registry.yaml
├── _runner.py          subprocess target for shaped-agent GPU tasks
├── authority.py        PR classification: auto / advisory / hold
├── brief.py            pm:brief synthesis + Pushover delivery
├── land.py             arc-doc synthesis to /srv/lapis/lapis-state/<tid>.md
├── registry.yaml       shaped-agent definitions (fixer, reviewer, scout)
└── smoke.sh            end-to-end smoke test
bin/lapis-pm             shell wrapper (sources env)
systemd/                 service + timer units
SPEC.md                  identity + what is owned vs. not + known couplings
CLAUDE.md                role spec for PM-shaped sessions in this repo
```

## Dependencies

- `agents-core` — shared primitives (llm, forgejo, notify, targets, comments, gpu, mem)
- `pyyaml` — registry.yaml parsing

`shaper.py` has a runtime dep on `chub_broker`, which still lives in
`/data/agents/scripts/` (deferred from agents-core day-one scope). See
`SPEC.md` § "Known couplings".
