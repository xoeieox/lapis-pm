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

### Adopting a freehand PR (post-cutover workflow)

During BRIX cutover, work done in freehand Claude Code sessions produces PRs on
`feat/*` branches outside the PM loop. Use `--adopt-pr` to bring them in without
manual branch re-homing:

```bash
lapis-pm bind <target_id> \
  --spec-from /srv/lapis/planning/specs/<target_id>.md \
  --repo <repo> --authority <auth> \
  --create --title "<title>" \
  --adopt-pr <pr_number>
```

The next tick perceives the adopted PR on its existing branch and dispatches a
**reviewer** (not the initial fixer). The loop then runs the normal
review → `fixer_retry` → review bounce against the freehand branch.

Constraints: the PR must be open (not closed/merged) and its repo must match
`--repo`. The `--adopt-pr` flag is mutually exclusive with an already-open
initial dispatch that has created a `lapis/<tid>/` branch.

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

## Tick Decision Taxonomy

Every tick emits exactly one decision log line per bound target:

```
[<target_id>] skipped=<bool> reason=<reason> encoded=<n> decision=<tag>
```

### Noop variants (`decision=noop:*`)

| Tag | Meaning |
|-----|---------|
| `noop:no_change` | Perceived state unchanged; nothing to do (healthy common case). |
| `noop:paused` | Target's `paused: true` flag is set. |
| `noop:reviewer_in_flight:pr=<n>:cycle=<k>` | PR open, reviewer agent dispatched, waiting for verdict. |
| `noop:fixer_in_flight:dispatch=<id>` | Fixer retry dispatched (reviewer returned `fixable`), queue job not yet terminal. |
| `noop:awaiting_chain_dependency:waiting_on=<tid>` | Chain-mode leg: `depends_on` not yet satisfied. |

### Action variants (`decision=action:*`)

| Tag | Meaning |
|-----|---------|
| `action:auto_merge:pr=<n>` | PR auto-merged by the daemon. |
| `action:merge_failed:<err>` | Auto-merge attempted but Forgejo returned an error. |
| `action:auto_land:pr=<n>:arc=<path>` | PR merged + arc doc written + target unbound. |
| `action:reviewer_dispatched:pr=<n>:cycle=<k>` | Opus reviewer agent dispatched for PR review cycle k. |
| `action:fixer_dispatched:source=retry:pr=<n>:cycle=<k>` | Fixer retry dispatched after reviewer returned `fixable`. |
| `action:fixer_dispatched:source=init:dispatch=<id>` | Initial or re-tried fixer dispatch (no open PR yet). |
| `action:brief_emitted:kind=<kind>:cid=<id>` | Brief synthesized and posted. `kind` ∈ `hold`, `advisory_clean`, `advisory_screen_issue`. |
| `action:brief_decision_applied:<brief_id>:<option_id>` | Click-to-resolve directive consumed and applied. |
| `action:directive_brief:cid=<id>` | Human directive received; brief synthesized for human review. |
| `action:abandon_brief:cid=<id>` | Fixer failed `MAX_DISPATCH_RETRIES` times; brief posted, dispatch abandoned. |
| `action:review_exhausted_brief:cid=<id>` | Review cycle budget exhausted; human judgment needed. |
| `action:review_gate_paused:cid=<id>` | Kill-switch threshold exceeded; review gate soft-paused. |
| `action:review_gate_pause:already_briefed` | Kill-switch already triggered this period; idempotent. |

### Skip variants (`skipped=True reason=*`)

| Reason | Meaning |
|--------|---------|
| `target not found` | Target ID not in the target store. |
| `target not pm_bound` | Target exists but `pm_bound` is false. |
| `paused` | Target is paused (`decision=noop:paused` is also set). |
| `forgejo_unreachable` | Forgejo health gate blocked the tick (added by `lapis-pm-forgejo-health-gate`). |
| `ratelimit` | Rate limiter blocked the tick. |
| `cursor_locked` | Concurrent-tick guard fired. |

**Invariants:** Every tick emits exactly one decision tag per target. The `noop:` and `action:` families are disjoint and machine-greppable. Bare `noop` (without `:` qualifier) is a programming error; unit tests enforce the closed enum.

## fixer_retry completion semantics

A `fixer_retry` is considered complete when the PR advances in either of two ways:

- **Commit-based (SHA advance):** the fixer pushes a new commit; the PR head SHA changes.
  Perceived via `pm:pr=<n>:sha=<sha>` observation.
- **Description-based (body advance):** the fixer edits the PR body to resolve the
  reviewer's issue (e.g. adding justification text); the body fingerprint changes.
  Perceived via `pm:pr=<n>:body=<fp>` observation (`fp = sha256(body)[:16]`).

Both paths flip the `fixer_retry` record to `processed`, call
`_close_slot_and_deposit` to close the Zephyr attribution slot, and advance the
reviewer-fixer bounce to the next reviewer cycle. A description-only completion
writes a distinct episodic observation:
`"Fixer retry for PR #N completed: PR description advanced after dispatch"`.

A `fixer_retry` that is terminal in the queue (job `processed`/`failed`) with
**neither** a SHA advance **nor** a description advance is classified as "lost" by
`_find_lost_fixer_dispatches` (which now covers `fixer_retry` in addition to the
initial `fixer`). Resolution: retry once (incrementing `lost_retry_count` on the
record and carrying `pr_number` to the child dispatch), then raise a brief. The
terminal path also calls `_close_slot_and_deposit` so the attribution slot closes
regardless of outcome. A `pending` fixer_retry is never preempted by the lost-net.

## Structure

```
lapis_pm/
├── __init__.py
├── cli.py              CLI dispatch (bind, tick, land, status, ...)
├── pm_core.py          perceive → score → encode → decide → act loop
├── episodic.py         CommentStore adapter (5-factor recall, tag taxonomy)
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

`agents_core.shaper` (now the canonical shaper — promoted from `lapis_pm/shaper.py`
2026-04-28) has a runtime dep on `chub_broker`, which still lives in
`/data/agents/scripts/` (deferred from agents-core day-one scope). See
`SPEC.md` § "Known couplings".
