# Lapis Scout v0

**Architectural fuzzing through narrative pseudocoding.**

Router designs a static scaffold (objective, sketch, conditions, available_tools,
scenario with time/external-state); Qwen simulates execution against the scaffold
over N runs; an aggregator distills runs into a failure-mode landscape Router
consults during spec design.

## Usage

```bash
# Run a full scaffold (all matrix cells)
lapis-pm scout simulate /srv/lapis/scout/sims/canonicalizer-v0.yaml

# Run a specific cell (smoke / iteration)
lapis-pm scout simulate /srv/lapis/scout/sims/canonicalizer-v0.yaml \
  --cell "opt=weight-by-recency,load=5,severity=degraded" --runs 1

# Digest all traces for a spec into a failure-mode map
lapis-pm scout digest canonicalizer-v0
```

## Scaffold YAML schema

See the spec at `/srv/lapis/planning/specs/lapis-scout-v0.md` for the full schema.
Quick reference:

```yaml
spec_id: my-architecture-v0
spec_version: v0
description: |
  One-paragraph description of what is being tested.

static_scaffold:
  objective: "What the architecture does"
  architecture_sketch: |
    Multi-line description of the architecture shape.
  necessary_steps:
    - id: step-id
      description: "What this step does"
  optional_steps:
    - id: opt-step-id
      description: "What this optional step does"
  available_tools:
    - id: tool_name
      description: "What this tool does"
  scenario:
    conditions:
      - "Constraint the scenario must satisfy"
    time_progression:
      t0: "initial state"
      t0+30s: "something changes"
    external_state:
      some_system:
        status: healthy
    utilization_pattern: "Pattern description"

generation_directive: |
  Simulate executing this architecture. Report strict JSON.

matrix:
  optional_steps_included:
    - []
    - [opt-step-id]
  external_state_severity: [healthy, degraded]
  concurrent_load: [1, 5]
  runs_per_cell: 5
```

## Storage

```
/srv/lapis/scout/
  sims/<spec-id>.yaml              # scaffold definitions
  traces/<spec-id>/<cell-id>/<run-id>.json  # raw traces (LapisToolReturn)
  maps/<spec-id>.yaml              # digested map (LapisToolReturn)
```

**`/srv/lapis/scout/` is NOT RoomRAG-indexed** — Qwen-narrated content stays out of
any RoomRAG-indexed tree per `feedback/qwen-narration-out-of-router-corpus`.

## Night queue

The night-queue orchestrator replaces `scripts/scout_night.sh` with a Python
process that gates on llama-server health, quarantines bad scaffolds, and
schedules cells in round-robin across scaffolds.

```bash
# Run until 09:00 local (default)
lapis-pm scout night run

# Run each scaffold once and exit (smoke / CI)
lapis-pm scout night run --once --sims-dir /srv/lapis/scout/sims

# Check status of the most recent night run
lapis-pm scout night status
lapis-pm scout night status --json   # structured output for claude-view

# Clear a quarantined scaffold after amending its YAML
lapis-pm scout night quarantine-clear <spec_id>
```

### Priority profiles

Each scaffold YAML may declare `priority_profile` at the top level:

| Value | Behaviour |
|---|---|
| `full-pass-once` (default) | Run each cell once, then done. |
| `variance-resolution` | Warmup pass + re-run cells whose break signatures varied across runs. |
| `continuous-baseline` | Cycle indefinitely until the run wall-clock expires. |

Unknown values raise `ValueError` at load time. Existing scaffolds without the
field default to `full-pass-once` and require no edits.

### Failure modes closed

1. **Health gate** — exponential back-off (5s base, 120s ceiling); 1-hour
   wall-clock abort budget if llama-server stays down.
2. **Quarantine** — 3 consecutive zero-parse runs → scaffold quarantined for
   the night; clears via `quarantine-clear` after operator amendment.
3. **Round-robin scheduling** — cells interleaved across scaffolds; 24-cell
   scaffolds no longer starve behind 216-cell scaffolds.
4. **SIGTERM** — handled between WorkUnits; in-flight simulate() runs to
   completion before the orchestrator exits.
5. **GPU contention** — reads GPUQueue read-only; yields when another process
   holds the GPU, resumes when clear.

## Provenance conformance

Every Scout output is a `LapisToolReturn` (payload + non-empty summary +
Provenance with required fields). See `Lapis/Architecture-Provenance-Schema-v0.md`.

- `manifest_hash`: sorted-keys JSON, no whitespace, shortest-round-trip floats.
- `prompt_hash`: sha256 of null-byte-delimited concatenation of tick prompts.
- `scaffold_hash`: sha256(canonical_json(static_scaffold + cell-resolved-matrix-row)).
- Digest: traces appear in BOTH `input_refs` (type=`tool_result`, `content_hash`) AND
  `upstream_calls` (`manifest_hash` composition pointer).
