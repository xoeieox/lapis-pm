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
  maps/<spec-id>.json              # digested map (LapisToolReturn)
```

**`/srv/lapis/scout/` is NOT RoomRAG-indexed** — Qwen-narrated content stays out of
any RoomRAG-indexed tree per `feedback/qwen-narration-out-of-router-corpus`.

## Provenance conformance

Every Scout output is a `LapisToolReturn` (payload + non-empty summary +
Provenance with required fields). See `Lapis/Architecture-Provenance-Schema-v0.md`.

- `manifest_hash`: sorted-keys JSON, no whitespace, shortest-round-trip floats.
- `prompt_hash`: sha256 of null-byte-delimited concatenation of tick prompts.
- `scaffold_hash`: sha256(canonical_json(static_scaffold + cell-resolved-matrix-row)).
- Digest: traces appear in BOTH `input_refs` (type=`tool_result`, `content_hash`) AND
  `upstream_calls` (`manifest_hash` composition pointer).
