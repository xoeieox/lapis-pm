# Backcaster v0

Backcaster is the inverse operator for goal-state decomposition in the Lapis ecosystem.

Given a goal-state (what you want to be true), it asks: **what must exist for
this goal-state to be reachable?**

The defining commitment: **resist the everything-is-software reflex.** Components
are categorized across 7 distinct kinds of work, and every run produces a
category histogram that makes software-bias visible at a glance.

## Usage

```bash
lapis-pm backcaster <goal-file> [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--axes ax1,ax2,...` | all 6 | Axes to decompose: `psychological,material,infrastructural,governance,social-norm,economic` |
| `--corpus PATH` | library/civic-theory | Corpus dir for Synapse retrieval |
| `--scenarios RUN_IDS` | (none) | Civic-sim run IDs (v0 no-op) |
| `--model qwen\|sonnet\|opus` | qwen | LLM model |
| `--out DIR` | auto | Override output directory |

### Stub mode

```bash
BACKCASTER_STUB=1 lapis-pm backcaster lapis_pm/backcaster/fixtures/stub-goal.md
```

Bypasses all LLM calls and returns canned outputs. Use for CI / smoke tests.

## Output layout

Each run writes to `/srv/lapis/backcaster/runs/<YYYY-MM-DD-HHMM>-<slug>/`:

```
goal.md             # goal-state text (input snapshot)
decomposition.yaml  # per-axis preconditions
gaps.yaml           # per-precondition gap analysis with citations
components.yaml     # derived components with category + effort + reversibility
roadmap.md          # human-readable synthesis (the primary output)
histogram.yaml      # category + financial-subtype distribution counts
run.yaml            # run metadata: model, corpus, timing, epistemic_caution,
                    # concentration_warning, degraded_paths, derive_fallback_count
```

## The 7 component categories

| Category | Meaning |
|---|---|
| `software` | A tool, system, code change |
| `policy` | A rule, agreement, contract |
| `community-formation` | A group, network, practice |
| `research` | Knowledge to be gathered or generated |
| `infrastructure` | Physical capacity (compute, space, energy) |
| `cultural-shift` | A normative change in practice or expectation |
| `financial` | A value-flow component (funding source, income structure, etc.) |

**Financial subtypes** (required for all `financial` components):

| Subtype | Meaning |
|---|---|
| `extractive` | Captures value out of communities to concentrate it |
| `distributive` | Circulates value through communities without concentration |
| `neutral` | No clear extractive/distributive shape |

## Category histogram inspection

The `histogram.yaml` is the primary inspection surface for "are we just
building software?" Open it immediately after a run:

```yaml
community-formation: 2
cultural-shift: 2
financial: 2
financial.distributive: 1
financial.extractive: 0
financial.neutral: 1
infrastructure: 0
policy: 2
research: 1
research.fallback: 0
software: 1
```

A `software-dominant` warning in `run.yaml:concentration_warning` means > 70%
of components landed as software — this is the bias to resist.

## Epistemic caution

`run.yaml:epistemic_caution` surfaces how sourced the gap analysis is:

- `low` - fewer than 20% of gaps are unsourced
- `medium` - 20-50% unsourced
- `high` - more than 50% unsourced (most common in v0 when Synapse is unavailable)

## Fail-soft behavior

Backcaster degrades gracefully:

- **Synapse unreachable**: `run.yaml:degraded_paths` includes `synapse`; gap analysis
  runs LLM-only with `unsourced: true` on all gaps.
- **mem.db unreachable**: `run.yaml:degraded_paths` includes `mem`; no ecosystem
  context provided to LLM.
- **LLM parse failure**: component deriver retries once, then writes a fallback
  component with `category: research, fallback: true`. Count in
  `run.yaml:derive_fallback_count`.

## Architecture

```
lapis_pm/backcaster/
  __init__.py
  schema.py       # Pydantic models: Precondition, Gap, Component, Histogram, RoadmapRun
  decompose.py    # Stage 1: goal-state → preconditions per axis
  gap_analyze.py  # Stage 2: precondition → gap (Synapse + mem retrieval)
  derive.py       # Stage 3: gap → components (retry-then-fallback)
  compose.py      # Stage 4: synthesis → roadmap.md + histogram
  runner.py       # Orchestration + run-dir writes
  cli.py          # CLI argument handlers
  fixtures/
    stub-goal.md  # DoD worked example goal-state
```

Backcaster is read-only with respect to the Lapis ecosystem — it does not open
PRs, bind targets, or write to mem.db. It is purely analytical.
