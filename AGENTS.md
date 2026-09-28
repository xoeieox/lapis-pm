# AGENTS.md - working notes for AI agents in this repo

This repo is **Lapis PM**: a dispatch-and-gate daemon plus CLI that runs
shaped coding agents against target repos, monitors their PRs, and routes
briefs back to the human owner for judgment. Two modes:

1. **Coordination (primary).** Shape a spec, pick the target repo, bind it,
   fire the initial dispatch. The daemon tick (`pm_core.tick`) then perceives
   PRs, classifies them against the authority gate, retries failed
   dispatches, and posts briefs.
2. **Daemon self-improvement.** Work on the `lapis_pm/` package itself with
   the same bind flow (`--repo lapis-pm`).

## Ground rules

- Conventional commits (`feat(scope): ...`, `fix(scope): ...`).
- The daemon tick is the SOLE dispatch/merge actor. Autopilot and helper
  units are perceive-and-write-state only.
- The authority gate (`lapis_pm/authority.py`) decides merge autonomy. Do not
  widen it without an explicit owner decision; briefs are resolved with
  `lapis-pm ratify <target_id> <outcome>`.
- Never store credentials in-repo. Host credentials live in a chmod-600 env
  file outside the repo (see `config/lapis-pm-autopilot.env.example`).

## Dev loop

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
pytest
```

Tests live in `tests/`. Keep new behaviour behind a failing test first;
`pytest` green is the bar for a PR.
