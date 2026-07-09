"""Golden-state sandbox code-oracle single-run executor.

Runs ONE real code-writing agent against ONE FixtureRecord inside a throwaway,
isolated git clone/worktree, judges the result with the independent
deterministic oracle from `lapis_pm.batched_fixer_eval`, records a structured
result, and resets - so different model choices can be A/B'd on identical,
held-constant tasks.

Model selection is by ENDPOINT, not by a model argument: `call_gw_agent` has
no `model` parameter, the model served is whatever `agent_backend_url` points
at. To compare e.g. Coder-Next vs Devstral, call `run_code_oracle_experiment`
twice with two different `agent_backend_url` values against the same
`fixture` and `run_id` prefix, and diff the two result dicts.

The oracle verdict (`oracle_outcome`, from `oracle_evaluate_candidate`) is the
ONLY authoritative pass/fail signal. The agent's own self-reported test run
(`agent_self_report`, from `fixer_result["last_test_outcome"]`) is recorded
for honesty-divergence analysis only - it is never used to decide pass/fail,
since the whole point of the oracle is that the agent can mis-scope or
misreport its own tests.

The shared `repo_path` working tree is never mutated: every run clones it to
a throwaway path (`dedicated_clone`) and works in a detached worktree off
that clone (`detached_worktree`); both are removed automatically when the
run's `with` blocks exit.
"""

from __future__ import annotations

import json
from pathlib import Path

from lapis_pm.batched_fixer_eval import (
    FixtureRecord,
    dedicated_clone,
    detached_worktree,
    oracle_evaluate_candidate,
)
from agents_core.gw_agent import call_gw_agent

FIXER_SYSTEM = """You are a fix agent working in a real git checkout.

You have exactly these tools: read_file, grep, write_file, apply_edit, run_tests.
You CAN and MUST edit files directly with write_file/apply_edit - you are not a
read-only reviewer. Inspect the relevant files, implement the fix for the task
described below, run the relevant tests with run_tests, and iterate until the
tests pass. Conclude when you are done.
"""


def run_code_oracle_experiment(
    fixture: FixtureRecord,
    agent_backend_url: str,
    *,
    repo_path: Path,
    out_dir: Path,
    max_steps: int = 60,
    timeout_s: int = 1800,
    run_id: str | None = None,
) -> dict:
    """Run one agent (selected by `agent_backend_url`) against one fixture.

    Sequence: throwaway clone (`dedicated_clone`) -> detached worktree at
    `fixture.parent_sha` (`detached_worktree`) -> real agent edits in place
    (`call_gw_agent(writeable=True)`) -> independent oracle verdict on the
    agent's own diff (`oracle_evaluate_candidate`, opened on its OWN fresh
    worktree at the same `parent_sha`) -> record to `out_dir` -> automatic
    reset when the clone/worktree context managers exit.

    `run_id` is used verbatim when provided (callers should pass a fixed id
    for deterministic/offline runs). If `None`, derives a stable id from the
    fixture identity: f"{fixture.repo}-{fixture.sha[:8]}".
    """
    if run_id is None:
        run_id = f"{fixture.repo}-{fixture.sha[:8]}"

    fixture_id = f"{fixture.repo}-{fixture.sha[:8]}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / f"{run_id}.transcript.json"

    with dedicated_clone(repo_path, run_id) as clone_path:
        with detached_worktree(clone_path, fixture.parent_sha, f"agent_{run_id}") as agent_wt:
            fixer_result, agent_transcript = call_gw_agent(
                prompt=(fixture.task_intent_paraphrased or fixture.task_intent_raw),
                system=FIXER_SYSTEM,
                cwd=str(agent_wt),
                writeable=True,
                backend_url=agent_backend_url,
                acquire_lease=False,
                timeout=timeout_s,
                max_steps=max_steps,
                work_id=run_id,
                on_wake_fail="skip",
            )

        apply_status, post_failures, oracle_outcome = oracle_evaluate_candidate(
            clone_path, fixture, fixer_result.get("final_diff", ""), f"oracle_{run_id}",
        )

    result = {
        "run_id": run_id,
        "fixture_id": fixture_id,
        "agent_backend_url": agent_backend_url,
        "apply_status": apply_status,
        "oracle_outcome": oracle_outcome,
        "post_failures": post_failures,
        "agent_self_report": fixer_result.get("last_test_outcome"),
        "final_diff": fixer_result.get("final_diff", ""),
        "agent_concluded": fixer_result.get("concluded"),
        "agent_max_steps_reached": fixer_result.get("max_steps_reached"),
        "agent_no_progress": fixer_result.get("no_progress"),
        "transcript_path": str(transcript_path),
    }

    result_path = out_dir / f"{run_id}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    transcript_path.write_text(json.dumps(agent_transcript, indent=2, default=str))

    return result
