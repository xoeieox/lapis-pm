"""Lapis Scout v0 — architectural fuzzing through narrative pseudocoding.

Scout designs a static scaffold (objective, sketch, conditions, available_tools,
scenario with time/external-state); Qwen simulates execution against the scaffold
over N runs; an aggregator distills runs into a failure-mode landscape Router
consults during spec design.

Public API:
  simulate(scaffold_path, ...)  → list[Path]   (runner.py)
  digest(spec_id, ...)          → Path          (digest.py)
  emit_run_ledger(...)          → str           (night_ledger.py, D1)
  emit_morning_digest(...)      → Path          (night_ledger.py, D3)

CLI: `lapis-pm scout simulate` / `lapis-pm scout digest`
"""
