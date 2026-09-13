"""Lapis PM tick orchestration.

One tick per bound target: perceive → score → encode → decide → act.

State across ticks lives in mem.db:
  pm/cursor/<target_id>            ISO timestamp; comments at or before this
                                   were already considered.
  pm/dispatched/<target_id>        JSON list of dispatch records.
  pm/pause-state/<target_id>       "paused" | "active" — transition detector.
  pm/outstanding-brief/<target_id> Comment id of the most recent unanswered brief.

Single-action discipline: at most one decision-action per tick. Encoding
percepts as comments is bookkeeping, not action.

Decision taxonomy (see README.md § "Tick Decision Taxonomy"):
  Noop:   noop:no_change | noop:paused | noop:reviewer_in_flight:pr=N:cycle=K
          | noop:fixer_in_flight:dispatch=ID | noop:awaiting_chain_dependency:waiting_on=TID
          | noop:reviewer_infra_backoff:until=ISO8601 (spaced infra-retry, see R2 below)
  Action: action:auto_merge:pr=N | action:auto_land:pr=N:arc=PATH
          | action:needs_review:pr=N:reason=... (merge conflict, cannot merge cleanly)
          | action:merge_attempted:mergeability_unknown:pr=N (mergeable=None, indeterminate)
          | action:merge_failed:... (merge API/transient failure)
          | action:reviewer_dispatched:pr=N:cycle=K
          | action:fixer_dispatched:source=(init|retry):...
          | action:brief_emitted:kind=KIND:cid=CID | action:brief_decision_applied:BID:OID
          | action:directive_brief:cid=CID | action:abandon_brief:cid=CID | ...
  Skip:   skipped=True reason=(target not found|target not pm_bound|paused
          |forgejo_unreachable|ratelimit|cursor_locked)
"""

from __future__ import annotations

import contextlib
import fcntl
import filecmp
import fnmatch
import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import yaml
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

from agents_core.targets import TargetStore
from agents_core.mem import MemoryStore
from agents_core.notify import Priority as NotifyPriority

try:
    from agents_core.forgejo import get_open_prs, merge_pr, get_pr as _forgejo_get_pr, get_branch as _forgejo_get_branch
except Exception:
    get_open_prs = None  # type: ignore
    merge_pr = None  # type: ignore
    _forgejo_get_pr = None  # type: ignore
    _forgejo_get_branch = None  # type: ignore

try:
    # Decoupled from the import block above: friction harvest is additive and
    # must never take down get_open_prs/merge_pr if get_pr_files is absent
    # from a given agents-core deploy.
    from agents_core.forgejo import get_pr_files as _forgejo_get_pr_files
except Exception:
    _forgejo_get_pr_files = None  # type: ignore

try:
    from agents_core.claude_queue import ClaudeQueue as _ClaudeQueue
except Exception:
    _ClaudeQueue = None  # type: ignore

try:
    # Reuse the runner's own stale-active arithmetic (same threshold
    # startup_sweep uses to decide a task's runner has died) rather than
    # inventing a second number. Import is best-effort — a missing symbol
    # falls back to the documented default (300s) rather than breaking
    # reconciliation entirely.
    from agents_core.claude_queue_runner import STARTUP_STALE_GRACE_S
except Exception:
    STARTUP_STALE_GRACE_S = 300  # type: ignore

from agents_core.shaper import Shaper, DispatchResult as _DispatchResult, _bool_val  # noqa: F401

try:
    from agents_core.room_paths import room_path, room_str
except ImportError as _e:
    raise RuntimeError("agents_core.room_paths missing — agents-core seam must be deployed first") from _e

from . import episodic, brief, authority, intent_artifact as _intent_artifact, steer
from . import node_identity
from . import signed_directive

try:
    from . import eval_gate as _eval_gate
except Exception:
    _eval_gate = None  # type: ignore

# Module-level singleton — constructed at import time so a malformed
# registry.yaml crashes the daemon immediately, not at first dispatch.
_SHAPER = Shaper(Path(__file__).parent / "registry.yaml")

# Targets for which a config-404 (repo-unresolved) signal has already been
# emitted this process lifetime. Prevents per-tick spam on persistent mis-binding.
_repo_unresolved_signalled: set[str] = set()


PACIFIC = ZoneInfo("America/Los_Angeles")
COMPLETED_DIR = room_path('gpu_queue.completed')
FAILED_DIR = room_path('gpu_queue.failed')
SHAPED_DIR = room_path('gpu_queue.shaped')  # shaped_runner meta sidecars
CLAUDE_QUEUE_DIR = room_path('claude_queue')  # claude-queue enqueue root
CLAUDE_QUEUE_COMPLETED_DIR = room_path('claude_queue.completed')
CLAUDE_QUEUE_FAILED_DIR = room_path('claude_queue.failed')
MAX_DISPATCH_RETRIES = 2

# Agent types that count as "initial fixer" for dispatch guards, lost-fixer
# detection, and concurrency checks. A single constant so all sites stay in sync.
_INITIAL_FIXER_TYPES = ("fixer", "fixer_local")

# Every agent_type that dispatches a local-reviewer read-only judge leg.
# reviewer_fresh_contractor (formerly the TOU-peak/dead-seat contractor
# route, lapis-pm-reviewer-peak-contractor-route-v0 /
# lapis-pm-reviewer-seat-phala-test-key) is kept here as a read-only
# parsing artifact only (lapis-pm-phala-test-key, D4) — no live
# code path produces this agent_type anymore. It stays in the tuple so
# historical dispatch records containing it still classify correctly
# (attempt ceiling, verdict decode, output-file
# processing) instead of silently falling outside that accounting.
_REVIEWER_AGENT_TYPES = ("reviewer", "reviewer_fresh", "reviewer_fresh_contractor")

# lapis-pm-reviewer-absence-grounding-rule-v0: flag-gated obligation block,
# rendered into the reviewer/reviewer_fresh system_template only when the
# registry's absence_grounding field is true (default false — see
# registry.yaml). Reference wording is the 2026-08-08 bakeoff prompt
# (finding/deepseek-vs-27b-reviewer-seat-bakeoff-outside-harness-2026-08-08),
# adapted: the lapis reviewer dispatch runs inside a worktree already
# checked out to the PR head, so the command form is the grep tool or
# `git grep` without a ref, not the bakeoff's branch-qualified form.
ABSENCE_GROUNDING_BLOCK = (
    "\n\n"
    "## HARD RULE — grounding absence claims\n\n"
    "Any finding of the form \"X is not defined\", \"X is never called\", "
    "\"X is not wired in\", \"I could not confirm X exists\", or \"there is "
    "no handler for X\" is INVALID unless you first run an explicit search "
    "over the FULL BRANCH and quote its output.\n\n"
    "Rules that follow from this:\n"
    "- A symbol being absent from the DIFF does not mean it is absent from "
    "the CODEBASE.\n"
    "- A symbol's definition and its call site are frequently hundreds of "
    "lines apart and in different hunks. Read the whole file before "
    "concluding anything is unwired.\n"
    "- If your search returns any hit you did not expect, the finding is "
    "withdrawn.\n"
    "- For every absence-style finding you report, include the exact "
    "command you ran and its output in the issue's `evidence` field "
    "(optional on every other issue; required here). No evidence, no "
    "finding.\n\n"
    "Use your grep tool, or `git grep` (no ref needed — you are already "
    "checked out to the branch head), to run the search. A failed or "
    "errored search is not evidence of absence; do not report the finding "
    "if the search itself failed.\n"
)


def _absence_grounding_enabled(agent_type: str) -> bool:
    """Read the flag-gated absence_grounding field for a reviewer registry entry.

    Reads registry.yaml directly rather than through _SHAPER.get_agent —
    ShapedAgent (agents_core/shaper.py) doesn't carry ad-hoc fields, only the
    ones it explicitly parses. Value is parsed through the existing shaper
    boolean parser (_bool_val, agents_core/shaper.py:68) rather than a new
    ad-hoc truthiness check, per DoD 1.

    reviewer_fresh_contractor has no field of its own — it shares the
    reviewer_fresh system_template via YAML anchor (registry.yaml), so its
    flag state is looked up under the "reviewer_fresh" family name by the
    caller (see _absence_grounding_block_for).
    """
    try:
        raw = yaml.safe_load((Path(__file__).parent / "registry.yaml").read_text()) or {}
        body = (raw.get("agents") or {}).get(agent_type) or {}
        return _bool_val(body.get("absence_grounding", False))
    except Exception:
        return False


def _absence_grounding_block_for(agent_type: str) -> str:
    """Return the rendered {absence_grounding_block} var value for a reviewer dispatch.

    Empty string when the flag is off (or agent_type isn't a reviewer seat at
    all) — this is what keeps the flag-off render byte-identical to today's
    template (DoD 2). reviewer_fresh_contractor maps to the "reviewer_fresh"
    family flag since it shares that template and carries no field of its own.
    """
    if agent_type not in _REVIEWER_AGENT_TYPES:
        return ""
    family = "reviewer_fresh" if agent_type in ("reviewer_fresh", "reviewer_fresh_contractor") else "reviewer"
    return ABSENCE_GROUNDING_BLOCK if _absence_grounding_enabled(family) else ""


# Synth-fail counter constants
_SYNTH_FAIL_KEY = "pm/brief/synth-fail-count/{}"
_SYNTH_FAIL_THRESHOLD = 3

# Cycle budgets per authority level (number of reviewer dispatches before exhausted)
_REVIEW_CYCLE_BUDGETS: dict[str, int] = {
    "advisory": 2,
    "hold": 4,
}

# Reviewer mode per authority level (fresh-reviewer = cold full-diff each cycle)
_REVIEWER_MODES: dict[str, str] = {
    "advisory": "same-reviewer",
    "hold": "fresh-reviewer",
}

# Reviewer attempt ceiling (lapis-pm-reviewer-attempt-ceiling-v0): bounds
# *attempts* (dispatches), not completed cycles. A reviewer that fails before
# ever writing a verdict never advances _reviewer_cycle_count, so without this
# a persistently-failing reviewer is redispatched every tick forever. Counter
# is PR+cycle-bound and survives new SHAs — only an explicit manual clear
# (`lapis-pm clear-reviewer-attempts <target_id>`) resets it. See spec
# "Reset scope" ruling: a SHA-triggered reset would grant amnesty to a
# code-level defect on every push.
REVIEWER_ATTEMPT_CEILING_DEFAULT = 2
REVIEWER_ATTEMPT_CEILING_ENV = "LAPIS_PM_REVIEWER_ATTEMPT_CEILING"

# Infrastructure non-runs (GravityWell unavailable, doorman disagreement, etc.)
# must not consume the reviewer-attempt ceiling above — that ceiling exists to
# bound a reviewer that keeps failing to produce a usable verdict, not a
# reviewer that never got to run at all (lapis-pm-reviewer-attempts-not-consumed-
# by-infra-v0). Infra-classified attempts get their own, more generous budget
# so an endlessly-absent GravityWell still produces a bounded number of
# dispatches rather than an unbounded loop. 6 is roughly 3x the reviewer
# ceiling (2) — generous enough to ride out a transient doorman/wake-gravitywell
# hiccup without burning attempts, but still a hard stop, not a retry-forever.
REVIEWER_INFRA_RETRY_BUDGET_DEFAULT = 6
REVIEWER_INFRA_RETRY_BUDGET_ENV = "LAPIS_PM_REVIEWER_INFRA_RETRY_BUDGET"

# Reviewer defer backoff (lapis-pm-reviewer-defer-backoff-v0, R1): the infra
# budget above bounds *how many* infra-classified attempts happen, but says
# nothing about *when* — before this, consecutive attempts fired every tick,
# so the whole budget spent itself in minutes against gate occupancies
# (spec-review protected leases on GravityWell) that run 40+ minutes. This
# explicit lookup table — not a formula, per the spec's gate finding — is
# indexed by infra_count (1-based: the Nth consecutive infra failure waits
# schedule[N-1] seconds before the next attempt). 2/4/8/16/20/20 minutes,
# ~70 min total span for the default budget of 6, outlasting every observed
# gate occupancy with margin. One entry per REVIEWER_INFRA_RETRY_BUDGET_DEFAULT
# attempt; _reviewer_infra_backoff_seconds clamps beyond the table's length.
REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC = [120, 240, 480, 960, 1200, 1200]

# Hardcoded allow-list of infrastructure-unavailability reasons, owned by
# lapis-pm and scoped to the vocabulary the reviewer's actual call path
# (shaped_runner._run_local_reviewer -> gw_agent.call_gw_agent) can produce:
# gw_unreachable (gw_agent.py), gw_not_serving (gw_agent.py), and
# gw_defer_timeout (gw_agent.py) — the doorman refused a `deferrable` lease
# acquire and the client's jittered-backoff retry budget
# (GW_DEFER_RETRY_BUDGET_SEC) ran out before admission. The reviewer never
# ran; it was refused GPU admission and timed out waiting. Deliberately NOT
# importing agents_core.llm.GW_PROVENANCE_PRECEDENCE — that tuple
# (gw_deferred_swarm, slot_queued_timeout, slot_pool_down, gw_not_serving)
# belongs to a different call path (the llm.py operator path) and most of its
# members can never reach a reviewer dispatch record. An unrecognised reason
# is treated as non-infrastructure and counts against the ceiling — fail
# closed, so the ceiling's bound is never silently defeated by a new or
# unexpected reason string.
REVIEWER_INFRA_FAIL_REASONS = frozenset({
    "gw_not_serving",
    "gw_unreachable",
    "gw_defer_timeout",
    # Emitted by agents-core-reviewer-seat-tool-call-probe-v0 (Unit A): a seat
    # that cannot emit a tool call never rendered a verdict, so this is a
    # non-run, not a failure — see lapis-pm-reviewer-seat-dead-token-infra-
    # classify-v0 (Unit B).
    "seat_no_tool_calls",
})

# Forgejo health gate constants
FORGEJO_CONSECUTIVE_FAILS_KEY = "pm/forgejo_consecutive_fails"
FORGEJO_UNREACHABLE_THRESHOLD = 3  # consecutive failed probes before Pushover

# D6/D7 stall thresholds (attestation-contract-v0 leg 2).
#
# TICK_COVERAGE_STALL_S: a legitimately-covered target advances its cursor
# at least every ~15-20 min (the measured tick_all pass runs 4-6 min; a
# target is ticked every pass). 45 min is ~3x the worst-case legitimate
# interval, so it is a tripwire for a genuinely broken tick loop (unit dead,
# a hung pass bounded by TimeoutStartSec, or a persistent exception), not
# for the normal cadence.
TICK_COVERAGE_STALL_S = 2700
# DIRECTIVE_OUTCOME_STALL_S: a directive seen but with no dispatch / PR-head
# advance / brief action within 15 min is an unattested outcome (the 09-07
# finding: a directive seen but never dispatched, with no signal).
DIRECTIVE_OUTCOME_STALL_S = 900

# Tick-local corroboration cache: (target_id, pr_number) -> corr_result dict.
# Populated by _encode_gpu_results; consumed by _persist_review_state_cache in
# the same tick to avoid re-scanning episodic for data already in hand.
_tick_corr_cache: dict[tuple[str, int], dict] = {}

# ---------------------------------------------------------------------------
# Post-land deploy hook
# ---------------------------------------------------------------------------

_POST_LAND_RESTART: dict[str, tuple[str, ...]] = {
    "agents-core": ("claude-queue-runner.service", "gpu-queue-runner.service"),
    # lapis-pm intentionally absent: every runtime unit is Type=oneshot (timer-fired).
    # Each fire spawns a fresh python3 process that imports from the deployed cwd,
    # so a code-only sync is picked up on the next scheduled fire with no restart.
    # INVARIANT: if a future lapis-pm unit adds Type=simple or Type=notify, it MUST
    # be listed here — code-only syncs will not restart it otherwise.
    # synapse.service is Type=simple (long-running uvicorn, -m synapse.service on :8401) — a pull alone
    # leaves the /serve daemon running STALE code. It must be restarted to pick up pulled code. (The
    # `ground` CLI is `python3 -m synapse.cli` invoked fresh per call and needs no restart, but the daemon
    # does.) System unit in /etc/systemd/system → restarted via `sudo -n systemctl restart` (the system
    # branch of _post_land_deploy_hook), not the --user branch. Restart FAILURE is stderr/journal best-
    # effort like every other unit here (synapse.service has Restart=on-failure as a self-heal backstop);
    # making restart-failure loud is a cross-cutting concern deferred to
    # lapis-pm-deploy-restart-gate-on-advance-v0 — see §2.
    "synapse": ("synapse.service",),
    # inertia-expert.service + lapis-expert.service are both Type=simple, System unit,
    # WorkingDirectory=/srv/git/experts (confirmed live via `systemctl show
    # inertia-expert.service lapis-expert.service -p Type,WorkingDirectory` 2026-07-23) —
    # a pull alone leaves them serving stale code until restarted, same as synapse above.
    # Closes the FOURTH deploy-currency gap (found 2026-07-10, manually fixed; never
    # closed at the map level until now — lapis-pm-deploy-inventory-auto-recovery-v0).
    "experts": ("inertia-expert.service", "lapis-expert.service"),
}

# --user units that import agents_core from /data/agents. Restarted via
# `systemctl --user restart` (no sudo) — the hook runs as user with
# XDG_RUNTIME_DIR=/run/user/1000 set by lapis-pm.service.
#
# §2.3 three-element note (agents-core-data-agents-auto-deploy-v0):
# 1. Architectural fact: the pip editable-install meta-path finder in /data/agents
#    dominates all `import agents_core` regardless of PYTHONPATH; /data/agents is
#    the actual import root for these units.
# 2. Failure consequence: a pull failure on /data/agents means these units run stale
#    code with no signal — hence agents-core is in _POST_LAND_PULL_CRITICAL above.
# 3. Growth obligation: when a third entry is added here, add a runtime audit guard
#    (grep ~/.config/systemd/user/*.service for agents_core importers; log loudly if
#    any is absent from this map). See spec §5 tripwire.
_POST_LAND_RESTART_USER: dict[str, tuple[str, ...]] = {
    "agents-core": ("doorman-server.service", "slot-server.service"),
    # cockpit.service: Type=simple long-running uvicorn (PM/Ops console + vitals rail),
    # binds 203.0.113.10:8409. Restart=on-failure is crash-only; it does not fire on a
    # clean stop, which is the gap this closes (2026-07-05 outage: service inactive ~3h
    # with fully current code underneath it). Restarts through the existing generic
    # immediate-restart path (not the claude-queue-runner in-flight-deferral path —
    # cockpit is not deferred).
    "cockpit": ("cockpit.service",),
    # loupe.service: Type=simple long-running FastAPI read surface (Desk/Sessions/
    # Landscape), same profile as cockpit above — a pull alone leaves it serving stale
    # code until restarted. Spec: navigator-loupe-serve-parity-n0-v0.
    "loupe": ("loupe.service",),
}

# Canonical deploy clone path for facets. Not pip-installed; the spec-review gate
# injects this path on PYTHONPATH for `python3 -m facets.adapter`.
_FACETS_DEPLOY_CLONE = "/srv/git/facets-working"

_POST_LAND_PULL: dict[str, list[str]] = {
    # Deploy clone only — this is the run path after the WorkingDirectory repoint.
    # cwd-precedence means python3 -m lapis_pm.cli imports from here, not the
    # editable install. /srv/lapis/lapis-pm is the dev/PM investigation
    # tree; it is intentionally NOT on the runtime path and pull failures there
    # must not block or alert.
    "lapis-pm":       ["/srv/git/lapis-pm"],
    # Two paths: the working clone (system-unit runners) + the editable-install
    # root /data/agents (--user unit runtime). Both must track origin/main.
    "agents-core":    ["/srv/git/agents-core-working", "/data/agents"],
    # code-reviewer services are Type=oneshot timer-fired; they re-import on each
    # fire, so a pull (no restart) is sufficient.  Pull failure → LOW signal so a
    # stale nightly sweep is attributable without polluting the critical channel.
    # Spec: lapis-pm-deploy-pull-code-reviewer-v0.
    "code-reviewer":  ["/srv/git/code-reviewer-working"],
    # facets adapter is invoked per-spec-review fire via `python3 -m facets.adapter`;
    # no long-running daemon. A pull (no restart) picks up the new code on next fire.
    # Pull failure → LOW signal (advisory-only, per-invocation).
    # NOTE: facets is NOT in _POST_LAND_RESTART/_RESTART_USER — timer re-imports on each
    # fire and picks up the pulled code naturally.
    # LIABILITIES: (1) facets.adapter must remain importable as python3 -m facets.adapter;
    # breaking module-level changes will break spec-review silently until detected.
    # (2) facets is treated as advisory-only, per-invocation; if it becomes authority-tier
    # or long-running, this classification is wrong — it would need _POST_LAND_PULL_CRITICAL
    # + a _POST_LAND_RESTART/_RESTART_USER entry. (3) this unit makes the current single-tree
    # topology correct; it does not solve the facets-prod-deploy-gap double-duty-tree
    # structure, which is a separate tracked concern.
    "facets":         [_FACETS_DEPLOY_CLONE],
    # synapse: central BRIX context-injection substrate (decision/synapse-central-on-brix-2026-05-30).
    # The serving clone IS /srv/git/synapse-working — synapse.service runs WorkingDirectory there and
    # the host pip-installs it editable, so a git pull into this tree updates the live import root.
    # Unlike lapis-pm's split dev/deploy trees, synapse has a SINGLE tree used for both PM-investigation
    # and runtime. Pull failure → stale runtime → see _POST_LAND_PULL_CRITICAL below.
    "synapse":        ["/srv/git/synapse-working"],
    # gardener is pip editable-installed from /srv/git/gardener-working; gardener-night.service
    # (Type=oneshot, 07:00 LA) re-imports from this tree on each fire, so a pull (no restart) picks
    # up new code on the next nightly. Pull failure → LOW signal (a degraded/stale nightly synthesis
    # is attributable, not a broken daemon). Spec: lapis-pm-deploy-pull-gardener-v0.
    # LIABILITY: this is the single double-duty tree (shared dev/PM-investigation + runtime), the same
    # structure flagged for facets - a /pm-pr-review that checks a branch OUT into this tree, or any
    # dirty/detached state, makes the ff-only pull fail (LOW alert) until the tree is restored to main.
    # The durable decouple to a dedicated deploy clone is a deferred follow-on (see §4), NOT this unit.
    "gardener":       ["/srv/git/gardener-working"],
    # conductor: NOT a code-only pull like the entries above. /srv/git/conductor is
    # conductor's DEPLOY CLONE (cf. lapis-pm's own /srv/git/lapis-pm vs -working split,
    # :174-179) — this pull keeps that reference tree current so _deploy_conductor_night_scripts
    # (see night-plan-conductor-deploy-sync-v0) has a fresh source to copy FROM. It does NOT
    # by itself deploy anything: the night timers execute from /data/agents/scripts
    # (agents-core), an unrelated tree this pull never touches — see the copy step below.
    # LIABILITY (inherited, not solved, by this unit — mirrors the gardener double-duty-tree
    # note above): if /srv/git/conductor is left checked out on a feature branch or dirty
    # (e.g. by a /pm-pr-review or worker checkout against this path instead of a scratch
    # clone), the ff-only pull fails (LOW alert, attributable) until restored to main. The
    # source-freshness gate in _deploy_conductor_night_scripts additionally refuses to copy
    # from a non-main/dirty source, so this liability cannot corrupt the runtime — it only
    # blocks delivery until remediated. See spec §Go-live step 1.
    "conductor":      ["/srv/git/conductor"],
    # cockpit is a single-tree PYTHONPATH-import service (like synapse), not a split
    # dev/deploy pair. cockpit.service is Type=simple, long-running --user unit; a pull
    # alone leaves it serving stale code until restarted (see _POST_LAND_RESTART_USER).
    # Pull failure -> LOW signal (advisory live console, not a silent load-bearing
    # substrate like synapse/agents-core) — see _POST_LAND_PULL_LOW_SIGNAL below.
    "cockpit":        ["/srv/git/cockpit-working"],
    # rag-ops's live deploy clone is /data/rag, a SEPARATE checkout from the PM working tree
    # /srv/git/rag-ops-working (see rag-ops-vault-rag-drift-reconcile-v0's Substrate for why these
    # two trees diverge). This pull keeps /data/rag's git ref current with origin/main so a future
    # /pm-pr-review's close-out doesn't have to hand-reconcile it (as happened for PR #5 and #6,
    # 2026-07-14) — it does NOT make docker re-read the pulled docker-compose.yml. docker compose
    # only applies compose-file changes at container create/recreate time (`docker compose up -d
    # [--force-recreate]`), never on a bare git pull against a running container. Actually deploying
    # a future rag-ops config change to the live services remains a manual post-merge step (matching
    # every rag-ops PR's own existing "Deploy note" convention) — this hook only prevents the git
    # tree itself from silently drifting behind what's merged. Spec: lapis-pm-deploy-pull-rag-ops-v0.
    "rag-ops":        ["/data/rag"],
    # experts: self-host Zephyr Expert production service (Dustin-facing, :8412),
    # WorkingDirectory=/srv/git/experts. Same signal tier as cockpit/synapse's restart
    # handling — see _POST_LAND_RESTART's experts entry for why a pull alone is
    # insufficient. Intentionally NOT in _POST_LAND_PULL_CRITICAL or
    # _POST_LAND_PULL_LOW_SIGNAL (spec: lapis-pm-deploy-inventory-auto-recovery-v0 Part A
    # item 3 — nothing in that spec depends on the critical/low-signal distinction here).
    "experts":        ["/srv/git/experts"],
    # loupe: single-tree PYTHONPATH-import service (like cockpit/synapse), not a split
    # dev/deploy pair for the read path itself — but the deploy CLONE is intentionally
    # separate from /srv/git/loupe-working (the PM investigation tree), mirroring the
    # lapis-pm/lapis-pm-working split above (:223), so that editing -working never
    # changes what's live. loupe.service is Type=simple, long-running --user unit; a
    # pull alone leaves it serving stale code until restarted (see
    # _POST_LAND_RESTART_USER). Pull failure -> LOW signal (advisory read-only console
    # for Erah, same tier as cockpit — see _POST_LAND_PULL_LOW_SIGNAL below).
    # Spec: navigator-loupe-serve-parity-n0-v0.
    "loupe":          ["/srv/git/loupe"],
}

# lapis-pm: failed pull → next tick runs stale code.
# agents-core: failed pull on /data/agents → --user units (doorman, slot) silently
# run stale code with no signal, which is the exact gap this unit closes.
# synapse: failed pull on /srv/git/synapse-working → the central context-injection
# substrate (decision/synapse-central-on-brix-2026-05-30) silently serves stale code to
# every node's UserPromptSubmit hook. Same silent-load-bearing-staleness profile → CRITICAL.
_POST_LAND_PULL_CRITICAL: frozenset[str] = frozenset({"lapis-pm", "agents-core", "synapse"})

# Repos whose pull failure emits a LOW-priority notification (not critical, not silent).
# code-reviewer: timer-oneshot services — stale code is a degraded nightly sweep, not
# a broken daemon.  LOW keeps the failure attributable without paging.
# facets: advisory-only, per-invocation gate (Haiku personas, spec-review). Stale
# code → degraded next spec-review fire. LOW keeps it attributable.
# gardener: timer-oneshot service (07:00 LA nightly synthesis) — stale code is a
# degraded synthesis, not a broken daemon. LOW keeps it attributable.
# conductor: night scripts are timer-oneshot (re-import on each fire); a stale/failed
# pull-then-copy is a degraded night, attributable via the U2a gate + success-predicates,
# not a broken daemon. NOTE (R10): conductor's actual alerting is owned by the
# source-freshness gate in _deploy_conductor_night_scripts, which alerts on every
# failure mode (transient fetch-fail, dirty, off-main) — the generic LOW pull-fail
# Pushover below is suppressed specifically for conductor to avoid a double-alert for
# one root cause. conductor stays in this set for classification purposes only.
# cockpit: advisory live/action console for Erah, not a silent substrate every session
# depends on (contrast synapse/agents-core, which are CRITICAL). A stale or dead
# cockpit is visible the moment Erah opens the UI — LOW keeps a pull failure
# attributable without polluting the critical Pushover channel.
# rag-ops: pure docker compose config + a bash script, no daemon imports it. A stale
# /data/rag git ref is attributable drift (§0 of lapis-pm-deploy-pull-rag-ops-v0), not a
# broken runtime — docker compose doesn't even re-read the pulled files until a manual
# recreate, so LOW keeps this failure attributable without polluting the critical channel.
# loupe: advisory read-only console for Erah, visible the moment he opens it — same
# tier reasoning as cockpit above. A stale pull is attributable, not a silent
# load-bearing-substrate failure. Spec: navigator-loupe-serve-parity-n0-v0.
_POST_LAND_PULL_LOW_SIGNAL: frozenset[str] = frozenset(
    {"code-reviewer", "facets", "gardener", "conductor", "cockpit", "rag-ops", "loupe"}
)

# ---------------------------------------------------------------------------
# conductor night-plan script closure (night-plan-conductor-deploy-sync-v0)
# ---------------------------------------------------------------------------
# Deploy clone conductor pulls into (see _POST_LAND_PULL["conductor"] above) and the
# root the R7 source-freshness gate fetches/rev-parses against. Module-level constant
# (not inlined) so tests can patch it to a temp git repo — see _CONDUCTOR_SCRIPTS_SRC.
_CONDUCTOR_DEPLOY_CLONE = "/srv/git/conductor"

# Copy source/dest. Module-level constants (not inlined literals) so the real-filesystem
# DoD tests can patch.object(pm_core, "_CONDUCTOR_SCRIPTS_DEST", <tmp_path>) exactly as the
# existing suite patches _DEPLOY_LOG (:235) — a fixer must never inline the production path
# in the helper body, or the tests would have to write to the live runtime to exercise it.
_CONDUCTOR_SCRIPTS_SRC = "/srv/git/conductor/scripts"
_CONDUCTOR_SCRIPTS_DEST = "/data/agents/scripts"

# The transitive scripts/-local import closure reachable from night_plan.py + every
# producer declared in night_producers.yaml (seed = {night_plan, night_coordinator, all
# yaml `module:` values}; edge = any `from X`/`import X` where scripts/X.py exists;
# fixpoint). Computed 2026-07-04 against conductor origin/main = bede656 — see spec
# night-plan-conductor-deploy-sync-v0 §"night_plan.py's runtime closure" for the full
# derivation and the deeper deps (gpu_lane, rsi_ingest, podcast_engine) a naive depth-2
# read misses.
#
# GROWTH OBLIGATION (mirrors the _POST_LAND_RESTART §2.3 note at :156-164): when a
# producer is added to (or a new scripts/-local import added by) conductor's
# night_producers.yaml, its module + every new sibling dep MUST be added here. An
# unmanifested file is a silent runtime gap the next conductor merge will not fill.
_CONDUCTOR_NIGHT_SCRIPTS: tuple[str, ...] = (
    "night_plan.py",
    "gw_phase_models.py",  # night_plan.py's own top-level import since 2026-08-01
                            # (c35e8f2) — GW phase->served-model map, a true leaf
                            # (lapis-pm-conductor-brix-gw-runtime-deploy-manifest-v0)
    "night_coordinator.py",
    "gpu_lane.py",
    "scout_producer.py",
    "arxiv_producer.py",
    "arxiv_watch.py",
    "rsi_ingest.py",
    "idea_collider_night_batch_producer.py",
    "idea_collider.py",
    "idea_collider_night_batch.py",
    "podcast_engine.py",
    "kami_producer.py",
    "kami_batch.py",
    "kami_selector.py",
    "kami_sweep.py",
    "kami_adjudicator.py",
    "kami_small.py",
    "enlightenment_producer.py",
    "enlightenment_reader.py",
    "research_headings.py",
    "night_producers.yaml",
    "night_task_menu.py", "night-task-menu.yaml", "night_plan_manager.py",   # Rung B menu (leftover) + Rung C.0 manager
)


# ---------------------------------------------------------------------------
# GravityWell host-script closure (host-script-deploy-on-land-v0, D1)
# ---------------------------------------------------------------------------
# Cross-host sibling of _CONDUCTOR_NIGHT_SCRIPTS above: these are conductor
# repo-held scripts whose runtime home is gravitywell:/usr/local/sbin/, not
# /data/agents/scripts. Same source of truth (_CONDUCTOR_SCRIPTS_SRC, the
# deploy clone — never -working) and the same shared freshness gate
# (_conductor_source_freshness_gate, D2), but a different destination
# (remote host, not a local directory) and a different delivery mechanism
# (atomic scp+install, D6, not a local temp-then-os.replace).
class _GwHostScript(NamedTuple):
    """src_name is repo-relative to _CONDUCTOR_SCRIPTS_SRC (scripts/<src_name>).
    self_attesting gates the extra `--help` self-hash verify (D6) — True iff
    the script prints a `self-hash:` line, the same mechanism
    gw_topology.py:_assert_helper_compatible checks. execution_sensitive
    gates the in_flight_flip skip (D5) — carried explicitly per entry, not
    inferred from in_flight_flip's own scope (which describes GravityWell's
    convergence state, the wrong granularity for a manifest that will grow)."""
    src_name: str
    dest: str
    mode: int
    owner: str
    self_attesting: bool
    execution_sensitive: bool


# GROWTH OBLIGATION (mirrors the _CONDUCTOR_NIGHT_SCRIPTS growth-obligation
# comment above): any new conductor script whose runtime home is
# gravitywell:/usr/local/sbin/ (or, per-entry, another remote path — the dest
# is arbitrary per entry) MUST be added here, or it gets no delivery (D1)
# and no drift report (D7 only covers manifested files).
# gw-dual (/usr/local/sbin) is deliberately NOT here — it has no repo source,
# so a manifest entry naming it would be a no-op at best (spec Out-of-scope).
# gw-serve (/usr/local/bin) gained a repo source (scripts/gw-serve,
# bootstrapped + GPU 1 stop-path extension) via gw-gpu1-berth-standing-seat-v0
# leg 1 (conductor PR #896), so it is manifested here: execution_sensitive=True
# because it flips serving state (the in-flight-flip skip, D5, applies).
# gw-gpu1-lease.sh is the tenancy fast-follow debt (gpu1-multi-tenant-lease
# arbitration landing 2026-08-22/23): it was hand-installed on GW and the
# manifest entry never landed; this entry makes subsequent conductor deploys
# deliver it durably.
_CONDUCTOR_GW_HOST_SCRIPTS: tuple[_GwHostScript, ...] = (
    _GwHostScript("gw-topology", "/usr/local/sbin/gw-topology", 0o755, "root:root", True, True),
    _GwHostScript("gw-idle-suspend.sh", "/usr/local/sbin/gw-idle-suspend.sh", 0o755, "root:root", False, True),
    _GwHostScript("gw_resume_grace.py", "/usr/local/sbin/gw_resume_grace.py", 0o755, "root:root", False, False),
    _GwHostScript("gw-gpu1-lease.sh", "/usr/local/sbin/gw-gpu1-lease.sh", 0o755, "root:root", False, False),
    _GwHostScript("gw-serve", "/usr/local/bin/gw-serve", 0o755, "root:root", False, True),
)


# ---------------------------------------------------------------------------
# BRIX-side GW actuator runtime closure
# (lapis-pm-conductor-brix-gw-runtime-deploy-manifest-v0)
# ---------------------------------------------------------------------------
# Local sibling of _CONDUCTOR_NIGHT_SCRIPTS above: same source of truth
# (_CONDUCTOR_SCRIPTS_SRC), same destination (_CONDUCTOR_SCRIPTS_DEST —
# /data/agents/scripts), same local-copy mechanics (_copy_one_conductor_script),
# and copied in the SAME pass as _CONDUCTOR_NIGHT_SCRIPTS by
# _deploy_conductor_night_scripts_impl (one shared D2/D3 freshness-gate call, so a
# fetch failure pages once — R10 — not twice). A SEPARATE manifest, not an append
# to _CONDUCTOR_NIGHT_SCRIPTS itself, because this family's consumers (doorman's
# _wake_generic_posture, flip-controller.service, gw_topology.py's own
# _local_helper_self_hash reference copy, scout-night.service) sit outside
# night_plan.py's producer-import closure — the two exact-equality closure tests
# above (test_fixture_closure_matches_manifest / test_live_conductor_closure_
# matches_fixture) would fail on a naive append.
#
# A plain filename tuple (like _CONDUCTOR_NIGHT_SCRIPTS), not a _GwHostScript
# NamedTuple: every entry shares one destination directory, so the per-entry
# dest/mode/owner fields _CONDUCTOR_GW_HOST_SCRIPTS needs (different destination,
# a remote host) are not needed here.
#
# `gw-topology` (no .py — the bash helper) is delivered here as the hash-reference
# copy _local_helper_self_hash reads on every reach() (gw_topology.py:1838-1842);
# it is NEVER executed on BRIX, only read for its hash.
#
# GROWTH OBLIGATION (mirrors the _CONDUCTOR_NIGHT_SCRIPTS growth-obligation
# comment above): any new conductor scripts/-local file the doorman/
# flip-controller/gw_topology runtime imports on BRIX MUST be added here, or it
# is a silent hand-copy gap — the defect this manifest closes.
#
# Deliberately excluded (spec Out-of-scope): gw_topology_cache.py (its systemd
# unit is not installed — inert), gw_enforce_rearm_ab.py (experiment artifact,
# no service consumer), gw-serve/gw-dual (no repo source).
_CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS: tuple[str, ...] = (
    "gw_topology.py",
    "gw_actuator.py",
    "gw_host_safety.py",
    "flip_controller.py",
    "gw-topology",
    "gw-night-pre.py",
    "gw-night-post.py",
    "scout-night-pre.py",
    # night-deploy-manifest-attestation-v0 (DoD-1/DoD-2): the three scripts the
    # night DAG actually executes that were NOT in any deploy manifest (the
    # measured 2026-09-07 hole — mini_1f916_night.py 2 days stale,
    # keeper_v0.py ~6 weeks stale, council_sweep.py absent from
    # /data/agents/scripts). Their consumers are the plan file's DAG commands
    # (absolute-path invocation), which sit outside night_plan.py's
    # producer-import closure — so the BRIX_GW_RUNTIME family, NOT
    # _CONDUCTOR_NIGHT_SCRIPTS (whose exact-equality closure tests would fail
    # on a naive append). The checker (lapis_pm/attestation.py) audits this
    # tuple; it is itself invoked as a module from the deploy clone and needs
    # no entry here.
    "mini_1f916_night.py",
    "keeper_v0.py",
    "council_sweep.py",
)

# D4: host-reachability policy — never wake, always retry on the next land.
# Short-timeout ssh probe pattern (mirrors conductor's gpu_lane.py restart_box_service:
# `ssh -o BatchMode=yes <host> ...`), never DoormanClient/WoL — a probe that can wake
# the host would violate D4 by construction.
_GW_SSH_HOST = os.environ.get("LAPIS_PM_GW_SSH_HOST", "gravitywell")
_GW_SSH_CONNECT_TIMEOUT_SECS = int(os.environ.get("LAPIS_PM_GW_SSH_CONNECT_TIMEOUT_SECS", "5"))
_GW_SSH_CMD_TIMEOUT_SECS = int(os.environ.get("LAPIS_PM_GW_SSH_CMD_TIMEOUT_SECS", "20"))
_GW_SCP_TIMEOUT_SECS = int(os.environ.get("LAPIS_PM_GW_SCP_TIMEOUT_SECS", "30"))

# Same self-hash line shape gw_topology.py's _assert_helper_compatible checks
# (scripts/gw-topology:94 — `echo "self-hash: $(sha256sum "$0" | cut -c1-12)"`).
_GW_SELF_HASH_RE = re.compile(r"^self-hash:\s*([0-9a-f]+)\s*$", re.MULTILINE)

# D7: drift ledger — dedups the Desk-gem deposit so a persisting undelivered
# script pages the Desk once, not on every conductor land (DoD 9). Module-level
# constant so tests can patch it to a tmp path exactly as the existing suite
# patches _DEPLOY_LOG (:235).
_GW_HOST_SCRIPT_DRIFT_LEDGER = room_path('lapis_state') / "gw-host-script-drift-ledger.json"
_GW_HOST_SCRIPT_DRIFT_SIGNATURE = "gw-host-script-drift"

_GW_GEM_OPTIONS = [
    {"key": "ack_watch", "title": "Known - keep watching",
     "sub": "stay quiet unless it recurs after a fix", "primary": True},
    {"key": "not_real", "title": "Not a real issue",
     "sub": "suppress this finding-class"},
]

_DEPLOY_LOG = room_path('lapis_state.deploy_log')
_DEPLOY_CURRENCY_STALE_KEY = "pm/deploy-currency-last-alert"
_DEPLOY_CURRENCY_COOLDOWN_SECS = 3600  # alert at most once per hour
# Quantitative commit-distance meter, written on every currency check (current
# or stale) — §D of agents-core-deploy-drift-backstop-v0. A visible-surface
# consumer (e.g. doorman's `/status`) reads this to render `stale: N`; wiring
# that read side up lives in agents-core's own repo, out of this repo's scope.
_DEPLOY_CURRENCY_STATUS_FILE = room_path('lapis_state') / "deploy-currency-status.json"

# Deploy-inventory reconciler (lapis-pm-deploy-inventory-reconciler-v0): derives
# the real deployment surface (systemd units + editable installs) and reconciles
# it against the maps above. Systemctl/unit enumeration is not free — cooldown
# gated like _DEPLOY_CURRENCY_COOLDOWN_SECS, not run every tick (spec item 7).
_DEPLOY_INVENTORY_COOLDOWN_SECS = 3600  # at most one full pass per hour
_DEPLOY_INVENTORY_LAST_RUN_KEY = "pm/deploy-inventory/last-run"

# Script-tier auto-recovery (lapis-pm-deploy-inventory-auto-recovery-v0): a mapped,
# stale-only, unlocked clone is pulled + restarted automatically via the existing
# _post_land_deploy_hook primitive rather than just alerted. _issue_system_restart is
# fire-and-forget and systemd restarts are asynchronous, so post-recovery verification
# polls `systemctl is-active` rather than trusting the restart call's return — checking
# immediately after issuing the restart would race the transition and false-positive as
# failed almost every time.
_AUTO_RECOVERY_RESTART_POLL_INTERVAL_SECS = 1.0
_AUTO_RECOVERY_RESTART_POLL_WINDOW_SECS = 10.0

# ---------------------------------------------------------------------------
# Robust pull: graded collision-safety model (agents-core-deploy-drift-backstop-v0)
# ---------------------------------------------------------------------------
# Zones that carry legitimate runtime state in a deploy clone (queue DBs, config,
# corpus mirrors, dashboard state, opencode context). An untracked file colliding
# with an incoming tracked upstream file here is NEVER silently moved — see
# _classify_untracked_collision. Filename-whitelisting alone is unsafe (a collision
# on a whitelisted name could mask a real conflict), hence zone + pattern together.
_DEPLOY_SACRED_ZONES: tuple[str, ...] = (
    "config/", "data/", "corpus-mirrors/", "dashboard/", ".opencode/",
)
# Basename globs that mark genuine runtime state (queue DBs, large data blobs) as
# opposed to stray debris (`0`, `0.7`, `=`) that happens to sit in a sacred zone.
_DEPLOY_RUNTIME_PATTERNS: tuple[str, ...] = ("*.db", "claude_queue.db", "big")

# Lock sentinel dir: written when a clone hits genuine lineage divergence (not an
# untracked-file collision) after quarantine. Its presence stops subsequent
# backstop ticks from re-attempting the pull (no loop, no `reset --hard`) until a
# human resolves the divergence and clears the lock.
_DEPLOY_PULL_LOCK_DIR = room_path('lapis_state') / "deploy-pull-lock"

# Checked once at module load so tests can patch the env before import.
_DEPLOY_HOOK_DISABLED = os.environ.get("LAPIS_PM_DEPLOY_HOOK_DISABLE") == "1"

# Bounded deferral for in-flight fixers before restarting claude-queue-runner.
# Tunable without a code change via LAPIS_RESTART_DEFER_MAX_S env var.
RESTART_DEFER_MAX_S: int = int(os.environ.get("LAPIS_RESTART_DEFER_MAX_S", "1800"))
_CLAUDE_QUEUE_ACTIVE_DIR = room_path('claude_queue.active')
_RESTART_PENDING_DIR = room_path('lapis_state.restart_pending')


def _ensure_head_branch_deleted(repo: str, pr_number: int, *, owner: str | None = None) -> None:
    """Ensure the head branch of a PR is deleted, idempotently.

    Best-effort. If Forgejo honors delete_branch_after_merge, the branch is
    already gone and get_branch returns 404 (normal path). If the flag is not
    honored or stale state persists, an explicit DELETE is issued. Either way,
    the branch ends deleted. If both probes fail, log and continue — do not
    raise. This guard is a safety backstop, not a critical operation.

    One exception: a branch still held by an open PR is NEVER deleted
    (2026-08-31 incident - every fixer for a target pushes to the SAME
    lapis/<target_id>/local branch, so deleting a merged PR's surviving
    branch auto-closed the in-flight PR on that branch and destroyed the
    work). The explicit DELETE is skipped with a warning naming the holding
    PR number(s). The open-PR probe is itself best-effort: a probe failure
    logs a warning and falls through to the DELETE, preserving today's
    behavior on a partial Forgejo outage.
    """
    if not _forgejo_get_pr or not _forgejo_get_branch:
        return  # forgejo not available
    try:
        pr = _forgejo_get_pr(repo, pr_number, owner=owner)
        ref = pr.get("head", {}).get("ref") if pr else None
        if not ref:
            return  # can't determine branch
        try:
            _forgejo_get_branch(repo, ref, owner=owner)
        except Exception as e:
            # 404 or other error — if 404, branch is already deleted (normal).
            # For other errors, log and continue (best-effort).
            if "404" in str(e) or "not found" in str(e).lower():
                return  # already deleted
            logger.warning("_ensure_head_branch_deleted: get_branch failed: %s", e)
            # fall through to explicit DELETE attempt
        # branch still exists — issue explicit DELETE
        # Open-PR clobber guard (2026-08-31 incident): the branch may still
        # hold an in-flight PR - the daemon reuses lapis/<target_id>/local
        # across every fixer dispatch for a target. Deleting it would make
        # Forgejo auto-close that PR. Refuse the delete while any open PR's
        # head ref matches. A probe failure is non-fatal: log and fall
        # through to the DELETE, exactly as today on a partial Forgejo
        # outage. (Protects both call sites: the post-merge merge_and_deploy
        # step and the per-tick _reconcile_surviving_head_branch.)
        if get_open_prs is not None:
            holding_prs = None
            try:
                holding_prs = [
                    pr_dict.get("number")
                    for pr_dict in get_open_prs(repo, owner=owner)
                    if (pr_dict.get("head") or {}).get("ref") == ref
                ]
            except Exception as e:
                logger.warning(
                    "_ensure_head_branch_deleted: open-PR probe failed (non-fatal): %s",
                    e,
                )
            if holding_prs:
                logger.warning(
                    "_ensure_head_branch_deleted: NOT deleting branch %s - open PR(s) %s still hold it",
                    ref,
                    holding_prs,
                )
                return
        try:
            import httpx
            default_owner = owner or (repo.split("/", 1)[0] if "/" in repo else "Erah")
            repo_name = repo.split("/", 1)[-1] if "/" in repo else repo
            forgejo_base = node_identity.resolve_node_identity().owned_forgejo
            url = f"{forgejo_base}/api/v1/repos/{default_owner}/{repo_name}/branches/{ref}"
            headers = {}
            token = os.environ.get("FORGEJO_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
            resp = httpx.delete(url, headers=headers, timeout=10, verify=False)
            if resp.status_code not in (204, 404):
                logger.warning(
                    "_ensure_head_branch_deleted: DELETE failed with status %d",
                    resp.status_code,
                )
        except Exception as e:
            logger.warning("_ensure_head_branch_deleted: DELETE attempt failed: %s", e)
    except Exception as e:
        logger.warning("_ensure_head_branch_deleted: outer exception (non-fatal): %s", e)


def merge_and_deploy(repo: str, pr_number: int, *, owner: str | None = None) -> dict:
    """Merge a PR and immediately bring its deploy clone(s) to origin/main.

    Single chokepoint for every merge path. Deploy currency must NOT depend on
    the daemon auto-land gate (merged + branch-deleted + no-pending + 60s race),
    which has silently left agents-core --user units on stale code.

    The hook is best-effort and never raises: a failed ff-only pull already
    alerts loudly (Pushover) inside _post_land_deploy_hook, and a merge that
    succeeded must never be reported as failed because the follow-on pull hit a
    dirty/diverged clone. _act_auto_land keeps its own hook call as an
    idempotent backstop — firing twice is a no-op (ff-only pull of an
    already-current clone changes nothing).
    """
    from agents_core.forgejo import FORGEJO_URL as _target_forgejo_url
    node_identity.ensure_owned_forgejo(_target_forgejo_url)
    result = merge_pr(repo, pr_number, owner=owner)  # raises on real merge failure
    try:
        _post_land_deploy_hook(repo, trigger="post-merge-hook")
        _ensure_head_branch_deleted(repo, pr_number, owner=owner)
    except Exception as e:
        logger.warning("merge_and_deploy post-merge step failed (non-fatal): %s", e)
    return result


def _write_deploy_log(tree: str, old_sha: str, new_sha: str, trigger: str) -> None:
    """Append one dated provenance line to the deploy log. Best-effort.

    L2.D4 (local-reviewer-identity-and-provenance-v0): the label is
    ``<seat-alias>:<served-model>`` — the seat alias resolves statically to
    the operator default (gravitywell; the brief composer passes no model
    kwarg) and the served model is the seam-filtered locality-ledger
    fallback, else the explicit not-reported marker. The dead _BRIEF_MODEL
    import dies here; the old brief.py:unknown label shape is dead.
    """
    try:
        from . import provenance as _provenance
        label = _provenance.deploy_log_label()
    except Exception:
        # Fail-soft: the label must never raise up the deploy path.
        label = "gravitywell:not-reported"
    ts = _now_iso()
    line = f"- `{ts}` | {tree} | synced {old_sha}..{new_sha} | brief.py:{label} | {trigger}\n"
    try:
        with open(_DEPLOY_LOG, "a") as f:
            f.write(line)
    except OSError as e:
        print(f"[post-land-pull] deploy log write failed: {e}", file=sys.stderr)


_DEPLOY_LOG_LINE_RE = re.compile(
    r"^- `[^`]+` \| (?P<tree>.+?) \| synced (?P<old>[0-9a-f]+)\.\.(?P<new>[0-9a-f]+) \| "
    r".*? \| (?P<trigger>.+)$"
)


def _last_deploy_log_shas(tree: str, trigger: str) -> tuple[str | None, str | None]:
    """Read back the most recent `_write_deploy_log` line for `tree`+`trigger`.
    Best-effort — a missing/unparseable log is a clean `(None, None)`."""
    try:
        lines = Path(_DEPLOY_LOG).read_text().splitlines()
    except OSError:
        return None, None
    for line in reversed(lines):
        m = _DEPLOY_LOG_LINE_RE.match(line)
        if m and m.group("tree") == tree and m.group("trigger") == trigger:
            return m.group("old"), m.group("new")
    return None, None


def _deploy_pull_lock_path(clone_path: str) -> Path:
    name = clone_path.strip("/").replace("/", "_") + ".json"
    return _DEPLOY_PULL_LOCK_DIR / name


def _deploy_pull_locked(clone_path: str) -> dict | None:
    """Return the lock record if `clone_path` is locked from a prior genuine
    divergence, else None. Best-effort — unreadable/missing lock reads as unlocked.
    """
    try:
        return json.loads(_deploy_pull_lock_path(clone_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_deploy_pull_lock(clone_path: str, reason: str, commits_behind: int | None) -> None:
    """Persist the divergence lock so subsequent backstop ticks skip this clone
    instead of re-attempting the ff-pull dance. Best-effort.
    """
    try:
        _DEPLOY_PULL_LOCK_DIR.mkdir(parents=True, exist_ok=True)
        _deploy_pull_lock_path(clone_path).write_text(json.dumps({
            "clone": clone_path,
            "reason": reason,
            "commits_behind": commits_behind,
            "locked_at": _now_iso(),
        }))
    except OSError as e:
        print(
            f"[post-land-pull] failed to write deploy-pull lock for {clone_path}: {e}",
            file=sys.stderr,
        )


def _clear_deploy_pull_lock(clone_path: str) -> None:
    """Remove the divergence lock — the manual-recovery escape hatch. Best-effort."""
    try:
        _deploy_pull_lock_path(clone_path).unlink(missing_ok=True)
    except OSError:
        pass


def _commit_distance(clone_path: str, ref_range: str = "HEAD..origin/main") -> int | None:
    """Count of commits in `ref_range` at `clone_path`. None if indeterminate."""
    try:
        result = subprocess.run(
            ["git", "-C", clone_path, "rev-list", "--count", ref_range],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return None


def _in_sacred_zone(relpath: str) -> bool:
    return any(relpath.startswith(zone) for zone in _DEPLOY_SACRED_ZONES)


def _matches_runtime_pattern(relpath: str) -> bool:
    basename = relpath.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(basename, pat) for pat in _DEPLOY_RUNTIME_PATTERNS)


def _classify_untracked_collision(relpath: str) -> str:
    """Classify one untracked-vs-incoming-tracked collision path.

    Returns "sacred_runtime" (halt, never move — a genuine runtime-state
    conflict), "sacred_junk" (inside a sacred zone but not a runtime pattern —
    quarantine), or "debris" (outside every sacred zone — quarantine).
    """
    if _in_sacred_zone(relpath):
        return "sacred_runtime" if _matches_runtime_pattern(relpath) else "sacred_junk"
    return "debris"


def _parse_untracked_collision_files(stderr: str) -> list[str]:
    """Extract the file list from git's "would be overwritten by merge/checkout"
    abort message. Returns [] if the message isn't present.
    """
    if "would be overwritten by" not in stderr:
        return []
    files: list[str] = []
    collecting = False
    for line in stderr.splitlines():
        if "would be overwritten by" in line:
            collecting = True
            continue
        if not collecting:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith(("Please", "Aborting", "error:", "hint:", "fatal:")):
            if files:
                break
            continue
        files.append(stripped)
    return files


def _quarantine_untracked_file(clone_path: str, relpath: str, batch_ts: str) -> bool:
    """Move a colliding untracked file to a timestamped quarantine dir inside the
    clone, clearing the way for the incoming tracked file. Never called for
    sacred_runtime collisions — those halt instead (see _post_land_git_pull).
    """
    src = Path(clone_path) / relpath
    dest = Path(clone_path) / ".deploy-quarantine" / batch_ts / relpath
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        print(
            f"[post-land-pull] quarantined debris {relpath} at {clone_path} -> {dest}",
            file=sys.stderr,
        )
        return True
    except OSError as e:
        print(
            f"[post-land-pull] quarantine failed for {relpath} at {clone_path}: {e}",
            file=sys.stderr,
        )
        return False


def _post_land_git_pull(repo: str | None, trigger: str = "post-land-hook") -> bool:
    """Git-pull each working clone mapped to `repo`.

    The tick runs from the deploy clone (/srv/git/lapis-pm) via cwd-precedence;
    that clone is the run path and its pull is runtime-critical. A failed pull
    there sends a Pushover alert at NORMAL priority so silent drift is impossible.
    A successful pull that advances HEAD appends a dated provenance line to the
    deploy log.

    Returns True if HEAD advanced on at least one mapped path (the caller uses this
    to gate service restarts so a no-op idempotent pull doesn't trigger a needless
    restart). Returns False when HEAD was already current, the pull failed, or repo
    is not mapped.

    Best-effort. Never raises. Uses --ff-only so a diverged clone fails
    loudly rather than silently creating a merge commit.
    """
    if repo is None:
        return False
    paths = _POST_LAND_PULL.get(repo)
    if not paths:
        return False
    is_critical_repo = repo in _POST_LAND_PULL_CRITICAL
    is_low_signal_repo = repo in _POST_LAND_PULL_LOW_SIGNAL
    any_advanced = False
    for path in paths:
        # A locked clone (prior genuine divergence) is skipped entirely — no
        # re-attempted pull, no repeated quarantine dance. A human clears the
        # lock (_clear_deploy_pull_lock) once the divergence is resolved.
        if is_critical_repo and _deploy_pull_locked(path):
            # lapis-pm-daemon-silent-gaps-v0 B4b: a locked clone is re-evaluated,
            # not skipped forever. The D1 classifier's EVIDENCE (fetch_rc /
            # porcelain / ahead) is consulted - never the class label's safe
            # set (the clean-ff-able state is `unknown`, which the module's I1
            # declares "never a safe class" for the module's own self-repair
            # actions; this is a different, explicitly-guarded decision path,
            # and the ff-only pull is the self-verification). Any None/empty
            # evidence field is still-stuck (never clear, never an error).
            # The clear decision does NOT re-implement classification logic.
            locked_reclear = False
            try:
                from . import deploy_pull_selfheal as _dps
                _cls, _ev = _dps.classify_pull_failure(path, trigger)
                _fetch_ok = _ev.get("fetch_rc") == 0
                _clean = not any(
                    line and not line.startswith("??")
                    for line in (_ev.get("porcelain") or [])
                )
                _ahead_zero = _ev.get("ahead") == 0
                if _fetch_ok and _clean and _ahead_zero:
                    # The ONLY clearable state (D1's clean:unknown arm).
                    # Clear the lock + proceed with the existing ff-only pull
                    # below; the ff-only guard self-verifies (a failed pull
                    # re-locks with fresh measurement + the existing HIGH
                    # page). Ledger entry marked `closed` (never deleted) so a
                    # re-lock pages as a fresh state transition.
                    _dps.close_ledger_entries_for_path(path, reason="b4b-self-clear")
                    _clear_deploy_pull_lock(path)
                    locked_reclear = True
                    print(
                        f"[post-land-pull] {path} lock self-cleared (clean-ff-able "
                        f"state: fetch ok, clean tree, ahead==0) — proceeding with "
                        f"ff-only pull",
                        file=sys.stderr,
                    )
                else:
                    # still-stuck: keep the lock, emit exactly one NORMAL page
                    # per (path, class) state transition via a NEW page site in
                    # the locked-clone path, deduped on the D3 ledger
                    # signature (compute_signature = sha256(path|class)); a
                    # stable class stuck for days pages exactly once
                    # (subsequent cycles are silent `bump:open`).
                    _dps.notify_locked_clone_stuck(repo, path, _cls, trigger)
                    print(
                        f"[post-land-pull] {path} is locked (genuine divergence, "
                        f"manual intervention required; re-evaluated class="
                        f"{_cls}) — skipping pull",
                        file=sys.stderr,
                    )
                    continue
            except Exception as _lock_exc:
                # Never an unhandled error: a classification failure keeps the
                # lock and degrades to the original skip (logged).
                print(
                    f"[post-land-pull] {path} locked-clone re-evaluation errored "
                    f"({type(_lock_exc).__name__}: {_lock_exc}) — skipping pull",
                    file=sys.stderr,
                )
                continue
            if not locked_reclear:
                continue

        pre_head = ""
        # deploy-pull-selfheal-core-v0 (Slice 1): whether this path's pull
        # failure was recorded by the machine (gem/ledger, no self-repair).
        # Initialized per-path so every failure branch (incl. the dirty-tree
        # skip that never reaches the selfheal block) has it defined.
        machine_recorded = False
        try:
            pre_result = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if pre_result.returncode == 0:
                pre_head = pre_result.stdout.strip()
        except Exception:
            pass

        def _finish_success(post_pre_head: str) -> None:
            nonlocal any_advanced
            post_result = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            post_head = (
                post_result.stdout.strip() if post_result.returncode == 0 else ""
            )
            if post_head and post_head != post_pre_head:
                any_advanced = True
                _write_deploy_log(path, post_pre_head[:8], post_head[:8], trigger)

        try:
            dirty_check = subprocess.run(
                ["git", "-C", path, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            # Untracked files alone are NOT "dirty" — the graded collision-safety
            # model below handles untracked-vs-incoming-tracked collisions
            # explicitly (quarantine or halt). Only actual tracked-file
            # modifications ("??"-prefixed lines are untracked; everything else
            # is a staged/unstaged change to a tracked file) block the pull.
            dirty = dirty_check.returncode == 0 and any(
                line and not line.startswith("??")
                for line in dirty_check.stdout.splitlines()
            )
            if dirty:
                print(
                    f"[post-land-pull] dirty working tree at {path}, skipping pull "
                    f"(uncommitted local changes present) — resolve manually",
                    file=sys.stderr,
                )
                # falls through to the existing is_critical_repo / is_low_signal_repo notify
                # branches below, exactly as a pull-command failure would — this only changes
                # the LOGGED MESSAGE from git's generic ff-only-failure text to an actionable,
                # specifically-named cause.
                result = subprocess.CompletedProcess(
                    args=["git", "-C", path, "pull", "--ff-only", "origin", "main"],
                    returncode=1,
                    stdout="",
                    stderr="dirty working tree, pull skipped",
                )
            else:
                result = subprocess.run(
                    ["git", "-C", path, "pull", "--ff-only", "origin", "main"],
                    capture_output=True, text=True, timeout=30,
                )
            if result.returncode != 0:
                print(
                    f"[post-land-pull] pull {path} failed rc={result.returncode}: "
                    f"{result.stderr[:200]}",
                    file=sys.stderr,
                )
                # Graded collision-safety model — critical clones only (agents-core-
                # deploy-drift-backstop-v0 §B). Untracked-vs-incoming-tracked
                # collisions are classified by zone + content, not blanket-
                # whitelisted, before either quarantining debris or halting on a
                # genuine runtime-state conflict. `dirty` failures (tracked-file
                # modifications) are a different class and are NOT collision-
                # handled here — they fall straight through to the generic alert.
                if is_critical_repo and not dirty:
                    collision_files = _parse_untracked_collision_files(result.stderr)
                    if collision_files:
                        classified = {
                            f: _classify_untracked_collision(f) for f in collision_files
                        }
                        sacred_runtime = [
                            f for f, cls in classified.items() if cls == "sacred_runtime"
                        ]
                        if sacred_runtime:
                            print(
                                f"[post-land-pull] HALT {path}: sacred runtime-state "
                                f"collision on {sacred_runtime} — pull aborted, file(s) "
                                f"NOT moved, manual resolution required",
                                file=sys.stderr,
                            )
                            try:
                                from agents_core.notify import send_notification, Priority as _P
                                send_notification(
                                    message=(
                                        f"post-land pull for {path} halted: untracked "
                                        f"runtime-state file(s) {sacred_runtime} collide "
                                        f"with incoming upstream tracked file(s). Not "
                                        f"auto-moved (data-loss risk) — needs manual "
                                        f"reconciliation."
                                    ),
                                    title=f"{repo}: deploy pull halted (runtime collision)",
                                    priority=_P.HIGH,
                                )
                            except Exception:
                                pass
                            continue
                        batch_ts = _now_iso().replace(":", "").replace("-", "")
                        quarantined_ok = all(
                            _quarantine_untracked_file(path, f, batch_ts)
                            for f in collision_files
                        )
                        if quarantined_ok:
                            retry = subprocess.run(
                                ["git", "-C", path, "pull", "--ff-only", "origin", "main"],
                                capture_output=True, text=True, timeout=30,
                            )
                            if retry.returncode == 0:
                                _finish_success(pre_head)
                                continue
                            result = retry  # fall through with the retry's failure below
                    # deploy-pull-selfheal-core-v0 (Slice 1): classify the failure and let
                    # the machine act on its lane (safe-zone self-repair, ledger +
                    # gem) BEFORE the legacy divergence-lockout / generic alert.
                    # A genuine divergence classifies `diverged` (NOT a safe-zone
                    # repair - the classifier returns it and the machine records a
                    # gem but does NOT repair it); the legacy lockout below still
                    # handles the actual lock. A self-REPAIR success (head
                    # advanced) is fully owned by the machine and skips the rest;
                    # a RECORDED-only failure (gem/ledger, no head advance - e.g.
                    # a divergence) suppresses the generic notify but still falls
                    # through to the divergence lockout. D4/D5/D6/D7 surfaces are
                    # removed in this slice.
                    selfheal_handled = False
                    if is_critical_repo:
                        try:
                            from . import deploy_pull_selfheal as _dps
                            selfheal_handled = _dps.pass_handled_failure(
                                repo, path, trigger, pull_rc=result.returncode,
                                pull_stderr=result.stderr or "",
                            )
                        except Exception as _dps_exc:
                            print(
                                f"[post-land-pull] selfheal pass errored for {path}: "
                                f"{_dps_exc}",
                                file=sys.stderr,
                            )
                    # Critical fix #4 (rev-2): a self-repair success MUST fire
                    # _finish_success(pre_head) so the restart gate (re-import into
                    # running services) + the canonical deploy-log line run - the
                    # head's block never made this call, so any_advanced stayed
                    # False and the self-repaired head was never re-imported
                    # (the stale-runtime class this slice closes).
                    if selfheal_handled:
                        _finish_success(pre_head)
                    # A self-REPAIR (head advanced) is fully owned - skip the
                    # divergence lockout AND the generic notify.
                    if selfheal_handled and any_advanced:
                        continue
                    # A RECORDED-only failure (gem/ledger, no head advance):
                    # suppress the generic notify below (the machine owns it)
                    # but STILL fall through to the divergence lockout.
                    machine_recorded = selfheal_handled and not any_advanced
                    if "Not possible to fast-forward" in result.stderr:
                        distance = _commit_distance(path)
                        print(
                            f"[post-land-pull] GENUINE DIVERGENCE at {path}: ff-only "
                            f"impossible after quarantine (commits behind={distance}) — "
                            f"locking, manual intervention required",
                            file=sys.stderr,
                        )
                        _write_deploy_pull_lock(path, "genuine_divergence", distance)
                        # lapis-pm-daemon-silent-gaps-v0 B4a: the EXISTING HIGH
                        # page is extended - not supplemented - with the
                        # commits_behind == 0 anomaly flag. Exactly one page
                        # per lock acquisition (this one). The bare
                        # try/except: pass is a named silent delivery path:
                        # send_notification writes the notify audit line either
                        # way (delivered=true/false is the verification
                        # surface - a swallowed send must be visible in the
                        # audit log).
                        _suspect_flag = ""
                        if distance == 0:
                            _suspect_flag = (
                                " suspect: divergence with zero commits behind "
                                "(classification anomaly)"
                            )
                        try:
                            from agents_core.notify import send_notification, Priority as _P
                            send_notification(
                                message=(
                                    f"post-land pull for {path} hit genuine lineage "
                                    f"divergence (not an untracked collision) — "
                                    f"{distance if distance is not None else '?'} commits "
                                    f"behind origin/main.{_suspect_flag} Locked; will "
                                    f"NOT retry or reset --hard. Needs manual "
                                    f"intervention, then clear the lock."
                                ),
                                title=f"{repo}: deploy pull diverged (locked)",
                                priority=_P.HIGH,
                            )
                        except Exception:
                            pass
                        continue
                # The machine recorded this failure (gem/ledger) but did not
                # self-repair it - suppress the generic notify (the machine owns
                # it; the divergence lockout above, if any, already handled the
                # physical lock).
                if is_critical_repo and not machine_recorded:
                    try:
                        from agents_core.notify import send_notification, Priority as _P
                        send_notification(
                            message=(
                                f"post-land pull failed for {path} "
                                f"(rc={result.returncode}): {result.stderr[:300]}"
                            ),
                            title=f"{repo}: deploy pull failed",
                            priority=_P.NORMAL,
                        )
                    except Exception:
                        pass
                # R10 (night-plan-conductor-deploy-sync-v0): conductor's alerting is
                # owned by the source-freshness gate in _deploy_conductor_night_scripts,
                # which fires on every conductor failure mode (transient, dirty,
                # off-main). Suppressing the generic LOW here avoids a double-alert
                # for one root cause — the stderr log above is retained either way.
                elif is_low_signal_repo and repo != "conductor":
                    try:
                        from agents_core.notify import send_notification, Priority as _P
                        send_notification(
                            message=(
                                f"post-land pull failed for {repo} ({path}) "
                                f"— git pull --ff-only rc={result.returncode}: "
                                f"{result.stderr[:300]}"
                            ),
                            title=f"{repo}: deploy pull failed",
                            priority=_P.LOW,
                        )
                    except Exception:
                        pass
            elif pre_head:
                _finish_success(pre_head)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"[post-land-pull] pull {path} errored: {e}", file=sys.stderr)
            if is_critical_repo:
                try:
                    from agents_core.notify import send_notification, Priority as _P
                    send_notification(
                        message=f"post-land pull errored for {path}: {e}",
                        title=f"{repo}: deploy pull failed",
                        priority=_P.NORMAL,
                    )
                except Exception:
                    pass
            elif is_low_signal_repo and repo != "conductor":  # R10 — see note above
                try:
                    from agents_core.notify import send_notification, Priority as _P
                    send_notification(
                        message=(
                            f"post-land pull errored for {repo} ({path}) "
                            f"— git pull --ff-only failed: {e}"
                        ),
                        title=f"{repo}: deploy pull failed",
                        priority=_P.LOW,
                    )
                except Exception:
                    pass
    return any_advanced


def _file_short_hash(path: Path) -> str:
    """8-char sha256 prefix of a file's contents, for copy-provenance logging."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:8]
    except OSError:
        return "unknown"


def _write_conductor_copy_log(fname: str, old_repr: str, new_sha: str, trigger: str) -> None:
    """Append one provenance line for a conductor night-plan script copy. Best-effort."""
    ts = _now_iso()
    change = "created" if old_repr == "absent" else f"{old_repr}..{new_sha}"
    line = f"- `{ts}` | conductor:scripts/{fname} | {change} | {trigger}\n"
    try:
        with open(_DEPLOY_LOG, "a") as f:
            f.write(line)
    except OSError as e:
        print(f"[post-land-deploy:conductor] deploy log write failed: {e}", file=sys.stderr)


def _conductor_copy_would_change(manifest: tuple[str, ...], src_dir: Path, dest_dir: Path) -> bool:
    """True if at least one manifested file's copy would create-or-update dest.

    Used only to pick alert priority when the R7 fetch itself fails (NORMAL if real
    delivery was blocked, LOW if the runtime already matches the local source).
    `manifest` is threaded explicitly (not read off a module global) so this is
    shared, unmodified, between the _CONDUCTOR_NIGHT_SCRIPTS and
    _CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS callers.
    """
    for fname in manifest:
        src = src_dir / fname
        dst = dest_dir / fname
        if not src.exists():
            continue
        if not dst.exists() or not filecmp.cmp(str(src), str(dst), shallow=False):
            return True
    return False


def _copy_one_conductor_script(fname: str, src_dir: Path, dest_dir: Path, trigger: str) -> None:
    """Create-or-update one manifested file atomically. Idempotent no-op if unchanged.

    R9: written via temp-in-dest-dir + os.replace so a concurrent night-timer read never
    observes a half-written file. The temp is unlinked if the write/replace fails, so a
    repeated failure does not accumulate stray .tmp files in the runtime directory.
    """
    src = src_dir / fname
    dst = dest_dir / fname
    if not src.exists():
        # A manifest entry naming a file not present in conductor is a manifest bug
        # (caught by the closure drift test), not a deploy-time failure.
        print(f"[post-land-deploy:conductor] manifest source missing: {src}", file=sys.stderr)
        return
    if dst.exists() and filecmp.cmp(str(src), str(dst), shallow=False):
        return  # idempotent no-op — no log, no restart

    old_repr = _file_short_hash(dst) if dst.exists() else "absent"
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(dest_dir), prefix=f".{fname}.", suffix=".tmp")
        with os.fdopen(fd, "wb") as tmp_f, open(src, "rb") as src_f:
            shutil.copyfileobj(src_f, tmp_f)
        shutil.copystat(str(src), tmp_path)
        os.replace(tmp_path, str(dst))
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    new_sha = _file_short_hash(dst)
    _write_conductor_copy_log(fname, old_repr, new_sha, trigger)


def _deploy_conductor_night_scripts(trigger: str = "post-land-hook") -> None:
    """Copy the night-plan script closure into the /data/agents/scripts runtime.

    lapis-pm-conductor-brix-gw-runtime-deploy-manifest-v0: this call ALSO carries
    the BRIX-side GW actuator family (_CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS) in the
    same combined copy pass, deliberately — not a second independent function —
    so a source fetch failure or dirty-source condition still pages exactly once
    (R10 single-alert-ownership; see TestConductorNightPlanDeploy's
    test_*_via_full_hook_emits_exactly_one_alert tests). The two manifests remain
    separate tuples (_CONDUCTOR_NIGHT_SCRIPTS stays the exact night_plan.py import
    closure the two closure tests assert against); only the copy pass is shared.

    R3/R7 — source-freshness gate: this function does its OWN `git fetch origin main`
    against the conductor deploy clone before copying anything. A fetch failure means
    freshness is unverifiable (transient/network/auth) — copying local refs in that
    state risks silently delivering stale content (the round-4 HIGH this gate closes),
    so the copy is skipped entirely and an alert fires. Only when the fetch succeeds
    AND post-fetch HEAD == origin/main AND the tree is clean does the copy proceed —
    otherwise the last-known-good runtime is preserved (skip-not-corrupt).

    R2/R9 — the copy itself is per-file isolated (one bad file must not skip the rest
    of the closure) and atomic (temp-in-dest-dir + os.replace).

    Never raises: an unexpected error anywhere in this function (including the git
    subprocess calls) is caught, logged, and alerted at LOW priority — a copy failure
    must never fail the merge that triggered it.
    """
    try:
        _deploy_conductor_night_scripts_impl(trigger)
    except Exception as e:
        print(
            f"[post-land-deploy:conductor] unexpected error (non-fatal): {e}",
            file=sys.stderr,
        )
        try:
            from agents_core.notify import send_notification, Priority as _P
            send_notification(
                message=f"conductor night-plan script deploy hit an unexpected error: {e}",
                title="conductor: night-plan deploy error",
                priority=_P.LOW,
            )
        except Exception:
            pass


def _conductor_source_freshness_gate(clone: str) -> tuple[bool, str]:
    """R7 source-freshness gate, extracted (host-script-deploy-on-land-v0 D2) so the
    local night-script copy and the GravityWell host-script install share ONE gate —
    a fetch failure or a not-clean-on-main source skips BOTH deliveries identically,
    never re-implemented twice.

    Returns (is_fresh, reason). reason is "" when fresh; otherwise a short
    machine-parseable string: "fetch_error:<exc>" or "fetch_failed:<stderr>" for a
    transport-level failure (caller decides alert priority by whether a real
    delivery was blocked), or "not clean-on-main (<branch>@<head8>)" for a
    dirty/diverged source (always NORMAL — see callers).

    D3: the tree-level check is `git status --porcelain --untracked-files=no` —
    untracked editor/tooling directories (.claude/, .vscode/) must not block
    delivery; only TRACKED modifications are load-bearing dirt. This is
    deliberately looser than before and is safe only because
    `_conductor_file_verified_clean` re-checks each manifested file individually
    before it is copied — the tree check alone would otherwise let an untracked
    manifested file slip through as though it were reviewed code.

    Never raises — a raised git-subprocess exception here reads as a fetch failure.
    """
    try:
        fetch_result = subprocess.run(
            ["git", "-C", clone, "fetch", "origin", "main"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"fetch_error:{e}"

    if fetch_result.returncode != 0:
        return False, f"fetch_failed:{fetch_result.stderr.strip()[:300]}"

    head_proc = subprocess.run(
        ["git", "-C", clone, "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=5,
    )
    main_proc = subprocess.run(
        ["git", "-C", clone, "rev-parse", "origin/main"],
        capture_output=True, text=True, timeout=5,
    )
    status_proc = subprocess.run(
        ["git", "-C", clone, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, timeout=15,
    )
    branch_proc = subprocess.run(
        ["git", "-C", clone, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, timeout=5,
    )

    head_sha = head_proc.stdout.strip() if head_proc.returncode == 0 else ""
    main_sha = main_proc.stdout.strip() if main_proc.returncode == 0 else ""
    dirty = bool(status_proc.stdout.strip()) if status_proc.returncode == 0 else True
    branch_name = branch_proc.stdout.strip() if branch_proc.returncode == 0 else "unknown"

    if bool(head_sha) and head_sha == main_sha and not dirty:
        return True, ""

    head8 = head_sha[:8] if head_sha else "unknown"
    return False, f"not clean-on-main ({branch_name}@{head8})"


def _conductor_file_verified_clean(clone: str, relpath: str) -> bool:
    """D3 per-file guarantee: `relpath` (repo-relative, e.g. "scripts/gw-topology")
    must be TRACKED and byte-identical to the deploy clone's own HEAD after the
    ff-only pull, before it is copied anywhere. Closes the one hole the looser
    --untracked-files=no tree check opens: an untracked or locally-modified
    manifested file must never be delivered as though it were reviewed code.

    Baseline is HEAD, not origin/main: during a land the local origin/main ref is
    behind by definition until the fetch above updates it, so diffing against it
    would compare reviewed content to a stale reference. HEAD after the ff-only
    pull is the merge commit's tree — the only truthful "this is reviewed code"
    reference available at that moment (spec D3). The tree-level gate above
    already proves HEAD == the fetched origin/main before any copy runs, so this
    is strictly tighter, not looser.

    Never raises — any git-subprocess failure (timeout, OSError) reads as "not
    verified clean", the conservative default.
    """
    try:
        tracked = subprocess.run(
            ["git", "-C", clone, "ls-files", "--error-unmatch", relpath],
            capture_output=True, text=True, timeout=10,
        )
        if tracked.returncode != 0:
            return False
        unmodified = subprocess.run(
            ["git", "-C", clone, "diff", "--quiet", "HEAD", "--", relpath],
            capture_output=True, timeout=10,
        )
        return unmodified.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _deploy_conductor_night_scripts_impl(trigger: str) -> None:
    clone = _CONDUCTOR_DEPLOY_CLONE
    src_dir = Path(_CONDUCTOR_SCRIPTS_SRC)
    dest_dir = Path(_CONDUCTOR_SCRIPTS_DEST)
    # lapis-pm-conductor-brix-gw-runtime-deploy-manifest-v0: one combined pass over
    # both manifests — see the docstring above for why this isn't two functions.
    manifest = _CONDUCTOR_NIGHT_SCRIPTS + _CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS

    is_fresh, reason = _conductor_source_freshness_gate(clone)
    if not is_fresh:
        is_fetch_reason = reason.startswith(("fetch_failed:", "fetch_error:"))
        if is_fetch_reason:
            _, _, err_detail = reason.partition(":")
            priority_low = not _conductor_copy_would_change(manifest, src_dir, dest_dir)
            print(
                f"[post-land-deploy:conductor] source fetch failed ({err_detail}); "
                f"night-plan copy SKIPPED, runtime preserved",
                file=sys.stderr,
            )
            msg = (
                f"conductor source fetch failed ({err_detail}); night-plan copy "
                f"SKIPPED, runtime preserved"
            )
        else:
            priority_low = False
            print(
                f"[post-land-deploy:conductor] source {reason}; "
                f"night-plan copy SKIPPED, runtime preserved",
                file=sys.stderr,
            )
            msg = (
                f"conductor deploy clone {reason}; night-plan script copy "
                f"SKIPPED, runtime preserved; restore /srv/git/conductor to main "
                f"— see spec §Go-live (night-plan-conductor-deploy-sync-v0)"
            )
        try:
            from agents_core.notify import send_notification, Priority as _P
            send_notification(
                message=msg,
                title="conductor: night-plan deploy skipped",
                priority=_P.LOW if priority_low else _P.NORMAL,
            )
        except Exception:
            pass
        return

    # --- R2/R9: source verified fresh — copy the closure, per-file isolated. ---
    for fname in manifest:
        try:
            # D3: per-file tracked-and-unmodified-vs-HEAD check, in addition to the
            # tree-level gate above — a manifested file that is itself untracked or
            # locally modified must never be delivered as though it were reviewed code.
            if not _conductor_file_verified_clean(clone, f"scripts/{fname}"):
                print(
                    f"[post-land-deploy:conductor] {fname} untracked or modified vs "
                    f"HEAD; skipping copy (D3)",
                    file=sys.stderr,
                )
                continue
            _copy_one_conductor_script(fname, src_dir, dest_dir, trigger)
        except OSError as e:
            print(f"[post-land-deploy:conductor] copy failed for {fname}: {e}", file=sys.stderr)
            continue


# ---------------------------------------------------------------------------
# GravityWell host-script install (host-script-deploy-on-land-v0)
# ---------------------------------------------------------------------------

def _gw_host_reachable() -> bool:
    """D4: passive reachability probe, NEVER a wake. BatchMode=yes fails fast on a
    key prompt; ConnectTimeout bounds the TCP handshake. No retry loop — a single
    failed probe means skip, full stop, until the next conductor land re-attempts.
    """
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes",
             "-o", f"ConnectTimeout={_GW_SSH_CONNECT_TIMEOUT_SECS}",
             "-o", "StrictHostKeyChecking=accept-new",
             _GW_SSH_HOST, "true"],
            capture_output=True, timeout=_GW_SSH_CONNECT_TIMEOUT_SECS + 5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _gw_ssh_capture(remote_cmd: list, timeout: int = _GW_SSH_CMD_TIMEOUT_SECS) -> str | None:
    """Run `ssh <host> <remote_cmd...>`, returning stdout on success or None on any
    failure (nonzero exit, timeout, transport error) — never raises."""
    try:
        result = subprocess.run(
            ["ssh", _GW_SSH_HOST] + list(remote_cmd),
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return result.stdout if result.returncode == 0 else None


def _gw_next_action(entry: _GwHostScript) -> str:
    """D7: the expected next action, per manifested script. Delivery always
    retries on the next conductor land (D4 — self-correcting, no retry loop
    needed within one pass); what happens meanwhile depends on whether the
    script self-attests: gw-topology's own guard (_assert_helper_compatible)
    refuses to converge through a stale copy, gw-idle-suspend.sh has no such
    guard and drifts silently."""
    guard = "will refuse reach() until delivered" if entry.self_attesting else "silent - no guard"
    return f"retries on next conductor land; {guard}"


def _gw_result(entry: _GwHostScript, *, status: str, reason: str | None = None) -> dict:
    """One manifest entry's outcome for this pass. status is one of "delivered",
    "noop", "skipped", "failed". reason is a structured token (D7 — never prose),
    populated for "skipped"/"failed" only."""
    return {
        "dest": entry.dest,
        "status": status,
        "reason": reason,
        "next_action": _gw_next_action(entry) if status in ("skipped", "failed") else None,
    }


def _install_one_gw_host_script(
    entry: _GwHostScript, clone: str, src_dir: Path, trigger: str, in_flight_flip: bool,
) -> dict:
    """D6: atomic remote install of one manifest entry, verified from the far side.

    Order: per-file tracked-and-unmodified check (D3) -> execution-sensitivity
    gate (D5) -> remote hash no-op check -> scp to a /tmp temp path -> `sudo
    install` (atomic replace, the remote equivalent of temp-in-dest-dir +
    os.replace) -> re-hash verify -> --help self-hash verify for self-attesting
    entries. A verify mismatch is reported "failed", never "delivered".
    """
    src = src_dir / entry.src_name
    if not src.exists():
        # A manifest entry naming a file conductor doesn't have is a manifest bug
        # (mirrors _copy_one_conductor_script's identical local-copy case).
        print(f"[post-land-deploy:conductor-gw] manifest source missing: {src}", file=sys.stderr)
        return _gw_result(entry, status="skipped", reason="manifest_source_missing")

    if not _conductor_file_verified_clean(clone, f"scripts/{entry.src_name}"):
        print(
            f"[post-land-deploy:conductor-gw] {entry.src_name} untracked or "
            f"modified vs HEAD; skipping install (D3)",
            file=sys.stderr,
        )
        return _gw_result(entry, status="skipped", reason="file_untracked_or_modified")

    if entry.execution_sensitive and in_flight_flip:
        # D5: never swap a script out from under a convergence. Per-file, not
        # global — only entries that declare execution_sensitive=True are gated.
        print(
            f"[post-land-deploy:conductor-gw] {entry.dest} skipped: GravityWell "
            f"in_flight_flip (D5)",
            file=sys.stderr,
        )
        return _gw_result(entry, status="skipped", reason="in_flight_flip")

    local_hash = hashlib.sha256(src.read_bytes()).hexdigest()

    pre_hash_out = _gw_ssh_capture(["sha256sum", entry.dest])
    pre_hash = pre_hash_out.split()[0] if pre_hash_out else None
    if pre_hash == local_hash:
        # D6: no-op when the hash already matches — log the no-op, never reinstall.
        print(f"[post-land-deploy:conductor-gw] {entry.dest} already current (no-op)", file=sys.stderr)
        return _gw_result(entry, status="noop")

    tmp_remote = f"/tmp/.lapis-pm-gw-deploy-{uuid.uuid4().hex}-{Path(entry.dest).name}"
    try:
        scp_result = subprocess.run(
            ["scp", "-p", str(src), f"{_GW_SSH_HOST}:{tmp_remote}"],
            capture_output=True, timeout=_GW_SCP_TIMEOUT_SECS,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[post-land-deploy:conductor-gw] scp to {entry.dest} errored: {e}", file=sys.stderr)
        return _gw_result(entry, status="failed", reason="scp_failed")
    if scp_result.returncode != 0:
        print(f"[post-land-deploy:conductor-gw] scp to {entry.dest} failed rc={scp_result.returncode}", file=sys.stderr)
        return _gw_result(entry, status="failed", reason="scp_failed")

    owner, _, group = entry.owner.partition(":")
    install_cmd = ["sudo", "-n", "install", "-o", owner]
    if group:
        install_cmd += ["-g", group]
    install_cmd += ["-m", format(entry.mode, "o"), tmp_remote, entry.dest]
    try:
        install_result = subprocess.run(
            ["ssh", _GW_SSH_HOST] + install_cmd,
            capture_output=True, text=True, timeout=_GW_SSH_CMD_TIMEOUT_SECS,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        install_result = None
        print(f"[post-land-deploy:conductor-gw] install on {entry.dest} errored: {e}", file=sys.stderr)
    finally:
        # Best-effort cleanup of the remote temp regardless of install outcome.
        _gw_ssh_capture(["rm", "-f", tmp_remote])

    if install_result is None or install_result.returncode != 0:
        detail = install_result.stderr.strip()[:200] if install_result is not None else "transport error"
        print(f"[post-land-deploy:conductor-gw] install failed for {entry.dest}: {detail}", file=sys.stderr)
        return _gw_result(entry, status="failed", reason="install_failed")

    # D6: always re-verify from the far side — a mismatch is a failed install,
    # never recorded as delivered, so the next land retries.
    post_hash_out = _gw_ssh_capture(["sha256sum", entry.dest])
    post_hash = post_hash_out.split()[0] if post_hash_out else None
    if post_hash != local_hash:
        print(
            f"[post-land-deploy:conductor-gw] post-install re-hash mismatch for "
            f"{entry.dest}: expected {local_hash[:12]}, got {(post_hash or 'unreadable')[:12]}",
            file=sys.stderr,
        )
        return _gw_result(entry, status="failed", reason="verify_hash_mismatch")

    if entry.self_attesting:
        # Same check gw_topology.py:_assert_helper_compatible performs — a green
        # verify here means the next reach() cannot refuse on drift. --help
        # touches nothing (no sudo, no GPU, no health calls).
        help_out = _gw_ssh_capture([entry.dest, "--help"])
        match = _GW_SELF_HASH_RE.search(help_out or "")
        remote_self_hash = match.group(1) if match else None
        local_self_hash = local_hash[:12]
        if remote_self_hash != local_self_hash:
            print(
                f"[post-land-deploy:conductor-gw] self-hash verify failed for "
                f"{entry.dest}: expected {local_self_hash}, got {remote_self_hash!r}",
                file=sys.stderr,
            )
            return _gw_result(entry, status="failed", reason="verify_self_hash_mismatch")

    # D8: provenance in the same deploy log every clone sync uses.
    old_repr = pre_hash[:8] if pre_hash else "absent"
    _write_deploy_log(f"gravitywell:{entry.dest}", old_repr, local_hash[:8], trigger)
    print(f"[post-land-deploy:conductor-gw] installed {entry.dest} ({local_hash[:12]})", file=sys.stderr)
    return _gw_result(entry, status="delivered")


def _read_gw_drift_ledger() -> dict:
    try:
        return json.loads(_GW_HOST_SCRIPT_DRIFT_LEDGER.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_gw_drift_ledger(ledger: dict) -> None:
    """Atomic write: write-to-temp-then-os.replace (mirrors deploy_inventory_repair's
    write_ledger — never a torn/partial file)."""
    try:
        _GW_HOST_SCRIPT_DRIFT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(_GW_HOST_SCRIPT_DRIFT_LEDGER.parent),
            prefix=".gw-host-script-drift-ledger-", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(ledger, indent=2, sort_keys=True))
            os.replace(tmp_name, str(_GW_HOST_SCRIPT_DRIFT_LEDGER))
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except OSError as e:
        print(f"[post-land-deploy:conductor-gw] drift ledger write failed: {e}", file=sys.stderr)


def _build_gw_host_script_gem_payload(results: list) -> dict:
    """D7: one gem per pass with an undelivered entry — reports BOTH the delivered
    and the undelivered sets (a partial delivery must never read as a clean state),
    each undelivered entry carrying its destination, structured reason, and
    expected next action."""
    delivered = [r for r in results if r["status"] in ("delivered", "noop")]
    undelivered = [r for r in results if r["status"] in ("skipped", "failed")]
    undelivered_lines = [f"{r['dest']}: {r['reason']} -> {r['next_action']}" for r in undelivered]
    delivered_lines = [f"{r['dest']}: {r['status']}" for r in delivered] or ["(none this pass)"]
    return {
        "title": f"gravitywell host-script drift: {len(undelivered)} undelivered",
        "ask": "; ".join(undelivered_lines),
        "why": (
            "Repo-held GravityWell scripts (gw-topology, gw-idle-suspend.sh) are "
            "delivered by this hook, not by hand. An undelivered script either "
            "refuses the next reach() (self-attesting) or drifts with no guard "
            "at all (see per-file reasons)."
        ),
        "context": [
            {"label": "Undelivered", "lines": undelivered_lines},
            {"label": "Delivered this pass", "lines": delivered_lines},
        ],
        "options": _GW_GEM_OPTIONS,
        "state": "needs",
        "origin": "from · gw-host-script-deploy",
        "agent": "repair-expert",
        "deposited_by": "lapis-pm-host-script-deploy-on-land-v0",
    }


def _report_gw_host_script_drift(results: list) -> None:
    """D7/D9: deposit a Desk gem for any undelivered manifest entry, deduped
    against a small local ledger so a persisting drift pages the Desk once —
    not on every conductor land (routes through the same Desk-gem path the
    deploy-inventory reconciler uses, not per-land Pushover).
    """
    undelivered = [r for r in results if r["status"] in ("skipped", "failed")]
    ledger = _read_gw_drift_ledger()
    entry = ledger.get(_GW_HOST_SCRIPT_DRIFT_SIGNATURE)
    now = _now_iso()

    if not undelivered:
        # Healthy pass: resolve any open ledger entry so a later regression
        # deposits a fresh gem instead of reading as still-suppressed.
        if entry is not None and entry.get("status") == "open":
            entry["status"] = "resolved"
            entry["last_seen"] = now
            _write_gw_drift_ledger(ledger)
        return

    would_deposit = entry is None or entry.get("status") in ("acked", "resolved")
    if not would_deposit:
        entry["last_seen"] = now
        entry["times_seen"] = entry.get("times_seen", 1) + 1
        _write_gw_drift_ledger(ledger)
        print(
            f"[post-land-deploy:conductor-gw] drift still open "
            f"({len(undelivered)} undelivered) — gem already on the Desk, not re-deposited",
            file=sys.stderr,
        )
        return

    payload = _build_gw_host_script_gem_payload(results)
    gem_id = None
    try:
        from . import deploy_inventory_repair
        gem_id = deploy_inventory_repair.deposit_gem(payload)
    except Exception as e:
        print(f"[post-land-deploy:conductor-gw] drift gem deposit errored: {e}", file=sys.stderr)

    if gem_id is None:
        print(
            "[post-land-deploy:conductor-gw] drift gem deposit failed or skipped; "
            "will retry next land",
            file=sys.stderr,
        )
        return  # ledger untouched — next pass retries the deposit

    ledger[_GW_HOST_SCRIPT_DRIFT_SIGNATURE] = {
        "gem_id": gem_id,
        "first_seen": entry.get("first_seen") if entry else now,
        "last_seen": now,
        "times_seen": (entry.get("times_seen", 0) + 1) if entry else 1,
        "status": "open",
    }
    _write_gw_drift_ledger(ledger)


# ---------------------------------------------------------------------------
# D3: land-time deploy reporting (night-deploy-manifest-attestation-v0)
# ---------------------------------------------------------------------------
# The land output states, in plain terms, what the deploy pass delivered
# (which host, which script, verified how) — or states explicitly that nothing
# was deployed and why. Silent-success land is the crack Erah named. The data
# source is the hook's existing result + _DEPLOY_LOG; no new deploy mechanism.
#
# Gate-3 amendments (folded rev 4):
#   * just-merged-PR gap naming — land lists any script named by the
#     just-merged PR's diff that is absent from the deployed set at land time.
#   * the mark reflects CURRENT LIVE STATE, not the historical ledger —
#     computed at land time from live inputs (fresh closure comparison + the
#     latest run-record verdict + the waiver file). The ledger remains the
#     enduring record (duration, waiver debt); the mark is a point-in-time
#     reading of it.
#   * the scar carries temporal context — the mark lands with its duration
#     from the ledger (DRIFT (3 nights, oldest 2026-09-05)).

_DEPLOY_ATTESTATION_MARKS = ("DRIFT", "WAIVER", "WAIVER_EXPIRED", "CLEAN")


def _just_merged_pr_scripts(repo: str, clone_paths: list[str]) -> list[str]:
    """Scripts named by the just-merged PR's diff (D3 gate-3 amendment).

    Derives the changed paths from the deploy-log line the hook just wrote
    (the same source `run_post_land_attestation` consumes — NOT an extra pair
    of git rev-parse calls competing with the hook's own HEAD checks). A
    .py path under the repo's scripts/ (or any .py the diff touched) is a
    candidate. Best-effort — an unreadable log / no diff is a clean [].
    """
    out: list[str] = []
    if not clone_paths:
        return out
    primary = clone_paths[0]
    try:
        old_sha, new_sha = _last_deploy_log_shas(primary, "post-land-hook")
        if not old_sha or not new_sha:
            return out
        diff = subprocess.run(
            ["git", "-C", primary, "diff", "--name-only", old_sha, new_sha],
            capture_output=True, text=True, timeout=10,
        )
        if diff.returncode != 0:
            return out
        # D9 (attestation-contract-v0 leg 2): intersect the diff candidates
        # with the files present at the merged HEAD. The diff range can name
        # files that are NOT in the merged tree (salvage-branch residue —
        # the #307 ghost DRIFT marks on cli.py / orchestrator.py), and a
        # file absent from the head cannot be a deployable that drifted.
        head_tree = subprocess.run(
            ["git", "-C", primary, "ls-tree", "-r", new_sha, "--name-only"],
            capture_output=True, text=True, timeout=10,
        )
        head_files = {
            ln.strip() for ln in head_tree.stdout.splitlines() if ln.strip()
        } if head_tree.returncode == 0 else None
        for ln in diff.stdout.splitlines():
            ln = ln.strip()
            if not ln.endswith(".py"):
                continue
            # Test-path files (a tests/ path component or a test_/conftest
            # basename) are never deployables — the #307 ghost mark on
            # test_council_resilience.py.
            parts = ln.split("/")
            if "tests" in parts[:-1]:
                continue
            base = parts[-1]
            if base.startswith("test_") or base == "conftest.py":
                continue
            if head_files is not None and ln not in head_files:
                continue
            if base not in out:
                out.append(base)
    except (subprocess.TimeoutExpired, OSError):
        pass
    return out


def _attestation_drift_duration(ledger_records: list) -> str | None:
    """Duration from the attestation ledger (the temporal scar, D3): how many
    nights the drift has persisted and the oldest observed date. None if the
    ledger has no drift record."""
    drift_dates: list[str] = []
    for rec in ledger_records:
        if not isinstance(rec, dict):
            continue
        if rec.get("verdict") == "held" and rec.get("reason") in (
            "drift", "waiver_expired", "waiver_forever",
        ):
            ts = rec.get("ts")
            if ts:
                drift_dates.append(str(ts)[:10])
    if not drift_dates:
        return None
    oldest = min(drift_dates)
    nights = len(drift_dates)
    return f"{nights} nights, oldest {oldest}"


def _latest_run_record_attestation(
    run_record_path: Path | None = None,
) -> dict | None:
    """The latest night-plan run record's attestation-node block (current live
    read of /data/slots/night-plan-runs/latest.json — the pinned DoD-0/D5
    consumer). Returns {"verdict", "reason", "ended_at"} or None.

    Best-effort — an unreadable/malformed run record is None, never a crash.
    """
    from . import attestation as _att

    run_record_path = run_record_path or _att.RUN_RECORD_PATH
    try:
        doc = json.loads(Path(run_record_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    nodes = doc.get("nodes")
    if not isinstance(nodes, list):
        return None
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == _att.ATTTESTATION_NODE_ID:
            return {
                "verdict": node.get("verdict"),
                "reason": node.get("hold_reason_class") or node.get("reason"),
                "ended_at": node.get("ended_at"),
            }
    return None


def _compute_land_attestation_mark(
    repo: str,
    pr_scripts: list[str],
    ledger_records: list | None = None,
    run_record: dict | None = None,
) -> tuple[str, str]:
    """Compute the land-time mark from CURRENT LIVE inputs (D3): fresh
    closure comparison + the latest run-record verdict + the waiver file.

    The mark covers the FULL audit surface, not just the just-merged PR's own
    scripts: the run-record verdict is consulted so a PRE-EXISTING drift in an
    unrelated script (the attestation node's latest verdict is `held`) reads
    DRIFT at land time, not CLEAN. The just-merged-PR closure check names the
    targeted crack (the gate-3 amendment); the run-record verdict carries the
    rest of the live state.

    Returns (mark, detail). mark is one of DRIFT / WAIVER / WAIVER_EXPIRED /
    CLEAN; detail carries the duration (the temporal scar).
    """
    from . import attestation as _att

    waivers, _problems = _att.load_waivers()
    ledger_records = ledger_records if ledger_records is not None else _att.read_ledger()
    run_record = run_record if run_record is not None else _latest_run_record_attestation()

    # Fresh closure comparison: which pr scripts are in a manifest / waived /
    # drifted right now.
    try:
        closure = _att.manifest_closure_from_pm_core()
    except Exception:
        closure = None
    local_manifested = closure.local_manifested() if closure else set()

    drifted: list[str] = []
    waived_now: list[str] = []
    waived_expired: list[str] = []
    for script in pr_scripts:
        base = Path(script).name
        w = _att.waiver_for(script, waivers)
        if w is not None:
            if w.is_expired():
                waived_expired.append(base)
            else:
                waived_now.append(base)
        # A pr script absent from every manifest and not waived is the named
        # gap — it is a drift row at land time.
        if base not in local_manifested and w is None:
            drifted.append(base)

    # The latest run-record verdict (current live state, D3): a `held`
    # attestation node means pre-existing drift on the audit surface that the
    # just-merged-PR subset does not cover.
    held_reason = ""
    if run_record and run_record.get("verdict") == "held":
        held_reason = str(run_record.get("reason") or "held")

    if waived_expired:
        return "WAIVER_EXPIRED", ", ".join(waived_expired)
    if drifted:
        detail = ", ".join(drifted)
        if held_reason:
            detail += f"; run-record held ({held_reason})"
        return "DRIFT", detail
    if held_reason:
        # No pr-script gap, but the latest night's attestation node held —
        # pre-existing drift in an unrelated script. Not CLEAN (D3: the mark
        # reflects current live state, not just the just-merged PR).
        return "DRIFT", f"pre-existing drift on the audit surface (run-record held: {held_reason})"
    if waived_now:
        return "WAIVER", ", ".join(waived_now)
    return "CLEAN", ""


def render_land_deploy_report(
    repo: str,
    pr_scripts: list[str] | None = None,
    ledger_records: list | None = None,
    run_record: dict | None = None,
) -> str:
    """D3: the plain-terms land output statement. States what the deploy pass
    delivered (host, script, verified how) — or explicitly that nothing was
    deployed and why — PLUS names any just-merged-PR script absent from the
    deployed set. The mark (DRIFT/WAIVER/WAIVER_EXPIRED/CLEAN) is computed at
    land time from current live inputs (fresh closure comparison + the latest
    run-record verdict + the waiver file) and carries duration from the
    ledger.

    Best-effort — a failure here must never fail the land (it lands on the
    land output before the archive step, but a reporting bug degrades to a
    minimal statement, never a crash).
    """
    from . import attestation as _att

    lines: list[str] = []
    lines.append("[land:deploy] deploy pass report")

    # --- what the deploy pass delivered (from _DEPLOY_LOG, current live) ---
    delivered = _deploy_log_lines_for_repo(repo)
    if delivered:
        lines.append(f"  delivered {len(delivered)} script(s) this pass:")
        for d in delivered:
            lines.append(f"    {d['tree']} | {d['change']} | verified: {d['verified']}")
    else:
        # Explicit: nothing was deployed, and why (D3 — no silent green).
        reason = _why_no_deploy(repo)
        lines.append(f"  nothing deployed this pass ({reason})")

    # --- just-merged-PR gap naming (D3 gate-3 amendment) ---
    pr_scripts = pr_scripts if pr_scripts is not None else _just_merged_pr_scripts(
        repo, _POST_LAND_PULL.get(repo) or []
    )
    if pr_scripts:
        try:
            closure = _att.manifest_closure_from_pm_core()
            local_manifested = closure.local_manifested()
        except Exception:
            local_manifested = set()
        absent = [s for s in pr_scripts if Path(s).name not in local_manifested]
        if absent:
            lines.append(
                f"  GAP: {len(absent)} script(s) from the just-merged PR are "
                f"absent from the deployed set at land time: {', '.join(absent)}"
                " — waived? (a silent-green land is the crack this report closes)"
            )
        else:
            lines.append(
                f"  just-merged PR scripts ({len(pr_scripts)}): all in the "
                "deployed set"
            )

    # --- the mark (current live state + duration from the ledger) ---
    if ledger_records is None:
        try:
            ledger_records = _att.read_ledger()
        except Exception:
            ledger_records = []
    if run_record is None:
        try:
            run_record = _latest_run_record_attestation()
        except Exception:
            run_record = None
    mark, detail = _compute_land_attestation_mark(
        repo, pr_scripts, ledger_records, run_record=run_record
    )
    duration = _attestation_drift_duration(ledger_records)
    if mark == "DRIFT" and duration:
        lines.append(f"  mark: DRIFT ({duration}) — {detail}")
    elif mark == "DRIFT":
        lines.append(f"  mark: DRIFT — {detail}")
    elif mark in ("WAIVER", "WAIVER_EXPIRED"):
        lines.append(f"  mark: {mark} — {detail}")
    else:
        lines.append("  mark: CLEAN")

    return "\n".join(lines)


def _deploy_log_lines_for_repo(repo: str) -> list[dict]:
    """The most recent deploy-log lines for `repo`'s trees (current live read
    of _DEPLOY_LOG). `verified` is how the pass verified delivery (the
    per-file tracked-and-unmodified D3 gate + the atomic copy)."""
    trees = set(_POST_LAND_PULL.get(repo) or [])
    out: list[dict] = []
    try:
        lines = Path(_DEPLOY_LOG).read_text().splitlines()
    except OSError:
        return out
    for line in reversed(lines):
        m = _DEPLOY_LOG_LINE_RE.match(line)
        if m and m.group("tree") in trees:
            change = m.group("old") + ".." + m.group("new")
            out.append({
                "tree": m.group("tree"),
                "change": change,
                "verified": "per-file tracked-and-unmodified vs HEAD + atomic copy",
            })
        if len(out) >= 8:
            break
    return out


def _why_no_deploy(repo: str) -> str:
    """Why nothing was deployed this pass (D3 — explicit, not silent)."""
    if _DEPLOY_HOOK_DISABLED:
        return "deploy hook disabled (LAPIS_PM_DEPLOY_HOOK_DISABLE=1)"
    if repo not in _POST_LAND_PULL:
        return f"{repo} has no deploy-clone pull mapping"
    return "HEAD unchanged (no-op pull) or source not fresh — runtime preserved"


def _deploy_conductor_gw_host_scripts_impl(trigger: str) -> None:
    clone = _CONDUCTOR_DEPLOY_CLONE
    src_dir = Path(_CONDUCTOR_SCRIPTS_SRC)

    is_fresh, reason = _conductor_source_freshness_gate(clone)
    if not is_fresh:
        print(
            f"[post-land-deploy:conductor-gw] source not fresh ({reason}); "
            f"host-script install SKIPPED for all manifest entries",
            file=sys.stderr,
        )
        results = [
            _gw_result(entry, status="skipped", reason="source_not_fresh")
            for entry in _CONDUCTOR_GW_HOST_SCRIPTS
        ]
        _report_gw_host_script_drift(results)
        return

    if not _gw_host_reachable():
        # D4: unreachable means skip — no WoL, no wake-gravitywell, no retry loop.
        # Self-correcting: re-attempted on every subsequent conductor land.
        print(
            "[post-land-deploy:conductor-gw] gravitywell unreachable; host-script "
            "install SKIPPED (no wake attempted, D4)",
            file=sys.stderr,
        )
        results = [
            _gw_result(entry, status="skipped", reason="host_unreachable")
            for entry in _CONDUCTOR_GW_HOST_SCRIPTS
        ]
        _report_gw_host_script_drift(results)
        return

    in_flight_flip = False
    try:
        from agents_core.llm import gw_serving_state
        state = gw_serving_state()
        in_flight_flip = bool(state.in_flight_flip)
    except Exception as e:
        print(
            f"[post-land-deploy:conductor-gw] gw_serving_state check failed "
            f"(treating as not in-flight): {e}",
            file=sys.stderr,
        )

    results = []
    for entry in _CONDUCTOR_GW_HOST_SCRIPTS:
        try:
            result = _install_one_gw_host_script(entry, clone, src_dir, trigger, in_flight_flip)
        except Exception as e:
            print(
                f"[post-land-deploy:conductor-gw] unexpected error installing "
                f"{entry.dest} (non-fatal): {e}",
                file=sys.stderr,
            )
            result = _gw_result(entry, status="failed", reason="unexpected_error")
        results.append(result)

    _report_gw_host_script_drift(results)


def _deploy_conductor_gw_host_scripts(trigger: str = "post-land-hook") -> None:
    """Cross-host sibling of _deploy_conductor_night_scripts: installs conductor's
    repo-held GravityWell scripts onto gravitywell:/usr/local/sbin/ (D1-D8).

    Never raises: an unexpected error anywhere in this function is caught, logged,
    and reported at LOW priority — this runs inside the post-land hook on the merge
    path for every repo, and a failure here must never fail the merge that
    triggered it (mirrors _deploy_conductor_night_scripts's own contract).

    Never converges or restarts a GravityWell serving unit, never wakes the host —
    installing a file is inert; reach() remains the only thing that converges
    topology (spec Out-of-scope).
    """
    try:
        _deploy_conductor_gw_host_scripts_impl(trigger)
    except Exception as e:
        print(
            f"[post-land-deploy:conductor-gw] unexpected error (non-fatal): {e}",
            file=sys.stderr,
        )
        try:
            from agents_core.notify import send_notification, Priority as _P
            send_notification(
                message=f"conductor GravityWell host-script install hit an unexpected error: {e}",
                title="conductor: gw host-script deploy error",
                priority=_P.LOW,
            )
        except Exception:
            pass


def _count_inflight_fixers() -> int:
    """Count active fixer dispatches from the claude-queue active/ dir. Best-effort."""
    try:
        return sum(1 for p in _CLAUDE_QUEUE_ACTIVE_DIR.iterdir() if p.suffix == ".yaml")
    except OSError:
        return 0


def _read_restart_pending(repo: str) -> dict | None:
    """Read the pending-restart marker for repo. Returns None if absent or unreadable."""
    path = _RESTART_PENDING_DIR / f"{repo}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_restart_pending(
    repo: str,
    units: tuple[str, ...],
    first_deferred_at: str,
) -> None:
    """Write the pending-restart marker for repo. Best-effort."""
    try:
        _RESTART_PENDING_DIR.mkdir(parents=True, exist_ok=True)
        path = _RESTART_PENDING_DIR / f"{repo}.json"
        path.write_text(json.dumps({
            "repo": repo,
            "units": list(units),
            "first_deferred_at": first_deferred_at,
            "post_head": "",
        }))
    except OSError as e:
        print(
            f"[post-land-deploy] WARNING: failed to write restart-pending marker for {repo}: {e}",
            file=sys.stderr,
        )


def _clear_restart_pending(repo: str) -> None:
    """Remove the pending-restart marker for repo. Best-effort."""
    try:
        (_RESTART_PENDING_DIR / f"{repo}.json").unlink(missing_ok=True)
    except OSError:
        pass


def _issue_system_restart(unit: str) -> None:
    """Run `sudo -n systemctl restart <unit>`. Logs failure to stderr; best-effort."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "systemctl", "restart", unit],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(
                f"[post-land-deploy] restart {unit} failed "
                f"rc={result.returncode}: {result.stderr.strip()[:200]}",
                file=sys.stderr,
            )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(
            f"[post-land-deploy] restart {unit} errored: {e}",
            file=sys.stderr,
        )


def _clone_currency_shas(clone_path: str) -> tuple[str, str]:
    """Fetch + return (local HEAD, origin/main HEAD) shas for `clone_path`.

    Returns ("", "") on any check failure (network/git error, unparseable
    output) — treated as indeterminate, never as "stale", so a transient error
    never fires a false alert.
    """
    try:
        subprocess.run(
            ["git", "-C", clone_path, "fetch", "origin", "main", "--quiet"],
            capture_output=True, timeout=15,
        )
        local_proc = subprocess.run(
            ["git", "-C", clone_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        remote_proc = subprocess.run(
            ["git", "-C", clone_path, "rev-parse", "origin/main"],
            capture_output=True, text=True, timeout=5,
        )
        local = local_proc.stdout.strip() if local_proc.returncode == 0 else ""
        remote = remote_proc.stdout.strip() if remote_proc.returncode == 0 else ""
    except Exception:
        return "", ""
    return local, remote


def _check_deploy_currency() -> None:
    """Alert via Pushover for every critical clone that is behind origin/main.

    Generalizes the original lapis-pm-only check to every path mapped for every
    repo in `_POST_LAND_PULL_CRITICAL` (lapis-pm, agents-core's two clones,
    synapse) — the un-loseable detector (agents-core-deploy-drift-backstop-v0
    §A). Because this runs inside lapis-pm's own tick, and lapis-pm's own clone
    is kept current by its own backstop timer (PR #116), it detects agents-core
    or synapse drift even when their own hook-based alerting is itself among the
    undeployed commits.

    Called once per tick_all(). Catches timer outage and persistent pull
    failures that would otherwise be silent. Cooldown: at most one alert per
    clone per `_DEPLOY_CURRENCY_COOLDOWN_SECS` — a different drifted clone
    alerts independently of another clone's cooldown.
    """
    for repo in sorted(_POST_LAND_PULL_CRITICAL):
        for clone_path in _POST_LAND_PULL.get(repo, []):
            _check_one_clone_currency(repo, clone_path)


def _write_deploy_currency_status(
    repo: str, clone_path: str, sha: str, distance: int | None,
) -> None:
    """Persist the latest per-clone currency snapshot to a shared status file.

    §D (agents-core-deploy-drift-backstop-v0): a quantitative commit-distance
    meter, not just a boolean alert. Written on every check (current or stale)
    so a consumer (e.g. an agents-core-owned `/status` surface, out of this
    repo's scope to wire up directly) can render `stale: N` without waiting for
    an alert. Best-effort — a write failure never blocks the currency check.
    """
    try:
        _DEPLOY_CURRENCY_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(_DEPLOY_CURRENCY_STATUS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        data[clone_path] = {
            "repo": repo,
            "sha": sha,
            "commits_behind": distance,
            "checked_at": _now_iso(),
        }
        _DEPLOY_CURRENCY_STATUS_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError as e:
        print(
            f"[deploy-currency] status file write failed for {clone_path}: {e}",
            file=sys.stderr,
        )


def _check_one_clone_currency(repo: str, clone_path: str) -> None:
    local, remote = _clone_currency_shas(clone_path)
    if not local or not remote:
        return  # indeterminate — never overwrite the status file on a transient error

    distance = 0 if local == remote else _commit_distance(clone_path)
    _write_deploy_currency_status(repo, clone_path, local, distance)

    if local == remote:
        return  # up to date

    # Per-clone cooldown key: a different drifted clone alerts independently.
    stale_key = f"{_DEPLOY_CURRENCY_STALE_KEY}:{clone_path}"
    try:
        last_raw = _mem().get(stale_key)
        if last_raw:
            last_ts = datetime.fromisoformat(last_raw["content"])
            now_ts = datetime.now(last_ts.tzinfo)
            if (now_ts - last_ts).total_seconds() < _DEPLOY_CURRENCY_COOLDOWN_SECS:
                return
    except Exception as e:
        logger.warning(
            "[deploy-currency] cooldown read failed for %s (failing open, alerting): %s",
            stale_key, e,
        )

    distance_repr = distance if distance is not None else "?"

    print(
        f"[deploy-currency] ALERT: {repo} clone at {clone_path} is at {local[:8]} "
        f"but origin/main is {remote[:8]} ({distance_repr} commits behind) — "
        f"deploy timer may be broken",
        file=sys.stderr,
        flush=True,
    )
    try:
        from agents_core.notify import send_notification, Priority as _P
        send_notification(
            message=(
                f"{repo} clone at {clone_path} is stale: running {local[:8]} but "
                f"origin/main is {remote[:8]} ({distance_repr} commits behind). "
                "Deploy timer may be broken or post-land pull failing silently."
            ),
            title=f"{repo}: deploy currency stale",
            priority=_P.NORMAL,
        )
    except Exception:
        pass
    try:
        _mem().set(
            stale_key,
            _now_iso(),
            tags=["lapis-pm", "deploy"],
        )
    except Exception:
        pass


def _deploy_inventory_high_kinds(clone: dict) -> set[str]:
    return {f["kind"] for f in clone.get("findings", []) if f.get("severity") == "HIGH"}


def _auto_recovery_eligible(clone: dict) -> bool:
    """A clone qualifies for script-tier auto-recovery iff it's mapped, its only
    HIGH finding is `stale_behind_origin` (no `stray_branch`/`tracked_dirty_tree`
    — equivalent to "on main, tree clean, only stale"), and no deploy-pull-lock
    is present. The lock check is independent and unconditional — it does not
    rely on _post_land_git_pull's own critical-repo-only lock check
    (pm_core.py:639), since an automated timer-fired trigger deserves the more
    conservative rule regardless of the lock's age or the repo's critical status.
    """
    if not clone.get("mapped"):
        return False
    if _deploy_inventory_high_kinds(clone) != {"stale_behind_origin"}:
        return False
    if _deploy_pull_locked(clone["path"]):
        return False
    return True


def _poll_unit_active(unit: str, *, user: bool) -> bool:
    """Poll `systemctl [--user] is-active <unit>` at most once per
    _AUTO_RECOVERY_RESTART_POLL_INTERVAL_SECS until it reports active or
    _AUTO_RECOVERY_RESTART_POLL_WINDOW_SECS elapses.
    """
    cmd = ["systemctl"] + (["--user"] if user else []) + ["is-active", unit]
    deadline = time.monotonic() + _AUTO_RECOVERY_RESTART_POLL_WINDOW_SECS
    while True:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip() == "active":
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(_AUTO_RECOVERY_RESTART_POLL_INTERVAL_SECS)


def _poll_restart_covered_units(repo: str) -> tuple[bool, list[str]]:
    """(all_active, failed_units) across every unit `repo` maps to in
    _POST_LAND_RESTART / _POST_LAND_RESTART_USER. An empty restart-covered set
    (code-reviewer, facets, gardener, conductor, rag-ops, lapis-pm — all
    timer-oneshot/per-invocation re-import, no restart needed) is vacuously
    successful by design: `all(...)` over `[]` is True, matching how
    _post_land_deploy_hook itself already treats these repos — not a gap to
    guard against.
    """
    failed: list[str] = []
    for unit in _POST_LAND_RESTART.get(repo, ()):
        if not _poll_unit_active(unit, user=False):
            failed.append(unit)
    for unit in _POST_LAND_RESTART_USER.get(repo, ()):
        if not _poll_unit_active(unit, user=True):
            failed.append(unit)
    return not failed, failed


def _run_deploy_inventory_auto_recovery(status: dict) -> None:
    """Script-tier auto-recovery pass (lapis-pm-deploy-inventory-auto-recovery-v0).

    For every eligible clone in `status["clones"]` (see _auto_recovery_eligible),
    pull + restart via the existing _post_land_deploy_hook primitive — never
    reimplement pull/restart logic here — then re-verify both git currency and
    restart-covered unit health before deciding success. _issue_system_restart is
    fire-and-forget, so relying on git currency alone would let a drifted-map
    restart failure silently read as a full success.

    Mutates each attempted clone's entry in `status["clones"]` in place with the
    fresh post-recovery state, so both the persisted status JSON and the
    HIGH-finding notify loop that runs after this see reality, not the stale
    pre-recovery snapshot.

    Success sends one Priority.LOW "auto-recovered" ping per clone. Failure
    (pull still fails, a new HIGH finding, or a restart-covered unit never
    reaches active) leaves the finding in place — or, if git currency alone
    looks clean but a restart did not take, synthesizes an
    `auto_recovery_restart_failed` HIGH finding so a drifted _POST_LAND_RESTART
    entry is never silently reported as success. Either way it flows through
    the existing alert path unchanged.

    Fail-soft per clone: an exception recovering one clone is logged and does
    not prevent other clones from being attempted.
    """
    from . import deploy_inventory

    reverse_map = deploy_inventory._reverse_pull_map(_POST_LAND_PULL)

    for clone in status.get("clones", []):
        try:
            if not _auto_recovery_eligible(clone):
                continue

            path = clone["path"]
            repo = reverse_map.get(path)
            if repo is None:
                continue

            pre_commits_behind = clone.get("commits_behind")

            _post_land_deploy_hook(repo, trigger="deploy-inventory-auto-recovery")

            currency = deploy_inventory.check_currency(path)
            clone["head"] = currency.head
            clone["branch"] = currency.branch
            clone["commits_behind"] = currency.commits_behind
            clone["tracked_dirty"] = currency.tracked_dirty
            clone["untracked_present"] = currency.untracked_present
            clone["findings"] = [asdict(f) for f in currency.findings]

            git_clean = currency.commits_behind == 0 and not any(
                f.severity == "HIGH" for f in currency.findings
            )
            restart_ok, failed_units = _poll_restart_covered_units(repo)

            if git_clean and restart_ok:
                try:
                    from agents_core.notify import send_notification, Priority as _P
                    behind_repr = pre_commits_behind if pre_commits_behind is not None else "?"
                    send_notification(
                        message=(
                            f"{path} was {behind_repr} commit(s) behind origin/main - "
                            f"auto-pulled and restarted."
                        ),
                        title="deploy-inventory: auto-recovered",
                        priority=_P.LOW,
                    )
                except Exception:
                    pass
            elif git_clean and not restart_ok:
                clone["findings"].append(asdict(deploy_inventory.Finding(
                    "auto_recovery_restart_failed", "HIGH",
                    f"{path} auto-recovery pulled to origin/main but "
                    f"{failed_units} did not reach active within "
                    f"{_AUTO_RECOVERY_RESTART_POLL_WINDOW_SECS:.0f}s",
                )))
            # else: git currency alone still shows a HIGH finding (pull failed,
            # still behind, or a new stray/dirty finding) — leave as-is; the
            # existing HIGH-finding notify loop below handles it unchanged.
        except Exception as e:
            logger.warning(
                "[deploy-inventory] auto-recovery failed for %s (non-fatal): %s",
                clone.get("path"), e,
            )


def _reconcile_deploy_inventory() -> None:
    """Cooldown-gated wrapper around deploy_inventory.run_reconcile_pass().

    Extends _check_deploy_currency (critical-repos-only) to every mapped-or-
    derived load-bearing clone, and closes the "no one ever compares the maps
    against the live host" gap (lapis-pm-deploy-inventory-reconciler-v0).
    Fail-soft: a reconciler failure degrades to a logged warning and never
    stalls tick_all() (spec item 7).

    Pushover notification is deduped against the prior status snapshot
    (lapis-pm-deploy-inventory-notify-dedup-v0): only HIGH findings new since
    the last pass are pushed. The stderr HIGH line is always emitted.
    """
    try:
        last_raw = _mem().get(_DEPLOY_INVENTORY_LAST_RUN_KEY)
        if last_raw:
            last_ts = datetime.fromisoformat(last_raw["content"])
            now_ts = datetime.now(last_ts.tzinfo)
            if (now_ts - last_ts).total_seconds() < _DEPLOY_INVENTORY_COOLDOWN_SECS:
                return
    except Exception as e:
        logger.warning(
            "[deploy-inventory] cooldown read failed (failing open, running pass): %s", e,
        )

    try:
        from . import deploy_inventory

        # Dedup (lapis-pm-deploy-inventory-notify-dedup-v0): only Pushover HIGH
        # findings that are new since the prior snapshot. `prior_high_keys` stays
        # None for a legitimate first-ever pass (bootstrap: notify everything once)
        # and also for a present-but-unparseable prior snapshot (corrupt: per-finding
        # notifications are suppressed below in favor of one loud audit push - never
        # fail-open to notifying every finding, never fail-silent about corruption).
        prior_high_keys = None
        corrupt = False
        try:
            prior_status = deploy_inventory.read_status_json()
            if prior_status is not None:
                prior_high_keys = deploy_inventory.high_finding_keys(prior_status)
        except Exception as e:
            corrupt = True
            logger.warning(
                "[deploy-inventory] prior status file unreadable, treating as corrupt: %s", e,
            )

        status = deploy_inventory.run_reconcile_pass(
            _POST_LAND_PULL, _POST_LAND_RESTART, _POST_LAND_RESTART_USER,
        )

        try:
            _run_deploy_inventory_auto_recovery(status)
        except Exception as e:
            logger.warning(
                "[deploy-inventory] auto-recovery step failed (non-fatal): %s", e,
            )

        deploy_inventory.write_status_json(status)

        for clone in status.get("clones", []):
            for finding in clone.get("findings", []):
                if finding.get("severity") != "HIGH":
                    continue
                print(
                    f"[deploy-inventory] HIGH: {clone['path']} {finding['kind']}: {finding['detail']}",
                    file=sys.stderr, flush=True,
                )

        # deploy-inventory-repair-proposer-v0: replaces the Pushover-to-Erah
        # relay (both the per-finding loop and the corrupt-status one-off) with
        # Desk-gem deposits. Fault-isolated per finding — a deposit/diagnosis
        # failure degrades to log+skip for that finding, never suppresses the
        # write_status_json/cooldown-timestamp writes that follow.
        try:
            from . import deploy_inventory_repair

            deploy_inventory_repair.run_repair_pass(
                status, prior_high_keys=prior_high_keys, corrupt=corrupt,
            )
        except Exception as e:
            logger.warning(
                "[deploy-inventory] repair-proposer pass failed (non-fatal): %s", e,
            )
    except Exception as e:
        logger.warning("[deploy-inventory] reconcile pass failed (non-fatal): %s", e)
        return

    try:
        _mem().set(_DEPLOY_INVENTORY_LAST_RUN_KEY, _now_iso(), tags=["lapis-pm", "deploy"])
    except Exception:
        pass


def _post_land_deploy_hook(
    repo: str | None, trigger: str = "post-land-hook", target_id: str | None = None,
) -> None:
    """Pull working clones then restart long-running services for `repo`.

    Best-effort. Failures (sudo unavailable, unit missing, restart timeout)
    are logged to stderr. Never raises — landing must complete even if the
    pull or restart fails.

    HEAD-advance gate: restarts are only issued when git pull actually advances
    HEAD. A no-op pull (already current) logs a one-line noop and skips restart,
    eliminating the double-restart class on idempotent backstop ticks.

    In-flight deferral (claude-queue-runner only): if fixers are in-flight when
    HEAD advances, the claude-queue-runner restart is deferred via a persistent
    marker under _RESTART_PENDING_DIR. Subsequent ticks fire it once the queue
    drains. The deferral is bounded by RESTART_DEFER_MAX_S (default 30 min); at
    the deadline a restart is forced anyway, and a CRITICAL-severity alert fires.
    Interrupted fixers are recoverable via the existing lost-fixer-retry path.

    system-spec-drift-attestation-v0: attached HERE, at the hook's
    *definition*, not at any one of its four call sites (:567, :1463,
    :3120, :3200) — attaching at a single call site would leave three of
    the four land paths accruing no drift (spec §Revision 2, amendment c).
    `target_id` is optional and only available from two of those call
    sites; when absent, drift provenance falls back to `trigger`. Best-
    effort and never raises, matching this function's own contract above.
    """
    if _DEPLOY_HOOK_DISABLED:
        print(
            "[post-land-deploy:disabled] skipping restart "
            "(LAPIS_PM_DEPLOY_HOOK_DISABLE=1)",
            file=sys.stderr,
        )
        return

    head_advanced = _post_land_git_pull(repo, trigger=trigger)

    # system-spec-drift-attestation-v0: derive the changed paths from the
    # deploy-log line `_post_land_git_pull` itself just wrote — deliberately
    # NOT an extra pair of `git rev-parse` calls bracketing that call, which
    # would otherwise compete with its own internal pre/post HEAD checks.
    if repo and head_advanced:
        try:
            _clone_paths = _POST_LAND_PULL.get(repo) or []
            if _clone_paths:
                primary = _clone_paths[0]
                old_sha, new_sha = _last_deploy_log_shas(primary, trigger)
                if old_sha and new_sha:
                    _diff = subprocess.run(
                        ["git", "-C", primary, "diff", "--name-only", old_sha, new_sha],
                        capture_output=True, text=True, timeout=10,
                    )
                    if _diff.returncode == 0:
                        changed = [ln for ln in _diff.stdout.splitlines() if ln.strip()]
                        from . import spec_attestation
                        spec_attestation.run_post_land_attestation(
                            repo, changed, target_id=target_id, trigger=trigger,
                        )
        except Exception as e:
            print(f"[post-land-deploy] spec-attestation pass failed (non-fatal): {e}", file=sys.stderr)

    if not repo:
        return

    # night-plan-conductor-deploy-sync-v0 (R2/R3): copy the night-plan script closure
    # into the /data/agents/scripts runtime. conductor is absent from both restart maps
    # below (R5 — night scripts are timer-oneshot, no restart needed), so this branch
    # does not interfere with the restart logic that follows.
    if repo == "conductor":
        # lapis-pm-conductor-brix-gw-runtime-deploy-manifest-v0: _deploy_conductor_
        # night_scripts below now also carries the GW actuator family
        # (_CONDUCTOR_BRIX_GW_RUNTIME_SCRIPTS) in the SAME copy pass — one combined
        # pass, not a second independent call, so a source fetch failure still
        # pages exactly once (R10 single-alert-ownership, see the closure tests).
        _deploy_conductor_night_scripts(trigger=trigger)
        # host-script-deploy-on-land-v0: cross-host sibling — installs conductor's
        # repo-held GravityWell scripts (gw-topology, gw-idle-suspend.sh) onto
        # gravitywell:/usr/local/sbin/. Independent of the local copy above (own
        # never-raises contract); never converges or restarts a GW serving unit.
        _deploy_conductor_gw_host_scripts(trigger=trigger)

    # night-deploy-manifest-attestation-v0 (D3): the land output states what
    # the deploy pass delivered (host, script, verified how) — or explicitly
    # that nothing was deployed and why — plus names any just-merged-PR script
    # absent from the deployed set. The mark (DRIFT/WAIVER/WAIVER_EXPIRED/
    # CLEAN) is computed at land time from current live inputs and carries
    # duration from the attestation ledger. Silent-success land is the crack
    # Erah named; this statement lands on the land output before archive.
    # Best-effort — a reporting bug degrades to a minimal statement, never a
    # crash, and never fails the land.
    try:
        print(render_land_deploy_report(repo), file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[post-land-deploy] D3 deploy report failed (non-fatal): {e}", file=sys.stderr)

    units = _POST_LAND_RESTART.get(repo)
    if units:
        # Only claude-queue-runner defers while fixers are in-flight;
        # all other units (gpu-queue-runner, synapse, etc.) restart immediately on advance.
        deferred_units = tuple(u for u in units if u == "claude-queue-runner.service")
        immediate_units = tuple(u for u in units if u != "claude-queue-runner.service")

        if deferred_units:
            # In-flight-aware path: check for an existing pending-restart marker first.
            # Guard: don't stack a new restart while a pending marker is live.
            pending = _read_restart_pending(repo)

            if pending:
                inflight = _count_inflight_fixers()
                try:
                    first_dt = datetime.fromisoformat(pending["first_deferred_at"])
                    elapsed = (datetime.now(first_dt.tzinfo) - first_dt).total_seconds()
                except Exception:
                    elapsed = RESTART_DEFER_MAX_S + 1  # parse failure → treat as budget exceeded
                pend_units = tuple(pending.get("units", deferred_units))

                if inflight == 0 or elapsed >= RESTART_DEFER_MAX_S:
                    if inflight > 0:
                        msg = (
                            f"defer budget exceeded ({elapsed:.0f}s/{RESTART_DEFER_MAX_S}s); "
                            f"forcing restart of {list(pend_units)} with {inflight} in-flight — "
                            f"interrupted fixers will be re-dispatched via lost-fixer-retry"
                        )
                        print(f"[post-land-deploy] CRITICAL: {msg}", file=sys.stderr)
                        try:
                            from agents_core.notify import send_notification, Priority as _P
                            send_notification(
                                message=msg,
                                title=f"{repo}: restart defer budget exceeded",
                                priority=_P.HIGH,
                            )
                        except Exception:
                            pass
                    else:
                        print(
                            f"[post-land-deploy] deferred restart now firing: "
                            f"{list(pend_units)} (queue idle, {elapsed:.0f}s after deferral)",
                            file=sys.stderr,
                        )
                    for unit in pend_units:
                        _issue_system_restart(unit)
                    _clear_restart_pending(repo)
                else:
                    print(
                        f"[post-land-deploy] deferring {list(pend_units)} restart: "
                        f"{inflight} fixers in-flight "
                        f"({elapsed:.0f}s / {RESTART_DEFER_MAX_S}s)",
                        file=sys.stderr,
                    )
                # A pending marker is live: do not add a new one even if HEAD advanced.

            elif not head_advanced:
                print(
                    f"[post-land-deploy] noop: HEAD unchanged, "
                    f"skipping {list(units)} restart",
                    file=sys.stderr,
                )

            else:
                # HEAD advanced, no pending marker: restart immediate units now.
                for unit in immediate_units:
                    _issue_system_restart(unit)

                # Deferred units: restart now if queue idle, else write marker.
                inflight = _count_inflight_fixers()
                if inflight == 0:
                    for unit in deferred_units:
                        _issue_system_restart(unit)
                else:
                    first_deferred_at = _now_iso()
                    _write_restart_pending(repo, deferred_units, first_deferred_at)
                    print(
                        f"[post-land-deploy] deferring {list(deferred_units)} restart: "
                        f"{inflight} fixers in-flight "
                        f"(deferred 0s / {RESTART_DEFER_MAX_S}s)",
                        file=sys.stderr,
                    )

        else:
            # No deferred units (e.g., synapse): simple HEAD-advance gate.
            if not head_advanced:
                print(
                    f"[post-land-deploy] noop: HEAD unchanged, "
                    f"skipping {list(units)} restart",
                    file=sys.stderr,
                )
            else:
                for unit in immediate_units:
                    _issue_system_restart(unit)

    user_units = _POST_LAND_RESTART_USER.get(repo)
    if user_units:
        if not head_advanced:
            # HEAD unchanged: skip user-unit restarts.
            # Noop already logged in system units block above (if units non-empty).
            if not units:
                print(
                    f"[post-land-deploy] noop: HEAD unchanged, "
                    f"skipping {list(user_units)} restart",
                    file=sys.stderr,
                )
        else:
            xdg = os.environ.get("XDG_RUNTIME_DIR")
            if not xdg:
                print(
                    f"[post-land-deploy] XDG_RUNTIME_DIR unset — cannot reach user bus, "
                    f"skipping user-unit restarts: {list(user_units)}",
                    file=sys.stderr,
                )
            else:
                for unit in user_units:
                    try:
                        result = subprocess.run(
                            ["systemctl", "--user", "restart", unit],
                            capture_output=True, text=True, timeout=60,
                        )
                        if result.returncode != 0:
                            print(
                                f"[post-land-deploy] user-unit restart {unit} failed "
                                f"rc={result.returncode}: {result.stderr.strip()[:200]}",
                                file=sys.stderr,
                            )
                    except (subprocess.TimeoutExpired, OSError) as e:
                        print(
                            f"[post-land-deploy] user-unit restart {unit} errored: {e}",
                            file=sys.stderr,
                        )
                        continue
                    # Post-restart liveness (best-effort). Runs whether restart rc was 0 or non-0.
                    try:
                        active = subprocess.run(
                            ["systemctl", "--user", "is-active", unit],
                            capture_output=True, text=True, timeout=10,
                        )
                        if active.stdout.strip() != "active":
                            print(
                                f"[post-land-deploy] user unit {unit} not active after restart "
                                f"(state={active.stdout.strip()!r})",
                                file=sys.stderr,
                            )
                    except (subprocess.TimeoutExpired, OSError):
                        pass


def _read_fixer_meta(spec_id: str) -> dict | None:
    """Read the {spec_id}-meta.json sidecar written by _runner.py for fixers."""
    p = SHAPED_DIR / f"{spec_id}-meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _consume_fixer_meta(spec_id: str) -> None:
    """Delete the meta sidecar after pm_core has consumed it. Best-effort."""
    p = SHAPED_DIR / f"{spec_id}-meta.json"
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def _read_fixer_verdict(spec_id: str) -> dict | None:
    """Read the {spec_id}-verdict.json sidecar copied from worktree by shaped_runner.

    Returns the parsed dict if present and valid JSON, None otherwise.
    Logs a WARN if the file exists but is unreadable or not a dict.
    """
    import sys
    p = SHAPED_DIR / f"{spec_id}-verdict.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            print(f"WARN: verdict sidecar {spec_id} is not a dict: {type(data).__name__}",
                  file=sys.stderr)
            return None
        return data
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: verdict sidecar {spec_id} unreadable: {exc}", file=sys.stderr)
        return None


def _consume_fixer_verdict(spec_id: str) -> None:
    """Delete the verdict sidecar after pm_core has consumed it. Best-effort."""
    p = SHAPED_DIR / f"{spec_id}-verdict.json"
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def _validate_already_satisfied_pr(repo: str, pr_num: int) -> bool:
    """Return True if pr_num exists in repo and is merged+closed in Forgejo."""
    if not repo or not _forgejo_get_pr:
        return False
    try:
        repo_name, owner = _repo_owner(repo)
        pr_data = _forgejo_get_pr(repo_name, pr_num, owner=owner)
        return bool(pr_data.get("merged") and pr_data.get("state") == "closed")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _mem() -> MemoryStore:
    return node_identity.writable_store()


def _check_tick_stalls(store, now: datetime) -> int:
    """D6 (attestation-contract-v0 leg 2): page HIGH once per ACTIVE bound
    target whose pm/cursor/<tid> watermark is aged past
    TICK_COVERAGE_STALL_S (the dedicated stall checker's cursor check, run
    from the tick pass — the `lapis-pm-stall-check` unit runs the same
    checks from `stall_check.py` on an independent 10-minute cadence).

    REV 2 pins (spec D6):
      * PAUSED targets are EXCLUDED by design — a paused target is
        intentionally not being ticked (tick() returns noop:paused before
        the cursor write), and the first tick after un-pause advances its
        cursor. Paging them would make the checker pure noise.
      * a target with NO cursor record uses its bound ts (the spec:bound
        comment ts) as the watermark (get_cursor returns None for those);
      * while the Forgejo consecutive-fail counter is at or above the page
        threshold (FORGEJO_UNREACHABLE_THRESHOLD) the checker skips paging —
        tick_all early-returns skipped:forgejo_unreachable for every target
        WITHOUT advancing any cursor, so after a Forgejo outage every
        cursor ages past the threshold and paging all of them would
        duplicate the existing Forgejo 3-strike page (I5).

    I5 page hygiene: one HIGH page per target per episode — dedup mem key
    pm/tick-stall/<tid> stamped with the last-paged ts; a cursor advance
    clears it. Returns the number of pages emitted (0 when suppressed).
    """
    pages = 0
    fails = _get_forgejo_consecutive_fails()
    if fails >= FORGEJO_UNREACHABLE_THRESHOLD:
        return 0
    for t in store.load_all():
        if not t.pm_bound or t.paused:
            continue
        tid = t.id
        cursor = get_cursor(tid)
        watermark = cursor if cursor else _spec_bound_ts(tid)
        if not watermark or watermark == "9999-99-99":
            continue
        try:
            wm = datetime.fromisoformat(watermark)
        except ValueError:
            continue
        if wm.tzinfo is None:
            wm = wm.replace(tzinfo=PACIFIC)
        age = (now - wm).total_seconds()
        if age < TICK_COVERAGE_STALL_S:
            continue
        # Episode dedup (I5): a cursor advance clears the episode; while the
        # stall persists, no second page.
        dedup = _mem().get(_tick_stall_key(tid))
        if dedup:
            try:
                last_paged = datetime.fromisoformat(dedup["content"])
            except (ValueError, KeyError, TypeError):
                last_paged = None
            if last_paged is not None and last_paged > wm:
                continue
        last_decision = _mem().get(_last_decision_key(tid))
        decision_line = (
            f" last-decision={last_decision['content']}"
            if last_decision and last_decision.get("content")
            else ""
        )
        try:
            from agents_core.notify import send_notification, Priority as _P
            send_notification(
                f"Target {tid} has not been ticked in "
                f"{int(age // 60)} min (cursor {watermark}{decision_line}). "
                f"Resume with: cd /srv/git/lapis-pm && "
                f"nohup python3 -m lapis_pm.cli tick --target {tid} "
                f">/dev/null 2>&1 &",
                title=f"lapis-pm: tick stall: {tid}",
                priority=_P.HIGH,
                source="lapis-pm-stall-check",
            )
            _mem().set(_tick_stall_key(tid), _now_iso(),
                       tags=["lapis-pm", "tick-stall"])
            pages += 1
        except Exception as e:
            logger.warning("[tick-stall] page failed for %s: %s", tid, e)
    return pages


def _check_directive_stalls(store, now: datetime) -> int:
    """D7 (attestation-contract-v0 leg 2): page HIGH once per directive
    episode with no outcome within DIRECTIVE_OUTCOME_STALL_S.

    The directive branch by design emits a brief and never dispatches
    (Standing ratification 1 — this detector does NOT make a directive
    dispatch work; it only attests the outcome). For each
    `pm:directive-seen` observation older than the threshold, an outcome
    holds when ANY of (a) a dispatch was recorded after the directive ts
    (timezone-aware comparison), (b) the open PR head advanced past the
    baseline recorded on the observation (`pm:pr-head=<sha>`; `none` +
    any open PR counts), or (c) the directive's OWN brief was consumed —
    REV 4 pin: outcome (c) is keyed on the specific brief id recorded on
    the observation (`pm:brief-id=<cid>`, `none` when the raise failed);
    an EMPTY outstanding-brief slot does NOT count (the over-broad
    `get_outstanding_brief is None -> True` reading suppressed the page
    for any target with no outstanding brief — the common steady state —
    defeating the detector).

    The detector is Forgejo-free (mem + episodic reads only — the current
    PR head comes from the pm/pr-head/<tid> record the tick's PR-perception
    site writes). PAUSED targets are excluded (parity with D6: a paused
    target is intentionally not being ticked, and the first tick after
    un-pause re-evaluates the directive). I5: one page per directive
    episode (dedup key pm/directive-stall/<tid>/<directive-id>). Returns
    the number of pages.
    """
    pages = 0
    for t in store.load_all():
        if not t.pm_bound or t.paused:
            continue
        tid = t.id
        for obs in _directive_seen_observations(tid):
            c = obs["comment"]
            try:
                obs_dt = datetime.fromisoformat(c.ts)
            except ValueError:
                continue
            if obs_dt.tzinfo is None:
                obs_dt = obs_dt.replace(tzinfo=PACIFIC)
            if now.tzinfo is None:
                # Defensive: `now` is tz-aware by construction (PACIFIC);
                # a naive `now` would make the age comparison raise
                # TypeError, so skip rather than mis-compare.
                continue
            age = (now - obs_dt).total_seconds()
            if age < DIRECTIVE_OUTCOME_STALL_S:
                continue
            if _directive_outcome_holds(
                tid, c.ts, obs["pr_head_baseline"],
                brief_id=obs["brief_id"],
            ):
                continue
            dedup_key = _directive_stall_key(tid, obs["directive_id"])
            if _mem().get(dedup_key):
                continue
            snippet = c.content.replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            try:
                from agents_core.notify import send_notification, Priority as _P
                send_notification(
                    f"Directive from {c.author} at {c.ts} on {tid} has no "
                    f"outcome for {int(age // 60)} min: {snippet}\n"
                    f"Force-dispatch: cd /srv/git/lapis-pm && "
                    f"python3 -m lapis_pm.cli tick --target {tid} "
                    f"--force-dispatch fixer:address the pending directive",
                    title=f"lapis-pm: directive outcome stall: {tid}",
                    priority=_P.HIGH,
                    source="lapis-pm-directive-stall",
                )
                _mem().set(dedup_key, _now_iso(),
                           tags=["lapis-pm", "directive-stall"])
                pages += 1
            except Exception as e:
                logger.warning(
                    "[directive-stall] page failed for %s: %s", tid, e
                )
    return pages


# ---------------------------------------------------------------------------
# Autonomy actuator hooks (lapis-pm-autonomy-actuator-v0, U1/U2/U3 wiring)
#
# Single dial, ships in shadow (Design 1): PM_AUTONOMY_ACTUATOR=shadow (default)
# | enforce. Unknown values fail-safe to shadow (same convention as
# PM_AUTO_RESOLVE, auto_resolve.py:19-25). Shadow: the hooks compute and record
# but never act. Enforce: U1 flips pm_verification from an observed pipeline
# run; U2 merges on a verified precedent match. Every hook is never-raise:
# a failed act attempt degrades to today's behavior (the brief is still
# emitted), never to a skipped brief.
# ---------------------------------------------------------------------------

_AUTONOMY_ACTUATOR_DIAL_ENV = "PM_AUTONOMY_ACTUATOR"
_AUTONOMY_ACTUATOR_KEY = "pm/autonomy-actuator/last-run"


def _autonomy_dial() -> str:
    """Read PM_AUTONOMY_ACTUATOR. Unknown values fail-safe to 'shadow'."""
    val = os.environ.get(_AUTONOMY_ACTUATOR_DIAL_ENV, "shadow").lower().strip()
    if val in ("shadow", "enforce"):
        return val
    logger.warning(
        "%s=%r is unknown; treating as 'shadow'",
        _AUTONOMY_ACTUATOR_DIAL_ENV, val,
    )
    return "shadow"


_AUTONOMY_INFLIGHT = False


def _autonomy_inflight() -> bool:
    """True while an autonomy op (attestation or backstop sweep) is in flight.

    Single shared concurrency guard (Design 9 / H2): the tick path never stacks
    two 900s autonomy ops — an in-flight op makes the other skip and retry next
    tick. Process-local (each timer-fired tick is a fresh process; a crashed
    op clears naturally on the next process start).
    """
    return _AUTONOMY_INFLIGHT


def _autonomy_set_inflight(value: bool) -> None:
    global _AUTONOMY_INFLIGHT
    _AUTONOMY_INFLIGHT = bool(value)


def _autonomy_op_daily_gated(key_suffix: str, window_s: float = 86400.0) -> bool:
    """Once-a-day gate on a mem key (same 'once a day, out of tick' pattern as
    hold_shadow_summary). Returns True when the op may run this tick."""
    key = f"{_AUTONOMY_ACTUATOR_KEY}/{key_suffix}"
    try:
        rec = _mem().get(key)
        if rec:
            last = datetime.fromisoformat(str(rec.get("content", "")).replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - last).total_seconds() < window_s:
                return False
    except Exception:
        pass  # fail-open: a read error lets the op run (it re-records the stamp)
    return True


def _autonomy_stamp(key_suffix: str) -> None:
    key = f"{_AUTONOMY_ACTUATOR_KEY}/{key_suffix}"
    try:
        _mem().set(key, _now_iso(), tags=["lapis-pm", "pm:autonomy-actuator"])
    except Exception as e:
        logger.warning("autonomy-actuator: stamp write failed for %s: %s", key, e)


def _attestation_hook(target_id: str, cls: authority.PRClassification,
                      payload: dict) -> None:
    """U1 hook: pipeline-attested machine verification (Design 2, U1.2).

    Fires only for non-hold advisory-clean targets whose pm_verification is
    still 'pm-live-test'. Runs (or reuses the head-SHA cache of) the
    two-worktree failure-set diff and, in enforce mode only, flips
    pm_verification to 'machine' on an attested result. unattested /
    inconclusive NEVER flip the field (fail-safe to the human brief in both
    dial modes). Never-raise.
    """
    try:
        dial = _autonomy_dial()
        from . import verification_attestation as _va

        # Fresh store read at the hook site (the hook precedes the shared read
        # by construction — H1: the shared read must reflect a same-tick grant).
        t = TargetStore().get(target_id)
        if t is None:
            return
        if t.data.get("pm_verification", "pm-live-test") != "pm-live-test":
            return  # already machine (or a non-default value) — nothing to do

        pr = payload.get("pr") or {}
        head_sha = (pr.get("head") or {}).get("sha", "")
        if not head_sha:
            return
        pr_number = cls.pr_number
        repo = cls.repo or ""
        if not repo:
            return

        # Derive the PR head branch slug from the payload's PR head ref
        # (e.g. "lapis/<tid>/local" -> slug "local"). attest() builds the
        # head branch as "lapis/<tid>/<slug>"; the default slug ("forced")
        # does not exist for real PR branches, so without the correct slug
        # worktree creation would always fail -> always inconclusive.
        # If the ref is missing or does not match the expected prefix, do
        # NOT call attest() at all — inconclusive by omission is fail-safe;
        # never call attest with a wrong slug.
        head_ref = (pr.get("head") or {}).get("ref", "")
        prefix = f"lapis/{target_id}/"
        if not head_ref or not head_ref.startswith(prefix):
            return
        slug = head_ref[len(prefix):]
        if not slug:
            return

        # One run per head SHA: reuse the cached result while the head is
        # unchanged (a force-push re-runs it).
        mem = _mem()
        cached = _va.read_cache(mem, target_id, pr_number, head_sha)
        if cached is None:
            if _autonomy_inflight():
                return  # shared concurrency guard — retry next tick
            _autonomy_set_inflight(True)
            try:
                att = _va.attest(
                    repo, pr_number, head_sha,
                    changed_paths=list(getattr(cls, "changed_paths", None) or []),
                    target_id=target_id,
                    slug=slug,
                )
            finally:
                _autonomy_set_inflight(False)
            _va.write_cache(mem, target_id, pr_number, head_sha, att,
                            shadow=(dial != "enforce"))
        else:
            att = cached

        # Pre-existing failures: recorded as finding/ keys per the skill's
        # recipe (deduped per (repo, test, day)).
        preexisting = att.get("preexisting") or []
        if preexisting:
            try:
                _va.record_preexisting(mem, repo, preexisting)
            except Exception as e:
                logger.warning("attestation: preexisting finding write failed: %s", e)

        if att.get("result") != "attested":
            # unattested / inconclusive — never flip the field (fail-safe).
            episodic.write_observation(
                target_id,
                f"attestation: result={att.get('result')} "
                f"(reason={att.get('reason') or ('pr_failures' if att.get('pr_failures') else 'n/a')}); "
                f"pm_verification unchanged — human brief as today",
                extra_tags=[
                    "pm:attestation", f"pm:attestation-result={att.get('result')}",
                    f"pm:pr={pr_number}", "pm:dial=" + dial,
                ],
            )
            return

        if dial != "enforce":
            episodic.write_observation(
                target_id,
                f"attestation (shadow): WOULD grant pm_verification=machine "
                f"(head={head_sha[:12]}, suite={att.get('suite', '?')}) — "
                f"field untouched in shadow mode",
                extra_tags=[
                    "pm:attestation", "pm:attestation-would-grant",
                    f"pm:pr={pr_number}", "pm:dial=shadow",
                ],
            )
            return

        # Enforce + attested: flip pm_verification from the OBSERVED run.
        t.data["pm_verification"] = "machine"
        t.save()
        episodic.write_observation(
            target_id,
            f"attestation (enforce): pm_verification flipped to machine from "
            f"observed pipeline run (head={head_sha[:12]}, suite={att.get('suite', '?')})",
            extra_tags=[
                "pm:attestation", "pm:attestation-granted",
                f"pm:pr={pr_number}", "pm:dial=enforce",
            ],
        )
    except Exception as e:
        logger.warning(
            "attestation hook failed for %s PR #%s (%s) — continuing to brief",
            target_id, cls.pr_number, e,
        )


def _precedent_hook(target_id: str, cls: authority.PRClassification,
                    payload: dict) -> str | None:
    """U2 hook: precedent actuator (Design 3/4/5, U2.5).

    Fires ONLY for effective_trigger == 'advisory-clean' and not hold (the
    merge-safe state; screen-issue and fixable-low forks keep the human in
    v0). Enforce mode: requires the fork class to be promoted AND a verified
    precedent match (Design 5 lineage chain), then merges via the proven
    brief._act_merge_pr path + writes the resolution record + returns the
    action string. Shadow mode: same derivation, records the would-be call to
    /srv/lapis/autonomy-shadow/precedent-calls.jsonl, never merges, returns None
    (the brief is still emitted). On ANY merge exception it returns None —
    fall through to the normal brief path, never swallow a brief.
    """
    try:
        dial = _autonomy_dial()
        from . import precedent as _pc

        fork = _pc.derive_fork_class(cls, "clean", repo=cls.repo)
        if fork.get("held_paths_class") != "none":
            _pc.record_shadow_call(
                target_id, cls.repo or "", cls.pr_number, fork,
                would_merge=False,
                would_merge_reason="held_paths_class!=none (unconditional human gate)",
                matched_precedent=None, no_match_reason="held_paths", dial=dial,
            )
            return None
        # Merge-safe state: reviewer clean, no issues, LOC within MAX_AUTO_LOC.
        if cls.issues:
            _pc.record_shadow_call(
                target_id, cls.repo or "", cls.pr_number, fork,
                would_merge=False,
                would_merge_reason="issues present (not merge-safe)",
                matched_precedent=None, no_match_reason="issues", dial=dial,
            )
            return None
        if int(getattr(cls, "diff_loc", 0) or 0) > _pc.MAX_AUTO_LOC:
            _pc.record_shadow_call(
                target_id, cls.repo or "", cls.pr_number, fork,
                would_merge=False,
                would_merge_reason=f"loc {getattr(cls, 'diff_loc', 0)} > MAX_AUTO_LOC",
                matched_precedent=None, no_match_reason="loc_bucket", dial=dial,
            )
            return None

        mem = _mem()
        rec = _pc.find_precedent(mem, cls.repo or "", fork)
        skipped = int(getattr(_pc.find_precedent, "last_skipped", 0) or 0)

        if dial != "enforce":
            _pc.record_shadow_call(
                target_id, cls.repo or "", cls.pr_number, fork,
                would_merge=rec is not None,
                would_merge_reason=(
                    "enforce would merge on verified precedent" if rec
                    else "no verified precedent match"
                ),
                matched_precedent=rec.get("_key") if rec else None,
                no_match_reason=None if rec else "no_match",
                dial="shadow", skipped=skipped,
            )
            return None

        if rec is None:
            _pc.record_shadow_call(
                target_id, cls.repo or "", cls.pr_number, fork,
                would_merge=False,
                would_merge_reason="no verified precedent match — brief as today",
                matched_precedent=None, no_match_reason="no_match",
                dial="enforce", skipped=skipped,
            )
            return None

        # Enforce + verified match: the fork class must be promoted (Design 8 —
        # a class nobody has promoted never auto-resolves).
        class_key = _pc.class_record_key(
            cls.repo or "", "clean", fork.get("change_classes", []),
            fork.get("loc_bucket", ""),
        )
        class_rec = None
        try:
            raw = mem.get(class_key)
            if raw:
                class_rec = json.loads(raw.get("content", ""))
        except (ValueError, TypeError):
            class_rec = None
        if not isinstance(class_rec, dict) or not class_rec.get("promoted"):
            _pc.record_shadow_call(
                target_id, cls.repo or "", cls.pr_number, fork,
                would_merge=False,
                would_merge_reason=(
                    "fork class not promoted (manual promotion required)"
                ),
                matched_precedent=rec.get("_key"), no_match_reason="class_not_promoted",
                dial="enforce", skipped=skipped,
            )
            return None

        # Merge via the proven path (Design 4). Any exception (incl. Forgejo 4xx
        # on a not-mergeable PR) falls through to the brief — never swallowed.
        brief._act_merge_pr(target_id, cls.pr_number)
        _pc.write_resolution(
            mem,
            target_id=target_id, pr_number=cls.pr_number, repo=cls.repo or "",
            fork=fork, matched_precedent=rec.get("_key", ""),
        )
        # Class counter increment (Design 8).
        try:
            data = dict(class_rec)
            data["auto_resolved_count"] = int(data.get("auto_resolved_count", 0)) + 1
            mem.set(class_key, json.dumps(data, ensure_ascii=False, sort_keys=True),
                    tags=["lapis-pm", "pm:autonomy-class"])
        except Exception as e:
            logger.warning("precedent hook: class counter update failed: %s", e)
        _pc.record_shadow_call(
            target_id, cls.repo or "", cls.pr_number, fork,
            would_merge=True,
            would_merge_reason="enforce: precedent match + promoted class — merged",
            matched_precedent=rec.get("_key"), no_match_reason=None,
            dial="enforce", skipped=skipped,
        )
        episodic.write_observation(
            target_id,
            f"Precedent-resolved: merged PR #{cls.pr_number} on verified precedent "
            f"{rec.get('_key', '?')} (fork={_pc.fork_hash(fork)}, dial=enforce)",
            extra_tags=["pm:precedent-resolved", f"pm:pr={cls.pr_number}"],
        )
        return f"action:precedent_resolved:fork={_pc.fork_hash(fork)}:pr={cls.pr_number}"
    except Exception as e:
        # Merge exception or hook failure: fall through to the normal brief
        # path — never swallow a brief (U2.6).
        logger.warning(
            "precedent hook failed for %s PR #%s (%s) — falling to brief",
            target_id, cls.pr_number, e,
        )
        return None


def _backstop_sweep_tick() -> None:
    """U3 tick-path invocation — D4 (lapis-pm-reviewer-leg-repair-v0): the
    backstop sweep is NO LONGER run from the 60s tick.

    The landed actuator's sweep was dead on arrival: a ~900s sweep inside a
    60s `Type=oneshot` tick capped at `TimeoutStartSec=300` was killed
    mid-run (8 consecutive `Failed with result 'timeout'` at exact 5-min
    intervals, 2026-09-07 13:03-13:38), wrote no completion stamp, and
    retried every tick. The sweep now runs on a dedicated
    `lapis-pm-backstop.timer` (15m cycle, `TimeoutStartSec=1200s`) wired to
    the `lapis-pm backstop-sweep` CLI (cli.py cmd_backstop_sweep), which
    calls `_run_backstop_sweep_standalone()` below.

    This function is retained as a no-op so the tick's step 1.5 call site and
    the existing never-raise test keep working; it logs once that the sweep
    moved to the dedicated unit.
    """
    logger.debug(
        "[autonomy-backstop] tick-path sweep is a no-op — the sweep runs on "
        "the dedicated lapis-pm-backstop timer (15m cycle, 1200s timeout); "
        "see _run_backstop_sweep_standalone()"
    )


def _run_backstop_sweep_standalone() -> dict:
    """D4 (lapis-pm-reviewer-leg-repair-v0): the backstop sweep on its
    dedicated 15m cycle (wired to the `lapis-pm backstop-sweep` CLI, which
    the dedicated `lapis-pm-backstop.timer` unit invokes).

    Serialization is enforced by the persistent `sweep_started_ts` stamp
    (written at sweep START, 15m minimum interval) — NOT by the process-local
    in-flight guard (each tick is a fresh process; the dedicated oneshot unit
    is one-at-a-time host-wide). Never-raise, fail-soft to logging.
    """
    try:
        from . import autonomy_backstop as _bs
        outcome = _bs.sweep(_mem())
        logger.info(
            "[autonomy-backstop] sweep: repo=%s status=%s suspect=%s demoted=%s "
            "change_class=%s sweep_count=%s",
            outcome.get("repo"), outcome.get("status"),
            outcome.get("suspect_prs"), outcome.get("demoted_classes"),
            outcome.get("change_class"), outcome.get("sweep_count"),
        )
        if outcome.get("status") == "red":
            try:
                _bs.open_regression_brief(
                    outcome.get("repo") or "?",
                    (outcome.get("suspect_prs") or [None])[0],
                    outcome.get("main_failures") or [],
                )
            except Exception as e:
                logger.warning("backstop regression brief failed: %s", e)
        return outcome
    except Exception as e:
        logger.warning("[autonomy-backstop] sweep failed (non-fatal): %s", e)
        return {"repo": None, "status": "error", "error": str(e),
                "suspect_prs": [], "demoted_classes": [], "findings": []}


def _ensure_dispatch_owned(repo: str) -> None:
    """Gate a fixer/reviewer dispatch on owned_queue_root (Design §3d, I4)."""
    paths = [str(SHAPED_DIR), str(CLAUDE_QUEUE_DIR)]
    if repo:
        paths.append(Shaper.resolve_repo_cwd(repo))
    node_identity.ensure_dispatch_target_owned(*paths)


def _cursor_key(target_id: str) -> str:
    return f"pm/cursor/{target_id}"


def _last_decision_key(target_id: str) -> str:
    """D6 (attestation-contract-v0 leg 2): the tick-exit decision tag for a
    target, persisted so the dedicated stall checker's page can carry it
    (line omitted when absent). Written at the tick-exit result-encoding
    site (tick()'s final return), read by stall_check.py."""
    return f"pm/last-decision/{target_id}"


def _pr_head_key(target_id: str) -> str:
    """D7 (attestation-contract-v0 leg 2): the per-target PR-state record
    the tick's PR-perception site writes (one write site, no extra Forgejo
    calls from the directive-outcome detector). JSON list of
    {"number", "head_sha"} for the target's open PRs."""
    return f"pm/pr-head/{target_id}"


def _tick_stall_key(target_id: str) -> str:
    """D6 episode-dedup key (I5): stamped with the last-paged ts; a cursor
    advance (new cursor ts > last-paged ts) clears the episode."""
    return f"pm/tick-stall/{target_id}"


def _directive_stall_key(target_id: str, directive_id: str) -> str:
    """D7 episode-dedup key (I5): one page per directive episode."""
    return f"pm/directive-stall/{target_id}/{directive_id or 'unknown'}"


def _dispatched_key(target_id: str) -> str:
    return f"pm/dispatched/{target_id}"


def _pause_key(target_id: str) -> str:
    return f"pm/pause-state/{target_id}"


def _brief_key(target_id: str) -> str:
    return f"pm/outstanding-brief/{target_id}"


def _classified_prs_key(target_id: str) -> str:
    return f"pm/classified-prs/{target_id}"


def _get_synth_fail_count(target_id: str) -> int:
    """Get the current synthesis-failure counter for target."""
    v = _mem().get(_SYNTH_FAIL_KEY.format(target_id))
    return int(v["content"]) if v else 0


def _inc_synth_fail_count(target_id: str) -> int:
    """Increment the synthesis-failure counter, return new count."""
    n = _get_synth_fail_count(target_id) + 1
    _mem().set(_SYNTH_FAIL_KEY.format(target_id), str(n), tags=["lapis-pm", "synth-fail"])
    return n


def _reset_synth_fail_count(target_id: str) -> None:
    """Reset the synthesis-failure counter to 0."""
    _mem().delete(_SYNTH_FAIL_KEY.format(target_id))


def _check_calcification(target_id: str) -> None:
    """Track consecutive dispatches with unresolved intent voids.

    Emits an advisory observation every CALCIFICATION_THRESHOLD dispatches
    (3, 6, 9, ...) where the intent artifact for target_id still has VOIDs.
    Resets when the artifact is absent or void-free.
    """
    content = _intent_artifact.load(target_id)
    key = f"{_intent_artifact.VOID_DISPATCH_KEY_PREFIX}/{target_id}"
    mem = _mem()
    if not content or not _intent_artifact.has_voids(content):
        mem.delete(key)
        return
    rec = mem.get(key)
    count = 0
    if rec:
        try:
            count = int(rec.get("content", "0"))
        except (ValueError, TypeError):
            count = 0
    count += 1
    mem.set(key, str(count), tags=["lapis-pm", "intent-void"])
    if count % _intent_artifact.CALCIFICATION_THRESHOLD == 0:
        n_voids = _intent_artifact.void_field_count(content)
        episodic.write_observation(
            target_id,
            (
                f"[intent-calcification-advisory] {n_voids} VOID field(s) in the intent "
                f"artifact for {target_id} have persisted across {count} consecutive worker "
                f"dispatches. These voids are machine-un-fillable. Erah should resolve them "
                f"or explicitly acknowledge them to prevent silent calcification into assumed intent."
            ),
            extra_tags=["pm:intent-void", "pm:calcification-advisory"],
        )


REVIEW_STATE_KEY_PREFIX = "pm/review-state/"


def _review_state_key(target_id: str) -> str:
    return f"{REVIEW_STATE_KEY_PREFIX}{target_id}"


# ---------------------------------------------------------------------------
# Corroboration follow-up pass helpers (cross-node-corroboration-v0, PR 3)
# ---------------------------------------------------------------------------

def _diff_text_for_corr(repo: str, pr_num: int | str) -> str:
    """Fetch PR diff for the corroboration pass. Returns '' on any failure."""
    try:
        from agents_core.forgejo import get_pr_diff as _get_diff
        return _get_diff(repo, int(pr_num))
    except Exception:
        return ""


def _run_corroboration_pass_sync(diff_text: str, repo: str) -> dict:
    """Run corroboration follow-up pass synchronously.

    Returns an uncertain-shaped dict on any error so the caller never
    needs to handle exceptions (Compost invariant: all outputs are nutrients).
    This function is a named wrapper so tests can patch it directly.
    """
    try:
        from .corroboration_adapter import run_corroboration_pass
        return run_corroboration_pass(diff_text, repo)
    except Exception as exc:
        return {
            "verdict": "uncertain",
            "claim": "",
            "citations": [],
            "freshness_stamp": _now_iso(),
            "scope_id": f"repo:{repo}",
            "drift_class": None,
            "notes": f"corroboration pass unavailable: {type(exc).__name__}",
            "primitive_decomposition": None,
            "leg_status": "pass_failed",
        }


def _persist_review_state_cache(target_id: str, target, open_prs: list[dict]) -> None:
    """Write or delete pm/review-state/<tid> from current tick state.

    Called once per tick after decide-step. Side-effect only; never raises
    (failures logged via episodic.write_observation with pm:error tag,
    but never abort the tick — cache is best-effort visibility).
    """
    state = _active_review_state(target_id, open_prs)
    key = _review_state_key(target_id)
    if state is None:
        # No active review loop — delete stale cache entry if present.
        _mem().delete(key)
        return

    authority_level = target.pm_authority or "advisory"
    budget = _REVIEW_CYCLE_BUDGETS.get(authority_level, 2)
    mode = _REVIEWER_MODES.get(authority_level, "same-reviewer")
    verdict = state.get("verdict")
    has_real_verdict = verdict and verdict != "pending"

    # Compact corroboration summary for quick display in claude-view.
    # Full evidence-packet lives in the episodic entry; cache stores verdict+drift only.
    last_corroboration: dict | None = None
    panel_starvation_summary: dict | None = None
    if has_real_verdict:
        pr_number = state.get("pr_number")
        if pr_number is not None:
            # Prefer tick-local cache (populated by _encode_gpu_results this same tick)
            # to avoid re-scanning episodic for data already in hand.
            corr = _tick_corr_cache.get((target_id, pr_number))
            verdict_info = None
            if corr is None:
                verdict_info = _last_review_verdict(target_id, pr_number)
                if verdict_info:
                    corr = verdict_info.get("corroboration_result")
            if corr:
                last_corroboration = {
                    "verdict": corr.get("verdict"),
                    "drift_class": corr.get("drift_class"),
                }
                # DoD-1 (lapis-pm-reviewer-leg-repair-v0): the mem cache
                # carries the node2 error-class tail (the four named classes)
                # so a genuine node2 down is distinguishable from a wiring
                # bug on the pm/review-state surface (today only the episodic
                # JSON did).
                _n2_notes = (corr.get("node2_corroboration") or {}).get("notes") or ""
                _n2_class = None
                for _ec in (
                    "node2_client_import_error", "node2_aci_unverified",
                    "node2_cap_refused", "node2_unavailable",
                ):
                    if _ec in _n2_notes:
                        _n2_class = _ec
                        break
                if _n2_class:
                    last_corroboration["node2_error_class"] = _n2_class
            # Leg 1 legibility (DoD #4): surface starvation in the same
            # compact cache claude-view already reads, without opening the
            # verdict JSON. Additive — absent on verdicts that predate this.
            if verdict_info is None:
                verdict_info = _last_review_verdict(target_id, pr_number)
            if verdict_info:
                _pstarv = verdict_info.get("panel_starvation")
                if _pstarv:
                    panel_starvation_summary = {
                        "starved": _pstarv.get("starved", False),
                        "legs_down": _pstarv.get("legs_down", []),
                        "confidence_raw": _pstarv.get("confidence_raw"),
                        "confidence_attenuated": verdict_info.get("confidence"),
                    }

    payload = {
        "pr_number": state["pr_number"],
        "cycle": state["cycle"],
        "budget": budget,
        "mode": mode,
        "last_verdict": verdict if has_real_verdict else None,
        "last_issues": state.get("issues") if (has_real_verdict and verdict == "fixable") else None,
        "updated_at": _now_iso(),
        "last_corroboration": last_corroboration,
        "panel_starvation": panel_starvation_summary,
    }
    _mem().set(key, json.dumps(payload),
               tags=["lapis-pm", "review-state"])


def _now_iso() -> str:
    """Microsecond-precision so it interleaves cleanly with comment timestamps."""
    return datetime.now(PACIFIC).isoformat(timespec="microseconds")


def _persist_pr_head_record(target_id: str, open_prs: list[dict]) -> None:
    """D7 (attestation-contract-v0 leg 2): write pm/pr-head/<tid> from the
    tick's open-PR perception.

    JSON list of {"number", "head_sha"} for each open PR (lowest number
    first — the same PR the decide phase drives). The directive-outcome
    detector reads this record and compares the current head against the
    baseline recorded on the directive-seen observation; it makes no
    Forgejo call of its own. PRs with no resolvable number are skipped
    (a None number would break the sort key and poison the record).
    """
    recs = []
    for pr in sorted(
        (p for p in (open_prs or []) if p.get("number") is not None),
        key=lambda p: p.get("number", 1 << 30),
    ):
        recs.append({
            "number": pr.get("number"),
            "head_sha": (pr.get("head") or {}).get("sha") or "",
        })
    _mem().set(_pr_head_key(target_id), json.dumps(recs, ensure_ascii=False),
               tags=["lapis-pm", "pr-head"])


def _dispatch_age(ts: str | None) -> timedelta | None:
    """Parse a dispatch record's ``ts`` (tz-aware Pacific, set at dispatch
    from _now_iso) and return its age, or None if ``ts`` is missing/unparseable.
    """
    if not ts:
        return None
    try:
        dispatched_at = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return datetime.now(PACIFIC) - dispatched_at


def _format_age(age: timedelta) -> str:
    """Render a timedelta as a short human string: '31h', '45m', '3d'."""
    total_s = int(age.total_seconds())
    if total_s < 0:
        total_s = 0
    days, rem = divmod(total_s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m"


def _dispatch_age_note(ts: str | None) -> str:
    """' pending for 31h' (Leg 5) — or '' if age can't be established."""
    age = _dispatch_age(ts)
    if age is None:
        return ""
    return f", pending for {_format_age(age)}"


# --- Project-slot blackboard: close the slot + emit attribution deposit ------
# Mirrors the *_DEPOSIT_RECORDER dynamic-import convention used by the weaver/mem
# servers: lapis-pm never statically imports zephyr — the recorder is resolved from
# an env spec "<module>:<callable>" so the attribution substrate stays decoupled.
_SLOT_DEPOSIT_RECORDER_SPEC = os.environ.get(
    "LAPIS_PM_DEPOSIT_RECORDER", "zephyr.attribution:get_recorder"
)
# PM dispatch terminal status -> slot lifecycle status.
_SLOT_STATUS_MAP = {"processed": "landed", "failed": "abandoned"}


def _close_slot_and_deposit(rec: dict, project_id: str) -> None:
    """Close the project-slot for a completed dispatch and emit its attribution
    deposit. Best-effort: never raises (mirrors router_portfolio's emit discipline).

    The slot_id is the dispatch spec_id; the contributor-of-record is the task_id
    (gpu_id). This is the seam that finally lands real (non-smoke) PM-origin rows in
    the Zephyr attribution log — every shaped slot that completes deposits provenance.
    The daemon runs on the BRIX master, so the slot transition is a local master write.
    """
    slot_id = rec.get("spec_id")
    task_id = rec.get("gpu_id")
    if not slot_id or not task_id:
        return  # legacy dispatch record without a slot; nothing to close
    slot_status = _SLOT_STATUS_MAP.get(rec.get("status"))
    if slot_status is None:
        return
    completed_at = rec.get("completed_at") or _now_iso()
    agent_type = rec.get("agent_type", "unknown")

    # 1) transition the slot (contributor-of-record write)
    try:
        from agents_core.slots import SlotStore

        SlotStore().update_status(slot_id, slot_status, by=task_id)
    except Exception as exc:  # never block result encoding
        logger.warning("[slots:close-failed] %s: %s", slot_id, exc)

    # 2) emit the attribution deposit (Zephyr deposit log)
    try:
        module_name, _, attr = _SLOT_DEPOSIT_RECORDER_SPEC.partition(":")
        recorder = getattr(importlib.import_module(module_name), attr)()
        manifest = "sha256:" + hashlib.sha256(
            f"slot:{slot_id}:{slot_status}:{completed_at}".encode("utf-8")
        ).hexdigest()
        # Source model from the shaper registry (dispatch records never store "model").
        _dep_model = None
        try:
            _dep_model = _SHAPER.get_agent(agent_type).model
        except Exception:
            pass
        prov = {
            "manifest_hash": manifest,
            "agent_id": task_id,
            "tool": f"lapis-pm:{agent_type}",
            "model": _dep_model,
            "timestamp": completed_at,
            "schema_version": "lapis-provenance-v0",
            "slot": {
                "slot_id": slot_id,
                "project_id": project_id,
                "status": slot_status,
                "agent_type": agent_type,
                "intent": rec.get("intent"),
                "pr_number": rec.get("pr_number"),
            },
        }
        recorder.record(prov, store_kind="slot", key=slot_id)
    except Exception as exc:  # never block result encoding
        logger.warning("[slots:deposit-failed] %s: %s", slot_id, exc)


def get_cursor(target_id: str) -> str | None:
    rec = _mem().get(_cursor_key(target_id))
    return rec["content"] if rec else None


def set_cursor(target_id: str, ts: str):
    _mem().set(_cursor_key(target_id), ts, tags=["lapis-pm", "cursor"])


def load_dispatched(target_id: str) -> list[dict]:
    rec = _mem().get(_dispatched_key(target_id))
    if not rec:
        return []
    try:
        data = json.loads(rec["content"])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_dispatched(target_id: str, records: list[dict]):
    _mem().set(_dispatched_key(target_id),
               json.dumps(records, ensure_ascii=False),
               tags=["lapis-pm", "dispatched"])


def append_dispatched(target_id: str, record: dict):
    records = load_dispatched(target_id)
    records.append(record)
    save_dispatched(target_id, records)


def compact_dispatched(target_id: str) -> None:
    """Replace the full dispatch record list for target_id with a compact stub.

    Stub shape (single-element list to preserve the list[dict] contract):
      [{"compacted": True, "compacted_at": "<iso>", "dispatch_count": N,
        "agent_types": [...], "first_dispatch": "<ts>",
        "last_completed": "<ts>"}]   # last_completed omitted if no record has it

    Idempotent: if the first element already has "compacted": True, returns
    immediately without writing. Does not touch pm/landed/<tid>.
    """
    records = load_dispatched(target_id)
    if not records:
        return
    if records[0].get("compacted"):
        return

    agent_types = [r.get("agent_type", "") for r in records]
    first_dispatch = records[0].get("ts", "")

    last_completed: str | None = None
    for r in records:
        ca = r.get("completed_at")
        if ca:
            if last_completed is None or ca > last_completed:
                last_completed = ca

    stub: dict = {
        "compacted": True,
        "compacted_at": _now_iso(),
        "dispatch_count": len(records),
        "agent_types": agent_types,
        "first_dispatch": first_dispatch,
    }
    if last_completed is not None:
        stub["last_completed"] = last_completed

    save_dispatched(target_id, [stub])


def compact_eligible_targets(min_age_days: int = 30) -> list[tuple[str, str]]:
    """Find all targets eligible for compaction and compact their dispatched records.

    Eligibility: pm/landed/<tid> exists AND landed_at (or ts for manual-land)
    is older than min_age_days days AND pm/dispatched/<tid> exists AND is not
    already compacted.

    Returns list of (target_id, outcome) pairs where outcome is one of:
      "compacted" | "already_compacted" | "no_dispatched_key"
    """
    from datetime import timezone, timedelta

    mem = _mem()
    cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)

    landed_entries = mem.list_by_prefix("pm/landed/", limit=500)
    results: list[tuple[str, str]] = []

    for entry in landed_entries:
        key = entry["key"]
        target_id = key[len("pm/landed/"):]

        try:
            landed_rec = json.loads(entry["content"])
        except (json.JSONDecodeError, TypeError):
            continue

        age_ts_str = landed_rec.get("landed_at") or landed_rec.get("ts")
        if not age_ts_str:
            continue  # unparseable age - skip rather than compact prematurely

        try:
            age_dt = datetime.fromisoformat(age_ts_str)
            if age_dt.tzinfo is None:
                from datetime import timezone as _tz
                age_dt = age_dt.replace(tzinfo=_tz.utc)
        except (ValueError, AttributeError):
            continue  # unparseable age - skip

        if age_dt > cutoff:
            continue  # too recent

        dispatched_rec = mem.get(_dispatched_key(target_id))
        if not dispatched_rec:
            results.append((target_id, "no_dispatched_key"))
            continue

        try:
            dispatched = json.loads(dispatched_rec["content"])
        except (json.JSONDecodeError, TypeError):
            dispatched = []

        if dispatched and dispatched[0].get("compacted"):
            results.append((target_id, "already_compacted"))
            continue

        compact_dispatched(target_id)
        results.append((target_id, "compacted"))

    return results


def force_dispatch(
    target_id: str,
    agent_type: str,
    intent: str,
    *,
    extra_episodic_tags: list[str] | None = None,
    episodic_message: str | None = None,
) -> str:
    """Dispatch a shaped agent, record it, and emit a router-portfolio event.

    Extracted from cmd_tick's force-dispatch block so that both the CLI and
    _act_force_dispatch_retry can call the same path.  Returns task_id.

    ``extra_episodic_tags`` and ``episodic_message`` let chain-mode callers
    (lapis_pm/cli.py's cmd_bind_chain, lapis_pm/chain.py's
    _fire_initial_dispatch) fold their chain-specific episodic write into
    this call's single ``episodic.write_dispatch`` instead of writing a
    second, competing entry. Non-chain callers get today's generic
    behavior unchanged.
    """
    target = TargetStore().get(target_id)
    if target is None:
        raise ValueError(f"target not found: {target_id}")
    # Guard: initial fixer must not fire for adopted targets.
    # The adopted PR already exists; the first action should be a reviewer dispatch.
    _adopted_pr_num = target.data.get("adopted_pr_number")
    if agent_type in _INITIAL_FIXER_TYPES and isinstance(_adopted_pr_num, int):
        existing_dispatches = load_dispatched(target_id)
        if not any(r.get("agent_type") in _INITIAL_FIXER_TYPES for r in existing_dispatches):
            raise ValueError(
                f"target {target_id!r} has an adopted PR "
                f"(#{_adopted_pr_num}) — "
                "dispatch a reviewer instead of the initial fixer"
            )

    # L1.D1: Extend initial-fixer guard to check for open lapis/<tid>/ PRs
    if agent_type in _INITIAL_FIXER_TYPES and target.pm_repo:
        try:
            from agents_core.forgejo import get_open_prs as _get_open_prs
            repo_name, owner = _repo_owner(target.pm_repo)
            _adopted_branch = target.data.get("adopted_head_branch") or ""
            for pr in _get_open_prs(repo_name, owner=owner):
                pr_ref = (pr.get("head") or {}).get("ref", "")
                if pr_ref.startswith(f"lapis/{target_id}/") or (
                    _adopted_branch and pr_ref == _adopted_branch
                ):
                    pr_num = pr.get("number", "?")
                    raise ValueError(
                        f"target {target_id} already has open PR #{pr_num} on {pr_ref} — "
                        "dispatch a reviewer/fixer_retry, not a new initial fixer"
                    )
        except ValueError:
            raise
        except Exception:
            # Degrade-open on Forgejo error; never wedge dispatch on transient API failure
            import sys
            print(f"[L1.D1-forgejo-warning] {target_id}: open-PR scan failed; proceeding with dispatch", file=sys.stderr)

    # Leg 2 (lapis-pm-stale-pending-dispatch-reaper-v0): reconcile before the
    # L1.D3 guard below evaluates, so the documented escape from a wedge
    # (tick --force-dispatch) is self-healing for every case the reconciler's
    # reap pass can prove (task gone from the queue entirely, or stuck in
    # active/ past its own timeout). Placed in force_dispatch itself — not
    # in the CLI — so cmd_bind_chain's call site and any future caller
    # inherit it for free.
    #
    # Failure handling is visibility, not a raise: if reconciliation returns
    # 0 because the queue read threw, warn to stderr and fall through to the
    # guard as today. Failing closed here would manufacture a second way to
    # wedge a target on an unreadable queue — exactly the defect class this
    # spec exists to remove.
    try:
        _flipped, _reconcile_ok = _reconcile_dispatched_with_queue_ex(target_id)
    except Exception:
        _reconcile_ok = False
    if not _reconcile_ok:
        import sys
        print(
            f"[L1.D3-reconcile-warning] {target_id}: dispatch reconciliation "
            "was skipped (queue read failed) before the concurrency guard; "
            "guard below may be evaluating stale state",
            file=sys.stderr,
        )

    # L1.D3: Target-level concurrency guard for initial fixers
    if agent_type in _INITIAL_FIXER_TYPES:
        existing_dispatches = load_dispatched(target_id)
        for record in existing_dispatches:
            if (record.get("status") == "pending" and
                record.get("agent_type") in _INITIAL_FIXER_TYPES + ("fixer_retry",)):
                gpu_id = record.get("gpu_id", "unknown")
                rec_agent = record.get("agent_type", "unknown")
                age = _dispatch_age(record.get("ts"))
                age_note = f" pending for {_format_age(age)}" if age is not None else " pending"
                raise ValueError(
                    f"target {target_id} has a {rec_agent} dispatch ({gpu_id}),"
                    f"{age_note} — if that fixer is dead: "
                    f"lapis-pm clear-dispatch {target_id}"
                )

    spec_sum = episodic.spec_summary(target_id)
    _adopted_branch = target.data.get("adopted_head_branch") or ""
    existing_branch = _adopted_branch if _adopted_branch else f"lapis/{target_id}/forced"
    base_branch = "main"
    slug = "forced"
    pr_number: int | None = None
    # L1.D2: Widen open-PR/canonical-branch reuse lookup to include fixer (not just fixer_retry).
    # Also resolve pr_number here for reviewer/reviewer_fresh/fixer_retry so cycle accounting
    # (_fixer_retry_count / _reviewer_cycle_count) can attribute force-dispatched records
    # to the correct PR, same as the daemon's own _act_dispatch_fixer_retry/_act_dispatch_reviewer.
    # fixer_staged (lapis-pm-fixers-harness-registry-v0, Leg 2): a BARE entry here — it must
    # resolve the parked PR's ref via this scan so the staged engine checks out the PR head,
    # but it must NOT be added to _INITIAL_FIXER_TYPES (that constant also gates the
    # open-PR ValueError guard above; extending it would block the staged dispatch against a
    # parked PR).
    if agent_type in _INITIAL_FIXER_TYPES + ("fixer_retry", "fixer_staged") + _REVIEWER_AGENT_TYPES and target.pm_repo:
        try:
            from agents_core.forgejo import get_open_prs as _get_open_prs
            repo_name, owner = _repo_owner(target.pm_repo)
            for pr in _get_open_prs(repo_name, owner=owner):
                pr_ref = (pr.get("head") or {}).get("ref", "")
                if pr_ref.startswith(f"lapis/{target_id}/") or (
                    _adopted_branch and pr_ref == _adopted_branch
                ):
                    existing_branch = pr_ref
                    base_branch = (pr.get("base") or {}).get("ref", "main")
                    pr_number = pr.get("number")
                    # Extract slug from the branch name (last component after /)
                    if pr_ref.startswith(f"lapis/{target_id}/"):
                        slug = pr_ref.split("/")[-1]
                    break
        except Exception:
            pass
    # Resolve the reviewer cycle (and, for agent_type == "reviewer", the matching
    # prior_review text) BEFORE building vars_/dispatching, so the template a
    # force-dispatched reviewer actually sees matches the cycle its record claims.
    # The `reviewer` template requires {prior_review} — `_SHAPER.dispatch` calls
    # str.format(**vars_) with no fallback, so a missing key raises KeyError before
    # this function ever reaches the record-construction code below. Build it the
    # same way the daemon's own _act_dispatch_reviewer does.
    reviewer_cycle: int | None = None
    if agent_type in _REVIEWER_AGENT_TYPES:
        reviewer_cycle = (_reviewer_cycle_count(target_id, pr_number) + 1) if pr_number is not None else 1

    prior_review_text = ""
    if agent_type == "reviewer":
        prior_review_text = _build_prior_review_text(target_id, pr_number, reviewer_cycle)

    vars_ = {
        "target_id": target_id,
        "spec_summary": spec_sum,
        "repo": target.pm_repo or "",
        "question": intent,
        "pr_number": str(pr_number) if pr_number is not None else "",
        "slug": slug,
        "existing_branch": existing_branch,
        "base_branch": base_branch,
        "intent_block": _intent_artifact.dispatch_block(target_id),
        "absence_grounding_block": _absence_grounding_block_for(agent_type),
    }
    if agent_type == "reviewer":
        vars_["prior_review"] = prior_review_text
    steer.inject_overlay(target_id, vars_, agent_type)
    _ensure_dispatch_owned(vars_.get("repo", ""))
    res = _SHAPER.dispatch(agent_type, target_id, intent, vars_=vars_)
    record = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": agent_type,
        "intent": intent,
        "repo": target.pm_repo or "",
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": 0,
    }
    reviewer_tags: list[str] = []
    if pr_number is not None:
        if agent_type == "fixer_retry":
            record["pr_number"] = pr_number
            record["cycle"] = _reviewer_cycle_count(target_id, pr_number)
        elif agent_type in _REVIEWER_AGENT_TYPES:
            record["pr_number"] = pr_number
            record["cycle"] = reviewer_cycle
            reviewer_tags = [f"pm:reviewer:pr={pr_number}:cycle={reviewer_cycle}:verdict=pending"]
    append_dispatched(target_id, record)
    _check_calcification(target_id)
    message = (
        episodic_message
        if episodic_message is not None
        else f"Forced dispatch: {agent_type} → {res.task_id}\nIntent: {intent}"
    )
    episodic.write_dispatch(
        target_id,
        message,
        extra_tags=[f"pm:gpu={res.task_id}", f"pm:agent={agent_type}"] + reviewer_tags + (extra_episodic_tags or []),
    )
    try:
        from .router_portfolio import emit_decision_dispatch as _emit_dispatch
        dispatched = load_dispatched(target_id)
        if agent_type in _INITIAL_FIXER_TYPES:
            frag = "kickoff" if len(dispatched) == 1 else "tick"
        elif agent_type == "reviewer":
            frag = "review-cycle"
        elif agent_type == "brief":
            frag = "human-judgment"
        else:
            frag = agent_type
        model = "unknown"
        try:
            model = _SHAPER.get_agent(agent_type).model
        except Exception:
            pass
        _TIER_MAP = {
            "haiku": "haiku", "sonnet": "sonnet", "opus": "opus",
            "qwen-3.6-35b-a3b": "qwen-local", "qwen3.6-35b-a3b": "qwen-local",
        }
        _emit_dispatch(
            target_id=target_id,
            fragment_id=frag,
            expert_chosen=_TIER_MAP.get(model.lower(), model.lower()),
            intent_summary=intent[:200],
        )
    except Exception as _e:
        import sys
        print(f"[router-portfolio:emit-failed] dispatch: {_e}", file=sys.stderr)
    return res.task_id


def get_pause_state(target_id: str) -> str | None:
    rec = _mem().get(_pause_key(target_id))
    return rec["content"] if rec else None


def set_pause_state(target_id: str, state: str):
    _mem().set(_pause_key(target_id), state, tags=["lapis-pm", "pause-state"])


def set_outstanding_brief(target_id: str, comment_id: str):
    _mem().set(_brief_key(target_id), comment_id, tags=["lapis-pm", "outstanding-brief"])


def _supersede_gem_for_cleared_brief(target_id: str, comment_id: str, reason: str) -> None:
    """Best-effort: supersede any live brief-gem tied to (target_id, comment_id).

    Fail-soft — any error here must never block the outstanding-brief clear.
    Uses brief_gem's own reverse-index lookup (the same one deposit_brief_gem
    uses for idempotency) so this is a single choke-point rather than four
    call-site patches.
    """
    import sys
    try:
        from . import brief_gem as _brief_gem

        mem = _mem()
        rkey = _brief_gem._reverse_key(target_id, comment_id)
        existing = mem.get(rkey)
        if not existing:
            return  # no gem ever deposited for this brief
        gem_id = existing["content"]
        result = _brief_gem._call_supersede_endpoint(
            gem_id, reason=f"brief cleared: {reason}", by="lapis-pm:brief-cleared",
        )
        if result is True:
            _brief_gem._update_map_status(
                gem_id, "superseded", annotation=f"brief cleared: {reason}",
            )
    except Exception as exc:
        print(
            f"[outstanding-brief:supersede-gem-error] tid={target_id} "
            f"cid={comment_id} err={exc!r}",
            file=sys.stderr,
        )


def clear_outstanding_brief(target_id: str, reason: str = "unspecified") -> None:
    import sys
    # Read the current value for the audit line. A read failure must not
    # block the delete — that would regress idempotency.
    try:
        observed = get_outstanding_brief(target_id)
    except Exception as exc:
        print(
            f"[outstanding-brief:clearing:read-error] tid={target_id} "
            f"reason={reason} err={exc!r}",
            file=sys.stderr,
        )
        observed = None
    if observed is not None:
        print(
            f"[outstanding-brief:clearing] tid={target_id} cid={observed} "
            f"reason={reason}",
            file=sys.stderr,
        )
        _supersede_gem_for_cleared_brief(target_id, observed, reason)
    else:
        print(
            f"[outstanding-brief:clearing:already-absent] tid={target_id} "
            f"reason={reason}",
            file=sys.stderr,
        )
    _mem().delete(_brief_key(target_id))


def get_outstanding_brief(target_id: str) -> str | None:
    rec = _mem().get(_brief_key(target_id))
    return rec["content"] if rec else None


class OutstandingBriefWriteError(RuntimeError):
    """set_outstanding_brief failed to persist after retry."""


def set_outstanding_brief_verified(target_id: str, comment_id: str) -> None:
    """Set the outstanding-brief mem key with a read-back verify.

    Writes the key, immediately reads it back, and on mismatch:
    1. Logs a structured WARN line to stderr (so journalctl picks it
       up — see `[router-portfolio:emit-failed]` precedent at
       pm_core.py:384) with target_id, comment_id, and the observed
       value (or 'missing').
    2. Retries the write exactly once.
    3. Re-reads. If still missing or mismatched after retry, raises
       OutstandingBriefWriteError so the caller surfaces the failure
       rather than logging a phantom action.

    The action-string returned by _act_brief is the operator's contract
    for "the brief is now resolvable" — silent persistence failure breaks
    that contract.
    """
    import sys
    set_outstanding_brief(target_id, comment_id)
    for attempt in range(1, 3):
        observed = get_outstanding_brief(target_id)
        if observed == comment_id:
            return
        observed_repr = observed if observed is not None else "missing"
        print(
            f"[outstanding-brief:write-mismatch] tid={target_id} cid={comment_id}"
            f" observed={observed_repr} attempt={attempt}",
            file=sys.stderr,
        )
        if attempt == 1:
            set_outstanding_brief(target_id, comment_id)
    raise OutstandingBriefWriteError(
        f"set_outstanding_brief failed to persist after retry: "
        f"tid={target_id} cid={comment_id}"
    )


def _set_brief_outstanding(
    target_id: str,
    b: brief.Brief,
    *,
    verified: bool = False,
) -> bool:
    """Set the outstanding brief, skipping if synthesis failed (within threshold).

    Returns True if the brief was set outstanding, False if skipped for retry.
    When synthesis keeps failing past _SYNTH_FAIL_THRESHOLD ticks, sets outstanding
    anyway so a human can force-clear or redirect.
    """
    if not b.synthesis_failed:
        _reset_synth_fail_count(target_id)
        if verified:
            set_outstanding_brief_verified(target_id, b.comment_id)
            _post_write_sweep_brief(target_id, b.comment_id)
        else:
            set_outstanding_brief(target_id, b.comment_id)
        try:
            from . import brief_gem as _brief_gem
            _brief_gem.deposit_brief_gem(target_id, b)
        except Exception as _bg_exc:
            logger.warning("brief-gem deposit failed (non-fatal): %s", _bg_exc)
        return True

    fail_n = _inc_synth_fail_count(target_id)
    if fail_n >= _SYNTH_FAIL_THRESHOLD:
        logger.warning(
            "brief synthesis failed %d consecutive ticks for %s — setting outstanding "
            "to unblock; clear manually or force-dispatch",
            fail_n, target_id,
        )
        if verified:
            set_outstanding_brief_verified(target_id, b.comment_id)
            _post_write_sweep_brief(target_id, b.comment_id)
        else:
            set_outstanding_brief(target_id, b.comment_id)
        try:
            from . import brief_gem as _brief_gem
            _brief_gem.deposit_brief_gem(target_id, b)
        except Exception as _bg_exc:
            logger.warning("brief-gem deposit failed (non-fatal): %s", _bg_exc)
        return True

    logger.warning(
        "brief synthesis failed (attempt %d/%d) for %s — skipping set_outstanding_brief, "
        "will retry next tick",
        fail_n, _SYNTH_FAIL_THRESHOLD, target_id,
    )
    episodic.write_observation(
        target_id,
        f"Brief synthesis failed (attempt {fail_n}/{_SYNTH_FAIL_THRESHOLD}); "
        "underlying state preserved - next tick will retry.",
        extra_tags=["pm:synthesis-failed"],
    )
    return False


def _post_write_sweep_brief(target_id: str, comment_id: str) -> None:
    """Read the brief key once more after _mark_pr_classified ran.

    This catches the external-deleter hypothesis the verify-and-retry
    cannot defend against: if some sibling process (sweeper, concurrent
    PM session, mem CLI invocation) deletes the key in the window
    between verify and the next status read, this sweep is the only
    on-tick surface that records the disappearance.

    Logs to stderr only; never raises. Sweep is observability, not
    guarantee — the verify-and-retry IS the guarantee.
    """
    import sys
    observed = get_outstanding_brief(target_id)
    if observed != comment_id:
        print(
            f"[outstanding-brief:disappeared-post-write] tid={target_id} cid={comment_id}",
            file=sys.stderr,
        )


def _landed_key(target_id: str) -> str:
    return f"pm/landed/{target_id}"


# ---------------------------------------------------------------------------
# Forgejo health gate
# ---------------------------------------------------------------------------

def probe_forgejo_health() -> tuple[bool, str]:
    """Probe Forgejo reachability with GET /api/v1/version (3-second timeout).

    Sends the same Authorization header as all other agents_core.forgejo calls
    (Forgejo returns 403 to anonymous requests on this instance).

    Called once per tick from tick_all() only; single-target tick() bypasses
    the probe entirely.

    Returns (True, "") on success.
    Returns (False, reason) on failure where reason is one of:
      "connect_error", "timeout", "http=<code>"
    """
    try:
        from agents_core.forgejo import FORGEJO_URL, FORGEJO_TOKEN
    except Exception:
        return True, ""  # agents_core unavailable — assume reachable
    try:
        import httpx
    except ImportError:
        return True, ""  # httpx unavailable — assume reachable
    try:
        r = httpx.get(
            f"{FORGEJO_URL}/api/v1/version",
            headers={"Authorization": f"token {FORGEJO_TOKEN}"},
            timeout=3.0,
        )
        if not (200 <= r.status_code < 300):
            return False, f"http={r.status_code}"
        return True, ""
    except httpx.TimeoutException:
        return False, "timeout"
    except Exception:
        return False, "connect_error"


def _get_forgejo_consecutive_fails() -> int:
    rec = _mem().get(FORGEJO_CONSECUTIVE_FAILS_KEY)
    if not rec:
        return 0
    try:
        return int(rec["content"])
    except (ValueError, KeyError, TypeError):
        return 0


def _directive_seen_observations(target_id: str) -> list:
    """D7 (attestation-contract-v0 leg 2): the target's `pm:directive-seen`
    observations, oldest first (CommentStore.list order). Each carries the
    directive id + ts and a `pm:pr-head=<sha>` tag (the open-PR head at
    directive time; `none` when no PR was open).

    REV 4 pin: the raised brief's comment id rides a companion
    `pm:directive-brief-raised` observation written on the same tick
    (`pm:brief-id=<cid>`, `pm:brief-id=none` when the raise failed) — the
    cid is paired onto the directive-seen observation by directive id (the
    latest such companion for the same `pm:directive-id`; `brief_id` is
    None when no companion exists, e.g. pre-rev-4 observations, and
    outcome (c) is then unprovable — fail-open to paging rather than to
    the over-broad suppression)."""
    out = []
    companions: dict[str, str | None] = {}
    for c in episodic.all_comments(target_id):
        if "pm:directive-brief-raised" not in c.tags:
            continue
        d_id = ""
        b_id = None
        for t in c.tags:
            if t.startswith("pm:directive-id="):
                d_id = t.split("=", 1)[1]
            elif t.startswith("pm:brief-id="):
                # Normalize the `pm:brief-id=none` literal (raise failed)
                # to None, same as an empty value - fail-open to paging.
                b_id = t.split("=", 1)[1]
                if not b_id or b_id == "none":
                    b_id = None
        if d_id:
            companions[d_id] = b_id
    for c in episodic.all_comments(target_id):
        if "pm:directive-seen" not in c.tags:
            continue
        directive_id = ""
        pr_head = None
        for t in c.tags:
            if t.startswith("pm:directive-id="):
                directive_id = t.split("=", 1)[1]
            elif t.startswith("pm:pr-head="):
                pr_head = t.split("=", 1)[1]
        out.append({
            "comment": c,
            "directive_id": directive_id,
            "pr_head_baseline": pr_head,
            "brief_id": companions.get(directive_id),
        })
    return out


def _parse_ts_aware(ts: str) -> datetime | None:
    """Parse an ISO timestamp into a tz-aware datetime (PACIFIC when naive),
    or None when unparseable."""
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=PACIFIC)
    return dt


def _directive_outcome_holds(
    target_id: str,
    directive_ts: str,
    pr_head_baseline: str | None,
    brief_id: str | None = None,
) -> bool:
    """D7 (attestation-contract-v0 leg 2): True when ANY of the three
    outcomes holds for the directive observed at `directive_ts`:

      (a) a dispatch was recorded after the directive ts
          (pm/dispatched/<tid> entry with a ts parsed TIMEZONE-AWARE and
          compared as datetimes — a string compare mis-orders timestamps
          across offset spellings, e.g. `+07:00` vs `+00:00`);
      (b) the target's open PR head advanced PAST THE BASELINE recorded on
          the directive-seen observation (pm/pr-head/<tid> record written
          at the tick's PR-perception site — the detector makes NO Forgejo
          call of its own); a `none` baseline + any open PR counts as an
          advance;
      (c) the directive's OWN brief was consumed — REV 4 pin: keyed on the
          specific brief id recorded on the observation
          (`pm:brief-id=<cid>`): a decision audit recorded against that cid
          (decision/brief-resolved/<cid>, written by the single brief
          resolution path) or a clear of that cid (the outstanding-brief
          slot holding a different cid at the directive's age means this
          one was consumed). An EMPTY outstanding-brief slot does NOT
          count — the over-broad `get_outstanding_brief is None -> True`
          reading suppressed the page for any target with no outstanding
          brief (the common steady state), defeating the detector.
          `brief_id=None` (pre-rev-4 observation or a failed raise) makes
          (c) unprovable: it holds only when the slot is non-empty, which
          is the conservative reading (no suppression on the common case).

    Forgejo-free by construction: mem reads + episodic reads only.
    """
    # (a) dispatch after the directive ts (timezone-aware compare)
    dir_dt = _parse_ts_aware(directive_ts)
    if dir_dt is not None:
        for r in load_dispatched(target_id):
            r_dt = _parse_ts_aware(r.get("ts") or "")
            if r_dt is not None and r_dt > dir_dt:
                return True
    # (b) PR head advanced past the baseline
    rec = _mem().get(_pr_head_key(target_id))
    if rec:
        try:
            heads = json.loads(rec["content"])
        except (json.JSONDecodeError, TypeError):
            heads = []
        if isinstance(heads, list):
            current = [
                h.get("head_sha") for h in heads
                if isinstance(h, dict) and h.get("head_sha")
            ]
            if pr_head_baseline in (None, "none"):
                if current:
                    return True
            elif current and any(s != pr_head_baseline for s in current):
                return True
    # (c) the directive's OWN brief was consumed (specific-cid contract)
    if brief_id:
        # A decision recorded against that cid (the single resolution path
        # writes the audit key before clearing the slot).
        audit = _mem().get(f"decision/brief-resolved/{brief_id}")
        if audit:
            return True
        # A clear of that cid: the slot now holds a DIFFERENT brief.
        if get_outstanding_brief(target_id) not in (None, brief_id):
            return True
    elif get_outstanding_brief(target_id) is not None:
        # No brief id recorded (pre-rev-4 observation / failed raise):
        # (c) is unprovable; a non-empty slot is the only conservative
        # suppression (an empty slot does NOT count).
        return True
    return False


def _set_forgejo_consecutive_fails(n: int) -> None:
    _mem().set(FORGEJO_CONSECUTIVE_FAILS_KEY, str(n), tags=["lapis-pm", "forgejo-health"])


def _notify_forgejo_unreachable() -> None:
    """Emit one HIGH-priority Pushover when Forgejo has been unreachable for ≥3 ticks."""
    try:
        from agents_core.notify import send_notification, Priority as _P
        send_notification(
            "Forgejo health probe has failed for 3 or more consecutive ticks. "
            "Daemon is skipping all target processing until Forgejo recovers.",
            title="lapis-pm: Forgejo unreachable for >3 ticks",
            priority=_P.HIGH,
        )
    except Exception:
        pass


def _classified_pr_ids(target_id: str) -> set[int]:
    """PR numbers that have already been classified (hold/advisory/merge) this target."""
    rec = _mem().get(_classified_prs_key(target_id))
    if not rec:
        return set()
    try:
        data = json.loads(rec["content"])
        return {int(x) for x in (data if isinstance(data, list) else [])}
    except (json.JSONDecodeError, ValueError):
        return set()


def _mark_pr_classified(target_id: str, pr_number: int):
    ids = _classified_pr_ids(target_id)
    ids.add(pr_number)
    _mem().set(_classified_prs_key(target_id),
               json.dumps(sorted(ids)),
               tags=["lapis-pm", "classified-prs"])


def clear_classified_prs(target_id: str):
    """Reset classified-PR state. Called on unbind/rebind so stale PR numbers don't block re-evaluation."""
    _mem().delete(_classified_prs_key(target_id))


def clear_landed_state(target_id: str) -> dict[str, int]:
    """Clear all PM mem state for a landed target.

    Called from `lapis-pm land` so archived targets stop polluting
    tick loops and `lapis-pm status`. Pending dispatched records are
    marked `status: "landed"` (not deleted) so the history stays
    queryable from mem for future arc-doc regeneration or audit.

    Clears: outstanding_brief, classified_prs, cursor, pause_state,
    dispatched_pending (marked landed), review_state cache.

    Returns a dict describing what was non-empty at clear time —
    for a caller (the `land` command) to print a useful summary.
    Idempotent: safe to call on a target that has no state.
    """
    mem = _mem()
    summary = {
        "outstanding_brief": 0,
        "classified_prs": 0,
        "cursor": 0,
        "pause_state": 0,
        "dispatched_pending": 0,
        "review_state": 0,
    }

    if get_outstanding_brief(target_id) is not None:
        summary["outstanding_brief"] = 1
    clear_outstanding_brief(target_id, reason="auto_land")

    summary["classified_prs"] = len(_classified_pr_ids(target_id))
    clear_classified_prs(target_id)

    if get_cursor(target_id) is not None:
        summary["cursor"] = 1
    mem.delete(_cursor_key(target_id))

    if get_pause_state(target_id) is not None:
        summary["pause_state"] = 1
    mem.delete(_pause_key(target_id))

    records = load_dispatched(target_id)
    pending = [r for r in records if r.get("status") == "pending"]
    if pending:
        for r in records:
            if r.get("status") == "pending":
                r["status"] = "landed"
        save_dispatched(target_id, records)
    summary["dispatched_pending"] = len(pending)

    if mem.get(_review_state_key(target_id)) is not None:
        summary["review_state"] = 1
    mem.delete(_review_state_key(target_id))

    mem.delete(_SYNTH_FAIL_KEY.format(target_id))

    return summary


# ---------------------------------------------------------------------------
# Debt-ledger close at land time (debt-ledger-merge-flip-v0)
# ---------------------------------------------------------------------------

# Anchored full-string match only — never used as a substring search. Greedy
# repo group + anchored 10-hex suffix recovers (repo, debt_id) from a
# mechanical-item tid (bundle_triage.py:549-554 mechanical_item_target_id);
# repos contain hyphens (e.g. "lapis-pm"), so the suffix anchor is what makes
# the split unambiguous.
_DEBT_ITEM_TID_RE = re.compile(r"^cr-bundle-item-(.+)-([0-9a-f]{10})$")


def close_resolved_debt_for_target(
    target_id: str,
    *,
    resolved_by_pr: int | None = None,
    ground: str,
) -> bool:
    """Flip review/debt/<repo>/<debt_id>'s ledger entry to resolved at land time.

    Called beside clear_landed_state at each of the three land choke points
    (cli.py cmd_land, pm_core._act_auto_land, pm_core._act_auto_land_already_
    satisfied) so a shipped bundle-item debt entry stops being re-harvested
    by the nightly `debt-open` tag-bucket sweep.

    Fail-soft throughout, mirroring the never-raise precedent at
    bundle_autodispatch._mark_debt_invalid: any failure returns False and is
    logged, never raised — a debt-flip failure must never alter the land
    outcome. `ground` is a required keyword so each call site's write stays
    distinguishable from any other reconciliation of the same key.
    """
    m = _DEBT_ITEM_TID_RE.match(target_id)
    if not m:
        return False
    repo, debt_id = m.group(1), m.group(2)
    key = f"review/debt/{repo}/{debt_id}"
    try:
        mem = _mem()
        rec = mem.get(key)
        if not rec:
            logger.warning(
                "[pm_core] debt-ledger-close: key %s not found for target %s",
                key, target_id,
            )
            return False
        body = yaml.safe_load(rec.get("content", "")) or {}
        if body.get("status") != "open":
            # Already resolved/invalid/deferred — no-clobber, idempotent.
            return False
        body["status"] = "resolved"
        body["resolved_by_pr"] = resolved_by_pr
        body["resolved_at"] = _now_iso()
        body["resolved_ground"] = str(ground)[:500]
        mem.set(
            key,
            yaml.safe_dump(body, sort_keys=False),
            tags=["review-debt", f"repo-{repo}", "debt-resolved"],
            source="lapis-pm-land-close",
        )
        return True
    except Exception as exc:
        logger.warning(
            "[pm_core] debt-ledger-close: failed for %s (target %s): %s",
            key, target_id, exc,
        )
        return False


# ---------------------------------------------------------------------------
# Already-satisfied verdict helpers
# ---------------------------------------------------------------------------

def _handle_already_satisfied_verdict(target_id: str, rec: dict, verdict_raw: dict) -> bool:
    """Evaluate an already_satisfied verdict sidecar. Returns True if handled.

    When True is returned, the caller should mark the dispatch processed and
    continue (skip confabulation check and normal write_result). Two sub-cases:

    - valid verdict + valid merged PR → write pm:already-satisfied observation
    - valid verdict + invalid/missing PR → write pm:already-satisfied:invalid
      observation (decide phase will brief the human)

    Malformed verdicts (non-dict, unrecognised verdict type, missing pr_num)
    return False so the caller falls through to the normal confabulation path.
    """
    import sys

    verdict_type = verdict_raw.get("verdict")
    if verdict_type != "already_satisfied":
        print(f"WARN: unrecognised verdict type in sidecar: {verdict_type!r}", file=sys.stderr)
        return False

    pr_num = verdict_raw.get("satisfied_by_pr")
    if not isinstance(pr_num, int) or pr_num <= 0:
        print(f"WARN: malformed satisfied_by_pr in verdict sidecar: {pr_num!r}", file=sys.stderr)
        return False

    evidence = str(verdict_raw.get("evidence", ""))[:500]
    repo = rec.get("repo", "")

    if _validate_already_satisfied_pr(repo, pr_num):
        episodic.write_observation(
            target_id,
            f"Fixer verdict: already_satisfied by PR #{pr_num}\nEvidence: {evidence}",
            extra_tags=["pm:already-satisfied", f"pm:already-satisfied:pr={pr_num}"],
        )
        return True

    # Forgejo validation failed — warn but still mark as handled so the fixer
    # is not re-retried (it intentionally produced no PR).
    print(
        f"WARN: already_satisfied PR #{pr_num} not found or not merged in {repo!r}",
        file=sys.stderr,
    )
    episodic.write_observation(
        target_id,
        f"Fixer claimed already_satisfied (PR #{pr_num}) but Forgejo validation failed "
        f"— will brief human instead of auto-landing. Evidence: {evidence}",
        extra_tags=["pm:already-satisfied:invalid", f"pm:already-satisfied:invalid:pr={pr_num}"],
    )
    return True


def _already_satisfied_pending(target_id: str) -> tuple[int, str, str] | None:
    """Return (pr_num, evidence, ts) if a valid already_satisfied verdict awaits action.

    Conditions: pm:already-satisfied:pr=N observation exists, target not yet
    landed, no pending dispatches.
    """
    if _mem().get(_landed_key(target_id)):
        return None
    if _has_pending_dispatch(target_id):
        return None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:already-satisfied:pr=") and "invalid" not in t:
                try:
                    pr_num = int(t.split("=", 1)[1])
                    content = c.content
                    evidence = ""
                    if "Evidence:" in content:
                        evidence = content.split("Evidence:", 1)[1].strip()
                    return (pr_num, evidence, c.ts)
                except (ValueError, IndexError):
                    pass
    return None


def _already_satisfied_invalid_pending(target_id: str) -> tuple[int, str] | None:
    """Return (pr_num, content) if an invalid already_satisfied verdict needs a brief.

    Returns None if already briefed (de-dup via pm:already-satisfied:invalid-briefed tag).
    """
    briefed = any(
        "pm:already-satisfied:invalid-briefed" in c.tags
        for c in episodic.all_comments(target_id)
    )
    if briefed:
        return None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:already-satisfied:invalid:pr="):
                try:
                    pr_num = int(t.split("=", 1)[1])
                    return (pr_num, c.content)
                except (ValueError, IndexError):
                    pass
    return None


# ---------------------------------------------------------------------------
# Auto-land helpers
# ---------------------------------------------------------------------------

def _merged_pr_numbers_observed(target_id: str) -> set[int]:
    """PR numbers for which a pm:pr-merged observation has been recorded in episodic."""
    out: set[int] = set()
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:pr-merged:"):
                try:
                    out.add(int(t.split(":", 2)[2]))
                except (ValueError, IndexError):
                    pass
    return out


def _merged_at_for_pr(target_id: str, pr_num: int) -> str:
    """Return the actual Forgejo merged_at timestamp for *pr_num* from episodic.

    Falls back to _now_iso() if the observation is missing (shouldn't happen
    in practice, but keeps _act_auto_land safe on retry after partial failure).
    """
    tag = f"pm:pr-merged:{pr_num}"
    for c in episodic.all_comments(target_id):
        if tag in c.tags:
            m = re.search(r"merged at (\S+),", c.content)
            if m:
                return m.group(1)
    return _now_iso()


def _encode_merged_prs(target_id: str, repo: str) -> int:
    """Check Forgejo for newly merged lapis PRs and write pm:pr-merged observations.

    Only calls get_pr for PR numbers not already noted. Returns count of new
    observations written. Gracefully degrades when Forgejo is unavailable.
    """
    if not repo or not _forgejo_get_pr:
        return 0
    seen = _seen_pr_ids(target_id)
    already_noted = _merged_pr_numbers_observed(target_id)
    new_obs = 0
    repo_name, owner = _repo_owner(repo)
    for pr_num in sorted(seen - already_noted):
        try:
            pr_data = _forgejo_get_pr(repo_name, pr_num, owner=owner)
        except Exception:
            continue
        if pr_data.get("merged") and pr_data.get("state") == "closed":
            merged_at = pr_data.get("merged_at") or _now_iso()
            created_at = pr_data.get("created_at") or ""
            head_sha = (pr_data.get("head") or {}).get("sha", "")
            episodic.write_observation(
                target_id,
                f"PR #{pr_num} merged at {merged_at}, created_at={created_at}, head_sha={head_sha}",
                extra_tags=[f"pm:pr-merged:{pr_num}", f"pm:pr={pr_num}"],
            )
            new_obs += 1
    return new_obs


def _detect_pr_count_from_spec(spec_text: str) -> int:
    """Count '^### PR \\d+' headers in spec; return max(count, 1). Regex-only, LLM-free."""
    matches = re.findall(r"^### PR \d+(?=\s|$)", spec_text, re.MULTILINE)
    return max(len(matches), 1)


def _has_auto_land_waiting_comment(target_id: str, pr_count: int) -> bool:
    """Return True if a pm:auto-land:waiting comment for this pr_count already exists."""
    tag = f"pm:auto-land:waiting:count={pr_count}"
    for c in episodic.all_comments(target_id):
        if tag in c.tags:
            return True
    return False


def _reconcile_surviving_head_branch(target_id: str, repo: str) -> None:
    """Delete a merged PR's surviving head branch so auto-land can proceed.

    A merge performed outside merge_and_deploy (e.g. a Forgejo web-UI merge with
    the delete-branch box unchecked) never gets the branch-deletion backstop, which
    leaves _is_auto_land_eligible permanently False. This reconciles that: bounded
    to the single most-recently-merged observed PR (one branch probe per tick, per
    target), reusing the existing idempotent, non-raising _ensure_head_branch_deleted
    primitive rather than adding a new Forgejo surface. Best-effort — a failure here
    is surfaced as a pm:error observation but never raises out of tick().
    """
    if not repo:
        return
    merged = _merged_pr_numbers_observed(target_id)
    if not merged:
        return
    pr_num = max(merged)
    try:
        repo_name, owner = _repo_owner(repo)
        _ensure_head_branch_deleted(repo_name, pr_num, owner=owner)
    except Exception as e:
        episodic.write_observation(
            target_id, f"Head branch reconcile failed for PR #{pr_num}: {e}",
            extra_tags=["pm:error", "pm:branch-reconcile-failed"],
        )


def _is_pr_merged(target_id: str, pr_num: int) -> bool:
    """Part C net (2026-08-24): the single merged-state predicate.

    A PR counts as merged when the daemon recorded a merged observation for
    it (the same observation set _is_auto_land_eligible is built from) OR a
    live Forgejo state check says merged. Fail-conservative: any exception,
    404, or not-found returns False (not merged) - never raises.
    """
    if pr_num in _merged_pr_numbers_observed(target_id):
        return True
    if _forgejo_get_pr is None:
        return False
    target = TargetStore().get(target_id)
    if target is None or not target.pm_repo:
        return False
    try:
        repo_name, owner = _repo_owner(target.pm_repo)
        pr_data = _forgejo_get_pr(repo_name, pr_num, owner=owner)
        return (pr_data.get("state") or "") == "merged"
    except Exception:
        return False


def _is_auto_land_eligible(target_id: str) -> bool:
    """Return True if this target meets all auto-land conditions.

    Conditions (all must hold):
      - No pm/landed/<tid> entry already exists
      - No pending dispatches
      - At least one merged PR is observed
      - The most recently seen PR is the merged one (no newer open PR)
      - All declared PRs have merged (len(merged) >= target.pr_count)
      - The PR's head branch is deleted (or Forgejo unavailable, per proxy)
    Paused check is applied by the caller: tick_all() excludes paused targets
    from auto-land pre-selection, and tick() itself also short-circuits to
    noop:paused before the decide phase that would call _act_auto_land.
    """
    if _mem().get(_landed_key(target_id)):
        return False
    if _has_pending_dispatch(target_id):
        return False
    seen = _seen_pr_ids(target_id)
    merged = _merged_pr_numbers_observed(target_id)
    if not merged or not seen:
        return False
    # Guard: a newer (higher-numbered) open PR must not exist
    if max(seen) != max(merged):
        return False
    # Guard: all declared PRs must have merged
    target = TargetStore().get(target_id)
    required = target.data.get("pr_count", 1) if target else 1
    if len(merged) < required:
        # Emit one audit comment per (target_id, pr_count) tuple — de-dup by tag
        if not _has_auto_land_waiting_comment(target_id, required):
            episodic.write_observation(
                target_id,
                f"auto-land deferred: {len(merged)}/{required} PRs merged on lapis/{target_id}/*",
                extra_tags=["pm:auto-land:waiting", f"pm:auto-land:waiting:count={required}"],
            )
        return False
    # Branch deletion check: confirm the PR's head branch is gone
    if _forgejo_get_pr is not None and _forgejo_get_branch is not None:
        if target and target.pm_repo:
            try:
                repo_name, owner = _repo_owner(target.pm_repo)
                pr_data = _forgejo_get_pr(repo_name, max(merged), owner=owner)
                head_ref = (pr_data.get("head") or {}).get("ref", "")
                if head_ref:
                    try:
                        _forgejo_get_branch(repo_name, head_ref, owner=owner)
                        # Branch still exists — not yet eligible
                        return False
                    except Exception:
                        pass  # 404 or network error → treat as deleted
            except Exception:
                pass  # Forgejo unavailable → fall through to merge-status proxy
    # Part C net (2026-08-24): the merged state is now expressed through the
    # single _is_pr_merged predicate. Behavior-neutral: max(merged) is in the
    # observed-merged set by construction, so the predicate's observation
    # short-circuit returns True exactly as the old literal True did.
    return _is_pr_merged(target_id, max(merged))


def _spec_bound_ts(target_id: str) -> str:
    """Return ts of the spec:bound comment (proxy for bind time); used for sorting."""
    for c in episodic.all_comments(target_id):
        if episodic.TAG_SPEC in c.tags:
            return c.ts
    return "9999-99-99"  # unbound targets sort last


def _act_auto_land(target_id: str) -> str:
    """Perform automatic landing: arc doc + archive + unbind + mem cleanup.

    Writes pm/landed/<tid> BEFORE unbinding so any partial failure between
    the mem write and unbind is observable and the target can be manually
    completed. The arc doc write is idempotent (overwrites on retry).
    """
    from . import land as land_module

    merged = _merged_pr_numbers_observed(target_id)
    pr_num = max(merged) if merged else 0
    merged_at = _merged_at_for_pr(target_id, pr_num)
    landed_at = _now_iso()

    # 1. Generate and write arc doc (idempotent on retry)
    arc = land_module.generate_arc_doc(target_id)
    path = land_module.write_arc_doc(arc)

    # 2. Write pm/landed/<tid> — commit marker (must succeed before unbind)
    _mem().set(
        _landed_key(target_id),
        json.dumps({"pr_num": pr_num, "merged_at": merged_at,
                    "landed_at": landed_at, "arc_path": str(path)}),
        tags=["lapis-pm", "landed"],
    )

    _deploy_target = TargetStore().get(target_id)
    _post_land_deploy_hook(_deploy_target.pm_repo if _deploy_target else None, target_id=target_id)

    # 3. Audit comment in the target JSONL (spec-required format)
    episodic.write(
        target_id,
        f"auto-landed: PR #{pr_num} merged at {merged_at}, "
        f"branch deleted, no pending dispatches. arc={path}",
        tags=["pm:auto-land"],
    )

    # 3a. Chain advance: update chain state + auto-fire dependent legs.
    # Read chain_group before archive — `archive()` reloads the target and
    # subsequent state changes can leave the in-memory copy stale.
    store = TargetStore()
    _target_for_chain = store.get(target_id)
    _chain_group = _target_for_chain.data.get("chain_group") or "" if _target_for_chain else ""
    try:
        from . import chain as _chain
        _chain.on_leg_landed(target_id, _chain_group)
        _chain.check_chain_advance(target_id)
    except Exception as _chain_err:
        episodic.write_observation(
            target_id,
            f"chain-advance error (non-fatal): {_chain_err}",
            extra_tags=["pm:chain-error"],
        )

    # 4. Archive + unbind + clear mem state
    store.archive(target_id)
    target = store.get(target_id)
    target.unbind_pm()
    target.save()
    close_resolved_debt_for_target(
        target_id, resolved_by_pr=pr_num, ground=f"auto-land pr={pr_num}",
    )
    clear_landed_state(target_id)

    # Opportunistic compaction of old landed targets (non-fatal).
    try:
        compact_eligible_targets()
    except Exception as _ce:
        logger.warning("compact_eligible_targets failed (non-fatal): %s", _ce)

    return f"action:auto_land:pr={pr_num}:arc={path}"


def _act_auto_land_already_satisfied(target_id: str) -> str:
    """Auto-land via an already_satisfied verdict: arc doc + archive + unbind.

    Mirrors _act_auto_land but uses the verdict's PR number (not a freshly
    merged lapis/* PR) and appends the spec-required "already satisfied" note
    to the arc doc's Origin section.
    """
    from . import land as land_module

    result = _already_satisfied_pending(target_id)
    if result is None:
        return "noop"
    pr_num, evidence, satisfied_ts = result
    landed_at = _now_iso()

    # Generate arc doc with the already-satisfied addendum to Origin.
    already_sat_note = (
        f"_Spec verified already satisfied by PR #{pr_num} at {satisfied_ts}; "
        "landed without new dispatch._"
    )
    arc = land_module.generate_arc_doc(target_id, extra_origin_note=already_sat_note)
    path = land_module.write_arc_doc(arc)

    # Commit marker (same structure as regular auto-land for symmetry)
    _mem().set(
        _landed_key(target_id),
        json.dumps({
            "pr_num": pr_num,
            "merged_at": satisfied_ts,
            "landed_at": landed_at,
            "arc_path": str(path),
            "via": "already_satisfied",
        }),
        tags=["lapis-pm", "landed"],
    )

    _deploy_target = TargetStore().get(target_id)
    _post_land_deploy_hook(_deploy_target.pm_repo if _deploy_target else None, target_id=target_id)

    # Audit comment (spec-required tag)
    episodic.write(
        target_id,
        f"auto-landed via already_satisfied verdict: PR #{pr_num} was merged before "
        f"dispatch ran. arc={path}",
        tags=["pm:auto-land", "pm:auto-land:already-satisfied"],
    )

    # Chain advance (same as regular auto-land)
    store = TargetStore()
    _target_for_chain = store.get(target_id)
    _chain_group = (
        _target_for_chain.data.get("chain_group") or ""
        if _target_for_chain else ""
    )
    try:
        from . import chain as _chain
        _chain.on_leg_landed(target_id, _chain_group)
        _chain.check_chain_advance(target_id)
    except Exception as _chain_err:
        episodic.write_observation(
            target_id,
            f"chain-advance error (non-fatal): {_chain_err}",
            extra_tags=["pm:chain-error"],
        )

    # Archive + unbind + clear mem state
    store.archive(target_id)
    target = store.get(target_id)
    target.unbind_pm()
    target.save()
    close_resolved_debt_for_target(
        target_id, resolved_by_pr=pr_num, ground="already-satisfied verdict",
    )
    clear_landed_state(target_id)

    # Opportunistic compaction of old landed targets (non-fatal).
    try:
        compact_eligible_targets()
    except Exception as _ce:
        logger.warning("compact_eligible_targets failed (non-fatal): %s", _ce)

    return f"auto_land:already_satisfied:pr={pr_num}:arc={path}"


def _act_brief_already_satisfied_invalid(target_id: str) -> str:
    """Post a brief for an already_satisfied verdict where the cited PR failed validation."""
    result = _already_satisfied_invalid_pending(target_id)
    if result is None:
        return "noop"
    pr_num, content = result

    b = brief.synthesize(
        target_id,
        trigger=f"fixer claimed already_satisfied but PR #{pr_num} not found or not merged",
        query=f"Fixer already-satisfied verdict: invalid PR #{pr_num}",
        notify=NotifyPriority.NORMAL,
    )
    _set_brief_outstanding(target_id, b)
    episodic.write_observation(
        target_id,
        f"Brief posted for invalid already_satisfied verdict (PR #{pr_num}): {b.comment_id}",
        extra_tags=["pm:already-satisfied:invalid-briefed"],
    )
    return f"already_satisfied_invalid_brief:{b.comment_id}"


# ---------------------------------------------------------------------------
# Reviewer attempt ceiling (lapis-pm-reviewer-attempt-ceiling-v0)
#
# Bounds *attempts*, not successes. Keyed the same way cycle accounting is
# (target_id, pr_number, cycle) so a failure loop that never advances the
# cycle (no verdict ever written) is still bounded. PR-bound, survives new
# SHAs, cleared only by an explicit manual clear — see spec "Reset scope".
# ---------------------------------------------------------------------------

def _reviewer_attempt_ceiling() -> int:
    raw = os.environ.get(REVIEWER_ATTEMPT_CEILING_ENV)
    if raw:
        try:
            parsed = int(raw)
            if parsed >= 1:
                return parsed
        except ValueError:
            pass
    return REVIEWER_ATTEMPT_CEILING_DEFAULT


def _reviewer_infra_retry_budget() -> int:
    raw = os.environ.get(REVIEWER_INFRA_RETRY_BUDGET_ENV)
    if raw:
        try:
            parsed = int(raw)
            if parsed >= 1:
                return parsed
        except ValueError:
            pass
    return REVIEWER_INFRA_RETRY_BUDGET_DEFAULT


def _classify_reviewer_infra_reason(reported_reason: str | None) -> str | None:
    """Return the matched infra reason if `reported_reason` names one of
    REVIEWER_INFRA_FAIL_REASONS, else None. Reason strings arrive embedded in
    a larger message, e.g. "ERROR: local reviewer produced no verdict
    (reason=gw_not_serving)" — extract via `reason=<value>` first, then fall
    back to a bare substring match against the allow-list for any other
    shape the reason might arrive in (e.g. straight from a queue error)."""
    if not reported_reason:
        return None
    m = re.search(r"reason=([\w.-]+)", reported_reason)
    if m and m.group(1) in REVIEWER_INFRA_FAIL_REASONS:
        return m.group(1)
    for candidate in REVIEWER_INFRA_FAIL_REASONS:
        if candidate in reported_reason:
            return candidate
    return None


def _reviewer_dispatch_wait_seconds(rec: dict) -> float | None:
    """Wall-clock seconds between dispatch (`rec["ts"]`) and terminal state
    (`rec["completed_at"]`), for the infra-wait telemetry (DoD 9). Returns
    None — never 0 — if either timestamp is missing or unparseable; a
    fabricated 0 would misrepresent an unmeasured duration as an instant
    refusal, defeating the point of the telemetry (distinguishing a
    momentary jam from systemic starvation)."""
    started = rec.get("ts")
    ended = rec.get("completed_at")
    if not started or not ended:
        return None
    try:
        delta = datetime.fromisoformat(ended) - datetime.fromisoformat(started)
    except (ValueError, TypeError):
        return None
    return delta.total_seconds()


def _reviewer_attempt_key(target_id: str, pr_number: int, cycle: int) -> str:
    return f"pm/reviewer-attempts/{target_id}/pr={pr_number}/cycle={cycle}"


def _reviewer_attempt_state(target_id: str, pr_number: int, cycle: int) -> dict:
    default = {
        "count": 0,
        "last_reason": None,
        "infra_count": 0,
        "last_infra_reason": None,
        "last_infra_wait_s": None,
        # R1 (lapis-pm-reviewer-defer-backoff-v0): missing == no backoff, so
        # records written before this unit landed keep working unchanged.
        "next_retry_at": None,
    }
    rec = _mem().get(_reviewer_attempt_key(target_id, pr_number, cycle))
    if not rec:
        return default
    try:
        data = json.loads(rec["content"])
        return {
            "count": int(data.get("count", 0)),
            "last_reason": data.get("last_reason"),
            "infra_count": int(data.get("infra_count", 0)),
            "last_infra_reason": data.get("last_infra_reason"),
            "last_infra_wait_s": data.get("last_infra_wait_s"),
            "next_retry_at": data.get("next_retry_at"),
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return default


def _reviewer_attempt_count(target_id: str, pr_number: int, cycle: int) -> int:
    return _reviewer_attempt_state(target_id, pr_number, cycle)["count"]


def _increment_reviewer_attempt(target_id: str, pr_number: int, cycle: int) -> int:
    """Record one reviewer dispatch attempt. Called at dispatch time, so it
    counts every attempt regardless of whether it later succeeds or fails."""
    state = _reviewer_attempt_state(target_id, pr_number, cycle)
    state["count"] += 1
    _mem().set(
        _reviewer_attempt_key(target_id, pr_number, cycle),
        json.dumps(state),
        tags=["lapis-pm", "reviewer-attempt-ceiling", f"target={target_id}"],
    )
    return state["count"]


def _record_reviewer_attempt_reason(target_id: str, pr_number: int, cycle: int, reason: str,
                                    wait_s: float | None = None,
                                    completed_at: str | None = None) -> None:
    """Stash the most recent failure reason for this pr+cycle, as-reported and
    unverified — see DoD 3b. Classifies the reason against
    REVIEWER_INFRA_FAIL_REASONS (D1): an infrastructure-classified failure
    bumps infra_count, which _reviewer_attempt_ceiling_check reads to exclude
    that attempt from the ceiling count (and to bound infra-only failures
    against the separate, more generous REVIEWER_INFRA_RETRY_BUDGET). Does
    not touch the raw `count` — that stays dispatch-time-incremented history,
    per the module docstring above.

    `wait_s` (DoD 9, lapis-pm-reviewer-defer-timeout-is-infra-v0): the
    wall-clock duration of the failed attempt, recorded beside
    last_infra_reason so a congestion pause (long wait, gw_defer_timeout) is
    distinguishable from an absence pause (short wait, gw_not_serving)
    without reading code. Only recorded when the reason classifies as infra —
    non-infra failures don't consume this field. Left as None (never
    defaulted to 0) when the caller has no measured duration — a 0 would be
    indistinguishable from an instant refusal and would defeat the
    telemetry's purpose.

    lapis-pm-reviewer-defer-backoff-v0 (R1): an infra-classified failure also
    stamps next_retry_at = completed_at + backoff(infra_count) via the
    REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC table — decide()'s redispatch path
    (_reviewer_infra_backoff_check) honours this before its next dispatch.
    A non-infra failure — a genuine review defect, not a non-run — clears
    next_retry_at: it breaks a consecutive-infra streak, so a mixed-failure
    sequence never silently spends the backoff window against a stale clock
    left over from an earlier infra run."""
    state = _reviewer_attempt_state(target_id, pr_number, cycle)
    state["last_reason"] = reason
    infra_reason = _classify_reviewer_infra_reason(reason)
    if infra_reason is not None:
        state["infra_count"] = state.get("infra_count", 0) + 1
        state["last_infra_reason"] = infra_reason
        state["last_infra_wait_s"] = wait_s
        state["next_retry_at"] = _reviewer_infra_backoff_until(completed_at, state["infra_count"])
    else:
        state["next_retry_at"] = None
    _mem().set(
        _reviewer_attempt_key(target_id, pr_number, cycle),
        json.dumps(state),
        tags=["lapis-pm", "reviewer-attempt-ceiling", f"target={target_id}"],
    )


def _reviewer_infra_backoff_seconds(infra_count: int) -> int:
    """Look up the backoff duration (seconds) for the Nth consecutive infra
    failure (1-indexed) in REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC. Clamps to the
    table's last entry for infra_count beyond its length — the infra budget
    stops at REVIEWER_INFRA_RETRY_BUDGET_DEFAULT == len(table), so this is a
    defensive floor, not a live path — and to the first entry for infra_count
    < 1 (shouldn't happen; fail toward the shortest wait rather than crash)."""
    table = REVIEWER_INFRA_BACKOFF_SCHEDULE_SEC
    index = min(max(infra_count, 1), len(table)) - 1
    return table[index]


def _reviewer_infra_backoff_until(completed_at: str | None, infra_count: int) -> str:
    """Compute next_retry_at = completed_at + backoff(infra_count) (R1).
    Falls back to now() if completed_at is missing/unparseable — an
    unmeasured completion time still deserves a backoff window, just
    anchored to when we noticed rather than when the attempt actually
    ended."""
    anchor = None
    if completed_at:
        try:
            anchor = datetime.fromisoformat(completed_at)
        except (ValueError, TypeError):
            anchor = None
    if anchor is None:
        anchor = datetime.now(PACIFIC)
    seconds = _reviewer_infra_backoff_seconds(infra_count)
    return (anchor + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def _clear_reviewer_infra_backoff(target_id: str, pr_number: int, cycle: int) -> None:
    """Clear next_retry_at for this attempt-state record without touching
    count/infra_count/last_infra_reason (R1) — those stay historical.
    Called on any non-infra completion of this pr+cycle: a real verdict
    (here) or a non-infra failure (inline in
    _record_reviewer_attempt_reason). A no-op, not a write, when there's
    nothing to clear."""
    state = _reviewer_attempt_state(target_id, pr_number, cycle)
    if state.get("next_retry_at") is None:
        return
    state["next_retry_at"] = None
    _mem().set(
        _reviewer_attempt_key(target_id, pr_number, cycle),
        json.dumps(state),
        tags=["lapis-pm", "reviewer-attempt-ceiling", f"target={target_id}"],
    )


def clear_reviewer_attempts(target_id: str) -> int:
    """Manual clear — the sole escape hatch once a target is ceiling-paused.

    Deliberately cheap: one call, one required argument (target_id), no
    confirmation prompt, no metadata. Wipes every attempt counter and
    ceiling-hit marker for this target (all PRs, all cycles). Does NOT
    resume the target — that remains a separate, explicit `lapis-pm resume`.
    Returns the number of keys cleared.
    """
    cleared = 0
    for prefix in (
        f"pm/reviewer-attempts/{target_id}/",
        f"pm/reviewer-attempt-ceiling/{target_id}/",
        f"pm/reviewer-infra-budget/{target_id}/",
    ):
        for rec in _mem().list_by_prefix(prefix, limit=10_000):
            if _mem().delete(rec["key"]):
                cleared += 1
    return cleared


def clear_dispatch(target_id: str) -> int:
    """Leg 3 (lapis-pm-stale-pending-dispatch-reaper-v0): the escape hatch
    for the residue Leg 1's reap pass cannot prove — a queue entry that is
    genuinely still running but whose PM record is wrong, or a queue whose
    files were moved by hand. Modeled on clear_reviewer_attempts: cheap,
    one required argument, no confirmation prompt (Erah's ruling — ceremony
    here inverts the point of an escape hatch).

    Flips every `pending` dispatch record for target_id to `failed` with
    failure_reason "cleared:operator", stamps completed_at, saves, and
    writes one episodic audit comment naming every cleared gpu_id. Never
    touches non-pending records, TargetStore, paused state, or the spec.
    Returns the number of records cleared.
    """
    records = load_dispatched(target_id)
    cleared_gpu_ids: list[str] = []
    now = _now_iso()
    for rec in records:
        if rec.get("status") != "pending":
            continue
        rec["status"] = "failed"
        rec["failure_reason"] = "cleared:operator"
        rec["completed_at"] = now
        cleared_gpu_ids.append(rec.get("gpu_id", "unknown"))

    if cleared_gpu_ids:
        save_dispatched(target_id, records)
        episodic.write_observation(
            target_id,
            f"Cleared {len(cleared_gpu_ids)} pending dispatch(es) via "
            f"clear-dispatch: {', '.join(cleared_gpu_ids)}",
            extra_tags=["pm:dispatch-cleared"],
        )

    return len(cleared_gpu_ids)


# ---------------------------------------------------------------------------
# Review-gate loop helpers
# ---------------------------------------------------------------------------

def _reviewer_cycle_count(target_id: str, pr_number: int) -> int:
    """Count completed reviewer cycles for this PR via episodic tags."""
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    count = 0
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                if verdict_val != "pending":
                    count += 1
    return count


def _last_review_verdict(target_id: str, pr_number: int) -> dict | None:
    """Return the most recent completed reviewer verdict dict for this PR, or None."""
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    last_comment = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                if verdict_val != "pending":
                    last_comment = c
    if last_comment is None:
        return None
    # Content format: "Reviewer verdict for PR #N:\n<json>"
    content = last_comment.content
    try:
        json_part = content.split("\n", 1)[-1].strip()
        return json.loads(json_part)
    except (json.JSONDecodeError, IndexError):
        # Fall back to extracting verdict from tag
        for t in last_comment.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                return {"verdict": verdict_val, "issues": [], "confidence": 0.0}
        return None


def _verdict_rendered_sha(target_id: str, pr_number: int, comment_ts: str | None) -> str | None:
    """D7: the head sha a reviewer verdict was rendered against.

    Preference order:
    1. the verdict comment's own `pm:reviewer-sha=<sha>` tag (D7's
       render-time tag — the exact sha the verdict was rendered against);
    2. else the latest `pm:pr=N:sha=` observation at or before the verdict
       comment's ts (computable: sha observations encode in the same tick
       BEFORE gpu results — the ordering is hard-required by the tick's
       encode-phase comment);
    3. else the latest `pm:pr=N:sha=` observation overall (pre-tag verdicts
       and the `sha=unknown` perceive-miss case — treated as fresh when it
       does not differ from the current head).

    Returns None when nothing is available — the caller treats that as
    FRESH (effectively unreachable), per the spec's `sha=unknown` rule.
    """
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    sha_prefix = f"pm:pr={pr_number}:sha="
    verdict_comment = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                if t.split(":verdict=")[-1] != "pending":
                    verdict_comment = c
    if verdict_comment is not None:
        for t in verdict_comment.tags:
            if t.startswith("pm:reviewer-sha="):
                return t[len("pm:reviewer-sha="):] or None
    sha: str | None = None
    for c in episodic.all_comments(target_id):
        if comment_ts is not None and c.ts > comment_ts:
            continue
        for t in c.tags:
            if t.startswith(sha_prefix):
                sha = t[len(sha_prefix):]
    if sha is not None:
        return sha
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(sha_prefix):
                sha = t[len(sha_prefix):]
    return sha


def _last_review_verdict_comment(target_id: str, pr_number: int):
    """The most recent completed reviewer-verdict Comment for this PR (or
    None). Same tag scan as _last_review_verdict — factored out so the
    verdict's ts is available for the D7 rendered-against-sha lookup."""
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    last_comment = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                if t.split(":verdict=")[-1] != "pending":
                    last_comment = c
    return last_comment


def _review_verdict_for_cycle(target_id: str, pr_number: int, cycle: int) -> dict | None:
    """Return the verdict dict for a specific reviewer cycle, or None if absent.

    Mirrors _last_review_verdict's tag-and-content parsing but filters by cycle.
    """
    prefix = f"pm:reviewer:pr={pr_number}:cycle={cycle}:verdict="
    target_comment = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix):
                verdict_val = t[len(prefix):]
                if verdict_val != "pending":
                    target_comment = c
    if target_comment is None:
        return None
    content = target_comment.content
    try:
        json_part = content.split("\n", 1)[-1].strip()
        return json.loads(json_part)
    except (json.JSONDecodeError, IndexError):
        for t in target_comment.tags:
            if t.startswith(prefix):
                verdict_val = t[len(prefix):]
                return {"verdict": verdict_val, "issues": [], "confidence": 0.0}
        return None


def _fixer_retry_count(target_id: str, pr_number: int) -> int:
    """Count fixer_retry dispatches that actually advanced the PR for this PR.

    Only `status == "processed"` records count — those are the dispatches the
    SHA-advance perceiver (~pm_core.py _perceive_pr_sha_advance) has confirmed
    pushed code. Failed dispatches did not push, so they do not advance cycle
    accounting.
    """
    return sum(
        1 for r in load_dispatched(target_id)
        if r.get("agent_type") == "fixer_retry"
        and r.get("pr_number") == pr_number
        and r.get("status") == "processed"
    )


def _has_pending_reviewer_for_pr(target_id: str, pr_number: int) -> bool:
    return any(
        r.get("status") == "pending"
        and r.get("agent_type") in _REVIEWER_AGENT_TYPES
        and r.get("pr_number") == pr_number
        for r in load_dispatched(target_id)
    )


def _has_pending_fixer_for_pr(target_id: str, pr_number: int) -> bool:
    return any(
        r.get("status") == "pending"
        and r.get("agent_type") == "fixer_retry"
        and r.get("pr_number") == pr_number
        for r in load_dispatched(target_id)
    )


def _collect_review_history(target_id: str, pr_number: int) -> list[dict]:
    """Collect all reviewer verdicts for a PR from episodic, in order."""
    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    history: list[dict] = []
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                if verdict_val == "pending":
                    continue
                cycle_part = t.split(":cycle=")[1].split(":")[0]
                try:
                    cycle_num = int(cycle_part)
                except ValueError:
                    cycle_num = 0
                entry: dict = {"cycle": cycle_num, "verdict": verdict_val, "issues": []}
                try:
                    json_part = c.content.split("\n", 1)[-1].strip()
                    data = json.loads(json_part)
                    entry["issues"] = data.get("issues", [])
                except (json.JSONDecodeError, IndexError):
                    pass
                history.append(entry)
    history.sort(key=lambda h: h["cycle"])
    return history


def _last_observed_pr_sha(target_id: str, pr_number: int) -> str | None:
    """Return the most recently observed head SHA for this PR from episodic, or None."""
    prefix = f"pm:pr={pr_number}:sha="
    last: str | None = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix):
                last = t[len(prefix):]
    return last


def _last_observed_pr_body_fp(target_id: str, pr_number: int) -> str | None:
    """Return the most recently observed body fingerprint for this PR from episodic, or None."""
    prefix = f"pm:pr={pr_number}:body="
    last: str | None = None
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix):
                last = t[len(prefix):]
    return last


def _fixer_sha_completion_ts(target_id: str, pr_number: int, dispatch_ts: str) -> str | None:
    """Return ts of first SHA-advance observation for PR N after dispatch_ts, or None."""
    prefix = f"pm:pr={pr_number}:sha="
    for c in episodic.all_comments(target_id):
        if c.ts > dispatch_ts:
            for t in c.tags:
                if t.startswith(prefix):
                    return c.ts
    return None


def _fixer_completion_ts(target_id: str, pr_number: int, dispatch_ts: str) -> str | None:
    """Return ts of first SHA-advance or body-advance observation for PR N after dispatch_ts."""
    sha_prefix = f"pm:pr={pr_number}:sha="
    body_prefix = f"pm:pr={pr_number}:body="
    for c in episodic.all_comments(target_id):
        if c.ts > dispatch_ts:
            for t in c.tags:
                if t.startswith(sha_prefix) or t.startswith(body_prefix):
                    return c.ts
    return None


def _reviewer_dispatch_ts(target_id: str, pr_number: int, cycle: int) -> str | None:
    """Return dispatch ts of reviewer at cycle K for PR N, or None."""
    for r in load_dispatched(target_id):
        if (r.get("agent_type") in _REVIEWER_AGENT_TYPES
                and r.get("pr_number") == pr_number
                and r.get("cycle") == cycle):
            return r.get("ts")
    return None


# ---------------------------------------------------------------------------
# Auditor (fixer-reception-v0, leg 2, D5/D6): the LLM-powered receiving role
# for non-passing fixer terminations (Erah 2026-09-07, ratified). The
# auditor investigates the actual state (three-head suite runs + file:line
# root-cause evidence) and returns a structured JSON audit brief; the daemon
# encodes it into the pm:auditor comment (D6c) — the auditor posts nothing.
#
# Invariants honored here (spec §Invariants):
# - the auditor is an ENCODE-phase trigger, not a reordering of the decide
#   phase's single FIFO action (the FIFO lowest-number rule is unchanged);
# - the auditor does NOT consume the review budget (it is not a reviewer
#   cycle and never touches the reviewer-attempt counters);
# - no schema change to Target/Comment: the audit brief is a
#   comment-with-tags + pm/audit/<tid>/pr=N/* mem keys, the same substrate
#   the reviewer verdicts use.
# ---------------------------------------------------------------------------

AUDITOR_AGENT_TYPE = "auditor"

# Per-target audit budget (D6a gate iii): at most N auditor dispatches per
# target per 12h. Env-var-resolved constant (gate finding, transmuter, run
# 2026-09-07-224357): the arbiter-tightening doctrine (spec Invariant 2)
# applies SYMMETRICALLY to loosening AND tightening — either direction tunes
# without a code change. Default 2 (the census in hand: 9 salvage PRs in one
# evening on 2026-09-07).
AUDIT_BUDGET_PER_12H_DEFAULT = 2
AUDIT_BUDGET_PER_12H_ENV = "GW_AGENT_AUDIT_BUDGET_PER_12H"
_AUDIT_BUDGET_WINDOW_HOURS = 12


def _audit_budget_per_12h() -> int:
    """Call-time env read (same discipline as the reviewer-attempt ceiling
    env reads): the per-target auditor dispatch budget per 12h window."""
    raw = os.environ.get(AUDIT_BUDGET_PER_12H_ENV)
    if raw is None:
        return AUDIT_BUDGET_PER_12H_DEFAULT
    try:
        n = int(raw)
        return n if n > 0 else AUDIT_BUDGET_PER_12H_DEFAULT
    except ValueError:
        return AUDIT_BUDGET_PER_12H_DEFAULT


def _audit_budget_key(target_id: str) -> str:
    return f"pm/audit/{target_id}/budget"


def _audit_dispatched_key(target_id: str, pr_number: int) -> str:
    """Gate (i) marker (D6a): the VALUE is the head sha that was audited.
    Re-triggers only when the current head sha differs from the stored
    value; a failed/timed-out audit clears it (D6c) so the un-audited head
    re-dispatches."""
    return f"pm/audit/{target_id}/pr={pr_number}/dispatched"


def _audit_brief_key(target_id: str, pr_number: int) -> str:
    """The encoded audit brief JSON (D6c): the five-section brief the
    advisory brief surface renders from."""
    return f"pm/audit/{target_id}/pr={pr_number}/brief"


def _audit_budget_record(target_id: str) -> dict:
    """Read + prune the per-target audit budget ledger (dispatch count +
    timestamps within the trailing AUDIT_BUDGET_PER_12H window)."""
    rec = _mem().get(_audit_budget_key(target_id))
    try:
        data = json.loads(rec["content"]) if rec else {}
        entries = data.get("entries", []) if isinstance(data, dict) else []
    except (json.JSONDecodeError, AttributeError, TypeError):
        entries = []
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=_AUDIT_BUDGET_WINDOW_HOURS)).isoformat()
    return {"entries": [e for e in entries if isinstance(e, str) and e >= cutoff]}


def _audit_budget_used(target_id: str) -> int:
    return len(_audit_budget_record(target_id)["entries"])


def _audit_budget_has_room(target_id: str) -> bool:
    return _audit_budget_used(target_id) < _audit_budget_per_12h()


def _record_audit_budget(target_id: str) -> None:
    rec = _audit_budget_record(target_id)
    rec["entries"].append(datetime.now(timezone.utc).isoformat())
    _mem().set(_audit_budget_key(target_id), json.dumps(rec),
               tags=["lapis-pm", "pm:audit-budget"])


def _audit_dispatched_head(target_id: str, pr_number: int) -> str | None:
    """Gate (i): the head sha the last audit for this PR covered, or None."""
    rec = _mem().get(_audit_dispatched_key(target_id, pr_number))
    if not rec:
        return None
    try:
        data = json.loads(rec["content"])
        return data.get("head_sha") if isinstance(data, dict) else None
    except (json.JSONDecodeError, AttributeError):
        return None


def _set_audit_dispatched(target_id: str, pr_number: int, head_sha: str) -> None:
    _mem().set(
        _audit_dispatched_key(target_id, pr_number),
        json.dumps({"head_sha": head_sha, "ts": _now_iso()}),
        tags=["lapis-pm", "pm:audit-dispatched"],
    )


def _clear_audit_dispatched(target_id: str, pr_number: int) -> None:
    """D6c: a failed/timed-out audit is retryable — the head is still
    un-audited, so the marker is cleared (not a timeout-only carve-out: a
    failed parse and a hard-kill both lose the audit)."""
    _mem().delete(_audit_dispatched_key(target_id, pr_number))


def _has_pending_auditor_for_pr(target_id: str, pr_number: int) -> bool:
    """Gate (ii) (D6a): same shape as _has_pending_reviewer_for_pr — a
    pending audit defers to the next tick (in-flight guard; the council's
    900s-timeout x trigger-density finding)."""
    return any(
        r.get("status") == "pending"
        and r.get("agent_type") == AUDITOR_AGENT_TYPE
        and r.get("pr_number") == pr_number
        for r in load_dispatched(target_id)
    )


def _is_salvage_pr(pr: dict) -> bool:
    """The salvage PR shape (spec Invariant 3 — consumed, not reshaped):
    head.ref ends in `-salvage-<sha8>` AND the body carries the
    `lapis-salvage: true` marker (agents_core/shaped_runner.py:683-685)."""
    head_ref = (pr.get("head") or {}).get("ref") or ""
    if "-salvage-" not in head_ref:
        return False
    body = pr.get("body") or ""
    return "lapis-salvage: true" in body


def _pr_head_sha(pr: dict) -> str | None:
    return (pr.get("head") or {}).get("sha") or None


def _previous_head_sha(target_id: str, pr_number: int, current_sha: str | None) -> str | None:
    """D5 input: the most recent prior salvage/fixer head sha for this PR
    from the target's pm:pr=N:sha= observations (all_comments order =
    chronological); for a fresh salvage with no prior head, the original
    tracked PR's fixer head (the first observed sha). None when no sha has
    ever been observed (the caller falls back to the current head)."""
    prefix = f"pm:pr={pr_number}:sha="
    shas: list[str] = []
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix):
                shas.append(t[len(prefix):])
    if not shas:
        return None
    if current_sha is not None and shas[-1] == current_sha:
        return shas[-2] if len(shas) >= 2 else shas[-1]
    return shas[-1]


def _baseline_main_sha(repo: str) -> str | None:
    """D5 input: the daemon's `git rev-parse origin/main` at dispatch time,
    in the repo's working clone. Best-effort — None when the clone or the
    ref is unavailable (the template renders 'unresolved')."""
    repo_name = repo.rsplit("/", 1)[-1]
    clone = Path(f"/srv/git/{repo_name}-working")
    if not clone.is_dir():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "origin/main"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _render_audit_brief(audit_json: dict, pr_number: int, head_sha: str | None) -> str:
    """D6c: render the auditor's five-section JSON into the advisory brief
    surface (the comment content). The JSON is the contract; this is a
    human-readable projection of it."""
    def _fmt_state(name: str, st: dict) -> str:
        return (f"{name} sha={st.get('sha', '?')[:12]} "
                f"passed={st.get('passed', '?')} failed={st.get('failed', '?')} "
                f"errors={st.get('errors', '?')}")

    states = audit_json.get("suite_states") or {}
    delta = audit_json.get("failure_set_delta") or {}
    root_cause = audit_json.get("root_cause") or []
    delta_to_green = audit_json.get("delta_to_green") or []
    deliverables = audit_json.get("deliverables_evaluation") or []

    lines = [
        f"## Audit brief — PR #{pr_number} (head {str(head_sha or '?')[:12]})",
        "",
        "### 1. Suite states (three-head runs)",
        _fmt_state("main", states.get("main") or {}),
        _fmt_state("previous_head", states.get("previous_head") or {}),
        _fmt_state("salvage_head", states.get("salvage_head") or {}),
        "",
        "### 2. Failure-set delta",
        f"classification: {delta.get('classification', '?')}",
        f"new_failures: {', '.join(delta.get('new_failures') or []) or '(none)'}",
        f"fixed_failures: {', '.join(delta.get('fixed_failures') or []) or '(none)'}",
        f"unchanged_count: {delta.get('unchanged_count', '?')}",
        "",
        "### 3. Root causes (claim + file:line evidence)",
    ]
    for rc in root_cause:
        verified = "verified" if rc.get("verified") else "UNVERIFIED (honest: false)"
        lines.append(f"- {rc.get('failure_class', '?')}: {rc.get('claim', '?')} "
                     f"[{rc.get('evidence', '?')}] ({verified})")
    if not root_cause:
        lines.append("- (none reported)")
    lines += ["", "### 4. Delta-to-green (the next fixer's brief)"]
    for unit in delta_to_green:
        lines.append(f"- {unit.get('unit', '?')} — gate: {unit.get('gate', '?')}")
    if not delta_to_green:
        lines.append("- (none reported)")
    lines += ["", "### 5. Deliverables evaluation"]
    for d in deliverables:
        lines.append(f"- {d.get('deliverable', '?')}: {d.get('status', '?')} "
                     f"[{d.get('evidence', '?')}]")
    if not deliverables:
        lines.append("- (none reported)")
    return "\n".join(lines)


def _act_dispatch_auditor(target_id: str, pr: dict, mode: str = "salvage") -> str:
    """D6: dispatch the auditor agent via the existing dispatch machinery.

    mode: "salvage" (D6a — a new salvage-shaped PR) or "noop" (D6b — a
    healthy-verdict fixer-retry no-op). The encode-phase dispatch is SEPARATE
    from the decide phase's single FIFO action; the FIFO action is
    unchanged (spec Invariant 4).

    The auditor is NOT a reviewer cycle: it never touches the
    reviewer-attempt counters or the review budget (spec Invariant 6). It
    does consume the per-target audit budget (D6a gate iii).
    """
    pr_number = pr["number"]
    repo = pr.get("repo") or ""
    pr_ref = (pr.get("head") or {}).get("ref") or ""
    head_sha = _pr_head_sha(pr)

    # Gate (iii): the per-target audit budget (at most N per 12h).
    if not _audit_budget_has_room(target_id):
        episodic.write_observation(
            target_id,
            f"Auditor dispatch deferred for PR #{pr_number}: per-target audit "
            f"budget exhausted ({_audit_budget_used(target_id)}/"
            f"{_audit_budget_per_12h()} in the trailing "
            f"{_AUDIT_BUDGET_WINDOW_HOURS}h window)",
            extra_tags=[
                f"pm:pr={pr_number}",
                "pm:auditor-budget-exhausted",
                f"pm:audit-mode={mode}",
            ],
        )
        return f"noop:auditor_budget_exhausted:pr={pr_number}"

    # D5 inputs (daemon-resolved at dispatch).
    spec_path = episodic.spec(target_id) or "(bound spec path unresolved)"
    vars_: dict = {
        "target_id": target_id,
        "repo": repo,
        "repo_cwd": Shaper.resolve_repo_cwd(repo),
        "pr_number": pr_number,
        "existing_branch": pr_ref,
        "salvage_head_sha": head_sha or "(unresolved)",
        "previous_head_sha": (
            _previous_head_sha(target_id, pr_number, head_sha) or head_sha or "(unresolved)"
        ),
        "baseline_main_sha": _baseline_main_sha(repo) or "(unresolved)",
        "bound_spec_path": spec_path,
    }

    intent = (
        f"audit PR #{pr_number} ({mode}): three-head suite states + root-cause "
        f"evidence + delta-to-green for the next fixer"
    )

    steer.inject_overlay(target_id, vars_, AUDITOR_AGENT_TYPE)
    _ensure_dispatch_owned(vars_.get("repo", ""))
    res = _SHAPER.dispatch(AUDITOR_AGENT_TYPE, target_id, intent, vars_=vars_)
    _check_calcification(target_id)

    record = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": AUDITOR_AGENT_TYPE,
        "intent": intent,
        "repo": repo,
        "pr_number": pr_number,
        "mode": mode,
        "head_sha": head_sha,
        "existing_branch": pr_ref,
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": 0,
    }
    append_dispatched(target_id, record)

    # Budget ledger (gate iii) is recorded at dispatch time — a lost audit
    # still cost a dispatch. The gate (i) marker (value = head sha) is set at
    # ENCODE time (D6c) so a failed/timed-out audit leaves no marker and the
    # head re-dispatches.
    _record_audit_budget(target_id)

    episodic.write_dispatch(
        target_id,
        f"Auditor dispatched ({mode}) for PR #{pr_number} (head "
        f"{str(head_sha or '?')[:12]}): {AUDITOR_AGENT_TYPE} → {res.task_id}\n"
        f"PR title: {pr.get('title', '')}",
        extra_tags=[
            f"pm:repo={repo}",
            f"pm:pr={pr_number}",
            "pm:auditor",
            f"pm:audit-mode={mode}",
        ],
    )
    return f"action:auditor_dispatched:pr={pr_number}:mode={mode}"


def _maybe_dispatch_auditor_salvage(target_id: str, repo: str, pr: dict) -> str | None:
    """D6a salvage trigger (encode phase): a NEW open PR on a pm_bound target
    matching the salvage shape triggers exactly ONE auditor dispatch via the
    shared gate sequence (i)/(ii)/(iii). Returns the action string on
    dispatch, or None when a gate held (the caller counts it as a noop)."""
    if not _is_salvage_pr(pr):
        return None
    return _maybe_dispatch_auditor_salvage_gates(target_id, pr, mode="salvage")


def _maybe_dispatch_auditor_noop(target_id: str, rec: dict, pr_num) -> str | None:
    """D6b no-op trigger: a fixer-retry no-op against a HEALTHY verdict
    dispatches the auditor for the same PR (audit why the retry produced no
    change + the delta-to-green). The starved/refuted case keeps its existing
    _escalate_noop_retry_if_degraded path UNCHANGED (spec Invariant 8).

    Once-per-pr+cycle marker in the existing escalation-marker shape
    (:7311-7312); the same three gates as D6a apply (the budget gate uses the
    no-op PR's head sha via the dispatch record's head_sha).
    """
    try:
        pr_number = pr_num if isinstance(pr_num, int) else int(pr_num)
    except (TypeError, ValueError):
        return None
    cycle = rec.get("cycle", 1)

    # Healthy-verdict partition: the degraded path owns starved/refuted.
    verdict_info = _review_verdict_for_cycle(target_id, pr_number, cycle)
    if verdict_info is None:
        return None
    from . import panel_starvation as _panel_starvation
    if _panel_starvation.verdict_is_starved(verdict_info):
        return None
    if verdict_info.get("refuted_absence_findings"):
        return None

    # Once-per-pr+cycle marker (the existing escalation-marker shape).
    marker_key = (
        f"pm/noop-audit/{target_id}/pr={pr_number}/cycle={cycle}/recorded"
    )
    if _mem().get(marker_key):
        return None

    # Reconstruct the PR dict the dispatch needs (head ref + sha). The
    # dispatch record's head_sha is the no-op PR's head sha (D6b: the budget
    # gate uses it); the head ref comes from the open PR list when
    # available, else the dispatch record.
    repo = rec.get("repo") or ""
    head_sha = rec.get("head_sha")
    pr_ref = rec.get("existing_branch") or ""
    title = f"PR #{pr_number} (noop audit)"
    if repo and head_sha:
        try:
            repo_name, owner = _repo_owner(repo)
            for open_pr in get_open_prs(repo_name, owner=owner):
                if open_pr.get("number") == pr_number:
                    head_sha = _pr_head_sha(open_pr) or head_sha
                    pr_ref = (open_pr.get("head") or {}).get("ref") or pr_ref
                    title = open_pr.get("title") or title
                    break
        except Exception:
            pass  # Forgejo unreachable: dispatch with the record's sha
    pr = {
        "number": pr_number,
        "repo": repo,
        "head": {"ref": pr_ref, "sha": head_sha},
        "title": title,
    }

    action = _maybe_dispatch_auditor_salvage_gates(target_id, pr, mode="noop")
    if action is None:
        return None
    _mem().set(
        marker_key,
        json.dumps({
            "target_id": target_id,
            "pr_number": pr_number,
            "cycle": cycle,
            "head_sha": head_sha,
            "action": action,
        }),
        tags=["lapis-pm", "noop-audit"],
    )
    return action


def _maybe_dispatch_auditor_salvage_gates(target_id: str, pr: dict, mode: str) -> str | None:
    """The D6a gate sequence (i)/(ii)/(iii) shared by the salvage trigger
    and the no-op trigger (D6b applies the same three gates; the no-op path
    skips the salvage-shape check because a no-op PR is not salvage-shaped)."""
    pr_number = pr.get("number")
    if pr_number is None:
        return None
    head_sha = _pr_head_sha(pr)
    audited_head = _audit_dispatched_head(target_id, pr_number)
    if audited_head is not None and (head_sha is None or audited_head == head_sha):
        return None
    if _has_pending_auditor_for_pr(target_id, pr_number):
        return None
    try:
        return _act_dispatch_auditor(target_id, pr, mode=mode)
    except Exception as exc:
        logger.warning("auditor dispatch failed for PR #%s (%s): %s",
                       pr_number, mode, exc)
        episodic.write_observation(
            target_id,
            f"Auditor dispatch failed for PR #{pr_number} ({mode}): "
            f"{type(exc).__name__}: {exc}",
            extra_tags=[f"pm:pr={pr_number}", "pm:auditor-dispatch-failed",
                        f"pm:audit-mode={mode}"],
        )
        return None


def _pr_advanced_since(target_id: str, pr_number: int, since_ts: str) -> bool:
    """Return True if a SHA-advance or body-advance observation for PR N exists after since_ts."""
    sha_prefix = f"pm:pr={pr_number}:sha="
    body_prefix = f"pm:pr={pr_number}:body="
    for c in episodic.all_comments(target_id):
        if c.ts > since_ts:
            for t in c.tags:
                if t.startswith(sha_prefix) or t.startswith(body_prefix):
                    return True
    return False


def _active_review_state(target_id: str, open_prs: list[dict]) -> dict | None:
    """Return active review-loop state for status display, or None."""
    classified_ids = _classified_pr_ids(target_id)
    for pr in open_prs:
        pr_number = pr.get("number")
        if pr_number in classified_ids:
            continue
        cycle = _reviewer_cycle_count(target_id, pr_number)
        if cycle == 0 and not _has_pending_reviewer_for_pr(target_id, pr_number):
            continue  # not started yet
        verdict_info = _last_review_verdict(target_id, pr_number)
        return {
            "pr_number": pr_number,
            "cycle": cycle,
            "verdict": verdict_info.get("verdict") if verdict_info else "pending",
            "issues": len(verdict_info.get("issues", [])) if verdict_info else 0,
        }
    return None


def _panel_health_summary(target_id: str, pr_number: int) -> dict | None:
    """Panel-health line for `lapis-pm status` (lapis-pm-panel-leg-survival-v0,
    Change 4): starved count over the last 5 reviewer verdicts for the active
    PR, plus the union of legs_down across ALL recent verdicts (AMENDED
    2026-09-10 — a second_node-only stretch renders
    `panel: 0/N starved (legs: second_node)`; the starved count itself keeps
    counting starved-only verdicts).

    Same episodic source and parsing rules as _last_review_verdict (tag
    `pm:reviewer:pr={N}:cycle=*:verdict=`, JSON body after the first newline,
    tag-fallback path included — the fallback verdict counts as starved via
    the missing-blocks rule, gate-consistent). Returns None when the PR has no
    completed reviewer verdicts (the line must not render then).
    """
    from . import panel_starvation as _panel_starvation

    prefix = f"pm:reviewer:pr={pr_number}:cycle="
    verdicts: list[dict] = []
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith(prefix) and ":verdict=" in t:
                verdict_val = t.split(":verdict=")[-1]
                if verdict_val == "pending":
                    continue
                try:
                    json_part = c.content.split("\n", 1)[-1].strip()
                    verdicts.append(json.loads(json_part))
                except (json.JSONDecodeError, IndexError):
                    verdicts.append(
                        {"verdict": verdict_val, "issues": [], "confidence": 0.0}
                    )
                break
    if not verdicts:
        return None
    # The starved COUNT is over the full stored verdict list (all cycles for
    # the PR) and counts starved-only verdicts (a GATE leg down); `total` is
    # the display window size (last 5). The legs union, however, is over ALL
    # verdicts' legs_down — AMENDED 2026-09-10 (decision/reviewer-single-leg-
    # local-phala-test-key): a second_node-only stretch must stay
    # visible on `lapis-pm status` (e.g. `panel: 0/5 starved (legs:
    # second_node)`) even though such verdicts are no longer starved.
    starved = [v for v in verdicts if _panel_starvation.verdict_is_starved(v)]
    legs = sorted({
        leg for v in verdicts for leg in _verdict_legs_down(v)
    })
    return {"starved": len(starved), "total": min(5, len(verdicts)), "legs": legs}


def _active_reviewer_backoff(target_id: str, open_prs: list[dict]) -> dict | None:
    """Return the currently-active reviewer-infra-backoff state for status
    display (R3, lapis-pm-reviewer-defer-backoff-v0), or None.

    Deliberately separate from _active_review_state: during a backoff window
    the failed attempt has already flipped pending -> failed (so
    _has_pending_reviewer_for_pr is False) and no verdict has ever landed
    (so cycle == 0) — exactly the shape _active_review_state's "not started
    yet" bail-out skips. Checking the attempt-state record directly avoids
    that blind spot, so a waiting target stays distinguishable from
    healthy-idle rather than silently reading the same as "nothing here yet".
    """
    classified_ids = _classified_pr_ids(target_id)
    for pr in open_prs:
        pr_number = pr.get("number")
        if pr_number in classified_ids:
            continue
        cycle = _reviewer_cycle_count(target_id, pr_number)
        attempt_cycle = cycle + 1
        state = _reviewer_attempt_state(target_id, pr_number, attempt_cycle)
        next_retry_at = state.get("next_retry_at")
        if not next_retry_at:
            continue
        try:
            deadline = datetime.fromisoformat(next_retry_at)
        except (ValueError, TypeError):
            continue
        now = datetime.now(deadline.tzinfo or PACIFIC)
        if now >= deadline:
            continue  # window elapsed — not "active" anymore
        return {
            "pr_number": pr_number,
            "cycle": attempt_cycle,
            "infra_count": state.get("infra_count", 0),
            "infra_budget": _reviewer_infra_retry_budget(),
            "last_infra_reason": state.get("last_infra_reason"),
            "next_retry_at": next_retry_at,
        }
    return None


# ---------------------------------------------------------------------------
# Perceive helpers
# ---------------------------------------------------------------------------

def _gpu_output_path(task_id: str) -> Path | None:
    for completed_dir, failed_dir in [
        (COMPLETED_DIR, FAILED_DIR),
        (CLAUDE_QUEUE_COMPLETED_DIR, CLAUDE_QUEUE_FAILED_DIR),
    ]:
        p = completed_dir / f"{task_id}-output.md"
        if p.exists():
            return p
        p = failed_dir / f"{task_id}-output.md"
        if p.exists():
            return p
    return None


def _repo_owner(pm_repo: str) -> tuple[str, str | None]:
    """Split pm_repo into (repo, owner) on first /; return (repo, None) for bare names.

    Examples: "lapis/coderag" -> ("coderag", "lapis"), "conductor" -> ("conductor", None)
    """
    if "/" in pm_repo:
        owner, repo = pm_repo.split("/", 1)
        return repo, owner
    return pm_repo, None


def _branch_belongs(target_id: str, branch: str) -> bool:
    return branch.startswith(f"lapis/{target_id}/")


def _extract_pr_markers(pr_body: str | None) -> dict[str, str | None]:
    """Extract lapis traceability markers from PR body.

    Returns {"gpu_id": <id> | None, "tid": <id> | None}
    Searches for HTML comments: <!-- lapis-gpu-id: <id> --> and <!-- lapis-tid: <id> -->
    """
    if not pr_body:
        return {"gpu_id": None, "tid": None}

    gpu_id = None
    tid = None

    # Search for lapis-gpu-id marker; capture non-whitespace until -->
    gpu_match = re.search(r'<!--\s*lapis-gpu-id:\s*(\S+?)\s*-->', pr_body)
    if gpu_match:
        gpu_id = gpu_match.group(1)

    # Search for lapis-tid marker; capture non-whitespace until -->
    tid_match = re.search(r'<!--\s*lapis-tid:\s*(\S+?)\s*-->', pr_body)
    if tid_match:
        tid = tid_match.group(1)

    return {"gpu_id": gpu_id, "tid": tid}


def _is_pr_traceable_to_target(target_id: str, pr: dict) -> bool:
    """Check if a PR is traceable back to this target via markers or dispatch records.

    Returns True if:
    1. PR body has lapis-tid marker matching this target, OR
    2. PR body has lapis-gpu-id marker matching a dispatched gpu_id for this target
    """
    pr_body = (pr.get("body") or "")
    markers = _extract_pr_markers(pr_body)

    # Explicit tid marker takes precedence
    if markers["tid"] == target_id:
        return True

    # Check if gpu_id matches any dispatch record for this target
    if markers["gpu_id"]:
        try:
            dispatched = load_dispatched(target_id)
            for record in dispatched:
                if record.get("gpu_id") == markers["gpu_id"]:
                    return True
        except Exception:
            pass

    return False


def _pr_owned_by_bound_sibling(pr: dict, self_target_id: str, repo: str) -> bool:
    """Check if a PR is owned by another bound target in this repo.

    Returns True if, for any sibling target (bound to the same repo and not self):
    - The PR's branch matches lapis/<sibling_id>/, OR
    - The PR is traceable to the sibling via markers/dispatch records

    Raises exception if TargetStore.load_all() fails — caller applies verify-failure policy.
    """
    store = TargetStore()
    all_targets = store.load_all()

    head = (pr.get("head") or {}).get("ref") or ""

    for sib in all_targets:
        if sib.id == self_target_id:
            continue  # skip self
        if not sib.pm_bound or sib.pm_repo != repo:
            continue  # skip targets not bound to this repo

        # Check canonical branch
        if _branch_belongs(sib.id, head):
            return True

        # Check traceability to sibling
        if _is_pr_traceable_to_target(sib.id, pr):
            return True

    return False


_RECONCILE_VERIFY_FAIL_KEY = "pm/reconcile-verify-fail-last-alert/{}"
_RECONCILE_VERIFY_FAIL_COOLDOWN_SECS = 3600


def _emit_reconcile_verify_failure(target_id: str, pr_number: int, err: Exception) -> None:
    """Emit a deduped verification-failure signal when TargetStore read fails.

    Uses cooldown to prevent notification storms during an outage. When the last
    alert for this target is within the cooldown window, logs and returns silently.
    Otherwise, sends one NORMAL-priority notification + episodic observation.
    """
    cooldown_key = _RECONCILE_VERIFY_FAIL_KEY.format(target_id)

    # Check cooldown
    try:
        last_raw = _mem().get(cooldown_key)
        if last_raw:
            last_ts = datetime.fromisoformat(last_raw["content"])
            now_ts = datetime.now(last_ts.tzinfo or PACIFIC)
            if (now_ts - last_ts).total_seconds() < _RECONCILE_VERIFY_FAIL_COOLDOWN_SECS:
                logger.debug(
                    "reconcile verify-fail alert for %s within cooldown; skipping notify",
                    target_id,
                )
                return
    except Exception:
        pass  # cooldown check failure is non-fatal

    # Set cooldown key
    try:
        _mem().set(
            cooldown_key,
            _now_iso(),
            tags=["lapis-pm", "reconcile-verify-failure"],
        )
    except Exception:
        pass  # cache failure is non-fatal

    # Send notification
    try:
        from agents_core.notify import send_notification
        send_notification(
            message=(
                f"reconcile could not verify PR ownership for `{target_id}` "
                f"(TargetStore read failed: {type(err).__name__}); "
                "orphan detection paused for this target until the store recovers."
            ),
            title=f"lapis-pm: reconcile verification failure for {target_id}",
            priority=NotifyPriority.NORMAL,
        )
    except Exception as notify_err:
        logger.warning("reconcile verify-fail notification failed: %s", notify_err)

    # Write episodic observation
    try:
        episodic.write_observation(
            target_id,
            f"Reconcile verify-failure: TargetStore read failed while checking PR #{pr_number} "
            f"ownership ({type(err).__name__}: {err}); "
            "orphan detection paused until store recovers.",
            extra_tags=["pm:reconcile-verify-failure", f"pm:pr={pr_number}"],
        )
    except Exception as obs_err:
        logger.warning("reconcile verify-fail observation failed: %s", obs_err)


def _reconcile_orphan_prs(target_id: str, target, repo: str, all_open_prs: list[dict]) -> None:
    """Reconciliation pass: check for deviant-branch PRs and auto-adopt traceable ones.

    For each open PR whose branch does NOT match lapis/<target_id>/:
    - If traceable via markers, auto-adopt (set adopted_head_branch + adopted_pr_number)
    - If owned by another bound target, skip silently
    - If not traceable to any target, raise a brief with adopt|close|ignore options

    Brief idempotency: if an outstanding brief already exists for this target
    and references this PR, skip re-synthesis to avoid LLM budget waste and
    episodic spam.
    """
    for pr in all_open_prs:
        pr_number = pr.get("number")
        head = (pr.get("head") or {}).get("ref") or ""

        # Skip canonical-branch PRs (already adopted by _perceive_prs logic)
        if _branch_belongs(target_id, head):
            continue

        # Skip already-adopted PRs
        if target.data.get("adopted_pr_number") == pr_number:
            continue

        # Check if this deviant PR is traceable to this target
        if _is_pr_traceable_to_target(target_id, pr):
            # Auto-adopt: set adopted_head_branch and adopted_pr_number
            target.data["adopted_head_branch"] = head
            target.data["adopted_pr_number"] = pr_number
            target.save()

            markers = _extract_pr_markers(pr.get("body") or "")
            episodic.write_observation(
                target_id,
                f"Orphan PR #{pr_number} on branch {head} auto-adopted "
                f"(traceable via lapis-tid={markers.get('tid') or 'N/A'} "
                f"lapis-gpu-id={markers.get('gpu_id') or 'N/A'})",
                extra_tags=["pm:orphan-adopted", f"pm:pr={pr_number}"],
            )
        else:
            # Not traceable to self: check if owned by another bound target
            try:
                owned_by_sibling = _pr_owned_by_bound_sibling(pr, target_id, repo)
            except Exception as e:
                _emit_reconcile_verify_failure(target_id, pr_number, e)
                continue

            if owned_by_sibling:
                # PR belongs to another bound target; skip silently
                continue

            # Not traceable to any target: check for outstanding brief idempotency guard
            # to avoid re-synthesizing brief.synthesize() on every tick
            existing_brief = get_outstanding_brief(target_id)
            if existing_brief:
                # A brief already exists; skip synthesis to avoid LLM waste
                continue

            # Not traceable: surface an outstanding brief with closed-form options
            markers = _extract_pr_markers(pr.get("body") or "")
            message = (
                f"Deviant-branch PR #{pr_number} on {head} is not traceable to target {target_id}.\n"
                f"Markers found: tid={markers.get('tid')}, gpu_id={markers.get('gpu_id')}\n"
                f"Options: adopt (link to this target), dismiss, or ignore."
            )
            b = brief.synthesize(
                target_id,
                trigger="orphan-pr-untraceable",
                query=message,
                pr_number=pr_number,
                notify=NotifyPriority.NORMAL
            )
            _set_brief_outstanding(target_id, b)

            episodic.write_observation(
                target_id,
                f"Orphan PR #{pr_number} on {head} not traceable; raised brief {b.comment_id}",
                extra_tags=["pm:orphan-untraceable", f"pm:pr={pr_number}", f"pm:brief={b.comment_id}"],
            )


def _perceive_prs(
    target_id: str, repo: str, adopted_pr_number: int | None = None
) -> tuple[list[dict], bool]:
    """Return (open_prs, forgejo_ok).  forgejo_ok=False means Forgejo was unreachable.

    When adopted_pr_number is set (bind --adopt-pr), the matching PR is included
    even if its head branch does not start with lapis/<target_id>/.
    """
    if not get_open_prs:
        return [], False
    try:
        repo_name, owner = _repo_owner(repo)
        prs = get_open_prs(repo_name, owner=owner)
    except Exception as e:
        try:
            import httpx as _httpx
            _is_404 = isinstance(e, _httpx.HTTPStatusError) and e.response.status_code == 404
        except Exception:
            _is_404 = False
        if _is_404:
            # Config-level 404: the bound repo does not resolve. Emit a distinct loud
            # signal once per process lifetime so the daemon does not silently loop.
            if target_id not in _repo_unresolved_signalled:
                _repo_unresolved_signalled.add(target_id)
                _resolved = f"{owner or 'Erah'}/{repo_name}"
                episodic.write_observation(
                    target_id,
                    f"[REPO-UNRESOLVED] PR fetch returned 404 for {_resolved!r} — "
                    f"this repo does not exist under its resolved owner. "
                    f"Rebind with the correct '<org>/repo' form to fix: "
                    f"`lapis-pm bind {target_id} --force --repo lapis/{repo_name}`",
                    extra_tags=["pm:error", "pm:repo-unresolved"],
                )
            return [], False
        episodic.write_observation(
            target_id, f"PR fetch failed for {repo}: {e}",
            extra_tags=["pm:error"],
        )
        return [], False
    out = []
    for pr in prs:
        head = (pr.get("head") or {}).get("ref") or ""
        if _branch_belongs(target_id, head):
            out.append(pr)
        elif adopted_pr_number is not None and pr.get("number") == adopted_pr_number:
            out.append(pr)
    return out, True


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    kind: str   # "merge" | "advisory_brief" | "hold_brief" | "retry" | "abandon_brief" | "directive_ack"
               #  | "noop_no_change" | "noop_reviewer_in_flight" | "noop_fixer_in_flight"
               #  | "reviewer_attempt_ceiling" | "reviewer_infra_budget_exhausted"
               #  | "reviewer_infra_backoff"
    payload: dict


def _reviewer_attempt_ceiling_check(target_id: str, pr_number: int, cycle: int) -> Decision | None:
    """Return a Decision if this pr+cycle has already hit either bound, else
    None. Checked immediately before every dispatch_reviewer return so a
    reviewer that never completes a verdict (and so never advances
    _reviewer_cycle_count) still stops on its own.

    Two separate bounds (D1):
    - infra budget: attempts classified as infrastructure non-runs
      (REVIEWER_INFRA_FAIL_REASONS) don't count against the reviewer-attempt
      ceiling, but they are still bounded by their own, more generous budget
      so an endlessly-absent GravityWell doesn't produce endless dispatches.
      Checked first — an infra-caused wedge must surface as an infra pause,
      never as a "clear-reviewer-attempts" ceiling pause.
    - reviewer-attempt ceiling: count of attempts minus infra-classified
      attempts. Unrecognised failure reasons count here (fail closed).
    """
    state = _reviewer_attempt_state(target_id, pr_number, cycle)

    infra_budget = _reviewer_infra_retry_budget()
    if state["infra_count"] >= infra_budget:
        return Decision("reviewer_infra_budget_exhausted", {
            "pr_number": pr_number,
            "cycle": cycle,
            "infra_attempts": state["infra_count"],
            "infra_budget": infra_budget,
            "reported_reason": state["last_infra_reason"] or state["last_reason"],
            "last_infra_wait_s": state["last_infra_wait_s"],
        })

    ceiling = _reviewer_attempt_ceiling()
    effective_count = state["count"] - state["infra_count"]
    if effective_count >= ceiling:
        return Decision("reviewer_attempt_ceiling", {
            "pr_number": pr_number,
            "cycle": cycle,
            "attempts": effective_count,
            "ceiling": ceiling,
            "reported_reason": state["last_reason"],
        })
    return None


def _reviewer_infra_backoff_check(target_id: str, pr_number: int, cycle: int) -> Decision | None:
    """Return a noop Decision while this pr+cycle's most recent infra failure
    hasn't cleared its backoff window yet (R2, lapis-pm-reviewer-defer-
    backoff-v0), else None. Reads next_retry_at from the attempt-state record
    (R1) — set on a consecutive infra failure, cleared on any non-infra
    completion — so a stale window never outlives the streak that set it.

    Deliberately a SEPARATE check from _reviewer_attempt_ceiling_check, not
    folded into it: every caller checks the ceiling first (an exhausted
    budget must still pause exactly as before — R4), then this. Kept
    separate rather than merged because the held-path caller maps any
    ceiling hit to hold_brief unconditionally (a held PR always needs the
    human hand once bounded) but must NOT do that for a mere pacing wait —
    a backoff noop is not a bound being hit.
    """
    state = _reviewer_attempt_state(target_id, pr_number, cycle)
    next_retry_at = state.get("next_retry_at")
    if not next_retry_at:
        return None
    try:
        deadline = datetime.fromisoformat(next_retry_at)
    except (ValueError, TypeError):
        return None
    now = datetime.now(deadline.tzinfo or PACIFIC)
    if now >= deadline:
        return None
    return Decision("reviewer_infra_backoff", {
        "pr_number": pr_number,
        "cycle": cycle,
        "infra_count": state.get("infra_count", 0),
        "infra_budget": _reviewer_infra_retry_budget(),
        "reason": state.get("last_infra_reason"),
        "next_retry_at": next_retry_at,
    })


def _decide_for_pr(target_id: str, repo: str, pr: dict, pm_authority: str,
                   verification: str = "pm-live-test") -> Decision:
    spec_summary = episodic.spec_summary(target_id)
    cls = authority.classify(repo, pr["number"], spec_summary, pm_authority=pm_authority,
                             verification=verification)
    payload = {"classification": cls, "pr": pr}

    # Static hold: a held path always ends in a hold brief, but it still gets a
    # machine review first — skipping it would make the PM's highest-blast-radius
    # files (lapis-pm-containment-held-paths-v0, Leg 3) the *least* reviewed
    # artifacts in the system. This reuses the same reviewer-cycle machinery as
    # the advisory/hold reviewer loop below, capped at exactly one dispatch:
    # a held PR is never fixer-retried (the fixer must not autonomously edit its
    # own cage), so reviewer_count can only ever be 0 or 1 here. Idempotency
    # ("dispatch once, not once per 60s tick") falls out of the same
    # _has_pending_reviewer_for_pr guard the normal loop uses, plus the fact that
    # only the terminal hold_brief marks a PR classified (dispatch_reviewer alone
    # does not), so every tick between dispatch and verdict is a cheap noop.
    if cls.static_outcome == authority.StaticOutcome.auto_hold_path:
        pr_number = pr["number"]
        if _has_pending_reviewer_for_pr(target_id, pr_number):
            cycle = _reviewer_cycle_count(target_id, pr_number)
            return Decision("noop_reviewer_in_flight",
                            {"pr_number": pr_number, "cycle": cycle})

        reviewer_count = _reviewer_cycle_count(target_id, pr_number)
        if reviewer_count == 0:
            ceiling_decision = _reviewer_attempt_ceiling_check(target_id, pr_number, 1)
            if ceiling_decision is not None:
                return Decision("hold_brief", payload)
            backoff_decision = _reviewer_infra_backoff_check(target_id, pr_number, 1)
            if backoff_decision is not None:
                return backoff_decision
            return Decision("dispatch_reviewer", {
                "pr": pr, "cls": cls, "mode": "fresh", "cycle": 1, "held_path": True,
            })

        # Reviewer already ran once for this PR — attach its verdict and raise
        # the hold brief. Never dispatched again regardless of verdict content
        # (fixable/needs-human included): held paths always need the human hand,
        # the reviewer's job here is only to give that human a machine read.
        verdict_info = _last_review_verdict(target_id, pr_number)
        if verdict_info is not None:
            cls.screen_verdict = verdict_info.get("verdict", "unknown")
            cls.issues = verdict_info.get("issues", [])
            payload["reviewer_verdict"] = verdict_info
        return Decision("hold_brief", payload)

    # Auto-merge path: inline local screen result drives action (unchanged)
    if pm_authority == "auto":
        if cls.verdict == "auto":
            return Decision("merge", payload)
        if cls.verdict == "hold":
            return Decision("hold_brief", payload)
        return Decision("advisory_brief", payload)

    # Advisory / hold path — reviewer loop
    pr_number = pr["number"]

    # Don't double-dispatch while reviewer or fixer is pending for this PR
    if _has_pending_reviewer_for_pr(target_id, pr_number):
        cycle = _reviewer_cycle_count(target_id, pr_number)
        return Decision("noop_reviewer_in_flight",
                        {"pr_number": pr_number, "cycle": cycle})
    if _has_pending_fixer_for_pr(target_id, pr_number):
        dispatch_id = next(
            (r.get("gpu_id", "unknown") for r in load_dispatched(target_id)
             if r.get("status") == "pending"
             and r.get("agent_type") == "fixer_retry"
             and r.get("pr_number") == pr_number),
            "unknown",
        )
        return Decision("noop_fixer_in_flight",
                        {"pr_number": pr_number, "dispatch_id": dispatch_id})

    reviewer_count = _reviewer_cycle_count(target_id, pr_number)
    fixer_count = _fixer_retry_count(target_id, pr_number)
    budget = _REVIEW_CYCLE_BUDGETS.get(pm_authority, 2)
    mode = "fresh" if pm_authority == "hold" else "same"

    if reviewer_count == fixer_count:
        # Either initial dispatch (both==0) or post-fixer dispatch (both==N).
        # For post-fixer (N>0): guard against dispatching reviewer K+1 when the
        # fixer didn't push any code (SHA unchanged since cycle K dispatch).
        # If reviewer_ts is unavailable (e.g. record cleared), default to proceed.
        if reviewer_count > 0:
            reviewer_ts = _reviewer_dispatch_ts(target_id, pr_number, reviewer_count)
            if reviewer_ts and not _pr_advanced_since(target_id, pr_number, reviewer_ts):
                return Decision("noop_no_change", {
                    "reason": (
                        f"PR #{pr_number} unchanged since reviewer "
                        f"cycle {reviewer_count} dispatch — waiting for fixer commit or description update"
                    )
                })
        if reviewer_count >= budget:
            # Both sides exhausted budget
            history = _collect_review_history(target_id, pr_number)
            return Decision("review_exhausted_brief", {
                "pr": pr, "cls": cls, "history": history,
            })
        next_cycle = reviewer_count + 1
        ceiling_decision = _reviewer_attempt_ceiling_check(target_id, pr_number, next_cycle)
        if ceiling_decision is not None:
            return ceiling_decision
        backoff_decision = _reviewer_infra_backoff_check(target_id, pr_number, next_cycle)
        if backoff_decision is not None:
            return backoff_decision
        return Decision("dispatch_reviewer", {
            "pr": pr, "cls": cls, "mode": mode, "cycle": next_cycle,
        })

    if reviewer_count < fixer_count:
        # One or more fixer_retry dispatches landed without an intervening review
        # (e.g. rapid force-dispatches against an open PR). Self-heal by dispatching
        # the next reviewer cycle rather than falling through to a silent noop.
        next_cycle = reviewer_count + 1
        ceiling_decision = _reviewer_attempt_ceiling_check(target_id, pr_number, next_cycle)
        if ceiling_decision is not None:
            return ceiling_decision
        backoff_decision = _reviewer_infra_backoff_check(target_id, pr_number, next_cycle)
        if backoff_decision is not None:
            return backoff_decision
        return Decision("dispatch_reviewer", {
            "pr": pr, "cls": cls, "mode": mode, "cycle": next_cycle,
        })

    # reviewer_count > fixer_count: reviewer has returned a verdict
    verdict_info = _last_review_verdict(target_id, pr_number)
    if verdict_info is None:
        # Shouldn't happen; defensively noop
        return Decision("noop_no_change", {"reason": "reviewer_count > fixer_count but no verdict found"})

    # ---------------------------------------------------------------------------
    # D7 (fixer-reception-v0, leg 2): the stale-verdict supersede.
    #
    # Verdicts are keyed by PR number, not head SHA — the R2 wedge: a
    # needs-human verdict rendered against an OLD head re-briefs forever
    # (live-observed 2026-09-07 20:41 PT). Before acting on a stored
    # verdict (clean/fixable/needs-human), compare its rendered-against
    # sha (the pm:reviewer-sha= render-time tag, falling back to the
    # latest pm:pr=N:sha= observation at or before the verdict's ts) to
    # the PR's current head sha (the perceive-phase PR data): if they
    # differ, the verdict is STALE — it neither blocks (hold_brief /
    # review_exhausted_brief) nor briefs from itself.
    #
    # Supersede partition (AC5):
    #   (i) budget room: dispatch a fresh reviewer on the current head
    #       through _act_dispatch_reviewer — it passes the existing
    #       ceiling/backoff checks exactly as a normal
    #       reviewer dispatch does (the re-review CONSUMES a reviewer
    #       cycle — spec Invariant 6, the named cost of superseding);
    #   (ii) budget exhausted: no dispatch — the normal exhausted path
    #       (review_exhausted_brief) surfaces to the human with the stale
    #       note attached.
    # Both cases log the pm:verdict-stale-superseded observation (both
    # shas) and do NOT re-emit the stale brief.
    #
    # Loop safety (verified): while a reviewer is pending,
    # _has_pending_reviewer_for_pr returns noop_reviewer_in_flight every
    # tick (checked above, before the verdict fork) — the supersede
    # re-fire cannot cascade. A matching sha (or no sha available — the
    # `sha=unknown` perceive-miss rule) behaves EXACTLY as today.
    # ---------------------------------------------------------------------------
    _d7_verdict_comment = _last_review_verdict_comment(target_id, pr_number)
    _d7_rendered_sha = _verdict_rendered_sha(
        target_id, pr_number,
        _d7_verdict_comment.ts if _d7_verdict_comment else None,
    )
    _d7_current_sha = (pr.get("head") or {}).get("sha")
    if (
        _d7_rendered_sha is not None
        and _d7_current_sha is not None
        and _d7_rendered_sha != _d7_current_sha
    ):
        episodic.write_observation(
            target_id,
            f"Verdict stale-superseded for PR #{pr_number}: the stored "
            f"verdict was rendered against head {_d7_rendered_sha[:12]} but "
            f"the current head is {_d7_current_sha[:12]} — the verdict "
            f"neither blocks nor briefs from its stale head",
            extra_tags=[
                f"pm:pr={pr_number}",
                "pm:verdict-stale-superseded",
                f"pm:verdict-stale-rendered-sha={_d7_rendered_sha}",
                f"pm:verdict-stale-current-sha={_d7_current_sha}",
            ],
        )
        if reviewer_count < budget:
            # Partition (i): budget room — a fresh reviewer on the
            # current head. The ceiling / backoff checks
            # run exactly as a normal reviewer dispatch (the re-review
            # consumes one reviewer cycle — Invariant 6).
            next_cycle = reviewer_count + 1
            ceiling_decision = _reviewer_attempt_ceiling_check(
                target_id, pr_number, next_cycle
            )
            if ceiling_decision is not None:
                return ceiling_decision
            backoff_decision = _reviewer_infra_backoff_check(
                target_id, pr_number, next_cycle
            )
            if backoff_decision is not None:
                return backoff_decision
            payload["stale_verdict"] = {
                "rendered_sha": _d7_rendered_sha,
                "current_sha": _d7_current_sha,
            }
            return Decision("dispatch_reviewer", {
                "pr": pr, "cls": cls, "mode": mode, "cycle": next_cycle,
            })
        # Partition (ii): budget exhausted — no dispatch; the normal
        # exhausted path surfaces with the stale note attached.
        history = _collect_review_history(target_id, pr_number)
        payload["stale_verdict"] = {
            "rendered_sha": _d7_rendered_sha,
            "current_sha": _d7_current_sha,
        }
        return Decision("review_exhausted_brief", {
            "pr": pr, "cls": cls, "history": history,
        })

    verdict = verdict_info.get("verdict", "needs-human")
    issues = verdict_info.get("issues", [])

    # ---------------------------------------------------------------------------
    # Panel-starvation gate (R1, ratified 2026-08-08; AMENDED 2026-09-10,
    # decision/reviewer-single-leg-local-phala-test-key). A GATE
    # leg down (local witness OR corroboration — the two local legs) drops
    # this verdict to advisory: it may inform, but it may never block
    # (hold_brief / review_exhausted_brief) and may never auto-dispatch a
    # fixer retry on its own authority. This is the exact failure class that
    # shipped PR #220's false HIGH finding at confidence=0.9 with all three
    # legs down — see
    # finding/reviewer-seat-false-high-finding-absence-from-diff-third-occurrence-2026-08-08.
    # A second_node (Phala TEE) absence alone does NOT starve: the verdict
    # stands on local legs, stays attenuated, and flags loudly on the brief —
    # it falls through to the normal clean/fixable/needs-human routing below.
    # Checked before the audit gate: a starved verdict's issues never reach
    # the fixable/needs-human branching below.
    # ---------------------------------------------------------------------------
    from . import panel_starvation as _panel_starvation
    if _panel_starvation.verdict_is_starved(verdict_info):
        cls.screen_verdict = verdict
        cls.issues = issues
        return Decision("advisory_brief", payload)

    # ---------------------------------------------------------------------------
    # §4 Audit gate — drop unsubstantiated still_present regurgitation
    # Runs only when there are prior issues to audit (cycle ≥ 2, same mode).
    # ---------------------------------------------------------------------------
    prior_issues_for_gate: list = []
    if mode == "same" and reviewer_count >= 2:
        prior_verdict_data = _review_verdict_for_cycle(target_id, pr_number, reviewer_count - 1)
        if prior_verdict_data is not None:
            prior_issues_for_gate = prior_verdict_data.get("issues") or []

    if prior_issues_for_gate:
        prior_resolution_raw: list = verdict_info.get("prior_resolution") or []

        # Validate resolution entries; drop malformed
        valid_resolutions: list = []
        for entry in prior_resolution_raw:
            idx = entry.get("prior_index")
            status = entry.get("status")
            evidence = entry.get("evidence", "")
            if not isinstance(idx, int) or not (0 <= idx < len(prior_issues_for_gate)):
                logger.warning(
                    "dropped malformed prior_resolution entry: prior_index=%r out of range "
                    "(prior_issues len=%d)",
                    idx, len(prior_issues_for_gate),
                )
                continue
            if status not in ("addressed", "still_present", "not_applicable"):
                logger.warning(
                    "dropped malformed prior_resolution entry: unknown status=%r prior_index=%d",
                    status, idx,
                )
                continue
            if not evidence:
                logger.warning(
                    "dropped malformed prior_resolution entry: empty evidence "
                    "prior_index=%d status=%r",
                    idx, status,
                )
                continue
            valid_resolutions.append({"prior_index": idx, "status": status, "evidence": evidence})

        # Build still_present_set from validated resolutions
        still_present_set: set = {
            r["prior_index"] for r in valid_resolutions if r["status"] == "still_present"
        }

        # Also build a set of prior paths that are still_present (for defense-in-depth)
        still_present_paths: set = {
            prior_issues_for_gate[i].get("path", "")
            for i in still_present_set
        }

        # Filter issues
        filtered_issues: list = []
        for iss in issues:
            pi = iss.get("prior_index")
            if pi is not None:
                # Issue cites a prior index — keep only if still_present
                if pi in still_present_set:
                    filtered_issues.append(iss)
                else:
                    logger.warning(
                        "dropped unsubstantiated regurgitated prior issue: prior_index=%d path=%r",
                        pi, iss.get("path"),
                    )
            else:
                # No prior_index — check path-only defense-in-depth
                iss_path = iss.get("path", "")
                prior_paths_all = {p.get("path", "") for p in prior_issues_for_gate}
                if iss_path in prior_paths_all and iss_path not in still_present_paths:
                    logger.warning(
                        "dropped uncited prior-path issue: path=%r", iss_path,
                    )
                else:
                    filtered_issues.append(iss)

        old_verdict = verdict
        issues = filtered_issues

        # Verdict downgrade after filtering
        if old_verdict == "fixable" and not issues:
            verdict = "clean"
            logger.info("verdict downgraded fixable→clean: no substantiated issues")
        elif old_verdict == "needs-human" and not issues and prior_issues_for_gate:
            # All priors addressed/not_applicable, no new issues
            all_resolved = all(
                r["status"] in ("addressed", "not_applicable") for r in valid_resolutions
            )
            if all_resolved and len(valid_resolutions) == len(prior_issues_for_gate):
                verdict = "clean"
                logger.info("verdict downgraded needs-human→clean: all priors resolved, no new issues")

        # §5 Telemetry — per-resolution observations
        for res_entry in valid_resolutions:
            episodic.write_observation(
                target_id,
                f"reviewer cycle {reviewer_count} prior_resolution: "
                f"index={res_entry['prior_index']} "
                f"status={res_entry['status']} "
                f"evidence={res_entry['evidence'][:120]}",
                extra_tags=[
                    "pm:reviewer-prior-resolution",
                    f"pm:cycle={reviewer_count}",
                    f"pm:status={res_entry['status']}",
                ],
            )

        # §5 Telemetry — rollup observation
        kept_count = len([i for i in issues if i.get("prior_index") in still_present_set])
        dropped_count = sum(
            1 for iss in verdict_info.get("issues", [])
            if iss.get("prior_index") is not None and iss.get("prior_index") not in still_present_set
        )
        episodic.write_observation(
            target_id,
            f"audit-gate: cycle={reviewer_count} priors={len(prior_issues_for_gate)} "
            f"still_present_kept={kept_count} "
            f"dropped_unsubstantiated={dropped_count} "
            f"verdict_downgrade={old_verdict}->{verdict}",
            extra_tags=["pm:reviewer-audit-gate", f"pm:cycle={reviewer_count}"],
        )
    # ---------------------------------------------------------------------------
    # End audit gate
    # ---------------------------------------------------------------------------

    if verdict == "clean":
        # Reviewer approved — surface per authority level
        cls.screen_verdict = "clean"
        cls.issues = issues
        if pm_authority == "hold":
            return Decision("hold_brief", payload)
        return Decision("advisory_brief", payload)

    if verdict == "fixable":
        if reviewer_count < budget:
            return Decision("dispatch_fixer_retry", {
                "pr": pr, "cls": cls, "issues": issues,
                "cycle": reviewer_count, "budget": budget,
            })
        # LOW-only exhaustion: advisory PRs with only low-severity issues remaining
        # emit advisory_brief (sweep-later framing) rather than review_exhausted_brief.
        # Hold authority always escalates regardless of severity.
        if (
            pm_authority != "hold"
            and issues
            and all(iss.get("severity", "high") == "low" for iss in issues)
        ):
            cls.screen_verdict = "fixable"
            cls.issues = issues
            return Decision("advisory_brief", payload)
        history = _collect_review_history(target_id, pr_number)
        return Decision("review_exhausted_brief", {
            "pr": pr, "cls": cls, "history": history,
        })

    # needs-human (or unknown verdict)
    cls.screen_verdict = verdict
    cls.issues = issues
    return Decision("hold_brief", payload)


def _has_pending_dispatch(target_id: str) -> bool:
    return any(d.get("status") == "pending" for d in load_dispatched(target_id))


# ---------------------------------------------------------------------------
# Act
# ---------------------------------------------------------------------------

def _pr_mergeable(repo_name: str, pr_number: int, owner: str | None = None) -> bool | None:
    """Check if a PR can merge cleanly via Forgejo.

    Returns:
      True   - PR can merge cleanly
      False  - PR has conflicts and cannot merge
      None   - Mergeability is indeterminate (Forgejo still computing, or get_pr failed)

    Tolerant of get_pr raising — returns None to allow fall-through to attempt-then-catch.
    """
    try:
        pr = _forgejo_get_pr(repo_name, pr_number, owner=owner)
        if pr:
            return pr.get("mergeable")
    except Exception:
        pass
    return None


def _act_needs_review(target_id: str, cls: authority.PRClassification, reason: str,
                      base_branch: str | None = None) -> str:
    """Route a PR with merge conflicts to NEEDS_REVIEW state (first-class, state-only).

    Writes a conflict-specific brief, marks PR classified, sets outstanding brief.
    Per spec: NEEDS_REVIEW is state-only (no buttons in this target); button wiring
    is the follow-on lapis-pm-needs-review-buttons-v0.
    """
    episodic.write_hold(
        target_id,
        f"PR #{cls.pr_number} cannot merge cleanly — {reason}.\n"
        f"Base branch: {base_branch or 'main'}\n"
        f"Title: {cls.title}\n{cls.html_url}\n\n"
        f"The branch has diverged from the base. Rebase and re-push, or merge manually.",
        extra_tags=[f"pm:repo={cls.repo}", f"pm:pr={cls.pr_number}", "pm:needs-review"],
    )
    b = brief.synthesize(
        target_id,
        trigger=f"PR #{cls.pr_number} needs review — cannot merge cleanly",
        query=f"PR #{cls.pr_number} merge conflict: {cls.title}",
        diff_snippet=cls.diff or None,
        screen_issues=None,
        notify=NotifyPriority.NORMAL,
    )
    _mark_pr_classified(target_id, cls.pr_number)
    _set_brief_outstanding(target_id, b, verified=True)
    return f"action:needs_review:pr={cls.pr_number}:reason={reason}"


def _act_merge(target_id: str, payload: dict) -> str:
    cls: authority.PRClassification = payload["classification"]
    pr: dict = payload.get("pr", {})
    repo_name, owner = _repo_owner(cls.repo)

    # Dry-run-merge discipline (guaardvark@51d9829c131d, merge_manager.check_conflicts):
    # never issue a merge we haven't proven applies cleanly. Forgejo already computes
    # mergeability — read it instead of merging blind. A not-mergeable PR is a
    # first-class NEEDS_REVIEW state, not a swallowed failure.
    mergeable = _pr_mergeable(repo_name, cls.pr_number, owner)
    if mergeable is False:
        base_branch = (pr.get("base") or {}).get("ref") or "main"
        return _act_needs_review(target_id, cls, reason="merge_conflict", base_branch=base_branch)
    # mergeable is True  -> clean path (unchanged below)
    # mergeable is None   -> indeterminate (still computing / field absent):
    #                        fall through to existing attempt-then-catch path
    #                        (no regression vs today). Per ⚑ Decision 2(A): emit the
    #                        distinct outcome string merge_attempted:mergeability_unknown
    #                        so audit record keeps the two epistemic states distinct.
    try:
        merge_and_deploy(repo_name, cls.pr_number, owner=owner)
    except Exception as e:
        episodic.write_hold(
            target_id,
            f"Auto-merge attempted for PR #{cls.pr_number} but failed: {e}",
            extra_tags=[f"pm:repo={cls.repo}"],
        )
        return f"action:merge_failed:{e}"

    # Success path — only reached if mergeable is True or None and merge succeeded
    episodic.write_merge(
        target_id,
        f"Auto-merged PR #{cls.pr_number} ({cls.title}) — "
        f"{cls.diff_loc} LOC, screen={cls.screen_verdict}\n{cls.html_url}",
        extra_tags=[f"pm:repo={cls.repo}", f"pm:pr={cls.pr_number}"],
    )
    _mark_pr_classified(target_id, cls.pr_number)
    if mergeable is None:
        return f"action:merge_attempted:mergeability_unknown:pr={cls.pr_number}"
    return f"action:auto_merge:pr={cls.pr_number}"


def _auto_resolve_record(target_id: str, pr_number: int) -> None:
    """Write audit trail after a conservative auto-resolve (observation + mem key)."""
    episodic.write_observation(
        target_id,
        f"Auto-resolved (conservative): merged PR #{pr_number} — reviewer clean, advisory, no held paths",
        extra_tags=["pm:auto-resolved", f"pm:pr={pr_number}"],
    )
    _mem().set(
        f"pm/auto-resolved/{target_id}/{pr_number}",
        f"merged PR #{pr_number} at advisory-clean auto-resolve",
        tags=["lapis-pm", "pm:auto-resolved"],
    )


def _act_brief(target_id: str, trigger: str, hold: bool, payload: dict) -> str:
    cls: authority.PRClassification = payload["classification"]
    hold_comment = None
    if hold:
        hold_comment = episodic.write_hold(
            target_id,
            f"PR #{cls.pr_number} held: {'; '.join(cls.reasons)}\n"
            f"Title: {cls.title}\n{cls.html_url}",
            extra_tags=[f"pm:repo={cls.repo}", f"pm:pr={cls.pr_number}"],
        )
    # For advisory (non-hold) briefs, derive a closed-form trigger string so
    # the synthesizer emits a pm:brief-options sibling with resolution buttons.
    effective_trigger = trigger
    if not hold:
        if cls.issues:
            effective_trigger = "advisory-screen-issue"
        else:
            effective_trigger = "advisory-clean"

    # U1 attestation hook (lapis-pm-autonomy-actuator-v0): pinned BEFORE the
    # shared TargetStore read so the read below reflects a same-tick
    # pm_verification grant (H1: persist-then-stale-read would make the grant
    # dead until a second tick). Never-raise; observation must never affect
    # the gate result or the brief already being built.
    if not hold and effective_trigger == "advisory-clean":
        _attestation_hook(target_id, cls, payload)

    # Single TargetStore read shared by the auto-resolve and FC hook blocks below.
    _target = TargetStore().get(target_id)

    # Conservative auto-resolve: merge unambiguous advisory-clean PRs without a gem.
    # Predicate is deterministic (no LLM). On any merge failure, falls through to
    # the normal brief/gem path — never swallows a brief.
    if not hold and effective_trigger == "advisory-clean":
        from . import auto_resolve as _ar
        if _target is not None:
            _should, _merge_opt = _ar.should_auto_resolve(
                cls, _target, _target.pm_repo or "",
            )
            if not _should and _merge_opt and "needs-pm-touch" in _merge_opt:
                episodic.write_observation(
                    target_id,
                    f"auto-resolve skipped: {_merge_opt} — routing to PM brief",
                    extra_tags=["pm:auto-resolve-skipped:verification", f"pm:pr={cls.pr_number}"],
                )
            if _should:
                try:
                    brief._act_merge_pr(target_id, cls.pr_number)
                    _auto_resolve_record(target_id, cls.pr_number)
                    _mark_pr_classified(target_id, cls.pr_number)
                    return f"action:auto_resolved:advisory_clean:pr={cls.pr_number}"
                except Exception as _ar_exc:
                    logger.warning(
                        "auto-resolve: merge failed for %s PR #%s (%s) — falling to brief/gem",
                        target_id, cls.pr_number, _ar_exc,
                    )
                    # fall through to normal brief/gem path

    # U2 precedent hook (lapis-pm-autonomy-actuator-v0): fires ONLY for
    # effective_trigger == 'advisory-clean' and not hold (the merge-safe state;
    # screen-issue and fixable-low forks keep the human in v0). Enforce mode
    # merges on a verified precedent match and returns the action string;
    # shadow mode records the would-be call and returns None. On any merge
    # exception it returns None — fall through to the normal brief path, never
    # swallow a brief.
    if not hold and effective_trigger == "advisory-clean":
        _precedent_action = _precedent_hook(target_id, cls, payload)
        if _precedent_action is not None:
            _mark_pr_classified(target_id, cls.pr_number)
            return _precedent_action

    reviewer_verdict_text: str | None = None
    reviewer_scope_text: str | None = None
    if effective_trigger == "advisory-clean" or hold:
        verdict_info = _last_review_verdict(target_id, cls.pr_number)
        if verdict_info:
            v = verdict_info.get("verdict", "?")
            conf = verdict_info.get("confidence", "?")
            n_issues = len(verdict_info.get("issues") or [])
            corr = verdict_info.get("corroboration_result") or {}
            corr_v = corr.get("verdict")
            # DoD-5 (lapis-pm-reviewer-leg-repair-v0): the brief carries
            # drift_class verbatim — the rev-1 brief rendered only
            # corroboration={verdict}, so a reader could not see the
            # per-pass drift class without opening the verdict JSON.
            corr_dc = corr.get("drift_class")
            parts = [f"local reviewer: verdict={v}", f"confidence={conf}", f"issues={n_issues}"]
            if corr_v:
                _corr_part = f"corroboration={corr_v}"
                if corr_dc:
                    _corr_part += f" (drift_class={corr_dc})"
                parts.append(_corr_part)
            # Leg 1 legibility (DoD #4): a reader must see degradation without
            # opening the verdict JSON. panel_starvation is additive — absent
            # on verdicts written before this landed, so guard with .get.
            _pstarv = verdict_info.get("panel_starvation") or {}
            _legs_down_list = _pstarv.get("legs_down") or []
            _raw_conf = _pstarv.get("confidence_raw", "?")
            # DoD-1 (lapis-pm-reviewer-leg-repair-v0): the DEGRADED PANEL
            # line carries the node2 error-class tail (the four named
            # classes: node2_unavailable / node2_aci_unverified /
            # node2_cap_refused / node2_client_import_error) so a genuine
            # node2 down is distinguishable from a wiring bug on a PM
            # surface (today only the episodic JSON did). Hoisted above the
            # fork: the DEGRADED line and the second_node loud line share
            # the read. Only the four allowlisted NODE2_ERROR_CLASSES
            # constants may render (containment scan of the node2 notes —
            # which are redacted at source); `unknown` when no class
            # matches — never empty parens, never free-form notes text.
            _n2_notes = (corr.get("node2_corroboration") or {}).get("notes") or ""
            _n2_class = ""
            for _ec in (
                "node2_client_import_error", "node2_aci_unverified",
                "node2_cap_refused", "node2_unavailable",
            ):
                if _ec in _n2_notes:
                    _n2_class = _ec
                    break
            if _pstarv.get("starved"):
                _legs_down = ", ".join(_legs_down_list) or "unknown"
                _n2_part = f"; node2={_n2_class}" if _n2_class else ""
                parts.append(
                    f"DEGRADED PANEL (legs down: {_legs_down}{_n2_part}; raw "
                    f"confidence {_raw_conf} attenuated to {conf}) — advisory "
                    f"only, not gated"
                )
            elif "second_node" in _legs_down_list:
                # AMENDED 2026-09-10 (decision/reviewer-single-leg-local-
                # phala-test-key): not starved (both local gate
                # legs up) but second_node (Phala TEE) absent — the loud
                # line. The verdict stands on local legs only; the absence
                # is flagged, not silently read as a healthy panel.
                parts.append(
                    f"second_node (Phala TEE) ABSENT ({_n2_class or 'unknown'}); "
                    f"raw confidence {_raw_conf} attenuated to {conf} — "
                    f"verdict stands on local legs only"
                )
            else:
                # DoD-1 (lapis-pm-reviewer-leg-repair-v0): a POSITIVE line
                # renders when not starved, so a healthy panel is
                # distinguishable from a legacy unannotated verdict.
                parts.append("panel: 3/3 legs reporting")
            _refuted = verdict_info.get("refuted_absence_findings") or []
            if _refuted:
                parts.append(f"refuted absence claims={len(_refuted)}")
            reviewer_verdict_text = "; ".join(parts)
            # D8 (attestation-contract-v0 leg 2): the reviewer's reported
            # verification scope (suite/files actually run + test count)
            # rides the brief's verdict surface verbatim. Absent on
            # verdicts written before this landed — guard with .get.
            _scope = verdict_info.get("scope")
            if _scope:
                reviewer_scope_text = str(_scope)

    # AC8: functional critic hook — fires when pm_verification == "agent-functional"
    functional_critic_text: str | None = None
    _pm_verification = (_target.data.get("pm_verification", "pm-live-test")
                        if _target else "pm-live-test")
    if _pm_verification == "agent-functional":
        try:
            from . import functional_critic as _fc
            _pr = payload.get("pr") or {}
            _head_sha = (_pr.get("head") or {}).get("sha", "")
            _spec_text = episodic.spec(target_id) or ""
            _diff_summary = (cls.diff or "")[:3000]
            _run_id = f"fc-{target_id}-pr{cls.pr_number}"
            if _head_sha:
                _verdict = _fc.run_functional_critic(
                    target_id=target_id,
                    pr_number=cls.pr_number,
                    head_sha=_head_sha,
                    repo=cls.repo,
                    spec_text=_spec_text,
                    run_id=_run_id,
                    diff_summary=_diff_summary,
                )
                _fc.write_verdict_artifact(_run_id, cls.pr_number, _verdict)
                functional_critic_text = _fc.verdict_summary_text(_verdict)
                # AC9: notify on partial or unverifiable
                _overall = _verdict.get("overall")
                if _overall in ("partial", "unverifiable"):
                    _ac_verdicts = _verdict.get("ac_verdicts") or []
                    if _ac_verdicts:
                        _nx_count = _fc.unexercised_ac_count(_verdict)
                        _nx_str = f"{_nx_count} unexercised AC(s)"
                    else:
                        # ac_verdicts is empty when GW was unavailable or worktree
                        # failed — every AC is unverifiable, not just 0
                        _nx_str = "all ACs unverifiable"
                    _brief_link = (cls.html_url or
                                   f"PR #{cls.pr_number} in {cls.repo}")
                    try:
                        from agents_core.notify import send_notification, Priority as _NP
                        send_notification(
                            message=(
                                f"Functional critic: overall={_overall} "
                                f"({_nx_str}). "
                                f"Spec: {target_id}. {_brief_link}"
                            ),
                            title=f"lapis-pm: functional critic {_overall}: {target_id}",
                            priority=_NP.NORMAL,
                        )
                    except Exception as _notif_exc:
                        logger.warning(
                            "functional_critic: AC9 notification failed: %s", _notif_exc
                        )
                episodic.write_observation(
                    target_id,
                    f"Functional critic: overall={_overall} for PR #{cls.pr_number}\n"
                    f"Run ID: {_run_id}",
                    extra_tags=[
                        f"pm:functional-critic:pr={cls.pr_number}",
                        f"pm:functional-critic:overall={_overall}",
                    ],
                )
            else:
                logger.warning(
                    "functional_critic: no head SHA for PR #%d — skipping critic",
                    cls.pr_number,
                )
        except Exception as _fc_exc:
            logger.warning(
                "functional_critic: hook failed for %s PR #%d (%s) — continuing to brief",
                target_id, cls.pr_number, _fc_exc,
            )

    # U2 fork-class persistence (lapis-pm-autonomy-actuator-v0, Design 6):
    # derive the deterministic fork shape at emission (classifier data is
    # fresh here) and persist it in the pm:brief-options sibling so
    # cmd_ratify can copy it without re-derivation or network. Never-raise —
    # a derivation failure leaves the sibling unchanged (fork_class=None).
    _fork_class: dict | None = None
    try:
        from . import precedent as _pc
        _fork_class = _pc.derive_fork_class(cls, cls.screen_verdict or "clean",
                                            repo=cls.repo)
    except Exception as e:
        logger.warning("fork-class derivation failed for %s: %s", target_id, e)

    b = brief.synthesize(
        target_id,
        trigger=effective_trigger,
        query=cls.title,
        diff_snippet=cls.diff or None,
        screen_issues=cls.issues or None,
        pr_number=cls.pr_number if not hold else None,
        notify=NotifyPriority.NORMAL if hold else None,
        reviewer_verdict_text=reviewer_verdict_text,
        reviewer_scope_text=reviewer_scope_text,
        functional_critic_text=functional_critic_text,
        fork_class=_fork_class,
    )
    _mark_pr_classified(target_id, cls.pr_number)
    _set_brief_outstanding(target_id, b, verified=True)
    if hold:
        kind = "hold"
        # lapis-pm-hold-shadow-observer-v0: act-free shadow record of this
        # PR hold (SHADOW MODE ONLY, Erah 2026-08-12). Both hold_comment_id
        # and brief_comment_id are known only here, at the end of the hold
        # branch — never-raise; cannot affect the brief already emitted.
        from . import hold_shadow as _hold_shadow
        _hold_shadow.observe_hold_fact(
            target_id=target_id,
            pr_number=cls.pr_number,
            repo=cls.repo,
            hold_reasons=cls.reasons,
            hold_comment_id=hold_comment.id if hold_comment is not None else "",
            brief_comment_id=b.comment_id,
            pm_authority=_target.pm_authority if _target is not None else "hold",
            spec_bound_ts=_spec_bound_ts(target_id),
            has_issues=bool(cls.issues),
        )
    elif cls.issues:
        kind = "advisory_screen_issue"
    else:
        kind = "advisory_clean"
    return f"action:brief_emitted:kind={kind}:cid={b.comment_id}"


def _act_retry(target_id: str, dispatch_record: dict) -> str:
    """Redispatch a failed shaped agent (≤MAX_DISPATCH_RETRIES)."""
    agent_type = dispatch_record.get("agent_type", "scout")
    intent = dispatch_record.get("intent", "(no intent)")
    spec_summary = episodic.spec_summary(target_id)
    user_prompt = (
        f"Previous attempt failed. Retry intent: {intent}\n"
        f"Note: previous failure observation is in the thread comments."
    )
    vars_ = {
        "target_id": target_id,
        "spec_summary": spec_summary,
        "repo": dispatch_record.get("repo", ""),
        "question": intent,
        "pr_number": dispatch_record.get("pr_number", ""),
        "slug": dispatch_record.get("slug", "retry"),
        "intent_block": _intent_artifact.dispatch_block(target_id),
    }
    steer.inject_overlay(target_id, vars_, agent_type)
    _ensure_dispatch_owned(vars_.get("repo", ""))
    res = _SHAPER.dispatch(agent_type, target_id, user_prompt, vars_=vars_)
    new_record = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": agent_type,
        "intent": intent,
        "repo": dispatch_record.get("repo", ""),
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": dispatch_record.get("retry_count", 0) + 1,
        "parent_gpu_id": dispatch_record.get("gpu_id"),
    }
    append_dispatched(target_id, new_record)
    _check_calcification(target_id)
    episodic.write_retry(
        target_id,
        f"Retry #{new_record['retry_count']} dispatched: {agent_type} → {res.task_id}\n"
        f"Intent: {intent}",
        extra_tags=[f"pm:gpu={res.task_id}", f"pm:agent={agent_type}"],
    )
    return f"action:fixer_dispatched:source=init:dispatch={res.task_id}"


def _build_prior_review_text(target_id: str, pr_number: int, cycle: int) -> str:
    """Build the '## Prior review context' block for same-reviewer mode, cycle > 1.

    Shared by _act_dispatch_reviewer (daemon path) and force_dispatch (manual
    --force-dispatch path) so the ``reviewer`` template's {prior_review}
    placeholder is populated identically regardless of dispatch path.
    """
    if cycle <= 1:
        return ""
    prior = _review_verdict_for_cycle(target_id, pr_number, cycle - 1)
    if not prior:
        return ""
    prior_issues = prior.get("issues", [])
    indexed_issues = "\n".join(
        f"  [{i}] {iss.get('severity', '?').upper()} {iss.get('path', '?')} — {iss.get('note', '')}"
        for i, iss in enumerate(prior_issues)
    ) if prior_issues else "  (none)"
    return (
        f"\n## Prior review context (cycle {cycle - 1})\n\n"
        f"The prior reviewer cycle returned this verdict on an EARLIER state of this branch:\n\n"
        f"Verdict: {prior.get('verdict')}\n"
        f"Prior issues (indexed):\n{indexed_issues}\n\n"
        f"The diff in the user prompt is the CURRENT state. Your task is to classify\n"
        f"EACH prior issue against the current diff. Then return your own fresh\n"
        f"verdict on the current diff.\n\n"
        f"For each prior issue, you MUST emit a `prior_resolution` entry with:\n"
        f'  - "prior_index": the index above\n'
        f'  - "status": "addressed" | "still_present" | "not_applicable"\n'
        f'  - "evidence": for "still_present", a current-diff line/path citation;\n'
        f'                for "addressed", the line/path that fixes it;\n'
        f'                for "not_applicable", a one-sentence reason\n'
        f'                (empty string is NOT acceptable for any status)\n\n'
        f"Then your `issues` array must contain ONLY:\n"
        f'  - prior issues you classified as "still_present" (re-stated, with the\n'
        f"    same path/severity, but `note` updated to reference the current-diff\n"
        f"    evidence), AND\n"
        f"  - any new issues you find in the current diff that were not in the\n"
        f"    prior set.\n\n"
        f'Issues you classified as "addressed" or "not_applicable" must NOT appear\n'
        f"in `issues`. Reviewer cycles are explicit deltas, not stateless re-reads.\n\n"
        f'When re-stating a prior issue in `issues`, include `prior_index: <i>`\n'
        f"pointing to the prior set; new issues omit `prior_index`.\n"
    )


def _act_dispatch_reviewer(target_id: str, pr: dict, cls: authority.PRClassification,
                            mode: str = "same", cycle: int = 1) -> str:
    """Dispatch a reviewer agent for the given PR."""
    pr_number = pr["number"]
    repo = cls.repo
    spec_summary = episodic.spec_summary(target_id)

    agent_type = "reviewer_fresh" if mode == "fresh" else "reviewer"

    # Build prior_review context for same-reviewer mode
    prior_review_text = _build_prior_review_text(target_id, pr_number, cycle) if mode == "same" else ""

    existing_branch = (pr.get("head") or {}).get("ref") or f"lapis/{target_id}/pr{pr_number}"
    base_branch = (pr.get("base") or {}).get("ref") or "main"

    vars_: dict = {
        "target_id": target_id,
        "spec_summary": spec_summary,
        "repo": repo,
        "repo_cwd": Shaper.resolve_repo_cwd(repo),
        "pr_number": pr_number,
        "slug": f"pr{pr_number}-review-c{cycle}",
        "question": f"review PR #{pr_number}",
        "prior_review": prior_review_text,
        "existing_branch": existing_branch,
        "base_branch": base_branch,
        "intent_block": _intent_artifact.dispatch_block(target_id),
        "absence_grounding_block": _absence_grounding_block_for(agent_type),
    }

    # Get the diff for the reviewer prompt
    try:
        from agents_core.forgejo import get_pr_diff as _get_diff
        diff_text = _get_diff(repo, pr_number)
        if len(diff_text) > authority.DIFF_INLINE_CAP:
            diff_text = diff_text[:authority.DIFF_INLINE_CAP] + "\n\n... (diff truncated)"
    except Exception:
        diff_text = "(diff unavailable)"

    if mode == "same" and cycle >= 2:
        schema_extra = (
            ', "prior_resolution": ['
            '{"prior_index": <int>, "status": "addressed"|"still_present"|"not_applicable", '
            '"evidence": "<non-empty string>"}]'
            " (required when prior issues were provided); "
            'issues that re-state a prior issue carry "prior_index": <int>; '
            "new issues omit prior_index"
        )
    else:
        schema_extra = ""
    pr_body = (pr.get("body") or "").strip()
    pr_body_section = f"\n\n**PR Description:**\n{pr_body}" if pr_body else ""
    user_prompt = (
        f"Review PR #{pr_number} in {repo}. This is reviewer cycle {cycle}.\n\n"
        f"```diff\n{diff_text}\n```"
        f"{pr_body_section}\n\n"
        f'Return JSON: {{"verdict": "clean" | "fixable" | "needs-human", '
        f'"issues": [{{"severity": "high"|"med"|"low", "path": "...", "note": "..."'
        f'{"," if schema_extra else ""}{"prior_index?: <int>" if schema_extra else ""}'
        f'}}]{schema_extra}, "confidence": 0.0-1.0}}'
    )

    # Increment the PR+cycle-bound attempt ceiling counter (counts this
    # dispatch regardless of whether it later succeeds or fails).
    _increment_reviewer_attempt(target_id, pr_number, cycle)

    steer.inject_overlay(target_id, vars_, agent_type)
    _ensure_dispatch_owned(vars_.get("repo", ""))
    res = _SHAPER.dispatch(agent_type, target_id, user_prompt, vars_=vars_)

    # D2/D5 provenance: a verdict whose record does not say which model
    # produced it is not auditable. Look up the registry-declared model/seat
    # for this agent_type (never hardcode — the registry is the single
    # source of truth). Post-lapis-pm-phala-test-key, every
    # fresh-review verdict comes from the single local reviewer_fresh seat —
    # there is no longer a contractor counterpart to distinguish. NOTE:
    # "reviewer_fresh_contractor" still appears in _REVIEWER_AGENT_TYPES and
    # the family mapping above (D4) as a read-only parsing artifact for
    # historical dispatch records; no live code path in this function (or
    # anywhere else) produces that agent_type anymore.
    try:
        shaped_agent = _SHAPER.get_agent(agent_type)
    except Exception:
        shaped_agent = None
    record = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": agent_type,
        "intent": f"review PR #{pr_number} cycle {cycle}",
        "repo": repo,
        "pr_number": pr_number,
        "cycle": cycle,
        "mode": mode,
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": 0,
        "model": shaped_agent.model if shaped_agent else None,
    }
    append_dispatched(target_id, record)

    # Dead-seat visibility (lapis-pm-phala-test-key, D3.4):
    # the removed gap-spanner used to silently reroute a dead
    # reviewer_fresh seat to the contractor seat. There is no alternate
    # seat anymore, so a prior seat_no_tool_calls failure on this exact
    # PR/cycle must be surfaced here instead of hidden — this dispatch is
    # about to retry the same dead seat, and an operator directive (wait
    # for seat recovery, or move the PR) is the only remedy.
    _dead_seat_note = ""
    if agent_type == "reviewer_fresh":
        _prior_state = _reviewer_attempt_state(target_id, pr_number, cycle)
        if _prior_state["last_infra_reason"] == "seat_no_tool_calls":
            _dead_seat_note = (
                f"Reviewer seat dead: prior attempt on PR #{pr_number} cycle "
                f"{cycle} failed with seat_no_tool_calls. No alternate seat "
                f"is configured — this dispatch retries the same seat. "
                f"Remedy is an operator directive (re-dispatch after seat "
                f"recovery, or a directive to move the PR).\n"
            )
    episodic.write_dispatch(
        target_id,
        f"{_dead_seat_note}"
        f"Reviewer dispatched (cycle {cycle}, mode={mode}): {agent_type} → {res.task_id}\n"
        f"PR #{pr_number}: {pr.get('title', '')}",
        extra_tags=[
            f"pm:repo={repo}",
            f"pm:pr={pr_number}",
            f"pm:reviewer:pr={pr_number}:cycle={cycle}:verdict=pending",
        ],
    )
    _check_calcification(target_id)
    return f"action:reviewer_dispatched:pr={pr_number}:cycle={cycle}"


def _act_dispatch_fixer_retry(target_id: str, payload: dict) -> str:
    """Dispatch a fixer_retry agent to fix reviewer-flagged issues on an existing PR."""
    pr = payload["pr"]
    cls = payload["cls"]
    issues = payload["issues"]
    cycle = payload.get("cycle", 1)  # reviewer cycle that returned fixable

    pr_number = pr["number"]
    pr_branch = (pr.get("head") or {}).get("ref") or f"lapis/{target_id}/pr{pr_number}"

    issues_text = "\n".join(
        f"- [{i.get('severity', '?')}] {i.get('path', '?')}: {i.get('note', '')}"
        for i in issues
    ) or "(no issues listed)"

    spec_summary = episodic.spec_summary(target_id)

    vars_: dict = {
        "target_id": target_id,
        "spec_summary": spec_summary,
        "repo": cls.repo,
        "repo_cwd": Shaper.resolve_repo_cwd(cls.repo),
        "pr_number": pr_number,
        "slug": f"pr{pr_number}-fix-c{cycle}",
        "existing_branch": pr_branch,
        "question": f"fix reviewer issues on PR #{pr_number}",
        "intent_block": _intent_artifact.dispatch_block(target_id),
    }

    user_prompt = (
        f"PR #{pr_number} reviewer returned `fixable` on cycle {cycle}. "
        f"Fix the listed issues on existing branch `{pr_branch}`.\n\n"
        f"Issues:\n{issues_text}\n\n"
        f"Push to the existing branch — do NOT create a new branch or new PR."
    )

    steer.inject_overlay(target_id, vars_, "fixer_retry")
    _ensure_dispatch_owned(vars_.get("repo", ""))
    res = _SHAPER.dispatch("fixer_retry", target_id, user_prompt, vars_=vars_)
    _check_calcification(target_id)

    record = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": "fixer_retry",
        "intent": f"fix PR #{pr_number} after reviewer cycle {cycle}",
        "repo": cls.repo,
        "pr_number": pr_number,
        "cycle": cycle,
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": 0,
    }
    append_dispatched(target_id, record)

    episodic.write_dispatch(
        target_id,
        f"Fixer retry dispatched (reviewer cycle {cycle}): fixer_retry → {res.task_id}\n"
        f"PR #{pr_number} issues:\n{issues_text}",
        extra_tags=[
            f"pm:repo={cls.repo}",
            f"pm:pr={pr_number}",
            "pm:fixer-retry",
        ],
    )
    return f"action:fixer_dispatched:source=retry:pr={pr_number}:cycle={cycle}"


def _noop_retry_escalation_marker_key(target_id: str, pr_number: int, cycle: int) -> str:
    return f"pm/noop-retry-escalation/{target_id}/pr={pr_number}/cycle={cycle}/recorded"


# Simultaneous-blip window (lapis-pm-panel-leg-survival-v0, Change 3; Council
# 2026-09-01-110546-fdaec4, Marcus's suggestion): two starved verdicts with
# DISJOINT legs_down closer than this apart are labelled "transient panel
# degradation (simultaneous blip)" — a common transient cause is the working
# hypothesis and a simple retry is the suggested action. Farther apart (the
# common case — review cycles are separated by a fixer round-trip, minutes+)
# they are labelled "intermittent panel instability". 60s = one tick.
PANEL_BLIP_WINDOW_SECONDS = 60

# Cross-cycle window for the degraded predicate: the last min(3, available)
# reviewer verdicts for the PR, including the current cycle.
_PANEL_DEGRADED_WINDOW = 3


def _reviewer_verdicts_by_cycle(target_id: str, pr_number: int) -> dict[int, dict]:
    """Map reviewer cycle -> verdict dict for this PR, in `all_comments` order.

    Same episodic source and parsing rules as _review_verdict_for_cycle (tag
    prefix `pm:reviewer:pr={N}:cycle={C}:verdict=`, JSON body after the first
    newline, tag-fallback path included). The tag-fallback verdict has no
    `panel_starvation` block, so `verdict_is_starved` counts it as starved via
    the missing-blocks rule — gate-consistent.
    """
    out: dict[int, dict] = {}
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            m = re.match(
                r"^pm:reviewer:pr=%d:cycle=(\d+):verdict=(.+)$" % pr_number, t
            )
            if not m:
                continue
            cyc, verdict_val = int(m.group(1)), m.group(2)
            if verdict_val == "pending":
                continue
            try:
                json_part = c.content.split("\n", 1)[-1].strip()
                out[cyc] = json.loads(json_part)
            except (json.JSONDecodeError, IndexError):
                out[cyc] = {"verdict": verdict_val, "issues": [], "confidence": 0.0}
    return out


def _verdict_legs_down(verdict: dict) -> list[str]:
    """A verdict's `legs_down`: the stored `panel_starvation` annotation when
    present, else recomputed via panel_starvation.starved_legs() (pre-annotation
    verdicts)."""
    from . import panel_starvation as _panel_starvation
    pstarv = verdict.get("panel_starvation")
    if pstarv is not None:
        legs = pstarv.get("legs_down") or []
        if legs:
            return list(legs)
    return _panel_starvation.starved_legs(verdict)


def _panel_blip_window_seconds() -> float:
    """Call-time read so tests can monkeypatch the value (same discipline as
    the LOCAL_LLM_URL/PHALA_API_KEY env reads)."""
    return float(os.environ.get("PANEL_BLIP_WINDOW_SECONDS", PANEL_BLIP_WINDOW_SECONDS))


def _parse_comment_ts(c) -> datetime | None:
    """Best-effort parse of a comment's `ts` field (ISO 8601, naive assumed
    UTC). Returns None when unparseable — the caller treats that as
    temporally-separated (the conservative label)."""
    try:
        return datetime.fromisoformat(str(c.ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _panel_degradation_label(
    starved: list[dict],
    cycle_by_verdict: dict[int, int],
    comment_ts_by_cycle: dict[int, datetime | None],
) -> str:
    """Diagnostic label for a degraded panel (Council reclassification,
    lapis-pm-panel-leg-survival-v0 Change 3):

    - persistent: the same leg appears in the legs_down of >= 2 starved
      verdicts — "persistent panel degradation", inspect that leg.
    - intermittent: disjoint legs, temporally separated (beyond the blip
      window) — "intermittent panel instability", inspect legs individually.
    - simultaneous blip: disjoint legs, starved verdicts within
      PANEL_BLIP_WINDOW_SECONDS of each other — "transient panel degradation
      (simultaneous blip)", a simple retry is the suggested action.
    """
    leg_counts: dict[str, int] = {}
    for v in starved:
        for leg in set(_verdict_legs_down(v)):
            leg_counts[leg] = leg_counts.get(leg, 0) + 1
    if any(n >= 2 for n in leg_counts.values()):
        return "persistent panel degradation"

    # Disjoint legs across the starved verdicts: are they temporally separated?
    timestamps = [
        comment_ts_by_cycle.get(cycle_by_verdict.get(id(v), -1))
        for v in starved
    ]
    known = [t for t in timestamps if t is not None]
    if len(known) >= 2 and min(known) and max(known):
        span = (max(known) - min(known)).total_seconds()
        if span <= _panel_blip_window_seconds():
            return "transient panel degradation (simultaneous blip)"
    return "intermittent panel instability"


def _escalate_noop_retry_if_degraded(target_id: str, rec: dict, pr_num) -> str | None:
    """Leg 3: escalate a no-op fixer_retry when the verdict that drove it was
    starved (Leg 1) or carried a refuted absence claim (Leg 2).

    Ontology (Council reclassification, run 2026-09-01-110546-fdaec4): a
    starved verdict is a VOID — the panel could not corroborate — not a
    FALSEHOOD. No brief in any case may conclude "the verdict was false"; the
    correct claim is about the process (corroboration failed), never about the
    content (the verdict is wrong). The PR #220 incident remains the worked
    example for the refuted-absence-claim branch (Leg 2); the starved branch
    now reads cross-cycle context before drawing a conclusion.

    Degraded predicate (broadened, Council): the panel is degraded and the
    verdict unreliable iff at least two of the last min(3, available)
    reviewer verdicts for this PR (including the current cycle) are starved —
    same-leg or disjoint alike. The brief directs the operator to suspend
    trust in the verdict and not gate on it (panel repair / PM second
    opinion); it carries its own premise (starved count, cycle numbers, union
    of legs_down) and a diagnostic label (persistent / intermittent /
    simultaneous blip). R1 is untouched: this conclusion never gates or
    auto-dispatches.

    Reuses the existing outstanding-brief mechanism (R3) rather than
    inventing a new state or surface — same shape as the
    reviewer_infra_budget_exhausted worked example: a distinct named
    condition, an idempotent marker so it fires exactly once per pr+cycle,
    and a human-reachable escalation via brief.synthesize +
    _set_brief_outstanding.

    Returns the action string if an escalation fired, else None (nothing to
    escalate — verdict was healthy and unrefuted — or already recorded).
    """
    if pr_num is None:
        return None
    try:
        pr_number = pr_num if isinstance(pr_num, int) else int(pr_num)
    except (TypeError, ValueError):
        return None
    cycle = rec.get("cycle", 1)

    verdict_info = _review_verdict_for_cycle(target_id, pr_number, cycle)
    if verdict_info is None:
        return None

    from . import panel_starvation as _panel_starvation
    starved = _panel_starvation.verdict_is_starved(verdict_info)
    refuted = verdict_info.get("refuted_absence_findings") or []
    if not starved and not refuted:
        return None

    marker_key = _noop_retry_escalation_marker_key(target_id, pr_number, cycle)
    if _mem().get(marker_key):
        return None

    # Cross-cycle window (Change 3): the last min(3, available) reviewer
    # verdicts for this PR, including the current cycle, in all_comments
    # order. The current cycle's verdict is the one just read.
    by_cycle = _reviewer_verdicts_by_cycle(target_id, pr_number)
    window_verdicts = list(by_cycle.values())[-_PANEL_DEGRADED_WINDOW:]
    current_cycle = cycle
    # The current cycle's verdict (just read) is always in the window — the
    # window is the last min(3, available) verdicts INCLUDING this cycle.
    if verdict_info not in window_verdicts:
        window_verdicts = window_verdicts + [verdict_info]
    window_verdicts = window_verdicts[-_PANEL_DEGRADED_WINDOW:]
    starved_in_window = [
        v for v in window_verdicts if _panel_starvation.verdict_is_starved(v)
    ]

    # Cycle number per verdict in the window (for the brief's premise).
    cycle_by_verdict: dict[int, int] = {}
    for c, v in by_cycle.items():
        cycle_by_verdict.setdefault(id(v), c)
    if id(verdict_info) not in cycle_by_verdict:
        cycle_by_verdict[id(verdict_info)] = current_cycle

    # Comment timestamps per cycle (for the blip-window label).
    comment_ts_by_cycle: dict[int, datetime | None] = {}
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            m = re.match(r"^pm:reviewer:pr=%d:cycle=(\d+):verdict=" % pr_number, t)
            if m:
                comment_ts_by_cycle.setdefault(int(m.group(1)), _parse_comment_ts(c))

    label: str | None = None
    if starved and len(starved_in_window) >= 2:
        # Degraded: >= 2 of the window starved, same-leg or disjoint alike.
        label = _panel_degradation_label(starved_in_window, cycle_by_verdict, comment_ts_by_cycle)

    reasons = []
    if starved:
        legs = ", ".join(_verdict_legs_down(verdict_info)) or "unknown"
        reasons.append(f"panel starved on this cycle (legs down: {legs})")
    if refuted:
        reasons.append(f"{len(refuted)} absence claim(s) refuted against the PR-head file")
    reason_text = "; ".join(reasons)

    if starved and label:
        starved_cycles = sorted(
            cycle_by_verdict.get(id(v), -1) for v in starved_in_window
        )
        cycle_range = (
            f"{starved_cycles[0]}" if len(starved_cycles) == 1
            else f"{starved_cycles[0]}-{starved_cycles[-1]}"
        )
        legs_union = sorted({
            leg for v in starved_in_window for leg in _verdict_legs_down(v)
        })
        legs_union_text = ", ".join(legs_union) or "unknown"
        premise = (
            f"panel starved on {len(starved_in_window)} of last {len(window_verdicts)} "
            f"verdicts, PR #{pr_number} (cycles {cycle_range}), legs down: "
            f"{legs_union_text}"
        )
        if label == "persistent panel degradation":
            diagnosis = (
                "persistent panel degradation — a leg is down across multiple "
                "cycles; inspect that leg and repair the panel"
            )
        elif label == "transient panel degradation (simultaneous blip)":
            diagnosis = (
                "transient panel degradation (simultaneous blip) — a common "
                "transient cause is the working hypothesis; a simple retry is "
                "the suggested action"
            )
        else:
            diagnosis = (
                "intermittent panel instability — disjoint legs across cycles; "
                "inspect legs individually"
            )
        hold_text = (
            f"PR #{pr_number} fixer retry (cycle {cycle}) made no code or "
            f"description change, against a verdict that was starved. "
            f"{premise}. {diagnosis}. Suspend trust in the verdict and do not "
            f"gate on it — it is uncorroborated, not proven wrong; panel "
            f"repair or PM second opinion, not an override of a false verdict."
        )
    elif starved:
        # Single-cycle starved: the void framing without the degraded label.
        legs = ", ".join(_verdict_legs_down(verdict_info)) or "unknown"
        hold_text = (
            f"PR #{pr_number} fixer retry (cycle {cycle}) made no code or "
            f"description change, against a verdict that was starved. Panel "
            f"starved on this cycle (legs down: {legs}); this verdict is "
            f"uncorroborated — do not gate on it; it stands as advisory."
        )
    else:
        hold_text = (
            f"PR #{pr_number} fixer retry (cycle {cycle}) made no code or "
            f"description change, against a verdict that was {reason_text}. "
            f"The retry found nothing to change; a real stall has unaddressed "
            f"valid issues, this has valid-looking invalid ones. Needs human "
            f"judgment on the refuted finding(s)."
        )

    episodic.write_hold(
        target_id,
        content=hold_text,
        extra_tags=[f"pm:pr={pr_number}", "pm:noop-retry-degraded-verdict"],
    )
    b = brief.synthesize(
        target_id,
        trigger=(
            f"No-op fixer retry against a starved or refuted verdict on PR "
            f"#{pr_number} cycle {cycle} ({reason_text}) — human judgment needed"
        ),
        query=(
            f"PR #{pr_number} no-op retry: verdict uncorroborated or refuted "
            f"({reason_text})"
        ),
        pr_number=pr_number,
        notify=NotifyPriority.HIGH,
    )
    _set_brief_outstanding(target_id, b)
    _mem().set(
        marker_key,
        json.dumps({
            "target_id": target_id,
            "pr_number": pr_number,
            "cycle": cycle,
            "reasons": reasons,
            "degraded_label": label,
            "comment_id": b.comment_id,
        }),
        tags=["lapis-pm", "noop-retry-escalation"],
    )
    return f"action:noop_retry_degraded_verdict_brief:pr={pr_number}:cycle={cycle}:cid={b.comment_id}"


def _act_brief_review_exhausted(target_id: str, payload: dict) -> str:
    """Human brief when review cycle budget is exhausted."""
    pr = payload["pr"]
    cls = payload["cls"]
    history = payload.get("history", [])

    history_text = "\n\n".join(
        f"Cycle {h.get('cycle')}: verdict={h.get('verdict')}\n"
        f"Issues: {json.dumps(h.get('issues', []))}"
        for h in history
    ) or "(no history)"

    episodic.write_hold(
        target_id,
        f"PR #{cls.pr_number} review budget exhausted after {len(history)} cycle(s).\n"
        f"Title: {cls.title}\n{cls.html_url}\n\nHistory:\n{history_text}",
        extra_tags=[
            f"pm:repo={cls.repo}",
            f"pm:pr={cls.pr_number}",
            "pm:review-exhausted",
        ],
    )
    b = brief.synthesize(
        target_id,
        trigger=f"Review budget exhausted for PR #{cls.pr_number} — human judgment needed",
        query=f"PR #{cls.pr_number} review exhausted: {cls.title}",
        diff_snippet=cls.diff or None,
        screen_issues=None,
        notify=NotifyPriority.HIGH,
    )
    _set_brief_outstanding(target_id, b)
    _mark_pr_classified(target_id, cls.pr_number)
    return f"action:review_exhausted_brief:cid={b.comment_id}"


def _reviewer_attempt_ceiling_marker_key(target_id: str, pr_number: int, cycle: int) -> str:
    return f"pm/reviewer-attempt-ceiling/{target_id}/pr={pr_number}/cycle={cycle}/recorded"


def _reviewer_infra_budget_marker_key(target_id: str, pr_number: int, cycle: int) -> str:
    return f"pm/reviewer-infra-budget/{target_id}/pr={pr_number}/cycle={cycle}/recorded"


# The human-judgment escape hatch that needs no automated verdict at all.
# D2 requires every ceiling/infra-budget pause to name this — #834 and #835
# both had briefs available the entire time they were wedged, and nothing
# said so.
_PM_PR_REVIEW_ESCAPE = (
    "This does not require a reviewer verdict to resolve: walk the "
    "outstanding brief via `pm-pr-review` for a human-judgment call on the PR."
)


def _fallback_reviewer_fail_reason(target_id: str, pr_number: int, cycle: int) -> str | None:
    """D3: when no per-attempt reason was recorded — e.g. the reviewer job
    crashed before writing an output file, so the queue-reconcile path (not
    the out_path-based encode path) owned the terminal flip and the ordering
    left last_reason unset for that race — fall back to the last failed
    reviewer dispatch record's `error` field, which the queue populates from
    the job's own stderr and reliably holds the real cause even when no
    output file was ever written."""
    records = [
        r for r in load_dispatched(target_id)
        if r.get("agent_type") in _REVIEWER_AGENT_TYPES
        and r.get("pr_number") == pr_number
        and r.get("cycle", 1) == cycle
        and r.get("status") == "failed"
        and r.get("error")
    ]
    if not records:
        return None
    return records[-1]["error"]


def _act_reviewer_attempt_ceiling_pause(target_id: str, payload: dict) -> str:
    """Auto-pause the target and emit exactly ONE structured record when a
    reviewer's per-PR-cycle attempt count (excluding infra-classified
    non-runs, see D1) exceeds the ceiling without ever producing a completed
    verdict (lapis-pm-reviewer-attempt-ceiling-v0).

    Per the boundary ruling: pause is the audible state (not silent-stop),
    the record must be machine-consumable, and this must NOT auto-unpause —
    the only escape hatch is `lapis-pm clear-reviewer-attempts <target_id>`
    followed by an explicit `lapis-pm resume <target_id>`.
    """
    pr_number = payload["pr_number"]
    cycle = payload["cycle"]

    # Idempotent: emit the record at most once per pr+cycle. The tick loop
    # already early-returns "noop:paused" for a paused target, so this is a
    # defensive belt-and-suspenders guard, not the primary mechanism.
    marker_key = _reviewer_attempt_ceiling_marker_key(target_id, pr_number, cycle)
    if _mem().get(marker_key):
        return "action:reviewer_attempt_ceiling:already_recorded"

    attempts = payload["attempts"]
    ceiling = payload["ceiling"]
    # D3: a ceiling pause must never render reported_reason=None. If no
    # per-attempt reason was recorded, fall back to the dispatch record's
    # `error` field before giving up.
    reported_reason = payload.get("reported_reason") or _fallback_reviewer_fail_reason(
        target_id, pr_number, cycle,
    )
    clear_cmd = f"lapis-pm clear-reviewer-attempts {target_id}"
    # This is the genuine-failure ceiling (infra attempts are excluded from
    # `attempts` by _reviewer_attempt_ceiling_check), so clear-reviewer-attempts
    # is a legitimate remedy here — a reviewer producing bad/unparseable
    # verdicts is exactly what clearing the counters and retrying can fix.
    reason_clause = (
        f"reported_reason={reported_reason!r}"
        if reported_reason
        else "no reason was reported by any failing attempt (unresolved)"
    )

    record = {
        "target_id": target_id,
        "pr_number": pr_number,
        "cycle": cycle,
        "attempts": attempts,
        "ceiling": ceiling,
        # As-reported by the failing reviewer leg — NOT a verified cause.
        # This mechanism does not investigate why the reviewer failed; it
        # only counts that it did, N times in a row. See DoD 3b.
        "reported_reason": reported_reason,
        "reported_reason_note": "as-reported by the failing leg; unverified by this mechanism",
        "paused": True,
        "clear_command": clear_cmd,
        "escape_hatch": _PM_PR_REVIEW_ESCAPE,
    }

    store = TargetStore()
    target = store.get(target_id)
    if target is not None:
        target.set_paused(
            True,
            reason=(
                f"reviewer-attempt-ceiling: PR #{pr_number} cycle {cycle} hit "
                f"{attempts}/{ceiling} attempts with no completed verdict "
                f"({reason_clause}, unverified). "
                f"Clear with `{clear_cmd}`, then `lapis-pm resume {target_id}`. "
                f"{_PM_PR_REVIEW_ESCAPE}"
            ),
        )
        target.save()

    episodic.write_observation(
        target_id,
        f"Reviewer attempt ceiling reached on PR #{pr_number} cycle {cycle} "
        f"({attempts}/{ceiling} attempts, no completed verdict) — target auto-paused.\n"
        f"{json.dumps(record, indent=2)}",
        extra_tags=[
            "pm:reviewer-attempt-ceiling",
            f"pm:pr={pr_number}",
            f"pm:reviewer-attempt-ceiling:pr={pr_number}:cycle={cycle}",
        ],
    )
    _mem().set(marker_key, json.dumps(record),
               tags=["lapis-pm", "reviewer-attempt-ceiling", f"target={target_id}"])
    return f"action:reviewer_attempt_ceiling_paused:pr={pr_number}:cycle={cycle}:attempts={attempts}"


def _act_reviewer_infra_budget_pause(target_id: str, payload: dict) -> str:
    """Auto-pause the target when the *infrastructure* retry budget (D1) is
    exhausted — repeated reviewer non-runs (GravityWell not serving /
    unreachable), never a reviewer that ran and produced a bad verdict.

    Deliberately a distinct action from _act_reviewer_attempt_ceiling_pause:
    per D2, an infra-caused pause must name the infrastructure cause and must
    NOT recommend `clear-reviewer-attempts` — clearing the ceiling counters
    cannot fix an unavailable GravityWell, and recommending it here is
    exactly the "retry, not a repair" trap that wedged #834 on 2026-08-03.
    """
    pr_number = payload["pr_number"]
    cycle = payload["cycle"]

    marker_key = _reviewer_infra_budget_marker_key(target_id, pr_number, cycle)
    if _mem().get(marker_key):
        return "action:reviewer_infra_budget_exhausted:already_recorded"

    infra_attempts = payload["infra_attempts"]
    infra_budget = payload["infra_budget"]
    reported_reason = payload.get("reported_reason") or _fallback_reviewer_fail_reason(
        target_id, pr_number, cycle,
    )
    reason_clause = reported_reason or "infrastructure non-run (reason unresolved)"

    record = {
        "target_id": target_id,
        "pr_number": pr_number,
        "cycle": cycle,
        "infra_attempts": infra_attempts,
        "infra_budget": infra_budget,
        "reported_reason": reported_reason,
        "reported_reason_note": "as-reported by the failing leg; unverified by this mechanism",
        "cause_class": "infrastructure",
        "paused": True,
        "escape_hatch": _PM_PR_REVIEW_ESCAPE,
    }

    store = TargetStore()
    target = store.get(target_id)
    if target is not None:
        target.set_paused(
            True,
            reason=(
                f"reviewer-infra-budget: PR #{pr_number} cycle {cycle} hit "
                f"{infra_attempts}/{infra_budget} infrastructure non-runs "
                f"({reason_clause}) — the reviewer never ran, this is not a "
                f"bad review. Clearing reviewer attempts will not help; this "
                f"is an infrastructure availability problem "
                f"(out of scope for lapis-pm — see the gw_not_serving/doorman "
                f"divergence follow-up). {_PM_PR_REVIEW_ESCAPE} "
                f"Resume once the infrastructure is confirmed available: "
                f"`lapis-pm resume {target_id}`."
            ),
        )
        target.save()

    episodic.write_observation(
        target_id,
        f"Reviewer infra-retry budget exhausted on PR #{pr_number} cycle {cycle} "
        f"({infra_attempts}/{infra_budget} infrastructure non-runs, "
        f"cause={reason_clause}) — target auto-paused.\n"
        f"{json.dumps(record, indent=2)}",
        extra_tags=[
            "pm:reviewer-infra-budget",
            f"pm:pr={pr_number}",
            f"pm:reviewer-infra-budget:pr={pr_number}:cycle={cycle}",
        ],
    )
    _mem().set(marker_key, json.dumps(record),
               tags=["lapis-pm", "reviewer-infra-budget", f"target={target_id}"])
    return (
        f"action:reviewer_infra_budget_paused:pr={pr_number}:cycle={cycle}:"
        f"infra_attempts={infra_attempts}"
    )


# ---------------------------------------------------------------------------
# Encode percepts
# ---------------------------------------------------------------------------

def _encode_user_comments(target_id: str, comments: list) -> list:
    """Return list of comments that triggered any state change worth acting on.

    User comments themselves are already in the JSONL — we don't re-encode
    them as observations. We return the honored directive comments for
    `decide` use.

    Acceptance gate (Bridge A, spec lapis-pm-signed-directive-acceptance-v0
    §4.3): every human:directive comment is verified via
    signed_directive.verify_directive — five checks (deposit tag present,
    row exists, signature verified, anchored to Erah's root, content-bound to
    this live comment). A fault is always surfaced (a diagnostic held-fault
    observation, plus a brief when well-formed per LAPIS_PM_DIRECTIVE_BRIEF_ON)
    — never silently dropped. Only the *consequence* of a fault depends on
    LAPIS_PM_DIRECTIVE_ENFORCEMENT: 'observe' (default) still honors it
    (loud warning, no lockout while keys are provisioned); 'enforce'
    quarantines it.
    """
    enforcement = signed_directive.directive_enforcement_mode()
    brief_on = signed_directive.directive_brief_on()
    directives = []
    for c in comments:
        if episodic.TAG_HUMAN_DIRECTIVE not in c.tags:
            continue
        verdict = signed_directive.verify_directive(target_id, c)
        if verdict.ok:
            directives.append(c)
            continue

        quarantined = enforcement == "enforce"
        fault_tag = "directive:quarantined" if quarantined else "directive:observed-unsigned"
        episodic.write_observation(
            target_id,
            f"HELD FAULT: directive {c.id} — {verdict.reason} "
            f"(signer={verdict.pubkey_id or 'none'}). Re-anchor: {signed_directive.RUNBOOK_HINT}",
            extra_tags=["pm:held-fault", fault_tag],
        )
        if not quarantined:
            directives.append(c)  # observe mode: honor with loud warning

        well_formed = signed_directive.deposit_ref(c.tags) is not None
        should_brief = brief_on == "all" or (brief_on == "well_formed" and well_formed)
        if should_brief:
            b = brief.synthesize(
                target_id,
                trigger=f"directive held fault: {verdict.reason}",
                query=c.content,
                notify=NotifyPriority.HIGH,
            )
            _set_brief_outstanding(target_id, b)

    acks = [c for c in comments if episodic.TAG_HUMAN_ACK in c.tags]
    if acks:
        clear_outstanding_brief(target_id, reason="user_ack")
        episodic.write_observation(
            target_id, f"Brief acknowledged by user ({len(acks)} ack(s)).",
        )
    return directives


_GPU_FAIL_PREFIXES = ("ERROR", "EXIT ", "TIMEOUT")


def _encode_pr_sha_updates(target_id: str, open_prs: list[dict]) -> int:
    """Write SHA-advance observation when a PR's head SHA changes. Returns count written."""
    written = 0
    for pr in open_prs:
        pr_num = pr.get("number")
        sha = (pr.get("head") or {}).get("sha")
        if not sha or not pr_num:
            continue
        if sha != _last_observed_pr_sha(target_id, pr_num):
            episodic.write_observation(
                target_id,
                f"PR #{pr_num} head SHA: {sha}",
                extra_tags=[f"pm:pr={pr_num}", f"pm:pr={pr_num}:sha={sha}"],
            )
            # SHA changed → invalidate classification so the next decide loop
            # re-screens the new code instead of noop'ing on a stale brief.
            ids = _classified_pr_ids(target_id)
            if pr_num in ids:
                ids.discard(pr_num)
                _mem().set(
                    _classified_prs_key(target_id),
                    json.dumps(sorted(ids)),
                    tags=["lapis-pm", "classified-prs"],
                )
            written += 1
    return written


def _encode_pr_body_updates(target_id: str, open_prs: list[dict]) -> int:
    """Write body-fingerprint observation when a PR's description changes. Returns count written."""
    written = 0
    for pr in open_prs:
        pr_num = pr.get("number")
        body = pr.get("body") or ""
        if not pr_num:
            continue
        fp = hashlib.sha256(body.encode()).hexdigest()[:16]
        if fp != _last_observed_pr_body_fp(target_id, pr_num):
            episodic.write_observation(
                target_id,
                f"PR #{pr_num} description changed",
                extra_tags=[f"pm:pr={pr_num}", f"pm:pr={pr_num}:body={fp}"],
            )
            # Body changed → invalidate classification so decide loop re-screens
            ids = _classified_pr_ids(target_id)
            if pr_num in ids:
                ids.discard(pr_num)
                _mem().set(
                    _classified_prs_key(target_id),
                    json.dumps(sorted(ids)),
                    tags=["lapis-pm", "classified-prs"],
                )
            written += 1
    return written


# ---------------------------------------------------------------------------
# Friction capture (operational-learning-friction-capture-v0, Unit 1a)
#
# Pure capture + queue, no distillation. Fixer deposits an optional
# friction.json sidecar committed alongside its PR; harvest sanitizes the
# agent-authored narrative fields (untrusted), stamps machine-authored
# provenance from the dispatch record lapis-pm already holds, and appends
# idempotently to /srv/lapis/friction/queue.jsonl. Absence of a sidecar is itself
# captured as a distinct "silent-gap" record — never conflated with success.
#
# `environment` carries latency_ms/step_depth/load_pct/pr_diff_lines, all
# null in this unit; Unit 1b (operational-learning-friction-environment-v0)
# backfills them from pm_core.py dispatch records, which this unit does not
# touch.
# ---------------------------------------------------------------------------

FRICTION_MAX_RECORDS = 8
_FRICTION_MAX_OBSTACLE_LEN = 1000
_FRICTION_MAX_PATH_TAKEN_LEN = 1000
_FRICTION_MAX_ARTIFACT_LEN = 300
_FRICTION_ALLOWED_COST_HINTS = {"trivial", "minor", "costly"}
_FRICTION_ALLOWED_CONFIDENCE = {"hunch", "likely", "certain"}
_FRICTION_SILENT_GAP_INDEX = -1  # reserved index; never collides with a real 0..N-1 array index
_FRICTION_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

_FRICTION_NULL_ENVIRONMENT = {
    "latency_ms": None,
    "step_depth": None,
    "load_pct": None,
    "pr_diff_lines": None,
}


def _friction_queue_path() -> Path:
    """Resolve /srv/lapis/friction/queue.jsonl.

    Prefers the agents-core room_paths classmap 'friction' key; falls back to
    the same env-var/ROOM_ROOT resolution room_paths.py itself uses when that
    key has not landed there yet (classmap changes are out of scope for this
    lapis-pm-only unit). Picks up the upstream key automatically once added.
    """
    try:
        base = room_path('friction')
    except KeyError:
        base = Path(os.environ.get('ROOM_ROOT', '/room')) / 'friction'
    return base / "queue.jsonl"


def _sanitize_friction_str(val, max_len: int) -> str | None:
    if not isinstance(val, str):
        return None
    cleaned = _FRICTION_CONTROL_CHAR_RE.sub('', val).strip()
    if not cleaned:
        return None
    return cleaned[:max_len]


def _sanitize_friction_record(raw: dict) -> dict | None:
    """Zero-trust sanitize+re-type one agent-authored friction record.

    Only the agent-authored key set is ever consulted; any other key on
    `raw` is ignored (never copied through). Returns None if the record is
    malformed or its obstacle sanitizes to empty (drop, not partial trust).
    """
    if not isinstance(raw, dict):
        return None
    obstacle = _sanitize_friction_str(raw.get("obstacle"), _FRICTION_MAX_OBSTACLE_LEN)
    if not obstacle:
        return None
    out: dict = {"obstacle": obstacle}
    path_taken = _sanitize_friction_str(raw.get("path_taken"), _FRICTION_MAX_PATH_TAKEN_LEN)
    out["path_taken"] = path_taken or ""
    artifact = _sanitize_friction_str(raw.get("artifact"), _FRICTION_MAX_ARTIFACT_LEN)
    if artifact:
        out["artifact"] = artifact
    cost_hint = raw.get("cost_hint")
    if cost_hint in _FRICTION_ALLOWED_COST_HINTS:
        out["cost_hint"] = cost_hint
    confidence = raw.get("confidence")
    if confidence in _FRICTION_ALLOWED_CONFIDENCE:
        out["confidence"] = confidence
    return out


def _sanitize_friction_array(raw) -> list[dict]:
    """Sanitize a whole friction.json payload. Never raises.

    A non-list payload, or one exceeding FRICTION_MAX_RECORDS, is logged and
    capped/dropped rather than partially trusted. Per-record index reflects
    position in the (capped) raw array so re-harvesting the same content is
    idempotent.
    """
    if not isinstance(raw, list):
        logger.warning("friction harvest: friction.json is not a JSON array; treating as absent")
        return []
    if len(raw) > FRICTION_MAX_RECORDS:
        logger.warning(
            "friction harvest: friction.json has %d entries, capping to %d",
            len(raw), FRICTION_MAX_RECORDS,
        )
    sanitized = []
    for idx, raw_rec in enumerate(raw[:FRICTION_MAX_RECORDS]):
        cleaned = _sanitize_friction_record(raw_rec)
        if cleaned is None:
            logger.info("friction harvest: dropped malformed/empty record at index %d", idx)
            continue
        cleaned["index"] = idx
        sanitized.append(cleaned)
    return sanitized


def _friction_provenance(target_id: str, repo: str, pr_number: int, dispatch_record: dict) -> dict:
    return {
        "task_id": dispatch_record.get("gpu_id"),
        "spec_id": dispatch_record.get("spec_id"),
        "target_id": target_id,
        "repo": repo,
        "pr": pr_number,
        "agent_type": dispatch_record.get("agent_type"),
        "captured_at": _now_iso(),
    }


def _find_dispatch_record_for_pr(target_id: str, pr: dict) -> dict | None:
    """Match an open PR to its originating dispatch record via the lapis-gpu-id marker."""
    markers = _extract_pr_markers(pr.get("body") or "")
    gpu_id = markers.get("gpu_id")
    if not gpu_id:
        return None
    for record in load_dispatched(target_id):
        if record.get("gpu_id") == gpu_id:
            return record
    return None


def _fetch_friction_sidecar(repo: str, pr_number: int) -> str | None:
    """Fetch friction.json raw content for a PR, if present. None if absent/unfetchable."""
    if _forgejo_get_pr_files is None:
        return None
    try:
        repo_name, owner = _repo_owner(repo)
        files = _forgejo_get_pr_files(repo_name, pr_number, owner=owner)
    except Exception as e:
        logger.warning("friction harvest: get_pr_files failed for %s#%s: %s", repo, pr_number, e)
        return None
    friction_file = next((f for f in files if f.get("filename") == "friction.json"), None)
    if friction_file is None:
        return None
    raw_url = friction_file.get("raw_url")
    if not raw_url:
        return None
    try:
        from agents_core.forgejo import FORGEJO_TOKEN
        import httpx
        r = httpx.get(raw_url, headers={"Authorization": f"token {FORGEJO_TOKEN}"}, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning("friction harvest: failed to fetch friction.json content: %s", e)
        return None


def _friction_normalize_text(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _friction_content_key(obstacle: str | None, path_taken: str | None) -> str:
    """Content key for cross-target inheritance detection (D2/D3).

    sha256(normalize(obstacle) + NUL + normalize(path_taken))[:16]. Deliberately
    excludes silent-gap records from every call site — see finding 5: they
    share identical text by design and are not obstacle content.
    """
    norm = _friction_normalize_text(obstacle) + "\x00" + _friction_normalize_text(path_taken)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _existing_friction_keys(queue_path: Path) -> tuple[set[tuple], dict[str, dict]]:
    """Scan the queue for identity keys and content keys.

    Identity keys are (task_id, index) across all record sources — the
    existing same-task idempotence guard, unchanged.

    Content keys map a normalized (obstacle, path_taken) hash to the
    provenance of the first `agent-deposit` record found with that content —
    used to suppress entries inherited (with false provenance) from an
    earlier target (finding 2). Silent-gap records never participate: they
    share identical text by design (finding 5).
    """
    identity_keys: set[tuple] = set()
    content_index: dict[str, dict] = {}
    if not queue_path.exists():
        return identity_keys, content_index
    try:
        with queue_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prov = rec.get("provenance") or {}
                identity_keys.add((prov.get("task_id"), rec.get("index")))
                if rec.get("record_source") == "agent-deposit":
                    ckey = _friction_content_key(rec.get("obstacle"), rec.get("path_taken"))
                    if ckey not in content_index:
                        content_index[ckey] = prov
    except OSError:
        pass
    return identity_keys, content_index


_FRICTION_LOCK_TIMEOUT_DEFAULT = 10.0


class _FrictionLockTimeout(Exception):
    """Raised when the friction queue flock is not acquired within the timeout."""


def _friction_lock_path() -> Path:
    return _friction_queue_path().parent / ".queue.lock"


@contextlib.contextmanager
def _friction_queue_lock(timeout_s: float | None = None):
    """Exclusive flock serializing the friction queue's two writers (D2a):
    harvest append (`_append_friction_records`) and the D3 backfill's
    whole-file rewrite. Prevents a backfill rewrite from silently dropping a
    concurrently-appended record.

    Raises `_FrictionLockTimeout` if not acquired within `timeout_s`. Callers
    decide the failure mode: harvest must never raise (warn + skip — safe to
    retry since harvest is idempotent); backfill fails closed (abort, zero
    bytes changed — see `friction_backfill_provenance`).

    `timeout_s` defaults to `_FRICTION_LOCK_TIMEOUT_DEFAULT`, read at call
    time (not bound as a function-signature default) so tests can override
    the module-level constant and have it take effect immediately.
    """
    if timeout_s is None:
        timeout_s = _FRICTION_LOCK_TIMEOUT_DEFAULT
    lock_path = _friction_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise _FrictionLockTimeout(
                        f"friction queue lock timed out after {timeout_s}s: {lock_path}"
                    )
                time.sleep(0.1)
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _write_inherited_entries_marker(task_id: str | None, suppressed: list[dict]) -> None:
    """Durable marker for a harvest that suppressed >=1 inherited entry (D2 fold).

    Loud by design — never called on the clean path (no suppression -> no
    marker, no noise).
    """
    marker_path = _friction_queue_path().parent / f"inherited-entries-{task_id or 'unknown'}.json"
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(suppressed, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning("friction harvest: failed to write inherited-entries marker %s: %s", marker_path, e)


def _append_friction_records_locked(records: list[dict], queue_path: Path) -> int:
    identity_keys, content_index = _existing_friction_keys(queue_path)
    written = 0
    suppressed: list[dict] = []
    incoming_task_id = None
    with queue_path.open("a", encoding="utf-8") as f:
        for rec in records:
            prov = rec.get("provenance") or {}
            incoming_task_id = incoming_task_id or prov.get("task_id")
            ikey = (prov.get("task_id"), rec.get("index"))
            if ikey in identity_keys:
                continue
            if rec.get("record_source") == "agent-deposit":
                ckey = _friction_content_key(rec.get("obstacle"), rec.get("path_taken"))
                existing_prov = content_index.get(ckey)
                if existing_prov is not None and existing_prov.get("task_id") != prov.get("task_id"):
                    logger.warning(
                        "friction harvest: suppressed inherited entry %s "
                        "(incoming task=%s target=%s) already recorded under task=%s target=%s",
                        ckey, prov.get("task_id"), prov.get("target_id"),
                        existing_prov.get("task_id"), existing_prov.get("target_id"),
                    )
                    suppressed.append({
                        "hash": ckey,
                        "incoming": {
                            "task_id": prov.get("task_id"), "target_id": prov.get("target_id"),
                            "repo": prov.get("repo"), "pr": prov.get("pr"),
                        },
                        "existing": {
                            "task_id": existing_prov.get("task_id"), "target_id": existing_prov.get("target_id"),
                            "repo": existing_prov.get("repo"), "pr": existing_prov.get("pr"),
                        },
                    })
                    continue
                content_index[ckey] = prov
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            identity_keys.add(ikey)
            written += 1
    if suppressed:
        _write_inherited_entries_marker(incoming_task_id, suppressed)
    return written


def _append_friction_records(records: list[dict]) -> int:
    """Append-only idempotent write, deduped on (task_id, index) plus
    content-level suppression of cross-target inherited entries (D2).

    Guarded by the D2a lock shared with the backfill rewrite. Never raises —
    on lock timeout, logs a WARNING and returns 0; the caller (harvest) will
    observe the same PR again next tick and safely re-append (harvest is
    idempotent), so no data is lost, only delayed.
    """
    if not records:
        return 0
    queue_path = _friction_queue_path()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _friction_queue_lock():
            return _append_friction_records_locked(records, queue_path)
    except _FrictionLockTimeout as e:
        logger.warning("friction harvest: %s — skipping this batch, will retry next observation", e)
        return 0


def _harvest_friction_for_pr(target_id: str, repo: str, pr: dict) -> int:
    """Read+sanitize an observed PR's friction.json (if any) and append to the queue.

    Never raises — a harvest failure must never affect the review/merge flow.
    Absence of a sidecar (or a sidecar with zero valid records after
    sanitization) is captured as one machine-authored silent-gap record, not
    a drop and not conflated with success.
    """
    pr_number = pr.get("number")
    if pr_number is None:
        return 0
    dispatch_record = _find_dispatch_record_for_pr(target_id, pr)
    if dispatch_record is None:
        # Can't stamp trusted provenance without a matched dispatch — skip.
        return 0

    try:
        raw_text = _fetch_friction_sidecar(repo, pr_number)
        raw_json = json.loads(raw_text) if raw_text is not None else None
    except json.JSONDecodeError:
        logger.warning("friction harvest: friction.json for %s#%s is not valid JSON", repo, pr_number)
        raw_json = None

    sanitized = _sanitize_friction_array(raw_json) if raw_json is not None else []
    provenance = _friction_provenance(target_id, repo, pr_number, dispatch_record)

    if not sanitized:
        records = [{
            "record_source": "silent-gap",
            "obstacle": "no_friction_reported",
            "derived_confidence": "zero",
            "environment": dict(_FRICTION_NULL_ENVIRONMENT),
            "provenance": provenance,
            "index": _FRICTION_SILENT_GAP_INDEX,
        }]
    else:
        records = [
            {
                "record_source": "agent-deposit",
                **rec,
                "environment": dict(_FRICTION_NULL_ENVIRONMENT),
                "provenance": provenance,
            }
            for rec in sanitized
        ]

    return _append_friction_records(records)


def _encode_friction_harvest(target_id: str, repo: str, open_prs: list[dict]) -> int:
    """Encode-phase step: harvest friction for every currently-open PR. Bookkeeping, not action."""
    written = 0
    for pr in open_prs:
        try:
            written += _harvest_friction_for_pr(target_id, repo, pr)
        except Exception as e:
            logger.warning(
                "friction harvest: unexpected error for %s PR #%s: %s",
                repo, pr.get("number"), e,
            )
    return written


def read_friction_records(
    target_id: str | None = None, repo: str | None = None, limit: int | None = None,
) -> list[dict]:
    """Read friction queue records, newest first, optionally filtered.

    Read-only; used by `lapis-pm friction list`.
    """
    queue_path = _friction_queue_path()
    if not queue_path.exists():
        return []
    records: list[dict] = []
    try:
        with queue_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    records.reverse()  # append-only file → last line is newest
    if target_id:
        records = [r for r in records if (r.get("provenance") or {}).get("target_id") == target_id]
    if repo:
        records = [r for r in records if (r.get("provenance") or {}).get("repo") == repo]
    if limit is not None:
        records = records[:limit]
    return records


# ---------------------------------------------------------------------------
# D3 — one-shot provenance backfill (friction-sidecar-per-pr-provenance-v0)
#
# Post-deploy archaeology for the corpus written before D1+D2 closed the
# inheritance hole: tags (never deletes or reorders) non-canonical members of
# a cross-target content group with provenance_suspect + canonical_task_id.
# Manual-only — invoked by hand, once, by a PM session. No systemd unit, no
# timer, no daemon/night-pass call site (see
# tests/test_friction_capture.py::TestBackfillManualOnly).
# ---------------------------------------------------------------------------

def _friction_group_key_for_backfill(rec) -> str | None:
    """Content key for D3 grouping, or None if this record is excluded from
    grouping entirely (finding 5 / D3 fold): only `agent-deposit` records are
    grouped. Silent-gap records (`obstacle == "no_friction_reported"`) share
    identical text across every target by design and must never collapse
    into a bogus group.
    """
    if not isinstance(rec, dict):
        return None
    if rec.get("record_source") != "agent-deposit":
        return None
    return _friction_content_key(rec.get("obstacle"), rec.get("path_taken"))


def _compute_friction_backfill_tags(records: list) -> tuple[dict[int, dict], int]:
    """Pure function: given the parsed queue in file order (index == line
    number; malformed lines are `None`), return {index: tag_dict} for every
    non-canonical member of a group whose content key is shared across >=2
    distinct task_ids, plus the number of such cross-attribution groups.

    Canonical = earliest by `provenance.captured_at` within the group;
    canonical members are never present in the returned tags dict.
    """
    groups: dict[str, list[tuple[int, dict]]] = {}
    for idx, rec in enumerate(records):
        key = _friction_group_key_for_backfill(rec)
        if key is None:
            continue
        groups.setdefault(key, []).append((idx, rec))

    tags: dict[int, dict] = {}
    group_count = 0
    for members in groups.values():
        task_ids = {(rec.get("provenance") or {}).get("task_id") for _, rec in members}
        if len(task_ids) < 2:
            continue  # same-task repeats only — not cross-attribution
        group_count += 1
        ordered = sorted(members, key=lambda m: (m[1].get("provenance") or {}).get("captured_at") or "")
        canonical_task_id = (ordered[0][1].get("provenance") or {}).get("task_id")
        for idx, _rec in ordered[1:]:
            tags[idx] = {"provenance_suspect": True, "canonical_task_id": canonical_task_id}
    return tags, group_count


def friction_backfill_provenance(*, apply: bool = False) -> dict:
    """One-shot D3 backfill. Dry-run by default; `apply=True` writes tags.

    Rewrites the queue atomically (tmp + rename) under the D2a lock shared
    with harvest appends. No record is ever deleted or reordered — line
    count and ordering are identical before/after. Idempotent: re-running
    against an already-tagged queue recomputes the same tags and leaves the
    file byte-identical.

    Raises `_FrictionLockTimeout` on lock contention — unlike harvest, the
    backfill does NOT swallow it: callers (the CLI) must abort non-zero
    having changed zero bytes, per D2a.
    """
    queue_path = _friction_queue_path()
    with _friction_queue_lock():
        if not queue_path.exists():
            return {"total_records": 0, "tagged": 0, "groups": 0, "applied": False}

        raw_lines = queue_path.read_text(encoding="utf-8").splitlines()
        records: list[dict | None] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                records.append(None)
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                records.append(None)

        tags, group_count = _compute_friction_backfill_tags(records)
        result = {
            "total_records": len(records),
            "tagged": len(tags),
            "groups": group_count,
            "applied": False,
        }
        if not apply:
            return result

        new_lines = []
        for idx, raw_line in enumerate(raw_lines):
            tag = tags.get(idx)
            rec = records[idx]
            if tag is not None and rec is not None:
                new_lines.append(json.dumps({**rec, **tag}, ensure_ascii=False))
            else:
                new_lines.append(raw_line)

        tmp_path = queue_path.with_name(queue_path.name + ".tmp")
        tmp_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        os.replace(tmp_path, queue_path)
        result["applied"] = True
        return result


def _recover_reviewer_verdict(raw: str) -> dict | None:
    """Attempt to recover a reviewer verdict dict from malformed JSON.

    Recovery strategies tried in order:
    1. Strip code fences (``` or ```json) and re-parse. (Mostly redundant with
       the eager fence-strip in _encode_gpu_results' live path; retained so
       the helper is also useful when called directly.)
    2. Extract outermost {...} via string-aware balanced-brace scan and
       re-parse. Braces inside JSON string literals are ignored so prose like
       `"note": "} oops"` doesn't close the object early.
    3. Regex extraction of verdict literal as last resort — returns
       issues=[], confidence=0.0 if the structure is otherwise unparseable.

    Returns a parsed dict on success, None if all strategies fail.
    """
    # Strategy 1: strip code fences
    stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    stripped = re.sub(r"\s*```\s*$", "", stripped.strip())
    if stripped != raw.strip():
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Strategy 2: extract outermost {...} via string-aware balanced-brace scan
    start = raw.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(raw[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # Strategy 3: regex extraction of verdict field (verdict-only recovery).
    # confidence=0.0 signals "structure was lost, only the verdict survived"
    # so any downstream consumer can treat it as low-trust if needed.
    m = re.search(r'"verdict"\s*:\s*"(clean|fixable|needs-human|blocked)"', raw)
    if m:
        return {"verdict": m.group(1), "issues": [], "confidence": 0.0}

    return None


# L2.D2 (local-reviewer-identity-and-provenance-v0): the agent types whose
# dispatch records carry the ACTUAL served model (stamped from the completed
# queue task yaml). Scoped to the dispatch-verified agent types — other
# agent types reaching a record-close site must NOT gain the field.
_SERVED_MODEL_STAMP_AGENT_TYPES = frozenset(
    set(_REVIEWER_AGENT_TYPES) | {"fixer", "fixer_retry"}
)


def _stamp_served_model(rec: dict, target_id: str) -> None:
    """L2.D2: stamp ``served_model`` onto a terminal dispatch record from the
    completed queue task yaml (Leg 1's L1.D3 field).

    Null-tolerant best-effort: a missing yaml / missing field / null echo
    leaves served_model as None — NEVER the seat alias (Erah 2026-09-06
    explicit-void adjudication). Scoped to the dispatch-verified agent types
    (reviewer/reviewer_fresh/fixer/fixer_retry). Must never fail
    verdict-encode (the idiom at 8082/8103/8121).
    """
    if rec.get("agent_type") not in _SERVED_MODEL_STAMP_AGENT_TYPES:
        return
    try:
        from . import provenance as _provenance
        rec["served_model"] = _provenance.read_served_model_from_queue_yaml(
            rec.get("gpu_id")
        )
    except Exception as _prov_exc:
        rec["served_model"] = None
        episodic.write_observation(
            target_id,
            f"served-model provenance read skipped for {rec.get('gpu_id')}: "
            f"{type(_prov_exc).__name__}",
            extra_tags=["pm:served-provenance-skipped"],
        )  # best-effort; never fail verdict-encode


def _encode_gpu_results(target_id: str) -> tuple[int, list[dict]]:
    """For each pending dispatch, check for completion and encode result.

    Returns (total_encoded, failed_for_retry).
    """
    # Clear any stale tick-local corroboration entries for this target so
    # _persist_review_state_cache always sees fresh data from this tick.
    for k in list(_tick_corr_cache):
        if k[0] == target_id:
            del _tick_corr_cache[k]
    records = load_dispatched(target_id)
    failed_for_retry: list[dict] = []
    total_encoded = 0
    changed = False
    for rec in records:
        if rec.get("status") != "pending":
            continue

        # fixer_retry uses PR SHA/body-advance as completion signal (preferred per spec)
        # rather than GPU output file, so we can detect completion even when the
        # fixer pushed code or edited the PR description (claude-queue or API path).
        if rec.get("agent_type") == "fixer_retry":
            pr_num = rec.get("pr_number")
            dispatch_ts = rec.get("ts", "")
            if pr_num is not None:
                completion_ts = _fixer_completion_ts(target_id, pr_num, dispatch_ts)
                if completion_ts:
                    rec["status"] = "processed"
                    rec["completed_at"] = completion_ts
                    changed = True
                    total_encoded += 1
                    _stamp_served_model(rec, target_id)
                    _close_slot_and_deposit(rec, target_id)
                    # Distinguish SHA vs description-only advance for episodic trail
                    body_prefix = f"pm:pr={pr_num}:body="
                    is_body_advance = any(
                        t.startswith(body_prefix)
                        for c in episodic.all_comments(target_id)
                        if c.ts == completion_ts
                        for t in c.tags
                    )
                    completion_note = (
                        "PR description advanced after dispatch"
                        if is_body_advance
                        else "head SHA advanced after dispatch"
                    )
                    episodic.write_result(
                        target_id,
                        f"Fixer retry for PR #{pr_num} completed: {completion_note}",
                        extra_tags=[
                            f"pm:gpu={rec['gpu_id']}",
                            "pm:agent=fixer_retry",
                            f"pm:pr={pr_num}",
                        ],
                    )
                    continue  # handled via advance signal
                # No advance yet — check if job is terminal via output file.
                # If terminal but no advance, flip to processed so the
                # lost-dispatch net can classify and handle it.
                out_path = _gpu_output_path(rec["gpu_id"])
                if out_path is not None:
                    rec["status"] = "processed"
                    rec["completed_at"] = _now_iso()
                    changed = True
                    total_encoded += 1
                    _stamp_served_model(rec, target_id)
                    episodic.write_observation(
                        target_id,
                        f"Fixer retry for PR #{pr_num} job complete:"
                        " no code or description change observed",
                        extra_tags=[
                            f"pm:gpu={rec['gpu_id']}",
                            "pm:agent=fixer_retry",
                            f"pm:pr={pr_num}",
                            "pm:fixer-retry-noop",
                        ],
                    )
                    # Leg 3: a no-op retry against a degraded/refuted verdict
                    # is strong evidence the verdict was false, not that
                    # there's nothing to fix. Escalate rather than leave the
                    # target wedged with nothing scheduled to advance it.
                    try:
                        _escalate_noop_retry_if_degraded(target_id, rec, pr_num)
                    except Exception as _esc_exc:
                        episodic.write_observation(
                            target_id,
                            f"noop-retry escalation check skipped for PR #{pr_num}: "
                            f"{type(_esc_exc).__name__}",
                            extra_tags=["pm:noop-retry-escalation-skipped"],
                        )  # best-effort; never fail verdict encoding
                    # D6b (fixer-reception-v0, leg 2): a no-op retry against a
                    # HEALTHY verdict (the starved/refuted case is owned by the
                    # escalation above — Invariant 8, untouched) dispatches the
                    # auditor for the same PR: audit why the retry produced no
                    # change + the delta-to-green. Best-effort, same discipline
                    # as the escalation check.
                    try:
                        _maybe_dispatch_auditor_noop(target_id, rec, pr_num)
                    except Exception as _aud_exc:
                        episodic.write_observation(
                            target_id,
                            f"noop-audit check skipped for PR #{pr_num}: "
                            f"{type(_aud_exc).__name__}",
                            extra_tags=["pm:noop-audit-skipped"],
                        )  # best-effort; never fail verdict encoding
                continue  # fixer_retry never falls to reviewer verdict path
            # fixer_retry without pr_number: fall through to GPU output file path
            # as defensive fallback (shouldn't happen in practice).

        out_path = _gpu_output_path(rec["gpu_id"])
        if not out_path:
            continue
        try:
            text = out_path.read_text(encoding="utf-8")
        except OSError:
            continue

        is_failure = (
            out_path.parent in (FAILED_DIR, CLAUDE_QUEUE_FAILED_DIR)
            or any(text.lstrip().startswith(p) for p in _GPU_FAIL_PREFIXES)
        )

        # Already-satisfied verdict check: fixer may write a machine-readable
        # verdict sidecar instead of opening a PR (lapis-pm-fixer-already-done-verdict).
        # Check before confabulation so a valid verdict is never marked confabulated.
        # fixer_local is intentionally excluded: the GW engine signals completion via
        # empty-diff/concluded=False rather than writing a verdict sidecar. Its no-PR
        # failure mode is caught by _find_lost_fixer_dispatches, not this path.
        _fixer_spec_id = rec.get("spec_id") if rec.get("agent_type") == "fixer" else None
        if _fixer_spec_id and not is_failure:
            _verdict_raw = _read_fixer_verdict(_fixer_spec_id)
            _consume_fixer_verdict(_fixer_spec_id)
            if _verdict_raw is not None:
                _verdict_handled = _handle_already_satisfied_verdict(
                    target_id, rec, _verdict_raw
                )
                if _verdict_handled:
                    # Also consume meta sidecar (verdict takes priority)
                    _consume_fixer_meta(_fixer_spec_id)
                    rec["status"] = "processed"
                    rec["completed_at"] = _now_iso()
                    changed = True
                    total_encoded += 1
                    _stamp_served_model(rec, target_id)
                    _close_slot_and_deposit(rec, target_id)
                    continue  # Skip confabulation + normal result encoding

        # Confabulation check: fixer agents that produced substantial prose
        # without using tools are stochastic failures and should be retried.
        # Meta sidecar is only written for fixers (capture_meta=true in registry.yaml).
        confabulation_note = ""
        spec_id = rec.get("spec_id") if rec.get("agent_type") == "fixer" else None
        if spec_id and not is_failure:
            meta = _read_fixer_meta(spec_id)
            if meta and meta.get("confabulated"):
                is_failure = True
                confabulation_note = (
                    f"CONFABULATED — fixer produced "
                    f"{meta.get('char_count', 0)} chars with no PR URL "
                    f"or git evidence (num_turns={meta.get('num_turns')}, "
                    f"basis={meta.get('decision_basis', '?')})\n\n"
                )
        # Whether we read the meta or it was missing/clean, drop the sidecar
        # now that the dispatch is no longer pending. Avoids /srv/lapis/gpu-queue/
        # shaped/ accumulating stale meta files indefinitely.
        if spec_id:
            _consume_fixer_meta(spec_id)

        rec["status"] = "failed" if is_failure else "processed"
        rec["completed_at"] = _now_iso()
        changed = True
        # L2.D2: stamp the ACTUAL served model from the completed queue task
        # yaml (scoped to the dispatch-verified agent types; null-tolerant,
        # never the seat alias, never fails verdict-encode).
        _stamp_served_model(rec, target_id)
        _close_slot_and_deposit(rec, target_id)

        snippet = text.strip()
        if len(snippet) > 1500:
            snippet = snippet[:1500] + "\n…[truncated, full output: " + str(out_path) + "]"
        snippet = confabulation_note + snippet
        tags = [
            f"pm:gpu={rec['gpu_id']}",
            f"pm:agent={rec.get('agent_type', 'unknown')}",
        ]
        if is_failure:
            episodic.write_result(
                target_id,
                f"FAILED — {rec.get('agent_type')} task {rec['gpu_id']}\n"
                f"Intent: {rec.get('intent')}\n\n{snippet}",
                extra_tags=tags + ["pm:failure"],
            )
            if rec.get("agent_type") in _REVIEWER_AGENT_TYPES and rec.get("pr_number") is not None:
                # Stash the as-reported reason for the attempt-ceiling record
                # (DoD 3b: recorded as claimed, never as a verified cause).
                reported_reason = (text.strip().splitlines() or [""])[0][:500]
                _record_reviewer_attempt_reason(
                    target_id, rec["pr_number"], rec.get("cycle", 1), reported_reason,
                    wait_s=_reviewer_dispatch_wait_seconds(rec),
                    completed_at=rec.get("completed_at"),
                )
            failed_for_retry.append(rec)
        else:
            # Reviewer agents: parse JSON output and write reviewer-tagged episodic entry
            if rec.get("agent_type") in _REVIEWER_AGENT_TYPES:
                pr_num = rec.get("pr_number", "?")
                cycle_num = rec.get("cycle", 1)
                # R1 (lapis-pm-reviewer-defer-backoff-v0): a real verdict is a
                # non-infra completion — clear any backoff left over from
                # earlier infra failures on this same pr+cycle so a stale
                # next_retry_at never lingers past a successful attempt.
                if isinstance(pr_num, int):
                    _clear_reviewer_infra_backoff(target_id, pr_num, cycle_num)
                raw_output = text.strip()
                if raw_output.startswith("```"):
                    raw_output = re.sub(r"^```(?:json)?\s*", "", raw_output)
                    raw_output = re.sub(r"\s*```$", "", raw_output.strip())
                parse_recovered = False
                try:
                    verdict_data = json.loads(raw_output)
                    verdict_val = verdict_data.get("verdict", "needs-human")
                    stored_json = json.dumps(verdict_data)
                except json.JSONDecodeError:
                    recovered = _recover_reviewer_verdict(raw_output)
                    if recovered is not None:
                        verdict_val = recovered.get("verdict", "needs-human")
                        stored_json = json.dumps(recovered)
                        parse_recovered = True
                    else:
                        verdict_val = "needs-human"
                        stored_json = json.dumps({"verdict": "needs-human", "issues": [],
                                                  "confidence": 0.0,
                                                  "_parse_error": raw_output[:200]})
                # Corroboration follow-up pass (q4 resolution: two LLM calls per
                # review; prompt-cached overlap; reviewer read-only invariant preserved
                # — corroboration_result is additive only, never mutates mainline fields).
                try:
                    _corr_diff = _diff_text_for_corr(rec.get("repo", ""), pr_num)
                    _corr_result = _run_corroboration_pass_sync(_corr_diff, rec.get("repo", ""))
                    _stored_dict = json.loads(stored_json)
                    _stored_dict["corroboration_result"] = _corr_result
                    stored_json = json.dumps(_stored_dict)
                    if isinstance(pr_num, int):
                        _tick_corr_cache[(target_id, pr_num)] = _corr_result
                except Exception as _corr_exc:
                    episodic.write_observation(
                        target_id,
                        f"corroboration pass skipped for PR #{pr_num}: {type(_corr_exc).__name__}",
                        extra_tags=["pm:corroboration-skipped"],
                    )  # best-effort; never fail verdict encoding
                # Local-reviewer witness pass — additive, never mutates mainline fields.
                try:
                    from lapis_pm.local_reviewer_witness import (
                        run_local_reviewer_witness as _run_local_reviewer_witness,
                        _format_divergence_note as _fmt_div_note,
                    )
                    _wit_diff = _diff_text_for_corr(rec.get("repo", ""), pr_num)
                    _wit_spec = episodic.spec_summary(target_id) or ""
                    _claude_verdict_dict = json.loads(stored_json)
                    _wit_result = _run_local_reviewer_witness(
                        diff_text=_wit_diff,
                        repo=rec.get("repo", ""),
                        pr_number=pr_num if isinstance(pr_num, int) else int(pr_num),
                        spec_summary=_wit_spec,
                        claude_verdict=_claude_verdict_dict,
                    )
                    _stored_dict2 = json.loads(stored_json)
                    _stored_dict2["local_reviewer_witness"] = _wit_result.to_dict()
                    stored_json = json.dumps(_stored_dict2)
                    if _wit_result.agreement == "diverge_major":
                        _div_content = _fmt_div_note(_claude_verdict_dict, _wit_result)
                        episodic.write_observation(
                            target_id,
                            _div_content,
                            extra_tags=[
                                "pm:reviewer-divergence",
                                f"pm:reviewer-divergence:pr={pr_num}:type=major",
                                f"pm:pr={pr_num}",
                            ],
                        )
                except Exception as _wit_exc:
                    episodic.write_observation(
                        target_id,
                        f"local witness pass skipped for PR #{pr_num}: {type(_wit_exc).__name__}",
                        extra_tags=["pm:local-witness-skipped"],
                    )  # best-effort; never fail verdict encoding

                # Panel starvation + confidence attenuation (Leg 1). Additive —
                # attaches "panel_starvation" and rewrites "confidence" to the
                # attenuated value, keeping the raw value at
                # panel_starvation.confidence_raw for audit. Never fails verdict
                # encoding; a starved-detection error just skips the annotation
                # (the raw verdict still ships, unattenuated — same as before
                # this leg existed).
                try:
                    from . import panel_starvation as _panel_starvation
                    _stored_dict3 = json.loads(stored_json)
                    _panel_starvation.apply_panel_starvation(_stored_dict3)
                    stored_json = json.dumps(_stored_dict3)
                except Exception as _starv_exc:
                    episodic.write_observation(
                        target_id,
                        f"panel-starvation check skipped for PR #{pr_num}: {type(_starv_exc).__name__}",
                        extra_tags=["pm:panel-starvation-skipped"],
                    )  # best-effort; never fail verdict encoding

                # Absence-claim grounding (Leg 2). Checks any "X is not
                # defined/missing/removed" finding against the full file at
                # the PR head rather than the diff. Refuted findings are
                # dropped from the actionable `issues` list and moved to
                # `refuted_absence_findings` with their location; an
                # unrunnable check leaves the finding actionable, marked
                # unverified (R2) — never a fail-verdict-encoding condition.
                try:
                    from . import absence_ground as _absence_ground
                    _pr_head_sha = None
                    try:
                        from agents_core.forgejo import get_pr as _get_pr
                        _pr_int = pr_num if isinstance(pr_num, int) else int(pr_num)
                        _pr_head_sha = ((_get_pr(rec.get("repo", ""), _pr_int) or {})
                                        .get("head") or {}).get("sha")
                    except Exception:
                        _pr_head_sha = None
                    _stored_dict4 = json.loads(stored_json)
                    _absence_ground.ground_verdict_issues(
                        _stored_dict4, rec.get("repo", ""), pr_num, _pr_head_sha,
                    )
                    stored_json = json.dumps(_stored_dict4)
                except Exception as _ground_exc:
                    episodic.write_observation(
                        target_id,
                        f"absence-claim grounding skipped for PR #{pr_num}: {type(_ground_exc).__name__}",
                        extra_tags=["pm:absence-grounding-skipped"],
                    )  # best-effort; never fail verdict encoding

                # D7 (fixer-reception-v0, leg 2): tag the verdict with the
                # head sha it was rendered against — the latest pm:pr=N:sha=
                # observation at or before this comment's ts (sha
                # observations encode in the same tick BEFORE gpu results,
                # so the verdict's own sha is already in the stream). The
                # tag is ADDITIVE: _last_review_verdict / the cycle parsers
                # read the tag prefix + JSON body only, so existing
                # consumers are unaffected. If no sha observation exists
                # (a Forgejo perceive miss) the tag is omitted and the
                # verdict is treated as FRESH by _decide_for_pr
                # (effectively unreachable).
                _rendered_sha = (
                    _last_observed_pr_sha(target_id, pr_num)
                    if isinstance(pr_num, int) else None
                )
                result_tags = tags + [
                    f"pm:reviewer:pr={pr_num}:cycle={cycle_num}:verdict={verdict_val}",
                    f"pm:pr={pr_num}",
                ]
                if _rendered_sha:
                    result_tags.append(f"pm:reviewer-sha={_rendered_sha}")
                # L2.D3 (local-reviewer-identity-and-provenance-v0): the
                # verdict write-back comment gains a pm:served=<model> tag
                # when the served model is known (stamped onto `rec` above).
                # ABSENCE is expected, not an error (pre-Leg-1 yamls and the
                # null-echo cases); the value is never the seat alias.
                try:
                    from . import provenance as _provenance
                    _served_tag = _provenance.served_tag_for(rec)
                    if _served_tag:
                        result_tags.append(_served_tag)
                except Exception as _served_exc:
                    pass  # best-effort; tag absence is a non-error
                if parse_recovered:
                    result_tags.append("pm:reviewer:parse-recovered")
                episodic.write_result(
                    target_id,
                    f"Reviewer verdict for PR #{pr_num}:\n{stored_json}",
                    extra_tags=result_tags,
                )
            elif rec.get("agent_type") == AUDITOR_AGENT_TYPE:
                # D6c (fixer-reception-v0, leg 2): the auditor's result
                # encode. The agent posts NOTHING — the daemon is the writer
                # (no agent has a comment-posting tool). On successful JSON
                # parse: the pm:auditor:pr=N:sha=<head> tagged comment (the
                # five-section brief rendered from the JSON, formatted for
                # the advisory brief surface) + the mem keys
                # pm/audit/<tid>/pr=N/dispatched (value = head sha) and
                # pm/audit/<tid>/pr=N/brief (the JSON). On None/parse-fail:
                # a pm:auditor:pr=N:failed observation and NO dispatched
                # marker (a failed audit is retryable next tick).
                _aud_pr_num = rec.get("pr_number")
                _aud_head_sha = rec.get("head_sha")
                _aud_result_tags = tags + [
                    "pm:auditor",
                    f"pm:pr={_aud_pr_num}",
                ]
                if _aud_head_sha:
                    _aud_result_tags.append(f"pm:auditor:pr={_aud_pr_num}:sha={_aud_head_sha}")
                _aud_raw = text.strip()
                if _aud_raw.startswith("```"):
                    _aud_raw = re.sub(r"^```(?:json)?\s*", "", _aud_raw)
                    _aud_raw = re.sub(r"\s*```$", "", _aud_raw.strip())
                _aud_json = None
                if not is_failure and _aud_raw:
                    try:
                        _aud_json = json.loads(_aud_raw)
                    except json.JSONDecodeError:
                        _aud_json = None
                if isinstance(_aud_json, dict):
                    _aud_brief = _render_audit_brief(
                        _aud_json, _aud_pr_num, _aud_head_sha
                    )
                    episodic.write_result(
                        target_id,
                        _aud_brief,
                        extra_tags=_aud_result_tags,
                    )
                    # Gate (i) marker (value = head sha) + the brief JSON.
                    if _aud_head_sha:
                        _set_audit_dispatched(
                            target_id, _aud_pr_num, _aud_head_sha
                        )
                    _mem().set(
                        _audit_brief_key(target_id, _aud_pr_num),
                        json.dumps(_aud_json, ensure_ascii=False),
                        tags=["lapis-pm", "pm:audit-brief"],
                    )
                else:
                    # None/parse-fail: retryable — the failed observation is
                    # written, the dispatched marker is NOT set (and is
                    # cleared if a prior partial state left one), so the
                    # next tick may re-dispatch the un-audited head.
                    _clear_audit_dispatched(target_id, _aud_pr_num)
                    episodic.write_result(
                        target_id,
                        f"FAILED — auditor task {rec['gpu_id']} produced no "
                        f"parseable audit JSON (output: {snippet[:400]})",
                        extra_tags=_aud_result_tags + [
                            "pm:auditor:pr=%s:failed" % _aud_pr_num,
                            "pm:failure",
                        ],
                    )
            else:
                episodic.write_result(
                    target_id,
                    f"Completed — {rec.get('agent_type')} task {rec['gpu_id']}\n"
                    f"Intent: {rec.get('intent')}\n\n{snippet}",
                    extra_tags=tags,
                )
        total_encoded += 1
    if changed:
        save_dispatched(target_id, records)
    return total_encoded, failed_for_retry


def _encode_new_prs(target_id: str, repo: str, prs: list[dict],
                    seen_ids: set[int]) -> list[dict]:
    """Write pm:observation for newly-seen PRs. Returns the new ones."""
    new_prs = [pr for pr in prs if pr.get("number") not in seen_ids]
    for pr in new_prs:
        episodic.write_observation(
            target_id,
            f"PR #{pr.get('number')} opened in {repo}: {pr.get('title')}\n"
            f"{pr.get('html_url', '')}",
            extra_tags=[f"pm:repo={repo}", f"pm:pr={pr.get('number')}"],
        )
    return new_prs


def _seen_pr_ids(target_id: str) -> set[int]:
    """Recover already-encoded PRs by scanning prior comment tags."""
    out: set[int] = set()
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:pr="):
                try:
                    out.add(int(t.split("=", 1)[1]))
                except (ValueError, IndexError):
                    pass
    return out


# ---------------------------------------------------------------------------
# Dispatch–queue reconciliation
# ---------------------------------------------------------------------------

def _reap_verdict(
    gpu_id: str,
    active_by_id: dict[str, dict],
    pending_ids: set[str],
) -> dict | None:
    """Leg 1 reap pass: decide whether a pending record with no terminal
    queue match (not in completed/, not in failed/) should be force-flipped
    to failed anyway, because the queue state proves the work is gone.

    Returns a verdict dict shaped like the ``terminal`` entries built in
    ``_reconcile_dispatched_with_queue`` (state/error/completed_at), plus a
    ``failure_reason`` key, or None if the record must stay pending.

    Case A — task_absent_from_queue: gpu_id is in none of pending / active /
    completed / failed. The task no longer exists anywhere in the queue.
    Case B — active_past_timeout: gpu_id is in active/ and its age exceeds
    its own declared timeout_seconds + STARTUP_STALE_GRACE_S — the same
    arithmetic agents_core.claude_queue_runner.startup_sweep uses to decide
    a runner died mid-task. A live, legitimately-slow fixer within its own
    timeout is never reaped (Case B does not fire) — see spec §Leg 1.
    """
    task = active_by_id.get(gpu_id)
    if task is not None:
        started_at = task.get("started_at")
        timeout = int(task.get("timeout_seconds", 300))
        if not started_at:
            # No started_at recorded at all — can't establish age, so we
            # cannot prove staleness. Degrade open: leave it pending.
            return None
        try:
            started_dt = datetime.fromisoformat(started_at)
            age_s = (datetime.now(PACIFIC) - started_dt).total_seconds()
        except ValueError:
            return None
        if age_s > (timeout + STARTUP_STALE_GRACE_S):
            return {
                "state": "failed",
                "error": None,
                "completed_at": _now_iso(),
                "failure_reason": "reaped:active_past_timeout",
            }
        return None  # active and within its own timeout — never reaped

    if gpu_id in pending_ids:
        # Still queued, waiting to be claimed. Not gone, not stuck.
        return None

    # Not in pending/, not in active/, and (by construction — this is only
    # called when `terminal.get(gpu_id) is None`) not in completed/ or
    # failed/ either. The task no longer exists in the queue at all.
    return {
        "state": "failed",
        "error": None,
        "completed_at": _now_iso(),
        "failure_reason": "reaped:task_absent_from_queue",
    }


def _reconcile_dispatched_with_queue(target_id: str) -> int:
    """Public entry point — see _reconcile_dispatched_with_queue_ex for the
    real implementation and full docstring. This wrapper preserves the
    existing int-only contract for tick() and any other caller that only
    needs the flipped count and is content with degrade-open-to-0 on a
    queue-read failure.
    """
    flipped, _ok = _reconcile_dispatched_with_queue_ex(target_id)
    return flipped


def _reconcile_dispatched_with_queue_ex(target_id: str) -> tuple[int, bool]:
    """Flip pending dispatch records to terminal state based on ClaudeQueue.

    Returns (flipped_count, ok). ``ok`` is False only when the queue read
    itself raised — Leg 2 (force_dispatch) uses it to decide whether to
    warn that reconciliation was skipped. A normal no-op call (nothing to
    reconcile) returns (0, True), distinguishable from a failed read's
    (0, False). Idempotent — calling twice in a row with no new queue
    activity is a no-op (returns 0 the second time).

    Reads get_recent_failed(limit=50) and get_recent_completed(limit=50)
    once per call (not once per record). Matches on gpu_id (the queue's
    `id` field) — covers all agent types naturally without agent-specific
    logic. Only `pending` records are eligible; already-terminal records
    are never re-flipped.

    Limit=50: large enough that a 10-min tick interval plus typical queue
    throughput cannot push a terminal record past the window before we
    observe it. Raise to 200 if get_recent_* proves cheap and throughput
    grows significantly.

    Carve-outs (two):
    - fixer_retry + completed: left pending so the SHA-advance perceiver in
      _encode_gpu_results owns the terminal flip.
    - reviewer/reviewer_fresh + completed: left pending so the output-file
      verdict-encoder in _encode_gpu_results owns the terminal flip (reads
      the output file, parses JSON, writes "Reviewer verdict for PR #N:").
    Failed flips for both carve-out types are still permitted — a crashed
    job produces no output file, so the record must fail rather than wait.

    Reap pass (Leg 1, lapis-pm-stale-pending-dispatch-reaper-v0): a pending
    record with no terminal match above is also checked against get_active()
    and get_pending() — see _reap_verdict. A record whose gpu_id is provably
    gone from every queue directory, or stuck in active/ past its own
    timeout_seconds + STARTUP_STALE_GRACE_S, is flipped to failed with a
    failure_reason of "reaped:task_absent_from_queue" or
    "reaped:active_past_timeout" respectively. The carve-outs above are
    unaffected — reap only ever produces "failed", never "processed", so it
    never bypasses either carve-out's reasoning.
    """
    if _ClaudeQueue is None:
        return 0, True

    try:
        cq = _ClaudeQueue()
        failed_entries = cq.get_recent_failed(limit=50)
        completed_entries = cq.get_recent_completed(limit=50)
        active_entries = cq.get_active()
        pending_entries = cq.get_pending()
    except Exception:
        return 0, False

    # Build gpu_id → terminal-state index from both lists.
    # failed wins if a task somehow appears in both (shouldn't happen).
    terminal: dict[str, dict] = {}
    for entry in completed_entries:
        task_id = entry.get("id")
        if task_id:
            terminal[task_id] = {
                "state": "processed",
                "error": None,
                "completed_at": entry.get("completed_at"),
            }
    for entry in failed_entries:
        task_id = entry.get("id")
        if task_id:
            terminal[task_id] = {
                "state": "failed",
                "error": entry.get("error"),
                "completed_at": entry.get("completed_at"),
            }

    # Live-queue index for the reap pass below (Leg 1 of the stale-pending
    # reaper spec). A record whose gpu_id shows up in none of these four
    # buckets — pending / active / completed / failed — is provably gone;
    # one whose gpu_id sits in active/ past its own declared timeout is
    # evidence the runner that owned it died. Liveness is decided from this
    # queue state ONLY — no ps, no worktree stat, no filesystem probing.
    active_by_id: dict[str, dict] = {}
    for entry in active_entries:
        task_id = entry.get("id")
        if task_id:
            active_by_id[task_id] = entry
    pending_ids: set[str] = {e.get("id") for e in pending_entries if e.get("id")}

    records = load_dispatched(target_id)
    flipped = 0
    changed = False

    # Pre-load existing reconcile tags for de-dup: one audit comment per
    # gpu_id, never written again if the tag already exists in the JSONL.
    existing_dedup_tags: set[str] = set()
    for c in episodic.all_comments(target_id):
        for t in c.tags:
            if t.startswith("pm:dispatch-reconciled:gpu="):
                existing_dedup_tags.add(t)

    for rec in records:
        if rec.get("status") != "pending":
            # Only pending records are touched. Terminal records are left alone.
            continue
        gpu_id = rec.get("gpu_id")
        if not gpu_id:
            continue
        t = terminal.get(gpu_id)
        if t is None:
            t = _reap_verdict(gpu_id, active_by_id, pending_ids)
            if t is None:
                continue

        # fixer_retry carve-out: advance-perceiver (_fixer_completion_ts) is the
        # sole authority for fixer_retry → processed. If the queue says "completed"
        # for a fixer_retry, we leave the record as pending — _encode_gpu_results
        # will flip it when it confirms the PR advanced (SHA or body) or the job
        # is terminal with no advance (lost-dispatch net path).
        # Failed flips for fixer_retry are still permitted (the job crashed or
        # was rejected; that doesn't advance the cycle regardless).
        if rec.get("agent_type") == "fixer_retry" and t["state"] == "processed":
            continue  # advance-perceiver owns fixer_retry → processed

        # reviewer/reviewer_fresh carve-out: output-file verdict-encoder in
        # _encode_gpu_results is the sole authority for reviewer → processed.
        # If the queue says "completed" for a reviewer/reviewer_fresh, leave
        # the record as pending — _encode_gpu_results reads the output file,
        # parses the JSON verdict, writes the "Reviewer verdict for PR #N:"
        # episodic entry, and then flips to processed.
        # Failed flips for reviewers are still permitted (crashed job → no
        # output file coming; must fail fast rather than wait forever).
        if rec.get("agent_type") in _REVIEWER_AGENT_TYPES and t["state"] == "processed":
            continue  # output-file verdict-encoder owns reviewer → processed

        # auditor carve-out (fixer-reception-v0, leg 2, D6c): the output-file
        # result-encoder in _encode_gpu_results is the sole authority for
        # auditor → processed (it writes the pm:auditor comment + the
        # gate-(i) marker). Leave a "completed" auditor pending, exactly as
        # the reviewer carve-out above. Failed flips for auditors are still
        # permitted (a hard-killed/timeout job produces no output file; the
        # marker clear below makes the un-audited head re-dispatch).
        if rec.get("agent_type") == AUDITOR_AGENT_TYPE and t["state"] == "processed":
            continue  # output-file result-encoder owns auditor → processed

        # Flip the record to the terminal state.
        rec["status"] = t["state"]
        if t["error"] is not None:
            rec["error"] = t["error"]
        if t["completed_at"] is not None:
            rec["completed_at"] = t["completed_at"]
        if t.get("failure_reason") is not None:
            rec["failure_reason"] = t["failure_reason"]
        # L2.D2: stamp the served model at the terminal flip (scoped to the
        # dispatch-verified agent types; the carve-outs above keep
        # reviewer/fixer_retry processed-flips owned by _encode_gpu_results).
        _stamp_served_model(rec, target_id)
        changed = True
        flipped += 1

        # auditor failure marker clear (fixer-reception-v0, leg 2, D6c): on a
        # hard-kill / timeout flip the audit is lost but the head is still
        # un-audited — clear any pending pm/audit/<tid>/pr=N/dispatched
        # marker so the next tick re-dispatches (a failed audit is
        # retryable, unlike a completed one).
        if (rec.get("agent_type") == AUDITOR_AGENT_TYPE
                and t["state"] == "failed"
                and rec.get("pr_number") is not None):
            _clear_audit_dispatched(target_id, rec["pr_number"])

        # D3 fix (lapis-pm-reviewer-attempts-not-consumed-by-infra-v0): a
        # reviewer/reviewer_fresh job that crashed before writing an output
        # file (e.g. GravityWell reported not-serving) is flipped to "failed"
        # right here — status was already != "pending" by the time
        # _encode_gpu_results ran later this same tick, so its is_failure
        # branch (the only other call site of _record_reviewer_attempt_reason)
        # never sees it and last_reason stays None forever. Record the reason
        # here too, from the queue's `error` field — the one surface that DID
        # hold the real cause (verified live 2026-08-03: the queue's error
        # string was the verbatim "ERROR: local reviewer produced no verdict
        # (reason=gw_not_serving)"). Also classifies against the infra
        # allow-list (D1), same as the out_path-based path.
        if (rec.get("agent_type") in _REVIEWER_AGENT_TYPES
                and t["state"] == "failed"
                and rec.get("pr_number") is not None
                and t["error"]):
            _record_reviewer_attempt_reason(
                target_id, rec["pr_number"], rec.get("cycle", 1), t["error"],
                wait_s=_reviewer_dispatch_wait_seconds(rec),
                completed_at=rec.get("completed_at"),
            )

        # Close the project-slot + emit its deposit for slots whose terminal flip
        # is owned by THIS reconciler rather than _encode_gpu_results. The two
        # carve-outs above (fixer_retry/reviewer → processed) returned early and
        # are closed by _encode_gpu_results; the remaining case — a plain `fixer`
        # that handed off a PR and was marked completed by the queue — lands here.
        # Without this, fixer slots stayed `dispatched` forever (never auto-landed
        # /deposited), unlike reviewer slots which already close via the encoder.
        # Best-effort + idempotent (keyed on slot_id), so a double-call is safe.
        _close_slot_and_deposit(rec, target_id)

        # Audit comment — one per flip, de-duped by tag so a second call
        # with the same gpu_id in a later tick writes nothing.
        dedup_tag = f"pm:dispatch-reconciled:gpu={gpu_id}"
        if dedup_tag not in existing_dedup_tags:
            reaped_reason = t.get("failure_reason")
            if reaped_reason is not None:
                # Reap path (Leg 1): no queue error string to report — the
                # task is either absent from every queue directory or stuck
                # past its own timeout. Tag distinctly from a normal
                # reconcile flip so the audit trail names cause, not just effect.
                episodic.write_observation(
                    target_id,
                    f"Reaped dispatch {gpu_id}: pending → {t['state']} ({reaped_reason})",
                    extra_tags=["pm:dispatch-reconciled", "pm:dispatch-reaped", dedup_tag],
                )
            else:
                error_note = f" ({t['error']})" if t["error"] else ""
                episodic.write_observation(
                    target_id,
                    f"Reconciled dispatch {gpu_id}: pending → {t['state']}{error_note}",
                    extra_tags=["pm:dispatch-reconciled", dedup_tag],
                )
            existing_dedup_tags.add(dedup_tag)

    if changed:
        save_dispatched(target_id, records)

    return flipped, True


# ---------------------------------------------------------------------------
# Lost-dispatch detection and handling
# ---------------------------------------------------------------------------

def _collect_merged_pr_created_ats(target_id: str) -> list[str]:
    """Return PR creation timestamps from pm:pr-merged episodic observations.

    Parses 'created_at=<ts>' from observation content (written by
    _encode_merged_prs since the lost-dispatch feature landed). Falls back to
    the observation's own timestamp for older records that pre-date this field.
    """
    result: list[str] = []
    for c in episodic.all_comments(target_id):
        if not any(t.startswith("pm:pr-merged:") for t in c.tags):
            continue
        m = re.search(r"created_at=(\S+)", c.content)
        result.append(m.group(1) if m else c.ts)
    return result


def _earliest_fixer_retry_dispatch_ts(records: list[dict], pr_num: int) -> str:
    """Part A net (2026-08-24): the retry-chain start for a PR.

    Chain start = the earliest dispatch ts among records with
    agent_type == "fixer_retry" for this pr_number. Initial-fixer and
    reviewer records are excluded - that exclusion is the strict chain
    isolation that keeps unrelated prior advances (original fixer, other
    retry chains) from masking a genuinely lost retry. Returns "" when
    there is no matching record.
    """
    ts_list = [
        r.get("ts", "")
        for r in records
        if r.get("agent_type") == "fixer_retry" and r.get("pr_number") == pr_num
    ]
    ts_list = [t for t in ts_list if t]
    return min(ts_list) if ts_list else ""


def _find_lost_fixer_dispatches(
    target_id: str,
    records: list[dict],
    open_prs: list[dict],
    forgejo_ok: bool,
    merged_pr_nums: set[int] | None = None,
) -> tuple[list[dict], list[tuple[dict, dict | None]]]:
    """Classify terminal fixer dispatches with no corresponding PR as lost.

    Returns (needs_retry, needs_brief):
      needs_retry: original fixer records where lost_retry_count == 0 and no
                   pending retry child exists (never preempt a live fixer).
      needs_brief: (original, retry_child_or_None) pairs where lost_retry_count >= 1
                   and the retry child is absent or also terminal.

    A `lost` classification requires forgejo_ok=True — stale empty PR list is
    indistinguishable from real empty; returns ([], []) if Forgejo was unreachable.

    Only original dispatches (no parent_gpu_id) are classified; retry children
    are located by parent pointer and returned as the second tuple element for
    the brief path (so both dispatch IDs can appear in the brief body).
    """
    if not forgejo_ok:
        return [], []

    # Lazily loaded on first miss — avoids scanning episodic on ticks where no
    # terminal fixer dispatch exists or all are covered by open_prs.
    merged_pr_created_ats: list[str] | None = None
    # New: lazy caches for merge-state and head-SHA-advance gates (Change 1).
    seen_prs: set[int] | None = None
    merged_prs_for_target: set[int] | None = None

    needs_retry: list[dict] = []
    needs_brief: list[tuple[dict, dict | None]] = []

    for rec in records:
        if rec.get("agent_type") not in _INITIAL_FIXER_TYPES + ("fixer_retry",):
            continue
        if rec.get("parent_gpu_id"):
            continue  # Only classify originals, not retry children
        if rec.get("status") not in ("processed", "failed"):
            continue  # Not terminal — job may still open a PR (fixer) or be pending (fixer_retry)

        # fixer_retry lost classification: different criterion from initial fixer.
        # A fixer_retry is "lost" when it is terminal but produced neither a SHA
        # advance nor a description advance after its dispatch_ts.
        # (A processed fixer_retry with an advance is a normal completion — not lost.)
        if rec.get("agent_type") == "fixer_retry":
            dispatch_ts = rec.get("ts", "")
            pr_num = rec.get("pr_number")
            if pr_num is None:
                # lapis-pm-daemon-silent-gaps-v0 B2 (rev 3): a pr_number-None
                # fixer_retry that died before a PR number was recorded is no
                # longer invisible to lost-detection. The sibling criterion
                # (the pr_number-present path above) applies with the
                # pr_number-keyed nets reduced: terminal (the status gate above
                # already excludes in-flight records), no advance observed for
                # the target after dispatch_ts (a pr=None record has no
                # per-PR completion observation to check - the has_pr checks
                # below are the advance net), no PR opened/merged for the
                # target at/after dispatch_ts (guards against a PR that opened
                # but failed to stamp the record), and no pending retry child
                # (checked below). No new time gate - the sibling path has
                # none. fixer_staged records keep the rev-2 skip behavior
                # (this branch only runs for agent_type == "fixer_retry").
                has_pr = any(
                    pr.get("created_at", "") >= dispatch_ts for pr in open_prs
                )
                if not has_pr:
                    if merged_pr_created_ats is None:
                        merged_pr_created_ats = _collect_merged_pr_created_ats(target_id)
                    has_pr = any(ts >= dispatch_ts for ts in merged_pr_created_ats)
                if has_pr:
                    continue  # PR exists — not lost
            else:
                if _fixer_completion_ts(target_id, pr_num, dispatch_ts):
                    continue  # Advance observed — normal completion, not lost
                # Part A net (2026-08-24): a sibling in the SAME retry chain
                # advanced the PR after the chain started - the no-op retry's
                # work was not lost.
                chain_start = _earliest_fixer_retry_dispatch_ts(records, pr_num)
                if chain_start and _pr_advanced_since(target_id, pr_num, chain_start):
                    continue  # Sibling-in-chain advance - normal completion, not lost
                # Part C net (2026-08-24): the PR is live-merged - the retry's
                # work demonstrably landed even though no advance observation
                # was recorded.
                if merged_pr_nums is not None and pr_num in merged_pr_nums:
                    continue  # Merged PR - normal completion, not lost

            orig_gpu_id = rec.get("gpu_id", "")
            retry_child: dict | None = next(
                (r for r in records
                 if r.get("parent_gpu_id") == orig_gpu_id
                 and r.get("agent_type") == "fixer_retry"),
                None,
            )
            if retry_child is not None and retry_child.get("status") == "pending":
                continue  # Never preempt a live fixer_retry

            lost_retry_count = rec.get("lost_retry_count", 0)
            if lost_retry_count == 0:
                needs_retry.append(rec)
            else:
                needs_brief.append((rec, retry_child))
            continue  # fixer_retry classification done; skip initial-fixer logic below

        dispatch_ts = rec.get("ts", "")

        # Does a matching PR exist (created at or after this dispatch)?
        has_pr = any(pr.get("created_at", "") >= dispatch_ts for pr in open_prs)
        if not has_pr:
            if merged_pr_created_ats is None:
                merged_pr_created_ats = _collect_merged_pr_created_ats(target_id)
            has_pr = any(ts >= dispatch_ts for ts in merged_pr_created_ats)

        # Gate 1 (new): a merged PR's merged_at >= dispatch_ts, restricted to
        # _seen_pr_ids(target_id).  Covers the foyer-shaped case where the fixer
        # pushed onto an *existing* branch rather than opening a new PR.
        if not has_pr:
            if seen_prs is None:
                seen_prs = _seen_pr_ids(target_id)
            if merged_prs_for_target is None:
                merged_prs_for_target = _merged_pr_numbers_observed(target_id)
            for n in merged_prs_for_target & seen_prs:
                if _merged_at_for_pr(target_id, n) >= dispatch_ts:
                    has_pr = True
                    break

        # Gate 2 (new): a head-SHA-advance observation (pm:pr=<n>:sha=<sha>) at
        # ts >= dispatch_ts, restricted to _seen_pr_ids(target_id).  Covers the
        # case where the fixer pushed a commit and the PR is still open.
        if not has_pr:
            if seen_prs is None:
                seen_prs = _seen_pr_ids(target_id)
            for c in episodic.all_comments(target_id):
                if c.ts >= dispatch_ts:
                    for t in c.tags:
                        if t.startswith("pm:pr=") and ":sha=" in t:
                            try:
                                n = int(t[len("pm:pr="):].split(":sha=")[0])
                                if n in seen_prs:
                                    has_pr = True
                                    break
                            except (ValueError, IndexError):
                                pass
                if has_pr:
                    break

        if has_pr:
            continue  # PR exists; not lost

        # Locate the youngest fixer retry child (if any)
        orig_gpu_id = rec.get("gpu_id", "")
        retry_child: dict | None = next(
            (r for r in records
             if r.get("parent_gpu_id") == orig_gpu_id
             and r.get("agent_type") in _INITIAL_FIXER_TYPES),
            None,
        )
        # Never preempt a live fixer (child pending = retry in flight)
        if retry_child is not None and retry_child.get("status") == "pending":
            continue

        lost_retry_count = rec.get("lost_retry_count", 0)
        if lost_retry_count == 0:
            needs_retry.append(rec)
        else:
            needs_brief.append((rec, retry_child))

    return needs_retry, needs_brief


def _act_lost_fixer_retry(target_id: str, rec: dict) -> str:
    """Re-dispatch a lost fixer with the same intent. Retry budget: exactly 1.

    Increments lost_retry_count on the original record and appends a child
    dispatch record with parent_gpu_id pointing back to the original.
    Logs decision=fixer_lost:retrying:dispatch=<id>.
    """
    agent_type = rec.get("agent_type", "fixer")
    if agent_type == "fixer_retry":
        # Part B (2026-08-24): a genuinely lost fixer_retry no longer aborts the
        # tick (the raise used to propagate to tick_all and loop on
        # skipped=exception). Full fixer_retry re-dispatch (branch derivation
        # from the PR head ref) remains a deferred follow-up; this is a loud,
        # visible, one-shot skip so the tick completes and the case stays seen.
        pr_num = rec.get("pr_number")
        dispatch_id = rec.get("gpu_id", "?")
        # lapis-pm-daemon-silent-gaps-v0 B3: the skip_tag degenerates to
        # per-target when pr_number is None (a second stuck pr=None chain on
        # the same target would page nothing) - carry the dispatch gpu_id in
        # the tag for the pr=None class so each chain is one-page-per-chain.
        if pr_num is None:
            skip_tag = f"pm:lost:fixer_retry:skipped:pr=None:gpu={dispatch_id}"
        else:
            skip_tag = f"pm:lost:fixer_retry:skipped:pr={pr_num}"
        print(
            f"WARN: _act_lost_fixer_retry: lost fixer_retry target={target_id} "
            f"pr={pr_num} dispatch={dispatch_id} - no fixer_retry re-dispatch "
            "(deferred follow-up); loud skip, tick continues.",
            file=sys.stderr,
        )
        if not any(skip_tag in (c.tags or []) for c in episodic.all_comments(target_id)):
            # B3: the page body (and the upgraded observation) must carry the
            # exact recovery command template + the record's original intent -
            # the operator/LLM first-responder must be able to act from the
            # page alone. The HIGH page for the underlying death itself comes
            # from the queue layer (B1 + sibling A1); this NORMAL page is the
            # daemon-side signal that the retry chain is stuck and needs a
            # human. Deduped by the SAME skip_tag that gates the observation.
            intent = rec.get("intent", "(no intent)")
            recovery_cmd = (
                f"lapis-pm tick --target {target_id} --force-dispatch "
                f"fixer:\"{intent}\""
            )
            observation_text = (
                f"lost fixer_retry: pr={pr_num} dispatch={dispatch_id} - "
                "fixer_retry re-dispatch is a deferred follow-up; loud skip "
                "(tick no longer aborts). The retry chain needs a manual "
                "force-dispatch or review.\n"
                f"Recovery: {recovery_cmd}\n"
                f"Original intent: {intent}"
            )
            episodic.write_observation(
                target_id,
                observation_text,
                extra_tags=["pm:error", skip_tag],
            )
            try:
                from agents_core.notify import send_notification, Priority as _P
                send_notification(
                    message=(
                        f"lost fixer_retry: target={target_id} pr={pr_num} "
                        f"dispatch={dispatch_id} - fixer_retry re-dispatch is "
                        "a deferred follow-up; the retry chain is stuck and "
                        "needs a human.\n"
                        f"Recovery: {recovery_cmd}\n"
                        f"Original intent: {intent}"
                    ),
                    title=f"{target_id}: lost fixer_retry stuck (manual force-dispatch needed)",
                    priority=_P.NORMAL,
                )
            except Exception:
                pass
        return f"skip:lost_fixer_retry_undispatchable:pr={pr_num}"
    intent = rec.get("intent", "(no intent)")
    spec_summary = episodic.spec_summary(target_id)
    # For fixer_retry, carry pr_number and note that prior attempt was a no-op
    pr_number_val = str(rec.get("pr_number", "")) if agent_type == "fixer_retry" else ""
    dispatch_intent = intent
    if agent_type == "fixer_retry":
        dispatch_intent = (
            intent + "\n\nNote: prior attempt completed without pushing any code "
            "or updating the PR description."
        )
    vars_ = {
        "target_id": target_id,
        "spec_summary": spec_summary,
        "repo": rec.get("repo", ""),
        "question": dispatch_intent,
        "pr_number": pr_number_val,
        "slug": rec.get("slug", "forced"),
        "base_branch": "main",
        "existing_branch": f"lapis/{target_id}/forced",
        "intent_block": _intent_artifact.dispatch_block(target_id),
    }
    steer.inject_overlay(target_id, vars_, agent_type)
    _ensure_dispatch_owned(vars_.get("repo", ""))
    res = _SHAPER.dispatch(agent_type, target_id, dispatch_intent, vars_=vars_)
    _check_calcification(target_id)

    orig_gpu_id = rec.get("gpu_id", "?")
    new_record: dict = {
        "gpu_id": res.task_id,
        "spec_id": res.spec_id,
        "agent_type": agent_type,
        "intent": intent,
        "repo": rec.get("repo", ""),
        "ts": _now_iso(),
        "status": "pending",
        "retry_count": 0,
        "lost_retry_count": 0,
        "parent_gpu_id": orig_gpu_id,
    }
    # Carry pr_number so the child is perceived by _encode_gpu_results consistently
    if pr_number_val:
        new_record["pr_number"] = rec.get("pr_number")

    # Increment lost_retry_count on original and append child in one save
    records = load_dispatched(target_id)
    for r in records:
        if r.get("gpu_id") == orig_gpu_id:
            r["lost_retry_count"] = 1
            break
    records.append(new_record)
    save_dispatched(target_id, records)

    episodic.write_observation(
        target_id,
        f"Lost dispatch {orig_gpu_id}: retrying → {res.task_id}\nIntent: {intent}",
        extra_tags=["pm:lost-dispatch-retry", f"pm:gpu={res.task_id}",
                    f"pm:agent={agent_type}"],
    )
    return f"fixer_lost:retrying:dispatch={orig_gpu_id}"


_LOST_BRIEF_RESURFACE_WINDOW_HOURS = 24


def _lost_brief_resurface_attempt_tag(target_id: str, orig_id: str) -> str | None:
    """Return the ts of the most recent lost-brief resurface attempt for
    (target, orig_gpu), or None.

    The attempt record is the pm:lost-brief-resurface attempt observation
    (written when the resurface path runs) or the pm:lost-brief-resurface-failed
    observation (written when the key-set inside the resurface path fails).
    Both carry the pm:lost-original-gpu tag.
    """
    gpu_tag = f"pm:lost-original-gpu={orig_id}"
    latest: str | None = None
    for c in episodic.all_comments(target_id):
        if gpu_tag not in c.tags:
            continue
        if (
            "pm:lost-brief-resurface" in c.tags
            or "pm:lost-brief-resurface-failed" in c.tags
        ):
            if latest is None or c.ts > latest:
                latest = c.ts
    return latest


def _lost_brief_resurface_allowed(
    target_id: str, orig_id: str, now: datetime
) -> bool:
    """At most one resurface attempt per 24h per orig_gpu_id (spec
    lapis-pm-lost-brief-must-surface-v0 §3). A missing/unparseable attempt ts
    is treated as "no attempt yet" so the first resurface always runs."""
    latest = _lost_brief_resurface_attempt_tag(target_id, orig_id)
    if latest is None:
        return True
    try:
        last = datetime.fromisoformat(latest)
    except ValueError:
        return True
    # Normalize to the same tz-awareness as `now` so subtraction never
    # raises (production `now` is a naive Pacific wall clock; the attempt
    # ts is tz-aware Pacific from CommentStore).
    if last.tzinfo is not None and now.tzinfo is None:
        last = last.replace(tzinfo=None)
    elif last.tzinfo is None and now.tzinfo is not None:
        last = last.replace(tzinfo=now.tzinfo)
    return (now - last) >= timedelta(hours=_LOST_BRIEF_RESURFACE_WINDOW_HOURS)


def _act_lost_brief(
    target_id: str, original_rec: dict, retry_rec: dict | None
) -> str:
    """Emit a lost-dispatch brief (both attempts terminated, no PR). NORMAL priority.

    Brief body includes both dispatch IDs, error strings, and spec reference.
    Options: retry-again, amend-spec-and-retry, unbind.
    Logs decision=fixer_lost:briefing:dispatches=<id1>,<id2>.

    Idempotency guard: if a pm:brief-options comment tagged
    pm:lost-original-gpu=<orig_gpu_id> already exists for this dispatch,
    and the outstanding-brief mem key still matches, the brief is not
    re-composed.  Returns noop:lost-brief-suppressed:gpu=<id> in that case.

    Resurface (lapis-pm-lost-brief-must-surface-v0): when the guard finds a
    prior lost-dispatch brief-options comment but the outstanding-brief mem
    key is absent or stale, the brief may NEVER have been delivered (the
    2026-08-28 case: comment written, key never set).  The stale/absent
    branch then re-emits the brief through the normal compose/surface path
    below (bounded to one attempt per 24h per orig_gpu_id) instead of
    suppressing forever.
    """
    orig_id = original_rec.get("gpu_id", "unknown")
    orig_error = original_rec.get("error") or "no error recorded"
    retry_id = retry_rec.get("gpu_id", "none") if retry_rec else "none"
    retry_error = (retry_rec.get("error") or "no error recorded") if retry_rec else ""

    # --- Idempotency guard (Change 2) ---
    gpu_tag = f"pm:lost-original-gpu={orig_id}"
    existing_brief_id: str | None = None
    already_suppressed = False
    for c in episodic.all_comments(target_id):
        if gpu_tag not in c.tags:
            continue
        if "pm:lost-brief-suppressed" in c.tags:
            # A prior stale-mem tick already wrote the observation for this
            # orig_gpu. Don't write another — that's the unbounded-write bug.
            already_suppressed = True
            continue
        if "pm:brief-options" not in c.tags:
            continue
        try:
            data = json.loads(c.content)
        except (json.JSONDecodeError, AttributeError):
            continue
        if data.get("trigger") == "lost-dispatch":
            existing_brief_id = data.get("brief_id")

    if already_suppressed:
        # Already reported once — the mismatch stays visible from that first
        # write. No mem key is touched here; state repair is out of scope.
        return f"noop:lost-brief-suppressed:gpu={orig_id}"

    if existing_brief_id is not None:
        current_outstanding = get_outstanding_brief(target_id)
        if current_outstanding == existing_brief_id:
            # Brief is outstanding and matches — suppress entirely (noop).
            return f"noop:lost-brief-suppressed:gpu={orig_id}"
        else:
            # Mem key is stale or absent.  The guard cannot distinguish "the
            # brief was delivered and the key went stale" from "the brief was
            # never delivered and the key was never set" (the 2026-08-28
            # case).  Re-emit through the normal compose/surface path below
            # instead of suppressing forever — bounded to one attempt per
            # 24h per orig_gpu_id so a persistently failing key-set degrades
            # to a once-a-day loud observation, not per-tick spam.
            now = datetime.now(PACIFIC)
            if not _lost_brief_resurface_allowed(target_id, orig_id, now):
                return f"noop:lost-brief-suppressed:gpu={orig_id}"
            episodic.write_observation(
                target_id,
                f"Lost-brief resurface attempt: orig_gpu={orig_id} "
                f"existing_brief={existing_brief_id} "
                f"mem_key={current_outstanding!r} "
                f"attempt_ts={now.isoformat(timespec='microseconds')}",
                extra_tags=["pm:lost-brief-resurface", gpu_tag],
            )
            # Fall through: re-emit the brief via the normal path below.
    # --- End idempotency guard ---

    # Close the attribution slot for terminal fixer_retry records. The slot was
    # opened at dispatch and must close regardless of outcome. For initial fixer
    # records, the reconciler already called _close_slot_and_deposit; the call
    # here is idempotent and safe (keyed on slot_id, no double-deposit).
    # L2.D2: stamp the served model on the terminal record (idempotent — the
    # read is best-effort and the field is set from the queue yaml).
    _stamp_served_model(original_rec, target_id)
    _close_slot_and_deposit(original_rec, target_id)

    spec_ref = (episodic.spec(target_id) or "")[:80] or "(spec not found)"
    spec_path = room_str('planning.specs', f'{target_id}.md')

    query = (
        f"Two fixer dispatches for {target_id} terminated without opening a PR.\n"
        f"Original dispatch: {orig_id} — error: {orig_error}\n"
        f"Retry dispatch: {retry_id}"
        + (f" — error: {retry_error}" if retry_error else "")
        + f"\nSpec: {spec_path} — {spec_ref}"
    )

    b = brief.synthesize(
        target_id,
        trigger="lost-dispatch",
        query=query,
        notify=NotifyPriority.NORMAL,
        options_extra_tags=[gpu_tag],
    )
    # The key-set is part of the same surface path (the contract: the key's
    # presence means the brief is outstanding/surfaced).  If it fails to
    # persist, the resurface attempt record above already bounds re-fires to
    # one per 24h; a loud observation keeps the failure visible in
    # claude-view instead of a silent suppression.
    try:
        _set_brief_outstanding(target_id, b)
    except Exception as exc:
        episodic.write_observation(
            target_id,
            f"Lost-brief resurface key-set failed: orig_gpu={orig_id} "
            f"brief={b.comment_id} err={exc!r} — resurface re-bounded to "
            f"{_LOST_BRIEF_RESURFACE_WINDOW_HOURS}h",
            extra_tags=["pm:error", "pm:lost-brief-resurface-failed", gpu_tag],
        )
        raise

    dispatch_ids = f"{orig_id},{retry_id}" if retry_rec else orig_id
    return f"fixer_lost:briefing:dispatches={dispatch_ids}"


# ---------------------------------------------------------------------------
# Brief-decision directive consumer
# ---------------------------------------------------------------------------

_DIRECTIVES_BASE = room_path('directives.brief_decisions')


def _consume_brief_decisions(target_id: str) -> str | None:
    """Consume at most one brief-decision directive for target_id this tick.

    Glob: /srv/lapis/directives/brief-decisions/<target_id>__*.json  (top-level = pending).
    Also recovers any processing/<target_id>__*.json left by a prior crashed tick.

    Returns decision_str "brief_decision_applied:<brief_id>:<option_id>" if a
    directive was consumed, else None.
    """
    base = _DIRECTIVES_BASE
    pending_dir = base
    processing_dir = base / "processing"
    applied_dir = base / "applied"
    failed_dir = base / "failed"

    for d in (processing_dir, applied_dir, failed_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _load_candidates(directory: Path, glob_pattern: str) -> list[Path]:
        try:
            return sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
        except OSError:
            return []

    # Recover any file stuck in processing/ from a prior crashed tick.
    processing_candidates = _load_candidates(processing_dir, f"{target_id}__*.json")
    # Also pick up top-level pending files.
    pending_candidates = _load_candidates(pending_dir, f"{target_id}__*.json")

    # Process recovery files first (they're already renamed); then pending.
    candidates = [(f, True) for f in processing_candidates] + \
                 [(f, False) for f in pending_candidates]

    for src_path, already_processing in candidates:
        stem = src_path.stem  # <target_id>__<brief_id>
        processing_path = processing_dir / src_path.name
        applied_path = applied_dir / src_path.name
        failed_path = failed_dir / src_path.name

        # Load directive JSON.
        try:
            directive = json.loads(src_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            episodic.write_observation(
                target_id,
                f"brief-decision directive unreadable ({src_path.name}): {exc}",
                extra_tags=["pm:error"],
            )
            continue

        brief_id = directive.get("brief_id", "")
        option_id = directive.get("option_id", "")

        if not already_processing:
            # Atomic rename to processing/ before calling apply_decision.
            try:
                os.rename(src_path, processing_path)
            except OSError as exc:
                episodic.write_observation(
                    target_id,
                    f"brief-decision rename-to-processing failed ({src_path.name}): {exc}",
                    extra_tags=["pm:error"],
                )
                continue

        # Apply the decision (idempotent).
        result = brief.apply_decision(target_id, brief_id, option_id)

        ts = _now_iso()
        if result.get("ok"):
            # Write augmented payload (directive + result + applied_ts) into
            # applied/ then unlink processing/.  Earlier code wrote the augmented
            # payload then os.rename'd processing_path over it, which silently
            # clobbered the result/applied_ts fields.
            result_payload = json.dumps({**directive, "result": result, "applied_ts": ts},
                                        ensure_ascii=False)
            try:
                applied_path.write_text(result_payload)
                processing_path.unlink(missing_ok=True)
            except OSError:
                # best-effort: leave in processing if write/unlink fails
                pass
            return f"action:brief_decision_applied:{brief_id}:{option_id}"
        else:
            err = result.get("error", "unknown")
            failed_payload = json.dumps({**directive, "error": err, "failed_ts": ts},
                                        ensure_ascii=False)
            try:
                failed_path.write_text(failed_payload)
                processing_path.unlink(missing_ok=True)
            except OSError:
                pass
            episodic.write_observation(
                target_id,
                f"brief-decision directive failed ({brief_id}/{option_id}): {err}",
                extra_tags=["pm:error"],
            )
            # A failed directive is still "consumed" this tick — return None
            # so the rest of the decide chain can run.
            return None

    return None


# ---------------------------------------------------------------------------
# Eval-gate helpers (synapse-eval-gate-v1)
# ---------------------------------------------------------------------------

def _act_eval_gate_brief(
    target_id: str, result, repo: str
) -> str:
    """Emit an advisory brief for a regressed or unverified eval-gate result.

    Calls brief.synthesize() with the documented parameter mapping:
      trigger  = "synapse_eval_regressed" | "synapse_eval_unverified"
      query    = "PR #N retrieval-quality check"
      diff_snippet = result.summary_text   (delta table)
      screen_issues = []
      pr_number = result.pr_number
      notify    = None    (advisory only; no Pushover per invariant)

    Marks the result actioned in the cache and sets the outstanding brief.
    Does NOT mark the PR classified so the reviewer/fixer can proceed on the
    next tick.
    """
    trigger = (
        "synapse_eval_regressed"
        if result.status == "regressed"
        else "synapse_eval_unverified"
    )
    b = brief.synthesize(
        target_id,
        trigger=trigger,
        query=f"PR #{result.pr_number} retrieval-quality check",
        diff_snippet=result.summary_text,
        screen_issues=[],
        pr_number=result.pr_number,
        notify=None,
    )
    if _eval_gate:
        _eval_gate.mark_eval_actioned(result.head_sha)
    episodic.write_observation(
        target_id,
        f"Eval-gate brief emitted: PR #{result.pr_number} "
        f"sha={result.head_sha[:8]} status={result.status} cid={b.comment_id}",
        extra_tags=[
            f"pm:synapse-eval:pr={result.pr_number}",
            f"pm:synapse-eval:sha={result.head_sha[:8]}",
            f"pm:synapse-eval:status={result.status}",
        ],
    )
    _set_brief_outstanding(target_id, b, verified=True)
    return (
        f"action:eval_gate_brief:pr={result.pr_number}"
        f":status={result.status}:cid={b.comment_id}"
    )


def _act_regenerate_synapse_baseline(target_id: str) -> str:
    """Regenerate the synapse eval baseline at origin/main HEAD.

    Blocking operation with REGEN_TIMEOUT_S ceiling (240s).  Tears down
    worktree + ephemeral process unconditionally.  On success, writes
    /data/synapse/eval/baselines/main.json and logs a journal observation.
    On failure, leaves the prior baseline in place.

    Rate-limited: at most one retry per 30 minutes after a failure (tracked
    via pm/synapse-eval/regen-last-attempt in mem.db).
    """
    if not _eval_gate:
        return "noop:eval_gate_unavailable"

    episodic.write_observation(
        target_id,
        "Baseline regeneration started (synapse-eval-gate-v1)",
        extra_tags=["pm:synapse-eval:regen-started"],
    )
    result = _eval_gate.regenerate_baseline()
    if result is not None:
        _eval_gate.record_regen_attempt(success=True)
        sha = result.get("sha", "?")[:12]
        episodic.write_observation(
            target_id,
            f"Baseline regeneration succeeded: sha={sha} "
            f"metrics={list(result.get('metrics', {}).keys())}",
            extra_tags=["pm:synapse-eval:regen-succeeded", f"pm:synapse-eval:sha={sha}"],
        )
        return f"action:regenerate_synapse_baseline:succeeded:sha={sha}"
    else:
        _eval_gate.record_regen_attempt(success=False)
        episodic.write_observation(
            target_id,
            "Baseline regeneration failed - prior baseline preserved; "
            "will retry after rate-limit window (30 min)",
            extra_tags=["pm:synapse-eval:regen-failed", "pm:error"],
        )
        return "action:regenerate_synapse_baseline:failed"


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    target_id: str
    skipped: bool
    reason: str
    encoded: int          # number of percepts encoded as PM comments
    decision: str         # the action taken or noop:<reason> — see README.md § "Tick Decision Taxonomy"
    reconciled: int = 0   # number of dispatch records flipped this tick


def tick(target_id: str, allow_auto_land: bool = True) -> TickResult:
    store = TargetStore()
    target = store.get(target_id)
    if target is None:
        return TickResult(target_id, True, "target not found", 0, "noop:no_change")
    if not target.pm_bound:
        return TickResult(target_id, True, "target not pm_bound", 0, "noop:no_change")

    # 1. Pause guard with transition detection
    prev_state = get_pause_state(target_id) or "active"
    cur_state = "paused" if target.paused else "active"
    if cur_state != prev_state:
        episodic.write_observation(
            target_id,
            f"Tick state transition: {prev_state} → {cur_state}"
            + (f" (reason: {target.paused_reason})" if target.paused and target.paused_reason else ""),
        )
        set_pause_state(target_id, cur_state)
    if cur_state == "paused":
        return TickResult(target_id, True, "paused", 0, "noop:paused")

    # 1.5 Autonomy backstop sweep — D4 (lapis-pm-reviewer-leg-repair-v0): the
    # sweep is NO LONGER run from the 60s tick. A ~900s sweep inside a 60s
    # `Type=oneshot` tick capped at `TimeoutStartSec=300` was killed mid-run
    # (the 8x-timeout loop, 2026-09-07 13:03-13:38). The sweep now runs on a
    # dedicated `lapis-pm-backstop.timer` (15m cycle, `TimeoutStartSec=1200s`)
    # wired to the `lapis-pm backstop-sweep` CLI. `_backstop_sweep_tick()` is
    # a no-op (retained for the tick call site + the never-raise test).
    try:
        _backstop_sweep_tick()
    except Exception as e:
        logger.warning("[autonomy-backstop] tick-path sweep failed (non-fatal): %s", e)

    # 2. Cursor + perceive
    # Reconcile pending dispatch records against ClaudeQueue terminal state
    # before any encode pass that reads dispatched (prevents stale-pending wedge).
    reconciled = _reconcile_dispatched_with_queue(target_id)

    cursor = get_cursor(target_id)
    new_comments = [
        c for c in episodic.since(target_id, cursor)
        if c.author != episodic.PM_AUTHOR
    ]

    repo = target.pm_repo or ""
    _adopted_pr_number: int | None = target.data.get("adopted_pr_number")
    open_prs, forgejo_ok = (
        _perceive_prs(target_id, repo, _adopted_pr_number) if repo else ([], False)
    )

    # 2.5. Orphan PR reconciliation (L2.D1): detect and auto-adopt traceable deviant-branch PRs
    if repo and forgejo_ok:
        try:
            repo_name, owner = _repo_owner(repo)
            all_open_prs = get_open_prs(repo_name, owner=owner)
            _reconcile_orphan_prs(target_id, target, repo, all_open_prs)
            # Re-fetch the canonical open_prs in case reconciliation updated adopted_pr_number
            _adopted_pr_number = target.data.get("adopted_pr_number")
            open_prs, _ = _perceive_prs(target_id, repo, _adopted_pr_number)
        except Exception as e:
            episodic.write_observation(
                target_id, f"Orphan PR reconciliation failed: {e}",
                extra_tags=["pm:error", "pm:orphan-reconcile-failed"],
            )

    # 3. Encode
    encoded = 0
    directives = _encode_user_comments(target_id, new_comments)
    encoded += len(directives)  # user-authored already in JSONL; count them

    # 3.x Steer channel — context and directive are non-actions; consume them here
    # so they are registered before any decide-phase early-return can skip them.
    # Emergency is NOT consumed here; its file stays pending until the decide-phase
    # pause commits (durability — see step 4.0a below).
    for s in steer.consume_pending_steers(target_id, types=("context", "directive")):
        stype = s.get("type")
        if stype == "context":
            episodic.write_observation(
                target_id,
                f"[steer:context] {s['message']}",
                extra_tags=["pm:steer:context", f"pm:steer:ts={s['ts']}"],
            )
            encoded += 1
        elif stype == "directive":
            steer.set_directive_overlay(target_id, s["message"])
            episodic.write_observation(
                target_id,
                f"[steer:directive] Registered overlay for next fixer dispatch: {s['message']}",
                extra_tags=["pm:steer:directive", f"pm:steer:ts={s['ts']}"],
            )
            encoded += 1

    seen_pr_ids = _seen_pr_ids(target_id)
    new_prs = _encode_new_prs(target_id, repo, open_prs, seen_pr_ids)
    encoded += len(new_prs)

    # Track PR head SHA advances (must precede _encode_gpu_results so the SHA
    # observation is visible when fixer_retry completion is checked below).
    encoded += _encode_pr_sha_updates(target_id, open_prs)

    # D6a (fixer-reception-v0, leg 2): a NEW salvage-shaped PR on this
    # pm_bound target triggers the auditor (the LLM-powered receiving role
    # for non-passing fixer terminations). ENCODE-phase dispatch — separate
    # from the decide phase's single FIFO action, which is unchanged (spec
    # Invariant 4). Runs AFTER _encode_pr_sha_updates so the new head's sha
    # observation is visible to the gate (i) marker check (the marker's
    # VALUE is the audited head sha; a fresh salvage with no marker always
    # triggers, and a same-head re-dispatch is blocked by the marker). The
    # three gates (marker/pending/budget) bound the dispatch to exactly one
    # per head.
    for _new_pr in new_prs:
        try:
            _auditor_action = _maybe_dispatch_auditor_salvage(
                target_id, repo, _new_pr
            )
        except Exception as _auditor_exc:
            logger.warning("auditor salvage trigger failed: %s", _auditor_exc)
            _auditor_action = None
        if _auditor_action is not None:
            encoded += 1
            # D6a is an encode-phase side effect (fixer-reception-v0 Invariant 4: the
            # decide-phase FIFO action is unchanged). It must NOT assign decision_str,
            # which is first bound in the decide phase below - the prior read here was a
            # reference-before-init UnboundLocalError. The dispatch is already recorded
            # via episodic.write_dispatch and counted in `encoded`.

    # Track PR description changes (body fingerprint), also before _encode_gpu_results
    # so a description-only fixer_retry is perceived as complete this tick.
    encoded += _encode_pr_body_updates(target_id, open_prs)

    # Friction capture harvest (operational-learning-friction-capture-v0, Unit 1a).
    # Bookkeeping only — writes to the friction queue, never to pm/* decide state.
    encoded += _encode_friction_harvest(target_id, repo, open_prs)

    gpu_encoded, failed_dispatches = _encode_gpu_results(target_id)
    encoded += gpu_encoded

    # Check Forgejo for newly merged PRs and write pm:pr-merged observations.
    # This is bookkeeping (encoding), not action — safe to do before decide.
    # Capture the count so the decide phase knows whether a merge just happened
    # (used by the baseline-regeneration trigger, which requires condition (a)
    # "a Synapse PR was just encoded as merged" per spec §Deliverables 2).
    _merged_this_tick = _encode_merged_prs(target_id, repo)
    encoded += _merged_this_tick

    # Reconcile a surviving head branch on a confirmed merge (L1). Bookkeeping,
    # not a decide-phase action — scoped to this single target_id only. See
    # _reconcile_surviving_head_branch for why this is bounded to one probe.
    _reconcile_surviving_head_branch(target_id, repo)

    # Classify lost fixer dispatches (terminal job, no PR produced).
    # Must run after encode so freshly-flipped records are visible.
    _lost_all_records = load_dispatched(target_id)
    # Part C net (2026-08-24): live-merged PR numbers unblock the lost-classifier.
    # One Forgejo call per distinct PR number; catch-all per PR (an exception
    # leaves the PR out of the set - it must never re-enter skipped=exception).
    _merged_pr_nums: set[int] = set()
    _retry_pr_nums = sorted({
        r.get("pr_number") for r in _lost_all_records
        if r.get("agent_type") == "fixer_retry" and r.get("pr_number") is not None
    })
    for _pn in _retry_pr_nums:
        try:
            if _is_pr_merged(target_id, _pn):
                _merged_pr_nums.add(_pn)
        except Exception:
            pass
    _lost_needs_retry, _lost_needs_brief = _find_lost_fixer_dispatches(
        target_id, _lost_all_records, open_prs, forgejo_ok,
        merged_pr_nums=_merged_pr_nums,
    )

    # 4. Decide (priority order, single action)
    decision_str = "noop:no_change"
    pm_authority = target.pm_authority
    pm_verification = target.data.get("pm_verification", "pm-live-test")

    # 4.0a Emergency steer — halt the run (highest-priority decide branch).
    # File is read non-destructively; pause is committed and saved FIRST;
    # only then is the file moved to applied/ so no crash window loses the halt.
    _emergencies = steer.peek_emergencies(target_id)
    if _emergencies:
        _emergency = _emergencies[0]
        episodic.write_observation(
            target_id,
            f"[steer:emergency] {_emergency['message']}",
            extra_tags=["pm:steer:emergency", f"pm:steer:ts={_emergency['ts']}"],
        )
        set_pause_state(target_id, "paused")
        target.set_paused(True, reason=f"steer:emergency: {_emergency['message'][:100]}")
        target.save()
        steer.mark_applied(_emergency["_path"], disposition="emergency_paused")
        _ts = _emergency["ts"]
        set_cursor(target_id, _now_iso())
        return TickResult(target_id, False, "ok", encoded,
                          f"action:steer_emergency_applied:{_ts}",
                          reconciled=reconciled)

    # 4.0 Brief-decision directive consumer — highest-priority decide branch.
    # If a directive file exists for this target, apply it and skip the rest.
    brief_decision_str = _consume_brief_decisions(target_id)
    if brief_decision_str is not None:
        set_cursor(target_id, _now_iso())
        return TickResult(target_id, False, "ok", encoded, brief_decision_str,
                          reconciled=reconciled)

    if directives:
        # v1: surface directives as a brief if any are recent and we don't
        # already have an outstanding brief; never auto-execute imperatives.
        # Held-fault diagnostics (unanchored/tampered/etc.) are already
        # surfaced by _encode_user_comments (spec
        # lapis-pm-signed-directive-acceptance-v0 §4.4); this is just the
        # routine "a directive arrived" observation.
        d = directives[-1]
        # D7 (attestation-contract-v0 leg 2): the directive-seen observation
        # carries the open-PR head baseline at directive time (REV 2 pin) so
        # the outcome detector can later compare the current head against it
        # without any Forgejo call of its own. `pm:pr-head=none` when no PR
        # is open (any later open PR counts as an advance).
        _dir_pr_heads = [
            ((p.get("head") or {}).get("sha") or "")
            for p in (open_prs or [])
        ]
        _dir_baseline = _dir_pr_heads[0] if _dir_pr_heads else "none"
        episodic.write_observation(
            target_id,
            f"Directive received from {d.author} at {d.ts}:\n{d.content[:400]}",
            extra_tags=[
                "pm:directive-seen",
                f"pm:directive-id={d.id}",
                f"pm:pr-head={_dir_baseline}",
            ],
        )
        encoded += 1
        b = brief.synthesize(target_id, trigger=f"user directive: {d.content[:80]}",
                             query=d.content, notify=None)
        _set_brief_outstanding(target_id, b)
        # D7 REV 4 pin: the directive-seen observation additionally records
        # the raised brief's comment id (pm:brief-id=<cid>, or
        # pm:brief-id=none if the raise failed) so the outcome detector's
        # outcome (c) is keyed on the SPECIFIC brief. The cid is read back
        # from the mem slot the raise just wrote (best-effort: a failed
        # raise records `none`); the amend is a separate tagged observation
        # on the same target, so the detector can pair it with the
        # directive-seen observation it belongs to.
        _dir_brief_cid = get_outstanding_brief(target_id) or "none"
        try:
            episodic.write_observation(
                target_id,
                f"Directive brief raised: cid={_dir_brief_cid}",
                extra_tags=[
                    "pm:directive-brief-raised",
                    f"pm:brief-id={_dir_brief_cid}",
                    f"pm:directive-id={d.id}",
                ],
            )
        except Exception as _e:
            logger.warning(
                "[directive] brief-id amend failed for %s: %s", target_id, _e
            )
        decision_str = f"action:directive_brief:cid={b.comment_id}"

    elif open_prs:
        # Skip PRs already classified this binding — prevents re-screening a
        # held PR every 10 min. Classification is reset when the PR is closed
        # or the target is rebound with a fresh spec.
        classified_ids = _classified_pr_ids(target_id)
        actionable_prs = [p for p in open_prs if p.get("number") not in classified_ids]
        if not actionable_prs:
            decision_str = "noop:no_change"
        else:
            # Pick the lowest-numbered PR (FIFO) so the same one drives action
            # until resolved.
            pr = min(actionable_prs, key=lambda p: p.get("number", 1 << 30))
            pr_number_sel = pr.get("number", 0)

            # Eval-gate: for synapse PRs, run quality check before reviewer dispatch.
            # Skip if reviewer is already pending (avoid race with in-flight review).
            _eval_gate_handled = False
            if _eval_gate and repo == "synapse":
                if not _has_pending_reviewer_for_pr(target_id, pr_number_sel):
                    try:
                        _eg_result = _eval_gate.evaluate_pr(target_id, pr, repo)
                    except Exception as _eg_exc:
                        logger.warning("eval_gate.evaluate_pr raised: %s", _eg_exc)
                        _eg_result = None
                    if _eg_result is not None:
                        _already_actioned = _eval_gate.is_eval_actioned(_eg_result.head_sha)
                        if not _already_actioned:
                            if _eg_result.status == "clean":
                                # Percept tag only; fall through to normal dispatch
                                episodic.write_observation(
                                    target_id,
                                    f"Eval-gate: PR #{pr_number_sel} "
                                    f"sha={_eg_result.head_sha[:8]} status=clean",
                                    extra_tags=[
                                        f"pm:synapse-eval:pr={pr_number_sel}",
                                        f"pm:synapse-eval:sha={_eg_result.head_sha[:8]}",
                                        "pm:synapse-eval:status=clean",
                                    ],
                                )
                                _eval_gate.mark_eval_actioned(_eg_result.head_sha)
                                # Fall through to _decide_for_pr below
                            elif _eg_result.status in ("regressed", "unverified"):
                                # Advisory brief — consumes single-action slot
                                decision_str = _act_eval_gate_brief(
                                    target_id, _eg_result, repo
                                )
                                _eval_gate_handled = True

            if not _eval_gate_handled:
                decision = _decide_for_pr(target_id, repo, pr, pm_authority, pm_verification)
                if decision.kind == "merge":
                    decision_str = _act_merge(target_id, decision.payload)
                elif decision.kind == "hold_brief":
                    decision_str = _act_brief(target_id, trigger="held PR", hold=True,
                                              payload=decision.payload)
                elif decision.kind == "advisory_brief":
                    decision_str = _act_brief(target_id, trigger="advisory PR", hold=False,
                                              payload=decision.payload)
                elif decision.kind == "dispatch_reviewer":
                    p = decision.payload
                    decision_str = _act_dispatch_reviewer(
                        target_id, p["pr"], p["cls"], p["mode"], p["cycle"],
                    )
                elif decision.kind == "dispatch_fixer_retry":
                    decision_str = _act_dispatch_fixer_retry(target_id, decision.payload)
                elif decision.kind == "review_exhausted_brief":
                    decision_str = _act_brief_review_exhausted(target_id, decision.payload)
                elif decision.kind == "reviewer_attempt_ceiling":
                    decision_str = _act_reviewer_attempt_ceiling_pause(target_id, decision.payload)
                elif decision.kind == "reviewer_infra_budget_exhausted":
                    decision_str = _act_reviewer_infra_budget_pause(target_id, decision.payload)
                elif decision.kind == "reviewer_infra_backoff":
                    p = decision.payload
                    decision_str = f"noop:reviewer_infra_backoff:until={p['next_retry_at']}"
                elif decision.kind == "noop_reviewer_in_flight":
                    p = decision.payload
                    decision_str = f"noop:reviewer_in_flight:pr={p['pr_number']}:cycle={p['cycle']}"
                elif decision.kind == "noop_fixer_in_flight":
                    p = decision.payload
                    decision_str = f"noop:fixer_in_flight:dispatch={p['dispatch_id']}"
                else:
                    # noop_no_change or unknown — single-action discipline: do nothing
                    decision_str = "noop:no_change"

    elif failed_dispatches:
        rec = failed_dispatches[-1]
        if rec.get("retry_count", 0) < MAX_DISPATCH_RETRIES:
            decision_str = _act_retry(target_id, rec)
        else:
            b = brief.synthesize(
                target_id,
                trigger=f"shaped agent {rec.get('agent_type')} failed after {rec.get('retry_count')} retries",
                query=rec.get("intent", ""),
                notify=NotifyPriority.HIGH,
            )
            _set_brief_outstanding(target_id, b)
            decision_str = f"action:abandon_brief:cid={b.comment_id}"

    elif _already_satisfied_pending(target_id) is not None:
        decision_str = _act_auto_land_already_satisfied(target_id)

    elif _already_satisfied_invalid_pending(target_id) is not None:
        decision_str = _act_brief_already_satisfied_invalid(target_id)

    elif _lost_needs_retry:
        # Lost dispatch: terminal fixer job, no PR produced, first loss → retry once.
        decision_str = _act_lost_fixer_retry(target_id, _lost_needs_retry[0])

    elif _lost_needs_brief:
        # Lost dispatch: retry also produced no PR → surface to human.
        _orig, _retry_rec = _lost_needs_brief[0]
        decision_str = _act_lost_brief(target_id, _orig, _retry_rec)

    elif (
        _eval_gate
        and _merged_this_tick > 0
        and _eval_gate.should_regenerate_baseline(repo)
    ):
        # Baseline regeneration: fires post-merge when baseline is stale/missing.
        # Spec §Deliverables 2: requires BOTH (a) a Synapse PR just encoded as
        # merged this tick AND (b) the baseline is stale or missing.
        # Mirrors auto-land shape — consumes the single-action slot.
        decision_str = _act_regenerate_synapse_baseline(target_id)

    elif allow_auto_land and _is_auto_land_eligible(target_id):
        decision_str = _act_auto_land(target_id)

    # 4.5 Refine noop decision for chain-mode legs waiting on dependencies.
    if decision_str == "noop:no_change":
        _deps = target.data.get("depends_on") or []
        if _deps:
            from . import chain as _chain_mod
            _landed = _chain_mod.landed_tids()
            _unsatisfied = [d for d in _deps if d not in _landed]
            if _unsatisfied:
                decision_str = f"noop:awaiting_chain_dependency:waiting_on={_unsatisfied[0]}"

    # 4.6. Persist review-state cache (best-effort visibility for claude-view).
    try:
        _persist_review_state_cache(target_id, target, open_prs)
    except Exception as e:
        episodic.write_observation(
            target_id, f"review-state cache write failed: {e}",
            extra_tags=["pm:error"],
        )

    # 4.7. Persist the per-target PR-state record (D7, attestation-contract-v0
    # leg 2): the tick's PR-perception site writes pm/pr-head/<tid> once per
    # tick (one write site; the directive-outcome detector reads it and makes
    # NO Forgejo call of its own). Best-effort — never block the tick.
    try:
        _persist_pr_head_record(target_id, open_prs)
    except Exception as e:
        episodic.write_observation(
            target_id, f"pr-head record write failed: {e}",
            extra_tags=["pm:error"],
        )

    # 5. Advance cursor to now (we've considered everything as of this tick).
    set_cursor(target_id, _now_iso())
    # D6 (attestation-contract-v0 leg 2): persist the decision tag at the
    # tick-exit result-encoding site so the dedicated stall checker's page
    # can carry the target's last decision (best-effort — never block exit).
    try:
        _mem().set(_last_decision_key(target_id), decision_str,
                   tags=["lapis-pm", "last-decision"])
    except Exception:
        pass
    return TickResult(target_id, False, "ok", encoded, decision_str, reconciled=reconciled)


def tick_all() -> list[TickResult]:
    """Tick every pm_bound target. Returns one TickResult per target.

    At most one auto-land fires per tick_all call. When multiple targets are
    land-eligible, the oldest-bound (earliest spec:bound comment) is chosen;
    others have auto-land suppressed and become eligible next tick.
    """
    # Pre-perceive health probe — one probe per tick_all call, not per target.
    reachable, probe_reason = probe_forgejo_health()
    if not reachable:
        print(f"[forgejo:unreachable] reason={probe_reason}", flush=True)
        fails = _get_forgejo_consecutive_fails() + 1
        _set_forgejo_consecutive_fails(fails)
        if fails == FORGEJO_UNREACHABLE_THRESHOLD:
            _notify_forgejo_unreachable()
        store = TargetStore()
        bound = [t for t in store.load_all() if t.pm_bound]
        # Cursors are NOT advanced — next healthy tick re-perceives from the same point.
        skipped = [
            TickResult(t.id, True, "forgejo_unreachable", 0, "skipped:forgejo_unreachable")
            for t in bound
        ]
        for r in skipped:
            print(
                f"[{r.target_id}] skipped={r.skipped} reason={r.reason}"
                f" encoded={r.encoded} decision={r.decision}",
                flush=True,
            )
        return skipped
    # Probe succeeded — reset consecutive-fail counter.
    _set_forgejo_consecutive_fails(0)

    # Deploy currency check — alerts if the deploy clone has drifted behind origin/main.
    # Runs once per tick_all so timer outages or persistent pull failures are never silent.
    try:
        _check_deploy_currency()
    except Exception:
        pass  # best-effort — must not block target processing

    # Deploy-inventory reconciler — derives the real deployment surface and
    # reconciles it against the maps above (lapis-pm-deploy-inventory-reconciler-v0).
    # Cooldown-gated internally; runs once per tick_all so a full pass is cheap.
    try:
        _reconcile_deploy_inventory()
    except Exception:
        pass  # best-effort — must not block target processing

    # Decided-gem reconciler: process brief-gems Erah has decided on the Desk.
    # Runs once per tick_all (cross-target). Fail-soft — never blocks target processing.
    try:
        from . import brief_gem as _brief_gem
        _gem_actions = _brief_gem.reconcile_decided_gems()
        if _gem_actions:
            logger.info("[brief-gem:reconcile] %s", "; ".join(_gem_actions))
    except Exception as _bg_exc:
        logger.warning("[brief-gem:reconcile] failed (non-fatal): %s", _bg_exc)

    store = TargetStore()
    bound = [t for t in store.load_all() if t.pm_bound]

    # D6/D7 stall detectors (attestation-contract-v0 leg 2): run at the
    # tick-pass location, AFTER the Forgejo probe succeeded (the
    # forgejo_unreachable early-return above owns that episode and the
    # detectors' Forgejo-down suppression pin skips paging while the fail
    # counter is at/above threshold). Best-effort — a detector failure
    # never blocks target processing.
    try:
        _check_tick_stalls(store, datetime.now(PACIFIC))
        _check_directive_stalls(store, datetime.now(PACIFIC))
    except Exception as e:
        logger.warning("[stall-check] detectors failed (non-fatal): %s", e)

    # Pre-select which target (if any) gets the auto-land slot this tick.
    # This check uses observations from previous ticks; targets that become
    # eligible for the first time this tick are deferred to the next tick.
    # Paused targets are excluded here (not just inside tick()): a paused
    # target that meets every other auto-land condition would otherwise sort
    # first by oldest bind, win the single slot, and immediately noop:paused —
    # starving every other land-eligible target for as long as it stays paused.
    eligible_ids = sorted(
        [t.id for t in bound if not t.paused and _is_auto_land_eligible(t.id)],
        key=_spec_bound_ts,
    )
    auto_land_chosen = eligible_ids[0] if eligible_ids else None

    results: list[TickResult] = []
    for t in bound:
        try:
            results.append(tick(t.id, allow_auto_land=(t.id == auto_land_chosen)))
        except Exception as e:
            # Advance cursor even on exception so the same percepts don't get
            # re-encoded as duplicate observations on the next tick.
            try:
                set_cursor(t.id, _now_iso())
            except Exception:
                pass
            results.append(TickResult(t.id, True, f"exception: {e}", 0, "noop:no_change"))
    return results
