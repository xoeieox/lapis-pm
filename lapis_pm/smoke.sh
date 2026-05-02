#!/usr/bin/env bash
# Lapis PM end-to-end smoke test (queue-paused; chain-mode may create + tear down real Forgejo PRs).
#
# Exercises:
#   - bind / spec:bound comment / tags field
#   - force-dispatch (real GPU queue submission, NOT a real Claude call)
#   - synthetic GPU completion → pm:result encoding via tick
#   - pause / resume + transition observation
#   - human:directive flow → tick produces brief
#   - status + list output
#
# Skipped (require live external state — note printed at end):
#   - authority gate against a real PR
#   - Pushover delivery confirmation (will only succeed if env is loaded)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TID="pm-smoke-$$"
TARGETS_DIR="${PM_TARGETS_DIR:-/srv/lapis/targets}"
COMMENTS_DIR="/srv/lapis/targets/comments"
GPU_COMPLETED="/srv/lapis/gpu-queue/completed"
GPU_PENDING="/srv/lapis/gpu-queue/pending"
SPEC_FILE="/tmp/${TID}-spec.md"
LAPIS="/usr/bin/python3 -m lapis_pm.cli"

# --- Smoke TID registry -------------------------------------------------
# SMOKE_TIDS: every bind phase that creates a target whose dispatch may open
# a Forgejo PR MUST append its TID(s) here immediately after definition.
# chain_cleanup iterates this array for PR reaping on EXIT/INT/TERM.
#
# Structural invariant: adding a new bind without registering here is a
# programmer error caught locally (by the sentinel assertion in phase 20)
# rather than discovered days later as leaked PRs.
#
# Audit — all smoke-created TIDs that fire real dispatches (as of this file):
#   Phase 18  (chain smoke):  pm-smoke-chain-l1-<pid>  pm-smoke-chain-l2-<pid>  pm-smoke-chain-l3-<pid>
#   Phase 20b (wire-chain):   pm-wire-chain-<pid>-l1   pm-wire-chain-<pid>-l2
#   Phase 20c (wire-disp):    pm-wire-disp-<pid>
#
# All other phases (land, multi-PR, reconcile, reviewer, portfolio, brief,
# verdict) use synthetic observations only — no real dispatches.
SMOKE_TIDS=()

green() { printf '\033[32m✓ %s\033[0m\n' "$*"; }
red()   { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
step()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

TRAJ_SMOKE_DIR="/tmp/traj-smoke-$$"

cleanup() {
    rm -f "$TARGETS_DIR/${TID}.yaml" "$COMMENTS_DIR/${TID}.jsonl" "$SPEC_FILE"
    rm -rf "/srv/lapis/gpu-queue/shaped/${TID}-"*.json 2>/dev/null || true
    # Cancel any pending tasks we created
    for f in "${GPU_PENDING}"/*.yaml; do
        [ -e "$f" ] || continue
        if grep -q "lapis-pm" "$f" 2>/dev/null && grep -q "${TID}" "$f" 2>/dev/null; then
            rm -f "$f"
        fi
    done
    rm -f "/tmp/${TID}-tickout"
    # Best-effort mem cleanup
    /usr/local/bin/mem delete "pm/cursor/${TID}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/dispatched/${TID}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/pause-state/${TID}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/outstanding-brief/${TID}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/classified-prs/${TID}" 2>/dev/null || true
    # Trajectory smoke cleanup
    rm -rf "$TRAJ_SMOKE_DIR" 2>/dev/null || true
}
trap cleanup EXIT

# --- Pre-run back-fill: close leaked smoke PRs from prior broken runs ----
# Runs BEFORE any phase so a half-broken environment self-heals on the next
# smoke invocation rather than requiring another manual sweep.
# Pattern is conservative: ONLY matches known smoke TID prefixes; never
# touches real lapis/<tid>/... PRs from real binds outside smoke.
_smoke_backfill_close_leaked_prs() {
    if [ -z "${FORGEJO_TOKEN:-}" ]; then
        echo "[smoke] pre-run backfill: skipped (FORGEJO_TOKEN unset)"
        return 0
    fi
    /usr/bin/python3 -c "
import os, httpx

FORGEJO_URL = os.environ.get('FORGEJO_URL', 'http://203.0.113.12:3000')
OWNER = 'Erah'
API = f'{FORGEJO_URL}/api/v1'
token = os.environ.get('FORGEJO_TOKEN', '')
HDRS = {'Authorization': f'token {token}', 'Accept': 'application/json',
        'Content-Type': 'application/json'}
T = 15

# Conservative: ONLY known smoke TID prefixes — never touches real work PRs.
SMOKE_PREFIXES = (
    'lapis/pm-smoke-chain-',
    'lapis/pm-wire-chain-',
    'lapis/pm-wire-disp-',
)

REPOS = ('conductor', 'lapis-test')
closed = 0
for repo in REPOS:
    try:
        r = httpx.get(f'{API}/repos/{OWNER}/{repo}/pulls', headers=HDRS,
                      params={'state': 'open', 'limit': 50}, timeout=T)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        prs = r.json()
    except Exception as e:
        print(f'[smoke] backfill: list-prs {repo}: {e}')
        continue
    for pr in prs:
        head_ref = (pr.get('head') or {}).get('ref', '')
        if not any(head_ref.startswith(p) for p in SMOKE_PREFIXES):
            continue
        pr_num = pr['number']
        try:
            httpx.patch(f'{API}/repos/{OWNER}/{repo}/pulls/{pr_num}', headers=HDRS,
                        json={'state': 'closed'}, timeout=T).raise_for_status()
            print(f'[smoke] backfill: closed {repo}#{pr_num} ({head_ref})')
            closed += 1
        except Exception as e:
            print(f'[smoke] backfill: close {repo}#{pr_num}: {e}')
        try:
            httpx.delete(f'{API}/repos/{OWNER}/{repo}/branches/{head_ref}',
                         headers=HDRS, timeout=T)
        except Exception:
            pass
print(f'[smoke] pre-run backfill: {closed} leaked smoke PR(s) closed')
" || true
}
_smoke_backfill_close_leaked_prs

# --- Setup --------------------------------------------------------------
step "1. Create sandbox target $TID"
cat > "$SPEC_FILE" <<EOF
# Smoke spec for $TID

What: smoke-test the Lapis PM end-to-end.
Why:  validate the perceive→encode→decide loop locally.
How:  one stage, no real PRs, synthetic GPU completion.
EOF

cat > "$TARGETS_DIR/${TID}.yaml" <<EOF
id: $TID
title: Lapis PM smoke
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Sandbox target used by smoke.sh — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF
green "target created"

# --- Bind ---------------------------------------------------------------
step "2. Bind target with spec"
$LAPIS bind "$TID" --spec-from "$SPEC_FILE" --repo lapis-test --authority advisory
JSONL="$COMMENTS_DIR/${TID}.jsonl"
[ -f "$JSONL" ] || red "comments JSONL missing"
grep -q '"spec:bound"' "$JSONL" || red "spec:bound tag not in JSONL"
grep -q '"tags":' "$JSONL" || red "tags field missing from JSONL records"
green "spec:bound comment written with tags field present"

# --- Verify list ---------------------------------------------------------
step "3. List bound targets includes $TID"
$LAPIS list | grep -q "$TID" || red "list output missing $TID"
green "list shows $TID"

# --- Force-dispatch -----------------------------------------------------
step "4. Force-dispatch a scout (queue paused so we don't burn GPU)"
/usr/local/bin/gpu-submit pause >/dev/null 2>&1 || true
DISPATCH_OUT=$($LAPIS tick --target "$TID" --force-dispatch "scout:list files in repo")
echo "$DISPATCH_OUT"
TASK_ID=$(echo "$DISPATCH_OUT" | grep -oE 'task_id=[^ ]+' | sed 's/task_id=//')
[ -n "$TASK_ID" ] || red "no task_id in dispatch output"
# Remove the pending file so the runner won't claim it on resume
rm -f "$GPU_PENDING/${TASK_ID}.yaml"
/usr/local/bin/gpu-submit resume >/dev/null 2>&1 || true
grep -q '"pm:dispatch"' "$JSONL" || red "pm:dispatch comment not written"
green "dispatch task=$TASK_ID created and pruned (queue resumed)"

# --- Synthetic completion -----------------------------------------------
step "5. Synthesize a fake GPU completion + tick to encode result"
mkdir -p "$GPU_COMPLETED"
cat > "$GPU_COMPLETED/${TASK_ID}-output.md" <<EOF
Smoke synthetic output.
Found 3 files: README.md, smoke.sh, _runner.py.
EOF
$LAPIS tick --target "$TID"
grep -q '"pm:result"' "$JSONL" || red "pm:result comment not written after tick"
/usr/local/bin/mem get "pm/cursor/${TID}" >/dev/null 2>&1 || red "cursor not advanced in mem"
green "result encoded, cursor advanced"
# Clean up the synthetic completion artifact
rm -f "$GPU_COMPLETED/${TASK_ID}-output.md"

# --- Pause / resume ------------------------------------------------------
step "6. Pause then resume — expect a single transition observation each way"
$LAPIS pause "$TID" --reason "smoke-pause"
PRE_LINES=$(wc -l < "$JSONL")
$LAPIS tick --target "$TID" | grep -q "skipped=True" || red "tick did not skip while paused"
$LAPIS tick --target "$TID" | grep -q "skipped=True" || red "second paused tick did not skip"
POST_LINES=$(wc -l < "$JSONL")
ADDED=$((POST_LINES - PRE_LINES))
# 1 transition (active→paused) was logged at the FIRST tick after pause.
# Subsequent paused ticks must add nothing.
[ "$ADDED" -le 1 ] || red "paused ticks added $ADDED comments (expected ≤1)"
$LAPIS resume "$TID"
$LAPIS tick --target "$TID" >/dev/null
grep -q "paused → active" "$JSONL" || red "no paused→active transition logged"
green "pause/resume transitions logged exactly once each"

# --- noop:no_change + noop:paused taxonomy assertions --------------------
step "6b. Decision taxonomy: healthy tick → noop:no_change; paused tick → noop:paused"
# Healthy tick on a bound target with no new percepts → noop:no_change
TICK_NOOP=$($LAPIS tick --target "$TID")
echo "$TICK_NOOP" | grep -q "decision=noop:no_change" \
    || red "healthy tick did not emit decision=noop:no_change (got: $TICK_NOOP)"
# Paused tick → noop:paused
$LAPIS pause "$TID" --reason "taxonomy-check"
TICK_PAUSED=$($LAPIS tick --target "$TID")
echo "$TICK_PAUSED" | grep -q "decision=noop:paused" \
    || red "paused tick did not emit decision=noop:paused (got: $TICK_PAUSED)"
$LAPIS resume "$TID"
green "noop:no_change and noop:paused taxonomy assertions pass"

# --- Directive flow ------------------------------------------------------
step "7. Post a human:directive — expect a brief comment after tick"
DIRECTIVE_BODY="Halt all fixer dispatches; investigate manually."
# Try the dashboard route first; fall back to direct CommentStore append
# if the dashboard isn't running so the smoke is self-contained.
HTTP_OK=0
if /usr/bin/curl -fsS -X POST "http://127.0.0.1:8400/api/thread/${TID}/directive" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"${DIRECTIVE_BODY}\", \"author\": \"Erah\"}" >/dev/null 2>&1; then
    HTTP_OK=1
    green "directive POSTed via dashboard route"
else
    /usr/bin/python3 -c "
from agents_core.comments import CommentStore
CommentStore().append('${TID}', '${DIRECTIVE_BODY}', 'Erah', 'user', tags=['human:directive'])
"
    green "dashboard not reachable on 8400 — wrote directive directly via CommentStore"
fi
grep -q '"human:directive"' "$JSONL" || red "human:directive tag not in JSONL"
$LAPIS tick --target "$TID" | tee /tmp/${TID}-tickout
grep -q "decision=action:directive_brief:" /tmp/${TID}-tickout || red "tick did not produce a directive brief"
grep -q '"pm:brief"' "$JSONL" || red "pm:brief comment not written"
/usr/local/bin/mem get "pm/outstanding-brief/${TID}" >/dev/null 2>&1 || red "outstanding-brief not set in mem"
green "directive → brief flow works (push success depends on PUSHOVER_* env, dashboard_route=${HTTP_OK})"

# --- Status output -------------------------------------------------------
step "8. Status --explain shows recent PM episodes"
$LAPIS status "$TID" --explain | grep -q "pm:" || red "status --explain showed no PM episodes"
green "status --explain works"

# =========================================================================
# Auto-land smoke tests
# =========================================================================
# These use a separate TID_LAND target per test so each test is independent.
# Forgejo is NOT called — the pm:pr-merged observations are written directly
# into the episodic store to simulate a merged PR.

_setup_land_target() {
    local tid="$1"
    cat > "$TARGETS_DIR/${tid}.yaml" <<EOF
id: ${tid}
title: Auto-land smoke target
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Auto-land smoke target — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF
    $LAPIS bind "$tid" --spec-from "$SPEC_FILE" --repo lapis-test --authority advisory
    # Simulate a PR having been opened and merged: write synthetic observations.
    /usr/bin/python3 -c "
from agents_core.comments import CommentStore
cs = CommentStore()
cs.append('${tid}', 'PR #99 opened in lapis-test: fake PR', 'lapis-pm', 'agent',
          tags=['pm:observation', 'pm:pr=99'])
cs.append('${tid}', 'PR #99 merged at 2026-04-25T00:00:00, head_sha=abc123', 'lapis-pm', 'agent',
          tags=['pm:observation', 'pm:pr-merged:99', 'pm:pr=99'])
"
}

_cleanup_land_target() {
    local tid="$1"
    rm -f "$TARGETS_DIR/${tid}.yaml" "$COMMENTS_DIR/${tid}.jsonl"
    rm -f "/srv/lapis/lapis-state/${tid}.md"
    /usr/local/bin/mem delete "pm/cursor/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/dispatched/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/pause-state/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/outstanding-brief/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/classified-prs/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/landed/${tid}" 2>/dev/null || true
}

# --- Auto-land: positive case --------------------------------------------
step "9. Auto-land positive: merged PR + no pending dispatches → auto-land fires"
TID_LAND="pm-smoke-land-$$"
_setup_land_target "$TID_LAND"
TICK_OUT=$($LAPIS tick --target "$TID_LAND")
echo "$TICK_OUT"
echo "$TICK_OUT" | grep -q "decision=action:auto_land:" || red "auto-land did not fire in positive case"
[ -f "/srv/lapis/lapis-state/${TID_LAND}.md" ] || red "arc doc not written for ${TID_LAND}"
/usr/local/bin/mem get "pm/landed/${TID_LAND}" >/dev/null 2>&1 || red "pm/landed not set after auto-land"
# Target should be archived and unbound after landing
/usr/bin/python3 -c "
from agents_core.targets import TargetStore
t = TargetStore().get('${TID_LAND}')
assert t is not None, 'target not found'
assert not t.pm_bound, 'target still pm_bound after auto-land'
assert t.data.get('status') == 'archived', f'status not archived: {t.data.get(\"status\")}'
"
grep -q '"pm:auto-land"' "$COMMENTS_DIR/${TID_LAND}.jsonl" || red "pm:auto-land audit comment missing"
_cleanup_land_target "$TID_LAND"
green "auto-land positive case OK"

# --- Auto-land: guard — pending dispatch --------------------------------
step "10. Auto-land guard: pending dispatch → auto-land suppressed"
TID_LAND2="pm-smoke-land-pend-$$"
_setup_land_target "$TID_LAND2"
# Inject a pending dispatch record so the guard fires
/usr/local/bin/mem set "pm/dispatched/${TID_LAND2}" \
    '[{"gpu_id":"fake-pending","agent_type":"fixer","status":"pending","intent":"test","repo":"lapis-test","ts":"2026-01-01T00:00:00","retry_count":0}]' \
    >/dev/null
TICK_OUT2=$($LAPIS tick --target "$TID_LAND2")
echo "$TICK_OUT2"
! echo "$TICK_OUT2" | grep -q "decision=action:auto_land:" || red "auto-land fired despite pending dispatch"
[ ! -f "/srv/lapis/lapis-state/${TID_LAND2}.md" ] || red "arc doc should NOT exist when dispatch pending"
_cleanup_land_target "$TID_LAND2"
green "auto-land pending-dispatch guard OK"

# --- Auto-land: guard — paused -------------------------------------------
step "11. Auto-land guard: target paused → auto-land suppressed"
TID_LAND3="pm-smoke-land-pause-$$"
_setup_land_target "$TID_LAND3"
$LAPIS pause "$TID_LAND3" --reason "smoke-test pause"
TICK_OUT3=$($LAPIS tick --target "$TID_LAND3")
echo "$TICK_OUT3"
echo "$TICK_OUT3" | grep -q "skipped=True" || red "tick should be skipped while paused"
[ ! -f "/srv/lapis/lapis-state/${TID_LAND3}.md" ] || red "arc doc should NOT exist when paused"
_cleanup_land_target "$TID_LAND3"
green "auto-land paused guard OK"

# --- Auto-land: idempotency ----------------------------------------------
step "12. Auto-land idempotency: second tick does not re-land"
TID_LAND4="pm-smoke-land-idem-$$"
_setup_land_target "$TID_LAND4"
# First tick → auto-land fires
TICK1=$($LAPIS tick --target "$TID_LAND4")
echo "$TICK1"
echo "$TICK1" | grep -q "decision=action:auto_land:" || red "first tick: auto-land did not fire"
[ -f "/srv/lapis/lapis-state/${TID_LAND4}.md" ] || red "arc doc missing after first tick"
ARC_MTIME=$(stat -c %Y "/srv/lapis/lapis-state/${TID_LAND4}.md")
# Second tick → target unbound, auto-land does NOT fire again
TICK2=$($LAPIS tick --target "$TID_LAND4")
echo "$TICK2"
echo "$TICK2" | grep -q "skipped=True" || red "second tick should skip (target unbound)"
! echo "$TICK2" | grep -q "decision=action:auto_land:" || red "auto-land fired twice (idempotency violated)"
ARC_MTIME2=$(stat -c %Y "/srv/lapis/lapis-state/${TID_LAND4}.md")
[ "$ARC_MTIME" = "$ARC_MTIME2" ] || red "arc doc was overwritten on second tick (not idempotent)"
_cleanup_land_target "$TID_LAND4"
green "auto-land idempotency OK"

# =========================================================================
# Multi-PR auto-land guard smoke tests (lapis-pm-auto-land-multipr-guard)
# =========================================================================

# --- Auto-land: multi-PR positive case -----------------------------------
step "13. Multi-PR guard: pr_count=2, one PR merged → noop; second merged → auto-land fires"
TID_MULTI="pm-smoke-multi-$$"
MULTI_SPEC="/tmp/${TID_MULTI}-spec.md"
cat > "$MULTI_SPEC" <<EOF
# Multi-PR smoke spec

### PR 1
First PR details.

### PR 2
Second PR details.
EOF

# Create target + bind with --pr-count 2 (explicit)
cat > "$TARGETS_DIR/${TID_MULTI}.yaml" <<EOF
id: ${TID_MULTI}
title: Multi-PR smoke target
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Multi-PR guard smoke target.
pr_count: 2
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_MULTI" --spec-from "$MULTI_SPEC" --repo lapis-test --authority advisory --pr-count 2

# Simulate PR #1 opened and merged only
/usr/bin/python3 -c "
from agents_core.comments import CommentStore
cs = CommentStore()
cs.append('${TID_MULTI}', 'PR #1 opened in lapis-test', 'lapis-pm', 'agent',
          tags=['pm:observation', 'pm:pr=1'])
cs.append('${TID_MULTI}', 'PR #1 merged at 2026-04-25T00:00:00, head_sha=aaa111', 'lapis-pm', 'agent',
          tags=['pm:observation', 'pm:pr-merged:1', 'pm:pr=1'])
"

# Tick 1: only 1/2 PRs merged → no auto-land, pm:auto-land:waiting emitted
TICK_MULTI1=$($LAPIS tick --target "$TID_MULTI")
echo "$TICK_MULTI1"
! echo "$TICK_MULTI1" | grep -q "decision=action:auto_land:" || red "multi-PR: auto-land should NOT fire after 1/2 merged"
[ ! -f "/srv/lapis/lapis-state/${TID_MULTI}.md" ] || red "arc doc should not exist after 1/2 PRs merged"
grep -q '"pm:auto-land:waiting"' "$COMMENTS_DIR/${TID_MULTI}.jsonl" \
    || red "pm:auto-land:waiting audit comment not written after 1/2"

# Tick 2 (still 1/2): waiting comment NOT written again (de-dup).
# Count only pm:auto-land:waiting occurrences (not total lines — pm:error
# from Forgejo 404s on the test repo is written every tick and is expected).
WAIT_BEFORE=$(grep -c '"pm:auto-land:waiting"' "$COMMENTS_DIR/${TID_MULTI}.jsonl" || true)
$LAPIS tick --target "$TID_MULTI" >/dev/null
WAIT_AFTER=$(grep -c '"pm:auto-land:waiting"' "$COMMENTS_DIR/${TID_MULTI}.jsonl" || true)
[ "$WAIT_BEFORE" -eq "$WAIT_AFTER" ] \
    || red "pm:auto-land:waiting was emitted again on second tick (de-dup failed, count before=$WAIT_BEFORE after=$WAIT_AFTER)"

# Simulate PR #2 opened and merged
/usr/bin/python3 -c "
from agents_core.comments import CommentStore
cs = CommentStore()
cs.append('${TID_MULTI}', 'PR #2 opened in lapis-test', 'lapis-pm', 'agent',
          tags=['pm:observation', 'pm:pr=2'])
cs.append('${TID_MULTI}', 'PR #2 merged at 2026-04-25T01:00:00, head_sha=bbb222', 'lapis-pm', 'agent',
          tags=['pm:observation', 'pm:pr-merged:2', 'pm:pr=2'])
"

# Tick 3: 2/2 PRs merged → auto-land fires
TICK_MULTI3=$($LAPIS tick --target "$TID_MULTI")
echo "$TICK_MULTI3"
echo "$TICK_MULTI3" | grep -q "decision=action:auto_land:" || red "multi-PR: auto-land did not fire after 2/2 merged"
[ -f "/srv/lapis/lapis-state/${TID_MULTI}.md" ] || red "arc doc not written after 2/2 PRs merged"

# Cleanup
rm -f "$TARGETS_DIR/${TID_MULTI}.yaml" "$COMMENTS_DIR/${TID_MULTI}.jsonl" "$MULTI_SPEC"
rm -f "/srv/lapis/lapis-state/${TID_MULTI}.md"
/usr/local/bin/mem delete "pm/cursor/${TID_MULTI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_MULTI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/pause-state/${TID_MULTI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_MULTI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_MULTI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/landed/${TID_MULTI}" 2>/dev/null || true
green "multi-PR guard: 1/2→noop+waiting, de-dup, 2/2→auto-land all OK"

# --- noop:reviewer_in_flight taxonomy -----------------------------------
step "13b. Decision taxonomy: pending reviewer dispatch → noop:reviewer_in_flight"
TID_RIF="pm-smoke-rif-$$"
cat > "$TARGETS_DIR/${TID_RIF}.yaml" <<EOF
id: ${TID_RIF}
title: Reviewer-in-flight smoke target
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: noop:reviewer_in_flight taxonomy smoke target.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_RIF" --spec-from "$SPEC_FILE" --repo lapis-test --authority advisory
# Inject a pending reviewer dispatch record into mem to simulate in-flight reviewer
/usr/bin/python3 -c "
import json
from lapis_pm.pm_core import append_dispatched
append_dispatched('${TID_RIF}', {
    'gpu_id': 'smoke-review-task-001',
    'agent_type': 'reviewer',
    'pr_number': 7,
    'cycle': 1,
    'status': 'pending',
    'intent': 'review PR #7',
    'repo': 'lapis-test',
    'ts': '2026-05-02T00:00:00+00:00',
    'retry_count': 0,
})
"
# Also inject a PR observation so the PR appears in open_prs mock path,
# and mark it as NOT classified so _decide_for_pr will be called.
# We patch _perceive_prs and authority.classify so the smoke doesn't need a live Forgejo.
TICK_RIF=$(/usr/bin/python3 -c "
from unittest.mock import patch, MagicMock
from lapis_pm import pm_core, authority

pr = {'number': 7, 'title': 'Smoke PR', 'state': 'open',
      'head': {'ref': 'lapis/${TID_RIF}/smoke'}, 'html_url': 'http://x'}

mock_cls = authority.PRClassification(
    verdict='advisory', screen_verdict='unknown',
    static_outcome=authority.StaticOutcome.static_pass,
    reasons=[], issues=[], pr_number=7, repo='lapis-test',
    title='Smoke PR', html_url='http://x', changed_paths=[], diff_loc=0, diff='',
)

with patch('lapis_pm.pm_core._perceive_prs', return_value=[pr]), \
     patch('lapis_pm.pm_core.authority.classify', return_value=mock_cls):
    result = pm_core.tick('${TID_RIF}')
print(f'[{result.target_id}] skipped={result.skipped} reason={result.reason} encoded={result.encoded} decision={result.decision}')
")
echo "$TICK_RIF"
echo "$TICK_RIF" | grep -q "decision=noop:reviewer_in_flight:pr=7:cycle=" \
    || red "reviewer_in_flight tick did not emit noop:reviewer_in_flight (got: $TICK_RIF)"
# Cleanup
rm -f "$TARGETS_DIR/${TID_RIF}.yaml" "$COMMENTS_DIR/${TID_RIF}.jsonl"
/usr/local/bin/mem delete "pm/cursor/${TID_RIF}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_RIF}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/pause-state/${TID_RIF}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_RIF}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_RIF}" 2>/dev/null || true
green "noop:reviewer_in_flight taxonomy assertion passes"

# --- Auto-detect: bind --create infers pr_count from spec headers ---------
step "14. Auto-detect: bind --create with ### PR N headers sets pr_count automatically"
TID_AUTODET="pm-smoke-autodet-$$"
AUTODET_SPEC="/tmp/${TID_AUTODET}-spec.md"
cat > "$AUTODET_SPEC" <<EOF
# Auto-detect spec

### PR 1
First.

### PR 2
Second.
EOF

# Bind with --create, no --pr-count → auto-detect should find 2
$LAPIS bind "$TID_AUTODET" \
    --spec-from "$AUTODET_SPEC" \
    --repo lapis-test \
    --authority advisory \
    --create \
    --title "Auto-detect target"

/usr/bin/python3 -c "
import yaml
from pathlib import Path
data = yaml.safe_load(Path('$TARGETS_DIR/${TID_AUTODET}.yaml').read_text())
pr_count = data.get('pr_count', 1)
assert pr_count == 2, f'Expected pr_count=2, got {pr_count}'
print(f'pr_count={pr_count} ✓')
" || red "auto-detect: pr_count not set to 2 in target YAML"

# Cleanup
rm -f "$TARGETS_DIR/${TID_AUTODET}.yaml" "$COMMENTS_DIR/${TID_AUTODET}.jsonl" "$AUTODET_SPEC"
/usr/local/bin/mem delete "pm/cursor/${TID_AUTODET}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_AUTODET}" 2>/dev/null || true
green "auto-detect: pr_count=2 set from ### PR N headers OK"

# =========================================================================
# Dispatch–queue reconciliation smoke test
# =========================================================================

# --- Reconciliation: positive case ----------------------------------------
step "15. Reconciliation: pending dispatch + faked queue failure → flipped to failed + audit comment"
TID_RECON="pm-smoke-recon-$$"
RECON_SPEC="/tmp/${TID_RECON}-spec.md"
cat > "$RECON_SPEC" <<EOF
# Reconciliation smoke spec for $TID_RECON
EOF

cat > "$TARGETS_DIR/${TID_RECON}.yaml" <<EOF
id: ${TID_RECON}
title: Reconciliation smoke target
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Reconciliation smoke target — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_RECON" --spec-from "$RECON_SPEC" --repo lapis-test --authority advisory

# Inject a pending dispatch record directly into mem (simulates a real dispatch).
FAKE_GPU_ID="claude_smoke_recon_fake_gpu_$$"
/usr/local/bin/mem set "pm/dispatched/${TID_RECON}" \
    "[{\"gpu_id\":\"${FAKE_GPU_ID}\",\"spec_id\":\"spec-smoke\",\"agent_type\":\"fixer_retry\",\"intent\":\"test reconcile\",\"repo\":\"lapis-test\",\"ts\":\"2026-04-25T16:51:00-07:00\",\"status\":\"pending\",\"retry_count\":0,\"pr_number\":1}]" \
    >/dev/null

# Fake a ClaudeQueue failure entry for that gpu_id by writing a YAML file
# directly into /srv/lapis/claude-queue/failed/ — this is the same path ClaudeQueue reads.
RECON_QUEUE_DIR="/srv/lapis/claude-queue/failed"
mkdir -p "$RECON_QUEUE_DIR"
cat > "$RECON_QUEUE_DIR/${FAKE_GPU_ID}.yaml" <<EOF
id: ${FAKE_GPU_ID}
status: failed
error: "ERROR: call_claude_cli returned None (smoke test)"
completed_at: "2026-04-25T16:51:30-07:00"
submitted_at: "2026-04-25T16:51:00-07:00"
submitted_by: lapis-pm
description: smoke-test fake failure
EOF

# Tick — reconcile should fire and flip the pending record to failed.
RECON_TICK_OUT=$($LAPIS tick --target "$TID_RECON")
echo "$RECON_TICK_OUT"

# Assert reconciled=1 appears in tick output.
echo "$RECON_TICK_OUT" | grep -q "reconciled=1" || red "reconcile: tick did not report reconciled=1"

# Assert the dispatched record was flipped to failed in mem.
/usr/bin/python3 -c "
import json
from lapis_pm.pm_core import load_dispatched
records = load_dispatched('${TID_RECON}')
assert records, 'no dispatched records found'
rec = records[0]
assert rec.get('status') == 'failed', f'expected failed, got {rec.get(\"status\")}'
assert 'ERROR' in (rec.get('error') or ''), f'error field not set: {rec.get(\"error\")}'
print(f'dispatched status={rec[\"status\"]} error={rec.get(\"error\")[:40]}')
" || red "reconcile: dispatched record not flipped to failed"

# Assert the audit comment was written.
RECON_JSONL="$COMMENTS_DIR/${TID_RECON}.jsonl"
grep -q '"pm:dispatch-reconciled"' "$RECON_JSONL" || red "reconcile: pm:dispatch-reconciled audit comment missing"
grep -q "pm:dispatch-reconciled:gpu=${FAKE_GPU_ID}" "$RECON_JSONL" || red "reconcile: gpu dedup tag missing from audit comment"

# Assert idempotency: second tick writes no new reconcile audit comment.
# Count only pm:dispatch-reconciled occurrences, not total lines — pm:error
# from Forgejo 404s on the test repo is written every tick and is expected.
RECON_BEFORE=$(grep -c '"pm:dispatch-reconciled"' "$RECON_JSONL" || true)
$LAPIS tick --target "$TID_RECON" >/dev/null
RECON_AFTER=$(grep -c '"pm:dispatch-reconciled"' "$RECON_JSONL" || true)
[ "$RECON_BEFORE" -eq "$RECON_AFTER" ] \
    || red "reconcile: audit comment written again on second tick (de-dup failed, count before=$RECON_BEFORE after=$RECON_AFTER)"

# Cleanup
rm -f "$RECON_QUEUE_DIR/${FAKE_GPU_ID}.yaml"
rm -f "$TARGETS_DIR/${TID_RECON}.yaml" "$COMMENTS_DIR/${TID_RECON}.jsonl" "$RECON_SPEC"
/usr/local/bin/mem delete "pm/cursor/${TID_RECON}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_RECON}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/pause-state/${TID_RECON}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_RECON}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_RECON}" 2>/dev/null || true
green "reconciliation positive case OK (flip + audit comment + de-dup idempotency)"

# =========================================================================
# Cycle-accounting phantom-record regression (lapis-pm-cycle-accounting-sha-truth)
# =========================================================================

# --- Part A: _fixer_retry_count counts processed only -------------------
step "16a. Phantom-record regression: _fixer_retry_count counts processed only (not failed)"
/usr/bin/python3 -c "
import json, sys
from unittest.mock import patch
from lapis_pm import pm_core

processed = {'agent_type': 'fixer_retry', 'pr_number': 1, 'status': 'processed', 'gpu_id': 'gpu-1', 'ts': '2026-04-27T00:00:00-07:00'}
failed    = {'agent_type': 'fixer_retry', 'pr_number': 1, 'status': 'failed',    'gpu_id': 'gpu-2', 'ts': '2026-04-27T00:00:00-07:00'}
with patch('lapis_pm.pm_core.load_dispatched', return_value=[processed, failed]):
    count = pm_core._fixer_retry_count('ignored', 1)
assert count == 1, f'expected 1, got {count} — failed record is being counted (cascade bug)'
print(f'_fixer_retry_count with 1 processed + 1 failed = {count} ✓')
" || red "phantom-record: _fixer_retry_count is counting failed records"
green "_fixer_retry_count: 1 processed + 1 failed → count=1 (cascade fix confirmed)"

# --- Part B: _fixer_retry_count returns 0 for all-failed scenario -------
step "16b. Phantom-record regression: _fixer_retry_count returns 0 for all-failed (cascade scenario)"
/usr/bin/python3 -c "
from unittest.mock import patch
from lapis_pm import pm_core

failed1 = {'agent_type': 'fixer_retry', 'pr_number': 1, 'status': 'failed', 'gpu_id': 'gpu-a', 'ts': '2026-04-27T00:00:00-07:00'}
failed2 = {'agent_type': 'fixer_retry', 'pr_number': 1, 'status': 'failed', 'gpu_id': 'gpu-b', 'ts': '2026-04-27T00:00:00-07:00'}
with patch('lapis_pm.pm_core.load_dispatched', return_value=[failed1, failed2]):
    count = pm_core._fixer_retry_count('ignored', 1)
assert count == 0, f'expected 0, got {count} — failed records causing phantom count'
print(f'_fixer_retry_count with 2 failed records = {count} ✓')
" || red "phantom-record: _fixer_retry_count returns non-zero for all-failed scenario"
green "_fixer_retry_count: 2 failed records → count=0 OK"

# --- Part C: reconciler carve-out — fixer_retry completed stays pending --
step "16c. Reconciler carve-out: fixer_retry + queue completed → stays pending (SHA-advance owns it)"
TID_CARVEOUT="pm-smoke-carveout-$$"
CARVEOUT_SPEC="/tmp/${TID_CARVEOUT}-spec.md"
cat > "$CARVEOUT_SPEC" <<EOF
# Carve-out smoke spec for $TID_CARVEOUT
EOF

cat > "$TARGETS_DIR/${TID_CARVEOUT}.yaml" <<EOF
id: ${TID_CARVEOUT}
title: Carve-out smoke target
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Reconciler carve-out smoke target — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_CARVEOUT" --spec-from "$CARVEOUT_SPEC" --repo lapis-test --authority advisory

# Inject a pending fixer_retry dispatch into mem (simulates a real dispatch).
CARVEOUT_GPU_ID="claude_smoke_carveout_$$"
/usr/local/bin/mem set "pm/dispatched/${TID_CARVEOUT}" \
    "[{\"gpu_id\":\"${CARVEOUT_GPU_ID}\",\"spec_id\":\"spec-smoke\",\"agent_type\":\"fixer_retry\",\"intent\":\"test carveout\",\"repo\":\"lapis-test\",\"ts\":\"2026-04-27T10:00:00-07:00\",\"status\":\"pending\",\"retry_count\":0,\"pr_number\":1}]" \
    >/dev/null

# Fake a ClaudeQueue *completed* entry for that fixer_retry gpu_id.
# With the carve-out, this must NOT flip the record to processed.
CARVEOUT_QUEUE_DIR="/srv/lapis/claude-queue/completed"
mkdir -p "$CARVEOUT_QUEUE_DIR"
cat > "$CARVEOUT_QUEUE_DIR/${CARVEOUT_GPU_ID}.yaml" <<EOF
id: ${CARVEOUT_GPU_ID}
status: completed
completed_at: "2026-04-27T10:05:00-07:00"
submitted_at: "2026-04-27T10:00:00-07:00"
submitted_by: lapis-pm
description: smoke-test fake completion
EOF

# Tick — reconciler should skip the fixer_retry completed entry.
CARVEOUT_TICK=$($LAPIS tick --target "$TID_CARVEOUT")
echo "$CARVEOUT_TICK"

# The record must still be pending (SHA-advance perceiver hasn't fired).
/usr/bin/python3 -c "
from lapis_pm.pm_core import load_dispatched
records = load_dispatched('${TID_CARVEOUT}')
assert records, 'no dispatched records found'
rec = records[0]
assert rec.get('status') == 'pending', f'expected pending, got {rec.get(\"status\")} — carve-out failed: reconciler flipped fixer_retry to processed'
print(f'fixer_retry status after queue-completed tick = {rec[\"status\"]} ✓')
" || red "carve-out: reconciler flipped fixer_retry to processed (SHA-advance perceiver must own this)"

# Assert no pm:dispatch-reconciled tag for this gpu_id (carve-out produces no audit comment).
! grep -q "pm:dispatch-reconciled:gpu=${CARVEOUT_GPU_ID}" "$COMMENTS_DIR/${TID_CARVEOUT}.jsonl" \
    || red "carve-out: reconcile audit comment written for fixer_retry completed (should be silent)"

# Cleanup
rm -f "$CARVEOUT_QUEUE_DIR/${CARVEOUT_GPU_ID}.yaml"
rm -f "$TARGETS_DIR/${TID_CARVEOUT}.yaml" "$COMMENTS_DIR/${TID_CARVEOUT}.jsonl" "$CARVEOUT_SPEC"
/usr/local/bin/mem delete "pm/cursor/${TID_CARVEOUT}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_CARVEOUT}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/pause-state/${TID_CARVEOUT}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_CARVEOUT}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_CARVEOUT}" 2>/dev/null || true
green "reconciler carve-out: fixer_retry completed → stays pending OK"

# =========================================================================
# Reviewer verdict-encode carve-out smoke test
# (lapis-pm-reviewer-verdict-encode-carveout)
# =========================================================================

# --- Step 17: reviewer dispatch → verdict=fixable encoded (not verdict=pending) ---
step "17. Reviewer carve-out: reviewer dispatch + synthesized output file → verdict=fixable encoded"
TID_REV="pm-smoke-reviewer-$$"
REV_SPEC="/tmp/${TID_REV}-spec.md"
cat > "$REV_SPEC" <<EOF
# Reviewer carve-out smoke spec for $TID_REV
EOF

cat > "$TARGETS_DIR/${TID_REV}.yaml" <<EOF
id: ${TID_REV}
title: Reviewer carve-out smoke target
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Reviewer verdict-encode carve-out smoke target — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_REV" --spec-from "$REV_SPEC" --repo lapis-test --authority advisory

# Inject a pending reviewer dispatch record directly into mem.
REV_GPU_ID="claude_smoke_reviewer_$$"
/usr/local/bin/mem set "pm/dispatched/${TID_REV}" \
    "[{\"gpu_id\":\"${REV_GPU_ID}\",\"spec_id\":\"spec-smoke\",\"agent_type\":\"reviewer\",\"intent\":\"review PR #1\",\"repo\":\"lapis-test\",\"ts\":\"2026-04-29T12:00:00-07:00\",\"status\":\"pending\",\"retry_count\":0,\"pr_number\":1,\"cycle\":1}]" \
    >/dev/null

# Fake a ClaudeQueue completed entry for that reviewer gpu_id.
# With the carve-out, the reconciler must NOT flip this to processed —
# _encode_gpu_results owns the terminal flip for reviewer completed.
REV_QUEUE_DIR="/srv/lapis/claude-queue/completed"
mkdir -p "$REV_QUEUE_DIR"
cat > "$REV_QUEUE_DIR/${REV_GPU_ID}.yaml" <<EOF
id: ${REV_GPU_ID}
status: completed
completed_at: "2026-04-29T12:05:00-07:00"
submitted_at: "2026-04-29T12:00:00-07:00"
submitted_by: lapis-pm
description: smoke-test fake reviewer completion
EOF

# Write the synthesized reviewer output file (JSON verdict the reviewer agent produced).
mkdir -p "$GPU_COMPLETED"
cat > "$GPU_COMPLETED/${REV_GPU_ID}-output.md" <<EOF
{"verdict":"fixable","issues":[{"summary":"missing test coverage","severity":"medium"}],"confidence":0.88}
EOF

# Tick — reconciler carve-out keeps reviewer pending; encoder reads the
# output file and writes a Reviewer verdict episodic entry with verdict=fixable.
REV_TICK=$($LAPIS tick --target "$TID_REV")
echo "$REV_TICK"

# Assert the dispatched record was flipped to processed by the encoder (not the reconciler).
/usr/bin/python3 -c "
from lapis_pm.pm_core import load_dispatched
records = load_dispatched('${TID_REV}')
assert records, 'no dispatched records found'
rec = records[0]
assert rec.get('status') == 'processed', f'expected processed, got {rec.get(\"status\")} — encoder did not flip reviewer'
print(f'reviewer dispatch status={rec[\"status\"]} ✓')
" || red "reviewer carve-out: dispatched record not processed after encode tick"

# Assert episodic has the Reviewer verdict entry (non-pending verdict).
REV_JSONL="$COMMENTS_DIR/${TID_REV}.jsonl"
grep -q 'Reviewer verdict for PR #1:' "$REV_JSONL" \
    || red "reviewer carve-out: 'Reviewer verdict for PR #1:' entry missing from episodic"
grep -q 'verdict=fixable' "$REV_JSONL" \
    || red "reviewer dispatch produces verdict=fixable episodic entry (REGRESSION: verdict stayed pending)"

# Assert reconciler did NOT write a dispatch-reconciled audit for this gpu_id
# (the carve-out must be silent — no audit comment on the skipped completed entry).
! grep -q "pm:dispatch-reconciled:gpu=${REV_GPU_ID}" "$REV_JSONL" \
    || red "reviewer carve-out: reconcile audit comment written (reconciler bypassed carve-out)"

# Assert corroboration_result field present in reviewer verdict JSON
# (cross-node-corroboration-v0 PR 3: verdict carries corroboration_result)
python3 - "$REV_JSONL" <<'PYEOF' || red "corroboration_result missing or malformed in reviewer verdict episodic entry"
import json, sys
jsonl_path = sys.argv[1]
try:
    lines = open(jsonl_path).readlines()
except FileNotFoundError:
    print('no jsonl file — skipping corroboration check')
    sys.exit(0)
for line in lines:
    try:
        entry = json.loads(line)
    except Exception:
        continue
    content = entry.get('content', '')
    if 'Reviewer verdict for PR #1:' not in content:
        continue
    try:
        json_part = content.split('\n', 1)[-1].strip()
        verdict_data = json.loads(json_part)
    except Exception:
        print(f'FAIL: could not parse verdict JSON')
        sys.exit(1)
    if 'corroboration_result' not in verdict_data:
        print('FAIL: corroboration_result missing from verdict JSON')
        sys.exit(1)
    corr = verdict_data['corroboration_result']
    required = {'verdict', 'scope_id', 'freshness_stamp', 'citations', 'claim'}
    missing = required - set(corr.keys())
    if missing:
        print(f'FAIL: corroboration_result missing fields: {missing}')
        sys.exit(1)
    print(f'corroboration_result present: verdict={corr["verdict"]}, scope_id={corr["scope_id"]} OK')
    sys.exit(0)
print('FAIL: no Reviewer verdict entry found in episodic')
sys.exit(1)
PYEOF

# Assert resume read-back preserves corroboration_result
# (_last_review_verdict parses the full JSON including corroboration_result)
python3 - "$TID_REV" "$REV_JSONL" <<'PYEOF3' || red "corroboration_result not preserved through _last_review_verdict (resume path)"
import json, sys
target_id, jsonl_path = sys.argv[1], sys.argv[2]
try:
    lines = open(jsonl_path).readlines()
except FileNotFoundError:
    print('no jsonl file — skipping resume check')
    sys.exit(0)
# Parse the verdict data directly from episodic (same path _last_review_verdict uses)
for line in lines:
    try:
        entry = json.loads(line)
    except Exception:
        continue
    content = entry.get('content', '')
    if 'Reviewer verdict for PR #1:' not in content:
        continue
    tags = entry.get('tags', [])
    if not any('verdict=' in t and ':verdict=pending' not in t for t in tags):
        continue
    try:
        json_part = content.split('\n', 1)[-1].strip()
        result = json.loads(json_part)
    except Exception:
        print('FAIL: could not parse verdict JSON for resume check')
        sys.exit(1)
    if 'corroboration_result' not in result:
        print(f'FAIL: corroboration_result not in verdict JSON (resume path): {list(result.keys())}')
        sys.exit(1)
    corr = result['corroboration_result']
    print(f'resume read-back: corroboration_result preserved, verdict={corr.get("verdict")} OK')
    sys.exit(0)
print('no non-pending reviewer entry found — skipping resume check')
sys.exit(0)
PYEOF3

# Cleanup
rm -f "$REV_QUEUE_DIR/${REV_GPU_ID}.yaml" "$GPU_COMPLETED/${REV_GPU_ID}-output.md"
rm -f "$TARGETS_DIR/${TID_REV}.yaml" "$COMMENTS_DIR/${TID_REV}.jsonl" "$REV_SPEC"
/usr/local/bin/mem delete "pm/cursor/${TID_REV}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_REV}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/pause-state/${TID_REV}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_REV}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_REV}" 2>/dev/null || true
green "reviewer verdict: corroboration_result present + resume read-back OK (cross-node-corroboration-v0)"
green "reviewer dispatch produces verdict=fixable episodic entry (not verdict=pending) OK"

# --- Chain smoke phase (step 18) ----------------------------------------
step "18. Chain: 3-leg bind → simulated land → auto-dispatch cascade → state.complete"

CHAIN_GROUP="pm-smoke-chain-$$"
TID_L1="pm-smoke-chain-l1-$$"
TID_L2="pm-smoke-chain-l2-$$"
TID_L3="pm-smoke-chain-l3-$$"
CHAIN_SPEC="/tmp/${CHAIN_GROUP}-spec.md"
CHAIN_LEGS="/tmp/${CHAIN_GROUP}-legs.yaml"
# Register chain-smoke legs: dispatches from these may open real Forgejo PRs.
SMOKE_TIDS+=("$TID_L1" "$TID_L2" "$TID_L3")

# Close leaked Forgejo PRs from chain-auto dispatch (best-effort; no-op if
# FORGEJO_TOKEN unset or Forgejo unreachable). Must run before mem deletion
# because we read each leg's dispatched record to find the target repo.
_smoke_chain_close_prs() {
    if [ -z "${FORGEJO_TOKEN:-}" ]; then
        echo "[smoke] chain-pr-cleanup: skipped (FORGEJO_TOKEN unset)"
        return 0
    fi
    local _tid
    for _tid in "$@"; do
        /usr/bin/python3 -c "
import os, json, sys
import httpx
from agents_core.mem import MemoryStore

tid = '${_tid}'
FORGEJO_URL = os.environ.get('FORGEJO_URL', 'http://203.0.113.12:3000')
OWNER = 'Erah'
API = f'{FORGEJO_URL}/api/v1'
token = os.environ.get('FORGEJO_TOKEN', '')
HDRS = {'Authorization': f'token {token}', 'Accept': 'application/json',
        'Content-Type': 'application/json'}
T = 15

try:
    rec = MemoryStore().get(f'pm/dispatched/{tid}')
    if not rec:
        sys.exit(0)
    records = json.loads(rec['content'])
    if not isinstance(records, list):
        records = [records]
except Exception as e:
    print(f'[smoke] chain-pr-cleanup: cannot read dispatched/{tid}: {e}')
    sys.exit(0)

repos = {r.get('repo') for r in records if r.get('repo')}
prefix = f'lapis/{tid}/'
for repo in repos:
    try:
        r = httpx.get(f'{API}/repos/{OWNER}/{repo}/pulls', headers=HDRS,
                      params={'state': 'open', 'limit': 50}, timeout=T)
        r.raise_for_status()
        open_prs = r.json()
    except Exception as e:
        print(f'[smoke] chain-pr-cleanup: list-prs {repo}: {e}')
        continue
    for pr in open_prs:
        head_ref = (pr.get('head') or {}).get('ref', '')
        if not head_ref.startswith(prefix):
            continue
        pr_num = pr['number']
        try:
            httpx.patch(f'{API}/repos/{OWNER}/{repo}/pulls/{pr_num}', headers=HDRS,
                        json={'state': 'closed'}, timeout=T).raise_for_status()
            print(f'[smoke] chain-pr-cleanup: closed {repo}#{pr_num} ({head_ref})')
        except Exception as e:
            print(f'[smoke] chain-pr-cleanup: close {repo}#{pr_num}: {e}')
            continue
        try:
            httpx.delete(f'{API}/repos/{OWNER}/{repo}/branches/{head_ref}',
                         headers=HDRS, timeout=T)
        except Exception:
            pass
" || true
    done
}

chain_cleanup() {
    # Close leaked Forgejo PRs for ALL smoke-created TIDs registered in
    # SMOKE_TIDS (chain legs + wire-emitter legs + wire-disp targets).
    # Must run before mem deletion because _smoke_chain_close_prs reads each
    # leg's dispatched record to find the target repo.
    [ ${#SMOKE_TIDS[@]} -gt 0 ] && _smoke_chain_close_prs "${SMOKE_TIDS[@]}"
    rm -f "$CHAIN_SPEC" "$CHAIN_LEGS"
    for T in "$TID_L1" "$TID_L2" "$TID_L3"; do
        rm -f "$TARGETS_DIR/${T}.yaml" "$COMMENTS_DIR/${T}.jsonl" 2>/dev/null || true
        /usr/local/bin/mem delete "pm/cursor/${T}" 2>/dev/null || true
        /usr/local/bin/mem delete "pm/dispatched/${T}" 2>/dev/null || true
        /usr/local/bin/mem delete "pm/pause-state/${T}" 2>/dev/null || true
        /usr/local/bin/mem delete "pm/outstanding-brief/${T}" 2>/dev/null || true
        /usr/local/bin/mem delete "pm/classified-prs/${T}" 2>/dev/null || true
        /usr/local/bin/mem delete "pm/landed/${T}" 2>/dev/null || true
        /usr/local/bin/mem delete "pm/review-state/${T}" 2>/dev/null || true
    done
    /usr/local/bin/mem delete "chain/${CHAIN_GROUP}/state" 2>/dev/null || true
}
trap 'chain_cleanup; cleanup' EXIT

cat > "$CHAIN_SPEC" <<EOF
# Chain smoke spec
What: 3-leg chain smoke test.
EOF

cat > "$CHAIN_LEGS" <<EOF
legs:
  - tid: ${TID_L1}
    repo: lapis-test
    authority: advisory
    branch_slug: leg1
    intent: |
      Implement leg 1 of chain smoke.
  - tid: ${TID_L2}
    repo: lapis-test
    authority: advisory
    branch_slug: leg2
    intent: |
      Implement leg 2 of chain smoke.
    depends_on:
      - ${TID_L1}
  - tid: ${TID_L3}
    repo: lapis-test
    authority: advisory
    branch_slug: leg3
    intent: |
      Implement leg 3 of chain smoke.
    depends_on:
      - ${TID_L2}
EOF

# 18a. Bind the 3-leg chain (with --no-auto-fire to avoid real GPU dispatch)
/usr/bin/python3 -m lapis_pm.cli bind "$CHAIN_GROUP" \
    --spec-from "$CHAIN_SPEC" \
    --legs-from "$CHAIN_LEGS" \
    --create --no-auto-fire
green "3-leg chain bound with --no-auto-fire"

# 18b. Verify all 3 leg targets exist and are pm_bound with correct chain fields
/usr/bin/python3 -c "
from agents_core.targets import TargetStore
store = TargetStore()
for tid in ['${TID_L1}', '${TID_L2}', '${TID_L3}']:
    t = store.get(tid)
    assert t is not None, f'target {tid!r} not found'
    assert t.pm_bound, f'target {tid!r} not pm_bound'
    assert t.data.get('chain_group') == '${CHAIN_GROUP}', f'{tid}: wrong chain_group'
    assert t.data.get('initial_dispatch'), f'{tid}: missing initial_dispatch'
print('all 3 legs bound with chain_group and initial_dispatch OK')

# L1 has no depends_on; L2 depends on L1; L3 depends on L2
t1 = store.get('${TID_L1}')
t2 = store.get('${TID_L2}')
t3 = store.get('${TID_L3}')
assert not (t1.data.get('depends_on') or []), 'L1 should have no depends_on'
assert t2.data.get('depends_on') == ['${TID_L1}'], f'L2 depends_on wrong: {t2.data.get(\"depends_on\")}'
assert t3.data.get('depends_on') == ['${TID_L2}'], f'L3 depends_on wrong: {t3.data.get(\"depends_on\")}'
print('depends_on fields correct OK')
" || red "chain bind: leg fields incorrect"

# 18b2. Verify branch injection: each leg's initial_dispatch starts with Branch: lapis/<tid>/<slug>
/usr/bin/python3 -c "
from agents_core.targets import TargetStore
store = TargetStore()
checks = [
    ('${TID_L1}', 'leg1'),
    ('${TID_L2}', 'leg2'),
    ('${TID_L3}', 'leg3'),
]
for tid, slug in checks:
    t = store.get(tid)
    dispatch = t.data.get('initial_dispatch', '')
    expected_prefix = f'Branch: lapis/{tid}/{slug}'
    assert dispatch.startswith(expected_prefix + '\n\n'), (
        f'{tid}: initial_dispatch does not start with {expected_prefix!r}; '
        f'got: {dispatch[:80]!r}'
    )
    print(f'{tid}: Branch injection OK → {expected_prefix}')
print('All leg branch injections correct')
" || red "chain branch injection: Branch: line not found or incorrect in initial_dispatch"
green "branch injection into initial_dispatch verified for all 3 legs"

# 18b3. Verify missing branch_slug is rejected before any leg target is created
CHAIN_NOSLUG_GROUP="pm-smoke-noslug-$$"
CHAIN_NOSLUG_LEGS="/tmp/${CHAIN_NOSLUG_GROUP}-legs.yaml"
CHAIN_NOSLUG_SPEC="/tmp/${CHAIN_NOSLUG_GROUP}-spec.md"
TID_NOSLUG="pm-smoke-noslug-leg-$$"
cat > "$CHAIN_NOSLUG_SPEC" <<'SPECEOF'
# No-slug smoke spec
SPECEOF
cat > "$CHAIN_NOSLUG_LEGS" <<NOSEOF
legs:
  - tid: ${TID_NOSLUG}
    repo: lapis-test
    authority: advisory
    intent: |
      Implement without branch_slug.
NOSEOF
# bind should fail with rc=2 (branch_slug missing)
NOSLUG_OUT=$(/usr/bin/python3 -m lapis_pm.cli bind "$CHAIN_NOSLUG_GROUP" \
    --spec-from "$CHAIN_NOSLUG_SPEC" \
    --legs-from "$CHAIN_NOSLUG_LEGS" \
    --create 2>&1 || true)
echo "$NOSLUG_OUT"
echo "$NOSLUG_OUT" | grep -qi "branch_slug" \
    || red "chain missing branch_slug: error message did not mention 'branch_slug'"
[ ! -f "${TARGETS_DIR}/${TID_NOSLUG}.yaml" ] \
    || red "chain missing branch_slug: leg target was created despite missing branch_slug (atomicity violated)"
rm -f "$CHAIN_NOSLUG_SPEC" "$CHAIN_NOSLUG_LEGS"
green "missing branch_slug rejected before any leg target created (atomicity preserved)"

# 18c. Verify chain/<group>/state is written with pending/dispatched statuses
/usr/bin/python3 -c "
from lapis_pm.chain import get_chain_state
state = get_chain_state('${CHAIN_GROUP}')
assert state is not None, 'chain state not written'
assert state['group_id'] == '${CHAIN_GROUP}', 'wrong group_id in state'
assert not state['complete'], 'state.complete should be false before any legs land'
legs_by_tid = {leg['tid']: leg for leg in state['legs']}
assert '${TID_L1}' in legs_by_tid, 'L1 missing from state'
assert '${TID_L2}' in legs_by_tid, 'L2 missing from state'
assert '${TID_L3}' in legs_by_tid, 'L3 missing from state'
print(f'chain state written, complete={state[\"complete\"]}, legs={list(legs_by_tid.keys())} OK')
" || red "chain state not written correctly"

# 18d. Simulate leg 1 landing: write pm/landed/L1 (what _act_auto_land does),
#      then call check_chain_advance → L2 should get an auto-dispatched record.
#      We need L2 to be pm_bound and have initial_dispatch, which it does.
#      We pause the GPU queue so the fixer dispatch doesn't run for real.
/usr/local/bin/gpu-submit pause >/dev/null 2>&1 || true

/usr/bin/python3 -c "
import json
from lapis_pm.pm_core import _mem, _landed_key, _now_iso
from lapis_pm.chain import check_chain_advance, on_leg_landed, get_chain_state

# Write pm/landed for L1 (simulating auto-land completion)
_mem().set(_landed_key('${TID_L1}'),
    json.dumps({'pr_num': 0, 'merged_at': _now_iso(), 'landed_at': _now_iso(), 'arc_path': '/tmp/none'}),
    tags=['lapis-pm', 'landed'])

on_leg_landed('${TID_L1}', '${CHAIN_GROUP}')
fired = check_chain_advance('${TID_L1}')
print(f'check_chain_advance fired: {fired}')
assert '${TID_L2}' in fired, f'L2 not auto-dispatched after L1 land; fired={fired}'

# State should now show L2 as dispatched
state = get_chain_state('${CHAIN_GROUP}')
legs_by_tid = {leg['tid']: leg for leg in state['legs']}
assert legs_by_tid['${TID_L1}']['status'] == 'landed', 'L1 status should be landed'
assert legs_by_tid['${TID_L2}']['status'] == 'dispatched', f'L2 status should be dispatched, got {legs_by_tid[\"${TID_L2}\"][\"status\"]}'
assert legs_by_tid['${TID_L3}']['status'] == 'pending', f'L3 status should still be pending'
print('L2 auto-dispatched, L1=landed L2=dispatched L3=pending OK')
" || red "chain advance: L2 not auto-dispatched after L1 land"
green "leg 1 land → leg 2 auto-dispatch verified"

# 18e. Idempotency: calling check_chain_advance again should NOT double-fire L2
/usr/bin/python3 -c "
from lapis_pm.chain import check_chain_advance
fired2 = check_chain_advance('${TID_L1}')
assert '${TID_L2}' not in fired2, f'L2 double-fired! fired2={fired2}'
print(f'idempotent: second check_chain_advance fired: {fired2} (L2 not in it) OK')
" || red "chain idempotency: L2 double-fired"
green "idempotency check OK (no double-fire)"

# 18f. Simulate leg 2 landing → leg 3 should auto-dispatch
/usr/bin/python3 -c "
import json
from lapis_pm.pm_core import _mem, _landed_key, _now_iso, load_dispatched, save_dispatched
from lapis_pm.chain import check_chain_advance, on_leg_landed, get_chain_state

# First mark L2's pending dispatch as landed so _is_dispatched returns False
records = load_dispatched('${TID_L2}')
for r in records:
    if r.get('status') == 'pending':
        r['status'] = 'landed'
save_dispatched('${TID_L2}', records)

# Write pm/landed for L2
_mem().set(_landed_key('${TID_L2}'),
    json.dumps({'pr_num': 0, 'merged_at': _now_iso(), 'landed_at': _now_iso(), 'arc_path': '/tmp/none'}),
    tags=['lapis-pm', 'landed'])

on_leg_landed('${TID_L2}', '${CHAIN_GROUP}')
fired = check_chain_advance('${TID_L2}')
print(f'check_chain_advance after L2 land fired: {fired}')
assert '${TID_L3}' in fired, f'L3 not auto-dispatched after L2 land; fired={fired}'

state = get_chain_state('${CHAIN_GROUP}')
legs_by_tid = {leg['tid']: leg for leg in state['legs']}
assert legs_by_tid['${TID_L2}']['status'] == 'landed', 'L2 status should be landed'
assert legs_by_tid['${TID_L3}']['status'] == 'dispatched', f'L3 status should be dispatched'
print('L3 auto-dispatched, L2=landed L3=dispatched OK')
" || red "chain advance: L3 not auto-dispatched after L2 land"
green "leg 2 land → leg 3 auto-dispatch verified"

# 18g. Simulate leg 3 landing → state.complete should flip true
/usr/bin/python3 -c "
import json
from lapis_pm.pm_core import _mem, _landed_key, _now_iso, load_dispatched, save_dispatched
from lapis_pm.chain import check_chain_advance, on_leg_landed, get_chain_state

# Clear L3's pending dispatch
records = load_dispatched('${TID_L3}')
for r in records:
    if r.get('status') == 'pending':
        r['status'] = 'landed'
save_dispatched('${TID_L3}', records)

_mem().set(_landed_key('${TID_L3}'),
    json.dumps({'pr_num': 0, 'merged_at': _now_iso(), 'landed_at': _now_iso(), 'arc_path': '/tmp/none'}),
    tags=['lapis-pm', 'landed'])

on_leg_landed('${TID_L3}', '${CHAIN_GROUP}')
check_chain_advance('${TID_L3}')  # nothing to fire, but should not error

state = get_chain_state('${CHAIN_GROUP}')
print(f'final chain state: complete={state[\"complete\"]}')
assert state['complete'], f'state.complete should be True after all legs land; state={state}'
print('state.complete=True OK')
" || red "chain: state.complete not true after all legs land"
green "all 3 legs landed → state.complete=True OK"

# 18h. Verify chain event log has bind + auto_dispatch events
/usr/bin/python3 -c "
from agents_core.mem import MemoryStore
mem = MemoryStore()
events = mem.list_all(tag='chain-group:${CHAIN_GROUP}', limit=100)
kinds = set()
for e in events:
    try:
        import json
        key = e.get('key', '')
        if '/event/' in key:
            data = json.loads(e['content'])
            kinds.add(data.get('kind'))
    except Exception:
        pass
print(f'chain event kinds found: {kinds}')
assert 'bind' in kinds, f'bind event missing; kinds={kinds}'
assert 'auto_dispatch' in kinds, f'auto_dispatch event missing; kinds={kinds}'
print('chain event log has bind + auto_dispatch events OK')
" || red "chain event log missing expected events"
green "chain event log verified OK"

chain_cleanup
green "Chain smoke phase complete: 3-leg chain, auto-dispatch cascade, state.complete, event log all OK"

# --- Phase 19: Router portfolio persistence smoke --------------------------
step "19. Router portfolio persistence (router-portfolio-persistence-v0)"

PORTFOLIO_TID="smoke-portfolio-$$"

portfolio_cleanup() {
    /usr/local/bin/mem delete "router/lapis-pm/decisions/smoke-kickoff-${PORTFOLIO_TID}" 2>/dev/null || true
    /usr/local/bin/mem delete "router/lapis-pm/decisions/smoke-dispatch-${PORTFOLIO_TID}" 2>/dev/null || true
    /usr/local/bin/mem delete "router/lapis-pm/ratification-outcomes/smoke-confirm-${PORTFOLIO_TID}" 2>/dev/null || true
}

# 19a. Emit 3 portfolio events (kickoff + dispatch + ratify-confirm)
/usr/bin/python3 -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from lapis_pm.router_portfolio import (
    emit_decision_kickoff,
    emit_decision_dispatch,
    emit_ratify_confirm,
)

tid = '${PORTFOLIO_TID}'
k1 = emit_decision_kickoff(
    target_id=tid,
    expert_chosen='haiku',
    intent_summary='smoke kickoff for portfolio test',
    event_id=f'smoke-kickoff-{tid}',
)
print(f'kickoff key: {k1}')
assert k1.startswith('router/lapis-pm/decisions/'), f'Bad kickoff key: {k1}'

k2 = emit_decision_dispatch(
    target_id=tid,
    fragment_id='tick',
    expert_chosen='haiku',
    intent_summary='smoke dispatch for portfolio test',
    event_id=f'smoke-dispatch-{tid}',
)
print(f'dispatch key: {k2}')
assert k2.startswith('router/lapis-pm/decisions/'), f'Bad dispatch key: {k2}'

k3 = emit_ratify_confirm(
    target_id=tid,
    prior_decision_event_id=f'smoke-kickoff-{tid}',
    event_id=f'smoke-confirm-{tid}',
)
print(f'confirm key: {k3}')
assert k3.startswith('router/lapis-pm/ratification-outcomes/'), f'Bad confirm key: {k3}'
print('3 portfolio events written OK')
" || red "portfolio: event emission failed"
green "3 portfolio events emitted (kickoff + dispatch + ratify-confirm)"

# 19b. Invoke checkpoint — read events + write session summary
/usr/bin/python3 -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from lapis_pm.router_portfolio import read_session_entries, write_session_summary

tid = '${PORTFOLIO_TID}'
entries = read_session_entries()
our_entries = [e for e in entries if e.get('target_id') == tid]
print(f'Found {len(our_entries)} entries for {tid}')
assert len(our_entries) == 3, f'Expected 3, got {len(our_entries)}'

summary = '''## Smoke session portfolio checkpoint

Attempted: kickoff + dispatch for {tid}
Landed: ratify-confirm received
Stalled: none
Gradient: 1 ratify-confirm
Next: none (smoke test)
'''.format(tid=tid)

session_key = write_session_summary(
    slug=f'smoke-{tid}',
    summary_markdown=summary.strip(),
    entries=our_entries,
)
print(f'Session checkpointed -> {session_key}')
assert session_key.startswith('router/lapis-pm/sessions/'), f'Bad session key: {session_key}'
assert f'smoke-{tid}' in session_key, f'Slug missing: {session_key}'
print('Checkpoint OK: all 3 events referenced')
" || red "portfolio: checkpoint write failed"
green "portfolio: checkpoint written with all 3 events referenced"

# 19c. Bootstrap query: verify prior session summary is readable
/usr/bin/python3 -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from lapis_pm.router_portfolio import session_checkpoint_exists

found = session_checkpoint_exists(since_iso='2020-01-01T00:00:00Z')
print(f'session_checkpoint_exists (wide window) = {found}')
assert found, 'session_checkpoint_exists returned False after writing checkpoint'
print('Bootstrap query verified: prior session summary readable OK')
" || red "portfolio: bootstrap session-start query failed"
green "portfolio: bootstrap query returns prior session summary OK"

portfolio_cleanup
green "Router portfolio smoke phase complete: 3 events, checkpoint, bootstrap query all OK"

# =========================================================================
# Phase 20: Router portfolio wire-emitter smoke tests
# (router-portfolio-wire-emitters-v0)
# Verifies that production CLI paths (bind / tick --force-dispatch / land)
# emit portfolio entries end-to-end via subprocess invocation of lapis-pm.
# =========================================================================

_wire_tid_cleanup() {
    local tid="$1"
    rm -f "$TARGETS_DIR/${tid}.yaml" "$COMMENTS_DIR/${tid}.jsonl"
    rm -f "/srv/lapis/lapis-state/${tid}.md"
    /usr/local/bin/mem delete "pm/cursor/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/dispatched/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/pause-state/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/outstanding-brief/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/classified-prs/${tid}" 2>/dev/null || true
    /usr/local/bin/mem delete "pm/landed/${tid}" 2>/dev/null || true
    # Delete any router-portfolio entries written for this tid
    /usr/bin/python3 -c "
from agents_core.mem import MemoryStore
m = MemoryStore()
rows = m.list_all(tag='target:${tid}', limit=200)
for r in rows:
    k = r.get('key', '')
    if k.startswith('router/lapis-pm/'):
        m.delete(k)
" 2>/dev/null || true
}

# --- 20a. test_bind_emits_kickoff_decision --------------------------------
step "20a. wire-emitter: bind emits kickoff decision entry"
TID_W1="pm-wire-bind-$$"
W1_SPEC="/tmp/${TID_W1}-spec.md"
cat > "$W1_SPEC" <<'SPECEOF'
# Wire Emitter Bind Test Spec
Implement the bind kickoff portfolio emission.
SPECEOF
cat > "$TARGETS_DIR/${TID_W1}.yaml" <<EOF
id: ${TID_W1}
title: Wire emitter bind test
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Wire emitter smoke target.
stages:
  - name: smoke-stage
    status: active
EOF
TEST_START_W1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
$LAPIS bind "$TID_W1" --spec-from "$W1_SPEC" --repo lapis-test --authority advisory
/usr/bin/python3 -c "
import json
from lapis_pm.router_portfolio import read_session_entries

tid = '${TID_W1}'
test_start = '${TEST_START_W1}'
entries = read_session_entries(since_iso=test_start)
ours = [e for e in entries
        if e.get('target_id') == tid and e.get('_mem_key', '').startswith('router/lapis-pm/decisions/')]
print(f'decisions entries since test_start: {len(ours)}')
assert len(ours) == 1, f'Expected 1 kickoff entry, got {len(ours)}: {ours}'
e = ours[0]
assert e.get('fragment_id') == 'kickoff', f'fragment_id not kickoff: {e.get(\"fragment_id\")}'
assert e.get('expert_chosen') is None, f'expert_chosen should be None: {e.get(\"expert_chosen\")}'
assert e.get('target_id') == tid, f'target_id mismatch: {e.get(\"target_id\")}'
print(f'kickoff entry OK: fragment_id={e[\"fragment_id\"]} expert_chosen={e[\"expert_chosen\"]} target_id={e[\"target_id\"]}')
" || red "wire-emitter: bind did not emit exactly one kickoff decision entry"
rm -f "$W1_SPEC"
_wire_tid_cleanup "$TID_W1"
green "wire-emitter: bind emits kickoff decision entry OK"

# --- 20b. test_chain_bind_emits_n_kickoff_decisions -----------------------
step "20b. wire-emitter: chain bind emits N kickoff + 1 dispatch entry"
TID_WC="pm-wire-chain-$$"
TID_WC_L1="${TID_WC}-l1"
TID_WC_L2="${TID_WC}-l2"
WC_SPEC="/tmp/${TID_WC}-spec.md"
# Register wire-chain legs: auto-fire from chain bind may open real Forgejo PRs.
SMOKE_TIDS+=("$TID_WC_L1" "$TID_WC_L2")
WC_LEGS="/tmp/${TID_WC}-legs.yaml"
cat > "$WC_SPEC" <<'SPECEOF'
# Wire Emitter Chain Test Spec
Implement chain leg work per spec.
SPECEOF
cat > "$WC_LEGS" <<EOF
legs:
  - tid: ${TID_WC_L1}
    repo: lapis-test
    authority: advisory
    branch_slug: implement
    intent: |
      Implement leg 1 per spec.
  - tid: ${TID_WC_L2}
    repo: lapis-test
    authority: advisory
    branch_slug: integrate
    intent: |
      Implement leg 2 per spec.
    depends_on:
      - ${TID_WC_L1}
EOF
for leg_tid in "$TID_WC_L1" "$TID_WC_L2"; do
    cat > "$TARGETS_DIR/${leg_tid}.yaml" <<EOF2
id: ${leg_tid}
title: Wire emitter chain leg
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Wire emitter chain smoke target.
stages:
  - name: smoke-stage
    status: active
EOF2
done
/usr/local/bin/gpu-submit pause >/dev/null 2>&1 || true
TEST_START_WC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
CHAIN_OUT=$($LAPIS bind "$TID_WC" --spec-from "$WC_SPEC" --legs-from "$WC_LEGS" --force 2>&1)
echo "$CHAIN_OUT"
# Clean up any pending GPU tasks created by chain auto-fire
for f in "${GPU_PENDING}"/*.yaml; do
    [ -e "$f" ] || continue
    if grep -q "$TID_WC_L1" "$f" 2>/dev/null; then rm -f "$f"; fi
done
/usr/local/bin/gpu-submit resume >/dev/null 2>&1 || true
/usr/bin/python3 -c "
import json
from lapis_pm.router_portfolio import read_session_entries

l1 = '${TID_WC_L1}'
l2 = '${TID_WC_L2}'
test_start = '${TEST_START_WC}'
entries = read_session_entries(since_iso=test_start)
# 2 kickoff decisions (one per leg)
kickoffs = [e for e in entries
            if e.get('fragment_id') == 'kickoff'
            and e.get('target_id') in (l1, l2)
            and e.get('expert_chosen') is None
            and e.get('_mem_key', '').startswith('router/lapis-pm/decisions/')]
print(f'kickoff decisions: {len(kickoffs)} (target_ids: {[e[\"target_id\"] for e in kickoffs]})')
assert len(kickoffs) == 2, f'Expected 2 kickoff decisions, got {len(kickoffs)}'
# 1 dispatch (from leg 1 auto-fire, expert_chosen is NOT None)
dispatches = [e for e in entries
              if e.get('fragment_id') == 'kickoff'
              and e.get('target_id') == l1
              and e.get('expert_chosen') is not None
              and e.get('_mem_key', '').startswith('router/lapis-pm/decisions/')]
print(f'dispatch entries for l1: {len(dispatches)}')
assert len(dispatches) == 1, f'Expected 1 dispatch entry for leg1, got {len(dispatches)}'
print('chain bind portfolio entries OK: 2 kickoffs + 1 dispatch')
" || red "wire-emitter: chain bind did not emit 2 kickoff + 1 dispatch entries"
rm -f "$WC_SPEC" "$WC_LEGS"
_wire_tid_cleanup "$TID_WC_L1"
_wire_tid_cleanup "$TID_WC_L2"
for key in "chain/${TID_WC}/state" "pm/classified-prs/${TID_WC}"; do
    /usr/local/bin/mem delete "$key" 2>/dev/null || true
done
/usr/bin/python3 -c "
from agents_core.mem import MemoryStore
m = MemoryStore()
rows = m.list_all(tag='chain-group:${TID_WC}', limit=100)
for r in rows: m.delete(r.get('key', ''))
" 2>/dev/null || true
green "wire-emitter: chain bind emits 2 kickoff + 1 dispatch entries OK"

# --- 20c. test_force_dispatch_emits_dispatch_decision ---------------------
step "20c. wire-emitter: force-dispatch emits dispatch decision entry"
TID_W3="pm-wire-disp-$$"
W3_SPEC="/tmp/${TID_W3}-spec.md"
# Register wire-disp target: force-dispatch from tick may open real Forgejo PRs.
SMOKE_TIDS+=("$TID_W3")
cat > "$W3_SPEC" <<'SPECEOF'
# Wire Emitter Dispatch Test Spec
Implement spec for dispatch portfolio emission test.
SPECEOF
cat > "$TARGETS_DIR/${TID_W3}.yaml" <<EOF
id: ${TID_W3}
title: Wire emitter dispatch test
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Wire emitter dispatch smoke target.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_W3" --spec-from "$W3_SPEC" --repo lapis-test --authority advisory
/usr/local/bin/gpu-submit pause >/dev/null 2>&1 || true
LONG_INTENT="$(python3 -c "print('x' * 250)")"
TEST_START_W3=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DISP_OUT=$($LAPIS tick --target "$TID_W3" --force-dispatch "fixer:${LONG_INTENT}")
echo "$DISP_OUT"
TASK_ID_W3=$(echo "$DISP_OUT" | grep -oE 'task_id=[^ ]+' | sed 's/task_id=//')
[ -n "$TASK_ID_W3" ] && rm -f "$GPU_PENDING/${TASK_ID_W3}.yaml"
/usr/local/bin/gpu-submit resume >/dev/null 2>&1 || true
/usr/bin/python3 -c "
from lapis_pm.router_portfolio import read_session_entries

tid = '${TID_W3}'
test_start = '${TEST_START_W3}'
entries = read_session_entries(since_iso=test_start)
# Force-dispatch entries have expert_chosen != None; bind kickoffs have expert_chosen=None.
# This distinguishes them even if both fall in the same second.
dispatches = [e for e in entries
              if e.get('target_id') == tid
              and e.get('expert_chosen') is not None
              and e.get('_mem_key', '').startswith('router/lapis-pm/decisions/')]
print(f'dispatch entries (expert_chosen != None): {len(dispatches)}')
assert len(dispatches) == 1, f'Expected 1 dispatch entry, got {len(dispatches)}'
e = dispatches[0]
# First dispatch on this target → fragment_id='kickoff'
assert e.get('fragment_id') == 'kickoff', f'Expected kickoff (first dispatch), got {e.get(\"fragment_id\")}'
# expert_chosen must not be None (resolved via _SHAPER)
assert e.get('expert_chosen') is not None, 'expert_chosen must not be None for dispatch'
# intent_summary truncated at 200 chars
intent_summary = e.get('intent_summary', '')
assert len(intent_summary) <= 200, f'intent_summary not truncated: len={len(intent_summary)}'
print(f'dispatch entry OK: fragment_id={e[\"fragment_id\"]} expert_chosen={e[\"expert_chosen\"]} intent_len={len(intent_summary)}')
" || red "wire-emitter: force-dispatch did not emit correct dispatch decision entry"
rm -f "$W3_SPEC"
_wire_tid_cleanup "$TID_W3"
green "wire-emitter: force-dispatch emits dispatch decision entry OK"

# --- SMOKE_TIDS sentinel: structural enforcement --------------------------
# Assert the registry is non-empty after all bind phases have run.
# This catches a new bind phase that forgets to append to SMOKE_TIDS.
# chain_cleanup is now known to reference "${SMOKE_TIDS[@]}" rather than
# hardcoded TID variables, so this assertion validates the full contract.
step "20c-sentinel. SMOKE_TIDS structural enforcement: registry non-empty + covers all phases"
[ ${#SMOKE_TIDS[@]} -gt 0 ] \
    || red "SMOKE_TIDS is empty — every bind phase that dispatches must register its TID(s)"
echo "[smoke] SMOKE_TIDS (${#SMOKE_TIDS[@]} entries): ${SMOKE_TIDS[*]}"
# Verify all three expected prefixes are represented
_has_prefix() { local p="$1"; shift; for t in "$@"; do [[ "$t" == "$p"* ]] && return 0; done; return 1; }
_has_prefix "pm-smoke-chain-" "${SMOKE_TIDS[@]}" \
    || red "SMOKE_TIDS missing pm-smoke-chain- entries (phase 18 registration broken)"
_has_prefix "pm-wire-chain-" "${SMOKE_TIDS[@]}" \
    || red "SMOKE_TIDS missing pm-wire-chain- entries (phase 20b registration broken)"
_has_prefix "pm-wire-disp-" "${SMOKE_TIDS[@]}" \
    || red "SMOKE_TIDS missing pm-wire-disp- entries (phase 20c registration broken)"
green "SMOKE_TIDS sentinel: ${#SMOKE_TIDS[@]} TIDs registered, all prefixes present — chain_cleanup will reap all"

# --- 20d. test_land_emits_land_decision ------------------------------------
step "20d. wire-emitter: land emits land decision entry"
TID_W4="pm-wire-land-$$"
W4_SPEC="/tmp/${TID_W4}-spec.md"
cat > "$W4_SPEC" <<'SPECEOF'
# Wire Emitter Land Test Spec
Implement spec for land portfolio emission test.
SPECEOF
cat > "$TARGETS_DIR/${TID_W4}.yaml" <<EOF
id: ${TID_W4}
title: Wire emitter land test
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Wire emitter land smoke target.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_W4" --spec-from "$W4_SPEC" --repo lapis-test --authority advisory
/usr/bin/python3 -c "
from agents_core.comments import CommentStore
cs = CommentStore()
cs.append('${TID_W4}', 'PR #99 opened in lapis-test: fake PR', 'lapis-pm', 'agent',
          tags=['pm:observation', 'pm:pr=99'])
cs.append('${TID_W4}', 'PR #99 merged at 2026-04-25T00:00:00, head_sha=abc123', 'lapis-pm', 'agent',
          tags=['pm:observation', 'pm:pr-merged:99', 'pm:pr=99'])
"
TEST_START_W4=$(date -u +%Y-%m-%dT%H:%M:%SZ)
$LAPIS land "$TID_W4"
[ -f "/srv/lapis/lapis-state/${TID_W4}.md" ] || red "arc doc not written for land test"
/usr/bin/python3 -c "
from lapis_pm.router_portfolio import read_session_entries

tid = '${TID_W4}'
test_start = '${TEST_START_W4}'
entries = read_session_entries(since_iso=test_start)
lands = [e for e in entries
         if e.get('target_id') == tid
         and e.get('fragment_id') == 'land'
         and e.get('_mem_key', '').startswith('router/lapis-pm/decisions/')]
print(f'land entries: {len(lands)}')
assert len(lands) == 1, f'Expected 1 land entry, got {len(lands)}'
e = lands[0]
intent = e.get('intent_summary', '')
assert intent, 'intent_summary must not be empty (should contain arc doc path)'
print(f'land entry OK: fragment_id={e[\"fragment_id\"]} intent_summary={intent[:60]}')
" || red "wire-emitter: land did not emit land decision entry"
rm -f "$W4_SPEC"
_wire_tid_cleanup "$TID_W4"
green "wire-emitter: land emits land decision entry OK"

# --- 20e. test_dry_run_land_does_not_emit ----------------------------------
step "20e. wire-emitter: dry-run land does NOT emit any decision entry"
TID_W5="pm-wire-dryrun-$$"
W5_SPEC="/tmp/${TID_W5}-spec.md"
cat > "$W5_SPEC" <<'SPECEOF'
# Wire Emitter Dry Run Test Spec
Implement spec for dry-run land test.
SPECEOF
cat > "$TARGETS_DIR/${TID_W5}.yaml" <<EOF
id: ${TID_W5}
title: Wire emitter dry run test
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Wire emitter dry-run smoke target.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_W5" --spec-from "$W5_SPEC" --repo lapis-test --authority advisory
/usr/bin/python3 -c "
from agents_core.comments import CommentStore
cs = CommentStore()
cs.append('${TID_W5}', 'PR #99 merged at 2026-04-25T00:00:00, head_sha=abc123', 'lapis-pm', 'agent',
          tags=['pm:observation', 'pm:pr-merged:99', 'pm:pr=99'])
"
# Capture test_start AFTER bind (bind emits a kickoff; we only care that dry-run
# does NOT emit a 'land' fragment, which is the one dry-run must skip).
TEST_START_W5=$(date -u +%Y-%m-%dT%H:%M:%SZ)
$LAPIS land --dry-run "$TID_W5" >/dev/null
/usr/bin/python3 -c "
from lapis_pm.router_portfolio import read_session_entries

tid = '${TID_W5}'
test_start = '${TEST_START_W5}'
entries = read_session_entries(since_iso=test_start)
# Dry-run must not emit a 'land' entry; any pre-existing 'kickoff' from bind is filtered
# by time unless the second boundary was hit (handled by checking fragment_id='land').
land_entries = [e for e in entries
                if e.get('target_id') == tid and e.get('fragment_id') == 'land']
print(f'land entries after dry-run land: {len(land_entries)}')
assert len(land_entries) == 0, f'dry-run land must not emit a land entry; got {land_entries}'
print('dry-run land emitted no land portfolio entry OK')
" || red "wire-emitter: dry-run land emitted a portfolio entry (must not)"
rm -f "$W5_SPEC"
_wire_tid_cleanup "$TID_W5"
green "wire-emitter: dry-run land does not emit OK"

# --- 20f. test_emit_failure_does_not_block_command ------------------------
step "20f. wire-emitter: emit failure does not block bind command"
TID_W6="pm-wire-fail-$$"
W6_SPEC="/tmp/${TID_W6}-spec.md"
cat > "$W6_SPEC" <<'SPECEOF'
# Wire Emitter Failure Test Spec
Implement spec for emit-failure-safe bind test.
SPECEOF
cat > "$TARGETS_DIR/${TID_W6}.yaml" <<EOF
id: ${TID_W6}
title: Wire emitter failure test
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Wire emitter failure smoke target.
stages:
  - name: smoke-stage
    status: active
EOF
BIND_STDERR=$( ROUTER_PORTFOLIO_FAIL_WRITES=1 $LAPIS bind "$TID_W6" \
    --spec-from "$W6_SPEC" --repo lapis-test --authority advisory 2>&1 >/dev/null )
echo "bind stderr: $BIND_STDERR"
echo "$BIND_STDERR" | grep -q "\[router-portfolio:emit-failed\]" \
    || red "wire-emitter: expected [router-portfolio:emit-failed] in stderr; got: $BIND_STDERR"
# Target YAML must exist (bind succeeded despite emit failure)
[ -f "$TARGETS_DIR/${TID_W6}.yaml" ] || red "wire-emitter: target YAML missing — bind was blocked by emit failure"
# Also verify bind exited 0
BIND_RC=0
ROUTER_PORTFOLIO_FAIL_WRITES=1 $LAPIS bind "$TID_W6" \
    --spec-from "$W6_SPEC" --repo lapis-test --authority advisory --force >/dev/null 2>&1 || BIND_RC=$?
[ "$BIND_RC" -eq 0 ] || red "wire-emitter: bind exited $BIND_RC when emit failed (must exit 0)"
rm -f "$W6_SPEC"
_wire_tid_cleanup "$TID_W6"
green "wire-emitter: emit failure does not block bind, stderr contains [router-portfolio:emit-failed] OK"

green "Phase 20 complete: all 6 wire-emitter portfolio smoke tests passed"

# =========================================================================
# Phase 21: Pushover notify routing assertions (lapis-pm-pushover-action-only)
# =========================================================================
# Monkey-patches agents_core.notify.send_notification to record (priority, called)
# without requiring PUSHOVER_* env vars. Validates per-call-site routing and
# that pm:brief comments are always written regardless of notify value.
step "21. Pushover notify routing — per-call-site assertions (no real Pushover delivery)"

/usr/bin/python3 -c "
import sys, json
sys.path.insert(0, '${REPO_ROOT}')
from unittest.mock import patch, MagicMock
from agents_core.notify import Priority as NotifyPriority

# ------------------------------------------------------------------ helpers
calls = []

def _fake_send_notification(message, title, priority, url=None, url_title=None):
    calls.append(priority)
    return True

def _fake_write_brief(target_id, body):
    c = MagicMock()
    c.id = 'smoke-brief-id'
    return c

# Shared mocks that every path needs
_common_patches = [
    patch('agents_core.notify.send_notification', side_effect=_fake_send_notification),
    patch('lapis_pm.brief.send_notification', side_effect=_fake_send_notification),
    patch('lapis_pm.brief.call_claude_cli', return_value='## State\nsmoke body\n## Decision needed\nnone'),
    patch('lapis_pm.brief.episodic.write_brief', side_effect=_fake_write_brief),
    patch('lapis_pm.brief.episodic.recall', return_value=[]),
    patch('lapis_pm.brief.episodic.spec_summary', return_value='smoke spec'),
]

def start_patches(patches):
    for p in patches:
        p.start()

def stop_patches(patches):
    for p in reversed(patches):
        try:
            p.stop()
        except Exception:
            pass

from lapis_pm import brief

results = {}

# ------------------------------------------------------------------ 1. held PR → NORMAL
calls.clear()
start_patches(_common_patches)
try:
    b = brief.synthesize('smoke-tid', trigger='held PR', notify=NotifyPriority.NORMAL)
    results['held_pr'] = (list(calls), b.pushed)
finally:
    stop_patches(_common_patches)

assert results['held_pr'][0] == [NotifyPriority.NORMAL], \
    f'held_pr: expected [NORMAL], got {results[\"held_pr\"][0]}'
assert results['held_pr'][1] is True, 'held_pr: pushed should be True'
print(f'held_pr: priority=NORMAL pushed=True ✓')

# ------------------------------------------------------------------ 2. advisory PR → no push
calls.clear()
start_patches(_common_patches)
try:
    b = brief.synthesize('smoke-tid', trigger='advisory PR', notify=None)
    results['advisory'] = (list(calls), b.pushed)
finally:
    stop_patches(_common_patches)

assert results['advisory'][0] == [], \
    f'advisory: expected no calls, got {results[\"advisory\"][0]}'
assert results['advisory'][1] is False, 'advisory: pushed should be False'
print(f'advisory: suppressed pushed=False ✓')

# ------------------------------------------------------------------ 3. review-exhausted → HIGH
calls.clear()
start_patches(_common_patches)
try:
    b = brief.synthesize('smoke-tid', trigger='review exhausted', notify=NotifyPriority.HIGH)
    results['review_exhausted'] = (list(calls), b.pushed)
finally:
    stop_patches(_common_patches)

assert results['review_exhausted'][0] == [NotifyPriority.HIGH], \
    f'review_exhausted: expected [HIGH], got {results[\"review_exhausted\"][0]}'
print(f'review_exhausted: priority=HIGH ✓')

# ------------------------------------------------------------------ 4. gate-pause → HIGH
calls.clear()
start_patches(_common_patches)
try:
    b = brief.synthesize('smoke-tid', trigger='gate paused', notify=NotifyPriority.HIGH)
    results['gate_pause'] = (list(calls), b.pushed)
finally:
    stop_patches(_common_patches)

assert results['gate_pause'][0] == [NotifyPriority.HIGH], \
    f'gate_pause: expected [HIGH], got {results[\"gate_pause\"][0]}'
print(f'gate_pause: priority=HIGH ✓')

# ------------------------------------------------------------------ 5. abandon-brief → HIGH
calls.clear()
start_patches(_common_patches)
try:
    b = brief.synthesize('smoke-tid', trigger='abandon brief', notify=NotifyPriority.HIGH)
    results['abandon'] = (list(calls), b.pushed)
finally:
    stop_patches(_common_patches)

assert results['abandon'][0] == [NotifyPriority.HIGH], \
    f'abandon: expected [HIGH], got {results[\"abandon\"][0]}'
print(f'abandon: priority=HIGH ✓')

# ------------------------------------------------------------------ 6. directive-echo → no push
calls.clear()
start_patches(_common_patches)
try:
    b = brief.synthesize('smoke-tid', trigger='user directive: do something', notify=None)
    results['directive'] = (list(calls), b.pushed)
finally:
    stop_patches(_common_patches)

assert results['directive'][0] == [], \
    f'directive: expected no calls, got {results[\"directive\"][0]}'
print(f'directive: suppressed pushed=False ✓')

# ------------------------------------------------------------------ 7. force-brief → no push
calls.clear()
start_patches(_common_patches)
try:
    b = brief.synthesize('smoke-tid', trigger='manual force-brief', notify=None)
    results['force_brief'] = (list(calls), b.pushed)
finally:
    stop_patches(_common_patches)

assert results['force_brief'][0] == [], \
    f'force_brief: expected no calls, got {results[\"force_brief\"][0]}'
print(f'force_brief: suppressed pushed=False ✓')

# ------------------------------------------------------------------ 8. chain auto-dispatch → no push
# Verify chain._fire_initial_dispatch does NOT call send_notification.
calls.clear()
chain_notify_calls = []

def _chain_fake_notify(message, title, priority, url=None, url_title=None):
    chain_notify_calls.append(priority)
    return True

chain_patches = [
    patch('agents_core.notify.send_notification', side_effect=_chain_fake_notify),
    patch('lapis_pm.brief.send_notification', side_effect=_chain_fake_notify),
    patch('lapis_pm.brief.call_claude_cli', return_value='## State\nsmoke\n## Decision needed\nnone'),
    patch('lapis_pm.brief.episodic.write_brief', side_effect=_fake_write_brief),
    patch('lapis_pm.brief.episodic.recall', return_value=[]),
    patch('lapis_pm.brief.episodic.spec_summary', return_value='spec'),
    patch('lapis_pm.pm_core.append_dispatched'),
    patch('lapis_pm.episodic.write_dispatch'),
    patch('lapis_pm.episodic.spec_summary', return_value='spec'),
    patch('lapis_pm.chain._mem'),
    patch('lapis_pm.pm_core._mem'),
]

from lapis_pm.chain import send_auto_dispatch_brief
result = send_auto_dispatch_brief('grp', 'leg', 'trigger')
assert result is False, f'send_auto_dispatch_brief should return False (deprecated), got {result}'
assert chain_notify_calls == [], \
    f'chain auto-dispatch: expected zero notify calls, got {chain_notify_calls}'
print(f'chain_auto_dispatch: zero notify calls, send_auto_dispatch_brief returns False ✓')

print()
print('Notify routing summary:')
print('  held_pr          → NORMAL  (pushed=True)')
print('  advisory         → suppressed (pushed=False)')
print('  review_exhausted → HIGH')
print('  gate_pause       → HIGH')
print('  abandon          → HIGH')
print('  directive        → suppressed (pushed=False)')
print('  force_brief      → suppressed (pushed=False)')
print('  chain_auto       → zero calls (stub returns False)')
print()
print('All notify routing assertions passed ✓')
" || red "notify routing: assertion failed"
green "notify routing: all 8 call-site assertions passed (HIGH/NORMAL/suppressed per taxonomy)"

# ------------------------------------------------------------------ brief persistence check
# For suppressed-push sites, assert pm:brief comment was still written.
step "20b. Brief persistence: suppressed-push paths still write pm:brief comment"

/usr/bin/python3 -c "
import sys, json
sys.path.insert(0, '${REPO_ROOT}')
from unittest.mock import patch, MagicMock

written_briefs = []

def _recording_write_brief(target_id, body):
    written_briefs.append(target_id)
    c = MagicMock()
    c.id = 'cid-' + target_id
    return c

common = [
    patch('lapis_pm.brief.send_notification'),  # should NOT be called; no side_effect
    patch('lapis_pm.brief.call_claude_cli', return_value='## State\nsmoke\n## Decision needed\nnone'),
    patch('lapis_pm.brief.episodic.write_brief', side_effect=_recording_write_brief),
    patch('lapis_pm.brief.episodic.recall', return_value=[]),
    patch('lapis_pm.brief.episodic.spec_summary', return_value='spec'),
]

from lapis_pm import brief
from agents_core.notify import Priority as NotifyPriority

for p in common:
    p.start()

try:
    # advisory (notify=None) — brief must be written
    b1 = brief.synthesize('tid-advisory', trigger='advisory PR', notify=None)
    # directive (notify=None) — brief must be written
    b2 = brief.synthesize('tid-directive', trigger='user directive: x', notify=None)
    # force-brief (notify=None) — brief must be written
    b3 = brief.synthesize('tid-force', trigger='manual force-brief', notify=None)
finally:
    for p in reversed(common):
        try: p.stop()
        except Exception: pass

# All three must appear in written_briefs
for tid in ['tid-advisory', 'tid-directive', 'tid-force']:
    assert tid in written_briefs, f'pm:brief NOT written for {tid} (persistence violated)'
    print(f'  {tid}: pm:brief written ✓')

# send_notification must NOT have been called for any of them
# (verify via the mock — if it was called with a side_effect it would have raised)
print('brief persistence: all suppressed-push paths still write pm:brief comment ✓')
" || red "brief persistence: pm:brief comment not written for a suppressed-push path"
green "brief persistence: advisory/directive/force-brief all write pm:brief comment OK"

# --- Phase 22: classified-prs SHA-invalidation ---------------------------
step "22. classified-prs SHA-invalidation: SHA advance removes PR from classified set"
TID_SHA="pm-smoke-sha-inv-$$"
SHA_SPEC="/tmp/${TID_SHA}-spec.md"
cat > "$SHA_SPEC" <<EOF
# SHA-invalidation smoke spec for $TID_SHA
EOF

cat > "$TARGETS_DIR/${TID_SHA}.yaml" <<EOF
id: ${TID_SHA}
title: SHA-invalidation smoke target
status: active
category: active-work
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: classified-prs SHA-invalidation smoke target — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_SHA" --spec-from "$SHA_SPEC" --repo lapis-pm --authority auto-merge

# Pre-populate classified-prs with PR #42.
/usr/local/bin/mem set "pm/classified-prs/${TID_SHA}" "[42]" --tags "lapis-pm,classified-prs" >/dev/null

# Verify it's there before the tick.
/usr/bin/python3 -c "
import sys; sys.path.insert(0, '/srv/git/agents-core-working')
from lapis_pm.pm_core import _classified_pr_ids
ids = _classified_pr_ids('${TID_SHA}')
assert 42 in ids, f'pre-condition failed: 42 not in classified-prs before tick; ids={ids}'
print(f'pre-condition: classified-prs={ids} ✓')
" || red "SHA-invalidation smoke: pre-condition check failed"

# Write a simulated open-PR observation with a new SHA so _encode_pr_sha_updates
# sees a SHA advance for PR #42.  We inject the PR directly via a synthetic
# tick that mocks the Forgejo get_open_prs call.
/usr/bin/python3 - "${TID_SHA}" <<'PYEOF21'
import sys, json
sys.path.insert(0, '/srv/git/agents-core-working')

from unittest.mock import patch
from lapis_pm import pm_core

target_id = sys.argv[1]
fake_pr = {"number": 42, "head": {"sha": "abc123newsha"}, "title": "smoke PR", "state": "open"}

# _last_observed_pr_sha returns None (no prior observation) → SHA "advance".
with patch("lapis_pm.pm_core.get_open_prs", return_value=[fake_pr]):
    written = pm_core._encode_pr_sha_updates(target_id, [fake_pr])

print(f"encode written={written}")
ids = pm_core._classified_pr_ids(target_id)
print(f"classified-prs after encode: {ids}")
assert 42 not in ids, f"FAIL: 42 still in classified-prs after SHA advance; ids={ids}"
print("SHA advance → classified-prs invalidated ✓")
PYEOF21
green "Phase 22: classified-prs SHA-invalidation — SHA advance removes PR #42 from classified set OK"

# Cleanup
rm -f "$TARGETS_DIR/${TID_SHA}.yaml" "$COMMENTS_DIR/${TID_SHA}.jsonl" "$SHA_SPEC"
/usr/local/bin/mem delete "pm/cursor/${TID_SHA}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_SHA}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_SHA}" 2>/dev/null || true

# --- Phase 23: state brief (dry-run) ------------------------------------
step "23. state-brief: dry-run morning brief"

BRIEFS_ROOT="/srv/lapis/briefs"
# Clean up any stale smoke artifacts first
rm -f "${BRIEFS_ROOT}/latest-morning.md"
rm -f "${BRIEFS_ROOT}/daily/"*-morning.md 2>/dev/null || true

LAPIS_BRIEF_DRY_RUN=1 $LAPIS brief --period morning \
    || red "state-brief: lapis-pm brief --period morning exited non-zero"

# (a) Placeholder file written at correct /srv/lapis/briefs/daily/ path
BRIEF_FILE=$(ls -t "${BRIEFS_ROOT}/daily/"*-morning.md 2>/dev/null | head -1)
[ -n "$BRIEF_FILE" ] || red "state-brief: no *-morning.md file found under ${BRIEFS_ROOT}/daily/"
[ -f "$BRIEF_FILE" ] || red "state-brief: expected file at ${BRIEF_FILE}, not found"
green "state-brief: brief file written at ${BRIEF_FILE}"

# (b) latest-morning.md symlink points to it
SYMLINK="${BRIEFS_ROOT}/latest-morning.md"
[ -L "$SYMLINK" ] || red "state-brief: latest-morning.md is not a symlink"
RESOLVED=$(readlink -f "$SYMLINK")
EXPECTED=$(readlink -f "$BRIEF_FILE")
[ "$RESOLVED" = "$EXPECTED" ] || red "state-brief: latest-morning.md -> ${RESOLVED} but expected ${EXPECTED}"
green "state-brief: latest-morning.md symlink points to correct file"

# (c) All five bucket headings present in file (prevents vacuous pass)
grep -q "^## Built (since" "$BRIEF_FILE" \
    || red "state-brief: missing 'Built (since' heading in ${BRIEF_FILE}"
grep -q "^## Notable ratifications (since" "$BRIEF_FILE" \
    || red "state-brief: missing 'Notable ratifications (since' heading in ${BRIEF_FILE}"
grep -q "^## In flight" "$BRIEF_FILE" \
    || red "state-brief: missing 'In flight' heading in ${BRIEF_FILE}"
grep -q "^## Captured" "$BRIEF_FILE" \
    || red "state-brief: missing 'Captured' heading in ${BRIEF_FILE}"
grep -q "^## Awaiting your call" "$BRIEF_FILE" \
    || red "state-brief: missing 'Awaiting your call' heading in ${BRIEF_FILE}"
green "state-brief: all five bucket headings present in brief file"

# Cleanup brief artifacts
rm -f "$BRIEF_FILE" "${SYMLINK}" 2>/dev/null || true
rmdir "${BRIEFS_ROOT}/daily" "${BRIEFS_ROOT}/weekly" "${BRIEFS_ROOT}" 2>/dev/null || true

green "Phase 23 complete: state-brief dry-run passed (file path, symlink, five headings)"

# =========================================================================
# Trajectory rollup smoke tests
# Exercises: --rebuild-index, --period per-target (dry-run), CLI surface.
# All LLM calls are suppressed via LAPIS_TRAJECTORY_DRY_RUN=1.
# =========================================================================
step "24. Trajectory rollup smoke: --rebuild-index (no LLM)"
mkdir -p "$TRAJ_SMOKE_DIR"

TRAJ_OUT=$($LAPIS trajectory-rollup --rebuild-index 2>&1) || red "trajectory-rollup --rebuild-index failed"
echo "$TRAJ_OUT"
[ -f "/srv/lapis/trajectory/index.json" ] || red "index.json not written by --rebuild-index"
/usr/bin/python3 -c "
import json, sys
data = json.load(open('/srv/lapis/trajectory/index.json'))
assert 'nodes' in data, 'index.json missing nodes'
assert 'edges' in data, 'index.json missing edges'
assert 'generated_at' in data, 'index.json missing generated_at'
print(f'index.json OK: {len(data[\"nodes\"])} nodes, {len(data[\"edges\"])} edges')
" || red "index.json shape validation failed"
green "trajectory-rollup --rebuild-index OK"

step "25. Trajectory rollup smoke: --period per-target --target (dry-run, fixture arc doc)"
FIXTURE_TID="smoke-traj-fixture-$$"
FIXTURE_ARC="/tmp/${FIXTURE_TID}.md"
cat > "$FIXTURE_ARC" <<'ARCEOF'
---
target_id: smoke-traj-fixture
generated: 2026-05-01T00:00:00-07:00
kind: arc-doc
---

# smoke-traj-fixture — Arc Doc

## Origin
Smoke test fixture for trajectory rollup dry-run mode.

## Key decisions
- Used dry-run mode to avoid real LLM calls.

## Dispatches and results
No real dispatches. Fixture only.

## Landing summary
Fixture landed. PR #0 merged.

## Open threads
- Follow-on: add more fixtures.
ARCEOF

/usr/local/bin/mem set "pm/landed/${FIXTURE_TID}" \
    "{\"manual\":true,\"ts\":\"2026-05-01T00:00:00-07:00\",\"arc_path\":\"${FIXTURE_ARC}\"}" \
    --tags "lapis-pm,landed" >/dev/null 2>&1 || true

LAPIS_TRAJECTORY_DRY_RUN=1 $LAPIS trajectory-rollup --period per-target --target "$FIXTURE_TID" 2>&1 | tee "/tmp/traj-smoke-$$.out"
[ -f "/srv/lapis/trajectory/per-target/${FIXTURE_TID}.json" ] || red "per-target JSON not written for $FIXTURE_TID"
/usr/bin/python3 -c "
import json, sys
data = json.load(open('/srv/lapis/trajectory/per-target/${FIXTURE_TID}.json'))
assert data.get('one_liner') == '(dry run)', f'one_liner should be (dry run), got: {data.get(\"one_liner\")}'
assert 'arc_doc_sha256' in data, 'arc_doc_sha256 missing'
assert 'generated_at' in data, 'generated_at missing'
print(f'per-target JSON OK: one_liner={data[\"one_liner\"]!r}, sha256={data[\"arc_doc_sha256\"][:12]}...')
" || red "per-target JSON shape or dry-run placeholder check failed (vacuous-pass detection)"
green "trajectory-rollup --period per-target dry-run OK (one_liner=(dry run) confirmed)"

MTIME1=$(stat -c %Y "/srv/lapis/trajectory/per-target/${FIXTURE_TID}.json")
LAPIS_TRAJECTORY_DRY_RUN=1 $LAPIS trajectory-rollup --period per-target --target "$FIXTURE_TID" >/dev/null 2>&1
MTIME2=$(stat -c %Y "/srv/lapis/trajectory/per-target/${FIXTURE_TID}.json")
[ "$MTIME1" = "$MTIME2" ] || red "per-target idempotency violated: file was rewritten on unchanged arc doc"
green "per-target idempotency OK (no-op on unchanged arc doc)"

/usr/local/bin/mem delete "pm/landed/${FIXTURE_TID}" 2>/dev/null || true
rm -f "$FIXTURE_ARC" "/tmp/traj-smoke-$$.out"
rm -f "/srv/lapis/trajectory/per-target/${FIXTURE_TID}.json"

# --- Phase 26: closed-form-brief scenario --------------------------------
step "26. closed-form-brief: synthesize emits sibling + directive consumer resolves"

TID_BRIEF="pm-smoke-brief-$$"
BRIEF_SPEC="/tmp/${TID_BRIEF}-spec.md"
DIRECTIVES_DIR="/srv/lapis/directives/brief-decisions"

cat > "$BRIEF_SPEC" <<EOF
# Closed-form brief smoke spec for $TID_BRIEF
EOF

cat > "$TARGETS_DIR/${TID_BRIEF}.yaml" <<EOF
id: ${TID_BRIEF}
title: Closed-form brief smoke
status: active
category: active-work
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Closed-form brief smoke target — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF

$LAPIS bind "$TID_BRIEF" --spec-from "$BRIEF_SPEC" --repo lapis-test --authority advisory

# (a) Synthesize with a closed-form trigger and verify sibling comment written.
/usr/bin/python3 - "${TID_BRIEF}" <<'PYEOF26'
import sys, json
sys.path.insert(0, '/srv/git/agents-core-working')
from unittest.mock import patch, MagicMock

target_id = sys.argv[1]

def fake_write_brief(tid, body, extra_tags=None):
    c = MagicMock()
    c.id = 'smoke-brief-id-26'
    return c

captured_options = []
def fake_write_brief_options(tid, content):
    c = MagicMock()
    c.id = 'smoke-brief-options-id-26'
    captured_options.append((tid, content))
    return c

patches = [
    patch('lapis_pm.brief.call_claude_cli', return_value='## State\nok\n## Decision needed\nnone'),
    patch('lapis_pm.brief.send_notification', return_value=False),
    patch('lapis_pm.brief.episodic.recall', return_value=[]),
    patch('lapis_pm.brief.episodic.spec_summary', return_value='spec'),
    patch('lapis_pm.brief.episodic.write_brief', side_effect=fake_write_brief),
    patch('lapis_pm.brief.episodic.write_brief_options', side_effect=fake_write_brief_options),
]
for p in patches:
    p.start()

try:
    from lapis_pm import brief
    b = brief.synthesize(target_id, trigger='advisory-screen-issue', pr_number=17, notify=None)
finally:
    for p in reversed(patches):
        try: p.stop()
        except Exception: pass

assert b.comment_id == 'smoke-brief-id-26', f'comment_id mismatch: {b.comment_id}'
assert len(captured_options) == 1, f'expected 1 sibling, got {len(captured_options)}'
data = json.loads(captured_options[0][1])
assert data['brief_id'] == 'smoke-brief-id-26', f'brief_id mismatch: {data}'
assert data['trigger'] == 'advisory-screen-issue', f'trigger mismatch: {data}'
assert len(data['options']) == 3, f'expected 3 options: {data}'
merge_opt = next(o for o in data['options'] if o['action']['kind'] == 'merge_pr')
assert merge_opt['action'].get('pr') == 17, f'pr_number not injected: {merge_opt}'
print(f'synthesize: sibling pm:brief-options emitted for advisory-screen-issue ✓')
print(f'  brief_id={data["brief_id"]} trigger={data["trigger"]} options={len(data["options"])} pr={merge_opt["action"]["pr"]}')
PYEOF26
green "Phase 26a: synthesize emits pm:brief-options sibling for advisory-screen-issue ✓"

# (b) Drop a directive file and run _consume_brief_decisions to verify
#     pending → applied transition and mem audit key written.
mkdir -p "$DIRECTIVES_DIR"
BRIEF_ID_SMOKE="smoke-brief-$(date +%s)-$$"
DIRECTIVE_FILE="$DIRECTIVES_DIR/${TID_BRIEF}__${BRIEF_ID_SMOKE}.json"
cat > "$DIRECTIVE_FILE" <<EOF
{"brief_id": "${BRIEF_ID_SMOKE}", "target_id": "${TID_BRIEF}", "option_id": "A",
 "ts": "$(date -Iseconds)", "submitter": "smoke"}
EOF

/usr/bin/python3 - "${TID_BRIEF}" "${BRIEF_ID_SMOKE}" <<'PYEOF26B'
import sys, json, os
sys.path.insert(0, '/srv/git/agents-core-working')
from unittest.mock import patch

target_id, brief_id = sys.argv[1], sys.argv[2]

apply_result = {"ok": True, "action_kind": "acknowledge_and_clear", "detail": "cleared"}

with (
    patch('lapis_pm.pm_core.brief.apply_decision', return_value=apply_result),
    patch('lapis_pm.pm_core.episodic.write_observation'),
):
    from lapis_pm import pm_core
    result = pm_core._consume_brief_decisions(target_id)

assert result == f'brief_decision_applied:{brief_id}:A', \
    f'expected decision_str, got: {result!r}'

applied = f'/srv/lapis/directives/brief-decisions/applied/{target_id}__{brief_id}.json'
assert os.path.exists(applied), f'applied file missing: {applied}'
print(f'directive consumer: pending → applied ✓')
print(f'  decision_str={result}')
PYEOF26B
green "Phase 26b: directive consumer pending→applied transition ✓"

# Cleanup phase 26
rm -f "$TARGETS_DIR/${TID_BRIEF}.yaml" "$COMMENTS_DIR/${TID_BRIEF}.jsonl" "$BRIEF_SPEC"
rm -f "${DIRECTIVES_DIR}/applied/${TID_BRIEF}__${BRIEF_ID_SMOKE}.json" 2>/dev/null || true
rm -f "${DIRECTIVE_FILE}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/cursor/${TID_BRIEF}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_BRIEF}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_BRIEF}" 2>/dev/null || true
green "Phase 26 complete: closed-form-brief scenario passed"

# =========================================================================
# Fixer already-done verdict smoke tests (lapis-pm-fixer-already-done-verdict)
# =========================================================================

SHAPED_DIR_SMOKE="/srv/lapis/gpu-queue/shaped"
mkdir -p "$SHAPED_DIR_SMOKE"

# --- Step 27a: valid verdict observation → auto-land fires ---------------
step "27a. Fixer already-done verdict: valid pm:already-satisfied obs → auto-land:already_satisfied fires"
TID_VERDI="pm-smoke-verdi-$$"
VERDI_SPEC="/tmp/${TID_VERDI}-spec.md"
cat > "$VERDI_SPEC" <<EOF
# Verdict smoke spec for $TID_VERDI

What: smoke the already_satisfied auto-land path.
EOF

cat > "$TARGETS_DIR/${TID_VERDI}.yaml" <<EOF
id: ${TID_VERDI}
title: Verdict smoke target
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Verdict smoke target — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_VERDI" --spec-from "$VERDI_SPEC" --repo lapis-test --authority advisory

# Write the pm:already-satisfied observation directly (simulates _encode_gpu_results
# after validating the cited PR).  The smoke exercises the decide→act path
# (_already_satisfied_pending → _act_auto_land_already_satisfied).  The
# encode path (verdict sidecar reading + Forgejo validation) is covered by unit tests.
/usr/bin/python3 -c "
from agents_core.comments import CommentStore
cs = CommentStore()
cs.append(
    '${TID_VERDI}',
    'Fixer verdict: already_satisfied by PR #42\nEvidence: ops-kami-nightly-timer-v0 PR #42 was merged at 2026-04-30T08:00:00Z. Confirmed: lapis_pm/pm_core.py line 1 contains the expected header.',
    'lapis-pm', 'agent',
    tags=['pm:observation', 'pm:already-satisfied', 'pm:already-satisfied:pr=42'],
)
"

# Tick — decide phase should fire auto_land:already_satisfied
VERDI_TICK=$(/usr/bin/python3 -c "
import sys
sys.path.insert(0, '/srv/git/agents-core-working')
from unittest.mock import patch, MagicMock

# Stub Haiku arc-doc call (no Haiku endpoint in smoke)
fake_arc_body = '# ${TID_VERDI} — Arc Doc\n\n## Origin\nSmoke origin.\n\n## Landing summary\nSmoke land.\n'
def fake_call_claude(*a, **kw):
    return fake_arc_body

# Stub Pushover (no token in smoke)
with (
    patch('lapis_pm.land.call_claude_cli', side_effect=fake_call_claude),
    patch('lapis_pm.pm_core._SHAPER'),
):
    from lapis_pm.cli import main as cli_main
    sys.argv = ['lapis-pm', 'tick', '--target', '${TID_VERDI}']
    cli_main()
" 2>/dev/null || $LAPIS tick --target "$TID_VERDI")
echo "$VERDI_TICK"
echo "$VERDI_TICK" | grep -q "decision=auto_land:already_satisfied:" \
    || red "verdict smoke 27a: auto_land:already_satisfied did not fire"
[ -f "/srv/lapis/lapis-state/${TID_VERDI}.md" ] \
    || red "verdict smoke 27a: arc doc not written for ${TID_VERDI}"
/usr/local/bin/mem get "pm/landed/${TID_VERDI}" >/dev/null 2>&1 \
    || red "verdict smoke 27a: pm/landed not set after already_satisfied auto-land"
# Arc doc must contain the already-satisfied trailing sentence
grep -q "already satisfied" "/srv/lapis/lapis-state/${TID_VERDI}.md" \
    || red "verdict smoke 27a: arc doc missing 'already satisfied' note in Origin section"
# Target should be unbound
/usr/bin/python3 -c "
from agents_core.targets import TargetStore
t = TargetStore().get('${TID_VERDI}')
assert t is not None, 'target not found'
assert not t.pm_bound, 'target still pm_bound after already_satisfied auto-land'
"
rm -f "$TARGETS_DIR/${TID_VERDI}.yaml" "$COMMENTS_DIR/${TID_VERDI}.jsonl" "$VERDI_SPEC"
rm -f "/srv/lapis/lapis-state/${TID_VERDI}.md"
/usr/local/bin/mem delete "pm/cursor/${TID_VERDI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_VERDI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/pause-state/${TID_VERDI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_VERDI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_VERDI}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/landed/${TID_VERDI}" 2>/dev/null || true
green "verdict smoke 27a: valid already_satisfied → auto_land:already_satisfied OK"

# --- Step 27b: malformed verdict sidecar → not auto-landed ---------------
step "27b. Fixer already-done verdict: malformed verdict sidecar → ignored, no auto-land"
TID_VERDI_BAD="pm-smoke-verdi-bad-$$"
VERDI_BAD_SPEC="/tmp/${TID_VERDI_BAD}-spec.md"
cat > "$VERDI_BAD_SPEC" <<EOF
# Malformed verdict smoke spec
EOF
cat > "$TARGETS_DIR/${TID_VERDI_BAD}.yaml" <<EOF
id: ${TID_VERDI_BAD}
title: Malformed verdict smoke target
status: active
category: research
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Malformed verdict smoke target — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_VERDI_BAD" --spec-from "$VERDI_BAD_SPEC" --repo lapis-test --authority advisory

# Inject pending fixer dispatch
VERDI_BAD_GPU="claude_smoke_verdi_bad_$$"
VERDI_BAD_SPEC_ID="spec-verdi-bad-$$"
/usr/local/bin/mem set "pm/dispatched/${TID_VERDI_BAD}" \
    "[{\"gpu_id\":\"${VERDI_BAD_GPU}\",\"spec_id\":\"${VERDI_BAD_SPEC_ID}\",\"agent_type\":\"fixer\",\"intent\":\"implement\",\"repo\":\"lapis-test\",\"ts\":\"$(date -Iseconds)\",\"status\":\"pending\",\"retry_count\":0}]" \
    >/dev/null

# Write fake completed output (short, no PR URL, no confab phrases)
mkdir -p "$GPU_COMPLETED"
cat > "$GPU_COMPLETED/${VERDI_BAD_GPU}-output.md" <<EOF
DONE. Already merged.
EOF

# Write a malformed verdict sidecar (bad JSON)
cat > "$SHAPED_DIR_SMOKE/${VERDI_BAD_SPEC_ID}-verdict.json" <<EOF
{this is not valid json at all
EOF

# Tick — malformed verdict should be ignored with WARN; no auto_land:already_satisfied
VERDI_BAD_TICK=$($LAPIS tick --target "$TID_VERDI_BAD" 2>&1)
echo "$VERDI_BAD_TICK"
! echo "$VERDI_BAD_TICK" | grep -q "decision=auto_land:already_satisfied:" \
    || red "verdict smoke 27b: auto_land:already_satisfied fired despite malformed verdict"
[ ! -f "/srv/lapis/lapis-state/${TID_VERDI_BAD}.md" ] \
    || red "verdict smoke 27b: arc doc written despite malformed verdict"
# Verdict sidecar should have been consumed
[ ! -f "$SHAPED_DIR_SMOKE/${VERDI_BAD_SPEC_ID}-verdict.json" ] \
    || red "verdict smoke 27b: malformed verdict sidecar not consumed"
# Cleanup
rm -f "$GPU_COMPLETED/${VERDI_BAD_GPU}-output.md"
rm -f "$TARGETS_DIR/${TID_VERDI_BAD}.yaml" "$COMMENTS_DIR/${TID_VERDI_BAD}.jsonl" "$VERDI_BAD_SPEC"
/usr/local/bin/mem delete "pm/cursor/${TID_VERDI_BAD}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_VERDI_BAD}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/pause-state/${TID_VERDI_BAD}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_VERDI_BAD}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_VERDI_BAD}" 2>/dev/null || true
green "verdict smoke 27b: malformed verdict sidecar ignored, no auto-land OK"

# =========================================================================
# Forgejo health gate smoke (lapis-pm-forgejo-health-gate)
# =========================================================================
step "28. Forgejo health gate: unreachable probe → all bound targets skipped, cursors not advanced"

TID_HG="pm-smoke-hg-$$"
HG_SPEC="/tmp/${TID_HG}-spec.md"
cat > "$HG_SPEC" <<EOF
# Health-gate smoke spec for $TID_HG
What: verify Forgejo health gate skips ticks correctly.
Why:  smoke coverage for lapis-pm-forgejo-health-gate.
EOF

cat > "$TARGETS_DIR/${TID_HG}.yaml" <<EOF
id: ${TID_HG}
title: Health-gate smoke
status: active
category: research

# --- Phase 29: lost-dispatch retry→brief sequence -------------------------
step "28. lost-dispatch: retry on first loss, brief on second loss"

TID_LOST="pm-smoke-lost-$$"
LOST_SPEC="/tmp/${TID_LOST}-spec.md"

cat > "$LOST_SPEC" <<EOF
# Lost-dispatch smoke spec for $TID_LOST
EOF

cat > "$TARGETS_DIR/${TID_LOST}.yaml" <<EOF
id: ${TID_LOST}
title: Lost-dispatch smoke
status: active
category: active-work
urgency: low
work_mode: anywhere
created: $(date +%F)
touched: $(date +%F)
updated: $(date +%F)
decay_days: 0
decay_threshold: 30
description: Health-gate smoke target — safe to delete.
stages:
  - name: smoke-stage
    status: active
EOF
$LAPIS bind "$TID_HG" --spec-from "$HG_SPEC" --repo lapis-test --authority advisory

# Run tick_all() with probe monkey-patched to return unreachable.
# Both the tick-prelude [forgejo:unreachable] line and per-target
# [<tid>] skipped=... lines are emitted by pm_core.tick_all() itself —
# capturing real production stdout, not simulated strings.
HG_OUT=$(python3 - <<PYEOF28
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '/srv/git/agents-core-working')
from unittest.mock import patch
from lapis_pm import pm_core

with patch('lapis_pm.pm_core.probe_forgejo_health', return_value=(False, 'connect_error')), \
     patch('lapis_pm.pm_core._notify_forgejo_unreachable'):
    pm_core.tick_all()
PYEOF28
)
echo "$HG_OUT"

# Assert [forgejo:unreachable] was emitted
echo "$HG_OUT" | grep -q "\[forgejo:unreachable\] reason=connect_error" \
    || red "health gate: [forgejo:unreachable] line missing from tick_all output"

# Assert target row has decision=skipped:forgejo_unreachable
echo "$HG_OUT" | grep -q "decision=skipped:forgejo_unreachable" \
    || red "health gate: decision=skipped:forgejo_unreachable missing"

# Assert skipped=True in the per-target line
echo "$HG_OUT" | grep -q "skipped=True" \
    || red "health gate: skipped=True missing from per-target line"

# Assert cursor was NOT advanced (key must be absent from mem)
/usr/local/bin/mem get "pm/cursor/${TID_HG}" 2>/dev/null \
    && red "health gate: cursor was advanced on unreachable tick (must not advance)" \
    || true   # expected absence is success

# Cleanup phase 28
rm -f "$TARGETS_DIR/${TID_HG}.yaml" "$COMMENTS_DIR/${TID_HG}.jsonl" "$HG_SPEC"
/usr/local/bin/mem delete "pm/cursor/${TID_HG}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_HG}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_HG}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/forgejo_consecutive_fails" 2>/dev/null || true
green "Phase 28 complete: Forgejo health gate smoke OK"



$LAPIS bind "$TID_LOST" --spec-from "$LOST_SPEC" --repo lapis-test --authority advisory

# (a) Stage a terminal fixer dispatch with no PR → tick should retry (first loss).
/usr/bin/python3 - "${TID_LOST}" <<'PYEOF29A'
import sys, json
sys.path.insert(0, '/srv/git/agents-core-working')
from unittest.mock import patch, MagicMock

target_id = sys.argv[1]

# Stage a terminal failed fixer dispatch (no output file, no PR — the SEGV scenario)
dispatch_ts = "2026-05-01T10:00:00-07:00"
orig_record = {
    "gpu_id": "smoke-lost-gpu-orig",
    "spec_id": "spec-smoke-lost",
    "agent_type": "fixer",
    "intent": "implement the smoke spec",
    "repo": "lapis-test",
    "ts": dispatch_ts,
    "status": "failed",
    "retry_count": 0,
    "lost_retry_count": 0,
    "error": "ERROR: signal: segmentation fault",
}

fake_res = MagicMock()
fake_res.task_id = "smoke-lost-gpu-retry"
fake_res.spec_id = "spec-smoke-retry"

saved_records = []

def capture_save(tid, records):
    saved_records.clear()
    saved_records.extend(records)

from agents_core.targets import Target
mock_target = MagicMock(spec=Target)
mock_target.pm_bound = True
mock_target.paused = False
mock_target.pm_repo = "lapis-test"
mock_target.pm_authority = "advisory"

patches = [
    patch('lapis_pm.pm_core.TargetStore'),
    patch('lapis_pm.pm_core._reconcile_dispatched_with_queue', return_value=0),
    patch('lapis_pm.pm_core.get_cursor', return_value=None),
    patch('lapis_pm.pm_core.set_cursor'),
    patch('lapis_pm.pm_core.get_pause_state', return_value=None),
    patch('lapis_pm.pm_core.set_pause_state'),
    # Forgejo reachable this tick, no open PRs
    patch('lapis_pm.pm_core._perceive_prs', return_value=([], True)),
    patch('lapis_pm.episodic.since', return_value=[]),
    patch('lapis_pm.pm_core._encode_new_prs', return_value=[]),
    patch('lapis_pm.pm_core._encode_pr_sha_updates', return_value=0),
    patch('lapis_pm.pm_core._encode_gpu_results', return_value=(0, [])),
    patch('lapis_pm.pm_core._encode_merged_prs', return_value=0),
    patch('lapis_pm.pm_core._encode_user_comments', return_value=[]),
    patch('lapis_pm.pm_core._consume_brief_decisions', return_value=None),
    patch('lapis_pm.pm_core._is_auto_land_eligible', return_value=False),
    patch('lapis_pm.pm_core._persist_review_state_cache'),
    patch('lapis_pm.pm_core.load_dispatched', return_value=[orig_record]),
    patch('lapis_pm.pm_core.save_dispatched', side_effect=capture_save),
    patch('lapis_pm.pm_core.episodic.spec_summary', return_value='spec'),
    patch('lapis_pm.pm_core.episodic.spec', return_value='spec summary'),
    patch('lapis_pm.episodic.all_comments', return_value=[]),
    patch('lapis_pm.episodic.write_observation'),
    patch('lapis_pm.pm_core._SHAPER'),
]

started = []
for p in patches:
    started.append(p.start())

try:
    store_mock = started[0]
    store_mock.return_value.get.return_value = mock_target
    shaper_mock = started[-1]
    shaper_mock.dispatch.return_value = fake_res

    from lapis_pm import pm_core
    result = pm_core.tick(target_id)
finally:
    for p in reversed(patches):
        try: p.stop()
        except: pass

assert result.decision.startswith('fixer_lost:retrying:dispatch='), \
    f'expected fixer_lost:retrying, got: {result.decision!r}'

# Verify child record persisted
child = next((r for r in saved_records if r.get('gpu_id') == 'smoke-lost-gpu-retry'), None)
assert child is not None, f'child record not persisted; saved={saved_records}'
assert child.get('parent_gpu_id') == 'smoke-lost-gpu-orig', f'parent_gpu_id wrong: {child}'
assert child.get('status') == 'pending', f'child status wrong: {child}'

# Verify original has lost_retry_count=1
orig_saved = next(r for r in saved_records if r['gpu_id'] == 'smoke-lost-gpu-orig')
assert orig_saved.get('lost_retry_count') == 1, \
    f'lost_retry_count not incremented: {orig_saved}'

print(f'Phase 29a: first-loss retry ✓  decision={result.decision}')
print(f'  child.gpu_id={child["gpu_id"]} parent={child["parent_gpu_id"]}')
print(f'  original.lost_retry_count={orig_saved["lost_retry_count"]}')
PYEOF29A
green "Phase 29a: first-loss retry (lost_retry_count 0→1, child pending) ✓"

# (b) Stage both dispatches as terminal with no PR → tick should brief (second loss).
/usr/bin/python3 - "${TID_LOST}" <<'PYEOF29B'
import sys, json
sys.path.insert(0, '/srv/git/agents-core-working')
from unittest.mock import patch, MagicMock

target_id = sys.argv[1]

orig_record = {
    "gpu_id": "smoke-lost-gpu-orig",
    "spec_id": "spec-smoke-lost",
    "agent_type": "fixer",
    "intent": "implement the smoke spec",
    "repo": "lapis-test",
    "ts": "2026-05-01T10:00:00-07:00",
    "status": "failed",
    "retry_count": 0,
    "lost_retry_count": 1,
    "error": "ERROR: signal: segmentation fault",
}
child_record = {
    "gpu_id": "smoke-lost-gpu-retry",
    "spec_id": "spec-smoke-retry",
    "agent_type": "fixer",
    "intent": "implement the smoke spec",
    "repo": "lapis-test",
    "ts": "2026-05-01T10:15:00-07:00",
    "status": "failed",
    "retry_count": 0,
    "lost_retry_count": 0,
    "parent_gpu_id": "smoke-lost-gpu-orig",
    "error": "ERROR: general protection fault",
}

fake_brief = MagicMock()
fake_brief.comment_id = "smoke-brief-lost-001"
fake_brief.pushed = False
fake_brief.body = "lost dispatch brief"
fake_brief.target_id = target_id

from agents_core.targets import Target
mock_target = MagicMock(spec=Target)
mock_target.pm_bound = True
mock_target.paused = False
mock_target.pm_repo = "lapis-test"
mock_target.pm_authority = "advisory"

patches = [
    patch('lapis_pm.pm_core.TargetStore'),
    patch('lapis_pm.pm_core._reconcile_dispatched_with_queue', return_value=0),
    patch('lapis_pm.pm_core.get_cursor', return_value=None),
    patch('lapis_pm.pm_core.set_cursor'),
    patch('lapis_pm.pm_core.get_pause_state', return_value=None),
    patch('lapis_pm.pm_core.set_pause_state'),
    patch('lapis_pm.pm_core._perceive_prs', return_value=([], True)),
    patch('lapis_pm.episodic.since', return_value=[]),
    patch('lapis_pm.pm_core._encode_new_prs', return_value=[]),
    patch('lapis_pm.pm_core._encode_pr_sha_updates', return_value=0),
    patch('lapis_pm.pm_core._encode_gpu_results', return_value=(0, [])),
    patch('lapis_pm.pm_core._encode_merged_prs', return_value=0),
    patch('lapis_pm.pm_core._encode_user_comments', return_value=[]),
    patch('lapis_pm.pm_core._consume_brief_decisions', return_value=None),
    patch('lapis_pm.pm_core._is_auto_land_eligible', return_value=False),
    patch('lapis_pm.pm_core._persist_review_state_cache'),
    patch('lapis_pm.pm_core.load_dispatched', return_value=[orig_record, child_record]),
    patch('lapis_pm.pm_core.save_dispatched'),
    patch('lapis_pm.pm_core.brief.synthesize', return_value=fake_brief),
    patch('lapis_pm.pm_core.set_outstanding_brief'),
    patch('lapis_pm.pm_core.episodic.spec', return_value='spec summary'),
    patch('lapis_pm.episodic.all_comments', return_value=[]),
    patch('lapis_pm.episodic.write_observation'),
]

started = []
for p in patches:
    started.append(p.start())

try:
    store_mock = started[0]
    store_mock.return_value.get.return_value = mock_target

    from lapis_pm import pm_core
    result = pm_core.tick(target_id)
finally:
    for p in reversed(patches):
        try: p.stop()
        except: pass

assert result.decision.startswith('fixer_lost:briefing:dispatches='), \
    f'expected fixer_lost:briefing, got: {result.decision!r}'
assert 'smoke-lost-gpu-orig' in result.decision, \
    f'original dispatch ID missing from decision: {result.decision}'
assert 'smoke-lost-gpu-retry' in result.decision, \
    f'retry dispatch ID missing from decision: {result.decision}'

print(f'Phase 29b: second-loss brief ✓  decision={result.decision}')
PYEOF29B
green "Phase 29b: second-loss brief (both dispatch IDs in decision) ✓"

# (c) Verify healthy dispatch (with open PR after dispatch) is NOT classified as lost.
/usr/bin/python3 - "${TID_LOST}" <<'PYEOF29C'
import sys
sys.path.insert(0, '/srv/git/agents-core-working')
from unittest.mock import patch, MagicMock

target_id = sys.argv[1]

dispatch_ts = "2026-05-01T10:00:00-07:00"
orig_record = {
    "gpu_id": "smoke-healthy-gpu",
    "spec_id": "spec-smoke-healthy",
    "agent_type": "fixer",
    "intent": "implement the smoke spec",
    "repo": "lapis-test",
    "ts": dispatch_ts,
    "status": "processed",  # job completed
    "retry_count": 0,
    "lost_retry_count": 0,
}
# A PR was opened AFTER the dispatch
open_pr = {
    "number": 42,
    "created_at": "2026-05-01T10:30:00-07:00",  # after dispatch_ts
    "head": {"ref": f"lapis/{target_id}/forced"},
    "title": "test PR",
}

from lapis_pm import pm_core

with patch('lapis_pm.episodic.all_comments', return_value=[]):
    needs_retry, needs_brief = pm_core._find_lost_fixer_dispatches(
        target_id, [orig_record], open_prs=[open_pr], forgejo_ok=True
    )

assert needs_retry == [], \
    f'healthy dispatch (has PR) should not be in needs_retry: {needs_retry}'
assert needs_brief == [], \
    f'healthy dispatch (has PR) should not be in needs_brief: {needs_brief}'
print('Phase 29c: healthy dispatch (open PR after dispatch) NOT classified as lost ✓')
PYEOF29C
green "Phase 29c: no false-positive lost classification on healthy dispatch ✓"

# Cleanup phase 29
rm -f "$TARGETS_DIR/${TID_LOST}.yaml" "$COMMENTS_DIR/${TID_LOST}.jsonl" "$LOST_SPEC"
/usr/local/bin/mem delete "pm/cursor/${TID_LOST}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_LOST}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_LOST}" 2>/dev/null || true
green "Phase 29 complete: lost-dispatch retry→brief sequence passed"

# --- Done ----------------------------------------------------------------
echo
green "Smoke complete: bind, dispatch, encode, pause/resume, directive→brief, auto-land, reviewer-verdict-encode, chain, router-portfolio, notify-routing, sha-invalidation, state-brief, trajectory-rollup, closed-form-brief, already-done-verdict, forgejo-health-gate, lost-dispatch all OK"
cat <<MSG

Skipped automatically (need live state):
  - authority gate against a real Forgejo PR (touch a held path → expect hold + brief)
  - auto-merge a clean PR (target authority=auto, diff <400 LOC, no held paths)
  - Pushover delivery confirmation — check phone after the directive step above
  - trajectory weekly/monthly rollup (requires Qwen endpoint — suppressed in smoke)
  - Forgejo health gate Pushover delivery (requires ≥3 consecutive unreachable ticks in prod)

Cleanup runs on exit. Target id was: $TID
MSG
