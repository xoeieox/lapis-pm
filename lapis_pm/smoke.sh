#!/usr/bin/env bash
# Lapis PM end-to-end smoke test (no real Forgejo PRs, no Pushover assertion).
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

TID="pm-smoke-$$"
TARGETS_DIR="${PM_TARGETS_DIR:-/srv/lapis/targets}"
COMMENTS_DIR="/srv/lapis/targets/comments"
GPU_COMPLETED="/srv/lapis/gpu-queue/completed"
GPU_PENDING="/srv/lapis/gpu-queue/pending"
SPEC_FILE="/tmp/${TID}-spec.md"
LAPIS="/usr/bin/python3 -m lapis_pm.cli"

green() { printf '\033[32m✓ %s\033[0m\n' "$*"; }
red()   { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
step()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

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
}
trap cleanup EXIT

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
grep -q "directive_brief" /tmp/${TID}-tickout || red "tick did not produce a directive brief"
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
echo "$TICK_OUT" | grep -q "decision=auto_land:" || red "auto-land did not fire in positive case"
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
! echo "$TICK_OUT2" | grep -q "decision=auto_land:" || red "auto-land fired despite pending dispatch"
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
echo "$TICK1" | grep -q "decision=auto_land:" || red "first tick: auto-land did not fire"
[ -f "/srv/lapis/lapis-state/${TID_LAND4}.md" ] || red "arc doc missing after first tick"
ARC_MTIME=$(stat -c %Y "/srv/lapis/lapis-state/${TID_LAND4}.md")
# Second tick → target unbound, auto-land does NOT fire again
TICK2=$($LAPIS tick --target "$TID_LAND4")
echo "$TICK2"
echo "$TICK2" | grep -q "skipped=True" || red "second tick should skip (target unbound)"
! echo "$TICK2" | grep -q "decision=auto_land:" || red "auto-land fired twice (idempotency violated)"
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
! echo "$TICK_MULTI1" | grep -q "decision=auto_land:" || red "multi-PR: auto-land should NOT fire after 1/2 merged"
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
echo "$TICK_MULTI3" | grep -q "decision=auto_land:" || red "multi-PR: auto-land did not fire after 2/2 merged"
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

# Cleanup
rm -f "$REV_QUEUE_DIR/${REV_GPU_ID}.yaml" "$GPU_COMPLETED/${REV_GPU_ID}-output.md"
rm -f "$TARGETS_DIR/${TID_REV}.yaml" "$COMMENTS_DIR/${TID_REV}.jsonl" "$REV_SPEC"
/usr/local/bin/mem delete "pm/cursor/${TID_REV}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/dispatched/${TID_REV}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/pause-state/${TID_REV}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/outstanding-brief/${TID_REV}" 2>/dev/null || true
/usr/local/bin/mem delete "pm/classified-prs/${TID_REV}" 2>/dev/null || true
green "reviewer dispatch produces verdict=fixable episodic entry (not verdict=pending) OK"

# --- Chain smoke phase (step 18) ----------------------------------------
step "18. Chain: 3-leg bind → simulated land → auto-dispatch cascade → state.complete"

CHAIN_GROUP="pm-smoke-chain-$$"
TID_L1="pm-smoke-chain-l1-$$"
TID_L2="pm-smoke-chain-l2-$$"
TID_L3="pm-smoke-chain-l3-$$"
CHAIN_SPEC="/tmp/${CHAIN_GROUP}-spec.md"
CHAIN_LEGS="/tmp/${CHAIN_GROUP}-legs.yaml"

chain_cleanup() {
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

# --- Done ----------------------------------------------------------------
echo
green "Smoke complete: bind, dispatch, encode, pause/resume, directive→brief, auto-land, reviewer-verdict-encode, chain all OK"
cat <<MSG

Skipped automatically (need live state):
  - authority gate against a real Forgejo PR (touch a held path → expect hold + brief)
  - auto-merge a clean PR (target authority=auto, diff <400 LOC, no held paths)
  - Pushover delivery confirmation — check phone after the directive step above

Cleanup runs on exit. Target id was: $TID
MSG
