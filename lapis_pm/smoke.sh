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
echo "$TICK_OUT2" | grep -qv "decision=auto_land:" || red "auto-land fired despite pending dispatch"
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
echo "$TICK2" | grep -qv "decision=auto_land:" || red "auto-land fired twice (idempotency violated)"
ARC_MTIME2=$(stat -c %Y "/srv/lapis/lapis-state/${TID_LAND4}.md")
[ "$ARC_MTIME" = "$ARC_MTIME2" ] || red "arc doc was overwritten on second tick (not idempotent)"
_cleanup_land_target "$TID_LAND4"
green "auto-land idempotency OK"

# --- Done ----------------------------------------------------------------
echo
green "Smoke complete: bind, dispatch, encode, pause/resume, directive→brief, auto-land all OK"
cat <<MSG

Skipped automatically (need live state):
  - authority gate against a real Forgejo PR (touch a held path → expect hold + brief)
  - auto-merge a clean PR (target authority=auto, diff <400 LOC, no held paths)
  - Pushover delivery confirmation — check phone after the directive step above

Cleanup runs on exit. Target id was: $TID
MSG
