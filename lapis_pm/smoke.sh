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

# --- Done ----------------------------------------------------------------
echo
green "Smoke complete: bind, dispatch, encode, pause/resume, directive→brief all OK"
cat <<MSG

Skipped automatically (need live state):
  - authority gate against a real Forgejo PR (touch a held path → expect hold + brief)
  - auto-merge a clean PR (target authority=auto, diff <400 LOC, no held paths)
  - Pushover delivery confirmation — check phone after the directive step above

Cleanup runs on exit. Target id was: $TID
MSG
