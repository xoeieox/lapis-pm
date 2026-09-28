#!/usr/bin/env bash
# Parallel-dispatch smoke test for ClaudeQueue + worktree isolation.
#
# This is the load-bearing gate for the 2026-04-23 race fix: two fixer
# dispatches running concurrently must each land their own commits on their
# own branches without interleaving via the shared /srv/git/<repo>-working/
# tree. Pre-fix (run without worktree isolation), running two fixers in
# parallel on 2026-04-22 produced PR #15 (empty) and PR #16 (both components'
# work mashed into two commits).
#
# Prerequisites (assumed present on the host):
#   - claude-queue-runner running against /tmp/lapis-pm-parallel-test-repo/
#   - CLAUDE_QUEUE_WORKERS >= 2 in the runner env
#   - PUSHOVER_* env may be unset; notify is best-effort
#   - a working `claude` CLI with API credentials
#
# Exit codes:
#   0 — smoke passed (both branches have correct disjoint edits)
#   1 — assertion failure (see stderr for which check tripped)
#   2 — setup/teardown error

set -euo pipefail

TEST_REPO=/tmp/lapis-pm-parallel-test-repo
TEST_BARE=/tmp/lapis-pm-parallel-test-repo.bare
CLONE_SYMLINK=/srv/git/lapis-pm-parallel-test-working
TID_A="parallel-smoke-A-$$"
TID_B="parallel-smoke-B-$$"
WORKTREE_ROOT=/tmp/lapis-pm-worktrees
POLL_TIMEOUT=600   # 10min

green() { printf '\033[32mo %s\033[0m\n' "$*"; }
red()   { printf '\033[31mx %s\033[0m\n' "$*" >&2; exit 1; }
step()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

cleanup() {
    set +e
    python3 - <<PY
from agents_core.claude_queue import ClaudeQueue
q = ClaudeQueue()
q.cancel("$TID_A"); q.cancel("$TID_B")
PY
    rm -rf "$TEST_REPO" "$TEST_BARE"
    rm -f "$CLONE_SYMLINK"
}
trap cleanup EXIT INT TERM

# --- 1. Set up throwaway repo -------------------------------------------
step "1. Set up throwaway repo at $TEST_REPO"
rm -rf "$TEST_REPO" "$TEST_BARE"
git init --bare -b main "$TEST_BARE" >/dev/null
git clone "$TEST_BARE" "$TEST_REPO" >/dev/null 2>&1

cd "$TEST_REPO"
cat > AGENTS.md <<'EOF'
# Parallel smoke sandbox
Minimal AGENTS.md so setup_worktree's tripwire passes.
EOF
mkdir -p .claude
cat > .claude/settings.json <<'EOF'
{"hooks": {}}
EOF
echo "original content in a" > a.md
echo "original content in b" > b.md
git add -A
git -c user.email=smoke@test -c user.name=smoke commit -m "initial" >/dev/null
git push origin main >/dev/null 2>&1
cd - >/dev/null

# Symlink so Shaper.resolve_repo_cwd("lapis-pm-parallel-test") finds the clone.
# Pre-flight guard: refuse to clobber a real working-clone by that name.
if [ -e "$CLONE_SYMLINK" ] && [ ! -L "$CLONE_SYMLINK" ]; then
    red "$CLONE_SYMLINK exists and is not a symlink — refusing to clobber"
fi
ln -sfn "$TEST_REPO" "$CLONE_SYMLINK"

# --- 2. Verify daemon is up ---------------------------------------------
step "2. Verify claude-queue-runner is active"
systemctl is-active --quiet claude-queue-runner \
    || red "claude-queue-runner.service is not active"

# --- 3. Dispatch two fixers back-to-back --------------------------------
step "3. Dispatch fixer for $TID_A and $TID_B"

# Create sandbox targets bound to the throwaway repo
export PM_TARGETS_DIR=/tmp/lapis-pm-parallel-targets
mkdir -p "$PM_TARGETS_DIR"

for TID in "$TID_A" "$TID_B"; do
    FILE=$([ "$TID" = "$TID_A" ] && echo "a.md" || echo "b.md")
    WORD=$([ "$TID" = "$TID_A" ] && echo "hello" || echo "world")
    cat > "$PM_TARGETS_DIR/${TID}.yaml" <<EOF
target_id: $TID
repo: lapis-pm-parallel-test
status: active
stages:
  - name: stage-one
    intent: "edit ${FILE} to say ${WORD}"
tags: [test]
EOF
done

python3 - <<PY || red "dispatch script failed"
from pathlib import Path
from agents_core.shaper import Shaper
_shaper = Shaper(Path("/srv/lapis/lapis-pm/lapis_pm/registry.yaml"))
_shaper.dispatch("fixer", "$TID_A",
         "edit a.md so its sole content is the word 'hello'. "
         "Do NOT open a Forgejo PR — the test repo is not on the Forgejo server. "
         "Stop after \`git push\`.",
         vars_={
             "repo": "lapis-pm-parallel-test",
             "target_id": "$TID_A",
             "spec_summary": "Parallel-dispatch smoke test. Edit a.md to contain the single word 'hello'.",
             "slug": "smoke-a",
         })
_shaper.dispatch("fixer", "$TID_B",
         "edit b.md so its sole content is the word 'world'. "
         "Do NOT open a Forgejo PR — the test repo is not on the Forgejo server. "
         "Stop after \`git push\`.",
         vars_={
             "repo": "lapis-pm-parallel-test",
             "target_id": "$TID_B",
             "spec_summary": "Parallel-dispatch smoke test. Edit b.md to contain the single word 'world'.",
             "slug": "smoke-b",
         })
print("dispatched")
PY

green "both dispatches submitted"

# --- 4. Poll until both complete ----------------------------------------
step "4. Poll ClaudeQueue status until both tasks finish (max ${POLL_TIMEOUT}s)"
deadline=$(($(date +%s) + POLL_TIMEOUT))
while true; do
    remaining=$(python3 - <<'PY'
from agents_core.claude_queue import ClaudeQueue
st = ClaudeQueue().status()
# Rough: pending depth + in-flight count. Good enough as a gate.
print(st["depth"] + len(st["in_flight"]))
PY
)
    if [ "$remaining" = "0" ]; then
        break
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        red "timeout waiting for tasks to drain (remaining=$remaining)"
    fi
    sleep 10
done
green "both tasks drained"

# --- 5. Assertions ------------------------------------------------------
step "5. Assert: clean worktrees, disjoint branches, no shared commits"

# 5a. No leftover worktrees in the test repo
cd "$TEST_REPO"
wt_count=$(git worktree list | wc -l)
[ "$wt_count" = "1" ] || red "leftover worktrees: $(git worktree list)"
green "git worktree list shows only the main clone"

# 5b. /tmp/lapis-pm-worktrees is empty (aside from other concurrent tests)
if [ -d "$WORKTREE_ROOT" ]; then
    leftover=$(find "$WORKTREE_ROOT" -maxdepth 1 -mindepth 1 -type d \
        \( -name "*-fixer-${TID_A}" -o -name "*-fixer-${TID_B}" \) | wc -l)
    [ "$leftover" = "0" ] \
        || red "leftover worktree dirs in $WORKTREE_ROOT for this smoke"
fi
green "no smoke-owned leftovers in $WORKTREE_ROOT"

# 5c. Fetch pushed branches
git fetch origin >/dev/null 2>&1

BRANCH_A=$(git branch -r | grep -E "origin/lapis/${TID_A}/" | head -1 | xargs || true)
BRANCH_B=$(git branch -r | grep -E "origin/lapis/${TID_B}/" | head -1 | xargs || true)
[ -n "$BRANCH_A" ] || red "no branch matched origin/lapis/${TID_A}/*"
[ -n "$BRANCH_B" ] || red "no branch matched origin/lapis/${TID_B}/*"
green "branches: A=$BRANCH_A  B=$BRANCH_B"

# 5d. Branch A touched only a.md (and maybe meta commit messages, but not b.md)
FILES_A=$(git log --name-only --pretty=format: "origin/main..${BRANCH_A}" \
    | grep -v '^$' | sort -u)
FILES_B=$(git log --name-only --pretty=format: "origin/main..${BRANCH_B}" \
    | grep -v '^$' | sort -u)

echo "$FILES_A" | grep -qx "a.md" || red "branch A did not touch a.md (got: $FILES_A)"
echo "$FILES_A" | grep -qx "b.md" && red "branch A touched b.md — race leaked (files: $FILES_A)"
echo "$FILES_B" | grep -qx "b.md" || red "branch B did not touch b.md (got: $FILES_B)"
echo "$FILES_B" | grep -qx "a.md" && red "branch B touched a.md — race leaked (files: $FILES_B)"
green "branch A touched only a.md; branch B touched only b.md"

# 5e. No shared commits between the two branches beyond main fork point
SHARED=$(git log --format=%H "$BRANCH_A" "$BRANCH_B" --not origin/main 2>/dev/null \
    | sort | uniq -d | wc -l)
[ "$SHARED" = "0" ] || red "branches share $SHARED commit(s) beyond main — race leaked"
green "no shared commits between branches beyond main"

cd - >/dev/null

step "Smoke passed: parallel dispatch with worktree isolation works"
