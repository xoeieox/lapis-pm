#!/usr/bin/env bash
# deploy/daemon-sync.sh — idempotent deploy sync for long-running daemon repos
# Usage: deploy/daemon-sync.sh [--dry-run] [--help]
#
# Brings each repo in daemon-manifest.yaml to origin/main, restarts its
# services when HEAD advances, and surfaces anomalies to Active Work.md.
# v0: weaver only (weaver-watcher.service + weaver-server.service).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/daemon-manifest.yaml"
DEPLOY_LOG="/srv/lapis/lapis-state/lapis-pm-deploy-log.md"
ACTIVE_WORK="/srv/git/inertia-vault-working/Active Work.md"
FORGEJO_BASE="http://203.0.113.10:3000"
FORGEJO_TIMEOUT=10
FORGEJO_LAST_OK_FILE="/tmp/daemon-sync-forgejo-last-ok"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: deploy/daemon-sync.sh [--dry-run] [--help]

Idempotent deploy sync for long-running daemon repos (v0: weaver only).
Reads deploy/daemon-manifest.yaml; for each repo:
  1. Manifest drift check: verify all listed units are known to systemd
  2. Dirty guard: warn+skip if deploy-source has uncommitted changes.
     Logs changed paths, last-commit time, and open-PR status via Forgejo.
     A Forgejo timeout fails to "anomalous" (never to "clean").
  3. git fetch + merge --ff-only origin/main (aborts loudly on divergence)
  4. pip install -e only if pyproject.toml changed in pulled range
  5. If HEAD advanced: restart each listed service in correct scope
  6. Liveness check: confirm active + fresh PID; ANOMALOUS if not
  7. Append deploy-log line when HEAD advances
Anomalies (failed pull, failed restart, dirty-without-open-PR, unknown unit)
surface as named next-steps in Active Work.md.

Options:
  --dry-run   Print planned actions, mutate nothing
  --help      Show this help and exit

Assumes FORGEJO_TOKEN is set (from EnvironmentFile conductor.env).
Assumes DBUS_SESSION_BUS_ADDRESS is set for --user scope systemctl calls.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; usage >&2; exit 1 ;;
    esac
done

log()  { echo "[daemon-sync] $*"; }
warn() { echo "[daemon-sync] WARN: $*" >&2; }

sudo_or_warn() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] sudo $*"
        return
    fi
    if ! sudo -n "$@"; then
        warn "sudo -n $* failed — run manually: sudo $*"
    fi
}

now_iso() {
    python3 -c "
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
" 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ
}

# ── Manifest helpers ──────────────────────────────────────────────────────────

get_repos() {
    python3 -c "
import yaml
with open('$MANIFEST') as f:
    m = yaml.safe_load(f)
for repo in m.get('repos', {}).keys():
    print(repo)
"
}

get_deploy_source() {
    local repo="$1"
    python3 -c "
import yaml
with open('$MANIFEST') as f:
    m = yaml.safe_load(f)
print(m['repos']['$repo']['deploy_source'])
"
}

get_services() {
    # Emit TSV: unit \t scope
    local repo="$1"
    python3 -c "
import yaml
with open('$MANIFEST') as f:
    m = yaml.safe_load(f)
for s in m['repos']['$repo']['services']:
    print(s['unit'] + '\t' + s['scope'])
"
}

# ── Anomaly surfacing ─────────────────────────────────────────────────────────

append_anomaly_to_active_work() {
    local entry="$1"
    [[ -f "$ACTIVE_WORK" ]] || { warn "Active Work.md not found at $ACTIVE_WORK — anomaly logged here only"; return; }
    [[ "$DRY_RUN" -eq 1 ]] && { echo "[dry-run] would append to Active Work.md: $entry"; return; }
    python3 - "$ACTIVE_WORK" "$entry" <<'PYEOF'
import sys
path, entry = sys.argv[1], sys.argv[2]
with open(path) as f:
    content = f.read()
# Insert as the first bullet after "### Awaiting your call"
for marker in ("### Awaiting your call\n\n", "### Awaiting your call\n"):
    if marker in content:
        idx = content.index(marker) + len(marker)
        # Skip past any existing blank line
        insert_at = idx if content[idx] != "\n" else idx + 1
        content = content[:insert_at] + entry + "\n" + content[insert_at:]
        break
else:
    content += "\n### Awaiting your call\n\n" + entry + "\n"
with open(path, "w") as f:
    f.write(content)
PYEOF
    log "Anomaly surfaced to Active Work.md"
}

surface_anomaly() {
    local msg="$1"
    local ts
    ts=$(now_iso)
    warn "ANOMALOUS: $msg"
    local entry="- **daemon-deploy ANOMALOUS ($ts):** $msg"
    append_anomaly_to_active_work "$entry"
}

# ── Step 1: Manifest drift check ──────────────────────────────────────────────
log "Step 1: manifest drift check"
while IFS=$'\t' read -r unit scope; do
    if [[ "$scope" == "system" ]]; then
        load_state=$(systemctl show "$unit" --property=LoadState 2>/dev/null | cut -d= -f2 || echo "unknown")
    else
        load_state=$(systemctl --user show "$unit" --property=LoadState 2>/dev/null | cut -d= -f2 || echo "unknown")
    fi
    if [[ "$load_state" != "loaded" ]]; then
        surface_anomaly "manifest unit $unit (scope=$scope) has LoadState='$load_state' — not known to systemd. Verify unit name or install the unit before this entry can be managed."
    else
        log "Manifest drift: $unit (scope=$scope) OK (loaded)"
    fi
done < <(get_repos | while read -r r; do get_services "$r"; done)

# ── Per-repo sync ─────────────────────────────────────────────────────────────
while read -r repo; do
    deploy_source=$(get_deploy_source "$repo")
    log "── Repo: $repo (source: $deploy_source) ──"

    # ── Step 2: Dirty guard with provenance ───────────────────────────────────
    log "Step 2: dirty guard"
    IS_DIRTY=0
    if ! git -C "$deploy_source" diff --quiet 2>/dev/null || \
       ! git -C "$deploy_source" diff --cached --quiet 2>/dev/null; then
        IS_DIRTY=1
    fi

    if [[ "$IS_DIRTY" -eq 1 ]]; then
        DIRTY_PATHS=$(git -C "$deploy_source" diff --name-only HEAD 2>/dev/null | head -20 | tr '\n' ' ')
        LAST_COMMIT=$(git -C "$deploy_source" log -1 --format="%ci %s" 2>/dev/null || echo "unknown")

        # Forgejo open-PR query — timeout fails to "anomalous", never to "clean"
        FORGEJO_RESULT="unknown"
        FORGEJO_ERROR=""
        if [[ -n "${FORGEJO_TOKEN:-}" ]]; then
            PR_RESPONSE=$(curl -s -m "$FORGEJO_TIMEOUT" \
                "$FORGEJO_BASE/api/v1/repos/Erah/$repo/pulls?state=open&limit=50" \
                -H "Authorization: token $FORGEJO_TOKEN" 2>&1) && CURL_EXIT=0 || CURL_EXIT=$?
            if [[ "$CURL_EXIT" -eq 0 ]]; then
                PR_COUNT=$(echo "$PR_RESPONSE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(len(d))
except Exception as e:
    print('err:' + str(e))
" 2>/dev/null || echo "err:python")
                if [[ "$PR_COUNT" =~ ^[0-9]+$ ]]; then
                    echo "$(now_iso)" > "$FORGEJO_LAST_OK_FILE"
                    [[ "$PR_COUNT" -gt 0 ]] && FORGEJO_RESULT="open-pr" || FORGEJO_RESULT="no-open-pr"
                else
                    FORGEJO_ERROR="JSON parse failed: $PR_COUNT"
                    FORGEJO_RESULT="anomalous"
                fi
            else
                FORGEJO_ERROR="curl exit $CURL_EXIT (timeout or connection error)"
                FORGEJO_RESULT="anomalous"
            fi
        else
            FORGEJO_ERROR="FORGEJO_TOKEN not set"
            FORGEJO_RESULT="anomalous"
        fi

        LAST_OK_NOTE=""
        if [[ "$FORGEJO_RESULT" == "anomalous" ]]; then
            if [[ -f "$FORGEJO_LAST_OK_FILE" ]]; then
                LAST_OK_NOTE=" (Forgejo last-ok: $(cat "$FORGEJO_LAST_OK_FILE"))"
            else
                LAST_OK_NOTE=" (Forgejo last-ok: never)"
            fi
        fi

        warn "Deploy source $deploy_source is DIRTY — skipping $repo"
        warn "  Dirty paths: ${DIRTY_PATHS:-none recorded}"
        warn "  Last commit: $LAST_COMMIT"
        warn "  Forgejo PR status: $FORGEJO_RESULT${FORGEJO_ERROR:+ ($FORGEJO_ERROR)}${LAST_OK_NOTE}"

        if [[ "$FORGEJO_RESULT" == "no-open-pr" || "$FORGEJO_RESULT" == "anomalous" ]]; then
            surface_anomaly "$repo deploy-source ($deploy_source) is dirty with no confirmed open PR (Forgejo status=$FORGEJO_RESULT${FORGEJO_ERROR:+, $FORGEJO_ERROR}${LAST_OK_NOTE}). Last commit: $LAST_COMMIT. Dirty paths: ${DIRTY_PATHS:-see deploy log}. Inspect deploy source — dirty tree blocks automatic sync."
        else
            log "Dirty tree with open PR — expected WIP provenance; skip is normal"
        fi
        continue
    fi

    # ── Step 3: fetch + ff-only merge ────────────────────────────────────────
    log "Step 3: code sync"
    PRE_REV=$(git -C "$deploy_source" rev-parse HEAD)

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] git -C $deploy_source fetch origin"
        echo "[dry-run] git -C $deploy_source merge --ff-only origin/main"
        POST_REV="$PRE_REV"
    else
        if ! git -C "$deploy_source" fetch origin 2>&1; then
            surface_anomaly "$repo — git fetch failed for $deploy_source. Repo may be unreachable. Manual: git -C $deploy_source fetch origin"
            log "Skipping $repo after fetch failure"
            continue
        fi
        if ! git -C "$deploy_source" merge --ff-only origin/main 2>&1; then
            surface_anomaly "$repo — git merge --ff-only origin/main failed for $deploy_source (diverged from main?). Manual: git -C $deploy_source merge --ff-only origin/main"
            log "Skipping $repo after merge failure"
            continue
        fi
        POST_REV=$(git -C "$deploy_source" rev-parse HEAD)
    fi

    CODE_CHANGED=0
    if [[ "$PRE_REV" != "$POST_REV" ]]; then
        log "Code: synced $PRE_REV..$POST_REV"
        CODE_CHANGED=1
    else
        log "Code: already up to date ($PRE_REV)"
    fi

    # ── Step 4: conditional reinstall ────────────────────────────────────────
    log "Step 4: checking pyproject.toml for dep/entry-point changes"
    if [[ "$CODE_CHANGED" -eq 1 ]]; then
        if ! git -C "$deploy_source" diff --quiet "${PRE_REV}..${POST_REV}" -- pyproject.toml 2>/dev/null; then
            log "pyproject.toml changed — reinstalling editable package"
            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "[dry-run] pip install -e $deploy_source"
            else
                pip install -e "$deploy_source"
            fi
        else
            log "No dep/entry-point change — reinstall skipped"
        fi
    else
        log "No new commits — reinstall skipped"
    fi

    # ── Steps 5+6: restart services + liveness check ─────────────────────────
    if [[ "$CODE_CHANGED" -eq 0 ]]; then
        log "Step 5: no new commits — restarts skipped"
    else
        log "Step 5: restarting services for $repo (HEAD advanced $PRE_REV..$POST_REV)"
        while IFS=$'\t' read -r unit scope; do
            # Capture pre-restart PID for liveness comparison
            PRE_PID=""
            if [[ "$scope" == "system" ]]; then
                PRE_PID=$(systemctl show "$unit" --property=MainPID 2>/dev/null | cut -d= -f2 || echo "")
            else
                PRE_PID=$(systemctl --user show "$unit" --property=MainPID 2>/dev/null | cut -d= -f2 || echo "")
            fi

            log "Restarting $unit (scope=$scope, pre-restart PID=${PRE_PID:-unknown})"
            if [[ "$DRY_RUN" -eq 1 ]]; then
                [[ "$scope" == "system" ]] \
                    && echo "[dry-run] sudo -n systemctl restart $unit" \
                    || echo "[dry-run] systemctl --user restart $unit"
                continue
            fi

            RESTART_OK=1
            if [[ "$scope" == "system" ]]; then
                sudo_or_warn systemctl restart "$unit" || RESTART_OK=0
            else
                if ! systemctl --user restart "$unit" 2>/dev/null; then
                    warn "systemctl --user restart $unit failed — run manually: systemctl --user restart $unit"
                    RESTART_OK=0
                fi
            fi

            if [[ "$RESTART_OK" -eq 0 ]]; then
                surface_anomaly "$repo — $unit restart command failed (scope=$scope). Service may still be running old code. Manual: $([ "$scope" = "system" ] && echo "sudo systemctl restart $unit" || echo "systemctl --user restart $unit")"
                continue
            fi

            # Step 6: liveness check — verify active + fresh PID
            log "Step 6: liveness check for $unit"
            sleep 3

            POST_STATE="unknown"
            POST_PID=""
            if [[ "$scope" == "system" ]]; then
                POST_STATE=$(systemctl show "$unit" --property=ActiveState 2>/dev/null | cut -d= -f2 || echo "unknown")
                POST_PID=$(systemctl show "$unit" --property=MainPID 2>/dev/null | cut -d= -f2 || echo "")
            else
                POST_STATE=$(systemctl --user show "$unit" --property=ActiveState 2>/dev/null | cut -d= -f2 || echo "unknown")
                POST_PID=$(systemctl --user show "$unit" --property=MainPID 2>/dev/null | cut -d= -f2 || echo "")
            fi

            if [[ "$POST_STATE" != "active" ]]; then
                surface_anomaly "$repo — $unit is not active after restart (state=$POST_STATE). Restart did not take. Manual: $([ "$scope" = "system" ] && echo "sudo systemctl restart $unit && systemctl status $unit" || echo "systemctl --user restart $unit && systemctl --user status $unit")"
            elif [[ -n "$PRE_PID" && "$PRE_PID" != "0" && "$POST_PID" == "$PRE_PID" ]]; then
                surface_anomaly "$repo — $unit PID unchanged after restart (PID=$POST_PID). Service may be running old code. Manual: $([ "$scope" = "system" ] && echo "sudo systemctl restart $unit" || echo "systemctl --user restart $unit")"
            else
                log "Liveness OK: $unit active (state=$POST_STATE, PID ${PRE_PID:-?}→${POST_PID:-?})"
            fi
        done < <(get_services "$repo")
    fi

    # ── Deploy log ────────────────────────────────────────────────────────────
    if [[ "$CODE_CHANGED" -eq 1 && "$DRY_RUN" -eq 0 && -f "$DEPLOY_LOG" ]]; then
        LOG_TS=$(now_iso)
        echo "- \`${LOG_TS}\` | ${deploy_source} | synced ${PRE_REV:0:8}..${POST_REV:0:8} | daemon-deploy-timer" >> "$DEPLOY_LOG"
        log "Deploy log updated"
    fi

    log "Summary ($repo): code=$([ "$CODE_CHANGED" -eq 1 ] && echo "synced $PRE_REV..$POST_REV" || echo unchanged)"

done < <(get_repos)

log "daemon-sync complete"
