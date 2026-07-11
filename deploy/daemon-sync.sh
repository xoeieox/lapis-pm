#!/usr/bin/env bash
# deploy/daemon-sync.sh — idempotent deploy sync for long-running daemon repos
# Usage: deploy/daemon-sync.sh [--dry-run] [--help]
#
# Brings each repo in daemon-manifest.yaml to origin/main, restarts its
# services when HEAD advances, and surfaces anomalies to Active Work.md.
# Supports both service-mode (weaver) and copy-mode (conductor) deployments.
set -euo pipefail

# Track temporary files for cleanup
CLEANUP_FILES=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/daemon-manifest.yaml}"
DEPLOY_LOG="/srv/lapis/lapis-state/lapis-pm-deploy-log.md"
ACTIVE_WORK="/srv/git/inertia-vault-working/Active Work.md"
FORGEJO_BASE="http://203.0.113.10:3000"
FORGEJO_TIMEOUT=10
FORGEJO_LAST_OK_FILE="/tmp/daemon-sync-forgejo-last-ok"
# Minimum interval between consecutive restarts of a given unit (fixed, not
# adaptive — Erah-decided 2026-07-11). Override in tests to avoid real sleeps.
DAEMON_SYNC_RESTART_HYSTERESIS_SEC="${DAEMON_SYNC_RESTART_HYSTERESIS_SEC:-60}"
DRY_RUN=0

cleanup() {
    for f in "${CLEANUP_FILES[@]}"; do
        [[ -f "$f" ]] && rm -f "$f"
    done
}

trap cleanup EXIT

usage() {
    cat <<'EOF'
Usage: deploy/daemon-sync.sh [--dry-run] [--help]

Idempotent deploy sync for long-running daemon repos (service and copy modes).
Reads deploy/daemon-manifest.yaml; for each repo:
  1. Manifest drift check: verify all listed units are known to systemd
  2. Dirty guard: warn+skip if deploy-source has uncommitted changes.
     Logs changed paths, last-commit time, and open-PR status via Forgejo.
     A Forgejo timeout fails to "anomalous" (never to "clean").
  3. git fetch + merge --ff-only origin/main (aborts loudly on divergence)
  4. For service-mode: pip install -e only if pyproject.toml changed
     For copy-mode: compute transitive import closure and rsync deployed files
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

get_deploy_mode() {
    local repo="$1"
    python3 -c "
import yaml
with open('$MANIFEST') as f:
    m = yaml.safe_load(f)
mode = m['repos']['$repo'].get('deploy_mode', 'service')
print(mode)
"
}

get_services() {
    # Emit TSV: unit \t scope
    # Tolerates missing 'services' key (returns empty for copy-mode repos)
    local repo="$1"
    python3 -c "
import yaml
with open('$MANIFEST') as f:
    m = yaml.safe_load(f)
for s in m['repos']['$repo'].get('services', []):
    print(s['unit'] + '\t' + s['scope'])
"
}

# ── Copy-mode helpers ────────────────────────────────────────────────────────

get_copy_from() {
    local repo="$1"
    python3 -c "
import yaml
with open('$MANIFEST') as f:
    m = yaml.safe_load(f)
copy_from = m['repos']['$repo'].get('copy_from')
if copy_from:
    print(copy_from)
"
}

get_copy_to() {
    local repo="$1"
    python3 -c "
import yaml
with open('$MANIFEST') as f:
    m = yaml.safe_load(f)
copy_to = m['repos']['$repo'].get('copy_to')
if copy_to:
    print(copy_to)
"
}

get_entry_points() {
    local repo="$1"
    python3 -c "
import yaml
with open('$MANIFEST') as f:
    m = yaml.safe_load(f)
for ep in m['repos']['$repo'].get('entry_points', []):
    print(ep)
"
}

get_seed_manifests() {
    local repo="$1"
    python3 -c "
import yaml
with open('$MANIFEST') as f:
    m = yaml.safe_load(f)
for sm in m['repos']['$repo'].get('seed_manifests', []):
    print(sm)
"
}

extract_producer_modules() {
    local manifest_file="$1"
    python3 -c "
import yaml, sys
try:
    with open('$manifest_file') as f:
        m = yaml.safe_load(f)
    for p in m.get('producers', []):
        print(p['module'])
except Exception as e:
    print(f'Error parsing $manifest_file: {e}', file=sys.stderr)
    sys.exit(1)
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

# ── Restart mechanics (single source of truth; service-mode and copy-mode both
#    call restart_service_unit — the actual systemctl + liveness + anomaly code
#    lives here exactly once) ────────────────────────────────────────────────

restart_service_unit() {
    # restart_service_unit <repo> <unit> <scope> <reason>
    # Restart one unit, compare pre/post PID for liveness, surface anomalies.
    local repo="$1" unit="$2" scope="$3" reason="$4"

    local PRE_PID=""
    if [[ "$scope" == "system" ]]; then
        PRE_PID=$(systemctl show "$unit" --property=MainPID 2>/dev/null | cut -d= -f2 || echo "")
    else
        PRE_PID=$(systemctl --user show "$unit" --property=MainPID 2>/dev/null | cut -d= -f2 || echo "")
    fi

    log "Restarting $unit (scope=$scope, reason=$reason, pre-restart PID=${PRE_PID:-unknown})"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        [[ "$scope" == "system" ]] \
            && echo "[dry-run] sudo -n systemctl restart $unit" \
            || echo "[dry-run] systemctl --user restart $unit"
        return 0
    fi

    local RESTART_OK=1
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
        return 1
    fi

    log "Liveness check for $unit"
    sleep 3

    local POST_STATE="unknown" POST_PID=""
    if [[ "$scope" == "system" ]]; then
        POST_STATE=$(systemctl show "$unit" --property=ActiveState 2>/dev/null | cut -d= -f2 || echo "unknown")
        POST_PID=$(systemctl show "$unit" --property=MainPID 2>/dev/null | cut -d= -f2 || echo "")
    else
        POST_STATE=$(systemctl --user show "$unit" --property=ActiveState 2>/dev/null | cut -d= -f2 || echo "unknown")
        POST_PID=$(systemctl --user show "$unit" --property=MainPID 2>/dev/null | cut -d= -f2 || echo "")
    fi

    if [[ "$POST_STATE" != "active" ]]; then
        surface_anomaly "$repo — $unit is not active after restart (state=$POST_STATE). Restart did not take. Manual: $([ "$scope" = "system" ] && echo "sudo systemctl restart $unit && systemctl status $unit" || echo "systemctl --user restart $unit && systemctl --user status $unit")"
        return 1
    elif [[ -n "$PRE_PID" && "$PRE_PID" != "0" && "$POST_PID" == "$PRE_PID" ]]; then
        surface_anomaly "$repo — $unit PID unchanged after restart (PID=$POST_PID). Service may be running old code. Manual: $([ "$scope" = "system" ] && echo "sudo systemctl restart $unit" || echo "systemctl --user restart $unit")"
        return 1
    else
        log "Liveness OK: $unit active (state=$POST_STATE, PID ${PRE_PID:-?}→${POST_PID:-?})"
        return 0
    fi
}

restart_services() {
    # restart_services <repo> <reason> — restart every unit listed in the
    # manifest for <repo>, unconditionally (no hysteresis/env gating — that's
    # copy-mode-only and is checked explicitly at its call site, not here).
    local repo="$1" reason="$2"
    while IFS=$'\t' read -r unit scope; do
        [[ -z "$unit" ]] && continue
        restart_service_unit "$repo" "$unit" "$scope" "$reason"
    done < <(get_services "$repo")
}

restart_marker_path() {
    echo "/tmp/daemon-sync-restart-$1"
}

restart_hysteresis_ok() {
    # Returns 0 (ok to restart) unless a restart of $1 was recorded within
    # DAEMON_SYNC_RESTART_HYSTERESIS_SEC. Missing/unreadable/corrupt marker
    # => assume-safe-and-proceed (worst case is one extra restart).
    local unit="$1"
    local marker
    marker=$(restart_marker_path "$unit")
    local last_restart
    last_restart=$(cat "$marker" 2>/dev/null) || return 0
    [[ "$last_restart" =~ ^[0-9]+$ ]] || return 0
    local now elapsed
    now=$(date +%s)
    elapsed=$(( now - last_restart ))
    [[ "$elapsed" -ge "$DAEMON_SYNC_RESTART_HYSTERESIS_SEC" ]]
}

record_restart_marker() {
    # Atomic write (temp + mv). Best-effort: a write failure is not critical
    # (loses hysteresis history, never blocks a restart).
    local unit="$1"
    local marker
    marker=$(restart_marker_path "$unit")
    local tmp
    tmp=$(mktemp "${marker}.XXXXXX" 2>/dev/null) || return 0
    if date +%s > "$tmp" 2>/dev/null; then
        mv -f "$tmp" "$marker" 2>/dev/null || rm -f "$tmp" 2>/dev/null
    else
        rm -f "$tmp" 2>/dev/null
    fi
}

# ── Step 1: Manifest drift check ──────────────────────────────────────────────
log "Step 1: manifest drift check"
while IFS=$'\t' read -r unit scope; do
    [[ -z "$unit" ]] && continue  # Skip empty lines (copy-mode repos with no services)
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
    REPO_DEPLOY_INCOMPLETE=0

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

    # ── Determine deploy mode and branch ──────────────────────────────────────
    DEPLOY_MODE=$(get_deploy_mode "$repo")
    log "Deploy mode: $DEPLOY_MODE"

    if [[ "$DEPLOY_MODE" == "copy" ]]; then
        # ── COPY MODE ────────────────────────────────────────────────────────
        log "Step 4: copy-mode deploy (transitive closure + rsync)"

        COPY_FROM_REL=$(get_copy_from "$repo")
        COPY_TO=$(get_copy_to "$repo")
        COPY_FROM="${deploy_source}/${COPY_FROM_REL}"

        if [[ -z "$COPY_FROM_REL" || -z "$COPY_TO" ]]; then
            surface_anomaly "$repo — copy-mode entry missing required fields: copy_from=$COPY_FROM_REL, copy_to=$COPY_TO. Check manifest."
            continue
        fi

        if [[ ! -d "$COPY_FROM" ]]; then
            surface_anomaly "$repo — copy_from directory not found: $COPY_FROM (deploy_source may be unmounted or misconfigured)"
            continue
        fi

        # Collect seeds: entry_points + modules from seed_manifests
        SEEDS=()
        while read -r ep; do
            [[ -n "$ep" ]] && SEEDS+=("$ep")
        done < <(get_entry_points "$repo")

        while read -r manifest; do
            [[ -n "$manifest" ]] || continue
            MANIFEST_FILE="${deploy_source}/${COPY_FROM_REL}/${manifest}"
            if [[ ! -f "$MANIFEST_FILE" ]]; then
                surface_anomaly "$repo — seed_manifest not found: $MANIFEST_FILE"
                continue
            fi
            while read -r module; do
                [[ -n "$module" ]] && SEEDS+=("$module")
            done < <(extract_producer_modules "$MANIFEST_FILE")
        done < <(get_seed_manifests "$repo")

        if [[ ${#SEEDS[@]} -eq 0 ]]; then
            surface_anomaly "$repo — no entry_points or seed_manifests configured. Check manifest."
            continue
        fi

        log "Computing import closure for seeds: ${SEEDS[*]}"
        CLOSURE_FILE=$(mktemp)
        CLEANUP_FILES+=("$CLOSURE_FILE")
        if ! python3 "$SCRIPT_DIR/import_closure.py" "$COPY_FROM" "${SEEDS[@]}" > "$CLOSURE_FILE" 2>&1; then
            surface_anomaly "$repo — import_closure failed: $(cat "$CLOSURE_FILE")"
            continue
        fi

        CLOSURE_COUNT=$(wc -l < "$CLOSURE_FILE")
        log "Closure computed: $CLOSURE_COUNT files"

        # Rsync closure + config files
        RSYNC_OUTPUT=$(mktemp)
        CLEANUP_FILES+=("$RSYNC_OUTPUT")
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[dry-run] mkdir -p $COPY_TO"
            echo "[dry-run] rsync -a --no-relative --exclude='__pycache__' --files-from=$CLOSURE_FILE $COPY_FROM/ $COPY_TO/"
            while read -r manifest; do
                [[ -n "$manifest" ]] && echo "[dry-run] cp -p ${deploy_source}/${COPY_FROM_REL}/$manifest $COPY_TO/$manifest"
            done < <(get_seed_manifests "$repo")
        else
            mkdir -p "$COPY_TO"
            if rsync -a --no-relative --exclude='__pycache__' --itemize-changes --files-from="$CLOSURE_FILE" "$COPY_FROM/" "$COPY_TO/" 2>&1 | tee "$RSYNC_OUTPUT"; then
                RSYNC_CHANGES=$(grep -c '>' "$RSYNC_OUTPUT" || true)
                RSYNC_CHANGES="${RSYNC_CHANGES:-0}"
                log "Rsync complete: $RSYNC_CHANGES file(s) changed/added"
            else
                surface_anomaly "$repo — rsync failed for $COPY_FROM to $COPY_TO. Check filesystem permissions and disk space."
                continue
            fi

            while read -r manifest; do
                [[ -n "$manifest" ]] || continue
                SRC="${deploy_source}/${COPY_FROM_REL}/$manifest"
                DST="$COPY_TO/$manifest"
                if [[ -f "$SRC" ]]; then
                    cp -p "$SRC" "$DST"
                    log "Copied config: $manifest"
                fi
            done < <(get_seed_manifests "$repo")
        fi

        # Step 5: Deploy-time smoke-check (import validation)
        log "Step 5: deploy-time import smoke-check"
        if [[ "$DRY_RUN" -eq 0 ]]; then
            SMOKE_ERR_FILE=$(mktemp)
            CLEANUP_FILES+=("$SMOKE_ERR_FILE")
            if PYTHONPATH="$COPY_TO:${PYTHONPATH:-}" python3 - "$COPY_TO" "${SEEDS[@]}" 2>"$SMOKE_ERR_FILE" <<'SMOKE_CHECK'
import sys, importlib
copy_to = sys.argv[1]
seeds = sys.argv[2:]
failed = []
for seed in seeds:
    module_name = seed.replace('.py', '')
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        failed.append(f"Module '{module_name}' not found: {e}")
    except Exception as e:
        failed.append(f"Module '{module_name}' import error: {e}")
if failed:
    for msg in failed:
        print(msg, file=sys.stderr)
    sys.exit(1)
SMOKE_CHECK
            then
                SMOKE_EXIT=0
            else
                SMOKE_EXIT=1
            fi
            if [[ $SMOKE_EXIT -ne 0 ]]; then
                SMOKE_ERR=$(cat "$SMOKE_ERR_FILE" 2>/dev/null | tr '\n' ' ' || echo "unknown error")
                surface_anomaly "$repo — import smoke-check failed. Deployed scripts cannot import dependencies: $SMOKE_ERR"
                continue
            else
                log "Smoke-check OK: all entry points and seed modules importable"
            fi
        fi

        # ── Step 6: copy-mode restart-on-change ───────────────────────────────
        # Gates are explicit here (not hidden in restart_service_unit/restart_services):
        # RSYNC_CHANGES>0, smoke-check passed (a failed smoke-check already `continue`d
        # out of this repo above, so reaching here means it passed), per-unit restart
        # hysteresis, and the --user-scope env prerequisite (skip-and-flag, never halt
        # the run).
        log "Step 6: copy-mode restart-on-change"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            while IFS=$'\t' read -r unit scope; do
                [[ -z "$unit" ]] && continue
                [[ "$scope" == "system" ]] \
                    && echo "[dry-run] sudo -n systemctl restart $unit" \
                    || echo "[dry-run] systemctl --user restart $unit"
            done < <(get_services "$repo")
        elif [[ "${RSYNC_CHANGES:-0}" -eq 0 ]]; then
            log "Rsync changed 0 files — restart skipped"
        else
            while IFS=$'\t' read -r unit scope; do
                [[ -z "$unit" ]] && continue

                if [[ "$scope" == "user" ]] && { [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]] || [[ -z "${XDG_RUNTIME_DIR:-}" ]]; }; then
                    surface_anomaly "$repo — $unit restart SKIPPED: DBUS_SESSION_BUS_ADDRESS/XDG_RUNTIME_DIR not set for a --user scope restart. Files landed in $COPY_TO but the running service is knowingly still on old code (deploy result INCOMPLETE). Self-corrects on the next run once the env is present."
                    REPO_DEPLOY_INCOMPLETE=1
                    continue
                fi

                if ! restart_hysteresis_ok "$unit"; then
                    log "Restart hysteresis: $unit was restarted within the last ${DAEMON_SYNC_RESTART_HYSTERESIS_SEC}s — skipping"
                    continue
                fi

                restart_service_unit "$repo" "$unit" "$scope" "copy-mode restart-on-change ($RSYNC_CHANGES file(s) changed, smoke-check OK)"
                record_restart_marker "$unit"
            done < <(get_services "$repo")
        fi

    else
        # ── SERVICE MODE (default, weaver path unchanged) ─────────────────────
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

        # ── Steps 5+6: restart services + liveness check ─────────────────────
        if [[ "$CODE_CHANGED" -eq 0 ]]; then
            log "Step 5: no new commits — restarts skipped"
        else
            log "Step 5: restarting services for $repo (HEAD advanced $PRE_REV..$POST_REV)"
            restart_services "$repo" "code sync $PRE_REV..$POST_REV"
        fi
    fi

    # ── Deploy log ────────────────────────────────────────────────────────────
    if [[ "$CODE_CHANGED" -eq 1 && "$DRY_RUN" -eq 0 && -f "$DEPLOY_LOG" ]]; then
        LOG_TS=$(now_iso)
        echo "- \`${LOG_TS}\` | ${deploy_source} | synced ${PRE_REV:0:8}..${POST_REV:0:8} | daemon-deploy-timer" >> "$DEPLOY_LOG"
        log "Deploy log updated"
    fi

    DEPLOY_RESULT=$([ "$CODE_CHANGED" -eq 1 ] && echo "synced $PRE_REV..$POST_REV" || echo unchanged)
    if [[ "${REPO_DEPLOY_INCOMPLETE:-0}" -eq 1 ]]; then
        DEPLOY_RESULT="INCOMPLETE ($DEPLOY_RESULT; restart skipped — see anomaly)"
    fi
    log "Summary ($repo): code=$DEPLOY_RESULT"

done < <(get_repos)

log "daemon-sync complete"
