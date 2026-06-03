#!/usr/bin/env bash
# deploy/sync.sh — idempotent BRIX deploy procedure for lapis-pm
# Usage: deploy/sync.sh [--dry-run] [--help]
set -euo pipefail

REPO=/srv/git/lapis-pm
SYSTEMD=/etc/systemd/system
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_UNITS="$SCRIPT_DIR/../systemd"
DRY_RUN=0
CODE_CHANGED=0
NEED_RELOAD=0
UNITS_CHANGED=()

usage() {
    cat <<'EOF'
Usage: deploy/sync.sh [--dry-run] [--help]

Idempotent BRIX deploy for lapis-pm. Steps:
  1. Code sync:  git fetch + merge --ff-only origin/main at /srv/git/lapis-pm
  2. Reinstall:  pip install -e only if pyproject.toml changed in pulled range
  3. Units:      copy systemd/*.{service,timer} + *.service.d/*.conf to /etc/systemd/system/
                 daemon-reload if anything changed; content-compare skips in-sync files
  4. Dependency: warn if /usr/local/bin/wake-and-run is missing (drop-ins need it)
  5. Restart:    systemctl restart for timers whose unit/drop-in actually changed

Options:
  --dry-run   Print planned actions, mutate nothing
  --help      Show this help and exit

Assumes passwordless sudo -n for cp/systemctl/daemon-reload. If sudo is denied,
the exact manual command is logged and the script continues (best-effort).
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; usage >&2; exit 1 ;;
    esac
done

log()  { echo "[sync] $*"; }
warn() { echo "[sync] WARN: $*" >&2; }

sudo_or_warn() {
    # best-effort privileged op; logs manual fallback on failure, never aborts
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] sudo $*"
        return
    fi
    if ! sudo -n "$@"; then
        warn "sudo -n $* failed — run manually: sudo $*"
    fi
}

# ── Step 1: code sync ──────────────────────────────────────────────────────
log "Step 1: code sync at $REPO"
PRE_REV=$(git -C "$REPO" rev-parse HEAD)

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] git -C $REPO fetch origin"
    echo "[dry-run] git -C $REPO merge --ff-only origin/main"
    POST_REV="$PRE_REV"
else
    git -C "$REPO" fetch origin
    git -C "$REPO" merge --ff-only origin/main  # aborts loudly on divergence
    POST_REV=$(git -C "$REPO" rev-parse HEAD)
fi

if [[ "$PRE_REV" != "$POST_REV" ]]; then
    log "Code: synced $PRE_REV..$POST_REV"
    CODE_CHANGED=1
else
    log "Code: already up to date ($PRE_REV)"
fi

# ── Step 2: conditional reinstall ─────────────────────────────────────────
log "Step 2: checking pyproject.toml for dep/entry-point changes"
if [[ "$CODE_CHANGED" -eq 1 ]]; then
    if ! git -C "$REPO" diff --quiet "${PRE_REV}..${POST_REV}" -- pyproject.toml; then
        log "pyproject.toml changed — reinstalling editable package"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[dry-run] pip install -e $REPO"
        else
            pip install -e "$REPO"
        fi
    else
        log "no dep/entry-point change — reinstall skipped"
    fi
else
    log "no new commits — reinstall skipped"
fi

# ── Step 3: units + drop-ins ──────────────────────────────────────────────
log "Step 3: propagating systemd units and drop-ins"

# Copy flat unit files (.service, .timer)
for src in "$SRC_UNITS"/*.service "$SRC_UNITS"/*.timer; do
    [[ -f "$src" ]] || continue
    unit=$(basename "$src")
    dst="$SYSTEMD/$unit"
    if cmp -s "$src" "$dst" 2>/dev/null; then
        log "Unit $unit: in sync"
    else
        log "Unit $unit: differs — copying"
        sudo_or_warn cp "$src" "$dst"
        NEED_RELOAD=1
        if [[ "$unit" == *.timer ]]; then
            UNITS_CHANGED+=("$unit")
        else
            timer="${unit%.service}.timer"
            [[ -f "$SRC_UNITS/$timer" ]] && UNITS_CHANGED+=("$timer")
        fi
    fi
done

# Copy drop-in overrides (*.service.d/*.conf)
for src in "$SRC_UNITS"/*.service.d/*.conf; do
    [[ -f "$src" ]] || continue
    dropin_dir=$(basename "$(dirname "$src")")   # e.g. lapis-brief-morning.service.d
    conf=$(basename "$src")                      # e.g. wake.conf
    dst_dir="$SYSTEMD/$dropin_dir"
    dst="$dst_dir/$conf"
    sudo_or_warn mkdir -p "$dst_dir"
    if cmp -s "$src" "$dst" 2>/dev/null; then
        log "Drop-in $dropin_dir/$conf: in sync"
    else
        log "Drop-in $dropin_dir/$conf: differs — copying"
        sudo_or_warn cp "$src" "$dst"
        NEED_RELOAD=1
        unit="${dropin_dir%.d}"                  # e.g. lapis-brief-morning.service
        timer="${unit%.service}.timer"
        [[ -f "$SRC_UNITS/$timer" ]] && UNITS_CHANGED+=("$timer")
    fi
done

if [[ "$NEED_RELOAD" -eq 1 ]]; then
    log "daemon-reload"
    sudo_or_warn systemctl daemon-reload
fi

# ── Step 4: verify wake-and-run dependency ────────────────────────────────
log "Step 4: verifying wake-and-run"
if [[ ! -x /usr/local/bin/wake-and-run ]]; then
    warn "/usr/local/bin/wake-and-run not found or not executable"
    warn "Drop-ins require it. Install from: /srv/git/conductor/scripts/wake-and-run"
fi

# ── Step 5: restart changed timers ────────────────────────────────────────
log "Step 5: restarting changed timers"
declare -A _SEEN=()
RESTART=()
for t in "${UNITS_CHANGED[@]}"; do
    if [[ -z "${_SEEN[$t]+x}" ]]; then
        _SEEN[$t]=1
        RESTART+=("$t")
    fi
done

if [[ "${#RESTART[@]}" -eq 0 ]]; then
    log "No timers changed — skipping restarts"
else
    for timer in "${RESTART[@]}"; do
        log "Restarting $timer"
        sudo_or_warn systemctl restart "$timer"
    done
fi

# ── Summary ────────────────────────────────────────────────────────────────
log "Summary: code=$([ "$CODE_CHANGED" -eq 1 ] && echo "synced $PRE_REV..$POST_REV" || echo unchanged); units=$([ "$NEED_RELOAD" -eq 1 ] && echo propagated || echo in-sync); restarts=${#RESTART[@]}"
