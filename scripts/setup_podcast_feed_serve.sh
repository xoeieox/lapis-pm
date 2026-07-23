#!/usr/bin/env bash
# Idempotent host-op: mount /srv/lapis/briefs/audio on the tailnet at an
# unguessable path, so Erah's podcast app can subscribe over HTTPS
# (gardener-tts-feed-v0, R4).
#
# NOT run automatically by the publish pipeline or the timer/service — a
# deliberate, reviewable step run once (or re-run to rotate the token) as
# part of the pm-live-test verification pass (see the target spec's
# §Verification: the PM confirms the render + serving legs live before
# handing Erah the subscribe URL).
#
# Reads PODCAST_FEED_TOKEN from the environment — persistent, operator-set
# (the same value the service unit's conductor.env carries). Never
# generated here, never random: a runtime-random segment would silently
# break the subscription URL on every re-run (gate refinement).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="${TTS_FEED_AUDIO_DIR:-/srv/lapis/briefs/audio}"
TS_HOST="${TTS_FEED_TS_HOST:-brix.tail2e95fb.ts.net}"
# Tailnet IP — NEVER loopback. The existing cockpit mount already proxies to
# this same IP (203.0.113.10), not localhost; a bare 127.0.0.1 target is a
# known 502 trap on this host's tailscale/network setup.
PROXY_IP="${TTS_FEED_PROXY_IP:-203.0.113.10}"
FALLBACK_PORT="${TTS_FEED_FALLBACK_PORT:-8413}"

log() { echo "[setup_podcast_feed_serve] $*" >&2; }

if [[ -z "${PODCAST_FEED_TOKEN:-}" ]]; then
    log "ERROR: PODCAST_FEED_TOKEN is not set — refusing to mount without a persistent unguessable path segment"
    log "generate one (e.g. \`openssl rand -hex 8\`), add PODCAST_FEED_TOKEN=<token> to conductor.env, and re-run"
    exit 1
fi

MOUNT_PATH="/gardener-${PODCAST_FEED_TOKEN}"

# --- Detect tailscale serve static-dir capability -------------------------
# Explicit, deterministic checks (gate refinement) rather than just trying
# the mount and hoping: CLI present, daemon reachable/logged in, and this
# tailscale version's serve subcommand documents directory serving at all.
tailscale_serve_available() {
    if ! command -v tailscale >/dev/null 2>&1; then
        log "detect: tailscale CLI not found"
        return 1
    fi
    if ! tailscale serve status >/dev/null 2>&1; then
        log "detect: 'tailscale serve status' failed — daemon unreachable or not logged in"
        return 1
    fi
    if ! tailscale serve --help 2>&1 | grep -qi "directory"; then
        log "detect: this tailscale version's serve help doesn't document directory serving"
        return 1
    fi
    return 0
}

if tailscale_serve_available; then
    log "mounting ${AUDIO_DIR} directly at https://${TS_HOST}${MOUNT_PATH}/ (static-dir mount, no extra process)"
    tailscale serve --bg --set-path "${MOUNT_PATH}" "${AUDIO_DIR}"
else
    log "tailscale serve static-dir mount unavailable — fallback: dedicated static server + path-handler proxy"
    if [[ "${PROXY_IP}" == "127.0.0.1" || "${PROXY_IP}" == "localhost" || "${PROXY_IP}" == "::1" ]]; then
        log "ERROR: refusing to proxy to loopback (${PROXY_IP}) — known 502 trap on this host, must be a tailnet IP"
        exit 1
    fi
    log "start the static server first (not managed by this script):"
    log "  python3 ${SCRIPT_DIR}/podcast_static_server.py --bind ${PROXY_IP} --port ${FALLBACK_PORT} --dir ${AUDIO_DIR}"
    log "then front it:"
    PROXY_URL="http://${PROXY_IP}:${FALLBACK_PORT}"
    tailscale serve --bg --set-path "${MOUNT_PATH}" "${PROXY_URL}"
fi

log "done — feed will be reachable at https://${TS_HOST}${MOUNT_PATH}/feed.xml"
log "current serve config:"
tailscale serve status
