#!/usr/bin/env bash
# Idempotent, version-pinned TTS provisioning for gardener-tts-episode-v0.
#
# Piper leg: pip-installs a pinned piper-tts into the current CLI's
# environment, verifies import, and fetches one pinned English voice
# (.onnx + .onnx.json) to a temp path with an atomic rename — no
# destructive shared-env mutation, safe to run inside a fixer's isolated
# per-work clone.
#
# Kokoro leg: live-checks GravityWell first (a plain `ls` over an
# already-up host — never wakes GW). If the pinned model is already
# staged there, this is a no-op. If the live-check times out/errors,
# Kokoro is reported unavailable for this run and the script proceeds
# (Piper still delivers per per-engine fallback). Only fetches the
# ~330MB model to GW when absent, guarded against partial downloads.
#
# Safe to re-run: every step checks "already present" before acting.

set -euo pipefail

# --- Piper pins ---------------------------------------------------------
PIPER_TTS_PIN="${PIPER_TTS_PIN:-1.5.0}"
PIPER_VOICE_NAME="${TTS_PIPER_VOICE_NAME:-en_US-lessac-medium}"
PIPER_VOICE_DIR="${TTS_PIPER_VOICE_DIR:-$HOME/.local/share/piper-voices}"
PIPER_VOICE_BASE_URL="${PIPER_VOICE_BASE_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium}"

# --- Kokoro pins ---------------------------------------------------------
GW_SSH_HOST="${TTS_GW_SSH_HOST:-gravitywell}"
GW_KOKORO_MODELS_DIR="${TTS_GW_KOKORO_MODELS_DIR:-/data/models/kokoro}"
KOKORO_MODEL_PIN="${KOKORO_MODEL_PIN:-kokoro-v1.0}"
KOKORO_MODEL_URL="${KOKORO_MODEL_URL:-https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth}"

log() { echo "[provision_tts] $*" >&2; }

# =========================================================================
# Piper (BRIX-CPU, no GW lease)
# =========================================================================

provision_piper() {
    log "piper: checking piper-tts==${PIPER_TTS_PIN} import"
    if python3 -c "import piper" >/dev/null 2>&1; then
        installed_ver="$(python3 -c 'import importlib.metadata as m; print(m.version("piper-tts"))' 2>/dev/null || echo unknown)"
        log "piper: piper-tts already importable (version ${installed_ver})"
    else
        log "piper: installing piper-tts==${PIPER_TTS_PIN}"
        if ! python3 -m pip install --user "piper-tts==${PIPER_TTS_PIN}"; then
            log "piper: pip install FAILED — skipping Piper leg (per-engine fallback)"
            return 1
        fi
        if ! python3 -c "import piper" >/dev/null 2>&1; then
            log "piper: import verification FAILED after install — skipping Piper leg"
            return 1
        fi
    fi

    mkdir -p "${PIPER_VOICE_DIR}"
    model_path="${PIPER_VOICE_DIR}/${PIPER_VOICE_NAME}.onnx"
    config_path="${PIPER_VOICE_DIR}/${PIPER_VOICE_NAME}.onnx.json"

    if [[ -s "${model_path}" && -s "${config_path}" ]]; then
        log "piper: voice model already staged at ${model_path}"
        return 0
    fi

    log "piper: fetching voice model ${PIPER_VOICE_NAME}"
    tmp_model="$(mktemp "${PIPER_VOICE_DIR}/.${PIPER_VOICE_NAME}.onnx.XXXXXX")"
    tmp_config="$(mktemp "${PIPER_VOICE_DIR}/.${PIPER_VOICE_NAME}.onnx.json.XXXXXX")"
    trap 'rm -f "${tmp_model}" "${tmp_config}"' RETURN

    if curl -fsSL "${PIPER_VOICE_BASE_URL}/${PIPER_VOICE_NAME}.onnx" -o "${tmp_model}" \
        && curl -fsSL "${PIPER_VOICE_BASE_URL}/${PIPER_VOICE_NAME}.onnx.json" -o "${tmp_config}"; then
        mv -f "${tmp_model}" "${model_path}"
        mv -f "${tmp_config}" "${config_path}"
        log "piper: voice model staged at ${model_path}"
    else
        log "piper: voice model fetch FAILED — skipping Piper leg (per-engine fallback)"
        return 1
    fi
}

# =========================================================================
# Kokoro (GravityWell, off-peak, DoormanClient owns the actual lease at
# synth time — this script only stages the model file)
# =========================================================================

provision_kokoro() {
    log "kokoro: live-check (ls over an already-up host — never wakes GW)"
    if ! listing="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${GW_SSH_HOST}" \
        "ls -la ${GW_KOKORO_MODELS_DIR}/kokoro* 2>/dev/null" 2>/dev/null)"; then
        log "kokoro: live-check unreachable/timed out — Kokoro unavailable this run"
        return 1
    fi

    if [[ -n "${listing}" ]]; then
        log "kokoro: model already staged on GravityWell:"
        log "${listing}"
        return 0
    fi

    log "kokoro: no staged model found — fetching pinned ${KOKORO_MODEL_PIN} (~330MB)"
    remote_tmp="${GW_KOKORO_MODELS_DIR}/.${KOKORO_MODEL_PIN}.download.$$"
    remote_final="${GW_KOKORO_MODELS_DIR}/${KOKORO_MODEL_PIN}.pth"
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "${GW_SSH_HOST}" bash -s -- \
        "${remote_tmp}" "${remote_final}" "${KOKORO_MODEL_URL}" "${GW_KOKORO_MODELS_DIR}" <<'REMOTE'
set -euo pipefail
remote_tmp="$1"; remote_final="$2"; url="$3"; models_dir="$4"
mkdir -p "${models_dir}"
curl -fsSL "${url}" -o "${remote_tmp}"
mv -f "${remote_tmp}" "${remote_final}"
REMOTE
    then
        log "kokoro: model staged at ${remote_final}"
    else
        log "kokoro: model fetch FAILED — Kokoro unavailable this run (per-engine fallback)"
        return 1
    fi
}

# =========================================================================

main() {
    piper_ok=0
    kokoro_ok=0
    provision_piper && piper_ok=1 || true
    provision_kokoro && kokoro_ok=1 || true

    log "summary: piper=$([[ $piper_ok == 1 ]] && echo ok || echo unavailable) kokoro=$([[ $kokoro_ok == 1 ]] && echo ok || echo unavailable)"

    if [[ $piper_ok == 0 && $kokoro_ok == 0 ]]; then
        log "both engines unavailable — nothing provisioned"
        exit 1
    fi
    exit 0
}

main "$@"
