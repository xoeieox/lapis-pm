#!/usr/bin/env bash
# Idempotent, version-pinned TTS provisioning for gardener-tts-episode-v0.
#
# Piper leg: pip-installs a pinned piper-tts into the current CLI's
# environment, verifies import, and fetches one pinned English voice
# (.onnx + .onnx.json) to a temp path with an atomic rename — no
# destructive shared-env mutation, safe to run inside a fixer's isolated
# per-work clone.
#
# Kokoro leg: live-checks GravityWell first (a plain `test` over an
# already-up host — never wakes GW). If the pinned model + config + voice
# are already staged there, model staging is a no-op; otherwise fetches
# them (~330MB), guarded against partial downloads. Then (round-3)
# provisions an actual inference *runtime* — GW already runs vLLM out of
# dedicated venvs (e.g. /data/vllm-venv) for its serving lanes, but those
# are live systemd-managed environments; installing a third-party package
# into one risks a dependency conflict that could break serving, so this
# creates its own dedicated venv instead — and deploys the checked-in
# scripts/gw/kokoro_synth.py inference script. If the live-check times
# out/errors, or any Kokoro step fails, Kokoro is reported unavailable for
# this run and the script proceeds (Piper still delivers per per-engine
# fallback).
#
# Safe to re-run: every step checks "already present" before acting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Piper pins ---------------------------------------------------------
PIPER_TTS_PIN="${PIPER_TTS_PIN:-1.5.0}"
PIPER_VOICE_NAME="${TTS_PIPER_VOICE_NAME:-en_US-lessac-medium}"
PIPER_VOICE_DIR="${TTS_PIPER_VOICE_DIR:-$HOME/.local/share/piper-voices}"
PIPER_VOICE_BASE_URL="${PIPER_VOICE_BASE_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium}"

# --- Kokoro pins ---------------------------------------------------------
GW_SSH_HOST="${TTS_GW_SSH_HOST:-gravitywell}"
GW_KOKORO_MODELS_DIR="${TTS_GW_KOKORO_MODELS_DIR:-/data/models/kokoro}"
KOKORO_MODEL_PIN="${KOKORO_MODEL_PIN:-kokoro-v1.0}"
KOKORO_MODEL_FILE="kokoro-v1_0.pth"
KOKORO_CONFIG_FILE="config.json"
KOKORO_VOICE_NAME="${TTS_KOKORO_VOICE:-af_heart}"
KOKORO_VOICE_FILE="voices/${KOKORO_VOICE_NAME}.pt"
KOKORO_HF_BASE_URL="${KOKORO_HF_BASE_URL:-https://huggingface.co/hexgrad/Kokoro-82M/resolve/main}"

# --- Kokoro inference runtime pins (round-3) -----------------------------
KOKORO_PKG_PIN="${KOKORO_PKG_PIN:-0.9.4}"
GW_KOKORO_VENV_PY="${TTS_GW_KOKORO_VENV_PY:-/data/agents/tts/venv/bin/python}"
GW_KOKORO_VENV_DIR="${GW_KOKORO_VENV_PY%/bin/python}"
GW_KOKORO_SYNTH_SCRIPT="${TTS_GW_KOKORO_SYNTH_SCRIPT:-/data/agents/tts/kokoro_synth.py}"
LOCAL_KOKORO_SYNTH_SCRIPT="${SCRIPT_DIR}/gw/kokoro_synth.py"

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
# synth time — this script stages the model files + inference runtime)
# =========================================================================

_kokoro_model_staged_status() {
    # Echoes PRESENT/MISSING on stdout; returns non-zero only on
    # unreachable/timeout so callers can tell "no model yet" apart from
    # "couldn't even reach GW".
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${GW_SSH_HOST}" bash -s -- \
        "${GW_KOKORO_MODELS_DIR}" "${KOKORO_MODEL_FILE}" "${KOKORO_CONFIG_FILE}" "${KOKORO_VOICE_FILE}" <<'REMOTE'
set -uo pipefail
dir="$1"; model="$2"; config="$3"; voice="$4"
if [[ -s "${dir}/${model}" && -s "${dir}/${config}" && -s "${dir}/${voice}" ]]; then
    echo PRESENT
else
    echo MISSING
fi
REMOTE
}

_kokoro_fetch_model() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${GW_SSH_HOST}" bash -s -- \
        "${GW_KOKORO_MODELS_DIR}" "${KOKORO_MODEL_FILE}" "${KOKORO_CONFIG_FILE}" "${KOKORO_VOICE_FILE}" "${KOKORO_HF_BASE_URL}" <<'REMOTE'
set -euo pipefail
dir="$1"; model="$2"; config="$3"; voice="$4"; base_url="$5"
mkdir -p "${dir}/voices"
fetch_one() {
    local rel="$1"
    local dest="${dir}/${rel}"
    if [[ -s "${dest}" ]]; then return 0; fi
    local tmp
    tmp="$(mktemp "${dest}.download.XXXXXX")"
    curl -fsSL "${base_url}/${rel}" -o "${tmp}"
    mv -f "${tmp}" "${dest}"
}
fetch_one "${model}"
fetch_one "${config}"
fetch_one "${voice}"
REMOTE
}

_kokoro_provision_runtime() {
    log "kokoro: checking inference runtime at ${GW_KOKORO_VENV_DIR}"
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "${GW_SSH_HOST}" \
        "${GW_KOKORO_VENV_PY} -c 'import kokoro, torch; from misaki import en'" >/dev/null 2>&1; then
        log "kokoro: inference runtime already provisioned (kokoro importable in ${GW_KOKORO_VENV_DIR})"
        return 0
    fi

    # GW is a GPU host that already runs vLLM out of dedicated venvs
    # (/data/vllm-venv, /data/aw-serve-venv — confirmed live, torch
    # 2.11.0+cu130) for its serving lanes, but those are live
    # systemd-managed environments (vllm-slot1/slot2/swarm.service);
    # installing a third-party package into one risks a dependency
    # conflict that could break serving. Not safe to reuse — create a
    # dedicated venv instead, same discipline as Piper's "no destructive
    # shared-env mutation".
    log "kokoro: no reusable env — creating dedicated venv at ${GW_KOKORO_VENV_DIR}"
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${GW_SSH_HOST}" bash -s -- \
        "${GW_KOKORO_VENV_DIR}" "${KOKORO_PKG_PIN}" <<'REMOTE'
set -euo pipefail
venv_dir="$1"; pkg_pin="$2"
mkdir -p "$(dirname "${venv_dir}")"
if [[ ! -x "${venv_dir}/bin/python" ]]; then
    python3 -m venv "${venv_dir}"
fi
# PIP_USER=yes forces --user installs even inside an activated venv on
# some hosts (confirmed live on the fixer's BRIX worktree — see
# friction.json); a venv has no --user site-packages so that fails
# outright. Override defensively even though GW's global env was
# confirmed unset at investigation time (round-3).
#
# CPU-only torch: kokoro_synth.py runs on CPU by default (GW's GPU is
# typically saturated by the primary llama-server serving lane; an
# 82M-param model doesn't need to contend for it), so pull the much
# smaller CPU wheel instead of the default CUDA build.
PIP_USER=no "${venv_dir}/bin/python" -m pip install --quiet --upgrade pip
PIP_USER=no "${venv_dir}/bin/python" -m pip install --quiet \
    --index-url https://download.pytorch.org/whl/cpu torch
PIP_USER=no "${venv_dir}/bin/python" -m pip install --quiet "kokoro==${pkg_pin}"
# misaki's English G2P lazily pulls the spaCy en_core_web_sm model on its
# first tokenize call if absent — pre-fetch it here so that cost lands in
# provisioning, not in the first (or every fresh-venv) synth's measured
# latency.
PIP_USER=no "${venv_dir}/bin/python" -m spacy download en_core_web_sm --quiet || true
"${venv_dir}/bin/python" -c "import kokoro, torch; from misaki import en"
REMOTE
    then
        return 1
    fi

    # misaki's English G2P falls back to espeak-ng for out-of-dictionary
    # words. It bundles its own copy via espeakng-loader (the primary
    # path, already verified importable above), but install the system
    # package too as a defensive fallback — idempotent, and additive
    # (a fresh apt package) rather than a mutation of existing shared
    # state, so safe even though this touches the host outside the venv.
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${GW_SSH_HOST}" \
        "dpkg -s espeak-ng >/dev/null 2>&1 || sudo -n apt-get install -y -qq espeak-ng" \
        >/dev/null 2>&1 \
        || log "kokoro: espeak-ng system package install skipped/failed (non-fatal — misaki's bundled espeakng-loader is the primary path)"

    return 0
}

_kokoro_deploy_synth_script() {
    if [[ ! -f "${LOCAL_KOKORO_SYNTH_SCRIPT}" ]]; then
        log "kokoro: local kokoro_synth.py not found at ${LOCAL_KOKORO_SYNTH_SCRIPT}"
        return 1
    fi
    log "kokoro: deploying kokoro_synth.py to ${GW_KOKORO_SYNTH_SCRIPT}"
    local remote_dir
    remote_dir="$(dirname "${GW_KOKORO_SYNTH_SCRIPT}")"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${GW_SSH_HOST}" "mkdir -p '${remote_dir}'" \
        && scp -p -o BatchMode=yes -o ConnectTimeout=10 \
            "${LOCAL_KOKORO_SYNTH_SCRIPT}" "${GW_SSH_HOST}:${GW_KOKORO_SYNTH_SCRIPT}.tmp" \
        && ssh -o BatchMode=yes -o ConnectTimeout=10 "${GW_SSH_HOST}" \
            "mv -f '${GW_KOKORO_SYNTH_SCRIPT}.tmp' '${GW_KOKORO_SYNTH_SCRIPT}'"
}

provision_kokoro() {
    log "kokoro: live-check (test over an already-up host — never wakes GW)"
    local status
    if ! status="$(_kokoro_model_staged_status)"; then
        log "kokoro: live-check unreachable/timed out — Kokoro unavailable this run"
        return 1
    fi

    if [[ "${status}" == "PRESENT" ]]; then
        log "kokoro: model + config + voice already staged on GravityWell"
    else
        log "kokoro: staging pinned ${KOKORO_MODEL_PIN} (~330MB) + config + voice ${KOKORO_VOICE_NAME}"
        if ! _kokoro_fetch_model; then
            log "kokoro: model fetch FAILED — Kokoro unavailable this run (per-engine fallback)"
            return 1
        fi
    fi

    if ! _kokoro_provision_runtime; then
        log "kokoro: inference runtime provisioning FAILED — Kokoro unavailable this run (per-engine fallback)"
        return 1
    fi

    if ! _kokoro_deploy_synth_script; then
        log "kokoro: kokoro_synth.py deploy FAILED — Kokoro unavailable this run (per-engine fallback)"
        return 1
    fi

    log "kokoro: model + runtime + synth script all staged on GravityWell"
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
