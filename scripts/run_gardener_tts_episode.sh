#!/usr/bin/env bash
# ExecStart wrapper for gardener-tts-episode.service (Unit 3b,
# gardener-tts-feed-v0). Appends a run log and fires a Pushover on any
# UNEXPECTED crash — the CLI itself (lapis_pm.tts_feed.publish_episode)
# already notifies on every expected success/failure path (episode
# published, both engines failed, MP3 encode failed, feed config missing);
# this wrapper only needs to catch the class those paths can't: argparse
# errors, import failures, an unhandled exception before publish_episode
# ever runs.
set -uo pipefail

cd /srv/git/lapis-pm
set -a
. /data/agents/config/conductor.env
set +a
export TTS_PIPER_BIN="${TTS_PIPER_BIN:-/data/agents/tts/piper-venv/bin/piper}"

LOG="/data/agents/tts/gardener-tts-episode-run.log"
mkdir -p "$(dirname "${LOG}")"

{
  echo "=== $(date -Iseconds) ==="
  /usr/bin/python3 -m lapis_pm.cli tts-episode-publish --period morning
  rc=$?
  echo "=== exit ${rc} ==="
} >> "${LOG}" 2>&1

rc=${rc:-1}
# Only notify here if the CLI itself did NOT already fire its own detailed
# Pushover — cmd_tts_episode_publish prints this exact marker on every
# expected failure path (both-engines-down, encode failure, config error),
# each of which already called send_notification internally. Anything
# non-zero WITHOUT that marker is an unexpected crash (traceback, argparse
# error, import failure) the CLI never got a chance to notify about itself.
if [[ ${rc} -ne 0 ]] && ! tail -50 "${LOG}" | grep -q "tts-episode-publish: FAILED"; then
  /usr/bin/python3 -c "
from agents_core.notify import send_notification
send_notification(
    message='gardener-tts-episode.service exited ${rc} unexpectedly (outside the CLI\'s own notify paths) — check ${LOG}',
    title='Gardener TTS feed — service crashed',
    source='gardener-tts-episode.service',
)
" || true
fi

exit "${rc}"
