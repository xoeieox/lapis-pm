"""Spoken-episode synthesis for lapis-pm (Unit 3a, gardener-tts-episode-v0).

Renders an existing morning-brief body to a playable WAV via two candidate
TTS engines (Piper on BRIX-CPU, Kokoro on GravityWell) so Erah can hear both
on real content and pick a voice by ear. This module owns ALL synthesis +
delivery logic (normalizer, per-engine dispatch, vault-drop + Pushover) —
`state_brief.py` (content generation) is untouched; this module only imports
its output via `room_path('briefs')`.

Mirror Council invariant (warnings sharp, climate flows): `[Critical]`/
`[Warning]`-tagged lines render as short, isolated declarative sentences on
their own paragraph; everything else (climate/narrative prose, bullets)
flows into continuous spoken paragraphs. This is a structural law, not an
aesthetic option — see normalize_for_speech.

v0 scope: WAV only (no ffmpeg/MP3 — that's 3b), on-demand (no timer-fire —
3b), vault-drop + Pushover delivery (interim bakeoff surface — the durable
RSS feed is 3b). No LLM call anywhere in this module; the normalizer is
pure string manipulation and Piper/Kokoro are local TTS models, not LLMs.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from agents_core.notify import send_notification

logger = logging.getLogger(__name__)

PACIFIC = ZoneInfo("America/Los_Angeles")

VOICES = ("piper", "kokoro")

# --- Piper (BRIX-CPU, local subprocess, no GW lease) ------------------------

PIPER_BIN = os.environ.get("TTS_PIPER_BIN", "piper")
PIPER_VOICE_NAME = os.environ.get("TTS_PIPER_VOICE_NAME", "en_US-lessac-medium")
PIPER_VOICE_DIR = Path(
    os.environ.get("TTS_PIPER_VOICE_DIR", str(Path.home() / ".local/share/piper-voices"))
)
PIPER_MODEL_PATH = PIPER_VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx"
PIPER_MODEL_CONFIG_PATH = PIPER_VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx.json"
# Hard ceiling so a truly stuck subprocess can't hang the bakeoff forever.
# Well above the 45s soft-fail *reporting* threshold below — we still want
# the real measured time for a render that finishes between 45s and this.
PIPER_HARD_TIMEOUT_SEC = 180

# Facets consensus / Council reservation (round 2): a Piper cold-render of a
# full morning brief that exceeds this is a soft-fail — withhold the
# artifact (don't present a stall as a success), report the measured time,
# mark Piper a discard-candidate for 3b's auto-path. Kokoro still delivers.
PIPER_LATENCY_SOFT_FAIL_SEC = 45.0

# --- Kokoro (GravityWell, DoormanClient off-peak lease) ---------------------

GW_NODE = "gravitywell"
GW_SSH_HOST = os.environ.get("TTS_GW_SSH_HOST", "gravitywell")
GW_KOKORO_VENV_PY = os.environ.get("TTS_GW_KOKORO_VENV_PY", "/data/agents/tts/venv/bin/python")
GW_KOKORO_SYNTH_SCRIPT = os.environ.get(
    "TTS_GW_KOKORO_SYNTH_SCRIPT", "/data/agents/tts/kokoro_synth.py"
)
GW_KOKORO_MODELS_DIR = os.environ.get("TTS_GW_KOKORO_MODELS_DIR", "/data/models/kokoro")
GW_TMP_DIR = os.environ.get("TTS_GW_TMP_DIR", "/tmp")
KOKORO_LEASE_TTL_SEC = int(os.environ.get("TTS_KOKORO_LEASE_TTL_SEC", "600"))
KOKORO_SCP_TIMEOUT_SEC = int(os.environ.get("TTS_KOKORO_SCP_TIMEOUT_SEC", "60"))
KOKORO_SYNTH_TIMEOUT_SEC = int(os.environ.get("TTS_KOKORO_SYNTH_TIMEOUT_SEC", "300"))


class TTSEngineError(Exception):
    """Raised when a TTS engine fails to produce audio for its leg of the bakeoff."""


# ---------------------------------------------------------------------------
# normalize_for_speech — deterministic spoken-pacing normalizer
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r'^#{1,6}\s+(.*\S)\s*$')
_BULLET_RE = re.compile(r'^[-*]\s+(.*\S)\s*$')
_WARNING_RE = re.compile(r'^\[(Critical|Warning)\]\s*(.*)$', re.IGNORECASE)
_EVIDENCE_PAREN_RE = re.compile(r'\s*\(evidence:\s*[^)]*\)', re.IGNORECASE)
_BARE_PATH_RE = re.compile(r'(?<!\S)/[\w.\-]+(?:/[\w.\-]+)+/?(?!\S)')
_ID_PREFIX_RE = re.compile(r'^(?:arc-doc|chain):\s*', re.IGNORECASE)
_TRUNCATED_ID_PREFIX_RE = re.compile(r'^[0-9a-f]{6,40}:\s*', re.IGNORECASE)
_EMPHASIS_RE = re.compile(r'[*_`]+')
_WHITESPACE_RE = re.compile(r'[ \t]{2,}')


def _clean_text(text: str) -> str:
    """Strip evidence parentheticals, bare paths, id-prefixes, markdown emphasis."""
    text = _EVIDENCE_PAREN_RE.sub('', text)
    text = _BARE_PATH_RE.sub('', text)
    text = _ID_PREFIX_RE.sub('', text)
    text = _TRUNCATED_ID_PREFIX_RE.sub('', text)
    text = _EMPHASIS_RE.sub('', text)
    text = _WHITESPACE_RE.sub(' ', text)
    return text.strip()


def _as_sentence(text: str) -> str:
    if not text:
        return text
    if text[-1] not in ".!?":
        text += "."
    return text


def normalize_for_speech(body: str) -> str:
    """Convert a markdown brief body into clean spoken prose.

    Strips visual markup (headers, bullet markers, emphasis), evidence
    parentheticals, bare paths, and `arc-doc:`/`chain:`/truncated-id
    prefixes. Headers become their own standalone paragraph (spoken as a
    section break).

    Council invariant (warnings sharp, climate flows): any line whose
    (cleaned) text begins with `[Critical]` or `[Warning]` — bulleted or
    plain — becomes its own isolated declarative-sentence paragraph
    ("Critical: <text>."), never merged with surrounding prose. Everything
    else (climate/narrative bullets and paragraph text) accumulates into
    flowing paragraphs, broken only by headers, warning items, or blank
    lines in the source — paragraph structure is preserved, not flattened.
    """
    lines = body.replace("\r\n", "\n").split("\n")
    paragraphs: list[str] = []
    flow_buffer: list[str] = []

    def flush_flow() -> None:
        if flow_buffer:
            paragraphs.append(" ".join(flow_buffer))
            flow_buffer.clear()

    for raw_line in lines:
        line = raw_line.rstrip()

        if not line.strip():
            flush_flow()
            continue

        header_m = _HEADER_RE.match(line)
        if header_m:
            flush_flow()
            heading = _clean_text(header_m.group(1))
            if heading:
                paragraphs.append(heading)
            continue

        bullet_m = _BULLET_RE.match(line)
        content = bullet_m.group(1) if bullet_m else line

        cleaned = _clean_text(content)
        if not cleaned:
            continue

        warn_m = _WARNING_RE.match(cleaned)
        if warn_m:
            flush_flow()
            label = warn_m.group(1).capitalize()
            rest = warn_m.group(2).strip()
            if rest:
                paragraphs.append(_as_sentence(f"{label}: {rest}"))
            continue

        flow_buffer.append(_as_sentence(cleaned))

    flush_flow()
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Per-engine synthesis
# ---------------------------------------------------------------------------

def _synthesize_piper(text: str, out_path: Path) -> None:
    """Render `text` to a WAV at `out_path` via the local Piper subprocess.

    No GravityWell lease — Piper runs entirely on BRIX-CPU, so this leg is
    always TOU-clean. Raises TTSEngineError on any failure.
    """
    if not PIPER_MODEL_PATH.exists():
        raise TTSEngineError(
            f"piper voice model not found at {PIPER_MODEL_PATH} — run scripts/provision_tts.sh"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [PIPER_BIN, "--model", str(PIPER_MODEL_PATH), "--output_file", str(out_path)],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=PIPER_HARD_TIMEOUT_SEC,
        )
    except FileNotFoundError as e:
        raise TTSEngineError(
            f"piper binary not found ({PIPER_BIN}) — run scripts/provision_tts.sh"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise TTSEngineError(
            f"piper render exceeded hard timeout ({PIPER_HARD_TIMEOUT_SEC}s)"
        ) from e

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[:300]
        raise TTSEngineError(f"piper synth failed: {detail}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise TTSEngineError("piper produced no audio output")


def _synthesize_kokoro(text: str, out_path: Path, *, work_id: str) -> None:
    """Render `text` to a WAV at `out_path` via a GravityWell Kokoro lease.

    Pattern (reference: conductor-working/dashboard/voice_capture/app.py):
    acquire → assert status=="serving" → synth → release + close in a
    `finally`, so a lease is never leaked on failure. Raises TTSEngineError
    on any failure (unreachable doorman, non-serving status, synth/scp
    failure).
    """
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable

    client = DoormanClient()
    lease_acquired = False
    remote_text_path = f"{GW_TMP_DIR}/{work_id}.txt"
    remote_wav_path = f"{GW_TMP_DIR}/{work_id}.wav"
    local_text_path: Path | None = None
    try:
        try:
            resp = client.acquire(
                GW_NODE, work_id, ttl_sec=KOKORO_LEASE_TTL_SEC, reason="tts-episode",
            )
        except DoormanUnreachable as e:
            raise TTSEngineError(f"kokoro: doorman unreachable: {e}") from e

        status = resp.get("status", "")
        if status != "serving":
            raise TTSEngineError(f"kokoro: gravitywell not serving (status={status})")
        lease_acquired = True

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            local_text_path = Path(f.name)

        scp_up = subprocess.run(
            ["scp", "-p", str(local_text_path), f"{GW_SSH_HOST}:{remote_text_path}"],
            capture_output=True, timeout=KOKORO_SCP_TIMEOUT_SEC,
        )
        if scp_up.returncode != 0:
            raise TTSEngineError("kokoro: scp text to gravitywell failed")

        synth = subprocess.run(
            ["ssh", GW_SSH_HOST, GW_KOKORO_VENV_PY, GW_KOKORO_SYNTH_SCRIPT,
             "--text-file", remote_text_path,
             "--models-dir", GW_KOKORO_MODELS_DIR,
             "--output-file", remote_wav_path],
            capture_output=True, timeout=KOKORO_SYNTH_TIMEOUT_SEC,
        )
        if synth.returncode != 0:
            detail = synth.stderr.decode("utf-8", errors="replace")[:300]
            raise TTSEngineError(f"kokoro synth failed: {detail}")

        scp_down = subprocess.run(
            ["scp", "-p", f"{GW_SSH_HOST}:{remote_wav_path}", str(out_path)],
            capture_output=True, timeout=KOKORO_SCP_TIMEOUT_SEC,
        )
        if scp_down.returncode != 0:
            raise TTSEngineError("kokoro: scp wav from gravitywell failed")
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise TTSEngineError("kokoro produced no audio output")
    finally:
        if local_text_path is not None:
            local_text_path.unlink(missing_ok=True)
        try:
            subprocess.run(
                ["ssh", GW_SSH_HOST, "rm", "-f", remote_text_path, remote_wav_path],
                capture_output=True, timeout=KOKORO_SCP_TIMEOUT_SEC,
            )
        except Exception:
            pass
        if lease_acquired:
            client.release(GW_NODE, work_id)
        client.close()


def synthesize_episode(
    body: str,
    *,
    voice: str,
    out_dir: Path,
    filename: str | None = None,
    work_id: str | None = None,
) -> Path | None:
    """Normalize `body` and dispatch to the requested TTS engine.

    `voice` is "piper" (local subprocess, no lease) or "kokoro" (GravityWell
    via DoormanClient off-peak lease). Writes the WAV under `out_dir` and
    returns its path.

    Raises TTSEngineError on synthesis failure — callers doing a multi-engine
    bakeoff (see run_episode_bakeoff) catch this per engine so one engine's
    failure never blocks delivery of the other (per-engine fallback).
    """
    if voice not in VOICES:
        raise ValueError(f"unknown voice {voice!r} — expected one of {VOICES}")

    text = normalize_for_speech(body)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (filename or f"{voice}.wav")

    if voice == "piper":
        _synthesize_piper(text, out_path)
    else:
        wid = work_id or f"tts-episode-kokoro-{uuid.uuid4().hex[:12]}"
        _synthesize_kokoro(text, out_path, work_id=wid)

    return out_path


# ---------------------------------------------------------------------------
# Bakeoff orchestration — delivery (vault-drop + Pushover)
# ---------------------------------------------------------------------------

_ENGINE_LABELS = {"piper": "Piper", "kokoro": "Kokoro"}

_FEEDBACK_PROMPT = (
    "Which voice? Did the Critical warning snap your attention (y/n)? "
    "Did the climate flow (y/n)?"
)


@dataclass
class EngineOutcome:
    voice: str
    path: Path | None
    elapsed_sec: float
    error: str | None = None
    soft_failed: bool = False


@dataclass
class BakeoffOutcome:
    engines: list[EngineOutcome] = field(default_factory=list)
    notified: bool = False

    @property
    def succeeded(self) -> list[EngineOutcome]:
        return [e for e in self.engines if e.path is not None]

    @property
    def ok(self) -> bool:
        return bool(self.succeeded)


def _succeeded_set_label(succeeded_voices: list[str]) -> str:
    labels = [_ENGINE_LABELS[v] for v in VOICES if v in succeeded_voices]
    return " + ".join(labels) if len(labels) > 1 else f"{labels[0]} only"


def _build_pushover_message(outcome: BakeoffOutcome) -> tuple[str, str]:
    succeeded = outcome.succeeded
    title = f"Gardener TTS bakeoff — {_succeeded_set_label([e.voice for e in succeeded])}"

    lines: list[str] = []
    for e in outcome.engines:
        label = _ENGINE_LABELS[e.voice]
        if e.path is not None:
            lines.append(f"{label}: {e.path}")
        elif e.soft_failed:
            lines.append(f"⚠ {label} >{PIPER_LATENCY_SOFT_FAIL_SEC:.0f}s — withheld ({e.elapsed_sec:.1f}s)")
        elif e.error:
            lines.append(f"{label} failed: {e.error}")

    for e in outcome.engines:
        if e.voice == "piper" and e.path is not None:
            lines.append(f"Piper latency: {e.elapsed_sec:.1f}s")

    lines.append(_FEEDBACK_PROMPT)
    return title, "\n".join(lines)


def run_episode_bakeoff(
    body: str,
    *,
    voices: Sequence[str],
    out_dir: Path,
    period: str = "morning",
    now: datetime | None = None,
) -> BakeoffOutcome:
    """Render `body` via each requested engine and deliver the result.

    Per-engine fallback: one engine failing (or Piper soft-failing on
    latency) still delivers the other; only *both* failing skips the
    Pushover ping entirely (outcome.ok is False, caller should error out).
    Fires exactly one Pushover ping summarizing the run when at least one
    engine succeeded.
    """
    if now is None:
        now = datetime.now(tz=PACIFIC)
    ts = now.strftime("%Y-%m-%d-%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    outcome = BakeoffOutcome()

    for voice in voices:
        filename = f"{ts}-{period}.{voice}.wav"
        start = time.monotonic()
        path: Path | None = None
        error: str | None = None
        try:
            path = synthesize_episode(body, voice=voice, out_dir=out_dir, filename=filename)
        except TTSEngineError as e:
            error = str(e)
        except Exception as e:  # unexpected — still per-engine fallback, not fatal
            error = f"{type(e).__name__}: {e}"
        elapsed = time.monotonic() - start

        soft_failed = False
        if voice == "piper" and path is not None and elapsed > PIPER_LATENCY_SOFT_FAIL_SEC:
            soft_failed = True
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            path = None

        if error:
            logger.warning("tts_episode: %s engine failed: %s", voice, error)

        outcome.engines.append(
            EngineOutcome(voice=voice, path=path, elapsed_sec=elapsed, error=error, soft_failed=soft_failed)
        )

    if outcome.ok:
        title, message = _build_pushover_message(outcome)
        outcome.notified = send_notification(message=message, title=title, source="tts-episode")

    return outcome
