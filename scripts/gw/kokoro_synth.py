#!/usr/bin/env python3
"""Kokoro inference CLI for lapis-pm's gardener-tts-episode-v0 bakeoff.

Deployed to GravityWell by scripts/provision_tts.sh (scp'd from this
checked-in copy, not heredoc-generated, so it stays reviewable and
versioned like the rest of this unit). Invoked over SSH by
lapis_pm.tts_episode._synthesize_kokoro():

    kokoro_synth.py --text-file <path> --models-dir <dir> --output-file <path>

Loads the pinned Kokoro model + a single voice pack entirely from local
files under --models-dir (kokoro-v1_0.pth, config.json, voices/<voice>.pt
— see scripts/provision_tts.sh) so there is no HF network call at synth
time, synthesizes the text at --text-file, and writes a 16-bit PCM WAV at
--output-file using the stdlib `wave` module (no extra audio-IO
dependency needed for a WAV write).

Runs on CPU by default. GravityWell's GPU is typically saturated by the
primary llama-server serving lane (~80/98GB used); Kokoro is an 82M-param
model that renders fast enough on CPU that contending with that lane for
GPU time isn't worth it. Override via KOKORO_DEVICE=cuda if ever needed.

Exits non-zero with a clear stderr message on any failure (missing model
files, import/runtime error) — _synthesize_kokoro() treats a non-zero
exit as TTSEngineError and surfaces it through per-engine fallback.
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
DEFAULT_DEVICE = os.environ.get("KOKORO_DEVICE", "cpu")
SAMPLE_RATE = 24000


def _fail(msg: str) -> None:
    print(f"kokoro_synth: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", required=True, type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    args = parser.parse_args()

    if not args.text_file.exists():
        _fail(f"text file not found: {args.text_file}")
    text = args.text_file.read_text(encoding="utf-8").strip()
    if not text:
        _fail("text file is empty")

    config_path = args.models_dir / "config.json"
    model_path = args.models_dir / "kokoro-v1_0.pth"
    voice_path = args.models_dir / "voices" / f"{DEFAULT_VOICE}.pt"
    for p in (config_path, model_path, voice_path):
        if not p.exists():
            _fail(f"required model file missing: {p} — run provision_tts.sh's kokoro leg")

    try:
        import numpy as np
        import torch
        from kokoro import KModel, KPipeline
    except ImportError as e:
        _fail(f"kokoro/torch not importable: {e} — run provision_tts.sh's kokoro leg")

    try:
        model = KModel(config=str(config_path), model=str(model_path))
        model = model.to(DEFAULT_DEVICE).eval()
        pipeline = KPipeline(lang_code="a", model=model, repo_id="hexgrad/Kokoro-82M")

        chunks = []
        for result in pipeline(text, voice=str(voice_path)):
            if result.audio is not None:
                chunks.append(result.audio)
        if not chunks:
            _fail("kokoro produced no audio segments")

        audio = torch.cat(chunks, dim=0).detach().cpu().numpy()
    except SystemExit:
        raise
    except Exception as e:
        _fail(f"synthesis failed: {type(e).__name__}: {e}")

    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output_file.with_suffix(args.output_file.suffix + ".tmp")
    with wave.open(str(tmp_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    tmp_path.rename(args.output_file)

    print(f"kokoro_synth: wrote {args.output_file} ({len(pcm16) / SAMPLE_RATE:.1f}s audio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
