"""Durable private podcast RSS feed for the Gardener TTS episode (Unit 3b,
gardener-tts-feed-v0).

Wraps `tts_episode.run_episode_bakeoff` (called read-only — see that module
for the Piper/Kokoro synthesis + #227 humanize logic, which this module does
not touch) with the publish pipeline: Kokoro-primary / Piper-automatic-
fallback engine selection, PyAV MP3 transcode, a statelessly-regenerated
RSS 2.0 feed, and retention pruning.

Design (gardener-tts-feed-v0 spec + gate refinements):
  - Episode identity (ts/basename/GUID) is derived from the BRIEF's own
    generation timestamp (the `latest-<period>.md` symlink target), not the
    live publish-time clock, so re-running publish against an unchanged
    brief is a true no-op (idempotent — see `publish_episode`).
  - Kokoro is tried first (`voices=["kokoro"]`); only on failure is Piper
    tried (`voices=["piper"]`) — never both, so a normal day makes exactly
    one engine call and the Kokoro->Piper voice shift stays an honest signal
    rather than a routine two-engine render.
  - MP3 filenames carry the engine (`<ts>-<period>.<engine>.mp3`, mirroring
    the existing WAV convention) so the feed can show which voice rendered
    each episode even after the WAV itself has been pruned away.
  - feed.xml is regenerated from a timestamp-sorted directory listing every
    run (never appended-to), each candidate PyAV-probed before inclusion,
    text fields XML-escaped via ElementTree's own serializer (never
    hand-built strings).
  - PODCAST_FEED_TOKEN is a persistent, operator-set env value — never
    generated at runtime — so the subscription URL is stable across
    restarts.
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path

from agents_core.notify import send_notification

from . import tts_episode

logger = logging.getLogger(__name__)

PACIFIC = tts_episode.PACIFIC

# --- Config -------------------------------------------------------------

# Persistent, operator-set (e.g. via conductor.env / the service unit's
# Environment=) — never generated at runtime. Rotatable only by deliberately
# changing the env value (gate refinement: a runtime-random segment would
# silently break the subscription on every restart).
PODCAST_FEED_TOKEN = os.environ.get("PODCAST_FEED_TOKEN", "")

_DEFAULT_TAILNET_HOST = "brix.tail2e95fb.ts.net"

# PODCAST_BASE_URL defaults to a derivation from PODCAST_FEED_TOKEN so there
# is one source of truth for the unguessable path segment; override directly
# if the serving setup (e.g. the dedicated-static-server fallback path) puts
# the feed somewhere else.
PODCAST_BASE_URL = os.environ.get(
    "PODCAST_BASE_URL",
    f"https://{_DEFAULT_TAILNET_HOST}/gardener-{PODCAST_FEED_TOKEN}" if PODCAST_FEED_TOKEN else "",
)

RETENTION_KEEP = int(os.environ.get("PODCAST_RETENTION_KEEP", "14"))
MP3_BITRATE_BPS = int(os.environ.get("PODCAST_MP3_BITRATE", "64000"))

CHANNEL_TITLE = "Gardener Morning Brief"
CHANNEL_DESCRIPTION = "Erah's private daily Gardener morning brief, read aloud."
CHANNEL_AUTHOR = "Lapis Gardener"
CHANNEL_CATEGORY = "Technology"

ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
ET.register_namespace("itunes", ITUNES_NS)

_ENGINE_ITEM_LABEL = {"kokoro": "Kokoro", "piper": "Piper — fallback"}


class PodcastEncodeError(Exception):
    """Raised when MP3 encoding fails or the encoder is unavailable (R2 — fail loud)."""


class PodcastConfigError(Exception):
    """Raised when required feed config (PODCAST_BASE_URL/PODCAST_FEED_TOKEN) is missing."""


# ---------------------------------------------------------------------------
# MP3 encode + integrity probe
# ---------------------------------------------------------------------------

def encode_wav_to_mp3(wav_path: Path, mp3_path: Path, *, bitrate: int = MP3_BITRATE_BPS) -> None:
    """Transcode a mono PCM WAV to mono MP3 via PyAV. Raises PodcastEncodeError
    on any failure — an unavailable/broken encoder must never silently ship a
    WAV-in-MP3 envelope (R2).

    Writes to a `.tmp` sibling and renames into place only after a full,
    successful encode — write-then-read ordering, so `mp3_path` never exists
    in a half-written state (gate refinement).
    """
    try:
        import av
    except ImportError as e:
        raise PodcastEncodeError(f"PyAV not available — cannot encode MP3: {e}") from e

    tmp_path = mp3_path.with_name(mp3_path.name + ".tmp")
    in_container = None
    out_container = None
    try:
        in_container = av.open(str(wav_path))
        in_stream = in_container.streams.audio[0]

        out_container = av.open(str(tmp_path), mode="w", format="mp3")
        out_stream = out_container.add_stream("mp3", rate=in_stream.rate)
        out_stream.layout = "mono"
        out_stream.bit_rate = bitrate

        resampler = av.AudioResampler(format="s16p", layout="mono", rate=in_stream.rate)

        for frame in in_container.decode(in_stream):
            frame.pts = None
            for rframe in resampler.resample(frame):
                for packet in out_stream.encode(rframe):
                    out_container.mux(packet)
        for packet in out_stream.encode(None):
            out_container.mux(packet)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise PodcastEncodeError(f"mp3 encode failed ({wav_path} -> {mp3_path}): {e}") from e
    finally:
        if out_container is not None:
            out_container.close()
        if in_container is not None:
            in_container.close()

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise PodcastEncodeError(f"mp3 encode produced no output for {wav_path}")

    tmp_path.rename(mp3_path)


def probe_mp3(path: Path) -> float | None:
    """PyAV-probe an MP3 for container/codec validity and full decodability.

    Returns the duration in seconds on success, None on any failure
    (missing file, corrupt/half-written, wrong codec, empty decode). Never
    raises — callers treat None as "exclude from the feed" (gate refinement:
    never enclose a half-written or corrupt file).
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        import av

        container = av.open(str(path))
        try:
            streams = container.streams.audio
            # container.format.name (not codec_context.name — that's the
            # decoder variant, e.g. "mp3float") is the actual container/codec
            # check.
            if not streams or container.format.name != "mp3":
                return None
            frame_count = 0
            for _ in container.decode(streams[0]):
                frame_count += 1
            if frame_count == 0:
                return None
            duration_us = container.duration
            if not duration_us or duration_us <= 0:
                return None
            return duration_us / 1_000_000
        finally:
            container.close()
    except Exception as e:
        logger.warning("tts_feed: probe failed for %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Episode file bookkeeping (stateless — always re-derived from the directory)
# ---------------------------------------------------------------------------

_EPISODE_MP3_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}-\d{4})-(?P<period>[a-z]+)\.(?P<engine>[a-z]+)\.mp3$'
)
_EPISODE_WAV_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}-\d{4})-(?P<period>[a-z]+)\.(?P<engine>[a-z]+)\.wav$'
)


@dataclass
class EpisodeFile:
    path: Path
    ts: datetime
    period: str
    engine: str


def _parse_ts(ts_str: str) -> datetime | None:
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d-%H%M").replace(tzinfo=PACIFIC)
    except ValueError:
        return None


def _list_period_files(
    audio_dir: Path, period: str, pattern: re.Pattern, extension: str
) -> list[EpisodeFile]:
    files = []
    for p in sorted(audio_dir.glob(f"*-{period}.*.{extension}")):
        m = pattern.match(p.name)
        if not m or m.group("period") != period:
            continue
        ts = _parse_ts(m.group("ts"))
        if ts is None:
            continue
        files.append(EpisodeFile(path=p, ts=ts, period=period, engine=m.group("engine")))
    files.sort(key=lambda e: e.ts, reverse=True)
    return files


def list_period_mp3s(audio_dir: Path, period: str) -> list[EpisodeFile]:
    return _list_period_files(audio_dir, period, _EPISODE_MP3_RE, "mp3")


def list_period_wavs(audio_dir: Path, period: str) -> list[EpisodeFile]:
    return _list_period_files(audio_dir, period, _EPISODE_WAV_RE, "wav")


def prune_episodes(audio_dir: Path, period: str, *, keep: int = RETENTION_KEEP) -> list[Path]:
    """Delete morning MP3 episodes beyond the newest `keep` (by parsed
    filename timestamp — file-existence based, never a fragile age
    computation), and all but the single newest WAV for the period.

    Only ever deletes files this module can positively identify by its own
    naming convention — never touches unrelated files in the directory.
    """
    deleted: list[Path] = []

    mp3s = list_period_mp3s(audio_dir, period)
    for e in mp3s[keep:]:
        try:
            e.path.unlink()
            deleted.append(e.path)
        except OSError as ex:
            logger.warning("tts_feed: prune failed to delete %s: %s", e.path, ex)

    wavs = list_period_wavs(audio_dir, period)
    for e in wavs[1:]:
        try:
            e.path.unlink()
            deleted.append(e.path)
        except OSError as ex:
            logger.warning("tts_feed: prune failed to delete %s: %s", e.path, ex)

    return deleted


# ---------------------------------------------------------------------------
# Cover art (lazy, one-time placeholder — Pillow already installed)
# ---------------------------------------------------------------------------

def _ensure_cover_image(audio_dir: Path) -> None:
    cover_path = audio_dir / "cover.jpg"
    if cover_path.exists():
        return
    try:
        from PIL import Image
    except ImportError:
        logger.warning("tts_feed: Pillow unavailable — skipping cover.jpg placeholder")
        return
    tmp_path = cover_path.with_name(cover_path.name + ".tmp")
    img = Image.new("RGB", (1400, 1400), color=(46, 74, 58))
    img.save(tmp_path, "JPEG", quality=90)
    tmp_path.rename(cover_path)


# ---------------------------------------------------------------------------
# RSS 2.0 feed generation (stdlib ElementTree — zero new dep)
# ---------------------------------------------------------------------------

def _itunes(tag: str) -> str:
    return f"{{{ITUNES_NS}}}{tag}"


def _require_base_url() -> str:
    base = PODCAST_BASE_URL.rstrip("/") if PODCAST_BASE_URL else ""
    if not base:
        raise PodcastConfigError(
            "PODCAST_BASE_URL is not configured (and PODCAST_FEED_TOKEN is unset, so no "
            "default could be derived) — set PODCAST_FEED_TOKEN or PODCAST_BASE_URL before publishing"
        )
    return base


def _duration_str(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _episode_title(e: EpisodeFile) -> str:
    label = _ENGINE_ITEM_LABEL.get(e.engine, e.engine.title())
    return f"{CHANNEL_TITLE} — {e.ts.strftime('%a, %b %-d, %Y')} ({label})"


def _episode_description(e: EpisodeFile) -> str:
    label = _ENGINE_ITEM_LABEL.get(e.engine, e.engine.title())
    return f"Spoken rendition of the {e.period} Gardener brief, synthesized via {label}."


def regenerate_feed(audio_dir: Path, period: str) -> tuple[Path, str]:
    """Rebuild feed.xml from scratch from the current (already-pruned)
    directory contents — stateless, ordering by parsed filename timestamp
    (never raw glob order), each candidate PyAV-probed before inclusion.
    """
    base_url = _require_base_url()
    _ensure_cover_image(audio_dir)

    candidates = list_period_mp3s(audio_dir, period)
    episodes: list[tuple[EpisodeFile, float]] = []
    for e in candidates:
        duration = probe_mp3(e.path)
        if duration is None:
            logger.warning("tts_feed: excluding %s from feed — failed integrity probe", e.path)
            continue
        episodes.append((e, duration))

    # xmlns:itunes is emitted automatically by ET's serializer because of the
    # register_namespace() call above + the {ITUNES_NS}tag-qualified children
    # below — declaring it again here would duplicate the attribute.
    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = CHANNEL_TITLE
    ET.SubElement(channel, "link").text = f"{base_url}/"
    ET.SubElement(channel, "description").text = CHANNEL_DESCRIPTION
    ET.SubElement(channel, "language").text = "en-us"
    ET.SubElement(channel, _itunes("author")).text = CHANNEL_AUTHOR
    ET.SubElement(channel, _itunes("image"), {"href": f"{base_url}/cover.jpg"})
    ET.SubElement(channel, _itunes("category"), {"text": CHANNEL_CATEGORY})
    ET.SubElement(channel, _itunes("explicit")).text = "false"

    for e, duration in episodes:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = _episode_title(e)
        ET.SubElement(item, "description").text = _episode_description(e)
        ET.SubElement(item, "enclosure", {
            "url": f"{base_url}/{e.path.name}",
            "length": str(e.path.stat().st_size),
            "type": "audio/mpeg",
        })
        ET.SubElement(item, "guid", {"isPermaLink": "false"}).text = e.path.stem
        ET.SubElement(item, "pubDate").text = format_datetime(e.ts)
        ET.SubElement(item, _itunes("duration")).text = _duration_str(duration)

    feed_path = audio_dir / "feed.xml"
    tmp_path = feed_path.with_name(feed_path.name + ".tmp")
    tree = ET.ElementTree(rss)
    ET.indent(tree, space="  ")
    tree.write(tmp_path, encoding="UTF-8", xml_declaration=True)
    tmp_path.rename(feed_path)

    return feed_path, f"{base_url}/feed.xml"


# ---------------------------------------------------------------------------
# Publish orchestration
# ---------------------------------------------------------------------------

_BRIEF_STEM_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}-\d{4})-(?:morning|afternoon|weekly|live)$'
)


def _resolve_brief_timestamp(brief_path: Path, period: str) -> datetime:
    """Identity timestamp for this brief's content (not the live publish-time
    clock) — the basis for the episode basename/GUID, so re-publishing the
    same brief is idempotent rather than minting a new episode each run.

    Falls back to the current time if the symlink target doesn't match the
    expected `<ts>-<period>.md` shape (defensive, fail-open like the rest of
    this arc's target-id/timestamp handling).
    """
    try:
        if brief_path.is_symlink():
            stem = Path(os.readlink(brief_path)).stem
        else:
            stem = brief_path.stem
        m = _BRIEF_STEM_RE.match(stem)
        if m:
            ts = _parse_ts(m.group("ts"))
            if ts is not None:
                return ts
    except OSError as e:
        logger.warning("tts_feed: could not resolve brief timestamp for %s: %s", brief_path, e)
    return datetime.now(tz=PACIFIC)


def _find_existing_mp3(audio_dir: Path, basename: str) -> Path | None:
    matches = sorted(audio_dir.glob(f"{basename}.*.mp3"))
    return matches[0] if matches else None


@dataclass
class PublishResult:
    ok: bool
    period: str
    mp3_path: Path | None = None
    engine: str | None = None
    feed_path: Path | None = None
    feed_url: str | None = None
    pruned: list[Path] | None = None
    skipped_idempotent: bool = False
    error: str | None = None


def publish_episode(period: str = "morning") -> PublishResult:
    """R1 — render (Kokoro-primary, Piper-automatic-fallback), MP3-encode,
    regenerate feed.xml statelessly, prune to RETENTION_KEEP. Idempotent —
    re-running against an unchanged brief no-ops the render/encode and just
    regenerates the feed from disk.
    """
    from agents_core.room_paths import room_path

    briefs_root = room_path("briefs")
    audio_dir = briefs_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    brief_path = briefs_root / f"latest-{period}.md"
    body = brief_path.read_text(encoding="utf-8") if brief_path.exists() else ""
    if not body.strip():
        msg = f"no {period} brief to publish — run `lapis-pm brief --period {period}` first"
        logger.error("tts_feed: %s", msg)
        return PublishResult(ok=False, period=period, error=msg)

    brief_dt = _resolve_brief_timestamp(brief_path, period)
    ts = brief_dt.strftime("%Y-%m-%d-%H%M")
    basename = f"{ts}-{period}"

    existing = _find_existing_mp3(audio_dir, basename)
    if existing is not None and probe_mp3(existing) is not None:
        m = _EPISODE_MP3_RE.match(existing.name)
        engine = m.group("engine") if m else None
        pruned = prune_episodes(audio_dir, period)
        try:
            feed_path, feed_url = regenerate_feed(audio_dir, period)
        except PodcastConfigError as e:
            return PublishResult(ok=False, period=period, error=str(e))
        return PublishResult(
            ok=True, period=period, mp3_path=existing, engine=engine,
            feed_path=feed_path, feed_url=feed_url, pruned=pruned, skipped_idempotent=True,
        )

    wav_path: Path | None = None
    engine: str | None = None
    kokoro_error: str | None = None
    piper_error: str | None = None

    kokoro_outcome = tts_episode.run_episode_bakeoff(
        body, voices=["kokoro"], out_dir=audio_dir, period=period, now=brief_dt,
    )
    if kokoro_outcome.ok:
        wav_path = kokoro_outcome.succeeded[0].path
        engine = "kokoro"
    else:
        kokoro_error = next(
            (e.error or ("soft-failed" if e.soft_failed else "unknown") for e in kokoro_outcome.engines),
            "unknown",
        )
        piper_outcome = tts_episode.run_episode_bakeoff(
            body, voices=["piper"], out_dir=audio_dir, period=period, now=brief_dt,
        )
        if piper_outcome.ok:
            wav_path = piper_outcome.succeeded[0].path
            engine = "piper"
        else:
            piper_error = next(
                (e.error or ("soft-failed" if e.soft_failed else "unknown") for e in piper_outcome.engines),
                "unknown",
            )

    if wav_path is None or engine is None:
        msg = f"both engines failed — kokoro: {kokoro_error}; piper: {piper_error}"
        logger.error("tts_feed: %s", msg)
        send_notification(
            message=f"Gardener TTS feed publish FAILED for {period} — no episode delivered.\n{msg}",
            title="Gardener TTS feed — BOTH engines failed",
            source="tts-episode-publish",
        )
        return PublishResult(ok=False, period=period, error=msg)

    mp3_path = wav_path.with_name(f"{wav_path.stem}.mp3")
    try:
        encode_wav_to_mp3(wav_path, mp3_path)
        duration = probe_mp3(mp3_path)
        if duration is None:
            raise PodcastEncodeError("encoded mp3 failed post-encode integrity probe")
    except PodcastEncodeError as e:
        mp3_path.unlink(missing_ok=True)
        msg = str(e)
        logger.error("tts_feed: %s", msg)
        send_notification(
            message=f"Gardener TTS feed publish FAILED for {period} — {engine} rendered a WAV but MP3 encoding failed.\n{msg}",
            title="Gardener TTS feed — MP3 encode failed",
            source="tts-episode-publish",
        )
        return PublishResult(ok=False, period=period, engine=engine, error=msg)

    pruned = prune_episodes(audio_dir, period)
    try:
        feed_path, feed_url = regenerate_feed(audio_dir, period)
    except PodcastConfigError as e:
        msg = str(e)
        logger.error("tts_feed: %s", msg)
        send_notification(
            message=f"Gardener TTS feed publish FAILED for {period} — episode encoded but feed config is missing.\n{msg}",
            title="Gardener TTS feed — config error",
            source="tts-episode-publish",
        )
        return PublishResult(ok=False, period=period, engine=engine, mp3_path=mp3_path, error=msg)

    label = _ENGINE_ITEM_LABEL.get(engine, engine.title())
    send_notification(
        message=(
            f"Episode published via {label}: {mp3_path.name}\n"
            f"Feed: {feed_url}\n"
            f"Pruned: {len(pruned)} old file(s)"
        ),
        title="Gardener TTS feed — episode published",
        source="tts-episode-publish",
    )

    return PublishResult(
        ok=True, period=period, mp3_path=mp3_path, engine=engine,
        feed_path=feed_path, feed_url=feed_url, pruned=pruned, skipped_idempotent=False,
    )
