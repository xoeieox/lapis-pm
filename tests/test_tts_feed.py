"""Tests for lapis_pm.tts_feed (Unit 3b, gardener-tts-feed-v0).

Coverage:
  - encode_wav_to_mp3 / probe_mp3: real PyAV round-trip, corrupt-file
    rejection, half-written-file rejection (integration-marked — needs the
    real PyAV/libmp3lame encoder).
  - prune_episodes: file-existence-based retention (keep newest N MP3s +
    newest 1 WAV), never touches files outside its own naming convention.
  - regenerate_feed: stateless rebuild, timestamp ordering (not glob order),
    XML-escaping, excludes probe-failed candidates (integration-marked).
  - publish_episode: Kokoro-primary/Piper-automatic-fallback sequencing
    (never both engines rendered), idempotent re-run (no re-render), both-
    engines-fail loud Pushover, encode-failure loud Pushover.
"""

from __future__ import annotations

import wave
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lapis_pm import tts_episode, tts_feed

PACIFIC = ZoneInfo("America/Los_Angeles")


def _write_dummy_wav(path: Path, seconds: float = 0.5, rate: int = 22050) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


# ---------------------------------------------------------------------------
# encode_wav_to_mp3 / probe_mp3
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEncodeAndProbe:
    def test_round_trip_produces_probeable_mp3(self, tmp_path):
        wav_path = tmp_path / "x.wav"
        _write_dummy_wav(wav_path, seconds=1.0)
        mp3_path = tmp_path / "x.mp3"

        tts_feed.encode_wav_to_mp3(wav_path, mp3_path)

        assert mp3_path.exists()
        duration = tts_feed.probe_mp3(mp3_path)
        assert duration is not None
        assert duration == pytest.approx(1.0, abs=0.2)

    def test_encode_never_leaves_a_tmp_file_behind(self, tmp_path):
        wav_path = tmp_path / "x.wav"
        _write_dummy_wav(wav_path)
        mp3_path = tmp_path / "x.mp3"
        tts_feed.encode_wav_to_mp3(wav_path, mp3_path)
        assert not (tmp_path / "x.mp3.tmp").exists()

    def test_encode_missing_source_raises_encode_error(self, tmp_path):
        with pytest.raises(tts_feed.PodcastEncodeError):
            tts_feed.encode_wav_to_mp3(tmp_path / "nonexistent.wav", tmp_path / "out.mp3")

    def test_probe_rejects_corrupt_file(self, tmp_path):
        bad = tmp_path / "bad.mp3"
        bad.write_bytes(b"not an mp3 at all" * 5)
        assert tts_feed.probe_mp3(bad) is None

    def test_probe_rejects_missing_file(self, tmp_path):
        assert tts_feed.probe_mp3(tmp_path / "missing.mp3") is None

    def test_probe_rejects_empty_file(self, tmp_path):
        empty = tmp_path / "empty.mp3"
        empty.touch()
        assert tts_feed.probe_mp3(empty) is None


# ---------------------------------------------------------------------------
# prune_episodes — file-existence based, naming-convention-scoped
# ---------------------------------------------------------------------------

class TestPruneEpisodes:
    def _touch_mp3(self, audio_dir: Path, ts: str, engine: str = "kokoro", period: str = "morning") -> Path:
        p = audio_dir / f"{ts}-{period}.{engine}.mp3"
        p.write_bytes(b"x")
        return p

    def _touch_wav(self, audio_dir: Path, ts: str, engine: str = "kokoro", period: str = "morning") -> Path:
        p = audio_dir / f"{ts}-{period}.{engine}.wav"
        p.write_bytes(b"x")
        return p

    def test_keeps_newest_n_by_parsed_timestamp(self, tmp_path):
        for day in range(1, 21):
            self._touch_mp3(tmp_path, f"2026-07-{day:02d}-0800")
        deleted = tts_feed.prune_episodes(tmp_path, "morning", keep=14)
        assert len(deleted) == 6
        remaining = sorted(p.name for p in tmp_path.glob("*.mp3"))
        assert len(remaining) == 14
        # Newest 14 (days 7..20) survive; oldest 6 (days 1..6) are gone.
        assert "2026-07-01-0800-morning.kokoro.mp3" not in remaining
        assert "2026-07-20-0800-morning.kokoro.mp3" in remaining

    def test_keeps_only_newest_wav(self, tmp_path):
        self._touch_wav(tmp_path, "2026-07-01-0800")
        self._touch_wav(tmp_path, "2026-07-02-0800", engine="piper")
        newest = self._touch_wav(tmp_path, "2026-07-03-0800")
        tts_feed.prune_episodes(tmp_path, "morning", keep=14)
        remaining = sorted(p.name for p in tmp_path.glob("*.wav"))
        assert remaining == [newest.name]

    def test_never_touches_files_outside_naming_convention(self, tmp_path):
        odd = tmp_path / "demo-humanized.kokoro.wav"
        odd.write_bytes(b"x")
        odder = tmp_path / "2026-07-23-0810-morning.kokoro-humanized-fallback.wav"
        odder.write_bytes(b"x")
        self._touch_mp3(tmp_path, "2026-07-01-0800")
        tts_feed.prune_episodes(tmp_path, "morning", keep=14)
        assert odd.exists()
        assert odder.exists()

    def test_different_period_untouched(self, tmp_path):
        afternoon = self._touch_mp3(tmp_path, "2026-07-01-1300", period="afternoon")
        for day in range(1, 20):
            self._touch_mp3(tmp_path, f"2026-07-{day:02d}-0800")
        tts_feed.prune_episodes(tmp_path, "morning", keep=14)
        assert afternoon.exists()


# ---------------------------------------------------------------------------
# regenerate_feed
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRegenerateFeed:
    def _make_episode(self, audio_dir: Path, ts: str, engine: str = "kokoro", period: str = "morning") -> Path:
        wav_path = audio_dir / f"{ts}-{period}.{engine}.wav"
        _write_dummy_wav(wav_path, seconds=0.3)
        mp3_path = wav_path.with_name(f"{wav_path.stem}.mp3")
        tts_feed.encode_wav_to_mp3(wav_path, mp3_path)
        wav_path.unlink()
        return mp3_path

    def test_requires_base_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_feed, "PODCAST_BASE_URL", "")
        with pytest.raises(tts_feed.PodcastConfigError):
            tts_feed.regenerate_feed(tmp_path, "morning")

    def test_items_ordered_newest_first_by_parsed_timestamp_not_glob(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_feed, "PODCAST_BASE_URL", "https://example.ts.net/gardener-tok")
        # Create out of lexical/glob order on purpose.
        self._make_episode(tmp_path, "2026-07-05-0800")
        self._make_episode(tmp_path, "2026-07-20-0800")
        self._make_episode(tmp_path, "2026-07-01-0800")

        feed_path, feed_url = tts_feed.regenerate_feed(tmp_path, "morning")
        import xml.etree.ElementTree as ET
        root = ET.fromstring(feed_path.read_text())
        guids = [item.find("guid").text for item in root.find("channel").findall("item")]
        assert guids == [
            "2026-07-20-0800-morning.kokoro",
            "2026-07-05-0800-morning.kokoro",
            "2026-07-01-0800-morning.kokoro",
        ]
        assert feed_url == "https://example.ts.net/gardener-tok/feed.xml"

    def test_excludes_corrupt_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_feed, "PODCAST_BASE_URL", "https://example.ts.net/gardener-tok")
        good = self._make_episode(tmp_path, "2026-07-05-0800")
        bad = tmp_path / "2026-07-06-0800-morning.kokoro.mp3"
        bad.write_bytes(b"not an mp3")

        feed_path, _ = tts_feed.regenerate_feed(tmp_path, "morning")
        text = feed_path.read_text()
        assert good.name in text
        assert bad.name not in text

    def test_xml_special_characters_are_escaped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_feed, "PODCAST_BASE_URL", "https://example.ts.net/gardener-tok")
        monkeypatch.setattr(tts_feed, "CHANNEL_TITLE", "Gardener & <Friends>")
        self._make_episode(tmp_path, "2026-07-05-0800")

        feed_path, _ = tts_feed.regenerate_feed(tmp_path, "morning")
        raw = feed_path.read_bytes()
        assert b"Gardener & <Friends>" not in raw
        assert b"Gardener &amp; &lt;Friends&gt;" in raw
        # And it still parses as well-formed XML.
        import xml.etree.ElementTree as ET
        ET.fromstring(feed_path.read_text())


# ---------------------------------------------------------------------------
# publish_episode — Kokoro-primary / Piper-automatic-fallback, idempotency
# ---------------------------------------------------------------------------

class TestPublishEpisode:
    def _setup_room(self, tmp_path, monkeypatch, *, brief_body="hello brief", brief_ts="2026-07-21-0800"):
        briefs_root = tmp_path / "briefs"
        daily = briefs_root / "daily"
        daily.mkdir(parents=True)
        audio_dir = briefs_root / "audio"
        audio_dir.mkdir()

        brief_file = daily / f"{brief_ts}-morning.md"
        brief_file.write_text(brief_body)
        latest = briefs_root / "latest-morning.md"
        latest.symlink_to(f"daily/{brief_ts}-morning.md")

        monkeypatch.setattr(
            "agents_core.room_paths.room_path",
            lambda key, *p: briefs_root if key == "briefs" else tmp_path,
        )
        monkeypatch.setattr(tts_feed, "PODCAST_BASE_URL", "https://example.ts.net/gardener-tok")
        monkeypatch.setattr(tts_feed, "send_notification", lambda **kw: True)
        return audio_dir

    def test_kokoro_success_never_calls_piper(self, tmp_path, monkeypatch):
        audio_dir = self._setup_room(tmp_path, monkeypatch)
        calls = []

        def fake_bakeoff(body, *, voices, out_dir, period, now=None):
            calls.append(list(voices))
            wav_path = out_dir / f"{now.strftime('%Y-%m-%d-%H%M')}-{period}.{voices[0]}.wav"
            _write_dummy_wav(wav_path, seconds=0.2)
            return tts_episode.BakeoffOutcome(engines=[
                tts_episode.EngineOutcome(voice=voices[0], path=wav_path, elapsed_sec=1.0)
            ])

        monkeypatch.setattr(tts_episode, "run_episode_bakeoff", fake_bakeoff)

        result = tts_feed.publish_episode(period="morning")

        assert result.ok
        assert result.engine == "kokoro"
        assert calls == [["kokoro"]]  # Piper never attempted
        assert result.mp3_path.exists()
        assert result.feed_path.exists()

    def test_kokoro_fails_piper_fallback_succeeds(self, tmp_path, monkeypatch):
        audio_dir = self._setup_room(tmp_path, monkeypatch)
        calls = []

        def fake_bakeoff(body, *, voices, out_dir, period, now=None):
            calls.append(list(voices))
            if voices[0] == "kokoro":
                return tts_episode.BakeoffOutcome(engines=[
                    tts_episode.EngineOutcome(voice="kokoro", path=None, elapsed_sec=1.0, error="gw unreachable")
                ])
            wav_path = out_dir / f"{now.strftime('%Y-%m-%d-%H%M')}-{period}.piper.wav"
            _write_dummy_wav(wav_path, seconds=0.2)
            return tts_episode.BakeoffOutcome(engines=[
                tts_episode.EngineOutcome(voice="piper", path=wav_path, elapsed_sec=2.0)
            ])

        monkeypatch.setattr(tts_episode, "run_episode_bakeoff", fake_bakeoff)

        result = tts_feed.publish_episode(period="morning")

        assert result.ok
        assert result.engine == "piper"
        assert calls == [["kokoro"], ["piper"]]

    def test_both_engines_fail_sends_loud_pushover_and_fails(self, tmp_path, monkeypatch):
        self._setup_room(tmp_path, monkeypatch)
        notify_calls = []
        monkeypatch.setattr(tts_feed, "send_notification", lambda **kw: notify_calls.append(kw) or True)

        def fake_bakeoff(body, *, voices, out_dir, period, now=None):
            return tts_episode.BakeoffOutcome(engines=[
                tts_episode.EngineOutcome(voice=voices[0], path=None, elapsed_sec=1.0, error="boom")
            ])

        monkeypatch.setattr(tts_episode, "run_episode_bakeoff", fake_bakeoff)

        result = tts_feed.publish_episode(period="morning")

        assert not result.ok
        assert len(notify_calls) == 1
        assert "BOTH engines failed" in notify_calls[0]["title"]

    def test_idempotent_rerun_does_not_rerender(self, tmp_path, monkeypatch):
        # brief_ts defaults to "2026-07-21" — pin "today" to match so this
        # exercises the true same-day idempotent no-op path (R5a: the only
        # case that stays silent-success).
        self._setup_room(tmp_path, monkeypatch)
        monkeypatch.setattr(tts_feed, "_today_pacific", lambda: date(2026, 7, 21))
        calls = []
        notify_calls = []
        monkeypatch.setattr(tts_feed, "send_notification", lambda **kw: notify_calls.append(kw) or True)

        def fake_bakeoff(body, *, voices, out_dir, period, now=None):
            calls.append(list(voices))
            wav_path = out_dir / f"{now.strftime('%Y-%m-%d-%H%M')}-{period}.{voices[0]}.wav"
            _write_dummy_wav(wav_path, seconds=0.2)
            return tts_episode.BakeoffOutcome(engines=[
                tts_episode.EngineOutcome(voice=voices[0], path=wav_path, elapsed_sec=1.0)
            ])

        monkeypatch.setattr(tts_episode, "run_episode_bakeoff", fake_bakeoff)

        first = tts_feed.publish_episode(period="morning")
        notify_calls.clear()
        second = tts_feed.publish_episode(period="morning")

        assert first.ok and second.ok
        assert second.skipped_idempotent
        assert calls == [["kokoro"]]  # only rendered once across both calls
        assert first.mp3_path == second.mp3_path
        assert notify_calls == []  # same-day idempotent re-run stays silent-success (R5a)

    def test_no_brief_fails_and_notifies(self, tmp_path, monkeypatch):
        briefs_root = tmp_path / "briefs"
        (briefs_root / "audio").mkdir(parents=True)
        monkeypatch.setattr(
            "agents_core.room_paths.room_path",
            lambda key, *p: briefs_root if key == "briefs" else tmp_path,
        )
        notify_calls = []
        monkeypatch.setattr(tts_feed, "send_notification", lambda **kw: notify_calls.append(kw) or True)

        result = tts_feed.publish_episode(period="morning")

        assert not result.ok
        assert "no morning brief" in result.error
        # R5a: a run that produces no new episode for today must notify —
        # the empty-brief case is one of the two covered branches.
        assert len(notify_calls) == 1
        assert "no brief to publish" in notify_calls[0]["title"].lower()

    def test_stale_idempotent_skip_notifies(self, tmp_path, monkeypatch):
        # latest-morning.md resolves to 2026-07-21, but "today" is 2026-07-23
        # (the upstream brief job stalled) — a re-run against this unchanged,
        # already-published stale brief must notify every time, not just once.
        self._setup_room(tmp_path, monkeypatch, brief_ts="2026-07-21-0800")
        monkeypatch.setattr(tts_feed, "_today_pacific", lambda: date(2026, 7, 23))
        notify_calls = []
        monkeypatch.setattr(tts_feed, "send_notification", lambda **kw: notify_calls.append(kw) or True)

        def fake_bakeoff(body, *, voices, out_dir, period, now=None):
            wav_path = out_dir / f"{now.strftime('%Y-%m-%d-%H%M')}-{period}.{voices[0]}.wav"
            _write_dummy_wav(wav_path, seconds=0.2)
            return tts_episode.BakeoffOutcome(engines=[
                tts_episode.EngineOutcome(voice=voices[0], path=wav_path, elapsed_sec=1.0)
            ])

        monkeypatch.setattr(tts_episode, "run_episode_bakeoff", fake_bakeoff)

        first = tts_feed.publish_episode(period="morning")  # fresh render of the stale brief
        notify_calls.clear()
        second = tts_feed.publish_episode(period="morning")  # idempotent-skip, still stale -> notify

        assert first.ok and second.ok
        assert second.skipped_idempotent
        assert len(notify_calls) == 1
        assert "stale" in notify_calls[0]["title"].lower()

        notify_calls.clear()
        third = tts_feed.publish_episode(period="morning")  # still stale on a later re-run -> notify again

        assert third.ok and third.skipped_idempotent
        assert len(notify_calls) == 1


# ---------------------------------------------------------------------------
# _resolve_brief_timestamp
# ---------------------------------------------------------------------------

class TestResolveBriefTimestamp:
    def test_resolves_from_symlink_target(self, tmp_path):
        daily = tmp_path / "daily"
        daily.mkdir()
        (daily / "2026-07-21-0800-morning.md").write_text("body")
        latest = tmp_path / "latest-morning.md"
        latest.symlink_to("daily/2026-07-21-0800-morning.md")

        dt = tts_feed._resolve_brief_timestamp(latest, "morning")
        assert dt == datetime(2026, 7, 21, 8, 0, tzinfo=PACIFIC)

    def test_falls_back_to_now_for_unrecognized_shape(self, tmp_path):
        odd = tmp_path / "latest-morning.md"
        odd.write_text("body")  # not a symlink, plain stem "latest-morning"
        dt = tts_feed._resolve_brief_timestamp(odd, "morning")
        assert dt.tzinfo is not None
