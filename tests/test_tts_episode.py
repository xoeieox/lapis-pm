"""Tests for lapis_pm.tts_episode (gardener-tts-episode-v0, Unit 3a).

Coverage:
  - normalize_for_speech: markup stripping + the warning-impact invariant
    (Council: `[Critical]`/`[Warning]` isolated declarative paragraphs;
    climate/narrative flows).
  - synthesize_episode: unknown-voice guard, Piper missing-model failure,
    a real Piper cold-render smoke test (integration-marked, skipped if
    piper-tts isn't provisioned).
  - _synthesize_kokoro: DoormanClient acquire→guard-serving→synth→release
    ordering, release-in-finally on both not-serving and synth failure.
  - run_episode_bakeoff: per-engine fallback, Piper latency soft-fail,
    both-engines-fail (no Pushover), Pushover message template.
  - cmd_tts_episode: no-brief-exists exits clean with no partial Pushover.
"""

from __future__ import annotations

import argparse
import textwrap
from unittest.mock import MagicMock

import pytest

from lapis_pm import tts_episode


# ---------------------------------------------------------------------------
# normalize_for_speech
# ---------------------------------------------------------------------------

FIXTURE_BODY = textwrap.dedent("""\
    # Lapis Morning Brief — 2026-07-22 08:00 PT

    ## Built (since 2026-07-21 08:00 PT)
    - Shipped the arc-Climate reconciler (evidence: /srv/lapis/lapis-state/gardener-arc-climate-v0.md)
    - arc-doc: gardener-arc-climate-v0.md

    ## Climate
    - gardener-tts-episode-v0: 3d of silence since 'land the fixer PR'. The thread has gone slack.
    - chain: gardener-podcast-briefing-arc

    ## Gardener Cross-Cutting Observations
    - [Critical] The PM state machine is stuck in a stale-signal deadlock  (evidence: pm/cursor/foo-v0)
    - [Warning] Three Weaver threads are circling the same friction point
    """)


class TestNormalizeForSpeech:
    def test_clean_spoken_prose_zero_markup(self):
        result = tts_episode.normalize_for_speech(FIXTURE_BODY)
        for marker in ("#", "**", "`", "(evidence:", "arc-doc:", "chain:"):
            assert marker not in result, f"leftover markup {marker!r} in: {result!r}"
        assert "/srv/lapis/" not in result

    def test_critical_item_is_isolated_declarative_paragraph(self):
        """Council invariant: [Critical] never merges into surrounding prose."""
        result = tts_episode.normalize_for_speech(FIXTURE_BODY)
        paragraphs = result.split("\n\n")
        critical_paragraphs = [p for p in paragraphs if p.startswith("Critical:")]
        assert len(critical_paragraphs) == 1, f"expected exactly one Critical paragraph, got: {paragraphs}"
        critical = critical_paragraphs[0]
        assert critical == "Critical: The PM state machine is stuck in a stale-signal deadlock."
        # Not merged with the following Warning item or anything else.
        assert "Warning" not in critical

    def test_warning_item_is_isolated_declarative_paragraph(self):
        result = tts_episode.normalize_for_speech(FIXTURE_BODY)
        paragraphs = result.split("\n\n")
        warning_paragraphs = [p for p in paragraphs if p.startswith("Warning:")]
        assert len(warning_paragraphs) == 1
        assert warning_paragraphs[0] == "Warning: Three Weaver threads are circling the same friction point."

    def test_climate_narrative_flows_not_isolated(self):
        """Climate/narrative bullets merge into one flowing paragraph (not isolated)."""
        result = tts_episode.normalize_for_speech(FIXTURE_BODY)
        paragraphs = result.split("\n\n")
        climate_paragraphs = [p for p in paragraphs if "gone slack" in p]
        assert len(climate_paragraphs) == 1
        # The chain: bullet flowed into the SAME paragraph as the climate bullet
        # (both were consecutive list items with no blank line between them).
        assert "gardener-podcast-briefing-arc" in climate_paragraphs[0]

    def test_headers_become_own_paragraph(self):
        result = tts_episode.normalize_for_speech(FIXTURE_BODY)
        paragraphs = result.split("\n\n")
        assert "Lapis Morning Brief — 2026-07-22 08:00 PT" in paragraphs
        assert any(p.startswith("Built (since") for p in paragraphs)

    def test_paragraph_breaks_preserved(self):
        result = tts_episode.normalize_for_speech(FIXTURE_BODY)
        # Multiple distinct paragraphs, not one giant flattened blob.
        assert len(result.split("\n\n")) >= 5

    def test_empty_body_returns_empty(self):
        assert tts_episode.normalize_for_speech("") == ""

    def test_case_insensitive_warning_tag(self):
        result = tts_episode.normalize_for_speech("- [critical] lowercase tag test")
        assert result == "Critical: lowercase tag test."


# ---------------------------------------------------------------------------
# synthesize_episode — dispatch + Piper
# ---------------------------------------------------------------------------

class TestSynthesizeEpisodeDispatch:
    def test_unknown_voice_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="unknown voice"):
            tts_episode.synthesize_episode("hello", voice="festival", out_dir=tmp_path)

    def test_piper_missing_model_raises_engine_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_episode, "PIPER_MODEL_PATH", tmp_path / "nonexistent.onnx")
        with pytest.raises(tts_episode.TTSEngineError, match="provision_tts.sh"):
            tts_episode.synthesize_episode("hello", voice="piper", out_dir=tmp_path)


_WAV_MAGIC = b"RIFF"
_WAV_FORMAT = b"WAVE"


@pytest.mark.integration
class TestPiperColdRenderSmoke:
    """Real Piper subprocess render — skipped unless piper-tts + a voice model
    are actually provisioned (see scripts/provision_tts.sh)."""

    def test_piper_smoke_produces_valid_wav(self, tmp_path):
        if not tts_episode.PIPER_MODEL_PATH.exists():
            pytest.skip("piper voice model not provisioned — run scripts/provision_tts.sh")
        import shutil
        if shutil.which(tts_episode.PIPER_BIN) is None:
            pytest.skip("piper binary not on PATH — run scripts/provision_tts.sh")

        out_path = tts_episode.synthesize_episode(
            "Critical: this is a short smoke-test fixture.",
            voice="piper",
            out_dir=tmp_path,
            filename="smoke.piper.wav",
        )
        assert out_path is not None
        data = out_path.read_bytes()
        assert data[:4] == _WAV_MAGIC
        assert data[8:12] == _WAV_FORMAT
        assert len(data) > 44  # more than just a WAV header


# ---------------------------------------------------------------------------
# _synthesize_kokoro — DoormanClient boundary
# ---------------------------------------------------------------------------

def _make_fake_client(order, *, acquire_status="serving"):
    client = MagicMock()

    def _acquire(*a, **kw):
        order.append("acquire")
        return {"status": acquire_status}

    def _release(*a, **kw):
        order.append("release")

    def _close():
        order.append("close")

    client.acquire.side_effect = _acquire
    client.release.side_effect = _release
    client.close.side_effect = _close
    return client


class TestSynthesizeKokoroDoormanBoundary:
    def test_acquire_guard_synth_release_ordering_on_success(self, tmp_path, monkeypatch):
        order: list[str] = []
        fake_client = _make_fake_client(order)
        monkeypatch.setattr("agents_core.doorman_client.DoormanClient", lambda *a, **kw: fake_client)

        out_path = tmp_path / "episode.kokoro.wav"

        def _fake_run(cmd, **kwargs):
            if cmd[0] == "scp" and str(out_path) == cmd[-1]:
                order.append("scp_down")
                out_path.write_bytes(_WAV_MAGIC + b"....")
            elif cmd[0] == "scp":
                order.append("scp_up")
            elif cmd[0] == "ssh" and "rm" in cmd:
                order.append("ssh_cleanup")
            elif cmd[0] == "ssh":
                order.append("ssh_synth")
            return MagicMock(returncode=0, stderr=b"")

        monkeypatch.setattr(tts_episode.subprocess, "run", _fake_run)

        tts_episode._synthesize_kokoro("hello world", out_path, work_id="test-wid")

        # acquire -> (implicit status guard) -> synth (scp up, ssh synth, scp down) -> release -> close
        assert order.index("acquire") < order.index("scp_up") < order.index("ssh_synth") < order.index("scp_down")
        assert order.index("scp_down") < order.index("release") < order.index("close")

    def test_not_serving_raises_and_skips_release_but_still_closes(self, tmp_path, monkeypatch):
        order: list[str] = []
        fake_client = _make_fake_client(order, acquire_status="wake_failed")
        monkeypatch.setattr("agents_core.doorman_client.DoormanClient", lambda *a, **kw: fake_client)

        with pytest.raises(tts_episode.TTSEngineError, match="not serving"):
            tts_episode._synthesize_kokoro("hello", tmp_path / "out.wav", work_id="wid")

        assert "release" not in order
        assert "close" in order

    def test_synth_failure_still_releases_and_closes(self, tmp_path, monkeypatch):
        order: list[str] = []
        fake_client = _make_fake_client(order)
        monkeypatch.setattr("agents_core.doorman_client.DoormanClient", lambda *a, **kw: fake_client)

        def _fake_run(cmd, **kwargs):
            if cmd[0] == "ssh" and "rm" not in cmd:
                return MagicMock(returncode=1, stderr=b"kokoro synth exploded")
            return MagicMock(returncode=0, stderr=b"")

        monkeypatch.setattr(tts_episode.subprocess, "run", _fake_run)

        with pytest.raises(tts_episode.TTSEngineError, match="kokoro synth failed"):
            tts_episode._synthesize_kokoro("hello", tmp_path / "out.wav", work_id="wid")

        assert "release" in order
        assert "close" in order
        assert order.index("release") < order.index("close")

    def test_doorman_unreachable_raises_engine_error(self, tmp_path, monkeypatch):
        from agents_core.doorman_client import DoormanUnreachable

        order: list[str] = []
        fake_client = _make_fake_client(order)
        fake_client.acquire.side_effect = DoormanUnreachable("connection refused")
        monkeypatch.setattr("agents_core.doorman_client.DoormanClient", lambda *a, **kw: fake_client)

        with pytest.raises(tts_episode.TTSEngineError, match="doorman unreachable"):
            tts_episode._synthesize_kokoro("hello", tmp_path / "out.wav", work_id="wid")

        assert "release" not in order
        assert "close" in order


# ---------------------------------------------------------------------------
# run_episode_bakeoff — per-engine fallback, latency soft-fail, Pushover
# ---------------------------------------------------------------------------

def _fake_success(voice_that_succeeds, contents=b"RIFF....WAVEfake"):
    def _synth(body, *, voice, out_dir, filename=None, work_id=None):
        if voice != voice_that_succeeds:
            raise tts_episode.TTSEngineError(f"{voice}: simulated failure")
        out_path = out_dir / (filename or f"{voice}.wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(contents)
        return out_path
    return _synth


class TestRunEpisodeBakeoffFallback:
    def test_piper_fails_kokoro_still_delivered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_episode, "synthesize_episode", _fake_success("kokoro"))
        sent = {}

        def _fake_notify(*, message, title, source):
            sent["title"] = title
            sent["message"] = message
            return True

        monkeypatch.setattr(tts_episode, "send_notification", _fake_notify)

        outcome = tts_episode.run_episode_bakeoff(
            "brief body", voices=["piper", "kokoro"], out_dir=tmp_path,
        )

        assert outcome.ok
        piper_result = next(e for e in outcome.engines if e.voice == "piper")
        kokoro_result = next(e for e in outcome.engines if e.voice == "kokoro")
        assert piper_result.path is None
        assert piper_result.error is not None
        assert kokoro_result.path is not None
        assert outcome.notified is True
        assert sent["title"] == "Gardener TTS bakeoff — Kokoro only"
        assert "piper failed" in sent["message"].lower() or "failed" in sent["message"].lower()

    def test_both_engines_fail_no_pushover(self, tmp_path, monkeypatch):
        def _always_fail(body, *, voice, out_dir, filename=None, work_id=None):
            raise tts_episode.TTSEngineError(f"{voice}: nope")

        monkeypatch.setattr(tts_episode, "synthesize_episode", _always_fail)
        notify_mock = MagicMock()
        monkeypatch.setattr(tts_episode, "send_notification", notify_mock)

        outcome = tts_episode.run_episode_bakeoff(
            "brief body", voices=["piper", "kokoro"], out_dir=tmp_path,
        )

        assert not outcome.ok
        assert outcome.notified is False
        notify_mock.assert_not_called()


class TestPiperLatencySoftFail:
    def test_piper_over_45s_withheld_kokoro_still_delivered(self, tmp_path, monkeypatch):
        real_synth = _fake_success("kokoro")

        def _slow_piper(body, *, voice, out_dir, filename=None, work_id=None):
            if voice == "piper":
                out_path = out_dir / (filename or "piper.wav")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"RIFF....WAVEfake")
                return out_path
            return real_synth(body, voice=voice, out_dir=out_dir, filename=filename, work_id=work_id)

        monkeypatch.setattr(tts_episode, "synthesize_episode", _slow_piper)

        # Simulate elapsed wall-clock without an actual sleep: monotonic()
        # advances by 50s across the piper leg's start/end reads only.
        calls = {"n": 0}
        real_monotonic = tts_episode.time.monotonic

        def _fake_monotonic():
            calls["n"] += 1
            # First two monotonic() calls bracket the piper leg (>45s apart);
            # subsequent calls (kokoro leg) behave normally/fast.
            if calls["n"] == 1:
                return 1000.0
            if calls["n"] == 2:
                return 1052.0
            return real_monotonic()

        monkeypatch.setattr(tts_episode.time, "monotonic", _fake_monotonic)
        monkeypatch.setattr(tts_episode, "send_notification", lambda **kw: True)

        outcome = tts_episode.run_episode_bakeoff(
            "brief body", voices=["piper", "kokoro"], out_dir=tmp_path,
        )

        piper_result = next(e for e in outcome.engines if e.voice == "piper")
        kokoro_result = next(e for e in outcome.engines if e.voice == "kokoro")
        assert piper_result.soft_failed is True
        assert piper_result.path is None
        assert piper_result.elapsed_sec == pytest.approx(52.0)
        assert kokoro_result.path is not None
        assert outcome.ok

    def test_pushover_message_shows_withheld_line(self, tmp_path, monkeypatch):
        outcome = tts_episode.BakeoffOutcome(engines=[
            tts_episode.EngineOutcome(voice="piper", path=None, elapsed_sec=52.3, soft_failed=True),
            tts_episode.EngineOutcome(voice="kokoro", path=tmp_path / "k.wav", elapsed_sec=3.0),
        ])
        title, message = tts_episode._build_pushover_message(outcome)
        assert title == "Gardener TTS bakeoff — Kokoro only"
        assert "withheld" in message
        assert "52.3" in message
        assert "Which voice?" in message


# ---------------------------------------------------------------------------
# cmd_tts_episode — no-brief-exists
# ---------------------------------------------------------------------------

class TestCmdTTSEpisodeNoBrief:
    def test_no_brief_exists_exits_nonzero_no_pushover(self, tmp_path, monkeypatch, capsys):
        from lapis_pm import cli as cli_mod

        empty_briefs_root = tmp_path / "briefs"
        empty_briefs_root.mkdir()
        monkeypatch.setattr("agents_core.room_paths.room_path", lambda key, *p: empty_briefs_root if key == "briefs" else tmp_path)

        notify_mock = MagicMock()
        monkeypatch.setattr(tts_episode, "send_notification", notify_mock)
        run_mock = MagicMock()
        monkeypatch.setattr(tts_episode, "run_episode_bakeoff", run_mock)

        args = argparse.Namespace(period="morning", voice="both")
        rc = cli_mod.cmd_tts_episode(args)

        assert rc != 0
        run_mock.assert_not_called()
        notify_mock.assert_not_called()
        captured = capsys.readouterr()
        assert "no morning brief" in captured.err
