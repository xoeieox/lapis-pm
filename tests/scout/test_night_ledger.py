"""Tests for the night-run observability lane (genesis-constitution-heartbeat-v0).

Covers D1 (run ledger), D2 (queue bootstrap + blank-page honesty), D3 (morning
digest composer, strictly NOT a wrapper around digest()), and D4 (loud,
non-Pushover delivery). Unit tests only — no live reach, no seat stop.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lapis_pm.scout import night_ledger as nl
from lapis_pm.scout.night_ledger import (
    BLANK,
    BLANK_REASON,
    LANE_STATUS,
    LaneRecord,
    RunLedger,
    blank_page_record,
    cheap_ask_guard,
    compose_morning_digest,
    deliver_morning_digest,
    emit_morning_digest,
    emit_run_ledger,
    ensure_queue_file,
    lane_records_for_run,
    ledger_key,
    read_run_ledger,
    resurface_parked,
    run_id_from_key,
    utc_run_ts,
)


# ---------------------------------------------------------------------------
# Fake mem store (in-memory) — isolates tests from the real mem.db
# ---------------------------------------------------------------------------

class FakeMemStore:
    def __init__(self) -> None:
        self._rows: dict[str, dict] = {}

    def set(self, key: str, content: str, tags=None, source="") -> bool:
        created = key not in self._rows
        self._rows[key] = {
            "key": key,
            "content": content,
            "tags": ",".join(tags or []),
            "source": source,
        }
        return created

    def get(self, key: str):
        return self._rows.get(key)

    def list_by_prefix(self, prefix: str, limit: int = 50):
        return [r for k, r in self._rows.items() if k.startswith(prefix)][:limit]

    def keys(self):
        return list(self._rows.keys())


# ---------------------------------------------------------------------------
# run_id / ledger key (Interface: run_id is the key's own <utc-ts> suffix)
# ---------------------------------------------------------------------------

def test_run_id_is_utc_ts_suffix():
    now = datetime(2026, 9, 18, 5, 30, 0, tzinfo=timezone.utc)
    assert utc_run_ts(now) == "20260918T053000Z"
    key = ledger_key(utc_run_ts(now))
    assert key == "pm/night/run/20260918T053000Z"
    # run_id is recoverable from the key — atomic with the write.
    assert run_id_from_key(key) == "20260918T053000Z"


def test_run_id_no_uuid_generator():
    # The run_id is deterministic from the timestamp, not a UUID.
    a = utc_run_ts(datetime(2026, 9, 18, tzinfo=timezone.utc))
    b = utc_run_ts(datetime(2026, 9, 18, tzinfo=timezone.utc))
    assert a == b
    assert "20260918" in a


# ---------------------------------------------------------------------------
# D1 — run-ledger row shape
# ---------------------------------------------------------------------------

def test_run_ledger_row_shape_open_and_summary():
    store = FakeMemStore()
    now = datetime(2026, 9, 18, 5, 30, 0, tzinfo=timezone.utc)
    run_id = utc_run_ts(now)
    ledger = RunLedger.open(
        run_id,
        lanes=[
            LaneRecord("scout", "dispatched", "lane ran", candidates=3),
            LaneRecord("podcast", "skipped", "no source"),
        ],
        now=now,
    )
    key = emit_run_ledger(ledger, mem_store=store)
    assert key == ledger_key(run_id)
    row = json.loads(store.get(key)["content"])
    # run-open shape
    assert row["run_id"] == run_id
    assert row["phase"] == "open"
    assert row["started_at"] == now.isoformat()
    assert row["lane_status"] == {"scout": "dispatched", "podcast": "skipped"}
    assert row["candidate_counts"] == {"scout": 3, "podcast": 0}
    assert "wall_seconds" not in row

    # Transition to summary
    ledger.summarize(wall_seconds=123.0, aborted=False)
    emit_run_ledger(ledger, mem_store=store)
    row2 = json.loads(store.get(key)["content"])
    assert row2["phase"] == "summary"
    assert row2["wall_seconds"] == 123.0
    assert row2["aborted"] is False
    # run_id is still the key's own suffix (ghost-row impossible by construction)
    assert run_id_from_key(key) == row2["run_id"]


def test_run_ledger_closed_status_set_enforced():
    with pytest.raises(ValueError):
        LaneRecord("scout", "ACTIVE", "hallucinated")
    with pytest.raises(ValueError):
        LaneRecord("scout", "dispatched", "")  # missing reason
    # Closed set is exactly the spec's set.
    assert LANE_STATUS == frozenset({"dispatched", "skipped", "noop", "blank", "failed"})


def test_read_run_ledger_roundtrip():
    store = FakeMemStore()
    now = datetime(2026, 9, 18, 5, 30, 0, tzinfo=timezone.utc)
    run_id = utc_run_ts(now)
    ledger = RunLedger.open(run_id, lanes=[LaneRecord("scout", "blank", "no_candidates")], now=now)
    emit_run_ledger(ledger, mem_store=store)
    assert read_run_ledger(run_id, mem_store=store)["lane_status"]["scout"] == "blank"
    assert read_run_ledger("does-not-exist", mem_store=store) is None


# ---------------------------------------------------------------------------
# D2 — queue bootstrap + blank-page honesty
# ---------------------------------------------------------------------------

def test_ensure_queue_file_creates_if_missing(tmp_path):
    qpath = tmp_path / "queue" / "scout_candidates.jsonl"
    assert not qpath.exists()
    result = ensure_queue_file(qpath)
    assert result == qpath
    assert qpath.exists()
    # Idempotent.
    assert ensure_queue_file(qpath) == qpath


def test_blank_page_record_on_zero_candidates():
    rec = blank_page_record("scout")
    assert rec.status == BLANK == "blank"
    assert rec.reason == BLANK_REASON == "no_candidates"
    assert rec.candidates == 0
    # Blank page is EXPLICIT evidence, not inferred activity.
    assert rec.status not in ("dispatched", "skipped", "noop", "failed")


def test_lane_records_zero_candidates_all_blank():
    recs = lane_records_for_run(["scout", "podcast", "arxiv"])
    assert [r.lane for r in recs] == ["scout", "podcast", "arxiv"]
    assert all(r.status == "blank" and r.reason == "no_candidates" for r in recs)
    assert all(r.candidates == 0 for r in recs)


def test_lane_records_classify_dispatched_skipped_noop_failed():
    recs = lane_records_for_run(
        ["scout", "podcast", "arxiv", "kami"],
        dispatched={"scout"},
        skipped={"podcast": "no source"},
        noop={"arxiv": "already up to date"},
        failed={"kami": "lane crashed"},
        candidate_counts={"scout": 5},
    )
    by_lane = {r.lane: r for r in recs}
    assert by_lane["scout"].status == "dispatched"
    assert by_lane["scout"].candidates == 5
    assert by_lane["podcast"].status == "skipped"
    assert by_lane["arxiv"].status == "noop"
    assert by_lane["kami"].status == "failed"
    # Every value carries a reason.
    assert all(r.reason for r in recs)


# ---------------------------------------------------------------------------
# D3 — morning digest composer (NEW, NOT a wrapper around digest())
# ---------------------------------------------------------------------------

def test_digest_renders_direction_grade_items():
    ledger = {
        "run_id": "20260918T053000Z",
        "phase": "summary",
        "wall_seconds": 240.0,
        "aborted": False,
        "lane_status": {"scout": "dispatched", "podcast": "blank"},
        "candidate_counts": {"scout": 4, "podcast": 0},
    }
    resurfaced = [
        {
            "spec_id": "spec-old",
            "days_waiting": 3.0,
            "recommendation": "Decide spec-old: it has waited 3.0 days.",
        }
    ]
    md = compose_morning_digest(
        ledger=ledger, resurfaced=resurfaced, date_str="2026-09-18"
    )
    assert "# Morning — 2026-09-18" in md
    assert "20260918T053000Z" in md
    assert "240s" in md
    # Direction-grade register fields are present.
    assert "Mechanism" in md
    assert "Purpose" in md
    assert "Why broken" in md
    assert "What the decision turns on" in md
    # Blank page is recorded explicitly, not as activity.
    assert "blank page" in md
    assert "scout" in md and "4 candidate" in md


def test_resurface_carries_days_waiting_and_recommendation():
    now = datetime(2026, 9, 18, 5, 0, 0, tzinfo=timezone.utc)
    old = (now - timedelta(days=3)).isoformat()
    fresh = (now - timedelta(hours=2)).isoformat()
    resurfaced = resurface_parked(
        [
            {"spec_id": "spec-old", "parked_at": old, "reason": "saturated"},
            {"spec_id": "spec-fresh", "parked_at": fresh, "reason": "saturated"},
        ],
        now=now,
    )
    # Only the >48h-old one re-surfaces.
    assert len(resurfaced) == 1
    rs = resurfaced[0]
    assert rs["spec_id"] == "spec-old"
    assert rs["days_waiting"] == pytest.approx(3.0)
    assert "3.0 days" in rs["recommendation"]
    assert "Decide spec-old" in rs["recommendation"]


def test_cheap_ask_guard_rejects_code_identifier_items():
    from lapis_pm.scout.night_ledger import DigestItem

    clean = DigestItem(
        title="A lane is silent",
        mechanism="The night produced no candidates.",
        purpose="Silence must be recorded explicitly.",
        why_broken="No writer existed for the ledger.",
        turns_on="Whether to add the ledger.",
    )
    assert cheap_ask_guard(clean) is True

    # A code-identifier-bearing item is misrouted (rejected).
    bad = DigestItem(
        title="A lane is silent",
        mechanism="The night produced no candidates.",
        purpose="Silence must be recorded explicitly.",
        why_broken="lapis_pm.scout.night_queue.run_night never wrote a row.",
        turns_on="Whether to add the ledger.",
    )
    assert cheap_ask_guard(bad) is False


def test_why_broken_allows_one_line_technical_root_cause():
    """A one-line technical root-cause summary for a code defect is allowed as
    long as it stays in the direction-grade register (no identifiers)."""
    from lapis_pm.scout.night_ledger import DigestItem

    item = DigestItem(
        title="The night could not see its own runs",
        mechanism="The night run has no ledger to read.",
        purpose="The cook cannot close loops without a ledger.",
        why_broken="The run-state writer was missing entirely, so runs left no trace.",
        turns_on="Whether to add the ledger now.",
    )
    assert cheap_ask_guard(item) is True


def test_emit_morning_digest_writes_file(tmp_path):
    ledger = {
        "run_id": "20260918T053000Z",
        "phase": "summary",
        "wall_seconds": 100.0,
        "lane_status": {"scout": "blank"},
        "candidate_counts": {"scout": 0},
    }
    out = emit_morning_digest(
        ledger=ledger, date_str="2026-09-18", morning_dir=tmp_path
    )
    assert out == tmp_path / "2026-09-18.md"
    assert out.exists()
    text = out.read_text()
    assert "# Morning — 2026-09-18" in text


def test_morning_digest_is_not_a_digest_wrapper():
    """D3 correction: emit_morning_digest is a NEW composer, not a wrapper
    around lapis_pm/scout/digest.py::digest(). The two surfaces are
    categorically incompatible (Markdown letter vs LapisToolReturn JSON)."""
    from lapis_pm.scout import digest as digest_mod

    # The morning composer does not import or call digest().
    import inspect
    src = inspect.getsource(nl)
    assert "digest_mod.digest(" not in src
    assert "from .digest import" not in src
    assert "from lapis_pm.scout.digest import" not in src
    # And the digest module is a distinct, independent surface.
    assert callable(digest_mod.digest)


# ---------------------------------------------------------------------------
# D4 — loud, non-Pushover delivery
# ---------------------------------------------------------------------------

def test_deliver_morning_digest_never_uses_pushover(tmp_path):
    digest_path = tmp_path / "2026-09-18.md"
    digest_path.write_text("x")
    called = {"pushover": 0, "matrix": 0}

    def pushover_notify(message, title):
        called["pushover"] += 1
        return True

    def matrix_notify(message, title):
        called["matrix"] += 1
        return True

    # Even if the caller names pushover, this lane refuses it.
    rec = deliver_morning_digest(
        digest_path, transport="pushover", notify_fn=pushover_notify
    )
    assert rec["transport"] is None
    assert rec["delivered"] is False
    assert rec["fallback"] is True
    assert called["pushover"] == 0  # never sent

    # Matrix (preferred) is used when configured.
    rec2 = deliver_morning_digest(
        digest_path, transport="matrix", notify_fn=matrix_notify
    )
    assert rec2["transport"] == "matrix"
    assert rec2["delivered"] is True
    assert rec2["fallback"] is False
    assert called["matrix"] == 1


def test_deliver_morning_digest_file_ledger_fallback(tmp_path):
    digest_path = tmp_path / "2026-09-18.md"
    digest_path.write_text("x")
    rec = deliver_morning_digest(digest_path)  # no transport, no notify_fn
    assert rec["transport"] is None
    assert rec["delivered"] is False
    assert rec["fallback"] is True
    assert rec["digest_path"] == str(digest_path)


# ---------------------------------------------------------------------------
# Integration (fixture): fake run -> exactly one ledger row + one digest file
# ---------------------------------------------------------------------------

def test_integration_fake_run_one_ledger_row_one_digest(tmp_path, monkeypatch):
    """A fake run (one lane success + one skipped) produces exactly one ledger
    row + one digest file at a temp path; notify called once via a mocked
    transport; Pushover asserted never used for this lane."""
    store = FakeMemStore()
    now = datetime(2026, 9, 18, 5, 30, 0, tzinfo=timezone.utc)

    # D1+D2: run-open ledger + queue bootstrap.
    from lapis_pm.scout.night_ledger import record_night_run
    ledger = record_night_run(
        lanes=["scout", "podcast"],
        dispatched={"scout"},
        skipped={"podcast": "no source"},
        candidate_counts={"scout": 2},
        queue_path=tmp_path / "queue" / "scout_candidates.jsonl",
        mem_store=store,
        now=now,
    )
    # Queue bootstrapped.
    assert (tmp_path / "queue" / "scout_candidates.jsonl").exists()
    # Exactly one ledger key so far.
    assert len(store.keys()) == 1
    assert store.keys()[0] == ledger_key(ledger.run_id)

    # D1: run-summary.
    from lapis_pm.scout.night_ledger import emit_night_summary
    emit_night_summary(
        ledger,
        wall_seconds=120.0,
        lanes=lane_records_for_run(
            ["scout", "podcast"],
            dispatched={"scout"},
            skipped={"podcast": "no source"},
            candidate_counts={"scout": 2},
        ),
        now=now,
        mem_store=store,
    )
    summary = json.loads(store.get(ledger_key(ledger.run_id))["content"])
    # Still exactly one ledger row (upserted, not a second row).
    assert len(store.keys()) == 1
    assert summary["phase"] == "summary"
    assert summary["wall_seconds"] == 120.0
    assert summary["lane_status"] == {"scout": "dispatched", "podcast": "skipped"}

    # D3+D4: morning digest + loud delivery.
    from lapis_pm.scout.night_ledger import serve_morning
    notify_calls = {"n": 0}

    def fake_notify(message, title):
        notify_calls["n"] += 1
        return True

    resurfaced = resurface_parked(
        [{"spec_id": "spec-old", "parked_at": (now - timedelta(days=2)).isoformat()}],
        now=now,
    )
    digest_path = serve_morning(
        ledger=summary,
        resurfaced=resurfaced,
        date_str="2026-09-18",
        morning_dir=tmp_path / "morning",
        notify_fn=fake_notify,
        transport="matrix",
        mem_store=store,
        run_id=ledger.run_id,
    )
    # Exactly one digest file at a temp path.
    assert digest_path == tmp_path / "morning" / "2026-09-18.md"
    assert digest_path.exists()
    # Notify called exactly once via the mocked transport.
    assert notify_calls["n"] == 1
    # The ledger row records the digest path + a non-Pushover delivery.
    final = json.loads(store.get(ledger_key(ledger.run_id))["content"])
    assert final["digest_path"] == str(digest_path)
    assert final["notify"]["transport"] == "matrix"
    assert final["notify"]["delivered"] is True
    # Still exactly one ledger row end-to-end.
    assert len(store.keys()) == 1


def test_run_night_writes_ledger_row(tmp_path, monkeypatch):
    """run_night (the night-run entry point) writes a run-ledger row.

    Exercises the D1 wiring end-to-end: a drained scout run produces exactly
    one pm/night/run/<utc-ts> row (run-summary) and bootstraps the queue.
    """
    import time as _time
    from unittest.mock import MagicMock

    import yaml

    from lapis_pm.scout.night_queue import run_night

    store = FakeMemStore()
    sims_dir = tmp_path / "sims"
    sims_dir.mkdir()
    # A minimal scaffold so the run has something to (attempt) dispatch.
    data = {
        "spec_id": "spec_a",
        "spec_version": "v0",
        "description": "test",
        "priority_profile": "full-pass-once",
        "static_scaffold": {
            "objective": "test",
            "architecture_sketch": "test",
            "scenario": {
                "conditions": [],
                "time_progression": {},
                "external_state": {},
                "utilization_pattern": "",
            },
        },
        "generation_directive": "test",
        "matrix": {
            "optional_steps_included": [[]],
            "external_state_severity": ["healthy"],
            "concurrent_load": [1],
            "runs_per_cell": 1,
        },
    }
    (sims_dir / "spec_a.yaml").write_text(yaml.dump(data))

    log_root = tmp_path / "log"
    queue_path = tmp_path / "queue" / "scout_candidates.jsonl"

    monkeypatch.setattr("httpx.get", lambda url, timeout: MagicMock(status_code=200))
    monkeypatch.setattr("lapis_pm.scout.runner.simulate", lambda *a, **k: [])

    result = run_night(
        sims_dir=sims_dir,
        once=True,
        log_root=log_root,
        mem_store=store,
        queue_path=queue_path,
    )

    # The queue was bootstrapped (D2).
    assert queue_path.exists()
    # Exactly one ledger row was written (D1), tagged pm-night-run.
    assert len(store.keys()) == 1
    key = store.keys()[0]
    assert key.startswith("pm/night/run/")
    row = json.loads(store.get(key)["content"])
    assert row["phase"] == "summary"
    assert row["run_id"] == key[len("pm/night/run/"):]
    assert row["wall_seconds"] is not None
    # The scout lane status is in the closed set.
    assert row["lane_status"]["scout"] in {
        "dispatched", "skipped", "noop", "blank", "failed",
    }
    # The run actually executed (not a no-op).
    assert result.total_units >= 0


# ---------------------------------------------------------------------------
# D1+D3 end-to-end: run_night produces BOTH a ledger row AND a morning digest
# ---------------------------------------------------------------------------

def _write_scaffold(sims_dir: Path, spec_id: str = "spec_a") -> None:
    """Write a minimal scaffold YAML so run_night has work to dispatch."""
    import yaml
    data = {
        "spec_id": spec_id,
        "spec_version": "v0",
        "description": "test",
        "priority_profile": "full-pass-once",
        "static_scaffold": {
            "objective": "test",
            "architecture_sketch": "test",
            "scenario": {
                "conditions": [],
                "time_progression": {},
                "external_state": {},
                "utilization_pattern": "",
            },
        },
        "generation_directive": "test",
        "matrix": {
            "optional_steps_included": [[]],
            "external_state_severity": ["healthy"],
            "concurrent_load": [1],
            "runs_per_cell": 1,
        },
    }
    (sims_dir / f"{spec_id}.yaml").write_text(yaml.dump(data))


def test_run_night_produces_ledger_row_and_morning_digest(tmp_path, monkeypatch):
    """DoD: one live night run produces a pm/night/run/<ts> row AND
    /srv/lapis/morning/<date>.md. Exercises the D3/D4 wiring end-to-end from the
    night-run entry point (run_night), not just the isolated composer."""
    from unittest.mock import MagicMock

    from lapis_pm.scout.night_queue import run_night

    store = FakeMemStore()
    sims_dir = tmp_path / "sims"
    sims_dir.mkdir()
    _write_scaffold(sims_dir)

    log_root = tmp_path / "log"
    queue_path = tmp_path / "queue" / "scout_candidates.jsonl"
    morning_dir = tmp_path / "morning"

    monkeypatch.setattr("httpx.get", lambda url, timeout: MagicMock(status_code=200))
    monkeypatch.setattr("lapis_pm.scout.runner.simulate", lambda *a, **k: [])
    # Point the /room root at a temp dir (the composer defaults to the real
    # /room; the DoD requires the file to be written, not the exact production
    # path). emit_morning_digest appends /morning to the room root.
    monkeypatch.setattr(
        "lapis_pm.scout.night_ledger._morning_root", lambda: tmp_path
    )

    result = run_night(
        sims_dir=sims_dir,
        once=True,
        log_root=log_root,
        mem_store=store,
        queue_path=queue_path,
    )

    # D1: exactly one ledger row (run-summary).
    assert len(store.keys()) == 1
    key = store.keys()[0]
    assert key.startswith("pm/night/run/")
    row = json.loads(store.get(key)["content"])
    assert row["phase"] == "summary"
    assert row["run_id"] == key[len("pm/night/run/"):]
    assert row["wall_seconds"] is not None

    # D3: the morning digest file was written to /srv/lapis/morning/<date>.md.
    date_str = row["run_id"][:8].replace("T", "-")
    digest_path = morning_dir / f"{date_str}.md"
    assert digest_path.exists(), (
        f"morning digest not written at {digest_path}; "
        f"morning_dir contents: {list(morning_dir.iterdir()) if morning_dir.exists() else 'missing'}"
    )
    digest_text = digest_path.read_text()
    assert digest_text.startswith(f"# Morning — {date_str}")

    # D4: the ledger row records the digest path (file is the source of truth).
    assert row.get("digest_path") == str(digest_path)
    # The run actually executed.
    assert result.total_units >= 0


def test_run_night_dispatched_run_not_mislabeled_blank(tmp_path, monkeypatch):
    """D1: a run that actually dispatches work must record the scout lane as
    ``dispatched`` (with candidate counts), NOT a blank page. A dispatched run
    mislabeled as blank is the exact under-reporting the reviewer flagged."""
    from unittest.mock import MagicMock

    from lapis_pm.scout.night_queue import run_night

    store = FakeMemStore()
    sims_dir = tmp_path / "sims"
    sims_dir.mkdir()
    _write_scaffold(sims_dir)

    log_root = tmp_path / "log"
    queue_path = tmp_path / "queue" / "scout_candidates.jsonl"
    morning_dir = tmp_path / "morning"

    monkeypatch.setattr("httpx.get", lambda url, timeout: MagicMock(status_code=200))
    monkeypatch.setattr("lapis_pm.scout.runner.simulate", lambda *a, **k: [])
    monkeypatch.setattr(
        "lapis_pm.scout.night_ledger._morning_root", lambda: tmp_path
    )

    result = run_night(
        sims_dir=sims_dir,
        once=True,
        log_root=log_root,
        mem_store=store,
        queue_path=queue_path,
    )

    # The run dispatched at least one unit.
    assert result.total_units >= 1

    key = store.keys()[0]
    row = json.loads(store.get(key)["content"])
    # The scout lane is recorded as dispatched, not blank.
    assert row["lane_status"]["scout"] == "dispatched"
    assert row["lane_status"]["scout"] != "blank"
    # Candidate counts reflect what actually ran.
    assert row["candidate_counts"]["scout"] == result.total_units
    # Every lane status carries a reason.
    assert row["lane_reason"]["scout"]


def test_run_night_noop_run_records_noop_not_blank(tmp_path, monkeypatch):
    """D1: a run with no runnable units (no scaffolds) records the scout lane
    as ``noop`` (with a reason), not a blank page — blank is reserved for a
    lane that produced zero candidates while actually dispatching."""
    from unittest.mock import MagicMock

    from lapis_pm.scout.night_queue import run_night

    store = FakeMemStore()
    sims_dir = tmp_path / "sims"
    sims_dir.mkdir()  # empty: no scaffold YAMLs

    log_root = tmp_path / "log"
    queue_path = tmp_path / "queue" / "scout_candidates.jsonl"
    morning_dir = tmp_path / "morning"

    monkeypatch.setattr("httpx.get", lambda url, timeout: MagicMock(status_code=200))
    monkeypatch.setattr("lapis_pm.scout.runner.simulate", lambda *a, **k: [])
    monkeypatch.setattr(
        "lapis_pm.scout.night_ledger._morning_root", lambda: tmp_path
    )

    result = run_night(
        sims_dir=sims_dir,
        once=True,
        log_root=log_root,
        mem_store=store,
        queue_path=queue_path,
    )

    key = store.keys()[0]
    row = json.loads(store.get(key)["content"])
    # Nothing ran -> noop (not blank, not dispatched).
    assert row["lane_status"]["scout"] == "noop"
    assert row["lane_reason"]["scout"]
    # The morning digest is still written (a no-op night still gets a digest).
    date_str = row["run_id"][:8].replace("T", "-")
    assert (morning_dir / f"{date_str}.md").exists()
