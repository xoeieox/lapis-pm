"""Tests for `lapis-pm ratify` CLI subcommand (lapis-pm-ratification-wiring-v0).

Coverage (per spec §Tests):
1. happy path each outcome     — confirm/correct/override/redirect all write to
                                  router/lapis-pm/ratification-outcomes/ with the right verdict
2. auto-prior resolution       — seed two decisions; ratify correct --intent x (no --prior);
                                  assert citations contains the NEWER decision's mem key
3. explicit --prior wins        — seed a decision; pass explicit --prior for a different id;
                                  assert the explicit one is used
4. bare event_id normalization  — --prior abc12345 and --prior router/lapis-pm/decisions/abc12345
                                  produce the same citation
5. confirm without prior errors — exit 3, clear stderr message, nothing written
6. no-prior warns + writes      — correct/override/redirect with no decisions seeded:
                                  exit 0, stderr warning, writes with empty citations
7. --intent required            — correct (no --intent) exits 2
8. --json output                — --json prints valid JSON with key; default is human-readable
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_mem(store: dict | None = None) -> MagicMock:
    """Mock MemoryStore with .set / .get / .list_all wired up."""
    data: dict[str, dict] = {}
    if store:
        data.update(store)

    mock = MagicMock()

    def _set(key: str, content: str, tags: list | None = None, source: str = "") -> bool:
        data[key] = {"key": key, "content": content, "tags": tags or []}
        return True

    def _get(key: str) -> dict | None:
        return data.get(key)

    def _list_all(tag: str = "", tags: list | None = None, since: str = "", limit: int = 50) -> list[dict]:
        result = list(data.values())
        if tag:
            result = [r for r in result if tag in (r.get("tags") or [])]
        return result[:limit]

    mock.set.side_effect = _set
    mock.get.side_effect = _get
    mock.list_all.side_effect = _list_all
    mock._data = data
    return mock


def _make_decision_row(target_id: str, freshness_stamp: str, event_id: str) -> tuple[str, dict]:
    """Return (key, row_dict) for a seeded decision entry."""
    key = f"router/lapis-pm/decisions/{event_id}"
    entry = {
        "verdict": "proposed",
        "claim": f"decision for {target_id}",
        "citations": [],
        "freshness_stamp": freshness_stamp,
        "scope_id": "router/lapis-pm",
        "target_id": target_id,
        "fragment_id": "kickoff",
        "expert_chosen": None,
        "ratification_outcome": None,
        "intent_summary": "test decision",
        "drift_class": None,
        "primitive_decomposition": None,
    }
    row = {
        "key": key,
        "content": json.dumps(entry),
        "tags": ["lapis-pm", "router-portfolio", f"target:{target_id}"],
    }
    return key, row


def _run(argv: list[str], mock_mem) -> tuple[int, str, str]:
    """Run main(argv) with _mem patched; return (exit_code, stdout, stderr)."""
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    rc = 0
    with (
        patch("lapis_pm.router_portfolio._mem", return_value=mock_mem),
        redirect_stdout(out_buf),
        redirect_stderr(err_buf),
    ):
        try:
            rc = main(argv)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
    return rc, out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Happy path — all four outcomes
# ---------------------------------------------------------------------------

class TestHappyPathOutcomes:
    def test_confirm_writes_ratified_verdict(self) -> None:
        # Seed a prior decision so confirm can resolve it
        mock_mem = _make_mock_mem()
        k, row = _make_decision_row("t1", "2026-05-01T10:00:00Z", "event-prior-001")
        mock_mem._data[k] = row

        rc, out, err = _run(["ratify", "t1", "confirm"], mock_mem)

        assert rc == 0, f"Expected exit 0, got {rc}; stderr={err}"
        ratify_keys = [k for k in mock_mem._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        assert len(ratify_keys) == 1
        entry = json.loads(mock_mem._data[ratify_keys[0]]["content"])
        assert entry["verdict"] == "ratified"
        assert entry["ratification_outcome"] == "confirm"

    def test_correct_writes_corrected_verdict(self) -> None:
        mock_mem = _make_mock_mem()
        rc, out, err = _run(["ratify", "t1", "correct", "--intent", "spec was wrong"], mock_mem)

        assert rc == 0, f"Expected exit 0, got {rc}; stderr={err}"
        ratify_keys = [k for k in mock_mem._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        assert len(ratify_keys) == 1
        entry = json.loads(mock_mem._data[ratify_keys[0]]["content"])
        assert entry["verdict"] == "corrected"
        assert entry["ratification_outcome"] == "correct"
        assert entry["intent_summary"] == "spec was wrong"

    def test_override_writes_overridden_verdict(self) -> None:
        mock_mem = _make_mock_mem()
        rc, out, err = _run(["ratify", "t1", "override", "--intent", "wrong expert chosen"], mock_mem)

        assert rc == 0
        ratify_keys = [k for k in mock_mem._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        assert len(ratify_keys) == 1
        entry = json.loads(mock_mem._data[ratify_keys[0]]["content"])
        assert entry["verdict"] == "overridden"
        assert entry["ratification_outcome"] == "override"

    def test_redirect_writes_redirected_verdict(self) -> None:
        mock_mem = _make_mock_mem()
        rc, out, err = _run(["ratify", "t1", "redirect", "--intent", "scope change"], mock_mem)

        assert rc == 0
        ratify_keys = [k for k in mock_mem._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        assert len(ratify_keys) == 1
        entry = json.loads(mock_mem._data[ratify_keys[0]]["content"])
        assert entry["verdict"] == "redirected"
        assert entry["ratification_outcome"] == "redirect"

    def test_all_outcomes_write_to_ratification_outcomes_namespace(self) -> None:
        for outcome, intent in [
            ("confirm", None),
            ("correct", "intent x"),
            ("override", "intent y"),
            ("redirect", "intent z"),
        ]:
            mock_mem = _make_mock_mem()
            # Seed prior for confirm
            if outcome == "confirm":
                k, row = _make_decision_row("t1", "2026-05-01T10:00:00Z", "prior-001")
                mock_mem._data[k] = row
            argv = ["ratify", "t1", outcome]
            if intent:
                argv += ["--intent", intent]
            rc, out, err = _run(argv, mock_mem)
            assert rc == 0, f"outcome={outcome} rc={rc} err={err}"
            ratify_keys = [
                k for k in mock_mem._data
                if k.startswith("router/lapis-pm/ratification-outcomes/")
            ]
            assert len(ratify_keys) == 1, f"outcome={outcome}: expected 1 ratification entry"


# ---------------------------------------------------------------------------
# 2. Auto-prior resolution — newest decision wins
# ---------------------------------------------------------------------------

class TestAutoPriorResolution:
    def test_newer_decision_used_as_prior(self) -> None:
        """Seed two decisions for t1; ratify correct without --prior; assert newer cited."""
        mock_mem = _make_mock_mem()
        k_old, row_old = _make_decision_row("t1", "2026-05-01T08:00:00Z", "old-event-aaa")
        k_new, row_new = _make_decision_row("t1", "2026-05-02T10:00:00Z", "new-event-bbb")
        mock_mem._data[k_old] = row_old
        mock_mem._data[k_new] = row_new

        rc, out, err = _run(["ratify", "t1", "correct", "--intent", "scope fix"], mock_mem)

        assert rc == 0, f"err={err}"
        ratify_keys = [k for k in mock_mem._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        assert len(ratify_keys) == 1
        entry = json.loads(mock_mem._data[ratify_keys[0]]["content"])
        # citations should reference the NEWER decision's mem key
        assert "router/lapis-pm/decisions/new-event-bbb" in entry["citations"]
        assert "router/lapis-pm/decisions/old-event-aaa" not in entry["citations"]


# ---------------------------------------------------------------------------
# 3. Explicit --prior wins over auto-resolution
# ---------------------------------------------------------------------------

class TestExplicitPriorWins:
    def test_explicit_prior_used_instead_of_auto_resolved(self) -> None:
        """Seed a decision; pass explicit --prior for a different event_id."""
        mock_mem = _make_mock_mem()
        k, row = _make_decision_row("t1", "2026-05-01T10:00:00Z", "auto-resolved-ccc")
        mock_mem._data[k] = row

        rc, out, err = _run(
            ["ratify", "t1", "correct", "--intent", "x", "--prior", "explicit-event-ddd"],
            mock_mem,
        )

        assert rc == 0, f"err={err}"
        ratify_keys = [k for k in mock_mem._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        entry = json.loads(mock_mem._data[ratify_keys[0]]["content"])
        assert "router/lapis-pm/decisions/explicit-event-ddd" in entry["citations"]
        assert "router/lapis-pm/decisions/auto-resolved-ccc" not in entry["citations"]


# ---------------------------------------------------------------------------
# 4. Bare event_id normalization
# ---------------------------------------------------------------------------

class TestBareEventIdNormalization:
    def test_bare_id_and_full_key_produce_same_citation(self) -> None:
        """--prior abc12345 and --prior router/lapis-pm/decisions/abc12345 → same citation."""
        bare_event_id = "2026-05-01T10:00:00Z-abc12345"
        full_key = f"router/lapis-pm/decisions/{bare_event_id}"

        mock_mem_bare = _make_mock_mem()
        rc1, _, _ = _run(
            ["ratify", "t1", "correct", "--intent", "x", "--prior", bare_event_id],
            mock_mem_bare,
        )
        assert rc1 == 0
        ratify_keys_bare = [k for k in mock_mem_bare._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        entry_bare = json.loads(mock_mem_bare._data[ratify_keys_bare[0]]["content"])

        mock_mem_full = _make_mock_mem()
        rc2, _, _ = _run(
            ["ratify", "t1", "correct", "--intent", "x", "--prior", full_key],
            mock_mem_full,
        )
        assert rc2 == 0
        ratify_keys_full = [k for k in mock_mem_full._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        entry_full = json.loads(mock_mem_full._data[ratify_keys_full[0]]["content"])

        # Both must cite the same key
        assert entry_bare["citations"] == entry_full["citations"]
        assert full_key in entry_bare["citations"]


# ---------------------------------------------------------------------------
# 5. Confirm without prior exits 3
# ---------------------------------------------------------------------------

class TestConfirmWithoutPrior:
    def test_confirm_no_prior_exits_3(self) -> None:
        """No decisions seeded; lapis-pm ratify t1 confirm → exit 3, nothing written."""
        mock_mem = _make_mock_mem()
        rc, out, err = _run(["ratify", "t1", "confirm"], mock_mem)

        assert rc == 3
        assert "no prior decision found" in err
        assert "cannot confirm" in err
        # Nothing should be written to ratification-outcomes
        ratify_keys = [k for k in mock_mem._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        assert len(ratify_keys) == 0


# ---------------------------------------------------------------------------
# 6. Non-confirm without prior: warns + writes (exit 0)
# ---------------------------------------------------------------------------

class TestNonConfirmWithoutPrior:
    @pytest.mark.parametrize("outcome", ["correct", "override", "redirect"])
    def test_no_prior_warns_and_writes(self, outcome: str) -> None:
        """No decisions seeded; non-confirm ratify writes with empty citations + warns."""
        mock_mem = _make_mock_mem()
        rc, out, err = _run(["ratify", "t1", outcome, "--intent", "reason"], mock_mem)

        assert rc == 0, f"outcome={outcome} rc={rc}"
        assert "[ratify] no prior decision found" in err
        ratify_keys = [k for k in mock_mem._data if k.startswith("router/lapis-pm/ratification-outcomes/")]
        assert len(ratify_keys) == 1
        entry = json.loads(mock_mem._data[ratify_keys[0]]["content"])
        # citations should be empty (no prior)
        assert entry["citations"] == []


# ---------------------------------------------------------------------------
# 7. --intent required for non-confirm
# ---------------------------------------------------------------------------

class TestIntentRequired:
    @pytest.mark.parametrize("outcome", ["correct", "override", "redirect"])
    def test_missing_intent_exits_2(self, outcome: str) -> None:
        """ratify t1 <non-confirm> without --intent → exit 2."""
        mock_mem = _make_mock_mem()
        rc, out, err = _run(["ratify", "t1", outcome], mock_mem)

        assert rc == 2
        assert "--intent is required" in err
        # Nothing written
        assert not any(
            k.startswith("router/lapis-pm/ratification-outcomes/")
            for k in mock_mem._data
        )


# ---------------------------------------------------------------------------
# 8. --json output
# ---------------------------------------------------------------------------

class TestJsonOutput:
    def test_json_flag_prints_key_as_json(self) -> None:
        """--json mode prints valid JSON with 'key' field."""
        mock_mem = _make_mock_mem()
        rc, out, err = _run(
            ["ratify", "t1", "correct", "--intent", "x", "--json"],
            mock_mem,
        )

        assert rc == 0
        parsed = json.loads(out.strip())
        assert "key" in parsed
        assert parsed["key"].startswith("router/lapis-pm/ratification-outcomes/")

    def test_non_json_mode_is_human_readable(self) -> None:
        """Default (non-JSON) mode prints human-readable text, not JSON."""
        mock_mem = _make_mock_mem()
        rc, out, err = _run(
            ["ratify", "t1", "correct", "--intent", "x"],
            mock_mem,
        )

        assert rc == 0
        # Should not be parseable as JSON
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(out.strip())
        assert "Ratified" in out
        assert "t1" in out
