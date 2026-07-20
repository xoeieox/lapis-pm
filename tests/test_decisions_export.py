"""Tests for lapis_pm.decisions_export — 19 tests per spec."""
from __future__ import annotations

import io
import sys
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from lapis_pm.decisions_export import (
    collect,
    parse_since,
    parse_types,
    render,
    run,
)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(s: str) -> str:
    """Return an ISO 8601 string for a simple YYYY-MM-DD string at UTC midnight."""
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC).isoformat()


def _make_row(key: str, created: str, content: str = "body", tags: str = "") -> dict:
    return {
        "key": key,
        "created_at": _dt(created),
        "updated_at": _dt(created),
        "content": content,
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# 1. parse_since accepts `Nd` form
# ---------------------------------------------------------------------------

def test_parse_since_nd_form():
    now = datetime.now(UTC)
    result = parse_since("7d")
    delta = now - result
    assert abs(delta.total_seconds() - 7 * 86400) < 1  # within ±1s


# ---------------------------------------------------------------------------
# 2. parse_since accepts ISO date form
# ---------------------------------------------------------------------------

def test_parse_since_iso_date():
    result = parse_since("2026-05-09")
    assert result == datetime(2026, 5, 9, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 3. parse_since rejects bad forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["7days", "not-a-date", "2026/05/09", ""])
def test_parse_since_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_since(bad)


# ---------------------------------------------------------------------------
# 4. parse_types happy path
# ---------------------------------------------------------------------------

def test_parse_types_happy_path():
    result = parse_types("decision,feedback,pattern", "")
    assert result == ["decision", "feedback", "pattern"]


# ---------------------------------------------------------------------------
# 5. parse_types exclude subtracts
# ---------------------------------------------------------------------------

def test_parse_types_exclude_subtracts():
    result = parse_types("decision,feedback,pattern", "pattern")
    assert result == ["decision", "feedback"]


# ---------------------------------------------------------------------------
# 6. parse_types unknown token errors
# ---------------------------------------------------------------------------

def test_parse_types_unknown_token():
    with pytest.raises(ValueError, match="unknown type: foo"):
        parse_types("decision,foo", "")


# ---------------------------------------------------------------------------
# 7. parse_types empty result errors
# ---------------------------------------------------------------------------

def test_parse_types_empty_result():
    with pytest.raises(ValueError, match="no types selected"):
        parse_types("decision", "decision")


# ---------------------------------------------------------------------------
# 8. collect filters by date
# ---------------------------------------------------------------------------

def test_collect_filters_by_date(monkeypatch):
    rows = [
        _make_row("decision/a", "2026-05-01"),   # -15d from 2026-05-16
        _make_row("decision/b", "2026-05-11"),   # -5d
        _make_row("decision/c", "2026-05-15"),   # -1d
    ]
    monkeypatch.setattr(
        "lapis_pm.node_identity.writable_store",
        lambda: type("MS", (), {"list_by_prefix": lambda self, p, limit=None: rows})(),
    )
    since = datetime(2026, 5, 9, tzinfo=UTC)  # within -7d window
    result = collect(["decision"], since)
    keys = [r["key"] for r in result["decision"]]
    assert "decision/a" not in keys
    assert "decision/b" in keys
    assert "decision/c" in keys


# ---------------------------------------------------------------------------
# 9. collect filters by tag
# ---------------------------------------------------------------------------

def test_collect_filters_by_tag(monkeypatch):
    rows = [
        _make_row("decision/a", "2026-05-14", tags="lapis-pm,foo"),
        _make_row("decision/b", "2026-05-14", tags="other"),
        _make_row("decision/c", "2026-05-14", tags=""),
    ]
    monkeypatch.setattr(
        "lapis_pm.node_identity.writable_store",
        lambda: type("MS", (), {"list_by_prefix": lambda self, p, limit=None: rows})(),
    )
    since = datetime(2026, 5, 1, tzinfo=UTC)
    result = collect(["decision"], since, tag="lapis-pm")
    keys = [r["key"] for r in result["decision"]]
    assert keys == ["decision/a"]

    # No tag filter returns all
    result2 = collect(["decision"], since, tag="")
    assert len(result2["decision"]) == 3


# ---------------------------------------------------------------------------
# 10. collect sorts newest-first within type
# ---------------------------------------------------------------------------

def test_collect_sorts_newest_first(monkeypatch):
    rows = [
        _make_row("decision/c", "2026-05-10"),
        _make_row("decision/a", "2026-05-14"),
        _make_row("decision/b", "2026-05-12"),
    ]
    monkeypatch.setattr(
        "lapis_pm.node_identity.writable_store",
        lambda: type("MS", (), {"list_by_prefix": lambda self, p, limit=None: rows})(),
    )
    since = datetime(2026, 5, 1, tzinfo=UTC)
    result = collect(["decision"], since)
    dates = [r["created_at"][:10] for r in result["decision"]]
    assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# 11. collect ties broken by key ascending
# ---------------------------------------------------------------------------

def test_collect_tie_broken_by_key_ascending(monkeypatch):
    tied_date = "2026-05-14"
    rows = [
        _make_row("decision/z-last", tied_date),
        _make_row("decision/a-first", tied_date),
        _make_row("decision/m-mid", tied_date),
    ]
    monkeypatch.setattr(
        "lapis_pm.node_identity.writable_store",
        lambda: type("MS", (), {"list_by_prefix": lambda self, p, limit=None: rows})(),
    )
    since = datetime(2026, 5, 1, tzinfo=UTC)
    result = collect(["decision"], since)
    keys = [r["key"] for r in result["decision"]]
    assert keys == ["decision/a-first", "decision/m-mid", "decision/z-last"]


# ---------------------------------------------------------------------------
# 12. render empty window
# ---------------------------------------------------------------------------

def test_render_empty_window():
    since = datetime(2026, 5, 9, tzinfo=UTC)
    output = render({"decision": [], "feedback": [], "pattern": []}, since)
    assert "No entries in window." in output
    assert "## Decisions" not in output
    assert "## Feedback" not in output
    assert "## Patterns" not in output


# ---------------------------------------------------------------------------
# 13. render populated
# ---------------------------------------------------------------------------

def test_render_populated():
    since = datetime(2026, 5, 9, tzinfo=UTC)
    grouped = {
        "decision": [
            _make_row("decision/alpha", "2026-05-14", content="body alpha", tags="lapis-pm"),
            _make_row("decision/beta", "2026-05-12", content="body beta"),
        ],
        "pattern": [
            _make_row("pattern/gamma", "2026-05-13", content="body gamma"),
        ],
    }
    output = render(grouped, since)
    assert "## Decisions (2)" in output
    assert "## Patterns (1)" in output
    assert "### decision/alpha" in output
    assert "### decision/beta" in output
    assert "### pattern/gamma" in output
    assert "body alpha" in output
    assert "**Total entries:** 3" in output
    assert "decisions 2" in output
    assert "patterns 1" in output
    assert "---" in output


# ---------------------------------------------------------------------------
# 14. render excluded-types disclaimer
# ---------------------------------------------------------------------------

def test_render_excluded_types_disclaimer():
    since = datetime(2026, 5, 9, tzinfo=UTC)
    output = render({"decision": [_make_row("decision/x", "2026-05-14")]}, since)
    assert "**Excluded types:**" in output
    for t in ["architecture", "feedback", "incident", "pattern", "project", "reference"]:
        assert t in output
    assert "**Note:**" in output
    assert "curated trace" in output


# ---------------------------------------------------------------------------
# 15. render omits Updated line when equal to Created
# ---------------------------------------------------------------------------

def test_render_omits_updated_when_equal():
    since = datetime(2026, 5, 9, tzinfo=UTC)
    row = _make_row("decision/x", "2026-05-14")
    # created_at == updated_at by _make_row construction
    output = render({"decision": [row]}, since)
    assert "**Updated:**" not in output


# ---------------------------------------------------------------------------
# 16. run exit 0 on empty window
# ---------------------------------------------------------------------------

def test_run_exit_0_empty_window(monkeypatch, capsys):
    monkeypatch.setattr(
        "lapis_pm.node_identity.writable_store",
        lambda: type("MS", (), {"list_by_prefix": lambda self, p, limit=None: []})(),
    )
    code = run("1d", "decision,feedback,pattern", "", "", None)
    assert code == 0
    out = capsys.readouterr().out
    assert "No entries in window." in out


# ---------------------------------------------------------------------------
# 17. run exit 2 on bad --since
# ---------------------------------------------------------------------------

def test_run_exit_2_bad_since(capsys):
    code = run("not-a-date", "decision,feedback,pattern", "", "", None)
    assert code == 2
    err = capsys.readouterr().err
    assert err.strip() != ""


# ---------------------------------------------------------------------------
# 18. run --out writes file and prints summary
# ---------------------------------------------------------------------------

def test_run_out_writes_file(tmp_path, monkeypatch):
    row = _make_row("decision/smoke", "2026-05-15", content="the body", tags="smoke")
    monkeypatch.setattr(
        "lapis_pm.node_identity.writable_store",
        lambda: type("MS", (), {"list_by_prefix": lambda self, p, limit=None: [row]})(),
    )
    out_path = str(tmp_path / "out.md")
    captured_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_stdout)
    code = run("30d", "decision", "", "", out_path)
    assert code == 0
    content = open(out_path).read()
    assert "decision/smoke" in content
    assert "the body" in content
    printed = captured_stdout.getvalue()
    assert "wrote" in printed
    assert out_path in printed
    assert "1 entries" in printed


# ---------------------------------------------------------------------------
# 19. run exit 1 on --out OSError
# ---------------------------------------------------------------------------

def test_run_exit_1_out_oserror(monkeypatch, capsys):
    monkeypatch.setattr(
        "lapis_pm.node_identity.writable_store",
        lambda: type("MS", (), {"list_by_prefix": lambda self, p, limit=None: []})(),
    )
    code = run("1d", "decision", "", "", "/nonexistent-dir/foo.md")
    assert code == 1
    err = capsys.readouterr().err
    assert "write failed:" in err
    assert "/nonexistent-dir/foo.md" in err
