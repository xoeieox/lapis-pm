"""AC (lapis-pm-corroboration-thinking-parse-and-truncation-loudness-v0 DoD #3):
enforce that all three chat-completion parse sites route through the one shared
`lapis_pm.completion_text.extract_completion_text` helper instead of reading
`message["content"]` (or `.get("content")`) independently.

Why AST instead of a plain grep (mirrors tests/test_reference_leg_naming_boundary.py,
the repo's existing boundary-test idiom): a blind text grep for `["content"]` /
`.get("content"` false-positives on `contractor_seat.py`'s `cached_health()`, which
reads `row.get("content")` off a mem.db row — an entirely different "content" field
(the mem storage layer's stored value), not a chat-completion message. Walking the
AST lets us require the "content" key-read to be applied to an expression that
itself touches a "message" key/attr somewhere in its subtree — the actual shape of
every real parse-site regression (`resp_json["choices"][0]["message"]["content"]`,
`(choices[0].get("message") or {}).get("content")`) — while leaving unrelated
"content" reads alone.

Two directions, both required:
  1. FALSE direction: none of the three parse-site modules independently reads
     `message["content"]` / `message.get("content")` anymore.
  2. TRUE direction: all three still import and call `extract_completion_text`, and
     the shared helper module itself is the one place that still reads `content`
     off a message dict — the read didn't just vanish, it centralized.
"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_PARSE_SITE_FILES = [
    _REPO_ROOT / "lapis_pm" / "corroboration_adapter.py",
    _REPO_ROOT / "lapis_pm" / "local_reviewer_witness.py",
    _REPO_ROOT / "lapis_pm" / "contractor_seat.py",
]

_HELPER_FILE = _REPO_ROOT / "lapis_pm" / "completion_text.py"


def _subtree_mentions_message(node: ast.AST) -> bool:
    """True if any Constant('message') or Name/Attribute 'message' appears
    anywhere within this subtree — the marker that a "content" read hangs off
    a chat-completion `message` object rather than some unrelated dict."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value == "message":
            return True
        if isinstance(sub, ast.Name) and sub.id == "message":
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == "message":
            return True
    return False


def _find_message_content_reads(path: Path) -> list[int]:
    """Return line numbers of `message["content"]` / `message.get("content")`-
    shaped reads in `path` — a "content" key access applied to an expression
    whose subtree also mentions "message"."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[int] = []
    for node in ast.walk(tree):
        # x["content"]
        if isinstance(node, ast.Subscript):
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and slice_node.value == "content":
                if _subtree_mentions_message(node.value):
                    hits.append(node.lineno)
        # x.get("content", ...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and first_arg.value == "content":
                    if _subtree_mentions_message(node.func.value):
                        hits.append(node.lineno)
    return hits


def test_no_independent_message_content_read_in_parse_site_modules():
    """FALSE direction: corroboration_adapter.py, local_reviewer_witness.py, and
    contractor_seat.py no longer read `message["content"]` / `message.get("content")`
    independently — every read routes through extract_completion_text()."""
    for path in _PARSE_SITE_FILES:
        assert path.is_file(), f"expected file missing: {path}"
        hits = _find_message_content_reads(path)
        assert not hits, (
            f"{path.relative_to(_REPO_ROOT)} has a second independent "
            f"message-content read at line(s) {hits} — route it through "
            f"lapis_pm.completion_text.extract_completion_text() instead."
        )


def test_all_three_parse_sites_import_and_call_the_shared_helper():
    """TRUE direction: the read did not just disappear — all three modules
    still import and call extract_completion_text()."""
    for path in _PARSE_SITE_FILES:
        source = path.read_text(encoding="utf-8")
        assert "extract_completion_text" in source, (
            f"{path.relative_to(_REPO_ROOT)} no longer references "
            f"extract_completion_text — expected it to route parsing through "
            f"the shared helper."
        )
        tree = ast.parse(source, filename=str(path))
        called = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "extract_completion_text"
            for node in ast.walk(tree)
        )
        assert called, (
            f"{path.relative_to(_REPO_ROOT)} imports extract_completion_text "
            f"but never calls it."
        )


def test_shared_helper_is_the_one_place_that_still_reads_message_content():
    """TRUE direction: completion_text.py itself is the one legitimate site —
    the read centralized, it didn't vanish."""
    assert _HELPER_FILE.is_file()
    hits = _find_message_content_reads(_HELPER_FILE)
    assert hits, (
        "expected lapis_pm/completion_text.py to contain the one real "
        "message.get('content') read that all three parse sites now delegate to."
    )
