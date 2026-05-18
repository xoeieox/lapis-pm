"""Unit tests for episodic.spec() last-wins semantics."""

from __future__ import annotations

from unittest.mock import patch

from lapis_pm import episodic


def _make_store(tmp_path):
    from agents_core.comments import CommentStore
    return CommentStore(root=str(tmp_path))


def test_spec_last_wins(tmp_path):
    """After two write_spec() calls, spec() returns the second (latest) body."""
    store = _make_store(tmp_path)

    with patch("lapis_pm.episodic._store", return_value=store):
        episodic.write_spec("my-target", "first spec body")
        episodic.write_spec("my-target", "second spec body")

        result = episodic.spec("my-target")

    assert result == "second spec body"


def test_spec_single_entry(tmp_path):
    """spec() returns the body when only one spec:bound comment exists."""
    store = _make_store(tmp_path)

    with patch("lapis_pm.episodic._store", return_value=store):
        episodic.write_spec("my-target", "only spec body")
        result = episodic.spec("my-target")

    assert result == "only spec body"


def test_spec_no_entry(tmp_path):
    """spec() returns None when no spec:bound comment exists."""
    store = _make_store(tmp_path)

    with patch("lapis_pm.episodic._store", return_value=store):
        result = episodic.spec("my-target")

    assert result is None
