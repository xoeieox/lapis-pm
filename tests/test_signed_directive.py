"""Tests for lapis_pm.signed_directive (Bridge A — anchored-agent-signed PM
directives, verified at acceptance). Spec: lapis-pm-signed-directive-acceptance-v0.

All identities are synthetic. Every test runs against a scratch ZEPHYR_KEYS_DIR /
ZEPHYR_ATTRIBUTION_DB under tmp_path via the ``scratch_zephyr`` fixture, which
overrides the zephyr singleton getters directly (module-level default
parameter values are frozen at def time, so env-var patching alone is not
enough once zephyr.attribution / zephyr.registry have already been imported
in this test session) — never the real /data/zephyr/*.
"""

from __future__ import annotations

import json
import os
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import episodic, pm_core, signed_directive


# ---------------------------------------------------------------------------
# Scratch zephyr substrate
# ---------------------------------------------------------------------------


@pytest.fixture()
def scratch_zephyr(tmp_path, monkeypatch):
    import zephyr.attribution as zattr
    import zephyr.registry as zreg

    keys_dir = tmp_path / "keys"
    db_path = tmp_path / "attribution.db"
    registry = zreg.PubkeyRegistry(keys_dir)
    recorder = zattr.AttributionLog(db_path)

    monkeypatch.setenv("ZEPHYR_KEYS_DIR", str(keys_dir))
    monkeypatch.setenv("ZEPHYR_ATTRIBUTION_DB", str(db_path))
    monkeypatch.setattr(zreg, "get_registry", lambda: registry)
    monkeypatch.setattr(zattr, "get_recorder", lambda: recorder)
    monkeypatch.delenv("ZEPHYR_HUMAN_ROOT_ANCHOR", raising=False)
    monkeypatch.setattr(zreg, "ZEPHYR_HUMAN_ROOT_ANCHOR", None)

    return SimpleNamespace(keys_dir=keys_dir, db_path=db_path, registry=registry, recorder=recorder)


@pytest.fixture()
def root_signer(tmp_path):
    from zephyr.signing import AgentSigner

    return AgentSigner(tmp_path / "agent-keys" / "root.key")


@pytest.fixture()
def agent_signer(tmp_path):
    from zephyr.signing import AgentSigner

    return AgentSigner(tmp_path / "agent-keys" / "pm-review.key")


@pytest.fixture()
def pinned_root(monkeypatch, scratch_zephyr, root_signer):
    """Register root_signer as the human root and pin it. Returns its pkid."""
    import zephyr.registry as zreg

    root_pkid = root_signer.pubkey_id_str
    monkeypatch.setattr(zreg, "ZEPHYR_HUMAN_ROOT_ANCHOR", root_pkid)
    scratch_zephyr.registry.register(
        root_pkid,
        root_signer.public_key_bytes.hex(),
        "human:Erah-synthetic",
        role="human",
        assurance_tier="software",
    )
    return root_pkid


def _register_agent(scratch_zephyr, agent_signer, agent_id=signed_directive.PM_REVIEW_AGENT_ID):
    return scratch_zephyr.registry.register(
        agent_signer.pubkey_id_str,
        agent_signer.public_key_bytes.hex(),
        agent_id,
        role="agent",
    )


def _vouch(scratch_zephyr, root_signer, agent_entry):
    from zephyr.registry import agent_canonical_target

    target = agent_canonical_target(
        agent_entry["pubkey_id"],
        agent_entry["public_key_hex"],
        agent_entry["agent_id"],
        agent_entry.get("valid_from", ""),
    )
    sig = root_signer.sign(target)
    return scratch_zephyr.registry.set_registered_by(
        agent_entry["pubkey_id"], {"signer_pkid": root_signer.pubkey_id_str, "signature": sig}
    )


def _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root):
    entry = _register_agent(scratch_zephyr, agent_signer)
    return _vouch(scratch_zephyr, root_signer, entry)


# ---------------------------------------------------------------------------
# AC1/AC2 — emit_signed_directive (capture side)
# ---------------------------------------------------------------------------


def test_emit_signed_directive_writes_deposit_and_signed_comment(
    scratch_zephyr, root_signer, agent_signer, pinned_root
):
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)

    result = signed_directive.emit_signed_directive(
        "my-target", "do the surgical thing", signer=agent_signer
    )

    row = scratch_zephyr.recorder.get(result["manifest_hash"], registry=scratch_zephyr.registry)
    assert row is not None
    assert row["verified"] is True
    assert row["pubkey_id"] == agent_signer.pubkey_id_str
    assert row["store_kind"] == "pm_directive"

    comment = result["comment"]
    assert comment.author == signed_directive.PM_REVIEW_AGENT_ID
    assert comment.author_type == "agent"
    assert episodic.TAG_HUMAN_DIRECTIVE in comment.tags
    assert f"zephyr:deposit={result['manifest_hash']}" in comment.tags

    stored = episodic._store().list("my-target")
    assert len(stored) == 1
    assert stored[0].id == comment.id
    # Round-trip through the on-disk JSONL store (agents_core.comments._KNOWN_FIELDS
    # filters known top-level Comment fields, not individual tag strings — both the
    # deposit tag and the cid tag are members of the same `tags` list and must
    # survive persistence identically).
    assert f"zephyr:deposit={result['manifest_hash']}" in stored[0].tags
    assert signed_directive._cid_ref(stored[0].tags) == signed_directive._cid_ref(comment.tags)


def test_emit_signed_directive_deposit_first_orphan_on_comment_failure(
    scratch_zephyr, root_signer, agent_signer, pinned_root
):
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)

    with patch.object(episodic, "_store") as mock_store:
        mock_store.return_value.append.side_effect = RuntimeError("disk full")
        with pytest.raises(RuntimeError):
            signed_directive.emit_signed_directive(
                "my-target", "surgical change", signer=agent_signer
            )

    # The deposit itself is an inert orphan — recorded, but no comment references it.
    assert scratch_zephyr.recorder.count() == 1
    assert episodic._store().list("my-target") == []


# ---------------------------------------------------------------------------
# AC3 — happy-path honor
# ---------------------------------------------------------------------------


def test_verify_directive_honors_anchored_content_bound_directive(
    scratch_zephyr, root_signer, agent_signer, pinned_root
):
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "do the surgical thing", signer=agent_signer
    )

    verdict = signed_directive.verify_directive("my-target", result["comment"])
    assert verdict.ok is True
    assert verdict.reason == "anchored"
    assert verdict.root_tier == "software"
    assert verdict.pubkey_id == agent_signer.pubkey_id_str


def test_encode_user_comments_honors_accepted_directive(
    scratch_zephyr, root_signer, agent_signer, pinned_root, monkeypatch
):
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "enforce")
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "do the surgical thing", signer=agent_signer
    )

    with patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs:
        honored = pm_core._encode_user_comments("my-target", [result["comment"]])

    assert honored == [result["comment"]]
    for call in mock_obs.call_args_list:
        assert "HELD FAULT" not in call.args[1]


# ---------------------------------------------------------------------------
# AC4 — absent deposit tag ("unsigned")
# ---------------------------------------------------------------------------


def _plain_directive_comment(tid="my-target", content="hand-written directive"):
    return episodic._store().append(
        tid, content, author="Erah", author_type="user", tags=[episodic.TAG_HUMAN_DIRECTIVE]
    )


def test_verify_directive_absent_tag_is_unsigned_fault(scratch_zephyr):
    c = _plain_directive_comment()
    verdict = signed_directive.verify_directive("my-target", c)
    assert verdict.ok is False
    assert verdict.reason == "unsigned"


def test_encode_user_comments_enforce_quarantines_unsigned(scratch_zephyr, monkeypatch):
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "enforce")
    c = _plain_directive_comment()

    with patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs:
        honored = pm_core._encode_user_comments("my-target", [c])

    assert honored == []
    fault_calls = [call for call in mock_obs.call_args_list if "HELD FAULT" in call.args[1]]
    assert len(fault_calls) == 1
    assert "unsigned" in fault_calls[0].args[1]
    assert "directive:quarantined" in fault_calls[0].kwargs["extra_tags"]


def test_encode_user_comments_observe_honors_unsigned_with_warning(scratch_zephyr, monkeypatch):
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "observe")
    c = _plain_directive_comment()

    with patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs:
        honored = pm_core._encode_user_comments("my-target", [c])

    assert honored == [c]
    fault_calls = [call for call in mock_obs.call_args_list if "HELD FAULT" in call.args[1]]
    assert len(fault_calls) == 1
    assert "directive:observed-unsigned" in fault_calls[0].kwargs["extra_tags"]


# ---------------------------------------------------------------------------
# AC5 — self_asserted / revoked
# ---------------------------------------------------------------------------


def test_verify_directive_self_asserted_not_honored(scratch_zephyr, agent_signer):
    # Agent key minted but never vouched — no pinned root at all.
    _register_agent(scratch_zephyr, agent_signer)
    result = signed_directive.emit_signed_directive(
        "my-target", "unvouched directive", signer=agent_signer
    )

    verdict = signed_directive.verify_directive("my-target", result["comment"])
    assert verdict.ok is False
    assert verdict.reason == "state=self_asserted"


def test_verify_directive_revoked_key_not_honored(
    scratch_zephyr, root_signer, agent_signer, pinned_root
):
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "about to be revoked", signer=agent_signer
    )
    scratch_zephyr.registry.revoke(agent_signer.pubkey_id_str, reason="compromised")

    verdict = signed_directive.verify_directive("my-target", result["comment"])
    assert verdict.ok is False
    assert verdict.reason in ("suspect", "state=invalid")


# ---------------------------------------------------------------------------
# AC6 — content-binding / anti-splice
# ---------------------------------------------------------------------------


def test_verify_directive_content_mismatch_is_rejected(
    scratch_zephyr, root_signer, agent_signer, pinned_root
):
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "original content", signer=agent_signer
    )

    # Splice the valid, anchored deposit tag onto an unrelated comment (same
    # signer, different content/cid) — the classic anti-splice attack.
    forged = episodic._store().append(
        "my-target",
        "a completely different directive",
        author=signed_directive.PM_REVIEW_AGENT_ID,
        author_type="agent",
        tags=[episodic.TAG_HUMAN_DIRECTIVE, f"zephyr:deposit={result['manifest_hash']}"],
    )

    verdict = signed_directive.verify_directive("my-target", forged)
    assert verdict.ok is False
    assert verdict.reason == "content_mismatch"


def test_verify_directive_malformed_provenance_shape_fails_closed(
    scratch_zephyr, root_signer, agent_signer, pinned_root
):
    """A signing agent fully controls its own deposit payload shape (the
    signature covers manifest_hash, not a live re-hash of provenance_json —
    see zephyr.attribution.verify_row). An adversarial-but-anchored agent
    could shape input_refs as anything; the content-binding parse must fail
    closed as content_mismatch instead of raising AttributeError/TypeError
    out of the acceptance gate."""
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "original content", signer=agent_signer
    )

    conn = sqlite3.connect(scratch_zephyr.db_path)
    try:
        conn.execute(
            "UPDATE deposits SET provenance_json = ? WHERE manifest_hash = ?",
            (
                json.dumps({"scope_id": "my-target", "job_id": "whatever", "input_refs": ["not-a-dict"]}),
                result["manifest_hash"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    verdict = signed_directive.verify_directive("my-target", result["comment"])
    assert verdict.ok is False
    assert verdict.reason == "content_mismatch"


def test_verify_directive_missing_deposit_row(scratch_zephyr):
    c = episodic._store().append(
        "my-target",
        "claims a deposit that was never recorded",
        author=signed_directive.PM_REVIEW_AGENT_ID,
        author_type="agent",
        tags=[episodic.TAG_HUMAN_DIRECTIVE, "zephyr:deposit=sha256:doesnotexist"],
    )
    verdict = signed_directive.verify_directive("my-target", c)
    assert verdict.ok is False
    assert verdict.reason == "missing_deposit"


# ---------------------------------------------------------------------------
# AC7 — is_honored_directive / outstanding bookkeeping
# ---------------------------------------------------------------------------


def test_is_honored_directive_legacy_user_and_signed_agent(
    scratch_zephyr, root_signer, agent_signer, pinned_root
):
    legacy = _plain_directive_comment(content="legacy hand-write")
    assert episodic.is_honored_directive(legacy) is True

    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "signed directive", signer=agent_signer
    )
    assert episodic.is_honored_directive(result["comment"]) is True

    unsigned_agent = episodic._store().append(
        "my-target", "agent-authored but no deposit tag",
        author=signed_directive.PM_REVIEW_AGENT_ID, author_type="agent",
        tags=[episodic.TAG_HUMAN_DIRECTIVE],
    )
    assert episodic.is_honored_directive(unsigned_agent) is False


def test_has_outstanding_directive_counts_signed_agent_directive(
    scratch_zephyr, root_signer, agent_signer, pinned_root
):
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "signed directive", signer=agent_signer
    )
    out = episodic.has_outstanding_directive("my-target", None)
    assert out is not None
    assert out.id == result["comment"].id


# ---------------------------------------------------------------------------
# AC9 — no static zephyr import; read path performs zero writes
# ---------------------------------------------------------------------------


def test_no_static_zephyr_import_in_lapis_pm():
    import ast
    import pathlib

    pkg_dir = pathlib.Path(signed_directive.__file__).parent
    offenders = []
    for path in pkg_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "zephyr" or alias.name.startswith("zephyr."):
                        offenders.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "zephyr" or node.module.startswith("zephyr.")):
                    if node.col_offset == 0:
                        offenders.append(f"{path}: from {node.module} import ...")
    assert offenders == [], f"static zephyr import(s) found: {offenders}"


def test_verify_directive_read_path_is_read_only(
    scratch_zephyr, root_signer, agent_signer, pinned_root
):
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "read-only check", signer=agent_signer
    )
    before = scratch_zephyr.recorder.count()

    signed_directive.verify_directive("my-target", result["comment"])

    assert scratch_zephyr.recorder.count() == before


# ---------------------------------------------------------------------------
# AC11 — enforcement modes matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["observe", "enforce"])
def test_encode_user_comments_accepted_directive_honored_both_modes(
    scratch_zephyr, root_signer, agent_signer, pinned_root, monkeypatch, mode
):
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", mode)
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "accepted regardless of mode", signer=agent_signer
    )

    with patch("lapis_pm.pm_core.episodic.write_observation"):
        honored = pm_core._encode_user_comments("my-target", [result["comment"]])

    assert honored == [result["comment"]]


# ---------------------------------------------------------------------------
# AC12 — diagnostic held-fault content + briefing threshold
# ---------------------------------------------------------------------------


def test_held_fault_observation_contains_remediation_and_signer(scratch_zephyr, monkeypatch):
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "enforce")
    c = _plain_directive_comment()

    with patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs:
        pm_core._encode_user_comments("my-target", [c])

    fault_calls = [call for call in mock_obs.call_args_list if "HELD FAULT" in call.args[1]]
    assert len(fault_calls) == 1
    body = fault_calls[0].args[1]
    assert "unsigned" in body
    assert "signer=none" in body
    assert signed_directive.RUNBOOK_HINT in body


def test_brief_on_well_formed_absent_tag_logs_no_brief(scratch_zephyr, monkeypatch):
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "enforce")
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_BRIEF_ON", "well_formed")
    c = _plain_directive_comment()  # no deposit tag -> not well-formed

    with patch("lapis_pm.pm_core.episodic.write_observation"), \
         patch("lapis_pm.pm_core.brief.synthesize") as mock_synth, \
         patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set_outstanding:
        pm_core._encode_user_comments("my-target", [c])

    mock_synth.assert_not_called()
    mock_set_outstanding.assert_not_called()


def test_brief_on_well_formed_tagged_fault_briefs(
    scratch_zephyr, agent_signer, monkeypatch
):
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "enforce")
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_BRIEF_ON", "well_formed")
    # Agent key minted but unvouched -> self_asserted fault, but well-formed
    # (a real signing attempt, carries a deposit tag).
    _register_agent(scratch_zephyr, agent_signer)
    result = signed_directive.emit_signed_directive(
        "my-target", "well-formed but unvouched", signer=agent_signer
    )

    fake_brief = MagicMock()
    fake_brief.comment_id = "cmt-fault-1"
    with patch("lapis_pm.pm_core.episodic.write_observation"), \
         patch("lapis_pm.pm_core.brief.synthesize", return_value=fake_brief) as mock_synth, \
         patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set_outstanding:
        pm_core._encode_user_comments("my-target", [result["comment"]])

    mock_synth.assert_called_once()
    mock_set_outstanding.assert_called_once_with("my-target", fake_brief)


# ---------------------------------------------------------------------------
# AC13 — untagged directive-shaped comments surface a WARN (silent-drop
#        guard, spec cr-bundle-item-lapis-pm-1593db884c)
# ---------------------------------------------------------------------------


def _untagged_directive_shaped_comment(tid="my-target", content=None):
    """A directive-shaped JSON payload (the signed_directive deposit payload
    shape) deposited as a comment WITHOUT the human:directive tag."""
    if content is None:
        content = json.dumps(
            {
                "kind": "pm_directive",
                "target_id": tid,
                "cid": "cmt-untagged-1",
                "content": "surgical change",
            }
        )
    return episodic._store().append(
        tid, content, author="Erah", author_type="user", tags=[]
    )


def test_encode_user_comments_untagged_directive_shaped_logs_warn(
    scratch_zephyr, monkeypatch, caplog
):
    """An untagged directive-shaped comment is skipped (fail-open, no
    encoding) but the skip is surfaced: a WARN naming the comment id and
    the target id."""
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "observe")
    c = _untagged_directive_shaped_comment()

    with patch("lapis_pm.pm_core.episodic.write_observation") as mock_obs, \
         caplog.at_level("WARNING", logger="lapis_pm.pm_core"):
        honored = pm_core._encode_user_comments("my-target", [c])

    # Fail-open: the untagged comment is NOT honored (behavior unchanged).
    assert honored == []
    # No held-fault observation — the comment never entered the gate.
    assert not any("HELD FAULT" in call.args[1] for call in mock_obs.call_args_list)
    # The WARN names the comment id + target id.
    warns = [rec for rec in caplog.records if rec.levelname == "WARNING"]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    assert c.id in msg
    assert "my-target" in msg
    assert "human:directive" in msg


def test_encode_user_comments_untagged_plain_comment_no_warn(
    scratch_zephyr, monkeypatch, caplog
):
    """A non-directive-shaped untagged comment takes no WARN path (the
    shape check does not false-positive on ordinary chatter)."""
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "observe")
    c = episodic._store().append(
        "my-target", "just some chatter, not a directive",
        author="Erah", author_type="user", tags=[],
    )

    with caplog.at_level("WARNING", logger="lapis_pm.pm_core"):
        honored = pm_core._encode_user_comments("my-target", [c])

    assert honored == []
    assert not [rec for rec in caplog.records if rec.levelname == "WARNING"]


def test_encode_user_comments_tagged_directive_no_untagged_warn(
    scratch_zephyr, root_signer, agent_signer, pinned_root, monkeypatch, caplog
):
    """Tagged directive encoding is unchanged: the acceptance gate runs,
    the comment is honored, and the untagged-drop WARN is NOT taken."""
    monkeypatch.setenv("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "enforce")
    _vouched_agent(scratch_zephyr, root_signer, agent_signer, pinned_root)
    result = signed_directive.emit_signed_directive(
        "my-target", "do the surgical thing", signer=agent_signer
    )

    with patch("lapis_pm.pm_core.episodic.write_observation"), \
         caplog.at_level("WARNING", logger="lapis_pm.pm_core"):
        honored = pm_core._encode_user_comments("my-target", [result["comment"]])

    assert honored == [result["comment"]]
    warns = [rec for rec in caplog.records if rec.levelname == "WARNING"]
    assert not any("lacks the required" in rec.getMessage() for rec in warns)
