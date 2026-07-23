"""Anchored-agent-signed PM directives (Bridge A).

Capture side: `emit_signed_directive` deposits a signed `pm_directive` record
under the pm-review agent key (never `Erah` / `human:*`) and writes the
`human:directive` comment with an honest agent author, referencing the
deposit by a `zephyr:deposit=` tag.

Accept side: `verify_directive` composes the read-only verification sequence
(verify_row -> agent_acceptance -> content-binding) that `pm_core._encode_user_comments`
gates honor-vs-quarantine on.

lapis-pm never statically imports zephyr (or archetypes_core, whose
`InputRef` we use to carry directive content) - every callable is resolved
lazily via an env-spec "<module>:<attr>", mirroring the existing
`_SLOT_DEPOSIT_RECORDER_SPEC` convention in pm_core.py. This keeps the
attribution substrate decoupled and matches the fact that a missing
archetypes-core install should only fail at deposit time, not module load.
"""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from lapis_pm import episodic

PM_REVIEW_AGENT_ID = "lapis-pm:pm-review-actor"

TAG_DEPOSIT_PREFIX = "zephyr:deposit="
TAG_CID_PREFIX = "zephyr:cid="

RUNBOOK_HINT = (
    "re-run `vouch-agent` for the pm-review key (or fix the ZEPHYR_HUMAN_ROOT_ANCHOR "
    "pin), restart the daemon, then flip LAPIS_PM_DIRECTIVE_ENFORCEMENT back to "
    "'enforce'. Directives keep flowing under 'observe' while you fix it."
)

# --- Lazy env-spec import resolution (no static zephyr / archetypes_core import) ---

_ZEPHYR_SIGN_SPEC = os.environ.get("LAPIS_PM_ZEPHYR_SIGN", "zephyr.attribution:record_signed")
_ZEPHYR_VERIFY_SPEC = os.environ.get("LAPIS_PM_ZEPHYR_VERIFY", "zephyr.attribution:verify_row")
_ZEPHYR_ACCEPT_SPEC = os.environ.get("LAPIS_PM_ZEPHYR_ACCEPT", "zephyr.registry:agent_acceptance")
_ZEPHYR_SIGNER_SPEC = os.environ.get("LAPIS_PM_ZEPHYR_SIGNER", "zephyr.signing:AgentSigner")
_ARCHETYPES_INPUT_REF_SPEC = os.environ.get(
    "LAPIS_PM_ARCHETYPES_INPUT_REF", "archetypes_core.provenance:InputRef"
)

_DEFAULT_PM_REVIEW_AGENT_KEY = "/data/zephyr/agent-keys/lapis-pm_pm-review-actor.key"


def _resolve(spec: str):
    module_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _resolve_pm_review_signer():
    """AgentSigner over the pm-review agent key (path from LAPIS_PM_PM_REVIEW_AGENT_KEY)."""
    agent_signer_cls = _resolve(_ZEPHYR_SIGNER_SPEC)
    key_path = Path(os.environ.get("LAPIS_PM_PM_REVIEW_AGENT_KEY", _DEFAULT_PM_REVIEW_AGENT_KEY))
    return agent_signer_cls(key_path)


# --- Config (observe -> enforce rollout; briefing threshold) -----------------

def directive_enforcement_mode() -> str:
    """'observe' (default; honor-with-loud-warning) or 'enforce' (quarantine)."""
    return os.environ.get("LAPIS_PM_DIRECTIVE_ENFORCEMENT", "observe")


def directive_brief_on() -> str:
    """'well_formed' (default) | 'all' | 'none' — controls which faults raise a brief."""
    return os.environ.get("LAPIS_PM_DIRECTIVE_BRIEF_ON", "well_formed")


# --- Sign side (capture; always permitted) ------------------------------------

def emit_signed_directive(tid: str, content: str, *, signer=None) -> dict:
    """Deposit a signed pm_directive under the pm-review agent key, then write the
    human:directive comment with an honest agent author. Deposit-first ordering:
    if the comment append fails, the deposit is an inert, unreferenced orphan —
    never the reverse (never write an honored comment before its backing deposit
    exists).

    Returns {"manifest_hash", "pubkey_id", "comment"}.
    """
    record_signed = _resolve(_ZEPHYR_SIGN_SPEC)
    input_ref_cls = _resolve(_ARCHETYPES_INPUT_REF_SPEC)

    cid = str(uuid.uuid4())
    payload = {"kind": "pm_directive", "target_id": tid, "cid": cid, "content": content}

    result = record_signed(
        payload,
        agent_id=PM_REVIEW_AGENT_ID,
        tool="lapis-pm:pm-directive",
        store_kind="pm_directive",
        key=f"{tid}:{cid}",
        signer=signer or _resolve_pm_review_signer(),
        # scope_id/job_id/input_refs ride in Provenance.to_dict() (unlike the
        # payload arg, which only contributes to the manifest_hash — it is
        # never itself stored), so the accept-side content-binding check
        # (verify_directive) can read {target_id, cid, content} straight back
        # off the deposit row without hand-reconstructing the signing target.
        scope_id=tid,
        job_id=cid,
        input_refs=[input_ref_cls(ref=content, type="claim")],
    )
    manifest_hash = result["manifest_hash"]

    # CommentStore.append() mints its own Comment.id (no caller-supplied-id
    # hook) — cid is generated here, before the comment exists, so the
    # deposit-first ordering (below) can sign over it. The cid rides in its
    # own tag rather than relying on comment.id for the content-binding
    # check (verify_directive) to compare against.
    comment = episodic._store().append(
        tid,
        content,
        author=PM_REVIEW_AGENT_ID,
        author_type="agent",
        tags=[
            episodic.TAG_HUMAN_DIRECTIVE,
            f"{TAG_DEPOSIT_PREFIX}{manifest_hash}",
            f"{TAG_CID_PREFIX}{cid}",
        ],
    )
    return {"manifest_hash": manifest_hash, "pubkey_id": result.get("pubkey_id"), "comment": comment}


# --- Accept side (enforcement; verified at acceptance) ------------------------

@dataclass
class DirectiveVerdict:
    ok: bool
    reason: str
    pubkey_id: str | None = None
    root_tier: str | None = None


def deposit_ref(tags: list[str]) -> str | None:
    """Parse the "zephyr:deposit=<manifest_hash>" tag, or None if absent."""
    for t in tags:
        if t.startswith(TAG_DEPOSIT_PREFIX):
            return t[len(TAG_DEPOSIT_PREFIX):]
    return None


def _cid_ref(tags: list[str]) -> str | None:
    """Parse the "zephyr:cid=<cid>" tag, or None if absent."""
    for t in tags:
        if t.startswith(TAG_CID_PREFIX):
            return t[len(TAG_CID_PREFIX):]
    return None


def _fetch_deposit_row(manifest_hash: str) -> dict | None:
    """Read-only SELECT against ZEPHYR_ATTRIBUTION_DB. Never writes."""
    db_path = os.environ.get("ZEPHYR_ATTRIBUTION_DB", "/data/zephyr/attribution.db")
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM deposits WHERE manifest_hash = ?", (manifest_hash,)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def verify_directive(target_id: str, comment) -> DirectiveVerdict:
    """The five-check acceptance predicate (spec §4.3): deposit tag -> row ->
    signature valid -> anchored to Erah's root -> content-bound to this live
    comment (anti-splice). All five must pass to honor.
    """
    mh = deposit_ref(comment.tags)
    if mh is None:
        return DirectiveVerdict(False, "unsigned")

    row = _fetch_deposit_row(mh)
    if row is None:
        return DirectiveVerdict(False, "missing_deposit")

    verify_row = _resolve(_ZEPHYR_VERIFY_SPEC)
    vr = verify_row(row, registry=None)
    if not vr.verified:
        return DirectiveVerdict(False, vr.status, pubkey_id=row.get("pubkey_id"))

    agent_acceptance = _resolve(_ZEPHYR_ACCEPT_SPEC)
    verdict = agent_acceptance(row.get("pubkey_id"), registry=None)
    if not verdict["accept"]:
        return DirectiveVerdict(
            False, f"state={verdict['state']}", pubkey_id=row.get("pubkey_id")
        )

    try:
        prov = json.loads(row.get("provenance_json") or "{}")
    except json.JSONDecodeError:
        return DirectiveVerdict(False, "content_mismatch", pubkey_id=row.get("pubkey_id"))

    input_refs = prov.get("input_refs") or []
    stored_content = input_refs[0].get("ref") if input_refs else None
    live_cid = _cid_ref(comment.tags)
    if (
        prov.get("scope_id") != target_id
        or prov.get("job_id") != live_cid
        or stored_content != comment.content
    ):
        return DirectiveVerdict(False, "content_mismatch", pubkey_id=row.get("pubkey_id"))

    return DirectiveVerdict(
        True, "anchored", pubkey_id=row.get("pubkey_id"), root_tier=verdict["root_tier"]
    )
