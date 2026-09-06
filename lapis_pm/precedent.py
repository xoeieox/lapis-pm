"""Precedent actuator (U2, lapis-pm-autonomy-actuator-v0).

Erah's ratifications become structured adjudication records in mem; before a
brief is emitted, the daemon matches the current fork shape against the
ledger and, on a match where the precedent's recorded choice is the
merge-class resolution (confirm only), resolves the fork to that choice and
merges — no brief, no gem. New fork classes (no precedent) still brief as
today.

Fork class (Design 3): a deterministic tuple
`(repo, verdict_class, change_classes, loc_bucket, held_paths_class)`.
Matching is conservative by construction: ANY parse failure, missing field,
or ambiguity = no match = brief as today. The matcher never widens its own
criteria.

Verification chain (Design 5 / B2): a candidate record matches only if ALL
hold, each verified live at match time:
  (i)   body parses as well-formed `adjudication/v1` JSON;
  (ii)  `source == "human"` (precedent-sourced records are lineage-only,
        never a primary match — no self-bootstrapping chains);
  (iii) `citations` contains a `router/lapis-pm/ratification-outcomes/<eid>`
        key that exists, parses, has `ratification_outcome == "confirm"`,
        and whose `target_id` matches the record's;
  (iv)  the record's `ts` is within 24h of that outcome key's `created`.
Intent is audit-only, never matchable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# change_classes buckets (Design 3).
_CONFIG_RE = re.compile(r"(\.service$|\.timer$|\.sql$|^schema/|^systemd/)")
_TEST_RE = re.compile(r"(^tests/|/tests/|test_.*\.py$)")
_DEPLOY_RE = re.compile(r"^deploy/")
_DOCS_RE = re.compile(r"\.md$")

# loc_bucket boundaries (aligned with MAX_AUTO_LOC = 400, authority.py:61).
MAX_AUTO_LOC = 400

# 24h lineage window (Design 5 (iv)).
LINEAGE_WINDOW_H = 24

# Shadow file (Design 10 / U2.5): /srv/lapis/autonomy-shadow/precedent-calls.jsonl
SHADOW_DIR = Path("/srv/lapis/autonomy-shadow")
SHADOW_PRECEDENT_CALLS = SHADOW_DIR / "precedent-calls.jsonl"

# Lazy mem accessor (same pattern as router_portfolio._mem): resolved on first
# use so the leaf module imports cleanly off the mem master and in tests.
_mem_store = None


def _mem():
    """Return the shared module-level mem accessor (lazy init).

    On the mem master (BRIX) this is a direct MemoryStore (local = master). Off
    master it is the pm_core shared store (node-identity-checked), so precedent
    writes never diverge into a local sqlite that reverse-replication would
    clobber. Import is best-effort — a missing pm_core (test isolation) falls
    back to a fresh MemoryStore."""
    global _mem_store
    if _mem_store is None:
        try:
            from . import pm_core as _pm_core
            _mem_store = _pm_core._mem()
        except Exception:
            from agents_core.mem import MemoryStore
            _mem_store = MemoryStore()
    return _mem_store


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def derive_change_classes(changed_paths: list[str]) -> list[str]:
    """Derive the change_classes set from changed_paths (Design 3)."""
    classes = set()
    for p in changed_paths or []:
        if _CONFIG_RE.search(p):
            classes.add("config")
        if _TEST_RE.search(p):
            classes.add("test")
        if _DEPLOY_RE.search(p):
            classes.add("deploy")
        if _DOCS_RE.search(p):
            classes.add("docs")
        if not (_CONFIG_RE.search(p) or _TEST_RE.search(p)
                or _DEPLOY_RE.search(p) or _DOCS_RE.search(p)):
            classes.add("code")
    return sorted(classes)


def derive_loc_bucket(loc: int) -> str:
    """loc_bucket in {lt100, 100_400, gt400} (Design 3)."""
    if loc < 100:
        return "lt100"
    if loc <= MAX_AUTO_LOC:
        return "100_400"
    return "gt400"


def derive_fork_class(cls, verdict_class: str, *, repo: str | None = None) -> dict:
    """Derive the deterministic fork class from classifier data (Design 3).

    `cls` is a classifier object with `changed_paths` and `loc` (or an
    equivalent mapping). Pure function; no network, no LLM. Returns:
        {"repo": ..., "verdict_class": ..., "change_classes": [...],
         "loc_bucket": ..., "held_paths_class": "none"|"held"}
    """
    repo = repo or getattr(cls, "repo", None)
    changed = getattr(cls, "changed_paths", None)
    if changed is None and isinstance(cls, dict):
        changed = cls.get("changed_paths")
    loc = getattr(cls, "loc", None)
    if loc is None and isinstance(cls, dict):
        loc = cls.get("loc", 0)
    try:
        loc = int(loc or 0)
    except (TypeError, ValueError):
        loc = 0
    held = getattr(cls, "held_paths", None)
    if held is None and isinstance(cls, dict):
        held = cls.get("held_paths")
    held_class = "held" if held else "none"
    return {
        "repo": repo,
        "verdict_class": verdict_class,
        "change_classes": derive_change_classes(changed or []),
        "loc_bucket": derive_loc_bucket(loc),
        "held_paths_class": held_class,
    }


def fork_hash(fork: dict) -> str:
    """Stable short hash of the fork tuple (for action strings / records)."""
    canonical = json.dumps(
        {
            "repo": fork.get("repo"),
            "verdict_class": fork.get("verdict_class"),
            "change_classes": fork.get("change_classes", []),
            "loc_bucket": fork.get("loc_bucket"),
            "held_paths_class": fork.get("held_paths_class"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def _loc_bucket_coarser(bucket: str) -> int:
    """Rank for the 'not coarser than the precedent's' check (Design 3).

    A candidate's loc_bucket must be at least as large (coarser) than the
    precedent's for the precedent to apply — a precedent covering a larger
    LOC range covers a smaller one.
    """
    return {"lt100": 0, "100_400": 1, "gt400": 2}.get(bucket, -1)


def _matches(repo: str, fork: dict, rec: dict) -> bool:
    """Deterministic fork-tuple match (Design 3). Intent is NOT consulted."""
    if rec.get("schema") != "adjudication/v1":
        return False
    if rec.get("source") != "human":
        return False
    if rec.get("chosen") != "confirm":
        return False
    rf = rec.get("fork")
    if not isinstance(rf, dict):
        return False
    if rf.get("repo") != repo:
        return False
    if rf.get("verdict_class") != "clean":
        return False
    if rf.get("verdict_class") != fork.get("verdict_class"):
        return False
    if set(rf.get("change_classes", [])) != set(fork.get("change_classes", [])):
        return False
    if _loc_bucket_coarser(fork.get("loc_bucket")) < _loc_bucket_coarser(
            rf.get("loc_bucket")):
        return False
    if rf.get("held_paths_class") != "none":
        return False
    if fork.get("held_paths_class") != "none":
        return False
    return True


def _lineage_ok(mem, rec: dict) -> bool:
    """Design 5 (iii)+(iv): verify the cited ratify outcome live at match time."""
    citations = rec.get("citations") or []
    if not isinstance(citations, list):
        return False
    rec_ts = _parse_iso(rec.get("ts"))
    if rec_ts is None:
        return False
    for cit in citations:
        if not isinstance(cit, str) or not cit.startswith(
                "router/lapis-pm/ratification-outcomes/"):
            continue
        try:
            outcome = mem.get(cit)
        except Exception:
            return False
        if not outcome:
            return False
        try:
            odata = json.loads(outcome.get("content", ""))
        except (ValueError, TypeError):
            return False
        if not isinstance(odata, dict):
            return False
        if odata.get("ratification_outcome") != "confirm":
            return False
        if odata.get("target_id") != rec.get("target_id"):
            return False
        created = _parse_iso(outcome.get("created") or odata.get("created"))
        if created is None:
            return False
        if abs((rec_ts - created).total_seconds()) > LINEAGE_WINDOW_H * 3600:
            return False
        return True
    return False


def find_precedent(mem, repo: str, fork: dict) -> dict | None:
    """Query mem for `adjudication`-tagged keys, apply the Design 5 chain.

    Returns the first record that matches the fork AND passes the full
    verification chain, else None. Malformed / lineage-failed records are
    skipped (counted in the returned debug field via the module-level
    `find_precedent.last_skipped` for the shadow file)."""
    find_precedent.last_skipped = 0
    candidates = []
    try:
        candidates = mem.search("adjudication") or []
    except Exception:
        return None
    for key in candidates:
        try:
            raw = mem.get(key)
            if not raw:
                continue
            body = raw.get("content", "")
            # Extract the fenced JSON block (Design 5 body carries a fenced JSON
            # block); fall back to the whole body if no fence.
            m = re.search(r"```json\s*(\{.*?\})\s*```", body, re.DOTALL)
            payload_text = m.group(1) if m else body
            rec = json.loads(payload_text)
            if not isinstance(rec, dict):
                find_precedent.last_skipped += 1
                continue
            if not _matches(repo, fork, rec):
                continue
            if not _lineage_ok(mem, rec):
                find_precedent.last_skipped += 1
                continue
            rec["_key"] = key
            return rec
        except (ValueError, TypeError, AttributeError):
            find_precedent.last_skipped += 1
            continue
    return None


def _fork_from_options_sibling(mem, target_id: str) -> dict | None:
    """Copy the fork block from the `pm:brief-options` sibling (Design 5).

    Returns None when the sibling is missing/unparseable (briefs predating
    this spec, non-PR briefs) — the record is still written with `fork: null`.
    """
    try:
        raw = mem.get(f"pm:brief-options:{target_id}")
        if not raw:
            return None
        data = json.loads(raw.get("content", ""))
        fork = data.get("fork_class") if isinstance(data, dict) else None
        return fork if isinstance(fork, dict) else None
    except Exception:
        return None


def write_adjudication(mem, *, target_id: str, pr_number, outcome: str,
                       repo: str, fork: dict | None, intent: str,
                       citations: list[str], invoked_interactive: bool = False,
                       source: str = "human",
                       precedent_of: str | None = None) -> str:
    """Write (or idempotently update) the adjudication record (Design 5).

    Key: `decision/adjudication/<tid>-<pr>-<outcome>`. Idempotent per
    (tid, pr, outcome): re-ratifying the same triple updates `ts` and
    `intent`, never creates a second key.
    """
    key = f"decision/adjudication/{target_id}-{pr_number}-{outcome}"
    body = {
        "schema": "adjudication/v1",
        "ts": _now_iso(),
        "repo": repo,
        "target_id": target_id,
        "fork": fork,
        "chosen": outcome,
        "intent": intent or "",
        "source": source,
        "pr": pr_number,
        "citations": list(citations or []),
        "invoked_interactive": bool(invoked_interactive),
    }
    if source == "precedent" and precedent_of:
        body["precedent_of"] = precedent_of
    content = (
        f"<!-- adjudication record, schema adjudication/v1 -->\n"
        f"```json\n{json.dumps(body, ensure_ascii=False, sort_keys=True)}\n```\n"
    )
    tags = ["lapis-pm", "adjudication", f"target:{target_id}"]
    mem.set(key, content, tags=tags)
    return key


def resolution_key(target_id: str, pr_number, fork: dict) -> str:
    """mem key for the actuator's own resolution record (source: precedent)."""
    return f"decision/adjudication/{target_id}-{pr_number}-precedent-{fork_hash(fork)}"


def write_resolution(mem, *, target_id: str, pr_number, repo: str, fork: dict,
                     matched_precedent: str) -> str:
    """Write the actuator's resolution record (source: precedent,
    precedent_of: the matched key). The lineage is grep-able."""
    key = resolution_key(target_id, pr_number, fork)
    body = {
        "schema": "adjudication/v1",
        "ts": _now_iso(),
        "repo": repo,
        "fork": fork,
        "chosen": "confirm",
        "intent": "",
        "source": "precedent",
        "pr": pr_number,
        "citations": [matched_precedent],
        "precedent_of": matched_precedent,
    }
    content = (
        f"<!-- adjudication record, schema adjudication/v1 -->\n"
        f"```json\n{json.dumps(body, ensure_ascii=False, sort_keys=True)}\n```\n"
    )
    tags = ["lapis-pm", "adjudication", f"target:{target_id}", "precedent"]
    mem.set(key, content, tags=tags)
    return key


def record_shadow_call(target_id: str, repo: str, pr_number, fork: dict,
                       *, would_merge: bool, would_merge_reason: str,
                       matched_precedent: str | None, no_match_reason: str | None,
                       dial: str, skipped: int = 0) -> None:
    """Append a would-be call to /srv/lapis/autonomy-shadow/precedent-calls.jsonl
    (schema precedent-call/v1, mirroring the hold_shadow record style).
    Never raises."""
    try:
        SHADOW_DIR.mkdir(parents=True, exist_ok=True)
        line = {
            "schema_version": "precedent-call/v1",
            "ts_utc": _now_iso(),
            "target_id": target_id,
            "repo": repo,
            "pr_number": pr_number,
            "fork": fork,
            "would_merge": bool(would_merge),
            "would_merge_reason": would_merge_reason,
            "matched_precedent": matched_precedent,
            "no_match_reason": no_match_reason,
            "dial": dial,
            "skipped_malformed": int(skipped or 0),
        }
        with open(SHADOW_PRECEDENT_CALLS, "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("precedent shadow call write failed: %s", e)


def class_record_key(repo: str, verdict_class: str,
                     change_classes: list[str], loc_bucket: str) -> str:
    """mem key for the fork-class record (Design 8)."""
    slug = ",".join(sorted(change_classes or [])) or "none"
    return (f"pm/autonomy-class/{repo}/{verdict_class}/{slug}/{loc_bucket}")
