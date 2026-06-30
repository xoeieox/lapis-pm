"""Prior-Art Scout runner.

Nightly scan of committed roadmap items against the open web via Dowser.

Flip sequencing (local-only path):
  swarm flip -> doorman serving-gate -> read_batch (QUEST operator)
  -> big flip -> doorman serving-gate -> critique_batch (GW 122B)

Two-tier priority per run:
  Priority 1: architecture/* + strategy/* in-flight — exhaustive each night.
  Priority 2: decision/* in-flight — rotating cursor, 30/night.

Stub mode: PRIOR_ART_SCOUT_STUB=1 bypasses all LLM + Dowser calls.
"""
from __future__ import annotations

import bisect
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .query_gen import item_to_request
from .schema import ScoutItem, ScoutRun
from .sidecar import is_known_hopeless, load_sidecar, write_sidecar

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FLIP_CONTROLLER_URL = os.environ.get("FLIP_CONTROLLER_URL", "http://203.0.113.10:8408")
FLIP_CONTROLLER_TOKEN = os.environ.get("FLIP_CONTROLLER_TOKEN", "")
DOORMAN_URL = os.environ.get("DOORMAN_URL", "http://127.0.0.1:8407")
FLIP_SERVE_TIMEOUT = int(os.environ.get("FLIP_SERVE_TIMEOUT", "240"))

_CURSOR_PATH = Path("/srv/lapis/prior-art-scout/cursor.json")
_PRIORITY2_BATCH_SIZE = 30
_SNAPSHOT_CAP = 500

_CREDIBILITY_FLOOR = 0.3
_DENYLIST_SUBSTRINGS = [
    "reddit.com", "quora.com", "pinterest.com", "medium.com/tag",
    "fiverr.com", "upwork.com", "guru.com", "freelancer.com",
    "blogspot", "wordpress.com/20",
    "seo", "affiliate",
]

# ---------------------------------------------------------------------------
# GW flip + doorman gate (mirrors backcaster quest_leg exactly)
# ---------------------------------------------------------------------------


def _flip_gw(mode: str) -> bool:
    try:
        import httpx
        url = f"{FLIP_CONTROLLER_URL}/v0/nodes/gravitywell/flip"
        headers: dict[str, str] = {}
        if FLIP_CONTROLLER_TOKEN:
            headers["Authorization"] = f"Bearer {FLIP_CONTROLLER_TOKEN}"
        resp = httpx.post(
            url,
            json={"mode": mode, "source": "prior-art-scout"},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        log.info("[prior_art_scout] GW flip -> %s OK", mode)
        return True
    except Exception as exc:
        log.warning("[prior_art_scout] GW flip -> %s failed: %s", mode, exc)
        return False


def _gate_doorman_serving(timeout_s: int = FLIP_SERVE_TIMEOUT) -> bool:
    try:
        import httpx as _httpx
    except ImportError:
        log.warning("[prior_art_scout] httpx not available; cannot gate on doorman")
        return False

    deadline = time.monotonic() + timeout_s
    poll_interval = 5
    while time.monotonic() < deadline:
        try:
            resp = _httpx.get(f"{DOORMAN_URL}/status", timeout=5)
            if resp.status_code == 200:
                gw = resp.json().get("nodes", {}).get("gravitywell", {})
                if gw.get("serving") or gw.get("serving_mode") == "deferred":
                    return True
        except Exception as exc:
            log.debug("[prior_art_scout] doorman poll error: %s", exc)
        time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
    log.warning("[prior_art_scout] doorman serving-gate timed out after %ds", timeout_s)
    return False


# ---------------------------------------------------------------------------
# Provenance audit (mirrors quest_leg)
# ---------------------------------------------------------------------------


def _on_denylist(url: str) -> bool:
    url_lower = url.lower()
    return any(s in url_lower for s in _DENYLIST_SUBSTRINGS)


def _credibility_score(url: str) -> float:
    try:
        from agents_core.dowser import _credibility_score as _ds_cred
        return _ds_cred(url)
    except Exception:
        return 0.5


def _audit_citation(citation: dict) -> bool:
    url = citation.get("url", "")
    excerpt = citation.get("excerpt", "")
    if not url or not excerpt:
        return False
    if _on_denylist(url):
        return False
    return _credibility_score(url) >= _CREDIBILITY_FLOOR


# ---------------------------------------------------------------------------
# Phase helpers (mirrors quest_leg)
# ---------------------------------------------------------------------------


def _phase_swarm_read(
    requests: list[dict],
    read_operator: str,
    dowser: Any,
    flip_fn: Any,
    gate_fn: Any,
) -> tuple[list[dict] | None, bool]:
    if not flip_fn("swarm"):
        log.warning("[prior_art_scout] swarm flip failed")
        return None, False
    if not gate_fn(FLIP_SERVE_TIMEOUT):
        log.warning("[prior_art_scout] serving-gate timed out after swarm flip; failing toward big")
        flip_fn("big")
        return None, True
    try:
        result = dowser.read_batch(requests, read_operator=read_operator)
        return result.get("drafts", []), False
    except Exception as exc:
        log.warning("[prior_art_scout] read_batch failed: %s", exc)
        flip_fn("big")
        return None, False


def _phase_big_critique(
    drafts: list[dict],
    critic_operator: str,
    dowser: Any,
    flip_fn: Any,
    gate_fn: Any,
    is_local: bool,
) -> tuple[list[dict] | None, bool]:
    if is_local:
        if not flip_fn("big"):
            log.warning("[prior_art_scout] big flip failed before critique")
            return None, False
        if not gate_fn(FLIP_SERVE_TIMEOUT):
            log.warning("[prior_art_scout] serving-gate timed out after big flip")
            return None, True
    try:
        result = dowser.critique_batch(drafts, critic_operator=critic_operator)
        return result.get("verdicts", []), False
    except Exception as exc:
        log.warning("[prior_art_scout] critique_batch failed: %s", exc)
        return None, False


# ---------------------------------------------------------------------------
# Cursor management
# ---------------------------------------------------------------------------


def _load_cursor() -> str:
    if _CURSOR_PATH.exists():
        try:
            data = json.loads(_CURSOR_PATH.read_text())
            return data.get("cursor", "")
        except Exception:
            return ""
    return ""


def _save_cursor(key: str) -> None:
    _CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CURSOR_PATH.write_text(json.dumps({"cursor": key}))


def _select_priority2_batch(
    in_flight_keys: list[str],  # sorted
    cursor: str,
    batch_size: int,
) -> tuple[list[str], str]:
    """Return (batch_keys, new_cursor).

    Selects keys strictly greater than cursor via bisect; wraps to start when exhausted.
    new_cursor is the last key taken (or "" on wrap with no items).
    """
    if not in_flight_keys:
        return [], cursor

    pos = bisect.bisect_right(in_flight_keys, cursor)
    if pos >= len(in_flight_keys):
        # Wrap
        pos = 0

    batch = in_flight_keys[pos: pos + batch_size]
    new_cursor = batch[-1] if batch else ""
    return batch, new_cursor


# ---------------------------------------------------------------------------
# Lean classification
# ---------------------------------------------------------------------------


def _classify_lean(verdict_item: dict) -> str:
    """Extract lean from critic verdict if available; default pending."""
    lean = (verdict_item or {}).get("lean", "")
    if lean in ("adopt-pattern", "adopt-tool", "not-relevant"):
        return lean
    return "pending"


# ---------------------------------------------------------------------------
# Stub mode
# ---------------------------------------------------------------------------


_STUB_CANNED_FINDINGS = (
    "Tenet (github.com/JeiKeiLim/tenet): TypeScript DAG orchestration with "
    "heartbeat/crash recovery, typed steer-messages, spec-driven loop."
)
_STUB_CANNED_CITATIONS = [
    {
        "url": "https://github.com/JeiKeiLim/tenet",
        "excerpt": "TypeScript DAG orchestration harness with heartbeat recovery.",
        "credibility": 0.8,
    }
]


def _stub_run(
    items: list[dict],
    is_local: bool,
    today: str,
) -> tuple[list[ScoutItem], int, int]:
    """Return (scout_items, sourced_count, honest_null_count) with canned outputs."""
    scout_items: list[ScoutItem] = []
    sourced = 0
    honest_null = 0
    for item in items:
        key = item["key"]
        write_sidecar(key, {
            "scout_attempted": today,
            "verdict": "sourced",
            "findings": _STUB_CANNED_FINDINGS,
            "citations": _STUB_CANNED_CITATIONS,
            "lean": "adopt-pattern",
        })
        si = ScoutItem(
            key=key,
            namespace=item["namespace"],
            summary=item["summary"],
            status=item["status"],
            findings=_STUB_CANNED_FINDINGS,
            citations=_STUB_CANNED_CITATIONS,
            outcome="sources-found",
            lean="adopt-pattern",
        )
        scout_items.append(si)
        sourced += 1
    return scout_items, sourced, honest_null


# ---------------------------------------------------------------------------
# Counter-query helper (mirrors quest_leg._try_counter_query)
# ---------------------------------------------------------------------------


def _try_counter_query(
    key: str,
    today: str,
    counter_req: dict,
    read_operator: str,
    critic_operator: str,
    dowser: Any,
    is_local: bool,
    flip_fn: Any,
    gate_fn: Any,
    orig_prov: dict,
) -> tuple[bool, list[dict], str]:
    """Issue adversarial counter-query.

    Returns (sourced, good_citations, lean).
    """
    try:
        if is_local:
            counter_drafts, _ = _phase_swarm_read(
                [counter_req], read_operator, dowser, flip_fn, gate_fn
            )
        else:
            r = dowser.read_batch([counter_req], read_operator=read_operator)
            counter_drafts = r.get("drafts", [])

        if not counter_drafts:
            _write_known_hopeless(key, today, orig_prov, "counter-query read failed")
            return False, [], ""

        cdraft = counter_drafts[0]
        counter_vs, _ = _phase_big_critique(
            [cdraft], critic_operator, dowser, flip_fn, gate_fn, is_local
        )
        if not counter_vs:
            _write_known_hopeless(key, today, orig_prov, "counter-query critique failed")
            return False, [], ""

        if counter_vs[0].get("status") == "pass":
            good = [c for c in cdraft.get("citations", []) if _audit_citation(c)]
            if good:
                lean = _classify_lean(counter_vs[0])
                write_sidecar(key, {
                    "scout_attempted": today,
                    "verdict": "sourced",
                    "findings": cdraft.get("findings", ""),
                    "citations": good,
                    "provenance": cdraft.get("provenance", {}),
                    "lean": lean,
                    "note": "sourced via adversarial counter-query",
                })
                return True, good, lean

        diagnosis = (counter_vs[0].get("diagnosis", "") if counter_vs else "")
        _write_known_hopeless(key, today, orig_prov, diagnosis)
        return False, [], ""

    except Exception as exc:
        log.warning("[prior_art_scout] counter-query failed for %s: %s", key, exc)
        _write_known_hopeless(key, today, orig_prov, str(exc))
        return False, [], ""


def _write_known_hopeless(key: str, today: str, prov: dict, reason: str) -> None:
    write_sidecar(key, {
        "scout_attempted": today,
        "verdict": "insufficient-sources",
        "search_strings": prov.get("search_strings", []),
        "hits_count": prov.get("hits_count", 0),
        "critic_reason": reason,
        "note": "no credible sources after adversarial counter-query",
    })


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def scout_run(
    *,
    escalate: bool = False,
    dry_run: bool = False,
    wall_budget: int | None = None,
    _dowser: Any = None,
    _flip_fn: Any = None,
    _gate_fn: Any = None,
) -> ScoutRun:
    """Execute one prior-art scout run.

    Parameters
    ----------
    escalate    : use read_operator=sonnet (paid, morning targeted path)
    dry_run     : load snapshot and select items but skip all LLM calls
    wall_budget : cap total items processed (smoke/testing)
    _dowser     : injectable Dowser module (tests)
    _flip_fn    : injectable flip function (tests)
    _gate_fn    : injectable gate function (tests)
    """
    from agents_core.roadmap import materialize_committed_plan

    is_stub = os.environ.get("PRIOR_ART_SCOUT_STUB") == "1"
    today = datetime.now(timezone.utc).date().isoformat()

    if _dowser is None and not is_stub and not dry_run:
        import agents_core.dowser as _dowser
    if _flip_fn is None:
        _flip_fn = _flip_gw
    if _gate_fn is None:
        _gate_fn = _gate_doorman_serving

    # -----------------------------------------------------------------------
    # 1. Load snapshot
    # -----------------------------------------------------------------------
    artifact = None
    try:
        artifact = materialize_committed_plan()
    except Exception as exc:
        log.error("[prior_art_scout] materialize_committed_plan failed: %s", exc)

    if artifact is None:
        log.error("[prior_art_scout] snapshot unavailable; aborting run")
        return ScoutRun(
            run_date=today,
            priority1_count=0,
            priority2_batch="0/0 (snapshot unavailable)",
            skipped_hopeless=0,
            items_targeted=0,
            items_sourced=0,
            items_honest_null=0,
            caution="high",
            model_policy="local-only",
        )

    all_items: list[dict] = artifact.get("items", [])

    # -----------------------------------------------------------------------
    # 1b. Saturation check
    # -----------------------------------------------------------------------
    swept_namespaces = {"architecture", "strategy", "decision"}
    ns_counts: dict[str, int] = {}
    for item in all_items:
        ns = item["namespace"]
        if ns in swept_namespaces:
            ns_counts[ns] = ns_counts.get(ns, 0) + 1

    saturated_namespaces = [ns for ns, cnt in ns_counts.items() if cnt >= _SNAPSHOT_CAP]

    # -----------------------------------------------------------------------
    # 2. Select priority-1 items
    # -----------------------------------------------------------------------
    skipped_hopeless = 0
    p1_items: list[dict] = []
    for item in all_items:
        if item["namespace"] not in {"architecture", "strategy"}:
            continue
        if item["status"] != "in-flight":
            continue
        if is_known_hopeless(item["key"]):
            skipped_hopeless += 1
            continue
        p1_items.append(item)

    # -----------------------------------------------------------------------
    # 3. Select priority-2 items (decision/* rotating cursor)
    # -----------------------------------------------------------------------
    decision_items = [
        item for item in all_items
        if item["namespace"] == "decision" and item["status"] == "in-flight"
    ]
    decision_items.sort(key=lambda i: i["key"])
    decision_in_flight_keys = [i["key"] for i in decision_items]

    cursor = _load_cursor()
    p2_batch_keys, new_cursor = _select_priority2_batch(
        decision_in_flight_keys, cursor, _PRIORITY2_BATCH_SIZE
    )

    p2_items: list[dict] = []
    key_to_decision = {i["key"]: i for i in decision_items}
    for k in p2_batch_keys:
        item = key_to_decision[k]
        if is_known_hopeless(k):
            skipped_hopeless += 1
            continue
        p2_items.append(item)

    all_selected = p1_items + p2_items

    # Apply wall_budget cap
    if wall_budget is not None and len(all_selected) > wall_budget:
        all_selected = all_selected[:wall_budget]
        log.info("[prior_art_scout] wall_budget=%d applied; trimmed to %d items", wall_budget, len(all_selected))

    p2_batch_str = (
        f"{len(p2_items)}/{len(decision_in_flight_keys)}"
        f" (cursor @{new_cursor or 'start'})"
    )

    # -----------------------------------------------------------------------
    # Operator / policy
    # -----------------------------------------------------------------------
    if escalate:
        read_operator = "sonnet"
        model_policy = "allow-escalation"
    else:
        read_operator = "quest"
        model_policy = "local-only"
    critic_operator = "gravitywell"
    is_local = model_policy == "local-only"

    # -----------------------------------------------------------------------
    # Dry-run or stub short-circuit
    # -----------------------------------------------------------------------
    if dry_run:
        _save_cursor(new_cursor)
        return ScoutRun(
            run_date=today,
            priority1_count=len(p1_items),
            priority2_batch=p2_batch_str,
            skipped_hopeless=skipped_hopeless,
            items_targeted=len(all_selected),
            items_sourced=0,
            items_honest_null=0,
            caution="high",
            model_policy=model_policy,
            saturated_namespaces=saturated_namespaces,
            wall_budget_applied=wall_budget,
        )

    if is_stub:
        stub_items, sourced_count, honest_null_count = _stub_run(all_selected, is_local, today)
        _save_cursor(new_cursor)
        caution = _compute_caution(sourced_count, honest_null_count, len(all_selected))
        run = ScoutRun(
            run_date=today,
            priority1_count=len(p1_items),
            priority2_batch=p2_batch_str,
            skipped_hopeless=skipped_hopeless,
            items_targeted=len(all_selected),
            items_sourced=sourced_count,
            items_honest_null=honest_null_count,
            caution=caution,
            model_policy=model_policy,
            saturated_namespaces=saturated_namespaces,
            wall_budget_applied=wall_budget,
        )
        _write_brief_and_yaml(stub_items, run, today)
        return run

    if not all_selected:
        _save_cursor(new_cursor)
        return ScoutRun(
            run_date=today,
            priority1_count=len(p1_items),
            priority2_batch=p2_batch_str,
            skipped_hopeless=skipped_hopeless,
            items_targeted=0,
            items_sourced=0,
            items_honest_null=0,
            caution="low",
            model_policy=model_policy,
            saturated_namespaces=saturated_namespaces,
        )

    # -----------------------------------------------------------------------
    # 4. Build Dowser requests
    # -----------------------------------------------------------------------
    key_order: list[str] = []
    clean_requests: list[dict] = []
    for item in all_selected:
        clean_requests.append(item_to_request(item))
        key_order.append(item["key"])

    item_by_key = {item["key"]: item for item in all_selected}

    # -----------------------------------------------------------------------
    # Any-exit big-flip guard (mirrors quest_leg)
    # -----------------------------------------------------------------------
    import signal as _sig
    _prev_sigterm = _sig.getsignal(_sig.SIGTERM)

    def _sigterm_handler(signum, frame):
        raise SystemExit(128 + signum)

    if is_local:
        try:
            _sig.signal(_sig.SIGTERM, _sigterm_handler)
        except (OSError, ValueError):
            pass

    try:
        # -------------------------------------------------------------------
        # Phase 1: Read batch (swarm flip)
        # -------------------------------------------------------------------
        read_drafts: list[dict] | None = None
        read_timed_out = False

        if is_local:
            read_drafts, read_timed_out = _phase_swarm_read(
                clean_requests, read_operator, _dowser, _flip_fn, _gate_fn
            )
        else:
            try:
                result = _dowser.read_batch(clean_requests, read_operator=read_operator)
                read_drafts = result.get("drafts", [])
            except Exception as exc:
                log.warning("[prior_art_scout] read_batch (escalation) failed: %s", exc)

        if not read_drafts:
            verdict = "infra-unavailable"
            note = "read phase timed out" if read_timed_out else "read phase failed"
            for k in key_order:
                write_sidecar(k, {"scout_attempted": today, "verdict": verdict, "note": note})
            if is_local:
                _flip_fn("big")
            _save_cursor(new_cursor)
            n = len(key_order)
            run = ScoutRun(
                run_date=today,
                priority1_count=len(p1_items),
                priority2_batch=p2_batch_str,
                skipped_hopeless=skipped_hopeless,
                items_targeted=n,
                items_sourced=0,
                items_honest_null=n,
                caution="high",
                model_policy=model_policy,
                saturated_namespaces=saturated_namespaces,
                wall_budget_applied=wall_budget,
            )
            _write_brief_and_yaml([], run, today)
            return run

        # -------------------------------------------------------------------
        # Phase 2: Critique batch (big flip)
        # -------------------------------------------------------------------
        critique_verdicts: list[dict] | None = None
        critique_timed_out = False

        critique_verdicts, critique_timed_out = _phase_big_critique(
            read_drafts, critic_operator, _dowser, _flip_fn, _gate_fn, is_local
        )

        # -------------------------------------------------------------------
        # Phase 3: Retry for subpar (diagnosis-driven)
        # -------------------------------------------------------------------
        retries_used = 0
        if critique_verdicts:
            subpar_idx = [
                i for i, v in enumerate(critique_verdicts)
                if v.get("status") != "pass"
            ]
            if subpar_idx:
                retry_reqs = []
                for i in subpar_idx:
                    diagnosis = critique_verdicts[i].get("diagnosis", "") or "subpar"
                    retry_reqs.append({**clean_requests[i], "prior_diagnosis": diagnosis})

                retry_drafts: list[dict] | None = None
                if is_local:
                    retry_drafts, _ = _phase_swarm_read(
                        retry_reqs, read_operator, _dowser, _flip_fn, _gate_fn
                    )
                else:
                    try:
                        r = _dowser.read_batch(retry_reqs, read_operator=read_operator)
                        retry_drafts = r.get("drafts", [])
                    except Exception as exc:
                        log.warning("[prior_art_scout] retry read_batch failed: %s", exc)

                if retry_drafts:
                    retries_used = 1
                    retry_verdicts, _ = _phase_big_critique(
                        retry_drafts, critic_operator, _dowser, _flip_fn, _gate_fn, is_local
                    )
                    if retry_verdicts:
                        for j, i in enumerate(subpar_idx):
                            if j < len(retry_verdicts) and retry_verdicts[j].get("status") == "pass":
                                critique_verdicts[i] = retry_verdicts[j]
                                if j < len(retry_drafts):
                                    read_drafts[i] = retry_drafts[j]

        # Ensure GW on big before write-back
        if is_local:
            _flip_fn("big")

        # -------------------------------------------------------------------
        # Phase 4: Write-back per item
        # -------------------------------------------------------------------
        scout_items: list[ScoutItem] = []
        items_sourced = 0
        items_honest_null = 0

        for i, key in enumerate(key_order):
            snap_item = item_by_key[key]
            draft = read_drafts[i] if i < len(read_drafts) else {}
            verdict_item = (
                (critique_verdicts or [])[i]
                if critique_verdicts and i < len(critique_verdicts)
                else {}
            )

            si = ScoutItem(
                key=key,
                namespace=snap_item["namespace"],
                summary=snap_item["summary"],
                status=snap_item["status"],
            )

            if critique_timed_out or not critique_verdicts:
                write_sidecar(key, {
                    "scout_attempted": today,
                    "verdict": "infra-unavailable",
                    "note": "critique timed out" if critique_timed_out else "critique failed",
                })
                si.outcome = "infra-unavailable"
                items_honest_null += 1
                scout_items.append(si)
                continue

            status = verdict_item.get("status", "subpar")
            outcome = draft.get("outcome", "no-credible-sources")
            prov = draft.get("provenance", {})

            if status == "pass":
                raw_cits = draft.get("citations", [])
                good_cits = [c for c in raw_cits if _audit_citation(c)]

                if not good_cits:
                    write_sidecar(key, {
                        "scout_attempted": today,
                        "verdict": "provenance-audit-no-survivors",
                        "note": "pass verdict but no citations survived provenance audit",
                    })
                    si.outcome = "no-credible-sources"
                    items_honest_null += 1
                    scout_items.append(si)
                    continue

                lean = _classify_lean(verdict_item)
                findings = draft.get("findings", "")
                write_sidecar(key, {
                    "scout_attempted": today,
                    "verdict": "sourced",
                    "findings": findings,
                    "citations": good_cits,
                    "lean": lean,
                    "provenance": prov,
                    "critic_verdict": verdict_item.get("verdict", {}),
                })
                si.outcome = "sources-found"
                si.findings = findings
                si.citations = good_cits
                si.lean = lean
                items_sourced += 1

            else:
                if outcome == "no-credible-sources":
                    counter_req = {
                        **clean_requests[i],
                        "prior_diagnosis": (
                            "Previous pass found no credible sources. "
                            "Reframe with an opposing angle or alternate vocabulary. "
                            f"Prior critic reason: {verdict_item.get('diagnosis', '')}"
                        ),
                    }
                    sourced, good_cits, lean = _try_counter_query(
                        key, today, counter_req,
                        read_operator, critic_operator, _dowser, is_local,
                        _flip_fn, _gate_fn, prov,
                    )
                    if sourced:
                        sidecar = load_sidecar(key) or {}
                        si.outcome = "sources-found"
                        si.findings = sidecar.get("findings", "")
                        si.citations = good_cits
                        si.lean = lean
                        items_sourced += 1
                    else:
                        si.outcome = "no-credible-sources"
                        items_honest_null += 1

                elif outcome == "high-friction":
                    write_sidecar(key, {
                        "scout_attempted": today,
                        "verdict": "high-friction-retry-candidate",
                        "search_strings": prov.get("search_strings", []),
                        "hits_count": prov.get("hits_count", 0),
                        "critic_reason": verdict_item.get("diagnosis", ""),
                    })
                    si.outcome = "high-friction"
                    items_honest_null += 1

                else:
                    write_sidecar(key, {
                        "scout_attempted": today,
                        "verdict": "infra-unavailable",
                        "note": f"outcome={outcome}; transient",
                    })
                    si.outcome = "infra-unavailable"
                    items_honest_null += 1

            scout_items.append(si)

        # Final big-flip backstop
        if is_local:
            _flip_fn("big")

        # -------------------------------------------------------------------
        # Phase 5: Compose brief
        # -------------------------------------------------------------------
        _save_cursor(new_cursor)
        caution = _compute_caution(items_sourced, items_honest_null, len(all_selected))
        run = ScoutRun(
            run_date=today,
            priority1_count=len(p1_items),
            priority2_batch=p2_batch_str,
            skipped_hopeless=skipped_hopeless,
            items_targeted=len(all_selected),
            items_sourced=items_sourced,
            items_honest_null=items_honest_null,
            caution=caution,
            model_policy=model_policy,
            saturated_namespaces=saturated_namespaces,
            wall_budget_applied=wall_budget,
        )
        _write_brief_and_yaml(scout_items, run, today)
        return run

    finally:
        if is_local:
            _flip_fn("big")
        try:
            _sig.signal(_sig.SIGTERM, _prev_sigterm)
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_caution(sourced: int, honest_null: int, total: int) -> str:
    if total == 0:
        return "high"
    null_ratio = honest_null / total
    if null_ratio < 0.20:
        return "low"
    elif null_ratio <= 0.50:
        return "medium"
    return "high"


def _write_brief_and_yaml(scout_items: list[ScoutItem], run: ScoutRun, today: str) -> None:
    from .compose import write_brief
    run_dir = Path(f"/srv/lapis/prior-art-scout/runs/{today}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_brief(run_dir, scout_items, run)
