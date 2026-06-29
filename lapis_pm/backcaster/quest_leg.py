"""Backcaster QUEST leg — post-hoc web sourcing for unsourced gaps.

Takes a completed Backcaster run's unsourced gaps, sources them via Dowser
(read_batch + critique_batch), writes web citations back, and recomputes
epistemic_caution.

Flip sequencing (local-only path):
  swarm flip -> doorman serving-gate -> read_batch
  -> big flip -> doorman serving-gate -> critique_batch

The leg owns GW flips directly (per decision/h5-elevator-queue-design).
Dowser is topology-agnostic; we sequence around its two phases.

Anti-deadlock: every serving-gate has a hard timeout (FLIP_SERVE_TIMEOUT,
default 240s). On timeout: abort phase, mark gaps infra-unavailable (transient),
fail toward big (never strand GW mid-flip in swarm).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .compose import compute_epistemic_caution
from .schema import BackcasterCitation, Gap

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FLIP_CONTROLLER_URL = os.environ.get(
    "FLIP_CONTROLLER_URL", "http://203.0.113.10:8408"
)
FLIP_CONTROLLER_TOKEN = os.environ.get("FLIP_CONTROLLER_TOKEN", "")
DOORMAN_URL = os.environ.get("DOORMAN_URL", "http://127.0.0.1:8407")
FLIP_SERVE_TIMEOUT = int(os.environ.get("FLIP_SERVE_TIMEOUT", "240"))

# Credibility floor for provenance audit (mirrors Dowser's threshold)
_CREDIBILITY_FLOOR = 0.3

# Domain denylist (mirrors Dowser's _JUNK_PATTERNS; kept explicit for audit clarity)
_DENYLIST_SUBSTRINGS = [
    "reddit.com", "quora.com", "pinterest.com", "medium.com/tag",
    "fiverr.com", "upwork.com", "guru.com", "freelancer.com",
    "blogspot", "wordpress.com/20",  # year-path WordPress spam
    "seo", "affiliate",
]


# ---------------------------------------------------------------------------
# GW flip + doorman serving-gate
# ---------------------------------------------------------------------------

def _flip_gw(mode: str) -> bool:
    """POST to flip-controller; return True on success."""
    try:
        import httpx
        url = f"{FLIP_CONTROLLER_URL}/v0/nodes/gravitywell/flip"
        headers: dict[str, str] = {}
        if FLIP_CONTROLLER_TOKEN:
            headers["Authorization"] = f"Bearer {FLIP_CONTROLLER_TOKEN}"
        resp = httpx.post(
            url,
            json={"mode": mode, "source": "backcaster-quest-leg"},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        log.info("[quest_leg] GW flip -> %s OK", mode)
        return True
    except Exception as exc:
        log.warning("[quest_leg] GW flip -> %s failed: %s", mode, exc)
        return False


def _gate_doorman_serving(timeout_s: int = FLIP_SERVE_TIMEOUT) -> bool:
    """Poll doorman /status until nodes.gravitywell.serving == True.

    Returns False only on hard timeout (never raises).
    """
    try:
        import httpx as _httpx
    except ImportError:
        log.warning("[quest_leg] httpx not available; cannot gate on doorman")
        return False

    deadline = time.monotonic() + timeout_s
    poll_interval = 5
    while time.monotonic() < deadline:
        try:
            resp = _httpx.get(f"{DOORMAN_URL}/status", timeout=5)
            if resp.status_code == 200:
                gw = resp.json().get("nodes", {}).get("gravitywell", {})
                if gw.get("serving"):
                    return True
        except Exception as exc:
            log.debug("[quest_leg] doorman poll error: %s", exc)
        time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
    log.warning("[quest_leg] doorman serving-gate timed out after %ds", timeout_s)
    return False


# ---------------------------------------------------------------------------
# Provenance audit
# ---------------------------------------------------------------------------

def _on_denylist(url: str) -> bool:
    url_lower = url.lower()
    return any(s in url_lower for s in _DENYLIST_SUBSTRINGS)


def _credibility_score(url: str) -> float:
    """Re-use Dowser's domain prior without creating a cross-dependency loop."""
    try:
        from agents_core.dowser import _credibility_score as _ds_cred
        return _ds_cred(url)
    except Exception:
        # Fallback: pessimistic score
        return 0.5


def _audit_citation(citation: dict) -> bool:
    """Return True if citation passes the provenance audit.

    Checks: non-empty URL + excerpt, not on denylist, credibility above floor.
    """
    url = citation.get("url", "")
    excerpt = citation.get("excerpt", "")
    if not url or not excerpt:
        return False
    if _on_denylist(url):
        return False
    return _credibility_score(url) >= _CREDIBILITY_FLOOR


# ---------------------------------------------------------------------------
# Run-dir IO helpers
# ---------------------------------------------------------------------------

def _sorted_yaml(obj: Any) -> str:
    return yaml.dump(obj, default_flow_style=False, sort_keys=True, allow_unicode=True)


def _load_gaps(run_dir: Path) -> list[Gap]:
    path = run_dir / "gaps.yaml"
    if not path.exists():
        raise FileNotFoundError(f"gaps.yaml not found in {run_dir}")
    raw = yaml.safe_load(path.read_text()) or {}
    return [Gap(**g) for g in raw.get("gaps", [])]


def _write_gaps(run_dir: Path, gaps: list[Gap]) -> None:
    data = {"gaps": sorted([g.to_dict() for g in gaps], key=lambda x: x["precondition_id"])}
    (run_dir / "gaps.yaml").write_text(_sorted_yaml(data))


def _load_run(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "run.yaml").read_text()) or {}


def _write_run(run_dir: Path, data: dict) -> None:
    (run_dir / "run.yaml").write_text(_sorted_yaml(data))


def _load_precondition_statements(run_dir: Path) -> dict[str, str]:
    """pid -> statement from decomposition.yaml."""
    path = run_dir / "decomposition.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {p["id"]: p.get("statement", "") for p in raw.get("preconditions", [])}


# ---------------------------------------------------------------------------
# Quest sidecar (quest/<pid>.yaml) — marker + provenance store
# ---------------------------------------------------------------------------

def _quest_dir(run_dir: Path) -> Path:
    return run_dir / "quest"


def _load_sidecar(run_dir: Path, pid: str) -> dict | None:
    path = _quest_dir(run_dir) / f"{pid}.yaml"
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return None


def _write_sidecar(run_dir: Path, pid: str, data: dict) -> None:
    qdir = _quest_dir(run_dir)
    qdir.mkdir(exist_ok=True)
    (qdir / f"{pid}.yaml").write_text(_sorted_yaml(data))


def _is_known_hopeless(run_dir: Path, pid: str) -> bool:
    sidecar = _load_sidecar(run_dir, pid)
    return bool(sidecar and sidecar.get("verdict") == "insufficient-sources")


# ---------------------------------------------------------------------------
# Re-derive components for a sourced gap
# ---------------------------------------------------------------------------

def _rederive_gap(gap: Gap, run_dir: Path) -> None:
    from .derive import derive_components

    comps_path = run_dir / "components.yaml"
    if not comps_path.exists():
        return
    raw = yaml.safe_load(comps_path.read_text()) or {}
    surviving = [
        c for c in raw.get("components", [])
        if c.get("gap_precondition_id") != gap.precondition_id
    ]
    new_comps, _, _ = derive_components([gap])
    all_comps = surviving + [c.to_dict() for c in new_comps]
    comps_path.write_text(_sorted_yaml({"components": sorted(all_comps, key=lambda x: x["id"])}))


# ---------------------------------------------------------------------------
# Core flip+gate phase helpers (return False on timeout/error)
# ---------------------------------------------------------------------------

def _phase_swarm_read(
    clean_requests: list[dict],
    read_operator: str,
    dowser: Any,
    flip_fn: Any,
    gate_fn: Any,
) -> tuple[list[dict] | None, bool]:
    """Flip to swarm, gate, read. Returns (drafts, timed_out).

    On timeout: flips back to big before returning.
    """
    if not flip_fn("swarm"):
        log.warning("[quest_leg] swarm flip failed")
        return None, False
    if not gate_fn(FLIP_SERVE_TIMEOUT):
        log.warning("[quest_leg] serving-gate timed out after swarm flip; failing toward big")
        flip_fn("big")
        return None, True
    try:
        result = dowser.read_batch(clean_requests, read_operator=read_operator)
        return result.get("drafts", []), False
    except Exception as exc:
        log.warning("[quest_leg] read_batch failed: %s", exc)
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
    """Flip to big (if local path), gate, critique. Returns (verdicts, timed_out)."""
    if is_local:
        if not flip_fn("big"):
            log.warning("[quest_leg] big flip failed before critique")
            return None, False
        if not gate_fn(FLIP_SERVE_TIMEOUT):
            log.warning("[quest_leg] serving-gate timed out after big flip")
            return None, True
    try:
        result = dowser.critique_batch(drafts, critic_operator=critic_operator)
        return result.get("verdicts", []), False
    except Exception as exc:
        log.warning("[quest_leg] critique_batch failed: %s", exc)
        return None, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def quest_source_run(
    run_id: str,
    *,
    gap_ids: list[str] | None = None,
    all_unsourced: bool = False,
    escalate: bool = False,
    _dowser: Any = None,
    _flip_fn: Any = None,
    _gate_fn: Any = None,
    _wall_budget: int | None = None,
) -> dict:
    """Source unsourced gaps from a completed Backcaster run via Dowser.

    Parameters
    ----------
    run_id       : the run directory name under /srv/lapis/backcaster/runs/
    gap_ids      : explicit list of precondition IDs to target (overrides all_unsourced)
    all_unsourced: source all unsourced gaps
    escalate     : use read_operator=sonnet (paid); no swarm flip; morning path only
    _dowser      : injectable Dowser module (tests); defaults to agents_core.dowser
    _flip_fn     : injectable flip function (tests); signature: (mode: str) -> bool
    _gate_fn     : injectable gate function (tests); signature: (timeout_s: int) -> bool
    _wall_budget : max gaps to process this run (None = unlimited)

    Returns a human-readable summary dict.
    """
    from agents_core.room_paths import room_path

    runs_root = Path(room_path("backcaster.runs"))
    run_dir = runs_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    if _dowser is None:
        import agents_core.dowser as _dowser
    if _flip_fn is None:
        _flip_fn = _flip_gw
    if _gate_fn is None:
        _gate_fn = _gate_doorman_serving

    # -----------------------------------------------------------------------
    # Select target gaps
    # -----------------------------------------------------------------------
    all_gaps = _load_gaps(run_dir)
    run_data = _load_run(run_dir)
    caution_before = run_data.get("epistemic_caution", "high")
    precondition_statements = _load_precondition_statements(run_dir)

    if gap_ids:
        target_gaps = [g for g in all_gaps if g.precondition_id in gap_ids and g.unsourced]
    elif all_unsourced:
        target_gaps = [g for g in all_gaps if g.unsourced]
    else:
        raise ValueError("must specify gap_ids or all_unsourced=True")

    # Skip known-hopeless (insufficient-sources verdict from a prior run)
    target_gaps = [
        g for g in target_gaps
        if not _is_known_hopeless(run_dir, g.precondition_id)
    ]
    if _wall_budget is not None:
        target_gaps = target_gaps[:_wall_budget]

    if not target_gaps:
        return {
            "gaps_targeted": 0,
            "gaps_sourced": 0,
            "gaps_honest_null": 0,
            "citations_added": 0,
            "caution_before": caution_before,
            "caution_after": caution_before,
            "retries_used": 0,
            "note": "no targetable gaps",
        }

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
    # Any-exit flip guard (criterion 6c): SIGTERM handler + try/finally ensure
    # GW is never stranded on swarm on any exit path (normal, exception, SIGTERM).
    # -----------------------------------------------------------------------
    import signal as _sig
    _prev_sigterm = _sig.getsignal(_sig.SIGTERM)

    def _sigterm_handler(signum, frame):
        raise SystemExit(128 + signum)

    if is_local:
        try:
            _sig.signal(_sig.SIGTERM, _sigterm_handler)
        except (OSError, ValueError):
            pass  # not in main thread; cannot install signal handler

    try:
        # -----------------------------------------------------------------------
        # Build Dowser requests (positionally aligned to target_gaps)
        # -----------------------------------------------------------------------
        pid_order: list[str] = []  # index -> pid; matches request/draft/verdict order
        clean_requests: list[dict] = []

        for gap in target_gaps:
            pid = gap.precondition_id
            intent = precondition_statements.get(pid) or gap.what_missing[:400]
            context = (
                f"what_exists: {gap.what_exists}\n"
                f"what_missing: {gap.what_missing}\n"
                f"what_miswired: {gap.what_miswired}"
            )
            sub_intents = [
                f"What exists: {gap.what_exists[:200]}",
                f"What is missing: {gap.what_missing[:200]}",
                f"What is miswired: {gap.what_miswired[:200]}",
            ]
            clean_requests.append({"intent": intent, "context": context, "sub_intents": sub_intents})
            pid_order.append(pid)

        # -----------------------------------------------------------------------
        # Phase 1: Read batch
        # -----------------------------------------------------------------------
        today = datetime.now(timezone.utc).date().isoformat()
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
                log.warning("[quest_leg] read_batch (escalation) failed: %s", exc)

        if not read_drafts:
            verdict = "infra-unavailable"
            note = "read phase timed out" if read_timed_out else "read phase failed"
            for gap in target_gaps:
                _write_sidecar(run_dir, gap.precondition_id, {
                    "quest_attempted": today, "verdict": verdict, "note": note,
                })
            # Fail toward big
            if is_local:
                _flip_fn("big")
            return _empty_summary(caution_before, len(target_gaps), note)

        # -----------------------------------------------------------------------
        # Phase 2: Critique batch
        # -----------------------------------------------------------------------
        critique_verdicts: list[dict] | None = None
        critique_timed_out = False

        critique_verdicts, critique_timed_out = _phase_big_critique(
            read_drafts, critic_operator, _dowser, _flip_fn, _gate_fn, is_local
        )

        # -----------------------------------------------------------------------
        # Phase 3: One retry for subpar verdicts (diagnosis-driven)
        # -----------------------------------------------------------------------
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
                    retry_reqs.append({
                        **clean_requests[i],
                        "prior_diagnosis": diagnosis,
                    })

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
                        log.warning("[quest_leg] retry read_batch failed: %s", exc)

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

        # Mid-loop backstop: leave GW on big before write-back loop (which may
        # re-flip to swarm via counter-queries for no-credible-sources gaps).
        if is_local:
            _flip_fn("big")

        # -----------------------------------------------------------------------
        # Phase 4: Write-back per gap
        # -----------------------------------------------------------------------
        pid_to_gap = {g.precondition_id: g for g in target_gaps}
        gaps_sourced = 0
        gaps_honest_null = 0
        citations_added = 0

        for i, pid in enumerate(pid_order):
            gap = pid_to_gap[pid]
            draft = read_drafts[i] if i < len(read_drafts) else {}
            verdict_item = (critique_verdicts or [])[i] if critique_verdicts and i < len(critique_verdicts) else {}

            if critique_timed_out or not critique_verdicts:
                _write_sidecar(run_dir, pid, {
                    "quest_attempted": today,
                    "verdict": "infra-unavailable",
                    "note": "critique phase timed out" if critique_timed_out else "critique phase failed",
                })
                gaps_honest_null += 1
                continue

            status = verdict_item.get("status", "subpar")
            outcome = draft.get("outcome", "no-credible-sources")
            prov = draft.get("provenance", {})

            if status == "pass":
                # Provenance audit first
                raw_cits = draft.get("citations", [])
                good_cits = [c for c in raw_cits if _audit_citation(c)]

                if not good_cits:
                    # Transient marker: all citations failed provenance audit, but Dowser
                    # may return different (non-denylist) URLs on a future run — do NOT
                    # use insufficient-sources (known-hopeless) here.
                    _write_sidecar(run_dir, pid, {
                        "quest_attempted": today,
                        "verdict": "provenance-audit-no-survivors",
                        "note": "pass verdict but no citations survived provenance audit",
                        "search_strings": prov.get("search_strings", []),
                        "hits_count": prov.get("hits_count", 0),
                    })
                    gaps_honest_null += 1
                    continue

                # Attach citations FIRST, then clear unsourced (validator order)
                for c in good_cits:
                    gap.citations.append(BackcasterCitation(type="web", ref=c["url"]))
                    citations_added += 1
                gap.unsourced = False

                _write_sidecar(run_dir, pid, {
                    "quest_attempted": today,
                    "verdict": "sourced",
                    "findings": draft.get("findings", ""),
                    "citations": good_cits,
                    "provenance": prov,
                    "critic_verdict": verdict_item.get("verdict", {}),
                })

                try:
                    _rederive_gap(gap, run_dir)
                except Exception as exc:
                    log.warning("[quest_leg] re-derive failed for %s: %s", pid, exc)

                gaps_sourced += 1

            else:
                # Honest-null, routed by Dowser's typed outcome
                if outcome == "no-credible-sources":
                    # Adversarial counter-query: treat null as hypothesis to disprove
                    counter_req = {
                        **clean_requests[i],
                        "prior_diagnosis": (
                            "Previous pass found no credible sources. "
                            "Reframe with an opposing angle or alternate vocabulary. "
                            f"Prior critic reason: {verdict_item.get('diagnosis', '')}"
                        ),
                    }
                    prior_web = sum(1 for c in gap.citations if c.type == "web")
                    counter_sourced = _try_counter_query(
                        gap, pid, run_dir, today, counter_req,
                        read_operator, critic_operator, _dowser, is_local,
                        _flip_fn, _gate_fn, prov,
                    )
                    if counter_sourced:
                        citations_added += sum(1 for c in gap.citations if c.type == "web") - prior_web
                        gaps_sourced += 1
                    else:
                        gaps_honest_null += 1

                elif outcome == "high-friction":
                    _write_sidecar(run_dir, pid, {
                        "quest_attempted": today,
                        "verdict": "high-friction-retry-candidate",
                        "search_strings": prov.get("search_strings", []),
                        "hits_count": prov.get("hits_count", 0),
                        "critic_reason": verdict_item.get("diagnosis", ""),
                        "friction_ratio": prov.get("friction_ratio", 0.0),
                    })
                    gaps_honest_null += 1

                else:
                    # infra-unavailable or unknown transient
                    _write_sidecar(run_dir, pid, {
                        "quest_attempted": today,
                        "verdict": "infra-unavailable",
                        "note": f"outcome={outcome}; transient",
                    })
                    gaps_honest_null += 1

        # Final backstop: ensure GW on big after write-back loop.
        # Counter-queries inside the loop can flip to swarm; a flip POST returning
        # non-200 must not leave GW stranded. The try/finally catches exception
        # + SIGTERM, but this explicit call closes the mid-loop POST-failure edge.
        if is_local:
            _flip_fn("big")

        # -----------------------------------------------------------------------
        # Phase 5: Recompute caution + write back
        # -----------------------------------------------------------------------
        updated_map = {g.precondition_id: g for g in target_gaps}
        final_gaps = [updated_map.get(g.precondition_id, g) for g in all_gaps]
        caution_after = compute_epistemic_caution(final_gaps)
        _write_gaps(run_dir, final_gaps)
        run_data["epistemic_caution"] = caution_after
        _write_run(run_dir, run_data)

        # -----------------------------------------------------------------------
        # Build summary
        # -----------------------------------------------------------------------
        null_outcomes: dict[str, int] = {
            "insufficient-sources": 0,
            "high-friction": 0,
            "infra-unavailable": 0,
        }
        escalate_candidates: list[str] = []
        for gap in target_gaps:
            sidecar = _load_sidecar(run_dir, gap.precondition_id) or {}
            v = sidecar.get("verdict", "")
            if v == "insufficient-sources":
                null_outcomes["insufficient-sources"] += 1
            elif v == "high-friction-retry-candidate":
                null_outcomes["high-friction"] += 1
                escalate_candidates.append(gap.precondition_id)
            elif v == "infra-unavailable":
                null_outcomes["infra-unavailable"] += 1

        summary: dict[str, Any] = {
            "gaps_targeted": len(target_gaps),
            "gaps_sourced": gaps_sourced,
            "gaps_honest_null": gaps_honest_null,
            "citations_added": citations_added,
            "caution_before": caution_before,
            "caution_after": caution_after,
            "retries_used": retries_used,
            "null_outcomes": null_outcomes,
        }
        if escalate_candidates:
            summary["escalate_candidates"] = escalate_candidates
            summary["escalate_note"] = (
                f"Gap(s) {escalate_candidates} found high-friction: insufficient credible "
                "web evidence under the local-only operator. Worth a morning `--escalate` pass."
            )
        return summary

    finally:
        if is_local:
            _flip_fn("big")  # best-effort; idempotent; covers exception + SIGTERM exit paths
        try:
            _sig.signal(_sig.SIGTERM, _prev_sigterm)
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_counter_query(
    gap: Gap,
    pid: str,
    run_dir: Path,
    today: str,
    counter_req: dict,
    read_operator: str,
    critic_operator: str,
    dowser: Any,
    is_local: bool,
    flip_fn: Any,
    gate_fn: Any,
    orig_prov: dict,
) -> bool:
    """Issue one adversarial counter-query for a no-credible-sources null.

    Returns True if the counter-query succeeded and the gap was sourced.
    Writes the appropriate sidecar in all branches.
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
            _write_known_hopeless(run_dir, pid, today, orig_prov, "counter-query read failed")
            return False

        cdraft = counter_drafts[0]
        counter_vs, _ = _phase_big_critique(
            [cdraft], critic_operator, dowser, flip_fn, gate_fn, is_local
        )
        if not counter_vs:
            _write_known_hopeless(run_dir, pid, today, orig_prov, "counter-query critique failed")
            return False

        if counter_vs[0].get("status") == "pass":
            good = [c for c in cdraft.get("citations", []) if _audit_citation(c)]
            if good:
                for c in good:
                    gap.citations.append(BackcasterCitation(type="web", ref=c["url"]))
                gap.unsourced = False
                _write_sidecar(run_dir, pid, {
                    "quest_attempted": today,
                    "verdict": "sourced",
                    "findings": cdraft.get("findings", ""),
                    "citations": good,
                    "provenance": cdraft.get("provenance", {}),
                    "note": "sourced via adversarial counter-query",
                })
                try:
                    _rederive_gap(gap, run_dir)
                except Exception as exc:
                    log.warning("[quest_leg] re-derive failed for %s: %s", pid, exc)
                return True

        # Still empty after counter-query -> known-hopeless
        diagnosis = (counter_vs[0].get("diagnosis", "") if counter_vs else "")
        _write_known_hopeless(run_dir, pid, today, orig_prov, diagnosis)
        return False

    except Exception as exc:
        log.warning("[quest_leg] counter-query failed for %s: %s", pid, exc)
        _write_known_hopeless(run_dir, pid, today, orig_prov, str(exc))
        return False


def _write_known_hopeless(
    run_dir: Path,
    pid: str,
    today: str,
    prov: dict,
    critic_reason: str,
) -> None:
    _write_sidecar(run_dir, pid, {
        "quest_attempted": today,
        "verdict": "insufficient-sources",
        "search_strings": prov.get("search_strings", []),
        "hits_count": prov.get("hits_count", 0),
        "critic_reason": critic_reason,
        "friction_ratio": prov.get("friction_ratio", 0.0),
        "note": "no credible sources after adversarial counter-query",
    })


def _empty_summary(caution: str, n_gaps: int, note: str) -> dict:
    return {
        "gaps_targeted": n_gaps,
        "gaps_sourced": 0,
        "gaps_honest_null": n_gaps,
        "citations_added": 0,
        "caution_before": caution,
        "caution_after": caution,
        "retries_used": 0,
        "null_outcomes": {"insufficient-sources": 0, "high-friction": 0, "infra-unavailable": n_gaps},
        "note": note,
    }
