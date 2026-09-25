"""Registry-resolved gate-lane resolver (S1 vendored shim).

gate-lanes-registry-driven-flashnext-v0, S1: gate legs (spec_review facets/
council, corroboration node2, many-eyes lens pins) resolve their model lane
through the gw-seats registry (the :8408 reality_view, served-model-name
pins), never through hardcoded ports. :8081 stays the default only as the
registry-absent fallback — the same pattern leg-1 established for the fixer
lane (fixer_flash registry row f0fb039, served-id + backend :30000).

Protocol boundary (re-gate fold, Rev 2026-09-25): this module IS the seam.
When the companion bind lands (gate-lanes-registry-driven-flashnext-v0-
agents-core, which promotes this shim into ``agents_core.lane_registry``),
``resolve_gate_lane()`` transparently delegates to the agents-core helper
when it is importable and byte-identical to the local implementation
otherwise (fail-closed: an import/attribute error on the companion side
falls back to the local implementation, never to a guess).

Caller contract (re-gate fold; resolves the council stand-asides):

* ``None`` (registry blind — the registry is unreachable, the payload is
  malformed, or no gate lane is registered) is the ONLY case in which
  callers fall back to the existing GW_URL behavior byte-identically. A
  blind registry is "no information", and the legacy env-var-driven path
  (GW_URL / SWARM_URL) is what a blind gate leg ran on before this target —
  so the blind path must be indistinguishable from today.
* When the registry is readable and an explicitly-requested lane (e.g.
  ``--facets-operator flashnext``) is inactive or not serving, the caller
  records an honest ``leg_down`` (S5) and NEVER falls back to the legacy
  lane — a requested-but-dead lane is reported as absent, not masked.
  A silent legacy fallback here is a "lying leg": it degrades the truth
  of the seat's absence (murasaki-shikibu-canonical stand-aside,
  honesty invariant).

Registry URL discovery is explicit in code: ``GW_SEATS_URL`` env override,
default ``http://203.0.113.11:8408`` (mirroring the GW_URL/SWARM_URL
env-default pattern; the Erah-canonical stand-aside asked for the registry
URL to be marked in code rather than assumed).

Registry shape (verified live 2026-09-25 against the :8408 status):
the payload carries ``reality_view`` (``reality`` in {"slot1-solo",
"flashnext-solo", ...}, ``primary`` with the 27B seat's port/model_root)
and ``seats`` — one row per seat port (8081, 8082, 8500, 30000) with
``state`` ("serving" | "down"), ``model`` (served id), ``model_root``,
``bind``. A gate lane is a seat row whose port is a known gate-lane port
and whose state is "serving". The flashnext row follows the f0fb039 row
shape: served id ``Qwen3.8-Flash-Next-NVFP4-SSD-Stream`` on backend
:30000.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

# Registry base URL — explicit env override, documented default (the
# Erah-canonical stand-aside: discovery mechanism marked in code).
DEFAULT_GW_SEATS_URL = "http://203.0.113.11:8408"
GW_SEATS_TIMEOUT_S = 4.0

# Gate-lane ports. :8081 is the 27B slot1 seat (the legacy gravitywell
# lane); :30000 is the flash-next sglang seat (the flashnext lane,
# f0fb039 row shape). Only these two ports are gate lanes — :8082 (slot2
# reference leg) and :8500 are not.
GATE_LANE_PORTS = (8081, 30000)

# The flashnext lane identity (f0fb039 row shape). The served id is the
# EXACT id :30000/v1/models data[0].id advertises (the sglang seat serves
# the concrete id — the slot-alias pitfall is a vLLM-seat property).
FLASHNEXT_LANE_NAME = "flashnext"
FLASHNEXT_LANE_PORT = 30000
FLASHNEXT_SERVED_ID = "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
FLASHNEXT_MODEL_ROOT_MARKER = "flash"  # model_root substring (case-insensitive)

# The 27B lane identity (slot1, :8081).
SLOT1_LANE_NAME = "slot1"
SLOT1_LANE_PORT = 8081
SLOT1_MODEL_ROOT_MARKER = "27b"


@dataclass
class GateLane:
    """One registry-resolved gate lane.

    ``name``: the lane name ("flashnext" | "slot1").
    ``base_url``: the lane's base URL (no trailing path) — the endpoint
        callers build their /v1/* probes and completions POSTs against.
    ``served_model``: the served model id the lane advertises (the
        registry's served-model-name pin; ``None`` when the registry row
        did not carry one and the caller must probe /v1/models live).
    """
    name: str
    base_url: str
    served_model: Optional[str] = None


def _gw_seats_url() -> str:
    """The gw-seats registry base URL, read at call time (env-overridable
    so tests never hit the live registry)."""
    return os.environ.get("GW_SEATS_URL", DEFAULT_GW_SEATS_URL)


def _registry_host() -> str:
    """The CLIENT-REACHABLE host for lane dialing: the gw-seats registry's
    own origin host (GW_SEATS_URL), default the GW tailscale IP.

    Why not the seat row's ``bind``? ``bind`` is the SERVING bind (the
    sglang/vllm bind-address on the GW box) — the live :8408 row for the
    serving :30000 seat advertises ``bind: "0.0.0.0"`` (gw-seats v0,
    verified live 2026-09-25), which is a wildcard listen address, not a
    dialable client host. The f0fb039 fixer_flash precedent pins the full
    URL (http://203.0.113.11:30000) for exactly this reason."""
    try:
        from urllib.parse import urlparse

        host = urlparse(_gw_seats_url()).hostname
        if isinstance(host, str) and host:
            return host
    except Exception:
        pass
    return "203.0.113.11"


def _lane_name_for_port(port: int) -> str:
    if port == FLASHNEXT_LANE_PORT:
        return FLASHNEXT_LANE_NAME
    if port == SLOT1_LANE_PORT:
        return SLOT1_LANE_NAME
    return f"port-{port}"


def _seat_rows(payload: dict) -> list[dict]:
    """The registry payload's seat rows (defensive: malformed -> [])."""
    seats = payload.get("seats")
    if not isinstance(seats, list):
        return []
    return [s for s in seats if isinstance(s, dict)]


def _lane_from_seats(seats: list[dict], lane_name: str) -> Optional[GateLane]:
    """Build a GateLane for ``lane_name`` from the seat rows, or None when
    the lane is absent or not serving.

    "Serving" is the registry's own state (state == "serving") — the
    registry is the seat-state contract's source of truth. A lane the
    registry declares down is NOT a lane; the caller's honest leg_down
    (S5) fires, never a masked legacy fallback.
    """
    for seat in seats:
        port = seat.get("port")
        if port != _lane_port(lane_name):
            continue
        if seat.get("state") != "serving":
            return None
        # ``bind`` is the serving-bind, NOT a client host (review HIGH-1,
        # 2026-09-25): the live serving row advertises "0.0.0.0", which
        # would build a phantom http://0.0.0.0:<port> lane. Wildcards (and
        # absent bind) dial the registry's own origin host instead; a
        # concrete bind is still honored (registry-forwarded rows).
        host = seat.get("bind")
        if not isinstance(host, str) or host.strip() in ("", "0.0.0.0", "::", "*", "[::]"):
            host = _registry_host()
        base_url = f"http://{host}:{port}"
        served_model = seat.get("model")
        if not isinstance(served_model, str) or not served_model:
            served_model = None
        return GateLane(name=lane_name, base_url=base_url, served_model=served_model)
    return None


def _lane_port(lane_name: str) -> int:
    if lane_name == FLASHNEXT_LANE_NAME:
        return FLASHNEXT_LANE_PORT
    if lane_name == SLOT1_LANE_NAME:
        return SLOT1_LANE_PORT
    raise ValueError(f"unknown gate lane {lane_name!r}")


def _reality_is_flashnext_solo(view: dict) -> bool:
    """True when the registry's reality_view says flashnext holds the seat
    (flashnext-solo: the 27B is down, :30000 is the live lane)."""
    reality = str(view.get("reality", "")).lower()
    if "flashnext" in reality:
        return True
    # Belt-and-braces: a reality string that names the flash model root as
    # the anchor is flashnext reality even if the label drifts.
    anchor = str(view.get("anchor", "")).lower()
    return FLASHNEXT_MODEL_ROOT_MARKER in anchor


def _reality_is_slot1(view: dict) -> bool:
    """True when the registry's reality_view says the 27B slot1 seat holds
    the seat (slot1-solo / 27B up)."""
    reality = str(view.get("reality", "")).lower()
    if "slot1" in reality or "27b" in reality:
        return True
    anchor = str(view.get("anchor", "")).lower()
    return SLOT1_MODEL_ROOT_MARKER in anchor


def _resolve_from_payload(payload: dict, lane: Optional[str] = None) -> Optional[GateLane]:
    """Resolve a gate lane from a parsed registry payload (the pure
    function the fetcher seam feeds).

    ``lane``: "flashnext" | "slot1" | None. None = "the live lane": the
    flashnext lane when reality is flashnext-solo, else the slot1 lane
    when the 27B is up, else None (registry readable but no gate lane is
    serving — callers treat this as the requested-lane-absent case, NOT
    as blind).

    Returns None ONLY for the blind/malformed cases (payload not a dict,
    no seats) — a readable registry with no serving gate lane returns
    None through the same shape, and callers distinguish the two by
    probing the registry separately when the distinction matters (S5).
    """
    if not isinstance(payload, dict):
        return None
    seats = _seat_rows(payload)
    if not seats:
        return None

    if lane is None:
        view = payload.get("reality_view")
        view = view if isinstance(view, dict) else {}
        if _reality_is_flashnext_solo(view):
            return _lane_from_seats(seats, FLASHNEXT_LANE_NAME)
        if _reality_is_slot1(view):
            return _lane_from_seats(seats, SLOT1_LANE_NAME)
        # Reality label unrecognized: fall back to the seat-state truth —
        # whichever known gate lane is actually serving. (A readable
        # registry with an unlabeled reality is NOT blind.)
        for name in (FLASHNEXT_LANE_NAME, SLOT1_LANE_NAME):
            lane_obj = _lane_from_seats(seats, name)
            if lane_obj is not None:
                return lane_obj
        return None

    if lane not in (FLASHNEXT_LANE_NAME, SLOT1_LANE_NAME):
        return None
    return _lane_from_seats(seats, lane)


def _local_voicing_lease_free() -> tuple[bool, str]:
    """S8: the local-voicing lease-free guard.

    A LOCAL voicing must not acquire a seat-node lease. The guard probes
    the doorman for either of the two shapes that make the acquire die 409
    on node "gravitywell" (the node the legacy gravitywell voicing leases,
    and the node the live 409 fired on):

    (a) an ACTIVE lease on that node — refused as
        ``local_seat_lease_refused``;
    (b) the LIVE 409 shape (review HIGH-2, 2026-09-25): the GW seat is NOT
        serving and the doorman's flashnext window sub-view is UP
        (seat_state up_registered/up_unverified, or window == "active") —
        the observed 409 fired with lease_count 0
        (finding/council-lease-409-under-flashnext-window-2026-09-25;
        doorman ensure_serving refuses every acquire in this state), so
        keying only on held leases pins the wrong shape. Refused as
        ``local_voicing_flashnext_window``.

    Either shape REFUSES the local voicing (leg_down, honest) — the
    doorman /lease/acquire call site is pinned
    unreachable on this path, and the refusal is a leg_down, NEVER a
    fallback-run on the gravitywell lane.

    Scope: the /status payload carries per-node leases
    (``nodes.<node>.leases``); the guard PREFERS the gravitywell node's
    lease list (node-scoped — a lease held on some OTHER seat node does
    not contend with the gravitywell seat the local leg would race). A
    client whose /status shape lacks per-node leases falls back to the
    GLOBAL lease_count (the doorman-wide count) — a coarser but
    conservative refusal, and the honest shape for that client.

    Blind never refuses: the doorman is unreachable -> (True, "") — the
    local voicing runs lease-free exactly as today (the 409 only fires
    when the doorman IS reachable and the seat is leased, which is the
    shape this guard refuses BEFORE the leg would acquire).

    Returns (ok, reason): (True, "") = lease-free, run the local leg;
    (False, "local_seat_lease_refused") = a gravitywell-seat lease is
    held, refuse the local leg (honest leg_down);
    (False, "local_voicing_flashnext_window") = the live-409 seat state
    (GW seat not serving + flashnext window up), refuse the local leg
    (honest leg_down).
    """
    try:
        from agents_core.doorman_client import DoormanClient
        client = DoormanClient()
        # A read-only seat-state SENSE (GET /status), never an acquire: the
        # guard must not itself take the lease it is guarding against.
        # DoormanClient.status() is the no-arg /status snapshot (the
        # gravitywell doorman's own seat state — the node the legacy
        # gravitywell voicing leases and the node the live 409 fired on).
        status = client.status()
    except Exception:
        # Blind doorman (unreachable / malformed) -> never refuses. The
        # local voicing runs lease-free as today; the 409 shape requires a
        # reachable doorman with a held lease, which this path never
        # produces because it never acquires.
        return (True, "")
    try:
        if not isinstance(status, dict):
            return (True, "")
        # Prefer the gravitywell node's lease list (the /status payload
        # carries per-node leases: nodes.<node>.leases). A lease held on a
        # different node does not contend with the gravitywell seat.
        nodes = status.get("nodes")
        if isinstance(nodes, dict):
            gw_node = nodes.get("gravitywell")
            if isinstance(gw_node, dict):
                # (b) HIGH-2 fold: the live 409 shape is NOT a held lease —
                # it fires when the GW seat is not serving and the doorman
                # sees the flashnext window UP (ensure_serving refuses every
                # acquire in that state; lease_count was 0 in the observed
                # 409). Sense that predicate from the same /status snapshot
                # and refuse BEFORE the leg would acquire. Fail-soft note:
                # this reads the doorman's cached snapshot; when it is stale
                # the refusal is a conservative honest leg_down, never a
                # fallback-run.
                if not gw_node.get("serving", False):
                    fx = gw_node.get("flashnext")
                    if isinstance(fx, dict):
                        _st = fx.get("seat_state")
                        _window_up = (
                            (isinstance(_st, str) and _st.startswith("up_"))
                            or fx.get("window") == "active"
                            or (_st is None and bool(fx.get("seat_health")))
                        )
                        if _window_up:
                            return (False, "local_voicing_flashnext_window")
                leases = gw_node.get("leases")
                if isinstance(leases, list):
                    return (
                        (False, "local_seat_lease_refused") if leases else (True, "")
                    )
                lease_count = gw_node.get("lease_count", 0)
                if isinstance(lease_count, int) and lease_count > 0:
                    return (False, "local_seat_lease_refused")
                return (True, "")
        # The client's /status shape lacks per-node leases: fall back to
        # the GLOBAL lease_count (the doorman-wide count — coarser but
        # conservative; the docstring documents this fallback shape).
        if status.get("lease_count", 0) > 0:
            return (False, "local_seat_lease_refused")
    except Exception:
        pass
    return (True, "")


def _fetch_payload(timeout: float = GW_SEATS_TIMEOUT_S) -> dict:
    """Live registry read (GET {GW_SEATS_URL}/). Any error -> {} (blind).

    The registry is a SENSE read (GET), never a seat flip. Fail-soft:
    unreachable / non-200 / malformed JSON all collapse to the empty
    payload, which ``_resolve_from_payload`` maps to None (blind).
    """
    import httpx

    resp = httpx.get(f"{_gw_seats_url()}/", timeout=timeout)
    if resp.status_code != 200:
        return {}
    payload = resp.json()
    return payload if isinstance(payload, dict) else {}


def resolve_gate_lane(
    lane: Optional[str] = None,
    fetcher: Optional[Callable[[], dict]] = None,
) -> Optional[GateLane]:
    """Resolve a gate lane through the gw-seats registry.

    Args:
        lane: "flashnext" | "slot1" | None. None = the live lane (the
            flashnext row when reality is flashnext-solo, the slot1 row
            when the 27B is up).
        fetcher: injectable callable returning the raw registry payload
            (the test seam — never a live call from a test). None = the
            live httpx read of ``GW_SEATS_URL``.

    Returns:
        GateLane (name, base_url, served_model) when the registry is
        readable and the lane is registered + serving.

        None in the ONLY blind case: the registry is unreachable, the
        payload is malformed, no seats are registered, or (for an
        explicit ``lane``) the lane is not registered / not serving.
        Callers fall back to the existing GW_URL behavior byte-identically
        on None — and an explicitly-requested inactive lane is reported
        as an honest leg_down by the caller, NEVER a silent legacy
        fallback (the "lying leg" the re-gate fold kills).
    """
    # Protocol boundary: delegate to the companion agents-core helper when
    # it exists (gate-lanes-registry-driven-flashnext-v0-agents-core
    # promotes this shim into agents_core.lane_registry). Fail-closed: any
    # import/attribute/shape error on the companion side falls back to the
    # local implementation below — never a guess, never a partial result.
    try:
        from agents_core import lane_registry as _companion
        companion = _companion.resolve_gate_lane
        if callable(companion):
            if fetcher is not None:
                return companion(lane=lane, fetcher=fetcher)
            return companion(lane=lane)
    except Exception:
        # Companion-side any-error (import, attribute, shape, RUNTIME):
        # fall through to the local implementation below. A companion bug
        # must degrade the shim, never crash the gate (LOW-7, review
        # 2026-09-25 — the previous (ImportError, AttributeError, TypeError)
        # catch let a companion runtime error escape to the gate).
        pass

    if fetcher is not None:
        payload = fetcher()
    else:
        try:
            payload = _fetch_payload()
        except Exception:
            return None  # blind — transport error
    return _resolve_from_payload(payload, lane)


def gate_lane_serving(
    lane: Optional[str] = None,
    fetcher: Optional[Callable[[], dict]] = None,
    probe: Optional[Callable[[str], Optional[str]]] = None,
) -> tuple[bool, str]:
    """Lane-aware serving probe (S5).

    Probes the lane's actual /v1/models for the served-model-name instead
    of the hardcoded :8081 SWARM_URL probe. Returns (serving, reason):

    * (True, "") — the lane is serving and (when the registry pinned a
      served id) the live /v1/models advertises it.
    * (False, "registry_blind") — the registry is unreachable/malformed
      (the ONLY case the caller may fall back to legacy behavior).
    * (False, "<lane>_not_serving") — the registry is readable and the
      lane is absent or not serving, or the live probe failed / does not
      advertise the pinned served id. An honest leg_down, never a silent
      mis-skip, never a masked legacy fallback.

    ``probe`` is injectable: a callable taking the lane base_url and
    returning the live served model id (or None when /v1/models is not
    serving). None = the live agents_core.llm.swarm_model probe.
    """
    if probe is None:
        def probe(base_url: str) -> Optional[str]:  # type: ignore[no-redef]
            try:
                from agents_core.llm import swarm_model
                return swarm_model(base_url)
            except Exception:
                return None

    # S8 (gate-lanes-registry-driven-flashnext-v0, hardened acceptance):
    # a LOCAL-voiced council leg must be lease-free. The live 409 evidence
    # (finding/council-lease-409-under-flashnext-window-2026-09-25): with
    # flash-next holding the seat, the council leg under --council-voicing
    # local still POSTed /lease/acquire on node "gravitywell" and died with
    # 409 -> "Failed to submit council". The doorman seat-state contract
    # applies to every gate leg: a local voicing acquires NO seat-node
    # lease. This guard is the lapis-pm-side pin: it makes the doorman
    # /lease/acquire call site UNREACHABLE on the local-voicing code path —
    # the 409 contention shape is pinned unreachable, not merely that a
    # lease-call count is zero (a zero-count assertion on a path the
    # acquire never reaches is vacuous; the live 409 fired before the
    # voicing branch was evaluated).
    #
    # The guard is a fail-closed REFUSE, never a fallback-run: a local
    # voicing that finds either 409 shape on the doorman snapshot — a
    # gravitywell-seat lease held, or the live-409 seat state (GW seat not
    # serving + flashnext window up, the shape that fired 409 with
    # lease_count 0) — is a leg_down (the honest
    # "local_seat_lease_refused" / "local_voicing_flashnext_window"), and
    # the caller must NOT re-route the leg to the gravitywell lane to
    # "keep going".
    #
    # Blind never refuses: the doorman is unreachable (or the lane is
    # registry-blind) -> the guard is a no-op and the local voicing runs
    # lease-free as today.
    if lane == "local":
        return _local_voicing_lease_free()

    lane_obj = resolve_gate_lane(lane=lane, fetcher=fetcher)
    if lane_obj is None:
        # Distinguish blind (registry unreadable) from lane-absent: a
        # readable registry with a down lane must NOT report blind — the
        # caller's honest leg_down depends on the distinction.
        if fetcher is not None:
            payload = fetcher()
        else:
            try:
                payload = _fetch_payload()
            except Exception:
                payload = None
        if isinstance(payload, dict) and _seat_rows(payload):
            return (False, f"{lane or 'gate'}_not_serving")
        return (False, "registry_blind")

    served = probe(lane_obj.base_url)
    if served is None:
        return (False, f"{lane_obj.name}_not_serving")
    if lane_obj.served_model and served != lane_obj.served_model:
        # The lane serves a DIFFERENT model than the registry pinned —
        # the pin is the contract (served-model-name pins); a mismatched
        # served id is not the requested lane.
        return (False, f"{lane_obj.name}_not_serving")
    return (True, "")
