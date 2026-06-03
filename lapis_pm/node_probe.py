"""Fast connect-probe for GPU inference endpoints.

node_reachable() gates expensive GPU POSTs by probing the endpoint's origin
with a short httpx.get timeout. Results are TTL-memoized per (host, port) so
corroboration + witness probes within one tick encode share a single result.
"""

from __future__ import annotations

import time
from urllib.parse import urlparse

_TTL = 15.0  # seconds; shorter than 60s tick, longer than one encode's duration
_probe_cache: dict[tuple[str, int], tuple[bool, float]] = {}


def node_reachable(url: str, *, timeout: float = 3.0) -> bool:
    """Return True if the host:port is accepting connections, False otherwise.

    Derives a probe target from the URL's scheme+host+port and GETs
    ``{origin}/health``. Any HTTP response (including 404) counts as reachable
    — we detect TCP connectivity, not endpoint health. The failure mode we're
    defending against is a dropped SYN, which results in no response at all.

    Transport is ``httpx.get`` so the test suite's ``_block_sleeping_node_http``
    autouse fixture covers sleep-capable nodes with zero additional guard.

    Results are memoized per (host, port) with a 15 s TTL. Never raises.
    """
    import httpx

    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    cache_key = (host, port)

    now = time.monotonic()
    cached = _probe_cache.get(cache_key)
    if cached is not None:
        result, ts = cached
        if now - ts < _TTL:
            return result

    probe_url = f"{parsed.scheme}://{parsed.netloc}/health"
    try:
        httpx.get(probe_url, timeout=timeout)
        reachable = True
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException, OSError):
        reachable = False
    except Exception:
        reachable = False

    _probe_cache[cache_key] = (reachable, now)
    return reachable
