#!/usr/bin/env python3
"""Minimal static file server for /srv/lapis/briefs/audio.

ONLY used if tailscale serve's static-directory mount is unavailable — see
scripts/setup_podcast_feed_serve.sh for the detection logic that decides
which path to use (gardener-tts-feed-v0, R4). Binds the tailnet IP
explicitly, never loopback: a bare 127.0.0.1 bind hits a known 502 trap when
tailscale serve proxies to it on this host (the existing voice-PWA mount
already proxies to 203.0.113.10, not localhost — same rule applies here).

Stdlib only — no new dependency for a process that only exists as a
fallback.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os

_LOOPBACK_ADDRESSES = {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bind", default=os.environ.get("TTS_FEED_PROXY_IP", "203.0.113.10"),
        help="Tailnet IP to bind — never loopback (502 trap).",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("TTS_FEED_FALLBACK_PORT", "8413")),
    )
    parser.add_argument(
        "--dir", default=os.environ.get("TTS_FEED_AUDIO_DIR", "/srv/lapis/briefs/audio"),
    )
    args = parser.parse_args()

    if args.bind in _LOOPBACK_ADDRESSES:
        raise SystemExit(
            f"refusing to bind loopback ({args.bind}) — known 502 trap when tailscale "
            f"serve proxies to it on this host; bind the tailnet IP instead"
        )

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=args.dir)
    server = http.server.ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"podcast_static_server: serving {args.dir} on {args.bind}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
