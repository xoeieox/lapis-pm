"""Root conftest: register the user site-packages directory so that the
agents-core editable install (whose .pth file lives in
~/.local/lib/python3.12/site-packages) is activated before test collection.

Without this, the worktree's .pyuserbase overrides getusersitepackages() and
the .pth-based editable finder for agents_core is never executed.

Also registers archetypes-core (git clone, not yet in user site-packages)
so that lapis_pm modules that import archetypes_core.* are resolvable.
"""

import sys
import site as _site

_site.addsitedir('/home/user/.local/lib/python3.12/site-packages')

# archetypes-core is a git-managed dependency not installed as a user editable
# package; add the source tree directly so imports resolve in tests.
_ARCHETYPES_CORE_PATH = '/srv/git/archetypes-core'
if _ARCHETYPES_CORE_PATH not in sys.path:
    sys.path.insert(0, _ARCHETYPES_CORE_PATH)


import pytest as _pytest


# Sleep-capable cluster endpoints. Real HTTP to these hangs 45-120s when the node
# is asleep — StarHouse qwen (:8081) and Synapse (:8401) sleep under the Gate-6
# all-hours bind, which is the steady state. The :19999 marker is the deliberately
# unreachable port used by backcaster degraded-path tests. The production code that
# calls these (local_reviewer_witness, corroboration_adapter, gap_analyze) is all
# fail-soft on ConnectError, so converting a real call into an *instant* ConnectError
# keeps the suite hermetic and fast without changing exercised behaviour. A test that
# wants a specific response patches httpx.post/get itself — that inner patch shadows
# this guard within the test's `with` block.
_BLOCKED_ENDPOINT_MARKERS = (":8081", ":8401", ":19999")


@_pytest.fixture(autouse=True)
def _block_sleeping_node_http(monkeypatch):
    import httpx

    _real_post = httpx.post
    _real_get = httpx.get

    def _is_blocked(url) -> bool:
        return any(m in str(url) for m in _BLOCKED_ENDPOINT_MARKERS)

    def _guarded(real):
        def _wrapped(url, *args, **kwargs):
            if _is_blocked(url):
                raise httpx.ConnectError(
                    f"blocked by test guard (sleep-capable node): {url}"
                )
            return real(url, *args, **kwargs)
        return _wrapped

    monkeypatch.setattr(httpx, "post", _guarded(_real_post))
    monkeypatch.setattr(httpx, "get", _guarded(_real_get))


@_pytest.fixture(autouse=True)
def _clear_tick_corr_cache():
    """pm_core._tick_corr_cache is a module-global keyed by (target_id, pr_num).
    A real tick clears it per target at the top of _encode_gpu_results, but tests
    that call _encode_gpu_results / _persist_review_state_cache directly can leave
    entries behind that pollute a later same-target test. Clear it around every
    test so ordering can't change outcomes."""
    try:
        from lapis_pm import pm_core
    except Exception:
        yield
        return
    pm_core._tick_corr_cache.clear()
    yield
    pm_core._tick_corr_cache.clear()
