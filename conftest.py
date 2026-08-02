"""Root conftest: register the user site-packages directory so that the
agents-core editable install (whose .pth file lives in
~/.local/lib/python3.12/site-packages) is activated before test collection.

Without this, the worktree's .pyuserbase overrides getusersitepackages() and
the .pth-based editable finder for agents_core is never executed.

Also registers archetypes-core (git clone, not yet in user site-packages)
so that lapis_pm modules that import archetypes_core.* are resolvable.
"""

import os
import sys
import tempfile
import site as _site

# Substrate isolation: redirect the canonical Zephyr attribution log and the slot
# blackboard to throwaway temp DBs for the whole test session BEFORE any test imports
# zephyr.attribution / agents_core.slots (both bind their default DB path at import /
# first-recorder construction). Without this, tests that exercise the gate-5
# slot-close + deposit path write into the PRODUCTION /data/zephyr/attribution.db and
# /data/slots/slots.db. Set only if a test runner hasn't already pinned them.
_test_substrate_dir = tempfile.mkdtemp(prefix="lapis-pm-test-substrate-")
# Unconditional set — silently no-op'ing on already-exported vars risks writes to
# /data/zephyr/attribution.db, /data/slots/slots.db, and /data/memory/mem.db.
os.environ["ZEPHYR_ATTRIBUTION_DB"] = os.path.join(_test_substrate_dir, "attribution.db")
os.environ["SLOTS_DB_PATH"] = os.path.join(_test_substrate_dir, "slots.db")
# MEM_DB_PATH must be set before agents_core.mem is first imported (it resolves
# DB_PATH at module load time). Setting it here — before sys.path manipulation and
# any pytest import — mirrors the ZEPHYR_ATTRIBUTION_DB / SLOTS_DB_PATH pattern.
os.environ["MEM_DB_PATH"] = os.path.join(_test_substrate_dir, "mem.db")
# Default test posture = master (today's pre-node-identity behavior): the existing
# suite exercises write/merge/dispatch/flip paths that were unconditional before
# lapis_pm.node_identity existed and must keep behaving exactly as before (spec
# lapis-pm-node-write-ownership-v0, I6). setdefault so a test/run that explicitly
# wants the independent posture (e.g. the negative-smoke AC9 scenario) can still
# set LAPIS_PM_NODE_ROLE=independent before this module is imported.
os.environ.setdefault("LAPIS_PM_NODE_ROLE", "master")

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
def _reset_node_identity_cache():
    """lapis_pm.node_identity caches the resolved NodeIdentity as a module
    global after the first call in a process. Reset it around every test so
    env-var/monkeypatch changes a test makes are actually re-resolved instead
    of leaking a stale identity from an earlier test."""
    try:
        from lapis_pm import node_identity
    except Exception:
        yield
        return
    node_identity._reset_for_tests()
    yield
    node_identity._reset_for_tests()


@_pytest.fixture(autouse=True)
def _clear_node_probe_cache():
    """Clear node_probe's TTL cache between tests for deterministic isolation."""
    try:
        from lapis_pm import node_probe
    except Exception:
        yield
        return
    node_probe._probe_cache.clear()
    yield
    node_probe._probe_cache.clear()


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


@_pytest.fixture(autouse=True)
def _clear_reviewer_attempt_ceiling_state():
    """Reviewer-attempt-ceiling counters (lapis-pm-reviewer-attempt-ceiling-v0)
    are deliberately persistent — they must survive new SHAs, so nothing in
    production code resets them. That means the shared session-wide mem.db
    (see MEM_DB_PATH above) accumulates them across tests. Many tests reuse
    the same target_id/pr_number defaults (e.g. "tid"/42), so without this a
    test earlier in a randomized run can push a later, unrelated test's PR
    over the ceiling. Wipe both key namespaces around every test."""
    try:
        from lapis_pm import pm_core
    except Exception:
        yield
        return

    def _wipe():
        try:
            store = pm_core._mem()
            for prefix in ("pm/reviewer-attempts/", "pm/reviewer-attempt-ceiling/"):
                for rec in store.list_by_prefix(prefix, limit=10_000):
                    store.delete(rec["key"])
        except Exception:
            pass

    _wipe()
    yield
    _wipe()


@_pytest.fixture(autouse=True)
def _isolate_comment_store(monkeypatch, tmp_path):
    """Redirect lapis_pm.episodic._store to a per-test tmp CommentStore.

    Without this, any code path that calls episodic.write_observation / write_dispatch /
    write_result / etc. (all delegate to _store()) writes to the production
    /srv/lapis/targets/comments/ directory. Tests that exercise _set_brief_outstanding with
    a synthesis-failed brief are the canonical leaker (pm:synthesis-failed observations
    for my-target). This autouse fixture ensures every test gets an isolated store.
    """
    try:
        from agents_core.comments import CommentStore
        from lapis_pm import episodic
    except Exception:
        yield
        return
    monkeypatch.setattr(episodic, "_store", lambda: CommentStore(root=tmp_path))
    yield


@_pytest.fixture(autouse=True)
def _block_real_pushover(monkeypatch):
    """Safety net: no test in this suite may perform a real Pushover send.

    agents_core.notify.send_notification has THREE independently-bound call sites in
    this codebase, not one: lapis_pm/pm_core.py resolves it via a deferred
    `from agents_core.notify import send_notification` *inside* each function body
    (so patching the agents_core.notify module attribute is sufficient there), but
    lapis_pm/brief.py and lapis_pm/brief_gem.py both do a *module-level* import,
    binding their own independent name at import time — patching the source module
    attribute does NOT affect those already-bound references. All three must be
    patched or a test exercising brief.py/brief_gem.py that forgets an explicit
    patch will still fire a real Pushover send. Individual tests may still assert
    on notify behavior by patching any of these three references themselves — that
    inner patch wins for the duration of its own `with`/monkeypatch scope.
    """
    import agents_core.notify as _notify_mod
    from lapis_pm import brief as _brief_mod
    from lapis_pm import brief_gem as _brief_gem_mod

    monkeypatch.setattr(_notify_mod, "send_notification", lambda *a, **kw: False)
    monkeypatch.setattr(_brief_mod, "send_notification", lambda *a, **kw: False)
    monkeypatch.setattr(_brief_gem_mod, "send_notification", lambda *a, **kw: False)
