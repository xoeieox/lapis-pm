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
# /data/zephyr/attribution.db and /data/slots/slots.db (the production DBs).
os.environ["ZEPHYR_ATTRIBUTION_DB"] = os.path.join(_test_substrate_dir, "attribution.db")
os.environ["SLOTS_DB_PATH"] = os.path.join(_test_substrate_dir, "slots.db")

_site.addsitedir('/home/user/.local/lib/python3.12/site-packages')

# archetypes-core is a git-managed dependency not installed as a user editable
# package; add the source tree directly so imports resolve in tests.
_ARCHETYPES_CORE_PATH = '/srv/git/archetypes-core'
if _ARCHETYPES_CORE_PATH not in sys.path:
    sys.path.insert(0, _ARCHETYPES_CORE_PATH)
