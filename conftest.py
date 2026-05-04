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
