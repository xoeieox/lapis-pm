"""Root conftest: register the user site-packages directory so that the
agents-core editable install (whose .pth file lives in
~/.local/lib/python3.12/site-packages) is activated before test collection.

Without this, the worktree's .pyuserbase overrides getusersitepackages() and
the .pth-based editable finder for agents_core is never executed.
"""

import site as _site
_site.addsitedir('/home/user/.local/lib/python3.12/site-packages')
