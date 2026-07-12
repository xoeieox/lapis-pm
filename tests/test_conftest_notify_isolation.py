"""Guards the root conftest's `_block_real_pushover` autouse fixture.

Without that fixture, `_post_land_git_pull` on a critical repo whose simulated
`git pull` fails reaches the real `agents_core.notify.send_notification`, which
(when PUSHOVER_* creds are present in the environment) fires a real Pushover
push. This test deliberately does NOT patch `send_notification` itself — the
whole point is to prove the autouse fixture in conftest.py is the thing
blocking the real send, not an explicit per-test patch.
"""

import subprocess
from unittest.mock import patch

from lapis_pm import pm_core


def _make_completed_process(returncode=0, stderr="", stdout=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_post_land_git_pull_failure_never_reaches_real_send_notification(monkeypatch):
    """A failing pull on a critical repo must not execute the real
    agents_core.notify.send_notification body (spied via _capture_event,
    which only the real function calls)."""
    import agents_core.notify as notify_mod

    capture_spy = []
    monkeypatch.setattr(notify_mod, "_capture_event", lambda **kw: capture_spy.append(kw))

    def fake_run(cmd, **kwargs):
        if cmd[0] == "git" and "rev-parse" in cmd:
            return _make_completed_process(returncode=0, stdout="somesha")
        if cmd[0] == "git" and "pull" in cmd:
            return _make_completed_process(returncode=1, stderr="simulated pull failure")
        return _make_completed_process(returncode=0, stdout="active")

    with patch("lapis_pm.pm_core.subprocess.run", side_effect=fake_run):
        pm_core._post_land_git_pull("agents-core")

    assert capture_spy == [], (
        "real agents_core.notify.send_notification body executed (via _capture_event) "
        "— the autouse _block_real_pushover fixture in conftest.py did not intercept it"
    )
