"""Test notify_policy configuration in registry.yaml."""

from pathlib import Path

from agents_core.shaper import Shaper


def test_fixer_reviewer_notify_policy_infra_only():
    """Verify fixer and reviewer are configured with notify_policy: infra-only."""
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    shaper = Shaper(registry_path)

    assert shaper.get_agent("fixer").notify_policy == "infra-only"
    assert shaper.get_agent("reviewer").notify_policy == "infra-only"


def test_retry_and_staged_notify_policy_infra_only():
    """lapis-pm-daemon-silent-gaps-v0 B1: the retry classes carry the same
    notify shape as fixer (notify: true + notify_policy: infra-only) so the
    queue seam's infra-class death pages fire on them; completions under
    infra-only stay audit-log-silent so successful retries do not page."""
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    shaper = Shaper(registry_path)

    assert shaper.get_agent("fixer_retry").notify is True
    assert shaper.get_agent("fixer_retry").notify_policy == "infra-only"
    assert shaper.get_agent("fixer_staged").notify is True
    assert shaper.get_agent("fixer_staged").notify_policy == "infra-only"


def test_other_agents_notify_policy_always():
    """Regression test: untouched agents should default to always."""
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    shaper = Shaper(registry_path)

    assert shaper.get_agent("scout").notify_policy == "always"
