"""Test notify_policy configuration in registry.yaml."""

from pathlib import Path

from agents_core.shaper import Shaper


def test_fixer_reviewer_notify_policy_infra_only():
    """Verify fixer and reviewer are configured with notify_policy: infra-only."""
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    shaper = Shaper(registry_path)

    assert shaper.get_agent("fixer").notify_policy == "infra-only"
    assert shaper.get_agent("reviewer").notify_policy == "infra-only"


def test_other_agents_notify_policy_always():
    """Regression test: untouched agents should default to always."""
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    shaper = Shaper(registry_path)

    assert shaper.get_agent("fixer_retry").notify_policy == "always"
    assert shaper.get_agent("scout").notify_policy == "always"
