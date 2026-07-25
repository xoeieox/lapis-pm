"""Test the fixer anti-stall / single-turn-completion contract in registry.yaml.

Guards against the initial claude -p fixer ending its turn mid-task with
deferred prose ("I'll report back once tests finish") and zero commits/PR.
See spec: fixer-antistall-prompt-hardening-v0.
"""

from pathlib import Path

from agents_core.shaper import Shaper


def _registry_and_fixer():
    registry_path = Path(__file__).parent.parent / "lapis_pm" / "registry.yaml"
    shaper = Shaper(registry_path)
    return shaper, shaper.get_agent("fixer")


def test_shared_preamble_states_no_later_turn_contract():
    shaper, _ = _registry_and_fixer()
    preamble = shaper._shared_preamble.lower()

    assert "no later turn" in preamble
    assert "single-response" in preamble or "single response" in preamble


def test_shared_preamble_forbids_backgrounding_and_deferral_phrases():
    shaper, _ = _registry_and_fixer()
    preamble = shaper._shared_preamble.lower()

    # Backgrounding forbidden.
    assert "run_in_background" in preamble
    assert "nohup" in preamble

    # Deferral phrases named as forbidden.
    assert "i'll report back" in preamble
    assert "pausing here" in preamble


def test_shared_preamble_names_two_terminal_states():
    shaper, _ = _registry_and_fixer()
    preamble = shaper._shared_preamble.lower()

    assert "clean abort" in preamble
    assert "no third option" in preamble


def test_fixer_step5_requires_synchronous_foreground_tests():
    _, fixer = _registry_and_fixer()
    template = fixer.system_template.lower()

    assert "synchronously in the foreground" in template
    assert "do not background the test" in template


def test_fixer_registry_still_format_interpolates_cleanly():
    """Regression: any literal brace added to the preamble/template must be
    doubled, or str.format raises at dispatch time."""
    shaper, fixer = _registry_and_fixer()
    vars_ = dict(
        repo="lapis-pm",
        repo_cwd="/srv/lapis/lapis-pm",
        target_id="t",
        pr_number=1,
        spec_summary="s",
        intent_block="",
        steer_directive_block="",
        base_branch="main",
        existing_branch="e",
        slug="slug",
    )
    composed = shaper._compose_system(fixer, vars_)
    assert "no later turn" in composed.lower()
