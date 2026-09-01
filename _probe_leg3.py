import json
from unittest.mock import MagicMock, patch
from tests.test_panel_leg_survival import _verdict_comment, _starved_verdict
from lapis_pm import pm_core

comments = [
    _verdict_comment("2026-09-01T10:00:00+00:00", 7, 1, _starved_verdict(("local_witness",))),
    _verdict_comment("2026-09-01T11:00:00+00:00", 7, 2, _starved_verdict(("local_witness", "corroboration"))),
    _verdict_comment("2026-09-01T12:00:00+00:00", 7, 3, _starved_verdict(("local_witness",))),
]

def fake_for_cycle(target_id, pr_number, cyc):
    for c in comments:
        for t in c.tags:
            if t == f"pm:reviewer:pr={pr_number}:cycle={cyc}:verdict=fixable":
                return json.loads(c.content.split("\n", 1)[-1].strip())
    return None

with (
    patch("lapis_pm.pm_core._review_verdict_for_cycle", side_effect=fake_for_cycle),
    patch("lapis_pm.pm_core.episodic.all_comments", return_value=comments),
    patch("lapis_pm.pm_core._mem") as mock_mem,
    patch("lapis_pm.pm_core.episodic.write_hold") as mock_hold,
    patch("lapis_pm.pm_core.brief.synthesize") as mock_synth,
    patch("lapis_pm.pm_core._set_brief_outstanding") as mock_set_outstanding,
):
    mock_mem.return_value.get.return_value = None
    mock_brief = MagicMock()
    mock_brief.comment_id = "cid-test"
    mock_synth.return_value = mock_brief
    action = pm_core._escalate_noop_retry_if_degraded("tid", {"gpu_id": "task-1", "agent_type": "fixer_retry", "pr_number": 7, "cycle": 3}, 7)
    print("ACTION:", action)
    print("HOLD CALLS:", mock_hold.call_count)
    if mock_hold.call_count:
        print("HOLD ARGS:", mock_hold.call_args.args)
        print("HOLD KWARGS:", mock_hold.call_args.kwargs)
    print("SYNTH CALLS:", mock_synth.call_count)
    if mock_synth.call_count:
        print("SYNTH KWARGS:", mock_synth.call_args.kwargs)
