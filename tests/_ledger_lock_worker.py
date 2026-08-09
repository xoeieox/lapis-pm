"""Real-OS-process worker for the ledger-lock cross-process tests
(deploy-inventory-repair-ledger-lock-v0, DoD-1/2/5).

Invoked as a subprocess (`python3 tests/_ledger_lock_worker.py <args>`), never
imported by pytest directly - it needs to be a genuine separate process, not
a thread, per DoD-1 ("must be tested with genuine OS processes... not two
threads and not a mocked lock").

Args: <mode> <ledger_path> <lock_path> <clone_path> <finding_kind> <sleep_s> [<timeout_s>]

mode == "repair": runs run_repair_pass with one fresh HIGH finding, a mocked
    diagnose_finding that sleeps `sleep_s` (simulating the GW diagnosis call)
    before returning, and mocked deposit_gem/emit_provenance/reconcile/escalate
    so no network is touched. Prints the actions list as JSON.
mode == "corrupt": runs propose_corrupt_status once (mocked deposit_gem),
    printing the returned action as JSON.
"""
import json
import site
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# agents-core is an editable install whose .pth file lives in the user
# site-packages dir; a plain sys.path entry does not get .pth files
# processed (only `site` does that) - mirrors tests/conftest.py's own
# addsitedir call, needed here too since this runs as its own interpreter.
site.addsitedir('/home/user/.local/lib/python3.12/site-packages')
_ARCHETYPES_CORE_PATH = '/srv/git/archetypes-core'
if _ARCHETYPES_CORE_PATH not in sys.path:
    sys.path.insert(0, _ARCHETYPES_CORE_PATH)

from lapis_pm import deploy_inventory_repair as dir_mod  # noqa: E402


def main():
    mode = sys.argv[1]
    ledger_path = Path(sys.argv[2])
    lock_path = Path(sys.argv[3])

    dir_mod._LEDGER_FILE = ledger_path
    dir_mod._LEDGER_LOCK_FILE = lock_path

    if len(sys.argv) > 7 and sys.argv[7]:
        dir_mod.DEPLOY_INVENTORY_LEDGER_LOCK_TIMEOUT_SECS = float(sys.argv[7])

    if mode == "repair":
        clone_path = sys.argv[4]
        finding_kind = sys.argv[5]
        sleep_s = float(sys.argv[6])

        def _slow_diagnose(*a, **kw):
            import time
            time.sleep(sleep_s)
            finding = kw.get("finding") if "finding" in kw else a[1]
            return dir_mod._degrade(finding), "mechanical"

        status = {
            "clones": [{
                "path": clone_path, "mapped": True, "branch": "main",
                "commits_behind": 1, "backing_units": [],
                "findings": [{"kind": finding_kind, "severity": "HIGH", "detail": "x"}],
            }],
        }
        with (
            patch.object(dir_mod, "diagnose_finding", side_effect=_slow_diagnose),
            patch.object(dir_mod, "deposit_gem", return_value=f"gem-{finding_kind}"),
            patch.object(dir_mod, "emit_provenance"),
            patch.object(dir_mod, "reconcile_terminal_gems"),
            patch.object(dir_mod, "escalate_stale_gems"),
        ):
            actions = dir_mod.run_repair_pass(status, prior_high_keys=set(), corrupt=False)
        print(json.dumps(actions))
    elif mode == "corrupt":
        with patch.object(dir_mod, "deposit_gem", return_value="gem-corrupt"):
            action = dir_mod.propose_corrupt_status()
        print(json.dumps(action))
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
