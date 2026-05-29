#!/usr/bin/env bash
# smoke_fixtures/spec_review/facets_dispatch_smoke.sh
#
# Exercises the actual facets-dispatch subprocess in a real spec-review run.
# Asserts:
#   - subprocess returncode 0
#   - deliberation envelope context dict contains spec_text key
#   - personas_invoked == ["technical-integrity", "trickster"] (mirror-rep absent)
#   - synthesis.parse_failed == False
#   - each persona stance's claim is non-empty
#
# Gate: set SMOKE_NO_LLM=1 to skip (LLM call disabled in CI without model access).
# Default (SMOKE_NO_LLM unset or 0) runs the full dispatch path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAPIS_PM_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

red()   { echo -e "\033[0;31m$*\033[0m" >&2; exit 1; }
green() { echo -e "\033[0;32m$*\033[0m"; }

if [ "${SMOKE_NO_LLM:-0}" = "1" ]; then
    echo "SMOKE_NO_LLM=1: skipping facets dispatch smoke (LLM calls disabled)"
    exit 0
fi

FIXTURE="${LAPIS_PM_DIR}/smoke_fixtures/spec_review/valid-fixture.md"
DELIBDIR="/srv/lapis/facets/deliberations"

# Run spec-review with real facets dispatch (no --no-facets, no FACETS_DISPATCH_DISABLED)
echo "[facets-dispatch-smoke] running spec-review with real facets dispatch..."
REVIEW_OUT="$(
    COUNCIL_ENGINE_STUB=1 COUNCIL_STUB_POSITIONS=agree,agree \
    timeout 660 python3 -m lapis_pm.cli spec-review \
    --timeout 600 \
    "${FIXTURE}" 2>&1
)" || red "[facets-dispatch-smoke] spec-review exited non-zero: $REVIEW_OUT"

# Extract deliberation_id from the [spec-review:facets-complete] log line
DELIB_ID="$(echo "$REVIEW_OUT" | grep -oP '(?<=deliberation_id=)[^\s]+' | head -1 || true)"
if [ -z "$DELIB_ID" ]; then
    red "[facets-dispatch-smoke] could not extract deliberation_id from output:\n$REVIEW_OUT"
fi
echo "[facets-dispatch-smoke] deliberation_id=$DELIB_ID"

ENVELOPE_PATH="${DELIBDIR}/${DELIB_ID}.json"
if [ ! -f "$ENVELOPE_PATH" ]; then
    red "[facets-dispatch-smoke] envelope not found at $ENVELOPE_PATH"
fi

# Assert envelope structure using python3
python3 - "$ENVELOPE_PATH" <<'PYEOF'
import json, sys
path = sys.argv[1]
envelope = json.loads(open(path).read())

# 1. context contains spec_text
ctx = envelope.get("context", {})
assert "spec_text" in ctx, f"context missing spec_text key; context keys: {list(ctx.keys())}"

# 2. personas_invoked == ["technical-integrity", "trickster"] (mirror-rep absent)
invoked = envelope.get("personas_invoked", [])
assert invoked == ["technical-integrity", "trickster"], (
    f"personas_invoked mismatch: expected ['technical-integrity', 'trickster'], got {invoked}"
)
assert "mirror-rep" not in invoked, "mirror-rep must not be in personas_invoked"

# 3. synthesis.parse_failed == False
synthesis = envelope.get("synthesis", {})
assert synthesis.get("parse_failed") is False, (
    f"synthesis.parse_failed is not False: {synthesis.get('parse_failed')!r}"
)

# 4. each persona stance's claim is non-empty
stances = envelope.get("stances", [])
assert len(stances) > 0, "no stances in envelope"
for stance in stances:
    claim = stance.get("claim", "")
    assert claim and claim.strip(), (
        f"persona {stance.get('persona')!r} has empty claim (got: {claim!r}); "
        "this indicates 'permission denied' or LLM non-engagement"
    )

print("[facets-dispatch-smoke] all assertions passed")
PYEOF

green "[facets-dispatch-smoke] facets dispatch smoke OK (deliberation_id=$DELIB_ID)"
