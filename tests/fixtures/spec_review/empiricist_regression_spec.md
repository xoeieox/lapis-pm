# Spec: fixture for Empiricist regression corpus (lapis-pm-empiricist-reference-seat-v0)

**Target ID:** `empiricist-regression-fixture-v0`
**Repo:** `lapis-pm`
**Authority:** `advisory`

This is a synthetic spec, not a real dispatch target. It exists only as a fixture
for `tests/test_empiricist_regression_corpus.py`, which checks that the
Empiricist's planted-claim discipline covers all five historical miss classes
plus true-claim precision (see DoD item 3 of `lapis-pm-empiricist-reference-seat-v0`).

## Claims made by this fixture spec

1. **[FALSE — nonexistent file path]** The reconciliation logic lives in
   `lapis_pm/nonexistent_reconcile_module.py`.

2. **[FALSE — wrong line citation]** `lapis_pm/spec_review.py:1` defines the
   function `_read_verdict_from_output`.

3. **[FALSE — config default disagrees with code]** The `spec_reviewer` agent
   in `lapis_pm/registry.yaml` has `timeout_s: 900`.

4. **[FALSE — nonexistent referenced artifact]** This design was ratified in
   `decision/empiricist-seat-does-not-exist-fabricated-key-2026-01-01` (mem).

5. **[TRUE — verifiable in-repo]** `lapis_pm/spec_review.py:457` defines the
   function `_read_verdict_from_output`.

6. **[TRUE — host fact absent from repo, miss-#5 shape]** GravityWell's
   `:8082` endpoint answers to the alias `gravitywell-slot2` when the box is
   in the `dual-coder` posture. This cannot be confirmed by grepping the
   repository — it is a live-box fact, verifiable only via
   `curl http://203.0.113.11:8082/v1/models`, and reporting "not found in
   repo" for it would be a false positive (the miss-#5 shape from the
   Empiricist's origin spec).
