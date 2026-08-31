"""DEPRECATED - content moved to tests/test_registry_fixer_seat.py.

PR for lapis-pm-fixer-seat-gpu0-default-v0 (2026-08-31): the fixer pair's
durable seat is the GPU0 default (the 2026-08-29 stopgap shape became the
durable default; the berth is dormant until on-demand overflow). The pin
moved to test_registry_fixer_seat.py, re-framed for the overflow posture.

This file is an intentional deprecation stub: the local-fixer toolset has
no file-delete primitive (write_file/apply_edit/run_tests only,
agents_core/gw_agent.py DEFAULT_FIXER_TOOLS), so the rename is a content
move, not a delete. It carries NO tests on purpose - pytest collects zero
items from here, and the positive-only test gate keys on FAILED/ERROR node
IDs, which this file cannot produce. Delete this file in a follow-up that
has shell access.
"""
