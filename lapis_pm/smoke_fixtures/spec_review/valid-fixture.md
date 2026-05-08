# Spec: Smoke Fixture for Spec Review

**Target ID:** `spec-review-smoke-fixture`
**Repo:** `lapis-pm`
**Branch:** `lapis/spec-review-smoke-fixture/implement`
**Authority:** `advisory`
**Bind trigger:** auto

## Goal

Fixture spec for smoke-testing `lapis-pm spec-review`. Not a real target.

## Scope

**In scope:**
- Validate that spec-review parses this frontmatter correctly.
- Validate that the combined brief is rendered to stdout.

**Explicitly out of scope:**
- Actual implementation work.

## Architecture

No architecture. This is a smoke fixture.

## Tests

None.

## Invariants

1. This file is read-only. Never modify it without updating smoke phases 39-42.

## Definition of Done

- Smoke phases 39-42 pass.
