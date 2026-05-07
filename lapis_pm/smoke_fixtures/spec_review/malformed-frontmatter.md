# Spec With Malformed Frontmatter

**Target ID:** `has whitespace inside`
**Repo:** `lapis-pm`

This spec has a malformed Target ID (contains spaces, which the regex rejects).
Used to test that lapis-pm spec-review exits with code 2 on malformed values.
