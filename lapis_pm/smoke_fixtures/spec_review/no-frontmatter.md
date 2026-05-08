# Spec Without Frontmatter

This spec document intentionally omits the required **Target ID:** and **Repo:** fields.
It is used to test that lapis-pm spec-review exits with code 2 and prints
SpecFrontmatterError to stderr.

## Goal

Trigger frontmatter parse failure.
