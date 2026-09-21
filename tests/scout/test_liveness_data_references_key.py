"""Tests for _data_references_key (lapis_pm/scout/liveness.py).

cr-bundle-lapis-pm-2026-09-21 item 4fd83ffdba: the ref-spend classifier must
only fire on STRUCTURED fields (covers / derived_from) with exact element
equality. The old substring match against free-text fields (description,
tags, spec_from) let a short key like "auth" or "v0" false-positive against
prose and permanently park scaffolds whose underlying work was never
implemented.
"""
from __future__ import annotations

from lapis_pm.scout.liveness import _data_references_key


class TestDataReferencesKeyExactMatch:
    def test_exact_element_in_covers(self):
        data = {"covers": ["auth", "other"]}
        assert _data_references_key(data, "auth") is True

    def test_exact_element_in_derived_from(self):
        data = {"derived_from": ["decision:auth"]}
        assert _data_references_key(data, "auth") is True

    def test_canonical_form_in_covers(self):
        """A ref written in full canonical form (decision:<key>) is an
        unambiguous reference and must match."""
        data = {"covers": ["decision:auth"]}
        assert _data_references_key(data, "auth") is True

    def test_substring_prose_in_description_does_not_match(self):
        """The core regression: 'auth' inside description prose must NOT
        classify the ref as spent."""
        data = {"description": "this scaffold covers the auth flow redesign"}
        assert _data_references_key(data, "auth") is False

    def test_substring_in_tags_does_not_match(self):
        data = {"tags": ["lapis-pm", "auth-related"]}
        assert _data_references_key(data, "auth") is False

    def test_substring_in_spec_from_does_not_match(self):
        data = {"spec_from": "/srv/lapis/planning/specs/auth-redesign-v0.md"}
        assert _data_references_key(data, "auth") is False

    def test_substring_element_in_covers_does_not_match(self):
        """Exact-element equality: 'authentication' in covers is not 'auth'."""
        data = {"covers": ["authentication"]}
        assert _data_references_key(data, "auth") is False

    def test_short_key_v0_in_prose_does_not_match(self):
        data = {"description": "v0 of the pipeline; v1 planned"}
        assert _data_references_key(data, "v0") is False

    def test_empty_and_missing_fields(self):
        assert _data_references_key({}, "auth") is False
        assert _data_references_key({"covers": [], "derived_from": []}, "auth") is False
        assert _data_references_key({"covers": None}, "auth") is False

    def test_non_string_list_items_ignored(self):
        data = {"covers": [42, None, {"key": "auth"}]}
        assert _data_references_key(data, "auth") is False
