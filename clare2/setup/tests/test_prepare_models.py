"""Tests for model revision resolution."""

from __future__ import annotations

import unittest

from prepare_models import revision


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str]] = []

    def model_info(self, model: str, revision: str | None, token: str):
        self.calls.append((model, revision, token))
        return type("ModelInfo", (), {"sha": "resolved-sha"})()


class RevisionTests(unittest.TestCase):
    def test_placeholder_resolves_default_revision(self) -> None:
        api = FakeApi()

        result = revision(api, "org/model", "REPLACE_WITH_REVISION", "token")

        self.assertEqual(result, "resolved-sha")
        self.assertEqual(api.calls, [("org/model", None, "token")])

    def test_explicit_revision_is_resolved_to_sha(self) -> None:
        api = FakeApi()

        result = revision(api, "org/model", "release-tag", "token")

        self.assertEqual(result, "resolved-sha")
        self.assertEqual(api.calls, [("org/model", "release-tag", "token")])

    def test_blank_revision_resolves_default_revision(self) -> None:
        api = FakeApi()

        revision(api, "org/model", "  ", "token")

        self.assertEqual(api.calls, [("org/model", None, "token")])


if __name__ == "__main__":
    unittest.main()
