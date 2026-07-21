"""Invariant tests for image-api's app.py request handling, focused on the
caption+concepts vision response contract. These are constraint tests: they
assert what must always be true of a malformed/well-formed vision response,
not just what the current implementation happens to return.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import app


def _chat_response(payload: dict) -> AsyncMock:
    response = AsyncMock()
    response.raise_for_status = lambda: None
    response.json = lambda: {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    return response


class ConceptsCaptionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._token_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".token", delete=False
        )
        self._token_file.write("test-token")
        self._token_file.close()
        self._token_patch = patch.object(
            app, "POLICY_TOKEN_FILE", self._token_file.name
        )
        self._token_patch.start()

    def tearDown(self):
        self._token_patch.stop()
        Path(self._token_file.name).unlink(missing_ok=True)

    async def test_returns_both_caption_and_concepts_from_one_call(self):
        payload = {
            "caption": "A red bicycle leaning against a brick wall.",
            "sam3_prompts": ["red bicycle", "brick wall"],
        }
        with patch("app.httpx.AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=_chat_response(payload)
            )
            result = await app.concepts(b"fake-image-bytes", "image/png")

        self.assertEqual(result, {
            "caption": "A red bicycle leaning against a brick wall.",
            "concepts": ["red bicycle", "brick wall"],
        })

    async def test_concepts_raises_if_caption_missing_from_model_response(self):
        payload = {"sam3_prompts": ["red bicycle"]}
        with patch("app.httpx.AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=_chat_response(payload)
            )
            with self.assertRaises(app.HTTPException):
                await app.concepts(b"fake-image-bytes", "image/png")

    async def test_concepts_raises_if_sam3_prompts_missing_from_model_response(self):
        payload = {"caption": "A red bicycle leaning against a brick wall."}
        with patch("app.httpx.AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=_chat_response(payload)
            )
            with self.assertRaises(app.HTTPException):
                await app.concepts(b"fake-image-bytes", "image/png")

    async def test_concepts_raises_if_caption_is_blank(self):
        payload = {"caption": "   ", "sam3_prompts": ["red bicycle"]}
        with patch("app.httpx.AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=_chat_response(payload)
            )
            with self.assertRaises(app.HTTPException):
                await app.concepts(b"fake-image-bytes", "image/png")


if __name__ == "__main__":
    unittest.main()
