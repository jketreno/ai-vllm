"""Invariant tests for image-api's app.py request handling, focused on the
caption+concepts vision response contract. These are constraint tests: they
assert what must always be true of a malformed/well-formed vision response,
not just what the current implementation happens to return.
"""

import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import app
import resource_lease


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


class ResourceLeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_lease_is_released_when_inference_fails(self):
        response = unittest.mock.Mock(status_code=200)
        response.json.return_value = {"lease_id": "lease-1"}
        response.raise_for_status.return_value = None
        with patch.object(resource_lease, "_token", return_value="token"), patch.object(
            resource_lease.httpx, "AsyncClient"
        ) as client_class:
            client = client_class.return_value.__aenter__.return_value
            client.post = AsyncMock(return_value=response)
            client.delete = AsyncMock()

            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                async with resource_lease.image_edit_lease("request-1"):
                    raise RuntimeError("inference failed")

        client.delete.assert_awaited_once()
        self.assertTrue(client.delete.call_args.args[0].endswith("/lease-1"))

    async def test_edit_invocation_holds_resource_lease_and_preserves_request_id(self):
        events = []

        @asynccontextmanager
        async def lease(request_id):
            events.append(("acquire", request_id))
            try:
                yield
            finally:
                events.append(("release", request_id))

        rpc_result = {
            "protocol_version": "1", "status": "ok",
            "data": {"width": 1, "height": 1},
            "attachments": [{
                "name": "image", "media_type": "image/png",
                "data_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            }],
        }
        image = app.Image.new("RGB", (1, 1))
        with patch.object(app, "image_edit_lease", lease), patch.object(
            app.editor, "invoke", new_callable=AsyncMock, return_value=rpc_result
        ) as invoke:
            await app.invoke_edit(
                "inpaint", image, "image/png", {}, request_id="request-1"
            )

        self.assertEqual(events, [("acquire", "request-1"), ("release", "request-1")])
        self.assertEqual(invoke.call_args.kwargs["request_id"], "request-1")

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
