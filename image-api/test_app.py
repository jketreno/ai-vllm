"""Invariant tests for image-api's app.py request handling, focused on the
caption+concepts vision response contract. These are constraint tests: they
assert what must always be true of a malformed/well-formed vision response,
not just what the current implementation happens to return.
"""

import importlib
import importlib.util
import base64
import io
import json
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
spec = importlib.util.spec_from_file_location("image_api_app", MODULE_DIR / "app.py")
app = importlib.util.module_from_spec(spec)
sys.modules["image_api_app"] = app
spec.loader.exec_module(app)
resource_lease = importlib.import_module("resource_lease")


def _mask_data_uri(size=(4, 4), fill=255):
    buffer = io.BytesIO()
    Image.new("L", size, fill).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


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
        with patch.object(app.httpx, "AsyncClient") as client_cls:
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
        with patch.object(app.httpx, "AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=_chat_response(payload)
            )
            with self.assertRaises(app.HTTPException):
                await app.concepts(b"fake-image-bytes", "image/png")


class CapabilityReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_capabilities_identify_the_unavailable_service(self):
        with (
            patch.object(app.sam, "ready", new_callable=AsyncMock, return_value=True),
            patch.object(
                app.editor, "ready", new_callable=AsyncMock, return_value=False
            ),
            patch.object(
                app, "policy_ready", new_callable=AsyncMock, return_value=True
            ),
        ):
            result = await app.capabilities()

        self.assertEqual(result["capabilities"]["inpaint"], "unavailable")
        self.assertEqual(
            result["services"]["qwen-image-edit-worker"],
            {
                "name": "Qwen image edit worker",
                "status": "unavailable",
                "capabilities": ["edit", "inpaint", "outpaint"],
            },
        )

    async def test_ready_reports_dependencies_even_while_degraded(self):
        with (
            patch.object(app.sam, "ready", new_callable=AsyncMock, return_value=False),
            patch.object(
                app.editor, "ready", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                app, "policy_ready", new_callable=AsyncMock, return_value=True
            ),
        ):
            result = await app.ready()

        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["services"]["sam3-worker"]["status"], "unavailable")


class MaskValidationTests(unittest.TestCase):
    def test_mask_dimensions_must_match_source(self):
        with self.assertRaisesRegex(app.HTTPException, "dimensions"):
            app.decode_mask(_mask_data_uri((3, 4)), expected_size=(4, 4))

    def test_mask_must_contain_an_editable_pixel(self):
        with self.assertRaisesRegex(app.HTTPException, "editable pixel"):
            app.decode_mask(_mask_data_uri(fill=0), expected_size=(4, 4))

    def test_valid_mask_is_normalized_to_png(self):
        decoded = app.decode_mask(_mask_data_uri(), expected_size=(4, 4))
        self.assertEqual(Image.open(io.BytesIO(decoded)).mode, "L")


class ResourceLeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_lease_is_released_when_inference_fails(self):
        response = unittest.mock.Mock(status_code=200)
        response.json.return_value = {"lease_id": "lease-1"}
        response.raise_for_status.return_value = None
        with patch.object(resource_lease, "EXCLUSIVE_VLLM", True), patch.object(
            resource_lease, "_token", return_value="token"
        ), patch.object(resource_lease.httpx, "AsyncClient") as client_class:
            client = client_class.return_value.__aenter__.return_value
            client.post = AsyncMock(return_value=response)
            client.delete = AsyncMock()

            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                async with resource_lease.image_edit_lease("request-1"):
                    raise RuntimeError("inference failed")

        client.delete.assert_awaited_once()
        self.assertTrue(client.delete.call_args.args[0].endswith("/lease-1"))

    async def test_lease_is_a_no_op_when_exclusive_vllm_disabled(self):
        with patch.object(resource_lease, "EXCLUSIVE_VLLM", False), patch.object(
            resource_lease.httpx, "AsyncClient"
        ) as client_class:
            async with resource_lease.image_edit_lease("request-1"):
                pass

        client_class.assert_not_called()

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
                "data_base64": (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
                    "QVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                ),
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

    async def test_invoke_edit_surfaces_effective_generation_params_when_overridden(
        self,
    ):
        """The worker silently overrides steps/CFG under its lightning profile
        (see qwen-image-edit/api.py _generation_settings) and reports what it
        actually used in metadata.effective_*. Callers need that surfaced here --
        otherwise a caller who requested true_cfg_scale=6.5 has no way to learn
        the worker actually ran with 1.0, and ends up recording a value that was
        never applied."""

        @asynccontextmanager
        async def lease(request_id):
            yield

        rpc_result = {
            "protocol_version": "1", "status": "ok",
            "data": {"width": 1, "height": 1},
            "attachments": [{
                "name": "image", "media_type": "image/png",
                "data_base64": (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
                    "QVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                ),
            }],
            "metadata": {
                "duration_seconds": 0.1,
                "effective_num_inference_steps": 4,
                "effective_true_cfg_scale": 1.0,
            },
        }
        image = app.Image.new("RGB", (1, 1))
        with patch.object(app, "image_edit_lease", lease), patch.object(
            app.editor, "invoke", new_callable=AsyncMock, return_value=rpc_result
        ):
            response = await app.invoke_edit(
                "inpaint", image, "image/png", {"true_cfg_scale": 6.5}
            )

        self.assertEqual(response["effective_num_inference_steps"], 4)
        self.assertEqual(response["effective_true_cfg_scale"], 1.0)

    async def test_invoke_edit_omits_effective_params_when_worker_doesnt_report_them(
        self,
    ):
        """Older/other workers may not send metadata.effective_* at all -- must not
        KeyError, and must not fabricate values the worker never reported."""

        @asynccontextmanager
        async def lease(request_id):
            yield

        rpc_result = {
            "protocol_version": "1", "status": "ok",
            "data": {"width": 1, "height": 1},
            "attachments": [{
                "name": "image", "media_type": "image/png",
                "data_base64": (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
                    "QVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                ),
            }],
            "metadata": {"duration_seconds": 0.1},
        }
        image = app.Image.new("RGB", (1, 1))
        with patch.object(app, "image_edit_lease", lease), patch.object(
            app.editor, "invoke", new_callable=AsyncMock, return_value=rpc_result
        ):
            response = await app.invoke_edit("edit", image, "image/png", {})

        self.assertNotIn("effective_num_inference_steps", response)
        self.assertNotIn("effective_true_cfg_scale", response)

    async def test_concepts_raises_if_sam3_prompts_missing_from_model_response(self):
        payload = {"caption": "A red bicycle leaning against a brick wall."}
        with patch.object(app.httpx, "AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=_chat_response(payload)
            )
            with self.assertRaises(app.HTTPException):
                await app.concepts(b"fake-image-bytes", "image/png")

    async def test_concepts_raises_if_caption_is_blank(self):
        payload = {"caption": "   ", "sam3_prompts": ["red bicycle"]}
        with patch.object(app.httpx, "AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=_chat_response(payload)
            )
            with self.assertRaises(app.HTTPException):
                await app.concepts(b"fake-image-bytes", "image/png")


if __name__ == "__main__":
    unittest.main()
