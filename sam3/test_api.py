"""Unit tests for SAM3 worker mask attachment serialization."""

import asyncio
import base64
import importlib
import io
import json
import sys
import types
import unittest
from unittest import mock

import numpy as np
from fastapi import Response
from PIL import Image


managers = types.ModuleType("managers")
annotation_manager = types.ModuleType("managers.annotation_manager")
annotation_manager.SAM3Annotator = mock.Mock
managers.annotation_manager = annotation_manager
sys.modules.setdefault("managers", managers)
sys.modules.setdefault("managers.annotation_manager", annotation_manager)
api = importlib.import_module("sam3.api")


class MaskSerializationTests(unittest.TestCase):
    def test_mask_attachment_round_trips_every_pixel(self):
        mask = np.array([[False, True, False], [True, True, False]], dtype=bool)
        attachment = api._mask_attachment(mask, "mask:0")
        decoded = np.asarray(
            Image.open(io.BytesIO(base64.b64decode(attachment["data_base64"])))
        )
        self.assertEqual(attachment["name"], "mask:0")
        self.assertEqual(attachment["media_type"], "image/png")
        np.testing.assert_array_equal(decoded, mask.astype(np.uint8) * 255)

    def test_capabilities_report_selected_runtime(self):
        capabilities = api.capabilities()
        self.assertEqual(capabilities["runtime"]["platform"], "gb10")
        self.assertEqual(capabilities["runtime"]["device"], "cuda")
        self.assertEqual(capabilities["runtime"]["precision"], "bf16-weight")
        self.assertEqual(capabilities["runtime"]["resolution"], 1008)


class ReadyEndpointTests(unittest.TestCase):
    def setUp(self):
        self.model_patch = mock.patch.object(api.annotator, "model", object())
        self.model_patch.start()
        self.error_patch = mock.patch.object(api, "last_inference_error", None)
        self.error_patch.start()

    def tearDown(self):
        mock.patch.stopall()

    def test_reports_loading_before_the_model_is_loaded(self):
        with mock.patch.object(api.annotator, "model", None):
            response = Response()
            body = api.ready(response)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(body["status"], "loading")
        self.assertFalse(body["model_loaded"])
        self.assertNotIn("last_inference_error", body)

    def test_reports_ready_once_loaded_with_no_inference_errors(self):
        response = Response()
        body = api.ready(response)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "ready")
        self.assertTrue(body["model_loaded"])

    def test_surfaces_the_last_inference_error_instead_of_reporting_ready(self):
        with mock.patch.object(
            api,
            "last_inference_error",
            {
                "type": "RuntimeError",
                "message": "level_zero backend failed with error: 39 "
                "(UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)",
                "at": 1234.5,
            },
        ):
            response = Response()
            body = api.ready(response)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(body["status"], "error")
        self.assertTrue(body["model_loaded"])
        self.assertIn(
            "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
            body["last_inference_error"]["message"],
        )


class InvokeErrorTrackingTests(unittest.TestCase):
    def setUp(self):
        self.error_patch = mock.patch.object(api, "last_inference_error", None)
        self.error_patch.start()

    def tearDown(self):
        mock.patch.stopall()

    def _run(self, coro):
        return asyncio.run(coro)

    def test_records_error_type_and_message_when_segment_raises(self):
        manifest = {
            "protocol_version": api.PROTOCOL_VERSION,
            "request_id": "test-1",
            "operation": "segment",
            "parameters": {"prompts": ["a shape"], "threshold": 0.15},
            "attachments": [{"name": "image"}],
        }
        image_bytes = io.BytesIO()
        Image.new("RGB", (4, 4)).save(image_bytes, format="PNG")

        upload = mock.Mock()
        upload.read = mock.AsyncMock(return_value=image_bytes.getvalue())

        with mock.patch.object(
            api,
            "_segment",
            side_effect=RuntimeError(
                "level_zero backend failed with error: 39 "
                "(UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)"
            ),
        ):
            with self.assertRaises(RuntimeError):
                self._run(
                    api.invoke(manifest=json.dumps(manifest), attachments=[upload])
                )

        self.assertIsNotNone(api.last_inference_error)
        self.assertEqual(api.last_inference_error["type"], "RuntimeError")
        self.assertIn("OUT_OF_DEVICE_MEMORY", api.last_inference_error["message"])

    def test_clears_the_prior_error_on_a_successful_inference(self):
        api.last_inference_error = {"type": "RuntimeError", "message": "stale", "at": 0}
        manifest = {
            "protocol_version": api.PROTOCOL_VERSION,
            "request_id": "test-2",
            "operation": "segment",
            "parameters": {"prompts": ["a shape"], "threshold": 0.15},
            "attachments": [{"name": "image"}],
        }
        image_bytes = io.BytesIO()
        Image.new("RGB", (4, 4)).save(image_bytes, format="PNG")

        upload = mock.Mock()
        upload.read = mock.AsyncMock(return_value=image_bytes.getvalue())

        with mock.patch.object(api, "_segment", return_value=([], [])):
            self._run(api.invoke(manifest=json.dumps(manifest), attachments=[upload]))

        self.assertIsNone(api.last_inference_error)


if __name__ == "__main__":
    unittest.main()
