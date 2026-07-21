"""Unit tests for qwen-image-edit's non-model request handling: mask decoding,
outpaint canvas/mask construction, and the pure-Pillow /v1/transform crop math.
These are constraint tests on invariants the endpoints must never violate
(mask semantics, canvas bounds, crop bounds) rather than confirmations of
current behavior.
"""

import base64
import importlib
import io
import sys
import types
import unittest

import numpy as np
from PIL import Image

torch_stub = types.ModuleType("torch")
torch_stub.bfloat16 = object()


class _FakeGenerator:
    def __init__(self, device=None):
        pass

    def manual_seed(self, seed):
        return self


class _FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


torch_stub.Generator = _FakeGenerator
torch_stub.inference_mode = _FakeInferenceMode
torch_stub.cuda = types.SimpleNamespace(
    memory_allocated=lambda: 0, memory_reserved=lambda: 0
)
sys.modules.setdefault("torch", torch_stub)

diffusers_stub = types.ModuleType("diffusers")
for name in (
    "AutoModel",
    "DiffusionPipeline",
    "QwenImageEditInpaintPipeline",
    "QwenImageEditPlusPipeline",
    "TorchAoConfig",
):
    setattr(diffusers_stub, name, type(name, (), {}))
sys.modules.setdefault("diffusers", diffusers_stub)

torchao_stub = types.ModuleType("torchao")
torchao_quantization_stub = types.ModuleType("torchao.quantization")
torchao_quantization_stub.Float8WeightOnlyConfig = type(
    "Float8WeightOnlyConfig", (), {}
)
torchao_stub.quantization = torchao_quantization_stub
sys.modules.setdefault("torchao", torchao_stub)
sys.modules.setdefault("torchao.quantization", torchao_quantization_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.Qwen2_5_VLForConditionalGeneration = type(
    "Qwen2_5_VLForConditionalGeneration", (), {}
)
sys.modules.setdefault("transformers", transformers_stub)

prometheus_stub = types.ModuleType("prometheus_client")


class _FakeHistogram:
    def __init__(self, *args, **kwargs):
        pass

    def time(self):
        return _FakeInferenceMode()


class _FakeGauge:
    def __init__(self, *args, **kwargs):
        pass

    def set(self, value):
        pass


prometheus_stub.Histogram = _FakeHistogram
prometheus_stub.Gauge = _FakeGauge
prometheus_stub.start_http_server = lambda *args, **kwargs: None
sys.modules.setdefault("prometheus_client", prometheus_stub)

try:
    import fastapi  # noqa: F401
except ImportError:
    fastapi_stub = types.ModuleType("fastapi")

    class _FakeHTTPException(Exception):
        def __init__(self, status_code, detail=None):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class _FakeFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def _noop_decorator(self, *args, **kwargs):
            def wrap(fn):
                return fn

            return wrap

        get = _noop_decorator
        post = _noop_decorator

        def on_event(self, *args, **kwargs):
            def wrap(fn):
                return fn

            return wrap

    fastapi_stub.FastAPI = _FakeFastAPI
    fastapi_stub.HTTPException = _FakeHTTPException
    fastapi_stub.File = lambda *args, **kwargs: None
    fastapi_stub.Form = lambda *args, **kwargs: None
    fastapi_stub.UploadFile = type("UploadFile", (), {})
    sys.modules.setdefault("fastapi", fastapi_stub)

api = importlib.import_module("api")


def _make_image(width, height, color=(255, 0, 0)):
    return Image.new("RGB", (width, height), color)


def _mask_data_uri(mask_array):
    encoded = io.BytesIO()
    Image.fromarray(mask_array.astype(np.uint8) * 255).save(encoded, format="PNG")
    payload = base64.b64encode(encoded.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


class MaskDecodingTests(unittest.TestCase):
    def test_decodes_sam3_style_data_uri_losslessly(self):
        mask = np.array([[False, True], [True, False]], dtype=bool)
        data_uri = _mask_data_uri(mask)

        decoded = api._decode_mask(data_uri)

        self.assertEqual(decoded.mode, "L")
        np.testing.assert_array_equal(np.asarray(decoded), mask.astype(np.uint8) * 255)

    def test_rejects_non_data_uri_string(self):
        with self.assertRaises(api.HTTPException):
            api._decode_mask("not-a-data-uri")

    def test_rejects_invalid_base64_payload(self):
        with self.assertRaises(api.HTTPException):
            api._decode_mask("data:image/png;base64,not-valid-base64!!!")


class OutpaintCanvasTests(unittest.TestCase):
    def test_center_anchor_preserves_original_pixels_untouched_by_mask(self):
        source = _make_image(10, 10, color=(1, 2, 3))

        canvas, mask = api._outpaint_canvas(source, 20, 20, "center")

        left, top = 5, 5
        self.assertEqual(
            canvas.crop((left, top, left + 10, top + 10)).tobytes(), source.tobytes()
        )
        mask_array = np.asarray(mask)
        self.assertTrue(np.all(mask_array[top : top + 10, left : left + 10] == 0))

    def test_mask_marks_every_non_original_pixel_for_repaint(self):
        source = _make_image(4, 4)

        canvas, mask = api._outpaint_canvas(source, 10, 8, "top-left")

        mask_array = np.asarray(mask)
        expected = np.full((8, 10), 255, dtype=np.uint8)
        expected[0:4, 0:4] = 0
        np.testing.assert_array_equal(mask_array, expected)

    def test_rejects_target_smaller_than_source(self):
        source = _make_image(10, 10)

        with self.assertRaises(api.HTTPException):
            api._outpaint_canvas(source, 5, 20, "center")

    def test_rejects_unknown_anchor(self):
        source = _make_image(10, 10)

        with self.assertRaises(api.HTTPException):
            api._outpaint_canvas(source, 20, 20, "diagonal")


class ClampStepsTests(unittest.TestCase):
    def test_never_returns_a_value_outside_one_to_one_hundred(self):
        self.assertEqual(api._clamp_steps(-5), 1)
        self.assertEqual(api._clamp_steps(0), 1)
        self.assertEqual(api._clamp_steps(500), 100)
        self.assertEqual(api._clamp_steps(50), 50)


class InvokeProgressTests(unittest.TestCase):
    def setUp(self):
        api._invoke_progress.clear()

    def tearDown(self):
        api._invoke_progress.clear()

    def test_step_callback_progress_is_monotonically_increasing_within_a_request(self):
        callback = api._make_step_callback("req-1", total_steps=10)
        seen_steps = []
        for step in range(10):
            callback(None, step, None, {})
            seen_steps.append(api._invoke_progress["req-1"]["step"])

        self.assertEqual(seen_steps, sorted(seen_steps))
        self.assertEqual(seen_steps, list(range(1, 11)))

    def test_invoke_progress_endpoint_returns_404_once_request_completes(self):
        callback = api._make_step_callback("req-2", total_steps=5)
        callback(None, 0, None, {})
        self.assertEqual(api.invoke_progress("req-2"), {"step": 1, "total": 5})

        api._invoke_progress.pop("req-2", None)

        with self.assertRaises(api.HTTPException):
            api.invoke_progress("req-2")

    def test_invoke_progress_endpoint_raises_for_unknown_request_id(self):
        with self.assertRaises(api.HTTPException):
            api.invoke_progress("never-seen")


if __name__ == "__main__":
    unittest.main()
