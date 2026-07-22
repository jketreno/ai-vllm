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
from unittest import mock

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
    "FlowMatchEulerDiscreteScheduler",
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

    def labels(self, *args, **kwargs):
        return self

    def observe(self, value):
        pass


class _FakeGauge:
    def __init__(self, *args, **kwargs):
        pass

    def set(self, value):
        pass

    def labels(self, *args, **kwargs):
        return self

    def inc(self):
        pass


prometheus_stub.Histogram = _FakeHistogram
prometheus_stub.Gauge = _FakeGauge
prometheus_stub.Counter = _FakeGauge
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


class RpcResultTests(unittest.TestCase):
    def test_includes_pre_composite_image_when_worker_returns_one(self):
        final = base64.b64encode(b"final").decode("ascii")
        pre_composite = base64.b64encode(b"generated").decode("ascii")

        response = api._rpc_result(
            {"request_id": "request-id"},
            {
                "width": 4,
                "height": 4,
                "image_png_base64": final,
                "pre_composite_image_png_base64": pre_composite,
            },
            api.time.monotonic(),
        )

        attachments = {item["name"]: item for item in response["attachments"]}
        self.assertEqual(attachments["image"]["data_base64"], final)
        self.assertEqual(
            attachments["pre_composite_image"]["data_base64"], pre_composite
        )

    def test_includes_conditioning_image_when_worker_returns_one(self):
        final = base64.b64encode(b"final").decode("ascii")
        conditioning = base64.b64encode(b"conditioning").decode("ascii")

        response = api._rpc_result(
            {"request_id": "request-id"},
            {
                "width": 4,
                "height": 4,
                "image_png_base64": final,
                "conditioning_image_png_base64": conditioning,
            },
            api.time.monotonic(),
        )

        attachments = {item["name"]: item for item in response["attachments"]}
        self.assertEqual(
            attachments["conditioning_image"]["data_base64"], conditioning
        )


class InpaintCompositionTests(unittest.TestCase):
    def test_portrait_crop_keeps_image_and_mask_dimensions_aligned(self):
        source = _make_image(1204, 1599, color=(10, 20, 30))
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (500, 700, 620, 840))

        cropped_image, cropped_mask, box = api._inpaint_region(source, mask, 64)

        self.assertEqual(cropped_image.size, cropped_mask.size)
        self.assertEqual(box, (436, 636, 684, 904))

    def test_composite_preserves_every_unmasked_source_pixel(self):
        source = _make_image(80, 120, color=(10, 20, 30))
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (20, 30, 40, 50))
        cropped_image, cropped_mask, box = api._inpaint_region(source, mask, 8)
        generated = _make_image(1024, 1024, color=(200, 100, 50))

        result = api._composite_generated_region(source, mask, generated, box)

        result_pixels = np.asarray(result)
        source_pixels = np.asarray(source)
        mask_pixels = np.asarray(mask)
        np.testing.assert_array_equal(
            result_pixels[mask_pixels == 0], source_pixels[mask_pixels == 0]
        )
        self.assertEqual(result.size, source.size)

    def test_strength_blends_generated_pixels_inside_mask(self):
        source = _make_image(4, 4, color=(0, 0, 0))
        mask = Image.new("L", source.size, 255)
        generated = _make_image(4, 4, color=(200, 100, 50))

        result = api._composite_generated_region(
            source, mask, generated, (0, 0, 4, 4), strength=0.5
        )

        self.assertEqual(result.getpixel((0, 0)), (100, 50, 25))

    def test_visual_marker_surrounds_mask_without_changing_selected_pixels(self):
        source = _make_image(100, 100, color=(10, 20, 30))
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (30, 30, 70, 70))

        annotated = api._annotate_inpaint_region(source, mask)

        self.assertEqual(annotated.getpixel((50, 50)), source.getpixel((50, 50)))
        self.assertEqual(annotated.getpixel((28, 50)), api.MARKER_COLOR)
        self.assertEqual(annotated.getpixel((20, 50)), api.MARKER_HALO_COLOR)
        self.assertEqual(annotated.getpixel((0, 0)), source.getpixel((0, 0)))

    def test_marker_width_scales_and_is_bounded(self):
        self.assertEqual(api._visual_marker_width(_make_image(100, 100)), 6)
        self.assertEqual(api._visual_marker_width(_make_image(1000, 800)), 6)
        self.assertEqual(api._visual_marker_width(_make_image(4000, 4000)), 16)

    def test_empty_mask_uses_full_canvas_without_changing_dimensions(self):
        source = _make_image(1204, 1599)
        mask = Image.new("L", source.size, 0)

        cropped_image, cropped_mask, box = api._inpaint_region(source, mask, 64)

        self.assertEqual(cropped_image.size, source.size)
        self.assertEqual(cropped_mask.size, source.size)
        self.assertEqual(box, (0, 0, 1204, 1599))

    def test_landscape_edge_touching_mask_is_clamped_to_canvas(self):
        source = _make_image(1599, 1204)
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (0, 100, 20, 300))

        cropped_image, cropped_mask, box = api._inpaint_region(source, mask, 64)

        self.assertEqual(box, (0, 36, 84, 364))
        self.assertEqual(cropped_image.size, cropped_mask.size)

    def test_full_mask_and_multi_object_mask_preserve_source_dimensions(self):
        source = _make_image(320, 180, color=(1, 2, 3))
        generated = _make_image(512, 512, color=(9, 8, 7))
        masks = [Image.new("L", source.size, 255), Image.new("L", source.size, 0)]
        masks[1].paste(255, (10, 10, 30, 30))
        masks[1].paste(255, (250, 120, 300, 160))

        for mask in masks:
            _, _, box = api._inpaint_region(source, mask, 64)
            result = api._composite_generated_region(source, mask, generated, box)
            self.assertEqual(result.size, source.size)


class InpaintPipelineTests(unittest.TestCase):
    def test_uses_edit_plus_pipeline_without_legacy_mask_arguments(self):
        class RecordingPipeline:
            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return types.SimpleNamespace(
                    images=[_make_image(8, 8, color=(200, 100, 50))]
                )

        pipeline = RecordingPipeline()
        with mock.patch.object(api, "_pipeline", pipeline):
            api._edit_plus_image(
                _make_image(8, 8),
                "replace with fireworks",
                " ",
                2,
                4.0,
                api.torch.Generator(device="cuda").manual_seed(0),
                None,
            )

        self.assertEqual(pipeline.kwargs["num_inference_steps"], 2)
        self.assertEqual(pipeline.kwargs["true_cfg_scale"], 4.0)
        self.assertEqual(pipeline.kwargs["guidance_scale"], 1.0)
        self.assertEqual(pipeline.kwargs["prompt"], "replace with fireworks")
        self.assertEqual(pipeline.kwargs["negative_prompt"], " ")
        self.assertNotIn("mask_image", pipeline.kwargs)
        self.assertNotIn("strength", pipeline.kwargs)


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

    def test_step_callback_estimates_remaining_wall_time(self):
        with mock.patch.object(
            api.time, "monotonic", side_effect=[100.0, 104.0, 110.0, 115.0]
        ):
            callback = api._make_step_callback("req-eta", total_steps=4)

            callback(None, 0, None, {})
            self.assertEqual(
                api._invoke_progress["req-eta"]["estimated_seconds_remaining"],
                12,
            )

            callback(None, 1, None, {})
            progress = api._invoke_progress["req-eta"]
            self.assertEqual(progress["step_duration_seconds"], 6.0)
            self.assertEqual(progress["estimated_seconds_remaining"], 10)

            callback(None, 2, None, {})
            self.assertEqual(
                api._invoke_progress["req-eta"]["estimated_seconds_remaining"],
                5,
            )

    def test_invoke_progress_endpoint_retains_terminal_state(self):
        callback = api._make_step_callback("req-2", total_steps=5)
        callback(None, 0, None, {})
        api._finish_progress("req-2", "succeeded")

        progress = api.invoke_progress("req-2")
        self.assertEqual(progress["status"], "succeeded")
        self.assertEqual(progress["step"], 1)
        self.assertEqual(progress["total"], 5)

    def test_invoke_progress_endpoint_raises_for_unknown_request_id(self):
        with self.assertRaises(api.HTTPException):
            api.invoke_progress("never-seen")

    def test_expired_progress_is_pruned(self):
        api._invoke_progress["expired"] = {
            "status": "succeeded",
            "updated_at": 1,
        }

        api._prune_progress(now=api.PROGRESS_RETENTION_SECONDS + 2)

        self.assertNotIn("expired", api._invoke_progress)


if __name__ == "__main__":
    unittest.main()
