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
from importlib.machinery import ModuleSpec
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
    memory_allocated=lambda: 0,
    memory_reserved=lambda: 0,
    is_available=lambda: False,
    mem_get_info=lambda: (0, 0),
    empty_cache=lambda: None,
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
torchao_stub.__spec__ = ModuleSpec("torchao", loader=None, is_package=True)
torchao_quantization_stub.__spec__ = ModuleSpec(
    "torchao.quantization", loader=None
)
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

if "fastapi.responses" not in sys.modules:
    fastapi_responses_stub = types.ModuleType("fastapi.responses")

    class _FakeResponse:
        def __init__(self, content=None, media_type=None, headers=None):
            self.body = content
            self.media_type = media_type
            self.headers = headers or {}

    fastapi_responses_stub.Response = _FakeResponse
    sys.modules["fastapi.responses"] = fastapi_responses_stub

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
    def test_composite_preserves_every_unmasked_source_pixel(self):
        source = _make_image(80, 120, color=(10, 20, 30))
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (20, 30, 40, 50))
        box = (0, 0, source.width, source.height)
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

    def test_full_mask_and_multi_object_mask_preserve_source_dimensions(self):
        source = _make_image(320, 180, color=(1, 2, 3))
        generated = _make_image(512, 512, color=(9, 8, 7))
        masks = [Image.new("L", source.size, 255), Image.new("L", source.size, 0)]
        masks[1].paste(255, (10, 10, 30, 30))
        masks[1].paste(255, (250, 120, 300, 160))
        box = (0, 0, source.width, source.height)

        for mask in masks:
            result = api._composite_generated_region(source, mask, generated, box)
            self.assertEqual(result.size, source.size)


class FadePreviewTests(unittest.TestCase):
    def test_early_step_stays_close_to_the_original(self):
        original = _make_image(4, 4, color=(0, 0, 0))
        generated = _make_image(4, 4, color=(200, 100, 50))

        result = api._fade_preview(original, generated, step=1, total_steps=20)

        self.assertEqual(result.getpixel((0, 0)), (10, 5, 2))

    def test_final_step_equals_the_generated_frame(self):
        original = _make_image(4, 4, color=(0, 0, 0))
        generated = _make_image(4, 4, color=(200, 100, 50))

        result = api._fade_preview(original, generated, step=20, total_steps=20)

        self.assertEqual(result.getpixel((0, 0)), (200, 100, 50))

    def test_missing_total_steps_returns_the_generated_frame(self):
        original = _make_image(4, 4, color=(0, 0, 0))
        generated = _make_image(4, 4, color=(200, 100, 50))

        result = api._fade_preview(original, generated, step=None, total_steps=None)

        self.assertEqual(result.getpixel((0, 0)), (200, 100, 50))

    def test_resizes_generated_frame_to_match_original(self):
        original = _make_image(8, 8, color=(0, 0, 0))
        generated = _make_image(4, 4, color=(200, 100, 50))

        result = api._fade_preview(original, generated, step=20, total_steps=20)

        self.assertEqual(result.size, original.size)


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
                _FakeGenerator(device="cuda").manual_seed(0),
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
        api._invoke_previews.clear()
        api._cancelled_requests.clear()

    def tearDown(self):
        api._invoke_progress.clear()
        api._invoke_previews.clear()
        api._cancelled_requests.clear()

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

    def test_step_callback_publishes_latest_preview_by_version(self):
        callback = api._make_step_callback(
            "req-preview", total_steps=2,
            preview_renderer=lambda _pipe, _latents, **_kwargs: b"jpeg-preview",
        )

        callback(None, 0, None, {"latents": object()})

        progress = api.invoke_progress("req-preview")
        self.assertEqual(progress["preview_version"], 1)
        self.assertEqual(progress["preview_media_type"], "image/jpeg")
        self.assertEqual(api.invoke_preview("req-preview").body, b"jpeg-preview")

    def test_cancelled_request_stops_at_next_step_callback(self):
        callback = api._make_step_callback("req-cancel", total_steps=2)
        response = api.cancel_invoke("req-cancel")

        self.assertEqual(response["status"], "cancel_requested")
        with self.assertRaises(api.InferenceCancelled):
            callback(None, 0, None, {})

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


class DebugMemoryTests(unittest.TestCase):
    def test_snapshot_reports_allocated_reserved_and_system_available(self):
        with (
            mock.patch.object(api.torch.cuda, "memory_allocated", return_value=1 << 30),
            mock.patch.object(api.torch.cuda, "memory_reserved", return_value=2 << 30),
            mock.patch.object(api, "_mem_available_gib", return_value=12.5),
            mock.patch.object(api.torch.cuda, "is_available", return_value=False),
        ):
            snapshot = api._debug_memory_snapshot()

        self.assertEqual(snapshot["allocated_gib"], 1.0)
        self.assertEqual(snapshot["reserved_gib"], 2.0)
        self.assertEqual(snapshot["system_available_gib"], 12.5)
        self.assertNotIn("cuda_free_gib", snapshot)

    def test_snapshot_includes_cuda_free_when_cuda_available(self):
        with (
            mock.patch.object(api.torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                api.torch.cuda, "mem_get_info", return_value=(3 << 30, 8 << 30)
            ),
            mock.patch.object(api, "_mem_available_gib", return_value=12.5),
        ):
            snapshot = api._debug_memory_snapshot()

        self.assertEqual(snapshot["cuda_free_gib"], 3.0)

    def test_empty_cache_calls_torch_and_returns_fresh_snapshot(self):
        with (
            mock.patch.object(api.torch.cuda, "empty_cache") as empty_cache,
            mock.patch.object(api, "_mem_available_gib", return_value=20.0),
        ):
            snapshot = api.debug_empty_cache()

        empty_cache.assert_called_once()
        self.assertEqual(snapshot["system_available_gib"], 20.0)


if __name__ == "__main__":
    unittest.main()
