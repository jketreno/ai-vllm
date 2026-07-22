"""FastAPI wrapper around a torchao fp8-quantized Qwen-Image-Edit-2511 pipeline.

Loading is split into instrumented sections (transformer, text_encoder, pipeline
assembly). Before each section, free host memory is checked against that section's
expected requirement -- GB10 is a unified-memory system, so host RAM and GPU memory
are the same physical pool, and MemAvailable is a fair proxy for real headroom (which
has repeatedly measured lower than vLLM's nominal `1 - gpu-memory-utilization` figure
would suggest). If headroom looks insufficient, loading stops before the risky
allocation instead of attempting it and potentially OOMing the whole host. Progress
and measured memory deltas are persisted to disk after every section so a stalled or
aborted load leaves a clear record, and so future runs can use measured actuals
instead of guessed size requirements.
"""

import asyncio
import base64
import binascii
import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import time
import uuid

import torch
from diffusers import (
    AutoModel,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    TorchAoConfig,
)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageChops, ImageFilter, UnidentifiedImageError
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from torchao.quantization import Float8WeightOnlyConfig
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL_ID = os.environ.get("QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
METRICS_PORT = int(os.environ.get("QWEN_IMAGE_EDIT_METRICS_PORT", "9093"))
STATUS_PATH = os.environ.get(
    "QWEN_IMAGE_EDIT_STATUS_PATH", "/app/state/load_status.json"
)
# Quantizing the transformer from bf16 to fp8 on every startup takes several minutes
# and briefly holds both the bf16 source and the fp8 result in memory. Cache the
# already-quantized weights on first run and load them directly on subsequent starts
# -- torchao persists real fp8 tensors here (not bf16 + a re-quantize instruction), so
# reload skips quantization compute entirely. Lives on the shared HF cache volume,
# alongside the other cached model repos, not the load-status state volume.
QUANTIZED_TRANSFORMER_PATH = os.environ.get(
    "QWEN_IMAGE_EDIT_QUANTIZED_TRANSFORMER_PATH",
    "/root/.cache/huggingface/qwen-image-edit-2511-transformer-fp8",
)

# Expected host MemAvailable required for each section, in GiB, before attempting it.
# Defaults are measured actuals from a successful concurrent run alongside vllm-engine
# on this GB10 node (see STATUS_PATH history / project plan), rounded up slightly for
# margin: transformer 12.81, text_encoder 13.38, pipeline 0.66 GiB actually used.
# Override via env if a future run's STATUS_PATH actuals differ meaningfully.
REQUIRED_GIB = {
    "transformer": float(
        os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_TRANSFORMER_GIB", "14")
    ),
    "text_encoder": float(
        os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_TEXT_ENCODER_GIB", "15")
    ),
    "pipeline": float(os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_PIPELINE_GIB", "2")),
}
SAFETY_MARGIN_GIB = float(os.environ.get("QWEN_IMAGE_EDIT_SAFETY_MARGIN_GIB", "4"))
INFERENCE_REQUIRED_GIB = float(
    os.environ.get("QWEN_IMAGE_EDIT_INFERENCE_REQUIRED_GIB", "16")
)
PROFILE = os.environ.get("QWEN_IMAGE_EDIT_PROFILE", "base").strip().lower()
LIGHTNING_REPO = os.environ.get(
    "QWEN_IMAGE_EDIT_LIGHTNING_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning"
)
LIGHTNING_WEIGHT = os.environ.get(
    "QWEN_IMAGE_EDIT_LIGHTNING_WEIGHT",
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
)
log = logging.getLogger("qwen_image_edit")
PROGRESS_RETENTION_SECONDS = int(
    os.environ.get("QWEN_IMAGE_EDIT_PROGRESS_RETENTION_SECONDS", "86400")
)
PREVIEW_MAX_DIMENSION = int(
    os.environ.get("QWEN_IMAGE_EDIT_PREVIEW_MAX_DIMENSION", "512")
)
PREVIEW_JPEG_QUALITY = int(
    os.environ.get("QWEN_IMAGE_EDIT_PREVIEW_JPEG_QUALITY", "72")
)

MARKER_COLOR = (255, 0, 255)
MARKER_HALO_COLOR = (255, 255, 255)

LATENCY_BUCKETS = (10, 30, 60, 120, 300, 600, 900, 1200, 1800)

EDIT_LATENCY = Histogram(
    "qwen_image_edit_inference_seconds",
    "Time spent running one edit inference",
    buckets=LATENCY_BUCKETS,
)
INPAINT_LATENCY = Histogram(
    "qwen_image_edit_inpaint_inference_seconds",
    "Time spent running one inpaint/outpaint inference",
    buckets=LATENCY_BUCKETS,
)
INFERENCE_REQUESTS = Counter(
    "qwen_image_edit_requests_total",
    "Inference requests by operation and outcome",
    ["operation", "outcome", "profile"],
)
INFERENCE_MEMORY = Gauge(
    "qwen_image_edit_inference_mem_available_gib",
    "Host MemAvailable observed around inference",
    ["phase", "profile"],
)
STEP_LATENCY = Histogram(
    "qwen_image_edit_step_duration_seconds",
    "Time between model step callbacks",
    ["profile"],
    buckets=(1, 2, 5, 10, 20, 30, 45, 60, 90),
)
INFERENCE_DIMENSION = Histogram(
    "qwen_image_edit_dimension_pixels",
    "Input dimensions by operation and axis",
    ["operation", "axis", "profile"],
    buckets=(256, 512, 768, 1024, 1280, 1600, 2048, 3072, 4096),
)

CUDA_ALLOCATED = Gauge(
    "qwen_image_edit_cuda_memory_allocated_bytes",
    "CUDA memory allocated by this process",
)
CUDA_RESERVED = Gauge(
    "qwen_image_edit_cuda_memory_reserved_bytes", "CUDA memory reserved by this process"
)
CUDA_FREE = Gauge(
    "qwen_image_edit_cuda_memory_free_bytes", "CUDA memory currently free"
)


def update_cuda_metrics():
    """Update process and device CUDA memory gauges when CUDA is available."""
    try:
        if torch.cuda.is_available():
            CUDA_ALLOCATED.set(torch.cuda.memory_allocated())
            CUDA_RESERVED.set(torch.cuda.memory_reserved())
            free, _ = torch.cuda.mem_get_info()
            CUDA_FREE.set(free)
    except Exception:
        pass


MAX_CANVAS_DIMENSION = int(
    os.environ.get("QWEN_IMAGE_EDIT_MAX_CANVAS_DIMENSION", "4096")
)
PROTOCOL_VERSION = "1"
MAX_ATTACHMENT_BYTES = int(
    os.environ.get("MODEL_RPC_MAX_ATTACHMENT_BYTES", str(64 * 1024 * 1024))
)

app = FastAPI(title="ai-vllm Qwen-Image-Edit API", version="1.0.0")
_pipeline = None
_pipeline_lock = threading.Lock()

# Per-request diffusion-step progress, keyed by the RPC manifest's request_id (see
# rpc.py's WorkerClient.invoke, which generates this id before POSTing). Lets a
# caller poll GET /v1/invoke/{request_id}/progress while the single long-running
# /v1/invoke call is still in flight, mirroring the _LoadStatus file-based progress
# pattern above but held in memory since this is per-request, not per-process-load.
_invoke_progress: dict[str, dict] = {}
_invoke_previews: dict[str, bytes] = {}
_cancelled_requests: set[str] = set()
_invoke_progress_lock = threading.Lock()


class InferenceCancelled(HTTPException):
    def __init__(self):
        super().__init__(409, "image inference was cancelled")


def _prune_progress(now: float | None = None) -> None:
    cutoff = (now or time.time()) - PROGRESS_RETENTION_SECONDS
    with _invoke_progress_lock:
        expired = [
            request_id
            for request_id, progress in _invoke_progress.items()
            if progress.get("updated_at", 0) < cutoff
        ]
        for request_id in expired:
            del _invoke_progress[request_id]
            _invoke_previews.pop(request_id, None)
            _cancelled_requests.discard(request_id)


def _set_progress_stage(request_id: str | None, status: str, **details) -> None:
    if not request_id:
        return
    _prune_progress()
    with _invoke_progress_lock:
        current = _invoke_progress.get(request_id, {})
        _invoke_progress[request_id] = {
            **current,
            "status": status,
            "updated_at": time.time(),
            **details,
        }


def _pipeline_dimensions(image: Image.Image) -> tuple[int, int]:
    ratio = image.width / image.height
    raw_width = math.sqrt(1024 * 1024 * ratio)
    raw_height = raw_width / ratio
    width = round(raw_width / 32) * 32
    height = round(raw_height / 32) * 32
    return width, height


def _decode_step_preview(pipe, packed_latents, width: int, height: int) -> Image.Image:
    latents = pipe._unpack_latents(
        packed_latents, height, width, pipe.vae_scale_factor
    ).to(pipe.vae.dtype)
    preview_scale = min(1.0, PREVIEW_MAX_DIMENSION / max(width, height))
    if preview_scale < 1.0:
        latent_height = max(2, round(latents.shape[-2] * preview_scale))
        latent_width = max(2, round(latents.shape[-1] * preview_scale))
        latents = torch.nn.functional.interpolate(
            latents,
            size=(latents.shape[-3], latent_height, latent_width),
            mode="trilinear",
            align_corners=False,
        )
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    decoded = pipe.vae.decode(latents / latents_std + latents_mean, return_dict=False)[
        0
    ][:, :, 0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def _encode_step_preview(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((PREVIEW_MAX_DIMENSION, PREVIEW_MAX_DIMENSION))
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=max(20, min(PREVIEW_JPEG_QUALITY, 95)),
        optimize=True,
    )
    return buffer.getvalue()


def _preview_renderer(
    inference_image: Image.Image,
    compositor=None,
    step_aware: bool = False,
):
    width, height = _pipeline_dimensions(inference_image)

    def render(pipe, packed_latents, step=None, total_steps=None) -> bytes:
        generated = _decode_step_preview(pipe, packed_latents, width, height)
        if compositor is None:
            composited = generated
        elif step_aware:
            composited = compositor(generated, step, total_steps)
        else:
            composited = compositor(generated)
        return _encode_step_preview(composited)

    return render


def _request_cancelled(request_id: str | None) -> bool:
    if not request_id:
        return False
    with _invoke_progress_lock:
        return request_id in _cancelled_requests


def _make_step_callback(
    request_id: str | None, total_steps: int, preview_renderer=None
):
    started_at = time.monotonic()
    last_observed = [started_at]

    def _on_step_end(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
        if _request_cancelled(request_id):
            raise InferenceCancelled()
        now = time.monotonic()
        step_duration = now - last_observed[0]
        STEP_LATENCY.labels(PROFILE).observe(step_duration)
        last_observed[0] = now
        completed_steps = step + 1
        remaining_steps = max(total_steps - completed_steps, 0)
        average_step_duration = (now - started_at) / completed_steps
        estimated_seconds_remaining = math.ceil(average_step_duration * remaining_steps)
        preview_version = None
        if preview_renderer is not None and "latents" in callback_kwargs:
            try:
                preview = preview_renderer(
                    pipe,
                    callback_kwargs["latents"],
                    step=completed_steps,
                    total_steps=total_steps,
                )
                preview_version = completed_steps
            except Exception:  # noqa: BLE001 -- preview failure must not fail inference
                log.exception(
                    "request_id=%s step=%d preview decode failed",
                    request_id,
                    completed_steps,
                )
                preview = None
        else:
            preview = None
        if _request_cancelled(request_id):
            raise InferenceCancelled()
        with _invoke_progress_lock:
            if preview is not None:
                _invoke_previews[request_id] = preview
            _invoke_progress[request_id] = {
                "status": "running",
                "step": completed_steps,
                "total": total_steps,
                "step_duration_seconds": round(step_duration, 1),
                "estimated_seconds_remaining": estimated_seconds_remaining,
                "profile": PROFILE,
                "updated_at": time.time(),
                **(
                    {
                        "preview_version": preview_version,
                        "preview_media_type": "image/jpeg",
                    }
                    if preview_version is not None
                    else {}
                ),
            }
        return callback_kwargs

    return _on_step_end


def _finish_progress(request_id: str | None, status: str, detail: str | None = None):
    if not request_id:
        return
    with _invoke_progress_lock:
        current = _invoke_progress.get(request_id, {})
        _invoke_progress[request_id] = {
            **current,
            "status": status,
            "updated_at": time.time(),
            **({"detail": detail[:500]} if detail else {}),
        }


def _record_cancelled(operation: str, request_id: str | None) -> None:
    INFERENCE_REQUESTS.labels(operation, "cancelled", PROFILE).inc()
    _finish_progress(request_id, "cancelled", "cancelled by client")


def _mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _gate_inference(operation: str, request_id: str | None, total_steps: int) -> None:
    available = _mem_available_gib()
    INFERENCE_MEMORY.labels("before", PROFILE).set(available)
    if request_id:
        with _invoke_progress_lock:
            cancelled = request_id in _cancelled_requests
            _invoke_previews.pop(request_id, None)
            _invoke_progress[request_id] = {
                "status": "cancelled" if cancelled else "resource_check",
                "step": 0,
                "total": total_steps,
                "profile": PROFILE,
                "mem_available_gib": round(available, 2),
                "updated_at": time.time(),
            }
        if cancelled:
            INFERENCE_REQUESTS.labels(operation, "cancelled", PROFILE).inc()
            raise InferenceCancelled()
    if available < INFERENCE_REQUIRED_GIB:
        INFERENCE_REQUESTS.labels(operation, "rejected", PROFILE).inc()
        _finish_progress(request_id, "failed", "insufficient memory headroom")
        raise HTTPException(
            503,
            f"image inference needs at least {INFERENCE_REQUIRED_GIB:.1f} GiB "
            f"MemAvailable; only {available:.1f} GiB is available",
            headers={"Retry-After": "30"},
        )


def _observe_dimensions(operation: str, image: Image.Image) -> None:
    INFERENCE_DIMENSION.labels(operation, "width", PROFILE).observe(image.width)
    INFERENCE_DIMENSION.labels(operation, "height", PROFILE).observe(image.height)


def _observe_post_inference_memory() -> None:
    try:
        INFERENCE_MEMORY.labels("after", PROFILE).set(_mem_available_gib())
    except (OSError, RuntimeError):
        log.exception("failed to observe post-inference memory")


def _generation_settings(steps: int, true_cfg_scale: float) -> tuple[int, float]:
    if PROFILE == "lightning":
        return 4, 1.0
    return _clamp_steps(steps), true_cfg_scale


def _cuda_mem_gib():
    return {
        "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
    }


class LoadAborted(RuntimeError):
    pass


class _LoadStatus:
    """Persists load progress to disk after every section, and gates each section on
    measured free memory before attempting it, aborting cleanly if headroom is short."""

    def __init__(self, path):
        self.path = path
        self.sections = []
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._write({"state": "starting", "sections": []})

    def _write(self, extra):
        payload = {"sections": self.sections, "updated_at": time.time(), **extra}
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.path)

    def gate(self, section):
        required = REQUIRED_GIB[section]
        available = _mem_available_gib()
        threshold = required + SAFETY_MARGIN_GIB
        record = {
            "section": section,
            "required_gib": round(required, 2),
            "safety_margin_gib": SAFETY_MARGIN_GIB,
            "available_before_gib": round(available, 2),
        }
        if available < threshold:
            record["decision"] = "aborted_insufficient_memory"
            self.sections.append(record)
            self._write({"state": "aborted"})
            raise LoadAborted(
                f"Refusing to load '{section}': need ~{required:.1f} GiB + "
                f"{SAFETY_MARGIN_GIB:.1f} GiB safety margin = {threshold:.1f} GiB, "
                f"only {available:.1f} GiB host MemAvailable."
            )
        record["decision"] = "proceeding"
        self._pending = record
        self._write({"state": f"loading_{section}"})
        return record

    def note(self, key, value):
        self._pending[key] = value

    def record_done(self, section, extra_timing_s):
        available_after = _mem_available_gib()
        record = self._pending
        record["available_after_gib"] = round(available_after, 2)
        record["actual_used_gib"] = round(
            record["available_before_gib"] - available_after, 2
        )
        record["duration_s"] = round(extra_timing_s, 1)
        record.update(_cuda_mem_gib())
        self.sections.append(record)
        self._write({"state": f"loaded_{section}"})


def _load_transformer_cached(status):
    marker = os.path.join(QUANTIZED_TRANSFORMER_PATH, "config.json")
    t0 = time.time()
    if os.path.exists(marker):
        transformer = AutoModel.from_pretrained(
            QUANTIZED_TRANSFORMER_PATH,
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
            device_map="cuda",
        )
        status.note("transformer_source", "cache")
    else:
        quant_config = TorchAoConfig(quant_type=Float8WeightOnlyConfig())
        transformer = AutoModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        status.note("transformer_source", "quantized_fresh")
        # torchao's Float8WeightOnlyConfig tensor-subclass format isn't safetensors-
        # compatible yet, hence safe_serialization=False -- persists actual fp8
        # tensors so the next startup can load them directly (see comment above).
        transformer.save_pretrained(
            QUANTIZED_TRANSFORMER_PATH, safe_serialization=False
        )
    return transformer, time.time() - t0


def _load_pipeline():
    if PROFILE not in {"base", "lightning"}:
        raise RuntimeError("QWEN_IMAGE_EDIT_PROFILE must be 'base' or 'lightning'")
    status = _LoadStatus(STATUS_PATH)

    status.gate("transformer")
    transformer, elapsed = _load_transformer_cached(status)
    status.record_done("transformer", elapsed)

    status.gate("text_encoder")
    t0 = time.time()
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    status.record_done("text_encoder", time.time() - t0)

    status.gate("pipeline")
    t0 = time.time()
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")  # only moves the small remaining components (vae, tokenizer)
    if PROFILE == "lightning":
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            scheduler_config
        )
        pipeline.load_lora_weights(LIGHTNING_REPO, weight_name=LIGHTNING_WEIGHT)
        status.note("profile", PROFILE)
        status.note("lightning_weight", LIGHTNING_WEIGHT)
    status.record_done("pipeline", time.time() - t0)

    status._write({"state": "ready", "profile": PROFILE})
    return pipeline


@app.on_event("startup")
def startup():
    global _pipeline
    start_http_server(METRICS_PORT)
    try:
        _pipeline = _load_pipeline()
        update_cuda_metrics()
    except LoadAborted as error:
        # Leave _pipeline as None; /health reports not-ready and inference endpoints
        # return 503. The process stays up so status and logs remain inspectable.
        print(f"qwen-image-edit: {error}", flush=True)


_DATA_URI_RE = re.compile(
    r"^data:image/[a-zA-Z0-9.+-]+;base64,(?P<payload>.+)$", re.DOTALL
)


def _decode_image_bytes(raw: bytes, mode: str = "RGB") -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert(mode)
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "Invalid image") from error


async def _read_upload_image(file: UploadFile, mode: str = "RGB") -> Image.Image:
    return _decode_image_bytes(await file.read(), mode)


def _decode_mask(mask_data_uri: str) -> Image.Image:
    """Decode a mask given as a `data:image/png;base64,...` string -- the exact
    format SAM3's /v1/segment endpoint emits for each segment (see sam3/api.py
    `_mask_to_data_uri`), so a SAM3 mask can be forwarded here unmodified. White
    (255) marks the region to replace during compositing; black (0) is preserved."""
    match = _DATA_URI_RE.match(mask_data_uri.strip())
    if not match:
        raise HTTPException(400, "mask must be a data:image/...;base64,... URI")
    try:
        payload = base64.b64decode(match.group("payload"), validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(400, "mask base64 payload is invalid") from error
    return _decode_image_bytes(payload, mode="L")


def _encode_image_response(result: Image.Image) -> dict:
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return {
        "width": result.width,
        "height": result.height,
        "image_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _visual_marker_width(image: Image.Image) -> int:
    """Return a contour width that survives Qwen's vision-input downscaling."""
    return max(6, min(16, round(min(image.size) * 0.008)))


def _annotate_inpaint_region(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Mark the selected object for prompt-driven Qwen Edit Plus inference.

    The colored bands are outside the compositing mask, so even a model that
    reproduces the temporary annotation cannot leak it directly into the result.
    """
    mask = mask.convert("L").point(lambda value: 255 if value else 0)
    marker_width = _visual_marker_width(image)
    inner = mask.filter(ImageFilter.MaxFilter(marker_width * 2 + 1))
    outer = mask.filter(ImageFilter.MaxFilter(marker_width * 4 + 1))
    marker_band = ImageChops.subtract(inner, mask)
    halo_band = ImageChops.subtract(outer, inner)

    annotated = Image.composite(
        Image.new("RGB", image.size, MARKER_HALO_COLOR), image.convert("RGB"), halo_band
    )
    annotated = Image.composite(
        Image.new("RGB", image.size, MARKER_COLOR), annotated, marker_band
    )
    return annotated


def _fade_preview(
    original: Image.Image,
    generated: Image.Image,
    step: int | None,
    total_steps: int | None,
) -> Image.Image:
    """Cross-fade the original image into the live-decoded preview frame as
    denoising progresses, so the preview starts close to the source and
    converges to the generated frame by the final step."""
    alpha = 1.0 if not total_steps else max(0.0, min(1.0, (step or 0) / total_steps))
    original = original.convert("RGB")
    generated = generated.convert("RGB")
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
    return Image.blend(original, generated, alpha)


def _composite_generated_region(
    source: Image.Image,
    source_mask: Image.Image,
    generated: Image.Image,
    region_box: tuple[int, int, int, int],
    strength: float = 1.0,
) -> Image.Image:
    """Return the source-sized canvas with only masked pixels replaced."""
    left, top, right, bottom = region_box
    region_size = (right - left, bottom - top)
    generated = generated.convert("RGB")
    if generated.size != region_size:
        generated = generated.resize(region_size, Image.Resampling.LANCZOS)
    source_region = source.crop(region_box)
    mask_region = source_mask.crop(region_box)
    if strength < 1.0:
        mask_region = mask_region.point(lambda value: round(value * strength))
    composited_region = Image.composite(generated, source_region, mask_region)
    result = source.copy()
    result.paste(composited_region, (left, top))
    return result


def _clamp_steps(num_inference_steps: int) -> int:
    return max(1, min(num_inference_steps, 100))


def _edit_plus_image(
    image,
    prompt,
    negative_prompt,
    num_inference_steps,
    true_cfg_scale,
    generator,
    callback,
):
    return _pipeline(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=1.0,
        generator=generator,
        **({"callback_on_step_end": callback} if callback else {}),
    ).images[0]


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    if _pipeline is None:
        raise HTTPException(503, "model is still loading")
    return {
        "status": "ready",
        "model_loaded": _pipeline is not None,
        "inpaint_model_loaded": _pipeline is not None,
    }


@app.get("/v1/load-status")
def load_status():
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(404, "No load status recorded yet")


async def edit(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(20),
    true_cfg_scale: float = Form(4.0),
    seed: int = Form(0),
    reference_files: list[UploadFile] | None = File(None),
    request_id: str | None = None,
):
    """Whole-image, prompt-driven edit: text editing, object add/remove/move, pose
    changes, style transfer, detail enhancement. Pass 1-2 `reference_files` to fuse
    elements from additional images into the edit, per Qwen-Image-Edit-Plus's
    multi-image fusion support (up to 3 images total, including `file`). `request_id`
    (if given) is the /v1/invoke manifest's request_id, used as the correlation key
    for GET /v1/invoke/{request_id}/progress step polling."""
    if _pipeline is None:
        raise HTTPException(503, "model is not loaded")
    image = await _read_upload_image(file)
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")
    num_inference_steps, true_cfg_scale = _generation_settings(
        num_inference_steps, true_cfg_scale
    )
    _gate_inference("edit", request_id, num_inference_steps)
    _observe_dimensions("edit", image)

    images = [image]
    for reference_file in reference_files or []:
        images.append(await _read_upload_image(reference_file))
    if len(images) > 3:
        raise HTTPException(400, "at most 3 images total (file + 2 reference_files)")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    callback = (
        _make_step_callback(
            request_id,
            num_inference_steps,
            _preview_renderer(images[-1]),
        )
        if request_id
        else None
    )

    def _run():
        with _pipeline_lock, torch.inference_mode(), EDIT_LATENCY.time():
            try:
                out = _pipeline(
                    image=images if len(images) > 1 else images[0],
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=generator,
                    **({"callback_on_step_end": callback} if callback else {}),
                ).images[0]
                update_cuda_metrics()
                INFERENCE_REQUESTS.labels("edit", "success", PROFILE).inc()
                _finish_progress(request_id, "succeeded")
                return out
            except InferenceCancelled:
                _record_cancelled("edit", request_id)
                raise
            except Exception as error:
                INFERENCE_REQUESTS.labels("edit", "failure", PROFILE).inc()
                _finish_progress(request_id, "failed", str(error))
                raise
            finally:
                _observe_post_inference_memory()

    result = await asyncio.to_thread(_run)
    return _encode_image_response(result)


async def inpaint(
    file: UploadFile = File(...),
    mask: str = Form(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(1.0),
    num_inference_steps: int = Form(20),
    true_cfg_scale: float = Form(4.0),
    seed: int = Form(0),
    padding_mask_crop: int | None = Form(None),
    request_id: str | None = None,
):
    """Mask-guided region edit: repaint the masked area (e.g. a SAM-selected
    region) per `prompt`. `mask` is a `data:image/png;base64,...` string --
    SAM3's /v1/segment mask field can be passed through as-is. The full frame
    is sent to the model and its full-frame output is returned unmodified, so
    `strength` and `padding_mask_crop` are accepted for backward compatibility
    but have no effect. `request_id` (if given) is the /v1/invoke manifest's
    request_id, used as the correlation key for
    GET /v1/invoke/{request_id}/progress step polling."""
    if _pipeline is None:
        raise HTTPException(503, "model is not loaded")
    image = await _read_upload_image(file)
    mask_image = _decode_mask(mask)
    if mask_image.size != image.size:
        raise HTTPException(400, "mask dimensions must match the image")
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")
    num_inference_steps, true_cfg_scale = _generation_settings(
        num_inference_steps, true_cfg_scale
    )
    conditioning_image = _annotate_inpaint_region(image, mask_image)
    _gate_inference("inpaint", request_id, num_inference_steps)
    _observe_dimensions("inpaint", image)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    callback = (
        _make_step_callback(
            request_id,
            num_inference_steps,
            _preview_renderer(
                image,
                lambda generated, step, total_steps: _fade_preview(
                    image, generated, step, total_steps
                ),
                step_aware=True,
            ),
        )
        if request_id
        else None
    )

    def _run():
        with _pipeline_lock, torch.inference_mode(), INPAINT_LATENCY.time():
            try:
                # Qwen-Image-Edit-2511 is prompt-driven rather than mask-conditioned.
                # Give it a temporary visual selection marker; the model's full-frame
                # output is returned as-is, with no cropping or compositing.
                result = _edit_plus_image(
                    conditioning_image,
                    prompt.strip(),
                    negative_prompt,
                    num_inference_steps,
                    true_cfg_scale,
                    generator,
                    callback,
                )
                update_cuda_metrics()
                INFERENCE_REQUESTS.labels("inpaint", "success", PROFILE).inc()
                _finish_progress(request_id, "succeeded")
                return result
            except InferenceCancelled:
                _record_cancelled("inpaint", request_id)
                raise
            except Exception as error:
                INFERENCE_REQUESTS.labels("inpaint", "failure", PROFILE).inc()
                _finish_progress(request_id, "failed", str(error))
                raise
            finally:
                _observe_post_inference_memory()

    result = await asyncio.to_thread(_run)
    response = _encode_image_response(result)
    response["conditioning_image_png_base64"] = _encode_image_response(
        conditioning_image
    )["image_png_base64"]
    return response


def _outpaint_canvas(
    image: Image.Image, target_width: int, target_height: int, anchor: str
):
    """Paste `image` onto a new `target_width` x `target_height` canvas positioned by
    `anchor`, and build the companion mask: black (preserve) over the original image,
    white (repaint) over the newly exposed border. Returns (canvas, mask)."""
    src_w, src_h = image.size
    if target_width < src_w or target_height < src_h:
        raise HTTPException(
            400, "target dimensions must be >= the source image dimensions"
        )

    anchors = {
        "center": ((target_width - src_w) // 2, (target_height - src_h) // 2),
        "top-left": (0, 0),
        "top-right": (target_width - src_w, 0),
        "bottom-left": (0, target_height - src_h),
        "bottom-right": (target_width - src_w, target_height - src_h),
        "top": ((target_width - src_w) // 2, 0),
        "bottom": ((target_width - src_w) // 2, target_height - src_h),
        "left": (0, (target_height - src_h) // 2),
        "right": (target_width - src_w, (target_height - src_h) // 2),
    }
    if anchor not in anchors:
        raise HTTPException(400, f"anchor must be one of {sorted(anchors)}")
    left, top = anchors[anchor]

    canvas = Image.new("RGB", (target_width, target_height))
    canvas.paste(image, (left, top))

    mask = Image.new("L", (target_width, target_height), 255)
    mask.paste(Image.new("L", (src_w, src_h), 0), (left, top))

    return canvas, mask


async def outpaint(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    target_width: int = Form(...),
    target_height: int = Form(...),
    anchor: str = Form("center"),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(20),
    true_cfg_scale: float = Form(4.0),
    seed: int = Form(0),
    request_id: str | None = None,
):
    """Expand the canvas: place the source image at `anchor` within a new
    `target_width` x `target_height` canvas and fill the newly exposed border via
    prompt-driven editing followed by mask compositing.
    `prompt` should describe the extended scene (e.g. "extend the beach and sky").
    `request_id` (if given) is the /v1/invoke manifest's request_id, used as the
    correlation key for GET /v1/invoke/{request_id}/progress step polling."""
    if _pipeline is None:
        raise HTTPException(503, "model is not loaded")
    if target_width > MAX_CANVAS_DIMENSION or target_height > MAX_CANVAS_DIMENSION:
        raise HTTPException(
            400, f"target dimensions must be <= {MAX_CANVAS_DIMENSION}px"
        )
    image = await _read_upload_image(file)
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")
    num_inference_steps, true_cfg_scale = _generation_settings(
        num_inference_steps, true_cfg_scale
    )

    canvas, mask_image = _outpaint_canvas(image, target_width, target_height, anchor)
    _gate_inference("outpaint", request_id, num_inference_steps)
    _observe_dimensions("outpaint", canvas)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    callback = (
        _make_step_callback(
            request_id,
            num_inference_steps,
            _preview_renderer(
                canvas,
                lambda generated: _composite_generated_region(
                    canvas,
                    mask_image,
                    generated,
                    (0, 0, canvas.width, canvas.height),
                ),
            ),
        )
        if request_id
        else None
    )

    def _run():
        with _pipeline_lock, torch.inference_mode(), INPAINT_LATENCY.time():
            try:
                out = _edit_plus_image(
                    canvas,
                    prompt,
                    negative_prompt,
                    num_inference_steps,
                    true_cfg_scale,
                    generator,
                    callback,
                )
                _set_progress_stage(request_id, "compositing")
                result = _composite_generated_region(
                    canvas, mask_image, out, (0, 0, canvas.width, canvas.height)
                )
                update_cuda_metrics()
                INFERENCE_REQUESTS.labels("outpaint", "success", PROFILE).inc()
                _finish_progress(request_id, "succeeded")
                return result
            except InferenceCancelled:
                _record_cancelled("outpaint", request_id)
                raise
            except Exception as error:
                INFERENCE_REQUESTS.labels("outpaint", "failure", PROFILE).inc()
                _finish_progress(request_id, "failed", str(error))
                raise
            finally:
                _observe_post_inference_memory()

    result = await asyncio.to_thread(_run)
    return _encode_image_response(result)


async def transform(
    file: UploadFile = File(...),
    crop_left: int | None = Form(None),
    crop_top: int | None = Form(None),
    crop_width: int | None = Form(None),
    crop_height: int | None = Form(None),
    rotate_degrees: float = Form(0.0),
    expand_canvas: bool = Form(True),
):
    """Pure geometric transform -- crop and/or rotate -- with no model inference.
    Alibaba's API and the local Qwen-Image-Edit pipelines have no crop/rotate
    primitive; this is a deterministic Pillow operation exposed alongside them so
    clients (e.g. auto-sam) can compose crop/rotate/inpaint/outpaint without paying
    GPU time for operations that don't need a model. Crop is applied before rotate.
    """
    image = await _read_upload_image(file)

    crop_fields = (crop_left, crop_top, crop_width, crop_height)
    if any(value is not None for value in crop_fields):
        if any(value is None for value in crop_fields):
            raise HTTPException(
                400,
                "crop requires crop_left, crop_top, crop_width, and crop_height "
                "together",
            )
        left, top, width, height = (int(value) for value in crop_fields)
        if width <= 0 or height <= 0:
            raise HTTPException(400, "crop_width and crop_height must be positive")
        box = (left, top, left + width, top + height)
        if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
            raise HTTPException(400, "crop region falls outside the image bounds")
        image = image.crop(box)

    if rotate_degrees % 360 != 0:
        image = image.rotate(
            -rotate_degrees, expand=expand_canvas, resample=Image.Resampling.BICUBIC
        )

    return _encode_image_response(image)


@app.get("/v1/capabilities")
def capabilities():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "worker": "qwen-image-edit",
        "model": MODEL_ID,
        "profile": PROFILE,
        "generation_defaults": {
            "num_inference_steps": 4 if PROFILE == "lightning" else 20,
            "true_cfg_scale": 1.0 if PROFILE == "lightning" else 4.0,
            "overrides_enforced": PROFILE == "lightning",
        },
        "operations": {
            "edit": {
                "inputs": ["image", "reference:*"],
                "parameters": [
                    "prompt",
                    "negative_prompt",
                    "num_inference_steps",
                    "true_cfg_scale",
                    "seed",
                ],
                "outputs": ["image"],
            },
            "inpaint": {
                "inputs": ["image", "mask"],
                "parameters": [
                    "prompt",
                    "negative_prompt",
                    "strength",
                    "num_inference_steps",
                    "true_cfg_scale",
                    "seed",
                    "padding_mask_crop",
                ],
                "outputs": ["image"],
            },
        },
    }


async def _rpc_attachments(request, files):
    descriptors = request.get("attachments", [])
    if len(descriptors) != len(files):
        raise HTTPException(400, "attachment descriptors do not match uploaded files")
    result = {}
    for descriptor, upload in zip(descriptors, files):
        name = descriptor.get("name")
        if not isinstance(name, str) or not name or name in result:
            raise HTTPException(
                400, "attachment names must be unique non-empty strings"
            )
        payload = await upload.read(MAX_ATTACHMENT_BYTES + 1)
        if len(payload) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(413, f"attachment '{name}' is too large")
        result[name] = (
            payload,
            descriptor.get("media_type", "application/octet-stream"),
        )
    return result


def _upload(name, item):
    payload, media_type = item
    return UploadFile(
        filename=name, file=io.BytesIO(payload), headers={"content-type": media_type}
    )


def _rpc_result(request, result, started):
    image_payload = base64.b64decode(result["image_png_base64"])
    attachments = [
        {
            "name": "image",
            "media_type": "image/png",
            "data_base64": base64.b64encode(image_payload).decode("ascii"),
        }
    ]
    pre_composite = result.get("pre_composite_image_png_base64")
    if pre_composite:
        attachments.append(
            {
                "name": "pre_composite_image",
                "media_type": "image/png",
                "data_base64": pre_composite,
            }
        )
    conditioning = result.get("conditioning_image_png_base64")
    if conditioning:
        attachments.append(
            {
                "name": "conditioning_image",
                "media_type": "image/png",
                "data_base64": conditioning,
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request.get("request_id", str(uuid.uuid4())),
        "status": "ok",
        "data": {"width": result["width"], "height": result["height"]},
        "attachments": attachments,
        "metadata": {"duration_seconds": round(time.monotonic() - started, 4)},
    }


@app.post("/v1/invoke")
async def invoke(manifest: str = Form(...), attachments: list[UploadFile] = File(...)):
    started = time.monotonic()
    try:
        request = json.loads(manifest)
    except (TypeError, json.JSONDecodeError) as error:
        raise HTTPException(400, "manifest must be valid JSON") from error
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise HTTPException(400, "unsupported protocol_version")
    operation = request.get("operation")
    if operation not in {"edit", "inpaint"}:
        raise HTTPException(404, "unknown operation")
    uploaded = await _rpc_attachments(request, attachments)
    if "image" not in uploaded:
        raise HTTPException(400, "image attachment is required")
    parameters = request.get("parameters", {})
    prompt = str(parameters.get("prompt", ""))
    image = _upload("image", uploaded["image"])
    request_id = request.get("request_id")
    log.info(
        "request_id=%s operation=%s stage=started prompt_len=%d prompt_sha256=%s",
        request_id,
        operation,
        len(prompt),
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
    )
    if operation == "edit":
        references = [
            _upload(name, item)
            for name, item in sorted(uploaded.items())
            if name.startswith("reference:")
        ]
        result = await edit(
            file=image,
            prompt=prompt,
            negative_prompt=str(parameters.get("negative_prompt", "")),
            num_inference_steps=int(parameters.get("num_inference_steps", 20)),
            true_cfg_scale=float(parameters.get("true_cfg_scale", 4.0)),
            seed=int(parameters.get("seed", 0)),
            reference_files=references,
            request_id=request_id,
        )
    else:
        if "mask" not in uploaded:
            raise HTTPException(400, "mask attachment is required")
        mask_payload = base64.b64encode(uploaded["mask"][0]).decode("ascii")
        padding = parameters.get("padding_mask_crop")
        result = await inpaint(
            file=image,
            mask=f"data:image/png;base64,{mask_payload}",
            prompt=prompt,
            negative_prompt=str(parameters.get("negative_prompt", "")),
            strength=float(parameters.get("strength", 1.0)),
            num_inference_steps=int(parameters.get("num_inference_steps", 20)),
            true_cfg_scale=float(parameters.get("true_cfg_scale", 4.0)),
            seed=int(parameters.get("seed", 0)),
            padding_mask_crop=int(padding) if padding is not None else None,
            request_id=request_id,
        )
    response = _rpc_result(request, result, started)
    log.info("request_id=%s operation=%s stage=succeeded", request_id, operation)
    return response


@app.get("/v1/invoke/{request_id}/progress")
def invoke_progress(request_id: str):
    _prune_progress()
    with _invoke_progress_lock:
        progress = _invoke_progress.get(request_id)
    if progress is None:
        raise HTTPException(
            404,
            "no progress recorded for this request_id (unknown, not yet started, "
            "or already finished)",
        )
    return progress


@app.get("/v1/invoke/{request_id}/preview")
def invoke_preview(request_id: str):
    _prune_progress()
    with _invoke_progress_lock:
        preview = _invoke_previews.get(request_id)
    if preview is None:
        raise HTTPException(404, "no step preview is available for this request")
    return Response(
        content=preview,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/v1/invoke/{request_id}/cancel")
def cancel_invoke(request_id: str):
    _prune_progress()
    with _invoke_progress_lock:
        current = _invoke_progress.get(request_id, {})
        if current.get("status") in {"succeeded", "failed", "cancelled"}:
            return {"request_id": request_id, "status": current["status"]}
        _cancelled_requests.add(request_id)
        _invoke_progress[request_id] = {
            **current,
            "status": "cancel_requested",
            "updated_at": time.time(),
        }
    return {"request_id": request_id, "status": "cancel_requested"}
