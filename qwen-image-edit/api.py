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
import io
import json
import os
import re
import threading
import time
import uuid

import torch
from diffusers import (
    AutoModel,
    DiffusionPipeline,
    QwenImageEditInpaintPipeline,
    QwenImageEditPlusPipeline,
    TorchAoConfig,
)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import Gauge, Histogram, start_http_server
from torchao.quantization import Float8WeightOnlyConfig
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL_ID = os.environ.get("QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
METRICS_PORT = int(os.environ.get("QWEN_IMAGE_EDIT_METRICS_PORT", "9093"))
STATUS_PATH = os.environ.get("QWEN_IMAGE_EDIT_STATUS_PATH", "/app/state/load_status.json")
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
    "transformer": float(os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_TRANSFORMER_GIB", "14")),
    "text_encoder": float(os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_TEXT_ENCODER_GIB", "15")),
    "pipeline": float(os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_PIPELINE_GIB", "2")),
    # QwenImageEditInpaintPipeline is constructed directly from the edit pipeline's
    # already-loaded scheduler/vae/text_encoder/tokenizer/processor/transformer by
    # reference rather than loading a second copy -- the only new allocation is the
    # small Python wrapper object itself. (Not built via DiffusionPipeline.from_pipe():
    # it unconditionally ends with new_pipeline.to(dtype=...), which torchao's
    # fp8-quantized transformer rejects.)
    "inpaint_pipeline": float(os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_INPAINT_PIPELINE_GIB", "0.5")),
}
SAFETY_MARGIN_GIB = float(os.environ.get("QWEN_IMAGE_EDIT_SAFETY_MARGIN_GIB", "4"))

EDIT_LATENCY = Histogram(
    "qwen_image_edit_inference_seconds", "Time spent running one edit inference"
)
INPAINT_LATENCY = Histogram(
    "qwen_image_edit_inpaint_inference_seconds", "Time spent running one inpaint/outpaint inference"
)

CUDA_ALLOCATED = Gauge(
    "qwen_image_edit_cuda_memory_allocated_bytes", "CUDA memory allocated by this process"
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

MAX_CANVAS_DIMENSION = int(os.environ.get("QWEN_IMAGE_EDIT_MAX_CANVAS_DIMENSION", "4096"))
PROTOCOL_VERSION = "1"
MAX_ATTACHMENT_BYTES = int(os.environ.get("MODEL_RPC_MAX_ATTACHMENT_BYTES", str(64 * 1024 * 1024)))

app = FastAPI(title="ai-vllm Qwen-Image-Edit API", version="1.0.0")
_pipeline = None
_inpaint_pipeline = None
_pipeline_lock = threading.Lock()


def _mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


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
            torch_dtype=torch.bfloat16, use_safetensors=False, device_map="cuda",
        )
        status.note("transformer_source", "cache")
    else:
        quant_config = TorchAoConfig(quant_type=Float8WeightOnlyConfig())
        transformer = AutoModel.from_pretrained(
            MODEL_ID, subfolder="transformer",
            quantization_config=quant_config, torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        status.note("transformer_source", "quantized_fresh")
        # torchao's Float8WeightOnlyConfig tensor-subclass format isn't safetensors-
        # compatible yet, hence safe_serialization=False -- persists actual fp8
        # tensors so the next startup can load them directly (see comment above).
        transformer.save_pretrained(QUANTIZED_TRANSFORMER_PATH, safe_serialization=False)
    return transformer, time.time() - t0


def _load_pipeline():
    status = _LoadStatus(STATUS_PATH)

    status.gate("transformer")
    transformer, elapsed = _load_transformer_cached(status)
    status.record_done("transformer", elapsed)

    status.gate("text_encoder")
    t0 = time.time()
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder",
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    status.record_done("text_encoder", time.time() - t0)

    status.gate("pipeline")
    t0 = time.time()
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, transformer=transformer, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")  # only moves the small remaining components (vae, tokenizer)
    status.record_done("pipeline", time.time() - t0)

    status.gate("inpaint_pipeline")
    t0 = time.time()
    # Constructed directly (not via from_pipe()) sharing this pipeline's already-loaded
    # transformer/text_encoder/vae/scheduler by reference -- no second copy of the
    # weights is allocated. from_pipe() always ends by casting the new pipeline to a
    # dtype (torch.float32 unless overridden), which torchao's fp8-quantized
    # transformer rejects outright (ValueError: "Casting a quantized model to a new
    # dtype is unsupported"), so from_pipe() can never succeed here.
    inpaint_pipeline = QwenImageEditInpaintPipeline(
        scheduler=pipeline.scheduler,
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        processor=pipeline.processor,
        transformer=pipeline.transformer,
    )
    status.record_done("inpaint_pipeline", time.time() - t0)

    status._write({"state": "ready"})
    return pipeline, inpaint_pipeline


@app.on_event("startup")
def startup():
    global _pipeline, _inpaint_pipeline
    start_http_server(METRICS_PORT)
    try:
        _pipeline, _inpaint_pipeline = _load_pipeline()
        update_cuda_metrics()
    except LoadAborted as error:
        # Leave _pipeline/_inpaint_pipeline as None; /health reports not-ready and the
        # inference endpoints return 503. The process stays up so the status file and
        # logs remain inspectable rather than the container silently restart-looping.
        print(f"qwen-image-edit: {error}", flush=True)


_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(?P<payload>.+)$", re.DOTALL)


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
    (255) marks the region to repaint; black (0) is preserved, matching
    QwenImageEditInpaintPipeline's mask_image convention."""
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


def _clamp_steps(num_inference_steps: int) -> int:
    return max(1, min(num_inference_steps, 100))


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    if _pipeline is None or _inpaint_pipeline is None:
        raise HTTPException(503, "model is still loading")
    return {
        "status": "ready",
        "model_loaded": _pipeline is not None,
        "inpaint_model_loaded": _inpaint_pipeline is not None,
    }


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
):
    """Whole-image, prompt-driven edit: text editing, object add/remove/move, pose
    changes, style transfer, detail enhancement. Pass 1-2 `reference_files` to fuse
    elements from additional images into the edit, per Qwen-Image-Edit-Plus's
    multi-image fusion support (up to 3 images total, including `file`)."""
    if _pipeline is None:
        raise HTTPException(503, "model is not loaded")
    image = await _read_upload_image(file)
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")
    num_inference_steps = _clamp_steps(num_inference_steps)

    images = [image]
    for reference_file in reference_files or []:
        images.append(await _read_upload_image(reference_file))
    if len(images) > 3:
        raise HTTPException(400, "at most 3 images total (file + 2 reference_files)")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    def _run():
        with _pipeline_lock, torch.inference_mode(), EDIT_LATENCY.time():
            out = _pipeline(
                image=images if len(images) > 1 else images[0],
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
            ).images[0]
            update_cuda_metrics()
            return out

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
):
    """Mask-guided region edit: repaint only the masked area (e.g. a SAM-selected
    region) per `prompt`, leaving the rest of the image untouched. `mask` is a
    `data:image/png;base64,...` string -- SAM3's /v1/segment mask field can be
    passed through as-is. `strength` (0-1) controls how strongly the masked region
    is regenerated; 1.0 fully replaces it, matching Alibaba's "add/remove/move
    objects" edit but constrained to a specific region instead of the whole image."""
    if _pipeline is None or _inpaint_pipeline is None:
        raise HTTPException(503, "model is not loaded")
    image = await _read_upload_image(file)
    mask_image = _decode_mask(mask)
    if mask_image.size != image.size:
        raise HTTPException(400, "mask dimensions must match the image")
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")
    num_inference_steps = _clamp_steps(num_inference_steps)
    strength = max(0.0, min(strength, 1.0))

    generator = torch.Generator(device="cuda").manual_seed(seed)

    def _run():
        with _pipeline_lock, torch.inference_mode(), INPAINT_LATENCY.time():
            out = _inpaint_pipeline(
                image=image,
                mask_image=mask_image,
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                strength=strength,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                padding_mask_crop=padding_mask_crop,
                generator=generator,
            ).images[0]
            update_cuda_metrics()
            return out

    result = await asyncio.to_thread(_run)
    return _encode_image_response(result)


def _outpaint_canvas(image: Image.Image, target_width: int, target_height: int, anchor: str):
    """Paste `image` onto a new `target_width` x `target_height` canvas positioned by
    `anchor`, and build the companion mask: black (preserve) over the original image,
    white (repaint) over the newly exposed border. Returns (canvas, mask)."""
    src_w, src_h = image.size
    if target_width < src_w or target_height < src_h:
        raise HTTPException(400, "target dimensions must be >= the source image dimensions")

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
):
    """Expand the canvas: place the source image at `anchor` within a new
    `target_width` x `target_height` canvas and fill the newly exposed border via
    masked inpainting (strength fixed at 1.0, since the border starts blank).
    `prompt` should describe the extended scene (e.g. "extend the beach and sky")."""
    if _pipeline is None or _inpaint_pipeline is None:
        raise HTTPException(503, "model is not loaded")
    if target_width > MAX_CANVAS_DIMENSION or target_height > MAX_CANVAS_DIMENSION:
        raise HTTPException(400, f"target dimensions must be <= {MAX_CANVAS_DIMENSION}px")
    image = await _read_upload_image(file)
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")
    num_inference_steps = _clamp_steps(num_inference_steps)

    canvas, mask_image = _outpaint_canvas(image, target_width, target_height, anchor)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    def _run():
        with _pipeline_lock, torch.inference_mode(), INPAINT_LATENCY.time():
            out = _inpaint_pipeline(
                image=canvas,
                mask_image=mask_image,
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                strength=1.0,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
            ).images[0]
            update_cuda_metrics()
            return out

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
            raise HTTPException(400, "crop requires crop_left, crop_top, crop_width, and crop_height together")
        left, top, width, height = (int(value) for value in crop_fields)
        if width <= 0 or height <= 0:
            raise HTTPException(400, "crop_width and crop_height must be positive")
        box = (left, top, left + width, top + height)
        if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
            raise HTTPException(400, "crop region falls outside the image bounds")
        image = image.crop(box)

    if rotate_degrees % 360 != 0:
        image = image.rotate(-rotate_degrees, expand=expand_canvas, resample=Image.Resampling.BICUBIC)

    return _encode_image_response(image)


@app.get("/v1/capabilities")
def capabilities():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "worker": "qwen-image-edit",
        "model": MODEL_ID,
        "operations": {
            "edit": {
                "inputs": ["image", "reference:*"],
                "parameters": ["prompt", "negative_prompt", "num_inference_steps", "true_cfg_scale", "seed"],
                "outputs": ["image"],
            },
            "inpaint": {
                "inputs": ["image", "mask"],
                "parameters": ["prompt", "negative_prompt", "strength", "num_inference_steps", "true_cfg_scale", "seed", "padding_mask_crop"],
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
            raise HTTPException(400, "attachment names must be unique non-empty strings")
        payload = await upload.read(MAX_ATTACHMENT_BYTES + 1)
        if len(payload) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(413, f"attachment '{name}' is too large")
        result[name] = (payload, descriptor.get("media_type", "application/octet-stream"))
    return result


def _upload(name, item):
    payload, media_type = item
    return UploadFile(filename=name, file=io.BytesIO(payload), headers={"content-type": media_type})


def _rpc_result(request, result, started):
    image_payload = base64.b64decode(result["image_png_base64"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request.get("request_id", str(uuid.uuid4())),
        "status": "ok",
        "data": {"width": result["width"], "height": result["height"]},
        "attachments": [{
            "name": "image",
            "media_type": "image/png",
            "data_base64": base64.b64encode(image_payload).decode("ascii"),
        }],
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
    image = _upload("image", uploaded["image"])
    if operation == "edit":
        references = [_upload(name, item) for name, item in sorted(uploaded.items()) if name.startswith("reference:")]
        result = await edit(
            file=image,
            prompt=str(parameters.get("prompt", "")),
            negative_prompt=str(parameters.get("negative_prompt", "")),
            num_inference_steps=int(parameters.get("num_inference_steps", 20)),
            true_cfg_scale=float(parameters.get("true_cfg_scale", 4.0)),
            seed=int(parameters.get("seed", 0)),
            reference_files=references,
        )
    else:
        if "mask" not in uploaded:
            raise HTTPException(400, "mask attachment is required")
        mask_payload = base64.b64encode(uploaded["mask"][0]).decode("ascii")
        padding = parameters.get("padding_mask_crop")
        result = await inpaint(
            file=image,
            mask=f"data:image/png;base64,{mask_payload}",
            prompt=str(parameters.get("prompt", "")),
            negative_prompt=str(parameters.get("negative_prompt", "")),
            strength=float(parameters.get("strength", 1.0)),
            num_inference_steps=int(parameters.get("num_inference_steps", 20)),
            true_cfg_scale=float(parameters.get("true_cfg_scale", 4.0)),
            seed=int(parameters.get("seed", 0)),
            padding_mask_crop=int(padding) if padding is not None else None,
        )
    return _rpc_result(request, result, started)
