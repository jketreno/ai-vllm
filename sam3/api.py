"""Headless SAM3 model worker using the private capability RPC."""

import base64
import io
import json
import os
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Gauge, Histogram, start_http_server

try:
    from .runtime import (
        PlatformSAM3Annotator,
        empty_device_cache,
        inference_context,
        memory_snapshot,
        reset_peak_memory_stats,
        runtime_config,
    )
except ImportError:  # The container installs this API as /app/sam3_worker.py.
    from sam3_runtime import (
        PlatformSAM3Annotator,
        empty_device_cache,
        inference_context,
        memory_snapshot,
        reset_peak_memory_stats,
        runtime_config,
    )


PROTOCOL_VERSION = "1"
MAX_ATTACHMENT_BYTES = int(
    os.environ.get("MODEL_RPC_MAX_ATTACHMENT_BYTES", str(64 * 1024 * 1024))
)
# Sam3Processor._forward_grounding bilinear-upsamples every mask above
# confidence_threshold to the original image's full resolution before
# returning. A prompt matching many near-identical objects (e.g. a table
# full of playing cards) can keep far more candidates than any caller uses,
# and each full-resolution mask is multiple MB -- enough in aggregate to
# exhaust a 12GB Arc B580 on a single prompt. This bounds how many
# candidates per prompt ever reach that upsample step, independent of how
# many score above threshold.
MAX_CANDIDATES_PER_PROMPT = int(
    os.environ.get("SAM3_MAX_CANDIDATES_PER_PROMPT", "32")
)
app = FastAPI(title="SAM3 model worker", version="1.0.0")
runtime = runtime_config()
annotator = PlatformSAM3Annotator(runtime)
inference_lock = threading.Lock()

# Set by invoke() whenever a real inference call raises (as opposed to a bad
# request, which is rejected before this point via HTTPException). Cleared on
# the next successful inference. /health/ready surfaces this so a wedged
# accelerator (e.g. an out-of-memory error that leaves the device unusable)
# fails the Docker healthcheck with the actual error instead of reporting
# healthy just because the model loaded once at startup.
last_inference_error: dict | None = None

MODEL_LOADED = Gauge("sam3_model_loaded", "Whether the SAM3 model is loaded")
MODEL_LOADS = Counter("sam3_model_loads_total", "SAM3 model loads", ["status"])
MODEL_LOAD_SECONDS = Histogram("sam3_model_load_seconds", "SAM3 model load latency")
INFERENCE_SECONDS = Histogram(
    "sam3_annotation_duration_seconds", "SAM3 inference latency"
)
INFERENCE_REQUESTS = Counter(
    "sam3_annotation_requests_total", "SAM3 inference requests", ["status"]
)
CUDA_ALLOCATED = Gauge(
    "sam3_cuda_memory_allocated_bytes", "Accelerator memory allocated"
)
CUDA_RESERVED = Gauge(
    "sam3_cuda_memory_reserved_bytes", "Accelerator memory reserved"
)
CUDA_FREE = Gauge("sam3_cuda_memory_free_bytes", "Accelerator memory free")


def _update_device_metrics():
    snapshot = memory_snapshot(runtime)
    if not snapshot:
        return
    CUDA_ALLOCATED.set(snapshot["allocated"])
    CUDA_RESERVED.set(snapshot["reserved"])
    CUDA_FREE.set(snapshot["free"])


@app.on_event("startup")
def startup():
    start_http_server(int(os.environ.get("SAM3_METRICS_PORT", "9092")))
    started = time.monotonic()
    try:
        annotator.initialize()
        MODEL_LOADED.set(1)
        MODEL_LOADS.labels("success").inc()
    except Exception:
        MODEL_LOADED.set(0)
        MODEL_LOADS.labels("error").inc()
        raise
    finally:
        MODEL_LOAD_SECONDS.observe(time.monotonic() - started)
        _update_device_metrics()


@app.get("/health/live")
def live():
    return {"status": "ok", "platform": runtime.platform}


@app.get("/health/ready")
def ready(response: Response):
    loaded = annotator.model is not None
    if not loaded:
        response.status_code = 503
        return {
            "status": "loading",
            "model_loaded": False,
            "platform": runtime.platform,
            "device": runtime.device,
            "precision": runtime.precision,
        }
    if last_inference_error is not None:
        response.status_code = 503
        return {
            "status": "error",
            "model_loaded": True,
            "platform": runtime.platform,
            "device": runtime.device,
            "precision": runtime.precision,
            "last_inference_error": last_inference_error,
        }
    return {
        "status": "ready",
        "model_loaded": True,
        "platform": runtime.platform,
        "device": runtime.device,
        "precision": runtime.precision,
    }


@app.get("/v1/capabilities")
def capabilities():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "worker": "sam3",
        "runtime": {
            "platform": runtime.platform,
            "device": runtime.device,
            "precision": runtime.precision,
            "resolution": runtime.resolution,
        },
        "operations": {
            "segment": {
                "inputs": ["image"],
                "parameters": ["prompts", "threshold"],
                "outputs": ["segments", "mask:*"],
            }
        },
    }


async def _read_attachments(
    manifest: dict, files: list[UploadFile]
) -> dict[str, bytes]:
    descriptors = manifest.get("attachments", [])
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
        result[name] = payload
    return result


def _mask_attachment(mask: np.ndarray, name: str) -> dict:
    output = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(output, format="PNG")
    return {
        "name": name,
        "media_type": "image/png",
        "data_base64": base64.b64encode(output.getvalue()).decode("ascii"),
    }


def _grounding_for_prompt(processor, state: dict, concept: str, threshold: float):
    """Run text-prompt grounding and return capped, full-resolution results.

    Reimplements Sam3Processor.set_text_prompt/_forward_grounding's tail
    instead of calling them directly so we can cap the candidate count
    *before* each candidate's mask is bilinear-upsampled to the original
    image's resolution -- that upsample is what OOMs on a 12GB Arc B580 when
    a prompt matches many near-identical small objects (e.g. a table full of
    playing cards), even with a reasonable confidence threshold.
    """
    from sam3.model import box_ops
    from sam3.model.data_misc import interpolate

    model = processor.model
    text_outputs = model.backbone.forward_text([concept], device=processor.device)
    state["backbone_out"].update(text_outputs)
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = model._get_dummy_prompt()

    outputs = model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=processor.find_stage,
        geometric_prompt=state["geometric_prompt"],
        find_target=None,
    )
    out_logits = outputs["pred_logits"]
    out_probs = out_logits.sigmoid()
    presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    out_probs = (out_probs * presence_score).squeeze(-1)

    keep = out_probs > threshold
    out_probs = out_probs[keep]
    out_masks = outputs["pred_masks"][keep]
    out_bbox = outputs["pred_boxes"][keep]

    if out_probs.shape[0] > MAX_CANDIDATES_PER_PROMPT:
        top = torch.topk(out_probs, MAX_CANDIDATES_PER_PROMPT)
        out_probs = out_probs[top.indices]
        out_masks = out_masks[top.indices]
        out_bbox = out_bbox[top.indices]

    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    img_h, img_w = state["original_height"], state["original_width"]
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(processor.device)
    boxes = boxes * scale_fct[None, :]

    out_masks = interpolate(
        out_masks.unsqueeze(1), (img_h, img_w), mode="bilinear", align_corners=False
    ).sigmoid()

    return {"scores": out_probs, "boxes": boxes, "masks": out_masks > 0.5}


def _segment(image: Image.Image, prompts: list[str], threshold: float):
    _, processor = annotator.initialize()
    state = processor.set_image(image)
    segments = []
    attachments = []
    for concept in prompts:
        processor.reset_all_prompts(state)
        empty_device_cache(runtime)
        output = _grounding_for_prompt(processor, state, concept, threshold)
        scores = output["scores"].detach().float().cpu().numpy()
        boxes = output["boxes"].detach().float().cpu().numpy()
        masks = output["masks"].detach().float().cpu().numpy()
        for index, score in enumerate(scores):
            if float(score) < threshold:
                continue
            mask = np.squeeze(masks[index]).astype(bool)
            if mask.shape != (image.height, image.width):
                mask = np.asarray(
                    Image.fromarray(mask.astype(np.uint8)).resize(
                        image.size, Image.Resampling.NEAREST
                    )
                ).astype(bool)
            name = f"mask:{len(segments)}"
            attachments.append(_mask_attachment(mask, name))
            segments.append({
                "concept": concept,
                "score": round(float(score), 4),
                "box": [round(float(value), 1) for value in boxes[index]],
                "mask_attachment": name,
            })
    processor.reset_all_prompts(state)
    empty_device_cache(runtime)
    return segments, attachments


@app.post("/v1/invoke")
async def invoke(manifest: str = Form(...), attachments: list[UploadFile] = File(...)):
    started = time.monotonic()
    try:
        request = json.loads(manifest)
    except (TypeError, json.JSONDecodeError) as error:
        raise HTTPException(400, "manifest must be valid JSON") from error
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise HTTPException(400, "unsupported protocol_version")
    if request.get("operation") != "segment":
        raise HTTPException(404, "unknown operation")
    uploaded = await _read_attachments(request, attachments)
    try:
        image = Image.open(io.BytesIO(uploaded["image"])).convert("RGB")
    except (KeyError, UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "a valid image attachment is required") from error
    parameters = request.get("parameters", {})
    prompts = parameters.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise HTTPException(400, "prompts must be a non-empty list")
    prompts = [
        value.strip()
        for value in prompts
        if isinstance(value, str) and value.strip()
    ][:24]
    global last_inference_error
    threshold = float(parameters.get("threshold", 0.15))
    try:
        with inference_lock:
            reset_peak_memory_stats(runtime)
            empty_device_cache(runtime)
            try:
                with (
                    torch.inference_mode(),
                    inference_context(runtime),
                    INFERENCE_SECONDS.time(),
                ):
                    segments, outputs = _segment(image, prompts, threshold)
            except torch.OutOfMemoryError:
                # A single OOM doesn't mean the device is wedged -- the
                # caching allocator can be fragmented from a prior request
                # even though the model itself is fine. Retry once after a
                # full cache eviction before surfacing an error and forcing
                # the caller through a much slower container restart.
                empty_device_cache(runtime)
                with (
                    torch.inference_mode(),
                    inference_context(runtime),
                    INFERENCE_SECONDS.time(),
                ):
                    segments, outputs = _segment(image, prompts, threshold)
            inference_memory = memory_snapshot(runtime)
        INFERENCE_REQUESTS.labels("success").inc()
        _update_device_metrics()
        last_inference_error = None
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request.get("request_id"),
            "status": "ok",
            "data": {
                "segments": segments,
                "width": image.width,
                "height": image.height,
            },
            "attachments": outputs,
            "metadata": {
                "duration_seconds": round(time.monotonic() - started, 4),
                "platform": runtime.platform,
                "device": runtime.device,
                "precision": runtime.precision,
                "resolution": runtime.resolution,
                "accelerator_memory": inference_memory,
            },
        }
    except Exception as error:
        INFERENCE_REQUESTS.labels("error").inc()
        last_inference_error = {
            "type": type(error).__name__,
            "message": str(error)[:500],
            "at": time.time(),
        }
        raise
