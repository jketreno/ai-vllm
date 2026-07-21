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
        inference_context,
        memory_snapshot,
        reset_peak_memory_stats,
        runtime_config,
    )
except ImportError:  # The container installs this API as /app/sam3_worker.py.
    from sam3_runtime import (
        PlatformSAM3Annotator,
        inference_context,
        memory_snapshot,
        reset_peak_memory_stats,
        runtime_config,
    )


PROTOCOL_VERSION = "1"
MAX_ATTACHMENT_BYTES = int(
    os.environ.get("MODEL_RPC_MAX_ATTACHMENT_BYTES", str(64 * 1024 * 1024))
)
app = FastAPI(title="SAM3 model worker", version="1.0.0")
runtime = runtime_config()
annotator = PlatformSAM3Annotator(runtime)
inference_lock = threading.Lock()

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
        "status": "ready" if loaded else "loading",
        "model_loaded": loaded,
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


def _segment(image: Image.Image, prompts: list[str], threshold: float):
    _, processor = annotator.initialize()
    processor.confidence_threshold = 0.05
    state = processor.set_image(image)
    segments = []
    attachments = []
    for concept in prompts:
        processor.reset_all_prompts(state)
        output = processor.set_text_prompt(state=state, prompt=concept)
        scores = output["scores"].detach().cpu().numpy()
        boxes = output["boxes"].detach().cpu().numpy()
        masks = output["masks"].detach().cpu().numpy()
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
    try:
        with inference_lock:
            reset_peak_memory_stats(runtime)
            with (
                torch.inference_mode(),
                inference_context(runtime),
                INFERENCE_SECONDS.time(),
            ):
                segments, outputs = _segment(
                    image,
                    prompts,
                    float(parameters.get("threshold", 0.15)),
                )
            inference_memory = memory_snapshot(runtime)
        INFERENCE_REQUESTS.labels("success").inc()
        _update_device_metrics()
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
    except Exception:
        INFERENCE_REQUESTS.labels("error").inc()
        raise
