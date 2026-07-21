"""Unified public API for image analysis and editing."""

import base64
import io
import json
import os
import re

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from image_ops import (
    image_response,
    outpaint_canvas,
    png_bytes,
    rpc_image,
    segment_response,
    transform_image,
)
from rpc import WorkerClient


app = FastAPI(title="ai-vllm Image API", version="1.0.0")
sam = WorkerClient(os.environ.get("SAM3_WORKER_URL", "http://sam3-worker:8004"))
editor = WorkerClient(
    os.environ.get("QWEN_IMAGE_EDIT_WORKER_URL", "http://qwen-image-edit-worker:8006")
)
MAX_UPLOAD_BYTES = int(
    os.environ.get("IMAGE_API_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))
)
MAX_CANVAS_DIMENSION = int(os.environ.get("IMAGE_API_MAX_CANVAS_DIMENSION", "4096"))
POLICY_URL = os.environ.get("IMAGE_API_POLICY_URL", "http://clare2-policy:8000/v1")
POLICY_HEALTH_URL = os.environ.get(
    "IMAGE_API_POLICY_HEALTH_URL", "http://clare2-policy:8000/health"
)
POLICY_TOKEN_FILE = os.environ.get(
    "IMAGE_API_POLICY_TOKEN_FILE", "/run/secrets/clare2_proxy_token"
)
VISION_MODEL = os.environ.get("IMAGE_API_VISION_MODEL", "Qwen/Qwen3.6-27B-FP8")
CONCEPT_PROMPT = (
    "Identify concrete visibly segmentable regions and write a one-sentence "
    "caption of the image. Return only JSON: "
    "{\"caption\":\"one-sentence description\","
    "\"sam3_prompts\":[\"specific object\"]}. "
    "Avoid synonyms and cap sam3_prompts at 24 prompts."
)


async def policy_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(POLICY_HEALTH_URL)
        return response.is_success
    except httpx.HTTPError:
        return False


async def read_image(file: UploadFile) -> tuple[bytes, str, Image.Image]:
    media_type = file.content_type or ""
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "upload a JPEG, PNG, or WebP image")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image exceeds configured upload limit")
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "invalid image") from error
    return payload, media_type, image


def decode_mask(value: str) -> bytes:
    match = re.match(r"^data:image/[^;]+;base64,(.+)$", value, re.DOTALL)
    if not match:
        raise HTTPException(400, "mask must be an image data URI")
    try:
        return base64.b64decode(match.group(1), validate=True)
    except ValueError as error:
        raise HTTPException(400, "invalid mask base64") from error


async def concepts(payload: bytes, media_type: str) -> dict:
    """Return {"caption": str, "concepts": list[str]} from a single vision call."""
    try:
        with open(POLICY_TOKEN_FILE, encoding="utf-8") as token_file:
            token = token_file.read().strip()
        request = {
            "model": VISION_MODEL, "temperature": 0.1, "max_tokens": 700,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": CONCEPT_PROMPT},
                {"type": "image_url", "image_url": {"url": (
                    f"data:{media_type};base64,"
                    f"{base64.b64encode(payload).decode('ascii')}"
                )}},
            ]}],
        }
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{POLICY_URL}/chat/completions",
                json=request,
                headers={"Authorization": f"Bearer {token}"},
            )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(
            content.strip().removeprefix("```json").removesuffix("```").strip()
        )
        caption = parsed.get("caption")
        values = parsed.get("sam3_prompts", [])
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("no caption returned")
        result = list(dict.fromkeys(
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        ))[:24]
        if not result:
            raise ValueError("no concepts returned")
        return {"caption": caption.strip(), "concepts": result}
    except (
        OSError, httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError
    ) as error:
        raise HTTPException(
            503, f"vision concept extraction unavailable: {error}"
        ) from error


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
async def ready():
    sam_ready = await sam.ready()
    edit_ready = await editor.ready()
    vision_ready = await policy_ready()
    statuses = {
        "analyze": sam_ready and vision_ready,
        "segment": sam_ready,
        "edit": edit_ready,
    }
    return {
        "status": "ready" if all(statuses.values()) else "degraded",
        "capabilities": statuses,
    }


@app.get("/v1/capabilities")
async def capabilities():
    sam_ready = await sam.ready()
    edit_ready = await editor.ready()
    vision_ready = await policy_ready()
    return {"api_version": "1", "capabilities": {
        "analyze": "ready" if sam_ready and vision_ready else "unavailable",
        "segment": "ready" if sam_ready else "unavailable",
        "edit": "ready" if edit_ready else "unavailable",
        "inpaint": "ready" if edit_ready else "unavailable",
        "outpaint": "ready" if edit_ready else "unavailable",
        "transform": "ready",
    }}


async def run_segment(payload, media_type, image, prompts, threshold, fields):
    result = await sam.invoke(
        "segment",
        {"prompts": prompts, "threshold": threshold},
        [("image", payload, media_type)],
    )
    return segment_response(image, result, fields)


@app.post("/v1/images/analyze")
async def analyze(
    file: UploadFile = File(...),
    threshold: float = Form(0.15),
    fields: str = Form("concept,score,box,color,area_pixels,polygon,mask"),
):
    payload, media_type, image = await read_image(file)
    vision = await concepts(payload, media_type)
    result = await run_segment(
        payload, media_type, image, vision["concepts"], threshold,
        set(fields.split(",")),
    )
    return {"caption": vision["caption"], "concepts": vision["concepts"], **result}


@app.post("/v1/images/concepts")
async def image_concepts(file: UploadFile = File(...)):
    payload, media_type, _ = await read_image(file)
    return await concepts(payload, media_type)


@app.post("/v1/images/segment")
async def segment(
    file: UploadFile = File(...),
    prompts: str = Form(...),
    threshold: float = Form(0.15),
    fields: str = Form("concept,score,box,color,area_pixels"),
):
    payload, media_type, image = await read_image(file)
    try:
        prompt_list = json.loads(prompts)
    except json.JSONDecodeError as error:
        raise HTTPException(400, "prompts must be a JSON list") from error
    return await run_segment(
        payload, media_type, image, prompt_list, threshold, set(fields.split(","))
    )


async def invoke_edit(
    operation, image, media_type, params, extras=None, request_id=None
):
    attachments = [("image", png_bytes(image), media_type), *(extras or [])]
    result = await editor.invoke(operation, params, attachments, request_id=request_id)
    return image_response(rpc_image(result))


def edit_params(prompt, negative_prompt, steps, scale, seed, **extra):
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "true_cfg_scale": scale,
        "seed": seed,
        **extra,
    }


@app.post("/v1/images/edit")
async def edit_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(20),
    true_cfg_scale: float = Form(4.0),
    seed: int = Form(0),
    reference_files: list[UploadFile] | None = File(None),
    request_id: str | None = Form(None),
):
    _, media_type, image = await read_image(file)
    extras = []
    for index, reference in enumerate(reference_files or []):
        ref_payload, ref_type, _ = await read_image(reference)
        extras.append((f"reference:{index}", ref_payload, ref_type))
    params = edit_params(
        prompt, negative_prompt, num_inference_steps, true_cfg_scale, seed
    )
    return await invoke_edit(
        "edit", image, media_type, params, extras, request_id=request_id
    )


@app.post("/v1/images/inpaint")
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
    request_id: str | None = Form(None),
):
    _, media_type, image = await read_image(file)
    params = edit_params(
        prompt,
        negative_prompt,
        num_inference_steps,
        true_cfg_scale,
        seed,
        strength=strength,
        padding_mask_crop=padding_mask_crop,
    )
    return await invoke_edit(
        "inpaint",
        image,
        media_type,
        params,
        [("mask", decode_mask(mask), "image/png")],
        request_id=request_id,
    )


@app.get("/v1/images/invoke/{request_id}/progress")
async def invoke_progress(request_id: str):
    """Passthrough for the qwen-image-edit worker's per-request diffusion-step
    progress, so callers polling for inpaint/outpaint step progress don't need to
    know the worker's internal URL."""
    return await editor.invoke_progress(request_id)


@app.post("/v1/images/outpaint")
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
    request_id: str | None = Form(None),
):
    _, media_type, image = await read_image(file)
    if target_width > MAX_CANVAS_DIMENSION or target_height > MAX_CANVAS_DIMENSION:
        raise HTTPException(400, "target dimensions exceed configured maximum")
    canvas, mask = outpaint_canvas(image, target_width, target_height, anchor)
    params = edit_params(
        prompt, negative_prompt, num_inference_steps, true_cfg_scale, seed, strength=1.0
    )
    return await invoke_edit(
        "inpaint",
        canvas,
        media_type,
        params,
        [("mask", png_bytes(mask), "image/png")],
        request_id=request_id,
    )


@app.post("/v1/images/transform")
async def transform(
    file: UploadFile = File(...),
    crop_left: int | None = Form(None),
    crop_top: int | None = Form(None),
    crop_width: int | None = Form(None),
    crop_height: int | None = Form(None),
    rotate_degrees: float = Form(0.0),
    expand_canvas: bool = Form(True),
):
    _, _, image = await read_image(file)
    return image_response(transform_image(
        image,
        (crop_left, crop_top, crop_width, crop_height),
        rotate_degrees,
        expand_canvas,
    ))
