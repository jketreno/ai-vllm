"""Policy-proxy client and media inference provenance."""

import base64
import io
import json
import os
from pathlib import Path

import httpx
from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from media_contracts import SCHEMA_FINGERPRINT

POLICY_URL = os.environ.get("IMAGE_API_POLICY_URL", "http://clare2-policy:8000/v1")
POLICY_TOKEN_FILE = Path(
    os.environ.get(
        "IMAGE_API_POLICY_TOKEN_FILE", "/run/secrets/clare2_proxy_token"
    )
)
VISION_MODEL = os.environ.get("IMAGE_API_VISION_MODEL", "Qwen/Qwen3.6-27B-FP8")
INFERENCE_TIMEOUT_SECONDS = float(
    os.environ.get("IMAGE_API_INFERENCE_TIMEOUT_SECONDS", "900")
)
MODEL_REVISION = os.environ.get("CLARE2_INFERENCE_REVISION", "configured")
MAX_UPLOAD_BYTES = int(
    os.environ.get("IMAGE_API_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))
)
# Long-edge target for the vision model's input image, aligned to the Qwen
# ViT 28px patch grid (56 x 28 = 1568). 0 disables downscaling entirely.
VISION_MAX_EDGE = int(os.environ.get("IMAGE_API_VISION_MAX_EDGE", "1568"))
VISION_JPEG_QUALITY = int(os.environ.get("IMAGE_API_VISION_JPEG_QUALITY", "88"))
PROMPT_VERSION = "phai-media-v1"
CONTRACT_VERSION = "0.2.0"


def _policy_token() -> str:
    try:
        return POLICY_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise HTTPException(
            503, "media analysis policy token is unavailable"
        ) from error


def image_content(payload: bytes, media_type: str) -> dict:
    encoded = base64.b64encode(payload).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def _downscale_for_vision(payload: bytes, media_type: str) -> tuple[bytes, str]:
    if VISION_MAX_EDGE <= 0:
        return payload, media_type
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image = ImageOps.exif_transpose(image)
            if max(image.size) <= VISION_MAX_EDGE:
                return payload, media_type
            image.thumbnail((VISION_MAX_EDGE, VISION_MAX_EDGE), Image.LANCZOS)
            buffer = io.BytesIO()
            if media_type == "image/png":
                image.save(buffer, format="PNG", optimize=True)
                return buffer.getvalue(), media_type
            image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=VISION_JPEG_QUALITY)
            return buffer.getvalue(), "image/jpeg"
    except OSError:
        return payload, media_type


async def read_image(upload: UploadFile) -> tuple[bytes, str]:
    media_type = upload.content_type or ""
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "semantic inputs must be JPEG, PNG, or WebP")
    payload = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "semantic input exceeds configured limit")
    return _downscale_for_vision(payload, media_type)


async def completion(
    prompt: str,
    schema: dict,
    content: list[dict] | None = None,
    max_tokens: int = 2500,
    workload: str = "default",
) -> dict:
    request = {
        "model": VISION_MODEL,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "phai_response",
                "strict": True,
                "schema": schema,
            },
        },
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *(content or [])],
            }
        ],
    }
    try:
        timeout = httpx.Timeout(
            connect=10,
            read=INFERENCE_TIMEOUT_SECONDS,
            write=120,
            pool=10,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{POLICY_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {_policy_token()}",
                    "X-Inference-Workload": workload,
                },
                json=request,
            )
        if response.status_code == 429:
            raise HTTPException(
                429,
                "inference capacity saturated",
                headers={
                    "Retry-After": response.headers.get("Retry-After", "30")
                },
            )
        if 400 <= response.status_code < 500:
            raise HTTPException(
                response.status_code,
                f"inference request rejected: {response.text[:1000]}",
            )
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])
    except httpx.TimeoutException as error:
        raise HTTPException(
            503,
            f"structured media analysis timed out: {error}",
            headers={"Retry-After": "60"},
        ) from error
    except (
        OSError,
        httpx.HTTPError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise HTTPException(
            503,
            f"structured media analysis unavailable: {error}",
            headers={"Retry-After": "60"},
        ) from error


async def capacity_response(workload: str) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{POLICY_URL}/capacity",
                headers={
                    "Authorization": f"Bearer {_policy_token()}",
                    "X-Inference-Workload": workload,
                },
            )
        payload = response.json()
        retry_after = response.headers.get(
            "Retry-After", str(payload.get("retry_after", 30))
        )
        return JSONResponse(
            payload,
            status_code=response.status_code,
            headers={"Retry-After": retry_after}
            if response.status_code == 429
            else {},
        )
    except (OSError, ValueError, httpx.HTTPError) as error:
        return JSONResponse(
            {"available": False, "detail": str(error), "retry_after": 60},
            status_code=503,
            headers={"Retry-After": "60"},
        )


def with_provenance(result: dict, prompt_version: str = PROMPT_VERSION) -> dict:
    return {
        **result,
        "contract_version": CONTRACT_VERSION,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "schema_version": "1",
        "model": VISION_MODEL,
        "model_revision": MODEL_REVISION,
        "prompt_version": prompt_version,
    }
