"""Policy-proxy client and media inference provenance."""

import base64
import io
import json
import logging
import os
from pathlib import Path

import httpx
from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from media_contracts import SCHEMA_FINGERPRINT

logger = logging.getLogger(__name__)

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
        body = response.json()
        choice = body["choices"][0]
        content = choice["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            finish_reason = choice.get("finish_reason")
            logger.warning(
                "workload=%s model=%s finish_reason=%s content_chars=%d "
                "content_tail=%r",
                workload,
                body.get("model"),
                finish_reason,
                len(content),
                content[-200:],
            )
            if finish_reason == "length":
                repaired = _repair_truncated_json(content, schema)
                if repaired is not None:
                    logger.warning(
                        "workload=%s recovered a partial result after "
                        "max_tokens truncation (fields recovered: %s)",
                        workload,
                        sorted(repaired.keys()),
                    )
                    return repaired
                raise HTTPException(
                    503,
                    {
                        "reason_code": "output_truncated",
                        "message": (
                            f"structured media analysis hit max_tokens "
                            f"before completing valid JSON: {error}"
                        ),
                    },
                    headers={"Retry-After": "5"},
                ) from error
            raise
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
        logger.warning(
            "workload=%s completion failed (%s): %s",
            workload,
            type(error).__name__,
            error,
        )
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


def _schema_default(schema: dict) -> object:
    """A minimal value satisfying `schema`'s type, for a field the model
    never reached before truncation."""
    types = schema.get("type", "null")
    types = types if isinstance(types, list) else [types]
    if "array" in types:
        return []
    if "object" in types:
        return {
            key: _schema_default(subschema)
            for key, subschema in schema.get("properties", {}).items()
        }
    if "string" in types:
        return ""
    if "integer" in types or "number" in types:
        return 0
    return None


def _find_matching_bracket(text: str, open_index: int) -> int:
    """Index of the bracket that closes the one at `open_index`, or -1."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _find_scalar_end(text: str, start: int) -> int:
    """Index just past a scalar value (string/number/bool/null) starting at
    `start`, i.e. the position of its terminating `,` or `}` — skipping
    over commas and braces that appear inside a quoted string."""
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in ",}":
            return index
    return -1


def _repair_truncated_array(remainder: str) -> list:
    """Trim a cut-off array to its last syntactically complete element."""
    items: list = []
    index = 0
    while True:
        start = min(
            (i for i in (remainder.find("{", index), remainder.find("[", index))
             if i != -1),
            default=-1,
        )
        if start == -1:
            return items
        end = _find_matching_bracket(remainder, start)
        if end == -1:
            return items
        try:
            items.append(json.loads(remainder[start : end + 1]))
        except json.JSONDecodeError:
            pass
        index = end + 1


def _locate_field_value(content: str, cursor: int, key: str) -> int | None:
    """Start index of `key`'s value in `content` at or after `cursor`, or
    None if the key/colon/value never appears (the cutoff landed before
    this field even started)."""
    key_index = content.find(f'"{key}"', cursor)
    if key_index == -1:
        return None
    colon_index = content.find(":", key_index)
    if colon_index == -1:
        return None
    value_start = colon_index + 1
    while value_start < len(content) and content[value_start] in " \t\n\r":
        value_start += 1
    if value_start >= len(content):
        return None
    return value_start


def _recover_field(
    content: str, value_start: int, subschema: dict
) -> tuple[object, int] | None:
    """`(value, cursor_after_value)` for the value at `value_start`, or None
    if it was cut off before closing (the field the truncation landed on)."""
    opener = content[value_start]
    if opener in "{[":
        close_index = _find_matching_bracket(content, value_start)
        if close_index == -1:
            remainder = content[value_start + 1 :]
            partial = (
                _repair_truncated_array(remainder)
                if opener == "["
                else _walk_object_fields(remainder, subschema)
            )
            return partial, len(content)
        value_text = content[value_start : close_index + 1]
        cursor = close_index + 1
    else:
        end_index = _find_scalar_end(content, value_start)
        if end_index == -1:
            return None
        value_text = content[value_start:end_index]
        cursor = end_index
    try:
        return json.loads(value_text), cursor
    except json.JSONDecodeError:
        return None


def _walk_object_fields(content: str, schema: dict) -> dict:
    """Recover as many of `schema`'s top-level fields as closed cleanly in
    `content` before it was cut off, in schema property order. The field
    left mid-flight at the cutoff is recovered on a best-effort basis
    (trimmed to its last complete element for an array, recursed into for
    an object); every field never reached is backfilled with a schema-typed
    default so the result always matches `schema`'s required-field shape.
    """
    properties = schema.get("properties", {})
    recovered: dict[str, object] = {}
    cursor = 0
    for key, subschema in properties.items():
        value_start = _locate_field_value(content, cursor, key)
        if value_start is None:
            break
        outcome = _recover_field(content, value_start, subschema)
        if outcome is None:
            break
        recovered[key], cursor = outcome
        if cursor == len(content):
            break
    for key, subschema in properties.items():
        if key not in recovered:
            recovered[key] = _schema_default(subschema)
    return recovered


def _repair_truncated_json(content: str, schema: dict) -> dict | None:
    """Best-effort recovery from a `finish_reason=length` cutoff.

    Strict json_schema mode means one unclosed field invalidates the whole
    document even when every field before it is complete. Returns None,
    signaling a clean failure, unless the recovered document has real
    content in at least `caption` and `summary` — a cutoff too early to
    salvage a `focus_targets` bounding box, say, is still not worth
    returning over the existing 503.
    """
    recovered = _walk_object_fields(content, schema)
    if recovered.get("caption") and recovered.get("summary"):
        return recovered
    return None


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
