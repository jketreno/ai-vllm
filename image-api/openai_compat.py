"""OpenAI-compatible image endpoints, translating to this service's native
/v1/images/* operations. Covers the subset open-webui's OpenAI image backend
uses: POST /v1/images/generations and POST /v1/images/edits, both returning
b64_json. There is no text-to-image model behind this service, so
generations seeds the edit pipeline with a random-noise canvas.
"""

import io
import re
import time
import uuid

from fastapi import APIRouter, Body, Header, HTTPException, Request
from starlette.datastructures import UploadFile
from PIL import Image, UnidentifiedImageError

from image_ops import noise_canvas

router = APIRouter(prefix="/openai/v1")

DEFAULT_SIZE = (1024, 1024)
_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")


def _parse_size(size: str | None, max_dimension: int) -> tuple[int, int]:
    if not size or size == "auto":
        width, height = DEFAULT_SIZE
    else:
        match = _SIZE_RE.match(size)
        if not match:
            raise HTTPException(400, "size must be formatted as WIDTHxHEIGHT")
        width, height = int(match.group(1)), int(match.group(2))
    if width > max_dimension or height > max_dimension:
        raise HTTPException(400, "requested size exceeds configured maximum")
    return width, height


def _require_b64_json(response_format: str | None):
    if response_format and response_format != "b64_json":
        raise HTTPException(
            400, "only response_format=b64_json is supported by this backend"
        )


def _openai_response(images_b64: list[str]) -> dict:
    return {
        "created": int(time.time()),
        "data": [{"b64_json": image} for image in images_b64],
    }


def register(app, editor_invoke_edit, edit_params, max_canvas_dimension: int):
    """Mount the OpenAI-compatible router. Takes the host app's invoke_edit
    coroutine and edit_params builder so this module shares the same
    worker-invocation path (leasing, logging, response shaping) as the native
    /v1/images/* routes rather than re-implementing it."""

    @router.post("/images/generations")
    async def generations(
        prompt: str = Body(..., embed=True),
        n: int = Body(1, embed=True),
        size: str | None = Body(None, embed=True),
        response_format: str | None = Body(None, embed=True),
        negative_prompt: str = Body("", embed=True),
        _authorization: str | None = Header(None, alias="Authorization"),
    ):
        _require_b64_json(response_format)
        if n != 1:
            raise HTTPException(400, "only n=1 is supported by this backend")
        width, height = _parse_size(size, max_canvas_dimension)
        canvas = noise_canvas(width, height)
        params = edit_params(prompt, negative_prompt, 20, 4.0, 0)
        result = await editor_invoke_edit(
            "edit", canvas, "image/png", params, request_id=str(uuid.uuid4())
        )
        return _openai_response([result["image_png_base64"]])

    @router.post("/images/edits")
    async def edits(
        request: Request,
        _authorization: str | None = Header(None, alias="Authorization"),
    ):
        form = await request.form()
        images = [
            value
            for key in ("image[]", "image")
            for value in form.getlist(key)
            if isinstance(value, UploadFile)
        ]
        if not images:
            raise HTTPException(422, "image is required")
        prompt = form.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(422, "prompt is required")
        n = int(form.get("n", 1))
        response_format = form.get("response_format") or None

        _require_b64_json(response_format)
        if n != 1:
            raise HTTPException(400, "only n=1 is supported by this backend")

        loaded = []
        for upload in images:
            media_type = upload.content_type or ""
            if media_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(415, "upload a JPEG, PNG, or WebP image")
            payload = await upload.read()
            try:
                source_image = Image.open(io.BytesIO(payload)).convert("RGB")
            except (UnidentifiedImageError, OSError) as error:
                raise HTTPException(400, "invalid image") from error
            loaded.append((source_image, media_type, payload))

        source, media_type, _ = loaded[0]
        extras = [
            (f"reference:{index}", payload, ref_media_type)
            for index, (_, ref_media_type, payload) in enumerate(loaded[1:])
        ]
        params = edit_params(prompt, "", 20, 4.0, 0)
        result = await editor_invoke_edit(
            "edit", source, media_type, params, extras, request_id=str(uuid.uuid4())
        )
        return _openai_response([result["image_png_base64"]])

    app.include_router(router)
