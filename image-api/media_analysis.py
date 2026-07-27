"""Structured, archive-agnostic media analysis operations for ketr.phai."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/media", tags=["media intelligence"])
POLICY_URL = os.environ.get("IMAGE_API_POLICY_URL", "http://clare2-policy:8000/v1")
POLICY_TOKEN_FILE = Path(
    os.environ.get(
        "IMAGE_API_POLICY_TOKEN_FILE", "/run/secrets/clare2_proxy_token"
    )
)
VISION_MODEL = os.environ.get("IMAGE_API_VISION_MODEL", "Qwen/Qwen3.6-27B-FP8")
MODEL_REVISION = os.environ.get("CLARE2_INFERENCE_REVISION", "configured")
MAX_UPLOAD_BYTES = int(
    os.environ.get("IMAGE_API_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))
)
MAX_WINDOW_FRAMES = 12
PROMPT_VERSION = "phai-media-v1"
CONTRACT_VERSION = "0.1.0"
SCHEMA_FINGERPRINT = (
    "375289189b519a6590658764228b45bf7ecdb4002a7ac4f03cd008843ae51c69"
)


class EvidenceRequest(BaseModel):
    asset: dict[str, Any]
    observations: list[dict[str, Any]] = Field(max_length=500)


class ValuesRequest(BaseModel):
    values: list[str] = Field(min_length=1, max_length=500)


class IdentityCaptionRequest(BaseModel):
    neutral_caption: str = Field(min_length=1, max_length=10000)
    identities: list[dict[str, Any]] = Field(min_length=1, max_length=100)


def _policy_token() -> str:
    try:
        return POLICY_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise HTTPException(
            503, "media analysis policy token is unavailable"
        ) from error


def _image_content(payload: bytes, media_type: str) -> dict:
    encoded = base64.b64encode(payload).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


async def _read_image(upload: UploadFile) -> tuple[bytes, str]:
    media_type = upload.content_type or ""
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "semantic inputs must be JPEG, PNG, or WebP")
    payload = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "semantic input exceeds configured limit")
    return payload, media_type


OBSERVATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string"},
        "start_us": {"type": ["integer", "null"]},
        "end_us": {"type": ["integer", "null"]},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["type", "start_us", "end_us", "confidence", "summary", "evidence"],
}

SEMANTIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "caption": {"type": "string"},
        "concise_caption": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 100},
        "sam_prompts": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 24,
        },
        "visible_text": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 100,
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 50,
        },
        "observations": {
            "type": "array",
            "items": OBSERVATION_SCHEMA,
            "maxItems": 100,
        },
    },
    "required": [
        "caption",
        "concise_caption",
        "confidence",
        "concepts",
        "sam_prompts",
        "visible_text",
        "uncertainties",
        "observations",
    ],
}

REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "concise_summary": {"type": "string"},
        "known_facts": {"type": "array", "items": {"type": "string"}},
        "inferences": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "evidence_types": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary",
        "concise_summary",
        "known_facts",
        "inferences",
        "uncertainties",
        "evidence_types",
    ],
}

QUERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "media_types": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "events": {"type": "array", "items": {"type": "string"}},
        "places": {"type": "array", "items": {"type": "string"}},
        "concepts": {"type": "array", "items": {"type": "string"}},
        "date_text": {"type": ["string", "null"]},
        "semantic_query": {"type": ["string", "null"]},
        "unresolved": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "media_types",
        "people",
        "events",
        "places",
        "concepts",
        "date_text",
        "semantic_query",
        "unresolved",
    ],
}


async def _completion(
    prompt: str,
    schema: dict,
    content: list[dict] | None = None,
    max_tokens: int = 2500,
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
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{POLICY_URL}/chat/completions",
                headers={"Authorization": f"Bearer {_policy_token()}"},
                json=request,
            )
        response.raise_for_status()
        result = json.loads(response.json()["choices"][0]["message"]["content"])
    except (
        OSError,
        httpx.HTTPError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise HTTPException(
            503, f"structured media analysis unavailable: {error}"
        ) from error
    return result


def _with_provenance(result: dict) -> dict:
    return {
        **result,
        "contract_version": CONTRACT_VERSION,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "schema_version": "1",
        "model": VISION_MODEL,
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
    }


@router.get("/capabilities")
async def capabilities():
    return capability_document(await _media_ready())


def capability_document(ready: bool) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "schema_version": "1",
        "model": VISION_MODEL,
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "max_window_frames": MAX_WINDOW_FRAMES,
        "ready": ready,
        "operations": [
            "semantic_image_v1",
            "semantic_window_v1",
            "report_synthesis_v1",
            "taxonomy_normalize_v1",
            "event_inference_v1",
            "query_plan_v1",
            "identity_caption_v1",
        ],
    }


async def _media_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{POLICY_URL.rsplit('/v1', 1)[0]}/health")
        return response.is_success and bool(_policy_token())
    except (OSError, httpx.HTTPError):
        return False


@router.post("/semantic-image")
async def semantic_image(file: UploadFile = File(...)):
    payload, media_type = await _read_image(file)
    prompt = (
        "Analyze only what is supportable from this image. Do not identify real "
        "people by appearance. Separate direct visual evidence from uncertain "
        "interpretation. Extract objects, activities, scene, visible relationships, "
        "visible text, location/event clues, photographic role, and concise SAM "
        "segmentation prompts. Times must be null for a still image."
    )
    result = await _completion(
        prompt, SEMANTIC_SCHEMA, [_image_content(payload, media_type)]
    )
    return _with_provenance(result)


@router.post("/semantic-window")
async def semantic_window(
    frames: list[UploadFile] = File(...),
    timestamps_us: str = Form(...),
    transcript: str | None = Form(default=None),
):
    if not frames or len(frames) > MAX_WINDOW_FRAMES:
        raise HTTPException(400, f"provide 1-{MAX_WINDOW_FRAMES} frames")
    try:
        timestamps = json.loads(timestamps_us)
    except json.JSONDecodeError as error:
        raise HTTPException(400, "timestamps_us must be a JSON array") from error
    if not isinstance(timestamps, list) or len(timestamps) != len(frames):
        raise HTTPException(400, "timestamps must correspond to every frame")
    images = []
    for frame in frames:
        payload, media_type = await _read_image(frame)
        images.append(_image_content(payload, media_type))
    prompt = (
        "Analyze these chronological video frames at timestamps "
        f"{timestamps}. Return time-bounded direct observations and uncertainties. "
        "Do not infer real-world identity from appearance. Cover scenes, activities, "
        "objects, visible relationships, text, event/place clues, and narrative role."
    )
    if transcript:
        prompt += f" Timestamped transcript context: {transcript[:12000]}"
    result = await _completion(prompt, SEMANTIC_SCHEMA, images, max_tokens=3500)
    return _with_provenance(result)


@router.post("/report-synthesis")
async def report_synthesis(request: EvidenceRequest):
    serialized = request.model_dump_json()
    if len(serialized) > 1_000_000:
        raise HTTPException(413, "evidence request is too large")
    prompt = (
        "Synthesize a media report using only the supplied evidence. Classify "
        "metadata/direct observations as known facts, keep model interpretations "
        "under inferences, preserve contradictions and uncertainty, and never invent "
        f"names, dates, places, or events. Evidence JSON: {serialized}"
    )
    result = await _completion(prompt, REPORT_SCHEMA, max_tokens=2500)
    return _with_provenance(result)


@router.post("/taxonomy-normalize")
async def taxonomy_normalize(request: ValuesRequest):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "raw": {"type": "string"},
                        "canonical": {"type": "string"},
                        "parent": {"type": ["string", "null"]},
                    },
                    "required": ["raw", "canonical", "parent"],
                },
            }
        },
        "required": ["concepts"],
    }
    prompt = (
        "Normalize these visible-media labels without adding facts: "
        f"{request.values}"
    )
    return _with_provenance(await _completion(prompt, schema))


@router.post("/event-inference")
async def event_inference(request: EvidenceRequest):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "events": {"type": "array", "items": OBSERVATION_SCHEMA},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["events", "uncertainties"],
    }
    prompt = (
        "Propose candidate events using only time, place, people, and semantic "
        f"evidence. Never confirm an event. {request.model_dump_json()}"
    )
    return _with_provenance(await _completion(prompt, schema))


@router.post("/query-plan")
async def query_plan(query: str = Body(embed=True, min_length=1, max_length=10000)):
    prompt = (
        "Convert this media search into unresolved entity names and typed constraints. "
        f"Do not invent IDs or silently resolve ambiguity. Query: {query}"
    )
    return _with_provenance(await _completion(prompt, QUERY_SCHEMA))


@router.post("/identity-caption")
async def identity_caption(request: IdentityCaptionRequest):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"caption": {"type": "string"}},
        "required": ["caption"],
    }
    prompt = (
        "Insert only supplied confirmed identity names into the neutral caption. "
        f"Caption: {request.neutral_caption}; identities: {request.identities}"
    )
    return _with_provenance(await _completion(prompt, schema))
