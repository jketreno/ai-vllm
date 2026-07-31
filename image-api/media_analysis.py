"""Structured, archive-agnostic media analysis operations for ketr.phai."""

import json
import re
from typing import Literal

import httpx
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from media_contracts import (
    EvidenceRequest,
    IdentityCaptionRequest,
    OBSERVATION_SCHEMA,
    QUERY_SCHEMA,
    SEMANTIC_OBSERVATION_TYPES,
    SEMANTIC_REPORT_SCHEMA,
    ValuesRequest,
)
from media_inference import (
    CONTRACT_VERSION,
    INFERENCE_TIMEOUT_SECONDS,
    MODEL_REVISION,
    POLICY_URL,
    PROMPT_VERSION,
    SCHEMA_FINGERPRINT,
    VISION_MODEL,
    _policy_token,
    capacity_response,
    completion as _completion,
    image_content as _image_content,
    read_image as _read_image,
    with_provenance as _with_provenance,
)

router = APIRouter(prefix="/v1/media", tags=["media intelligence"])
MAX_WINDOW_FRAMES = 12
REPORT_PROMPT_VERSION = "phai-report-v3"


@router.get("/capacity/{workload}")
async def inference_capacity(
    workload: Literal["semantic_report"],
) -> JSONResponse:
    return await capacity_response(workload)


def _without_speech_capability_claims(value: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", value.strip())
    capability_terms = ("transcript", "diarization", "speaker separation")
    retained = [
        sentence
        for sentence in sentences
        if not any(term in sentence.lower() for term in capability_terms)
    ]
    return " ".join(retained).strip()


def _normalize_speech_capabilities(
    result: dict, observations: list[dict]
) -> dict:
    speech_status = next(
        (
            item.get("payload", {})
            for item in observations
            if item.get("observation_type") == "speech_status"
        ),
        {},
    )
    has_transcript = any(
        item.get("observation_type") == "transcript_segment"
        for item in observations
    )
    if speech_status.get("transcription") != "available" or has_transcript:
        return result

    normalized = dict(result)
    canonical = [
        "Transcription ran, but no transcript text segments were produced."
    ]
    if speech_status.get("diarization") == "unavailable":
        canonical.append(
            "Anonymous speaker separation was unavailable and did not affect "
            "transcription."
        )
    base_summary = _without_speech_capability_claims(
        str(normalized.get("summary", ""))
    )
    normalized["summary"] = " ".join([base_summary, *canonical]).strip()
    normalized["concise_summary"] = " ".join(canonical)
    capability_terms = ("transcript", "diarization", "speaker separation")
    for field in ("known_facts", "inferences", "uncertainties"):
        retained = [
            item
            for item in normalized.get(field, [])
            if not any(term in item.lower() for term in capability_terms)
        ]
        normalized[field] = retained
    normalized["known_facts"].extend(canonical)
    return normalized


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
        "inference_timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
        "max_window_frames": MAX_WINDOW_FRAMES,
        "ready": ready,
        "operations": [
            "semantic_image_v1",
            "semantic_window_v1",
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


_REPORT_SYNTHESIS_INSTRUCTIONS = (
    "In addition to the visual analysis above, synthesize a media report "
    "using both the image and the supplied prior evidence. Classify "
    "metadata/direct observations as known facts, keep model interpretations "
    "under inferences, preserve contradictions and uncertainty, and never "
    "invent names, dates, places, or events. Diarization availability never "
    "determines transcription availability; diarization only assigns "
    "anonymous speaker labels. If transcription is available but there are "
    "no transcript_segment observations, state that no transcript text was "
    "produced and do not attribute that to diarization."
)


def _parse_observations(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(400, "observations must be a JSON array") from error
    if not isinstance(parsed, list):
        raise HTTPException(400, "observations must be a JSON array")
    return parsed


def _evidence_prompt(observations: list[dict]) -> str:
    if not observations:
        return f"{_REPORT_SYNTHESIS_INSTRUCTIONS} No prior evidence was supplied."
    serialized = json.dumps(observations[:200])
    if len(serialized) > 1_000_000:
        raise HTTPException(413, "evidence request is too large")
    return f"{_REPORT_SYNTHESIS_INSTRUCTIONS} Evidence JSON: {serialized}"


@router.post("/semantic-image")
async def semantic_image(
    file: UploadFile = File(...),
    observations: str | None = Form(default=None),
):
    payload, media_type = await _read_image(file)
    parsed_observations = _parse_observations(observations)
    prompt = (
        "Analyze only what is supportable from this image. Do not identify real "
        "people by appearance. Separate direct visual evidence from uncertain "
        "interpretation. For each entry in `observations`, set `type` to exactly "
        f"one of: {', '.join(SEMANTIC_OBSERVATION_TYPES)}. Also produce concise SAM "
        "segmentation prompts. Times must be null for a still image. "
        f"{_evidence_prompt(parsed_observations)}"
    )
    result = await _completion(
        prompt,
        SEMANTIC_REPORT_SCHEMA,
        [_image_content(payload, media_type)],
        max_tokens=4000,
        workload="semantic_report",
    )
    result = _normalize_speech_capabilities(result, parsed_observations)
    return _with_provenance(result, REPORT_PROMPT_VERSION)


@router.post("/semantic-window")
async def semantic_window(
    frames: list[UploadFile] = File(...),
    timestamps_us: str = Form(...),
    transcript: str | None = Form(default=None),
    observations: str | None = Form(default=None),
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
    parsed_observations = _parse_observations(observations)
    prompt = (
        "Analyze these chronological video frames at timestamps "
        f"{timestamps}. Return time-bounded direct observations and uncertainties. "
        "Do not infer real-world identity from appearance. For each entry in "
        f"`observations`, set `type` to exactly one of: "
        f"{', '.join(SEMANTIC_OBSERVATION_TYPES)}. "
        f"{_evidence_prompt(parsed_observations)}"
    )
    if transcript:
        prompt += f" Timestamped transcript context: {transcript[:12000]}"
    result = await _completion(
        prompt,
        SEMANTIC_REPORT_SCHEMA,
        images,
        max_tokens=5000,
        workload="semantic_report",
    )
    result = _normalize_speech_capabilities(result, parsed_observations)
    return _with_provenance(result, REPORT_PROMPT_VERSION)


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
