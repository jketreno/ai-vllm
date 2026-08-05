"""Structured media request models and JSON schemas."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class EvidenceRequest(BaseModel):
    asset: dict[str, Any]
    observations: list[dict[str, Any]] = Field(max_length=500)


class ValuesRequest(BaseModel):
    values: list[str] = Field(min_length=1, max_length=500)


class IdentityCaptionRequest(BaseModel):
    neutral_caption: str = Field(min_length=1, max_length=10000)
    identities: list[dict[str, Any]] = Field(min_length=1, max_length=100)


class ContextReportRequest(BaseModel):
    asset: dict[str, Any]
    visual_semantics: dict[str, Any]
    observations: list[dict[str, Any]] = Field(max_length=128)
    accepted_identities: list[dict[str, Any]] = Field(max_length=100)
    location_override: dict[str, Any] | None = None


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

# Fixed vocabulary for semantic-image/semantic-window per-observation `type`.
# Without an enum here the model paraphrases the prompt's category names
# (e.g. "location_event_clue" vs "location/event clues" vs "location_event")
# and every variant is persisted verbatim as a distinct observation_type.
SEMANTIC_OBSERVATION_TYPES = (
    "scene",
    "activity",
    "object",
    "visible_relationship",
    "visible_text",
    "location_event_clue",
    "photographic_role",
    "narrative_role",
)

# Standalone (not spread over OBSERVATION_SCHEMA, which /event-inference still
# uses): drops the unbounded `evidence` array, which no consumer reads.
SEMANTIC_OBSERVATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "enum": list(SEMANTIC_OBSERVATION_TYPES)},
        "start_us": {"type": ["integer", "null"]},
        "end_us": {"type": ["integer", "null"]},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "summary": {"type": "string", "maxLength": 240},
    },
    "required": ["type", "start_us", "end_us", "confidence", "summary"],
}

PLACE_RESOLUTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {
            "type": "string",
            "enum": ["none", "possible", "resolved"],
        },
        "candidate_id": {"type": ["string", "null"]},
        "name": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "visual_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "spatial_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
    },
    "required": [
        "status",
        "candidate_id",
        "name",
        "confidence",
        "visual_evidence",
        "spatial_evidence",
    ],
}

BOX_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "x": {"type": "number", "minimum": 0, "maximum": 1},
        "y": {"type": "number", "minimum": 0, "maximum": 1},
        "w": {"type": "number", "minimum": 0, "maximum": 1},
        "h": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["x", "y", "w", "h"],
}

# `id` and `priority` are intentionally absent: _normalize_focus_plan assigns
# both server-side from array order, so asking the model for them is pure
# ceremony (see commit 3f3ade7 for the precedent of dropping model-facing
# fields that duplicate server-derived output).
FOCUS_TARGET_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "display_label": {"type": "string", "minLength": 1, "maxLength": 120},
        "sam_prompt": {
            "type": ["string", "null"],
            "minLength": 1,
            "maxLength": 80,
        },
        "role": {
            "type": "string",
            "enum": ["primary", "supporting", "context"],
        },
        "subject_type": {
            "type": "string",
            "enum": [
                "person",
                "group",
                "animal",
                "food",
                "vehicle",
                "object",
                "structure",
                "landscape",
                "text",
                "other",
            ],
        },
        "extent": {
            "type": "string",
            "enum": ["whole_subject", "group", "detail", "region"],
        },
        "segmentability": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "location": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "horizontal": {
                    "type": "string",
                    "enum": ["left", "center", "right", "full"],
                },
                "vertical": {
                    "type": "string",
                    "enum": ["top", "center", "bottom", "full"],
                },
            },
            "required": ["horizontal", "vertical"],
        },
        "box": BOX_SCHEMA,
        "gaze_direction": {
            "type": "string",
            "enum": [
                "toward_camera",
                "left",
                "right",
                "up",
                "down",
                "away",
                "not_applicable",
            ],
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 120},
    },
    "required": [
        "display_label",
        "sam_prompt",
        "role",
        "subject_type",
        "extent",
        "segmentability",
        "confidence",
        "location",
        "box",
        "gaze_direction",
        "reason",
    ],
}

NARRATIVE_ROLES = (
    "establishing",
    "detail",
    "portrait",
    "action",
    "transition",
    "climax",
    "closing",
)

SCALES = ("wide", "medium", "close", "detail")

SEMANTIC_REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "caption": {"type": "string", "maxLength": 700},
        "concise_caption": {"type": "string", "maxLength": 200},
        "narrative_role": {"type": "string", "enum": list(NARRATIVE_ROLES)},
        "scale": {"type": "string", "enum": list(SCALES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "concepts": {
            "type": "array",
            "items": {"type": "string", "maxLength": 64},
            "maxItems": 24,
        },
        "focus_targets": {
            "type": "array",
            "items": FOCUS_TARGET_SCHEMA,
            "maxItems": 8,
        },
        "visible_text": {
            "type": "array",
            "items": {"type": "string", "maxLength": 160},
            "maxItems": 32,
        },
        "observations": {
            "type": "array",
            "items": SEMANTIC_OBSERVATION_SCHEMA,
            "maxItems": 24,
        },
        "summary": {"type": "string", "maxLength": 1500},
        "concise_summary": {"type": "string", "maxLength": 300},
        "known_facts": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 16,
        },
        "inferences": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 16,
        },
        "evidence_types": {
            "type": "array",
            "items": {"type": "string", "maxLength": 40},
            "maxItems": 12,
        },
        "place_resolution": PLACE_RESOLUTION_SCHEMA,
    },
    "required": [
        "caption",
        "concise_caption",
        "narrative_role",
        "scale",
        "confidence",
        "concepts",
        "focus_targets",
        "visible_text",
        "observations",
        "summary",
        "concise_summary",
        "known_facts",
        "inferences",
        "evidence_types",
        "place_resolution",
    ],
}

_VISUAL_SEMANTIC_FIELDS = (
    "caption",
    "concise_caption",
    "narrative_role",
    "scale",
    "confidence",
    "concepts",
    "focus_targets",
    "visible_text",
    "observations",
)
VISUAL_SEMANTIC_SCHEMA = {
    **SEMANTIC_REPORT_SCHEMA,
    "properties": {
        key: SEMANTIC_REPORT_SCHEMA["properties"][key]
        for key in _VISUAL_SEMANTIC_FIELDS
    },
    "required": list(_VISUAL_SEMANTIC_FIELDS),
}

_CONTEXT_REPORT_FIELDS = (
    "summary",
    "concise_summary",
    "known_facts",
    "inferences",
    "evidence_types",
    "place_resolution",
)
CONTEXT_REPORT_SCHEMA = {
    **SEMANTIC_REPORT_SCHEMA,
    "properties": {
        key: SEMANTIC_REPORT_SCHEMA["properties"][key]
        for key in _CONTEXT_REPORT_FIELDS
    },
    "required": list(_CONTEXT_REPORT_FIELDS),
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


def schema_fingerprint() -> str:
    """Stable hash over model-facing schemas, recomputed on every import.

    Replaces a hand-maintained literal that could silently go stale after
    a schema edit.
    """
    payload = json.dumps(
        {
            "semantic_report": SEMANTIC_REPORT_SCHEMA,
            "visual_semantic": VISUAL_SEMANTIC_SCHEMA,
            "context_report": CONTEXT_REPORT_SCHEMA,
            "observation": OBSERVATION_SCHEMA,
            "query": QUERY_SCHEMA,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SCHEMA_FINGERPRINT = schema_fingerprint()
