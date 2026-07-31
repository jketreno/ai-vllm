"""Structured media request models and JSON schemas."""

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

SEMANTIC_OBSERVATION_SCHEMA = {
    **OBSERVATION_SCHEMA,
    "properties": {
        **OBSERVATION_SCHEMA["properties"],
        "type": {"type": "string", "enum": list(SEMANTIC_OBSERVATION_TYPES)},
    },
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
            "maxItems": 10,
        },
        "spatial_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
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

FOCUS_TARGET_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "pattern": "^focus-[1-8]$"},
        "priority": {"type": "integer", "minimum": 1, "maximum": 8},
        "display_label": {"type": "string", "minLength": 1, "maxLength": 160},
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
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    },
    "required": [
        "id",
        "priority",
        "display_label",
        "sam_prompt",
        "role",
        "subject_type",
        "extent",
        "segmentability",
        "confidence",
        "location",
        "reason",
    ],
}

SEMANTIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "caption": {"type": "string"},
        "concise_caption": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 100},
        "focus_targets": {
            "type": "array",
            "items": FOCUS_TARGET_SCHEMA,
            "maxItems": 8,
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
            "items": SEMANTIC_OBSERVATION_SCHEMA,
            "maxItems": 100,
        },
    },
    "required": [
        "caption",
        "concise_caption",
        "confidence",
        "concepts",
        "focus_targets",
        "visible_text",
        "uncertainties",
        "observations",
    ],
}

SEMANTIC_REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "caption": {"type": "string"},
        "concise_caption": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 100},
        "focus_targets": {
            "type": "array",
            "items": FOCUS_TARGET_SCHEMA,
            "maxItems": 8,
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
            "items": SEMANTIC_OBSERVATION_SCHEMA,
            "maxItems": 100,
        },
        "summary": {"type": "string"},
        "concise_summary": {"type": "string"},
        "known_facts": {"type": "array", "items": {"type": "string"}},
        "inferences": {"type": "array", "items": {"type": "string"}},
        "evidence_types": {"type": "array", "items": {"type": "string"}},
        "place_resolution": PLACE_RESOLUTION_SCHEMA,
    },
    "required": [
        "caption",
        "concise_caption",
        "confidence",
        "concepts",
        "focus_targets",
        "visible_text",
        "uncertainties",
        "observations",
        "summary",
        "concise_summary",
        "known_facts",
        "inferences",
        "evidence_types",
        "place_resolution",
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
