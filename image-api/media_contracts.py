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
