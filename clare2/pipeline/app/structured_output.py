"""Schema validation and one-shot repair for distillation outputs."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .json_output import parse_json_output

PatternCategory = Literal["style", "architecture", "antipattern", "domain"]
ParseOutcome = Literal["valid_first_try", "repaired", "failed"]


class PatternRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: PatternCategory
    pattern: str = Field(min_length=1)
    evidence_count: int = Field(ge=1)
    canonical_example: str = Field(min_length=1)
    first_seen: str = Field(min_length=1)
    last_seen: str = Field(min_length=1)


def parse_pattern_records_with_repair(
    text: str | None,
    repair: Callable[[str], str],
) -> tuple[list[dict[str, Any]], ParseOutcome]:
    try:
        return _validate_pattern_records(parse_json_output(text)), "valid_first_try"
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        try:
            repaired = repair(str(exc))
        except Exception:
            return [], "failed"

    try:
        return _validate_pattern_records(parse_json_output(repaired)), "repaired"
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        return [], "failed"


def repair_prompt(raw_output: str | None, error: str) -> str:
    schema = json.dumps(PatternRecord.model_json_schema(), indent=2)
    return (
        "Repair this LLM response so it is valid JSON matching the schema.\n"
        "Return ONLY a JSON array. Do not include markdown, prose, or code fences.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Validation error:\n{error}\n\n"
        f"Original response:\n{raw_output or ''}"
    )


def _validate_pattern_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "patterns" in value:
        value = value["patterns"]
    if not isinstance(value, list):
        raise TypeError("expected a JSON array or object with a patterns array")
    return [PatternRecord.model_validate(item).model_dump() for item in value]
