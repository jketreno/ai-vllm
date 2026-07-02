"""Helpers for parsing JSON returned by local LLM prompts."""

from __future__ import annotations

import json
from typing import Any


def parse_json_output(text: str | None) -> Any:
    """Parse a JSON value, tolerating common model wrappers around it."""
    if not isinstance(text, str):
        raise json.JSONDecodeError("expected JSON text", "", 0)
    cleaned = _strip_fence(text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return _decode_embedded_json(cleaned)


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if not lines:
        return text
    if lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def _decode_embedded_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise json.JSONDecodeError("no JSON object or array found", text, 0)
