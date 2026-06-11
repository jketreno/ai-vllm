"""Local Qwen3.5 generation through the private vLLM service."""

from __future__ import annotations

import os

import httpx

MODEL = os.environ.get("CLARE2_DISTILL_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
VLLM_URL = os.environ.get("CLARE2_VLLM_URL", "http://vllm-engine:8001")


def generate(prompt: str, *, max_tokens: int = 4096) -> str:
    response = httpx.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": max_tokens,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
