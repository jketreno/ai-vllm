"""Local Qwen3.5 generation through the private vLLM service."""

from __future__ import annotations

import os

import httpx

from .security import secret_value

MODEL = os.environ.get("CLARE2_DISTILL_MODEL", "Qwen/Qwen3.6-27B-FP8")
POLICY_URL = os.environ.get(
    "CLARE2_POLICY_INFERENCE_URL", "http://localhost:8000"
)


def generate(prompt: str, *, max_tokens: int = 4096) -> str:
    response = httpx.post(
        f"{POLICY_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {secret_value('CLARE2_PROXY_TOKEN')}",
            "X-Inference-Workload": "distillation",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=900,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"].get("content") or ""
