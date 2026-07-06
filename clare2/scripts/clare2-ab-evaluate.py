#!/usr/bin/env python3
"""Run a small CLARE2 base-vs-routed-LoRA prompt evaluation."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


PROMPTS = [
    {
        "id": "summary-path",
        "prompt": "Where should CLARE2 weekly summaries for ai-vllm be written?",
        "expected": ["summaries/weekly/ai-vllm", "2026-W"],
    },
    {
        "id": "corpus-mount",
        "prompt": "A Docker Compose CLARE2 corpus mount points to ./corpus. Is that correct?",
        "expected": ["CLARE2_CORPUS_ROOT", "/corpus"],
    },
    {
        "id": "humans-only",
        "prompt": "Before editing clare/scripts/clare2-capture-event.sh, what should an agent do?",
        "expected": ["humans-only", "autonomy.yml"],
    },
    {
        "id": "fingerprint",
        "prompt": "Why would an adapter fail with 'adapter base fingerprint mismatch: model_id'?",
        "expected": ["base", "model_id"],
    },
    {
        "id": "verify-ci",
        "prompt": "What command must be run after modifying files in this repo?",
        "expected": ["./clare/verify-ci.sh"],
    },
]


@dataclass(frozen=True)
class Config:
    base_url: str
    proxy_token: str
    internal_token: str
    operator_token: str
    project: str
    task_kind: str
    capabilities: list[str]
    max_tokens: int
    timeout: int


def _read_secret(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {detail}") from exc


def _chat(config: Config, prompt: str, route_id: str | None = None) -> str:
    headers = {
        "Authorization": f"Bearer {config.proxy_token}",
        "Content-Type": "application/json",
    }
    if route_id:
        headers["X-CLARE-Route-ID"] = route_id
    payload = {
        "model": "ignored-by-clare2-proxy",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": config.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{config.base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"chat failed: HTTP {exc.code}: {detail}") from exc
    content = body["choices"][0]["message"].get("content") or ""
    return content.strip() or f"[empty response after {time.monotonic() - started:.3f}s]"


def _score(text: str, expected: list[str]) -> dict[str, Any]:
    lowered = text.casefold()
    hits = [needle for needle in expected if needle.casefold() in lowered]
    return {
        "passed": len(hits) == len(expected),
        "hits": hits,
        "expected": expected,
    }


def _create_route(config: Config) -> dict[str, Any]:
    return _request(
        "POST",
        f"{config.base_url}/internal/routes",
        config.internal_token,
        {
            "project": config.project,
            "task_kind": config.task_kind,
            "capabilities": config.capabilities,
        },
        timeout=config.timeout,
    )


def _operator_adapters(config: Config) -> dict[str, Any]:
    return _request(
        "GET",
        f"{config.base_url}/operator/adapters",
        config.operator_token,
        timeout=config.timeout,
    )


def run(config: Config) -> dict[str, Any]:
    inventory = _operator_adapters(config)
    route = _create_route(config)
    if not route.get("adapter_id"):
        raise RuntimeError(
            "route selected base_fallback; no approved/loaded adapter matched "
            f"project={config.project!r} capabilities={config.capabilities!r}"
        )
    results = []
    for item in PROMPTS:
        base = _chat(config, item["prompt"])
        routed = _chat(config, item["prompt"], route_id=route["route_id"])
        results.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "base": {
                "response": base,
                "score": _score(base, item["expected"]),
            },
            "lora": {
                "response": routed,
                "score": _score(routed, item["expected"]),
            },
        })
    base_passed = sum(1 for item in results if item["base"]["score"]["passed"])
    lora_passed = sum(1 for item in results if item["lora"]["score"]["passed"])
    return {
        "route": route,
        "aliases": inventory["aliases"],
        "base": inventory["base"],
        "summary": {
            "total": len(results),
            "base_passed": base_passed,
            "lora_passed": lora_passed,
            "delta": lora_passed - base_passed,
        },
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--project", default="ai-vllm")
    parser.add_argument("--task-kind", default="review")
    parser.add_argument("--capability", action="append", default=["code", "review"])
    parser.add_argument("--secrets-dir", default="secrets")
    parser.add_argument("--output", default="-")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    secrets_dir = pathlib.Path(args.secrets_dir)
    config = Config(
        base_url=args.base_url.rstrip("/"),
        proxy_token=_read_secret(secrets_dir / "clare2_proxy_token"),
        internal_token=_read_secret(secrets_dir / "clare2_callback_secret"),
        operator_token=_read_secret(secrets_dir / "clare2_operator_token"),
        project=args.project,
        task_kind=args.task_kind,
        capabilities=args.capability,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    result = run(config)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        pathlib.Path(args.output).write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
