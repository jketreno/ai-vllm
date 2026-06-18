"""Authenticated policy proxy for chat-completions inference endpoints."""

from __future__ import annotations

import json
import logging
import os
import time

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from . import metrics
from .routing import RouteError
from .runtime import BASE_MODEL_ID, VLLM_URL, controller, maintenance, router
from .security import require_bearer, secret_value

log = logging.getLogger(__name__)
router_api = APIRouter()

ALLOWED_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
    "/health",
}
BLOCKED_MANAGEMENT_PARTS = {"load_lora_adapter", "unload_lora_adapter"}
DEFAULT_THINKING_TOKEN_BUDGET = 1024
MAX_THINKING_TOKEN_BUDGET = 2048


def parse_endpoint_and_route(path: str, header_route_id: str | None) -> tuple[str, str | None]:
    endpoint = "/" + path
    route_id = header_route_id
    first, sep, rest = path.partition("/")
    # Optional route-in-path form: /<route-id>/v1/... for clients that cannot send custom headers.
    if sep and rest and ("/" + rest) in ALLOWED_ENDPOINTS and first not in {"v1", "health"}:
        endpoint = "/" + rest
        route_id = first
    return endpoint, route_id


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        log.warning("invalid integer environment value name=%s value=%r", name, value)
        return default


def apply_thinking_defaults(payload: dict) -> None:
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
        payload["chat_template_kwargs"] = chat_template_kwargs
    elif not isinstance(chat_template_kwargs, dict):
        return

    thinking_requested = chat_template_kwargs.get("enable_thinking")
    if thinking_requested is None:
        thinking_requested = env_bool("CLARE2_DEFAULT_ENABLE_THINKING", True)
        chat_template_kwargs["enable_thinking"] = thinking_requested

    if thinking_requested is False:
        return

    max_budget = env_int("CLARE2_MAX_THINKING_TOKEN_BUDGET", MAX_THINKING_TOKEN_BUDGET)
    if "thinking_token_budget" in payload:
        budget = payload["thinking_token_budget"]
        if isinstance(budget, int) and not isinstance(budget, bool) and budget > max_budget:
            payload["thinking_token_budget"] = max_budget
        return

    default_budget = env_int("CLARE2_DEFAULT_THINKING_TOKEN_BUDGET", DEFAULT_THINKING_TOKEN_BUDGET)
    if default_budget > 0:
        payload["thinking_token_budget"] = min(default_budget, max_budget)


@router_api.api_route(
    "/{path:path}",
    methods=["GET", "POST"],
)
async def forward(
    path: str,
    request: Request,
    x_clare_route_id: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Response:
    endpoint, resolved_route_id = parse_endpoint_and_route(path, x_clare_route_id)
    if endpoint not in ALLOWED_ENDPOINTS or any(part in endpoint for part in BLOCKED_MANAGEMENT_PARTS):
        raise HTTPException(status_code=404, detail="endpoint is not available")
    if endpoint == "/health":
        return Response(content='{"status":"ok"}', media_type="application/json")
    require_bearer(secret_value("CLARE2_PROXY_TOKEN"), authorization)
    if maintenance.enabled and endpoint not in {"/health"}:
        return Response(
            content='{"detail":"inference maintenance"}',
            status_code=503,
            media_type="application/json",
            headers={"Retry-After": os.environ.get("CLARE2_RETRY_AFTER", "300")},
        )

    adapter_id = None
    policy_rule = "base_without_route"
    project_id = None
    if resolved_route_id:
        try:
            route = router.get(resolved_route_id)
        except RouteError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        adapter_id = route.adapter_id
        policy_rule = route.policy_rule
        project_id = route.project_id

    request_guard = maintenance.request()
    try:
        request_guard.__enter__()
    except RuntimeError as exc:
        if str(exc) == "maintenance":
            raise HTTPException(status_code=503, detail="inference maintenance") from exc
        raise
    guard_owned_by_stream = False
    try:
        if adapter_id:
            controller.ensure_loaded(adapter_id)
        body = await request.body()
        stream_requested = False
        if body and endpoint.startswith("/v1/") and endpoint != "/v1/models":
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="invalid JSON request") from exc
            payload["model"] = adapter_id or BASE_MODEL_ID
            if endpoint == "/v1/chat/completions":
                apply_thinking_defaults(payload)
            stream_requested = payload.get("stream") is True
            body = json.dumps(payload).encode()
        started = time.monotonic()
        upstream_url = f"{VLLM_URL}{endpoint}"
        upstream_headers = {
            "content-type": request.headers.get("content-type", "application/json")
        }
        if stream_requested:
            client = httpx.AsyncClient(timeout=300)
            try:
                upstream_request = client.build_request(
                    request.method,
                    upstream_url,
                    content=body,
                    headers=upstream_headers,
                )
                upstream = await client.send(upstream_request, stream=True)
            except Exception:
                await client.aclose()
                raise
            guard_owned_by_stream = True
            return streaming_response(
                upstream,
                client,
                request_guard,
                started,
                resolved_route_id,
                project_id,
                policy_rule,
                adapter_id,
            )

        async with httpx.AsyncClient(timeout=300) as client:
            upstream = await client.request(
                request.method,
                upstream_url,
                content=body,
                headers=upstream_headers,
            )
        record_outcome(
            started,
            resolved_route_id,
            project_id,
            policy_rule,
            adapter_id,
            upstream.status_code,
        )
        excluded = {"content-encoding", "transfer-encoding", "connection", "content-length"}
        headers = {key: value for key, value in upstream.headers.items() if key.lower() not in excluded}
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=headers,
            media_type=upstream.headers.get("content-type"),
        )
    finally:
        if not guard_owned_by_stream:
            request_guard.__exit__(None, None, None)


def record_outcome(
    started: float,
    route_id: str | None,
    project_id: str | None,
    policy_rule: str,
    adapter_id: str | None,
    status_code: int,
) -> None:
    metrics.routing_decisions.labels(rule=policy_rule).inc()
    if not adapter_id:
        metrics.base_fallbacks.inc()
    metrics.proxy_latency.observe(time.monotonic() - started)
    log.info(
        "route_decision route_id=%s project_id=%s policy_rule=%s adapter_id=%s outcome=%s",
        route_id,
        project_id,
        policy_rule,
        adapter_id,
        status_code,
    )


def streaming_response(
    upstream: httpx.Response,
    client: httpx.AsyncClient,
    request_guard,
    started: float,
    route_id: str | None,
    project_id: str | None,
    policy_rule: str,
    adapter_id: str | None,
) -> StreamingResponse:
    excluded = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    headers = {key: value for key, value in upstream.headers.items() if key.lower() not in excluded}
    record_outcome(
        started,
        route_id,
        project_id,
        policy_rule,
        adapter_id,
        upstream.status_code,
    )

    async def chunks():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            request_guard.__exit__(None, None, None)

    return StreamingResponse(
        chunks(),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type"),
    )
