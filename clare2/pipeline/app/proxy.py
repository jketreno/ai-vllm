"""Authenticated policy proxy for OpenAI-compatible inference endpoints."""

from __future__ import annotations

import json
import logging
import os
import time

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Response

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
    endpoint = "/" + path
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
    if x_clare_route_id:
        try:
            route = router.get(x_clare_route_id)
        except RouteError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        adapter_id = route.adapter_id
        policy_rule = route.policy_rule
        project_id = route.project_id

    try:
        with maintenance.request():
            if adapter_id:
                controller.ensure_loaded(adapter_id)
            body = await request.body()
            if body and endpoint.startswith("/v1/") and endpoint != "/v1/models":
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise HTTPException(status_code=400, detail="invalid JSON request") from exc
                payload["model"] = adapter_id or BASE_MODEL_ID
                body = json.dumps(payload).encode()
            started = time.monotonic()
            async with httpx.AsyncClient(timeout=300) as client:
                upstream = await client.request(
                    request.method,
                    f"{VLLM_URL}{endpoint}",
                    content=body,
                    headers={"content-type": request.headers.get("content-type", "application/json")},
                )
            metrics.routing_decisions.labels(rule=policy_rule).inc()
            if not adapter_id:
                metrics.base_fallbacks.inc()
            metrics.proxy_latency.observe(time.monotonic() - started)
            log.info(
                "route_decision route_id=%s project_id=%s policy_rule=%s adapter_id=%s outcome=%s",
                x_clare_route_id,
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
    except RuntimeError as exc:
        if str(exc) == "maintenance":
            raise HTTPException(status_code=503, detail="inference maintenance") from exc
        raise
