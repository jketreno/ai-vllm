"""Authenticated policy proxy for chat-completions inference endpoints."""

from __future__ import annotations

import json
import logging
import os
import time

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Response

from . import metrics
from .admission import (
    AdmissionRejected,
    ClientDisconnected,
    from_environment,
)
from .proxy_transport import dispatch as _dispatch
from .routing import RouteError
from .runtime import BASE_MODEL_ID, VLLM_URL, controller, maintenance, router
from .security import require_bearer, secret_value

log = logging.getLogger(__name__)
router_api = APIRouter()
admission = from_environment()

ALLOWED_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
    "/capacity",
    "/health",
}
BLOCKED_MANAGEMENT_PARTS = {"load_lora_adapter", "unload_lora_adapter"}


def _deep_merge(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_endpoint_and_route(
    path: str, header_route_id: str | None
) -> tuple[str, str | None]:
    endpoint = "/" + path
    route_id = header_route_id
    first, sep, rest = path.partition("/")
    # Optional route-in-path form: /<route-id>/v1/... for clients that cannot
    # send custom headers.
    if (
        sep
        and rest
        and ("/" + rest) in ALLOWED_ENDPOINTS
        and first not in {"v1", "health"}
    ):
        endpoint = "/" + rest
        route_id = first
    return endpoint, route_id


def _resolve_route(resolved_route_id: str | None) -> tuple[str | None, str, str | None]:
    if not resolved_route_id:
        return None, "base_without_route", None
    try:
        route = router.get(resolved_route_id)
    except RouteError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return route.adapter_id, route.policy_rule, route.project_id


def _maintenance_response() -> Response:
    return Response(
        content='{"detail":"inference maintenance"}',
        status_code=503,
        media_type="application/json",
        headers={"Retry-After": os.environ.get("CLARE2_RETRY_AFTER", "300")},
    )


async def _capacity_response(workload: str) -> Response:
    if maintenance.enabled:
        return _maintenance_response()
    snapshot = await admission.capacity(workload)
    status_code = 200 if snapshot["available"] else 429
    headers = (
        {}
        if snapshot["available"]
        else {"Retry-After": str(snapshot["retry_after"])}
    )
    return Response(
        content=json.dumps(snapshot),
        status_code=status_code,
        media_type="application/json",
        headers=headers,
    )


async def _acquire_admission(request: Request, workload: str):
    try:
        lease = await admission.acquire(workload, request.is_disconnected)
    except AdmissionRejected as error:
        metrics.inference_admission_outcomes.labels(
            workload, "rejected"
        ).inc()
        return Response(
            content='{"detail":"inference capacity saturated"}',
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": str(error.retry_after)},
        )
    except ClientDisconnected:
        metrics.inference_admission_outcomes.labels(
            workload, "disconnected_before_admission"
        ).inc()
        return Response(status_code=499)
    metrics.inference_admission_active.labels(workload).inc()
    metrics.inference_admission_outcomes.labels(workload, "admitted").inc()
    return lease


async def _dispatch_inference(
    request: Request,
    endpoint: str,
    resolved_route_id: str | None,
    adapter_id: str | None,
    project_id: str | None,
    policy_rule: str,
    x_clare2_params: str | None,
    request_guard,
    admission_lease,
) -> tuple[Response, bool]:
    try:
        if adapter_id:
            controller.ensure_loaded(adapter_id)
        body, stream_requested = await _prepare_body(
            request, endpoint, adapter_id, x_clare2_params
        )
        return await _dispatch(
            request,
            body,
            stream_requested,
            f"{VLLM_URL}{endpoint}",
            {
                "content-type": request.headers.get(
                    "content-type", "application/json"
                )
            },
            time.monotonic(),
            resolved_route_id,
            project_id,
            policy_rule,
            adapter_id,
            request_guard,
            admission_lease,
        )
    except ClientDisconnected:
        metrics.inference_admission_outcomes.labels(
            admission_lease.workload, "cancelled"
        ).inc()
        return Response(status_code=499), False
    except httpx.ConnectError:
        log.error("Unable to connect to vLLM engine at %s", VLLM_URL)
        return (
            Response(
                content='{"detail":"unable to connect to vLLM engine"}',
                status_code=503,
                media_type="application/json",
            ),
            False,
        )


async def _forward_inference(
    request: Request,
    endpoint: str,
    resolved_route_id: str | None,
    x_clare2_params: str | None,
    workload: str,
) -> Response:
    adapter_id, policy_rule, project_id = _resolve_route(resolved_route_id)
    admission_lease = await _acquire_admission(request, workload)
    if isinstance(admission_lease, Response):
        return admission_lease

    request_guard = maintenance.request()
    try:
        request_guard.__enter__()
    except RuntimeError as exc:
        await admission_lease.release()
        metrics.inference_admission_active.labels(workload).dec()
        if str(exc) == "maintenance":
            raise HTTPException(
                status_code=503, detail="inference maintenance"
            ) from exc
        raise

    owned_by_stream = False
    try:
        response, owned_by_stream = await _dispatch_inference(
            request,
            endpoint,
            resolved_route_id,
            adapter_id,
            project_id,
            policy_rule,
            x_clare2_params,
            request_guard,
            admission_lease,
        )
        return response
    finally:
        if not owned_by_stream:
            request_guard.__exit__(None, None, None)
            await admission_lease.release()
            metrics.inference_admission_active.labels(workload).dec()


async def _prepare_body(
    request: Request,
    endpoint: str,
    adapter_id: str | None,
    extra_params_header: str | None,
) -> tuple[bytes, bool]:
    body = await request.body()
    stream_requested = False
    if not (body and endpoint.startswith("/v1/") and endpoint != "/v1/models"):
        return body, stream_requested
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON request") from exc
    if extra_params_header:
        try:
            extra_params = json.loads(extra_params_header)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="invalid JSON in X-CLARE2-Params header"
            ) from exc
        if not isinstance(extra_params, dict):
            raise HTTPException(
                status_code=400, detail="X-CLARE2-Params header must be a JSON object"
            )
        _deep_merge(payload, extra_params)
    payload["model"] = adapter_id or BASE_MODEL_ID
    stream_requested = payload.get("stream") is True
    return json.dumps(payload).encode(), stream_requested


@router_api.api_route(
    "/{path:path}",
    methods=["GET", "POST"],
)
async def forward(
    path: str,
    request: Request,
    x_clare_route_id: str | None = Header(default=None),
    x_clare2_params: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Response:
    endpoint, resolved_route_id = parse_endpoint_and_route(path, x_clare_route_id)
    if endpoint not in ALLOWED_ENDPOINTS or any(
        part in endpoint for part in BLOCKED_MANAGEMENT_PARTS
    ):
        raise HTTPException(status_code=404, detail="endpoint is not available")
    if endpoint == "/health":
        return Response(content='{"status":"ok"}', media_type="application/json")
    require_bearer(secret_value("CLARE2_PROXY_TOKEN"), authorization)
    workload = request.headers.get("X-Inference-Workload", "default")[:64]
    if endpoint == "/capacity":
        return await _capacity_response(workload)
    if maintenance.enabled:
        return _maintenance_response()
    return await _forward_inference(
        request,
        endpoint,
        resolved_route_id,
        x_clare2_params,
        workload,
    )
