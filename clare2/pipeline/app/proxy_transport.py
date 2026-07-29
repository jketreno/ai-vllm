"""Cancellation-aware HTTP transport for the policy proxy."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from contextlib import suppress

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from . import metrics
from .admission import AdmissionLease, ClientDisconnected

log = logging.getLogger(__name__)
VLLM_TIMEOUT = httpx.Timeout(connect=10, read=None, write=30, pool=10)


async def dispatch(
    request: Request,
    body: bytes,
    stream_requested: bool,
    upstream_url: str,
    upstream_headers: dict,
    started: float,
    resolved_route_id: str | None,
    project_id: str | None,
    policy_rule: str,
    adapter_id: str | None,
    request_guard,
    admission_lease: AdmissionLease,
) -> tuple[Response, bool]:
    if stream_requested:
        client = httpx.AsyncClient(timeout=VLLM_TIMEOUT)
        try:
            upstream_request = client.build_request(
                request.method,
                upstream_url,
                content=body,
                headers=upstream_headers,
            )
            upstream = await cancel_on_disconnect(
                client.send(upstream_request, stream=True),
                request,
            )
        except Exception:
            await client.aclose()
            raise
        return (
            streaming_response(
                upstream,
                client,
                request_guard,
                started,
                resolved_route_id,
                project_id,
                policy_rule,
                adapter_id,
                admission_lease,
            ),
            True,
        )

    async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
        upstream = await cancel_on_disconnect(
            client.request(
                request.method,
                upstream_url,
                content=body,
                headers=upstream_headers,
            ),
            request,
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
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}
    return (
        Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=headers,
            media_type=upstream.headers.get("content-type"),
        ),
        False,
    )


async def request_until_disconnected(
    client: httpx.AsyncClient,
    downstream: Request,
    method: str,
    url: str,
    body: bytes,
    headers: dict,
) -> httpx.Response:
    return await cancel_on_disconnect(
        client.request(method, url, content=body, headers=headers),
        downstream,
    )


async def cancel_on_disconnect(
    upstream_request: Awaitable[httpx.Response],
    downstream: Request,
) -> httpx.Response:
    upstream_task = asyncio.ensure_future(upstream_request)

    async def wait_for_disconnect() -> None:
        while not await downstream.is_disconnected():
            await asyncio.sleep(0.1)

    disconnect_task = asyncio.create_task(wait_for_disconnect())
    done, _pending = await asyncio.wait(
        {upstream_task, disconnect_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if upstream_task in done:
        disconnect_task.cancel()
        return await upstream_task
    upstream_task.cancel()
    with suppress(asyncio.CancelledError):
        await upstream_task
    raise ClientDisconnected()


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
        "route_decision route_id=%s project_id=%s policy_rule=%s adapter_id=%s "
        "outcome=%s",
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
    admission_lease: AdmissionLease,
) -> StreamingResponse:
    excluded = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in excluded
    }
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
            await admission_lease.release()
            metrics.inference_admission_active.labels(
                admission_lease.workload
            ).dec()

    return StreamingResponse(
        chunks(),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type"),
    )
