"""Authenticated client for CLARE2's GB10 image-edit resource lease."""

import os
from contextlib import asynccontextmanager

import httpx
from fastapi import HTTPException


LEASE_URL = os.environ.get(
    "IMAGE_API_RESOURCE_LEASE_URL",
    "http://clare2-policy:8000/operator/resource-leases/image-edit",
)
LEASE_TOKEN_FILE = os.environ.get(
    "IMAGE_API_RESOURCE_LEASE_TOKEN_FILE", "/run/secrets/clare2_operator_token"
)
EXCLUSIVE_VLLM = os.environ.get("IMAGE_API_EXCLUSIVE_VLLM", "false").lower() in (
    "1", "true", "yes",
)


def _token() -> str:
    try:
        with open(LEASE_TOKEN_FILE, encoding="utf-8") as token_file:
            return token_file.read().strip()
    except OSError as error:
        raise HTTPException(
            503, "image resource coordinator credentials unavailable"
        ) from error


@asynccontextmanager
async def image_edit_lease(request_id: str):
    if not EXCLUSIVE_VLLM:
        yield
        return
    headers = {"Authorization": f"Bearer {_token()}"}
    timeout = httpx.Timeout(connect=10, read=900, write=30, pool=10)
    lease_id = None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                LEASE_URL, headers=headers, json={"request_id": request_id}
            )
        if response.status_code == 409:
            raise HTTPException(
                503, "image resources are busy", headers={"Retry-After": "30"}
            )
        response.raise_for_status()
        lease_id = response.json()["lease_id"]
        yield
    except httpx.HTTPError as error:
        raise HTTPException(
            503, f"image resource coordinator unavailable: {error}"
        ) from error
    finally:
        if lease_id:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    await client.delete(f"{LEASE_URL}/{lease_id}", headers=headers)
            except httpx.HTTPError:
                # Coordinator TTL reconciliation is the failure-safe fallback.
                pass
