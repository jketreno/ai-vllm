"""Client for the private model capability RPC."""

import json
import uuid

import httpx
from fastapi import HTTPException


PROTOCOL_VERSION = "1"


class WorkerClient:
    def __init__(self, base_url: str, timeout: float = 1800):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def capabilities(self) -> dict:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/v1/capabilities")
        response.raise_for_status()
        return response.json()

    async def ready(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/health/ready")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def invoke(
        self,
        operation: str,
        parameters: dict,
        attachments: list[tuple[str, bytes, str]],
        request_id: str | None = None,
    ) -> dict:
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id or str(uuid.uuid4()),
            "operation": operation,
            "parameters": parameters,
            "attachments": [
                {"name": name, "media_type": media_type}
                for name, _, media_type in attachments
            ],
        }
        files = [
            ("attachments", (name, payload, media_type))
            for name, payload, media_type in attachments
        ]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/invoke",
                    data={"manifest": json.dumps(manifest)},
                    files=files,
                )
        except httpx.HTTPError as error:
            raise HTTPException(503, f"model worker unavailable: {error}") from error
        if response.status_code == 503:
            detail = "model capability is not ready"
            try:
                worker_detail = response.json().get("detail")
            except (ValueError, AttributeError):
                worker_detail = None
            if isinstance(worker_detail, str) and worker_detail:
                detail = worker_detail
            raise HTTPException(503, detail)
        if response.status_code >= 400:
            raise HTTPException(
                502, f"model worker rejected request: {response.text[:500]}"
            )
        result = response.json()
        if (
            result.get("protocol_version") != PROTOCOL_VERSION
            or result.get("status") != "ok"
        ):
            raise HTTPException(502, "invalid model worker response")
        return result

    async def invoke_progress(self, request_id: str) -> dict:
        """Poll step progress for an in-flight /v1/invoke call. Raises
        HTTPException(404) if the worker has no progress recorded (unknown, not
        yet started, or finished)."""
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{self.base_url}/v1/invoke/{request_id}/progress"
            )
        if response.status_code == 404:
            raise HTTPException(404, "no progress recorded for this request_id")
        response.raise_for_status()
        return response.json()

    async def invoke_preview(self, request_id: str) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{self.base_url}/v1/invoke/{request_id}/preview"
            )
        if response.status_code == 404:
            raise HTTPException(404, "no preview is available for this request_id")
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "image/jpeg")

    async def cancel_invoke(self, request_id: str) -> dict:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{self.base_url}/v1/invoke/{request_id}/cancel"
            )
        response.raise_for_status()
        return response.json()
