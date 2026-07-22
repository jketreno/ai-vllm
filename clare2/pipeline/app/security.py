"""Bearer and HMAC authentication helpers."""

from __future__ import annotations

import hashlib
import hmac
import os
import pathlib
import time

from fastapi import Header, HTTPException
from starlette.responses import JSONResponse


def secret_value(name: str) -> str:
    direct = os.environ.get(name)
    if direct:
        return direct
    path = pathlib.Path(os.environ.get(f"{name}_FILE", f"/run/secrets/{name.lower()}"))
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def require_bearer(expected: str, authorization: str | None) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    if (
        scheme.lower() != "bearer"
        or not expected
        or not hmac.compare_digest(token, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid bearer token")


def verify_callback(
    secret: str,
    body: bytes,
    timestamp: str | None,
    signature: str | None,
    *,
    max_age_seconds: int = 300,
) -> None:
    if not secret or not timestamp or not signature:
        raise HTTPException(status_code=401, detail="missing callback authentication")
    try:
        sent_at = int(timestamp)
    except ValueError as exc:
        raise HTTPException(
            status_code=401, detail="invalid callback timestamp"
        ) from exc
    if abs(int(time.time()) - sent_at) > max_age_seconds:
        raise HTTPException(status_code=401, detail="expired callback")
    expected = hmac.new(
        secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="invalid callback signature")


def bearer_dependency(expected: str):
    def dependency(authorization: str | None = Header(default=None)) -> None:
        require_bearer(expected, authorization)

    return dependency


class BearerASGIMiddleware:
    """Require a fixed bearer token for HTTP requests to an ASGI app."""

    def __init__(self, app, expected: str) -> None:
        self.app = app
        self.expected = expected

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        try:
            require_bearer(self.expected, authorization)
        except HTTPException:
            response = JSONResponse({"detail": "invalid bearer token"}, status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
