"""Validation for Auto SAM user bearer tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from pathlib import Path


class AuthenticationError(ValueError):
    """Raised when an access token cannot be trusted."""


_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ISSUER = os.environ.get("IMAGE_API_AUTH_ISSUER", "auto-sam")
_AUDIENCE = os.environ.get("IMAGE_API_AUTH_AUDIENCE", "image-api")
_SECRET_FILE = Path(
    os.environ.get(
        "IMAGE_API_AUTH_TOKEN_SECRET_FILE",
        "/run/secrets/auto_sam_auth_token",
    )
)
_OPENWEBUI_SECRET_FILE = Path(
    os.environ.get(
        "IMAGE_API_OPENWEBUI_TOKEN_FILE",
        "/run/secrets/image_api_openwebui_token",
    )
)


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except (ValueError, TypeError) as error:
        raise AuthenticationError("invalid token encoding") from error


def _secret() -> bytes:
    try:
        value = _SECRET_FILE.read_bytes().strip()
    except OSError:
        value = os.environ.get("IMAGE_API_AUTH_TOKEN_SECRET", "").encode()
    if len(value) < 32:
        raise AuthenticationError("authentication token secret is not configured")
    return value


def _openwebui_secret() -> bytes:
    try:
        value = _OPENWEBUI_SECRET_FILE.read_bytes().strip()
    except OSError:
        value = os.environ.get("IMAGE_API_OPENWEBUI_TOKEN", "").encode()
    if len(value) < 32:
        raise AuthenticationError("Open WebUI service token is not configured")
    return value


def validate_openwebui_token(token: str) -> str:
    """Validate the dedicated service credential for OpenAI-compatible routes."""
    if not hmac.compare_digest(token.encode(), _openwebui_secret()):
        raise AuthenticationError("invalid Open WebUI service token")
    return "open-webui"


def validate_access_token(token: str) -> str:
    """Validate an HS256 token and return its safe, canonical username."""
    try:
        encoded_header, encoded_claims, encoded_signature = token.split(".")
        header = json.loads(_b64url_decode(encoded_header))
        claims = json.loads(_b64url_decode(encoded_claims))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise AuthenticationError("invalid bearer token") from error

    if header.get("alg") != "HS256" or header.get("typ") != "JWT":
        raise AuthenticationError("unsupported bearer token")
    signed = f"{encoded_header}.{encoded_claims}".encode()
    expected = hmac.new(_secret(), signed, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64url_decode(encoded_signature)):
        raise AuthenticationError("invalid bearer token signature")

    now = int(time.time())
    if claims.get("iss") != _ISSUER or claims.get("aud") != _AUDIENCE:
        raise AuthenticationError("invalid bearer token claims")
    if not isinstance(claims.get("iat"), int) or claims["iat"] > now + 60:
        raise AuthenticationError("invalid bearer token issue time")
    if not isinstance(claims.get("exp"), int) or claims["exp"] <= now:
        raise AuthenticationError("bearer token expired")
    username = claims.get("sub")
    if not isinstance(username, str) or not _USERNAME_RE.fullmatch(username):
        raise AuthenticationError("invalid bearer token subject")
    return username
