"""Security invariants for image-api bearer authentication."""

import base64
import hashlib
import hmac
import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request
from starlette.responses import Response

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
if "image_api_app" in sys.modules:
    app = sys.modules["image_api_app"]
else:
    spec = importlib.util.spec_from_file_location(
        "image_api_app", MODULE_DIR / "app.py"
    )
    app = importlib.util.module_from_spec(spec)
    sys.modules["image_api_app"] = app
    spec.loader.exec_module(app)


SECRET = b"a-test-secret-that-is-at-least-thirty-two-bytes"
SERVICE_SECRET = "open-webui-test-token-that-is-long-enough"


def _encode(value: dict) -> str:
    payload = json.dumps(value, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode()


def _token(**claim_overrides) -> str:
    now = int(time.time())
    header = _encode({"alg": "HS256", "typ": "JWT"})
    claims = _encode(
        {
            "iss": "auto-sam",
            "aud": "image-api",
            "sub": "alice",
            "iat": now,
            "exp": now + 600,
            **claim_overrides,
        }
    )
    signed = f"{header}.{claims}".encode()
    signature = base64.urlsafe_b64encode(
        hmac.new(SECRET, signed, hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    return f"{header}.{claims}.{signature}"


def _request(path: str, token: str | None = None) -> Request:
    headers = (
        [(b"authorization", f"Bearer {token}".encode())]
        if token
        else []
    )
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": headers,
            "query_string": b"",
            "server": ("test", 80),
            "client": ("test", 123),
            "scheme": "http",
        }
    )


class AuthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.secret = patch("auth._secret", return_value=SECRET)
        self.service_secret = patch(
            "auth._openwebui_secret",
            return_value=SERVICE_SECRET.encode(),
        )
        self.secret.start()
        self.service_secret.start()

    def tearDown(self):
        self.service_secret.stop()
        self.secret.stop()

    @staticmethod
    async def _next(_request):
        return Response(status_code=204)

    async def test_health_and_capabilities_are_public(self):
        response = await app.require_image_bearer(
            _request("/health/live"), self._next
        )
        self.assertEqual(response.status_code, 204)

    async def test_image_routes_require_a_bearer_token(self):
        response = await app.require_image_bearer(
            _request("/v1/images/edit"), self._next
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    async def test_valid_token_reaches_the_image_route(self):
        request = _request("/v1/images/edit", _token())
        response = await app.require_image_bearer(
            request,
            self._next,
        )
        self.assertEqual(response.status_code, 204)
        self.assertEqual(request.state.username, "alice")

    async def test_openai_routes_accept_only_the_openwebui_service_token(self):
        request = _request("/openai/v1/images/edits", SERVICE_SECRET)
        response = await app.require_image_bearer(request, self._next)
        self.assertEqual(response.status_code, 204)
        self.assertEqual(request.state.service, "open-webui")

        user_token_response = await app.require_image_bearer(
            _request("/openai/v1/images/edits", _token()),
            self._next,
        )
        self.assertEqual(user_token_response.status_code, 401)

    async def test_native_routes_reject_the_openwebui_service_token(self):
        response = await app.require_image_bearer(
            _request("/v1/images/edit", SERVICE_SECRET),
            self._next,
        )
        self.assertEqual(response.status_code, 401)

    async def test_expired_token_is_rejected(self):
        response = await app.require_image_bearer(
            _request("/v1/images/edit", _token(exp=int(time.time()) - 1)),
            self._next,
        )
        self.assertEqual(response.status_code, 401)

    async def test_path_like_subject_is_rejected(self):
        response = await app.require_image_bearer(
            _request("/v1/images/edit", _token(sub="../alice")),
            self._next,
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
