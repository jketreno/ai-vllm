import os
from typing import Any

import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP

from .security import BearerASGIMiddleware, secret_value

mcp = FastMCP("clare-temper", host="0.0.0.0", port=8002)
POLICY_URL = os.environ.get("CLARE2_POLICY_URL", "http://clare2-policy:8000")


def _policy_request(method: str, path: str, **kwargs: Any) -> Any:
    token = secret_value("CLARE2_CALLBACK_SECRET")
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    response = httpx.request(
        method,
        f"{POLICY_URL}{path}",
        headers=headers,
        timeout=30,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def clare_temper_route(project: str, task_kind: str, capabilities: list[str]) -> dict:
    """Create a session-pinned route to the approved adapter selected by CLARE policy.

    Use the configured project key, describe the task kind, and list required
    capabilities. The returned opaque route ID remains pinned to the selected
    immutable adapter for the route lifetime; callers cannot choose adapters.
    """
    return _policy_request(
        "POST",
        "/internal/routes",
        json={
            "project": project,
            "task_kind": task_kind,
            "capabilities": capabilities,
        },
    )


@mcp.tool()
def clare_temper_status(route_id: str) -> dict:
    """Report the immutable adapter and availability for an existing route ID.

    Use this to confirm that a previously created route remains valid and
    available before sending inference requests with X-CLARE-Route-ID.
    """
    return _policy_request("GET", f"/internal/routes/{route_id}")


@mcp.tool()
def clare_temper_list(project: str) -> list[dict]:
    """List approved adapters visible to a configured project for diagnostics.

    This inventory is read-only. It does not select, load, unload, or route to
    an adapter; use clare_temper_route for policy-controlled routing.
    """
    return _policy_request("GET", "/internal/routes", params={"project": project})


if __name__ == "__main__":
    app = BearerASGIMiddleware(
        mcp.streamable_http_app(),
        secret_value("CLARE2_MCP_TOKEN"),
    )
    uvicorn.run(app, host="0.0.0.0", port=8002)
