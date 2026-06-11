"""CLARE Temper MCP tools. Agents receive routes, never adapter controls."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .runtime import router

mcp = FastMCP("clare-temper", host="0.0.0.0", port=8002)


@mcp.tool()
def clare_temper_route(project: str, task_kind: str, capabilities: list[str]) -> dict:
    route = router.create_route(project, task_kind, capabilities)
    return {
        "route_id": route.route_id,
        "project_id": route.project_id,
        "adapter_id": route.adapter_id,
        "policy_rule": route.policy_rule,
        "expires_at": route.expires_at.isoformat(),
    }


@mcp.tool()
def clare_temper_status(route_id: str) -> dict:
    route = router.get(route_id)
    return {
        "route_id": route.route_id,
        "adapter_id": route.adapter_id,
        "policy_rule": route.policy_rule,
        "available": True,
        "expires_at": route.expires_at.isoformat(),
    }


@mcp.tool()
def clare_temper_list(project: str) -> list[dict]:
    return router.list_approved(project)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
