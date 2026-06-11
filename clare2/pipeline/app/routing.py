"""Deterministic adapter policy and route lifetime pinning."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .registry import AdapterRegistry


class RouteError(ValueError):
    """Route context is missing, expired, or unauthorized."""


@dataclass(frozen=True)
class Route:
    route_id: str
    project_id: str
    adapter_id: str | None
    policy_rule: str
    capabilities: tuple[str, ...]
    created_at: datetime
    expires_at: datetime


class Router:
    def __init__(
        self,
        registry: AdapterRegistry,
        projects: dict[str, str],
        *,
        route_ttl_seconds: int = 8 * 60 * 60,
    ) -> None:
        self.registry = registry
        self.projects = projects
        self.route_ttl = timedelta(seconds=route_ttl_seconds)
        self._routes: dict[str, Route] = {}
        self._lock = threading.RLock()

    def create_route(
        self,
        project: str,
        task_kind: str,
        capabilities: list[str] | tuple[str, ...],
    ) -> Route:
        del task_kind
        project_id = self._canonical_project(project)
        document = self.registry.read()
        adapter_id, rule = self._select(document, project_id, set(capabilities))
        now = datetime.now(tz=timezone.utc)
        route = Route(
            route_id=secrets.token_urlsafe(32),
            project_id=project_id,
            adapter_id=adapter_id,
            policy_rule=rule,
            capabilities=tuple(sorted(set(capabilities))),
            created_at=now,
            expires_at=now + self.route_ttl,
        )
        with self._lock:
            self._routes[route.route_id] = route
        return route

    def get(self, route_id: str) -> Route:
        with self._lock:
            route = self._routes.get(route_id)
            if route is None:
                raise RouteError("unknown route")
            if route.expires_at <= datetime.now(tz=timezone.utc):
                del self._routes[route_id]
                raise RouteError("expired route")
            return route

    def active_adapter_ids(self) -> set[str]:
        with self._lock:
            now = datetime.now(tz=timezone.utc)
            self._routes = {key: value for key, value in self._routes.items() if value.expires_at > now}
            return {route.adapter_id for route in self._routes.values() if route.adapter_id}

    def list_approved(self, project: str) -> list[dict[str, Any]]:
        project_id = self._canonical_project(project)
        adapters = self.registry.read()["adapters"].values()
        return [
            self._public_adapter(adapter)
            for adapter in adapters
            if adapter["status"] in {"approved", "loaded"}
            and adapter.get("project_scope") in {project_id, "global"}
        ]

    def _canonical_project(self, project: str) -> str:
        try:
            return self.projects[project]
        except KeyError as exc:
            raise RouteError("project is not in the configured repository map") from exc

    @staticmethod
    def _select(
        document: dict[str, Any],
        project_id: str,
        requested: set[str],
    ) -> tuple[str | None, str]:
        approved = [
            adapter
            for adapter in document["adapters"].values()
            if adapter["status"] in {"approved", "loaded"}
        ]
        project_matches = [
            adapter
            for adapter in approved
            if adapter.get("project_scope") == project_id
            and requested <= set(adapter.get("capabilities", []))
        ]
        if project_matches:
            selected = max(project_matches, key=lambda item: item["created_at"])
            return selected["id"], "project_capability_match"
        global_matches = [
            adapter
            for adapter in approved
            if adapter.get("project_scope") == "global"
            and requested <= set(adapter.get("capabilities", []))
        ]
        if global_matches:
            selected = max(global_matches, key=lambda item: item["created_at"])
            return selected["id"], "global_temper"
        return None, "base_fallback"

    @staticmethod
    def _public_adapter(adapter: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": adapter["id"],
            "created_at": adapter["created_at"],
            "project_scope": adapter.get("project_scope"),
            "capabilities": adapter.get("capabilities", []),
            "status": adapter["status"],
            "evaluation": adapter.get("evaluation"),
        }
