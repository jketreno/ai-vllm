"""Private vLLM adapter controller with serialized loads and LRU tracking."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Protocol

from . import metrics
from .registry import AdapterRegistry, RegistryError


class VllmClient(Protocol):
    def models(self) -> set[str]: ...
    def load(self, adapter_id: str, path: str) -> None: ...
    def unload(self, adapter_id: str) -> None: ...


class AdapterController:
    def __init__(
        self,
        registry: AdapterRegistry,
        client: VllmClient,
        pinned: Callable[[], set[str]],
        *,
        max_cpu_loras: int = 8,
    ) -> None:
        self.registry = registry
        self.client = client
        self.pinned = pinned
        self.max_cpu_loras = max_cpu_loras
        self._loaded: OrderedDict[str, float] = OrderedDict()
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.RLock()

    def reconcile(self) -> set[str]:
        upstream = self.client.models()
        registry = self.registry.read()
        known = set(registry["adapters"])
        with self._guard:
            self._loaded = OrderedDict(
                (adapter_id, time.monotonic())
                for adapter_id in sorted(upstream & known)
            )
        base_ids = {
            registry["base"]["model_id"],
            registry["base"].get("inference_model_id"),
        }
        unknown = upstream - known - base_ids
        if unknown:
            metrics.registry_reconciliation_errors.inc(len(unknown))
        return unknown

    def ensure_loaded(self, adapter_id: str | None) -> None:
        if adapter_id is None:
            return
        lock = self._adapter_lock(adapter_id)
        with lock:
            with self._guard:
                if adapter_id in self._loaded:
                    self._loaded.move_to_end(adapter_id)
                    metrics.adapter_operations.labels(operation="cache_hit", outcome="success").inc()
                    return
            registry = self.registry.read()
            adapter = registry["adapters"].get(adapter_id)
            if adapter is None or adapter["status"] not in {"approved", "loaded", "candidate"}:
                metrics.adapter_compatibility_failures.inc()
                raise RegistryError("adapter is not approved for loading")
            self.registry.validate_adapter(adapter)
            self._evict_if_needed(adapter_id)
            started = time.monotonic()
            try:
                self.client.load(adapter_id, str(self.registry.adapter_path(adapter)))
            except Exception:
                metrics.adapter_operations.labels(operation="load", outcome="failure").inc()
                raise
            metrics.adapter_operation_latency.labels(operation="load").observe(time.monotonic() - started)
            metrics.adapter_operations.labels(operation="load", outcome="success").inc()
            with self._guard:
                self._loaded[adapter_id] = time.monotonic()

    def unload(self, adapter_id: str) -> None:
        if adapter_id in self.pinned():
            raise RegistryError("cannot unload an adapter pinned by an active route")
        self.client.unload(adapter_id)
        with self._guard:
            self._loaded.pop(adapter_id, None)
        metrics.adapter_operations.labels(operation="unload", outcome="success").inc()

    def _evict_if_needed(self, incoming: str) -> None:
        with self._guard:
            if incoming in self._loaded or len(self._loaded) < self.max_cpu_loras:
                return
            pinned = self.pinned()
            victim = next((adapter_id for adapter_id in self._loaded if adapter_id not in pinned), None)
        if victim is None:
            raise RegistryError("adapter cache is full and every adapter is pinned")
        self.unload(victim)

    def _adapter_lock(self, adapter_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(adapter_id, threading.Lock())


class HttpVllmClient:
    def __init__(self, base_url: str, http_client: Any) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = http_client

    def models(self) -> set[str]:
        response = self.http.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return {item["id"] for item in response.json().get("data", [])}

    def load(self, adapter_id: str, path: str) -> None:
        response = self.http.post(
            f"{self.base_url}/v1/load_lora_adapter",
            json={"lora_name": adapter_id, "lora_path": path, "load_inplace": False},
        )
        response.raise_for_status()

    def unload(self, adapter_id: str) -> None:
        response = self.http.post(
            f"{self.base_url}/v1/unload_lora_adapter",
            json={"lora_name": adapter_id},
        )
        response.raise_for_status()
