"""Process-wide policy runtime shared by the proxy, MCP, and operator API."""

from __future__ import annotations

import json
import os
import pathlib
import threading
from contextlib import contextmanager
from typing import Iterator

import httpx

from . import metrics
from .controller import AdapterController, HttpVllmClient
from .registry import AdapterRegistry
from .routing import Router


class MaintenanceState:
    def __init__(self) -> None:
        self._enabled = False
        self._active = 0
        self._condition = threading.Condition()

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled

    @property
    def active(self) -> int:
        with self._condition:
            return self._active

    def enter(self) -> None:
        with self._condition:
            self._enabled = True
            metrics.maintenance_mode.set(1)

    def exit(self) -> None:
        with self._condition:
            self._enabled = False
            metrics.maintenance_mode.set(0)
            self._condition.notify_all()

    def wait_for_drain(self, timeout: float) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: self._active == 0, timeout=timeout)

    @contextmanager
    def request(self) -> Iterator[None]:
        with self._condition:
            if self._enabled:
                raise RuntimeError("maintenance")
            self._active += 1
            metrics.active_requests.inc()
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                metrics.active_requests.dec()
                self._condition.notify_all()


MODELS_ROOT = pathlib.Path(os.environ.get("MODELS_ROOT", "/models"))
VLLM_URL = os.environ.get("CLARE2_VLLM_URL", "http://vllm-engine:8001")
BASE_MODEL_ID = os.environ.get("CLARE2_INFERENCE_MODEL", "Qwen/Qwen3.6-27B-FP8")
PROJECTS = json.loads(
    os.environ.get("CLARE2_PROJECT_MAP", '{"clare":"github:jketreno/clare"}')
)

registry = AdapterRegistry(MODELS_ROOT)
router = Router(registry, PROJECTS)
maintenance = MaintenanceState()
http = httpx.Client(timeout=120)
controller = AdapterController(
    registry,
    HttpVllmClient(VLLM_URL, http),
    router.active_adapter_ids,
    max_cpu_loras=int(os.environ.get("CLARE2_MAX_CPU_LORAS", "8")),
)


def initialize_registry() -> None:
    registry.initialize(
        {
            "model_id": BASE_MODEL_ID,
            "inference_model_id": BASE_MODEL_ID,
            "revision": os.environ.get(
                "CLARE2_INFERENCE_REVISION",
                "REPLACE_WITH_INFERENCE_REVISION",
            ),
            "inference_revision": os.environ.get(
                "CLARE2_INFERENCE_REVISION",
                "REPLACE_WITH_INFERENCE_REVISION",
            ),
            "architecture": os.environ.get(
                "CLARE2_BASE_ARCHITECTURE",
                "Qwen3_5ForConditionalGeneration",
            ),
            "config_hash": os.environ.get(
                "CLARE2_BASE_CONFIG_HASH", "REPLACE_WITH_SHA256"
            ),
            "tokenizer_hash": os.environ.get(
                "CLARE2_TOKENIZER_HASH", "REPLACE_WITH_SHA256"
            ),
            "inference_quantization": "fp8",
        }
    )
