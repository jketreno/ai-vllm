"""Bounded admission control for shared vLLM inference."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


class AdmissionRejected(RuntimeError):
    def __init__(self, retry_after: int, reason: str = "inference saturated"):
        super().__init__(reason)
        self.retry_after = retry_after


class ClientDisconnected(RuntimeError):
    pass


@dataclass
class AdmissionLease:
    controller: AdmissionController
    workload: str
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        await self.controller.release(self.workload)


class AdmissionController:
    def __init__(
        self,
        max_active: int,
        max_waiting: int,
        wait_seconds: float,
        retry_after: int,
        workload_limits: dict[str, int] | None = None,
    ):
        self.max_active = max(1, max_active)
        self.max_waiting = max(0, max_waiting)
        self.wait_seconds = max(0.0, wait_seconds)
        self.retry_after = max(1, retry_after)
        self.workload_limits = workload_limits or {}
        self.active = 0
        self.waiting = 0
        self.active_by_workload: dict[str, int] = {}
        self._lock = threading.Lock()

    def _has_capacity(self, workload: str) -> bool:
        workload_limit = self.workload_limits.get(workload, self.max_active)
        return (
            self.active < self.max_active
            and self.active_by_workload.get(workload, 0) < workload_limit
        )

    async def capacity(self, workload: str) -> dict:
        with self._lock:
            return self._snapshot(workload)

    def _snapshot(self, workload: str) -> dict:
        return {
            "available": self._has_capacity(workload),
            "active": self.active,
            "max_active": self.max_active,
            "waiting": self.waiting,
            "max_waiting": self.max_waiting,
            "workload": workload,
            "workload_active": self.active_by_workload.get(workload, 0),
            "workload_limit": self.workload_limits.get(
                workload, self.max_active
            ),
            "retry_after": self.retry_after,
        }

    async def acquire(
        self,
        workload: str,
        disconnected: Callable[[], Awaitable[bool]],
    ) -> AdmissionLease:
        with self._lock:
            if self._has_capacity(workload):
                return self._reserve(workload)
            if self.waiting >= self.max_waiting or self.wait_seconds == 0:
                raise AdmissionRejected(self.retry_after)
            self.waiting += 1

        deadline = time.monotonic() + self.wait_seconds
        try:
            while time.monotonic() < deadline:
                if await disconnected():
                    raise ClientDisconnected()
                with self._lock:
                    if self._has_capacity(workload):
                        return self._reserve(workload)
                await asyncio.sleep(0.1)
        finally:
            with self._lock:
                self.waiting -= 1
        raise AdmissionRejected(self.retry_after)

    def _reserve(self, workload: str) -> AdmissionLease:
        self.active += 1
        self.active_by_workload[workload] = (
            self.active_by_workload.get(workload, 0) + 1
        )
        return AdmissionLease(self, workload)

    async def release(self, workload: str) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            remaining = self.active_by_workload.get(workload, 0) - 1
            if remaining > 0:
                self.active_by_workload[workload] = remaining
            else:
                self.active_by_workload.pop(workload, None)


def _workload_limits(max_active: int) -> dict[str, int]:
    raw = os.environ.get(
        "CLARE2_INFERENCE_WORKLOAD_LIMITS",
        '{"semantic":2,"report_assembly":1,"spam":1,"distillation":1,"default":1}',
    )
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("CLARE2_INFERENCE_WORKLOAD_LIMITS must be an object")
    return {
        str(key): min(max_active, max(1, int(value)))
        for key, value in parsed.items()
    }


def from_environment() -> AdmissionController:
    max_active = int(os.environ.get("CLARE2_INFERENCE_MAX_ACTIVE", "4"))
    return AdmissionController(
        max_active=max_active,
        max_waiting=int(os.environ.get("CLARE2_INFERENCE_MAX_WAITING", "0")),
        wait_seconds=float(
            os.environ.get("CLARE2_INFERENCE_WAIT_SECONDS", "0")
        ),
        retry_after=int(
            os.environ.get("CLARE2_INFERENCE_RETRY_AFTER_SECONDS", "30")
        ),
        workload_limits=_workload_limits(max_active),
    )
