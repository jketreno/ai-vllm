"""Immutable adapter registry and compatibility validation."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable

ADAPTER_STATES = {
    "training",
    "candidate",
    "approved",
    "rejected",
    "loaded",
    "retired",
    "failed",
}
ALLOWED_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
IMMUTABLE_ID = re.compile(r"^clare-[a-z0-9][a-z0-9-]*-\d{8}T\d{6}Z-[0-9a-f]{8,64}$")


class RegistryError(ValueError):
    """Registry contents or an adapter artifact are invalid."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_registry(base: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base": base,
        "adapters": {},
        "aliases": {"current": None, "rollback": None},
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


class AdapterRegistry:
    def __init__(
        self,
        models_root: pathlib.Path | str,
        *,
        safetensors_validator: Callable[[pathlib.Path], None] | None = None,
    ) -> None:
        self.models_root = pathlib.Path(models_root).resolve()
        self.adapters_root = (self.models_root / "adapters").resolve()
        self.path = self.adapters_root / "registry.json"
        self._lock = threading.RLock()
        self._validate_safetensors = safetensors_validator or self._basic_safetensors_validation

    def read(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                raise RegistryError(f"registry not found: {self.path}")
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RegistryError(f"cannot read registry: {exc}") from exc
            self.validate_document(data)
            return data

    def initialize(self, base: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.path.exists():
                document = self.read()
                if self._can_refresh_base(document, base):
                    document["base"] = base
                    document["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
                    self.validate_document(document)
                    self._atomic_write(document)
                return document
            document = empty_registry(base)
            self.validate_document(document)
            self._atomic_write(document)
            return document

    def update(self, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self._lock:
            document = self.read()
            updated = deepcopy(document)
            mutate(updated)
            updated["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
            self.validate_document(updated)
            self._atomic_write(updated)
            return updated

    def add_adapter(self, adapter: dict[str, Any]) -> dict[str, Any]:
        adapter_id = adapter.get("id", "")
        self.validate_adapter(adapter)

        def add(document: dict[str, Any]) -> None:
            if adapter_id in document["adapters"]:
                raise RegistryError(f"duplicate adapter id: {adapter_id}")
            self._validate_base_compatibility(document["base"], adapter)
            document["adapters"][adapter_id] = adapter

        return self.update(add)

    def transition(self, adapter_id: str, status: str) -> dict[str, Any]:
        if status not in ADAPTER_STATES:
            raise RegistryError(f"unsupported adapter status: {status}")

        def change(document: dict[str, Any]) -> None:
            adapter = self._adapter(document, adapter_id)
            adapter["status"] = status

        return self.update(change)

    def promote(self, adapter_id: str, evaluation: dict[str, Any]) -> dict[str, Any]:
        def change(document: dict[str, Any]) -> None:
            candidate = self._adapter(document, adapter_id)
            if candidate["status"] not in {"candidate", "loaded"}:
                raise RegistryError("only a candidate or loaded adapter may be promoted")
            previous = document["aliases"]["current"]
            if previous and previous != adapter_id:
                document["adapters"][previous]["status"] = "approved"
            candidate["status"] = "approved"
            candidate["evaluation"] = evaluation
            document["aliases"]["rollback"] = previous
            document["aliases"]["current"] = adapter_id

        return self.update(change)

    def rollback(self) -> tuple[dict[str, Any], str]:
        target: list[str] = []

        def change(document: dict[str, Any]) -> None:
            rollback_id = document["aliases"]["rollback"]
            if not rollback_id:
                raise RegistryError("no rollback adapter is registered")
            self._adapter(document, rollback_id)
            current = document["aliases"]["current"]
            document["aliases"]["current"] = rollback_id
            document["aliases"]["rollback"] = current
            target.append(rollback_id)

        return self.update(change), target[0]

    def adapter_path(self, adapter: dict[str, Any]) -> pathlib.Path:
        raw = adapter.get("directory")
        if not isinstance(raw, str) or not raw:
            raise RegistryError("adapter directory is required")
        if raw != adapter.get("id"):
            raise RegistryError("adapter directory must equal its immutable id")
        path = (self.adapters_root / raw).resolve()
        if path.parent != self.adapters_root:
            raise RegistryError("adapter directory must be an immutable direct child")
        if path.is_symlink() or not path.is_dir():
            raise RegistryError("adapter directory must be a real directory")
        try:
            path.relative_to(self.adapters_root)
        except ValueError as exc:
            raise RegistryError("adapter directory escapes models root") from exc
        return path

    def validate_adapter(self, adapter: dict[str, Any]) -> None:
        adapter_id = adapter.get("id", "")
        if not isinstance(adapter_id, str) or not IMMUTABLE_ID.fullmatch(adapter_id):
            raise RegistryError(f"malformed immutable adapter id: {adapter_id}")
        if adapter.get("status") not in ADAPTER_STATES:
            raise RegistryError("invalid adapter status")
        rank = adapter.get("peft", {}).get("rank")
        if not isinstance(rank, int) or rank < 1 or rank > 32:
            raise RegistryError("adapter rank must be between 1 and 32")
        modules = set(adapter.get("target_modules", []))
        if not modules or not modules <= ALLOWED_TARGET_MODULES:
            raise RegistryError("adapter target modules are unsupported")
        path = self.adapter_path(adapter)
        config_path = path / "adapter_config.json"
        weights_path = path / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise RegistryError("adapter config and safetensors are required")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"invalid adapter config: {exc}") from exc
        config_rank = config.get("r")
        if config_rank is not None and config_rank != rank:
            raise RegistryError("adapter rank does not match PEFT config")
        config_modules = set(config.get("target_modules", []))
        if config_modules and config_modules != modules:
            raise RegistryError("target modules do not match PEFT config")
        self._validate_safetensors(weights_path)

    def validate_document(self, document: dict[str, Any]) -> None:
        if document.get("schema_version") != 1:
            raise RegistryError("unsupported registry schema")
        base = document.get("base")
        required_base = {
            "model_id",
            "revision",
            "architecture",
            "config_hash",
            "tokenizer_hash",
            "inference_quantization",
        }
        if not isinstance(base, dict) or not required_base <= base.keys():
            raise RegistryError("base model fingerprint is incomplete")
        adapters = document.get("adapters")
        aliases = document.get("aliases")
        if not isinstance(adapters, dict) or not isinstance(aliases, dict):
            raise RegistryError("registry adapters and aliases must be objects")
        for adapter_id, adapter in adapters.items():
            if adapter.get("id") != adapter_id:
                raise RegistryError("adapter map key must match adapter id")
            self.validate_adapter(adapter)
        for alias in ("current", "rollback"):
            value = aliases.get(alias)
            if value is not None and value not in adapters:
                raise RegistryError(f"{alias} references an unknown adapter")

    @staticmethod
    def _adapter(document: dict[str, Any], adapter_id: str) -> dict[str, Any]:
        try:
            return document["adapters"][adapter_id]
        except KeyError as exc:
            raise RegistryError(f"unknown adapter: {adapter_id}") from exc

    @staticmethod
    def _can_refresh_base(document: dict[str, Any], base: dict[str, Any]) -> bool:
        adapters = document.get("adapters", {})
        inactive = all(
            adapter.get("status") in {"rejected", "failed", "retired"}
            for adapter in adapters.values()
        )
        return (
            document.get("base") != base
            and inactive
            and document.get("aliases", {}).get("current") is None
            and document.get("aliases", {}).get("rollback") is None
        )

    @staticmethod
    def _validate_base_compatibility(base: dict[str, Any], adapter: dict[str, Any]) -> None:
        adapter_base = adapter.get("inference_base") or adapter.get("base", {})
        for field in ("model_id", "revision", "config_hash", "tokenizer_hash"):
            if adapter_base.get(field) != base.get(field):
                raise RegistryError(f"adapter inference fingerprint mismatch: {field}")
        if adapter_base.get("architecture") and adapter_base.get("architecture") != base.get("architecture"):
            raise RegistryError("adapter inference fingerprint mismatch: architecture")
        if (
            adapter_base.get("inference_quantization")
            and adapter_base.get("inference_quantization") != base.get("inference_quantization")
        ):
            raise RegistryError("adapter inference fingerprint mismatch: inference_quantization")

    @staticmethod
    def _basic_safetensors_validation(path: pathlib.Path) -> None:
        size = path.stat().st_size
        if size < 9:
            raise RegistryError("safetensors file is truncated")
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), byteorder="little")
            if header_size <= 1 or header_size > size - 8:
                raise RegistryError("invalid safetensors header length")
            try:
                header = json.loads(handle.read(header_size))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RegistryError("invalid safetensors header") from exc
            if not isinstance(header, dict):
                raise RegistryError("invalid safetensors metadata")

    def _atomic_write(self, document: dict[str, Any]) -> None:
        self.adapters_root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=".registry.", dir=self.adapters_root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.adapters_root, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
