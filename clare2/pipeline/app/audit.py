"""Read-only, redacted CLARE2 production baseline report."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from collections import Counter
from typing import Any

import httpx

CORPUS_ROOT = pathlib.Path(os.environ.get("CORPUS_ROOT", "/corpus"))
DEPENDENCY_LOCK = pathlib.Path(
    os.environ.get(
        "CLARE2_TRAIN_DEPENDENCY_LOCK",
        "/audit/training-requirements.lock",
    )
)
MLFLOW_URL = os.environ.get("CLARE2_MLFLOW_URL", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "clare2-qlora",
)
DOCKER_PROXY_URL = os.environ.get(
    "CLARE2_DOCKER_PROXY_URL",
    "http://docker-socket-proxy:2375",
)
TRAIN_CONTAINER = os.environ.get("CLARE2_TRAIN_CONTAINER", "clare2-train")
VLLM_CONTAINER = os.environ.get("CLARE2_VLLM_CONTAINER", "vllm-engine")
BUILD_REVISION = os.environ.get("CLARE2_BUILD_REVISION", "unknown")
REQUESTED_TRAINING_MODE = os.environ.get(
    "CLARE2_REQUESTED_TRAINING_MODE",
    "qlora-4bit",
)
BASE_CONFIG_HASH = os.environ.get("CLARE2_BASE_CONFIG_HASH", "unknown")
TOKENIZER_HASH = os.environ.get("CLARE2_TOKENIZER_HASH", "unknown")


def _file_sha256(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _jsonl_record_count(paths: list[pathlib.Path]) -> int:
    count = 0
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                count += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return count


def _project_names() -> list[str]:
    projects: set[str] = set()
    for relative in ("sessions", "episodes", "training", "themes/active"):
        root = CORPUS_ROOT / relative
        if root.is_dir():
            projects.update(path.name for path in root.iterdir() if path.is_dir())
    for level in ("weekly", "monthly", "quarterly"):
        root = CORPUS_ROOT / "summaries" / level
        if root.is_dir():
            projects.update(path.name for path in root.iterdir() if path.is_dir())
    return sorted(projects)


def _summary_counts(project: str) -> dict[str, dict[str, int]]:
    result = {}
    for level in ("weekly", "monthly", "quarterly"):
        files = sorted((CORPUS_ROOT / "summaries" / level / project).glob("*.jsonl"))
        result[level] = {
            "files": len(files),
            "records": _jsonl_record_count(files),
        }
    return result


def _project_inventory(project: str) -> dict[str, Any]:
    train_file = CORPUS_ROOT / "training" / project / "current.jsonl"
    manifest_file = CORPUS_ROOT / "training" / project / "manifest.json"
    session_files = sorted((CORPUS_ROOT / "sessions" / project).glob("**/*.jsonl"))
    episode_files = sorted((CORPUS_ROOT / "episodes" / project).glob("**/*.jsonl"))
    theme_files = sorted((CORPUS_ROOT / "themes" / "active" / project).glob("*.jsonl"))
    manifest = _read_json(manifest_file, {})
    return {
        "sessions": {
            "files": len(session_files),
            "records": _jsonl_record_count(session_files),
        },
        "episodes": {
            "files": len(episode_files),
            "records": _jsonl_record_count(episode_files),
        },
        "summaries": _summary_counts(project),
        "active_themes": {
            "files": len(theme_files),
            "records": _jsonl_record_count(theme_files),
        },
        "training": {
            "records": _jsonl_record_count([train_file]),
            "sha256": _file_sha256(train_file),
            "manifest_sha256": _file_sha256(manifest_file),
            "manifest": {
                "last_updated": manifest.get("last_updated"),
                "total_sft_pairs": manifest.get("total_sft_pairs"),
                "total_tokens": manifest.get("total_tokens"),
            },
        },
    }


def _registry_inventory(document: dict[str, Any]) -> dict[str, Any]:
    adapters = document.get("adapters", {})
    statuses = Counter(
        adapter.get("status", "unknown") for adapter in adapters.values()
    )
    stale = sorted(
        adapter_id
        for adapter_id, adapter in adapters.items()
        if adapter.get("status") in {"candidate", "training"}
    )
    return {
        "base": document.get("base", {}),
        "aliases": document.get("aliases", {}),
        "adapter_statuses": dict(sorted(statuses.items())),
        "stale_adapter_ids": stale,
        "updated_at": document.get("updated_at"),
    }


def _key_values(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(entry.get("key")): entry.get("value")
        for entry in entries
        if entry.get("key") is not None
    }


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    info = run.get("info", {})
    data = run.get("data", {})
    params = _key_values(data.get("params", []))
    tags = _key_values(data.get("tags", []))
    return {
        "run_id": info.get("run_id"),
        "status": info.get("status"),
        "adapter_id": tags.get("clare2.adapter_id"),
        "project": tags.get("clare2.project_id"),
        "corpus_hash": params.get("corpus_hash"),
        "dependency_lock_hash": params.get("dependency_lock_hash"),
        "effective_training_mode": params.get("effective_training_mode"),
    }


def _mlflow_inventory() -> dict[str, Any]:
    try:
        experiment_response = httpx.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": MLFLOW_EXPERIMENT},
            timeout=10,
        )
        experiment_response.raise_for_status()
        experiment_id = experiment_response.json()["experiment"]["experiment_id"]
        runs_response = httpx.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": [experiment_id],
                "max_results": 1000,
                "order_by": ["attributes.start_time DESC"],
            },
            timeout=30,
        )
        runs_response.raise_for_status()
        runs = runs_response.json().get("runs", [])
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        return {"available": False, "error_type": type(error).__name__}

    statuses = Counter(run.get("info", {}).get("status", "UNKNOWN") for run in runs)
    running = sorted(
        str(run.get("info", {}).get("run_id"))
        for run in runs
        if run.get("info", {}).get("status") == "RUNNING"
    )
    return {
        "available": True,
        "experiment": MLFLOW_EXPERIMENT,
        "run_statuses": dict(sorted(statuses.items())),
        "stale_running_run_ids": running,
        "latest_run": _run_summary(runs[0]) if runs else None,
    }


def _container_inventory() -> dict[str, Any]:
    containers = {}
    for role, name in (("trainer", TRAIN_CONTAINER), ("inference", VLLM_CONTAINER)):
        try:
            response = httpx.get(
                f"{DOCKER_PROXY_URL}/containers/{name}/json",
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            state = payload.get("State", {})
            containers[role] = {
                "name": name,
                "image_id": payload.get("Image"),
                "running": state.get("Running"),
                "status": state.get("Status"),
                "exit_code": state.get("ExitCode"),
            }
        except (httpx.HTTPError, TypeError, ValueError) as error:
            containers[role] = {
                "name": name,
                "available": False,
                "error_type": type(error).__name__,
            }
    return containers


def _latest_failure(adapters_root: pathlib.Path) -> dict[str, Any] | None:
    failures = []
    for path in adapters_root.glob("*/failure.json"):
        record = _read_json(path, {})
        if record:
            failures.append((str(record.get("created_at", "")), path, record))
    if not failures:
        return None
    _created_at, path, record = max(failures, key=lambda item: (item[0], str(item[1])))
    return {
        "adapter_id": record.get("adapter_id"),
        "project": record.get("project"),
        "mlflow_run_id": record.get("mlflow_run_id"),
        "error_type": record.get("error_type"),
        "traceback_sha256": record.get("traceback_sha256"),
        "artifact": str(path),
        "created_at": record.get("created_at"),
    }


def _lifecycle_inventory(state: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = (
        "phase",
        "run_id",
        "outcome",
        "project",
        "candidate_id",
        "mlflow_run_id",
        "error_type",
        "traceback_sha256",
        "traceback_artifact",
        "training_enabled",
        "configuration_error",
        "trainer_start_requested",
    )
    return {
        field: state.get(field)
        for field in allowed_fields
        if state.get(field) is not None
    }


def _readiness(
    projects: dict[str, dict[str, Any]],
    mlflow: dict[str, Any],
) -> dict[str, Any]:
    from . import lifecycle

    blockers = []
    if lifecycle.TRAINING_CONFIGURATION_ERROR:
        blockers.append("invalid_training_configuration")
    if not lifecycle.TRAINING_ENABLED:
        blockers.append("training_disabled")
    if _file_sha256(DEPENDENCY_LOCK) is None:
        blockers.append("dependency_lock_unavailable")
    if not any(item["training"]["records"] for item in projects.values()):
        blockers.append("no_nonempty_project_corpus")
    if not mlflow.get("available"):
        blockers.append("mlflow_unavailable")
    latest = mlflow.get("latest_run") or {}
    effective_mode = latest.get("effective_training_mode")
    if effective_mode and effective_mode != REQUESTED_TRAINING_MODE:
        blockers.append("effective_training_mode_mismatch")
    return {"ready": not blockers, "blockers": blockers}


def build_report(registry: Any) -> dict[str, Any]:
    from . import lifecycle

    projects = {project: _project_inventory(project) for project in _project_names()}
    mlflow = _mlflow_inventory()
    return {
        "schema_version": 1,
        "build": {
            "repository_revision": BUILD_REVISION,
            "training_dependency_lock_sha256": _file_sha256(DEPENDENCY_LOCK),
            "base_config_sha256": BASE_CONFIG_HASH,
            "tokenizer_sha256": TOKENIZER_HASH,
            "containers": _container_inventory(),
        },
        "training": {
            "requested_mode": REQUESTED_TRAINING_MODE,
            "enabled": _training_enabled(),
        },
        "projects": projects,
        "registry": _registry_inventory(registry.read()),
        "lifecycle": _lifecycle_inventory(lifecycle.status()),
        "mlflow": mlflow,
        "latest_trainer_failure": _latest_failure(registry.adapters_root),
        "admission": _readiness(projects, mlflow),
    }


def _training_enabled() -> dict[str, Any]:
    from . import lifecycle

    return {
        "value": lifecycle.TRAINING_ENABLED,
        "configuration_error": lifecycle.TRAINING_CONFIGURATION_ERROR,
    }


def main() -> None:
    from .runtime import registry

    print(json.dumps(build_report(registry), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
