"""Durable, bounded metadata for a failed trainer process."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import traceback
from datetime import datetime, timezone
from typing import Any

CALLBACK_FIELDS = (
    "project",
    "adapter_id",
    "mlflow_run_id",
    "error_type",
    "traceback_sha256",
    "traceback_artifact",
)


def write_failure_artifact(
    output_dir: pathlib.Path,
    *,
    project: str,
    adapter_id: str,
    mlflow_run_id: str | None,
    error: Exception,
) -> dict[str, Any]:
    traceback_text = traceback.format_exc()
    traceback_sha256 = hashlib.sha256(traceback_text.encode("utf-8")).hexdigest()
    artifact_path = output_dir / "failure.json"
    record = {
        "schema_version": 1,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "project": project,
        "adapter_id": adapter_id,
        "mlflow_run_id": mlflow_run_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback_text,
        "traceback_sha256": traceback_sha256,
        "traceback_artifact": str(artifact_path),
    }
    temporary = output_dir / ".failure.json.tmp"
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, artifact_path)
    return record


def load_callback_context(path: pathlib.Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {field: record.get(field) for field in CALLBACK_FIELDS if record.get(field)}
