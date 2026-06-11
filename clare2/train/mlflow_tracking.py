"""Explicit MLflow tracking for immutable CLARE2 adapter training runs."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import mlflow


class TrainingTracker:
    def __init__(
        self,
        *,
        lifecycle_run_id: str,
        adapter_id: str,
        project_id: str,
    ) -> None:
        self.lifecycle_run_id = lifecycle_run_id
        self.adapter_id = adapter_id
        self.project_id = project_id
        self.tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://mlflow:5000",
        )
        self.experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            "clare2-qlora",
        )
        self.mlflow_run_id: str | None = None

    def start(self, params: dict[str, Any], tags: dict[str, Any]) -> str:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(
            run_name=self.adapter_id,
            tags={
                "clare2.lifecycle_run_id": self.lifecycle_run_id,
                "clare2.adapter_id": self.adapter_id,
                "clare2.project_id": self.project_id,
                **{key: str(value) for key, value in tags.items()},
            },
        )
        self.mlflow_run_id = run.info.run_id
        self.log_params(params)
        return self.mlflow_run_id

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params({key: self._param(value) for key, value in params.items()})

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_dict(self, value: dict[str, Any], artifact_file: str) -> None:
        mlflow.log_dict(value, artifact_file)

    def log_adapter_artifacts(self, output_dir: pathlib.Path) -> None:
        for artifact in sorted(output_dir.iterdir()):
            if artifact.is_file():
                mlflow.log_artifact(str(artifact), artifact_path="adapter")

    def finish(self, status: str = "FINISHED") -> None:
        if self.mlflow_run_id is not None:
            mlflow.end_run(status=status)

    @staticmethod
    def _param(value: Any) -> str | int | float | bool:
        if isinstance(value, (str, int, float, bool)):
            return value
        return json.dumps(value, sort_keys=True)
