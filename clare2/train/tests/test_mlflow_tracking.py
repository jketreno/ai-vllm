from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import mlflow_tracking


class TrainingTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mlflow = MagicMock()
        self.mlflow.start_run.return_value.info.run_id = "mlflow-run-1"
        self.patch = patch.object(mlflow_tracking, "mlflow", self.mlflow)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_tracks_lineage_metrics_and_artifacts(self) -> None:
        tracker = mlflow_tracking.TrainingTracker(
            lifecycle_run_id="lifecycle-run-1",
            adapter_id="clare-project-20260611T000000Z-12345678",
            project_id="github:example/project",
        )

        run_id = tracker.start(
            {"target_modules": ["q_proj", "v_proj"], "rank": 32},
            {"clare2.stage": "training"},
        )
        tracker.log_metric("train.loss", 0.25, step=7)
        tracker.log_params({"config_hash": "abc"})
        tracker.log_dict({"blank": 2}, "corpus/skipped_records.json")
        with tempfile.TemporaryDirectory() as directory:
            pathlib.Path(directory, "adapter_config.json").write_text("{}")
            pathlib.Path(directory, "checkpoint-1").mkdir()
            tracker.log_adapter_artifacts(pathlib.Path(directory))
        tracker.finish()

        self.assertEqual(run_id, "mlflow-run-1")
        self.mlflow.set_experiment.assert_called_once_with("clare2-qlora")
        tags = self.mlflow.start_run.call_args.kwargs["tags"]
        self.assertEqual(tags["clare2.lifecycle_run_id"], "lifecycle-run-1")
        self.assertEqual(tags["clare2.project_id"], "github:example/project")
        params = self.mlflow.log_params.call_args_list[0].args[0]
        self.assertEqual(params["target_modules"], '["q_proj", "v_proj"]')
        self.assertEqual(self.mlflow.log_params.call_args_list[1].args[0]["config_hash"], "abc")
        self.mlflow.log_metric.assert_called_once_with("train.loss", 0.25, step=7)
        self.mlflow.log_artifact.assert_called_once()
        self.assertIn("adapter_config.json", self.mlflow.log_artifact.call_args.args[0])
        self.mlflow.end_run.assert_called_once_with(status="FINISHED")

    def test_failed_run_is_explicitly_terminated(self) -> None:
        tracker = mlflow_tracking.TrainingTracker(
            lifecycle_run_id="lifecycle-run-2",
            adapter_id="adapter",
            project_id="global",
        )
        tracker.start({}, {})
        tracker.finish("FAILED")
        self.mlflow.end_run.assert_called_once_with(status="FAILED")


if __name__ == "__main__":
    unittest.main()
