from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import app.lifecycle as lifecycle
import app.metrics as metrics
from app.registry import AdapterRegistry


class CompleteTrainingSkippedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_root = pathlib.Path(self.temp.name)
        self.state_patch = patch.object(lifecycle, "STATE_ROOT", self.state_root)
        self.state_path_patch = patch.object(lifecycle, "STATE_PATH", self.state_root / "lifecycle.json")
        self.lock_path_patch = patch.object(lifecycle, "LOCK_PATH", self.state_root / "lifecycle.lock")
        self.state_patch.start()
        self.state_path_patch.start()
        self.lock_path_patch.start()

        self.container_patch = patch.object(lifecycle, "_container")
        self.wait_patch = patch.object(lifecycle, "_wait_for_vllm")
        self.controller_patch = patch.object(lifecycle, "controller")
        self.maintenance_patch = patch.object(lifecycle, "maintenance")
        self.container_patch.start()
        self.wait_patch.start()
        self.controller_patch.start()
        self.maintenance_patch.start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def _set_training_state(self, run_id: str) -> None:
        lifecycle._set_state("training", run_id=run_id)

    def test_returns_to_idle_without_touching_registry(self):
        self._set_training_state("run-1")
        result = lifecycle.complete_training_skipped("run-1")
        self.assertEqual(result["phase"], "idle")
        self.assertEqual(result["outcome"], "skipped_no_new_content")
        self.assertEqual(result["completed_adapter_id"], "skipped:run-1")

    def test_restarts_vllm_and_reconciles(self):
        self._set_training_state("run-1")
        lifecycle.complete_training_skipped("run-1")
        lifecycle._container.assert_any_call("start", lifecycle.VLLM_CONTAINER)
        lifecycle.controller.reconcile.assert_called_once()

    def test_exits_maintenance(self):
        self._set_training_state("run-1")
        lifecycle.complete_training_skipped("run-1")
        lifecycle.maintenance.exit.assert_called_once()

    def test_idempotent_on_repeated_callback(self):
        self._set_training_state("run-1")
        lifecycle.complete_training_skipped("run-1")
        lifecycle.controller.reset_mock()
        result = lifecycle.complete_training_skipped("run-1")
        self.assertEqual(result["completed_adapter_id"], "skipped:run-1")
        lifecycle.controller.reconcile.assert_not_called()

    def test_rejects_mismatched_run_id(self):
        self._set_training_state("run-1")
        with self.assertRaises(RuntimeError):
            lifecycle.complete_training_skipped("run-2")

    def test_recovers_on_failure(self):
        self._set_training_state("run-1")
        lifecycle.controller.reconcile.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            lifecycle.complete_training_skipped("run-1")
        state = lifecycle.status()
        self.assertEqual(state["phase"], "failed")


class RecordTrainingMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.models = pathlib.Path(self.temp.name)
        self.registry_patch = patch.object(lifecycle, "registry", AdapterRegistry(self.models))
        self.registry_patch.start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def _make_adapter_dir(self, adapter_id: str, duration_seconds: float | None = 12.5) -> pathlib.Path:
        adapter_dir = self.models / "adapters" / adapter_id
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"0" * 1000)
        meta = {"duration_seconds": duration_seconds} if duration_seconds is not None else {}
        (adapter_dir / "training_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return adapter_dir

    def test_labels_loss_metrics_by_project(self):
        self._make_adapter_dir("clare-ai-vllm-20260101T000000Z-aaaa")
        lifecycle._record_training_metrics(
            "clare-ai-vllm-20260101T000000Z-aaaa", "ai-vllm", 0.42, [1.0, 0.7, 0.42]
        )
        self.assertEqual(metrics.training_loss_final.labels(project="ai-vllm")._value.get(), 0.42)
        self.assertEqual(
            metrics.training_loss_by_epoch.labels(project="ai-vllm", epoch="3")._value.get(), 0.42
        )

    def test_labels_duration_from_training_meta(self):
        self._make_adapter_dir("clare-ai-vllm-20260101T000000Z-bbbb", duration_seconds=99.0)
        lifecycle._record_training_metrics(
            "clare-ai-vllm-20260101T000000Z-bbbb", "ai-vllm", None, []
        )
        self.assertEqual(
            metrics.training_duration_seconds.labels(project="ai-vllm")._value.get(), 99.0
        )

    def test_labels_adapter_size_by_project(self):
        self._make_adapter_dir("clare-clare-20260101T000000Z-cccc")
        lifecycle._record_training_metrics(
            "clare-clare-20260101T000000Z-cccc", "clare", None, []
        )
        size = metrics.adapter_size_bytes.labels(project="clare")._value.get()
        self.assertGreaterEqual(size, 1000)

    def test_missing_training_meta_does_not_raise(self):
        adapter_dir = self.models / "adapters" / "clare-ai-vllm-20260101T000000Z-dddd"
        adapter_dir.mkdir(parents=True)
        lifecycle._record_training_metrics(
            "clare-ai-vllm-20260101T000000Z-dddd", "ai-vllm", 0.1, [0.1]
        )
        # loss metrics are still recorded even though training_meta.json is absent
        self.assertEqual(metrics.training_loss_final.labels(project="ai-vllm")._value.get(), 0.1)


if __name__ == "__main__":
    unittest.main()
