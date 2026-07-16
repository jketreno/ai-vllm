from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import httpx

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
        self.notify_patch = patch.object(lifecycle, "notify")
        self.container_patch.start()
        self.wait_patch.start()
        self.controller_patch.start()
        self.maintenance_patch.start()
        self.notify_patch.start()

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

    def test_sends_skipped_notification(self):
        self._set_training_state("run-1")
        lifecycle.complete_training_skipped("run-1")
        lifecycle.notify.send_run_notification.assert_called_once_with(
            "skipped_no_new_content", run_id="run-1"
        )

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

    def test_sends_failed_notification_on_recovery(self):
        self._set_training_state("run-1")
        lifecycle.controller.reconcile.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            lifecycle.complete_training_skipped("run-1")
        lifecycle.notify.send_run_notification.assert_called_once_with(
            "failed", run_id="run-1", adapter_id=None, error="boom"
        )

    def test_reconciles_terminal_outcome_with_stale_phase(self):
        lifecycle._set_state("evaluating", run_id="run-1", outcome="rejected")
        state = lifecycle.reconcile_terminal_state()
        self.assertEqual(state["phase"], "idle")
        self.assertEqual(state["outcome"], "rejected")

    def test_reconciles_rejected_candidate_registry_state(self):
        adapter_id = "adapter-1"
        with patch.object(lifecycle, "registry") as registry:
            registry.read.return_value = {"adapters": {adapter_id: {"status": "candidate"}}}
            lifecycle._set_state("evaluating", run_id="run-1", candidate_id=adapter_id, outcome="rejected")
            lifecycle.reconcile_terminal_state()
            registry.transition.assert_called_once_with(adapter_id, "rejected")


class NightlyTrainingAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_root = pathlib.Path(self.temp.name)
        patch.object(lifecycle, "STATE_ROOT", self.state_root).start()
        patch.object(lifecycle, "STATE_PATH", self.state_root / "lifecycle.json").start()
        patch.object(lifecycle, "LOCK_PATH", self.state_root / "lifecycle.lock").start()
        patch.object(lifecycle, "TRAINING_RETRY_INTERVAL", 0).start()
        patch.object(lifecycle, "maintenance").start()
        patch.object(lifecycle, "notify").start()
        patch.object(lifecycle, "corpus").start()
        patch.object(lifecycle, "_container").start()
        patch.object(lifecycle.time, "sleep").start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def test_postpones_once_then_refreshes_corpus_and_trains(self):
        with patch.object(lifecycle, "_active_inference_sessions", side_effect=[2, 1, 0]):
            lifecycle.run_nightly_training()

        lifecycle.notify.send_run_notification.assert_called_once()
        outcome, = lifecycle.notify.send_run_notification.call_args.args
        self.assertEqual(outcome, "postponed")
        self.assertEqual(lifecycle.time.sleep.call_count, 2)
        lifecycle.corpus.assemble.assert_called_once_with()
        self.assertEqual(
            lifecycle._container.call_args_list,
            [
                unittest.mock.call("stop", lifecycle.VLLM_CONTAINER),
                unittest.mock.call("start", lifecycle.TRAIN_CONTAINER),
            ],
        )

    def test_prometheus_failure_is_fail_closed_without_email(self):
        with patch.object(
            lifecycle, "_active_inference_sessions", side_effect=[httpx.ConnectError("down"), 0]
        ):
            lifecycle.run_nightly_training()

        lifecycle.notify.send_run_notification.assert_not_called()
        lifecycle.time.sleep.assert_called_once_with(0)

    def test_active_session_query_sums_prometheus_results(self):
        response = unittest.mock.Mock()
        response.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": [1, "2"]}, {"value": [1, "1.0"]}]},
        }
        with patch.object(lifecycle.httpx, "get", return_value=response) as get:
            self.assertEqual(lifecycle._active_inference_sessions(), 3)
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(get.call_args.kwargs["params"], {"query": lifecycle.ACTIVE_INFERENCE_QUERY})

    def test_active_session_query_rejects_missing_metric(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"status": "success", "data": {"result": []}}
        with patch.object(lifecycle.httpx, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "no active-inference metric"):
                lifecycle._active_inference_sessions()


class ApplyEvaluationNotificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_root = pathlib.Path(self.temp.name)
        self.state_patch = patch.object(lifecycle, "STATE_ROOT", self.state_root)
        self.state_path_patch = patch.object(lifecycle, "STATE_PATH", self.state_root / "lifecycle.json")
        self.lock_path_patch = patch.object(lifecycle, "LOCK_PATH", self.state_root / "lifecycle.lock")
        self.state_patch.start()
        self.state_path_patch.start()
        self.lock_path_patch.start()

        self.registry_patch = patch.object(lifecycle, "registry")
        self.controller_patch = patch.object(lifecycle, "controller")
        self.notify_patch = patch.object(lifecycle, "notify")
        self.registry_patch.start()
        self.controller_patch.start()
        self.notify_patch.start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def test_sends_rejected_notification_with_report_and_project(self):
        report = {"approved": False, "candidate": {"pass_rate": 0.1}}
        lifecycle._apply_evaluation("adapter-1", "run-1", "mlflow-1", report, project="ai-vllm")
        lifecycle.notify.send_run_notification.assert_called_once_with(
            "rejected",
            adapter_id="adapter-1",
            run_id="run-1",
            mlflow_run_id="mlflow-1",
            report=report,
            project="ai-vllm",
        )

    def test_sends_promoted_notification_with_report_and_project(self):
        report = {"approved": True, "candidate": {"pass_rate": 1.0}}
        lifecycle._apply_evaluation("adapter-1", "run-1", "mlflow-1", report, project="ai-vllm")
        lifecycle.notify.send_run_notification.assert_called_once_with(
            "promoted",
            adapter_id="adapter-1",
            run_id="run-1",
            mlflow_run_id="mlflow-1",
            report=report,
            project="ai-vllm",
        )


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
