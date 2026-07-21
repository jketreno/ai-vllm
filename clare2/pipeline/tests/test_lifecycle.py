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


class ImageEditLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.temp.name)
        patch.object(lifecycle, "STATE_ROOT", root).start()
        patch.object(lifecycle, "STATE_PATH", root / "lifecycle.json").start()
        patch.object(lifecycle, "LOCK_PATH", root / "lifecycle.lock").start()
        patch.object(lifecycle, "maintenance").start()
        patch.object(lifecycle, "_container").start()
        patch.object(lifecycle, "_wait_for_vllm").start()
        patch.object(lifecycle, "_wait_for_image_memory", return_value=24.0).start()
        patch.object(lifecycle, "DRAIN_TIMEOUT", 0).start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def test_acquire_stops_vllm_and_records_exclusive_lease(self):
        result = lifecycle.acquire_image_edit_lease("request-1")

        self.assertEqual(result["phase"], "image_edit")
        self.assertEqual(result["request_id"], "request-1")
        self.assertEqual(result["mem_available_gib"], 24.0)
        lifecycle.maintenance.enter.assert_called_once()
        lifecycle._container.assert_called_once_with("stop", lifecycle.VLLM_CONTAINER)

    def test_release_restarts_vllm_and_is_idempotent(self):
        lease = lifecycle.acquire_image_edit_lease("request-1")
        lifecycle._container.reset_mock()

        released = lifecycle.release_image_edit_lease(lease["lease_id"])
        repeated = lifecycle.release_image_edit_lease(lease["lease_id"])

        self.assertEqual(released["status"], "released")
        self.assertEqual(repeated["status"], "already_released")
        lifecycle._container.assert_called_once_with("start", lifecycle.VLLM_CONTAINER)
        lifecycle._wait_for_vllm.assert_called_once()
        lifecycle.maintenance.exit.assert_called_once()

    def test_second_acquire_times_out_while_lease_is_active(self):
        lifecycle.acquire_image_edit_lease("request-1")

        with self.assertRaisesRegex(RuntimeError, "timed out waiting"):
            lifecycle.acquire_image_edit_lease("request-2")

    def test_failed_restore_keeps_maintenance_and_schedules_ttl_retry(self):
        lease = lifecycle.acquire_image_edit_lease("request-1")
        lifecycle._wait_for_vllm.side_effect = RuntimeError("not healthy")

        with self.assertRaisesRegex(RuntimeError, "not healthy"):
            lifecycle.release_image_edit_lease(lease["lease_id"])

        self.assertEqual(lifecycle.status()["phase"], "image_edit")
        lifecycle.maintenance.exit.assert_not_called()


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
        self.assertFalse(result["trainer_start_requested"])

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
        self.container_exists_patch = patch.object(
            lifecycle, "_container_exists", return_value=True
        )
        self.container_exists_patch.start()
        patch.object(lifecycle.time, "sleep").start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def test_postpones_once_then_refreshes_corpus_and_trains(self):
        lifecycle._set_state(
            "idle",
            run_id="old-run",
            dream_mode=True,
            outcome="rejected",
            evaluation={"approved": False},
        )
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
        state = lifecycle.status()
        self.assertNotIn("dream_mode", state)
        self.assertNotIn("outcome", state)
        self.assertNotIn("evaluation", state)
        self.assertEqual(state["phase"], "training")
        self.assertTrue(state["trainer_start_requested"])

    def test_waits_for_training_container_before_stopping_inference(self):
        with patch.object(lifecycle, "_active_inference_sessions", return_value=0), patch.object(
            lifecycle, "_container_exists", side_effect=[False, True]
        ):
            lifecycle.run_nightly_training()

        lifecycle.time.sleep.assert_called_once_with(lifecycle.TRAINING_RETRY_INTERVAL)
        self.assertEqual(
            lifecycle._container.call_args_list,
            [
                unittest.mock.call("stop", lifecycle.VLLM_CONTAINER),
                unittest.mock.call("start", lifecycle.TRAIN_CONTAINER),
            ],
        )
        self.assertEqual(lifecycle.status()["phase"], "training")

    def test_retries_when_training_container_disappears_during_start(self):
        missing = httpx.Response(
            404,
            request=httpx.Request("POST", "http://docker/containers/clare2-train/start"),
        )
        lifecycle._container.side_effect = [
            None,
            httpx.HTTPStatusError("missing", request=missing.request, response=missing),
            None,
        ]

        with patch.object(lifecycle, "_active_inference_sessions", return_value=0):
            lifecycle.run_nightly_training()

        lifecycle.time.sleep.assert_called_once_with(lifecycle.TRAINING_RETRY_INTERVAL)
        self.assertEqual(lifecycle.status()["phase"], "training")
        self.assertTrue(lifecycle.status()["trainer_start_requested"])

    def test_container_existence_maps_not_found_to_false(self):
        self.container_exists_patch.stop()
        response = unittest.mock.Mock(status_code=404)
        with patch.object(lifecycle.httpx, "get", return_value=response):
            self.assertFalse(lifecycle._container_exists(lifecycle.TRAIN_CONTAINER))
        response.raise_for_status.assert_not_called()

    def test_prometheus_failure_is_fail_closed_without_email(self):
        with patch.object(
            lifecycle, "_active_inference_sessions", side_effect=[httpx.ConnectError("down"), 0]
        ):
            lifecycle.run_nightly_training()

        lifecycle.notify.send_run_notification.assert_not_called()
        lifecycle.time.sleep.assert_called_once_with(0)

    def test_failed_run_does_not_block_a_fresh_nightly_run(self):
        lifecycle._set_state("failed", run_id="failed-run", error="old failure")
        with patch.object(lifecycle, "_active_inference_sessions", return_value=0):
            lifecycle.run_nightly_training()

        state = lifecycle.status()
        self.assertEqual(state["phase"], "training")
        self.assertNotEqual(state["run_id"], "failed-run")
        self.assertNotIn("error", state)

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


class ApplyEvaluationTests(unittest.TestCase):
    """_apply_evaluation only mutates the registry and reports the outcome;
    sending a notification is the caller's responsibility (single-adapter vs.
    batch callers each compose their own notification around it)."""

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
        self.registry = self.registry_patch.start()
        self.controller_patch.start()
        self.notify_patch.start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def test_rejects_and_returns_outcome_without_notifying(self):
        report = {"approved": False, "candidate": {"pass_rate": 0.1}}
        outcome = lifecycle._apply_evaluation("adapter-1", "run-1", "mlflow-1", report, project="ai-vllm")
        self.assertEqual(outcome, "rejected")
        self.registry.transition.assert_called_once_with("adapter-1", "rejected")
        lifecycle.notify.send_run_notification.assert_not_called()

    def test_promotes_and_returns_outcome_without_notifying(self):
        report = {"approved": True, "candidate": {"pass_rate": 1.0}}
        outcome = lifecycle._apply_evaluation("adapter-1", "run-1", "mlflow-1", report, project="ai-vllm")
        self.assertEqual(outcome, "promoted")
        self.registry.promote.assert_called_once_with("adapter-1", report)
        lifecycle.notify.send_run_notification.assert_not_called()


class CompleteTrainingBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_root = pathlib.Path(self.temp.name) / "state"
        self.models_root = pathlib.Path(self.temp.name) / "models"
        self.state_patch = patch.object(lifecycle, "STATE_ROOT", self.state_root)
        self.state_path_patch = patch.object(lifecycle, "STATE_PATH", self.state_root / "lifecycle.json")
        self.lock_path_patch = patch.object(lifecycle, "LOCK_PATH", self.state_root / "lifecycle.lock")
        self.state_patch.start()
        self.state_path_patch.start()
        self.lock_path_patch.start()

        self.registry_patch = patch.object(lifecycle, "registry")
        self.registry = self.registry_patch.start()
        self.registry.adapters_root = self.models_root / "adapters"
        self.registry.read.return_value = {"adapters": {}, "aliases": {"current": None}}

        self.controller_patch = patch.object(lifecycle, "controller")
        self.controller_patch.start()
        self.maintenance_patch = patch.object(lifecycle, "maintenance")
        self.maintenance_patch.start()
        self.notify_patch = patch.object(lifecycle, "notify")
        self.notify_patch.start()
        self.container_patch = patch.object(lifecycle, "_container")
        self.container_patch.start()
        self.wait_patch = patch.object(lifecycle, "_wait_for_vllm")
        self.wait_patch.start()

        self.evaluator_patch = patch.object(lifecycle, "evaluator")
        self.evaluator = self.evaluator_patch.start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def _write_candidate(self, adapter_id: str, project: str) -> None:
        adapter_dir = self.registry.adapters_root / adapter_id
        adapter_dir.mkdir(parents=True)
        manifest = {"id": adapter_id, "project_scope": project, "status": "candidate"}
        (adapter_dir / "candidate_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _set_training_state(self, run_id: str) -> None:
        lifecycle._set_state("training", run_id=run_id)

    def test_evaluates_every_project_independently(self):
        self._write_candidate("clare-ai-vllm-1", "ai-vllm")
        self._write_candidate("clare-clare-1", "clare")
        self._set_training_state("run-1")
        self.evaluator.compare.side_effect = [
            {"approved": False, "candidate": {}, "baseline": {}},
            {"approved": True, "candidate": {}, "baseline": {}},
        ]

        result = lifecycle.complete_training_batch(
            "run-1",
            [{"adapter_id": "clare-ai-vllm-1"}, {"adapter_id": "clare-clare-1"}],
        )

        self.assertEqual(result["outcome"], "batch_complete")
        self.assertFalse(result["trainer_start_requested"])
        outcomes = {r["project"]: r["outcome"] for r in result["batch_results"]}
        self.assertEqual(outcomes, {"ai-vllm": "rejected", "clare": "promoted"})
        self.registry.transition.assert_called_once_with("clare-ai-vllm-1", "rejected")
        self.registry.promote.assert_called_once()

    def test_restarts_vllm_exactly_once_for_the_whole_batch(self):
        self._write_candidate("clare-ai-vllm-1", "ai-vllm")
        self._write_candidate("clare-clare-1", "clare")
        self._set_training_state("run-1")
        self.evaluator.compare.return_value = {"approved": False, "candidate": {}, "baseline": {}}

        lifecycle.complete_training_batch(
            "run-1",
            [{"adapter_id": "clare-ai-vllm-1"}, {"adapter_id": "clare-clare-1"}],
        )

        lifecycle._container.assert_called_once_with("start", lifecycle.VLLM_CONTAINER)
        lifecycle._wait_for_vllm.assert_called_once()

    def test_sends_one_batch_notification(self):
        self._write_candidate("clare-ai-vllm-1", "ai-vllm")
        self._set_training_state("run-1")
        self.evaluator.compare.return_value = {"approved": True, "candidate": {}, "baseline": {}}

        lifecycle.complete_training_batch("run-1", [{"adapter_id": "clare-ai-vllm-1"}])

        lifecycle.notify.send_batch_run_notification.assert_called_once()
        lifecycle.notify.send_run_notification.assert_not_called()

    def test_idempotent_on_repeated_callback(self):
        self._write_candidate("clare-ai-vllm-1", "ai-vllm")
        self._set_training_state("run-1")
        self.evaluator.compare.return_value = {"approved": True, "candidate": {}, "baseline": {}}

        lifecycle.complete_training_batch("run-1", [{"adapter_id": "clare-ai-vllm-1"}])
        lifecycle._container.reset_mock()
        result = lifecycle.complete_training_batch("run-1", [{"adapter_id": "clare-ai-vllm-1"}])

        self.assertEqual(result["completed_run_id"], "run-1")
        lifecycle._container.assert_not_called()

    def test_recovers_on_mid_batch_failure_without_losing_earlier_outcome(self):
        self._write_candidate("clare-ai-vllm-1", "ai-vllm")
        # "missing-adapter" has no candidate_manifest.json on disk, so its
        # read fails after clare-ai-vllm-1 has already been promoted.
        self._set_training_state("run-1")
        self.evaluator.compare.return_value = {"approved": True, "candidate": {}, "baseline": {}}

        with self.assertRaises(FileNotFoundError):
            lifecycle.complete_training_batch(
                "run-1",
                [{"adapter_id": "clare-ai-vllm-1"}, {"adapter_id": "missing-adapter"}],
            )

        self.registry.promote.assert_called_once_with("clare-ai-vllm-1", unittest.mock.ANY)
        state = lifecycle.status()
        self.assertEqual(state["phase"], "failed")
        self.assertEqual(state["candidate_id"], "missing-adapter")

    def test_rejects_mismatched_run_id(self):
        self._set_training_state("run-1")
        with self.assertRaises(RuntimeError):
            lifecycle.complete_training_batch("run-2", [])


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
