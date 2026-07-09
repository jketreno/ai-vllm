from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import app.lifecycle as lifecycle


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


if __name__ == "__main__":
    unittest.main()
