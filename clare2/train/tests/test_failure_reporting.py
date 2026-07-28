from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from failure_reporting import load_callback_context, write_failure_artifact


class FailureReportingTests(unittest.TestCase):
    def test_writes_full_traceback_and_returns_redacted_callback_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = pathlib.Path(temporary)
            try:
                raise RuntimeError("backward pass failed")
            except RuntimeError as error:
                record = write_failure_artifact(
                    output_dir,
                    project="Zoo-Code",
                    adapter_id="adapter-1",
                    mlflow_run_id="mlflow-1",
                    error=error,
                )

            artifact = output_dir / "failure.json"
            persisted = json.loads(artifact.read_text(encoding="utf-8"))
            callback = load_callback_context(artifact)

        self.assertEqual(record, persisted)
        self.assertIn("RuntimeError: backward pass failed", persisted["traceback"])
        self.assertEqual(len(persisted["traceback_sha256"]), 64)
        self.assertEqual(callback["project"], "Zoo-Code")
        self.assertEqual(callback["adapter_id"], "adapter-1")
        self.assertNotIn("error", callback)
        self.assertNotIn("traceback", callback)

    def test_missing_or_invalid_artifact_has_empty_callback_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "failure.json"
            self.assertEqual(load_callback_context(path), {})
            path.write_text("{invalid", encoding="utf-8")
            self.assertEqual(load_callback_context(path), {})


if __name__ == "__main__":
    unittest.main()
