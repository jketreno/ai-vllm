from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import app.audit as audit
import app.lifecycle as lifecycle


class _Registry:
    def __init__(self, adapters_root: pathlib.Path) -> None:
        self.adapters_root = adapters_root

    def read(self) -> dict:
        return {
            "base": {"model_id": "test/model"},
            "aliases": {"current": "approved-1", "rollback": None},
            "adapters": {
                "approved-1": {"status": "approved"},
                "stale-1": {"status": "candidate"},
            },
            "updated_at": "2026-07-28T00:00:00Z",
        }


class AuditReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.corpus_root = self.root / "corpus"
        self.adapters_root = self.root / "adapters"
        self.lock_path = self.root / "requirements.lock"
        self.lock_path.write_text("pinned==1.0\n", encoding="utf-8")
        patch.object(audit, "CORPUS_ROOT", self.corpus_root).start()
        patch.object(audit, "DEPENDENCY_LOCK", self.lock_path).start()
        patch.object(audit, "_container_inventory", return_value={}).start()
        patch.object(lifecycle, "STATE_PATH", self.root / "lifecycle.json").start()
        patch.object(
            audit,
            "_mlflow_inventory",
            return_value={
                "available": True,
                "run_statuses": {"FAILED": 3},
                "stale_running_run_ids": [],
                "latest_run": {
                    "run_id": "mlflow-1",
                    "effective_training_mode": "qlora-4bit",
                },
            },
        ).start()
        patch.object(lifecycle, "TRAINING_ENABLED", False).start()
        patch.object(lifecycle, "TRAINING_CONFIGURATION_ERROR", None).start()

    def tearDown(self):
        patch.stopall()
        self.temp.cleanup()

    def _write_project(self) -> None:
        training = self.corpus_root / "training" / "Zoo-Code"
        training.mkdir(parents=True)
        (training / "current.jsonl").write_text(
            '{"prompt":"secret source","response":"secret response"}\n',
            encoding="utf-8",
        )
        (training / "manifest.json").write_text(
            json.dumps(
                {
                    "last_updated": "2026-07-28T07:00:00Z",
                    "total_sft_pairs": 1,
                    "total_tokens": 12,
                    "private_note": "must not leak",
                }
            ),
            encoding="utf-8",
        )

    def test_report_is_deterministic_and_redacts_corpus_content(self):
        self._write_project()
        registry = _Registry(self.adapters_root)

        first = audit.build_report(registry)
        second = audit.build_report(registry)

        self.assertEqual(first, second)
        serialized = json.dumps(first, sort_keys=True)
        self.assertNotIn("secret source", serialized)
        self.assertNotIn("secret response", serialized)
        self.assertNotIn("private_note", serialized)
        project = first["projects"]["Zoo-Code"]
        self.assertEqual(project["training"]["records"], 1)
        self.assertEqual(len(project["training"]["sha256"]), 64)
        self.assertEqual(
            first["registry"]["stale_adapter_ids"],
            ["stale-1"],
        )

    def test_disabled_training_is_a_readiness_blocker_not_a_learning_blocker(self):
        self._write_project()

        report = audit.build_report(_Registry(self.adapters_root))

        self.assertFalse(report["admission"]["ready"])
        self.assertIn("training_disabled", report["admission"]["blockers"])
        self.assertEqual(report["projects"]["Zoo-Code"]["training"]["records"], 1)

    def test_latest_failure_exposes_fingerprint_but_not_traceback(self):
        failure_dir = self.adapters_root / "failed-1"
        failure_dir.mkdir(parents=True)
        (failure_dir / "failure.json").write_text(
            json.dumps(
                {
                    "created_at": "2026-07-28T07:00:00Z",
                    "project": "Zoo-Code",
                    "adapter_id": "failed-1",
                    "mlflow_run_id": "mlflow-1",
                    "error_type": "RuntimeError",
                    "error": "sensitive error detail",
                    "traceback": "sensitive traceback",
                    "traceback_sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )

        report = audit.build_report(_Registry(self.adapters_root))
        serialized = json.dumps(report, sort_keys=True)

        self.assertNotIn("sensitive error detail", serialized)
        self.assertNotIn("sensitive traceback", serialized)
        self.assertEqual(
            report["latest_trainer_failure"]["traceback_sha256"],
            "b" * 64,
        )

    def test_lifecycle_error_text_is_redacted(self):
        lifecycle.STATE_PATH.write_text(
            json.dumps(
                {
                    "phase": "failed",
                    "run_id": "run-1",
                    "error": "credential-like secret",
                    "error_type": "RuntimeError",
                    "traceback_sha256": "c" * 64,
                }
            ),
            encoding="utf-8",
        )

        report = audit.build_report(_Registry(self.adapters_root))
        serialized = json.dumps(report, sort_keys=True)

        self.assertNotIn("credential-like secret", serialized)
        self.assertEqual(report["lifecycle"]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
