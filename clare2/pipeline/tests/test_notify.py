from __future__ import annotations

import json
import pathlib
import smtplib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import app.corpus as corpus
import app.metrics as metrics
import app.notify as notify


class SendRunNotificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.corpus_root = pathlib.Path(self.temp.name)
        self.corpus_patch = patch.object(corpus, "CORPUS_ROOT", self.corpus_root)
        self.corpus_patch.start()

        self.to_patch = patch.object(notify, "NOTIFY_TO", "james_clare2@ketrenos.com")
        self.to_patch.start()

        self.smtp_instance = MagicMock()
        self.smtp_instance.send_message.return_value = {}
        self.smtp_cm = MagicMock()
        self.smtp_cm.__enter__.return_value = self.smtp_instance
        self.smtp_cm.__exit__.return_value = False
        self.smtp_patch = patch.object(notify.smtplib, "SMTP", return_value=self.smtp_cm)
        self.smtp_mock = self.smtp_patch.start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def _write_corpus_stats(self) -> None:
        stats_path = self.corpus_root / "meta" / "corpus_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps({"episodes": {"domain": 16}, "last_distillation": "2026-07-12T05:00:28Z"}),
            encoding="utf-8",
        )

    def test_sends_message_for_rejected_outcome(self):
        self._write_corpus_stats()
        notify.send_run_notification(
            "rejected",
            adapter_id="adapter-1",
            run_id="run-1",
            mlflow_run_id="mlflow-1",
            report={
                "candidate": {"pass_rate": 0.1, "passed": 2, "total": 20},
                "baseline": {"pass_rate": 0.1, "passed": 2, "total": 20},
                "mandatory_pass": False,
                "no_category_regression": True,
                "approved": False,
            },
            project="ai-vllm",
        )
        self.smtp_instance.send_message.assert_called_once()
        sent_msg = self.smtp_instance.send_message.call_args[0][0]
        self.assertIn("REJECTED", sent_msg["Subject"])
        self.assertEqual(sent_msg["To"], "james_clare2@ketrenos.com")

    def test_no_op_when_notify_to_empty(self):
        with patch.object(notify, "NOTIFY_TO", ""):
            notify.send_run_notification("failed", run_id="run-1", error="boom")
        self.smtp_mock.assert_not_called()

    def test_increments_ok_metric_on_success(self):
        before = metrics.notification_sent.labels(outcome="promoted", status="ok")._value.get()
        notify.send_run_notification(
            "promoted",
            adapter_id="adapter-1",
            run_id="run-1",
            mlflow_run_id=None,
            report={
                "candidate": {"pass_rate": 1.0, "passed": 20, "total": 20},
                "baseline": {"pass_rate": 0.1, "passed": 2, "total": 20},
                "mandatory_pass": True,
                "no_category_regression": True,
                "approved": True,
            },
            project="ai-vllm",
        )
        after = metrics.notification_sent.labels(outcome="promoted", status="ok")._value.get()
        self.assertEqual(after, before + 1)

    def test_smtp_failure_is_caught_and_counted_not_raised(self):
        self.smtp_patch.stop()
        self.smtp_patch = patch.object(
            notify.smtplib, "SMTP", side_effect=smtplib.SMTPConnectError(421, "down")
        )
        self.smtp_patch.start()
        before = metrics.notification_sent.labels(outcome="failed", status="error")._value.get()
        notify.send_run_notification("failed", run_id="run-1", error="boom")
        after = metrics.notification_sent.labels(outcome="failed", status="error")._value.get()
        self.assertEqual(after, before + 1)

    def test_missing_corpus_stats_does_not_raise(self):
        notify.send_run_notification("skipped_no_new_content", run_id="run-1")
        self.smtp_instance.send_message.assert_called_once()

    def test_composes_postponement_notice_with_active_count(self):
        notify.send_run_notification("postponed", run_id="run-1", active_sessions=2)
        sent_msg = self.smtp_instance.send_message.call_args[0][0]
        self.assertIn("POSTPONED", sent_msg["Subject"])
        self.assertIn("2 inference request(s) are active", sent_msg.get_content())


if __name__ == "__main__":
    unittest.main()
