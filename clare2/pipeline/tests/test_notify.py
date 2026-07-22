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


def _text_body(msg):
    return msg.get_body(preferencelist=("plain",)).get_content()


def _html_body(msg):
    part = msg.get_body(preferencelist=("html",))
    return part.get_content() if part else None


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
        self.smtp_patch = patch.object(
            notify.smtplib, "SMTP", return_value=self.smtp_cm
        )
        self.smtp_mock = self.smtp_patch.start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def _write_corpus_stats(self) -> None:
        stats_path = self.corpus_root / "meta" / "corpus_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(
                {
                    "projects": {
                        "ai-vllm": {
                            "episodes": {"domain": 16},
                            "last_distillation": "2026-07-12T05:00:28Z",
                        }
                    }
                }
            ),
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
        before = metrics.notification_sent.labels(
            outcome="promoted", status="ok"
        )._value.get()
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
        after = metrics.notification_sent.labels(
            outcome="promoted", status="ok"
        )._value.get()
        self.assertEqual(after, before + 1)

    def test_smtp_failure_is_caught_and_counted_not_raised(self):
        self.smtp_patch.stop()
        self.smtp_patch = patch.object(
            notify.smtplib, "SMTP", side_effect=smtplib.SMTPConnectError(421, "down")
        )
        self.smtp_patch.start()
        before = metrics.notification_sent.labels(
            outcome="failed", status="error"
        )._value.get()
        notify.send_run_notification("failed", run_id="run-1", error="boom")
        after = metrics.notification_sent.labels(
            outcome="failed", status="error"
        )._value.get()
        self.assertEqual(after, before + 1)

    def test_missing_corpus_stats_does_not_raise(self):
        notify.send_run_notification("skipped_no_new_content", run_id="run-1")
        self.smtp_instance.send_message.assert_called_once()

    def test_composes_postponement_notice_with_active_count(self):
        notify.send_run_notification("postponed", run_id="run-1", active_sessions=2)
        sent_msg = self.smtp_instance.send_message.call_args[0][0]
        self.assertIn("POSTPONED", sent_msg["Subject"])
        self.assertIn("2 inference request(s) are active", _text_body(sent_msg))

    def test_distillation_lines_report_every_project(self):
        stats_path = self.corpus_root / "meta" / "corpus_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(
                {
                    "projects": {
                        "ai-vllm": {
                            "episodes": {"domain": 25},
                            "last_distillation": "2026-07-16T05:02:22Z",
                        },
                        "clare": {
                            "episodes": {"style": 3},
                            "last_distillation": "2026-07-16T05:03:10Z",
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        notify.send_run_notification("skipped_no_new_content", run_id="run-1")
        body = _text_body(self.smtp_instance.send_message.call_args[0][0])
        self.assertIn("ai-vllm:", body)
        self.assertIn("clare:", body)
        self.assertIn("domain: 25", body)
        self.assertIn("style: 3", body)

    def test_skipped_run_reports_a_reason_for_each_known_project(self):
        session_path = self.corpus_root / "sessions/backstory/2026/07/06/session.jsonl"
        session_path.parent.mkdir(parents=True)
        session_path.write_text("{}\n")
        notify.send_run_notification("skipped_no_new_content", run_id="run-1")
        body = _text_body(self.smtp_instance.send_message.call_args[0][0])
        self.assertIn(
            "backstory: 1 captured session(s) remain pending distillation", body
        )

    def test_sends_multipart_message_with_html_alternative(self):
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
        sent_msg = self.smtp_instance.send_message.call_args[0][0]
        self.assertTrue(sent_msg.is_multipart())
        html = _html_body(sent_msg)
        self.assertIsNotNone(html)
        self.assertIn("<table", html)
        self.assertIn("ai-vllm", html)
        self.assertIn("adapter-1", html)
        self.assertIn("REJECTED", html)


class SendBatchRunNotificationTests(unittest.TestCase):
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
        self.smtp_patch = patch.object(
            notify.smtplib, "SMTP", return_value=self.smtp_cm
        )
        self.smtp_mock = self.smtp_patch.start()

    def tearDown(self):
        self.temp.cleanup()
        patch.stopall()

    def _results(self):
        return [
            {
                "adapter_id": "clare-ai-vllm-1",
                "project": "ai-vllm",
                "mlflow_run_id": "mlflow-1",
                "outcome": "rejected",
                "report": {
                    "candidate": {"pass_rate": 0.1, "passed": 2, "total": 20},
                    "baseline": {"pass_rate": 0.1, "passed": 2, "total": 20},
                    "mandatory_pass": False,
                    "no_category_regression": True,
                    "approved": False,
                },
            },
            {
                "adapter_id": "clare-clare-1",
                "project": "clare",
                "mlflow_run_id": "mlflow-2",
                "outcome": "promoted",
                "report": {
                    "candidate": {"pass_rate": 1.0, "passed": 20, "total": 20},
                    "baseline": {"pass_rate": 0.5, "passed": 10, "total": 20},
                    "mandatory_pass": True,
                    "no_category_regression": True,
                    "approved": True,
                },
            },
        ]

    def _write_project_inventory(self):
        stats_path = self.corpus_root / "meta" / "corpus_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(
                {
                    "projects": {
                        "ai-vllm": {
                            "episodes": {"domain": 8},
                            "last_distillation": "2026-07-18T05:04:40Z",
                        }
                    }
                }
            )
        )
        index_path = self.corpus_root / "meta" / "session_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "sessions": [
                        {
                            "project": "backstory",
                            "date": "2026-07-06",
                            "session_id": "session-1",
                        }
                    ]
                }
            )
        )
        session_path = (
            self.corpus_root / "sessions/backstory/2026/07/06/session-1.jsonl"
        )
        session_path.parent.mkdir(parents=True)
        session_path.write_text("{}\n")
        ze_session = self.corpus_root / "sessions/ze-monitor/2026/07/11/session-2.jsonl"
        ze_session.parent.mkdir(parents=True)
        ze_session.write_text("{}\n")
        manifest_path = self.corpus_root / "training/ai-vllm/manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "last_updated": "2026-07-18T05:05:00Z",
                    "total_sft_pairs": 15,
                    "total_tokens": 4200,
                }
            )
        )

    def test_sends_one_email_covering_every_project(self):
        self._write_project_inventory()
        notify.send_batch_run_notification(run_id="run-1", results=self._results())
        self.smtp_instance.send_message.assert_called_once()
        sent_msg = self.smtp_instance.send_message.call_args[0][0]
        body = _text_body(sent_msg)
        self.assertIn("ai-vllm", body)
        self.assertIn("clare", body)
        self.assertIn("REJECTED", body)
        self.assertIn("PROMOTED", body)

    def test_reports_projects_with_sessions_but_no_corpus_or_training(self):
        self._write_project_inventory()
        notify.send_batch_run_notification(run_id="run-1", results=self._results())
        body = _text_body(self.smtp_instance.send_message.call_args[0][0])

        self.assertIn("backstory:", body)
        self.assertIn("1 captured, 1 processed, 0 pending; latest: 2026-07-06", body)
        self.assertIn(
            "current corpus: 0 SFT pair(s), ~0 tokens; last updated: never", body
        )
        self.assertIn("TRAINING / EVALUATION — backstory (NOT TRAINED)", body)
        self.assertIn("no accepted distilled patterns produced an SFT corpus", body)

        self.assertIn("ze-monitor:", body)
        self.assertIn("1 captured, 0 processed, 1 pending; latest: 2026-07-11", body)
        self.assertIn("1 captured session(s) remain pending distillation", body)

    def test_reports_latest_corpus_information(self):
        self._write_project_inventory()
        notify.send_batch_run_notification(run_id="run-1", results=self._results())
        body = _text_body(self.smtp_instance.send_message.call_args[0][0])
        self.assertIn(
            "current corpus: 15 SFT pair(s), ~4200 tokens; "
            "last updated: 2026-07-18T05:05:00Z",
            body,
        )

    def test_distillation_blocks_are_separated_by_blank_lines(self):
        self._write_project_inventory()
        notify.send_batch_run_notification(run_id="run-1", results=self._results())
        body = _text_body(self.smtp_instance.send_message.call_args[0][0])
        lines = body.split("\n")
        project_header_indices = [
            i
            for i, line in enumerate(lines)
            if line.strip().endswith(":") and line.startswith("  ")
        ]
        self.assertGreater(len(project_header_indices), 1)
        for idx in project_header_indices[1:]:
            self.assertEqual(lines[idx - 1], "")

    def test_sends_multipart_message_with_html_table(self):
        self._write_project_inventory()
        notify.send_batch_run_notification(run_id="run-1", results=self._results())
        sent_msg = self.smtp_instance.send_message.call_args[0][0]
        self.assertTrue(sent_msg.is_multipart())
        html = _html_body(sent_msg)
        self.assertIsNotNone(html)
        self.assertIn("<table", html)
        self.assertIn("ai-vllm", html)
        self.assertIn("clare", html)
        self.assertIn("backstory", html)
        self.assertIn("REJECTED", html)
        self.assertIn("PROMOTED", html)
        self.assertIn("NOT TRAINED", html)
        self.assertIn("mlflow-1", html)
        self.assertIn("run-1", html)

    def test_no_op_when_notify_to_empty(self):
        with patch.object(notify, "NOTIFY_TO", ""):
            notify.send_batch_run_notification(run_id="run-1", results=self._results())
        self.smtp_mock.assert_not_called()

    def test_increments_ok_metric_on_success(self):
        before = metrics.notification_sent.labels(
            outcome="batch_complete", status="ok"
        )._value.get()
        notify.send_batch_run_notification(run_id="run-1", results=self._results())
        after = metrics.notification_sent.labels(
            outcome="batch_complete", status="ok"
        )._value.get()
        self.assertEqual(after, before + 1)

    def test_empty_results_does_not_raise(self):
        notify.send_batch_run_notification(run_id="run-1", results=[])
        self.smtp_instance.send_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
