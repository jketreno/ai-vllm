from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import app.distiller as distiller
import app.metrics as metrics


class DistillerCatchUpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_session(self, project: str, date: str, session_id: str) -> None:
        year, month, day = date.split("-")
        path = self.root / "sessions" / project / year / month / day / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"role": "user", "content": f"{project}:{date}"}) + "\n")

    def _patterns(self, output: str) -> list[dict]:
        return [
            {
                "category": "domain",
                "pattern": f"Pattern from {output}",
                "evidence_count": 2,
                "canonical_example": "example",
            }
        ]

    def test_run_daily_catches_up_all_unprocessed_dates(self):
        self._write_session("ai-vllm", "2026-07-01", "same-session")
        self._write_session("ai-vllm", "2026-07-02", "same-session")

        with patch.object(distiller, "CORPUS_ROOT", self.root), patch.object(
            distiller, "_load_distill_prompt", return_value="{{SESSION_CONTENT}}"
        ), patch.object(distiller, "_call_distill_llm", side_effect=["one", "two"]), patch.object(
            distiller, "_parse_patterns", side_effect=self._patterns
        ):
            result = distiller.run_daily()

        self.assertEqual(result["sessions"], 2)
        self.assertTrue((self.root / "episodes/ai-vllm/2026/07/01.jsonl").exists())
        self.assertTrue((self.root / "episodes/ai-vllm/2026/07/02.jsonl").exists())
        index = json.loads((self.root / "meta/session_index.json").read_text())
        self.assertEqual(len(index["sessions"]), 2)
        self.assertEqual({record["date"] for record in index["sessions"]}, {"2026-07-01", "2026-07-02"})

    def test_explicit_date_limits_processing_to_that_date(self):
        self._write_session("ai-vllm", "2026-07-01", "old")
        self._write_session("ai-vllm", "2026-07-02", "target")

        with patch.object(distiller, "CORPUS_ROOT", self.root), patch.object(
            distiller, "_load_distill_prompt", return_value="{{SESSION_CONTENT}}"
        ), patch.object(distiller, "_call_distill_llm", return_value="ok"), patch.object(
            distiller, "_parse_patterns", side_effect=self._patterns
        ):
            result = distiller.run_daily(datetime(2026, 7, 2, tzinfo=timezone.utc))

        self.assertEqual(result["sessions"], 1)
        self.assertFalse((self.root / "episodes/ai-vllm/2026/07/01.jsonl").exists())
        self.assertTrue((self.root / "episodes/ai-vllm/2026/07/02.jsonl").exists())

    def test_corpus_stats_are_tracked_per_project(self):
        self._write_session("ai-vllm", "2026-07-01", "session-a")
        self._write_session("clare", "2026-07-01", "session-b")

        with patch.object(distiller, "CORPUS_ROOT", self.root), patch.object(
            distiller, "_load_distill_prompt", return_value="{{SESSION_CONTENT}}"
        ), patch.object(distiller, "_call_distill_llm", side_effect=["one", "two"]), patch.object(
            distiller, "_parse_patterns", side_effect=self._patterns
        ):
            distiller.run_daily()

        stats = json.loads((self.root / "meta/corpus_stats.json").read_text())
        self.assertEqual(set(stats["projects"]), {"ai-vllm", "clare"})
        self.assertEqual(stats["projects"]["ai-vllm"]["episodes"]["domain"], 1)
        self.assertEqual(stats["projects"]["clare"]["episodes"]["domain"], 1)
        self.assertNotEqual(
            stats["projects"]["ai-vllm"]["last_distillation"],
            None,
        )

    def test_corpus_stats_for_one_project_do_not_clobber_another(self):
        self._write_session("ai-vllm", "2026-07-01", "session-a")

        with patch.object(distiller, "CORPUS_ROOT", self.root), patch.object(
            distiller, "_load_distill_prompt", return_value="{{SESSION_CONTENT}}"
        ), patch.object(distiller, "_call_distill_llm", return_value="one"), patch.object(
            distiller, "_parse_patterns", side_effect=self._patterns
        ):
            distiller.run_daily()

        self._write_session("clare", "2026-07-02", "session-b")
        with patch.object(distiller, "CORPUS_ROOT", self.root), patch.object(
            distiller, "_load_distill_prompt", return_value="{{SESSION_CONTENT}}"
        ), patch.object(distiller, "_call_distill_llm", return_value="two"), patch.object(
            distiller, "_parse_patterns", side_effect=self._patterns
        ):
            distiller.run_daily()

        stats = json.loads((self.root / "meta/corpus_stats.json").read_text())
        self.assertEqual(set(stats["projects"]), {"ai-vllm", "clare"})
        self.assertEqual(stats["projects"]["ai-vllm"]["episodes"]["domain"], 1)
        self.assertEqual(stats["projects"]["clare"]["episodes"]["domain"], 1)

    def test_records_last_run_metrics_when_no_new_sessions(self):
        self._write_session("ai-vllm", "2026-07-01", "processed")
        index_path = self.root / "meta/session_index.json"
        index_path.parent.mkdir(parents=True)
        index_path.write_text(
            json.dumps({
                "sessions": [{
                    "session_id": "processed",
                    "project": "ai-vllm",
                    "date": "2026-07-01",
                    "path": "sessions/ai-vllm/2026/07/01/processed.jsonl",
                }]
            }),
            encoding="utf-8",
        )

        with patch.object(distiller, "CORPUS_ROOT", self.root), patch.object(
            distiller, "_load_distill_prompt", return_value="{{SESSION_CONTENT}}"
        ):
            result = distiller.run_daily()

        self.assertEqual(result["sessions"], 0)
        self.assertEqual(metrics.distillation_sessions_last.labels(project="ai-vllm")._value.get(), 0)
        self.assertEqual(metrics.distillation_patterns_extracted_last.labels(project="ai-vllm")._value.get(), 0)
        self.assertEqual(metrics.distillation_patterns_gated_out_last.labels(project="ai-vllm")._value.get(), 0)


if __name__ == "__main__":
    unittest.main()
