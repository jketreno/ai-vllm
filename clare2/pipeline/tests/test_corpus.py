from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import app.corpus as corpus
import app.metrics as metrics


class AssembleProjectMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_theme(self, project: str, records: list[dict]) -> None:
        path = self.root / "themes" / "active" / project / "domain.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def test_each_project_gets_its_own_labeled_metric_value(self):
        self._write_theme(
            "ai-vllm",
            [{"category": "domain", "pattern": "one pattern for ai-vllm"}],
        )
        self._write_theme(
            "clare",
            [
                {"category": "domain", "pattern": "first clare pattern"},
                {"category": "domain", "pattern": "second clare pattern"},
            ],
        )
        with patch.object(corpus, "CORPUS_ROOT", self.root):
            corpus.assemble()

        ai_vllm_pairs = metrics.corpus_sft_pairs.labels(project="ai-vllm")._value.get()
        clare_pairs = metrics.corpus_sft_pairs.labels(project="clare")._value.get()
        self.assertEqual(ai_vllm_pairs, 1)
        self.assertEqual(clare_pairs, 2)

    def test_project_with_no_patterns_reports_zero_not_the_other_projects_total(self):
        self._write_theme("ai-vllm", [{"category": "domain", "pattern": "has content"}])
        self._write_theme("empty-project", [])
        with patch.object(corpus, "CORPUS_ROOT", self.root):
            corpus.assemble()

        empty_pairs = metrics.corpus_sft_pairs.labels(project="empty-project")._value.get()
        self.assertEqual(empty_pairs, 0)


if __name__ == "__main__":
    unittest.main()
