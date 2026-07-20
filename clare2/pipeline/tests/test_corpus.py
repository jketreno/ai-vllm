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

    def _write_episode(self, project: str, date: str, records: list[dict]) -> None:
        year, month, day = date.split("-")
        path = self.root / "episodes" / project / year / month / f"{day}.jsonl"
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

    def test_sft_pair_contains_actionable_context_and_source_metadata(self):
        pair = corpus._pattern_to_sft_pair({
            "project": "ai-vllm",
            "category": "antipattern",
            "pattern": "Do not bypass CLARE verification.",
            "canonical_example": "Run ./clare/verify-ci.sh before completion.",
            "evidence_count": 3,
            "session_id": "codex-session",
            "session_date": "2026-07-09",
            "_source_type": "episode",
        })

        self.assertIsNotNone(pair)
        self.assertIn("Project: ai-vllm", pair["prompt"])
        self.assertIn("Anti-pattern", pair["prompt"])
        self.assertIn("Do not bypass CLARE verification.", pair["completion"])
        self.assertIn("Run ./clare/verify-ci.sh before completion.", pair["completion"])
        self.assertEqual(pair["source_session"], "codex-session")
        self.assertEqual(pair["source_date"], "2026-07-09")
        self.assertEqual(pair["evidence_count"], 3)
        self.assertEqual(pair["weight"], 1.5)

    def test_sft_completions_vary_across_patterns_in_the_same_category(self):
        completions = {
            corpus._pattern_to_sft_pair({
                "project": "ai-vllm",
                "category": "domain",
                "pattern": f"Distinct domain rule number {i}.",
                "session_id": f"session-{i}",
            })["completion"]
            for i in range(6)
        }

        self.assertGreater(len(completions), 1)

    def test_backfilled_episode_is_used_when_it_is_project_latest(self):
        self._write_episode(
            "clare",
            "2026-07-01",
            [{"project": "clare", "category": "domain", "pattern": "Keep installer copies in sync."}],
        )

        with patch.object(corpus, "CORPUS_ROOT", self.root):
            result = corpus.assemble()

        self.assertEqual(result["sft_pairs"], 1)
        self.assertEqual((self.root / "training" / "clare" / "current.jsonl").read_text().count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
