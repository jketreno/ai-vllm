"""Deterministic comparative evaluation for candidate promotion."""

from __future__ import annotations

import json
import logging
import pathlib
from collections import defaultdict
from typing import Any, Callable

from . import metrics

log = logging.getLogger(__name__)
PROBES_PATH = pathlib.Path("/app/prompts/eval_probes.jsonl")


def load_probes(path: pathlib.Path = PROBES_PATH) -> list[dict[str, Any]]:
    probes = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                probe = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed probe at line {line_number}") from exc
            if not {"id", "prompt"} <= probe.keys():
                raise ValueError(f"incomplete probe at line {line_number}")
            probes.append(probe)
    if not probes:
        raise ValueError("evaluation suite is empty")
    return probes


def compare(
    candidate_id: str,
    baseline_id: str,
    invoke: Callable[[str, dict[str, Any]], str],
    probes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    probes = probes or load_probes()
    candidate_results = [_score(probe, invoke(candidate_id, probe)) for probe in probes]
    baseline_results = [_score(probe, invoke(baseline_id, probe)) for probe in probes]
    candidate = _summary(candidate_results)
    baseline = _summary(baseline_results)
    categories = set(candidate["categories"]) | set(baseline["categories"])
    no_regression = all(
        candidate["categories"].get(category, 0) >= baseline["categories"].get(category, 0)
        for category in categories
    )
    mandatory_pass = all(item["passed"] for item in candidate_results if item["mandatory"])
    approved = mandatory_pass and candidate["pass_rate"] >= 0.90 and no_regression
    for category, score in candidate["categories"].items():
        metrics.evaluation_score.labels(adapter_id=candidate_id, category=category).set(score)
    return {
        "candidate_id": candidate_id,
        "baseline_id": baseline_id,
        "candidate": candidate,
        "baseline": baseline,
        "mandatory_pass": mandatory_pass,
        "no_category_regression": no_regression,
        "approved": approved,
        "results": candidate_results,
    }


def _score(probe: dict[str, Any], completion: str | None) -> dict[str, Any]:
    expected = probe.get("expected_keyword")
    content = completion or ""
    passed = True if not expected else expected.casefold() in content.casefold()
    return {
        "id": probe["id"],
        "category": probe.get("category", "general"),
        "mandatory": bool(probe.get("mandatory", True)),
        "passed": passed,
    }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        totals[result["category"]].append(result["passed"])
    passed = sum(item["passed"] for item in results)
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results),
        "categories": {
            category: sum(values) / len(values)
            for category, values in sorted(totals.items())
        },
    }
