"""Hierarchical summarizer: daily episodes → weekly → monthly → quarterly → themes."""

import json
import logging
import os
import pathlib
from datetime import datetime, timedelta, timezone

from . import metrics
from .local_llm import generate
from .structured_output import repair_prompt, parse_pattern_records_with_repair

log = logging.getLogger(__name__)

CORPUS_ROOT = pathlib.Path(os.environ.get("CORPUS_ROOT", "/corpus"))
WEEKLY_GATE = 2
MONTHLY_GATE = 3
QUARTERLY_GATE = 2

CATEGORIES = ("style", "architecture", "antipattern", "domain")


def _call_llm_merge(records: list[dict], level: str) -> list[dict]:
    """Ask the LLM to merge semantically similar patterns within the same category."""
    if not records:
        return []

    prompt = (
        f"You are compressing a {level} summary of AI session patterns for fine-tuning signal.\n"
        "Input: a JSON array of pattern records. Each has: category, pattern (description), "
        "evidence_count, canonical_example, first_seen, last_seen.\n"
        "Task:\n"
        "1. Group records with the same category AND similar semantic meaning.\n"
        "2. For each group, merge into one record: sum evidence_count, keep the canonical_example "
        "from the highest-evidence record, use the earliest first_seen and latest last_seen.\n"
        "3. Drop any merged record whose final evidence_count is below the threshold for this level "
        f"({WEEKLY_GATE if level == 'weekly' else MONTHLY_GATE if level == 'monthly' else QUARTERLY_GATE}).\n"
        "Return ONLY a JSON array of merged records. No commentary.\n\n"
        f"Input records:\n{json.dumps(records, indent=2)}"
    )

    text = generate(prompt, max_tokens=2048)

    result, outcome = parse_pattern_records_with_repair(
        text,
        lambda error: generate(repair_prompt(text, error), max_tokens=2048),
    )
    metrics.structured_output_attempts.labels(stage=f"summary_{level}", outcome=outcome).inc()
    if outcome == "failed":
        metrics.summary_parse_errors.labels(level=level).inc()
        log.error("Summarizer LLM output not valid JSON")
    return result


def _iso_week(date: datetime) -> str:
    """Return ISO week string like 2025-W42."""
    year, week, _ = date.isocalendar()
    return f"{year}-W{week:02d}"


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _record_summary_metrics(
    project: str,
    level: str,
    input_records: int,
    output_records: int,
    outcome: str,
) -> None:
    metrics.summary_records_input.labels(project=project, level=level).set(input_records)
    metrics.summary_records_output.labels(project=project, level=level).set(output_records)
    metrics.summary_runs.labels(project=project, level=level, outcome=outcome).inc()
    metrics.summary_last_run_timestamp.labels(project=project, level=level).set(
        datetime.now(tz=timezone.utc).timestamp()
    )


def _episode_projects() -> list[str]:
    episodes_dir = CORPUS_ROOT / "episodes"
    if not episodes_dir.exists():
        return []
    return sorted(p.name for p in episodes_dir.iterdir() if p.is_dir())


def _summary_projects(level: str) -> list[str]:
    summary_dir = CORPUS_ROOT / "summaries" / level
    if not summary_dir.exists():
        return []
    return sorted(p.name for p in summary_dir.iterdir() if p.is_dir())


def _run_weekly_project(project: str, reference_date: datetime) -> dict:
    day_records: list[dict] = []
    for offset in range(7):
        day = reference_date - timedelta(days=offset)
        ep_path = CORPUS_ROOT / "episodes" / project / day.strftime("%Y/%m/%d.jsonl")
        if ep_path.exists():
            day_records.extend(_load_jsonl(ep_path))

    if not day_records:
        log.info("No daily episodes found for project %s week ending %s", project, reference_date.date())
        _record_summary_metrics(project, "weekly", 0, 0, "no_input")
        return {"input_records": 0, "output_records": 0}

    merged = _call_llm_merge(day_records, "weekly")
    week_str = _iso_week(reference_date)
    _write_jsonl(CORPUS_ROOT / "summaries" / "weekly" / project / f"{week_str}.jsonl", merged)
    _record_summary_metrics(project, "weekly", len(day_records), len(merged), "success")

    log.info("Weekly summary %s for %s: %d → %d records", week_str, project, len(day_records), len(merged))
    return {"input_records": len(day_records), "output_records": len(merged)}


def run_weekly(reference_date: datetime | None = None) -> dict:
    """Compress the 7 daily episode files ending on reference_date into project summaries."""
    if reference_date is None:
        reference_date = datetime.now(tz=timezone.utc)

    totals = {"input_records": 0, "output_records": 0}
    for project in _episode_projects():
        result = _run_weekly_project(project, reference_date)
        for key in totals:
            totals[key] += result[key]
    if totals["input_records"] == 0:
        log.info("No project daily episodes found for week ending %s", reference_date.date())
    return totals


def _weekly_files_for_month(project: str, year: int, month: int) -> list[pathlib.Path]:
    weekly_dir = CORPUS_ROOT / "summaries" / "weekly" / project
    result = []
    for weekly_file in sorted(weekly_dir.glob(f"{year}-W*.jsonl")):
        try:
            _, week_num = weekly_file.stem.split("-W")
            week_start = datetime.strptime(f"{year}-W{int(week_num)}-1", "%Y-W%W-%w")
            if week_start.month == month:
                result.append(weekly_file)
        except (ValueError, AttributeError):
            continue
    return result


def _run_monthly_project(project: str, reference_date: datetime) -> dict:
    first_of_this_month = reference_date.replace(day=1)
    last_of_prev_month = first_of_this_month - timedelta(days=1)
    year = last_of_prev_month.year
    month = last_of_prev_month.month

    week_records: list[dict] = []
    for weekly_file in _weekly_files_for_month(project, year, month):
        week_records.extend(_load_jsonl(weekly_file))

    if not week_records:
        log.info("No weekly summaries found for project %s %d-%02d", project, year, month)
        _record_summary_metrics(project, "monthly", 0, 0, "no_input")
        return {"input_records": 0, "output_records": 0}

    merged = _call_llm_merge(week_records, "monthly")
    _write_jsonl(CORPUS_ROOT / "summaries" / "monthly" / project / f"{year}-{month:02d}.jsonl", merged)
    _record_summary_metrics(project, "monthly", len(week_records), len(merged), "success")

    log.info("Monthly summary %d-%02d for %s: %d → %d records", year, month, project, len(week_records), len(merged))
    return {"input_records": len(week_records), "output_records": len(merged)}


def run_monthly(reference_date: datetime | None = None) -> dict:
    """Compress weekly summaries for the prior month into project monthly summaries."""
    if reference_date is None:
        reference_date = datetime.now(tz=timezone.utc)

    totals = {"input_records": 0, "output_records": 0}
    for project in _summary_projects("weekly"):
        result = _run_monthly_project(project, reference_date)
        for key in totals:
            totals[key] += result[key]
    if totals["input_records"] == 0:
        log.info("No project weekly summaries found for monthly summarization")
    return totals


def _prev_quarter(reference_date: datetime) -> tuple[int, int]:
    current_quarter = (reference_date.month - 1) // 3 + 1
    year = reference_date.year
    if current_quarter == 1:
        return year - 1, 4
    return year, current_quarter - 1


def _run_quarterly_project(project: str, reference_date: datetime) -> dict:
    year, target_quarter = _prev_quarter(reference_date)
    monthly_dir = CORPUS_ROOT / "summaries" / "monthly" / project
    month_records: list[dict] = []
    for m in range((target_quarter - 1) * 3 + 1, (target_quarter - 1) * 3 + 4):
        mf = monthly_dir / f"{year}-{m:02d}.jsonl"
        if mf.exists():
            month_records.extend(_load_jsonl(mf))

    if not month_records:
        log.info("No monthly summaries found for project %s Q%d %d", project, target_quarter, year)
        _record_summary_metrics(project, "quarterly", 0, 0, "no_input")
        return {"input_records": 0, "output_records": 0, "themes_promoted": 0}

    merged = _call_llm_merge(month_records, "quarterly")
    _write_jsonl(CORPUS_ROOT / "summaries" / "quarterly" / project / f"{year}-Q{target_quarter}.jsonl", merged)
    themes_promoted = _promote_themes(project)
    _record_summary_metrics(project, "quarterly", len(month_records), len(merged), "success")

    log.info(
        "Quarterly summary Q%d %d for %s: %d → %d records, %d themes promoted",
        target_quarter, year, project, len(month_records), len(merged), themes_promoted,
    )
    return {"input_records": len(month_records), "output_records": len(merged), "themes_promoted": themes_promoted}


def run_quarterly(reference_date: datetime | None = None) -> dict:
    """Compress monthly summaries for the prior quarter and promote project themes."""
    if reference_date is None:
        reference_date = datetime.now(tz=timezone.utc)

    totals = {"input_records": 0, "output_records": 0, "themes_promoted": 0}
    for project in _summary_projects("monthly"):
        result = _run_quarterly_project(project, reference_date)
        for key in totals:
            totals[key] += result[key]
    if totals["input_records"] == 0:
        log.info("No project monthly summaries found for quarterly summarization")
    return totals


def _load_quarterly_records(project: str) -> list[dict]:
    quarterly_dir = CORPUS_ROOT / "summaries" / "quarterly" / project
    records: list[dict] = []
    for qf in sorted(quarterly_dir.glob("*.jsonl")):
        for r in _load_jsonl(qf):
            r["_quarter_file"] = qf.stem
            records.append(r)
    return records


def _archive_theme(project: str, theme_path: pathlib.Path, now_str: str) -> None:
    archive_dir = CORPUS_ROOT / "themes" / "archive" / project / now_str
    archive_dir.mkdir(parents=True, exist_ok=True)
    theme_path.rename(archive_dir / theme_path.name)


def _promote_category(project: str, cat: str, records: list[dict], now_str: str) -> int:
    to_promote = [r for r in records if r.get("evidence_count", 0) >= QUARTERLY_GATE]
    if not to_promote:
        return 0
    theme_path = CORPUS_ROOT / "themes" / "active" / project / f"{cat}.jsonl"
    theme_path.parent.mkdir(parents=True, exist_ok=True)
    if theme_path.exists():
        _archive_theme(project, theme_path, now_str)
    for r in to_promote:
        r.pop("_quarter_file", None)
    _write_jsonl(theme_path, to_promote)
    metrics.themes_active.labels(project=project, category=cat).set(len(to_promote))
    return len(to_promote)


def _promote_themes(project: str) -> int:
    """Scan project quarterly summaries and promote cross-quarter patterns to active themes."""
    all_quarterly = _load_quarterly_records(project)
    if not all_quarterly:
        return 0

    by_category: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for r in all_quarterly:
        cat = r.get("category", "unknown")
        if cat in by_category:
            by_category[cat].append(r)

    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return sum(_promote_category(project, cat, records, now_str) for cat, records in by_category.items())


def run_scheduled(reference_date: datetime | None = None) -> dict:
    """Run whichever summarization passes are due for the given date."""
    if reference_date is None:
        reference_date = datetime.now(tz=timezone.utc)

    results: dict = {}

    # Weekly: run every Sunday
    if reference_date.weekday() == 6:
        results["weekly"] = run_weekly(reference_date)

    # Monthly: run on the 1st of each month
    if reference_date.day == 1:
        results["monthly"] = run_monthly(reference_date)

    # Quarterly: run on the 1st day of each new quarter (Jan, Apr, Jul, Oct)
    if reference_date.day == 1 and reference_date.month in (1, 4, 7, 10):
        results["quarterly"] = run_quarterly(reference_date)

    return results
