"""Hierarchical summarizer: daily episodes → weekly → monthly → quarterly → themes."""

import json
import logging
import os
import pathlib
from datetime import datetime, timedelta, timezone

import anthropic
import openai

from . import metrics
from .security import secret_value

log = logging.getLogger(__name__)

CORPUS_ROOT = pathlib.Path(os.environ.get("CORPUS_ROOT", "/corpus"))
DISTILL_MODEL = os.environ.get("CLARE2_DISTILL_MODEL", "claude-haiku-4-5")

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

    if DISTILL_MODEL.startswith("claude"):
        client = anthropic.Anthropic(api_key=secret_value("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=DISTILL_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
    else:
        local_url = os.environ.get("CLARE2_LOCAL_LLM_URL", "http://clare2-policy:8000/v1")
        client = openai.OpenAI(
            base_url=local_url,
            api_key=secret_value("CLARE2_PROXY_TOKEN"),
        )
        resp = client.chat.completions.create(
            model=DISTILL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        text = resp.choices[0].message.content

    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "patterns" in result:
            result = result["patterns"]
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        log.error("Summarizer LLM output not valid JSON")
        return []


def _iso_week(date: datetime) -> str:
    """Return ISO week string like 2025-W42."""
    year, week, _ = date.isocalendar()
    return f"{year}-W{week:02d}"


def run_weekly(reference_date: datetime | None = None) -> dict:
    """Compress the 7 daily episode files ending on reference_date (Sunday) into a weekly summary."""
    if reference_date is None:
        reference_date = datetime.now(tz=timezone.utc)

    # Collect the 7 days ending on (and including) reference_date
    day_records: list[dict] = []
    for offset in range(7):
        day = reference_date - timedelta(days=offset)
        ep_path = CORPUS_ROOT / "episodes" / day.strftime("%Y/%m/%d.jsonl")
        if ep_path.exists():
            with open(ep_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            day_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    if not day_records:
        log.info("No daily episodes found for week ending %s", reference_date.date())
        return {"input_records": 0, "output_records": 0}

    merged = _call_llm_merge(day_records, "weekly")

    week_str = _iso_week(reference_date)
    out_path = CORPUS_ROOT / "summaries" / "weekly" / f"{week_str}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")

    log.info("Weekly summary %s: %d → %d records", week_str, len(day_records), len(merged))
    return {"input_records": len(day_records), "output_records": len(merged)}


def run_monthly(reference_date: datetime | None = None) -> dict:
    """Compress 4–5 weekly summaries for the prior month into a monthly summary."""
    if reference_date is None:
        reference_date = datetime.now(tz=timezone.utc)

    # Target the previous month
    first_of_this_month = reference_date.replace(day=1)
    last_of_prev_month = first_of_this_month - timedelta(days=1)
    year = last_of_prev_month.year
    month = last_of_prev_month.month

    weekly_dir = CORPUS_ROOT / "summaries" / "weekly"
    week_records: list[dict] = []
    for weekly_file in sorted(weekly_dir.glob(f"{year}-W*.jsonl")):
        # Parse the week date to check if it falls in target month
        week_label = weekly_file.stem  # e.g. 2025-W42
        try:
            _, week_num = week_label.split("-W")
            week_start = datetime.strptime(f"{year}-W{int(week_num)}-1", "%Y-W%W-%w")
            if week_start.month == month:
                with open(weekly_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                week_records.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except (ValueError, AttributeError):
            continue

    if not week_records:
        log.info("No weekly summaries found for %d-%02d", year, month)
        return {"input_records": 0, "output_records": 0}

    merged = _call_llm_merge(week_records, "monthly")

    out_path = CORPUS_ROOT / "summaries" / "monthly" / f"{year}-{month:02d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")

    log.info("Monthly summary %d-%02d: %d → %d records", year, month, len(week_records), len(merged))
    return {"input_records": len(week_records), "output_records": len(merged)}


def run_quarterly(reference_date: datetime | None = None) -> dict:
    """Compress monthly summaries for the prior quarter + promote themes."""
    if reference_date is None:
        reference_date = datetime.now(tz=timezone.utc)

    # Determine the previous quarter
    current_quarter = (reference_date.month - 1) // 3 + 1
    year = reference_date.year
    if current_quarter == 1:
        target_quarter = 4
        year -= 1
    else:
        target_quarter = current_quarter - 1

    quarter_months = range((target_quarter - 1) * 3 + 1, (target_quarter - 1) * 3 + 4)
    monthly_dir = CORPUS_ROOT / "summaries" / "monthly"
    month_records: list[dict] = []
    for m in quarter_months:
        mf = monthly_dir / f"{year}-{m:02d}.jsonl"
        if mf.exists():
            with open(mf) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            month_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    if not month_records:
        log.info("No monthly summaries found for Q%d %d", target_quarter, year)
        return {"input_records": 0, "output_records": 0, "themes_promoted": 0}

    merged = _call_llm_merge(month_records, "quarterly")

    out_path = CORPUS_ROOT / "summaries" / "quarterly" / f"{year}-Q{target_quarter}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")

    # Theme promotion: scan all quarterly summaries, promote patterns seen in 2+ quarters
    themes_promoted = _promote_themes()

    log.info(
        "Quarterly summary Q%d %d: %d → %d records, %d themes promoted",
        target_quarter, year, len(month_records), len(merged), themes_promoted,
    )
    return {"input_records": len(month_records), "output_records": len(merged), "themes_promoted": themes_promoted}


def _promote_themes() -> int:
    """Scan all quarterly summaries and promote cross-quarter patterns to active themes."""
    quarterly_dir = CORPUS_ROOT / "summaries" / "quarterly"
    all_quarterly: list[dict] = []
    for qf in sorted(quarterly_dir.glob("*.jsonl")):
        with open(qf) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        r["_quarter_file"] = qf.stem
                        all_quarterly.append(r)
                    except json.JSONDecodeError:
                        pass

    if not all_quarterly:
        return 0

    # Group by category, collect patterns with evidence across multiple quarters
    by_category: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for r in all_quarterly:
        cat = r.get("category", "unknown")
        if cat in by_category:
            by_category[cat].append(r)

    promoted = 0
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    for cat, records in by_category.items():
        # Simple promotion: evidence_count >= quarterly gate
        to_promote = [r for r in records if r.get("evidence_count", 0) >= QUARTERLY_GATE]
        if not to_promote:
            continue

        theme_path = CORPUS_ROOT / "themes" / "active" / f"{cat}.jsonl"
        theme_path.parent.mkdir(parents=True, exist_ok=True)

        # Archive existing theme file before overwriting
        if theme_path.exists():
            archive_dir = CORPUS_ROOT / "themes" / "archive" / now_str
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"{cat}.jsonl"
            theme_path.rename(archive_path)

        with open(theme_path, "w") as fh:
            for r in to_promote:
                r.pop("_quarter_file", None)
                fh.write(json.dumps(r) + "\n")

        metrics.themes_active.labels(category=cat).set(len(to_promote))
        promoted += len(to_promote)

    return promoted


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
