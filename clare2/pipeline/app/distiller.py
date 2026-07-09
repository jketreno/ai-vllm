"""Daily distillation pass: raw sessions → episode store."""

import json
import logging
import os
import pathlib
from datetime import datetime, timezone

from . import metrics
from .local_llm import generate
from .structured_output import repair_prompt, parse_pattern_records_with_repair

log = logging.getLogger(__name__)

CORPUS_ROOT = pathlib.Path(os.environ.get("CORPUS_ROOT", "/corpus"))
RECURRENCE_GATE = 2  # minimum evidence_count within a session to pass gate


def _load_distill_prompt() -> str:
    prompt_path = pathlib.Path("/app/prompts/distill.txt")
    return prompt_path.read_text()


def _session_projects() -> list[str]:
    """Return sorted list of project names that have session data."""
    sessions_dir = CORPUS_ROOT / "sessions"
    if not sessions_dir.exists():
        return []
    return sorted(p.name for p in sessions_dir.iterdir() if p.is_dir())


def _session_files_for_date(date: datetime, project: str) -> list[pathlib.Path]:
    day_dir = CORPUS_ROOT / "sessions" / project / date.strftime("%Y/%m/%d")
    if not day_dir.exists():
        return []
    return sorted(day_dir.glob("*.jsonl"))


def _session_file_date(path: pathlib.Path) -> str:
    relative = path.relative_to(CORPUS_ROOT / "sessions")
    year, month, day = relative.parts[1:4]
    return f"{year}-{month}-{day}"


def _all_session_files(project: str) -> list[pathlib.Path]:
    project_dir = CORPUS_ROOT / "sessions" / project
    if not project_dir.exists():
        return []
    return sorted(project_dir.glob("*/*/*/*.jsonl"))


def _unprocessed_session_files(project: str, date: datetime | None = None) -> list[pathlib.Path]:
    session_files = (
        _session_files_for_date(date, project) if date is not None else _all_session_files(project)
    )
    processed_keys = _processed_session_keys(project)
    return [
        path
        for path in session_files
        if _session_key(project, _session_file_date(path), path.stem) not in processed_keys
    ]


def _read_session(path: pathlib.Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Skipping malformed JSONL line in %s", path)
    return records


def _call_distill_llm(session_text: str, prompt_template: str) -> str:
    """Call local Qwen3.5 with the session content."""
    prompt = prompt_template.replace("{{SESSION_CONTENT}}", session_text)
    return generate(prompt, max_tokens=1536)


def _parse_patterns(llm_output: str) -> list[dict]:
    """Extract JSON pattern list from LLM output."""
    patterns, outcome = parse_pattern_records_with_repair(
        llm_output,
        lambda error: generate(repair_prompt(llm_output, error), max_tokens=1536),
    )
    metrics.structured_output_attempts.labels(stage="distillation", outcome=outcome).inc()
    if outcome == "failed":
        metrics.distillation_parse_errors.inc()
        log.error("LLM distillation output was not valid JSON:\n%s", llm_output[:500])
    return patterns


def _patterns_from_session(
    project: str,
    session_path: pathlib.Path,
    prompt_template: str,
) -> tuple[str, list[dict]]:
    records = _read_session(session_path)
    if not records:
        metrics.distillation_sessions.labels(project=project, outcome="empty").inc()
        return "empty", []

    session_text = "\n".join(json.dumps(r) for r in records)
    try:
        llm_output = _call_distill_llm(session_text, prompt_template)
    except Exception:
        metrics.distillation_sessions.labels(project=project, outcome="llm_error").inc()
        log.exception("LLM call failed for session %s", session_path.name)
        return "llm_error", []

    metrics.distillation_sessions.labels(project=project, outcome="processed").inc()
    raw_patterns = _parse_patterns(llm_output)
    metrics.distillation_patterns_raw.labels(project=project).inc(len(raw_patterns))
    return "processed", raw_patterns


def _accepted_patterns(
    project: str,
    session_path: pathlib.Path,
    raw_patterns: list[dict],
) -> tuple[list[dict], int]:
    accepted: list[dict] = []
    gated_out = 0
    session_date = _session_file_date(session_path)
    for pattern in raw_patterns:
        evidence = pattern.get("evidence_count", 1)
        if evidence < RECURRENCE_GATE:
            gated_out += 1
            metrics.distillation_patterns_gated_out.inc()
            continue

        pattern["session_id"] = session_path.stem
        pattern["project"] = project
        pattern["session_date"] = session_date
        pattern["distilled_at"] = datetime.now(tz=timezone.utc).isoformat()
        accepted.append(pattern)

        category = pattern.get("category", "unknown")
        metrics.distillation_patterns_extracted.labels(project=project, category=category).inc()
    return accepted, gated_out


def _write_episode_patterns(project: str, patterns: list[dict]) -> None:
    patterns_by_date: dict[str, list[dict]] = {}
    for pattern in patterns:
        patterns_by_date.setdefault(pattern["session_date"], []).append(pattern)

    for session_date, dated_patterns in sorted(patterns_by_date.items()):
        year, month, day = session_date.split("-")
        episode_path = CORPUS_ROOT / "episodes" / project / year / month / f"{day}.jsonl"
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        with open(episode_path, "a", encoding="utf-8") as fh:
            for pattern in dated_patterns:
                fh.write(json.dumps(pattern) + "\n")


def _distill_project(project: str, date: datetime | None, prompt_template: str) -> dict:
    """Distill unprocessed sessions for one project."""
    session_files = _unprocessed_session_files(project, date)
    metrics.distillation_sessions_pending.labels(project=project).set(len(session_files))
    if not session_files:
        _record_last_distillation_metrics(project, 0, 0, 0)
        scope = date.date() if date is not None else "any date"
        log.info("No new sessions for project %s on %s", project, scope)
        return {"sessions": 0, "patterns_extracted": 0, "patterns_gated": 0}

    all_patterns: list[dict] = []
    gated_out = 0
    processed_files: list[pathlib.Path] = []

    for session_path in session_files:
        outcome, raw_patterns = _patterns_from_session(project, session_path, prompt_template)
        if outcome == "llm_error":
            continue
        processed_files.append(session_path)
        accepted, session_gated_out = _accepted_patterns(project, session_path, raw_patterns)
        all_patterns.extend(accepted)
        gated_out += session_gated_out

    if not all_patterns:
        _update_session_index(processed_files, project)
        _record_last_distillation_metrics(project, len(session_files), 0, gated_out)
        scope = date.date() if date is not None else "unprocessed sessions"
        log.info("No patterns passed the recurrence gate for project %s on %s", project, scope)
        return {"sessions": len(session_files), "patterns_extracted": 0, "patterns_gated": gated_out}

    _write_episode_patterns(project, all_patterns)

    log.info(
        "Distillation complete for %s: %d sessions, %d patterns saved, %d gated out",
        project,
        len(session_files),
        len(all_patterns),
        gated_out,
    )

    _update_session_index(processed_files, project)
    _update_corpus_stats(all_patterns)
    _record_last_distillation_metrics(project, len(session_files), len(all_patterns), gated_out)

    return {
        "sessions": len(session_files),
        "patterns_extracted": len(all_patterns),
        "patterns_gated": gated_out,
    }


def run_daily(date: datetime | None = None) -> dict:
    """Distill unprocessed sessions for all projects.

    When date is provided, only that date is processed. The scheduled path
    intentionally omits date so missed syncs, restarts, and local-time offsets
    are repaired by catching up every unprocessed session.
    """
    projects = _session_projects()
    if not projects:
        log.info("No project session directories found — skipping distillation")
        metrics.distillation_runs.labels(outcome="no_projects").inc()
        metrics.distillation_last_run_timestamp.set(datetime.now(tz=timezone.utc).timestamp())
        return {"sessions": 0, "patterns_extracted": 0, "patterns_gated": 0}

    prompt_template = _load_distill_prompt()
    totals = {"sessions": 0, "patterns_extracted": 0, "patterns_gated": 0}

    for project in projects:
        result = _distill_project(project, date, prompt_template)
        for key in totals:
            totals[key] += result[key]

    outcome = "patterns_extracted" if totals["patterns_extracted"] else "no_patterns"
    metrics.distillation_runs.labels(outcome=outcome).inc()
    metrics.distillation_last_run_timestamp.set(datetime.now(tz=timezone.utc).timestamp())
    return totals


def _session_key(project: str, date: str, session_id: str) -> tuple[str, str, str]:
    return (project, date, session_id)


def _processed_session_keys(project: str = "") -> set[tuple[str, str, str]]:
    index_path = CORPUS_ROOT / "meta" / "session_index.json"
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {"sessions": []}
    except json.JSONDecodeError:
        return set()
    sessions = index.get("sessions", [])
    if project:
        sessions = [s for s in sessions if s.get("project") == project]
    return {
        _session_key(record.get("project", ""), record.get("date", ""), record.get("session_id", ""))
        for record in sessions
        if record.get("project") and record.get("date") and record.get("session_id")
    }


def _update_session_index(session_files: list[pathlib.Path], project: str) -> None:
    index_path = CORPUS_ROOT / "meta" / "session_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {"sessions": []}
    except json.JSONDecodeError:
        index = {"sessions": []}

    existing_keys = {
        _session_key(s.get("project", ""), s.get("date", ""), s.get("session_id", ""))
        for s in index["sessions"]
    }
    for f in session_files:
        session_date = _session_file_date(f)
        key = _session_key(project, session_date, f.stem)
        if key not in existing_keys:
            index["sessions"].append({
                "session_id": f.stem,
                "project": project,
                "date": session_date,
                "path": str(f),
            })
            existing_keys.add(key)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _update_corpus_stats(patterns: list[dict]) -> None:
    stats_path = CORPUS_ROOT / "meta" / "corpus_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    except json.JSONDecodeError:
        stats = {}

    episodes = stats.get("episodes", {"style": 0, "architecture": 0, "antipattern": 0, "domain": 0})
    for p in patterns:
        cat = p.get("category", "unknown")
        episodes[cat] = episodes.get(cat, 0) + 1
    stats["episodes"] = episodes
    stats["last_distillation"] = datetime.now(tz=timezone.utc).isoformat()
    stats_path.write_text(json.dumps(stats, indent=2))


def _record_last_distillation_metrics(
    project: str,
    sessions: int,
    patterns_extracted: int,
    patterns_gated: int,
) -> None:
    metrics.distillation_sessions_last.labels(project=project).set(sessions)
    metrics.distillation_patterns_extracted_last.labels(project=project).set(patterns_extracted)
    metrics.distillation_patterns_gated_out_last.labels(project=project).set(patterns_gated)
