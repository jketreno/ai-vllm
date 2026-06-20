"""Daily distillation pass: raw sessions → episode store."""

import json
import logging
import os
import pathlib
from datetime import datetime, timezone

from . import metrics
from .local_llm import generate

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
    return generate(prompt)


def _parse_patterns(llm_output: str) -> list[dict]:
    """Extract JSON pattern list from LLM output."""
    # The LLM is prompted to return a JSON array; strip any markdown fencing
    text = llm_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        patterns = json.loads(text)
        if isinstance(patterns, dict) and "patterns" in patterns:
            patterns = patterns["patterns"]
        return patterns if isinstance(patterns, list) else []
    except json.JSONDecodeError:
        log.error("LLM distillation output was not valid JSON:\n%s", llm_output[:500])
        return []


def _distill_project(project: str, date: datetime, prompt_template: str) -> dict:
    """Distill all sessions for one project on the given date."""
    session_files = _session_files_for_date(date, project)
    processed_ids = _processed_session_ids(project)
    session_files = [p for p in session_files if p.stem not in processed_ids]
    if not session_files:
        log.info("No new sessions for project %s on %s", project, date.date())
        return {"sessions": 0, "patterns_extracted": 0, "patterns_gated": 0}

    all_patterns: list[dict] = []
    gated_out = 0

    for session_path in session_files:
        records = _read_session(session_path)
        if not records:
            continue

        session_text = "\n".join(json.dumps(r) for r in records)
        try:
            llm_output = _call_distill_llm(session_text, prompt_template)
        except Exception:
            log.exception("LLM call failed for session %s", session_path.name)
            continue

        raw_patterns = _parse_patterns(llm_output)

        for p in raw_patterns:
            evidence = p.get("evidence_count", 1)
            if evidence < RECURRENCE_GATE:
                gated_out += 1
                metrics.distillation_patterns_gated_out.inc()
                continue

            p["session_id"] = session_path.stem
            p["project"] = project
            p["distilled_at"] = datetime.now(tz=timezone.utc).isoformat()
            all_patterns.append(p)

            cat = p.get("category", "unknown")
            metrics.distillation_patterns_extracted.labels(category=cat).inc()

    if not all_patterns:
        _update_session_index(session_files, date, project)
        log.info("No patterns passed the recurrence gate for project %s on %s", project, date.date())
        return {"sessions": len(session_files), "patterns_extracted": 0, "patterns_gated": gated_out}

    episode_path = CORPUS_ROOT / "episodes" / project / date.strftime("%Y/%m/%d.jsonl")
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    with open(episode_path, "a") as fh:
        for p in all_patterns:
            fh.write(json.dumps(p) + "\n")

    log.info(
        "Distillation complete for %s: %d sessions, %d patterns saved, %d gated out",
        project,
        len(session_files),
        len(all_patterns),
        gated_out,
    )

    _update_session_index(session_files, date, project)
    _update_corpus_stats(all_patterns)

    return {
        "sessions": len(session_files),
        "patterns_extracted": len(all_patterns),
        "patterns_gated": gated_out,
    }


def run_daily(date: datetime | None = None) -> dict:
    """Distill all sessions for all projects for the given date."""
    if date is None:
        date = datetime.now(tz=timezone.utc)

    projects = _session_projects()
    if not projects:
        log.info("No project session directories found — skipping distillation")
        return {"sessions": 0, "patterns_extracted": 0, "patterns_gated": 0}

    prompt_template = _load_distill_prompt()
    totals = {"sessions": 0, "patterns_extracted": 0, "patterns_gated": 0}

    for project in projects:
        result = _distill_project(project, date, prompt_template)
        for key in totals:
            totals[key] += result[key]

    return totals


def _processed_session_ids(project: str = "") -> set[str]:
    index_path = CORPUS_ROOT / "meta" / "session_index.json"
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {"sessions": []}
    except json.JSONDecodeError:
        return set()
    sessions = index.get("sessions", [])
    if project:
        sessions = [s for s in sessions if s.get("project") == project]
    return {record.get("session_id") for record in sessions if record.get("session_id")}


def _update_session_index(session_files: list[pathlib.Path], date: datetime, project: str) -> None:
    index_path = CORPUS_ROOT / "meta" / "session_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {"sessions": []}
    except json.JSONDecodeError:
        index = {"sessions": []}

    existing_ids = {s.get("session_id") for s in index["sessions"]}
    for f in session_files:
        if f.stem not in existing_ids:
            index["sessions"].append({
                "session_id": f.stem,
                "project": project,
                "date": date.strftime("%Y-%m-%d"),
                "path": str(f),
            })
    index_path.write_text(json.dumps(index, indent=2))


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
