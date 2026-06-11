"""Daily distillation pass: raw sessions → episode store."""

import json
import logging
import os
import pathlib
from datetime import datetime, timezone

import anthropic
import openai

from . import metrics
from .security import secret_value

log = logging.getLogger(__name__)

CORPUS_ROOT = pathlib.Path(os.environ.get("CORPUS_ROOT", "/corpus"))
DISTILL_MODEL = os.environ.get("CLARE2_DISTILL_MODEL", "claude-haiku-4-5")
RECURRENCE_GATE = 2  # minimum evidence_count within a session to pass gate


def _load_distill_prompt() -> str:
    prompt_path = pathlib.Path("/app/prompts/distill.txt")
    return prompt_path.read_text()


def _session_files_for_date(date: datetime) -> list[pathlib.Path]:
    day_dir = CORPUS_ROOT / "sessions" / date.strftime("%Y/%m/%d")
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
    """Call the distillation LLM with the session content."""
    prompt = prompt_template.replace("{{SESSION_CONTENT}}", session_text)

    if DISTILL_MODEL.startswith("claude"):
        client = anthropic.Anthropic(api_key=secret_value("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=DISTILL_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    else:
        # Local model via OpenAI-compatible endpoint
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
        return resp.choices[0].message.content


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


def run_daily(date: datetime | None = None) -> dict:
    """Distill all sessions for the given date into an episode file."""
    if date is None:
        date = datetime.now(tz=timezone.utc)

    session_files = _session_files_for_date(date)
    processed_ids = _processed_session_ids()
    session_files = [path for path in session_files if path.stem not in processed_ids]
    if not session_files:
        log.info("No session files found for %s — skipping distillation", date.date())
        return {"sessions": 0, "patterns_extracted": 0, "patterns_gated": 0}

    prompt_template = _load_distill_prompt()
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
            p["distilled_at"] = datetime.now(tz=timezone.utc).isoformat()
            all_patterns.append(p)

            cat = p.get("category", "unknown")
            metrics.distillation_patterns_extracted.labels(category=cat).inc()

    if not all_patterns:
        _update_session_index(session_files, date)
        log.info("No patterns passed the recurrence gate for %s", date.date())
        return {"sessions": len(session_files), "patterns_extracted": 0, "patterns_gated": gated_out}

    episode_path = CORPUS_ROOT / "episodes" / date.strftime("%Y/%m/%d.jsonl")
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    with open(episode_path, "a") as fh:
        for p in all_patterns:
            fh.write(json.dumps(p) + "\n")

    log.info(
        "Distillation complete: %d sessions, %d patterns saved, %d gated out",
        len(session_files),
        len(all_patterns),
        gated_out,
    )

    _update_session_index(session_files, date)
    _update_corpus_stats(all_patterns)

    return {
        "sessions": len(session_files),
        "patterns_extracted": len(all_patterns),
        "patterns_gated": gated_out,
    }


def _processed_session_ids() -> set[str]:
    index_path = CORPUS_ROOT / "meta" / "session_index.json"
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {"sessions": []}
    except json.JSONDecodeError:
        return set()
    return {record.get("session_id") for record in index.get("sessions", []) if record.get("session_id")}


def _update_session_index(session_files: list[pathlib.Path], date: datetime) -> None:
    index_path = CORPUS_ROOT / "meta" / "session_index.json"
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {"sessions": []}
    except json.JSONDecodeError:
        index = {"sessions": []}

    existing_ids = {s.get("session_id") for s in index["sessions"]}
    for f in session_files:
        if f.stem not in existing_ids:
            index["sessions"].append({
                "session_id": f.stem,
                "date": date.strftime("%Y-%m-%d"),
                "path": str(f),
            })
    index_path.write_text(json.dumps(index, indent=2))


def _update_corpus_stats(patterns: list[dict]) -> None:
    stats_path = CORPUS_ROOT / "meta" / "corpus_stats.json"
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
