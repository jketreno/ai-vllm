"""Corpus assembler: active themes + recent episodes → SFT pairs for training."""

import json
import logging
import os
import pathlib
from datetime import datetime, timedelta, timezone

from . import metrics

log = logging.getLogger(__name__)

CORPUS_ROOT = pathlib.Path(os.environ.get("CORPUS_ROOT", "/corpus"))

# Category weights — antipatterns get 1.5× signal weight
CATEGORY_WEIGHTS: dict[str, float] = {
    "style": 1.0,
    "architecture": 1.0,
    "antipattern": 1.5,
    "domain": 1.0,
}

# How many days of recent episodes to include alongside themes
RECENT_EPISODE_WINDOW_DAYS = 7


def _corpus_projects() -> list[str]:
    """Return sorted list of project names that have episodes or active themes."""
    projects: set[str] = set()
    for subdir in (CORPUS_ROOT / "episodes", CORPUS_ROOT / "themes" / "active"):
        if subdir.exists():
            for p in subdir.iterdir():
                if p.is_dir():
                    projects.add(p.name)
    return sorted(projects)


def _load_active_themes(project: str) -> list[dict]:
    themes: list[dict] = []
    theme_dir = CORPUS_ROOT / "themes" / "active" / project
    if not theme_dir.exists():
        return themes
    for tf in sorted(theme_dir.glob("*.jsonl")):
        with open(tf) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        r["_source_type"] = "theme"
                        r["_source_file"] = tf.name
                        themes.append(r)
                    except json.JSONDecodeError:
                        pass
    return themes


def _load_recent_episodes(project: str, days: int = RECENT_EPISODE_WINDOW_DAYS) -> list[dict]:
    episodes: list[dict] = []
    now = datetime.now(tz=timezone.utc)
    for offset in range(days):
        day = now - timedelta(days=offset)
        ep_path = CORPUS_ROOT / "episodes" / project / day.strftime("%Y/%m/%d.jsonl")
        if ep_path.exists():
            with open(ep_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                            r["_source_type"] = "episode"
                            episodes.append(r)
                        except json.JSONDecodeError:
                            pass
    return episodes


def _pattern_to_sft_pair(pattern: dict) -> dict | None:
    """Convert a distilled pattern record into an SFT prompt/completion pair."""
    category = pattern.get("category", "unknown")
    description = pattern.get("pattern", pattern.get("pattern_description", ""))
    canonical = pattern.get("canonical_example", "")

    if not description:
        return None

    weight = CATEGORY_WEIGHTS.get(category, 1.0)
    source_theme = pattern.get("source_theme", pattern.get("_source_file", ""))

    # SFT pair: the prompt is the trigger scenario; the completion is correct behavior
    if category == "antipattern":
        prompt = (
            f"The following code or approach has been identified as an anti-pattern in this codebase. "
            f"Describe what should be done instead.\n\nAnti-pattern: {description}"
        )
        completion = (
            f"Avoid this pattern. {description} "
            f"{'Example of the issue: ' + canonical if canonical else ''}"
        )
    elif category == "style":
        prompt = (
            f"Apply the project's naming and style conventions to the following. "
            f"Convention: {description}"
        )
        completion = (
            f"Follow this style convention: {description}. "
            f"{'Reference: ' + canonical if canonical else ''}"
        )
    elif category == "architecture":
        prompt = (
            f"The project uses a specific architectural pattern. "
            f"Apply it correctly. Pattern: {description}"
        )
        completion = (
            f"Use this architectural pattern: {description}. "
            f"{'Example: ' + canonical if canonical else ''}"
        )
    else:  # domain
        prompt = (
            f"Use the correct domain terminology for this project. "
            f"Convention: {description}"
        )
        completion = (
            f"Use this domain term/concept correctly: {description}. "
            f"{'Reference: ' + canonical if canonical else ''}"
        )

    return {
        "prompt": prompt,
        "completion": completion,
        "category": category,
        "weight": weight,
        "source_theme": source_theme,
    }


def _assemble_project(project: str) -> dict:
    """Assemble training/current.jsonl for a single project."""
    themes = _load_active_themes(project)
    episodes = _load_recent_episodes(project)
    all_patterns = themes + episodes

    sft_pairs: list[dict] = []
    for p in all_patterns:
        pair = _pattern_to_sft_pair(p)
        if pair:
            sft_pairs.append(pair)

    if not sft_pairs:
        log.warning("No SFT pairs generated for project %s — corpus may be empty", project)

    training_dir = CORPUS_ROOT / "training" / project
    training_dir.mkdir(parents=True, exist_ok=True)
    current_path = training_dir / "current.jsonl"

    # Snapshot the previous current.jsonl before overwriting
    if current_path.exists():
        snapshots_dir = training_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        snap_path = snapshots_dir / f"{now_str}.jsonl"
        current_path.rename(snap_path)

    with open(current_path, "w") as fh:
        for pair in sft_pairs:
            fh.write(json.dumps(pair) + "\n")

    # Rough token estimate: ~1 token per 4 chars
    total_chars = sum(len(p["prompt"]) + len(p["completion"]) for p in sft_pairs)
    estimated_tokens = total_chars // 4

    _update_manifest(project, sft_pairs, estimated_tokens)

    log.info(
        "Corpus assembled for %s: %d SFT pairs, ~%d tokens",
        project,
        len(sft_pairs),
        estimated_tokens,
    )
    return {"sft_pairs": len(sft_pairs), "estimated_tokens": estimated_tokens}


def assemble() -> dict:
    """Generate per-project training/current.jsonl from active themes + recent episodes."""
    projects = _corpus_projects()
    if not projects:
        log.warning("No projects found in episodes or themes/active — corpus is empty")
        return {"sft_pairs": 0, "estimated_tokens": 0}

    totals = {"sft_pairs": 0, "estimated_tokens": 0}
    for project in projects:
        result = _assemble_project(project)
        for key in totals:
            totals[key] += result[key]

    metrics.corpus_sft_pairs.set(totals["sft_pairs"])
    metrics.corpus_tokens_total.set(totals["estimated_tokens"])
    log.info(
        "Corpus assembly complete: %d projects, %d total SFT pairs, ~%d total tokens",
        len(projects),
        totals["sft_pairs"],
        totals["estimated_tokens"],
    )
    return totals


def _update_manifest(project: str, sft_pairs: list[dict], estimated_tokens: int) -> None:
    manifest_path = CORPUS_ROOT / "training" / project / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    except json.JSONDecodeError:
        manifest = {"runs": []}

    manifest["last_updated"] = datetime.now(tz=timezone.utc).isoformat()
    manifest["total_sft_pairs"] = len(sft_pairs)
    manifest["total_tokens"] = estimated_tokens
    manifest_path.write_text(json.dumps(manifest, indent=2))
