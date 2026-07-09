"""Corpus assembler: active themes + recent episodes → SFT pairs for training."""

import json
import logging
import os
import pathlib
from datetime import datetime, timezone

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


def _episode_date(path: pathlib.Path) -> datetime:
    relative = path.relative_to(CORPUS_ROOT / "episodes")
    year, month, filename = relative.parts[1:4]
    day = pathlib.Path(filename).stem
    return datetime(int(year), int(month), int(day), tzinfo=timezone.utc)


def _recent_episode_paths(project: str, days: int) -> list[pathlib.Path]:
    project_dir = CORPUS_ROOT / "episodes" / project
    if not project_dir.exists():
        return []
    dated_paths = [(_episode_date(path), path) for path in project_dir.glob("*/*/*.jsonl")]
    return [path for _date, path in sorted(dated_paths, reverse=True)[:days]]


def _load_recent_episodes(project: str, days: int = RECENT_EPISODE_WINDOW_DAYS) -> list[dict]:
    episodes: list[dict] = []
    for ep_path in _recent_episode_paths(project, days):
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
    project = pattern.get("project", "this project")
    evidence_count = pattern.get("evidence_count", 1)
    session_id = pattern.get("session_id", "")
    source_type = pattern.get("_source_type", "pattern")
    reference = f"\nReference example: {canonical}" if canonical else ""
    context = (
        f"Project: {project}\n"
        f"Signal category: {category}\n"
        f"Evidence count: {evidence_count}\n"
        f"Source: {source_type}"
        f"{f' ({session_id})' if session_id else ''}\n"
    )

    # SFT pair: prompt presents a realistic decision point; completion models the desired action.
    if category == "antipattern":
        prompt = (
            f"{context}\n"
            "You are modifying this repository and encounter the following anti-pattern.\n"
            f"Anti-pattern: {description}\n\n"
            "Respond with the concrete behavior you should apply instead."
        )
        completion = (
            f"Avoid the anti-pattern: {description}\n"
            "Instead, verify the behavior through the repository's authoritative path, "
            "preserve the existing project constraints, and explain the safer implementation choice."
            f"{reference}"
        )
    elif category == "style":
        prompt = (
            f"{context}\n"
            "You are preparing a code or documentation change for this repository.\n"
            f"Style convention: {description}\n\n"
            "Respond with how you should apply this convention."
        )
        completion = (
            f"Apply the style convention consistently: {description}\n"
            "Use the convention where it affects prose or code users will see, while preserving literal "
            "identifiers, paths, environment variables, and protocol values."
            f"{reference}"
        )
    elif category == "architecture":
        prompt = (
            f"{context}\n"
            "You are designing or changing a component in this repository.\n"
            f"Architectural pattern: {description}\n\n"
            "Respond with the implementation approach that respects this architecture."
        )
        completion = (
            f"Follow the established architecture: {description}\n"
            "Keep ownership boundaries explicit, derive generated artifacts from their sources, and add "
            "focused verification for the invariant being protected."
            f"{reference}"
        )
    else:  # domain
        prompt = (
            f"{context}\n"
            "You are answering or implementing a change for this repository.\n"
            f"Domain rule: {description}\n\n"
            "Respond with the correct project-specific handling."
        )
        completion = (
            f"Use the project-specific rule directly: {description}\n"
            "Prefer the repository's configured source of truth over assumptions, and make the resulting "
            "behavior observable through the normal project workflow."
            f"{reference}"
        )

    return {
        "prompt": prompt,
        "completion": completion,
        "category": category,
        "weight": weight,
        "source_theme": source_theme,
        "source_session": session_id,
        "source_date": pattern.get("session_date", ""),
        "evidence_count": evidence_count,
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
        metrics.corpus_sft_pairs.labels(project=project).set(result["sft_pairs"])
        metrics.corpus_tokens_total.labels(project=project).set(result["estimated_tokens"])

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
