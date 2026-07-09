"""Keep Docker-managed OpenAI settings out of Open WebUI's config database."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


def _load_config(connection: sqlite3.Connection):
    row = connection.execute(
        "SELECT id, data FROM config ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None, None

    row_id, raw_data = row
    data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    return row_id, data


def _save_config(connection: sqlite3.Connection, row_id, data) -> None:
    connection.execute(
        "UPDATE config SET data = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (json.dumps(data), row_id),
    )


def _delete_keys(section, keys: tuple[str, ...]) -> bool:
    changed = False
    for key in keys:
        if key in section:
            del section[key]
            changed = True
    return changed


def remove_managed_openai_settings(database: Path) -> bool:
    if not database.is_file():
        return False

    with sqlite3.connect(database) as connection:
        row_id, data = _load_config(connection)
        if data is None:
            return False

        openai = data.get("openai")
        if not isinstance(openai, dict):
            return False

        changed = _delete_keys(openai, ("api_base_urls", "api_keys"))

        if changed:
            _save_config(connection, row_id, data)
        return changed


def remove_managed_image_generation_settings(database: Path) -> bool:
    if not database.is_file():
        return False

    with sqlite3.connect(database) as connection:
        row_id, data = _load_config(connection)
        if data is None:
            return False

        image_generation = data.get("image_generation")
        if not isinstance(image_generation, dict):
            return False

        changed = _delete_keys(image_generation, ("engine", "model", "size", "steps"))

        comfyui = image_generation.get("comfyui")
        if isinstance(comfyui, dict):
            changed = _delete_keys(comfyui, ("base_url", "workflow", "nodes")) or changed

        if changed:
            _save_config(connection, row_id, data)
        return changed


if __name__ == "__main__":
    data_dir = Path(os.environ.get("DATA_DIR", "/app/backend/data"))
    database = data_dir / "webui.db"
    remove_managed_openai_settings(database)
    remove_managed_image_generation_settings(database)
