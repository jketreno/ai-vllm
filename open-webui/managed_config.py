"""Keep Docker-managed OpenAI settings out of Open WebUI's config database."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


def _delete_normalized_keys(
    connection: sqlite3.Connection, keys: tuple[str, ...]
) -> bool | None:
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(config)").fetchall()
    }
    if not {"key", "value"}.issubset(columns):
        return None

    placeholders = ", ".join("?" for _ in keys)
    cursor = connection.execute(
        f'DELETE FROM config WHERE "key" IN ({placeholders})', keys
    )
    return cursor.rowcount > 0


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
        changed = _delete_normalized_keys(
            connection, ("openai.api_base_urls", "openai.api_keys")
        )
        if changed is not None:
            return changed

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


def _strip_image_provider_sections(
    section, top_level_keys: tuple[str, ...]
) -> bool:
    if not isinstance(section, dict):
        return False

    changed = _delete_keys(section, top_level_keys)

    comfyui = section.get("comfyui")
    if isinstance(comfyui, dict):
        changed = _delete_keys(comfyui, ("base_url", "workflow", "nodes")) or changed

    openai = section.get("openai")
    if isinstance(openai, dict):
        changed = _delete_keys(openai, ("api_base_url", "api_key")) or changed

    return changed


def remove_managed_image_generation_settings(database: Path) -> bool:
    if not database.is_file():
        return False

    with sqlite3.connect(database) as connection:
        changed = _delete_normalized_keys(
            connection,
            (
                "image_generation.engine",
                "image_generation.model",
                "image_generation.size",
                "image_generation.steps",
                "image_generation.openai.api_base_url",
                "image_generation.openai.api_key",
                "image_generation.comfyui.base_url",
                "image_generation.comfyui.workflow",
                "image_generation.comfyui.nodes",
            ),
        )
        if changed is not None:
            return changed

        row_id, data = _load_config(connection)
        if data is None:
            return False

        changed = _strip_image_provider_sections(
            data.get("image_generation"), ("engine", "model", "size", "steps")
        )

        if changed:
            _save_config(connection, row_id, data)
        return changed


def remove_managed_image_edit_settings(database: Path) -> bool:
    if not database.is_file():
        return False

    with sqlite3.connect(database) as connection:
        changed = _delete_normalized_keys(
            connection,
            (
                "images.edit.engine",
                "images.edit.model",
                "images.edit.size",
                "images.edit.openai.api_base_url",
                "images.edit.openai.api_key",
                "images.edit.comfyui.base_url",
                "images.edit.comfyui.workflow",
                "images.edit.comfyui.nodes",
            ),
        )
        if changed is not None:
            return changed

        row_id, data = _load_config(connection)
        if data is None:
            return False

        images = data.get("images")
        edit = images.get("edit") if isinstance(images, dict) else None
        changed = _strip_image_provider_sections(edit, ("engine", "model", "size"))

        if changed:
            _save_config(connection, row_id, data)
        return changed


if __name__ == "__main__":
    data_dir = Path(os.environ.get("DATA_DIR", "/app/backend/data"))
    database = data_dir / "webui.db"
    remove_managed_openai_settings(database)
    remove_managed_image_generation_settings(database)
    remove_managed_image_edit_settings(database)
