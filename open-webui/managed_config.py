"""Keep Docker-managed OpenAI settings out of Open WebUI's config database."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


def remove_managed_openai_settings(database: Path) -> bool:
    if not database.is_file():
        return False

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT id, data FROM config ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return False

        row_id, raw_data = row
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        openai = data.get("openai")
        if not isinstance(openai, dict):
            return False

        changed = False
        for key in ("api_base_urls", "api_keys"):
            if key in openai:
                del openai[key]
                changed = True

        if changed:
            connection.execute(
                "UPDATE config SET data = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(data), row_id),
            )
        return changed


if __name__ == "__main__":
    data_dir = Path(os.environ.get("DATA_DIR", "/app/backend/data"))
    remove_managed_openai_settings(data_dir / "webui.db")
