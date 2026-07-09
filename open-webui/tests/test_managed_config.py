"""Tests for Docker-managed Open WebUI configuration."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "managed_config.py"
SPEC = importlib.util.spec_from_file_location("managed_config", MODULE_PATH)
assert SPEC and SPEC.loader
MANAGED_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANAGED_CONFIG)


class ManagedConfigTests(unittest.TestCase):
    def test_removes_only_compose_managed_openai_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "webui.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TABLE config "
                    "(id INTEGER PRIMARY KEY, data JSON NOT NULL, "
                    "updated_at DATETIME)"
                )
                connection.execute(
                    "INSERT INTO config (data) VALUES (?)",
                    (
                        json.dumps(
                            {
                                "openai": {
                                    "enable": True,
                                    "api_base_urls": ["http://old/v1"],
                                    "api_keys": ["old-secret"],
                                    "api_configs": {"0": {"enable": True}},
                                },
                                "ui": {"enable_signup": False},
                            }
                        ),
                    ),
                )

            changed = MANAGED_CONFIG.remove_managed_openai_settings(database)

            self.assertTrue(changed)
            with sqlite3.connect(database) as connection:
                data = json.loads(connection.execute("SELECT data FROM config").fetchone()[0])
            self.assertNotIn("api_base_urls", data["openai"])
            self.assertNotIn("api_keys", data["openai"])
            self.assertEqual(data["openai"]["api_configs"], {"0": {"enable": True}})
            self.assertEqual(data["ui"], {"enable_signup": False})

    def test_missing_database_is_ignored(self) -> None:
        self.assertFalse(
            MANAGED_CONFIG.remove_managed_openai_settings(Path("/missing/webui.db"))
        )

    def test_removes_only_compose_managed_image_generation_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "webui.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TABLE config "
                    "(id INTEGER PRIMARY KEY, data JSON NOT NULL, "
                    "updated_at DATETIME)"
                )
                connection.execute(
                    "INSERT INTO config (data) VALUES (?)",
                    (
                        json.dumps(
                            {
                                "image_generation": {
                                    "enable": True,
                                    "engine": "comfyui",
                                    "model": "model.safetensors",
                                    "size": "512x512",
                                    "steps": 50,
                                    "comfyui": {
                                        "api_key": "",
                                        "base_url": "http://old:8188",
                                        "workflow": "{}",
                                        "nodes": [],
                                    },
                                },
                                "ui": {"enable_signup": False},
                            }
                        ),
                    ),
                )

            changed = MANAGED_CONFIG.remove_managed_image_generation_settings(database)

            self.assertTrue(changed)
            with sqlite3.connect(database) as connection:
                data = json.loads(connection.execute("SELECT data FROM config").fetchone()[0])
            image_generation = data["image_generation"]
            self.assertNotIn("engine", image_generation)
            self.assertNotIn("model", image_generation)
            self.assertNotIn("size", image_generation)
            self.assertNotIn("steps", image_generation)
            self.assertNotIn("base_url", image_generation["comfyui"])
            self.assertNotIn("workflow", image_generation["comfyui"])
            self.assertNotIn("nodes", image_generation["comfyui"])
            self.assertEqual(image_generation["comfyui"]["api_key"], "")
            self.assertTrue(image_generation["enable"])
            self.assertEqual(data["ui"], {"enable_signup": False})

    def test_missing_image_generation_database_is_ignored(self) -> None:
        self.assertFalse(
            MANAGED_CONFIG.remove_managed_image_generation_settings(
                Path("/missing/webui.db")
            )
        )


if __name__ == "__main__":
    unittest.main()
