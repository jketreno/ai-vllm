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
    def test_removes_managed_values_from_normalized_config(self) -> None:
        managed_keys = {
            "openai.api_base_urls",
            "openai.api_keys",
            "image_generation.engine",
            "image_generation.model",
            "image_generation.size",
            "image_generation.steps",
            "image_generation.openai.api_base_url",
            "image_generation.openai.api_key",
            "image_generation.comfyui.base_url",
            "image_generation.comfyui.workflow",
            "image_generation.comfyui.nodes",
            "images.edit.engine",
            "images.edit.model",
            "images.edit.size",
            "images.edit.openai.api_base_url",
            "images.edit.openai.api_key",
            "images.edit.comfyui.base_url",
            "images.edit.comfyui.workflow",
            "images.edit.comfyui.nodes",
        }

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "webui.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    'CREATE TABLE config ("key" TEXT PRIMARY KEY, '
                    "value JSON NOT NULL, updated_at BIGINT)"
                )
                connection.executemany(
                    'INSERT INTO config ("key", value) VALUES (?, ?)',
                    [(key, json.dumps("managed")) for key in managed_keys]
                    + [("ui.enable_signup", "false")],
                )

            self.assertTrue(MANAGED_CONFIG.remove_managed_openai_settings(database))
            self.assertTrue(
                MANAGED_CONFIG.remove_managed_image_generation_settings(database)
            )
            self.assertTrue(MANAGED_CONFIG.remove_managed_image_edit_settings(database))

            with sqlite3.connect(database) as connection:
                remaining_keys = {
                    row[0] for row in connection.execute('SELECT "key" FROM config')
                }
            self.assertEqual(remaining_keys, {"ui.enable_signup"})

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
                data = json.loads(
                    connection.execute("SELECT data FROM config").fetchone()[0]
                )
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
                                    "openai": {
                                        "api_base_url": "http://old/openai/v1",
                                        "api_key": "old-image-secret",
                                    },
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
                data = json.loads(
                    connection.execute("SELECT data FROM config").fetchone()[0]
                )
            image_generation = data["image_generation"]
            self.assertNotIn("engine", image_generation)
            self.assertNotIn("model", image_generation)
            self.assertNotIn("size", image_generation)
            self.assertNotIn("steps", image_generation)
            self.assertNotIn("api_base_url", image_generation["openai"])
            self.assertNotIn("api_key", image_generation["openai"])
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

    def test_removes_only_compose_managed_image_edit_values(self) -> None:
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
                                "images": {
                                    "edit": {
                                        "enable": False,
                                        "engine": "openai",
                                        "model": "",
                                        "size": "",
                                        "openai": {
                                            "api_base_url": "http://old/openai/v1",
                                            "api_key": "old-edit-secret",
                                        },
                                        "comfyui": {
                                            "api_key": "",
                                            "base_url": "",
                                            "workflow": "",
                                            "nodes": [],
                                        },
                                    }
                                },
                                "ui": {"enable_signup": False},
                            }
                        ),
                    ),
                )

            changed = MANAGED_CONFIG.remove_managed_image_edit_settings(database)

            self.assertTrue(changed)
            with sqlite3.connect(database) as connection:
                data = json.loads(
                    connection.execute("SELECT data FROM config").fetchone()[0]
                )
            edit = data["images"]["edit"]
            self.assertNotIn("engine", edit)
            self.assertNotIn("model", edit)
            self.assertNotIn("size", edit)
            self.assertNotIn("api_base_url", edit["openai"])
            self.assertNotIn("api_key", edit["openai"])
            self.assertNotIn("base_url", edit["comfyui"])
            self.assertNotIn("workflow", edit["comfyui"])
            self.assertNotIn("nodes", edit["comfyui"])
            self.assertEqual(edit["comfyui"]["api_key"], "")
            self.assertFalse(edit["enable"])
            self.assertEqual(data["ui"], {"enable_signup": False})

    def test_missing_image_edit_database_is_ignored(self) -> None:
        self.assertFalse(
            MANAGED_CONFIG.remove_managed_image_edit_settings(Path("/missing/webui.db"))
        )


if __name__ == "__main__":
    unittest.main()
