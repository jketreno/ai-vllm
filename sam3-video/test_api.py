import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

MODULE_DIR = Path(__file__).parent
spec = importlib.util.spec_from_file_location("sam3_video_api", MODULE_DIR / "api.py")
api = importlib.util.module_from_spec(spec)
sys.modules["sam3_video_api"] = api
spec.loader.exec_module(api)


def test_capabilities_expose_session_lifecycle():
    with patch.object(api, "_token", return_value="x" * 32):
        result = api.capabilities()
    assert "propagate" in result["operations"]
    assert "masklet_export" in result["operations"]


def test_service_auth_rejects_browser_tokens():
    with (
        patch.object(api, "_token", return_value="s" * 32),
        pytest.raises(HTTPException, match="invalid service"),
    ):
        api.require_service("Bearer browser-token")


def test_session_path_cannot_escape_root(tmp_path: Path):
    with (
        patch.object(api, "SESSION_ROOT", tmp_path),
        pytest.raises(HTTPException, match="session not found"),
    ):
        api._session("../../etc")
