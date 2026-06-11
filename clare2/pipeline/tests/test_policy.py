from __future__ import annotations

import hashlib
import hmac
import json
import pathlib
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.controller import AdapterController
from app.evaluator import compare
from app.local_llm import generate
from app.proxy import router_api
from app.registry import AdapterRegistry, RegistryError
from app.routing import Router
from app.security import require_bearer, verify_callback

BASE = {
    "model_id": "Qwen/Qwen3.5-35B-A3B-FP8",
    "revision": "abc123",
    "architecture": "Qwen3_5MoeForConditionalGeneration",
    "config_hash": "config",
    "tokenizer_hash": "tokenizer",
    "inference_quantization": "fp8",
}


def safetensors(path: pathlib.Path) -> None:
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def adapter(models: pathlib.Path, adapter_id: str, **overrides) -> dict:
    directory = models / "adapters" / adapter_id
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})
    )
    safetensors(directory / "adapter_model.safetensors")
    result = {
        "id": adapter_id,
        "directory": adapter_id,
        "created_at": "2026-06-11T00:00:00+00:00",
        "corpus_hash": "c" * 64,
        "base": {
            "model_id": BASE["model_id"],
            "revision": BASE["revision"],
            "config_hash": BASE["config_hash"],
            "tokenizer_hash": BASE["tokenizer_hash"],
        },
        "peft": {"rank": 32, "alpha": 64, "dropout": 0.05},
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "evaluation": None,
        "project_scope": "github:example/project",
        "capabilities": ["code", "review"],
        "status": "approved",
    }
    result.update(overrides)
    return result


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.models = pathlib.Path(self.temp.name)
        self.registry = AdapterRegistry(self.models)
        self.registry.initialize(BASE)

    def tearDown(self):
        self.temp.cleanup()

    def test_registry_write_is_valid_and_leaves_no_temporary_file(self):
        item = adapter(self.models, "clare-project-20260611T000000Z-12345678")
        self.registry.add_adapter(item)
        self.assertEqual(self.registry.read()["adapters"][item["id"]]["id"], item["id"])
        self.assertEqual(list((self.models / "adapters").glob(".registry.*")), [])

    def test_rejects_duplicate_and_wrong_base(self):
        item = adapter(self.models, "clare-project-20260611T000000Z-12345678")
        self.registry.add_adapter(item)
        with self.assertRaises(RegistryError):
            self.registry.add_adapter(item)
        wrong = adapter(self.models, "clare-project-20260611T000001Z-12345679")
        wrong["base"]["revision"] = "different"
        with self.assertRaisesRegex(RegistryError, "fingerprint"):
            self.registry.add_adapter(wrong)

    def test_rejects_symlink_escape_and_unsupported_module(self):
        outside = self.models / "outside"
        outside.mkdir()
        (outside / "adapter_config.json").write_text("{}")
        safetensors(outside / "adapter_model.safetensors")
        adapter_id = "clare-project-20260611T000000Z-12345678"
        (self.models / "adapters" / adapter_id).symlink_to(outside, target_is_directory=True)
        item = {
            **adapter(self.models, "clare-project-20260611T000001Z-12345679"),
            "id": adapter_id,
            "directory": adapter_id,
        }
        with self.assertRaises(RegistryError):
            self.registry.validate_adapter(item)
        item["id"] = "clare-project-20260611T000001Z-12345679"
        item["directory"] = item["id"]
        item["target_modules"] = ["gate_proj"]
        with self.assertRaisesRegex(RegistryError, "unsupported"):
            self.registry.validate_adapter(item)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.models = pathlib.Path(self.temp.name)
        self.registry = AdapterRegistry(self.models)
        self.registry.initialize(BASE)
        self.project = adapter(self.models, "clare-project-20260611T000000Z-12345678")
        self.global_adapter = adapter(
            self.models,
            "clare-global-20260611T000001Z-12345679",
            project_scope="global",
        )
        self.registry.add_adapter(self.project)
        self.registry.add_adapter(self.global_adapter)
        self.router = Router(self.registry, {"repo": "github:example/project"})

    def tearDown(self):
        self.temp.cleanup()

    def test_project_precedes_global_and_route_stays_pinned(self):
        route = self.router.create_route("repo", "review", ["code"])
        self.assertEqual(route.adapter_id, self.project["id"])
        newer = adapter(
            self.models,
            "clare-project-20260611T000002Z-12345670",
            created_at="2026-06-11T00:02:00+00:00",
        )
        self.registry.add_adapter(newer)
        self.assertEqual(self.router.get(route.route_id).adapter_id, self.project["id"])

    def test_base_fallback_when_capabilities_do_not_match(self):
        route = self.router.create_route("repo", "vision", ["vision"])
        self.assertIsNone(route.adapter_id)
        self.assertEqual(route.policy_rule, "base_fallback")


class FakeVllm:
    def __init__(self):
        self.loaded: set[str] = set()
        self.load_calls: list[str] = []
        self.unload_calls: list[str] = []
        self.lock = threading.Lock()

    def models(self):
        return set(self.loaded)

    def load(self, adapter_id, path):
        del path
        with self.lock:
            time.sleep(0.01)
            self.load_calls.append(adapter_id)
            self.loaded.add(adapter_id)

    def unload(self, adapter_id):
        self.unload_calls.append(adapter_id)
        self.loaded.remove(adapter_id)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.models = pathlib.Path(self.temp.name)
        self.registry = AdapterRegistry(self.models)
        self.registry.initialize(BASE)
        self.items = [
            adapter(self.models, f"clare-project-20260611T00000{i}Z-1234567{i}")
            for i in range(3)
        ]
        for item in self.items:
            self.registry.add_adapter(item)
        self.fake = FakeVllm()
        self.pinned: set[str] = set()
        self.controller = AdapterController(
            self.registry,
            self.fake,
            lambda: self.pinned,
            max_cpu_loras=2,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_concurrent_first_load_is_serialized(self):
        threads = [
            threading.Thread(target=self.controller.ensure_loaded, args=(self.items[0]["id"],))
            for _ in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(self.fake.load_calls, [self.items[0]["id"]])

    def test_lru_never_evicts_pinned_adapter(self):
        self.controller.ensure_loaded(self.items[0]["id"])
        self.controller.ensure_loaded(self.items[1]["id"])
        self.pinned.add(self.items[0]["id"])
        self.controller.ensure_loaded(self.items[2]["id"])
        self.assertEqual(self.fake.unload_calls, [self.items[1]["id"]])


class SecurityAndEvaluationTests(unittest.TestCase):
    def test_bearer_and_callback_hmac(self):
        require_bearer("secret", "Bearer secret")
        with self.assertRaises(HTTPException):
            require_bearer("secret", "Bearer wrong")
        body = b'{"ok":true}'
        timestamp = str(int(time.time()))
        signature = hmac.new(b"secret", timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
        verify_callback("secret", body, timestamp, signature)
        with self.assertRaises(HTTPException):
            verify_callback("secret", body, timestamp, "wrong")

    def test_promotion_requires_threshold_and_no_category_regression(self):
        probes = [
            {"id": f"p{i}", "prompt": str(i), "expected_keyword": "pass", "category": "code"}
            for i in range(10)
        ]
        report = compare(
            "candidate",
            "baseline",
            lambda model, probe: "pass" if model == "candidate" or probe["id"] != "p0" else "fail",
            probes,
        )
        self.assertTrue(report["approved"])
        regressed = compare(
            "candidate",
            "baseline",
            lambda model, probe: "fail" if model == "candidate" and probe["id"] == "p0" else "pass",
            probes,
        )
        self.assertFalse(regressed["approved"])

    def test_local_generation_uses_deterministic_qwen_request(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"choices": [{"message": {"content": "[]"}}]}
        with patch("app.local_llm.httpx.post", return_value=response) as post:
            self.assertEqual(generate("distill this"), "[]")
        response.raise_for_status.assert_called_once()
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "Qwen/Qwen3.5-35B-A3B-FP8")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 42)


class ProxyIntegrationTests(unittest.TestCase):
    def test_proxy_overwrites_model_and_blocks_management_routes(self):
        captured: dict = {}

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                del kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def request(self, method, url, content, headers):
                del method, url, headers
                captured.update(json.loads(content))
                return httpx.Response(
                    200,
                    json={"choices": []},
                    headers={"content-type": "application/json"},
                )

        app = FastAPI()
        app.include_router(router_api)
        client = TestClient(app)
        with patch.dict("os.environ", {"CLARE2_PROXY_TOKEN": "secret"}), patch(
            "app.proxy.httpx.AsyncClient",
            FakeAsyncClient,
        ):
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={"model": "attacker-selected", "messages": []},
            )
            blocked = client.post(
                "/v1/load_lora_adapter",
                headers={"Authorization": "Bearer secret"},
                json={"lora_path": "/tmp/escape"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["model"], "Qwen/Qwen3.5-35B-A3B-FP8")
        self.assertEqual(blocked.status_code, 404)


if __name__ == "__main__":
    unittest.main()
