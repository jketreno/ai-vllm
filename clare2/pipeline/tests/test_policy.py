from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import pathlib
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import app.mcp_server as mcp_server
import app.metrics as metrics
import app.main as main
import app.summarizer as summarizer
from app.controller import AdapterController
from app.evaluator import compare
from app.json_output import parse_json_output
from app.local_llm import generate
from app.mcp_server import mcp
from app.proxy import router_api
from app.registry import AdapterRegistry, RegistryError
from app.routing import RouteError, Router
from app.runtime import maintenance
from app.security import BearerASGIMiddleware, require_bearer, verify_callback
from app.structured_output import parse_pattern_records_with_repair

BASE = {
    "model_id": "Qwen/Qwen3.6-27B-FP8",
    "revision": "abc123",
    "architecture": "Qwen3_5MoeForConditionalGeneration",
    "config_hash": "config",
    "tokenizer_hash": "tokenizer",
    "inference_quantization": "fp8",
}


class JsonOutputTests(unittest.TestCase):
    def test_parses_fenced_and_embedded_json(self):
        self.assertEqual(parse_json_output('```json\n[{"ok": true}]\n```'), [{"ok": True}])
        self.assertEqual(
            parse_json_output('Here is the result:\n[{"category": "domain"}]\nDone.'),
            [{"category": "domain"}],
        )

    def test_rejects_missing_json(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_json_output("no structured payload")

    def test_pattern_schema_repairs_invalid_output_once(self):
        repaired = json.dumps([
            {
                "category": "domain",
                "pattern": "Use CLARE2_CORPUS_ROOT for shared corpus mounts.",
                "evidence_count": 2,
                "canonical_example": "Mount ${CLARE2_CORPUS_ROOT}:/corpus.",
                "first_seen": "2026-07-02T18:01:27Z",
                "last_seen": "2026-07-02T18:14:07Z",
            }
        ])
        records, outcome = parse_pattern_records_with_repair(
            "[{\"category\":\"domain\"}]",
            lambda error: repaired,
        )
        self.assertEqual(outcome, "repaired")
        self.assertEqual(records[0]["pattern"], "Use CLARE2_CORPUS_ROOT for shared corpus mounts.")

    def test_pattern_schema_fails_after_repair_attempt(self):
        records, outcome = parse_pattern_records_with_repair("not json", lambda error: "still not json")
        self.assertEqual(outcome, "failed")
        self.assertEqual(records, [])

    def test_pattern_schema_handles_null_model_content(self):
        records, outcome = parse_pattern_records_with_repair(None, lambda error: "[]")
        self.assertEqual(outcome, "repaired")
        self.assertEqual(records, [])


class SummarizerPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_jsonl(self, path: pathlib.Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records))

    def test_weekly_summaries_are_project_scoped(self):
        record = {
            "category": "domain",
            "pattern": "Use project-scoped corpus paths.",
            "evidence_count": 2,
            "canonical_example": "episodes/ai-vllm/2026/07/05.jsonl",
            "first_seen": "2026-07-05T00:00:00Z",
            "last_seen": "2026-07-05T00:01:00Z",
        }
        self._write_jsonl(self.root / "episodes" / "ai-vllm" / "2026/07/05.jsonl", [record])
        with patch.object(summarizer, "CORPUS_ROOT", self.root), patch.object(
            summarizer, "_call_llm_merge", return_value=[record]
        ):
            result = summarizer.run_weekly(datetime(2026, 7, 5, tzinfo=timezone.utc))
        self.assertEqual(result["input_records"], 1)
        self.assertTrue((self.root / "summaries" / "weekly" / "ai-vllm" / "2026-W27.jsonl").exists())
        self.assertFalse((self.root / "summaries" / "weekly" / "2026-W27.jsonl").exists())

    def test_quarterly_themes_are_project_scoped(self):
        record = {
            "category": "domain",
            "pattern": "Use project-scoped active themes.",
            "evidence_count": 2,
            "canonical_example": "themes/active/ai-vllm/domain.jsonl",
            "first_seen": "2026-01-01T00:00:00Z",
            "last_seen": "2026-01-02T00:00:00Z",
        }
        self._write_jsonl(self.root / "summaries" / "monthly" / "ai-vllm" / "2026-01.jsonl", [record])
        with patch.object(summarizer, "CORPUS_ROOT", self.root), patch.object(
            summarizer, "_call_llm_merge", return_value=[record]
        ):
            result = summarizer.run_quarterly(datetime(2026, 4, 1, tzinfo=timezone.utc))
        self.assertEqual(result["themes_promoted"], 1)
        self.assertTrue((self.root / "summaries" / "quarterly" / "ai-vllm" / "2026-Q1.jsonl").exists())
        self.assertTrue((self.root / "themes" / "active" / "ai-vllm" / "domain.jsonl").exists())
        self.assertFalse((self.root / "themes" / "active" / "domain.jsonl").exists())


class IngestFlowTests(unittest.TestCase):
    def test_sync_distill_and_assemble_runs_in_order(self):
        calls = []
        sync = lambda: calls.append("sync") or {"succeeded": 1}
        distill = lambda: calls.append("distill") or {"sessions": 2}
        assemble = lambda: calls.append("assemble") or {"sft_pairs": 3}
        with patch.object(main.corpus_sync, "sync_all", side_effect=sync), patch.object(
            main.distiller, "run_daily", side_effect=distill
        ), patch.object(main.corpus, "assemble", side_effect=assemble):
            result = main.sync_distill_and_assemble()

        self.assertEqual(calls, ["sync", "distill", "assemble"])
        self.assertEqual(result["sync"]["succeeded"], 1)
        self.assertEqual(result["distill"]["sessions"], 2)
        self.assertEqual(result["assemble"]["sft_pairs"], 3)


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

    def test_initialize_refreshes_empty_registry_base(self):
        new_base = {**BASE, "model_id": "Qwen/Qwen3.6-27B-FP8", "revision": "def456"}
        self.registry.initialize(new_base)
        self.assertEqual(self.registry.read()["base"], new_base)

    def test_initialize_preserves_non_empty_registry_base(self):
        item = adapter(self.models, "clare-project-20260611T000000Z-12345678")
        self.registry.add_adapter(item)
        new_base = {**BASE, "model_id": "Qwen/Qwen3.6-27B-FP8", "revision": "def456"}
        self.registry.initialize(new_base)
        self.assertEqual(self.registry.read()["base"], BASE)

    def test_initialize_refreshes_registry_when_adapters_are_inactive(self):
        item = adapter(
            self.models,
            "clare-project-20260611T000000Z-12345678",
            status="rejected",
        )
        self.registry.add_adapter(item)
        new_base = {**BASE, "architecture": "Qwen3_5ForConditionalGeneration"}
        self.registry.initialize(new_base)
        self.assertEqual(self.registry.read()["base"], new_base)

    def test_rejects_duplicate_and_wrong_base(self):
        item = adapter(self.models, "clare-project-20260611T000000Z-12345678")
        self.registry.add_adapter(item)
        with self.assertRaises(RegistryError):
            self.registry.add_adapter(item)
        wrong = adapter(self.models, "clare-project-20260611T000001Z-12345679")
        wrong["base"]["revision"] = "different"
        with self.assertRaisesRegex(RegistryError, "fingerprint"):
            self.registry.add_adapter(wrong)

    def test_accepts_distinct_train_base_when_inference_base_matches(self):
        item = adapter(
            self.models,
            "clare-project-20260611T000000Z-12345678",
            train_base={
                "model_id": "Qwen/Qwen3.6-27B",
                "revision": "non-fp8",
                "architecture": BASE["architecture"],
                "config_hash": "train-config",
                "tokenizer_hash": BASE["tokenizer_hash"],
            },
            inference_base={
                "model_id": BASE["model_id"],
                "revision": BASE["revision"],
                "architecture": BASE["architecture"],
                "config_hash": BASE["config_hash"],
                "tokenizer_hash": BASE["tokenizer_hash"],
                "inference_quantization": BASE["inference_quantization"],
            },
        )
        self.registry.add_adapter(item)
        stored = self.registry.read()["adapters"][item["id"]]
        self.assertEqual(stored["train_base"]["model_id"], "Qwen/Qwen3.6-27B")
        self.assertEqual(stored["inference_base"]["model_id"], BASE["model_id"])

    def test_rejects_mismatched_inference_tokenizer(self):
        item = adapter(
            self.models,
            "clare-project-20260611T000000Z-12345678",
            inference_base={
                "model_id": BASE["model_id"],
                "revision": BASE["revision"],
                "architecture": BASE["architecture"],
                "config_hash": BASE["config_hash"],
                "tokenizer_hash": "different-tokenizer",
                "inference_quantization": BASE["inference_quantization"],
            },
        )
        with self.assertRaisesRegex(RegistryError, "tokenizer_hash"):
            self.registry.add_adapter(item)

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

    def test_rejects_rank_mismatch(self):
        item = adapter(
            self.models,
            "clare-project-20260611T000000Z-12345678",
            peft={"rank": 16, "alpha": 64, "dropout": 0.05},
        )
        with self.assertRaisesRegex(RegistryError, "rank"):
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
    def test_mcp_tools_advertise_descriptions(self):
        tools = asyncio.run(mcp.list_tools())
        descriptions = {tool.name: tool.description for tool in tools}
        expected = {
            "clare_temper_route",
            "clare_temper_status",
            "clare_temper_list",
        }
        self.assertEqual(set(descriptions), expected)
        for name in expected:
            self.assertTrue(descriptions[name], f"{name} must have a description")

    def test_asgi_bearer_middleware(self):
        app = FastAPI()

        @app.get("/health")
        def health():
            return {"status": "ok"}

        client = TestClient(BearerASGIMiddleware(app, "secret"))
        self.assertEqual(client.get("/health").status_code, 401)
        response = client.get("/health", headers={"Authorization": "Bearer secret"})
        self.assertEqual(response.status_code, 200)

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

    def test_mcp_route_tool_calls_policy_internal_api(self):
        def fake_request(method, url, headers=None, timeout=None, **kwargs):
            self.assertEqual(method, "POST")
            self.assertEqual(url, "http://policy.local/internal/routes")
            self.assertEqual(headers, {"Authorization": "Bearer internal-secret"})
            self.assertEqual(timeout, 30)
            self.assertEqual(
                kwargs["json"],
                {
                    "project": "clare",
                    "task_kind": "review",
                    "capabilities": ["chat"],
                },
            )
            return httpx.Response(
                200,
                json={
                    "route_id": "r1",
                    "project_id": "github:jketreno/clare",
                    "adapter_id": None,
                    "policy_rule": "base_fallback",
                    "expires_at": "2026-06-12T00:00:00+00:00",
                },
                request=httpx.Request(method, url),
            )

        with patch.dict(
            "os.environ",
            {
                "CLARE2_POLICY_URL": "http://policy.local",
                "CLARE2_CALLBACK_SECRET": "internal-secret",
            },
            clear=False,
        ), patch("app.mcp_server.httpx.request", side_effect=fake_request):
            original_url = mcp_server.POLICY_URL
            mcp_server.POLICY_URL = "http://policy.local"
            try:
                result = mcp_server.clare_temper_route("clare", "review", ["chat"])
            finally:
                mcp_server.POLICY_URL = original_url

        self.assertEqual(result["route_id"], "r1")

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

    def test_evaluation_score_is_labeled_by_project(self):
        compare(
            "candidate-x",
            "baseline-x",
            lambda model, probe: "pass",
            [{"id": "p0", "prompt": "p", "expected_keyword": "pass", "category": "code"}],
            project="ai-vllm",
        )
        score = metrics.evaluation_score.labels(
            adapter_id="candidate-x", project="ai-vllm", category="code"
        )._value.get()
        self.assertEqual(score, 1.0)

    def test_evaluation_score_defaults_project_to_unknown(self):
        compare(
            "candidate-y",
            "baseline-y",
            lambda model, probe: "pass",
            [{"id": "p0", "prompt": "p", "expected_keyword": "pass", "category": "code"}],
        )
        score = metrics.evaluation_score.labels(
            adapter_id="candidate-y", project="unknown", category="code"
        )._value.get()
        self.assertEqual(score, 1.0)

    def test_evaluation_treats_null_completion_as_empty_text(self):
        report = compare(
            "candidate",
            "baseline",
            lambda model, probe: None if model == "candidate" else "pass",
            [{"id": "p0", "prompt": "p", "expected_keyword": "pass", "category": "code"}],
        )
        self.assertFalse(report["approved"])
        self.assertFalse(report["results"][0]["passed"])

    def test_local_generation_uses_deterministic_qwen_request(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"choices": [{"message": {"content": "[]"}}]}
        with patch("app.local_llm.httpx.post", return_value=response) as post:
            self.assertEqual(generate("distill this"), "[]")
        response.raise_for_status.assert_called_once()
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "Qwen/Qwen3.6-27B-FP8")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_local_generation_converts_null_content_to_empty_string(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"choices": [{"message": {"content": None}}]}
        with patch("app.local_llm.httpx.post", return_value=response):
            self.assertEqual(generate("distill this"), "")


class ProxyIntegrationTests(unittest.TestCase):
    def test_proxy_streams_upstream_chunks_and_releases_request(self):
        captured = {"stream": False, "response_closed": False, "client_closed": False}

        class ChunkStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            async def aclose(self):
                captured["response_closed"] = True

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                del kwargs

            def build_request(self, method, url, content, headers):
                return httpx.Request(method, url, content=content, headers=headers)

            async def send(self, request, *, stream):
                captured["stream"] = stream
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=ChunkStream(),
                    request=request,
                )

            async def aclose(self):
                captured["client_closed"] = True

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
                json={"model": "ignored", "messages": [], "stream": True},
            )

        self.assertTrue(captured["stream"])
        self.assertIn('data: {"choices"', response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertTrue(captured["response_closed"])
        self.assertTrue(captured["client_closed"])
        self.assertEqual(maintenance.active, 0)

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
        self.assertEqual(captured["model"], "Qwen/Qwen3.6-27B-FP8")
        self.assertNotIn("thinking_token_budget", captured)
        self.assertNotIn("chat_template_kwargs", captured)
        self.assertEqual(blocked.status_code, 404)

    def test_proxy_preserves_client_thinking_configuration(self):
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
            "app.proxy.httpx.AsyncClient", FakeAsyncClient
        ):
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={
                    "model": "ignored",
                    "messages": [],
                    "thinking_token_budget": 512,
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                        "preserve_thinking": False,
                    },
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["thinking_token_budget"], 512)
        self.assertEqual(
            captured["chat_template_kwargs"],
            {"enable_thinking": False, "preserve_thinking": False},
        )

    def test_proxy_accepts_route_id_in_path(self):
        captured: dict = {}

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                del kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def request(self, method, url, content, headers):
                del method, headers
                captured["url"] = url
                captured.update(json.loads(content))
                return httpx.Response(
                    200,
                    json={"choices": []},
                    headers={"content-type": "application/json"},
                )

        app = FastAPI()
        app.include_router(router_api)
        client = TestClient(app)
        route = SimpleNamespace(
            adapter_id="adapter-from-route",
            policy_rule="project_capability_match",
            project_id="github:jketreno/clare",
        )
        with patch.dict("os.environ", {"CLARE2_PROXY_TOKEN": "secret"}), patch(
            "app.proxy.httpx.AsyncClient",
            FakeAsyncClient,
        ), patch("app.proxy.router.get", return_value=route) as get_route, patch(
            "app.proxy.controller.ensure_loaded"
        ) as ensure_loaded:
            response = client.post(
                "/route-from-path/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={"model": "ignored", "messages": []},
            )
        self.assertEqual(response.status_code, 200)
        get_route.assert_called_once_with("route-from-path")
        ensure_loaded.assert_called_once_with("adapter-from-route")
        self.assertEqual(captured["url"], "http://vllm-engine:8001/v1/chat/completions")
        self.assertEqual(captured["model"], "adapter-from-route")

    def test_proxy_path_route_id_takes_precedence_over_header(self):
        app = FastAPI()
        app.include_router(router_api)
        client = TestClient(app)
        with patch.dict("os.environ", {"CLARE2_PROXY_TOKEN": "secret"}), patch(
            "app.proxy.router.get", side_effect=RouteError("unknown route")
        ) as get_route:
            response = client.post(
                "/path-route/v1/chat/completions",
                headers={
                    "Authorization": "Bearer secret",
                    "X-CLARE-Route-ID": "header-route",
                },
                json={"model": "ignored", "messages": []},
            )
        self.assertEqual(response.status_code, 401)
        get_route.assert_called_once_with("path-route")


if __name__ == "__main__":
    unittest.main()
