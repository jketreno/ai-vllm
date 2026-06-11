# ai-vllm with CLARE₂

This stack serves `Qwen/Qwen3.5-35B-A3B-FP8` through an authenticated CLARE₂
policy proxy. Raw vLLM and its runtime LoRA management endpoints are reachable
only on the private `inference` Docker network.

## Security Setup

Copy `.env.example` to `.env`, then set the exact Hugging Face revision and
base/tokenizer hashes. Create these files with mode `0600`:

```text
secrets/anthropic_api_key
secrets/huggingface_token
secrets/ldap_app_password
secrets/clare2_proxy_token
secrets/clare2_operator_token
secrets/clare2_callback_secret
```

Generate the three CLARE₂ tokens independently with at least 32 random bytes.
Credentials formerly stored in `.env` must be rotated at their providers.
This repository has no tracked `.env` history; verify that remains true with:

```bash
git log --all -- .env
```

## Start

The trainer container must exist in a stopped state so the restricted Docker
proxy only needs container inspect/start/stop access:

```bash
docker compose --profile training build
docker compose --profile training create clare2-train
docker compose up -d
```

Public bindings:

- `127.0.0.1:8000`: authenticated OpenAI-compatible policy proxy and operator API
- `127.0.0.1:8002`: CLARE Temper MCP server
- `127.0.0.1:9091`: Prometheus metrics
- `0.0.0.0:8080`: Open WebUI

There is no host binding for raw vLLM.

## Agent Routing

Agents call `clare_temper_route(project, task_kind, capabilities)` and retain
the returned opaque route ID for the session. Requests send:

```text
Authorization: Bearer <clare2_proxy_token>
X-CLARE-Route-ID: <opaque route id>
```

The proxy ignores the client's `model`, loads the pinned approved adapter when
needed, and forwards the immutable adapter ID upstream. Missing route context
uses the pinned base model. Management routes are never proxied.

Policy order:

1. Approved project adapter matching all requested capabilities.
2. Approved global adapter matching all requested capabilities.
3. Base Qwen3.5 model.

## Adapter Registry

`models/adapters/registry.json` is the source of truth. It records the exact
base fingerprint, immutable adapter metadata, lifecycle state, evaluation, and
the `current`/`rollback` aliases. Adapter directories must have the same name
as their immutable ID and may not be symlinks.

The service creates the initial registry from `.env`; replace every placeholder
fingerprint before training. `registry.example.json` documents the schema.

## Nightly Lifecycle

The persisted, single-run state machine performs:

```text
maintenance -> drain -> stop vLLM -> train -> restart base -> reconcile
-> load current and candidate -> deterministic comparison -> promote/reject
-> resume
```

Promotion requires all mandatory probes, pass rate `>= 0.90`, and no category
regression. Failure restarts the prior approved adapter and preserves the
candidate directory.

Operator calls require `Authorization: Bearer <clare2_operator_token>`:

```text
GET  /operator/adapters
GET  /operator/status
POST /operator/promote/{adapter_id}
POST /operator/rollback
POST /operator/maintenance/{enter|exit}
```

Training callbacks use a timestamped HMAC and are idempotent.

## Verification

```bash
PYTHONPATH=clare2/pipeline python -m unittest discover -s clare2/pipeline/tests
docker compose config --quiet
```

GPU acceptance should use a small Qwen-compatible base and two distinguishable
LoRA fixtures before enabling the nightly schedule. Confirm each route selects
its fixture, unload/reload works, restart reconciliation succeeds, and requests
without a route use the base model.
