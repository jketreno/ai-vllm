# ai-vllm with CLARE₂

This stack serves `Qwen/Qwen3.5-35B-A3B-FP8` through an authenticated CLARE₂
policy proxy. Raw vLLM and its runtime LoRA management endpoints are reachable
only on the private `inference` Docker network. The same local Qwen3.5 service
performs distillation, summarization, evaluation, and agent inference.

## Automated Setup

Run the Dockerized bootstrap. It creates secrets, resolves and downloads pinned
models into a repository bind mount, computes model fingerprints, builds the
services, and starts the core stack:

```bash
HF_TOKEN='<Hugging Face token>' \
CLARE2_PROJECT_MAP='{"clare":"github:jketreno/clare"}' \
CLARE2_PROJECT_ID='github:jketreno/clare' \
./setup-clare2.sh --capture-project /path/to/clare
```

Models are stored under `models/huggingface/` by default and bind-mounted into
both vLLM and the trainer. Set `CLARE2_MODEL_CACHE` to use another host
directory. Use `./setup-clare2.sh --no-start` to prepare without startup.
No Python packages are installed on the host.

Inference and MCP bind to localhost by default. Set
`CLARE2_BIND_ADDRESS=0.0.0.0` during setup for authenticated LAN access, and
restrict ports `8000` and `8002` with the host firewall.

Secrets remain under `secrets/` with mode `0600`; `.env` contains no
credentials.

Launch Codex through the capture wrapper so its lifecycle hooks and CLARE
verification events write to the same session:

```bash
./clare2/scripts/clare2-agent.sh codex /path/to/clare
```

Codex's private transcript directory is not used as an ingestion API. The
project hooks normalize prompts and final responses directly into
`corpus/sessions/YYYY/MM/DD/<session-id>.jsonl`.

Public bindings:

- `127.0.0.1:8000`: authenticated inference policy proxy and operator API
- `127.0.0.1:8002`: CLARE Temper MCP server
- `127.0.0.1:5000`: MLflow experiment tracking UI
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

## MLflow Tracking

The private `mlflow` service stores run metadata in `mlflow/data/mlflow.db` and
artifacts under `mlflow/artifacts/`. The UI is bound only to localhost:

```text
http://127.0.0.1:5000
```

Every QLoRA run logs its immutable adapter and lifecycle IDs, project, exact
base revision, corpus and dependency hashes, hyperparameters, skipped-record
counts, per-step/final loss, duration, metadata, and generated adapter files.
The raw training corpus is not uploaded to MLflow.

## Verification

```bash
PYTHONPATH=clare2/pipeline python -m unittest discover -s clare2/pipeline/tests
docker compose config --quiet
```

GPU acceptance should use a small Qwen-compatible base and two distinguishable
LoRA fixtures before enabling the nightly schedule. Confirm each route selects
its fixture, unload/reload works, restart reconciliation succeeds, and requests
without a route use the base model.
