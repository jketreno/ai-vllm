# ai-vllm with [CLARE₂](https://github.com/jketreno/clare)

This stack serves `Qwen/Qwen3.6-27B-FP8` through an authenticated
[CLARE₂](https://github.com/jketreno/clare) policy proxy. Raw vLLM and its runtime LoRA management endpoints are reachable
only on the private `inference` Docker network. The same local Qwen3.6 service
performs distillation, summarization, evaluation, agent inference, and spam
classification. The authenticated spam-classification API uses vLLM
JSON-schema constrained decoding without loading a second GPU model.

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

Launch Codex through the capture wrapper so its lifecycle hooks and
[CLARE](https://github.com/jketreno/clare) verification events write to the
same session:

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
- `127.0.0.1:8003`: authenticated spam-classification API
- `0.0.0.0:8080`: Open WebUI
- `127.0.0.1:8501`: SAM3 Auto Annotator (optional `sam3` profile)

There is no host binding for raw vLLM.

## SAM3 Annotation

The optional `sam3` profile runs the pinned
[SAM3 Auto Annotator](https://github.com/data-with-shobhit/sam3-auto-annotator)
Streamlit frontend and GPU-backed SAM3 inference in one container. Model weights
share the existing Hugging Face cache; projects, trained models, and logs use
named volumes. Keep the UI's annotation worker count at one on the GB10 to avoid
loading concurrent SAM3 model copies into unified memory.

SAM3 is gated on Hugging Face. Accept Meta's model terms for the account behind
`secrets/huggingface_token`, then build and start the profile:

```bash
docker compose --profile sam3 build sam3-annotator
docker compose --profile sam3 up -d sam3-annotator
```

Open `https://ai.ketrenos.com/sam3/`. The direct loopback endpoint remains
available at `http://127.0.0.1:8501/sam3/`. Persistent
projects, exports, database state, and caches use named volumes. Stop the SAM3
profile before CLARE2 training because both workloads require substantial
unified GPU memory.

SAM3 exports Prometheus metrics on its private monitoring-network port `9092`.
Prometheus scrapes them under the `sam3` job, and Grafana provisions the
`SAM3 Annotator` dashboard for model, inference, data-pipeline, training, GPU
memory, and process-health telemetry.

The same container exposes an internal API at `http://sam3-annotator:8004` on
the frontend network. `POST /v1/segment` accepts a multipart image, a JSON
`prompts` array, and an optional `threshold`; it returns mask metadata and a
base64 PNG overlay. Each segment's `mask` is a lossless, full-resolution
monochrome PNG data URI with background pixels set to 0 and matched pixels set
to 255, suitable for direct use as an overlay or alpha mask. API and Streamlit
requests share one model instance and are serialized to prevent duplicate GPU
allocations.

For interactive exploration of text masks, point prompts, and video tracking,
start the separately pinned SAM3-Demo frontend:

```bash
docker compose --profile sam3-demo up -d --build sam3-demo
```

Open `https://ai.ketrenos.com/sam3-demo/`. The demo loads only the model variant
needed by the active mode and releases the prior variant when switching modes,
avoiding three simultaneous SAM3 copies on the GPU.

## Spam Classification

The classifier accepts parsed email data rather than raw RFC 822 input. This
keeps MIME parsing and mail mutation out of the model-facing service:

```bash
SPAM_TOKEN=$(<secrets/spam_api_token)
curl --fail --silent \
  -H "Authorization: Bearer ${SPAM_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "envelope_from": "billing@lookalike.example",
    "envelope_to": ["user@example.com"],
    "headers": [
      {"name": "From", "value": "Example Billing <billing@lookalike.example>"},
      {"name": "Authentication-Results", "value": "spf=fail; dkim=fail"}
    ],
    "subject": "Your account will be closed today",
    "text_body": "Verify your password immediately at the supplied link."
  }' \
  http://127.0.0.1:8003/v1/classify
```

The response contract is:

```json
{
  "schema_version": "1",
  "classification": "SPAM",
  "spam_score": 0.97,
  "threshold": 0.8,
  "reasons": ["Urgent credential-verification request", "SPF and DKIM failed"],
  "model": "Qwen/Qwen3.6-27B-FP8"
}
```

`classification` is produced by the model, not inferred from the score alone.
The constrained model schema asks for evidence first, then classification, then
`spam_score`; if the numeric score contradicts the classification, the API
normalizes the score to the matching side of the threshold. The score is still
the model's estimate, not a calibrated probability. Tune
`SPAM_THRESHOLD` against a representative labeled mailbox before allowing the
mail-server integration to modify messages. A model or API failure returns an
error and must not be interpreted as spam.

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

Qwen thinking is enabled but bounded by default for chat-completion callers.
See `THINKING-CONFIG.md` for the caller-configurable `extra_body` fields,
LangGraph examples, and no-thinking request form.

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

## Monitoring

Prometheus scrapes `clare2-policy`, `vllm-engine`, and `nvidia-exporter` by
container name automatically. The `node` job reaches the host-networked
node-exporter through Docker's `host-gateway` mapping and supplies the
`System Resources` dashboard. cAdvisor supplies per-service container
working-set memory for that dashboard's allocation pie chart; model weights
are attributed to the service hosting the model rather than individual tensors.
The model-memory exporter reads vLLM's runtime profiler through the restricted
Docker proxy and supplies a separate accelerator-memory pie for model weights,
KV-cache capacity, CUDA graphs, and SAM3's live PyTorch reservation. Keeping
the charts separate avoids double counting on coherent unified-memory systems
such as NVIDIA GB10.

The `comfyui_flux_arc` job scrapes the FLUX ComfyUI metrics sidecar on
`battle-linux.ketrenos.com:9190`. Grafana provisions the `ComfyUI FLUX Arc`
dashboard with service health, queue depth, prompt duration, retained-history
outcomes, host memory, and Intel XPU memory. The sidecar reads ComfyUI's
read-only status APIs and does not proxy generation requests.

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
