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
- `127.0.0.1:8005`: unified Image API

There is no host binding for raw vLLM.

## Image API and SAM3

`image-api` is the CPU-only public facade for analysis, segmentation, editing,
inpainting, outpainting, and deterministic transforms. Host clients use
`http://127.0.0.1:8005`; containers such as Auto SAM use
`http://image-api:8000`. Restarting it does not reload model workers.

The optional `sam3` profile runs a headless `sam3-worker`. Its capability RPC
and metrics are private to Docker networks; no Streamlit UI is installed.

SAM3 is gated on Hugging Face. Accept Meta's model terms for the account behind
`secrets/huggingface_token`, then build and start the profile:

```bash
docker compose --profile sam3 build sam3-worker image-api
docker compose --profile sam3 up -d sam3-worker image-api
```

Stop the SAM3 profile before CLARE2 training because both workloads require
substantial unified GPU memory.

SAM3 exports Prometheus metrics on its private monitoring-network port `9092`.
Prometheus scrapes them under the `sam3` job, and Grafana provisions the
SAM3 dashboard for model, inference, GPU memory, and process-health telemetry.

Use `POST /v1/images/analyze` for concept discovery plus segmentation, or
`POST /v1/images/segment` with an explicit JSON prompt list.

## Qwen-Image-Edit

The optional `qwen-image-edit` profile runs a private persistent worker for
`Qwen/Qwen-Image-Edit-2511`. The Image API exposes its capabilities as domain
operations. It shares the same unified-memory GPU as
`vllm-engine`, so `CLARE2_GPU_MEMORY_UTILIZATION` is deliberately kept low
(`0.45` by default — see `.env`) to leave headroom for it. The transformer is
quantized to fp8 at load time with `torchao`'s `Float8WeightOnlyConfig`
(weights fp8-resident, activations bf16); the text encoder, VAE, and tokenizer
load from the same upstream repo in bf16. Measured on this GB10 node in
isolation (vLLM stopped): peak ~43 GiB CUDA memory for one 1024px edit.

A community single-file FP8 checkpoint (`1038lab/Qwen-Image-Edit-2511-FP8`) was
evaluated first and rejected: loading it via `diffusers`' `from_single_file`
with `torch_dtype=torch.bfloat16` silently upcasts the weights to bf16 instead
of keeping them fp8-resident, which OOM'd on this hardware. The `torchao`
quantize-on-load approach used here was verified to actually reduce memory and
to produce numerically correct (non-NaN) output on this GB10's sm_121 GPU,
which is not an officially supported PyTorch/torchao target as of this writing
— re-verify output quality after any base image, diffusers, or torchao upgrade.

**Do not run `qwen-image-edit` concurrently with `vllm-engine` without care.**
On this GB10 node, real available memory under load has repeatedly measured
lower than `(1 - CLARE2_GPU_MEMORY_UTILIZATION) × 121.63 GiB` implies — a first
concurrent attempt OOM'd the container; a second attempt (after a loading-code
fix) triggered a **full host lockup requiring a hard reboot**. Docker `mem_limit`
was investigated as a safety backstop and rejected: it is documented as
unreliable for CUDA/UVM allocations specifically on GB10's unified-memory
architecture (cgroup memory accounting doesn't see UVM allocations, or
`cudaMemGetInfo` misreports the full 128 GB pool as free regardless of the
container's limit). Also do not run it at the same time as `sam3` or
`clare2-train`.

Instead, model loading is split into three instrumented sections (transformer,
text encoder, pipeline assembly). Before each section, the service checks host
`MemAvailable` (a fair proxy for real headroom on unified memory) against that
section's expected requirement plus a safety margin
(`QWEN_IMAGE_EDIT_SAFETY_MARGIN_GIB`, default 4 GiB); if headroom looks short it
aborts loading *before* attempting the risky allocation instead of after. Every
section's outcome — required/available/actual-used memory, CUDA
allocated/reserved, duration, and (for the transformer) whether it loaded from
cache or quantized fresh — is persisted to `/app/state/load_status.json` (named
volume `qwen-image-edit-state`) after each step, so a partial or aborted load
leaves a clear, inspectable record instead of silently retrying or taking the
host down. Per-section size requirements (`QWEN_IMAGE_EDIT_REQUIRED_TRANSFORMER_GIB`,
`..._TEXT_ENCODER_GIB`, `..._PIPELINE_GIB`) default to measured actuals from a
successful run on this node (14/15/2 GiB with margin) — override via env if a
future run's `actual_used_gib` values differ meaningfully.

Quantizing the transformer from bf16 to fp8 takes several minutes on every
startup by default. To avoid repeating that, the service caches the quantized
weights on first run at `QWEN_IMAGE_EDIT_QUANTIZED_TRANSFORMER_PATH` (default
`/root/.cache/huggingface/qwen-image-edit-2511-transformer-fp8`, on the shared
`CLARE2_MODEL_CACHE` volume) via `torchao`'s `save_pretrained(...,
safe_serialization=False)`, which persists real fp8 tensors rather than bf16 +
a re-quantize instruction. Subsequent starts detect the cached checkpoint and
load it directly, skipping quantization compute entirely. Measured on this
node: first run (quantize + save) took 586.6s for the transformer section;
a second run loading from cache took 297.4s — roughly halved, and the load
status records `"transformer_source": "cache"` vs `"quantized_fresh"` so you
can confirm which path a given startup took. Delete the cache directory to
force re-quantization (e.g. after a `diffusers`/`torchao` upgrade you want to
re-validate against).

**First-time setup: do the initial quantize-to-disk run with `vllm-engine`
stopped.** A cold start (no cached fp8 checkpoint yet) briefly holds both the
bf16 source weights and the fp8 result in memory during quantization, on top
of the normal per-section footprint, and takes several minutes longer than a
cached load. Combined with `vllm-engine` already holding its own weights and
KV cache, that peak is exactly the kind of concurrent memory contention
described above that has caused an OOM and, once, a full host lockup on this
node. The in-process `MemAvailable` gate reduces but does not eliminate this
risk — a lockup can happen fast enough that the gate's own abort doesn't help.
Onboarding should always do the first run in isolation:

```bash
# 1. Make sure vllm-engine (and sam3 / clare2-train) are stopped
docker compose stop vllm-engine sam3-worker clare2-train

# 2. Build and start qwen-image-edit alone, and let it finish quantizing +
#    caching to disk (watch for "transformer_source": "quantized_fresh" and
#    then state "ready" in /v1/load-status; can take upward of 15 minutes)
docker compose --profile qwen-image-edit build qwen-image-edit-worker image-api
docker compose --profile qwen-image-edit up -d qwen-image-edit-worker image-api
docker compose exec qwen-image-edit-worker curl -s http://localhost:8006/health/ready

# 3. Once cached, subsequent starts load fp8 weights directly (no
#    bf16+fp8 double-hold), so vllm-engine can be brought back up and run
#    concurrently with qwen-image-edit per the guidance below
docker compose up -d vllm-engine
```

After this one-time step, the cached checkpoint persists on the
`CLARE2_MODEL_CACHE` volume, so it survives container rebuilds/restarts and
this isolation step does not need to be repeated unless the cache is deleted.

Build and start the profile:

```bash
docker compose --profile qwen-image-edit build qwen-image-edit-worker image-api
docker compose --profile qwen-image-edit up -d qwen-image-edit-worker image-api
```

Before starting it alongside a live `vllm-engine`, check real headroom
yourself first (`free -h`, looking at `available`) — don't rely solely on the
in-process gate, since a lockup can happen fast enough that the gate's own
abort doesn't help if the very first section's allocation is what overwhelms
the host. The service loads its model on startup, which can take upward of 15
minutes on a cold Hugging Face cache. Its private `/health/ready` endpoint
reports readiness; load progress remains persisted in the state volume.
Submit an edit through the Image API:

```bash
curl -s http://127.0.0.1:8005/v1/images/edit \
  -F file=@input.png \
  -F prompt='add a small red circle in the center' \
  | python3 -c 'import sys,json,base64; d=json.load(sys.stdin); open("out.png","wb").write(base64.b64decode(d["image_png_base64"]))'
```

`qwen-image-edit` exports Prometheus metrics on its private monitoring-network
port `9093` (job `qwen_image_edit`), following the same pattern as the SAM3
services' `9092`.

### Image API

Alibaba's hosted Qwen-Image-Edit API (see
[the Model Studio docs](https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit-api))
is purely prompt-driven — it has no mask, inpaint, outpaint, crop, or rotate
parameter. This service exposes the same prompt-driven editing plus additional
endpoints built on `diffusers`' `QwenImageEditInpaintPipeline`, which *does*
support a mask, so mask-guided workflows (e.g. `sam3`-selected regions from
`auto-sam`) don't need to be implemented as bespoke composition logic in every
client. `QwenImageEditInpaintPipeline` is constructed once at startup directly
from the edit pipeline's already-loaded scheduler/vae/text_encoder/tokenizer/
processor/transformer, which shares them by reference rather than loading a
second copy. Load status records it as a fourth section (`inpaint_pipeline`).
(Not built via `DiffusionPipeline.from_pipe()`: it
unconditionally ends with a `.to(dtype=...)` cast, which torchao's
fp8-quantized transformer rejects with `ValueError: Casting a quantized model
to a new dtype is unsupported`.)

- `POST /v1/images/edit` — whole-image, prompt-driven edit (text editing, object
  add/remove/move, pose changes, style transfer, detail enhancement). Accepts
  optional `reference_files` (0-2 additional images) for Qwen-Image-Edit-Plus's
  multi-image fusion, up to 3 images total.
- `POST /v1/images/inpaint` — mask-guided region edit. `mask` is a
  `data:image/png;base64,...` string where white pixels are repainted and
  black pixels are preserved — exactly the format returned by
  `/v1/images/segment`, so a SAM mask can be forwarded unmodified. `strength`
  (0.0-1.0, default 1.0) controls how strongly the masked region is
  regenerated.
- `POST /v1/images/outpaint` — canvas expansion ("expand image"). Give a
  `target_width`/`target_height` and an `anchor`
  (`center`/`top`/`bottom`/`left`/`right`/`top-left`/`top-right`/`bottom-left`/`bottom-right`);
  the service pads the canvas, derives the border mask itself (no
  client-side mask painting needed), and inpaints the new border at
  `strength=1.0` per `prompt`.
- `POST /v1/images/transform` — pure Pillow crop and/or rotate, no model inference
  and no GPU time. Crop (`crop_left`/`crop_top`/`crop_width`/`crop_height`)
  is applied before rotate (`rotate_degrees`, `expand_canvas`). Alibaba's API
  and the local pipelines have no geometric-transform primitive, so this is
  a deterministic local operation clients compose alongside the model-backed
  endpoints rather than paying for a diffusion pass on a task that doesn't
  need one.

All endpoints return `{"width", "height", "image_png_base64"}`. Example
inpaint call using a SAM3 mask:

```bash
curl -s http://127.0.0.1:8005/v1/images/inpaint \
  -F file=@input.png \
  -F mask="$(python3 -c 'import json,sys; print(json.load(open("segment.json"))["segments"][0]["mask"])')" \
  -F prompt='a black cat sitting' \
  | python3 -c 'import sys,json,base64; d=json.load(sys.stdin); open("out.png","wb").write(base64.b64decode(d["image_png_base64"]))'
```

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

`CLARE2_GPU_MEMORY_UTILIZATION` in `.env` governs how much of the node's 128 GB
unified memory (121.63 GiB usable) `vllm-engine` reserves; the remainder is
shared with either `sam3` or `qwen-image-edit` when those services are running.
Lower it further if a new GPU workload doesn't fit; raise it back toward `0.70`
if those optional profiles are retired.

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
