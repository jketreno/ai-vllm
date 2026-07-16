#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT=/corpus/training
STATE=/corpus/meta/lifecycle.json
REGISTRY=/models/adapters/registry.json
MODEL=${CLARE2_TRAIN_MODEL:-Qwen/Qwen3.6-27B-FP8}
REVISION=${CLARE2_TRAIN_REVISION:?CLARE2_TRAIN_REVISION must pin the training base revision}
MODEL_CACHE=${HF_HUB_CACHE:-${HF_HOME:-/root/.cache/huggingface}/hub}
MIN_TRAIN_RECORDS=${CLARE2_MIN_TRAIN_RECORDS:-8}

[[ -s "$STATE" ]] || { echo "lifecycle state is missing" >&2; exit 1; }
RUN_ID=$(python3 -c 'import json; print(json.load(open("'"$STATE"'"))["run_id"])')

# Discover per-project corpora: training/{project}/current.jsonl
mapfile -t PROJECT_DIRS < <(find "$TRAINING_ROOT" -mindepth 2 -maxdepth 2 -name current.jsonl -size +0c 2>/dev/null | sort)

if [[ ${#PROJECT_DIRS[@]} -eq 0 ]]; then
  echo "No non-empty per-project corpus files found under $TRAINING_ROOT" >&2
  exit 1
fi

# True if the project's most recent adapter for this exact corpus hash
# already produced a durable outcome (approved, loaded, retired, or still
# training) — not just any hash match. A rejected, failed, or unevaluated
# candidate adapter for the same hash does not block a retry, since an
# interrupted callback can leave stale candidates behind.
already_trained_for_hash() {
  local project="$1" corpus_hash="$2"
  [[ -s "$REGISTRY" ]] || return 1
  python3 - "$REGISTRY" "$project" "$corpus_hash" <<'PY'
import json
import sys

registry_path, project, corpus_hash = sys.argv[1:4]
BLOCKING_STATUSES = {"training", "approved", "loaded", "retired"}

with open(registry_path, encoding="utf-8") as fh:
    document = json.load(fh)

matches = [
    adapter
    for adapter in document.get("adapters", {}).values()
    if adapter.get("project_scope") == project and adapter.get("corpus_hash") == corpus_hash
]
latest = max(matches, key=lambda a: a.get("created_at", ""), default=None)
sys.exit(0 if latest and latest.get("status") in BLOCKING_STATUSES else 1)
PY
}

_send_batch_callback() {
  local payload
  payload=$(python3 - "$RUN_ID" "${TRAINED_META_PATHS[@]}" <<'PY'
import json, sys
run_id, meta_paths = sys.argv[1], sys.argv[2:]
adapters = []
for meta_path in meta_paths:
    meta = json.load(open(meta_path))
    adapters.append({
        "adapter_id": meta["adapter_id"],
        "mlflow_run_id": meta["mlflow_run_id"],
        "loss": meta["final_loss"],
        "epoch_losses": [entry["loss"] for entry in meta["loss_history"]],
    })
print(json.dumps({"run_id": run_id, "adapters": adapters}, separators=(",", ":")))
PY
  )
  local timestamp signature
  timestamp=$(date +%s)
  signature=$(printf '%s.%s' "$timestamp" "$payload" |
    openssl dgst -sha256 -hmac "$(cat /run/secrets/clare2_callback_secret)" -hex |
    awk '{print $2}')
  curl --fail --silent --show-error \
    -H "Content-Type: application/json" \
    -H "X-CLARE-Timestamp: ${timestamp}" \
    -H "X-CLARE-Signature: ${signature}" \
    --data "$payload" \
    "${CLARE2_PIPELINE_URL:-http://clare2-policy:8000}/training/batch-done"
}

_send_skipped_callback() {
  local payload timestamp signature
  payload=$(printf '{"run_id":"%s"}' "$RUN_ID")
  timestamp=$(date +%s)
  signature=$(printf '%s.%s' "$timestamp" "$payload" |
    openssl dgst -sha256 -hmac "$(cat /run/secrets/clare2_callback_secret)" -hex |
    awk '{print $2}')
  curl --fail --silent --show-error \
    -H "Content-Type: application/json" \
    -H "X-CLARE-Timestamp: ${timestamp}" \
    -H "X-CLARE-Signature: ${signature}" \
    --data "$payload" \
    "${CLARE2_PIPELINE_URL:-http://clare2-policy:8000}/training/skipped"
}

TRAINED_ANY=0
TRAINED_META_PATHS=()

for corpus_file in "${PROJECT_DIRS[@]}"; do
  # Extract project name from path: training/{project}/current.jsonl
  project=$(basename "$(dirname "$corpus_file")")
  safe_project=$(printf '%s' "$project" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')
  record_count=$(wc -l < "$corpus_file")
  if (( record_count < MIN_TRAIN_RECORDS )); then
    echo "Project $project has only $record_count records; need at least $MIN_TRAIN_RECORDS — skipping" >&2
    continue
  fi
  full_corpus_hash=$(sha256sum "$corpus_file" | cut -d' ' -f1)
  corpus_hash=${full_corpus_hash:0:12}

  if already_trained_for_hash "$project" "$full_corpus_hash"; then
    echo "No new content for project: $project (corpus unchanged since last trained adapter) — skipping" >&2
    continue
  fi

  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  adapter_id="clare-${safe_project}-${stamp}-${corpus_hash}"
  adapter_out="/models/adapters/${adapter_id}"
  model_path="$MODEL"
  model_cache_dir="${MODEL_CACHE}/models--${MODEL//\//--}/snapshots/${REVISION}"
  if [[ -d "$model_cache_dir" ]]; then
    model_path="$model_cache_dir"
  fi

  echo "Training adapter for project: $project (corpus: $corpus_file, model: $model_path)" >&2

  python /app/train.py \
    --model_name "$model_path" \
    --base_model_id "$MODEL" \
    --revision "$REVISION" \
    --train_file "$corpus_file" \
    --output_dir "$adapter_out" \
    --adapter_id "$adapter_id" \
    --run_id "$RUN_ID" \
    --project_id "$project" \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --max_seq_length 2048

  TRAINED_ANY=1
  TRAINED_META_PATHS+=("$adapter_out/training_meta.json")

  if [[ "${CLARE2_TRAIN_SKIP_CALLBACK:-0}" == "1" ]]; then
    echo "Skipping training callback for dream-mode run: $adapter_id" >&2
  fi
done

if [[ "$TRAINED_ANY" -eq 1 && "${CLARE2_TRAIN_SKIP_CALLBACK:-0}" != "1" ]]; then
  _send_batch_callback
fi

if [[ "$TRAINED_ANY" -eq 0 ]]; then
  echo "No project had new corpus content since its last trained adapter — nothing to train" >&2
  if [[ "${CLARE2_TRAIN_SKIP_CALLBACK:-0}" == "1" ]]; then
    echo "Skipping training-skipped callback for dream-mode run: $RUN_ID" >&2
  else
    _send_skipped_callback
  fi
fi
