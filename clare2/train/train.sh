#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT=/corpus/training
STATE=/corpus/meta/lifecycle.json
MODEL=${CLARE2_TRAIN_MODEL:-Qwen/Qwen3.5-35B-A3B}
REVISION=${CLARE2_TRAIN_REVISION:?CLARE2_TRAIN_REVISION must pin the non-FP8 revision}

[[ -s "$STATE" ]] || { echo "lifecycle state is missing" >&2; exit 1; }
RUN_ID=$(python3 -c 'import json; print(json.load(open("'"$STATE"'"))["run_id"])')

# Discover per-project corpora: training/{project}/current.jsonl
mapfile -t PROJECT_DIRS < <(find "$TRAINING_ROOT" -mindepth 2 -maxdepth 2 -name current.jsonl -size +0c 2>/dev/null | sort)

if [[ ${#PROJECT_DIRS[@]} -eq 0 ]]; then
  echo "No non-empty per-project corpus files found under $TRAINING_ROOT" >&2
  exit 1
fi

_send_callback() {
  local meta_path="$1"
  local payload
  payload=$(python3 - "$meta_path" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
print(json.dumps({
    "adapter_id": meta["adapter_id"],
    "run_id": meta["run_id"],
    "mlflow_run_id": meta["mlflow_run_id"],
    "loss": meta["final_loss"],
    "epoch_losses": [entry["loss"] for entry in meta["loss_history"]],
}, separators=(",", ":")))
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
    "${CLARE2_PIPELINE_URL:-http://clare2-policy:8000}/training/done"
}

for corpus_file in "${PROJECT_DIRS[@]}"; do
  # Extract project name from path: training/{project}/current.jsonl
  project=$(basename "$(dirname "$corpus_file")")
  safe_project=$(printf '%s' "$project" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')
  corpus_hash=$(sha256sum "$corpus_file" | cut -c1-12)
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  adapter_id="clare-${safe_project}-${stamp}-${corpus_hash}"
  adapter_out="/models/adapters/${adapter_id}"

  echo "Training adapter for project: $project (corpus: $corpus_file)" >&2

  python /app/train.py \
    --model_name "$MODEL" \
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

  _send_callback "$adapter_out/training_meta.json"
done
