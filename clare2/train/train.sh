#!/usr/bin/env bash
set -euo pipefail

CORPUS=/corpus/training/current.jsonl
STATE=/corpus/meta/lifecycle.json
MODEL=${CLARE2_TRAIN_MODEL:-Qwen/Qwen3.5-35B-A3B}
REVISION=${CLARE2_TRAIN_REVISION:?CLARE2_TRAIN_REVISION must pin the non-FP8 revision}
PROJECT=${CLARE2_PROJECT_ID:-global}

[[ -s "$CORPUS" ]] || { echo "training corpus is missing or empty" >&2; exit 1; }
[[ -s "$STATE" ]] || { echo "lifecycle state is missing" >&2; exit 1; }

RUN_ID=$(python3 -c 'import json; print(json.load(open("'"$STATE"'"))["run_id"])')
CORPUS_HASH=$(sha256sum "$CORPUS" | cut -c1-12)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SAFE_PROJECT=$(printf '%s' "$PROJECT" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')
ADAPTER_ID="clare-${SAFE_PROJECT}-${STAMP}-${CORPUS_HASH}"
ADAPTER_OUT="/models/adapters/${ADAPTER_ID}"

python /app/train.py \
  --model_name "$MODEL" \
  --revision "$REVISION" \
  --train_file "$CORPUS" \
  --output_dir "$ADAPTER_OUT" \
  --adapter_id "$ADAPTER_ID" \
  --run_id "$RUN_ID" \
  --project_id "$PROJECT" \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --max_seq_length 2048

PAYLOAD=$(python3 - "$ADAPTER_OUT/training_meta.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
print(json.dumps({
    "adapter_id": meta["adapter_id"],
    "run_id": meta["run_id"],
    "loss": meta["final_loss"],
    "epoch_losses": [entry["loss"] for entry in meta["loss_history"]],
}, separators=(",", ":")))
PY
)
TIMESTAMP=$(date +%s)
SIGNATURE=$(printf '%s.%s' "$TIMESTAMP" "$PAYLOAD" |
  openssl dgst -sha256 -hmac "$(cat /run/secrets/clare2_callback_secret)" -hex |
  awk '{print $2}')

curl --fail --silent --show-error \
  -H "Content-Type: application/json" \
  -H "X-CLARE-Timestamp: ${TIMESTAMP}" \
  -H "X-CLARE-Signature: ${SIGNATURE}" \
  --data "$PAYLOAD" \
  "${CLARE2_PIPELINE_URL:-http://clare2-policy:8000}/training/done"
