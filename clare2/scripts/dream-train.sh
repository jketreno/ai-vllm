#!/usr/bin/env bash
set -euo pipefail

KEEP_SERVICES=(
  spam-vllm
  spam-classifier
  prometheus
  grafana
  node-exporter
  nvidia-exporter
  nginx
  docker-socket-proxy
)

STOP_SERVICES=(
  vllm-engine
  clare2-policy
  clare2-mcp
  open-webui
  ollama
  qdrant
  redis
  mlflow
)

WAKE_SERVICES=(
  redis
  mlflow
  vllm-engine
  clare2-policy
  clare2-mcp
  open-webui
  ollama
  qdrant
)

DRY_RUN=0
SUMMARY_DIR=${CLARE2_DREAM_SUMMARY_DIR:-./logs/clare2/dream}
POLICY_URL=${CLARE2_POLICY_URL:-http://127.0.0.1:${CLARE2_PROXY_PORT:-8000}}
ASSEMBLE_WAIT_SECONDS=${CLARE2_DREAM_ASSEMBLE_WAIT_SECONDS:-10}
WAKE_TIMEOUT_SECONDS=${CLARE2_DREAM_WAKE_TIMEOUT_SECONDS:-900}

usage() {
  cat <<'EOF'
Usage: clare2/scripts/dream-train.sh [--dry-run]

Stops normal AI services, runs clare2-train offline, wakes CLARE2, then registers
and evaluates produced adapters. The dry run prints the exact service plan only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

json_array() {
  python3 - "$@" <<'PY'
import json, sys
print(json.dumps(sys.argv[1:]))
PY
}

memory_snapshot() {
  local stage="$1"
  python3 - "$stage" <<'PY'
import json, pathlib, subprocess, sys, time
stage = sys.argv[1]
snapshot = {"stage": stage, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            snapshot[key] = value.strip()
except OSError:
    pass
try:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    snapshot["gpus"] = [line.strip() for line in output.splitlines() if line.strip()]
except Exception:
    snapshot["gpus"] = []
print(json.dumps(snapshot, sort_keys=True))
PY
}

adapter_meta_paths() {
  local run_id="$1"
  python3 - "$run_id" <<'PY'
import json, pathlib, sys
run_id = sys.argv[1]
root = pathlib.Path("models/adapters")
if not root.exists():
    raise SystemExit(0)
for meta in sorted(root.glob("*/training_meta.json")):
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if data.get("run_id") == run_id:
        print(meta)
PY
}

callback_payload() {
  local meta_path="$1"
  python3 - "$meta_path" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "adapter_id": meta["adapter_id"],
    "run_id": meta["run_id"],
    "mlflow_run_id": meta.get("mlflow_run_id"),
    "loss": meta.get("final_loss"),
    "epoch_losses": [entry["loss"] for entry in meta.get("loss_history", [])],
}, separators=(",", ":")))
PY
}

sign_and_post_callback() {
  local payload="$1"
  local token_file="${CLARE2_CALLBACK_SECRET_FILE:-./secrets/clare2_callback_secret}"
  local timestamp signature
  timestamp=$(date +%s)
  signature=$(printf '%s.%s' "$timestamp" "$payload" |
    openssl dgst -sha256 -hmac "$(tr -d '\r\n' < "$token_file")" -hex |
    awk '{print $2}')
  curl --fail --silent --show-error \
    -H "Content-Type: application/json" \
    -H "X-CLARE-Timestamp: ${timestamp}" \
    -H "X-CLARE-Signature: ${signature}" \
    --data "$payload" \
    "${POLICY_URL}/training/done" >/dev/null
}

wait_for_policy() {
  local deadline=$((SECONDS + WAKE_TIMEOUT_SECONDS))
  until curl -fsS "${POLICY_URL}/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for CLARE2 policy at ${POLICY_URL}" >&2
      return 1
    fi
    sleep 5
  done
}

wait_for_adapter_outcome() {
  local adapter_id="$1"
  local token_file="${CLARE2_OPERATOR_TOKEN_FILE:-./secrets/clare2_operator_token}"
  local token deadline status
  token=$(tr -d '\r\n' < "$token_file")
  deadline=$((SECONDS + WAKE_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    status=$(curl -fsS -H "Authorization: Bearer ${token}" "${POLICY_URL}/operator/status" || true)
    if python3 - "$status" "$adapter_id" <<'PY'
import json, sys
try:
    state = json.loads(sys.argv[1]).get("lifecycle", {})
except json.JSONDecodeError:
    raise SystemExit(1)
adapter_id = sys.argv[2]
if state.get("candidate_id") == adapter_id and state.get("phase") in {"idle", "failed"}:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 10
  done
  echo "Timed out waiting for adapter outcome: ${adapter_id}" >&2
  return 1
}

if [[ "$DRY_RUN" == "1" ]]; then
  python3 - <<PY
import json
print(json.dumps({
    "dry_run": True,
    "keep_services": $(json_array "${KEEP_SERVICES[@]}"),
    "stop_services": $(json_array "${STOP_SERVICES[@]}"),
    "wake_services": $(json_array "${WAKE_SERVICES[@]}"),
    "training_command": "docker compose --profile training run --no-deps --rm clare2-train",
}, indent=2, sort_keys=True))
PY
  exit 0
fi

[[ -n "${CLARE2_CORPUS_ROOT:-}" ]] || {
  echo "CLARE2_CORPUS_ROOT must point to the shared CLARE2 corpus directory" >&2
  exit 1
}

OPERATOR_TOKEN=$(tr -d '\r\n' < "${CLARE2_OPERATOR_TOKEN_FILE:-./secrets/clare2_operator_token}")
RUN_ID=${CLARE2_RUN_ID:-run-$(date -u +%Y%m%dT%H%M%SZ)-dream}
SUMMARY_PATH="${SUMMARY_DIR}/${RUN_ID}.json"
mkdir -p "$SUMMARY_DIR"

SNAPSHOTS=()
SNAPSHOTS+=("$(memory_snapshot before_sleep)")

curl -fsS -X POST \
  -H "Authorization: Bearer ${OPERATOR_TOKEN}" \
  "${POLICY_URL}/corpus/assemble" >/dev/null
sleep "$ASSEMBLE_WAIT_SECONDS"

curl -fsS -X POST \
  -H "Authorization: Bearer ${OPERATOR_TOKEN}" \
  -H "Content-Type: application/json" \
  --data "{\"run_id\":\"${RUN_ID}\"}" \
  "${POLICY_URL}/operator/training/dream/start" >/dev/null

docker compose up -d "${KEEP_SERVICES[@]}"
docker compose stop "${STOP_SERVICES[@]}"
SNAPSHOTS+=("$(memory_snapshot before_training)")

docker compose --profile training run --no-deps --rm \
  -e CLARE2_TRAIN_SKIP_CALLBACK=1 \
  -e CLARE2_TRAIN_MLFLOW_DISABLED=1 \
  clare2-train

SNAPSHOTS+=("$(memory_snapshot after_training)")

docker compose up -d "${WAKE_SERVICES[@]}"
wait_for_policy
SNAPSHOTS+=("$(memory_snapshot after_wake)")

mapfile -t META_PATHS < <(adapter_meta_paths "$RUN_ID")
if [[ ${#META_PATHS[@]} -eq 0 ]]; then
  echo "No adapter metadata found for run ${RUN_ID}" >&2
  exit 1
fi

ADAPTER_IDS=()
OUTCOMES=()
for meta_path in "${META_PATHS[@]}"; do
  payload=$(callback_payload "$meta_path")
  adapter_id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["adapter_id"])' "$payload")
  ADAPTER_IDS+=("$adapter_id")
  sign_and_post_callback "$payload"
  wait_for_adapter_outcome "$adapter_id"
  OUTCOMES+=("$(curl -fsS -H "Authorization: Bearer ${OPERATOR_TOKEN}" "${POLICY_URL}/operator/status")")
done

python3 - "$SUMMARY_PATH" "$RUN_ID" "$(json_array "${STOP_SERVICES[@]}")" \
  "$(json_array "${KEEP_SERVICES[@]}")" "$(json_array "${WAKE_SERVICES[@]}")" \
  "$(json_array "${ADAPTER_IDS[@]}")" "${SNAPSHOTS[@]}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
snapshots = [json.loads(item) for item in sys.argv[7:]]
summary = {
    "run_id": sys.argv[2],
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "stopped_services": json.loads(sys.argv[3]),
    "kept_services": json.loads(sys.argv[4]),
    "started_services": json.loads(sys.argv[5]),
    "adapter_ids": json.loads(sys.argv[6]),
    "memory_snapshots": snapshots,
}
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
