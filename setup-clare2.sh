#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE="${ROOT}/.env"
SECRETS_DIR="${ROOT}/secrets"
MODEL_CACHE="${CLARE2_MODEL_CACHE:-${ROOT}/models/huggingface}"
SETUP_IMAGE="${CLARE2_SETUP_IMAGE:-clare2-model-setup:local}"
START_SERVICES=true
CAPTURE_PROJECTS=()

usage() {
  cat <<'EOF'
Usage: ./setup-clare2.sh [--no-start] [--capture-project PATH]...

Environment:
  HF_TOKEN                 Hugging Face access token (prompted when interactive)
  CLARE2_MODEL_CACHE       Host model cache (default: ./models/huggingface)
  CLARE2_BIND_ADDRESS      Inference/MCP host bind (default: 127.0.0.1)
  CLARE2_INFERENCE_MODEL   Serving model repository
  CLARE2_TRAIN_MODEL       Training model repository
  CLARE2_PROJECT_MAP       Canonical repository map JSON
  CLARE2_PROJECT_ID        Adapter project scope
  CLARE2_DOCKER_GID        Docker socket group id (default: stat /var/run/docker.sock)
EOF
}

while (($#)); do
  case "$1" in
    --no-start) START_SERVICES=false ;;
    --capture-project)
      [[ $# -ge 2 ]] || {
        echo "--capture-project requires a path" >&2
        exit 2
      }
      CAPTURE_PROJECTS+=("$2")
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
  shift
done

for command in docker openssl curl jq flock; do
  command -v "$command" >/dev/null || {
    echo "Required command not found: $command" >&2
    exit 1
  }
done
docker compose version >/dev/null
docker info >/dev/null

cd "$ROOT"
[[ -f "$ENV_FILE" ]] || cp .env.example "$ENV_FILE"
mkdir -p "$SECRETS_DIR" "$MODEL_CACHE" "$ROOT/mlflow/data" "$ROOT/mlflow/artifacts"
chmod 700 "$SECRETS_DIR"

read_env() {
  local key=$1
  local fallback=$2
  local value
  value=$(sed -n "s/^${key}=//p" "$ENV_FILE" | tail -1)
  printf '%s' "${value:-$fallback}"
}

write_env() {
  local key=$1
  local value=$2
  local temporary
  temporary=$(mktemp "${ENV_FILE}.XXXXXX")
  awk -v key="$key" -v value="$value" '
    BEGIN { replaced = 0 }
    index($0, key "=") == 1 {
      if (!replaced) print key "=" value
      replaced = 1
      next
    }
    { print }
    END { if (!replaced) print key "=" value }
  ' "$ENV_FILE" >"$temporary"
  chmod --reference="$ENV_FILE" "$temporary"
  mv "$temporary" "$ENV_FILE"
}

secret() {
  local name=$1
  local path="${SECRETS_DIR}/${name}"
  [[ -s "$path" ]] || openssl rand -hex 32 >"$path"
  chmod 600 "$path"
}

HF_TOKEN_FILE="${SECRETS_DIR}/huggingface_token"
if [[ -n "${HF_TOKEN:-}" ]]; then
  printf '%s' "$HF_TOKEN" >"$HF_TOKEN_FILE"
elif [[ ! -s "$HF_TOKEN_FILE" ]]; then
  if [[ ! -t 0 ]]; then
    echo "Set HF_TOKEN for non-interactive setup" >&2
    exit 1
  fi
  read -r -s -p "Hugging Face token: " HF_TOKEN
  printf '\n'
  printf '%s' "$HF_TOKEN" >"$HF_TOKEN_FILE"
fi
chmod 600 "$HF_TOKEN_FILE"

secret clare2_proxy_token
secret clare2_mcp_token
secret clare2_operator_token
secret clare2_callback_secret
secret spam_api_token
secret grafana_admin_password
[[ -e "${SECRETS_DIR}/ldap_app_password" ]] ||
  printf '%s' "disabled" >"${SECRETS_DIR}/ldap_app_password"
chmod 600 "${SECRETS_DIR}/ldap_app_password"

for project in "${CAPTURE_PROJECTS[@]}"; do
  "${ROOT}/clare2/scripts/clare2-install-hooks.sh" "$project"
done

INFERENCE_MODEL="${CLARE2_INFERENCE_MODEL:-$(read_env CLARE2_INFERENCE_MODEL Qwen/Qwen3.6-27B-FP8)}"
TRAIN_MODEL="${CLARE2_TRAIN_MODEL:-$(read_env CLARE2_TRAIN_MODEL Qwen/Qwen3.6-27B-FP8)}"
INFERENCE_REVISION="${CLARE2_INFERENCE_REVISION:-$(read_env CLARE2_INFERENCE_REVISION "")}"
TRAIN_REVISION="${CLARE2_TRAIN_REVISION:-$(read_env CLARE2_TRAIN_REVISION "")}"
MLFLOW_PORT="${CLARE2_MLFLOW_PORT:-$(read_env CLARE2_MLFLOW_PORT 5000)}"
BIND_ADDRESS="${CLARE2_BIND_ADDRESS:-$(read_env CLARE2_BIND_ADDRESS 127.0.0.1)}"
DOCKER_GID="${CLARE2_DOCKER_GID:-$(read_env CLARE2_DOCKER_GID "")}"
if [[ -z "$DOCKER_GID" || "$DOCKER_GID" == "REPLACE_WITH_DOCKER_SOCKET_GROUP_ID" ]]; then
  DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
fi

write_env CLARE2_INFERENCE_MODEL "$INFERENCE_MODEL"
write_env CLARE2_DISTILL_MODEL "$INFERENCE_MODEL"
write_env CLARE2_TRAIN_MODEL "$TRAIN_MODEL"
write_env CLARE2_MODEL_CACHE "$MODEL_CACHE"
write_env CLARE2_MLFLOW_PORT "$MLFLOW_PORT"
write_env CLARE2_BIND_ADDRESS "$BIND_ADDRESS"
write_env CLARE2_DOCKER_GID "$DOCKER_GID"
[[ -z "${CLARE2_PROJECT_MAP:-}" ]] || write_env CLARE2_PROJECT_MAP "$CLARE2_PROJECT_MAP"
[[ -z "${CLARE2_PROJECT_ID:-}" ]] || write_env CLARE2_PROJECT_ID "$CLARE2_PROJECT_ID"

echo "Building Dockerized model setup helper..."
docker build -t "$SETUP_IMAGE" clare2/setup

OUTPUT_DIR=$(mktemp -d)
trap 'rm -rf "$OUTPUT_DIR"' EXIT
echo "Downloading pinned Qwen snapshots into $MODEL_CACHE"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "${MODEL_CACHE}:/cache" \
  -v "${HF_TOKEN_FILE}:/run/secrets/huggingface_token:ro" \
  -v "${OUTPUT_DIR}:/output" \
  "$SETUP_IMAGE" \
  --inference-model "$INFERENCE_MODEL" \
  --training-model "$TRAIN_MODEL" \
  --inference-revision "$INFERENCE_REVISION" \
  --training-revision "$TRAIN_REVISION"

while IFS='=' read -r key value; do
  [[ -n "$key" ]] && write_env "$key" "$value"
done <"${OUTPUT_DIR}/model.env"

echo "Validating Compose configuration..."
docker compose config --quiet
echo "Building CLARE₂ services..."
docker compose --profile training build \
  mlflow docker-socket-proxy clare2-policy clare2-mcp clare2-train spam-classifier
docker compose --profile training create clare2-train

if $START_SERVICES; then
  docker compose up -d mlflow redis docker-socket-proxy vllm-engine clare2-policy clare2-mcp spam-classifier
  echo "Waiting for MLflow..."
  for _ in $(seq 1 60); do
    curl --fail --silent "http://127.0.0.1:${MLFLOW_PORT}/health" >/dev/null && break
    sleep 2
  done
  curl --fail --silent "http://127.0.0.1:${MLFLOW_PORT}/health" >/dev/null
  echo "Waiting for the policy proxy..."
  for _ in $(seq 1 120); do
    curl --fail --silent http://127.0.0.1:8000/health >/dev/null && break
    sleep 5
  done
  curl --fail --silent http://127.0.0.1:8000/health
  printf '\n'
  echo "Waiting for the spam classifier..."
  for _ in $(seq 1 120); do
    curl --fail --silent http://127.0.0.1:8003/health >/dev/null && break
    sleep 5
  done
  curl --fail --silent http://127.0.0.1:8003/health
  printf '\n'

  OPERATOR_TOKEN=$(<"${SECRETS_DIR}/clare2_operator_token")
  curl --fail --silent \
    -H "Authorization: Bearer ${OPERATOR_TOKEN}" \
    http://127.0.0.1:8000/operator/status
  printf '\n'
fi

cat <<EOF
CLARE₂ setup complete.
Model cache: $MODEL_CACHE
Inference/MCP bind address: $BIND_ADDRESS
MCP port: 8002 (/mcp/)
Policy port: 8000
Spam classifier port: 8003
MLflow UI: http://127.0.0.1:${MLFLOW_PORT}
Agent wrapper: ${ROOT}/clare2/scripts/clare2-agent.sh
EOF
