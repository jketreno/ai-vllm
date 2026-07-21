#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  . ./.env
  set +a
fi

sam3_profile="sam3"
intel_compose="docker-compose.intel-sam3.yml"
intel_device="${SAM3_INTEL_DEVICE:-/dev/dri/renderD128}"

uses_intel_sam3() {
  case "${SAM3_PLATFORM:-}" in
    intel | intel_arc | b580 | xpu)
      return 0
      ;;
  esac

  [[ -e "$intel_device" && -n "${SAM3_INTEL_DEVICE_GID:-}" ]]
}

if [[ -n "${SAM3_WORKER_URL:-}" ]]; then
  echo "Using remote SAM3 worker: $SAM3_WORKER_URL"
  docker compose --profile "$sam3_profile" stop sam3-worker
  docker compose up -d --build
elif uses_intel_sam3; then
  echo "Starting local SAM3 worker on Intel Arc device: $intel_device"
  docker compose \
    -f "$intel_compose" \
    --profile "$sam3_profile" \
    up -d --build
else
  echo "Starting main stack with the local GB10 SAM3 worker"
  docker compose --profile "$sam3_profile" up -d --build
fi
