#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Read the handful of vars this script's own branching logic needs directly
# from .env, without shell-sourcing the file. Bash-sourcing (`. .env`) runs
# every line through the shell parser, which strips quote characters from
# values like `{"a":"b"}` (unescaped `"..."` is shell quoting syntax, not
# literal text) -- silently corrupting valid JSON before docker compose ever
# sees it. `docker compose` reads .env natively (no shell involved) and does
# not have this problem, so container-bound values are left for it to parse;
# only these few values are needed here in the shell itself.
env_value() {
  local key="$1"
  [[ -f .env ]] || return 0
  sed -n "s/^${key}=//p" .env | tail -n1
}

SAM3_WORKER_URL="$(env_value SAM3_WORKER_URL)"
SAM3_PLATFORM="$(env_value SAM3_PLATFORM)"
SAM3_INTEL_DEVICE="$(env_value SAM3_INTEL_DEVICE)"
SAM3_INTEL_DEVICE_GID="$(env_value SAM3_INTEL_DEVICE_GID)"

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
