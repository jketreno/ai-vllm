#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
  echo "Usage: clare2-agent.sh <agent-command> [arguments...]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CORPUS_ROOT="${CORPUS_ROOT:-${SCRIPT_DIR}/../../corpus}"
eval "$("${SCRIPT_DIR}/clare2-session-start.sh")"

exec "$@"
