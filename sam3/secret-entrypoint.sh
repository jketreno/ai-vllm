#!/bin/sh
set -eu

if [ -n "${HF_TOKEN_FILE:-}" ]; then
  HF_TOKEN=$(tr -d '\r\n' < "$HF_TOKEN_FILE")
  export HF_TOKEN
fi

exec "$@"
