#!/usr/bin/env bash
set -euo pipefail

PIPELINE_URL=${CLARE2_PIPELINE_URL:-http://127.0.0.1:8000}
TOKEN_FILE=${CLARE2_OPERATOR_TOKEN_FILE:-secrets/clare2_operator_token}

[[ -r "$TOKEN_FILE" ]] || {
  echo "operator token is not readable: $TOKEN_FILE" >&2
  exit 1
}

curl --fail --silent --show-error \
  -X POST \
  -H "Authorization: Bearer $(<"$TOKEN_FILE")" \
  "${PIPELINE_URL}/operator/rollback"
printf '\n'
