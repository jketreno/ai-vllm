#!/usr/bin/env bash
set -euo pipefail

read_secret() {
  local variable=$1
  local file_variable="${variable}_FILE"
  local path=${!file_variable:-}
  if [[ -n "$path" && -r "$path" ]]; then
    export "$variable=$(<"$path")"
  fi
}

read_secret OPENAI_API_KEY
read_secret LDAP_APP_PASSWORD

python /usr/local/lib/clare-open-webui-managed-config.py
exec bash /app/backend/start.sh "$@"
