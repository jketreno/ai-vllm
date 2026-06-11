#!/usr/bin/env bash
set -euo pipefail

SESSION_FILE="${CLARE2_SESSION_FILE:-}"
[[ -n "$SESSION_FILE" ]] || exit 0

EVENT="${1:-}"
case "$EVENT" in
  UserPromptSubmit)
    FILTER="{type:\"interaction\",role:\"user\",content:.prompt,\
session_id:.session_id,turn_id:.turn_id,ts:\$ts}"
    ;;
  Stop)
    FILTER="{type:\"interaction\",role:\"assistant\",\
content:.last_assistant_message,session_id:.session_id,\
turn_id:.turn_id,ts:\$ts}"
    ;;
  *)
    exit 0
    ;;
esac

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RECORD=$(jq -c --arg ts "$TS" "($FILTER) | select(.content != null)")
[[ -n "$RECORD" ]] || exit 0

mkdir -p "$(dirname "$SESSION_FILE")"
(
  flock -x 9
  printf '%s\n' "$RECORD" >>"$SESSION_FILE"
) 9>"${SESSION_FILE}.lock"
