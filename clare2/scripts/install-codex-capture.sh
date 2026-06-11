#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "Usage: install-codex-capture.sh <project-directory>" >&2
  exit 2
fi

PROJECT=$(cd "$1" && pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAPTURE_SCRIPT="${SCRIPT_DIR}/clare2-capture-hook.sh"
HOOK_DIR="${PROJECT}/.codex"
HOOK_FILE="${HOOK_DIR}/hooks.json"
TEMP_FILE=$(mktemp)
trap 'rm -f "$TEMP_FILE"' EXIT

PROMPT_COMMAND=$(printf '%q %q' "$CAPTURE_SCRIPT" UserPromptSubmit)
STOP_COMMAND=$(printf '%q %q' "$CAPTURE_SCRIPT" Stop)
NEW_HOOKS=$(jq -n \
  --arg prompt_command "$PROMPT_COMMAND" \
  --arg stop_command "$STOP_COMMAND" \
  '{
    hooks: {
      UserPromptSubmit: [{
        hooks: [{type: "command", command: $prompt_command, timeout: 10}]
      }],
      Stop: [{
        hooks: [{type: "command", command: $stop_command, timeout: 10}]
      }]
    }
  }')

mkdir -p "$HOOK_DIR"
if [[ -f "$HOOK_FILE" ]]; then
  jq --argjson additions "$NEW_HOOKS" '
    reduce ($additions.hooks | keys[]) as $event (.;
      (.hooks[$event] // []) as $existing
      | [$existing[].hooks[].command] as $commands
      | .hooks[$event] = (
        $existing +
        [
          $additions.hooks[$event][]
          | select(.hooks[0].command as $command | $commands | index($command) | not)
        ]
      )
    )
  ' "$HOOK_FILE" >"$TEMP_FILE"
else
  printf '%s\n' "$NEW_HOOKS" >"$TEMP_FILE"
fi

jq -e . "$TEMP_FILE" >/dev/null
chmod 0644 "$TEMP_FILE"
mv "$TEMP_FILE" "$HOOK_FILE"
trap - EXIT
echo "Installed Codex capture hooks in $HOOK_FILE"
