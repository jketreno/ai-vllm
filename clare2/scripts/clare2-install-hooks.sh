#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "Usage: clare2-install-hooks.sh <project-directory>" >&2
  exit 2
fi

PROJECT=$(cd "$1" && pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEMPLATES_DIR="${SCRIPT_DIR}/../templates/hooks"

command -v jq >/dev/null 2>&1 || {
  echo "jq is required to merge CLARE2 hook configuration" >&2
  exit 1
}

merge_hooks() {
  local template=$1
  local target=$2
  local temporary
  temporary=$(mktemp)

  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" ]]; then
    jq --slurpfile additions "$template" '
      reduce ($additions[0].hooks | keys[]) as $event (.;
        (.hooks[$event] // []) as $existing
        | [$existing[].hooks[]? | .command] as $commands
        | .hooks[$event] = (
            $existing +
            [
              $additions[0].hooks[$event][]
              | .hooks[0].command as $new_command
              | select($commands | index($new_command) | not)
            ]
          )
      )
    ' "$target" >"$temporary"
  else
    cp "$template" "$temporary"
  fi
  jq -e . "$temporary" >/dev/null
  chmod 0644 "$temporary"
  mv "$temporary" "$target"
}

merge_hooks "$TEMPLATES_DIR/codex-hooks.json" "$PROJECT/.codex/hooks.json"
merge_hooks "$TEMPLATES_DIR/claude-hooks.json" "$PROJECT/.claude/settings.json"
mkdir -p "$PROJECT/.github/hooks"
cp "$TEMPLATES_DIR/copilot-hooks.json" "$PROJECT/.github/hooks/clare2-corpus.json"

echo "Installed CLARE2 hooks for Codex, Claude Code, and GitHub Copilot."
