#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
TEMPLATES="${ROOT}/clare/templates/hooks"
SCRIPTS_TEMPLATES="${ROOT}/clare/templates/scripts"
SCRIPTS_TARGET="${ROOT}/clare/scripts"

command -v jq >/dev/null 2>&1 || {
  echo "jq is required to merge CLARE2 hook configuration" >&2
  exit 1
}

# merge_hooks <template> <target>
#   Merges the hook entries in $template into $target's "hooks" map,
#   skipping entries whose command already exists for that event. Supports
#   both nested-command layouts (Codex/Claude: entry.hooks[0].command) and
#   flat-command layouts (Copilot: entry.bash).
merge_hooks() {
  local template=$1
  local target=$2
  local relative="${target#"$ROOT"/}"
  local temporary

  if [[ -e "$target" ]] \
    && [[ -n "$(git -C "$ROOT" status --porcelain -- "$relative" 2>/dev/null || true)" ]]; then
    echo "Skipped CLARE2 hook merge: $relative has local changes" >&2
    return 0
  fi

  temporary=$(mktemp)

  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" ]]; then
    jq --slurpfile additions "$template" '
      def entry_command: if .bash then .bash elif (.hooks[0].command // null) != null then .hooks[0].command else null end;
      reduce ($additions[0].hooks | keys[]) as $event (.;
        (.hooks[$event] // []) as $existing
        | [$existing[] | entry_command | select(. != null)] as $commands
        | .hooks[$event] = (
            $existing +
            [
              $additions[0].hooks[$event][]
              | select(entry_command as $new_command | $new_command != null and ($commands | index($new_command) | not))
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

mkdir -p "$SCRIPTS_TARGET"
cp "$SCRIPTS_TEMPLATES/clare2-capture-event.sh" "$SCRIPTS_TARGET/clare2-capture-event.sh"
chmod 0755 "$SCRIPTS_TARGET/clare2-capture-event.sh"

merge_hooks "$TEMPLATES/codex-hooks.json" "$ROOT/.codex/hooks.json"
merge_hooks "$TEMPLATES/claude-hooks.json" "$ROOT/.claude/settings.json"
merge_hooks "$TEMPLATES/copilot-hooks.json" "$ROOT/.github/hooks/clare2-corpus.json"

echo "Installed CLARE2 hooks for Codex, Claude Code, and GitHub Copilot."
echo "Set CLARE2_CORPUS_ROOT or CLARE2_ROOT before starting an agent."
