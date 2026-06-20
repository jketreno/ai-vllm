---
applyTo: "scripts/**,clare/verify-*.sh"
---
# Script Authoring Rules (CLARE — Constrained)

All scripts in `scripts/` follow these rules. Violations cause `verify-ci.sh` to fail.

## Required

- **Shebang**: `#!/usr/bin/env bash` (not `/bin/bash`)
- **Safety flags**: `set -euo pipefail` at the top of every script
- **Self-documenting header**: Block comment describing purpose, usage, and options
- **Executable**: Scripts must be `chmod +x`

## Quoting and Variables

- Always quote variables: `"$var"` not `$var`
- Use `[[ ]]` for conditionals, not `[ ]`
- Use `$()` for command substitution, not backticks
- Use `readonly` or `local` to scope variables appropriately

## Error Handling

- Check return codes explicitly when `set -e` would mask the intent
- Use `|| true` only when failure is genuinely acceptable and comment why
- Provide human-readable error messages before exiting

## Output

- Use color codes for status (see verify-ci.sh helper pattern)
- Use `>&2` to write errors to stderr
- Keep stdout machine-parseable when scripts are used in pipelines

## Portability

- Target bash 4.0+ (available on all modern Linux/macOS with Homebrew)
- Do not assume GNU-specific flags for `sed`, `awk`, `date` without checking
- Use `command -v tool` to test for tool availability before use

## Prohibited

- `rm -rf` without explicit user confirmation or `--force` flag
- Hardcoded credentials or tokens
- `curl | bash` patterns
- Suppressing errors with `2>/dev/null` unless explicitly justified in a comment
