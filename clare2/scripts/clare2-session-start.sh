#!/usr/bin/env bash
# CLARE₂ session capture initializer
# Creates a new session JSONL file and prints the export command for CLARE2_SESSION_FILE.
#
# Usage:
#   eval "$(./clare2/scripts/clare2-session-start.sh)"
#
# This sets CLARE2_SESSION_FILE in the current shell.
# verify-ci.sh checks this variable and appends CI event records when it is set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS_ROOT="${CORPUS_ROOT:-${SCRIPT_DIR}/../../corpus}"

# Generate a session ID using /proc/sys/kernel/random/uuid if available, else Python
if [[ -r /proc/sys/kernel/random/uuid ]]; then
  SESSION_ID=$(cat /proc/sys/kernel/random/uuid)
elif command -v python3 &>/dev/null; then
  SESSION_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
else
  SESSION_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
fi

TODAY=$(date -u +%Y/%m/%d)
SESSION_DIR="${CORPUS_ROOT}/sessions/${TODAY}"
mkdir -p "${SESSION_DIR}"

SESSION_FILE="${SESSION_DIR}/${SESSION_ID}.jsonl"

# Gather context
PROJECT=$(basename "$(git rev-parse --show-toplevel 2>/dev/null || pwd)")
CLARE1_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
MODEL="${CLARE2_MODEL:-unknown}"
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Write session_meta header record
jq -n \
  --arg type "session_meta" \
  --arg session_id "${SESSION_ID}" \
  --arg project "${PROJECT}" \
  --arg started_at "${TS}" \
  --arg model "${MODEL}" \
  --arg clare1_commit "${CLARE1_COMMIT}" \
  '{type:$type, session_id:$session_id, project:$project, started_at:$started_at, model:$model, clare1_commit:$clare1_commit}' \
  >> "${SESSION_FILE}"

# Output the export command — caller should eval this
echo "export CLARE2_SESSION_FILE=${SESSION_FILE}"
echo "export CLARE2_SESSION_ID=${SESSION_ID}"
