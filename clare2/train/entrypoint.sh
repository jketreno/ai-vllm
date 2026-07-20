#!/usr/bin/env bash
set -euo pipefail

STATE=${CLARE2_LIFECYCLE_STATE_PATH:-/corpus/meta/lifecycle.json}

if [[ "${CLARE2_TRAIN_AUTHORIZED:-0}" == "1" ]]; then
  exec "$@"
fi

if [[ ! -s "$STATE" ]]; then
  echo "Training is not authorized: lifecycle state is unavailable"
  exit 0
fi

AUTHORIZED=$(python3 - "$STATE" <<'PY'
import json
import sys

try:
    state = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print(0)
else:
    authorized_phase = state.get("phase") in {"starting_training", "training"}
    print(int(authorized_phase and state.get("trainer_start_requested") is True))
PY
)

if [[ "$AUTHORIZED" != "1" ]]; then
  echo "Training is not authorized by the current lifecycle state"
  exit 0
fi

exec "$@"
