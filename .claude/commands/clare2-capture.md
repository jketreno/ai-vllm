# /project:clare2-capture — Start or stop a CLARE₂ session capture

Initializes a new JSONL session file in `corpus/sessions/YYYY/MM/DD/` and sets
`CLARE2_SESSION_FILE` in the environment. Once set, `verify-ci.sh` automatically
appends structured CI event records to this file after each run.

## Usage

```
/project:clare2-capture start    # Begin capturing this session
/project:clare2-capture stop     # End the session (optional — file is already written)
/project:clare2-capture status   # Show the current session file path
```

## Instructions

Parse the argument passed as `$ARGUMENTS`:

### `start` (or no argument)

Run the session initializer and activate capture:

```bash
AI_VLLM_ROOT="${CLARE2_ROOT:-../ai-vllm}"
eval "$("${AI_VLLM_ROOT}/clare2/scripts/clare2-session-start.sh")"
echo "CLARE₂ session capture started."
echo "Session file: $CLARE2_SESSION_FILE"
echo "Session ID:   $CLARE2_SESSION_ID"
echo ""
echo "verify-ci.sh will now emit structured CI event records to this file."
echo "Run /project:distill at any time to distill today's sessions on demand."
```

Report:
- The session file path
- The session ID (UUID)
- A reminder that `verify-ci.sh` is now wired up

### `stop`

```bash
echo "Session file: ${CLARE2_SESSION_FILE:-'(no active session)'}"
unset CLARE2_SESSION_FILE
unset CLARE2_SESSION_ID
echo "CLARE₂ session capture stopped. CI events will no longer be recorded."
```

### `status`

```bash
if [[ -n "${CLARE2_SESSION_FILE:-}" ]]; then
  LINE_COUNT=$(wc -l < "$CLARE2_SESSION_FILE" 2>/dev/null || echo 0)
  echo "Active session: $CLARE2_SESSION_FILE"
  echo "Session ID:     ${CLARE2_SESSION_ID:-unknown}"
  echo "Records so far: $LINE_COUNT"
else
  echo "No active CLARE₂ session. Run /project:clare2-capture start to begin."
fi
```

## Notes

- The JSONL file is append-only. Stopping a session does not delete or truncate it.
- Raw session files are excluded from git (see `corpus/.gitignore`).
- The nightly distillation cron at 22:00 UTC processes today's sessions automatically.
  Use `/project:distill` for an on-demand run.
