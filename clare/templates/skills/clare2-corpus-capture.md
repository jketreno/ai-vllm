---
name: clare2-corpus-capture
description: "Capture normalized agent interactions for local CLARE2 distillation"
mode: agent
---

# CLARE2 Corpus Capture

Use deterministic lifecycle hooks to send useful agent interactions into the
local CLARE2 corpus. Sessions are captured per-project under a shared user
corpus at `~/.config/clare/corpus`, partitioned by project name. Each project
builds its own adapter.

## Agent support matrix

| Agent | Capture supported | Notes |
|---|---|---|
| Claude Code | Yes | Hooks in `.claude/settings.json` |
| Codex | Yes | Hooks in `.codex/hooks.json` |
| GitHub CoPilot | Unverified | Template written to `.github/hooks/clare2-corpus.json`; confirm this path matches your `gh copilot` version |
| Zoo Code | No | VSCode extension API does not expose compatible lifecycle hooks |

## Configuration

`clare-installer.sh` sets `CLARE2_CORPUS_ROOT` automatically, creates
`~/.config/clare/corpus`, and symlinks the project's `corpus/` directory there
so the pipeline container resolves the same data. Run:

```bash
./scripts/clare-installer.sh --update --target /path/to/project
```

If you need to set it manually:

```bash
export CLARE2_CORPUS_ROOT="$HOME/.config/clare/corpus"
```

`CLARE2_SESSION_FILE` takes precedence when an existing wrapper or command has
already initialized a session. `CLARE2_PROJECT_ID` overrides the project name
derived from `git rev-parse --show-toplevel` basename (used by default).

Text content is redacted and limited to 12,000 characters by default. Set
`CLARE2_CAPTURE_MAX_CHARS` to a smaller non-negative integer, or `0` to disable
prompt, response, and correction text capture while retaining lifecycle and
tool outcome events.

Reinstall or repair the project hooks with:

```bash
./clare/scripts/clare2-install-hooks.sh
```

Hooks are fail-open. If the corpus is unavailable, normal agent work continues.

## Capture Rules

Capture observable evidence:

- submitted user prompts
- final assistant responses when the provider exposes them
- tool names and success/failure outcomes, without raw arguments or output
- session lifecycle events
- `verify-ci.sh` results and corrections already emitted by CLARE

Never capture:

- credentials, authorization headers, environment dumps, or secret files
- hidden reasoning or chain-of-thought
- raw tool arguments, command output, or complete source files
- private provider transcript files or undocumented cache formats
- internal CLARE2 distillation, evaluation, or summarization requests

Treat `corpus/` as sensitive local data. Do not commit it.

## Agent Guidance

When deterministic hooks are active, do not duplicate their interaction
records. Continue to emit explicit, concise correction records when a user
rejects an approach or states a preferred replacement:

```bash
printf '%s' '{
  "type": "correction",
  "problem": "Used mutable global state",
  "preferred": "Inject an immutable dependency"
}' | ./clare/scripts/clare2-capture-event.sh generic correction
```

For agents without compatible hooks, launch them with the environment variables
above and call `clare2-capture-event.sh` from their documented prompt and
completed-turn extension points. Do not scrape their private log directories.

## Verification

1. Submit one harmless prompt.
2. Check for a dated JSONL file:

   ```bash
   find "$CLARE2_CORPUS_ROOT/sessions" -name '*.jsonl' -type f | tail -1
   ```

3. Validate every line:

   ```bash
   jq -e . <path-to-session.jsonl> >/dev/null
   ```

4. Confirm the file contains `session_meta` and `interaction` records.
