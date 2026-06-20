# CLARE with Multi-Agent Pipelines and MCP

> CLARE was designed for single-session AI tools, but its core primitives — `verify-ci.sh` and `autonomy.yml` — apply equally to orchestrated multi-agent workflows. This guide explains how.

---

## The Multi-Agent Problem

Single-session AI tools (Claude Code, Cursor, Copilot, Codex) load your AGENTS.md, CLAUDE.md, or rule files at startup and carry context throughout the session. Multi-agent pipelines are different:

- **Orchestrators** spawn sub-agents that may not share session context
- **Parallel agents** modify code concurrently — autonomy violations become race conditions
- **Ephemeral agents** complete one task and exit — there's no persistent "memory" of the rules
- **Headless agents** (running in CI, cron jobs, or triggered by events) have no interactive session at all

The question: **how do you enforce CLARE principles when there's no persistent session to hold the rules?**

The answer: **make enforcement stateless**.

`verify-ci.sh` is already stateless — it's a script that runs against the current state of the filesystem. `autonomy.yml` is a static YAML file any process can read. Neither requires a loaded AI session to be enforced.

### Orchestration Pseudocode

```text
function runClearAgentPipeline(tasks):
  restricted = clare_list_humans_only()

  for task in tasks:
    candidatePaths = predict_changed_paths(task)
    for p in candidatePaths:
      level = clare_check_autonomy(p)
      if level == "humans-only":
        stop_and_request_human(task, p)

  results = run_subagents_in_parallel(tasks)
  if any(result.status != "done" for result in results):
    fail_pipeline("sub-agent failure")

  verify = clare_verify()
  if verify.status != "passed":
    fail_pipeline("post-flight verify failed")

  return "ready-for-review"
```

### Enforcement Model Comparison

| Dimension | Single-agent session | Multi-agent pipeline |
|---|---|---|
| Autonomy check timing | Agent checks before edits in-session | Orchestrator pre-flight + sub-agent guard |
| Verification gate | Agent runs `./clare/verify-ci.sh` once before completion | Each sub-agent runs verify + final orchestrator verify |
| Rule propagation | Prompt/session memory | Tool contract (`clare_check_autonomy`, `clare_verify`) |
| Failure containment | One session rollback | Isolated task retry + post-merge verification |
| Humans-only enforcement | Prompt-based refusal | Delegation block at orchestrator boundary |

---

## Autonomy Boundaries in Multi-Agent Workflows

### The risk: agents that skip the check

In a single session, the AI reads CLARE config at startup and checks autonomy.yml before every file modification. In an orchestrated pipeline, a sub-agent may:
- Have no knowledge of CLARE or autonomy.yml
- Be a different AI system entirely (GPT-4, Gemini, local model)
- Be spawned with a minimal system prompt focused only on its subtask

**The solution: enforce autonomy at the orchestrator level.**

Before delegating a subtask to a sub-agent, the orchestrator (or a pre-flight step) should:

1. Identify which files the sub-task will likely touch
2. Look up their autonomy levels in `clare/autonomy.yml`
3. Refuse to delegate if any path is `humans-only`
4. Include `supervised` path warnings in the sub-agent's instructions

**Example orchestrator pre-flight prompt:**

```
Before assigning this task to a sub-agent, check clare/autonomy.yml.

Task: "Add rate limiting to the payment endpoint"
Files likely touched: src/payment/processor.ts, src/middleware/rate-limit.ts

For each file:
- If humans-only: do NOT delegate. Flag for human attention.
- If supervised: include a note in the sub-agent instructions: 
  "⚠ This path requires human review before commit."
- If full-autonomy: proceed normally.

Only delegate if no humans-only paths are involved.
```

### Architecture test: autonomy guard for pipelines

The `clare/templates/architecture-tests/autonomy-guard.test.js` template can be run as a pre-commit or CI gate, catching any changes that touch `humans-only` paths regardless of which agent made them:

```javascript
// tests/architecture/autonomy-guard.test.js
test('no commits touch humans-only paths without explicit override', () => {
  const changedFiles = getChangedFiles(); // git diff --name-only HEAD
  const humansOnly = getHumansOnlyPaths('clare/autonomy.yml');
  
  const violations = changedFiles.filter(f => 
    humansOnly.some(p => f.startsWith(p))
  );
  
  expect(violations).toHaveLength(0);
});
```

This is your last line of defense — runs in CI even if no agent ever checked autonomy.yml.

---

## verify-ci.sh as the Universal Gate

`verify-ci.sh` is the most portable CLARE primitive for multi-agent use. Every agent, regardless of its AI provider or session state, should run it before reporting work complete.

### Passing the gate requirement to any agent

For sub-agents in your pipeline, include this in every task prompt:

```
After completing your changes, run:
  ./clare/verify-ci.sh

Do not report the task as complete until all checks pass.
If the script fails, fix the errors and run again.
```

This is provider-agnostic. It works whether the sub-agent is Claude, Codex, GPT-4, a local model, or a scripted tool.

### CI/CD as the final backstop

Even if a sub-agent skips `verify-ci.sh`, the GitHub Actions workflow in `clare/templates/github-actions/ci.yml` catches it at PR time. The pipeline runs the same checks — no agent can merge code that fails them.

---

## MCP Integration

### CLARE₂ Temper routing

When CLARE₂ is available, agents call
`clare_temper_route(project, task_kind, capabilities)` for an opaque,
session-pinned route ID. `clare_temper_status` reports its immutable adapter,
and `clare_temper_list` is diagnostic inventory only. Send the route as
`X-CLARE-Route-ID` to the authenticated policy proxy.

Repository identities are configured canonical IDs, not filesystem paths.
Agents never select or load adapter IDs, provide adapter paths, access Docker
control, call raw vLLM management endpoints, or bypass the proxy. Requests
without a route intentionally use the base model.

[MCP (Model Context Protocol)](https://modelcontextprotocol.io/) is the emerging standard for giving AI agents structured, typed access to tools and data sources. Exposing CLARE as MCP tools makes enforcement available to any MCP-compatible agent or orchestrator without requiring bash execution or file reading.

### Why MCP for CLARE?

| Without MCP | With MCP |
|-------------|----------|
| Agent must parse autonomy.yml manually | Agent calls `clare_check_autonomy(path)` → gets typed result |
| Orchestrator runs bash to invoke verify-ci.sh | Orchestrator calls `clare_verify()` → gets structured pass/fail + error list |
| Sub-agents have no discoverable enforcement interface | CLARE tools are discoverable in the MCP server manifest |
| Provider-specific prompt engineering required | Any MCP-compatible agent uses the same tool interface |

### The three CLARE MCP tools

**`clare_verify`** — runs `clare/verify-ci.sh` and returns structured results:
```json
{
  "status": "failed",
  "passed": ["TypeScript build", "ESLint"],
  "failed": [{"check": "Jest", "output": "3 tests failed:\n  ..."}],
  "summary": "2/3 checks passed"
}
```

**`clare_check_autonomy`** — looks up a path in `clare/autonomy.yml`:
```json
{
  "path": "src/payment/processor.ts",
  "matched_rule": "src/payment",
  "level": "humans-only",
  "reason": "Money movement; AI errors are financial risk"
}
```

**`clare_list_humans_only`** — returns all `humans-only` paths; useful for pre-flight checks:
```json
{
  "humans_only_paths": ["src/payment", "src/auth/core", "ORIGIN.md"]
}
```

### Scaffolding a CLARE MCP server

Use the skill template at `clare/templates/skills/mcp-server.md` to generate a minimal MCP server for your project. The template generates a Node.js or Python server that wraps these three tools.

```
Follow clare/templates/skills/mcp-server.md to scaffold a CLARE MCP server
for this project.
```

Register the server in your Claude Code settings:

```json
// .claude/settings.json (or ~/.claude/settings.json for global)
{
  "mcpServers": {
    "clare": {
      "command": "node",
      "args": ["./mcp/clare-server.js"]
    }
  }
}
```

Once registered, Claude Code (and any other MCP client) can call these tools directly:
- `mcp__clare__clare_verify` — run CI checks
- `mcp__clare__clare_check_autonomy` — check a path
- `mcp__clare__clare_list_humans_only` — list restricted paths

---

## Multi-Agent Patterns with CLARE

### Pattern 1: Orchestrator + Sub-agents

```
Orchestrator
  ├── Pre-flight: clare_list_humans_only → identify restricted zones
  ├── Task assignment: include autonomy level in each sub-agent prompt
  ├── Sub-agent A: implements feature (runs clare_verify before reporting done)
  ├── Sub-agent B: writes tests (runs clare_verify before reporting done)
  └── Post-flight: orchestrator calls clare_verify as final gate
```

**Key rule:** The orchestrator enforces autonomy. Sub-agents enforce verification. Both run `clare_verify` — sub-agents after their own work, orchestrator after all sub-agents complete.

### Pattern 2: Parallel Code Generation

When multiple agents modify different parts of the codebase in parallel:

1. Each agent checks its target paths via `clare_check_autonomy` before starting
2. `humans-only` paths block delegation entirely — the agent should not start
3. `supervised` paths get a flag in the agent's instructions
4. Each agent runs `clare_verify` independently when done
5. A merge/integration step runs `clare_verify` one final time on the combined result

This prevents one agent's changes from breaking another agent's work only after they're combined.

### Pattern 3: Headless / Event-Driven Agents

Agents triggered by webhooks, CI events, or scheduled jobs have no interactive session. They should:

1. Read `clare/autonomy.yml` at startup and refuse to touch `humans-only` paths
2. Run `./clare/verify-ci.sh` (or call `clare_verify` via MCP) as the final step
3. Fail loudly (non-zero exit) if verification fails — let the pipeline catch it

The GitHub Actions template at `clare/templates/github-actions/ci.yml` is exactly this pattern: a headless agent that runs `verify-ci.sh` on every PR.

### Pattern 4: Agentic Code Review

Instead of a human reviewing every AI-generated PR, use an agent that:

1. Calls `clare_verify` — if it fails, the PR is not ready for review
2. Calls `clare_list_humans_only` — flags any changes to restricted paths
3. Checks that new tests are constraint tests, not confirmation tests (see [docs/principles/assertive.md](principles/assertive.md))
4. Posts a structured report: "CLARE checks: ✅ Passed / ❌ Autonomy violation in src/payment"

This turns CLARE enforcement from a human responsibility into an automated gate.

---

## What CLARE Does Not Solve in Multi-Agent Contexts

CLARE is an architectural guardrail, not a coordination protocol. It does not:

- **Manage agent communication** — use a proper orchestration framework (LangGraph, CrewAI, Claude Agents SDK, Codex workflows) for that
- **Prevent conflicting concurrent writes** — handle file locking or task partitioning at the orchestrator level
- **Replace integration tests** — `verify-ci.sh` runs unit/architecture tests locally; full integration tests still need a real environment
- **Audit which agent made which change** — use `git blame` and commit messages for attribution; structure agent commits with clear messages

CLARE's job is to ensure that whatever any agent generates, it passes your architectural invariants before it reaches review or merge. The coordination of *how* agents work together is a separate problem.

---

## Checklist: Multi-Agent CLARE Setup

- [ ] `clare/autonomy.yml` configured with `humans-only` paths for sensitive modules
- [ ] `clare/verify-ci.sh` tested and passing on current codebase
- [ ] Architecture tests cover key invariants (not just behavior)
- [ ] `autonomy-guard.test.js` added to CI as final backstop
- [ ] GitHub Actions workflow runs `verify-ci.sh` on every PR
- [ ] (Optional) CLARE MCP server scaffolded from `clare/templates/skills/mcp-server.md`
- [ ] Orchestrator prompt template includes autonomy pre-flight check
- [ ] Sub-agent prompts include `clare/verify-ci.sh` requirement

---

## Further Reading

| Topic | Document |
|-------|---------|
| Autonomy boundaries in depth | [docs/principles/limited.md](principles/limited.md) |
| Writing constraint tests for agentic code | [docs/principles/assertive.md](principles/assertive.md) |
| Generated code workflows | [docs/principles/ephemeral.md](principles/ephemeral.md) |
| Claude Code + MCP setup | [docs/ai-tools/claude.md](ai-tools/claude.md) |
| Codex setup | [docs/ai-tools/codex.md](ai-tools/codex.md) |
| MCP server skill template | [clare/templates/skills/mcp-server.md](../install/clare/templates/skills/mcp-server.md) |
