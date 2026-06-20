# CLARE Principles — Quick Reference

> Keep this file in your repository root. AI tools are instructed to read it.
> Full documentation: docs/principles/

---

## [C] Constrained — Enforced, not suggested

Rules exist in code and tests, not in code review comments or docs.

**Mechanism:** `clare/verify-ci.sh` runs before any work is marked complete.  
**Implementation:** Architecture tests, linters, type checkers, build validation.  
**AI rule:** Never report work as complete if `./clare/verify-ci.sh` fails.

---

## [L] Limited — Boundaries are explicit

Each module has a declared autonomy level. AI checks before every change.

**Mechanism:** `clare/autonomy.yml` maps paths to autonomy levels.  
**Levels:**
- `full-autonomy` — AI proceeds freely
- `supervised` — AI generates; human reviews before commit
- `humans-only` — AI stops and alerts the user

**AI rule:** Read `clare/autonomy.yml` before touching any file. Refuse to generate code in `humans-only` paths.

---

## [A] Assertive — Tests define invariants, not confirmations

Tests capture what must always be true, not what the implementation currently does.

**Mechanism:** Architecture tests, property-based tests, schema-lock tests.  
**Test quality check:** Delete the implementation, regenerate it with AI, run tests. If tests fail, they're doing their job.  
**AI rule:** Write tests that would catch a correct-but-different implementation. Prefer invariants ("this can never happen") over confirmations ("this currently returns X").

---

## [R] Reality-Aligned — Single source of truth

Each domain concept has one authoritative source. Everything derives from it.

**Mechanism:** `clare/autonomy.yml` `sources_of_truth` section.  
**Implementation:** Reality tests run against external systems in staging/nightly.  
**AI rule:** When generating code for a domain concept, find its `source_of_truth` in `clare/autonomy.yml` and derive from that. Never invent a representation.

---

## [E] Ephemeral — Generated code is regenerated, not hand-edited

Code derived from a source of truth is never manually patched.

**Mechanism:** Skill files define regeneration rules. AI follows them exactly.  
**Implementation:** When models change, run the skill — don't patch the output.  
**AI rule:** If a file is marked as generated (header comment, autonomy level), regenerate from source rather than editing in place.

---

## Workflow Summary

```
You: "Implement X"
  ↓
AI checks clare/autonomy.yml for affected paths
  ↓
AI checks clare/principles.md for applicable rules
  ↓
AI generates code following constraints
  ↓
AI runs ./clare/verify-ci.sh
  ↓
If fails → AI fixes and reruns
  ↓
All checks pass → AI reports: "Complete. All CI checks pass."
You: review the diff, then commit
```

---

## Multi-Agent and MCP

CLARE principles apply equally to orchestrated multi-agent pipelines. Key rules:

- **Orchestrators:** check `clare/autonomy.yml` before delegating any subtask; never delegate a `humans-only` path
- **Sub-agents:** run `./clare/verify-ci.sh` before reporting a subtask complete, regardless of AI provider
- **Headless agents:** read `clare/autonomy.yml` at startup; exit non-zero if verification fails — let CI catch it
- **MCP:** expose `verify-ci.sh` and `autonomy.yml` as MCP tools (`clare_verify`, `clare_check_autonomy`) for structured agent access

See [docs/agentic.md](../docs/agentic.md) for full patterns and the `clare/templates/skills/mcp-server.md` skill to scaffold a CLARE MCP server.

---

## Files in This Repository

| File | Purpose |
|------|---------|
| `clare/autonomy.yml` | Module autonomy boundaries + sources of truth |
| `clare/principles.md` | This file — AI quick reference |
| `clare/verify-ci.sh` | Local CI/CD enforcement — CLARE-owned, auto-updated |
| `clare/verify-local.sh` | Project-specific checks — yours to edit |
| `clare/templates/architecture-tests/` | Generic architecture tests (autonomy guard) |
| `clare/examples/architecture-tests/` | Domain-specific test examples (API rules, type sync, module boundaries) |
| `clare/templates/skills/` | Generic AI skills (MCP server, code review) |
| `clare/examples/skills/` | Domain-specific skill illustrations (copy and customize) |
| `clare/templates/github-actions/` | CI/CD workflow templates |
| `clare/templates/linting/` | ESLint config templates |
| `clare/docs/agentic.md` | Multi-agent pipelines and MCP integration guide |
| `AGENTS.md` | Codex configuration (auto-read at startup) |
| `.github/copilot-instructions.md` | GitHub Copilot / VS Code workspace config |
| `CLAUDE.md` | Claude Code configuration (auto-read at startup) |
| `.cursor/rules/` | Cursor AI rules (MDC format) |
| `clare/templates/skills/mcp-server.md` | Skill to scaffold a CLARE MCP server |
