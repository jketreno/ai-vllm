# CLARE — Codex Configuration

This project uses the **CLARE** framework for AI-assisted development.

## Deployment Topology

This checkout is a **development system**. The `docker-compose.yml` services
(`qwen-image-edit-worker`, `sam3-worker`, `image-api`, `vllm-engine`, etc.) do
**not** run here — expect `docker compose ps` / `docker ps` on this host to
show them absent. Those services run on a separate **production system**.
Do not assume a service is broken or misconfigured just because it has no
running container on this dev machine; ask the user how to reach the
production host (e.g. remote docker context, SSH) if logs or live status are
needed.

## The verify-ci.sh Rule

**When to run it:** only in a turn where you created, updated, or deleted a file. If you edited no files this turn — including any plan-mode, research, or read-only turn — do not run it.

**How to run it (a terminating sequence, not a loop):**

1. Run `./clare/verify-ci.sh` once.
2. Exit code 0 → report PASS and stop.
3. Non-zero → fix the reported failures, then run it once more.

Do not invoke it again for any other reason, and do not report implementation work complete while it fails.

**Reporting:** if you ran verify-ci.sh this turn, report:
- verify-ci.sh result: PASS/FAIL
- command used
- whether `--fast`, `--fix`, or `--fail-slow` was used, if applicable

If you edited no files this turn, state `no edits this turn — verify-ci.sh not run` and do not invoke it. If running in read-only mode, state: `verify-ci.sh not run because session is read-only.`

## Cross-Agent Sync Rule

When CLARE agent configuration changes, update all supported agent environments in the same change.

Supported environments:
- Copilot: `.github/copilot-instructions.md` and `install/.github/copilot-instructions.md`
- Claude: `CLAUDE.md` and `install/root/CLAUDE.md`
- Codex: `AGENTS.md` and `install/root/AGENTS.md`
- Cursor: `.cursorrules` and `install/root/.cursorrules`

This includes rule updates, skill/workflow guidance, and CLARE installer-facing agent instructions.

## Session Start Checklist

At the start of every session, before making code changes:

1. Read `clare/autonomy.yml` and identify `humans-only` paths.
2. Read `clare/principles.md`.
3. Check `sources_of_truth` in `clare/autonomy.yml` for relevant domain concepts.

## Autonomy Boundaries

Before modifying any file, check its path in `clare/autonomy.yml`:

| Level | Action |
|-------|--------|
| `full-autonomy` | Proceed freely |
| `supervised` | Generate code, then note that human review is required |
| `humans-only` | Stop. Do not generate code for this path |

If a file is marked `humans-only`, say:

> This path is marked `humans-only` in `clare/autonomy.yml`. I won't generate code here.

## CLARE Principles

- **Constrained** — Rules are enforced by `clare/verify-ci.sh`.
- **Limited** — Autonomy levels in `clare/autonomy.yml` define editable boundaries.
- **Assertive** — Tests should enforce invariants, not confirm current implementation details.
- **Reality-Aligned** — Declared sources of truth in `clare/autonomy.yml` win over assumptions.
- **Ephemeral** — Generated files must be regenerated from their source, not hand-edited.

## Linter Integrity

Fix what the linters and `clare/verify-ci.sh` flag — never circumvent it. When complexity or other checks are enabled, do not silence findings with per-file/per-line suppressions (`// eslint-disable complexity`, `# noqa`, `# type: ignore`, `nolint`), do not raise thresholds in `clare/extensions.yml`, and do not exclude a file to dodge a fix. Refactor to satisfy the check; if a limit is genuinely wrong, raise it with the user rather than changing it silently to get green.

## Key Files

- `clare/autonomy.yml` — autonomy boundaries and sources of truth
- `clare/principles.md` — CLARE principles quick reference
- `clare/verify-ci.sh` — required verification gate before completion
- `clare/verify-local.sh` — project-specific checks
- `clare/templates/skills/` — reference skill templates

## CLARE₂ Commands

When the CLARE₂ pipeline container is running:

- `/project:clare2-capture start` — initialize session JSONL capture; wires verify-ci.sh to emit CI events
- `/project:distill` — trigger an on-demand distillation pass (also runs nightly at 22:00 UTC)

## CLARE₂ Temper Routing

Use `clare_temper_route(project, task_kind, capabilities)` for an opaque route
ID and send `X-CLARE-Route-ID` to the policy proxy. Never select/load adapters,
provide adapter paths, call raw vLLM management endpoints, or access Docker.
