# Example Skills — Copy and Customize

These are **domain-specific illustrations** of how to write CLARE skill files. They are not meant to be used as-is — each one assumes a specific tech stack and project layout.

**To use one:** copy it into your project, then edit the paths, types, and patterns to match your codebase. Look for `UPDATE:` comments marking the lines you need to change.

| Example | Demonstrates | Assumes |
|---------|-------------|---------|
| `api-endpoint.md` | [C] Constrained + [A] Assertive — enforcing endpoint standards | Express or FastAPI, Zod, rate limiting |
| `type-sync.md` | [A] Assertive + [R] Reality-Aligned + [E] Ephemeral — regenerating derived types | Python Pydantic backend, TypeScript frontend |
| `proto-visualization.md` | [L] Limited + [E] Ephemeral — full-autonomy regeneration | Protobuf APIs, React UI components |
| `reality-test.md` | [R] Reality-Aligned — verifying alignment with external systems | IAM provider, PostgreSQL, staging env |

## Writing your own skill

A skill file tells your AI tool exactly how to perform a repeatable task. Structure:

1. **When to apply** — trigger phrases and conditions
2. **The rule** — what source of truth to follow
3. **Step-by-step process** — what to generate and how
4. **Verification** — how to confirm correctness (usually `verify-ci.sh`)

See the generic skills in `clare/templates/skills/` (`autonomy-bootstrap.md`, `mcp-server.md`, `review.md`) for project-agnostic skills that work without customization.

## Installing examples into your project

Copy individually:
```bash
cp clare/examples/skills/api-endpoint.md .github/prompts/api-endpoint.prompt.md
```
