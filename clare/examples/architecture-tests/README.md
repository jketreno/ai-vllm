# Example Architecture Tests — Copy and Customize

These are **domain-specific illustrations** of architecture tests that enforce structural rules. They are not meant to be used as-is — each one assumes a specific tech stack and project layout.

**To use one:** copy it into your project's `tests/architecture/` directory, then edit the paths, patterns, and rules to match your codebase. Look for `UPDATE:` comments marking the lines you need to change.

| Example | Demonstrates | Assumes |
|---------|-------------|---------|
| `api-rules.test.js` | [C] Constrained — enforcing API endpoint standards | Express routes in `src/api/`, rate limiting, Zod validation |
| `module-boundaries.test.js` | [L] Limited — enforcing import restrictions between modules | Specific `src/` directory structure with domain/auth/payment modules |
| `type-compatibility.test.ts` | [R] Reality-Aligned — detecting type drift between systems | Python Pydantic backend, TypeScript frontend |

## Generic architecture tests

The `clare/templates/architecture-tests/` directory contains tests that work on any CLARE project without customization:

- `autonomy-guard.test.js` — reads `autonomy.yml` and checks staged files against `humans-only` paths

## Installing examples into your project

Extract examples on demand:
```bash
cp clare/examples/architecture-tests/api-rules.test.js tests/architecture/
```
