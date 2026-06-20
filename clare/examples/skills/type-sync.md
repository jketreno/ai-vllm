# CLARE Skill: Python ↔ TypeScript Type Synchronization
# =======================================================
# CLARE Principles: [A] Assertive — [R] Reality-Aligned — [E] Ephemeral
#
# USAGE: Copy this file into your project's skills/rules directory
#   VS Code/Copilot: .github/instructions/type-sync.instructions.md
#   Claude:          CLAUDE.md (paste the content) or .claude/commands/type-sync.md
#   Cursor:          .cursor/rules/type-sync.mdc
#   Codex:           .codex/skills/type-sync/SKILL.md or AGENTS.md
#
# This skill tells your AI exactly how to keep Python and TypeScript types in sync.
# The Python Pydantic models are the SOURCE OF TRUTH.

---

## When to Apply This Skill

Apply this skill whenever:
- A new Pydantic model is added to `backend/models/api/` (UPDATE: your path)
- A Pydantic model field is added, removed, renamed, or its type changes
- A user says "update types", "sync types", "regenerate TS types"
- `verify-ci.sh` fails with "Field X missing in TypeScript"

---

## The Rule

**Python Pydantic models are the source of truth.**

When Python and TypeScript disagree, the Python model is correct. Regenerate the TypeScript from Python — never the reverse.

---

## Directory Layout

```
backend/
  models/
    api/           ← Python Pydantic models (SOURCE OF TRUTH)
      user.py
      order.py

frontend/
  src/
    types/
      api/         ← Generated TypeScript interfaces (DERIVED)
        user.ts    ← Generated from user.py
        order.ts   ← Generated from order.py
      schemas/     ← Generated Zod validation schemas (DERIVED)
        user.ts
        order.ts
```

UPDATE: adjust these paths to match your project structure.

---

## Pydantic → TypeScript Mapping Rules

### Type Mapping

| Python (Pydantic) | TypeScript | Zod |
|-------------------|------------|-----|
| `str` | `string` | `z.string()` |
| `int` | `number` | `z.number().int()` |
| `float` | `number` | `z.number()` |
| `bool` | `boolean` | `z.boolean()` |
| `list[X]` | `X[]` | `z.array(ZodX)` |
| `dict[str, X]` | `Record<string, X>` | `z.record(ZodX)` |
| `Optional[X]` | `X \| null` | `ZodX.nullable()` |
| `X \| None` | `X \| null` | `ZodX.nullable()` |
| `Literal["a", "b"]` | `"a" \| "b"` | `z.enum(["a", "b"])` |
| `EmailStr` | `string` | `z.string().email()` |
| `datetime` | `string` (ISO 8601) | `z.string().datetime()` |
| `UUID` | `string` | `z.string().uuid()` |
| `Decimal` | `number` | `z.number()` |

### Field Name Mapping

Always convert: `snake_case` (Python) → `camelCase` (TypeScript)

```
user_id        → userId
subscription_status → subscriptionStatus
created_at     → createdAt
```

### Validation Rules Mapping

| Pydantic | Zod |
|----------|-----|
| `Field(min_length=1)` | `.min(1)` |
| `Field(max_length=100)` | `.max(100)` |
| `Field(ge=0)` | `.nonnegative()` |
| `Field(gt=0)` | `.positive()` |
| `Field(le=100)` | `.max(100)` |
| `Field(pattern=r"...")` | `.regex(/.../)` |
| `Field(description="...")` | `// comment` |

---

## Generation Template

Given this Python model:

```python
# backend/models/api/user.py
class User(BaseModel):
    """User account information."""
    user_id: str = Field(min_length=1, description="Unique identifier")
    email: EmailStr = Field(description="User email address")  
    display_name: str = Field(min_length=1, max_length=100)
    subscription_status: Literal["free", "premium", "enterprise"]
    created_at: datetime
    phone_number: Optional[str] = None
    
    model_config = ConfigDict(json_schema_extra={"ts_interface": "User"})
```

Generate this TypeScript interface:

```typescript
// frontend/src/types/api/user.ts
// @generated — DO NOT EDIT DIRECTLY
// Source: backend/models/api/user.py
// Regenerate with: [your regeneration command]

/** User account information. */
export interface User {
  /** Unique identifier */
  userId: string;
  /** User email address */
  email: string;
  displayName: string;
  subscriptionStatus: "free" | "premium" | "enterprise";
  createdAt: string; // ISO 8601
  phoneNumber: string | null;
}
```

And this Zod schema:

```typescript
// frontend/src/types/schemas/user.ts
// @generated — DO NOT EDIT DIRECTLY
// Source: backend/models/api/user.py

import { z } from 'zod';

export const UserSchema = z.object({
  userId: z.string().min(1),
  email: z.string().email(),
  displayName: z.string().min(1).max(100),
  subscriptionStatus: z.enum(["free", "premium", "enterprise"]),
  createdAt: z.string().datetime(),
  phoneNumber: z.string().nullable(),
});

export type User = z.infer<typeof UserSchema>;
```

---

## Step-by-Step Process

When asked to sync types after a Python model change:

1. **Read** the changed Pydantic model(s) from `backend/models/api/`
2. **Identify** all changed/added/removed fields
3. **Generate** the TypeScript interface using the type mapping above
4. **Generate** the Zod schema to match
5. **Update** both `frontend/src/types/api/[model].ts` and `frontend/src/types/schemas/[model].ts`
6. **Add** the `@generated` header to all output files
7. **Run** `./clare/verify-ci.sh` — the type-compatibility architecture test must pass
8. **Report** which models were updated and which TS files were changed

---

## Drift Detection

The architecture test `clare/examples/architecture-tests/type-compatibility.test.ts` catches drift.
To run it standalone:

```bash
npx jest tests/architecture/type-compatibility.test.ts
```

If it fails: re-run this skill on the failing models.

---

## What NOT to Do

- ❌ Do not hand-edit generated TypeScript files — regenerate from Python
- ❌ Do not change TypeScript types when the Python model is the source of truth
- ❌ Do not omit the `@generated` header from output files
- ❌ Do not add TypeScript-only fields not present in the Python model (document exclusions explicitly if needed)
