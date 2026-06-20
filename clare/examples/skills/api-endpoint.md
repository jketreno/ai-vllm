# CLARE Skill: API Endpoint Generation
# ======================================
# CLARE Principles: [C] Constrained — [A] Assertive — [E] Ephemeral
#
# USAGE: Copy this file into your project's AI instructions directory.
# This skill ensures every generated endpoint includes rate limiting,
# input validation, tests, and OpenAPI documentation.
#
# Wire into verify-ci.sh: the architecture test checks rate limiting on every PR.

---

## When to Apply This Skill

Apply when a user says:
- "add a [X] endpoint"
- "create a POST /users route"
- "implement the [feature] API"
- verify-ci.sh fails with "endpoint missing rate limiting"

---

## Non-Negotiables for Every Endpoint

Every generated endpoint MUST include:

1. **Rate limiting** — no exceptions
2. **Input validation** — validate all inputs against a schema
3. **Authentication check** — unless explicitly marked as public
4. **Standardized error format** — same structure as all other endpoints
5. **OpenAPI documentation** — JSDoc or decorators matching the spec
6. **Tests** — at least one unit test + one constraint test
7. **Audit logging** — for state-modifying endpoints (POST/PUT/PATCH/DELETE)

If you cannot satisfy all of these, stop and ask which constraint to relax.

---

## Express/Node.js Template

### Route file

```typescript
// src/api/routes/users.ts

import { Router, Request, Response, NextFunction } from 'express';
import { rateLimit } from 'express-rate-limit';
import { z } from 'zod';
import { createUser } from '../../services/user.service';
import { ApiError } from '../../utils/api-error';
import { audit } from '../../utils/audit';
import { authenticate } from '../../middleware/auth';

const router = Router();

// Rate limiting — adjust limits to match your SLA and abuse risk
const createUserLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10,                   // 10 requests per window
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests', code: 'RATE_LIMITED' },
});

// Input schema — the single source of validation truth for this endpoint
const CreateUserSchema = z.object({
  email: z.string().email({ message: 'Invalid email address' }),
  displayName: z.string().min(1).max(100),
  role: z.enum(['user', 'admin']).optional().default('user'),
});

/**
 * @openapi
 * /users:
 *   post:
 *     summary: Create a new user
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/CreateUserRequest'
 *     responses:
 *       201:
 *         description: User created
 *       400:
 *         description: Validation error
 *       409:
 *         description: Email already exists
 *       429:
 *         description: Rate limit exceeded
 */
router.post(
  '/users',
  createUserLimiter,
  authenticate,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      // Validate input
      const body = CreateUserSchema.safeParse(req.body);
      if (!body.success) {
        throw new ApiError(400, 'Validation error', body.error.flatten());
      }

      // Business logic
      const user = await createUser(body.data);

      // Audit log for state mutation
      audit.log('user.created', { actorId: req.user?.id, targetId: user.id });

      res.status(201).json({ data: user });
    } catch (err) {
      next(err);
    }
  }
);

export default router;
```

### Test file

```typescript
// src/api/routes/users.test.ts

import request from 'supertest';
import { app } from '../../app';
import { seedUser, clearUsers } from '../../../tests/fixtures/users';

describe('POST /users', () => {
  beforeEach(() => clearUsers());

  // ── Constraint tests (invariants) ──

  test('never creates two users with the same email', async () => {
    const email = 'dup@example.com';
    await request(app).post('/users')
      .set('Authorization', 'Bearer test-token')
      .send({ email, displayName: 'First' })
      .expect(201);

    await request(app).post('/users')
      .set('Authorization', 'Bearer test-token')
      .send({ email, displayName: 'Second' })
      .expect(409);
  });

  test('always rejects requests without authentication', async () => {
    await request(app).post('/users')
      .send({ email: 'a@b.com', displayName: 'X' })
      .expect(401);
  });

  test('always validates email format', async () => {
    const invalids = ['notanemail', '@nodomain', 'no@', ''];
    for (const email of invalids) {
      await request(app).post('/users')
        .set('Authorization', 'Bearer test-token')
        .send({ email, displayName: 'X' })
        .expect(400);
    }
  });

  test('rate limiter blocks after 10 requests in 15 minutes', async () => {
    // Send 11 requests — the 11th should be rate-limited
    const requests = Array.from({ length: 11 }, (_, i) =>
      request(app).post('/users')
        .set('Authorization', 'Bearer test-token')
        .send({ email: `user${i}@example.com`, displayName: 'X' })
    );
    const responses = await Promise.all(requests);
    expect(responses.some((r) => r.status === 429)).toBe(true);
  });
});
```

---

## FastAPI/Python Template

```python
# backend/api/routes/users.py

from fastapi import APIRouter, Depends, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, EmailStr, Field

from ..dependencies.auth import get_current_user
from ..services.user_service import UserService, DuplicateEmailError
from ..utils.audit import audit_log

router = APIRouter(prefix="/users", tags=["users"])
limiter = Limiter(key_func=get_remote_address)


class CreateUserRequest(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=100)
    role: str = Field(default="user", pattern="^(user|admin)$")


@router.post("/", status_code=status.HTTP_201_CREATED)
@limiter.limit("10/15minutes")
async def create_user(
    request: CreateUserRequest,
    current_user=Depends(get_current_user),
    user_service: UserService = Depends(),
):
    """Create a new user account."""
    try:
        user = await user_service.create(request)
    except DuplicateEmailError:
        raise HTTPException(status_code=409, detail="Email already registered")

    audit_log("user.created", actor_id=current_user.id, target_id=user.id)
    return user
```

---

## Checklist Before Completing

After generating an endpoint, verify:
- [ ] Rate limiting middleware is applied
- [ ] Authentication is required (or explicitly exempted with a reason)
- [ ] All inputs are validated with a schema
- [ ] Errors use the standard `ApiError` / HTTPException format
- [ ] OpenAPI documentation is complete
- [ ] At least one constraint test (invariant) is written
- [ ] `verify-ci.sh` passes (architecture test: all endpoints have rate limiting)
