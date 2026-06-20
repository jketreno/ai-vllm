---
applyTo: "**/*.test.ts,**/*.test.js,**/*.spec.ts,**/*.spec.js,tests/**,test/**"
---
# Testing Standards (CLARE — Assertive)

CLARE requires **constraint tests**, not confirmation tests. Every test file must enforce invariants — things that must always be true — not just describe what the current implementation does.

## The Core Distinction

```ts
// ❌ Confirmation test — describes implementation, not invariant
it('returns the user', () => {
  const user = getUser(1);
  expect(user.id).toBe(1);
});

// ✅ Constraint test — enforces an invariant
it('always returns a user matching the requested ID', () => {
  fc.assert(
    fc.property(fc.integer({ min: 1, max: 10000 }), (id) => {
      seedUser(id);
      const user = getUser(id);
      expect(user.id).toBe(id);
    })
  );
});
```

## Rules for Every Test File

1. **Name tests as invariants**, not as implementations:
   - ❌ `'creates a user'`
   - ✅ `'never allows duplicate emails'`

2. **Add at least one property-based test** per critical module using `fast-check` (JS) or `hypothesis` (Python)

3. **Architecture tests** go in `tests/architecture/` and run in `verify-ci.sh`; they must check structural constraints, not behavior:
   ```ts
   // Architecture test example
   test('all API endpoints have rate limiting', () => {
     const endpoints = loadEndpoints();
     endpoints.forEach(ep => expect(ep).toHaveRateLimiting());
   });
   ```

4. **Schema-lock tests**: For any external contract (API, event schema, proto), add a test that would fail if the schema changes unexpectedly:
   ```ts
   test('User API response matches expected schema', () => {
     const response = getUser(seed.userId);
     expect(response).toMatchSnapshot(); // or use Zod schema assertion
   });
   ```

5. **Test the test** — After writing tests, delete the implementation and regenerate it with AI. If tests fail, they are working. If they pass, verify the constraints are strong enough.

## Property-Based Testing Quick Reference

```ts
// fast-check (TypeScript/JavaScript)
import fc from 'fast-check';

test('never creates negative balances', () => {
  fc.assert(
    fc.property(
      fc.integer({ min: 1, max: 10000 }),
      fc.integer({ min: 1, max: 10000 }),
      (initialBalance, deductAmount) => {
        fc.pre(deductAmount <= initialBalance); // precondition
        const result = deduct(initialBalance, deductAmount);
        expect(result).toBeGreaterThanOrEqual(0);
      }
    )
  );
});
```

```python
# hypothesis (Python)
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=10000))
def test_never_creates_duplicate_ids(user_count):
    ids = [create_user(f"user{i}@example.com").id for i in range(user_count)]
    assert len(ids) == len(set(ids)), "IDs must be unique"
```

## What Makes a Test Worth Keeping

Extract invariants from PR review comments and team discussions. Look for:
- "this should never happen"
- "we always guarantee..."
- "this assumes..."

Turn each one into a named test. These are the tests that protect you.
