# CLARE Skill: Reality Tests Against External Services
# ======================================================
# CLARE Principle: [R] Reality-Aligned
#
# USAGE: Copy this file into your project's AI instructions directory.
# This skill generates reality tests — tests that verify your local
# implementation matches the actual state of an external system.
#
# Reality tests run in staging or nightly CI, not on every PR.
# Their purpose: catch drift between your code and reality before
# that drift becomes institutionalized.

---

## When to Apply This Skill

Apply when a user says:
- "add a reality test for [service]"
- "verify [OAuth/IAM / database / external API] alignment"
- "we think our [concept] is drifting from [source]"
- Adding a new `sources_of_truth` entry to `clare/autonomy.yml`

---

## The Three-Question Drift Diagnostic

Before writing a reality test, answer:
1. **Where is [concept] defined?**
2. **Where is [concept] validated?**
3. **Where is [concept] enforced?**

If those answers point to different files/systems → you have drift. The reality test closes the loop.

---

## Reality Test Directory

Reality tests live in `tests/reality/` and run in staging CI only:

```yaml
# .github/workflows/nightly.yml (or your CI equivalent)
- name: Reality tests (staging)
  run: npm run test:reality
  env:
    NODE_ENV: staging
    # Secrets injected from staging environment
```

---

## IAM Permission Reality Test

```typescript
// tests/reality/iam-permissions.reality.test.ts

/**
 * Reality Test: Local user roles match IAM provider state
 * ========================================================
 * Source of truth: OAuth/IAM provider (declared in clare/autonomy.yml)
 * Runs: staging CI, nightly
 *
 * This test verifies that our local permission records accurately
 * reflect the state in the IAM provider. If they diverge, the IdP is correct.
 */

import { IAMClient } from '../../src/infrastructure/iam-client';
import { getUserRoles } from '../../src/services/permission.service';
import { testUserIds } from '../fixtures/reality-seeds';

// Only run in staging — never against production data
if (process.env.NODE_ENV !== 'staging') {
  throw new Error('Reality tests must only run in staging environment');
}

const iam = new IAMClient({
  endpoint: process.env.IAM_ENDPOINT!,
  clientSecret: process.env.IAM_CLIENT_SECRET!,
});

/** Normalize both representations to a comparable shape */
function normalizePermissions(perms: {
  roles: string[];
  isActive: boolean;
  groups: string[];
}) {
  return {
    // Sort arrays for stable comparison
    roles: [...perms.roles].sort(),
    isActive: perms.isActive,
    groups: [...perms.groups].sort(),
  };
}

describe('IAM Permission Reality Alignment', () => {
  let idpUsers: Map<string, { roles: string[]; enabled: boolean; groups: string[] }>;

  beforeAll(async () => {
    idpUsers = new Map();
    // Fetch all test users from IAM provider
    for (const id of testUserIds) {
      const user = await iam.users.get(id);
      idpUsers.set(id, user);
    }
  });

  test('local user roles match IAM provider for all test IDs', async () => {
    const drifts: string[] = [];

    for (const [id, idpUser] of idpUsers) {
      const local = await getUserRoles(id);
      if (!local) {
        drifts.push(`  ${id}: exists in IAM provider but NOT in local DB`);
        continue;
      }

      const idpNorm = normalizePermissions({
        roles: idpUser.roles,
        isActive: idpUser.enabled,
        groups: idpUser.groups,
      });

      const localNorm = normalizePermissions({
        roles: local.roles,
        isActive: local.isActive,
        groups: local.groups,
      });

      if (JSON.stringify(idpNorm) !== JSON.stringify(localNorm)) {
        drifts.push(
          `  ${id}: DRIFT DETECTED\n` +
          `    IAM:   ${JSON.stringify(idpNorm)}\n` +
          `    Local: ${JSON.stringify(localNorm)}`
        );
      }
    }

    if (drifts.length > 0) {
      throw new Error(
        `Permission drift detected (IAM provider is the source of truth):\n\n` +
        drifts.join('\n\n') +
        `\n\nRun the sync job or investigate the webhook handler.`
      );
    }
  });

  test('local DB has no role grants that do not exist in IAM provider', async () => {
    // Check for "ghost" permissions — local records with no IAM counterpart
    for (const id of testUserIds) {
      const local = await getUserRoles(id);
      if (local && !idpUsers.has(id)) {
        throw new Error(`Ghost permission found: ${id} has roles locally but not in IAM provider`);
      }
    }
  });
});
```

---

## Database Schema Reality Test

```typescript
// tests/reality/db-schema.reality.test.ts

/**
 * Reality Test: Application code matches live database schema
 * ============================================================
 * Source of truth: database schema
 * Runs: staging CI, nightly
 */

import { pool } from '../../src/infrastructure/db';
import * as models from '../../src/models'; // Your ORM models / type definitions

describe('Database Schema Alignment', () => {
  afterAll(() => pool.end());

  test('all ORM models have corresponding database tables', async () => {
    const result = await pool.query(`
      SELECT tablename FROM pg_tables 
      WHERE schemaname = 'public'
    `);
    const dbTables = new Set(result.rows.map((r: { tablename: string }) => r.tablename));
    const modelTables = Object.keys(models); // UPDATE: your model-to-table mapping

    const missing = modelTables.filter((t) => !dbTables.has(t));
    if (missing.length > 0) {
      throw new Error(
        `ORM models reference tables that don't exist in the database:\n` +
        missing.map((t) => `  • ${t}`).join('\n')
      );
    }
  });

  test('User table has all expected columns', async () => {
    const result = await pool.query(`
      SELECT column_name, data_type, is_nullable
      FROM information_schema.columns
      WHERE table_name = 'users'
      ORDER BY ordinal_position
    `);

    const columns = new Map(result.rows.map((r: { column_name: string; data_type: string }) =>
      [r.column_name, r.data_type]
    ));

    // These MUST always exist — add columns your code depends on
    const required = ['id', 'email', 'display_name', 'created_at'];
    const missing = required.filter((col) => !columns.has(col));

    if (missing.length > 0) {
      throw new Error(
        `User table is missing expected columns: ${missing.join(', ')}\n` +
        `Check if a migration was missed.`
      );
    }
  });
});
```

---

## Generating a Reality Test for Your Service

When asked to create a reality test, follow this process:

1. **Identify the source of truth** from `clare/autonomy.yml` → `sources_of_truth`
2. **Fetch the canonical state** from the external system (IAM provider, database, external API)
3. **Fetch the local state** from your service/database
4. **Normalize both** to a comparable structure (strip timestamps, sort arrays, etc.)
5. **Compare** — flag any differences as drift
6. **Guard the environment** — add `if NODE_ENV !== 'staging' throw` at the top
7. **Add to nightly CI** — not the PR check workflow

---

## Nightly CI Configuration (GitHub Actions)

```yaml
# .github/workflows/nightly.yml — COPY from clare/templates/github-actions/
name: Nightly Reality Checks
on:
  schedule:
    - cron: '0 2 * * *' # 2am UTC daily
  workflow_dispatch:     # Allow manual trigger

jobs:
  reality-tests:
    name: Reality Alignment Tests
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci
      - name: Run reality tests
        run: npm run test:reality
        env:
          NODE_ENV: staging
          IAM_ENDPOINT: ${{ secrets.STAGING_IAM_ENDPOINT }}
          IAM_CLIENT_SECRET: ${{ secrets.STAGING_IAM_CLIENT_SECRET }}
          DATABASE_URL: ${{ secrets.STAGING_DATABASE_URL }}
```
