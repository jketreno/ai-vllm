# CLARE Skill: Protobuf → UI Component Generation
# =================================================
# CLARE Principles: [L] Limited — [R] Reality-Aligned — [E] Ephemeral
#
# USAGE: Copy this file into your project's AI instructions directory.
# This skill tells your AI how to automatically update React components
# when protobuf definitions change.
#
# The .proto files are the SOURCE OF TRUTH.
# All UI components are DERIVED (full-autonomy, regeneratable at any time).

---

## When to Apply This Skill

Apply when a user says:
- "update the protos"
- "the protos changed — update the UI"
- "implement [service name] proto"
- `verify-ci.sh` fails with "Component missing for message X"

---

## The Rule

**Protobuf definitions are the source of truth.**
The UI must match them exactly. No manual interpretation. No guessing at types.

Direct mapping: `.proto field` → `TypeScript type` → `React prop` → `Zod validation`

---

## Directory Layout

```
proto/                      ← Protobuf definitions (SOURCE OF TRUTH)
  services/
    user.proto
    order.proto

src/
  components/
    proto/                  ← Generated React components (DERIVED — full-autonomy)
      user/
        UserMessage.tsx     ← Visualizes the User proto message
        UserMessage.test.tsx
      order/
        OrderMessage.tsx
        OrderMessage.test.tsx
  types/
    proto/                  ← Generated TypeScript types (DERIVED)
      user.ts
      order.ts
```

UPDATE: adjust paths to match your project.

---

## Proto → TypeScript Type Mapping

| Proto type | TypeScript | Notes |
|-----------|------------|-------|
| `string` | `string` | |
| `int32`, `int64` | `number` | Note: int64 may lose precision with JSON |
| `uint32`, `uint64` | `number` | |
| `float`, `double` | `number` | |
| `bool` | `boolean` | |
| `bytes` | `Uint8Array` | or `string` if base64-encoded |
| `repeated X` | `X[]` | |
| `map<K, V>` | `Record<K, V>` | |
| `enum X` | `enum X` or `"A" \| "B"` | |
| `message X` | `X` (nested interface) | |
| `google.protobuf.Timestamp` | `string` (ISO 8601) | |
| `google.protobuf.Any` | `unknown` | |
| `oneof` | discriminated union | |

---

## Step-by-Step Update Process

When proto files change:

### 1. Read the diff

```bash
git diff HEAD -- proto/**/*.proto
# or: compare the new .proto files to what's currently in src/types/proto/
```

### 2. Identify changes

For each `.proto` file:
- **New message** → create new component + type + test
- **Added field** → add to existing component and interface
- **Removed field** → remove from component, interface, and tests
- **Renamed field** → update throughout the component tree
- **Type change** → update TypeScript type + Zod schema + prop validation
- **Deleted message** → remove component, type, and all references

### 3. Generate TypeScript types

In `src/types/proto/[service].ts`:

```typescript
// @generated — DO NOT EDIT DIRECTLY
// Source: proto/services/user.proto
// Regenerate by asking AI to "update the protos"

export interface User {
  id: string;
  email: string;
  displayName: string;
  role: UserRole;
  createdAt: string; // ISO 8601
}

export enum UserRole {
  USER = 'USER',
  ADMIN = 'ADMIN',
  MODERATOR = 'MODERATOR',
}

export const UserSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  displayName: z.string(),
  role: z.nativeEnum(UserRole),
  createdAt: z.string().datetime(),
});
```

### 4. Generate React component

In `src/components/proto/user/UserMessage.tsx`:

```tsx
// @generated — DO NOT EDIT DIRECTLY
// Source: proto/services/user.proto

import React from 'react';
import { User } from '../../../types/proto/user';

interface UserMessageProps {
  message: User;
  className?: string;
}

/**
 * Renders a User proto message for inspection and testing.
 * Generated from: proto/services/user.proto
 */
export function UserMessage({ message, className }: UserMessageProps) {
  return (
    <div className={`proto-message ${className ?? ''}`} data-testid="user-message">
      <h3 className="proto-message__title">User</h3>
      <table className="proto-message__fields">
        <tbody>
          <tr><td className="field-name">id</td><td className="field-value">{message.id}</td></tr>
          <tr><td className="field-name">email</td><td className="field-value">{message.email}</td></tr>
          <tr><td className="field-name">displayName</td><td className="field-value">{message.displayName}</td></tr>
          <tr><td className="field-name">role</td><td className="field-value">{message.role}</td></tr>
          <tr><td className="field-name">createdAt</td><td className="field-value">{message.createdAt}</td></tr>
        </tbody>
      </table>
    </div>
  );
}
```

### 5. Generate tests

In `src/components/proto/user/UserMessage.test.tsx`:

```tsx
// @generated — DO NOT EDIT DIRECTLY

import { render, screen } from '@testing-library/react';
import { UserMessage } from './UserMessage';
import { UserRole } from '../../../types/proto/user';

const testUser = {
  id: 'usr_123',
  email: 'test@example.com',
  displayName: 'Test User',
  role: UserRole.USER,
  createdAt: '2024-01-01T00:00:00Z',
};

describe('UserMessage', () => {
  test('renders all proto fields', () => {
    render(<UserMessage message={testUser} />);
    expect(screen.getByText('usr_123')).toBeInTheDocument();
    expect(screen.getByText('test@example.com')).toBeInTheDocument();
    expect(screen.getByText('Test User')).toBeInTheDocument();
  });

  test('renders with testid for integration testing', () => {
    render(<UserMessage message={testUser} />);
    expect(screen.getByTestId('user-message')).toBeInTheDocument();
  });
});
```

### 6. Register the component

Update `src/components/proto/index.ts` to export the new component.

### 7. Run verification

```bash
./clare/verify-ci.sh
# Architecture test checks: all proto messages have components
# All fields are rendered
# All tests pass
```

---

## Architecture Test

`clare/examples/architecture-tests/api-rules.test.js` (adapted for proto):

```js
test('all proto messages have corresponding React components', () => {
  const protoMessages = loadProtoMessages(); // from proto/ directory
  const components = loadComponentRegistry(); // from src/components/proto/
  
  protoMessages.forEach(message => {
    expect(components).toContain(message.name);
  });
});
```

---

## What NOT to Do

- ❌ Do not hand-edit generated components — re-run this skill
- ❌ Do not interpret proto field semantics; render them as-is
- ❌ Do not add display logic or formatting beyond field-name → value rendering in the base component
- ❌ Do not skip test generation for new components
