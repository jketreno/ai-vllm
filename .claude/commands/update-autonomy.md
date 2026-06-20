# /project:update-autonomy — Guide autonomy boundary update

Guides the user through updating `clare/autonomy.yml` to add, modify, or remove an autonomy boundary.

## Instructions

1. First, read and display the current contents of `clare/autonomy.yml`
2. Ask the user:
   - **What path** are they adding/modifying? (e.g., `src/billing`)
   - **What level**? (`full-autonomy` / `supervised` / `humans-only`)
   - **Why**? (reason for this boundary)
   - **Add, modify, or remove** the entry?

3. Show the proposed YAML change before applying it:

```yaml
# Proposed addition to clare/autonomy.yml modules section:
  - path: "[path]"
    level: [level]
    reason: "[reason]"
```

4. Ask for confirmation: "Apply this change to `clare/autonomy.yml`?"
5. On confirmation, edit `clare/autonomy.yml` to add/modify/remove the entry
6. Confirm: "Updated `clare/autonomy.yml`. The `[path]` path is now `[level]`."

## Guidance for Choosing Levels

Suggest based on path characteristics:
- **full-autonomy**: Generated code, pure utilities, test fixtures, anything fully derived from a source of truth
- **supervised**: Business logic, API handlers, database queries, UI components with business rules
- **humans-only**: Money movement, authentication, cryptography, core domain models, regulatory compliance code

## After Updating

Remind the user: "Consider adding an architecture test that validates this boundary is respected. See `clare/templates/architecture-tests/autonomy-guard.test.js` for an example."
