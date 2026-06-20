# /project:verify — Run CLARE verification

Runs `./clare/verify-ci.sh` and reports a structured summary of results.

## Instructions

Execute the following and report results clearly:

```bash
./clare/verify-ci.sh 2>&1
# Or to continue past failures and collect all results before reviewing:
./clare/verify-ci.sh --fail-slow 2>&1
```

After running:
1. State clearly: **PASSED** or **FAILED**
2. If failed, list each failing check by name
3. For each failure, show the first few lines of the captured error output (the script prints a concise snippet). Include a suggested "quick command" the developer/agent can run to iterate on that specific failure (e.g., `npx eslint --fix .`, `npx tsc --noEmit`, `pytest -q`).
4. Propose specific fixes for each failure
5. Offer to fix and re-run: "Shall I fix these issues and run verify again?"

## Format

```
## Verify CI Results

Status: ✅ PASSED / ❌ FAILED

### Failing Checks
- **[Check name]**: [brief error description]
  Fix: [specific action needed]

### Next Step
[Offer to fix and rerun, or confirm completion if passed]
```

If all checks pass, confirm: "All CI checks pass. Work is complete. Please review the diff and commit."
