# Phase 01 — Versioned Capture and Verified-Correction Contract

Status: proposed  
Depends on: Phase 00  
Produces: schema-v2 capture records and explicit correction/verification
lineage

## Goal

Capture the high-value learning signal described by the CLARE₂ concept:
failed assistant work linked to a developer-approved correction and objective
verification evidence.

## Record Types

Define and schema-validate at least:

- `session`;
- `interaction`;
- `tool_outcome`;
- `assistant_attempt`;
- `developer_correction`;
- `code_change`;
- `verification_run`;
- `turn_outcome`; and
- `redaction_event`.

Every record receives a stable ID. Relationships use IDs, not timestamps or
LLM-inferred association.

## Correction Episode Contract

A correction-backed episode should link:

1. task/request ID;
2. assistant attempt ID;
3. optional failed patch hash and bounded diff hunks;
4. developer correction ID;
5. accepted patch hash and bounded diff hunks;
6. verification run IDs;
7. repository revision before and after;
8. affected paths;
9. explicit acceptance source; and
10. redaction/sensitivity metadata.

The system must distinguish:

- explicit developer rejection and replacement;
- an assistant self-revision without developer confirmation;
- CI failure followed by a verified fix;
- environmental/flaky failure;
- abandoned work; and
- successful first attempt.

Only the first and third are correction-backed training candidates by default.

## Privacy and Retention

- Capture minimal diff hunks rather than complete files.
- Reject binary data, secrets, credentials, authorization headers, environment
  dumps, and configured sensitive paths.
- Hash full artifacts when content cannot be retained.
- Bound text and diff sizes.
- Make retention class explicit.
- Support tombstones that prevent deleted sensitive material from being
  reintroduced during sync.

## Primary File Targets

- new versioned schemas under `clare2/pipeline/app`
- `clare2/pipeline/app/structured_output.py`
- `clare2/pipeline/app/corpus_sync.py`
- `clare2/scripts/clare2-capture-hook.sh`
- capture documentation and fixtures
- provider hook specifications in `clare2/templates/hooks` — **human-only**
- `clare/scripts/clare2-capture-event.sh` — **human-only**

AI implementation must define and test the contract, then stop for human input
before changing either humans-only area.

## Compatibility

- Continue accepting v1 records during migration.
- Tag inferred v1 relationships as unverified.
- Never silently upgrade a v1 narrative into a verified correction.
- Sync preserves record IDs and detects conflicting content for the same ID.

## Tests

- Schema round trips for every record type.
- Duplicate delivery is idempotent.
- Conflicting duplicate IDs are quarantined.
- Correction lineage cannot reference a missing attempt.
- Passing verification can be linked to the accepted patch.
- Flaky/environmental outcomes are distinguishable from semantic failures.
- Redaction removes seeded secrets and retains a redaction audit record.
- Oversized and malformed diffs are rejected or safely truncated.
- v1 input remains readable but is marked unverified.

## Production Validation

Capture controlled sessions containing:

1. a successful first attempt;
2. an explicit developer correction;
3. a CI failure fixed by a developer patch;
4. an environmental failure; and
5. an abandoned task.

Verify that only cases 2 and 3 produce verified correction lineages.

## Rollback

Disable v2 emission while retaining v2 records. Continue v1 capture. Do not
rewrite or delete synchronized session data.

## Exit Criteria

- A correction can be traced from task through accepted patch and verification.
- No LLM is required to determine whether a correction record exists.
- Privacy tests pass.
- Provider hook humans-only work is completed or explicitly recorded as a
  blocking human action.
- `verify-ci.sh` passes.

