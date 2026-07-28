# Phase 02 — Deterministic Evidence and Distillation Quality

Status: proposed  
Depends on: Phase 01  
Produces: provenance-backed patterns with code-derived evidence

## Goal

Keep semantic interpretation in the LLM while moving recurrence, provenance,
eligibility, and safety decisions into deterministic code.

## Design

The distillation model may propose:

- category;
- precise behavioral rule;
- canonical example selection;
- stability candidate;
- source-of-truth candidate; and
- semantic grouping key.

It may not author authoritative values for:

- evidence count;
- distinct-session count;
- correction status;
- verification status;
- sensitivity;
- eligibility; or
- source event IDs.

These fields are calculated from validated source records.

## Evidence Policy

Maintain separate values:

- `observation_count`;
- `distinct_session_count`;
- `explicit_correction_count`;
- `verified_fix_count`;
- `first_observed_at`;
- `last_observed_at`; and
- `supporting_event_ids`.

Initial eligibility policy:

- verified developer correction: eligible as high-confidence single evidence;
- automatically inferred pattern: requires recurrence across at least two
  distinct sessions;
- repeated wording inside one assistant response: one observation;
- tool outcome without relevant content: not semantic evidence;
- environmental failure: excluded from behavioral antipatterns unless the
  learned behavior concerns diagnosing environmental failures.

Thresholds must be configuration with strict validation, not prompt text.

## Quality Gates

- Require evidence excerpts to resolve to retained source event IDs.
- Reject a proposed pattern that contradicts its source excerpt.
- Quarantine unknown categories and malformed timestamps.
- Detect likely incident-specific or obsolete wording.
- Record distillation model ID, revision, prompt hash, and parse outcome.
- Never mark a session processed after an LLM transport or validation failure.
- Make reprocessing idempotent by session content hash and distiller version.

## Primary File Targets

- `clare2/pipeline/app/distiller.py`
- `clare2/pipeline/app/structured_output.py`
- `clare2/pipeline/prompts/distill.txt`
- `clare2/pipeline/app/metrics.py`
- `clare2/pipeline/tests/test_distiller.py`
- new provenance/evidence modules and fixtures

## Tests

- An LLM-supplied inflated evidence count has no effect.
- Two excerpts in one event count once.
- Two distinct sessions count twice.
- A verified correction is eligible without falsifying recurrence.
- Missing source event IDs cause quarantine.
- Failed distillation remains pending.
- Reprocessing identical input emits no duplicate episode.
- Changing distiller version produces a new lineage without overwriting v1.
- Environmental and semantic failures follow different paths.

## Production Validation

Run v2 distillation in shadow mode over the controlled Phase 01 sessions and a
sample of existing sessions. Produce a comparison report showing:

- v1 accepted patterns;
- v2 accepted patterns;
- gated patterns and exact reasons;
- distinct-session evidence;
- correction-backed evidence; and
- unmatched or unverifiable canonical examples.

## Rollback

Keep v1 distillation active and retain shadow v2 output under a separate
versioned root. Never replace v1 episode files during this phase.

## Exit Criteria

- No accepted pattern depends on an LLM-authored count.
- Every accepted pattern resolves to source event IDs.
- Failed sessions remain retryable.
- Shadow comparison is reviewed.
- `verify-ci.sh` passes.

